//! Student-T distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, StudentT as RandStudentT};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::constants::{LOG_PI, trigamma};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentT {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl StudentT {
    pub fn new(
        stabilization: Stabilization,
        df_response_fn: ResponseFn,
        scale_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("df", df_response_fn),
            DistributionParam::new("loc", ResponseFn::Identity),
            DistributionParam::new("scale", scale_response_fn),
        ];

        Self {
            params,
            stabilization,
            loss_fn,
            initialize,
        }
    }

    pub fn default() -> Self {
        Self::new(
            Stabilization::None,
            ResponseFn::ExpDf,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        )
    }

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let df = params[0];
        let loc = params[1];
        let scale = params[2];

        if df <= 0.0 || scale <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if target.is_infinite() {
            return f64::NEG_INFINITY;
        }

        let d = (target - loc) / scale;
        ln_gamma((df + 1.0) / 2.0)
            - 0.5 * ((df + 1.0) * (1.0 + d * d / df).ln())
            - ln_gamma(df / 2.0)
            - 0.5 * (df.ln() + LOG_PI)
            - scale.ln()
    }
}

#[typetag::serde]
impl Distribution for StudentT {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "StudentT"
    }

    fn n_params(&self) -> usize {
        3
    }

    fn params(&self) -> &[DistributionParam] {
        &self.params
    }

    fn loss_fn(&self) -> LossFn {
        self.loss_fn
    }

    fn stabilization(&self) -> Stabilization {
        self.stabilization
    }

    fn should_initialize(&self) -> bool {
        self.initialize
    }

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_scalar(params, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                let df_col = params.column(0);
                let loc_col = params.column(1);
                let scale_col = params.column(2);
                crate::distributions::util::par_sum(y.len(), |i| {
                    -self.log_prob_scalar(&[df_col[i], loc_col[i], scale_col[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("StudentT is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let df = params[[j, 0]];
            let loc = params[[j, 1]];
            let scale = params[[j, 2]];

            if df > 0.0 && scale > 0.0 {
                // rand_distr::StudentT only takes df, so we sample and transform
                if let Ok(dist) = RandStudentT::new(df) {
                    for i in 0..n_samples {
                        let sample: f64 = dist.sample(&mut rng);
                        result[[i, j]] = loc + scale * sample;
                    }
                }
            }
        }
        result
    }

    /// Analytical gradients for Student-T distribution.
    ///
    /// NLL = -ln_gamma((ν+1)/2) + 0.5*(ν+1)*ln(1+d²/ν) + ln_gamma(ν/2) + 0.5*ln(νπ) + ln(σ)
    /// where d = (y-μ)/σ, ν = df.
    ///
    /// Gradients w.r.t. distribution parameters (let z = 1 + d²/ν):
    /// - ∂NLL/∂μ = -(ν+1)*d / (σ*(ν+d²))
    /// - ∂NLL/∂σ = 1/σ - (ν+1)*d² / (σ*(ν+d²))
    /// - ∂NLL/∂ν = -0.5*digamma((ν+1)/2) + 0.5*digamma(ν/2) + 0.5*ln(z) - 0.5*(ν+1)*d²/(ν²*z) + 0.5/ν
    fn analytical_gradients(
        &self,
        predictions: &ArrayView2<f64>,
        transformed: &ArrayView2<f64>,
        target: &ResponseData,
    ) -> Option<(Array2<f64>, Array2<f64>)> {
        if self.loss_fn != LossFn::Nll {
            return None;
        }

        let y = match target {
            ResponseData::Univariate(arr) => arr,
            ResponseData::Multivariate(_) => return None,
        };

        let n_samples = predictions.nrows();

        let df_response_fn = &self.params[0].response_fn;
        let scale_response_fn = &self.params[2].response_fn;

        let t_df = transformed.column(0);
        let t_loc = transformed.column(1);
        let t_scale = transformed.column(2);
        let p_df = predictions.column(0);
        let p_scale = predictions.column(2);

        let mut gradients = Array2::zeros((n_samples, 3));
        let mut hessians = Array2::zeros((n_samples, 3));

        if n_samples >= 4096 {
            let (rd_df, rsd_df) = df_response_fn.derivative_batches(&p_df);
            let (rd_scale, rsd_scale) = scale_response_fn.derivative_batches(&p_scale);

            let compute_sample = |i: usize| -> ([f64; 3], [f64; 3]) {
                let nu = t_df[i].max(1e-6);
                let mu = t_loc[i];
                let sigma = t_scale[i].max(1e-6);
                let yi = y[i];

                if !yi.is_finite() {
                    return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
                }

                let d = (yi - mu) / sigma;
                let d2 = d * d;
                let nu_plus_d2 = nu + d2;
                let z = 1.0 + d2 / nu;
                let nu_p1 = nu + 1.0;
                let sigma_sq = sigma * sigma;

                let g1 = -(nu_p1) * d / (sigma * nu_plus_d2);
                let h1 = nu_p1 / sigma_sq * (nu - d2) / (nu_plus_d2 * nu_plus_d2);

                let grad_sigma = 1.0 / sigma - nu_p1 * d2 / (sigma * nu_plus_d2);
                let hess_sigma_param = -1.0 / sigma_sq
                    + nu_p1 * d2 * (3.0 * nu + d2) / (sigma_sq * nu_plus_d2 * nu_plus_d2);
                let rs = rd_scale[i];
                let rss = rsd_scale[i];
                let g2 = grad_sigma * rs;
                let h2 = hess_sigma_param * rs * rs + grad_sigma * rss;

                let half_nu_p1 = nu_p1 / 2.0;
                let half_nu = nu / 2.0;
                let ln_z = z.ln();
                let grad_nu = -0.5 * digamma(half_nu_p1) + 0.5 * digamma(half_nu) + 0.5 * ln_z
                    - 0.5 * nu_p1 * d2 / (nu * nu * z)
                    + 0.5 / nu;
                let hess_nu_param =
                    -0.25 * trigamma(half_nu_p1) + 0.25 * trigamma(half_nu) - 0.5 / (nu * nu);
                let rd = rd_df[i];
                let rsd = rsd_df[i];
                let g0 = grad_nu * rd;
                let h0 = hess_nu_param * rd * rd + grad_nu * rsd;

                ([g0, g1, g2], [h0, h1, h2])
            };

            let results: Vec<_> = (0..n_samples).into_par_iter().map(compute_sample).collect();
            for (i, (g, h)) in results.into_iter().enumerate() {
                gradients[[i, 0]] = g[0];
                gradients[[i, 1]] = g[1];
                gradients[[i, 2]] = g[2];
                hessians[[i, 0]] = h[0];
                hessians[[i, 1]] = h[1];
                hessians[[i, 2]] = h[2];
            }
        } else {
            for i in 0..n_samples {
                let nu = t_df[i].max(1e-6);
                let mu = t_loc[i];
                let sigma = t_scale[i].max(1e-6);
                let yi = y[i];

                if !yi.is_finite() {
                    hessians[[i, 0]] = 1e-6;
                    hessians[[i, 1]] = 1e-6;
                    hessians[[i, 2]] = 1e-6;
                    continue;
                }

                let d = (yi - mu) / sigma;
                let d2 = d * d;
                let nu_plus_d2 = nu + d2;
                let z = 1.0 + d2 / nu;
                let nu_p1 = nu + 1.0;
                let sigma_sq = sigma * sigma;

                gradients[[i, 1]] = -(nu_p1) * d / (sigma * nu_plus_d2);
                hessians[[i, 1]] =
                    nu_p1 / sigma_sq * (nu - d2) / (nu_plus_d2 * nu_plus_d2);

                let grad_sigma = 1.0 / sigma - nu_p1 * d2 / (sigma * nu_plus_d2);
                let hess_sigma_param = -1.0 / sigma_sq
                    + nu_p1 * d2 * (3.0 * nu + d2) / (sigma_sq * nu_plus_d2 * nu_plus_d2);
                let pred_scale = p_scale[i];
                let rs = scale_response_fn.derivative(pred_scale);
                let rss = scale_response_fn.second_derivative(pred_scale);
                gradients[[i, 2]] = grad_sigma * rs;
                hessians[[i, 2]] = hess_sigma_param * rs * rs + grad_sigma * rss;

                let half_nu_p1 = nu_p1 / 2.0;
                let half_nu = nu / 2.0;
                let ln_z = z.ln();
                let grad_nu = -0.5 * digamma(half_nu_p1) + 0.5 * digamma(half_nu) + 0.5 * ln_z
                    - 0.5 * nu_p1 * d2 / (nu * nu * z)
                    + 0.5 / nu;
                let hess_nu_param =
                    -0.25 * trigamma(half_nu_p1) + 0.25 * trigamma(half_nu) - 0.5 / (nu * nu);
                let pred_df = p_df[i];
                let rd = df_response_fn.derivative(pred_df);
                let rsd = df_response_fn.second_derivative(pred_df);
                gradients[[i, 0]] = grad_nu * rd;
                hessians[[i, 0]] = hess_nu_param * rd * rd + grad_nu * rsd;
            }
        }

        Some((gradients, hessians))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ResponseData;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_student_t_creation() {
        let dist = StudentT::default();
        assert_eq!(dist.n_params(), 3);
        assert_eq!(dist.param_names(), vec!["df", "loc", "scale"]);
        assert!(!dist.should_initialize());
    }

    #[test]
    fn test_student_t_log_prob() {
        let dist = StudentT::default();
        let log_p = dist.log_prob_scalar(&[5.0, 0.0, 1.0], 0.0);
        // StudentT(df=5, loc=0, scale=1) at x=0: d=0
        // ln_gamma(3) - 0.5*(6*ln(1)) - ln_gamma(2.5) - 0.5*ln(5π) - ln(1)
        let expected = ln_gamma(3.0) - 0.0 - ln_gamma(2.5) - 0.5 * (5.0_f64.ln() + LOG_PI);
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_student_t_nll() {
        let dist = StudentT::default();
        let params = array![[5.0, 0.0, 1.0], [5.0, 0.0, 1.0]];
        let target = array![0.0, 0.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(ln_gamma(3.0) - ln_gamma(2.5) - 0.5 * (5.0_f64.ln() + LOG_PI));
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }

    #[test]
    fn test_student_t_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        let dist = StudentT::default();
        // Use predictions that produce reasonable transformed params
        let predictions = array![[0.5, 1.0, 0.3], [-0.2, 0.5, 0.8], [1.0, -0.5, 0.1]];
        let targets = array![1.5, 0.8, -1.0];
        let target = ResponseData::Univariate(&targets.view());

        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");

        let numerical = dist
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("Should return numerical gradients");

        // Compare gradients (allow looser tolerance for df parameter which involves digamma)
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-2);
            }
        }
    }

    #[test]
    fn test_student_t_sample() {
        let dist = StudentT::default();
        let params = array![[5.0, 0.0, 1.0], [10.0, 5.0, 0.5]];
        let samples = dist.sample(&params.view(), 10000, 123);

        assert_eq!(samples.dim(), (10000, 2));

        let mean_0: f64 = samples.column(0).mean().unwrap();
        let mean_1: f64 = samples.column(1).mean().unwrap();

        assert_relative_eq!(mean_0, 0.0, epsilon = 0.1);
        assert_relative_eq!(mean_1, 5.0, epsilon = 0.1);
    }
}
