//! Weibull distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Weibull as RandWeibull};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
// Direct formula - no statrs constructor needed

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weibull {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Weibull {
    pub fn new(
        stabilization: Stabilization,
        scale_response_fn: ResponseFn,
        concentration_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("scale", scale_response_fn),
            DistributionParam::new("concentration", concentration_response_fn),
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
            ResponseFn::Exp,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        )
    }

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let scale = params[0]; // lambda
        let k = params[1]; // concentration/shape

        if scale <= 0.0 || k <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if target < 0.0 || target.is_infinite() {
            return f64::NEG_INFINITY;
        }
        if target == 0.0 {
            if (k - 1.0).abs() < f64::EPSILON {
                return -scale.ln(); // exponential special case
            }
            return f64::NEG_INFINITY;
        }

        // ln_pdf = ln(k) + (k-1)*ln(x/λ) - (x/λ)^k - ln(λ)
        k.ln() + (k - 1.0) * (target / scale).ln() - (target / scale).powf(k) - scale.ln()
    }
}

#[typetag::serde]
impl Distribution for Weibull {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Weibull"
    }

    fn n_params(&self) -> usize {
        2
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
                let scale_col = params.column(0);
                let conc_col = params.column(1);
                let mut total = 0.0;
                for (i, &y_val) in y.iter().enumerate() {
                    if y_val < 0.0 {
                        return f64::INFINITY;
                    }
                    total -= self.log_prob_scalar(&[scale_col[i], conc_col[i]], y_val);
                }
                total
            }
            ResponseData::Multivariate(_) => panic!("Weibull is a univariate distribution."),
        }
    }

    /// Analytical gradients for Weibull distribution.
    ///
    /// NLL = -ln(k) - (k-1)*ln(x/λ) + (x/λ)^k + ln(λ)
    ///
    /// Let t = x/λ. Gradients w.r.t. distribution parameters:
    /// - dNLL/dλ = k*(1 - t^k)/λ
    /// - dNLL/dk = -1/k + ln(t)*(t^k - 1)
    ///
    /// Hessians:
    /// - d²NLL/dλ² = k*((k+1)*t^k - k) / λ²
    /// - d²NLL/dk² = 1/k² + t^k * ln(t)²
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

        let scale_response_fn = &self.params[0].response_fn;
        let conc_response_fn = &self.params[1].response_fn;

        let t_scale = transformed.column(0);
        let t_conc = transformed.column(1);
        let p_scale = predictions.column(0);
        let p_conc = predictions.column(1);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        if n_samples >= 4096 {
            // Pre-compute batch derivatives for both parameters
            let rd_s = scale_response_fn.derivative_batch(&p_scale);
            let rsd_s = scale_response_fn.second_derivative_batch(&p_scale);
            let rd_k = conc_response_fn.derivative_batch(&p_conc);
            let rsd_k = conc_response_fn.second_derivative_batch(&p_conc);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let lambda = t_scale[i].max(1e-6);
                let k = t_conc[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || !yi.is_finite() {
                    return (0.0, 1e-6, 0.0, 1e-6);
                }

                let t = yi / lambda;
                let ln_t = t.ln();
                let t_k = t.powf(k);

                let grad_lambda = k * (1.0 - t_k) / lambda;
                let hess_lambda = k * ((k + 1.0) * t_k - k) / (lambda * lambda);
                let grad_k = -1.0 / k + ln_t * (t_k - 1.0);
                let hess_k = 1.0 / (k * k) + t_k * ln_t * ln_t;

                let r0 = rd_s[i];
                let rs0 = rsd_s[i];
                let g0 = grad_lambda * r0;
                let h0 = (hess_lambda * r0 * r0 + grad_lambda * rs0).max(1e-6);

                let r1 = rd_k[i];
                let rs1 = rsd_k[i];
                let g1 = grad_k * r1;
                let h1 = (hess_k * r1 * r1 + grad_k * rs1).max(1e-6);

                (g0, h0, g1, h1)
            };

            let results: Vec<_> = (0..n_samples).into_par_iter().map(compute_sample).collect();
            for (i, (g0, h0, g1, h1)) in results.into_iter().enumerate() {
                gradients[[i, 0]] = g0;
                hessians[[i, 0]] = h0;
                gradients[[i, 1]] = g1;
                hessians[[i, 1]] = h1;
            }
        } else {
            for i in 0..n_samples {
                let lambda = t_scale[i].max(1e-6);
                let k = t_conc[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || !yi.is_finite() {
                    gradients[[i, 0]] = 0.0;
                    hessians[[i, 0]] = 1e-6;
                    gradients[[i, 1]] = 0.0;
                    hessians[[i, 1]] = 1e-6;
                    continue;
                }

                let t = yi / lambda;
                let ln_t = t.ln();
                let t_k = t.powf(k);

                let grad_lambda = k * (1.0 - t_k) / lambda;
                let hess_lambda = k * ((k + 1.0) * t_k - k) / (lambda * lambda);
                let grad_k = -1.0 / k + ln_t * (t_k - 1.0);
                let hess_k = 1.0 / (k * k) + t_k * ln_t * ln_t;

                let r0 = scale_response_fn.derivative(p_scale[i]);
                let rs0 = scale_response_fn.second_derivative(p_scale[i]);
                let g0 = grad_lambda * r0;
                let h0 = (hess_lambda * r0 * r0 + grad_lambda * rs0).max(1e-6);

                let r1 = conc_response_fn.derivative(p_conc[i]);
                let rs1 = conc_response_fn.second_derivative(p_conc[i]);
                let g1 = grad_k * r1;
                let h1 = (hess_k * r1 * r1 + grad_k * rs1).max(1e-6);

                gradients[[i, 0]] = g0;
                hessians[[i, 0]] = h0;
                gradients[[i, 1]] = g1;
                hessians[[i, 1]] = h1;
            }
        }

        Some((gradients, hessians))
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let scale = params[[j, 0]];
            let concentration = params[[j, 1]];

            if scale > 0.0 && concentration > 0.0 {
                // rand_distr::Weibull uses (scale, shape) parameterization
                if let Ok(dist) = RandWeibull::new(scale, concentration) {
                    for i in 0..n_samples {
                        result[[i, j]] = dist.sample(&mut rng);
                    }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ResponseData;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_weibull_creation() {
        let dist = Weibull::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["scale", "concentration"]);
    }

    #[test]
    fn test_weibull_log_prob() {
        let dist = Weibull::default();
        let log_p = dist.log_prob_scalar(&[1.0, 2.0], 1.0);
        // Weibull(shape=2, scale=1) at x=1: ln(2) + 1*ln(1) - 1^2 - ln(1) = ln(2) - 1
        let expected = 2.0_f64.ln() - 1.0;
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_weibull_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        let dist = Weibull::default();
        let predictions = array![[0.5, 0.3], [0.8, 0.5], [0.3, 0.8]];
        let targets = array![1.5, 0.8, 2.0];
        let target = ResponseData::Univariate(&targets.view());

        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");

        let numerical = dist
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("Should return numerical gradients");

        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-2);
            }
        }
    }

    #[test]
    fn test_weibull_nll() {
        let dist = Weibull::default();
        let params = array![[1.0, 2.0], [1.0, 2.0]];
        let target = array![1.0, 1.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(2.0_f64.ln() - 1.0);
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}
