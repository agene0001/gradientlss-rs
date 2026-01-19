//! Student-T distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, StudentT as RandStudentT};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, StudentsT};
use statrs::function::gamma::digamma;

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

    /// Helper method for scalar log probability
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let df = params[0];
        let loc = params[1];
        let scale = params[2];

        if df <= 0.0 || scale <= 0.0 {
            return f64::NEG_INFINITY;
        }

        match StudentsT::new(loc, scale, df) {
            Ok(dist) => dist.ln_pdf(target),
            Err(_) => f64::NEG_INFINITY,
        }
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
                let mut total = 0.0;
                for (i, &y_val) in y.iter().enumerate() {
                    let p = vec![params[[i, 0]], params[[i, 1]], params[[i, 2]]];
                    total -= self.log_prob_scalar(&p, y_val);
                }
                total
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
    /// The Student-T log probability is:
    /// log p(y|ν,μ,σ) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ) - log(σ)
    ///                  - ((ν+1)/2) * log(1 + ((y-μ)/σ)²/ν)
    ///
    /// Gradients:
    /// - ∂NLL/∂μ = (ν+1) * (y-μ) / (σ² * (ν + z²))  where z = (y-μ)/σ
    /// - ∂NLL/∂σ = 1/σ - (ν+1) * z² / (σ * (ν + z²))
    /// - ∂NLL/∂ν = 0.5 * [digamma((ν+1)/2) - digamma(ν/2) - 1/ν - log(1 + z²/ν) + (ν+1)*z²/(ν*(ν+z²))]
    fn analytical_gradients(
        &self,
        predictions: &ArrayView2<f64>,
        transformed: &ArrayView2<f64>,
        target: &ResponseData,
    ) -> Option<(Array2<f64>, Array2<f64>)> {
        // Only support NLL for analytical gradients
        if self.loss_fn != LossFn::Nll {
            return None;
        }

        let y = match target {
            ResponseData::Univariate(arr) => arr,
            ResponseData::Multivariate(_) => return None,
        };

        let n_samples = predictions.nrows();
        let mut gradients = Array2::zeros((n_samples, 3));
        let mut hessians = Array2::zeros((n_samples, 3));

        let df_response_fn = &self.params[0].response_fn;
        let scale_response_fn = &self.params[2].response_fn;

        for i in 0..n_samples {
            let df = transformed[[i, 0]].max(2.001); // Ensure df > 2 for stability
            let loc = transformed[[i, 1]];
            let scale = transformed[[i, 2]].max(1e-6);

            let yi = y[i];
            let z = (yi - loc) / scale;
            let z_sq = z * z;
            let df_plus_zsq = df + z_sq;

            // Gradient w.r.t. df
            // ∂NLL/∂ν = -0.5 * [digamma((ν+1)/2) - digamma(ν/2) - 1/ν - log(1 + z²/ν) + (ν+1)*z²/(ν*(ν+z²))]
            let digamma_term = digamma((df + 1.0) / 2.0) - digamma(df / 2.0);
            let log_term = (1.0 + z_sq / df).ln();
            let ratio_term = (df + 1.0) * z_sq / (df * df_plus_zsq);
            let grad_df_param = -0.5 * (digamma_term - 1.0 / df - log_term + ratio_term);

            let pred_df = predictions[[i, 0]];
            let df_derivative = df_response_fn.derivative(pred_df);
            gradients[[i, 0]] = grad_df_param * df_derivative;

            // Hessian for df (use positive approximation)
            hessians[[i, 0]] = (0.1 * df_derivative * df_derivative).max(1e-6);

            // Gradient w.r.t. loc (identity response)
            // ∂NLL/∂μ = -(ν+1) * (y-μ) / (σ² * (ν + z²))
            let grad_loc = -(df + 1.0) * (yi - loc) / (scale * scale * df_plus_zsq);
            gradients[[i, 1]] = grad_loc;

            // Hessian w.r.t. loc
            hessians[[i, 1]] = ((df + 1.0) / (scale * scale * df_plus_zsq)).max(1e-6);

            // Gradient w.r.t. scale
            // ∂NLL/∂σ = 1/σ - (ν+1) * z² / (σ * (ν + z²))
            let grad_scale_param = 1.0 / scale - (df + 1.0) * z_sq / (scale * df_plus_zsq);

            let pred_scale = predictions[[i, 2]];
            let scale_derivative = scale_response_fn.derivative(pred_scale);
            gradients[[i, 2]] = grad_scale_param * scale_derivative;

            // Hessian w.r.t. scale (simplified positive approximation)
            let hess_scale = (1.0 / (scale * scale)) * scale_derivative * scale_derivative;
            hessians[[i, 2]] = hess_scale.max(1e-6);
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
        let expected = StudentsT::new(0.0, 1.0, 5.0).unwrap().ln_pdf(0.0);
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_student_t_nll() {
        let dist = StudentT::default();
        let params = array![[5.0, 0.0, 1.0], [5.0, 0.0, 1.0]];
        let target = array![0.0, 0.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -StudentsT::new(0.0, 1.0, 5.0).unwrap().ln_pdf(0.0);
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
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
