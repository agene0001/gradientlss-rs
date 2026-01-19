//! Gaussian (Normal) distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Gaussian (Normal) distribution for distributional regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Gaussian {
    pub fn new(
        stabilization: Stabilization,
        scale_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
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
        Self::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false)
    }

    /// Compute log probability for a single scalar target.
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let loc = params[0];
        let scale = params[1];

        if scale <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let z = (target - loc) / scale;
        -0.5 * (2.0 * PI).ln() - scale.ln() - 0.5 * z * z
    }
}

#[typetag::serde]
impl Distribution for Gaussian {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Gaussian"
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
                let mut total = 0.0;
                for (i, &y_val) in y.iter().enumerate() {
                    let p = if params.nrows() == 1 {
                        vec![params[[0, 0]], params[[0, 1]]]
                    } else {
                        vec![params[[i, 0]], params[[i, 1]]]
                    };
                    total -= self.log_prob_scalar(&p, y_val);
                }
                total
            }
            ResponseData::Multivariate(_) => panic!("Gaussian is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let loc = params[[j, 0]];
            let scale = params[[j, 1]].max(1e-6);

            if let Ok(normal) = Normal::new(loc, scale) {
                for i in 0..n_samples {
                    result[[i, j]] = normal.sample(&mut rng);
                }
            }
        }
        result
    }

    /// Analytical gradients for Gaussian distribution.
    ///
    /// For NLL = -log p(y|μ,σ) = 0.5*log(2π) + log(σ) + 0.5*(y-μ)²/σ²
    ///
    /// Gradients w.r.t. transformed parameters:
    /// - ∂NLL/∂μ = (μ - y) / σ²
    /// - ∂NLL/∂σ = 1/σ - (y - μ)² / σ³
    ///
    /// We apply the chain rule: ∂NLL/∂pred = ∂NLL/∂param * ∂param/∂pred
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
        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        let scale_response_fn = &self.params[1].response_fn;

        for i in 0..n_samples {
            let loc = transformed[[i, 0]];
            let scale = transformed[[i, 1]].max(1e-6);
            let scale_sq = scale * scale;
            let scale_cu = scale_sq * scale;
            let yi = y[i];
            let diff = yi - loc;
            let diff_sq = diff * diff;

            // Gradient w.r.t. loc (identity response, so derivative is 1)
            // ∂NLL/∂loc = -diff / σ² = (loc - y) / σ²
            let grad_loc = -diff / scale_sq;
            gradients[[i, 0]] = grad_loc;

            // Hessian w.r.t. loc
            // ∂²NLL/∂loc² = 1 / σ²
            hessians[[i, 0]] = (1.0 / scale_sq).max(1e-6);

            // Gradient w.r.t. scale (need chain rule for response function)
            // ∂NLL/∂σ = 1/σ - (y-μ)²/σ³
            let grad_scale_param = 1.0 / scale - diff_sq / scale_cu;

            // Chain rule: ∂NLL/∂pred_scale = ∂NLL/∂σ * ∂σ/∂pred_scale
            let pred_scale = predictions[[i, 1]];
            let response_derivative = scale_response_fn.derivative(pred_scale);
            gradients[[i, 1]] = grad_scale_param * response_derivative;

            // Hessian w.r.t. scale (simplified - using diagonal approximation)
            // ∂²NLL/∂σ² = -1/σ² + 3*(y-μ)²/σ⁴
            let hess_scale_param = -1.0 / scale_sq + 3.0 * diff_sq / (scale_sq * scale_sq);
            // Apply chain rule squared for Hessian (Gauss-Newton approximation)
            let hess_scale = hess_scale_param * response_derivative * response_derivative;
            hessians[[i, 1]] = hess_scale.max(1e-6);
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
    fn test_gaussian_creation() {
        let dist = Gaussian::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
        assert!(!dist.should_initialize());
    }

    #[test]
    fn test_gaussian_log_prob() {
        let dist = Gaussian::default();

        let log_p = dist.log_prob_scalar(&[0.0, 1.0], 0.0);
        assert_relative_eq!(log_p, -0.5 * (2.0 * PI).ln(), epsilon = 1e-10);

        let log_p_wide = dist.log_prob_scalar(&[0.0, 2.0], 0.0);
        let log_p_narrow = dist.log_prob_scalar(&[0.0, 0.5], 0.0);
        assert!(log_p_narrow > log_p_wide);
    }

    #[test]
    fn test_gaussian_nll() {
        let dist = Gaussian::default();
        let params = array![[0.0, 1.0], [0.0, 1.0]];
        let target = array![0.0, 0.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected = 2.0 * -dist.log_prob_scalar(&[0.0, 1.0], 0.0);
        assert_relative_eq!(nll, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_sample() {
        let dist = Gaussian::default();
        let params = array![[0.0, 1.0], [5.0, 0.5]];
        let samples = dist.sample(&params.view(), 1000, 123);

        assert_eq!(samples.dim(), (1000, 2));

        let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;

        assert_relative_eq!(mean_0, 0.0, epsilon = 0.1);
        assert_relative_eq!(mean_1, 5.0, epsilon = 0.1);
    }

    #[test]
    fn test_gaussian_transform_params() {
        let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        let predictions = array![[0.0, 0.0], [1.0, 1.0]];
        let transformed = dist.transform_params(&predictions.view());

        assert_relative_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-6);
        assert_relative_eq!(transformed[[1, 0]], 1.0, epsilon = 1e-6);

        assert!(transformed[[0, 1]] > 0.0);
        assert!(transformed[[1, 1]] > transformed[[0, 1]]);
    }
}
