//! Laplace distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Exp};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, Laplace as StatrsLaplace};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Laplace {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Laplace {
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

    /// Helper method for scalar log probability
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let loc = params[0];
        let scale = params[1];

        if scale <= 0.0 {
            return f64::NEG_INFINITY;
        }

        match StatrsLaplace::new(loc, scale) {
            Ok(dist) => dist.ln_pdf(target),
            Err(_) => f64::NEG_INFINITY,
        }
    }
}

#[typetag::serde]
impl Distribution for Laplace {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Laplace"
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
                    let p = vec![params[[i, 0]], params[[i, 1]]];
                    total -= self.log_prob_scalar(&p, y_val);
                }
                total
            }
            ResponseData::Multivariate(_) => panic!("Laplace is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_obs, n_samples));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let loc = params[[j, 0]];
            let scale = params[[j, 1]];

            if scale > 0.0 {
                // Sample from Laplace using inverse transform: loc - scale * sign(u-0.5) * ln(1 - 2*|u-0.5|)
                if let Ok(exp_dist) = Exp::new(1.0 / scale) {
                    for i in 0..n_samples {
                        let e1: f64 = exp_dist.sample(&mut rng);
                        let e2: f64 = exp_dist.sample(&mut rng);
                        result[[j, i]] = loc + e1 - e2;
                    }
                }
            }
        }
        result
    }

    /// Analytical gradients for Laplace distribution.
    ///
    /// The Laplace log probability is:
    /// log p(y|μ,b) = -log(2b) - |y-μ|/b
    ///
    /// Gradients:
    /// - ∂NLL/∂μ = sign(y-μ) / b
    /// - ∂NLL/∂b = 1/b - |y-μ| / b²
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
            let yi = y[i];
            let diff = yi - loc;
            let abs_diff = diff.abs();

            // Gradient w.r.t. loc (identity response)
            // ∂NLL/∂μ = -sign(y-μ) / b
            let sign_diff = if diff > 0.0 {
                1.0
            } else if diff < 0.0 {
                -1.0
            } else {
                0.0
            };
            gradients[[i, 0]] = -sign_diff / scale;

            // Hessian w.r.t. loc (Laplace has 0 second derivative except at μ=y)
            // Use a small positive value for stability
            hessians[[i, 0]] = (1.0 / (scale * scale)).max(1e-6);

            // Gradient w.r.t. scale
            // ∂NLL/∂b = 1/b - |y-μ| / b²
            let grad_scale_param = 1.0 / scale - abs_diff / (scale * scale);

            let pred_scale = predictions[[i, 1]];
            let scale_derivative = scale_response_fn.derivative(pred_scale);
            gradients[[i, 1]] = grad_scale_param * scale_derivative;

            // Hessian w.r.t. scale
            let hess_scale = (1.0 / (scale * scale)) * scale_derivative * scale_derivative;
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
    fn test_laplace_creation() {
        let dist = Laplace::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
        assert!(!dist.should_initialize());
    }

    #[test]
    fn test_laplace_log_prob() {
        let dist = Laplace::default();
        let log_p = dist.log_prob_scalar(&[0.0, 1.0], 0.0);
        let expected = StatrsLaplace::new(0.0, 1.0).unwrap().ln_pdf(0.0);
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_nll() {
        let dist = Laplace::default();
        let params = array![[0.0, 1.0], [0.0, 1.0]];
        let target = array![0.0, 0.0];
        let target_response = ResponseData::Univariate(&target.view());
        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -StatrsLaplace::new(0.0, 1.0).unwrap().ln_pdf(0.0);
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_sample() {
        let dist = Laplace::default();
        let params = array![[0.0, 1.0], [2.0, 3.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        assert_eq!(samples.dim(), (2, 1000));

        let mean_0: f64 = samples.row(0).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_0, 0.0, epsilon = 0.2);

        let mean_1: f64 = samples.row(1).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_1, 2.0, epsilon = 0.2);
    }
}
