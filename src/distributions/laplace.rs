//! Laplace distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Exp};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
// Direct formula - no statrs constructor needed

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

    /// Helper method for scalar log probability (direct formula)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let loc = params[0];
        let scale = params[1];

        if scale <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // ln_pdf = -|x - loc| / scale - ln(2 * scale)
        -((target - loc).abs() / scale) - (2.0 * scale).ln()
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
                let loc_col = params.column(0);
                let scale_col = params.column(1);
                crate::distributions::util::par_sum(y.len(), |i| {
                    -self.log_prob_scalar(&[loc_col[i], scale_col[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("Laplace is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
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
                        result[[i, j]] = loc + e1 - e2;
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

        let scale_response_fn = &self.params[1].response_fn;

        let t_loc = transformed.column(0);
        let t_scale = transformed.column(1);
        let p_scale = predictions.column(1);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        if n_samples >= 4096 {
            // Pre-compute batch derivatives for scale (loc uses Identity: deriv=1, second=0)
            let resp_deriv = scale_response_fn.derivative_batch(&p_scale);
            let resp_second = scale_response_fn.second_derivative_batch(&p_scale);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let loc = t_loc[i];
                let scale = t_scale[i].max(1e-6);
                let yi = y[i];
                let diff = yi - loc;
                let abs_diff = diff.abs();
                let scale_sq = scale * scale;

                // Gradient/Hessian w.r.t. loc (identity response)
                let g0 = -diff.signum() / scale;
                let h0 = (1.0 / scale_sq).max(1e-6);

                // Gradient/Hessian w.r.t. scale with chain rule
                let grad_scale_param = 1.0 / scale - abs_diff / scale_sq;
                let hess_scale_param = -1.0 / scale_sq + 2.0 * abs_diff / (scale_sq * scale);
                let rd = resp_deriv[i];
                let rsd = resp_second[i];
                let g1 = grad_scale_param * rd;
                let h1 = (hess_scale_param * rd * rd + grad_scale_param * rsd).max(1e-6);

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
                let loc = t_loc[i];
                let scale = t_scale[i].max(1e-6);
                let yi = y[i];
                let diff = yi - loc;
                let abs_diff = diff.abs();
                let scale_sq = scale * scale;

                // Gradient/Hessian w.r.t. loc (identity response)
                let g0 = -diff.signum() / scale;
                let h0 = (1.0 / scale_sq).max(1e-6);

                // Gradient/Hessian w.r.t. scale with chain rule
                let pred_scale = p_scale[i];
                let rd = scale_response_fn.derivative(pred_scale);
                let rsd = scale_response_fn.second_derivative(pred_scale);
                let grad_scale_param = 1.0 / scale - abs_diff / scale_sq;
                let hess_scale_param = -1.0 / scale_sq + 2.0 * abs_diff / (scale_sq * scale);
                let g1 = grad_scale_param * rd;
                let h1 = (hess_scale_param * rd * rd + grad_scale_param * rsd).max(1e-6);

                gradients[[i, 0]] = g0;
                hessians[[i, 0]] = h0;
                gradients[[i, 1]] = g1;
                hessians[[i, 1]] = h1;
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
        // Laplace(loc=0, scale=1) at x=0: -|0|/1 - ln(2) = -ln(2)
        let expected = -(2.0_f64).ln();
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_nll() {
        let dist = Laplace::default();
        let params = array![[0.0, 1.0], [0.0, 1.0]];
        let target = array![0.0, 0.0];
        let target_response = ResponseData::Univariate(&target.view());
        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = (2.0_f64).ln();
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_sample() {
        let dist = Laplace::default();
        let params = array![[0.0, 1.0], [2.0, 3.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        assert_eq!(samples.dim(), (1000, 2));

        let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_0, 0.0, epsilon = 0.2);

        let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_1, 2.0, epsilon = 0.2);
    }
}
