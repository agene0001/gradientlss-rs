// Gumbel distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
// Direct log-space formula - avoids statrs computing pdf then taking ln()

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gumbel {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Gumbel {
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

    /// Helper method for scalar log probability (direct log-space formula)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let loc = params[0];
        let scale = params[1];

        if scale <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // ln_pdf = -z - exp(-z) - ln(scale), where z = (x - loc) / scale
        let z = (target - loc) / scale;
        -z - (-z).exp() - scale.ln()
    }
}

#[typetag::serde]
impl Distribution for Gumbel {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Gumbel"
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
                let col0 = params.column(0);
                let col1 = params.column(1);
                crate::distributions::util::par_sum(y.len(), |i| {
                    -self.log_prob_scalar(&[col0[i], col1[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("Gumbel is a univariate distribution."),
        }
    }

    /// Analytical gradients for Gumbel distribution.
    ///
    /// For NLL = z + exp(-z) + ln(s), where z = (x - μ) / s
    ///
    /// Gradients:
    /// - ∂NLL/∂μ = (exp(-z) - 1) / s
    /// - ∂NLL/∂s = (z*exp(-z) - z + 1) / s
    ///
    /// Hessians:
    /// - ∂²NLL/∂μ² = exp(-z) / s²
    /// - ∂²NLL/∂s² = (z²*exp(-z) - 2z*exp(-z) + 2z - 1) / s²
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

        let scale_response_fn = &self.params[1].response_fn;

        let t_loc = transformed.column(0);
        let t_scale = transformed.column(1);
        let p_scale = predictions.column(1);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        if n_samples >= 4096 {
            // Pre-compute batch derivatives for scale (loc uses Identity: deriv=1, second=0)
            let (resp_deriv, resp_second) = scale_response_fn.derivative_batches(&p_scale);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let loc = t_loc[i];
                let scale = t_scale[i].max(1e-6);
                let yi = y[i];

                let z = (yi - loc) / scale;
                let u = (-z).exp();
                let scale_sq = scale * scale;

                // Gradient/Hessian w.r.t. loc (identity response)
                let g0 = (u - 1.0) / scale;
                let h0 = u / scale_sq;

                // Gradient/Hessian w.r.t. scale with chain rule
                let grad_scale_param = (z * u - z + 1.0) / scale;
                let hess_scale_param = (z * z * u - 2.0 * z * u + 2.0 * z - 1.0) / scale_sq;
                let rd = resp_deriv[i];
                let rsd = resp_second[i];
                let g1 = grad_scale_param * rd;
                let h1 = hess_scale_param * rd * rd + grad_scale_param * rsd;

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

                let z = (yi - loc) / scale;
                let u = (-z).exp();
                let scale_sq = scale * scale;

                // Gradient/Hessian w.r.t. loc (identity response)
                let g0 = (u - 1.0) / scale;
                let h0 = u / scale_sq;

                // Gradient/Hessian w.r.t. scale with chain rule
                let pred_scale = p_scale[i];
                let rd = scale_response_fn.derivative(pred_scale);
                let rsd = scale_response_fn.second_derivative(pred_scale);
                let grad_scale_param = (z * u - z + 1.0) / scale;
                let hess_scale_param = (z * z * u - 2.0 * z * u + 2.0 * z - 1.0) / scale_sq;
                let g1 = grad_scale_param * rd;
                let h1 = hess_scale_param * rd * rd + grad_scale_param * rsd;

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
            let loc = params[[j, 0]];
            let scale = params[[j, 1]];

            if scale > 0.0 {
                for i in 0..n_samples {
                    let u: f64 = rng.random_range(0.0..1.0);
                    result[[i, j]] = loc - scale * (-(u.ln())).ln();
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
    const EULER_MASCHERONI: f64 = 0.5772156649015329;

    #[test]
    fn test_gumbel_creation() {
        let dist = Gumbel::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
    }

    #[test]
    fn test_gumbel_log_prob() {
        let dist = Gumbel::default();
        let log_p = dist.log_prob_scalar(&[0.0, 1.0], 1.0);
        // Gumbel(loc=0, scale=1) at x=1: z=1, -1 - exp(-1) - ln(1) = -1 - 1/e
        let expected = -1.0 - (-1.0_f64).exp();
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gumbel_nll() {
        let dist = Gumbel::default();
        let params = array![[0.0, 1.0], [0.0, 1.0]];
        let target = array![1.0, 1.0];
        let target_response = ResponseData::Univariate(&target.view());
        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(-1.0 - (-1.0_f64).exp());
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }

    #[test]
    fn test_gumbel_sample() {
        let dist = Gumbel::default();
        let params = array![[0.0, 1.0], [2.0, 3.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        assert_eq!(samples.dim(), (1000, 2));

        let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        let expected_mean_0 = 0.0 + 1.0 * EULER_MASCHERONI; // loc + scale * euler_mascheroni
        assert_relative_eq!(mean_0, expected_mean_0, epsilon = 0.2);

        let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;
        let expected_mean_1 = 2.0 + 3.0 * EULER_MASCHERONI; // loc + scale * euler_mascheroni
        assert_relative_eq!(mean_1, expected_mean_1, epsilon = 0.2);
    }
}
