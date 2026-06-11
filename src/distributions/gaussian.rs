//! Gaussian (Normal) distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::constants::LOG_2PI;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
        -0.5 * LOG_2PI - scale.ln() - 0.5 * z * z
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
                let loc_col = params.column(0);
                let scale_col = params.column(1);
                if params.nrows() == 1 {
                    let p = [loc_col[0], scale_col[0]];
                    crate::distributions::util::par_sum(y.len(), |i| -self.log_prob_scalar(&p, y[i]))
                } else {
                    crate::distributions::util::par_sum(y.len(), |i| {
                        -self.log_prob_scalar(&[loc_col[i], scale_col[i]], y[i])
                    })
                }
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

        let scale_response_fn = &self.params[1].response_fn;

        let t_loc = transformed.column(0);
        let t_scale = transformed.column(1);
        let p_scale = predictions.column(1);

        // Batch response-fn derivatives (auto-vectorized, one shared `exp`),
        // hoisted above the branch so the sequential path no longer pays two
        // scalar `exp` calls per sample via `derivative`/`second_derivative`.
        let (resp_deriv, resp_second) = scale_response_fn.derivative_batches(&p_scale);

        let compute = |i: usize| -> (f64, f64, f64, f64) {
            let loc = t_loc[i];
            let scale = t_scale[i].max(1e-6);
            let scale_sq = scale * scale;
            let yi = y[i];
            let diff = yi - loc;
            let diff_sq = diff * diff;
            let g0 = -diff / scale_sq;
            let h0 = 1.0 / scale_sq;
            let grad_scale_param = 1.0 / scale - diff_sq / (scale_sq * scale);
            let hess_scale_param = -1.0 / scale_sq + 3.0 * diff_sq / (scale_sq * scale_sq);
            let rd = resp_deriv[i];
            let rsd = resp_second[i];
            let g1 = grad_scale_param * rd;
            let h1 = hess_scale_param * rd * rd + grad_scale_param * rsd;
            (g0, h0, g1, h1)
        };

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));
        // C-order rows are [param0, param1] per sample; write straight into the
        // contiguous backing slices (same pattern as NegativeBinomial).
        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");

        // Threshold matches the other analytical paths (256).
        const PAR_MIN: usize = 256;
        if n_samples >= PAR_MIN {
            g_slice
                .par_chunks_mut(2)
                .zip(h_slice.par_chunks_mut(2))
                .enumerate()
                .for_each(|(i, (gc, hc))| {
                    let (g0, h0, g1, h1) = compute(i);
                    gc[0] = g0;
                    gc[1] = g1;
                    hc[0] = h0;
                    hc[1] = h1;
                });
        } else {
            g_slice
                .chunks_mut(2)
                .zip(h_slice.chunks_mut(2))
                .enumerate()
                .for_each(|(i, (gc, hc))| {
                    let (g0, h0, g1, h1) = compute(i);
                    gc[0] = g0;
                    gc[1] = g1;
                    hc[0] = h0;
                    hc[1] = h1;
                });
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
        assert_relative_eq!(log_p, -0.5 * LOG_2PI, epsilon = 1e-10);

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
