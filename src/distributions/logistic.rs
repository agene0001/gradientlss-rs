//! Logistic distribution implementation.

use super::base::{Distribution, DistributionParam, HessianMode, LossFn, Stabilization};
use super::location_scale::{self, LocationScaleStart, LsFamily};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logistic {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
    /// Curvature handed to the booster. `Damped` by default; a model saved before the
    /// field existed loads as `Observed`, what it was trained with.
    #[serde(default = "crate::distributions::location_scale::legacy_hessian_mode")]
    hessian_mode: HessianMode,
    /// Start values used without `initialize`. `LocationScale` by default; a model saved
    /// before the field existed loads as `Raw`.
    #[serde(default)]
    start: LocationScaleStart,
}

impl Logistic {
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
            // Defaults since 2026-11-13 (research proposal 021, Felix's decision): the
            // observed curvature and the raw 0.5 start stall on targets far from zero.
            hessian_mode: HessianMode::Damped,
            start: LocationScaleStart::LocationScale,
        }
    }

    /// Curvature (default `Damped`; `Observed` restores the behaviour before 2026-11-13):
    /// `Fisher` replaces the observed
    /// second derivative with the expected information, `Damped` takes the row-wise maximum.
    pub fn with_hessian_mode(mut self, mode: HessianMode) -> Self {
        self.hessian_mode = mode;
        self
    }

    pub fn hessian_mode(&self) -> HessianMode {
        self.hessian_mode
    }

    /// Closed-form start values used without `initialize` (default `LocationScale`; `Raw`
    /// restores the behaviour before 2026-11-13; see [`LocationScaleStart`]).
    pub fn with_start(mut self, start: LocationScaleStart) -> Self {
        self.start = start;
        self
    }

    pub fn start(&self) -> LocationScaleStart {
        self.start
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

        let z = (target - loc) / scale;
        -z - scale.ln() - 2.0 * (1.0 + (-z).exp()).ln()
    }
}

#[typetag::serde]
impl Distribution for Logistic {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Logistic"
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

    fn default_start_values(&self, target: &ResponseData) -> Option<ndarray::Array1<f64>> {
        location_scale::start_values(self.start, LsFamily::Logistic, target, &self.params[1].response_fn)
    }

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_scalar(params, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                let col0 = params.column(0);
                let col1 = params.column(1);
                crate::distributions::util::par_nansum(y.len(), |i| {
                    -self.log_prob_scalar(&[col0[i], col1[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("Logistic is a univariate distribution."),
        }
    }

    /// Analytical gradients for Logistic distribution.
    ///
    /// For NLL = z + ln(s) + 2*ln(1 + exp(-z)), where z = (y - μ) / s
    ///
    /// Gradients:
    /// - ∂NLL/∂μ = (1 - 2σ(z)) / s, where σ(z) = 1/(1+exp(-z))
    /// - ∂NLL/∂s = (z*(1-2σ(z)) + 1) / s
    ///
    /// Hessians:
    /// - ∂²NLL/∂μ² = 2σ(z)(1-σ(z)) / s²
    /// - ∂²NLL/∂s² computed with full chain rule
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
            let (resp_deriv, resp_second) =
                scale_response_fn.derivative_batches_from_transformed(&p_scale, &t_scale);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let loc = t_loc[i];
                let scale = t_scale[i].max(1e-6);
                let yi = y[i];

                let z = (yi - loc) / scale;
                // Numerically stable sigmoid
                let sig_z = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                };
                let sig_1m = 1.0 - sig_z;
                let one_minus_2sig = 1.0 - 2.0 * sig_z;
                let scale_sq = scale * scale;

                // Gradient/Hessian w.r.t. loc (identity response)
                let g0 = one_minus_2sig / scale;
                let h0 = 2.0 * sig_z * sig_1m / scale_sq;

                // Gradient/Hessian w.r.t. scale with chain rule
                let grad_scale_param = (z * one_minus_2sig + 1.0) / scale;
                let hess_scale_param =
                    (-2.0 * z * one_minus_2sig + 2.0 * z * z * sig_z * sig_1m - 1.0) / scale_sq;
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
                // Numerically stable sigmoid
                let sig_z = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                };
                let sig_1m = 1.0 - sig_z;
                let one_minus_2sig = 1.0 - 2.0 * sig_z;
                let scale_sq = scale * scale;

                // Gradient/Hessian w.r.t. loc (identity response)
                let g0 = one_minus_2sig / scale;
                let h0 = 2.0 * sig_z * sig_1m / scale_sq;

                // Gradient/Hessian w.r.t. scale with chain rule
                let pred_scale = p_scale[i];
                let rd = scale_response_fn.derivative(pred_scale);
                let rsd = scale_response_fn.second_derivative(pred_scale);
                let grad_scale_param = (z * one_minus_2sig + 1.0) / scale;
                let hess_scale_param =
                    (-2.0 * z * one_minus_2sig + 2.0 * z * z * sig_z * sig_1m - 1.0) / scale_sq;
                let g1 = grad_scale_param * rd;
                let h1 = hess_scale_param * rd * rd + grad_scale_param * rsd;

                gradients[[i, 0]] = g0;
                hessians[[i, 0]] = h0;
                gradients[[i, 1]] = g1;
                hessians[[i, 1]] = h1;
            }
        }

        location_scale::apply_hessian_mode(
            self.hessian_mode,
            LsFamily::Logistic,
            &mut hessians,
            &predictions.column(1),
            &transformed.column(1),
            scale_response_fn,
        );
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
                    result[[i, j]] = loc + scale * (u / (1.0 - u)).ln();
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_logistic_creation() {
        let dist = Logistic::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
    }

    #[test]
    fn test_logistic_log_prob() {
        let dist = Logistic::default();
        let log_p = dist.log_prob_scalar(&[0.0, 1.0], 0.0);
        let expected = -2.0 * (2.0f64).ln(); // -ln(4)
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_logistic_sample() {
        let dist = Logistic::default();
        let params = array![[0.0, 1.0], [2.0, 3.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        assert_eq!(samples.dim(), (1000, 2));

        let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_0, 0.0, epsilon = 0.2);

        let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_1, 2.0, epsilon = 0.2);
    }
}
