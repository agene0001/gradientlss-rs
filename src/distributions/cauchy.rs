//! Cauchy distribution implementation.

use super::base::{Distribution, DistributionParam, HessianMode, LossFn, Stabilization};
use super::location_scale::{self, LocationScaleStart, LsFamily};
use crate::constants::LOG_PI;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cauchy {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
    /// Curvature handed to the booster. `Damped` by default; a model saved before the
    /// field existed loads as `Observed`, what it was trained with.
    #[serde(default = "crate::distributions::location_scale::legacy_hessian_mode")]
    hessian_mode: HessianMode,
    /// Start values used without `initialize`. `Location` by default; a model saved
    /// before the field existed loads as `Raw`.
    #[serde(default)]
    start: LocationScaleStart,
}

impl Cauchy {
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
            start: LocationScaleStart::Location,
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

    /// Closed-form start values used without `initialize` (default `Location`; `Raw`
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

        -LOG_PI - scale.ln() - (1.0 + ((target - loc) / scale).powi(2)).ln()
    }
}

#[typetag::serde]
impl Distribution for Cauchy {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Cauchy"
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
        location_scale::start_values(self.start, LsFamily::Cauchy, target, &self.params[1].response_fn)
    }

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_scalar(params, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                let loc_col = params.column(0);
                let scale_col = params.column(1);
                crate::distributions::util::par_nansum(y.len(), |i| {
                    -self.log_prob_scalar(&[loc_col[i], scale_col[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("Cauchy is a univariate distribution."),
        }
    }

    /// Analytical gradients for Cauchy distribution.
    ///
    /// NLL = ln(π) + ln(s) + ln(1 + z²), where z = (x - μ) / s
    ///
    /// Gradients:
    /// - ∂NLL/∂μ = -2z / (s*(1+z²))
    /// - ∂NLL/∂s = (1 - z²) / (s*(1+z²))
    ///
    /// Hessians:
    /// - ∂²NLL/∂μ² = 2*(1-z²) / (s²*(1+z²)²)
    /// - ∂²NLL/∂s² = (z⁴ + 4z² - 1) / (s²*(1+z²)²)
    ///   (note z depends on s: with c = x-μ, NLL(s) = ln(s²+c²) - ln(s) up to consts)
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
                let z2 = z * z;
                let denom = 1.0 + z2;
                let scale_sq = scale * scale;
                let denom_sq = denom * denom;

                // Gradient/Hessian w.r.t. loc (identity response)
                let g0 = -2.0 * z / (scale * denom);
                let h0 = 2.0 * (1.0 - z2) / (scale_sq * denom_sq);

                // Gradient/Hessian w.r.t. scale with chain rule
                let grad_scale_param = (1.0 - z2) / (scale * denom);
                let z4 = z2 * z2;
                let hess_scale_param = (z4 + 4.0 * z2 - 1.0) / (scale_sq * denom_sq);
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
                let z2 = z * z;
                let denom = 1.0 + z2;
                let scale_sq = scale * scale;
                let denom_sq = denom * denom;

                // Gradient/Hessian w.r.t. loc (identity response)
                let g0 = -2.0 * z / (scale * denom);
                let h0 = 2.0 * (1.0 - z2) / (scale_sq * denom_sq);

                // Gradient/Hessian w.r.t. scale with chain rule
                let pred_scale = p_scale[i];
                let rd = scale_response_fn.derivative(pred_scale);
                let rsd = scale_response_fn.second_derivative(pred_scale);
                let grad_scale_param = (1.0 - z2) / (scale * denom);
                let z4 = z2 * z2;
                let hess_scale_param = (z4 + 4.0 * z2 - 1.0) / (scale_sq * denom_sq);
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
            LsFamily::Cauchy,
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
                    result[[i, j]] = loc + scale * (PI * (u - 0.5)).tan();
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

    #[test]
    fn test_cauchy_creation() {
        let dist = Cauchy::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
    }

    #[test]
    fn test_cauchy_log_prob() {
        let dist = Cauchy::default();
        let log_p = dist.log_prob_scalar(&[0.0, 1.0], 0.0);
        let expected = -LOG_PI;
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cauchy_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;
        use crate::types::ResponseData;
        use ndarray::array;

        // the observed second derivative is what the numerical path differentiates;
        // the default (Damped) raises it to the Fisher value where it is smaller
        let dist = Cauchy::default().with_hessian_mode(HessianMode::Observed);
        let predictions = array![[0.0, 0.0], [0.5, 0.3], [-0.2, 0.6]];
        let targets = array![1.0, 2.0, -1.5];
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
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-3);
                // The scale Hessian previously used (3z⁴-6z²-1) instead of
                // (z⁴+4z²-1) in the numerator — wrong sign at |z| ≈ 1.
                assert_relative_eq!(
                    analytical.1[[i, j]],
                    numerical.1[[i, j]],
                    epsilon = 1e-2,
                    max_relative = 1e-2
                );
            }
        }
    }
}
