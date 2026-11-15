// Gumbel distribution implementation.

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
// Direct log-space formula - avoids statrs computing pdf then taking ln()

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gumbel {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
    /// Curvature handed to the booster. `Damped` by default since 2026-11-14; a model
    /// saved before the field existed loads as `Observed`, what it was trained with.
    #[serde(default = "crate::distributions::location_scale::legacy_hessian_mode")]
    hessian_mode: HessianMode,
    /// Start values used without `initialize`. `LocationScale` by default since
    /// 2026-11-14; a model saved before the field existed loads as `Raw` (0.5, 0.5).
    #[serde(default)]
    start: LocationScaleStart,
    /// Multiplier on the moment-matched start scale of `LocationScaleStart::LocationScale`.
    /// [`GUMBEL_START_SCALE_FACTOR`] by default; 1.0 for a model saved before the field existed.
    #[serde(default = "one")]
    start_scale_factor: f64,
}

/// Default start-scale factor of [`Gumbel`]: the start scale is this many times the
/// moment-matched one (research proposal 028: chosen on validation NLL over 1–100;
/// naval −5.13 at 1 against −5.47 at 100, flat where the start does not matter).
pub const GUMBEL_START_SCALE_FACTOR: f64 = 100.0;

fn one() -> f64 {
    1.0
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
            // Defaults since 2026-11-14 (research proposals 028, 031, Felix's decision): the
            // observed curvature and the raw 0.5 start do not fit a target away from zero.
            hessian_mode: HessianMode::Damped,
            start: LocationScaleStart::LocationScale,
            start_scale_factor: GUMBEL_START_SCALE_FACTOR,
        }
    }

    /// With `LocationScaleStart::LocationScale`, start the scale at `factor` times the
    /// moment-matched value (default [`GUMBEL_START_SCALE_FACTOR`]). The lower tail of the
    /// Gumbel density costs e^{-z}, so a start that is too narrow for some rows begins with
    /// an NLL in the thousands; a start that is too wide costs only ln(factor) and makes
    /// the first rounds a least-squares fit of the location.
    pub fn with_start_scale_factor(mut self, factor: f64) -> Self {
        assert!(factor.is_finite() && factor > 0.0, "start scale factor must be positive");
        self.start_scale_factor = factor;
        self
    }

    pub fn start_scale_factor(&self) -> f64 {
        self.start_scale_factor
    }

    /// Curvature: `Damped` (the default) takes the row-wise maximum of the observed second
    /// derivative and the expected information, `Fisher` the expected information alone,
    /// `Observed` the behaviour before 2026-11-14.
    pub fn with_hessian_mode(mut self, mode: HessianMode) -> Self {
        self.hessian_mode = mode;
        self
    }

    pub fn hessian_mode(&self) -> HessianMode {
        self.hessian_mode
    }

    /// Closed-form start values used without `initialize` (see [`LocationScaleStart`]);
    /// `LocationScale` by default, `Raw` restores the behaviour before 2026-11-14.
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

    fn default_start_values(&self, target: &ResponseData) -> Option<ndarray::Array1<f64>> {
        let scale_fn = &self.params[1].response_fn;
        let mut v = location_scale::start_values(self.start, LsFamily::Gumbel, target, scale_fn)?;
        if self.start == LocationScaleStart::LocationScale && self.start_scale_factor != 1.0 {
            let wide = scale_fn.apply_scalar(v[1]) * self.start_scale_factor;
            v[1] = super::student_t::invert_response(scale_fn, wide);
        }
        Some(v)
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
            let (resp_deriv, resp_second) =
                scale_response_fn.derivative_batches_from_transformed(&p_scale, &t_scale);

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

        location_scale::apply_hessian_mode(
            self.hessian_mode,
            LsFamily::Gumbel,
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

    /// Proposal 028: the start-scale factor widens the moment-matched scale only, follows
    /// the label units, is inert for the other start rules, and old files load with 1.
    #[test]
    fn start_scale_factor_widens_only_the_matched_scale() {
        let y: ndarray::Array1<f64> = (0..500).map(|i| 400.0 + 17.0 * ((i * 37 % 101) as f64 / 101.0 - 0.5)).collect();
        let start = |d: &Gumbel, y: &ndarray::Array1<f64>| {
            let v = y.view();
            d.default_start_values(&ResponseData::Univariate(&v))
        };
        assert_eq!(Gumbel::default().start_scale_factor(), GUMBEL_START_SCALE_FACTOR);
        assert_eq!(Gumbel::default().start(), LocationScaleStart::LocationScale);
        let ls = Gumbel::default().with_start_scale_factor(1.0);
        let (a, b) = (start(&ls, &y).unwrap(), start(&ls.clone().with_start_scale_factor(100.0), &y).unwrap());
        assert_relative_eq!(b[0], a[0], epsilon = 1e-12);
        assert_relative_eq!(b[1], a[1] + 100f64.ln(), epsilon = 1e-6); // Exp link (inverted numerically)
        // labels in other units: location and scale follow
        let y2 = y.mapv(|v| 1000.0 * v);
        let c = start(&ls.clone().with_start_scale_factor(100.0), &y2).unwrap();
        assert_relative_eq!(c[0], 1000.0 * b[0], max_relative = 1e-9);
        assert_relative_eq!(c[1], b[1] + 1000f64.ln(), epsilon = 1e-6);
        // inert without the LocationScale rule
        assert!(start(&Gumbel::default().with_start(LocationScaleStart::Raw), &y).is_none());
        let loc = Gumbel::default().with_start(LocationScaleStart::Location);
        assert_eq!(start(&loc, &y), start(&loc.clone().with_start_scale_factor(100.0), &y));
        // a model saved before the field existed
        let mut v = serde_json::to_value(Gumbel::default().with_start_scale_factor(7.0)).unwrap();
        v.as_object_mut().unwrap().remove("start_scale_factor");
        let back: Gumbel = serde_json::from_value(v).unwrap();
        assert_eq!(back.start_scale_factor(), 1.0);
    }
}
