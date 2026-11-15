//! Gamma distribution implementation.

use super::base::{Distribution, DistributionParam, HessianMode, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Gamma as RandGamma};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::constants::trigamma;

/// Boosting coordinates of the Gamma head.
///
/// `ConcentrationRate` (the default; XGBoostLSS parity) boosts (log α, log β).
/// Those coordinates are nearly collinear: the Fisher information is
/// `[[α²ψ′(α), −α], [−α, α]]` (correlation → −1 as α grows; −0.9992 at α=330),
/// and α moves the MEAN α/β as well as the dispersion. A diagonal Newton
/// booster then sees almost only mean-information on the α coordinate and
/// learns the dispersion O(1/α) too slowly — measured: power_plant needs
/// α ≈ 12 000 from a start of 330, and the one-step move is 0.0015 of the
/// 3.6 required (90 % coverage stuck at 1.000).
///
/// `MeanShape` boosts (log µ, log α), β = α/µ — Cox & Reid (1987)
/// orthogonal coordinates with the Fisher information exactly
/// `diag(α, α²ψ′(α) − α)`, so tree 0 moves the mean at fixed shape and tree 1
/// the shape at fixed mean (same idea as `NbParameterization::MeanDispersion`).
/// Start values are the exact Gamma MLE. Transformed / predicted parameters
/// are STILL reported as `(concentration, rate)`. NLL only.
///
/// Default since 2026-11-10 (`Gamma::new` with `ResponseFn::Exp` and the NLL
/// loss, and `Gamma::default()`): MeanShape beat ConcentrationRate by 0.6–4.7
/// nats on power_plant / naval / synthetic Gamma and lost kin8nm by 0.05 (won
/// it with a validation-tuned shape freeze). Opt out with
/// `.with_parameterization(GammaParameterization::ConcentrationRate)`. Model
/// files saved before the field existed still load as `ConcentrationRate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GammaParameterization {
    ConcentrationRate,
    #[default]
    MeanShape,
}

impl GammaParameterization {
    /// The serde default for files written before the field existed.
    fn legacy() -> Self {
        GammaParameterization::ConcentrationRate
    }
}

/// Raw-coordinate clamps in `MeanShape` mode: log α ∈ [−15, 25] (α up to 7e10),
/// log µ ∈ [−300, 300]. Outside them the gradient of that coordinate is zeroed.
const GAMMA_LOG_SHAPE_CLAMP: (f64, f64) = (-15.0, 25.0);
const GAMMA_LOG_MEAN_CLAMP: (f64, f64) = (-300.0, 300.0);

/// ψ(α) − ln α, cancellation-safe for large α.
pub(crate) fn digamma_minus_ln(alpha: f64) -> f64 {
    if alpha > 50.0 {
        let a2 = alpha * alpha;
        -1.0 / (2.0 * alpha) - 1.0 / (12.0 * a2) + 1.0 / (120.0 * a2 * a2) - 1.0 / (252.0 * a2 * a2 * a2)
    } else {
        digamma(alpha) - alpha.ln()
    }
}

/// α²ψ′(α) − α (the Fisher information of log α at fixed mean), cancellation-safe.
/// Always positive; → 1/2 as α → ∞.
pub(crate) fn shape_fisher(alpha: f64) -> f64 {
    if alpha > 50.0 {
        let a2 = alpha * alpha;
        0.5 + 1.0 / (6.0 * alpha) - 1.0 / (30.0 * a2 * alpha) + 1.0 / (42.0 * a2 * a2 * alpha)
    } else {
        alpha * alpha * trigamma(alpha) - alpha
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gamma {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
    /// Curvature handed to the booster (see [`HessianMode`]); defaults to `Damped`.
    #[serde(default)]
    hessian_mode: HessianMode,
    /// Boosting coordinates (see [`GammaParameterization`]); old files load as
    /// `ConcentrationRate`.
    #[serde(default = "GammaParameterization::legacy")]
    parameterization: GammaParameterization,
    /// MeanShape without `initialize`: start log µ at ln ȳ (shape stays at
    /// the raw 0.5). Only affects training.
    #[serde(default)]
    location_start: bool,
    /// MeanShape without `initialize`: the unit-free start [ln ȳ + 0.5, 0.5]
    /// (µ₀ = 1.65·ȳ, α₀ = 1.65). Default on; models saved before the field
    /// existed load with the raw [0.5, 0.5] start. `location_start` takes
    /// precedence. Only affects training.
    #[serde(default)]
    wide_start: bool,
}

/// Offset of the unit-free start's log mean above ln ȳ (see `wide_start`).
pub const GAMMA_WIDE_START_LOG_MEAN_OFFSET: f64 = 0.5;

impl Gamma {
    /// `MeanShape` coordinates (the default, see [`GammaParameterization`]) when
    /// `response_fn` is `Exp` and the loss is NLL; any other response function
    /// or the CRPS loss keeps the per-parameter `ConcentrationRate` head.
    pub fn new(
        stabilization: Stabilization,
        response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("concentration", response_fn),
            DistributionParam::new("rate", response_fn),
        ];
        Self {
            params,
            stabilization,
            loss_fn,
            initialize,
            hessian_mode: HessianMode::default(),
            parameterization: if response_fn == ResponseFn::Exp && loss_fn == LossFn::Nll {
                GammaParameterization::default()
            } else {
                GammaParameterization::ConcentrationRate
            },
            location_start: false,
            wide_start: true,
        }
    }

    /// See the `location_start` field.
    pub fn with_location_start(mut self, on: bool) -> Self {
        self.location_start = on;
        self
    }

    pub fn location_start(&self) -> bool {
        self.location_start
    }

    /// See the `wide_start` field. `false` restores the raw [0.5, 0.5]
    /// start, whose position relative to the data depends on the target's units.
    pub fn with_wide_start(mut self, on: bool) -> Self {
        self.wide_start = on;
        self
    }

    pub fn wide_start(&self) -> bool {
        self.wide_start
    }

    /// Choose the boosting coordinates. `MeanShape` forces `Exp` responses
    /// (its map is (log µ, log α)) and is ignored, with a warning, for the
    /// CRPS loss.
    pub fn with_parameterization(mut self, parameterization: GammaParameterization) -> Self {
        if parameterization == GammaParameterization::MeanShape {
            if self.loss_fn != LossFn::Nll {
                eprintln!(
                    "gradientlss: Gamma MeanShape supports NLL only — keeping the \
                     concentration/rate parameterization"
                );
                return self;
            }
            for p in &mut self.params {
                p.response_fn = ResponseFn::Exp;
            }
        }
        self.parameterization = parameterization;
        self
    }

    /// The `MeanShape` head (see [`GammaParameterization`]): boosting in
    /// (log µ, log α), exact-MLE start values, outputs still reported as
    /// (concentration, rate). `LossFn::Crps` is not supported in this mode
    /// (falls back to the legacy parameterization with a warning).
    pub fn new_mean_shape(stabilization: Stabilization, loss_fn: LossFn) -> Self {
        if loss_fn == LossFn::Crps {
            eprintln!(
                "gradientlss: Gamma MeanShape supports NLL only — using the \
                 concentration/rate parameterization for the CRPS loss"
            );
            return Self::new(stabilization, ResponseFn::Exp, loss_fn, true);
        }
        // Column NAMES keep the output convention (see the enum docs).
        Self::new(stabilization, ResponseFn::Exp, loss_fn, true)
            .with_parameterization(GammaParameterization::MeanShape)
    }

    pub fn parameterization(&self) -> GammaParameterization {
        self.parameterization
    }

    /// `MeanShape`: raw (f, g) = (log µ, log α) → (α, β = α/µ), clamped.
    fn mean_shape_to_conc_rate(f: f64, g: f64) -> (f64, f64) {
        let f = if f.is_finite() { f.clamp(GAMMA_LOG_MEAN_CLAMP.0, GAMMA_LOG_MEAN_CLAMP.1) } else { 0.0 };
        let g = if g.is_finite() { g.clamp(GAMMA_LOG_SHAPE_CLAMP.0, GAMMA_LOG_SHAPE_CLAMP.1) } else { 0.0 };
        let alpha = g.exp();
        (alpha, (g - f).exp())
    }

    /// Exact Gamma MLE on `y > 0`: µ̂ = ȳ; α̂ solves ln α − ψ(α) = s with
    /// s = ln ȳ − mean(ln y) ≥ 0 (Newton in ln α from Minka's closed-form
    /// approximation). Returns raw (ln µ̂, ln α̂).
    fn mean_shape_mle(y: &[f64]) -> (f64, f64) {
        let pos: Vec<f64> = y.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
        if pos.is_empty() {
            return (0.0, 0.0);
        }
        let n = pos.len() as f64;
        let mean = pos.iter().sum::<f64>() / n;
        let mean_ln = pos.iter().map(|v| v.ln()).sum::<f64>() / n;
        let s = (mean.ln() - mean_ln).max(1e-12);
        // Minka (2002) starting point
        let mut a = (3.0 - s + ((s - 3.0) * (s - 3.0) + 24.0 * s).sqrt()) / (12.0 * s);
        for _ in 0..50 {
            // h(ln a) = ln a − ψ(a) − s ; dh/d ln a = 1 − aψ′(a) = −(a²ψ′−a)/a
            let h = -digamma_minus_ln(a) - s;
            let dh = -shape_fisher(a) / a;
            let step = h / dh;
            let new_ln = (a.ln() - step).clamp(GAMMA_LOG_SHAPE_CLAMP.0, GAMMA_LOG_SHAPE_CLAMP.1);
            let done = (new_ln - a.ln()).abs() < 1e-12;
            a = new_ln.exp();
            if done {
                break;
            }
        }
        (mean.ln(), a.ln())
    }

    /// `MeanShape` NLL gradients / Hessians in raw (f, g) = (log µ, log α).
    /// With r = y/µ, NLL = −α ln α + α ln µ − (α−1) ln y + α r + lnΓ(α):
    ///   ∂/∂f = α(1 − r)                      obs α r,   Fisher α
    ///   ∂/∂g = α[(ψ(α) − ln α) + (r − 1 − ln r)]
    ///   ∂²/∂g² = ∂/∂g + α²ψ′(α) − α           Fisher α²ψ′(α) − α
    fn mean_shape_gradients(
        &self,
        predictions: &ArrayView2<f64>,
        y: &ndarray::ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let n = predictions.nrows();
        let mode = self.hessian_mode;
        let compute = |i: usize| -> [f64; 4] {
            let yi = y[i];
            let f_raw = predictions[[i, 0]];
            let g_raw = predictions[[i, 1]];
            let f = if f_raw.is_finite() { f_raw.clamp(GAMMA_LOG_MEAN_CLAMP.0, GAMMA_LOG_MEAN_CLAMP.1) } else { 0.0 };
            let g = if g_raw.is_finite() { g_raw.clamp(GAMMA_LOG_SHAPE_CLAMP.0, GAMMA_LOG_SHAPE_CLAMP.1) } else { 0.0 };
            let alpha = g.exp();
            let h_f_fisher = alpha;
            let h_g_fisher = shape_fisher(alpha);
            if !yi.is_finite() || yi <= 0.0 {
                return [0.0, h_f_fisher, 0.0, h_g_fisher];
            }
            let ln_r = yi.ln() - f;
            let r = ln_r.exp();
            let gf = alpha * (1.0 - r);
            // r − 1 − ln r, cancellation-safe near r = 1
            let dev = if ln_r.abs() < 1e-3 {
                let x = ln_r;
                x * x / 2.0 + x * x * x / 6.0 + x * x * x * x / 24.0
            } else {
                r - 1.0 - ln_r
            };
            let gg = alpha * (digamma_minus_ln(alpha) + dev);
            let hf_obs = alpha * r;
            let hg_obs = gg + h_g_fisher;
            let (hf, hg) = match mode {
                HessianMode::Observed => (hf_obs, hg_obs),
                HessianMode::Fisher => (h_f_fisher, h_g_fisher),
                HessianMode::Damped => (hf_obs.max(h_f_fisher), hg_obs.max(h_g_fisher)),
            };
            // a clamped coordinate has no gradient there
            let gf = if f != f_raw { 0.0 } else { gf };
            let gg = if g != g_raw { 0.0 } else { gg };
            [gf, hf, gg, hg]
        };
        let rows: Vec<[f64; 4]> = if n >= 256 {
            (0..n).into_par_iter().map(compute).collect()
        } else {
            (0..n).map(compute).collect()
        };
        let mut gradients = Array2::zeros((n, 2));
        let mut hessians = Array2::zeros((n, 2));
        for (i, r) in rows.into_iter().enumerate() {
            gradients[[i, 0]] = r[0];
            hessians[[i, 0]] = r[1];
            gradients[[i, 1]] = r[2];
            hessians[[i, 1]] = r[3];
        }
        (gradients, hessians)
    }

    /// Choose the curvature handed to the Newton step (see [`HessianMode`]).
    pub fn with_hessian_mode(mut self, mode: HessianMode) -> Self {
        self.hessian_mode = mode;
        self
    }

    pub fn hessian_mode(&self) -> HessianMode {
        self.hessian_mode
    }

    /// Fisher information of −log f in RAW space, [concentration, rate]:
    /// I_αα = ψ′(α), I_ββ = α/β², chained with f′² (the f″ chain term has
    /// expectation zero). Under `Exp`: α²ψ′(α) and α.
    fn fisher_hessians_raw(alpha: f64, beta: f64, rd0: f64, rd1: f64) -> [f64; 2] {
        [trigamma(alpha) * rd0 * rd0, alpha / (beta * beta) * rd1 * rd1]
    }

    pub fn default() -> Self {
        Self::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false)
    }

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let shape = params[0]; // concentration
        let rate = params[1];

        if shape <= 0.0 || rate <= 0.0 || target < 0.0 {
            return f64::NEG_INFINITY;
        }
        if target == 0.0 {
            // shape=1 is the exponential special case
            if (shape - 1.0).abs() < f64::EPSILON {
                return rate.ln() - rate * target;
            }
            return f64::NEG_INFINITY;
        }
        if target.is_infinite() {
            return f64::NEG_INFINITY;
        }

        // ln_pdf = shape * ln(rate) + (shape - 1) * ln(x) - rate * x - ln_gamma(shape)
        shape * rate.ln() + (shape - 1.0) * target.ln() - rate * target - ln_gamma(shape)
    }
}

#[typetag::serde]
impl Distribution for Gamma {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Gamma"
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

    /// Measured recipe (research cycles 5–6): watch the shape of the
    /// mean/shape head; the legacy (concentration, rate) head gets none. The
    /// mean stays under the global early stopping. NLL loss only.
    fn default_param_early_stopping_params(&self) -> Option<Vec<usize>> {
        (self.parameterization == GammaParameterization::MeanShape && self.loss_fn() == LossFn::Nll).then(|| vec![1])
    }

    fn should_initialize(&self) -> bool {
        self.initialize
    }

    /// The mean coordinate of the orthogonal head (none for the legacy head).
    fn location_param_index(&self) -> Option<usize> {
        (self.parameterization == GammaParameterization::MeanShape).then_some(0)
    }

    fn default_start_values(&self, target: &ResponseData) -> Option<ndarray::Array1<f64>> {
        if !(self.location_start || self.wide_start) || self.parameterization != GammaParameterization::MeanShape {
            return None;
        }
        let ResponseData::Univariate(y) = target else { return None };
        let pos: Vec<f64> = y.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
        if pos.is_empty() {
            return None;
        }
        let mean = pos.iter().sum::<f64>() / pos.len() as f64;
        let offset = if self.location_start { 0.0 } else { GAMMA_WIDE_START_LOG_MEAN_OFFSET };
        Some(ndarray::Array1::from_vec(vec![mean.ln() + offset, 0.5]))
    }

    /// Sampling uses rejection sampling (Marsaglia–Tsang), which is not a
    /// smooth function of the parameters under a fixed seed — CRPS finite
    /// differences through it are meaningless (torch has no rsample here either).
    fn has_reparameterizable_sampler(&self) -> bool {
        false
    }

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_scalar(params, target[0])
    }

    fn transform_params(&self, predictions: &ArrayView2<f64>) -> Array2<f64> {
        if self.parameterization == GammaParameterization::ConcentrationRate {
            let mut result = Array2::zeros(predictions.dim());
            for (i, param) in self.params.iter().enumerate() {
                let col = predictions.column(i);
                let mut out_col = result.column_mut(i);
                param.response_fn.apply_into(&col, &mut out_col);
            }
            return result;
        }
        // MeanShape: a JOINT map — the rate needs both raw scores.
        let n = predictions.nrows();
        let mut result = Array2::zeros((n, 2));
        for i in 0..n {
            let (a, b) = Self::mean_shape_to_conc_rate(predictions[[i, 0]], predictions[[i, 1]]);
            result[[i, 0]] = a;
            result[[i, 1]] = b;
        }
        result
    }

    fn calculate_start_values(
        &self,
        target: &ResponseData,
        max_iter: usize,
    ) -> crate::error::Result<(f64, ndarray::Array1<f64>)> {
        if self.parameterization == GammaParameterization::ConcentrationRate {
            return self.calculate_start_values_default(target, max_iter);
        }
        let y: Vec<f64> = match target {
            ResponseData::Univariate(arr) => arr.to_vec(),
            ResponseData::Multivariate(_) => {
                return Err(crate::error::GradientLSSError::InvalidInput(
                    "Gamma is univariate".to_string(),
                ));
            }
        };
        let (f, g) = Self::mean_shape_mle(&y);
        let (a, b) = Self::mean_shape_to_conc_rate(f, g);
        Ok((self.compute_total_loss(&[a, b], target), ndarray::Array1::from_vec(vec![f, g])))
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                // Inline NLL to avoid per-element function call + [[i,j]] indexing overhead.
                // NLL_i = -shape*ln(rate) - (shape-1)*ln(y) + rate*y + ln_gamma(shape)
                let conc_col = params.column(0);
                let rate_col = params.column(1);
                crate::distributions::util::par_nansum(y.len(), |i| {
                    let y_val = y[i];
                    if y_val < 0.0 {
                        f64::INFINITY
                    } else {
                        -self.log_prob_scalar(&[conc_col[i], rate_col[i]], y_val)
                    }
                })
            }
            ResponseData::Multivariate(_) => panic!("Gamma is a univariate distribution."),
        }
    }

    /// Analytical gradients for Gamma distribution.
    ///
    /// NLL = ln_gamma(α) - α*ln(β) - (α-1)*ln(y) + β*y
    ///
    /// Gradients w.r.t. distribution parameters:
    /// - ∂NLL/∂α = digamma(α) - ln(β) - ln(y)
    /// - ∂NLL/∂β = -α/β + y
    ///
    /// Hessians w.r.t. distribution parameters:
    /// - ∂²NLL/∂α² = trigamma(α)
    /// - ∂²NLL/∂β² = α/β²
    ///
    /// Chain rule applied for response function: ∂NLL/∂pred = ∂NLL/∂param * ∂param/∂pred
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

        if self.parameterization == GammaParameterization::MeanShape {
            return Some(self.mean_shape_gradients(predictions, y));
        }

        let n_samples = predictions.nrows();

        let conc_response_fn = &self.params[0].response_fn;
        let rate_response_fn = &self.params[1].response_fn;

        let t_conc = transformed.column(0);
        let t_rate = transformed.column(1);
        let p_conc = predictions.column(0);
        let p_rate = predictions.column(1);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        if n_samples >= 4096 {
            // Pre-compute batch derivatives for both parameters
            let (rd_conc, rsd_conc) =
                conc_response_fn.derivative_batches_from_transformed(&p_conc, &t_conc);
            let (rd_rate, rsd_rate) =
                rate_response_fn.derivative_batches_from_transformed(&p_rate, &t_rate);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let alpha = t_conc[i].max(1e-6);
                let beta = t_rate[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || !yi.is_finite() {
                    return (0.0, 0.0, 0.0, 0.0);
                }

                let ln_y = yi.ln();
                let ln_beta = beta.ln();

                let grad_alpha = digamma(alpha) - ln_beta - ln_y;
                let hess_alpha = trigamma(alpha);
                let grad_beta = -alpha / beta + yi;
                let hess_beta = alpha / (beta * beta);

                let rd0 = rd_conc[i];
                let rsd0 = rsd_conc[i];
                let g0 = grad_alpha * rd0;
                let h0 = hess_alpha * rd0 * rd0 + grad_alpha * rsd0;

                let rd1 = rd_rate[i];
                let rsd1 = rsd_rate[i];
                let g1 = grad_beta * rd1;
                let h1 = hess_beta * rd1 * rd1 + grad_beta * rsd1;

                let (h0, h1) = match self.hessian_mode {
                    HessianMode::Observed => (h0, h1),
                    HessianMode::Fisher => {
                        let f = Self::fisher_hessians_raw(alpha, beta, rd0, rd1);
                        (f[0], f[1])
                    }
                    HessianMode::Damped => {
                        let f = Self::fisher_hessians_raw(alpha, beta, rd0, rd1);
                        (h0.max(f[0]), h1.max(f[1]))
                    }
                };
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
                let alpha = t_conc[i].max(1e-6);
                let beta = t_rate[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || !yi.is_finite() {
                    gradients[[i, 0]] = 0.0;
                    hessians[[i, 0]] = 1e-6;
                    gradients[[i, 1]] = 0.0;
                    hessians[[i, 1]] = 1e-6;
                    continue;
                }

                let ln_y = yi.ln();
                let ln_beta = beta.ln();

                let grad_alpha = digamma(alpha) - ln_beta - ln_y;
                let hess_alpha = trigamma(alpha);
                let grad_beta = -alpha / beta + yi;
                let hess_beta = alpha / (beta * beta);

                let rd0 = conc_response_fn.derivative(p_conc[i]);
                let rsd0 = conc_response_fn.second_derivative(p_conc[i]);
                let g0 = grad_alpha * rd0;
                let h0 = hess_alpha * rd0 * rd0 + grad_alpha * rsd0;

                let rd1 = rate_response_fn.derivative(p_rate[i]);
                let rsd1 = rate_response_fn.second_derivative(p_rate[i]);
                let g1 = grad_beta * rd1;
                let h1 = hess_beta * rd1 * rd1 + grad_beta * rsd1;

                let (h0, h1) = match self.hessian_mode {
                    HessianMode::Observed => (h0, h1),
                    HessianMode::Fisher => {
                        let f = Self::fisher_hessians_raw(alpha, beta, rd0, rd1);
                        (f[0], f[1])
                    }
                    HessianMode::Damped => {
                        let f = Self::fisher_hessians_raw(alpha, beta, rd0, rd1);
                        (h0.max(f[0]), h1.max(f[1]))
                    }
                };
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
            let concentration = params[[j, 0]];
            let rate = params[[j, 1]];

            if concentration > 0.0 && rate > 0.0 {
                // rand_distr::Gamma uses shape and scale (1/rate) parameterization
                if let Ok(dist) = RandGamma::new(concentration, 1.0 / rate) {
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
    fn test_gamma_creation() {
        let dist = Gamma::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["concentration", "rate"]);
        assert!(!dist.should_initialize());
    }

    #[test]
    fn test_gamma_log_prob() {
        let dist = Gamma::default();
        let log_p = dist.log_prob_scalar(&[2.0, 1.0], 1.0);
        // Gamma(shape=2, rate=1) at x=1: shape*ln(rate) + (shape-1)*ln(x) - rate*x - ln_gamma(shape)
        // = 2*ln(1) + 1*ln(1) - 1*1 - ln_gamma(2) = 0 + 0 - 1 - 0 = -1.0
        let expected = -1.0;
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_nll() {
        let dist = Gamma::default();
        let params = array![[2.0, 1.0], [2.0, 1.0]];
        let target = array![1.0, 1.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        // Gamma(shape=2, rate=1) at x=1: ln_pdf = -1.0
        let expected_single = 1.0;
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }

    #[test]
    fn test_trigamma() {
        // trigamma(1) = π²/6
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert_relative_eq!(trigamma(1.0), expected, epsilon = 1e-8);
        // trigamma(2) = π²/6 - 1
        assert_relative_eq!(trigamma(2.0), expected - 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_gamma_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        // Legacy head: the numerical path differentiates per-parameter
        // response functions, which the joint MeanShape map does not have
        // (MeanShape has its own FD test in `mean_shape_tests`).
        let dist = Gamma::default().with_parameterization(GammaParameterization::ConcentrationRate);
        let predictions = array![[0.5, 0.3], [-0.2, 0.8], [1.0, -0.5]];
        let targets = array![1.5, 0.8, 2.0];
        let target = ResponseData::Univariate(&targets.view());

        // Get analytical gradients
        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");

        // Get numerical gradients (call the numerical method directly)
        let numerical = dist
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("Should return numerical gradients");

        // Compare gradients
        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn test_gamma_sample() {
        let dist = Gamma::default();
        let params = array![[2.0, 1.0], [3.0, 2.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        assert_eq!(samples.dim(), (1000, 2));

        let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_0, 2.0, epsilon = 0.2);

        let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_1, 1.5, epsilon = 0.2);
    }
}

#[cfg(test)]
mod hessian_mode_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution as _, Gamma as RandGamma};

    /// E_y[observed raw-space Hessian] = Fisher for both Gamma parameters
    /// (Exp links): α²ψ′(α) and α. Gradients identical across modes; Damped is
    /// the elementwise max.
    #[test]
    fn fisher_is_the_expectation_of_observed_and_damped_is_the_max() {
        for &(alpha, beta) in &[(0.7, 2.0), (3.0, 0.5), (15.0, 4.0)] {
            let n = 300_000;
            let mut rng = ChaCha8Rng::seed_from_u64(9);
            let g = RandGamma::new(alpha, 1.0 / beta).unwrap(); // rand_distr uses scale = 1/rate
            let y: Array1<f64> = Array1::from_iter((0..n).map(|_| g.sample(&mut rng)));
            let preds = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { alpha.ln() } else { beta.ln() });
            let target = ResponseData::Univariate(&y.view());
            let mk = |m| Gamma::default().with_parameterization(GammaParameterization::ConcentrationRate).with_hessian_mode(m);
            let tr = mk(HessianMode::Observed).transform_params(&preds.view());
            let (g_o, h_o) = mk(HessianMode::Observed).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
            let (g_f, h_f) = mk(HessianMode::Fisher).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
            let (g_d, h_d) = mk(HessianMode::Damped).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
            assert_eq!(g_o, g_f);
            assert_eq!(g_o, g_d);
            let expect = [alpha * alpha * trigamma(alpha), alpha];
            for j in 0..2 {
                let fis = h_f[[0, j]];
                assert!((fis - expect[j]).abs() < 1e-6 * expect[j].max(1.0) + 1e-5, "{j}: {fis} vs {}", expect[j]);
                let mean_obs = h_o.column(j).mean().unwrap();
                assert!((mean_obs - fis).abs() / fis < 0.03, "α={alpha} β={beta} param {j}: E[obs]={mean_obs} Fisher={fis}");
                for i in 0..n {
                    assert_eq!(h_d[[i, j]], h_o[[i, j]].max(h_f[[i, j]]));
                }
            }
            // the observed concentration curvature really does go negative for large yβ
            assert!(h_o.column(0).iter().any(|&h| h < 0.0) || alpha > 10.0);
        }
    }

    #[test]
    fn default_is_damped_and_serde_default_holds() {
        assert_eq!(Gamma::default().hessian_mode(), HessianMode::Damped);
        let mut v: serde_json::Value = serde_json::to_value(Gamma::default()).unwrap();
        v.as_object_mut().unwrap().remove("hessian_mode");
        let back: Gamma = serde_json::from_value(v).unwrap();
        assert_eq!(back.hessian_mode(), HessianMode::Damped);
    }
}

#[cfg(test)]
mod mean_shape_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution as _, Gamma as RandGamma};

    fn draws(alpha: f64, mean: f64, n: usize, seed: u64) -> Array1<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let g = RandGamma::new(alpha, mean / alpha).unwrap();
        Array1::from_iter((0..n).map(|_| g.sample(&mut rng)))
    }

    /// Analytic gradients/Hessians match central finite differences of the
    /// per-row NLL in raw (log µ, log α) — including the large-α series paths.
    #[test]
    fn gradients_match_finite_differences() {
        let d = Gamma::new_mean_shape(Stabilization::None, LossFn::Nll).with_hessian_mode(HessianMode::Observed);
        for &(alpha, mean) in &[(0.8, 2.0), (6.0, 0.7), (330.0, 450.0), (12000.0, 450.0)] {
            let y = draws(alpha * 1.3, mean * 1.05, 40, 3);
            let (f0, g0) = (mean.ln(), alpha.ln());
            let preds = Array2::from_shape_fn((y.len(), 2), |(_, j)| if j == 0 { f0 } else { g0 });
            let (g, h) = d.analytical_gradients(&preds.view(), &d.transform_params(&preds.view()).view(), &ResponseData::Univariate(&y.view())).unwrap();
            let nll = |f: f64, gg: f64, yi: f64| {
                let (a, b) = Gamma::mean_shape_to_conc_rate(f, gg);
                -d.log_prob(&[a, b], &[yi])
            };
            let eps = 1e-4;
            for i in 0..y.len() {
                let yi = y[i];
                let df = (nll(f0 + eps, g0, yi) - nll(f0 - eps, g0, yi)) / (2.0 * eps);
                let dg = (nll(f0, g0 + eps, yi) - nll(f0, g0 - eps, yi)) / (2.0 * eps);
                let hff = (nll(f0 + eps, g0, yi) - 2.0 * nll(f0, g0, yi) + nll(f0 - eps, g0, yi)) / (eps * eps);
                let hgg = (nll(f0, g0 + eps, yi) - 2.0 * nll(f0, g0, yi) + nll(f0, g0 - eps, yi)) / (eps * eps);
                let tol = |x: f64| 1e-4 * x.abs().max(1.0) + 5e-3 * (alpha / 1000.0).max(1e-3);
                assert!((g[[i, 0]] - df).abs() < tol(df), "α={alpha} grad f {} vs {df}", g[[i, 0]]);
                assert!((g[[i, 1]] - dg).abs() < tol(dg), "α={alpha} grad g {} vs {dg}", g[[i, 1]]);
                assert!((h[[i, 0]] - hff).abs() < tol(hff) * 10.0, "α={alpha} hess f {} vs {hff}", h[[i, 0]]);
                if alpha < 1000.0 {
                    // FD second differences of a 1e5-sized NLL are noise at α = 1.2e4
                    assert!((h[[i, 1]] - hgg).abs() < tol(hgg) * 10.0, "α={alpha} hess g {} vs {hgg}", h[[i, 1]]);
                }
            }
        }
    }

    /// Fisher mode is the expectation of Observed; the mixed term is zero
    /// (orthogonality) — checked by Monte Carlo on the gradient product.
    #[test]
    fn fisher_is_diagonal_and_matches_observed_expectation() {
        for &(alpha, mean) in &[(2.0, 3.0), (330.0, 450.0)] {
            let n = 300_000;
            let y = draws(alpha, mean, n, 11);
            let preds = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { mean.ln() } else { alpha.ln() });
            let obs = Gamma::new_mean_shape(Stabilization::None, LossFn::Nll).with_hessian_mode(HessianMode::Observed);
            let fis = Gamma::new_mean_shape(Stabilization::None, LossFn::Nll).with_hessian_mode(HessianMode::Fisher);
            let tr = obs.transform_params(&preds.view());
            let t = ResponseData::Univariate(&y.view());
            let (g, h_o) = obs.analytical_gradients(&preds.view(), &tr.view(), &t).unwrap();
            let (_, h_f) = fis.analytical_gradients(&preds.view(), &tr.view(), &t).unwrap();
            for j in 0..2 {
                let rel = (h_o.column(j).mean().unwrap() - h_f[[0, j]]).abs() / h_f[[0, j]];
                assert!(rel < 0.03, "α={alpha} param {j}: E[obs] vs Fisher rel {rel}");
            }
            // E[g_f g_g] ≈ 0 relative to sqrt(I_ff I_gg)
            let cross = (0..n).map(|i| g[[i, 0]] * g[[i, 1]]).sum::<f64>() / n as f64;
            let scale = (h_f[[0, 0]] * h_f[[0, 1]]).sqrt();
            assert!(cross.abs() / scale < 0.02, "α={alpha}: cross-information {cross} vs scale {scale}");
        }
    }

    /// Start values are the Gamma MLE (µ̂ = ȳ; α̂ the profile root), recovered
    /// across shape scales including α ≈ 1e4.
    #[test]
    fn start_values_recover_the_mle() {
        for &(alpha, mean) in &[(0.6, 5.0), (4.0, 0.01), (330.0, 450.0), (12000.0, 450.0)] {
            let y = draws(alpha, mean, 20_000, 5);
            let d = Gamma::new_mean_shape(Stabilization::None, LossFn::Nll);
            let (loss, sv) = d.calculate_start_values(&ResponseData::Univariate(&y.view()), 50).unwrap();
            assert!(loss.is_finite());
            assert!((sv[0].exp() / y.mean().unwrap() - 1.0).abs() < 1e-12);
            let a_hat = sv[1].exp();
            assert!((a_hat / alpha - 1.0).abs() < 0.06, "α̂ {a_hat} vs {alpha}");
            // and it really is the stationary point of the profile in α
            let s = y.mean().unwrap().ln() - y.mapv(f64::ln).mean().unwrap();
            assert!((-digamma_minus_ln(a_hat) - s).abs() < 1e-9 * s.max(1e-6), "profile equation residual");
        }
    }

    /// The `(concentration, rate)` output convention is preserved, and the
    /// new field defaults for old serialized models.
    #[test]
    fn output_convention_and_serde_default() {
        let d = Gamma::new_mean_shape(Stabilization::None, LossFn::Nll);
        let preds = Array2::from_shape_vec((1, 2), vec![450f64.ln(), 330f64.ln()]).unwrap();
        let t = d.transform_params(&preds.view());
        assert!((t[[0, 0]] - 330.0).abs() < 1e-9 && (t[[0, 1]] - 330.0 / 450.0).abs() < 1e-12);
        let mut v: serde_json::Value = serde_json::to_value(Gamma::default()).unwrap();
        v.as_object_mut().unwrap().remove("parameterization");
        let back: Gamma = serde_json::from_value(v).unwrap();
        assert_eq!(back.parameterization(), GammaParameterization::ConcentrationRate);
    }

    /// 2026-11-10 default: Exp + NLL → MeanShape, identical to the dedicated
    /// constructor; other response functions / CRPS stay ConcentrationRate;
    /// the opt-out is exact.
    #[test]
    fn mean_shape_is_the_exp_nll_default() {
        let d = Gamma::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);
        assert_eq!(d.parameterization(), GammaParameterization::MeanShape);
        assert_eq!(Gamma::default().parameterization(), GammaParameterization::MeanShape);
        for (rf, lf) in [(ResponseFn::Softplus, LossFn::Nll), (ResponseFn::Exp, LossFn::Crps)] {
            assert_eq!(Gamma::new(Stabilization::None, rf, lf, true).parameterization(), GammaParameterization::ConcentrationRate);
        }
        let ms = Gamma::new_mean_shape(Stabilization::None, LossFn::Nll);
        let preds = Array2::from_shape_vec((2, 2), vec![6.1, 5.8, -0.3, 0.4]).unwrap();
        let y = Array1::from_vec(vec![450.0, 0.9]);
        let target = ResponseData::Univariate(&y.view());
        let (t_d, t_ms) = (d.transform_params(&preds.view()), ms.transform_params(&preds.view()));
        assert_eq!(t_d, t_ms);
        assert_eq!(
            d.analytical_gradients(&preds.view(), &t_d.view(), &target),
            ms.analytical_gradients(&preds.view(), &t_ms.view(), &target)
        );
        let legacy = d.with_parameterization(GammaParameterization::ConcentrationRate);
        let t_l = legacy.transform_params(&preds.view());
        assert!((t_l[[0, 0]] - ResponseFn::Exp.apply_scalar(6.1)).abs() < 1e-9, "{}", t_l[[0, 0]]);
        assert!((t_l[[0, 1]] - ResponseFn::Exp.apply_scalar(5.8)).abs() < 1e-9, "{}", t_l[[0, 1]]);
    }
}

#[cfg(test)]
mod location_start_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn location_start_sets_log_mean_only() {
        let y = array![400.0, 450.0, 500.0];
        let t = ResponseData::Univariate(&y.view());
        assert!(Gamma::default().with_wide_start(false).default_start_values(&t).is_none());
        let sv = Gamma::default().with_location_start(true).default_start_values(&t).unwrap();
        assert_eq!(sv.to_vec(), vec![450.0f64.ln(), 0.5]);
        // the legacy head has no location coordinate
        let cr = Gamma::default().with_parameterization(GammaParameterization::ConcentrationRate).with_location_start(true);
        assert!(cr.default_start_values(&t).is_none());
    }

    /// The default start is unit-free: rescaling y shifts log µ₀ by ln c only.
    #[test]
    fn wide_start_is_the_default_and_unit_free() {
        let y = array![400.0, 450.0, 500.0];
        let t = ResponseData::Univariate(&y.view());
        let d = Gamma::default();
        assert!(d.wide_start());
        let sv = d.default_start_values(&t).unwrap();
        assert!((sv[0] - (450.0f64.ln() + 0.5)).abs() < 1e-12 && sv[1] == 0.5);
        let y2 = &y * 100.0;
        let t2 = ResponseData::Univariate(&y2.view());
        let sv2 = d.default_start_values(&t2).unwrap();
        assert!((sv2[0] - sv[0] - 100.0f64.ln()).abs() < 1e-12 && sv2[1] == sv[1]);
    }

    #[test]
    fn serde_default_is_the_raw_start() {
        let mut v = serde_json::to_value(Gamma::default()).unwrap();
        v.as_object_mut().unwrap().remove("wide_start");
        let back: Gamma = serde_json::from_value(v).unwrap();
        assert!(!back.wide_start());
    }
}
