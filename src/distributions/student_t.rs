//! Student-T distribution implementation.

use super::base::{Distribution, DistributionParam, HessianMode, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, StudentT as RandStudentT};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::constants::{LOG_PI, trigamma};

/// Above this ν the distribution is treated as its Gaussian limit: the
/// digamma/trigamma/ln_gamma differences that make up the df score and
/// curvature are O(1/ν²) true values computed from O(ln ν) terms, so f64
/// cancellation noise dominates, and the ExpDf chain factor e^η ≈ ν then
/// amplifies that noise into O(1) per-row garbage that can throw the df margin
/// to ±∞ in a single boosting round. ν is not identifiable there anyway.
const LARGE_DF: f64 = 1e8;

/// Lowest raw `ExpDf` start value handed to the booster: ν₀ = 2 + e^{ln 0.5} = 2.5.
const MIN_START_DF_RAW: f64 = -0.693_147_180_559_945_3; // ln(0.5)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentT {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
    /// Curvature handed to the booster (see [`HessianMode`]); defaults to
    /// `Damped`. The field only affects training, so models saved before it
    /// existed load fine.
    #[serde(default)]
    hessian_mode: HessianMode,
    /// Cap on the fitted unconditional start value of ν (`initialize: true`
    /// only). On Gaussian-like marginals the unconditional MLE is ν → ∞,
    /// where the df score is ≈ 0 and the df tree can never learn conditional
    /// heavy tails; starting at a moderate ν keeps that gradient alive.
    /// Default `Some(10.0)` since 2026-11-09: in 5-seed paired runs it was
    /// better on protein (−0.067 nats) and power_plant and within noise
    /// everywhere else (research/gradientlss/reports/003-cycle2-report.md).
    /// `None` disables the cap. The field only affects training, so saved
    /// models load unchanged.
    #[serde(default = "StudentT::default_max_start_df")]
    max_start_df: Option<f64>,
    /// Per-parameter gradient multipliers (effective per-parameter learning
    /// rates), see `Distribution::param_gradient_scales`. `None` = all ones.
    #[serde(default)]
    param_gradient_scales: Option<Vec<f64>>,
    /// β-NLL row weighting (Seitzer et al. 2022): every row's gradient and
    /// Hessian are multiplied by σ̂^{2β}, σ̂ treated as a constant. β = 0 (or
    /// `None`) is plain NLL; β = 1 gives the location tree MSE-like row
    /// influence; β = 0.5 is the paper's recommendation.
    #[serde(default)]
    beta_nll: Option<f64>,
    /// Start values used without `initialize` (see [`StudentTStart`]).
    /// Models saved before the field existed load with `Raw`; it only
    /// affects training.
    #[serde(default = "StudentTStart::legacy")]
    start_rule: StudentTStart,
    /// Start ν for the closed-form starts (`Median`, `MedianMad`); `None` =
    /// the raw 0.5 (ν ≈ 3.65).
    #[serde(default)]
    start_df: Option<f64>,
}

/// Closed-form start values for a `StudentT` built without `initialize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudentTStart {
    /// Raw 0.5 for every parameter (ν ≈ 3.65, µ = 0.5, σ ≈ 1.65). On a
    /// target far from 0.5 the location never moves: the t score
    /// ∝ r/(νσ² + r²) vanishes for large residuals (power_plant, y ≈ 450:
    /// RMSE 448).
    Raw,
    /// Location at the median, df and scale raw.
    Median,
    /// Location at the median, scale at 1.4826·MAD, df raw.
    MedianMad,
    /// `Median`, but the scale starts at 3·sd(y) when the labels' sample
    /// excess kurtosis is below 1 ("approach the scale from above" only on
    /// light-tailed targets). Research option (proposal 013).
    Adaptive,
}

/// Excess-kurtosis gate of [`StudentTStart::Adaptive`] (a t₉ has 1).
pub const ADAPTIVE_KURTOSIS_GATE: f64 = 1.0;

/// Population standard deviation and sample excess kurtosis g₂ = m₄/m₂² − 3.
pub fn sd_and_excess_kurtosis(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    if n < 4.0 {
        return (0.0, f64::INFINITY);
    }
    let mean = v.iter().sum::<f64>() / n;
    let (mut m2, mut m4) = (0.0, 0.0);
    for x in v {
        let d = (x - mean) * (x - mean);
        m2 += d;
        m4 += d * d;
    }
    m2 /= n;
    m4 /= n;
    if !(m2 > 0.0) {
        return (0.0, f64::INFINITY);
    }
    (m2.sqrt(), m4 / (m2 * m2) - 3.0)
}

impl StudentTStart {
    fn legacy() -> Self {
        StudentTStart::Raw
    }
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
            hessian_mode: HessianMode::default(),
            max_start_df: Self::default_max_start_df(),
            param_gradient_scales: None,
            beta_nll: None,
            start_rule: StudentTStart::Median,
            start_df: None,
        }
    }

    /// Start ν used by the closed-form starts (see [`StudentTStart`]).
    pub fn with_start_df(mut self, df: f64) -> Self {
        self.start_df = Some(df);
        self
    }

    pub fn start_df(&self) -> Option<f64> {
        self.start_df
    }

    fn raw_start_df(&self) -> f64 {
        match self.start_df {
            None => 0.5,
            Some(df) => match self.params[0].response_fn {
                ResponseFn::ExpDf => (df - 2.0).max(1e-6).ln(),
                ref f => invert_response(f, df),
            },
        }
    }

    /// Start values used without `initialize` (see [`StudentTStart`]).
    pub fn with_start_rule(mut self, rule: StudentTStart) -> Self {
        self.start_rule = rule;
        self
    }

    pub fn start_rule(&self) -> StudentTStart {
        self.start_rule
    }

    /// Enable β-NLL row weighting with exponent `beta` (see the field docs).
    pub fn with_beta_nll(mut self, beta: f64) -> Self {
        self.beta_nll = Some(beta);
        self
    }

    pub fn beta_nll(&self) -> Option<f64> {
        self.beta_nll
    }

    /// Multiply the gradient of each parameter by `scales[j]` (order: df, loc,
    /// scale): an effective per-parameter learning rate η·scales[j]. Use e.g.
    /// `[2.0, 1.0, 0.5]` to speed the df tree and slow the scale tree.
    pub fn with_param_gradient_scales(mut self, scales: [f64; 3]) -> Self {
        self.param_gradient_scales = Some(scales.to_vec());
        self
    }

    /// The default start-df cap (see the field docs).
    pub const fn default_max_start_df() -> Option<f64> {
        Some(10.0)
    }

    /// Cap the fitted start value of ν at `max_df` (only affects
    /// `initialize: true`; see the field docs).
    pub fn with_max_start_df(mut self, max_df: f64) -> Self {
        self.max_start_df = Some(max_df);
        self
    }

    /// Disable the start-df cap: use the unconditional MLE of ν as the start.
    pub fn without_max_start_df(mut self) -> Self {
        self.max_start_df = None;
        self
    }

    pub fn max_start_df(&self) -> Option<f64> {
        self.max_start_df
    }

    /// Choose the curvature handed to the Newton step (see [`HessianMode`]).
    /// Gradients are unaffected.
    pub fn with_hessian_mode(mut self, mode: HessianMode) -> Self {
        self.hessian_mode = mode;
        self
    }

    pub fn hessian_mode(&self) -> HessianMode {
        self.hessian_mode
    }

    /// Diagonal of the expected (Fisher) information of −log f in RAW space,
    /// ordered [df, loc, scale]. `rd_df`/`rd_scale` are the response
    /// derivatives dν/dη, dσ/dη; the loc response is the identity.
    ///
    /// Per observation, in distribution space:
    ///   I_μμ = (ν+1)/((ν+3)σ²),  I_σσ = 2ν/((ν+3)σ²),
    ///   I_νν = ¼[ψ′(ν/2) − ψ′((ν+1)/2)] − (ν+5)/(2ν(ν+1)(ν+3)).
    /// Chained with `E[∂ℓ/∂θ]=0`, the raw-space value is `I_θθ·f′(η)²` (the
    /// `∂ℓ/∂θ·f″` term has expectation zero). The ν entry is a difference of
    /// two O(1/ν²) quantities whose result is O(1/ν⁴); it is floored at a
    /// tiny positive value so cancellation noise can never make it negative.
    fn fisher_hessians_raw(nu: f64, sigma: f64, rd_df: f64, rd_scale: f64) -> [f64; 3] {
        let half_nu = 0.5 * nu;
        let i_nu = 0.25 * (trigamma(half_nu) - trigamma(half_nu + 0.5))
            - (nu + 5.0) / (2.0 * nu * (nu + 1.0) * (nu + 3.0));
        let sigma_sq = sigma * sigma;
        let i_mu = (nu + 1.0) / ((nu + 3.0) * sigma_sq);
        let i_sigma = 2.0 * nu / ((nu + 3.0) * sigma_sq);
        [
            if nu > LARGE_DF { 0.0 } else { (i_nu * rd_df * rd_df).max(1e-12) },
            i_mu,
            i_sigma * rd_scale * rd_scale,
        ]
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

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let df = params[0];
        let loc = params[1];
        let scale = params[2];

        if df <= 0.0 || scale <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if target.is_infinite() {
            return f64::NEG_INFINITY;
        }

        let d = (target - loc) / scale;
        if df > LARGE_DF {
            // Gaussian limit. Past ~1e12 `ln_gamma((ν+1)/2) − ln_gamma(ν/2)`
            // is the difference of two ~1e14-sized numbers and carries O(1)
            // absolute error; the t density is within 1/ν of the normal
            // long before that.
            return -0.5 * d * d - 0.5 * (LOG_PI + 2.0f64.ln()) - scale.ln();
        }
        ln_gamma((df + 1.0) / 2.0)
            - 0.5 * ((df + 1.0) * (1.0 + d * d / df).ln())
            - ln_gamma(df / 2.0)
            - 0.5 * (df.ln() + LOG_PI)
            - scale.ln()
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

    fn param_gradient_scales(&self) -> Option<&[f64]> {
        self.param_gradient_scales.as_deref()
    }

    fn row_gradient_weights(&self, transformed: &ArrayView2<f64>) -> Option<Array1<f64>> {
        let beta = self.beta_nll?;
        if beta == 0.0 {
            return None;
        }
        let two_beta = 2.0 * beta;
        Some(transformed.column(2).mapv(|s| s.max(1e-12).powf(two_beta)))
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

    /// Measured recipe (research cycle 6): watch df and scale (annealed;
    /// a hard df freeze was harmful); the location stays under the global
    /// early stopping. NLL loss only.
    fn default_param_early_stopping_params(&self) -> Option<Vec<usize>> {
        (self.loss_fn() == LossFn::Nll).then(|| vec![0, 2])
    }

    fn should_initialize(&self) -> bool {
        self.initialize
    }

    fn default_start_values(&self, target: &ResponseData) -> Option<ndarray::Array1<f64>> {
        if self.start_rule == StudentTStart::Raw {
            return None;
        }
        let ResponseData::Univariate(y) = target else { return None };
        let mut v: Vec<f64> = y.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return None;
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            let n = v.len();
            if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
        };
        let med = median(&mut v);
        let df0 = self.raw_start_df();
        if self.start_rule == StudentTStart::Median {
            return Some(ndarray::Array1::from_vec(vec![df0, med, 0.5]));
        }
        if self.start_rule == StudentTStart::Adaptive {
            let (sd, g2) = sd_and_excess_kurtosis(&v);
            if g2 < ADAPTIVE_KURTOSIS_GATE && sd > 0.0 {
                let raw_scale = invert_response(&self.params[2].response_fn, 3.0 * sd);
                return Some(ndarray::Array1::from_vec(vec![df0, med, raw_scale]));
            }
            return Some(ndarray::Array1::from_vec(vec![df0, med, 0.5]));
        }
        let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
        let mut scale = 1.4826 * median(&mut dev);
        if !(scale > 0.0) {
            // ≥ half the labels tie: fall back to the standard deviation, then 1
            let n = v.len() as f64;
            let mean = v.iter().sum::<f64>() / n;
            scale = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
            if !(scale > 0.0) {
                scale = 1.0;
            }
        }
        let raw_scale = invert_response(&self.params[2].response_fn, scale);
        Some(ndarray::Array1::from_vec(vec![df0, med, raw_scale]))
    }

    /// Sampling uses rejection sampling (chi-square/Gamma draw for the df), which is not a
    /// smooth function of the parameters under a fixed seed — CRPS finite
    /// differences through it are meaningless (torch has no rsample here either).
    fn has_reparameterizable_sampler(&self) -> bool {
        false
    }

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_scalar(params, target[0])
    }

    fn calculate_start_values(
        &self,
        target: &ResponseData,
        max_iter: usize,
    ) -> crate::error::Result<(f64, ndarray::Array1<f64>)> {
        let (loss, mut sv) = self.calculate_start_values_default(target, max_iter)?;
        // Never START inside the ExpDf dead zone: at ν = 2 + e^η with η ≪ 0 the
        // chain factor dν/dη = ν − 2 kills both the df score and its curvature,
        // so a boosted model can never leave such a start. The unconditional
        // optimiser lands exactly there on very heavy-tailed residuals (naval);
        // ν₀ = 2.5 (dν/dη = 0.5) keeps the df learnable. Only the df coordinate
        // is touched, and only from below.
        let mut clamped = false;
        if self.params[0].response_fn == ResponseFn::ExpDf && sv[0] < MIN_START_DF_RAW {
            sv[0] = MIN_START_DF_RAW;
            clamped = true;
        }
        if clamped && self.max_start_df.is_none() {
            let transformed: Vec<f64> = sv
                .iter()
                .zip(self.params.iter())
                .map(|(&p, dp)| dp.response_fn.apply_scalar(p))
                .collect();
            return Ok((self.compute_total_loss(&transformed, target), sv));
        }
        if let Some(cap) = self.max_start_df {
            // Invert the df response at the cap (ExpDf: ν = e^η + 2).
            let raw_cap = match self.params[0].response_fn {
                ResponseFn::ExpDf => (cap - 2.0).max(1e-6).ln(),
                ResponseFn::Exp => cap.max(1e-6).ln(),
                _ => {
                    // generic: bisect the response for the raw value giving `cap`
                    let f = &self.params[0].response_fn;
                    let (mut lo, mut hi) = (-50.0, 50.0);
                    for _ in 0..200 {
                        let mid = 0.5 * (lo + hi);
                        if f.apply_scalar(mid) < cap { lo = mid } else { hi = mid }
                    }
                    0.5 * (lo + hi)
                }
            };
            if sv[0] > raw_cap {
                sv[0] = raw_cap;
                let transformed: Vec<f64> = sv
                    .iter()
                    .zip(self.params.iter())
                    .map(|(&p, dp)| dp.response_fn.apply_scalar(p))
                    .collect();
                return Ok((self.compute_total_loss(&transformed, target), sv));
            }
        }
        Ok((loss, sv))
    }

    /// Ranked by `dist_select` at the free maximum-likelihood fit: the start-df
    /// cap and the dead-zone floor are training devices, not part of the fit.
    fn marginal_fit(
        &self,
        target: &ResponseData,
        max_iter: usize,
    ) -> crate::error::Result<(f64, ndarray::Array1<f64>)> {
        self.calculate_start_values_default(target, max_iter)
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                let df_col = params.column(0);
                let loc_col = params.column(1);
                let scale_col = params.column(2);
                crate::distributions::util::par_nansum(y.len(), |i| {
                    -self.log_prob_scalar(&[df_col[i], loc_col[i], scale_col[i]], y[i])
                })
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
    /// NLL = -ln_gamma((ν+1)/2) + 0.5*(ν+1)*ln(1+d²/ν) + ln_gamma(ν/2) + 0.5*ln(νπ) + ln(σ)
    /// where d = (y-μ)/σ, ν = df.
    ///
    /// Gradients w.r.t. distribution parameters (let z = 1 + d²/ν):
    /// - ∂NLL/∂μ = -(ν+1)*d / (σ*(ν+d²))
    /// - ∂NLL/∂σ = 1/σ - (ν+1)*d² / (σ*(ν+d²))
    /// - ∂NLL/∂ν = -0.5*digamma((ν+1)/2) + 0.5*digamma(ν/2) + 0.5*ln(z) - 0.5*(ν+1)*d²/(ν²*z) + 0.5/ν
    ///
    /// Hessians w.r.t. distribution parameters:
    /// - ∂²NLL/∂μ² = (ν+1)*(ν-d²) / (σ²*(ν+d²)²)
    /// - ∂²NLL/∂σ² = -1/σ² + (ν+1)*d²*(3ν+d²) / (σ²*(ν+d²)²)
    /// - ∂²NLL/∂ν² = -0.25*trigamma((ν+1)/2) + 0.25*trigamma(ν/2) - 0.5/ν²
    ///               + d²*(2ν + d²*(1-ν)) / (2ν²*(ν+d²)²)
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

        let df_response_fn = &self.params[0].response_fn;
        let scale_response_fn = &self.params[2].response_fn;

        let t_df = transformed.column(0);
        let t_loc = transformed.column(1);
        let t_scale = transformed.column(2);
        let p_df = predictions.column(0);
        let p_scale = predictions.column(2);

        let mut gradients = Array2::zeros((n_samples, 3));
        let mut hessians = Array2::zeros((n_samples, 3));

        if n_samples >= 4096 {
            let (rd_df, rsd_df) = df_response_fn.derivative_batches_from_transformed(&p_df, &t_df);
            let (rd_scale, rsd_scale) =
                scale_response_fn.derivative_batches_from_transformed(&p_scale, &t_scale);

            let compute_sample = |i: usize| -> ([f64; 3], [f64; 3]) {
                let nu = t_df[i].max(1e-6);
                let mu = t_loc[i];
                let sigma = t_scale[i].max(1e-6);
                let yi = y[i];

                if !yi.is_finite() {
                    return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
                }

                let d = (yi - mu) / sigma;
                let d2 = d * d;
                let nu_plus_d2 = nu + d2;
                let z = 1.0 + d2 / nu;
                let nu_p1 = nu + 1.0;
                let sigma_sq = sigma * sigma;

                let g1 = -(nu_p1) * d / (sigma * nu_plus_d2);
                let h1 = nu_p1 / sigma_sq * (nu - d2) / (nu_plus_d2 * nu_plus_d2);

                let grad_sigma = 1.0 / sigma - nu_p1 * d2 / (sigma * nu_plus_d2);
                let hess_sigma_param = -1.0 / sigma_sq
                    + nu_p1 * d2 * (3.0 * nu + d2) / (sigma_sq * nu_plus_d2 * nu_plus_d2);
                let rs = rd_scale[i];
                let rss = rsd_scale[i];
                let g2 = grad_sigma * rs;
                let h2 = hess_sigma_param * rs * rs + grad_sigma * rss;

                let half_nu_p1 = nu_p1 / 2.0;
                let half_nu = nu / 2.0;
                let ln_z = z.ln();
                let grad_nu = -0.5 * digamma(half_nu_p1) + 0.5 * digamma(half_nu) + 0.5 * ln_z
                    - 0.5 * nu_p1 * d2 / (nu * nu * z)
                    + 0.5 / nu;
                // d/dν of the two data-dependent grad terms (0.5·ln z and
                // −(ν+1)d²/(2ν²z)) combines to d²(2ν + d²(1−ν))/(2ν²(ν+d²)²).
                let hess_nu_param = -0.25 * trigamma(half_nu_p1) + 0.25 * trigamma(half_nu)
                    - 0.5 / (nu * nu)
                    + d2 * (2.0 * nu + d2 * (1.0 - nu)) / (2.0 * nu * nu * nu_plus_d2 * nu_plus_d2);
                let rd = rd_df[i];
                let rsd = rsd_df[i];
                let (g0, h0) = if nu > LARGE_DF {
                    (0.0, 0.0)
                } else {
                    (grad_nu * rd, hess_nu_param * rd * rd + grad_nu * rsd)
                };

                match self.hessian_mode {
                    HessianMode::Observed => ([g0, g1, g2], [h0, h1, h2]),
                    HessianMode::Fisher => {
                        ([g0, g1, g2], Self::fisher_hessians_raw(nu, sigma, rd, rs))
                    }
                    HessianMode::Damped => {
                        let f = Self::fisher_hessians_raw(nu, sigma, rd, rs);
                        ([g0, g1, g2], [h0.max(f[0]), h1.max(f[1]), h2.max(f[2])])
                    }
                }
            };

            let results: Vec<_> = (0..n_samples).into_par_iter().map(compute_sample).collect();
            for (i, (g, h)) in results.into_iter().enumerate() {
                gradients[[i, 0]] = g[0];
                gradients[[i, 1]] = g[1];
                gradients[[i, 2]] = g[2];
                hessians[[i, 0]] = h[0];
                hessians[[i, 1]] = h[1];
                hessians[[i, 2]] = h[2];
            }
        } else {
            for i in 0..n_samples {
                let nu = t_df[i].max(1e-6);
                let mu = t_loc[i];
                let sigma = t_scale[i].max(1e-6);
                let yi = y[i];

                if !yi.is_finite() {
                    hessians[[i, 0]] = 1e-6;
                    hessians[[i, 1]] = 1e-6;
                    hessians[[i, 2]] = 1e-6;
                    continue;
                }

                let d = (yi - mu) / sigma;
                let d2 = d * d;
                let nu_plus_d2 = nu + d2;
                let z = 1.0 + d2 / nu;
                let nu_p1 = nu + 1.0;
                let sigma_sq = sigma * sigma;

                gradients[[i, 1]] = -(nu_p1) * d / (sigma * nu_plus_d2);
                hessians[[i, 1]] = nu_p1 / sigma_sq * (nu - d2) / (nu_plus_d2 * nu_plus_d2);

                let grad_sigma = 1.0 / sigma - nu_p1 * d2 / (sigma * nu_plus_d2);
                let hess_sigma_param = -1.0 / sigma_sq
                    + nu_p1 * d2 * (3.0 * nu + d2) / (sigma_sq * nu_plus_d2 * nu_plus_d2);
                let pred_scale = p_scale[i];
                let rs = scale_response_fn.derivative(pred_scale);
                let rss = scale_response_fn.second_derivative(pred_scale);
                gradients[[i, 2]] = grad_sigma * rs;
                hessians[[i, 2]] = hess_sigma_param * rs * rs + grad_sigma * rss;

                let half_nu_p1 = nu_p1 / 2.0;
                let half_nu = nu / 2.0;
                let ln_z = z.ln();
                let grad_nu = -0.5 * digamma(half_nu_p1) + 0.5 * digamma(half_nu) + 0.5 * ln_z
                    - 0.5 * nu_p1 * d2 / (nu * nu * z)
                    + 0.5 / nu;
                // See the parallel path: includes the data-dependent term from
                // differentiating 0.5·ln z − (ν+1)d²/(2ν²z) w.r.t. ν.
                let hess_nu_param = -0.25 * trigamma(half_nu_p1) + 0.25 * trigamma(half_nu)
                    - 0.5 / (nu * nu)
                    + d2 * (2.0 * nu + d2 * (1.0 - nu)) / (2.0 * nu * nu * nu_plus_d2 * nu_plus_d2);
                let pred_df = p_df[i];
                let rd = df_response_fn.derivative(pred_df);
                let rsd = df_response_fn.second_derivative(pred_df);
                if nu > LARGE_DF {
                    gradients[[i, 0]] = 0.0;
                    hessians[[i, 0]] = 0.0;
                } else {
                    gradients[[i, 0]] = grad_nu * rd;
                    hessians[[i, 0]] = hess_nu_param * rd * rd + grad_nu * rsd;
                }

                match self.hessian_mode {
                    HessianMode::Observed => {}
                    HessianMode::Fisher => {
                        let h = Self::fisher_hessians_raw(nu, sigma, rd, rs);
                        hessians[[i, 0]] = h[0];
                        hessians[[i, 1]] = h[1];
                        hessians[[i, 2]] = h[2];
                    }
                    HessianMode::Damped => {
                        let h = Self::fisher_hessians_raw(nu, sigma, rd, rs);
                        for j in 0..3 {
                            hessians[[i, j]] = hessians[[i, j]].max(h[j]);
                        }
                    }
                }
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
        // StudentT(df=5, loc=0, scale=1) at x=0: d=0
        // ln_gamma(3) - 0.5*(6*ln(1)) - ln_gamma(2.5) - 0.5*ln(5π) - ln(1)
        let expected = ln_gamma(3.0) - 0.0 - ln_gamma(2.5) - 0.5 * (5.0_f64.ln() + LOG_PI);
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_student_t_nll() {
        let dist = StudentT::default();
        let params = array![[5.0, 0.0, 1.0], [5.0, 0.0, 1.0]];
        let target = array![0.0, 0.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(ln_gamma(3.0) - ln_gamma(2.5) - 0.5 * (5.0_f64.ln() + LOG_PI));
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }

    #[test]
    fn test_student_t_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        // the numerical path differentiates the NLL, i.e. the OBSERVED curvature
        let dist = StudentT::default().with_hessian_mode(HessianMode::Observed);
        // Use predictions that produce reasonable transformed params
        let predictions = array![[0.5, 1.0, 0.3], [-0.2, 0.5, 0.8], [1.0, -0.5, 0.1]];
        let targets = array![1.5, 0.8, -1.0];
        let target = ResponseData::Univariate(&targets.view());

        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");

        let numerical = dist
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("Should return numerical gradients");

        // Compare gradients (allow looser tolerance for df parameter which involves digamma)
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-2);
                // Hessians must agree too — the df Hessian previously dropped its
                // data-dependent terms, which only a Hessian comparison catches.
                assert_relative_eq!(
                    analytical.1[[i, j]],
                    numerical.1[[i, j]],
                    epsilon = 1e-2,
                    max_relative = 1e-2
                );
            }
        }
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

#[cfg(test)]
mod hessian_mode_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::StudentT as RandT;

    fn raw_preds(n: usize, nu: f64, mu: f64, sigma: f64) -> Array2<f64> {
        // ExpDf: ν = e^η + 2 ; Exp: σ = e^η (ε-shifts negligible at this tolerance)
        Array2::from_shape_fn((n, 3), |(_, j)| match j {
            0 => (nu - 2.0).ln(),
            1 => mu,
            _ => sigma.ln(),
        })
    }

    /// The observed information averaged over y ~ t(ν, µ, σ) must equal the
    /// Fisher information in raw space, parameter by parameter. This ties the
    /// two modes together mathematically: Fisher is the expectation of the
    /// default. Observed-mode summands are bounded in d, so plain Monte Carlo
    /// converges at the usual rate.
    #[test]
    fn fisher_is_the_expectation_of_observed_in_raw_space() {
        for &(nu, mu, sigma) in &[(3.0, 0.3, 1.5), (6.0, -1.0, 0.4), (25.0, 2.0, 3.0)] {
            let n = 400_000;
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            let t = RandT::new(nu).unwrap();
            let y: Array1<f64> = Array1::from_iter((0..n).map(|_| mu + sigma * t.sample(&mut rng)));
            let preds = raw_preds(n, nu, mu, sigma);

            let observed = StudentT::default().with_hessian_mode(HessianMode::Observed);
            let fisher = StudentT::default().with_hessian_mode(HessianMode::Fisher);
            let transformed = observed.transform_params(&preds.view());
            let target = ResponseData::Univariate(&y.view());
            let (g_obs, h_obs) = observed
                .analytical_gradients(&preds.view(), &transformed.view(), &target)
                .unwrap();
            let (g_fis, h_fis) = fisher
                .analytical_gradients(&preds.view(), &transformed.view(), &target)
                .unwrap();

            // gradients are bitwise identical between modes
            assert_eq!(g_obs, g_fis);
            // Fisher hessians are constant across rows (same params) and > 0
            for j in 0..3 {
                let col = h_fis.column(j);
                assert!(col.iter().all(|&h| h > 0.0 && h.is_finite()), "param {j} not positive");
                let spread = col.iter().fold(0.0f64, |m, &h| m.max((h - col[0]).abs()));
                assert!(spread < 1e-12);
                let mean_obs = h_obs.column(j).mean().unwrap();
                let rel = (mean_obs - col[0]).abs() / col[0];
                assert!(
                    rel < 0.03,
                    "ν={nu} param {j}: E[observed]={mean_obs:.6e} vs Fisher={:.6e} (rel {rel:.3})",
                    col[0]
                );
            }
        }
    }

    /// Spot-check the raw-space closed forms against the distribution-space
    /// Fisher information chained by hand (ν=4, σ=1.3).
    #[test]
    fn fisher_raw_space_matches_hand_chain_rule() {
        let nu = 4.0;
        let sigma = 1.3;
        let h = StudentT::fisher_hessians_raw(nu, sigma, nu - 2.0, sigma);
        // I_νν(4) = ¼[ψ′(2) − ψ′(2.5)] − 9/(2·4·5·7); scipy: 0.0065012
        assert!((h[0] - 0.0065012 * (nu - 2.0).powi(2)).abs() < 1e-6, "{}", h[0]);
        assert!((h[1] - 5.0 / (7.0 * sigma * sigma)).abs() < 1e-12);
        assert!((h[2] - 8.0 / 7.0).abs() < 1e-12);
    }

    /// Both execution paths (serial < 4096 rows, parallel ≥ 4096) agree.
    #[test]
    fn fisher_serial_and_parallel_paths_agree() {
        let fisher = StudentT::default().with_hessian_mode(HessianMode::Fisher);
        let big = 4096;
        let preds = Array2::from_shape_fn((big, 3), |(i, j)| match j {
            0 => 0.3 + 0.001 * (i % 17) as f64,
            1 => (i % 5) as f64 * 0.2,
            _ => -0.2 + 0.0005 * (i % 11) as f64,
        });
        let y = Array1::from_iter((0..big).map(|i| ((i * 7919) % 101) as f64 / 20.0 - 2.5));
        let transformed = fisher.transform_params(&preds.view());
        let target = ResponseData::Univariate(&y.view());
        let (_, h_par) = fisher
            .analytical_gradients(&preds.view(), &transformed.view(), &target)
            .unwrap();
        let small = big - 1;
        let (_, h_ser) = fisher
            .analytical_gradients(
                &preds.slice(ndarray::s![..small, ..]),
                &transformed.slice(ndarray::s![..small, ..]),
                &ResponseData::Univariate(&y.slice(ndarray::s![..small])),
            )
            .unwrap();
        for i in 0..small {
            for j in 0..3 {
                assert!((h_par[[i, j]] - h_ser[[i, j]]).abs() <= 1e-12 * h_par[[i, j]].abs().max(1.0));
            }
        }
    }

    /// Above LARGE_DF the density is the Gaussian limit (continuous at the
    /// switch to ~1/ν) and the df score/curvature are exactly zero in every
    /// mode; log_prob stays sane where the naive formula is garbage.
    #[test]
    fn large_df_branch_is_continuous_and_silences_the_df_score() {
        let d = StudentT::default();
        for &x in &[-3.0, -0.3, 0.3, 2.5] {
            let below = d.log_prob(&[LARGE_DF * 0.999, 0.1, 1.3], &[x]);
            let above = d.log_prob(&[LARGE_DF * 1.001, 0.1, 1.3], &[x]);
            assert!((below - above).abs() < 1e-6, "{x}: {below} vs {above}");
            let huge = d.log_prob(&[1e16, 0.1, 1.3], &[x]);
            let gauss = -0.5 * ((x - 0.1f64) / 1.3).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln() - 1.3f64.ln();
            assert!((huge - gauss).abs() < 1e-9, "{x}: {huge} vs {gauss}");
        }
        for mode in [HessianMode::Observed, HessianMode::Fisher, HessianMode::Damped] {
            let dist = StudentT::default().with_hessian_mode(mode);
            let preds = Array2::from_shape_fn((8, 3), |(i, j)| match j {
                0 => 30.0 + i as f64, // ν = e^30.. ≫ LARGE_DF
                1 => 0.0,
                _ => 0.0,
            });
            let y = Array1::from_vec(vec![-2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0, 5.0]);
            let tr = dist.transform_params(&preds.view());
            let (g, h) = dist
                .analytical_gradients(&preds.view(), &tr.view(), &ResponseData::Univariate(&y.view()))
                .unwrap();
            assert!(g.column(0).iter().all(|&v| v == 0.0) && h.column(0).iter().all(|&v| v == 0.0), "{mode:?}");
            assert!(g.column(1).iter().chain(h.column(1).iter()).chain(g.column(2).iter()).chain(h.column(2).iter()).all(|v| v.is_finite()));
        }
    }

    /// Damped is the elementwise max of the other two modes on every row.
    #[test]
    fn damped_is_elementwise_max_of_observed_and_fisher() {
        let n = 5000; // exercises both the serial and (via slicing) parallel paths
        let preds = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => -0.5 + 0.0004 * (i % 23) as f64,
            1 => (i % 7) as f64 * 0.3,
            _ => -0.4 + 0.0003 * (i % 13) as f64,
        });
        let y = Array1::from_iter((0..n).map(|i| ((i * 7919) % 211) as f64 / 10.0 - 10.0));
        let target = ResponseData::Univariate(&y.view());
        let mk = |m| StudentT::default().with_hessian_mode(m);
        let tr = mk(HessianMode::Observed).transform_params(&preds.view());
        let (g_o, h_o) = mk(HessianMode::Observed).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
        let (g_f, h_f) = mk(HessianMode::Fisher).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
        let (g_d, h_d) = mk(HessianMode::Damped).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
        assert_eq!(g_o, g_d);
        assert_eq!(g_f, g_d);
        let mut negatives = 0;
        for i in 0..n {
            for j in 0..3 {
                assert_eq!(h_d[[i, j]], h_o[[i, j]].max(h_f[[i, j]]));
                if h_o[[i, j]] < 0.0 {
                    negatives += 1;
                }
            }
        }
        assert!(negatives > 0, "test data should include negative observed curvature");
        assert!(h_d.iter().all(|&h| h > 0.0));
    }

    /// A start value is never placed in the ExpDf dead zone: on data whose
    /// unconditional t-fit runs to the ν = 2 floor the df start is 2.5.
    #[test]
    fn start_df_is_floored_above_the_dead_zone() {
        // extremely heavy-tailed: Cauchy-like draws → unconditional ν̂ → floor
        use rand::SeedableRng;
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let c = RandT::new(1.0).unwrap();
        let y: Array1<f64> = Array1::from_iter((0..4000).map(|_| 0.3 + 0.01 * c.sample(&mut rng)));
        let dist = StudentT::new(Stabilization::None, ResponseFn::ExpDf, ResponseFn::Exp, LossFn::Nll, true);
        let (loss, sv) = dist.calculate_start_values(&ResponseData::Univariate(&y.view()), 50).unwrap();
        assert!(sv[0] >= MIN_START_DF_RAW - 1e-12, "df raw {} below the floor", sv[0]);
        assert!(loss.is_finite());
        let nu = ResponseFn::ExpDf.apply_scalar(sv[0]);
        assert!((2.4..=3.0).contains(&nu) || nu > 3.0, "ν₀ = {nu}");
    }

    /// The start-df cap clamps only the df coordinate and only from above.
    #[test]
    fn max_start_df_caps_the_fitted_df() {
        use rand::SeedableRng;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let nd = rand_distr::Normal::new(5.0, 2.0).unwrap();
        let y: Array1<f64> = Array1::from_iter((0..3000).map(|_| nd.sample(&mut rng)));
        let target = ResponseData::Univariate(&y.view());
        let free = StudentT::new(Stabilization::None, ResponseFn::ExpDf, ResponseFn::Exp, LossFn::Nll, true)
            .without_max_start_df();
        let capped = free.clone().with_max_start_df(100.0);
        let (_, sv_free) = free.calculate_start_values(&target, 50).unwrap();
        let (loss_c, sv_cap) = capped.calculate_start_values(&target, 50).unwrap();
        assert!(sv_free[0] > (98.0f64).ln(), "Gaussian data should fit a large ν: {}", sv_free[0]);
        assert!((sv_cap[0] - (98.0f64).ln()).abs() < 1e-12);
        assert_eq!(sv_cap[1], sv_free[1]);
        assert_eq!(sv_cap[2], sv_free[2]);
        assert!(loss_c.is_finite());
        // serde: a blob without the field gets the default cap; an explicit None round-trips
        let mut v: serde_json::Value = serde_json::to_value(&free).unwrap();
        v.as_object_mut().unwrap().remove("max_start_df");
        let back: StudentT = serde_json::from_value(v).unwrap();
        assert_eq!(back.max_start_df, Some(10.0));
        let back2: StudentT = serde_json::from_value(serde_json::to_value(&free).unwrap()).unwrap();
        assert_eq!(back2.max_start_df, None);
        assert_eq!(StudentT::default().max_start_df(), Some(10.0));
    }

    /// Default is Damped; a JSON blob without the field (an older saved
    /// model) also deserializes to Damped, and an explicit mode round-trips.
    #[test]
    fn hessian_mode_serde_default_is_damped() {
        let d = StudentT::default();
        assert_eq!(d.hessian_mode(), HessianMode::Damped);
        let mut v: serde_json::Value = serde_json::to_value(&d).unwrap();
        v.as_object_mut().unwrap().remove("hessian_mode");
        let back: StudentT = serde_json::from_value(v).unwrap();
        assert_eq!(back.hessian_mode(), HessianMode::Damped);
        let f: StudentT = serde_json::from_value(
            serde_json::to_value(d.with_hessian_mode(HessianMode::Fisher)).unwrap(),
        )
        .unwrap();
        assert_eq!(f.hessian_mode(), HessianMode::Fisher);
    }
}

#[cfg(test)]
mod param_gradient_scale_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;

    #[test]
    fn scales_multiply_gradient_columns_only() {
        let n = 300;
        let preds = Array2::from_shape_fn((n, 3), |(i, j)| 0.1 * ((i + j) % 7) as f64 - 0.2);
        let y = Array1::from_iter((0..n).map(|i| ((i * 37) % 101) as f64 / 20.0 - 2.5));
        let target = ResponseData::Univariate(&y.view());
        let plain = StudentT::default();
        let scaled = StudentT::default().with_param_gradient_scales([2.0, 1.0, 0.5]);
        let a = plain.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        let b = scaled.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        let sc = [2.0, 1.0, 0.5];
        for j in 0..3 {
            for i in 0..n {
                assert_eq!(b.gradients[[i, j]], a.gradients[[i, j]] * sc[j]);
                assert_eq!(b.hessians[[i, j]], a.hessians[[i, j]]);
            }
        }
        // serde: absent field → None; explicit value round-trips
        let mut v: serde_json::Value = serde_json::to_value(&plain).unwrap();
        v.as_object_mut().unwrap().remove("param_gradient_scales");
        let back: StudentT = serde_json::from_value(v).unwrap();
        assert!(back.param_gradient_scales().is_none());
        let back2: StudentT = serde_json::from_value(serde_json::to_value(&scaled).unwrap()).unwrap();
        assert_eq!(back2.param_gradient_scales(), Some(&sc[..]));
    }
}

#[cfg(test)]
mod beta_nll_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;

    /// β-NLL multiplies every column of row i by σ̂_i^{2β}; β = 0 is a no-op.
    #[test]
    fn beta_nll_weights_rows_by_scale_power() {
        let n = 200;
        let preds = Array2::from_shape_fn((n, 3), |(i, j)| if j == 2 { -0.5 + 0.01 * i as f64 } else { 0.1 * (j as f64) });
        let y = Array1::from_iter((0..n).map(|i| ((i * 53) % 97) as f64 / 25.0 - 2.0));
        let target = ResponseData::Univariate(&y.view());
        let plain = StudentT::default();
        let b = StudentT::default().with_beta_nll(0.5);
        let zero = StudentT::default().with_beta_nll(0.0);
        let tr = plain.transform_params(&preds.view());
        let a = plain.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        let w = b.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        let z = zero.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        assert_eq!(a.gradients, z.gradients);
        let mean_sigma = tr.column(2).mean().unwrap();
        for i in 0..n {
            let wi = tr[[i, 2]] / mean_sigma; // σ^{2·0.5} = σ, normalised to mean 1
            for j in 0..3 {
                assert!((w.gradients[[i, j]] - a.gradients[[i, j]] * wi).abs() <= 1e-12 * a.gradients[[i, j]].abs().max(1.0));
                assert!((w.hessians[[i, j]] - a.hessians[[i, j]] * wi).abs() <= 1e-12 * a.hessians[[i, j]].abs().max(1.0));
            }
        }
        let mut v: serde_json::Value = serde_json::to_value(&plain).unwrap();
        v.as_object_mut().unwrap().remove("beta_nll");
        let back: StudentT = serde_json::from_value(v).unwrap();
        assert_eq!(back.beta_nll(), None);
    }
}

/// Raw value whose response is `target` (closed form for Exp / Softplus,
/// bisection otherwise).
pub(crate) fn invert_response(f: &ResponseFn, target: f64) -> f64 {
    match f {
        ResponseFn::Exp => target.ln(),
        ResponseFn::Softplus => {
            if target > 30.0 { target } else { target.exp_m1().ln() }
        }
        _ => {
            let (mut lo, mut hi) = (-50.0, 50.0);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if f.apply_scalar(mid) < target { lo = mid } else { hi = mid }
            }
            0.5 * (lo + hi)
        }
    }
}

#[cfg(test)]
mod start_rule_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn median_start_is_the_default_and_moves_only_the_location() {
        let y = array![440.0, 455.0, 448.0, 470.0, 452.0];
        let target = ResponseData::Univariate(&y.view());
        let d = StudentT::default();
        assert_eq!(d.start_rule(), StudentTStart::Median);
        assert_eq!(d.default_start_values(&target).unwrap().to_vec(), vec![0.5, 452.0, 0.5]);
        let raw = d.clone().with_start_rule(StudentTStart::Raw);
        assert!(raw.default_start_values(&target).is_none());
        let mad = d.with_start_rule(StudentTStart::MedianMad);
        let sv = mad.default_start_values(&target).unwrap();
        // MAD = median(|y − 452|) = median(12, 3, 4, 18, 0) = 4
        assert!((sv[2] - (1.4826f64 * 4.0).ln()).abs() < 1e-9 && sv[1] == 452.0 && sv[0] == 0.5);
    }

    #[test]
    fn adaptive_start_inflates_the_scale_only_on_light_tails() {
        // uniform-ish labels: g2 ≈ −1.2 → scale 3·sd
        let y = ndarray::Array1::from_shape_fn(1000, |i| i as f64 / 1000.0);
        let t = ResponseData::Univariate(&y.view());
        let d = StudentT::default().with_start_rule(StudentTStart::Adaptive);
        let sv = d.default_start_values(&t).unwrap();
        let (sd, g2) = sd_and_excess_kurtosis(y.as_slice().unwrap());
        assert!(g2 < -1.0);
        assert!((sv[2] - (3.0 * sd).ln()).abs() < 1e-9);
        // one far outlier: g2 large → median start unchanged
        let mut y2 = y.clone();
        y2[0] = 1.0e3;
        let t2 = ResponseData::Univariate(&y2.view());
        assert!(sd_and_excess_kurtosis(y2.as_slice().unwrap()).1 > 1.0);
        assert_eq!(d.default_start_values(&t2).unwrap()[2], 0.5);
    }

    #[test]
    fn serde_default_is_the_raw_start() {
        let mut v = serde_json::to_value(StudentT::default()).unwrap();
        v.as_object_mut().unwrap().remove("start_rule");
        let back: StudentT = serde_json::from_value(v).unwrap();
        assert_eq!(back.start_rule(), StudentTStart::Raw);
    }
}

#[cfg(test)]
mod start_df_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn start_df_sets_the_raw_df_of_the_closed_form_starts() {
        let y = array![1.0, 2.0, 3.0];
        let t = ResponseData::Univariate(&y.view());
        let d = StudentT::default().with_start_df(10.0);
        let sv = d.default_start_values(&t).unwrap();
        assert!((ResponseFn::ExpDf.apply_scalar(sv[0]) - 10.0).abs() < 1e-5, "{sv:?}");
        assert_eq!(sv[1], 2.0);
        // Raw ignores it (no closed-form start at all)
        assert!(d.with_start_rule(StudentTStart::Raw).default_start_values(&t).is_none());
    }
}
