//! NegativeBinomial distribution implementation.

use super::base::{Distribution, DistributionParam, HessianMode, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Gamma as RandGamma, Poisson as RandPoisson};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::constants::trigamma;

/// For the `total_count` (r) gradient/hessian the NB NLL needs
/// `digamma(r+k) - digamma(r)` and `trigamma(r+k) - trigamma(r)`. For integer k
/// these collapse to finite rational sums, ~2x cheaper than two special-function
/// evaluations each:
///   digamma(r+k)  - digamma(r)  =  Σ_{j=0}^{k-1} 1/(r+j)
///   trigamma(r+k) - trigamma(r) = -Σ_{j=0}^{k-1} 1/(r+j)²
///
/// Returns the pair actually used below — `(-digamma(r+k)+digamma(r),
/// -trigamma(r+k)+trigamma(r))` = `(-Σ1/(r+j), Σ1/(r+j)²)` — so `hess_r` is
/// nonnegative by construction. Falls back to `None` for non-integer or large k,
/// where the loop would lose its edge over the asymptotic special functions.
///
/// The k cutoff is measured, not guessed: on Apple Silicon the digamma+trigamma
/// pair costs ~16 ns while the division-bound loop costs ~0.4 ns/iteration, so
/// the loop wins below k ≈ 40 (k=32: 0.73x, k=64: 1.67x of the special-function
/// cost) — both branches are exact, so the cutoff is purely a speed knob.
///
/// Shared with the Zero-Inflated NB (`ZINB`), whose positive-count branch has the
/// same NB `total_count` gradient/hessian.
#[inline]
pub(crate) fn nb_psi_diff(r: f64, k: f64) -> Option<(f64, f64)> {
    if k >= 0.0 && k <= 40.0 && k == k.trunc() {
        let kk = k as u32;
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for j in 0..kk {
            let inv = 1.0 / (r + j as f64);
            s1 += inv;
            s2 += inv * inv;
        }
        Some((-s1, s2))
    } else {
        None
    }
}

/// `ln_gamma(r+k) - ln_gamma(r)` for the same integer-k band as `nb_psi_diff`:
/// Γ(r+k)/Γ(r) = Π_{j=0}^{k-1} (r+j), so the lgamma difference collapses to k
/// multiplies and a single `ln` instead of two lgamma evaluations. The `r`
/// bound keeps the running product below f64 overflow: it is at most
/// (r+40)^40, which stays finite for r < 1e7 — response functions never push
/// `total_count` near that in practice, and the fallback is exact anyway.
///
/// Shared with `ZINB` (same NB positive-count log-pmf).
#[inline]
pub(crate) fn nb_lgamma_ratio(r: f64, k: f64) -> Option<f64> {
    if k >= 0.0 && k <= 40.0 && k == k.trunc() && r < 1e7 {
        let kk = k as u32;
        let mut prod = 1.0;
        for j in 0..kk {
            prod *= r + j as f64;
        }
        Some(prod.ln())
    } else {
        None
    }
}

/// Which coordinates the two boosters descend in (2026-10-31).
///
/// `TotalCountProbs` is the torch convention this head always had: tree 0 is
/// `total_count` (r), tree 1 is `probs`, and the mean r·p/(1−p) depends on
/// BOTH — so every r step moves the mean and the p tree then corrects it; the
/// Fisher information is not diagonal in (r, p). On NBA points the head
/// settled at θ ≈ 2.0 against ~4.5 and sat 4 % CRPS behind the served
/// Poisson-plus-width tree.
///
/// `MeanDispersion` descends in (log µ, log θ) — the NB2 coordinates in which
/// the Fisher information IS diagonal, so the two trees do not fight: tree 0
/// moves the mean with θ fixed, tree 1 moves the width with µ fixed. The
/// Hessians are the expected Fisher information for log µ (µθ/(µ+θ)) and the
/// observed information for log θ floored at `NB_THETA_HESS_FLOOR` (the usual
/// LSS stabilisation — the observed form can go negative). Start values are
/// the closed-form moments (no L-BFGS). A plain xgboost twin of this fit beat
/// the served Poisson + width tree on CRPS and NLL 6/6 on the NBA points folds
/// (`python_tests/nba_lit_width_experiments.py`, EXP=nbsched). The transformed
/// / predicted parameters are STILL reported as `(total_count, probs)` — every
/// consumer of the (n, p) convention is untouched; only the boosting
/// coordinates differ. Old model files deserialize to `TotalCountProbs`.
///
/// The CRPS loss uses these coordinates too since 2026-11-14 (research
/// proposals 029, 031): its gradient is the chain rule through
/// (r, probs) = (θ, µ/(µ+θ)) and its Hessian the expected CRPS curvature
/// 2 Σ_k (∂F(k)/∂η)² for every row. Against (total_count, probs) the test
/// CRPS was 2 % to 25 % lower on five datasets (25 of 25 fits).
///
/// Default since 2026-11-10 for `NegativeBinomial::new` / `default()` with the
/// NLL loss (measured: −0.07 nats on µ ∈ [1.5, 60], −0.86 on µ ∈ [100, 4000]
/// against `TotalCountProbs`, oracle-known Gamma–Poisson counts), and since
/// 2026-11-14 with the CRPS loss. Opt out with
/// `.with_parameterization(NbParameterization::TotalCountProbs)`, which
/// restores the response functions passed to `new`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NbParameterization {
    TotalCountProbs,
    #[default]
    MeanDispersion,
}

impl NbParameterization {
    /// The serde default for files written before the field existed.
    fn legacy() -> Self {
        NbParameterization::TotalCountProbs
    }
}

/// Floor on the log-θ observed information in `MeanDispersion` mode.
pub const NB_THETA_HESS_FLOOR: f64 = 1e-2;
/// Floor on the log-µ Fisher information in `MeanDispersion` mode.
pub const NB_MU_HESS_FLOOR: f64 = 1e-6;
/// Raw-coordinate clamps of `MeanDispersion`: log µ ∈ [−10, 15] (µ up to
/// 3.3e6), log θ ∈ [−6, 12] (θ up to 1.6e5). Widened 2026-11-09 from
/// [−10, 6] / [−6, 8]: the old µ ≤ e⁶ ≈ 403 cap silently truncated every count
/// target with a conditional mean above 403 (measured: synth µ ∈ [100, 4000],
/// MeanDispersion RMSE 1432 vs 656 for the default head). Outside the clamp
/// a coordinate's gradient is zeroed — the margin no longer affects the fit,
/// so pushing it further is meaningless.
pub const NB_LOG_MU_CLAMP: (f64, f64) = (-10.0, 15.0);
pub const NB_LOG_THETA_CLAMP: (f64, f64) = (-6.0, 12.0);

/// θ start value clip (method of moments can explode on under-dispersed data).
pub const NB_THETA_START_CLIP: (f64, f64) = (0.2, 50.0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegativeBinomial {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
    #[serde(default = "NbParameterization::legacy")]
    parameterization: NbParameterization,
    /// log-θ curvature of the `MeanDispersion` head (log µ always uses its
    /// closed-form Fisher information). `Observed` = the observed information
    /// floored at `NB_THETA_HESS_FLOOR` (the behaviour before this field
    /// existed, kept for files saved without it); constructors default to
    /// `Damped` (2026-11-10: better or neutral on three synthetic NB DGPs and
    /// two real count targets — see `nb_log_theta_fisher`).
    #[serde(default = "NegativeBinomial::legacy_hessian_mode")]
    hessian_mode: HessianMode,
    /// MeanDispersion without `initialize`: start log µ at ln ȳ (log θ stays
    /// at the raw 0.5). Default `true` since 2026-11-10: from the raw 0.5
    /// start (µ ≈ 1.65) the fit was far worse on counts with a large mean —
    /// bike_cnt −0.18, bike_casual −0.08, synthetic µ 100–4000 −1.45 nats with
    /// the location start, which ties the fitted `initialize` start and fits
    /// ~2× faster. Old files load with `false`; only affects training.
    #[serde(default)]
    location_start: bool,
}

/// Expected (Fisher) information of log θ under NB2(µ, θ):
///   I = θ² E[ψ′(θ) − ψ′(Y + θ)] − θµ/(µ + θ)
/// (no closed form; limits θ²ψ′(θ) − θ as µ → ∞ and µ²/(2(µ + θ)²) as
/// θ/µ → ∞). With ψ′(θ) − ψ′(θ + y) = ∫ t e^{−θt}(1 − e^{−yt})/(1 − e^{−t}) dt,
/// the NB probability generating function E[e^{−Yt}] = (θ/(θ + µ(1 − e^{−t})))^θ
/// and θµ/(µ+θ) = θ² ∫ e^{−θt}(1 − e^{−µt}) dt, this is the µ-independent-cost
/// integral (u = θt)
///   I = θ ∫₀^∞ e^{−u} [ t/(1 − e^{−t}) · (1 − PGF(t)) − (1 − e^{−µt}) ] du,  t = u/θ,
/// evaluated with a 25-node exp-sinh rule (h = 1/4, |x| ≤ 3). Accuracy over
/// log µ ∈ [−10, 15], log θ ∈ [−6, 12] against a 640-node reference: 99th
/// percentile relative error 3e-3, worst 4 % only where I < 1e-15.
pub fn nb_log_theta_fisher(mu: f64, theta: f64) -> f64 {
    use std::sync::OnceLock;
    static NODES: OnceLock<[(f64, f64); 25]> = OnceLock::new();
    if !(mu > 0.0 && theta > 0.0 && mu.is_finite() && theta.is_finite()) {
        return 0.0;
    }
    let nodes = NODES.get_or_init(|| {
        let h = 0.25f64;
        let mut out = [(0.0, 0.0); 25];
        for (k, o) in out.iter_mut().enumerate() {
            let x = -3.0 + h * k as f64;
            let u = (std::f64::consts::FRAC_PI_2 * x.sinh()).exp();
            *o = (u, u * std::f64::consts::FRAC_PI_2 * x.cosh() * h);
        }
        out
    });
    let mut acc = 0.0;
    for &(u, w) in nodes.iter() {
        let t = u / theta;
        let om = -(-t).exp_m1(); // 1 − e^{−t}
        let one_m_pgf = -(-theta * (mu * om / theta).ln_1p()).exp_m1();
        let f = (-u).exp() * (t / om * one_m_pgf + (-mu * t).exp_m1());
        acc += w * f;
    }
    (theta * acc).max(0.0)
}

impl NegativeBinomial {
    /// `MeanDispersion` coordinates (the default, see [`NbParameterization`])
    /// for both losses; the response functions are kept and used only by the
    /// `TotalCountProbs` head (after
    /// `.with_parameterization(NbParameterization::TotalCountProbs)`).
    pub fn new(
        stabilization: Stabilization,
        total_count_response_fn: ResponseFn,
        probs_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("total_count", total_count_response_fn),
            DistributionParam::new("probs", probs_response_fn),
        ];
        Self {
            params,
            stabilization,
            loss_fn,
            initialize,
            hessian_mode: Self::default_hessian_mode(),
            parameterization: NbParameterization::default(),
            location_start: true,
        }
    }

    /// Whether training fits the start values (`initialize`).
    pub fn with_initialize(mut self, on: bool) -> Self {
        self.initialize = on;
        self
    }

    /// See the `location_start` field.
    pub fn with_location_start(mut self, on: bool) -> Self {
        self.location_start = on;
        self
    }

    pub fn location_start(&self) -> bool {
        self.location_start
    }

    /// Choose the boosting coordinates, for either loss. `MeanDispersion`
    /// never reads the per-parameter response functions, so switching back to
    /// `TotalCountProbs` restores the ones passed to `new`.
    pub fn with_parameterization(mut self, parameterization: NbParameterization) -> Self {
        self.parameterization = parameterization;
        self
    }

    /// The `MeanDispersion` head (see [`NbParameterization`]): boosting in
    /// (log µ, log θ) with `Exp` responses, moments start values, outputs
    /// reported as (total_count, probs). Both losses (since 2026-11-14 the
    /// CRPS too: research proposals 029, 031).
    pub fn new_mean_dispersion(stabilization: Stabilization, loss_fn: LossFn) -> Self {
        let params = vec![
            // The NAMES stay the output convention: column 0 is reported as
            // total_count (= θ), column 1 as probs (= µ/(µ+θ)); the response
            // functions here are the per-coordinate maps raw → (µ, θ) that
            // `transform_params` composes with the joint reparameterization.
            DistributionParam::new("total_count", ResponseFn::Exp),
            DistributionParam::new("probs", ResponseFn::Exp),
        ];
        Self {
            params,
            stabilization,
            loss_fn,
            initialize: true,
            parameterization: NbParameterization::MeanDispersion,
            hessian_mode: Self::default_hessian_mode(),
            location_start: true,
        }
    }

    fn default_hessian_mode() -> HessianMode {
        HessianMode::Damped
    }

    fn legacy_hessian_mode() -> HessianMode {
        HessianMode::Observed
    }

    /// log-θ curvature of the `MeanDispersion` head (no effect on the
    /// `TotalCountProbs` head). `Fisher` / `Damped` evaluate
    /// [`nb_log_theta_fisher`] (25-node quadrature) per row and drop the 0.01 floor (to 1e-8).
    pub fn with_hessian_mode(mut self, mode: HessianMode) -> Self {
        self.hessian_mode = mode;
        self
    }

    pub fn hessian_mode(&self) -> HessianMode {
        self.hessian_mode
    }

    pub fn parameterization(&self) -> NbParameterization {
        self.parameterization
    }

    /// `MeanDispersion`: raw (f, g) = (log µ, log θ) → (r, probs) = (θ, µ/(µ+θ)).
    fn mean_dispersion_to_r_probs(f: f64, g: f64) -> (f64, f64) {
        let mu = f.clamp(NB_LOG_MU_CLAMP.0, NB_LOG_MU_CLAMP.1).exp();
        let theta = g.clamp(NB_LOG_THETA_CLAMP.0, NB_LOG_THETA_CLAMP.1).exp();
        (theta, (mu / (mu + theta)).clamp(1e-6, 1.0 - 1e-6))
    }

    pub fn default() -> Self {
        Self::new(
            Stabilization::None,
            ResponseFn::Relu,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        )
    }

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let r = params[0]; // total_count
        let probs = params[1];

        if r <= 0.0 || probs <= 0.0 || probs >= 1.0 {
            return f64::NEG_INFINITY;
        }

        // Use the target as-is rather than truncating to u64. PyTorch's NB
        // log_prob is lgamma-based and well-defined for non-integer counts, and
        // the analytical-gradient path already uses the raw target — truncating
        // here made the early-stopping NLL disagree with the training gradient.
        // Negative or non-finite counts are outside the support → -inf.
        let k = target;
        if !k.is_finite() || k < 0.0 {
            return f64::NEG_INFINITY;
        }
        // PyTorch NB uses probs = P(success/counted event), mean = r*p/(1-p)
        // statrs NB uses p = P(success/stopping event), mean = r*(1-p)/p
        // They are complementary: statrs_p = 1 - probs
        let p = 1.0 - probs;
        // ln_pmf = ln_gamma(r+k) - ln_gamma(r) - ln_gamma(k+1) + r*ln(p) + k*ln(1-p)
        nb_lgamma_ratio(r, k).unwrap_or_else(|| ln_gamma(r + k) - ln_gamma(r))
            - crate::constants::ln_factorial(k).unwrap_or_else(|| ln_gamma(k + 1.0))
            + r * p.ln()
            + k * (-p).ln_1p()
    }
}

/// First index of a CRPS table for NB(r, probs): 0, or `mean - 10 sd - 10`
/// when pmf(0) = (1 - probs)^r would underflow.
#[inline]
fn nb_window_start(r: f64, ln_1mp: f64, mean: f64, sd: f64) -> usize {
    if r * -ln_1mp < 600.0 { 0 } else { (mean - 10.0 * sd - 10.0).floor().max(0.0) as usize }
}

/// pmf(k) in the convention of the CRPS recurrences: pmf(0) = (1 - probs)^r,
/// pmf(k) / pmf(k-1) = probs (r + k - 1) / k.
#[inline]
fn nb_pmf_at(r: f64, probs: f64, ln_1mp: f64, k: usize) -> f64 {
    if k == 0 {
        (r * ln_1mp).exp()
    } else {
        let kf = k as f64;
        (ln_gamma(r + kf) - ln_gamma(r) - ln_gamma(kf + 1.0) + r * ln_1mp + kf * probs.ln()).exp()
    }
}

/// The CDF-sum CRPS over `[lo, hi]` for rows whose sum needs more than the
/// 150 terms of the original loop. Below the table F = 0 and the term is
/// 1{k >= y}; above it F = 1 and the term is 1{k < y}.
fn nb_crps_windowed(r: f64, probs: f64, mean: f64, sd: f64, y_int: usize, max_k: usize) -> f64 {
    let ln_1mp = (-probs).ln_1p();
    let lo = nb_window_start(r, ln_1mp, mean, sd);
    let mut hi = max_k.max(lo);
    if hi - lo + 1 > super::util::CRPS_MAX_TERMS {
        super::util::warn_crps_truncated("NegativeBinomial", mean);
        hi = lo + super::util::CRPS_MAX_TERMS - 1;
    }
    let mut pmf = nb_pmf_at(r, probs, ln_1mp, lo);
    let mut cdf = 0.0;
    let mut crps = lo.saturating_sub(y_int) as f64;
    for k in lo..=hi {
        if k > lo {
            pmf *= probs * (r + k as f64 - 1.0) / k as f64;
        }
        cdf += pmf;
        let d = cdf.min(1.0) - if k >= y_int { 1.0 } else { 0.0 };
        crps += d * d;
        if k >= y_int && 1.0 - cdf < 1e-9 {
            break;
        }
    }
    crps + y_int.saturating_sub(hi + 1) as f64
}

/// Scratch for one row of the NB CRPS closed-form pass — pmf/cdf tables and
/// the per-parameter dF/dtheta prefix tables, reused across rows.
#[derive(Default)]
struct NbCrpsScratch {
    pmf: Vec<f64>,
    cdf: Vec<f64>,
    /// F_r[k]     = sum_{i<=k} dpmf(i)/dr        (prefix of dpmf/dr)
    fr: Vec<f64>,
    /// F_p[k]     = sum_{i<=k} dpmf(i)/dprobs    (prefix of dpmf/dprobs)
    fp_: Vec<f64>,
    /// suffix sums of F_r / F_p (S_j(y) = sum_{k>=y} dF(k)/dtheta_j)
    suf_r: Vec<f64>,
    suf_p: Vec<f64>,
}

impl NegativeBinomial {
    /// Closed-form CRPS gradients AND metric hessians for the two-parameter
    /// NB, replacing the FD path (which trained with unit hessians — measured
    /// on real strikeout data: tuned NB-CRPS test CRPS 1.293 vs 1.173 for the
    /// NLL twin, the same deficiency natural gradients fixed for Poisson).
    ///
    /// # The identities
    ///
    /// With CRPS(theta) = sum_k (F(k) - 1{k>=y})^2 and Ftheta(k) = dF(k)/dtheta:
    ///
    /// ```text
    ///  dCRPS/dtheta = 2 [ sum_k F(k) Ftheta(k)  -  sum_{k>=y} Ftheta(k) ]
    ///               = 2 [ A - S(y) ]
    /// ```
    ///
    /// so ONE set of prefix/suffix tables serves every y — both the observed
    /// gradient (y = y_obs) and the metric expectation over hypothetical y.
    /// The per-parameter pmf derivatives need no special functions:
    ///
    /// ```text
    ///  dpmf(i)/dprobs = pmf(i) * (i/probs - r/(1-probs))
    ///  dpmf(i)/dr     = pmf(i) * (H_i + ln(1-probs)),
    ///      H_i = psi(r+i) - psi(r) = sum_{t<i} 1/(r+t)   (telescoped —
    ///      the digamma difference is a plain running sum).
    /// ```
    ///
    /// # The hessian
    ///
    /// NGBoost's CRPS metric, diagonal per parameter in MARGIN space:
    ///
    /// ```text
    ///  M_jj = E_{y~NB(theta)} [ ( dCRPS/dtheta_j * dtheta_j/dm_j )^2 ]
    /// ```
    ///
    /// positive by construction (a Riemannian metric, not a curvature), same
    /// floor rationale as the Poisson port: XGBoost leaf values are ~ g/h
    /// with no line search, and the metric can vanish faster than an observed
    /// tail gradient at degenerate parameter corners.
    fn analytical_crps_gradients(
        &self,
        predictions: &ArrayView2<f64>,
        transformed: &ArrayView2<f64>,
        target: &ResponseData,
    ) -> Option<(Array2<f64>, Array2<f64>)> {
        self.analytical_crps_gradients_impl(predictions, transformed, target, false)
    }

    /// [`analytical_crps_gradients`] with the row-parallel fill forced off —
    /// the reference the parity test compares against.
    fn analytical_crps_gradients_impl(
        &self,
        predictions: &ArrayView2<f64>,
        transformed: &ArrayView2<f64>,
        target: &ResponseData,
        force_serial: bool,
    ) -> Option<(Array2<f64>, Array2<f64>)> {
        let y = match target {
            ResponseData::Univariate(arr) => arr,
            ResponseData::Multivariate(_) => return None,
        };
        let n_samples = predictions.nrows();
        let r_response_fn = &self.params[0].response_fn;
        let probs_response_fn = &self.params[1].response_fn;
        let t_r = transformed.column(0);
        let t_probs = transformed.column(1);
        let p_r = predictions.column(0);
        let p_probs = predictions.column(1);
        // MeanDispersion: the raw scores are (log µ, log θ) and `transformed` comes from the
        // joint map, so the per-parameter response derivatives are not used.
        let md = self.parameterization == NbParameterization::MeanDispersion;
        let (rd_r, rd_p) = if md {
            (ndarray::Array1::ones(n_samples), ndarray::Array1::ones(n_samples))
        } else {
            (
                r_response_fn.derivative_batches_from_transformed(&p_r, &t_r).0,
                probs_response_fn.derivative_batches_from_transformed(&p_probs, &t_probs).0,
            )
        };

        const CRPS_MIN_METRIC: f64 = 0.05;
        const LEGACY_CAP: usize = 220;
        let unit_hess = std::env::var("GRADIENTLSS_CRPS_HESS")
            .map(|v| v.trim().eq_ignore_ascii_case("unit"))
            .unwrap_or(false);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::ones((n_samples, 2));
        // One row: writes its (r, probs) gradient / hessian into the 2-wide
        // chunks it owns, with its own scratch tables. Rows are independent,
        // so the parallel driver below is bit-identical to the serial one
        // (pinned by `crps_gradients_parallel_matches_serial`). Serial until
        // 2026-09-03 — ~4 ms per boosting round at 18.6k rows on one core.
        let row = |i: usize, sc: &mut NbCrpsScratch, gc: &mut [f64], hc: &mut [f64]| {
            let r = t_r[i];
            let probs = t_probs[i];
            let yi = y[i];
            if !(r > 0.0) || !(probs > 0.0) || !(probs < 1.0) || !yi.is_finite() || yi < 0.0 {
                // Out of support: FD saw the flat 1e6 penalty — zero gradient,
                // unit hessian.
                return;
            }
            let mean = r * probs / (1.0 - probs);
            let var = mean / (1.0 - probs);
            let sd = var.sqrt();
            let y_int = yi.round().max(0.0) as usize;
            // Table long enough for BOTH the observed-y gradient and the
            // metric expectation over hypothetical y (mean + 8 sigma). Until
            // 2026-11-13 both were cut at 220 terms, which left a row with a
            // mean above ~100 a gradient from a CDF that had not reached 1
            // (research proposal 023). A row that fits under the old cap
            // builds exactly the table it always did.
            let mut y_hyp_max = ((mean + 8.0 * sd).ceil() as usize).max(5);
            let mut k_total = ((yi.max(mean) + 6.0 * sd).ceil() as usize).max(y_hyp_max + 10);
            // a row the original 220-entry table could hold
            let legacy_row = y_hyp_max <= LEGACY_CAP - 11 && k_total <= LEGACY_CAP;
            let ln_1mp = (-probs).ln_1p(); // ln(1 - probs)
            // The table starts at 0 unless pmf(0) = (1 - probs)^r would underflow
            // (large r with a large mean); then at mean - 10 sd, below which the
            // pmf and its parameter derivatives vanish.
            let lo = nb_window_start(r, ln_1mp, mean, sd);
            if k_total - lo + 1 > super::util::CRPS_MAX_TERMS {
                super::util::warn_crps_truncated("NegativeBinomial", mean);
                k_total = lo + super::util::CRPS_MAX_TERMS - 1;
                y_hyp_max = y_hyp_max.min(k_total - 10);
            }
            let len = k_total - lo + 1;

            // ---- one pass: pmf, cdf, and the two dF/dtheta prefixes -------
            sc.pmf.clear();
            sc.cdf.clear();
            sc.fr.clear();
            sc.fp_.clear();
            let mut pmf = nb_pmf_at(r, probs, ln_1mp, lo);
            let mut cdf = 0.0;
            // psi(r+i) - psi(r), telescoped
            let mut h_i = if lo == 0 { 0.0 } else { digamma(r + lo as f64) - digamma(r) };
            let mut acc_r = 0.0;
            let mut acc_p = 0.0;
            for k in lo..=k_total {
                if k > lo {
                    pmf *= probs * (r + k as f64 - 1.0) / k as f64;
                    h_i += 1.0 / (r + k as f64 - 1.0);
                }
                cdf += pmf;
                acc_r += pmf * (h_i + ln_1mp);
                acc_p += pmf * (k as f64 / probs - r / (1.0 - probs));
                sc.pmf.push(pmf);
                sc.cdf.push(cdf.min(1.0));
                sc.fr.push(acc_r);
                sc.fp_.push(acc_p);
            }

            // A_j = sum_k F(k) * Ftheta_j(k); suffix sums S_j
            sc.suf_r.clear();
            sc.suf_r.resize(len + 1, 0.0);
            sc.suf_p.clear();
            sc.suf_p.resize(len + 1, 0.0);
            let mut a_r = 0.0;
            let mut a_p = 0.0;
            for k in (0..len).rev() {
                sc.suf_r[k] = sc.suf_r[k + 1] + sc.fr[k];
                sc.suf_p[k] = sc.suf_p[k + 1] + sc.fp_[k];
            }
            for k in 0..len {
                a_r += sc.cdf[k] * sc.fr[k];
                a_p += sc.cdf[k] * sc.fp_[k];
            }

            // ---- observed gradient: g_j = 2 (A_j - S_j(y_obs)) ------------
            let yo = y_int.clamp(lo, k_total) - lo;
            let g_r = 2.0 * (a_r - sc.suf_r[yo]);
            let g_p = 2.0 * (a_p - sc.suf_p[yo]);

            // ---- (log µ, log θ) coordinates (the default; proposal 029) ----
            // r = θ, probs = µ/(µ+θ):  ∂probs/∂f = c, ∂probs/∂g = −c, ∂r/∂g = r,
            // with c = probs (1 − probs). So ∂F/∂f = c F_p and ∂F/∂g = r F_r − c F_p,
            // and the same for the gradient. Curvature: expected, for every row.
            if md {
                let c = probs * (1.0 - probs);
                let inside = |v: f64, (lo, hi): (f64, f64)| v.is_finite() && v > lo && v < hi;
                // a clamped coordinate has no gradient there (as in the NLL head)
                gc[0] = if inside(p_r[i], NB_LOG_MU_CLAMP) { c * g_p } else { 0.0 };
                gc[1] = if inside(p_probs[i], NB_LOG_THETA_CLAMP) { r * g_r - c * g_p } else { 0.0 };
                if !unit_hess {
                    let mut e_f = 0.0;
                    let mut e_g = 0.0;
                    for k in 0..len {
                        let df = c * sc.fp_[k];
                        let dg = r * sc.fr[k] - df;
                        e_f += df * df;
                        e_g += dg * dg;
                    }
                    hc[0] = (2.0 * e_f).max(CRPS_MIN_METRIC);
                    hc[1] = (2.0 * e_g).max(CRPS_MIN_METRIC);
                }
                return;
            }
            gc[0] = g_r * rd_r[i];
            gc[1] = g_p * rd_p[i];

            // ---- large counts: expected CRPS curvature ---------------------
            // H_jj = d2/dtheta_j'2 E_y[CRPS(theta', y)] at theta' = theta
            //      = 2 sum_k (dF(k)/dtheta_j)^2,
            // the CRPS counterpart of the Fisher information. The metric below
            // is in units of CRPS^2 and outgrows the curvature as the counts
            // grow (see the Poisson counterpart), which shrinks every step.
            if !unit_hess && !legacy_row {
                let mut e_r = 0.0;
                let mut e_p = 0.0;
                for k in 0..len {
                    e_r += sc.fr[k] * sc.fr[k];
                    e_p += sc.fp_[k] * sc.fp_[k];
                }
                let dr = rd_r[i];
                let dp = rd_p[i];
                hc[0] = (2.0 * e_r * dr * dr).max(CRPS_MIN_METRIC);
                hc[1] = (2.0 * e_p * dp * dp).max(CRPS_MIN_METRIC);
                return;
            }

            // ---- metric hessians: M_jj = E_y[(g_j(y) * dtheta/dm)^2] ------
            if !unit_hess {
                let mut m_r = 0.0;
                let mut m_p = 0.0;
                for yh in 0..=(y_hyp_max - lo) {
                    let py = sc.pmf[yh];
                    if py < 1e-300 {
                        continue;
                    }
                    let gr = 2.0 * (a_r - sc.suf_r[yh]);
                    let gp = 2.0 * (a_p - sc.suf_p[yh]);
                    m_r += gr * gr * py;
                    m_p += gp * gp * py;
                }
                let dr = rd_r[i];
                let dp = rd_p[i];
                hc[0] = (m_r * dr * dr).max(CRPS_MIN_METRIC);
                hc[1] = (m_p * dp * dp).max(CRPS_MIN_METRIC);
            }
        };
        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let use_parallel = !force_serial && n_samples >= 256;
        if use_parallel {
            g_slice
                .par_chunks_mut(2)
                .zip(h_slice.par_chunks_mut(2))
                .enumerate()
                .for_each_init(NbCrpsScratch::default, |sc, (i, (gc, hc))| row(i, sc, gc, hc));
        } else {
            let mut sc = NbCrpsScratch::default();
            for (i, (gc, hc)) in g_slice.chunks_mut(2).zip(h_slice.chunks_mut(2)).enumerate() {
                row(i, &mut sc, gc, hc);
            }
        }
        Some((gradients, hessians))
    }
}

impl NegativeBinomial {
    /// The legacy L-BFGS start values (the trait default, kept callable from
    /// the override so the two parameterizations can share one method).
    fn calculate_start_values_lbfgs(
        &self,
        target: &ResponseData,
        max_iter: usize,
    ) -> crate::error::Result<(f64, ndarray::Array1<f64>)> {
        // Re-dispatch through a legacy-mode clone so the trait default sees
        // per-parameter response functions it can apply.
        let mut legacy = self.clone();
        legacy.parameterization = NbParameterization::TotalCountProbs;
        Distribution::calculate_start_values_default(&legacy, target, max_iter)
    }

    /// `MeanDispersion` NLL gradients / Hessians in the raw coordinates
    /// (f, g) = (log µ, log θ), NB2 log-likelihood
    /// ℓ = lnΓ(y+θ) − lnΓ(θ) − lnΓ(y+1) + θ ln θ + y ln µ − (y+θ) ln(µ+θ):
    ///   ∂ℓ/∂f = (y − µ) θ / (µ + θ)                 Fisher E[−∂²ℓ/∂f²] = µθ/(µ+θ)
    ///   ∂ℓ/∂g = θ·a,  a = ψ(y+θ) − ψ(θ) + ln θ + 1 − ln(µ+θ) − (y+θ)/(µ+θ)
    ///   ∂²ℓ/∂g² = θ·a + θ²·(ψ′(y+θ) − ψ′(θ) + 1/θ − 1/(µ+θ) − (µ−y)/(µ+θ)²)
    /// Returned as the NLL (negated) gradient and the positive Hessians.
    fn mean_dispersion_gradients(
        &self,
        predictions: &ArrayView2<f64>,
        y: &ndarray::ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let n_samples = predictions.nrows();
        let f_col = predictions.column(0);
        let g_col = predictions.column(1);
        let mode = self.hessian_mode;
        let compute = |i: usize| -> (f64, f64, f64, f64) {
            let k = y[i];
            if !k.is_finite() || k < 0.0 {
                return (0.0, NB_MU_HESS_FLOOR, 0.0, NB_THETA_HESS_FLOOR);
            }
            let f = if f_col[i].is_finite() { f_col[i] } else { 0.0 };
            let g = if g_col[i].is_finite() { g_col[i] } else { 0.0 };
            let f_c = f.clamp(NB_LOG_MU_CLAMP.0, NB_LOG_MU_CLAMP.1);
            let g_c = g.clamp(NB_LOG_THETA_CLAMP.0, NB_LOG_THETA_CLAMP.1);
            let mu = f_c.exp();
            let th = g_c.exp();
            let s = mu + th;
            // log mu
            let grad_f = (mu - k) * th / s;
            let hess_f = (mu * th / s).max(NB_MU_HESS_FLOOR);
            // log theta: psi differences (exact rational sums for integer k)
            let (neg_dpsi, pos_dtri) = nb_psi_diff(th, k)
                .unwrap_or_else(|| (-digamma(th + k) + digamma(th), -trigamma(th + k) + trigamma(th)));
            let dpsi = -neg_dpsi; // ψ(y+θ) − ψ(θ)
            let dtri = -pos_dtri; // ψ′(y+θ) − ψ′(θ)
            let a = dpsi + th.ln() + 1.0 - s.ln() - (k + th) / s;
            let dl_dg = th * a;
            let d2 = th * a + th * th * (dtri + 1.0 / th - 1.0 / s - (mu - k) / (s * s));
            let grad_g = -dl_dg;
            let hess_g = match mode {
                HessianMode::Observed => (-d2).max(NB_THETA_HESS_FLOOR),
                HessianMode::Fisher => nb_log_theta_fisher(mu, th).max(1e-8),
                HessianMode::Damped => (-d2).max(nb_log_theta_fisher(mu, th)).max(1e-8),
            };
            // a clamped coordinate has no gradient there
            let grad_f = if f_c != f { 0.0 } else { grad_f };
            let grad_g = if g_c != g { 0.0 } else { grad_g };
            (grad_f, hess_f, grad_g, hess_g)
        };
        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));
        let g_slice = gradients.as_slice_mut().expect("contiguous");
        let h_slice = hessians.as_slice_mut().expect("contiguous");
        const PAR_MIN: usize = 256;
        if n_samples >= PAR_MIN {
            g_slice.par_chunks_mut(2).zip(h_slice.par_chunks_mut(2)).enumerate().for_each(|(i, (gc, hc))| {
                let (g0, h0, g1, h1) = compute(i);
                gc[0] = g0; gc[1] = g1; hc[0] = h0; hc[1] = h1;
            });
        } else {
            g_slice.chunks_mut(2).zip(h_slice.chunks_mut(2)).enumerate().for_each(|(i, (gc, hc))| {
                let (g0, h0, g1, h1) = compute(i);
                gc[0] = g0; gc[1] = g1; hc[0] = h0; hc[1] = h1;
            });
        }
        (gradients, hessians)
    }
}

#[typetag::serde]
impl Distribution for NegativeBinomial {
    fn transform_params(&self, predictions: &ArrayView2<f64>) -> Array2<f64> {
        if self.parameterization == NbParameterization::TotalCountProbs {
            let mut result = Array2::zeros(predictions.dim());
            for (i, param) in self.params.iter().enumerate() {
                let col = predictions.column(i);
                let mut out_col = result.column_mut(i);
                param.response_fn.apply_into(&col, &mut out_col);
            }
            return result;
        }
        // MeanDispersion: a JOINT map — probs needs both raw scores.
        let n = predictions.nrows();
        let mut result = Array2::zeros((n, 2));
        for i in 0..n {
            let f = predictions[[i, 0]];
            let g = predictions[[i, 1]];
            let (f, g) = (if f.is_finite() { f } else { 0.0 }, if g.is_finite() { g } else { 0.0 });
            let (r, probs) = Self::mean_dispersion_to_r_probs(f, g);
            result[[i, 0]] = r;
            result[[i, 1]] = probs;
        }
        result
    }

    fn calculate_start_values(
        &self,
        target: &ResponseData,
        max_iter: usize,
    ) -> crate::error::Result<(f64, ndarray::Array1<f64>)> {
        if self.parameterization == NbParameterization::TotalCountProbs {
            return self.calculate_start_values_lbfgs(target, max_iter);
        }
        // MeanDispersion: closed-form method of moments in the raw coordinates
        // (log mean, log of the NB2 dispersion µ²/(var − µ)), θ clipped.
        let y: Vec<f64> = match target {
            ResponseData::Univariate(arr) => arr.iter().copied().filter(|v| v.is_finite() && *v >= 0.0).collect(),
            ResponseData::Multivariate(_) => {
                return Err(crate::error::GradientLSSError::InvalidInput(
                    "NegativeBinomial is univariate".to_string(),
                ))
            }
        };
        let n = y.len().max(1) as f64;
        let mean = (y.iter().sum::<f64>() / n).max(1e-3);
        let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let theta = (mean * mean / (var - mean).max(1e-3)).clamp(NB_THETA_START_CLIP.0, NB_THETA_START_CLIP.1);
        let raw = ndarray::Array1::from_vec(vec![mean.ln(), theta.ln()]);
        let (r, probs) = Self::mean_dispersion_to_r_probs(raw[0], raw[1]);
        let loss = self.compute_total_loss(&[r, probs], target);
        Ok((loss, raw))
    }

    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "NegativeBinomial"
    }

    fn is_discrete(&self) -> bool {
        true
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

    /// Measured recipe (research cycles 5–6): watch log θ of the
    /// MeanDispersion head; the legacy (total_count, probs) head gets none.
    /// The mean stays under the global early stopping. NLL loss only.
    fn default_param_early_stopping_params(&self) -> Option<Vec<usize>> {
        (self.parameterization == NbParameterization::MeanDispersion && self.loss_fn() == LossFn::Nll).then(|| vec![1])
    }

    fn should_initialize(&self) -> bool {
        self.initialize
    }

    /// The mean coordinate of the orthogonal head (none for the legacy head).
    fn location_param_index(&self) -> Option<usize> {
        (self.parameterization == NbParameterization::MeanDispersion).then_some(0)
    }

    fn default_start_values(&self, target: &ResponseData) -> Option<ndarray::Array1<f64>> {
        if !self.location_start || self.parameterization != NbParameterization::MeanDispersion {
            return None;
        }
        let ResponseData::Univariate(y) = target else { return None };
        let v: Vec<f64> = y.iter().copied().filter(|t| t.is_finite() && *t >= 0.0).collect();
        if v.is_empty() {
            return None;
        }
        let mean = (v.iter().sum::<f64>() / v.len() as f64).max(1e-4);
        Some(ndarray::Array1::from_vec(vec![
            mean.ln().clamp(NB_LOG_MU_CLAMP.0, NB_LOG_MU_CLAMP.1),
            0.5,
        ]))
    }

    /// Sampling uses a discrete Gamma–Poisson draw, which is not a
    /// smooth function of the parameters under a fixed seed — CRPS finite
    /// differences through it are meaningless (torch has no rsample here either).
    /// Truncated CDF-sum CRPS, mirroring the consumer's NB eval kernel:
    /// K = max(y, mean) + 4*sqrt(var) (until 2026-11-13 capped at 150, which
    /// cut the sum short for larger counts). Params are the torch
    /// convention `(r, probs)` used everywhere in this file (mean =
    /// r*probs/(1-probs)); the pmf comes from `log_prob_scalar`, so the CRPS
    /// is by construction consistent with the NLL this distribution trains
    /// under. Smooth in both params - see the Poisson counterpart.
    fn analytic_crps(&self, params: &[f64], target: &[f64]) -> Option<f64> {
        let r = params[0];
        let probs = params[1];
        let y = target[0];
        if !(r > 0.0) || !(probs > 0.0) || !(probs < 1.0) || !y.is_finite() || y < 0.0 {
            return Some(1e6);
        }
        let mean = r * probs / (1.0 - probs);
        let var = mean / (1.0 - probs);
        let y_int = y.round().max(0.0) as usize;
        let max_k = (y.max(mean) + 4.0 * var.sqrt()).ceil() as usize;
        if max_k > 150 {
            // more than the original loop's 150 terms: do not stop short
            return Some(nb_crps_windowed(r, probs, mean, var.sqrt(), y_int, max_k));
        }
        // pmf recurrence — one multiply per term instead of a full
        // ln_gamma + exp per term (see the Poisson counterpart; the FD
        // gradient path evaluates this 4x per sample per round for the two
        // params, so the per-term cost is the whole ballgame). pmf(0) =
        // (1 - probs)^r in the torch parameterization; pmf(k)/pmf(k-1) =
        // probs * (r + k - 1) / k. No underflow here: this branch has a mean
        // below 150.
        let mut pmf = (r * (-probs).ln_1p()).exp();
        let mut cdf = 0.0;
        let mut crps = 0.0;
        for k in 0..=max_k {
            if k > 0 {
                pmf *= probs * (r + k as f64 - 1.0) / k as f64;
            }
            cdf += pmf;
            let indicator = if k >= y_int { 1.0 } else { 0.0 };
            let d = cdf - indicator;
            crps += d * d;
            if k >= y_int && 1.0 - cdf < 1e-9 {
                break;
            }
        }
        Some(crps)
    }

    fn has_analytic_crps(&self) -> bool {
        true
    }

    fn has_reparameterizable_sampler(&self) -> bool {
        false
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
            ResponseData::Multivariate(_) => {
                panic!("NegativeBinomial is a univariate distribution.")
            }
        }
    }

    /// Analytical gradients for Negative Binomial distribution.
    ///
    /// Using the parameterization where p = 1 - probs (internal), the NLL is:
    /// NLL = -ln_gamma(r+k) + ln_gamma(r) + ln_gamma(k+1) - r*ln(1-probs) - k*ln(probs)
    ///
    /// Gradients w.r.t. distribution parameters:
    /// - dNLL/dr = -digamma(r+k) + digamma(r) - ln(1-probs)
    /// - dNLL/dprobs = r/(1-probs) - k/probs
    ///
    /// Hessians w.r.t. distribution parameters:
    /// - d²NLL/dr² = -trigamma(r+k) + trigamma(r)
    /// - d²NLL/dprobs² = r/(1-probs)² + k/probs²
    fn analytical_gradients(
        &self,
        predictions: &ArrayView2<f64>,
        transformed: &ArrayView2<f64>,
        target: &ResponseData,
    ) -> Option<(Array2<f64>, Array2<f64>)> {
        if self.loss_fn == LossFn::Crps {
            return self.analytical_crps_gradients(predictions, transformed, target);
        }
        if self.loss_fn != LossFn::Nll {
            return None;
        }

        let y = match target {
            ResponseData::Univariate(arr) => arr,
            ResponseData::Multivariate(_) => return None,
        };

        let n_samples = predictions.nrows();

        if self.parameterization == NbParameterization::MeanDispersion {
            return Some(self.mean_dispersion_gradients(predictions, y));
        }

        let r_response_fn = &self.params[0].response_fn;
        let probs_response_fn = &self.params[1].response_fn;

        let t_r = transformed.column(0);
        let t_probs = transformed.column(1);
        let p_r = predictions.column(0);
        let p_probs = predictions.column(1);

        // Batch response-fn derivatives (auto-vectorized); shared by the
        // sequential and parallel paths so we no longer branch on the per-sample
        // scalar derivative calls.
        let (rd_r, rsd_r) = r_response_fn.derivative_batches_from_transformed(&p_r, &t_r);
        let (rd_p, rsd_p) =
            probs_response_fn.derivative_batches_from_transformed(&p_probs, &t_probs);

        // Per-sample (grad_r, hess_r, grad_probs, hess_probs) in prediction space.
        let compute = |i: usize| -> (f64, f64, f64, f64) {
            let r = t_r[i].max(1e-6);
            let probs = t_probs[i].clamp(1e-6, 1.0 - 1e-6);
            let k = y[i];

            if !k.is_finite() || k < 0.0 {
                return (0.0, 0.0, 0.0, 0.0);
            }

            let one_minus_probs = 1.0 - probs;

            // r-parameter digamma/trigamma differences: exact rational sums for
            // integer k, else the asymptotic special functions.
            let (grad_r_psi, hess_r) = nb_psi_diff(r, k)
                .unwrap_or_else(|| (-digamma(r + k) + digamma(r), -trigamma(r + k) + trigamma(r)));
            let grad_r = grad_r_psi - one_minus_probs.ln();
            // Two reciprocals instead of four independent divisions — f64
            // division is the throughput bottleneck of this loop.
            let inv_1mp = 1.0 / one_minus_probs;
            let inv_p = 1.0 / probs;
            let grad_probs = r * inv_1mp - k * inv_p;
            let hess_probs = r * (inv_1mp * inv_1mp) + k * (inv_p * inv_p);

            let r0 = rd_r[i];
            let rs0 = rsd_r[i];
            let g0 = grad_r * r0;
            let h0 = hess_r * r0 * r0 + grad_r * rs0;

            let r1 = rd_p[i];
            let rs1 = rsd_p[i];
            let g1 = grad_probs * r1;
            let h1 = hess_probs * r1 * r1 + grad_probs * rs1;

            (g0, h0, g1, h1)
        };

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));
        // C-order rows are [param0, param1] per sample, so write straight into
        // the contiguous backing slices — no intermediate Vec of tuples.
        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");

        // Threshold matches the numerical path (256): the per-sample work here
        // (digamma / short rational sums) amortizes rayon at the same size.
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

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();

        // Sample each observation's column independently in parallel. Each column
        // gets its own ChaCha stream seeded from `seed + j`, so results are
        // deterministic and independent of how rayon schedules the observations
        // (the previous single shared RNG forced a sequential pass).
        let cols: Vec<Vec<f64>> = (0..n_obs)
            .into_par_iter()
            .map(|j| {
                let total_count = params[[j, 0]];
                let probs = params[[j, 1]];
                let mut col = vec![0.0; n_samples];

                if total_count > 0.0 && probs > 0.0 && probs < 1.0 {
                    // NegativeBinomial as Gamma-Poisson mixture.
                    // rand_distr::Gamma takes (shape, scale), where scale = 1/rate.
                    // For NB(total_count, probs): Gamma(total_count, probs/(1-probs)).
                    let scale = probs / (1.0 - probs);
                    if let Ok(gamma_dist) = RandGamma::new(total_count, scale) {
                        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(j as u64));
                        for v in col.iter_mut() {
                            let lambda: f64 = gamma_dist.sample(&mut rng);
                            if let Ok(poisson_dist) = RandPoisson::new(lambda) {
                                *v = poisson_dist.sample(&mut rng) as f64;
                            }
                        }
                    }
                }
                col
            })
            .collect();

        let mut result = Array2::zeros((n_samples, n_obs));
        for (j, col) in cols.into_iter().enumerate() {
            result.column_mut(j).assign(&ndarray::Array1::from(col));
        }
        result
    }
}

#[cfg(test)]
mod tests {

    /// The closed-form NB CRPS gradients must agree with FD over the analytic
    /// loss for BOTH parameters. FD carries O(eps^2) truncation error and the
    /// two paths use slightly different tail bounds (4 vs 6 sigma), hence the
    /// modest tolerance.
    #[test]
    fn nb_closed_form_crps_gradients_match_finite_differences() {
        use crate::types::ResponseData;
        use ndarray::array;
        let d = NegativeBinomial::new(
            Stabilization::None,
            ResponseFn::Softplus,
            ResponseFn::Sigmoid,
            LossFn::Crps,
            false,
        )
        // the (total_count, probs) head: the CRPS default until 2026-11-14
        .with_parameterization(NbParameterization::TotalCountProbs);
        // raw margins; softplus/sigmoid map them to (r, probs)
        let predictions = array![[2.0_f64, 0.3], [4.0, -0.5], [1.0, 0.8], [3.0, 0.0]];
        let transformed = {
            let mut t = predictions.clone();
            for i in 0..t.nrows() {
                let m0 = predictions[[i, 0]];
                t[[i, 0]] = (m0.exp().ln_1p()).max(1e-12); // softplus
                let m1 = predictions[[i, 1]];
                t[[i, 1]] = 1.0 / (1.0 + (-m1).exp()); // sigmoid
            }
            t
        };
        let y = array![3.0_f64, 8.0, 0.0, 5.0];
        let target = ResponseData::Univariate(&y.view());

        let (g_an, h_an) = d
            .analytical_crps_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("closed-form NB CRPS gradients");
        let (g_fd, _h_fd) = d
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("fd gradients");
        for i in 0..y.len() {
            for j in 0..2 {
                let (a, f) = (g_an[[i, j]], g_fd[[i, j]]);
                assert!(
                    (a - f).abs() <= 2e-3 * f.abs().max(1e-2),
                    "row {i} param {j}: analytic {a} vs fd {f}"
                );
                assert!(
                    h_an[[i, j]] >= 0.05 && h_an[[i, j]].is_finite(),
                    "row {i} param {j}: metric hessian {} invalid",
                    h_an[[i, j]]
                );
            }
        }
    }

    /// Counts above the old 150 / 220-term caps (research proposals 023, 026):
    /// the loss equals a brute-force sum, the gradients its finite difference,
    /// and neither vanishes.
    #[test]
    fn crps_is_exact_for_counts_above_the_old_caps() {
        use crate::types::ResponseData;
        use approx::assert_relative_eq;
        use ndarray::array;
        let d = NegativeBinomial::new(Stabilization::None, ResponseFn::Softplus, ResponseFn::Sigmoid, LossFn::Crps, false)
            .with_parameterization(NbParameterization::TotalCountProbs);
        // (r, mean, y)
        let cases = [(2.0_f64, 700.0_f64, 650.0_f64), (1.5, 190.0, 420.0), (8.0, 400.0, 30.0), (40.0, 5000.0, 5200.0), (3.0, 120.0, 0.0)];
        for (r, mean, y) in cases {
            let probs = mean / (mean + r);
            let var = mean / (1.0 - probs);
            let got = d.analytic_crps(&[r, probs], &[y]).unwrap();
            let top = (y.max(mean) + 4.0 * var.sqrt()).ceil() as usize;
            let y_int = y.round() as usize;
            let (mut cdf, mut want) = (0.0, 0.0);
            for k in 0..=top {
                cdf += d.log_prob_scalar(&[r, probs], k as f64).exp();
                let ind = if k >= y_int { 1.0 } else { 0.0 };
                want += (cdf.min(1.0) - ind).powi(2);
            }
            assert_relative_eq!(got, want, max_relative = 1e-7);
        }
        // margins for (softplus, sigmoid) giving those (r, probs)
        let inv_softplus = |v: f64| (v.exp() - 1.0).ln();
        let logit = |p: f64| (p / (1.0 - p)).ln();
        let mut predictions = ndarray::Array2::zeros((cases.len(), 2));
        let mut transformed = predictions.clone();
        let mut y = ndarray::Array1::zeros(cases.len());
        for (i, (r, mean, yi)) in cases.iter().enumerate() {
            let probs = mean / (mean + r);
            predictions[[i, 0]] = inv_softplus(*r);
            predictions[[i, 1]] = logit(probs);
            transformed[[i, 0]] = *r;
            transformed[[i, 1]] = probs;
            y[i] = *yi;
        }
        let yv = y.view();
        let target = ResponseData::Univariate(&yv);
        let (g_an, h_an) = d.analytical_crps_gradients(&predictions.view(), &transformed.view(), &target).unwrap();
        let (g_fd, _) = d.numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target).unwrap();
        for i in 0..cases.len() {
            for j in 0..2 {
                let (a, f) = (g_an[[i, j]], g_fd[[i, j]]);
                // the loss stops at 4 sd and the gradient table at 6 sd; with r = 2 the
                // tail between them is worth ~0.6 % of the FD gradient
                assert!((a - f).abs() <= 2e-2 * f.abs().max(1e-2), "row {i} param {j}: analytic {a} vs fd {f}");
                assert!(h_an[[i, j]].is_finite() && h_an[[i, j]] >= 0.05, "row {i} param {j}: metric {}", h_an[[i, j]]);
            }
            assert!(g_an[[i, 1]].abs() > 1e-2, "row {i}: probs gradient vanished ({})", g_an[[i, 1]]);
        }
        // large-count rows carry the expected curvature 2 sum_k (dF(k)/dm)^2; recompute it
        // from finite differences of the CDF in the probs margin (row 1: r = 1.5, mean 190).
        // (E_y[(F'(k) - 1{k>=y})^2] = (F'(k) - F(k))^2 + F(k)(1 - F(k)), so the second
        // derivative of the expected CRPS at F' = F is 2 sum_k (dF/dm)^2.)
        {
            let (r, mean) = (1.5_f64, 190.0_f64);
            let m1 = logit(mean / (mean + r));
            let eps = 1e-5;
            let sig = |m: f64| 1.0 / (1.0 + (-m).exp());
            let cdf = |m: f64| -> Vec<f64> {
                let mut acc = 0.0;
                (0..4000usize)
                    .map(|k| {
                        acc += d.log_prob_scalar(&[r, sig(m)], k as f64).exp();
                        acc
                    })
                    .collect()
            };
            let (up, dn) = (cdf(m1 + eps), cdf(m1 - eps));
            let want: f64 = up.iter().zip(&dn).map(|(a, b)| 2.0 * ((a - b) / (2.0 * eps)).powi(2)).sum();
            assert!((h_an[[1, 1]] / want - 1.0).abs() < 0.01, "expected curvature {} vs {want}", h_an[[1, 1]]);
        }
        let _ = array![0.0];
        // a table that starts above 0 (pmf(0) underflows) agrees with the brute-force loss
        let (r, mean, yv) = (4000.0_f64, 5000.0_f64, 5100.0_f64);
        let probs = mean / (mean + r);
        assert!(r * -(-probs).ln_1p() > 600.0);
        let got = d.analytic_crps(&[r, probs], &[yv]).unwrap();
        let var = mean / (1.0 - probs);
        let top = (yv.max(mean) + 4.0 * var.sqrt()).ceil() as usize;
        let (mut cdf, mut want) = (0.0, 0.0);
        for k in 0..=top {
            cdf += d.log_prob_scalar(&[r, probs], k as f64).exp();
            let ind = if k as f64 >= yv { 1.0 } else { 0.0 };
            want += (cdf.min(1.0) - ind).powi(2);
        }
        assert_relative_eq!(got, want, max_relative = 1e-6);
    }

    #[test]
    fn analytic_crps_matches_brute_force_nb() {
        use approx::assert_relative_eq;
        let d = NegativeBinomial::new(
            Stabilization::None,
            ResponseFn::Softplus,
            ResponseFn::Sigmoid,
            LossFn::Crps,
            false,
        )
        // the (total_count, probs) head: the CRPS default until 2026-11-14
        .with_parameterization(NbParameterization::TotalCountProbs);
        for (r, probs, y) in [(4.0_f64, 0.5_f64, 3.0_f64), (2.5, 0.7, 8.0), (10.0, 0.3, 1.0)] {
            let got = d.analytic_crps(&[r, probs], &[y]).expect("nb has analytic crps");
            let mean = r * probs / (1.0 - probs);
            let var = mean / (1.0 - probs);
            let max_k = ((y.max(mean) + 4.0 * var.sqrt()).ceil() as usize).min(150);
            let y_int = y.round() as usize;
            let mut cdf = 0.0;
            let mut want = 0.0;
            for k in 0..=max_k {
                let pmf = d.log_prob_scalar(&[r, probs], k as f64).exp();
                cdf += pmf;
                let ind = if k >= y_int { 1.0 } else { 0.0 };
                want += (cdf - ind) * (cdf - ind);
            }
            assert_relative_eq!(got, want, max_relative = 1e-9);
        }
    }

    /// Both NB params must receive finite, nonzero CRPS gradients.
    #[test]
    fn nb_crps_numerical_gradients_are_finite_and_nonzero() {
        use crate::types::ResponseData;
        use ndarray::array;
        let d = NegativeBinomial::new(
            Stabilization::None,
            ResponseFn::Softplus,
            ResponseFn::Sigmoid,
            LossFn::Crps,
            false,
        )
        // the (total_count, probs) head: the CRPS default until 2026-11-14
        .with_parameterization(NbParameterization::TotalCountProbs);
        let transformed = array![[4.0_f64, 0.5], [2.5, 0.6]];
        // raw = inverse of the response fns; for the gradient smoke it is fine
        // to reuse transformed as raw when responses are monotone — the FD
        // perturbs raw and re-applies the response per param.
        let predictions = transformed.clone();
        let y = array![3.0_f64, 6.0];
        let target = ResponseData::Univariate(&y.view());
        let (grads, hess) = d
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("gradients");
        for g in grads.iter().chain(hess.iter()) {
            assert!(g.is_finite(), "non-finite grad/hess: {g}");
        }
        let gsum: f64 = grads.iter().map(|g| g.abs()).sum();
        assert!(gsum > 1e-8, "gradients must be nonzero, got sum {gsum}");
    }
    use super::*;
    use crate::types::ResponseData;
    use approx::assert_relative_eq;
    use ndarray::array;

    /// MeanDispersion (2026-10-31): the analytical (log µ, log θ) gradients match
    /// central finite differences of the NLL taken through `transform_params`
    /// (the joint map), and both Hessians are positive.
    #[test]
    fn mean_dispersion_gradients_match_finite_differences_through_the_joint_transform() {
        let dist = NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Nll);
        let y = ndarray::array![0.0, 3.0, 7.0, 12.0, 25.0, 1.0];
        let yv = y.view();
        let target = ResponseData::Univariate(&yv);
        let raw = ndarray::array![
            [1.5f64, 1.0], [2.0, 0.5], [2.3, 1.6], [2.6, 2.2], [3.1, 0.2], [0.4, -0.5]
        ];
        let (g, h) = dist.analytical_gradients(&raw.view(), &dist.transform_params(&raw.view()).view(), &target).unwrap();
        let eps = 1e-5;
        for i in 0..raw.nrows() {
            for j in 0..2 {
                let mut up = raw.clone(); up[[i, j]] += eps;
                let mut dn = raw.clone(); dn[[i, j]] -= eps;
                let tu = dist.transform_params(&up.view());
                let td = dist.transform_params(&dn.view());
                let nll_u = -dist.log_prob(&[tu[[i, 0]], tu[[i, 1]]], &[y[i]]);
                let nll_d = -dist.log_prob(&[td[[i, 0]], td[[i, 1]]], &[y[i]]);
                let fd = (nll_u - nll_d) / (2.0 * eps);
                assert!((g[[i, j]] - fd).abs() < 1e-5 * (1.0 + fd.abs()), "row {i} param {j}: analytical {} vs FD {fd}", g[[i, j]]);
                assert!(h[[i, j]] > 0.0, "row {i} param {j}: hessian {} must be positive", h[[i, j]]);
            }
        }
    }

    /// The joint transform reports the legacy (total_count, probs) convention —
    /// mean = r·p/(1−p) = µ, variance = µ + µ²/θ — so every downstream consumer
    /// of (n, p) is untouched; the moments start values land on the raw
    /// (log µ̄, log θ_mom) and are clipped.
    #[test]
    fn mean_dispersion_transform_and_start_values_keep_the_output_convention() {
        let dist = NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Nll);
        let raw = ndarray::array![[2.0f64, 1.0], [0.0, 3.0]];
        let t = dist.transform_params(&raw.view());
        for i in 0..2 {
            let (mu, th) = (raw[[i, 0]].exp(), raw[[i, 1]].exp());
            let (r, probs) = (t[[i, 0]], t[[i, 1]]);
            assert!((r - th).abs() < 1e-12);
            let mean = r * probs / (1.0 - probs);
            let var = mean / (1.0 - probs);
            assert!((mean - mu).abs() < 1e-9, "mean {mean} vs mu {mu}");
            assert!((var - (mu + mu * mu / th)).abs() < 1e-8, "var {var} vs NB2 {}", mu + mu * mu / th);
        }
        // start values: mean 10, var 40 -> theta = 100 / 30
        let y = ndarray::Array1::from_vec(vec![4.0, 6.0, 10.0, 10.0, 14.0, 16.0, 2.0, 18.0, 8.0, 12.0]);
        let yv = y.view();
        let (_, sv) = dist.calculate_start_values(&ResponseData::Univariate(&yv), 10).unwrap();
        let mean = y.mean().unwrap();
        let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / y.len() as f64;
        assert!((sv[0] - mean.ln()).abs() < 1e-12);
        assert!((sv[1] - (mean * mean / (var - mean)).clamp(NB_THETA_START_CLIP.0, NB_THETA_START_CLIP.1).ln()).abs() < 1e-12);
        // legacy mode still answers through the L-BFGS default
        let legacy = NegativeBinomial::new(Stabilization::None, ResponseFn::Softplus, ResponseFn::Sigmoid, LossFn::Nll, true)
            .with_parameterization(NbParameterization::TotalCountProbs);
        assert_eq!(legacy.parameterization(), NbParameterization::TotalCountProbs);
        let (_, sv2) = legacy.calculate_start_values(&ResponseData::Univariate(&yv), 10).unwrap();
        assert_eq!(sv2.len(), 2);
    }

    /// A serialized legacy model (no `parameterization` field) deserializes to
    /// `TotalCountProbs`; a MeanDispersion head round-trips.
    #[test]
    fn parameterization_serde_defaults_to_the_legacy_convention() {
        let legacy_json = r#"{"params":[{"name":"total_count","response_fn":"Softplus"},{"name":"probs","response_fn":"Sigmoid"}],"stabilization":"None","loss_fn":"Nll","initialize":true}"#;
        let d: NegativeBinomial = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(d.parameterization(), NbParameterization::TotalCountProbs);
        let md = NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Nll);
        let back: NegativeBinomial = serde_json::from_str(&serde_json::to_string(&md).unwrap()).unwrap();
        assert_eq!(back.parameterization(), NbParameterization::MeanDispersion);
    }

    /// 2026-11-10 default: `new`/`default()` with NLL build the MeanDispersion
    /// head and KEEP the caller's response functions, so opting out restores
    /// the exact legacy head; the CRPS loss follows the same rule.
    #[test]
    fn mean_dispersion_is_the_nll_default_and_opt_out_is_exact() {
        let d = NegativeBinomial::new(Stabilization::None, ResponseFn::Softplus, ResponseFn::Sigmoid, LossFn::Nll, true);
        assert_eq!(d.parameterization(), NbParameterization::MeanDispersion);
        assert_eq!(NegativeBinomial::default().parameterization(), NbParameterization::MeanDispersion);
        // 2026-11-14 (proposal 031): the CRPS loss uses the same head, and opts out the same way.
        let crps = NegativeBinomial::new(Stabilization::None, ResponseFn::Softplus, ResponseFn::Sigmoid, LossFn::Crps, true);
        assert_eq!(crps.parameterization(), NbParameterization::MeanDispersion);
        assert_eq!(NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Crps).parameterization(), NbParameterization::MeanDispersion);
        let crps_legacy = crps.clone().with_parameterization(NbParameterization::TotalCountProbs);
        assert_eq!(crps_legacy.parameterization(), NbParameterization::TotalCountProbs);
        assert_eq!(crps_legacy.params()[0].response_fn, ResponseFn::Softplus);
        assert_eq!(crps_legacy.params()[1].response_fn, ResponseFn::Sigmoid);
        // a CRPS model saved before the change carries its coordinates and loads with them
        let saved: NegativeBinomial = serde_json::from_value(serde_json::to_value(&crps_legacy).unwrap()).unwrap();
        assert_eq!(saved.parameterization(), NbParameterization::TotalCountProbs);
        // Default MD == the dedicated constructor on transforms / gradients.
        let md = NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Nll);
        let preds = array![[2.0, 0.3], [5.0, -1.0], [0.1, 2.0]];
        let y = array![6.0, 140.0, 0.0];
        let target = ResponseData::Univariate(&y.view());
        let (t_d, t_md) = (d.transform_params(&preds.view()), md.transform_params(&preds.view()));
        assert_eq!(t_d, t_md);
        assert_eq!(
            d.analytical_gradients(&preds.view(), &t_d.view(), &target),
            md.analytical_gradients(&preds.view(), &t_md.view(), &target)
        );
        // Opt-out == a head built the legacy way (response fns preserved).
        let legacy = d.with_parameterization(NbParameterization::TotalCountProbs);
        let t_l = legacy.transform_params(&preds.view());
        assert!((t_l[[0, 0]] - ResponseFn::Softplus.apply_scalar(2.0)).abs() < 1e-12);
        assert!((t_l[[0, 1]] - ResponseFn::Sigmoid.apply_scalar(0.3)).abs() < 1e-12);
    }

    /// θ RECOVERY on synthetic NB2 data (the measured failure: the legacy head
    /// settled at θ ≈ 2.0 on NBA points where ~4.5 fits). µ = exp(1.5 + 0.8·x₀),
    /// θ = 4 everywhere, 6,000 rows; both heads boosted 300 rounds at eta 0.05.
    /// The MeanDispersion head must land within 25 % of the true θ in the
    /// median and must not be further from it than the legacy head.
    #[cfg(feature = "xgboost")]
    #[test]
    fn mean_dispersion_recovers_theta_on_synthetic_nb2_data() {
        use crate::backend::{Backend, BackendDataset, BackendParams, ParamValue, PredictionOutput, TrainConfig, XGBoostBackend};
        use crate::model::{GradientLSS, PredType};
        use rand::RngExt;
        use std::sync::Arc;
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let n = 6000usize;
        let theta_true = 4.0f64;
        let mut x = Array2::<f64>::zeros((n, 3));
        let mut y = ndarray::Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..3 { x[[i, j]] = rng.random_range(-1.0..1.0); }
            let mu = (1.5 + 0.8 * x[[i, 0]]).exp();
            // NB2 draw: Gamma(theta, mu/theta) rate, then Poisson
            let g = RandGamma::new(theta_true, mu / theta_true).unwrap().sample(&mut rng);
            y[i] = RandPoisson::new(g.max(1e-9)).unwrap().sample(&mut rng);
        }
        let fit = |dist: NegativeBinomial| -> (f64, f64) {
            let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(dist));
            let mut ds = <XGBoostBackend as Backend>::Dataset::from_data(x.view(), y.view()).unwrap();
            let mut params = XGBoostBackend::create_params(model.n_params());
            params.set("eta", ParamValue::Float(0.05));
            params.set("max_depth", ParamValue::Int(3));
            let cfg = TrainConfig { num_boost_round: 300, early_stopping_rounds: None, verbose: false, collect_train_metrics: false,
            param_freeze_rounds: None, seed: 3 };
            model.train(&mut ds, None, params, cfg).unwrap();
            let out = match model.predict(&x.view(), PredType::Parameters, 0, &[], 1).unwrap() {
                PredictionOutput::Parameters(p) => p,
                _ => unreachable!(),
            };
            // total_count column = theta in both conventions
            let mut th: Vec<f64> = out.column(0).to_vec();
            th.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_theta = th[th.len() / 2];
            let mean_abs_mu_err = (0..n).map(|i| {
                let (r, pr) = (out[[i, 0]], out[[i, 1]]);
                ((r * pr / (1.0 - pr)) - (1.5 + 0.8 * x[[i, 0]]).exp()).abs()
            }).sum::<f64>() / n as f64;
            (med_theta, mean_abs_mu_err)
        };
        let (th_md, mu_err_md) = fit(NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Nll));
        let (th_legacy, mu_err_legacy) = fit(NegativeBinomial::new(Stabilization::None, ResponseFn::Softplus, ResponseFn::Sigmoid, LossFn::Nll, true).with_parameterization(NbParameterization::TotalCountProbs));
        eprintln!("theta: mean-dispersion {th_md:.2} legacy {th_legacy:.2} (true {theta_true}); mean |mu err| {mu_err_md:.3} vs {mu_err_legacy:.3}");
        assert!((th_md - theta_true).abs() / theta_true < 0.25, "mean-dispersion theta {th_md} vs true {theta_true}");
        assert!((th_md - theta_true).abs() <= (th_legacy - theta_true).abs() + 0.25, "legacy {th_legacy} closer than mean-dispersion {th_md}");
        assert!(mu_err_md < 1.5 * mu_err_legacy + 0.05, "the mean must not degrade: {mu_err_md} vs {mu_err_legacy}");
    }

    #[test]
    fn test_negative_binomial_creation() {
        let dist = NegativeBinomial::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["total_count", "probs"]);
        assert!(dist.is_discrete());
    }

    #[test]
    fn test_negative_binomial_log_prob() {
        let dist = NegativeBinomial::default();
        let log_p = dist.log_prob_scalar(&[5.0, 0.5], 3.0);
        // NB(r=5, probs=0.5) at k=3, statrs p=0.5:
        // ln_gamma(8) - ln_gamma(5) - ln_gamma(4) + 5*ln(0.5) + 3*ln(0.5)
        let expected = ln_gamma(8.0) - ln_gamma(5.0) - ln_gamma(4.0) + 8.0 * 0.5_f64.ln();
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_negative_binomial_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        let dist = NegativeBinomial::new(
            Stabilization::None,
            ResponseFn::Exp,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        )
        .with_parameterization(NbParameterization::TotalCountProbs);
        let predictions = array![[0.5, 0.0], [0.8, -0.5], [1.0, 0.5]];
        let targets = array![3.0, 1.0, 5.0];
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
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-2);
                // True (unfloored) Hessians: both paths must agree on value AND sign.
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
    fn test_nb_lgamma_ratio_matches_ln_gamma() {
        // The product form is exact math; the two forms differ only by float
        // rounding, and at small r the two-lgamma difference itself carries
        // ~1e-9 relative cancellation error (the product side is the more
        // accurate one there). Cover small/large r across the k band, plus the
        // fallback conditions.
        for &r in &[1e-6, 0.37, 1.0, 5.5, 123.4, 9.9e6] {
            for &k in &[0.0, 1.0, 2.0, 17.0, 40.0] {
                let got = nb_lgamma_ratio(r, k).unwrap();
                let want = ln_gamma(r + k) - ln_gamma(r);
                assert_relative_eq!(got, want, epsilon = 1e-8, max_relative = 1e-8);
            }
        }
        assert_eq!(nb_lgamma_ratio(5.0, 41.0), None);
        assert_eq!(nb_lgamma_ratio(5.0, 2.5), None);
        assert_eq!(nb_lgamma_ratio(5.0, -1.0), None);
        assert_eq!(nb_lgamma_ratio(1e7, 3.0), None);
    }

    #[test]
    fn test_negative_binomial_nll() {
        let dist = NegativeBinomial::default();
        let params = array![[5.0, 0.5], [5.0, 0.5]];
        let target = array![3.0, 3.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(ln_gamma(8.0) - ln_gamma(5.0) - ln_gamma(4.0) + 8.0 * 0.5_f64.ln());
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod crps_par_tests {
    use super::*;
    use ndarray::Array1;

    /// Row-parallel NB CRPS gradients must be bit-identical to the serial
    /// fill (both parameters, gradients and metric hessians).
    #[test]
    fn crps_gradients_parallel_matches_serial() {
        let d = NegativeBinomial::new(
            Stabilization::None,
            ResponseFn::Softplus,
            ResponseFn::Sigmoid,
            LossFn::Crps,
            false,
        )
        // the (total_count, probs) head: the CRPS default until 2026-11-14
        .with_parameterization(NbParameterization::TotalCountProbs);
        let n = 3_000usize;
        let mut state = 11u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (1u64 << 31) as f64
        };
        let margins = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { next() * 6.0 - 1.0 } else { next() * 6.0 - 3.0 });
        let transformed = Array2::from_shape_fn((n, 2), |(i, j)| {
            let m = margins[[i, j]];
            if j == 0 { (1.0 + m.exp()).ln() } else { 1.0 / (1.0 + (-m).exp()) }
        });
        let mut y = Array1::from_shape_fn(n, |_| (next() * 25.0).floor());
        y[5] = -2.0;
        let yv = y.view();
        let target = ResponseData::Univariate(&yv);
        let (gp, hp) = d
            .analytical_crps_gradients_impl(&margins.view(), &transformed.view(), &target, false)
            .unwrap();
        let (gs, hs) = d
            .analytical_crps_gradients_impl(&margins.view(), &transformed.view(), &target, true)
            .unwrap();
        assert_eq!(gp, gs, "gradients differ between parallel and serial");
        assert_eq!(hp, hs, "hessians differ between parallel and serial");
        assert!(gp.iter().any(|v| *v != 0.0));
    }
}


#[cfg(test)]
mod mean_dispersion_clamp_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;

    /// Means far above the old e⁶ ≈ 403 cap are representable, the gradients
    /// at such means match finite differences of the NLL, and a margin beyond
    /// the clamp gets zero gradient.
    #[test]
    fn large_means_are_representable_and_clamped_margins_are_inert() {
        let d = NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Nll);
        let (mu, th) = (3000.0f64, 25.0f64);
        let preds = Array2::from_shape_vec((1, 2), vec![mu.ln(), th.ln()]).unwrap();
        let t = d.transform_params(&preds.view());
        let mean = t[[0, 0]] * t[[0, 1]] / (1.0 - t[[0, 1]]);
        assert!((mean / mu - 1.0).abs() < 1e-9, "reported mean {mean} vs {mu}");
        let y = Array1::from_vec(vec![2500.0]);
        let (g, _) = d
            .analytical_gradients(&preds.view(), &t.view(), &ResponseData::Univariate(&y.view()))
            .unwrap();
        let nll = |f: f64, gg: f64| {
            let p = Array2::from_shape_vec((1, 2), vec![f, gg]).unwrap();
            let tt = d.transform_params(&p.view());
            -d.log_prob(&[tt[[0, 0]], tt[[0, 1]]], &[2500.0])
        };
        let e = 1e-5;
        let df = (nll(mu.ln() + e, th.ln()) - nll(mu.ln() - e, th.ln())) / (2.0 * e);
        let dg = (nll(mu.ln(), th.ln() + e) - nll(mu.ln(), th.ln() - e)) / (2.0 * e);
        assert!((g[[0, 0]] - df).abs() < 1e-3 * df.abs().max(1.0), "{} vs {df}", g[[0, 0]]);
        assert!((g[[0, 1]] - dg).abs() < 1e-3 * dg.abs().max(1.0), "{} vs {dg}", g[[0, 1]]);
        // beyond the clamp: zero gradient for that coordinate
        let far = Array2::from_shape_vec((1, 2), vec![20.0, 15.0]).unwrap();
        let tf = d.transform_params(&far.view());
        let (gf, _) = d
            .analytical_gradients(&far.view(), &tf.view(), &ResponseData::Univariate(&y.view()))
            .unwrap();
        assert_eq!(gf[[0, 0]], 0.0);
        assert_eq!(gf[[0, 1]], 0.0);
    }
}

#[cfg(test)]
mod log_theta_fisher_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;

    /// The exact-sum Fisher information equals the Monte-Carlo mean of the
    /// observed information, matches both limits, and `Damped` ≥ both pieces.
    #[test]
    fn fisher_matches_monte_carlo_and_limits() {
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        for &(mu, th) in &[(3.0f64, 1.0f64), (40.0, 4.0), (5.0, 60.0), (800.0, 20.0)] {
            let n = 200_000;
            let y: Array1<f64> = Array1::from_iter((0..n).map(|_| {
                let lam = RandGamma::new(th, mu / th).unwrap().sample(&mut rng);
                RandPoisson::new(lam).unwrap().sample(&mut rng)
            }));
            let preds = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { mu.ln() } else { th.ln() });
            let target = ResponseData::Univariate(&y.view());
            let mk = |m| NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Nll).with_hessian_mode(m);
            // Observed with the floor removed is not exposed; use Fisher/Damped/Observed relations.
            let tr = mk(HessianMode::Fisher).transform_params(&preds.view());
            let (g_f, h_f) = mk(HessianMode::Fisher).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
            let (g_o, h_o) = mk(HessianMode::Observed).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
            let (g_d, h_d) = mk(HessianMode::Damped).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
            assert_eq!(g_f, g_o);
            assert_eq!(g_f, g_d);
            let fis = nb_log_theta_fisher(mu, th);
            assert!((h_f[[0, 1]] / fis.max(1e-8) - 1.0).abs() < 1e-9, "{} vs {fis}", h_f[[0, 1]]);
            // E[score²] = Fisher (the score has no floor, unlike the observed Hessian).
            let e_s2 = g_f.column(1).iter().map(|v| v * v).sum::<f64>() / n as f64;
            assert!((e_s2 - fis).abs() / fis < 0.05, "µ={mu} θ={th}: E[s²]={e_s2} Fisher={fis}");
            for i in 0..n {
                assert!(h_d[[i, 1]] >= h_f[[i, 1]] - 1e-15);
                assert!(h_d[[i, 1]] + 1e-12 >= h_o[[i, 1]].min(h_d[[i, 1]]));
            }
        }
        // limits
        let glim = |th: f64| th * th * trigamma(th) - th;
        assert!((nb_log_theta_fisher(1e5, 2.0) / glim(2.0) - 1.0).abs() < 2e-3);
        let (mu, th) = (10.0, 300.0);
        let pois = mu * mu / (2.0 * (mu + th) * (mu + th));
        assert!((nb_log_theta_fisher(mu, th) / pois - 1.0).abs() < 0.05);
        assert_eq!(nb_log_theta_fisher(0.0, 1.0), 0.0);
    }

    /// The quadrature against the direct series θ² Σ_j P(Y > j)/(θ + j)² − θµ/(µ+θ)
    /// in the region where the series is numerically safe.
    #[test]
    fn quadrature_matches_the_exact_series() {
        fn series(mu: f64, theta: f64) -> f64 {
            let s = mu + theta;
            let ln_q = (mu / s).ln();
            let mut ln_p = theta * (theta / s).ln();
            let (mut cdf, mut acc) = (0.0f64, 0.0f64);
            let max_j = (mu + 60.0 * (mu + mu * mu / theta).sqrt() + 100.0) as usize;
            for j in 0..=max_j {
                let jf = j as f64;
                cdf += ln_p.exp();
                let sf = (1.0 - cdf).max(0.0);
                acc += sf / ((theta + jf) * (theta + jf));
                if jf > mu && sf < 1e-14 {
                    break;
                }
                ln_p += ((jf + theta) / (jf + 1.0)).ln() + ln_q;
            }
            theta * theta * acc - theta * mu / s
        }
        for &th in &[0.05f64, 0.3, 1.0, 3.0, 10.0, 50.0, 300.0] {
            for &mu in &[0.1f64, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0, 4000.0] {
                let ex = series(mu, th);
                if ex < 1e-6 {
                    continue; // the series cancels catastrophically there
                }
                let q = nb_log_theta_fisher(mu, th);
                assert!((q / ex - 1.0).abs() < 1e-2, "µ={mu} θ={th}: quad {q} series {ex}");
            }
        }
    }

    /// Old files (no `hessian_mode`) keep the floored observed curvature.
    #[test]
    fn serde_default_is_observed() {
        let mut v: serde_json::Value =
            serde_json::to_value(NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Nll)).unwrap();
        v.as_object_mut().unwrap().remove("hessian_mode");
        let back: NegativeBinomial = serde_json::from_value(v).unwrap();
        assert_eq!(back.hessian_mode(), HessianMode::Observed);
    }
}

#[cfg(test)]
mod hessian_mode_default_tests {
    use super::*;

    #[test]
    fn constructors_default_to_damped() {
        assert_eq!(NegativeBinomial::default().hessian_mode(), HessianMode::Damped);
        assert_eq!(
            NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Nll).hessian_mode(),
            HessianMode::Damped
        );
    }
}

#[cfg(test)]
mod location_start_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn location_start_sets_log_mean_only() {
        let y = array![0.0, 4.0, 8.0];
        let t = ResponseData::Univariate(&y.view());
        assert!(NegativeBinomial::default().with_location_start(false).default_start_values(&t).is_none());
        let sv = NegativeBinomial::default().default_start_values(&t).unwrap();
        assert_eq!(sv.to_vec(), vec![4.0f64.ln(), 0.5]);
    }
}

#[cfg(test)]
mod location_start_default_tests {
    use super::*;

    #[test]
    fn serde_default_is_off_and_legacy_head_ignores_it() {
        let mut v = serde_json::to_value(NegativeBinomial::default()).unwrap();
        v.as_object_mut().unwrap().remove("location_start");
        let back: NegativeBinomial = serde_json::from_value(v).unwrap();
        assert!(!back.location_start());
        let y = ndarray::array![1.0, 2.0];
        let legacy = NegativeBinomial::default().with_parameterization(NbParameterization::TotalCountProbs);
        assert!(legacy.default_start_values(&ResponseData::Univariate(&y.view())).is_none());
    }

    /// Proposals 029, 031: CRPS in (log µ, log θ). The gradient is the finite
    /// difference of the loss through the joint transform and the Hessian is
    /// 2 Σ_k (∂F(k)/∂η)² recomputed from finite differences of the CDF.
    #[test]
    fn crps_mean_dispersion_gradients_match_finite_differences() {
        use crate::types::ResponseData;
        let d = NegativeBinomial::new_mean_dispersion(Stabilization::None, LossFn::Crps);
        assert_eq!(d.parameterization(), NbParameterization::MeanDispersion);
        // (µ, θ, y): small and large counts, near-Poisson and overdispersed
        let cases = [(9.5_f64, 40.0_f64, 12.0_f64), (3.0, 1.2, 0.0), (25.0, 4.0, 60.0), (190.0, 1.5, 420.0), (700.0, 6.0, 650.0), (9.0, 400.0, 4.0)];
        let n = cases.len();
        let mut raw = ndarray::Array2::zeros((n, 2));
        let mut y = ndarray::Array1::zeros(n);
        for (i, (mu, th, yi)) in cases.iter().enumerate() {
            raw[[i, 0]] = mu.ln();
            raw[[i, 1]] = th.ln();
            y[i] = *yi;
        }
        let tr = d.transform_params(&raw.view());
        let yv = y.view();
        let (g, h) = d.analytical_gradients(&raw.view(), &tr.view(), &ResponseData::Univariate(&yv)).expect("analytic");
        let crps = |f: f64, gg: f64, yi: f64| {
            let (r, p) = NegativeBinomial::mean_dispersion_to_r_probs(f, gg);
            d.analytic_crps(&[r, p], &[yi]).unwrap()
        };
        let cdf = |f: f64, gg: f64, top: usize| -> Vec<f64> {
            let (r, p) = NegativeBinomial::mean_dispersion_to_r_probs(f, gg);
            let mut acc = 0.0;
            (0..=top).map(|k| { acc += d.log_prob_scalar(&[r, p], k as f64).exp(); acc }).collect()
        };
        let eps = 1e-5;
        for (i, (mu, th, yi)) in cases.iter().enumerate() {
            let (f, gg) = (mu.ln(), th.ln());
            let fd = [
                (crps(f + eps, gg, *yi) - crps(f - eps, gg, *yi)) / (2.0 * eps),
                (crps(f, gg + eps, *yi) - crps(f, gg - eps, *yi)) / (2.0 * eps),
            ];
            let sd = (mu + mu * mu / th).sqrt();
            let top = (yi.max(*mu) + 12.0 * sd).ceil() as usize;
            let (c0, c_f, c_g) = (cdf(f, gg, top), [cdf(f + eps, gg, top), cdf(f - eps, gg, top)], [cdf(f, gg + eps, top), cdf(f, gg - eps, top)]);
            let curv = |c: &[Vec<f64>; 2]| 2.0 * (0..c0.len()).map(|k| ((c[0][k] - c[1][k]) / (2.0 * eps)).powi(2)).sum::<f64>();
            let want_h = [curv(&c_f), curv(&c_g)];
            for j in 0..2 {
                // the loss stops at 4 sd, the gradient table at 6 sd (see the test above)
                assert!((g[[i, j]] - fd[j]).abs() <= 2e-2 * fd[j].abs().max(1e-2), "row {i} param {j}: analytic {} vs fd {}", g[[i, j]], fd[j]);
                let want = want_h[j].max(0.05);
                assert!((h[[i, j]] - want).abs() <= 1e-2 * want, "row {i} param {j}: hessian {} vs {}", h[[i, j]], want);
            }
        }
        // a coordinate outside its clamp has no gradient
        let mut far = raw.clone();
        far[[0, 0]] = NB_LOG_MU_CLAMP.1 + 1.0;
        far[[1, 1]] = NB_LOG_THETA_CLAMP.0 - 1.0;
        let tr = d.transform_params(&far.view());
        let (g, _) = d.analytical_gradients(&far.view(), &tr.view(), &ResponseData::Univariate(&yv)).unwrap();
        assert_eq!(g[[0, 0]], 0.0);
        assert_eq!(g[[1, 1]], 0.0);
    }
}
