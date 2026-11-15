//! Beta distribution implementation.

use super::base::{Distribution, DistributionParam, HessianMode, LossFn, Stabilization};
use super::gamma::{digamma_minus_ln, shape_fisher};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta as RandBeta, Distribution as RandDistribution};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::constants::trigamma;

/// Boosting coordinates of [`Beta`].
///
/// `Concentrations` boosts each of (concentration1 a, concentration0 b)
/// through its own response function. Under `Exp` the Fisher information of
/// (log a, log b) has correlation → −1 as the precision φ = a + b grows
/// (−0.95 at φ = 20, −0.99 at φ = 100): a and b both move the mean a/φ, so a
/// diagonal Newton booster learns the precision far too slowly (same
/// collinearity as Gamma's (concentration, rate); research cycle 7 audit).
///
/// `MeanPrecision` boosts (logit µ, log φ) with a = µφ, b = (1 − µ)φ
/// (Ferrari & Cribari-Neto 2004). Its Fisher correlation is small exactly
/// where the concentrations' is bad (|ρ| ≤ 0.24 at φ ≥ 20). Start values
/// are method-of-moments. Transformed / predicted parameters are STILL
/// reported as `(concentration1, concentration0)`. NLL only.
///
/// Default since 2026-11-10 (`Beta::new` with `ResponseFn::Exp` and the NLL
/// loss, and `Beta::default()`): with Damped curvature it beat the
/// concentrations head 5/5 seeds on every dataset measured — synthetic
/// φ 55–1100 −0.27 nats, φ 3–55 −0.10, bike-sharing casual share −0.04 vs
/// the Damped concentrations head (the Observed one diverged there). Opt out
/// with `.with_parameterization(BetaParameterization::Concentrations)`;
/// model files saved before the field existed load as `Concentrations`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BetaParameterization {
    Concentrations,
    MeanPrecision,
}

impl BetaParameterization {
    fn legacy() -> Self {
        BetaParameterization::Concentrations
    }
}

/// `MeanPrecision` raw clamps: logit µ ∈ [−30, 30], log φ ∈ [−10, 25].
/// Outside them the gradient of that coordinate is zeroed.
const BETA_LOGIT_MEAN_CLAMP: (f64, f64) = (-30.0, 30.0);
const BETA_LOG_PREC_CLAMP: (f64, f64) = (-10.0, 25.0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Beta {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
    /// Curvature handed to the booster (see [`HessianMode`]). Files written
    /// before the field existed load as `Observed` (the historical curvature).
    #[serde(default = "Beta::legacy_hessian_mode")]
    hessian_mode: HessianMode,
    /// Boosting coordinates (see [`BetaParameterization`]); old files load
    /// as `Concentrations`.
    #[serde(default = "BetaParameterization::legacy")]
    parameterization: BetaParameterization,
}

impl Beta {
    pub fn new(
        stabilization: Stabilization,
        response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("concentration1", response_fn),
            DistributionParam::new("concentration0", response_fn),
        ];
        Self {
            params,
            stabilization,
            loss_fn,
            initialize,
            hessian_mode: Self::default_hessian_mode(),
            parameterization: if response_fn == ResponseFn::Exp && loss_fn == LossFn::Nll {
                BetaParameterization::MeanPrecision
            } else {
                BetaParameterization::Concentrations
            },
        }
    }

    fn legacy_hessian_mode() -> HessianMode {
        HessianMode::Observed
    }

    /// `Damped` since 2026-11-10 (research cycle 7): the observed curvature
    /// diverged on a real proportion target (bike_share: NLL 2545 vs −1.52).
    fn default_hessian_mode() -> HessianMode {
        HessianMode::Damped
    }

    pub fn default() -> Self {
        Self::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false)
    }

    /// Choose the curvature handed to the Newton step (see [`HessianMode`]).
    pub fn with_hessian_mode(mut self, mode: HessianMode) -> Self {
        self.hessian_mode = mode;
        self
    }

    pub fn hessian_mode(&self) -> HessianMode {
        self.hessian_mode
    }

    /// Choose the boosting coordinates (see [`BetaParameterization`]).
    /// `MeanPrecision` needs the NLL loss; with CRPS it warns and keeps
    /// `Concentrations`.
    pub fn with_parameterization(mut self, parameterization: BetaParameterization) -> Self {
        if parameterization == BetaParameterization::MeanPrecision && self.loss_fn != LossFn::Nll {
            eprintln!(
                "gradientlss: Beta MeanPrecision supports NLL only — keeping the \
                 Concentrations head"
            );
            return self;
        }
        self.parameterization = parameterization;
        self
    }

    pub fn parameterization(&self) -> BetaParameterization {
        self.parameterization
    }

    /// Raw (f, g) = (logit µ, log φ) → (a, b), clamped.
    fn mean_precision_to_ab(f: f64, g: f64) -> (f64, f64) {
        let f = if f.is_finite() { f.clamp(BETA_LOGIT_MEAN_CLAMP.0, BETA_LOGIT_MEAN_CLAMP.1) } else { 0.0 };
        let g = if g.is_finite() { g.clamp(BETA_LOG_PREC_CLAMP.0, BETA_LOG_PREC_CLAMP.1) } else { 0.0 };
        let phi = g.exp();
        let mu = 1.0 / (1.0 + (-f).exp());
        (mu * phi, (1.0 - mu) * phi)
    }

    /// Method-of-moments start in raw (logit µ, log φ) on the labels in (0, 1).
    fn mean_precision_start(y: &[f64]) -> (f64, f64) {
        let v: Vec<f64> = y.iter().copied().filter(|t| t.is_finite() && *t > 0.0 && *t < 1.0).collect();
        if v.len() < 2 {
            return (0.0, 0.0);
        }
        let n = v.len() as f64;
        let m = (v.iter().sum::<f64>() / n).clamp(1e-6, 1.0 - 1e-6);
        let var = v.iter().map(|t| (t - m).powi(2)).sum::<f64>() / n;
        let phi = if var > 0.0 { (m * (1.0 - m) / var - 1.0).max(0.1) } else { 1e3 };
        ((m / (1.0 - m)).ln(), phi.ln())
    }

    /// `MeanPrecision` NLL gradients / Hessians in raw (f, g) = (logit µ, log φ).
    /// With a = µφ, b = (1−µ)φ, m = φµ(1−µ), h(x) = ψ(x) − ln x,
    /// F(x) = x²ψ′(x) − x (cancellation-safe forms):
    ///   ∂/∂f = m[logit µ − logit y + h(a) − h(b)]
    ///          Fisher m²(ψ′(a) + ψ′(b)); observed adds (1 − 2µ)·∂/∂f
    ///   ∂/∂g = a(ln µ − ln y) + b(ln(1−µ) − ln(1−y)) + a h(a) + b h(b) − φ h(φ)
    ///          Fisher F(a) + F(b) − F(φ); observed adds ∂/∂g
    fn mean_precision_gradients(
        &self,
        predictions: &ArrayView2<f64>,
        y: &ndarray::ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let n = predictions.nrows();
        let mode = self.hessian_mode;
        let compute = |i: usize| -> [f64; 4] {
            let f_raw = predictions[[i, 0]];
            let g_raw = predictions[[i, 1]];
            let f = if f_raw.is_finite() { f_raw.clamp(BETA_LOGIT_MEAN_CLAMP.0, BETA_LOGIT_MEAN_CLAMP.1) } else { 0.0 };
            let g = if g_raw.is_finite() { g_raw.clamp(BETA_LOG_PREC_CLAMP.0, BETA_LOG_PREC_CLAMP.1) } else { 0.0 };
            let phi = g.exp();
            // ln µ, ln(1−µ) stably from the logit
            let ln_mu = -(-f).exp().ln_1p();
            let ln_1mmu = -f.exp().ln_1p();
            let (mu, one_m_mu) = (ln_mu.exp(), ln_1mmu.exp());
            let (a, b) = (mu * phi, one_m_mu * phi);
            let m = phi * mu * one_m_mu;
            let h_f_fisher = (m * m * (trigamma(a) + trigamma(b))).max(1e-300);
            let h_g_fisher = (shape_fisher(a) + shape_fisher(b) - shape_fisher(phi)).max(1e-12);
            let yi = y[i];
            if !yi.is_finite() || yi <= 0.0 || yi >= 1.0 {
                return [0.0, h_f_fisher, 0.0, h_g_fisher];
            }
            let (ln_y, ln_1my) = (yi.ln(), (-yi).ln_1p());
            let gf = m * (f - (ln_y - ln_1my) + digamma_minus_ln(a) - digamma_minus_ln(b));
            let gg = a * (ln_mu - ln_y) + b * (ln_1mmu - ln_1my) + a * digamma_minus_ln(a)
                + b * digamma_minus_ln(b)
                - phi * digamma_minus_ln(phi);
            let hf_obs = h_f_fisher + (1.0 - 2.0 * mu) * gf;
            let hg_obs = h_g_fisher + gg;
            let (hf, hg) = match mode {
                HessianMode::Observed => (hf_obs, hg_obs),
                HessianMode::Fisher => (h_f_fisher, h_g_fisher),
                HessianMode::Damped => (hf_obs.max(h_f_fisher), hg_obs.max(h_g_fisher)),
            };
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

    /// Apply the Hessian mode to a Concentrations-head row: `obs` is the
    /// observed raw curvature, the Fisher one is I·r′² (the r″ chain term
    /// has expectation zero).
    fn concentration_curvature(&self, obs: f64, fisher: f64) -> f64 {
        match self.hessian_mode {
            HessianMode::Observed => obs,
            HessianMode::Fisher => fisher,
            HessianMode::Damped => obs.max(fisher),
        }
    }

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let a = params[0]; // concentration1
        let b = params[1]; // concentration0

        if a <= 0.0 || b <= 0.0 || a.is_infinite() || b.is_infinite() {
            return f64::NEG_INFINITY;
        }
        if !(0.0..=1.0).contains(&target) {
            return f64::NEG_INFINITY;
        }

        // ln_pdf = ln_gamma(a+b) - ln_gamma(a) - ln_gamma(b) + (a-1)*ln(x) + (b-1)*ln(1-x)
        let log_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
        let ln_x = if target == 0.0 {
            if a > 1.0 {
                f64::NEG_INFINITY
            } else if (a - 1.0).abs() < f64::EPSILON {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            (a - 1.0) * target.ln()
        };
        let ln_1mx = if (target - 1.0).abs() < f64::EPSILON {
            if b > 1.0 {
                f64::NEG_INFINITY
            } else if (b - 1.0).abs() < f64::EPSILON {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            (b - 1.0) * (1.0 - target).ln()
        };

        -log_beta + ln_x + ln_1mx
    }
}

#[typetag::serde]
impl Distribution for Beta {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Beta"
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

    /// The mean coordinate of the orthogonal head (none for the legacy head).
    fn location_param_index(&self) -> Option<usize> {
        (self.parameterization == BetaParameterization::MeanPrecision).then_some(0)
    }

    /// Measured recipe (research cycle 8): watch log φ of the MeanPrecision
    /// head (bike_share −0.019, 5/5; simulations unchanged). The
    /// Concentrations head gets none. NLL loss only.
    fn default_param_early_stopping_params(&self) -> Option<Vec<usize>> {
        (self.parameterization == BetaParameterization::MeanPrecision && self.loss_fn == LossFn::Nll)
            .then(|| vec![1])
    }

    fn transform_params(&self, predictions: &ArrayView2<f64>) -> Array2<f64> {
        if self.parameterization == BetaParameterization::Concentrations {
            let mut result = Array2::zeros(predictions.dim());
            for (i, param) in self.params.iter().enumerate() {
                let col = predictions.column(i);
                let mut out_col = result.column_mut(i);
                param.response_fn.apply_into(&col, &mut out_col);
            }
            return result;
        }
        let n = predictions.nrows();
        let mut result = Array2::zeros((n, 2));
        for i in 0..n {
            let (a, b) = Self::mean_precision_to_ab(predictions[[i, 0]], predictions[[i, 1]]);
            result[[i, 0]] = a;
            result[[i, 1]] = b;
        }
        result
    }

    fn default_start_values(&self, target: &ResponseData) -> Option<ndarray::Array1<f64>> {
        if self.parameterization != BetaParameterization::MeanPrecision {
            return None;
        }
        let ResponseData::Univariate(y) = target else { return None };
        let (f, g) = Self::mean_precision_start(&y.to_vec());
        Some(ndarray::Array1::from_vec(vec![f, g]))
    }

    fn calculate_start_values(
        &self,
        target: &ResponseData,
        max_iter: usize,
    ) -> crate::error::Result<(f64, ndarray::Array1<f64>)> {
        if self.parameterization == BetaParameterization::Concentrations {
            return self.calculate_start_values_default(target, max_iter);
        }
        // MeanPrecision: refine the moments start with the generic optimiser
        // (it works in the raw (logit µ, log φ) coordinates via transform_params).
        let (loss, sv) = self.calculate_start_values_default(target, max_iter)?;
        if sv.iter().all(|v| v.is_finite()) {
            return Ok((loss, sv));
        }
        let start = self.default_start_values(target).expect("MeanPrecision");
        let (a, b) = Self::mean_precision_to_ab(start[0], start[1]);
        Ok((self.compute_total_loss(&[a, b], target), start))
    }

    /// Sampling uses rejection sampling (built on Gamma draws), which is not a
    /// smooth function of the parameters under a fixed seed — CRPS finite
    /// differences through it are meaningless (torch has no rsample here either).
    fn has_reparameterizable_sampler(&self) -> bool {
        false
    }

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_scalar(params, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                let conc1_col = params.column(0);
                let conc0_col = params.column(1);
                crate::distributions::util::par_nansum(y.len(), |i| {
                    let y_val = y[i];
                    if y_val < 0.0 || y_val > 1.0 {
                        f64::INFINITY
                    } else {
                        -self.log_prob_scalar(&[conc1_col[i], conc0_col[i]], y_val)
                    }
                })
            }
            ResponseData::Multivariate(_) => panic!("Beta is a univariate distribution."),
        }
    }

    /// Analytical gradients for Beta distribution.
    ///
    /// NLL = ln_gamma(a) + ln_gamma(b) - ln_gamma(a+b) - (a-1)*ln(x) - (b-1)*ln(1-x)
    ///
    /// Gradients w.r.t. distribution parameters:
    /// - dNLL/da = digamma(a) - digamma(a+b) - ln(x)
    /// - dNLL/db = digamma(b) - digamma(a+b) - ln(1-x)
    ///
    /// Hessians w.r.t. distribution parameters:
    /// - d²NLL/da² = trigamma(a) - trigamma(a+b)
    /// - d²NLL/db² = trigamma(b) - trigamma(a+b)
    ///
    /// Chain rule applied for response function: dNLL/dpred = dNLL/dparam * dparam/dpred
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

        if self.parameterization == BetaParameterization::MeanPrecision {
            return Some(self.mean_precision_gradients(predictions, y));
        }

        let n_samples = predictions.nrows();

        let conc1_response_fn = &self.params[0].response_fn;
        let conc0_response_fn = &self.params[1].response_fn;

        let t_conc1 = transformed.column(0);
        let t_conc0 = transformed.column(1);
        let p_conc1 = predictions.column(0);
        let p_conc0 = predictions.column(1);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        if n_samples >= 4096 {
            // Pre-compute batch derivatives for both parameters
            let (rd_a, rsd_a) =
                conc1_response_fn.derivative_batches_from_transformed(&p_conc1, &t_conc1);
            let (rd_b, rsd_b) =
                conc0_response_fn.derivative_batches_from_transformed(&p_conc0, &t_conc0);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let a = t_conc1[i].max(1e-6);
                let b = t_conc0[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || yi >= 1.0 || !yi.is_finite() {
                    return (0.0, 0.0, 0.0, 0.0);
                }

                let ln_y = yi.ln();
                let ln_1my = (1.0 - yi).ln();
                let ab = a + b;
                let digamma_ab = digamma(ab);
                let trigamma_ab = trigamma(ab);

                let grad_a = digamma(a) - digamma_ab - ln_y;
                let hess_a = trigamma(a) - trigamma_ab;
                let grad_b = digamma(b) - digamma_ab - ln_1my;
                let hess_b = trigamma(b) - trigamma_ab;

                let r0 = rd_a[i];
                let rs0 = rsd_a[i];
                let g0 = grad_a * r0;
                let h0 = self.concentration_curvature(hess_a * r0 * r0 + grad_a * rs0, hess_a * r0 * r0);

                let r1 = rd_b[i];
                let rs1 = rsd_b[i];
                let g1 = grad_b * r1;
                let h1 = self.concentration_curvature(hess_b * r1 * r1 + grad_b * rs1, hess_b * r1 * r1);

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
                let a = t_conc1[i].max(1e-6);
                let b = t_conc0[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || yi >= 1.0 || !yi.is_finite() {
                    gradients[[i, 0]] = 0.0;
                    hessians[[i, 0]] = 1e-6;
                    gradients[[i, 1]] = 0.0;
                    hessians[[i, 1]] = 1e-6;
                    continue;
                }

                let ln_y = yi.ln();
                let ln_1my = (1.0 - yi).ln();
                let ab = a + b;
                let digamma_ab = digamma(ab);
                let trigamma_ab = trigamma(ab);

                let grad_a = digamma(a) - digamma_ab - ln_y;
                let hess_a = trigamma(a) - trigamma_ab;
                let grad_b = digamma(b) - digamma_ab - ln_1my;
                let hess_b = trigamma(b) - trigamma_ab;

                let r0 = conc1_response_fn.derivative(p_conc1[i]);
                let rs0 = conc1_response_fn.second_derivative(p_conc1[i]);
                let g0 = grad_a * r0;
                let h0 = self.concentration_curvature(hess_a * r0 * r0 + grad_a * rs0, hess_a * r0 * r0);

                let r1 = conc0_response_fn.derivative(p_conc0[i]);
                let rs1 = conc0_response_fn.second_derivative(p_conc0[i]);
                let g1 = grad_b * r1;
                let h1 = self.concentration_curvature(hess_b * r1 * r1 + grad_b * rs1, hess_b * r1 * r1);

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
            let concentration1 = params[[j, 0]];
            let concentration0 = params[[j, 1]];

            if concentration1 > 0.0 && concentration0 > 0.0 {
                if let Ok(dist) = RandBeta::new(concentration1, concentration0) {
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
    fn test_beta_creation() {
        let dist = Beta::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["concentration1", "concentration0"]);
    }

    #[test]
    fn test_beta_log_prob() {
        let dist = Beta::default();
        let log_p = dist.log_prob_scalar(&[2.0, 2.0], 0.5);
        // Beta(2,2) at x=0.5: ln_gamma(4)-ln_gamma(2)-ln_gamma(2)+(2-1)*ln(0.5)+(2-1)*ln(0.5)
        let expected = ln_gamma(4.0) - ln_gamma(2.0) - ln_gamma(2.0)
            + 1.0 * (0.5_f64).ln()
            + 1.0 * (0.5_f64).ln();
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_beta_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        // numerical path uses per-parameter response functions (the
        // Concentrations head); the observed curvature is what FD sees
        let dist = Beta::default()
            .with_parameterization(BetaParameterization::Concentrations)
            .with_hessian_mode(HessianMode::Observed);
        let predictions = array![[0.5, 0.3], [-0.2, 0.8], [1.0, -0.5]];
        let targets = array![0.3, 0.7, 0.5];
        let target = ResponseData::Univariate(&targets.view());

        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");

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
    fn test_beta_nll() {
        let dist = Beta::default();
        let params = array![[2.0, 2.0], [2.0, 2.0]];
        let target = array![0.5, 0.5];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(ln_gamma(4.0) - ln_gamma(2.0) - ln_gamma(2.0)
            + 1.0 * (0.5_f64).ln()
            + 1.0 * (0.5_f64).ln());
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod mean_precision_tests {
    use super::*;
    use ndarray::{Array1, array};

    fn mp(mode: HessianMode) -> Beta {
        Beta::default().with_parameterization(BetaParameterization::MeanPrecision).with_hessian_mode(mode)
    }

    fn nll_raw(d: &Beta, f: f64, g: f64, y: f64) -> f64 {
        let (a, b) = Beta::mean_precision_to_ab(f, g);
        -d.log_prob_scalar(&[a, b], y)
    }

    /// Gradients and observed curvature match central differences, from a
    /// diffuse Beta to φ = 1e6 (where the naive digamma forms cancel).
    #[test]
    fn gradients_match_finite_differences() {
        let d = mp(HessianMode::Observed);
        for &(f, g, y) in &[(0.3, 0.5, 0.4), (-2.0, 3.0, 0.08), (1.5, 6.0, 0.83), (0.2, 13.8, 0.551)] {
            let pred = array![[f, g]];
            let yy = array![y];
            let (gr, h) = d.mean_precision_gradients(&pred.view(), &yy.view());
            let e = 1e-4;
            let fd_f = (nll_raw(&d, f + e, g, y) - nll_raw(&d, f - e, g, y)) / (2.0 * e);
            let fd_g = (nll_raw(&d, f, g + e, y) - nll_raw(&d, f, g - e, y)) / (2.0 * e);
            let c = nll_raw(&d, f, g, y);
            let fd_ff = (nll_raw(&d, f + e, g, y) - 2.0 * c + nll_raw(&d, f - e, g, y)) / (e * e);
            let fd_gg = (nll_raw(&d, f, g + e, y) - 2.0 * c + nll_raw(&d, f, g - e, y)) / (e * e);
            let tol = |x: f64| 1e-4 * (1.0 + x.abs()) + 1e-6 * (g.exp());
            assert!((gr[[0, 0]] - fd_f).abs() < tol(fd_f), "∂f at {f},{g},{y}: {} vs {fd_f}", gr[[0, 0]]);
            assert!((gr[[0, 1]] - fd_g).abs() < tol(fd_g), "∂g at {f},{g},{y}: {} vs {fd_g}", gr[[0, 1]]);
            if g < 7.0 {
                // second differences lose precision at very large φ
                assert!((h[[0, 0]] - fd_ff).abs() < 1e-2 * (1.0 + fd_ff.abs()), "∂²f: {} vs {fd_ff}", h[[0, 0]]);
                assert!((h[[0, 1]] - fd_gg).abs() < 1e-2 * (1.0 + fd_gg.abs()), "∂²g: {} vs {fd_gg}", h[[0, 1]]);
            }
        }
    }

    /// E[observed] = Fisher, E[score] = 0, and the cross information is small
    /// for a concentrated Beta (orthogonality where it matters).
    #[test]
    fn fisher_is_the_expected_observed_and_coordinates_are_near_orthogonal() {
        let (f, g) = (0.4_f64, 4.0_f64); // µ ≈ 0.60, φ ≈ 55
        let (a, b) = Beta::mean_precision_to_ab(f, g);
        let n = 200_000;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let dist = RandBeta::new(a, b).unwrap();
        let y = Array1::from_iter((0..n).map(|_| dist.sample(&mut rng)));
        let pred = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { f } else { g });
        let (gr, h_obs) = mp(HessianMode::Observed).mean_precision_gradients(&pred.view(), &y.view());
        let (_, h_fis) = mp(HessianMode::Fisher).mean_precision_gradients(&pred.view(), &y.view());
        for j in 0..2 {
            let mean_obs = h_obs.column(j).mean().unwrap();
            let fis = h_fis[[0, j]];
            assert!((mean_obs / fis - 1.0).abs() < 0.02, "param {j}: E[obs] {mean_obs} vs Fisher {fis}");
            let score_var = gr.column(j).mapv(|v| v * v).mean().unwrap();
            assert!((score_var / fis - 1.0).abs() < 0.03, "param {j}: E[s²] {score_var} vs Fisher {fis}");
        }
        let cross = (&gr.column(0) * &gr.column(1)).mean().unwrap();
        let rho = cross / (h_fis[[0, 0]] * h_fis[[0, 1]]).sqrt();
        assert!(rho.abs() < 0.1, "mean/precision correlation {rho}");
        // the concentrations' (log a, log b) correlation at the same point is ≈ −0.97
        let ip = trigamma(a + b);
        let rho_ab = -a * b * ip / ((a * a * (trigamma(a) - ip)) * (b * b * (trigamma(b) - ip))).sqrt();
        assert!(rho_ab < -0.95, "{rho_ab}");
    }

    #[test]
    fn moments_start_and_output_convention() {
        let d = mp(HessianMode::Damped);
        let (a, b) = (6.0, 14.0);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let dist = RandBeta::new(a, b).unwrap();
        let y = Array1::from_iter((0..50_000).map(|_| dist.sample(&mut rng)));
        let sv = d.default_start_values(&ResponseData::Univariate(&y.view())).unwrap();
        let (a0, b0) = Beta::mean_precision_to_ab(sv[0], sv[1]);
        assert!((a0 / (a0 + b0) - 0.3).abs() < 0.005 && ((a0 + b0) / 20.0 - 1.0).abs() < 0.05, "{a0} {b0}");
        // transformed params are (concentration1, concentration0)
        let t = d.transform_params(&array![[sv[0], sv[1]]].view());
        assert!((t[[0, 0]] - a0).abs() < 1e-12 && (t[[0, 1]] - b0).abs() < 1e-12);
        // the Concentrations head has no closed-form default start
        assert!(Beta::default().with_parameterization(BetaParameterization::Concentrations).default_start_values(&ResponseData::Univariate(&y.view())).is_none());
    }

    #[test]
    fn mean_precision_damped_is_the_exp_nll_default() {
        let d = Beta::default();
        assert_eq!((d.parameterization(), d.hessian_mode()), (BetaParameterization::MeanPrecision, HessianMode::Damped));
        let crps = Beta::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);
        assert_eq!(crps.parameterization(), BetaParameterization::Concentrations);
        let sp = Beta::new(Stabilization::None, ResponseFn::Softplus, LossFn::Nll, false);
        assert_eq!((sp.parameterization(), sp.hessian_mode()), (BetaParameterization::Concentrations, HessianMode::Damped));
    }

    #[test]
    fn serde_defaults_are_the_legacy_head() {
        let mut v = serde_json::to_value(mp(HessianMode::Damped)).unwrap();
        v.as_object_mut().unwrap().remove("hessian_mode");
        v.as_object_mut().unwrap().remove("parameterization");
        let back: Beta = serde_json::from_value(v).unwrap();
        assert_eq!(back.hessian_mode(), HessianMode::Observed);
        assert_eq!(back.parameterization(), BetaParameterization::Concentrations);
    }

    /// Damped on the Concentrations head never goes below the Fisher value.
    #[test]
    fn damped_concentrations_curvature_is_positive() {
        let conc = || Beta::default().with_parameterization(BetaParameterization::Concentrations);
        let d = conc().with_hessian_mode(HessianMode::Damped);
        let pred = array![[2.0, 3.0], [-1.0, 0.5], [4.0, 4.0]];
        let y = array![0.999, 0.001, 0.5];
        let t = d.transform_params(&pred.view());
        let (_, h) = d.analytical_gradients(&pred.view(), &t.view(), &ResponseData::Univariate(&y.view())).unwrap();
        assert!(h.iter().all(|v| *v > 0.0), "{h:?}");
        let (_, h_obs) = conc()
            .with_hessian_mode(HessianMode::Observed)
            .analytical_gradients(&pred.view(), &t.view(), &ResponseData::Univariate(&y.view()))
            .unwrap();
        assert!(h_obs.iter().any(|v| *v < 0.0), "the observed curvature does go negative here: {h_obs:?}");
    }
}

#[cfg(test)]
mod pes_default_tests {
    use super::*;

    #[test]
    fn precision_is_watched_on_the_mean_precision_head_only() {
        assert_eq!(Beta::default().default_param_early_stopping_params(), Some(vec![1]));
        let conc = Beta::default().with_parameterization(BetaParameterization::Concentrations);
        assert_eq!(conc.default_param_early_stopping_params(), None);
        assert_eq!(super::super::Weibull::default().default_param_early_stopping_params(), Some(vec![1]));
    }
}
