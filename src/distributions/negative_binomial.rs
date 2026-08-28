//! NegativeBinomial distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegativeBinomial {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl NegativeBinomial {
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
        }
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
    ///     dCRPS/dtheta = 2 [ sum_k F(k) Ftheta(k)  -  sum_{k>=y} Ftheta(k) ]
    ///                  = 2 [ A - S(y) ]
    ///
    /// so ONE set of prefix/suffix tables serves every y — both the observed
    /// gradient (y = y_obs) and the metric expectation over hypothetical y.
    /// The per-parameter pmf derivatives need no special functions:
    ///
    ///     dpmf(i)/dprobs = pmf(i) * (i/probs - r/(1-probs))
    ///     dpmf(i)/dr     = pmf(i) * (H_i + ln(1-probs)),
    ///         H_i = psi(r+i) - psi(r) = sum_{t<i} 1/(r+t)   (telescoped —
    ///         the digamma difference is a plain running sum).
    ///
    /// # The hessian
    ///
    /// NGBoost's CRPS metric, diagonal per parameter in MARGIN space:
    ///
    ///     M_jj = E_{y~NB(theta)} [ ( dCRPS/dtheta_j * dtheta_j/dm_j )^2 ]
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
        let (rd_r, _) = r_response_fn.derivative_batches_from_transformed(&p_r, &t_r);
        let (rd_p, _) = probs_response_fn.derivative_batches_from_transformed(&p_probs, &t_probs);

        const CRPS_MIN_METRIC: f64 = 0.05;
        const CAP: usize = 220;
        let unit_hess = std::env::var("GRADIENTLSS_CRPS_HESS")
            .map(|v| v.trim().eq_ignore_ascii_case("unit"))
            .unwrap_or(false);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::ones((n_samples, 2));
        let mut sc = NbCrpsScratch::default();

        for i in 0..n_samples {
            let r = t_r[i];
            let probs = t_probs[i];
            let yi = y[i];
            if !(r > 0.0) || !(probs > 0.0) || !(probs < 1.0) || !yi.is_finite() || yi < 0.0 {
                // Out of support: FD saw the flat 1e6 penalty — zero gradient,
                // unit hessian.
                continue;
            }
            let mean = r * probs / (1.0 - probs);
            let var = mean / (1.0 - probs);
            let sd = var.sqrt();
            let y_int = yi.round().max(0.0) as usize;
            // Table long enough for BOTH the observed-y gradient and the
            // metric expectation over hypothetical y (mean + 8 sigma).
            let y_hyp_max = (((mean + 8.0 * sd).ceil() as usize).max(5)).min(CAP - 11);
            let k_total = (((yi.max(mean) + 6.0 * sd).ceil() as usize).max(y_hyp_max + 10)).min(CAP);
            let len = k_total + 1;

            // ---- one pass: pmf, cdf, and the two dF/dtheta prefixes -------
            sc.pmf.clear();
            sc.cdf.clear();
            sc.fr.clear();
            sc.fp_.clear();
            let ln_1mp = (-probs).ln_1p(); // ln(1 - probs)
            let mut pmf = (r * ln_1mp).exp();
            let mut cdf = 0.0;
            let mut h_i = 0.0; // psi(r+i) - psi(r), telescoped
            let mut acc_r = 0.0;
            let mut acc_p = 0.0;
            for k in 0..len {
                if k > 0 {
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
            let yo = y_int.min(len - 1);
            let g_r = 2.0 * (a_r - sc.suf_r[yo]);
            let g_p = 2.0 * (a_p - sc.suf_p[yo]);
            gradients[[i, 0]] = g_r * rd_r[i];
            gradients[[i, 1]] = g_p * rd_p[i];

            // ---- metric hessians: M_jj = E_y[(g_j(y) * dtheta/dm)^2] ------
            if !unit_hess {
                let mut m_r = 0.0;
                let mut m_p = 0.0;
                for yh in 0..=y_hyp_max {
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
                hessians[[i, 0]] = (m_r * dr * dr).max(CRPS_MIN_METRIC);
                hessians[[i, 1]] = (m_p * dp * dp).max(CRPS_MIN_METRIC);
            }
        }
        Some((gradients, hessians))
    }
}

#[typetag::serde]
impl Distribution for NegativeBinomial {
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

    fn should_initialize(&self) -> bool {
        self.initialize
    }

    /// Sampling uses a discrete Gamma–Poisson draw, which is not a
    /// smooth function of the parameters under a fixed seed — CRPS finite
    /// differences through it are meaningless (torch has no rsample here either).
    /// Truncated CDF-sum CRPS, mirroring the consumer's NB eval kernel:
    /// K = max(y, mean) + 4*sqrt(var), capped at 150. Params are the torch
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
        let max_k = ((y.max(mean) + 4.0 * var.sqrt()).ceil() as usize).min(150);
        // pmf recurrence — one multiply per term instead of a full
        // ln_gamma + exp per term (see the Poisson counterpart; the FD
        // gradient path evaluates this 4x per sample per round for the two
        // params, so the per-term cost is the whole ballgame). pmf(0) =
        // (1 - probs)^r in the torch parameterization; pmf(k)/pmf(k-1) =
        // probs * (r + k - 1) / k. Underflow at extreme means is benign under
        // the k <= 150 cap, as in the Poisson case.
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
        );
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

    #[test]
    fn analytic_crps_matches_brute_force_nb() {
        use approx::assert_relative_eq;
        let d = NegativeBinomial::new(
            Stabilization::None,
            ResponseFn::Softplus,
            ResponseFn::Sigmoid,
            LossFn::Crps,
            false,
        );
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
        );
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
        );
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
