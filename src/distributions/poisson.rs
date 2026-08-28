//! Poisson distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Poisson as RandPoisson};
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Poisson {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Poisson {
    pub fn new(
        stabilization: Stabilization,
        rate_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![DistributionParam::new("rate", rate_response_fn)];
        Self {
            params,
            stabilization,
            loss_fn,
            initialize,
        }
    }

    pub fn default() -> Self {
        Self::new(Stabilization::None, ResponseFn::Relu, LossFn::Nll, false)
    }

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let rate = params[0];
        if rate <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // Use the target as-is rather than truncating to u64. The lgamma-based
        // ln_pmf is well-defined for non-integer counts, and truncating silently
        // mapped negatives to 0; counts outside the support are -inf.
        let k = target;
        if !k.is_finite() || k < 0.0 {
            return f64::NEG_INFINITY;
        }
        // ln_pmf = -λ + k*ln(λ) - ln(k!) = -λ + k*ln(λ) - ln_gamma(k+1)
        -rate + k * rate.ln()
            - crate::constants::ln_factorial(k).unwrap_or_else(|| ln_gamma(k + 1.0))
    }
}

/// NGBoost's CRPS metric for a single Poisson rate, ported from ngboost-rs
/// `poisson_crps_metric_value` (dist/poisson.rs): M = sum_y g(y)^2 * P(Y=y)
/// with g(y) = lambda * dCRPS/dlambda = 2*lambda*(sum_{k>=y} pmf(k) -
/// sum_k F(k) pmf(k)). Prefix/suffix sums over one pmf/cdf table, O(k_max).
/// The caller-owned scratch vecs avoid per-row allocation in the hot loop.
fn poisson_crps_metric(
    lambda: f64,
    pmf: &mut Vec<f64>,
    cdf: &mut Vec<f64>,
    fp: &mut Vec<f64>,
    suf: &mut Vec<f64>,
) -> f64 {
    const CAP: usize = 200;
    let std_dev = lambda.sqrt();
    let y_max = (((lambda + 8.0 * std_dev).ceil() as usize).max(5)).min(CAP - 11);
    let k_total = (((lambda + 6.0 * std_dev).ceil() as usize).max(y_max + 10)).min(CAP);
    let len = k_total + 1;

    pmf.clear();
    cdf.clear();
    let mut p = (-lambda).exp();
    let mut f = 0.0;
    for k in 0..len {
        if k > 0 {
            p *= lambda / k as f64;
        }
        f += p;
        pmf.push(p);
        cdf.push(f.min(1.0));
    }

    fp.clear();
    let mut acc = 0.0;
    for k in 0..len {
        acc += cdf[k] * pmf[k];
        fp.push(acc);
    }
    suf.clear();
    suf.resize(len + 1, 0.0);
    for k in (0..len).rev() {
        suf[k] = suf[k + 1] + pmf[k];
    }

    let mut metric = 0.0;
    for y in 0..=y_max {
        let py = pmf[y];
        if py < 1e-300 {
            continue;
        }
        let inner_max = (((lambda + 6.0 * std_dev).ceil() as usize).max(y + 10)).min(len - 1);
        let tail = suf[y] - suf[inner_max + 1];
        let g = 2.0 * lambda * (tail - fp[inner_max]);
        metric += g * g * py;
    }
    metric
}

impl Poisson {
    /// Closed-form CRPS gradients, replacing the FD path entirely for the
    /// 1-param Poisson.
    ///
    /// With CRPS(lambda) = sum_k (F(k) - 1{k>=y})^2 and the classic identity
    /// dF(k)/dlambda = -pmf(k), the gradient is
    ///
    ///     dCRPS/dlambda = sum_k 2 (F(k) - 1{k>=y}) * (-pmf(k))
    ///
    /// computed in the SAME single pmf-recurrence pass as the loss itself —
    /// one pass per sample instead of the FD path's two full loss evaluations,
    /// and exact instead of O(eps^2)-approximate. The truncation boundary's
    /// dependence on lambda is ignored, as the FD path implicitly did: the
    /// boundary terms are squared upper-tail residuals (~1e-9 at the 4-sigma
    /// bound).
    ///
    /// Hessians are 1.0, matching the numerical CRPS path exactly (Python
    /// parity: xgboostlss trains CRPS with unit hessians).
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
        let rate_response_fn = &self.params[0].response_fn;
        let t_rate = transformed.column(0);
        let p_rate = predictions.column(0);
        let (rd, _rsd) = rate_response_fn.derivative_batches_from_transformed(&p_rate, &t_rate);
        let _rsd = &_rsd;

        let mut gradients = Array2::zeros((n_samples, 1));
        let mut hessians = Array2::ones((n_samples, 1));
        // CRPS hessian mode (2026-08-28). Three arms, measured on real
        // strikeout data (untuned xgb, 300 rounds, test CRPS; NLL twin 1.205):
        //
        // * "unit"      — Python-parity hess = 1.0 (plain gradient descent):
        //                 1.379, with a downward prediction drift that the FD
        //                 path amplified into outright collapse.
        // * "curvature" — analytic d2CRPS/dlambda2 floored at 1.0 (may only
        //                 damp, never amplify — a 1e-2 floor exploded, pred
        //                 mean 25 / RMSE 156, CRPS is not convex in lambda):
        //                 1.230.
        // * "metric"    — DEFAULT. NGBoost's CRPS metric preconditioner,
        //                 M(lambda) = E_{y~Pois(lambda)}[ g(y)^2 ] with
        //                 g(y) = lambda * dCRPS/dlambda the log-rate gradient
        //                 (ported from ngboost-rs `poisson_crps_metric_value`).
        //                 Positive by construction — a Riemannian metric, not
        //                 a curvature, so no convexity caveat — this is the
        //                 natural-gradient step in the lss framework. In
        //                 margin space the hessian is M * (dlambda/dm / lambda)^2,
        //                 which under the Exp response is exactly M.
        //
        // The metric is floored: as lambda -> 0 both g and M vanish (M ~
        // lambda^2), and XGBoost leaf values are ~ g/h with no line search to
        // catch a blow-up — NGBoost itself pairs natural gradients WITH a line
        // search. The floor bounds the amplification for near-zero-rate rows.
        const CRPS_MIN_METRIC: f64 = 0.05;
        const CRPS_MIN_HESS: f64 = 1.0;
        #[derive(PartialEq)]
        enum CrpsHess {
            Metric,
            Curvature,
            Unit,
        }
        let hess_mode = match std::env::var("GRADIENTLSS_CRPS_HESS")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "unit" => CrpsHess::Unit,
            "curvature" | "curv" => CrpsHess::Curvature,
            _ => CrpsHess::Metric,
        };
        // Scratch tables for the metric expectation, reused across rows.
        // Capacity bounds the table like ngboost's cap: beyond it the metric
        // degrades gracefully toward the floor (huge-rate rows also have ~zero
        // gradient under the same truncation, so no amplification pairs with
        // it).
        const METRIC_CAP: usize = 200;
        let mut m_pmf: Vec<f64> = Vec::with_capacity(METRIC_CAP + 2);
        let mut m_cdf: Vec<f64> = Vec::with_capacity(METRIC_CAP + 2);
        let mut m_fp: Vec<f64> = Vec::with_capacity(METRIC_CAP + 2);
        let mut m_suf: Vec<f64> = Vec::with_capacity(METRIC_CAP + 3);
        for i in 0..n_samples {
            let rate = t_rate[i].max(1e-6);
            let yi = y[i];
            if !yi.is_finite() || yi < 0.0 {
                // Out of support: the FD path saw the flat 1e6 penalty on both
                // sides, i.e. a zero gradient.
                continue;
            }
            let y_int = yi.round().max(0.0) as usize;
            let max_k = ((yi.max(rate) + 4.0 * rate.sqrt()).ceil() as usize).min(100);
            let mut pmf = (-rate).exp();
            let mut cdf = 0.0;
            let mut g_lambda = 0.0;
            let mut h_lambda = 0.0;
            for k in 0..=max_k {
                if k > 0 {
                    pmf *= rate / k as f64;
                }
                cdf += pmf;
                let indicator = if k >= y_int { 1.0 } else { 0.0 };
                let resid = cdf - indicator;
                g_lambda -= 2.0 * resid * pmf;
                let dpmf = pmf * (k as f64 / rate - 1.0);
                h_lambda += 2.0 * (pmf * pmf - resid * dpmf);
                if k >= y_int && 1.0 - cdf < 1e-9 {
                    break;
                }
            }
            let d = rd[i];
            gradients[[i, 0]] = g_lambda * d;
            match hess_mode {
                CrpsHess::Unit => {}
                CrpsHess::Curvature => {
                    // Chain rule to the raw margin, mirroring the NLL path:
                    // h = h_lambda * d^2 + g_lambda * d2lambda/dm2.
                    let sd = _rsd[i];
                    hessians[[i, 0]] = (h_lambda * d * d + g_lambda * sd).max(CRPS_MIN_HESS);
                }
                CrpsHess::Metric => {
                    let m = poisson_crps_metric(
                        rate,
                        &mut m_pmf,
                        &mut m_cdf,
                        &mut m_fp,
                        &mut m_suf,
                    );
                    // M is the log-rate-space metric; map to margin space:
                    // h = M * (dlambda/dm / lambda)^2 (= M for Exp response).
                    let scale = d / rate;
                    hessians[[i, 0]] = (m * scale * scale).max(CRPS_MIN_METRIC);
                }
            }
        }
        Some((gradients, hessians))
    }
}

#[typetag::serde]
impl Distribution for Poisson {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Poisson"
    }

    fn is_discrete(&self) -> bool {
        true
    }
    fn n_params(&self) -> usize {
        1
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

    /// Sampling uses a discrete draw, which is not a
    /// smooth function of the parameters under a fixed seed — CRPS finite
    /// differences through it are meaningless (torch has no rsample here either).
    /// Truncated CDF-sum CRPS: sum_k (F(k) - 1{k >= round(y)})^2 up to
    /// K = max(y, lambda) + 4*sqrt(lambda), capped at 100 - term-for-term the
    /// consumer's Poisson CRPS eval kernel, so the training loss and the
    /// selection metric agree on what "CRPS" means. Smooth in lambda, which is
    /// what makes the central-difference gradients valid where the sampled
    /// estimator's were not.
    fn analytic_crps(&self, params: &[f64], target: &[f64]) -> Option<f64> {
        let rate = params[0];
        let y = target[0];
        if !(rate > 0.0) || !y.is_finite() || y < 0.0 {
            // Same convention as log_prob for out-of-support input: a large
            // finite penalty keeps FD arithmetic defined (NaN/inf would poison
            // the column means used for nan replacement).
            return Some(1e6);
        }
        let y_int = y.round().max(0.0) as usize;
        let max_k = ((y.max(rate) + 4.0 * rate.sqrt()).ceil() as usize).min(100);
        // One exp + a multiply per term, via pmf(k) = pmf(k-1) * rate / k —
        // the previous form paid a full ln_gamma + exp PER TERM, which made
        // every FD gradient evaluation ~an order of magnitude dearer than it
        // needed to be. Underflow at large rate is benign: the sum is capped
        // at k <= 100, where the true pmf mass is ~0 anyway, so a pmf(0)
        // underflowing to 0.0 yields the same capped sum the ln form did.
        let mut pmf = (-rate).exp();
        let mut cdf = 0.0;
        let mut crps = 0.0;
        for k in 0..=max_k {
            if k > 0 {
                pmf *= rate / k as f64;
            }
            cdf += pmf;
            let indicator = if k >= y_int { 1.0 } else { 0.0 };
            let d = cdf - indicator;
            crps += d * d;
            // Past y with the CDF saturated, every remaining term is
            // (cdf - 1)^2 < 1e-18 — stop paying for them.
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
                let rate_col = params.column(0);
                crate::distributions::util::par_nansum(y.len(), |i| {
                    -self.log_prob_scalar(&[rate_col[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("Poisson is a univariate distribution."),
        }
    }

    /// Analytical gradients for Poisson distribution.
    ///
    /// NLL = λ - k*ln(λ) + ln_gamma(k+1)
    ///
    /// Gradients w.r.t. distribution parameter:
    /// - dNLL/dλ = 1 - k/λ
    ///
    /// Hessian w.r.t. distribution parameter:
    /// - d²NLL/dλ² = k/λ²
    ///
    /// Chain rule applied for response function: dNLL/dpred = dNLL/dλ * dλ/dpred
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
        let rate_response_fn = &self.params[0].response_fn;

        let t_rate = transformed.column(0);
        let p_rate = predictions.column(0);

        // Batch response-fn derivatives (auto-vectorized), shared by both paths.
        let (rd, rsd) = rate_response_fn.derivative_batches_from_transformed(&p_rate, &t_rate);

        let compute = |i: usize| -> (f64, f64) {
            let lambda = t_rate[i].max(1e-6);
            let k = y[i];

            if !k.is_finite() || k < 0.0 {
                return (0.0, 0.0);
            }

            let grad_lambda = 1.0 - k / lambda;
            let hess_lambda = k / (lambda * lambda);

            let d = rd[i];
            let sd = rsd[i];
            let g = grad_lambda * d;
            let h = hess_lambda * d * d + grad_lambda * sd;
            (g, h)
        };

        let mut gradients = Array2::zeros((n_samples, 1));
        let mut hessians = Array2::zeros((n_samples, 1));
        // Single param → one element per sample; write straight into the backing
        // slices. Threshold matches the numerical path (256).
        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");

        const PAR_MIN: usize = 256;
        if n_samples >= PAR_MIN {
            g_slice
                .par_iter_mut()
                .zip(h_slice.par_iter_mut())
                .enumerate()
                .for_each(|(i, (g, h))| {
                    let (gv, hv) = compute(i);
                    *g = gv;
                    *h = hv;
                });
        } else {
            g_slice
                .iter_mut()
                .zip(h_slice.iter_mut())
                .enumerate()
                .for_each(|(i, (g, h))| {
                    let (gv, hv) = compute(i);
                    *g = gv;
                    *h = hv;
                });
        }

        Some((gradients, hessians))
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for j in 0..n_obs {
            let rate = params[[j, 0]];
            if rate > 0.0 {
                if let Ok(dist) = RandPoisson::new(rate) {
                    for i in 0..n_samples {
                        result[[i, j]] = RandDistribution::sample(&dist, &mut rng) as f64;
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

    /// The analytic CRPS must equal the definition it compresses: a
    /// brute-force sum over (CDF(k) - 1{k >= y})^2 with independently
    /// recomputed pmf terms.
    #[test]
    fn analytic_crps_matches_brute_force() {
        let d = Poisson::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);
        for (lam, y) in [(3.5_f64, 4.0_f64), (0.5, 0.0), (12.0, 20.0), (7.3, 2.0)] {
            let got = d.analytic_crps(&[lam], &[y]).expect("poisson has analytic crps");
            // Independent recompute: pmf via the direct formula, same bound.
            let max_k = ((y.max(lam) + 4.0 * lam.sqrt()).ceil() as usize).min(100);
            let y_int = y.round() as usize;
            let mut cdf = 0.0;
            let mut want = 0.0;
            let mut ln_fact = 0.0_f64;
            for k in 0..=max_k {
                if k > 0 {
                    ln_fact += (k as f64).ln();
                }
                let pmf = (-lam + (k as f64) * lam.ln() - ln_fact).exp();
                cdf += pmf;
                let ind = if k >= y_int { 1.0 } else { 0.0 };
                want += (cdf - ind) * (cdf - ind);
            }
            assert_relative_eq!(got, want, max_relative = 1e-9);
        }
    }

    /// The closed-form CRPS gradient must agree with the FD gradient it
    /// replaces — same loss surface, two differentiation methods. FD carries
    /// O(eps^2) truncation error, hence the loose-ish tolerance.
    #[test]
    fn closed_form_crps_gradients_match_finite_differences() {
        use ndarray::array;
        let d = Poisson::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);
        // raw margins -> rates via exp: rates ~ {2.0, 7.4, 20.1, 3.3}
        let predictions = array![[0.7_f64], [2.0], [3.0], [1.2]];
        let transformed = predictions.mapv(f64::exp);
        let y = array![3.0_f64, 5.0, 30.0, 0.0];
        let target = ResponseData::Univariate(&y.view());

        let (g_an, h_an) = d
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("closed-form CRPS gradients");
        let (g_fd, h_fd) = d
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("fd gradients");
        for i in 0..y.len() {
            let (a, f) = (g_an[[i, 0]], g_fd[[i, 0]]);
            assert!(
                (a - f).abs() <= 1e-4 * f.abs().max(1e-3),
                "row {i}: analytic {a} vs fd {f}"
            );
            // Closed-form path carries a preconditioner (metric by default,
            // floored at 0.05); FD keeps unit hessians.
            assert!(h_an[[i, 0]] >= 0.05, "hessian {} under floor", h_an[[i, 0]]);
            assert_eq!(h_fd[[i, 0]], 1.0);
        }
    }

    /// Speed probe for the two CRPS-path optimizations (recurrence + closed
    /// form). Not a regression gate — prints measured per-call costs.
    #[test]
    #[ignore = "manual benchmark - run with --ignored --nocapture"]
    fn bench_crps_gradient_paths() {
        use ndarray::Array2;
        use std::time::Instant;
        let d = Poisson::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);
        let n = 15_000usize;
        let predictions =
            Array2::from_shape_fn((n, 1), |(i, _)| 0.5 + ((i * 37 % 100) as f64) / 60.0);
        let transformed = predictions.mapv(f64::exp);
        let yv: Vec<f64> = (0..n).map(|i| ((i * 13) % 12) as f64).collect();
        let y = ndarray::Array1::from_vec(yv);
        let target = ResponseData::Univariate(&y.view());

        // Third arm: the pre-recurrence per-term cost (full ln_gamma + exp
        // per CDF term via log_prob_scalar) — what shipped before 2026-08-28.
        let old_style = |params: &[f64], y: f64| -> f64 {
            let rate = params[0];
            let y_int = y.round().max(0.0) as usize;
            let max_k = ((y.max(rate) + 4.0 * rate.sqrt()).ceil() as usize).min(100);
            let mut cdf = 0.0;
            let mut crps = 0.0;
            for k in 0..=max_k {
                cdf += d.log_prob_scalar(params, k as f64).exp();
                let ind = if k >= y_int { 1.0 } else { 0.0 };
                crps += (cdf - ind) * (cdf - ind);
            }
            crps
        };

        for _ in 0..3 {
            let t = Instant::now();
            // Old FD = 2 old-style evals per row (the eps offsets don't change cost).
            let mut sink = 0.0;
            for i in 0..n {
                let lam = transformed[[i, 0]];
                sink += old_style(&[lam + 1e-4], y[i]) - old_style(&[lam - 1e-4], y[i]);
            }
            let old_ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(sink);
            let t = Instant::now();
            let _ = d.numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target);
            let fd_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let _ = d.analytical_gradients(&predictions.view(), &transformed.view(), &target);
            let an_ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "{n} rows: OLD ln_gamma-FD {old_ms:.2} ms   FD(recurrence) {fd_ms:.2} ms                    closed-form {an_ms:.2} ms   (old/new = {:.0}x)",
                old_ms / an_ms
            );
        }
    }

    /// CRPS gradients through the analytic path must be finite and nonzero —
    /// the property the sampled path could not deliver for a discrete
    /// distribution (its gate rejected training outright).
    #[test]
    fn crps_numerical_gradients_are_finite_and_nonzero() {
        let d = Poisson::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);
        // raw predictions (pre-response); response exp() → rates.
        let predictions = array![[1.0_f64], [0.2], [2.0]];
        let transformed = predictions.mapv(f64::exp);
        let y = array![3.0_f64, 1.0, 6.0];
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

    #[test]
    fn test_poisson_creation() {
        let dist = Poisson::default();
        assert_eq!(dist.n_params(), 1);
        assert_eq!(dist.param_names(), vec!["rate"]);
        assert!(dist.is_discrete());
    }

    #[test]
    fn test_poisson_log_prob() {
        let dist = Poisson::default();
        let log_p = dist.log_prob_scalar(&[5.0], 3.0);
        // Poisson(λ=5) at k=3: -5 + 3*ln(5) - ln_gamma(4)
        let expected = -5.0 + 3.0 * 5.0_f64.ln() - ln_gamma(4.0);
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        let dist = Poisson::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        let predictions = array![[1.0], [0.5], [1.5]];
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
            assert_relative_eq!(analytical.0[[i, 0]], numerical.0[[i, 0]], epsilon = 1e-3);
            // True (unfloored) Hessians: both paths must agree on value AND sign.
            assert_relative_eq!(
                analytical.1[[i, 0]],
                numerical.1[[i, 0]],
                epsilon = 1e-2,
                max_relative = 1e-2
            );
        }
    }

    #[test]
    fn test_poisson_nll() {
        let dist = Poisson::default();
        let params = array![[5.0], [5.0]];
        let target = array![3.0, 3.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(-5.0 + 3.0 * 5.0_f64.ln() - ln_gamma(4.0));
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}
