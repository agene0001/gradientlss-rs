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
