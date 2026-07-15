//! Mixture distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::constants::LOG_2PI;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Mixture distribution for distributional regression.
///
/// This implements a mixture of Gaussian distributions as a simplified version
/// of the full mixture model. The mixture is parameterized by component means,
/// variances, and mixing probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mixture {
    n_components: usize,
    temperature: f64,
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Mixture {
    pub fn new(
        n_components: usize,
        temperature: f64,
        stabilization: Stabilization,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        if n_components < 2 {
            panic!("Mixture requires at least 2 components");
        }
        if temperature <= 0.0 {
            panic!("Temperature must be greater than 0");
        }

        let mut params = Vec::new();

        // Component means (loc first, matching Python ordering)
        for i in 0..n_components {
            params.push(DistributionParam::new(
                format!("loc_{}", i + 1),
                ResponseFn::Identity,
            ));
        }

        // Component scales (using exp response function)
        for i in 0..n_components {
            params.push(DistributionParam::new(
                format!("scale_{}", i + 1),
                ResponseFn::Exp,
            ));
        }

        // Mixing probabilities (last, using gumbel_softmax via Identity;
        // gumbel_softmax is applied in transform_dist_params, matching Python)
        for i in 0..n_components {
            params.push(DistributionParam::new(
                format!("mix_prob_{}", i + 1),
                ResponseFn::Identity,
            ));
        }

        Self {
            n_components,
            temperature,
            params,
            stabilization,
            loss_fn,
            initialize,
        }
    }

    pub fn default() -> Self {
        Self::new(2, 1.0, Stabilization::None, LossFn::Nll, false)
    }

    /// Transform parameters to the distribution parameter space.
    /// Parameter ordering matches Python: [loc_1..M, scale_1..M, mix_prob_1..M]
    fn transform_dist_params(&self, params: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let m = self.n_components;

        // Extract location parameters (first M params)
        let locs: Vec<f64> = params[0..m].to_vec();

        // Scale parameters are already transformed by transform_params before reaching here
        let scales: Vec<f64> = params[m..2 * m].to_vec();

        // Extract mixing probability logits (last M params) and apply gumbel_softmax
        let mix_logits: Vec<f64> = params[2 * m..3 * m].to_vec();
        let mix_probs = Self::gumbel_softmax_from_logits(&mix_logits, self.temperature);

        (mix_probs, locs, scales)
    }

    /// The fixed Gumbel(0,1) noise vector: g_i = -log(-log(u_i)) with u drawn
    /// from ChaCha8(seed=123) — the deterministic analogue of Python's
    /// `torch.manual_seed(123)` inside gumbel_softmax_fn. The noise depends
    /// only on the component count, so hot paths draw it ONCE per pass instead
    /// of reseeding an RNG per row (values are identical either way).
    fn gumbel_noise(m: usize) -> Vec<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        (0..m)
            .map(|_| {
                let u: f64 = rng.random();
                -(-u.ln()).ln()
            })
            .collect()
    }

    /// Gumbel-softmax with precomputed noise, written into `out`
    /// (allocation-free): softmax((sanitize(logits) + noise) / temperature).
    /// NaN logits are replaced with the row's finite mean first — Python's
    /// gumbel_softmax_fn runs nan_to_num(predt) (replacement = nanmean), and
    /// without it one NaN logit poisons every downstream log_prob.
    fn gumbel_softmax_with_noise_into(
        logits: &[f64],
        noise: &[f64],
        temperature: f64,
        out: &mut [f64],
    ) {
        let (sum, count) = logits.iter().fold((0.0, 0usize), |(s, c), &v| {
            if v.is_finite() {
                (s + v, c + 1)
            } else {
                (s, c)
            }
        });
        let nan_replacement = if count == 0 { 0.0 } else { sum / count as f64 };

        let mut max_val = f64::NEG_INFINITY;
        for ((o, &x), &g) in out.iter_mut().zip(logits).zip(noise) {
            let x = if x.is_finite() { x } else { nan_replacement };
            let perturbed = x + g;
            *o = perturbed;
            max_val = max_val.max(perturbed);
        }
        let mut sum_exp = 0.0;
        for o in out.iter_mut() {
            *o = ((*o - max_val) / temperature).exp();
            sum_exp += *o;
        }
        for o in out.iter_mut() {
            *o /= sum_exp;
        }
    }

    /// Apply Gumbel-Softmax to logits for differentiable categorical sampling.
    /// Matches Python's gumbel_softmax_fn which uses torch.manual_seed(123)
    /// and torch.nn.functional.gumbel_softmax on every call, making the Gumbel
    /// noise deterministic and reproducible.
    fn gumbel_softmax_from_logits(logits: &[f64], temperature: f64) -> Vec<f64> {
        let noise = Self::gumbel_noise(logits.len());
        let mut out = vec![0.0; logits.len()];
        Self::gumbel_softmax_with_noise_into(logits, &noise, temperature, &mut out);
        out
    }

    /// Apply softmax function with temperature.
    fn softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
        // Subtract max for numerical stability
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f64> = logits
            .iter()
            .map(|&x| ((x - max_logit) / temperature).exp())
            .collect();

        let sum_exp: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum_exp).collect()
    }

    /// Compute the log probability for mixture distribution.
    fn log_prob_mixture(
        &self,
        mix_probs: &[f64],
        locs: &[f64],
        scales: &[f64],
        target: f64,
    ) -> f64 {
        let mut total_log_prob = f64::NEG_INFINITY;

        for i in 0..self.n_components {
            let loc = locs[i];
            let scale = scales[i];
            let mix_prob = mix_probs[i];

            if scale <= 0.0 {
                return f64::NEG_INFINITY;
            }

            // Compute Gaussian log probability for this component
            let z = (target - loc) / scale;
            let log_prob_component = -0.5 * z * z - 0.5 * LOG_2PI - scale.ln();

            // Weight by mixing probability
            let weighted_log_prob = log_prob_component + mix_prob.ln();

            // Use log-sum-exp for numerical stability
            if i == 0 {
                total_log_prob = weighted_log_prob;
            } else {
                total_log_prob = Self::log_sum_exp(total_log_prob, weighted_log_prob);
            }
        }

        total_log_prob
    }

    /// Compute log(sum(exp(a), exp(b))) for numerical stability.
    fn log_sum_exp(a: f64, b: f64) -> f64 {
        if a > b {
            a + (b - a).exp().ln_1p()
        } else {
            b + (a - b).exp().ln_1p()
        }
    }
}

#[typetag::serde]
impl Distribution for Mixture {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Mixture"
    }

    fn is_univariate(&self) -> bool {
        true
    }

    fn n_params(&self) -> usize {
        self.params.len()
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

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        if target.len() != 1 {
            return f64::NEG_INFINITY;
        }

        let (mix_probs, locs, scales) = self.transform_dist_params(params);
        self.log_prob_mixture(&mix_probs, &locs, &scales, target[0])
    }

    /// Analytical gradients/Hessians of the NLL w.r.t. the RAW margins —
    /// exactly what PyTorch autograd computes for Python's Mixture under the
    /// default `hessian_mode="individual"` (diagonal second derivatives, no
    /// positivity floor). Replaces the numerical central-difference fallback,
    /// which cost 2·n_params `log_prob` evaluations (each allocating several
    /// Vecs) per sample per boosting round.
    ///
    /// Derivation, per observation (fixed gumbel noise g, temperature τ):
    ///   π = softmax((z + g)/τ),  φ_j = N(y; μ_j, σ_j),  S = Σ_j π_j φ_j,
    ///   responsibilities w_j = π_j φ_j / S,  L = ln S,  NLL = −L, and with
    ///   ζ_j = (y−μ_j)/σ_j,  A_j = (y−μ_j)/σ_j²,  B_j = (ζ_j²−1)/σ_j:
    ///
    ///   ∂L/∂μ_j  = w_j A_j          ∂²L/∂μ_j²  = w_j(1−w_j)A_j² − w_j/σ_j²
    ///   ∂L/∂σ_j  = w_j B_j          ∂²L/∂σ_j²  = w_j(1−w_j)B_j² + w_j(1−3ζ_j²)/σ_j²
    ///   ∂L/∂z_m  = (w_m − π_m)/τ    ∂²L/∂z_m²  = [w_m(1−w_m) − π_m(1−π_m)]/τ²
    ///
    /// Locs and logits use the identity response (no chain rule); scales chain
    /// through σ = e^s + ε with σ' = σ'' = e^s:
    ///   d(−L)/ds = −(∂L/∂σ)σ',  d²(−L)/ds² = −[(∂²L/∂σ²)σ'² + (∂L/∂σ)σ'].
    ///
    /// The softmax derivatives fall out of ∂π_j/∂z_m = π_j(δ_jm − π_m)/τ:
    /// ∂S/∂z_m = S(w_m − π_m)/τ and ∂w_m/∂z_m = w_m(1−w_m)/τ.
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

        let m = self.n_components;
        let n_params = 3 * m;
        let tau = self.temperature;
        let n_samples = predictions.nrows();

        // Fixed per-call gumbel noise (identical for every row — the accepted
        // per-row-reuse deviation, same values the numerical path sees).
        let noise = Self::gumbel_noise(m);

        // dσ/ds = e^s, recovered from the transformed value without an exp
        // (σ − ε); Mixture scales always use the Exp response.
        let mut sigma_prime = Array2::zeros((n_samples, m));
        for j in 0..m {
            let (d1, _d2) = ResponseFn::Exp.derivative_batches_from_transformed(
                &predictions.column(m + j),
                &transformed.column(m + j),
            );
            sigma_prime.column_mut(j).assign(&d1);
        }

        let mut gradients = Array2::zeros((n_samples, n_params));
        let mut hessians = Array2::zeros((n_samples, n_params));
        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");

        // Per-row fill. Scratch: (row copy, π buffer, log-weight buffer).
        // A NaN target/margin propagates to NaN outputs, which stabilization
        // replaces — identical to the numerical path's behavior.
        let fill = |scratch: &mut (Vec<f64>, Vec<f64>, Vec<f64>),
                    i: usize,
                    gc: &mut [f64],
                    hc: &mut [f64]| {
            let (row_buf, pi, lw) = scratch;
            for (k, &v) in transformed.row(i).iter().enumerate() {
                row_buf[k] = v;
            }
            let yi = y[i];

            Self::gumbel_softmax_with_noise_into(&row_buf[2 * m..3 * m], &noise, tau, pi);

            // Responsibilities in log space: lw_j = ln π_j + ln φ_j, shifted
            // by the max so w_j = exp(lw_j − a)/Σ never over/underflows.
            for j in 0..m {
                let sig = row_buf[m + j];
                let zeta = (yi - row_buf[j]) / sig;
                lw[j] = -0.5 * zeta * zeta - 0.5 * LOG_2PI - sig.ln() + pi[j].ln();
            }
            let a = lw.iter().fold(f64::NEG_INFINITY, |acc, &v| acc.max(v));
            let mut sum_w = 0.0;
            for v in lw.iter_mut() {
                *v = (*v - a).exp();
                sum_w += *v;
            }

            for j in 0..m {
                let w = lw[j] / sum_w;
                let sig = row_buf[m + j];
                let sig_sq = sig * sig;
                let diff = yi - row_buf[j];
                let zeta = diff / sig;
                let a_j = diff / sig_sq;
                gc[j] = -w * a_j;
                hc[j] = w / sig_sq - w * (1.0 - w) * a_j * a_j;

                let b_j = (zeta * zeta - 1.0) / sig;
                let sp = sigma_prime[[i, j]];
                gc[m + j] = -w * b_j * sp;
                hc[m + j] = -((w * (1.0 - w) * b_j * b_j + w * (1.0 - 3.0 * zeta * zeta) / sig_sq)
                    * sp
                    * sp
                    + w * b_j * sp);

                let p = pi[j];
                gc[2 * m + j] = (p - w) / tau;
                hc[2 * m + j] = (p * (1.0 - p) - w * (1.0 - w)) / (tau * tau);
            }
        };

        let make_scratch = || (vec![0.0; n_params], vec![0.0; m], vec![0.0; m]);

        // Threshold matches the other analytical paths (256).
        const PAR_MIN: usize = 256;
        if n_samples >= PAR_MIN {
            g_slice
                .par_chunks_mut(n_params)
                .zip(h_slice.par_chunks_mut(n_params))
                .enumerate()
                .for_each_init(make_scratch, |scratch, (i, (gc, hc))| {
                    fill(scratch, i, gc, hc)
                });
        } else {
            let mut scratch = make_scratch();
            for (i, (gc, hc)) in g_slice
                .chunks_mut(n_params)
                .zip(h_slice.chunks_mut(n_params))
                .enumerate()
            {
                fill(&mut scratch, i, gc, hc);
            }
        }

        Some((gradients, hessians))
    }

    /// User-facing predicted parameters: apply the gumbel-softmax to the
    /// mixing-logit block, matching Python's `predict_dist` (whose `param_dict`
    /// response for `mix_prob` is `gumbel_softmax_fn`). Without this, users get
    /// unbounded raw logits that neither sum to 1 nor look like the Python
    /// output. Internal consumers (`nll`/`log_prob`/`sample`) keep receiving
    /// logits and softmax them in `transform_dist_params` — do not feed the
    /// finalized parameters back into those.
    fn finalize_predicted_params(&self, mut params: Array2<f64>) -> Array2<f64> {
        let m = self.n_components;
        let mut logits = vec![0.0f64; m];
        for mut row in params.rows_mut() {
            for k in 0..m {
                logits[k] = row[2 * m + k];
            }
            let probs = Self::gumbel_softmax_from_logits(&logits, self.temperature);
            for (k, &p) in probs.iter().enumerate() {
                row[2 * m + k] = p;
            }
        }
        params
    }

    /// Component selection is a discrete draw — CRPS finite differences through
    /// `sample` are not meaningful (torch's MixtureSameFamily has no rsample).
    fn has_reparameterizable_sampler(&self) -> bool {
        false
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(arr) => {
                let mut total_nll = 0.0;
                let n_samples = params.nrows();
                let m = self.n_components;

                // Hoisted per-call state: the gumbel noise is fixed (seeded
                // per call in Python; identical for every row here — the
                // accepted per-row-reuse deviation), so draw it once instead
                // of reseeding an RNG per row, and reuse one probability
                // buffer instead of the 4 Vecs `transform_dist_params`
                // allocated per row. This runs every boosting round on both
                // the train and validation sets.
                let noise = Self::gumbel_noise(m);
                let mut probs_buf = vec![0.0f64; m];
                // Fallback buffer for non-contiguous rows
                let n_params = self.n_params();
                let mut params_buf = vec![0.0f64; n_params];

                for i in 0..n_samples {
                    let row = params.row(i);
                    let row_params: &[f64] = match row.as_slice() {
                        Some(s) => s,
                        None => {
                            for (k, &v) in row.iter().enumerate() {
                                params_buf[k] = v;
                            }
                            &params_buf[..n_params]
                        }
                    };

                    Self::gumbel_softmax_with_noise_into(
                        &row_params[2 * m..3 * m],
                        &noise,
                        self.temperature,
                        &mut probs_buf,
                    );
                    let log_prob = self.log_prob_mixture(
                        &probs_buf,
                        &row_params[0..m],
                        &row_params[m..2 * m],
                        arr[i],
                    );
                    // torch.nansum parity: a NaN log-prob contributes 0.
                    if !log_prob.is_nan() {
                        total_nll -= log_prob;
                    }
                }

                total_nll
            }
            ResponseData::Multivariate(_) => {
                panic!("Mixture is a univariate distribution")
            }
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let n_components = self.n_components;

        // For mixture, we return samples with shape (n_samples, n_obs)
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let row_params: Vec<f64> = params.row(j).to_vec();
            let (mix_probs, locs, scales) = self.transform_dist_params(&row_params);

            // Sample from the mixture distribution
            for s in 0..n_samples {
                // Exact categorical draw from the mixing probabilities,
                // matching Python's MixtureSameFamily.sample() (a plain
                // Categorical(probs) draw). Re-perturbing the already-softmaxed
                // probabilities with a second Gumbel + temperature softmax (as
                // this used to do) biases component selection toward uniform —
                // e.g. weights (0.9, 0.1) at tau=1 selected component 1 with
                // probability ~0.82 instead of 0.9.
                let rand_val: f64 = rng.random();
                let mut cum_prob = 0.0;
                // Fall back to the last component if floating-point rounding
                // leaves the cumulative sum a hair below rand_val.
                let mut component_idx = n_components - 1;
                for (i, &prob) in mix_probs.iter().enumerate() {
                    cum_prob += prob;
                    if rand_val <= cum_prob {
                        component_idx = i;
                        break;
                    }
                }

                // Sample from the selected component
                let loc = locs[component_idx];
                let scale = scales[component_idx];
                let normal_dist = Normal::new(loc, scale).unwrap();
                result[[s, j]] = normal_dist.sample(&mut rng);
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
    fn test_mixture_creation() {
        let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);
        // 2 locs + 2 scales + 2 mix_probs = 6 parameters
        assert_eq!(dist.n_params(), 6);
        assert_eq!(dist.n_components, 2);
        assert_relative_eq!(dist.temperature, 1.0);
        assert!(dist.is_univariate());
    }

    #[test]
    fn test_mixture_log_prob() {
        let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);

        // Test with equal mixing probabilities and similar components
        // Order: [loc_1, loc_2, scale_1, scale_2, mix_prob_1, mix_prob_2]
        // Params are in already-transformed space (scales already exp'd)
        let params = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]; // locs=[0,0], scales=[1,1], mix_probs=[0.5,0.5]
        let target = vec![0.0];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_mixture_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let softmax_result = Mixture::softmax(&logits, 1.0);

        let sum: f64 = softmax_result.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

        // Check that probabilities are in [0, 1]
        for prob in &softmax_result {
            assert!(*prob >= 0.0 && *prob <= 1.0);
        }
    }

    #[test]
    fn test_mixture_nll() {
        let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);
        // Order: [loc_1, loc_2, scale_1, scale_2, mix_prob_1, mix_prob_2]
        // Params are in already-transformed space (scales already exp'd)
        let params = array![
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        ];
        let target = array![0.0, 1.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        assert!(nll.is_finite());
    }

    /// The analytical gradients/Hessians must agree with the numerical
    /// central-difference path (the autograd-equivalent reference) across
    /// component counts and temperatures — this is the sweep that catches
    /// Hessian-only bugs the grad-only tests miss.
    #[test]
    fn test_mixture_analytical_matches_numerical() {
        for &(m, tau) in &[(2usize, 1.0f64), (3, 0.7), (4, 2.5)] {
            let dist = Mixture::new(m, tau, Stabilization::None, LossFn::Nll, false);
            let n_params = dist.n_params();
            let n = 40usize;

            // Deterministic raw margins in a moderate range and targets that
            // straddle the component means.
            let mut state: u64 = 0xfeed_beef ^ (m as u64) << 8;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f64) / (u32::MAX as f64)
            };
            let mut preds = Array2::zeros((n, n_params));
            let mut y = ndarray::Array1::zeros(n);
            for i in 0..n {
                for j in 0..n_params {
                    preds[[i, j]] = (next() - 0.5) * 3.0;
                }
                y[i] = (next() - 0.5) * 5.0;
            }

            let transformed = dist.transform_params(&preds.view());
            let yv = y.view();
            let target = ResponseData::Univariate(&yv);

            let (ga, ha) = dist
                .analytical_gradients(&preds.view(), &transformed.view(), &target)
                .expect("analytical gradients available for NLL");
            let (gn, hn) = dist
                .numerical_gradients_hessians(&preds.view(), &transformed.view(), &target)
                .unwrap();

            for i in 0..n {
                for j in 0..n_params {
                    let (a, nref) = (ga[[i, j]], gn[[i, j]]);
                    assert!(
                        (a - nref).abs() <= 1e-6 + 1e-4 * nref.abs(),
                        "grad mismatch m={m} tau={tau} [{i},{j}]: analytical={a}, numerical={nref}"
                    );
                    let (a, nref) = (ha[[i, j]], hn[[i, j]]);
                    assert!(
                        (a - nref).abs() <= 1e-4 + 1e-3 * nref.abs(),
                        "hess mismatch m={m} tau={tau} [{i},{j}]: analytical={a}, numerical={nref}"
                    );
                }
            }
        }
    }

    /// Component selection must be an exact categorical draw from the mixing
    /// probabilities. The old sampler layered a second Gumbel perturbation and
    /// temperature softmax on top of the already-softmaxed probabilities, which
    /// biased selection toward uniform (0.9/0.1 sampled at ~0.82/0.18).
    #[test]
    fn test_mixture_sample_component_frequencies() {
        let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);
        // Well-separated components so every sample is attributable:
        // loc_1=0, loc_2=100, scales=1 (pre-transformed), strong logit gap.
        let params = array![[0.0, 100.0, 1.0, 1.0, 3.0, 0.0]];
        let (mix_probs, _, _) = dist.transform_dist_params(&params.row(0).to_vec());

        let n = 20_000usize;
        let samples = dist.sample(&params.view(), n, 42);
        let frac_comp0 = samples.column(0).iter().filter(|&&v| v < 50.0).count() as f64 / n as f64;

        assert!(
            (frac_comp0 - mix_probs[0]).abs() < 0.02,
            "component 0 frequency {frac_comp0} should match mixing probability {}",
            mix_probs[0]
        );
    }

    /// `finalize_predicted_params` must softmax ONLY the mixing-logit block
    /// (rows sum to 1) and leave locs/scales untouched — matching Python's
    /// predict output where mix_prob columns are probabilities.
    #[test]
    fn test_mixture_finalize_predicted_params() {
        let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);
        let params = array![
            [0.0, 1.0, 1.0, 2.0, 2.0, -1.0],
            [3.0, -1.0, 0.5, 0.5, 0.0, 0.0]
        ];
        let out = dist.finalize_predicted_params(params.clone());

        for i in 0..2 {
            // locs/scales unchanged
            for j in 0..4 {
                assert_relative_eq!(out[[i, j]], params[[i, j]]);
            }
            // mixing block: valid probabilities summing to 1
            let p1 = out[[i, 4]];
            let p2 = out[[i, 5]];
            assert!(p1 > 0.0 && p1 < 1.0);
            assert!(p2 > 0.0 && p2 < 1.0);
            assert_relative_eq!(p1 + p2, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mixture_sample() {
        let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);
        // Order: [loc_1, loc_2, scale_1, scale_2, mix_prob_1, mix_prob_2]
        // Params are in already-transformed space (scales already exp'd)
        let params = array![
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        ];
        let samples = dist.sample(&params.view(), 1000, 123);

        // Should have shape (n_samples, n_obs) = (1000, 2)
        assert_eq!(samples.dim(), (1000, 2));

        // Check that samples are reasonable
        let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;

        assert!(mean_0.abs() < 10.0); // Should be around 0
        assert!(mean_1.abs() < 10.0); // Should be around 1
    }
}
