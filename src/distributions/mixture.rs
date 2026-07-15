//! Mixture distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::constants::LOG_2PI;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
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

    /// Apply Gumbel-Softmax to logits for differentiable categorical sampling.
    /// Matches Python's gumbel_softmax_fn which uses torch.manual_seed(123)
    /// and torch.nn.functional.gumbel_softmax on every call, making the Gumbel
    /// noise deterministic and reproducible.
    fn gumbel_softmax_from_logits(logits: &[f64], temperature: f64) -> Vec<f64> {
        // Python's gumbel_softmax_fn runs nan_to_num(predt) first (replacement
        // = nanmean); mirror that per row so a NaN logit doesn't turn the whole
        // probability vector — and every downstream log_prob — into NaN.
        let (sum, count) = logits.iter().fold((0.0, 0usize), |(s, c), &v| {
            if v.is_finite() {
                (s + v, c + 1)
            } else {
                (s, c)
            }
        });
        let nan_replacement = if count == 0 { 0.0 } else { sum / count as f64 };

        // Fixed seed matching Python's torch.manual_seed(123) in gumbel_softmax_fn
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        // Add Gumbel(0,1) noise to logits: g_i = -log(-log(u_i)) where u_i ~ Uniform(0,1)
        let perturbed: Vec<f64> = logits
            .iter()
            .map(|&x| {
                let x = if x.is_finite() { x } else { nan_replacement };
                let u: f64 = rng.random();
                x + (-(-u.ln()).ln())
            })
            .collect();

        Self::softmax(&perturbed, temperature)
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

                // Pre-allocate fallback buffer for non-contiguous rows
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
                    let target_val = arr[i];

                    let log_prob = self.log_prob(row_params, &[target_val]);
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
