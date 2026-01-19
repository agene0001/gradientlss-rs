//! Mixture distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

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

        // Mixing probabilities (using softmax response function)
        for i in 0..n_components {
            params.push(DistributionParam::new(
                format!("mix_prob_{}", i + 1),
                ResponseFn::Identity, // Will apply softmax in transform
            ));
        }

        // Component means
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
    fn transform_dist_params(&self, params: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n_params = self.n_params();
        let mut mix_probs = Vec::with_capacity(self.n_components);
        let mut locs = Vec::with_capacity(self.n_components);
        let mut scales = Vec::with_capacity(self.n_components);

        // Extract and transform mixing probabilities (apply softmax)
        let start_mix = 0;
        let end_mix = self.n_components;
        let mix_params: Vec<f64> = params[start_mix..end_mix].to_vec();

        // Apply softmax to get valid probabilities
        let mix_probs_softmax = Self::softmax(&mix_params, self.temperature);
        for prob in mix_probs_softmax {
            mix_probs.push(prob);
        }

        // Extract location parameters
        let start_loc = end_mix;
        let end_loc = start_loc + self.n_components;
        for i in start_loc..end_loc {
            locs.push(params[i]);
        }

        // Extract and transform scale parameters
        for i in end_loc..n_params {
            let param = &self.params[i];
            scales.push(param.response_fn.apply_scalar(params[i]));
        }

        (mix_probs, locs, scales)
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
            let log_prob_component = -0.5 * z * z - 0.5 * (2.0 * PI).ln() - scale.ln();

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

    /// Sample from Gumbel(0, 1) distribution.
    fn sample_gumbel(rng: &mut ChaCha8Rng) -> f64 {
        let uniform: f64 = rng.random();
        -(-uniform.ln()).ln()
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

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(arr) => {
                let mut total_nll = 0.0;
                let n_samples = params.nrows();

                for i in 0..n_samples {
                    let row_params: Vec<f64> = params.row(i).to_vec();
                    let target_val = arr[i];

                    let log_prob = self.log_prob(&row_params, &[target_val]);
                    total_nll -= log_prob;
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
                // Sample component index using Gumbel-Softmax trick
                let gumbel_samples: Vec<f64> = (0..n_components)
                    .map(|_| Self::sample_gumbel(&mut rng))
                    .collect();

                let logits_with_gumbel: Vec<f64> = mix_probs
                    .iter()
                    .zip(gumbel_samples.iter())
                    .map(|(&prob, &gumbel)| prob.ln() + gumbel)
                    .collect();

                let component_probs = Self::softmax(&logits_with_gumbel, self.temperature);

                // Sample component index
                let mut cum_prob = 0.0;
                let mut component_idx = 0;
                let rand_val: f64 = rng.random();

                for (i, &prob) in component_probs.iter().enumerate() {
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
        // 2 mix_probs + 2 locs + 2 scales = 6 parameters
        assert_eq!(dist.n_params(), 6);
        assert_eq!(dist.n_components, 2);
        assert_relative_eq!(dist.temperature, 1.0);
        assert!(dist.is_univariate());
    }

    #[test]
    fn test_mixture_log_prob() {
        let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);

        // Test with equal mixing probabilities and similar components
        let params = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // mix_probs=[0.5,0.5], locs=[0,0], scales=[1,1]
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
        let params = array![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ];
        let target = array![0.0, 1.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_mixture_sample() {
        let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);
        let params = array![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
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
