//! Zero-Adjusted LogNormal distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::constants::LOG_2PI;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Bernoulli, Distribution as RandDistribution, LogNormal};
use serde::{Deserialize, Serialize};

/// Zero-Adjusted LogNormal distribution for distributional regression.
///
/// The zero-adjusted LogNormal distribution allows zeros as values, combining
/// a Bernoulli distribution for the zero probability and a LogNormal distribution
/// for the positive continuous part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZALN {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl ZALN {
    pub fn new(
        stabilization: Stabilization,
        scale_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("loc", ResponseFn::Identity),
            DistributionParam::new("scale", scale_response_fn),
            DistributionParam::new("gate", ResponseFn::Sigmoid), // Gate probability (0, 1)
        ];

        Self {
            params,
            stabilization,
            loss_fn,
            initialize,
        }
    }

    pub fn default() -> Self {
        Self::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false)
    }

    /// Compute the log probability for zero-adjusted LogNormal distribution.
    /// Matches Python's ZeroInflatedDistribution.log_prob with epsilon clamping.
    fn log_prob_zaln(&self, loc: f64, scale: f64, gate: f64, target: f64) -> f64 {
        // Check that scale parameter is positive
        if scale <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Check that gate probability is valid
        if !(0.0 < gate && gate < 1.0) {
            return f64::NEG_INFINITY;
        }

        let is_zero = target == 0.0;

        // Handle negative values
        if target < 0.0 {
            return f64::NEG_INFINITY;
        }

        // Clamp value away from 0 for continuous distributions (matches Python's epsilon clamp)
        let clamped_target = if target <= 0.0 {
            crate::constants::ZERO_CLAMP_EPS
        } else {
            target
        };

        // Compute LogNormal log probability
        let log_target = clamped_target.ln();
        let log_prob_lognormal =
            -0.5 * ((log_target - loc) / scale).powi(2) - 0.5 * LOG_2PI - scale.ln() - log_target;

        let log_one_minus_gate = (1.0 - gate).ln();
        let log_prob = log_one_minus_gate + log_prob_lognormal;

        if is_zero {
            // log(gate + (1-gate) * base_pdf(epsilon))
            (gate + log_prob.exp()).ln()
        } else {
            log_prob
        }
    }
}

#[typetag::serde]
impl Distribution for ZALN {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "ZALN"
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

    /// Sampling uses a discrete zero-gate, which is not a
    /// smooth function of the parameters under a fixed seed — CRPS finite
    /// differences through it are meaningless (torch has no rsample here either).
    fn has_reparameterizable_sampler(&self) -> bool {
        false
    }

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        if target.len() != 1 {
            return f64::NEG_INFINITY;
        }

        // params are already transformed by the caller (numerical_gradients_hessians
        // applies response functions before calling log_prob)
        let loc = params[0];
        let scale = params[1];
        let gate = params[2];
        self.log_prob_zaln(loc, scale, gate, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(arr) => {
                crate::distributions::util::par_nansum(params.nrows(), |i| {
                    let row_vec: Vec<f64> = params.row(i).to_vec();
                    -self.log_prob(&row_vec, &[arr[i]])
                })
            }
            ResponseData::Multivariate(_) => {
                panic!("ZALN is a univariate distribution")
            }
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();

        // For ZALN, we return samples with shape (n_samples, n_obs)
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            // params are already transformed by the caller, use directly
            let loc = params[[j, 0]];
            let scale = params[[j, 1]].max(0.1);
            let gate = params[[j, 2]].clamp(0.01, 0.99);

            for s in 0..n_samples {
                // Sample from Bernoulli to decide if zero or LogNormal
                let bernoulli_dist = Bernoulli::new(gate).unwrap();
                let is_zero = bernoulli_dist.sample(&mut rng);

                if is_zero {
                    result[[s, j]] = 0.0;
                } else {
                    // Sample from LogNormal distribution
                    match LogNormal::new(loc, scale) {
                        Ok(lognormal_dist) => result[[s, j]] = lognormal_dist.sample(&mut rng),
                        Err(_) => result[[s, j]] = 0.0, // Fallback to zero if LogNormal creation fails
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
    use ndarray::array;

    #[test]
    fn test_zaln_creation() {
        let dist = ZALN::new(
            Stabilization::None,
            ResponseFn::Identity,
            LossFn::Nll,
            false,
        );
        assert_eq!(dist.n_params(), 3); // loc, scale, gate
        assert!(dist.is_univariate());
    }

    #[test]
    fn test_zaln_log_prob_zero() {
        let dist = ZALN::new(
            Stabilization::None,
            ResponseFn::Identity,
            LossFn::Nll,
            false,
        );

        // Test with zero target - params are already transformed
        let params = vec![0.0, 1.0, 0.5]; // loc=0, scale=1, gate=0.5
        let target = vec![0.0];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_zaln_log_prob_positive() {
        let dist = ZALN::new(
            Stabilization::None,
            ResponseFn::Identity,
            LossFn::Nll,
            false,
        );

        // Test with positive target - params are already transformed
        let params = vec![0.0, 1.0, 0.5]; // loc=0, scale=1, gate=0.5
        let target = vec![1.0];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_zaln_invalid_target() {
        let dist = ZALN::new(
            Stabilization::None,
            ResponseFn::Identity,
            LossFn::Nll,
            false,
        );

        // Test with negative target - params are already transformed
        let params = vec![0.0, 1.0, 0.5];
        let target = vec![-1.0];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p == f64::NEG_INFINITY);
    }

    #[test]
    fn test_zaln_nll() {
        let dist = ZALN::new(
            Stabilization::None,
            ResponseFn::Identity,
            LossFn::Nll,
            false,
        );
        // Already transformed params: loc, scale, gate
        let params = array![[0.0, 1.0, 0.5], [1.0, 0.5, 0.3]];
        let target = array![0.0, 1.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_zaln_sample() {
        let dist = ZALN::new(
            Stabilization::None,
            ResponseFn::Identity,
            LossFn::Nll,
            false,
        );
        // Params are in already-transformed space: gate uses Sigmoid, so pass sigmoid outputs directly
        // gate=0.5 -> ~50% zeros, gate=0.731 -> ~73% zeros
        let params = array![[0.0, 1.0, 0.5], [1.0, 0.5, 0.731]];
        let samples = dist.sample(&params.view(), 1000, 123);

        // Should have shape (n_samples, n_obs) = (1000, 2)
        assert_eq!(samples.dim(), (1000, 2));

        // Check that we have a mix of zeros and positive values
        let zero_count_0 = samples.column(0).iter().filter(|&&x| x == 0.0).count();
        let zero_count_1 = samples.column(1).iter().filter(|&&x| x == 0.0).count();

        // First observation should have ~50% zeros (gate=0.5)
        assert!(zero_count_0 > 300 && zero_count_0 < 700);
        // Second observation should have ~73% zeros (gate=0.731)
        assert!(zero_count_1 > 600 && zero_count_1 < 850);

        // Check that positive samples are reasonable
        let positive_samples: Vec<f64> = samples
            .column(0)
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x.clamp(0.0, 100.0))
            .collect();

        if !positive_samples.is_empty() {
            let mean_positive: f64 =
                positive_samples.iter().sum::<f64>() / positive_samples.len() as f64;
            assert!(mean_positive > 0.1 && mean_positive < 10.0);
        }
    }
}
