//! Zero-Adjusted LogNormal distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Bernoulli, Distribution as RandDistribution, LogNormal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

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
        response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("loc", response_fn),
            DistributionParam::new("scale", response_fn),
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
        Self::new(
            Stabilization::None,
            ResponseFn::Identity,
            LossFn::Nll,
            false,
        )
    }

    /// Transform parameters to the distribution parameter space.
    fn transform_dist_params(&self, params: &[f64]) -> (f64, f64, f64) {
        let loc = self.params[0].response_fn.apply_scalar(params[0]);
        let scale = self.params[1].response_fn.apply_scalar(params[1]);
        let gate = self.params[2].response_fn.apply_scalar(params[2]);

        (loc, scale, gate)
    }

    /// Compute the log probability for zero-adjusted LogNormal distribution.
    fn log_prob_zaln(&self, loc: f64, scale: f64, gate: f64, target: f64) -> f64 {
        // Handle zero case
        if target == 0.0 {
            return gate.ln(); // Probability of zero
        }

        // Handle positive continuous case (y > 0)
        if target <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Check that scale parameter is positive
        if scale <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Check that gate probability is valid
        if !(0.0 < gate && gate < 1.0) {
            return f64::NEG_INFINITY;
        }

        // Compute LogNormal log probability
        let log_target = target.ln();
        let log_prob_lognormal = -0.5 * ((log_target - loc) / scale).powi(2)
            - 0.5 * (2.0 * PI).ln()
            - scale.ln()
            - log_target;

        // Combine with gate probability (probability of non-zero)
        (1.0 - gate).ln() + log_prob_lognormal
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

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        if target.len() != 1 {
            return f64::NEG_INFINITY;
        }

        let (loc, scale, gate) = self.transform_dist_params(params);
        self.log_prob_zaln(loc, scale, gate, target[0])
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
            let row_params: Vec<f64> = params.row(j).to_vec();
            let (loc, scale, gate) = self.transform_dist_params(&row_params);

            // Ensure valid parameters
            let scale = scale.max(0.1);
            let gate = gate.clamp(0.01, 0.99);

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

        // Test with zero target
        let params = vec![0.0, 1.0, 0.0]; // loc=0, scale=exp(1), gate=sigmoid(0)=0.5
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

        // Test with positive target
        let params = vec![0.0, 1.0, 0.0]; // loc=0, scale=exp(1), gate=sigmoid(0)=0.5
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

        // Test with negative target
        let params = vec![0.0, 1.0, 0.0];
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
        let params = array![[0.0, 1.0, 0.0], [1.0, 0.5, 1.0]];
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
        let params = array![[0.0, 1.0, 0.0], [1.0, 0.5, 1.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        // Should have shape (n_samples, n_obs) = (1000, 2)
        assert_eq!(samples.dim(), (1000, 2));

        // Check that we have a mix of zeros and positive values
        let zero_count_0 = samples.column(0).iter().filter(|&&x| x == 0.0).count();
        let zero_count_1 = samples.column(1).iter().filter(|&&x| x == 0.0).count();

        // First observation should have ~50% zeros (gate=0.5)
        assert!(zero_count_0 > 300 && zero_count_0 < 700);
        // Second observation should have ~70% zeros (gate=sigmoid(1)≈0.73)
        assert!(zero_count_1 > 600 && zero_count_1 < 800);

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
