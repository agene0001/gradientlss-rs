//! Zero-Adjusted Beta distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Bernoulli, Beta, Distribution as RandDistribution};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

/// Zero-Adjusted Beta distribution for distributional regression.
///
/// The zero-adjusted Beta distribution allows zeros as values, combining
/// a Bernoulli distribution for the zero probability and a Beta distribution
/// for the continuous part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZABeta {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl ZABeta {
    pub fn new(
        stabilization: Stabilization,
        response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("concentration1", response_fn),
            DistributionParam::new("concentration0", response_fn),
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

    /// Compute the log probability for zero-adjusted Beta distribution.
    /// Matches Python's ZeroInflatedDistribution.log_prob with epsilon clamping.
    fn log_prob_zabeta(
        &self,
        concentration1: f64,
        concentration0: f64,
        gate: f64,
        target: f64,
    ) -> f64 {
        // Check that concentration parameters are positive
        if concentration1 <= 0.0 || concentration0 <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Check that gate probability is valid
        if !(0.0 < gate && gate < 1.0) {
            return f64::NEG_INFINITY;
        }

        let is_zero = target == 0.0;

        // Handle out-of-range values
        if target < 0.0 || target > 1.0 {
            return f64::NEG_INFINITY;
        }

        // Clamp value away from boundaries for continuous distributions (matches Python's epsilon clamp)
        let clamped_target = if target <= 0.0 {
            f64::EPSILON
        } else if target >= 1.0 {
            1.0 - f64::EPSILON
        } else {
            target
        };

        // Compute Beta log probability using statrs ln_gamma
        let log_beta = ln_gamma(concentration1) + ln_gamma(concentration0)
            - ln_gamma(concentration1 + concentration0);
        let log_prob_beta = (concentration1 - 1.0) * clamped_target.ln()
            + (concentration0 - 1.0) * (1.0 - clamped_target).ln()
            - log_beta;

        let log_one_minus_gate = (1.0 - gate).ln();
        let log_prob = log_one_minus_gate + log_prob_beta;

        if is_zero {
            // log(gate + (1-gate) * base_pdf(epsilon))
            (gate + log_prob.exp()).ln()
        } else {
            log_prob
        }
    }
}

#[typetag::serde]
impl Distribution for ZABeta {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "ZABeta"
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

        // params are already transformed by the caller (numerical_gradients_hessians
        // applies response functions before calling log_prob)
        let concentration1 = params[0];
        let concentration0 = params[1];
        let gate = params[2];
        self.log_prob_zabeta(concentration1, concentration0, gate, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(arr) => {
                let mut total_nll = 0.0;
                let n_samples = params.nrows();

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
                    total_nll -= log_prob;
                }

                total_nll
            }
            ResponseData::Multivariate(_) => {
                panic!("ZABeta is a univariate distribution")
            }
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();

        // For ZABeta, we return samples with shape (n_samples, n_obs)
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            // params are already transformed by the caller, use directly
            let concentration1 = params[[j, 0]].max(0.1);
            let concentration0 = params[[j, 1]].max(0.1);
            let gate = params[[j, 2]].clamp(0.01, 0.99);

            for s in 0..n_samples {
                // Sample from Bernoulli to decide if zero or Beta
                let bernoulli_dist = Bernoulli::new(gate).unwrap();
                let is_zero = bernoulli_dist.sample(&mut rng);

                if is_zero {
                    result[[s, j]] = 0.0;
                } else {
                    // Sample from Beta distribution
                    match Beta::new(concentration1, concentration0) {
                        Ok(beta_dist) => result[[s, j]] = beta_dist.sample(&mut rng),
                        Err(_) => result[[s, j]] = 0.0, // Fallback to zero if Beta creation fails
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
    fn test_zabeta_creation() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        assert_eq!(dist.n_params(), 3); // concentration1, concentration0, gate
        assert!(dist.is_univariate());
    }

    #[test]
    fn test_zabeta_log_prob_zero() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // Test with zero target - params are already transformed
        let params = vec![2.718281828, 2.718281828, 0.5]; // concentration1=e, concentration0=e, gate=0.5
        let target = vec![0.0];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_zabeta_log_prob_continuous() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // Test with continuous target - params are already transformed
        let params = vec![2.718281828, 2.718281828, 0.5]; // concentration1=e, concentration0=e, gate=0.5
        let target = vec![0.5];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_zabeta_invalid_target() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // Test with invalid target (> 1) - params are already transformed
        let params = vec![2.718281828, 2.718281828, 0.5];
        let target = vec![1.5];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p == f64::NEG_INFINITY);
    }

    #[test]
    fn test_zabeta_nll() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        // Already transformed params: concentration1, concentration0, gate
        let params = array![[2.0, 2.0, 0.5], [1.0, 1.0, 0.7]];
        let target = array![0.0, 0.5];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_zabeta_sample() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        // Params are in already-transformed space: gate uses Sigmoid, so pass sigmoid outputs directly
        // gate=0.5 -> ~50% zeros, gate=0.731 -> ~73% zeros
        let params = array![[1.0, 1.0, 0.5], [1.0, 1.0, 0.731]];
        let samples = dist.sample(&params.view(), 1000, 123);

        // Should have shape (n_samples, n_obs) = (1000, 2)
        assert_eq!(samples.dim(), (1000, 2));

        // Check that we have a mix of zeros and continuous values
        let zero_count_0 = samples.column(0).iter().filter(|&&x| x == 0.0).count();
        let zero_count_1 = samples.column(1).iter().filter(|&&x| x == 0.0).count();

        // First observation should have ~50% zeros (gate=0.5)
        assert!(zero_count_0 > 300 && zero_count_0 < 700);
        // Second observation should have ~73% zeros (gate=0.731)
        assert!(zero_count_1 > 600 && zero_count_1 < 850);
    }
}
