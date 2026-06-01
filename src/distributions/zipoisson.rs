//! Zero-Inflated Poisson distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Poisson as RandPoisson};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZIPoisson {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl ZIPoisson {
    pub fn new(
        stabilization: Stabilization,
        rate_response_fn: ResponseFn,
        gate_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("rate", rate_response_fn),
            DistributionParam::new("gate", gate_response_fn),
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

    /// Poisson ln_pmf inlined: -λ + k*ln(λ) - ln_gamma(k+1)
    fn poisson_ln_pmf(rate: f64, k: u64) -> f64 {
        -rate + (k as f64) * rate.ln() - ln_gamma(k as f64 + 1.0)
    }

    /// Helper method for scalar log probability (inlined formula)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let rate = params[0];
        let gate = params[1];

        if rate <= 0.0 || gate < 0.0 || gate > 1.0 {
            return f64::NEG_INFINITY;
        }

        if target == 0.0 {
            // pmf(0) = exp(-rate), so gate + (1-gate)*exp(-rate)
            (gate + (1.0 - gate) * (-rate).exp()).ln()
        } else {
            let k = target as u64;
            (1.0 - gate).ln() + Self::poisson_ln_pmf(rate, k)
        }
    }
}

#[typetag::serde]
impl Distribution for ZIPoisson {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "ZIPoisson"
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

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_scalar(params, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                let col0 = params.column(0);
                let col1 = params.column(1);
                crate::distributions::util::par_sum(y.len(), |i| {
                    -self.log_prob_scalar(&[col0[i], col1[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("ZIPoisson is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let rate = params[[j, 0]];
            let gate = params[[j, 1]];

            if rate > 0.0 && gate >= 0.0 && gate <= 1.0 {
                if let Ok(poisson_dist) = RandPoisson::new(rate) {
                    for i in 0..n_samples {
                        if rng.random_bool(gate) {
                            result[[i, j]] = 0.0;
                        } else {
                            result[[i, j]] = poisson_dist.sample(&mut rng) as f64;
                        }
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
    use approx::assert_relative_eq;

    #[test]
    fn test_zipoisson_creation() {
        let dist = ZIPoisson::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["rate", "gate"]);
        assert!(dist.is_discrete());
    }

    #[test]
    fn test_zipoisson_log_prob() {
        let dist = ZIPoisson::default();

        let log_p_zero = dist.log_prob_scalar(&[5.0, 0.1], 0.0);
        // pmf(0) = exp(-5), so gate + (1-gate)*exp(-5) = 0.1 + 0.9*exp(-5)
        let expected_zero = (0.1 + 0.9 * (-5.0_f64).exp()).ln();
        assert_relative_eq!(log_p_zero, expected_zero, epsilon = 1e-10);

        let log_p_non_zero = dist.log_prob_scalar(&[5.0, 0.1], 3.0);
        // ln(1-gate) + poisson_ln_pmf(5, 3) = ln(0.9) + (-5 + 3*ln(5) - ln_gamma(4))
        let expected_non_zero = 0.9_f64.ln() + (-5.0 + 3.0 * 5.0_f64.ln() - ln_gamma(4.0));
        assert_relative_eq!(log_p_non_zero, expected_non_zero, epsilon = 1e-10);
    }
}
