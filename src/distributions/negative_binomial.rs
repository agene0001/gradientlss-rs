//! NegativeBinomial distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Gamma as RandGamma, Poisson as RandPoisson};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Discrete, NegativeBinomial as StatrsNegativeBinomial};

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
            ResponseFn::Exp,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        )
    }

    /// Helper method for scalar log probability
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let total_count = params[0];
        let probs = params[1];

        if total_count <= 0.0 || probs <= 0.0 || probs >= 1.0 {
            return f64::NEG_INFINITY;
        }

        match StatrsNegativeBinomial::new(total_count, probs) {
            Ok(dist) => dist.ln_pmf(target as u64),
            Err(_) => f64::NEG_INFINITY,
        }
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

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_scalar(params, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                let mut total = 0.0;
                for (i, &y_val) in y.iter().enumerate() {
                    let p = vec![params[[i, 0]], params[[i, 1]]];
                    total -= self.log_prob_scalar(&p, y_val);
                }
                total
            }
            ResponseData::Multivariate(_) => {
                panic!("NegativeBinomial is a univariate distribution.")
            }
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let total_count = params[[j, 0]];
            let probs = params[[j, 1]];

            if total_count > 0.0 && probs > 0.0 && probs < 1.0 {
                // NegativeBinomial as Gamma-Poisson mixture
                // probs is success probability, so rate = (1-probs)/probs
                let rate = (1.0 - probs) / probs;
                if let Ok(gamma_dist) = RandGamma::new(total_count, rate) {
                    for i in 0..n_samples {
                        let lambda: f64 = gamma_dist.sample(&mut rng);
                        if let Ok(poisson_dist) = RandPoisson::new(lambda) {
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
        let expected = StatrsNegativeBinomial::new(5.0, 0.5).unwrap().ln_pmf(3);
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_negative_binomial_nll() {
        let dist = NegativeBinomial::default();
        let params = array![[5.0, 0.5], [5.0, 0.5]];
        let target = array![3.0, 3.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -StatrsNegativeBinomial::new(5.0, 0.5).unwrap().ln_pmf(3);
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}
