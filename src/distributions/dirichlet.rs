//! Dirichlet distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Gamma as RandGamma};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

/// Dirichlet distribution for distributional regression.
///
/// The Dirichlet distribution is used for modeling compositional data (proportions
/// that sum to 1). It extends the beta distribution to multiple dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dirichlet {
    n_targets: usize,
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Dirichlet {
    pub fn new(
        n_targets: usize,
        stabilization: Stabilization,
        response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        if n_targets < 2 {
            panic!("Dirichlet requires at least 2 targets");
        }

        let mut params = Vec::new();

        // Concentration parameters (alpha) - must be positive
        for i in 0..n_targets {
            params.push(DistributionParam::new(
                format!("concentration_{}", i + 1),
                response_fn,
            ));
        }

        Self {
            n_targets,
            params,
            stabilization,
            loss_fn,
            initialize,
        }
    }

    pub fn default() -> Self {
        Self::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false)
    }

    /// Transform parameters to the distribution parameter space.
    fn transform_dist_params(&self, params: &[f64]) -> Vec<f64> {
        params
            .iter()
            .zip(self.params.iter())
            .map(|(&p, param)| param.response_fn.apply_scalar(p))
            .collect()
    }

    /// Compute the log probability for Dirichlet distribution.
    fn log_prob_dirichlet(&self, concentration: &[f64], target: &[f64]) -> f64 {
        // Check that target sums to 1 (with some tolerance)
        let target_sum: f64 = target.iter().sum();
        if !(target_sum > 0.99 && target_sum < 1.01) {
            return f64::NEG_INFINITY;
        }

        // Check that all target values are in [0, 1]
        for &t in target {
            if t < 0.0 || t > 1.0 {
                return f64::NEG_INFINITY;
            }
        }

        // Check that all concentration parameters are positive
        for &alpha in concentration {
            if alpha <= 0.0 {
                return f64::NEG_INFINITY;
            }
        }

        // Compute log probability
        let log_beta_normalization = Self::log_beta(concentration);
        let log_prod = concentration
            .iter()
            .zip(target.iter())
            .map(|(&alpha, &t)| (alpha - 1.0) * t.ln())
            .sum::<f64>();

        log_prod - log_beta_normalization
    }

    /// Compute the log of the beta function (normalization constant).
    fn log_beta(concentration: &[f64]) -> f64 {
        let sum_alpha: f64 = concentration.iter().sum();
        let sum_log_gamma_alpha: f64 = concentration.iter().map(|&a| Self::log_gamma(a)).sum();
        let log_gamma_sum_alpha = Self::log_gamma(sum_alpha);

        sum_log_gamma_alpha - log_gamma_sum_alpha
    }

    /// Approximate log gamma function (Lanczos approximation).
    fn log_gamma(x: f64) -> f64 {
        ln_gamma(x)
    }
}

#[typetag::serde]
impl Distribution for Dirichlet {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Dirichlet"
    }

    fn is_univariate(&self) -> bool {
        false
    }

    fn n_params(&self) -> usize {
        self.params.len()
    }

    fn n_targets(&self) -> usize {
        self.n_targets
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
        if target.len() != self.n_targets {
            return f64::NEG_INFINITY;
        }

        let concentration = self.transform_dist_params(params);
        self.log_prob_dirichlet(&concentration, target)
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(_) => {
                panic!("Dirichlet requires multivariate targets")
            }
            ResponseData::Multivariate(arr) => {
                let mut total_nll = 0.0;
                let n_samples = params.nrows();

                for i in 0..n_samples {
                    let row_params: Vec<f64> = params.row(i).to_vec();
                    let target_row: Vec<f64> = arr.row(i).to_vec();

                    let log_prob = self.log_prob(&row_params, &target_row);
                    total_nll -= log_prob;
                }

                total_nll
            }
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let n_targets = self.n_targets;

        // For Dirichlet, we return samples with shape (n_obs * n_targets, n_samples)
        let mut result = Array2::zeros((n_obs * n_targets, n_samples));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let row_params: Vec<f64> = params.row(j).to_vec();
            let concentration = self.transform_dist_params(&row_params);

            // Ensure all concentration parameters are positive
            let concentration: Vec<f64> = concentration
                .iter()
                .map(|&c| c.max(1e-6)) // Ensure positive
                .collect();

            // Sample from Dirichlet using Gamma distributions
            // Dirichlet(alpha) can be sampled by: X_i ~ Gamma(alpha_i, 1), then normalize
            for s in 0..n_samples {
                let mut gamma_samples = Vec::with_capacity(n_targets);
                let mut sum = 0.0;
                for &alpha in &concentration {
                    if let Ok(gamma_dist) = RandGamma::new(alpha, 1.0) {
                        let g: f64 = gamma_dist.sample(&mut rng);
                        gamma_samples.push(g);
                        sum += g;
                    } else {
                        gamma_samples.push(1.0);
                        sum += 1.0;
                    }
                }
                // Normalize to get Dirichlet sample
                for t in 0..n_targets {
                    result[[j * n_targets + t, s]] = gamma_samples[t] / sum;
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
    fn test_dirichlet_creation() {
        let dist = Dirichlet::new(3, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        assert_eq!(dist.n_params(), 3); // 3 concentration parameters
        assert_eq!(dist.n_targets(), 3);
        assert!(!dist.is_univariate());
        assert!(!dist.should_initialize());
    }

    #[test]
    fn test_dirichlet_log_prob() {
        let dist = Dirichlet::new(3, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // Test with valid concentration parameters and target that sums to 1
        let params = vec![1.0, 1.0, 1.0]; // concentration = [exp(1), exp(1), exp(1)]
        let target = vec![0.3, 0.4, 0.3]; // Sums to 1.0

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_dirichlet_invalid_target() {
        let dist = Dirichlet::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // Test with target that doesn't sum to 1
        let params = vec![0.0, 0.0]; // concentration = [1, 1]
        let target = vec![0.6, 0.3]; // Sums to 0.9, not 1.0

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p == f64::NEG_INFINITY);
    }

    #[test]
    fn test_dirichlet_nll() {
        let dist = Dirichlet::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        let params = array![[0.0, 0.0], [1.0, 0.0]];
        let target = array![[0.5, 0.5], [0.7, 0.3]];
        let target_response = ResponseData::Multivariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_dirichlet_sample() {
        let dist = Dirichlet::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        let params = array![[0.0, 0.0], [1.0, 0.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        // Should have shape (n_obs * n_targets, n_samples) = (2*2, 1000) = (4, 1000)
        assert_eq!(samples.dim(), (4, 1000));

        // Check that samples for the first observation sum to approximately 1
        let sum_0: f64 = samples
            .row(0)
            .iter()
            .zip(samples.row(1).iter())
            .map(|(&a, &b)| a + b)
            .sum::<f64>()
            / 1000.0;
        assert_relative_eq!(sum_0, 1.0, epsilon = 0.05);

        // Check that samples for the second observation sum to approximately 1
        let sum_1: f64 = samples
            .row(2)
            .iter()
            .zip(samples.row(3).iter())
            .map(|(&a, &b)| a + b)
            .sum::<f64>()
            / 1000.0;
        assert_relative_eq!(sum_1, 1.0, epsilon = 0.05);
    }
}
