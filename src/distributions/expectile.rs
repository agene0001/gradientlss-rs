//! Expectile distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use serde::{Deserialize, Serialize};

/// Expectile distribution for distributional regression.
///
/// Expectiles are quantile-like measures that minimize asymmetric least squares
/// rather than least absolute deviations. This implementation supports multiple
/// expectiles with optional crossing penalty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expectile {
    expectiles: Vec<f64>,
    penalize_crossing: bool,
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Expectile {
    pub fn new(
        expectiles: Vec<f64>,
        penalize_crossing: bool,
        stabilization: Stabilization,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        // Validate expectiles
        if expectiles.is_empty() {
            panic!("Expectiles list cannot be empty");
        }

        for &tau in &expectiles {
            if !(0.0 < tau && tau < 1.0) {
                panic!("Expectiles must be between 0 and 1");
            }
        }

        // Sort expectiles
        let mut expectiles = expectiles;
        expectiles.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Create parameters for each expectile
        let mut params = Vec::new();
        for &tau in &expectiles {
            params.push(DistributionParam::new(
                format!("expectile_{}", tau),
                ResponseFn::Identity,
            ));
        }

        Self {
            expectiles,
            penalize_crossing,
            params,
            stabilization,
            loss_fn,
            initialize,
        }
    }

    pub fn default() -> Self {
        Self::new(
            vec![0.1, 0.5, 0.9],
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        )
    }

    /// Compute the expectile loss function.
    fn expectile_loss(&self, params: &[f64], target: f64) -> f64 {
        let mut total_loss = 0.0;

        for (i, &tau) in self.expectiles.iter().enumerate() {
            let expectile_value = params[i];
            let weight = if target >= expectile_value {
                tau
            } else {
                1.0 - tau
            };
            let loss = weight * (target - expectile_value).powi(2);
            total_loss += loss;
        }

        // Add crossing penalty if enabled
        let mut penalty = 0.0;
        if self.penalize_crossing && self.expectiles.len() > 1 {
            for i in 1..self.expectiles.len() {
                if params[i] < params[i - 1] {
                    penalty = 1.0;
                    break;
                }
            }
        }

        total_loss = total_loss * (1.0 + penalty) / self.expectiles.len() as f64;

        total_loss
    }

    /// Compute the log probability (negative expectile loss).
    fn log_prob_expectile(&self, params: &[f64], target: f64) -> f64 {
        -self.expectile_loss(params, target)
    }

    /// Transform parameters to the distribution parameter space.
    fn transform_dist_params(&self, params: &[f64]) -> Vec<f64> {
        params.to_vec() // Expectiles use identity response function
    }
}

#[typetag::serde]
impl Distribution for Expectile {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Expectile"
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

        let transformed = self.transform_dist_params(params);
        self.log_prob_expectile(&transformed, target[0])
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
                panic!("Expectile is a univariate distribution")
            }
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_obs, n_samples));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Find the index of the expectile closest to 0.5
        let median_expectile_index = self
            .expectiles
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - 0.5).abs().partial_cmp(&((*b - 0.5).abs())).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0);

        for j in 0..n_obs {
            let expectile_value = params[[j, median_expectile_index]];
            let normal_dist = Normal::new(expectile_value, 1.0).unwrap(); // Assume variance of 1.0

            for i in 0..n_samples {
                result[[j, i]] = normal_dist.sample(&mut rng);
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
    fn test_expectile_creation() {
        let dist = Expectile::new(
            vec![0.1, 0.5, 0.9],
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        assert_eq!(dist.n_params(), 3);
        assert_eq!(dist.expectiles, vec![0.1, 0.5, 0.9]);
        assert!(!dist.penalize_crossing);
        assert!(dist.is_univariate());
    }

    #[test]
    fn test_expectile_log_prob() {
        let dist = Expectile::new(vec![0.5], false, Stabilization::None, LossFn::Nll, false);

        // Test with target equal to expectile value
        let params = vec![1.0]; // expectile_0.5 = 1.0
        let target = vec![1.0];

        let log_p = dist.log_prob(&params, &target);
        // Should be 0 loss for perfect prediction
        assert_relative_eq!(log_p, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectile_loss() {
        let dist = Expectile::new(
            vec![0.1, 0.9],
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        );

        // Test asymmetric loss
        let params = vec![1.0, 2.0]; // expectile_0.1 = 1.0, expectile_0.9 = 2.0
        let target = vec![1.5];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p < 0.0); // Should have some loss
    }

    #[test]
    fn test_expectile_crossing_penalty() {
        let dist = Expectile::new(
            vec![0.1, 0.5, 0.9],
            true,
            Stabilization::None,
            LossFn::Nll,
            false,
        );

        // Test with crossed expectiles (should have penalty)
        let params = vec![2.0, 1.5, 1.0]; // Decreasing order
        let target = vec![1.5];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p < 0.0); // Should have penalty
    }

    #[test]
    fn test_expectile_nll() {
        let dist = Expectile::new(
            vec![0.1, 0.5, 0.9],
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        let params = array![[1.0, 1.5, 2.0], [1.1, 1.6, 2.1]];
        let target = array![1.5, 1.6];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_expectile_sample() {
        let dist = Expectile::new(
            vec![0.1, 0.5, 0.9],
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        let params = array![[1.0, 1.5, 2.0], [1.1, 1.6, 2.1]];
        let samples = dist.sample(&params.view(), 1000, 123);

        // Should have shape (n_obs, n_samples) = (2, 1000)
        assert_eq!(samples.dim(), (2, 1000));

        // Check that samples for first observation are centered around the median expectile value
        let mean_0: f64 = samples.row(0).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_0, 1.5, epsilon = 0.1);

        // Check that samples for second observation are centered around the median expectile value
        let mean_1: f64 = samples.row(1).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_1, 1.6, epsilon = 0.1);
    }
}
