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

    /// Compute the expectile loss function for a single observation.
    /// The crossing penalty is NOT applied here; it is computed at the batch level
    /// in the `nll` method, matching Python's batch-mean penalty:
    ///   penalty = torch.mean((~torch.all(torch.diff(predt, dim=1) > 0, dim=1)).float())
    fn expectile_loss(&self, params: &[f64], target: f64, penalty: f64) -> f64 {
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

        total_loss = total_loss * (1.0 + penalty) / self.expectiles.len() as f64;

        total_loss
    }

    /// Compute per-observation crossing indicator.
    /// Returns 1.0 if any consecutive pair of expectiles crosses (non-increasing), 0.0 otherwise.
    fn has_crossing(&self, params: &[f64]) -> f64 {
        if self.expectiles.len() <= 1 {
            return 0.0;
        }
        let crossed = (1..self.expectiles.len()).any(|i| params[i] <= params[i - 1]);
        if crossed { 1.0 } else { 0.0 }
    }

    /// Compute the log probability (negative expectile loss).
    /// Note: When called per-observation (e.g. from log_prob), penalty is 0.0.
    /// The actual batch penalty is computed in `nll`.
    fn log_prob_expectile(&self, params: &[f64], target: f64, penalty: f64) -> f64 {
        -self.expectile_loss(params, target, penalty)
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
        // Per-observation log_prob does not include crossing penalty (penalty=0.0).
        // Crossing penalty is computed at the batch level in nll().
        self.log_prob_expectile(&transformed, target[0], 0.0)
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(arr) => {
                let n_samples = params.nrows();

                // Compute batch-level crossing penalty matching Python:
                //   penalty = torch.mean((~torch.all(torch.diff(predt, dim=1) > 0, dim=1)).float())
                let penalty = if self.penalize_crossing && self.expectiles.len() > 1 {
                    let crossing_count: f64 = crate::distributions::util::par_sum(n_samples, |i| {
                        self.has_crossing(&params.row(i).to_vec())
                    });
                    crossing_count / n_samples as f64
                } else {
                    0.0
                };

                crate::distributions::util::par_nansum(n_samples, |i| {
                    let row_params: Vec<f64> = params.row(i).to_vec();
                    let transformed = self.transform_dist_params(&row_params);
                    self.expectile_loss(&transformed, arr[i], penalty)
                })
            }
            ResponseData::Multivariate(_) => {
                panic!("Expectile is a univariate distribution")
            }
        }
    }

    /// Analytical gradients for the expectile loss, matching PyTorch autograd of
    /// Python's `Expectile_Torch.log_prob`:
    ///
    ///   loss = [Σ_p nansum_n w_{n,p} (y_n - e_{n,p})²] · (1 + penalty) / M
    ///
    /// with w = τ if y - e ≥ 0 else 1-τ. The crossing penalty is an indicator,
    /// so autograd treats it as constant — but it still *scales* the loss, and
    /// therefore every gradient and Hessian, by (1 + penalty). The previous
    /// numerical fallback differentiated per-observation `log_prob`, which
    /// hardcodes penalty = 0 and silently dropped that factor whenever
    /// `penalize_crossing` was set.
    ///
    ///   dL/de  = -2 w (y - e) (1 + penalty) / M
    ///   d²L/de² =  2 w (1 + penalty) / M    (w is piecewise-constant)
    ///
    /// Expectile parameters always use the identity response, so no chain rule.
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

        let n_samples = predictions.nrows();
        let m = self.expectiles.len();

        // Batch-level crossing penalty, matching Python:
        //   penalty = mean((~all(diff(predt, dim=1) > 0, dim=1)).float())
        let penalty = if self.penalize_crossing && m > 1 {
            let crossing_count: f64 = crate::distributions::util::par_sum(n_samples, |i| {
                let row = transformed.row(i);
                let crossed = (1..m).any(|p| row[p] <= row[p - 1]);
                if crossed { 1.0 } else { 0.0 }
            });
            crossing_count / n_samples as f64
        } else {
            0.0
        };
        let scale = (1.0 + penalty) / m as f64;

        let mut gradients = Array2::zeros((n_samples, m));
        let mut hessians = Array2::zeros((n_samples, m));
        for i in 0..n_samples {
            let yi = y[i];
            if !yi.is_finite() {
                // nansum masks these observations in Python → zero grad/hess.
                continue;
            }
            for (p, &tau) in self.expectiles.iter().enumerate() {
                let e = transformed[[i, p]];
                let w = if yi - e >= 0.0 { tau } else { 1.0 - tau };
                gradients[[i, p]] = -2.0 * w * (yi - e) * scale;
                hessians[[i, p]] = 2.0 * w * scale;
            }
        }

        Some((gradients, hessians))
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
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
                result[[i, j]] = normal_dist.sample(&mut rng);
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
    fn test_expectile_analytical_matches_numerical() {
        let dist = Expectile::new(
            vec![0.1, 0.5, 0.9],
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        let predictions = array![[1.0, 1.5, 2.0], [2.2, 1.8, 1.1]];
        let targets = array![1.2, 1.4];
        let target = ResponseData::Univariate(&targets.view());

        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");
        let numerical = dist
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("Should return numerical gradients");

        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-3);
                assert_relative_eq!(analytical.1[[i, j]], numerical.1[[i, j]], epsilon = 1e-2);
            }
        }
    }

    #[test]
    fn test_expectile_crossing_penalty_scales_gradients() {
        // One of the two rows has crossed expectiles → penalty = 0.5. Python's
        // autograd keeps the (1 + penalty) loss scaling in every gradient and
        // hessian; verify ours does too.
        let predictions = array![[1.0, 1.5, 2.0], [2.0, 1.5, 1.0]];
        let targets = array![1.2, 1.4];
        let target = ResponseData::Univariate(&targets.view());

        let base = Expectile::new(
            vec![0.1, 0.5, 0.9],
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        let penalized = Expectile::new(
            vec![0.1, 0.5, 0.9],
            true,
            Stabilization::None,
            LossFn::Nll,
            false,
        );

        let transformed = base.transform_params(&predictions.view());
        let (g0, h0) = base
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .unwrap();
        let (g1, h1) = penalized
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .unwrap();

        for (a, b) in g0.iter().zip(g1.iter()) {
            assert_relative_eq!(*b, a * 1.5, epsilon = 1e-12);
        }
        for (a, b) in h0.iter().zip(h1.iter()) {
            assert_relative_eq!(*b, a * 1.5, epsilon = 1e-12);
        }
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

        // Should have shape (n_samples, n_obs) = (1000, 2)
        assert_eq!(samples.dim(), (1000, 2));

        // Check that samples for first observation are centered around the median expectile value
        let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_0, 1.5, epsilon = 0.1);

        // Check that samples for second observation are centered around the median expectile value
        let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_1, 1.6, epsilon = 0.1);
    }
}
