//! Multivariate Normal distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Multivariate Normal distribution for distributional regression.
///
/// This distribution is parameterized by a mean vector and a lower-triangular
/// matrix L with positive-valued diagonal entries, such that Σ = LL'.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MVN {
    n_targets: usize,
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl MVN {
    pub fn new(
        n_targets: usize,
        stabilization: Stabilization,
        scale_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        if n_targets < 2 {
            panic!("MVN requires at least 2 targets");
        }

        let mut params = Vec::new();

        // Location parameters (mean vector)
        for i in 0..n_targets {
            params.push(DistributionParam::new(
                format!("location_{}", i + 1),
                ResponseFn::Identity,
            ));
        }

        // Scale tril parameters (lower triangular matrix)
        let tril_indices = Self::get_tril_indices(n_targets);

        for (_, (row, col)) in tril_indices.iter().enumerate() {
            let param_name = if row == col {
                format!("scale_tril_diag_{}", col + 1)
            } else {
                format!("scale_tril_offdiag_{}{}", col + 1, row + 1)
            };

            let response_fn = if row == col {
                scale_response_fn
            } else {
                ResponseFn::Identity
            };

            params.push(DistributionParam::new(param_name, response_fn));
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

    /// Get the lower triangular indices for a given dimension.
    fn get_tril_indices(n: usize) -> Vec<(usize, usize)> {
        let mut indices = Vec::new();
        for col in 0..n {
            for row in col..n {
                indices.push((row, col));
            }
        }
        indices
    }

    /// Transform parameters to the distribution parameter space.
    fn transform_dist_params(&self, params: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n_params = self.n_params();
        let mut loc = Vec::with_capacity(self.n_targets);
        let mut tril = Vec::with_capacity(n_params - self.n_targets);

        // Extract location parameters
        for i in 0..self.n_targets {
            loc.push(params[i]);
        }

        // Extract and transform tril parameters
        for i in self.n_targets..n_params {
            let param = &self.params[i];
            tril.push(param.response_fn.apply_scalar(params[i]));
        }

        (loc, tril)
    }

    /// Forward substitution to solve L*x = b where L is lower triangular.
    fn forward_substitution(l: &Array2<f64>, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut x = vec![0.0; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[[i, j]] * x[j];
            }
            if l[[i, i]].abs() > 1e-10 {
                x[i] = sum / l[[i, i]];
            } else {
                x[i] = 0.0; // Handle singular case
            }
        }
        x
    }

    /// Compute the log probability for multivariate normal distribution.
    fn log_prob_mvn(&self, loc: &[f64], tril: &[f64], target: &[f64]) -> f64 {
        let n = self.n_targets;
        let mut cov = Array2::zeros((n, n));

        // Reconstruct the lower triangular matrix
        let tril_indices = Self::get_tril_indices(n);
        for (i, (row, col)) in tril_indices.iter().enumerate() {
            cov[[*row, *col]] = tril[i];
        }

        // Compute covariance matrix: Σ = LL^T

        // Compute the difference vector
        let diff: Vec<f64> = target.iter().zip(loc.iter()).map(|(t, l)| t - l).collect();

        // Compute the log determinant of the covariance matrix
        let log_det = 2.0
            * tril
                .iter()
                .enumerate()
                .filter(|(i, _)| tril_indices[*i].0 == tril_indices[*i].1) // Only diagonal elements
                .map(|(_, &val)| val.ln())
                .sum::<f64>();

        // Compute the quadratic form: (y-μ)^T Σ^{-1} (y-μ)
        // Using the Cholesky factor: Σ^{-1} = (LL^T)^{-1} = L^{-T}L^{-1}
        // So (y-μ)^T Σ^{-1} (y-μ) = ||L^{-1}(y-μ)||^2
        // Solve L*z = diff for z using forward substitution
        let z = Self::forward_substitution(&cov, &diff);

        let quadratic_form = z.iter().map(|&x| x * x).sum::<f64>();

        // Compute log probability
        -0.5 * (n as f64) * (2.0 * PI).ln() - 0.5 * log_det - 0.5 * quadratic_form
    }
}

#[typetag::serde]
impl Distribution for MVN {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "MVN"
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

        let (loc, tril) = self.transform_dist_params(params);
        self.log_prob_mvn(&loc, &tril, target)
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(_) => {
                panic!("MVN requires multivariate targets")
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

        // For multivariate, we return samples with shape (n_samples, n_obs * n_targets)
        let mut result = Array2::zeros((n_samples, n_obs * n_targets));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let row_params: Vec<f64> = params.row(j).to_vec();
            let (loc, tril) = self.transform_dist_params(&row_params);

            // Reconstruct the lower triangular matrix
            let tril_indices = Self::get_tril_indices(n_targets);
            let mut cov = Array2::zeros((n_targets, n_targets));
            for (i, (row, col)) in tril_indices.iter().enumerate() {
                cov[[*row, *col]] = tril[i];
            }

            // Compute covariance matrix: Σ = LL^T

            // Sample from multivariate normal
            let mut samples = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                let mut sample = Vec::with_capacity(n_targets);

                // Sample from standard normal
                let standard_normal = Normal::new(0.0, 1.0).unwrap();
                let z: Vec<f64> = (0..n_targets)
                    .map(|_| standard_normal.sample(&mut rng))
                    .collect();

                // Transform using Cholesky decomposition: L * z + μ
                for i in 0..n_targets {
                    let mut sum = 0.0;
                    for k in 0..n_targets {
                        sum += cov[[i, k]] * z[k];
                    }
                    sample.push(loc[i] + sum);
                }

                samples.push(sample);
            }

            // Store samples in the result array
            for s in 0..n_samples {
                for t in 0..n_targets {
                    result[[s, j * n_targets + t]] = samples[s][t];
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
    fn test_mvn_creation() {
        let dist = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        assert_eq!(dist.n_params(), 5); // 2 loc + 3 tril (2 diag + 1 off-diag)
        assert_eq!(dist.n_targets(), 2);
        assert!(!dist.is_univariate());
        assert!(!dist.should_initialize());
    }

    #[test]
    fn test_mvn_log_prob() {
        let dist = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // Simple case: identity covariance after transformation
        // With ResponseFn::Exp, we need params that result in tril=[[1,0],[0,1]]
        // So diagonal elements should be log(1) = 0.0
        let params = vec![0.0, 0.0, 0.0, 0.0, 0.0]; // loc=[0,0], tril=[[exp(0),0],[0,exp(0)]] = [[1,0],[0,1]]
        let target = vec![0.0, 0.0];

        let log_p = dist.log_prob(&params, &target);
        // For 2D standard normal at mean, should be similar to 2 * log_prob of 1D standard normal
        let expected = -2.0 * (0.5 * (2.0 * PI).ln());
        assert_relative_eq!(log_p, expected, epsilon = 0.1);
    }

    #[test]
    fn test_mvn_nll() {
        let dist = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        let params = array![[0.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0]];
        let target = array![[0.0, 0.0], [1.0, 1.0]];
        let target_response = ResponseData::Multivariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_mvn_sample() {
        let dist = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        let params = array![[0.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        // Should have shape (n_samples, n_obs * n_targets) = (1000, 2*2) = (1000, 4)
        assert_eq!(samples.dim(), (1000, 4));

        // Check that first observation samples have mean around [0, 0]
        let mean_0_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        let mean_0_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;

        assert_relative_eq!(mean_0_0, 0.0, epsilon = 0.1);
        assert_relative_eq!(mean_0_1, 0.0, epsilon = 0.1);

        // Check that second observation samples have mean around [1, 1]
        let mean_1_0: f64 = samples.column(2).iter().sum::<f64>() / 1000.0;
        let mean_1_1: f64 = samples.column(3).iter().sum::<f64>() / 1000.0;

        assert_relative_eq!(mean_1_0, 1.0, epsilon = 0.1);
        assert_relative_eq!(mean_1_1, 1.0, epsilon = 0.1);
    }
}
