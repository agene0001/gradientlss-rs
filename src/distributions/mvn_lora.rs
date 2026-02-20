//! Low-Rank Multivariate Normal distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::constants::LOG_2PI;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use serde::{Deserialize, Serialize};

/// Low-Rank Multivariate Normal distribution for distributional regression.
///
/// This distribution uses a low-rank form of the covariance matrix:
/// covariance_matrix = cov_factor @ cov_factor.T + cov_diag
/// which is more efficient for high-dimensional data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MVNLoRa {
    n_targets: usize,
    rank: usize,
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl MVNLoRa {
    pub fn new(
        n_targets: usize,
        rank: usize,
        stabilization: Stabilization,
        response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        if n_targets < 2 {
            panic!("MVN_LoRa requires at least 2 targets");
        }
        if rank >= n_targets {
            panic!("Rank must be less than number of targets");
        }
        if rank == 0 {
            panic!("Rank must be at least 1");
        }

        let mut params = Vec::new();

        // Location parameters (mean vector)
        for i in 0..n_targets {
            params.push(DistributionParam::new(
                format!("location_{}", i + 1),
                ResponseFn::Identity,
            ));
        }

        // Covariance factor parameters (low-rank part)
        for r in 0..rank {
            for i in 0..n_targets {
                params.push(DistributionParam::new(
                    format!("cov_factor_{}_{}", r + 1, i + 1),
                    ResponseFn::Identity,
                ));
            }
        }

        // Covariance diagonal parameters (diagonal part)
        for i in 0..n_targets {
            params.push(DistributionParam::new(
                format!("cov_diag_{}", i + 1),
                response_fn,
            ));
        }

        Self {
            n_targets,
            rank,
            params,
            stabilization,
            loss_fn,
            initialize,
        }
    }

    pub fn default() -> Self {
        Self::new(
            3,
            2,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        )
    }

    /// Transform parameters to the distribution parameter space.
    fn transform_dist_params(&self, params: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n_params = self.n_params();
        let mut loc = Vec::with_capacity(self.n_targets);
        let mut cov_factor = Vec::with_capacity(self.rank * self.n_targets);
        let mut cov_diag = Vec::with_capacity(self.n_targets);

        // Extract location parameters
        for i in 0..self.n_targets {
            loc.push(params[i]);
        }

        // Extract covariance factor parameters
        let start_factor = self.n_targets;
        let end_factor = start_factor + self.rank * self.n_targets;
        for i in start_factor..end_factor {
            cov_factor.push(params[i]);
        }

        // Cov diagonal params are already transformed by transform_params before reaching here
        for i in end_factor..n_params {
            cov_diag.push(params[i]);
        }

        (loc, cov_factor, cov_diag)
    }

    /// Compute the log probability for low-rank multivariate normal distribution.
    fn log_prob_mvn_lora(
        &self,
        loc: &[f64],
        cov_factor: &[f64],
        cov_diag: &[f64],
        target: &[f64],
    ) -> f64 {
        let n = self.n_targets;
        let r = self.rank;

        // Reconstruct covariance factor matrix (n x r)
        let mut factor_matrix = Array2::zeros((n, r));
        for i in 0..n {
            for j in 0..r {
                factor_matrix[[i, j]] = cov_factor[j * n + i];
            }
        }

        // Compute covariance matrix: Σ = FF^T + D
        let factor_part = factor_matrix.dot(&factor_matrix.t());
        let mut cov_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                cov_matrix[[i, j]] = factor_part[[i, j]];
                if i == j {
                    cov_matrix[[i, j]] += cov_diag[i];
                }
            }
        }

        // Compute the difference vector
        let diff: Vec<f64> = target.iter().zip(loc.iter()).map(|(t, l)| t - l).collect();

        // Compute the log determinant using matrix determinant lemma
        let log_det = Self::log_det_low_rank(&factor_matrix, &cov_diag);

        // Compute the quadratic form: (y-μ)^T Σ^{-1} (y-μ)
        // Using simple Gaussian elimination to solve Σ*x = diff
        let x = Self::solve_linear_system(&cov_matrix, &diff);

        let quadratic_form = diff
            .iter()
            .zip(x.iter())
            .map(|(&d, &xi)| d * xi)
            .sum::<f64>();

        // Compute log probability
        -0.5 * (n as f64) * LOG_2PI - 0.5 * log_det - 0.5 * quadratic_form
    }

    /// Solve linear system A*x = b using Gaussian elimination with partial pivoting.
    fn solve_linear_system(a: &Array2<f64>, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut aug = Array2::zeros((n, n + 1));

        // Create augmented matrix [A|b]
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Gaussian elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            for j in 0..=n {
                let tmp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }

            // Eliminate column
            if aug[[k, k]].abs() > 1e-10 {
                for i in (k + 1)..n {
                    let factor = aug[[i, k]] / aug[[k, k]];
                    for j in k..=n {
                        aug[[i, j]] -= factor * aug[[k, j]];
                    }
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in (i + 1)..n {
                sum -= aug[[i, j]] * x[j];
            }
            if aug[[i, i]].abs() > 1e-10 {
                x[i] = sum / aug[[i, i]];
            }
        }

        x
    }

    /// Compute log determinant using matrix determinant lemma for low-rank updates.
    fn log_det_low_rank(factor_matrix: &Array2<f64>, cov_diag: &[f64]) -> f64 {
        let n = factor_matrix.nrows();

        // Compute D (diagonal matrix)
        let mut d_matrix = Array2::zeros((n, n));
        for i in 0..n {
            d_matrix[[i, i]] = cov_diag[i];
        }

        // Compute F^T D^{-1} F
        let mut d_inv = Array2::zeros((n, n));
        for i in 0..n {
            d_inv[[i, i]] = 1.0 / cov_diag[i];
        }

        let f_t_d_inv = factor_matrix.t().dot(&d_inv);
        let f_t_d_inv_f = f_t_d_inv.dot(factor_matrix);

        // Compute log|D| + log|I + F^T D^{-1} F|
        let log_det_d: f64 = cov_diag.iter().map(|&d| d.ln()).sum();

        // Compute log|I + F^T D^{-1} F| using eigenvalues
        let eigenvalues = Self::eigenvalues(&f_t_d_inv_f);
        let log_det_update: f64 = eigenvalues.iter().map(|&lambda| (1.0 + lambda).ln()).sum();

        log_det_d + log_det_update
    }

    /// Compute eigenvalues of a small symmetric matrix using Jacobi iteration.
    fn eigenvalues(matrix: &Array2<f64>) -> Vec<f64> {
        let n = matrix.nrows();

        if n == 1 {
            return vec![matrix[[0, 0]]];
        }

        // Jacobi eigenvalue algorithm for small symmetric matrices
        let mut a = matrix.clone();
        let max_iter = 100;

        for _ in 0..max_iter {
            // Find largest off-diagonal element
            let mut max_val = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[[i, j]].abs() > max_val {
                        max_val = a[[i, j]].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            // Check convergence
            if max_val < 1e-12 {
                break;
            }

            // Compute rotation angle
            let theta = if (a[[p, p]] - a[[q, q]]).abs() < 1e-15 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * a[[p, q]] / (a[[p, p]] - a[[q, q]])).atan()
            };

            let c = theta.cos();
            let s = theta.sin();

            // Apply Jacobi rotation
            let mut new_a = a.clone();
            for i in 0..n {
                if i != p && i != q {
                    new_a[[i, p]] = c * a[[i, p]] + s * a[[i, q]];
                    new_a[[p, i]] = new_a[[i, p]];
                    new_a[[i, q]] = -s * a[[i, p]] + c * a[[i, q]];
                    new_a[[q, i]] = new_a[[i, q]];
                }
            }
            new_a[[p, p]] = c * c * a[[p, p]] + 2.0 * s * c * a[[p, q]] + s * s * a[[q, q]];
            new_a[[q, q]] = s * s * a[[p, p]] - 2.0 * s * c * a[[p, q]] + c * c * a[[q, q]];
            new_a[[p, q]] = 0.0;
            new_a[[q, p]] = 0.0;

            a = new_a;
        }

        let mut eigenvalues = Vec::with_capacity(n);
        for i in 0..n {
            eigenvalues.push(a[[i, i]]);
        }

        eigenvalues
    }
}

#[typetag::serde]
impl Distribution for MVNLoRa {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "MVNLoRa"
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

        let (loc, cov_factor, cov_diag) = self.transform_dist_params(params);
        self.log_prob_mvn_lora(&loc, &cov_factor, &cov_diag, target)
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(_) => {
                panic!("MVN_LoRa requires multivariate targets")
            }
            ResponseData::Multivariate(arr) => {
                let mut total_nll = 0.0;
                let n_samples = params.nrows();

                for i in 0..n_samples {
                    let row_params = params.row(i);
                    let target_row = arr.row(i);

                    // Avoid allocation when rows are contiguous (standard layout)
                    let log_prob = match (row_params.as_slice(), target_row.as_slice()) {
                        (Some(rp), Some(tr)) => self.log_prob(rp, tr),
                        _ => {
                            let rp = row_params.to_vec();
                            let tr = target_row.to_vec();
                            self.log_prob(&rp, &tr)
                        }
                    };
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

        let standard_normal = Normal::new(0.0, 1.0).unwrap();
        let mut z = vec![0.0f64; n_targets];

        for j in 0..n_obs {
            let row_params: Vec<f64> = params.row(j).to_vec();
            let (loc, cov_factor, cov_diag) = self.transform_dist_params(&row_params);

            // Reconstruct covariance factor matrix (n x rank)
            let mut factor_matrix = Array2::zeros((n_targets, self.rank));
            for i in 0..n_targets {
                for r in 0..self.rank {
                    factor_matrix[[i, r]] = cov_factor[r * n_targets + i];
                }
            }

            // Compute covariance matrix: Σ = FF^T + D
            let factor_part = factor_matrix.dot(&factor_matrix.t());
            let mut cov_matrix = Array2::zeros((n_targets, n_targets));

            for i in 0..n_targets {
                for jj in 0..n_targets {
                    cov_matrix[[i, jj]] = factor_part[[i, jj]];
                    if i == jj {
                        cov_matrix[[i, jj]] += cov_diag[i];
                    }
                }
            }

            // Compute Cholesky decomposition of covariance matrix: Σ = LL^T
            let mut chol = Array2::zeros((n_targets, n_targets));
            for i in 0..n_targets {
                for jj in 0..=i {
                    let mut sum = cov_matrix[[i, jj]];
                    for k in 0..jj {
                        sum -= chol[[i, k]] * chol[[jj, k]];
                    }
                    if i == jj {
                        chol[[i, jj]] = if sum > 0.0 { sum.sqrt() } else { 1e-6 };
                    } else {
                        chol[[i, jj]] = sum / chol[[jj, jj]];
                    }
                }
            }

            // Sample from multivariate normal using Cholesky factor: L * z + μ
            for s in 0..n_samples {
                // Sample z from standard normal (reuse buffer)
                for zi in z.iter_mut() {
                    *zi = standard_normal.sample(&mut rng);
                }

                // Transform using Cholesky factor: L * z + μ
                for t in 0..n_targets {
                    let mut sum = 0.0;
                    for k in 0..=t {
                        sum += chol[[t, k]] * z[k];
                    }
                    result[[s, j * n_targets + t]] = loc[t] + sum;
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
    fn test_mvn_lora_creation() {
        let dist = MVNLoRa::new(
            3,
            2,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        );
        // 3 loc + 3*2 cov_factor + 3 cov_diag = 3 + 6 + 3 = 12 parameters
        assert_eq!(dist.n_params(), 12);
        assert_eq!(dist.n_targets(), 3);
        assert_eq!(dist.rank, 2);
        assert!(!dist.is_univariate());
        assert!(!dist.should_initialize());
    }

    #[test]
    fn test_mvn_lora_log_prob() {
        let dist = MVNLoRa::new(
            2,
            1,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        );

        // Simple case: rank=1, 2 targets
        // loc = [0, 0], cov_factor = [0, 0], cov_diag = [1, 1]
        // Params are in already-transformed space (cov_diag already exp'd)
        let params = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let target = vec![0.0, 0.0];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_mvn_lora_nll() {
        let dist = MVNLoRa::new(
            2,
            1,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        );
        // Params are in already-transformed space (cov_diag already exp'd)
        let params = array![
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        ];
        let target = array![[0.0, 0.0], [1.0, 1.0]];
        let target_response = ResponseData::Multivariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_mvn_lora_sample() {
        let dist = MVNLoRa::new(
            2,
            1,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        );
        // Params are in already-transformed space (cov_diag already exp'd)
        let params = array![
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        ];
        let samples = dist.sample(&params.view(), 1000, 123);

        // Should have shape (n_samples, n_obs * n_targets) = (1000, 2*2) = (1000, 4)
        assert_eq!(samples.dim(), (1000, 4));

        // Check that first observation samples have mean around [0, 0]
        let mean_0_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        let mean_0_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;

        assert_relative_eq!(mean_0_0, 0.0, epsilon = 0.2);
        assert_relative_eq!(mean_0_1, 0.0, epsilon = 0.2);
    }
}
