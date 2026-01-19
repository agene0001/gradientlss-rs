//! Zero-Adjusted Beta distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Bernoulli, Beta, Distribution as RandDistribution};
use serde::{Deserialize, Serialize};

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

    /// Transform parameters to the distribution parameter space.
    fn transform_dist_params(&self, params: &[f64]) -> (f64, f64, f64) {
        let concentration1 = self.params[0].response_fn.apply_scalar(params[0]);
        let concentration0 = self.params[1].response_fn.apply_scalar(params[1]);
        let gate = self.params[2].response_fn.apply_scalar(params[2]);

        (concentration1, concentration0, gate)
    }

    /// Compute the log probability for zero-adjusted Beta distribution.
    fn log_prob_zabeta(
        &self,
        concentration1: f64,
        concentration0: f64,
        gate: f64,
        target: f64,
    ) -> f64 {
        // Handle zero case
        if target == 0.0 {
            return gate.ln(); // Probability of zero
        }

        // Handle continuous case (0 < y < 1)
        if !(0.0 < target && target < 1.0) {
            return f64::NEG_INFINITY;
        }

        // Check that concentration parameters are positive
        if concentration1 <= 0.0 || concentration0 <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Check that gate probability is valid
        if !(0.0 < gate && gate < 1.0) {
            return f64::NEG_INFINITY;
        }

        // Compute Beta log probability
        let log_beta = Self::log_beta(concentration1, concentration0);
        let log_prob_beta = (concentration1 - 1.0) * target.ln()
            + (concentration0 - 1.0) * (1.0 - target).ln()
            - log_beta;

        // Combine with gate probability (probability of non-zero)
        (1.0 - gate).ln() + log_prob_beta
    }

    /// Compute the log of the beta function.
    fn log_beta(a: f64, b: f64) -> f64 {
        Self::log_gamma(a) + Self::log_gamma(b) - Self::log_gamma(a + b)
    }

    /// Approximate log gamma function (Lanczos approximation).
    fn log_gamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Lanczos approximation coefficients
        let g = 7.0;
        let p = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        if x < 0.5 {
            return Self::log_gamma(x + 1.0) - x.ln();
        }

        let x_adj = x - 1.0;
        let mut a = p[0];
        for i in 1..p.len() {
            a += p[i] / (x_adj + i as f64);
        }

        let t = x_adj + g + 0.5;
        (x_adj + 0.5).ln() * (2.4041138063191886 * x).sqrt() - t + a.ln()
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

        let (concentration1, concentration0, gate) = self.transform_dist_params(params);
        self.log_prob_zabeta(concentration1, concentration0, gate, target[0])
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
            let row_params: Vec<f64> = params.row(j).to_vec();
            let (concentration1, concentration0, gate) = self.transform_dist_params(&row_params);

            // Ensure valid parameters
            let concentration1 = concentration1.max(0.1);
            let concentration0 = concentration0.max(0.1);
            let gate = gate.clamp(0.01, 0.99);

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

        // Test with zero target
        let params = vec![1.0, 1.0, 0.0]; // concentration1=exp(1), concentration0=exp(1), gate=sigmoid(0)=0.5
        let target = vec![0.0];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_zabeta_log_prob_continuous() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // Test with continuous target
        let params = vec![1.0, 1.0, 0.0]; // concentration1=exp(1), concentration0=exp(1), gate=sigmoid(0)=0.5
        let target = vec![0.5];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_zabeta_invalid_target() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // Test with invalid target (> 1)
        let params = vec![1.0, 1.0, 0.0];
        let target = vec![1.5];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p == f64::NEG_INFINITY);
    }

    #[test]
    fn test_zabeta_nll() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        let params = array![[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let target = array![0.0, 0.5];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_zabeta_sample() {
        let dist = ZABeta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        let params = array![[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        // Should have shape (n_samples, n_obs) = (1000, 2)
        assert_eq!(samples.dim(), (1000, 2));

        // Check that we have a mix of zeros and continuous values
        let zero_count_0 = samples.column(0).iter().filter(|&&x| x == 0.0).count();
        let zero_count_1 = samples.column(1).iter().filter(|&&x| x == 0.0).count();

        // First observation should have ~50% zeros (gate=0.5)
        assert!(zero_count_0 > 300 && zero_count_0 < 700);
        // Second observation should have ~90% zeros (gate=sigmoid(1)≈0.73)
        assert!(zero_count_1 > 600 && zero_count_1 < 800);
    }
}
