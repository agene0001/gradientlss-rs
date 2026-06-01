//! Zero-Inflated Negative Binomial distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Gamma as RandGamma, Poisson as RandPoisson};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZINB {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl ZINB {
    pub fn new(
        stabilization: Stabilization,
        total_count_response_fn: ResponseFn,
        probs_response_fn: ResponseFn,
        gate_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("total_count", total_count_response_fn),
            DistributionParam::new("probs", probs_response_fn),
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
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        )
    }

    /// NB ln_pmf inlined: ln_gamma(r+k) - ln_gamma(r) - ln_gamma(k+1) + r*ln(p) + k*ln(1-p)
    fn nb_ln_pmf(r: f64, p: f64, k: u64) -> f64 {
        let kf = k as f64;
        ln_gamma(r + kf) - ln_gamma(r) - ln_gamma(kf + 1.0) + r * p.ln() + kf * (-p).ln_1p()
    }

    /// Helper method for scalar log probability (inlined formula)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let r = params[0]; // total_count
        let probs = params[1];
        let gate = params[2];

        if r <= 0.0 || probs <= 0.0 || probs >= 1.0 || gate < 0.0 || gate > 1.0 {
            return f64::NEG_INFINITY;
        }

        // PyTorch probs → statrs p = 1 - probs
        let p = 1.0 - probs;

        if target == 0.0 {
            // NB pmf(0) = p^r, so gate + (1-gate)*p^r
            (gate + (1.0 - gate) * p.powf(r)).ln()
        } else {
            (1.0 - gate).ln() + Self::nb_ln_pmf(r, p, target as u64)
        }
    }
}

#[typetag::serde]
impl Distribution for ZINB {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "ZINB"
    }

    fn is_discrete(&self) -> bool {
        true
    }

    fn n_params(&self) -> usize {
        3
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
                let col2 = params.column(2);
                crate::distributions::util::par_sum(y.len(), |i| {
                    -self.log_prob_scalar(&[col0[i], col1[i], col2[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("ZINB is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let total_count = params[[j, 0]];
            let probs = params[[j, 1]];
            let gate = params[[j, 2]];

            if total_count > 0.0 && probs > 0.0 && probs < 1.0 && gate >= 0.0 && gate <= 1.0 {
                // NegativeBinomial as Gamma-Poisson mixture
                // rand_distr::Gamma takes (shape, scale), where scale = 1/rate
                // For NB(total_count, probs): Gamma(shape=total_count, scale=probs/(1-probs))
                let scale = probs / (1.0 - probs);
                if let Ok(gamma_dist) = RandGamma::new(total_count, scale) {
                    for i in 0..n_samples {
                        if rng.random_bool(gate) {
                            result[[i, j]] = 0.0;
                        } else {
                            let lambda: f64 = gamma_dist.sample(&mut rng);
                            if let Ok(poisson_dist) = RandPoisson::new(lambda) {
                                result[[i, j]] = poisson_dist.sample(&mut rng) as f64;
                            }
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
    fn test_zinb_creation() {
        let dist = ZINB::default();
        assert_eq!(dist.n_params(), 3);
        assert_eq!(dist.param_names(), vec!["total_count", "probs", "gate"]);
        assert!(dist.is_discrete());
    }

    #[test]
    fn test_zinb_log_prob() {
        let dist = ZINB::default();
        // r=5, probs=0.5 → statrs p=0.5, pmf(0) = 0.5^5 = 0.03125

        let log_p_zero = dist.log_prob_scalar(&[5.0, 0.5, 0.1], 0.0);
        let expected_zero = (0.1 + 0.9 * 0.5_f64.powi(5)).ln();
        assert_relative_eq!(log_p_zero, expected_zero, epsilon = 1e-10);

        let log_p_non_zero = dist.log_prob_scalar(&[5.0, 0.5, 0.1], 3.0);
        // ln(0.9) + nb_ln_pmf(5, 0.5, 3)
        let expected_non_zero = 0.9_f64.ln() + ln_gamma(8.0) - ln_gamma(5.0) - ln_gamma(4.0)
            + 5.0 * 0.5_f64.ln()
            + 3.0 * (-0.5_f64).ln_1p();
        assert_relative_eq!(log_p_non_zero, expected_non_zero, epsilon = 1e-10);
    }
}
