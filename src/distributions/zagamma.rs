//! Zero-Adjusted Gamma distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Gamma as RandGamma};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZAGamma {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl ZAGamma {
    pub fn new(
        stabilization: Stabilization,
        concentration_response_fn: ResponseFn,
        rate_response_fn: ResponseFn,
        gate_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("concentration", concentration_response_fn),
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
            ResponseFn::Exp,
            ResponseFn::Exp,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        )
    }

    /// Helper method for scalar log probability.
    /// Matches Python's ZeroInflatedDistribution.log_prob which uses the mixture formula
    /// for both zero and non-zero values. For zero values, continuous distributions
    /// clamp the value to epsilon before evaluating the base distribution.
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let concentration = params[0];
        let rate = params[1];
        let gate = params[2];

        if concentration <= 0.0 || rate <= 0.0 || gate < 0.0 || gate > 1.0 {
            return f64::NEG_INFINITY;
        }

        let is_zero = target == 0.0;

        // Clamp value away from 0 for continuous distributions (matches Python's epsilon clamp)
        let clamped_target = if target <= 0.0 {
            crate::constants::ZERO_CLAMP_EPS
        } else {
            target
        };

        // Gamma ln_pdf inlined
        let base_log_prob = concentration * rate.ln() + (concentration - 1.0) * clamped_target.ln()
            - rate * clamped_target
            - ln_gamma(concentration);
        let log_one_minus_gate = (1.0 - gate).ln();
        let log_prob = log_one_minus_gate + base_log_prob;

        if is_zero {
            // log(gate + (1-gate) * base_pdf(epsilon))
            (gate + log_prob.exp()).ln()
        } else {
            log_prob
        }
    }
}

#[typetag::serde]
impl Distribution for ZAGamma {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "ZAGamma"
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
                    let y_val = y[i];
                    if y_val < 0.0 {
                        f64::INFINITY
                    } else {
                        -self.log_prob_scalar(&[col0[i], col1[i], col2[i]], y_val)
                    }
                })
            }
            ResponseData::Multivariate(_) => panic!("ZAGamma is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let concentration = params[[j, 0]];
            let rate = params[[j, 1]];
            let gate = params[[j, 2]];

            if concentration > 0.0 && rate > 0.0 && gate >= 0.0 && gate <= 1.0 {
                // rand_distr::Gamma uses shape and scale (1/rate) parameterization
                if let Ok(gamma_dist) = RandGamma::new(concentration, 1.0 / rate) {
                    for i in 0..n_samples {
                        if rng.random_bool(gate) {
                            result[[i, j]] = 0.0;
                        } else {
                            result[[i, j]] = gamma_dist.sample(&mut rng);
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
    fn test_zagamma_creation() {
        let dist = ZAGamma::default();
        assert_eq!(dist.n_params(), 3);
        assert_eq!(dist.param_names(), vec!["concentration", "rate", "gate"]);
    }

    #[test]
    fn test_zagamma_log_prob() {
        let dist = ZAGamma::default();
        // Gamma(shape=2, rate=1) ln_pdf helper
        let gamma_ln_pdf =
            |x: f64| -> f64 { 2.0 * 1.0_f64.ln() + 1.0 * x.ln() - 1.0 * x - ln_gamma(2.0) };

        // Zero case: log(gate + (1-gate) * base_pdf(epsilon))
        let log_p_zero = dist.log_prob_scalar(&[2.0, 1.0, 0.1], 0.0);
        let base_pdf_eps = gamma_ln_pdf(f32::EPSILON as f64).exp();
        let expected_zero = (0.1 + 0.9 * base_pdf_eps).ln();
        assert_relative_eq!(log_p_zero, expected_zero, epsilon = 1e-10);

        let log_p_non_zero = dist.log_prob_scalar(&[2.0, 1.0, 0.1], 1.5);
        let expected_non_zero = 0.9_f64.ln() + gamma_ln_pdf(1.5);
        assert_relative_eq!(log_p_non_zero, expected_non_zero, epsilon = 1e-10);
    }
}
