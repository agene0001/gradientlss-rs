//! NegativeBinomial distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Gamma as RandGamma, Poisson as RandPoisson};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::constants::trigamma;

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
            ResponseFn::Relu,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        )
    }

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let r = params[0]; // total_count
        let probs = params[1];

        if r <= 0.0 || probs <= 0.0 || probs >= 1.0 {
            return f64::NEG_INFINITY;
        }

        let k = target as u64 as f64;
        // PyTorch NB uses probs = P(success/counted event), mean = r*p/(1-p)
        // statrs NB uses p = P(success/stopping event), mean = r*(1-p)/p
        // They are complementary: statrs_p = 1 - probs
        let p = 1.0 - probs;
        // ln_pmf = ln_gamma(r+k) - ln_gamma(r) - ln_gamma(k+1) + r*ln(p) + k*ln(1-p)
        ln_gamma(r + k) - ln_gamma(r) - ln_gamma(k + 1.0) + r * p.ln() + k * (-p).ln_1p()
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
                let col0 = params.column(0);
                let col1 = params.column(1);
                let n_samples = y.len();

                // Parallelize the per-sample ln_gamma sum above the same threshold
                // `analytical_gradients` uses. The metric runs every boosting round
                // on both train and validation, so the sequential loop was the next
                // largest per-round cost once the O(rounds²) predict was removed.
                if n_samples >= 4096 {
                    (0..n_samples)
                        .into_par_iter()
                        .map(|i| -self.log_prob_scalar(&[col0[i], col1[i]], y[i]))
                        .sum()
                } else {
                    let mut total = 0.0;
                    for (i, &y_val) in y.iter().enumerate() {
                        total -= self.log_prob_scalar(&[col0[i], col1[i]], y_val);
                    }
                    total
                }
            }
            ResponseData::Multivariate(_) => {
                panic!("NegativeBinomial is a univariate distribution.")
            }
        }
    }

    /// Analytical gradients for Negative Binomial distribution.
    ///
    /// Using the parameterization where p = 1 - probs (internal), the NLL is:
    /// NLL = -ln_gamma(r+k) + ln_gamma(r) + ln_gamma(k+1) - r*ln(1-probs) - k*ln(probs)
    ///
    /// Gradients w.r.t. distribution parameters:
    /// - dNLL/dr = -digamma(r+k) + digamma(r) - ln(1-probs)
    /// - dNLL/dprobs = r/(1-probs) - k/probs
    ///
    /// Hessians w.r.t. distribution parameters:
    /// - d²NLL/dr² = -trigamma(r+k) + trigamma(r)
    /// - d²NLL/dprobs² = r/(1-probs)² + k/probs²
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

        let r_response_fn = &self.params[0].response_fn;
        let probs_response_fn = &self.params[1].response_fn;

        let t_r = transformed.column(0);
        let t_probs = transformed.column(1);
        let p_r = predictions.column(0);
        let p_probs = predictions.column(1);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        if n_samples >= 4096 {
            // Pre-compute batch derivatives for both parameters
            let rd_r = r_response_fn.derivative_batch(&p_r);
            let rsd_r = r_response_fn.second_derivative_batch(&p_r);
            let rd_p = probs_response_fn.derivative_batch(&p_probs);
            let rsd_p = probs_response_fn.second_derivative_batch(&p_probs);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let r = t_r[i].max(1e-6);
                let probs = t_probs[i].clamp(1e-6, 1.0 - 1e-6);
                let k = y[i];

                if !k.is_finite() || k < 0.0 {
                    return (0.0, 1e-6, 0.0, 1e-6);
                }

                let one_minus_probs = 1.0 - probs;

                let grad_r = -digamma(r + k) + digamma(r) - one_minus_probs.ln();
                let hess_r = -trigamma(r + k) + trigamma(r);
                let grad_probs = r / one_minus_probs - k / probs;
                let hess_probs = r / (one_minus_probs * one_minus_probs) + k / (probs * probs);

                let r0 = rd_r[i];
                let rs0 = rsd_r[i];
                let g0 = grad_r * r0;
                let h0 = (hess_r * r0 * r0 + grad_r * rs0).max(1e-6);

                let r1 = rd_p[i];
                let rs1 = rsd_p[i];
                let g1 = grad_probs * r1;
                let h1 = (hess_probs * r1 * r1 + grad_probs * rs1).max(1e-6);

                (g0, h0, g1, h1)
            };

            let results: Vec<_> = (0..n_samples).into_par_iter().map(compute_sample).collect();
            for (i, (g0, h0, g1, h1)) in results.into_iter().enumerate() {
                gradients[[i, 0]] = g0;
                hessians[[i, 0]] = h0;
                gradients[[i, 1]] = g1;
                hessians[[i, 1]] = h1;
            }
        } else {
            for i in 0..n_samples {
                let r = t_r[i].max(1e-6);
                let probs = t_probs[i].clamp(1e-6, 1.0 - 1e-6);
                let k = y[i];

                if !k.is_finite() || k < 0.0 {
                    gradients[[i, 0]] = 0.0;
                    hessians[[i, 0]] = 1e-6;
                    gradients[[i, 1]] = 0.0;
                    hessians[[i, 1]] = 1e-6;
                    continue;
                }

                let one_minus_probs = 1.0 - probs;

                let grad_r = -digamma(r + k) + digamma(r) - one_minus_probs.ln();
                let hess_r = -trigamma(r + k) + trigamma(r);
                let grad_probs = r / one_minus_probs - k / probs;
                let hess_probs = r / (one_minus_probs * one_minus_probs) + k / (probs * probs);

                let r0 = r_response_fn.derivative(p_r[i]);
                let rs0 = r_response_fn.second_derivative(p_r[i]);
                let g0 = grad_r * r0;
                let h0 = (hess_r * r0 * r0 + grad_r * rs0).max(1e-6);

                let r1 = probs_response_fn.derivative(p_probs[i]);
                let rs1 = probs_response_fn.second_derivative(p_probs[i]);
                let g1 = grad_probs * r1;
                let h1 = (hess_probs * r1 * r1 + grad_probs * rs1).max(1e-6);

                gradients[[i, 0]] = g0;
                hessians[[i, 0]] = h0;
                gradients[[i, 1]] = g1;
                hessians[[i, 1]] = h1;
            }
        }

        Some((gradients, hessians))
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
                // rand_distr::Gamma takes (shape, scale), where scale = 1/rate
                // For NB(total_count, probs): Gamma(shape=total_count, scale=probs/(1-probs))
                let scale = probs / (1.0 - probs);
                if let Ok(gamma_dist) = RandGamma::new(total_count, scale) {
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
        // NB(r=5, probs=0.5) at k=3, statrs p=0.5:
        // ln_gamma(8) - ln_gamma(5) - ln_gamma(4) + 5*ln(0.5) + 3*ln(0.5)
        let expected = ln_gamma(8.0) - ln_gamma(5.0) - ln_gamma(4.0) + 8.0 * 0.5_f64.ln();
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_negative_binomial_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        let dist = NegativeBinomial::new(
            Stabilization::None,
            ResponseFn::Exp,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        );
        let predictions = array![[0.5, 0.0], [0.8, -0.5], [1.0, 0.5]];
        let targets = array![3.0, 1.0, 5.0];
        let target = ResponseData::Univariate(&targets.view());

        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");

        let numerical = dist
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("Should return numerical gradients");

        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-2);
            }
        }
    }

    #[test]
    fn test_negative_binomial_nll() {
        let dist = NegativeBinomial::default();
        let params = array![[5.0, 0.5], [5.0, 0.5]];
        let target = array![3.0, 3.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(ln_gamma(8.0) - ln_gamma(5.0) - ln_gamma(4.0) + 8.0 * 0.5_f64.ln());
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}
