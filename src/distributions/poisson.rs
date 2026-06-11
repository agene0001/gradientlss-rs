//! Poisson distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Poisson as RandPoisson};
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Poisson {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Poisson {
    pub fn new(
        stabilization: Stabilization,
        rate_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![DistributionParam::new("rate", rate_response_fn)];
        Self {
            params,
            stabilization,
            loss_fn,
            initialize,
        }
    }

    pub fn default() -> Self {
        Self::new(Stabilization::None, ResponseFn::Relu, LossFn::Nll, false)
    }

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let rate = params[0];
        if rate <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // Use the target as-is rather than truncating to u64. The lgamma-based
        // ln_pmf is well-defined for non-integer counts, and truncating silently
        // mapped negatives to 0; counts outside the support are -inf.
        let k = target;
        if !k.is_finite() || k < 0.0 {
            return f64::NEG_INFINITY;
        }
        // ln_pmf = -λ + k*ln(λ) - ln(k!) = -λ + k*ln(λ) - ln_gamma(k+1)
        -rate + k * rate.ln() - crate::constants::ln_factorial(k).unwrap_or_else(|| ln_gamma(k + 1.0))
    }
}

#[typetag::serde]
impl Distribution for Poisson {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Poisson"
    }

    fn is_discrete(&self) -> bool {
        true
    }
    fn n_params(&self) -> usize {
        1
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
                let rate_col = params.column(0);
                crate::distributions::util::par_sum(y.len(), |i| {
                    -self.log_prob_scalar(&[rate_col[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("Poisson is a univariate distribution."),
        }
    }

    /// Analytical gradients for Poisson distribution.
    ///
    /// NLL = λ - k*ln(λ) + ln_gamma(k+1)
    ///
    /// Gradients w.r.t. distribution parameter:
    /// - dNLL/dλ = 1 - k/λ
    ///
    /// Hessian w.r.t. distribution parameter:
    /// - d²NLL/dλ² = k/λ²
    ///
    /// Chain rule applied for response function: dNLL/dpred = dNLL/dλ * dλ/dpred
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
        let rate_response_fn = &self.params[0].response_fn;

        let t_rate = transformed.column(0);
        let p_rate = predictions.column(0);

        // Batch response-fn derivatives (auto-vectorized), shared by both paths.
        let (rd, rsd) = rate_response_fn.derivative_batches(&p_rate);

        let compute = |i: usize| -> (f64, f64) {
            let lambda = t_rate[i].max(1e-6);
            let k = y[i];

            if !k.is_finite() || k < 0.0 {
                return (0.0, 0.0);
            }

            let grad_lambda = 1.0 - k / lambda;
            let hess_lambda = k / (lambda * lambda);

            let d = rd[i];
            let sd = rsd[i];
            let g = grad_lambda * d;
            let h = hess_lambda * d * d + grad_lambda * sd;
            (g, h)
        };

        let mut gradients = Array2::zeros((n_samples, 1));
        let mut hessians = Array2::zeros((n_samples, 1));
        // Single param → one element per sample; write straight into the backing
        // slices. Threshold matches the numerical path (256).
        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");

        const PAR_MIN: usize = 256;
        if n_samples >= PAR_MIN {
            g_slice
                .par_iter_mut()
                .zip(h_slice.par_iter_mut())
                .enumerate()
                .for_each(|(i, (g, h))| {
                    let (gv, hv) = compute(i);
                    *g = gv;
                    *h = hv;
                });
        } else {
            g_slice
                .iter_mut()
                .zip(h_slice.iter_mut())
                .enumerate()
                .for_each(|(i, (g, h))| {
                    let (gv, hv) = compute(i);
                    *g = gv;
                    *h = hv;
                });
        }

        Some((gradients, hessians))
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for j in 0..n_obs {
            let rate = params[[j, 0]];
            if rate > 0.0 {
                if let Ok(dist) = RandPoisson::new(rate) {
                    for i in 0..n_samples {
                        result[[i, j]] = RandDistribution::sample(&dist, &mut rng) as f64;
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
    fn test_poisson_creation() {
        let dist = Poisson::default();
        assert_eq!(dist.n_params(), 1);
        assert_eq!(dist.param_names(), vec!["rate"]);
        assert!(dist.is_discrete());
    }

    #[test]
    fn test_poisson_log_prob() {
        let dist = Poisson::default();
        let log_p = dist.log_prob_scalar(&[5.0], 3.0);
        // Poisson(λ=5) at k=3: -5 + 3*ln(5) - ln_gamma(4)
        let expected = -5.0 + 3.0 * 5.0_f64.ln() - ln_gamma(4.0);
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        let dist = Poisson::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        let predictions = array![[1.0], [0.5], [1.5]];
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
            assert_relative_eq!(analytical.0[[i, 0]], numerical.0[[i, 0]], epsilon = 1e-3);
            // True (unfloored) Hessians: both paths must agree on value AND sign.
            assert_relative_eq!(
                analytical.1[[i, 0]],
                numerical.1[[i, 0]],
                epsilon = 1e-2,
                max_relative = 1e-2
            );
        }
    }

    #[test]
    fn test_poisson_nll() {
        let dist = Poisson::default();
        let params = array![[5.0], [5.0]];
        let target = array![3.0, 3.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(-5.0 + 3.0 * 5.0_f64.ln() - ln_gamma(4.0));
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}
