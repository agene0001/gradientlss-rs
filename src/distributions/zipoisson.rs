//! Zero-Inflated Poisson distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Poisson as RandPoisson};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZIPoisson {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl ZIPoisson {
    pub fn new(
        stabilization: Stabilization,
        rate_response_fn: ResponseFn,
        gate_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
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
            ResponseFn::Relu,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        )
    }

    /// Poisson ln_pmf inlined: -λ + k*ln(λ) - ln_gamma(k+1)
    fn poisson_ln_pmf(rate: f64, k: f64) -> f64 {
        -rate + k * rate.ln() - ln_gamma(k + 1.0)
    }

    /// Helper method for scalar log probability (inlined formula)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let rate = params[0];
        let gate = params[1];

        if rate <= 0.0 || gate < 0.0 || gate > 1.0 {
            return f64::NEG_INFINITY;
        }

        // Counts outside the support are -inf. Previously the target was
        // truncated via `as u64`, mapping negatives to 0 and dropping the
        // fractional part; the lgamma-based ln_pmf is valid for non-integer k.
        if !target.is_finite() || target < 0.0 {
            return f64::NEG_INFINITY;
        }

        if target == 0.0 {
            // pmf(0) = exp(-rate), so gate + (1-gate)*exp(-rate)
            (gate + (1.0 - gate) * (-rate).exp()).ln()
        } else {
            (1.0 - gate).ln() + Self::poisson_ln_pmf(rate, target)
        }
    }
}

#[typetag::serde]
impl Distribution for ZIPoisson {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "ZIPoisson"
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
                crate::distributions::util::par_sum(y.len(), |i| {
                    -self.log_prob_scalar(&[col0[i], col1[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("ZIPoisson is a univariate distribution."),
        }
    }

    /// Analytical gradients for the Zero-Inflated Poisson.
    ///
    /// With rate λ and gate π the NLL splits on whether the count is zero. Writing
    /// `A = π + (1-π) e^{-λ}` for the zero branch, the diagonal gradients and
    /// Hessians (matching the diagonal-Hessian convention the other distributions
    /// use) are:
    ///
    /// - k = 0:
    ///   - dNLL/dλ = (1-π) e^{-λ} / A
    ///   - dNLL/dπ = -(1 - e^{-λ}) / A
    ///   - d²NLL/dλ² = [(1-π) e^{-λ} / A]² - (1-π) e^{-λ} / A
    ///   - d²NLL/dπ² = [(1 - e^{-λ}) / A]²
    /// - k > 0 (the zero-inflation drops out):
    ///   - dNLL/dλ = 1 - k/λ,   d²NLL/dλ² = k/λ²
    ///   - dNLL/dπ = 1/(1-π),   d²NLL/dπ² = 1/(1-π)²
    ///
    /// each then chain-ruled through the response functions:
    /// `g = dNLL/dθ · f'(pred)`, `h = d²NLL/dθ² · f'(pred)² + dNLL/dθ · f''(pred)`.
    /// Replaces the numerical finite-difference fallback this distribution used
    /// before (two perturbed `log_prob` evaluations per parameter per sample).
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
        let gate_response_fn = &self.params[1].response_fn;

        let t_rate = transformed.column(0);
        let t_gate = transformed.column(1);
        let p_rate = predictions.column(0);
        let p_gate = predictions.column(1);

        // Batch response-fn derivatives (auto-vectorized), shared by both paths.
        let rd_r = rate_response_fn.derivative_batch(&p_rate);
        let rsd_r = rate_response_fn.second_derivative_batch(&p_rate);
        let rd_g = gate_response_fn.derivative_batch(&p_gate);
        let rsd_g = gate_response_fn.second_derivative_batch(&p_gate);

        let compute = |i: usize| -> (f64, f64, f64, f64) {
            let lambda = t_rate[i].max(1e-6);
            let gate = t_gate[i].clamp(1e-6, 1.0 - 1e-6);
            let k = y[i];

            if !k.is_finite() || k < 0.0 {
                return (0.0, 1e-6, 0.0, 1e-6);
            }

            let one_minus_gate = 1.0 - gate;

            let (grad_rate, hess_rate, grad_gate, hess_gate) = if k == 0.0 {
                // Zero branch: A = gate + (1-gate) e^{-λ}.
                let e_neg = (-lambda).exp();
                let a = gate + one_minus_gate * e_neg;
                let gr = one_minus_gate * e_neg / a; // dNLL/dλ
                let gg = -(1.0 - e_neg) / a; // dNLL/dπ
                // d²NLL/dλ² = gr² - gr;  d²NLL/dπ² = gg²
                (gr, gr * gr - gr, gg, gg * gg)
            } else {
                // Positive counts: zero-inflation term vanishes (plain Poisson in
                // λ, plus the ln(1-π) gate term).
                let gr = 1.0 - k / lambda;
                let hr = k / (lambda * lambda);
                let gg = 1.0 / one_minus_gate;
                (gr, hr, gg, gg * gg)
            };

            // Chain rule through the rate response function.
            let dr = rd_r[i];
            let sdr = rsd_r[i];
            let g0 = grad_rate * dr;
            let h0 = (hess_rate * dr * dr + grad_rate * sdr).max(1e-6);

            // Chain rule through the gate response function.
            let dg = rd_g[i];
            let sdg = rsd_g[i];
            let g1 = grad_gate * dg;
            let h1 = (hess_gate * dg * dg + grad_gate * sdg).max(1e-6);

            (g0, h0, g1, h1)
        };

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));
        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");

        const PAR_MIN: usize = 256;
        if n_samples >= PAR_MIN {
            g_slice
                .par_chunks_mut(2)
                .zip(h_slice.par_chunks_mut(2))
                .enumerate()
                .for_each(|(i, (gc, hc))| {
                    let (g0, h0, g1, h1) = compute(i);
                    gc[0] = g0;
                    gc[1] = g1;
                    hc[0] = h0;
                    hc[1] = h1;
                });
        } else {
            g_slice
                .chunks_mut(2)
                .zip(h_slice.chunks_mut(2))
                .enumerate()
                .for_each(|(i, (gc, hc))| {
                    let (g0, h0, g1, h1) = compute(i);
                    gc[0] = g0;
                    gc[1] = g1;
                    hc[0] = h0;
                    hc[1] = h1;
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
            let gate = params[[j, 1]];

            if rate > 0.0 && gate >= 0.0 && gate <= 1.0 {
                if let Ok(poisson_dist) = RandPoisson::new(rate) {
                    for i in 0..n_samples {
                        if rng.random_bool(gate) {
                            result[[i, j]] = 0.0;
                        } else {
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
    use approx::assert_relative_eq;

    #[test]
    fn test_zipoisson_creation() {
        let dist = ZIPoisson::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["rate", "gate"]);
        assert!(dist.is_discrete());
    }

    #[test]
    fn test_zipoisson_log_prob() {
        let dist = ZIPoisson::default();

        let log_p_zero = dist.log_prob_scalar(&[5.0, 0.1], 0.0);
        // pmf(0) = exp(-5), so gate + (1-gate)*exp(-5) = 0.1 + 0.9*exp(-5)
        let expected_zero = (0.1 + 0.9 * (-5.0_f64).exp()).ln();
        assert_relative_eq!(log_p_zero, expected_zero, epsilon = 1e-10);

        let log_p_non_zero = dist.log_prob_scalar(&[5.0, 0.1], 3.0);
        // ln(1-gate) + poisson_ln_pmf(5, 3) = ln(0.9) + (-5 + 3*ln(5) - ln_gamma(4))
        let expected_non_zero = 0.9_f64.ln() + (-5.0 + 3.0 * 5.0_f64.ln() - ln_gamma(4.0));
        assert_relative_eq!(log_p_non_zero, expected_non_zero, epsilon = 1e-10);
    }

    #[test]
    fn test_zipoisson_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;
        use ndarray::array;

        let dist = ZIPoisson::new(
            Stabilization::None,
            ResponseFn::Exp,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        );
        // Mix of zero and positive counts to exercise both NLL branches.
        let predictions = array![[1.0, 0.0], [0.5, -0.5], [1.5, 0.5], [0.8, 0.2]];
        let targets = array![0.0, 2.0, 0.0, 4.0];
        let target = ResponseData::Univariate(&targets.view());

        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");
        let numerical = dist
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("Should return numerical gradients");

        for i in 0..targets.len() {
            for j in 0..2 {
                assert_relative_eq!(
                    analytical.0[[i, j]],
                    numerical.0[[i, j]],
                    epsilon = 1e-3
                );
            }
        }
    }
}
