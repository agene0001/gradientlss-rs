//! Zero-Inflated Negative Binomial distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Gamma as RandGamma, Poisson as RandPoisson};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::constants::trigamma;
use crate::distributions::negative_binomial::nb_psi_diff;

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
    fn nb_ln_pmf(r: f64, p: f64, k: f64) -> f64 {
        ln_gamma(r + k)
            - ln_gamma(r)
            - crate::constants::ln_factorial(k).unwrap_or_else(|| ln_gamma(k + 1.0))
            + r * p.ln()
            + k * (-p).ln_1p()
    }

    /// Helper method for scalar log probability (inlined formula)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let r = params[0]; // total_count
        let probs = params[1];
        let gate = params[2];

        if r <= 0.0 || probs <= 0.0 || probs >= 1.0 || gate < 0.0 || gate > 1.0 {
            return f64::NEG_INFINITY;
        }

        // Counts outside the support are -inf. Previously the target was
        // truncated via `as u64`, which silently mapped negatives to 0 and
        // dropped the fractional part; use it as-is (the NB ln_pmf is
        // lgamma-based and valid for non-integer counts).
        if !target.is_finite() || target < 0.0 {
            return f64::NEG_INFINITY;
        }

        // PyTorch probs → statrs p = 1 - probs
        let p = 1.0 - probs;

        if target == 0.0 {
            // NB pmf(0) = p^r, so gate + (1-gate)*p^r
            (gate + (1.0 - gate) * p.powf(r)).ln()
        } else {
            (1.0 - gate).ln() + Self::nb_ln_pmf(r, p, target)
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

    /// Sampling uses a discrete zero-inflation gate and Gamma–Poisson draw, which is not a
    /// smooth function of the parameters under a fixed seed — CRPS finite
    /// differences through it are meaningless (torch has no rsample here either).
    fn has_reparameterizable_sampler(&self) -> bool {
        false
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
                crate::distributions::util::par_nansum(y.len(), |i| {
                    -self.log_prob_scalar(&[col0[i], col1[i], col2[i]], y[i])
                })
            }
            ResponseData::Multivariate(_) => panic!("ZINB is a univariate distribution."),
        }
    }

    /// Analytical gradients for the Zero-Inflated Negative Binomial.
    ///
    /// Parameters are `total_count` (r), `probs`, and `gate`. Internally the NB
    /// uses `p = 1 - probs`, so NB pmf(0) = pᵣ. The NLL splits on the count.
    ///
    /// For k > 0 the zero-inflation contributes only `-ln(1-gate)`, so r and probs
    /// take their plain-NB gradients/Hessians (reusing [`nb_psi_diff`] for the
    /// digamma/trigamma differences), while the gate adds
    /// `dNLL/dgate = 1/(1-gate)`, `d²NLL/dgate² = 1/(1-gate)²`.
    ///
    /// For k = 0, with `B = pʳ` and `A = gate + (1-gate) B`:
    /// - dNLL/dr     = -(1-gate) B ln p / A
    /// - dNLL/dprobs =  (1-gate) r B / (p A)
    /// - dNLL/dgate  = -(1 - B) / A
    /// - d²NLL/dr²     = (dNLL/dr)²    - (1-gate) B ln²p / A
    /// - d²NLL/dprobs² = (dNLL/dprobs)² - (1-gate) r (r-1) B / (p² A)
    /// - d²NLL/dgate²  = (dNLL/dgate)²
    ///
    /// each chain-ruled through the response functions. Replaces the numerical
    /// finite-difference fallback (two perturbed `log_prob` evaluations per
    /// parameter per sample, each with three `ln_gamma` calls).
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
        let gate_response_fn = &self.params[2].response_fn;

        let t_r = transformed.column(0);
        let t_probs = transformed.column(1);
        let t_gate = transformed.column(2);
        let p_r = predictions.column(0);
        let p_probs = predictions.column(1);
        let p_gate = predictions.column(2);

        // Batch response-fn derivatives (auto-vectorized), shared by both paths.
        let (rd_r, rsd_r) = r_response_fn.derivative_batches_from_transformed(&p_r, &t_r);
        let (rd_p, rsd_p) =
            probs_response_fn.derivative_batches_from_transformed(&p_probs, &t_probs);
        let (rd_g, rsd_g) = gate_response_fn.derivative_batches_from_transformed(&p_gate, &t_gate);

        let compute = |i: usize| -> (f64, f64, f64, f64, f64, f64) {
            let r = t_r[i].max(1e-6);
            let probs = t_probs[i].clamp(1e-6, 1.0 - 1e-6);
            let gate = t_gate[i].clamp(1e-6, 1.0 - 1e-6);
            let k = y[i];

            if !k.is_finite() || k < 0.0 {
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }

            let p = 1.0 - probs; // statrs convention; NB pmf(0) = p^r
            let one_minus_gate = 1.0 - gate;

            let (grad_r, hess_r, grad_probs, hess_probs, grad_gate, hess_gate) = if k == 0.0 {
                // Zero branch: A = gate + (1-gate) * p^r.
                let lnp = p.ln();
                let b = p.powf(r);
                let a = gate + one_minus_gate * b;

                let gr = -one_minus_gate * b * lnp / a;
                let gp = one_minus_gate * r * b / (p * a);
                let gg = -(1.0 - b) / a;

                let hr = gr * gr - one_minus_gate * b * lnp * lnp / a;
                let hp = gp * gp - one_minus_gate * r * (r - 1.0) * b / (p * p * a);
                let hg = gg * gg;

                (gr, hr, gp, hp, gg, hg)
            } else {
                // Positive counts: plain NB in (r, probs), plus the ln(1-gate) term.
                let (grad_r_psi, hess_r) = nb_psi_diff(r, k).unwrap_or_else(|| {
                    (-digamma(r + k) + digamma(r), -trigamma(r + k) + trigamma(r))
                });
                let gr = grad_r_psi - p.ln();
                let gp = r / p - k / probs;
                let hp = r / (p * p) + k / (probs * probs);
                let gg = 1.0 / one_minus_gate;
                let hg = gg * gg;

                (gr, hess_r, gp, hp, gg, hg)
            };

            // Chain rule through each parameter's response function.
            let dr = rd_r[i];
            let sdr = rsd_r[i];
            let g0 = grad_r * dr;
            let h0 = hess_r * dr * dr + grad_r * sdr;

            let dp = rd_p[i];
            let sdp = rsd_p[i];
            let g1 = grad_probs * dp;
            let h1 = hess_probs * dp * dp + grad_probs * sdp;

            let dg = rd_g[i];
            let sdg = rsd_g[i];
            let g2 = grad_gate * dg;
            let h2 = hess_gate * dg * dg + grad_gate * sdg;

            (g0, h0, g1, h1, g2, h2)
        };

        let mut gradients = Array2::zeros((n_samples, 3));
        let mut hessians = Array2::zeros((n_samples, 3));
        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");

        const PAR_MIN: usize = 256;
        if n_samples >= PAR_MIN {
            g_slice
                .par_chunks_mut(3)
                .zip(h_slice.par_chunks_mut(3))
                .enumerate()
                .for_each(|(i, (gc, hc))| {
                    let (g0, h0, g1, h1, g2, h2) = compute(i);
                    gc[0] = g0;
                    gc[1] = g1;
                    gc[2] = g2;
                    hc[0] = h0;
                    hc[1] = h1;
                    hc[2] = h2;
                });
        } else {
            g_slice
                .chunks_mut(3)
                .zip(h_slice.chunks_mut(3))
                .enumerate()
                .for_each(|(i, (gc, hc))| {
                    let (g0, h0, g1, h1, g2, h2) = compute(i);
                    gc[0] = g0;
                    gc[1] = g1;
                    gc[2] = g2;
                    hc[0] = h0;
                    hc[1] = h1;
                    hc[2] = h2;
                });
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

    #[test]
    fn test_zinb_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;
        use ndarray::array;

        let dist = ZINB::new(
            Stabilization::None,
            ResponseFn::Exp,
            ResponseFn::Sigmoid,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            false,
        );
        // Mix of zero and positive counts to exercise both NLL branches.
        let predictions = array![
            [0.5, 0.0, 0.0],
            [0.8, -0.5, 0.3],
            [1.0, 0.5, -0.2],
            [0.3, 0.2, 0.1]
        ];
        let targets = array![0.0, 2.0, 0.0, 5.0];
        let target = ResponseData::Univariate(&targets.view());

        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");
        let numerical = dist
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("Should return numerical gradients");

        for i in 0..targets.len() {
            for j in 0..3 {
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-2);
                // True (unfloored) Hessians: both paths must agree on value AND sign.
                assert_relative_eq!(
                    analytical.1[[i, j]],
                    numerical.1[[i, j]],
                    epsilon = 1e-2,
                    max_relative = 1e-2
                );
            }
        }
    }
}
