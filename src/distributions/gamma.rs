//! Gamma distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Gamma as RandGamma};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::constants::trigamma;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gamma {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Gamma {
    pub fn new(
        stabilization: Stabilization,
        response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("concentration", response_fn),
            DistributionParam::new("rate", response_fn),
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

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let shape = params[0]; // concentration
        let rate = params[1];

        if shape <= 0.0 || rate <= 0.0 || target < 0.0 {
            return f64::NEG_INFINITY;
        }
        if target == 0.0 {
            // shape=1 is the exponential special case
            if (shape - 1.0).abs() < f64::EPSILON {
                return rate.ln() - rate * target;
            }
            return f64::NEG_INFINITY;
        }
        if target.is_infinite() {
            return f64::NEG_INFINITY;
        }

        // ln_pdf = shape * ln(rate) + (shape - 1) * ln(x) - rate * x - ln_gamma(shape)
        shape * rate.ln() + (shape - 1.0) * target.ln() - rate * target - ln_gamma(shape)
    }
}

#[typetag::serde]
impl Distribution for Gamma {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Gamma"
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
                // Inline NLL to avoid per-element function call + [[i,j]] indexing overhead.
                // NLL_i = -shape*ln(rate) - (shape-1)*ln(y) + rate*y + ln_gamma(shape)
                let conc_col = params.column(0);
                let rate_col = params.column(1);
                crate::distributions::util::par_sum(y.len(), |i| {
                    let y_val = y[i];
                    if y_val < 0.0 {
                        f64::INFINITY
                    } else {
                        -self.log_prob_scalar(&[conc_col[i], rate_col[i]], y_val)
                    }
                })
            }
            ResponseData::Multivariate(_) => panic!("Gamma is a univariate distribution."),
        }
    }

    /// Analytical gradients for Gamma distribution.
    ///
    /// NLL = ln_gamma(α) - α*ln(β) - (α-1)*ln(y) + β*y
    ///
    /// Gradients w.r.t. distribution parameters:
    /// - ∂NLL/∂α = digamma(α) - ln(β) - ln(y)
    /// - ∂NLL/∂β = -α/β + y
    ///
    /// Hessians w.r.t. distribution parameters:
    /// - ∂²NLL/∂α² = trigamma(α)
    /// - ∂²NLL/∂β² = α/β²
    ///
    /// Chain rule applied for response function: ∂NLL/∂pred = ∂NLL/∂param * ∂param/∂pred
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

        let conc_response_fn = &self.params[0].response_fn;
        let rate_response_fn = &self.params[1].response_fn;

        let t_conc = transformed.column(0);
        let t_rate = transformed.column(1);
        let p_conc = predictions.column(0);
        let p_rate = predictions.column(1);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        if n_samples >= 4096 {
            // Pre-compute batch derivatives for both parameters
            let (rd_conc, rsd_conc) = conc_response_fn.derivative_batches_from_transformed(&p_conc, &t_conc);
            let (rd_rate, rsd_rate) = rate_response_fn.derivative_batches_from_transformed(&p_rate, &t_rate);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let alpha = t_conc[i].max(1e-6);
                let beta = t_rate[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || !yi.is_finite() {
                    return (0.0, 0.0, 0.0, 0.0);
                }

                let ln_y = yi.ln();
                let ln_beta = beta.ln();

                let grad_alpha = digamma(alpha) - ln_beta - ln_y;
                let hess_alpha = trigamma(alpha);
                let grad_beta = -alpha / beta + yi;
                let hess_beta = alpha / (beta * beta);

                let rd0 = rd_conc[i];
                let rsd0 = rsd_conc[i];
                let g0 = grad_alpha * rd0;
                let h0 = hess_alpha * rd0 * rd0 + grad_alpha * rsd0;

                let rd1 = rd_rate[i];
                let rsd1 = rsd_rate[i];
                let g1 = grad_beta * rd1;
                let h1 = hess_beta * rd1 * rd1 + grad_beta * rsd1;

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
                let alpha = t_conc[i].max(1e-6);
                let beta = t_rate[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || !yi.is_finite() {
                    gradients[[i, 0]] = 0.0;
                    hessians[[i, 0]] = 1e-6;
                    gradients[[i, 1]] = 0.0;
                    hessians[[i, 1]] = 1e-6;
                    continue;
                }

                let ln_y = yi.ln();
                let ln_beta = beta.ln();

                let grad_alpha = digamma(alpha) - ln_beta - ln_y;
                let hess_alpha = trigamma(alpha);
                let grad_beta = -alpha / beta + yi;
                let hess_beta = alpha / (beta * beta);

                let rd0 = conc_response_fn.derivative(p_conc[i]);
                let rsd0 = conc_response_fn.second_derivative(p_conc[i]);
                let g0 = grad_alpha * rd0;
                let h0 = hess_alpha * rd0 * rd0 + grad_alpha * rsd0;

                let rd1 = rate_response_fn.derivative(p_rate[i]);
                let rsd1 = rate_response_fn.second_derivative(p_rate[i]);
                let g1 = grad_beta * rd1;
                let h1 = hess_beta * rd1 * rd1 + grad_beta * rsd1;

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
            let concentration = params[[j, 0]];
            let rate = params[[j, 1]];

            if concentration > 0.0 && rate > 0.0 {
                // rand_distr::Gamma uses shape and scale (1/rate) parameterization
                if let Ok(dist) = RandGamma::new(concentration, 1.0 / rate) {
                    for i in 0..n_samples {
                        result[[i, j]] = dist.sample(&mut rng);
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
    fn test_gamma_creation() {
        let dist = Gamma::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["concentration", "rate"]);
        assert!(!dist.should_initialize());
    }

    #[test]
    fn test_gamma_log_prob() {
        let dist = Gamma::default();
        let log_p = dist.log_prob_scalar(&[2.0, 1.0], 1.0);
        // Gamma(shape=2, rate=1) at x=1: shape*ln(rate) + (shape-1)*ln(x) - rate*x - ln_gamma(shape)
        // = 2*ln(1) + 1*ln(1) - 1*1 - ln_gamma(2) = 0 + 0 - 1 - 0 = -1.0
        let expected = -1.0;
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_nll() {
        let dist = Gamma::default();
        let params = array![[2.0, 1.0], [2.0, 1.0]];
        let target = array![1.0, 1.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        // Gamma(shape=2, rate=1) at x=1: ln_pdf = -1.0
        let expected_single = 1.0;
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }

    #[test]
    fn test_trigamma() {
        // trigamma(1) = π²/6
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert_relative_eq!(trigamma(1.0), expected, epsilon = 1e-8);
        // trigamma(2) = π²/6 - 1
        assert_relative_eq!(trigamma(2.0), expected - 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_gamma_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        let dist = Gamma::default();
        let predictions = array![[0.5, 0.3], [-0.2, 0.8], [1.0, -0.5]];
        let targets = array![1.5, 0.8, 2.0];
        let target = ResponseData::Univariate(&targets.view());

        // Get analytical gradients
        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");

        // Get numerical gradients (call the numerical method directly)
        let numerical = dist
            .numerical_gradients_hessians(&predictions.view(), &transformed.view(), &target)
            .expect("Should return numerical gradients");

        // Compare gradients
        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(analytical.0[[i, j]], numerical.0[[i, j]], epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn test_gamma_sample() {
        let dist = Gamma::default();
        let params = array![[2.0, 1.0], [3.0, 2.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        assert_eq!(samples.dim(), (1000, 2));

        let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_0, 2.0, epsilon = 0.2);

        let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_1, 1.5, epsilon = 0.2);
    }
}
