//! Beta distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta as RandBeta, Distribution as RandDistribution};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::constants::trigamma;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Beta {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Beta {
    pub fn new(
        stabilization: Stabilization,
        response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("concentration1", response_fn),
            DistributionParam::new("concentration0", response_fn),
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
        let a = params[0]; // concentration1
        let b = params[1]; // concentration0

        if a <= 0.0 || b <= 0.0 || a.is_infinite() || b.is_infinite() {
            return f64::NEG_INFINITY;
        }
        if !(0.0..=1.0).contains(&target) {
            return f64::NEG_INFINITY;
        }

        // ln_pdf = ln_gamma(a+b) - ln_gamma(a) - ln_gamma(b) + (a-1)*ln(x) + (b-1)*ln(1-x)
        let log_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
        let ln_x = if target == 0.0 {
            if a > 1.0 {
                f64::NEG_INFINITY
            } else if (a - 1.0).abs() < f64::EPSILON {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            (a - 1.0) * target.ln()
        };
        let ln_1mx = if (target - 1.0).abs() < f64::EPSILON {
            if b > 1.0 {
                f64::NEG_INFINITY
            } else if (b - 1.0).abs() < f64::EPSILON {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            (b - 1.0) * (1.0 - target).ln()
        };

        -log_beta + ln_x + ln_1mx
    }
}

#[typetag::serde]
impl Distribution for Beta {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Beta"
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
                let conc1_col = params.column(0);
                let conc0_col = params.column(1);
                crate::distributions::util::par_sum(y.len(), |i| {
                    let y_val = y[i];
                    if y_val < 0.0 || y_val > 1.0 {
                        f64::INFINITY
                    } else {
                        -self.log_prob_scalar(&[conc1_col[i], conc0_col[i]], y_val)
                    }
                })
            }
            ResponseData::Multivariate(_) => panic!("Beta is a univariate distribution."),
        }
    }

    /// Analytical gradients for Beta distribution.
    ///
    /// NLL = ln_gamma(a) + ln_gamma(b) - ln_gamma(a+b) - (a-1)*ln(x) - (b-1)*ln(1-x)
    ///
    /// Gradients w.r.t. distribution parameters:
    /// - dNLL/da = digamma(a) - digamma(a+b) - ln(x)
    /// - dNLL/db = digamma(b) - digamma(a+b) - ln(1-x)
    ///
    /// Hessians w.r.t. distribution parameters:
    /// - d²NLL/da² = trigamma(a) - trigamma(a+b)
    /// - d²NLL/db² = trigamma(b) - trigamma(a+b)
    ///
    /// Chain rule applied for response function: dNLL/dpred = dNLL/dparam * dparam/dpred
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

        let conc1_response_fn = &self.params[0].response_fn;
        let conc0_response_fn = &self.params[1].response_fn;

        let t_conc1 = transformed.column(0);
        let t_conc0 = transformed.column(1);
        let p_conc1 = predictions.column(0);
        let p_conc0 = predictions.column(1);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        if n_samples >= 4096 {
            // Pre-compute batch derivatives for both parameters
            let (rd_a, rsd_a) = conc1_response_fn.derivative_batches(&p_conc1);
            let (rd_b, rsd_b) = conc0_response_fn.derivative_batches(&p_conc0);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let a = t_conc1[i].max(1e-6);
                let b = t_conc0[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || yi >= 1.0 || !yi.is_finite() {
                    return (0.0, 0.0, 0.0, 0.0);
                }

                let ln_y = yi.ln();
                let ln_1my = (1.0 - yi).ln();
                let ab = a + b;
                let digamma_ab = digamma(ab);
                let trigamma_ab = trigamma(ab);

                let grad_a = digamma(a) - digamma_ab - ln_y;
                let hess_a = trigamma(a) - trigamma_ab;
                let grad_b = digamma(b) - digamma_ab - ln_1my;
                let hess_b = trigamma(b) - trigamma_ab;

                let r0 = rd_a[i];
                let rs0 = rsd_a[i];
                let g0 = grad_a * r0;
                let h0 = hess_a * r0 * r0 + grad_a * rs0;

                let r1 = rd_b[i];
                let rs1 = rsd_b[i];
                let g1 = grad_b * r1;
                let h1 = hess_b * r1 * r1 + grad_b * rs1;

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
                let a = t_conc1[i].max(1e-6);
                let b = t_conc0[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || yi >= 1.0 || !yi.is_finite() {
                    gradients[[i, 0]] = 0.0;
                    hessians[[i, 0]] = 1e-6;
                    gradients[[i, 1]] = 0.0;
                    hessians[[i, 1]] = 1e-6;
                    continue;
                }

                let ln_y = yi.ln();
                let ln_1my = (1.0 - yi).ln();
                let ab = a + b;
                let digamma_ab = digamma(ab);
                let trigamma_ab = trigamma(ab);

                let grad_a = digamma(a) - digamma_ab - ln_y;
                let hess_a = trigamma(a) - trigamma_ab;
                let grad_b = digamma(b) - digamma_ab - ln_1my;
                let hess_b = trigamma(b) - trigamma_ab;

                let r0 = conc1_response_fn.derivative(p_conc1[i]);
                let rs0 = conc1_response_fn.second_derivative(p_conc1[i]);
                let g0 = grad_a * r0;
                let h0 = hess_a * r0 * r0 + grad_a * rs0;

                let r1 = conc0_response_fn.derivative(p_conc0[i]);
                let rs1 = conc0_response_fn.second_derivative(p_conc0[i]);
                let g1 = grad_b * r1;
                let h1 = hess_b * r1 * r1 + grad_b * rs1;

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
            let concentration1 = params[[j, 0]];
            let concentration0 = params[[j, 1]];

            if concentration1 > 0.0 && concentration0 > 0.0 {
                if let Ok(dist) = RandBeta::new(concentration1, concentration0) {
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
    fn test_beta_creation() {
        let dist = Beta::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["concentration1", "concentration0"]);
    }

    #[test]
    fn test_beta_log_prob() {
        let dist = Beta::default();
        let log_p = dist.log_prob_scalar(&[2.0, 2.0], 0.5);
        // Beta(2,2) at x=0.5: ln_gamma(4)-ln_gamma(2)-ln_gamma(2)+(2-1)*ln(0.5)+(2-1)*ln(0.5)
        let expected = ln_gamma(4.0) - ln_gamma(2.0) - ln_gamma(2.0)
            + 1.0 * (0.5_f64).ln()
            + 1.0 * (0.5_f64).ln();
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_beta_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        let dist = Beta::default();
        let predictions = array![[0.5, 0.3], [-0.2, 0.8], [1.0, -0.5]];
        let targets = array![0.3, 0.7, 0.5];
        let target = ResponseData::Univariate(&targets.view());

        let transformed = dist.transform_params(&predictions.view());
        let analytical = dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)
            .expect("Should return analytical gradients");

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
    fn test_beta_nll() {
        let dist = Beta::default();
        let params = array![[2.0, 2.0], [2.0, 2.0]];
        let target = array![0.5, 0.5];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(ln_gamma(4.0) - ln_gamma(2.0) - ln_gamma(2.0)
            + 1.0 * (0.5_f64).ln()
            + 1.0 * (0.5_f64).ln());
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}
