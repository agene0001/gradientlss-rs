//! Weibull distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Weibull as RandWeibull};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, Weibull as StatrsWeibull};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weibull {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Weibull {
    pub fn new(
        stabilization: Stabilization,
        scale_response_fn: ResponseFn,
        concentration_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("scale", scale_response_fn),
            DistributionParam::new("concentration", concentration_response_fn),
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
            LossFn::Nll,
            false,
        )
    }

    /// Helper method for scalar log probability
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let scale = params[0];
        let concentration = params[1];

        if scale <= 0.0 || concentration <= 0.0 {
            return f64::NEG_INFINITY;
        }

        match StatrsWeibull::new(concentration, scale) {
            Ok(dist) => dist.ln_pdf(target),
            Err(_) => f64::NEG_INFINITY,
        }
    }
}

#[typetag::serde]
impl Distribution for Weibull {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Weibull"
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
                let mut total = 0.0;
                for (i, &y_val) in y.iter().enumerate() {
                    if y_val < 0.0 {
                        return f64::INFINITY;
                    }
                    let p = vec![params[[i, 0]], params[[i, 1]]];
                    total -= self.log_prob_scalar(&p, y_val);
                }
                total
            }
            ResponseData::Multivariate(_) => panic!("Weibull is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let scale = params[[j, 0]];
            let concentration = params[[j, 1]];

            if scale > 0.0 && concentration > 0.0 {
                // rand_distr::Weibull uses (scale, shape) parameterization
                if let Ok(dist) = RandWeibull::new(scale, concentration) {
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
    fn test_weibull_creation() {
        let dist = Weibull::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["scale", "concentration"]);
    }

    #[test]
    fn test_weibull_log_prob() {
        let dist = Weibull::default();
        let log_p = dist.log_prob_scalar(&[1.0, 2.0], 1.0);
        let expected = StatrsWeibull::new(2.0, 1.0).unwrap().ln_pdf(1.0);
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_weibull_nll() {
        let dist = Weibull::default();
        let params = array![[1.0, 2.0], [1.0, 2.0]];
        let target = array![1.0, 1.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -StatrsWeibull::new(2.0, 1.0).unwrap().ln_pdf(1.0);
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}
