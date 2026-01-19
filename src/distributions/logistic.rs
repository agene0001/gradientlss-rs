//! Logistic distribution implementation.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logistic {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
}

impl Logistic {
    pub fn new(
        stabilization: Stabilization,
        scale_response_fn: ResponseFn,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let params = vec![
            DistributionParam::new("loc", ResponseFn::Identity),
            DistributionParam::new("scale", scale_response_fn),
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

    /// Helper method for scalar log probability
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let loc = params[0];
        let scale = params[1];

        if scale <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let z = (target - loc) / scale;
        -z - scale.ln() - 2.0 * (1.0 + (-z).exp()).ln()
    }
}

#[typetag::serde]
impl Distribution for Logistic {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Logistic"
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
                    let p = vec![params[[i, 0]], params[[i, 1]]];
                    total -= self.log_prob_scalar(&p, y_val);
                }
                total
            }
            ResponseData::Multivariate(_) => panic!("Logistic is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_obs, n_samples));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let loc = params[[j, 0]];
            let scale = params[[j, 1]];

            if scale > 0.0 {
                for i in 0..n_samples {
                    let u: f64 = rng.random_range(0.0..1.0);
                    result[[j, i]] = loc + scale * (u / (1.0 - u)).ln();
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
    use ndarray::array;

    #[test]
    fn test_logistic_creation() {
        let dist = Logistic::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
    }

    #[test]
    fn test_logistic_log_prob() {
        let dist = Logistic::default();
        let log_p = dist.log_prob_scalar(&[0.0, 1.0], 0.0);
        let expected = -2.0 * (2.0f64).ln(); // -ln(4)
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_logistic_sample() {
        let dist = Logistic::default();
        let params = array![[0.0, 1.0], [2.0, 3.0]];
        let samples = dist.sample(&params.view(), 1000, 123);

        assert_eq!(samples.dim(), (2, 1000));

        let mean_0: f64 = samples.row(0).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_0, 0.0, epsilon = 0.2);

        let mean_1: f64 = samples.row(1).iter().sum::<f64>() / 1000.0;
        assert_relative_eq!(mean_1, 2.0, epsilon = 0.2);
    }
}
