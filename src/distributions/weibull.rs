//! Weibull distribution implementation.

use super::base::{Distribution, DistributionParam, HessianMode, LossFn, Stabilization};
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Weibull as RandWeibull};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
// Direct formula - no statrs constructor needed

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weibull {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
    /// Curvature handed to the booster (see [`HessianMode`]). Files written
    /// before the field existed load as `Observed` (the historical curvature).
    #[serde(default = "Weibull::legacy_hessian_mode")]
    hessian_mode: HessianMode,
    /// Without `initialize`: start the scale so the start distribution's
    /// median is the labels' median (shape stays at the raw 0.5). Only
    /// affects training.
    #[serde(default)]
    location_start: bool,
    /// Optional cap on the shape k (rows above it get k = cap and no shape
    /// gradient). `None` (default) = no cap.
    #[serde(default)]
    max_concentration: Option<f64>,
}

/// Fisher information of log k: π²/6 + (1 − γ)² (constant). The Fisher of
/// log λ is k², and their correlation is a constant −(1−γ)/√(π²/6+(1−γ)²)
/// = −0.313 (research cycle 7 audit: mild, no reparameterisation needed).
const WEIBULL_LOG_SHAPE_FISHER: f64 = 1.823_680_660_852_879_3;

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
            hessian_mode: Self::default_hessian_mode(),
            location_start: false,
            max_concentration: None,
        }
    }

    /// Cap the shape k at `cap` (see the field docs).
    pub fn with_max_concentration(mut self, cap: f64) -> Self {
        self.max_concentration = Some(cap);
        self
    }

    pub fn max_concentration(&self) -> Option<f64> {
        self.max_concentration
    }

    /// See the `location_start` field.
    pub fn with_location_start(mut self, on: bool) -> Self {
        self.location_start = on;
        self
    }

    pub fn location_start(&self) -> bool {
        self.location_start
    }

    fn legacy_hessian_mode() -> HessianMode {
        HessianMode::Observed
    }

    /// `Damped` since 2026-11-10 (research cycle 7): better held-out NLL than
    /// the observed curvature on every dataset measured (power_plant −0.15,
    /// kin8nm −0.08, synthetic −0.05, naval −0.05; 5/5 seeds except naval 4/5).
    fn default_hessian_mode() -> HessianMode {
        HessianMode::Damped
    }

    /// Choose the curvature handed to the Newton step (see [`HessianMode`]).
    pub fn with_hessian_mode(mut self, mode: HessianMode) -> Self {
        self.hessian_mode = mode;
        self
    }

    pub fn hessian_mode(&self) -> HessianMode {
        self.hessian_mode
    }

    fn curvature(&self, obs: f64, fisher: f64) -> f64 {
        match self.hessian_mode {
            HessianMode::Observed => obs,
            HessianMode::Fisher => fisher,
            HessianMode::Damped => obs.max(fisher),
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

    /// Helper method for scalar log probability (inlined formula, no statrs constructor)
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let scale = params[0]; // lambda
        let k = params[1]; // concentration/shape

        if scale <= 0.0 || k <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if target < 0.0 || target.is_infinite() {
            return f64::NEG_INFINITY;
        }
        if target == 0.0 {
            if (k - 1.0).abs() < f64::EPSILON {
                return -scale.ln(); // exponential special case
            }
            return f64::NEG_INFINITY;
        }

        // ln_pdf = ln(k) + (k-1)*ln(x/λ) - (x/λ)^k - ln(λ)
        k.ln() + (k - 1.0) * (target / scale).ln() - (target / scale).powf(k) - scale.ln()
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

    /// The scale λ carries the location.
    fn location_param_index(&self) -> Option<usize> {
        Some(0)
    }

    fn transform_params(&self, predictions: &ArrayView2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(predictions.dim());
        for (i, param) in self.params.iter().enumerate() {
            let col = predictions.column(i);
            let mut out_col = result.column_mut(i);
            param.response_fn.apply_into(&col, &mut out_col);
        }
        if let Some(cap) = self.max_concentration {
            result.column_mut(1).mapv_inplace(|k| k.min(cap));
        }
        result
    }

    /// Measured recipe (research cycle 8): watch the shape (naval −0.81,
    /// kin8nm −0.035, 5/5; power_plant −0.013 over 10 seeds, one test-only
    /// outlier; simulation unchanged). NLL loss only.
    fn default_param_early_stopping_params(&self) -> Option<Vec<usize>> {
        (self.loss_fn == LossFn::Nll).then(|| vec![1])
    }

    fn default_start_values(&self, target: &ResponseData) -> Option<ndarray::Array1<f64>> {
        if !self.location_start || self.params[0].response_fn != ResponseFn::Exp {
            return None;
        }
        let ResponseData::Univariate(y) = target else { return None };
        let mut v: Vec<f64> = y.iter().copied().filter(|t| t.is_finite() && *t > 0.0).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len();
        let med = if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) };
        // median of Weibull(λ, k) = λ (ln 2)^{1/k}; k at its raw start
        let k0 = self.params[1].response_fn.apply_scalar(0.5);
        let lam = med / std::f64::consts::LN_2.powf(1.0 / k0);
        Some(ndarray::Array1::from_vec(vec![lam.ln(), 0.5]))
    }

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_scalar(params, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                let scale_col = params.column(0);
                let conc_col = params.column(1);
                crate::distributions::util::par_nansum(y.len(), |i| {
                    let y_val = y[i];
                    if y_val < 0.0 {
                        f64::INFINITY
                    } else {
                        -self.log_prob_scalar(&[scale_col[i], conc_col[i]], y_val)
                    }
                })
            }
            ResponseData::Multivariate(_) => panic!("Weibull is a univariate distribution."),
        }
    }

    /// Analytical gradients for Weibull distribution.
    ///
    /// NLL = -ln(k) - (k-1)*ln(x/λ) + (x/λ)^k + ln(λ)
    ///
    /// Let t = x/λ. Gradients w.r.t. distribution parameters:
    /// - dNLL/dλ = k*(1 - t^k)/λ
    /// - dNLL/dk = -1/k + ln(t)*(t^k - 1)
    ///
    /// Hessians:
    /// - d²NLL/dλ² = k*((k+1)*t^k - 1) / λ²
    /// - d²NLL/dk² = 1/k² + t^k * ln(t)²
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

        let scale_response_fn = &self.params[0].response_fn;
        let conc_response_fn = &self.params[1].response_fn;

        let t_scale = transformed.column(0);
        let t_conc = transformed.column(1);
        let p_scale = predictions.column(0);
        let p_conc = predictions.column(1);

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));

        if n_samples >= 4096 {
            // Pre-compute batch derivatives for both parameters
            let (rd_s, rsd_s) =
                scale_response_fn.derivative_batches_from_transformed(&p_scale, &t_scale);
            let (rd_k, rsd_k) =
                conc_response_fn.derivative_batches_from_transformed(&p_conc, &t_conc);

            let compute_sample = |i: usize| -> (f64, f64, f64, f64) {
                let lambda = t_scale[i].max(1e-6);
                let k = t_conc[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || !yi.is_finite() {
                    return (0.0, 0.0, 0.0, 0.0);
                }

                let t = yi / lambda;
                let ln_t = t.ln();
                let t_k = t.powf(k);

                let grad_lambda = k * (1.0 - t_k) / lambda;
                let hess_lambda = k * ((k + 1.0) * t_k - 1.0) / (lambda * lambda);
                let grad_k = -1.0 / k + ln_t * (t_k - 1.0);
                let hess_k = 1.0 / (k * k) + t_k * ln_t * ln_t;

                let r0 = rd_s[i];
                let rs0 = rsd_s[i];
                let g0 = grad_lambda * r0;
                let h0 = self.curvature(
                    hess_lambda * r0 * r0 + grad_lambda * rs0,
                    k * k / (lambda * lambda) * r0 * r0,
                );

                let r1 = rd_k[i];
                let rs1 = rsd_k[i];
                let g1 = grad_k * r1;
                let h1 = self.curvature(
                    hess_k * r1 * r1 + grad_k * rs1,
                    WEIBULL_LOG_SHAPE_FISHER / (k * k) * r1 * r1,
                );

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
                let lambda = t_scale[i].max(1e-6);
                let k = t_conc[i].max(1e-6);
                let yi = y[i];

                if yi <= 0.0 || !yi.is_finite() {
                    gradients[[i, 0]] = 0.0;
                    hessians[[i, 0]] = 1e-6;
                    gradients[[i, 1]] = 0.0;
                    hessians[[i, 1]] = 1e-6;
                    continue;
                }

                let t = yi / lambda;
                let ln_t = t.ln();
                let t_k = t.powf(k);

                let grad_lambda = k * (1.0 - t_k) / lambda;
                let hess_lambda = k * ((k + 1.0) * t_k - 1.0) / (lambda * lambda);
                let grad_k = -1.0 / k + ln_t * (t_k - 1.0);
                let hess_k = 1.0 / (k * k) + t_k * ln_t * ln_t;

                let r0 = scale_response_fn.derivative(p_scale[i]);
                let rs0 = scale_response_fn.second_derivative(p_scale[i]);
                let g0 = grad_lambda * r0;
                let h0 = self.curvature(
                    hess_lambda * r0 * r0 + grad_lambda * rs0,
                    k * k / (lambda * lambda) * r0 * r0,
                );

                let r1 = conc_response_fn.derivative(p_conc[i]);
                let rs1 = conc_response_fn.second_derivative(p_conc[i]);
                let g1 = grad_k * r1;
                let h1 = self.curvature(
                    hess_k * r1 * r1 + grad_k * rs1,
                    WEIBULL_LOG_SHAPE_FISHER / (k * k) * r1 * r1,
                );

                gradients[[i, 0]] = g0;
                hessians[[i, 0]] = h0;
                gradients[[i, 1]] = g1;
                hessians[[i, 1]] = h1;
            }
        }

        if let Some(cap) = self.max_concentration {
            // a capped row's shape does not respond to its raw score
            for i in 0..n_samples {
                if t_conc[i] >= cap {
                    gradients[[i, 1]] = 0.0;
                }
            }
        }
        Some((gradients, hessians))
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
        // Weibull(shape=2, scale=1) at x=1: ln(2) + 1*ln(1) - 1^2 - ln(1) = ln(2) - 1
        let expected = 2.0_f64.ln() - 1.0;
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_weibull_analytical_vs_numerical_gradients() {
        use crate::distributions::base::Distribution;

        // finite differences see the observed curvature
        let dist = Weibull::default().with_hessian_mode(HessianMode::Observed);
        let predictions = array![[0.5, 0.3], [0.8, 0.5], [0.3, 0.8]];
        let targets = array![1.5, 0.8, 2.0];
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
                // Hessians must agree too — the scale Hessian previously used
                // k*((k+1)*t^k - k)/λ² (only correct at k=1) instead of
                // k*((k+1)*t^k - 1)/λ².
                assert_relative_eq!(
                    analytical.1[[i, j]],
                    numerical.1[[i, j]],
                    epsilon = 1e-2,
                    max_relative = 1e-2
                );
            }
        }
    }

    #[test]
    fn test_weibull_nll() {
        let dist = Weibull::default();
        let params = array![[1.0, 2.0], [1.0, 2.0]];
        let target = array![1.0, 1.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected_single = -(2.0_f64.ln() - 1.0);
        assert_relative_eq!(nll, 2.0 * expected_single, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod hessian_mode_tests {
    use super::*;
    use ndarray::Array1;

    /// Under Exp responses E[observed raw curvature] = (k², π²/6 + (1−γ)²).
    #[test]
    fn fisher_is_the_expected_observed() {
        let (lam, k) = (3.0_f64, 1.7_f64);
        let n = 200_000;
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let w = RandWeibull::new(lam, k).unwrap();
        let y = Array1::from_iter((0..n).map(|_| w.sample(&mut rng)));
        let pred = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { lam.ln() } else { k.ln() });
        let obs = Weibull::default().with_hessian_mode(HessianMode::Observed);
        let fis = Weibull::default().with_hessian_mode(HessianMode::Fisher);
        let t = obs.transform_params(&pred.view());
        let target = ResponseData::Univariate(&y.view());
        let (_, h_obs) = obs.analytical_gradients(&pred.view(), &t.view(), &target).unwrap();
        let (_, h_fis) = fis.analytical_gradients(&pred.view(), &t.view(), &target).unwrap();
        assert!((h_fis[[0, 0]] / (k * k) - 1.0).abs() < 1e-4 && (h_fis[[0, 1]] / WEIBULL_LOG_SHAPE_FISHER - 1.0).abs() < 1e-4, "{} {}", h_fis[[0, 0]], h_fis[[0, 1]]);
        for j in 0..2 {
            let m = h_obs.column(j).mean().unwrap();
            assert!((m / h_fis[[0, j]] - 1.0).abs() < 0.02, "param {j}: {m} vs {}", h_fis[[0, j]]);
        }
        let damped = Weibull::default().with_hessian_mode(HessianMode::Damped);
        let (_, h_d) = damped.analytical_gradients(&pred.view(), &t.view(), &target).unwrap();
        assert!(h_d.iter().zip(h_fis.iter()).all(|(d, f)| d >= f));
        assert!(h_obs.iter().any(|v| *v < 0.0), "observed curvature goes negative on some rows");
    }

    #[test]
    fn damped_is_the_default() {
        assert_eq!(Weibull::default().hessian_mode(), HessianMode::Damped);
    }

    #[test]
    fn serde_default_is_observed() {
        let mut v = serde_json::to_value(Weibull::default().with_hessian_mode(HessianMode::Damped)).unwrap();
        v.as_object_mut().unwrap().remove("hessian_mode");
        let back: Weibull = serde_json::from_value(v).unwrap();
        assert_eq!(back.hessian_mode(), HessianMode::Observed);
    }
}

#[cfg(test)]
mod location_start_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn location_start_matches_the_median() {
        let y = array![420.0, 450.0, 470.0];
        let d = Weibull::default().with_location_start(true);
        let sv = d.default_start_values(&ResponseData::Univariate(&y.view())).unwrap();
        let t = d.transform_params(&array![[sv[0], sv[1]]].view());
        let (lam, k) = (t[[0, 0]], t[[0, 1]]);
        assert!((lam * std::f64::consts::LN_2.powf(1.0 / k) - 450.0).abs() < 1e-3, "{lam} {k}");
        assert_eq!(sv[1], 0.5);
        assert!(Weibull::default().default_start_values(&ResponseData::Univariate(&y.view())).is_none());
    }
}

#[cfg(test)]
mod cap_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cap_clamps_the_shape_and_zeroes_its_gradient() {
        let d = Weibull::default().with_max_concentration(5.0);
        let pred = array![[1.0, 3.0], [1.0, 0.5]]; // k = e³ ≈ 20 (capped), e^0.5 ≈ 1.65
        let t = d.transform_params(&pred.view());
        assert_eq!(t[[0, 1]], 5.0);
        assert!((t[[1, 1]] - 0.5f64.exp()).abs() < 1e-5);
        let y = array![2.0, 2.0];
        let (g, _) = d.analytical_gradients(&pred.view(), &t.view(), &ResponseData::Univariate(&y.view())).unwrap();
        assert_eq!(g[[0, 1]], 0.0);
        assert!(g[[1, 1]] != 0.0);
        assert_eq!(Weibull::default().max_concentration(), None);
    }
}
