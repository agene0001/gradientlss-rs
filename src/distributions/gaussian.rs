//! Gaussian (Normal) distribution implementation.

use super::base::{Distribution, DistributionParam, HessianMode, LossFn, Stabilization};
use crate::constants::LOG_2PI;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Gaussian (Normal) distribution for distributional regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian {
    params: Vec<DistributionParam>,
    stabilization: Stabilization,
    loss_fn: LossFn,
    initialize: bool,
    /// Curvature handed to the booster (see [`HessianMode`]); defaults to
    /// `Damped`. The field only affects training, so models saved before it
    /// existed load fine.
    #[serde(default)]
    hessian_mode: HessianMode,
    /// Per-parameter gradient multipliers (effective per-parameter learning
    /// rates), see `Distribution::param_gradient_scales`. `None` = all ones.
    #[serde(default)]
    param_gradient_scales: Option<Vec<f64>>,
    /// β-NLL row weighting (Seitzer et al. 2022): every row's gradient and
    /// Hessian are multiplied by σ̂^{2β}, σ̂ treated as a constant. β = 0 (or
    /// `None`) is plain NLL; β = 1 gives the location tree MSE-like row
    /// influence; β = 0.5 is the paper's recommendation.
    #[serde(default)]
    beta_nll: Option<f64>,
}

impl Gaussian {
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
            hessian_mode: HessianMode::default(),
            param_gradient_scales: None,
            beta_nll: None,
        }
    }

    /// Enable β-NLL row weighting with exponent `beta` (see the field docs).
    pub fn with_beta_nll(mut self, beta: f64) -> Self {
        self.beta_nll = Some(beta);
        self
    }

    pub fn beta_nll(&self) -> Option<f64> {
        self.beta_nll
    }

    /// Multiply the gradient of each parameter by `scales[j]` (order: loc,
    /// scale): an effective per-parameter learning rate η·scales[j]. Use e.g.
    /// `[1.0, 0.5]` to slow the scale tree relative to the location tree.
    pub fn with_param_gradient_scales(mut self, scales: [f64; 2]) -> Self {
        self.param_gradient_scales = Some(scales.to_vec());
        self
    }

    /// Choose the curvature handed to the Newton step (see [`HessianMode`]).
    /// `Observed` restores the autograd-style `−1/σ² + 3(y−µ)²/σ⁴` scale
    /// Hessian (plus the response chain term); gradients are unaffected.
    pub fn with_hessian_mode(mut self, mode: HessianMode) -> Self {
        self.hessian_mode = mode;
        self
    }

    pub fn hessian_mode(&self) -> HessianMode {
        self.hessian_mode
    }

    pub fn default() -> Self {
        Self::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false)
    }

    /// Compute log probability for a single scalar target.
    fn log_prob_scalar(&self, params: &[f64], target: f64) -> f64 {
        let loc = params[0];
        let scale = params[1];

        if scale <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let z = (target - loc) / scale;
        -0.5 * LOG_2PI - scale.ln() - 0.5 * z * z
    }
}

#[typetag::serde]
impl Distribution for Gaussian {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Gaussian"
    }

    fn param_gradient_scales(&self) -> Option<&[f64]> {
        self.param_gradient_scales.as_deref()
    }

    fn row_gradient_weights(&self, transformed: &ArrayView2<f64>) -> Option<Array1<f64>> {
        let beta = self.beta_nll?;
        if beta == 0.0 {
            return None;
        }
        let two_beta = 2.0 * beta;
        Some(transformed.column(1).mapv(|s| s.max(1e-12).powf(two_beta)))
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

    /// Measured recipe (research cycles 5–6): watch the scale only; the
    /// location stays under the global early stopping. NLL loss only.
    fn default_param_early_stopping_params(&self) -> Option<Vec<usize>> {
        (self.loss_fn() == LossFn::Nll).then(|| vec![1])
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
                let loc_col = params.column(0);
                let scale_col = params.column(1);
                if params.nrows() == 1 {
                    let p = [loc_col[0], scale_col[0]];
                    crate::distributions::util::par_nansum(y.len(), |i| {
                        -self.log_prob_scalar(&p, y[i])
                    })
                } else {
                    crate::distributions::util::par_nansum(y.len(), |i| {
                        -self.log_prob_scalar(&[loc_col[i], scale_col[i]], y[i])
                    })
                }
            }
            ResponseData::Multivariate(_) => panic!("Gaussian is a univariate distribution."),
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for j in 0..n_obs {
            let loc = params[[j, 0]];
            let scale = params[[j, 1]].max(1e-6);

            if let Ok(normal) = Normal::new(loc, scale) {
                for i in 0..n_samples {
                    result[[i, j]] = normal.sample(&mut rng);
                }
            }
        }
        result
    }

    /// Analytical gradients for Gaussian distribution.
    ///
    /// For NLL = -log p(y|μ,σ) = 0.5*log(2π) + log(σ) + 0.5*(y-μ)²/σ²
    ///
    /// Gradients w.r.t. transformed parameters:
    /// - ∂NLL/∂μ = (μ - y) / σ²
    /// - ∂NLL/∂σ = 1/σ - (y - μ)² / σ³
    ///
    /// We apply the chain rule: ∂NLL/∂pred = ∂NLL/∂param * ∂param/∂pred
    fn analytical_gradients(
        &self,
        predictions: &ArrayView2<f64>,
        transformed: &ArrayView2<f64>,
        target: &ResponseData,
    ) -> Option<(Array2<f64>, Array2<f64>)> {
        // Only support NLL for analytical gradients
        if self.loss_fn != LossFn::Nll {
            return None;
        }

        let y = match target {
            ResponseData::Univariate(arr) => arr,
            ResponseData::Multivariate(_) => return None,
        };

        let n_samples = predictions.nrows();

        let scale_response_fn = &self.params[1].response_fn;

        let t_loc = transformed.column(0);
        let t_scale = transformed.column(1);
        let p_scale = predictions.column(1);

        // Batch response-fn derivatives (auto-vectorized, one shared `exp`),
        // hoisted above the branch so the sequential path no longer pays two
        // scalar `exp` calls per sample via `derivative`/`second_derivative`.
        let (resp_deriv, resp_second) =
            scale_response_fn.derivative_batches_from_transformed(&p_scale, &t_scale);
        let hessian_mode = self.hessian_mode;

        let compute = |i: usize| -> (f64, f64, f64, f64) {
            let loc = t_loc[i];
            let scale = t_scale[i].max(1e-6);
            let scale_sq = scale * scale;
            let yi = y[i];
            let diff = yi - loc;
            let diff_sq = diff * diff;
            let g0 = -diff / scale_sq;
            let h0 = 1.0 / scale_sq;
            let grad_scale_param = 1.0 / scale - diff_sq / (scale_sq * scale);
            let rd = resp_deriv[i];
            // Scale curvature per `HessianMode`. Fisher = the EXPECTED
            // information 2/σ² chained through the response derivative (the
            // constant 2 in log σ under `Exp`, the textbook LSS Gaussian).
            // Observed = −1/σ² + 3(y−µ)²/σ⁴ plus the grad·response″ chain term;
            // under `Exp` the two combine to 2z², which is ≈ 0 for a well-fit
            // row (its Newton step g/h then blows up) but large for a row far
            // from the fit (a free trust region). Damped takes the max of the
            // two per row — see HessianMode.
            let fisher = (2.0 / scale_sq) * rd * rd;
            let observed = || {
                (-1.0 / scale_sq + 3.0 * diff_sq / (scale_sq * scale_sq)) * rd * rd
                    + grad_scale_param * resp_second[i]
            };
            let h1 = match hessian_mode {
                HessianMode::Fisher => fisher,
                HessianMode::Observed => observed(),
                HessianMode::Damped => observed().max(fisher),
            };
            let g1 = grad_scale_param * rd;
            (g0, h0, g1, h1)
        };

        let mut gradients = Array2::zeros((n_samples, 2));
        let mut hessians = Array2::zeros((n_samples, 2));
        // C-order rows are [param0, param1] per sample; write straight into the
        // contiguous backing slices (same pattern as NegativeBinomial).
        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");

        // Threshold matches the other analytical paths (256).
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ResponseData;
    use approx::assert_relative_eq;
    use ndarray::array;

    /// The scale Hessian is the Fisher information: with the `Exp` response it
    /// is exactly 2 for every row, whatever the residual (the observed form was
    /// ~0 or negative for small residuals), and the loc Hessian stays 1/σ².
    #[test]
    fn scale_hessian_is_the_fisher_information() {
        let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false)
            .with_hessian_mode(HessianMode::Fisher);
        let y = ndarray::array![1.0, 1.05, 3.0, -2.0];
        let yv = y.view();
        let raw = ndarray::array![[1.0f64, 0.0], [1.0, 0.0], [1.0, 0.7], [1.0, -0.5]];
        let t = dist.transform_params(&raw.view());
        let (_, h) = dist.analytical_gradients(&raw.view(), &t.view(), &ResponseData::Univariate(&yv)).unwrap();
        for i in 0..4 {
            // the Exp response adds EPSILON (sigma = e^eta + eps), so 2*sigma^2/(sigma+eps)^2, not 2 to 1e-9
            assert!((h[[i, 1]] - 2.0).abs() < 1e-5, "row {i}: scale hessian {}", h[[i, 1]]);
            let sigma = t[[i, 1]]; // the transformed scale (e^eta + EPSILON)
            assert!((h[[i, 0]] - 1.0 / (sigma * sigma)).abs() < 1e-9);
        }
    }

    #[test]
    fn test_gaussian_creation() {
        let dist = Gaussian::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
        assert!(!dist.should_initialize());
    }

    #[test]
    fn test_gaussian_log_prob() {
        let dist = Gaussian::default();

        let log_p = dist.log_prob_scalar(&[0.0, 1.0], 0.0);
        assert_relative_eq!(log_p, -0.5 * LOG_2PI, epsilon = 1e-10);

        let log_p_wide = dist.log_prob_scalar(&[0.0, 2.0], 0.0);
        let log_p_narrow = dist.log_prob_scalar(&[0.0, 0.5], 0.0);
        assert!(log_p_narrow > log_p_wide);
    }

    #[test]
    fn test_gaussian_nll() {
        let dist = Gaussian::default();
        let params = array![[0.0, 1.0], [0.0, 1.0]];
        let target = array![0.0, 0.0];
        let target_response = ResponseData::Univariate(&target.view());

        let nll = dist.nll(&params.view(), &target_response);
        let expected = 2.0 * -dist.log_prob_scalar(&[0.0, 1.0], 0.0);
        assert_relative_eq!(nll, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_sample() {
        let dist = Gaussian::default();
        let params = array![[0.0, 1.0], [5.0, 0.5]];
        let samples = dist.sample(&params.view(), 1000, 123);

        assert_eq!(samples.dim(), (1000, 2));

        let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 1000.0;
        let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 1000.0;

        assert_relative_eq!(mean_0, 0.0, epsilon = 0.1);
        assert_relative_eq!(mean_1, 5.0, epsilon = 0.1);
    }

    #[test]
    fn test_gaussian_transform_params() {
        let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        let predictions = array![[0.0, 0.0], [1.0, 1.0]];
        let transformed = dist.transform_params(&predictions.view());

        assert_relative_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-6);
        assert_relative_eq!(transformed[[1, 0]], 1.0, epsilon = 1e-6);

        assert!(transformed[[0, 1]] > 0.0);
        assert!(transformed[[1, 1]] > transformed[[0, 1]]);
    }
}

#[cfg(test)]
mod hessian_mode_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::Distribution as _;

    /// `Observed` mode's scale Hessian averages to the Fisher value 2 (Exp
    /// link) over y ~ N(µ, σ²); gradients are identical between modes.
    #[test]
    fn observed_scale_hessian_has_fisher_expectation() {
        let (mu, sigma) = (0.7, 1.9);
        let n = 400_000;
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let nd = Normal::new(mu, sigma).unwrap();
        let y: Array1<f64> = Array1::from_iter((0..n).map(|_| nd.sample(&mut rng)));
        let preds = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { mu } else { sigma.ln() });
        let fisher = Gaussian::default().with_hessian_mode(HessianMode::Fisher);
        let observed = Gaussian::default().with_hessian_mode(HessianMode::Observed);
        let transformed = fisher.transform_params(&preds.view());
        let target = ResponseData::Univariate(&y.view());
        let (g_f, h_f) = fisher
            .analytical_gradients(&preds.view(), &transformed.view(), &target)
            .unwrap();
        let (g_o, h_o) = observed
            .analytical_gradients(&preds.view(), &transformed.view(), &target)
            .unwrap();
        assert_eq!(g_f, g_o);
        assert_eq!(h_f.column(0), h_o.column(0)); // loc Hessian is the same in both
        assert!((h_f[[0, 1]] - 2.0).abs() < 1e-5);
        let mean_obs = h_o.column(1).mean().unwrap();
        assert!((mean_obs - 2.0).abs() < 0.02, "E[observed scale h] = {mean_obs}");
        // Under the Exp link the chain term cancels the −1: observed raw-space
        // h = (−1 + 3z²) + (1 − z²) = 2z² ≥ 0 — never negative, but ≈ 0 for
        // small residuals, which is what makes the Newton ratio g/h blow up.
        assert!(h_o.column(1).iter().all(|&h| h >= -1e-9));
        assert!(h_o.column(1).iter().filter(|&&h| h < 0.05).count() > n / 20);
    }

    #[test]
    fn damped_is_max_of_observed_and_fisher() {
        let n = 1000;
        let preds = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 0.1 * (i % 9) as f64 } else { 0.2 });
        let y = Array1::from_iter((0..n).map(|i| ((i * 31) % 97) as f64 / 10.0 - 4.0));
        let target = ResponseData::Univariate(&y.view());
        let mk = |m| Gaussian::default().with_hessian_mode(m);
        let tr = mk(HessianMode::Fisher).transform_params(&preds.view());
        let (_, h_o) = mk(HessianMode::Observed).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
        let (_, h_f) = mk(HessianMode::Fisher).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
        let (_, h_d) = mk(HessianMode::Damped).analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
        for i in 0..n {
            assert_eq!(h_d[[i, 0]], h_o[[i, 0]]);
            assert_eq!(h_d[[i, 1]], h_o[[i, 1]].max(h_f[[i, 1]]));
        }
        assert!(h_o.column(1).iter().any(|&h| h < 1.0) && h_o.column(1).iter().any(|&h| h > 3.0));
    }

    #[test]
    fn hessian_mode_serde_default_is_damped() {
        let d = Gaussian::default();
        assert_eq!(d.hessian_mode(), HessianMode::Damped);
        let mut v: serde_json::Value = serde_json::to_value(&d).unwrap();
        v.as_object_mut().unwrap().remove("hessian_mode");
        let back: Gaussian = serde_json::from_value(v).unwrap();
        assert_eq!(back.hessian_mode(), HessianMode::Damped);
    }
}

#[cfg(test)]
mod param_gradient_scale_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;

    #[test]
    fn scales_multiply_gradient_columns_only() {
        let n = 300;
        let preds = Array2::from_shape_fn((n, 2), |(i, j)| 0.1 * ((i + j) % 7) as f64 - 0.2);
        let y = Array1::from_iter((0..n).map(|i| ((i * 37) % 101) as f64 / 20.0 - 2.5));
        let target = ResponseData::Univariate(&y.view());
        let plain = Gaussian::default();
        let scaled = Gaussian::default().with_param_gradient_scales([1.0, 0.5]);
        let a = plain.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        let b = scaled.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        let sc = [1.0, 0.5];
        for j in 0..2 {
            for i in 0..n {
                assert_eq!(b.gradients[[i, j]], a.gradients[[i, j]] * sc[j]);
                assert_eq!(b.hessians[[i, j]], a.hessians[[i, j]]);
            }
        }
        // serde: absent field → None; explicit value round-trips
        let mut v: serde_json::Value = serde_json::to_value(&plain).unwrap();
        v.as_object_mut().unwrap().remove("param_gradient_scales");
        let back: Gaussian = serde_json::from_value(v).unwrap();
        assert!(back.param_gradient_scales().is_none());
        let back2: Gaussian = serde_json::from_value(serde_json::to_value(&scaled).unwrap()).unwrap();
        assert_eq!(back2.param_gradient_scales(), Some(&sc[..]));
    }
}

#[cfg(test)]
mod beta_nll_tests {
    use super::*;
    use crate::types::ResponseData;
    use ndarray::Array1;

    /// β-NLL multiplies every column of row i by σ̂_i^{2β}; β = 0 is a no-op.
    #[test]
    fn beta_nll_weights_rows_by_scale_power() {
        let n = 200;
        let preds = Array2::from_shape_fn((n, 2), |(i, j)| if j == 1 { -0.5 + 0.01 * i as f64 } else { 0.1 * (j as f64) });
        let y = Array1::from_iter((0..n).map(|i| ((i * 53) % 97) as f64 / 25.0 - 2.0));
        let target = ResponseData::Univariate(&y.view());
        let plain = Gaussian::default();
        let b = Gaussian::default().with_beta_nll(0.5);
        let zero = Gaussian::default().with_beta_nll(0.0);
        let tr = plain.transform_params(&preds.view());
        let a = plain.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        let w = b.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        let z = zero.compute_gradients_and_hessians(&preds.view(), &target, None).unwrap();
        assert_eq!(a.gradients, z.gradients);
        let mean_sigma = tr.column(1).mean().unwrap();
        for i in 0..n {
            let wi = tr[[i, 1]] / mean_sigma; // σ^{2·0.5} = σ, normalised to mean 1
            for j in 0..2 {
                assert!((w.gradients[[i, j]] - a.gradients[[i, j]] * wi).abs() <= 1e-12 * a.gradients[[i, j]].abs().max(1.0));
                assert!((w.hessians[[i, j]] - a.hessians[[i, j]] * wi).abs() <= 1e-12 * a.hessians[[i, j]].abs().max(1.0));
            }
        }
        let mut v: serde_json::Value = serde_json::to_value(&plain).unwrap();
        v.as_object_mut().unwrap().remove("beta_nll");
        let back: Gaussian = serde_json::from_value(v).unwrap();
        assert_eq!(back.beta_nll(), None);
    }
}
