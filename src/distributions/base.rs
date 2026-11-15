//! Base distribution trait and types.

use crate::error::Result;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Loss function types for distributional regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFn {
    /// Negative log-likelihood loss.
    Nll,
    /// Continuous Ranked Probability Score.
    Crps,
}

impl LossFn {
    /// Get the string name of the loss function.
    pub fn name(&self) -> &'static str {
        match self {
            LossFn::Nll => "nll",
            LossFn::Crps => "crps",
        }
    }
}

/// Stabilization methods for gradients and hessians.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stabilization {
    /// No stabilization.
    None,
    /// Median Absolute Deviation stabilization.
    Mad,
    /// L2 norm stabilization.
    L2,
}

/// Which curvature a distribution hands the Newton step as its "Hessian".
///
/// Both backends build leaves as `−Σg / (Σh + λ)` and score splits by
/// `Σg²/(Σh+λ)`, so `h` must be positive to act as curvature. The two
/// choices differ only for families whose observed second derivative is
/// unsigned (StudentT, LogNormal, Gamma, ...); gradients are identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HessianMode {
    /// The observed information: `∂²(−log f)/∂η²` at the current fit, what
    /// torch autograd (and so XGBoostLSS/LightGBMLSS) produces. Can be
    /// negative row by row.
    Observed,
    /// The expected (Fisher) information `E_y[∂²(−log f)/∂η²] = I_θθ·f′(η)²`,
    /// strictly positive, the curvature NGBoost and GAMLSS use.
    Fisher,
    /// `max(Observed, Fisher)` per row and parameter: the Fisher value floors
    /// the zero/negative observed curvature near the optimum, while far from
    /// it the (larger) observed curvature keeps the Newton step bounded —
    /// a Levenberg–Marquardt-style damping that needs no extra tuning knob.
    /// The default since 2026-11-09: in 700 paired fits it never lost to the
    /// previous defaults by more than 0.001 nats, won by 0.05–0.95 nats on
    /// four of five real targets and never diverged
    /// (research/gradientlss/reports/002-cycle1-report.md).
    #[default]
    Damped,
}

/// A distributional parameter with its response function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionParam {
    /// Name of the parameter (e.g., "loc", "scale").
    pub name: String,
    /// Response function to transform predictions to parameter space.
    pub response_fn: ResponseFn,
}

impl DistributionParam {
    /// Create a new distribution parameter.
    pub fn new(name: impl Into<String>, response_fn: ResponseFn) -> Self {
        Self {
            name: name.into(),
            response_fn,
        }
    }
}

/// Container for gradients and hessians.
#[derive(Debug, Clone)]
pub struct GradientsAndHessians {
    /// Gradients with shape (n_samples, n_params).
    pub gradients: Array2<f64>,
    /// Hessians with shape (n_samples, n_params).
    pub hessians: Array2<f64>,
}

/// Core trait for probability distributions in GradientLSS.
///
/// This trait defines the interface that all distributions must implement
/// to be used with the gradient boosting framework.
#[typetag::serde(tag = "type")]
pub trait Distribution: Send + Sync {
    /// Clone the distribution into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Distribution>;

    /// Clone the distribution into an Arc.
    fn clone_arc(&self) -> std::sync::Arc<dyn Distribution> {
        std::sync::Arc::from(self.clone_box())
    }

    /// Get the name of this distribution (e.g., "Gaussian", "Gamma").
    fn name(&self) -> &'static str;

    /// Whether this is a univariate distribution.
    fn is_univariate(&self) -> bool {
        true
    }

    /// Whether this distribution has discrete support.
    fn is_discrete(&self) -> bool {
        false
    }

    /// Number of distributional parameters.
    fn n_params(&self) -> usize;

    /// Number of target dimensions (1 for univariate, >1 for multivariate).
    fn n_targets(&self) -> usize {
        1
    }

    /// Get the distribution parameters with their response functions.
    fn params(&self) -> &[DistributionParam];

    /// Get the parameter names.
    fn param_names(&self) -> Vec<&str> {
        self.params().iter().map(|p| p.name.as_str()).collect()
    }

    /// Get the loss function.
    fn loss_fn(&self) -> LossFn;

    /// Get the stabilization method.
    fn stabilization(&self) -> Stabilization;

    /// Whether to initialize with start values.
    fn should_initialize(&self) -> bool;

    /// Compute the negative log-likelihood for given parameters and targets.
    ///
    /// # Arguments
    /// * `params` - Transformed parameters with shape (n_samples, n_params)
    /// * `target` - Target values
    ///
    /// # Returns
    /// The total negative log-likelihood.
    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64;

    /// Compute the log probability for a single observation.
    ///
    /// # Arguments
    /// * `params` - Parameter values for this observation
    /// * `target` - Target value(s)
    ///
    /// # Returns
    /// The log probability.
    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64;

    /// Draw samples from the distribution given parameters.
    ///
    /// # Arguments
    /// * `params` - Distribution parameters with shape (n_samples, n_params)
    /// * `n_samples` - Number of samples to draw per observation
    /// * `seed` - Random seed
    ///
    /// # Returns
    /// Samples with shape (n_samples, n_observations * n_targets) for multivariate,
    /// or (n_samples, n_observations) for univariate
    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64>;

    /// Transform raw predictions to the parameter space.
    ///
    /// # Arguments
    /// * `predictions` - Raw predictions with shape (n_samples, n_params)
    ///
    /// # Returns
    /// Transformed parameters.
    fn transform_params(&self, predictions: &ArrayView2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(predictions.dim());
        for (i, param) in self.params().iter().enumerate() {
            let col = predictions.column(i);
            let mut out_col = result.column_mut(i);
            param.response_fn.apply_into(&col, &mut out_col);
        }
        result
    }

    /// Post-process transformed parameters into their USER-FACING form for
    /// `predict(PredType::Parameters)` output. Default: identity.
    ///
    /// Most distributions' internal parameter representation (what `nll`,
    /// `log_prob`, and `sample` consume) is already what users expect. Mixture
    /// is the exception: its per-column response functions leave the mixing
    /// weights as raw logits (the softmax couples columns, which the scalar
    /// response-function machinery can't express), and `log_prob`/`sample`
    /// apply the softmax internally. Python's `predict_dist` returns softmaxed
    /// probabilities, so Mixture overrides this to match. Internal consumers
    /// must NOT call this — feeding finalized parameters back into
    /// `nll`/`log_prob`/`sample` would double-apply the group transform.
    fn finalize_predicted_params(&self, params: Array2<f64>) -> Array2<f64> {
        params
    }

    /// Whether `sample` is a smooth (pathwise/reparameterizable) function of
    /// the distribution parameters when the RNG seed is held fixed.
    ///
    /// The CRPS loss differentiates the sampled empirical CRPS by central
    /// differences with a fixed seed. That is only meaningful when parameters
    /// enter the sampler smoothly (loc-scale transforms, inverse-CDF draws) —
    /// the analogue of torch's `rsample`. Rejection samplers (Gamma and
    /// everything built on it) and discrete draws (Poisson, mixture gates)
    /// flip accept/reject decisions between the ±ε evaluations, and the
    /// resulting sample jump is amplified by 1/(2ε) into garbage gradients —
    /// distributions like those override this to `false`, and training errors
    /// out early for `LossFn::Crps` (Python errors likewise: torch
    /// distributions without `rsample` reject `.rsample()`).
    fn has_reparameterizable_sampler(&self) -> bool {
        true
    }

    /// Optional per-parameter gradient multipliers `c_j` (see
    /// `compute_gradients_and_hessians_with_transformed`): the Newton step of
    /// parameter j is scaled by `c_j`, an effective per-parameter learning
    /// rate. `None` means all ones.
    fn param_gradient_scales(&self) -> Option<&[f64]> {
        None
    }

    /// Optional per-ROW weights on gradients and Hessians, computed from the
    /// current transformed parameters (β-NLL, Seitzer et al. 2022: weight
    /// row i by σ̂_i^{2β} with the weight treated as a constant). Changes
    /// which rows drive the splits — high-variance rows regain influence on
    /// the location tree — without changing any leaf's Newton ratio. `None`
    /// means all ones.
    fn row_gradient_weights(&self, _transformed: &ArrayView2<f64>) -> Option<Array1<f64>> {
        None
    }

    /// Closed-form (deterministic truncated-sum) CRPS for ONE observation, on
    /// TRANSFORMED params. `None` = no closed form; the sampled estimator is
    /// used instead.
    ///
    /// This is what unlocks `LossFn::Crps` for the DISCRETE distributions:
    /// their samplers are non-reparameterizable (finite differences through a
    /// discrete draw are meaningless - see `has_reparameterizable_sampler`),
    /// but the analytic CRPS is a smooth function of the parameters, so the
    /// same central-difference machinery applies to it directly. It is also
    /// strictly better where available: exact instead of a 30-sample estimate,
    /// and cheaper than sampling + sorting.
    ///
    /// The truncation bound inside an implementation may shift by one term
    /// when FD perturbs the params; the affected term is a squared upper-tail
    /// CDF residual (~1e-9 at the 4-sigma bounds used), which is far below the
    /// FD step's own truncation error - harmless.
    fn analytic_crps(&self, params: &[f64], target: &[f64]) -> Option<f64> {
        let _ = (params, target);
        None
    }

    /// Column of the boosting margin that carries the location (used by the
    /// location-only warm-up, `GradientLSS::with_warmup_rounds`). Default:
    /// the parameter named "loc".
    fn location_param_index(&self) -> Option<usize> {
        self.params().iter().position(|p| p.name == "loc")
    }

    /// Start values used when `should_initialize()` is false; `None` (the
    /// default) = raw 0.5 for every parameter, the historical behaviour.
    /// Cheap closed forms only — the fitted (L-BFGS) start values are what
    /// `initialize: true` is for.
    fn default_start_values(&self, _target: &ResponseData) -> Option<ndarray::Array1<f64>> {
        None
    }

    /// Parameters that per-parameter early stopping
    /// (`model::ParamEarlyStoppingSetting::Auto`) watches by default when a
    /// validation set is passed: the scale/shape/dispersion parameters of
    /// families where the recipe was measured to be better or neutral.
    /// `None` (default) = no per-parameter early stopping unless configured.
    fn default_param_early_stopping_params(&self) -> Option<Vec<usize>> {
        None
    }

    /// Whether [`Self::analytic_crps`] returns `Some` - used by the training
    /// gate (`LossFn::Crps` support check) and the metric path without probing
    /// with dummy params.
    fn has_analytic_crps(&self) -> bool {
        false
    }

    /// Compute analytical gradients and hessians if available.
    ///
    /// Distributions can override this to provide exact analytical gradients
    /// instead of numerical approximations. This matches PyTorch's autograd behavior.
    ///
    /// # Arguments
    /// * `predictions` - Raw predictions with shape (n_samples, n_params)
    /// * `transformed` - Transformed parameters with shape (n_samples, n_params)
    /// * `target` - Target values
    ///
    /// # Returns
    /// Some((gradients, hessians)) if analytical gradients are available, None otherwise.
    fn analytical_gradients(
        &self,
        _predictions: &ArrayView2<f64>,
        _transformed: &ArrayView2<f64>,
        _target: &ResponseData,
    ) -> Option<(Array2<f64>, Array2<f64>)> {
        None // Default: no analytical gradients available
    }

    /// Compute gradients and hessians for the objective function.
    ///
    /// This method first tries to use analytical gradients (matching PyTorch autograd),
    /// and falls back to numerical differentiation if not available.
    ///
    /// # Arguments
    /// * `predictions` - Raw predictions with shape (n_samples, n_params)
    /// * `target` - Target values
    /// * `weights` - Optional sample weights
    ///
    /// # Returns
    /// Gradients and hessians.
    fn compute_gradients_and_hessians(
        &self,
        predictions: &ArrayView2<f64>,
        target: &ResponseData,
        weights: Option<&ArrayView1<f64>>,
    ) -> Result<GradientsAndHessians> {
        // Transform predictions to parameter space
        let transformed = self.transform_params(predictions);
        self.compute_gradients_and_hessians_with_transformed(
            predictions,
            &transformed.view(),
            target,
            weights,
        )
    }

    /// Like [`Self::compute_gradients_and_hessians`], but takes the transformed
    /// parameters as input instead of recomputing them. The training loop's
    /// metric pass transforms the post-update margin every round, and the next
    /// round's objective receives that exact margin — the model layer caches the
    /// transform across the two calls and enters here, skipping one full
    /// response-transform pass per boosting round.
    ///
    /// `transformed` MUST equal `self.transform_params(predictions)`.
    fn compute_gradients_and_hessians_with_transformed(
        &self,
        predictions: &ArrayView2<f64>,
        transformed: &ArrayView2<f64>,
        target: &ResponseData,
        weights: Option<&ArrayView1<f64>>,
    ) -> Result<GradientsAndHessians> {
        let n_samples = predictions.nrows();
        let n_params = self.n_params();

        // Try analytical gradients first, fall back to numerical
        let (mut gradients, mut hessians) = self
            .analytical_gradients(predictions, transformed, target)
            .unwrap_or_else(|| {
                self.numerical_gradients_hessians(predictions, transformed, target)
                    .unwrap_or_else(|_| {
                        (
                            Array2::zeros((n_samples, n_params)),
                            Array2::ones((n_samples, n_params)),
                        )
                    })
            });

        // Apply stabilization
        self.stabilize_derivatives(&mut gradients, &mut hessians);

        // Per-parameter shrinkage: scale the GRADIENT column only, so the leaf
        // value −Σg/(Σh+λ) — and with it the Newton step of that parameter —
        // is multiplied by c_j while split gains and min_child_weight keep the
        // Hessian scale. An effective per-parameter learning rate η·c_j, the
        // step length ν_k of gamboostLSS. None = all ones (unchanged).
        if let Some(scales) = self.param_gradient_scales() {
            for (j, &c) in scales.iter().enumerate().take(gradients.ncols()) {
                if c != 1.0 {
                    gradients.column_mut(j).mapv_inplace(|g| g * c);
                }
            }
        }

        // Per-row weights (β-NLL style) on both g and h — see
        // `row_gradient_weights`. Applied like sample weights.
        if let Some(mut rw) = self.row_gradient_weights(transformed) {
            // Normalise to mean 1: the raw σ̂^{2β} weights carry the target's
            // scale (σ̂ ≈ 1e-3 would shrink every leaf by 1e-6 and freeze the
            // fit), whereas only their RATIOS between rows are the point.
            let mean = rw.iter().filter(|v| v.is_finite()).sum::<f64>() / rw.len().max(1) as f64;
            if mean.is_finite() && mean > 0.0 {
                rw.mapv_inplace(|w| if w.is_finite() { w / mean } else { 1.0 });
                let rw_col = rw.view().insert_axis(ndarray::Axis(1));
                gradients *= &rw_col;
                hessians *= &rw_col;
            }
        }

        // Apply weights if provided — broadcast column vector across columns.
        // insert_axis on a reborrowed view: no per-round copy of the weights.
        if let Some(w) = weights {
            let w_col = w.view().insert_axis(ndarray::Axis(1));
            gradients *= &w_col;
            hessians *= &w_col;
        }

        Ok(GradientsAndHessians {
            gradients,
            hessians,
        })
    }

    /// Compute numerical gradients and hessians.
    fn numerical_gradients_hessians(
        &self,
        predictions: &ArrayView2<f64>,
        transformed: &ArrayView2<f64>,
        target: &ResponseData,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let n_samples = predictions.nrows();
        let n_params = self.n_params();
        // Step size for central differences. The Hessian uses the second
        // difference (f⁺ - 2f⁰ + f⁻)/h², whose round-off error is ~4·ε_machine·|f|/h²,
        // so the optimal h is ~ε_machine^(1/4) ≈ 1e-4. The previous 1e-7 amplified
        // f64 round-off by 1e14 and made the Hessians mostly noise. At h = 1e-4 the
        // gradient's central difference is also still accurate (truncation
        // ~h²·|f‴|/6, round-off ~ε_machine·|f|/h — both negligible).
        let eps = 1e-4;
        let use_crps = self.loss_fn() == LossFn::Crps;

        let mut gradients = Array2::zeros((n_samples, n_params));
        let mut hessians = Array2::zeros((n_samples, n_params));

        // Precompute column means for nan_to_num replacement,
        // matching Python's torch.nanmean(predt) behavior.
        // Single-pass mean computation avoids intermediate Vec allocation.
        let col_means: Vec<f64> = (0..n_params)
            .map(|p| {
                let col = predictions.column(p);
                let (sum, count) = col.iter().fold((0.0, 0usize), |(s, c), &v| {
                    if v.is_finite() {
                        (s + v, c + 1)
                    } else {
                        (s, c)
                    }
                });
                if count == 0 { 0.0 } else { sum / count as f64 }
            })
            .collect();

        let n_targets = self.n_targets();
        let params_list = self.params();

        // Precompute reciprocals to avoid repeated division in hot loop
        let inv_2eps = 1.0 / (2.0 * eps);
        let inv_eps_sq = 1.0 / (eps * eps);

        // Parallel threshold: for small sample counts, rayon overhead exceeds benefit.
        let use_parallel = n_samples >= 256;

        // Per-sample fill, writing straight into the output rows. Scratch is
        // per-thread (`for_each_init`) rather than per-sample: the old shape —
        // 4 fresh Vecs per sample plus a collected Vec of row pairs and a
        // copy-out loop — allocated ~4·n_samples times per boosting round.
        let fill = |scratch: &mut (Vec<f64>, Vec<f64>, Vec<f64>),
                    i: usize,
                    grad_row: &mut [f64],
                    hess_row: &mut [f64]| {
            let (params_buf, y_buf, crps_sort_buf) = scratch;

            // Copy transformed params for this sample
            for (j, v) in transformed.row(i).iter().enumerate() {
                params_buf[j] = *v;
            }

            // Fill target buffer
            match target {
                ResponseData::Univariate(arr) => {
                    y_buf[0] = arr[i];
                }
                ResponseData::Multivariate(arr) => {
                    for (j, &v) in arr.row(i).iter().enumerate() {
                        y_buf[j] = v;
                    }
                }
            }
            let y_buf = &y_buf[..n_targets];

            // For NLL path, compute base loss once per sample (shared across params)
            let loss_center = if !use_crps {
                -self.log_prob(params_buf, y_buf)
            } else {
                0.0
            };

            for p in 0..n_params {
                let pred_val = predictions[[i, p]];
                let response_fn = &params_list[p].response_fn;
                let nan_replacement = col_means[p];

                let pred_plus = pred_val + eps;
                let pred_minus = pred_val - eps;

                let param_plus =
                    response_fn.apply_scalar_with_nan_replacement(pred_plus, nan_replacement);
                let param_minus =
                    response_fn.apply_scalar_with_nan_replacement(pred_minus, nan_replacement);

                // Save original and perturb in-place (avoids full copy_from_slice)
                let original = params_buf[p];
                params_buf[p] = param_plus;

                if use_crps {
                    let n_crps_samples: usize = 30;
                    let seed: u64 = 123;

                    // Analytic CRPS when the distribution has one (exact and
                    // smooth in the params - the only valid option for the
                    // discrete distributions); sampled estimator otherwise.
                    let loss_plus = self.analytic_crps(params_buf, y_buf).unwrap_or_else(|| {
                        self.crps_single_observation_buffered(
                            params_buf,
                            y_buf,
                            n_crps_samples,
                            seed,
                            crps_sort_buf,
                        )
                    });
                    params_buf[p] = param_minus;
                    let loss_minus = self.analytic_crps(params_buf, y_buf).unwrap_or_else(|| {
                        self.crps_single_observation_buffered(
                            params_buf,
                            y_buf,
                            n_crps_samples,
                            seed,
                            crps_sort_buf,
                        )
                    });

                    grad_row[p] = (loss_plus - loss_minus) * inv_2eps;
                    hess_row[p] = 1.0;
                } else {
                    let loss_plus = -self.log_prob(params_buf, y_buf);
                    params_buf[p] = param_minus;
                    let loss_minus = -self.log_prob(params_buf, y_buf);

                    grad_row[p] = (loss_plus - loss_minus) * inv_2eps;
                    // True second difference, no positivity floor: PyTorch autograd
                    // hands XGBoost/LightGBM the exact (possibly negative) diagonal
                    // Hessian, and both boosters accept it — flooring changed leaf
                    // values relative to the Python implementations.
                    hess_row[p] = (loss_plus - 2.0 * loss_center + loss_minus) * inv_eps_sq;
                }

                // Restore original value
                params_buf[p] = original;
            }
        };

        let make_scratch = || {
            (
                vec![0.0f64; n_params],
                vec![0.0f64; n_targets],
                Vec::<f64>::new(),
            )
        };

        let g_slice = gradients
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");
        let h_slice = hessians
            .as_slice_mut()
            .expect("freshly-allocated Array2 is contiguous");

        if use_parallel {
            g_slice
                .par_chunks_mut(n_params)
                .zip(h_slice.par_chunks_mut(n_params))
                .enumerate()
                .for_each_init(make_scratch, |scratch, (i, (gc, hc))| {
                    fill(scratch, i, gc, hc)
                });
        } else {
            let mut scratch = make_scratch();
            for (i, (gc, hc)) in g_slice
                .chunks_mut(n_params)
                .zip(h_slice.chunks_mut(n_params))
                .enumerate()
            {
                fill(&mut scratch, i, gc, hc);
            }
        }

        Ok((gradients, hessians))
    }

    /// Compute CRPS for a single observation using sampling.
    /// This matches Python's approach of sampling from the distribution
    /// and computing empirical CRPS.
    fn crps_single_observation(
        &self,
        params: &[f64],
        target: &[f64],
        n_samples: usize,
        seed: u64,
    ) -> f64 {
        let mut sort_buf = Vec::new();
        self.crps_single_observation_buffered(params, target, n_samples, seed, &mut sort_buf)
    }

    /// Like [`Self::crps_single_observation`], but reuses a caller-owned sort
    /// buffer. The numerical CRPS gradient calls this 2·n_params times per
    /// observation per round — a fresh params array and sample Vec each time
    /// were the dominant allocations on that path (the params view is now
    /// borrowed from the slice, allocation-free).
    fn crps_single_observation_buffered(
        &self,
        params: &[f64],
        target: &[f64],
        n_samples: usize,
        seed: u64,
        sort_buf: &mut Vec<f64>,
    ) -> f64 {
        // 1-row view straight over the params slice — no copy.
        let params_view =
            ArrayView2::from_shape((1, params.len()), params).expect("slice reshapes to one row");

        let samples = self.sample(&params_view, n_samples, seed);

        // Compute empirical CRPS for the single observation
        let y = target[0];
        sort_buf.clear();
        sort_buf.extend(samples.column(0).iter().copied());
        sort_buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut crps = 0.0;
        let mut yhat_prev = 0.0;
        let mut yhat_cdf = 0.0;
        let mut y_cdf = 0.0;

        for &yhat in sort_buf.iter() {
            let flag = y_cdf == 0.0 && y < yhat;

            if flag {
                crps += (y - yhat_prev) * yhat_cdf * yhat_cdf;
                crps += (yhat - y) * (yhat_cdf - 1.0) * (yhat_cdf - 1.0);
                y_cdf = 1.0;
            } else {
                crps += (yhat - yhat_prev) * (yhat_cdf - y_cdf) * (yhat_cdf - y_cdf);
            }

            yhat_cdf += 1.0 / n_samples as f64;
            yhat_prev = yhat;
        }

        // Handle case where y > all samples
        if y_cdf == 0.0 {
            crps += y - sort_buf.last().unwrap_or(&0.0);
        }

        crps
    }

    /// Get target values for a specific observation.
    ///
    /// # Arguments
    /// * `target` - Target data
    /// * `obs_idx` - Observation index
    ///
    /// # Returns
    /// Target values for this observation
    fn get_target_for_observation(&self, target: &ResponseData, obs_idx: usize) -> Vec<f64> {
        match target {
            ResponseData::Univariate(arr) => vec![arr[obs_idx]],
            ResponseData::Multivariate(arr) => arr.row(obs_idx).to_vec(),
        }
    }

    /// Stabilize gradients and hessians.
    fn stabilize_derivatives(&self, gradients: &mut Array2<f64>, hessians: &mut Array2<f64>) {
        match self.stabilization() {
            Stabilization::None => {
                // Just replace NaNs with column means
                replace_nans_with_mean(gradients);
                replace_nans_with_mean(hessians);
            }
            Stabilization::Mad => {
                stabilize_mad(gradients);
                stabilize_mad(hessians);
            }
            Stabilization::L2 => {
                stabilize_l2(gradients);
                stabilize_l2(hessians);
            }
        }
    }

    /// Calculate unconditional start values for distributional parameters.
    ///
    /// Uses L-BFGS optimization to match Python's torch.optim.LBFGS behavior.
    /// This uses the actual distribution's log_prob for optimization, matching
    /// the Python implementation which calls self.distribution(*params).log_prob(target).
    ///
    /// # Arguments
    /// * `target` - Target values
    /// * `max_iter` - Maximum optimization iterations
    ///
    /// # Returns
    /// Tuple of (loss, start_values)
    fn calculate_start_values(
        &self,
        target: &ResponseData,
        max_iter: usize,
    ) -> Result<(f64, Array1<f64>)> {
        self.calculate_start_values_default(target, max_iter)
    }

    /// The unconditional maximum-likelihood fit that `dist_select` ranks
    /// families by: `(total NLL, raw parameters)`.
    ///
    /// Defaults to [`Self::calculate_start_values`]. A family whose training
    /// start values deliberately differ from its maximum-likelihood fit
    /// overrides this, so that it is ranked at its best fit and not at its
    /// start (StudentT caps the start df at 10; ranked at the cap it lost to
    /// Gaussian on near-Gaussian targets it fits at least as well).
    fn marginal_fit(&self, target: &ResponseData, max_iter: usize) -> Result<(f64, Array1<f64>)> {
        self.calculate_start_values(target, max_iter)
    }

    /// The L-BFGS start-value search over the per-parameter response
    /// functions — the default body of [`Self::calculate_start_values`], kept
    /// callable so a distribution that overrides the start values for one of
    /// its parameterizations can still reach it for the others
    /// (`NegativeBinomial::MeanDispersion`, 2026-10-31).
    fn calculate_start_values_default(
        &self,
        target: &ResponseData,
        max_iter: usize,
    ) -> Result<(f64, Array1<f64>)> {
        let n_params = self.n_params();

        // Clone target data for the optimization problem
        let target_data: Vec<f64> = match target {
            ResponseData::Univariate(arr) => arr.to_vec(),
            ResponseData::Multivariate(arr) => arr.iter().copied().collect(),
        };
        let is_multivariate = !self.is_univariate();
        let n_targets = self.n_targets();

        // Collect response functions info for the problem
        let response_fns: Vec<ResponseFn> = self.params().iter().map(|p| p.response_fn).collect();

        // Use the actual distribution's log_prob via clone_box
        let dist = self.clone_box();

        // Create the optimization problem using the actual distribution
        let mut problem = StartValueProblem {
            target_data,
            is_multivariate,
            n_targets,
            n_params,
            response_fns,
            dist,
            min_finite_rows: 0,
        };
        // Rows whose log-prob is finite at the initial point. The cost skips
        // NaN rows (torch.nansum parity), so a point where EVERY row is NaN
        // would score 0 — below any real minimum — and L-BFGS would keep it as
        // "best" (seen with Gamma on y ≈ 450: start values (−4356, 6021)).
        // A candidate with fewer finite rows than the start is rejected.
        problem.min_finite_rows = problem.finite_row_count(&vec![0.5; n_params]);

        // Initial guess, then a damped diagonal-Newton pre-phase. L-BFGS +
        // MoreThuente's first trial step along −g has length ‖g‖, which for a
        // target not already O(1) (e.g. y ≈ 450, ‖g‖ ≈ 7e8) is NaN/Inf; the
        // line search then aborts and argmin hands back the INITIAL point as
        // "best" — raw 0.5 for every parameter, silently. The pre-phase takes
        // per-coordinate Newton steps (the same step boosting itself takes,
        // so it is well-scaled by construction) with Armijo backtracking until
        // the gradient is small or a step is rejected, and L-BFGS polishes.
        let init_params = diagonal_newton_prephase(&problem, vec![0.5; n_params]);

        // Set up L-BFGS with More-Thuente line search (similar to PyTorch's strong_wolfe)
        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 7); // 7 is a common default for L-BFGS memory

        // Python uses max_iter outer epochs x min(max_iter/4, 20) inner L-BFGS iterations.
        // For the default max_iter=50, that's 50 * 12 = 600 total L-BFGS iterations.
        // We match this total budget.
        let inner_iters = (max_iter / 4).min(20);
        let total_iters = max_iter * inner_iters;

        // Run the optimizer. No target_cost: argmin treats it as "stop once
        // best cost <= target", and a total NLL is legitimately negative
        // whenever the ML density exceeds 1 (e.g. small-scale continuous
        // targets) — a 0.0 target froze those fits at their first
        // zero-crossing iterate, corrupting dist_select rankings and
        // `initialize: true` start values. Python runs the full epoch budget.
        let result = Executor::new(problem, solver)
            .configure(|state| state.param(init_params).max_iters(total_iters as u64))
            .run();

        match result {
            Ok(res) => {
                let best_params: Vec<f64> = res
                    .state()
                    .get_best_param()
                    .cloned()
                    .unwrap_or_else(|| vec![0.5; n_params]);
                let best_cost: f64 = res.state().get_best_cost();

                // Convert to Array1 and replace any NaNs
                let mut params_arr: Array1<f64> = Array1::from_vec(best_params);
                for v in params_arr.iter_mut() {
                    if !v.is_finite() {
                        *v = 0.5;
                    }
                }

                Ok((best_cost, params_arr))
            }
            Err(_) => {
                // Fall back to simple initialization if L-BFGS fails
                let params_arr = Array1::from_elem(n_params, 0.5);
                let transformed: Vec<f64> = params_arr
                    .iter()
                    .zip(self.params().iter())
                    .map(|(&p, param)| param.response_fn.apply_scalar(p))
                    .collect();
                let loss = self.compute_total_loss(&transformed, target);
                Ok((loss, params_arr))
            }
        }
    }

    /// Compute total loss for given parameters and target data.
    ///
    /// # Arguments
    /// * `params` - Parameter values
    /// * `target` - Target data
    ///
    /// # Returns
    /// Total negative log-likelihood
    fn compute_total_loss(&self, params: &[f64], target: &ResponseData) -> f64 {
        // torch.nansum parity (Python's loss_fn_start_values): NaN terms → 0.
        let nan_to_zero = |v: f64| if v.is_nan() { 0.0 } else { v };
        match target {
            ResponseData::Univariate(arr) => arr
                .iter()
                .map(|&y| nan_to_zero(-self.log_prob(params, &[y])))
                .sum(),
            ResponseData::Multivariate(arr) => (0..arr.nrows())
                .map(|i| {
                    let y = arr.row(i).to_vec();
                    nan_to_zero(-self.log_prob(params, &y))
                })
                .sum(),
        }
    }

    /// Compute CRPS score using sampling.
    ///
    /// # Arguments
    /// * `target` - Target values
    /// * `samples` - Samples from predicted distribution (n_samples, n_observations * n_targets for multivariate)
    fn crps_score(&self, target: &ResponseData, samples: &ArrayView2<f64>) -> f64 {
        match target {
            ResponseData::Univariate(arr) => {
                let n_obs = arr.len();
                let n_samples = samples.nrows();

                let mut total_crps = 0.0;

                for j in 0..n_obs {
                    let y = arr[j];

                    // Get samples for this observation and sort them
                    let mut obs_samples: Vec<f64> = samples.column(j).to_vec();
                    obs_samples
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    // Compute CRPS using the sorted samples
                    let mut crps = 0.0;
                    let mut yhat_prev = 0.0;
                    let mut yhat_cdf = 0.0;
                    let mut y_cdf = 0.0;

                    for &yhat in &obs_samples {
                        let flag = y_cdf == 0.0 && y < yhat;

                        if flag {
                            crps += (y - yhat_prev) * yhat_cdf * yhat_cdf;
                            crps += (yhat - y) * (yhat_cdf - 1.0) * (yhat_cdf - 1.0);
                            y_cdf = 1.0;
                        } else {
                            crps += (yhat - yhat_prev) * (yhat_cdf - y_cdf) * (yhat_cdf - y_cdf);
                        }

                        yhat_cdf += 1.0 / n_samples as f64;
                        yhat_prev = yhat;
                    }

                    // Handle case where y > all samples
                    if y_cdf == 0.0 {
                        crps += y - obs_samples.last().unwrap_or(&0.0);
                    }

                    // torch.nansum parity: a NaN per-observation CRPS → 0.
                    if !crps.is_nan() {
                        total_crps += crps;
                    }
                }

                total_crps
            }
            ResponseData::Multivariate(arr) => {
                // For multivariate CRPS, we compute it per target dimension
                let n_obs = arr.nrows();
                let n_targets = arr.ncols();
                let n_samples = samples.nrows();

                let mut total_crps = 0.0;

                for t in 0..n_targets {
                    for j in 0..n_obs {
                        let y = arr[[j, t]];

                        // Get samples for this observation and target dimension
                        let mut obs_samples: Vec<f64> = samples.column(j * n_targets + t).to_vec();
                        obs_samples
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        // Compute CRPS using the sorted samples
                        let mut crps = 0.0;
                        let mut yhat_prev = 0.0;
                        let mut yhat_cdf = 0.0;
                        let mut y_cdf = 0.0;

                        for &yhat in &obs_samples {
                            let flag = y_cdf == 0.0 && y < yhat;

                            if flag {
                                crps += (y - yhat_prev) * yhat_cdf * yhat_cdf;
                                crps += (yhat - y) * (yhat_cdf - 1.0) * (yhat_cdf - 1.0);
                                y_cdf = 1.0;
                            } else {
                                crps +=
                                    (yhat - yhat_prev) * (yhat_cdf - y_cdf) * (yhat_cdf - y_cdf);
                            }

                            yhat_cdf += 1.0 / n_samples as f64;
                            yhat_prev = yhat;
                        }

                        // Handle case where y > all samples
                        if y_cdf == 0.0 {
                            crps += y - obs_samples.last().unwrap_or(&0.0);
                        }

                        // torch.nansum parity: a NaN per-observation CRPS → 0.
                        if !crps.is_nan() {
                            total_crps += crps;
                        }
                    }
                }

                total_crps
            }
        }
    }
}

/// Replace NaN values in array with column means.
fn replace_nans_with_mean(arr: &mut Array2<f64>) {
    // Fast clean-path: a single contiguous, branch-light scan over the whole
    // backing buffer. The analytical gradient paths clamp their inputs, so
    // non-finite values are the exception, not the rule —
    // and this runs on both gradients and hessians every boosting round.
    // When the buffer is already finite (the common case) we skip the strided
    // per-column mean/fix-up work entirely. C-order arrays are contiguous, so
    // `as_slice` is Some; a non-contiguous view falls through to the columns.
    if let Some(slice) = arr.as_slice() {
        // OR-reduce instead of short-circuiting: no per-element branch, fully
        // auto-vectorizable. NaN/±inf are exactly the non-finite values.
        let any_bad = slice.iter().fold(false, |acc, &v| acc | !v.is_finite());
        if !any_bad {
            return;
        }
    }

    for mut col in arr.columns_mut() {
        // Single-pass mean computation avoids intermediate Vec allocation
        let (sum, count) = col.iter().fold((0.0, 0usize), |(s, c), &v| {
            if v.is_finite() {
                (s + v, c + 1)
            } else {
                (s, c)
            }
        });
        let mean = if count == 0 { 0.0 } else { sum / count as f64 };
        for v in col.iter_mut() {
            if !v.is_finite() {
                *v = mean;
            }
        }
    }
}

/// MAD stabilization.
fn stabilize_mad(arr: &mut Array2<f64>) {
    replace_nans_with_mean(arr);

    // Reuse a single buffer for deviations across columns
    let n_rows = arr.nrows();
    let mut buf = vec![0.0f64; n_rows];

    for mut col in arr.columns_mut() {
        // Copy column to buffer and select the median in-place (no allocation)
        for (j, &v) in col.iter().enumerate() {
            buf[j] = v;
        }
        let median = median_inplace(&mut buf);

        // Compute deviations in-place in the buffer, then select for MAD
        for (j, &v) in col.iter().enumerate() {
            buf[j] = (v - median).abs();
        }
        let mad = median_inplace(&mut buf);

        // A (near-)constant column has no spread to normalize: dividing by
        // the 1e-4 clamp would multiply it by ~1e4 instead. This is not
        // hypothetical — the CRPS metric hessian depends only on the fitted
        // parameters, so at round 0 (constant init margins) every row is
        // identical, and unit-hess mode is constant EVERY round; Mad-stab
        // was silently crushing those leaf values ~1e-4x. Python XGBoostLSS
        // has the same clamp bug; parity is preserved for every
        // non-degenerate column.
        let inv_mad = if mad < 1e-12 { 1.0 } else { 1.0 / mad.max(1e-4) };
        for v in col.iter_mut() {
            *v *= inv_mad;
        }
    }
}

/// Compute the lower median via quickselect — O(n) instead of a full
/// O(n log n) sort, and this runs twice per parameter column every boosting
/// round under MAD stabilization. Lower median for even lengths matches
/// torch.nanmedian; (len-1)/2 selects it for both parities.
/// Assumes all values are finite (NaNs already replaced by replace_nans_with_mean).
fn median_inplace(buf: &mut [f64]) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    let mid = (buf.len() - 1) / 2;
    let (_, m, _) = buf.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    *m
}

/// L2 stabilization.
fn stabilize_l2(arr: &mut Array2<f64>) {
    replace_nans_with_mean(arr);

    for mut col in arr.columns_mut() {
        let sum_sq: f64 = col.iter().map(|v| v * v).sum();
        let l2 = (sum_sq / col.len() as f64).sqrt().clamp(1e-4, 10000.0);

        let inv_l2 = 1.0 / l2;
        for v in col.iter_mut() {
            *v *= inv_l2;
        }
    }
}

// ============================================================================
// L-BFGS Optimization for Start Values (matching Python's torch.optim.LBFGS)
// ============================================================================

/// Optimization problem for finding start values using L-BFGS.
/// This matches the behavior of Python's torch.optim.LBFGS with strong_wolfe line search.
/// Uses the actual distribution's log_prob method for computing the cost,
/// matching Python's loss_fn_start_values which calls self.distribution(*params).log_prob(target).
struct StartValueProblem {
    /// Finite-row count at the initial point; a candidate below it costs f64::MAX.
    min_finite_rows: usize,
    target_data: Vec<f64>,
    is_multivariate: bool,
    n_targets: usize,
    n_params: usize,
    response_fns: Vec<ResponseFn>,
    dist: Box<dyn Distribution>,
}

/// Damped diagonal-Newton descent on the start-value problem from `x`.
///
/// Per iteration: central-difference gradient `g` and diagonal second
/// difference `h` (the `cost(x ± εe_i)` evaluations are shared), direction
/// `d_i = −g_i / (|h_i| + 1e-8)` (negative curvature → use |h|, zero → a
/// gradient step), Armijo backtracking (ρ = ½, c₁ = 1e-4, non-finite cost is
/// a rejection). Stops after `MAX_ITERS`, when ‖g‖ < 1e-6, or when no step
/// length is accepted. Returns the best point seen — never worse than `x`.
fn diagonal_newton_prephase(problem: &StartValueProblem, mut x: Vec<f64>) -> Vec<f64> {
    // Budget: each iteration is (1 + 2p) cost passes plus ≤ 2 finite-row
    // passes per trial step — small next to the 600-iteration L-BFGS budget
    // that follows. 100 iterations let the trust radius grow from 2 to 2^k
    // past a location offset of 1e6 even with a few backtracks on the way.
    const MAX_ITERS: usize = 100;
    const EPS: f64 = 1e-4;
    let cost = |p: &Vec<f64>| problem.cost(p).unwrap_or(f64::MAX);
    let n = x.len();
    let mut c = cost(&x);
    if !c.is_finite() {
        return x;
    }
    // The cost skips rows whose log-prob is NaN (torch.nansum parity), so a
    // parameter vector that breaks EVERY row (e.g. an overflowed `Exp`
    // response, ν = ∞) scores 0 and would look like a huge improvement along
    // a flat likelihood direction. A step may not lose finite rows.
    let finite_rows_at_start = problem.finite_row_count(&x);
    // Trust radius on the raw-space step (∞-norm). Where the curvature is ≈ 0
    // (a flat df direction, a saturated response) the Newton step is
    // arbitrarily long and Armijo would still accept it if it lowers the
    // (initially terrible) cost a little — landing in a dead zone such as the
    // ExpDf floor ν = 2, where e^η ≈ 0 kills the gradient for good. Doubled
    // on every full-step acceptance, so a location 450 units away is still
    // reached in ~8 iterations.
    let mut radius: f64 = 2.0;
    for _ in 0..MAX_ITERS {
        let mut g = vec![0.0; n];
        let mut d = vec![0.0; n];
        match problem.analytic_diag(&x) {
            Some((ga, ha)) => {
                // Curvature per coordinate = max(|analytic h|, FD second
                // difference). |analytic h| keeps the exact step where the
                // curvature is tiny but real (a saturated t location far from
                // the data, below FD noise); the FD value is the OBSERVED
                // total curvature and takes over where the analytic one is
                // smaller — StudentT's df under Fisher curvature, which
                // otherwise out-ran σ into the ν = 2 dead zone of `ExpDf` and
                // made the start values depend on the training HessianMode.
                // A negative observed curvature still points the Newton step
                // toward the data, hence the |·|.
                for i in 0..n {
                    g[i] = if ga[i].is_finite() { ga[i] } else { 0.0 };
                    let h_an = if ha[i].is_finite() { ha[i].abs() } else { 0.0 };
                    let mut plus = x.clone();
                    let mut minus = x.clone();
                    plus[i] += EPS;
                    minus[i] -= EPS;
                    let h_fd = (cost(&plus) - 2.0 * c + cost(&minus)) / (EPS * EPS);
                    let h_fd = if h_fd.is_finite() { h_fd.max(0.0) } else { 0.0 };
                    d[i] = -g[i] / (h_an.max(h_fd) + 1e-12);
                }
            }
            None => {
                for i in 0..n {
                    let mut plus = x.clone();
                    let mut minus = x.clone();
                    plus[i] += EPS;
                    minus[i] -= EPS;
                    let (cp, cm) = (cost(&plus), cost(&minus));
                    g[i] = (cp - cm) / (2.0 * EPS);
                    let h = (cp - 2.0 * c + cm) / (EPS * EPS);
                    if !g[i].is_finite() {
                        g[i] = 0.0;
                    }
                    let h = if h.is_finite() { h.abs() } else { 0.0 };
                    d[i] = -g[i] / (h + 1e-8);
                }
            }
        }
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < 1e-6 {
            break;
        }
        let step_inf = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if step_inf > radius {
            let shrink = radius / step_inf;
            d.iter_mut().for_each(|v| *v *= shrink);
        }
        let slope: f64 = g.iter().zip(&d).map(|(g, d)| g * d).sum();
        let c_prev = c;
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..40 {
            let xn: Vec<f64> = x.iter().zip(&d).map(|(x, d)| x + alpha * d).collect();
            let cn = cost(&xn);
            if cn.is_finite()
                && cn <= c + 1e-4 * alpha * slope
                && problem.transformed_are_finite(&xn)
                && problem.finite_row_count(&xn) >= finite_rows_at_start
            {
                x = xn;
                c = cn;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
        // Stalled: the accepted step no longer changes the cost measurably.
        if (c_prev - c).abs() <= 1e-12 * c.abs().max(1.0) {
            break;
        }
        if alpha == 1.0 {
            radius = (radius * 2.0).min(1e6);
        } else {
            radius = (alpha * step_inf.min(radius)).max(1e-3);
        }
    }
    x
}

impl StartValueProblem {
    fn transformed(&self, params: &[f64]) -> Vec<f64> {
        params
            .iter()
            .zip(self.response_fns.iter())
            .map(|(&p, response_fn)| response_fn.apply_scalar(if p.is_finite() { p } else { 0.5 }))
            .collect()
    }

    fn transformed_are_finite(&self, params: &[f64]) -> bool {
        self.transformed(params).iter().all(|v| v.is_finite())
    }

    /// Exact summed gradient and diagonal Hessian in raw space from the
    /// distribution's analytical per-row derivatives, when it has them and
    /// the target is univariate. Finite differences cannot resolve a tiny
    /// true curvature (e.g. a location with n/σ² ≈ 1e-8 once σ has been
    /// inflated) against rounding noise in a cost of size 1e4; the analytic
    /// path has no such floor.
    fn analytic_diag(&self, params: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        if self.is_multivariate {
            return None;
        }
        let n_obs = self.target_data.len();
        let predictions = Array2::from_shape_fn((n_obs, self.n_params), |(_, j)| params[j]);
        let transformed = self.dist.transform_params(&predictions.view());
        let y = ndarray::ArrayView1::from(&self.target_data[..]);
        let target = ResponseData::Univariate(&y);
        let (g, h) = self
            .dist
            .analytical_gradients(&predictions.view(), &transformed.view(), &target)?;
        let sum_finite = |col: ndarray::ArrayView1<f64>| col.iter().filter(|v| v.is_finite()).sum::<f64>();
        let gs: Vec<f64> = (0..self.n_params).map(|j| sum_finite(g.column(j))).collect();
        let hs: Vec<f64> = (0..self.n_params).map(|j| sum_finite(h.column(j))).collect();
        Some((gs, hs))
    }

    /// Number of observations whose log-prob is finite at `params`.
    fn finite_row_count(&self, params: &[f64]) -> usize {
        let transformed = self.transformed(params);
        if self.is_multivariate {
            let n_obs = self.target_data.len() / self.n_targets;
            (0..n_obs)
                .into_par_iter()
                .filter(|&i| {
                    let start = i * self.n_targets;
                    self.dist
                        .log_prob(&transformed, &self.target_data[start..start + self.n_targets])
                        .is_finite()
                })
                .count()
        } else {
            self.target_data
                .par_iter()
                .filter(|&&y| self.dist.log_prob(&transformed, &[y]).is_finite())
                .count()
        }
    }
}

impl CostFunction for StartValueProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> std::result::Result<Self::Output, ArgminError> {
        // Replace NaN/Inf with 0.5 (matches Python's nan_inf handling in loss_fn_start_values)
        let safe_params: Vec<f64> = params
            .iter()
            .map(|&p| if p.is_finite() { p } else { 0.5 })
            .collect();

        // Transform parameters to response scale using the actual response functions
        let transformed: Vec<f64> = safe_params
            .iter()
            .zip(self.response_fns.iter())
            .map(|(&p, response_fn)| response_fn.apply_scalar(p))
            .collect();

        // Total NLL via the actual distribution's log_prob. par_nansum both
        // parallelizes the full-dataset pass (this cost — and, via the
        // finite-difference `gradient` below, 2·n_params of these per L-BFGS
        // iteration — dominated `initialize: true` fits) and matches Python's
        // `-torch.nansum(...)` in loss_fn_start_values: a NaN per-sample
        // log-prob is skipped, not allowed to poison the whole loss into the
        // f64::MAX clamp. The parallel reduction reorders float adds (~1e-12),
        // which is immaterial to the optimizer.
        let loss: f64 = if self.is_multivariate {
            let n_obs = self.target_data.len() / self.n_targets;
            crate::distributions::util::par_nansum(n_obs, |i| {
                let start = i * self.n_targets;
                let end = start + self.n_targets;
                let target_slice = &self.target_data[start..end];
                -self.dist.log_prob(&transformed, target_slice)
            })
        } else {
            crate::distributions::util::par_nansum(self.target_data.len(), |i| {
                -self.dist.log_prob(&transformed, &[self.target_data[i]])
            })
        };

        if loss.is_finite() {
            // Reject points that lost finite rows relative to the start (see
            // `min_finite_rows`); only re-count when the sum looks suspiciously
            // low to keep the common path at one pass.
            if self.min_finite_rows > 0 && self.finite_row_count(&safe_params) < self.min_finite_rows {
                return Ok(f64::MAX);
            }
            Ok(loss)
        } else {
            Ok(f64::MAX)
        }
    }
}

impl Gradient for StartValueProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, params: &Self::Param) -> std::result::Result<Self::Gradient, ArgminError> {
        let eps = 1e-5;
        let mut grad = vec![0.0; self.n_params];

        for i in 0..self.n_params {
            // Central difference: (f(x+h) - f(x-h)) / (2h)
            // Better approximation of PyTorch autograd than forward difference
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;
            let cost_plus = self.cost(&params_plus)?;
            let cost_minus = self.cost(&params_minus)?;
            grad[i] = (cost_plus - cost_minus) / (2.0 * eps);

            if !grad[i].is_finite() {
                grad[i] = 0.0;
            }
        }

        Ok(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_fn_name() {
        assert_eq!(LossFn::Nll.name(), "nll");
        assert_eq!(LossFn::Crps.name(), "crps");
    }

    #[test]
    fn test_median_inplace() {
        assert_eq!(median_inplace(&mut [1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median_inplace(&mut [3.0, 1.0, 2.0]), 2.0);
        // Even-length: lower median (matches torch.nanmedian)
        assert_eq!(median_inplace(&mut [1.0, 2.0, 3.0, 4.0]), 2.0);
        assert_eq!(median_inplace(&mut [4.0, 2.0, 3.0, 1.0]), 2.0);
        assert_eq!(median_inplace(&mut []), 0.0);
    }

    #[test]
    fn test_distribution_param() {
        let param = DistributionParam::new("loc", ResponseFn::Identity);
        assert_eq!(param.name, "loc");
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct MockDistribution {
        params: Vec<DistributionParam>,
    }

    impl Default for MockDistribution {
        fn default() -> Self {
            Self {
                params: vec![
                    DistributionParam::new("loc", ResponseFn::Identity),
                    DistributionParam::new("scale", ResponseFn::Softplus),
                ],
            }
        }
    }

    #[typetag::serde]
    impl Distribution for MockDistribution {
        fn clone_box(&self) -> Box<dyn Distribution> {
            Box::new(self.clone())
        }

        fn name(&self) -> &'static str {
            "MockDistribution"
        }

        fn n_params(&self) -> usize {
            self.params.len()
        }

        fn params(&self) -> &[DistributionParam] {
            &self.params
        }

        fn loss_fn(&self) -> LossFn {
            LossFn::Nll
        }

        fn stabilization(&self) -> Stabilization {
            Stabilization::None
        }

        fn should_initialize(&self) -> bool {
            true
        }

        fn nll(&self, _params: &ArrayView2<f64>, _target: &ResponseData) -> f64 {
            0.0
        }

        fn log_prob(&self, _params: &[f64], _target: &[f64]) -> f64 {
            0.0
        }

        fn sample(&self, _params: &ArrayView2<f64>, _n_samples: usize, _seed: u64) -> Array2<f64> {
            Array2::zeros((10, 100))
        }
    }

    #[test]
    fn test_crps_score() {
        let dist = MockDistribution::default();
        let target_array = Array1::from_vec(vec![0.5; 10]);
        let target = ResponseData::Univariate(&target_array.view());
        let samples = Array2::from_elem((100, 10), 0.5);
        let score = dist.crps_score(&target, &samples.view());

        assert!(score >= 0.0);
        assert!(score.is_finite());
    }
}

#[cfg(test)]
mod start_value_prephase_tests {
    /// Start values are a property of the data and the family, not of the
    /// training curvature: all three HessianModes must agree. (Before the
    /// relative curvature floor, Fisher's tiny df curvature sent ν into the
    /// ExpDf dead zone at 2 while Damped/Observed went to ν → ∞.)
    #[test]
    fn student_t_start_values_do_not_depend_on_hessian_mode() {
        use crate::distributions::StudentT;
        use super::HessianMode;
        // heavy-ish tails: Gaussian core plus a few wide rows
        let mut y = sample(3000, 7.0, 6.0);
        for i in (0..3000).step_by(37) {
            y[i] = 7.0 + 40.0 * if i % 2 == 0 { 1.0 } else { -1.0 } * ((i % 5) as f64 + 1.0) / 5.0;
        }
        let target = ResponseData::Univariate(&y.view());
        let mut results = Vec::new();
        for mode in [HessianMode::Observed, HessianMode::Fisher, HessianMode::Damped] {
            let dist = StudentT::new(Stabilization::None, ResponseFn::ExpDf, ResponseFn::Exp, LossFn::Nll, true)
                .with_hessian_mode(mode)
                .without_max_start_df();
            let (loss, sv) = dist.calculate_start_values(&target, 50).unwrap();
            results.push((mode, loss, sv));
        }
        let (_, l0, s0) = &results[0];
        for (mode, l, sv) in &results[1..] {
            assert!((l - l0).abs() <= 1e-6 * l0.abs().max(1.0), "{mode:?}: loss {l} vs {l0}");
            // loc and log-scale agree tightly; df may sit on the flat tail so compare its transformed value loosely
            assert!((sv[1] - s0[1]).abs() < 1e-4 && (sv[2] - s0[2]).abs() < 1e-4, "{mode:?}: {sv:?} vs {s0:?}");
        }
        // and none of them is in the ν = 2 dead zone
        for (mode, _, sv) in &results {
            assert!(sv[0] > -5.0, "{mode:?}: df raw {} is in the ExpDf dead zone", sv[0]);
        }
    }

    /// StudentT on near-Gaussian data: the likelihood is flat in ν for large ν
    /// and the ExpDf response overflows past η ≈ 710, where every row's
    /// log-prob is NaN and the nansum cost is 0. The pre-phase must never
    /// accept such a point: transformed params finite, cost ≤ cost at 0.5.
    #[test]
    fn student_t_start_values_stay_finite_on_light_tailed_data() {
        use crate::distributions::StudentT;
        for &(mean, sd) in &[(0.7, 0.26), (454.0, 17.0), (4.5e5, 1.7e4)] {
            let y = sample(3000, mean, sd);
            // no start cap here: this test checks the pre-phase, so the fitted ν must be free
            let dist = StudentT::new(Stabilization::None, ResponseFn::ExpDf, ResponseFn::Exp, LossFn::Nll, true)
                .without_max_start_df();
            let target = ResponseData::Univariate(&y.view());
            let (loss, sv) = dist.calculate_start_values(&target, 50).unwrap();
            let transformed: Vec<f64> = sv.iter().zip(dist.params()).map(|(&p, dp)| dp.response_fn.apply_scalar(p)).collect();
            assert!(transformed.iter().all(|v| v.is_finite()), "scale {sd}: {transformed:?}");
            let loss_at_half = dist.compute_total_loss(
                &[ResponseFn::ExpDf.apply_scalar(0.5), 0.5, ResponseFn::Exp.apply_scalar(0.5)],
                &target,
            );
            assert!(loss.is_finite() && loss <= loss_at_half, "scale {sd}: {loss} vs {loss_at_half}");
            // location and scale recovered; ν large (Gaussian-like) but finite
            assert!((sv[1] - y.mean().unwrap()).abs() <= 1e-2 * sd, "scale {sd}: loc {}", sv[1]);
            assert!((sv[2] - y.std(0.0).ln()).abs() <= 0.1, "scale {sd}: log scale {}", sv[2]);
            assert!(transformed[0] > 20.0, "scale {sd}: ν {} should be large on Gaussian data", transformed[0]);
        }
    }

    use super::*;
    use crate::distributions::Gaussian;
    use crate::types::ResponseData;
    use crate::utils::ResponseFn;

    /// Deterministic pseudo-normal sample (sum of 12 LCG uniforms).
    fn sample(n: usize, mean: f64, sd: f64) -> Array1<f64> {
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        Array1::from_iter((0..n).map(|_| {
            let z: f64 = (0..12).map(|_| next()).sum::<f64>() - 6.0;
            mean + sd * z
        }))
    }

    /// The fitted Gaussian start values must equal (ȳ, ln s_y) regardless of
    /// the target's scale — the exact case that used to return raw 0.5 for
    /// anything not already O(1).
    #[test]
    fn gaussian_start_values_are_scale_invariant() {
        for &(mean, sd) in &[(0.7, 0.26), (10.0, 2.8), (454.0, 17.0), (4.5e5, 1.7e4), (7e-4, 2.6e-4)] {
            let y = sample(4000, mean, sd);
            let ybar = y.mean().unwrap();
            let s_y = y.std(0.0);
            let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);
            let (loss, sv) = dist
                .calculate_start_values(&ResponseData::Univariate(&y.view()), 50)
                .unwrap();
            let loss_at_half = dist.compute_total_loss(
                &[0.5, ResponseFn::Exp.apply_scalar(0.5)],
                &ResponseData::Univariate(&y.view()),
            );
            assert!(loss <= loss_at_half, "scale {sd}: {loss} > {loss_at_half}");
            assert!(
                (sv[0] - ybar).abs() <= 1e-4 * sd.max(1e-9) + 1e-9,
                "scale {sd}: loc {} vs {ybar}",
                sv[0]
            );
            // Exp response is σ = e^η + 1e-6, so the ML raw value is ln(s_y − 1e-6).
            let expected = (s_y - 1e-6).ln();
            assert!(
                (sv[1] - expected).abs() <= 1e-3,
                "scale {sd}: log scale {} vs {expected}",
                sv[1]
            );
        }
    }
}
