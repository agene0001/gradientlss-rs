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

        // Per-sample computation closure. Each sample is independent.
        // Returns (grad_row, hess_row) for sample i.
        let compute_sample = |i: usize| -> (Vec<f64>, Vec<f64>) {
            let mut grad_row = vec![0.0f64; n_params];
            let mut hess_row = vec![0.0f64; n_params];

            // Thread-local buffers (stack-allocated for small n_params)
            let mut params_buf = vec![0.0f64; n_params];
            let mut y_buf = vec![0.0f64; n_targets];

            // Copy transformed params for this sample
            let row = transformed.row(i);
            for (j, v) in row.iter().enumerate() {
                params_buf[j] = *v;
            }

            // Fill target buffer
            match target {
                ResponseData::Univariate(arr) => {
                    y_buf[0] = arr[i];
                }
                ResponseData::Multivariate(arr) => {
                    let target_row = arr.row(i);
                    for (j, &v) in target_row.iter().enumerate() {
                        y_buf[j] = v;
                    }
                }
            }

            let y_slice = &y_buf[..n_targets];

            // For NLL path, compute base loss once per sample (shared across params)
            let loss_center = if !use_crps {
                -self.log_prob(&params_buf, y_slice)
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

                    let loss_plus =
                        self.crps_single_observation(&params_buf, y_slice, n_crps_samples, seed);
                    params_buf[p] = param_minus;
                    let loss_minus =
                        self.crps_single_observation(&params_buf, y_slice, n_crps_samples, seed);

                    grad_row[p] = (loss_plus - loss_minus) * inv_2eps;
                    hess_row[p] = 1.0;
                } else {
                    let loss_plus = -self.log_prob(&params_buf, y_slice);
                    params_buf[p] = param_minus;
                    let loss_minus = -self.log_prob(&params_buf, y_slice);

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

            (grad_row, hess_row)
        };

        // Compute gradients/hessians, using parallel iteration for large sample counts
        let results: Vec<(Vec<f64>, Vec<f64>)> = if use_parallel {
            (0..n_samples).into_par_iter().map(compute_sample).collect()
        } else {
            (0..n_samples).map(compute_sample).collect()
        };

        // Assemble results into output arrays
        for (i, (grad_row, hess_row)) in results.into_iter().enumerate() {
            for p in 0..n_params {
                gradients[[i, p]] = grad_row[p];
                hessians[[i, p]] = hess_row[p];
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
        // Create a 1-row params array for the sample method
        let params_arr = Array2::from_shape_vec((1, params.len()), params.to_vec()).unwrap();

        let samples = self.sample(&params_arr.view(), n_samples, seed);

        // Compute empirical CRPS for the single observation
        let y = target[0];
        let mut obs_samples: Vec<f64> = samples.column(0).to_vec();
        obs_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        let problem = StartValueProblem {
            target_data,
            is_multivariate,
            n_targets,
            n_params,
            response_fns,
            dist,
        };

        // Initial guess
        let init_params: Vec<f64> = vec![0.5; n_params];

        // Set up L-BFGS with More-Thuente line search (similar to PyTorch's strong_wolfe)
        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 7); // 7 is a common default for L-BFGS memory

        // Python uses max_iter outer epochs x min(max_iter/4, 20) inner L-BFGS iterations.
        // For the default max_iter=50, that's 50 * 12 = 600 total L-BFGS iterations.
        // We match this total budget.
        let inner_iters = (max_iter / 4).min(20);
        let total_iters = max_iter * inner_iters;

        // Run the optimizer
        let result = Executor::new(problem, solver)
            .configure(|state| {
                state
                    .param(init_params)
                    .max_iters(total_iters as u64)
                    .target_cost(0.0)
            })
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
        let mad = median_inplace(&mut buf).max(1e-4);

        let inv_mad = 1.0 / mad;
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
    target_data: Vec<f64>,
    is_multivariate: bool,
    n_targets: usize,
    n_params: usize,
    response_fns: Vec<ResponseFn>,
    dist: Box<dyn Distribution>,
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

        // Compute total negative log-likelihood using the actual distribution's log_prob
        let loss: f64 = if self.is_multivariate {
            let n_obs = self.target_data.len() / self.n_targets;
            (0..n_obs)
                .map(|i| {
                    let start = i * self.n_targets;
                    let end = start + self.n_targets;
                    let target_slice = &self.target_data[start..end];
                    -self.dist.log_prob(&transformed, target_slice)
                })
                .sum()
        } else {
            self.target_data
                .iter()
                .map(|&y| -self.dist.log_prob(&transformed, &[y]))
                .sum()
        };

        if loss.is_finite() {
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
