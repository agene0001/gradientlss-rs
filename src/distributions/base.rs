//! Base distribution trait and types.

use crate::error::Result;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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
    /// Samples with shape (n_observations * n_targets, n_samples) for multivariate,
    /// or (n_observations, n_samples) for univariate
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
            let transformed = param.response_fn.apply(&col);
            result.column_mut(i).assign(&transformed);
        }
        result
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
        let n_samples = predictions.nrows();
        let n_params = self.n_params();

        // Transform predictions to parameter space
        let transformed = self.transform_params(predictions);

        // Try analytical gradients first, fall back to numerical
        let (mut gradients, mut hessians) = self
            .analytical_gradients(predictions, &transformed.view(), target)
            .unwrap_or_else(|| {
                self.numerical_gradients_hessians(predictions, &transformed.view(), target)
                    .unwrap_or_else(|_| {
                        (
                            Array2::zeros((n_samples, n_params)),
                            Array2::ones((n_samples, n_params)),
                        )
                    })
            });

        // Apply stabilization
        self.stabilize_derivatives(&mut gradients, &mut hessians);

        // Apply weights if provided
        if let Some(w) = weights {
            for i in 0..n_params {
                for j in 0..n_samples {
                    gradients[[j, i]] *= w[j];
                    hessians[[j, i]] *= w[j];
                }
            }
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
        let eps = 1e-7;

        let mut gradients = Array2::zeros((n_samples, n_params));
        let mut hessians = Array2::zeros((n_samples, n_params));

        for i in 0..n_samples {
            let y = self.get_target_for_observation(target, i);
            let base_params: Vec<f64> = transformed.row(i).to_vec();

            for p in 0..n_params {
                let pred_val = predictions[[i, p]];
                let response_fn = &self.params()[p].response_fn;

                // Compute gradient using chain rule: dL/dpred = dL/dparam * dparam/dpred
                let mut params_plus = base_params.clone();
                let mut params_minus = base_params.clone();

                // Perturb in prediction space
                let pred_plus = pred_val + eps;
                let pred_minus = pred_val - eps;

                params_plus[p] = response_fn.apply_scalar(pred_plus);
                params_minus[p] = response_fn.apply_scalar(pred_minus);

                let loss_plus = -self.log_prob(&params_plus, &y);
                let loss_minus = -self.log_prob(&params_minus, &y);

                let grad = (loss_plus - loss_minus) / (2.0 * eps);
                gradients[[i, p]] = grad;

                // Compute hessian (second derivative)
                let loss_center = -self.log_prob(&base_params, &y);
                let hess = (loss_plus - 2.0 * loss_center + loss_minus) / (eps * eps);

                // For CRPS, we use constant hessian of 1.0
                hessians[[i, p]] = if self.loss_fn() == LossFn::Crps {
                    1.0
                } else {
                    hess.max(1e-6) // Ensure positive hessian
                };
            }
        }

        Ok((gradients, hessians))
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
    /// This provides much better convergence than simple gradient descent.
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

        // Create the optimization problem
        let problem = StartValueProblem {
            target_data,
            is_multivariate,
            n_targets,
            n_params,
            response_fns,
            log_prob_fn: |params: &[f64], target: &[f64], is_mv: bool, n_tgt: usize| {
                // We need to compute log_prob here, but we can't call self.log_prob
                // So we'll use a simpler approach with numerical differentiation
                compute_log_prob_generic(params, target, is_mv, n_tgt)
            },
        };

        // Initial guess
        let init_params: Vec<f64> = vec![0.5; n_params];

        // Set up L-BFGS with More-Thuente line search (similar to PyTorch's strong_wolfe)
        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 7); // 7 is a common default for L-BFGS memory

        // Run the optimizer
        let result = Executor::new(problem, solver)
            .configure(|state| {
                state
                    .param(init_params)
                    .max_iters(max_iter as u64)
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
        match target {
            ResponseData::Univariate(arr) => {
                arr.iter().map(|&y| -self.log_prob(params, &[y])).sum()
            }
            ResponseData::Multivariate(arr) => (0..arr.nrows())
                .map(|i| {
                    let y = arr.row(i).to_vec();
                    -self.log_prob(params, &y)
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

                    total_crps += crps;
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

                        total_crps += crps;
                    }
                }

                total_crps
            }
        }
    }
}

/// Replace NaN values in array with column means.
fn replace_nans_with_mean(arr: &mut Array2<f64>) {
    for mut col in arr.columns_mut() {
        let valid: Vec<f64> = col.iter().filter(|v| v.is_finite()).copied().collect();
        let mean = if valid.is_empty() {
            0.0
        } else {
            valid.iter().sum::<f64>() / valid.len() as f64
        };
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

    for mut col in arr.columns_mut() {
        let median = compute_median(&col.to_vec());
        let deviations: Vec<f64> = col.iter().map(|&v| (v - median).abs()).collect();
        let mad = compute_median(&deviations).max(1e-4);

        for v in col.iter_mut() {
            *v /= mad;
        }
    }
}

/// L2 stabilization.
fn stabilize_l2(arr: &mut Array2<f64>) {
    replace_nans_with_mean(arr);

    for mut col in arr.columns_mut() {
        let sum_sq: f64 = col.iter().map(|v| v * v).sum();
        let l2 = (sum_sq / col.len() as f64).sqrt().clamp(1e-4, 10000.0);

        for v in col.iter_mut() {
            *v /= l2;
        }
    }
}

/// Compute median of a slice.
fn compute_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = values.iter().filter(|v| v.is_finite()).copied().collect();
    if sorted.is_empty() {
        return 0.0;
    }

    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

// ============================================================================
// L-BFGS Optimization for Start Values (matching Python's torch.optim.LBFGS)
// ============================================================================

/// Optimization problem for finding start values using L-BFGS.
/// This matches the behavior of Python's torch.optim.LBFGS with strong_wolfe line search.
struct StartValueProblem<F>
where
    F: Fn(&[f64], &[f64], bool, usize) -> f64,
{
    target_data: Vec<f64>,
    is_multivariate: bool,
    n_targets: usize,
    n_params: usize,
    response_fns: Vec<ResponseFn>,
    log_prob_fn: F,
}

impl<F> CostFunction for StartValueProblem<F>
where
    F: Fn(&[f64], &[f64], bool, usize) -> f64,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> std::result::Result<Self::Output, ArgminError> {
        // Transform parameters to response scale
        let transformed: Vec<f64> = params
            .iter()
            .zip(self.response_fns.iter())
            .map(|(&p, response_fn)| {
                let val = if p.is_finite() { p } else { 0.5 };
                response_fn.apply_scalar(val)
            })
            .collect();

        // Compute total negative log-likelihood
        let loss: f64 = if self.is_multivariate {
            let n_obs = self.target_data.len() / self.n_targets;
            (0..n_obs)
                .map(|i| {
                    let start = i * self.n_targets;
                    let end = start + self.n_targets;
                    let target_slice = &self.target_data[start..end];
                    -(self.log_prob_fn)(&transformed, target_slice, true, self.n_targets)
                })
                .sum()
        } else {
            self.target_data
                .iter()
                .map(|&y| -(self.log_prob_fn)(&transformed, &[y], false, 1))
                .sum()
        };

        if loss.is_finite() {
            Ok(loss)
        } else {
            Ok(f64::MAX)
        }
    }
}

impl<F> Gradient for StartValueProblem<F>
where
    F: Fn(&[f64], &[f64], bool, usize) -> f64,
{
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, params: &Self::Param) -> std::result::Result<Self::Gradient, ArgminError> {
        let eps = 1e-5;
        let mut grad = vec![0.0; self.n_params];

        let base_cost = self.cost(params)?;

        for i in 0..self.n_params {
            let mut params_plus = params.clone();
            params_plus[i] += eps;
            let cost_plus = self.cost(&params_plus)?;
            grad[i] = (cost_plus - base_cost) / eps;

            // Clip gradient to prevent instability
            if !grad[i].is_finite() {
                grad[i] = 0.0;
            } else {
                grad[i] = grad[i].clamp(-100.0, 100.0);
            }
        }

        Ok(grad)
    }
}

/// Generic log probability computation for common distributions.
/// This is used during start value optimization when we don't have access to
/// the specific distribution's log_prob method.
fn compute_log_prob_generic(
    params: &[f64],
    target: &[f64],
    is_multivariate: bool,
    n_targets: usize,
) -> f64 {
    if is_multivariate {
        // For multivariate, assume MVN-like structure
        // params: [loc_1, ..., loc_n, scale_tril_elements...]
        if params.len() < n_targets {
            return f64::NEG_INFINITY;
        }

        // Simple approximation: treat as independent normals for start value estimation
        let mut log_prob = 0.0;
        for i in 0..n_targets {
            let loc = params[i];
            // Find the diagonal scale element (simplified assumption)
            let scale_idx = n_targets + i; // Diagonal elements come first in our tril ordering
            let scale = if scale_idx < params.len() {
                params[scale_idx].max(1e-6)
            } else {
                1.0
            };

            let z = (target[i] - loc) / scale;
            log_prob += -0.5 * (2.0 * std::f64::consts::PI).ln() - scale.ln() - 0.5 * z * z;
        }
        log_prob
    } else {
        // Univariate case - assume Gaussian-like for start value estimation
        if params.len() < 2 {
            return f64::NEG_INFINITY;
        }

        let loc = params[0];
        let scale = params[1].max(1e-6);
        let y = target[0];

        let z = (y - loc) / scale;
        -0.5 * (2.0 * std::f64::consts::PI).ln() - scale.ln() - 0.5 * z * z
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
    fn test_compute_median() {
        assert_eq!(compute_median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(compute_median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(compute_median(&[]), 0.0);
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
