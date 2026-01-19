//! Backend trait definitions.

use crate::distributions::GradientsAndHessians;
use crate::error::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Training configuration for gradient boosting.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of boosting rounds.
    pub num_boost_round: usize,
    /// Early stopping rounds (None to disable).
    /// Note: This is a simple counter-based early stopping. For more sophisticated
    /// early stopping (e.g., based on validation loss), use an EarlyStoppingCallback.
    pub early_stopping_rounds: Option<usize>,
    /// Whether to print verbose output.
    pub verbose: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_boost_round: 100,
            early_stopping_rounds: Some(20),
            verbose: true,
            seed: 123,
        }
    }
}

/// Training result containing metadata about the training run.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Total number of iterations completed.
    pub n_iterations: usize,
    /// Best iteration (if early stopping was used).
    pub best_iteration: Option<usize>,
    /// Best score achieved (if early stopping was used).
    pub best_score: Option<f64>,
    /// Training loss history.
    pub train_history: Vec<f64>,
    /// Validation loss history (if validation data was provided).
    pub valid_history: Vec<f64>,
    /// Whether training stopped early.
    pub stopped_early: bool,
}

/// Prediction output types.
#[derive(Debug, Clone)]
pub enum PredictionOutput {
    /// Raw distributional parameters.
    Parameters(Array2<f64>),
    /// Samples drawn from the predicted distribution.
    Samples(Array2<f64>),
    /// Quantiles computed from samples.
    Quantiles(Array2<f64>),
    /// Expectile values (for Expectile distribution).
    Expectiles(Array2<f64>),
    /// Distribution info with parameters for PDF/CDF evaluation.
    Distribution(DistributionInfo),
}

/// Distribution information for PDF/CDF evaluation.
///
/// This struct provides all the information needed to evaluate the predicted
/// distributions, including parameters and metadata about the distribution type.
#[derive(Debug, Clone)]
pub struct DistributionInfo {
    /// Name of the distribution (e.g., "Gaussian", "Gamma").
    pub dist_name: String,
    /// Predicted parameters (n_samples x n_params).
    pub params: Array2<f64>,
    /// Parameter names in order (e.g., ["loc", "scale"]).
    pub param_names: Vec<String>,
    /// Whether the distribution is univariate.
    pub is_univariate: bool,
    /// Number of target dimensions (1 for univariate).
    pub n_targets: usize,
}

/// Feature importance types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureImportanceType {
    /// Gain-based importance (default for tree models).
    Gain,
    /// Split count importance.
    Split,
    /// Cover-based importance (number of samples affected).
    Cover,
}

/// Feature importance result.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Feature indices.
    pub feature_indices: Vec<usize>,
    /// Feature names (if available).
    pub feature_names: Option<Vec<String>>,
    /// Importance scores per feature, per distributional parameter.
    /// Shape: (n_features, n_params) or (n_features,) for aggregated.
    pub scores: Array2<f64>,
    /// Type of importance computed.
    pub importance_type: FeatureImportanceType,
}

/// Training callback trait for monitoring and early stopping.
///
/// Callbacks are invoked during training to monitor progress, record metrics,
/// implement custom early stopping logic, or perform other actions.
///
/// # Example
///
/// ```rust,ignore
/// use gradientlss::backend::{TrainingCallback, CallbackAction};
///
/// struct MyCallback {
///     threshold: f64,
/// }
///
/// impl TrainingCallback for MyCallback {
///     fn on_iteration_end(
///         &mut self,
///         iteration: usize,
///         train_loss: f64,
///         valid_loss: Option<f64>,
///     ) -> CallbackAction {
///         if let Some(vl) = valid_loss {
///             if vl < self.threshold {
///                 return CallbackAction::Stop;
///             }
///         }
///         CallbackAction::Continue
///     }
/// }
/// ```
pub trait TrainingCallback: Send {
    /// Called at the start of training.
    ///
    /// # Arguments
    /// * `n_iterations` - The planned number of iterations (may not complete if early stopped)
    fn on_training_start(&mut self, _n_iterations: usize) {}

    /// Called at the end of each boosting iteration.
    ///
    /// # Arguments
    /// * `iteration` - The current iteration number (0-indexed)
    /// * `train_loss` - The training loss for this iteration
    /// * `valid_loss` - The validation loss for this iteration (if validation data was provided)
    ///
    /// # Returns
    /// A `CallbackAction` indicating whether to continue training or stop early.
    fn on_iteration_end(
        &mut self,
        iteration: usize,
        train_loss: f64,
        valid_loss: Option<f64>,
    ) -> CallbackAction;

    /// Called when training completes (either normally or due to early stopping).
    ///
    /// # Arguments
    /// * `total_iterations` - The total number of iterations completed
    /// * `stopped_early` - Whether training stopped early
    fn on_training_end(&mut self, _total_iterations: usize, _stopped_early: bool) {}

    /// Get the name of this callback for logging purposes.
    fn name(&self) -> &str {
        "TrainingCallback"
    }
}

/// Action returned by a callback to control training flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackAction {
    /// Continue training.
    Continue,
    /// Stop training immediately.
    Stop,
}

/// A simple callback that prints progress during training.
#[derive(Debug, Clone)]
pub struct PrintCallback {
    /// Print every N iterations.
    pub print_every: usize,
}

impl Default for PrintCallback {
    fn default() -> Self {
        Self { print_every: 10 }
    }
}

impl TrainingCallback for PrintCallback {
    fn on_iteration_end(
        &mut self,
        iteration: usize,
        train_loss: f64,
        valid_loss: Option<f64>,
    ) -> CallbackAction {
        if iteration % self.print_every == 0 || iteration == 0 {
            match valid_loss {
                Some(vl) => println!(
                    "[{}] train_loss: {:.6}, valid_loss: {:.6}",
                    iteration, train_loss, vl
                ),
                None => println!("[{}] train_loss: {:.6}", iteration, train_loss),
            }
        }
        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "PrintCallback"
    }
}

/// A callback that records training history for later analysis.
#[derive(Debug, Clone, Default)]
pub struct HistoryCallback {
    /// Training loss history.
    pub train_history: Vec<f64>,
    /// Validation loss history (if validation data provided).
    pub valid_history: Vec<f64>,
}

impl HistoryCallback {
    /// Create a new HistoryCallback.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the best training loss and its iteration.
    pub fn best_train(&self) -> Option<(usize, f64)> {
        self.train_history
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
    }

    /// Get the best validation loss and its iteration.
    pub fn best_valid(&self) -> Option<(usize, f64)> {
        self.valid_history
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
    }
}

impl TrainingCallback for HistoryCallback {
    fn on_training_start(&mut self, _n_iterations: usize) {
        self.train_history.clear();
        self.valid_history.clear();
    }

    fn on_iteration_end(
        &mut self,
        _iteration: usize,
        train_loss: f64,
        valid_loss: Option<f64>,
    ) -> CallbackAction {
        self.train_history.push(train_loss);
        if let Some(vl) = valid_loss {
            self.valid_history.push(vl);
        }
        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "HistoryCallback"
    }
}

/// Early stopping callback that monitors validation loss and stops when it stops improving.
///
/// This callback implements patience-based early stopping:
/// - Monitors validation loss (or training loss if no validation data)
/// - Stops training if the loss doesn't improve for `patience` consecutive iterations
/// - Optionally requires minimum improvement delta
///
/// # Example
///
/// ```rust,ignore
/// use gradientlss::backend::EarlyStoppingCallback;
///
/// // Stop if validation loss doesn't improve for 10 iterations
/// let callback = EarlyStoppingCallback::new(10);
///
/// // Stop if validation loss doesn't improve by at least 0.001 for 20 iterations
/// let callback = EarlyStoppingCallback::new(20).with_min_delta(0.001);
/// ```
#[derive(Debug, Clone)]
pub struct EarlyStoppingCallback {
    /// Number of iterations to wait for improvement before stopping.
    pub patience: usize,
    /// Minimum improvement required to reset the patience counter.
    pub min_delta: f64,
    /// Whether to monitor validation loss (true) or training loss (false).
    /// If true but no validation data is provided, falls back to training loss.
    pub monitor_validation: bool,
    /// Whether lower is better (true for loss, false for accuracy).
    pub minimize: bool,
    /// Best value seen so far.
    best_value: f64,
    /// Iteration at which best value was observed.
    best_iteration: usize,
    /// Number of iterations without improvement.
    iterations_without_improvement: usize,
    /// Whether to print when stopping.
    pub verbose: bool,
}

impl EarlyStoppingCallback {
    /// Create a new EarlyStoppingCallback with the given patience.
    pub fn new(patience: usize) -> Self {
        Self {
            patience,
            min_delta: 0.0,
            monitor_validation: true,
            minimize: true,
            best_value: f64::INFINITY,
            best_iteration: 0,
            iterations_without_improvement: 0,
            verbose: true,
        }
    }

    /// Set the minimum delta for improvement.
    pub fn with_min_delta(mut self, min_delta: f64) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// Set whether to monitor validation loss (default: true).
    /// If true but no validation data is provided, falls back to training loss.
    pub fn with_monitor_validation(mut self, monitor_validation: bool) -> Self {
        self.monitor_validation = monitor_validation;
        self
    }

    /// Set whether to minimize (default: true for loss).
    pub fn with_minimize(mut self, minimize: bool) -> Self {
        self.minimize = minimize;
        if !minimize {
            self.best_value = f64::NEG_INFINITY;
        }
        self
    }

    /// Set verbosity.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Get the best value observed during training.
    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// Get the iteration at which the best value was observed.
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }

    fn is_improvement(&self, current: f64) -> bool {
        if self.minimize {
            current < self.best_value - self.min_delta
        } else {
            current > self.best_value + self.min_delta
        }
    }
}

impl TrainingCallback for EarlyStoppingCallback {
    fn on_training_start(&mut self, _n_iterations: usize) {
        self.best_value = if self.minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        self.best_iteration = 0;
        self.iterations_without_improvement = 0;
    }

    fn on_iteration_end(
        &mut self,
        iteration: usize,
        train_loss: f64,
        valid_loss: Option<f64>,
    ) -> CallbackAction {
        // Determine which value to monitor
        let current_value = if self.monitor_validation {
            valid_loss.unwrap_or(train_loss)
        } else {
            train_loss
        };

        if self.is_improvement(current_value) {
            self.best_value = current_value;
            self.best_iteration = iteration;
            self.iterations_without_improvement = 0;
        } else {
            self.iterations_without_improvement += 1;

            if self.iterations_without_improvement >= self.patience {
                if self.verbose {
                    let metric_name = if self.monitor_validation && valid_loss.is_some() {
                        "validation loss"
                    } else {
                        "training loss"
                    };
                    println!(
                        "Early stopping: {} has not improved from {:.6} for {} iterations. \
                         Best iteration: {}",
                        metric_name, self.best_value, self.patience, self.best_iteration
                    );
                }
                return CallbackAction::Stop;
            }
        }

        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "EarlyStoppingCallback"
    }
}

/// A composite callback that runs multiple callbacks in sequence.
///
/// If any callback returns `CallbackAction::Stop`, training will stop.
pub struct CallbackList {
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl CallbackList {
    /// Create a new empty CallbackList.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback to the list.
    pub fn add<C: TrainingCallback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
    }

    /// Add a callback to the list (builder pattern).
    pub fn with<C: TrainingCallback + 'static>(mut self, callback: C) -> Self {
        self.add(callback);
        self
    }

    /// Get the number of callbacks.
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Check if the callback list is empty.
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }
}

impl Default for CallbackList {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingCallback for CallbackList {
    fn on_training_start(&mut self, n_iterations: usize) {
        for callback in &mut self.callbacks {
            callback.on_training_start(n_iterations);
        }
    }

    fn on_iteration_end(
        &mut self,
        iteration: usize,
        train_loss: f64,
        valid_loss: Option<f64>,
    ) -> CallbackAction {
        for callback in &mut self.callbacks {
            if callback.on_iteration_end(iteration, train_loss, valid_loss) == CallbackAction::Stop
            {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    fn on_training_end(&mut self, total_iterations: usize, stopped_early: bool) {
        for callback in &mut self.callbacks {
            callback.on_training_end(total_iterations, stopped_early);
        }
    }

    fn name(&self) -> &str {
        "CallbackList"
    }
}

/// Learning rate scheduler callback that adjusts learning rate during training.
///
/// Note: This callback stores the scheduled learning rate values, but the actual
/// application of the learning rate depends on backend support.
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    /// The learning rate schedule (iteration -> learning rate).
    schedule: LearningRateSchedule,
    /// Initial learning rate.
    initial_lr: f64,
    /// Current learning rate.
    pub current_lr: f64,
}

/// Learning rate schedule types.
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate (no scheduling).
    Constant,
    /// Step decay: multiply by factor every step_size iterations.
    StepDecay { step_size: usize, factor: f64 },
    /// Exponential decay: lr = initial_lr * decay^iteration.
    ExponentialDecay { decay: f64 },
    /// Cosine annealing: lr decreases following a cosine curve.
    CosineAnnealing { t_max: usize, eta_min: f64 },
    /// Custom schedule function.
    Custom(fn(usize, f64) -> f64),
}

impl LearningRateScheduler {
    /// Create a new learning rate scheduler.
    pub fn new(initial_lr: f64, schedule: LearningRateSchedule) -> Self {
        Self {
            schedule,
            initial_lr,
            current_lr: initial_lr,
        }
    }

    /// Create a step decay scheduler.
    pub fn step_decay(initial_lr: f64, step_size: usize, factor: f64) -> Self {
        Self::new(
            initial_lr,
            LearningRateSchedule::StepDecay { step_size, factor },
        )
    }

    /// Create an exponential decay scheduler.
    pub fn exponential_decay(initial_lr: f64, decay: f64) -> Self {
        Self::new(initial_lr, LearningRateSchedule::ExponentialDecay { decay })
    }

    /// Create a cosine annealing scheduler.
    pub fn cosine_annealing(initial_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self::new(
            initial_lr,
            LearningRateSchedule::CosineAnnealing { t_max, eta_min },
        )
    }

    fn compute_lr(&self, iteration: usize) -> f64 {
        match &self.schedule {
            LearningRateSchedule::Constant => self.initial_lr,
            LearningRateSchedule::StepDecay { step_size, factor } => {
                self.initial_lr * factor.powi((iteration / step_size) as i32)
            }
            LearningRateSchedule::ExponentialDecay { decay } => {
                self.initial_lr * decay.powi(iteration as i32)
            }
            LearningRateSchedule::CosineAnnealing { t_max, eta_min } => {
                let t_max = *t_max as f64;
                let eta_min = *eta_min;
                let iteration = iteration as f64;
                eta_min
                    + (self.initial_lr - eta_min)
                        * (1.0 + (std::f64::consts::PI * iteration / t_max).cos())
                        / 2.0
            }
            LearningRateSchedule::Custom(f) => f(iteration, self.initial_lr),
        }
    }
}

impl TrainingCallback for LearningRateScheduler {
    fn on_training_start(&mut self, _n_iterations: usize) {
        self.current_lr = self.initial_lr;
    }

    fn on_iteration_end(
        &mut self,
        iteration: usize,
        _train_loss: f64,
        _valid_loss: Option<f64>,
    ) -> CallbackAction {
        self.current_lr = self.compute_lr(iteration + 1);
        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "LearningRateScheduler"
    }
}

/// Backend-specific hyperparameters.
pub trait BackendParams: Default + Clone {
    /// Set a parameter by name.
    fn set(&mut self, key: &str, value: ParamValue);

    /// Get a parameter value.
    fn get(&self, key: &str) -> Option<&ParamValue>;

    /// Convert to a hashmap for the backend.
    fn to_map(&self) -> HashMap<String, ParamValue>;
}

/// Parameter value types.
#[derive(Debug, Clone)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        ParamValue::Int(v)
    }
}

impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        ParamValue::Float(v)
    }
}

impl From<&str> for ParamValue {
    fn from(v: &str) -> Self {
        ParamValue::String(v.to_string())
    }
}

impl From<bool> for ParamValue {
    fn from(v: bool) -> Self {
        ParamValue::Bool(v)
    }
}

/// Backend-specific dataset wrapper.
pub trait BackendDataset: Sized {
    /// Create a dataset from features and labels.
    fn from_data(features: ArrayView2<f64>, labels: ArrayView1<f64>) -> Result<Self>;

    /// Set the initial score/base margin.
    fn set_init_score(&mut self, init_score: &Array1<f64>) -> Result<()>;

    /// Get the number of rows.
    fn num_rows(&self) -> usize;

    /// Get the labels.
    fn get_labels(&self) -> Result<Array1<f64>>;

    /// Get the number of target dimensions (for multivariate support).
    fn n_targets(&self) -> usize {
        1
    }
}

/// Backend-specific model wrapper.
pub trait BackendModel: Sized {
    type Dataset: BackendDataset;
    type Params: BackendParams;

    /// Train a model with custom objective.
    ///
    /// This is the basic training method without callback support.
    /// For training with callbacks, use `train_with_objective_and_callbacks`.
    fn train_with_objective<F, M>(
        params: &Self::Params,
        train_data: &mut Self::Dataset,
        valid_data: Option<&mut Self::Dataset>,
        config: &TrainConfig,
        objective_fn: F,
        metric_fn: M,
        start_values: Option<&Array1<f64>>,
    ) -> Result<Self>
    where
        F: Fn(&Array2<f64>, &Array1<f64>, Option<&Array1<f64>>) -> Result<GradientsAndHessians>,
        M: Fn(&Array2<f64>, &Array1<f64>) -> f64;

    /// Train a model with custom objective and callbacks.
    ///
    /// This method supports training with validation data monitoring and custom callbacks
    /// for advanced early stopping, logging, learning rate scheduling, etc.
    ///
    /// # Arguments
    /// * `params` - Backend-specific training parameters
    /// * `train_data` - Training dataset
    /// * `valid_data` - Optional validation dataset for monitoring
    /// * `config` - Training configuration
    /// * `objective_fn` - Custom objective function returning gradients and hessians
    /// * `metric_fn` - Metric function for evaluation
    /// * `start_values` - Optional start values for distributional parameters
    /// * `callbacks` - Optional callbacks for monitoring and control
    ///
    /// # Returns
    /// A tuple of (trained model, training result with history and metadata)
    fn train_with_objective_and_callbacks<F, M, C>(
        params: &Self::Params,
        train_data: &mut Self::Dataset,
        valid_data: Option<&mut Self::Dataset>,
        config: &TrainConfig,
        objective_fn: F,
        metric_fn: M,
        start_values: Option<&Array1<f64>>,
        callbacks: Option<&mut C>,
    ) -> Result<(Self, TrainingResult)>
    where
        F: Fn(&Array2<f64>, &Array1<f64>, Option<&Array1<f64>>) -> Result<GradientsAndHessians>,
        M: Fn(&Array2<f64>, &Array1<f64>) -> f64,
        C: TrainingCallback;

    /// Predict raw margin/scores.
    fn predict_raw(&self, data: &ArrayView2<f64>) -> Result<Array2<f64>>;

    /// Save model to a writer.
    fn save_to_writer<W: std::io::Write>(&self, writer: &mut W) -> Result<()>;

    /// Load model from a reader.
    fn load_from_reader<R: std::io::Read>(reader: &mut R) -> Result<Self>;

    /// Get feature importance scores.
    ///
    /// Returns importance scores for each feature across all distributional parameters.
    fn feature_importance(
        &self,
        importance_type: FeatureImportanceType,
        feature_names: Option<Vec<String>>,
    ) -> Result<FeatureImportance>;

    /// Get the number of features the model was trained on.
    fn num_features(&self) -> usize;

    /// Get the number of distributional parameters.
    fn num_params(&self) -> usize;
}

/// The main backend trait that ties everything together.
pub trait Backend: Sized + 'static {
    /// The dataset type for this backend.
    type Dataset: BackendDataset;
    /// The model type for this backend.
    type Model: BackendModel<Dataset = Self::Dataset, Params = Self::Params>;
    /// The parameter type for this backend.
    type Params: BackendParams;

    /// Get the backend name.
    fn name() -> &'static str;

    /// Create default parameters adjusted for distributional regression.
    fn create_params(n_dist_params: usize) -> Self::Params;

    /// Reshape gradients and hessians for this backend.
    ///
    /// Different backends expect gradients in different formats:
    /// - XGBoost: 2D array (n_samples, n_params)
    /// - LightGBM: Flattened with Fortran ordering
    fn reshape_gradients(
        gradients: &Array2<f64>,
        hessians: &Array2<f64>,
    ) -> (Array1<f64>, Array1<f64>);
}

/// Simple parameter storage implementation.
#[derive(Debug, Clone, Default)]
pub struct SimpleParams {
    params: HashMap<String, ParamValue>,
}

impl BackendParams for SimpleParams {
    fn set(&mut self, key: &str, value: ParamValue) {
        self.params.insert(key.to_string(), value);
    }

    fn get(&self, key: &str) -> Option<&ParamValue> {
        self.params.get(key)
    }

    fn to_map(&self) -> HashMap<String, ParamValue> {
        self.params.clone()
    }
}
