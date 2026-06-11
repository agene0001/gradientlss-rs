use crate::backend::{
    Backend, BackendDataset, BackendModel, FeatureImportance, FeatureImportanceType,
    PredictionOutput, TrainConfig, TrainingCallback, TrainingResult,
};
use crate::distributions::{Distribution, GradientsAndHessians, LossFn};
use crate::error::{GradientLSSError, Result};
use crate::hyper_opt;
use crate::types::ResponseData;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Cache mapping a raw training margin to its transformed parameters, shared
/// between the metric closure (writer) and the objective closure (reader).
///
/// Every boosting round the metric transforms the post-update train margin, and
/// the next round's objective receives that *exact* margin — so caching the
/// transform skips one full response-transform pass (exp/sigmoid over
/// n_samples × n_params) per round. Entries are keyed by exact value equality
/// on the raw margin, so a mismatch is never wrong, just a recompute. Two slots
/// because the validation metric runs right after the train metric each round
/// and must not evict the train entry.
type TransformCache = Arc<Mutex<Vec<(Array2<f64>, Array2<f64>)>>>;

/// Prediction types for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredType {
    /// Return distributional parameters.
    Parameters,
    /// Draw samples from the predicted distribution.
    Samples,
    /// Calculate quantiles from samples.
    Quantiles,
    /// Return expectile values (for Expectile distribution, same as Parameters).
    Expectiles,
    /// Return distribution info for PDF/CDF evaluation.
    /// Returns parameters along with distribution metadata (name, param names, bounds).
    Distribution,
}

/// Struct to hold the serializable state of the model
#[derive(Serialize, Deserialize)]
struct SerializableState {
    dist: Box<dyn Distribution>,
    start_values: Option<Array1<f64>>,
}

/// GradientLSS model with a specific backend.
pub struct GradientLSS<B: Backend> {
    dist: Arc<dyn Distribution>,
    model: Option<B::Model>,
    start_values: Option<Array1<f64>>,
    _backend: PhantomData<B>,
}

impl<B: Backend> GradientLSS<B> {
    pub fn new(dist: Arc<dyn Distribution>) -> Self {
        Self {
            dist,
            model: None,
            start_values: None,
            _backend: PhantomData,
        }
    }

    /// Save the model to a file.
    pub fn save(&self, path: &str) -> Result<()> {
        let model = self
            .model
            .as_ref()
            .ok_or(GradientLSSError::ModelNotTrained)?;

        // Create a serializable state object
        let state = SerializableState {
            dist: self.dist.clone_box(),
            start_values: self.start_values.clone(),
        };

        // Serialize the state using bincode
        let state_bytes = bincode::serialize(&state)
            .map_err(|e| GradientLSSError::SerializationError(e.to_string()))?;

        // Write state and model to a single file, with a separator
        let mut file = File::create(path).map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        // Write state length, then state
        let state_len = state_bytes.len() as u64;
        file.write_all(&state_len.to_le_bytes())
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        file.write_all(&state_bytes)
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        // Let the backend model save itself to the same file
        model.save_to_writer(&mut file)?;

        Ok(())
    }

    /// Load the model from a file.
    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        // Read state length and state
        let mut state_len_bytes = [0u8; 8];
        file.read_exact(&mut state_len_bytes)
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        let state_len = u64::from_le_bytes(state_len_bytes) as usize;

        let mut state_bytes = vec![0u8; state_len];
        file.read_exact(&mut state_bytes)
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        // Deserialize the state using bincode
        let state: SerializableState = bincode::deserialize(&state_bytes)
            .map_err(|e| GradientLSSError::SerializationError(e.to_string()))?;

        // Let the backend model load itself from the rest of the file
        let model = B::Model::load_from_reader(&mut file)?;

        Ok(Self {
            dist: Arc::from(state.dist),
            model: Some(model),
            start_values: state.start_values,
            _backend: PhantomData,
        })
    }

    pub fn distribution(&self) -> &dyn Distribution {
        self.dist.as_ref()
    }

    pub fn n_params(&self) -> usize {
        self.dist.n_params()
    }

    pub fn is_trained(&self) -> bool {
        self.model.is_some()
    }

    fn calculate_start_values(&mut self, labels: &ResponseData) -> Result<()> {
        if self.dist.should_initialize() {
            let (_, start_vals) = self.dist.calculate_start_values(labels, 50)?;
            self.start_values = Some(start_vals);
        } else {
            self.start_values = Some(Array1::from_elem(self.n_params(), 0.5));
        }
        Ok(())
    }

    fn set_init_score(&self, dataset: &mut B::Dataset) -> Result<()> {
        if let Some(ref start_vals) = self.start_values {
            let n_samples = dataset.num_rows();
            let n_params = self.n_params();
            // Row-major (sample-major) layout: [s0_p0, s0_p1, ..., s1_p0, ...],
            // matching what the backends' `set_init_score` / `set_base_margin`
            // expect for multi-output. (The XGBoost training path overwrites this
            // and LightGBM's setter is a no-op, so this is currently inert — but
            // the column-major layout it previously built would be a latent bug.)
            let mut init_score = Array1::zeros(n_samples * n_params);
            for i in 0..n_samples {
                for j in 0..n_params {
                    init_score[i * n_params + j] = start_vals[j];
                }
            }
            dataset.set_init_score(&init_score)?;
        }
        Ok(())
    }

    /// Compute the NLL of already-transformed parameters against `labels`,
    /// handling the univariate/multivariate reshape. Shared by the metric closure
    /// (both the cache-hit and recompute paths).
    fn nll_on_transformed(&self, transformed: &Array2<f64>, labels: &Array1<f64>) -> f64 {
        let n_targets = self.dist.n_targets();
        if n_targets == 1 {
            let target = ResponseData::Univariate(&labels.view());
            self.dist.nll(&transformed.view(), &target)
        } else {
            let n_samples = labels.len() / n_targets;
            match labels.view().into_shape_with_order((n_samples, n_targets)) {
                Ok(reshaped) => {
                    let target = ResponseData::Multivariate(&reshaped);
                    self.dist.nll(&transformed.view(), &target)
                }
                Err(_) => f64::INFINITY,
            }
        }
    }

    /// Compute the configured training loss on transformed parameters. Python's
    /// `metric_fn` evaluates whichever loss the distribution was set up with —
    /// for "crps" it draws 30 samples (seed 123) and sums the empirical CRPS,
    /// not the NLL.
    fn metric_on_transformed(&self, transformed: &Array2<f64>, labels: &Array1<f64>) -> f64 {
        match self.dist.loss_fn() {
            LossFn::Nll => self.nll_on_transformed(transformed, labels),
            LossFn::Crps => {
                let samples = self.dist.sample(&transformed.view(), 30, 123);
                let n_targets = self.dist.n_targets();
                if n_targets == 1 {
                    let target = ResponseData::Univariate(&labels.view());
                    self.dist.crps_score(&target, &samples.view())
                } else {
                    let n_samples = labels.len() / n_targets;
                    match labels.view().into_shape_with_order((n_samples, n_targets)) {
                        Ok(reshaped) => {
                            let target = ResponseData::Multivariate(&reshaped);
                            self.dist.crps_score(&target, &samples.view())
                        }
                        Err(_) => f64::INFINITY,
                    }
                }
            }
        }
    }

    /// Replace non-finite raw margins with the per-parameter unconditional start
    /// values, mirroring Python's `get_params_loss`:
    ///   `predt[nan_inf_mask] = np.take(start_values, np.where(nan_inf_mask)[1])`
    /// Returns `None` when nothing needs replacing (the common case) so callers
    /// can avoid a copy.
    fn clean_predictions(&self, predictions: &Array2<f64>) -> Option<Array2<f64>> {
        let sv = self.start_values.as_ref()?;
        if predictions.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut cleaned = predictions.clone();
        for mut row in cleaned.rows_mut() {
            for (j, v) in row.iter_mut().enumerate() {
                if !v.is_finite() {
                    *v = sv[j];
                }
            }
        }
        Some(cleaned)
    }

    fn create_objective_fn(
        &self,
        cache: TransformCache,
    ) -> impl Fn(&Array2<f64>, &Array1<f64>, Option<&Array1<f64>>) -> Result<GradientsAndHessians> + '_
    {
        let n_targets = self.dist.n_targets();
        move |predictions, labels, weights| {
            // Reuse the transform the metric pass computed for this exact margin
            // at the end of the previous round (keyed on the raw, pre-clean
            // margin; a miss just recomputes).
            let cached = {
                let mut guard = cache.lock().unwrap();
                guard
                    .iter()
                    .position(|(key, _)| key == predictions)
                    .map(|i| guard.swap_remove(i).1)
            };

            // Non-finite margins are replaced with the start values, as in Python.
            let cleaned = self.clean_predictions(predictions);
            let predictions = cleaned.as_ref().unwrap_or(predictions);
            let transformed =
                cached.unwrap_or_else(|| self.dist.transform_params(&predictions.view()));
            let w = weights.map(|w| w.view());
            if n_targets == 1 {
                let target = ResponseData::Univariate(&labels.view());
                self.dist.compute_gradients_and_hessians_with_transformed(
                    &predictions.view(),
                    &transformed.view(),
                    &target,
                    w.as_ref(),
                )
            } else {
                let n_samples = labels.len() / n_targets;
                let reshaped = labels
                    .view()
                    .into_shape_with_order((n_samples, n_targets))
                    .map_err(GradientLSSError::from)?;
                let target = ResponseData::Multivariate(&reshaped);
                self.dist.compute_gradients_and_hessians_with_transformed(
                    &predictions.view(),
                    &transformed.view(),
                    &target,
                    w.as_ref(),
                )
            }
        }
    }

    fn create_metric_fn(
        &self,
        cache: TransformCache,
    ) -> impl Fn(&Array2<f64>, &Array1<f64>) -> f64 + '_ {
        move |predictions, labels| {
            // Non-finite margins are replaced with the start values, as in Python.
            let cleaned = self.clean_predictions(predictions);
            let cleaned_preds = cleaned.as_ref().unwrap_or(predictions);
            let transformed = self.dist.transform_params(&cleaned_preds.view());
            let loss = self.metric_on_transformed(&transformed, labels);

            // Stash (raw margin → transform) for the next round's objective; see
            // `TransformCache`. Oldest entry is evicted at 2 slots.
            let mut guard = cache.lock().unwrap();
            if let Some(i) = guard.iter().position(|(key, _)| key == predictions) {
                guard.swap_remove(i);
            }
            if guard.len() >= 2 {
                guard.remove(0);
            }
            guard.push((predictions.clone(), transformed));

            loss
        }
    }

    /// Train the model on the provided data.
    ///
    /// This is the basic training method without validation monitoring or callbacks.
    /// For more advanced training with validation-based early stopping and callbacks,
    /// use `train_with_callbacks`.
    ///
    /// # Arguments
    /// * `train_data` - Training dataset
    /// * `valid_data` - Optional validation dataset (currently unused, use `train_with_callbacks` for validation monitoring)
    /// * `params` - Backend-specific parameters
    /// * `config` - Training configuration
    pub fn train(
        &mut self,
        train_data: &mut B::Dataset,
        valid_data: Option<&mut B::Dataset>,
        params: B::Params,
        config: TrainConfig,
    ) -> Result<()> {
        let (_, _) = self.train_with_callbacks(
            train_data,
            valid_data,
            params,
            config,
            None::<&mut crate::backend::HistoryCallback>,
        )?;
        Ok(())
    }

    /// Train the model with validation monitoring and callbacks.
    ///
    /// This method provides full control over the training process, including:
    /// - Validation data monitoring for early stopping
    /// - Custom callbacks for logging, learning rate scheduling, etc.
    /// - Training history tracking
    ///
    /// # Arguments
    /// * `train_data` - Training dataset
    /// * `valid_data` - Optional validation dataset for monitoring
    /// * `params` - Backend-specific parameters
    /// * `config` - Training configuration
    /// * `callbacks` - Optional callback(s) for training control and monitoring
    ///
    /// # Returns
    /// A tuple of ((), TrainingResult) containing training history and metadata.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use gradientlss::backend::{EarlyStoppingCallback, HistoryCallback, CallbackList};
    ///
    /// // Simple early stopping on validation loss
    /// let mut early_stopping = EarlyStoppingCallback::new(10);
    /// let (_, result) = model.train_with_callbacks(
    ///     &mut train_data,
    ///     Some(&mut valid_data),
    ///     params,
    ///     config,
    ///     Some(&mut early_stopping),
    /// )?;
    /// println!("Best iteration: {:?}", result.best_iteration);
    ///
    /// // Multiple callbacks
    /// let mut callbacks = CallbackList::new()
    ///     .with(EarlyStoppingCallback::new(10))
    ///     .with(HistoryCallback::new());
    /// let (_, result) = model.train_with_callbacks(
    ///     &mut train_data,
    ///     Some(&mut valid_data),
    ///     params,
    ///     config,
    ///     Some(&mut callbacks),
    /// )?;
    /// ```
    pub fn train_with_callbacks<C: TrainingCallback>(
        &mut self,
        train_data: &mut B::Dataset,
        valid_data: Option<&mut B::Dataset>,
        params: B::Params,
        config: TrainConfig,
        callbacks: Option<&mut C>,
    ) -> Result<((), TrainingResult)> {
        let labels = train_data.get_labels()?;
        if self.dist.is_univariate() {
            let target = ResponseData::Univariate(&labels.view());
            self.calculate_start_values(&target)?;
        } else {
            // For multivariate, we need to reshape the labels
            // This assumes the backend provides labels in the correct multivariate format
            let n_samples = labels.len() / self.dist.n_targets();
            let reshaped = labels
                .view()
                .into_shape_with_order((n_samples, self.dist.n_targets()))?;
            let target = ResponseData::Multivariate(&reshaped);
            self.calculate_start_values(&target)?;
        }

        self.set_init_score(train_data)?;

        // Shared per-run transform cache — see `TransformCache`.
        let transform_cache: TransformCache = Arc::new(Mutex::new(Vec::with_capacity(2)));

        let (model, result) = B::Model::train_with_objective_and_callbacks(
            &params,
            train_data,
            valid_data,
            &config,
            self.create_objective_fn(Arc::clone(&transform_cache)),
            self.create_metric_fn(transform_cache),
            self.start_values.as_ref(),
            callbacks,
        )?;

        self.model = Some(model);
        Ok(((), result))
    }

    pub fn cv(
        &mut self,
        features: &Array2<f64>,
        labels: &Array1<f64>,
        n_folds: usize,
        params: B::Params,
        config: TrainConfig,
    ) -> Result<f64> {
        let n_samples = features.nrows();
        let fold_size = n_samples / n_folds;
        let mut scores = Vec::new();

        for i in 0..n_folds {
            let test_start = i * fold_size;
            let test_end = if i == n_folds - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test_features = features.slice(s![test_start..test_end, ..]);
            let test_labels = labels.slice(s![test_start..test_end]);

            let train_features_1 = features.slice(s![..test_start, ..]);
            let train_labels_1 = labels.slice(s![..test_start]);
            let train_features_2 = features.slice(s![test_end.., ..]);
            let train_labels_2 = labels.slice(s![test_end..]);

            let train_features =
                ndarray::concatenate(Axis(0), &[train_features_1, train_features_2])?;
            let train_labels = ndarray::concatenate(Axis(0), &[train_labels_1, train_labels_2])?;

            let mut train_data = B::Dataset::from_data(train_features.view(), train_labels.view())?;

            let mut model = GradientLSS::<B>::new(self.dist.clone());
            model.train(&mut train_data, None, params.clone(), config.clone())?;

            // Score on the predicted *parameters*, which add the model's start
            // values (base margin) before transforming — `predict_raw` omits them,
            // so scoring its output transforms the wrong point. Mirrors hyper_opt.
            let preds = match model.predict(&test_features, PredType::Parameters, 0, &[], 0)? {
                PredictionOutput::Parameters(p) => p,
                _ => unreachable!("PredType::Parameters yields Parameters"),
            };
            let target = ResponseData::Univariate(&test_labels);
            let score = self.dist.nll(&preds.view(), &target);
            scores.push(score);
        }

        let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
        Ok(avg_score)
    }

    /// Cross-validation with stratified sampling.
    ///
    /// This method performs stratified k-fold cross-validation, which attempts
    /// to preserve the distribution of the target variable in each fold.
    /// Useful when the target has imbalanced classes or distinct groups.
    ///
    /// # Arguments
    /// * `features` - Feature matrix
    /// * `labels` - Target labels
    /// * `n_folds` - Number of folds
    /// * `n_bins` - Number of bins for stratification (continuous targets are binned)
    /// * `params` - Backend parameters
    /// * `config` - Training configuration
    /// * `shuffle` - Whether to shuffle before splitting
    /// * `seed` - Random seed for shuffling
    pub fn cv_stratified(
        &mut self,
        features: &Array2<f64>,
        labels: &Array1<f64>,
        n_folds: usize,
        n_bins: usize,
        params: B::Params,
        config: TrainConfig,
        shuffle: bool,
        seed: u64,
    ) -> Result<f64> {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;

        let n_samples = features.nrows();

        // Create indices
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Optionally shuffle
        if shuffle {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }

        // Bin labels for stratification
        let min_label = labels.iter().copied().fold(f64::INFINITY, f64::min);
        let max_label = labels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bin_width = (max_label - min_label) / n_bins as f64;

        // Assign each sample to a bin
        let bins: Vec<usize> = labels
            .iter()
            .map(|&l| {
                let bin = ((l - min_label) / bin_width).floor() as usize;
                bin.min(n_bins - 1)
            })
            .collect();

        // Group indices by bin
        let mut bin_indices: Vec<Vec<usize>> = vec![Vec::new(); n_bins];
        for &idx in &indices {
            let bin = bins[idx];
            bin_indices[bin].push(idx);
        }

        // Create stratified folds
        let mut folds: Vec<Vec<usize>> = vec![Vec::new(); n_folds];
        for bin_idx_list in &bin_indices {
            for (i, &idx) in bin_idx_list.iter().enumerate() {
                folds[i % n_folds].push(idx);
            }
        }

        let mut scores = Vec::new();

        for fold_idx in 0..n_folds {
            let test_indices = &folds[fold_idx];
            let train_indices: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != fold_idx)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();

            // Extract train and test data
            let n_train = train_indices.len();
            let n_test = test_indices.len();
            let n_cols = features.ncols();

            let mut train_features = Array2::zeros((n_train, n_cols));
            let mut train_labels = Array1::zeros(n_train);
            for (i, &idx) in train_indices.iter().enumerate() {
                train_features.row_mut(i).assign(&features.row(idx));
                train_labels[i] = labels[idx];
            }

            let mut test_features = Array2::zeros((n_test, n_cols));
            let mut test_labels = Array1::zeros(n_test);
            for (i, &idx) in test_indices.iter().enumerate() {
                test_features.row_mut(i).assign(&features.row(idx));
                test_labels[i] = labels[idx];
            }

            let mut train_data = B::Dataset::from_data(train_features.view(), train_labels.view())?;

            let mut model = GradientLSS::<B>::new(self.dist.clone());
            model.train(&mut train_data, None, params.clone(), config.clone())?;

            // Score on predicted parameters (start values added) — see `cv`.
            let preds = match model.predict(&test_features.view(), PredType::Parameters, 0, &[], 0)? {
                PredictionOutput::Parameters(p) => p,
                _ => unreachable!("PredType::Parameters yields Parameters"),
            };
            let test_labels_view = test_labels.view();
            let target = ResponseData::Univariate(&test_labels_view);
            let score = self.dist.nll(&preds.view(), &target);
            scores.push(score);
        }

        let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
        Ok(avg_score)
    }

    pub fn hyper_opt(
        &mut self,
        features: &Array2<f64>,
        labels: &Array1<f64>,
        hp_dict: &HashMap<String, Value>,
        n_trials: u32,
        n_folds: usize,
    ) -> Result<HashMap<String, Value>> {
        hyper_opt::hyper_opt(self, features, labels, hp_dict, n_trials, n_folds, 42)
            .map(|result| result.best_params)
            .map_err(|e| GradientLSSError::HyperOptError(e.to_string()))
    }

    pub fn predict(
        &self,
        features: &ArrayView2<f64>,
        pred_type: PredType,
        n_samples: usize,
        quantiles: &[f64],
        seed: u64,
    ) -> Result<PredictionOutput> {
        let model = self
            .model
            .as_ref()
            .ok_or(GradientLSSError::ModelNotTrained)?;

        // `predict_raw` hands us an owned array — add the start values in place
        // rather than cloning a second full matrix.
        let mut predictions = model.predict_raw(features)?;
        if let Some(ref start_vals) = self.start_values {
            for mut row in predictions.rows_mut() {
                for (j, val) in row.iter_mut().enumerate() {
                    *val += start_vals[j];
                }
            }
        }

        let params = self.dist.transform_params(&predictions.view());

        match pred_type {
            PredType::Parameters => Ok(PredictionOutput::Parameters(params)),

            PredType::Samples => {
                let samples = self.dist.sample(&params.view(), n_samples, seed);
                Ok(PredictionOutput::Samples(samples))
            }

            PredType::Quantiles => {
                let samples = self.dist.sample(&params.view(), n_samples, seed);
                let n_obs = features.nrows();
                let n_quantiles = quantiles.len();
                let mut quant_result = Array2::zeros((n_obs, n_quantiles));

                for i in 0..n_obs {
                    let mut obs_samples: Vec<f64> = samples.column(i).to_vec();
                    obs_samples
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let n = obs_samples.len();
                    if n == 0 {
                        continue;
                    }
                    for (q_idx, &q) in quantiles.iter().enumerate() {
                        // Linear interpolation between order statistics, matching
                        // numpy/pandas' default `quantile` (interpolation="linear").
                        // A floor index biases discrete/tail quantiles low.
                        let pos = (n as f64 - 1.0) * q.clamp(0.0, 1.0);
                        let lo = pos.floor() as usize;
                        let hi = pos.ceil() as usize;
                        let frac = pos - lo as f64;
                        quant_result[[i, q_idx]] =
                            obs_samples[lo] + frac * (obs_samples[hi] - obs_samples[lo]);
                    }
                }
                // Python truncates quantiles to integers for discrete supports
                // (`pred_quant_df.astype(int)`).
                if self.dist.is_discrete() {
                    quant_result.mapv_inplace(|v| v.trunc());
                }
                Ok(PredictionOutput::Quantiles(quant_result))
            }

            PredType::Expectiles => {
                // For Expectile distribution, the parameters ARE the expectiles.
                // For other distributions, this returns the same as Parameters.
                Ok(PredictionOutput::Expectiles(params))
            }

            PredType::Distribution => {
                // Return full distribution info for PDF/CDF evaluation
                let dist_info = crate::backend::DistributionInfo {
                    dist_name: self.dist.name().to_string(),
                    params,
                    param_names: self.param_names(),
                    is_univariate: self.dist.is_univariate(),
                    n_targets: self.dist.n_targets(),
                };
                Ok(PredictionOutput::Distribution(dist_info))
            }
        }
    }

    pub fn start_values(&self) -> Option<&Array1<f64>> {
        self.start_values.as_ref()
    }

    /// Get feature importance scores from the trained model.
    ///
    /// # Arguments
    /// * `importance_type` - Type of importance to compute (Gain, Split, or Cover)
    /// * `feature_names` - Optional feature names for labeling
    ///
    /// # Returns
    /// Feature importance scores for each feature across distributional parameters.
    pub fn feature_importance(
        &self,
        importance_type: FeatureImportanceType,
        feature_names: Option<Vec<String>>,
    ) -> Result<FeatureImportance> {
        let model = self
            .model
            .as_ref()
            .ok_or(GradientLSSError::ModelNotTrained)?;

        model.feature_importance(importance_type, feature_names)
    }

    /// Get the number of features the model was trained on.
    pub fn num_features(&self) -> Result<usize> {
        let model = self
            .model
            .as_ref()
            .ok_or(GradientLSSError::ModelNotTrained)?;
        Ok(model.num_features())
    }

    /// Get parameter names from the distribution.
    pub fn param_names(&self) -> Vec<String> {
        self.dist
            .param_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(all(test, feature = "lightgbm"))]
mod lightgbm_tests {
    use super::*;
    use crate::backend::lightgbm_backend::LightGBMBackend;
    use crate::distributions::{
        Dirichlet, Expectile, Gaussian, LossFn, MVN, MVNLoRa, MVT, Stabilization,
    };
    use crate::utils::ResponseFn;
    use ndarray::array;
    use std::sync::Arc;

    #[test]
    fn test_univariate_training() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::new(
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        )));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![1.0, 2.0, 3.0];

        // Fix: Use fully qualified syntax <LightGBMBackend as Backend>::Dataset
        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_multivariate_mvn_creation() {
        let dist = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        let model = GradientLSS::<LightGBMBackend>::new(Arc::new(dist));

        assert_eq!(model.n_params(), 5); // 2 loc + 3 tril
        assert!(!model.distribution().is_univariate());
        assert_eq!(model.distribution().n_targets(), 2);
    }

    #[test]
    fn test_multivariate_training() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(MVN::new(
            2,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        )));

        // For multivariate with 2 targets, we need 2 * n_samples labels
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        // Labels are flattened: [y1_obs1, y2_obs1, y1_obs2, y2_obs2, ...]
        let labels = array![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];

        // Fix: Use fully qualified syntax
        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_mvt_multivariate_training() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(MVT::new(
            2,
            Stabilization::None,
            ResponseFn::Exp,
            ResponseFn::ExpDf,
            LossFn::Nll,
            false,
        )));

        // For multivariate with 2 targets, we need 2 * n_samples labels
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        // Labels are flattened: [y1_obs1, y2_obs1, y1_obs2, y2_obs2, ...]
        let labels = array![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];

        // Fix: Use fully qualified syntax
        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_dirichlet_multivariate_training() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Dirichlet::new(
            3,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        )));

        // For Dirichlet with 3 targets (compositional data), we need 3 * n_samples labels
        // that sum to 1 for each observation
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        // Labels are flattened: [p1_obs1, p2_obs1, p3_obs1, p1_obs2, p2_obs2, p3_obs2, ...]
        // where p1 + p2 + p3 = 1 for each observation
        let labels = array![0.3, 0.4, 0.3, 0.2, 0.5, 0.3, 0.1, 0.6, 0.3];

        // Fix: Use fully qualified syntax
        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_expectile_training() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Expectile::new(
            vec![0.1, 0.5, 0.9],
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        )));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![1.5, 2.5, 3.5];

        // Fix: Use fully qualified syntax
        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_mvn_lora_training() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(MVNLoRa::new(
            3,
            2,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        )));

        // For MVN_LoRa with 3 targets, we need 3 * n_samples labels
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        // Labels are flattened: [y1_obs1, y2_obs1, y3_obs1, y1_obs2, y2_obs2, y3_obs2, ...]
        let labels = array![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0];

        // Fix: Use fully qualified syntax
        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_comprehensive_prediction_validation() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::default()));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let labels = array![1.0, 2.0, 3.0, 4.0];

        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        // Test different prediction types
        let test_features = array![[2.0, 3.0], [4.0, 5.0]];

        // Test parameter prediction
        let params_result =
            model.predict(&test_features.view(), PredType::Parameters, 100, &[], 42);
        if let Ok(PredictionOutput::Parameters(params)) = params_result {
            assert_eq!(params.nrows(), 2); // 2 test samples
            assert_eq!(params.ncols(), 2); // 2 parameters (loc, scale)
            assert!(params.iter().all(|&v| v.is_finite() && !v.is_nan()));
        } else {
            panic!("Parameter prediction failed");
        }

        // Test sample prediction
        let samples_result = model.predict(&test_features.view(), PredType::Samples, 100, &[], 42);
        if let Ok(PredictionOutput::Samples(samples)) = samples_result {
            assert_eq!(samples.nrows(), 100); // 100 samples
            assert_eq!(samples.ncols(), 2); // 2 test observations
            assert!(samples.iter().all(|&v| v.is_finite() && !v.is_nan()));
        } else {
            panic!("Sample prediction failed");
        }

        // Test quantile prediction
        let quantiles = vec![0.1, 0.5, 0.9];
        let quantiles_result = model.predict(
            &test_features.view(),
            PredType::Quantiles,
            100,
            &quantiles,
            42,
        );
        if let Ok(PredictionOutput::Quantiles(quantiles_pred)) = quantiles_result {
            assert_eq!(quantiles_pred.nrows(), 2); // 2 test samples
            assert_eq!(quantiles_pred.ncols(), 3); // 3 quantiles
            assert!(quantiles_pred.iter().all(|&v| v.is_finite() && !v.is_nan()));

            // Check that quantiles are ordered correctly
            for row in quantiles_pred.rows() {
                assert!(row[0] <= row[1] && row[1] <= row[2]);
            }
        } else {
            panic!("Quantile prediction failed");
        }
    }

    #[test]
    fn test_early_stopping() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::default()));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let labels = array![1.0, 2.0, 3.0, 4.0];

        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 100,
            early_stopping_rounds: Some(5),
            verbose: false,
            collect_train_metrics: true,
            seed: 42,
        };

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_hyper_opt() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::default()));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let labels = array![1.0, 2.0, 3.0, 4.0];

        let mut hp_dict = HashMap::new();
        hp_dict.insert(
            "num_leaves".to_string(),
            serde_json::to_value(vec![2, 8]).unwrap(),
        );

        let result = model.hyper_opt(&features, &labels, &hp_dict, 2, 2);

        assert!(result.is_ok());
        let best_params = result.unwrap();
        assert!(best_params.contains_key("num_leaves"));
    }

    #[test]
    fn test_mixture_training() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(
            crate::distributions::Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, true),
        ));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let labels = array![1.0, 2.0, 3.0, 4.0];

        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_spline_flow_training() {
        let mut model =
            GradientLSS::<LightGBMBackend>::new(Arc::new(crate::distributions::SplineFlow::new(
                crate::distributions::spline_flow::TargetSupport::Real,
                2,
                1.0,
                crate::distributions::spline_flow::SplineOrder::Quadratic,
                Stabilization::None,
                LossFn::Nll,
                true,
            )));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let labels = array![1.0, 2.0, 3.0, 4.0];

        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }
}

#[cfg(all(test, feature = "xgboost"))]
mod xgboost_tests {
    use super::*;
    use crate::backend::BackendDataset;
    use crate::distributions::{Gaussian, LossFn, MVN, Stabilization};
    use crate::utils::ResponseFn;
    use ndarray::array;
    use std::sync::Arc;
    #[test]
    fn test_xgboost_univariate_training() {
        let mut model =
            GradientLSS::<crate::backend::XGBoostBackend>::new(Arc::new(Gaussian::default()));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![1.0, 2.0, 3.0];

        let mut train_data =
            <crate::backend::XGBoostBackend as crate::backend::Backend>::Dataset::from_data(
                features.view(),
                labels.view(),
            )
            .unwrap();

        let params = crate::backend::XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_xgboost_multivariate_training() {
        let mut model = GradientLSS::<crate::backend::XGBoostBackend>::new(Arc::new(MVN::new(
            2,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        )));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];

        let mut train_data =
            <crate::backend::XGBoostBackend as crate::backend::Backend>::Dataset::from_data(
                features.view(),
                labels.view(),
            )
            .unwrap();

        let params = crate::backend::XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.n_params(), 5); // 2 loc + 3 tril
    }

    #[test]
    fn test_xgboost_comprehensive_prediction() {
        let mut model =
            GradientLSS::<crate::backend::XGBoostBackend>::new(Arc::new(Gaussian::default()));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let labels = array![1.0, 2.0, 3.0, 4.0];

        let mut train_data =
            <crate::backend::XGBoostBackend as crate::backend::Backend>::Dataset::from_data(
                features.view(),
                labels.view(),
            )
            .unwrap();

        let params = crate::backend::XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        // Test predictions
        let test_features = array![[2.0, 3.0], [4.0, 5.0]];

        // Test parameter prediction
        let params_result =
            model.predict(&test_features.view(), PredType::Parameters, 100, &[], 42);
        assert!(matches!(params_result, Ok(PredictionOutput::Parameters(_))));

        // Test sample prediction
        let samples_result = model.predict(&test_features.view(), PredType::Samples, 100, &[], 42);
        assert!(matches!(samples_result, Ok(PredictionOutput::Samples(_))));

        // Test quantile prediction
        let quantiles = vec![0.1, 0.5, 0.9];
        let quantiles_result = model.predict(
            &test_features.view(),
            PredType::Quantiles,
            100,
            &quantiles,
            42,
        );
        assert!(matches!(
            quantiles_result,
            Ok(PredictionOutput::Quantiles(_))
        ));
    }
}
