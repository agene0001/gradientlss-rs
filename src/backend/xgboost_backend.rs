//! XGBoost backend implementation.
//!
//! This backend trains separate boosters for each distribution parameter,
//! which is the standard approach for distributional regression with XGBoost.

use super::traits::{
    Backend, BackendDataset, BackendModel, BackendParams, CallbackAction, FeatureImportance,
    FeatureImportanceType, ParamValue, TrainConfig, TrainingCallback, TrainingResult,
};
use crate::distributions::GradientsAndHessians;
use crate::error::{GradientLSSError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Write};
use tempfile::NamedTempFile;

use xgb::parameters::BoosterParameters;
use xgb::{Booster, DMatrix};

// Thread-local storage to pass gradients/hessians to the strict function pointer callback
thread_local! {
    static OBJECTIVE_DATA: RefCell<Option<(Vec<f32>, Vec<f32>)>> = RefCell::new(None);
}

/// Trampoline function that matches the signature required by xgboost::update_custom
fn objective_trampoline(_preds: &[f32], _dtrain: &DMatrix) -> (Vec<f32>, Vec<f32>) {
    OBJECTIVE_DATA.with(|data| {
        data.borrow()
            .as_ref()
            .expect("Objective data was not set before update_custom call")
            .clone()
    })
}

/// XGBoost backend for GradientLSS.
#[derive(Debug, Clone)]
pub struct XGBoostBackend;

/// XGBoost-specific parameters.
#[derive(Debug, Clone)]
pub struct XGBoostParams {
    inner: HashMap<String, String>,
    n_dist_params: usize,
}

impl Default for XGBoostParams {
    fn default() -> Self {
        let mut inner = HashMap::new();
        inner.insert("booster".to_string(), "gbtree".to_string());
        inner.insert("eta".to_string(), "0.1".to_string());
        inner.insert("max_depth".to_string(), "6".to_string());
        inner.insert("base_score".to_string(), "0.0".to_string());
        inner.insert(
            "disable_default_eval_metric".to_string(),
            "true".to_string(),
        );
        Self {
            inner,
            n_dist_params: 1,
        }
    }
}

impl BackendParams for XGBoostParams {
    fn set(&mut self, key: &str, value: ParamValue) {
        let str_value = match value {
            ParamValue::Int(v) => v.to_string(),
            ParamValue::Float(v) => v.to_string(),
            ParamValue::String(v) => v,
            ParamValue::Bool(v) => v.to_string(),
        };
        self.inner.insert(key.to_string(), str_value);
    }

    fn get(&self, _key: &str) -> Option<&ParamValue> {
        None
    }

    fn to_map(&self) -> HashMap<String, ParamValue> {
        self.inner
            .iter()
            .map(|(k, v)| (k.clone(), ParamValue::String(v.clone())))
            .collect()
    }
}

impl XGBoostParams {
    /// Get the inner HashMap for xgboost-rs.
    pub fn to_xgb_params(&self) -> HashMap<String, String> {
        self.inner.clone()
    }

    /// Set the number of distribution parameters.
    pub fn set_n_dist_params(&mut self, n: usize) {
        self.n_dist_params = n;
    }

    /// Get the number of distribution parameters.
    pub fn n_dist_params(&self) -> usize {
        self.n_dist_params
    }
}

/// XGBoost dataset wrapper around DMatrix.
pub struct XGBoostDataset {
    dmatrix: DMatrix,
    n_rows: usize,
    n_cols: usize,
    features: Vec<f32>,
    /// Full labels (may be longer than n_rows for multivariate targets)
    full_labels: Vec<f64>,
}

impl std::fmt::Debug for XGBoostDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XGBoostDataset")
            .field("n_rows", &self.n_rows)
            .field("n_cols", &self.n_cols)
            .finish()
    }
}

impl BackendDataset for XGBoostDataset {
    fn from_data(features: ArrayView2<f64>, labels: ArrayView1<f64>) -> Result<Self> {
        let n_rows = features.nrows();
        let n_cols = features.ncols();

        // Convert to f32 for xgboost
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();

        // For XGBoost with separate boosters per distribution parameter, we only use
        // single-target labels. If labels has more elements than n_rows (multivariate case),
        // only use the first n_rows labels. This prevents XGBoost from thinking it's a
        // multi-output problem. The actual gradients are passed via update_custom.
        let labels_f32: Vec<f32> = labels.iter().take(n_rows).map(|&x| x as f32).collect();

        // Create DMatrix from dense array (row-major)
        let mut dmatrix = DMatrix::from_dense(&features_f32, n_rows).map_err(|e| {
            GradientLSSError::BackendError(format!("Failed to create DMatrix: {}", e))
        })?;

        // Set labels (only n_rows labels to keep XGBoost in single-output mode)
        dmatrix
            .set_labels(&labels_f32)
            .map_err(|e| GradientLSSError::BackendError(format!("Failed to set labels: {}", e)))?;

        Ok(Self {
            dmatrix,
            n_rows,
            n_cols,
            features: features_f32,
            full_labels: labels.to_vec(),
        })
    }

    fn set_init_score(&mut self, _init_score: &Array1<f64>) -> Result<()> {
        // Note: We intentionally don't use base_margin here because:
        // 1. The XGBoost backend trains separate boosters for each distribution parameter
        // 2. Each booster expects base_margin of shape (n_samples, 1), not (n_samples * n_params)
        // 3. The training loop already handles start values by adding them to predictions
        // 4. Using base_margin with the wrong shape causes errors in newer XGBoost versions
        Ok(())
    }

    fn num_rows(&self) -> usize {
        self.n_rows
    }

    fn get_labels(&self) -> Result<Array1<f64>> {
        // Return the full labels (which may be longer than n_rows for multivariate targets)
        // rather than the truncated labels stored in the DMatrix
        Ok(Array1::from(self.full_labels.clone()))
    }
}

impl XGBoostDataset {
    /// Get a reference to the underlying DMatrix.
    pub fn dmatrix(&self) -> &DMatrix {
        &self.dmatrix
    }

    /// Get a mutable reference to the underlying DMatrix.
    pub fn dmatrix_mut(&mut self) -> &mut DMatrix {
        &mut self.dmatrix
    }

    /// Get the stored features for creating new DMatrix instances.
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    /// Get the number of columns (features).
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
}

/// XGBoost model wrapper - contains one booster per distribution parameter.
pub struct XGBoostModel {
    /// One booster for each distribution parameter
    boosters: Vec<Booster>,
    n_params: usize,
}

impl std::fmt::Debug for XGBoostModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XGBoostModel")
            .field("n_params", &self.n_params)
            .field("n_boosters", &self.boosters.len())
            .finish()
    }
}

impl BackendModel for XGBoostModel {
    type Dataset = XGBoostDataset;
    type Params = XGBoostParams;

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
        M: Fn(&Array2<f64>, &Array1<f64>) -> f64,
    {
        // Use the callback version with no callbacks
        let (model, _) =
            Self::train_with_objective_and_callbacks::<F, M, super::traits::HistoryCallback>(
                params,
                train_data,
                valid_data,
                config,
                objective_fn,
                metric_fn,
                start_values,
                None,
            )?;
        Ok(model)
    }

    fn train_with_objective_and_callbacks<F, M, C>(
        params: &Self::Params,
        train_data: &mut Self::Dataset,
        valid_data: Option<&mut Self::Dataset>,
        config: &TrainConfig,
        objective_fn: F,
        metric_fn: M,
        start_values: Option<&Array1<f64>>,
        mut callbacks: Option<&mut C>,
    ) -> Result<(Self, TrainingResult)>
    where
        F: Fn(&Array2<f64>, &Array1<f64>, Option<&Array1<f64>>) -> Result<GradientsAndHessians>,
        M: Fn(&Array2<f64>, &Array1<f64>) -> f64,
        C: TrainingCallback,
    {
        let n_params = params.n_dist_params();
        let n_samples = train_data.num_rows();

        // Create one booster per distribution parameter
        let mut boosters: Vec<Booster> = Vec::with_capacity(n_params);
        for _ in 0..n_params {
            let booster = Booster::new_with_cached_dmats(
                &BoosterParameters::default(),
                &[&train_data.dmatrix],
            )
            .map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to create booster: {}", e))
            })?;
            boosters.push(booster);
        }

        let labels = train_data.get_labels()?;

        // For validation-based early stopping
        let (valid_features, valid_labels) = if let Some(ref vd) = valid_data {
            let vl = vd.get_labels()?;
            Some((vd.features().to_vec(), vd.n_rows, vd.n_cols, vl))
        } else {
            None
        }
        .map_or((None, None), |(f, r, c, l)| (Some((f, r, c)), Some(l)));

        let mut best_loss = f64::INFINITY;
        let mut best_iteration = 0usize;
        let mut rounds_without_improvement = 0;
        let mut stopped_early = false;

        // Training history
        let mut train_history = Vec::with_capacity(config.num_boost_round);
        let mut valid_history = Vec::with_capacity(config.num_boost_round);

        // Initialize predictions with start values or zeros
        let mut predictions = Array2::zeros((n_samples, n_params));
        if let Some(sv) = start_values {
            for i in 0..n_samples {
                for j in 0..n_params {
                    predictions[[i, j]] = sv[j];
                }
            }
        }

        // Notify callbacks of training start
        if let Some(ref mut cb) = callbacks {
            cb.on_training_start(config.num_boost_round);
        }

        let mut final_round = 0;

        // Training loop
        for round in 0..config.num_boost_round {
            final_round = round;

            // Get current predictions from all boosters
            for (param_idx, booster) in boosters.iter().enumerate() {
                let preds = booster.predict(&train_data.dmatrix).map_err(|e| {
                    GradientLSSError::BackendError(format!("Prediction failed: {}", e))
                })?;

                // Only use first n_samples predictions (XGBoost may return more based on labels)
                for i in 0..n_samples {
                    predictions[[i, param_idx]] = preds[i] as f64;
                }
            }

            // Add start values to predictions if provided
            if let Some(sv) = start_values {
                for i in 0..n_samples {
                    for j in 0..n_params {
                        predictions[[i, j]] += sv[j];
                    }
                }
            }

            // Compute gradients and hessians for all parameters
            let gh = objective_fn(&predictions, &labels, None)?;

            // Update each booster with its corresponding gradients
            for (param_idx, booster) in boosters.iter_mut().enumerate() {
                let mut grad_f32 = Vec::with_capacity(n_samples);
                let mut hess_f32 = Vec::with_capacity(n_samples);

                for i in 0..n_samples {
                    grad_f32.push(gh.gradients[[i, param_idx]] as f32);
                    hess_f32.push(gh.hessians[[i, param_idx]] as f32);
                }

                // Set gradient data for this parameter's booster
                OBJECTIVE_DATA.with(|data| {
                    *data.borrow_mut() = Some((grad_f32, hess_f32));
                });

                // Update the booster
                booster
                    .update_custom(
                        &train_data.dmatrix,
                        round as i32, // <--- Add this argument
                        objective_trampoline
                    )
                    .map_err(|e| {
                        GradientLSSError::BackendError(format!(
                            "Update failed for param {}: {}",
                            param_idx, e
                        ))
                    })?;
            }

            // Compute training loss
            let train_loss = metric_fn(&predictions, &labels);
            train_history.push(train_loss);

            // Compute validation loss if validation data is available
            let valid_loss = if let (Some((vf, vr, _vc)), Some(vl)) =
                (&valid_features, &valid_labels)
            {
                // Compute predictions on validation set
                let valid_dmat = DMatrix::from_dense(vf, *vr).map_err(|e| {
                    GradientLSSError::BackendError(format!("Failed to create valid DMatrix: {}", e))
                })?;

                let mut valid_preds = Array2::zeros((*vr, n_params));
                for (param_idx, booster) in boosters.iter().enumerate() {
                    let preds = booster.predict(&valid_dmat).map_err(|e| {
                        GradientLSSError::BackendError(format!("Valid prediction failed: {}", e))
                    })?;

                    // Only use first vr predictions
                    for i in 0..*vr {
                        valid_preds[[i, param_idx]] = preds[i] as f64;
                    }
                }

                // Add start values
                if let Some(sv) = start_values {
                    for i in 0..*vr {
                        for j in 0..n_params {
                            valid_preds[[i, j]] += sv[j];
                        }
                    }
                }

                let vl_loss = metric_fn(&valid_preds, vl);
                valid_history.push(vl_loss);
                Some(vl_loss)
            } else {
                None
            };

            // Determine the evaluation loss for early stopping
            let eval_loss = valid_loss.unwrap_or(train_loss);

            if config.verbose && round % 10 == 0 {
                match valid_loss {
                    Some(vl) => println!(
                        "[{}] train_loss: {:.6}, valid_loss: {:.6}",
                        round, train_loss, vl
                    ),
                    None => println!("[{}] train_loss: {:.6}", round, train_loss),
                }
            }

            // Invoke callbacks
            if let Some(ref mut cb) = callbacks {
                if cb.on_iteration_end(round, train_loss, valid_loss) == CallbackAction::Stop {
                    stopped_early = true;
                    break;
                }
            }

            // Built-in early stopping (if no callbacks or callbacks didn't stop)
            if let Some(early_stopping) = config.early_stopping_rounds {
                if eval_loss < best_loss {
                    best_loss = eval_loss;
                    best_iteration = round;
                    rounds_without_improvement = 0;
                } else {
                    rounds_without_improvement += 1;
                    if rounds_without_improvement >= early_stopping {
                        if config.verbose {
                            println!("Early stopping at round {}", round);
                        }
                        stopped_early = true;
                        break;
                    }
                }
            } else if eval_loss < best_loss {
                best_loss = eval_loss;
                best_iteration = round;
            }
        }

        // Notify callbacks of training end
        if let Some(ref mut cb) = callbacks {
            cb.on_training_end(final_round + 1, stopped_early);
        }

        let result = TrainingResult {
            n_iterations: final_round + 1,
            best_iteration: Some(best_iteration),
            best_score: Some(best_loss),
            train_history,
            valid_history,
            stopped_early,
        };

        Ok((Self { boosters, n_params }, result))
    }

    fn predict_raw(&self, data: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.nrows();

        // Create DMatrix from features
        let features_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let dmatrix = DMatrix::from_dense(&features_f32, n_samples).map_err(|e| {
            GradientLSSError::BackendError(format!("Failed to create DMatrix: {}", e))
        })?;

        // Get predictions from each booster
        let mut result = Array2::zeros((n_samples, self.n_params));

        for (param_idx, booster) in self.boosters.iter().enumerate() {
            let preds = booster
                .predict(&dmatrix)
                .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;

            // Only use first n_samples predictions
            for i in 0..n_samples {
                result[[i, param_idx]] = preds[i] as f64;
            }
        }

        Ok(result)
    }

    fn save_to_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write number of boosters
        writer
            .write_all(&(self.boosters.len() as u64).to_le_bytes())
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        // Write each booster using temp file workaround (xgboost crate lacks save_buffer)
        for booster in &self.boosters {
            let temp_file =
                NamedTempFile::new().map_err(|e| GradientLSSError::IoError(e.to_string()))?;
            let temp_path = temp_file.path();

            booster.save(temp_path).map_err(|e| {
                GradientLSSError::BackendError(format!(
                    "Failed to save booster to temp file: {}",
                    e
                ))
            })?;

            let model_bytes =
                std::fs::read(temp_path).map_err(|e| GradientLSSError::IoError(e.to_string()))?;

            writer
                .write_all(&(model_bytes.len() as u64).to_le_bytes())
                .map_err(|e| GradientLSSError::IoError(e.to_string()))?;
            writer
                .write_all(&model_bytes)
                .map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    fn load_from_reader<R: Read>(reader: &mut R) -> Result<Self> {
        // Read number of boosters
        let mut n_boosters_bytes = [0u8; 8];
        reader
            .read_exact(&mut n_boosters_bytes)
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        let n_boosters = u64::from_le_bytes(n_boosters_bytes) as usize;

        // Read each booster
        let mut boosters = Vec::with_capacity(n_boosters);
        for _ in 0..n_boosters {
            let mut len_bytes = [0u8; 8];
            reader
                .read_exact(&mut len_bytes)
                .map_err(|e| GradientLSSError::IoError(e.to_string()))?;
            let len = u64::from_le_bytes(len_bytes) as usize;

            let mut model_bytes = vec![0u8; len];
            reader
                .read_exact(&mut model_bytes)
                .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

            let booster = Booster::load_buffer(&model_bytes).map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to load booster from buffer: {}", e))
            })?;
            boosters.push(booster);
        }

        Ok(Self {
            boosters,
            n_params: n_boosters,
        })
    }

    fn feature_importance(
        &self,
        importance_type: FeatureImportanceType,
        feature_names: Option<Vec<String>>,
    ) -> Result<FeatureImportance> {
        if self.boosters.is_empty() {
            return Err(GradientLSSError::ModelNotTrained);
        }

        // XGBoost Rust crate doesn't expose feature importance directly.
        // We parse the model dump to extract feature usage statistics.
        // This is a simplified implementation that counts feature occurrences.

        let n_params = self.n_params;

        // Parse feature importance from model dumps
        let mut all_importance: Vec<HashMap<String, f64>> = Vec::with_capacity(n_params);

        for booster in &self.boosters {
            let mut importance: HashMap<String, f64> = HashMap::new();

            // Dump the model to text and parse feature usage
            if let Ok(model_dump) = booster.dump_model(true, None) {
                for line in model_dump.lines() {
                    // Look for lines like "[f0<0.5]" or "f0<0.5"
                    if let Some(start) = line.find("[f") {
                        if let Some(end) =
                            line[start..].find('<').or_else(|| line[start..].find(']'))
                        {
                            let feature_name = &line[start + 1..start + end];
                            *importance.entry(feature_name.to_string()).or_insert(0.0) += 1.0;
                        }
                    }
                }
            }

            all_importance.push(importance);
        }

        // Find all unique feature names across boosters
        let mut all_features: Vec<String> = all_importance
            .iter()
            .flat_map(|m| m.keys().cloned())
            .collect();
        all_features.sort();
        all_features.dedup();

        // If no features found, create a placeholder
        if all_features.is_empty() {
            return Ok(FeatureImportance {
                feature_indices: vec![],
                feature_names,
                scores: Array2::zeros((0, n_params)),
                importance_type,
            });
        }

        let n_features = all_features.len();

        // Build scores matrix (n_features x n_params)
        let mut scores_vec = Vec::with_capacity(n_features * n_params);
        for feat in &all_features {
            for imp_map in &all_importance {
                let score = imp_map.get(feat).copied().unwrap_or(0.0);
                scores_vec.push(score);
            }
        }

        let scores = Array2::from_shape_vec((n_features, n_params), scores_vec).map_err(|e| {
            GradientLSSError::ShapeMismatch {
                expected_shape: format!("({}, {})", n_features, n_params),
                actual_shape: e.to_string(),
            }
        })?;

        // Extract feature indices from names (XGBoost names features as "f0", "f1", etc.)
        let feature_indices: Vec<usize> = all_features
            .iter()
            .map(|f| {
                f.strip_prefix('f')
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
            })
            .collect();

        Ok(FeatureImportance {
            feature_indices,
            feature_names,
            scores,
            importance_type,
        })
    }

    fn num_features(&self) -> usize {
        // XGBoost doesn't directly expose num_features, return 0 if unknown
        // Users should track this themselves or use feature_importance which discovers features
        0
    }

    fn num_params(&self) -> usize {
        self.n_params
    }
}

impl Backend for XGBoostBackend {
    type Dataset = XGBoostDataset;
    type Model = XGBoostModel;
    type Params = XGBoostParams;

    fn name() -> &'static str {
        "XGBoost"
    }

    fn create_params(n_dist_params: usize) -> Self::Params {
        let mut params = XGBoostParams::default();
        params.set_n_dist_params(n_dist_params);
        params
    }

    fn reshape_gradients(
        gradients: &Array2<f64>,
        hessians: &Array2<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        // XGBoost expects gradients in C order (row-major)
        let grad_flat = Array1::from_iter(gradients.iter().copied());
        let hess_flat = Array1::from_iter(hessians.iter().copied());
        (grad_flat, hess_flat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xgboost_params_default() {
        let params = XGBoostParams::default();
        assert!(params.inner.contains_key("booster"));
    }

    #[test]
    fn test_reshape_gradients() {
        let gradients = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let hessians = Array2::ones((3, 2));

        let (grad_flat, _) = XGBoostBackend::reshape_gradients(&gradients, &hessians);

        // C order: [1, 2, 3, 4, 5, 6]
        assert_eq!(grad_flat.len(), 6);
        assert_eq!(grad_flat[0], 1.0);
        assert_eq!(grad_flat[1], 2.0);
    }

    #[test]
    fn test_xgboost_params_n_dist_params() {
        let mut params = XGBoostParams::default();
        params.set_n_dist_params(3);
        assert_eq!(params.n_dist_params(), 3);
    }
}
