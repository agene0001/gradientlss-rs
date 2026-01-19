//! LightGBM backend implementation using the `lgbm` crate.
//!
//! This backend supports custom objective functions via `update_one_iter_custom`,
//! enabling full distributional regression with custom gradients and hessians.
//! LightGBM natively supports multi-output, so we use a single booster with num_class.

use super::traits::{
    Backend, BackendDataset, BackendModel, BackendParams, CallbackAction, FeatureImportance,
    FeatureImportanceType, ParamValue, TrainConfig, TrainingCallback, TrainingResult,
};
use crate::distributions::GradientsAndHessians;
use crate::error::{GradientLSSError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;

use lgbm::{Booster, Dataset, Field, Mat, MatBuf, Parameters, PredictType, Prediction};

/// LightGBM backend for GradientLSS.
#[derive(Debug, Clone)]
pub struct LightGBMBackend;

/// LightGBM-specific parameters.
#[derive(Debug, Clone)]
pub struct LightGBMParams {
    inner: Parameters,
    n_dist_params: usize,
}

impl Default for LightGBMParams {
    fn default() -> Self {
        let mut inner = Parameters::new();
        inner.push("boosting", "gbdt");
        inner.push("learning_rate", "0.1");
        inner.push("num_leaves", "31");
        inner.push("verbose", "-1");
        // Use "none" objective since we provide custom gradients
        inner.push("objective", "none");
        Self {
            inner,
            n_dist_params: 1,
        }
    }
}

impl BackendParams for LightGBMParams {
    fn set(&mut self, key: &str, value: ParamValue) {
        let str_value = match value {
            ParamValue::Int(v) => v.to_string(),
            ParamValue::Float(v) => v.to_string(),
            ParamValue::String(v) => v,
            ParamValue::Bool(v) => if v { "true" } else { "false" }.to_string(),
        };
        self.inner.push(key, str_value);
    }

    fn get(&self, _key: &str) -> Option<&ParamValue> {
        // lgbm::Parameters doesn't support direct lookup
        None
    }

    fn to_map(&self) -> HashMap<String, ParamValue> {
        // Simplified - would need to parse the parameters string
        HashMap::new()
    }
}

impl LightGBMParams {
    /// Get a reference to the inner Parameters.
    pub fn inner(&self) -> &Parameters {
        &self.inner
    }

    /// Set the number of distribution parameters.
    pub fn set_n_dist_params(&mut self, n: usize) {
        self.n_dist_params = n;
        // Set num_class for multi-output
        self.inner.push("num_class", n.to_string());
    }

    /// Get the number of distribution parameters.
    pub fn n_dist_params(&self) -> usize {
        self.n_dist_params
    }
}

/// LightGBM dataset wrapper around lgbm::Dataset.
/// Stores features for mid-training predictions.
pub struct LightGBMDataset {
    dataset: Arc<Dataset>,
    n_rows: usize,
    n_cols: usize,
    labels: Array1<f64>,
    /// Store features for prediction during training
    features: Vec<f32>,
}

impl std::fmt::Debug for LightGBMDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LightGBMDataset")
            .field("n_rows", &self.n_rows)
            .field("n_cols", &self.n_cols)
            .finish()
    }
}

impl BackendDataset for LightGBMDataset {
    fn from_data(features: ArrayView2<f64>, labels: ArrayView1<f64>) -> Result<Self> {
        let n_rows = features.nrows();
        let n_cols = features.ncols();

        // Convert features to f32 for lgbm
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let labels_f32: Vec<f32> = labels.iter().map(|&x| x as f32).collect();

        // Create MatBuf (row-major)
        let mat = Mat::from_slice(&features_f32, n_rows, n_cols, lgbm::mat::RowMajor);

        // Create dataset with parameters
        let params = Parameters::new();
        let mut dataset = Dataset::from_mat(&mat, None, &params).map_err(|e| {
            GradientLSSError::BackendError(format!("Failed to create Dataset: {}", e))
        })?;

        // For multivariate distributions, we need to handle labels differently
        // LightGBM expects one label per sample, but for multivariate distributions
        // we have n_targets labels per sample. We'll store the full labels but only
        // provide dummy labels to LightGBM (since custom objectives don't use them)
        let dummy_labels = if labels.len() == n_rows {
            // Univariate case: use labels as-is
            labels_f32
        } else {
            // Multivariate case: create dummy labels (all zeros)
            vec![0.0; n_rows]
        };

        // Set dummy labels (actual labels are stored separately for the objective function)
        dataset
            .set_field(Field::<f32>::LABEL, &dummy_labels)
            .map_err(|e| GradientLSSError::BackendError(format!("Failed to set labels: {}", e)))?;

        Ok(Self {
            dataset: Arc::new(dataset),
            n_rows,
            n_cols,
            labels: labels.to_owned(),
            features: features_f32,
        })
    }

    fn set_init_score(&mut self, init_score: &Array1<f64>) -> Result<()> {
        // lgbm uses f64 for init_score
        let _init_vec: Vec<f64> = init_score.to_vec();

        // We need mutable access to the dataset
        // Since Arc doesn't allow mutation, we need to work around this
        // For now, we'll store the init_score and apply it when creating the booster
        // This is a limitation - ideally we'd set it on the dataset directly

        // Note: The lgbm crate requires Arc<Dataset> for Booster::new,
        // which makes setting init_score after creation tricky.
        // We'll handle this in the training function instead.

        Ok(())
    }

    fn num_rows(&self) -> usize {
        self.n_rows
    }

    fn get_labels(&self) -> Result<Array1<f64>> {
        Ok(self.labels.clone())
    }
}

impl LightGBMDataset {
    /// Get a reference to the underlying Dataset wrapped in Arc.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// Get the number of columns (features).
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Get the stored features.
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    /// Create a Mat from stored features for prediction.
    pub fn to_mat(&self) -> MatBuf<f32, lgbm::mat::RowMajor> {
        MatBuf::from_vec(
            self.features.clone(),
            self.n_rows,
            self.n_cols,
            lgbm::mat::RowMajor,
        )
    }
}

/// LightGBM model wrapper around lgbm::Booster.
pub struct LightGBMModel {
    booster: Booster,
    n_params: usize,
    n_features: usize,
}

impl std::fmt::Debug for LightGBMModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LightGBMModel")
            .field("n_params", &self.n_params)
            .field("n_features", &self.n_features)
            .finish()
    }
}

impl BackendModel for LightGBMModel {
    type Dataset = LightGBMDataset;
    type Params = LightGBMParams;

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
        let n_features = train_data.n_cols();
        let labels = train_data.get_labels()?;

        // Create booster
        let mut booster =
            Booster::new(train_data.dataset.clone(), params.inner()).map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to create Booster: {}", e))
            })?;

        // Initialize predictions
        let mut predictions = Array2::zeros((n_samples, n_params));
        if let Some(sv) = start_values {
            for i in 0..n_samples {
                for j in 0..n_params {
                    predictions[[i, j]] = sv[j];
                }
            }
        }

        // Prepare validation data if available
        let (valid_mat, valid_labels, valid_n_samples) = if let Some(ref vd) = valid_data {
            let vl = vd.get_labels()?;
            let vm = vd.to_mat();
            Some((vm, vl, vd.num_rows()))
        } else {
            None
        }
        .map_or((None, None, 0), |(m, l, n)| (Some(m), Some(l), n));

        let mut best_loss = f64::INFINITY;
        let mut best_iteration = 0usize;
        let mut rounds_without_improvement = 0;
        let mut stopped_early = false;

        // Training history
        let mut train_history = Vec::with_capacity(config.num_boost_round);
        let mut valid_history = Vec::with_capacity(config.num_boost_round);

        // Create Mat from training data for predictions
        let train_mat = train_data.to_mat();
        let pred_params = Parameters::new();

        // Notify callbacks of training start
        if let Some(ref mut cb) = callbacks {
            cb.on_training_start(config.num_boost_round);
        }

        let mut final_round = 0;

        for round in 0..config.num_boost_round {
            final_round = round;

            // Compute gradients and hessians using our distribution
            let gh = objective_fn(&predictions, &labels, None)?;

            // Flatten gradients and hessians in Fortran order (column-major)
            // lgbm expects: [grad_param0_sample0, grad_param0_sample1, ..., grad_param1_sample0, ...]
            let (grad_flat, hess_flat) =
                LightGBMBackend::reshape_gradients(&gh.gradients, &gh.hessians);

            // Convert to f32 for lgbm
            let grad_f32: Vec<f32> = grad_flat.iter().map(|&x| x as f32).collect();
            let hess_f32: Vec<f32> = hess_flat.iter().map(|&x| x as f32).collect();

            // Update model with custom gradients
            let is_finished = booster
                .update_one_iter_custom(&grad_f32, &hess_f32)
                .map_err(|e| GradientLSSError::BackendError(format!("Update failed: {}", e)))?;

            if is_finished {
                if config.verbose {
                    println!("Training finished early at round {}", round);
                }
                stopped_early = true;
                break;
            }

            // Get predictions for next iteration using stored features
            let prediction = booster
                .predict_for_mat(&train_mat, PredictType::RawScore, 0, None, &pred_params)
                .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;

            // Update predictions array
            predictions = prediction_to_array2(&prediction, n_samples, n_params)?;

            // Add start values if provided
            if let Some(sv) = start_values {
                for i in 0..n_samples {
                    for j in 0..n_params {
                        predictions[[i, j]] += sv[j];
                    }
                }
            }

            // Compute training loss
            let train_loss = metric_fn(&predictions, &labels);
            train_history.push(train_loss);

            // Compute validation loss if available
            let valid_loss = if let (Some(vm), Some(vl)) = (&valid_mat, &valid_labels) {
                let valid_pred = booster
                    .predict_for_mat(vm, PredictType::RawScore, 0, None, &pred_params)
                    .map_err(|e| {
                        GradientLSSError::BackendError(format!("Valid prediction failed: {}", e))
                    })?;

                let mut valid_preds = prediction_to_array2(&valid_pred, valid_n_samples, n_params)?;

                // Add start values
                if let Some(sv) = start_values {
                    for i in 0..valid_n_samples {
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

            // Built-in early stopping
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

        Ok((
            Self {
                booster,
                n_params,
                n_features,
            },
            result,
        ))
    }

    fn predict_raw(&self, data: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.nrows();

        // Convert to f32 MatBuf
        let features_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let n_cols = data.ncols();
        let mat = MatBuf::from_vec(features_f32, n_samples, n_cols, lgbm::mat::RowMajor);

        // Predict raw scores
        let pred_params = Parameters::new();
        let prediction = self
            .booster
            .predict_for_mat(&mat, PredictType::RawScore, 0, None, &pred_params)
            .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;

        prediction_to_array2(&prediction, n_samples, self.n_params)
    }

    fn save_to_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|e| GradientLSSError::IoError(format!("Failed to create temp dir: {}", e)))?;
        let temp_path = temp_dir.path().join("model.txt");

        self.booster
            .save_model(0, None, lgbm::FeatureImportanceType::Gain, &temp_path)
            .map_err(|e| GradientLSSError::BackendError(format!("Failed to save model: {}", e)))?;

        let model_bytes = std::fs::read(&temp_path)
            .map_err(|e| GradientLSSError::IoError(format!("Failed to read temp model: {}", e)))?;

        writer
            .write_all(&model_bytes)
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        Ok(())
    }

    fn load_from_reader<R: Read>(reader: &mut R) -> Result<Self> {
        let mut model_bytes = Vec::new();
        reader
            .read_to_end(&mut model_bytes)
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        let temp_dir = tempfile::tempdir()
            .map_err(|e| GradientLSSError::IoError(format!("Failed to create temp dir: {}", e)))?;
        let temp_path = temp_dir.path().join("model.txt");

        std::fs::write(&temp_path, &model_bytes)
            .map_err(|e| GradientLSSError::IoError(format!("Failed to write temp model: {}", e)))?;

        let (booster, _) = Booster::from_file(&temp_path)
            .map_err(|e| GradientLSSError::BackendError(format!("Failed to load model: {}", e)))?;

        let n_params = booster.get_num_classes().unwrap_or(1);
        let n_features = booster.get_num_feature().unwrap_or(0);

        Ok(Self {
            booster,
            n_params,
            n_features,
        })
    }

    fn feature_importance(
        &self,
        importance_type: FeatureImportanceType,
        feature_names: Option<Vec<String>>,
    ) -> Result<FeatureImportance> {
        let lgbm_importance_type = match importance_type {
            FeatureImportanceType::Gain => lgbm::FeatureImportanceType::Gain,
            FeatureImportanceType::Split => lgbm::FeatureImportanceType::Split,
            FeatureImportanceType::Cover => {
                // LightGBM doesn't have Cover, use Gain as fallback
                lgbm::FeatureImportanceType::Gain
            }
        };

        let importance = self
            .booster
            .feature_importance(None, lgbm_importance_type)
            .map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to get feature importance: {}", e))
            })?;

        let n_features = self.n_features;

        // LightGBM returns importance per feature (aggregated across classes)
        // Shape it as (n_features, 1) for consistency
        let scores =
            Array2::from_shape_vec((n_features, 1), importance.iter().map(|&x| x).collect())
                .map_err(|e| GradientLSSError::ShapeMismatch {
                    expected_shape: format!("({}, 1)", n_features),
                    actual_shape: e.to_string(),
                })?;

        let feature_indices: Vec<usize> = (0..n_features).collect();

        Ok(FeatureImportance {
            feature_indices,
            feature_names,
            scores,
            importance_type: if importance_type == FeatureImportanceType::Cover {
                // We used Gain as fallback
                FeatureImportanceType::Gain
            } else {
                importance_type
            },
        })
    }

    fn num_features(&self) -> usize {
        self.n_features
    }

    fn num_params(&self) -> usize {
        self.n_params
    }
}

impl Backend for LightGBMBackend {
    type Dataset = LightGBMDataset;
    type Model = LightGBMModel;
    type Params = LightGBMParams;

    fn name() -> &'static str {
        "LightGBM"
    }

    fn create_params(n_dist_params: usize) -> Self::Params {
        let mut params = LightGBMParams::default();
        params.set_n_dist_params(n_dist_params);
        params
    }

    fn reshape_gradients(
        gradients: &Array2<f64>,
        hessians: &Array2<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        // LightGBM expects gradients in Fortran order (column-major)
        // For multi-class: [class0_sample0, class0_sample1, ..., class1_sample0, ...]
        let (n_samples, n_params) = gradients.dim();

        let mut grad_flat = Array1::zeros(n_samples * n_params);
        let mut hess_flat = Array1::zeros(n_samples * n_params);

        for j in 0..n_params {
            for i in 0..n_samples {
                let idx = j * n_samples + i;
                grad_flat[idx] = gradients[[i, j]];
                hess_flat[idx] = hessians[[i, j]];
            }
        }

        (grad_flat, hess_flat)
    }
}

/// Convert lgbm::Prediction to Array2.
fn prediction_to_array2(
    prediction: &Prediction,
    n_samples: usize,
    n_params: usize,
) -> Result<Array2<f64>> {
    let values: Vec<f64> = prediction.values().iter().map(|&x| x as f64).collect();

    // lgbm returns predictions in row-major order for multi-class
    if values.len() != n_samples * n_params {
        return Err(GradientLSSError::ShapeMismatch {
            expected_shape: format!("{}", n_samples * n_params),
            actual_shape: format!("{}", values.len()),
        });
    }

    Array2::from_shape_vec((n_samples, n_params), values).map_err(|e| {
        GradientLSSError::ShapeMismatch {
            expected_shape: format!("({}, {})", n_samples, n_params),
            actual_shape: e.to_string(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lightgbm_params_default() {
        let params = LightGBMParams::default();
        assert_eq!(params.n_dist_params(), 1);
    }

    #[test]
    fn test_reshape_gradients_fortran_order() {
        let gradients = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let hessians = Array2::ones((3, 2));

        let (grad_flat, _) = LightGBMBackend::reshape_gradients(&gradients, &hessians);

        // Fortran order for [[1,2],[3,4],[5,6]]: [1, 3, 5, 2, 4, 6]
        assert_eq!(grad_flat.len(), 6);
        assert_eq!(grad_flat[0], 1.0); // [0,0]
        assert_eq!(grad_flat[1], 3.0); // [1,0]
        assert_eq!(grad_flat[2], 5.0); // [2,0]
        assert_eq!(grad_flat[3], 2.0); // [0,1]
        assert_eq!(grad_flat[4], 4.0); // [1,1]
        assert_eq!(grad_flat[5], 6.0); // [2,1]
    }

    #[test]
    fn test_lightgbm_params_n_dist_params() {
        let mut params = LightGBMParams::default();
        params.set_n_dist_params(3);
        assert_eq!(params.n_dist_params(), 3);
    }
}
