//! XGBoost backend implementation.
//!
//! This backend trains a single multi-output booster with num_target set to
//! the number of distribution parameters, matching Python XGBoostLSS.

use super::traits::{
    Backend, BackendDataset, BackendModel, BackendParams, CallbackAction, EARLY_STOP_REL_DELTA,
    FeatureImportance, FeatureImportanceType, ParamValue, TrainConfig, TrainingCallback,
    TrainingResult, is_significant_improvement,
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
        // Move the buffers out rather than cloning — they're rebuilt each round
        // before `update_custom`, so there's no need to keep a copy here. Avoids
        // a per-round memcpy of both (n_samples * n_params) f32 vectors.
        data.borrow_mut()
            .take()
            .expect("Objective data was not set before update_custom call")
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
        // Histogram split-finding — the same algorithm LightGBM uses. Without
        // this, XGBoost defaults to `auto` (exact/approx greedy), which is ~10x
        // slower here (esp. for the 2-parameter NegativeBinomial). This is the
        // single biggest training-speed lever for the XGBoost backend.
        inner.insert("tree_method".to_string(), "hist".to_string());
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
    /// Per-sample training weights, applied to grad/hess in the objective.
    weights: Option<Array1<f64>>,
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

        // Use first n_rows labels for XGBoost DMatrix.
        // For multivariate targets the full label array is stored separately.
        let labels_f32: Vec<f32> = labels.iter().take(n_rows).map(|&x| x as f32).collect();

        // Create DMatrix from dense array (row-major)
        let mut dmatrix = DMatrix::from_dense(&features_f32, n_rows).map_err(|e| {
            GradientLSSError::BackendError(format!("Failed to create DMatrix: {}", e))
        })?;

        dmatrix
            .set_labels(&labels_f32)
            .map_err(|e| GradientLSSError::BackendError(format!("Failed to set labels: {}", e)))?;

        Ok(Self {
            dmatrix,
            n_rows,
            n_cols,
            features: features_f32,
            full_labels: labels.to_vec(),
            weights: None,
        })
    }

    fn set_init_score(&mut self, init_score: &Array1<f64>) -> Result<()> {
        // Set base_margin on the DMatrix, matching Python's set_base_margin.
        // For multi-output, base_margin is flattened row-major: n_samples * n_params.
        let margin_f32: Vec<f32> = init_score.iter().map(|&x| x as f32).collect();
        self.dmatrix.set_base_margin(&margin_f32).map_err(|e| {
            GradientLSSError::BackendError(format!("Failed to set base_margin: {}", e))
        })?;
        Ok(())
    }

    /// Plumb per-sample weights into the underlying DMatrix. Mirrors XGBoost's
    /// `DMatrix::set_weights` — the booster picks weights up automatically during
    /// training and rescales gradients/hessians per-sample.
    fn set_weights(&mut self, weights: ArrayView1<f64>) -> Result<()> {
        if weights.len() != self.n_rows {
            return Err(GradientLSSError::BackendError(format!(
                "set_weights: expected {} weights, got {}",
                self.n_rows,
                weights.len()
            )));
        }
        let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
        self.dmatrix.set_weights(&weights_f32).map_err(|e| {
            GradientLSSError::BackendError(format!("Failed to set weights: {}", e))
        })?;
        // Keep a copy: XGBoost ignores DMatrix weights for custom objectives, so
        // the training loop reads these back and weights grad/hess itself.
        self.weights = Some(weights.to_owned());
        Ok(())
    }

    fn supports_weights() -> bool {
        true
    }

    fn num_rows(&self) -> usize {
        self.n_rows
    }

    fn get_labels(&self) -> Result<Array1<f64>> {
        Ok(Array1::from(self.full_labels.clone()))
    }

    fn get_weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
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

/// XGBoost model wrapper - single multi-output booster matching Python XGBoostLSS.
pub struct XGBoostModel {
    booster: Booster,
    n_params: usize,
}

impl std::fmt::Debug for XGBoostModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XGBoostModel")
            .field("n_params", &self.n_params)
            .finish()
    }
}

/// Helper to reshape flat predictions from XGBoost into (n_samples, n_params).
fn prediction_to_array2(preds: &[f32], n_samples: usize, n_params: usize) -> Array2<f64> {
    let mut result = Array2::zeros((n_samples, n_params));
    // XGBoost multi-output returns predictions row-major: [s0_p0, s0_p1, ..., s1_p0, ...]
    for i in 0..n_samples {
        for j in 0..n_params {
            result[[i, j]] = preds[i * n_params + j] as f64;
        }
    }
    result
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

        // Set base_margin on DMatrix if start values are provided.
        // This matches Python: base_margin = np.ones((n_rows, 1)) * start_values, flattened.
        if let Some(sv) = start_values {
            let mut margin = Vec::with_capacity(n_samples * n_params);
            for _i in 0..n_samples {
                for j in 0..n_params {
                    margin.push(sv[j] as f32);
                }
            }
            train_data.dmatrix.set_base_margin(&margin).map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to set base_margin: {}", e))
            })?;
        }

        let labels = train_data.get_labels()?;

        // Per-sample weights (if any). XGBoost does not apply DMatrix weights to
        // custom-objective grad/hess, so we read them here and hand them to the
        // objective, which scales grad/hess per-sample — matching XGBoostLSS.
        let train_weights = train_data.get_weights().cloned();

        // For validation-based early stopping.
        let (valid_features, valid_labels) = if let Some(ref vd) = valid_data {
            let vl = vd.get_labels()?;
            Some((vd.features().to_vec(), vd.n_rows, vd.n_cols, vl))
        } else {
            None
        }
        .map_or((None, None), |(f, r, c, l)| (Some((f, r, c)), Some(l)));

        // Build the validation DMatrix ONCE — its contents and base_margin are
        // constant across boosting rounds, so rebuilding it every round (as the
        // old loop did) was pure waste.
        let valid_dmat: Option<DMatrix> = match &valid_features {
            Some((vf, vr, _vc)) => {
                let mut dm = DMatrix::from_dense(vf, *vr).map_err(|e| {
                    GradientLSSError::BackendError(format!("Failed to create valid DMatrix: {}", e))
                })?;
                if let Some(sv) = start_values {
                    let mut margin = Vec::with_capacity(*vr * n_params);
                    for _i in 0..*vr {
                        for j in 0..n_params {
                            margin.push(sv[j] as f32);
                        }
                    }
                    dm.set_base_margin(&margin).map_err(|e| {
                        GradientLSSError::BackendError(format!(
                            "Failed to set valid base_margin: {}",
                            e
                        ))
                    })?;
                }
                Some(dm)
            }
            None => None,
        };

        // Create the booster, caching BOTH the train and (when present) validation
        // DMatrices. XGBoost keeps an incremental prediction cache for cached
        // DMatrices and refreshes it inside each TrainOneIter, so `predict()` on
        // either inside the loop is an O(new-trees) cache read rather than a full
        // re-prediction — this is what keeps the loop O(rounds), not O(rounds²).
        // (XGBoost-specific: the LightGBM backend has no such cache, which is why
        // it accumulates per-iteration margins instead.)
        let mut xgb_params = params.to_xgb_params();
        xgb_params.insert("num_target".to_string(), n_params.to_string());

        let cached_dmats: Vec<&DMatrix> = match &valid_dmat {
            Some(vdm) => vec![&train_data.dmatrix, vdm],
            None => vec![&train_data.dmatrix],
        };
        let mut booster =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &cached_dmats).map_err(
                |e| GradientLSSError::BackendError(format!("Failed to create booster: {}", e)),
            )?;

        // Apply all user-specified parameters
        for (key, value) in &xgb_params {
            booster.set_param(key, value).map_err(|e| {
                GradientLSSError::BackendError(format!(
                    "Failed to set param {}={}: {}",
                    key, value, e
                ))
            })?;
        }

        // Whether the per-round train metric is needed at all: it drives early
        // stopping when there is no validation set, feeds verbose logging and
        // callbacks, and is recorded when `collect_train_metrics` asks for it.
        // When none of those apply, skipping it saves a full NLL pass over the
        // training set every round.
        let need_train_metric = config.collect_train_metrics
            || config.verbose
            || callbacks.is_some()
            || valid_dmat.is_none();

        let mut best_loss = f64::INFINITY;
        let mut best_iteration = 0usize;
        let mut rounds_without_improvement = 0;
        let mut stopped_early = false;

        let mut train_history = Vec::with_capacity(config.num_boost_round);
        let mut valid_history = Vec::with_capacity(config.num_boost_round);

        if let Some(ref mut cb) = callbacks {
            cb.on_training_start(config.num_boost_round);
        }

        let mut final_round = 0;

        // Initial train margin = trees [0, 0) (just the base_margin). We carry the
        // margin across rounds: each round's post-update prediction is reused as
        // the next round's pre-update (gradient) input, so the train matrix is
        // predicted once per round (＋1 here) rather than twice. The train DMatrix
        // is cached at booster creation, so each predict is an O(new-trees) cache
        // read.
        let mut predictions = {
            let raw_preds = booster
                .predict(&train_data.dmatrix)
                .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;
            prediction_to_array2(&raw_preds, n_samples, n_params)
        };

        for round in 0..config.num_boost_round {
            final_round = round;

            // Gradients/hessians on the current margin = trees [0, round).
            let gh = objective_fn(&predictions, &labels, train_weights.as_ref())?;

            // Flatten gradients/hessians row-major: [g_s0_p0, g_s0_p1, ..., g_s1_p0, ...]
            // This matches Python which concatenates per-param gradients along axis=1.
            // The arrays are C-order, so ndarray's iter IS row-major order — a single
            // contiguous cast pass that LLVM can vectorize, instead of per-element
            // `[[i, j]]` index arithmetic.
            let grad_f32: Vec<f32> = gh.gradients.iter().map(|&v| v as f32).collect();
            let hess_f32: Vec<f32> = gh.hessians.iter().map(|&v| v as f32).collect();

            // Set gradient data for the trampoline
            OBJECTIVE_DATA.with(|data| {
                *data.borrow_mut() = Some((grad_f32, hess_f32));
            });

            // Update the single booster with all parameters' gradients
            booster
                .update_custom(&train_data.dmatrix, round as i32, objective_trampoline)
                .map_err(|e| GradientLSSError::BackendError(format!("Update failed: {}", e)))?;

            // Refresh the train margin to the post-update model (trees [0, round]).
            // The train loss is therefore measured AFTER this round's tree is added,
            // matching the validation loss below (both reflect trees [0, round]) —
            // so train-only early stopping no longer lags the model by one tree.
            // This same array is carried into the next round as its pre-update
            // (gradient) margin, keeping it at one predict per round.
            let raw_preds = booster
                .predict(&train_data.dmatrix)
                .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;
            predictions = prediction_to_array2(&raw_preds, n_samples, n_params);

            let train_loss = if need_train_metric {
                let loss = metric_fn(&predictions, &labels);
                train_history.push(loss);
                Some(loss)
            } else {
                None
            };

            // Validation loss (post-update → trees [0, round]). The validation
            // DMatrix is also cached, so this is likewise a cache read.
            let valid_loss = if let (Some(vdm), Some((_vf, vr, _vc)), Some(vl)) =
                (&valid_dmat, &valid_features, &valid_labels)
            {
                let valid_raw = booster.predict(vdm).map_err(|e| {
                    GradientLSSError::BackendError(format!("Valid prediction failed: {}", e))
                })?;
                let valid_preds = prediction_to_array2(&valid_raw, *vr, n_params);
                let vl_loss = metric_fn(&valid_preds, vl);
                valid_history.push(vl_loss);
                Some(vl_loss)
            } else {
                None
            };

            let eval_loss = valid_loss
                .or(train_loss)
                .expect("train metric is computed whenever no validation set is present");

            if config.verbose && round % 10 == 0 {
                // `need_train_metric` is true when verbose, so this is Some.
                let tl = train_loss.unwrap_or(f64::NAN);
                match valid_loss {
                    Some(vl) => println!(
                        "[{}] train_loss: {:.6}, valid_loss: {:.6}",
                        round, tl, vl
                    ),
                    None => println!("[{}] train_loss: {:.6}", round, tl),
                }
            }

            if let Some(ref mut cb) = callbacks {
                // `need_train_metric` is true when callbacks are present.
                let tl = train_loss.unwrap_or(f64::NAN);
                if cb.on_iteration_end(round, tl, valid_loss) == CallbackAction::Stop {
                    stopped_early = true;
                    break;
                }
            }

            // Count this round as progress only if it beats the best by a
            // relative margin (see `is_significant_improvement`). Without the
            // min-delta, a noisy sub-1e-4 NLL tick reset the patience counter
            // every round, so NegBinom fits rode `num_boost_round` to the end
            // instead of early-stopping.
            if is_significant_improvement(eval_loss, best_loss, EARLY_STOP_REL_DELTA) {
                best_loss = eval_loss;
                best_iteration = round;
                rounds_without_improvement = 0;
            } else {
                rounds_without_improvement += 1;
            }

            if let Some(early_stopping) = config.early_stopping_rounds {
                if rounds_without_improvement >= early_stopping {
                    if config.verbose {
                        println!("Early stopping at round {}", round);
                    }
                    stopped_early = true;
                    break;
                }
            }
        }

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

        Ok((Self { booster, n_params }, result))
    }

    fn predict_raw(&self, data: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.nrows();

        let features_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let dmatrix = DMatrix::from_dense(&features_f32, n_samples).map_err(|e| {
            GradientLSSError::BackendError(format!("Failed to create DMatrix: {}", e))
        })?;

        let raw_preds = self
            .booster
            .predict(&dmatrix)
            .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;

        Ok(prediction_to_array2(&raw_preds, n_samples, self.n_params))
    }

    fn save_to_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write n_params so we know how to reshape predictions on load
        writer
            .write_all(&(self.n_params as u64).to_le_bytes())
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        // Save the single booster
        let temp_file =
            NamedTempFile::new().map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        let temp_path = temp_file.path();

        self.booster.save(temp_path).map_err(|e| {
            GradientLSSError::BackendError(format!("Failed to save booster to temp file: {}", e))
        })?;

        let model_bytes =
            std::fs::read(temp_path).map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        writer
            .write_all(&(model_bytes.len() as u64).to_le_bytes())
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        writer
            .write_all(&model_bytes)
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        Ok(())
    }

    fn load_from_reader<R: Read>(reader: &mut R) -> Result<Self> {
        // Read n_params
        let mut n_params_bytes = [0u8; 8];
        reader
            .read_exact(&mut n_params_bytes)
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        let n_params = u64::from_le_bytes(n_params_bytes) as usize;

        // Read the single booster
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

        Ok(Self { booster, n_params })
    }

    fn feature_importance(
        &self,
        importance_type: FeatureImportanceType,
        feature_names: Option<Vec<String>>,
    ) -> Result<FeatureImportance> {
        let n_params = self.n_params;

        let mut importance: HashMap<String, f64> = HashMap::new();

        // Parse feature importance from model dump
        if let Ok(model_dump) = self.booster.dump_model(true, None) {
            for line in model_dump.lines() {
                if let Some(start) = line.find("[f") {
                    if let Some(end) = line[start..].find('<').or_else(|| line[start..].find(']')) {
                        let feature_name = &line[start + 1..start + end];
                        *importance.entry(feature_name.to_string()).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        let mut all_features: Vec<String> = importance.keys().cloned().collect();
        all_features.sort();

        if all_features.is_empty() {
            return Ok(FeatureImportance {
                feature_indices: vec![],
                feature_names,
                scores: Array2::zeros((0, n_params)),
                importance_type,
            });
        }

        let n_features = all_features.len();

        // For a single multi-output booster, feature importance is shared across params.
        // Replicate the scores for each param column to match the expected shape.
        let mut scores_vec = Vec::with_capacity(n_features * n_params);
        for feat in &all_features {
            let score = importance.get(feat).copied().unwrap_or(0.0);
            for _ in 0..n_params {
                scores_vec.push(score);
            }
        }

        let scores = Array2::from_shape_vec((n_features, n_params), scores_vec).map_err(|e| {
            GradientLSSError::ShapeMismatch {
                expected_shape: format!("({}, {})", n_features, n_params),
                actual_shape: e.to_string(),
            }
        })?;

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
        assert_eq!(params.n_dist_params(), 1);
        params.set_n_dist_params(3);
        assert_eq!(params.n_dist_params(), 3);
    }

    /// Manual benchmark proving the prediction-cache claim:
    /// `cargo test --release --features full --lib xgb_predict_cache_timing -- --ignored --nocapture`
    ///
    /// Strategy A is the restored loop — `predict()` on the cached train DMatrix
    /// each round. Strategy B is the reverted-out loop — `predict_matrix(r..r+1)`
    /// each round. Both build identical trees (`update`); only the per-round
    /// prediction differs. If XGBoost's prediction cache is real, A's accumulated
    /// predict time stays ~flat while B's grows with the model, so B ≫ A.
    #[test]
    #[ignore]
    fn xgb_predict_cache_timing() {
        use std::time::{Duration, Instant};
        use xgb::{PredictConfig, PredictType};

        let n = 12000usize;
        let f = 60usize;
        let rounds = 300usize;

        let mut feats = vec![0f32; n * f];
        for i in 0..n {
            for j in 0..f {
                feats[i * f + j] = (((i * 7 + j * 13) % 101) as f32) / 101.0;
            }
        }
        let labels: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let params = [
            ("tree_method", "hist"),
            ("max_depth", "6"),
            ("eta", "0.1"),
            ("objective", "reg:squarederror"),
        ];

        // Strategy A: cached predict().
        let mut dtrain_a = DMatrix::from_dense(&feats, n).unwrap();
        dtrain_a.set_labels(&labels).unwrap();
        let mut b_a =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dtrain_a]).unwrap();
        for (k, v) in params {
            b_a.set_param(k, v).unwrap();
        }
        let mut predict_a = Duration::ZERO;
        let total_a = Instant::now();
        for r in 0..rounds {
            let t = Instant::now();
            let _ = b_a.predict(&dtrain_a).unwrap();
            predict_a += t.elapsed();
            b_a.update(&dtrain_a, r as i32).unwrap();
        }
        let total_a = total_a.elapsed();

        // Strategy B: predict_matrix(iteration_begin=r, end=r+1) each round.
        let mut dtrain_b = DMatrix::from_dense(&feats, n).unwrap();
        dtrain_b.set_labels(&labels).unwrap();
        let mut b_b =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dtrain_b]).unwrap();
        for (k, v) in params {
            b_b.set_param(k, v).unwrap();
        }
        let mut predict_b = Duration::ZERO;
        let total_b = Instant::now();
        for r in 0..rounds {
            // Mirror the regressed loop: train the round, THEN predict the
            // just-added iteration (so iteration_end = r+1 is in range).
            b_b.update(&dtrain_b, r as i32).unwrap();
            let cfg = PredictConfig {
                _type: PredictType::OutputMargin,
                training: false,
                iteration_begin: r as i64,
                iteration_end: r as i64 + 1,
                strict_shape: false,
            };
            let t = Instant::now();
            let _ = b_b.predict_matrix(&dtrain_b, &cfg.as_json()).unwrap();
            predict_b += t.elapsed();
        }
        let total_b = total_b.elapsed();

        eprintln!("XGB {rounds} rounds, {n}x{f}:");
        eprintln!("  A predict() cached:        predict={predict_a:?}  total={total_a:?}");
        eprintln!("  B predict_matrix(r..r+1):  predict={predict_b:?}  total={total_b:?}");
        eprintln!(
            "  predict-time ratio B/A = {:.1}x",
            predict_b.as_secs_f64() / predict_a.as_secs_f64().max(1e-9)
        );
    }

    #[test]
    fn test_prediction_to_array2() {
        let preds = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = prediction_to_array2(&preds, 3, 2);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[1, 0]], 3.0);
        assert_eq!(result[[1, 1]], 4.0);
        assert_eq!(result[[2, 0]], 5.0);
        assert_eq!(result[[2, 1]], 6.0);
    }

}
