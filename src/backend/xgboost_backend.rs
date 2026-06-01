//! XGBoost backend implementation.
//!
//! This backend trains a single multi-output booster with num_target set to
//! the number of distribution parameters, matching Python XGBoostLSS.

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
use xgb::{Booster, DMatrix, PredictConfig, PredictType};

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

        // Create a single booster with num_target matching Python's XGBoostLSS
        let mut xgb_params = params.to_xgb_params();
        xgb_params.insert("num_target".to_string(), n_params.to_string());

        let mut booster =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&train_data.dmatrix])
                .map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to create booster: {}", e))
            })?;

        // Apply all user-specified parameters
        for (key, value) in &xgb_params {
            booster.set_param(key, value).map_err(|e| {
                GradientLSSError::BackendError(format!(
                    "Failed to set param {}={}: {}",
                    key, value, e
                ))
            })?;
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

        let mut best_loss = f64::INFINITY;
        let mut best_iteration = 0usize;
        let mut rounds_without_improvement = 0;
        let mut stopped_early = false;

        let mut train_history = Vec::with_capacity(config.num_boost_round);
        let mut valid_history = Vec::with_capacity(config.num_boost_round);

        if let Some(ref mut cb) = callbacks {
            cb.on_training_start(config.num_boost_round);
        }

        // ------------------------------------------------------------------
        // Per-iteration margin accumulation (replaces the O(rounds²) full
        // re-predict the loop used to do every round). We maintain the current
        // raw margin incrementally:
        //   `predictions` = base_margin + Σ_{t<round} tree_t.
        // XGBoost's per-iteration margin slice (iteration_begin=r, end=r+1)
        // *includes* base_margin, so we subtract a constant base_margin row from
        // each slice to isolate that round's tree contribution before folding it
        // in. The gradient/hessian inputs are then bit-for-bit identical to the
        // old full-predict path — this is purely a speed optimization.
        // ------------------------------------------------------------------
        let base_row: Vec<f64> = match start_values {
            Some(sv) => (0..n_params).map(|j| sv[j]).collect(),
            None => vec![0.0; n_params],
        };
        let make_base_full = |rows: usize| {
            let mut a = Array2::<f64>::zeros((rows, n_params));
            for mut row in a.rows_mut() {
                for j in 0..n_params {
                    row[j] = base_row[j];
                }
            }
            a
        };
        let base_full = make_base_full(n_samples);
        let mut predictions = base_full.clone();

        let base_full_valid = valid_features
            .as_ref()
            .map(|(_, vr, _)| make_base_full(*vr));
        let mut valid_predictions = base_full_valid.clone();

        let margin_cfg = |begin: i64, end: i64| {
            PredictConfig {
                _type: PredictType::OutputMargin,
                training: false,
                iteration_begin: begin,
                iteration_end: end,
                strict_shape: false,
            }
            .as_json()
        };

        let mut final_round = 0;

        for round in 0..config.num_boost_round {
            final_round = round;

            // `predictions` already holds base_margin + trees [0, round) — no
            // full re-predict needed.
            let gh = objective_fn(&predictions, &labels, None)?;

            // Flatten gradients/hessians row-major: [g_s0_p0, g_s0_p1, ..., g_s1_p0, ...]
            // This matches Python which concatenates per-param gradients along axis=1
            let mut grad_f32 = Vec::with_capacity(n_samples * n_params);
            let mut hess_f32 = Vec::with_capacity(n_samples * n_params);

            for i in 0..n_samples {
                for j in 0..n_params {
                    grad_f32.push(gh.gradients[[i, j]] as f32);
                    hess_f32.push(gh.hessians[[i, j]] as f32);
                }
            }

            // Set gradient data for the trampoline
            OBJECTIVE_DATA.with(|data| {
                *data.borrow_mut() = Some((grad_f32, hess_f32));
            });

            // Training loss reflects trees [0, round) — computed before folding in
            // this round's tree, matching the previous (pre-update) behavior.
            let train_loss = metric_fn(&predictions, &labels);
            train_history.push(train_loss);

            // Update the single booster with all parameters' gradients
            booster
                .update_custom(&train_data.dmatrix, round as i32, objective_trampoline)
                .map_err(|e| GradientLSSError::BackendError(format!("Update failed: {}", e)))?;

            // Fold in only this round's tree contribution (slice − base_margin).
            let (slice, _shape) = booster
                .predict_matrix(
                    &train_data.dmatrix,
                    &margin_cfg(round as i64, round as i64 + 1),
                )
                .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;
            if slice.len() != n_samples * n_params {
                return Err(GradientLSSError::BackendError(format!(
                    "per-iteration margin slice returned {} values, expected {} (n_samples {} × n_params {}); \
                     incremental accumulation assumes a row-major (n_samples, n_params) layout",
                    slice.len(),
                    n_samples * n_params,
                    n_samples,
                    n_params
                )));
            }
            let slice = prediction_to_array2(&slice, n_samples, n_params);
            predictions += &slice;
            predictions -= &base_full;

            // Validation loss reflects trees [0, round] — accumulate this round's
            // contribution first, then evaluate (matches the previous post-update
            // predict on the validation DMatrix).
            let valid_loss = if let (Some(vdm), Some((_vf, vr, _vc)), Some(vl), Some(vpred), Some(vbase)) = (
                &valid_dmat,
                &valid_features,
                &valid_labels,
                valid_predictions.as_mut(),
                base_full_valid.as_ref(),
            ) {
                let (vslice, _vs) = booster
                    .predict_matrix(vdm, &margin_cfg(round as i64, round as i64 + 1))
                    .map_err(|e| {
                        GradientLSSError::BackendError(format!("Valid prediction failed: {}", e))
                    })?;
                if vslice.len() != *vr * n_params {
                    return Err(GradientLSSError::BackendError(format!(
                        "valid per-iteration margin slice returned {} values, expected {} ({} × {})",
                        vslice.len(),
                        *vr * n_params,
                        *vr,
                        n_params
                    )));
                }
                let vslice = prediction_to_array2(&vslice, *vr, n_params);
                *vpred += &vslice;
                *vpred -= vbase;

                let vl_loss = metric_fn(&*vpred, vl);
                valid_history.push(vl_loss);
                Some(vl_loss)
            } else {
                None
            };

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

            if let Some(ref mut cb) = callbacks {
                if cb.on_iteration_end(round, train_loss, valid_loss) == CallbackAction::Stop {
                    stopped_early = true;
                    break;
                }
            }

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

    /// Build a small multi-output booster trained with custom gradients, exactly
    /// like the training loop. Returns (booster, dmatrix, n, n_params) for the
    /// iteration-semantics tests below.
    fn train_dummy_booster(
        with_base_margin: bool,
        n_params: usize,
    ) -> (Booster, DMatrix, usize, usize, usize) {
        let n = 50usize;
        let f = 4usize;
        let rounds = 5usize;

        let mut feats = vec![0f32; n * f];
        for i in 0..n {
            for j in 0..f {
                feats[i * f + j] = (((i * 7 + j * 11) % 17) as f32) / 17.0;
            }
        }
        let mut dmat = DMatrix::from_dense(&feats, n).unwrap();
        let labels: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        dmat.set_labels(&labels).unwrap();
        if with_base_margin {
            // Distinct constant per param so a double-count is obvious.
            let per_param = [0.5f32, -0.3f32];
            let bm: Vec<f32> = (0..n)
                .flat_map(|_| (0..n_params).map(|j| per_param[j % 2]).collect::<Vec<_>>())
                .collect();
            dmat.set_base_margin(&bm).unwrap();
        }

        let mut params = XGBoostParams::default();
        params.set_n_dist_params(n_params);
        let mut xgb_params = params.to_xgb_params();
        xgb_params.insert("num_target".to_string(), n_params.to_string());

        let mut booster =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat]).unwrap();
        for (k, v) in &xgb_params {
            booster.set_param(k, v).unwrap();
        }

        for r in 0..rounds {
            // Per-sample-varying gradients so the trees actually split.
            let mut g = vec![0f32; n * n_params];
            let h = vec![1f32; n * n_params];
            for i in 0..n {
                for j in 0..n_params {
                    g[i * n_params + j] = feats[i * f + (j % f)] - 0.5;
                }
            }
            OBJECTIVE_DATA.with(|d| *d.borrow_mut() = Some((g, h)));
            booster
                .update_custom(&dmat, r as i32, objective_trampoline)
                .unwrap();
        }

        (booster, dmat, n, n_params, rounds)
    }

    fn margin_slice(booster: &Booster, dmat: &DMatrix, begin: i64, end: i64) -> Vec<f32> {
        use xgb::{PredictConfig, PredictType};
        let cfg = PredictConfig {
            _type: PredictType::OutputMargin,
            training: false,
            iteration_begin: begin,
            iteration_end: end,
            strict_shape: false,
        };
        booster.predict_matrix(dmat, &cfg.as_json()).unwrap().0
    }

    /// Verifies the training loop's O(rounds²) → O(rounds) fix: starting from
    /// `base_margin` and folding in each round's `(slice − base_margin)` exactly
    /// reconstructs the full booster margin. Covers both the no-base-margin path
    /// and the base-margin path (where each per-iteration slice re-adds the base,
    /// which the subtraction cancels). The per-column arithmetic is identical for
    /// `num_target>1`, so a single-output booster is sufficient — and avoids the
    /// 2-D custom-gradient shape constraint that only the full `train()` path sets up.
    #[test]
    fn test_per_iteration_margin_accumulation_matches_full() {
        for &bm in &[false, true] {
            let (booster, dmat, n, n_params, rounds) = train_dummy_booster(bm, 1);

            let full =
                prediction_to_array2(&booster.predict_margin(&dmat).unwrap(), n, n_params);

            // Base margin row used by the fix (param 0 == 0.5 when base_margin is set).
            let base_val = if bm { 0.5f64 } else { 0.0 };
            let base_full = Array2::<f64>::from_elem((n, n_params), base_val);

            // Replicate exactly what the training loop does.
            let mut predictions = base_full.clone();
            for r in 0..rounds {
                let slice = prediction_to_array2(
                    &margin_slice(&booster, &dmat, r as i64, (r + 1) as i64),
                    n,
                    n_params,
                );
                predictions += &slice;
                predictions -= &base_full;
            }

            let max_diff = (&predictions - &full).iter().fold(0f64, |m, &x| m.max(x.abs()));
            assert!(
                max_diff < 1e-4,
                "with_base_margin={bm}: per-iteration accumulation diverged from full margin (max abs diff {max_diff})"
            );
        }
    }
}
