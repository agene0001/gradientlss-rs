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
use std::collections::HashMap;
use std::io::{Read, Write};
use tempfile::NamedTempFile;

use xgb::parameters::BoosterParameters;
use xgb::{Booster, DMatrix};

/// XGBoost backend for GradientLSS.
#[derive(Debug, Clone)]
pub struct XGBoostBackend;

/// Render a `ParamValue` the way XGBoost's string config expects it.
fn param_value_to_string(value: &ParamValue) -> String {
    match value {
        ParamValue::Int(v) => v.to_string(),
        ParamValue::Float(v) => v.to_string(),
        ParamValue::String(v) => v.clone(),
        ParamValue::Bool(v) => v.to_string(),
    }
}

/// XGBoost-specific parameters.
#[derive(Debug, Clone)]
pub struct XGBoostParams {
    inner: HashMap<String, ParamValue>,
    n_dist_params: usize,
}

impl Default for XGBoostParams {
    fn default() -> Self {
        let mut inner = HashMap::new();
        inner.insert("booster".to_string(), ParamValue::from("gbtree"));
        // Histogram split-finding — the same algorithm LightGBM uses. Without
        // this, XGBoost defaults to `auto` (exact/approx greedy), which is ~10x
        // slower here (esp. for the 2-parameter NegativeBinomial). This is the
        // single biggest training-speed lever for the XGBoost backend.
        inner.insert("tree_method".to_string(), ParamValue::from("hist"));
        inner.insert("eta".to_string(), ParamValue::Float(0.1));
        inner.insert("max_depth".to_string(), ParamValue::Int(6));
        Self {
            inner,
            n_dist_params: 1,
        }
    }
}

impl BackendParams for XGBoostParams {
    fn set(&mut self, key: &str, value: ParamValue) {
        self.inner.insert(key.to_string(), value);
    }

    fn get(&self, key: &str) -> Option<&ParamValue> {
        self.inner.get(key)
    }

    fn to_map(&self) -> HashMap<String, ParamValue> {
        self.inner.clone()
    }
}

impl XGBoostParams {
    /// Get the params as the string map xgboost-rs consumes.
    pub fn to_xgb_params(&self) -> HashMap<String, String> {
        self.inner
            .iter()
            .map(|(k, v)| (k.clone(), param_value_to_string(v)))
            .collect()
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

/// XGBoost dataset wrapper.
///
/// Holds the raw features/labels; the training `DMatrix` is built lazily at
/// TRAIN time (see [`Self::build_train_dmatrix`]) so it can be a
/// `QuantileDMatrix` under the `hist` tree method — pre-binned values use ~1
/// byte per feature value instead of 4 and skip the sketching pass at the
/// start of training. Validation data never needs a stored DMatrix either
/// (the training loop builds one plain prediction matrix from `features()`),
/// so nothing is constructed eagerly.
pub struct XGBoostDataset {
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

        Ok(Self {
            n_rows,
            n_cols,
            features: features_f32,
            full_labels: labels.to_vec(),
            weights: None,
        })
    }

    fn set_init_score(&mut self, _init_score: &Array1<f64>) -> Result<()> {
        // Intentional no-op: the training path sets the start values as the
        // base margin on the train/valid DMatrices it builds (see
        // `build_train_dmatrix`), and `predict` adds them back after
        // `predict_raw` — there is no stored DMatrix to set anything on.
        Ok(())
    }

    /// Store per-sample weights. They are set on the training `DMatrix` when
    /// `build_train_dmatrix` runs, and the training loop additionally hands
    /// them to the objective, which scales grad/hess per-sample — XGBoost does
    /// not apply DMatrix weights to custom-objective gradients (matching
    /// XGBoostLSS's `grad *= weights; hess *= weights`).
    fn set_weights(&mut self, weights: ArrayView1<f64>) -> Result<()> {
        if weights.len() != self.n_rows {
            return Err(GradientLSSError::BackendError(format!(
                "set_weights: expected {} weights, got {}",
                self.n_rows,
                weights.len()
            )));
        }
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
    /// Build the training `DMatrix` with base margin, labels, and weights.
    ///
    /// Under the `hist` tree method this is a `QuantileDMatrix`
    /// (`from_dense_quantile`): pre-binned storage, ~4x less memory, no
    /// separate sketching pass — with `max_bin` taken from the training params
    /// so the binning always matches the booster (a mismatch is a hard XGBoost
    /// error). Any other tree method gets a plain `DMatrix`, since quantile
    /// matrices are `hist`-only. Prediction on a quantile matrix is exact for
    /// its own trees (hist split thresholds are bin boundaries), so the
    /// training loop's cached per-round predicts are unaffected.
    fn build_train_dmatrix(
        &self,
        xgb_params: &HashMap<String, String>,
        n_params: usize,
        start_values: Option<&Array1<f64>>,
    ) -> Result<DMatrix> {
        // First n_rows labels; for multivariate targets these are dummies (the
        // objective reads the real labels from `full_labels`).
        let labels_f32: Vec<f32> = self
            .full_labels
            .iter()
            .take(self.n_rows)
            .map(|&x| x as f32)
            .collect();

        let use_quantile = xgb_params.get("tree_method").map(String::as_str) == Some("hist");
        let mut dmatrix = if use_quantile {
            let max_bin: u32 = xgb_params
                .get("max_bin")
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);
            DMatrix::from_dense_quantile(&self.features, self.n_rows, Some(&labels_f32), max_bin)
                .map_err(|e| {
                    GradientLSSError::BackendError(format!(
                        "Failed to create QuantileDMatrix: {}",
                        e
                    ))
                })?
        } else {
            let mut dm = DMatrix::from_dense(&self.features, self.n_rows).map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to create DMatrix: {}", e))
            })?;
            dm.set_labels(&labels_f32).map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to set labels: {}", e))
            })?;
            dm
        };

        // Start values as base margin, matching Python:
        // base_margin = np.ones((n_rows, 1)) * start_values, flattened row-major.
        if let Some(sv) = start_values {
            let mut margin = Vec::with_capacity(self.n_rows * n_params);
            for _i in 0..self.n_rows {
                for j in 0..n_params {
                    margin.push(sv[j] as f32);
                }
            }
            dmatrix.set_base_margin(&margin).map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to set base_margin: {}", e))
            })?;
        }

        if let Some(ref w) = self.weights {
            let weights_f32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
            dmatrix.set_weights(&weights_f32).map_err(|e| {
                GradientLSSError::BackendError(format!("Failed to set weights: {}", e))
            })?;
        }

        Ok(dmatrix)
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
    /// Number of features the model was trained on. 0 when unknown (models
    /// saved before this field existed — see `load_from_reader`).
    n_features: usize,
    /// 0-based index of the best boosting iteration selected by early stopping.
    /// `Some(b)` means predictions must use trees for iterations `[0, b]`
    /// (i.e. `b + 1` iterations), truncating the overfit tail that early
    /// stopping trained past the best. `None` means no early stopping was in
    /// effect — predict with all trees.
    best_iteration: Option<usize>,
}

impl std::fmt::Debug for XGBoostModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XGBoostModel")
            .field("n_params", &self.n_params)
            .field("best_iteration", &self.best_iteration)
            .finish()
    }
}

/// Copy a flat row-major f32 prediction buffer into an existing (n_samples,
/// n_params) f64 array — the reused-buffer counterpart of
/// `prediction_to_array2` for the training loop's persistent margin arrays.
fn copy_preds_into(preds: &[f32], out: &mut Array2<f64>) -> Result<()> {
    if preds.len() != out.len() {
        return Err(GradientLSSError::ShapeMismatch {
            expected_shape: format!("{}", out.len()),
            actual_shape: format!("{}", preds.len()),
        });
    }
    // Both are C-order row-major, so element order matches.
    for (o, &p) in out.iter_mut().zip(preds) {
        *o = p as f64;
    }
    Ok(())
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

        let labels = train_data.get_labels()?;

        // Per-sample weights (if any). XGBoost does not apply DMatrix weights to
        // custom-objective grad/hess, so we read them here and hand them to the
        // objective, which scales grad/hess per-sample — matching XGBoostLSS.
        let train_weights = train_data.get_weights().cloned();

        let mut xgb_params = params.to_xgb_params();
        // Forced params, matching Python's `set_params_adj` which unconditionally
        // overwrites the user dict. These are correctness requirements, not
        // defaults: a user-carried `objective` (e.g. "count:poisson") makes every
        // predict return link-transformed values while boosting stays on margin
        // scale — silently corrupting gradients and predictions — and a nonzero
        // `base_score` is inert during training (base_margin overrides it) but
        // added by predict, shifting all outputs under the start values.
        xgb_params.remove("objective");
        xgb_params.insert("base_score".to_string(), "0.0".to_string());
        xgb_params.insert(
            "disable_default_eval_metric".to_string(),
            "true".to_string(),
        );
        xgb_params.insert("num_target".to_string(), n_params.to_string());
        // Wire TrainConfig::seed (subsample/colsample draws); an explicit
        // user-set "seed" param still wins.
        xgb_params
            .entry("seed".to_string())
            .or_insert_with(|| config.seed.to_string());

        // Training DMatrix, built with the forced params in hand so the
        // quantile binning (hist) sees the same max_bin as the booster;
        // carries base margin (start values) and weights.
        let train_dmat = train_data.build_train_dmatrix(&xgb_params, n_params, start_values)?;

        // For validation-based early stopping.
        let (valid_features, valid_labels) = if let Some(ref vd) = valid_data {
            let vl = vd.get_labels()?;
            Some((vd.n_rows, vd.n_cols, vl))
        } else {
            None
        }
        .map_or((None, None), |(r, c, l)| (Some((r, c)), Some(l)));

        // Build the validation DMatrix ONCE — its contents and base_margin are
        // constant across boosting rounds, so rebuilding it every round (as the
        // old loop did) was pure waste. Plain (non-quantile) DMatrix: it is
        // prediction-only, and prediction on raw values is exact for
        // hist-trained trees.
        let valid_dmat: Option<DMatrix> = match (&valid_data, &valid_features) {
            (Some(vd), Some((vr, _vc))) => {
                let mut dm = DMatrix::from_dense(vd.features(), *vr).map_err(|e| {
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
            _ => None,
        };

        // Create the booster, caching BOTH the train and (when present) validation
        // DMatrices. XGBoost keeps an incremental prediction cache for cached
        // DMatrices and refreshes it inside each TrainOneIter, so `predict()` on
        // either inside the loop is an O(new-trees) cache read rather than a full
        // re-prediction — this is what keeps the loop O(rounds), not O(rounds²).
        // (XGBoost-specific: the LightGBM backend has no such cache, which is why
        // it accumulates per-iteration margins instead.)
        let cached_dmats: Vec<&DMatrix> = match &valid_dmat {
            Some(vdm) => vec![&train_dmat, vdm],
            None => vec![&train_dmat],
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
        // Last loss that beat its predecessor by the relative min-delta; drives
        // ONLY the patience counter, never best_iteration/best_loss.
        let mut significant_best_loss = f64::INFINITY;
        let mut rounds_without_improvement = 0;
        let mut stopped_early = false;

        let mut train_history = Vec::with_capacity(config.num_boost_round);
        let mut valid_history = Vec::with_capacity(config.num_boost_round);

        if let Some(ref mut cb) = callbacks {
            cb.on_training_start(config.num_boost_round);
        }

        let mut final_round = 0;

        // Reused per-round buffers: grad/hess f32 staging for `boost`, one f32
        // prediction buffer for `predict_into`, and persistent train/valid
        // margin arrays refreshed in place. Steady-state, the loop allocates
        // nothing in the wrapper.
        let mut grad_f32: Vec<f32> = Vec::with_capacity(n_samples * n_params);
        let mut hess_f32: Vec<f32> = Vec::with_capacity(n_samples * n_params);
        let mut pred_buf: Vec<f32> = Vec::new();
        let valid_rows = valid_features.map_or(0, |(r, _c)| r);
        let mut valid_preds = Array2::<f64>::zeros((valid_rows, n_params));

        // Initial train margin = trees [0, 0) (just the base_margin). We carry the
        // margin across rounds: each round's post-update prediction is reused as
        // the next round's pre-update (gradient) input, so the train matrix is
        // predicted once per round (＋1 here) rather than twice. The train DMatrix
        // is cached at booster creation, so each predict is an O(new-trees) cache
        // read.
        let mut predictions = Array2::<f64>::zeros((n_samples, n_params));
        booster
            .predict_into(&train_dmat, &mut pred_buf)
            .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;
        copy_preds_into(&pred_buf, &mut predictions)?;

        for round in 0..config.num_boost_round {
            final_round = round;

            // Gradients/hessians on the current margin = trees [0, round).
            let gh = objective_fn(&predictions, &labels, train_weights.as_ref())?;

            // Flatten gradients/hessians row-major: [g_s0_p0, g_s0_p1, ..., g_s1_p0, ...]
            // This matches Python which concatenates per-param gradients along axis=1.
            // The arrays are C-order, so ndarray's iter IS row-major order — a single
            // contiguous cast pass that LLVM can vectorize, instead of per-element
            // `[[i, j]]` index arithmetic.
            grad_f32.clear();
            grad_f32.extend(gh.gradients.iter().map(|&v| v as f32));
            hess_f32.clear();
            hess_f32.extend(gh.hessians.iter().map(|&v| v as f32));

            // Update the single booster with all parameters' gradients. `boost`
            // takes them directly — `update_custom` would first re-predict on
            // the train matrix for its callback, a wasted cache read since the
            // carried `predictions` margin already holds that result.
            booster
                .boost(&train_dmat, round as i32, &grad_f32, &hess_f32)
                .map_err(|e| GradientLSSError::BackendError(format!("Update failed: {}", e)))?;

            // Refresh the train margin to the post-update model (trees [0, round]).
            // The train loss is therefore measured AFTER this round's tree is added,
            // matching the validation loss below (both reflect trees [0, round]) —
            // so train-only early stopping no longer lags the model by one tree.
            // This same array is carried into the next round as its pre-update
            // (gradient) margin, keeping it at one predict per round.
            booster
                .predict_into(&train_dmat, &mut pred_buf)
                .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;
            copy_preds_into(&pred_buf, &mut predictions)?;

            let train_loss = if need_train_metric {
                let loss = metric_fn(&predictions, &labels);
                train_history.push(loss);
                Some(loss)
            } else {
                None
            };

            // Validation loss (post-update → trees [0, round]). The validation
            // DMatrix is also cached, so this is likewise a cache read.
            let valid_loss = if let (Some(vdm), Some(vl)) = (&valid_dmat, &valid_labels) {
                booster.predict_into(vdm, &mut pred_buf).map_err(|e| {
                    GradientLSSError::BackendError(format!("Valid prediction failed: {}", e))
                })?;
                copy_preds_into(&pred_buf, &mut valid_preds)?;
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
                    Some(vl) => {
                        println!("[{}] train_loss: {:.6}, valid_loss: {:.6}", round, tl, vl)
                    }
                    None => println!("[{}] train_loss: {:.6}", round, tl),
                }
            }

            // best_iteration/best_loss track the TRUE argmin of the eval curve
            // (min_delta = 0, matching Python's xgboost/lightgbm early-stopping
            // bookkeeping) — predict-time truncation must keep every tree that
            // improved the monitored loss, however slightly.
            if eval_loss < best_loss {
                best_loss = eval_loss;
                best_iteration = round;
            }

            // The patience counter, by contrast, only counts a round as progress
            // if it beats the last significant best by a relative margin (see
            // `is_significant_improvement`). Without the min-delta, a noisy
            // sub-1e-4 NLL tick reset the counter every round, so NegBinom fits
            // rode `num_boost_round` to the end instead of early-stopping.
            if is_significant_improvement(eval_loss, significant_best_loss, EARLY_STOP_REL_DELTA) {
                significant_best_loss = eval_loss;
                rounds_without_improvement = 0;
            } else {
                rounds_without_improvement += 1;
            }

            // Callbacks run AFTER the argmin/patience bookkeeping: a callback
            // that stops on a new-minimum round must not exclude that round
            // from best_iteration (its loss is already in the histories above).
            if let Some(ref mut cb) = callbacks {
                // `need_train_metric` is true when callbacks are present.
                let tl = train_loss.unwrap_or(f64::NAN);
                if cb.on_iteration_end(round, tl, valid_loss) == CallbackAction::Stop {
                    stopped_early = true;
                    break;
                }
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

        // `final_round` stays 0 when the loop never ran (num_boost_round == 0),
        // so `final_round + 1` would fabricate one completed iteration.
        let n_iterations = if config.num_boost_round == 0 {
            0
        } else {
            final_round + 1
        };

        if let Some(ref mut cb) = callbacks {
            cb.on_training_end(n_iterations, stopped_early);
        }

        let result = TrainingResult {
            n_iterations,
            best_iteration: Some(best_iteration),
            best_score: Some(best_loss),
            train_history,
            valid_history,
            stopped_early,
        };

        // Only honor best_iteration at predict time when early stopping was
        // actually in effect: there must be a validation set to score against
        // AND `early_stopping_rounds` configured. Otherwise the model was
        // trained to the full `num_boost_round` and predictions should use all
        // trees (None).
        let effective_best_iteration =
            if valid_dmat.is_some() && config.early_stopping_rounds.is_some() {
                Some(best_iteration)
            } else {
                None
            };

        // Release XGBoost's internal training caches (gradient buffers, the
        // prediction caches for the cached DMatrices). The model is unaffected;
        // this just hands back memory the inference-only booster never needs.
        booster
            .reset()
            .map_err(|e| GradientLSSError::BackendError(format!("Booster reset failed: {}", e)))?;

        Ok((
            Self {
                booster,
                n_params,
                n_features: train_data.n_cols,
                best_iteration: effective_best_iteration,
            },
            result,
        ))
    }

    fn predict_raw(&self, data: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.nrows();

        let features_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

        let raw_preds = match self.best_iteration {
            // Early stopping selected a best iteration: predict over trees for
            // iterations [0, best_iteration] only. XGBoost's iteration_range is
            // [begin, end) (half-open), so end = best_iteration + 1. This drops
            // the overfit tail (up to early_stopping_rounds extra iterations)
            // that training added past the best. This is the one path that
            // still needs a DMatrix: inplace prediction has no iteration-range
            // knob. `PredictType::Normal` matches the semantics of the inplace
            // predict used below.
            Some(best_iteration) => {
                let dmatrix = DMatrix::from_dense(&features_f32, n_samples).map_err(|e| {
                    GradientLSSError::BackendError(format!("Failed to create DMatrix: {}", e))
                })?;
                let cfg = xgb::PredictConfig {
                    _type: xgb::PredictType::Normal,
                    training: false,
                    iteration_begin: 0,
                    iteration_end: (best_iteration + 1) as i64,
                    strict_shape: false,
                };
                let (preds, _shape) = self
                    .booster
                    .predict_matrix(&dmatrix, &cfg.as_json())
                    .map_err(|e| {
                        GradientLSSError::BackendError(format!("Prediction failed: {}", e))
                    })?;
                preds
            }
            // No early stopping in effect: all trees, predicted inplace
            // straight off the feature slice — constructing a DMatrix per
            // predict call copies and re-indexes the data for nothing and
            // dominates small-batch latency. NaN-as-missing matches the
            // DMatrix path.
            None => {
                let (preds, _shape) = self
                    .booster
                    .predict_from_dense(&features_f32, n_samples)
                    .map_err(|e| {
                        GradientLSSError::BackendError(format!("Prediction failed: {}", e))
                    })?;
                preds
            }
        };

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

        // Append best_iteration at the END of the format so older readers that
        // stop after the booster bytes are unaffected, and this reader can fall
        // back to None on EOF for models saved before this field existed.
        // Sentinel: -1 => None (predict all trees); >= 0 => Some(value).
        let best_iter_sentinel: i64 = match self.best_iteration {
            Some(b) => b as i64,
            None => -1,
        };
        writer
            .write_all(&best_iter_sentinel.to_le_bytes())
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

        // n_features, appended after best_iteration with the same trailing-
        // optional convention (older readers stop before it; this reader falls
        // back to 0 on EOF for models saved before the field existed).
        writer
            .write_all(&(self.n_features as u64).to_le_bytes())
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

        // Read the trailing best_iteration field. Older saved models predate
        // this field, so a clean EOF here means "no early stopping recorded"
        // → None (predict all trees). Any other read error is a real failure.
        let mut best_iter_bytes = [0u8; 8];
        let best_iteration = match reader.read_exact(&mut best_iter_bytes) {
            Ok(()) => {
                let sentinel = i64::from_le_bytes(best_iter_bytes);
                if sentinel < 0 {
                    None
                } else {
                    Some(sentinel as usize)
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => None,
            Err(e) => return Err(GradientLSSError::IoError(e.to_string())),
        };

        // Trailing n_features (same optional convention): 0 = unknown for
        // models saved before the field existed.
        let mut n_features_bytes = [0u8; 8];
        let n_features = match reader.read_exact(&mut n_features_bytes) {
            Ok(()) => u64::from_le_bytes(n_features_bytes) as usize,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => 0,
            Err(e) => return Err(GradientLSSError::IoError(e.to_string())),
        };

        Ok(Self {
            booster,
            n_params,
            n_features,
            best_iteration,
        })
    }

    fn feature_importance(
        &self,
        importance_type: FeatureImportanceType,
        feature_names: Option<Vec<String>>,
    ) -> Result<FeatureImportance> {
        let n_params = self.n_params;

        // Per-feature (split count, total gain, total cover) from the
        // with-stats model dump. Branch lines look like
        //   0:[f2<3.5] yes=1,no=2,missing=1,gain=100.5,cover=58
        // Python XGBoostLSS uses booster.get_score(importance_type="gain"),
        // where "gain"/"cover" are per-split AVERAGES — matched below.
        let mut importance: HashMap<String, (f64, f64, f64)> = HashMap::new();

        if let Ok(model_dump) = self.booster.dump_model(true, None) {
            for line in model_dump.lines() {
                if let Some(start) = line.find("[f") {
                    if let Some(end) = line[start..].find('<').or_else(|| line[start..].find(']')) {
                        let feature_name = &line[start + 1..start + end];
                        let gain = parse_dump_stat(line, "gain=").unwrap_or(0.0);
                        let cover = parse_dump_stat(line, "cover=").unwrap_or(0.0);
                        let entry = importance
                            .entry(feature_name.to_string())
                            .or_insert((0.0, 0.0, 0.0));
                        entry.0 += 1.0;
                        entry.1 += gain;
                        entry.2 += cover;
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
            let (count, gain, cover) = importance.get(feat).copied().unwrap_or((0.0, 0.0, 0.0));
            let score = match importance_type {
                FeatureImportanceType::Split => count,
                // Averages over splits, matching xgboost get_score semantics.
                FeatureImportanceType::Gain => {
                    if count > 0.0 {
                        gain / count
                    } else {
                        0.0
                    }
                }
                FeatureImportanceType::Cover => {
                    if count > 0.0 {
                        cover / count
                    } else {
                        0.0
                    }
                }
            };
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
        self.n_features
    }

    fn num_params(&self) -> usize {
        self.n_params
    }
}

/// Parse a numeric `key=value` stat (e.g. `gain=`, `cover=`) out of a model
/// dump branch line; the value runs to the next ',' or end of line.
fn parse_dump_stat(line: &str, key: &str) -> Option<f64> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest.find(',').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
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

    // ---- best_iteration / early-stopping truncation tests ----

    /// Build a noisy single-parameter regression problem where the model
    /// provably overfits: the validation loss bottoms out early, then rises as
    /// later trees fit training noise. Returns (train_feats, train_labels,
    /// valid_feats, valid_labels).
    fn overfit_dataset() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
        let f = 8usize;
        let n_train = 200usize;
        let n_valid = 120usize;

        // Simple deterministic LCG so the test is reproducible without rand deps.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (u32::MAX as f64)
        };

        let make = |rows: usize, next: &mut dyn FnMut() -> f64| {
            let mut feats = Array2::<f64>::zeros((rows, f));
            let mut labels = Array1::<f64>::zeros(rows);
            for i in 0..rows {
                let mut signal = 0.0;
                for j in 0..f {
                    let x = next();
                    feats[[i, j]] = x;
                    if j < 2 {
                        signal += x; // only 2 features carry real signal
                    }
                }
                // Heavy label noise so extra trees fit noise, not signal.
                let noise = (next() - 0.5) * 4.0;
                labels[i] = signal + noise;
            }
            (feats, labels)
        };

        let (tf, tl) = make(n_train, &mut next);
        let (vf, vl) = make(n_valid, &mut next);
        (tf, tl, vf, vl)
    }

    /// Gaussian-mean squared-error objective (1 param): gradient = pred - label,
    /// hessian = 1. Metric = MSE.
    fn sq_objective(
        preds: &Array2<f64>,
        labels: &Array1<f64>,
        _w: Option<&Array1<f64>>,
    ) -> Result<GradientsAndHessians> {
        let n = preds.nrows();
        let mut g = Array2::<f64>::zeros((n, 1));
        let mut h = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            g[[i, 0]] = preds[[i, 0]] - labels[i];
            h[[i, 0]] = 1.0;
        }
        Ok(GradientsAndHessians {
            gradients: g,
            hessians: h,
        })
    }

    fn sq_metric(preds: &Array2<f64>, labels: &Array1<f64>) -> f64 {
        let n = preds.nrows();
        let mut s = 0.0;
        for i in 0..n {
            let d = preds[[i, 0]] - labels[i];
            s += d * d;
        }
        s / n as f64
    }

    fn train_overfit_model() -> (XGBoostModel, TrainingResult) {
        let (tf, tl, vf, vl) = overfit_dataset();
        let mut train = XGBoostDataset::from_data(tf.view(), tl.view()).unwrap();
        let mut valid = XGBoostDataset::from_data(vf.view(), vl.view()).unwrap();

        let mut params = XGBoostParams::default();
        params.set_n_dist_params(1);
        // Aggressive learning + deep trees + many rounds => overfits well before
        // num_boost_round, so early stopping fires with a best in the interior.
        params.set("eta", ParamValue::Float(0.3));
        params.set("max_depth", ParamValue::Int(8));

        let config = TrainConfig {
            num_boost_round: 80,
            early_stopping_rounds: Some(5),
            verbose: false,
            seed: 7,
            collect_train_metrics: false,
        };

        XGBoostModel::train_with_objective_and_callbacks::<
            _,
            _,
            super::super::traits::HistoryCallback,
        >(
            &params,
            &mut train,
            Some(&mut valid),
            &config,
            sq_objective,
            sq_metric,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_best_iteration_truncates_prediction() {
        let (model, result) = train_overfit_model();

        // Early stopping selected an interior best.
        let best = model.best_iteration.expect("best_iteration should be Some");
        assert!(
            best + 1 < result.n_iterations,
            "expected best_iteration ({}) + 1 < total trees ({}); early stopping did not overfit past best",
            best,
            result.n_iterations
        );

        // Reference prediction bounded to exactly best+1 trees, computed directly.
        let (tf, _, _, _) = overfit_dataset();
        let feats_f32: Vec<f32> = tf.iter().map(|&x| x as f32).collect();
        let dmat = DMatrix::from_dense(&feats_f32, tf.nrows()).unwrap();

        let bounded_cfg = xgb::PredictConfig {
            _type: xgb::PredictType::Normal,
            training: false,
            iteration_begin: 0,
            iteration_end: (best + 1) as i64,
            strict_shape: false,
        };
        let (bounded_ref, _) = model
            .booster
            .predict_matrix(&dmat, &bounded_cfg.as_json())
            .unwrap();
        let bounded_ref = prediction_to_array2(&bounded_ref, tf.nrows(), 1);

        // All-trees prediction (what the buggy predict_raw did).
        let all_trees = model.booster.predict(&dmat).unwrap();
        let all_trees = prediction_to_array2(&all_trees, tf.nrows(), 1);

        // predict_raw must equal the bounded reference...
        let pr = model.predict_raw(&tf.view()).unwrap();
        let max_diff_bounded = (&pr - &bounded_ref)
            .iter()
            .fold(0f64, |m, &x| m.max(x.abs()));
        assert!(
            max_diff_bounded < 1e-5,
            "predict_raw must match best_iteration-bounded prediction (max diff {max_diff_bounded})"
        );

        // ...and DIFFER from the all-trees prediction (proves truncation happens).
        let max_diff_all = (&pr - &all_trees).iter().fold(0f64, |m, &x| m.max(x.abs()));
        assert!(
            max_diff_all > 1e-6,
            "predict_raw should differ from all-trees prediction, but max diff was {max_diff_all}"
        );
    }

    #[test]
    fn test_best_iteration_survives_save_load() {
        let (model, _) = train_overfit_model();
        let best = model.best_iteration.expect("best_iteration should be Some");

        let (tf, _, _, _) = overfit_dataset();
        let pre = model.predict_raw(&tf.view()).unwrap();

        let mut buf: Vec<u8> = Vec::new();
        model.save_to_writer(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let loaded = XGBoostModel::load_from_reader(&mut cursor).unwrap();

        assert_eq!(
            loaded.best_iteration,
            Some(best),
            "best_iteration must round-trip through save/load"
        );

        let post = loaded.predict_raw(&tf.view()).unwrap();
        let max_diff = (&pre - &post).iter().fold(0f64, |m, &x| m.max(x.abs()));
        assert!(
            max_diff < 1e-5,
            "post-load predictions must match pre-load (max diff {max_diff})"
        );
    }

    #[test]
    fn test_no_early_stopping_predicts_all_trees() {
        // With early_stopping_rounds = None, best_iteration must be None so
        // predict_raw uses all trees (unchanged legacy behavior).
        let (tf, tl, vf, vl) = overfit_dataset();
        let mut train = XGBoostDataset::from_data(tf.view(), tl.view()).unwrap();
        let mut valid = XGBoostDataset::from_data(vf.view(), vl.view()).unwrap();

        let mut params = XGBoostParams::default();
        params.set_n_dist_params(1);

        let config = TrainConfig {
            num_boost_round: 15,
            early_stopping_rounds: None,
            verbose: false,
            seed: 7,
            collect_train_metrics: false,
        };

        let (model, _) = XGBoostModel::train_with_objective_and_callbacks::<
            _,
            _,
            super::super::traits::HistoryCallback,
        >(
            &params,
            &mut train,
            Some(&mut valid),
            &config,
            sq_objective,
            sq_metric,
            None,
            None,
        )
        .unwrap();

        assert!(model.best_iteration.is_none());

        // predict_raw should equal the plain all-trees predict().
        let feats_f32: Vec<f32> = tf.iter().map(|&x| x as f32).collect();
        let dmat = DMatrix::from_dense(&feats_f32, tf.nrows()).unwrap();
        let all_trees = prediction_to_array2(&model.booster.predict(&dmat).unwrap(), tf.nrows(), 1);
        let pr = model.predict_raw(&tf.view()).unwrap();
        let max_diff = (&pr - &all_trees).iter().fold(0f64, |m, &x| m.max(x.abs()));
        assert!(max_diff < 1e-6, "no-early-stopping must predict all trees");
    }
}
