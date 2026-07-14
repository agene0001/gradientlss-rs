//! LightGBM backend implementation using the `lgbm` crate.
//!
//! This backend supports custom objective functions via `update_one_iter_custom`,
//! enabling full distributional regression with custom gradients and hessians.
//! LightGBM natively supports multi-output, so we use a single booster with num_class.

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
use std::sync::Arc;

use lgbm::{Booster, Dataset, Field, Mat, MatBuf, Parameters, PredictType, Prediction};

/// LightGBM backend for GradientLSS.
#[derive(Debug, Clone)]
pub struct LightGBMBackend;

/// LightGBM-specific parameters.
#[derive(Debug, Clone)]
pub struct LightGBMParams {
    inner: Parameters,
    /// Typed shadow of `inner` so `BackendParams::get`/`to_map` work.
    typed: HashMap<String, ParamValue>,
    n_dist_params: usize,
}

impl Default for LightGBMParams {
    fn default() -> Self {
        let mut params = Self {
            inner: Parameters::new(),
            typed: HashMap::new(),
            n_dist_params: 1,
        };
        params.set("boosting", ParamValue::from("gbdt"));
        params.set("learning_rate", ParamValue::Float(0.1));
        params.set("num_leaves", ParamValue::Int(31));
        params.set("verbose", ParamValue::Int(-1));
        // Use "none" objective since we provide custom gradients
        params.set("objective", ParamValue::from("none"));
        params
    }
}

impl BackendParams for LightGBMParams {
    fn set(&mut self, key: &str, value: ParamValue) {
        let str_value = match &value {
            ParamValue::Int(v) => v.to_string(),
            ParamValue::Float(v) => v.to_string(),
            ParamValue::String(v) => v.clone(),
            ParamValue::Bool(v) => if *v { "true" } else { "false" }.to_string(),
        };
        // REPLACE in place, never push a duplicate: LightGBM's config parser
        // keeps the FIRST occurrence of a duplicated key and only logs a
        // warning — which the default `verbose=-1` (parsed first) suppresses.
        // Appending meant every override of a default (learning_rate,
        // num_leaves, ... including all hyper_opt-tuned values) was silently
        // discarded.
        self.inner.0.retain(|(k, _)| k != key);
        self.inner.push(key, str_value);
        self.typed.insert(key.to_string(), value);
    }

    fn get(&self, key: &str) -> Option<&ParamValue> {
        self.typed.get(key)
    }

    fn to_map(&self) -> HashMap<String, ParamValue> {
        self.typed.clone()
    }
}

impl LightGBMParams {
    /// Get a reference to the inner Parameters.
    pub fn inner(&self) -> &Parameters {
        &self.inner
    }

    /// Whether any of LightGBM's seed keys was set explicitly.
    fn has_seed_param(&self) -> bool {
        self.inner.0.iter().any(|(k, _)| {
            matches!(
                k.as_str(),
                "seed"
                    | "random_seed"
                    | "random_state"
                    | "data_random_seed"
                    | "bagging_seed"
                    | "feature_fraction_seed"
                    | "extra_seed"
                    | "drop_seed"
                    | "objective_seed"
            )
        })
    }

    /// Set the number of distribution parameters.
    pub fn set_n_dist_params(&mut self, n: usize) {
        self.n_dist_params = n;
        // Set num_class for multi-output
        self.set("num_class", ParamValue::Int(n as i64));
    }

    /// Get the number of distribution parameters.
    pub fn n_dist_params(&self) -> usize {
        self.n_dist_params
    }
}

/// LightGBM dataset wrapper.
///
/// Holds the raw features/labels; the binned `lgbm::Dataset` is built lazily at
/// TRAIN time (see [`Self::build_dataset`]) so it can see the full training
/// params — mirroring Python, where `lgb.train` constructs the lazy
/// `lgb.Dataset` with the training params. Dataset-level params (`max_bin`,
/// `feature_pre_filter`, `min_data_in_leaf`, `categorical_feature`,
/// `zero_as_missing`, ...) shape the bin mappers, and LightGBM cannot change
/// them after construction — building eagerly with empty params silently
/// pinned them all to their defaults.
pub struct LightGBMDataset {
    n_rows: usize,
    n_cols: usize,
    labels: Array1<f64>,
    /// Row-major f32 features; used to build the binned Dataset at train time
    /// and the Mats for mid-training predictions.
    features: Vec<f32>,
    /// Per-sample training weights, applied to grad/hess in the objective.
    weights: Option<Array1<f64>>,
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

        Ok(Self {
            n_rows,
            n_cols,
            labels: labels.to_owned(),
            features: features_f32,
            weights: None,
        })
    }

    fn set_init_score(&mut self, _init_score: &Array1<f64>) -> Result<()> {
        // Intentional no-op: the LightGBM training loop seeds the accumulated
        // train/valid margins with the start values and `predict` adds them
        // back after `predict_raw` — together exactly Python's Dataset
        // init_score + predict-time add-back, with nothing to set here.
        Ok(())
    }

    fn num_rows(&self) -> usize {
        self.n_rows
    }

    fn get_labels(&self) -> Result<Array1<f64>> {
        Ok(self.labels.clone())
    }

    /// Store per-sample weights. They are plumbed into the binned
    /// `lgbm::Dataset` (`Field::<f32>::WEIGHT`) when `build_dataset` runs at
    /// train time; the training loop additionally reads them back and hands
    /// them to the objective, which scales grad/hess per-sample — LightGBM does
    /// not apply dataset weights to custom-objective gradients (matching
    /// LightGBMLSS's `grad *= weights; hess *= weights`).
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

    fn get_weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
    }
}

impl LightGBMDataset {
    /// Build the binned `lgbm::Dataset` with the given (training) params —
    /// the bin mappers must see the same params as the booster, exactly like
    /// `lgb.train`'s lazy `lgb.Dataset` construction in Python. Called once per
    /// training run, for the training data only (validation data is predicted
    /// through Mats and never needs binning).
    fn build_dataset(&self, params: &Parameters) -> Result<Dataset> {
        let mat = Mat::from_slice(
            &self.features,
            self.n_rows,
            self.n_cols,
            lgbm::mat::RowMajor,
        );
        let mut dataset = Dataset::from_mat(&mat, None, params).map_err(|e| {
            GradientLSSError::BackendError(format!("Failed to create Dataset: {}", e))
        })?;

        // For multivariate distributions we have n_targets labels per sample,
        // but LightGBM expects one label per sample — give it dummy zeros (the
        // custom objective reads the real labels from `self.labels`).
        let dummy_labels: Vec<f32> = if self.labels.len() == self.n_rows {
            self.labels.iter().map(|&x| x as f32).collect()
        } else {
            vec![0.0; self.n_rows]
        };
        dataset
            .set_field(Field::<f32>::LABEL, &dummy_labels)
            .map_err(|e| GradientLSSError::BackendError(format!("Failed to set labels: {}", e)))?;

        if let Some(ref w) = self.weights {
            let weights_f32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
            dataset
                .set_field(Field::<f32>::WEIGHT, &weights_f32)
                .map_err(|e| {
                    GradientLSSError::BackendError(format!("Failed to set weights: {}", e))
                })?;
        }

        Ok(dataset)
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
    /// Best iteration found by early stopping, if early stopping was enabled.
    /// LightGBM's Python `Booster.predict` defaults `num_iteration` to
    /// `best_iteration` when it exists, so LightGBMLSS predicts with the best
    /// model rather than all built trees (best + patience extras). We mirror
    /// that in `predict_raw`.
    best_iteration: Option<usize>,
}

impl std::fmt::Debug for LightGBMModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LightGBMModel")
            .field("n_params", &self.n_params)
            .field("n_features", &self.n_features)
            .field("best_iteration", &self.best_iteration)
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

        // Per-sample weights (if any). LightGBM does not apply dataset weights to
        // custom-objective grad/hess, so we read them here and hand them to the
        // objective, which scales grad/hess per-sample — matching LightGBMLSS.
        let train_weights = train_data.get_weights().cloned();

        // Full booster params: user params plus TrainConfig::seed unless a seed
        // was set explicitly (Python forces random_seed=123; we honor the
        // config field instead, and an explicit user seed always wins).
        let mut full_params = params.inner().clone();
        if !params.has_seed_param() {
            full_params.push("seed", config.seed.to_string());
        }

        // Build the binned Dataset with the SAME params as the booster (see
        // `build_dataset`), then the booster on top of it.
        let train_dataset = Arc::new(train_data.build_dataset(&full_params)?);
        let mut booster = Booster::new(Arc::clone(&train_dataset), &full_params).map_err(|e| {
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

        // Running validation margin, accumulated one iteration at a time (mirrors
        // `predictions`). Seeded with start_values; zero-sized when there's no
        // validation set.
        let mut valid_predictions = Array2::zeros((valid_n_samples, n_params));
        if let Some(sv) = start_values {
            for i in 0..valid_n_samples {
                for j in 0..n_params {
                    valid_predictions[[i, j]] = sv[j];
                }
            }
        }

        // Whether the per-round train metric is needed at all: it drives early
        // stopping when there is no validation set, feeds verbose logging and
        // callbacks, and is recorded when `collect_train_metrics` asks for it.
        // When none of those apply, skipping it saves a full NLL pass over the
        // training set every round.
        let need_train_metric = config.collect_train_metrics
            || config.verbose
            || callbacks.is_some()
            || valid_mat.is_none();

        let mut best_loss = f64::INFINITY;
        let mut best_iteration = 0usize;
        // Last loss that beat its predecessor by the relative min-delta; drives
        // ONLY the patience counter, never best_iteration/best_loss.
        let mut significant_best_loss = f64::INFINITY;
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
            let gh = objective_fn(&predictions, &labels, train_weights.as_ref())?;

            // Flatten gradients and hessians in Fortran order (column-major):
            // lgbm expects [grad_param0_sample0, grad_param0_sample1, ..., grad_param1_sample0, ...].
            // Transpose and cast to f32 in a single pass — the previous
            // `reshape_gradients` + `map` route allocated two intermediate f64
            // arrays per round.
            let mut grad_f32 = vec![0f32; n_samples * n_params];
            let mut hess_f32 = vec![0f32; n_samples * n_params];
            for (i, (g_row, h_row)) in gh
                .gradients
                .rows()
                .into_iter()
                .zip(gh.hessians.rows())
                .enumerate()
            {
                for j in 0..n_params {
                    grad_f32[j * n_samples + i] = g_row[j] as f32;
                    hess_f32[j * n_samples + i] = h_row[j] as f32;
                }
            }

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

            // Predict ONLY the iteration just added and accumulate into the running
            // margin. RawScore is additive across trees, so summing each iteration's
            // contribution equals re-predicting all trees — but at O(n) per round
            // instead of O(n * trees_so_far), turning the loop from O(rounds^2) to
            // O(rounds). This matters most for multi-output (num_class>1, e.g. the
            // NegativeBinomial), where the full re-predict over 2x the trees was the
            // dominant cost. `predictions` already carries start_values from init, so
            // we no longer re-add them each round.
            let delta = booster
                .predict_for_mat(
                    &train_mat,
                    PredictType::RawScore,
                    round,
                    Some(1),
                    &pred_params,
                )
                .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;
            let delta = prediction_to_array2(&delta, n_samples, n_params)?;
            predictions += &delta;

            // Compute training loss (only when something consumes it)
            let train_loss = if need_train_metric {
                let loss = metric_fn(&predictions, &labels);
                train_history.push(loss);
                Some(loss)
            } else {
                None
            };

            // Compute validation loss if available — same per-iteration accumulation
            // into the running `valid_predictions` margin (carries start_values).
            let valid_loss = if let (Some(vm), Some(vl)) = (&valid_mat, &valid_labels) {
                let vdelta = booster
                    .predict_for_mat(vm, PredictType::RawScore, round, Some(1), &pred_params)
                    .map_err(|e| {
                        GradientLSSError::BackendError(format!("Valid prediction failed: {}", e))
                    })?;
                let vdelta = prediction_to_array2(&vdelta, valid_n_samples, n_params)?;
                valid_predictions += &vdelta;

                let vl_loss = metric_fn(&valid_predictions, vl);
                valid_history.push(vl_loss);
                Some(vl_loss)
            } else {
                None
            };

            // Determine the evaluation loss for early stopping
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

            // Built-in early stopping. The patience counter only counts a round
            // as progress if it beats the last significant best by a relative
            // margin (see `is_significant_improvement`). Without the min-delta,
            // a noisy sub-1e-4 NLL tick reset the patience counter every round,
            // so NegBinom fits rode `num_boost_round` to the end instead of
            // early-stopping.
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

        // Notify callbacks of training end
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
        // AND `early_stopping_rounds` configured (matching the XGBoost backend).
        // Without a validation set the monitored loss is the train loss, which
        // keeps descending — truncating on it would discard useful trees.
        let model_best_iteration = if valid_mat.is_some() && config.early_stopping_rounds.is_some()
        {
            Some(best_iteration)
        } else {
            None
        };

        Ok((
            Self {
                booster,
                n_params,
                n_features,
                best_iteration: model_best_iteration,
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

        // Predict raw scores. When early stopping recorded a best iteration,
        // use trees [0, best_iteration]; LightGBM's Python predict does the
        // same by defaulting num_iteration to best_iteration.
        let pred_params = Parameters::new();
        let num_iteration = self.best_iteration.map(|b| b + 1);
        let prediction = self
            .booster
            .predict_for_mat(&mat, PredictType::RawScore, 0, num_iteration, &pred_params)
            .map_err(|e| GradientLSSError::BackendError(format!("Prediction failed: {}", e)))?;

        prediction_to_array2(&prediction, n_samples, self.n_params)
    }

    fn save_to_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Persist best_iteration first (u64::MAX = none): LightGBM's model text
        // format does not record it, but prediction parity requires it.
        let best_iter_tag: u64 = self.best_iteration.map_or(u64::MAX, |b| b as u64);
        writer
            .write_all(&best_iter_tag.to_le_bytes())
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;

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
        let mut best_iter_bytes = [0u8; 8];
        reader
            .read_exact(&mut best_iter_bytes)
            .map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        let best_iter_tag = u64::from_le_bytes(best_iter_bytes);
        let best_iteration = if best_iter_tag == u64::MAX {
            None
        } else {
            Some(best_iter_tag as usize)
        };

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
            best_iteration,
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

    /// LightGBM's config parser keeps the FIRST occurrence of a duplicated key
    /// (warning suppressed by verbose=-1), so `set` must REPLACE the default
    /// in place — a pushed duplicate would be silently ignored, discarding
    /// every user/hyper_opt override of a default param.
    #[test]
    fn test_params_set_overrides_defaults_in_place() {
        let mut params = LightGBMParams::default();
        params.set("learning_rate", ParamValue::Float(0.9));
        params.set("learning_rate", ParamValue::Float(0.5));

        let occurrences: Vec<String> = params
            .inner()
            .0
            .iter()
            .filter(|(k, _)| k == "learning_rate")
            .map(|(_, v)| v.to_string())
            .collect();
        assert_eq!(
            occurrences,
            vec!["0.5".to_string()],
            "set must leave exactly one occurrence holding the latest value"
        );
        assert!(
            matches!(params.get("learning_rate"), Some(ParamValue::Float(v)) if *v == 0.5),
            "typed get must reflect the latest value"
        );

        // set_n_dist_params re-invocations must not duplicate num_class either.
        params.set_n_dist_params(2);
        params.set_n_dist_params(3);
        let num_class: Vec<String> = params
            .inner()
            .0
            .iter()
            .filter(|(k, _)| k == "num_class")
            .map(|(_, v)| v.to_string())
            .collect();
        assert_eq!(num_class, vec!["3".to_string()]);
    }

    /// The training loop's speed fix replaces a full re-predict each round with
    /// summing per-iteration RawScore contributions. This verifies the invariant
    /// it relies on: `Σ_r predict(start=r, num=1) == predict(start=0, num=all)`,
    /// for a multi-output (`num_class=2`) booster — i.e. the NegativeBinomial path.
    #[test]
    fn test_per_iteration_accumulation_matches_full_predict() {
        let n = 60usize;
        let f = 5usize;
        let num_class = 2usize;

        // Deterministic dummy features + labels.
        let mut feats = Array2::<f64>::zeros((n, f));
        for i in 0..n {
            for j in 0..f {
                feats[[i, j]] = (((i * 7 + j * 11) % 17) as f64) / 17.0;
            }
        }
        let labels = Array1::from_vec((0..n).map(|i| (i % 4) as f64).collect());

        let ds = LightGBMDataset::from_data(feats.view(), labels.view()).expect("dataset");
        let mut params = LightGBMParams::default();
        params.set_n_dist_params(num_class);
        let dataset = Arc::new(ds.build_dataset(params.inner()).expect("binned dataset"));
        let mut booster = Booster::new(dataset, params.inner()).expect("booster");

        // A few rounds with per-sample-varying custom gradients (column-major by
        // class) so the trees actually split — constant gradients build trivial trees.
        let rounds = 6usize;
        for _ in 0..rounds {
            let mut grad = vec![0f32; n * num_class];
            let hess = vec![1f32; n * num_class];
            for c in 0..num_class {
                for i in 0..n {
                    grad[c * n + i] = (feats[[i, c % f]] - 0.5) as f32;
                }
            }
            booster
                .update_one_iter_custom(&grad, &hess)
                .expect("update_one_iter_custom");
        }

        let mat = ds.to_mat();
        let pp = Parameters::new();

        let full = booster
            .predict_for_mat(&mat, PredictType::RawScore, 0, None, &pp)
            .expect("full predict");
        let full = prediction_to_array2(&full, n, num_class).expect("full arr");

        let mut acc = Array2::<f64>::zeros((n, num_class));
        for r in 0..rounds {
            let d = booster
                .predict_for_mat(&mat, PredictType::RawScore, r, Some(1), &pp)
                .expect("per-iteration predict");
            acc += &prediction_to_array2(&d, n, num_class).expect("iter arr");
        }

        let max_diff = (&acc - &full).iter().fold(0f64, |m, &x| m.max(x.abs()));
        assert!(
            max_diff < 1e-4,
            "per-iteration accumulation diverged from full predict (max abs diff {max_diff})"
        );
    }

    // ---- best_iteration / early-stopping truncation tests ----

    /// Noisy single-parameter regression where the model provably overfits:
    /// validation loss bottoms out early, later trees fit training noise.
    fn overfit_dataset() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
        let f = 8usize;
        let n_train = 300usize;
        let n_valid = 150usize;

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
                        signal += x;
                    }
                }
                let noise = (next() - 0.5) * 4.0;
                labels[i] = signal + noise;
            }
            (feats, labels)
        };

        let (tf, tl) = make(n_train, &mut next);
        let (vf, vl) = make(n_valid, &mut next);
        (tf, tl, vf, vl)
    }

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

    fn train_overfit_model() -> (LightGBMModel, TrainingResult) {
        let (tf, tl, vf, vl) = overfit_dataset();
        let mut train = LightGBMDataset::from_data(tf.view(), tl.view()).unwrap();
        let mut valid = LightGBMDataset::from_data(vf.view(), vl.view()).unwrap();

        let mut params = LightGBMParams::default();
        params.set_n_dist_params(1);
        params.set("learning_rate", ParamValue::Float(0.3));
        params.set("num_leaves", ParamValue::Int(63));
        params.set("min_data_in_leaf", ParamValue::Int(1));

        let config = TrainConfig {
            num_boost_round: 80,
            early_stopping_rounds: Some(5),
            verbose: false,
            seed: 7,
            collect_train_metrics: false,
        };

        LightGBMModel::train_with_objective_and_callbacks::<
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

        let best = model.best_iteration.expect("best_iteration should be Some");
        assert!(
            best + 1 < result.n_iterations,
            "expected best_iteration ({}) + 1 < total trees ({})",
            best,
            result.n_iterations
        );

        let (tf, _, _, _) = overfit_dataset();
        let feats_f32: Vec<f32> = tf.iter().map(|&x| x as f32).collect();
        let mat = MatBuf::from_vec(feats_f32, tf.nrows(), tf.ncols(), lgbm::mat::RowMajor);
        let pp = Parameters::new();

        // Reference bounded to best+1 iterations from the start.
        let bounded = model
            .booster
            .predict_for_mat(&mat, PredictType::RawScore, 0, Some(best + 1), &pp)
            .unwrap();
        let bounded = prediction_to_array2(&bounded, tf.nrows(), 1).unwrap();

        // All-trees prediction (what the buggy predict_raw did).
        let all_trees = model
            .booster
            .predict_for_mat(&mat, PredictType::RawScore, 0, None, &pp)
            .unwrap();
        let all_trees = prediction_to_array2(&all_trees, tf.nrows(), 1).unwrap();

        let pr = model.predict_raw(&tf.view()).unwrap();
        let max_diff_bounded = (&pr - &bounded).iter().fold(0f64, |m, &x| m.max(x.abs()));
        assert!(
            max_diff_bounded < 1e-5,
            "predict_raw must match best_iteration-bounded prediction (max diff {max_diff_bounded})"
        );

        let max_diff_all = (&pr - &all_trees).iter().fold(0f64, |m, &x| m.max(x.abs()));
        assert!(
            max_diff_all > 1e-6,
            "predict_raw should differ from all-trees prediction (max diff {max_diff_all})"
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
        let loaded = LightGBMModel::load_from_reader(&mut cursor).unwrap();

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
        let (tf, tl, vf, vl) = overfit_dataset();
        let mut train = LightGBMDataset::from_data(tf.view(), tl.view()).unwrap();
        let mut valid = LightGBMDataset::from_data(vf.view(), vl.view()).unwrap();

        let mut params = LightGBMParams::default();
        params.set_n_dist_params(1);

        let config = TrainConfig {
            num_boost_round: 15,
            early_stopping_rounds: None,
            verbose: false,
            seed: 7,
            collect_train_metrics: false,
        };

        let (model, _) = LightGBMModel::train_with_objective_and_callbacks::<
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

        let feats_f32: Vec<f32> = tf.iter().map(|&x| x as f32).collect();
        let mat = MatBuf::from_vec(feats_f32, tf.nrows(), tf.ncols(), lgbm::mat::RowMajor);
        let pp = Parameters::new();
        let all_trees = model
            .booster
            .predict_for_mat(&mat, PredictType::RawScore, 0, None, &pp)
            .unwrap();
        let all_trees = prediction_to_array2(&all_trees, tf.nrows(), 1).unwrap();
        let pr = model.predict_raw(&tf.view()).unwrap();
        let max_diff = (&pr - &all_trees).iter().fold(0f64, |m, &x| m.max(x.abs()));
        assert!(max_diff < 1e-6, "no-early-stopping must predict all trees");
    }

    /// Manual benchmark proving the LightGBM per-iteration fix is a real speedup
    /// (no XGBoost-style regression):
    /// `cargo test --release --features full --lib lgb_predict_strategy_timing -- --ignored --nocapture`
    ///
    /// Strategy A is the OLD loop — full `predict_for_mat(start=0, num=None)` each
    /// round (re-walks every tree). Strategy B is the shipped fix —
    /// `predict_for_mat(start=r, num=Some(1))` (the round's tree only). Unlike
    /// XGBoost's cached `predict()`, `predict_for_mat` runs from scratch on the
    /// passed `Mat`, so A grows O(rounds) per call (O(rounds²) total) while B stays
    /// O(1). If B ≪ A, the fix helps and cannot regress.
    #[test]
    #[ignore]
    fn lgb_predict_strategy_timing() {
        use std::time::{Duration, Instant};

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
        let mat = Mat::from_slice(&feats, n, f, lgbm::mat::RowMajor);

        let build_booster = || {
            let dparams = Parameters::new();
            let mut dataset = Dataset::from_mat(&mat, None, &dparams).unwrap();
            dataset.set_field(Field::<f32>::LABEL, &labels).unwrap();
            let mut bparams = Parameters::new();
            bparams.push("objective", "regression");
            bparams.push("boosting", "gbdt");
            bparams.push("num_leaves", "31");
            bparams.push("learning_rate", "0.1");
            bparams.push("verbose", "-1");
            Booster::new(Arc::new(dataset), &bparams).unwrap()
        };
        let pred_params = Parameters::new();

        // Strategy A: full re-predict each round.
        let mut b_a = build_booster();
        let mut predict_a = Duration::ZERO;
        let total_a = Instant::now();
        for _ in 0..rounds {
            b_a.update_one_iter().unwrap();
            let t = Instant::now();
            let _ = b_a
                .predict_for_mat(&mat, PredictType::RawScore, 0, None, &pred_params)
                .unwrap();
            predict_a += t.elapsed();
        }
        let total_a = total_a.elapsed();

        // Strategy B: per-iteration predict (the shipped fix).
        let mut b_b = build_booster();
        let mut predict_b = Duration::ZERO;
        let total_b = Instant::now();
        for r in 0..rounds {
            b_b.update_one_iter().unwrap();
            let t = Instant::now();
            let _ = b_b
                .predict_for_mat(&mat, PredictType::RawScore, r, Some(1), &pred_params)
                .unwrap();
            predict_b += t.elapsed();
        }
        let total_b = total_b.elapsed();

        eprintln!("LGB {rounds} rounds, {n}x{f}:");
        eprintln!("  A full predict_for_mat(0,None):  predict={predict_a:?}  total={total_a:?}");
        eprintln!("  B predict_for_mat(r,1) [fix]:    predict={predict_b:?}  total={total_b:?}");
        eprintln!(
            "  predict-time ratio A/B = {:.1}x (>1 means the fix is faster)",
            predict_a.as_secs_f64() / predict_b.as_secs_f64().max(1e-9)
        );
    }
}
