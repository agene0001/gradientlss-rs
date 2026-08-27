//! Hyperparameter optimization for GradientLSS.
//!
//! Provides Tree-structured Parzen Estimator (TPE) based optimization
//! for finding optimal hyperparameters with optional trial pruning.

use crate::backend::{Backend, BackendParams, ParamValue, TrainConfig};
use crate::model::GradientLSS;
use ndarray::{Array1, Array2};
// tpe 0.3 uses the same rand as the rest of the crate (the old rand_compat
// shim was only needed for tpe 0.2's rand 0.8).
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::Value;
use std::collections::HashMap;
use tpe::{TpeOptimizer, categorical_range, histogram_estimator, parzen_estimator, range};

/// Hyperparameter specification for optimization.
#[derive(Debug, Clone)]
pub struct HyperParamSpec {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: HyperParamType,
}

/// Types of hyperparameters that can be optimized.
#[derive(Debug, Clone)]
pub enum HyperParamType {
    /// Continuous float parameter with bounds [low, high]
    Float { low: f64, high: f64, log: bool },
    /// Integer parameter with bounds [low, high]
    Int { low: i64, high: i64, log: bool },
    /// Categorical parameter with choices
    Categorical { choices: Vec<Value> },
}

impl HyperParamSpec {
    /// Create a float hyperparameter specification.
    pub fn float(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            param_type: HyperParamType::Float {
                low,
                high,
                log: false,
            },
        }
    }

    /// Create a log-scale float hyperparameter specification.
    pub fn log_float(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            param_type: HyperParamType::Float {
                low,
                high,
                log: true,
            },
        }
    }

    /// Create an integer hyperparameter specification.
    pub fn int(name: impl Into<String>, low: i64, high: i64) -> Self {
        Self {
            name: name.into(),
            param_type: HyperParamType::Int {
                low,
                high,
                log: false,
            },
        }
    }

    /// Create a categorical hyperparameter specification.
    pub fn categorical(name: impl Into<String>, choices: Vec<Value>) -> Self {
        Self {
            name: name.into(),
            param_type: HyperParamType::Categorical { choices },
        }
    }
}

/// Result of hyperparameter optimization.
#[derive(Debug, Clone)]
pub struct HyperOptResult {
    /// Best hyperparameters found
    pub best_params: HashMap<String, Value>,
    /// Best score (loss) achieved
    pub best_score: f64,
    /// Optimal number of boosting rounds from the best trial
    pub opt_rounds: usize,
    /// History of all trials
    pub trials: Vec<TrialResult>,
}

/// Result of a single trial.
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Parameters used in this trial
    pub params: HashMap<String, Value>,
    /// Score achieved
    pub score: f64,
    /// Whether the trial was pruned
    pub pruned: bool,
    /// Intermediate scores (for pruning history)
    pub intermediate_scores: Vec<f64>,
}

/// Pruning strategy for early termination of unpromising trials.
#[derive(Debug, Clone)]
pub enum PruningStrategy {
    /// No pruning - run all trials to completion
    None,
    /// Median pruning: prune if intermediate score is worse than median of completed trials
    /// at the same step. The parameter is the number of warmup trials before pruning starts.
    Median { n_warmup_trials: usize },
    /// Percentile pruning: prune if intermediate score is worse than given percentile.
    /// For example, percentile=25 keeps only top 25% of trials at each step.
    Percentile {
        percentile: f64,
        n_warmup_trials: usize,
    },
    /// Threshold pruning: prune if intermediate score exceeds threshold
    Threshold { threshold: f64 },
}

impl Default for PruningStrategy {
    fn default() -> Self {
        PruningStrategy::Median { n_warmup_trials: 5 }
    }
}

/// Pruner state for tracking intermediate results across trials.
#[derive(Debug, Clone, Default)]
struct PrunerState {
    /// Intermediate scores for each completed trial at each step
    /// Structure: step -> list of scores at that step
    intermediate_scores: HashMap<usize, Vec<f64>>,
}

impl PrunerState {
    fn new() -> Self {
        Self {
            intermediate_scores: HashMap::new(),
        }
    }

    /// Record an intermediate score for tracking
    fn record(&mut self, step: usize, score: f64) {
        self.intermediate_scores
            .entry(step)
            .or_insert_with(Vec::new)
            .push(score);
    }

    /// Check if a trial should be pruned based on the strategy
    fn should_prune(
        &self,
        strategy: &PruningStrategy,
        step: usize,
        score: f64,
        n_completed_trials: usize,
    ) -> bool {
        match strategy {
            PruningStrategy::None => false,
            PruningStrategy::Median { n_warmup_trials } => {
                if n_completed_trials < *n_warmup_trials {
                    return false;
                }
                if let Some(scores) = self.intermediate_scores.get(&step) {
                    if scores.is_empty() {
                        return false;
                    }
                    let mut sorted = scores.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = sorted[sorted.len() / 2];
                    score > median
                } else {
                    false
                }
            }
            PruningStrategy::Percentile {
                percentile,
                n_warmup_trials,
            } => {
                if n_completed_trials < *n_warmup_trials {
                    return false;
                }
                if let Some(scores) = self.intermediate_scores.get(&step) {
                    if scores.is_empty() {
                        return false;
                    }
                    let mut sorted = scores.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let idx = ((sorted.len() as f64 * percentile / 100.0).ceil() as usize)
                        .saturating_sub(1)
                        .min(sorted.len() - 1);
                    let threshold = sorted[idx];
                    score > threshold
                } else {
                    false
                }
            }
            PruningStrategy::Threshold { threshold } => score > *threshold,
        }
    }
}

/// How cross-validation folds are partitioned during the search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CvScheme {
    /// Standard k-fold: each fold trains on all other rows. Assumes rows are
    /// exchangeable (IID) — the correct default for general tabular data.
    #[default]
    KFold,
    /// Forward-chaining (expanding window) for time-ordered data: fold `i` trains
    /// only on rows strictly before its test segment, so no future row informs a
    /// prediction of the past. Use when row order encodes time (e.g. time series).
    TimeSeries,
}

/// A data-dependent transform fitted INSIDE each CV fold.
///
/// Preprocessing that learns from the data — imputation being the motivating
/// case — must be fitted on the fold's TRAIN rows only. Fitting it once on the
/// whole matrix before folding lets the fitted state encode the very rows each
/// fold then scores against, which quietly flatters every CV number and so
/// biases model selection.
///
/// Implementors receive the fold's train/valid feature blocks and the trial's
/// sampled parameters, so the transform's own hyperparameters (matrix-imputer
/// `rank`, KNN `k`, …) can be tuned in the same search as the model's. Called
/// once per (trial, fold); keep it cheap.
///
/// Labels are deliberately not passed: a transform that reads y is a target
/// leak waiting to happen.
pub trait FoldTransform: Sync {
    /// Fit on `train_x` and apply to both blocks, in place.
    ///
    /// `trial_params` holds every value the study sampled for this trial,
    /// including any the transform declared. Returning `Err` marks the fold
    /// unscorable (recorded as infinite loss) rather than aborting the study.
    /// `train_idx` / `valid_idx` are the rows' positions in the ORIGINAL
    /// feature matrix, in the same order as the blocks handed over. Transforms
    /// that key off per-row identity — a per-player median, a forward fill
    /// within a player's chronological games — need them to recover which
    /// entity each row belongs to; purely numeric transforms can ignore them.
    fn fit_transform(
        &self,
        train_x: &mut Array2<f64>,
        valid_x: &mut Array2<f64>,
        train_idx: &[usize],
        valid_idx: &[usize],
        trial_params: &HashMap<String, Value>,
    ) -> Result<(), String>;
}

/// Search-space keys with this prefix are handed to the [`FoldTransform`] only
/// and are NOT forwarded to the boosting backend. Use it for anything that
/// parameterizes preprocessing (imputer rank, neighbour count, …) so a single
/// TPE study can tune the pipeline and the model together.
pub const TRANSFORM_PARAM_PREFIX: &str = "__";

/// Configuration for hyperparameter optimization.
#[derive(Debug, Clone)]
pub struct HyperOptConfig {
    /// Number of optimization trials
    pub n_trials: u32,
    /// Number of cross-validation folds
    pub n_folds: usize,
    /// How CV folds are partitioned (`KFold` default; `TimeSeries` for forward
    /// chaining on time-ordered rows).
    pub cv_scheme: CvScheme,
    /// Random seed for CV fold generation
    pub seed: u64,
    /// Separate seed for hyperparameter sampling (if None, uses seed)
    pub hp_seed: Option<u64>,
    /// Pruning strategy for early termination
    pub pruning: PruningStrategy,
    /// Number of boosting rounds per trial
    pub num_boost_round: usize,
    /// Early stopping rounds within each trial's CV
    pub early_stopping_rounds: Option<usize>,
    /// Maximum time budget in minutes (None = no limit)
    pub max_minutes: Option<f64>,
    /// Stop the study after this many CONSECUTIVE COMPLETED trials without a
    /// new best. `None` (the default) runs every trial, as before.
    ///
    /// This is study-level early stopping, and it is distinct from
    /// [`PruningStrategy`]: pruning kills an individual BAD trial partway
    /// through its folds, but never ends the search — so a study that found
    /// its optimum at trial 10 still pays for the remaining 90.
    ///
    /// Measured on the consumer's 2026-08-26 sweep: searches routinely hit
    /// their best score in the first 10-20 trials and then spent the rest of
    /// the budget improving by <0.11%. Whether those late gains survive to
    /// held-out data is a separate (open) question, so the recommended
    /// setting is generous — large enough to keep real late improvements,
    /// small enough to cut a search that has plainly converged.
    ///
    /// Only COMPLETED trials count: a pruned trial was killed precisely
    /// because it was bad, and letting those advance the counter would end a
    /// search that is still exploring productively.
    pub plateau_patience: Option<u32>,
    /// A previous search's winning params + score, replayed into the TPE
    /// optimizers as an observation BEFORE trial 1.
    ///
    /// A sweep that runs many searches over the same data with only a
    /// preprocessing or scoring change between them re-derives the same
    /// hyperparameters from scratch every time. Seeding the optimizers with a
    /// sibling search's answer biases the early trials toward a region already
    /// known to work, without constraining the search: TPE still explores, so
    /// a run that genuinely wants different params still finds them.
    ///
    /// This is deliberately NOT a way to skip searching. The prior is told to
    /// the optimizers only; it never seeds `best_params`, so it cannot be
    /// returned as a result without being re-drawn and re-scored on this run's
    /// own folds. Combined with [`HyperOptConfig::plateau_patience`], a search
    /// whose prior still wins exits in a handful of trials instead of all of
    /// them.
    ///
    /// Keys name params in the same `hp_dict` namespace; unknown keys are
    /// ignored, and values outside a spec's range are clamped into it. `None`
    /// (the default) searches cold.
    pub warm_start: Option<(HashMap<String, Value>, f64)>,
    /// Whether to print progress
    pub verbose: bool,
}

impl Default for HyperOptConfig {
    fn default() -> Self {
        Self {
            n_trials: 100,
            n_folds: 5,
            cv_scheme: CvScheme::KFold,
            seed: 42,
            hp_seed: None,
            pruning: PruningStrategy::None,
            num_boost_round: 100,
            early_stopping_rounds: Some(10),
            max_minutes: None,
            plateau_patience: None,
            warm_start: None,
            verbose: true,
        }
    }
}

/// Perform hyperparameter optimization using TPE.
///
/// # Arguments
/// * `model` - The GradientLSS model to optimize
/// * `features` - Training features
/// * `labels` - Training labels
/// * `hp_dict` - Hyperparameter search space as JSON values
///   - For numeric params: `[low, high]` or `{"low": x, "high": y, "log": true/false}`
///   - For categorical params: `{"choices": [val1, val2, ...]}`
/// * `n_trials` - Number of optimization trials
/// * `n_folds` - Number of cross-validation folds
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// The best hyperparameters found during optimization.
pub fn hyper_opt<B: Backend>(
    model: &mut GradientLSS<B>,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    hp_dict: &HashMap<String, Value>,
    n_trials: u32,
    n_folds: usize,
    seed: u64,
) -> Result<HyperOptResult, Box<dyn std::error::Error>> {
    let config = HyperOptConfig {
        n_trials,
        n_folds,
        seed,
        pruning: PruningStrategy::None,
        ..Default::default()
    };
    hyper_opt_with_config(model, features, labels, hp_dict, config)
}

/// Perform hyperparameter optimization with pruning support.
///
/// This is the full-featured version that supports trial pruning for early
/// termination of unpromising trials, similar to Optuna's pruning API.
///
/// # Arguments
/// * `model` - The GradientLSS model to optimize
/// * `features` - Training features
/// * `labels` - Training labels
/// * `hp_dict` - Hyperparameter search space
/// * `config` - Optimization configuration including pruning strategy
///
/// # Returns
/// The best hyperparameters found during optimization.
///
/// # Example
/// ```ignore
/// use gradientlss::hyper_opt::{hyper_opt_with_config, HyperOptConfig, PruningStrategy};
///
/// let config = HyperOptConfig {
///     n_trials: 100,
///     n_folds: 5,
///     pruning: PruningStrategy::Median { n_warmup_trials: 10 },
///     ..Default::default()
/// };
///
/// let result = hyper_opt_with_config(&mut model, &features, &labels, &hp_dict, config)?;
/// ```
pub fn hyper_opt_with_config<B: Backend>(
    model: &mut GradientLSS<B>,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    hp_dict: &HashMap<String, Value>,
    config: HyperOptConfig,
) -> Result<HyperOptResult, Box<dyn std::error::Error>> {
    hyper_opt_with_transform(model, features, labels, hp_dict, config, None)
}

/// [`hyper_opt_with_config`] plus a [`FoldTransform`] fitted inside each CV
/// fold.
///
/// Use this when preprocessing learns from the data — imputation above all.
/// The plain entry point above passes `None` and is exactly equivalent to the
/// previous behaviour, so existing callers are unaffected.
///
/// Because the transform sees each trial's sampled parameters, its own
/// hyperparameters can be declared in `hp_dict` and tuned jointly with the
/// model's, scored on folds the transform never fitted on.
/// Encode a caller-facing JSON value into the TPE optimizer's raw space — the
/// inverse of the raw -> param conversion in the trial loop (ln for log
/// scales, choice index for categoricals). Used to replay a prior observation
/// via [`HyperOptConfig::warm_start`].
///
/// Returns `None` when the value cannot be placed in this param's space — a
/// wrong JSON type, or a categorical the choice list no longer contains.
/// Out-of-range numerics are CLAMPED rather than rejected: a stale prior near
/// a moved range boundary is still a useful hint, and an out-of-range `tell`
/// would error, letting a stale cache entry break an otherwise fine search.
fn value_to_raw(param_type: &HyperParamType, value: &Value) -> Option<f64> {
    match param_type {
        HyperParamType::Float { low, high, log } => {
            let v = value.as_f64()?.clamp(*low, *high);
            Some(if *log { v.ln() } else { v })
        }
        HyperParamType::Int { low, high, log } => {
            // Accept a float too: JSON round-trips can turn 8 into 8.0.
            let v = value
                .as_i64()
                .or_else(|| value.as_f64().map(|f| f.round() as i64))?
                .clamp(*low, *high) as f64;
            Some(if *log { v.ln() } else { v })
        }
        // Match by VALUE, not index: a reordered choice list must not silently
        // seed a different option.
        HyperParamType::Categorical { choices } => {
            choices.iter().position(|c| c == value).map(|i| i as f64)
        }
    }
}

pub fn hyper_opt_with_transform<B: Backend>(
    model: &mut GradientLSS<B>,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    hp_dict: &HashMap<String, Value>,
    config: HyperOptConfig,
    fold_transform: Option<&dyn FoldTransform>,
) -> Result<HyperOptResult, Box<dyn std::error::Error>> {
    use std::time::Instant;

    // Use hp_seed for hyperparameter sampling, falling back to seed
    let hp_seed = config.hp_seed.unwrap_or(config.seed);
    let mut rng = StdRng::seed_from_u64(hp_seed);

    // Track start time for max_minutes budget
    let start_time = Instant::now();

    // Parse hyperparameter specifications and create optimizers
    let specs = parse_hp_specs(hp_dict)?;
    let mut optimizers: HashMap<String, TpeOptimizer> = HashMap::new();

    for spec in &specs {
        match &spec.param_type {
            HyperParamType::Float { low, high, log } => {
                let (opt_low, opt_high) = if *log {
                    (low.ln(), high.ln())
                } else {
                    (*low, *high)
                };
                let optimizer = TpeOptimizer::new(parzen_estimator(), range(opt_low, opt_high)?);
                optimizers.insert(spec.name.clone(), optimizer);
            }
            HyperParamType::Int { low, high, log } => {
                let (opt_low, opt_high) = if *log {
                    ((*low as f64).ln(), (*high as f64).ln())
                } else {
                    (*low as f64, *high as f64)
                };
                let optimizer = TpeOptimizer::new(parzen_estimator(), range(opt_low, opt_high)?);
                optimizers.insert(spec.name.clone(), optimizer);
            }
            HyperParamType::Categorical { choices } => {
                // Categorical choices are unordered: model them with the
                // histogram estimator over [0, n) index bins (like Optuna's
                // suggest_categorical) instead of a parzen estimator, which
                // would treat adjacent indices as "similar" values.
                let n_choices = choices.len();
                if n_choices > 0 {
                    let optimizer =
                        TpeOptimizer::new(histogram_estimator(), categorical_range(n_choices)?);
                    optimizers.insert(spec.name.clone(), optimizer);
                }
            }
        }
    }

    // WARM START: replay a prior search's winner as an observation on every
    // optimizer, before any trial runs. The encoding below is the exact
    // INVERSE of the raw -> param conversion in the trial loop (ln for log
    // scales, choice index for categoricals), so the prior lands in the same
    // space the optimizer samples from.
    if let Some((prior_params, prior_score)) = &config.warm_start {
        let mut seeded = 0usize;
        for spec in &specs {
            let Some(value) = prior_params.get(&spec.name) else {
                continue;
            };
            let Some(optimizer) = optimizers.get_mut(&spec.name) else {
                continue;
            };
            let Some(raw) = value_to_raw(&spec.param_type, value) else {
                continue;
            };
            if optimizer.tell(raw, *prior_score).is_ok() {
                seeded += 1;
            }
        }
        if config.verbose {
            eprintln!(
                "Warm start: seeded {}/{} params from a prior search (score {:.6})",
                seeded,
                specs.len(),
                prior_score
            );
        }
    }

    // One shuffled row permutation, fixed for the entire search, so `seed`
    // actually drives CV fold generation as documented and every trial is
    // scored on the SAME folds (like xgb.cv / Optuna). TimeSeries keeps row
    // order (shuffling would leak future rows); the single holdout keeps its
    // documented "first 80% / last 20%" split.
    let fold_perm: Option<Vec<usize>> =
        if matches!(config.cv_scheme, CvScheme::KFold) && config.n_folds >= 2 {
            use rand::seq::SliceRandom;
            let mut perm: Vec<usize> = (0..features.nrows()).collect();
            perm.shuffle(&mut StdRng::seed_from_u64(config.seed));
            Some(perm)
        } else {
            None
        };

    // Start values depend only on each fold's train labels, which are fixed
    // across trials — compute them on the first trial that reaches a fold and
    // reuse them afterwards. Saves one L-BFGS start-value fit per trial × fold
    // (significant for distributions with numerical-gradient fits like
    // NegativeBinomial on large data).
    let n_fold_slots = if config.n_folds <= 1 {
        1
    } else {
        config.n_folds
    };
    let mut fold_start_values: Vec<Option<Array1<f64>>> = vec![None; n_fold_slots];

    let mut best_score = f64::INFINITY;
    let mut trials_since_best: u32 = 0;
    let mut best_params: HashMap<String, Value> = HashMap::new();
    let mut best_rounds: usize = config.num_boost_round;
    let mut trials: Vec<TrialResult> = Vec::with_capacity(config.n_trials as usize);
    let mut pruner_state = PrunerState::new();
    let mut n_completed_trials = 0usize;

    for trial in 0..config.n_trials {
        // Check time budget
        if let Some(max_mins) = config.max_minutes {
            let elapsed_mins = start_time.elapsed().as_secs_f64() / 60.0;
            if elapsed_mins >= max_mins {
                if config.verbose {
                    eprintln!(
                        "Time budget of {:.1} minutes exceeded after {} trials",
                        max_mins, trial
                    );
                }
                break;
            }
        }
        // Sample hyperparameters
        let mut sampled_params: HashMap<String, f64> = HashMap::new();
        let mut trial_params: HashMap<String, Value> = HashMap::new();

        for spec in &specs {
            if let Some(optimizer) = optimizers.get_mut(&spec.name) {
                let raw_value = optimizer.ask(&mut rng)?;
                sampled_params.insert(spec.name.clone(), raw_value);

                // Convert raw value to actual parameter value
                let param_value = match &spec.param_type {
                    HyperParamType::Float { log, .. } => {
                        let v = if *log { raw_value.exp() } else { raw_value };
                        Value::from(v)
                    }
                    HyperParamType::Int { log, .. } => {
                        let v = if *log {
                            raw_value.exp().round() as i64
                        } else {
                            raw_value.round() as i64
                        };
                        Value::from(v)
                    }
                    HyperParamType::Categorical { choices } => {
                        let idx = (raw_value.floor() as usize).min(choices.len() - 1);
                        choices[idx].clone()
                    }
                };
                trial_params.insert(spec.name.clone(), param_value);
            }
        }

        // Create backend params with the sampled values
        let mut backend_params = B::create_params(model.n_params());

        for (key, value) in &trial_params {
            // Transform-only axes (reserved `__` prefix) configure the
            // `FoldTransform`, not the booster. Forwarding them would make
            // XGBoost / LightGBM log an "unrecognized parameter" warning on
            // every trial, and a strict backend could reject the fit outright.
            if key.starts_with(TRANSFORM_PARAM_PREFIX) {
                continue;
            }
            let param_value = json_to_param_value(value);
            backend_params.set(key, param_value);
        }

        // Run cross-validation with pruning support
        let train_config = TrainConfig {
            num_boost_round: config.num_boost_round,
            early_stopping_rounds: config.early_stopping_rounds,
            verbose: false,
            collect_train_metrics: false,
            seed: config.seed + trial as u64,
        };

        let (cv_score, pruned, intermediate_scores, fold_best_rounds) = cv_with_pruning(
            model,
            features,
            labels,
            config.n_folds,
            config.cv_scheme,
            fold_perm.as_deref(),
            &mut fold_start_values,
            backend_params,
            train_config,
            &config.pruning,
            &pruner_state,
            n_completed_trials,
            fold_transform,
            &trial_params,
        );

        // Report result to optimizers (even for pruned trials). `cv_score` is
        // the mean over the folds run so far, so pruned and completed trials
        // feed the TPE estimator the same quantity on the same scale.
        let mut report_score = cv_score;

        // Prevent TPE from crashing on NaN values
        if report_score.is_nan() {
            report_score = f64::INFINITY;
        }

        for (name, raw_value) in &sampled_params {
            if let Some(optimizer) = optimizers.get_mut(name) {
                optimizer.tell(*raw_value, report_score)?;
            }
        }

        // Track best (only from completed trials)
        if !pruned {
            trials_since_best = trials_since_best.saturating_add(1);
        }
        if !pruned && cv_score < best_score {
            best_score = cv_score;
            trials_since_best = 0;
            best_params = trial_params.clone();
            // Like Python's hyper_opt (test-mean idxmin + 1), take opt_rounds
            // from the best trial's early-stopped optimum: the mean of the
            // folds' best iteration counts.
            best_rounds = if fold_best_rounds.is_empty() {
                config.num_boost_round
            } else {
                let mean =
                    fold_best_rounds.iter().sum::<usize>() as f64 / fold_best_rounds.len() as f64;
                (mean.round() as usize).max(1)
            };
        }

        // Record intermediate values for future pruning decisions. Store the
        // CUMULATIVE mean at each step — the same quantity `should_prune`
        // receives — so median/percentile pruning compares like with like.
        if !pruned {
            let mut cum = 0.0;
            for (step, &score) in intermediate_scores.iter().enumerate() {
                cum += score;
                pruner_state.record(step, cum / (step + 1) as f64);
            }
            n_completed_trials += 1;
        }

        trials.push(TrialResult {
            params: trial_params,
            score: cv_score,
            pruned,
            intermediate_scores,
        });

        // Optional: print progress
        if config.verbose && (trial + 1) % 10 == 0 {
            let pruned_count = trials.iter().filter(|t| t.pruned).count();
            eprintln!(
                "Trial {}/{}: score = {:.6}, best = {:.6}, pruned = {}/{}",
                trial + 1,
                config.n_trials,
                cv_score,
                best_score,
                pruned_count,
                trial + 1
            );
        }

        // Study-level early stop. Placed after the progress print so the last
        // line before stopping still reports the state that triggered it.
        if let Some(patience) = config.plateau_patience
            && patience > 0
            && trials_since_best >= patience
        {
            if config.verbose {
                eprintln!(
                    "Stopping at trial {}/{}: {} completed trials with no new best                      (best = {:.6})",
                    trial + 1,
                    config.n_trials,
                    trials_since_best,
                    best_score
                );
            }
            break;
        }
    }

    Ok(HyperOptResult {
        best_params,
        best_score,
        opt_rounds: best_rounds,
        trials,
    })
}

/// Cross-validation with pruning support.
///
/// `n_folds >= 2` runs standard k-fold CV. `n_folds <= 1` runs a single 80/20
/// holdout (train on the first 80% of the data, validate/score on the last
/// 20%) — one fit, no cross-fold pruning. The holdout mode is much cheaper and
/// matches a plain train/val split for hyperparameter search.
///
/// `fold_perm` is an optional seeded row permutation: when present, k-fold
/// test/train rows are gathered through it (shuffled folds); when absent the
/// folds are contiguous row ranges.
///
/// Returns (final_score, was_pruned, intermediate_scores, fold_best_rounds)
/// where `fold_best_rounds` holds each successful fold's early-stopped
/// optimal round count (`best_iteration + 1`, or the rounds actually run).
fn cv_with_pruning<B: Backend>(
    model: &GradientLSS<B>,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    n_folds: usize,
    cv_scheme: CvScheme,
    fold_perm: Option<&[usize]>,
    fold_start_values: &mut [Option<Array1<f64>>],
    params: B::Params,
    config: TrainConfig,
    pruning_strategy: &PruningStrategy,
    pruner_state: &PrunerState,
    n_completed_trials: usize,
    fold_transform: Option<&dyn FoldTransform>,
    trial_params: &HashMap<String, Value>,
) -> (f64, bool, Vec<f64>, Vec<usize>) {
    use crate::backend::BackendDataset;
    use ndarray::{Axis, s};

    let n_samples = features.nrows();
    // `n_folds <= 1` ⇒ single 80/20 holdout (one iteration); otherwise k-fold.
    let single_holdout = n_folds <= 1;
    let n_iters = if single_holdout { 1 } else { n_folds };
    let fold_size = n_samples / n_iters;
    // Forward-chaining splits into `n_folds + 1` time segments; disabled when there
    // are too few rows to segment (falls back to k-fold/holdout behavior).
    let ts_seg = n_samples / (n_folds + 1);
    let time_series = matches!(cv_scheme, CvScheme::TimeSeries) && !single_holdout && ts_seg > 0;
    let mut intermediate_scores = Vec::with_capacity(n_iters);
    let mut fold_best_rounds: Vec<usize> = Vec::with_capacity(n_iters);
    let mut pruned = false;

    for i in 0..n_iters {
        // Partition fold `i` into (train, test). `TimeSeries` trains only on rows
        // strictly before the test segment (no future leakage); `KFold`/holdout
        // trains on all other rows.
        let (train_features, train_labels, test_features, test_labels, train_idx, valid_idx) =
            if time_series {
            let train_end = (i + 1) * ts_seg;
            let test_end = if i == n_iters - 1 {
                n_samples
            } else {
                train_end + ts_seg
            };
            (
                features.slice(s![..train_end, ..]).to_owned(),
                labels.slice(s![..train_end]).to_owned(),
                features.slice(s![train_end..test_end, ..]).to_owned(),
                labels.slice(s![train_end..test_end]).to_owned(),
                (0..train_end).collect::<Vec<usize>>(),
                (train_end..test_end).collect::<Vec<usize>>(),
            )
        } else {
            let (test_start, test_end) = if single_holdout {
                // Last 20% is the held-out validation/scoring fold.
                (((n_samples as f64) * 0.8) as usize, n_samples)
            } else if i == n_iters - 1 {
                (i * fold_size, n_samples)
            } else {
                (i * fold_size, (i + 1) * fold_size)
            };

            if let Some(p) = fold_perm {
                // Shuffled k-fold: gather rows through the seeded permutation.
                let test_idx = &p[test_start..test_end];
                let train_idx: Vec<usize> = p[..test_start]
                    .iter()
                    .chain(p[test_end..].iter())
                    .copied()
                    .collect();
                (
                    features.select(Axis(0), &train_idx),
                    labels.select(Axis(0), &train_idx),
                    features.select(Axis(0), test_idx),
                    labels.select(Axis(0), test_idx),
                    train_idx.clone(),
                    test_idx.to_vec(),
                )
            } else {
                // Contiguous split (single holdout, or no permutation supplied).
                let test_features = features.slice(s![test_start..test_end, ..]).to_owned();
                let test_labels = labels.slice(s![test_start..test_end]).to_owned();

                let train_features_1 = features.slice(s![..test_start, ..]);
                let train_labels_1 = labels.slice(s![..test_start]);
                let train_features_2 = features.slice(s![test_end.., ..]);
                let train_labels_2 = labels.slice(s![test_end..]);

                let train_features =
                    match ndarray::concatenate(Axis(0), &[train_features_1, train_features_2]) {
                        Ok(f) => f,
                        Err(_) => {
                            intermediate_scores.push(f64::INFINITY);
                            continue;
                        }
                    };
                let train_labels =
                    match ndarray::concatenate(Axis(0), &[train_labels_1, train_labels_2]) {
                        Ok(l) => l,
                        Err(_) => {
                            intermediate_scores.push(f64::INFINITY);
                            continue;
                        }
                    };
                // Contiguous arms: indices are the ranges themselves, in the
                // same order the blocks were concatenated.
                let train_idx: Vec<usize> =
                    (0..test_start).chain(test_end..n_samples).collect();
                let valid_idx: Vec<usize> = (test_start..test_end).collect();
                (
                    train_features,
                    train_labels,
                    test_features,
                    test_labels,
                    train_idx,
                    valid_idx,
                )
            }
        };

        // Fit-inside-the-fold preprocessing. Runs AFTER the split so the
        // transform only ever sees this fold's training rows — the whole point
        // of the hook. A failure here makes the fold unscorable rather than
        // killing the study, matching how the dataset-construction failures
        // below are handled.
        let (mut train_features, mut test_features) = (train_features, test_features);
        if let Some(t) = fold_transform
            && let Err(e) = t.fit_transform(
                &mut train_features,
                &mut test_features,
                &train_idx,
                &valid_idx,
                trial_params,
            )
        {
            if std::env::var("GRADIENTLSS_VERBOSE").is_ok() {
                eprintln!("fold {i}: fold transform failed ({e}); scoring as infinite");
            }
            intermediate_scores.push(f64::INFINITY);
            continue;
        }

        let mut train_data = match B::Dataset::from_data(train_features.view(), train_labels.view())
        {
            Ok(d) => d,
            Err(_) => {
                intermediate_scores.push(f64::INFINITY);
                continue;
            }
        };

        // Validation dataset = the held-out fold. Passing it to `train` enables
        // the backend's early stopping (`config.early_stopping_rounds`). Without
        // a validation set XGBoost/LightGBM cannot early-stop, so every fold
        // runs the full `num_boost_round` even after the model has converged.
        // Reusing the scoring fold as the early-stopping monitor makes each
        // trial's score mildly optimistic, but uniformly so across trials — the
        // TPE ranking (all that matters for HPO) is unaffected.
        let mut val_data = match B::Dataset::from_data(test_features.view(), test_labels.view()) {
            Ok(d) => d,
            Err(_) => {
                intermediate_scores.push(f64::INFINITY);
                continue;
            }
        };

        let mut fold_model = GradientLSS::<B>::new(model.distribution().clone_arc());
        // Reuse this fold's start values from an earlier trial when available —
        // they depend only on the fold's train labels, which are fixed for the
        // whole search.
        if let Some(sv) = fold_start_values.get(i).and_then(|s| s.as_ref()) {
            fold_model.preset_start_values(sv.clone());
        }
        let train_result = match fold_model.train_with_callbacks(
            &mut train_data,
            Some(&mut val_data),
            params.clone(),
            config.clone(),
            None::<&mut crate::backend::HistoryCallback>,
        ) {
            Ok(((), result)) => result,
            Err(_) => {
                intermediate_scores.push(f64::INFINITY);
                continue;
            }
        };
        if let Some(slot) = fold_start_values.get_mut(i) {
            if slot.is_none() {
                *slot = fold_model.start_values().cloned();
            }
        }

        // Rounds this fold's early stopping actually selected (argmin of its
        // validation curve); falls back to the number of rounds run.
        fold_best_rounds.push(
            train_result
                .best_iteration
                .map(|b| b + 1)
                .unwrap_or(train_result.n_iterations),
        );

        // Score on the INTERNAL transformed parameters (start values added, no
        // user-facing finalize — Mixture's `nll` expects raw mixing logits).
        //
        // Scored with the model's CONFIGURED loss via `metric_on_transformed`,
        // not a hardcoded `nll`. Under `LossFn::Nll` this is byte-identical to
        // the old direct `nll` call (it delegates to the same function on the
        // same transformed params). Under `LossFn::Crps` the old code ranked
        // CRPS-trained trials — and chose their round counts — by an objective
        // they were not optimizing, which selected hyperparameters that LOOK
        // NLL-reasonable while the CRPS fit itself is unconverged or diverged
        // (observed 2026-08-27: a tuned xgboost-lss Poisson CRPS model with
        // test RMSE worse than predicting the training mean).
        let score = match fold_model.predict_transformed(&test_features.view()) {
            Ok(preds) => {
                let test_labels_owned = test_labels.to_owned();
                fold_model.metric_on_transformed(&preds, &test_labels_owned)
            }
            Err(_) => f64::INFINITY,
        };

        intermediate_scores.push(score);

        // Check if we should prune after this fold. Skipped in single-holdout
        // mode — there are no further folds to save by pruning.
        if !single_holdout {
            let current_avg =
                intermediate_scores.iter().sum::<f64>() / intermediate_scores.len() as f64;
            if pruner_state.should_prune(pruning_strategy, i, current_avg, n_completed_trials) {
                pruned = true;
                break;
            }
        }
    }

    let final_score = if intermediate_scores.is_empty() {
        f64::INFINITY
    } else {
        intermediate_scores.iter().sum::<f64>() / intermediate_scores.len() as f64
    };

    (final_score, pruned, intermediate_scores, fold_best_rounds)
}

/// Parse hyperparameter specifications from JSON values.
fn parse_hp_specs(hp_dict: &HashMap<String, Value>) -> Result<Vec<HyperParamSpec>, String> {
    let mut specs = Vec::new();

    for (name, value) in hp_dict {
        let spec = match value {
            // Array format: [low, high]. Integer bounds (JSON `2`, not `2.0`)
            // mean an INT range — optuna-suggest_int semantics for params like
            // num_leaves/max_depth, which LightGBM rejects as floats ("Parameter
            // num_leaves should be of type int, got 5.07..."). Any float bound
            // makes it a float range.
            Value::Array(arr) if arr.len() == 2 => {
                if let (Some(low), Some(high)) = (arr[0].as_i64(), arr[1].as_i64()) {
                    HyperParamSpec {
                        name: name.clone(),
                        param_type: HyperParamType::Int {
                            low,
                            high,
                            log: false,
                        },
                    }
                } else {
                    let low = arr[0]
                        .as_f64()
                        .ok_or_else(|| format!("Invalid low value for {}", name))?;
                    let high = arr[1]
                        .as_f64()
                        .ok_or_else(|| format!("Invalid high value for {}", name))?;
                    HyperParamSpec::float(name.clone(), low, high)
                }
            }
            // Object format with more options
            Value::Object(obj) => {
                if let Some(choices) = obj.get("choices") {
                    // Categorical parameter
                    let choices = choices
                        .as_array()
                        .ok_or_else(|| format!("Invalid choices for {}", name))?
                        .clone();
                    HyperParamSpec::categorical(name.clone(), choices)
                } else {
                    // Numeric parameter with options
                    let low = obj
                        .get("low")
                        .and_then(|v| v.as_f64())
                        .ok_or_else(|| format!("Missing 'low' for {}", name))?;
                    let high = obj
                        .get("high")
                        .and_then(|v| v.as_f64())
                        .ok_or_else(|| format!("Missing 'high' for {}", name))?;
                    let log = obj.get("log").and_then(|v| v.as_bool()).unwrap_or(false);
                    let is_int = obj.get("type").and_then(|v| v.as_str()) == Some("int");

                    if is_int {
                        HyperParamSpec {
                            name: name.clone(),
                            param_type: HyperParamType::Int {
                                low: low as i64,
                                high: high as i64,
                                log,
                            },
                        }
                    } else if log {
                        HyperParamSpec::log_float(name.clone(), low, high)
                    } else {
                        HyperParamSpec::float(name.clone(), low, high)
                    }
                }
            }
            _ => return Err(format!("Invalid hyperparameter format for {}", name)),
        };
        specs.push(spec);
    }

    // HashMap iteration order is randomized per process, and the specs order
    // sets the order in which parameters consume the shared RNG stream — sort
    // by name so seeded runs are reproducible.
    specs.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(specs)
}

/// Convert a JSON Value to a ParamValue.
fn json_to_param_value(value: &Value) -> ParamValue {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ParamValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                ParamValue::Float(f)
            } else {
                ParamValue::String(n.to_string())
            }
        }
        Value::String(s) => ParamValue::String(s.clone()),
        Value::Bool(b) => ParamValue::Bool(*b),
        _ => ParamValue::String(value.to_string()),
    }
}

/// Convenience function for simple hyperparameter optimization.
///
/// Uses default settings and a simple search space format.
pub fn hyper_opt_simple<B: Backend>(
    model: &mut GradientLSS<B>,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    hp_dict: &HashMap<String, Value>,
    n_trials: u32,
    n_folds: usize,
) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
    let result = hyper_opt(model, features, labels, hp_dict, n_trials, n_folds, 42)?;
    Ok(result.best_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hp_specs_array() {
        let mut hp_dict = HashMap::new();
        hp_dict.insert("learning_rate".to_string(), serde_json::json!([0.01, 0.3]));

        let specs = parse_hp_specs(&hp_dict).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "learning_rate");

        match &specs[0].param_type {
            HyperParamType::Float { low, high, log } => {
                assert_eq!(*low, 0.01);
                assert_eq!(*high, 0.3);
                assert!(!log);
            }
            _ => panic!("Expected Float type"),
        }
    }

    #[test]
    fn test_parse_hp_specs_object() {
        let mut hp_dict = HashMap::new();
        hp_dict.insert(
            "learning_rate".to_string(),
            serde_json::json!({"low": 0.001, "high": 0.1, "log": true}),
        );

        let specs = parse_hp_specs(&hp_dict).unwrap();
        assert_eq!(specs.len(), 1);

        match &specs[0].param_type {
            HyperParamType::Float { low, high, log } => {
                assert_eq!(*low, 0.001);
                assert_eq!(*high, 0.1);
                assert!(log);
            }
            _ => panic!("Expected Float type"),
        }
    }

    #[test]
    fn test_parse_hp_specs_categorical() {
        let mut hp_dict = HashMap::new();
        hp_dict.insert(
            "booster".to_string(),
            serde_json::json!({"choices": ["gbtree", "dart"]}),
        );

        let specs = parse_hp_specs(&hp_dict).unwrap();
        assert_eq!(specs.len(), 1);

        match &specs[0].param_type {
            HyperParamType::Categorical { choices } => {
                assert_eq!(choices.len(), 2);
            }
            _ => panic!("Expected Categorical type"),
        }
    }

    #[test]
    fn test_json_to_param_value() {
        assert!(matches!(
            json_to_param_value(&serde_json::json!(42)),
            ParamValue::Int(42)
        ));
        assert!(matches!(
            json_to_param_value(&serde_json::json!(3.5)),
            ParamValue::Float(_)
        ));
        assert!(matches!(
            json_to_param_value(&serde_json::json!("test")),
            ParamValue::String(_)
        ));
        assert!(matches!(
            json_to_param_value(&serde_json::json!(true)),
            ParamValue::Bool(true)
        ));
    }

    #[test]
    fn test_parse_hp_specs_sorted_by_name() {
        // HashMap iteration order is randomized per process; specs must come
        // back name-sorted so seeded searches consume the RNG in a stable order.
        let mut hp_dict = HashMap::new();
        hp_dict.insert("eta".to_string(), serde_json::json!([0.01, 0.3]));
        hp_dict.insert("max_depth".to_string(), serde_json::json!([2.0, 8.0]));
        hp_dict.insert("alpha".to_string(), serde_json::json!([0.0, 1.0]));
        hp_dict.insert("lambda".to_string(), serde_json::json!([0.0, 1.0]));

        let specs = parse_hp_specs(&hp_dict).unwrap();
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "eta", "lambda", "max_depth"]);
    }
}

#[cfg(all(test, feature = "xgboost"))]
mod xgboost_hyper_opt_tests {
    use super::*;
    use crate::backend::XGBoostBackend;
    use crate::distributions::Gaussian;
    use ndarray::{Array1, Array2};
    use std::sync::Arc;

    /// Deterministic noisy regression data (same LCG scheme as the backend
    /// best_iteration tests) — noisy enough that CV early stopping fires well
    /// before `num_boost_round`.
    fn noisy_dataset() -> (Array2<f64>, Array1<f64>) {
        let n = 200usize;
        let f = 4usize;
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (u32::MAX as f64)
        };

        let mut feats = Array2::<f64>::zeros((n, f));
        let mut labels = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut signal = 0.0;
            for j in 0..f {
                let x = next();
                feats[[i, j]] = x;
                if j < 2 {
                    signal += x;
                }
            }
            labels[i] = signal + (next() - 0.5) * 4.0;
        }
        (feats, labels)
    }

    fn run_search(seed: u64) -> HyperOptResult {
        let (features, labels) = noisy_dataset();
        let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));

        let mut hp_dict = HashMap::new();
        hp_dict.insert("eta".to_string(), serde_json::json!([0.05, 0.5]));
        hp_dict.insert(
            "max_depth".to_string(),
            serde_json::json!({"low": 2.0, "high": 6.0, "type": "int"}),
        );

        let config = HyperOptConfig {
            n_trials: 4,
            n_folds: 2,
            seed,
            num_boost_round: 60,
            early_stopping_rounds: Some(5),
            verbose: false,
            ..Default::default()
        };

        hyper_opt_with_config(&mut model, &features, &labels, &hp_dict, config).unwrap()
    }

    #[test]
    fn test_seeded_search_is_reproducible() {
        let a = run_search(11);
        let b = run_search(11);

        assert_eq!(a.trials.len(), b.trials.len());
        for (ta, tb) in a.trials.iter().zip(&b.trials) {
            assert_eq!(
                ta.params, tb.params,
                "same seed must sample identical hyperparameters"
            );
            assert_eq!(ta.score.to_bits(), tb.score.to_bits());
        }
        assert_eq!(a.best_score.to_bits(), b.best_score.to_bits());
        assert_eq!(a.opt_rounds, b.opt_rounds);
    }

    /// Manual benchmark proving the per-fold start-value cache is a real
    /// speedup. Uses NegativeBinomial with `initialize: true` (the L-BFGS
    /// start-value path; distributions with `initialize: false` skip the fit
    /// entirely and the cache is a no-op for them). True A/B: `&mut []` as the
    /// cache slice disables caching (no slots to read or write).
    /// `cargo test --release --features xgboost --lib start_value_cache_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn start_value_cache_timing() {
        use crate::backend::Backend;
        use crate::distributions::{LossFn, NegativeBinomial, Stabilization};
        use crate::utils::ResponseFn;
        use std::time::Instant;

        // ~50k-row count dataset (LCG, deterministic).
        let n = 50_000usize;
        let f = 4usize;
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (u32::MAX as f64)
        };
        let mut features = Array2::<f64>::zeros((n, f));
        let mut labels = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut rate = 1.0;
            for j in 0..f {
                let x = next();
                features[[i, j]] = x;
                rate += x;
            }
            labels[i] = (rate * next() * 4.0).floor(); // counts in [0, ~20)
        }

        let nb = NegativeBinomial::new(
            Stabilization::Mad,
            ResponseFn::Relu,
            ResponseFn::Sigmoid,
            LossFn::Nll,
            true, // initialize: engage the L-BFGS start-value fit
        );
        let model = GradientLSS::<XGBoostBackend>::new(Arc::new(nb));

        let n_trials = 6usize;
        let n_folds = 3usize;
        let config = TrainConfig {
            num_boost_round: 10,
            early_stopping_rounds: None,
            verbose: false,
            seed: 7,
            collect_train_metrics: false,
        };
        let pruner_state = PrunerState::new();

        let run = |cache: &mut [Option<Array1<f64>>]| -> std::time::Duration {
            let t = Instant::now();
            for _ in 0..n_trials {
                let params = XGBoostBackend::create_params(model.n_params());
                let (score, ..) = cv_with_pruning(
                    &model,
                    &features,
                    &labels,
                    n_folds,
                    CvScheme::KFold,
                    None,
                    cache,
                    params,
                    config.clone(),
                    &PruningStrategy::None,
                    &pruner_state,
                    0,
                    None,
                    &HashMap::new(),
                );
                assert!(score.is_finite());
            }
            t.elapsed()
        };

        // Warm-up (allocator, xgboost init).
        run(&mut []);

        let uncached = run(&mut []);
        let mut cache: Vec<Option<Array1<f64>>> = vec![None; n_folds];
        let cached = run(&mut cache);

        println!(
            "start-value cache A/B over {n_trials} trials x {n_folds} folds @ {n} rows:\n\
             uncached: {uncached:?}\n\
             cached:   {cached:?}\n\
             saved:    {:?} ({:.1}%)",
            uncached.saturating_sub(cached),
            100.0 * (1.0 - cached.as_secs_f64() / uncached.as_secs_f64()),
        );
        assert!(
            cached < uncached,
            "cached ({cached:?}) should beat uncached ({uncached:?})"
        );
    }

    #[test]
    fn test_opt_rounds_reflects_early_stopping() {
        let result = run_search(11);

        assert!(result.best_score.is_finite());
        assert!(result.opt_rounds >= 1);
        // On heavily noisy labels the CV folds early-stop far below the
        // budget; opt_rounds must come from that optimum, not num_boost_round.
        assert!(
            result.opt_rounds < 60,
            "opt_rounds ({}) should reflect the folds' early-stopped optimum, \
             not the num_boost_round budget",
            result.opt_rounds
        );
    }
}

#[cfg(test)]
mod warm_start_tests {
    use super::*;

    /// The correctness claim for warm start: a prior must land where the
    /// optimizer would itself have sampled that value from, so encoding and
    /// materializing back must be the identity.
    #[test]
    fn value_to_raw_round_trips_for_every_param_type() {
        // Float, linear and log.
        let pt = HyperParamType::Float { low: 0.01, high: 0.5, log: false };
        assert_eq!(value_to_raw(&pt, &Value::from(0.25)), Some(0.25));
        let pt = HyperParamType::Float { low: 1e-6, high: 1e-1, log: true };
        let raw = value_to_raw(&pt, &Value::from(1e-3)).unwrap();
        assert!((raw.exp() - 1e-3).abs() < 1e-12, "log float round-trip: {}", raw.exp());

        // Int, linear and log.
        let pt = HyperParamType::Int { low: 2, high: 12, log: false };
        assert_eq!(value_to_raw(&pt, &Value::from(7)), Some(7.0));
        let pt = HyperParamType::Int { low: 10, high: 1000, log: true };
        let raw = value_to_raw(&pt, &Value::from(100)).unwrap();
        assert_eq!(raw.exp().round() as i64, 100);

        // Categorical -> index.
        let pt = HyperParamType::Categorical {
            choices: vec![Value::from("a"), Value::from("b"), Value::from("c")],
        };
        assert_eq!(value_to_raw(&pt, &Value::from("b")), Some(1.0));
    }

    /// A stale prior outside a since-moved range must be pulled INTO range,
    /// not told out-of-range (which `tell` errors on).
    #[test]
    fn out_of_range_values_are_clamped_not_rejected() {
        let pt = HyperParamType::Float { low: 0.1, high: 0.5, log: false };
        assert_eq!(value_to_raw(&pt, &Value::from(9.9)), Some(0.5));
        assert_eq!(value_to_raw(&pt, &Value::from(-9.9)), Some(0.1));
    }

    /// A choice list that changed between searches must not silently seed a
    /// DIFFERENT option.
    #[test]
    fn categorical_misses_when_the_choice_is_gone() {
        let pt = HyperParamType::Categorical {
            choices: vec![Value::from("gbtree"), Value::from("dart")],
        };
        assert_eq!(value_to_raw(&pt, &Value::from("gblinear")), None);
    }

    /// Cache entries are untrusted input: a wrong type is a miss, not a panic.
    #[test]
    fn wrong_type_is_a_miss() {
        let pt = HyperParamType::Float { low: 0.0, high: 1.0, log: false };
        assert_eq!(value_to_raw(&pt, &Value::from("not a number")), None);
    }

    /// An int spec must accept a float that round-tripped through JSON.
    #[test]
    fn int_accepts_a_json_float() {
        let pt = HyperParamType::Int { low: 1, high: 10, log: false };
        assert_eq!(value_to_raw(&pt, &Value::from(8.0)), Some(8.0));
    }
}
