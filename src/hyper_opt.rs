//! Hyperparameter optimization for GradientLSS.
//!
//! Provides Tree-structured Parzen Estimator (TPE) based optimization
//! for finding optimal hyperparameters with optional trial pruning.

use crate::backend::{Backend, BackendParams, ParamValue, TrainConfig};
use crate::model::GradientLSS;
use ndarray::{Array1, Array2};
// Note: tpe uses rand 0.8, so we use rand_compat (rand 0.8) for RNG compatibility
use rand_compat::SeedableRng;
use rand_compat::rngs::StdRng;
use serde_json::Value;
use std::collections::HashMap;
use tpe::{TpeOptimizer, parzen_estimator, range};

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

/// Configuration for hyperparameter optimization.
#[derive(Debug, Clone)]
pub struct HyperOptConfig {
    /// Number of optimization trials
    pub n_trials: u32,
    /// Number of cross-validation folds
    pub n_folds: usize,
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
    /// Whether to print progress
    pub verbose: bool,
}

impl Default for HyperOptConfig {
    fn default() -> Self {
        Self {
            n_trials: 100,
            n_folds: 5,
            seed: 42,
            hp_seed: None,
            pruning: PruningStrategy::None,
            num_boost_round: 100,
            early_stopping_rounds: Some(10),
            max_minutes: None,
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
                // For categorical, use indices as float range
                let n_choices = choices.len();
                if n_choices > 0 {
                    let optimizer =
                        TpeOptimizer::new(parzen_estimator(), range(0.0, n_choices as f64 - 0.01)?);
                    optimizers.insert(spec.name.clone(), optimizer);
                }
            }
        }
    }

    let mut best_score = f64::INFINITY;
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
            let param_value = json_to_param_value(value);
            backend_params.set(key, param_value);
        }

        // Run cross-validation with pruning support
        let train_config = TrainConfig {
            num_boost_round: config.num_boost_round,
            early_stopping_rounds: config.early_stopping_rounds,
            verbose: false,
            seed: config.seed + trial as u64,
        };

        let (cv_score, pruned, intermediate_scores) = cv_with_pruning(
            model,
            features,
            labels,
            config.n_folds,
            backend_params,
            train_config,
            &config.pruning,
            &pruner_state,
            n_completed_trials,
        );

        // Report result to optimizers (even for pruned trials)
        let report_score = if pruned {
            // For pruned trials, use the last intermediate score or infinity
            intermediate_scores.last().copied().unwrap_or(f64::INFINITY)
        } else {
            cv_score
        };

        for (name, raw_value) in &sampled_params {
            if let Some(optimizer) = optimizers.get_mut(name) {
                optimizer.tell(*raw_value, report_score)?;
            }
        }

        // Track best (only from completed trials)
        if !pruned && cv_score < best_score {
            best_score = cv_score;
            best_params = trial_params.clone();
            best_rounds = config.num_boost_round;
        }

        // Record intermediate scores for future pruning decisions
        if !pruned {
            for (step, &score) in intermediate_scores.iter().enumerate() {
                pruner_state.record(step, score);
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
/// Returns (final_score, was_pruned, intermediate_scores)
fn cv_with_pruning<B: Backend>(
    model: &GradientLSS<B>,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    n_folds: usize,
    params: B::Params,
    config: TrainConfig,
    pruning_strategy: &PruningStrategy,
    pruner_state: &PrunerState,
    n_completed_trials: usize,
) -> (f64, bool, Vec<f64>) {
    use crate::backend::BackendDataset;
    use ndarray::{Axis, s};

    let n_samples = features.nrows();
    let fold_size = n_samples / n_folds;
    let mut intermediate_scores = Vec::with_capacity(n_folds);
    let mut pruned = false;

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
            match ndarray::concatenate(Axis(0), &[train_features_1, train_features_2]) {
                Ok(f) => f,
                Err(_) => {
                    intermediate_scores.push(f64::INFINITY);
                    continue;
                }
            };
        let train_labels = match ndarray::concatenate(Axis(0), &[train_labels_1, train_labels_2]) {
            Ok(l) => l,
            Err(_) => {
                intermediate_scores.push(f64::INFINITY);
                continue;
            }
        };

        let mut train_data = match B::Dataset::from_data(train_features.view(), train_labels.view())
        {
            Ok(d) => d,
            Err(_) => {
                intermediate_scores.push(f64::INFINITY);
                continue;
            }
        };

        let mut fold_model = GradientLSS::<B>::new(model.distribution().clone_arc());
        if fold_model
            .train(&mut train_data, None, params.clone(), config.clone())
            .is_err()
        {
            intermediate_scores.push(f64::INFINITY);
            continue;
        }

        // Get prediction using public API and compute score
        let score = match fold_model.predict(
            &test_features,
            crate::model::PredType::Parameters,
            0,
            &[],
            0,
        ) {
            Ok(crate::backend::PredictionOutput::Parameters(preds)) => {
                let target = crate::types::ResponseData::Univariate(&test_labels);
                fold_model.distribution().nll(&preds.view(), &target)
            }
            _ => f64::INFINITY,
        };

        intermediate_scores.push(score);

        // Check if we should prune after this fold
        let current_avg =
            intermediate_scores.iter().sum::<f64>() / intermediate_scores.len() as f64;
        if pruner_state.should_prune(pruning_strategy, i, current_avg, n_completed_trials) {
            pruned = true;
            break;
        }
    }

    let final_score = if intermediate_scores.is_empty() {
        f64::INFINITY
    } else {
        intermediate_scores.iter().sum::<f64>() / intermediate_scores.len() as f64
    };

    (final_score, pruned, intermediate_scores)
}

/// Parse hyperparameter specifications from JSON values.
fn parse_hp_specs(hp_dict: &HashMap<String, Value>) -> Result<Vec<HyperParamSpec>, String> {
    let mut specs = Vec::new();

    for (name, value) in hp_dict {
        let spec = match value {
            // Array format: [low, high] - treated as float range
            Value::Array(arr) if arr.len() == 2 => {
                let low = arr[0]
                    .as_f64()
                    .ok_or_else(|| format!("Invalid low value for {}", name))?;
                let high = arr[1]
                    .as_f64()
                    .ok_or_else(|| format!("Invalid high value for {}", name))?;
                HyperParamSpec::float(name.clone(), low, high)
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
            json_to_param_value(&serde_json::json!(3.14)),
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
}
