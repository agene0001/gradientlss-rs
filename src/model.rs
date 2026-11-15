use crate::backend::{
    Backend, BackendDataset, BackendModel, BackendParams, FeatureImportance, FeatureImportanceType,
    CallbackAction, PredictionOutput, TrainConfig, TrainingCallback, TrainingResult,
};
use crate::distributions::{Distribution, Gaussian, GradientsAndHessians, LossFn, Stabilization};
use crate::utils::ResponseFn;
use crate::error::{GradientLSSError, Result};
use crate::hyper_opt;
use crate::types::ResponseData;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Write};
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
    /// One-shot flag: the next `train` call reuses the preset `start_values`
    /// instead of recomputing them (see `preset_start_values`). Consumed by
    /// that call, so any later train recomputes as usual. Not serialized.
    start_values_preset: bool,
    /// Set by `train_with_callbacks_and_offsets` when per-row offsets were
    /// supplied: the trees then encode ADJUSTMENTS around a per-row prior and
    /// only predict correctly when `predict_with_offset` receives that row's
    /// offset again. In-process guard only — not serialized (the bincode state
    /// is not self-describing, so adding a field would break every saved
    /// model); a loaded model's caller owns this knowledge.
    trained_with_row_offsets: bool,
    /// Set by [`train_with_residual_warm_start`]: the stage-1 location model
    /// whose predictions are this model's per-row location offsets. Prediction
    /// recomputes those offsets from the features automatically, and `save`
    /// persists the stage-1 model (format v2, see `write_to`).
    warm_start: Option<Box<WarmStartState<B>>>,
    /// Per-parameter early stopping (see [`ParamEarlyStopping`]); a training
    /// option, not serialized.
    param_early_stopping: ParamEarlyStoppingSetting,
    param_freeze_log: Option<ParamFreezeLog>,
    param_es_triggers: Option<Vec<Vec<usize>>>,
    /// Location-only warm-up rounds (see [`Self::with_warmup_rounds`]); a
    /// training option, not serialized.
    warmup_rounds: usize,
    /// Tail-robust stopping metric (see [`Self::with_stopping_row_cap`]).
    stopping_row_cap: Option<f64>,
    /// Scale-relative tree regularisation (see [`HessianNormalization`]).
    hessian_normalization: HessianNormalization,
    /// Per-round adaptive ridge on the Hessians (see [`GradientRegularization`]).
    gradient_regularization: Option<GradientRegularization>,
    _backend: PhantomData<B>,
}

/// Gradient-regularised Newton boosting (research prototype, proposal 030;
/// after Zozoulenko, Falkowski, Cass and Gonon, arXiv 2605.00581): every
/// round, each parameter's per-row Hessians get an additive term that grows
/// with that parameter's gradient norm,
///
/// ```text
/// h_ij ← h_ij + strength · h̄_j · ρ_j^{1/4},   ρ_j = mean_i g_ij² / h̄_j,
/// ```
///
/// with h̄_j the mean per-row Hessian of the round. This is the paper's
/// λ_k = sqrt(M ‖g_k‖) added to the quadratic's L² penalty (a per-row term,
/// so a leaf's denominator grows by λ_k times its row count), applied per
/// distribution parameter and with the Hessian-Lipschitz constant M taken as
/// h̄_j^{3/2}, which makes the term independent of the label units. For a
/// log-likelihood at the true parameters ρ_j ≈ 1 (E g² = E h), so the term
/// does not vanish at the statistical optimum; `excess_only` subtracts that
/// floor (ρ^{1/4} − 1, at least 0), leaving only the early damping.
/// Training-only; off by default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientRegularization {
    /// Multiplier on the adaptive term (≥ 0; the paper's sqrt(M) in units of h̄).
    pub strength: f64,
    /// Subtract the noise floor ρ = 1, so the term is zero near the optimum.
    pub excess_only: bool,
}

impl GradientRegularization {
    /// Add this round's term to `gh` (raw loss units, before any rescaling).
    fn apply(&self, gh: &mut GradientsAndHessians) {
        if !(self.strength > 0.0) {
            return;
        }
        for j in 0..gh.gradients.ncols() {
            let (mut g2, mut hs, mut n) = (0.0, 0.0, 0usize);
            for (g, h) in gh.gradients.column(j).iter().zip(gh.hessians.column(j).iter()) {
                if g.is_finite() && h.is_finite() {
                    g2 += g * g;
                    hs += h;
                    n += 1;
                }
            }
            if n == 0 {
                continue;
            }
            let hbar = hs / n as f64;
            let rho = g2 / n as f64 / hbar;
            if !(hbar > 0.0) || !rho.is_finite() {
                continue;
            }
            let root = rho.sqrt().sqrt();
            let factor = if self.excess_only { (root - 1.0).max(0.0) } else { root };
            let lam = self.strength * hbar * factor;
            if lam > 0.0 && lam.is_finite() {
                gh.hessians.column_mut(j).mapv_inplace(|h| h + lam);
            }
        }
    }
}

/// Reference point for [`GradientLSS::with_hessian_normalization`]: where
/// the mean per-row Hessian κⱼ⁻¹ is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HessianNormalization {
    /// Raw Hessians (XGBoost/LightGBM semantics; the default).
    #[default]
    Off,
    /// At the start values (round 0).
    Start,
    /// At the marginal (intercept-only) MLE, `Distribution::calculate_start_values`:
    /// independent of the start and of the label units.
    Marginal,
}

/// Per-parameter early stopping: after every round, for each eligible
/// parameter j, the validation loss is re-evaluated with column j of the
/// validation margin reverted to its value `window` rounds earlier. If undoing
/// parameter j's last `window` trees would LOWER the validation loss, j is
/// frozen from the next round on (its gradients and Hessians are zeroed, as
/// with `TrainConfig::param_freeze_rounds`) while the other parameters keep
/// training — or, with [`Self::with_anneal`], its learning rate is cut instead.
/// The automatic counterpart of a validation-tuned freeze round (gamboostLSS
/// per-parameter `mstop`). Needs a validation set; costs one extra
/// validation-loss evaluation per eligible, unfrozen parameter per round.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamEarlyStopping {
    /// Look-back in rounds (≥ 1).
    pub window: usize,
    /// Parameters that may be frozen; `None` = all. Recommended: the
    /// scale/shape/dispersion parameters only — the location is already
    /// governed by the global early stopping.
    pub params: Option<Vec<usize>>,
    /// Consecutive rounds the revert test must succeed before freezing (≥ 1).
    /// A single success can be transient (the scale briefly ahead of a still
    /// improving location) and a permanent freeze then leaves σ̂ stale.
    pub patience: usize,
    /// `0.0` (default): a trigger freezes the parameter. `f ∈ (0, 1)`: a
    /// trigger multiplies the parameter's gradient column by `f` (its
    /// effective learning rate), the parameter is re-tested after a full
    /// `window` of trees at the new rate, and it is frozen once the
    /// multiplier falls below [`PES_ANNEAL_FREEZE_BELOW`].
    pub anneal: f64,
    /// Every trigger resets the global `early_stopping_rounds` patience
    /// counter (best iteration unaffected), so the fit is not ended while a
    /// parameter's rate is still being cut, and the other parameters get a
    /// full patience window after each cut or freeze.
    pub hold_early_stopping: bool,
}

/// An annealed parameter is frozen once its gradient multiplier falls below this.
pub const PES_ANNEAL_FREEZE_BELOW: f64 = 1.0 / 64.0;

impl ParamEarlyStopping {
    /// `window` look-back, `patience` 1, freeze on trigger, global early
    /// stopping held on each trigger.
    pub fn new(window: usize, params: Option<Vec<usize>>) -> Self {
        Self { window, params, patience: 1, anneal: 0.0, hold_early_stopping: true }
    }

    /// See [`Self::hold_early_stopping`].
    pub fn with_hold_early_stopping(mut self, hold: bool) -> Self {
        self.hold_early_stopping = hold;
        self
    }

    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Cut the learning rate by `factor` on each trigger instead of freezing
    /// (`0.0` = freeze).
    pub fn with_anneal(mut self, factor: f64) -> Self {
        self.anneal = factor;
        self
    }

    /// Apply `hyper_opt` trial axes on top of `base`: `__pes_window` (0 = off),
    /// `__pes_patience`, `__pes_anneal`. Missing axes keep `base`'s values
    /// (or the recommended recipe when `base` is off); `params` comes from
    /// `base` or else `default_params`. Use it on the winning params for the
    /// final fit, as `hyper_opt` does for every fold model.
    pub fn from_trial_params(
        base: Option<&ParamEarlyStopping>,
        default_params: Option<Vec<usize>>,
        trial: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<ParamEarlyStopping> {
        let w = trial.get("__pes_window").and_then(|v| v.as_f64());
        let p = trial.get("__pes_patience").and_then(|v| v.as_f64());
        let a = trial.get("__pes_anneal").and_then(|v| v.as_f64());
        if w.is_none() && p.is_none() && a.is_none() {
            return base.cloned();
        }
        if matches!(w, Some(w) if w.round() < 1.0) {
            return None;
        }
        let mut pes = base.cloned().unwrap_or_else(|| ParamEarlyStopping::recommended(default_params));
        if let Some(w) = w {
            pes.window = w.round() as usize;
        }
        if let Some(p) = p {
            pes.patience = (p.round() as usize).max(1);
        }
        if let Some(a) = a {
            pes.anneal = a.clamp(0.0, 1.0);
        }
        Some(pes)
    }

    /// The recipe used by [`ParamEarlyStoppingSetting::Auto`]: window
    /// [`PES_DEFAULT_WINDOW`], patience 1, anneal [`PES_DEFAULT_ANNEAL`].
    /// Annealing rather than freezing was the only uniform recipe that was
    /// better or neutral on every dataset measured (research cycle 6): a
    /// permanent freeze left StudentT's scale stale while df kept moving
    /// (gaussian_sim +0.39) and was noisy on Gamma naval.
    pub fn recommended(params: Option<Vec<usize>>) -> Self {
        Self::new(PES_DEFAULT_WINDOW, params).with_anneal(PES_DEFAULT_ANNEAL)
    }
}

/// Window of the default per-parameter early stopping recipe.
pub const PES_DEFAULT_WINDOW: usize = 20;
/// Anneal factor of the default per-parameter early stopping recipe.
pub const PES_DEFAULT_ANNEAL: f64 = 0.5;

/// How a model chooses its per-parameter early stopping.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ParamEarlyStoppingSetting {
    /// Use the distribution's measured recipe
    /// ([`Distribution::default_param_early_stopping_params`] with
    /// [`ParamEarlyStopping::recommended`]) whenever a validation set is
    /// passed; no-op for distributions without one.
    #[default]
    Auto,
    /// Never.
    Off,
    /// This configuration whenever a validation set is passed.
    Custom(ParamEarlyStopping),
}

/// Rounds at which parameters were frozen by [`ParamEarlyStopping`] in the
/// last training run (`None` = never frozen).
pub type ParamFreezeLog = Vec<Option<usize>>;

/// Shared state between the metric pass (decides) and the objective (applies).
struct ParamEsState {
    window: usize,
    patience: usize,
    anneal: f64,
    streak: Mutex<Vec<usize>>,
    eligible: Vec<bool>,
    valid_labels: Array1<f64>,
    history: Mutex<std::collections::VecDeque<Array2<f64>>>,
    frozen_at: Mutex<Vec<Option<usize>>>,
    /// Gradient multiplier per parameter (anneal); 1 = untouched.
    grad_scale: Mutex<Vec<f64>>,
    /// First validation round at which each parameter may be tested again.
    next_check: Mutex<Vec<usize>>,
    /// Every trigger round per parameter.
    triggers: Mutex<Vec<Vec<usize>>>,
    hold_early_stopping: bool,
    /// Set by a trigger, consumed by [`PesHoldCallback`].
    hold_pending: std::sync::atomic::AtomicBool,
    valid_round: std::sync::atomic::AtomicUsize,
}

/// Wraps the caller's callback (if any) and reports pending per-parameter
/// early-stopping triggers to the backend's patience counter.
struct PesHoldCallback<'a, C: TrainingCallback> {
    inner: Option<&'a mut C>,
    state: Arc<ParamEsState>,
}

impl<C: TrainingCallback> TrainingCallback for PesHoldCallback<'_, C> {
    fn on_training_start(&mut self, n_iterations: usize) {
        if let Some(c) = self.inner.as_mut() {
            c.on_training_start(n_iterations);
        }
    }

    fn on_iteration_end(&mut self, iteration: usize, train_loss: f64, valid_loss: Option<f64>) -> CallbackAction {
        match self.inner.as_mut() {
            Some(c) => c.on_iteration_end(iteration, train_loss, valid_loss),
            None => CallbackAction::Continue,
        }
    }

    fn on_training_end(&mut self, total_iterations: usize, stopped_early: bool) {
        if let Some(c) = self.inner.as_mut() {
            c.on_training_end(total_iterations, stopped_early);
        }
    }

    fn needs_train_metric(&self) -> bool {
        self.inner.as_ref().is_some_and(|c| c.needs_train_metric())
    }

    fn holds_early_stopping(&mut self) -> bool {
        let inner = self.inner.as_mut().is_some_and(|c| c.holds_early_stopping());
        self.state.hold_pending.swap(false, std::sync::atomic::Ordering::Relaxed) | inner
    }

    fn name(&self) -> &str {
        "PesHoldCallback"
    }
}

/// Stage-1 model and bookkeeping of a residual warm start
/// (see [`GradientLSS::train_with_residual_warm_start`]).
pub struct WarmStartState<B: Backend> {
    stage1: GradientLSS<B>,
    loc_index: usize,
}

impl<B: Backend> WarmStartState<B> {
    /// The stage-1 (Gaussian) location model.
    pub fn stage1(&self) -> &GradientLSS<B> {
        &self.stage1
    }

    /// Column of the distribution's parameters that carries the location.
    pub fn loc_index(&self) -> usize {
        self.loc_index
    }

    /// `(n_rows, n_params)` raw-space offsets for `features`: the stage-1
    /// location in the `loc` column, zero elsewhere.
    pub fn offsets(&self, features: &ArrayView2<f64>, n_params: usize) -> Result<Array2<f64>> {
        let yhat = self.stage1.predict_transformed(features)?;
        let mut off = Array2::<f64>::zeros((features.nrows(), n_params));
        off.column_mut(self.loc_index).assign(&yhat.column(0));
        Ok(off)
    }
}

/// Which rows supply the residuals that the stage-2 start values are fitted
/// on in a residual warm start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidualSource {
    /// Validation residuals (honest: the stage-1 model did not see them).
    /// Falls back to training residuals when no validation set is given.
    #[default]
    Validation,
    /// In-sample training residuals. Over-fitted stage-1 residuals are more
    /// peaked than the truth, so this tends to under-estimate σ₀ and ν₀.
    Train,
}

/// Options for [`GradientLSS::train_with_residual_warm_start`].
pub struct ResidualWarmStart<B: Backend> {
    /// Backend parameters for the stage-1 Gaussian fit; `None` reuses the
    /// stage-2 parameters.
    pub stage1_params: Option<B::Params>,
    /// Residuals the stage-2 start values are fitted on.
    pub residuals: ResidualSource,
    /// `max_iter` handed to the distribution's `calculate_start_values` on the residuals.
    pub start_value_iters: usize,
}

impl<B: Backend> Default for ResidualWarmStart<B> {
    fn default() -> Self {
        Self {
            stage1_params: None,
            residuals: ResidualSource::Validation,
            start_value_iters: 50,
        }
    }
}

/// First word of a saved file that marks the v2 layout (warm start present).
/// A legacy file starts with its bincode state length, which can never be
/// `u64::MAX`, so old models keep loading through the legacy branch.
const SAVE_FORMAT_V2_SENTINEL: u64 = u64::MAX;
const SAVE_FORMAT_V2: u32 = 2;

impl<B: Backend> GradientLSS<B> {
    pub fn new(dist: Arc<dyn Distribution>) -> Self {
        Self {
            dist,
            model: None,
            start_values: None,
            start_values_preset: false,
            trained_with_row_offsets: false,
            warm_start: None,
            param_early_stopping: ParamEarlyStoppingSetting::Auto,
            param_freeze_log: None,
            param_es_triggers: None,
            warmup_rounds: 0,
            stopping_row_cap: None,
            hessian_normalization: HessianNormalization::Off,
            gradient_regularization: None,
            _backend: PhantomData,
        }
    }

    /// Enable per-parameter early stopping (see [`ParamEarlyStopping`]) with
    /// this configuration for subsequent training runs that have a
    /// validation set (overrides the default recipe).
    pub fn with_param_early_stopping(mut self, pes: ParamEarlyStopping) -> Self {
        self.param_early_stopping = ParamEarlyStoppingSetting::Custom(pes);
        self
    }

    /// Train only the location for the first `n` rounds: every other
    /// parameter's gradients and Hessians are zeroed until round `n`
    /// (`Distribution::location_param_index`; no-op for distributions
    /// without a location coordinate). Default 0.
    pub fn with_warmup_rounds(mut self, n: usize) -> Self {
        self.warmup_rounds = n;
        self
    }

    pub fn set_warmup_rounds(&mut self, n: usize) {
        self.warmup_rounds = n;
    }

    /// Make the *stopping decision* robust to single extreme rows: the
    /// monitored metric (early stopping, best iteration, the per-parameter
    /// revert test) softens each row's NLL above `m + k`, with `m` the
    /// median row NLL of that evaluation, to `m + k + ln(1 + ℓ − m − k)`.
    /// Gradients and any later scoring are unaffected; `valid_history` and
    /// `best_score` report the softened metric. Unit-free (differences of
    /// log densities). Univariate NLL only; otherwise ignored. Default off.
    pub fn with_stopping_row_cap(mut self, k: f64) -> Self {
        self.set_stopping_row_cap(Some(k));
        self
    }

    pub fn set_stopping_row_cap(&mut self, k: Option<f64>) {
        if let Some(k) = k {
            assert!(k.is_finite() && k >= 0.0, "stopping row cap must be finite and >= 0");
        }
        self.stopping_row_cap = k;
    }

    /// The configured stopping row cap (copied to `hyper_opt`'s fold models).
    pub fn stopping_row_cap(&self) -> Option<f64> {
        self.stopping_row_cap
    }

    /// Express the booster's absolute regularisation (`min_child_weight`,
    /// `lambda`, `gamma`) in "rows' worth of information" per parameter: each
    /// parameter's gradient and Hessian columns are multiplied by a constant
    /// κⱼ = 1 / mean per-row Hessian (see [`HessianNormalization`]). The
    /// unregularised Newton step and `eta` are unchanged; leaf values become
    /// −Σg / (Σh + λ/κⱼ). Training-only; trees and predictions keep their
    /// meaning. Default `Off`.
    pub fn with_hessian_normalization(mut self, mode: HessianNormalization) -> Self {
        self.hessian_normalization = mode;
        self
    }

    pub fn set_hessian_normalization(&mut self, mode: HessianNormalization) {
        self.hessian_normalization = mode;
    }

    /// Opt-in gradient-regularised Newton boosting (see [`GradientRegularization`]).
    pub fn with_gradient_regularization(mut self, reg: Option<GradientRegularization>) -> Self {
        self.gradient_regularization = reg;
        self
    }

    pub fn set_gradient_regularization(&mut self, reg: Option<GradientRegularization>) {
        self.gradient_regularization = reg;
    }

    pub fn gradient_regularization(&self) -> Option<GradientRegularization> {
        self.gradient_regularization
    }

    /// The configured [`HessianNormalization`] (copied to `hyper_opt`'s fold models).
    pub fn hessian_normalization(&self) -> HessianNormalization {
        self.hessian_normalization
    }

    /// κⱼ = 1 / mean per-row Hessian of `gh`; 1 where that mean is not a
    /// positive finite number.
    fn hessian_kappa(gh: &GradientsAndHessians) -> Vec<f64> {
        (0..gh.hessians.ncols())
            .map(|j| {
                let col = gh.hessians.column(j);
                let (sum, n) = col.iter().filter(|v| v.is_finite()).fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
                let m = if n > 0 { sum / n as f64 } else { f64::NAN };
                if m.is_finite() && m > 0.0 { 1.0 / m } else { 1.0 }
            })
            .collect()
    }

    /// Gradients/Hessians with every row at the same raw values `v`.
    fn constant_gh(&self, v: &Array1<f64>, labels: &Array1<f64>) -> Result<GradientsAndHessians> {
        let n_targets = self.dist.n_targets().max(1);
        let n = labels.len() / n_targets;
        let mut preds = Array2::<f64>::zeros((n, v.len()));
        for mut row in preds.rows_mut() {
            row.assign(v);
        }
        let t = self.dist.transform_params(&preds.view());
        if n_targets == 1 {
            let target = ResponseData::Univariate(&labels.view());
            self.dist.compute_gradients_and_hessians_with_transformed(&preds.view(), &t.view(), &target, None)
        } else {
            let reshaped = labels.view().into_shape_with_order((n, n_targets)).map_err(GradientLSSError::from)?;
            let target = ResponseData::Multivariate(&reshaped);
            self.dist.compute_gradients_and_hessians_with_transformed(&preds.view(), &t.view(), &target, None)
        }
    }

    /// The monitored (stopping) metric: [`Self::metric_on_transformed`], or
    /// its tail-softened form when a stopping row cap is set.
    fn stopping_metric(&self, transformed: &Array2<f64>, labels: &Array1<f64>) -> f64 {
        let Some(k) = self.stopping_row_cap else {
            return self.metric_on_transformed(transformed, labels);
        };
        let n = labels.len();
        if self.dist.n_targets() != 1 || !matches!(self.dist.loss_fn(), LossFn::Nll) || transformed.nrows() != n {
            return self.metric_on_transformed(transformed, labels);
        }
        let mut rows: Vec<f64> = (0..n)
            .map(|i| -self.dist.log_prob(&transformed.row(i).to_vec(), &[labels[i]]))
            .filter(|l| !l.is_nan())
            .collect();
        if rows.is_empty() {
            return f64::NAN;
        }
        let mut sorted = rows.clone();
        let mid = sorted.len() / 2;
        let (_, m, _) = sorted.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        let thr = *m + k;
        for l in rows.iter_mut() {
            if *l > thr {
                *l = thr + (1.0 + (*l - thr)).ln();
            }
        }
        rows.iter().sum::<f64>() / rows.len() as f64
    }

    /// Disable per-parameter early stopping, including the default recipe.
    pub fn without_param_early_stopping(mut self) -> Self {
        self.param_early_stopping = ParamEarlyStoppingSetting::Off;
        self
    }

    /// `Some` = this configuration, `None` = off (use
    /// [`Self::set_param_early_stopping_setting`] to restore `Auto`).
    pub fn set_param_early_stopping(&mut self, pes: Option<ParamEarlyStopping>) {
        self.param_early_stopping = match pes {
            Some(p) => ParamEarlyStoppingSetting::Custom(p),
            None => ParamEarlyStoppingSetting::Off,
        };
    }

    pub fn set_param_early_stopping_setting(&mut self, setting: ParamEarlyStoppingSetting) {
        self.param_early_stopping = setting;
    }

    pub fn param_early_stopping_setting(&self) -> &ParamEarlyStoppingSetting {
        &self.param_early_stopping
    }

    /// The configuration a training run with a validation set will use.
    pub fn effective_param_early_stopping(&self) -> Option<ParamEarlyStopping> {
        match &self.param_early_stopping {
            ParamEarlyStoppingSetting::Off => None,
            ParamEarlyStoppingSetting::Custom(p) => Some(p.clone()),
            ParamEarlyStoppingSetting::Auto => self
                .dist
                .default_param_early_stopping_params()
                .map(|ps| ParamEarlyStopping::recommended(Some(ps))),
        }
    }

    /// Freeze rounds chosen by per-parameter early stopping in the last run.
    pub fn param_freeze_log(&self) -> Option<&ParamFreezeLog> {
        self.param_freeze_log.as_ref()
    }

    /// Every round at which the per-parameter revert test fired, per
    /// parameter, in the last run (with `anneal > 0` a parameter can fire
    /// several times before it freezes).
    pub fn param_early_stopping_triggers(&self) -> Option<&Vec<Vec<usize>>> {
        self.param_es_triggers.as_ref()
    }

    /// Pre-set the unconditional start values so the NEXT `train` call skips
    /// the L-BFGS start-value fit. Intended for callers that retrain on the
    /// same labels repeatedly (e.g. `hyper_opt` reuses each CV fold's start
    /// values across trials — they depend only on the fold's train labels),
    /// and for OFFSET models: the built-in start is fit to the labels alone,
    /// so with per-row offsets (`train_with_callbacks_and_offsets`) the
    /// boosting starts `mean(offset)` too high in the location column. A
    /// caller that knows the link can pass the offset-aware intercept (for
    /// a Poisson under `exp` with offsets `o_i`: `ln(Σ y_i / Σ exp(o_i))`).
    /// Starting that far off is not merely slow — when the fitted mean sits
    /// several times above every label, each row's gradient is proportional
    /// to its hessian, no split has positive gain under an L2 penalty, and
    /// LightGBM reports the fit FINISHED at round 0 with zero trees (XGBoost
    /// emits constant stumps and walks the intercept down instead).
    /// The preset applies to exactly one train call and is saved with the
    /// model like a computed start value, so prediction adds it back.
    pub fn preset_start_values(&mut self, start_values: Array1<f64>) {
        self.start_values = Some(start_values);
        self.start_values_preset = true;
    }

    /// Save the model to a file.
    ///
    /// Layout: `[state_len u64][bincode state][backend model]`. A model
    /// trained with a residual warm start instead starts with the sentinel
    /// `u64::MAX`, a format version, the stage-1 model as a length-prefixed
    /// blob (same layout, recursively), and then the legacy layout — see
    /// `write_to`. Old files load unchanged.
    pub fn save(&self, path: &str) -> Result<()> {
        let mut file = File::create(path).map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        self.write_to(&mut file)
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let model = self
            .model
            .as_ref()
            .ok_or(GradientLSSError::ModelNotTrained)?;
        let io = |e: std::io::Error| GradientLSSError::IoError(e.to_string());

        if let Some(ws) = &self.warm_start {
            w.write_all(&SAVE_FORMAT_V2_SENTINEL.to_le_bytes()).map_err(io)?;
            w.write_all(&SAVE_FORMAT_V2.to_le_bytes()).map_err(io)?;
            w.write_all(&(ws.loc_index as u64).to_le_bytes()).map_err(io)?;
            // The stage-1 model as a self-contained blob; a length prefix is
            // needed because a backend's loader may read to the end of its
            // reader (LightGBM does).
            let mut blob = Vec::new();
            ws.stage1.write_to(&mut blob)?;
            w.write_all(&(blob.len() as u64).to_le_bytes()).map_err(io)?;
            w.write_all(&blob).map_err(io)?;
        }

        let state = SerializableState {
            dist: self.dist.clone_box(),
            start_values: self.start_values.clone(),
        };
        let state_bytes = bincode::serialize(&state)
            .map_err(|e| GradientLSSError::SerializationError(e.to_string()))?;
        w.write_all(&(state_bytes.len() as u64).to_le_bytes()).map_err(io)?;
        w.write_all(&state_bytes).map_err(io)?;
        model.save_to_writer(w)
    }

    /// Load the model from a file (either layout, see [`save`]).
    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        Self::read_from(&mut file)
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let io = |e: std::io::Error| GradientLSSError::IoError(e.to_string());
        let mut u64_buf = [0u8; 8];
        r.read_exact(&mut u64_buf).map_err(io)?;
        let mut first = u64::from_le_bytes(u64_buf);

        let mut warm_start = None;
        if first == SAVE_FORMAT_V2_SENTINEL {
            let mut u32_buf = [0u8; 4];
            r.read_exact(&mut u32_buf).map_err(io)?;
            let version = u32::from_le_bytes(u32_buf);
            if version != SAVE_FORMAT_V2 {
                return Err(GradientLSSError::SerializationError(format!(
                    "unsupported GradientLSS save format version {version}"
                )));
            }
            r.read_exact(&mut u64_buf).map_err(io)?;
            let loc_index = u64::from_le_bytes(u64_buf) as usize;
            r.read_exact(&mut u64_buf).map_err(io)?;
            let blob_len = u64::from_le_bytes(u64_buf) as usize;
            let mut blob = vec![0u8; blob_len];
            r.read_exact(&mut blob).map_err(io)?;
            let stage1 = Self::read_from(&mut Cursor::new(blob))?;
            warm_start = Some(Box::new(WarmStartState { stage1, loc_index }));
            r.read_exact(&mut u64_buf).map_err(io)?;
            first = u64::from_le_bytes(u64_buf);
        }

        let state_len = first as usize;
        let mut state_bytes = vec![0u8; state_len];
        r.read_exact(&mut state_bytes).map_err(io)?;
        let state: SerializableState = bincode::deserialize(&state_bytes)
            .map_err(|e| GradientLSSError::SerializationError(e.to_string()))?;
        let model = B::Model::load_from_reader(r)?;

        Ok(Self {
            dist: Arc::from(state.dist),
            model: Some(model),
            start_values: state.start_values,
            start_values_preset: false,
            trained_with_row_offsets: false,
            warm_start,
            param_early_stopping: ParamEarlyStoppingSetting::Auto,
            param_freeze_log: None,
            param_es_triggers: None,
            warmup_rounds: 0,
            stopping_row_cap: None,
            hessian_normalization: HessianNormalization::Off,
            gradient_regularization: None,
            _backend: PhantomData,
        })
    }

    /// The residual warm start this model was trained with, if any.
    pub fn warm_start(&self) -> Option<&WarmStartState<B>> {
        self.warm_start.as_deref()
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
            self.start_values = Some(
                self.dist
                    .default_start_values(labels)
                    .unwrap_or_else(|| Array1::from_elem(self.n_params(), 0.5)),
            );
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
    /// PER-OBSERVATION metric (mean, not sum).
    ///
    /// Every raw scorer in this fork (`nll`, `crps_score`, `analytic_crps`)
    /// totals over observations; this one divides by the observation count
    /// before returning. Three reasons:
    ///
    /// * **Fold weighting.** `cv` averages per-fold scores. With sums, a fold
    ///   holding more rows silently counts for more — forward-chaining splits
    ///   are not exactly equal (e.g. 6213/6213/6215 on 24,854 rows), so the
    ///   last fold was slightly overweighted. Means weight folds equally.
    /// * **Warm-start priors survive a growing dataset.** Consumers cache a
    ///   search's best score and replay it into later searches. A summed score
    ///   scales with row count, so as the dataset grows the prior reads as
    ///   worse than it was and biases TPE AWAY from a good config. A mean is
    ///   scale-free.
    /// * **Comparability.** 13,485 and 2.16 described equally good models; only
    ///   the units differed from the other model families.
    ///
    /// Safe for every existing caller because they all compare metrics
    /// RELATIVELY: early stopping across rounds on one validation set
    /// (`is_significant_improvement` uses a relative delta), median pruning
    /// across trials, and hyper-parameter ranking within a search — all
    /// unchanged by a constant divisor.
    pub(crate) fn metric_on_transformed(&self, transformed: &Array2<f64>, labels: &Array1<f64>) -> f64 {
        let n_obs = {
            let t = self.dist.n_targets().max(1);
            (labels.len() / t).max(1)
        };
        self.metric_total_on_transformed(transformed, labels) / n_obs as f64
    }

    /// [`Self::metric_on_transformed`] as a WEIGHTED mean over observations:
    /// `Σ wᵢ ℓᵢ / Σ wᵢ` with `ℓᵢ` the row's NLL (`−log_prob`) or analytic CRPS
    /// — the configured loss, row by row. With unit weights it equals the
    /// unweighted metric (NaN rows are skipped like `par_nansum`). Falls back
    /// to the unweighted metric where no per-row form exists (multivariate
    /// targets, sample-based CRPS) or when the weights do not fit the rows.
    pub(crate) fn metric_on_transformed_weighted(
        &self,
        transformed: &Array2<f64>,
        labels: &Array1<f64>,
        weights: &[f64],
    ) -> f64 {
        let n = labels.len();
        let per_row_crps = matches!(self.dist.loss_fn(), LossFn::Crps);
        if self.dist.n_targets() != 1
            || weights.len() != n
            || transformed.nrows() != n
            || (per_row_crps && !self.dist.has_analytic_crps())
        {
            return self.metric_on_transformed(transformed, labels);
        }
        let (mut num, mut den) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            let w = weights[i];
            if !(w.is_finite() && w > 0.0) {
                continue;
            }
            let params: Vec<f64> = transformed.row(i).to_vec();
            let loss = if per_row_crps {
                match self.dist.analytic_crps(&params, &[labels[i]]) {
                    Some(v) => v,
                    None => return f64::NAN,
                }
            } else {
                -self.dist.log_prob(&params, &[labels[i]])
            };
            if loss.is_nan() {
                continue;
            }
            num += w * loss;
            den += w;
        }
        if den > 0.0 { num / den } else { f64::NAN }
    }

    /// The raw total the scorers produce, before per-observation averaging.
    fn metric_total_on_transformed(&self, transformed: &Array2<f64>, labels: &Array1<f64>) -> f64 {
        match self.dist.loss_fn() {
            LossFn::Nll => self.nll_on_transformed(transformed, labels),
            LossFn::Crps => {
                let n_targets = self.dist.n_targets();
                // Analytic CRPS when available: the metric then agrees EXACTLY
                // with the quantity the FD gradients descend, instead of being
                // a 30-sample estimate of it (which would make early stopping
                // noisier than the training signal it monitors).
                if n_targets == 1 && self.dist.has_analytic_crps() {
                    let n = labels.len();
                    if n == 0 {
                        return f64::NAN;
                    }
                    // SUM here; `metric_on_transformed` divides by the
                    // observation count on the way out.
                    let mut total = 0.0;
                    for i in 0..n {
                        let params: Vec<f64> = transformed.row(i).to_vec();
                        match self.dist.analytic_crps(&params, &[labels[i]]) {
                            Some(v) => total += v,
                            None => return f64::NAN,
                        }
                    }
                    return total;
                }
                let samples = self.dist.sample(&transformed.view(), 30, 123);
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
        freeze: Option<Vec<Option<usize>>>,
        pes: Option<Arc<ParamEsState>>,
    ) -> impl Fn(&Array2<f64>, &Array1<f64>, Option<&Array1<f64>>) -> Result<GradientsAndHessians> + '_
    {
        let n_targets = self.dist.n_targets();
        let warmup = self.warmup_rounds;
        let loc_index = self.dist.location_param_index();
        // One objective call per boosting round, so a call counter is the
        // round index for the per-parameter freeze schedule
        // (`TrainConfig::param_freeze_rounds`).
        let round = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hnorm = self.hessian_normalization;
        let grn = self.gradient_regularization;
        let kappa: Arc<std::sync::Mutex<Option<Vec<f64>>>> = Arc::new(std::sync::Mutex::new(None));
        move |predictions, labels, weights| {
            let this_round = round.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
            let mut gh = if n_targets == 1 {
                let target = ResponseData::Univariate(&labels.view());
                self.dist.compute_gradients_and_hessians_with_transformed(
                    &predictions.view(),
                    &transformed.view(),
                    &target,
                    w.as_ref(),
                )?
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
                )?
            };
            if let Some(reg) = &grn {
                reg.apply(&mut gh);
            }
            if hnorm != HessianNormalization::Off {
                let mut k = kappa.lock().unwrap();
                if k.is_none() {
                    *k = Some(match hnorm {
                        HessianNormalization::Start => Self::hessian_kappa(&gh),
                        _ => {
                            let target_arr;
                            let target = if n_targets == 1 {
                                ResponseData::Univariate(&labels.view())
                            } else {
                                let n_samples = labels.len() / n_targets;
                                target_arr = labels
                                    .view()
                                    .into_shape_with_order((n_samples, n_targets))
                                    .map_err(GradientLSSError::from)?;
                                ResponseData::Multivariate(&target_arr)
                            };
                            let (_, marginal) = self.dist.calculate_start_values(&target, 50)?;
                            Self::hessian_kappa(&self.constant_gh(&marginal, labels)?)
                        }
                    });
                }
                let k = k.as_ref().expect("set above");
                for (j, &kj) in k.iter().enumerate().take(gh.gradients.ncols()) {
                    if kj != 1.0 {
                        gh.gradients.column_mut(j).mapv_inplace(|g| g * kj);
                        gh.hessians.column_mut(j).mapv_inplace(|h| h * kj);
                    }
                }
            }
            if let Some(st) = &pes {
                let frozen = st.frozen_at.lock().unwrap();
                let scale = st.grad_scale.lock().unwrap();
                for (j, f) in frozen.iter().enumerate().take(gh.gradients.ncols()) {
                    if f.is_some() {
                        gh.gradients.column_mut(j).fill(0.0);
                        gh.hessians.column_mut(j).fill(0.0);
                    } else if scale[j] != 1.0 {
                        gh.gradients.column_mut(j).mapv_inplace(|g| g * scale[j]);
                    }
                }
            }
            if this_round < warmup
                && let Some(loc) = loc_index
            {
                for j in 0..gh.gradients.ncols() {
                    if j != loc {
                        gh.gradients.column_mut(j).fill(0.0);
                        gh.hessians.column_mut(j).fill(0.0);
                    }
                }
            }
            if let Some(fr) = &freeze {
                for (j, m) in fr.iter().enumerate().take(gh.gradients.ncols()) {
                    if let Some(m) = m
                        && this_round >= *m
                    {
                        gh.gradients.column_mut(j).fill(0.0);
                        gh.hessians.column_mut(j).fill(0.0);
                    }
                }
            }
            Ok(gh)
        }
    }

    /// One validation-round update of [`ParamEarlyStopping`]: compare the
    /// current loss with the loss after reverting each eligible parameter's
    /// last `window` rounds; freeze the parameters whose revert helps.
    fn param_early_stopping_step(&self, st: &ParamEsState, margin: &Array2<f64>, labels: &Array1<f64>, loss: f64) {
        let round = st.valid_round.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut hist = st.history.lock().unwrap();
        hist.push_back(margin.clone());
        if hist.len() > st.window + 1 {
            hist.pop_front();
        }
        if hist.len() < st.window + 1 || !loss.is_finite() {
            return;
        }
        let old = hist.front().expect("non-empty");
        let mut frozen = st.frozen_at.lock().unwrap();
        let mut scale = st.grad_scale.lock().unwrap();
        let mut next_check = st.next_check.lock().unwrap();
        let mut streak = st.streak.lock().unwrap();
        for j in 0..margin.ncols().min(st.eligible.len()) {
            if !st.eligible[j] || frozen[j].is_some() || round < next_check[j] {
                continue;
            }
            let mut reverted = margin.clone();
            reverted.column_mut(j).assign(&old.column(j));
            let t = self.dist.transform_params(&reverted.view());
            let l_rev = self.stopping_metric(&t, labels);
            if l_rev < loss {
                streak[j] += 1;
                if streak[j] >= st.patience {
                    st.triggers.lock().unwrap()[j].push(round + 1);
                    if st.hold_early_stopping {
                        st.hold_pending.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    streak[j] = 0;
                    let s_new = scale[j] * st.anneal;
                    if st.anneal <= 0.0 || s_new < PES_ANNEAL_FREEZE_BELOW {
                        frozen[j] = Some(round + 1);
                    } else {
                        // Trees from round+1 on use the new rate; test again
                        // once a whole window of them is in the margin.
                        scale[j] = s_new;
                        next_check[j] = round + st.window;
                    }
                }
            } else {
                streak[j] = 0;
            }
        }
    }

    fn create_metric_fn(
        &self,
        cache: TransformCache,
        pes: Option<Arc<ParamEsState>>,
    ) -> impl Fn(&Array2<f64>, &Array1<f64>) -> f64 + '_ {
        move |predictions, labels| {
            // Non-finite margins are replaced with the start values, as in Python.
            let cleaned = self.clean_predictions(predictions);
            let cleaned_preds = cleaned.as_ref().unwrap_or(predictions);
            let transformed = self.dist.transform_params(&cleaned_preds.view());
            let loss = self.stopping_metric(&transformed, labels);

            if let Some(st) = &pes
                && labels.len() == st.valid_labels.len()
                && labels == &st.valid_labels
            {
                self.param_early_stopping_step(st, cleaned_preds, labels, loss);
            }

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
        self.train_with_callbacks_and_offsets(
            train_data, valid_data, params, config, callbacks, None, None,
        )
    }

    /// Two-stage **residual warm start** for location–scale(–shape) families.
    ///
    /// Start values fitted on the marginal of `y` are wrong for a conditional
    /// model: σ₀ is the marginal sd (too large once the trees explain
    /// variance) and, for StudentT, the df start is the marginal MLE ν → ∞
    /// where the df score vanishes — the shape tree then never learns
    /// conditional heavy tails. This method instead
    /// 1. fits a Gaussian location model (`stage1_params`, else `params`) on
    ///    the same data with early stopping on `valid_data`,
    /// 2. fits the distribution's unconditional start values on the
    ///    *residuals* `y − ŷ(x)` (validation residuals by default), and
    /// 3. trains this model with those start values and the stage-1 location
    ///    as a per-row offset in the `loc` column.
    /// Prediction recomputes the offsets from the stage-1 model, so `predict`
    /// works as usual; `save`/`load` persist the stage-1 model.
    ///
    /// Requires a univariate distribution with a parameter named `"loc"` under
    /// the `Identity` response (Gaussian, StudentT, Laplace, Logistic, Gumbel,
    /// Cauchy). Measured on StudentT (research/gradientlss/reports/003): better
    /// held-out NLL than the marginal start on 6 of 7 datasets (0.004–0.13
    /// nats), worse by 0.017 on data whose conditional law is exactly Gaussian.
    /// `cv`/`hyper_opt` do not use it.
    pub fn train_with_residual_warm_start(
        &mut self,
        train_data: &mut B::Dataset,
        mut valid_data: Option<&mut B::Dataset>,
        params: B::Params,
        config: TrainConfig,
        opts: ResidualWarmStart<B>,
    ) -> Result<TrainingResult> {
        if !self.dist.is_univariate() {
            return Err(GradientLSSError::InvalidParameter(
                "residual warm start supports univariate distributions only".to_string(),
            ));
        }
        let loc_index = self
            .dist
            .params()
            .iter()
            .position(|p| p.name == "loc" && p.response_fn == ResponseFn::Identity)
            .ok_or_else(|| {
                GradientLSSError::InvalidParameter(format!(
                    "residual warm start needs a parameter named \"loc\" with the Identity \
                     response (Gaussian, StudentT, Laplace, Logistic, Gumbel, Cauchy); {} has none",
                    self.dist.name()
                ))
            })?;

        // Stage 1: Gaussian location model on the same rows.
        let stage1_dist: Arc<dyn Distribution> =
            Arc::new(Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true));
        // Stage 1 keeps its cycle-3 behaviour: no per-parameter early stopping.
        let mut stage1 = GradientLSS::<B>::new(stage1_dist).without_param_early_stopping();
        let mut stage1_params = opts.stage1_params.unwrap_or_else(|| params.clone());
        stage1_params.set_n_dist_params(2);
        stage1.train_with_callbacks(
            train_data,
            valid_data.as_deref_mut(),
            stage1_params,
            config.clone(),
            None::<&mut crate::backend::HistoryCallback>,
        )?;

        let x_tr = train_data.get_features()?;
        let yhat_tr = stage1.predict_transformed(&x_tr.view())?.column(0).to_owned();
        let valid_fit: Option<(Array1<f64>, Array1<f64>)> = match valid_data.as_deref() {
            Some(v) => {
                let x = v.get_features()?;
                let yh = stage1.predict_transformed(&x.view())?.column(0).to_owned();
                Some((v.get_labels()?, yh))
            }
            None => None,
        };

        // Stage 2: start values from residuals.
        let resid: Array1<f64> = match (opts.residuals, &valid_fit) {
            (ResidualSource::Validation, Some((yv, yhv))) => yv - yhv,
            _ => &train_data.get_labels()? - &yhat_tr,
        };
        let (_, sv) = self
            .dist
            .calculate_start_values(&ResponseData::Univariate(&resid.view()), opts.start_value_iters)?;
        self.preset_start_values(sv);

        let n_params = self.dist.n_params();
        let offsets_of = |yh: &Array1<f64>| {
            let mut o = Array2::<f64>::zeros((yh.len(), n_params));
            o.column_mut(loc_index).assign(yh);
            o
        };
        let off_tr = offsets_of(&yhat_tr);
        let off_va = valid_fit.as_ref().map(|(_, yh)| offsets_of(yh));

        let (_, result) = self.train_with_callbacks_and_offsets(
            train_data,
            valid_data,
            params,
            config,
            None::<&mut crate::backend::HistoryCallback>,
            Some(off_tr.view()),
            off_va.as_ref().map(|a| a.view()),
        )?;
        self.warm_start = Some(Box::new(WarmStartState { stage1, loc_index }));
        Ok(result)
    }

    /// [`train_with_callbacks`] with a per-row boosting OFFSET.
    ///
    /// `row_offsets` is `(n_train_rows, n_params)` in RAW (pre-response)
    /// parameter space and is added on top of the unconditional start values
    /// as each row's base margin: the trees then learn adjustments around a
    /// per-row prior instead of around one global constant. The classic use is
    /// a location column holding `link⁻¹(prior expected value)` — e.g.
    /// `ln(expected count)` under an `Exp` response — with zeros elsewhere.
    /// `valid_row_offsets` must accompany `valid_data`. Predictions from a
    /// model trained this way need the row's offset back:
    /// [`predict_with_offset`] / [`predict_transformed_with_offset`].
    ///
    /// `cv` / `hyper_opt` do not thread offsets (they call the plain train);
    /// a tuned run wanting offsets must supply them per fold itself.
    pub fn train_with_callbacks_and_offsets<C: TrainingCallback>(
        &mut self,
        train_data: &mut B::Dataset,
        valid_data: Option<&mut B::Dataset>,
        params: B::Params,
        config: TrainConfig,
        callbacks: Option<&mut C>,
        row_offsets: Option<ArrayView2<'_, f64>>,
        valid_row_offsets: Option<ArrayView2<'_, f64>>,
    ) -> Result<((), TrainingResult)> {
        // CRPS gradients are finite differences through `sample` with a fixed
        // seed — only meaningful when the sampler is smooth in its parameters
        // (torch rsample analogue). Rejection/discrete samplers would turn each
        // accept/reject flip into a 1/(2ε)-amplified garbage gradient, so fail
        // fast instead (Python fails the same way: no rsample → error).
        if self.dist.loss_fn() == LossFn::Crps
            && !self.dist.has_reparameterizable_sampler()
            // An analytic CRPS sidesteps the sampler entirely, so the
            // rejection/discrete-sampling objection does not apply.
            && !self.dist.has_analytic_crps() {
            return Err(GradientLSSError::InvalidParameter(format!(
                "LossFn::Crps is not supported for {}: its sampler uses rejection sampling or \
                 discrete draws, which are not differentiable (torch has no rsample for it \
                 either). Use LossFn::Nll instead.",
                self.dist.name()
            )));
        }

        let labels = train_data.get_labels()?;
        // A preset (see `preset_start_values`) is consumed by exactly this one
        // train call; the caller vouches that the preset values were computed
        // on these labels.
        let use_preset =
            std::mem::take(&mut self.start_values_preset) && self.start_values.is_some();
        if !use_preset {
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
        }

        // Start values are handed to the backend below, which sets them as the
        // base margin (XGBoost) or seeds the accumulated margins (LightGBM) on
        // both the train and validation sets — no separate init-score pass.

        // Shared per-run transform cache — see `TransformCache`.
        let transform_cache: TransformCache = Arc::new(Mutex::new(Vec::with_capacity(2)));

        // Per-parameter early stopping needs the validation labels to tell
        // validation metric calls from training ones.
        let n_dist = self.dist.n_params();
        let pes_state: Option<Arc<ParamEsState>> = match (self.effective_param_early_stopping(), valid_data.as_deref()) {
            (Some(pes), Some(vd)) => {
                if pes.window == 0 || pes.patience == 0 || !(0.0..1.0).contains(&pes.anneal) {
                    return Err(GradientLSSError::InvalidParameter(
                        "ParamEarlyStopping: window and patience must be ≥ 1 and anneal in [0, 1)".to_string(),
                    ));
                }
                let mut eligible = vec![pes.params.is_none(); n_dist];
                if let Some(ps) = &pes.params {
                    for &j in ps {
                        if j >= n_dist {
                            return Err(GradientLSSError::InvalidParameter(format!(
                                "ParamEarlyStopping: parameter {j} out of range ({n_dist} parameters)"
                            )));
                        }
                        eligible[j] = true;
                    }
                }
                Some(Arc::new(ParamEsState {
                    window: pes.window,
                    patience: pes.patience,
                    anneal: pes.anneal,
                    streak: Mutex::new(vec![0; n_dist]),
                    eligible,
                    valid_labels: vd.get_labels()?,
                    history: Mutex::new(std::collections::VecDeque::with_capacity(pes.window + 1)),
                    frozen_at: Mutex::new(vec![None; n_dist]),
                    grad_scale: Mutex::new(vec![1.0; n_dist]),
                    next_check: Mutex::new(vec![0; n_dist]),
                    triggers: Mutex::new(vec![Vec::new(); n_dist]),
                    hold_early_stopping: pes.hold_early_stopping,
                    hold_pending: std::sync::atomic::AtomicBool::new(false),
                    valid_round: std::sync::atomic::AtomicUsize::new(0),
                }))
            }
            _ => None,
        };

        let objective = self.create_objective_fn(
            Arc::clone(&transform_cache),
            config.param_freeze_rounds.clone(),
            pes_state.clone(),
        );
        let metric = self.create_metric_fn(transform_cache, pes_state.clone());
        let (model, result) = match pes_state.as_ref().filter(|st| st.hold_early_stopping) {
            Some(st) => {
                let mut hold = PesHoldCallback { inner: callbacks, state: Arc::clone(st) };
                B::Model::train_with_objective_and_callbacks(
                    &params,
                    train_data,
                    valid_data,
                    &config,
                    objective,
                    metric,
                    self.start_values.as_ref(),
                    row_offsets,
                    valid_row_offsets,
                    Some(&mut hold),
                )?
            }
            None => B::Model::train_with_objective_and_callbacks(
                &params,
                train_data,
                valid_data,
                &config,
                objective,
                metric,
                self.start_values.as_ref(),
                row_offsets,
                valid_row_offsets,
                callbacks,
            )?,
        };

        self.model = Some(model);
        self.trained_with_row_offsets = row_offsets.is_some();
        self.param_freeze_log = pes_state.as_ref().map(|st| st.frozen_at.lock().unwrap().clone());
        self.param_es_triggers = pes_state.map(|st| st.triggers.lock().unwrap().clone());
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

            // Score on the INTERNAL transformed parameters, which add the
            // model's start values (base margin) before transforming —
            // `predict_raw` omits them, so scoring its output transforms the
            // wrong point; and the user-facing `predict(Parameters)` output is
            // finalized (Mixture: softmaxed), which `nll` must not receive.
            // Mirrors hyper_opt.
            let preds = model.predict_transformed(&test_features)?;
            // Configured loss, not hardcoded NLL — see hyper_opt's fold
            // scoring for the failure mode a mismatch causes.
            let test_labels_owned = test_labels.to_owned();
            let score = model.metric_on_transformed(&preds, &test_labels_owned);
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

            // Score on the internal transformed parameters (start values
            // added, no user-facing finalize) — see `cv`.
            let preds = model.predict_transformed(&test_features.view())?;
            // Configured loss, not hardcoded NLL — see `cv`.
            let score = model.metric_on_transformed(&preds, &test_labels);
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

    /// Predict the INTERNAL transformed parameters: raw margin + start values,
    /// run through the per-parameter response functions — the representation
    /// `nll`/`log_prob`/`sample` consume. For most distributions this equals
    /// the user-facing `predict(Parameters)` output; for Mixture the user-facing
    /// form additionally softmaxes the mixing logits (see
    /// `Distribution::finalize_predicted_params`), which must NOT be fed back
    /// into `nll`. CV/hyper-opt scoring therefore uses this method.
    pub(crate) fn predict_transformed(&self, features: &ArrayView2<f64>) -> Result<Array2<f64>> {
        self.predict_transformed_with_offset(features, None)
    }

    /// [`predict_transformed`] for a model trained with per-row offsets: the
    /// same `(n_rows, n_params)` raw-space offsets the rows would have carried
    /// in training are added back before the response functions. A model that
    /// was trained with offsets in this process refuses to predict without
    /// them (it would silently drop every row's prior).
    pub fn predict_transformed_with_offset(
        &self,
        features: &ArrayView2<f64>,
        row_offsets: Option<ArrayView2<'_, f64>>,
    ) -> Result<Array2<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or(GradientLSSError::ModelNotTrained)?;
        // A residual-warm-started model carries its own offsets.
        if row_offsets.is_none()
            && let Some(ws) = &self.warm_start
        {
            let off = ws.offsets(features, self.dist.n_params())?;
            return self.predict_transformed_with_offset(features, Some(off.view()));
        }
        if self.trained_with_row_offsets && row_offsets.is_none() {
            return Err(GradientLSSError::InvalidParameter(
                "model was trained with row offsets; predict_with_offset needs them back"
                    .to_string(),
            ));
        }

        // `predict_raw` hands us an owned array — add the start values (and
        // offsets) in place rather than cloning a second full matrix.
        let mut predictions = model.predict_raw(features)?;
        if let Some(off) = row_offsets {
            if off.nrows() != predictions.nrows() || off.ncols() != predictions.ncols() {
                return Err(GradientLSSError::InvalidParameter(format!(
                    "row_offsets shape ({}, {}) does not match predictions ({}, {})",
                    off.nrows(),
                    off.ncols(),
                    predictions.nrows(),
                    predictions.ncols()
                )));
            }
            predictions += &off;
        }
        if let Some(ref start_vals) = self.start_values {
            for mut row in predictions.rows_mut() {
                for (j, val) in row.iter_mut().enumerate() {
                    *val += start_vals[j];
                }
            }
        }

        Ok(self.dist.transform_params(&predictions.view()))
    }

    pub fn predict(
        &self,
        features: &ArrayView2<f64>,
        pred_type: PredType,
        n_samples: usize,
        quantiles: &[f64],
        seed: u64,
    ) -> Result<PredictionOutput> {
        self.predict_with_offset(features, None, pred_type, n_samples, quantiles, seed)
    }

    /// [`predict`] with the per-row offsets a model trained through
    /// [`train_with_callbacks_and_offsets`] needs back.
    pub fn predict_with_offset(
        &self,
        features: &ArrayView2<f64>,
        row_offsets: Option<ArrayView2<'_, f64>>,
        pred_type: PredType,
        n_samples: usize,
        quantiles: &[f64],
        seed: u64,
    ) -> Result<PredictionOutput> {
        let params = self.predict_transformed_with_offset(features, row_offsets)?;

        match pred_type {
            PredType::Parameters => Ok(PredictionOutput::Parameters(
                self.dist.finalize_predicted_params(params),
            )),

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
                Ok(PredictionOutput::Expectiles(
                    self.dist.finalize_predicted_params(params),
                ))
            }

            PredType::Distribution => {
                // Return full distribution info for PDF/CDF evaluation.
                // NOTE: these are the INTERNAL transformed parameters (what
                // `log_prob`/`sample` consume for PDF/CDF evaluation), so no
                // finalize here — Mixture's block keeps raw mixing logits.
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
    use crate::backend::BackendDataset;
    #[cfg(feature = "xgboost")]
    use crate::backend::XGBoostBackend;
    use crate::backend::lightgbm_backend::LightGBMBackend;
    use crate::distributions::{
        Dirichlet, Expectile, Gaussian, LossFn, MVN, MVNLoRa, MVT, Stabilization,
    };
    use crate::utils::ResponseFn;
    use ndarray::array;
    use std::sync::Arc;

    /// Row offsets: all-zero offsets must reproduce the offset-free model
    /// exactly, and shifting every row's location offset by δ must move the
    /// predicted (identity-linked) Gaussian mean by exactly δ — the trees see
    /// identical gradients either way, only the margin differs.
    fn row_offset_contract<B: Backend>()
    where
        B::Dataset: BackendDataset,
    {
        let dist = || {
            Arc::new(Gaussian::new(
                Stabilization::None,
                ResponseFn::Exp,
                LossFn::Nll,
                false,
            )) as Arc<dyn Distribution>
        };
        // 400 rows: LightGBM's default min_data_in_leaf (20) makes no split at
        // all on a few dozen rows, leaving both models at the constant start
        // value — nothing to compare.
        let n = 400usize;
        let features: Array2<f64> =
            Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64 / 11.0);
        let labels: Array1<f64> = Array1::from_shape_fn(n, |i| 2.0 + features[[i, 0]] * 3.0 - features[[i, 1]]);
        let params = B::create_params(2);
        let mut config = TrainConfig::default();
        config.num_boost_round = 20;
        config.early_stopping_rounds = None;

        let mut plain = GradientLSS::<B>::new(dist());
        let mut d = B::Dataset::from_data(features.view(), labels.view()).unwrap();
        plain.train(&mut d, None, params.clone(), config.clone()).unwrap();
        let p_plain = plain.predict_transformed(&features.view()).unwrap();

        let zeros = Array2::<f64>::zeros((n, 2));
        let mut with_zero = GradientLSS::<B>::new(dist());
        let mut d = B::Dataset::from_data(features.view(), labels.view()).unwrap();
        with_zero
            .train_with_callbacks_and_offsets(
                &mut d,
                None,
                params.clone(),
                config.clone(),
                None::<&mut crate::backend::HistoryCallback>,
                Some(zeros.view()),
                None,
            )
            .unwrap();
        assert!(
            with_zero.predict_transformed(&features.view()).is_err(),
            "offset-trained model must refuse to predict without offsets"
        );
        let p_zero = with_zero
            .predict_transformed_with_offset(&features.view(), Some(zeros.view()))
            .unwrap();
        assert_eq!(p_plain, p_zero, "zero offsets must equal no offsets, bitwise");

        // δ on the location column (param 0; identity link for Gaussian loc).
        let delta = 1.5;
        let mut off = Array2::<f64>::zeros((n, 2));
        off.column_mut(0).fill(delta);
        let mut shifted = GradientLSS::<B>::new(dist());
        let mut d = B::Dataset::from_data(features.view(), labels.view()).unwrap();
        shifted
            .train_with_callbacks_and_offsets(
                &mut d,
                None,
                params,
                config,
                None::<&mut crate::backend::HistoryCallback>,
                Some(off.view()),
                None,
            )
            .unwrap();
        // Additivity: for ONE trained model the offset is added back in raw
        // space, so predicting with every location offset moved by +1 must move
        // the identity-linked Gaussian mean by exactly 1 and leave the scale
        // untouched. (A δ-shifted margin does NOT reproduce the offset-free
        // model after a fixed number of rounds — the trees only absorb the
        // shift at convergence — so that is deliberately not asserted.)
        let p_shift = shifted
            .predict_transformed_with_offset(&features.view(), Some(off.view()))
            .unwrap();
        let mut off_plus = off.clone();
        off_plus.column_mut(0).mapv_inplace(|v| v + 1.0);
        let p_plus = shifted
            .predict_transformed_with_offset(&features.view(), Some(off_plus.view()))
            .unwrap();
        for i in 0..n {
            assert!(
                ((p_plus[[i, 0]] - p_shift[[i, 0]]) - 1.0).abs() < 1e-9,
                "row {i}: loc moved by {} not 1.0",
                p_plus[[i, 0]] - p_shift[[i, 0]]
            );
            assert_eq!(p_plus[[i, 1]], p_shift[[i, 1]], "row {i}: scale must not move");
        }
        // And the prior genuinely entered training: the shifted model's raw
        // trees differ from the plain model's (they fit a different residual).
        let raw_shift = shifted
            .predict_transformed_with_offset(&features.view(), Some(zeros.view()))
            .unwrap();
        let max_diff = (0..n)
            .map(|i| (raw_shift[[i, 0]] - p_plain[[i, 0]]).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "[row_offset_contract] max |shifted-trees − plain| = {max_diff:.6}; plain loc[0..3] = {:?}; shifted loc[0..3] = {:?}",
            &p_plain.column(0).to_vec()[..3],
            &raw_shift.column(0).to_vec()[..3]
        );
        assert!(
            max_diff > 1e-6,
            "offset-trained trees must differ from the plain model's (max diff {max_diff})"
        );
    }

    #[cfg(feature = "xgboost")]
    #[test]
    fn row_offsets_xgboost_contract() {
        row_offset_contract::<XGBoostBackend>();
    }

    #[test]
    fn row_offsets_lightgbm_contract() {
        row_offset_contract::<LightGBMBackend>();
    }

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
            collect_train_metrics: false,
            param_freeze_rounds: None,
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

#[cfg(all(test, feature = "xgboost"))]
mod analytic_metric_probe {
    use super::*;
    use crate::backend::XGBoostBackend;
    use crate::distributions::{LossFn, Poisson, Stabilization};
    use crate::utils::ResponseFn;
    use ndarray::array;

    /// Does the metric path actually reach the analytic-CRPS mean branch for
    /// a Crps-configured Poisson? (2026-08-27 sweep showed sum-scale scores,
    /// implying it does not.)
    #[test]
    fn metric_uses_analytic_mean_for_poisson_crps() {
        let dist = Poisson::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);
        eprintln!("has_analytic (concrete): {}", dist.has_analytic_crps());
        let arced: std::sync::Arc<dyn Distribution> = std::sync::Arc::new(dist);
        eprintln!("has_analytic (dyn):      {}", arced.has_analytic_crps());
        let model = GradientLSS::<XGBoostBackend>::new(arced);
        // 4 rows, rate 3.0 each
        let transformed = array![[3.0_f64], [3.0], [3.0], [3.0]];
        let labels = array![2.0_f64, 3.0, 4.0, 1.0];
        let m = model.metric_on_transformed(&transformed, &labels);
        eprintln!("metric value for 4 rows: {m:.4}");
        // Must be the SUM of the per-row analytic values (fork convention).
        let want: f64 = (0..4)
            .map(|i| {
                model
                    .distribution()
                    .analytic_crps(&[3.0], &[labels[i]])
                    .expect("poisson analytic")
            })
            .sum::<f64>()
            / 4.0;
        // `metric_on_transformed` is a PER-OBSERVATION mean since 985cec7
        // (2026-08-31); this expectation was left at the raw sum.
        assert!((m - want).abs() < 1e-9, "metric {m} != analytic mean {want}");
    }
}

#[cfg(all(test, feature = "xgboost"))]
mod residual_warm_start_tests {
    use super::*;
    use crate::backend::{Backend, BackendDataset, XGBoostBackend};
    use crate::distributions::{Gamma, StudentT};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution as _, StudentT as RandT};

    /// y = 2 + 3x₀ − x₁ + 0.5·t₃ noise on 3 U(0,1) features (x₂ is noise).
    fn data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let t = RandT::new(3.0).unwrap();
        let mut state: u64 = 0x9E3779B97F4A7C15 ^ seed;
        let mut u = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let x = Array2::from_shape_fn((n, 3), |_| u());
        let y = Array1::from_shape_fn(n, |i| 2.0 + 3.0 * x[[i, 0]] - x[[i, 1]] + 0.5 * t.sample(&mut rng));
        (x, y)
    }

    fn config() -> TrainConfig {
        TrainConfig {
            num_boost_round: 60,
            early_stopping_rounds: Some(10),
            verbose: false,
            collect_train_metrics: false,
            param_freeze_rounds: None,
            seed: 1,
        }
    }

    #[test]
    fn warm_start_trains_predicts_without_offsets_and_round_trips() {
        let (x, y) = data(1500, 3);
        let (xt, yt) = data(400, 4);
        let (xv, yv) = data(300, 5);
        let mut dtr = <XGBoostBackend as Backend>::Dataset::from_data(x.view(), y.view()).unwrap();
        let mut dva = <XGBoostBackend as Backend>::Dataset::from_data(xv.view(), yv.view()).unwrap();

        let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(StudentT::default()));
        let res = model
            .train_with_residual_warm_start(
                &mut dtr,
                Some(&mut dva),
                XGBoostBackend::create_params(3),
                config(),
                ResidualWarmStart::default(),
            )
            .unwrap();
        assert!(res.n_iterations > 0);

        // start values came from residuals: location start near 0, not near mean(y) ≈ 3
        let sv = model.start_values().unwrap();
        assert!(sv[1].abs() < 0.5, "loc start {} should be a residual mean", sv[1]);
        // the df start is the capped residual fit, inside (2.5, 10]
        let nu0 = ResponseFn::ExpDf.apply_scalar(sv[0]);
        assert!(nu0 > 2.4 && nu0 <= 10.0 + 1e-9, "ν₀ = {nu0}");

        // predict without offsets == predict with the stage-1 offsets built by hand
        let p_auto = model.predict_transformed(&xt.view()).unwrap();
        let ws = model.warm_start().expect("warm start recorded");
        assert_eq!(ws.loc_index(), 1);
        let yhat = ws.stage1().predict_transformed(&xt.view()).unwrap();
        let mut off = Array2::<f64>::zeros((xt.nrows(), 3));
        off.column_mut(1).assign(&yhat.column(0));
        let p_manual = model.predict_transformed_with_offset(&xt.view(), Some(off.view())).unwrap();
        assert_eq!(p_auto, p_manual);

        // the location is learned (RMSE well below sd(y)) and the fit is sane
        let rmse = (p_auto.column(1).iter().zip(yt.iter()).map(|(m, y)| (m - y).powi(2)).sum::<f64>()
            / yt.len() as f64)
            .sqrt();
        // irreducible noise sd = 0.5·√3 ≈ 0.87; sd(y) ≈ 1.29
        assert!(rmse < 1.0, "rmse {rmse} vs sd(y) {}", yt.std(0.0));
        assert!(p_auto.iter().all(|v| v.is_finite()));

        // save / load keeps the stage-1 model: identical predictions, no offsets needed
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("warm.bin");
        model.save(path.to_str().unwrap()).unwrap();
        let loaded = GradientLSS::<XGBoostBackend>::load(path.to_str().unwrap()).unwrap();
        assert!(loaded.warm_start().is_some());
        let p_loaded = loaded.predict_transformed(&xt.view()).unwrap();
        assert_eq!(p_loaded, p_auto);

        // a plain model still writes/reads the legacy layout
        let mut plain = GradientLSS::<XGBoostBackend>::new(Arc::new(StudentT::default()));
        let mut d2 = <XGBoostBackend as Backend>::Dataset::from_data(x.view(), y.view()).unwrap();
        plain.train(&mut d2, None, XGBoostBackend::create_params(3), config()).unwrap();
        let path2 = dir.path().join("plain.bin");
        plain.save(path2.to_str().unwrap()).unwrap();
        let bytes = std::fs::read(&path2).unwrap();
        assert_ne!(u64::from_le_bytes(bytes[..8].try_into().unwrap()), u64::MAX);
        let loaded2 = GradientLSS::<XGBoostBackend>::load(path2.to_str().unwrap()).unwrap();
        assert!(loaded2.warm_start().is_none());
        assert_eq!(
            loaded2.predict_transformed(&xt.view()).unwrap(),
            plain.predict_transformed(&xt.view()).unwrap()
        );
    }

    #[test]
    fn warm_start_rejects_families_without_an_identity_location() {
        let (x, y) = data(300, 7);
        let y = y.mapv(|v| v.abs() + 0.1);
        let mut d = <XGBoostBackend as Backend>::Dataset::from_data(x.view(), y.view()).unwrap();
        let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(Gamma::default()));
        let err = model
            .train_with_residual_warm_start(&mut d, None, XGBoostBackend::create_params(2), config(), ResidualWarmStart::default())
            .unwrap_err();
        assert!(err.to_string().contains("loc"), "{err}");
    }
}

#[cfg(all(test, feature = "xgboost"))]
mod param_gradient_scale_booster_tests {
    use super::*;
    use crate::backend::{Backend, BackendDataset, XGBoostBackend};
    use crate::distributions::Gaussian;
    use crate::utils::ResponseFn;

    /// On the real booster a gradient multiplier c on parameter j scales that
    /// parameter's one-round update by exactly c (leaf = −Σ(c·g)/(Σh+λ)) and
    /// leaves the other parameter's trees bit-identical: one tree per target,
    /// so the scaled column never enters the other column's split search.
    #[test]
    fn multiplier_scales_one_round_update_exactly() {
        let n = 1200;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| ((i * 7 + j * 13) % 97) as f64 / 97.0);
        let y = Array1::from_shape_fn(n, |i| 1.0 + 2.0 * x[[i, 0]] + (0.3 + x[[i, 1]]) * (((i * 31) % 17) as f64 / 8.0 - 1.0));
        let xt = x.slice(s![..200, ..]).to_owned();
        let fit = |scales: Option<[f64; 2]>| {
            let mut d = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);
            if let Some(sc) = scales {
                d = d.with_param_gradient_scales(sc);
            }
            let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(d));
            let mut ds = <XGBoostBackend as Backend>::Dataset::from_data(x.view(), y.view()).unwrap();
            let config = TrainConfig { num_boost_round: 1, early_stopping_rounds: None, verbose: false, collect_train_metrics: false,
            param_freeze_rounds: None, seed: 1 };
            m.train(&mut ds, None, XGBoostBackend::create_params(2), config).unwrap();
            let sv = m.start_values().unwrap().clone();
            // raw one-round update = raw prediction − start value
            let mut raw = m.predict_transformed(&xt.view()).unwrap();
            // undo the response to get raw margins: loc identity, scale = exp(η)+ε
            for mut row in raw.rows_mut() {
                row[1] = (row[1] - 1e-6).ln();
                row[0] -= sv[0];
                row[1] -= sv[1];
            }
            (sv, raw)
        };
        let (sv1, u1) = fit(None);
        let (sv2, u2) = fit(Some([1.0, 0.5]));
        assert_eq!(sv1, sv2, "start values are independent of the multipliers");
        let mut max_loc_diff = 0.0f64;
        let mut max_scale_ratio_err = 0.0f64;
        let mut moved = 0;
        for i in 0..u1.nrows() {
            max_loc_diff = max_loc_diff.max((u1[[i, 0]] - u2[[i, 0]]).abs());
            if u1[[i, 1]].abs() > 1e-4 {
                moved += 1;
                max_scale_ratio_err = max_scale_ratio_err.max((u2[[i, 1]] / u1[[i, 1]] - 0.5).abs());
            }
        }
        assert!(moved > 100, "scale tree should have moved most rows");
        assert!(max_loc_diff < 1e-6, "loc update changed by {max_loc_diff}");
        assert!(max_scale_ratio_err < 2e-3, "scale update ratio off by {max_scale_ratio_err} (f32 booster)");
    }
}

#[cfg(all(test, feature = "xgboost"))]
mod param_freeze_tests {
    use super::*;
    use crate::backend::{Backend, BackendDataset, XGBoostBackend};
    use crate::distributions::Gaussian;

    /// Freezing the scale at round k leaves the scale column exactly where a
    /// k-round fit left it while the location keeps moving.
    #[test]
    fn frozen_parameter_stops_at_its_round() {
        let n = 800;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 13) % 97) as f64 / 97.0);
        let y = Array1::from_shape_fn(n, |i| 1.0 + 2.0 * x[[i, 0]] + (0.3 + x[[i, 1]]) * (((i * 31) % 17) as f64 / 8.0 - 1.0));
        let fit = |rounds: usize, freeze: Option<Vec<Option<usize>>>| {
            let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));
            let mut ds = <XGBoostBackend as Backend>::Dataset::from_data(x.view(), y.view()).unwrap();
            let config = TrainConfig { num_boost_round: rounds, early_stopping_rounds: None, verbose: false, collect_train_metrics: false, seed: 1, param_freeze_rounds: freeze };
            m.train(&mut ds, None, XGBoostBackend::create_params(2), config).unwrap();
            m.predict_transformed(&x.view()).unwrap()
        };
        let p10 = fit(10, None);
        let p30_frozen = fit(30, Some(vec![None, Some(10)]));
        let p30 = fit(30, None);
        // scale column: identical to the 10-round fit (null trees afterwards)
        let max_scale_diff = p10.column(1).iter().zip(p30_frozen.column(1).iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(max_scale_diff < 1e-5, "scale moved after freeze by {max_scale_diff}");
        // location column: kept learning (differs from the 10-round fit, matches the 30-round fit's direction)
        let moved = p10.column(0).iter().zip(p30_frozen.column(0).iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(moved > 1e-3, "location did not keep learning");
        let unfrozen_scale_moved = p10.column(1).iter().zip(p30.column(1).iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(unfrozen_scale_moved > 1e-3, "control: the unfrozen scale should move");
    }
}

#[cfg(all(test, feature = "xgboost"))]
mod param_early_stopping_tests {
    use super::*;
    use crate::backend::{Backend, BackendDataset, XGBoostBackend};
    use crate::distributions::Gaussian;

    fn data(n: usize, salt: usize) -> (Array2<f64>, Array1<f64>) {
        // homoscedastic noise: any learned σ(x) structure is over-fitting
        let x = Array2::from_shape_fn((n, 3), |(i, j)| (((i + salt) * (7 + 5 * j) + j * 13) % 101) as f64 / 101.0);
        let y = Array1::from_shape_fn(n, |i| {
            let u = (((i + salt) * 2654435761usize) % 1000) as f64 / 1000.0 - 0.5;
            3.0 * x[[i, 0]] + 0.5 * u
        });
        (x, y)
    }

    fn config(rounds: usize, freeze: Option<Vec<Option<usize>>>) -> TrainConfig {
        TrainConfig { num_boost_round: rounds, early_stopping_rounds: None, verbose: false, collect_train_metrics: false, seed: 1, param_freeze_rounds: freeze }
    }

    /// The dynamic freeze picks a round for the scale only, and the fit is
    /// identical to a static `param_freeze_rounds` freeze at that round.
    #[test]
    fn dynamic_freeze_equals_static_freeze_at_the_chosen_round() {
        let (xt, yt) = data(600, 0);
        let (xv, yv) = data(300, 7777);
        let fit = |pes: Option<ParamEarlyStopping>, freeze: Option<Vec<Option<usize>>>| {
            let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));
            m.set_param_early_stopping(pes);
            let mut tr = <XGBoostBackend as Backend>::Dataset::from_data(xt.view(), yt.view()).unwrap();
            let mut va = <XGBoostBackend as Backend>::Dataset::from_data(xv.view(), yv.view()).unwrap();
            let mut p = XGBoostBackend::create_params(2);
            p.set("max_depth", crate::backend::ParamValue::Int(5));
            p.set("eta", crate::backend::ParamValue::Float(0.3));
            m.train_with_callbacks(&mut tr, Some(&mut va), p, config(120, freeze), None::<&mut crate::backend::CallbackList>).unwrap();
            (m.predict_transformed(&xv.view()).unwrap(), m.param_freeze_log().cloned())
        };
        let (p_dyn, log) = fit(Some(ParamEarlyStopping::new(5, Some(vec![1]))), None);
        let log = log.expect("validation set given → log recorded");
        assert_eq!(log[0], None, "location is not eligible");
        let k = log[1].expect("the over-fitting scale must freeze");
        assert!(k < 120);
        let (p_static, log2) = fit(None, Some(vec![None, Some(k)]));
        assert!(log2.is_none());
        let diff = p_dyn.iter().zip(p_static.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(diff < 1e-9, "dynamic ≠ static freeze at {k}: {diff}");
        // patience delays (never advances) the freeze
        let (_, log_p) = fit(Some(ParamEarlyStopping::new(5, Some(vec![1])).with_patience(4)), None);
        if let Some(kp) = log_p.unwrap()[1] {
            assert!(kp >= k + 3, "patience 4 froze at {kp}, patience 1 at {k}");
        }
    }

    fn fit_pes(pes: Option<ParamEarlyStopping>, rounds: usize) -> GradientLSS<XGBoostBackend> {
        let (xt, yt) = data(600, 0);
        let (xv, yv) = data(300, 7777);
        let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));
        m.set_param_early_stopping(pes);
        let mut tr = <XGBoostBackend as Backend>::Dataset::from_data(xt.view(), yt.view()).unwrap();
        let mut va = <XGBoostBackend as Backend>::Dataset::from_data(xv.view(), yv.view()).unwrap();
        let mut p = XGBoostBackend::create_params(2);
        p.set("max_depth", crate::backend::ParamValue::Int(5));
        p.set("eta", crate::backend::ParamValue::Float(0.3));
        m.train_with_callbacks(&mut tr, Some(&mut va), p, config(rounds, None), None::<&mut crate::backend::CallbackList>).unwrap();
        m
    }

    /// Anneal: the first trigger is the freeze round of the freeze rule (same
    /// decisions until then); later triggers are a full window apart; the
    /// parameter freezes only on the trigger that takes the rate below 1/64
    /// (0.5^7 < 1/64 ≤ 0.5^6 → the 7th), and the fit differs from the freeze fit.
    #[test]
    fn anneal_cuts_the_rate_before_freezing() {
        let frz = fit_pes(Some(ParamEarlyStopping::new(5, Some(vec![1]))), 400);
        let k = frz.param_freeze_log().unwrap()[1].expect("freeze rule fires");
        assert_eq!(frz.param_early_stopping_triggers().unwrap()[1], vec![k]);
        let ann = fit_pes(Some(ParamEarlyStopping::new(5, Some(vec![1])).with_anneal(0.5)), 400);
        let trig = ann.param_early_stopping_triggers().unwrap()[1].clone();
        assert_eq!(trig[0], k);
        assert!(trig.windows(2).all(|w| w[1] - w[0] >= 5), "{trig:?}");
        match ann.param_freeze_log().unwrap()[1] {
            Some(f) => {
                assert_eq!(trig.len(), 7, "{trig:?}");
                assert_eq!(*trig.last().unwrap(), f);
            }
            None => assert!(trig.len() < 7),
        }
        assert!(trig.len() >= 2, "an over-fitting scale should trigger again: {trig:?}");
        let (xv, _) = data(300, 7777);
        let a = ann.predict_transformed(&xv.view()).unwrap();
        let b = frz.predict_transformed(&xv.view()).unwrap();
        assert!(a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    }

    /// With the hold, the global early stopping never fires within
    /// `early_stopping_rounds` of a trigger; without it, it can.
    #[test]
    fn hold_keeps_the_fit_alive_after_every_trigger() {
        let (xt, yt) = data(600, 0);
        let (xv, yv) = data(300, 7777);
        let es = 8;
        let fit = |hold: bool| {
            let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));
            m.set_param_early_stopping(Some(
                ParamEarlyStopping::new(5, Some(vec![1])).with_anneal(0.5).with_hold_early_stopping(hold),
            ));
            let mut tr = <XGBoostBackend as Backend>::Dataset::from_data(xt.view(), yt.view()).unwrap();
            let mut va = <XGBoostBackend as Backend>::Dataset::from_data(xv.view(), yv.view()).unwrap();
            let mut p = XGBoostBackend::create_params(2);
            p.set("max_depth", crate::backend::ParamValue::Int(5));
            p.set("eta", crate::backend::ParamValue::Float(0.3));
            let mut c = config(400, None);
            c.early_stopping_rounds = Some(es);
            let ((), r) = m
                .train_with_callbacks(&mut tr, Some(&mut va), p, c, None::<&mut crate::backend::CallbackList>)
                .unwrap();
            (r, m.param_early_stopping_triggers().unwrap()[1].clone())
        };
        let (r_hold, trig) = fit(true);
        assert!(!trig.is_empty());
        // triggers are recorded as "round + 1"; the patience counter restarts there
        let last = *trig.last().unwrap();
        assert!(r_hold.n_iterations >= last + es - 1 || r_hold.n_iterations == 400, "{} vs trigger {last}", r_hold.n_iterations);
        let (r_free, _) = fit(false);
        assert!(r_hold.n_iterations >= r_free.n_iterations);
        // the hold never makes the selected validation loss worse
        assert!(r_hold.best_score.unwrap() <= r_free.best_score.unwrap() + 1e-12);
    }

    /// During the warm-up only the location column moves: after exactly
    /// `n` rounds the predicted scale is still the (constant) start value.
    #[test]
    fn warmup_trains_only_the_location() {
        let (xt, yt) = data(400, 0);
        let fit = |rounds: usize, warmup: usize| {
            let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()))
                .without_param_early_stopping()
                .with_warmup_rounds(warmup);
            let mut tr = <XGBoostBackend as Backend>::Dataset::from_data(xt.view(), yt.view()).unwrap();
            m.train(&mut tr, None, XGBoostBackend::create_params(2), config(rounds, None)).unwrap();
            m.predict_transformed(&xt.view()).unwrap()
        };
        let spread = |c: ndarray::ArrayView1<f64>| {
            c.iter().cloned().fold(f64::MIN, f64::max) - c.iter().cloned().fold(f64::MAX, f64::min)
        };
        let p = fit(10, 10);
        assert!(spread(p.column(1)) < 1e-9, "scale moved during warm-up");
        assert!(spread(p.column(0)) > 1e-2, "location did not move");
        assert!(spread(fit(15, 10).column(1)) > 1e-6, "scale never starts");
        assert!(spread(fit(10, 0).column(1)) > 1e-6, "without warm-up the scale trains");
    }

    /// Proposal 030: the adaptive term is h̄ ρ^{1/4} per parameter (ρ = mean g² / h̄),
    /// zero at ρ ≤ 1 in the excess-only form, unit-free, and off means untouched.
    #[test]
    fn gradient_regularization_adds_the_documented_term() {
        use crate::distributions::GradientsAndHessians;
        let gh = |scale: f64| GradientsAndHessians {
            // column 0: mean g² = 16, h̄ = 1 → ρ = 16; column 1: mean g² = 0.25, h̄ = 1 → ρ = 0.25
            gradients: ndarray::array![[4.0 * scale, 0.5], [-4.0 * scale, -0.5]],
            hessians: ndarray::array![[0.5 * scale * scale, 1.5], [1.5 * scale * scale, 0.5]],
        };
        let mut a = gh(1.0);
        GradientRegularization { strength: 0.5, excess_only: false }.apply(&mut a);
        // 0.5 · 1 · 16^{1/4} = 1 and 0.5 · 1 · 0.25^{1/4}
        assert!((a.hessians[[0, 0]] - 1.5).abs() < 1e-12 && (a.hessians[[1, 0]] - 2.5).abs() < 1e-12);
        assert!((a.hessians[[0, 1]] - (1.5 + 0.5 * 0.25f64.powf(0.25))).abs() < 1e-12);
        let mut b = gh(1.0);
        GradientRegularization { strength: 0.5, excess_only: true }.apply(&mut b);
        // 0.5 · (2 − 1) on column 0, nothing on column 1 (ρ < 1)
        assert!((b.hessians[[0, 0]] - 1.0).abs() < 1e-12);
        assert_eq!(b.hessians.column(1), gh(1.0).hessians.column(1));
        // a location margin measured in other units (g ∝ 1/c, h ∝ 1/c²): the relative change is the same
        let mut c = gh(0.01);
        GradientRegularization { strength: 0.5, excess_only: false }.apply(&mut c);
        assert!((c.hessians[[0, 0]] / gh(0.01).hessians[[0, 0]] - 3.0).abs() < 1e-9);
        let mut d = gh(1.0);
        GradientRegularization { strength: 0.0, excess_only: false }.apply(&mut d);
        assert_eq!(d.hessians, gh(1.0).hessians);
        assert_eq!(a.gradients, gh(1.0).gradients, "gradients are never touched");

        // end to end: the option changes a fit and keeps it finite
        let (xt, yt) = data(400, 3);
        let fit = |reg: Option<GradientRegularization>| {
            let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default())).with_gradient_regularization(reg);
            let mut tr = <XGBoostBackend as Backend>::Dataset::from_data(xt.view(), yt.view()).unwrap();
            m.train(&mut tr, None, XGBoostBackend::create_params(2), config(15, None)).unwrap();
            m.predict_transformed(&xt.view()).unwrap()
        };
        let (off, on) = (fit(None), fit(Some(GradientRegularization { strength: 1.0, excess_only: false })));
        assert!(on.iter().all(|v| v.is_finite()));
        assert!((&off - &on).iter().any(|d| d.abs() > 1e-6));
    }

    /// With no tree regularisation (min_child_weight = lambda = 0) the
    /// normalisation leaves the Newton step unchanged; with the default
    /// regularisation it changes the fit.
    #[test]
    fn hessian_normalization_only_rescales_regularisation() {
        use crate::backend::{BackendParams, ParamValue};
        let (xt, yt0) = data(600, 3);
        let yt = yt0.mapv(|v| 50.0 * v); // location Hessians ≪ 1
        let fit = |mode: HessianNormalization, unregularised: bool| {
            let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()))
                .without_param_early_stopping()
                .with_hessian_normalization(mode);
            let mut p = XGBoostBackend::create_params(2);
            if unregularised {
                p.set("min_child_weight", ParamValue::Float(0.0));
                p.set("lambda", ParamValue::Float(0.0));
            }
            let mut tr = <XGBoostBackend as Backend>::Dataset::from_data(xt.view(), yt.view()).unwrap();
            m.train(&mut tr, None, p, config(20, None)).unwrap();
            m.predict_transformed(&xt.view()).unwrap()
        };
        let rel = |a: &Array2<f64>, b: &Array2<f64>| {
            (a - b).iter().zip(a.iter()).map(|(d, v)| d.abs() / v.abs().max(1.0)).fold(0.0, f64::max)
        };
        for mode in [HessianNormalization::Start, HessianNormalization::Marginal] {
            // f32 gradient staging in the booster leaves ~1e-3 of drift
            let unreg = rel(&fit(HessianNormalization::Off, true), &fit(mode, true));
            let reg = rel(&fit(HessianNormalization::Off, false), &fit(mode, false));
            eprintln!("{mode:?}: unregularised {unreg:.2e}, regularised {reg:.2e}");
            assert!(unreg < 1e-2, "{mode:?}: unregularised fit moved by {unreg}");
            assert!(reg > 10.0 * unreg, "{mode:?}: regularised change {reg} not above f32 drift {unreg}");
        }
    }

    /// The tail-robust stopping metric equals the plain metric when no row
    /// exceeds the cap, and bounds a single extreme row's influence.
    #[test]
    fn stopping_row_cap_softens_only_extreme_rows() {
        let m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));
        let n = 50;
        let mut t = Array2::zeros((n, 2));
        t.column_mut(1).fill(1.0);
        let mut y = Array1::from_shape_fn(n, |i| (i as f64 / n as f64) - 0.5);
        let plain = m.metric_on_transformed(&t, &y);
        let capped = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default())).with_stopping_row_cap(10.0);
        assert!((capped.stopping_metric(&t, &y) - plain).abs() < 1e-12);
        y[0] = 1.0e3; // NLL ≈ 5e5 nats
        let plain_out = m.metric_on_transformed(&t, &y);
        let robust_out = capped.stopping_metric(&t, &y);
        assert!(plain_out > 1e4);
        assert!(robust_out < plain + 1.0, "one row moved the robust metric by {}", robust_out - plain);
        assert!(robust_out > plain, "softening is monotone, the outlier still counts");
        // without a cap the stopping metric is the plain one
        assert_eq!(m.stopping_metric(&t, &y), plain_out);
    }

    /// `Auto` resolves to the distribution's measured recipe; `Off` and an
    /// explicit configuration override it.
    #[test]
    fn auto_setting_resolves_to_the_distribution_recipe() {
        let m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));
        assert_eq!(m.param_early_stopping_setting(), &ParamEarlyStoppingSetting::Auto);
        assert_eq!(
            m.effective_param_early_stopping(),
            Gaussian::default().default_param_early_stopping_params().map(|p| ParamEarlyStopping::recommended(Some(p)))
        );
        let m = m.without_param_early_stopping();
        assert_eq!(m.effective_param_early_stopping(), None);
        let custom = ParamEarlyStopping::new(7, Some(vec![1])).with_anneal(0.25);
        let m = m.with_param_early_stopping(custom.clone());
        assert_eq!(m.effective_param_early_stopping(), Some(custom));
        // a distribution without a recipe stays off under Auto
        let m = GradientLSS::<XGBoostBackend>::new(Arc::new(crate::distributions::Poisson::default()));
        assert_eq!(m.effective_param_early_stopping(), None);
    }

    #[test]
    fn trial_axes_override_the_base_configuration() {
        use serde_json::json;
        let base = ParamEarlyStopping::new(20, Some(vec![2])).with_patience(3);
        let mut t = std::collections::HashMap::new();
        assert_eq!(ParamEarlyStopping::from_trial_params(Some(&base), None, &t), Some(base.clone()));
        t.insert("__pes_window".to_string(), json!(50));
        t.insert("__pes_anneal".to_string(), json!(0.5));
        let got = ParamEarlyStopping::from_trial_params(Some(&base), None, &t).unwrap();
        assert_eq!((got.window, got.patience, got.anneal, got.params.clone()), (50, 3, 0.5, Some(vec![2])));
        // base off → recommended recipe on the distribution's default params
        let got = ParamEarlyStopping::from_trial_params(None, Some(vec![1]), &t).unwrap();
        assert_eq!((got.window, got.params), (50, Some(vec![1])));
        t.insert("__pes_window".to_string(), json!(0));
        assert_eq!(ParamEarlyStopping::from_trial_params(Some(&base), None, &t), None);
    }

    #[test]
    fn needs_a_valid_window_and_parameter_index() {
        let (x, y) = data(100, 0);
        for pes in [
            ParamEarlyStopping::new(0, None),
            ParamEarlyStopping::new(3, Some(vec![2])),
            ParamEarlyStopping::new(3, None).with_patience(0),
            ParamEarlyStopping::new(3, None).with_anneal(1.0),
        ] {
            let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default())).with_param_early_stopping(pes);
            let mut tr = <XGBoostBackend as Backend>::Dataset::from_data(x.view(), y.view()).unwrap();
            let mut va = <XGBoostBackend as Backend>::Dataset::from_data(x.view(), y.view()).unwrap();
            let r = m.train_with_callbacks(&mut tr, Some(&mut va), XGBoostBackend::create_params(2), config(3, None), None::<&mut crate::backend::CallbackList>);
            assert!(r.is_err());
        }
        // no validation set: silently inactive, no log
        let mut m = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()))
            .with_param_early_stopping(ParamEarlyStopping::new(3, None));
        let mut tr = <XGBoostBackend as Backend>::Dataset::from_data(x.view(), y.view()).unwrap();
        m.train(&mut tr, None, XGBoostBackend::create_params(2), config(3, None)).unwrap();
        assert!(m.param_freeze_log().is_none());
    }
}

#[cfg(all(test, feature = "lightgbm"))]
mod param_early_stopping_lightgbm_tests {
    use super::*;
    use crate::backend::lightgbm_backend::LightGBMBackend;
    use crate::backend::{Backend, BackendDataset};
    use crate::distributions::Gaussian;

    /// LightGBM drives the metric closure differently from XGBoost; the
    /// validation-call detection must still fire and the dynamic freeze must
    /// equal a static freeze at the chosen round.
    #[test]
    fn dynamic_freeze_equals_static_freeze_on_lightgbm() {
        let mk = |n: usize, salt: usize| {
            let x = Array2::from_shape_fn((n, 3), |(i, j)| (((i + salt) * (7 + 5 * j) + j * 13) % 101) as f64 / 101.0);
            let y = Array1::from_shape_fn(n, |i| {
                3.0 * x[[i, 0]] + 0.5 * ((((i + salt) * 2654435761usize) % 1000) as f64 / 1000.0 - 0.5)
            });
            (x, y)
        };
        let (xt, yt) = mk(600, 0);
        let (xv, yv) = mk(300, 7777);
        let fit = |pes: Option<ParamEarlyStopping>, freeze: Option<Vec<Option<usize>>>| {
            let mut m = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::default()));
            m.set_param_early_stopping(pes);
            let mut tr = <LightGBMBackend as Backend>::Dataset::from_data(xt.view(), yt.view()).unwrap();
            let mut va = <LightGBMBackend as Backend>::Dataset::from_data(xv.view(), yv.view()).unwrap();
            let mut p = LightGBMBackend::create_params(2);
            p.set("learning_rate", crate::backend::ParamValue::Float(0.3));
            p.set("min_data_in_leaf", crate::backend::ParamValue::Int(2));
            let config = TrainConfig { num_boost_round: 120, early_stopping_rounds: None, verbose: false, collect_train_metrics: false, seed: 1, param_freeze_rounds: freeze };
            m.train_with_callbacks(&mut tr, Some(&mut va), p, config, None::<&mut crate::backend::CallbackList>).unwrap();
            (m.predict_transformed(&xv.view()).unwrap(), m.param_freeze_log().cloned())
        };
        let (p_dyn, log) = fit(Some(ParamEarlyStopping::new(5, Some(vec![1]))), None);
        let k = log.expect("validation set given")[1].expect("the over-fitting scale must freeze on LightGBM");
        let (p_static, _) = fit(None, Some(vec![None, Some(k)]));
        let diff = p_dyn.iter().zip(p_static.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(diff < 1e-9, "dynamic ≠ static freeze at {k}: {diff}");
    }
}
