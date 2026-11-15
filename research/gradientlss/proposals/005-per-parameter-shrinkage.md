# Proposal 005 — Per-parameter shrinkage (effective learning rate per distribution parameter)

Status: **proposed, pre-registered, not yet implemented** (cycle 3, 2026-11-09)

## Measured limitation
Without early stopping every mode's held-out NLL eventually explodes while RMSE keeps
improving and the mean predicted σ̂ keeps shrinking (cycle-2 trajectories: kin8nm all
modes by R = 800, protein Damped/Observed by R = 400). The scale tree over-fits
|residual| on the training rows — the model becomes over-confident — long before the
location tree over-fits. One shared learning rate for all parameters forces a single
stopping round to serve both; early stopping then truncates the location/df learning to
protect the scale. Related: with a heavy-tailed start the df tree may need *larger*
steps (cycle-3 H-g), the opposite adjustment.

## Proposed rule
Per-parameter multipliers `c_j` on the gradient column only, applied after stabilisation
and before the backend: `g_j ← c_j · g_j`, `h_j` unchanged. Since the leaf value is
`−Σg/(Σh+λ)`, this scales the Newton step of parameter j by `c_j` — an effective
per-parameter learning rate `η·c_j` — while leaving the Hessian-based split gain scale
and `min_child_weight` untouched (scaling `h` too would reduce to a pure learning-rate
change *and* alter splits). Default `c_j = 1` for all j (unchanged behaviour).

In GAMLSS boosting terms this is the per-parameter step length `ν_k` of gamboostLSS
(Mayr et al. 2012), which the R package exposes and the XGBoostLSS/LightGBMLSS ports do
not. It is **not** β-NLL: β-NLL re-weights rows by σ^{2β}; this re-weights parameters.
(β-NLL-style row weighting is a separate candidate; see Open.)

## Pre-registered predictions (Gaussian and StudentT, Damped, η=0.05/depth 4, 5 seeds)
1. Scale shrinkage `c_scale = 0.5` (others 1): with early stopping, held-out NLL improves
   or ties on kin8nm/naval/protein/power_plant (where the scale over-fits first) and 90 %
   coverage moves toward nominal; best_iter grows. Without early stopping the test-NLL
   blow-up is delayed by ≈ 2× in rounds.
2. `c_scale = 0.25` is not better than 0.5 on NLL (too slow) — a dose–response check.
3. df boost `c_df = 2` on StudentT from the warm start: gaussian_sim's penalty
   (+0.017) shrinks (df climbs faster) without hurting protein/power_plant by > 0.01.
4. Null: differences within ±0.005 everywhere → shared η is fine, reject.

## Implementation sketch
`Distribution` default method `param_gradient_scales(&self) -> Option<&[f64]>` (None =
all 1) with builders on Gaussian/StudentT (`with_param_shrinkage([..])`), applied in
`compute_gradients_and_hessians_with_transformed` after stabilisation. Serialised with
`#[serde(default)]`. Harness knob `BENCH_PARAM_SCALES=1,0.5` (per parameter order).

## Open
- Row-weighting alternative (β-NLL, Seitzer et al. 2022): multiply *all* columns of row i
  by σ̂_i^{2β}; changes which rows drive splits, not step sizes. Test after this.
- A validation-residual σ floor (never predict σ below α·s_resid) as a hard guard.
