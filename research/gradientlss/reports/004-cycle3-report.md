# Cycle 3 report — warm start productised, df step size, scale over-fitting

Date: 2026-11-09. Follows `003-cycle2-report.md`. Rows: `experiments_v2.csv` tags `c2f_*`,
`c2g_*`, `c3a_*`; detail in `research_log.md` "Cycle 3".

## 1. What Felix asked for, and what landed
- **ν₀ ≤ 10 start cap is now the StudentT default** (`StudentT::max_start_df = Some(10)`,
  `.without_max_start_df()` opts out). Better on protein/power_plant, neutral elsewhere.
- **Residual warm start is a crate feature**: `GradientLSS::train_with_residual_warm_start(
  train, valid, params, config, ResidualWarmStart { stage1_params, residuals, start_value_iters })`.
  Stage-1 Gaussian → per-row location offsets; the distribution's start values are fitted on
  validation residuals; `predict*` recompute the offsets; `save`/`load` persist the stage-1
  model in a versioned layout (legacy files unchanged). Trait additions:
  `BackendDataset::get_features`, `BackendParams::set_n_dist_params`. Tests cover
  auto-offset ≡ manual-offset predictions, save/load round-trip, legacy layout, rejection of
  families without an Identity `loc`. The API path reproduces the harness recipe exactly
  (tag `c2f_api` vs `c2d_warmvalid_cap10`: identical on 5 of 7 datasets, ≤ 0.008 on the two
  whose start now sits at the ν₀ ≥ 2.5 floor).
- **Per-parameter shrinkage** (proposal 005): `with_param_gradient_scales([..])` on Gaussian
  and StudentT — gradient-column multipliers = effective per-parameter learning rate.
  Default `None`.
- Suites: 220 lib (both backends), 129 / 137 integration (xgboost / full) green.

## 2. Measured limitations addressed
1. StudentT df frozen at a Gaussian start (cycle 2) → warm start + cap (above).
2. **Scale over-fitting**: without early stopping every family's held-out NLL explodes while
   σ̂ shrinks and coverage collapses (gaussian_sim warm-start trajectory: NLL 2.119 at R = 50 →
   3.1 at R = 400, 90 % coverage 0.91 → 0.71). The stopping round is dictated by the scale
   tree, which truncates location/df learning.

## 3. Competing explanations for the slow df on gaussian_sim (H-g)
| candidate | prediction | result |
|---|---|---|
| early stopping too early | df_med rises with R and NLL keeps improving past R = 60 | df rises (2.9 → 31) but NLL is minimal at R ≈ 50: stopping is *correct*; the scale over-fits first |
| gradient cancellation from unlearned σ(x) | df stalls | rejected — df moves monotonically |
| df curvature too large | df ×2 multiplier closes the +0.017 gap | **falsified**: −0.002 only |

## 4. Fair experiments — per-parameter shrinkage (Damped, η=0.05/d4, 5 paired seeds)
Gaussian, ΔNLL vs shared η:

| dataset | scale ×0.5 | scale ×0.25 | validation-selected c ∈ {1, .5, .25} |
|---|---|---|---|
| kin8nm | −0.036 ± 0.007 (5/5) | −0.081 ± 0.006 (5/5) | −0.081 (picks .25 ×5) |
| naval | +0.020 ± 0.029 | −0.107 ± 0.025 (5/5) | −0.107 (picks .25 ×5) |
| power_plant | −0.003 ± 0.011 | −0.046 ± 0.007 (5/5) | −0.046 |
| protein | −0.007 ± 0.004 | −0.022 ± 0.010 (4/5) | −0.022 |
| gaussian_sim | +0.007 (0/5) | +0.017 (0/5) | 0.000 (picks 1 ×5) |
| synth_t3 | +0.001 | +0.008 (0/5) | 0.000 (picks 1 ×5) |
| studentt_sim | 0.000 | +0.002 (0/5) | 0.000 |

Coverage at ×0.25: kin8nm 0.82 → 0.86, naval 0.81 → 0.83 (toward nominal 0.90); CRPS moves
with NLL. StudentT (warm start): scale ×0.5 −0.014/−0.016 on kin8nm/naval (5/5), ≈ 0 to
+0.005 elsewhere; validation selection −0.017/−0.018 there and 0 elsewhere.

Reading: real targets over-fit the scale; the simulations have strong, fast-learnable σ(x)
and don't. Validation selection resolves the sign flip with no loss anywhere — the knob is
strictly non-worse as a tuned hyperparameter.

## 5. Decision
- Ship `with_param_gradient_scales` as a documented knob; recommend tuning the scale factor
  on validation NLL over {1, 0.5, 0.25}. Not a default (dataset-dependent sign).
- Warm start: recommended StudentT recipe, now one method call; not a default (two-stage
  object; +0.017 on exactly-Gaussian conditional data).
- The gaussian_sim penalty of the warm start is not a df-speed problem; it is the price of a
  heavy-tailed start under a scale-driven stopping round. Acceptable for a heavy-tail recipe.

## 6. Next autonomous investigations (ranked)
1. Wire the scale-shrinkage factor (and `HessianMode`, warm start on/off) into `hyper_opt`'s
   search space so the validation selection above is automatic.
2. Scale regularisation that does not need tuning: a σ floor from validation residuals, or a
   separate early-stopping criterion per parameter (stop the scale tree earlier than the
   location tree) — needs backend support for freezing a target.
3. β-NLL-style row weighting (σ̂^{2β}) for the location tree — changes *which rows* drive
   splits; test against the shrinkage knob on kin8nm/protein.
4. Extend Damped curvature and the warm start to Gamma / NegativeBinomial dispersion.
5. Deterministic `par_nansum` for reproducible start values.

---

## Addendum (same day) — Felix's follow-ups: implementation scrutiny and the §6 items

### Scrutiny of the per-parameter shrinkage implementation
- **Booster-level test** (`multiplier_scales_one_round_update_exactly`): with scales [1, 0.5]
  the one-round raw update of the scale column is 0.5× the unscaled one on every row (f32
  error < 2e-3) and the location column is bit-identical. Start values unaffected. Caveat: with
  `multi_strategy=multi_output_tree` the targets would share trees and the separation would not
  hold.
- **gamboostLSS at source** (boost-R/gamboostLSS `R/mboostLSS.R`, `R/helpers.R`, man page):
  `nu` is "a (named) vector or a (named) list with separate values for each component",
  applied per component through `control$nu <- nu[[j]]`; `stabilize_ngradient` divides the
  negative gradient only (MAD floored 1e-4, L2 clamped [1e-4, 1e4]) — gamboostLSS has no
  Hessian, which is why the inherited MAD/L2 recipe misbehaves in a Newton booster; the
  noncyclic method advances all parameters tentatively and keeps the best inner loss; the
  cyclic method accepts a per-parameter `mstop`. Our `c_j` is the Newton analogue of `nu_j`;
  `TrainConfig::param_freeze_rounds` is the per-parameter `mstop`.

### Landed (all tested; 230 lib with both backends, 137/129 integration)
`TrainConfig::param_freeze_rounds` (new field) · `HyperOptConfig::trial_dist_fn` + `__freeze_<j>`
axes · `Gaussian/StudentT::with_beta_nll` (row weights σ̂^{2β}, normalised to mean 1) · Gamma
`HessianMode` (default Damped) · deterministic `par_sum`/`par_nansum` · start-value cost guard
against the all-NaN "cost 0" trap (fixed Gamma on y ≈ 450) · README updates.

### Results (Damped, η=0.05/d4, 5 paired seeds; ΔNLL vs the respective baseline)
| knob | kin8nm | naval | power_plant | protein | sims |
|---|---|---|---|---|---|
| Gamma Damped − Observed | −0.021 (5/5) | **−2.88** (5/5) | **−0.41** (5/5) | — | — |
| scale freeze, validation-selected k ∈ {∅,50,100,200} | **−0.157** (5/5) | 0 (picks ∅) | **−0.116** (5/5) | **−0.098** (5/5) | 0 |
| shrinkage, validation-selected c ∈ {1,.5,.25} | −0.081 | 0 | −0.046 | −0.022 | 0 |
| β-NLL, validation-selected β ∈ {0,.5,1} | −0.024 | −0.014 | +0.021 (noise) | −0.068 | ≤ +0.006 |
| β-NLL = 1, RMSE | −0.007 | 0 | **−0.21** | **−0.36** | +0.003…+0.006 |

Freeze also raises 90 % coverage toward nominal (kin8nm 0.82 → 0.95 at k=50, 0.89 at k=200)
and improves RMSE (protein −0.53, power_plant −0.52): once the scale stops, early stopping no
longer fires on the scale's over-fit and the location keeps training. naval is the
counter-example (its tiny σ(x) carries structure; any freeze is catastrophic), which is why
all three stay validation-tuned knobs rather than defaults.

### Decisions
- Gamma: `Damped` default confirmed (3/3 datasets).
- Scale freeze is the strongest scale-side lever found; expose through `hyper_opt`
  (`__freeze_<j>`), not as a default.
- β-NLL: keep; its reliable benefit is location accuracy.
- Open for cycle 4: Gamma's concentration learns too slowly from the marginal start
  (power_plant: coverage 1.0, needs α ≈ 12 000 from 330) — a Gamma warm start / shape-step
  study, mirroring the StudentT df work.
