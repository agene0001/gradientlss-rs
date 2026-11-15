# Cycle 2 report — StudentT degrees of freedom: why it never learns, and how to start it

Date: 2026-11-09 (same day as cycle 1). Base: working tree after cycle 1 with
`HessianMode::Damped` made the default (Felix's approval). Rows: `experiments_v2.csv`
tags `c2a_*` … `c2e_*`; console logs under `reports/c2*.txt`; detail in `research_log.md`
"Cycle 2".

## 1. Measured limitation
With fitted start values the StudentT models in cycle 1 were Gaussians in disguise: ν₀
landed at 2e7–1e9 on every real target and the df tree never moved (StudentT cells equal
the Gaussian cells to 4 decimals). Yet one accidental start at ν = 2 (Fisher mode) beat
those fits by 0.12–0.20 nats on protein and power_plant — so a learnable df is worth a
lot and the crate was not providing it.

## 2. Competing explanations and what killed them
| hypothesis | test | verdict |
|---|---|---|
| H-a: df moves, just slowly | fixed-round trajectories R=1…800, no early stopping | rejected — df does not move at all (score ≈ 0 at ν₀ ≈ 1e7) |
| H-b: df direction flips as σ shrinks | same trajectories | rejected — no non-monotone path |
| H-c: gradual run-off into the ν=2 dead zone | same | half right — the "collapse" is already present at R=1 |
| H-d: first-round leaf catapult from cancellation noise | per-mode Σg/Σh probe at ν₀ | rejected — root steps are 1e-6 in every mode |
| **truth** | `BENCH_DEBUG` at R=1 | the START VALUES differed by HessianMode: Fisher's pre-phase (cycle 1) ran the df into the ExpDf dead zone (ν=2) before σ adapted; Damped's ran it to ν→∞ |
| unconditional ν profile (scipy) | protein / power_plant / kin8nm | unimodal, **ν → ∞ is the marginal MLE** on all three; the ν=2 start was a worse unconditional fit that happened to be a better *conditional* start |

Core insight: start values from the *marginal* of y put σ₀ too high and ν₀ at ∞; at that
σ the df gradient points up, ν climbs to ~1e3 (information ≈ 1e-5/row) and is frozen by
the time the scale tree has shrunk σ. The conditional optimum has finite ν.

## 3. Fixes landed in `src/` (all tested; 218 lib + 129 integration green)
- Pre-phase curvature = `max(|analytic h|, FD second difference)` per coordinate: start
  values no longer depend on the training `HessianMode` (test
  `student_t_start_values_do_not_depend_on_hessian_mode`) and the df can no longer
  out-run σ into the dead zone; the exact analytic step is kept where FD noise would
  stall a saturated t location (`gaussian_start_values_are_scale_invariant` still holds
  at 7e-4 … 4.5e5).
- `StudentT::with_max_start_df(ν)` (experimental, non-default) and a hard **start
  floor ν₀ ≥ 2.5** (`MIN_START_DF_RAW = ln 0.5`): a start exactly at the ExpDf floor is a
  dead state (dν/dη ≈ 0) the booster can never leave; observed on naval residuals.
- `Damped` is now the default `HessianMode` (cycle-1 result, approved).

## 4. Experiments (StudentT, Damped, η=0.05/depth 4, 5 paired seeds, ΔNLL vs the fitted marginal start)
| recipe | protein | power_plant | kin8nm | naval | synth_t3 | studentt_sim | gaussian_sim |
|---|---|---|---|---|---|---|---|
| ν₀ ≤ 10 cap (c2b) | −0.067 ± 0.015 (5/5) | −0.052 ± 0.053 | −0.001 | −0.010 ± 0.017 | 0 | 0 | 0 |
| warm start, train residuals (c2c) | −0.124 ± 0.014 (5/5) | −0.130 ± 0.079 (5/5) | −0.008 ± 0.006 | −0.114 (ν₀ at floor) | −0.016 ± 0.002 (5/5) | −0.004 (5/5) | **+0.018 ± 0.002 (0/5)** |
| warm start, validation residuals (c2d) | −0.087 ± 0.026 (5/5) | −0.127 ± 0.077 (5/5) | **−0.020 ± 0.002 (5/5)** | −0.118 ± 0.025 (ν₀ at floor) | −0.017 ± 0.002 (5/5) | −0.004 (5/5) | +0.017 ± 0.005 (0/5) |
| … + ν₀ ≤ 10 | −0.115 (→ 2.779) | −0.130 | −0.018 | −0.118 | −0.017 | −0.004 | +0.017 |
| naval / gaussian_sim with the start floor (c2e) | — | — | — | _see §4a_ | — | — | _see §4a_ |

Absolute NLL, best StudentT recipe vs the Gaussian-Damped default: protein 2.78 vs 2.89,
power_plant 2.71 vs 2.81, kin8nm −0.51 vs −0.49, naval −5.87 vs −5.74, synth_t3 1.40 vs
1.47, studentt_sim 1.98 vs 2.03; gaussian_sim 2.13 vs 2.10. CRPS and PIT-KS move with NLL
(protein CRPS −0.12 to −0.15, PIT-KS 0.082 → 0.066–0.070). Coverage: 90% intervals cover
0.81–0.90; kin8nm under-covers in every recipe (a model-family issue).

Why gaussian_sim loses: its conditional law is exactly Gaussian but σ(x) ∈ [1,5] makes
the residual marginal a scale mixture, so ν₀ ≈ 3 is the *correct* unconditional answer
and the df tree climbs back too slowly before early stopping (≈ round 60). This is the
mirror image of the frozen-df problem and the open item for cycle 3.

### 4a. Start floor re-check (tag c2e_floor_warmvalid_cap10; warm start, validation residuals, ν₀ ≤ 10)
With the ν₀ ≥ 2.5 floor the naval warm start no longer sits at the ExpDf dead zone: predicted df
is learned (median 10–23, 5 % quantile 3.4–5.6 across seeds) at the same NLL (−5.866 vs −5.868),
so the naval gain is legitimate, not a degenerate ν = 2 state. gaussian_sim improves slightly
(2.125 vs 2.133; df median 3.4–7.6, still climbing too slowly to reach the Gaussian limit).

## 5. Decision
- **Recommend** the residual warm start (validation residuals, ν₀ ≤ 10) for StudentT:
  better than the marginal start on 6/7 datasets (0.004–0.13 nats) and better than the
  Gaussian-Damped default on 6/7; harness recipe `BENCH_WARM_START=1 BENCH_WARM_RESID=valid
  BENCH_T_START_DF_CAP=10`; crate API proposed in `proposals/004`. **Not** made a default:
  it needs a two-stage object and loses 0.017 on exactly-Gaussian data.
- Keep `with_max_start_df` non-default (neutral on light tails, cap30 slightly worse on naval).
- Land the pre-phase mode-independence fix and the ν₀ ≥ 2.5 floor (corrections).
- Second finding worth its own cycle: without early stopping every mode's test NLL
  eventually explodes as σ̂ keeps shrinking (scale overfitting; Observed worst, Fisher
  latest) — early stopping at ≈ 200 rounds is doing a lot of work.

## 6. Next autonomous investigations (ranked)
1. **df step size from a heavy-tailed start** (gaussian_sim): the df tree needs ~10×
   larger effective steps at ν ≈ 3; test a df-specific curvature floor / learning rate, or
   drop the `(ν−2)` chain factor via a Softplus-type df link without a dead zone.
2. **Scale overfitting / overconfidence** (all families): β-NLL-style down-weighting of
   the scale gradient, or a σ floor from the validation residuals.
3. Productise the warm start (proposal 004) and extend it to Gamma/NB dispersion.
4. Deterministic `par_nansum` (start-value reproducibility).
5. Extend Damped to the remaining families (LogNormal, Gamma, Laplace, Gumbel, Logistic,
   Weibull, NB-TotalCountProbs).
