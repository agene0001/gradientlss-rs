# Cycle 6 report — per-parameter early stopping: anneal, StudentT, hyper_opt, on by default

Date: 2026-11-10. Rows: `experiments_v2.csv` tags `c6*`; detail in `research_log.md` "Cycle 6";
matrix: `c6_matrix.py`. Proposal 008 (+ Addendum A, pre-registered before batch c6d).
Suites: 256 lib + 137 integration (both backends), 223 + 129 (xgboost); new tests listed in §4.

## 1. Measured limitation
Cycle 5's per-parameter freeze is permanent. A transient trigger leaves σ̂ stale (Gaussian
power_plant, W 50: +0.256). It was noisy on Gamma naval (one seed +0.146), untested on StudentT,
not tunable in `hyper_opt`, and opt-in.

## 2. Competing explanations / designs
| question | candidates | outcome |
|---|---|---|
| what to do on a trigger | freeze; anneal ×0.5; anneal ×0.25 | anneal ×0.5: the only one that is never materially worse |
| StudentT watched set | scale; df; both | both (annealed) — post hoc, replicated in 3 configs |
| why anneal sometimes "does nothing" | — | global ES (patience 50) fires while the scale is being annealed → falls back to the no-PES fit |

## 3. Math / rule
On a trigger for parameter j, its gradient column is multiplied by f (an effective
per-parameter learning rate, as in `with_param_gradient_scales`). The next test only runs
once W trees have been grown at the new rate, so the revert test compares like with like.
The parameter freezes when the multiplier falls below 1/64. f = 0 reproduces the cycle-5
freeze bit-for-bit (checked).

## 4. Implementation and tests
- `ParamEarlyStopping { window, params, patience, anneal }`, `with_anneal`, `recommended`,
  `from_trial_params`.
- `ParamEarlyStoppingSetting { Auto (default), Off, Custom }`.
- `GradientLSS::without_param_early_stopping`, `effective_param_early_stopping`,
  `param_early_stopping_triggers`.
- `Distribution::default_param_early_stopping_params` (default `None`):
  - Gaussian [scale], StudentT [df, scale], Gamma MeanShape [shape], NB MeanDispersion [log θ];
  - NLL only; legacy heads and every other family return `None`.
- Residual warm start: stage 1 is explicitly Off, so its behaviour is unchanged.
- hyper_opt:
  - fold models inherit the model's setting, and the `__pes_window/_patience/_anneal` axes
    override it;
  - those axes keep the fold-data cache.
- Tests:
  - anneal: the first trigger equals the freeze round, later triggers are ≥ W apart,
    7 triggers to freeze at 0.5, and the fit differs from freeze;
  - Auto, Off and Custom resolution;
  - trial-axis override;
  - invalid anneal is rejected;
  - hyper_opt: window 0 ≡ off, and window 3 changes the CV score;
  - LightGBM: dynamic freeze ≡ static freeze.
- Harness: `BENCH_PES=auto` reproduces the explicit arms bit-for-bit (`c6v_auto`, 6 cells).

## 5. Results (5 paired seeds, ΔNLL vs no per-parameter stopping; mean, worst seed)
| family / dataset | freeze W20 | **anneal .5 W20 (default)** |
|---|---|---|
| Gaussian kin8nm / power_plant / protein / naval | −.137 / −.172 / −.136 / −.267 | −.138 / −.143 / −.101 / **−.368** |
| Gaussian sims (3) | ≤ +.001 | 0 |
| Gamma MS kin8nm / power_plant / naval | −.139 / −.147 / −.013 (+.146) | −.143 / −.112 / **−.194 (−.112)** |
| Gamma MS synth_gamma | .000 (+.007) | +.001 (+.007) |
| NB MD bike_cnt / bike_casual | −.041 / −.020 | **−.060 / −.026** |
| NB MD sims (3) | ≤ +.001 | 0 |
| StudentT (init 0) gaussian_sim | scale: **+.387 (+.491)** | df+scale: +.002 (+.005) |
| StudentT (init 0) kin8nm / protein / naval | −.136 / −.128 / −.293 (both) | −.148 / −.112 / −.389 |
| StudentT (init 1) kin8nm / power_plant / protein / naval | scale: −.142 / −.144 / −.057 / −.078 | −.166 / −.132 / −.130 / −.177 |
| StudentT (warm API) kin8nm / power_plant / protein / naval | both: −.124 / −.166 / −.077 / −.336 | −.138 / −.165 / −.074 / −.340 |

- Calibration: 90 % coverage moves toward 0.90:
  - Gaussian kin8nm .82 → .89, naval .82 → .94, power_plant .90 → .94;
  - StudentT kin8nm .82 → .91;
  - NB bike_cnt .88 → .92.
- RMSE improves where the scale used to stop the fit early: power_plant 3.95 → 3.52,
  protein 4.69 → 4.33, bike_cnt 41.2 → 39.2. The sims are unchanged.
- Cost: 2–4× more rounds on the real datasets. Per-round overhead is within timing noise.
- Anneal at W 50 repairs the cycle-5 failure: power_plant −0.143 (5/5), where freeze gave +0.256.

Prediction scorecard:
- A1 ✓.
- A2 partial: anneal is ahead of freeze on naval, behind on power_plant/protein by 0.03.
- A3 ✗.
- T1 ✗.
- T2 ✗: freezing df hurts (default-init power_plant +0.208).
- T3 ✗: df+scale ≫ scale on protein.
- T4 partial.
- T5 ✓.
- D ✓ (anneal .5).
- H ✓.

Deviation, reported: the pre-registered StudentT default set was {scale}. The data chose
{df, scale}, which beat scale alone in all three configurations (protein by 0.06–0.10) and
was neutral on the sims.

## 6. Decision (landed)
- Per-parameter early stopping is **on by default** whenever a validation set is passed:
  W 20, patience 1, anneal 0.5, on the watched sets above.
  - It is the only uniform recipe within +0.005 of "none" on every dataset of the four
    families. Freeze fails StudentT gaussian_sim (+0.39); anneal .25 fails it too (+0.009).
  - Opt out with `.without_param_early_stopping()`.
  - Hard freeze stays available: `with_anneal(0.0)` (best on Gaussian protein/power_plant).
- `hyper_opt` can tune the window, patience and anneal factor.

## 7. Side finding (not fixed — needs Felix)
`StudentT::default()` (`initialize = false`, raw 0.5 start) fails on non-centred targets:
- On power_plant (y ≈ 450), RMSE is 448, cov90 is 1.000 and NLL is 7.52, against 2.78 with
  `initialize = true`.
- The t-score saturates for large residuals, so the location never leaves 0.5.
- Gaussian is unaffected (its score is linear in the residual), and the Gamma/NB orthogonal
  heads fit MLE starts anyway.
- Candidate fix: default StudentT start values to the closed-form (median, MAD-scaled,
  ν₀ = 10) or to `initialize = true`. That is a Python-parity / default question.

## 8. Next investigations (ranked)
1. StudentT start values (§7). This is the largest remaining defect measured anywhere in the
   harness.
2. Global early stopping should not end the fit while a parameter is mid-anneal. Delaying
   the ES counter after a trigger would remove anneal's "fallback" seeds, worth about
   0.03–0.04 on power_plant/protein.
3. naval hits the 2000-round cap under the default (the location keeps improving). Check
   with a larger cap.
4. Beta / Weibull orthogonality audit, then extend the default watched sets to them.
