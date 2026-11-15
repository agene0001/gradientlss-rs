# Proposal 026 — applying the decisions of report 015 §10

Status: **run** 2026-11-13 (D1 ✓, D2 ✗, D3 ✗, R1 ✗ by the letter, R2–R5 ✓; amended once before the CRPS run). **Pre-registered** 2026-11-13, after Felix approved all six decisions ("you can apply all
of these") and before any verification run. Where a decision was an either/or, the more complete
option is taken: fix (not error), helper (not documentation only), fix (not documentation only).

## Changes
| # | decision | change |
|---|---|---|
| 1 | four family defaults | `new()` for Cauchy and LogNormal: Damped + Location; Logistic and Laplace: Damped + LocationScale. Gumbel unchanged. Serde defaults stay Observed / Raw (old saved models load as trained). |
| 2 | CRPS training for large counts | Poisson and NB: the loss, its gradient and the metric Hessian are summed over a window that covers the distribution instead of stopping at 100 / 150 / 200 / 220 terms. Rows the old caps did not bind keep the old code path. |
| 3 | validation guard | opt-in helper next to `hyper_opt`: fit the tuned and the untuned configuration with validation early stopping, return the better. |
| 4 | two code leads | `spline_flow.rs` start fit without `target_cost(0.0)`; LogNormal rows with y ≤ 0: gradient 0, Hessian 1e-6 on both paths and in every curvature mode. |
| 5 | `dist_select` | ranks by `Distribution::marginal_fit` (new trait method, default = `calculate_start_values`); StudentT overrides it with the free ML fit (no start-df cap). |
| 6 | normalised tuning axis | README recommendation, with cycle 13's numbers and caveats. |

## Verification runs and predictions
**Defaults (1).** `BENCH_LS_PORT=2` builds each family with `new()` only. Tag `c23_default`:
the four families × the five unseen sets and × power_plant, kin8nm, naval, protein (LogNormal
without protein), 5 seeds.
- **D1.** Every cell reproduces the tested recipe's row bit for bit (`c18_port` on the unseen
  sets; `c15_d_loc` for Cauchy and LogNormal, `c15_d_ls` for Logistic and Laplace on the
  development sets).

**`dist_select` (5).** The marginal pass of proposal 018 re-run with the new binary (no training);
the trained models' rows are reused.
- **D2.** protein's marginal pick changes from Gaussian to StudentT in 5/5 seeds.
- **D3.** Marginal pick = best-on-test family in at least 27/45 cells (was 22/45), mean regret
  ≤ 0.040 (was 0.0545). It stays far behind validation selection (42/45, 0.0005).

**CRPS (2).** Proposal 023's protocol, CRPS arm only, new binary, tag `c24_crps`; NLL arm reused
(`c20_nll`).
- **R1 (nothing moves below the caps).** Poisson c_abalone and synth_pois and NB c_abalone and
  synth_nb_lo: test CRPS equal to `c20_crps` to 1e-6 relative.
- **R2 (Poisson trains).** bike_cnt and c_seoulbike: best round > 0 in 10/10 cells and test CRPS
  within 10 % of the NLL-trained model's (was +640 %, +562 %).
- **R3 (NB above the cap).** bike_cnt and c_seoulbike: the gap to NLL training falls below +50 %
  (was +116 %, +485 %).
- **R4 (NB is still not competitive).** NB stays worse than NLL training on all five sets: the
  fallback to (total count, probs) and the slow convergence below the cap are not addressed here.
- **R5.** No non-finite row in either family.

## Decision rule
- D1 fails on any cell: the default change is not what was tested; find the difference before
  anything else.
- R2 fails: revert the Poisson change and return an error for `LossFn::Crps` instead, as the
  either/or allowed. R3 fails: the same for NB.
- R4 holds: the README keeps "train on the NLL" for NB and says why.

## Not in scope
CRPS gradients for NB in (mean, dispersion) coordinates; Gumbel; the wide-space questions of 025.

## Amendment (2026-11-13, before the CRPS run; after two smoke cells)
Smoke cell, Poisson bike_cnt seed 0, sums fixed, metric Hessian kept: the model now trains but is
at the round cap with test CRPS 56.6 (NLL-trained mean 18.2; before the fix 134.8 at round 0). R2
would fail. Cause, from the formulas: the metric E_y[g²] is in units of CRPS² and grows like λ²/3
in the log rate; the curvature of the CRPS grows like λ^1.5/√π. The step g/M is too small by
about 0.6·√λ: 8× at λ = 190, 16× at 700. (At λ ≈ 5, where the metric was tuned, the two agree
within 1.3×.) NGBoost pairs this metric with a line search; here there is none.

**Change added.** For rows that take the large-count path (the ones the old caps cut), the
Hessian is the expected CRPS curvature, 2 Σ_k (∂F(k)/∂θ_j)² in margin space — the CRPS
counterpart of the Fisher information — floored at 0.05 as before. Rows under the old caps keep
the metric and the old code path bit for bit. Second smoke (seed 0): bike_cnt 18.80, c_seoulbike
77.66 (NLL-trained means 18.23, 76.46), both at the round cap.

R1–R5 stand as written. The decision rule stands: R2 or R3 failing on the full run means an error
for large counts instead.

## Result (2026-11-13, report 015 §11)
Library tests: 264 unit and 129 integration tests pass (XGBoost); the LightGBM build compiles.

**Defaults (1).** **D1 ✓.** 190/190 cells of `c23_default` equal the tested recipe's rows bit for
bit (NLL, CRPS, validation NLL, best round, RMSE).

**`dist_select` (5).** **D2 ✗, D3 ✗.** The picks did not change: best-on-test family in 22/45
cells, mean regret 0.0543 (was 22/45, 0.0545). On protein the free StudentT fit ties with Gaussian
(3.2232 each) and Gaussian is still listed first. The cap was an artefact (0.011 NLL per row on
Gaussian labels, unit test) but not the cause of the mis-ranking: the labels' marginal carries no
information about heavy-tailed conditional residuals. My diagnosis in report 015 §2 was wrong on
that point. The change is kept (it ranks the right quantity) and the documentation says shortlist.

**CRPS (2).** Test CRPS, 5 seeds; relative to the NLL-trained model.

| family | dataset | NLL-trained | before | now | now, relative | best round, min–max |
|---|---|---|---|---|---|---|
| Poisson | synth_pois | 1.654 | 1.669 | 1.668 | +0.8 % | 401–486 |
| Poisson | c_abalone | 1.2009 | 1.2005 | 1.2005 | −0.04 % | 155–309 |
| Poisson | bike_casual | 6.44 | 9.50 | 6.77 | +5.2 % | 1994–1999 |
| Poisson | bike_cnt | 18.23 | 134.8 | 19.40 | +6.5 % | 1905–1999 |
| Poisson | c_seoulbike | 76.46 | 505.6 | 80.01 | +4.7 % | 1792–1999 |
| NB | synth_nb_lo | 4.017 | 4.126 | 4.136 | +3.0 % | 182–289 |
| NB | c_abalone | 1.207 | 1.612 | 1.612 | +33.6 % | 1999 |
| NB | bike_casual | 6.07 | 6.56 | 6.29 | +3.7 % | 1162–1999 |
| NB | bike_cnt | 17.22 | 37.17 | 19.19 | +11.5 % | 1997–1999 |
| NB | c_seoulbike | 69.79 | 407.7 | 70.62 | +1.2 % ± 1.4 (2/5) | 579–1999 |

- **R1 ✗ by the letter.** c_abalone is identical in 10/10 cells. The two simulations are not
  (synth_pois up to 0.8 %, synth_nb_lo up to 1.2 % per cell; means 1.669 → 1.668 and
  4.126 → 4.136): they contain rows above the old caps, which now take the new path. My
  prediction assumed they did not.
- **R2 ✓.** Poisson trains (best round ≥ 1792 in 10/10 cells) and is within 10 % of NLL training.
- **R3 ✓.** NB above the cap: +11.5 % and +1.2 % (were +116 %, +485 %).
- **R4 ✓.** NB is worse than NLL training on all five means; c_seoulbike is inside the noise.
- **R5 ✓.** No non-finite row.
- CRPS training still beats NLL training by more than 1 % on 0/10 pairs. 29/50 cells are at the
  round cap.

**Decision (by the rule).** D1 holds: the default change stands. R2 and R3 hold: the CRPS change
stands (no error path needed). R4 holds: the README keeps "train on the NLL" and says the NB
coordinates are not addressed.
