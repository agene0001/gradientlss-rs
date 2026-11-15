# Proposal 031 — applying the decisions of report 016 §6

Status: **run** 2026-11-14 (E1 ✓, E2 ✓, E3 ✓, E4 ✓). **Pre-registered** 2026-11-14, after Felix answered "yes to all of these" and before any
verification run. Decisions 3 and 4 were either/or questions; "yes" is read as the first option
of each (3: confirm on fresh data before recommending, proposal 032; 4: keep the option).

## Changes
| # | decision | change |
|---|---|---|
| 1 | Gumbel default | `Gumbel::new`: Damped curvature, `LocationScale` start, start-scale factor 100. Serde defaults stay Observed / Raw / 1 (old saved models load as trained). `.with_hessian_mode(Observed).with_start(Raw)` restores the old behaviour. |
| 2 | NB CRPS coordinates | `NegativeBinomial::new` uses `MeanDispersion` for `LossFn::Crps` as it does for the NLL; `new_mean_dispersion(_, Crps)` and `with_parameterization(MeanDispersion)` no longer fall back. The prototype method `with_mean_dispersion_crps()` (never committed) is removed: `with_parameterization` does it. `.with_parameterization(TotalCountProbs)` on `new(..)` restores the old CRPS path. Saved models carry their parameterization and load as trained. |
| 3 | subsampling axes | confirmation on fresh datasets: proposal 032. README advice follows its rule. |
| 4 | `GradientRegularization` | kept in the tree, documented as tested and not recommended, not in the prelude. No change. |

## Verification runs and predictions
Binary `/tmp/rb_c28` (built after the changes).
- **E1 (Gumbel).** `BENCH_LS_PORT=2` (the family built with `new` only), tag `c28_gb_default`, the
  11 pairs of 028 × 5 seeds: every cell equals `c26_gb_ls100` bit for bit (NLL, CRPS, validation
  NLL, best round).
- **E2 (NB, new default).** NB under `BENCH_LOSS=crps` with no switch, tag `c28_nb_crps`, 023's
  five sets × 5 seeds: every cell equals `c26_nbmd` bit for bit.
- **E3 (NB, opt-out).** The same with `BENCH_NB_CRPS_TCP=1` (`new(..).with_parameterization(
  TotalCountProbs)`), tag `c28_nb_crps_tcp`: every cell equals `c24_crps` bit for bit.
- **E4.** The NLL paths are untouched: NB and Gumbel-free cells are not re-run; the unit and
  integration suites pass, with the tests that pinned the old defaults rewritten to pin the new
  ones and the old ones under their opt-outs.

## Decision rule
E1, E2 or E3 failing on any cell: the change is not what was tested; find the difference before
anything else.

## Result (2026-11-14, report 016 §8)
- **E1 ✓.** `Gumbel::new` as is: 55/55 cells equal `c26_gb_ls100` bit for bit (NLL, CRPS,
  validation NLL, best round, RMSE).
- **E2 ✓.** NB under `LossFn::Crps` with no switch: 25/25 cells equal `c26_nbmd`.
- **E3 ✓.** `new(..).with_parameterization(TotalCountProbs)` under the CRPS loss: 25/25 cells
  equal `c24_crps`.
- **E4 ✓.** 267 unit and 129 integration tests pass (XGBoost); the LightGBM library build
  compiles. Rewritten to the new defaults: `location_scale::defaults_and_legacy_loading` (Gumbel
  Damped / LocationScale / factor 100; a file without the three fields loads as Observed / Raw /
  1), `gumbel::start_scale_factor_widens_only_the_matched_scale`,
  `negative_binomial::mean_dispersion_is_the_nll_default_and_opt_out_is_exact` (CRPS builds the
  MeanDispersion head; the opt-out restores the caller's response functions; a saved
  TotalCountProbs model loads as one). Five NB CRPS tests of the (total_count, probs) head now
  ask for that head explicitly. `with_mean_dispersion_crps()` is gone.
- Each verification stream wrote its own CSV, appended to `experiments_v2.csv` afterwards.

**Decision (by the rule).** E1–E3 hold on every cell: both default changes are what was tested.
Decision 3 is proposal 032. Decision 4 needed no change.
