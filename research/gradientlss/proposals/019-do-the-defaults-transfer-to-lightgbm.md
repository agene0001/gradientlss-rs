# Proposal 019 — do the cycle 1–13 defaults transfer to the LightGBM backend?

Status: **run** 2026-11-13 (L1 ✓, L2 ✓, L3 ✓). **Pre-registered** 2026-11-13, before the run (one smoke cell was seen while checking that the
LightGBM build works: kin8nm seed 0, Gaussian, Observed 0.716 against Damped −0.701). Skill:
`optimization-algorithm-research`.

## Question
Every measured cell of cycles 1–13 used `XGBoostBackend` (`examples/research_bench.rs` hard-coded it).
The defaults those cycles changed live in the distributions and in `GradientLSS`, so LightGBM users
get them too, unmeasured. LightGBM grows leaf-wise, has `min_data_in_leaf = 20` and `lambda_l2 = 0`
where XGBoost has none and λ = 1, so the curvature and regularisation findings need not carry over.

## Harness change
`research_bench` now names its backend through one alias (`Bk`): `--features xgboost` is unchanged,
`--features lightgbm` alone builds the same bench on LightGBM with the same parameter names (eta,
max_depth, min_child_weight are LightGBM aliases). Binaries `/tmp/rb_c15` (XGBoost) and
`/tmp/rb_c16_lgbm`.

## Protocol
Two recipes × two backends, 5 seeds, the usual splits, eta 0.05, depth 4, mcw 1, 2000 rounds /
patience 50, nothing tuned.

| family | `old` (before cycle 1) | `new` (today's defaults) |
|---|---|---|
| Gaussian | Observed, no per-parameter stopping | Damped, PES Auto |
| StudentT | Observed, raw start, no PES | Damped, median start, PES Auto |
| Gamma | concentration/rate, Observed, no PES | mean/shape, Damped, PES Auto |
| NB | total_count/probs, no PES | mean/dispersion, location start, PES Auto |
| Poisson | ReLU link, Observed, no PES | Exp link, ȳ start, Damped, PES Auto |

Datasets: Gaussian kin8nm, power_plant, protein, naval, gaussian_sim; StudentT those plus
studentt_sim, synth_t3; Gamma kin8nm, naval, power_plant, synth_gamma; NB bike_cnt, bike_casual,
synth_nb_lo; Poisson bike_cnt, synth_pois. 21 (family, dataset) pairs.
Tags `c16_<xgb|lgb>_<old|new>`; script `run_c16.sh`; summary `pair_tags.py`-style paired differences.

## Predictions
- **L1 (the rule the defaults were held to).** On LightGBM, `new` − `old` test NLL is ≤ +0.005 on
  every pair (failures: non-finite rows count as losses).
- **L2 (same direction).** Where |`new` − `old`| > 0.01 on XGBoost, the sign on LightGBM agrees on at
  least 80 % of pairs.
- **L3 (size).** The mean gain on LightGBM is at least half the mean gain on XGBoost.

## Decision rule
No library change. A pair that fails L1 is a LightGBM-specific regression to investigate (cycle 17
candidate) and is listed in the README's backend notes. If L1–L3 hold, the README can say the
defaults were measured on both backends.

## Cost
420 fits; two single-thread streams; about an hour on a loaded machine.

## Result (2026-11-13, report 015 §3)
- 0/21 pairs break the +0.005 rule on XGBoost and on LightGBM; the sign agrees on 19/19 pairs with
  |Δ| > 0.01; mean Δ −1.006 (XGBoost), −1.095 (LightGBM).
- LightGBM with the old defaults: StudentT power_plant non-finite in 5/5 seeds; Gamma power_plant and
  synth_gamma in 1/5 each. No failure with today's defaults.
- Caveat: LightGBM Gaussian protein −0.015 ± 0.126 (3/5); seed 2 is 3.915 (new) against 3.455 (old).
- L1 ✓, L2 ✓, L3 ✓. **Decision.** No change; the defaults are not XGBoost-specific.
