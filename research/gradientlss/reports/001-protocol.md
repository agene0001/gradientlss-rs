# Benchmark protocol — cycle 1 (Fisher vs Observed curvature)

Reproduce (release, XGBoost backend; `mount-build` first on this machine):

```bash
BENCH_TAG=c1_eta05_d4 cargo run --release --features xgboost --example research_bench
BENCH_TAG=c1_eta10_d6 BENCH_ETA=0.1 BENCH_DEPTH=6 cargo run --release --features xgboost --example research_bench
```

Rows are appended to `research/gradientlss/experiments.csv`; console output is
saved under `reports/c1_*.txt`.

## Data

| name | source | n (total) | features | target | oracle |
|---|---|---|---|---|---|
| studentt_sim | XGBoostLSS `studentT_{train,test}_sim.csv` | 8000 / 2000 fixed test | x, x2 | y | loc, scale, df |
| gaussian_sim | XGBoostLSS `gaussian_{train,test}_sim.csv` | 7000 / 3000 fixed test | x_true + 10 noise | y | loc, scale |
| synth_t3 | generated in harness, seed-dependent | 8000 | 5 × U(−1,1) | µ=sin(πx₁)+x₂², log σ=0.5x₃−0.3, ν=2.5+3σ(3x₄) | yes |
| kin8nm | UCI via ngboost | 8192 | 8 | y | — |
| power_plant | UCI CCPP via ngboost | 9568 | 4 | PE | — |
| naval | UCI via ngboost | 11934 | 16 | c17 (turbine decay) | — |
| protein | UCI CASP via ngboost, fixed 12000-row subsample (seed 2026) | 12000 | 9 | RMSD | — |

## Splits

Seed s ∈ {0..4}: one permutation per seed (ChaCha8, seed 1000+s). Fixed-test
datasets: 85/15 train/valid of the training file; test file untouched. Others:
20% test, then 15% valid, rest train. The *same* split is used for every
(family, mode) cell of that seed, so differences are paired.

## Model

`GradientLSS<XGBoostBackend>`, `Stabilization::None`, start values = raw 0.5
(`initialize: false`, the crate default), `LossFn::Nll`. XGBoost: `eta`,
`max_depth`, `min_child_weight=1`, `lambda=1` (defaults otherwise). Up to 2000
rounds, early stopping on the validation NLL with patience 50; prediction uses
`best_iteration`. Families: Gaussian (loc Identity, scale Exp) and StudentT (df
ExpDf, loc Identity, scale Exp). Modes: `HessianMode::Observed` and `Fisher`.
Nothing is tuned per cell; both modes see identical settings.

## Scores (all on the held-out test split)

- `nll` — mean −log f(y).
- `crps` — closed form (Gaussian; t for ν > 1, Gneiting & Raftery 2007), mean.
- `cov50`, `cov90` — share of PIT values in [0.25, 0.75] / [0.05, 0.95]
  (nominal 0.50 / 0.90).
- `pit_ks` — Kolmogorov distance of PIT values to U(0,1).
- `rmse` — of the predicted location.
- `oracle_nll`, `oracle_crps` — same scores from the true DGP parameters.
- `nonfinite_rows` — test rows with non-finite params or scores (excluded
  from means; a failure count, never hidden).
- `best_iter`, `fit_secs`.

## Decision rule

Paired per-seed differences Fisher − Observed, mean ± s.e. over 5 seeds, per
dataset × family. A method "wins" a dataset if |mean| > 2 s.e. and the sign
is consistent across the two hyperparameter settings. Claims are made on
NLL and CRPS; coverage/KS are reported for calibration context.
