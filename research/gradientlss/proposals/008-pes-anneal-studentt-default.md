# Proposal 008 — Per-parameter early stopping: anneal vs freeze, StudentT, hyper_opt, default-on

Status: **resolved** (cycle 6, 2026-11-10) — see reports/007-cycle6-report.md. Pre-registered before any cycle-6 run. Felix approved
"on by default whenever a validation set is passed", subject to the usual rule: the default
config must be better or neutral on every measured dataset.

## Measured limitation (cycle 5)
- Freeze is permanent. W = 50 / patience 1 froze power_plant's σ at round ≈ 95 on a transient
  signal (+0.256 vs none): σ̂ then cannot follow a location that keeps improving.
- Gamma MeanShape naval at W = 20 is noisy (−0.013 ± 0.079, one seed +0.15).
- StudentT untested (two non-location parameters, df and scale; order is df, loc, scale).
- Window / patience are hand-set; `hyper_opt` cannot tune them.

## Rule (anneal)
`ParamEarlyStopping::with_anneal(f)`: when the revert test fires for j, multiply j's gradient
column by f (the per-parameter effective learning rate, as `with_param_gradient_scales`)
instead of zeroing it. j is re-tested only once a full window of trees has been grown at the
new rate (next test at trigger round + W). Once the multiplier drops below 1/64 the parameter
is frozen. f = 0 (default) reproduces today's freeze bit-for-bit.

## Pre-registered predictions (Damped, η .05/d4, ES 50, 5 paired seeds, scale/shape only)
- A1: anneal 0.5, W = 50, patience 1 repairs the power_plant failure: Gaussian ≤ −0.10 vs none
  (freeze at those settings: +0.256).
- A2: anneal 0.5, W = 20 is within 0.02 of freeze W = 20 on the Gaussian real targets
  (−0.137 / −0.172 / −0.136 / −0.267), and its Gamma naval spread (s.d. over seeds) is lower.
- T1: StudentT, scale-only (j = 2), W = 20: ≤ −0.07 on kin8nm / power_plant / protein vs none
  (as the Gaussian); |Δ| ≤ 0.01 on the three sims.
- T2: StudentT, df-only (j = 0): |Δ| ≤ 0.01 everywhere (df moves slowly; the cycle-3 df study
  found no df over-fitting).
- T3: StudentT both (0, 2) is within 0.01 of scale-only.
- T4: with the residual warm start (the recommended StudentT recipe) scale-only still helps
  on kin8nm / power_plant / protein (≤ −0.03).
- D (default): the default recipe is the best of {freeze W20, anneal-0.5 W20} that is better
  or neutral (mean Δ ≤ +0.005) on every measured dataset of Gaussian, StudentT, Gamma
  MeanShape and NB MeanDispersion. Families / heads without a measured eligible set
  (Poisson, legacy heads, mixtures, multivariate, …) keep no per-parameter stopping. If no
  candidate passes on all datasets, PES stays opt-in and this is reported.
- H (hyper_opt): `__pes_window` (0 = off), `__pes_patience`, `__pes_anneal` axes reach the
  fold models, and a helper applies the winning values to the final fit (unit-tested).

## Addendum A (after batch c6a/c6b, before batch c6d) — pre-registered
Observed: anneal 0.5 needs ~7 windows to reach the freeze, and on 1/5 seeds of power_plant
and protein the global early stopping (patience 50) fired first, reverting the fit to "none"
(+0.14 vs freeze). StudentT freeze on the scale is catastrophic on gaussian_sim (+0.387, 0/5;
df still drifting up while σ is frozen), anneal 0.5 is not (+0.003). StudentT defaults
(`initialize = false`, raw 0.5 start) are degenerate on power_plant (RMSE 448, cov90 1.0)
with or without PES — a separate defect, not PES's.
- A3: anneal 0.25 (freeze on the 4th trigger) at W = 20 keeps every Gaussian / Gamma / NB
  dataset within +0.02 of freeze W20 *or* better, and keeps ≥ half of anneal-0.5's naval gain
  over freeze (Gaussian −0.10, Gamma −0.18).
- T5: StudentT with `initialize = true`: scale-only anneal (0.5 or 0.25) is ≤ +0.005 on
  gaussian_sim / synth_t3 / studentt_sim and ≤ −0.03 on kin8nm / naval / protein / power_plant.
- Default candidates now: {freeze W20, anneal-0.5 W20, anneal-0.25 W20}; StudentT eligible set
  = scale only (df freezing was harmful on default-init power_plant, +0.208, and neutral elsewhere).
