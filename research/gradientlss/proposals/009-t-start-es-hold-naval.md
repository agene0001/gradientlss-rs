# Proposal 009 — StudentT default start, early-stopping hold during anneal, naval round cap

Status: **resolved** — reports/008-cycle7-report.md. Was **pre-registered** (cycle 7, 2026-11-10), before any cycle-7 run. Felix: Python parity
is not a constraint; defaults still need better-or-neutral on every measured dataset.

## S — StudentT start values without `initialize`
Limitation: raw 0.5 start; on power_plant (y ≈ 450) the t score r/(νσ² + r²) vanishes and the
location never moves (RMSE 448, NLL 7.52). Measured already (c6b vs c6e): `initialize = true`
fixes it (−4.73) but is worse on protein (+0.068), kin8nm (+0.018), gaussian_sim (+0.013), and
with PES naval (+0.136) — the fitted marginal df start. So it fails the default rule.
Candidate: closed-form location = median, scale = 1.4826·MAD, df keeps raw 0.5 (ν ≈ 3.65).
- S1: vs raw start (PES off, init 0): power_plant ≤ −4; every other dataset ≤ +0.005.
- S2: same with the cycle-6 StudentT PES recipe (df+scale, W20, anneal .5).
Competing explanation: the raw start's tiny scale (1.65) might be *helping* on small-scale
targets (kin8nm y sd 0.26 → scale start is too large there instead); then S fails on kin8nm.

## H — global early stopping held on each PES trigger
Limitation: with anneal .5, 1/5 seeds of power_plant / protein (Gaussian) and Gamma power_plant
fell back to "none" because patience 50 ran out while the scale was being annealed.
Rule: every trigger resets the patience counter (`TrainingCallback::holds_early_stopping`);
best iteration untouched, so the worst case is still the no-PES fit, 50 rounds later.
- H1: Gaussian power_plant ≤ −0.16, protein ≤ −0.12; Gamma power_plant ≤ −0.13 (vs none).
- H2: no dataset of Gaussian / Gamma MS / NB MD / StudentT worse than no-hold by > +0.005.
- H3: sims unchanged (|Δ| ≤ 0.002 vs no-hold).
Decision: hold joins the default recipe iff H2 and H3 hold.

## N — naval round cap
Under the default PES recipe naval hits the 2000-round cap (Gaussian, Gamma MS, StudentT).
- N1: with 6000 rounds, default PES improves further on naval by ≥ 0.05 vs its 2000-round fit,
  while "none" is unchanged (it early-stops at ≈ 800).
This is a harness question (it changes no default) but sizes the gain users miss with low caps.

## Addendum S3 (after c7s, before c7s3) — pre-registered
S1 ✗ / S2 ✗: median+MAD start is worse than raw on kin8nm (+0.023 / +0.013), protein (+0.032 /
+0.025), gaussian_sim (+0.007 / +0.005) and, with PES, naval (+0.112, 0/5); it fixes power_plant
(−4.7). Reading: the raw scale start (1.65) keeps early location learning in the t-score's
linear (Gaussian-like) regime; what breaks power_plant is only the location being 450 away.
- S3: location = median, df and scale raw (0.5). Prediction: power_plant ≤ −4 vs raw; every
  other dataset ≤ +0.005 vs raw, with and without PES. If naval/protein move by more than that,
  the scale-regime reading is incomplete (their medians are also far from 0.5 in σ units).
