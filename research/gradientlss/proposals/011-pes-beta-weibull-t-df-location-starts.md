# Proposal 011 — PES for Beta/Weibull, StudentT gaussian_sim cost, location starts for Weibull/Gamma/NB

Status: **resolved** — reports/009-cycle8-report.md. Was **pre-registered** (cycle 8, 2026-11-10), before any cycle-8 run.

## P — per-parameter early stopping for Beta (log φ) and Weibull (log k)
Default recipe (W20, patience 1, anneal .5, hold) on param 1 vs none; Beta MeanPrecision and
Weibull, both Damped (their defaults). Datasets: synth_beta, synth_beta_hi, bike_share;
synth_weibull, kin8nm, power_plant, naval.
- P1: ≤ +0.005 on every dataset (then added to the default watched sets).
- P2: ≤ −0.03 on at least one real dataset of each family (kin8nm/power_plant over-fit the
  scale for every other family).

## T — what causes StudentT's +0.008 on gaussian_sim with the median start
Observed (c6b_t_0 vs c7s3_t_med_0): final df median 100 (raw) vs 11 (median start), df q05 16.7
vs 3.2; best_iter 120 vs 59. Mechanism hypothesis (T-a): with the raw start the location starts
~10 away, the scale first inflates, residuals look light-tailed against it, and df is pushed up
early; with the median start df stays near its heavy-tailed start (ν ≈ 3.65) until the
scale-driven early stop. Competing (T-b): the median start just stops earlier (scale over-fits
sooner) — fewer rounds for every parameter, df included.
- T1 (T-a): median start with ν₀ = 10 closes the gaussian_sim gap (≤ +0.002 vs raw) and leaves
  df_med ≥ 30 there.
- T2 (side effects): with ν₀ = 10, synth_t3 / studentt_sim / kin8nm / protein / naval within
  +0.005 of the median start (ν₀ = 3.65), and power_plant keeps ≤ −4.
- T3 (T-b check): the median start with df gradient ×2 (`with_param_gradient_scales([2,1,1])`)
  also closes the gap if T-b is right; if only ν₀ = 10 closes it, T-a.
Decision: the start df becomes 10 for the median start iff T1 and T2 hold (with and without PES).

## L — location-only start for Weibull, Gamma MeanShape, NB MeanDispersion (no `initialize`)
Location raw start from the labels, other parameters raw 0.5 (as StudentT):
Gamma MS log µ = ln ȳ; NB MD log µ = ln ȳ; Weibull ln λ = ln(median / (ln 2)^{1/k₀}), k₀ = e^0.5.
- L1: ≤ +0.005 on every dataset of each family (Gamma MS: kin8nm, power_plant, naval,
  synth_gamma; NB MD: bike_cnt, bike_casual, synth_nb_lo/hi/ps; Weibull: synth_weibull, kin8nm,
  power_plant, naval), with the families' default PES (Gamma/NB) and none (Weibull).
- L2: Weibull improves where y is far from 1.65 (power_plant ≤ −0.02); for Gamma/NB, whose
  location score does not saturate, expect neutrality (|Δ| ≤ 0.005) and fewer rounds.
Decision per family: default iff L1.

## Addendum L (before any run)
Finding while implementing: the harness families `gamma_ms` / `nb_md` were built with
`Gamma::new_mean_shape` / `NegativeBinomial::new_mean_dispersion`, which set `initialize: true`
(exact Gamma MLE / NB moments starts). Every Gamma MS / NB MD number since cycle 4 is therefore
the FITTED-start configuration, while `Gamma::default()`, `NegativeBinomial::default()` and
`new(.., false)` start at raw 0.5. The families now build through `new(.., BENCH_INIT)`; with
BENCH_INIT=1 they must reproduce the old rows bit-for-bit (checked first).
Arms per family: raw (init 0, today's user default), loc (init 0 + location start), fitted
(init 1). L1/L2 compare loc vs raw. Also reported: fitted vs raw — if fitted is better or neutral
everywhere and loc is not, the default start for these heads becomes the fitted closed form.

## Addendum T4 (after c8t, before c8t4) — pre-registered
T1 ✗ (ν₀ = 10 closes 0.003 of 0.008; hurts synth_t3 +0.012, studentt_sim +0.007, power_plant
+0.017), T3: df ×2 raises gaussian_sim df_med 11 → 30 with ΔNLL +0.0002 ⇒ df is NOT the cause.
Remaining difference: mean predicted scale 1.90 (median) vs 2.28 (raw), cov90 0.881 vs 0.892,
best_iter 59 vs 120. Hypothesis T-c: gaussian_sim's y is centred at 10; the raw start's location
spends its first ~60 rounds travelling 9.5 units, an implicit warm-up that regularises the scale.
The median start is translation-equivariant; the raw start is not.
- T4a: gaussian_sim shifted by −9.5 (y centred at 0.5): raw start NLL within 0.002 of the median
  start's (the raw advantage disappears).
- T4b: the median start's NLL is unchanged by the shift (|Δ| ≤ 0.001).
- T4c: shifted by +20 (y ≈ 30), the raw start is worse than the median start.
If T4a–b hold, the +0.008 is an artefact of the raw start's position, not a cost of the fix.

## Addendum P3 (after c8p, before c8p10) — post-hoc extension, disclosed
Weibull PES on power_plant: +0.043 mean, driven by one seed (+0.56 on test, while its validation
NLL was 0.05 BETTER); validation and RMSE improve on all 5 seeds. P1 strictly ✗. Extension: 10
seeds on power_plant. If the 10-seed mean is ≤ +0.005, Weibull's shape joins the default watched
set, reported as a post-hoc decision; otherwise Weibull stays opt-in.
