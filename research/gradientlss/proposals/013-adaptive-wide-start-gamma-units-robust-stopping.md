# Proposal 013 — adaptive wide start, Gamma start units, tail-robust stopping

Status: **resolved** — reports/011-cycle10-report.md. Was pre-registered (cycle 10, 2026-11-11) before any cycle-10 run and before computing any label kurtosis.

Rule (unchanged): a default changes only if it is better or neutral (≤ +0.005 NLL mean) on every
measured dataset.

## A — adaptive "start wide" for StudentT
c9: a start scale of 3·sd(y) reproduces the raw start's gaussian_sim gain (−0.0088) but hurts
heavy-tailed targets (protein +0.091, power_plant +0.054, synth_t3 +0.020, studentt_sim +0.013).
Gate (fixed now, from theory, not tuned): inflate the start scale to 3·sd(y) iff the training
labels' sample excess kurtosis g₂ < 1 (a Gaussian has 0; a t₅ has 6, a t₉ has 1). Otherwise use
the median start unchanged. `StudentTStart::Adaptive`.

Known confound: g₂ is the *marginal* kurtosis. It mixes the signal's distribution with the noise
tails, so a Gaussian-noise target with a kurtotic mean function reads as heavy-tailed, and a
heavy-tailed target with a large bounded signal can read as light-tailed (power_plant is
bimodal-ish).

Three new held-out simulations (never used before, n = 8000, 5 seeds), all with
heteroscedastic noise:
- `synth_g_sin`: Gaussian noise, sin mean, light marginal. The gate should inflate.
- `synth_g_exp`: Gaussian noise, mean exp(2x₀) (kurtotic marginal). This is the confound case:
  the gate should *not* inflate, though the data is Gaussian.
- `synth_t4_flat`: t₄ noise, small signal. The gate should not inflate.

Predictions (PES off, as in c9w; then also with the default PES):
- A1: Adaptive is ≤ +0.005 against Median on all 10 datasets (7 old + 3 new), and better on
  gaussian_sim (≈ −0.009). It fails if the gate inflates protein or power_plant, or any heavy
  set.
- A2: synth_g_sin improves under Adaptive (< −0.003). This tests whether the gaussian_sim effect
  generalises at all.
- A3: on synth_g_exp the forced wide start (`t_scale:3`) helps, but the gate misses it. This
  documents the confound.

Default change only if A1 holds both with PES off and with the default PES.

## G — Gamma MS raw start: unit dependence and a deliberate wide start
The raw start [log µ, log α] = [0.5, 0.5] has µ₀ = 1.65 in the target's units, so it is not
scale-equivariant. On naval (ȳ ≈ 0.975) it happens to sit 1.7× above the mean with α₀ = 1.65.
New bench knob `BENCH_Y_SCALE=c` multiplies every label by c. The reported NLL is compared after
subtracting ln c, and CRPS after dividing by c.

Arms (default PES + hold; kin8nm, power_plant, naval, synth_gamma; 5 seeds; c ∈ {1, 0.01, 100}):
- raw (the default);
- wide_b = [ln ȳ + 0.5, 0.5]: unit-free, and mimics raw on naval;
- wide_a = [ln ȳ, 0]: α₀ = 1, mean at ȳ;
- wide_m = [ln ȳ − 0.5, 0.5]: tests whether the mean offset's *direction* matters.

Predictions:
- G1: raw is unit-dependent. On naval, |raw(c = 100) − raw(c = 1)| > 0.05 after the ln c
  adjustment.
- G2: the wide_* arms are unit-free: identical across c up to ≤ 0.002 (floating point and
  tree-split noise).
- G3: wide_b ≈ raw at c = 1 on naval (within 0.03), and is ≤ +0.005 against raw on every
  dataset at c = 1.
- G4: wide_m is worse than wide_b on naval, i.e. starting *above* the mean matters.

Default change (Gamma MS start → wide_b) only if wide_b is ≤ +0.005 against raw on every
(dataset, c) cell. Since raw has no unit invariance, wide_b would then remove a unit dependence
at no cost.

## R — tail-robust stopping metric
c9: Weibull test blow-ups are single rows (e^z tail). The stopping decision (best iteration,
patience counter, per-parameter revert test) uses the mean validation NLL, which a single
validation row can dominate in the same way.

Robust stopping metric, for the stopping decision only:
- Row losses ℓᵢ = −log p(yᵢ), with m = their median on the validation set this round.
- Rows above m + K are softened: ℓ̃ = m + K + ln(1 + ℓ − m − K).
- The metric is mean(ℓ̃).

Implementation details:
- `GradientLSS::with_stopping_row_cap(K)`. The metric is unit-free because it works on
  differences of log densities.
- It is used for global early stopping and for the PES test. It never affects gradients or the
  test scoring.
- `valid_history` and `best_score` then report the robust value; this is documented.

Arms: K ∈ {10, 25}.
- Weibull: synth_weibull, kin8nm, naval, power_plant (10 seeds).
- Neutrality checks: Gaussian (kin8nm, power_plant, protein, naval, gaussian_sim), Gamma MS
  (kin8nm, power_plant, naval, synth_gamma), StudentT (power_plant, protein, synth_t3).
- Default recipe throughout.

Predictions:
- R1: on Weibull the robust metric is ≤ +0.005 everywhere. The c9 blow-up rows are *test* rows,
  so the measured gain is expected to be small: |Δ| < 0.02 except where a validation row was
  extreme.
- R2: on the non-Weibull families it is bit-identical or within ±0.002. No validation row
  exceeds m + 10 nats in a well-specified fit.

Default change only if one K is ≤ +0.005 everywhere *and* better somewhere by more than noise.
Otherwise it ships as an opt-in.

## Addendum A′ (after printing train-label g₂, before any cycle-10 fit) — pre-registered
Train-label excess kurtosis (seed 0): synth_g_sin −0.81, synth_g_exp 0.53, synth_t4_flat 5.14,
gaussian_sim **3.57**, kin8nm −0.57, power_plant **−1.06**, protein **−1.12**, naval −1.22,
synth_t3 20.2, studentt_sim 195.
The marginal gate is inverted on the old sets:
- It inflates protein and power_plant, where c9's 3·sd start cost +0.091 / +0.054.
- It does not inflate gaussian_sim (g₂ = 3.57: a heavy marginal from the signal), the one set
  that gained.

So Adaptive ≡ t_scale:3 on kin8nm/power_plant/protein/naval and ≡ Median elsewhere, and
A1 fails from c9 rows already (protein +0.091). A3's premise is also wrong: synth_g_exp reads as
light (0.53). The marginal-kurtosis gate is rejected without a fit. `StudentTStart::Adaptive`
stays out of the defaults (research option, documented as failing).

Replacement gate A′: fit both starts (Median, 3·sd) and keep the one with the lower best
validation NLL. This costs 2× and uses train + validation only.
- Evaluated on existing paired rows c7s3_t_med_0 (Median) and c9w_s3 (3·sd), selected per seed
  by `valid_nll`, plus the new sims.
- A′1: the selection is ≤ +0.005 against Median on all 10 datasets.
- A′2: it keeps ≥ half of the gaussian_sim gain (≤ −0.0044).

Even if both hold, A′ is a 2× fitting cost. It would be an opt-in helper, not a default.

New-sim runs still go ahead: Median vs t_scale:3 on synth_g_sin / synth_g_exp / synth_t4_flat
(PES off). That is A2: does "approach from above" generalise beyond gaussian_sim on Gaussian
noise?
