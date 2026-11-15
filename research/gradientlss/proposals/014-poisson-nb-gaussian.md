# Proposal 014 — Poisson, NB and Gaussian: link, curvature, starts, units

Status: **resolved** — reports/012-cycle11-report.md. Was pre-registered (cycle 11, 2026-11-11) after one smoke fit per arm (listed below) and before any multi-seed run.

Rule (Felix): a default changes only if it is better or neutral (≤ +0.005 NLL mean) on every measured
dataset. A noise-level miss that removes a real defect goes to Felix as a decision.

## 0. Harness bug: the discrete PIT uniform (found before any hypothesis run)
- **What it was:** the randomized PIT for count families used u = frac(0.618·y + 10⁻³µ), which is a
  function of y. Every y = 0 row got u ≈ 0, so zero counts landed in the lower tail.
- **Effect:** the true model's own cov90 was 0.82 (synth_nb_ps), 0.84 (synth_nb_lo) and 0.74
  (synth_pois). With a splitmix hash of (y, µ) (`BENCH_PIT_U=hash`) it is 0.90 / 0.89 / 0.89.
- **Consequence:** NB coverage reported in cycles 4–10 is biased low; NLL and CRPS are unaffected.
- **From here on:** all cycle-11 runs use `BENCH_PIT_U=hash`, and `BENCH_ORACLE_COV=1` prints the
  oracle's coverage.

## P — Poisson
The Poisson family was never benchmarked in cycles 1–10. Defaults: `ResponseFn::Relu` link, raw start
0.5 (λ₀ = 0.5 whatever the counts), observed curvature, no `initialize`.

Defects found by reading the code:
- **(a) Dead rows under ReLU.** For margin m ≤ 0, λ = 10⁻⁶ and λ′(m) = 0, so the row's gradient
  and Hessian are exactly 0 and it can never recover. One positive count there costs y·13.8 nats.
- **(b) Zero curvature on zero counts.** Under ReLU the observed curvature k/λ² is 0 for k = 0,
  against a Fisher value of 1/λ. Leaves dominated by zeros take steps limited only by
  `lambda`/`min_child_weight`, which pushes margins below 0 and into (a).
- **(c) Far start on large counts.** A raw start of 0.5 on counts near 190 (bike) needs many rounds
  at η = 0.1.

Under the canonical Exp link, observed = Fisher = λ exactly (no curvature defect) and no row can die.

Smoke results (seed 0, Observed):

| | synth_pois | bike_cnt |
|---|---|---|
| ReLU | 2.248 | 9.31 |
| Exp | 2.209 | 6.71 |
| NB MD on synth_pois | 2.186 | |
| oracle | 2.154 | |

Arms (5 seeds; datasets synth_pois, synth_nb_ps, synth_nb_lo, bike_cnt, bike_casual):
1. relu-observed (the current default);
2. relu-damped;
3. exp (the curvature mode is irrelevant);
4. softplus-damped;
5. each of 1–4 with `location_start` (λ₀ = ȳ through the inverse link).

Predictions:
- P1: exp beats relu-observed on every dataset; on bike_cnt by > 1 nat.
- P2: relu-damped beats relu-observed on synth_pois and synth_nb_lo (many zeros) but stays behind exp.
- P3: location start is ≤ +0.005 everywhere and wins on bike_cnt / bike_casual under every link
  (the same mechanism as NB in cycle 8).
- P4 (null for the NB-beats-Poisson gap): on synth_pois, Poisson exp + location start comes within
  0.01 of NB MD. If it stays > 0.02 behind, the gap is an optimisation defect specific to the Poisson
  head (a step/curvature issue), not a family limitation.

Default change candidate: Exp link (+ location start) as the Poisson default, if it is better or
neutral on all five sets. Saved models keep their link, since links are serialized per parameter.

## N — NegativeBinomial
- N1 (re-scoring): with the hashed PIT, NB MD's cov90 on the three sims is within ±0.02 of the
  oracle's. The cycle-4–10 under-coverage was the harness.
- N2: on true Poisson data (synth_pois), NB MD's log θ climbs with no limit. PES (the default) keeps
  NLL within 0.04 of the oracle. Without PES it is worse by ≥ 0.01. This checks whether PES is what
  saves NB at the Poisson boundary.

These are measurements; no default change is planned for NB.

## G — Gaussian start (the same unit defect as Gamma)
`Gaussian::default()` starts at raw [µ, log σ] = [0.5, 0.5], so σ₀ = 1.65 whatever the units:
220× too wide on naval (sd 0.0075) and 10× too narrow on power_plant (sd 17).

Arms (default PES; kin8nm, power_plant, protein, naval, gaussian_sim; 5 seeds):
- raw (the default);
- g_sd:1 = [ȳ, ln sd];
- g_sd:3 = [ȳ, ln 3·sd].

Each is run at label scale c ∈ {1, 100} (`BENCH_Y_SCALE`), with NLL compared after subtracting ln c.

Predictions:
- G1: raw is unit-dependent: |raw(100) − raw(1)| > 0.02 on at least two datasets.
- G2: g_sd:3 is ≤ +0.005 against raw at c = 1 everywhere, and better at c = 100. Cycle 9's
  "approach from above" result applies, and the Gaussian family has no df to be pushed up.
- G3: g_sd:1 is worse than g_sd:3 on gaussian_sim (approach from above).

Default change candidate: g_sd:3 as the Gaussian start without `initialize` (saved models keep raw),
if G2 holds.

## Addendum G′ (after c11g, before any standardized run) — pre-registered
c11g results:
- G2 ✗ at c = 1: sd3 − raw on naval +0.099 (1/5); sd1 naval +0.156.
- Both unit-free starts are far *more* unit-dependent than raw: c100 − c1 on power_plant is +1.30
  (sd3, sd1) against +0.20 (raw), and on protein +0.39 against +0.06.
- A unit-free start cannot produce this, so another part of the fit depends on units.

Diagnosis (from the code, before testing):
- XGBoost's `min_child_weight` (1) and L2 `lambda` (1) act on raw Hessian sums. The Gaussian
  location's Hessian is 1/σ², which scales as c⁻².
- At c = 100 on power_plant, h ≈ 6·10⁻⁶ per row, so even a 5000-row leaf has Σh < 1 and the location
  trees can barely split.
- At native units the implicit regularisation already spans ~10⁷ across datasets: h ≈ 0.06 per row on
  power_plant / protein, ≈ 100 on kin8nm, ≈ 10⁶ on naval.
- The log σ coordinate's Hessian (≈ 2) is unit-free.
- Raw is less affected at c = 100 because its σ₀ = 1.65 is tiny there. The early location Hessians
  are then large and the first location trees do get grown.

Test: `BENCH_Y_STD=1` standardizes the labels with training-split mean and sd (y → (y − ȳ)/s). This
makes every location Hessian unit-free, and NLL is compared after adding ln s (printed per seed).
Arms: standardized + default raw start, Gaussian, 5 datasets, 5 seeds, at c = 1. Invariance in c
then holds by construction (checked on one dataset).

Predictions:
- G′1: standardized − raw ≤ +0.005 on gaussian_sim and kin8nm, and better on power_plant and
  protein, where native units over-regularise the location. Naval: no prediction (native h ≈ 10⁶,
  which is effectively unregularised in both).
- G′2: standardized at c = 100 equals standardized at c = 1 to ≤ 0.002 (power_plant).

If G′1 holds, the library candidate is a unit-free location coordinate: boost µ = ȳ + s·m with s from
the training labels, which is equivalent to standardizing internally. That is an invasive change
(prediction path and serialization) and needs Felix's approval before it is implemented. This cycle
only measures it.

## Addendum G″ (cycle 11b, 2026-11-11, after G′, before any 6000-round run) — pre-registered
G′'s naval loss (+0.066, 1/5) mixes three changes: the round budget (both arms sit at 2000), the start
(σ₀ = 1.65 in data units ≈ 220 sd vs 1.65 sd), and the Hessian scale.

Naval, Gaussian, default PES, 5 seeds, `BENCH_ROUNDS=6000`:
- A: native units, default raw start;
- B: standardized labels, default start (µ₀ = ȳ + 0.5·sd, σ₀ = 1.65·sd in data units);
- D: native units, preset `g_std_equiv` = B's start in data units [ȳ + 0.5·sd, ln(1.65·sd)].
  B and D differ only in the Hessian scale (and f32 rounding).

Predictions:
- H1: at 6000 rounds, B − A on naval is ≤ +0.02 (the 2000-round loss was mostly budget/start).
- H2: B − D is ≤ +0.005, i.e. Hessian scaling itself does not hurt naval.
- H3: if B − A stays > +0.02, D − A carries it (the start, not the curvature).

Decision: if H1 holds, internal standardization of the Gaussian location is a candidate default and
goes to Felix with the full table.

### G″ results (naval, 6000 rounds, 5 seeds)
| | comparison | mean | se | seeds better |
|---|---|---|---|---|
| H1 ✗ | B − A | +0.073 | 0.050 | 2/5 |
| H2 ✓ | B − D (curvature alone, same start) | −0.019 | 0.016 | 4/5 |
| H3 ✓ | D − A (start alone) | +0.091 | 0.048 | 1/5 |

Naval's loss under standardization is the start: raw σ₀ ≈ 220 sd is an extreme "approach from above".
The unit-free curvature itself helps.

## Addendum G‴ (before any C run) — pre-registered
Arm C: standardized labels (unit-free location curvature), with the start held at the *current* raw
start mapped into standardized units: µ₀ = (0.5 − ȳ)/s, σ₀ = 1.65/s (preset `g_raw_in_std`). This is
the minimal library change: curvature unit-free, starts unchanged.

Runs (Gaussian, default PES, 5 seeds): C on kin8nm, power_plant, protein, naval, gaussian_sim at
2000 rounds (against c11g_raw_c1), plus naval at 6000 (against c11h_A_native).

Predictions:
- C1: C − raw ≤ +0.005 on all five sets at 2000 rounds, and on naval at 6000. By H2 the curvature
  alone does not hurt.
- C2: G′'s gains on power_plant/protein come at least half from the curvature, i.e. C − raw ≤ −0.02
  on both. If C2 fails, those gains were the start (σ₀ = 1.65·sd vs raw), and the curvature change is
  only a unit-robustness fix.

Decision: if C1 holds, internal location standardization with unchanged starts goes to Felix as a
candidate default.

### G‴ results
C (standardized curvature, raw start mapped) − raw, 2000 rounds:

| dataset | mean | seeds better |
|---|---|---|
| gaussian_sim | +0.0088 | 0/5 |
| kin8nm | −0.0055 | 3/5 |
| naval | +0.067 | 1/5 |
| power_plant | −0.036 | 2/5 |
| protein | −0.011 | 2/5 |

Naval at 6000 rounds: +0.054 (1/5).
- C1 ✗ (naval, gaussian_sim). C2 ✗ for protein: its G′ gain was mostly the start.
- Reading: H2's curvature gain on naval held only with the standardized start. With the raw start
  mapped (σ₀ ≈ 220 sd) the standardized location Hessians start at ~2·10⁻⁵ per row, so
  `min_child_weight` blocks early location splits.
- Neither unit system's absolute thresholds are "right". The outcome is an interaction of start,
  curvature scale and `min_child_weight`/`lambda`.

**Decision: no library change.** The README documents that Gaussian fits depend on label units through
XGBoost's absolute Hessian thresholds, and that standardizing the labels (training split only) is an
option users can tune.
