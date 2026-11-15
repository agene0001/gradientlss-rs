# Proposal 015 — scale-relative tree regularisation (Hessian normalisation)

Status: **resolved** — reports/013-cycle12-report.md. Was pre-registered (cycle 12, 2026-11-11) before any implementation run; Felix approved investigating the semantics change.

## Measured limitation (cycle 11)
XGBoost's `min_child_weight` (mcw = 1) and L2 `lambda` (= 1) act on raw Hessian sums, and those
sums carry the parameter's units:
- **Gaussian location:** h = 1/σ². That is ≈ 0.06 per row on power_plant and ≈ 10⁶ on naval, so the
  same hyperparameters mean wildly different regularisation. Labels × 100 cost power_plant 0.2 nats
  with the raw start, and 1.3 with a unit-free start.
- **Other families:** Gamma MeanShape's log µ has h ≈ α, about 10⁴ on naval. Poisson's log-link
  Hessian is λ, about 190 on bike_cnt. NB's is µθ/(µ + θ).

## Math
For parameter j, multiply its gradient and Hessian columns by a constant κⱼ > 0. The leaf value is
then

  −κΣg / (κΣh + λ) = −Σg / (Σh + λ/κ),

which is the same Newton step with `lambda` → λ/κⱼ. The split gain scales by κⱼ, so `gamma` → γ/κⱼ,
and the child-weight test becomes Σh ≥ mcw/κⱼ.

Choosing κⱼ = 1 / (mean per-row Hessian of parameter j) makes mcw and λ count "rows' worth of
information", as in squared-error regression where h ≡ 1. `eta` and the unregularised step are
unchanged. Trees, prediction and serialization are untouched: it is a training-time rescaling of
(g, h), applied before per-parameter stopping and freezes.

Two reference points for the mean Hessian:
- **Start:** the Hessians at the start values (round 0). This is cheap, but inherits a
  unit-dependent start; with the raw Gaussian start κ_loc = σ₀² whatever the units.
- **Marginal:** the Hessians at the marginal MLE (`Distribution::calculate_start_values`, the fitted
  intercept-only model). It does not depend on the start or the units: for Gaussian, κ_loc = s²,
  κ_logσ = ½.

API: `GradientLSS::with_hessian_normalization(HessianNormalization::{Off, Start, Marginal})`.
Default Off.

Null explanation to keep in mind: any effect is a change of effective regularisation strength, which
ordinary tuning of mcw/λ could also reach per dataset. The claim is that one *fixed* setting behaves
consistently across units and datasets, not that it beats per-dataset tuning.

## Experiments (5 seeds, default recipe incl. PES, Damped)
- Gaussian: kin8nm, power_plant, protein, naval, gaussian_sim.
- Poisson (new default): synth_pois, synth_nb_ps, synth_nb_lo, bike_cnt, bike_casual
  (`BENCH_PIT_U=hash`).
- NB MD (location start): the same five count sets.
- Gamma MS (default wide start): kin8nm, power_plant, naval, synth_gamma.
- Invariance: Gaussian with preset `g_std_equiv` (a unit-free start) + Marginal at c = 1 and c = 100,
  on power_plant and kin8nm.

Arms: Off (baseline), Start, Marginal.

## Predictions
- R0 (invariance): with a unit-free start + Marginal, c = 100 equals c = 1 to ≤ 0.002 mean |Δ| per
  dataset (f32 noise). Without normalisation the same start gave +1.30 on power_plant (c11g sd1).
- R1: Marginal is ≤ +0.005 against Off on every dataset, and better on Gaussian power_plant and
  protein. Those have native h ≪ 1, so they are over-regularised now.
- R2: where the native Hessians are large (Gaussian naval, Gamma naval, Poisson/NB bike), mcw = 1
  and λ = 1 are negligible both before and after relative to leaf sums of thousands of rows:
  |Δ| < 0.01.
- R3: Start and Marginal agree within 0.01 where the start is close to the marginal fit (Gamma, NB,
  Poisson location starts), and differ on Gaussian (raw start).

Default change only if Marginal (or Start) passes the rule everywhere. Since it changes what mcw/λ
mean, it goes to Felix even if it passes.

## Results (c12; resolved — reports/013-cycle12-report.md)
Marginal − Off (5 seeds):

| family | dataset | ΔNLL | seeds better |
|---|---|---|---|
| Gaussian | kin8nm | −0.018 | 5/5 |
| Gaussian | power_plant | −0.044 | 4/5 |
| Gaussian | protein | −0.018 | 1/5 |
| Gaussian | naval | +0.016 | 2/5 |
| Gaussian | gaussian_sim | **+0.0087** | 0/5 |
| Gamma | naval | −0.043 | 3/5 |
| Gamma | kin8nm | −0.013 | 4/5 |
| Gamma | power_plant | +0.005 | |
| Gamma | synth_gamma | +0.001 | |
| NB | bike_cnt | −0.008 | 4/5 |
| NB | other sets | ±0.002 | |
| Poisson | bike_cnt | **+0.066** | 0/5 |
| Poisson | bike_casual | +0.003 | |
| Poisson | synth_nb_lo | −0.006 | |
| Poisson | other sets | ±0.001 | |

Start − Off: Gaussian power_plant **+0.236** (0/5; the raw start's σ₀ makes κ_loc = σ₀² ≈ 2.7, much
stronger regularisation), Poisson bike_cnt +0.052, protein −0.046.

Predictions:
- R0 ✗ strictly, but near: a unit-free start + Marginal gives c100 − c1 of kin8nm −0.002 and
  power_plant +0.006 mean (per seed ≤ 0.025). Without normalisation the same start gave +1.30. The
  remainder is f32 and the relative early-stopping min-delta.
- R1 ✗: gaussian_sim +0.009 and Poisson bike_cnt +0.066 fail the band.
- R2 ✗: large Hessians are *not* inert. "One row's worth" of λ is λ·h̄ in raw units, so Poisson
  bike_cnt (h̄ ≈ λ ≈ 190) gets ~190× stronger L2 per leaf and loses.
- R3 ✓ for the count/Gamma location starts; Gaussian differs.

Reading: the normalisation fixes the units dependence, but the right regularisation strength still
differs by dataset (the null explanation: effectively a tuning change).

**Decision:** opt-in only (`with_hessian_normalization`), default Off. It is useful for unit robustness
and as a better-conditioned axis for tuning mcw/λ.
