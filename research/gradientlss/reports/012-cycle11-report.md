# Cycle 11 report — Poisson, NegativeBinomial, Gaussian

Date: 2026-11-11. Rows: `experiments_v2.csv` tags `c11*` (all with `BENCH_PIT_U=hash`); detail in
`research_log.md` "Cycle 11". Proposal 014 (+ Addendum G′).
**One default change: Poisson. Gaussian location units: investigated to the end, no library change (§3b).**

## 0. Harness bug: discrete coverage was biased
- **What was wrong:** the randomized PIT for counts drew its uniform from y itself, so zero counts
  always landed in the lower tail. Even the true model scored cov90 = 0.74–0.84 on the count
  simulations.
- **Fix:** with a hashed uniform the true model scores 0.89–0.90.
- **Consequence:** NB coverage figures from cycles 4–10 were too low. NLL and CRPS were never
  affected.

## 1. Poisson (landed)
- **Measured limitation:** the Poisson family had never been benchmarked. `Poisson::default()` used:
  - the ReLU link. A row whose margin falls to ≤ 0 has zero gradient and never recovers, and zero
    counts give zero curvature.
  - a raw start of λ₀ = 0.5 whatever the counts.

  On true Poisson data NB fitted better than Poisson.
- **Fair experiment:** 3 links × {observed, damped} × {raw, ȳ start}; 5 count sets × 5 seeds;
  identical splits.
- **Result:** log link + ȳ start against the old default:

  | dataset | ΔNLL | seeds better |
  |---|---|---|
  | bike_cnt | −2.68 | 5/5 |
  | bike_casual | −1.23 | 5/5 |
  | synth_pois | −0.054 | 5/5 |
  | synth_nb_lo | −0.021 | 4/5 |
  | synth_nb_ps | −0.003 | 4/5 |

  - CRPS and RMSE improve as well (bike_cnt RMSE −14.5).
  - Poisson now beats NB on Poisson data (2.146 vs 2.150; oracle 2.115).
- **Other arms:**
  - Damped curvature fixes most of ReLU's damage on the simulations, but the link's dead rows remain.
  - The ȳ start hurts under ReLU (bike_cnt +0.87), so it is on by default only for smooth links.
- **Decision:** `Poisson::default()` uses the Exp link, starts at ȳ and uses Damped curvature. ReLU is
  still available through `new`. Saved models are unchanged.

## 2. NegativeBinomial (no change)
- **Calibration:** with the fixed PIT, cov90 on the simulations is 0.905–0.907 against the oracle's
  0.89–0.90. NB was calibrated all along.
- **Per-parameter stopping:**
  - on true Poisson data it makes no difference (θ climbing is harmless);
  - on real counts it helps: bike_cnt −0.055, cov90 0.88 → 0.93.

## 3. Gaussian
- **Unit-free start** (σ₀ = 3·sd or sd): rejected. At native units naval is +0.10 / +0.16.
- **The real unit defect:** XGBoost's `min_child_weight` and `lambda` act on raw Hessian sums. The
  Gaussian location's Hessian 1/σ² scales as units⁻². At native units the implicit location
  regularisation therefore differs by ~10⁷ across datasets: ≈ 0.06 per row on power_plant against
  ≈ 10⁶ on naval.
- **Test:** standardizing the labels (training-split statistics only) makes the fit exactly
  unit-invariant (checked at ×100), and against raw at native units:

  | dataset | ΔNLL | seeds better |
  |---|---|---|
  | power_plant | −0.042 | 4/5 |
  | protein | −0.067 | 5/5 (RMSE −0.11) |
  | kin8nm | −0.0003 | |
  | gaussian_sim | +0.0006 | |
  | naval | **+0.066** | 1/5 |

  Naval is confounded. It is budget-capped at 2000 rounds in both arms, and standardization also
  moves its start from σ₀ ≈ 220 sd to 1.65 sd.
- **Decision (Felix):** a library fix would boost the location in standardized units internally,
  µ = ȳ + s·m with s from the training labels. It touches prediction and serialization, so it needs
  approval. It fails the strict rule on naval as measured.

## 3b. Follow-up (Felix asked): naval at 6000 rounds, start held equal
| comparison | ΔNLL | seeds better |
|---|---|---|
| standardized − native, 6000 rounds | +0.073 | 2/5 |
| curvature only (same start) | −0.019 | 4/5 |
| start only | +0.091 | 1/5 |

- The loss is real, and it comes from the start: raw σ₀ ≈ 220 sd is a lucky "approach from above" on
  naval.
- Standardized curvature with the old start held is not neutral either:
  - naval +0.067 (+0.054 at 6000), gaussian_sim +0.009 (0/5);
  - protein −0.011, so its earlier gain was mostly the start.
- With σ₀ ≈ 220 sd the standardized location Hessians are ~10⁻⁵ per row, and `min_child_weight` then
  blocks early splits.

The start, the curvature scale and XGBoost's absolute `min_child_weight`/`lambda` interact, and no
unit-free arrangement was neutral everywhere. **No library change.** The README notes the units
dependence and that standardizing labels is a user-side option to try.

## 4. Next investigations (ranked)
1. Done (§3b): no unit-free Gaussian arrangement was neutral everywhere. A possible follow-up is to
   express `min_child_weight` relative to the start's Hessian (scale-aware regularisation). That
   would be a hyperparameter-semantics change for Felix.
2. The same Hessian-scale question for StudentT's location and Gamma's MeanShape (log µ has a unit-free
   Hessian, so Gamma should already be invariant apart from f32).
3. Early stopping's relative min-delta (from cycle 10): also not unit-free.
