# Cycles 14–22 report — the tuner, family selection, backends, unseen data, more families, CRPS

Date: 2026-11-13. Proposals 017–025. Rows: `experiments_v2.csv`, tags `c14_*`, `c14f`, `c15_*`,
`c16_*`, `c17_*`, `c18_*`, `c19_*`, `c20_*`, `c21_*`, `c22_*`. First cycles run under the project skills
`automl-hyperparameter-research` and `optimization-algorithm-research`.
Cycles 14–22 changed no default and every addition was opt-in; the decisions of §10 were then
approved and applied (§11), which changes four family defaults and the CRPS path.

All numbers are test NLL, mean over 5 seeds, paired by seed, unless a line says otherwise. "The
rule" is the project's: a change must be better or neutral (≤ +0.005) on every dataset.

## 0. What changed in the code (uncommitted)
| where | what | default |
|---|---|---|
| `src/hyper_opt.rs` | `HyperOptConfig::sampler` (`Sampler::{Tpe, Random, TpeAfterRandom{n_startup}}`), `HyperOptConfig::initial_trials` | `Tpe`, empty: the default path replays a cycle 13 cell line for line |
| `src/distributions/location_scale.rs` (new) | Fisher constants, Damped / Fisher curvature and closed-form starts for five families | — |
| `cauchy.rs`, `logistic.rs`, `gumbel.rs`, `laplace.rs`, `log_normal.rs` | `with_hessian_mode`, `with_start(LocationScaleStart)` | Observed, Raw: unchanged; old saved models load |
| `examples/research_bench.rs` | samplers, default-first, family-selection trace, five families, LightGBM build, `c_<name>` datasets, `BENCH_LOSS`, `BENCH_HPO_SPACE` | — |

`HyperOptConfig` gained two public fields. Code that builds it with a full struct literal (no
`..Default::default()`) needs them added. This table is the state after cycle 22; §11 lists what
changed when the decisions were applied (four of these defaults among them).

## 1. Is the search the bottleneck? (proposal 017, cycle 14)
Cycle 13's raw arm, re-run with three other samplers: 10 sets × 5 seeds, 3-fold CV, 20 trials over
mcw and λ.

**Tier 1 — best CV score, random − TPE** (negative = random better):
| budget | sets where random is better | median difference |
|---|---|---|
| 5 trials | 4/10 | +0.003 |
| 10 trials | 0/10 | +0.004 |
| 20 trials | 0/10 | +0.003 |

TPE's odd first five trials (trial 2 repeats trial 1; trials 3–4 go to the range ends) cost nothing
measurable. TPE is not worse than random anywhere and slightly better on naval.

**Tier 2 — test NLL at 20 trials, tuned − untuned default:**
| sampler | mean over sets | sets worse than the default by > 0.005 |
|---|---|---|
| TPE (library default) | −0.008 | 2/10 (Gaussian naval +0.053, Poisson bike_cnt +0.030) |
| random | −0.004 | 2/10 |
| default first, then TPE | −0.006 | 2/10 |
| default first, 10 random, then TPE | −0.009 | 2/10 |

**What the search believes against what it gets** (600 refits; the default's CV score comes from the
arms that score it first, on the same folds):
| budget | CV gain over the default | validation gain of the refit | test gain of the refit |
|---|---|---|---|
| 5 | −0.034 | −0.005 | −0.005 |
| 10 | −0.039 | −0.007 | −0.006 |
| 20 | −0.047 | −0.007 | −0.007 |

- More trials raise the believed gain by 0.013 and the real one by 0.002.
- Within a set, the CV gain of a pick does not predict its test gain: correlation −0.18 to +0.36.
- The refit's validation gain does predict it: 0.86–0.98 on both naval sets, both bike_cnt sets and
  Gamma kin8nm (0.23–0.69 on the other five, where the gains are small).
- **Diagnosis, by the rule fixed in advance:** Poisson bike_cnt and Gaussian naval are selection
  failures. The default's CV score is worse than the pick's in 5/5 seeds (by 0.052 and 0.095);
  validation and test prefer the default. On naval 4 of 5 picks have mcw ≈ 700–990.
- The cause of the disagreement is not established. A fold model sees 2/3 of the training rows and
  the two axes are in absolute Hessian units; §6 tests that with 5 folds (it is not the cause).

**Predictions:** P1 ✗ (random better at 5 trials on 4/10, needed 7), P2 ✗ (7/10 within 0.005; the
three misses favour TPE), P3 ✗ (scoring the default first: +0.014 to +0.032 on four sets, −0.049 on
naval), P4 ✓.

## 2. `dist_select` against a held-out choice (proposal 018, cycle 14f)
9 sets × 5 seeds, 2–4 candidate families each. `dist_select` ranks by an unconditional fit to the
training labels; the alternative fits each candidate and compares validation NLL.

| selector | picks the best-on-test family | mean regret (test NLL) |
|---|---|---|
| marginal fit (`dist_select`) | 22/45 cells | 0.0545 |
| validation NLL of fitted models | 42/45 cells | 0.0005 |

| dataset | marginal pick | regret | best on test |
|---|---|---|---|
| protein | Gaussian (5/5) | 0.211 | StudentT |
| kin8nm | Weibull (5/5) | 0.137 | StudentT |
| power_plant | Gamma (5/5) | 0.104 | StudentT |
| naval | Gaussian (3/5) | 0.034 | Gamma |
| gaussian_sim | StudentT (5/5) | 0.006 | Gaussian |

The count sets and the two StudentT simulations are picked correctly by both. F1 ✓, F2 ✓, F3 ✓.
Side finding: `dist_select` scores StudentT at the training start values, whose df is capped at 10,
so on protein and power_plant its marginal NLL comes out worse than Gaussian's. (Fixed in §11; it
was not the cause of the wrong picks.)

## 3. Do the defaults of cycles 1–13 transfer to LightGBM? (proposal 019, cycle 16)
Old recipe (Observed curvature, raw starts, no per-parameter early stopping) against today's
defaults, 21 pairs, both backends.

| | XGBoost | LightGBM |
|---|---|---|
| pairs breaking the rule | 0/21 | 0/21 |
| mean change over common pairs | −1.006 | −1.095 |
| old recipe fails outright | — | StudentT power_plant 5/5 seeds; Gamma power_plant and synth_gamma 1/5 each |
| today's defaults fail | 0 | 0 |

Sign agreement on the 19 pairs with |Δ| > 0.01: 19/19. L1 ✓, L2 ✓, L3 ✓.
Caveat: LightGBM Gaussian protein is −0.015 ± 0.126 (3/5); seed 2 is 3.915 with today's defaults
against 3.455 with the old (XGBoost 2.945). The pair mean hides it.

## 4. The defaults on datasets no cycle has seen (proposal 020, cycle 17)
Seven UCI sets fetched for this (`data_confirm/`): airfoil, Parkinson's telemonitoring, gas-turbine
NOx and CO, superconductivity (12 000 rows), abalone, Seoul bike. 19 (family, set) pairs.

| family | pairs | mean change | worst pair |
|---|---|---|---|
| Gaussian | 5 | −0.57 | c_gt_co −0.073 ± 0.047 (4/5) |
| StudentT | 5 | −6.6 | c_gt_co −0.158 |
| Gamma | 5 | −2.19 | c_gt_co −0.731 |
| NB | 2 | −0.26 | c_abalone −0.120 |
| Poisson | 2 | −14.4 | c_abalone +0.0017 ± 0.0011 (2/5) |

0/19 pairs break the rule; no non-finite row. C1 ✓, C3 ✓. C2 ✗ by the letter for StudentT (the
development mean −42.9 is one pair, power_plant −299).
Not a defaults question: Poisson on c_seoulbike scores 19.6 with 90 % coverage 0.41; NB scores 5.97.

## 5. Cauchy, Logistic, Gumbel, Laplace, LogNormal (proposal 021, cycles 15 and 18)
None of the earlier cycles touched these. At the crate defaults (Gaussian on the same splits for scale):

| dataset | Gaussian | Cauchy | Logistic | Gumbel | Laplace | LogNormal |
|---|---|---|---|---|---|---|
| power_plant (y ≈ 454) | 2.68 | 12.88 | 7.61 | 7.43 | 7.65 | 2.84 |
| kin8nm | −0.656 | −0.069 | −0.098 | −0.159 | −0.366 | −0.328 |
| naval | −6.05 | −4.78 | −5.58 | −4.59 | −5.20 | −5.33 |
| protein | 2.859 | 2.941 | 2.901 | 73 681 | 2.987 | — |

**Cause, from the code.** Far from the target the location Hessian underflows (Logistic
2e^{−|z|}/s², Gumbel e^{−z}/s²) or is negative (Cauchy for |z| > 1, i.e. half of all rows even at
the true parameters). XGBoost returns a zero leaf. `initialize: true` is not the fix: it is worse
than the default on 9/23 pairs.

**Prototype.** Per row, h ← max(observed, Fisher) (Damped), with I_μμ = c_loc/s² and
I_{ln s} = c_scale: Cauchy (1/2, 1/2), Logistic (1/3, (π² + 3)/9), Gumbel (1, π²/6 + (1 − γ)²),
Laplace (1, 1), LogNormal (1, 2). Starts in closed form (median; sd·√3/π, IQR/2, mean |y − median|).
Monte Carlo check: mean observed Hessian = the Fisher constant within 2 % for all ten.

**Development sets** (recipe chosen per family by validation rank, never by test):
| family | chosen | power_plant | kin8nm | naval | protein | pairs worse than default |
|---|---|---|---|---|---|---|
| Cauchy | Damped + Location | 12.88 → 2.82 | −0.069 → −0.274 | −4.78 → −5.50 | 2.941 → 2.842 | 0/5 |
| Logistic | Damped + LocationScale | 7.61 → 2.69 | −0.098 → −0.477 | −5.58 → −5.81 | 2.901 → 2.846 | 0/5 |
| Laplace | Damped + LocationScale | 7.65 → 2.73 | −0.366 → −0.425 | −5.20 → −5.27 | 2.987 → 2.835 | 0/5 |
| LogNormal | Damped + Location | 2.84 → 2.75 | −0.328 → −0.416 | −5.33 → −5.66 | — | 0/3 |
| Gumbel | Damped + Raw | 7.43 → 7.27 | −0.159 → −0.307 | −4.59 → −5.41 | 73 681 → 3.12 | 0/5 |

**Unseen sets, run once with the recipe fixed** (`c18`):
| family | pairs ≤ default + 0.005 | largest gain | smallest effect |
|---|---|---|---|
| Cauchy | 5/5 | c_gt_nox 18.68 → 3.09 | c_gt_co −0.068 ± 0.010 |
| Logistic | 5/5 | c_airfoil 6.32 → 2.19 | c_gt_co −0.007 ± 0.013 (2/5) |
| Laplace | 5/5 | c_airfoil 6.27 → 2.28 | c_gt_co −0.013 ± 0.008 |
| LogNormal | 5/5 | c_airfoil 2.87 → 2.16 | c_gt_co −0.040 ± 0.008 |
| Gumbel | **4/5** | c_supercond 49 992 → 3.76 | **c_gt_co +0.016 ± 0.008 (1/5)** |

- **Gumbel is not solved.** With a data start it still blows up on single seeds (power_plant 557 and
  20 802; naval seed 1 21.0), and it is 3.9 behind Gaussian on c_airfoil. Its e^{−z} tail needs its
  own treatment, as Weibull's did in cycle 9.
- **Laplace caution.** Damped with a raw or location-only start is worse than the default on naval
  (+0.75, +1.14). Only Damped + LocationScale passes.
- **Predictions:** H1 ✗ (Gumbel), H2 ✓ for four families, H3 ✓ (curvature alone leaves Logistic
  power_plant at 7.34), H4 ✓, H5 ✗ (Gumbel), H6 ✓ for four families, ✗ for Gumbel.

## 6. A validation guard for tuned models (proposal 022, cycle 19)
The rule: after the search, fit the pick and the untuned default the way the final model is fitted
(validation early stopping) and keep the one with the lower validation NLL. It came out of §1 after
the fact (development sets: losses 2/10 → 0/10, mean gain −0.008 → −0.019), so it was run once on
the 14 unseen pairs. Two axes, TPE, 20 trials; test NLL − untuned default.

| | plain | guarded |
|---|---|---|
| pairs worse than the default by > 0.005 | 2/14 (Gaussian c_parkinsons +0.052, c_gt_co +0.020) | **0/14** |
| mean over pairs | −0.017 | −0.023 |
| median over pairs | −0.009 | −0.018 |
| same search with the default scored first: losses | 2/14 | 0/14 |
| same, mean | −0.033 | −0.024 |

- **The guard is safe with two axes** (0/14, 0/14, and 0/2 in the 5-fold arm) **but not free.** On
  Poisson c_seoulbike it gives up half the gain (−0.140 → −0.071; −0.316 → −0.141). Poisson is
  badly calibrated there (90 % coverage 0.41) and its validation NLL is noisy.
- **The search still overestimates.** CV gain over the default −0.082, test gain −0.022: 27 %
  survives. Correlation of CV gain with test gain ≤ 0.2 on 8/14 pairs.
- **Fold size is not the cause.** With 5 folds instead of 3: Gaussian naval +0.053 → +0.012,
  Poisson bike_cnt +0.030 → +0.053. The mechanism stays open.
- **With the wide space the guard is not enough** (§8).
- G1 ✓, G2 ✓, G3 ✓ (✗ for the default-first arm), G4 ✓, M1 ✗.

## 7. Training on the CRPS: Poisson and NegativeBinomial (proposal 023, cycle 20)
`LossFn::Crps` has a closed form only for these two. 10 (family, set) pairs × 5 seeds, same budget
as everywhere; the NLL arm reproduces the earlier default rows bit for bit. Test CRPS; relative =
CRPS-trained / NLL-trained − 1.

| family | dataset (mean count) | NLL-trained | CRPS-trained | relative | CRPS-trained rounds |
|---|---|---|---|---|---|
| Poisson | c_abalone (10) | 1.2009 | 1.2005 | −0.04 % | 239 |
| Poisson | synth_pois | 1.654 | 1.669 | +0.9 % | 549 |
| Poisson | bike_casual (36) | 6.44 | 9.50 | +48 % | 1999 (cap, 5/5) |
| Poisson | bike_cnt (190) | 18.2 | 134.8 | +640 % | **0** (5/5) |
| Poisson | c_seoulbike (700) | 76.5 | 505.6 | +562 % | **0** (5/5) |
| NB | synth_nb_lo | 4.017 | 4.126 | +2.7 % | 269 |
| NB | bike_casual | 6.07 | 6.56 | +8.2 % | 1999 (cap, 5/5) |
| NB | c_abalone | 1.207 | 1.612 | +34 % | 1999 (cap, 5/5) |
| NB | bike_cnt | 17.2 | 37.2 | +116 % | 1999 (cap, 5/5) |
| NB | c_seoulbike | 69.8 | 407.7 | +485 % | 1941 (cap, 4/5) |

- CRPS training beats NLL training on test CRPS by more than 1 % on 0/10 pairs.
- **Defect 1 — Poisson above about 100.** The gradient sums stop at
  k = min(max(y, λ) + 4√λ, 100) (`poisson.rs`, `analytical_crps_gradients_impl`). On bike_cnt and
  c_seoulbike the model never leaves its start values (best round 0 in 10/10 cells; test NLL 86 and
  275 against 6.4 and 19.6). Nothing warns the user.
- **Defect 2 — NB.** Table cap 220; above it +116 % and +485 %. Below it the model is still at the
  round cap after 2000 rounds. `new_mean_dispersion` silently falls back to (total count, probs),
  the coordinates cycle 4 found collinear, so none of the later NB work reaches this path.
- K1 ✓, K2 ✓, K3 ✓, K4 ✓. No code was changed; a fix needs its own proposal.

## 8. A wider search space at the same budget (proposals 024 and 025, cycles 21 and 22)
Two axes (mcw, λ) against five (plus max_depth 2–8, subsample and colsample_bytree in [0.5, 1]).
TPE, 3 folds, 20 trials, the ten development pairs. Test NLL − untuned default.

| family | dataset | two axes | wide | wide − two axes (seeds better) | median depth picked |
|---|---|---|---|---|---|
| Gamma | kin8nm | −0.053 | −0.127 | −0.073 ± 0.018 (5/5) | 5 |
| Gamma | naval | −0.025 | −0.178 | −0.153 ± 0.045 (5/5) | 5 |
| Gaussian | gaussian_sim | −0.008 | −0.008 | +0.000 ± 0.002 (3/5) | 2 |
| Gaussian | kin8nm | −0.025 | −0.143 | −0.118 ± 0.021 (5/5) | 6 |
| Gaussian | naval | **+0.053** | −0.127 | −0.180 ± 0.013 (5/5) | 5 |
| Gaussian | power_plant | −0.030 | −0.059 | −0.029 ± 0.016 (4/5) | 6 |
| Gaussian | protein | −0.021 | −0.064 | −0.043 ± 0.023 (4/5) | 4 |
| NB | bike_cnt | −0.004 | −0.057 | −0.053 ± 0.002 (5/5) | 6 |
| Poisson | bike_cnt | **+0.030** | −0.124 | −0.154 ± 0.012 (5/5) | 5 |
| Poisson | synth_pois | −0.001 | −0.007 | −0.006 ± 0.003 (3/5) | 2 |
| mean | | −0.008 | **−0.089** | | |

- Development pairs: wide is worse than the default on 0/10; two axes on 2/10. Both selection
  failures of §1 disappear without the guard.
- Wide at 10 trials (−0.071) beats two axes at 20 (−0.008). The tuning gain of cycles 13–14 was
  small because of the space, not the sampler or the budget.
- Best CV score: wide lower on 9/10 (median −0.044).
- **Not shown: which axis.** Within a search the fold score moves with depth (best at 5–6 on six
  pairs, at 2 on both simulations) and only weakly, with unstable sign, with either subsampling
  axis. A depth-only and a subsampling-only arm would settle it; not run.
- **Caveat, naval.** Every arm, the default included, stops at the 2000-round cap there. Deeper
  trees do more per round, so part of naval's gain is a larger effective budget.
- S1 ✓, **S2 ✗** (I predicted the wide space would lose to the default at least as often as two
  axes), S3 ✓, S4 ✓.

**Confirmation on the 14 unseen pairs (proposal 025, run once):**
| family | dataset | two axes | wide | wide − two axes (seeds better) |
|---|---|---|---|---|
| Gamma | c_airfoil | −0.041 | −0.104 | −0.063 ± 0.035 (4/5) |
| Gamma | c_gt_co | −0.014 | −0.032 | −0.018 ± 0.004 (5/5) |
| Gamma | c_gt_nox | −0.002 | −0.024 | −0.022 ± 0.008 (4/5) |
| Gamma | c_parkinsons | −0.044 | −0.334 | −0.290 ± 0.053 (5/5) |
| Gamma | c_supercond | −0.028 | −0.055 | −0.027 ± 0.008 (5/5) |
| Gaussian | c_airfoil | −0.026 | −0.143 | −0.118 ± 0.044 (5/5) |
| Gaussian | c_gt_co | +0.020 | **+0.190** | **+0.170 ± 0.117 (0/5)** |
| Gaussian | c_gt_nox | −0.006 | −0.053 | −0.047 ± 0.007 (5/5) |
| Gaussian | c_parkinsons | +0.052 | −0.199 | −0.251 ± 0.032 (5/5) |
| Gaussian | c_supercond | +0.003 | **+0.025** | +0.023 ± 0.047 (3/5) |
| NB | c_abalone | −0.001 | −0.005 | −0.004 ± 0.001 (5/5) |
| NB | c_seoulbike | −0.012 | −0.067 | −0.055 ± 0.003 (5/5) |
| Poisson | c_abalone | −0.003 | −0.007 | −0.004 ± 0.001 (5/5) |
| Poisson | c_seoulbike | −0.140 | −1.029 | −0.890 ± 0.178 (5/5) |
| mean (median) | | −0.017 (−0.009) | **−0.131 (−0.054)** | |

- The size of the gain holds: mean −0.131 against −0.017 (−0.062 against −0.008 without Poisson
  c_seoulbike); wide beats two axes on 12/14 pairs.
- **It is not safe.** Wide is worse than the untuned default on 2/14 pairs, both Gaussian, and the
  validation guard keeps the bad picks (guarded wide: c_gt_co +0.166, c_supercond +0.015).
- **Gaussian c_gt_co.** Validation NLL 1.11–1.64 by seed, test 1.27–3.20; one seed moves from 2.63
  to 3.20 while its CRPS improves (0.454 → 0.433). The target is heavy-tailed and Gaussian is the
  wrong family (untuned: Gaussian 1.68, Gamma 1.11, StudentT 1.00). A few test rows carry the
  Gaussian NLL, and neither the folds nor the validation split see them. Gamma on the same data
  gains (−0.032).
- W1 ✓ (13/14), **W2 ✗** (2 losses, predicted ≤ 1), **W3 ✗** (10/14 better by > 0.01, 2/14 worse),
  W4 ✓. By the rule fixed in advance: no recommendation; both tables go in the README.

## 9. Looked at and not run; leads from reading the code
- **Adaptive per-round λ.** Feasible in the XGBoost backend (the booster takes parameters between
  rounds). Not prototyped: the published rules assume a fixed number of rounds, and here λ
  interacts with early stopping and with the Damped curvature, so a rule would need its own
  derivation and proposal. §8 also makes it a low priority: the best λ spans four decades across
  sets, and depth moved the fold score far more than λ did.
- **Which wide-space axis carries the gain** (depth-only and subsampling-only arms).
- **`spline_flow.rs:1076`** still passes `.target_cost(0.0)` to its start-value L-BFGS, the pattern
  fixed in `base.rs` on July 23. Not touched. (Fixed in §11.)
- **`log_normal.rs`, rows with y ≤ 0:** the parallel branch (n ≥ 4096) returns Hessian 0, the
  serial branch 1e-6. Not touched. (Fixed in §11.)
- **`dist_select`** scores StudentT at the capped start df (§2). (Fixed in §11.)
- **Unverified** (from a code-reading pass, not confirmed by a run or by me line by line):
  MVN_LoRa rank collapse; ZIPoisson / ZINB negative curvature on zero rows; Mixture curvature
  unsigned; ZAGamma / ZABeta in collinear coordinates.

## 10. Decisions for Felix (all approved 2026-11-13; outcome in §11)
1. **Default change for four families?** Cauchy and LogNormal → Damped + Location; Logistic and
   Laplace → Damped + LocationScale. Evidence: 0/18 development pairs and 20/20 unseen pairs
   within the rule; gains up to 15 nats on non-centred targets (§5). Results change for these
   families. Gumbel stays as it is.
2. **CRPS training of Poisson and NB (§7).** Fix (own proposal), or return an error / warning for
   counts above the caps? Today it fails silently.
3. **The validation guard (§6).** Document only (done in the README draft), or add an opt-in
   helper next to `hyper_opt`? It is safe for the two-axis space and not for the wide one.
4. **Search-space guidance (§8).** The README states both results and recommends nothing. A
   recommendation needs the depth-only / subsampling-only arms and a selection score that is
   robust to tail rows.
5. **Two small code leads (§9):** `spline_flow.rs:1076` `target_cost(0.0)`; LogNormal Hessian for
   y ≤ 0.
6. **`dist_select`:** score StudentT at a fitted df instead of the capped start (§2), or document
   it as a shortlist only (README draft does the latter).
7. **Still open from cycle 13:** whether to recommend the normalised tuning axis.

## 11. The decisions, applied (proposal 026; Felix: "apply all of these")
Where §10 offered an either/or, the more complete option was taken. Library tests: 264 unit and
129 integration tests pass; the LightGBM build compiles. Nothing is committed.

| # | decision | what was done | check |
|---|---|---|---|
| 1 | four family defaults | `new()`: Cauchy and LogNormal Damped + Location; Logistic and Laplace Damped + LocationScale. Old saved models load as trained (Observed, Raw). Gumbel unchanged. | 190/190 cells bit-identical to the tested recipe (D1 ✓) |
| 2 | CRPS for large counts | Poisson and NB: loss, gradient and Hessian summed over the whole distribution; large-count rows use the expected CRPS curvature. Rows under the old caps run the old code. | table below (R2–R5 ✓) |
| 3 | validation guard | `hyper_opt::validation_guard` (opt-in), also in the prelude | unit test |
| 4 | two code leads | spline-flow start fit without `target_cost(0.0)`; LogNormal y ≤ 0 rows inert on both paths | unit tests (the spline test fails with the old line: −0.61 against −2.25 per row) |
| 5 | `dist_select` | ranks by the new `Distribution::marginal_fit`; StudentT overrides it with the free fit | picks unchanged, 22/45 (D2 ✗, D3 ✗) |
| 6 | normalised tuning axis | README recommendation for the two-parameter search, with cycle 13's numbers and its limits | — |

**CRPS training after the fix** (test CRPS, 5 seeds; relative to NLL training):
| family | dataset | NLL-trained | before | now | now, relative |
|---|---|---|---|---|---|
| Poisson | bike_casual | 6.44 | 9.50 | 6.77 | +5.2 % |
| Poisson | bike_cnt | 18.23 | 134.8 | 19.40 | +6.5 % |
| Poisson | c_seoulbike | 76.46 | 505.6 | 80.01 | +4.7 % |
| NB | bike_casual | 6.07 | 6.56 | 6.29 | +3.7 % |
| NB | bike_cnt | 17.22 | 37.17 | 19.19 | +11.5 % |
| NB | c_seoulbike | 69.79 | 407.7 | 70.62 | +1.2 % ± 1.4 |

- **The truncation was not the whole problem.** With the sums fixed and the metric Hessian kept,
  Poisson bike_cnt trained but stood at 56.6 after 2000 rounds. The metric E[g²] grows like λ²/3
  in the log rate, the curvature of the CRPS like λ^1.5/√π, so the step shrinks by about 0.6·√λ.
  The expected curvature 2 Σ_k (∂F(k)/∂θ)² has the right units at every rate. This was added
  after a smoke cell and written into the proposal before the full run.
- **Not addressed:** small counts are unchanged (c_abalone identical in 10/10 cells; NB there is
  still +34 %, at the round cap, in the collinear coordinates). CRPS training beats NLL training
  by more than 1 % on 0/10 pairs, and 29/50 cells are at the round cap. The README keeps "train on
  the NLL".
- **Two simulations moved slightly** (synth_pois 1.669 → 1.668, synth_nb_lo 4.126 → 4.136): they
  contain rows above the old caps. I had predicted they would not move (R1 ✗ by the letter).
- **`dist_select`: my diagnosis in §2 was wrong.** The capped start df cost StudentT 0.011 NLL per
  row on Gaussian labels, but removing it changes no pick: on protein the free fit ties with
  Gaussian. The marginal of the labels does not show heavy-tailed conditional residuals.

**Still open after this:** Gumbel; NB CRPS in (mean, dispersion) coordinates; which wide-space axis
carries the gain and a selection score robust to tail rows; adaptive λ; the unverified leads of §9.
