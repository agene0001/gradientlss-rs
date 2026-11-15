# Cycle 13 report — Hessian normalisation as a tuning axis

Date: 2026-11-12. Proposal 016. Rows: `experiments_v2.csv` tags `c13_raw_k{5,10,20}` and
`c13_norm_k{5,10,20}`; script `run_c13.sh`; search diagnostics `hpo_summary.py`.
**No default change. One library fix (hyper_opt fold models).**

## 0. Library fix found while preparing
- **Bug:** `hyper_opt`'s fold models ignored `hessian_normalization` and `stopping_row_cap`. A search
  on a normalised model therefore tuned the unnormalised model.
- **Fix:** both settings are now copied in `src/hyper_opt.rs`, with getters on `GradientLSS`.
- **Test:** `hessian_normalization_reaches_the_fold_models`, checked to fail without the fix.

## 1. Design
- **Search:** tune `min_child_weight` and `lambda` with hyper_opt (TPE, 3-fold CV on the training
  split, log range [10⁻³, 10³] for both, 20 trials).
  - The range is the same on both axes: raw (Off) and normalised (Marginal).
  - Everything else is the default recipe.
- **Refit:** the best-so-far params after 5, 10 and 20 trials, with validation early stopping;
  scored on test.
- **Scope:** 10 dataset/family pairs × 5 seeds.

## 2. Results (test NLL, norm − raw; negative = the normalised axis is better)
| family | dataset | k = 5 | k = 20 | seeds better (k = 20) |
|---|---|---|---|---|
| Gaussian | kin8nm | +0.001 | **−0.035** | 5/5 |
| Gaussian | power_plant | −0.028 | −0.022 | 4/5 |
| Gaussian | protein | −0.028 | −0.024 | 3/5 |
| Gaussian | naval | **+1.22** | −0.049 | 4/5 |
| Gaussian | gaussian_sim | −0.002 | −0.000 | 4/5 |
| Gamma | kin8nm | −0.021 | **−0.025** | 5/5 |
| Gamma | naval | **+1.28** | +0.022 ± 0.028 | 2/5 |
| NB | bike_cnt | −0.001 | −0.008 | 5/5 |
| Poisson | bike_cnt | +0.014 | +0.017 ± 0.027 | 3/5 |
| Poisson | synth_pois | +0.000 | +0.001 | 1/5 |

Against the untuned default (mcw = λ = 1, Off), tuned-norm at k = 20 gives:
- Gaussian kin8nm −0.059, power_plant −0.052, protein −0.045, gaussian_sim −0.008, naval +0.004;
- Gamma kin8nm −0.079, naval −0.003;
- NB bike −0.012;
- Poisson bike_cnt **+0.046**.

Of the four recipes (untuned Off, untuned Marginal, tuned raw, tuned norm), tuned-norm is the best
on 6/10 sets.

## 3. Predictions
- **T1 ✗ (narrowly).** It wins clearly on 5–6 sets, but Poisson bike_cnt (+0.017 ± 0.027) and Gamma
  naval (+0.022 ± 0.028) miss the +0.005 band, each within one s.e.
- **T2 ✗.** Speed is reversed at small budgets on naval. The normalised range's top end (10³ rows'
  worth) is crushing regularisation where per-row Hessians are ~10⁶; the raw range cannot reach it.
- **T3 ✓.** CV and test signs agree on 7/10 sets.
- **T4 ✓.** Tuning matters: tuned raw beats untuned Off by ≥ 0.01 on 5/10 sets.

## 4. Readings
1. **20 trials or more:** the normalised axis is the better tuning axis for the Gaussian, Gamma and
   NB location scales on 7/8 of those sets. That is consistent with cycle 12: the optimum is
   "rows' worth", not raw units.
2. **Poisson bike_cnt:** tuning mcw/λ does not transfer from CV to test on either axis. Both tuned
   arms are worse than the untuned default (+0.030 raw, +0.046 norm). This is a selection problem
   on that dataset, not an axis problem.
3. **Small budgets with a wide range:** the normalised axis is dangerous unless the upper end is
   narrowed. A range like [10⁻³, 10] rows' worth would exclude the destructive corner, but that is
   post hoc and untested.

## 5. Decision
- No default change.
- The README reports the measured result and its caveats.
- Recommending the normalised axis for tuning is Felix's call: the two misses are noise-level.

## 6. Next (ranked)
1. Done (§7): the narrowed range trades the small-budget failure for most of the gains.
2. Poisson bike_cnt CV → test transfer: check whether KFold CV on time-ordered bike rows is
   optimistic against the bench's random test split, e.g. by comparing CV and test NLL gaps.
3. Early stopping's relative min-delta (cycle 10): still not unit-free.

## 7. Follow-up (cycle 13b): narrowed normalised range [10⁻³, 10]
Addendum N in proposal 016.
- **Fixed:** the naval small-budget failure (k = 5: +0.015 / +0.016, against +1.2).
- **The cost at k = 20 is larger.** Against the wide normalised range: kin8nm +0.035, protein
  +0.046, naval +0.045, Gamma kin8nm +0.031. Against raw: Gamma naval +0.034, gaussian_sim +0.006,
  Poisson bike +0.043.
- **Why:** the best λ in rows' worth ranges from ≈ 0.05 (naval) to ≈ 700 (kin8nm). Normalisation
  removes units, not the dataset-to-dataset spread, so no fixed range is both safe at 5 trials and
  complete at 20.
- **Decision:** the narrowed range is not recommended. The README advises the wide range with
  ≥ 20 trials. Whether to recommend the wide normalised axis is still Felix's decision.
