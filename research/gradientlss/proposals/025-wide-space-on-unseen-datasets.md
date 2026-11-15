# Proposal 025 — the wide search space on the unseen datasets (confirmation of 024)

Status: **run** 2026-11-13 (W1 ✓, W2 ✗, W3 ✗, W4 ✓). **Pre-registered** 2026-11-13, after proposal 024's result on the development sets and
before any wide-space run on an unseen set. The two-axis reference on these sets (`c19_tpe`,
proposal 022) was still running and its summary had not been read. Skill:
`automl-hyperparameter-research` (confirmation; Tier 1 and Tier 2).

## Question
On the ten development pairs the wide space (mcw, λ, max_depth 2–8, subsample and colsample_bytree
in [0.5, 1]) gave a mean test gain over the untuned default of −0.089 at 20 trials, against −0.008
for the two axes, and no pair worse than the default (024). Those pairs are the ones every default
of cycles 1–13 was developed on, at depth 4. Does the gain hold on sets no cycle has tuned anything
on?

## Protocol
The 14 unseen pairs of proposal 022: Gaussian and Gamma (mean, shape) × {c_airfoil, c_parkinsons,
c_gt_nox, c_gt_co, c_supercond}; NB (mean, dispersion) and Poisson × {c_abalone, c_seoulbike}.
5 seeds, 3-fold CV, TPE, 20 trials, refit at k ∈ {10, 20} with validation early stopping. Same
binary and settings as 024 (`/tmp/rb_c20`, `BENCH_HPO_SPACE=wide`), tag `c22_wide_k<k>`.
Reference: `c19_tpe_k<k>` (two axes, same sets, seeds, folds). Untuned default: `c19_def`.
Run once. Both arms read plain and with the validation guard.

## Predictions
- **W1 (Tier 1).** At 20 trials the wide space has the lower best CV score on at least 10/14 pairs.
- **W2 (plain).** Plain wide at 20 trials is worse than the untuned default by more than 0.005 on
  at most 1/14 pairs.
- **W3 (guarded).** Guarded wide beats guarded two-axis by more than 0.01 test NLL on at least 7/14
  pairs and is worse by more than 0.01 on none.
- **W4 (size).** The mean test gain over pairs of plain wide is at least 0.02 better than the mean
  gain of plain two-axis.

## Decision rule
No default change (the library has no default space). W2 and W3 hold: the README's tuning section
recommends searching depth and subsampling together with mcw and λ, with both tables. Either
fails: the README reports the development result and the unseen result side by side and makes no
recommendation.

## Not answered here
Which of the three added axes carries the gain (024's trace: the fold score is lowest at depth 5–6
on six of ten pairs and at depth 2 on the two simulations). That needs a depth-only and a
subsampling-only arm.

## Cost
70 searches; c_supercond (81 features) dominates. ≈ 8 stream-hours.

## Result (2026-11-13, report 015 §8)
70 wide searches, 140 refits, no non-finite row. Test NLL − untuned default at 20 trials, mean
over 5 seeds.

| family | dataset | two axes | wide | wide − two axes (seeds better) | median depth |
|---|---|---|---|---|---|
| Gamma | c_airfoil | −0.041 | −0.104 | −0.063 ± 0.035 (4/5) | 4 |
| Gamma | c_gt_co | −0.014 | −0.032 | −0.018 ± 0.004 (5/5) | 7 |
| Gamma | c_gt_nox | −0.002 | −0.024 | −0.022 ± 0.008 (4/5) | 4 |
| Gamma | c_parkinsons | −0.044 | −0.334 | −0.290 ± 0.053 (5/5) | 7 |
| Gamma | c_supercond | −0.028 | −0.055 | −0.027 ± 0.008 (5/5) | 5 |
| Gaussian | c_airfoil | −0.026 | −0.143 | −0.118 ± 0.044 (5/5) | 6 |
| Gaussian | c_gt_co | +0.020 | **+0.190** | **+0.170 ± 0.117 (0/5)** | 4 |
| Gaussian | c_gt_nox | −0.006 | −0.053 | −0.047 ± 0.007 (5/5) | 5 |
| Gaussian | c_parkinsons | +0.052 | −0.199 | −0.251 ± 0.032 (5/5) | 6 |
| Gaussian | c_supercond | +0.003 | **+0.025** | +0.023 ± 0.047 (3/5) | 3 |
| NB | c_abalone | −0.001 | −0.005 | −0.004 ± 0.001 (5/5) | 3 |
| NB | c_seoulbike | −0.012 | −0.067 | −0.055 ± 0.003 (5/5) | 5 |
| Poisson | c_abalone | −0.003 | −0.007 | −0.004 ± 0.001 (5/5) | 4 |
| Poisson | c_seoulbike | −0.140 | −1.029 | −0.890 ± 0.178 (5/5) | 5 |
| mean (median) | | −0.017 (−0.009) | −0.131 (−0.054) | | |

- **W1 ✓.** Lower best CV score on 13/14 (median −0.028); the exception is Gaussian c_gt_co.
- **W2 ✗.** Plain wide is worse than the default on 2/14 (predicted ≤ 1): Gaussian c_gt_co +0.190
  (worse in 5/5 seeds against two axes), Gaussian c_supercond +0.025.
- **W3 ✗.** Guarded wide beats guarded two-axis by > 0.01 on 10/14, but is worse by > 0.01 on 2/14
  (the same two; +0.188 and +0.050). The guard kept the wide pick on c_gt_co although it was worse
  on test: validation NLL 1.11–1.64 by seed, test 1.27–3.20, and one seed alone moves from 2.63 to
  3.20. That target is heavy-tailed and Gaussian is the wrong family for it (untuned test NLL:
  Gaussian 1.68, Gamma 1.11, StudentT 1.00; `c17_new`); a few test rows carry the Gaussian NLL
  and no held-in score sees them. The wide pick's test CRPS on that seed is the better one
  (0.433 against 0.454).
- **W4 ✓.** Mean −0.131 against −0.017. Without Poisson c_seoulbike: −0.062 against −0.008.

**Decision (by the rule).** W2 and W3 fail, so the README makes no recommendation and shows both
tables: the wide space beats the two-axis search on 9/10 development pairs and 12/14 unseen
pairs, often by a lot, and is worse than both the default and the two-axis search on two unseen
Gaussian pairs, where the validation guard does not catch it.
