# Proposal 024 — does a wider search space pay at the same budget?

Status: **run** 2026-11-13 (S1 ✓, S2 ✗, S3 ✓, S4 ✓; confirmation on unseen sets is proposal 025). **Pre-registered** 2026-11-13, after proposal 017's results and one smoke search (synth_pois,
4 trials, to check the plumbing), before any run. Skill: `automl-hyperparameter-research`
(search-space design; Tier 1 and Tier 2).

## Question
Cycles 13–14 tuned two axes (min_child_weight, λ; log [10⁻³, 10³]) with depth 4, no row or column
subsampling. Tree depth and subsampling are the usual next axes. At the same 20 trials, does adding
them give a better held-out model, or does it only give the search more ways to fit the folds
(cycle 14: the CV gain of a pick does not predict its test gain)?

## Spaces
| space | axes |
|---|---|
| `2d` (cycle 13 raw arm, reused) | mcw, λ log-uniform [10⁻³, 10³]; depth 4; no subsampling |
| `wide` | the same two, plus max_depth integer 2–8, subsample and colsample_bytree uniform [0.5, 1] |

eta stays 0.05 with 2000 rounds / patience 50, so depth is not confounded with the round cap more
than it is in any user's search. Sampler: TPE (the library default), one optimiser per axis.

## Protocol
Cycle 13's 10 (family, dataset) pairs × 5 seeds, 3-fold CV on the training split, 20 trials, refit
at k ∈ {10, 20} with validation early stopping. Bench `BENCH_HPO_SPACE=wide`, binary `/tmp/rb_c20`,
tags `c21_wide_k<k>`; reference `c13_raw_k<k>`; untuned default `c12_off`. Both arms are read plain
and with the validation guard of proposal 022.

## Predictions
- **S1 (Tier 1).** At 20 trials the best CV score of `wide` is lower than `2d`'s on at least 7/10
  sets: more axes, more to gain on the folds.
- **S2 (Tier 2, plain).** Plain `wide` at 20 trials is worse than the untuned default by more than
  0.005 on at least as many sets as plain `2d` (2/10).
- **S3 (Tier 2, guarded).** Guarded `wide` beats guarded `2d` by more than 0.01 test NLL on at
  least 3/10 sets, and is worse by more than 0.01 on none.
- **S4.** The picks use the new axes: the median chosen depth differs from 4 on at least 5/10 sets.

## Decision rule
No default change (the library has no default space). S3 holds: recommend the wide space with the
guard in the README's tuning section, with the table. S3 fails: say that depth and subsampling did
not pay at 20 trials on these sets, and keep the two-axis recommendation.

## Cost
Deeper trees and five axes: about twice cycle 14's arm, ≈ 5 stream-hours.

## Result (2026-11-13, report 015 §8)
50 wide searches, 100 refits, no non-finite row. Test NLL − untuned default, mean over 5 seeds.

| family | dataset | two axes, 20 trials | wide, 20 trials | wide − two axes (seeds better) | median depth picked |
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

- **S1 ✓.** Wide has the lower best CV score on 9/10 sets at 20 trials (median −0.044); the
  exception is gaussian_sim (+0.002).
- **S2 ✗.** Predicted: plain wide worse than the default on at least 2/10. Observed: 0/10 at 10
  and at 20 trials. Both selection failures of cycle 14 (Gaussian naval, Poisson bike_cnt) are gone.
- **S3 ✓.** Guarded wide beats guarded two-axis by more than 0.01 on 8/10 (7/10 at 10 trials),
  worse by more than 0.01 on 0/10. The guard changes nothing for wide at 20 trials (every pick kept).
- **S4 ✓.** Median depth ≠ 4 on 9/10 (protein stays at 4). Subsample medians 0.68–0.86,
  colsample 0.71–0.92.
- Wide at 10 trials (−0.071) already beats two axes at 20 (−0.008).

**What this does not show.**
- Which axis carries it. Within a search the fold score moves with depth (best at 5–6 on six sets,
  at 2 on the two simulations; depth 2 and 8 are 0.1–0.5 worse on naval) and only weakly with
  either subsampling axis (rank correlation −0.5 to +0.4, sign not stable across sets).
- On naval every arm, including the default, stops at the 2000-round cap. A deeper tree does more
  per round, so naval's gain is partly a larger effective budget.
- Unseen data: proposal 025.

**Decision (by the rule).** S3 holds on the development pairs. The confirmation on unseen sets
(proposal 025) then failed two of its four predictions, so the README shows both results and
recommends nothing. No default changes (the library has no default space). My prediction S2 was wrong: I
expected more axes to give the folds more ways to mislead; they did not.
