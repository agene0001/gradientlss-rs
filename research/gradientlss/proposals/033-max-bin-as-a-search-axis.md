# Proposal 033 — should `max_bin` be in the search space?

Status: **run** 2026-11-14 (M0 ✓, M1a ✗, M1b ✓ / ✗, M2 ✗ / ✓, M3 ✗; inclusion rule fails (a), (b), (c);
stage 2 not run). **Pre-registered** 2026-11-14, after one plumbing check on a dataset outside the test
pairs (below) and before any `max_bin` fit on any of the 36 pairs. Skill:
`automl-hyperparameter-research` (search-space design; Tier 2). Felix: "lets look into whether it
is a parameter that should be included in search space".

## Question
Every fit of cycles 1–28 used the backend's 256 histogram bins per feature. `max_bin` is a
dataset-level parameter: fewer bins coarsen the candidate splits (a regulariser, and faster),
more bins refine them. The library has no built-in search space; the question is whether the
README's tuning advice should name `max_bin` next to `min_child_weight`, `lambda` and the two
subsampling axes.

## Data
The 36 (family, dataset) pairs of 024 / 025 / 032 (10 development, 14 unseen, 12 fresh). No
`max_bin` result has been seen on any of them, the range and the rule below are fixed now, and
nothing about the arm is tuned on them, so they serve as a test set for this question. (I told
Felix a fresh download would be needed; that was stricter than necessary. What is spent is their
use for the subsampling claim, which was derived from them.)

## Plumbing check (done before this pre-registration, Gamma synth_gamma, seed 0, not one of the 36)
`BENCH_MAX_BIN=256` equals the unset default bit for bit (NLL 2.4343); `BENCH_MAX_BIN=16` gives
2.6982; a 4-trial `bin` search draws max_bin 123, 124, 555, 18 and its refit runs. So the
parameter reaches the booster, and 16 bins cost 0.26 on that simulation: M1b below is written
knowing this one number.

## Stages
- **Stage 0, sensitivity without a search.** The untuned default at max_bin 16, 64, 1024, and at
  256 as a reproduction check (`c29_mb<k>`): 36 pairs × 5 seeds each.
- **Stage 1, `bin`.** mcw, λ (log-uniform [10⁻³, 10³]) and max_bin (log-uniform integer
  16–1024), 20 trials, TPE, 3 folds, refits at k ∈ {10, 20}; everything else as 027 / 032. Tags
  `c29_bin_k<k>`. References: the two-axis arms `c13_raw`, `c19_tpe`, `c27_2d`; untuned
  `c12_off`, `c19_def` / `c17_new`, `c27_def`. "Guarded" keeps a refit only if its validation NLL
  beats the untuned fit's.
- **Stage 2, `subsbin`** (subsample, colsample_bytree and max_bin on top of the two axes) against
  `subs`: run only if stage 1 passes the inclusion rule.
Binary `/tmp/rb_c29`, one CSV per stream.

## Predictions
- **M0 (reproduction).** `c29_mb256` equals the untuned reference rows bit for bit on all 180 cells.
- **M1a (more bins do nothing).** Untuned, max_bin 1024 is within ±0.005 of 256 on at least 30/36
  pairs.
- **M1b (few bins cost).** Untuned, max_bin 16 is worse than 256 by more than 0.005 on at least
  18/36 pairs and better by more than 0.005 on at most 6/36.
- **M2 (the search does not use it).** Plain `bin` beats the plain two-axis search by more than
  0.005 on at most 9/36 pairs and is worse by more than 0.005 on at least 6/36 (a third axis
  takes trials from the two that matter).
- **M3 (no stable optimum).** On at least 24/36 pairs the max_bin picked at 20 trials differs by
  a factor of 8 or more between the lowest and the highest of the five seeds.

## Inclusion rule
`max_bin` goes into the README's tuning advice only if, at 20 trials, all three hold over the 36
pairs: (a) guarded `bin` is worse than the untuned default by more than 0.005 on 0 pairs;
(b) guarded `bin` is better than the guarded two-axis search on at least 24 pairs; (c) the median
over pairs of guarded `bin` − guarded two axes is ≤ −0.005. Then stage 2 runs and the advice is
worded from both. Otherwise the README says it was tested and is not worth a search axis, with
the stage 0 table, and stage 2 is not run. No default changes either way (256 is the backends').

## Cost
720 untuned fits and 180 three-axis searches whose folds are re-binned per trial: about 6
stream-hours on a free machine; the machine is shared.

## Result (2026-11-14, report 016 §9)
720 untuned fits, 180 searches, 360 refits, no non-finite row, one CSV per stream. Summaries:
`c29_maxbin_summary.py <csv> bin=<logs>`. Stage 2 was not run (the rule failed).

**Stage 0: untuned, test NLL at max_bin k minus at 256, mean over 5 seeds, 36 pairs.**

| max_bin | better by > 0.005 | worse by > 0.005 | within ±0.005 | median | beyond 2 s.e.: better / worse |
|---|---|---|---|---|---|
| 16 | 7 | 25 | 4 | +0.040 | 3 / 19 |
| 64 | 12 | 12 | 12 | −0.001 | 5 / 5 |
| 1024 | 4 | 16 | 16 | +0.003 | 1 / 6 |

Largest effects: 16 bins: Poisson n_metro +22.8, Poisson c_seoulbike +4.6, Poisson bike_cnt
+3.0, naval +1.43 / +1.44, kin8nm −0.085 / −0.101. 64 bins: naval +0.58 (both families), Gaussian
c_supercond +0.14, Poisson n_metro −2.06, kin8nm −0.058 / −0.052, Gaussian n_gridstab −0.048.
1024 bins: Poisson c_seoulbike +0.39, Gaussian protein +0.086 (± 0.119), power_plant +0.051,
naval +0.044 / +0.041, kin8nm +0.016 / +0.029. On six pairs 1024 equals 256 bit for bit (c_airfoil,
bike_cnt, n_sgemm in both families: no feature has more than 256 distinct values); on n_sgemm
every setting is identical (no feature has more than 16).

**Stage 1: `bin` against the two-axis search, test NLL minus the untuned default, 36 pairs.**

| | 10 trials: two axes | 10 trials: `bin` | 20 trials: two axes | 20 trials: `bin` |
|---|---|---|---|---|
| plain: mean (median) | −0.046 (−0.013) | −0.070 (−0.020) | −0.054 (−0.017) | −0.055 (−0.017) |
| plain: worse than default by > 0.005 | 7 | 5 | 6 | 5 |
| guarded: mean (median) | −0.041 (−0.012) | −0.059 (−0.019) | −0.056 (−0.021) | −0.041 (−0.016) |
| guarded: worse than default by > 0.005 | 2 | 3 | 1 | 2 |

By set at 20 trials, guarded `bin` − guarded two axes: development better on 9/10 (median
−0.008), unseen 7/14 (0.000), fresh 7/12 (−0.006).

- **M0 ✓.** `c29_mb256` equals the untuned reference rows on 180/180 cells (NLL, CRPS, validation
  NLL, best iteration).
- **M1a ✗.** 1024 bins is within ±0.005 of 256 on 16/36 pairs, not ≥ 30, and worse on 16. More
  bins are not free.
- **M1b ✓ / ✗.** 16 bins is worse on 25/36 (≥ 18 ✓) and better on 7/36 (≤ 6 ✗: both kin8nm
  pairs, Gaussian protein, c_gt_co, n_gridstab, Gamma n_concrete, NB n_news).
- **M2 ✗ / ✓.** Plain `bin` beats the plain two-axis search by more than 0.005 on 17/36 pairs
  (predicted ≤ 9) and is worse on 8/36 (≥ 6 ✓). The search does use the axis.
- **M3 ✗.** The picked max_bin spans a factor ≥ 8 over the seeds on 16/36 pairs, not ≥ 24. On
  some datasets the pick is stable and agrees with stage 0: kin8nm 16–23 and 16–30, Gamma
  c_parkinsons 25–32, Gaussian n_gridstab 27–48, naval 332–595 and 234–972. Median pick over all
  searches 89; below 64 in 71/180, above 512 in 26/180.

**The inclusion rule, at 20 trials: fails all three.**
- (a) Guarded `bin` is worse than the untuned default on 2/36 pairs, needed 0: Gaussian protein
  +0.123 (seed 2: validation NLL 2.802 against 2.807 untuned, test NLL 3.69 against 2.95, test
  CRPS within 2 %: a few test rows; the search picked max_bin 208 there) and Poisson c_seoulbike
  +0.078 (untuned NLL 19.6, the wrong family; picks 28–235). The guarded two-axis search: 1/36.
- (b) Guarded `bin` is better than the guarded two-axis search on 23/36 pairs, needed 24.
- (c) The median of guarded `bin` − guarded two axes is −0.003, needed ≤ −0.005 (mean +0.014:
  Poisson n_metro +0.59, Gaussian protein +0.15, Poisson c_seoulbike +0.15).

**What it is and is not.**
- Not a null axis. Where stage 0 shows a real preference the search finds it (plain `bin` −
  plain two axes, seeds better): kin8nm −0.090 (Gaussian) and −0.058 (Gamma), 5/5 seeds each; Gamma c_parkinsons −0.101, 5/5;
  Gaussian naval −0.094, 5/5; Gaussian n_gridstab −0.031, 4/5.
- It costs trials where it cannot matter: on Gaussian n_sgemm, where every max_bin gives the same
  model, `bin` is +0.048 ± 0.026 worse than two axes (1/5 seeds better). That is the price of a
  third axis at 20 trials, measured with the axis switched off by the data.
- At 10 trials `bin` was ahead on the means (guarded −0.059 against −0.041) and at 20 trials
  behind (−0.041 against −0.056); the count of pairs barely moved (22 → 23). The means are three
  pairs. The `bin` refit has a lower validation NLL than the two-axis refit on 16/36 pairs at 20
  trials (19/36 at 10).
- Within a search the fold score barely moves with max_bin (rank correlation with log max_bin:
  median −0.10; only the ≤ 32 bucket is visibly worse, +0.007 against the search's median).
- Exploratory, not pre-registered: choosing among {16, 64, 256, 1024} by validation NLL alone,
  per seed, is better than 256 by more than 0.005 on 14/36 pairs and worse on 8. The stage-0
  difference (64 − 256) does not predict where `bin` beats two axes (rank correlation 0.09).

**Decision (by the rule).** `max_bin` does not go into the README's tuning advice as a search
axis. The README records the test, the stage 0 table, and two facts a user can act on without a
search: 16 bins is usually much worse, and 1024 is not a safe "more is better". No default
changes. M1a, M2 and M3 were wrong in the direction of the axis mattering more than I thought;
the rule still fails: (b) by one pair, (c) by 0.002, (a) on two losses the guard did not catch.
