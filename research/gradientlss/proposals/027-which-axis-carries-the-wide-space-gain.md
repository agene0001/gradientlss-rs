# Proposal 027 — which of the added axes carries the wide-space gain?

Status: **run** 2026-11-14 (A1 ✗, A2 ✗, A3 ✗, A4 ✗ / ✓, A5 ✗). **Pre-registered** 2026-11-14, before any run of the two new arms. Skill:
`automl-hyperparameter-research` (search-space design; Tier 1 and Tier 2). Follows 024 and 025.

## Question
The wide space (mcw, λ, max_depth 2–8, subsample and colsample_bytree in [0.5, 1]) gave a mean
test gain over the untuned default of −0.089 (development) and −0.131 (unseen) at 20 trials,
against −0.008 and −0.017 for the two axes, and was worse than the default on two unseen Gaussian
pairs (c_gt_co +0.190, c_supercond +0.025). 024 could not say which axis does it: within a search
the fold score moves with depth and only weakly with either subsampling axis, but that is a
within-search correlation, not an arm.

## Arms
| arm | axes | fixed |
|---|---|---|
| `2d` (reused: `c13_raw`, `c19_tpe`) | mcw, λ | depth 4, no subsampling |
| `depth` (new) | mcw, λ, max_depth integer 2–8 | no subsampling |
| `subs` (new) | mcw, λ, subsample and colsample_bytree uniform [0.5, 1] | depth 4 |
| `wide` (reused: `c21_wide`, `c22_wide`) | all five | |
| `d6` (new, no search) | none: the untuned default at max_depth 6, XGBoost's own default depth | |

Ranges, sampler (TPE, one optimiser per axis), 20 trials, 3 folds, eta 0.05, 2000 rounds,
patience 50, seeds 0–4 and the refits at k ∈ {10, 20} are those of 024 / 025. Bench
`BENCH_HPO_SPACE=depth|subs`, binary `/tmp/rb_c25`, tags `c25_depth_k<k>`, `c25_subs_k<k>`,
`c25_d6`. Untuned reference: `c12_off` (development), `c19_def` (unseen). The 10 development and
14 unseen pairs of 024 / 025. The unseen pairs are run once, with the rule below fixed now.

## Predictions
"Gain" is test NLL minus the untuned default's, mean over 5 seeds, then over pairs, at 20 trials,
without the guard.
- **A1 (depth carries it, development).** `depth` reaches at least 70 % of `wide`'s mean gain
  (≤ −0.062); `subs` reaches less than 30 % (> −0.027).
- **A2 (unseen).** The same split on the 14 unseen pairs: `depth` ≤ −0.092, `subs` > −0.039. Also
  stated without Poisson c_seoulbike, which is half of the mean: `depth` ≤ −0.043, `subs` > −0.019.
- **A3 (the two unseen losses are not depth).** The wide picks there had median depth 4 and 3. On
  Gaussian c_gt_co `subs` is worse than the default by more than 0.05 and `depth` by at most 0.05.
- **A4 (a fixed deeper tree is not the same thing).** `d6` reaches between 30 % and 70 % of
  `depth`'s mean gain on the development pairs and is worse than the depth-4 default by more than
  0.005 on at least 2 of the 24 pairs.
- **A5 (guard).** Guarded `depth` (keep the refit only if its validation NLL beats the untuned
  fit's) is worse than the untuned default by more than 0.005 on 0/10 development pairs and at
  most 1/14 unseen pairs.

## Decision rule
No default changes: the library has no default space, and the depth is the user's.
- Guarded `depth` worse than the default by more than 0.005 on 0/10 and 0/14 pairs **and** its
  mean gain at least twice the guarded two-axis gain on both sets of pairs: the README's tuning
  section recommends adding `max_depth` to the two axes, with the guard, and gives the table.
- Otherwise the README gets the attribution table and no recommendation.
- `d6` is reported, not recommended, whatever it shows (a depth is not chosen from test data).

## Not answered here
Whether a larger budget changes the split (20 trials only); eta; LightGBM.

## Cost
96 searches at three or four axes and 120 untuned fits. From the wide logs (4.8 + 4.2 search
hours for one arm): about 18 stream-hours, in waves of at most 10 single-thread streams, each
stream under the 2-hour task limit.

## Result (2026-11-14, report 016 §1)
240 searches (two arms × 24 pairs × 5 seeds), 480 refits, 120 untuned depth-6 fits, no non-finite
row. Test NLL minus the untuned default at 20 trials, mean over 5 seeds; "guarded" keeps a refit
only if its validation NLL beats the untuned fit's. Summaries: `c25_axis_summary.py`.

| arm | development: mean | share of wide | worse than default | guarded: mean (worse) | unseen: mean | share of wide | worse than default | guarded: mean (worse) |
|---|---|---|---|---|---|---|---|---|
| two axes | −0.008 | 9 % | 2/10 | −0.019 (0/10) | −0.017 | 13 % | 2/14 | −0.023 (0/14) |
| + max_depth (`depth`) | −0.055 | 61 % | 1/10 | −0.057 (1/10) | −0.081 | 61 % | 2/14 | −0.076 (1/14) |
| + subsample, colsample (`subs`) | −0.054 | 61 % | 0/10 | −0.052 (0/10) | −0.055 | 42 % | 2/14 | −0.045 (0/14) |
| all five (`wide`) | −0.089 | | 0/10 | −0.089 (0/10) | −0.131 | | 2/14 | −0.121 (2/14) |
| untuned, depth 6 (`d6`) | −0.002 | | 5/10 | | −0.032 | | 4/14 | |

Per pair (plain, 20 trials):

| family | dataset | two axes | depth | subs | wide | d6 | depth picked (median) |
|---|---|---|---|---|---|---|---|
| Gamma | kin8nm | −0.053 | −0.091 | −0.103 | −0.127 | −0.025 | 5 |
| Gamma | naval | −0.025 | −0.093 | −0.092 | −0.178 | −0.049 | 5 |
| Gaussian | gaussian_sim | −0.008 | −0.009 | −0.006 | −0.008 | +0.030 | 2 |
| Gaussian | kin8nm | −0.025 | −0.164 | −0.083 | −0.143 | +0.037 | 8 |
| Gaussian | naval | +0.053 | −0.104 | −0.050 | −0.127 | −0.047 | 5 |
| Gaussian | power_plant | −0.030 | −0.078 | −0.053 | −0.059 | +0.017 | 6 |
| Gaussian | protein | −0.021 | **+0.029** | −0.068 | −0.064 | +0.054 | 3 |
| NB | bike_cnt | −0.004 | −0.020 | −0.028 | −0.057 | −0.027 | 5 |
| Poisson | bike_cnt | +0.030 | −0.012 | −0.056 | −0.124 | −0.026 | 6 |
| Poisson | synth_pois | −0.001 | −0.004 | −0.002 | −0.007 | +0.015 | 2 |
| Gamma | c_airfoil | −0.041 | −0.020 | −0.096 | −0.104 | +0.069 | 6 |
| Gamma | c_gt_co | −0.014 | −0.036 | −0.015 | −0.032 | −0.019 | 7 |
| Gamma | c_gt_nox | −0.002 | +0.003 | −0.014 | −0.024 | +0.021 | 4 |
| Gamma | c_parkinsons | −0.044 | −0.444 | −0.111 | −0.334 | −0.298 | 7 |
| Gamma | c_supercond | −0.028 | −0.040 | −0.040 | −0.055 | +0.016 | 5 |
| Gaussian | c_airfoil | −0.026 | −0.058 | −0.092 | −0.143 | −0.038 | 5 |
| Gaussian | c_gt_co | +0.020 | **+0.168** | +0.011 | +0.190 | −0.043 | 4 |
| Gaussian | c_gt_nox | −0.006 | −0.023 | −0.028 | −0.053 | −0.049 | 6 |
| Gaussian | c_parkinsons | +0.052 | −0.255 | +0.026 | −0.199 | −0.260 | 6 |
| Gaussian | c_supercond | +0.003 | **+0.067** | −0.024 | +0.025 | +0.581 | 3 |
| NB | c_abalone | −0.001 | −0.002 | −0.005 | −0.005 | −0.000 | 3 |
| NB | c_seoulbike | −0.012 | −0.034 | −0.038 | −0.067 | +0.002 | 6 |
| Poisson | c_abalone | −0.003 | −0.001 | −0.006 | −0.007 | +0.003 | 4 |
| Poisson | c_seoulbike | −0.140 | −0.454 | −0.338 | −1.029 | −0.434 | 6 |

- **A1 ✗.** Development: `depth` carries 61 % of `wide`'s mean gain, not ≥ 70 %; `subs` also 61 %,
  not < 30 %. The two halves overlap (122 %): either one gets most of what both get.
- **A2 ✗.** Unseen: `depth` −0.081 (needed ≤ −0.092), `subs` −0.055 (needed > −0.039). Without
  Poisson c_seoulbike `depth` is −0.052 (✓ on its own) and `subs` −0.033 (✗).
- **A3 ✗, the opposite.** Gaussian c_gt_co: `depth` +0.168, `subs` +0.011. Gaussian c_supercond:
  `depth` +0.067, `subs` −0.024. Both unseen losses of the wide space are the depth axis, although
  the picked depths there are unremarkable (median 4 and 3, ranges 2–7 and 3–4). The same on the
  development side: Gaussian protein `depth` +0.029 (depth 3 picked), `subs` −0.068.
- **A4 ✗ / ✓.** `d6` reaches 4 % of `depth`'s gain on the development pairs (predicted 30–70 %):
  a fixed deeper tree is not the gain; the gain is in choosing the depth per dataset (2 on the
  simulations, 8 on Gaussian kin8nm). It is worse than the depth-4 default on 9/24 pairs (≥ 2 ✓),
  Gaussian c_supercond by +0.58.
- **A5 ✗.** Guarded `depth` is worse than the default on 1/10 development pairs (Gaussian
  protein +0.007; predicted 0) and 1/14 unseen (Gaussian c_gt_co +0.149).
- **Tier 1.** Lower best CV score than the two-axis search: `subs` on 9/10 and 13/14 pairs,
  `depth` on 8/10 and 8/14, `wide` on 9/10 and 13/14. `depth` has the higher CV score than two
  axes on six unseen pairs and still the better test NLL on three of them.

**Not predicted, seen after the fact (needs fresh data before it is a rule).** The subsampling
axes never cost anything against the two-axis search: `subs` − two axes is negative or within
0.002 on 24/24 pairs, plain `subs` is worse than the default on 0/10 and 2/14 (+0.011, +0.026:
both smaller than the two-axis losses on the same pairs), and guarded `subs` on 0/10 and 0/14
with a gain of −0.052 and −0.045 against −0.019 and −0.023 for the guarded two axes. The safety
half of the criterion I had written for `depth` is met by the arm I expected to do nothing; the
size half (twice the guarded two-axis gain) is met on the development pairs (2.8×) and missed by a
hair on the unseen ones (1.95×).

**Decision (by the rule).** Guarded `depth` is worse than the default on 1/10 and 1/14, so no
recommendation for it; the README gets the attribution table. `d6` is reported only. No default
changes. My model of the wide space was wrong on every prediction about which axis does what.
