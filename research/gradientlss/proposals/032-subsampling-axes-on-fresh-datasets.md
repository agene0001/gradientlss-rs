# Proposal 032 — the subsampling axes on datasets no cycle has used (confirmation of 027's observation)

Status: **run** 2026-11-14 (C1 ✓, C2 ✗, C3 ✗ on the ratio of means). **Pre-registered** 2026-11-14, after Felix answered "yes to all of these" to report 016
§6 (decision 3 read as its first option: confirm on fresh data before recommending) and before
any fit on the new datasets. Skill: `automl-hyperparameter-research` (confirmation; Tier 2).

## What is being confirmed
027, after the fact: adding `subsample` and `colsample_bytree` (uniform [0.5, 1]) to the two-axis
search (mcw, λ) was never worse than the two-axis search (≤ +0.002 on 24/24 pairs), and with the
validation guard never worse than the untuned default (0/24), with a guarded mean gain of −0.052
(development) and −0.045 (unseen) against −0.019 and −0.023 for the guarded two axes. All 24
pairs had been looked at before the observation was made.

## Fresh data
Seven UCI datasets fetched 2026-11-14 and never used in cycles 1–26 (`prep_confirm_data2.py`,
`data_confirm/n_*.csv`, bench names `c_n_<name>`), chosen before any fit:

| dataset | rows | features | target |
|---|---|---|---|
| n_concrete | 1 030 | 8 | compressive strength |
| n_wine (white) | 4 898 | 11 | quality score 3–9 |
| n_appliances | 12 000 of 19 735 | 27 | appliance energy (Wh), heavy upper tail |
| n_sgemm | 12 000 of 241 600 | 14 | kernel run time (ms) |
| n_gridstab | 10 000 | 12 | stability margin (can be negative) |
| n_metro | 12 000 of 48 204 | 20 | hourly traffic volume (count, mean 3 265) |
| n_news | 12 000 of 39 644 | 58 | article shares (count, mean 3 433, max 843 300) |

12 (family, dataset) pairs: Gaussian × {concrete, wine, appliances, sgemm, gridstab}; Gamma (mean,
shape) × {concrete, wine, appliances, sgemm}; Poisson × metro; NB (mean, dispersion) × {metro,
news}. Poisson is not run on n_news (a count with that tail under Poisson would decide every
mean, as Poisson c_seoulbike did in 025).

## Protocol
027's, unchanged: 5 seeds, 3-fold CV, TPE, 20 trials, refits at k ∈ {10, 20} with validation
early stopping, eta 0.05, depth 4, 2000 rounds, patience 50, binary `/tmp/rb_c25`. Arms: untuned
default (`c27_def`), two axes (`c27_2d`), `subs` (`c27_subs`). Run once.

## Predictions (20 trials)
- **C1 (safe with the guard).** Guarded `subs` is worse than the untuned default by more than
  0.005 on 0/12 pairs.
- **C2 (never costs against two axes).** Plain `subs` − plain two axes ≤ +0.005 on at least 11/12
  pairs.
- **C3 (worth it).** The guarded `subs` gain is better than the guarded two-axis gain on at least
  8/12 pairs, and its mean over pairs is at least 1.5 times the guarded two-axis mean.

## Decision rule
- C1, C2 and C3 hold: the README's tuning advice becomes "search `subsample` and
  `colsample_bytree` together with `min_child_weight` and `lambda`, and keep the result only if
  it beats the untuned model on validation", with the three tables (development, unseen, fresh).
- C1 fails: no recommendation; the README says the observation did not hold on fresh data.
- C1 holds and C2 or C3 fails: the README says the axes are safe with the check but not
  reliably better, with the table.
No default changes (the library has no default space).

## Cost
120 searches and 60 untuned fits on at most 12 000 rows: about 4 stream-hours.

## Result (2026-11-14, report 016 §8)
120 searches, 240 refits, 60 untuned fits, no non-finite row. Each stream wrote its own CSV
(appended to `experiments_v2.csv` afterwards). `c27_fresh_summary.py`. Test NLL minus the untuned
default at 20 trials, mean over 5 seeds.

| family | dataset | untuned NLL | two axes | subs | guarded two axes | guarded subs | subs − two axes (seeds better) |
|---|---|---|---|---|---|---|---|
| Gamma | n_appliances | 4.864 | −0.009 | −0.016 | −0.007 | −0.014 | −0.007 ± 0.007 (4/5) |
| Gamma | n_concrete | 2.996 | −0.096 | −0.150 | −0.022 | −0.089 | −0.054 ± 0.037 (4/5) |
| Gamma | n_sgemm | 4.597 | −0.005 | −0.014 | −0.007 | −0.014 | −0.010 ± 0.006 (4/5) |
| Gamma | n_wine | 1.012 | −0.021 | −0.034 | −0.015 | −0.034 | −0.014 ± 0.004 (5/5) |
| Gaussian | n_appliances | 5.391 | **+0.034** | −0.010 | +0.008 | −0.010 | −0.043 ± 0.037 (4/5) |
| Gaussian | n_concrete | 2.917 | **+0.069** | −0.055 | −0.013 | −0.055 | −0.124 ± 0.040 (5/5) |
| Gaussian | n_gridstab | −3.511 | −0.028 | −0.101 | −0.028 | −0.101 | −0.073 ± 0.007 (5/5) |
| Gaussian | n_sgemm | 5.027 | −0.198 | −0.221 | −0.198 | −0.221 | −0.023 ± 0.009 (4/5) |
| Gaussian | n_wine | 1.005 | −0.023 | −0.038 | −0.019 | −0.038 | −0.015 ± 0.007 (4/5) |
| NB | n_metro | 7.299 | −0.035 | −0.042 | −0.035 | −0.042 | −0.007 ± 0.019 (4/5) |
| NB | n_news | 9.036 | −0.055 | −0.044 | −0.055 | −0.044 | **+0.011 ± 0.006 (2/5)** |
| Poisson | n_metro | 39.25 | −1.237 | −1.072 | −1.098 | −1.006 | **+0.165 ± 0.255 (2/5)** |
| mean (median) | | | −0.134 (−0.026) | −0.150 (−0.043) | −0.124 (−0.021) | −0.139 (−0.043) | |

- **C1 ✓.** Guarded `subs` is worse than the untuned default on 0/12 pairs. So is plain `subs`
  (0/12). The two-axis search is worse on 2/12 plain and, for the first time in these cycles, on
  1/12 with the guard (Gaussian n_appliances +0.008).
- **C2 ✗.** Plain `subs` − plain two axes ≤ +0.005 on 10/12, not 11: NB n_news +0.011 ± 0.006 and
  Poisson n_metro +0.165 ± 0.255.
- **C3 ✗ on the ratio.** Guarded `subs` beats guarded two axes on 10/12 pairs (needed 8), but the
  ratio of the means is 1.12, not 1.5. Poisson n_metro (untuned NLL 39: the wrong family for a
  count with mean 3 265) is 74 % and 60 % of the two means; without it the means are −0.036 and −0.060
  (1.69×), and the medians are −0.021 and −0.043. The rule was written on the mean of all pairs.
- At 10 trials: guarded `subs` 0/12 worse than the default, better than guarded two axes on 9/12,
  ratio 1.32.

**Decision (by the rule).** C1 holds, C2 and C3 fail: the README says the two axes are safe with
the validation check (0/36 pairs over the three sets of data) and usually but not reliably
better than the two-axis search (better on 33 of 36 pairs, one tie, two losses, both on count
data), with the table. It is not written as "always add them".
