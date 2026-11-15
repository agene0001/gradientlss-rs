# Proposal 039 — the leaf-wise axis (`lossguide`, `max_leaves`) on datasets it has not seen

Status: **run** 2026-11-15 (F0–F6 all ✓; the rule passes: 0/12, 12/12, −0.068).
**Pre-registered** 2026-11-15, before any leaf-wise fit on the datasets below. Skill:
`automl-hyperparameter-research` (search-space design; Tier 2, confirmation). Follows 035, whose
rule says a pass on the 36 pairs is development evidence only.

## Question
In 035 the `leaf` arm (mcw, λ, and `max_leaves` log-uniform integer 4–64 with
`grow_policy = lossguide`, `max_depth = 0`; 20 trials) passed the inclusion rule on the 36 pairs:
guarded arm worse than the untuned default on 0/36, better than the guarded two-axis search on
34/36, median difference −0.038. The plain refit was worse than the default on 0/36 as well. The
untuned variant with 16 leaves (`u_lg16`) was better than the depth-4 default by more than 0.005
on 27/36 pairs and worse on 5. Eight arms were screened under one rule in 035, so this pass needs
a batch that had no part in it. Does it hold there?

## Data
The 12 `c_m_*` pairs of 038 (seven UCI datasets fetched 2026-11-14: m_energy, m_steel, m_fishtox,
m_tempbias, m_video, m_pm25, m_fbcomments; Gaussian 5, Gamma 4, Poisson 1, NegativeBinomial 2).
They are not freshly downloaded for this proposal: 038 used them for `max_bin`, so their untuned
and two-axis results have been seen (including that Gamma m_energy, 768 rows, is where a guard
missed for `max_bin`). No fit with `lossguide`, `max_leaves` or any other 035 parameter has been
run on them, and nothing about the `leaf` arm (range, budget, rule) was chosen with them. The
references are 038's stored rows, so this batch costs only the new arms.

## Arms
| arm | setting | tags |
|---|---|---|
| `u_repro` | untuned default spelled out (`max_delta_step=0,gamma=0`) | `c35_u_repro` |
| `u_lg16` | untuned, `grow_policy=lossguide`, `max_leaves=16`, `max_depth=0` | `c35_u_lg16` |
| `leaf` | mcw, λ, max_leaves 4–64 (log, integer), lossguide, max_depth 0; 20 trials, refits at 10 and 20 | `c35_leaf_k10`, `c35_leaf_k20` |

References: untuned `c34_def`; two axes `c34_2d_k20` (the 20-trial refit of 038's 50-trial
search, which equals a 20-trial search bit for bit by the prefix property checked in 034).
Everything else as 035: TPE, 3 folds, eta 0.05, 2000 rounds, patience 50, seeds 0–4, binary
`/tmp/rb_c31`, `run_c35.sh`, one job per (arm, pair, seed) in `/tmp/c35/`.

## Rule (035's, scaled from 36 to 12 pairs), at 20 trials
(a) guarded `leaf` worse than the untuned default by more than 0.005 on 0/12 pairs; (b) guarded
`leaf` better than the guarded two-axis search on at least 8/12; (c) the median over pairs of
that difference ≤ −0.005.

## Predictions
- **F0.** `u_repro` equals `c34_def` bit for bit on 60/60 cells (the binary of 035 against the
  binary of 034/038).
- **F1.** (a) holds, 0/12, and the plain refit is worse than the default on at most 1/12.
- **F2.** (b) at least 10/12.
- **F3.** (c) between −0.080 and −0.015. With F1 and F2: the rule passes. (038's lesson: a single
  guard miss on a small dataset is the likeliest way to be wrong; m_energy and m_fishtox have
  under 1000 rows.)
- **F4 (`u_lg16`).** Better than the default by more than 0.005 on at least 7/12 pairs and worse
  on at least 1/12.
- **F5 (the search is needed).** Plain `leaf` is better than `u_lg16` by more than 0.005 on at
  least 7/12 pairs.
- **F6 (picks).** The median picked max_leaves over seeds is at most 8 on at least 1/12 pairs and
  at least 32 on at least 2/12.

## Decision
No default changes from this proposal.
- **Pass:** the README's tuning section recommends the leaf-wise axis next to mcw and λ (with the
  validation guard), with both tables. Whether `hyper_opt` should offer it as a ready-made space,
  and whether a leaf-wise default deserves its own trial (`u_lg16` is worse on 5/36 pairs, so it
  does not meet the default rule as it stands), go to Felix as questions.
- **Fail:** the README records that the arm passed on the 36 pairs and failed here, with the
  counts; no recommendation.

## Cost
60 twenty-trial searches and 120 untuned fits: about 3 stream-hours. Runs when 035's last arms
finish, before 036.

## Result (2026-11-15, report 016 §13)
180 jobs (60 twenty-trial searches with refits at 10 and 20, 120 untuned fits), no non-finite
row, no failed job. Scored with `c35_score.py`.

| arm (20 trials) | plain: mean (median) | worse than default | guarded: mean (median) | guarded: worse than default |
|---|---|---|---|---|
| two axes (stored, `c34_2d_k20`) | −0.033 (−0.028) | 0/12 | −0.045 (−0.028) | 1/12 |
| `leaf` | −0.225 (−0.078) | 0/12 | −0.232 (−0.130) | 0/12 |
| `u_lg16` (untuned) | −0.157 (−0.084) | 2/12 | | |

**Rule at 20 trials: (a) 0/12, (b) 12/12, (c) −0.068 → pass.** At 10 trials against the same
reference: 0/12, 11/12, −0.059.

| pair | `leaf` plain | `leaf` guarded | `u_lg16` | two axes guarded | plain `leaf` − two axes (± s.e., seeds better) | median max_leaves |
|---|---|---|---|---|---|---|
| Gamma m_energy | −0.037 | −0.041 | −0.083 | −0.025 | −0.006 ± 0.065 (2/5) | 7 |
| Gamma m_steel | −0.202 | −0.181 | −0.142 | −0.129 | −0.073 ± 0.015 (5/5) | 21 |
| Gamma m_tempbias | −0.078 | −0.078 | −0.072 | −0.031 | −0.047 ± 0.005 (5/5) | 15 |
| Gamma m_video | −0.225 | −0.225 | −0.171 | −0.034 | −0.200 ± 0.021 (5/5) | 32 |
| Gaussian m_energy | −0.077 | −0.181 | −0.086 | −0.084 | −0.025 ± 0.101 (2/5) | 8 |
| Gaussian m_fishtox | −0.018 | −0.018 | +0.080 | +0.006 | −0.018 ± 0.016 (3/5) | 4 |
| Gaussian m_steel | −0.312 | −0.312 | −0.252 | −0.076 | −0.280 ± 0.052 (5/5) | 24 |
| Gaussian m_tempbias | −0.079 | −0.079 | −0.075 | −0.007 | −0.076 ± 0.010 (5/5) | 19 |
| Gaussian m_video | −0.263 | −0.263 | −0.173 | −0.020 | −0.243 ± 0.038 (5/5) | 30 |
| NegativeBinomial m_fbcomments | −0.023 | −0.023 | +0.007 | −0.015 | +0.000 ± 0.005 (3/5) | 4 |
| NegativeBinomial m_pm25 | −0.071 | −0.071 | −0.043 | −0.007 | −0.064 ± 0.006 (5/5) | 31 |
| Poisson m_pm25 | −1.312 | −1.312 | −0.875 | −0.119 | −1.276 ± 0.129 (5/5) | 43 |

The three pairs where the paired difference is inside its standard error are the two on the
768-row m_energy and NegativeBinomial m_fbcomments; (b) counts them because the guarded means
favour `leaf` (by 0.016, 0.097 and 0.009). Without them (b) would be 9/9. The refit's validation
NLL was lower than the two-axis refit's on 12/12 pairs.

### Predictions
- **F0 ✓.** 60/60 identical.
- **F1 ✓.** (a) 0/12; plain worse than the default on 0/12.
- **F2 ✓.** (b) 12/12.
- **F3 ✓.** (c) −0.068, inside −0.080 … −0.015; the rule passes.
- **F4 ✓.** `u_lg16` better on 10/12, worse on 2 (Gaussian m_fishtox +0.080, NegativeBinomial
  m_fbcomments +0.007).
- **F5 ✓.** Plain `leaf` better than `u_lg16` by more than 0.005 on 9/12, worse on 2 (both
  m_energy pairs, where the search picked 7–8 leaves and the fixed 16 did better: Gamma −0.037
  against −0.083).
- **F6 ✓.** Median max_leaves at most 8 on 4/12, at least 32 on 2/12.

All seven held. They were written after seeing the 36 pairs of 035, so this is a confirmation of
that result on other data, not a blind forecast of the method.

### Over 035 and 039 together (48 pairs, 20 trials)
`leaf`: worse than the untuned default on 0/48 plain and 0/48 guarded; guarded better than the
guarded two-axis search on 46/48. Two axes: guarded worse than the default on 2/48. `u_lg16`:
better than the default on 37/48, worse on 7.

### Decision
- **Pass → the README recommends the leaf-wise axis** next to mcw and λ with the validation
  guard (XGBoost backend), with both tables.
- Limits stated there: 20 trials; not tested together with the subsampling axes; the 12 pairs
  were fetched for 038, not for this.
- **For Felix:** (1) should `hyper_opt` offer this space ready-made (today it is three entries in
  the user's dictionary)? (2) is a leaf-wise default worth its own trial? `u_lg16` loses on 7/48
  pairs, so it fails the default rule as it stands; a smaller fixed leaf count has not been
  tried.

### Addendum, 2026-11-15
Felix asked for my recommendation on both questions. (1) Yes: `hyper_opt::leaf_wise_search_space()`
added and exported in the prelude. (2) Not now: no trial of a leaf-wise default was run (report
016 §16, "Decisions").
