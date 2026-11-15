# Proposal 036 — LightGBM's own parameters as search axes

Status: **run** 2026-11-15 (G0 ✗ / ✓, G1 ✓, G2 ✗ / ✓, G3 ✗; all four arms fail; a side finding on
`lambda` under LightGBM). **Pre-registered** 2026-11-14, after plumbing checks on a dataset outside the test pairs
(below) and before any LightGBM search on any of the 36 pairs. Runs after 035. Skill:
`automl-hyperparameter-research` (search-space design; Tier 2). Felix's queue, item 6.

## Question
All search-space work so far was on XGBoost. On LightGBM, are num_leaves, min_data_in_leaf,
feature_fraction or bagging_fraction worth a search axis next to the two axes
(min_child_weight = min_sum_hessian_in_leaf, lambda = lambda_l2)?

## Data
The 36 (family, dataset) pairs of 033. LightGBM has been fitted untuned on the development
datasets (019: 21 pairs, old against new defaults); no LightGBM search has been run on any pair.
Four arms are screened under one rule: a **pass** is confirmed on fresh datasets before any
README advice; a **fail** is final at this budget.

## Arms (LightGBM backend; eta 0.05, max_depth 4, 2000 rounds, patience 50, seeds 0–4; 20 trials, TPE, 3 folds; refits at 10 and 20)
| arm | space | note |
|---|---|---|
| `l_def` | none | the untuned default on LightGBM: the reference for "worse than untuned" |
| `l_2d` | mcw, λ log-uniform [10⁻³, 10³] | the reference search |
| `lleaves` | + num_leaves log-uniform integer [4, 64], max_depth = −1 | leaf-wise size instead of the depth limit |
| `lmdl` | + min_data_in_leaf log-uniform integer [1, 100] | LightGBM's default is 20 |
| `lff` | + feature_fraction uniform [0.5, 1] | per tree |
| `lbag` | + bagging_fraction uniform [0.5, 1] | bagging_freq = 1 (the backend's default) |

Binary `/tmp/rb_c32_lgbm`, `run_c32.sh`, one job per (arm, pair, seed) in `/tmp/c32/`, tags
`c32_<arm>[_k<k>]`. "Guarded" keeps a refit only if its validation NLL beats `l_def`'s.

## Plumbing (Gamma synth_gamma, seed 0, not one of the 36)
Untuned NLL 2.44319; `min_data_in_leaf=20` and `num_leaves=16` (at depth 4) each equal it bit for
bit; `min_data_in_leaf=1` 2.44508; `num_leaves=4, max_depth=-1` 2.41148. Each arm draws its
parameter (3 trials) and its refit runs. Seen when the predictions were written.

## Inclusion rule (033's), per arm at 20 trials over the 36 pairs
(a) guarded arm worse than `l_def` by more than 0.005 on 0 pairs; (b) guarded arm better than
guarded `l_2d` on at least 24 pairs; (c) the median over pairs of that difference ≤ −0.005.

## Predictions
- **G0 (the two axes behave as on XGBoost).** Plain `l_2d`: mean gain over `l_def` between −0.10
  and 0; worse than `l_def` by more than 0.005 on 3 to 9 of 36 pairs (XGBoost: −0.054, 6).
- **G1 (`lleaves` is the depth axis again).** Plain `lleaves` beats plain `l_2d` by more than
  0.005 on at least 15/36 pairs; guarded `lleaves` is worse than `l_def` on at least one pair, so
  (a) fails.
- **G2 (`lmdl` matters where the data are small).** Plain `lmdl` beats plain `l_2d` by more than
  0.005 on at least 12/36 pairs; the rule fails on (c) (median above −0.005).
- **G3 (the two sampling axes are the safe ones).** Guarded `lff` and guarded `lbag` are each
  worse than `l_def` on 0 pairs, each reaches (b) ≥ 24, and at least one of the two passes the
  whole rule.

## Also reported, not a test
`l_def` against the XGBoost untuned default on the 36 pairs.

## Decision
Pass → fresh-data confirmation (new proposal) before the README names the axis for LightGBM.
Fail → the README lists it as tested on LightGBM with the counts. No default changes.

## Cost
900 searches and 180 untuned fits, about 35 stream-hours on a free machine, resumable waves.

## Result (2026-11-15, report 016 §14)
1080 jobs (900 twenty-trial searches with refits at 10 and 20, 180 untuned fits), no failed job,
no non-finite test NLL. Scored with `c32_score.py`.

### The references
- **`l_def` against the XGBoost untuned default** (same eta, depth, rounds; LightGBM's
  lambda_l2 is 0, XGBoost's lambda 1): LightGBM better by more than 0.005 on 21/36 pairs, worse
  on 8, within on 7; mean −0.072, median −0.019. Largest: Poisson c_n_metro −1.75, c_seoulbike
  −0.60, Gamma c_parkinsons −0.13 for LightGBM; Gaussian protein +0.21, Gamma c_n_concrete
  +0.20, Gaussian c_gt_co +0.07 against it.
- **`l_2d` against `l_def`:** plain mean **+0.178**, median −0.004, worse by more than 0.005 on
  5/36, better on 17. Guarded: mean −0.021, median −0.006, worse on 3/36 (Gamma c_airfoil
  +0.010, Gaussian c_n_appliances +0.006, NegativeBinomial c_n_metro +0.006). The XGBoost two-axis
  search: plain −0.054, guarded −0.056, 1/36.
- **The plain mean is one pair.** On Gaussian naval every tuned refit of every arm, in all five
  seeds, at 10 and at 20 trials (50 cells), stops after 77–99 rounds at a test NLL of +0.46 to
  +1.42; `l_def` runs 1391–1999 rounds to −6.1. Without that pair `l_2d`'s plain mean is −0.015. Probe (untuned, seed 0): lambda = 0
  → −6.11 (1900 rounds); lambda_l2 = 0.001 → +0.95 (92 rounds); lambda = 1 → +1.46 (0 rounds,
  validation NLL infinite); mcw 0.001 or 16 at lambda 0 → −6.11, −6.08. Any L2 penalty in the
  search range [10⁻³, 10³] ends the fit early, so every trial of every search on this pair is
  broken and cross-validation cannot see it (the best CV score in seed 0 of `l_2d` is +1.11). The guard discards the
  refit in every cell. XGBoost at its default lambda = 1 fits this pair (−5.97 to −6.13 over the seeds). The
  mechanism inside LightGBM was not traced.

### The four arms at 20 trials against `l_2d`
| arm | plain mean | guarded mean | guarded median | (a) guarded worse than `l_def` | (b) better than guarded `l_2d` | (c) median difference | plain better / worse than `l_2d` by > 0.005 | rule |
|---|---|---|---|---|---|---|---|---|
| `l_2d` | +0.178 | −0.021 | −0.006 | 3 | | | | |
| `lleaves` | +0.145 | −0.135 | −0.031 | 1 | 25 | −0.0238 | 28 / 3 | fail |
| `lmdl` | +0.179 | −0.020 | −0.009 | 2 | 19 | −0.0002 | 11 / 13 | fail |
| `lff` | +0.174 | −0.022 | −0.013 | 2 | 22 | −0.0027 | 17 / 7 | fail |
| `lbag` | +0.184 | −0.036 | −0.013 | 4 | 24 | −0.0065 | 21 / 9 | fail |

Guard misses, (a): `lleaves` Gaussian power_plant +0.0058; `lmdl` Poisson c_n_metro +0.079,
c_seoulbike +0.045; `lff` Poisson c_n_metro +0.128, NegativeBinomial c_n_metro +0.008; `lbag`
Poisson c_n_metro +0.041, Gaussian c_n_concrete +0.010, power_plant +0.007, c_n_appliances
+0.006. At 10 trials: `lleaves` 3 / 28 / −0.0135, `lmdl` 3 / 20 / −0.0005, `lff` 1 / 16 /
+0.0005, `lbag` 4 / 20 / −0.0003; none passes.

`lleaves` by set, (b) and (c): development 6/10, −0.014; unseen 11/14, −0.028; fresh 8/12,
−0.022. Picked values, quartiles over the 180 searches: num_leaves 9 / 13 / 31;
min_data_in_leaf 8 / 23 / 51; feature_fraction 0.63 / 0.80 / 0.90; bagging_fraction 0.69 / 0.80
/ 0.90.

### Predictions
- **G0 ✗ / ✓.** Plain `l_2d` mean +0.178, not between −0.10 and 0 (the naval pair; −0.015
  without it); worse than `l_def` on 5 pairs (3 to 9 ✓).
- **G1 ✓.** Plain `lleaves` beats plain `l_2d` on 28/36; guarded it is worse than `l_def` on one
  pair, so (a) fails — by 0.0008 over the 0.005 line. I wrote this prediction before 035 showed
  the XGBoost leaf axis losing nowhere; on LightGBM the leaf axis does lose once, barely.
- **G2 ✗ / ✓.** Plain `lmdl` beats plain `l_2d` on 11/36, one short of the 12 predicted, and is
  worse on 13; the rule fails on (c) as predicted (and on (a) and (b)).
- **G3 ✗.** I called the two sampling axes the safe ones. Guarded `lff` is worse than `l_def` on
  2 pairs and `lbag` on 4; `lff` reaches (b) 22, `lbag` 24; neither passes.

### What I had wrong
- I assumed the two-axis search carries over to LightGBM. It does not carry over cleanly: its
  lambda range excludes LightGBM's default (0), one pair breaks completely, and even guarded it
  loses to the untuned default on 3 pairs and gains less than half of what it gains on XGBoost
  (−0.021 against −0.056).
- Subsampling was safe with the guard on XGBoost (0/36); on LightGBM both sampling axes have
  guard misses.

### Decision
- **All four fail, final at 20 trials.** The README lists them as tested on LightGBM with the
  counts. `lleaves` is the near miss (one pair, +0.0058) and has the same shape as the XGBoost
  leaf axis that passed: for Felix, whether it is worth a rerun at a larger budget or with the
  lambda range fixed first.
- **The lambda range on LightGBM is the larger finding.** The README says so. For Felix: should
  the documented two-axis space for LightGBM let lambda be 0 (a choice between 0 and a
  log-uniform range), and is the early stop on Gaussian naval worth a diagnosis of its own?
- No default changes.
