# Proposal 022 — a validation guard for tuned models, confirmed on unseen datasets

Status: **run** 2026-11-13 (G1–G4 ✓, M1 ✗). **Pre-registered** 2026-11-13, after proposal 017's results and before any run on the unseen
sets. Skill: `automl-hyperparameter-research` (Tier 2: does the tuned model generalise).

## Observed limitation (proposal 017, 10 development sets × 5 seeds × 4 samplers × k ∈ {5, 10, 20})
- The search believes it gained −0.034 to −0.047 over the untuned default (CV score of the pick −
  CV score of the default on the same folds). The refit keeps −0.005 to −0.007 on validation and test.
- Within a set, the CV gain of a pick does not predict its test gain: correlation ≤ 0.15 on 6 of 10
  sets (−0.18 … +0.36 overall). The refit's own validation gain does: 0.86–0.98 on the five sets
  where tuning is unstable (both naval sets, Poisson and NB bike_cnt, Gamma kin8nm).
- Every sampler is worse than the untuned default by more than 0.005 on 2 of 10 sets at 20 trials.
- By the diagnosis rule of 017 both losing sets are **selection failures**: the default's CV score is
  worse than the pick's in 5/5 seeds, while validation and test prefer the default.

## Candidate (post hoc on the development sets, hence this confirmation)
After the search, fit the pick and the untuned default the way the final model is fitted (training
split, early stopping on the validation split) and keep the one with the lower validation NLL.

    guard(pick) = pick     if  NLL_valid(refit of pick) < NLL_valid(refit of default)
                  default  otherwise

Cost: one extra fit. No test data is used. The validation split is also the early-stopping monitor
for both fits, so the comparison is between two equally optimistic numbers.

On the development sets (same refits, re-read, `c14_selection_analysis.py`): sets worse than the
default by > 0.005 go from 2/10 to 0/10 for `tpe`, `rand`, `dtpe` and to 1/10 for `ds10` (protein
+0.024); the mean gain over sets goes from −0.008 to −0.019 for `tpe`.

## Protocol
The (family, dataset) pairs of proposal 020 that have a current-default row (`c17_new`), StudentT left out:
Gaussian and Gamma (mean, shape) on c_airfoil, c_parkinsons, c_gt_nox, c_gt_co, c_supercond; Poisson
and NB (mean, dispersion) on c_abalone, c_seoulbike: 14 pairs.
5 seeds, splits as in 020. Search exactly as in 017: 3-fold CV on the training split, 20 trials, mcw
and λ log-uniform on [10⁻³, 10³], refit at k ∈ {5, 10, 20}. Binary `/tmp/rb_c15c`, one thread.

| arm | sampler | trial 1 | purpose |
|---|---|---|---|
| `tpe` | TPE (library default) | drawn | the guard's test |
| `dtpe` | TPE | the untuned default | the default's CV score on the search's folds (G4) |

The untuned default is re-fitted by the same binary (`c19_def`) so the comparison is within one build.
Tags `c19_<arm>_k<k>`, `c19_def`; script `run_c19.sh`; analysis `c19_guard_summary.py`.

Mechanism arm on two development sets (Gaussian naval, Poisson bike_cnt): `tpe` with **5 folds**
(tag `c19_tpe5`). A fold model sees 2/3 of the training rows, the refit all of them, and mcw and λ
are in absolute Hessian units; with 5 folds the fold model sees 4/5.

## Predictions
- **G1 (the problem is real off the development sets).** Plain `tpe` at 20 trials is worse than the
  untuned default by more than 0.005 test NLL on at least one of the 14 pairs.
- **G2 (the guard is safe).** Guarded `tpe` at 20 trials is ≤ default + 0.005 on 14/14 pairs.
- **G3 (the guard is free).** Mean over pairs: guarded ≤ plain.
- **G4 (the CV gain does not survive).** `dtpe`, pooled over pairs and k: mean test gain is less
  than half of the mean CV gain.
- **M1 (fold size).** With 5 folds the loss of tuning halves: tuned − default at 20 trials is
  ≤ +0.026 on Gaussian naval (3 folds: +0.053) and ≤ +0.015 on Poisson bike_cnt (+0.030).

## Decision rule
- No default change. `hyper_opt` returns parameters, not a model, so the guard is a usage rule.
- G2 and G3 hold: document the guard in the README's tuning section (compare the tuned refit with
  the untuned fit on the validation split, keep the better).
- G2 fails: report where validation mis-selects; no recommendation.
- M1 holds: note that `n_folds` matters for the absolute-unit axes; M1 fails: the mechanism stays open.

## Result (2026-11-13, report 015 §6)
150 searches, 450 refits, no non-finite row. `c19_def` reproduces `c17_new` bit for bit. The
background task hit its 2 h limit with Gamma c_supercond seeds 2–4 missing in both arms; that one
set was re-run (`run_c19.sh <arm> 5`); seeds 0–1 of the re-run repeat the first run line for line.

Two axes, TPE, 20 trials; test NLL − untuned default, mean over 5 seeds:

| family | dataset | plain | guarded | picks kept |
|---|---|---|---|---|
| Gamma | c_airfoil | −0.041 | −0.048 | 2/5 |
| Gamma | c_gt_co | −0.014 | −0.009 | 3/5 |
| Gamma | c_gt_nox | −0.002 | −0.001 | 3/5 |
| Gamma | c_parkinsons | −0.044 | −0.047 | 4/5 |
| Gamma | c_supercond | −0.028 | −0.025 | 4/5 |
| Gaussian | c_airfoil | −0.026 | −0.043 | 4/5 |
| Gaussian | c_gt_co | **+0.020** | −0.022 | 2/5 |
| Gaussian | c_gt_nox | −0.006 | −0.007 | 3/5 |
| Gaussian | c_parkinsons | **+0.052** | 0.000 | 0/5 |
| Gaussian | c_supercond | +0.003 | −0.036 | 3/5 |
| NB | c_abalone | −0.001 | −0.001 | 2/5 |
| NB | c_seoulbike | −0.012 | −0.013 | 3/5 |
| Poisson | c_abalone | −0.003 | −0.001 | 4/5 |
| Poisson | c_seoulbike | −0.140 | −0.071 | 2/5 |
| mean (median) | | −0.017 (−0.009) | −0.023 (−0.018) | |

- **G1 ✓.** Plain tuning is worse than the default by > 0.005 on 2/14 unseen pairs.
- **G2 ✓.** Guarded: 0/14 (also 0/14 for `dtpe`, 0/2 for the 5-fold arm).
- **G3 ✓** for `tpe` (−0.023 against −0.017). Not for `dtpe`: −0.024 against −0.033, all of it
  Poisson c_seoulbike (plain −0.316, guarded −0.141), where the Poisson fit is badly calibrated
  (90 % coverage 0.41) and its validation NLL is noisy. The guard is safe, not free.
- **G4 ✓.** `dtpe`, pooled: CV gain −0.082, validation −0.015, test −0.022: 27 % survives
  (medians −0.021 and −0.006). Correlation of CV gain with test gain: −0.52 to +0.70, ≤ 0.2 on 8/14.
- **M1 ✗.** 5 folds: Gaussian naval +0.012 (3 folds +0.053; within the bound), Poisson bike_cnt
  +0.053 (3 folds +0.030; worse). Fold size is not the mechanism; it stays open.

**Qualification from proposal 025.** With the wide space the guard does not hold on unseen data:
guarded wide is worse than the default on 2/14 pairs (Gaussian c_gt_co +0.166, c_supercond +0.015).
G2 is a result for the two-axis space.

**Decision (by the rule).** G2 and G3 hold: the README documents the guard as a usage rule for the
two-axis search, with the qualification. Whether to build it into `hyper_opt` is Felix's call.

**Applied 2026-11-13 (Felix's decision; proposal 026):** `hyper_opt::validation_guard` (opt-in).
