# Proposal 034 — `max_bin`: 50 trials, and a choice between 64 and 256

Status: **run** 2026-11-14 (R0 ✓, B1 ✗ / ✓, B2 ✗, B3 ✗, B4 ✓, C1 ✓, C2 ✓, C3 ✓, C4 ✓; `bin50` passes the rule, `bin2` fails; confirmation: 038).
**Pre-registered** 2026-11-14, after two plumbing checks on a dataset outside the test
pairs (below) and before any run of the three arms. Skill: `automl-hyperparameter-research`
(search budget and search-space design; Tier 2). Follows 033. Felix: "lets test this" (the two
items 033 left open).

## Questions
1. **Budget.** 033 failed its inclusion rule at 20 trials by one pair on (b), by 0.002 on (c) and
   on two guard misses on (a). Does `bin` (mcw, λ, max_bin log-uniform integer 16–1024) pass the
   same rule at 50 trials, against the two-axis search at 50 trials?
2. **Two values.** Does a categorical choice max_bin ∈ {64, 256} on top of the two axes (`bin2`)
   keep the gains `bin` had where the bin count matters (kin8nm, Gamma c_parkinsons) without the
   dilution cost it had where it does not (Gaussian n_sgemm +0.048)?

## Data, and what it can and cannot show
The 36 (family, dataset) pairs of 033. They are no longer untouched for this parameter: the
50-trial question and the value 64 both come from 033's results on them (64 is the value that was
better on 12 and worse on 12 pairs untuned, and the kin8nm gains were seen). So:
- a **fail** of either arm on these pairs is taken as final;
- a **pass** is development evidence only, and is confirmed on freshly downloaded datasets under
  the same rule before the README's advice changes (as 032 did for the subsampling axes).

## Arms
| arm | space | trials | refits at | tags |
|---|---|---|---|---|
| `2d50` | mcw, λ | 50 | 20, 50 | `c30_2d_k20`, `c30_2d_k50` |
| `bin50` | mcw, λ, max_bin 16–1024 (log, integer) | 50 | 20, 50 | `c30_bin_k20`, `c30_bin_k50` |
| `bin2` | mcw, λ, max_bin ∈ {64, 256} | 20 | 10, 20 | `c30_bin2_k10`, `c30_bin2_k20` |

TPE, 3 folds, eta 0.05, 2000 rounds, patience 50, seeds 0–4, as 033. One job per (arm, pair,
seed), each with its own CSV (`/tmp/c30/`), binary `/tmp/rb_c30`, `run_c30.sh`. References:
untuned `c12_off` / `c19_def` / `c17_new` / `c27_def`; two axes at 20 trials `c13_raw` / `c19_tpe`
/ `c27_2d`; `bin` at 20 trials `c29_bin`. "Guarded" keeps a refit only if its validation NLL
beats the untuned fit's.

## Plumbing checks (Gamma synth_gamma, not one of the 36)
- `bin2`, 4 trials, seed 0: draws 256, 256, 256, 64; the refit runs.
- The first 4 trials and the k = 4 refit of a 6-trial search equal the 4-trial search bit for bit,
  so the k = 20 rows of a 50-trial search should reproduce the stored 20-trial rows.
- `BENCH_SEED_START=3 BENCH_SEEDS=4` runs seed 3 alone.

## Predictions
- **R0 (reproduction).** `c30_2d_k20` equals the stored two-axis rows and `c30_bin_k20` equals
  `c29_bin_k20`, bit for bit, on all 180 cells each.
- **B1 (two axes are saturated).** For the two-axis search the mean over pairs of the plain test
  gain moves by less than 0.010 between 20 and 50 trials, and the number of pairs worse than the
  untuned default (plain, > 0.005) does not fall below 4.
- **B2 (`bin` at 50 still fails).** Rule (a) fails: guarded `bin50` is worse than the untuned
  default by more than 0.005 on at least one pair. (c) fails: the median of guarded `bin50` −
  guarded `2d50` is above −0.005. (b): better on 20 to 26 of 36 pairs.
- **B3 (dilution shrinks with budget).** On Gaussian n_sgemm plain `bin50` − plain `2d50` is
  within ±0.020 (it was +0.048 at 20 trials).
- **B4 (the real gains stay).** On both kin8nm pairs plain `bin50` − plain `2d50` ≤ −0.040.
- **C1 (`bin2` finds the side).** At 20 trials `bin2` picks 64 in at least 4 of 5 seeds on both
  kin8nm pairs, and 256 in at least 4 of 5 on both naval pairs.
- **C2 (it keeps part of the gain).** Plain `bin2` − plain two axes (20 trials) ≤ −0.025 on both
  kin8nm pairs; on Gamma c_parkinsons ≤ −0.020.
- **C3 (less dilution).** On Gaussian n_sgemm plain `bin2` − plain two axes is within ±0.025.
- **C4 (`bin2` fails the rule on (c)).** (b) guarded `bin2` beats the guarded two-axis search on
  at least 24 of 36 pairs, but (c) the median difference is above −0.005, and (a) fails with at
  least one pair worse than the untuned default.

## Inclusion rule (033's, unchanged), applied to each arm at its largest budget
(a) guarded arm worse than the untuned default by more than 0.005 on 0 pairs; (b) guarded arm
better than the guarded two-axis search of the same budget on at least 24 of 36 pairs; (c) the
median over pairs of that difference ≤ −0.005.
- An arm that fails: the README adds one sentence to the `max_bin` paragraph saying the larger
  budget / the two-value choice was tested too and failed, with the counts. Nothing else changes.
- An arm that passes: no README advice yet; fresh datasets are fetched and the arm is run on them
  under the same rule (a new proposal).
No default changes either way.

## Also reported, not a test
The no-search version of question 2: fit the untuned default at 64 and at 256 bins and keep the
one with the lower validation NLL. All of it is computable from 033's stage 0 rows, which I have
already looked at per pair, so it is a description of seen data and is labelled as such.

## Cost
540 jobs: 360 fifty-trial searches and 180 twenty-trial ones, about 36 stream-hours on a free
machine, in resumable waves of 10 single-thread streams, each wave under the 2-hour task limit.

## Result (2026-11-14, report 016 §10)
540 jobs: 360 fifty-trial searches and 180 twenty-trial ones, 1080 refits, no non-finite row.
Test NLL minus the untuned default, mean over 5 seeds, then over the 36 pairs; "guarded" keeps a
refit only if its validation NLL beats the untuned fit's. Summaries: `c30_summary.py`,
`c30_score.py`.

| arm (trials) | plain: mean | median | worse than default | guarded: mean | median | worse than default |
|---|---|---|---|---|---|---|
| two axes (20, stored) | −0.054 | −0.017 | 6/36 | −0.056 | −0.021 | 1/36 |
| two axes (50) | −0.031 | −0.013 | 7/36 | −0.027 | −0.018 | 3/36 |
| + max_bin 16–1024 (20; = 033) | −0.055 | −0.017 | 5/36 | −0.041 | −0.016 | 2/36 |
| + max_bin 16–1024 (50), `bin50` | −0.075 | −0.025 | 3/36 | −0.062 | −0.031 | **0/36** |
| + max_bin ∈ {64, 256} (20), `bin2` | −0.077 | −0.019 | 6/36 | −0.072 | −0.020 | 3/36 |

The rule, each arm at its largest budget:

| arm | reference | (a) guarded worse than default | (b) better than guarded reference | (c) median difference | |
|---|---|---|---|---|---|
| `bin50` | two axes, 50 trials | 0/36 | 30/36 | −0.009 | **passes** |
| `bin50` (not the registered reference) | two axes, 20 trials | 0/36 | 28/36 | −0.007 | passes |
| `bin2` | two axes, 20 trials | 3/36 (Gaussian c_n_concrete +0.091, c_supercond +0.057, c_gt_co +0.011) | 24/36 | −0.002 | **fails** (a), (c) |

`bin50` by set, (b) and (c): development 10/10, −0.018; unseen 10/14, −0.002; fresh 10/12, −0.016.

- **R0 ✓.** `c30_2d_k20` equals the stored two-axis rows and `c30_bin_k20` equals `c29_bin_k20` on
  180/180 cells each.
- **B1 ✗ / ✓.** The two-axis mean plain gain moves by +0.023 from 20 to 50 trials (predicted
  < 0.010), and in the wrong direction: thirty more trials make the two-axis search worse. Two
  Poisson pairs carry the mean (c_n_metro +0.63, c_seoulbike +0.13), but the guarded two-axis
  search at 50 trials is also no better than at 20 on the pair count (better on 17/36, median
  +0.000) and is worse than the default on 3 pairs against 1. Pairs worse than the default
  (plain): 6 → 7 (not below 4 ✓). In 67 % of the searches the best trial is one of the last 30.
- **B2 ✗.** I predicted `bin50` would fail on (a) and (c) with (b) between 20 and 26. It passes
  all three: 0/36, 30/36, −0.009. The two guard misses of 033 at 20 trials (Gaussian protein
  +0.123, Poisson c_seoulbike +0.078) are gone at 50 (−0.051, −0.191).
- **B3 ✗ (in the arm's favour).** Gaussian n_sgemm, where max_bin cannot matter: plain `bin50` −
  `2d50` = −0.029 ± 0.008 (5/5 seeds), outside the predicted ±0.020; the +0.048 dilution cost
  of 033 is gone. The difference is the search path, not the bins.
- **B4 ✓.** Gaussian kin8nm −0.071 ± 0.007, Gamma kin8nm −0.064 ± 0.010 (5/5 seeds each).
- **C1 ✓.** `bin2` picks 64 in 5/5 seeds on both kin8nm pairs and 256 in 5/5 on both naval pairs
  (64 in 97 of 180 searches overall).
- **C2 ✓.** Plain `bin2` − two axes: Gaussian kin8nm −0.062, Gamma kin8nm −0.049, Gamma
  c_parkinsons −0.048.
- **C3 ✓.** Gaussian n_sgemm −0.002.
- **C4 ✓.** `bin2`: (b) 24/36, (c) −0.002, (a) 3 pairs. The two-value choice keeps the kin8nm
  gains and loses the dilution, but its guard misses are larger than the continuous arm's.
- **Picks at 50 trials.** kin8nm 16–30, Gamma c_parkinsons 22–26, naval 239–903, the simulation
  403–994; max / min over seeds ≥ 8 on 12/36 pairs. The refit's validation NLL is lower than the
  two-axis refit's on 27/36 pairs.

**Also reported, not a test (seen data, from 033's stage 0 rows).** Fitting the untuned default at
64 and at 256 bins and keeping the lower validation NLL picks 64 in 100/180 cells and is better
than always-256 by more than 0.005 on 13/36 pairs, worse on 7 (largest Poisson c_seoulbike
+0.134); median −0.001.

**What I had wrong.** I treated the 20-trial failure as a property of the parameter. It was a
property of the budget: three axes need more than 20 trials, and at 50 the guard has no misses.
I also expected the two-axis search to be saturated at 20 trials; it is not saturated, it is
past its best (more trials select harder on a noisy fold score).

**Decision (by the rule).** `bin2` fails: final, one README sentence. `bin50` passes on pairs
that had already been used for this parameter, so it is development evidence only: no README
advice yet; proposal 038 runs `2d50` and `bin50` on freshly downloaded datasets under the same
rule. No default changes.
