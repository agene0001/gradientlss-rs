# Proposal 017 — is the search itself the bottleneck? Random control, start-up phase, defaults first

Status: **run** 2026-11-13 (P1 ✗, P2 ✗, P3 ✗, P4 ✓; both losing sets are selection failures). **Pre-registered** 2026-11-13, before any controlled run. Skill: `automl-hyperparameter-research`
(first cycle run under it; kept in `research/gradientlss/` with the existing numbering).

## Question
Cycle 13 judged a tuning *axis* with `hyper_opt` but never judged `hyper_opt`. Two things were left open:
1. **Tier 1 (tuner).** Does the sampler beat a random search on this problem at 5, 10 and 20 trials?
2. **Tier 2 (generalisation).** On raw Poisson bike_cnt and raw Gaussian naval the tuned model is
   *worse* than the untuned default. Is that the search (it never found anything as good as the
   default) or the selection (CV prefers a configuration that held-out data does not)?

## What `hyper_opt` does today (read from the code, 2026-11-13)
- One independent `TpeOptimizer` per parameter (`src/hyper_opt.rs`, `search`), `tpe` 0.3.1 defaults:
  γ = 0.1, 24 candidates, Parzen prior at the middle of the range. No random start-up phase and no
  joint model, so an mcw × λ interaction cannot be represented.
- A trial's score is the mean over folds of the fold's validation loss at its early-stopped round;
  the scoring fold is also the early-stopping monitor.
- The winner is the plain argmin of that mean. The untuned defaults are not evaluated unless the
  sampler happens to draw them.

## Evidence already in hand (zero compute; `c14_trace_analysis.py` on the cycle 13 logs)
100 searches (10 sets × 5 seeds × 2 axes), 20 trials, two log axes over six decades.
- **Trial 2 repeats trial 1.** kin8nm seed 0: (mcw 16.4, λ 0.131) then (21.5, 0.204); seed 1:
  (0.0221, 0.0635) then (0.0152, 0.0555). With one observation the "good" set is that observation.
- **Trials 3–4 go to the range ends.** Share of draws within half a decade of an end on either axis
  (a log-uniform sampler: 0.31): trial 1 0.20, trial 2 0.20, **trial 3 0.80, trial 4 0.95**, trial 5
  0.55, trials 6–10 0.36, trials 11–20 0.22.
- **After that it is close to uniform.** Draws 6–20 have sd 1.55 decades per axis (uniform 1.73) and
  visit 10.4 of 16 grid cells (uniform ≈ 11.6). The final best was found by trial 5 in 18/100
  searches and by trial 10 in 32/100.
- **Validation agrees with test.** Tuned (k = 20) − untuned, paired by seed, on the bench's
  early-stopping split and on test: same sign and similar size on 18 of 20 (axis, set) rows. The two
  raw-axis losses appear on both splits: Poisson bike_cnt +0.024 ± 0.026 / +0.030 ± 0.018, Gaussian
  naval +0.064 ± 0.033 / +0.053 ± 0.029. So they are not test noise alone.

Reading before the run: the first five trials are one prior draw, one near-duplicate and two or three
corners. That is the small-budget failure of cycle 13 (naval at k = 5 on the normalised axis was
forced to visit λ ≈ 10³ rows' worth), and it is a property of the sampler, not of the axis.

## Library change (opt-in; defaults unchanged)
`HyperOptConfig::sampler: Sampler::{Tpe (default), Random, TpeAfterRandom { n_startup }}` and
`HyperOptConfig::initial_trials` (configurations evaluated first, on this run's folds; Optuna's
`enqueue_trial`). Tests: `initial_trials_are_evaluated_first`, `random_and_startup_samplers`.
Check that the default path did not move: the cycle 13 cell (synth_pois, Poisson, raw, seeds 0–1) was
replayed with the new binary and its 40 trial lines are identical to the cycle 13 log.

## Protocol
Exactly cycle 13's raw arm (Off): 10 (family, dataset) pairs × 5 seeds, 3-fold KFold on the training
split, fold seed = `hp_seed` = bench seed, 20 trials, mcw and λ log-uniform on [10⁻³, 10³], eta 0.05,
depth 4, 2000 rounds / patience 50, PES Auto, Damped, `OMP_NUM_THREADS=3`. Refit the best-so-far
after k ∈ {5, 10, 20} trials with validation early stopping; score on test.

| arm | sampler | trial 1 |
|---|---|---|
| `tpe` | TPE | drawn (this is `c13_raw`, reused, not re-run) |
| `rand` | Random | drawn |
| `dtpe` | TPE | the untuned default (mcw = 1, λ = 1) |
| `ds10` | TpeAfterRandom{10} | the untuned default |

`rand` vs `tpe` is the Tier 1 control. `dtpe` vs `tpe` isolates "defaults first". `ds10` vs `dtpe`
isolates the start-up phase. `dtpe`'s first trial also gives the default's CV score on the search's
own folds, which the diagnosis below needs.

Tags `c14_<arm>_k<k>`; script `run_c14.sh`; logs `/tmp/c14_<arm>_<group>.log`.

## Predictions
- **P1 (start-up defect).** At k = 5 the best CV score of `rand` is lower than `tpe`'s on at least
  7/10 sets (mean over seeds).
- **P2 (no model advantage at this budget).** At k = 20, |`rand` − `tpe`| in best CV score is ≤ 0.005
  on at least 8/10 sets.
- **P3 (defaults first).** At k = 20, test NLL of `dtpe` − `tpe` is ≤ +0.005 on every set, and
  ≤ −0.01 on raw Poisson bike_cnt and raw Gaussian naval.
- **P4 (start-up by 20).** At k = 20, |`ds10` − `dtpe`| in best CV score is ≤ 0.005 on at least 8/10.

## Diagnosis rule for the two losing sets (fixed in advance)
With c = the default's CV score and p = the 20-trial pick's CV score in `tpe`, per seed:
- **search failure** if c < p in ≥ 4/5 seeds (the search never matched the default);
- **selection failure** if c > p in ≥ 4/5 seeds while validation and test prefer the default;
- otherwise **undetermined at 5 seeds**.

## Decision rule
- No default change in this cycle.
- If P3 holds: recommend `initial_trials` with the untuned defaults in the README's tuning section.
- If P1 holds: propose a start-up phase to Felix as a candidate default, with the k = 5/10/20 table.
- If P2 fails in TPE's favour by more than 0.005 on ≥ 3 sets, TPE earns its keep at 20 trials and the
  start-up question is limited to k ≤ 10.

## Cost
Three new arms × ≈ 3.2 stream-hours each (cycle 13's `HPO_SECS`), four streams: ≈ 2.4 h wall.

## Result (2026-11-13, report 015 §1)
- **P1 ✗.** Random beats TPE on best CV score at k = 5 on 4/10 sets (needed 7).
- **P2 ✗.** At k = 20, |random − TPE| ≤ 0.005 on 7/10 (needed 8). The three misses favour TPE
  (Gaussian naval +0.023, Gamma naval +0.012, power_plant +0.006), and random is better on 0/10 sets
  at k = 10 and at k = 20. By the decision rule TPE earns its keep at 20 trials.
- **P3 ✗.** Test NLL, dtpe − tpe at k = 20: +0.014 Gamma kin8nm, +0.021 Gamma naval, +0.032 protein,
  +0.029 Poisson bike_cnt; −0.049 Gaussian naval, −0.025 Gaussian kin8nm. Scoring the default first
  does not make the result safer.
- **P4 ✓.** |ds10 − dtpe| ≤ 0.005 in best CV score on 8/10 sets.
- **Diagnosis.** Poisson bike_cnt and Gaussian naval: the default's CV score is worse than the
  20-trial pick's in 5/5 seeds (by 0.052 and 0.095), while validation (+0.024, +0.064) and test
  (+0.030, +0.053) prefer the default. **Selection failure** on both, not a search failure.
- **Beyond the predictions** (`c14_selection_analysis.py`): over 600 refits the CV gain over the
  default is −0.034 to −0.047 and the refit keeps −0.005 to −0.007; the CV gain does not predict the
  test gain within a set, the refit's validation gain does. Followed up in proposal 022.
- **Decision.** No start-up phase proposed; `initial_trials` not recommended. Both options stay
  (opt-in, tested).
