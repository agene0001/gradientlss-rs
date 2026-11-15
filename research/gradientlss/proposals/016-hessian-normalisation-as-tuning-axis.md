# Proposal 016 — Hessian normalisation as a tuning axis (equal-budget hyper_opt)

Status: **resolved** — reports/014-cycle13-report.md. Was pre-registered (cycle 13, 2026-11-11) before any search run; Felix approved the test.

## Question
Cycle 12 (proposal 015) found that `HessianNormalization::Marginal` with the *fixed* defaults
(mcw = 1, λ = 1) helps some sets and hurts others: Poisson bike_cnt +0.066, gaussian_sim +0.009.
The null explanation was that the right regularisation strength still differs by dataset.

So judge it as a tuning axis instead. When `min_child_weight` and `lambda` are tuned by `hyper_opt`
with the same number of trials and the same search range, does the normalised axis
- reach an equal or better tuned model (test NLL), and
- get there in fewer trials?

Reasoning: on the raw axis the useful range of mcw/λ moves with the Hessian scale. That is
≈ 0.06 per row on power_plant, ≈ 10⁶ on naval (Gaussian location), and ≈ 190 on Poisson bike_cnt.
A fixed search range then covers the useful region on some datasets and misses it on others. On the
normalised axis the same range means "10⁻³ to 10³ rows' worth" everywhere.

## Harness fix found while preparing (before any run)
`hyper_opt`'s fold models did not inherit `hessian_normalization` (or `stopping_row_cap`), so a search
on a normalised model tuned the *un*normalised model.
- Fixed in `src/hyper_opt.rs`.
- New getters `GradientLSS::hessian_normalization()` and `stopping_row_cap()`.
- Test `hessian_normalization_reaches_the_fold_models`, checked to fail without the fix.

## Protocol
Bench `BENCH_HPO=<n_trials>`, per (dataset, family, seed):
1. **Search.** `hyper_opt_with_config` on the training split.
   - 3-fold KFold (fold seed = bench seed, `hp_seed` = bench seed);
   - no pruning; the same rounds cap and early stopping as the bench (2000 / 50);
   - the default recipe: PES Auto, Damped;
   - eta 0.05 and depth 4 fixed.
   - `min_child_weight` ∈ [10⁻³, 10³] and `lambda` ∈ [10⁻³, 10³], log-uniform, TPE. The range is the
     same in both arms.
2. **Refit at budgets** k ∈ {5, 10, 20}. Take the best-so-far params after k trials, refit on the
   training split with validation early stopping (as in every bench cell) and score on test. Rows
   are tagged `c13_<arm>_k<k>`.
3. **Search diagnostics.** HPO lines on stderr give each trial's CV score and params.

Arms: `raw` (Off) and `norm` (`BENCH_HNORM=marginal`). TPE draws the same raw values in both arms
for the same seed, but scores differ, so later draws diverge.

Datasets (5 seeds each):
- Gaussian: kin8nm, power_plant, protein, naval, gaussian_sim;
- Poisson: bike_cnt, synth_pois;
- NB MD (`BENCH_LOC_START=1`): bike_cnt;
- Gamma MS: naval, kin8nm.

Count families use `BENCH_PIT_U=hash`.

Reference: the untuned defaults (c12_off / c12_marginal, same splits).

## Predictions
- T1 (main): at k = 20, norm − raw ≤ +0.005 test NLL on every dataset, and < −0.01 on at least two.
  The expected winners are Gaussian naval and Poisson bike_cnt, where the raw range sits far from
  the Hessian scale.
- T2 (speed): at k = 5, norm's advantage is larger than at k = 20 (a mean over datasets of
  norm − raw that is more negative at k = 5).
- T3 (CV agreement): the sign of norm − raw in best CV score at k = 20 matches the sign in test NLL
  on ≥ 7 of 10 datasets. Otherwise the gains are selection noise.
- T4 (tuning beats defaults): tuned raw at k = 20 beats the untuned Off default by ≥ 0.01 on at
  least half the datasets. If tuning these two parameters barely matters at all, the axis question
  is moot, and the conclusion should say so.

## Decision rule
- If T1 holds, the README recommends `HessianNormalization::Marginal` when tuning
  `min_child_weight`/`lambda`.
- No default change follows from this cycle: the fixed-default result (015) stands.
- If T1 fails, the opt-in stays documented as a unit-robustness option only.

## Results (c13; 5 seeds; runs `run_c13.sh`, logs /tmp/c13_{off,marginal}_{a,b}.log)
Test NLL, norm − raw (negative = the normalised axis is better):

| family | dataset | k = 5 | k = 10 | k = 20 | seeds better (k = 20) |
|---|---|---|---|---|---|
| Gaussian | kin8nm | +0.001 | −0.003 | **−0.035** | 5/5 |
| Gaussian | power_plant | −0.028 | +0.004 | −0.022 | 4/5 |
| Gaussian | protein | −0.028 | −0.029 | −0.024 | 3/5 |
| Gaussian | naval | **+1.22** | −0.030 | −0.049 | 4/5 |
| Gaussian | gaussian_sim | −0.002 | −0.002 | −0.000 | 4/5 |
| Gamma | kin8nm | −0.021 | −0.022 | **−0.025** | 5/5 |
| Gamma | naval | **+1.28** | −0.007 | +0.022 ± 0.028 | 2/5 |
| NB | bike_cnt | −0.001 | +0.001 | −0.008 | 5/5 |
| Poisson | bike_cnt | +0.014 | +0.005 | +0.017 ± 0.027 | 3/5 |
| Poisson | synth_pois | +0.000 | −0.000 | +0.001 | 1/5 |

Against the untuned defaults (c12_off), at k = 20:
- **Tuned raw:** gaussian_sim −0.008, kin8nm −0.025, power_plant −0.030, protein −0.021, Gamma kin8nm
  −0.053, Gamma naval −0.025, NB bike −0.004, synth_pois −0.001. It is worse on Gaussian naval
  (+0.053) and Poisson bike_cnt (+0.030).
- **Tuned norm:** gaussian_sim −0.008, kin8nm −0.059, power_plant −0.052, protein −0.045, Gamma
  kin8nm −0.079, Gamma naval −0.003, NB bike −0.012, Gaussian naval +0.004, synth_pois −0.000. It is
  worse on Poisson bike_cnt (+0.046).

Search diagnostics:
- **naval:** the first draws on the normalised axis include heavy regularisation (up to 10³ rows'
  worth), which is catastrophic there (k = 1 CV +3.9). On the raw axis the whole [10⁻³, 10³] range
  is light, because the native Hessian is ~10⁶ per row.
- **Chosen values (geometric mean at k = 20):**
  - normalised: naval mcw 0.07, λ 0.01 rows' worth; kin8nm λ ≈ 730 (the range edge on both axes).
  - Poisson bike_cnt: λ 0.37 rows' worth ≈ 70 raw, against 1.8 on the raw axis.

Predictions:
- **T1 ✗ (narrowly).** Better by < −0.01 on 6 sets (Gaussian kin8nm, power_plant, protein, naval;
  Gamma kin8nm; NB bike −0.008 ≈). Two sets miss the +0.005 band, both within one s.e.: Poisson
  bike_cnt +0.017 ± 0.027 and Gamma naval +0.022 ± 0.028.
- **T2 ✗.** The speed claim is reversed at k = 5: naval loses > 1 nat. The fixed normalised range
  admits a destructive heavy-regularisation corner on high-information data, which the raw range
  cannot reach. Elsewhere the k = 5 gaps are similar to the k = 20 gaps.
- **T3 ✓ (7/10 signs agree).** The exceptions are NB bike, gaussian_sim (both tiny) and Gaussian
  naval. On Gaussian naval CV prefers raw by 0.05 but test prefers norm by 0.05: naval's selection
  is noisy.
- **T4 ✓.** Tuned raw beats untuned Off by ≥ 0.01 on 5/10 sets (the normalised axis on 6/10).

Reading:
- With 20 trials, the normalised axis is the better place to tune `min_child_weight`/`lambda` on
  Gaussian, Gamma kin8nm and NB. Among the four recipes (untuned Off, untuned Marginal, tuned raw,
  tuned norm), tuned-norm is the best on 6/10 sets. The exceptions are naval (Gaussian: untuned
  Off; Gamma: untuned Marginal), Poisson bike_cnt (untuned Off) and synth_pois (tuned raw, by
  0.001).
- On Poisson bike_cnt, tuning these two parameters does not transfer from CV to test on either axis.
  Both tuned arms are worse than the untuned default.
- At very small budgets the normalised axis is risky where the per-row Hessians are huge, unless the
  range's upper end is narrowed.

**Decision:** no default change, and the library already ships the option.
- The README reports the equal-budget result as measured, including the Poisson bike_cnt and
  small-budget caveats.
- Whether to *recommend* the normalised axis for tuning goes to Felix: the two misses are
  noise-level, but they miss.

## Addendum N (cycle 13b, 2026-11-13, before any run) — pre-registered
Felix asked to re-run with the normalised range narrowed to [10⁻³, 10] for both mcw and λ. The aim
is to remove the destructive heavy-regularisation corner seen on naval at k = 5.

Arm `norm10`: Marginal normalisation, mcw and λ ∈ [10⁻³, 10] (log, `BENCH_HPO_HIGH=10`), with the same
20 trials, folds, seeds, datasets and refits at k ∈ {5, 10, 20}.

Comparator: the existing `c13_raw` rows, raw axis over [10⁻³, 10³]. The question is whether tuning
on the narrowed normalised range beats tuning the raw parameters as users do today. The raw arm is
not re-run with a narrowed range. On naval and Poisson bike_cnt the raw [10⁻³, 10³] range is
already light, so narrowing it would mostly remove options.

Predictions:
- N1: at k = 5, norm10 − raw on Gaussian and Gamma naval is ≤ +0.02 (the > 1 nat failure is gone).
- N2: at k = 20, norm10 − raw ≤ +0.005 on Gamma naval. That would show the c13 miss was the range.
- N3 (the expected cost): kin8nm (Gaussian) loses part of its c13 gain. The c13 normalised search
  chose λ ≈ 730 rows' worth there, which is now out of range. norm10 − norm at k = 20 is > 0 on
  kin8nm.
- N4 (rule): norm10 − raw at k = 20 is ≤ +0.005 on every dataset except possibly Poisson bike_cnt.
  Tuning does not transfer there on either axis (c13).

Decision: if N1, N2 and N4 hold, the README recommends the normalised axis with a range topping out
at 10 rows' worth, with the Poisson bike_cnt caveat. This goes to Felix as a documentation decision.
There is still no default change.

### Addendum N results (c13_norm10; 5 seeds; `run_c13b.sh`, logs /tmp/c13b_{1..4}.log)
Test NLL against the raw axis [10⁻³, 10³], and against the wide normalised range:

| family | dataset | norm10 − raw, k = 5 | norm10 − raw, k = 20 | norm10 − norm (wide), k = 20 |
|---|---|---|---|---|
| Gaussian | kin8nm | +0.015 | −0.000 | **+0.035** (1/5) |
| Gaussian | power_plant | −0.006 | −0.000 | +0.022 |
| Gaussian | protein | +0.037 | +0.022 ± 0.053 | +0.046 |
| Gaussian | naval | **+0.015** (c13: +1.22) | −0.004 | +0.045 |
| Gaussian | gaussian_sim | +0.005 | **+0.006** (0/5) | +0.006 |
| Gamma | kin8nm | −0.006 | +0.006 | **+0.031** (0/5) |
| Gamma | naval | **+0.016** (c13: +1.28) | **+0.034** ± 0.042 | +0.011 |
| NB | bike_cnt | +0.009 | −0.004 | +0.005 |
| Poisson | bike_cnt | +0.039 | **+0.043** (1/5) | +0.026 |
| Poisson | synth_pois | +0.001 | −0.001 | −0.002 |

- **N1 ✓.** The > 1 nat failure at k = 5 on naval is gone: +0.015 (Gaussian), +0.016 (Gamma).
- **N2 ✗.** Gamma naval at k = 20 is +0.034, worse than with the wide range (+0.022).
- **N3 ✓, and broader than predicted.** The narrowed range loses most of the wide range's gains:
  kin8nm +0.035, protein +0.046, naval +0.045, power_plant +0.022, Gamma kin8nm +0.031.
  - Search diagnostics: the chosen λ sits at 5.6–8.7 rows' worth on most sets, i.e. against the
    new upper edge.
  - Naval's first-trial CV loss is still +1.3: even 10 rows' worth is heavy regularisation there.
- **N4 ✗.** Over +0.005: gaussian_sim +0.006 (0/5), Gamma kin8nm +0.006, protein +0.022, Gamma naval
  +0.034, Poisson bike_cnt +0.043.

Reading:
- The best regularisation in rows' worth still spans about four decades across datasets:
  - naval: λ ≈ 0.01–0.1 rows' worth;
  - kin8nm: λ ≈ 700;
  - protein: λ ≈ 100.
- Normalisation removes the units, but not the dataset-to-dataset spread. No fixed range is both
  safe at 5 trials and complete at 20.
- The wide range with ≥ 20 trials remains the better recipe.

**Decision (pre-registered rule):** N2 and N4 fail, so the narrowed range is not recommended.
- The README says the wide normalised range needs ≥ 20 trials; narrowing it costs more than it
  saves.
- The open decision for Felix is unchanged: whether to recommend the wide normalised axis.
