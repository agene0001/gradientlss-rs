# Proposal 035 — the untested XGBoost parameters, one search axis at a time

Status: **run** 2026-11-14/15 (S0 ✓, E0 ✓, E1 ✓, E2 ✓ except one naval pair, R1 ✓ for gamma and ✗ for
alpha's (b), K1 ✓ / ✓ / ✗, D1 ✓, L0 ✓, L1 ✓ / ✗; seven arms fail, `leaf` passes → 039).
**Pre-registered** 2026-11-14, after plumbing checks on a dataset outside the test pairs
(below) and before any run of any arm on the 36 pairs. Runs after 034 finishes. Skill:
`automl-hyperparameter-research` (search-space design; Tier 2). Felix: "queue these to test after
we are done with the current running test" (items 1–5 of his list; items 6 and 7 are 036 and 037,
see "Queue" at the end).

## Question
Which booster parameters that no cycle has searched are worth a search axis next to
`min_child_weight` and `lambda`? Tested so far: max_depth (gains and all the losses), subsample
with colsample_bytree (safe with the guard, usually better), max_bin (033: no).

## Data, and what a pass means
The 36 (family, dataset) pairs of 033 (10 development, 14 unseen, 12 fresh). None of them has
seen a search over any parameter below. Seen before: eta at fixed values 0.15, 0.3, 0.5, 1.0 on
12 of the pairs (030), which is why E1 is not a blind prediction on those. Eight arms are screened
under one rule, so one arm passing by chance is likely enough to matter: a **pass** is development
evidence and is confirmed on freshly downloaded datasets before the README's advice changes; a
**fail** is final at this budget.

## Arms (each: mcw and λ log-uniform [10⁻³, 10³] plus the one parameter; 20 trials, TPE, 3 folds)
| arm | added parameter | range |
|---|---|---|
| `eta` | eta | log-uniform [0.02, 0.3]; round cap 2000 as everywhere |
| `gamma` | gamma (min split loss) | log-uniform [10⁻³, 10²] |
| `alpha` | alpha (L1) | log-uniform [10⁻³, 10²] |
| `ctree` | colsample_bytree | uniform [0.5, 1] |
| `clevel` | colsample_bylevel | uniform [0.5, 1] |
| `cnode` | colsample_bynode | uniform [0.5, 1] |
| `mds` | max_delta_step | log-uniform [0.1, 10] (the default 0 = no cap is not in the range) |
| `leaf` | grow_policy = lossguide, max_depth = 0, max_leaves | log-uniform integer [4, 64] |

Untuned arms (no search, 180 fits each): `u_repro` (`max_delta_step=0,gamma=0`, i.e. the default
spelled out), `u_cap6000` (round cap 6000), `u_eta015` (eta 0.15), `u_lg16` (lossguide, 16 leaves:
the leaf count of the default depth 4).

Everything else as 033: eta 0.05, depth 4, 2000 rounds, patience 50, seeds 0–4, refits at
k ∈ {10, 20}. Bench `BENCH_HPO_SPACE=<arm>`, `BENCH_XGB=key=value,...`, binary `/tmp/rb_c31`,
`run_c31.sh` (one job per arm, pair and seed, own CSV in `/tmp/c31/`), tags `c31_<arm>_k<k>`,
`c31_u_*`. References: untuned `c12_off` / `c19_def` / `c17_new` / `c27_def`; two axes `c13_raw` /
`c19_tpe` / `c27_2d`; `subs` (`c25_subs`, `c27_subs`) and `depth` (`c25_depth`, 24 pairs)
descriptively. "Guarded" keeps a refit only if its validation NLL beats the untuned fit's.

## Plumbing (Gamma synth_gamma, seed 0, not one of the 36)
Every arm draws its parameter and its refit runs (3 trials each); `ctree`, `clevel`, `cnode` give
three different fold scores at the same mcw, λ and fraction; `BENCH_XGB=max_delta_step=0,gamma=0.0`
equals the default bit for bit (NLL 2.43434); lossguide with 16 leaves 2.4375, with 4 leaves
2.4092, gamma = 5 2.4197, eta 0.01 with cap 6000 2.4276. These numbers were seen when the
predictions were written.

## Inclusion rule (033's), per arm at 20 trials over the 36 pairs
(a) guarded arm worse than the untuned default by more than 0.005 on 0 pairs; (b) guarded arm
better than the guarded two-axis search on at least 24 pairs; (c) the median over pairs of that
difference ≤ −0.005. Pass → fresh-data confirmation (new proposal). Fail → the README lists the
parameter as tested and not worth an axis, with the counts. No default changes.

## Predictions
- **S0.** `u_repro` equals the stored untuned rows bit for bit on 180/180 cells.
- **E0 (the cap binds on a few).** `u_cap6000` is better than the default by more than 0.005 on
  at least 6/36 pairs and worse on at most 2.
- **E1.** `u_eta015` is worse than the default by more than 0.005 on at least 15/36 pairs and
  better on at least 4.
- **E2 (`eta` axis).** The search picks eta above 0.1 (median over seeds) on both naval pairs;
  plain `eta` beats plain two axes by more than 0.005 on at least 12/36 pairs; the rule fails:
  (a) at least one pair, and (b) fewer than 24.
- **R1 (`gamma`, `alpha`: nothing).** For each, the median over pairs of guarded arm − guarded two
  axes is within ±0.005 and (b) is below 24.
- **K1 (the three colsample axes are one axis).** The guarded medians (arm − two axes) of `ctree`,
  `clevel` and `cnode` lie within 0.005 of each other; at least one of the three reaches (b) ≥ 24;
  none passes (c).
- **D1 (`mds` hurts).** Plain `mds` is worse than plain two axes by more than 0.005 on at least
  10/36 pairs; the rule fails.
- **L0.** `u_lg16` is within ±0.005 of the depth-4 default on fewer than 18/36 pairs.
- **L1 (`leaf` behaves like depth).** Plain `leaf` beats plain two axes by more than 0.005 on at
  least 15/36 pairs, and guarded `leaf` is worse than the untuned default on at least one pair, so
  (a) fails.

## Cost
1440 searches and 720 untuned fits; about 60 stream-hours on a free machine, in resumable waves
of 10 single-thread streams under the 2-hour task limit. Arms run in the order of the table, so
each can be scored as it completes.

## Queue (Felix's list, in his order)
1. eta with the round cap → `eta`, `u_cap6000`, `u_eta015` (this proposal).
2. gamma, alpha → `gamma`, `alpha` (this proposal).
3. colsample_bylevel / colsample_bynode → `ctree`, `clevel`, `cnode` (this proposal).
4. max_delta_step → `mds` (this proposal).
5. grow_policy with max_leaves → `leaf`, `u_lg16` (this proposal).
6. LightGBM's own parameters (num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction)
   → proposal 036, to be pre-registered before it runs: needs a LightGBM build of the bench and
   LightGBM reference arms (untuned, two axes) on the 36 pairs first.
7. Library-level settings chosen per dataset (per-parameter early stopping window and anneal via
   hyper_opt's `__pes_*` axes; curvature mode, start values and family as a validation choice
   among fitted candidates) → proposal 037, to be pre-registered before it runs.

## Result (2026-11-15, report 016 §12)
2160 jobs (1440 twenty-trial searches with refits at 10 and 20, 720 untuned fits), no non-finite
row, no failed job. Two drive disconnects and three 2-hour task limits interrupted the waves; the
jobs they killed were rerun from scratch and every finished job was checked (2 rows, 34 fields,
empty error field). Scored with `c31_score.py`.

### Untuned arms (test NLL − the untuned default, 36 pairs, pair means over 5 seeds)
| arm | better by > 0.005 | worse by > 0.005 | within | mean | median |
|---|---|---|---|---|---|
| `u_repro` | identical on 180/180 cells | | | | |
| `u_cap6000` (round cap 6000) | 6 | 0 | 30 | −0.005 | 0.000 |
| `u_eta015` (eta 0.15) | 9 | 22 | 5 | +0.024 | +0.018 |
| `u_lg16` (lossguide, 16 leaves, max_depth 0) | 27 | 5 | 4 | −0.145 | −0.058 |

The default stops at the 2000-round cap in at least one seed on 11 pairs; the six gains of
`u_cap6000` are among them (naval −0.052 and −0.050, SGEMM −0.022 and −0.016, superconductivity
−0.019, airfoil −0.006). `u_lg16` by set, better / worse: development 8 / 0, unseen 11 / 1, fresh
8 / 4. Its losses: Gaussian c_n_concrete +0.081, c_n_appliances +0.042, Gamma c_n_concrete +0.036,
NegativeBinomial c_n_news +0.033, Gaussian c_gt_co +0.006. Largest gains: Poisson c_n_metro −1.99,
SGEMM −0.50 and −0.42, Parkinson's −0.38 and −0.28.

### Search arms at 20 trials against the stored two-axis search
"Plain" = the refit as it is; "guarded" = kept only if its validation NLL beats the untuned
fit's. Two axes for reference: plain mean −0.054, guarded mean −0.056, guarded median −0.021,
guarded worse than default on 1/36.

| arm | plain mean | guarded mean | guarded median | (a) guarded worse than default | (b) better than guarded two axes | (c) median difference | plain better / worse than two axes by > 0.005 | rule |
|---|---|---|---|---|---|---|---|---|
| `eta` | −0.062 | −0.039 | −0.022 | 2 | 20 | −0.0003 | 17 / 13 | fail |
| `gamma` | −0.043 | −0.031 | −0.013 | 2 | 14 | +0.0036 | 7 / 14 | fail |
| `alpha` | −0.072 | −0.073 | −0.026 | 3 | 28 | −0.0045 | 20 / 5 | fail |
| `ctree` | −0.042 | −0.047 | −0.026 | 0 | 22 | −0.0044 | 18 / 7 | fail |
| `clevel` | −0.061 | −0.060 | −0.016 | 3 | 22 | −0.0032 | 19 / 5 | fail |
| `cnode` | −0.078 | −0.075 | −0.029 | 1 | 28 | −0.0080 | 21 / 3 | fail |
| `mds` | +0.034 | −0.034 | −0.014 | 2 | 15 | +0.0007 | 6 / 17 | fail |
| `leaf` | −0.229 | −0.208 | −0.057 | 0 | 34 | −0.0381 | 30 / 0 | **pass** |

Guard misses, (a): `eta` Gaussian c_n_concrete +0.014, c_n_appliances +0.012; `gamma` Gaussian
c_n_concrete +0.030, c_n_appliances +0.026; `alpha` Gaussian c_n_concrete +0.068, protein +0.010,
Poisson bike_cnt +0.009; `clevel` Gaussian c_supercond +0.028, c_n_appliances +0.022, Poisson
c_seoulbike +0.008; `cnode` Gaussian c_parkinsons +0.009; `mds` Gaussian c_gt_co +0.040,
power_plant +0.014.

At 10 trials (not the budget the rule is applied at): `alpha` 0 / 29 / −0.0062 and `leaf`
0 / 35 / −0.042 meet all three conditions; `cnode` 1 / 29 / −0.0069 and `clevel` 3 / 24 / −0.0051
miss only (a); the rest miss (b) and (c).

`leaf` by set, (b) and (c): development 10/10, −0.060; unseen 12/14, −0.039; fresh 12/12,
−0.030. The plain `leaf` refit was worse than the untuned default on 0/36 pairs (two axes: 6).
Against the stored searches of cycles 25–27, guarded: better than the depth search on 20 of its
24 pairs (median −0.029; the depth search has 2 guard misses there, `leaf` none) and better than
the four-axis subsampling search on 26/36 (median −0.019). Its refits cost 4.7 s on average
against 3.2 s for the default.

### Picked values (best trial of each 20-trial search, quartiles over the 180 searches)
eta 0.039 / 0.062 / 0.104; gamma 0.002 / 0.010 / 0.041 (near the lower bound: the search turns
it off); alpha 0.08 / 3.6 / 15.6; colsample_bytree 0.68 / 0.82 / 0.93; colsample_bylevel 0.60 /
0.72 / 0.84; colsample_bynode 0.59 / 0.69 / 0.86; max_delta_step 1.05 / 2.5 / 5.4; max_leaves
10 / 16 / 28. The median max_leaves by pair ranges from 4–6 (Gaussian c_n_appliances,
c_n_concrete, gaussian_sim, Gamma c_n_concrete, NegativeBinomial c_n_news; four of these five are
pairs where `u_lg16` lost) to 43–61 (SGEMM, wine, gas-turbine CO, Poisson bike_cnt).

### Predictions
- **S0 ✓.** 180/180 identical.
- **E0 ✓.** Better on 6, worse on 0.
- **E1 ✓.** Worse on 22, better on 9.
- **E2 ✓ except one detail.** Median picked eta: Gaussian naval 0.149, Gamma naval 0.097 (just
  under the 0.1 I predicted for both). Plain `eta` beat plain two axes on 17/36 (≥ 12 ✓); the rule
  failed with (a) 2 and (b) 20 ✓. The gains are where the cap binds (Gamma c_n_sgemm −0.136,
  Gaussian c_n_sgemm −0.063, Gamma naval −0.057); elsewhere a third axis mostly costs.
- **R1 ✓ for `gamma`** ((c) +0.0036, (b) 14). **For `alpha`:** (c) −0.0045 is inside ±0.005 ✓, but
  (b) is 28, not below 24 ✗. L1 regularisation is not "nothing": plain `alpha` beat plain two axes
  on 20 pairs and lost on 5. It fails on three guard misses.
- **K1 ✓ / ✓ / ✗.** The three guarded medians are −0.0044, −0.0032, −0.0080 (spread 0.0048 ✓);
  `cnode` reaches (b) 28 ✓; but `cnode` also meets (c), which I predicted none would ✗. By-node
  sampling was the best of the three on every column of the table, and fails only on one pair
  (+0.009).
- **D1 ✓.** Plain `mds` worse than plain two axes on 17/36; the rule fails on all three counts.
- **L0 ✓.** `u_lg16` within ±0.005 on 4/36.
- **L1 ✓ / ✗.** Plain `leaf` beat plain two axes on 30/36 (≥ 15 ✓). I predicted at least one
  guard miss, as with depth ✗: there was none, with or without the guard.

### What I had wrong
- I treated the leaf count as depth under another name. Depth as an axis carried all the losses
  of cycles 25–26; a leaf budget with best-first growth carried none here. The two searches
  differ in three ways at once: the size range (depth 2–8 is 4 to 256 leaves, uniform over
  depths; `leaf` is 4 to 64 leaves, log-uniform), the resolution (seven depths against any leaf
  count), and where the splits go (level by level against best first). This run does not
  separate them.
- I expected the untuned leaf-wise tree to differ from the default but not to be better on 27 of
  36 pairs. It still loses on 5, so it does not meet the default rule (≤ +0.005 everywhere).
- alpha and by-node sampling are closer to useful than I predicted. Both fail the rule as
  written, on (a).

### Decision
- **`eta`, `gamma`, `alpha`, `ctree`, `clevel`, `cnode`, `mds`: fail, final at 20 trials.** The
  README lists them as tested with the counts. 034 and 038 showed that a verdict can change with
  the budget, so each is stated with its budget. `alpha` and `cnode` are the two that would be
  worth a larger-budget test if Felix wants one.
- **`leaf`: pass → proposal 039** (confirmation on the 12 `c_m_*` pairs, which no leaf-wise fit
  has touched), before the README's advice changes. 039 passed (0/12, 12/12, −0.068); the README
  now recommends the axis.
- No default changes. The round cap (E0: 6 pairs gain up to 0.05 from a cap of 6000, none lose)
  and a leaf-wise default (`u_lg16`) are questions for Felix, not changes made here.
