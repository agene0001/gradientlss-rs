# Proposal 037 — library-level settings chosen per dataset

Status: **run** 2026-11-15 (P1 ✓, P2 ✓, V1 ✓ / ✗, V2 ✓, V3 ✓, F1 ✓; every part fails its rule; `V-start`
misses by one pair at +0.0052). **Pre-registered** 2026-11-14, after plumbing checks on a dataset outside the test pairs
(below) and before any run of any arm on the 36 pairs. Runs after 036. Skill:
`automl-hyperparameter-research` (Tier 2) with `optimization-algorithm-research` (the settings
are the optimiser's). Felix's queue, item 7.

## Question
The curvature mode, the start values, per-parameter early stopping (PES) and the family were
fixed as defaults across datasets (cycles 1–13) or chosen once per dataset by hand. Does choosing
them per dataset, by a search or by validation, beat leaving the defaults?

## Data
The 36 pairs of 033. They are not equally clean here: the defaults were chosen on the 10
development pairs (so those favour the default), the 14 unseen pairs saw one comparison of
today's defaults against the pre-research recipe (019/020), and the 12 fresh pairs have only
ever been fitted with the defaults. Everything is reported by set; the fresh pairs are the clean
part. A pass is confirmed on fresh downloads before any README advice.

## Parts
**P — PES inside the search.** Arm `pes`: mcw, λ, `__pes_window` ∈ {0, 5, 10, 20, 40, 80}
(0 = off; the default recipe is 20) and `__pes_anneal` uniform [0, 0.9] (default 0.5), 20 trials,
TPE, 3 folds, refits at 10 and 20; the refit applies the picked values the way hyper_opt applies
them to its fold models. Against the stored two-axis search under 033's rule: (a) guarded arm
worse than the untuned default by more than 0.005 on 0 pairs; (b) better than the guarded
two-axis search on at least 24; (c) median of that difference ≤ −0.005.

**V — a validation choice among untuned candidates.** Five untuned arms, each changing one
setting of the default: `v_obs` and `v_fis` (curvature Observed, Fisher instead of Damped),
`v_pesoff` (PES off instead of Auto), `v_init` (`initialize: true`, the L-BFGS start), `v_locflip`
(the other location-start setting: on for Gamma, off for Poisson and NegativeBinomial; Gaussian has
none and serves as an identity check). Selections, per seed, by the lowest validation NLL, the
default always among the candidates:
`V-mode` {default, obs, fis}; `V-start` {default, init, locflip}; `V-pes` {default, pesoff};
`V-all` all six. Rule for a selection, over the 36 pairs (mean over seeds of test NLL of the
pick minus the default's): worse by more than 0.005 on 0 pairs **and** better by more than 0.005
on at least 9.

**F — the family by validation (stored rows, no new fits).** On the 15 datasets that have two
families among the 36 pairs (11 with Gaussian and Gamma, 4 with Poisson and NegativeBinomial):
per seed, the family with the lower validation NLL of the untuned fit, against the family with
the lower test NLL. I have seen these pairs' mean test NLLs, not the validation picks: a new
statistic on seen data, reported as such.

Bench: `BENCH_HPO_SPACE=pes`, `BENCH_MODES`, `BENCH_PES=0`, `BENCH_INIT=1`, `BENCH_LOC_START`;
binary `/tmp/rb_c33`, `run_c33.sh`, jobs in `/tmp/c33/`, tags `c33_pes_k<k>`, `c33_v_*`.
The family is not put inside hyper_opt as an axis: with two candidates, fitting both and
comparing on validation is the same choice without spending trials.

## Plumbing (Gamma synth_gamma, seed 0, not one of the 36)
Default 2.43434 (equal to the earlier binaries); Observed 2.49086, Fisher 2.44144, PES off
2.43370, initialize 2.44283, location start 2.44606; a 5-trial `pes` search draws windows 40, 40,
80, 40, 10 and its refit runs. Seen when the predictions were written.

## Predictions
- **P1 (`pes` does not pay).** Rule fails: (b) below 24 and (c) above −0.005. Plain `pes` is
  worse than plain two axes by more than 0.005 on at least 8/36 pairs (two more axes at 20
  trials).
- **P2.** The picked window is 0 (off) in fewer than 20 % of the 180 searches.
- **V1 (each alternative loses more often than it wins).** Untuned, against the default, worse
  by more than 0.005 on at least: `v_obs` 12, `v_fis` 8, `v_init` 10, `v_pesoff` 8 of 36 pairs;
  each is better by more than 0.005 on at least 3.
- **V2 (`V-pes` is a no-op).** `V-pes` is within ±0.005 of the default on at least 28/36 pairs.
- **V3 (`V-all` fails on safety).** `V-all` is better than the default by more than 0.005 on at
  least 10/36 pairs and worse on at least 3, so the rule fails; on the 12 fresh pairs alone it
  is better on at least 4.
- **F1.** The validation pick is the family with the lower test NLL in at least 80 % of the 75
  (dataset, seed) cells, and its mean regret per dataset is below 0.01 on at least 12/15.

## Decision
A part that passes its rule goes to a fresh-data confirmation (new proposal); the README says
nothing until then. A part that fails: the README's tuning section records it as tested, with
the counts. No default changes from this proposal.

## Cost
180 four-axis searches (about 8 stream-hours) and 900 untuned fits (about 3).

## Result (2026-11-15, report 016 §15)
1080 jobs (180 four-axis searches with refits at 10 and 20, 900 untuned fits), no failed job.
`v_fis` is non-finite on Gaussian power_plant in 5/5 seeds; every other cell is finite. Scored
with `c33_score.py` and `c30_summary.py … c33_pes=2d@20`.

### P — PES window and anneal inside the search (20 trials, against the stored two axes)
| | plain mean | guarded mean | guarded median | (a) | (b) | (c) | plain better / worse than two axes by > 0.005 |
|---|---|---|---|---|---|---|---|
| two axes | −0.054 | −0.056 | −0.021 | 1 | | | |
| `pes` | −0.047 | −0.036 | −0.019 | 1 | 20 | −0.0003 | 13 / 11 |

Rule: fail on (a) (Gaussian c_n_appliances +0.028), (b) and (c). By set, (b) and (c):
development 8/10, −0.012; unseen 7/14, −0.000; fresh 5/12, +0.004. At 10 trials 5 / 18 /
−0.0002. Picked window over the 180 searches: 0 (off) 7, 5 → 43, 10 → 36, 20 (the default) → 31,
40 → 40, 80 → 23; no value is preferred, and the draws were spread evenly (12–20 % each). Picked
anneal quartiles 0.20 / 0.41 / 0.64 (default 0.5).

### V — untuned alternatives against the default (test NLL, 36 pairs unless noted)
| arm | better by > 0.005 | worse by > 0.005 | within | median | note |
|---|---|---|---|---|---|
| `v_obs` (Observed curvature) | 1 | 28 | 7 | +0.031 | Gaussian protein +3.0, c_supercond +2.6 |
| `v_fis` (Fisher) | 7 | 15 | 13 | +0.002 | 35 finite pairs; Gaussian power_plant non-finite; Gaussian c_n_appliances 1.9·10¹¹, c_n_sgemm +117, c_airfoil +92, c_supercond +16 |
| `v_pesoff` (PES off) | 0 | 28 | 8 | +0.097 | the 8 are the 5 Poisson pairs (one parameter: PES cannot act) and 3 near zero |
| `v_init` (L-BFGS start) | 11 | 13 | 12 | 0.000 | Gaussian c_n_sgemm +1.54, c_parkinsons +0.18; protein −0.070, airfoil −0.061 and −0.062 |
| `v_locflip` (other location start) | 3 | 13 | 5 | +0.017 | 21 non-Gaussian pairs; Poisson c_seoulbike 3.7·10¹², c_n_metro 2.4·10⁴; Gaussian identical 75/75 |

Selections per seed by validation NLL (default always a candidate); test NLL of the pick minus
the default's, pair means:

| selection | better by > 0.005 | worse by > 0.005 | within | mean | picks of the default | rule |
|---|---|---|---|---|---|---|
| `V-mode` {default, obs, fis} | 12 | 9 | 15 | −0.002 | 97/180 | fail |
| `V-start` {default, init, locflip} | 16 | 1 | 19 | −0.010 | 94/180 | fail |
| `V-pes` {default, pesoff} | 0 | 1 | 35 | +0.001 | 174/180 | fail |
| `V-all` (six candidates) | 20 | 5 | 11 | −0.009 | 61/180 | fail |

- `V-start`'s one loss is Gamma c_n_wine +0.0052. By set, better / worse: development 4 / 0,
  unseen 7 / 0, fresh 5 / 1. Choosing by test NLL instead (not available in practice) would give
  −0.016.
- `V-mode`'s losses: Gaussian c_gt_co +0.045, NegativeBinomial c_n_news +0.013, Gaussian
  c_n_appliances +0.012, Gamma c_n_concrete +0.012, c_gt_co +0.011, c_n_appliances +0.010, c_gt_nox
  +0.008, Gaussian c_gt_nox +0.006, NegativeBinomial c_n_metro +0.005. Development 6 / 0, unseen
  3 / 4, fresh 3 / 5: the gains are on the pairs the defaults were chosen on.
- `V-all`'s losses: Gaussian c_gt_co +0.047, NegativeBinomial c_n_metro +0.011, Gaussian
  c_n_appliances +0.011, Gamma c_n_concrete +0.011, c_n_wine +0.005. Fresh pairs: 6 / 4.
- Validation never picked one of `v_fis`'s or `v_locflip`'s blow-ups.

### F — the family by validation (stored rows)
Right family in 65/75 (dataset, seed) cells (86.7 %); mean regret per dataset below 0.01 on
12/15; largest 0.294 (Gaussian against Gamma on c_gt_co), then c_n_concrete 0.061, c_parkinsons
0.020. Mean regret over datasets: validation pick 0.026; always Gamma / NegativeBinomial 0.024;
always Gaussian / Poisson 3.28. The pick matters on the count data (Poisson is 2 to 32 nats
worse on three of four); between Gaussian and Gamma it is no better than always taking Gamma.

### Predictions
- **P1 ✓.** (b) 20, (c) −0.0003; plain `pes` worse than plain two axes on 11/36.
- **P2 ✓.** Window 0 picked in 7/180 searches (3.9 %).
- **V1 ✓ / ✗.** Worse on 28, 15 (plus one non-finite), 13 and 28 pairs (predicted at least 12,
  8, 10, 8 ✓). "Each better on at least 3" ✗ for `v_obs` (1) and `v_pesoff` (0): I underrated
  how completely the default dominates those two.
- **V2 ✓.** `V-pes` within ±0.005 on 35/36.
- **V3 ✓.** `V-all` better on 20, worse on 5; on the fresh pairs better on 6.
- **F1 ✓.** 86.7 % and 12/15.

### Decision
- **Every part fails its rule; final for this proposal.** The README records them as tested.
- **`V-start` is a near miss** (16 better, one worse by 0.0002 over the line, clean on the
  unseen and fresh sets apart from that pair). For Felix: whether a validation choice between
  the default start and the L-BFGS start (`initialize: true`) is worth a fresh-data test under
  the same rule; by the rule as written it is not recommended.
- **Not to do:** Fisher curvature on Gaussian (non-finite or 2.9 to 10¹¹ nats worse on 8 of 15
  datasets), the other location start on Poisson and NegativeBinomial, Observed curvature, PES
  off. These are what the defaults already avoid; this run confirms them on 26 pairs the
  defaults were not chosen on.
- No default changes.
