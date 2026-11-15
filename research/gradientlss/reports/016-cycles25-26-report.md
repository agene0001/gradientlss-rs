# Cycles 25–26 report — the four items left open by report 015

Date: 2026-11-14. Proposals 027–030. Rows: `experiments_v2.csv`, tags `c25_*` (search-space arms)
and `c26_*` (Gumbel, NB CRPS, gradient regularisation). Skills: `automl-hyperparameter-research`
(027), `optimization-algorithm-research` (028–030). Felix asked to look into the four items report
015 §11 listed as still open: Gumbel, NB CRPS in mean/dispersion coordinates, which added search
axis carries the wide-space gain, adaptive λ.

Cycles 25–26 changed no default and every addition was opt-in; the decisions of §6 were then
approved and applied (§8), which changes the Gumbel default and the coordinates NegativeBinomial
uses under the CRPS loss. Nothing is committed.

All numbers are means over 5 seeds, paired by seed, on XGBoost. "The rule" is the project's: a
change must be better or neutral (≤ +0.005 test NLL) on every dataset.

## 0. What changed in the code (uncommitted)
| where | what | default |
|---|---|---|
| `src/distributions/gumbel.rs` | `Gumbel::with_start_scale_factor(k)` / `start_scale_factor()`: with the `LocationScale` start, the start scale is k × the moment-matched one | 1.0 (serde default 1.0): unchanged |
| `src/distributions/negative_binomial.rs` | `NegativeBinomial::with_mean_dispersion_crps()`: CRPS gradients and expected-curvature Hessians in (log µ, log θ) | off: CRPS still boosts in (total_count, probs) |
| `src/model.rs` | `GradientRegularization { strength, excess_only }`, `GradientLSS::{with,set}_gradient_regularization`, `gradient_regularization()` | `None`; not in the prelude |
| `examples/research_bench.rs` | `BENCH_HPO_SPACE=depth\|subs`, `BENCH_LS_SCALE_K`, `BENCH_NB_CRPS_MD`, `BENCH_GRN`, `BENCH_SEED_START`, dataset `synth_gumbel` (Gumbel oracle), trimmed-NLL debug line | — |

This table is the state after cycle 26; §8 lists what changed when the decisions were applied
(the first two rows' defaults among them).

Tests: 267 unit and 129 integration tests pass (`cargo test --release --features xgboost --lib
--tests`); three are new (NB CRPS gradients and curvature against finite differences; the
regulariser's formula and unit-freeness; the Gumbel start factor).

## 1. Which axis carries the wide-space gain? (proposal 027, cycle 25)
The five-axis search of 024 / 025 split into its halves at the same 20 trials: `depth` (mcw, λ,
max_depth 2–8), `subs` (mcw, λ, subsample, colsample_bytree in [0.5, 1]), and `d6`, the untuned
default at depth 6. 240 searches. Test NLL minus the untuned default; "guarded" keeps a refit only
if it beats the untuned fit on validation.

| arm | development: mean | worse than default | guarded: mean (worse) | unseen: mean | worse than default | guarded: mean (worse) |
|---|---|---|---|---|---|---|
| two axes | −0.008 | 2/10 | −0.019 (0/10) | −0.017 | 2/14 | −0.023 (0/14) |
| `depth` | −0.055 | 1/10 | −0.057 (1/10) | −0.081 | 2/14 | −0.076 (1/14) |
| `subs` | −0.054 | 0/10 | −0.052 (0/10) | −0.055 | 2/14 | −0.045 (0/14) |
| all five | −0.089 | 0/10 | −0.089 (0/10) | −0.131 | 2/14 | −0.121 (2/14) |
| `d6` | −0.002 | 5/10 | | −0.032 | 4/14 | |

- **Each half carries about 60 %** of the five-axis gain on the development pairs (61 % and 61 %);
  on the unseen pairs depth 61 %, subsampling 42 %. They overlap.
- **The losses are the depth axis.** Gaussian c_gt_co: `depth` +0.168, `subs` +0.011. Gaussian
  c_supercond: +0.067, −0.024. Gaussian protein (development): +0.029, −0.068. The guard does
  not catch c_gt_co (+0.149). The largest gains are depth too (c_parkinsons −0.44 and −0.25).
- **`subs` never lost to the two-axis search** (better or within 0.002 on 24/24 pairs) and with
  the guard is worse than the default on 0/24. Seen after the fact; not a tested rule.
- **`d6`:** a fixed deeper tree is not the gain (4 % of `depth`'s on development; worse than
  depth 4 on 9/24 pairs, Gaussian c_supercond +0.58). The gain is in picking the depth per dataset.
- **Tier 1:** `subs` has the lower best CV score than two axes on 9/10 and 13/14 pairs, `depth` on
  8/10 and 8/14.
- **Predictions: A1 ✗, A2 ✗, A3 ✗ (the opposite), A4 ✗ / ✓, A5 ✗.** I expected depth to carry the
  gain and subsampling the losses.
- **By the rule:** no recommendation for `depth`; README gets the table.

## 2. Gumbel (proposal 028, cycle 26)
**Diagnosis.** 021's "blow-ups on single seeds" are single test rows 10–17 scales below the
predicted location. The Gumbel NLL charges e^{−z} there: one row costs up to 9·10⁸, the cell's
CRPS does not move. All 27 blow-up cells of the experiment (a cell whose mean drops by more than
0.5 without its worst row) have that row at z < −5. It is the density, not the optimiser.

Separately, the default (observed curvature, raw 0.5 start) **stalls** on any target far from
0.5: on data simulated from a Gumbel at location ≈ 120 it ends 3.50 nats above the oracle at a
scale of 115.

**Prototype.** Damped curvature, `LocationScale` start, start scale = k × the matched one. k
chosen on six development sets by validation NLL (grid 1–100 after one amendment; statistic
changed from mean rank to mean NLL in a second, both before the unseen run): **k = 100**.

| dataset | default: NLL | k = 100: NLL | median seed | trimmed NLL (worst 0.5 % of rows dropped) | CRPS |
|---|---|---|---|---|---|
| synth_gumbel (oracle 2.597) | 6.097 | 2.683 | 6.097 → 2.698 | 6.095 → 2.646 | 50.98 → 2.22 |
| gaussian_sim | 2.356 | 2.217 | 2.325 → 2.219 | 2.230 → 2.154 | 1.466 → 1.410 |
| kin8nm | −0.159 | −0.367 | −0.169 → −0.362 | −0.233 → −0.415 | 0.103 → 0.088 |
| naval | −4.59 | −5.42 | −4.73 → −5.43 | −5.32 → −5.55 | 0.0008 → 0.0007 |
| power_plant | 7.43 | **2622** | 7.43 → 3.12 | 7.43 → 2.92 | 190.5 → 2.51 |
| protein | 73 681 | 2.934 | 2.864 → 2.832 | 2.736 → 2.716 | 2.642 → 2.559 |
| c_airfoil | 6.13 | 2.51 | 6.13 → 2.56 | 6.13 → 2.40 | 52.3 → 1.70 |
| c_gt_co | 1.120 | 1.122 (+0.002 ± 0.005) | 1.123 → 1.133 | 1.040 → 1.034 | 0.473 → 0.468 |
| c_gt_nox | 7.25 | 3.17 | 3.29 → 3.17 | 3.12 → 3.06 | 3.35 → 3.19 |
| c_parkinsons | 14.77 | 2.91 | 14.88 → 2.90 | 14.70 → 2.78 | 25.9 → 3.08 |
| c_supercond | 49 992 | 4.29 | 4.45 → 3.63 | 841 → 3.59 | 12.6 → 7.33 |

- The rule on mean NLL holds on 10/11 pairs and fails on power_plant, through two single-row
  cells (8.4 and 13 091; the other three seeds 3.07–3.13). The default cannot blow up there
  because it never fits (scale 430). Trimmed NLL: 11/11. CRPS equal or better: 11/11.
- The default has 6 blow-up cells of its own (naval, protein ×2, c_gt_nox, c_supercond ×2); the
  recipe 4.
- Why wide: at a scale far above the residuals the Damped location step is η·(y − μ), least
  squares; the log-scale falls by about η per round, so k = 100 buys about 92 rounds of location
  fitting before the e^{−z} tail weights single rows. Validation NLL improved monotonically in k
  on naval and kin8nm and was flat elsewhere; the trend is not exhausted at 100.
- **Predictions: G1 ✗ on its threshold** (0.087 above the oracle, not within 0.05), **G2 ✓, G3 ✓,
  G4 ✓, G5 ✓.**
- Caveats: the unseen sets had been used once before for Gumbel (021's recipe); the selection
  statistic was changed after seeing which arm it favoured (the as-written pick, k = 4, is
  reported in the proposal: worse on three unseen sets).
- **By the rule:** no default change; the decision goes to Felix (§6).

## 3. NegativeBinomial: CRPS training in (log µ, log θ) (proposal 029, cycle 26)
Gradient by the chain rule through (r, probs) = (θ, µ/(µ+θ)); Hessian the expected CRPS curvature
2 Σ_k (∂F(k)/∂η)² for every row. Test CRPS:

| dataset (mean count) | NLL-trained | CRPS, fallback coordinates | CRPS, (log µ, log θ) | against the fallback | against NLL training |
|---|---|---|---|---|---|
| synth_nb_lo | 4.017 | 4.136 | 4.040 | −2.3 % ± 0.3 (5/5) | +0.6 % ± 0.1 (0/5) |
| c_abalone (10) | 1.2069 | 1.612 | 1.2082 | −25.1 % ± 0.7 (5/5) | +0.1 % ± 0.3 (3/5) |
| bike_casual (36) | 6.07 | 6.29 | 6.00 | −4.7 % ± 1.1 (5/5) | −1.2 % ± 0.9 (4/5) |
| bike_cnt (190) | 17.22 | 19.19 | 16.77 | −12.7 % ± 0.8 (5/5) | −2.7 % ± 1.1 (4/5) |
| c_seoulbike (700) | 69.79 | 70.62 | 65.66 | −7.0 % ± 1.2 (5/5) | −5.9 % ± 1.0 (5/5) |

- Better than the fallback in 25/25 cells; no fit at the round cap (c_abalone was 5/5 at it).
- **CRPS training now beats NLL training on the three real sets with larger counts** (N4 ✗: I
  predicted at most 1/5). It pays in the log score (+0.06, +0.13, +0.63 nats) and in tail
  coverage: the 90 % interval covers 0.865 / 0.847 / 0.785 of the labels against 0.899 / 0.928 /
  0.934 for the NLL-trained model.
- Fit times were measured under load at one thread and are not a result; every row now builds its
  table, so it is not cheaper than the fallback.
- **Predictions: N1 ✓, N2 ✓, N3 ✓, N4 ✗, N5 ✓, N6 ✓.**
- **By the rule:** goes to Felix (§6); the README's "train on the NLL" is qualified for NB.

## 4. Adaptive λ: gradient-regularised Newton boosting (proposal 030, cycle 26)
Zozoulenko et al. (arXiv 2605.00581): λ_k = sqrt(M‖g_k‖) added to the round's L² penalty, i.e. to
every row's Hessian. Here per parameter with M = h̄^{3/2}: λ = c·h̄·ρ^{1/4}, ρ = mean g²/h̄ (`grn`),
and an excess-only form with ρ^{1/4} − 1 (`grnx`); c = 1. 12 pairs, eta ∈ {0.05, 0.3, 1}, plus
vanilla at 0.15 and 0.5 as half-step controls (added while running). 660 fits.

| comparison (test NLL) | better by > 0.005 | worse by > 0.005 | mean |
|---|---|---|---|
| vanilla eta 0.3 − eta 0.05 (threshold 0.02) | 0/12 | 9/12 | +0.080 |
| `grn` − vanilla at eta 0.3 | 8/12 | 0/12 | −0.040 |
| `grn` − vanilla at eta 1 | 12/12 | 0/12 | −0.169 |
| `grnx` − vanilla at eta 0.3 | 4/12 | 0/12 | −0.019 |
| `grn` − vanilla at eta 0.05 | 1/12 | 8/12 | +0.108 |
| **`grn` at eta 0.3 − vanilla at eta 0.15** | 2/12 | 8/12 | +0.027 |
| **`grn` at eta 1 − vanilla at eta 0.5** | 3/12 | 5/12 | +0.065 |

- Failed cells at eta 1 (test NLL more than 1 above eta 0.05): vanilla 11/60, `grn` 3, `grnx` 6.
- **It is a smaller step by another name.** What `grn` gains over vanilla at the same eta, vanilla
  at half the eta gains and more. Nothing beats vanilla at eta 0.05 on more than 2/12 pairs.
- `grnx` is exactly vanilla on five pairs: with a data start and Damped curvature ρ ≤ 1 from the
  first round. The early instability the method targets is already handled.
- eta 0.3 against 0.05 costs more than 0.02 on 9/12 pairs with early stopping in both.
- **Predictions: R1 ✓, R2 ✓ for `grn` / ✗ for `grnx`, R3 ✗ / ✓, R4 ✗, R5 ✓.**
- **By the rule:** stays out of the prelude and is documented as tested and not recommended.

## 5. How the runs went
- **Three amendments, all recorded in the proposals before the runs they affect:** 028's grid
  (k = 30, 100 added) and selection statistic (mean rank → mean validation NLL), both before the
  unseen run; 030's half-step control arms, added after partial tables for eta 1 and 0.3 were seen.
- **One data repair.** Three streams of sub-second cells appending at once interleaved two CSV
  rows (lines of 40 and 28 fields). The two cells (Gumbel c_airfoil, `base` seed 2 and `ls100`
  seed 0) were re-run and the garbled lines removed after all streams had stopped; a copy of the
  file before the repair is `/tmp/experiments_v2.before_repair.csv`. 75 Gumbel cells that repeat
  021's arms equal its rows bit for bit.
- The machine was shared with another session's jobs (load up to 280 on 12 cores). No stream hit
  the 2-hour limit. `BENCH_SEED_START` was added for resuming a cut stream (one search and its
  refits reproduce the staged binary's rows exactly); it was not needed.

## 6. Decisions for Felix (all answered "yes" on 2026-11-14; outcome in §8)
1. **Gumbel default.** `Gumbel::new` → Damped + `LocationScale` start + start-scale factor 100?
   For: 10/11 pairs within the rule, nine of them by 0.14 to 74 000 nats; the default does not fit
   targets away from zero at all. Against: power_plant fails the rule on the mean (2622 against
   7.43) because two seeds contain one lower-tail test row each; the median seed goes 7.43 → 3.12.
   The rule as written says no. My recommendation: change it and document the tail, as was done
   for Weibull's upper tail in cycle 9; a model that never leaves its start is not a safe default.
2. **NB CRPS coordinates.** Make (log µ, log θ) what `LossFn::Crps` uses for NegativeBinomial
   (`new`, `new_mean_dispersion`, `with_parameterization`)? Better than today's CRPS path on 5/5
   sets, 25/25 cells. Results change for CRPS-trained NB. My recommendation: yes.
3. **Subsampling axes in the tuning advice.** The README now reports that adding `subsample` and
   `colsample_bytree` to the two axes never lost on 24 pairs. Recommending it would need a
   confirmation on data not used for this report (all 24 pairs are spent). Run it, or leave the
   observation as it is?
4. **`GradientRegularization`.** Keep the option in the tree (documented as not recommended) or
   remove it?

## 7. Still open
- Depth: a selection rule that keeps its gains (c_parkinsons −0.44) without its losses (c_gt_co
  +0.17 under the wrong family). 025's lead stands: settle the family before tuning depth.
- Gumbel: k above 100; whether `Location` + wide scale does as well as `LocationScale` + wide.
- CRPS-trained NB: the coverage loss (0.79 on c_seoulbike) and whether per-parameter early
  stopping on log θ recovers it; Poisson CRPS (still 5–7 % behind NLL training).
- eta: 0.15 beat 0.05 where the 2000-round cap binds and lost elsewhere; `grn` at 0.05 on protein
  suggests a step below 0.05 helps there. A round budget tied to eta was not tested.
- From report 015 and untouched here: a selection score robust to tail rows; the unverified audit
  leads (MVN_LoRa, ZIPoisson / ZINB, Mixture, ZAGamma / ZABeta).

## 8. The decisions, applied (proposals 031 and 032; Felix: "yes to all of these")
Decisions 3 and 4 were either/or questions; "yes" was read as the first option of each (confirm
on fresh data; keep the option).

| # | decision | what was done | check |
|---|---|---|---|
| 1 | Gumbel default | `Gumbel::new`: Damped, `LocationScale` start, start-scale factor 100 (`GUMBEL_START_SCALE_FACTOR`). Serde defaults stay Observed / Raw / 1. | E1 ✓: 55/55 cells equal the tested recipe bit for bit |
| 2 | NB CRPS coordinates | `LossFn::Crps` builds the MeanDispersion head in `new`, `new_mean_dispersion`, `with_parameterization`. Opt-out: `new(..).with_parameterization(TotalCountProbs)`. The prototype method `with_mean_dispersion_crps()` is removed. | E2 ✓: 25/25 equal the tested coordinates; E3 ✓: the opt-out equals the old path in 25/25 |
| 3 | subsampling axes | confirmation on 12 fresh pairs (seven UCI datasets fetched 2026-11-14, `prep_confirm_data2.py`) | C1 ✓, C2 ✗, C3 ✗: below |
| 4 | `GradientRegularization` | kept, documented as tested and not recommended, not in the prelude | — |

Tests after the changes: 267 unit and 129 integration tests pass; the LightGBM library build
compiles. Tests that pinned the old defaults now pin the new ones and the old ones under their
opt-outs (list in proposal 031).

**Decision 3: the subsampling axes on fresh data (proposal 032).** Test NLL minus the untuned
default at 20 trials, 12 pairs.

| arm | mean (median) | worse than default | guarded: mean (median) | guarded: worse than default |
|---|---|---|---|---|
| two axes | −0.134 (−0.026) | 2/12 | −0.124 (−0.021) | 1/12 |
| `subs` | −0.150 (−0.043) | 0/12 | −0.139 (−0.043) | 0/12 |

- **C1 ✓.** Guarded `subs` is worse than the untuned default on 0/12 (0/36 over the three sets of
  data). Plain `subs` 0/12 as well.
- **C2 ✗.** `subs` is worse than the two-axis search by more than 0.005 on 2/12, both counts: NB
  n_news +0.011 ± 0.006, Poisson n_metro +0.165 ± 0.255. Predicted: at most one.
- **C3 ✗ on the ratio.** Guarded `subs` beats guarded two axes on 10/12 pairs (33/36 overall), but
  the ratio of the means is 1.12, not 1.5: Poisson n_metro (untuned NLL 39, the wrong family)
  is most of both means. Without it 1.69; medians 2.1.
- New: the guard left one two-axis loss in place (Gaussian n_appliances +0.008), the first in 36
  pairs.
- **By the rule:** the README says the two axes are safe with the validation check and usually,
  not always, better than the two-axis search. It does not say "always add them".

**What Felix should know about the applied changes.**
- Gumbel: the rule on mean test NLL fails on power_plant (two single-row cells); this was the
  stated reason against, and the change was made with it documented in the README.
- NB under the CRPS loss: results change for anyone training NB with `LossFn::Crps`; the model is
  sharper and its 90 % interval covers less than the NLL-trained one's (0.79 against 0.93 on
  c_seoulbike). The README says to train on the CRPS only if the CRPS is all that is scored.
- `initialize: true` together with the new defaults was not benchmarked for either family.

## 9. `max_bin` as a search axis (proposal 033, cycle 29; Felix: "lets look into whether it is a parameter that should be included in search space")
Not included. All 36 pairs (10 development, 14 unseen, 12 fresh), which had never seen a
`max_bin` fit; range, predictions and rule fixed first. Bench: `BENCH_MAX_BIN=k` (untuned),
`BENCH_HPO_SPACE=bin` (mcw, λ, max_bin log-uniform integer 16–1024); `run_c29.sh`, binary
`/tmp/rb_c29`, tags `c29_mb<k>`, `c29_bin_k<k>`; `c29_maxbin_summary.py`.

**Untuned, test NLL at max_bin k minus at 256.**

| max_bin | better by > 0.005 | worse by > 0.005 | within | median | beyond 2 s.e.: better / worse |
|---|---|---|---|---|---|
| 16 | 7 | 25 | 4 | +0.040 | 3 / 19 |
| 64 | 12 | 12 | 12 | −0.001 | 5 / 5 |
| 1024 | 4 | 16 | 16 | +0.003 | 1 / 6 |

The 256 rerun equals the stored untuned rows on 180/180 cells.

**Searched, 20 trials, test NLL minus the untuned default.**

| arm | plain: mean (median) | worse than default | guarded: mean (median) | guarded: worse than default |
|---|---|---|---|---|
| two axes | −0.054 (−0.017) | 6/36 | −0.056 (−0.021) | 1/36 |
| `bin` | −0.055 (−0.017) | 5/36 | −0.041 (−0.016) | 2/36 |

- **Rule: fails (a), (b), (c).** Guarded `bin` worse than the default on 2/36 (Gaussian protein
  +0.123, one seed whose validation NLL improved by 0.005 and whose test NLL went from 2.95 to
  3.69; Poisson c_seoulbike +0.078). Better than guarded two axes on 23/36 (24 needed). Median
  difference −0.003 (−0.005 needed).
- **Predictions:** M0 ✓; M1a ✗ (1024 within ±0.005 on 16/36, predicted ≥ 30); M1b ✓ / ✗ (16 bins
  worse on 25, better on 7 where ≤ 6 was predicted); M2 ✗ / ✓ (plain `bin` beats two axes by
  > 0.005 on 17/36, predicted ≤ 9; worse on 8); M3 ✗ (picks span a factor ≥ 8 on 16/36, predicted
  ≥ 24).
- **The axis is real, the search is too short for it.** Stable picks that agree with the untuned
  table: kin8nm 16–30, naval 234–972, Gamma c_parkinsons 25–32. Gains against two axes there:
  −0.090, −0.058, −0.101, 5/5 seeds each. Cost where the axis is switched off by the data
  (Gaussian n_sgemm, no feature above 16 distinct values): +0.048 ± 0.026.
- **10 against 20 trials:** `bin` led on the guarded mean at 10 trials (−0.059 / −0.041) and
  trailed at 20 (−0.041 / −0.056); the pair count went 22 → 23. Three pairs carry the means.
- Stage 2 (`subsbin` against `subs`) was not run, as the rule says.
- README: "tested, not recommended", the untuned table, and "leave 256; 1024 is not free; 16
  usually costs".

Open after this: whether a larger budget (50 trials) or a two-value choice (64 or 256, by
validation) recovers the kin8nm-type gains without the losses. The validation-only choice among
four values, looked at after the fact, was better than 256 on 14 pairs and worse on 8, so it is
not safe as it stands.

## 10. `max_bin`: 50 trials, and a choice between 64 and 256 (proposal 034, cycle 30; Felix: "lets test this")
540 jobs on the 36 pairs: the two-axis search and `bin` at 50 trials (refits at 20 and 50), and
`bin2` (mcw, λ, max_bin ∈ {64, 256}) at 20. The 20-trial refits of both 50-trial searches equal
the stored rows on 180/180 cells.

| arm (trials) | plain: mean (median) | worse than default | guarded: mean (median) | guarded: worse than default |
|---|---|---|---|---|
| two axes (20) | −0.054 (−0.017) | 6/36 | −0.056 (−0.021) | 1/36 |
| two axes (50) | −0.031 (−0.013) | 7/36 | −0.027 (−0.018) | 3/36 |
| `bin` (20) | −0.055 (−0.017) | 5/36 | −0.041 (−0.016) | 2/36 |
| `bin` (50) | −0.075 (−0.025) | 3/36 | −0.062 (−0.031) | 0/36 |
| `bin2` (20) | −0.077 (−0.019) | 6/36 | −0.072 (−0.020) | 3/36 |

- **`bin` at 50 trials passes the rule** against two axes at 50: (a) 0/36, (b) 30/36, (c) −0.009.
  Against the stored two axes at 20: 0/36, 28/36, −0.007. By set, (b) and (c): development
  10/10, −0.018; unseen 10/14, −0.002; fresh 10/12, −0.016.
- **`bin2` fails:** (a) 3/36 (Gaussian c_n_concrete +0.091, c_supercond +0.057, c_gt_co +0.011),
  (b) 24/36, (c) −0.002. It picks 64 on both kin8nm pairs and 256 on both naval pairs in 5/5
  seeds and keeps the kin8nm gains (−0.062, −0.049), with no dilution on Gaussian n_sgemm
  (−0.002).
- **Two axes do not improve from 20 to 50 trials:** guarded, better on 17/36, median +0.000,
  worse than the default on 3 pairs against 1. The plain mean moves by +0.023 (Poisson c_n_metro
  +0.63, c_seoulbike +0.13).
- **Predictions:** R0 ✓; B1 ✗ / ✓; B2 ✗ (I predicted a fail); B3 ✗ (Gaussian n_sgemm −0.029, not
  within ±0.020, in the arm's favour); B4 ✓; C1 ✓; C2 ✓; C3 ✓; C4 ✓.
- A pass on pairs already used for this parameter is development evidence only → §11.

## 11. `max_bin` at 50 trials on fresh datasets (proposal 038, cycle 30)
Seven UCI datasets fetched for this (Energy Efficiency, Steel Industry Energy, QSAR fish
toxicity, temperature-forecast bias correction, video transcoding time, Beijing PM2.5, Facebook
comment volume; `prep_confirm_data3.py`), 12 pairs, fixed before any fit. 120 fifty-trial
searches, 60 untuned fits.

| arm (trials) | plain: mean (median) | worse than default | guarded: mean (median) | guarded: worse than default |
|---|---|---|---|---|
| two axes (20) | −0.033 (−0.028) | 0/12 | −0.045 (−0.028) | 1/12 |
| two axes (50) | −0.051 (−0.026) | 1/12 | −0.055 (−0.025) | 0/12 |
| `bin` (20) | −0.037 (−0.019) | 2/12 | −0.043 (−0.031) | 0/12 |
| `bin` (50) | −0.061 (−0.034) | 0/12 | −0.062 (−0.044) | 1/12 |

- **The rule at 50 trials fails on (a):** 1/12 (Gamma m_energy +0.029), (b) 10/12, (c) −0.015.
  The miss is one seed on a 768-row dataset (115 validation rows): validation NLL 0.39 against
  0.70, test NLL 1.03 against 0.87; the guard also discarded the two seeds where the refit was
  better on test, so the plain arm is −0.017 there.
- At 20 trials the same arm passes the scaled rule (0/12, 9/12, −0.005): the opposite order to
  the 36 pairs.
- **Predictions:** D1 ✗, D2 ✓, D3 ✓ / ✗ (the size was right, the pass was not), D4 ✓, D5 ✗, D6 ✓
  (median pick below 64 on 5/12, above 256 on 3/12), D7 ✓.
- Over the 48 pairs at 50 trials: guarded `bin` better than guarded two axes on 40, one guard
  miss; the two-axis search has three.
- **Decision (by the rule):** not recommended; README says what was found, including the 40 of
  48 and the one miss. For Felix: whether a rule that allows no loss at all should keep this
  parameter out; changing the rule would need another fresh batch.

## 12. Eight untested XGBoost parameters, one axis at a time (proposal 035, cycle 31; Felix: "queue these to test after we are done with the current running test")
Each parameter as a third axis next to mcw and λ, 20 trials, the 36 pairs, 033's rule;
1440 searches and 720 untuned fits. Two drive disconnects and three 2-hour task limits
interrupted the waves; killed jobs were rerun and every finished job checked.

| arm | plain mean | guarded mean (median) | (a) | (b) | (c) | plain better / worse than two axes | rule |
|---|---|---|---|---|---|---|---|
| two axes (stored) | −0.054 | −0.056 (−0.021) | 1 | | | | |
| `eta` 0.02–0.3 | −0.062 | −0.039 (−0.022) | 2 | 20 | −0.0003 | 17 / 13 | fail |
| `gamma` 10⁻³–10² | −0.043 | −0.031 (−0.013) | 2 | 14 | +0.0036 | 7 / 14 | fail |
| `alpha` 10⁻³–10² | −0.072 | −0.073 (−0.026) | 3 | 28 | −0.0045 | 20 / 5 | fail |
| `ctree` 0.5–1 | −0.042 | −0.047 (−0.026) | 0 | 22 | −0.0044 | 18 / 7 | fail |
| `clevel` 0.5–1 | −0.061 | −0.060 (−0.016) | 3 | 22 | −0.0032 | 19 / 5 | fail |
| `cnode` 0.5–1 | −0.078 | −0.075 (−0.029) | 1 | 28 | −0.0080 | 21 / 3 | fail |
| `mds` 0.1–10 | +0.034 | −0.034 (−0.014) | 2 | 15 | +0.0007 | 6 / 17 | fail |
| `leaf` (lossguide, max_leaves 4–64) | −0.229 | −0.208 (−0.057) | 0 | 34 | −0.0381 | 30 / 0 | **pass** |

- **Untuned arms:** `u_repro` identical on 180/180; round cap 6000 better on 6, worse on 0 (the
  default is at the cap on 11 pairs); eta 0.15 worse on 22, better on 9; lossguide with 16 leaves
  better on 27, worse on 5 (Gaussian c_n_concrete +0.081, c_n_appliances +0.042, Gamma
  c_n_concrete +0.036, NegativeBinomial c_n_news +0.033, Gaussian c_gt_co +0.006).
- **`leaf`** by set, (b) and (c): development 10/10, −0.060; unseen 12/14, −0.039; fresh 12/12,
  −0.030. Plain refit worse than the default on 0/36. Against the stored depth search (24
  pairs): better on 20, median −0.029. Against the subsampling search: better on 26/36, median
  −0.019. Picked max_leaves quartiles 10 / 16 / 28.
- **Near misses:** `cnode` fails on (a) alone (Gaussian c_parkinsons +0.009); `alpha` on (a)
  (three pairs) with (c) at −0.0045. At 10 trials `alpha` meets all three conditions; the rule is
  applied at 20.
- **Predictions:** S0 ✓, E0 ✓, E1 ✓, E2 ✓ except Gamma naval's median eta (0.097, predicted above
  0.1), R1 ✓ for gamma, ✗ for alpha's (b), K1 ✓ / ✓ / ✗ (cnode meets (c)), D1 ✓, L0 ✓, L1 ✓ / ✗
  (I predicted a guard miss for `leaf`; there was none).
- Seven fails are final at 20 trials. `leaf` → §13.

## 13. The leaf-wise axis on 12 pairs it had not seen (proposal 039, cycle 31)
The `c_m_*` pairs of §11: fetched for `max_bin`, never fitted leaf-wise, references stored. 60
searches, 120 untuned fits.

| arm (20 trials) | plain: mean (median) | worse than default | guarded: mean (median) | guarded: worse than default |
|---|---|---|---|---|
| two axes (stored) | −0.033 (−0.028) | 0/12 | −0.045 (−0.028) | 1/12 |
| `leaf` | −0.225 (−0.078) | 0/12 | −0.232 (−0.130) | 0/12 |
| `u_lg16` (untuned) | −0.157 (−0.084) | 2/12 | | |

- **The rule passes:** (a) 0/12, (b) 12/12, (c) −0.068. On three pairs (both m_energy, 768 rows;
  NegativeBinomial m_fbcomments) the paired difference is inside its standard error.
- `u_lg16` worse on Gaussian m_fishtox (+0.080) and NegativeBinomial m_fbcomments (+0.007); the
  search picked 4 leaves on both.
- **Predictions:** F0–F6 all ✓ (written after seeing §12, so a confirmation on other data, not a
  blind forecast).
- Over 48 pairs: `leaf` worse than the default on 0 (plain and guarded), better than guarded two
  axes on 46. `u_lg16` better than the default on 37, worse on 7.
- **Decision (by the rule):** the README recommends the axis for the XGBoost backend, with its
  limits (20 trials, not combined with subsampling, batch not fetched for this). For Felix:
  whether `hyper_opt` should offer the space ready-made, and whether a leaf-wise default is worth
  a trial (16 leaves fails the default rule on 7/48).

## 14. LightGBM's own parameters as search axes (proposal 036, cycle 32; item 6 of Felix's queue)
LightGBM build of the bench, the 36 pairs, 20 trials; 900 searches and 180 untuned fits.

| arm | plain mean | guarded mean (median) | (a) | (b) | (c) | plain better / worse than `l_2d` | rule |
|---|---|---|---|---|---|---|---|
| `l_2d` (mcw, λ) | +0.178 | −0.021 (−0.006) | 3 | | | | |
| `lleaves` num_leaves 4–64 | +0.145 | −0.135 (−0.031) | 1 | 25 | −0.0238 | 28 / 3 | fail |
| `lmdl` min_data_in_leaf 1–100 | +0.179 | −0.020 (−0.009) | 2 | 19 | −0.0002 | 11 / 13 | fail |
| `lff` feature_fraction 0.5–1 | +0.174 | −0.022 (−0.013) | 2 | 22 | −0.0027 | 17 / 7 | fail |
| `lbag` bagging_fraction 0.5–1 | +0.184 | −0.036 (−0.013) | 4 | 24 | −0.0065 | 21 / 9 | fail |

- **Untuned LightGBM against untuned XGBoost:** better on 21/36, worse on 8, median −0.019.
- **Gaussian naval breaks under any λ ≥ 0.001 on LightGBM:** every tuned refit (50 cells) stops after
  77–99 rounds at +0.46 to +1.42 against −6.1 untuned; probe: λ = 0 → −6.11, λ = 0.001 → +0.95 (92
  rounds), λ = 1 → +1.46 (0 rounds); mcw 0.001 and 16 are harmless. The search range has no 0, so
  CV cannot avoid it; the guard discards every such refit. This pair is the whole positive plain
  mean (`l_2d` without it: −0.015). Mechanism not traced.
- `lleaves` fails on one pair (Gaussian power_plant +0.0058).
- **Predictions:** G0 ✗ / ✓, G1 ✓, G2 ✗ / ✓ (11 pairs, 12 predicted), G3 ✗ (the sampling axes
  have 2 and 4 guard misses).
- **Decision:** all four fail at 20 trials; README has the table and the λ warning. For Felix:
  a λ range that includes 0 for LightGBM; whether `lleaves` deserves a rerun after that.

## 15. Library-level settings chosen per dataset (proposal 037, cycle 33; item 7 of Felix's queue)
180 four-axis searches and 900 untuned fits on the 36 pairs (XGBoost).

- **P (`pes`: mcw, λ, PES window, PES anneal; 20 trials):** (a) 1 (Gaussian c_n_appliances
  +0.028), (b) 20, (c) −0.0003 → fail. Guarded mean −0.036 against −0.056 for two axes. Picked
  windows 0 / 5 / 10 / 20 / 40 / 80: 7 / 43 / 36 / 31 / 40 / 23 of 180.
- **V (validation choice among untuned candidates), better / worse / within:** `V-mode`
  12 / 9 / 15; `V-start` 16 / 1 / 19 (Gamma c_n_wine +0.0052; mean −0.010); `V-pes` 0 / 1 / 35;
  `V-all` 20 / 5 / 11. All fail "0 worse and at least 9 better".
- **Untuned alternatives against the default, better / worse:** Observed 1 / 28; Fisher 7 / 15
  with Gaussian power_plant non-finite and blow-ups up to 1.9·10¹¹; PES off 0 / 28; L-BFGS start
  11 / 13; other location start 3 / 13 of 21 (Poisson c_seoulbike 3.7·10¹²).
- **F (family by validation, stored rows):** right in 65/75 cells; mean regret 0.026 against
  0.024 for always Gamma / NegativeBinomial and 3.28 for always Gaussian / Poisson; largest
  regret 0.294 (c_gt_co).
- **Predictions:** P1 ✓, P2 ✓, V1 ✓ / ✗ (Observed and PES off are better on 1 and 0 pairs, I
  predicted at least 3), V2 ✓, V3 ✓, F1 ✓.
- **Decision:** nothing passes; README records it. For Felix: whether the start-value choice
  (one pair over the line by 0.0002) is worth a fresh-data test.

## 16. The queue of 2026-11-14, in one table
| item | proposal | outcome at 20 trials |
|---|---|---|
| 1 eta with the round cap | 035 | fail (2 / 20 / −0.0003); cap 6000 helps 6 pairs, hurts none |
| 2 gamma, alpha | 035 | fail (2 / 14 / +0.004; 3 / 28 / −0.0045) |
| 3 colsample_bylevel, bynode (and bytree alone) | 035 | fail (3 / 22 / −0.003; 1 / 28 / −0.008; 0 / 22 / −0.004) |
| 4 max_delta_step | 035 | fail (2 / 15 / +0.001) |
| 5 grow_policy with max_leaves | 035, 039 | **pass** (0 / 34 / −0.038), confirmed on 12 other pairs (0 / 12 / −0.068) |
| 6 LightGBM parameters | 036 | fail ×4; num_leaves 1 / 25 / −0.024; λ ≥ 0.001 breaks Gaussian naval |
| 7 library-level settings | 037 | fail ×5; start-value choice 16 better, 1 worse |

Open decisions for Felix: (1) a ready-made leaf-wise space in `hyper_opt`; (2) a trial of a
leaf-wise default; (3) the round cap; (4) whether one guard miss in 48 should keep `max_bin`
out (§11); (5) a λ range that includes 0 for LightGBM; (6) a rerun of LightGBM `num_leaves`
after that; (7) a fresh-data test of the start-value choice.

### Decisions, 2026-11-15
Felix answered (4) himself and asked for my recommendation on the rest.

| # | question | decision | what was done |
|---|---|---|---|
| 1 | ready-made leaf-wise space | yes | `hyper_opt::leaf_wise_search_space()` (mcw and λ log 1e-3–1e3, lossguide, max_depth 0, max_leaves 4–64 log integer), exported in the prelude, unit-tested |
| 2 | trial of a leaf-wise default | not now | Fixed 16 leaves loses on 7/48. The searches picked 4–8 leaves on some pairs and 43–61 on others, so a smaller fixed count is expected to lose on the second group. The search space gets the gain with 0/48 losses. Stays open as a possible proposal |
| 3 | round cap | advice only | The library has no cap of its own (`num_boost_round` is the caller's; 2000 was the bench's). README: if `best_iteration` is at the cap, raise it |
| 4 | does one guard miss in 48 keep `max_bin` out | **no (Felix)** | README recommends `max_bin` 16–1024 at ≥ 50 trials with the guard; `hyper_opt::max_bin_search_space()` added. This departs from 038's pre-registered rule, under which the arm failed; the README says so and names the pair |
| 5 | λ = 0 reachable on LightGBM | advice only | The library has no built-in space. README: give λ as a choice list containing 0, or leave it out; untested, the guard stays mandatory. The cause on Gaussian naval is still untraced |
| 6 | rerun LightGBM `num_leaves` | not now | Needs a new proposal and about a day of compute on a backend-specific axis; worth it only if LightGBM becomes the working backend |
| 7 | fresh-data test of the start-value choice | not now | Mean gain 0.010 for three fits instead of one; small against the leaf-wise axis (median 0.06–0.14 by family) |

Leaf-wise search by family over the 48 pairs (guarded, median against untuned; two axes in
brackets; pairs better than two axes): Gaussian −0.081 (−0.021), 19/20; Poisson −0.136 (−0.036),
6/6; NegativeBinomial −0.057 (−0.013), 6/7; Gamma −0.078 (−0.025), 15/15. No family has a loss
against the untuned default.
