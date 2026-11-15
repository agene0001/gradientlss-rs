# Proposal 028 — Gumbel: the blow-ups are single lower-tail rows; a wide data start for the stall

Status: **run** 2026-11-14 (G1 ✗ on its 0.05 threshold, G2 ✓, G3 ✓, G4 ✓, G5 ✓; amended twice before the unseen run). **Pre-registered** 2026-11-14, after the diagnosis below (existing rows of 021 and one
debug pass over two of 021's arms, no new arm) and before any run of the new start. Skill:
`optimization-algorithm-research`. Follows 021, where Gumbel was the one family left unsolved.

## What 021 left
Gumbel keeps observed curvature and the raw 0.5 start. With Damped curvature and a data start it
"still blows up on single seeds (power_plant 557 and 20 802; naval seed 1 21.0)", and on unseen
c_gt_co the Damped + Raw recipe was +0.016 ± 0.008 worse.

## Diagnosis (done before this pre-registration)
Per-row test NLL of `c15_d_loc` and `c15_d_ls` re-run with `BENCH_DEBUG_WORST=1` (same binary
settings; the cell means reproduce 021's):

| cell | mean test NLL | without its single worst row | the worst row |
|---|---|---|---|
| power_plant, Location start, seed 3 | 20 802 | 11.1 | y 438.6, predicted location 480.0, scale 2.37 (z = −17.5): 99.9 % of the total |
| power_plant, Location start, seed 2 | 15.8 | 8.45 | y 434.1, location 460.8, scale 2.79 (z = −9.6) |
| power_plant, LocationScale start, seed 3 | 557 | 5.57 | the same row as seed 3 above (z = −13.9) |
| naval, LocationScale start, seed 1 | 21.0 | −5.04 | y 0.980, location 0.98101, scale 9.1e-5 (z = −11.0) |

- Every blow-up is one test row (or a handful) **below** the predicted location by 10–17 scales.
  The Gumbel NLL is z + e^{−z} + ln s: the lower tail costs e^{−z}, so one row at z = −17 costs
  4·10⁷. The same outlier in a Gaussian costs z²/2 ≈ 50–150 for the row. CRPS is unaffected
  (power_plant 2.52–2.55 in the blown-up arms). This is the family's density meeting a real
  outlier in the test split, not a training divergence: no in-sample change can see that row.
- Separately, the **stall** of 021 is real and is a start problem: from the raw start the scale
  grows (log-scale step ≈ 1 per unit of z) while the location creeps, ending at scale 367 on
  power_plant (NLL 7.27, CRPS 163) and 92 on c_airfoil (5.88; Gaussian 1.97). The Location start
  with the raw scale e^0.5 = 1.65 begins power_plant at a validation NLL of 5·10⁴–2·10⁵ (half the
  rows are in the e^{−z} tail) and reaches a median of 4.30; the LocationScale start reaches 3.17
  there but is worse on naval (median −4.85 against −5.49) and kin8nm (−0.273 against −0.301),
  where the raw scale happened to be 140× and 8× too wide: starting wide is cheap (ln k) and
  starting narrow is not (e^{−z}).

## Prototype
`Gumbel::with_start_scale_factor(k)`: with `LocationScaleStart::LocationScale`, the start scale is
k × the moment-matched one (sd·√6/π); the location stays ȳ − γ·sd·√6/π. Damped curvature.
Opt-in; `Gumbel::new` is unchanged. Bench `BENCH_LS_SCALE_K`.

## Protocol
Arms: `base` (Observed, Raw: today's default), `ls1`, `ls2`, `ls4`, `ls10` (Damped, LocationScale,
k = 1, 2, 4, 10). Tags `c26_gb_<arm>`, binary `/tmp/rb_c26`, 5 seeds, eta 0.05, depth 4, 2000
rounds, patience 50, PES auto.
- Development: power_plant, kin8nm, naval, protein, gaussian_sim, and a new simulation
  `synth_gumbel` (the family is right: μ = 120 + 25 sin(πx₁) + 10x₂, ln s = 1 + 0.8x₃, 8000 rows,
  oracle known). k is chosen on these by mean validation-NLL rank, never by test.
- Unseen: c_airfoil, c_parkinsons, c_gt_nox, c_gt_co, c_supercond, `base` and the chosen k only,
  run once. **This is these sets' second look for Gumbel** (the first was 021's Damped + Raw).
- Read-outs per cell: mean test NLL (the rule's metric), the mean without the model's worst 0.5 %
  of rows ("trimmed"), CRPS, and the worst row's z. A **blow-up cell** is one whose mean test NLL
  drops by more than 0.5 when its single worst row is removed.

## Predictions
- **G1 (right family).** synth_gumbel: `base` is more than 1 nat above the oracle NLL; the chosen
  arm is within 0.05 of it in 5/5 seeds and has no blow-up cell.
- **G2 (the stall goes).** The chosen arm's CRPS is at most `base`'s + 1 % and its trimmed NLL at
  most `base`'s + 0.005 on all 11 pairs.
- **G3 (the tail stays).** The chosen arm still has at least one blow-up cell among its 55, and
  fails the +0.005 rule on mean test NLL on at least one pair.
- **G4 (cause).** In every blow-up cell of any arm the worst row has z < −5.
- **G5 (wide beats matched).** The chosen k is > 1, and its validation NLL beats `ls1` by more than
  0.05 on naval and kin8nm without losing more than 0.05 on power_plant.

## Decision rule
- The +0.005 rule on mean test NLL holds on all 11 pairs: propose the recipe as Gumbel's default.
- G2 holds and the mean-NLL rule fails only through blow-up cells: no default change by the rule;
  put the decision to Felix with both read-outs (change the default and document that Gumbel's
  mean NLL is fragile to lower-tail outliers, or keep it).
- G2 fails: Gumbel stays as it is; the README gets the diagnosis.

## Not in scope
A scale floor (the analogue of Weibull's shape cap: it widens every prediction and was a loss in
cycle 9), a robust training loss (it would not be the NLL), Gumbel-min.

## Amendment (2026-11-14, after the development arms, before any unseen run)
The grid k ∈ {1, 2, 4, 10} was too short. Validation NLL was still falling at its upper edge on
the two sets where the start matters (naval: −5.126, −5.176, −5.258, −5.331; kin8nm: −0.269,
−0.268, −0.290, −0.313) and is flat for k ≥ 2 on the other four. The rule as written picks k = 4
(mean validation rank 1.67; k = 10 2.17), and k = 4 fails G2 on naval (trimmed NLL +0.033,
CRPS +30 % against `base`). 021's Damped + Raw arm, whose raw scale happens to be about 140× the
matched one on naval, is better there (−5.41) than any arm of this grid.

**Added arms:** `ls30`, `ls100` on the six development sets. The selection rule is unchanged
(mean validation-NLL rank over all six arms, development sets only); the unseen sets are still
untouched. G1–G5 and the decision rule stand and are scored for whichever k the rule picks.

## Second amendment (2026-11-14, after `ls30` / `ls100` on the development sets, before any unseen run)
Mean validation-NLL **rank** over the six arms picks k = 4 (2.33) ahead of k = 100 (2.67). The
rank counts a 0.001 lead on four sets where every k ≥ 2 is within 0.01 (gaussian_sim, power_plant,
protein, synth_gumbel) the same as a 0.21 lead on naval and 0.075 on kin8nm. Mean validation NLL
over the six sets: k = 1 2.057, 2 2.040, 4 2.006, 10 1.977, 30 1.962, **100 1.944**.

I am changing the selection statistic from mean rank to mean validation NLL, on development
validation scores only and before any unseen run. It is a change made after seeing which arm it
favours, so both picks go to the unseen sets: **`ls100` is the candidate** the predictions and
the decision rule are scored on; `ls4`, the pick of the rule as first written, is reported next
to it and is not a candidate. The trend is not exhausted at k = 100 (naval −5.331, −5.386, −5.473
at k = 10, 30, 100); the grid stops here.

Why wider keeps helping, from the gradients: at a scale far above the residuals every z is near 0,
e^{−z} ≈ 1, and the Damped location step is η·(y − μ): least-squares boosting. The log-scale
falls by about η per round, so k buys about ln(k)/η rounds (92 at k = 100) of location fitting
before the e^{−z} tail starts to weight single rows.

## Result (2026-11-14, report 016 §2)
385 cells (7 arms × 6 development sets, 3 arms × 5 unseen sets, 5 seeds), no non-finite row. The
75 cells that repeat 021's arms (`base`, `ls1`) equal its rows bit for bit. Two rows of the unseen
run were interleaved in the CSV by concurrent writers; c_airfoil was re-run for `base` and `ls100`.

Candidate `ls100` (Damped, LocationScale, start scale 100 × the matched one) against `base`
(today's default). Mean over 5 seeds; "trimmed" drops each model's worst 0.5 % of test rows.

| dataset | Gaussian | base: NLL | ls100: NLL | NLL, median seed | trimmed NLL | CRPS |
|---|---|---|---|---|---|---|
| synth_gumbel (oracle 2.597) | — | 6.097 | 2.683 | 6.097 → 2.698 | 6.095 → 2.646 | 50.98 → 2.22 |
| gaussian_sim | 2.101 | 2.356 | 2.217 | 2.325 → 2.219 | 2.230 → 2.154 | 1.466 → 1.410 |
| kin8nm | −0.656 | −0.159 | −0.367 | −0.169 → −0.362 | −0.233 → −0.415 | 0.103 → 0.088 |
| naval | −6.05 | −4.59 | −5.42 | −4.73 → −5.43 | −5.32 → −5.55 | 0.0008 → 0.0007 |
| power_plant | 2.68 | 7.43 | **2622** | 7.43 → 3.12 | 7.43 → 2.92 | 190.5 → 2.51 |
| protein | 2.859 | 73 681 | 2.934 | 2.864 → 2.832 | 2.736 → 2.716 | 2.642 → 2.559 |
| c_airfoil | 1.97 | 6.13 | 2.51 | 6.13 → 2.56 | 6.13 → 2.40 | 52.3 → 1.70 |
| c_gt_co | 1.68 | 1.120 | 1.122 (+0.002 ± 0.005, 1/5) | 1.123 → 1.133 | 1.040 → 1.034 | 0.473 → 0.468 |
| c_gt_nox | 2.89 | 7.25 | 3.17 | 3.29 → 3.17 | 3.12 → 3.06 | 3.35 → 3.19 |
| c_parkinsons | 2.18 | 14.77 | 2.91 | 14.88 → 2.90 | 14.70 → 2.78 | 25.9 → 3.08 |
| c_supercond | 3.37 | 49 992 | 4.29 | 4.45 → 3.63 | 841 → 3.59 | 12.6 → 7.33 |

- **G1 ✗ on its threshold.** synth_gumbel: `base` is 3.50 above the oracle (it never leaves the
  raw start: scale 115). `ls100` is 0.087 above it (0.068–0.097 by seed), not within 0.05; no
  blow-up cell. Every k ≥ 4 is at 0.087; k = 1 is at 0.112.
- **G2 ✓.** Trimmed NLL ≤ `base` + 0.005 on 11/11 pairs and CRPS ≤ `base` + 1 % on 11/11 (the
  smallest gains: c_gt_co −0.006 and −1.0 %).
- **G3 ✓.** 4 of `ls100`'s 55 cells are blow-up cells (power_plant seeds 2 and 3, protein seed 4,
  c_supercond seed 3; `base` has 6), and the +0.005 rule on mean NLL fails on 1/11: power_plant,
  where the two cells are 8.4 and 13 091 and the other three 3.07–3.13. `base` cannot blow up
  there: its scale is 430.
- **G4 ✓.** All 27 blow-up cells of all arms have their worst row at z < −5 (−5.6 to −20.6): the
  lower tail. power_plant seed 3 is the same row in every arm (y 438.6, predicted 480).
- **G5 ✓.** Validation NLL, k = 100 against k = 1: naval −5.473 / −5.126, kin8nm −0.365 / −0.269,
  power_plant 3.043 / 3.061.
- **`ls4`, the pick of the rule as first written (reported, not a candidate):** the mean-NLL rule
  fails on 2/11 (power_plant; c_gt_co +0.008 ± 0.009), trimmed NLL on 1/11 (naval +0.033), CRPS on
  1/11 (naval +30 %), and it has 6 blow-up cells. The amended statistic picked the better arm on
  the unseen sets as well (c_airfoil 2.51 against 2.73, c_parkinsons 2.91 against 3.43,
  c_supercond 4.29 against 48.4); this is one comparison, on sets used once before for Gumbel.

**What it does not fix.** Gumbel stays 0.3–0.6 behind Gaussian on the symmetric targets, as it
should (wrong family), and its mean NLL stays at the mercy of single lower-tail test rows. No
in-sample change can remove that: it is the density.

**Decision (by the rule).** G2 holds and the mean-NLL rule fails only through blow-up cells, so the
rule makes no default change and the decision goes to Felix: `Gumbel::new` → Damped +
LocationScale with start-scale factor 100 (10/11 pairs within the rule, 9 of them by 0.14 to
74 000 nats; power_plant fails on the mean through two single-row cells while its median seed
goes 7.43 → 3.12 and its CRPS 190 → 2.5), or keep the stalled default. Until then it is opt-in:
`Gumbel::default().with_hessian_mode(HessianMode::Damped).with_start(LocationScaleStart::LocationScale).with_start_scale_factor(100.0)`.
