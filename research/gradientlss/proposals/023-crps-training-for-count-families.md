# Proposal 023 — does training on the CRPS help? Poisson and NegativeBinomial

Status: **run** 2026-11-13 (K1–K4 all hold; two defects reported). **Pre-registered** 2026-11-13, after reading the code and one smoke cell per family
(c_abalone, seed 0), before any other run. Skill: `optimization-algorithm-research`.

## Question
`LossFn::Crps` trains on the CRPS instead of the NLL. Only Poisson and NegativeBinomial have a closed
form for it (every other family goes through finite differences of a 30-draw estimate). Does a
CRPS-trained model give a better held-out CRPS than the NLL-trained default, and at what NLL cost?

## What the code does (read 2026-11-13)
- Gradients: closed-form dCRPS/dθ through the link. Hessian: the diagonal metric
  M_jj = E_{y∼model}[(dCRPS/dη_j)²], floored (NB: 0.05). It is a metric, not a curvature.
- **The sums are truncated.** Poisson: k ≤ min(max(y, λ) + 4√λ, **100**)
  (`src/distributions/poisson.rs`, `analytical_crps_gradients_impl`). NB: table cap **220**.
  A row with y or λ above the cap gets a gradient from a CDF that has not reached 1.
- **NB falls back to (total count, probs).** `new_mean_dispersion` with CRPS prints "MeanDispersion
  supports NLL only" and uses the coordinates cycle 4 found collinear; none of the later NB work
  (orthogonal coordinates, Damped, mean start) reaches the CRPS path.

Smoke cells (c_abalone, counts 1–29, seed 0; NLL-trained NB: test CRPS 1.213, 212 rounds):
CRPS-trained Poisson 1.209 at 222 rounds; CRPS-trained NB 1.622 at the 2000-round cap; NB in its
native coordinates 1.722 at the cap.

## Protocol
XGBoost, 5 seeds, eta 0.05, depth 4, 2000 rounds / patience 50, PES Auto, Damped, the default starts.
Arms `c20_nll` (today's defaults) and `c20_crps` (`BENCH_LOSS=crps`), same binary (`/tmp/rb_c20`).
- Poisson (Exp link): synth_pois, c_abalone (counts < 100); bike_casual (ȳ ≈ 36, max 367);
  bike_cnt (ȳ ≈ 190), c_seoulbike (ȳ ≈ 700).
- NB (mean, dispersion): synth_nb_lo, c_abalone, bike_casual, bike_cnt, c_seoulbike.
Scores on test: CRPS (closed-form sums in the bench, not truncated), NLL, 90 % coverage, RMSE.
Early stopping monitors each arm's own loss; `valid_nll` in the CSV is that loss.

## Predictions
- **K1 (Poisson, small counts).** On synth_pois and c_abalone the CRPS-trained Poisson's test CRPS is
  within 1 % of the NLL-trained one's.
- **K2 (Poisson, truncation).** On bike_cnt and c_seoulbike the CRPS-trained Poisson's test CRPS is
  more than 5 % worse than the NLL-trained one's.
- **K3 (NB).** The CRPS-trained NB has a worse test CRPS than the NLL-trained NB on all five sets,
  and stops at the round cap (best round ≥ 1990) in at least half of the 25 cells.
- **K4.** On no (family, set) pair does CRPS training beat NLL training on test CRPS by more than 1 %.

## Decision rule
No default change. K2 holds: report the truncation as a defect for counts above 100 (a fix needs
its own proposal: scale the table with λ, or a continuous approximation for large λ). K3 holds:
document that CRPS training of NB is not converged at the bench budget and that NLL training gives
the better CRPS. K4 fails somewhere: report where CRPS training pays.

## Result (2026-11-13, report 015 §7)
50 cells, no non-finite row. `c20_nll` reproduces the earlier default rows bit for bit (37 cells).
Test CRPS, mean over 5 seeds; relative = CRPS-trained / NLL-trained − 1 (positive = worse).

| family | dataset (mean count) | NLL-trained | CRPS-trained | relative | CRPS-trained rounds |
|---|---|---|---|---|---|
| Poisson | c_abalone (10) | 1.2009 | 1.2005 | −0.04 % ± 0.31 (2/5) | 239 |
| Poisson | synth_pois | 1.654 | 1.669 | +0.9 % ± 0.5 (1/5) | 549 |
| Poisson | bike_casual (36) | 6.44 | 9.50 | +47.6 % | 1999 (cap, 5/5) |
| Poisson | bike_cnt (190) | 18.2 | 134.8 | +640 % | **0** (5/5) |
| Poisson | c_seoulbike (700) | 76.5 | 505.6 | +562 % | **0** (5/5) |
| NB | synth_nb_lo | 4.017 | 4.126 | +2.7 % | 269 |
| NB | bike_casual | 6.07 | 6.56 | +8.2 % | 1999 (cap, 5/5) |
| NB | c_abalone | 1.207 | 1.612 | +33.6 % | 1999 (cap, 5/5) |
| NB | bike_cnt | 17.2 | 37.2 | +116 % | 1999 (cap, 5/5) |
| NB | c_seoulbike | 69.8 | 407.7 | +485 % | 1941 (cap, 4/5) |

- **K1 ✓** (−0.04 %, +0.9 %; synth_pois is close to the 1 % line).
- **K2 ✓.** Above the 100-term cap the CRPS-trained Poisson does not train: best round 0 in 10/10
  cells, test NLL 86 and 275 against 6.4 and 19.6, 90 % coverage 0.09 and 0.04. It stays at its
  start values. bike_casual (mean 36, max 367) is not in K2 and is still 48 % worse, at the cap.
- **K3 ✓.** NB worse on 5/5 sets; 19/25 cells at the cap. The two sets above the 220 table cap are
  the worst (+116 %, +485 %; c_seoulbike coverage 0.46).
- **K4 ✓.** 0/10 pairs where CRPS training beats NLL training by more than 1 %.

**Decision (by the rule).** No default change, no code change. Reported as defects for Felix:
(1) the Poisson CRPS path is unusable for counts above about 100 and fails silently; (2) the NB
CRPS path is unusable above about 220 and unconverged below it at 2000 rounds / eta 0.05, in
coordinates cycle 4 found collinear. Until a fix has its own proposal, the README says: train on
the NLL; it gives the better CRPS on every set here.

**Follow-up 2026-11-13 (Felix's decision; proposal 026):** the truncation is fixed and large-count
rows use the expected CRPS curvature. bike_cnt and c_seoulbike: Poisson +6.5 %, +4.7 %; NB
+11.5 %, +1.2 % (were +640 %, +562 %, +116 %, +485 %). The NB coordinate fallback is not addressed.
