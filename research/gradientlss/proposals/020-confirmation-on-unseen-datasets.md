# Proposal 020 — do the cycle 1–13 gains hold on datasets the research never saw?

Status: **run** 2026-11-13 (C1 ✓, C2 ✗ for StudentT by the letter, C3 ✓). **Pre-registered** 2026-11-13, before the run (one smoke cell per backend plumbing check was
seen: `new` Gaussian on c_airfoil and c_parkinsons, seed 0; no `old` cell). Skills: both.

## Question
Every default change of cycles 1–13 was decided on test NLL over the same sources: four UCI
regression sets (kin8nm, power_plant, naval, protein), UCI bike sharing, and the crate's synthetic
sets, with seeds 0–9. Thirteen rounds of decisions on one test bed can fit the test bed. Nothing in
`research/gradientlss` checks the accumulated recipe on data that took no part in any decision.

## Data (fetched 2026-11-13 from the UCI repository; `prep_confirm_data.py`)
| name | rows | features | target |
|---|---|---|---|
| c_airfoil | 1 503 | 5 | sound pressure level, dB (103–141) |
| c_supercond | 12 000 of 21 263 | 81 | critical temperature, K (> 0) |
| c_parkinsons | 5 875 | 19 | total UPDRS (7–55) |
| c_gt_nox | 12 000 of 36 733 | 9 | NOx emission (27–120) |
| c_gt_co | 12 000 of 36 733 | 9 | CO emission (0.007–38, skewed) |
| c_abalone | 4 177 | 10 | rings (count, 1–29) |
| c_seoulbike | 8 465 | 13 | bikes rented per hour (count, 2–3 556) |

Random 65/15/20 splits by seed, as for every non-fixed-test set. (Parkinsons has repeated subjects
and the bike and turbine sets are time-ordered, so absolute scores are optimistic; the comparison
is paired within a split.)

## Protocol
The `old` and `new` recipes of proposal 019, XGBoost, 5 seeds, eta 0.05, depth 4, 2000 rounds /
patience 50, nothing tuned:
- Gaussian, StudentT, Gamma on the five continuous sets (all targets are positive);
- NB, Poisson on the two count sets.
19 (family, dataset) pairs. Tags `c17_old`, `c17_new`; script `run_c17.sh`.
The development-set reference is `c16_xgb_old` / `c16_xgb_new` (same binary source, same recipes).

## Predictions
- **C1.** `new` − `old` ≤ +0.005 test NLL on every pair (non-finite rows count as a loss for the arm
  that has them).
- **C2.** Per family, the mean gain over the unseen sets is at least half the mean gain over the
  development sets.
- **C3.** `new` has no non-finite test rows on any pair.

## Decision rule
No library change. A failed C1 pair is reported with its seeds and becomes a cycle 17 question. If
C2 fails for a family, that family's recipe is flagged in the README as measured on the development
sets only.

## Cost
190 fits, one single-thread stream; under an hour on a loaded machine.

## Result (2026-11-13, report 015 §4)
- 0/19 unseen pairs break the rule, no non-finite row. Family means: Gamma −2.19, Gaussian −0.57,
  NB −0.26, Poisson −14.4, StudentT −6.6. Only non-improving pair: Poisson c_abalone +0.0017 ± 0.0011.
- C1 ✓, C3 ✓. C2 ✗ by the letter for StudentT: the development mean (−42.9) is dominated by one pair
  (power_plant −299); with medians it would pass, which is a post hoc reading.
- **Decision.** No change. The seven sets stay available as `c_<name>` for later confirmations.
