# Proposal 018 — choosing the family: marginal fit (`dist_select`) against held-out conditional score

Status: **run** 2026-11-13 (F1 ✓, F2 ✓, F3 ✓). **Pre-registered** 2026-11-13, before the run. Skill: `automl-hyperparameter-research`.

## Question
`dist_select` ranks candidate families by the NLL of an *unconditional* fit to the training labels
(`src/dist_select.rs`: it calls `calculate_start_values` and sorts). The model that is then trained is
*conditional*. The two can disagree: a Gaussian whose variance depends on x has a heavy-tailed
marginal, so the marginal fit prefers StudentT (or a skewed family) although the conditional
distribution is Gaussian. The marginal fit also has no held-out data, so a family with more
parameters is never penalised.

Does the marginal ranking pick the family that scores best out of sample, and if not, how much does
it cost against picking by a held-out conditional score?

## Protocol
Per (dataset, seed), same splits as every bench cell (5 seeds), the current default recipe
(Damped, PES Auto, eta 0.05, depth 4, 2000 rounds / patience 50), nothing tuned:
- **Continuous, positive targets** (kin8nm, power_plant, naval): Gaussian, StudentT, Gamma (mean/shape),
  Weibull.
- **Continuous, any sign** (protein, gaussian_sim, studentt_sim, synth_t3): Gaussian, StudentT.
- **Counts** (bike_cnt, bike_casual): Poisson (Exp link), NB (mean/dispersion).

Three selectors per cell:
1. `marginal` — lowest unconditional NLL on the training labels (`BENCH_DIST_SELECT=1`, 100 iterations);
2. `valid` — lowest NLL of the trained model on the validation split (the early-stopping split, so
   optimistic for every family alike);
3. `oracle` — lowest test NLL (the reference, not a usable rule).

Outcome: regret = test NLL of the selected family − test NLL of the oracle family, per cell.
Rows tagged `c14f`; DSEL lines in `/tmp/c14f_dsel.log`; summary `c14f_family_selection.py`.

## Predictions
- **F1.** `marginal` picks a family other than the oracle's on at least 3 of the 9 datasets
  (majority over seeds).
- **F2.** Mean regret of `valid` is smaller than mean regret of `marginal` on every dataset where they
  pick differently, and `valid` matches the oracle in ≥ 80 % of cells.
- **F3.** On gaussian_sim (a heteroscedastic Gaussian DGP) `marginal` picks StudentT.

## Decision rule
No library change in this cycle. If F1 and F2 hold, document that `dist_select` is a screen for the
*support and rough shape* of the target, and that the family should be chosen by held-out score
(`HyperOptConfig::trial_dist_fn` makes it a search axis). A CV-scored `dist_select` variant would be
proposed to Felix separately.

## Cost
≈ 130 fits of a few seconds each plus the marginal fits; one stream, under 20 minutes.

## Result (2026-11-13, report 015 §2)
- The marginal ranking picks the family with the best test NLL in 22/45 cells; validation NLL of
  fitted models in 42/45. Mean regret 0.0545 against 0.0005.
- Marginal picks that lose: protein (Gaussian; regret 0.211), kin8nm (Weibull; 0.137), power_plant
  (Gamma; 0.104), naval (0.034), gaussian_sim (StudentT; 0.006).
- F1 ✓, F2 ✓, F3 ✓.
- Side finding: `dist_select` scores StudentT at the capped training start for df.
- **Decision.** README note: pick the family by validation NLL of fitted models.

**Follow-up 2026-11-13 (proposal 026):** `dist_select` now ranks StudentT at its free fit
(`Distribution::marginal_fit`). The picks did not change (22/45, regret 0.0543): the side finding
about the capped start df was an artefact, not the cause.
