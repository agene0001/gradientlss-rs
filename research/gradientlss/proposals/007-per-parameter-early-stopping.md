# Proposal 007 — Per-parameter early stopping (automatic shape/scale freeze)

Status: **pre-registered** (cycle 5, 2026-11-10), after a 1-seed smoke on kin8nm only.

## Measured limitation
The strongest scale-side lever found so far (cycle 3/4) is freezing the scale/shape tree at a
validation-TUNED round k ∈ {∅, 50, 100, 200}: kin8nm −0.157, power_plant −0.116, protein −0.098
(Gaussian); Gamma MeanShape kin8nm −0.150. It needs one full fit per candidate k, and naval needs
k = ∅ (any freeze is catastrophic there), so no fixed k is safe as a default.

## Rule
`GradientLSS::with_param_early_stopping(ParamEarlyStopping { window: W, params })`. After every
round, for each eligible, unfrozen parameter j: evaluate the validation loss with column j of the
validation margin reverted to its value W rounds ago. If that is lower than the current loss
(parameter j's last W trees hurt held-out loss), freeze j from the next round (gradient and Hessian
zeroed). This is the per-parameter analogue of early stopping (patience W) and an automatic
gamboostLSS per-parameter `mstop`. Location stays under the ordinary global early stopping.

Smoke (kin8nm, seed 0): scale-only W=20 → −0.157 (frozen at 171, best_iter 206 → 1449);
all-parameter W=20 → −0.098 (location frozen at 449 too) ⇒ use scale/shape-only eligibility.

## Pre-registered predictions (Damped, η .05/d4, ES 50, 5 paired seeds, eligible = scale/shape only)
- P1: Gaussian, W=20: within 0.03 of the validation-TUNED freeze on kin8nm/power_plant/protein
  (−0.157/−0.116/−0.098 vs no freeze), i.e. ≤ −0.07 on each.
- P2 (safety): naval, gaussian_sim, synth_t3, studentt_sim: |Δ| ≤ 0.01 vs no freeze, or better.
  naval is the critical case (any fixed freeze cost +0.3…+1.4).
- P3: W=50 is no worse than W=20 by > 0.01 anywhere (robust to the window).
- P4: Gamma MeanShape (shape = param 1): kin8nm ≤ −0.10 vs no freeze; naval/synth_gamma/power_plant
  within 0.01 or better.
- P5: NB MeanDispersion (log θ = param 1) on bike_cnt/bike_casual: 90 % coverage rises toward 0.9
  and NLL ties or improves; synth NB DGPs within 0.01.
- Decision rule: if P1–P4 hold, recommend it as the standard recipe (not a default: it needs a
  validation set and changes the training object).
