# Proposal 002 — Damped curvature: h = max(observed, Fisher)

Status: **implemented as `HessianMode::Damped`, benchmarked (tags c1c_*, c1d_*, c1e_*): dominant or tied in every cell, 0 divergences** — recommended experimental setting (cycle 1, 2026-11-09)

## Limitation it addresses
Proposal 001's pure Fisher curvature is constant per row for the scale column
(2 for Gaussian/Exp, 2ν/(ν+3) for t/Exp) and so does not shrink the step when
the gradient is enormous far from the optimum (power_plant: σ → ∞ in one
round). The observed curvature `2z²` does shrink it, but is ≈ 0 near the
optimum (zero-curvature leaves, inflated g/h) and negative for the t
location/df when d² > ν.

## Proposed rule (per row, per parameter, raw space)
```
h_damped = max(h_observed, h_fisher)
```
- Near the optimum (|z| < 1): h_fisher ≥ h_observed → Fisher step, no blow-up.
- Far from it (|z| > 1): h_observed ≥ h_fisher → the observed, self-regulating
  step (Gaussian: Newton step in log σ is (z²−1)/(2z²) → ½, bounded).
- Negative observed values are replaced by the positive Fisher value.
Interpretation: a Levenberg–Marquardt / trust-region damping with the Fisher
information as the floor, cf. TRBoost's claim that the Hessian need not be PD
if the step is constrained, and GRN's λ_k ∝ √‖g‖ (ours is per-row, not
per-round, and needs no backend change).

## Predictions
1. Matches Fisher where Fisher wins (kin8nm, naval, synth_t3) within 1 s.e.
2. Does **not** produce NaN on power_plant with raw-0.5 starts (Gaussian) and
   does not diverge (StudentT); may still stall on the t location (the
   saturating score is a property of the t gradient, not of the curvature).
3. Gaussian on `studentt_sim`/StudentT on `gaussian_sim`, where Observed was
   slightly better: damped ≥ Fisher.

## Cost
One `max` per row per parameter. Implementation: `HessianMode::Damped` arm in
the same `analytical_gradients` blocks (both families).
