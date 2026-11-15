# Proposal 004 — Residual warm start for shape/scale parameters (StudentT first)

Status: **validated in the harness (tags c2c_*, c2d_*), crate API proposed, not implemented in `src/`** (cycle 2, 2026-11-09)

## Measured limitation
The crate initialises every distribution parameter from the *marginal* of y
(`calculate_start_values`). For a location–scale–shape family this is wrong as
soon as the trees explain variance: σ₀ is the marginal sd (too large), and the
df gradient evaluated at that σ points toward *larger* ν, so the df tree first
climbs to ν ≈ 1e3 (information ≈ 0) and is frozen by the time the scale tree has
shrunk σ enough for heavy tails to show (cycle-2 probe + trajectories). Result:
StudentT ≡ Gaussian on every real target (protein, power_plant, kin8nm, naval),
losing 0.02–0.13 nats that a learnable df recovers.

## Proposed procedure (what the harness does)
1. Stage 1: fit a Gaussian `GradientLSS` (same backend params, early stopping on
   the validation split) → location ŷ(x) for train/valid/test.
2. Stage 2: unconditional t-fit on the **validation** residuals y − ŷ
   (`StudentT::calculate_start_values`, `initialize: true`) → (ν₀, µ_r, σ₀);
   clamp ν₀ ≤ 10 (`with_max_start_df`), ν₀ ≥ 2.5 (floor, now in the crate).
3. Stage 3: train the StudentT with `initialize: false` and per-row raw offsets
   `[ln(ν₀−2) − 0.5, ŷ(x) + µ_r − 0.5, ln σ₀ − 0.5]` via
   `train_with_callbacks_and_offsets`; predict with `predict_with_offset` using
   the stage-1 location on the new rows.
Only existing public machinery is used (the offset path added for per-row
priors). Cost: one extra Gaussian fit.

## Evidence (StudentT, Damped, η=0.05/depth 4, 5 paired seeds; ΔNLL vs marginal start)
kin8nm −0.020 ± 0.002 (5/5) · naval −0.118 ± 0.025 (5/5) · power_plant −0.127 ± 0.077 (5/5) ·
protein −0.087 ± 0.026 (5/5; −0.115 with ν₀ ≤ 10) · synth_t3 −0.017 ± 0.002 (5/5) ·
studentt_sim −0.004 ± 0.001 (5/5) · gaussian_sim **+0.017 ± 0.005 (0/5)**.
Against the Gaussian-Damped default: better on 6/7 (−0.02 … −0.13), worse on
gaussian_sim (+0.035). CRPS moves the same way; PIT-KS improves or is unchanged.

Why gaussian_sim loses: its conditional law is exactly Gaussian but σ(x) ∈ [1,5]
makes the residual marginal a scale mixture — leptokurtic — so ν₀ ≈ 3 is the
correct unconditional answer and the df tree climbs back too slowly before early
stopping (round ≈ 60). In-sample residuals (overfit, more peaked) make this
worse (+0.018) and naval/gaussian_sim start at the ν = 2 floor; validation
residuals are the honest choice.

## Proposed crate API (for Felix)
```rust
// on GradientLSS<B> (StudentT, later Gamma/NB shape params):
pub fn warm_start_from_residuals(&mut self, stage1: &GradientLSS<B>, train, valid) -> Result<WarmStart>;
// WarmStart carries (ν₀, µ_r, σ₀) and builds the per-row offsets for train/valid/predict.
```
Alternatively a `TrainConfig::residual_warm_start: Option<WarmStartSpec>` that
runs the stage-1 Gaussian internally and threads the offsets through
`train`/`predict`, so users do not have to manage offsets. Either way the
prediction path must carry the stage-1 location, so a saved model needs to
serialise the stage-1 booster alongside (or the two-stage object).

## Open issues / next
- df climbs slowly from a heavy-tailed start (gaussian_sim): a larger df step
  (lower curvature floor on the df column, or a df-specific learning rate) is
  the complementary fix; test on gaussian_sim + protein jointly.
- Out-of-fold residuals (k-fold stage 1) would remove the validation-size
  dependence of (ν₀, σ₀); not needed at 1–2k validation rows here.
- Same idea applies to Gamma/NegativeBinomial dispersion starts.
