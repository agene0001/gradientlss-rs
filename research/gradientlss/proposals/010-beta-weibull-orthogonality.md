# Proposal 010 — Beta / Weibull orthogonality audit

Status: **resolved** — reports/008-cycle7-report.md. Was **pre-registered** (cycle 7, 2026-11-10), after the analytic audit
(`scripts/audit_beta_weibull.py`), before any Beta/Weibull boosting run.

## Audit (exact Fisher, Monte-Carlo checked)
- Beta, crate coordinates (log a, log b): ρ = −0.50 (µ=.5, φ=1), −0.95 (φ=20), −0.99 (φ=100),
  −0.999 (φ=1000) — one diagonal Newton step moves 1 − ρ² = 9.5 % / 2 % / 0.2 % of the way
  along the slow (precision) direction. Same collinearity as Gamma (cycle 4).
  Mean/precision (logit µ, log φ): |ρ| ≤ 0.24 for φ ≥ 20; worse than the crate only for
  diffuse Betas with extreme mean (µ=.1, φ=1: +0.86 vs −0.31).
- Weibull (log λ, log k): ρ = −0.313 everywhere (1 − ρ² = 0.90). η = log λ + (1−γ)/k is exactly
  orthogonal (MC |ρ| < 0.005) but the expected gain is small → no new head; only curvature.
- Both heads hand the booster OBSERVED curvature, which goes negative row-wise (tested).

## Implementation (experiment first; constructor defaults unchanged until measured)
`Beta::with_hessian_mode`, `BetaParameterization::{Concentrations, MeanPrecision}` (moments start
via `default_start_values`, cancellation-safe ψ(x) − ln x and x²ψ′(x) − x forms, outputs still
(a, b)); `Weibull::with_hessian_mode` (Fisher diag(k², π²/6 + (1−γ)²) in log coordinates).

## Data
synth_beta (logit µ = 0.8 sin πx₁ + 0.5x₂, log φ = 2.5 + 1.5x₃: φ 2.7–55), synth_beta_hi (log φ =
5.5 + 1.5x₃: φ 55–1100), bike_share (casual / cnt on rows with 0 < casual < cnt; real);
synth_weibull (log λ = 1 + 0.6 sin πx₁ + 0.3x₂, log k = 0.7 + 0.9x₃: k 0.8–5), kin8nm, power_plant,
naval (positive real targets).

## Predictions (5 paired seeds, η .05/d4, ES 50, crate-default init)
- B1: Beta Concentrations, Damped − Observed ≤ +0.005 everywhere.
- B2: MeanPrecision (Damped) − Concentrations (Observed, today's default): synth_beta_hi ≤ −0.10,
  synth_beta ≤ −0.02, bike_share ≤ +0.005.
- W1: Weibull Damped − Observed ≤ +0.005 everywhere; < −0.01 on at least one real target.
- Decision: a default flips only if better/neutral (≤ +0.005) on every dataset of that family.
