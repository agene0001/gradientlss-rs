# Proposal 006 — NB MeanDispersion log-θ curvature (Observed+floor vs Fisher vs Damped)

Status: **pre-registered** (cycle 5, 2026-11-10). MeanDispersion is the NB NLL default as of today.

## Measured limitation / motivation
The MeanDispersion head hands the booster the *observed* log-θ curvature floored at
`NB_THETA_HESS_FLOOR = 0.01`. Every other family now uses `HessianMode::Damped`
(max(observed, Fisher)), which won all cycle-1/3 comparisons. The NB log-θ Fisher
information has no closed form:

  I_gg(µ, θ) = θ² Σ_{j≥0} P(Y > j)/(θ + j)² − θµ/(µ + θ),

→ θ²ψ′(θ) − θ (the Gamma shape information) as µ → ∞ and → µ²/(2(µ+θ)²) as θ/µ → ∞
(near-Poisson). Tabulated (python, exact sums): I_gg / (θ²ψ′(θ) − θ) is 0.001–0.36 at θ = 0.05,
0.13 at µ = θ = 1, 0.06 at θ = 300, µ = 100. So in the near-Poisson corner the true information
(≈ 5e-4 at θ = 300, µ = 10) is **20× below the 0.01 floor** — there the floor, not the data,
sets the log-θ step, and steps are ~20× shorter than Newton's.

## Competing explanations of whatever we observe
1. Curvature is irrelevant where µ ≳ θ (observed curvature mostly positive, ≈ Fisher on average).
2. The floor throttles θ only in the near-Poisson corner, where θ is weakly identified anyway
   (fast steps may just add noise → Fisher could *hurt* there).
3. Remaining oracle gaps (synth_nb_hi 0.078, synth_nb_lo 0.025) are mean-side, not θ-side.

## Pre-registered predictions (MeanDispersion, 5 paired seeds, η 0.05/d4, ES on)
- P1 (null for µ ≳ θ): on synth_nb_lo and synth_nb_hi, |ΔNLL| (Fisher or Damped − Observed) < 0.01.
- P2 (near-Poisson, new DGP synth_nb_ps: µ ∈ ~[1, 8], θ ∈ ~[12, 245]): Damped − Observed ≤ −0.003
  with no seed worse by > 0.005. If instead it is ≥ +0.003 → explanation 2 (noise) wins and the
  floor is a useful regulariser.
- P3: Fisher alone is no better than Damped anywhere (as in cycle 1).
- Decision rule: Damped becomes the NB default only if better/neutral on all three DGPs; if it
  only ties, keep Observed (no reason to pay for the sum).

## Implementation (experiment first)
`NegativeBinomial::with_hessian_mode(HessianMode)`; serde/constructor default `Observed`
(= today's behaviour, floor 0.01). Fisher/Damped use the exact sum, log-space pmf recursion,
stopped once P(Y > j) < 1e-12 past the mean (slow; only to measure). A fast approximation
(table or asymptotic blend) is built only if P2 holds.
