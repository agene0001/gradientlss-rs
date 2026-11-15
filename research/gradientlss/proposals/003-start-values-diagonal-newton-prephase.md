# Proposal 003 — Make unconditional start values actually fit: damped diagonal-Newton pre-phase before L-BFGS

Status: **implemented in base.rs (`diagonal_newton_prephase`: analytic diagonal Newton + trust radius + finite-row guard → L-BFGS), tested at target scales 7e-4…4.5e5; awaiting Felix sign-off as a behavioural correction** (cycle 1, 2026-11-09)

## Measured limitation
`Distribution::calculate_start_values` (base.rs `calculate_start_values_default`) runs argmin
L-BFGS + MoreThuente from raw 0.5 on the summed NLL. For any target whose scale is not already
O(1) the first line-search trial step has length ‖g‖ (7e8 on power_plant, 2.5e5 on
gaussian_sim), produces NaN/Inf, the line search aborts, and argmin returns **the initial
point as the best point** with `Ok(...)` — so the crate silently uses raw 0.5 everywhere.
Observed with `examples/research_probe_start_values.rs` and reproduced standalone in `examples/research_probe_lbfgs.rs`:

| target | y scale | L-BFGS+MoreThuente from 0.5 | moments start | L-BFGS+Armijo backtracking | diag-Newton pre-phase → L-BFGS |
|---|---|---|---|---|---|
| power_plant PE | 454 ± 17 | returns 0.5 (NaN/Inf in line search) | converges, 3 it | **wrong optimum** (log σ → 693) | converges (454.365, 2.837) |
| gaussian_sim y | 10 ± 2.8 | returns 0.5 | — | converges | converges |
| kin8nm y | 0.71 ± 0.26 | converges | — | converges | converges |
| power_plant ×1000 | 4.5e5 | — | — | — | converges (L-BFGS 26 it) |
| kin8nm ÷1000 | 7e-4 | — | — | — | converges |

Consequences today: `initialize: true` is a no-op on most real targets; `dist_select` ranks
candidates by their NLL *at raw 0.5*, i.e. it ranks link functions, not fits; boosting then
starts 450 units away from the data on power_plant, which is what made both curvature modes
fail there (research_log, cycle 1).

## Why not the obvious fixes
- Moment-based start: needs to know which parameter is the location/scale — not expressible
  through the `Distribution` trait, wrong for Gamma/Beta/NB parameterisations.
- Mean instead of sum NLL: helps only when ‖g‖ is moderate (works for gaussian_sim, still fails
  for power_plant). The return value must stay the summed NLL anyway (dist_select contract).
- Backtracking line search alone: accepts a huge first step in the scale coordinate, after
  which the location gradient is ~0 and it "converges" at cost 6.6e6 vs 4.1e4.

## Proposed algorithm (distribution-agnostic, raw space)
For k = 1..K (K ≤ 30), at the current point x with cost c:
1. FD gradient g (already computed today) and FD diagonal second difference
   `h_i = (c(x+εe_i) − 2c + c(x−εe_i))/ε²` (one extra cost evaluation per parameter — the
   `c(x±εe_i)` are shared with the gradient).
2. Damped Newton direction `d_i = −g_i / (max(h_i, |h_i|, 1e-8) + 1e-8)` (negative curvature
   → use |h|; zero → gradient step).
3. Armijo backtracking (ρ = ½, c₁ = 1e-4, ≤ 40 halvings, non-finite cost = reject) on the full
   step; stop when the step is rejected or ‖g‖ < 1e-6.
Then run the existing L-BFGS from the resulting point with the existing iteration budget.
The pre-phase is exactly the per-coordinate Newton step boosting itself takes, so it is
well-scaled by construction: for a Gaussian it jumps µ to ȳ in one step and moves log σ by
`(z²−1)/(2z²) ≈ ½`.

## Invariants / tests
- Returned cost ≤ cost at raw 0.5 (never worse than today).
- Gaussian: start values equal (ȳ, ln s_y) to 1e-6 for targets at scales 1e-3 … 1e5.
- Poisson/NB/Gamma on their existing test fixtures: unchanged or lower cost.
- `dist_select` ranking on the existing fixture unchanged where fits already converged.

## Behavioural change (needs Felix's sign-off before merging)
`initialize: true` and `dist_select` will start producing fitted values on real-scale targets
where they previously returned 0.5. That is a correction of a broken path, not a default change,
but every downstream number moves. Cost: ≤ 30 × (1 + 2p) extra NLL evaluations.
