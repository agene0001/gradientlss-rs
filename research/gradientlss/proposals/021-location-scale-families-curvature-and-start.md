# Proposal 021 — Cauchy, Logistic, Gumbel, Laplace, LogNormal: positive curvature and a data start

Status: **run** 2026-11-13 (H2 and H6 ✓ for Cauchy, Logistic, Laplace, LogNormal; ✗ for Gumbel). **Pre-registered** 2026-11-13 after the baseline (c15_base / c15_init) and four smoke cells of
the prototype (Logistic and Cauchy, seed 0, power_plant and kin8nm, Damped + LocationScale), before
any arm was run. Skill: `optimization-algorithm-research`.

## Observed limitation (c15_base, crate defaults, 5 seeds; Gaussian on the same splits for scale)
| dataset | Gaussian | Cauchy | Logistic | Gumbel | Laplace | LogNormal |
|---|---|---|---|---|---|---|
| power_plant (y ≈ 454) | 2.68 | **12.88** (stops at round 1, cov90 0.000) | **7.61** | **7.43** | **7.65** | 2.84 |
| kin8nm | −0.656 | −0.069 | −0.098 | −0.159 | −0.366 | −0.328 |
| naval | −6.05 | −4.78 | −5.58 | −4.59 | −5.20 | −5.33 |
| protein | 2.859 | 2.941 | 2.901 | **73 681** (one seed 368 394) | 2.987 | — |
| gaussian_sim | 2.101 | 2.273 | 2.108 | 2.356 | 2.154 | — |

`initialize: true` (c15_init) is not a fix: it repairs power_plant for Cauchy, Logistic and Laplace
(2.82 / 2.71 / 2.75) but gives Gumbel 142 761 there and 8.5 on kin8nm, takes Cauchy on naval from
−4.78 to −2.87 (stops at round 1) and Logistic on naval from −5.58 to −4.75.

## Baseline mathematics (read from the code; z = (y − μ)/s, η₁ = μ, η₂ = ln s)
Leaves are −Σg/(Σh + λ); XGBoost returns 0 for a leaf with Σh ≤ 0 or < `min_child_weight`.
- Logistic: h_μ = 2σ(z)(1 − σ(z))/s² ~ 2e^{−|z|}/s², while |g_μ| → 1/s. At the raw start on
  power_plant z ≈ 275, so h_μ underflows to 0: no step.
- Gumbel: h_μ = e^{−z}/s²: 0 for z ≫ 0, overflow for z ≪ 0.
- Cauchy: h_μ = 2(1 − z²)/(s²(1 + z²)²) < 0 whenever |z| > 1, i.e. for half of all rows *at the true
  parameters*, and for every row at a far start.
- Laplace: h_μ is already 1/b² (the Fisher value); the raw start still leaves 450 units to cover
  in steps of η·b.
- LogNormal: h_μ = 1/σ², h_{ln σ} = 2d² with d = (ln y − μ)/σ: non-negative, zero at d = 0.

## Candidate
Per row and parameter, with the Fisher information I(η) in predictor space:

    Fisher:  h ← I(η)          Damped:  h ← max(h_observed, I(η))   (I(η) where h_observed is not finite)

I_μμ = c_loc/s², I_ss = c_scale/s², I_{η₂η₂} = I_ss·(ds/dη₂)² (= c_scale under the Exp link):

| family | c_loc | c_scale |
|---|---|---|
| Cauchy | 1/2 | 1/2 |
| Logistic | 1/3 | (π² + 3)/9 |
| Gumbel | 1 | π²/6 + (1 − γ)² |
| Laplace | 1 | 1 |
| LogNormal (μ, σ of ln y) | 1 | 2 |

Start rule without `initialize` (closed forms, no optimiser):
- `Location`: μ₀ = median (Gumbel: median + s₀ ln ln 2; LogNormal: median of ln y), scale raw 0.5.
- `LocationScale`: Logistic (median, sd·√3/π); Cauchy (median, IQR/2); Laplace (median, mean |y − median|);
  Gumbel (ȳ − γs, s = sd·√6/π); LogNormal (mean, sd of ln y).

Objective unchanged (NLL, a proper score); gradients unchanged; model class unchanged. Cost: one
pass over the rows per round. No convergence proof; the argument is the one of proposal 002 (the
Fisher value floors vanishing or negative curvature; the larger observed value still bounds steps).

Failure mode to watch: a Fisher floor makes the location step at most g/I = (1/s)/(c/s²) ∝ s per
round, so far from the target the crawl is slow; that is what the start rule is for.

## Verification (done, `src/distributions/location_scale.rs` tests)
- Monte Carlo, 400 000 draws at the true parameters: mean observed Hessian = Fisher constant within
  2 % for both parameters of all five families.
- Damped = row-wise max, bit for bit; always positive.
- Defaults unchanged: `default_start_values` is `None`, a model serialised without the new fields
  loads as Observed / Raw, and the bench with the port off reproduces a c15_base cell exactly
  (Logistic, kin8nm, seed 0: −0.1334 at 141 rounds).
- Starts recover the generating (loc, scale) within 5 % and are shift-equivariant.

## Protocol
XGBoost, 5 seeds, eta 0.05, depth 4, 2000 rounds / patience 50. Development sets: kin8nm,
power_plant, naval, protein, gaussian_sim (LogNormal: the three positive ones).

| arm | curvature | start |
|---|---|---|
| `base` (c15_base) | Observed | Raw |
| `d_raw` | Damped | Raw |
| `d_loc` | Damped | Location |
| `d_ls` | Damped | LocationScale |
| `o_ls` | Observed | LocationScale |

Tags `c15_<arm>`. **Selection without test:** the recipe is chosen per family by mean rank of
validation NLL over the development sets. The chosen recipe is then run once, with `base`, on the
five unseen continuous sets of proposal 020 (tags `c18_base`, `c18_port`).

## Predictions
- **H1.** `d_ls` on power_plant: Cauchy, Logistic, Gumbel, Laplace all below 3.2 (base 7.4–12.9).
- **H2.** `d_ls` ≤ `base` + 0.005 test NLL on every (family, development set).
- **H3.** Curvature alone is not enough far from the target: `d_raw` Logistic on power_plant stays
  above 5.
- **H4.** `d_ls` Logistic on kin8nm is within 0.25 of Gaussian (base: 0.56 behind).
- **H5.** No arm with Damped has a mean test NLL above Gaussian + 1 on any set (the Gumbel blow-ups go).
- **H6 (unseen sets).** The chosen recipe ≤ `base` + 0.005 on every unseen pair.

## Decision rule
The options stay opt-in. If H2 and H6 hold for a family, propose its chosen recipe to Felix as that
family's default, with the tables. A family that fails H2 or H6 keeps its default and gets a note.

## Result (2026-11-13, report 015 §5)
- **H1 ✗** (Gumbel 114 on power_plant, one seed 557; the other three 2.69–2.82). **H2 ✓** for Cauchy,
  Logistic, Laplace, LogNormal; **✗** for Gumbel (naval +4.76, power_plant +107). **H3 ✓** (7.34).
  **H4 ✓** (0.18 behind Gaussian). **H5 ✗** (Gumbel).
- Chosen by validation rank: Cauchy `d_loc`, LogNormal `d_loc`, Logistic `d_ls`, Laplace `d_ls`,
  Gumbel `d_raw`. Each chosen recipe is ≤ `base` on every development pair of its family.
- **H6 (unseen, run once): ✓ for Cauchy, Logistic, Laplace, LogNormal (20/20 pairs); ✗ for Gumbel**
  (c_gt_co +0.016 ± 0.008, 1/5).
- Laplace: Damped with a raw or location-only start is worse than `base` on naval (+0.75, +1.14).
- **Decision.** Options stay opt-in. Default change for the four passing families goes to Felix with
  the tables. Gumbel keeps its default and a README note.

**Applied 2026-11-13 (Felix's decision; proposal 026):** the recipes are now the defaults of
`Cauchy::new`, `LogNormal::new`, `Logistic::new`, `Laplace::new`. Gumbel unchanged.
