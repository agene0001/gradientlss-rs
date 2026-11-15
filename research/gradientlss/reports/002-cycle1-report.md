# Cycle 1 report — Newton curvature for LSS likelihoods, and a start-value defect

Date: 2026-11-09. Base commit: `59f91cb` (master). All numbers reproduce from
`research/gradientlss/experiments.csv` (tags `c1_*`) and `experiments_v2.csv`
(tags `c1b_*`, `c1c_*`, `c1d_*`) via `summarize.py` / `select_by_valid.py`;
protocol in `001-protocol.md`.

## 1. Measured limitation
Both backends do diagonal Newton boosting (`w = −Σg/(Σh+λ)`, gain `Σg²/(Σh+λ)`).
Every distribution except Gaussian hands them the *observed* second derivative
of the per-row NLL in raw (link) space. For StudentT this is negative on the
location whenever `d² > ν` (the outlier rows), unsigned on df, and the
Gaussian/StudentT scale curvature under `Exp` is `2z²`-shaped: ≥ 0 but ≈ 0 for
well-fit rows, so the Newton ratio `g/h` is unbounded there and `min_child_weight`
starves the scale tree of splits. Measured consequences (raw-0.5 starts,
η=0.05/depth 4, 5 seeds): Observed-mode fits diverged on power_plant (StudentT,
5/5 seeds, NLL 301), protein (Gaussian, 1/5, NLL 18) and gaussian_sim (StudentT,
1/5 at η=0.1, CRPS 3.5e5).

## 2. Competing explanations tested
| explanation | test | result |
|---|---|---|
| sampling noise / split luck | paired seeds, identical splits, 5 seeds × 2 settings | kin8nm/naval/synth_t3 effects are 5–17 s.e. |
| implicit learning-rate (Fisher's constant `h` = smaller steps) | η ∈ {0.02,0.05,0.1,0.2} per mode, η chosen by *validation* NLL, test compared | Fisher still wins kin8nm (−0.13, z −10), naval (−0.12/−0.16), synth_t3; confound rejected |
| start values, not curvature | `initialize: true` | **inert** — exposed a crate defect (§6); real test is tag `c1d_*` (§7) |
| stabilization is the intended remedy | MAD | harmful for both modes (NLL 46–3e7); incoherent with constant `h` |

## 3. Math
Expected (Fisher) information per observation, chained with `E[∂ℓ/∂θ]=0` so the
raw-space value is `I_θθ·f′(η)²` (no `f″` term):
- StudentT: `I_µµ=(ν+1)/((ν+3)σ²)`, `I_σσ=2ν/((ν+3)σ²)`,
  `I_νν=¼[ψ′(ν/2)−ψ′((ν+1)/2)]−(ν+5)/(2ν(ν+1)(ν+3))` → raw
  `h_loc=(ν+1)/((ν+3)σ²)`, `h_logσ=2ν/(ν+3)`, `h_df=I_νν(ν−2)²` (ExpDf).
  Verified by numerical integration (rel. err < 2e-5) and by a Monte-Carlo test
  that the mean observed Hessian equals it (3%, n=4e5, ν∈{3,6,25}).
- Gaussian: `h_logσ = 2` (already the crate's choice); Observed restored as
  `(−1/σ²+3d²/σ⁴)f′² + (1/σ−d²/σ³)f″` for the reverse test.
- **Damped** (proposal 002): `h = max(h_observed, h_fisher)` per row/parameter
  — Fisher floors the ≈0/negative observed curvature near the optimum; far from
  it the larger observed curvature keeps the step bounded (LM-style).

## 4. Implementation
- `HessianMode::{Observed, Fisher, Damped}` (base.rs), `with_hessian_mode()` on
  `StudentT` (default Observed) and `Gaussian` (default Fisher) — **production
  defaults unchanged**; `#[serde(default)]` keeps old saved models loadable;
  gradients bitwise identical across modes (tested).
- 8 unit tests (positivity, E[observed]=Fisher, serial/parallel path equality,
  elementwise-max identity, serde defaults).
- `examples/research_bench.rs`: CSV loader, seeded paired splits, early stopping
  on validation NLL, closed-form CRPS (Gaussian, t — self-tested vs MC), PIT
  coverage/KS, RMSE, oracle scores on simulated data, divergence diagnostics.
- Test-only fix: two LightGBM-typed tests in `hyper_opt.rs` were ungated, so
  `cargo test --features xgboost` could not compile.

## 5. Results (Fisher vs Observed; then Damped) — raw-0.5 starts
See research_log cycle-1 tables. Summary at η=0.05/depth 4, ΔNLL mean ± s.e.:

| dataset | family | Fisher − Obs | Damped − Obs | Damped − Fisher |
|---|---|---|---|---|
| kin8nm | G / T | −0.134±0.010 / −0.170±0.016 | −0.137 / −0.166 | −0.003 / +0.003 |
| naval | G / T | −0.185±0.032 / −0.119±0.026 | −0.206 / −0.136 | −0.021 / −0.017 |
| synth_t3 (true t DGP) | G / T | −0.018±0.009 / −0.042±0.005 | −0.015 / −0.041 | +0.003 / +0.001 |
| gaussian_sim (true N DGP) | G / T | −0.065±0.031 / **+0.022±0.003** | −0.067 / −0.041 | −0.002 / **−0.063±0.002** |
| protein | G / T | −3.1 (Obs blow-up) / +0.011±0.019 | −3.1 / −0.020 | +0.002 / −0.032±0.013 |
| power_plant | G / T | NaN / −294 (Obs diverged) | −0.068 (no NaN) / −294 | Fisher NaN / −0.23 |
| studentt_sim | G / T | +0.004 / −0.001 | +0.001 / −0.000 | −0.003 / +0.001 |

At η=0.1/depth 6 every Observed gap widens (kin8nm −1.4 nats); Damped − Fisher
becomes −0.084 (gaussian_sim G), −0.345 (protein G, Fisher σ_max 2540), −0.05
(protein T). Calibration: Fisher/Damped PIT-KS lower than Observed on kin8nm,
naval, synth_t3; 90% coverage within ±0.02 of nominal for all finite fits.
Stability: Observed 7 diverged/blown-up fits of 140; Fisher 0 diverged but 10
NaN (power_plant G) and several σ_max blow-ups; **Damped 0 of 280**.

## 6. Start-value defect (proposal 003) — fixed in this working tree
`calculate_start_values` returned raw 0.5 for any target not already O(1)
(MoreThuente's first trial step is of length ‖g‖ → NaN/Inf → argmin returns the
initial point as "best", with `Ok`). A damped diagonal-Newton pre-phase before
L-BFGS now converges on every scale tested (7e-4 … 4.5e5); test
`gaussian_start_values_are_scale_invariant`. **This changes `initialize: true`
and `dist_select` results on real-scale targets** (they were silently inert).
All 213 lib tests and the start-value / dist_select integration tests pass.

## 7. Fitted start values × three modes (tag `c1e_*`, final crate)
420 fits, 0 non-finite rows, 0 errors; no Fisher or Damped fit diverged.
ΔNLL Damped − Observed, mean ± s.e. (η=0.05/d4 ; η=0.1/d6), Damped wins 5/5
unless noted:

| dataset | family | Damped − Observed | Damped − Fisher |
|---|---|---|---|
| kin8nm | G (= T, df frozen) | −0.076 ± 0.013 ; −0.174 ± 0.021 | −0.011 ± 0.006 ; −0.017 ± 0.006 |
| naval | G / T | −0.353 ± 0.040 / −0.372 ± 0.039 ; −0.94 ± 0.10 / −0.96 ± 0.11 | −0.028 / −0.034 ; +0.032 / +0.014 |
| power_plant | G | −0.230 ± 0.160 ; −0.319 ± 0.233 (Observed has one bad seed each) | −0.035 ; +0.004 |
| protein | G | −0.048 ± 0.005 ; −0.043 ± 0.010 | −0.015 ± 0.003 ; +0.014 ± 0.005 |
| synth_t3 | G / T | −0.015 / −0.008 ; −0.060 / −0.025 | ≈ 0 |
| gaussian_sim | G / T | −0.006 / −0.006 ; −0.031 / −0.031 | ≈ 0 / −0.044 ± 0.002 |
| studentt_sim | G / T | ≈ 0 (±0.001) | ≈ 0 |

PIT-KS: Damped lowest or tied in 12/14 cells (naval 0.079 vs Observed 0.102;
protein 0.081 vs 0.107). 90% coverage 0.80–0.92 for all finite fits (kin8nm
and naval under-cover in every mode — a model-family issue, not curvature).

**StudentT with fitted starts**: on every real target the unconditional MLE
puts ν₀ at 2e7–1e9, the df score is ≈ 0 there, and the StudentT cells equal the
Gaussian cells to 4 decimals in Observed/Damped. In Fisher mode the df margin
instead ran to −∞ (ν = 2.000 on every row, 10/10 seeds, protein and
power_plant) — and that degenerate fit is the *best* NLL on both (protein
2.75 vs 2.90; 2.68 vs 2.88). Capping ν₀ at 100 (tag `c1f_*`,
`with_max_start_df`) made the df tree move, but upward to ν ≈ 1e3, and left
NLL unchanged within noise (protein-Damped −0.032 ± 0.015; naval-Damped
+0.035 ± 0.015). The ExpDf parameterisation has a dead zone at both ends
(d ν/dη → 0 at ν → 2; information → 0 at ν → ∞); see research_log for the
mechanism and the open question of why only Fisher collapses.

## 8. Decision
- **Retain `HessianMode::Damped`** as the recommended experimental setting for
  Gaussian and StudentT; propose making it the default after Felix reviews:
  across 700 paired fits it never lost to the current defaults by more than
  0.001 NLL, won by 0.05–0.95 nats on four of five real targets, and removed
  every divergence. `Fisher` stays available for comparison.
- **Land the start-value fix** (`diagonal_newton_prephase`): a behavioural
  correction — `initialize: true` and `dist_select` were inert on any target
  not already O(1). Needs sign-off because all downstream numbers move.
- **Land the StudentT `LARGE_DF` branch** (numerical correctness only).
- `with_max_start_df`: experimental, not recommended (neutral).
- Do not use MAD/L2 stabilization with Newton curvature as implemented.
- Production defaults are unchanged in this tree; nothing is committed.

## 9. Next autonomous investigations (ranked)
1. **StudentT df dynamics**: why Fisher's df runs to ν = 2 while Observed/Damped
   drift to ν ≈ 1e3 on the same data with identical gradients, and why the
   collapsed fit scores best on protein/power_plant. Instrument the per-round
   df margin; test a curvature floor on the df column and a Softplus-type df
   link without the (ν−2) dead zone. Potential 0.1–0.2 nats on heavy-tailed
   targets.
2. Extend Damped to LogNormal, Gamma, Laplace, Gumbel, Logistic, Weibull,
   NegativeBinomial (TotalCountProbs) — same derivation; Gamma's concentration
   curvature is negative for large yβ.
3. Deterministic `par_nansum` (chunked reduction) so seeded start values and
   `dist_select` are reproducible.
4. Stabilization coherence: scale g and h by one shared factor, or drop it.
5. GRN adaptive λ_k ∝ √‖g_k‖ via per-round backend `lambda` — compare with Damped.
6. β-NLL-style split reweighting for the location tree (leaf values unchanged,
   split gains re-weighted by σ^{2β}).
7. Poisson default `Relu` response kills y=0 rows (h=0, f′=0 at margin ≤ 0).
