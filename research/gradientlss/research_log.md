# GradientLSS research log

Persistent memory for the autonomous algorithm-research cycles run under
`.agents/skills/gradientlss-research`. Newest cycle at the bottom. Every
claim below is tagged **[measured]**, **[derived]**, **[literature]** or
**[untested]**.

---

## Cycle 1 — 2026-11-09 — repository analysis + baseline + first hypothesis

### Repository state at start
- master @ `59f91cb` (xgb pinned to 3.3.0 so the rust-xgboost fork `[patch]` applies).
- Working tree: only a comment in `Cargo.toml` (uncommitted, Felix's).
- No `research/` workspace existed before this cycle.

### Local evaluation assets found (no network needed)
| source | file | n | notes |
|---|---|---|---|
| ngboost clone | `../ngboost/ngboost/data/uci/kin8nm.csv` | 8192 | 8 feats, continuous y (header row) |
| ngboost clone | `../ngboost/ngboost/data/uci/protein.csv` | 45730 | CASP protein |
| ngboost clone | `../ngboost/ngboost/data/uci/naval-propulsion.txt` | 11934 | whitespace-separated, 16 cols |
| ngboost clone | `../ngboost/ngboost/data/uci/power-plant.xlsx` | — | needs conversion |
| ngboost clone | `../ngboost/ngboost/data/surv/*.csv` | ~8–9k each | survival; not used yet |
| XGBoostLSS | `../Lss-Models/XGBoostLSS/xgboostlss/datasets/gaussian_{train,test}_sim.csv` | 7000/3000 | known DGP: y ~ N(loc(x_true), scale(x_true)) + 10 noise feats; `loc`,`scale` columns present → oracle NLL computable |
| XGBoostLSS | `.../studentT_{train,test}_sim.csv` | 8000/2000 | y ~ t(df, loc, scale), columns `loc, scale, df, x, x2` → oracle available |
| XGBoostLSS | `.../sim_triv_{gaussian,studentT}.csv` | 10000 | trivariate, oracle params present |

Python references on disk: `../Lss-Models/XGBoostLSS`, `../Lss-Models/LightGBMLSS`
(for parity checks only; policy: see memory `gradientlss-parity-policy`).

### Literature gathered via Lacuna MCP (cycle 1)
All retrieved 2026-11-09; summaries are Lacuna's, verified only where noted.

- **NGBoost** (Duan et al., ICML 2020) — natural gradient (Fisher preconditioning) for
  distributional boosting. https://lacuna.tiptreesystems.com/work/ngboost-natural-gradient-boosting-for-probabilistic/version/art_8bb559093868439e9df690e9dc7ebc1e
  Relevance: this crate already uses an NGBoost-style metric preconditioner for Poisson CRPS
  (commit e505e13). Question: is the NLL path's Hessian the *observed* or *expected* information?
- **Gradient Regularized Newton Boosting Trees with Global Convergence** (Zozoulenko, Falkowski,
  Cass, Gonon, arXiv 2605.00581, May 2026). Vanilla Newton boosting can diverge; adding an
  adaptive ℓ2 term λ_k ≥ sqrt(M‖g_k‖) to the per-round quadratic gives O(1/k²) global convergence
  with no change to split finding — only the leaf denominator Σh+λ_k changes.
  https://lacuna.tiptreesystems.com/work/gradient-regularized-newton-boosting-trees-with-global
  Relevance: a *per-round adaptive* `lambda` for the backend is implementable from the objective
  side (we see g each round). Falsifiable claim: fewer divergent/oscillating fits for heavy-tailed
  dists (StudentT df, NB dispersion) at larger learning rates. **[literature, untested]**
- **TRBoost** (Luo et al., 2022/2023) — trust-region subproblem per round; Hessian need not be PD;
  adaptive radius from actual/predicted reduction ratio ρ.
  https://lacuna.tiptreesystems.com/work/trboost-a-generic-gradient-boosting-machine-based/version/art_8fc0d64aa78d4c958e7108118f3fa271
  Relevance: observed Hessians of LSS likelihoods are indefinite in the unconstrained space
  (e.g. Gaussian scale when |z|<1 after exp link is fine, but StudentT df / NB dispersion are not).
- **Noncyclical gamboostLSS** (Thomas, Mayr, Bischl, Schmid, Smith, Hofner, Stat. Comput. 2018) —
  update only the distribution parameter whose base-learner most reduces risk each iteration;
  collapses m_stop tuning to 1-D, same final solution, faster early risk reduction.
  https://lacuna.tiptreesystems.com/work/gradient-boosting-for-distributional-regression-faster/version/art_b2bb664e25e9477da5ee90522e50844c
  Relevance: both backends here fit one tree per parameter *every* round (XGB `num_target`,
  LGBM `num_class`). A "parameter-selective" round would need backend support → invasive.
- **β-NLL / Pitfalls of heteroscedastic uncertainty estimation** (Seitzer et al., ICLR 2022) —
  NLL training lets the scale head "explain away" mean error; loss weight stop_grad(σ^{2β}),
  β=0.5 best; mean gradient recovers MSE at β=1.
  https://lacuna.tiptreesystems.com/work/on-the-pitfalls-of-heteroscedastic-uncertainty-estimation/version/art_abef037adc544a84b33e6627fb3928d9
- **Faithful Heteroscedastic Regression** (Stirn et al., AISTATS 2023) — scale the mean gradient
  by the predicted variance (Newton-like step for the mean) + stop-grad shielding so the variance
  objective cannot corrupt the mean.
  https://lacuna.tiptreesystems.com/work/faithful-heteroscedastic-regression-with-neural-networks-sd27/version/art_caafc23609f7415b8058637aaba8aa0c
  Relevance to boosting: in Newton boosting the leaf value is −Σg/Σh. For a Gaussian loc with
  g = −(y−μ)/σ², h = 1/σ², the leaf step is already the σ²-scaled "faithful" step *within a leaf*,
  but the **split gain** Σg²/Σh is weighted by 1/σ² → high-variance rows have little say in where
  the loc tree splits. That is the boosting analogue of the "rich-get-richer" pathology.
  Hypothesis candidate, see H2. **[derived, untested]**
- **Distributional regression: CRPS-error bounds** (Dombry & Zaoui, NeurIPS 2024) — CRPS is
  2-Lipschitz in W1; validation-set CRPS model selection has O(sqrt(log M / N)) regret.
  https://lacuna.tiptreesystems.com/work/distributional-regression-crps-error-bounds-for-model/version/art_3f2b9b9f91044490b8ff92634bd01086
  Use: justifies CRPS on a held-out split as the selection criterion in our benchmark harness.
- **Large-Scale Study of Probabilistic Calibration in NN Regression** (ICML 2023) — PIT-based
  calibration metrics at scale. https://lacuna.tiptreesystems.com/work/a-large-scale-study-of-probabilistic-calibration/version/art_22fad57cdd2d4ff3995aa4bf837e820b
  Use: PIT histogram / coverage are what our harness must report; none exists in the crate yet.
- Lacuna hypothesis "Outlier-Aware Robust Splits for Boosted Regression Trees" (af98214cbb619d7c,
  AI-generated, speculative) — Huber-inside-adversarial-split; about point regression under
  feature attack, **not applicable** to distributional NLL here. Recorded, rejected.

Python reference check (parity, not a bug hunt): `XGBoostLSS/distributions/distribution_utils.py:415-500`
— NLL Hessian is the autograd *observed* information diag (d/dθ_i of Σ∂ℓ/∂θ_i), CRPS Hessian is
`ones_like`, stabilization MAD/L2 applied to grad AND hess separately, weights multiplied after.
So the Rust crate's design choices inherit: observed (possibly negative) Hessians, per-parameter
rescaling that destroys the Newton ratio g/h unless both are divided by the same constant (they
are not — MAD of g and MAD of h differ). **[verified in Python source]**

### Open questions to settle from the Rust mapping (pending agent reports)
1. Exact Hessian expression per distribution and whether any floor/abs is applied.
2. Whether stabilization divides g and h by *different* constants (changes the Newton step
   g/h by MAD(h)/MAD(g) — a hidden per-parameter learning-rate).
3. What metric drives early stopping and whether any calibration diagnostic exists.

### Cycle 1 — implementation notes (2026-11-09)
- Added `HessianMode::{Observed, Fisher}` (base.rs) + `with_hessian_mode()` on `StudentT`
  (default Observed = unchanged) and `Gaussian` (default Fisher = unchanged). `#[serde(default)]`
  keeps old saved models loadable. Gradients bitwise identical between modes (tested).
- **[measured]** Monte-Carlo test: mean of the observed raw-space Hessian over y ~ t(ν,µ,σ) equals
  the Fisher closed form within 3% for all three params at ν ∈ {3, 6, 25} (n = 4e5). So the Fisher
  mode is literally the expectation of the default — any benchmark difference is variance/sign,
  not bias.
- **[derived]** Sign analysis in raw space with the `Exp` link on σ: the response chain term
  `∂ℓ/∂σ·σ` cancels the `−1/σ²·σ²` term, so
  - Gaussian: `h_logσ = 2z²` (≥ 0, ≈ 0 when |z| small) — the gaussian.rs comment claiming
    negativity for |y−µ| < σ/√3 is only true *without* the chain term. The Gaussian Observed-mode
    pathology is therefore **near-zero curvature**, not sign flips: g/h = (1−z²)/(2z²) → ∞ as z→0.
  - StudentT: `h_logσ = 2ν(ν+1)d²/(ν+d²)² ≥ 0`, same zero-prone shape; `h_µ` is negative iff
    d² > ν; `h_logν` has both signs.
  Consequence: within a leaf, rows with tiny residuals contribute ≈0 curvature to the scale
  column and the Newton step is set by the few large-|z| rows; `min_child_weight` (default 1)
  counts Σh, so leaves of well-fit rows become un-splittable for the scale tree. Fisher gives every
  row the same curvature (2 for Gaussian, 2ν/(ν+3) for t).
- Fixed test-only build break: two LightGBM-typed tests in `hyper_opt.rs` were not gated on
  `feature = "lightgbm"`, so `cargo test --features xgboost` could not compile the lib tests.
- Benchmark harness: `examples/research_bench.rs` (see reports/ for protocol). Closed-form CRPS for
  Gaussian and t (self-test vs Monte Carlo), PIT coverage 50/90, KS-to-uniform, RMSE, oracle
  NLL/CRPS on simulated data, non-finite row counts, fit seconds; appends to experiments.csv.

### Cycle 1 — interim results, simulated datasets (2026-11-09) **[measured]**
Protocol: reports/001-protocol.md. Paired Fisher − Observed, 5 seeds, both settings
(η=0.05/depth 4 and η=0.1/depth 6). Full table: `uv run --with pandas python -I summarize.py`.

| dataset | family | η/depth | ΔNLL mean ± se | Fisher wins | note |
|---|---|---|---|---|---|
| synth_t3 (true t DGP) | StudentT | .05/4 | −0.042 ± 0.005 | 5/5 | ΔCRPS −0.0065 ± 0.0006 |
| synth_t3 | StudentT | .10/6 | −0.097 ± 0.012 | 5/5 | ΔCRPS −0.024 ± 0.003 |
| synth_t3 | Gaussian | .05/4 | −0.018 ± 0.009 | 5/5 | |
| synth_t3 | Gaussian | .10/6 | −0.066 ± 0.012 | 5/5 | |
| gaussian_sim (true N DGP) | Gaussian | .05/4 | −0.065 ± 0.031 | 5/5 | Fisher ran +49 more rounds before early stop |
| gaussian_sim | Gaussian | .10/6 | −0.023 ± 0.026 | 4/5 | |
| gaussian_sim | StudentT | .05/4 | **+0.022 ± 0.003** | 0/5 | Observed better; misspecified-light-tail case |
| gaussian_sim | StudentT | .10/6 | −0.19 ± 0.11 | 5/5 | Observed seed 4 **diverged**: mean σ̂ = 1.4e6, RMSE 13.6, CRPS 3.5e5 |
| studentt_sim | StudentT | both | −0.0014 / −0.0038 (z −5) | 5/5 | tiny; 2 features, oracle columns unreliable (see below) |
| studentt_sim | Gaussian | both | +0.0036 ± 0.001 | 0–1/5 | Observed slightly better |

Reading:
- Where the DGP is genuinely heavy-tailed and heteroscedastic with every parameter
  feature-dependent (synth_t3), Fisher curvature is a clear, consistent win for both families,
  larger at the more aggressive setting — consistent with H1 (observed curvature corrupts the
  Newton direction/splits most when steps are large).
- Prediction 2 (stability) confirmed once: an Observed StudentT fit diverged; no Fisher fit did
  (0 non-finite rows anywhere, 140 fits).
- The one systematic reversal (StudentT on Gaussian data, η=0.05) is small but 7 s.e. — the
  observed df/scale curvature on light-tailed data may act as a useful *larger* step toward
  ν→∞. Needs the fair-tuning sweep before concluding.
- Confound to kill: Fisher's constant scale curvature (2 vs 2z²) means smaller scale steps,
  i.e. an implicit learning-rate change; Fisher also early-stops later. Next: η sweep per mode,
  select each mode's η by *validation* NLL, compare test (equal tuning budget, no test leakage).

Data-quality note **[measured]**: XGBoostLSS's `studentT_*_sim.csv` oracle columns do not
describe the file's y: PIT under (loc, scale, df) is non-uniform (KS p = 1.5e-25 train,
7e-8 test; standardized residuals fit a t with scale ≈ 0.77). Models legitimately beat the
"oracle" NLL there. `gaussian_*_sim.csv` oracle is consistent (PIT KS p = 0.90 / 0.74). Only
synth_t3 and gaussian_sim carry a trustworthy oracle.

Prior art on the curvature choice (not retrievable via Lacuna, cited from memory — verify):
Rigby & Stasinopoulos (2005) GAMLSS RS algorithm defaults to expected information (Fisher
scoring) for the per-parameter updates; NGBoost (Duan et al. 2020) uses the Fisher metric;
XGBoostLSS/LightGBMLSS use autograd observed. Lacuna did not surface the 2005 JRSS-C paper.

### Cycle 1 — real-data results and a Fisher failure mode (2026-11-09) **[measured]**
η=0.05/depth 4 (5 seeds, paired):

| dataset | family | ΔNLL (F−O) | wins | Observed NLL | Fisher NLL |
|---|---|---|---|---|---|
| kin8nm | Gaussian | −0.134 ± 0.010 | 5/5 | −0.381 | −0.515 |
| kin8nm | StudentT | −0.170 ± 0.016 | 5/5 | −0.343 | −0.513 |
| naval | Gaussian | −0.185 ± 0.032 | 5/5 | −5.479 | −5.664 |
| naval | StudentT | −0.119 ± 0.026 | 5/5 | −5.520 | −5.639 |
| protein | Gaussian | −3.1 ± 3.0 | 4/5 | 6.06 (seed 2 = 18.2, partial divergence) | 2.96 |
| protein | StudentT | +0.011 ± 0.019 | 2/5 | 2.789 | 2.800 |
| power_plant | Gaussian | — | — | 2.75 (fine) | **NaN all seeds** |
| power_plant | StudentT | −294 | 5/5 | **301 (diverged, all seeds)** | 7.75 (loc never learned, RMSE 448) |

At η=0.1/depth 6 the kin8nm gaps grow to −1.42 ± 0.28 / −1.44 ± 0.08 (Observed degrades badly
with bigger steps), naval −0.39/−0.24, protein StudentT −0.062 ± 0.018 (3.4 se), power_plant
Observed-StudentT ran all 2000 rounds at NLL 401.

**power_plant anatomy [derived]** (y ≈ 420–495, start values raw 0.5 ⇒ µ₀=0.5, σ₀=e^0.5):
- Gaussian scale column: g = (1−z²)·1 ≈ −7.4e4 with z ≈ 270. Observed h = 2z² ≈ 1.5e5 ⇒ raw
  step ≈ η·0.5 (self-regulating Newton step in log σ). Fisher h = 2 ⇒ raw step ≈ η·7.4e4/3 ≈
  1200 ⇒ σ = e^1200 = ∞ ⇒ NaN after round 1. **Fisher scoring far from the optimum takes the
  textbook huge step; observed curvature is a free trust region here.**
- StudentT location: the t score saturates, g_µ ≈ −(ν+1)/(σd); Fisher h_µ = (ν+1)/((ν+3)σ²)
  ⇒ Newton step ≈ (ν+3)σ/d → ~0: the model cannot move µ by 450 and instead inflates σ until
  d is O(1) — it early-stops at RMSE 448. Observed h_µ ≈ −(ν+1)/(σ²d²) < 0 ⇒ step −g/h has the
  WRONG sign ⇒ divergence to NLL 300–400.
- Neither mode is acceptable there; the root cause is the crate default `initialize: false`
  (raw 0.5 start regardless of the target's scale). Test pending: BENCH_INIT=1.
- Honest scoring: for the Fisher/Observed question, power_plant is a *negative* result for
  pure Fisher (prediction 2 falsified in this regime). The fix candidates are (a) proper start
  values, (b) damped curvature h = max(observed, Fisher) (LM-style; see proposals/002), (c) GRN
  adaptive λ_k ∝ √‖g_k‖ (Zozoulenko et al. 2026) — (c) needs per-round backend param updates.

Confound batch launched (tag prefix `c1b_`, output experiments_v2.csv with valid_nll/df/scale
diagnostics): η ∈ {0.02, 0.05, 0.1, 0.2} at depth 4; `initialize: true`; MAD stabilization.

### Cycle 1 — confound batch results (2026-11-09) **[measured]**  (experiments_v2.csv, tags c1b_*)
**Equal tuning budget (η ∈ {0.02, 0.05, 0.1, 0.2} swept per mode, each mode's η picked by
validation NLL, test compared; `select_by_valid.py`):**

| dataset | family | ΔNLL Fisher−Observed | z | verdict |
|---|---|---|---|---|
| kin8nm | Gaussian / StudentT | −0.131 ± 0.013 / −0.133 ± 0.013 | −10 / −10 | Fisher |
| naval | Gaussian / StudentT | −0.120 ± 0.023 / −0.159 ± 0.035 | −5.3 / −4.6 | Fisher |
| synth_t3 | Gaussian / StudentT | −0.018 ± 0.009 / −0.014 ± 0.005 | −1.9 / −3.0 | Fisher (small) |
| gaussian_sim | Gaussian | −0.059 ± 0.033 | −1.8 | Fisher (weak) |
| gaussian_sim | StudentT | **+0.025 ± 0.003** | +9.9 | Observed |
| studentt_sim | Gaussian / StudentT | +0.002 ± 0.001 / −0.002 ± 0.002 | +3.3 / −1.0 | ≈ tie |
| protein | Gaussian | −3.1 ± 3.1 (one Observed blow-up) | — | Fisher by stability |
| protein | StudentT | +0.027 ± 0.015 | +1.8 | Observed (weak) |
| power_plant | both | Fisher-Gaussian NaN; Fisher-StudentT stalls; Observed-StudentT diverges | — | both fail |

→ The step-size confound is rejected: Fisher's wins survive per-mode η selection. The pattern is
also interpretable: Fisher wins when the data are heteroscedastic/heavy-tailed with learnable
structure (kin8nm, naval, synth_t3); Observed wins modestly when a t is fit to light-tailed data
(gaussian_sim), where its larger observed df-curvature steps help ν → ∞ faster.

**`initialize: true` (L-BFGS start values):** bit-identical to raw-0.5 on gaussian_sim,
studentt_sim, protein, power_plant; changed only kin8nm/naval/synth_t3. Probe
(`examples/research_probe_start_values.rs`): `calculate_start_values` **returns the initial point 0.5 with the
initial loss** for power_plant (y≈454, loss 3.6e8) and gaussian_sim (y≈10, loss 1.4e5); it
converges only on kin8nm (y≈0.7 → loc 0.714, log σ −1.33, correct). So start-value fitting and
therefore `dist_select` are silently inert for targets not already O(1). **Crate defect,
independent of Hessian mode** — see below for mechanism once read.

**MAD stabilization (η=0.05):** harmful in nearly every cell for *both* modes (Observed: kin8nm
NLL 83, naval 46, synth_t3-StudentT 3e7; Fisher: power_plant 1932, naval-StudentT 265). With
Fisher the per-column MAD of a constant h is 0, the code then leaves h unscaled while g is
divided by MAD(g) → the Newton ratio is destroyed. Stabilization as implemented (inherited from
gamboostLSS/XGBoostLSS: g and h scaled by *different* constants) is incoherent with Newton
boosting; if kept, g and h should share one scale factor. Not pursued further this cycle.

### Cycle 1 — start-value defect: mechanism + fix validated (2026-11-09) **[measured]**
`examples/research_probe_lbfgs.rs` (standalone Gaussian NLL, same argmin setup as base.rs): MoreThuente's
first trial step along −g has length ‖g‖ (7e8 on power_plant) → NaN/Inf → line search aborts →
argmin returns the INITIAL point as best with `Ok` → crate uses raw 0.5 (the `Err` fallback is
never reached). Armijo backtracking alone fixes gaussian_sim/kin8nm but converges to a wrong
point on power_plant (log σ → 693). A damped diagonal-Newton pre-phase (FD diag Hessian, Armijo)
followed by the existing L-BFGS reaches the exact optimum on every target tested incl. ×1000 and
÷1000 rescalings. Written up as proposals/003. Patch to base.rs pending (after c1c finishes).
Implication for the curvature study: the power_plant failures of BOTH modes were downstream of
this defect (start 450 units from the data). Re-run power_plant after the fix before any verdict.

### Cycle 1 — Damped curvature (proposal 002) results (2026-11-09) **[measured]**  (tags c1c_*)
3 modes × 7 datasets × 2 families × 5 seeds × 2 settings, raw-0.5 starts. ΔNLL = Damped − X:

| dataset | family | vs Observed (η.05/d4) | vs Fisher (η.05/d4) | vs Observed (η.1/d6) | vs Fisher (η.1/d6) |
|---|---|---|---|---|---|
| kin8nm | G / T | −0.137 / −0.166 (5/5, 5/5) | −0.003 / +0.003 (tie) | −1.42 / −1.44 | −0.003 / +0.004 (tie) |
| naval | G / T | −0.206 / −0.136 (5/5) | −0.021 / −0.017 (≈tie) | −0.378 / −0.264 | +0.009 / −0.028 |
| synth_t3 | G / T | −0.015 / −0.041 (5/5) | +0.003 / +0.001 (tie) | −0.057 / −0.094 | +0.009 / +0.003 (tie) |
| gaussian_sim | G | −0.067 (5/5) | −0.0015 | −0.107 | **−0.084 ± 0.018** (Fisher σ_max 95) |
| gaussian_sim | T | **−0.041 ± 0.003** (5/5) | **−0.063 ± 0.002** (5/5) | −0.24 | −0.050 ± 0.003 |
| protein | G | −3.1 (Observed blow-up) | +0.002 | −3.1 | **−0.345 ± 0.047** (Fisher σ_max 2540) |
| protein | T | −0.020 ± 0.014 | −0.032 ± 0.013 | −0.111 ± 0.016 | −0.049 ± 0.007 |
| power_plant | G | −0.068 (5/5, **no NaN**) | Fisher = NaN | −0.170 | Fisher = NaN |
| power_plant | T | −294 (Observed diverged) | −0.23 (both still stalled on loc → start-value defect) | −394 | −0.23 |
| studentt_sim | G / T | ±0.001 | −0.003 / +0.001 | −0.002 / −0.004 | −0.005 / 0.000 |

Reading: Damped dominates Observed everywhere that matters and never loses to it by more than
1e-3; it matches Fisher where Fisher is good and removes Fisher's far-from-optimum blow-ups. The
two reversals Fisher had (StudentT on light-tailed data, protein-T) flip to wins. No Damped fit
diverged or produced non-finite rows (280 fits). The remaining failure (power_plant-T stalls at
RMSE ≈ 448 in all modes) is the saturating t score far from the data, i.e. the start-value
defect (proposals/003), not a curvature issue.
Decision: **retain `HessianMode::Damped` as the experimental recommendation**; keep defaults
unchanged pending Felix's review. Fisher stays available for the literature comparison.

### Cycle 1 — fitted start values × 3 modes (tags c1d_*) and a large-ν StudentT instability (2026-11-09) **[measured]**
With the start-value fix, power_plant trains in every mode (η .05/d4: Observed 3.04/2.98,
Fisher 2.84/2.81, Damped 2.81/2.83 for G/T). Ranking unchanged: Damped/Fisher beat Observed on
kin8nm (−0.08/−0.10), naval (−0.35/−0.41), protein (−0.05/−0.05), synth_t3, gaussian_sim;
Damped vs Fisher within noise except kin8nm-G (−0.011 ± 0.006), naval-G (−0.028 ± 0.020),
protein (−0.015/−0.012 ± 0.004) at η .05 — and the reverse (+0.013/+0.018) on protein at η .1.
Observed's PIT-KS is consistently the worst (naval 0.10–0.12 vs 0.05–0.08).

NEW failure class, **all modes, StudentT only, only with fitted starts**: cells with every test
row non-finite at best_iter 0 (kin8nm F s0, D s1, O s4; synth_t3 F s2, D s2; power_plant D s0,s3,
O s2; protein O s1), plus seeds where ν collapses to exactly 2 (ExpDf floor) and one with
ν = 1.5e16 and NLL −111. Cause (probe (since removed; findings recorded here)): on near-Gaussian targets the
unconditional MLE sends ν → 1e5…1e8 (raw ExpDf margin 11–18). The df score is a difference of
O(1) digamma/log terms whose true value is O(1/ν²), and the ExpDf chain multiplies by e^η ≈ ν —
so f64 cancellation noise becomes O(1)-to-O(100) per row with random sign, and the df margin is
blown to ±∞ by the first leaf. `log_prob` itself is also garbage for ν ≳ 1e12 (lnΓ differences
of ~1e17-sized numbers). statrs' t CDF/PDF are fine to 1e9 (checked). Not a curvature issue;
a parameterisation/precision issue that correct start values now expose. Candidates: (a)
cancellation-safe large-ν branch for the StudentT score/Hessian/log_prob (asymptotic
expansions), (b) cap the df start value (e.g. ν₀ ≤ 1e3) — the unconditional ν is irrelevant
once trees condition on x. Harness now reports Gaussian-limit CRPS/PIT for ν > 1e6 and counts
non-finite PITs as failures.

### Cycle 1 — start-value pre-phase, final form (2026-11-09) **[measured]**
Three failure modes were found and fixed while validating proposal 003 on StudentT:
1. **nansum trap**: a step that makes EVERY row's log-prob NaN (ExpDf overflow, ν = ∞) scores
   cost 0 and Armijo accepts it as a huge improvement. Guard: a step may not reduce the number
   of finite rows, and transformed params must be finite.
2. **flat-direction catapult**: with curvature ≈ 0 on the df coordinate the Newton step is
   arbitrarily long; a slightly lower cost at raw η_df = −31 (ν = 2 + e^−31, the ExpDf floor)
   was accepted and the gradient died there. Fix: trust radius on the raw-space step (∞-norm,
   start 2, ×2 on full-step acceptance, shrink to the accepted step otherwise).
3. **FD curvature noise floor**: once σ has been inflated to ≈ 4e5 the true location curvature
   n/σ² ≈ 1e-8 is far below the second-difference noise (≈ ε_mach·|c|/ε² ≈ 1e-3), so the loc
   crawled at 0.03/iteration. Fix: use the distribution's analytical per-row gradients and
   Hessians summed over rows (exact diagonal Newton, |h| for the sign), FD only as fallback.
Result: Gaussian and StudentT start values recover (ȳ, ln s_y) on targets at scales 7e-4…4.5e5;
100-iteration budget with stall stop; 215 lib tests green.
Also fixed: **StudentT large-ν branch** (`LARGE_DF = 1e8`): Gaussian-limit log-density and a
zero df score/curvature beyond it. Probe showed the df score noise-dominated past ~5e7 and
`log_prob` garbage past ~1e12 (ln Γ differences of 1e17-sized numbers); statrs' t CDF is fine.
**Open design point [untested]**: on Gaussian-like marginals the unconditional MLE sends ν₀ to
≈ 1e8, where the df score is (correctly) zero — the df tree then cannot learn conditional
heavy tails at all. A cap on the df start value (e.g. ν₀ ≤ 1e2–1e3) would keep that gradient
alive; this is an initialization design choice, proposed for cycle 2.
**Reproducibility note**: start values differed run-to-run before the fix (raw df 10.97 /
11.74 / 11.44 on identical data): `par_nansum`'s rayon reduction order is nondeterministic, and
on a likelihood flat in ν the optimizer's endpoint follows the rounding noise. Deterministic
chunked reduction would fix seeded reproducibility; not changed this cycle.

### Cycle 1 — FINAL benchmark on the patched crate (tags c1e_*, fitted start values, 3 modes) **[measured]**
420 fits, **0 non-finite rows, 0 errors, 0 divergences in Fisher/Damped**; Observed still has
isolated bad seeds (power_plant G/T seed 3: 3.9/4.2 vs ~2.7). ΔNLL Damped − Observed (η .05/d4 ;
η .1/d6), all 5/5 unless noted: kin8nm −0.076±0.013 ; −0.174±0.021 · naval −0.35±0.04 ;
−0.94±0.10 · power_plant −0.23±0.16 ; −0.32±0.23 · protein −0.048±0.005 ; −0.043±0.010 ·
synth_t3 (G/T) −0.015/−0.008 ; −0.060/−0.025 · gaussian_sim −0.006/−0.006 ; −0.031/−0.031 ·
studentt_sim ≈ 0 (±0.001). CRPS moves the same way (protein −0.11/−0.13). Damped − Fisher is
within ±0.03 everywhere the df stayed frozen; Damped's PIT-KS is the lowest or tied in 12/14.

**New finding — the ExpDf dead zones.** With fitted starts, ν₀ ≈ 2e7–1e9 on every real target
(unconditional MLE on Gaussian-like marginals). In Observed/Damped the df tree then never moves
(score ≈ 0) and the StudentT cells equal the Gaussian cells to 4 decimals. In the Fisher cells
on protein and power_plant the df margin instead ran to −∞: predicted ν = 2.000 (ExpDf floor)
for EVERY test row, in 10/10 seeds. Mechanism: near ν = 2 the chain factor dν/dη = ν−2 → 0
makes both score and curvature vanish; with Σh ≈ 0 the leaf is −ηΣg/λ, unbounded, and once the
margin is far negative nothing can pull it back. On protein's heavy-tailed residuals that
degenerate t STILL beats the frozen-Gaussian Damped/Observed fits by 0.15 ± 0.01 (5/5) and by
0.20 at η .1 — i.e. **a learnable df is worth ≥ 0.15 nats on protein and the frozen df is
costing it**. (Why only Fisher moved is unresolved: start values are nondeterministic, so the
Fisher runs may simply have drawn ν₀ below the freeze; the bench cannot tell.) Two follow-ups:
(a) cap ν₀ so the df tree keeps a gradient — tested next (tag c1f); (b) guard the ν → 2 run-off
(curvature floor on the df column or a lower bound on the raw margin) — cycle 2.

### Cycle 1 — start-df cap experiment (tags c1f_cap100_*, StudentT, fitted starts, ν₀ ≤ 100) **[measured]**
Prediction was "a live df tree recovers protein's ≥ 0.15 nats". **Falsified / neutral.** With
ν₀ = 100 the df tree does move — but UP, to ν ≈ 1e3 (kin8nm, naval, power_plant, protein; Damped
and Observed alike) — and test NLL is unchanged within noise: protein-Damped −0.032 ± 0.015
(4/5), power_plant-Observed −0.13 ± 0.13 (noise), naval-Damped +0.035 ± 0.015 (0/5, worse),
kin8nm ±0.003, simulated datasets identical (their ν₀ was already < 100). Calibration unchanged.
Fisher cells are bit-identical to uncapped: Fisher's df runs to the ν = 2 floor from ANY start
on protein/power_plant, and that degenerate fit is still the best NLL there (2.68–2.75 vs
2.87–2.96, 0.1–0.2 nats). Open puzzle for cycle 2: with identical gradients, Damped/Observed
drift ν upward while Fisher collapses it downward on the same data — the difference must come
from the df Hessian magnitude (Fisher's I_νν(ν−2)² is tiny ⇒ large steps) interacting with the
loc/scale trajectory and early stopping; the ν = 2 result suggests Observed/Damped under-exploit
heavy tails, but a run-off to the floor is not an acceptable mechanism either.
Decision: `with_max_start_df` stays as an experimental, non-default knob; **not recommended**.

---

## Cycle 2 — 2026-11-09 — StudentT df dynamics and the ExpDf dead zones

Felix approved `HessianMode::Damped` as the default for Gaussian and StudentT (done; 216 lib +
129 integration tests green; README breaking-change notes added) and asked to continue with the
next investigation.

### Question
With fitted starts on protein/power_plant, Fisher's df margin runs to the ExpDf floor (ν = 2 on
every row) and that fit is the best NLL by 0.1–0.2 nats; Observed/Damped (identical gradients)
drift ν upward from a capped start of 100 to ≈ 1e3 and never get there. Why, and what is the
right fix?

### Pre-registered hypotheses (written before looking at the trajectories, tag c2a_traj_R*)
- **H-a (step size)**: Damped's df curvature (observed data terms, larger than Fisher's tiny
  `I_νν(ν−2)²`) makes the df leaf steps small; the df simply moves too slowly and early stopping
  (driven by loc/scale progress) fires first. Prediction: with no early stopping and enough rounds
  (R = 400–800) Damped's ν eventually turns downward and its NLL approaches Fisher's.
- **H-b (direction flips with σ)**: at ν₀ with σ fitted as the *Gaussian* sd (too large for a
  heavy-tailed fit), ∂NLL/∂ν at the current σ favours *larger* ν; only once the scale tree has
  shrunk σ does the preferred ν fall. Fisher's large df steps overshoot through this phase;
  Damped tracks the slowly moving optimum upward first. Prediction: Damped's df trajectory is
  non-monotone (up, then down) and its turn coincides with σ decreasing; Fisher's ν is already
  near 2 by R ≈ 20–50.
- **H-c (irreversible run-off)**: once ν is small, `dν/dη = ν−2 → 0` makes score and curvature
  vanish together and the leaf `−Σg/(Σh+λ)` is governed by λ alone; the margin then runs to −∞
  and cannot return. Prediction: in the Fisher cells df_q05 = df_med = 2.000 exactly (margin
  ≤ −40) rather than a finite small ν such as 2.3; NLL would be *better still* at a finite ν
  near 2–3 (checkable later by clamping the margin).
- **H0**: differences are seed/split noise. Prediction: the two seeds disagree in direction.
Candidate fixes, to be chosen by the evidence: curvature floor on the df column (bounded step),
a df link without the (ν−2) factor, or longer patience for the df.

### Cycle 2 — trajectories (tags c2a_traj_R1…R800; StudentT, fitted starts, no early stopping, 2 seeds) **[measured]**
| | Fisher | Damped / Observed |
|---|---|---|
| df (protein, power_plant, kin8nm) | **ν = 2.000 on every row already at R = 1**, and stays there | frozen at ν₀ (2e7–1e9) for all 800 rounds; moves by < 1% |
| protein NLL (R=50/100/200/400) | 2.92 / 2.84 / 2.78 / 2.78 | Damped 2.98 / 2.91 / 2.90 / 3.41 ; Observed 2.97 / 2.93 / 3.38 / 4.39 |
| power_plant NLL (R=50/100/200) | 2.89 / 2.75 / 2.71 | Damped 3.16 / 2.78 / 2.70 ; Observed 2.95 / 2.75 / 2.73 |
| kin8nm NLL (R=200/400/800) | −0.46 / −0.30 / 1.6 | Damped −0.46 / −0.34 / 2.5–33 ; Observed −0.37 / 0.4–3.8 / 11–4e4 |
| mean σ̂ (protein R=1→800) | 4.8 → 2.5 | 6.1 → 3.5 |

Verdicts on the pre-registered hypotheses: **H-a rejected** (Damped's df does not move slowly — it
does not move at all; its score at ν₀ ≈ 1e7 is ≈ 0). **H-b rejected** (no non-monotone df path;
σ falls monotonically in every mode). **H-c half right**: the run-off is irreversible, but it is
not a gradual slide — it is ONE first-round leaf. **H0 rejected** (seeds agree to 3 decimals).
Second finding, separate: without early stopping every mode eventually diverges on the test set
(kin8nm all modes by R = 800; protein Damped/Observed by R = 400) while mean σ̂ keeps shrinking and
RMSE keeps improving — scale overfitting (overconfidence), Observed earliest/worst, Fisher latest.
Early stopping at ≈ 200 rounds is what keeps the cycle-1 numbers sane; a scale-tree regulariser
(cf. β-NLL, Seitzer et al. 2022) is a candidate for a later cycle.

**New hypothesis H-d (pre-registered before the next probe)**: at ν₀ ≈ 1e7 the per-row df score
`grad_ν·(ν−2)` is digamma/ln cancellation noise × 1e7, i.e. O(1e-2–1e-1) per row with a common
sign, so |Σg_df| is O(1e2–1e3) on 5–12k rows. Fisher's Σh_df = Σ max(I_νν(ν−2)², 1e-12) ≈ 0, so the
root leaf is −η·Σg/λ ≈ −(10…50) raw units in ONE round → ν = 2 + e^(η−30) = 2.000. The observed
Σh_df is the same noise × (ν−2)² ≈ O(1e2–1e4) and (empirically) positive, so Damped's leaf is
≈ −η·Σg/Σh ≈ 0 → frozen. With ν₀ capped at 100 (c1f) Fisher still collapsed, so there the
gradient must be real and large relative to I_νν·98²·n — prediction: Σg_df(ν=100) > 0 on protein
(heavy tails want smaller ν) and |Σg|/(Σh_F+1) ≳ 100 → a 5-unit raw step at η = 0.05 → ν ≈ 2.7
after round 1, then run-off. Probe: `examples/research_probe_df_step.rs`.

### Cycle 2 — H-d rejected; the "collapse" is in the START VALUES, not in boosting (2026-11-09) **[measured]**
`research_probe_df_step.rs`: at the fitted ν₀ ≈ 1e7–9e7 the whole-column df sums are tiny in every
mode (|Σg| ≈ 1e-4, root step ≈ +4e-6 at η = 0.05, λ = 1); at ν₀ = 100 the df gradient points UP
(Σg < 0 → ν rises; Fisher +0.34/round, Damped +0.04/round) because σ fitted as the Gaussian sd
makes the data look light-tailed. No first-round tree can move the df margin by −28 units.
`BENCH_DEBUG` on protein, R = 1: **start values differ by Hessian mode** —
Fisher: raw (−29.39, 5.92, 1.57) ⇒ ν = 2.000, σ = 4.8; Damped: raw (31.15, 7.69, 1.80) ⇒
ν = 3.4e13, σ = 6.1. Cause: the cycle-1 pre-phase uses `analytical_gradients`, which honours
`hessian_mode`. Fisher's tiny df curvature moves ν FIRST (downward — at the initial σ = 1.65
the residuals look heavy-tailed); Damped's larger df curvature lets σ inflate first, after
which the df gradient reverses and ν runs to +∞. Two different local optima of the
unconditional t likelihood; the boosting phase then never leaves either (df score ≈ 0 at ν = ∞;
dν/dη ≈ 0 at ν = 2). Same explains c1f: the cap only clamps from above.
Consequences: (1) start values must not depend on the training-curvature choice — pre-phase
curvature should be a fixed, well-conditioned choice; (2) the unconditional t likelihood in ν is
multimodal/flat on real targets, so a 1-D ν scan (profile over loc/scale) is the principled
start — cheap, StudentT-specific; (3) the 0.1–0.2-nat "Fisher advantage" on protein/power_plant
was the better start mode, not the curvature.

### Cycle 2 — unconditional ν profile and hypothesis H-e (2026-11-09)
**[measured]** Profile NLL over ν with (loc, σ) re-optimised (scipy, Nelder–Mead), n = 12000/9568/8192:
protein 40286 (ν=2) → 39058 (10) → 38757 (100) → 38724 (1e8); power_plant 42070 → 40981 → 40747 →
40722; kin8nm 1527 → 829 → 712 → 701. **Unimodal, monotone: ν → ∞ is the unconditional MLE on all
three.** So Fisher's ν = 2 start was not a better mode but a pre-phase failure (df ran into the
ExpDf dead zone before σ adapted; irreversible since dν/dη → 0). Fixed in base.rs by a relative
curvature floor `h_i ≥ 1e-3·max_j h_j` in the pre-phase (+ test that all three HessianModes give
the same start values).
**The real design flaw [derived from the above]**: the unconditional MLE (ν = ∞, df score ≡ 0)
freezes the df tree, yet the CONDITIONAL optimum has finite ν — a ν = 2 start beat the ν = ∞ start
by 0.12–0.20 nats on protein/power_plant after boosting even though it was the worse unconditional
fit. The c1f cap at ν₀ = 100 was neutral because at ν = 100 the df information is already ~1e-6/row
and the gradient (at the unconditional σ) points up; the df drifted to 1e3 and froze again.
**H-e (pre-registered)**: a MODERATE start cap (ν₀ ≤ 10) keeps the df score alive during boosting so
the df tree can follow the conditional tails in either direction. Predictions at η=.05/d4, Damped,
5 seeds: protein ≤ 2.80 (vs 2.90 uncapped; ν=2-start gave 2.75), power_plant ≤ 2.75 (vs 2.81),
synth_t3 unchanged (its ν₀ is already < 10), kin8nm/naval/gaussian_sim worse by ≤ 0.01 (df must
climb; information at ν = 10 is ~1e-3/row so it can). Also test ν₀ ≤ 5 and ≤ 30 (dose–response).
Prior art (from memory, not retrievable via Lacuna): Lange, Little & Taylor (1989) note ν is
poorly determined and often fix it; Fernández & Steel (1999) analyse the t likelihood's
pathologies in ν. A fixed moderate ν (4–10) is common robust-regression practice.

### Cycle 2 — H-e results: ν₀ cap dose–response (tags c2b_cap{none,30,10,5}; StudentT, Damped, fitted starts, η=.05/d4, 5 seeds) **[measured]**
ΔNLL vs uncapped (uncapped ν₀ now ≈ 1e5–2e6 after the pre-phase fix, df score alive but ~0):
| dataset | cap30 | cap10 | cap5 | note |
|---|---|---|---|---|
| protein | −0.047 ± 0.024 (4/5) | **−0.067 ± 0.015 (5/5)** | −0.055 ± 0.022 (4/5) | CRPS −0.10 to −0.11; df_q05 ≈ 3.4–5, df_med 240–580 → conditional heavy tails learned for a subset of rows; PIT-KS 0.082 → 0.070 |
| power_plant | −0.059 ± 0.045 (5/5) | −0.052 ± 0.053 | −0.053 ± 0.042 | noisy (uncapped has a bad seed); df_q05 13–21 |
| kin8nm | +0.003 ± 0.002 | −0.001 ± 0.004 | +0.007 ± 0.008 | df climbs to ~2e3 in all; neutral |
| naval | +0.064 ± 0.037 (0/5) | −0.010 ± 0.017 | +0.012 ± 0.007 | mixed; df climbs to 2.8e3 |
| sims | 0 | 0 | 0 / −0.0015 | ν₀ already below the caps |
Verdict: H-e confirmed where it matters (protein 4.5 s.e., all caps; power_plant directionally),
neutral on light-tailed targets. Prediction "protein ≤ 2.80" NOT met (2.83): the global ν = 2
start (c1e Fisher, 2.75) still wins, so the df tree moves too slowly toward heavy tails.
Decision: `with_max_start_df` stays **non-default**; recommend ν₀ ≤ 10 for suspected heavy tails.
Not made default because it is not better on all datasets (naval cap30 0/5).

**Why the df is slow [derived from probe + trajectories]**: σ₀ and ν₀ come from the MARGINAL fit;
once loc(x) explains variance σ₀ is far too large, the df gradient at that σ points UP (probe:
Σg < 0 at ν = 10/100), ν climbs to ~300 where I_νν·(ν−2)² ≈ 1e-5/row, and only after the scale
tree has shrunk σ does the sign flip — by then the df is nearly frozen again. Start values
should come from CONDITIONAL residuals.
**H-f (pre-registered)**: two-stage residual warm start using the crate's per-row offsets —
stage 1: Gaussian(Damped) fit → ŷ(x) on train/valid/test; stage 2: t-fit on train residuals
→ (ν₀, σ₀); StudentT trained with `initialize: false` and offsets [ln(ν₀−2)−0.5, ŷ(x)−0.5,
ln σ₀−0.5] (0.5 cancels the default start), predicted with the same offsets. Predictions
(Damped, η=.05/d4, 5 seeds): protein ≤ 2.76 (beats cap10's 2.83, matches the ν=2 start),
power_plant ≤ 2.72, kin8nm/naval no worse than cap10 by > 0.01, synth_t3/studentt_sim
unchanged or better (their ν₀ is already conditional-ish). Risk: in-sample residuals understate
σ₀ (stage-1 overfit) → early over-confidence; mitigated by early stopping on the validation set.

### Cycle 2 — H-f results: residual warm start (tags c2c_warm, c2c_warm_cap10; StudentT, Damped, η=.05/d4, 5 seeds) **[measured]**
ΔNLL warm − marginal start (c2b_capnone) and warm − cap10:
| dataset | NLL warm | vs marginal | vs cap10 | note |
|---|---|---|---|---|
| protein | **2.770** | −0.124 ± 0.014 (5/5) | −0.057 ± 0.013 (5/5) | CRPS −0.146; PIT-KS 0.082→0.066; df_q05 3.6, df_med 74; ≈ the ν=2 start's 2.75 |
| power_plant | **2.714** | −0.130 ± 0.079 (5/5) | −0.078 ± 0.029 (5/5) | prediction ≤ 2.72 met; df_med 15 |
| naval | −5.864 | −0.114 ± 0.026 (5/5) | −0.104 ± 0.012 (5/5) | **df = 2.000 on every row** — the ExpDf dead zone entered via the residual fit; NLL "better" but a degenerate state (see debug below) |
| kin8nm | −0.498 | −0.008 ± 0.006 (3/5) | −0.007 ± 0.007 | neutral+; df_med 35 |
| synth_t3 | 1.398 | −0.016 ± 0.002 (5/5) | same | ν₀ ≈ 3.1 recovered (true 2.5–5.5) |
| studentt_sim | 1.976 | −0.004 ± 0.001 (5/5) | same | |
| gaussian_sim | 2.134 | **+0.018 ± 0.002 (0/5)** | same | ν₀ ≈ 3.4 on Gaussian data: in-sample stage-1 residuals are leptokurtic (overfit), the t-fit reads tails that are not there and boosting does not push ν back up |
Predictions: protein ≤ 2.76 → 2.770 (borderline), power_plant ≤ 2.72 ✓, kin8nm/naval not worse ✓,
synth_t3/studentt_sim better ✓; the stated risk (in-sample residuals understate/peak) materialised
on gaussian_sim and (as a dead-zone entry) on naval. Decision pending the honest-residual variant
(validation residuals for the t-fit) — tag c2d.

### Cycle 2 — honest-residual warm start (tags c2d_warmvalid, c2d_warmvalid_cap10) **[measured]**
(ν₀, σ₀) fitted on VALIDATION residuals of the stage-1 Gaussian. ΔNLL vs the marginal start
(c2b_capnone), StudentT, Damped, 5 seeds:
| dataset | NLL | Δ vs marginal | Δ vs Gaussian-Damped | + cap10 | note |
|---|---|---|---|---|---|
| kin8nm | −0.511 | **−0.020 ± 0.002 (5/5)** | −0.022 ± 0.004 | +0.002 | (train-residual version was only −0.008) |
| naval | −5.868 | −0.118 ± 0.025 (5/5) | −0.125 | 0 | still ν₀ = 2.000 → dead-zone start (floor added, see below) |
| power_plant | 2.717 | −0.127 ± 0.077 (5/5) | −0.088 ± 0.039 | −0.003 | |
| protein | 2.807 | −0.087 ± 0.026 (5/5) | −0.085 | **−0.028 (→ 2.779)** | train-residual version 2.770 |
| synth_t3 | 1.397 | −0.017 ± 0.002 (5/5) | −0.070 | 0 | |
| studentt_sim | 1.976 | −0.004 ± 0.001 (5/5) | −0.053 | 0 | |
| gaussian_sim | 2.133 | +0.017 ± 0.005 (0/5) | +0.035 | 0 | ν₀ ≈ 2.3–3.6 |
gaussian_sim is a legitimate, informative loss: its CONDITIONAL law is Gaussian but the residual
MARGINAL is a scale mixture (σ(x) ∈ [1, 5]) and therefore leptokurtic — a small ν₀ IS the correct
unconditional fit, and the df tree then climbs too slowly (df_med 3.6 at the early-stop round
60). Same limitation as the capped start, from the other side. Validation residuals did not fix
it (the kurtosis is real, not overfit), but they did fix kin8nm (−0.020 vs −0.008) and made
protein slightly worse than in-sample residuals (2.807 vs 2.770) — in-sample residuals' extra
kurtosis happened to help there. Calibration: PIT-KS improves on protein (0.082→0.070), synth_t3,
unchanged elsewhere; 90% coverage 0.81–0.90 (kin8nm under-covers in every recipe).
**Decision**: residual warm start (validation residuals, ν₀ ≤ 10) is the recommended StudentT
recipe: better than the marginal start on 6/7 datasets by 0.004–0.13 nats, and better than the
Gaussian-Damped default on 6/7; the one loss (+0.017) is on exactly-Gaussian conditional data.
Kept at harness level (`BENCH_WARM_START=1 BENCH_WARM_RESID=valid BENCH_T_START_DF_CAP=10`);
crate API proposed in proposals/004 for Felix.
Crate fix landed: `StudentT::calculate_start_values` floors the df START at ν₀ = 2.5 (raw ln 0.5)
— a start exactly at the ExpDf floor is a dead state (dν/dη ≈ 0) the booster can never leave.

### Cycle 2 — start floor re-check (tag c2e_floor_warmvalid_cap10) **[measured]**
naval: NLL −5.866 (unchanged) but df_med 10–23 / df_q05 3.4–5.6 instead of 2.000 everywhere — the
warm-start gain stands on a learned, non-degenerate df. gaussian_sim 2.125 (was 2.133), df_med
3.4–7.6. Floor kept. Cycle 2 closed; report: reports/003-cycle2-report.md.

---

## Cycle 3 — 2026-11-09 — productising the warm start; df step size; scale overfitting

Felix: keep the ν₀ ≤ 10 cap as the StudentT default (better/neutral everywhere), implement the
warm start as a crate feature, then continue with df step size and scale-overfitting control.

### Landed in `src/`
- `StudentT::max_start_df` defaults to `Some(10.0)` (`without_max_start_df()` opts out);
  `MIN_START_DF_RAW` floor (ν₀ ≥ 2.5) stays.
- `GradientLSS::train_with_residual_warm_start(train, valid, params, config, ResidualWarmStart)`:
  stage-1 Gaussian (same params, 2 targets) → ŷ(x); `dist.calculate_start_values` on validation
  residuals (`ResidualSource::{Validation, Train}`) → `preset_start_values`; stage 2 trained with
  the stage-1 location as the `loc`-column offset. `predict*` recompute the offsets from the
  stored stage-1 model; `save` writes a v2 layout (sentinel `u64::MAX`, version, loc index,
  length-prefixed stage-1 blob, then the legacy layout) and `load` accepts both layouts.
  Requires a univariate family with an Identity `"loc"` parameter (Gamma etc. → error).
- Trait additions: `BackendDataset::get_features()`, `BackendParams::set_n_dist_params()`.
- Tests: auto-offset prediction ≡ manual stage-1 offsets (bitwise), save/load round-trip with the
  stage-1 blob, legacy layout unchanged for plain models, Gamma rejected. README updated.
- Benchmark of the API path: tag c2f_api (pending at the time of writing; expected ≡ c2d_warmvalid_cap10).

### Next question (H-g, pre-registered): why does the df climb so slowly from ν₀ ≈ 3 on gaussian_sim?
Candidates: (a) early stopping fires at ≈ round 60 on the loc/scale plateau while the df is still
moving (prediction: with no early stopping, df_med rises with R and test NLL keeps improving past
R = 60); (b) the df gradient is genuinely small because the scale tree has not yet matched σ(x)
∈ [1, 5] — with σ too large for some rows and too small for others the per-row df gradients
cancel (prediction: df_med stalls even at R = 400 while NLL plateaus); (c) the Damped df curvature
is too large relative to the gradient (prediction: Fisher-mode df would move faster at equal R).
Diagnostic: tags c2g_apitraj_R{5…400}, gaussian_sim + protein, 2 seeds.

### Cycle 3 — API path check and H-g trajectories **[measured]**
- `train_with_residual_warm_start` (tag c2f_api) reproduces the manual harness recipe
  (c2d_warmvalid_cap10): identical NLL on kin8nm, power_plant, studentt_sim, synth_t3; naval
  +0.003 and gaussian_sim −0.008 (those two now start at the ν₀ ≥ 2.5 floor). The crate feature is
  the recipe. (protein still running at the time of writing.)
- H-g (tags c2g_apitraj_R5…400, warm start, no early stopping, 2 seeds):
  gaussian_sim df_med 2.9 → 4.6 → 7.8 → 10.7 → 26 → 31 (R = 5…400) — the df DOES climb; but test NLL
  is minimal at R ≈ 50 (2.119) and then rises to 2.37 (R = 200) and 3.1 (R = 400) while 90 % coverage
  falls 0.91 → 0.88 → 0.80 → 0.71 and mean σ̂ keeps drifting. protein: NLL minimum at R ≈ 100–200
  (2.81), df_q05 → 3 (heavy tails learned for a subset), coverage 0.96 → 0.81.
  Verdict: candidate (a) "early stopping fires too early" is wrong in the sense that it fires
  correctly for the total NLL — the SCALE tree over-fits (overconfidence) before the df tree has
  moved far enough, so the stopping round is set by the scale and the df is left at ν ≈ 8. (b) is
  rejected (df moves monotonically; no sign cancellation). (c) untested. The remedy is the one in
  proposals/005: slow the scale and/or speed the df — per-parameter shrinkage.

### Cycle 3 — proposal 005 results: per-parameter shrinkage (tags c3a_g_*, c3a_t_*; Damped, η=.05/d4, 5 seeds) **[measured]**
Gaussian, scale-gradient multiplier c (loc 1): ΔNLL vs c = 1
| dataset | c = 0.5 | c = 0.25 | Δcov90 (0.25) | Δbest_iter (0.25) |
|---|---|---|---|---|
| kin8nm | −0.036 ± 0.007 (5/5) | **−0.081 ± 0.006 (5/5)** | +0.037 (0.82 → 0.86) | +143 |
| naval | +0.020 ± 0.029 | **−0.107 ± 0.025 (5/5)** | +0.021 | +124 |
| power_plant | −0.003 ± 0.011 | **−0.046 ± 0.007 (5/5)** | −0.003 | +413 |
| protein | −0.007 ± 0.004 (4/5) | −0.022 ± 0.010 (4/5) | −0.005 | +239 |
| gaussian_sim | +0.007 ± 0.000 (0/5) | +0.017 ± 0.001 (0/5) | −0.010 | +178 |
| synth_t3 | +0.001 | +0.008 ± 0.002 (0/5) | +0.001 | +132 |
| studentt_sim | 0.000 | +0.002 ± 0.001 (0/5) | 0.000 | +175 |
Prediction 1 confirmed on real data (NLL and coverage improve, more rounds used); prediction 2
FALSIFIED (0.25 beats 0.5 on every real target); the simulations lose because their σ(x) carries
strong, quickly learnable structure and nothing over-fits within the early-stopping window.
StudentT (API warm start): scale ×0.5 helps kin8nm/naval (−0.014/−0.016, 5/5) and is neutral or
+0.001…+0.005 elsewhere; df ×2 moves gaussian_sim by only −0.002 → prediction 3 FALSIFIED: the df
step size is not what limits the heavy-start case; the scale tree's over-fitting timing is.
**Validation-selected c** (per dataset × seed by valid NLL among {1, .5, .25}): picks c = 1 on all
three simulations (Δ = 0.000) and c = 0.25 on all four real targets (full gains, equal to the
per-seed oracle except protein 2.8705 vs 2.8661). StudentT: −0.017/−0.018 on kin8nm/naval,
≈ 0 elsewhere. → **Strictly non-worse as a tuned hyperparameter.**
Decision: `with_param_gradient_scales` ships as a knob (default None, README documents it);
recommend tuning the scale factor on validation NLL over {1, 0.5, 0.25} — a natural addition to
`hyper_opt`'s search space (not wired in this cycle). Not a default: dataset-dependent sign.

### Cycle 3 — implementation review of per-parameter shrinkage, and new machinery (2026-11-09)
Felix asked for more scrutiny of the shrinkage implementation.
- **[measured]** On the real XGBoost booster (test `multiplier_scales_one_round_update_exactly`):
  with scales [1, 0.5] the one-round raw update of the scale column is exactly 0.5× the unscaled
  one on every row (ratio error < 2e-3, f32 booster) and the location column's update is
  bit-identical — one tree per target, so a scaled column never enters another column's split
  search. Start values are unaffected. (With `multi_strategy=multi_output_tree` the trees would
  be shared and this separation would NOT hold — documented caveat.)
- **[literature, verified at source]** gamboostLSS (boost-R/gamboostLSS, R/mboostLSS.R,
  R/helpers.R, man/mboostLSS.Rd): `nu` "a (named) vector or a (named) list with separate values for
  each component", recycled from a scalar, applied per component via `control$nu <- nu[[j]]` before
  each `mboost` fit (the multiplication onto the fitted negative gradient lives in mboost's update);
  `stabilize_ngradient` divides the NEGATIVE GRADIENT only (MAD floored 1e-4; L2 clamped
  [1e-4, 1e4]) — there is no Hessian in gamboostLSS; the noncyclic method advances every
  parameter tentatively and keeps the one with the smallest inner loss; the cyclic method takes a
  per-parameter `mstop` (e.g. `mstop = list(mu = 400, sigma = 300)`). So: our `c_j` is the Newton
  analogue of gamboostLSS's `nu_j`, and our MAD/L2 "stabilization" inherited a gradient-only
  recipe into a Newton method (it rescales g and h by different constants) — which is why it broke
  (cycle 1).
- New machinery landed (all tested): deterministic `par_sum`/`par_nansum` (fixed 1024-blocks,
  thread-count invariant — fixes start-value reproducibility); Gamma `HessianMode` (Fisher
  α²ψ′(α), α under Exp; MC E[observed] = Fisher; default Damped pending benchmark c3b_gamma);
  β-NLL row weighting `with_beta_nll(β)` on Gaussian/StudentT (rows × σ̂^{2β} on g and h);
  per-parameter freeze schedule `GradientLSS::set_param_freeze_rounds` (gamboostLSS cyclic
  `mstop` per parameter; frozen column's g,h = 0 → null trees); `HyperOptConfig::trial_dist_fn`
  (per-trial distribution builder reading `__`-prefixed axes; start-value cache bypassed).
**Pre-registered predictions for c3b** (Damped, η=.05/d4, 5 seeds):
- Gamma: Damped ≥ Observed on kin8nm/power_plant/naval (the observed concentration curvature
  goes negative for large yβ); Fisher may blow up far from the start as it did for Gaussian.
- β-NLL (Gaussian): β = 0.5 improves location RMSE on heteroscedastic data (kin8nm, synth_t3,
  gaussian_sim) and is neutral-to-slightly-worse on NLL; β = 1 worse on NLL (MSE-like row
  influence) — the paper's own finding transfers if the mechanism is the split weighting.
- Scale freeze (Gaussian, freeze scale at 50/100/200): mimics the shrinkage gains on the real
  targets (kin8nm/naval/power_plant/protein) with the right round; the simulations prefer no
  freeze; validation-selected freeze ≈ validation-selected shrinkage.

### Cycle 3 — c3b results: Gamma curvature, β-NLL, scale freeze (Damped, η=.05/d4, 5 seeds) **[measured]**
**Gamma `HessianMode`** (kin8nm, naval; power_plant failed in every mode — all test rows
non-finite, under investigation): Damped − Observed: kin8nm −0.021 ± 0.005 (5/5); naval
**−2.88 ± 0.001** (Observed −1.08 vs Damped −3.96: the negative observed concentration curvature
cripples it). Damped − Fisher: −0.006 / −0.0003 (tie). PIT-KS best for Damped. → Damped default
for Gamma stands (prediction confirmed).
**β-NLL (Gaussian)**, ΔNLL / Δrmse vs plain: β = 0.5: kin8nm −0.024 ± 0.003 (5/5) / −0.005;
protein −0.021 ± 0.011 / **−0.28**; power_plant 0.000 / **−0.18**; naval −0.022 ± 0.031; sims
+0.002…+0.003 (0/5). β = 1: protein −0.049 ± 0.009 (5/5) / −0.32; kin8nm −0.029; power_plant
−0.012 / −0.25; sims +0.002…+0.012; **naval +1.75 ± 0.04 (0/5, coverage → 1.0)**. Prediction
(RMSE gains on heteroscedastic data) confirmed and larger than expected; the NLL prediction held
except naval. Diagnosis of naval: the weights σ̂^{2β} are scale-DEPENDENT — naval's σ̂ ≈ 1e-3 gives
row weights ≈ 1e-6, the leaves collapse to ≈ 0 and the model never leaves its start values. The
paper's networks train with Adam (scale-invariant); a Newton booster needs the weights normalised
to mean 1 per round. Fix: normalise. StudentT (warm start) β = 0.5: kin8nm −0.020 (5/5), naval
−0.040 (5/5), others ≈ 0.
**Scale freeze at round k (Gaussian)** vs none: kin8nm k=100 −0.116, **k=200 −0.157 ± 0.010 (5/5)**,
cov90 +0.07…+0.16; power_plant k=100 **−0.110 ± 0.049 (5/5)**, rmse −0.52; protein k=100
**−0.095 ± 0.007 (5/5)**, rmse −0.53; naval any k catastrophic (+0.3…+1.4; its tiny structured σ
must keep learning); sims ≈ 0 at k ≥ 100, k=50 too early. RMSE improves when the scale is frozen
because early stopping is no longer triggered by the scale's over-fit and the LOCATION keeps
training. **Validation-selected k ∈ {none, 50, 100, 200}**: kin8nm −0.157 (5/5), power_plant
−0.116 (5/5), protein −0.098 (5/5), naval 0.000 (picks none ×5), sims 0.000 — strictly non-worse
and **about twice the gain of validation-selected shrinkage** (−0.081 / −0.046 / −0.022).
Prediction confirmed; magnitude exceeded. This is gamboostLSS's per-parameter `mstop`.
Decisions: Gamma Damped default kept (debug power_plant first); β-NLL weights to be normalised
and re-run; freeze rounds to move into `TrainConfig` so `hyper_opt` can tune `__freeze_<j>`.

### Cycle 3 — c3c: Gamma re-run and β-NLL with normalised weights (Damped, η=.05/d4, 5 seeds) **[measured]**
- Start-value cost guard (`min_finite_rows`, base.rs): Gamma on power_plant now starts at
  (α ≈ 328, β ≈ 0.72, mean 455) instead of (e^−4356, e^6021); 0 non-finite rows. Crate start values
  equal scipy's Gamma MLE on kin8nm to 4 decimals (α 6.2056, rate 8.6879, NLL 932.389) — the gap to
  the Gaussian (701) is family misfit, not the optimiser.
- Gamma power_plant: Damped − Observed **−0.408 ± 0.0004 (5/5)**, Damped − Fisher +0.001 (tie).
  Damped default for Gamma confirmed on 3/3 datasets. But the Gamma fit itself is poor there
  (NLL 4.11 vs Gaussian 2.81, 90 % coverage 1.000, PIT-KS 0.38): the conditional sd is ≈ 4 around
  450, i.e. α ≈ 12 000 is needed and the concentration tree (h_logα = α²ψ′(α) ≈ α, large) climbs
  too slowly from α₀ ≈ 330 before early stopping. Same theme as the StudentT df: a shape parameter
  started from the marginal and moving too slowly — cycle-4 item (Gamma warm start / shape step).
- β-NLL, normalised weights (ΔNLL / Δrmse vs β = 0), Gaussian: β = 0.5: kin8nm −0.019 ± 0.004
  (5/5) / −0.006; protein −0.029 ± 0.010 (5/5) / −0.21; naval +0.013 ± 0.033 (the +1.75 blow-up is
  gone); power_plant +0.033 ± 0.050 / **−0.19**; sims +0.0015…+0.005. β = 1: protein
  **−0.072 ± 0.007 (5/5) / −0.36**; kin8nm −0.019; power_plant +0.022 / −0.21; naval +0.059 ±
  0.032; sims +0.002…+0.008. StudentT (warm start) β = 0.5: kin8nm −0.010, protein −0.015 /
  rmse −0.11, power_plant rmse −0.13, else ≈ 0.
  Validation-selected β ∈ {0, .5, 1}: protein −0.068 ± 0.007 (5/5), kin8nm −0.024 ± 0.010 (5/5),
  naval −0.014 ± 0.015, power_plant +0.021 ± 0.038 (selection noise), sims +0.0003…+0.006.
  Verdict: β-NLL's reliable effect in Newton boosting is LOCATION accuracy on heteroscedastic
  real data (RMSE −0.2 to −0.36), as in the paper; the NLL effect is smaller and mixed. Keep as a
  knob (default None); it is complementary to, not a substitute for, the scale freeze.
Ranking of the three scale-side knobs by validation-selected gain (Gaussian): freeze (kin8nm
−0.157, protein −0.098, power_plant −0.116) > shrinkage (−0.081, −0.022, −0.046) > β-NLL
(−0.024, −0.068, +0.021); all ≈ 0 on the simulations and on naval.

---

## Cycle 4 — 2026-11-09 — Gamma: mean/shape (orthogonal) coordinates; then β-NLL × freeze; NB dispersion

### Measured limitation
Gamma on power_plant (cycle 3): 90 % coverage 1.000, PIT-KS 0.38, NLL 4.11 vs Gaussian 2.81; the
concentration must reach α ≈ 12 000 from the MLE start 330 and does not.

### Derivation [derived + verified numerically, scipy quad, 4 decimals]
(log α, log β) Fisher = [[α²ψ′(α), −α], [−α, α]]: correlation −0.9992 at α=330, i.e. α carries
the MEAN α/β. The diagonal-Newton step in log α at the right mean (E_y over the true law) is
0.0015 vs 3.59 needed (α 330 → 12 000) — ~2400× too slow; 0.084 vs 2.30 for α 5 → 50.
(log µ, log α), β = α/µ: Fisher = diag(α, α²ψ′(α) − α) EXACTLY (Cox & Reid 1987 parameter
orthogonality; not retrievable via Lacuna, cited from memory), with α²ψ′(α) − α → ½. The same
step is 0.97 (α 330 → 12 000) and −9.3 for α 50 → 5 (overshoot; η and Damped bound it).
This is the mechanism NegativeBinomial's `MeanDispersion` comment already records for (r, p).

### Implementation
`GammaParameterization::{ConcentrationRate (default), MeanShape}`, `Gamma::new_mean_shape`:
joint transform (outputs still (concentration, rate)), exact-MLE starts (µ̂ = ȳ; Newton on
ln α − ψ(α) = ln ȳ − mean ln y from Minka's start), analytic g/h with cancellation-safe series
for ψ(α) − ln α and α²ψ′(α) − α (α > 50), clamps with zeroed gradients outside. 4 tests: FD
derivatives (α 0.8 … 12 000), Monte-Carlo orthogonality and E[observed] = Fisher, MLE recovery,
output convention / serde default. Harness: `gamma_ms` family, closed-form Gamma CRPS (MC
self-test), `synth_gamma` DGP (ln µ = 3 + sin πx₁ + 0.5x₂, ln α = 4.2 + 3.4x₃ ⇒ α ∈ [2.2, 2000]).

### Pre-registered predictions (Damped unless stated, η=.05/d4, 5 seeds; tag c4a)
1. MeanShape − ConcentrationRate ΔNLL < 0 on synth_gamma, power_plant (large α) and naval by
   > 2 s.e.; power_plant 90 % coverage drops from 1.0 to within 0.05 of nominal.
2. kin8nm (α ≈ 6, modest) — small or zero gain (the slowdown factor ≈ 2α is small there).
3. Within MeanShape, curvature modes nearly tie (Fisher diagonal is exact here; observed ≈ Fisher
   except far from the fit); Observed may still be worst.
4. MeanShape needs far fewer boosting rounds (best_iter) for the same or better NLL.
Smoke (synth_gamma, 1 seed): ConcRate 3.110 / MeanShape 2.443 / oracle 2.317; PIT-KS 0.176 →
0.013; best_iter 859 → 173.

### Cycle 4 — NB dispersion: same pathology, partially solved already [measured, exact summation]
Expected-Fisher correlation of NB in the default raw coordinates (log r, logit p) vs (log µ, log θ):
µ=3,θ=1: +0.876 vs −2e-7 · µ=20,θ=4: +0.951 vs −2e-6 · µ=400,θ=20: +0.988 vs −1e-5 ·
µ=2000,θ=50: +0.995 vs +6e-6. So the default NB head (TotalCountProbs, Relu on r) has the Gamma
collinearity (worse with the mean), and `NbParameterization::MeanDispersion` is the orthogonal fix
that already exists — but (i) it is not the default, (ii) it CLAMPS µ ≤ e⁶ ≈ 403 and θ ≤ e⁸ ≈ 2981
(the gradient is still evaluated at the clamped point — flagged in the July review), so any count
target with conditional mean > 403 is silently capped, and (iii) Relu on r in the default kills
rows with margin ≤ 0. I_logθ → ½ as θ grows (same as Gamma's shape).
Plan (pre-registered, tag c4c, needs count DGPs since no local count data): synth_nb_lo
(µ ∈ [2, 60], θ(x) ∈ [0.5, 20]) and synth_nb_hi (µ ∈ [50, 5000], θ(x) ∈ [2, 200]); families nb
(TotalCountProbs default), nb_md (MeanDispersion as-is), nb_md_wide (clamps widened to µ ≤ e¹⁵,
θ ≤ e¹², gradients zeroed outside). Predictions: nb_md ≪ nb on both; nb_md ≈ nb_md_wide on _lo;
nb_md_wide ≪ nb_md on _hi (clamp bites at µ > 403); 90 % coverage of nb near 1.0 on _hi.

### Cycle 4 — c4a results: Gamma MeanShape vs ConcentrationRate (5 seeds × 2 settings) **[measured]**
ΔNLL MeanShape − ConcRate (both Damped), η.05/d4 ; η.1/d6:
| dataset | Δ NLL | ΔCRPS | cov90 CR → MS | PIT-KS CR → MS | best_iter CR → MS |
|---|---|---|---|---|---|
| power_plant | **−1.326 ± 0.047 (5/5) ; −1.363** | −3.78 | 1.000 → 0.901 | 0.383 → 0.025 | 1999 (cap) → 169 |
| naval | **−4.662 ± 0.047 (5/5) ; −4.363** | −0.032 | 1.000 → 0.813 | 0.491 → 0.080 | 1999 → 771 |
| synth_gamma | **−0.646 ± 0.011 (5/5) ; −0.617** | −0.78 | 0.926 → 0.876 | 0.166 → 0.024 | 939 → 173 |
| kin8nm | **+0.050 ± 0.008 (0/5) ; +0.063** | +0.012 | 0.915 → 0.839 | 0.088 → 0.076 | 1364 → 220 |
synth_gamma MeanShape 2.429 vs oracle 2.290 (CR 3.075). power_plant MeanShape-Damped 2.788 now
beats the Gaussian-Damped default (2.806); naval −5.72 vs Gaussian −5.74 (≈ tie).
Within MeanShape: Damped ≈ Fisher (±0.01), Observed worst (−0.04…−0.53) — prediction 3 ✓.
Predictions 1 ✓ (all > 20 s.e.), 4 ✓ (6–20× fewer rounds). Prediction 2 FALSIFIED: kin8nm is
WORSE with orthogonal coordinates, and its 90 % coverage falls 0.915 → 0.839 (over-confident).
The 1565 non-finite rows at η.1/d6 were in the LEGACY ConcRate/Observed cell (seed 2); MeanShape
had 0 failures in 120 fits.
Hypothesis H-k (pre-registered): the dispersion tree, now fast, over-fits the shape (α ↑ →
over-confidence, coverage 0.84) and early stopping fires at round ≈ 220 with the MEAN still
under-fit; in ConcRate the slow α tree could not over-fit and the mean got 1364 rounds through
BOTH trees. Prediction: freezing the shape at 50–200 rounds restores kin8nm to ≤ −0.49 (CR's
level), coverage → ~0.9, and costs synth_gamma/naval little (their dispersion structure is
learnable within 100–200 rounds) — power_plant unaffected. Tag c4b.

### Cycle 4 — c4b: MeanShape + shape freeze (H-k) **[measured]**
Freeze the shape tree (param 1) at k: kin8nm k=200 **−0.150 ± 0.007 vs MeanShape (5/5)**, → −0.593
(beats ConcRate −0.493 and Gaussian −0.488); best_iter 220 → 1018; coverage 0.84 → 0.89 — H-k
confirmed: the fast shape tree over-fits and early stopping truncated the mean. power_plant k=100
−0.120; synth_gamma k=200 +0.001 (k=50 +0.69 — too early); naval any k bad (+0.31…+1.41).
Validation-selected k ∈ {∅, 50, 100, 200}, MeanShape vs ConcRate (both Damped): kin8nm
−0.100 ± 0.009, power_plant −1.427 ± 0.035, naval −4.662 ± 0.047, synth_gamma −0.645 ± 0.012 —
**5/5 on all four**. Default decision deferred to Felix: MeanShape alone wins 3/4 by 0.6–4.7
nats but loses kin8nm by 0.05; with a validation-tuned shape freeze it wins everywhere.

### Cycle 4 — c4c: NB MeanDispersion clamp (5 seeds; oracle-known Gamma–Poisson DGPs) **[measured]**
synth_nb_lo (µ 1.5–60): MeanDispersion − default −0.074 ± 0.006 (5/5); old and widened clamp
bit-identical (no row beyond e⁶). synth_nb_hi (µ 100–4000): default 7.581 (cov90 0.995, KS 0.29);
MeanDispersion OLD clamp 7.803 (+0.222, 0/5; RMSE 1441 — the mean is capped at 403);
MeanDispersion WIDENED (µ ≤ e¹⁵, θ ≤ e¹², zero gradient beyond) **6.722, −0.859 ± 0.013 vs
default (5/5), −1.081 vs old clamp; oracle 6.644**; CRPS 306 → 206 (oracle 200); KS 0.029.
Predictions all ✓. Widening landed in negative_binomial.rs with a test (µ = 3000 representable,
FD-exact gradients there, zero gradient beyond the clamp).

### Cycle 4 — c4d: β-NLL × scale freeze (Gaussian, 5 seeds) **[measured]**
ΔNLL vs plain: protein frz200 −0.021, β1 −0.072, frz200+β1 −0.114 (≈ additive — the one dataset
where both act); kin8nm frz200 −0.157, +β adds nothing (−0.147…−0.160); power_plant frz100 −0.110,
best combo −0.128; naval/sims: β adds ≤ +0.01. Validation-selected over the full menu vs the
freeze-only menu: −0.009…−0.029 on real targets (< 1.5 s.e.), +0.007 on synth_t3. → overlapping;
tune the freeze first. Cycle 4 closed; report: reports/005-cycle4-report.md.

## Cycle 5 (2026-11-10)

### Defaults approved by Felix (2026-11-10)
`Gamma::new(_, Exp, Nll, _)` / `Gamma::default()` → `MeanShape`; `NegativeBinomial::new(.., Nll, _)` /
`default()` → `MeanDispersion` (response functions kept, used only by the legacy head). CRPS and
non-Exp Gamma keep the legacy heads; files saved without the field load with the legacy head
(`#[serde(default = "…::legacy")]`). Opt-out builders `with_parameterization(..)`. Unit tests:
default ≡ dedicated constructor (transforms + gradients), opt-out ≡ legacy head. Bench families
`gamma`/`nb` pinned to the legacy heads. 237 lib / 137 / 129 integration green.

### c5a/c5b — NB MeanDispersion log-θ curvature (proposal 006) **[measured]**
Fisher information of log θ: I = θ² E[ψ′(θ) − ψ′(Y+θ)] − θµ/(µ+θ) — no closed form; exact series
validated by Monte-Carlo E[score²] (4 points, < 5 %). Fast form via the NB PGF:
I = θ ∫ e^{−u}[t/(1−e^{−t})(1 − PGF(t)) − (1 − e^{−µt})] du, t = u/θ, 25-node exp-sinh — p99 rel.
error 3e-3 over the clamp domain (worst 4 % where I < 1e-15); cost ≈ observed (2.3 s vs 15 s for the
series). Near-Poisson corner: Fisher ≈ 5e-4 at θ = 300, µ = 10 — 20× below the old 0.01 floor.
5 seeds, Δ vs Observed+floor (c5b, quadrature; c5a series identical within 0.0005):
| DGP | Damped | Fisher | oracle gap (Obs) |
|---|---|---|---|
| synth_nb_hi | −0.0281 ± 0.0043 (5/5), CRPS −0.83, cov90 .869→.886 | −0.0246 (5/5) | 0.078 |
| synth_nb_lo | −0.0036 ± 0.0015 (5/5) | −0.0034 (5/5) | 0.025 |
| synth_nb_ps (new, µ 1–8, θ 12–245) | −0.0013 ± 0.0005 (4/5, max +0.0003) | −0.0032 (5/5) | 0.018 |
P1 (null where µ ≳ θ) **falsified** on hi (36 % of the oracle gap closed); P2 (≤ −0.003 on ps) not
met for Damped (−0.0013); P3 holds except ps (Fisher −0.003 vs Damped −0.001, both ≤ 0).

### c5c — real counts: UCI Bike Sharing hourly (17 379 rows; `cnt`, `casual`) **[measured]**
MeanDispersion(Observed) − legacy (r, p): bike_cnt +0.001 ± 0.010 (2/5, tie), bike_casual
−0.032 ± 0.006 (5/5) → the new default holds on real data. Curvature (Δ vs Observed): Damped
bike_cnt −0.0098 ± 0.0047 (4/5), bike_casual −0.0024 ± 0.0028 (4/5); Fisher −0.0105 / +0.0006.
Damped better/neutral on all 5 → **NB constructors now default to Damped** (serde default for old
files: Observed). Flag: MeanDispersion under-covers on bike (cov90 .874/.796 vs legacy .931/.851) —
over-fitted dispersion → per-parameter early stopping (proposal 007).

### c5d — per-parameter early stopping (proposal 007), Gaussian, scale only **[measured]**
`GradientLSS::with_param_early_stopping(ParamEarlyStopping::new(W, Some(vec![1])))`. Unit test:
dynamic freeze at the chosen round k ≡ static `param_freeze_rounds` at k (bit-identical).
Δ vs no per-parameter stopping (5 seeds, Damped, η .05/d4, ES 50):
| dataset | W = 20 | W = 50 | tuned freeze (c3) |
|---|---|---|---|
| kin8nm | −0.137 ± 0.011 (5/5), cov .82→.92, frozen ≈ 170 | −0.143 (5/5) | −0.157 |
| power_plant | −0.172 ± 0.035 (5/5), cov .90→.95, frozen 160–240 | **+0.256 ± 0.087 (1/5)**, frozen at 94–97, cov .997 | −0.116 |
| protein | −0.136 ± 0.014 (5/5) | −0.101 (4/5) | −0.098 |
| naval | **−0.267 ± 0.016 (5/5)**, frozen ≈ 410, best_iter 736 → 1999 | −0.300 (5/5) | 0 (every fixed k +0.3…+1.4) |
| gaussian_sim / synth_t3 / studentt_sim | +0.001 / −0.001 / +0.000 | 0 / 0 / 0 | 0 |
P1 ✓ (W=20 beats the tuned freeze on 3/4), P2 ✓ (sims within 0.001; naval *improves* — the adaptive
round is what no fixed k could find), **P3 ✗**: W=50 freezes power_plant at ≈ 95 on a transient
signal (scale briefly ahead of the still-learning location); the location then trains to ~1200
rounds under a stale, too-wide σ. Remedy under test (c5e): `patience` — the revert test must
succeed on P consecutive rounds before freezing.

### c5e — patience probe (Gaussian, scale only, real datasets) **[measured]**
Δ test NLL vs none: W20/P1 (c5d) kin8nm −0.137, naval −0.267, power_plant −0.172, protein −0.136;
W20/P5 −0.136 / −0.272 / −0.174 / −0.107; W20/P10 −0.138 / −0.279 / −0.136 / −0.103;
W50/P10 −0.152 / −0.306 / **+0.123** / −0.101; W50/P20 −0.152 / −0.320 / −0.126 / −0.084.
Patience repairs W=50 on power_plant only partly; nothing dominates W20/P1. The failure is visible on
validation (power_plant valid NLL 3.11 for W50/P1 vs 2.62 for W20/P1), so the window is safely
validation-tunable: per-seed selection over all arms → kin8nm −0.154, naval −0.320, power_plant
−0.169, protein −0.136 (all 5/5), sims ≤ +0.001. Recipe: W = 20, patience 1; tune W ∈ {20, 50/P20}
on validation if a run can afford it.

### c5f — per-parameter early stopping on Gamma MeanShape (shape) and NB MeanDispersion (log θ) **[measured]**
Δ vs none, W20/P1 ; W50/P20 ; validation-selected over {none, W20, W50/P20}:
Gamma kin8nm −0.139 (5/5) ; −0.139 ; −0.139 — power_plant −0.147 (5/5) ; −0.081 ; −0.133 (5/5) —
naval −0.013 ± 0.079 (2/5, max +0.146) ; −0.119 (4/5) ; −0.127 (3/5, none worse) — synth_gamma ±0.001.
NB bike_cnt −0.041 (5/5, cov90 .880→.942) ; −0.061 (5/5) ; −0.061 — bike_casual −0.020 (5/5, cov
.794→.832) ; −0.026 ; −0.026 — synth_nb_lo/hi/ps ≤ +0.001 (θ rarely or never frozen).
P4 ✓ except Gamma naval at W20 (noisy; validation selection fixes it); P5 ✓.
Cycle 5 closed; report: reports/006-cycle5-report.md.

## Cycle 6 — anneal vs freeze, StudentT, hyper_opt axes, default-on (2026-11-10)
Pre-registration: proposals/008-pes-anneal-studentt-default.md (+ Addendum A before c6d).
Felix approved default-on "whenever a validation set is passed", subject to the
better-or-neutral rule. Harness: frozen binary copies (/tmp/rb_c6, /tmp/rb_c6v), Damped,
η .05/d4, ES 50, crate-default init (0) unless noted. Matrix script: c6_matrix.py;
pairs: pair_tags.py.

### c6 implementation **[landed]**
`ParamEarlyStopping::anneal` (0 = freeze; f: multiply the gradient column by f per trigger,
re-test after a full window at the new rate, freeze below 1/64), `param_early_stopping_triggers()`,
`ParamEarlyStoppingSetting {Auto (default), Off, Custom}`, `without_param_early_stopping()`,
`Distribution::default_param_early_stopping_params()` (None by default), hyper_opt axes
`__pes_window/_patience/_anneal` (fold models inherit the model setting; `from_trial_params`
for the final fit; `__pes_*` keep the fold-data cache). Warm-start stage 1 is Off (unchanged).
Bit-identity: c6_check_pes20 ≡ c5d_pes_20 (anneal 0); c6v_auto ≡ explicit arms (6 cells).
LightGBM: dynamic ≡ static freeze test.

### c6a/c6d — anneal vs freeze, W = 20 (ΔNLL vs none, mean (worst seed)) **[measured]**
| | freeze | anneal .5 | anneal .25 |
|---|---|---|---|
| G kin8nm | −.137 (−.107) | −.138 (−.121) | −.136 (−.104) |
| G naval | −.267 (−.240) | **−.368** (−.306) | −.339 (−.286) |
| G power_plant | **−.172** (−.118) | −.143 (.000) | −.179 (−.120) |
| G protein | **−.136** (−.100) | −.101 (.000) | −.100 (.000) |
| G sims | ≤ +.001 | 0 | ≤ +.001 |
| Gm kin8nm | −.139 | −.143 | −.138 |
| Gm naval | −.013 (**+.146**) | **−.194** (−.112) | −.195 (−.099) |
| Gm power_plant | **−.147** | −.112 (.000) | −.120 (.000) |
| Gm synth_gamma | .000 (+.007) | +.001 (+.007) | +.002 (+.005) |
| NB bike_cnt | −.041 | **−.060** | −.049 |
| NB bike_casual | −.020 | −.026 | −.028 |
| NB sims | ≤ +.001 | 0 | 0 |
Anneal's "(.000)" seeds: the global early stopping (patience 50) fired while the scale was still
being annealed (7 windows to freeze at 0.5), reverting to the no-PES fit — never worse than none.
W = 50 anneal .5: Gaussian power_plant −0.143 (5/5) (freeze: +0.256) → **A1 ✓**.
A2: within 0.02 of freeze on kin8nm only; power_plant/protein −0.03 behind (fallback seeds),
naval −0.10 ahead; Gamma naval spread 0.079 → 0.044 sd ✓. **A3 ✗** (0.25 still trails freeze by
0.036 on protein, 0.027 on Gamma power_plant).

### c6b/c6c/c6e — StudentT (df 0, loc 1, scale 2) **[measured]**
init 0 (crate default): s.freeze gaussian_sim **+0.387 (0/5)** — σ frozen while df still rising;
df.freeze power_plant +0.208 (all seeds); both.ann.5: gaussian_sim +0.002, kin8nm −0.148,
naval −0.389, protein **−0.112** (scale-only ann.5: −0.008), sims ≤ +0.002.
init 1: both.ann.5 kin8nm −0.166, naval −0.177, power_plant −0.132 (worst +0.004), protein −0.130
(scale-only −0.058), sims ≤ +0.000.
warm start (API): both.ann.5 kin8nm −0.138, naval −0.340, power_plant −0.165, protein −0.074,
sims 0; scale-only ann.5 power_plant −0.031 (worst +0.060).
T1 ✗ (s.freeze — init 0: kin8nm −0.136 ✓, power_plant 0, protein −0.014, gaussian_sim +0.387;
init 1: kin8nm −0.142, power_plant −0.144 ✓, protein −0.057 ✗, sims ✓). T2 ✗
(df-freeze: power_plant +0.208, protein −0.027). T3 ✗ (both ≫ scale-only on protein).
T4 ✓ kin8nm/power_plant, ✗ protein (−0.009). T5 ✓.
**Deviation from Addendum A:** the default StudentT set is {df, scale}, not {scale}: annealed
df+scale beat scale-only in all three configurations (protein by −0.10/−0.07/−0.06) and was
neutral on the sims; post hoc, replicated across init 0 / init 1 / warm start.

### Side finding — StudentT crate default fails on non-centred targets **[measured, not fixed]**
`StudentT::default()` has `initialize = false` → raw 0.5 start for every parameter. On
power_plant (y ≈ 450) the t-score saturates (∂/∂µ ∝ r/(νσ² + r²)) and the location never
leaves 0.5: RMSE 448, cov90 1.000, NLL 7.52 (init 1: 2.78). Same in cycle 1 (c1c). Gaussian,
Gamma MS and NB MD are unaffected (Gaussian score is linear in r; MS/MD heads compute MLE starts).

### Decision D
Default = W 20, patience 1, anneal 0.5 (only uniform candidate within +0.005 of none on every
dataset of the four families; freeze fails StudentT gaussian_sim, anneal .25 fails it at +0.009).
Watched: Gaussian scale, StudentT df+scale, Gamma MS shape, NB MD log θ; NLL only.
Cost: 2–4× rounds on real data (location trains longer; RMSE improves), sims unchanged.

## Cycle 7 — StudentT start, ES hold, naval cap, Beta/Weibull audit (2026-11-10)
Felix: Python parity not required; StudentT start fix approved; then hold, naval, Beta/Weibull.
Proposals 009 (+ Addendum S3) and 010, pre-registered before the respective runs.

### c7s / c7s3 — StudentT start without `initialize` **[measured]**
Hook: `Distribution::default_start_values` (used only when `initialize` is false; the warm-start
API still fits its own). `StudentTStart {Raw, Median, MedianMad}`; serde default Raw.
Δ vs raw start (PES off ; with df+scale PES):
| | initialize=true | median+MAD (S1/S2) | **median (S3)** |
|---|---|---|---|
| power_plant | −4.73 ; −4.86 | −4.72 ; −4.85 | **−4.75 ; −4.95** |
| kin8nm | +.018 ; .000 | +.023 ; +.013 | +.006 ; −.006 |
| protein | +.068 ; +.050 | +.032 ; +.025 | .000 ; −.008 |
| naval | −.075 ; +.136 | −.092 ; +.112 | −.018 ; +.018 ± .028 |
| gaussian_sim | +.013 ; +.011 | +.007 ; +.005 | **+.008 ; +.006 (0/5)** |
| sims (2) | ≤ .005 | ≤ .009 | ≤ .001 |
S1 ✗, S2 ✗, S3 ✗ strictly (gaussian_sim +0.006–0.008, naval with PES leans worse) — landed
anyway as Felix asked for the fix: removes a −4.9-nat failure; costs flagged in the report.
check: c7_check_raw ≡ c6b_t_0.

### c7h — global early stopping held on each PES trigger **[measured, landed]**
`TrainingCallback::holds_early_stopping` (provided, default false; CallbackList ORs); the model
wraps the caller's callback (`PesHoldCallback`) when PES has hold on (now the default).
Δ hold − no hold (anneal .5 W20): Gaussian power_plant −0.034 (one seed −0.17, rest identical),
Gamma MS power_plant −0.032 (same seed), everything else bit-identical incl. all NB and StudentT.
protein's fallback seed is not rescued: the scale stops triggering and the location had peaked.
H1 partial (power_plant ✓, protein ✗), H2 ✓, H3 ✓.

### c7n — naval at a 6000-round cap **[measured]**
Default PES vs its 2000-cap fit: Gaussian −0.052, Gamma MS −0.028, StudentT −0.035 (converge at
2600–3200 rounds); "none" unchanged (stops ≈ 750). N1 ✓ Gaussian only. vs none at 6000: −0.42 /
−0.22 / −0.39.

### c7b / c7w2 — Beta / Weibull (proposal 010) **[measured, landed]**
Audit: Beta (log a, log b) ρ → −1 with φ (−0.95 at 20, −0.99 at 100); (logit µ, log φ) |ρ| ≤ 0.24
for φ ≥ 20. Weibull ρ = −0.313 constant; η = log λ + (1−γ)/k exactly orthogonal (MC) — not built.
Beta (5 seeds): Concentrations Observed (old default) diverged on bike_share (NLL 2545, best 0).
Damped − Observed: bike_share huge, synth_beta −0.023 (5/5), synth_beta_hi +0.004 (1/5) → B1 ✓.
MeanPrecision Damped − Concentrations Observed: −0.27 (synth_beta_hi), −0.10 (synth_beta),
bike_share huge, all 5/5; vs Concentrations Damped −0.28 / −0.075 / −0.038, all 5/5. B2 ✓.
Oracle gap synth_beta_hi 0.34 → 0.06. MeanPrecision Damped − Observed −0.06/−0.01/−0.01 (5/5).
Weibull Damped − Observed: power_plant −0.151 (5/5), kin8nm −0.083 (5/5), synth −0.047 (5/5),
naval −0.049 (4/5; one seed +1.43 on test only, its validation NLL was 0.8 better). W1 ✓.
Harness bug found and fixed: statrs' Weibull cdf returns NaN for shape ≳ 100, so power_plant
rows were silently dropped (c7w, superseded by c7w2 with the closed-form cdf).
Defaults: Beta Exp+NLL → MeanPrecision + Damped; Concentrations head → Damped; Weibull → Damped.

## Cycle 8 — PES for Beta/Weibull, StudentT gaussian_sim cost, location starts (2026-11-10)
Proposal 011 (+ Addenda L, T4, P3). Binaries /tmp/rb_c8, /tmp/rb_c8b.

### c8t / c8t4 — StudentT +0.008 on gaussian_sim **[measured, explained]**
raw vs median start: df_med 100 vs 11, mean scale 2.28 vs 1.90, best_iter 120 vs 59.
T1 ✗ ν₀ = 10: closes 0.003 of 0.008 (df_med 27), hurts synth_t3 +0.012, studentt_sim +0.007,
power_plant +0.017. T3: df ×2 → df_med 30, ΔNLL +0.0002 ⇒ df is not the cause.
T4 (y shift): at −9.5 (y centred at 0.5) raw − median = +0.0003 (T4a ✓); the median start is
exactly invariant to the shift (T4b ✓); at +20 the raw start is better again (−0.009, df 1222;
T4c ✗ but informative). ⇒ a distant location start is an accidental warm-up on Gaussian data
(scale inflates, df pushed up while the location travels); the +0.008 is the raw start's
position, not a defect of the median start. Median start kept; ν₀ unchanged.

### c8l — location-only starts (no `initialize`) **[measured]**
Harness fix (Addendum L): `gamma_ms` / `nb_md` were built with `initialize: true` since cycle 4;
now via `new(.., BENCH_INIT)`; BENCH_INIT=1 reproduces c7h_gm / c7h_nb bit-for-bit.
Gamma MS (default PES), loc − raw: kin8nm +0.019, naval +0.046, power_plant +0.005,
synth_gamma −0.015 → L1 ✗ (fitted − raw: naval +0.155, kin8nm +0.014) ⇒ keep raw.
NB MD, loc − raw: bike_cnt −0.184, bike_casual −0.082, synth_nb_hi −1.447, synth_nb_lo −0.006
(all 5/5), synth_nb_ps −0.001 → L1 ✓ (L2's "neutral" prediction ✗: big wins). loc ≈ fitted
(≤ +0.006) at ~½ the time; raw synth_nb_hi fits took ~480 s each. ⇒ NB MD location start default.
Weibull, loc − raw (PES off): naval +0.73, power_plant +0.039, kin8nm +0.017, synth −0.009 →
L1 ✗; fitted worse (naval +4.9) ⇒ keep raw.

### c8p / c8p10 — PES on Beta MP (log φ) and Weibull (log k) **[measured, landed]**
Beta MP: bike_share −0.019 (5/5), synth_beta 0.000, synth_beta_hi 0.000 → P1 ✓, P2 ✗ (−0.019).
Weibull: naval −0.815, kin8nm −0.035 (5/5), synth 0, power_plant +0.043 (one seed +0.56 on
test, its validation 0.05 better) → P1 ✗ at 5 seeds. Post-hoc 10 seeds (Addendum P3):
power_plant −0.013 (9/10; validation better 10/10) ⇒ added, disclosed as post hoc.

## Cycle 9 — Weibull tail, deliberate warm-up, Gamma raw vs fitted start (2026-11-10)
Proposal 012 (+ Addendum G3). Binary /tmp/rb_c9. No default changes.

### Weibull tail blow-ups **[diagnosed, measured]**
BENCH_DEBUG_WORST: power_plant seed 0 one row y 491.97, λ̂ 476.4, k̂ 222.8 → NLL 1314 (20 % of
total; next 30); naval seed 3 y = 1.000 (upper bound), λ̂ 0.996, k̂ 2169 → 5711. Cost (y/λ)^k
= e^z, z ≈ k ln(y/λ) in predicted-sd units: exponential in z for every k.
Caps (default recipe; power_plant 10 seeds), Δ vs none: K=50 naval +2.77, power_plant +0.48;
K=200 naval +1.40, power_plant −0.041 (5/10, max +0.002); K=1000 naval +0.083, power_plant 0;
kin8nm / synth 0. C1 ✓ (no cap neutral everywhere) ⇒ opt-in `with_max_concentration`.

### Warm-up for StudentT (median start, PES off) **[measured]**
Δ vs median start: location-only warm-up 20 / 50 rounds: gaussian_sim +0.004 / +0.009 (worse —
W1 ✗), kin8nm −0.010/−0.016, protein −0.003/−0.016, naval −0.020/−0.027, power_plant +0.024/
−0.012, sims −0.002…−0.007. Start scale 3·sd: gaussian_sim −0.0088 (5/5; df_med 185, scale
2.33, best_iter 102 — reproduces the raw start exactly) but studentt_sim +0.013, synth_t3
+0.020, protein +0.091, power_plant +0.054 → W2 mechanism ✓, as a default ✗.
⇒ the raw start's gaussian_sim benefit is "approach the scale from above", which helps when the
truth is Gaussian-like and hurts heavy tails. No warm-up default. `with_warmup_rounds` kept
as an opt-in (generic: `Distribution::location_param_index`).

### Gamma MS: raw vs fitted start on naval **[measured]**
Δ vs raw (default PES): loc +0.046, fitted +0.155, raw mean + fitted shape +0.044, fitted mean +
α̂/100 +0.037, fitted + 50-round warm-up +0.085 (all naval; kin8nm/power_plant/synth mixed,
≤ ±0.03). G1 ✗, G2 ✗. All naval fits sat at the 2000 cap ⇒ G3: at 6000 rounds (both converge,
best 3005 / 2562) raw − fitted = −0.175 (5/5) ⇒ not a budget artefact: path dependence. The
raw start is overdispersed (α 1.65 ≪ α̂ ~ 10⁴, mean offset); both components contribute about
equally — the same "approach the dispersion from above" effect as StudentT. Gamma already
defaults to the raw start: no change.

## Cycle 10 — adaptive wide start, Gamma start units, tail-robust stopping (2026-11-11)
Proposal 013 (+ Addendum A′). Binary /tmp/rb_c10. No default changes.

### A — adaptive "start wide" for StudentT **[rejected]**
- **Kurtosis gate (`StudentTStart::Adaptive`): rejected without a fit.** Train-label g₂: gaussian_sim 3.57,
  power_plant −1.06, protein −1.12, kin8nm −0.57, naval −1.22, synth_t3 20, studentt_sim 195, synth_g_sin
  −0.81, synth_g_exp 0.53, synth_t4_flat 5.1. The marginal is dominated by the signal, so the gate inflates
  protein and power_plant (c9: +0.091 / +0.054) and skips gaussian_sim.
- **A2 ✓ (new held-out sims, PES off; 3·sd vs median).** synth_g_sin −0.0120 (5/5), synth_g_exp −0.0024
  (5/5), synth_t4_flat +0.0153 (0/5). "Approach the scale from above" generalises to Gaussian noise.
- **A′ (choose the start by best valid NLL; `select_start_by_valid.py`).**
  - Wins: gaussian_sim −0.0088, naval −0.063, synth_g_sin −0.012, synth_g_exp −0.0024.
  - Neutral (0.000): protein, studentt_sim, synth_t3, synth_t4_flat; kin8nm −0.002.
  - power_plant +0.050: seeds 2 and 3 were better on validation (−0.004 / −0.020) and worse on test
    (+0.10 / +0.15); the wide start drives df to ~1000.
  - Verdict: A′1 ✗, A′2 ✓. It also doubles the cost, so it is not a default.

### G — Gamma MS start units **[measured]**
- `BENCH_Y_SCALE` rescales the labels; NLL is compared after subtracting ln c. raw c1 ≡ c8l_raw (bit-identical).
- **G1 (raw is unit-dependent):** ✓ but smaller than predicted.

  | | kin8nm | power_plant | naval | synth_gamma |
  |---|---|---|---|---|
  | raw(c) − raw(1), c = 100 | +0.066 (5/5 worse) | +0.007 | +0.005 | +0.005 |
  | raw(c) − raw(1), c = 0.01 | +0.026 | +0.010 | +0.008 | −0.010 |

  Naval's predicted >0.05 ✗.
- **G2 (wide_b unit-free to ≤ 0.002):** ✗. wide_b moves by ≤ 0.011 on average across c, but by up to
  0.1 per seed on naval.
  - The start is unit-free; the fit is not. Labels and gradients go to xgboost in f32, which flips near-tied
    splits.
  - Early stopping is also not unit-free: the patience rule's min-delta is relative (1e-4·|mean NLL|), and
    the mean NLL shifts by ln c. kin8nm stop rounds differ by up to 600 across c.
- **wide_b − raw:**

  | c | kin8nm | naval | power_plant | synth_gamma |
  |---|---|---|---|---|
  | 1 | +0.001 | +0.007 ± 0.042 | +0.002 | −0.016 (4/5) |
  | 0.01 | −0.017 | −0.010 | −0.007 | −0.006 (5/5) |
  | 100 | −0.056 (5/5) | −0.009 | −0.011 | −0.020 (5/5) |

  G3 ✓ within noise, but naval c1's +0.0072 is outside the +0.005 band, so the strict rule fails ⇒ no
  default change (decision left to Felix).
- **Other arms at c1 (vs raw):** wide_a [ln ȳ, 0] kin8nm +0.010, naval +0.057, power_plant +0.002,
  synth_gamma −0.014. wide_m [ln ȳ − 0.5, 0.5] vs wide_b: kin8nm +0.025 (4/5 worse), naval +0.006,
  power_plant +0.006 ⇒ G4 ✓ weakly: starting above the mean is better.

### R — tail-robust stopping metric **[rejected; opt-in]**
`GradientLSS::with_stopping_row_cap(K)` softens rows above median + K to median + K + ln(1 + excess), in
the stopping decision only.
- **K = 10:**
  - Weibull: naval +0.88 (one seed +4.2), power_plant +0.097 (10 seeds).
  - Others: StudentT power_plant +0.016, protein +0.011; Gaussian power_plant +0.004; Gamma naval +0.037
    (0/5).
- **K = 25:**
  - Weibull: power_plant +0.032, naval −0.003.
  - Others: Gaussian power_plant +0.005 (0/5), protein −0.013; Gamma power_plant −0.007; the rest
    identical.
- R1 ✗, R2 ✗.
- **Mechanism:** the extreme validation rows are the signal that the shape or scale is overconfident.
  Softening them lets the fit keep sharpening, and CRPS improves while the NLL tail worsens (Weibull
  power_plant ΔCRPS −0.07, ΔNLL +0.10).

### Decision (Felix, 2026-11-11): Gamma unit-free start landed **[default change]**
Felix accepted naval c1's +0.0072 ± 0.042 (0.17 s.e.) as neutral.
- **Change:** `Gamma` MeanShape without `initialize` now starts at [ln ȳ + 0.5, 0.5]
  (`wide_start`, default true). serde default false, so saved models keep the raw start.
- **API:** `with_wide_start(false)` restores the old start; `location_start` takes precedence.
- **Bench check (c10v):**
  - the default reproduces c10g_wb_c1 to floating point (power_plant, synth_gamma);
  - `BENCH_GM_RAW=1` reproduces c10g_raw_c1 bit-for-bit.
- **Tests:** `wide_start_is_the_default_and_unit_free`, `serde_default_is_the_raw_start` (gamma).

## Cycle 11 — Poisson, NB, Gaussian (2026-11-11)
Proposal 014 (+ Addendum G′). Binaries /tmp/rb_c11 and /tmp/rb_c11b (BENCH_Y_STD).
Literature: Lacuna had nothing direct (closest: "Data subsampling for Poisson regression with pth-root
link", NeurIPS 2024, tangential). The log link is standard in XGBoost `count:poisson`
(max_delta_step 0.7 by default), LightGBM `poisson` and GAMLSS PO (Rigby & Stasinopoulos 2005).

### Harness bug: the discrete PIT uniform **[fixed; `BENCH_PIT_U=hash`]**
- u = frac(0.618·y + 10⁻³µ) depends on y, so y = 0 always gets u ≈ 0.
- The oracle's own cov90 was 0.82 / 0.84 / 0.74 (synth_nb_ps / nb_lo / pois); with the hashed uniform it
  is 0.90 / 0.89 / 0.89.
- NB coverage in cycles 4–10 was biased low; NLL and CRPS are unaffected.

### P — Poisson (5 count sets × 5 seeds) **[landed]**
NLL by arm (bike_casual, bike_cnt, nb_lo, nb_ps, pois):

| arm | bike_casual | bike_cnt | nb_lo | nb_ps | pois |
|---|---|---|---|---|---|
| relu-observed (old default) | 5.350 | 9.091 | 4.595 | 1.925 | 2.200 |
| relu-damped | 5.275 | 8.961 | 4.576 | 1.923 | 2.160 |
| exp | 4.184 | 6.756 | 4.632 | 1.923 | 2.180 |
| exp + ȳ start | 4.123 | 6.413 | 4.575 | 1.922 | 2.146 |
| softplus-damped + ȳ start | 4.729 | 8.682 | 4.560 | 1.922 | 2.152 |
| relu + ȳ start | 5.229 | 9.826 | | | |

relu + ȳ start hurts bike_cnt (+0.87 against relu-damped).

exp + ȳ start against the old default:

| dataset | ΔNLL | seeds better |
|---|---|---|
| bike_casual | −1.227 | 5/5 |
| bike_cnt | −2.677 | 5/5 |
| synth_nb_lo | −0.021 | 4/5 |
| synth_nb_ps | −0.003 | 4/5 |
| synth_pois | −0.054 | 5/5 |

bike_cnt RMSE −14.5.

Predictions:
- P1 ✗ in part: exp without the start is +0.037 on nb_lo.
- P2 ✓: relu-damped beats relu-observed on every set (pois −0.040, 5/5), but stays behind exp.
- P3 ✓ under exp and softplus, ✗ under relu.
- P4 ✓: Poisson exp + start on synth_pois scores 2.146 against NB MD 2.150. The cycle-11 smoke gap was
  the ReLU link and raw start.

**Landed:** `Poisson::default()` uses Exp; `new` defaults to Damped; `location_start` defaults to
`link != Relu`. serde: Observed and no start. Tests in `poisson::cycle11_tests`.

### N — NegativeBinomial **[measured, no change]**
- N1 ✓: with the hashed PIT, NB MD cov90 on synth_nb_lo / ps / pois is 0.905 / 0.906 / 0.907 against
  oracle 0.89–0.90.
- N2 ✗ (as stated): PES vs no PES on synth_pois is 2.1502 vs 2.1503, so NB is already fine at the
  Poisson boundary without PES.
- On real counts PES helps: bike_cnt −0.055, cov90 0.879 → 0.928.

### G — Gaussian start units **[rejected as a default]**
- raw c100 − c1: power_plant +0.205 (0/5), protein +0.056, gaussian_sim +0.054, kin8nm −0.026.
- sd3 − raw at c = 1: naval +0.099 (1/5), protein −0.073, others ±0.006 ⇒ G2 ✗.
- The unit-free starts were *more* unit-dependent than raw (power_plant c100 − c1 +1.30).

### G′ — XGBoost's absolute regularisation is not unit-free **[measured, decision needed]**
- **Cause:** `min_child_weight` and `lambda` act on raw Hessian sums, and the Gaussian location Hessian
  is 1/σ² ∝ c⁻². At native units it spans ≈ 0.06 per row (power_plant) to ≈ 10⁶ (naval).
- **Standardized labels (train-split mean/sd), default start, − raw at c = 1:**
  - gaussian_sim +0.0006, kin8nm −0.0003;
  - power_plant −0.042 (4/5), protein −0.067 (5/5; RMSE −0.11);
  - naval +0.066 (1/5).
  - G′1 ✓ on the four predicted sets; naval is worse.
- **G′2 ✓:** standardized at c = 100 equals c = 1 exactly on power_plant.
- **Naval:** it sits at the 2000-round cap in both arms, and standardization also moves its start
  (σ₀ 220 sd → 1.65 sd), so the comparison mixes curvature, start and budget.
- A unit-free location coordinate would be an invasive change and is pending Felix.

### Cycle 11b — naval 6000 / start vs curvature (Addenda G″, G‴) **[no library change]**
Naval at 6000 rounds:
- B (standardized) − A (native) = +0.073 (2/5): the loss is real (H1 ✗).
- B − D (same start, curvature only) = −0.019 (4/5) (H2 ✓).
- D − A (start only) = +0.091 (1/5) (H3 ✓): raw σ₀ ≈ 220 sd is a very favourable "approach from above"
  on naval.

C (standardized curvature, raw start mapped) − raw:
- gaussian_sim +0.0088 (0/5), kin8nm −0.006, naval +0.067 (1/5; +0.054 at 6000), power_plant −0.036
  (2/5), protein −0.011 ⇒ C1 ✗, C2 ✗.
- The start, the curvature scale and `min_child_weight`/`lambda` interact; there is no unit-free
  arrangement that is neutral everywhere. Documented in the README; no library change.

## Cycle 12 — scale-relative tree regularisation (2026-11-11)
Proposal 015. Binary /tmp/rb_c12. Felix approved investigating the change of hyperparameter semantics.
- **Math:** scaling (g, h) of parameter j by κⱼ gives leaf = −Σg/(Σh + λ/κⱼ), mcw → mcw/κⱼ,
  γ → γ/κⱼ; the step is otherwise unchanged.
- **Test** `hessian_normalization_only_rescales_regularisation`: with mcw = λ = 0 the fit moves
  ≤ 3·10⁻³ (f32); with defaults it moves ≈ 0.2.
- **Marginal − Off:**
  - Gaussian: kin8nm −0.018 (5/5), power_plant −0.044 (4/5), protein −0.018, naval +0.016,
    gaussian_sim +0.0087 (0/5).
  - Gamma: naval −0.043, kin8nm −0.013.
  - NB: bike_cnt −0.008.
  - Poisson: bike_cnt +0.066 (0/5).
- **Start − Off:** Gaussian power_plant +0.236 (0/5).
- **Invariance** (unit-free start + Marginal, c100 − c1): ≤ 0.006 mean, against +1.30 without.
- Not a default: the gaussian_sim and Poisson bike_cnt losses fail the rule. The per-dataset optimum
  strength differs; for high-information rows "one row's worth" of λ is much stronger L2. Opt-in.
- **Next:** tune mcw/λ on the normalised axis in hyper_opt with an equal budget against tuning raw
  mcw/λ. Does the normalised axis find good settings with fewer trials? That is the honest test of
  its value.

## Cycle 13 — Hessian normalisation as a tuning axis (2026-11-11/12)
Proposal 016; report 014. Binary /tmp/rb_c13; `run_c13.sh` (4 streams, OMP_NUM_THREADS=3); `hpo_summary.py`.
- **Fix:** `hyper_opt` fold models now inherit `hessian_normalization` and `stopping_row_cap`. Test
  `hessian_normalization_reaches_the_fold_models` fails without the fix.
- **Bench:** `BENCH_HPO=n` tunes mcw/λ (log [10⁻³, 10³], TPE, 3-fold CV) and refits at
  `BENCH_HPO_CHECK` budgets; rows are tagged `<tag>_k<k>`.
- **Test NLL, norm − raw at k = 20:**
  - Gaussian: kin8nm −0.035 (5/5), power_plant −0.022, protein −0.024, naval −0.049,
    gaussian_sim −0.000;
  - Gamma: kin8nm −0.025 (5/5), naval +0.022 ± 0.028;
  - NB bike −0.008 (5/5);
  - Poisson: bike_cnt +0.017 ± 0.027, synth_pois +0.001.
- **At k = 5:** naval +1.2 (both families). The normalised range's heavy end is destructive there.
- **Predictions:** T1 ✗ narrowly, T2 ✗, T3 ✓ (7/10), T4 ✓.
- **Poisson bike_cnt:** both tuned arms are worse than the untuned default (selection does not
  transfer).
- No default change; README updated. Recommending the axis is Felix's call.

### Cycle 13b — narrowed normalised range [10⁻³, 10] (2026-11-13) **[no change]**
Addendum N; `run_c13b.sh`; tag `c13_norm10_k{5,10,20}`; bench `BENCH_HPO_HIGH`.
- **Naval at k = 5 is fixed:** +0.015 / +0.016, against +1.2 with the wide range (N1 ✓).
- **At k = 20 the narrowed range loses most of the wide range's gains** (norm10 − norm): kin8nm
  +0.035, protein +0.046, naval +0.045, power_plant +0.022, Gamma kin8nm +0.031.
- **Against raw:** gaussian_sim +0.006 (0/5), Gamma naval +0.034, Poisson bike +0.043 (N2 ✗, N4 ✗).
- **Chosen λ:** it hits the new edge (≈ 8 rows' worth). The optimum spans about four decades in
  rows' worth (naval ≈ 0.05, kin8nm ≈ 700).
- Not recommended. The README says to use the wide range with ≥ 20 trials.

## Cycles 14–18 — the tuner, family selection, other backends, unseen data, five more families (2026-11-13)
First cycles run under the two project skills (`automl-hyperparameter-research`,
`optimization-algorithm-research`). Proposals 017–021; report 015. Binaries `/tmp/rb_c14` … `/tmp/rb_c15c`
(one XGBoost thread per stream; trial lines are bit-identical to cycle 13's 3-thread logs).
**No default change. Library additions are opt-in.**

### Cycle 14 — is the search the bottleneck? (proposal 017) **[no change]**
`run_c14.sh`; tags `c14_{rand,dtpe,ds10}_k{5,10,20}`, `tpe` = `c13_raw`; `c14_trace_analysis.py`,
`c14_sampler_summary.py`, `c14_selection_analysis.py`.
- **Library (opt-in):** `HyperOptConfig::sampler` (`Sampler::{Tpe, Random, TpeAfterRandom{n_startup}}`)
  and `HyperOptConfig::initial_trials` (configurations scored first). The default path replays a
  cycle 13 cell line for line.
- **Trace re-read (no compute):** trial 2 repeats trial 1 and trials 3–4 go to the range ends (share
  of draws within half a decade of an end: 0.80 and 0.95; uniform 0.31). After trial 5 the draws are
  close to uniform.
- **Tier 1 (best CV score, random − TPE):** random is better on 4/10 sets at k = 5 and on **0/10 at
  k = 10 and k = 20**; median +0.003 to +0.004. The start-up oddity costs nothing measurable.
- **Tier 2 (test NLL):** every sampler is worse than the untuned default by > 0.005 on 2/10 sets at
  k = 20. Mean over sets: tpe −0.008, ds10 −0.009, dtpe −0.006, random −0.004.
- **The search believes more than it gets.** CV gain over the default (same folds) −0.034 to −0.047;
  the refit keeps −0.005 to −0.007 on validation and on test. Within a set the CV gain of a pick does
  not predict its test gain (correlation −0.18 … +0.36); the refit's validation gain does (0.86–0.98
  on both naval sets, both bike_cnt sets and Gamma kin8nm).
- **Diagnosis (rule fixed in advance):** raw Poisson bike_cnt and raw Gaussian naval are **selection
  failures**: the default's CV score is worse than the pick's in 5/5 seeds; validation and test prefer
  the default. On naval 4 of 5 picks sit at mcw ≈ 700–990.
- **Predictions:** P1 ✗ (4/10, needed 7), P2 ✗ (7/10; the three misses favour TPE), P3 ✗ (scoring the
  default first changes nothing reliable: dtpe − tpe is +0.014 … +0.032 on four sets, −0.049 on
  naval), P4 ✓ (8/10).
- **Post hoc:** keeping the tuned refit only when it beats the untuned fit on the validation split
  removes both losing sets for tpe, random and dtpe (0/10; ds10 1/10) and doubles the mean gain
  (tpe −0.008 → −0.019). Pre-registered for unseen data as proposal 022.
- No start-up phase proposed (P1 failed). `initial_trials` not recommended (P3 failed).

### Cycle 14f — `dist_select` against a held-out choice (proposal 018) **[no change]**
`run_c14f.sh`; tag `c14f`; `c14f_family_selection.py`. 9 sets × 5 seeds, 2–4 candidate families each.
- **Marginal fit (what `dist_select` ranks by) picks the family with the best test NLL in 22/45
  cells; validation NLL of a fitted model in 42/45.** Mean regret 0.0545 against 0.0005.
- **Where the marginal ranking is wrong:** protein (Gaussian over StudentT, regret 0.211), kin8nm
  (Weibull, 0.137), power_plant (Gamma, 0.104), naval (0.034), gaussian_sim (StudentT, 0.006).
- **Side finding:** `dist_select` scores StudentT at the training start values, whose df is capped;
  on protein and power_plant its marginal NLL is therefore worse than Gaussian's.
- **Predictions:** F1 ✓, F2 ✓, F3 ✓.
- README: choose the family by validation NLL of fitted models; `dist_select` is a pre-filter.

### Cycle 16 — do the cycle 1–13 defaults transfer to LightGBM? (proposal 019) **[no change]**
`run_c16.sh`; tags `c16_{xgb,lgb}_{old,new}`; `c16_recipe_transfer.py`; the bench builds on LightGBM
with `--features lightgbm` (backend alias `Bk`).
- **0/21 pairs break the +0.005 rule on either backend.** Sign agreement 19/19. Mean change
  XGBoost −1.006, LightGBM −1.095.
- **LightGBM with the old defaults fails outright** on StudentT power_plant (5/5 seeds non-finite)
  and on one seed each of Gamma power_plant and Gamma synth_gamma; the new defaults have no failure.
- **Caveat:** LightGBM Gaussian protein is −0.015 ± 0.126 (3/5): seed 2 is 3.915 with the new
  defaults against 3.455 with the old (XGBoost: 2.945).
- **Predictions:** L1 ✓, L2 ✓, L3 ✓.

### Cycle 17 — the defaults on datasets no cycle has seen (proposal 020) **[no change]**
`run_c17.sh`; tags `c17_{old,new}`; data `data_confirm/` (seven UCI sets, `prep_confirm_data.py`).
- **0/19 pairs break the rule; no non-finite row.** Mean change: Gamma −2.19, Gaussian −0.57, NB
  −0.26, Poisson −14.4, StudentT −6.6.
- **Smallest effect:** Poisson c_abalone +0.0017 ± 0.0011 (2/5), inside the rule.
- **Not a defaults question:** Poisson on c_seoulbike is 19.6 (90 % coverage 0.41) against NB 5.97.
- **Predictions:** C1 ✓, C3 ✓, C2 ✗ by the letter for StudentT (the development mean −42.9 is one
  pair, power_plant −299).

### Cycles 15 and 18 — Cauchy, Logistic, Gumbel, Laplace, LogNormal (proposal 021) **[opt-in; default change proposed for four]**
`run_c15.sh`, `run_c18.sh`; tags `c15_{base,init,d_raw,d_loc,d_ls,o_ls}`, `c18_{base,port}`;
`c15_port_summary.py`; `src/distributions/location_scale.rs`.
- **Baseline at the crate defaults (c15_base):** power_plant Cauchy 12.88 (stops at round 1),
  Logistic 7.61, Gumbel 7.43, Laplace 7.65, against Gaussian 2.68; protein Gumbel 73 681.
  `initialize: true` is worse than the default on 9/23 pairs.
- **Cause (from the code):** the location Hessian underflows (Logistic, Gumbel) or is negative
  (Cauchy, for |z| > 1) far from the target; XGBoost then makes no step.
- **Prototype (opt-in):** `with_hessian_mode` (Fisher / Damped, constants checked by Monte Carlo
  within 2 %) and `with_start(LocationScaleStart::{Raw, Location, LocationScale})` on the five
  families. Defaults stay Observed / Raw.
- **Development sets:** Damped + LocationScale ≤ default on every pair for Cauchy, Logistic, Laplace,
  LogNormal (power_plant 2.82 / 2.69 / 2.73). Curvature alone leaves Logistic power_plant at 7.34.
  Recipe chosen by validation rank: Cauchy and LogNormal Damped + Location, Logistic and Laplace
  Damped + LocationScale, Gumbel Damped + Raw.
- **Unseen sets, run once (c18):** the chosen recipe is ≤ default + 0.005 on **20/20 pairs for
  Cauchy, Logistic, Laplace, LogNormal** (c_airfoil: Cauchy 10.29 → 2.54, Logistic 6.32 → 2.19,
  Laplace 6.27 → 2.28; c_gt_nox Cauchy 18.68 → 3.09). Gumbel: 4/5; c_gt_co +0.016 ± 0.008 (1/5).
- **Gumbel is not solved:** single-seed blow-ups remain with a data start (power_plant 557 and
  20 802; naval seed 1 21.0), and it stays 3.9 behind Gaussian on c_airfoil.
- **Laplace caution:** Damped with a raw or location-only start is worse than the default on naval
  (+0.75, +1.14); only Damped + LocationScale passes.
- **Predictions:** H1 ✗ (Gumbel), H2 ✓ for four families ✗ for Gumbel, H3 ✓, H4 ✓, H5 ✗ (Gumbel),
  H6 ✓ for four families ✗ for Gumbel.
- Default change for Cauchy, Logistic, Laplace, LogNormal is Felix's call.

## Cycles 19–22 — the validation guard, CRPS training, a wider search space (2026-11-13)
Proposals 022–025; report 015 §6–§10. Binaries `/tmp/rb_c15c` (c19), `/tmp/rb_c20` (c20–c22; adds
`BENCH_LOSS`, `BENCH_HPO_SPACE`). `RAYON_NUM_THREADS=1` is bit-identical (37 cells, one search, and
the c19 re-run). **No default change. No library change in these cycles.**

### Cycle 19 — a validation guard for tuned models, on unseen data (proposal 022) **[no change; README guidance]**
`run_c19.sh`; tags `c19_{tpe,dtpe,tpe5}_k{5,10,20}`, `c19_def`; `c19_guard_summary.py`. 14 unseen
pairs × 5 seeds, two axes, 20 trials. The task hit its 2 h limit with Gamma c_supercond seeds 2–4
missing; that set was re-run (group 5), seeds 0–1 identical line for line.
- **Plain tuning is worse than the untuned default by > 0.005 on 2/14 unseen pairs** (Gaussian
  c_parkinsons +0.052, c_gt_co +0.020). Mean −0.017, median −0.009.
- **Guarded (keep the tuned refit only if its validation NLL beats the untuned fit's): 0/14.**
  Mean −0.023, median −0.018. Default-first arm: 0/14, but mean −0.024 against plain −0.033: on
  Poisson c_seoulbike the guard gives up half the gain (−0.316 → −0.141).
- **CV gain −0.082, test gain −0.022 (27 % survives).** Correlation of CV gain with test gain ≤ 0.2
  on 8/14 pairs.
- **5 folds instead of 3:** Gaussian naval +0.053 → +0.012; Poisson bike_cnt +0.030 → +0.053.
- **Predictions:** G1 ✓, G2 ✓, G3 ✓ (tpe; ✗ for dtpe), G4 ✓, M1 ✗ (fold size is not the mechanism).

### Cycle 20 — training on the CRPS: Poisson and NegativeBinomial (proposal 023) **[no change; two defects reported]**
`run_c20.sh`; tags `c20_{nll,crps}`; `c20_crps_summary.py`; bench `BENCH_LOSS=crps`. 10 pairs × 5 seeds.
`c20_nll` reproduces the earlier default rows bit for bit (37 cells).
- **CRPS training beats NLL training on test CRPS by > 1 % on 0/10 pairs.**
- **Poisson, small counts:** a tie (c_abalone −0.04 %, synth_pois +0.9 %).
- **Poisson above the 100-term cap does not train:** bike_cnt 134.8 against 18.2, c_seoulbike 505.6
  against 76.5; best round 0 in 10/10 cells. The gradient sums stop at
  k = min(max(y, λ) + 4√λ, 100) (`poisson.rs`, `analytical_crps_gradients_impl`). No warning.
  bike_casual (mean 36): +48 %, at the round cap.
- **NB:** worse on 5/5 sets (+2.7 % synth_nb_lo, +8.2 % bike_casual, +34 % c_abalone, +116 %
  bike_cnt, +485 % c_seoulbike); 19/25 cells at the 2000-round cap. Table cap 220.
  `new_mean_dispersion` with CRPS falls back to (total count, probs).
- **Predictions:** K1 ✓, K2 ✓, K3 ✓, K4 ✓.
- No code changed. README: train on the NLL. A fix (table scaled with λ or a continuous
  approximation; NB in orthogonal coordinates) needs its own proposal.

### Cycle 21 — a wider search space at the same budget (proposal 024) **[no change; confirmation in cycle 22]**
`run_c21.sh`; tags `c21_wide_k{10,20}`, reference `c13_raw_k{10,20}`; `c21_space_summary.py`; bench
`BENCH_HPO_SPACE=wide` (max_depth 2–8, subsample and colsample_bytree in [0.5, 1], plus mcw and λ).
The ten development pairs × 5 seeds, TPE, 3 folds, 20 trials.
- **Mean test gain over the untuned default: wide −0.089, two axes −0.008.** Wide at 10 trials
  (−0.071) beats two axes at 20.
- **Worse than the default by > 0.005: wide 0/10, two axes 2/10.** Gaussian naval +0.053 → −0.127;
  Poisson bike_cnt +0.030 → −0.124. The guard changes nothing for wide at 20 trials.
- **Paired, wide − two axes:** Gaussian kin8nm −0.118, naval −0.180, protein −0.043, power_plant
  −0.029; Gamma kin8nm −0.073, naval −0.153; NB bike_cnt −0.053; Poisson bike_cnt −0.154; the two
  simulations ±0.006.
- **Picks:** median depth 5–6 on seven pairs, 4 on protein, 2 on both simulations; subsample
  0.68–0.86, colsample 0.71–0.92.
- **Not shown:** which axis carries it (the fold score moves with depth; with subsampling only
  weakly and with unstable sign). On naval every arm stops at the 2000-round cap, so deeper trees
  also mean a larger effective budget there.
- **Predictions:** S1 ✓ (9/10), **S2 ✗** (predicted ≥ 2/10 losses, observed 0), S3 ✓ (8/10 better
  by > 0.01, 0 worse), S4 ✓ (9/10).

### Cycle 22 — the wide space on unseen data (proposal 025) **[no change; no recommendation]**
`run_c22.sh`; tags `c22_wide_k{10,20}`, reference `c19_tpe_k{10,20}`, default `c19_def`;
`c21_space_summary.py tags=2d:c19_tpe,wide:c22_wide base=c19_def,c17_new`. 14 unseen pairs × 5 seeds.
- **Mean test gain over the untuned default: wide −0.131, two axes −0.017** (medians −0.054 and
  −0.009; without Poisson c_seoulbike −0.062 and −0.008). Wide beats two axes on 12/14 pairs:
  Gamma c_parkinsons −0.290, Gaussian c_parkinsons −0.251, c_airfoil −0.118, Poisson c_seoulbike
  −0.890.
- **Wide is worse than the default on 2/14: Gaussian c_gt_co +0.190 (0/5 seeds better than two
  axes), Gaussian c_supercond +0.025.** The guard keeps both (guarded +0.166, +0.015).
- **c_gt_co:** heavy-tailed target, Gaussian is the wrong family (untuned Gaussian 1.68, Gamma
  1.11, StudentT 1.00). A few test rows carry the NLL; validation does not see them (validation
  1.11–1.64 by seed, test 1.27–3.20). Gamma on the same data gains −0.032.
- **Predictions:** W1 ✓ (13/14), **W2 ✗**, **W3 ✗** (10/14 better by > 0.01, 2/14 worse), W4 ✓.
- By the rule: the README shows the development and the unseen table and recommends nothing.
- **Open:** depth-only and subsampling-only arms; a selection score robust to tail rows; why the
  folds and the refit disagree (not fold size).

## Cycles 23–24 — applying the decisions of report 015 §10 (2026-11-13)
Felix approved all six decisions. Proposal 026; report 015 §11. Binaries `/tmp/rb_c23` (defaults,
`dist_select`), `/tmp/rb_c24` (CRPS). Tests: 264 unit, 129 integration (XGBoost); LightGBM compiles.

### Cycle 23 — four family defaults, `dist_select`, two code leads, the guard helper **[DEFAULT CHANGE]**
`run_c23.sh`; tags `c23_default`, DSEL log `/tmp/c23_dsel.log`; bench `BENCH_LS_PORT=2` (`new` as is).
- **Defaults:** `Cauchy::new`, `LogNormal::new` → Damped + Location; `Logistic::new`,
  `Laplace::new` → Damped + LocationScale. Serde defaults stay Observed / Raw. Gumbel unchanged.
  **190/190 cells equal `c18_port` / `c15_d_loc` / `c15_d_ls` bit for bit** (D1 ✓).
- **`dist_select`:** new trait method `Distribution::marginal_fit` (default
  `calculate_start_values`); StudentT returns the free fit. **Picks unchanged: 22/45, regret
  0.0543** (D2 ✗, D3 ✗). protein: free StudentT 3.2232 = Gaussian 3.2232. The capped start df was
  an artefact, not the cause.
- **Spline flow:** start fit without `target_cost(0.0)` (test: −0.61 → −2.25 NLL per row).
- **LogNormal:** y ≤ 0 rows get gradient 0 and Hessian 1e-6 on both paths and in every mode.
- **`hyper_opt::validation_guard`** (opt-in): fits the tuned and the baseline configuration with
  validation early stopping, returns the better; inherits the model settings a fold model does.
- One existing test adjusted (`test_cauchy_analytical_vs_numerical_gradients` now asks for
  Observed curvature explicitly).

### Cycle 24 — CRPS training for large counts **[BEHAVIOUR CHANGE for counts above the old caps]**
`run_c24.sh`; tag `c24_crps`, NLL arm `c20_nll`; `c20_crps_summary.py <csv> c24_crps`.
- **Sums:** Poisson and NB loss, gradient and Hessian tables cover the distribution (window of
  mean ± 10 sd for Poisson; NB table starts at 0 unless pmf(0) underflows); at most 200 000 terms
  per row, then one warning. Rows under the old caps (100 / 150 / 220) run the old loops.
- **Smoke, sums only:** Poisson bike_cnt trains but is at 56.6 after 2000 rounds. The metric
  E[g²] ~ λ²/3 outgrows the CRPS curvature ~ λ^1.5/√π: steps too small by ~0.6·√λ.
- **Added (proposal amended before the run):** large-count rows use the expected CRPS curvature
  2 Σ_k (∂F(k)/∂θ)². Tests: equals the pmf-weighted mean of the observed curvature (Poisson) and
  an independent CDF finite difference (NB) within 1 %.
- **Result (test CRPS, relative to NLL training):** Poisson bike_cnt +6.5 % (was +640 %),
  c_seoulbike +4.7 % (+562 %), bike_casual +5.2 % (+48 %); NB bike_cnt +11.5 % (+116 %),
  c_seoulbike +1.2 % ± 1.4 (+485 %), bike_casual +3.7 % (+8.2 %).
- **Unchanged:** c_abalone identical in 10/10 cells (NB there still +34 % at the cap). CRPS
  training beats NLL training by > 1 % on 0/10 pairs; 29/50 cells at the round cap.
- **Predictions:** R1 ✗ by the letter (synth_pois and synth_nb_lo hold rows above the old caps and
  move by ≤ 1.2 % per cell), R2 ✓, R3 ✓, R4 ✓, R5 ✓.
- README: new "results change" sections for the defaults and the CRPS path; `validation_guard`;
  `dist_select`; the normalised axis recommended for two-parameter searches (decision 6).

## Cycles 25–26 — the four open items of report 015 (2026-11-14)
Felix: "can we look into this" (Gumbel, NB CRPS in mean/dispersion coordinates, which added
search axis carries the wide-space gain, adaptive λ). Proposals 027–030, report
`reports/016-cycles25-26-report.md`. No default changed; nothing committed.

### Cycle 25 — which axis carries the wide-space gain (proposal 027) **[no recommendation; observation on subsampling]**
`run_c25.sh <depth|subs|d6> <dev|unseen|all> <group>`, binary `/tmp/rb_c25`; tags `c25_depth_k<k>`,
`c25_subs_k<k>`, `c25_d6`; `c25_axis_summary.py <csv> <dev|unseen> 2d=… depth=… subs=… wide=…`.
- Mean test gain over the untuned default at 20 trials, development / unseen: two axes −0.008 /
  −0.017, `depth` −0.055 / −0.081, `subs` −0.054 / −0.055, five axes −0.089 / −0.131, untuned
  depth 6 −0.002 / −0.032.
- **The losses are depth:** Gaussian c_gt_co `depth` +0.168 (`subs` +0.011), c_supercond +0.067
  (−0.024), protein +0.029 (−0.068). The guard misses c_gt_co (+0.149).
- **`subs` never lost to two axes** (≤ +0.002 on 24/24) and guarded is worse than the default on
  0/24 (gain −0.052 / −0.045 against −0.019 / −0.023). Post hoc.
- Depth 6 untuned is worse than depth 4 on 9/24 pairs (Gaussian c_supercond +0.58).
- **Predictions:** A1 ✗, A2 ✗, A3 ✗ (the opposite), A4 ✗ / ✓, A5 ✗.

### Cycle 26a — Gumbel (proposal 028) **[opt-in; default change put to Felix]**
`run_c26.sh gb <base|ls<k>> <dev|unseen>`, binary `/tmp/rb_c26`; tags `c26_gb_<arm>`;
`c26_gumbel_summary.py <csv> <arm>=<log> …` (logs carry the worst rows).
- **Diagnosis:** every blow-up cell (27 across arms) is a test row at z < −5, costing e^{−z}; CRPS
  unaffected. Family property. The default also stalls on targets away from 0.5 (synth_gumbel:
  3.50 above the oracle, scale 115).
- **`Gumbel::with_start_scale_factor(k)`** with Damped + LocationScale. Validation NLL improves
  monotonically in k on naval and kin8nm, flat elsewhere; k = 100 chosen (two amendments: grid
  extended, statistic mean rank → mean validation NLL; the as-written pick k = 4 reported).
- **k = 100 against the default:** mean-NLL rule 10/11 (power_plant 7.43 → 2622: seeds 2 and 3
  hold one lower-tail row each; median seed 3.12), trimmed NLL 11/11, CRPS 11/11. c_airfoil
  6.13 → 2.51, c_parkinsons 14.77 → 2.91, c_supercond 49 992 → 4.29, c_gt_co +0.002 ± 0.005.
- **Predictions:** G1 ✗ (0.087 above the oracle, threshold 0.05), G2 ✓, G3 ✓, G4 ✓, G5 ✓.
- 75 cells repeating 021's arms: identical.

### Cycle 26b — NB CRPS in (log µ, log θ) (proposal 029) **[opt-in; default change put to Felix]**
`run_c26.sh nbmd`; tag `c26_nbmd`; `c20_crps_summary.py <csv> c26_nbmd`.
- `NegativeBinomial::with_mean_dispersion_crps()`: chain rule through (θ, µ/(µ+θ)), expected CRPS
  curvature for every row.
- **Against the fallback coordinates:** better in 25/25 cells (−2.3 % to −25.1 %); c_abalone
  +33.6 % → +0.1 % of NLL training, no fit at the round cap.
- **Against NLL training:** bike_casual −1.2 % ± 0.9, bike_cnt −2.7 % ± 1.1, c_seoulbike −5.9 % ±
  1.0; test NLL +0.06 / +0.13 / +0.63; cov90 0.865 / 0.847 / 0.785 against 0.899 / 0.928 / 0.934.
- **Predictions:** N1 ✓, N2 ✓, N3 ✓, N4 ✗ (CRPS training wins on 3/5, predicted ≤ 1), N5 ✓, N6 ✓.

### Cycle 26c — gradient-regularised Newton boosting (proposal 030) **[rejected; kept as an undocumented-in-prelude option]**
`run_c26.sh grn <eta> <van|grn|grnx>`; tags `c26_e<eta>_<arm>`; `c26_grn_summary.py <csv>`.
- `GradientLSS::with_gradient_regularization`: h_ij += c·h̄_j·ρ_j^{1/4} (ρ = mean g²/h̄), or the
  excess over ρ = 1. Paper read in full text (Lacuna): the penalty is on the L² norm of the tree,
  so it is a per-row Hessian term, not XGBoost's per-leaf lambda.
- vanilla eta 0.3 is worse than 0.05 by > 0.02 on 9/12 pairs; eta 1 has 11/60 failed cells.
- `grn` beats vanilla at the same eta (8/12 at 0.3, 12/12 at 1) and **loses to vanilla at half the
  eta** (worse on 8/12 and 5/12). `grnx` equals vanilla on five pairs (ρ ≤ 1 from round 0).
- **Predictions:** R1 ✓, R2 ✓ `grn` / ✗ `grnx`, R3 ✗ / ✓, R4 ✗, R5 ✓.

### Operations
- Two CSV rows were interleaved by three concurrent streams of sub-second cells; cells re-run,
  garbled lines removed once nothing was writing. Check: `awk -F, 'NF != 34'`.
- `BENCH_SEED_START=k` added (resume a cut stream; verified bit-identical on one search).
- README: section "Research cycles 25–26"; three older statements now point to it.

## Cycles 27–28 — the decisions of report 016, applied (2026-11-14)
Felix: "yes to all of these" (report 016 §6). Either/or questions 3 and 4 read as their first
option. Proposals 031, 032; report 016 §8. Nothing committed.

### Cycle 28 — defaults **[BEHAVIOUR CHANGE: Gumbel; NegativeBinomial under LossFn::Crps]**
`run_c28.sh <gb|nb|nb_tcp>`, binary `/tmp/rb_c28`; tags `c28_gb_default`, `c28_nb_crps`,
`c28_nb_crps_tcp`.
- **Gumbel:** `Gumbel::new` → Damped + LocationScale + start-scale factor 100
  (`GUMBEL_START_SCALE_FACTOR`, exported). Serde defaults Observed / Raw / 1.
  **E1 ✓: 55/55 cells = `c26_gb_ls100` bit for bit.**
- **NB CRPS:** `LossFn::Crps` uses the MeanDispersion head in `new`, `new_mean_dispersion`,
  `with_parameterization`; `with_mean_dispersion_crps()` removed (never committed). Opt-out
  `new(..).with_parameterization(TotalCountProbs)`. **E2 ✓: 25/25 = `c26_nbmd`; E3 ✓: the opt-out
  = `c24_crps` in 25/25.**
- Tests: 267 + 129 pass; three rewritten to the new defaults, five NB CRPS tests now request the
  (total_count, probs) head.
- `GradientRegularization` kept as is.

### Cycle 27 — the subsampling axes on fresh datasets (proposal 032) **[safe with the guard; not reliably better]**
`run_c27.sh <def|2d|subs> <group>`, binary `/tmp/rb_c25`, one CSV per stream; tags `c27_def`,
`c27_2d_k<k>`, `c27_subs_k<k>`; `c27_fresh_summary.py <csv>`. Data: seven UCI sets fetched
2026-11-14 (`data_confirm/n_*`, `prep_confirm_data2.py`), 12 pairs.
- Guarded `subs` worse than the default on 0/12 (plain 0/12); two axes 2/12 plain, 1/12 guarded
  (Gaussian n_appliances +0.008: the guard's first miss on two axes).
- `subs` − two axes: better on 10/12; worse on NB n_news (+0.011 ± 0.006) and Poisson n_metro
  (+0.165 ± 0.255).
- Means −0.150 against −0.134 (guarded −0.139 / −0.124, ratio 1.12); medians −0.043 / −0.026.
- **Predictions:** C1 ✓, C2 ✗, C3 ✗ (ratio; 10/12 pairs ✓).
- README: tuning advice says safe with the check, usually better; not "always add them".

## Cycle 29 — `max_bin` as a search axis (2026-11-14)
Felix: "lets look into whether it is a parameter that should be included in search space".
Proposal 033; report 016 §9. Nothing committed.

### Cycle 29 — **[not included: the rule fails on all three counts]**
`run_c29.sh mb<k> all 1` and `run_c29.sh bin <dev|unseen|fresh> <group>`, binary `/tmp/rb_c29`,
one CSV per stream; tags `c29_mb{16,64,256,1024}`, `c29_bin_k{10,20}`;
`c29_maxbin_summary.py <csv> bin=<logs>`. Bench additions: `BENCH_MAX_BIN`,
`BENCH_HPO_SPACE=bin|subsbin`.
- Untuned against 256 bins, 36 pairs: 16 bins worse on 25 / better on 7; 64 bins 12 / 12; 1024
  bins worse on 16 / better on 4. 256 rerun identical on 180/180 cells.
- `bin` (mcw, λ, max_bin 16–1024) at 20 trials: guarded worse than the default on 2/36 (Gaussian
  protein +0.123, Poisson c_seoulbike +0.078); better than guarded two axes on 23/36; median
  difference −0.003. Plain: better by > 0.005 on 17, worse on 8; means −0.055 / −0.054.
- Stable picks on kin8nm (16–30), naval (234–972), Gamma c_parkinsons (25–32); +0.048 ± 0.026
  on Gaussian n_sgemm, where max_bin cannot change the model.
- **Predictions:** M0 ✓, M1a ✗, M1b ✓ / ✗, M2 ✗ / ✓, M3 ✗.
- Stage 2 (`subsbin`) not run. README: tested, not recommended; leave 256.

## Cycle 30 — `max_bin` at 50 trials, the two-value choice, and a fresh confirmation (2026-11-14)
Felix: "lets test this" (the two items cycle 29 left open). Proposals 034 and 038; report 016
§10–11. Nothing committed.

### Cycle 30a — **[`bin` at 50 trials passes on the 36 pairs; `bin2` fails]**
`run_c30.sh wave <streams> <minutes>` (one job per arm, pair, seed in `/tmp/c30/`, resumable),
binary `/tmp/rb_c30`; tags `c30_2d_k{20,50}`, `c30_bin_k{20,50}`, `c30_bin2_k{10,20}`;
`c30_summary.py <csv> <logdir> <arm>=<ref>@<k>` and `c30_score.py`. Bench addition:
`BENCH_HPO_SPACE=bin2`.
- `bin` at 50 against two axes at 50: guarded worse than the default 0/36, better 30/36, median
  −0.009. `bin2` at 20: 3/36, 24/36, −0.002.
- Two axes at 50 trials are not better than at 20 (guarded: better on 17/36, worse than the
  default on 3 against 1).
- **Predictions:** R0 ✓, B1 ✗ / ✓, B2 ✗, B3 ✗, B4 ✓, C1 ✓, C2 ✓, C3 ✓, C4 ✓.
- One wave was stopped at the 2-hour task limit with two long jobs unfinished; they were rerun.

### Cycle 30b — **[fresh data: the gain confirms, the rule fails on one guard miss]**
`prep_confirm_data3.py` (seven UCI sets, `c_m_*`), `run_c34.sh`, tags `c34_def`,
`c34_2d_k{20,50}`, `c34_bin_k{20,50}`; `c30_summary.py /tmp/c34 /tmp/c34 base=c34_def pairs=12
c34_bin=c34_2d@50`.
- `bin` at 50: guarded worse than the default 1/12 (Gamma m_energy +0.029, 768 rows), better
  than guarded two axes 10/12, median −0.015. At 20 trials: 0/12, 9/12, −0.005.
- **Predictions:** D1 ✗, D2 ✓, D3 ✓ / ✗, D4 ✓, D5 ✗, D6 ✓, D7 ✓.
- README: `max_bin` paragraph extended; advice unchanged ("leave 256"), with the 40 of 48.

## Cycle 31 — eight untested XGBoost parameters as search axes, and the leaf-wise confirmation (2026-11-14/15)
Felix: "queue these to test after we are done with the current running test" (items 1–5 of his
list). Proposals 035 and 039; report 016 §12–13. Nothing committed.

### Cycle 31a — **[seven axes fail at 20 trials; leaf-wise growth with a leaf budget passes]**
`run_c31.sh wave <streams> <minutes> [arm ...]` (one job per arm, pair, seed in `/tmp/c31/`),
binary `/tmp/rb_c31`; tags `c31_u_{repro,cap6000,eta015,lg16}`,
`c31_{eta,gamma,alpha,ctree,clevel,cnode,mds,leaf}_k{10,20}`; `c31_score.py <csv> <logdir>
<arm ...>`. Bench additions: `BENCH_XGB=key=value,...`, `BENCH_HPO_SPACE=eta|gamma|alpha|ctree|
clevel|cnode|mds|leaf`.
- (a) / (b) / (c) at 20 trials: eta 2 / 20 / −0.0003; gamma 2 / 14 / +0.0036; alpha 3 / 28 /
  −0.0045; ctree 0 / 22 / −0.0044; clevel 3 / 22 / −0.0032; cnode 1 / 28 / −0.0080; mds 2 / 15 /
  +0.0007; **leaf 0 / 34 / −0.0381**.
- Untuned: cap 6000 better on 6, worse on 0; eta 0.15 worse on 22; lossguide-16 better on 27,
  worse on 5.
- **Predictions:** S0 ✓, E0 ✓, E1 ✓, E2 ✓ (one naval pick ✗), R1 ✓ / ✗ (alpha's (b)), K1 ✓ / ✓ /
  ✗, D1 ✓, L0 ✓, L1 ✓ / ✗.
- Ops: two drive disconnects ended waves (xargs could not find the script). The run scripts now
  copy themselves to the job directory and each job waits for the drive.

### Cycle 31b — **[the leaf-wise axis confirms on the 12 `c_m_*` pairs]**
`run_c35.sh`, tags `c35_u_repro`, `c35_u_lg16`, `c35_leaf_k{10,20}`; references `c34_def`,
`c34_2d_k20`; `c35_score.py <csv>,/tmp/c35 /tmp/c35`.
- `leaf`: guarded worse than the default 0/12, better than guarded two axes 12/12, median
  −0.068. `u_lg16`: better on 10, worse on 2.
- **Predictions:** F0–F6 ✓.
- README: new paragraph in the cycles 25–26 section (the eight axes, the leaf-wise
  recommendation and its limits).

## Cycle 32 — LightGBM's own parameters as search axes (2026-11-15)
Felix's queue, item 6. Proposal 036; report 016 §14. Nothing committed.

### Cycle 32 — **[all four fail at 20 trials; any λ ≥ 0.001 breaks Gaussian naval on LightGBM]**
`run_c32.sh wave <streams> <minutes> [arm ...]`, binary `/tmp/rb_c32_lgbm` (`--features
lightgbm`), tags `c32_l_def`, `c32_{l_2d,lleaves,lmdl,lff,lbag}_k{10,20}`; `c32_score.py <csv>
/tmp/c32`. Bench additions: `BENCH_HPO_SPACE=lleaves|lmdl|lff|lbag`.
- (a) / (b) / (c) at 20 trials against `l_2d`: lleaves 1 / 25 / −0.0238; lmdl 2 / 19 / −0.0002;
  lff 2 / 22 / −0.0027; lbag 4 / 24 / −0.0065. `l_2d` itself: guarded worse than `l_def` on 3/36.
- `l_def` against XGBoost untuned: better on 21, worse on 8.
- Gaussian naval: tuned refits stop at 77–99 rounds (+0.46 to +1.42 against −6.1). Probe in
  `/tmp/c32probe`: `BENCH_XGB=lambda=0.001` reproduces it untuned; λ = 0 does not.
- **Predictions:** G0 ✗ / ✓, G1 ✓, G2 ✗ / ✓, G3 ✗.

## Cycle 33 — library-level settings chosen per dataset (2026-11-15)
Felix's queue, item 7. Proposal 037; report 016 §15–16. Nothing committed.

### Cycle 33 — **[nothing passes; the start-value choice misses by one pair]**
`run_c33.sh wave <streams> <minutes> [arm ...]`, binary `/tmp/rb_c33`, tags
`c33_v_{obs,fis,pesoff,init,locflip}`, `c33_pes_k{10,20}`; `c33_score.py <csv> /tmp/c33` (parts
P picks, V, F) and `c30_summary.py <csv> /tmp/c33 c33_pes=2d@20`. Bench additions:
`BENCH_HPO_SPACE=pes`, `BENCH_PES=0`, `BENCH_INIT=1`.
- `pes` at 20 trials: 1 / 20 / −0.0003. Picked windows spread over 5–80.
- Validation choices, better / worse: mode 12 / 9; start 16 / 1 (Gamma c_n_wine +0.0052); PES
  0 / 1; all six 20 / 5.
- Untuned: Observed and PES off worse on 28/36 each; Fisher non-finite on Gaussian power_plant.
- Family by validation: 65/75 right; no better than always Gamma / NegativeBinomial.
- **Predictions:** P1 ✓, P2 ✓, V1 ✓ / ✗, V2 ✓, V3 ✓, F1 ✓.

### 2026-11-15 — decisions on the queue's seven questions (report 016 §16)
- Felix: one guard miss in 48 does not keep `max_bin` out. README recommends it at ≥ 50 trials
  with the guard, as a stated exception to 038's rule (which it failed).
- On my recommendation: `leaf_wise_search_space()` and `max_bin_search_space()` added to
  `hyper_opt` (prelude, unit test); round cap and LightGBM λ = 0 as README advice only; no trial
  of a leaf-wise default, no rerun of LightGBM `num_leaves`, no fresh-data test of the start
  choice.
- README: "At a glance" table at the top of Breaking Changes and a section for the two spaces.
