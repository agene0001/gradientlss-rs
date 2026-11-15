# Cycle 4 report — orthogonal coordinates for shape/dispersion parameters (Gamma, NB); β-NLL × freeze

Date: 2026-11-09. Rows: `experiments_v2.csv` tags `c4a_*` … `c4d_*`; detail in
`research_log.md` "Cycle 4". Suites: 235 lib (both backends), 137 / 129 integration.

## 1. Measured limitation
Gamma on power_plant: 90 % coverage 1.000, PIT-KS 0.38, NLL 4.11 (Gaussian 2.81). The
concentration had to climb from the MLE start α ≈ 330 to ≈ 12 000 and did not.

## 2. Competing explanations
| explanation | test | verdict |
|---|---|---|
| bad start values | crate start = scipy MLE to 4 decimals | rejected |
| shape step too small (tuning) | derivation + numerics below | true, but the *cause* is the coordinates |
| **collinear boosting coordinates** | exact Fisher in both frames | **confirmed** |

## 3. Math (verified numerically: scipy quadrature for Gamma, exact summation for NB)
- Gamma, (log α, log β): Fisher `[[α²ψ′(α), −α], [−α, α]]`, correlation −0.9992 at α = 330 —
  α moves the mean α/β. One diagonal-Newton step in log α (at the right mean) moves 0.0015 of
  the 3.59 needed (≈ 2400× too slow).
- Gamma, (log µ, log α), β = α/µ: Fisher `diag(α, α²ψ′(α) − α)` **exactly** (Cox & Reid 1987
  parameter orthogonality); the same step is 0.97. α²ψ′(α) − α → ½.
- NB, (log r, logit p) default: correlation +0.876 / +0.951 / +0.988 / +0.995 for
  µ = 3 / 20 / 400 / 2000; (log µ, log θ): |corr| < 1e-5. The crate's existing
  `MeanDispersion` head is the orthogonal fix, but clamped µ ≤ e⁶ ≈ 403.

## 4. Implementation (all tested)
- `GammaParameterization::{ConcentrationRate (default), MeanShape}`, `Gamma::new_mean_shape`:
  joint transform (outputs still `(concentration, rate)`), exact-MLE starts (µ̂ = ȳ; Newton on
  ln α − ψ(α) = s from Minka's start), cancellation-safe series for ψ(α) − ln α and
  α²ψ′(α) − α, zero gradient outside clamps, `HessianMode` honoured. Tests: FD derivatives
  (α 0.8 … 12 000), Monte-Carlo orthogonality + E[observed] = Fisher, MLE recovery, serde default.
- NB `MeanDispersion` clamps widened to log µ ∈ [−10, 15], log θ ∈ [−6, 12] (constants
  `NB_LOG_MU_CLAMP`, `NB_LOG_THETA_CLAMP`), zero gradient beyond; test at µ = 3000.
- Harness: `gamma_ms`, `nb`, `nb_md` families; closed-form Gamma CRPS and exact discrete NB
  CRPS / randomized PIT (Monte-Carlo self-tests); `synth_gamma`, `synth_nb_lo`, `synth_nb_hi`
  DGPs with oracles.

## 5. Results (Damped, 5 paired seeds)
**Gamma, MeanShape − ConcRate** (η .05/d4 ; η .1/d6):

| dataset | ΔNLL | cov90 | PIT-KS | best_iter |
|---|---|---|---|---|
| power_plant | −1.326 ± 0.047 ; −1.363 (5/5) | 1.000 → 0.901 | 0.383 → 0.025 | 1999 → 169 |
| naval | −4.662 ± 0.047 ; −4.363 (5/5) | 1.000 → 0.813 | 0.491 → 0.080 | 1999 → 771 |
| synth_gamma | −0.646 ± 0.011 ; −0.617 (5/5); oracle gap 0.14 | 0.926 → 0.876 | 0.166 → 0.024 | 939 → 173 |
| kin8nm | **+0.050 ± 0.008 ; +0.063 (0/5)** | 0.915 → 0.839 | 0.088 → 0.076 | 1364 → 220 |

kin8nm (pre-registered H-k): the fast shape tree over-fits and early stopping truncates the
mean. Freezing the shape at 200 rounds: −0.150 vs MeanShape, −0.100 vs ConcRate (5/5).
With the freeze round validation-selected from {∅, 50, 100, 200}, MeanShape beats ConcRate
5/5 on all four datasets (kin8nm −0.100, power_plant −1.427, naval −4.662, synth_gamma −0.645).
MeanShape had 0 non-finite rows in 120 fits (the legacy head had one 1565-row failure).

**NB** (oracle-known Gamma–Poisson counts):

| dataset | default (r, p) | MeanDispersion, µ ≤ 403 | MeanDispersion, widened | oracle |
|---|---|---|---|---|
| synth_nb_lo (µ 1.5–60) | 3.011 | 2.937 (−0.074, 5/5) | 2.937 (bit-identical) | 2.912 |
| synth_nb_hi (µ 100–4000) | 7.581 (cov90 0.995) | 7.803 (+0.222; RMSE 1441) | **6.722 (−0.859, 5/5); CRPS 306 → 206** | 6.644 |

**β-NLL × scale freeze** (Gaussian): largely overlapping. Only protein stacks (−0.021 and
−0.072 alone, −0.114 together). Validation selection over the full 3×3 menu vs the freeze-only
menu: −0.009…−0.029 on the real targets (< 1.5 s.e. each), +0.007 on synth_t3 (selection noise).

## 6. Decisions
- **NB clamp widening landed** (a correction: identical in-range, removes a hidden cap).
- **Gamma MeanShape shipped, not default** — default change is Felix's call: alone it wins
  3/4 by 0.6–4.7 nats and loses kin8nm by 0.05; with a validation-tuned shape freeze it wins 4/4.
- **NB MeanDispersion vs default**: MeanDispersion wins both DGPs (−0.07, −0.86); making it the
  default is likewise Felix's call (it changes the head and drops CRPS support).
- Freeze first; β-NLL is a minor addition once the freeze is tuned.

## 7. Next investigations (ranked)
1. NB `MeanDispersion` log-θ curvature: observed with a 0.01 floor vs Fisher → ½ — apply
   `HessianMode::Damped` (needs E over y for the Fisher term; closed-form via trigamma sums).
2. Real count data for NB (no local count dataset yet — bike-sharing or similar).
3. Automatic shape freeze: stop each parameter on its own validation contribution instead of
   a tuned round (per-parameter early stopping).
4. Same orthogonalisation audit for the remaining 2-parameter shape families (Beta mean/precision,
   Weibull, LogNormal is already orthogonal).
