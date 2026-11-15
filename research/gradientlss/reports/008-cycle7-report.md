# Cycle 7 report — StudentT start, early-stopping hold, naval cap, Beta/Weibull audit

Date: 2026-11-10. Rows: `experiments_v2.csv` tags `c7*`; detail in `research_log.md`
"Cycle 7". Proposals 009 (+ Addendum S3) and 010. Felix: Python parity is not a constraint.

## 1. StudentT start values (the location starts at the median)
- **Limitation:** without `initialize`, every parameter started at a raw 0.5. On y ≈ 450
  (power_plant) the t score r/(νσ² + r²) vanishes for large residuals, so the location never
  moved: RMSE 448, NLL 7.52.
- **Candidates:**

  | candidate | power_plant | elsewhere | verdict |
  |---|---|---|---|
  | `initialize = true` (fitted) | −4.73 | protein +0.07, kin8nm +0.02, naval +0.14 (with stopping) | ✗ |
  | median + MAD scale | −4.72 | protein +0.03, kin8nm +0.02, naval +0.11 (with stopping) | ✗ |
  | **median location only** | **−4.75 / −4.95** | ≤ ±0.008; gaussian_sim +0.006–0.008; naval with stopping +0.018 ± 0.028 | ✗ strictly; landed |

- **Reading:** a scale that starts large relative to the residuals keeps early location
  learning in the t score's near-linear (Gaussian-like) regime. Only the location needs to
  move.
- **Decision:** `StudentTStart::Median` is the default (`Raw` / `MedianMad` available; saved
  models load as `Raw`). It misses the +0.005 neutral band on gaussian_sim, and naval with
  stopping is noisy and leans worse. **Felix: keep it, or revert to Raw?**
- **New hook:** `Distribution::default_start_values`, used only without `initialize`.

## 2. Early-stopping hold during anneal
- **Mechanism:** each per-parameter trigger restarts the `early_stopping_rounds` patience
  counter. It works through the new provided method
  `TrainingCallback::holds_early_stopping` (non-breaking); the model wraps the caller's
  callback.
- **Results:**
  - Gaussian and Gamma power_plant −0.03 (the one seed that fell back to "none", −0.17 on
    that seed).
  - Bit-identical everywhere else, including all NB and StudentT runs.
  - Protein's fallback seed is real: its location had peaked.
- **Decision:** on by default (`with_hold_early_stopping(false)` turns it off).

## 3. Naval round cap
With per-parameter stopping, naval converges at 2600–3200 rounds. Going from a 2000 to a
6000 cap gains Gaussian −0.05, Gamma −0.03, StudentT −0.04. Against no per-parameter
stopping the gains are −0.42 / −0.22 / −0.39. This changes no default; it is in the README.

## 4. Beta / Weibull orthogonality audit
- **Beta:** in (log a, log b) the correlation → −1 as φ grows (−0.95 at φ = 20, −0.99 at
  φ = 100). New `BetaParameterization::MeanPrecision` boosts (logit µ, log φ):
  - cancellation-safe forms, moments start, outputs still (a, b);
  - tested: finite differences up to φ = 10⁶; E[observed] = Fisher; |ρ| < 0.1 at φ = 55.
- **Beta results (5/5 seeds everywhere):**

  | dataset | MeanPrecision (Damped) vs old default | vs old head with Damped |
  |---|---|---|
  | synth_beta_hi (φ 55–1100) | −0.27 | −0.28 |
  | synth_beta (φ 3–55) | −0.10 | −0.075 |
  | bike_share (real) | old default diverged (NLL 2545) | −0.04 |

  The oracle gap on synth_beta_hi shrinks from 0.34 to 0.06.
- **Weibull:** the correlation is −0.31 everywhere, so a diagonal step keeps 90% efficiency.
  η = log λ + (1−γ)/k is exactly orthogonal but was not built. Damped vs Observed:
  power_plant −0.15, kin8nm −0.08, synthetic −0.05 (all 5/5); naval −0.05 (4/5, one test-only
  +1.43 seed).
- **Harness bug fixed:** statrs' Weibull CDF returns NaN for shape ≳ 100, so power_plant rows
  had been silently dropped. Superseded by c7w2.
- **Decision:**
  - Beta `Exp` + NLL defaults to MeanPrecision + Damped; the Concentrations head defaults to
    Damped.
  - Weibull defaults to Damped.
  - Old files load with the old head and observed curvature.

## 5. Next investigations (ranked)
1. Per-parameter early stopping for Beta (log φ) and Weibull (log k): add them to the
   default watched sets if they are better or neutral.
2. Whether StudentT's gaussian_sim cost is the start or the path. Test a start-scale
   multiplier, or ν₀ at the cap.
3. Weibull on targets far from 0 still starts at raw 0.5 (λ = 1.65 on y ≈ 450). A moments
   start through `default_start_values`, as for Beta.
4. Gamma MeanShape / NB MeanDispersion also start at raw 0.5 without `initialize`. Their MLE
   starts exist but only run with `initialize = true`.
