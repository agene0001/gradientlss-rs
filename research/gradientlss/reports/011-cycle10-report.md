# Cycle 10 report — adaptive wide start, Gamma start units, tail-robust stopping

Date: 2026-11-11. Rows: `experiments_v2.csv` tags `c10*`; detail in `research_log.md` "Cycle 10".
Proposal 013 (+ Addendum A′). **One default change, made after review: Gamma's unit-free start (§2).**

## 1. Adaptive "start wide" for StudentT
- **Kurtosis gate: rejected before any fit.**
  - The training labels' excess kurtosis measures the signal, not the noise:
    - gaussian_sim reads heavy (3.6);
    - protein and power_plant read light (−1.1, −1.06).
  - So the gate would inflate the two sets where a wide start hurts (+0.091, +0.054) and skip the one
    it helps.
  - `StudentTStart::Adaptive` ships as a documented failing research option.
- **"Approach the scale from above" generalises to Gaussian noise.** Three new held-out sims:
  - synth_g_sin −0.012 (5/5);
  - synth_g_exp −0.0024 (5/5);
  - t₄ noise +0.015 (0/5).
- **Picking the start by validation NLL** (fit both, 2× cost) is right on 9 of 10 sets:
  - gaussian_sim −0.0088, naval −0.063, synth_g_sin −0.012;
  - the heavy-tailed sets are untouched (0.000).
- **It fails on power_plant (+0.050).** On two seeds validation preferred the wide start, which had
  driven df to ~1000, and the test split's outliers then cost +0.10 / +0.15. Not a default.

## 2. Gamma start: unit dependence and a deliberate wide start
- **The old raw start depends on the target's units**, though less than predicted. At labels × 100
  (ln c adjusted):
  - kin8nm +0.066 (5/5 worse);
  - others +0.005 to +0.010.
- **The unit-free wide start [ln ȳ + 0.5, 0.5]** ties the old start at the original units and wins at
  rescaled units:

  | c | kin8nm | naval | power_plant | synth_gamma |
  |---|---|---|---|---|
  | 1 | +0.001 | +0.007 ± 0.042 | +0.002 | −0.016 |
  | 0.01 | −0.017 | −0.010 | −0.007 | −0.006 |
  | 100 | **−0.056 (5/5)** | −0.009 | −0.011 | −0.020 |

  Naval at c = 1 is +0.007. That is noise-level, but it is outside the +0.005 band, so under the strict
  rule it would stay. **Felix accepted it (2026-11-11): it is now the default** (`Gamma::with_wide_start`,
  on by default; saved models keep the raw start).
- **Starting the mean above ȳ beats starting below:** kin8nm by 0.025 (4/5). Shape α₀ = 1 is worse on
  naval (+0.057).
- **Side finding: the fit itself is not exactly unit-free**, even from a unit-free start (per-seed
  swings up to 0.1 on naval). Two causes:
  1. xgboost sees f32 labels and gradients.
  2. Early stopping's min-delta is relative to |mean NLL|, and the mean NLL shifts by ln c under
     rescaling. kin8nm's stopping round moved by up to 600.

  The traits.rs comment also still calls the metric a sum; it is a per-row mean now.

## 3. Capped per-row loss for the stopping decision
`GradientLSS::with_stopping_row_cap(K)` softens validation rows above median + K. It affects the
stopping decision only.

| K | Weibull | StudentT | Gaussian | Gamma |
|---|---|---|---|---|
| 10 | naval +0.88 (one seed +4.2), power_plant +0.097 | power_plant +0.016, protein +0.011 | power_plant +0.004 | naval +0.037 (0/5) |
| 25 | power_plant +0.032 | — | power_plant +0.005 (0/5) | — |

Everywhere not listed it is identical or within noise.

The extreme validation rows are the evidence that the fit is getting overconfident. Hiding them lets it
keep sharpening, so CRPS improves while the NLL tail gets worse. Rejected; documented as not
recommended.

## 4. Next investigations (ranked)
1. Make early stopping's patience threshold unit-free. The min-delta is relative to |mean NLL|, which
   is meaningless for a log-density (it can be near 0 or negative). Use an absolute per-row threshold
   in nats (e.g. 1e-4) and measure:
   - the change at c = 1 (it should be tiny);
   - unit invariance under BENCH_Y_SCALE.
2. Done: Gamma's wide start landed as the default (§2).
3. A cheaper validation-based start choice for StudentT. Run both starts for ~W rounds only and keep
   the better one, which brings the cost from 2× to ~1.1×. power_plant's test-only outliers remain the
   blocker.
