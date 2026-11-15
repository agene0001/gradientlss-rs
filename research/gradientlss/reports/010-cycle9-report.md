# Cycle 9 report — Weibull tail blow-ups, deliberate warm-up, Gamma raw vs fitted start

Date: 2026-11-10. Rows: `experiments_v2.csv` tags `c9*`; detail in `research_log.md`
"Cycle 9". Proposal 012 (+ Addendum G3). **No default changes this cycle.**

## 1. Weibull upper-tail blow-ups
- **Diagnosis:** a single test row dominates.
  - power_plant: y = 492 against λ̂ = 476, k̂ = 223, costing 1314 nats (20% of that fit's
    test NLL).
  - naval: a row at the target's upper bound, costing 5711 nats.
- **Math:** the upper-tail cost is (y/λ)^k = e^z, with z ≈ k ln(y/λ) the distance in predicted
  standard deviations. That is exponential in z for any shape (a Gaussian charges z²/2), so a
  shape cap can only widen every prediction.
- **Measured** (no cap → cap):

  | cap | naval | power_plant (10 seeds) |
  |---|---|---|
  | 50 | +2.77 | +0.48 |
  | 200 | +1.40 | −0.04 |
  | 1000 | +0.08 | 0 |

- **Decision:** the cap ships as an opt-in (`with_max_concentration`), and the README says to
  prefer Gamma when the upper tail has outliers. It is a property of the family, not a bug.

## 2. Deliberate warm-up: what the raw StudentT start was really doing
- **Location-only warm-up** (`with_warmup_rounds`): makes gaussian_sim *worse* (+0.009 at 50
  rounds), though it helps the real targets slightly. Rejected as an explanation and as a
  default; kept as an opt-in.
- **Starting the scale at 3× the target's standard deviation** reproduces the raw start's
  benefit exactly (gaussian_sim −0.0088; same df ≈ 185 and same scale). So the mechanism is
  "approach the scale from above".
- **But approaching from above hurts heavy-tailed and real targets:** protein +0.091,
  power_plant +0.054, synth_t3 +0.020, studentt_sim +0.013. It suits data that really is
  Gaussian, and the median start is the right default.

## 3. Gamma: why the raw start beats the fitted one on naval
- **Each half of the raw start is worth about the same:**
  - fitted mean only: +0.046
  - fitted shape only: +0.044
  - both: +0.155
  - the fitted start plus a warm-up recovers about half.
- **Not a round-budget artefact:** all naval fits had sat at the 2000-round cap. At 6000 rounds
  both converge (best 3005 / 2562), and the raw start is still 0.175 better (5/5).
- **Path dependence:** the raw start is heavily overdispersed (α = 1.65 against a fitted α of
  ~10⁴, plus a mean offset). This is the same "approach the dispersion from above" effect as for
  StudentT. Gamma's default is already the raw start, so nothing changes.

## 4. Cross-cutting finding
With early stopping and per-parameter stopping, where the dispersion parameter **starts** matters
beyond its role as an initial value:
- Starting overdispersed and tightening helps on Gaussian-like and Gamma data.
- On heavy-tailed data it hurts, because the inflated phase pushes df up.

## 5. Next investigations (ranked)
1. Make "approach from above" adaptive. Start the scale inflated only when a quick marginal
   check (e.g. marginal kurtosis) says the data is not heavy-tailed. This would be decided per
   dataset on train data, so it is not tuned on test.
2. Gamma on naval-like data: does a deliberately overdispersed start (α₀ = 1, mean at ȳ ×
   1.5?) beat the raw start, and is it translation-safe (Gamma has no translation; it is
   scale-equivariant)? Test the raw start's scale dependence by rescaling y.
3. A Weibull row-level tail guard in the *validation* metric (e.g. Huberised NLL for early
   stopping) to make stopping robust to the e^z tail. It would change only the stopping
   decision, never the scoring.
