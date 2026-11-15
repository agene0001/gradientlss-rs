# Cycle 8 report — PES for Beta/Weibull, StudentT's gaussian_sim cost, location starts

Date: 2026-11-10. Rows: `experiments_v2.csv` tags `c8*`; detail in `research_log.md`
"Cycle 8". Proposal 011 (+ Addenda L, T4, P3).

## 1. Per-parameter early stopping for Beta and Weibull (landed)
The default recipe (W20, anneal 0.5, hold) was applied to parameter 1 and compared with none.

| family | dataset | ΔNLL | note |
|---|---|---|---|
| Beta MP (log φ) | bike_share | −0.019 (5/5) | cov90 .876 → .894 |
| Beta MP (log φ) | synth_beta, synth_beta_hi | 0.000 | never triggers |
| Weibull (log k) | naval | −0.81 (5/5) | cov90 .88 → .97 |
| Weibull (log k) | kin8nm | −0.035 (5/5) | |
| Weibull (log k) | synth_weibull | 0.000 | |
| Weibull (log k) | power_plant | +0.043 (5 seeds) → **−0.013 (10 seeds, 9/10)** | one test-only outlier, validation better on 10/10 |

Weibull was added to the default watched set on the strength of the 10-seed extension.
That extension was decided **after** seeing the 5-seed result, and is disclosed here as such.

## 2. What causes StudentT's +0.008 on gaussian_sim
Neither df nor the start scale causes it.
- **Start df ν₀ = 10:** closes only 0.003 of the gap, and hurts the heavy-tailed sets
  (+0.007 / +0.012).
- **Doubling df's learning rate:** moves df from 11 to 30 with no NLL change (+0.0002).

What does explain it is a translation probe. gaussian_sim's target sits near 10, so with the
raw start the location travels ~9.5 units first. That acts as an accidental warm-up: the
scale inflates and df is pushed up.
- **Target recentred at 0.5:** the raw start does no better than the median start (+0.0003).
- **Shifted the other way (near 30):** the raw start's advantage grows (−0.009).
- **The median start is exactly translation-invariant.**

The cost is the raw start's luck, not a defect of the fix, so the median start stays.

## 3. Location starts without `initialize`

| head | location start vs raw start | decision |
|---|---|---|
| NB mean/dispersion | bike_cnt −0.18, bike_casual −0.08, synth µ 100–4000 −1.45, small-µ −0.006/−0.001 | **default** (ties the fitted start, ~2× faster) |
| Gamma mean/shape | kin8nm +0.019, naval +0.046 | keep raw (`with_location_start(true)` opt-in) |
| Weibull | naval +0.73, power_plant +0.04, kin8nm +0.02 | keep raw (opt-in) |

Harness finding: since cycle 4 the bench's Gamma (mean/shape) and NB (mean/dispersion)
families used `initialize: true`. Users of `default()` / `new(.., false)` started raw:
- **NB:** much worse than reported, now fixed by the location start.
- **Gamma:** slightly better than reported (fitted starts were +0.155 worse on naval).

The bench now builds both families the way users do. With `BENCH_INIT=1` it reproduces the
old rows bit-for-bit.

## 4. Next investigations (ranked)
1. Weibull's thin upper tail produces rare test-only NLL blow-ups (power_plant seed 0, naval
   seed 3 in cycle 7). Look at a shape cap, or a scoring guard on these targets.
2. The "far start = warm-up" effect on Gaussian-like data. Could a deliberate warm-up
   (e.g. a few location-only rounds before the scale/df trees start) give its benefit
   without the translation dependence?
3. Gamma's raw start beats its fitted MLE start on naval by 0.155. Find out why (the same
   warm-up effect?).
