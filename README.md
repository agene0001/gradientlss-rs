# GradientLSS-rs

Distributional Gradient Boosting for Location, Scale, and Shape (LSS) in Rust.

A Rust implementation of probabilistic gradient boosting, supporting XGBoost and LightGBM backends.

## Features

- **Unified Distribution Interface**: Common `Distribution` trait for all probability distributions
- **Multiple Backends**: Support for XGBoost and LightGBM through feature flags
- **Response Functions**: Identity, exp, softplus, sigmoid, relu, and more
- **Loss Functions**: Negative log-likelihood (NLL) and CRPS
- **Gradient Stabilization**: MAD and L2 stabilization methods
- **Multivariate Distributions**: Support for multivariate normal (MVN) and other multivariate distributions

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
gradientlss = "0.1"

# Enable backends as needed:
# gradientlss = { version = "0.1", features = ["xgboost"] }
# gradientlss = { version = "0.1", features = ["lightgbm"] }
# gradientlss = { version = "0.1", features = ["full"] }  # Both backends
```

### System Requirements for Backends

Both XGBoost and LightGBM backends require native libraries to be compiled:

1. **CMake** (required for both):
   ```bash
   # macOS
   brew install cmake
   
   # Ubuntu/Debian
   sudo apt-get install cmake
   
   # Windows
   # Download from https://cmake.org/download/
   ```

2. **Clang/LLVM** (for bindgen):
   ```bash
   # macOS - usually pre-installed with Xcode
   xcode-select --install
   
   # Ubuntu/Debian
   sudo apt-get install llvm-dev libclang-dev clang
   ```

## Breaking Changes

If you're updating from an earlier build, the following changes require action
or change numerical results:

**At a glance (2026-11-15).** The sections below have the measurements and
the opt-outs.

| change | who it affects | what to do |
|---|---|---|
| `TrainConfig` has a new field `param_freeze_rounds` | code that writes the struct out in full | add `param_freeze_rounds: None`, or end the literal with `..TrainConfig::default()` |
| `HyperOptConfig` has new fields `trial_dist_fn`, `sampler`, `initial_trials` | the same | end the literal with `..Default::default()` |
| `BackendDataset::get_features` and `BackendParams::set_n_dist_params` are now trait methods | custom backends | implement them |
| The default training recipe changed for Gaussian, StudentT, Gamma, Poisson, NegativeBinomial, Beta, Weibull, Gumbel, Cauchy, Logistic, Laplace and LogNormal | everyone who trains these | retrain; regenerate any pinned numbers. Each section names the call that restores the old behaviour |
| Per-parameter early stopping is on when a validation set is passed (Gaussian, StudentT, Gamma, NegativeBinomial, Beta, Weibull) | the same | expect 2–4× more rounds on real data; `.without_param_early_stopping()` turns it off |
| `dist_select` rankings and fitted start values changed | users of `dist_select` or `initialize: true` | none; rankings on targets far from order 1 were wrong before |
| Saved models | nobody | files from earlier builds load and keep the head, link and curvature they were trained with |

For the three families used most for counts and scores, the default changes
and what they measured against the previous defaults (held-out NLL per row, 5
seeds; negative is better):

| family | what changed by default | measured |
|---|---|---|
| Gaussian | Damped curvature; the scale stops on its own validation signal | curvature −0.05 to −0.95 on four real datasets, never worse by more than 0.001; scale stopping −0.10 to −0.37 on the same four, 0.000 on simulations |
| Poisson | log link instead of ReLU; the rate starts at ȳ | Bike Sharing −2.68 and −1.23; simulations −0.003 to −0.054 |
| NegativeBinomial | mean/dispersion head; the mean starts at ln ȳ; Damped curvature and its own stopping for log θ; the mean is no longer capped at 403 | start: Bike Sharing −0.18 and −0.08, counts with a mean of 100–4000 −1.45; curvature −0.001 to −0.028; stopping −0.03 and −0.06 |

### Ready-made search spaces for `hyper_opt` (new, opt-in)

```rust
use gradientlss::hyper_opt::{hyper_opt_with_config, leaf_wise_search_space};

// min_child_weight, lambda and leaf-wise growth with max_leaves 4–64 (XGBoost)
let mut hp = leaf_wise_search_space();
hp.insert("eta".into(), serde_json::json!({"choices": [0.05]})); // what you hold fixed
let result = hyper_opt_with_config(&mut model, &x_train, &y_train, &hp, config.clone())?;
// then validation_guard, as in the section further down
```

`leaf_wise_search_space()` is the search space that passed the inclusion test
at 20 trials on 48 (family, dataset) pairs: with `validation_guard` it was
worse than the untuned defaults on none and better than searching
`min_child_weight` and `lambda` alone on 46. By family (median test NLL
against the untuned defaults, with the two-axis search in brackets, and the
pairs on which it beat that search): Gaussian −0.08 (−0.02), 19 of 20;
Poisson −0.14 (−0.04), 6 of 6; NegativeBinomial −0.06 (−0.01), 6 of 7; Gamma
−0.08 (−0.03), 15 of 15.

`max_bin_search_space()` adds `max_bin` (16–1024) to the same two axes
instead. Give it 50 trials or more: at 20 it failed the test. At 50 it was
better than the two-axis search on 40 of 48 pairs and worse than the untuned
defaults on one, which `validation_guard` did not catch (Gamma on a 768-row
dataset, +0.029).

Both are for the XGBoost backend and neither sets `eta` or the round budget.
The two were not tested together, and the leaf-wise space was not tested with
`subsample` / `colsample_bytree`. Measurements: "Research cycles 25–26" below.

### Gumbel: Damped curvature and a wide data start by default (results change)

`Gumbel::new` now uses `HessianMode::Damped`, the
`LocationScaleStart::LocationScale` start, and a start scale 100 times the
moment-matched one (`with_start_scale_factor`, new;
`GUMBEL_START_SCALE_FACTOR`). Before, it used the observed curvature and a
raw 0.5 start, and did not fit a target far from zero: on data simulated from
a Gumbel with location near 120 it ended 3.5 nats above the true model, at a
scale of 115. Now it ends 0.09 above.

Test NLL and CRPS, 5 seeds, old default → new default:

| dataset | Gaussian | before: NLL | now: NLL | now: NLL of the median seed | CRPS, before → now |
|---|---|---|---|---|---|
| simulated Gumbel (true model 2.60) | — | 6.10 | 2.68 | 2.70 | 51.0 → 2.22 |
| kin8nm | −0.66 | −0.16 | −0.37 | −0.36 | 0.103 → 0.088 |
| naval | −6.05 | −4.59 | −5.42 | −5.43 | 0.0008 → 0.0007 |
| power_plant | 2.68 | 7.43 | **2622** | 3.12 | 190.5 → 2.51 |
| protein | 2.86 | 73 681 | 2.93 | 2.83 | 2.64 → 2.56 |
| airfoil (unseen) | 1.97 | 6.13 | 2.51 | 2.56 | 52.3 → 1.70 |
| gas-turbine CO (unseen) | 1.68 | 1.120 | 1.122 | 1.133 | 0.473 → 0.468 |
| gas-turbine NOx (unseen) | 2.89 | 7.25 | 3.17 | 3.17 | 3.35 → 3.19 |
| Parkinson's (unseen) | 2.18 | 14.77 | 2.91 | 2.90 | 25.9 → 3.08 |
| superconductivity (unseen) | 3.37 | 49 992 | 4.29 | 3.63 | 12.6 → 7.33 |

- **Why the start is wide.** The Gumbel density charges e^{−z} for a label
  below the location. A start scale that is too small for some rows begins
  with an NLL in the tens of thousands; one that is too large costs ln 100
  and makes the first rounds least-squares fitting of the location. The
  matched scale (factor 1) was worse than factor 100 on every set where the
  start mattered (naval −5.13 against −5.47 validation NLL).
- **What it cannot fix: single test rows.** A test label 10 to 17 scales
  below its predicted location costs e^{10} to e^{17} for that one row. That
  is what the 2622 on power_plant is: two of five seeds contain such a row
  (8.4 and 13 091); the other three are 3.07–3.13 and the CRPS is unaffected.
  All 27 such cells in the experiment have their worst row in the lower tail.
  The old default "avoided" it on power_plant only by never fitting (scale
  430), and had six such cells of its own elsewhere (naval, protein, NOx,
  superconductivity).
  If your target can have low outliers, Gumbel's mean NLL is fragile whatever
  the settings; use it for right-skewed data.
- Against the old default the new one is better or within 0.005 on 10 of 11
  datasets on mean test NLL (power_plant is the exception, for the reason
  above), on 11 of 11 without each model's worst 0.5 % of rows, and its CRPS
  is equal or better on 11 of 11. The unseen datasets had been used once
  before for Gumbel (with a different recipe).
- **To get the old behaviour:**
  `Gumbel::new(..).with_hessian_mode(HessianMode::Observed).with_start(LocationScaleStart::Raw)`.
  A model saved before this change loads with the settings it was trained
  with.

### NegativeBinomial: `LossFn::Crps` now trains in mean/dispersion coordinates (results change)

A NegativeBinomial built with `LossFn::Crps` now boosts in (log mean, log
dispersion), as the NLL-trained one has since 2026-11-10. Before, it fell
back to (total count, probs), with a warning from `new_mean_dispersion`.
Predictions are still reported as (total count, probs). This applies to
`NegativeBinomial::new`, `new_mean_dispersion` and
`with_parameterization(NbParameterization::MeanDispersion)`.

Test CRPS, 5 seeds (lower is better):

| dataset (mean count) | NLL-trained | CRPS-trained, before | CRPS-trained, now | now against NLL-trained |
|---|---|---|---|---|
| simulated NB | 4.017 | 4.136 | 4.040 | +0.6 % ± 0.1 |
| abalone (10) | 1.207 | 1.612 | 1.208 | +0.1 % ± 0.3 |
| bike_casual (36) | 6.07 | 6.29 | 6.00 | −1.2 % ± 0.9 |
| bike_cnt (190) | 17.22 | 19.19 | 16.77 | −2.7 % ± 1.1 |
| Seoul bike (700) | 69.79 | 70.62 | 65.66 | −5.9 % ± 1.0 |

- Better than the old coordinates on all five datasets and in 25 of 25 fits
  (2 % to 25 %), and no fit ends at the round cap (abalone: 5 of 5 did).
- **It changes the advice for NegativeBinomial.** On the three real datasets
  with larger counts, training on the CRPS now gives the better test CRPS (by
  1 % to 6 %). It pays for it elsewhere: the test NLL is worse (+0.06, +0.13,
  +0.63 nats) and the 90 % interval covers fewer labels (Seoul bike 79 %
  against 93 %; bike_cnt 85 % against 93 %). Train on the CRPS only if the
  CRPS is all you score; for calibrated intervals train on the NLL.
- Poisson is unchanged: its CRPS training stays 5–7 % behind NLL training.
- **To get the old behaviour:**
  `NegativeBinomial::new(stab, r_fn, p_fn, LossFn::Crps, init).with_parameterization(NbParameterization::TotalCountProbs)`.
  A saved model carries its coordinates and loads with them.
- The NLL path is not touched.

### Cauchy, Logistic, Laplace, LogNormal: Damped curvature and data starts by default (results change)

`Cauchy::new` and `LogNormal::new` now use `HessianMode::Damped` with a
`LocationScaleStart::Location` start; `Logistic::new` and `Laplace::new` use
`Damped` with `LocationScaleStart::LocationScale`. Before, these families
used the observed curvature and a raw 0.5 start, and stalled on targets far
from zero: far from the target the location Hessian underflows (Logistic) or
is negative (Cauchy, for half of all rows).

| family | power_plant (y ≈ 454), before → now | airfoil (unseen), before → now |
|---|---|---|
| Cauchy | 12.88 → 2.82 | 10.29 → 2.54 |
| Logistic | 7.61 → 2.69 | 6.32 → 2.19 |
| Laplace | 7.65 → 2.73 | 6.27 → 2.28 |
| LogNormal | 2.84 → 2.75 | 2.87 → 2.16 |

(Test NLL, 5 seeds; Gaussian scores 2.68 on power_plant.)

- **Evidence:** not worse than the old default on any of the 18 development
  pairs, and within +0.005 on 20 of 20 pairs from five datasets no research
  cycle had used. The new defaults reproduce the tested recipe bit for bit
  (190 of 190 cells).
- **Opt out:** `.with_hessian_mode(HessianMode::Observed).with_start(LocationScaleStart::Raw)`.
- **Saved models:** a model saved before this change loads with the settings
  it was trained with (Observed, Raw). Predictions of saved models do not change.
- **Gumbel is unchanged.** The same recipe passed on only 4 of 5 unseen
  datasets and single seeds still diverge; `with_hessian_mode` and
  `with_start` are available on it as options.
- **Laplace:** keep `Damped` together with `LocationScale`. With a raw or
  location-only start it was worse on naval (+0.75, +1.14).
- With `initialize: true` the start values come from the optimiser as before
  and only the curvature changes. That combination was not benchmarked.

### `LossFn::Crps` for Poisson and NegativeBinomial now works for large counts (results change)

The CRPS loss, its gradient and its curvature term were summed over at most
100 terms (Poisson) and 150–220 terms (NegativeBinomial). A row with a larger
count got a loss and a gradient from a CDF that had not reached 1. A Poisson
model with mean counts above about 100 did not train at all, without a
warning.

Test CRPS, 5 seeds (lower is better):

| family | dataset (mean count) | NLL-trained | CRPS-trained, before | CRPS-trained, now |
|---|---|---|---|---|
| Poisson | bike_casual (36) | 6.44 | 9.50 | 6.77 |
| Poisson | bike_cnt (190) | 18.2 | 134.8 (no training) | 19.4 |
| Poisson | Seoul bike (700) | 76.5 | 505.6 (no training) | 80.0 |
| NegativeBinomial | bike_casual | 6.07 | 6.56 | 6.29 |
| NegativeBinomial | bike_cnt | 17.2 | 37.2 | 19.2 |
| NegativeBinomial | Seoul bike | 69.8 | 407.7 | 70.6 |

- **What changed.** For a row whose sums do not fit under the old limits, the
  sums now cover the whole distribution, and the curvature term is the
  expected CRPS curvature, 2 Σₖ (∂F(k)/∂θ)², instead of the NGBoost metric
  E[g²]. The metric grows faster with the count than the curvature does and
  shrank every step: with the sums fixed but the metric kept, Poisson on
  bike counts was still at 3× the NLL-trained CRPS after 2000 rounds.
- **What did not change.** A row that fits under the old limits runs the old
  code and gives the same numbers (abalone, counts 1–29: identical in 10 of
  10 cells). Datasets that mix small and large counts change.
- **Limit.** One row's sum uses at most 200 000 terms (Poisson means up to
  about 10⁸; NegativeBinomial means up to a few 10⁴, depending on the
  dispersion). Beyond that a warning is printed once and the sum is cut.
- **Training on the NLL is still the better choice for these two families.**
  It gave an equal or better test CRPS on all 10 (family, dataset) pairs: CRPS
  training is 1 % to 12 % behind on the large-count datasets, mostly still
  improving at 2000 rounds. NegativeBinomial under `LossFn::Crps` also still
  falls back to the (total count, probs) coordinates and is 34 % behind on
  abalone. (Fixed on 2026-11-14: see the NegativeBinomial section above this
  one.)

### `validation_guard`: check a tuned configuration against an untuned one (new, opt-in)

```rust
use gradientlss::hyper_opt::validation_guard;

let result = hyper_opt_with_config(&mut model, &x_train, &y_train, &hp_dict, config.clone())?;
let checked = validation_guard(
    &model, &x_train, &y_train, &x_valid, &y_valid,
    &result.best_params, // tuned
    &untuned_params,     // what you would use without tuning
    &config,
)?;
// checked.params is the configuration to use; checked.kept_tuned says which one won
```

It fits both configurations with early stopping on the validation rows and
returns the one with the lower validation loss. With searches over
`min_child_weight` and `lambda` this removed every case where tuning was
worse than the untuned defaults (4 of 24 pairs without it, 0 of 24 with it).
On twelve further pairs (cycles 25–26) it left one of two (2 of 12 without
it, 1 of 12 with it, by +0.008). It is a check, not a guarantee: see "Tuning
with `hyper_opt`" below for the two cases it missed with a wider search
space.

### `dist_select` ranks each family at its maximum-likelihood fit (results may change)

`dist_select` and `dist_select_with_params` now call the new
`Distribution::marginal_fit` (default: `calculate_start_values`). StudentT
overrides it with the free fit. Before, StudentT was ranked at its training
start values, whose df is capped at 10, and scored about 0.011 NLL per row
worse than Gaussian on Gaussian labels that it fits equally well.

This removes that artefact and nothing more. On the 45 benchmark cells the
picks did not change (best-on-test family picked in 22 of 45 before and
after): the ranking still cannot see the features. Use `dist_select` to
shortlist, then compare trained models on validation (see below).

### Two small fixes (results may change)

- **SplineFlow start values.** The start-value optimiser stopped at the first
  iterate with a total NLL ≤ 0, which happens whenever the fitted density
  exceeds 1. It now runs its iteration budget (same fix as the other families
  got in July). On 400 labels packed into [0.45, 0.55] the start fit goes from
  −0.61 to −2.25 NLL per row.
- **LogNormal, rows with y ≤ 0.** Such a row now gets a zero gradient and a
  token Hessian (1e-6) on every path. Before, datasets of 4096 rows or more
  gave it a Hessian of 0 and smaller ones 1e-6.

### Newton curvature: `HessianMode::Damped` is now the default (results change)

Gaussian and StudentT now hand the booster `max(observed, Fisher)` curvature
per row and parameter instead of the observed second derivative (StudentT) or
the constant Fisher value (Gaussian). Observed curvature is negative on the t
location for outlier rows and ≈ 0 on the scale for well-fit rows, so leaf steps
`−Σg/(Σh+λ)` could blow up or flip sign; pure Fisher curvature is constant and
takes huge steps far from the optimum. The damped form floors the first with
the second. In 700 paired fits across 7 datasets it never lost to the previous
defaults by more than 0.001 nats, improved held-out NLL by 0.05–0.95 nats on
kin8nm, naval, power_plant and protein, and never diverged (details and the
reproducible harness: `research/gradientlss/reports/002-cycle1-report.md`,
`examples/research_bench.rs`).

- **Effect:** trained Gaussian/StudentT models differ (generally better);
  saved models load unchanged — the mode only affects training.
- **Opt out:** `Gaussian::new(..).with_hessian_mode(HessianMode::Observed)`
  (autograd/XGBoostLSS parity) or `HessianMode::Fisher`.

### Beta now boosts mean/precision; Beta and Weibull use Damped curvature (results change)

- **Beta:** `Beta::new(.., ResponseFn::Exp, LossFn::Nll, ..)` and
  `Beta::default()` now boost (logit µ, log φ), with µ the mean, φ = a + b the
  precision and a = µφ, b = (1 − µ)φ. Predicted parameters are still reported
  as `(concentration1, concentration0)`.
  - **Why:** in the old (log a, log b) coordinates the Fisher information is
    nearly collinear once the Beta is concentrated: correlation −0.95 at φ = 20
    and −0.99 at φ = 100, the same problem Gamma had.
  - **Results:** mean/precision won 5/5 seeds on every dataset (held-out NLL):
    synthetic φ 55–1100 −0.27 nats, φ 3–55 −0.10, and the Bike Sharing casual
    share −0.04 against the old head with Damped curvature.
  - **Old curvature on real data:** with its old observed curvature, the
    previous head diverged on the casual share (NLL 2545).
  - **Opt out:** `.with_parameterization(BetaParameterization::Concentrations)`.
    That head now defaults to Damped curvature (neutral to better).
- **Weibull:** now defaults to Damped curvature (`with_hessian_mode` is new).
  Held-out NLL: power_plant −0.15, kin8nm −0.08, synthetic −0.05, naval −0.05
  (one naval seed was worse on test only).
  - **No new head:** its (log λ, log k) coordinates are only mildly correlated
    (−0.31 everywhere), so no reparameterisation was added.

Models saved before these fields existed load with the old head and the
observed curvature.

### New opt-in knobs from research cycle 9 (no default changes)

- **`Weibull::with_max_concentration(k)`** caps the shape. A Weibull's upper
  tail is doubly exponential. A test point z predicted standard deviations
  above the scale costs about e^z nats (a Gaussian charges z²/2). With a large
  fitted shape, one outlier can dominate the test NLL: power_plant had a row
  costing 1314 nats, and a naval row at the target's upper bound cost 5711. A
  cap only widens every prediction, and no fixed cap was neutral: k ≤ 200
  rescued power_plant (−0.04) but cost naval +1.4. If your positive target
  has upper outliers, prefer Gamma, or tune the cap on data you trust.
- **`GradientLSS::with_warmup_rounds(n)`** trains only the location for the
  first `n` rounds (`Distribution::location_param_index` says which column).
  - **Where it helps:** on StudentT, 50 rounds helped kin8nm, protein and
    naval slightly (−0.016 to −0.027).
  - **Where it hurts:** it cost the Gaussian-noise simulation +0.009, so it is
    not a default.

### Poisson: log link and a mean start by default (results change)

`Poisson::default()` used the ReLU (identity) link with a raw 0.5 start. Under
ReLU:
- a row whose margin falls to ≤ 0 has a rate of 10⁻⁶ and a gradient of exactly
  0, so it never recovers;
- zero counts carry zero curvature.

The default is now the log link, which XGBoost's `count:poisson`, LightGBM and
GAMLSS all use. The rate starts at ȳ, and curvature is Damped (identical to
observed under the log link).

Measured against the old default (5 seeds, NLL per point):

| dataset | change |
|---|---|
| bike_cnt | −2.68 |
| bike_casual | −1.23 |
| simulated Poisson | −0.054 |
| simulated NB (θ small) | −0.021 |
| simulated NB (near-Poisson) | −0.003 |

`Poisson::new(.., ResponseFn::Relu, ..)` keeps the old link. With it the start
stays at 0.5, because the mean start hurt bike_cnt under ReLU. Saved models keep
their link and load with observed curvature.

### Gamma: the default start is now unit-free (results change)

Without `initialize`, the Gamma mean/shape head used to start at raw [0.5, 0.5],
so the starting mean was 1.65 whatever the target's units. Rescaling the same
data therefore changed the fitted model. kin8nm with labels × 100 was 0.066
nats worse per point, after correcting for the unit change.

It now starts at [ln ȳ + 0.5, 0.5]: mean 1.65·ȳ, shape 1.65. Rescaling y only
shifts this start, so the fit begins in the same place whatever the units.

Measured against the old start (5 seeds, NLL per point, negative = better):

| labels | kin8nm | naval | power_plant | synth_gamma |
|---|---|---|---|---|
| as given | +0.001 | +0.007 ± 0.042 | +0.002 | −0.016 |
| × 0.01 | −0.017 | −0.010 | −0.007 | −0.006 |
| × 100 | −0.056 | −0.009 | −0.011 | −0.020 |

- `Gamma::with_wide_start(false)` restores the old start.
- `with_location_start(true)` starts the mean at ȳ instead, and takes precedence.
- Models saved before this change load with the old start.

### Gaussian fits depend on the label units

XGBoost's `min_child_weight` and L2 `lambda` act on raw Hessian sums. The
Gaussian location's Hessian is 1/σ², so it scales with 1/units². The same data
in different units is regularised differently:
- power_plant with labels × 100 lost 0.2 nats;
- at native units the effective location regularisation already differs by
  ~10⁷ between datasets.

No built-in fix was neutral on every dataset tested (cycle 11). Standardizing
the labels yourself (training-split mean/sd; add ln sd back to the NLL)
helped protein and power_plant but hurt naval. Treat it, or a rescaled
`min_child_weight`, as something to tune on validation.

### Scale-relative tree regularisation (opt-in, cycle 12)

`GradientLSS::with_hessian_normalization(HessianNormalization::Marginal)`
multiplies each parameter's gradients and Hessians by 1 / (its mean Hessian at
the intercept-only fit). `min_child_weight` and `lambda` then count "rows'
worth of information", as in squared-error regression. The unregularised
Newton step is unchanged, and so are trees, predictions and saved models.

It makes fits nearly unit-free: labels × 100 change the NLL by ≤ 0.006,
against 1.3 without it. It is not a default, because the best regularisation
strength still differs by dataset:
- Gaussian kin8nm −0.018, power_plant −0.044, Gamma naval −0.043;
- Gaussian simulation +0.009, Poisson bike_cnt +0.066.

`HessianNormalization::Start` measures the mean Hessian at the start values
instead.

**Recommended when you tune `min_child_weight` and `lambda`.** Turn
normalisation on for the search and for the final model, search both over the
full log range [1e-3, 1e3], and give the search at least 20 trials:

```rust
let mut model = GradientLSS::<XGBoostBackend>::new(dist)
    .with_hessian_normalization(HessianNormalization::Marginal);
```

It is a recommendation for that two-parameter search, not a default: at fixed
`min_child_weight` = `lambda` = 1 it fails the better-or-neutral rule (above).
It has not been measured together with a search over tree depth and
subsampling, which gains far more (cycles 21–22 below).

**The evidence (cycle 13).** `hyper_opt` tuned `min_child_weight` and
`lambda` over the same log range [1e-3, 1e3] with 20 trials, once with
normalisation and once without (5 seeds). The search's fold models use the
model's normalisation setting.
- **With normalisation better:** Gaussian kin8nm −0.035, power_plant −0.022,
  protein −0.024, naval −0.049; Gamma kin8nm −0.025; NB bike_cnt −0.008.
- **With normalisation worse, within noise:** Poisson bike_cnt +0.017 and
  Gamma naval +0.022. On Poisson bike_cnt, tuning these two parameters was
  worse than the untuned defaults on either axis.
- **Give it at least ~20 trials over the wide range.** After 5 trials the
  normalised search lost > 1 nat on naval: 1000 "rows' worth" of
  regularisation is crushing there. Narrowing the range to [1e-3, 10] fixed
  that, but it lost most of the gains at 20 trials (kin8nm +0.035, protein
  +0.046). The best strength in rows' worth still spans about four decades
  across datasets: about 0.05 on naval, about 700 on kin8nm.

### Research cycles 14–22: tuning guidance and opt-in options

Details: `research/gradientlss/reports/015-cycles14-21-report.md`. All numbers
are test NLL, 5 seeds, XGBoost unless a line says otherwise. The changes that
came out of these cycles (four family defaults, CRPS for large counts,
`validation_guard`, `dist_select`) have their own sections at the top of this
list.

**Choose the family by validation NLL, not by `dist_select` alone.**
`dist_select` ranks families by an unconditional fit to the labels. On 9
datasets × 5 seeds it picked the family with the best test NLL in 22 of 45
cells; comparing the validation NLL of fitted models picked it in 42 of 45
(mean regret 0.055 against 0.0005). It preferred Gaussian or Gamma where
StudentT was better by 0.10–0.21 (protein, power_plant, kin8nm). Ranking
StudentT at its free fit instead of its capped start (now done) did not change
a single pick. Use `dist_select` to shortlist, then fit the shortlist and
compare on validation.

**Tuning with `hyper_opt`: what 20 trials bought.** TPE, 3 folds, 5 seeds;
numbers are test NLL of the tuned model minus the untuned defaults (negative
= tuning helped), on 10 development and 14 unseen (family, dataset) pairs.

| search space | development: mean | worse than untuned | unseen: mean | worse than untuned |
|---|---|---|---|---|
| `min_child_weight`, `lambda` | −0.008 | 2/10 | −0.017 | 2/14 |
| the same, kept only if better on validation | −0.019 | 0/10 | −0.023 | 0/14 |
| plus `max_depth` 2–8, `subsample`, `colsample_bytree` 0.5–1 | −0.089 | 0/10 | −0.131 | 2/14 |

("Worse than untuned" counts pairs worse by more than 0.005.)

- **The CV score overstates the gain.** With two axes the search believed it
  gained 0.047 (development) and 0.082 (unseen) over the defaults; the
  refitted model kept 0.007 and 0.022 on test. A configuration's CV gain did
  not predict its test gain.
- **Check a two-axis result on validation** (`validation_guard` does this).
  Fit the tuned and the untuned configuration the way you fit the final model
  and keep the one with the lower validation NLL. That removed every loss of
  the two-axis search on these pairs (0/10, 0/14); on twelve fresh pairs it
  left one of two (cycles 25–26 below).
  It can cost gain when the family fits badly (Poisson on Seoul bike counts:
  −0.140 → −0.071).
- **The three extra axes carry most of the gain, but not safely.** The
  five-axis search beat the two-axis search on 9/10 and 12/14 pairs (Gaussian
  naval +0.053 → −0.127, Poisson bike_cnt +0.030 → −0.124, Gamma Parkinson's
  −0.044 → −0.334). On two unseen Gaussian pairs it was worse than the untuned
  defaults (gas-turbine CO +0.190, superconductivity +0.025), and the
  validation check did not catch either. The CO target is heavy-tailed and
  Gaussian is the wrong family for it (untuned: Gaussian 1.68, StudentT 1.00);
  a few test rows carry the NLL. Settle the family first (see above), then
  tune.
- **Not established:** a rule that makes the five-axis search safe. Which of
  the three added axes matters was measured afterwards (cycles 25–26 below).
- More folds did not fix the two-axis losses (5 folds: Gaussian naval +0.012,
  Poisson bike_cnt +0.053).

**New `hyper_opt` options (opt-in).** `HyperOptConfig::sampler`
(`Sampler::Tpe` default, `Random`, `TpeAfterRandom { n_startup }`) and
`HyperOptConfig::initial_trials` (configurations scored before the sampler
draws). Neither is recommended over the default: random search was never
better than TPE on the best CV score at 10 or 20 trials (0 of 10 datasets),
and scoring the untuned defaults first did not help reliably. Code that builds
`HyperOptConfig` without `..Default::default()` must add the two fields.

**Cauchy, Logistic, Laplace, LogNormal, Gumbel.** `with_hessian_mode` and
`with_start(LocationScaleStart::{Raw, Location, LocationScale})` were added
to all five. For the first four the tested recipe is now the default (see the
section above); Gumbel's default followed on 2026-11-14 (its own section,
near the top of this list).

**CRPS training (Poisson, NegativeBinomial).** `LossFn::Crps` gave a better test CRPS than NLL
training on 0 of 10 (family, dataset) pairs, before and after the large-count
fix described above. Train on the NLL even if you score by CRPS. (For
NegativeBinomial this changed on 2026-11-14 with the mean/dispersion
coordinates: see the section near the top of this list.)

**The defaults hold on LightGBM and on unseen data.** Today's defaults
against the pre-research recipe: 0 of 21 pairs worse by more than 0.005 on
either backend, and 0 of 19 on seven datasets no cycle had used.

### Research cycles 25–26: search space and adaptive λ

Details: `research/gradientlss/reports/016-cycles25-26-report.md`. 5 seeds,
XGBoost. The two default changes that came out of these cycles (Gumbel,
NegativeBinomial under the CRPS loss) have their own sections at the top of
this list.

**Which part of the wider search space pays.** The five-axis search of
cycles 21–22 split in two, same 20 trials, folds and seeds; test NLL of the
tuned model minus the untuned defaults, 10 development and 14 unseen (family,
dataset) pairs. "Checked" keeps the tuned model only if it beats the untuned
one on validation (`validation_guard`).

| search space | development: mean | worse than untuned | checked: mean (worse) | unseen: mean | worse than untuned | checked: mean (worse) |
|---|---|---|---|---|---|---|
| `min_child_weight`, `lambda` | −0.008 | 2/10 | −0.019 (0/10) | −0.017 | 2/14 | −0.023 (0/14) |
| plus `max_depth` 2–8 | −0.055 | 1/10 | −0.057 (1/10) | −0.081 | 2/14 | −0.076 (1/14) |
| plus `subsample`, `colsample_bytree` 0.5–1 | −0.054 | 0/10 | −0.052 (0/10) | −0.055 | 2/14 | −0.045 (0/14) |
| all five | −0.089 | 0/10 | −0.089 (0/10) | −0.131 | 2/14 | −0.121 (2/14) |
| no search, `max_depth` 6 instead of 4 | −0.002 | 5/10 | | −0.032 | 4/14 | |

- **Either half gets about 60 % of the five-axis gain** on the development
  pairs (depth 61 %, subsampling 61 %); on the unseen pairs depth 61 %,
  subsampling 42 %.
- **The losses are the depth axis.** Both unseen losses of the five-axis
  search reappear with depth alone (Gaussian on gas-turbine CO +0.168,
  superconductivity +0.067) and not with subsampling alone (+0.011, −0.024);
  Gaussian protein lost with depth alone as well (+0.029). The validation
  check does not catch the CO case. Depth also gives the largest single
  gains (Parkinson's −0.44 and −0.25, Gaussian kin8nm −0.16).
- **Adding the two subsampling axes never cost anything against the two-axis
  search on these 24 pairs:** better or within 0.002 on all of them, and with
  the validation check worse than the untuned defaults on none. This was seen
  after the fact, so it was then tested on fresh data (next paragraph).
- **A fixed deeper tree is not the gain.** `max_depth` 6 without a search was
  worse than depth 4 on 9 of 24 pairs (superconductivity +0.58). The searches
  picked depth 2 on the simulations and 8 on Gaussian kin8nm.

**The subsampling axes on fresh data.** Twelve (family, dataset) pairs from
seven UCI datasets no cycle had used (concrete, white wine, appliance energy,
SGEMM run times, grid stability, metro traffic, news shares), same protocol,
predictions written down first.

| search space | mean (median) | worse than untuned | checked: mean (median) | checked: worse than untuned |
|---|---|---|---|---|
| `min_child_weight`, `lambda` | −0.134 (−0.026) | 2/12 | −0.124 (−0.021) | 1/12 |
| plus `subsample`, `colsample_bytree` 0.5–1 | −0.150 (−0.043) | 0/12 | −0.139 (−0.043) | 0/12 |

- **Safe:** with or without the validation check, the four-axis search was
  worse than the untuned defaults on 0 of 12 pairs. Over all three sets of
  data that is 0 of 36 with the check. The two-axis search lost on 2 of 12
  (Gaussian on concrete +0.069, on appliance energy +0.034), and once even
  with the check (+0.008).
- **Usually, not always, better than two axes:** better on 10 of 12 fresh
  pairs (33 of 36 overall). It lost on two count pairs: NegativeBinomial on
  news shares (+0.011 ± 0.006) and Poisson on metro traffic (+0.165 ± 0.255,
  where Poisson is the wrong family: untuned NLL 39).
- **Two of my three predictions failed:** I had predicted at most one such
  loss, and a mean gain 1.5 times the two-axis one (it was 1.12 times; the
  Poisson pair is most of both means, and the medians differ by a factor 2).
- **What to do with it:** if you tune `min_child_weight` and `lambda`, adding
  `subsample` and `colsample_bytree` in [0.5, 1] to the same search and
  keeping the result only if it beats the untuned model on validation
  (`validation_guard`) has not yet produced a model worse than the defaults.
  Do not expect it to beat the two-axis search every time. `max_depth` is
  where the larger gains and all the losses were.

**`max_bin` as a search axis: worth adding at 50 trials, not at 20.** Every result above
used the backend's 256 histogram bins per feature. First without a search:
the untuned defaults at other values, test NLL against 256 bins, all 36
(family, dataset) pairs.

| `max_bin` | better by > 0.005 | worse by > 0.005 | within ±0.005 | median difference | largest differences |
|---|---|---|---|---|---|
| 16 | 7 | 25 | 4 | +0.040 | naval +1.4, Poisson on bike counts +3.0; kin8nm −0.09 |
| 64 | 12 | 12 | 12 | −0.001 | naval +0.58; kin8nm −0.06, Poisson on metro traffic −2.1 |
| 1024 | 4 | 16 | 16 | +0.003 | Poisson on Seoul bikes +0.39, power_plant +0.05 |

Then as a third search axis (integers 16–1024, log-uniform) next to
`min_child_weight` and `lambda`, 20 trials, rule and predictions written down
first.

- **The search does use it.** It picked 16–30 bins on kin8nm and 230–970 on
  naval in every seed, in line with the table, and beat the two-axis search
  by more than 0.005 on 17 of 36 pairs (kin8nm −0.09 and −0.06, Gamma on
  Parkinson's −0.10). It was worse by more than 0.005 on 8.
- **It failed the rule on all three counts.** With the validation check it
  was worse than the untuned defaults on 2 of 36 pairs (Gaussian protein
  +0.12, Poisson on Seoul bikes +0.08; the two-axis search: 1 of 36), better
  than the checked two-axis search on 23 of 36 (24 needed), and the median
  difference was −0.003 (−0.005 needed). The mean over pairs is about the
  same with and without the axis (−0.055 against −0.054 unchecked).
- **A third axis takes trials from the other two.** On the SGEMM data no
  feature has more than 16 distinct values, so `max_bin` cannot change the
  model; the three-axis search was still 0.048 ± 0.026 worse there than the
  two-axis one.
- **At 50 trials it passed on the same 36 pairs and missed on new data.**
  With 50 trials for both searches the three-axis one, with the validation
  check, was worse than the untuned defaults on 0 of 36 pairs, better than
  the checked two-axis search on 30 and by a median of −0.009: the rule
  passed. Those pairs had been used for this parameter before, so it was run
  once more on 12 pairs from seven datasets downloaded for the purpose.
  There it was better than the checked two-axis search on 10 of 12 (median
  −0.015) but worse than the untuned defaults on one (Gamma on the 768-row
  Energy Efficiency data, +0.029), so the rule, which allows none, failed.
- **A choice between 64 and 256 bins** as the third axis (20 trials) picked
  64 on kin8nm and 256 on naval in every seed and kept the kin8nm gains,
  but was worse than the untuned defaults on 3 of 36 pairs with the
  validation check. It failed too.
- **More trials did not help the two-axis search.** At 50 trials instead of
  20 it was better on 17 of 36 pairs and worse than the untuned defaults on
  3 instead of 1; on the 12 new pairs better on 6.
- **What to do with it:** with 50 trials or more, add `max_bin` (16–1024,
  log-uniform) to the search and keep the validation check;
  `max_bin_search_space()` is that space. It was better than the checked
  two-axis search on 40 of 48 pairs. This recommendation is an exception to
  the rule the other axes were held to, which allows no loss: on one pair of
  48 it was worse than the untuned defaults and the check did not catch it
  (the Gamma pair above). The maintainer accepted that on 2026-11-15. With 20 trials
  leave it out. Without a search leave it at 256: 1024 was worse on 16 of 36
  pairs and 16 on 25. It was not tested together with the leaf-wise axis
  below.

**Leaf-wise growth with a leaf budget: the one further axis that passed, twice.**
Eight XGBoost parameters no earlier cycle had searched were each added as a
third axis next to `min_child_weight` and `lambda` (20 trials, same 36 pairs,
same rule, predictions written down first). "Checked" means the tuned model is
kept only if it beats the untuned one on validation (`validation_guard`).

| third axis (range) | checked: worse than untuned (0 needed) | checked: better than two axes (24 needed) | median difference (−0.005 needed) | rule |
|---|---|---|---|---|
| `eta` (0.02–0.3, log) | 2 | 20 | −0.000 | fail |
| `gamma` (0.001–100, log) | 2 | 14 | +0.004 | fail |
| `alpha` (0.001–100, log) | 3 | 28 | −0.005 | fail |
| `colsample_bytree` (0.5–1) | 0 | 22 | −0.004 | fail |
| `colsample_bylevel` (0.5–1) | 3 | 22 | −0.003 | fail |
| `colsample_bynode` (0.5–1) | 1 | 28 | −0.008 | fail |
| `max_delta_step` (0.1–10, log) | 2 | 15 | +0.001 | fail |
| `max_leaves` (4–64, log) with `grow_policy = lossguide`, `max_depth = 0` | 0 | 34 | −0.038 | **pass** |

- **Leaf-wise growth passed and then passed again on other data.** The pass
  on the 36 pairs was one of eight tries, so it was rerun on the 12 pairs of
  the `max_bin` confirmation, which no leaf-wise fit had touched: worse than
  the untuned defaults on 0 of 12, better than the checked two-axis search
  on 12 of 12, median difference −0.068. Over all 48 pairs it was worse than
  the untuned defaults on none, with or without the validation check, and
  better than the two-axis search on 46.

  | 48 pairs, 20 trials, checked | mean (median) on the 36 | mean (median) on the 12 | worse than untuned |
  |---|---|---|---|
  | `min_child_weight`, `lambda` | −0.056 (−0.021) | −0.045 (−0.028) | 2/48 |
  | plus `max_leaves`, leaf-wise | −0.208 (−0.057) | −0.232 (−0.130) | 0/48 |

- **It is not the depth axis under another name.** I predicted it would lose
  somewhere, as the `max_depth` search did; it did not. On the 24 pairs where
  both were run it beat the depth search on 20. The two differ in range
  (depth 2–8 is 4 to 256 leaves; this is 4 to 64), in resolution, and in
  where the splits go, and these runs do not say which matters.
- **The best leaf count depends on the dataset.** The searches picked 4–8
  leaves on concrete, news shares, fish toxicity, Energy Efficiency and
  Facebook comments, and 43–61 on SGEMM, wine and gas-turbine CO. A fixed 16 leaves without any search was better than the depth-4
  default on 37 of 48 pairs and worse on 7 (Gaussian on concrete +0.08, on
  fish toxicity +0.08), so it is not a new default.
- **`colsample_bynode` and `alpha` came close.** By-node sampling failed on
  one pair (Gaussian on Parkinson's, +0.009 with the check) and beat the
  two-axis search more often than the other two column-sampling variants
  (28 pairs against 22 and 22); `alpha` beat the
  two-axis search on 28 pairs and failed on three losses the check did not
  catch (Gaussian on concrete +0.068). `gamma` and `max_delta_step` made the
  search worse; the searches pushed `gamma` to its lower bound.
- **`eta` helps only where the round cap binds.** With 2,000 rounds at 0.05
  the untuned default stops at the cap on 11 of 36 pairs; raising the cap to
  6,000 improved 6 of them by up to 0.05 and changed nothing else. Searching
  `eta` recovered those gains and lost elsewhere. A fixed 0.15 was worse on
  22 of 36 pairs. The library has no round cap of its own: you pass
  `num_boost_round`. If `best_iteration` comes back at or next to it, the
  cap ended the fit, not early stopping; raise it (tripling it changed
  nothing on the pairs where it did not bind) and leave `eta` alone.
- **These verdicts are for 20 trials.** `max_bin` changed sides between 20
  and 50; none of the seven failures was rerun with more.
- **What to do with it (XGBoost backend):** add the leaf-wise axis to the
  search and keep the validation check. `leaf_wise_search_space()` returns
  the two axes plus these three entries:

  ```rust
  hp.insert("grow_policy".into(), json!({"choices": ["lossguide"]}));
  hp.insert("max_depth".into(), json!({"choices": [0]}));
  hp.insert("max_leaves".into(), json!({"low": 4, "high": 64, "log": true, "type": "int"}));
  ```

  It was not tested together with the subsampling axes or with `max_bin`.
  The default stays at depth-wise growth: no fixed leaf count has been shown
  to be safe, and a trial of smaller fixed counts was not run. LightGBM is
  the next paragraph.

**The same question on LightGBM: nothing passed, and `lambda` needs care.**
Same 36 pairs, 20 trials, references fitted on LightGBM (untuned, and the
two-axis search over `min_child_weight` and `lambda`).

| third axis on LightGBM (range) | checked: worse than untuned (0 needed) | checked: better than two axes (24 needed) | median difference (−0.005 needed) | rule |
|---|---|---|---|---|
| `num_leaves` (4–64, log) with `max_depth = -1` | 1 | 25 | −0.024 | fail |
| `min_data_in_leaf` (1–100, log) | 2 | 19 | −0.000 | fail |
| `feature_fraction` (0.5–1) | 2 | 22 | −0.003 | fail |
| `bagging_fraction` (0.5–1) | 4 | 24 | −0.007 | fail |

- **`num_leaves` missed by one pair,** Gaussian on power_plant at +0.0058
  against a limit of 0.005. Unchecked it beat the two-axis search on 28 of
  36 pairs. It was not rerun.
- **The two sampling axes are not the safe ones here.** On XGBoost
  subsampling with the check lost to the untuned defaults on 0 of 36 pairs;
  on LightGBM `feature_fraction` lost on 2 and `bagging_fraction` on 4.
- **Searching `lambda` can break a LightGBM fit.** LightGBM's default L2
  penalty is 0 and a log-uniform range cannot contain 0. For Gaussian on the
  naval data any penalty from 0.001 up stopped training within 100 rounds
  (test NLL about +1 against −6.1 untuned), in every trial of every search,
  so cross-validation could not see it. The validation check discarded all
  of those models. Without the check the two-axis search on LightGBM is 0.18
  nats worse than the untuned defaults on average over the 36 pairs, all of
  it from this pair; with it, 0.021 better, against 0.056 on XGBoost. The
  cause inside LightGBM was not traced.
- **Untuned, LightGBM was the better backend on more pairs than not:**
  better than untuned XGBoost by more than 0.005 on 21 of 36, worse on 8
  (median −0.019; Gaussian protein +0.21 and Gamma on concrete +0.20 the
  other way).
- **What to do with it:** on LightGBM always use `validation_guard` when
  `lambda` is searched, and expect less from the two-axis search than on
  XGBoost. To let the search reach LightGBM's default, give `lambda` as a
  choice list that contains 0 (`{"choices": [0, 0.01, 0.1, 1, 10, 100]}`)
  or leave it out of the search. Neither variant has been run through the
  36 pairs, so this is a precaution, not a tested space, and there is no
  ready-made LightGBM space.

**Library settings chosen per dataset: tested, not recommended.** The
curvature mode, the start values and per-parameter early stopping are fixed
defaults. Choosing them per dataset was tested three ways on the 36 pairs
(XGBoost), predictions first.

- **Inside the search:** adding the early-stopping window (0 = off, 5–80)
  and anneal (0–0.9) to the two-axis search, 20 trials, failed the rule on
  all three counts (worse than untuned on 1 of 36 with the check, better
  than two axes on 20, median −0.000). The searches showed no preferred
  window: every value from 5 to 80 was picked between 23 and 43 times in 180
  searches.
- **By validation among untuned fits** (the default always a candidate; "no
  pair worse by more than 0.005 and at least 9 better" to pass):

  | chosen per dataset | better | worse | within ±0.005 |
  |---|---|---|---|
  | curvature: default, Observed or Fisher | 12 | 9 | 15 |
  | start: default, L-BFGS (`initialize: true`) or the other location start | 16 | 1 | 19 |
  | per-parameter early stopping: on or off | 0 | 1 | 35 |
  | any of the six | 20 | 5 | 11 |

  The start choice missed by one pair (Gamma on wine, +0.0052) and was not
  rerun. The curvature choice gained on the datasets the defaults were
  developed on (6 better, 0 worse) and lost on the others (6 better, 9
  worse).
- **Each alternative on its own is worse than the default:** Observed
  curvature and early stopping off on 28 of 36 pairs each (better on 1 and
  0); the L-BFGS start on 13 (better on 11). Fisher curvature on Gaussian was
  non-finite on power_plant and between 2.9 and 10¹¹ nats worse on seven
  other datasets.
- **The family by validation NLL** (fit both, keep the one that is better
  on validation, as advised above; Gaussian or Gamma, Poisson or
  NegativeBinomial; 15 datasets) picked the family with the lower test NLL
  in 65 of 75 cases. It matters for counts (Poisson
  was 2 to 32 nats worse than NegativeBinomial on three of four datasets).
  Between Gaussian and Gamma it did no better than always taking Gamma
  (mean regret 0.026 against 0.024) and was wrong once by 0.29.
- **What to do with it:** leave these settings at their defaults. Choosing
  the family by validation NLL remains the advice; it is a large gain on
  counts and close to a coin flip between two families that both fit.

**Adaptive λ (gradient-regularised Newton boosting): tested, not recommended.**
`GradientLSS::with_gradient_regularization` adds a per-round ridge to every
row's Hessian that grows with the gradient norm (Zozoulenko et al., arXiv
2605.00581, adapted per parameter). At a learning rate of 1 it removed 8 of
11 failed fits, and at 0.3 it beat plain boosting on 8 of 12 (family,
dataset) pairs. Plain boosting at half the learning rate did better still
(better on 8 of 12 at 0.15 against 0.3, on 5 of 12 at 0.5 against 1.0), and
nothing beat a learning rate of 0.05 on more than 2 of 12 pairs. It is a
smaller step by another name. The option is left in for reproducibility and
is not in the prelude.

- A learning rate of 0.3 (XGBoost's default) cost more than 0.02 nats on 9 of
  12 pairs against 0.05 (kin8nm +0.17, protein +0.12, power_plant +0.11),
  with early stopping in both. Keep the learning rate small.

### Research options from cycle 10 (not recommended)

- **`GradientLSS::with_stopping_row_cap(k)`** softens validation rows more
  than `k` nats above the median row, for the stopping decision only (early
  stopping, best iteration, and the per-parameter test). Gradients and scoring
  are unchanged.
  - **Measured: it hurts.** At k = 10: Weibull naval +0.88, power_plant
    +0.10, StudentT power_plant +0.016. Even k = 25 costs Weibull power_plant
    +0.03.
  - **Why:** the extreme validation rows are the evidence that the fit is
    becoming overconfident. Softening them lets the fit keep sharpening.
- **`StudentTStart::Adaptive`** starts the scale at 3·sd(y) when the labels'
  excess kurtosis is below 1. It does not work: the labels' kurtosis mostly
  reflects the signal, not the noise (protein −1.1, the Gaussian simulation
  +3.6). It inflates exactly the targets that a wide start hurts.

### NegativeBinomial: the mean now starts at ln ȳ (results change)

`NegativeBinomial::default()` and `new(.., false)` used to start the
mean/dispersion head at raw 0.5, so µ ≈ 1.65 whatever the counts. The published
NB numbers came from `new_mean_dispersion`, which fits start values. The raw
start was far worse on counts with a large mean. The mean now starts at the
sample mean (log θ keeps its raw start). Held-out NLL vs the raw start:

- Bike Sharing: `cnt` −0.18, `casual` −0.08
- synthetic counts with µ 100–4000: −1.45 nats
- small-mean simulations: −0.006 and −0.001

All of these are 5/5 seeds except the last (3/5). It ties the fitted start and
fits about 2× faster than it. Opt out with `.with_location_start(false)`.

The same fix was measured and **not** adopted for Gamma (mean/shape) and
Weibull. Their location score does not saturate, so the raw start can move,
and starting at the data made them worse (Gamma naval +0.05, Weibull naval
+0.73). Both have `.with_location_start(true)` as an opt-in.

### StudentT: the location now starts at the median (results change)

Without `initialize`, `StudentT` used to start every parameter at a raw 0.5
(ν ≈ 3.65, µ = 0.5, σ ≈ 1.65). On a target far from 0.5 the location never
moved, because the t score ∝ r/(νσ² + r²) vanishes for large residuals. On
power_plant (y ≈ 450) the fit had RMSE 448, 90 % coverage 1.000 and NLL 7.5.
The location now starts at the median of the labels, and df and scale keep
their raw starts. Change in held-out NLL (5 seeds, no per-parameter stopping):

- power_plant −4.75
- naval −0.02, protein 0.00, the two t simulations ≤ 0.001
- kin8nm +0.006, gaussian_sim +0.008 (small but consistent; not a cost of
  the median start: gaussian_sim's target sits near 10. A far-away raw start
  gives Gaussian data an accidental warm-up that inflates the scale and
  pushes df up. With the target recentred at 0.5 the raw start does no better
  (+0.0003), and the median start's result is identical at every shift.)

Starting the scale at the MAD as well, or fitting all start values
(`initialize: true`), also fixes power_plant. Both were worse elsewhere:
protein +0.03 to +0.07, kin8nm +0.02, naval +0.11 to +0.14 with per-parameter
stopping. Starting with a scale that is large relative to the residuals keeps
early location learning in the t score's near-linear regime. Opt out with
`.with_start_rule(StudentTStart::Raw)`, or pick `StudentTStart::MedianMad`.
Saved models are unaffected; the field only changes training.

### StudentT: start df capped at ν₀ ≤ 10, and a residual warm start API

`StudentT` now caps the fitted unconditional start value of ν at 10 by default
(`initialize: true` only; `.without_max_start_df()` restores the raw MLE,
`.with_max_start_df(ν)` sets another cap). On Gaussian-like marginals the
unconditional MLE is ν → ∞, where the df score is zero and the df tree can
never learn conditional heavy tails; a moderate start keeps that gradient
alive. Measured: better held-out NLL on protein (−0.067 nats) and power_plant,
within noise elsewhere. A start exactly at the `ExpDf` floor (ν = 2) is also
prevented (ν₀ ≥ 2.5), since the booster can never leave it.

New: `GradientLSS::train_with_residual_warm_start` — a two-stage fit for
location–scale(–shape) families. A Gaussian stage-1 model supplies the per-row
location; the distribution's start values are fitted on the (validation)
residuals; the model is then trained with those start values and the stage-1
location as a per-row offset. `predict` recomputes the offsets itself and
`save`/`load` persist the stage-1 model (a new, versioned file layout — older
files still load). Measured on StudentT: better than the marginal start on 6 of
7 datasets (0.004–0.13 nats), worse by 0.017 on data whose conditional law is
exactly Gaussian. See `research/gradientlss/reports/003-cycle2-report.md`.

```rust
use gradientlss::model::ResidualWarmStart;
let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(StudentT::default()));
model.train_with_residual_warm_start(&mut train, Some(&mut valid), params, config,
                                     ResidualWarmStart::default())?;
let p = model.predict(&x_test.view(), PredType::Parameters, 0, &[], 0)?; // offsets handled
```

`BackendDataset` gained `get_features()` and `BackendParams` gained
`set_n_dist_params()` (both already existed as inherent methods on the
concrete types); custom backend implementations must add them.

### New knob: per-parameter shrinkage (`with_param_gradient_scales`)

`Gaussian` and `StudentT` accept per-parameter gradient multipliers — an
effective per-parameter learning rate (the step length ν_k of gamboostLSS). The
multiplier scales only the gradient column, so the leaf value `−Σg/(Σh+λ)` of
that parameter is scaled while split gains and `min_child_weight` keep their
Hessian scale. Typical use: slow the scale tree, which over-fits |residual| long
before the location tree does:

```rust
Gaussian::default().with_param_gradient_scales([1.0, 0.25])        // [loc, scale]
StudentT::default().with_param_gradient_scales([1.0, 1.0, 0.25])   // [df, loc, scale]
```

Measured (Damped, η=0.05, 5 seeds, held-out NLL vs the shared learning rate):
scale ×0.25 improves kin8nm −0.08, naval −0.11, power_plant −0.05, protein
−0.02 and 90 % coverage where the model under-covered, but costs +0.002…+0.017
on simulated data whose σ(x) must be learned quickly. Dataset-dependent, hence
not a default; tune it on validation NLL (`{1, 0.5, 0.25}`). Default `None`.

### Gamma and NegativeBinomial now boost in orthogonal (mean, shape) coordinates by default

**Breaking (training behaviour).** `Gamma::new(_, ResponseFn::Exp, LossFn::Nll, _)`
and `Gamma::default()` now build the `GammaParameterization::MeanShape` head, and
`NegativeBinomial::new(.., LossFn::Nll, _)` / `NegativeBinomial::default()` build
the `NbParameterization::MeanDispersion` head (reasons and measurements below).
Predictions keep the `(concentration, rate)` / `(total_count, probs)` output
convention. Unchanged: Gamma with any other response function, either family
with `LossFn::Crps`, and every model file saved before (they load with their
original head). In `MeanDispersion` the NB response functions passed to `new`
are kept but unused. To restore the old head:

```rust
let gamma = Gamma::new(stab, ResponseFn::Exp, LossFn::Nll, true)
    .with_parameterization(GammaParameterization::ConcentrationRate);
let nb = NegativeBinomial::new(stab, ResponseFn::Relu, ResponseFn::Sigmoid, LossFn::Nll, false)
    .with_parameterization(NbParameterization::TotalCountProbs);
```

#### Gamma: mean/shape head; NegativeBinomial MeanDispersion no longer caps the mean at 403

**`Gamma::new_mean_shape(stabilization, LossFn::Nll)`** boosts (log µ, log α)
instead of (log α, log β). In the default coordinates the Fisher information is
`[[α²ψ′(α), −α], [−α, α]]` — the two parameters are almost collinear
(correlation −0.999 at α = 330) because α also moves the mean α/β — so a
diagonal-Newton booster learns the dispersion O(1/α) too slowly. In (log µ, log α)
the information is exactly diagonal (Cox & Reid 1987 orthogonal coordinates).
Start values are the exact Gamma MLE; predictions are still reported as
`(concentration, rate)`. Measured (5 seeds, held-out NLL vs the default head):
power_plant −1.33 (90 % coverage 1.000 → 0.901), naval −4.66, synthetic Gamma
−0.65 (within 0.14 of the oracle), kin8nm +0.05 — the last recovers to −0.10
when the shape tree is frozen at ~200 rounds (`TrainConfig::param_freeze_rounds
= Some(vec![None, Some(200)])`; tune on validation). Default since 2026-11-10.

**`NegativeBinomial::new_mean_dispersion`** clamped log µ to [−10, 6], i.e. a
conditional mean of at most e⁶ ≈ 403, and kept pushing margins beyond the
clamp. The clamps are now log µ ∈ [−10, 15] and log θ ∈ [−6, 12], with zero
gradient beyond them. Fits whose means stayed below 403 are bit-identical; on
counts with means of 100–4000 held-out NLL improved by 1.08 nats (to within 0.08
of the oracle) and the head now beats the default (total_count, probs) head by
0.86 nats, whose coordinates are 0.88–0.995 correlated for the same reason as
Gamma's.

### NegativeBinomial: Damped log-θ curvature by default

**Breaking (training behaviour).** The `MeanDispersion` head now hands the booster
`HessianMode::Damped` curvature for log θ, max(observed, Fisher), in place of the
observed curvature floored at 0.01. The log-θ Fisher information has no closed
form. It is computed with a 25-node quadrature of a PGF identity
(`nb_log_theta_fisher`), so it costs about the same as the observed curvature.
Measured against the old curvature (5 seeds, held-out NLL): synthetic counts −0.028
(µ 100–4000), −0.004 (µ 1.5–60), −0.001 (near-Poisson); UCI Bike Sharing
`cnt` −0.010, `casual` −0.002. Restore the old behaviour with
`.with_hessian_mode(HessianMode::Observed)`. Files saved before the field existed
load as `Observed`.

### Per-parameter early stopping is on by default when a validation set is passed (results change)

For **Gaussian, StudentT, Gamma (mean/shape head) and NegativeBinomial
(MeanDispersion head)** with the NLL loss, any training run that is given a
validation set now also stops the scale/shape parameters on their own
validation signal. After every round the validation loss is recomputed with
each watched parameter's last 20 rounds undone. If undoing them would lower
the loss, that parameter's learning rate is halved. It is re-tested after
another 20 rounds and frozen once its rate falls below 1/64. The location keeps
training under the ordinary early stopping.

| family | watched | held-out NLL vs before (5 seeds) |
|---|---|---|
| Gaussian | scale | kin8nm −0.14, power_plant −0.14, protein −0.10, naval −0.37; simulations 0.000 |
| StudentT | df and scale | kin8nm −0.15, protein −0.11, naval −0.39; simulations ≤ +0.002 |
| Gamma (mean/shape) | shape | kin8nm −0.14, power_plant −0.11, naval −0.19; simulation +0.001 |
| NegativeBinomial (MeanDispersion) | log θ | Bike Sharing `cnt` −0.06, `casual` −0.03; simulations 0.000 |

90 % coverage moves toward nominal (Gaussian kin8nm 0.82 → 0.89, naval 0.82 →
0.94) and RMSE usually improves, because the location trains longer once the
scale stops over-fitting (power_plant 3.95 → 3.52). The cost is time: on real
data the fit runs 2–4× more rounds. The per-round overhead is one extra
validation-loss evaluation per watched parameter. Nothing changes without a
validation set, for other families, for the legacy Gamma/NB/Beta heads, or for
the CRPS loss.

**Also watched since cycle 8:** Beta's precision (mean/precision head;
Bike Sharing casual share −0.02, simulations unchanged) and Weibull's shape
(naval −0.81, kin8nm −0.035, power_plant −0.013 over 10 seeds, simulation
unchanged).

```rust
use gradientlss::model::ParamEarlyStopping;
// Opt out (previous behaviour):
let model = GradientLSS::<XGBoostBackend>::new(dist).without_param_early_stopping();
// Or configure: window, watched parameters, patience, anneal factor (0 = hard freeze).
let model = GradientLSS::<XGBoostBackend>::new(dist)
    .with_param_early_stopping(
        ParamEarlyStopping::new(20, Some(vec![1])).with_patience(1).with_anneal(0.5).with_hold_early_stopping(true),
    );
// After training:
model.param_freeze_log();              // round each parameter froze at, if any
model.param_early_stopping_triggers(); // every round each parameter's rate was cut
```

Use annealing rather than a hard freeze (`with_anneal(0.0)`), even though the
hard freeze is slightly better on some datasets (Gaussian protein −0.14). A
frozen StudentT scale cannot follow a df that is still moving (gaussian_sim
+0.39). A freeze triggered early by a transient signal leaves σ̂ stale
(Gaussian power_plant +0.26 at window 50). On Gamma naval the freeze was noisy
(one seed +0.15). With annealing, the worst seed on any dataset was +0.007.

**Each trigger also holds the global early stopping.** The
`early_stopping_rounds` patience counter restarts at every cut or freeze, so
the fit is not ended while a parameter's rate is still being cut. The best
iteration is still the validation argmin, so the worst case is the plain fit
50 rounds later. It recovered the one power_plant seed (Gaussian and Gamma)
where the stop had fired mid-anneal (−0.03 mean, −0.17 on that seed). It was
bit-identical everywhere else. Turn it off with
`.with_hold_early_stopping(false)`. Custom training callbacks can use the same
mechanism through the new provided method `TrainingCallback::holds_early_stopping`.

With per-parameter stopping, naval keeps improving past 2000 rounds. At a
6000-round cap it converged at 2600–3200 rounds and gained a further 0.03–0.05
nats. If your fits reach `num_boost_round`, raise it.

**hyper_opt** can tune it: the reserved axes `__pes_window` (`0` = off),
`__pes_patience` and `__pes_anneal` apply to every fold model on top of the
model's own setting. `ParamEarlyStopping::from_trial_params` applies the
winning values to the final fit.

### More training-side controls (cycle 3)

- **`TrainConfig` has a new field `param_freeze_rounds: Option<Vec<Option<usize>>>`**
  (exhaustive struct literals must add it; `..TrainConfig::default()` sidesteps
  it). It is gamboostLSS's per-parameter `mstop`: from round `rounds[j]` on,
  parameter j's gradients and Hessians are zeroed so its trees stop while the
  others keep learning. Freezing the Gaussian *scale* at 100–200 rounds improved
  held-out NLL by 0.10–0.16 nats on kin8nm, power_plant and protein (the location
  keeps training once the scale's over-fit no longer triggers early stopping) and
  hurt where σ(x) still carries structure (naval); tune it on validation NLL.
  In `hyper_opt`, a `__freeze_<j>` axis (e.g. `"__freeze_1": {"choices": [100, 200, 100000]}`)
  sets it per trial.
- `HyperOptConfig::trial_dist_fn: Option<TrialDistFn>` — a per-trial distribution
  builder: it receives the model's distribution and the trial's sampled params
  (including `__`-prefixed, transform-only axes) and returns the distribution the
  trial trains with. Use it to tune `HessianMode`, `with_param_gradient_scales`,
  `with_beta_nll` or the StudentT start cap in the same TPE study as the booster
  params, e.g. `"__scale_shrink": {"choices": [1.0, 0.5, 0.25]}`. The per-fold
  start-value cache is disabled while it is set.
- `Gaussian/StudentT::with_beta_nll(β)` — β-NLL row weighting (Seitzer et al.
  2022): every row's gradient and Hessian are multiplied by σ̂^{2β}, changing
  which rows drive the splits (high-variance rows regain influence on the
  location tree) without changing any leaf's Newton ratio.
- `Gamma` gained `HessianMode` (`with_hessian_mode`); default `Damped`, Fisher
  curvature `α²ψ′(α)` / `α` under `Exp` (the observed concentration curvature is
  negative for large yβ).
- The parallel NLL/start-value reductions are now deterministic (fixed-block
  sums), so seeded start values and `dist_select` are reproducible across runs
  and thread counts. Values move by ≲ 1e-12 relative versus before.

### Start values are now actually fitted on real-scale targets (results change)

`calculate_start_values` (used by `initialize: true` and `dist_select`) ran
L-BFGS from raw 0.5 and, for any target not already of order 1 (e.g. y ≈ 450),
the first line-search step overflowed and the optimizer silently returned the
initial point — so start values were 0.5 for every parameter and `dist_select`
ranked distributions by their loss *at 0.5*. A damped diagonal-Newton pre-phase
now precedes L-BFGS; start values are recovered correctly from target scales of
1e-3 to 1e5. Expect different (correct) start values, `dist_select` rankings and
early-round training behaviour on such targets.

StudentT additionally treats ν > 1e8 as its Gaussian limit (the df score there
was floating-point cancellation noise amplified by the `ExpDf` chain factor).

### `TrainConfig` has a new required field: `collect_train_metrics`

Any code constructing `TrainConfig` as an exhaustive struct literal will fail to
compile until the field is added:

```rust
let config = TrainConfig {
    num_boost_round: 100,
    early_stopping_rounds: Some(20),
    verbose: false,
    seed: 123,
    collect_train_metrics: false, // <-- new field
};

// or sidestep future additions entirely:
let config = TrainConfig {
    num_boost_round: 100,
    ..TrainConfig::default()
};
```

The field controls whether the per-round training-set metric is computed when
nothing else needs it. It **defaults to `false`** because skipping the pass
measured ~20% faster end-to-end training with a validation set (see
`examples/bench_collect_train_metrics.rs` and OPTIMIZATION.md). Consequences:

- `TrainingResult::train_history` is now **empty by default** when a validation
  set drives early stopping. Set `collect_train_metrics: true` to get the
  per-round train loss curve back.
- Nothing else changes: the train metric is still force-computed whenever it is
  actually consumed — no validation set (train loss drives early stopping),
  `verbose: true`, or registered callbacks — so early stopping, callbacks, and
  the trained model itself are identical either way.

### LightGBM saved-model format changed

LightGBM models now persist `best_iteration` (an 8-byte header before the model
text) so that predictions after early stopping use the best round, matching
LightGBM's Python behavior. **Models saved with an older build will fail to
load** — retrain and re-save them.

### Predictions may shift: `best_iteration` now tracks the true optimum

When early stopping is active, prediction truncates the ensemble to the best
boosting iteration. Previously that iteration only advanced when the validation
metric improved by the early-stopping *min-delta* (1e-4 relative), so it could
lag the real optimum and predict a slightly under-fit model. It now records the
true argmin of the validation curve (any improvement counts), matching
XGBoost/LightGBM in Python.

- **Effect:** predictions from a model trained *with a validation set + early
  stopping* can change slightly — generally a touch more accurate. Models
  trained without early stopping are **unchanged** (all trees are used).
- **Action:** none required. If you pinned expected prediction values in tests,
  regenerate the baselines.

The early-stopping *decision* is unchanged: the patience counter still uses the
min-delta, so training stops after the same number of rounds as before — only
which iteration is selected for prediction moved.

### Hyperparameter search now shuffles CV folds (results change)

`hyper_opt` with the default `CvScheme::KFold` now draws one seeded row
permutation (from `HyperOptConfig::seed`) and reuses it for every trial, instead
of using contiguous row-order folds. This makes `seed` actually control fold
composition (as documented) and matches `xgboost.cv`'s random partitioning.

- **Effect:** CV scores — and therefore which hyperparameters are selected —
  will differ from previous runs. This is a one-time shift toward more reliable
  estimates (contiguous folds were biased whenever rows were ordered by target).
- **Reproducibility:** a fixed `seed` (and `hp_seed`) now gives identical
  results run-to-run; a previous bug tied fold/RNG order to `HashMap` iteration
  order, so even seeded runs varied.

**Time-series data is unaffected — and this is the mechanism to use.** Shuffling
is applied *only* to `CvScheme::KFold`. Set `cv_scheme: CvScheme::TimeSeries` and
folds stay in row order with forward-chaining (expanding window): fold *i* trains
only on rows strictly before its test segment, so no future row ever informs a
past prediction. If your rows encode time, use `TimeSeries` and none of the
shuffling above applies:

```rust
let config = HyperOptConfig {
    cv_scheme: CvScheme::TimeSeries, // preserves order; no shuffle
    n_folds: 5,
    ..HyperOptConfig::default()      // KFold is the default
};
```

(The single 80/20 holdout, `n_folds <= 1`, is likewise never shuffled.)

### LightGBM: user-set parameters now actually apply (results change)

`LightGBMParams::set` previously *appended* to LightGBM's parameter list, and
LightGBM keeps the **first** occurrence of a duplicated key — so any override of
a default (`learning_rate`, `num_leaves`, `boosting`, `objective`, `verbose`,
plus `num_class` on repeated `set_n_dist_params`) was silently ignored, with the
warning suppressed by `verbose=-1`. This included **every hyper_opt-tuned
value**: LightGBM trials all trained with `learning_rate=0.1, num_leaves=31`
regardless of what the optimizer sampled. `set` now replaces in place.

- **Effect:** LightGBM models trained with non-default params (directly or via
  `hyper_opt`) will differ — they now honor your configuration. Re-evaluate any
  previously tuned "best" LightGBM hyperparameters: the scores they were
  selected by did not measure them.
- Relatedly, the binned `Dataset` is now constructed with the full training
  params (as `lgb.train` does in Python), so dataset-level params like
  `max_bin`, `min_data_in_leaf` (feature pre-filtering), and `zero_as_missing`
  take effect instead of being pinned to defaults.
- The `[low, high]` hyper_opt shorthand with **integer bounds** (e.g.
  `num_leaves: [2, 64]`) now samples integers (optuna `suggest_int` semantics);
  LightGBM rejects float-valued integer params, which the override fix exposed.

### XGBoost: `objective` and `base_score` are now forced (matching Python)

Python XGBoostLSS's `set_params_adj` unconditionally overwrites
`objective=None`, `base_score=0`, and `disable_default_eval_metric=True`; the
Rust backend now does the same at train time. A user-carried `objective` (e.g.
`"count:poisson"`) used to silently corrupt training — every internal predict
returned link-transformed values while boosting stayed on margin scale — and a
nonzero `base_score` shifted all predictions. If you were setting either, remove
them; the distribution's response functions are the link.

`TrainConfig::seed` is now wired into both backends (previously silently
ignored). Runs using `subsample`/`colsample_*`/`bagging_fraction` may differ; an
explicit `seed` set via backend params still wins.

### Mixture distribution fixes (predictions change)

- `predict(PredType::Samples | Quantiles)` drew mixture components with the
  wrong probabilities: a second Gumbel perturbation plus temperature softmax was
  layered on top of the already-softmaxed mixing weights, biasing selection
  toward uniform (weights 0.9/0.1 sampled at roughly 0.82/0.18). Components are
  now drawn as an exact categorical, matching `MixtureSameFamily.sample()`.
- `predict(PredType::Parameters)` now returns mixing **probabilities** (summing
  to 1) for the `mix_prob_*` columns, matching Python's `predict_dist`.
  Previously it returned raw unbounded logits.

### NLL metric now skips NaN terms (`torch.nansum` parity)

The evaluation metric aggregates per-sample log-probs with NaN treated as 0,
matching Python's `-torch.nansum(log_prob)`. Previously one NaN log-prob (e.g. a
0·inf at a support boundary) made the metric NaN for the rest of training, which
froze `best_iteration` and burned out the early-stopping patience where Python
trains on.

### CRPS loss now errors for non-reparameterizable distributions

`LossFn::Crps` computes gradients by finite differences through fixed-seed
sampling, which is only meaningful when the sampler is smooth in its parameters
(the `rsample` analogue). Training now fails fast with a clear error for
distributions whose samplers use rejection sampling or discrete draws (Gamma,
Beta, Dirichlet, StudentT, Poisson, NegativeBinomial, the ZI/ZA families, MVT,
Mixture) instead of silently producing garbage gradients. Torch errors the same
way — these have no `rsample`. Loc-scale and inverse-CDF families (Gaussian,
LogNormal, Logistic, Gumbel, Laplace, Cauchy, Weibull, Expectile, MVN, MVNLoRa,
SplineFlow) still support CRPS.

### Smaller behavioral fixes

- XGBoost `feature_importance` now honors the requested type: `Gain`/`Cover`
  are parsed from the model dump (per-split averages, matching
  `get_score`); previously every type silently returned split counts.
- XGBoost saved models append `n_features` (trailing, optional — old files
  still load, reporting 0); `GradientLSS::num_features()` now returns the real
  count instead of a stub 0.
- A `CallbackAction::Stop` on a new-minimum round no longer excludes that round
  from `best_iteration`.
- `num_boost_round: 0` now reports `n_iterations: 0` instead of 1.
- Sigmoid response derivatives are 0 in the clamp region (|x| ≳ 6.9), matching
  torch autograd through `torch.clamp` and this crate's own numerical path.
- Softplus keeps its tail below x = −20 (`exp(x) + ε`) instead of flooring to ε,
  matching torch.

## Usage

### Basic Example

```rust
use gradientlss::prelude::*;
use gradientlss::distributions::Gaussian;

// Create a Gaussian distribution
let dist = Gaussian::new(
    Stabilization::None,
    ResponseFn::Exp,  // For scale parameter
    LossFn::Nll,
    false,  // Don't initialize with start values
);

// With XGBoost backend
#[cfg(feature = "xgboost")]
{
    use gradientlss::backend::XGBoostBackend;
    
    let mut model = GradientLSS::<XGBoostBackend>::new(dist);
    
    // Create dataset
    let mut train_data = XGBoostDataset::from_data(
        features.view(),
        labels.view(),
    )?;
    
    // Train
    let params = XGBoostBackend::create_params(2);  // 2 params: loc, scale
    let config = TrainConfig {
        num_boost_round: 100,
        early_stopping_rounds: Some(20),
        verbose: true,
        seed: 123,
        collect_train_metrics: false,  // set true to record per-round train loss
    };
    
    model.train(&mut train_data, None, params, config)?;
    
    // Predict
    let predictions = model.predict(
        &test_features.view(),
        PredType::Parameters,
        1000,  // n_samples (for sampling)
        &[0.1, 0.5, 0.9],  // quantiles
        123,  // seed
    )?;
}
```

### Available Distributions

#### Univariate Distributions
- `Gaussian` - Normal distribution with loc (mean) and scale (std) parameters
- `Gamma` - Gamma distribution
- `Beta` - Beta distribution
- `StudentT` - Student's t distribution
- `Poisson` - Poisson distribution (discrete)
- `NegativeBinomial` - Negative binomial distribution (discrete)
- `Weibull` - Weibull distribution
- `LogNormal` - Log-normal distribution
- `Cauchy` - Cauchy distribution
- `Laplace` - Laplace distribution
- `Gumbel` - Gumbel distribution
- `Logistic` - Logistic distribution
- `ZAGamma` - Zero-adjusted Gamma distribution
- `ZINB` - Zero-inflated Negative Binomial distribution
- `ZIPoisson` - Zero-inflated Poisson distribution

#### Multivariate Distributions
- `MVN` - Multivariate Normal distribution with mean vector and Cholesky-decomposed covariance matrix
- `MVT` - Multivariate Student's T distribution with degrees of freedom, mean vector, and Cholesky-decomposed covariance matrix
- `Dirichlet` - Dirichlet distribution for compositional data (proportions that sum to 1)

More distributions can be added by implementing the `Distribution` trait.

### Response Functions

| Function | Description | Use Case |
|----------|-------------|----------|
| `Identity` | No transformation | Location parameters |
| `Exp` | Exponential | Strictly positive (scale) |
| `Softplus` | ln(1 + exp(x)) | Smooth positive |
| `Sigmoid` | 1/(1 + exp(-x)) | Bounded (0, 1) |
| `ExpDf` | exp(x) + 2 | Degrees of freedom |

### Multivariate Usage

For multivariate distributions, the target data must be provided in a flattened format:

#### MVN (Multivariate Normal)

```rust
use gradientlss::distributions::MVN;
use gradientlss::backend::lightgbm_backend::LightGBMBackend;
use gradientlss::model::GradientLSS;
use ndarray::array;

// Create a 2-target MVN distribution
let mvn = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
let mut model = GradientLSS::<LightGBMBackend>::new(mvn);

// For 3 observations with 2 targets each, labels should be:
// [y1_obs1, y2_obs1, y1_obs2, y2_obs2, y1_obs3, y2_obs3]
let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let labels = array![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]; // Flattened multivariate targets

let mut train_data = LightGBMBackend::Dataset::from_data(features.view(), labels.view())?;
```

#### MVT (Multivariate Student's T)

```rust
use gradientlss::distributions::{MVT, ResponseFn};

// Create a 2-target MVT distribution with separate response functions
let mvt = MVT::new(
    2,
    Stabilization::None,
    ResponseFn::Exp,      // For scale parameters
    ResponseFn::ExpDf,    // For degrees of freedom (ensures df > 2)
    LossFn::Nll,
    false
);
```

#### Dirichlet (Compositional Data)

```rust
use gradientlss::distributions::Dirichlet;

// Create a 3-target Dirichlet distribution for compositional data
let dirichlet = Dirichlet::new(3, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

// For Dirichlet, targets must sum to 1 for each observation
// [p1_obs1, p2_obs1, p3_obs1, p1_obs2, p2_obs2, p3_obs2, ...]
let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let labels = array![0.3, 0.4, 0.3, 0.2, 0.5, 0.3, 0.1, 0.6, 0.3];
```

The model will automatically handle the reshaping and parameter estimation for each multivariate distribution type.

## Model Interpretability

GradientLSS provides multiple approaches for model interpretation, matching the functionality of XGBoostLSS/LightGBMLSS.

### Built-in Interpretability Features

```rust
// Feature importance
let importance = model.feature_importance(
    FeatureImportanceType::Gain,
    Some(vec!["age".to_string(), "income".to_string()]),
)?;

// Partial dependence plots
let pdp = model.partial_dependence(
    &features.view(),
    0,  // feature index
    0,  // parameter index (e.g., 0 for 'loc')
    Some(50),  // grid size
    None,
    None,
)?;

// ICE (Individual Conditional Expectation) curves
let ice = model.ice_curves(
    &features.view(),
    0,  // feature index
    0,  // parameter index
    Some(50),
    None,
    None,
)?;
```

### Plotting (with `plotting` feature)

```rust
// Feature importance plot
model.plot_feature_importance(
    "importance.png",
    FeatureImportanceType::Gain,
    None,  // aggregated across parameters
    Some(feature_names),
    None,
)?;

// Partial dependence plot
model.plot_partial_dependence(
    &features.view(),
    0,  // feature index
    0,  // parameter index
    "pdp.png",
    None,
)?;

// Expectile plot (for Expectile distribution models)
model.expectile_plot(
    &features.view(),
    0,  // feature index
    "expectiles.png",
    Some(50),  // grid size
    Some("Age".to_string()),
    None,
    true,  // show confidence bands
)?;
```

### SHAP Integration

For advanced SHAP-based visualizations (beeswarm plots, dependency plots), GradientLSS provides a data export workflow compatible with Python's `shap` library:

```rust
// Export data for SHAP analysis
let shap_data = model.export_for_shap(
    &features.view(),
    Some(vec!["feature1".to_string(), "feature2".to_string()]),
)?;

// Save to JSON for Python consumption
shap_data.to_json_file("shap_export.json")?;
```

Then in Python:

```python
import json
import numpy as np
import shap
import xgboost as xgb

# Load the exported data
with open('shap_export.json', 'r') as f:
    data = json.load(f)

X = np.array(data['features'])
predictions = np.array(data['predictions'])
feature_names = data['feature_names']
param_names = data['param_names']

# Use feature importance directly from GradientLSS
if data['feature_importance']:
    for param, importance in data['feature_importance'].items():
        print(f"{param}: {importance}")

# Or train a surrogate model for full SHAP values
# (useful for beeswarm plots and SHAP dependency plots)
model_loc = xgb.XGBRegressor()
model_loc.fit(X, predictions[:, 0])  # Train on 'loc' predictions

explainer = shap.TreeExplainer(model_loc)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, feature_names=feature_names)
```

See `examples/shap_integration.rs` for a complete example.

## Backend Differences

### XGBoost (`--features xgboost`)
- Full support for custom objective functions
- Uses `update_custom()` for distributional gradient updates
- Row-major (C-order) gradient layout

### LightGBM (`--features lightgbm`)
- **Limited**: Current Rust bindings don't expose custom objective API
- Uses built-in objectives only
- Column-major (Fortran-order) gradient layout
- Consider using XGBoost for full distributional regression support

## Architecture

```
gradientlss/
├── distributions/     # Distribution implementations
│   ├── base.rs       # Distribution trait
│   └── gaussian.rs   # Gaussian distribution
├── backend/          # Gradient boosting backends
│   ├── traits.rs     # Backend trait definitions
│   ├── xgboost_backend.rs
│   └── lightgbm_backend.rs
├── utils.rs          # Response functions
├── model.rs          # GradientLSS model wrapper
└── error.rs          # Error types
```

## Adding New Distributions

1. Create a new file in `src/distributions/`
2. Implement the `Distribution` trait:

```rust
impl Distribution for MyDistribution {
    fn n_params(&self) -> usize { /* ... */ }
    fn params(&self) -> &[DistributionParam] { /* ... */ }
    fn log_prob(&self, params: &[f64], target: f64) -> f64 { /* ... */ }
    fn nll(&self, params: &ArrayView2<f64>, target: &ArrayView1<f64>) -> f64 { /* ... */ }
    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> { /* ... */ }
    // ... other required methods
}
```

3. Export from `src/distributions/mod.rs`

## License

MIT
