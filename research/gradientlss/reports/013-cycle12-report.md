# Cycle 12 report — scale-relative tree regularisation (Hessian normalisation)

Date: 2026-11-11. Rows: tags `c12*`. Proposal 015. **No default change; new opt-in.**

## Measured limitation
XGBoost's `min_child_weight` and `lambda` act on raw Hessian sums, so the same hyperparameters mean
different regularisation in different units. Gaussian power_plant with labels × 100 lost up to 1.3
nats (cycle 11).

## Math
Scaling parameter j's (g, h) by κⱼ gives leaf = −Σg/(Σh + λ/κⱼ), with mcw → mcw/κⱼ and γ → γ/κⱼ. The
unregularised Newton step and `eta` are unchanged.
- **Marginal:** κⱼ = 1 / (mean Hessian at the intercept-only MLE).
- **Start:** κⱼ = 1 / (mean Hessian at the start values).

## Implementation and tests
- `GradientLSS::with_hessian_normalization(HessianNormalization::{Off, Start, Marginal})`.
- It is applied in the objective before per-parameter stopping and freezes, so it works with both
  backends. Training-only; trees, predictions and serialization are unchanged.
- Test: with mcw = λ = 0 the fit moves ≤ 3·10⁻³ (f32); with the defaults it moves ≈ 0.2.

## Results (5 seeds; Marginal − Off)
| family | dataset | ΔNLL | seeds better |
|---|---|---|---|
| Gaussian | kin8nm | −0.018 | 5/5 |
| Gaussian | power_plant | −0.044 | 4/5 |
| Gaussian | protein | −0.018 | |
| Gaussian | naval | +0.016 | |
| Gaussian | gaussian_sim | **+0.009** | 0/5 |
| Gamma MS | naval | −0.043 | |
| Gamma MS | kin8nm | −0.013 | |
| Gamma MS | others | ≤ +0.005 | |
| NB MD | bike_cnt | −0.008 | |
| NB MD | others | ±0.002 | |
| Poisson | bike_cnt | **+0.066** | 0/5 |
| Poisson | others | ±0.006 | |

- **Start − Off:** Gaussian power_plant +0.236 (0/5). The raw start gives κ_loc = σ₀², which over-
  regularises.
- **Unit invariance** (unit-free start + Marginal, labels × 100 against × 1): ≤ 0.006 mean, against
  +1.30 without normalisation.

## Decision
- Not a default: it fails the rule on gaussian_sim and Poisson bike_cnt. For high-information rows,
  "one row's worth" of λ is much stronger L2 (Poisson bike: ~190× in raw units).
- What it fixes is the *units* dependence, not the choice of strength, which still differs by dataset.
  That matches the pre-registered null: it acts like a regularisation retune.
- Kept as an opt-in.

## Next investigations (ranked)
1. An equal-budget `hyper_opt` comparison: tune (mcw, λ) on the raw axis against the normalised axis,
   with the same number of trials. If the normalised axis reaches equal or better validation NLL in
   fewer trials, recommend it for tuning (it needs the hyper_opt fold models to inherit the setting).
2. Early stopping's relative min-delta (cycle 10): the remaining source of unit dependence.
