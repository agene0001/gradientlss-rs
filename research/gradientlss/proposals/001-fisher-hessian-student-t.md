# Proposal 001 — Expected (Fisher) information as the Newton curvature for heavy-tailed LSS likelihoods

Status: **implemented (non-default), benchmarked** — superseded as the recommendation by proposal 002 (`Damped`), which matches Fisher where it wins and removes its far-from-optimum blow-ups (cycle 1, 2026-11-09)

## Measured / observed limitation

Both backends run diagonal Newton boosting: leaf value `w = −Σg / (Σh + λ)`,
split gain `Σg²/(Σh+λ)`. The crate feeds them the *observed* information
(second derivative of the per-row NLL at the current fit) for every
distribution except Gaussian (`gaussian.rs:182-190` deliberately uses the
Fisher value `2/σ²`). For StudentT (`student_t.rs:222-247`) the observed
curvatures are, in distribution-parameter space,

| param | observed `∂²ℓ/∂θ²` | sign |
|---|---|---|
| μ | `(ν+1)(ν−d²) / (σ²(ν+d²)²)` | **negative when d² > ν** (exactly the outlier rows a t-model exists for) |
| σ | `−1/σ² + (ν+1)d²(3ν+d²)/(σ²(ν+d²)²)` | negative when d² is small |
| ν | `−¼ψ′((ν+1)/2) + ¼ψ′(ν/2) − 1/(2ν²) + d²(2ν+d²(1−ν))/(2ν²(ν+d²)²)` | sign depends on d² |

plus the chain term `∂ℓ/∂θ · f″(η)` in raw space, which is also unsigned.
Nothing floors these. In a leaf, rows with negative `h` cancel positive rows,
so `Σh` can approach 0 (or go negative), the leaf step is then governed by
`λ` alone and can flip sign relative to the gradient. XGBoost's
`min_child_weight` prunes leaves with `Σh < 1`, so negative-curvature rows
also *suppress splits* precisely where tail structure lives.

Python XGBoostLSS/LightGBMLSS inherit the same thing (autograd observed
Hessian, `distribution_utils.py:441-444`), so parity is not an argument
either way. The Gaussian code already deviates from Python for this reason.

## Competing explanations for any StudentT under-performance

- H0a: sampling error / hyperparameter confounding — handled by paired seeds,
  identical splits, identical params, oracle NLL on simulated data.
- H0b: the heavy-tail cases are just harder; a Fisher Hessian changes the
  step size, not the optimum, so with early stopping on a validation set it
  may make **no** difference. (Plausible. Then the result is "neutral".)
- H1: negative/erratic observed curvature degrades the Newton direction and
  the split selection → worse out-of-sample NLL/CRPS, more non-finite fits,
  larger seed variance. Fisher curvature fixes this at zero cost.

## Proposed math

Expected information per observation, parameters (μ, σ, ν), with
`d = (y−μ)/σ`:

```
I_μμ = (ν+1) / ((ν+3) σ²)
I_σσ = 2ν / ((ν+3) σ²)
I_νν = ¼[ψ′(ν/2) − ψ′((ν+1)/2)] − (ν+5) / (2ν(ν+1)(ν+3))
```

Verified by numerical integration of `E[−∂²log f]` at ν ∈ {2.5, 4, 8, 30}
(scipy, relative error < 2e-5 for all three; see research_log cycle 1).
Sources: standard result, e.g. Lange, Little & Taylor (1989) "Robust
Statistical Modeling Using the t Distribution", JASA; Fonseca, Ferreira &
Migon (2008) Biometrika (the `I_νν` form above). I could not retrieve these
through Lacuna; the formula is checked numerically instead.

Chain rule in raw (margin) space η with θ = f(η): because `E[∂ℓ/∂θ] = 0`,
the chain term vanishes in expectation and

```
h_η = I_θθ · f′(η)²
```

- loc  (Identity):  `h = (ν+1) / ((ν+3) σ²)`
- scale (Exp, σ = e^η):  `h = 2ν / (ν+3)`        (→ Gaussian's 2 as ν→∞ ✓)
- df (ExpDf, ν = e^η + 2):  `h = I_νν · (ν−2)²`

Gradients are unchanged. Hessians are strictly positive for ν > 0.

Also for the reverse experiment on Gaussian: `HessianMode::Observed` restores
`h_σ = (−1/σ² + 3d²/σ⁴) f′² + (1/σ − d²/σ³) f″` so the effect size can be
measured on the simplest family where the crate already chose Fisher.

## Falsifiable predictions

1. On `studentT_sim` (oracle params available): Fisher ≤ Observed test NLL in
   ≥ 4 of 5 seeds, with the paired mean difference > 2 s.e. **or** no
   difference (→ H0b, report neutral).
2. Fisher has no more non-finite/ diverged fits than Observed; Observed has
   ≥ 1 in the seed sweep *or* higher seed variance of test NLL.
3. On real UCI targets with a StudentT family (kin8nm, power-plant, naval,
   protein), the direction of Δ(NLL, CRPS) agrees with (1) on the majority.
4. Gaussian reverse test: Observed-mode Gaussian is worse than or equal to
   the current Fisher default; if it is *better*, the whole hypothesis is in
   doubt and must be reported.

## Implementation plan (minimal, preserves production behaviour)

- `base.rs`: `pub enum HessianMode { Observed, Fisher }`, Serialize/Deserialize.
- `StudentT`: new field `#[serde(default = "...observed")] hessian_mode`,
  builder `with_hessian_mode`. Default stays Observed. Old saved models load.
- `Gaussian`: same field, default stays Fisher.
- Unit tests: Fisher hessians positive; `E_y[observed]` ≈ Fisher by Monte
  Carlo at fixed params (ties the two modes together mathematically);
  serde round-trip of the default.
- Benchmark harness `examples/research_bench.rs` (xgboost feature) — see
  reports/.

## Invariants to test

- `h > 0` for all finite rows, both families, Fisher mode.
- Mean over 200k simulated y of the observed `h` equals the Fisher `h` within
  Monte-Carlo error, in raw space, for all three StudentT params.
- Gradients identical between modes (bitwise).
