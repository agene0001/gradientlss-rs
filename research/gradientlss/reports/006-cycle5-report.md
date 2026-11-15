# Cycle 5 report — new defaults, NB log-θ curvature, real count data, per-parameter early stopping

Date: 2026-11-10. Rows: `experiments_v2.csv` tags `c5a_*` … `c5f_*`; detail in `research_log.md`
"Cycle 5". Proposals 006 (NB curvature) and 007 (per-parameter early stopping). Suites: 243 lib
(both backends), 219 lib (xgboost), 137 / 129 integration.

## 1. Defaults Felix approved (landed)
- `Gamma::new(_, Exp, Nll, _)` / `Gamma::default()` → `MeanShape`; `NegativeBinomial::new(.., Nll, _)`
  / `default()` → `MeanDispersion`. CRPS (both) and non-Exp Gamma keep the legacy heads. Saved
  files without the field load with the legacy head. Opt-out: `.with_parameterization(..)`; for
  NB it restores the response functions passed to `new` exactly (tested).

## 2. NB log-θ curvature (proposal 006)
- **Limitation:** observed information floored at 0.01. The Fisher information
  I = θ²E[ψ′(θ) − ψ′(Y+θ)] − θµ/(µ+θ) has no closed form and falls to ~5e-4 near Poisson.
- **Math:** PGF identity → a µ-independent 1-D integral
  I = θ∫e^{−u}[t/(1−e^{−t})(1−PGF(t)) − (1−e^{−µt})]du, t = u/θ. A 25-node exp-sinh rule has p99
  relative error 3e-3 over the clamp domain. Validated against the exact series and Monte-Carlo
  E[score²]. Cost ≈ observed curvature.
- **Results:** Damped vs Observed (5 seeds):

  | data | ΔNLL | seeds |
  |---|---|---|
  | synth_nb_hi | −0.028 | 5/5 |
  | synth_nb_lo | −0.004 | 5/5 |
  | synth_nb_ps (near-Poisson) | −0.001 | 4/5 |
  | bike_cnt | −0.010 | 4/5 |
  | bike_casual | −0.002 | 4/5 |

- **Decision:** Damped is now the NB default (better or neutral everywhere). Old files load as
  Observed.

## 3. Real count data (UCI Bike Sharing, hourly, 17 379 rows)
MeanDispersion vs legacy (r, p): `cnt` tie (+0.001 ± 0.010), `casual` −0.032 (5/5). The default
change holds on real data. MeanDispersion under-covered there (cov90 0.87 / 0.79), and §4 fixes it.

## 4. Per-parameter early stopping (proposal 007)
- **API:** `GradientLSS::with_param_early_stopping(ParamEarlyStopping::new(W, Some(vec![j])))`.
- **Rule:** freeze parameter j when undoing its last W rounds lowers the validation loss
  (`with_patience(p)` requires p consecutive confirmations).
- **Test:** a dynamic freeze at round k is bit-identical to a static `param_freeze_rounds` freeze
  at k. The freeze rounds are exposed via `param_freeze_log()`.

ΔNLL vs none (W = 20, scale/shape/dispersion only, 5 seeds):

| family | dataset | ΔNLL (seeds) | cov90 | tuned fixed freeze |
|---|---|---|---|---|
| Gaussian | kin8nm | −0.137 (5/5) | .82 → .92 | −0.157 |
| Gaussian | power_plant | −0.172 (5/5) | .90 → .95 | −0.116 |
| Gaussian | protein | −0.136 (5/5) | | −0.098 |
| Gaussian | naval | **−0.267 (5/5)** | | 0 (every k hurt) |
| Gaussian | 3 sims | ≤ +0.001 | | 0 |
| Gamma MS | kin8nm | −0.139 (5/5) | | |
| Gamma MS | power_plant | −0.147 (5/5) | | |
| Gamma MS | naval | −0.013 ± 0.079 (noisy) | | |
| Gamma MS | synth_gamma | ±0.001 | | |
| NB MD | bike_cnt | −0.041 (5/5) | .88 → .94 | |
| NB MD | bike_casual | −0.020 (5/5) | .79 → .83 | |
| NB MD | synth NB ×3 | ≤ +0.001 | | |

- **Failure found:** W = 50, patience 1 froze power_plant's scale at round ≈ 95 on a transient
  signal, giving +0.256. Patience 20 repairs it to −0.126. It shows on validation (valid NLL
  3.11 vs 2.62).
- **Validation selection:** choosing over {none, W20, W50/P20} is never worse than none on any
  dataset by > 0.007. Gaussian: −0.15 / −0.17 / −0.14 / −0.32. Gamma: naval −0.127.
  NB bike: −0.061 / −0.026.

## 5. Decisions
- Landed defaults: Gamma MeanShape, NB MeanDispersion, NB Damped log-θ curvature.
- Per-parameter early stopping is shipped as an opt-in, recommended recipe (W = 20, scale /
  shape / dispersion only). It is not a default: it needs a validation set. It is now the
  strongest calibration lever in the crate and supersedes tuning `param_freeze_rounds` by hand.

## 6. Next investigations (ranked)
1. Wire `ParamEarlyStopping` (W, patience) into `hyper_opt` so the window is tuned with the rest.
2. Freeze → *anneal* instead (halve the parameter's gradient scale on each trigger) so a frozen
   σ̂ can still follow a location that keeps improving (the power_plant W=50 failure mode).
3. Per-parameter early stopping for StudentT (df and scale separately) and with the residual warm
   start.
4. Beta / Weibull orthogonality audit (carried over from cycle 4).
