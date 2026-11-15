# Proposal 030 — adaptive λ: gradient-regularised Newton boosting

Status: **run** 2026-11-14 (R1 ✓, R2 ✓ for the paper form and ✗ for the excess-only form, R3 ✗ / ✓, R4 ✗, R5 ✓; one control arm added while running). **Pre-registered** 2026-11-14, after the unit test of the mechanism and before any
training run with it or at eta ≠ 0.05 on these pairs. Skill: `optimization-algorithm-research`.
Source: Zozoulenko, Falkowski, Cass, Gonon, "Gradient Regularized Newton Boosting Trees with
Global Convergence", arXiv 2605.00581 (read in full text via Lacuna, 2026-11-14;
https://lacuna.tiptreesystems.com/work/gradient-regularized-newton-boosting-trees-with-global).
On the list since cycle 1 ("literature, untested").

## The method, and what it becomes here
The paper adds (λ_k/2)‖f‖² to the round's quadratic with λ_k = λ_base + sqrt(M‖g_k‖), ‖g_k‖ the
L² norm of the per-row gradients and 2M the Lipschitz constant of the Hessian; vanilla Newton
boosting can diverge at eta = 1 (Charbonnier loss), this cannot.
- ‖f‖² is the L² norm over rows, so the penalty adds λ_k to **every row's Hessian** (a leaf's
  denominator grows by λ_k × its row count). That needs no backend change: it is done in the
  objective. (It is not XGBoost's `lambda`, which is per leaf.)
- Here the margins of different parameters have different units, so the term is per parameter j.
  M is not known per family; taking M_j = h̄_j^{3/2} (h̄_j the round's mean Hessian) makes the term
  unit-free: **λ_kj = c · h̄_j · ρ_j^{1/4}, ρ_j = mean g_j² / h̄_j.** My adaptation, not the paper's.
- For a log-likelihood at the true parameters E g² = E h, so ρ ≈ 1 and the term does **not**
  vanish at the statistical optimum (the paper's optimum interpolates the training rows): late in
  training the paper form is about a learning rate of eta/(1 + c). The **excess-only** form,
  λ_kj = c · h̄_j · max(ρ_j^{1/4} − 1, 0), keeps only the early damping.

`GradientLSS::with_gradient_regularization(Some(GradientRegularization { strength, excess_only }))`,
opt-in, off by default; bench `BENCH_GRN=<c>` / `x:<c>`.

## Why it might matter for held-out accuracy, and why it might not
The cycles so far removed early instability case by case (Damped curvature, data starts, PES) at
eta 0.05. A user at XGBoost's default eta 0.3 has none of that margin. If large steps cost
accuracy or stability there, an adaptive ridge is the general remedy; if they do not, there is
nothing for it to fix.

## Protocol
12 pairs: the 10 development pairs of 024 plus StudentT × {synth_t3, protein}. 5 seeds, depth 4,
2000 rounds, patience 50, PES auto. eta ∈ {0.05, 0.3, 1.0} × arm ∈ {vanilla, `grn` (c = 1),
`grnx` (excess-only, c = 1)}: tags `c26_e<eta>_<arm>`, binary `/tmp/rb_c26`. A **failed cell** has
a non-finite row or a test NLL more than 1 above the same pair and seed at vanilla eta 0.05.

## Predictions
- **R1 (is there anything to fix?).** Vanilla at eta 0.3 is worse than at 0.05 by more than 0.02
  on at least 5/12 pairs; at eta 1.0 it has at least one failed cell.
- **R2 (stability).** At eta 1.0 both forms have fewer failed cells than vanilla and a lower mean
  test NLL on at least 8/12 pairs.
- **R3 (cost at the bench's rate).** At eta 0.05 `grnx` is within ±0.005 of vanilla on at least
  9/12 pairs; `grn` is worse by more than 0.005 on at least 3/12 (it halves the late step, and
  several fits already end at the round cap).
- **R4 (accuracy at eta 0.3).** `grnx` beats vanilla by more than 0.005 on at least 6/12 pairs and
  is worse by more than 0.005 on at most 1/12.
- **R5 (the ceiling).** No GRN arm at any eta beats vanilla at eta 0.05 by more than 0.005 on more
  than 3/12 pairs.

## Decision rule
No default changes whatever happens (a new mechanism, c untuned).
- R4 holds: the README's opt-in section describes it for learning rates above 0.1, with the table.
- R2 holds and R4 fails: described as a stability option for eta near 1 only.
- Neither: the option stays undocumented outside the research workspace and out of the prelude,
  and the report says it did not help.
A tuned c, LightGBM, and the HPO interaction are not tested.

## Amendment (2026-11-14, while the arms were running; partial tables seen for eta 1 and 0.3)
A control was missing. Late in training the paper form is close to vanilla at eta/(1 + c) = eta/2,
so "`grn` beats vanilla at the same eta" cannot separate an adaptive ridge from a smaller step.
**Added arms:** vanilla at eta 0.15 and 0.5 (tags `c26_e0.15_van`, `c26_e0.5_van`), the
half-step controls for `grn` at eta 0.3 and 1. R1–R5 are scored as written; the controls enter
the interpretation only: `grn` at eta counts as more than a step-size change only where it beats
vanilla at eta/2 by more than 0.005.

## Result (2026-11-14, report 016 §4)
660 fits (11 arm × eta cells, 12 pairs, 5 seeds), no non-finite row. Test NLL, mean over seeds.

| comparison | better by > 0.005 | worse by > 0.005 | mean difference |
|---|---|---|---|
| vanilla eta 0.3 − vanilla eta 0.05 (threshold 0.02) | 0/12 | 9/12 | +0.080 |
| vanilla eta 1 − vanilla eta 0.05 (threshold 0.02) | 0/12 | 12/12 | +0.427 |
| `grn` − vanilla, eta 0.05 | 1/12 | 8/12 | +0.108 |
| `grnx` − vanilla, eta 0.05 | 2/12 | 2/12 | +0.029 |
| `grn` − vanilla, eta 0.3 | 8/12 | 0/12 | −0.040 |
| `grnx` − vanilla, eta 0.3 | 4/12 | 0/12 | −0.019 |
| `grn` − vanilla, eta 1 | 12/12 | 0/12 | −0.169 |
| `grnx` − vanilla, eta 1 | 5/12 | 1/12 | −0.109 |
| **control: `grn` at eta 0.3 − vanilla at eta 0.15** | 2/12 | 8/12 | +0.027 |
| **control: `grn` at eta 1 − vanilla at eta 0.5** | 3/12 | 5/12 | +0.065 |

Failed cells (test NLL more than 1 above vanilla at eta 0.05; none non-finite): eta 1 vanilla
11/60 (naval under Gaussian and Gamma, Poisson bike_cnt), `grn` 3, `grnx` 6; none at eta ≤ 0.5.

- **R1 ✓.** There is something to fix above eta 0.05: 9/12 pairs lose more than 0.02 at eta 0.3
  (kin8nm +0.17, protein +0.12, power_plant +0.11), and eta 1 has 11 failed cells.
- **R2 ✓ for `grn`, ✗ for `grnx`.** Both have fewer failed cells at eta 1 (3 and 6 against 11).
  `grn` is better than vanilla on 12/12 pairs; `grnx` on 5/12: on five pairs it is vanilla to the
  last digit, because with a data start and Damped curvature ρ ≤ 1 from the first round and the
  excess term is zero.
- **R3 ✗ / ✓.** At eta 0.05 `grnx` is within ±0.005 on 8/12, not 9: Poisson bike_cnt +0.377 and
  power_plant +0.007 are worse, protein −0.031 and NB bike_cnt −0.008 better. `grn` is worse on
  8/12 (naval +0.15 and +0.19, Poisson bike_cnt +0.81: all at the round cap).
- **R4 ✗.** `grnx` at eta 0.3 is better on 4/12, not 6 (worse on none).
- **R5 ✓.** No GRN arm at any eta beats vanilla at eta 0.05 on more than 2/12 pairs.
- **The control decides it.** `grn` at eta 0.3 loses to vanilla at eta 0.15 on 8/12 pairs and at
  eta 1 to vanilla at eta 0.5 on 5/12 (wins 2 and 3). What `grn` gains over vanilla at the same
  eta is what a smaller step gains, and a smaller step gains more.

Side observations, not predictions: vanilla at eta 0.15 beats eta 0.05 where the 2000-round cap
binds (Gaussian naval −0.021, power_plant −0.010, NB bike_cnt −0.020) and loses where it does not
(protein +0.096, kin8nm +0.035); `grn` at eta 0.05 on Gaussian protein (−0.058, at the round cap)
says a step below 0.05 would help there. Both are learning-rate effects.

**Decision (by the rule).** R4 fails and R2 does not hold for both forms: the option stays
undocumented outside the research workspace and out of the prelude. It did not help: the
instability it addresses is already handled by Damped curvature and the data starts, and what
remains above eta 0.05 is a step-size effect that a smaller eta handles better. No default
changes. The README's advice stays: a small learning rate with enough rounds.
