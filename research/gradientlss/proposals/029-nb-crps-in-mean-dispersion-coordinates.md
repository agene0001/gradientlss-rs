# Proposal 029 — NegativeBinomial: CRPS training in (log µ, log θ)

Status: **run** 2026-11-14 (N1 ✓, N2 ✓, N3 ✓, N4 ✗, N5 ✓, N6 ✓). **Pre-registered** 2026-11-14, after the unit test of the new gradients and before any
training run with them. Skill: `optimization-algorithm-research`. Follows 023 and 026.

## What 026 left
`LossFn::Crps` for NB still boosts in (total_count, probs): `new_mean_dispersion(_, Crps)` falls
back with a warning. After the large-count fix it is +1 % to +12 % behind NLL training on test
CRPS and **+33.6 % on c_abalone**, where all five fits end at the 2000-round cap. Those two
coordinates are collinear (cycle 4), which is why the NLL head moved to (log µ, log θ).

## Prototype
`NegativeBinomial::with_mean_dispersion_crps()` (opt-in; nothing existing changes): the raw scores
are (f, g) = (log µ, log θ), mapped jointly to (r, probs) = (θ, µ/(µ+θ)). With c = probs(1 − probs),
∂F(k)/∂f = c·F_p(k) and ∂F(k)/∂g = r·F_r(k) − c·F_p(k), from the tables the existing code already
builds. Gradient 2(A − S(y)) in those directions; Hessian the expected CRPS curvature
2 Σ_k (∂F(k)/∂η)² for **every** row (026 used it only above the old caps), floored at 0.05; a
coordinate outside its clamp has no gradient, as in the NLL head. Unit test: gradient = finite
difference of the loss through the joint map (2 %), Hessian = the curvature recomputed from finite
differences of the CDF (1 %), six (µ, θ, y) cases from µ = 3 to 700.

## Protocol
023's: NB on synth_nb_lo, c_abalone, bike_casual, bike_cnt, c_seoulbike, 5 seeds, location start,
eta 0.05, depth 4, 2000 rounds, patience 50. Tag `c26_nbmd` (`BENCH_LOSS=crps BENCH_NB_CRPS_MD=1`,
binary `/tmp/rb_c26`). References: `c24_crps` (fallback coordinates, after 026's fix), `c20_nll`
(NLL-trained, the default head). Metric: test CRPS. No per-parameter early stopping (as for every
CRPS fit today).

## Predictions
- **N1 (c_abalone).** The gap to NLL training falls from +33.6 % to below +5 %, and at least 3/5
  fits stop before the round cap.
- **N2.** On all five sets the new coordinates are within +5 % of the NLL-trained test CRPS.
- **N3.** They beat the fallback coordinates (lower mean test CRPS) on at least 4/5 sets.
- **N4.** CRPS training still does not pay: it beats NLL training by more than 1 % on at most 1/5.
- **N5.** Its test NLL is worse than the NLL-trained model's by more than 0.02 on at least 3/5.
- **N6.** No non-finite row.

## Decision rule
- N2 and N3 hold and no set is worse than the fallback by more than 0.5 % test CRPS: put to Felix
  the change "NB CRPS trains in (log µ, log θ)": `new_mean_dispersion(_, Crps)` and
  `with_parameterization(MeanDispersion)` do what they say, and `new(.., Crps, ..)` uses them
  (results change for CRPS-trained NB).
- Otherwise it stays an opt-in with its table, or is removed if it is worse than the fallback on
  a majority.
- N4 holds: the README keeps "train on the NLL".

## Result (2026-11-14, report 016 §3)
25 fits, no non-finite row, none at the round cap. Test CRPS, mean over 5 seeds; relative
differences are per seed, mean ± s.e. (seeds better).

| dataset (mean count) | NLL-trained | CRPS, fallback coordinates | CRPS, (log µ, log θ) | against the fallback | against NLL training | best round, min–max |
|---|---|---|---|---|---|---|
| synth_nb_lo | 4.017 | 4.136 | 4.040 | −2.3 % ± 0.3 (5/5) | +0.6 % ± 0.1 (0/5) | 52–86 |
| c_abalone (10) | 1.2069 | 1.612 | 1.2082 | −25.1 % ± 0.7 (5/5) | +0.1 % ± 0.3 (3/5) | 145–265 |
| bike_casual (36) | 6.07 | 6.29 | 6.00 | −4.7 % ± 1.1 (5/5) | −1.2 % ± 0.9 (4/5) | 554–905 |
| bike_cnt (190) | 17.22 | 19.19 | 16.77 | −12.7 % ± 0.8 (5/5) | −2.7 % ± 1.1 (4/5) | 830–1452 |
| c_seoulbike (700) | 69.79 | 70.62 | 65.66 | −7.0 % ± 1.2 (5/5) | −5.9 % ± 1.0 (5/5) | 635–960 |

- **N1 ✓.** c_abalone: +33.6 % → +0.1 %, and 5/5 fits stop before the cap (the fallback: 5/5 at it).
- **N2 ✓.** Within +5 % of NLL training on 5/5 (the worst is +0.6 %).
- **N3 ✓.** Better than the fallback coordinates on 5/5 sets and in 25/25 cells.
- **N4 ✗.** Predicted: CRPS training beats NLL training by more than 1 % on at most 1/5.
  Observed: 3/5, the three real count sets with means above 30. I expected a tie.
- **N5 ✓.** Test NLL is worse than the NLL-trained model's by more than 0.02 on 4/5:
  synth_nb_lo +0.022, bike_casual +0.063, bike_cnt +0.134, c_seoulbike +0.634 (c_abalone +0.005).
- **N6 ✓.**
- **What the CRPS gain costs.** The 90 % interval covers less: bike_casual 0.865 against 0.899 for
  the NLL-trained model, bike_cnt 0.847 against 0.928, c_seoulbike 0.785 against 0.934 (c_abalone
  0.974 / 0.972, synth_nb_lo 0.913 / 0.904). The CRPS-trained model is sharper in the centre and
  too narrow in the tails; RMSE is 0.4–3 % lower on the four real sets.
- **Cost.** Every row now builds its table and the squared-derivative sum; the fits ran at one
  thread on a machine shared with other jobs (bike_cnt 210 s a fit against 45 s for the fallback at
  four threads on a quieter machine), so the times are not comparable and are not a result.

**Decision (by the rule).** N2 and N3 hold and no set is worse than the fallback, so the change
goes to Felix: NB under `LossFn::Crps` trains in (log µ, log θ) (`new`, `new_mean_dispersion` and
`with_parameterization(MeanDispersion)`; results change for CRPS-trained NB; old models load as
they were trained). Until then it is opt-in: `.with_mean_dispersion_crps()`. N4 failed, so "train
on the NLL even if you score by CRPS" no longer holds for NB with the new coordinates: on the
three real sets the CRPS-trained model has the better test CRPS and the worse NLL and tail
coverage. The README says both. Poisson is untouched (026: CRPS training 5–7 % behind).
