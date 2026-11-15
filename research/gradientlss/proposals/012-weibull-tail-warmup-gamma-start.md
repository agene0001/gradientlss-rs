# Proposal 012 — Weibull tail blow-ups, deliberate warm-up, Gamma raw-vs-fitted start

Status: **resolved** — reports/010-cycle9-report.md. Was **pre-registered** (cycle 9, 2026-11-10), after the Weibull worst-row diagnosis only.

## C — Weibull upper-tail blow-ups
Diagnosis (BENCH_DEBUG_WORST): power_plant seed 0 (PES): one test row y = 491.97, λ̂ = 476.4,
k̂ = 222.8 → NLL 1314 (20 % of the test total; next-worst 30). naval seed 3: y = 1.000 (the
target's upper bound), λ̂ = 0.996, k̂ = 2169 → NLL 5711. The Weibull upper-tail cost is
(y/λ)^k = e^z with z ≈ k ln(y/λ) the distance in predicted-sd units — exponential in z (a
Gaussian charges z²/2), whatever k is. A shape cap can only lower z by widening every
prediction; naval needs k in the thousands.
- C1: no fixed cap K ∈ {50, 200, 1000} (`Weibull::with_max_concentration`) is better or neutral
  on all four Weibull datasets (synth_weibull, kin8nm, power_plant 10 seeds, naval) with the
  default recipe. If that holds: no default change; the README documents the tail property
  and the cap as an opt-in.

## W — deliberate warm-up for StudentT (gaussian_sim raw-start benefit)
Two mechanisms are possible for the raw start's accidental benefit:
(W-a) location first: the scale/df trees only start after the location has done its job;
(W-b) approach from above: the scale is inflated early and then shrinks to the truth.
Arms (StudentT median start, PES off, 7 datasets): location-only warm-up of 20 and of 50
rounds (`GradientLSS::with_warmup_rounds`: non-location gradients zeroed for the first n
rounds); start scale = 3·sd(y) (preset).
- W1 (W-a): warm-up 50 closes the gaussian_sim gap to raw (≤ +0.002) and is ≤ +0.005 elsewhere.
- W2 (W-b): start scale 3·sd closes the gap (≤ +0.002) and is ≤ +0.005 elsewhere.
Exactly one of W1/W2 is expected to hold. Default change only if one holds everywhere, also
with the default PES.

## G — why Gamma MS's raw start beats the fitted start (naval +0.155)
c8l: naval, raw [log µ 0.5, log α 0.5] best; loc [ln ȳ, 0.5] +0.046; fitted [ln ȳ, ln α̂] +0.155.
(G-a) an overdispersed start (α = 1.65 ≪ α̂ ≈ 10⁴) that approaches the shape from below helps;
(G-b) the mean's travel from 0.5 is a warm-up (as for StudentT).
New arms (default PES): [0.5, ln α̂] (raw mean, fitted shape); [ln ȳ, ln α̂ − 4.6] (fitted mean,
100× more dispersed than the MLE); fitted + 50-round location-only warm-up.
- G1 (G-a): [0.5, ln α̂] is ≥ +0.08 worse than raw on naval; [ln ȳ, ln α̂ − 4.6] within 0.03 of
  loc.
- G2 (G-b): [0.5, ln α̂] within 0.03 of raw; the warm-up arm closes ≥ half of fitted's gap.

## Addendum G3 (after c9g, before c9g6k) — pre-registered
G1 ✗ (raw mean + fitted shape +0.044, not ≥ +0.08), G2 ✗ (warm-up closes 45 % < half).
Every naval Gamma fit sits at the 2000-round cap (best_iter ≈ 1998), so the comparison is of
progress within a budget. G3: at a 6000-round cap (default PES + hold), raw − fitted on naval
shrinks to |Δ| < 0.05 (finite-budget artefact); if it stays ≥ +0.10 in favour of raw, the start
changes the optimum reached (path dependence). Fitted at 6000 = c7n_hold (gamma_ms).
