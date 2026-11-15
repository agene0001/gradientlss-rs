# Proposal 038 — `max_bin` at 50 trials on fresh datasets

Status: **run** 2026-11-14 (D1 ✗, D2 ✓, D3 ✓ / ✗, D4 ✓, D5 ✗, D6 ✓, D7 ✓; the rule fails on (a): one guard miss in 12).
**Pre-registered** 2026-11-14, before any fit on the new datasets. Skill:
`automl-hyperparameter-research` (search-space design; Tier 2, confirmation). Follows 034, whose
rule says a pass on the 36 pairs is development evidence only.

## Question
On the 36 pairs, adding `max_bin` (log-uniform integer 16–1024) to the two axes passed the
inclusion rule at 50 trials (0/36 guard misses, better than the guarded two-axis search on 30/36,
median −0.009) after failing it at 20. Those pairs had been used for this parameter twice. Does
the pass hold on datasets that no cycle has touched?

## Data (fixed before any fit)
Seven UCI datasets fetched today, none used in cycles 1–30; `prep_confirm_data3.py` documents the
columns kept. Only the target's range, the row and feature counts and whether it is an integer
were looked at. The bench reads them as `c_m_<name>`.

| dataset | UCI id | rows | features | target | families |
|---|---|---|---|---|---|
| m_energy | 242 | 768 | 8 | heating load, 6–43 | Gaussian, Gamma |
| m_steel | 851 | 12 000 (subsample) | 10 | usage kWh, 2.5–153 | Gaussian, Gamma |
| m_fishtox | 504 | 908 | 6 | LC50, 0.05–9.6 | Gaussian |
| m_tempbias | 514 | 7 588 | 24 | next-day maximum temperature, 17–39 | Gaussian, Gamma |
| m_video | 335 | 12 000 (subsample) | 25 | transcoding time (s), 0.18–183 | Gaussian, Gamma |
| m_pm25 | 381 | 12 000 (subsample) | 14 | PM2.5, integer 2–980 | Poisson, NegativeBinomial |
| m_fbcomments | 363 | 12 000 (subsample) | 53 | comments, integer 0–1059, 56 % zeros | NegativeBinomial |

12 (family, dataset) pairs: Gaussian 5, Gamma 4, Poisson 1, NegativeBinomial 2 — the mix of the
12 fresh pairs of 032. If a pair cannot be fitted at all (untuned default non-finite on every
seed) it is dropped and reported; nothing is substituted.

## Arms
| arm | space | trials | refits at | tags |
|---|---|---|---|---|
| `def` | none (untuned default) | | | `c34_def` |
| `2d50` | mcw, λ | 50 | 20, 50 | `c34_2d_k20`, `c34_2d_k50` |
| `bin50` | mcw, λ, max_bin 16–1024 (log, integer) | 50 | 20, 50 | `c34_bin_k20`, `c34_bin_k50` |

Everything else as 034: TPE, 3 folds, eta 0.05, 2000 rounds, patience 50, seeds 0–4, binary
`/tmp/rb_c30`, one job per (arm, pair, seed) in `/tmp/c34/`, `run_c34.sh`.

## Rule (034's, scaled from 36 to 12 pairs), at 50 trials
(a) guarded `bin50` worse than the untuned default by more than 0.005 on 0/12 pairs; (b) guarded
`bin50` better than guarded `2d50` on at least 8/12; (c) the median over pairs of that difference
≤ −0.005.

## Predictions
- **D1.** (a) holds: 0/12.
- **D2.** (b) between 8 and 10 of 12.
- **D3.** (c) between −0.020 and −0.005. With D1 and D2: the rule passes, narrowly. (The unseen
  subset of 034 had (c) = −0.002, so a fail on (c) is the likeliest way to be wrong.)
- **D4 (more trials do not help two axes).** Guarded `2d50` at 50 trials is better than at 20 on
  at most 7/12 pairs, and the median of that difference is within ±0.003.
- **D5 (20 trials are not enough for three axes).** At 20 trials `bin50`'s refit fails the same
  rule against `2d50`'s 20-trial refit.
- **D6 (picks).** The median picked max_bin over seeds is below 64 on at least 3/12 pairs and
  above 256 on at least 2/12.
- **D7.** Plain `bin50` is worse than plain `2d50` by more than 0.005 on at most 3/12 pairs.

## Decision
No default changes: the library has no default space.
- **Pass:** the README's `max_bin` paragraph changes from "tested, not recommended" to: with a
  budget of about 50 trials and the validation guard, add `max_bin` (log-uniform 16–1024) to the
  two axes; at 20 trials do not. With both tables.
- **Fail:** the README records that the 50-trial arm passed on the 36 pairs and failed on fresh
  data, with the counts; the advice stays "leave 256".
Either way the two-axis finding (50 trials no better than 20) is reported with this batch's
numbers.

## Cost
120 fifty-trial searches and 60 untuned fits: about 12 stream-hours.

## Result (2026-11-14, report 016 §11)
180 jobs: 120 fifty-trial searches, 240 refits, 60 untuned fits, no non-finite row, no pair
dropped. Test NLL minus the untuned default, mean over 5 seeds; "guarded" keeps a refit only if
its validation NLL beats the untuned fit's. Summary: `c30_summary.py /tmp/c34 … base=c34_def
pairs=12`.

| arm (trials) | plain: mean | median | worse than default | guarded: mean | median | worse than default |
|---|---|---|---|---|---|---|
| two axes (20) | −0.033 | −0.028 | 0/12 | −0.045 | −0.028 | 1/12 |
| two axes (50) | −0.051 | −0.026 | 1/12 | −0.055 | −0.025 | 0/12 |
| + max_bin (20) | −0.037 | −0.019 | 2/12 | −0.043 | −0.031 | 0/12 |
| + max_bin (50), `bin50` | −0.061 | −0.034 | 0/12 | −0.062 | −0.044 | **1/12** |

| pair | plain `2d50` | plain `bin50` | guarded `2d50` | guarded `bin50` | guarded difference | max_bin picked (median) |
|---|---|---|---|---|---|---|
| Gamma m_energy | −0.016 | −0.017 | −0.020 | **+0.029** | +0.050 | 239 |
| Gamma m_steel | −0.155 | −0.177 | −0.155 | −0.177 | −0.022 | 722 |
| Gamma m_tempbias | −0.029 | −0.056 | −0.029 | −0.056 | −0.027 | 55 |
| Gamma m_video | −0.029 | −0.038 | −0.033 | −0.059 | −0.026 | 53 |
| Gaussian m_energy | −0.152 | −0.119 | −0.139 | −0.155 | −0.016 | 226 |
| Gaussian m_fishtox | −0.010 | −0.005 | −0.016 | −0.007 | +0.010 | 97 |
| Gaussian m_steel | −0.054 | −0.053 | −0.075 | −0.079 | −0.004 | 336 |
| Gaussian m_tempbias | −0.008 | −0.030 | −0.009 | −0.031 | −0.022 | 59 |
| Gaussian m_video | +0.007 | −0.022 | +0.002 | −0.013 | −0.015 | 43 |
| NB m_fbcomments | −0.022 | −0.026 | −0.021 | −0.023 | −0.002 | 274 |
| NB m_pm25 | −0.007 | −0.004 | −0.006 | −0.006 | −0.001 | 53 |
| Poisson m_pm25 | −0.137 | −0.186 | −0.155 | −0.169 | −0.014 | 67 |

**The rule at 50 trials: (a) 1/12, (b) 10/12, (c) −0.015 → fails on (a).**

- **D1 ✗.** One guard miss: Gamma m_energy +0.029. It is one seed (seed 1: validation NLL 0.39
  against the default's 0.70, test NLL 1.03 against 0.87); the guard also threw away the two
  seeds where the refit was better on test (−0.045, −0.188), so the plain arm is −0.017 on this
  pair and the guarded one +0.029. m_energy has 768 rows: 115 validation and 153 test rows, and
  the default's own test NLL runs from 0.14 to 0.87 over the seeds.
- **D2 ✓.** (b) 10/12.
- **D3 ✓ / ✗.** (c) −0.015, inside the predicted −0.020 to −0.005; the pass I predicted from D1–D3
  does not happen.
- **D4 ✓.** Guarded two axes at 50 trials are better than at 20 on 6/12 pairs, median −0.001.
  (The plain mean does improve here, −0.033 → −0.051, where it got worse on the 36 pairs.)
- **D5 ✗.** At 20 trials the arm passes the scaled rule against the 20-trial two-axis refit:
  0/12, 9/12, −0.005. On the 36 pairs it failed at 20 and passed at 50; here it is the other way
  round. Which budget passes is not stable; that it is close either way is.
- **D6 ✓.** Median picked max_bin below 64 on 5/12 pairs, above 256 on 3/12.
- **D7 ✓.** Plain `bin50` worse than plain `2d50` by more than 0.005 on 2/12 (Gaussian m_energy
  +0.033, m_fishtox +0.006), better on 6/12.

**Read-out.** The size of the gain confirms: guarded `bin50` beats guarded `2d50` on 10/12 fresh
pairs with a median of −0.015 (36 pairs: 30/36, −0.009), and plain `bin50` is worse than the
untuned default on 0/12. The safety clause does not confirm: one pair worse than the default
with the guard, on the smallest dataset, where the guard's 115 validation rows cannot rank
anything. Over 48 pairs the 50-trial arm has 1 guard miss; the two-axis search at 50 trials has
3.

**Decision (by the rule).** Fail: the README records that the 50-trial arm passed on the 36 pairs
and failed (a) on fresh data, with the counts; the advice stays "leave 256". No default changes.
Not decided here, for Felix: whether one miss in 48 on a 768-row dataset is a reason to keep a
parameter out that is better on 40 of 48 pairs. Relaxing (a) after seeing the result would need
its own fresh batch.

### Addendum, 2026-11-15: owner's decision
The arm failed the pre-registered rule on (a) (1/12, Gamma m_energy +0.029), and under "Fail"
above the advice was to stay "leave 256". Felix was asked whether that one uncaught loss in 48
pairs should keep `max_bin` out of the recommended search and answered no. The README now
recommends the axis at 50 trials or more with the guard, states that this is an exception to the
rule, and names the pair. `hyper_opt::max_bin_search_space()` was added. The result above is
unchanged: the rule failed.
