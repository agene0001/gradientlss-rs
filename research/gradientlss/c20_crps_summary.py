"""Proposal 023: CRPS training vs NLL training for Poisson and NB (tags c20_nll, c20_crps).

Usage: uv run --with pandas python -I c20_crps_summary.py <experiments_v2.csv> [crps tag, default c20_crps]
"""
import math
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1])
k = ["family", "dataset", "seed"]
dd = lambda t: d[d.tag == t].drop_duplicates(k, keep="last").set_index(k)
crps_tag = sys.argv[2] if len(sys.argv) > 2 else "c20_crps"   # proposal 026: c24_crps
n, c = dd("c20_nll"), dd(crps_tag)
cols = ["nll", "crps", "cov90", "rmse", "best_iter", "n_iter", "fit_secs", "nonfinite_rows"]
J = n[cols].join(c[cols], lsuffix="_n", rsuffix="_c", how="inner")
J["rel"] = J.crps_c / J.crps_n - 1
J["cap"] = J.best_iter_c >= 1990
pd.set_option("display.width", 250)
se = lambda s: s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")
g = J.groupby(level=[0, 1]).agg(
    seeds=("nll_n", "count"), crps_nll=("crps_n", "mean"), crps_crps=("crps_c", "mean"), crps_rel=("rel", "mean"), crps_rel_se=("rel", se),
    crps_better=("rel", lambda s: int((s < 0).sum())), nll_nll=("nll_n", "mean"), nll_crps=("nll_c", "mean"),
    cov90_nll=("cov90_n", "mean"), cov90_crps=("cov90_c", "mean"), rmse_nll=("rmse_n", "mean"), rmse_crps=("rmse_c", "mean"),
    rounds_nll=("best_iter_n", "mean"), rounds_crps=("best_iter_c", "mean"), at_cap=("cap", "sum"), secs_crps=("fit_secs_c", "mean"),
    nonfinite=("nonfinite_rows_c", "sum"))
print("test scores, mean over seeds; crps_rel = CRPS-trained / NLL-trained − 1 (positive = CRPS training is worse on CRPS)")
print(g.round(4).to_string())
print()
print(f"cells: {len(J)}; CRPS-trained cells at the round cap: {int(J.cap.sum())}; "
      f"pairs where CRPS training beats NLL training on test CRPS by > 1 %: {int((g.crps_rel < -0.01).sum())}/{len(g)}")
if crps_tag != "c20_crps":
    old = dd("c20_crps")
    Q = c[["crps", "nll", "best_iter"]].join(old[["crps", "nll", "best_iter"]], lsuffix="_new", rsuffix="_old", how="inner")
    Q["rel"] = (Q.crps_new / Q.crps_old - 1).abs()
    print()
    print(f"{crps_tag} against c20_crps (the CRPS arm before the fix): test CRPS")
    print(Q.groupby(level=[0, 1]).agg(cells=("rel", "size"), identical=("rel", lambda s: int((s == 0).sum())), max_rel_diff=("rel", "max"),
                                      crps_old=("crps_old", "mean"), crps_new=("crps_new", "mean"),
                                      rounds_old=("best_iter_old", "mean"), rounds_new=("best_iter_new", "mean")).round(6).to_string())
