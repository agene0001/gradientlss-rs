"""Paired summary of research_bench rows: Fisher − Observed per seed.

Usage: uv run --with pandas python -I summarize.py [experiments.csv] [--tag TAG ...]
Prints, per (tag, dataset, family): n seeds, mean ± s.e. of Δnll, Δcrps,
Δcov90, Δrmse, Δbest_iter, the sign count (seeds where Fisher's NLL is
lower), and the absolute levels for context (Observed NLL, oracle NLL).
"""
import sys

import pandas as pd

path = next((a for a in sys.argv[1:] if a.endswith(".csv")), "experiments.csv")
tags = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--tag"]

df = pd.read_csv(path)
df = df[df["error"].isna() | (df["error"] == "")]
if tags:
    df = df[df["tag"].isin(tags)]

keys = ["tag", "dataset", "family", "seed"]
wide = df.pivot_table(
    index=keys, columns="hessian_mode",
    values=["nll", "crps", "cov50", "cov90", "pit_ks", "rmse", "best_iter", "oracle_nll", "nonfinite_rows"],
)
wide = wide.dropna(subset=[("nll", "fisher"), ("nll", "observed")])

rows = []
for (tag, ds, fam), g in wide.groupby(level=["tag", "dataset", "family"]):
    n = len(g)

    def ms(col):
        d = g[(col, "fisher")] - g[(col, "observed")]
        m = d.mean()
        se = d.std(ddof=1) / n ** 0.5 if n > 1 else float("nan")
        return m, se

    dn, dnse = ms("nll")
    dc, dcse = ms("crps")
    dcov, dcovse = ms("cov90")
    dr, drse = ms("rmse")
    di, _ = ms("best_iter")
    wins = int((g[("nll", "fisher")] < g[("nll", "observed")]).sum())
    rows.append(dict(
        tag=tag, dataset=ds, family=fam, n=n,
        d_nll=f"{dn:+.4f} ± {dnse:.4f}", z=f"{dn / dnse:+.1f}" if dnse == dnse and dnse > 0 else "",
        d_crps=f"{dc:+.4f} ± {dcse:.4f}", d_cov90=f"{dcov:+.3f}", d_rmse=f"{dr:+.4f}", d_iter=f"{di:+.0f}",
        fisher_wins=f"{wins}/{n}",
        obs_nll=f"{g[('nll', 'observed')].mean():.4f}", fis_nll=f"{g[('nll', 'fisher')].mean():.4f}",
        oracle=f"{g[('oracle_nll', 'fisher')].mean():.4f}",
        nonfinite=int(g[("nonfinite_rows", "fisher")].sum() + g[("nonfinite_rows", "observed")].sum()),
    ))

out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(out.to_string(index=False))
