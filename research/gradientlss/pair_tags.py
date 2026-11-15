"""Paired comparison of two tags (B − A), per (dataset, family), matched on seed.

Usage: uv run --with pandas python -I pair_tags.py <csv> <tagA> <tagB> [--family f]
Prints ΔNLL mean ± s.e., sd, wins (B better), max Δ, cov90 A → B, ΔCRPS, mean valid ΔNLL.
"""
import sys

import pandas as pd

csv, a, b = sys.argv[1:4]
fam = sys.argv[sys.argv.index("--family") + 1] if "--family" in sys.argv else None
df = pd.read_csv(csv)
df = df[df["error"].isna() | (df["error"] == "")]
if fam:
    df = df[df["family"] == fam]
A = df[df.tag == a].drop_duplicates(["dataset", "family", "seed"], keep="last")
B = df[df.tag == b].drop_duplicates(["dataset", "family", "seed"], keep="last")
m = A.merge(B, on=["dataset", "family", "seed"], suffixes=("_a", "_b"))
rows = []
for (ds, f), g in m.groupby(["dataset", "family"]):
    d = g.nll_b - g.nll_a
    n = len(d)
    se = d.std(ddof=1) / n**0.5 if n > 1 else float("nan")
    rows.append(dict(
        dataset=ds, family=f, n=n, d_nll=f"{d.mean():+.4f} ± {se:.4f}", sd=f"{d.std(ddof=1):.3f}",
        wins=f"{int((d < 0).sum())}/{n}", max=f"{d.max():+.3f}",
        cov90=f"{g.cov90_a.mean():.3f}→{g.cov90_b.mean():.3f}",
        d_crps=f"{(g.crps_b - g.crps_a).mean():+.4f}" if g.crps_a.notna().all() else "",
        d_valid=f"{(g.valid_nll_b - g.valid_nll_a).mean():+.4f}",
        secs=f"{g.fit_secs_a.mean():.1f}→{g.fit_secs_b.mean():.1f}",
    ))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
