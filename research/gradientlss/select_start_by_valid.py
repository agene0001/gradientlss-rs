"""Proposal 013 A′: per (dataset, family, seed) keep whichever of two tags has the
lower best *validation* NLL; report the chosen cells' test ΔNLL against tag A.

Usage: uv run --with pandas python -I select_start_by_valid.py <csv> <tagA> <tagB> [--family f]
"""
import sys

import pandas as pd

csv, a, b = sys.argv[1:4]
fam = sys.argv[sys.argv.index("--family") + 1] if "--family" in sys.argv else None
df = pd.read_csv(csv)
df = df[df["error"].isna() | (df["error"] == "")]
if fam:
    df = df[df.family == fam]
A = df[df.tag == a].drop_duplicates(["dataset", "family", "seed"], keep="last")
B = df[df.tag == b].drop_duplicates(["dataset", "family", "seed"], keep="last")
m = A.merge(B, on=["dataset", "family", "seed"], suffixes=("_a", "_b"))
m["pick_b"] = m.valid_nll_b < m.valid_nll_a
m["nll_sel"] = m.nll_b.where(m.pick_b, m.nll_a)
rows = []
for (ds, f), g in m.groupby(["dataset", "family"]):
    d = g.nll_sel - g.nll_a
    db = g.nll_b - g.nll_a
    n = len(g)
    rows.append(dict(dataset=ds, family=f, n=n, picks_b=f"{int(g.pick_b.sum())}/{n}",
                     d_sel=f"{d.mean():+.4f} ± {d.std(ddof=1) / n**0.5:.4f}", max_sel=f"{d.max():+.3f}",
                     d_b=f"{db.mean():+.4f}", oracle_pick=f"{(g[['nll_a','nll_b']].min(axis=1) - g.nll_a).mean():+.4f}"))
print(pd.DataFrame(rows).to_string(index=False))
