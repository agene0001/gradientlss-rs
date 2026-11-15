"""Cycle 13 search diagnostics from the bench's stderr HPO lines.

Usage: uv run --with pandas python -I hpo_summary.py <raw.log>[,<raw.log>...] <norm.log>[,<norm.log>...]
Prints, per (dataset, family): the best-so-far CV score difference norm − raw after k trials (mean over
seeds; negative = the normalised axis found a better CV score), and the geometric-mean chosen mcw/λ at
k = 20 in each arm.
"""
import math
import re
import sys

import pandas as pd

PAT = re.compile(r"^HPO (\S+) (\S+) (\d+) trial (\d+) cv (\S+) mcw (\S+) lambda (\S+)")


def read(paths, arm):
    rows = []
    for p in paths.split(","):
        for line in open(p):
            m = PAT.match(line)
            if m:
                ds, fam, seed, t, cv, mcw, lam = m.groups()
                rows.append(dict(arm=arm, dataset=ds, family=fam, seed=int(seed), trial=int(t),
                                 cv=float(cv), mcw=float(mcw), lam=float(lam)))
    return rows


df = pd.DataFrame(read(sys.argv[1], "raw") + read(sys.argv[2], "norm"))
df = df.sort_values(["arm", "dataset", "family", "seed", "trial"])
df["cv"] = df["cv"].where(df["cv"].apply(math.isfinite), float("inf"))
df["best"] = df.groupby(["arm", "dataset", "family", "seed"])["cv"].cummin()
ks = [1, 3, 5, 10, 15, 20]
out = []
for (ds, fam), g in df.groupby(["dataset", "family"]):
    row = dict(dataset=ds, family=fam)
    for k in ks:
        b = g[g.trial == k - 1].pivot_table(index="seed", columns="arm", values="best")
        if {"raw", "norm"} <= set(b.columns):
            d = (b.norm - b.raw).dropna()
            row[f"k{k}"] = f"{d.mean():+.4f} ({int((d < 0).sum())}/{len(d)})"
    for arm in ["raw", "norm"]:
        a = g[g.arm == arm]
        idx = a.groupby("seed")["cv"].idxmin()
        sel = a.loc[idx]
        row[f"mcw_{arm}"] = f"{math.exp(sel.mcw.apply(math.log).mean()):.3g}"
        row[f"lam_{arm}"] = f"{math.exp(sel.lam.apply(math.log).mean()):.3g}"
    out.append(row)
pd.set_option("display.width", 250)
print("best-so-far CV score, norm − raw (mean over seeds; wins = norm better)")
print(pd.DataFrame(out).to_string(index=False))
