"""Proposal 021: curvature / start arms for Cauchy, Logistic, Gumbel, Laplace, LogNormal.

Usage: uv run --with pandas python -I c15_port_summary.py <experiments_v2.csv> [prefix=c15] [arms=base,init,d_raw,d_loc,d_ls,o_ls]

Prints test NLL per (family, dataset, arm) with the Gaussian reference, each arm's difference to
`base` (mean ± s.e., seeds better), and the per-family choice by mean rank of VALIDATION NLL.

The first Cauchy rows of c15_base / c15_init were written by a scorer that dropped every Cauchy row
(no finite CRPS); they are recognised (NaN NLL with every test row non-finite) and ignored.
"""
import math
import sys

import pandas as pd

csv_path = sys.argv[1]
opts = dict(a.split("=", 1) for a in sys.argv[2:])
prefix = opts.get("prefix", "c15")
arms = opts.get("arms", "base,init,d_raw,d_loc,d_ls,o_ls").split(",")
df = pd.read_csv(csv_path)
df = df[df.tag.isin([f"{prefix}_{a}" for a in arms])].copy()
df["arm"] = df.tag.str[len(prefix) + 1:]
artefact = (df.family == "cauchy") & df.nll.isna() & (df.nonfinite_rows == df.n_test)
df = df[~artefact].drop_duplicates(["arm", "dataset", "family", "seed"], keep="last")
df["failed"] = (df.error.notna() & (df.error != "")) | df.nll.isna() | (df.nonfinite_rows > 0)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)

fams = [f for f in ["cauchy", "logistic", "gumbel", "laplace", "lognormal"] if f in set(df.family)]
ref = df[(df.family == "gaussian")].groupby(["dataset", "arm"]).nll.mean().unstack()
print("failed cells (error, NaN or non-finite test rows) per arm:", df[df.family.isin(fams)].groupby("arm").failed.sum().to_dict())
print()
t = df[df.family.isin(fams)].pivot_table(index=["family", "dataset"], columns="arm", values="nll", aggfunc="mean")
t = t[[a for a in arms if a in t.columns]]
if "base" in ref.columns:
    t["gaussian"] = [ref["base"].get(ds, float("nan")) for _, ds in t.index]
print("mean test NLL over seeds")
print(t.round(3).to_string())
print()


def pm(d):
    d = d.dropna()
    if len(d) < 2:
        return "n/a"
    return f"{d.mean():+.3f} ± {d.std(ddof=1) / math.sqrt(len(d)):.3f} ({int((d < 0).sum())}/{len(d)})"


p = df[df.family.isin(fams)].pivot_table(index=["family", "dataset", "seed"], columns="arm", values="nll")
out = []
for (fam, ds), g in p.groupby(level=[0, 1]):
    row = dict(family=fam, dataset=ds)
    for a in arms:
        if a != "base" and a in g:
            row[a] = pm(g[a] - g["base"])
    out.append(row)
print("arm − base, test NLL: mean ± s.e. (seeds where the arm is better)")
print(pd.DataFrame(out).to_string(index=False))
print()
m = p.groupby(level=[0, 1]).mean()
for a in arms:
    if a != "base" and a in m:
        d = m[a] - m["base"]
        print(f"  {a}: worse than base by > 0.005 on {int((d > 0.005).sum())}/{len(d)} pairs "
              + ("" if not (d > 0.005).any() else "-> " + ", ".join(f"{f}/{ds} {v:+.3f}" for (f, ds), v in d[d > 0.005].items())))
print()

# selection by validation NLL: mean rank over datasets, per family
v = df[df.family.isin(fams)].pivot_table(index=["family", "dataset"], columns="arm", values="valid_nll", aggfunc="mean")
v = v[[a for a in arms if a in v.columns]]
rk = v.rank(axis=1).groupby(level=0).mean()
print("mean rank of validation NLL over datasets (1 = best); the recipe is chosen on this, not on test")
print(rk.round(2).to_string())
print("chosen:", {f: rk.loc[f].idxmin() for f in rk.index})
