"""Proposal 027: which axis carries the wide-space gain (arms 2d, depth, subs, wide, d6).

Usage: uv run --with pandas --with numpy python -I c25_axis_summary.py <experiments_v2.csv> <dev|unseen> \
           <arm>=<log,log,...> ...          (arms: 2d depth subs wide; logs give the Tier 1 traces)

Test rows: development `c13_raw` / `c25_depth` / `c25_subs` / `c21_wide` (+ `_k<k>`), default `c12_off`;
unseen `c19_tpe` / `c25_depth` / `c25_subs` / `c22_wide`, default `c19_def` (then `c17_new`). `c25_d6` is
the untuned default at depth 6. guard = keep the refit only if its validation NLL beats the untuned default's.
"""
import math
import re
import sys

import numpy as np
import pandas as pd

csv_path, which = sys.argv[1], sys.argv[2]
logs = dict(a.split("=", 1) for a in sys.argv[3:])
DEV = which == "dev"
TAGS = {"2d": "c13_raw" if DEV else "c19_tpe", "depth": "c25_depth", "subs": "c25_subs", "wide": "c21_wide" if DEV else "c22_wide"}
BASE = ["c12_off"] if DEV else ["c19_def", "c17_new"]
key = ["family", "dataset", "seed"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def se(s):
    s = s.dropna()
    return s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")


def pm(d):
    d = d.dropna()
    return "n/a" if len(d) < 2 else f"{d.mean():+.4f} ± {se(d):.4f} ({int((d < 0).sum())}/{len(d)})"


df = pd.read_csv(csv_path, low_memory=False)
df = df[df.error.isna() | (df.error == "")]
dedup = lambda t: t.drop_duplicates(key, keep="last")
base = pd.concat([dedup(df[df.tag == t]) for t in BASE]).drop_duplicates(key, keep="first").set_index(key)
is_unseen = df.dataset.str.startswith("c_")
parts = []
for arm, pre in TAGS.items():
    for k in (10, 20):
        t = dedup(df[(df.tag == f"{pre}_k{k}") & (is_unseen != DEV)])
        parts.append(t.assign(arm=arm, k=k))
T = pd.concat(parts)
T = T[pd.MultiIndex.from_frame(T[key]).isin(base.index)]
ix = pd.MultiIndex.from_frame(T[key])
T["val_gain"] = T.valid_nll.values - base.valid_nll.reindex(ix).values
T["test_gain"] = T.nll.values - base.nll.reindex(ix).values
T["guard"] = T.test_gain.where(T.val_gain < 0, 0.0)
print("refit rows:", T.groupby(["arm", "k"]).size().to_dict(), " non-finite:", int((T.nll.isna() | (T.nonfinite_rows > 0)).sum()))

d6 = dedup(df[(df.tag == "c25_d6") & (is_unseen != DEV)]).set_index(key)
d6 = d6[d6.index.isin(base.index)]
d6_gain = (d6.nll - base.nll.reindex(d6.index)).groupby(level=[0, 1]).mean()

ARMS = ["2d", "depth", "subs", "wide"]
for k in (10, 20):
    P = T[T.k == k].pivot_table(index=key, columns="arm", values=["test_gain", "guard"])
    have = [a for a in ARMS if a in P["test_gain"]]
    P = P.dropna()
    out = []
    for (fam, ds), g in P.groupby(level=[0, 1]):
        row = dict(family=fam, dataset=ds, seeds=len(g))
        for a in have:
            row[a] = g["test_gain"][a].mean()
        for a in have:
            row["g_" + a] = g["guard"][a].mean()
        row["d6"] = d6_gain.get((fam, ds), float("nan"))
        if "depth" in have and "wide" in have:
            row["depth_minus_wide"] = pm(g["test_gain"]["depth"] - g["test_gain"]["wide"])
        if "subs" in have and "2d" in have:
            row["subs_minus_2d"] = pm(g["test_gain"]["subs"] - g["test_gain"]["2d"])
        out.append(row)
    t = pd.DataFrame(out)
    print()
    print(f"TIER 2, k = {k} ({which}) — test NLL − untuned default, mean over seeds; g_ = with the validation guard; d6 = untuned depth 6")
    print(t.round(4).to_string(index=False))
    m = {a: t[a].mean() for a in have}
    print("   mean over pairs:", {a: round(v, 4) for a, v in m.items()}, " guarded:", {a: round(t["g_" + a].mean(), 4) for a in have},
          " d6:", round(t.d6.mean(), 4))
    print("   median over pairs:", {a: round(t[a].median(), 4) for a in have})
    if "wide" in m:
        print("   share of wide's mean gain:", {a: round(m[a] / m["wide"], 2) for a in have}, " d6 / depth:",
              round(t.d6.mean() / m["depth"], 2) if "depth" in m else None)
    if not DEV:
        x = t[~((t.family == "poisson") & (t.dataset == "c_seoulbike"))]
        print("   without Poisson c_seoulbike, mean:", {a: round(x[a].mean(), 4) for a in have})
    print("   worse than default by > 0.005:", {a: f"{int((t[a] > 0.005).sum())}/{len(t)}" for a in have},
          " guarded:", {a: f"{int((t['g_' + a] > 0.005).sum())}/{len(t)}" for a in have}, f" d6: {int((t.d6 > 0.005).sum())}/{len(t)}")
    for a in have:
        w = t[t[a] > 0.005]
        if len(w):
            print(f"      {a}: " + ", ".join(f"{r.family} {r.dataset} {r[a]:+.3f} (guarded {r['g_' + a]:+.3f})" for _, r in w.iterrows()))

# Tier 1 and the picks, from the logs
PAT = re.compile(r"^HPO (\S+) (\S+) (\d+) trial (\d+) cv (\S+) mcw (\S+) lambda (\S+)(.*)$")
rows = []
for arm, paths in logs.items():
    for p in paths.split(","):
        for line in open(p):
            m = PAT.match(line)
            if not m:
                continue
            ds, fam, seed, tno, cv, mcw, lam, rest = m.groups()
            extra = dict(zip(rest.split()[0::2], map(float, rest.split()[1::2])))
            rows.append(dict(arm=arm, dataset=ds, family=fam, seed=int(seed), trial=int(tno), cv=float(cv), mcw=float(mcw), lam=float(lam),
                             depth=extra.get("max_depth", 4.0), sub=extra.get("subsample", 1.0), col=extra.get("colsample_bytree", 1.0)))
if rows:
    tr = pd.DataFrame(rows).drop_duplicates(["arm"] + key + ["trial"], keep="last")
    tr = tr[(tr.trial < 20) & (tr.dataset.str.startswith("c_") != DEV)]
    tr["cv"] = tr.cv.where(np.isfinite(tr.cv), np.inf)
    print()
    print("complete 20-trial searches per arm:", {a: int(g.groupby(key).trial.max().eq(19).sum()) for a, g in tr.groupby("arm")})
    best = tr.groupby(["arm"] + key).cv.min().unstack(0)
    print("TIER 1 — best CV score at 20 trials, minus the two-axis search's; mean ± s.e. (seeds better)")
    out = []
    for (fam, ds), g in best.groupby(level=[0, 1]):
        row = dict(family=fam, dataset=ds)
        for a in ARMS[1:]:
            if a in g:
                row[a] = pm(g[a] - g["2d"])
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False))
    for a in ("depth", "subs"):
        if a in tr.arm.values:
            w = tr[tr.arm == a]
            pk = w.loc[w.groupby(key).cv.idxmin()]
            print()
            print(f"{a} picks at 20 trials (median over seeds)")
            print(pk.groupby(["family", "dataset"]).agg(depth=("depth", "median"), depth_min=("depth", "min"), depth_max=("depth", "max"),
                                                        sub=("sub", "median"), col=("col", "median")).round(2).to_string())
