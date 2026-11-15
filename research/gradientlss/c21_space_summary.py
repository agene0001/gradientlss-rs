"""Proposal 024: wide search space (mcw, lambda, max_depth, subsample, colsample_bytree) vs the two axes.

Usage: uv run --with pandas --with numpy python -I c21_space_summary.py <experiments_v2.csv> \
           2d=<c13_off logs,...> wide=<c21 logs,...>

Test rows: `c13_raw_k<k>` (2d), `c21_wide_k<k>` (wide), untuned default `c12_off`; for proposal 025 add
`tags=2d:c19_tpe,wide:c22_wide base=c19_def,c17_new`.
guard = keep the refit only if its validation NLL beats the untuned default's.
"""
import math
import re
import sys

import numpy as np
import pandas as pd

csv_path = sys.argv[1]
arms = dict(a.split("=", 1) for a in sys.argv[2:])
# optional: tags=2d:c19_tpe,wide:c22_wide  base=c19_def,c17_new   (proposal 025; defaults are proposal 024's)
TAGS = dict(t.split(":") for t in arms.pop("tags", "2d:c13_raw,wide:c21_wide").split(","))
BASE = arms.pop("base", "c12_off").split(",")
PAT = re.compile(r"^HPO (\S+) (\S+) (\d+) trial (\d+) cv (\S+) mcw (\S+) lambda (\S+)(?: max_depth (\S+) subsample (\S+) colsample_bytree (\S+))?")
key = ["family", "dataset", "seed"]
rows = []
for arm, paths in arms.items():
    for p in paths.split(","):
        for line in open(p):
            m = PAT.match(line)
            if m:
                ds, fam, seed, t, cv, mcw, lam, dep, sub, col = m.groups()
                rows.append(dict(arm=arm, dataset=ds, family=fam, seed=int(seed), trial=int(t), cv=float(cv), mcw=float(mcw),
                                 lam=float(lam), depth=float(dep) if dep else 4.0, sub=float(sub) if sub else 1.0,
                                 col=float(col) if col else 1.0))
tr = pd.DataFrame(rows).drop_duplicates(["arm"] + key + ["trial"], keep="last")
tr["cv"] = tr.cv.where(np.isfinite(tr.cv), np.inf)
tr = tr[tr.trial < 20]
print("complete 20-trial searches per arm:", {a: int(g.groupby(key).trial.max().eq(19).sum()) for a, g in tr.groupby("arm")})
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def se(s):
    s = s.dropna()
    return s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")


def pm(d):
    d = d.dropna()
    return "n/a" if len(d) < 2 else f"{d.mean():+.4f} ± {se(d):.4f} ({int((d < 0).sum())}/{len(d)})"


# Tier 1
best = {k: tr[tr.trial < k].groupby(["arm"] + key).cv.min().unstack(0) for k in (10, 20)}
print()
print("TIER 1 — best CV score, wide − 2d, mean ± s.e. (seeds where wide is better)")
out = []
for (fam, ds), _ in best[20].groupby(level=[0, 1]):
    out.append(dict(family=fam, dataset=ds, k10=pm((best[10]["wide"] - best[10]["2d"]).loc[(fam, ds)]),
                    k20=pm((best[20]["wide"] - best[20]["2d"]).loc[(fam, ds)])))
print(pd.DataFrame(out).to_string(index=False))
d = (best[20]["wide"] - best[20]["2d"]).groupby(level=[0, 1]).mean().dropna()
print(f"   k=20: wide better on {int((d < 0).sum())}/{len(d)} sets; median {d.median():+.4f}")

# Tier 2
df = pd.read_csv(csv_path)
df = df[df.error.isna() | (df.error == "")]
dedup = lambda t: t.drop_duplicates(["dataset", "family", "seed"], keep="last")
base = pd.concat([dedup(df[df.tag == t]) for t in BASE]).drop_duplicates(["dataset", "family", "seed"], keep="first").set_index(key)
parts = []
for arm, pre in TAGS.items():
    for k in (10, 20):
        parts.append(dedup(df[df.tag == f"{pre}_k{k}"]).assign(arm=arm, k=k))
T = pd.concat(parts)
ix = pd.MultiIndex.from_frame(T[key])
T["val_gain"] = T.valid_nll.values - base.valid_nll.reindex(ix).values
T["test_gain"] = T.nll.values - base.nll.reindex(ix).values
T["guard"] = T.test_gain.where(T.val_gain < 0, 0.0)
print()
print("refit rows:", T.groupby(["arm", "k"]).size().to_dict(), " non-finite:", int((T.nll.isna() | (T.nonfinite_rows > 0)).sum()))
for k in (10, 20):
    P = T[T.k == k].pivot_table(index=key, columns="arm", values=["test_gain", "guard"])
    if "wide" not in P["guard"]:
        continue
    P = P.dropna()
    out = []
    for (fam, ds), g in P.groupby(level=[0, 1]):
        out.append(dict(family=fam, dataset=ds, seeds=len(g), plain_2d=g["test_gain"]["2d"].mean(), plain_wide=g["test_gain"]["wide"].mean(),
                        guard_2d=g["guard"]["2d"].mean(), guard_wide=g["guard"]["wide"].mean(),
                        guard_wide_minus_2d=pm(g["guard"]["wide"] - g["guard"]["2d"]),
                        plain_wide_minus_2d=pm(g["test_gain"]["wide"] - g["test_gain"]["2d"])))
    t = pd.DataFrame(out)
    print()
    print(f"TIER 2, k = {k} — test NLL − untuned default (negative = tuning helps); paired differences wide − 2d")
    print(t.round(4).to_string(index=False))
    dg = t.guard_wide - t.guard_2d
    print(f"   worse than default by > 0.005: plain 2d {int((t.plain_2d > 0.005).sum())}/{len(t)}, plain wide {int((t.plain_wide > 0.005).sum())}/{len(t)}, "
          f"guard 2d {int((t.guard_2d > 0.005).sum())}/{len(t)}, guard wide {int((t.guard_wide > 0.005).sum())}/{len(t)}")
    print(f"   guarded wide − guarded 2d: better by > 0.01 on {int((dg < -0.01).sum())}/{len(t)}, worse by > 0.01 on {int((dg > 0.01).sum())}/{len(t)}; "
          f"mean over sets: plain 2d {t.plain_2d.mean():+.4f}, plain wide {t.plain_wide.mean():+.4f}, guard 2d {t.guard_2d.mean():+.4f}, guard wide {t.guard_wide.mean():+.4f}")

# what the wide picks look like
w = tr[tr.arm == "wide"]
pk = w.loc[w.groupby(key).cv.idxmin()]
print()
print("wide picks at k = 20 (median over seeds; depth also min–max)")
print(pk.groupby(["family", "dataset"]).agg(depth_med=("depth", "median"), depth_min=("depth", "min"), depth_max=("depth", "max"),
                                            sub=("sub", "median"), col=("col", "median"),
                                            log10_mcw=("mcw", lambda s: np.log10(s).median()), log10_lam=("lam", lambda s: np.log10(s).median())).round(2).to_string())
# CV score by depth, pooled: does the fold score move with depth?
w = w[np.isfinite(w.cv)].copy()
w["rel"] = w.cv - w.groupby(key).cv.transform("median")
print()
print("wide trials: CV score − the search's median, by depth (mean over all trials of a set)")
print(w.pivot_table(index=["family", "dataset"], columns="depth", values="rel", aggfunc="mean").round(3).to_string())
