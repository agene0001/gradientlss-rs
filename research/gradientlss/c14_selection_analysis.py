"""Proposal 017 follow-up (zero compute): how much of a CV gain over the default survives the refit?

Usage: uv run --with pandas --with numpy python -I c14_selection_analysis.py <experiments_v2.csv> \
           tpe=<log>[,<log>] rand=... dtpe=... ds10=...

Every arm uses the same folds (fold seed = bench seed), so the default's CV score from `dtpe` trial 0
is the default's score for every arm. For each refit (arm, k, set, seed):
    cv_gain   = CV score of the pick − CV score of the default        (what the search believed)
    val_gain  = validation NLL of the refit − of the untuned default  (bench early-stopping split)
    test_gain = test NLL of the refit − of the untuned default
Prints per set: mean gains, the share of the CV gain that survives, the within-set correlation, the
picks' location, and P4 (ds10 vs dtpe, best CV at k = 20).
"""
import math
import re
import sys

import numpy as np
import pandas as pd

csv_path = sys.argv[1]
arms = dict(a.split("=", 1) for a in sys.argv[2:])
PAT = re.compile(r"^HPO (\S+) (\S+) (\d+) trial (\d+) cv (\S+) mcw (\S+) lambda (\S+)")
KS = [5, 10, 20]
rows = []
for arm, paths in arms.items():
    for p in paths.split(","):
        for line in open(p):
            m = PAT.match(line)
            if m:
                ds, fam, seed, t, cv, mcw, lam = m.groups()
                rows.append(dict(arm=arm, dataset=ds, family=fam, seed=int(seed), trial=int(t), cv=float(cv),
                                 mcw=float(mcw), lam=float(lam)))
tr = pd.DataFrame(rows).drop_duplicates(["arm", "dataset", "family", "seed", "trial"], keep="last")
tr["cv"] = tr.cv.where(np.isfinite(tr.cv), np.inf)
key = ["family", "dataset", "seed"]
d0 = tr[(tr.arm == "dtpe") & (tr.trial == 0)].set_index(key).cv.rename("default_cv")

picks = []
for k in KS:
    t = tr[tr.trial < k]
    pk = t.loc[t.groupby(["arm"] + key).cv.idxmin()].assign(k=k)
    picks.append(pk)
pk = pd.concat(picks).rename(columns={"cv": "pick_cv"})

df = pd.read_csv(csv_path)
df = df[df.error.isna() | (df.error == "")]
base = df[df.tag == "c12_off"].drop_duplicates(["dataset", "family", "seed"], keep="last").set_index(key)
parts = []
for arm in arms:
    pre = "c13_raw" if arm == "tpe" else f"c14_{arm}"
    for k in KS:
        t = df[df.tag == f"{pre}_k{k}"].drop_duplicates(["dataset", "family", "seed"], keep="last")
        parts.append(t.assign(arm=arm, k=k)[["arm", "k"] + key + ["nll", "valid_nll", "best_iter"] if "best_iter" in t else ["arm", "k"] + key + ["nll", "valid_nll"]])
T = pd.concat(parts)
J = pk.merge(T, on=["arm", "k"] + key).join(d0, on=key)
J["cv_gain"] = J.pick_cv - J.default_cv
J["val_gain"] = J.valid_nll - base.valid_nll.reindex(pd.MultiIndex.from_frame(J[key])).values
J["test_gain"] = J.nll - base.nll.reindex(pd.MultiIndex.from_frame(J[key])).values
J["is_default"] = (J.arm.isin(["dtpe", "ds10"])) & (J.trial == 0)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def se(s):
    return s.std(ddof=1) / math.sqrt(len(s))


print(f"refits joined: {len(J)} (of {len(T)}); picks equal to the default: {int(J.is_default.sum())}")
print()
print("ALL ARMS AND BUDGETS POOLED, per set — gains over the untuned default (negative = better)")
out = []
for (fam, ds), g in J.groupby(["family", "dataset"]):
    h = g[~g.is_default]
    # within-seed correlation: remove seed means first
    c = h[["cv_gain", "test_gain"]] - h.groupby("seed")[["cv_gain", "test_gain"]].transform("mean")
    out.append(dict(family=fam, dataset=ds, n=len(h), cv_gain=h.cv_gain.mean(), val_gain=h.val_gain.mean(),
                    test_gain=h.test_gain.mean(), test_se=se(h.groupby("seed").test_gain.mean()),
                    kept=h.test_gain.mean() / h.cv_gain.mean() if h.cv_gain.mean() != 0 else np.nan,
                    corr_cv_test=h.cv_gain.corr(h.test_gain), corr_within_seed=c.cv_gain.corr(c.test_gain),
                    corr_val_test=h.val_gain.corr(h.test_gain)))
print(pd.DataFrame(out).round(4).to_string(index=False))
print()

print("BY BUDGET, mean over sets and arms")
print(J[~J.is_default].groupby("k")[["cv_gain", "val_gain", "test_gain"]].mean().round(4).to_string())
print()

print("WHERE THE PICKS ARE — log10 mcw / log10 lambda of the k = 20 pick, all arms (median, min, max), and the test gain by region")
J["lm"] = np.log10(J.mcw)
J["ll"] = np.log10(J.lam)
g20 = J[(J.k == 20) & ~J.is_default]
print(g20.groupby(["family", "dataset"]).agg(lm_med=("lm", "median"), lm_min=("lm", "min"), lm_max=("lm", "max"),
                                             ll_med=("ll", "median"), ll_min=("ll", "min"), ll_max=("ll", "max")).round(2).to_string())
print()
for fam, ds in [("poisson", "bike_cnt"), ("gaussian", "naval"), ("gaussian", "protein"), ("gamma_ms", "naval")]:
    h = J[(J.family == fam) & (J.dataset == ds) & ~J.is_default].copy()
    if h.empty:
        continue
    h["mcw_bin"] = pd.cut(h.lm, [-3.01, -1, 1, 3.01], labels=["mcw<0.1", "0.1-10", "mcw>10"])
    h["lam_bin"] = pd.cut(h.ll, [-3.01, -1, 1, 3.01], labels=["lam<0.1", "0.1-10", "lam>10"])
    t = h.pivot_table(index="mcw_bin", columns="lam_bin", values="test_gain", aggfunc=["mean", "count"], observed=False)
    print(f"-- {fam} {ds}: test gain of the refit by region of the pick (all arms, all k)")
    print(t.round(3).to_string())
    print()

print("PER-SEED rows for the two losing sets, tpe arm, k = 20")
h = J[(J.arm == "tpe") & (J.k == 20) & (((J.family == "poisson") & (J.dataset == "bike_cnt")) | ((J.family == "gaussian") & (J.dataset == "naval")))]
print(h[key + ["trial", "mcw", "lam", "default_cv", "pick_cv", "cv_gain", "val_gain", "test_gain"]].round(4).to_string(index=False))
print()

# P4
best = tr[tr.trial < 20].groupby(["arm"] + key).cv.min().unstack(0)
d = (best["ds10"] - best["dtpe"]).groupby(level=[0, 1]).mean()
print("P4 — best CV at k = 20, ds10 − dtpe, mean over seeds")
print(d.round(4).to_string())
print(f"   |diff| <= 0.005 on {int((d.abs() <= 0.005).sum())}/{len(d)}")
# exact values behind P3
p = T[T.k == 20].pivot_table(index=key, columns="arm", values="nll")
d3 = (p["dtpe"] - p["tpe"]).groupby(level=[0, 1]).mean()
print("P3 — test NLL at k = 20, dtpe − tpe")
print(d3.round(5).to_string())

# ---------------------------------------------------------------- post hoc: a validation guard
# guard:  keep the k-trial pick only if its refit beats the untuned default on the validation split.
# best-of: the lowest validation NLL among the default and the refits at k = 5, 10, 20.
print()
print("POST HOC (not pre-registered) — validation guard; test NLL − untuned default, mean over seeds")
out = []
for arm in arms:
    A = J[J.arm == arm]
    g20 = A[A.k == 20].set_index(key)
    guard = g20.test_gain.where(g20.val_gain < 0, 0.0)
    idx = A.groupby(key).val_gain.idxmin()
    bo = A.loc[idx].set_index(key)
    bestof = bo.test_gain.where(bo.val_gain < 0, 0.0)
    t = pd.DataFrame(dict(plain=g20.test_gain, guard=guard, bestof=bestof.reindex(g20.index))).groupby(level=[0, 1]).mean()
    kept = (g20.val_gain < 0).groupby(level=[0, 1]).sum()
    for (fam, ds), r in t.iterrows():
        out.append(dict(arm=arm, family=fam, dataset=ds, plain=r.plain, guard=r.guard, bestof=r.bestof, picks_kept=int(kept[(fam, ds)])))
G = pd.DataFrame(out)
print(G.pivot_table(index=["family", "dataset"], columns="arm", values=["plain", "guard", "bestof"]).round(4).to_string())
print()
for arm in arms:
    g = G[G.arm == arm]
    print(f"  {arm}: sets worse than the default by > 0.005 — plain {int((g.plain > 0.005).sum())}/10, guard {int((g.guard > 0.005).sum())}/10, "
          f"best-of {int((g.bestof > 0.005).sum())}/10; mean over sets — plain {g.plain.mean():+.4f}, guard {g.guard.mean():+.4f}, best-of {g.bestof.mean():+.4f}; "
          f"worst set — plain {g.plain.max():+.4f}, guard {g.guard.max():+.4f}, best-of {g.bestof.max():+.4f}")
# paired, per set: guard − plain over all arms (20 searches per set)
A = J[J.k == 20].copy()
A["guard"] = A.test_gain.where(A.val_gain < 0, 0.0)
d = (A.guard - A.test_gain).groupby([A.family, A.dataset]).agg(["mean", lambda s: s.std(ddof=1) / math.sqrt(len(s)), lambda s: int((s != 0).sum())])
d.columns = ["guard_minus_plain", "se", "searches_changed_of_20"]
print()
print(d.round(4).to_string())
