"""Proposal 017: sampler control and defaults-first. Tier 1 (CV best-so-far) and Tier 2 (test NLL).

Usage: uv run --with pandas --with numpy python -I c14_sampler_summary.py <experiments_v2.csv> \
           tpe=<log>[,<log>] rand=<log>[,..] dtpe=<log>[,..] ds10=<log>[,..]

`tpe` is cycle 13's raw arm (its logs hold both arms' runs back to back only if concatenated; pass the
`c13_off_*` logs). Test rows: `c13_raw_k<k>` for tpe, `c14_<arm>_k<k>` otherwise; untuned = `c12_off`.
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
tr = tr.sort_values(["arm", "family", "dataset", "seed", "trial"])
tr["best"] = tr.groupby(["arm", "family", "dataset", "seed"]).cv.cummin()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
cells = tr.groupby("arm").apply(lambda g: g.groupby(["family", "dataset", "seed"]).trial.max().eq(19).sum(), include_groups=False)
print("complete 20-trial searches per arm:", dict(cells))
print()


def pm(d):
    d = d.dropna()
    if len(d) < 2:
        return "n/a"
    return f"{d.mean():+.4f} ± {d.std(ddof=1) / math.sqrt(len(d)):.4f} ({int((d < 0).sum())}/{len(d)})"


# ---------------------------------------------------------------- Tier 1: best CV score after k trials
best = {k: tr[tr.trial == k - 1].pivot_table(index=["family", "dataset", "seed"], columns="arm", values="best") for k in KS}
print("TIER 1 — best CV score after k trials, arm − tpe, mean ± s.e. over seeds (seeds where the arm is better)")
for other in [a for a in arms if a != "tpe"]:
    out = []
    for (fam, ds), _ in best[20].groupby(level=[0, 1]):
        row = dict(family=fam, dataset=ds)
        for k in KS:
            b = best[k].loc[(fam, ds)]
            if other in b and "tpe" in b:
                row[f"k{k}"] = pm(b[other] - b["tpe"])
        out.append(row)
    t = pd.DataFrame(out)
    print(f"-- {other} − tpe")
    print(t.to_string(index=False))
    for k in KS:
        d = (best[k][other] - best[k]["tpe"]).groupby(level=[0, 1]).mean()
        print(f"   k={k}: {other} better on {int((d < 0).sum())}/{len(d)} sets; |diff| <= 0.005 on {int((d.abs() <= 0.005).sum())}/{len(d)}; "
              f"median diff {d.median():+.4f}")
    print()

# ---------------------------------------------------------------- Tier 2: test NLL of the refits
df = pd.read_csv(csv_path)
df = df[df.error.isna() | (df.error == "")]
parts = []
for arm in arms:
    pre = "c13_raw" if arm == "tpe" else f"c14_{arm}"
    for k in KS:
        t = df[df.tag == f"{pre}_k{k}"].drop_duplicates(["dataset", "family", "seed"], keep="last")
        parts.append(t.assign(arm=arm, k=k))
T = pd.concat(parts)
base = df[df.tag == "c12_off"].drop_duplicates(["dataset", "family", "seed"], keep="last").set_index(["family", "dataset", "seed"])
print("TIER 2 — test NLL of the refit at k trials")
for k in KS:
    p = T[T.k == k].pivot_table(index=["family", "dataset", "seed"], columns="arm", values="nll")
    v = T[T.k == k].pivot_table(index=["family", "dataset", "seed"], columns="arm", values="valid_nll")
    out = []
    for (fam, ds), g in p.groupby(level=[0, 1]):
        row = dict(family=fam, dataset=ds)
        for other in [a for a in arms if a != "tpe" and a in g]:
            row[f"{other}-tpe"] = pm(g[other] - g["tpe"])
        b = base.nll.reindex(g.index)
        for a in arms:
            if a in g:
                row[f"{a}-untuned"] = f"{(g[a] - b).mean():+.4f}"
        out.append(row)
    print(f"-- k = {k}  (arm − tpe: mean ± s.e. (seeds better);  arm − untuned default: mean)")
    print(pd.DataFrame(out).to_string(index=False))
    for a in arms:
        if a in p:
            d = (p[a] - base.nll.reindex(p.index)).groupby(level=[0, 1]).mean()
            print(f"   {a}: worse than the untuned default by > 0.005 on {int((d > 0.005).sum())}/{len(d)} sets; mean over sets {d.mean():+.4f}")
    print()

# ---------------------------------------------------------------- diagnosis: the default's own CV score
if "dtpe" in arms:
    d0 = tr[(tr.arm == "dtpe") & (tr.trial == 0)].set_index(["family", "dataset", "seed"])
    assert ((d0.mcw - 1).abs() < 1e-9).all() and ((d0.lam - 1).abs() < 1e-9).all(), "dtpe trial 0 is not the default"
    tp = tr[tr.arm == "tpe"]
    pick = tp.loc[tp.groupby(["family", "dataset", "seed"]).cv.idxmin()].set_index(["family", "dataset", "seed"])
    rank = tp.join(d0.cv.rename("dcv"), on=["family", "dataset", "seed"])
    rank = (rank.cv < rank.dcv).groupby([rank.family, rank.dataset, rank.seed]).sum()
    j = pd.DataFrame(dict(default_cv=d0.cv, pick_cv=pick.cv, trials_beating_default=rank)).dropna()
    j["default_better"] = j.default_cv < j.pick_cv
    k20 = T[(T.k == 20) & (T.arm == "tpe")].set_index(["family", "dataset", "seed"])
    j["test_pick_minus_default"] = k20.nll.reindex(j.index) - base.nll.reindex(j.index)
    j["valid_pick_minus_default"] = k20.valid_nll.reindex(j.index) - base.valid_nll.reindex(j.index)
    g = j.groupby(level=[0, 1]).agg(
        default_cv=("default_cv", "mean"), pick_cv=("pick_cv", "mean"),
        cv_pick_minus_default=("pick_cv", lambda s: (s - j.default_cv.reindex(s.index)).mean()),
        default_cv_better_seeds=("default_better", "sum"), seeds=("default_better", "count"),
        tpe_trials_beating_default=("trials_beating_default", "mean"),
        valid_pick_minus_default=("valid_pick_minus_default", "mean"),
        test_pick_minus_default=("test_pick_minus_default", "mean"),
    ).round(4)
    print("DIAGNOSIS — the untuned default's CV score on the search's own folds, against tpe's 20-trial pick")
    print("  (cv/valid/test pick − default: negative = the pick is better)")
    print(g.to_string())
    agree = np.sign(g.cv_pick_minus_default) == np.sign(g.test_pick_minus_default)
    print(f"\n  CV and test agree on the sign of pick − default on {int(agree.sum())}/{len(g)} sets")
