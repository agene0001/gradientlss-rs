"""Proposal 022: validation guard for tuned models, on the unseen sets (and the 5-fold mechanism arm).

Usage: uv run --with pandas --with numpy python -I c19_guard_summary.py <experiments_v2.csv> \
           tpe=<log>[,<log>...] dtpe=<log>[,...] [tpe5=<log>[,...]]

plain  = test NLL of the k-trial refit − the untuned default (`c19_def`; `c12_off` for tpe5's sets)
guard  = plain where the refit's validation NLL beats the default's, else 0
bestof = the lowest validation NLL among the default and the refits at k = 5, 10, 20
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
key = ["family", "dataset", "seed"]
rows = []
for arm, paths in arms.items():
    for p in paths.split(","):
        for line in open(p):
            m = PAT.match(line)
            if m:
                ds, fam, seed, t, cv, mcw, lam = m.groups()
                rows.append(dict(arm=arm, dataset=ds, family=fam, seed=int(seed), trial=int(t), cv=float(cv),
                                 mcw=float(mcw), lam=float(lam)))
tr = pd.DataFrame(rows).drop_duplicates(["arm"] + key + ["trial"], keep="last")
tr["cv"] = tr.cv.where(np.isfinite(tr.cv), np.inf)
print("complete 20-trial searches per arm:",
      {a: int(g.groupby(key).trial.max().eq(19).sum()) for a, g in tr.groupby("arm")})

df = pd.read_csv(csv_path)
df = df[df.error.isna() | (df.error == "")]
dedup = lambda t: t.drop_duplicates(["dataset", "family", "seed"], keep="last")
base = pd.concat([dedup(df[df.tag == "c19_def"]), dedup(df[df.tag == "c17_new"]), dedup(df[df.tag == "c12_off"])])
base = base.drop_duplicates(["dataset", "family", "seed"], keep="first").set_index(key)
parts = []
for arm in arms:
    for k in KS:
        parts.append(dedup(df[df.tag == f"c19_{arm}_k{k}"]).assign(arm=arm, k=k))
T = pd.concat(parts)
ix = pd.MultiIndex.from_frame(T[key])
T["val_gain"] = T.valid_nll.values - base.valid_nll.reindex(ix).values
T["test_gain"] = T.nll.values - base.nll.reindex(ix).values
T["bad"] = T.nll.isna() | (T.nonfinite_rows > 0)
print("refit rows:", T.groupby("arm").size().to_dict(), " non-finite refits:", int(T.bad.sum()),
      " rows without a default:", int(T.test_gain.isna().sum() - T.nll.isna().sum()))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def se(s):
    s = s.dropna()
    return s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")


def table(arm):
    A = T[T.arm == arm]
    g = A[A.k == 20].set_index(key)
    keep = g.val_gain < 0
    guard = g.test_gain.where(keep, 0.0)
    bo = A.loc[A.groupby(key).val_gain.idxmin()].set_index(key)
    bestof = bo.test_gain.where(bo.val_gain < 0, 0.0).reindex(g.index)
    out = []
    for (fam, ds), i in g.groupby(level=[0, 1]).groups.items():
        p, gd, b = g.test_gain[i], guard[i], bestof[i]
        out.append(dict(family=fam, dataset=ds, default_nll=base.nll.reindex(i).mean(),
                        plain=p.mean(), plain_se=se(p), plain_better=f"{int((p < 0).sum())}/{len(p)}",
                        guard=gd.mean(), guard_se=se(gd), picks_kept=f"{int(keep[i].sum())}/{len(i)}",
                        bestof=b.mean(), val_plain=g.val_gain[i].mean()))
    return pd.DataFrame(out)


for arm in arms:
    t = table(arm)
    if t.empty:
        continue
    print()
    print(f"== {arm}: test NLL − untuned default at 20 trials, mean over seeds (negative = tuning helps)")
    print(t.round(4).to_string(index=False))
    n = len(t)
    print(f"   worse than the default by > 0.005: plain {int((t.plain > 0.005).sum())}/{n}"
          + ("" if not (t.plain > 0.005).any() else " [" + ", ".join(f"{r.family}/{r.dataset} {r.plain:+.4f}" for r in t[t.plain > 0.005].itertuples()) + "]")
          + f", guard {int((t.guard > 0.005).sum())}/{n}"
          + ("" if not (t.guard > 0.005).any() else " [" + ", ".join(f"{r.family}/{r.dataset} {r.guard:+.4f}" for r in t[t.guard > 0.005].itertuples()) + "]")
          + f", best-of {int((t.bestof > 0.005).sum())}/{n}")
    print(f"   mean over pairs: plain {t.plain.mean():+.4f}, guard {t.guard.mean():+.4f}, best-of {t.bestof.mean():+.4f};"
          f"  median: plain {t.plain.median():+.4f}, guard {t.guard.median():+.4f}")

# G4: how much of the CV gain survives (dtpe: trial 0 is the default)
if "dtpe" in arms:
    d0 = tr[(tr.arm == "dtpe") & (tr.trial == 0)].set_index(key)
    assert ((d0.mcw - 1).abs() < 1e-9).all() and ((d0.lam - 1).abs() < 1e-9).all()
    pk = []
    for k in KS:
        t = tr[(tr.arm == "dtpe") & (tr.trial < k)]
        pk.append(t.loc[t.groupby(key).cv.idxmin()].assign(k=k))
    pk = pd.concat(pk)
    J = pk.merge(T[T.arm == "dtpe"], on=key + ["k"]).join(d0.cv.rename("default_cv"), on=key)
    J["cv_gain"] = J.cv - J.default_cv
    J = J[J.trial > 0]
    print()
    print("== G4 — dtpe: what the search believed (CV gain) against what the refit kept, picks other than the default")
    g = J.groupby(["family", "dataset"]).agg(n=("cv_gain", "size"), cv_gain=("cv_gain", "mean"), val_gain=("val_gain", "mean"),
                                             test_gain=("test_gain", "mean"))
    g["corr_cv_test"] = J.groupby(["family", "dataset"]).apply(lambda h: h.cv_gain.corr(h.test_gain), include_groups=False)
    g["corr_val_test"] = J.groupby(["family", "dataset"]).apply(lambda h: h.val_gain.corr(h.test_gain), include_groups=False)
    print(g.round(4).to_string())
    print(f"   pooled over pairs and k: CV gain {g.cv_gain.mean():+.4f}, validation {g.val_gain.mean():+.4f}, test {g.test_gain.mean():+.4f}"
          f"  -> kept {g.test_gain.mean() / g.cv_gain.mean():.2f} (medians: CV {g.cv_gain.median():+.4f}, test {g.test_gain.median():+.4f})")
