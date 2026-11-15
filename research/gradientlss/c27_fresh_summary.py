"""Proposal 032: the subsampling axes on fresh datasets. Tags c27_def, c27_2d_k<k>, c27_subs_k<k>.

Usage: uv run --with pandas --with numpy python -I c27_fresh_summary.py <experiments_v2.csv>

guard = keep the refit only if its validation NLL beats the untuned default's.
"""
import math
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
df = df[df.tag.str.startswith("c27_", na=False) & (df.error.isna() | (df.error == ""))]
key = ["family", "dataset", "seed"]
dd = lambda t: df[df.tag == t].drop_duplicates(key, keep="last").set_index(key)
base = dd("c27_def")
pd.set_option("display.width", 250)
print("cells:", df.groupby("tag").size().to_dict(), " non-finite rows:", int(df.nonfinite_rows.sum()))


def se(s):
    return s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")


def pm(d):
    d = d.dropna()
    return f"{d.mean():+.4f} ± {se(d):.4f} ({int((d < 0).sum())}/{len(d)})"


for k in (10, 20):
    arms = {a: dd(f"c27_{a}_k{k}") for a in ("2d", "subs")}
    if any(len(v) == 0 for v in arms.values()):
        continue
    G = {}
    for a, t in arms.items():
        gain = t.nll - base.nll.reindex(t.index)
        vgain = t.valid_nll - base.valid_nll.reindex(t.index)
        G[a] = pd.DataFrame({"plain": gain, "guard": gain.where(vgain < 0, 0.0), "kept": vgain < 0})
    out = []
    for (fam, ds), _ in G["subs"].groupby(level=[0, 1]):
        s2, ss = G["2d"].loc[(fam, ds)], G["subs"].loc[(fam, ds)]
        out.append(dict(family=fam, dataset=ds, seeds=len(ss), default_nll=base.nll.loc[(fam, ds)].mean(), plain_2d=s2.plain.mean(), plain_subs=ss.plain.mean(),
                        guard_2d=s2.guard.mean(), guard_subs=ss.guard.mean(), kept_subs=int(ss.kept.sum()),
                        subs_minus_2d=pm(ss.plain - s2.plain), guard_subs_minus_2d=pm(ss.guard - s2.guard)))
    t = pd.DataFrame(out)
    n = len(t)
    print()
    print(f"k = {k} — test NLL − untuned default, mean over seeds")
    print(t.round(4).to_string(index=False))
    print(f"   mean over pairs: plain 2d {t.plain_2d.mean():+.4f}, plain subs {t.plain_subs.mean():+.4f}, guard 2d {t.guard_2d.mean():+.4f}, guard subs {t.guard_subs.mean():+.4f}; "
          f"median: plain 2d {t.plain_2d.median():+.4f}, plain subs {t.plain_subs.median():+.4f}, guard 2d {t.guard_2d.median():+.4f}, guard subs {t.guard_subs.median():+.4f}")
    print(f"   worse than default by > 0.005: plain 2d {int((t.plain_2d > 0.005).sum())}/{n}, plain subs {int((t.plain_subs > 0.005).sum())}/{n}, "
          f"guard 2d {int((t.guard_2d > 0.005).sum())}/{n}, guard subs {int((t.guard_subs > 0.005).sum())}/{n}")
    d = t.plain_subs - t.plain_2d
    dg = t.guard_subs - t.guard_2d
    print(f"   C2: plain subs − plain 2d ≤ +0.005 on {int((d <= 0.005).sum())}/{n} (largest {d.max():+.4f})")
    print(f"   C3: guarded subs better than guarded 2d on {int((dg < 0).sum())}/{n}; ratio of means {t.guard_subs.mean() / t.guard_2d.mean():.2f}")
