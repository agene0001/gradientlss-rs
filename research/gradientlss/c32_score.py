"""Proposal 036: LightGBM arms against the LightGBM two-axis search (G0-G3), and l_def against XGBoost untuned.

Usage: uv run --with pandas --with numpy python -I c32_score.py <experiments_v2.csv>[,/tmp/c32] /tmp/c32 [arm ...]
"""
import sys

import pandas as pd

want = sys.argv[3:]
sys.argv = sys.argv[:3] + ["base=c32_l_def"]
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import c30_summary as S  # noqa: E402

base, which = S.base, S.which

# l_def against the XGBoost untuned default (reported, not a test)
xgb = S.first("c12_off", "c19_def", "c17_new", "c27_def").reindex(base.index)
d = (base.nll - xgb.nll).dropna().groupby(level=[0, 1]).mean()
print(f"l_def minus XGBoost untuned: {len(base)} cells, {len(d)} pairs; LightGBM better by > 0.005 on {int((d < -0.005).sum())}, worse on {int((d > 0.005).sum())}, within on {int((d.abs() <= 0.005).sum())}; mean {d.mean():+.4f}, median {d.median():+.4f}")
print("   largest:", {f"{f}/{ds}": round(v, 3) for (f, ds), v in pd.concat([d.sort_values().head(5), d.sort_values().tail(5)]).items()})
print(f"   best_iter median LightGBM {base.best_iter.median():.0f}, XGBoost {xgb.best_iter.median():.0f}; fit seconds mean {base.fit_secs.mean():.1f} / {xgb.fit_secs.mean():.1f}; non-finite cells {int(base.nll.isna().sum())}")

# G0: the two axes against l_def
for k in (20, 10):
    t = S.arm("c32_l_2d", k)
    g = (t.nll - base.nll.reindex(t.index)).dropna()
    p = g.groupby(level=[0, 1]).mean()
    gg = g.where(t.valid_nll.reindex(g.index) < base.valid_nll.reindex(g.index), 0.0).groupby(level=[0, 1]).mean()
    print(f"G0 l_2d k = {k}: {len(g)} cells, {len(p)} pairs; plain mean {p.mean():+.4f}, median {p.median():+.4f}, worse than l_def by > 0.005 on {int((p > 0.005).sum())}, better on {int((p < -0.005).sum())}"
          f" | guarded mean {gg.mean():+.4f}, median {gg.median():+.4f}, worse on {int((gg > 0.005).sum())} {gg[gg > 0.005].round(4).to_dict()}")
    if k == 20:
        print("   plain worse:", p[p > 0.005].round(4).to_dict())

KEYS = {"lleaves": "num_leaves", "lmdl": "min_data_in_leaf", "lff": "feature_fraction", "lbag": "bagging_fraction"}
med = {}
for a in want or list(KEYS):
    P = S.rule(f"c32_{a}", "c32_l_2d", 20, show=False)
    if P is None:
        continue
    med[a] = P
    S.rule(f"c32_{a}", "c32_l_2d", 10, show=False)
    pk = S.picks(a, 20)
    if pk is not None and KEYS[a] in pk[1]:
        tr, best = pk
        g = best.groupby(["family", "dataset"])[KEYS[a]].agg(["min", "median", "max"])
        print(f"   picked {KEYS[a]} (min / median / max over seeds) by pair:")
        print("   " + "; ".join(f"{f}/{d} {r['min']:.3g}/{r['median']:.3g}/{r['max']:.3g}" for (f, d), r in g.iterrows()))
        print(f"   over all searches: quartiles {best[KEYS[a]].quantile(.25):.3g}, {best[KEYS[a]].median():.3g}, {best[KEYS[a]].quantile(.75):.3g}")
        ta = S.arm(f"c32_{a}", 20)
        v = (ta.valid_nll - S.arm("c32_l_2d", 20).valid_nll.reindex(ta.index)).dropna().groupby(level=[0, 1]).mean()
        print(f"   refit validation NLL lower than two axes on {int((v < 0).sum())}/{len(v)} pairs")
if med:
    print("\nsummary (20 trials): arm, pairs, plain mean, guarded mean, guarded median, (a), (b), (c), plain better/worse than l_2d")
    for a, P in med.items():
        print(f"   {a:7s} {len(P):2d}  {P.plain.mean():+.4f}  {P.guard.mean():+.4f}  {P.guard.median():+.4f}  (a) {int((P.guard > 0.005).sum())}  (b) {int((P.d_guard < 0).sum())}  (c) {P.d_guard.median():+.4f}"
              f"  {int((P.d_plain < -0.005).sum())}/{int((P.d_plain > 0.005).sum())}   [l_2d: plain {P.ref_plain.mean():+.4f}, guarded {P.ref_guard.mean():+.4f}, median {P.ref_guard.median():+.4f}]")
