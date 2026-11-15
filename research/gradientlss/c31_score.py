"""Proposal 035: untuned arms (S0, E0, E1, L0), the rule per search arm, and the picked values.

Usage: uv run --with pandas --with numpy python -I c31_score.py <experiments_v2.csv>[,<dir>] <log dir> [arm ...]
"""
import sys

import numpy as np
import pandas as pd

want = sys.argv[3:]
sys.argv = sys.argv[:3]
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import c30_summary as S  # noqa: E402

base, dd, which = S.base, S.dd, S.which

t = dd("c31_u_repro")
if len(t):
    S.identical(t, base, "S0 u_repro vs untuned default")
U = {}
for a, label in (("u_cap6000", "round cap 6000"), ("u_eta015", "eta 0.15"), ("u_lg16", "lossguide, 16 leaves")):
    t = dd(f"c31_{a}")
    if len(t) == 0:
        continue
    b = base.reindex(t.index)
    d = (t.nll - b.nll).dropna().groupby(level=[0, 1]).mean()
    U[a] = d
    n = len(d)
    line = f"{a} ({label}) minus default: better by > 0.005 on {int((d < -0.005).sum())}/{n}, worse on {int((d > 0.005).sum())}/{n}, within on {int((d.abs() <= 0.005).sum())}/{n}; mean {d.mean():+.4f}, median {d.median():+.4f}"
    for s in ("dev", "unseen", "fresh"):
        g = d[[which(ds) == s for _, ds in d.index]]
        line += f" | {s} {int((g < -0.005).sum())}/{int((g > 0.005).sum())}/{len(g)}"
    print(line)
    hit = (b.best_iter >= 1990).groupby(level=[0, 1]).mean()
    print(f"    default at the 2000-round cap (best_iter >= 1990) in at least one seed on {int((hit > 0).sum())} pairs; "
          f"best_iter median default {b.best_iter.median():.0f}, arm {t.best_iter.median():.0f}; fit seconds mean default {b.fit_secs.mean():.1f}, arm {t.fit_secs.mean():.1f}")
if U:
    T = pd.DataFrame(U)
    T.insert(0, "default_best_iter", base.best_iter.groupby(level=[0, 1]).median().reindex(T.index))
    T.insert(0, "set", [which(ds) for _, ds in T.index])
    print(T.sort_values("set").round(4).to_string())

KEYS = {"eta": "eta", "gamma": "gamma", "alpha": "alpha", "ctree": "colsample_bytree", "clevel": "colsample_bylevel", "cnode": "colsample_bynode", "mds": "max_delta_step", "leaf": "max_leaves"}
med = {}
for a in want or list(KEYS):
    P = S.rule(f"c31_{a}", "2d", 20, show=False)
    if P is None:
        continue
    med[a] = P
    S.rule(f"c31_{a}", "2d", 10, show=False)
    pk = S.picks(a, 20)
    if pk is not None and KEYS[a] in pk[1]:
        tr, best = pk
        g = best.groupby(["family", "dataset"])[KEYS[a]].agg(["min", "median", "max"])
        print(f"   picked {KEYS[a]} (min / median / max over seeds) by pair:")
        print("   " + "; ".join(f"{f}/{d} {r['min']:.3g}/{r['median']:.3g}/{r['max']:.3g}" for (f, d), r in g.iterrows()))
        print(f"   over all searches: quartiles {best[KEYS[a]].quantile(.25):.3g}, {best[KEYS[a]].median():.3g}, {best[KEYS[a]].quantile(.75):.3g}")
        v = (S.arm(f"c31_{a}", 20).valid_nll - S.arm("2d", 20).valid_nll.reindex(S.arm(f"c31_{a}", 20).index)).dropna().groupby(level=[0, 1]).mean()
        print(f"   refit validation NLL lower than two axes on {int((v < 0).sum())}/{len(v)} pairs")
if med:
    print("\nsummary (20 trials): arm, pairs, plain mean, guarded mean, guarded median, (a), (b), (c), plain better/worse than two axes")
    for a, P in med.items():
        print(f"   {a:7s} {len(P):2d}  {P.plain.mean():+.4f}  {P.guard.mean():+.4f}  {P.guard.median():+.4f}  (a) {int((P.guard > 0.005).sum())}  (b) {int((P.d_guard < 0).sum())}  (c) {P.d_guard.median():+.4f}"
              f"  {int((P.d_plain < -0.005).sum())}/{int((P.d_plain > 0.005).sum())}   [two axes: plain {P.ref_plain.mean():+.4f}, guarded {P.ref_guard.mean():+.4f}, median {P.ref_guard.median():+.4f}]")
