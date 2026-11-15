"""Proposal 039: the leaf-wise axis on the c_m_* batch (predictions F0-F6).

Usage: uv run --with pandas --with numpy python -I c35_score.py <experiments_v2.csv>[,/tmp/c35] /tmp/c35
"""
import sys

import pandas as pd

sys.argv = sys.argv[:3] + ["base=c34_def", "pairs=12"]
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import c30_summary as S  # noqa: E402

base, dd = S.base, S.dd

# F0
S.identical(dd("c35_u_repro"), base, "F0 c35_u_repro vs c34_def")

# F1-F3: the rule at 20 trials (and at 10 against the 20-trial reference, descriptively)
P = S.rule("c35_leaf", "c34_2d", 20)
S.rule("c35_leaf", "c34_2d", 10, show=False, rk=20)

# F4: untuned lossguide-16 against the default
u = dd("c35_u_lg16")
d = (u.nll - base.nll.reindex(u.index)).dropna().groupby(level=[0, 1]).mean()
print(f"\nF4 u_lg16 minus default: better by > 0.005 on {int((d < -0.005).sum())}/{len(d)}, worse on {int((d > 0.005).sum())}/{len(d)}, mean {d.mean():+.4f}, median {d.median():+.4f}")
print(d.round(4).to_string())

# F5: plain leaf against untuned lossguide-16
if P is not None:
    e = (P.plain - d.reindex(P.index)).dropna()
    print(f"\nF5 plain leaf minus u_lg16: better by > 0.005 on {int((e < -0.005).sum())}/{len(e)}, worse on {int((e > 0.005).sum())}/{len(e)}, median {e.median():+.4f}")
    print(pd.DataFrame({"leaf_plain": P.plain, "leaf_guard": P.guard, "u_lg16": d.reindex(P.index), "two_axes_guard": P.ref_guard}).round(4).to_string())

# F6: picks
pk = S.picks("leaf", 20)
if pk is not None:
    tr, best = pk
    g = best.groupby(["family", "dataset"]).max_leaves.agg(["min", "median", "max", "count"])
    print("\nF6 picked max_leaves by pair")
    print(g.to_string())
    print(f"   median <= 8 on {int((g['median'] <= 8).sum())}/{len(g)}, >= 32 on {int((g['median'] >= 32).sum())}/{len(g)}; quartiles over searches {best.max_leaves.quantile([.25, .5, .75]).tolist()}")

# Tier 1
t, r0 = S.arm("c35_leaf", 20), S.arm("c34_2d", 20)
v = (t.valid_nll - r0.valid_nll.reindex(t.index)).dropna().groupby(level=[0, 1]).mean()
print(f"\nvalidation NLL of the refit, leaf - two axes at k = 20: lower on {int((v < 0).sum())}/{len(v)} pairs")
