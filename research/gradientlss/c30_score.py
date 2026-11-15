"""Proposal 034: predictions R0, B1-B4, C1-C4 (the rule tables come from c30_summary.py).

Usage: uv run --with pandas --with numpy python -I c30_score.py <experiments_v2.csv>[,<dir>] <log dir>
"""
import sys

import numpy as np
import pandas as pd

sys.argv = sys.argv[:3]
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import c30_summary as S  # noqa: E402

arm, base, rule = S.arm, S.base, S.rule

# R0
S.identical(arm("c30_2d", 20), arm("2d", 20), "c30_2d_k20 vs stored two axes k20")
S.identical(arm("c30_bin", 20), arm("c29_bin", 20), "c30_bin_k20 vs c29_bin_k20")


def plain(stem, k):
    t = arm(stem, k)
    return (t.nll - base.nll.reindex(t.index)).dropna().groupby(level=[0, 1]).mean()


# B1
p20, p50 = plain("c30_2d", 20), plain("c30_2d", 50)
print(f"\nB1: two axes plain mean gain k20 {p20.mean():+.4f} ({len(p20)} pairs), k50 {p50.mean():+.4f} ({len(p50)} pairs), move {p50.mean() - p20.mean():+.4f}; "
      f"median {p20.median():+.4f} -> {p50.median():+.4f}; worse than default > 0.005: k20 {int((p20 > 0.005).sum())}, k50 {int((p50 > 0.005).sum())}")
d = (p50 - p20).sort_values()
print("    largest moves k50 - k20:", {f"{f}/{ds}": round(v, 4) for (f, ds), v in pd.concat([d.head(4), d.tail(4)]).items()})

# B2-B4
P = rule("c30_bin", "c30_2d", 50, show=False)
if P is not None:
    for (f, ds) in [("gaussian", "c_n_sgemm"), ("gaussian", "kin8nm"), ("gamma_ms", "kin8nm"), ("gamma", "kin8nm"), ("gamma_ms", "c_parkinsons"), ("gamma", "c_parkinsons")]:
        if (f, ds) in P.index:
            r = P.loc[(f, ds)]
            print(f"   B3/B4 {f} {ds}: plain bin50 - 2d50 = {r.d_plain:+.4f} {r.paired}; guarded {r.d_guard:+.4f}")
    P20 = rule("c30_bin", "c30_2d", 20, show=False)

# C1-C4
Q = rule("c30_bin2", "2d", 20, show=False)
if Q is not None:
    for (f, ds) in [("gaussian", "kin8nm"), ("gamma_ms", "kin8nm"), ("gamma", "kin8nm"), ("gamma_ms", "c_parkinsons"), ("gamma", "c_parkinsons"), ("gaussian", "c_n_sgemm")]:
        if (f, ds) in Q.index:
            r = Q.loc[(f, ds)]
            print(f"   C2/C3 {f} {ds}: plain bin2 - two axes = {r.d_plain:+.4f} {r.paired}; guarded {r.d_guard:+.4f}")
    rule("c30_bin2", "2d", 10, show=False)

pk = S.picks("bin2", 20)
if pk is not None:
    tr, best = pk
    g = best.groupby(["family", "dataset"]).max_bin.agg(n64=lambda s: int((s == 64).sum()), n="count")
    print("\nC1: bin2 picks of 64 over seeds, by pair")
    print(g.T.to_string())
    for ds in ("kin8nm", "naval"):
        print(f"   {ds}:", {f: f"{int(r.n64)}/{int(r.n)} picked 64" for (f, d2), r in g.iterrows() if d2 == ds})
    print(f"   all searches: 64 picked in {int((best.max_bin == 64).sum())}/{len(best)}; share of trials drawn at 64: {float((tr.max_bin == 64).mean()):.2f}")

pk = S.picks("bin50", 50)
if pk is not None:
    tr, best = pk
    g = best.groupby(["family", "dataset"]).max_bin.agg(["min", "median", "max", "count"])
    g["ratio"] = g["max"] / g["min"]
    print("\nbin50: max_bin picked at 50 trials")
    print(g.round(1).to_string())
    print(f"   max / min >= 8 on {int((g.ratio >= 8).sum())}/{len(g)} pairs; best trial index median {best.trial.median():.0f}, share with best trial >= 20: {float((best.trial >= 20).mean()):.2f}")
for stem in ("2d50",):
    pk = S.picks(stem, 50)
    if pk is not None:
        tr, best = pk
        print(f"\n{stem}: best trial index median {best.trial.median():.0f}, share of searches whose best trial is >= 20: {float((best.trial >= 20).mean()):.2f}")

# Tier 1: refit validation NLL against the reference
for a, r, k in (("c30_bin", "c30_2d", 50), ("c30_bin2", "2d", 20)):
    t, r0 = arm(a, k), arm(r, k)
    if len(t):
        v = (t.valid_nll - r0.valid_nll.reindex(t.index)).dropna().groupby(level=[0, 1]).mean()
        print(f"\nvalidation NLL of the refit, {a} - {r} at k = {k}: lower on {int((v < 0).sum())}/{len(v)} pairs, mean {v.mean():+.4f}")
