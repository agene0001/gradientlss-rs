"""Proposal 037: parts P (picks), V (validation choice among untuned candidates), F (family by validation).

Usage: uv run --with pandas --with numpy python -I c33_score.py <experiments_v2.csv>[,<dir>] <log dir>
The rule table for part P comes from c30_summary.py (c33_pes=2d@20).
"""
import sys

import numpy as np
import pandas as pd

sys.argv = sys.argv[:3]
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import c30_summary as S  # noqa: E402

base, dd, which = S.base, S.dd, S.which
G = "kin8nm power_plant protein naval gaussian_sim c_airfoil c_parkinsons c_gt_nox c_gt_co c_supercond c_n_concrete c_n_wine c_n_appliances c_n_sgemm c_n_gridstab".split()
M = "naval kin8nm c_airfoil c_parkinsons c_gt_nox c_gt_co c_supercond c_n_concrete c_n_wine c_n_appliances c_n_sgemm".split()
PO = "synth_pois bike_cnt c_abalone c_seoulbike c_n_metro".split()
NB = "bike_cnt c_abalone c_seoulbike c_n_metro c_n_news".split()
pairs36 = {("gaussian", d) for d in G} | {("gamma_ms", d) for d in M} | {("poisson", d) for d in PO} | {("nb_md", d) for d in NB}
assert len(pairs36) == 36

# ---- P
P = S.rule("c33_pes", "2d", 20, show=False)
S.rule("c33_pes", "2d", 10, show=False)
pk = S.picks("pes", 20)
if pk is not None:
    tr, best = pk
    w = best["__pes_window"]
    print(f"\nP2: picked window over {len(best)} searches: {w.value_counts().sort_index().to_dict()}; off (0) in {float((w == 0).mean()):.1%}; "
          f"picked anneal median {best['__pes_anneal'].median():.2f} (quartiles {best['__pes_anneal'].quantile(.25):.2f}, {best['__pes_anneal'].quantile(.75):.2f})")
    print("    drawn windows:", tr["__pes_window"].value_counts(normalize=True).sort_index().round(2).to_dict())
    g = best.groupby(["family", "dataset"])["__pes_window"].agg(lambda s: " ".join(str(int(x)) for x in sorted(s)))
    print("    picked windows by pair:", {f"{f}/{d}": v for (f, d), v in g.items()})

# ---- V
ARMS = ["v_obs", "v_fis", "v_pesoff", "v_init", "v_locflip"]
C = {"default": base}
for a in ARMS:
    t = dd(f"c33_{a}")
    if len(t):
        C[a] = t
print("\ncells:", {k: len(v) for k, v in C.items()})
if "v_locflip" in C:
    g = C["v_locflip"]
    g = g[g.index.get_level_values(0) == "gaussian"]
    S.identical(g, base, "v_locflip on Gaussian vs default (identity check)")


def per_pair(d):
    return d.dropna().groupby(level=[0, 1]).mean()


def counts(d, name):
    n = len(d)
    out = f"{name}: better by > 0.005 on {int((d < -0.005).sum())}/{n}, worse on {int((d > 0.005).sum())}/{n}, within on {int((d.abs() <= 0.005).sum())}/{n}; mean {d.mean():+.4f}, median {d.median():+.4f}"
    for s in ("dev", "unseen", "fresh"):
        g = d[[which(ds) == s for _, ds in d.index]]
        out += f" | {s} {int((g < -0.005).sum())}/{int((g > 0.005).sum())}/{len(g)}"
    return out


print("\nV1 — untuned alternative minus default, test NLL, mean over seeds (per set: better/worse/pairs)")
T = {}
for a in ARMS:
    if a in C:
        idx = C[a].index
        if a == "v_locflip":
            idx = idx[idx.get_level_values(0) != "gaussian"]
        T[a] = per_pair(C[a].nll.reindex(idx) - base.nll.reindex(idx))
        print("  ", counts(T[a], a))
if T:
    print(pd.DataFrame(T).round(4).to_string())

SEL = {"V-mode": ["default", "v_obs", "v_fis"], "V-start": ["default", "v_init", "v_locflip"], "V-pes": ["default", "v_pesoff"], "V-all": ["default"] + ARMS}
print("\nselections by validation NLL (per seed), test NLL of the pick minus the default's")
R = {}
for name, cands in SEL.items():
    if not all(c in C for c in cands):
        continue
    idx = C[cands[1]].index
    V = pd.DataFrame({c: C[c].valid_nll.reindex(idx) for c in cands})
    N = pd.DataFrame({c: C[c].nll.reindex(idx) for c in cands})
    V = V.where(np.isfinite(V), np.inf).fillna(np.inf)
    pick = V.idxmin(axis=1)
    got = pd.Series(N.to_numpy()[np.arange(len(N)), [cands.index(p) for p in pick]], index=idx)
    d = per_pair(got - N["default"])
    R[name] = d
    oracle = per_pair(N.min(axis=1) - N["default"])
    ok = int((d > 0.005).sum()) == 0 and int((d < -0.005).sum()) >= 9 and len(d) == 36
    print("  ", counts(d, name), f"| picks {pick.value_counts().to_dict()} | test-oracle mean {oracle.mean():+.4f} → {'PASS' if ok else 'FAIL' if len(d) == 36 else 'incomplete'}")
    if (d > 0.005).any():
        print("      worse:", {f"{f}/{ds}": round(v, 4) for (f, ds), v in d[d > 0.005].items()})
if R:
    print(pd.DataFrame(R).round(4).to_string())

# ---- F
print("\nF — family chosen by validation NLL of the untuned fit (stored rows)")
b = base[[p in pairs36 for p in base.index.droplevel(2)]]
rows = []
for famA, famB in (("gaussian", "gamma_ms"), ("poisson", "nb_md")):
    if famA not in b.index.get_level_values(0) or famB not in b.index.get_level_values(0):
        continue
    A, B = b.xs(famA, level=0), b.xs(famB, level=0)
    j = A[["nll", "valid_nll"]].join(B[["nll", "valid_nll"]], lsuffix="_a", rsuffix="_b", how="inner")
    j["pick_b"] = j.valid_nll_b < j.valid_nll_a
    j["best_b"] = j.nll_b < j.nll_a
    j["got"] = np.where(j.pick_b, j.nll_b, j.nll_a)
    j["regret"] = j.got - np.minimum(j.nll_a, j.nll_b)
    j["pairname"] = f"{famA}/{famB}"
    rows.append(j)
if rows:
    J = pd.concat(rows)
    g = J.groupby(["pairname", "dataset"]).agg(seeds=("regret", "size"), nll_a=("nll_a", "mean"), nll_b=("nll_b", "mean"), picks_b=("pick_b", "sum"),
                                               right=("regret", lambda s: int((s == 0).sum())), regret=("regret", "mean"))
    g["gap"] = g.nll_b - g.nll_a
    print(g.round(4).to_string())
    print(f"   F1: right family in {int((J.pick_b == J.best_b).sum())}/{len(J)} cells ({float((J.pick_b == J.best_b).mean()):.1%}); mean regret < 0.01 on {int((g.regret < 0.01).sum())}/{len(g)} datasets; "
          f"largest regret {g.regret.max():.4f}; always-a mean regret {float((J.nll_a - np.minimum(J.nll_a, J.nll_b)).mean()):.4f}, always-b {float((J.nll_b - np.minimum(J.nll_a, J.nll_b)).mean()):.4f}, validation pick {J.regret.mean():.4f}")
