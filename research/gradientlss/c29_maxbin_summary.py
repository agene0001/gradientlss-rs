"""Proposal 033: max_bin as a search axis. Tags c29_mb<k> (untuned), c29_bin_k<k>, optionally c29_subsbin_k<k>.

Usage: uv run --with pandas --with numpy python -I c29_maxbin_summary.py <experiments_v2.csv> [bin=<log,log,...>]

References: untuned c12_off / c19_def / c17_new / c27_def; two axes c13_raw / c19_tpe / c27_2d; subs c25_subs / c27_subs.
guard = keep the refit only if its validation NLL beats the untuned default's.
"""
import math
import re
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
df = df[df.error.isna() | (df.error == "")]
logs = dict(a.split("=", 1) for a in sys.argv[2:])
key = ["family", "dataset", "seed"]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
dd = lambda t: df[df.tag == t].drop_duplicates(key, keep="last").set_index(key)


def first(*tags):
    t = pd.concat([dd(x) for x in tags])
    return t[~t.index.duplicated(keep="first")]


base = first("c12_off", "c19_def", "c17_new", "c27_def")
two = {k: first(f"c13_raw_k{k}", f"c19_tpe_k{k}", f"c27_2d_k{k}") for k in (10, 20)}
subs = {k: first(f"c25_subs_k{k}", f"c27_subs_k{k}") for k in (10, 20)}


def which(ds):
    return "fresh" if ds.startswith("c_n_") else "unseen" if ds.startswith("c_") else "dev"


def se(s):
    return s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")


# ---- stage 0
mb = {int(t[len("c29_mb"):]): dd(t) for t in sorted(df.tag.dropna().unique()) if re.fullmatch(r"c29_mb\d+", t)}
if 256 in mb:
    j = mb[256][["nll", "crps", "valid_nll", "best_iter"]].join(base[["nll", "crps", "valid_nll", "best_iter"]], rsuffix="_ref", how="inner")
    same = (j.nll == j.nll_ref) & (j.crps == j.crps_ref) & (j.valid_nll == j.valid_nll_ref) & (j.best_iter == j.best_iter_ref)
    print(f"M0: c29_mb256 cells {len(mb[256])}, matched to a reference {len(j)}, identical {int(same.sum())}")
    if (~same).any():
        print(j[~same].head(10).round(5).to_string())
rows = {}
for k, t in mb.items():
    if k == 256:
        continue
    d = (t.nll - base.nll.reindex(t.index)).dropna()
    rows[k] = d.groupby(level=[0, 1]).mean()
if rows:
    S = pd.DataFrame(rows)
    S.insert(0, "untuned_256", base.nll.groupby(level=[0, 1]).mean().reindex(S.index))
    S.insert(0, "set", [which(ds) for _, ds in S.index])
    print()
    print("STAGE 0 — untuned test NLL at max_bin k minus at 256, mean over 5 seeds")
    print(S.sort_values("set").round(4).to_string())
    for k in sorted(rows):
        d = S[k]
        print(f"   max_bin {k}: better by > 0.005 on {int((d < -0.005).sum())}/{len(d)}, worse by > 0.005 on {int((d > 0.005).sum())}/{len(d)}, "
              f"within on {int((d.abs() <= 0.005).sum())}/{len(d)}; mean {d.mean():+.4f}, median {d.median():+.4f}")
    secs = {k: t.fit_secs.mean() for k, t in mb.items()}
    print("   mean fit seconds (shared machine, indicative only):", {k: round(v, 1) for k, v in sorted(secs.items())})


# ---- stage 1 / 2
def tier2(arm, ref, ref_name, k):
    t = dd(f"c29_{arm}_k{k}")
    if len(t) == 0:
        return None
    r = ref[k].reindex(t.index)
    b = base.reindex(t.index)
    G = pd.DataFrame({"plain": t.nll - b.nll, "guard": (t.nll - b.nll).where(t.valid_nll < b.valid_nll, 0.0),
                      "ref_plain": r.nll - b.nll, "ref_guard": (r.nll - b.nll).where(r.valid_nll < b.valid_nll, 0.0)}).dropna()
    P = G.groupby(level=[0, 1]).mean()
    P["d_plain"] = P.plain - P.ref_plain
    P["d_guard"] = P.guard - P.ref_guard
    paired = (G.plain - G.ref_plain).groupby(level=[0, 1]).agg(lambda s: f"± {se(s):.4f} ({int((s < 0).sum())}/{len(s)})")
    P["paired"] = paired
    P.insert(0, "set", [which(ds) for _, ds in P.index])
    n = len(P)
    print()
    print(f"{arm} against {ref_name}, k = {k} — test NLL − untuned default, mean over seeds; d_ = {arm} − {ref_name}")
    print(P.sort_values("set").round(4).to_string())
    for name, g in [("all", P)] + [(s, P[P.set == s]) for s in ("dev", "unseen", "fresh")]:
        if len(g) == 0:
            continue
        print(f"   {name:6s} ({len(g):2d} pairs): mean plain {g.plain.mean():+.4f} vs {g.ref_plain.mean():+.4f}; guarded {g.guard.mean():+.4f} vs {g.ref_guard.mean():+.4f}; "
              f"median guarded {g.guard.median():+.4f} vs {g.ref_guard.median():+.4f}; worse than default > 0.005: plain {int((g.plain > 0.005).sum())}, guarded {int((g.guard > 0.005).sum())} "
              f"(ref {int((g.ref_plain > 0.005).sum())}, {int((g.ref_guard > 0.005).sum())})")
    print(f"   M2: plain {arm} better than plain {ref_name} by > 0.005 on {int((P.d_plain < -0.005).sum())}/{n}, worse by > 0.005 on {int((P.d_plain > 0.005).sum())}/{n}")
    print(f"   rule (a) guarded {arm} worse than default by > 0.005: {int((P.guard > 0.005).sum())}/{n};  (b) guarded {arm} better than guarded {ref_name}: "
          f"{int((P.d_guard < 0).sum())}/{n};  (c) median of guarded difference: {P.d_guard.median():+.4f} (mean {P.d_guard.mean():+.4f})")
    return P


for k in (10, 20):
    tier2("bin", two, "two axes", k)
    tier2("subsbin", subs, "subs", k)

# ---- picks (M3)
PAT = re.compile(r"^HPO (\S+) (\S+) (\d+) trial (\d+) cv (\S+) mcw (\S+) lambda (\S+)(.*)$")
for arm, paths in logs.items():
    rows = []
    for p in paths.split(","):
        for line in open(p):
            m = PAT.match(line)
            if not m:
                continue
            ds, fam, seed, tno, cv, mcw, lam, rest = m.groups()
            extra = dict(zip(rest.split()[0::2], map(float, rest.split()[1::2])))
            cv = float(cv)
            rows.append(dict(dataset=ds, family=fam, seed=int(seed), trial=int(tno), cv=cv if np.isfinite(cv) else np.inf, max_bin=extra.get("max_bin", np.nan)))
    tr = pd.DataFrame(rows).drop_duplicates(key + ["trial"], keep="last")
    tr = tr[tr.trial < 20]
    pk = tr.loc[tr.groupby(key).cv.idxmin()]
    g = pk.groupby(["family", "dataset"]).max_bin.agg(["min", "median", "max", "count"])
    g["ratio"] = g["max"] / g["min"]
    print()
    print(f"{arm}: max_bin picked at 20 trials over the seeds")
    print(g.round(1).to_string())
    print(f"   M3: max / min ≥ 8 on {int((g.ratio >= 8).sum())}/{len(g)} pairs; median pick over all searches {pk.max_bin.median():.0f}; "
          f"picks below 64: {int((pk.max_bin < 64).sum())}/{len(pk)}, above 512: {int((pk.max_bin > 512).sum())}/{len(pk)}")
    # does the fold score move with max_bin inside a search?
    tr = tr[np.isfinite(tr.cv)].copy()
    tr["rel"] = tr.cv - tr.groupby(key).cv.transform("median")
    tr["bucket"] = pd.cut(tr.max_bin, [0, 32, 64, 128, 256, 512, 2000], labels=["≤32", "33–64", "65–128", "129–256", "257–512", ">512"])
    print("   trials: CV score − the search's median, by max_bin bucket (median over all trials; mean in brackets is dominated by bad mcw/λ draws)")
    print("  ", tr.groupby("bucket", observed=True).rel.median().round(4).to_dict())
    rho = tr.groupby(key).apply(lambda t: t.cv.rank().corr(np.log(t.max_bin).rank()) if len(t) > 3 else np.nan)
    print(f"   rank correlation of CV score with log max_bin inside a search: median {rho.median():+.2f}; "
          f"by pair, share of searches with |ρ| > 0.3: {float((rho.abs() > 0.3).mean()):.2f}")
