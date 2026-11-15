"""Proposals 034 / 035: an arm against a reference search under the inclusion rule of 033.

Usage: uv run --with pandas --with numpy python -I c30_summary.py <experiments_v2.csv> <log dir> <arm>=<ref>[@k] ...
  arm, ref: tag stems; "2d" = the stored two-axis rows (c13_raw / c19_tpe / c27_2d), "subs", "depth" likewise.
  e.g.  c30_bin=c30_2d@50  c30_bin2=2d@20  c30_2d=2d@20  c31_eta=2d@20
base=<tag> replaces the untuned reference rows (036: base=c32_l_def); -q hides the per-pair table.
Extra CSVs may be appended to the first argument with commas (per-job files are concatenated).
guard = keep the refit only if its validation NLL beats the untuned default's.
"""
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd

paths = [q for p in sys.argv[1].split(",") for q in (sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p) else [p])]
df = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
df = df[df.error.isna() | (df.error == "")]
logdir = sys.argv[2]
key = ["family", "dataset", "seed"]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
dd = lambda t: df[df.tag == t].drop_duplicates(key, keep="last").set_index(key)


def first(*tags):
    t = pd.concat([dd(x) for x in tags])
    return t[~t.index.duplicated(keep="first")]


STORED = {"2d": ("c13_raw", "c19_tpe", "c27_2d"), "subs": ("c25_subs", "c27_subs"), "depth": ("c25_depth",)}
arm = lambda stem, k: first(*[f"{s}_k{k}" for s in STORED.get(stem, (stem,))])
NPAIRS = int(next((x.split("=", 1)[1] for x in sys.argv[3:] if x.startswith("pairs=")), 36))  # (b) needs two thirds of the pairs
BASE = next((x.split("=", 1)[1] for x in sys.argv[3:] if x.startswith("base=")), None)  # e.g. base=c32_l_def (036)
base = dd(BASE) if BASE else first("c12_off", "c19_def", "c17_new", "c27_def")
which = lambda ds: "fresh" if ds.startswith("c_n_") else "unseen" if ds.startswith("c_") else "dev"
se = lambda s: s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")
COLS = ["nll", "crps", "valid_nll", "best_iter"]


def identical(a, b, name):
    j = a[COLS].join(b[COLS], rsuffix="_ref", how="inner")
    same = np.logical_and.reduce([j[c] == j[c + "_ref"] for c in COLS])
    print(f"REPRO {name}: cells {len(a)}, matched {len(j)}, identical {int(same.sum())}")
    if (~same).any():
        print(j[~same].head(8).round(5).to_string())


def rule(a_stem, r_stem, k, show=True, rk=None):
    t, r0 = arm(a_stem, k), arm(r_stem, rk or k)
    if len(t) == 0:
        return None
    r, b = r0.reindex(t.index), base.reindex(t.index)
    G = pd.DataFrame({"plain": t.nll - b.nll, "guard": (t.nll - b.nll).where(t.valid_nll < b.valid_nll, 0.0),
                      "ref_plain": r.nll - b.nll, "ref_guard": (r.nll - b.nll).where(r.valid_nll < b.valid_nll, 0.0)}).dropna()
    P = G.groupby(level=[0, 1]).mean()
    P["d_plain"], P["d_guard"] = P.plain - P.ref_plain, P.guard - P.ref_guard
    P["paired"] = (G.plain - G.ref_plain).groupby(level=[0, 1]).agg(lambda s: f"± {se(s):.4f} ({int((s < 0).sum())}/{len(s)})")
    P["n"] = G.groupby(level=[0, 1]).size()
    P.insert(0, "set", [which(ds) for _, ds in P.index])
    n = len(P)
    print()
    print(f"=== {a_stem} against {r_stem}, k = {k}{f' (reference at k = {rk})' if rk else ''}: {len(G)} cells, {n} pairs — test NLL − untuned default; d_ = arm − reference")
    if show:
        print(P.sort_values("set").round(4).to_string())
    for name, g in [("all", P)] + [(s, P[P.set == s]) for s in ("dev", "unseen", "fresh")]:
        if len(g):
            print(f"   {name:6s} ({len(g):2d}): plain mean {g.plain.mean():+.4f} / {g.ref_plain.mean():+.4f}, median {g.plain.median():+.4f} / {g.ref_plain.median():+.4f}, worse than default {int((g.plain > 0.005).sum())} / {int((g.ref_plain > 0.005).sum())}"
                  f" | guarded mean {g.guard.mean():+.4f} / {g.ref_guard.mean():+.4f}, median {g.guard.median():+.4f} / {g.ref_guard.median():+.4f}, worse than default {int((g.guard > 0.005).sum())} / {int((g.ref_guard > 0.005).sum())}"
                  f" | (b) {int((g.d_guard < 0).sum())}, (c) {g.d_guard.median():+.4f}")
    print(f"   plain arm better than plain reference by > 0.005 on {int((P.d_plain < -0.005).sum())}/{n}, worse on {int((P.d_plain > 0.005).sum())}/{n}")
    a_, b_, c_ = int((P.guard > 0.005).sum()), int((P.d_guard < 0).sum()), P.d_guard.median()
    verdict = "PASS" if (a_ == 0 and 3 * b_ >= 2 * NPAIRS and c_ <= -0.005 and n == NPAIRS) else "FAIL" if n == NPAIRS else "incomplete"
    print(f"   RULE (a) guarded worse than default: {a_}/{n} {P[P.guard > 0.005].guard.round(4).to_dict()};  (b) better than guarded reference: {b_}/{n};  (c) median {c_:+.4f} (mean {P.d_guard.mean():+.4f})  → {verdict}")
    return P


def picks(stem, kmax):
    pat = re.compile(r"^HPO (\S+) (\S+) (\d+) trial (\d+) cv (\S+) mcw (\S+) lambda (\S+)(.*)$")
    rows = []
    for p in sorted(glob.glob(os.path.join(logdir, f"{stem}_*.log"))):
        for line in open(p):
            m = pat.match(line)
            if m:
                ds, fam, seed, tno, cv, mcw, lam, rest = m.groups()
                w = rest.split()
                cv = float(cv)
                rows.append(dict(dataset=ds, family=fam, seed=int(seed), trial=int(tno), cv=cv if np.isfinite(cv) else np.inf, mcw=float(mcw), lam=float(lam),
                                 **{w[i]: float(w[i + 1]) for i in range(0, len(w) - 1, 2)}))
    if not rows:
        return None
    tr = pd.DataFrame(rows).drop_duplicates(key + ["trial"], keep="last")
    tr = tr[tr.trial < kmax]
    return tr, tr.loc[tr.groupby(key).cv.idxmin()]


if __name__ == "__main__":
    for spec in sys.argv[3:]:
        if spec == "-q" or spec.startswith(("base=", "pairs=")):
            continue
        a_stem, rest = spec.split("=")
        r_stem, k = rest.split("@")
        ak = None
        if "@" in a_stem:  # arm@50=2d@20: arm at 50 trials against the reference at 20
            a_stem, ak = a_stem.split("@")
        rule(a_stem, r_stem, int(ak or k), show="-q" not in sys.argv, rk=int(k) if ak else None)
