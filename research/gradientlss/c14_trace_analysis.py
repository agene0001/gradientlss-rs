"""Cycle 14, zero-compute pass: what the cycle 13 search traces already say.

Usage: uv run --with pandas --with numpy python -I c14_trace_analysis.py <experiments_v2.csv> <log>[,<log>...]

Reads the bench's stderr `HPO ...` lines (one per trial) and the `c13_*` / `c12_*` rows of the
experiments table. Prints

  A. sampler behaviour: when the final best was found, how much the best CV score moved after trial 5,
     and how widely the draws were spread against a log-uniform sampler on the same range;
  B. transfer: tuned − untuned on validation and on test (paired by seed), and whether the CV gain
     of the 20-trial pick over the 5-trial pick shows up on test.

Nothing here is a new measurement; it re-reads cycle 13.
"""
import math
import re
import sys

import numpy as np
import pandas as pd

PAT = re.compile(r"^HPO (\S+) (\S+) (\d+) trial (\d+) cv (\S+) mcw (\S+) lambda (\S+)")
RUN = re.compile(r"^== .*BENCH_TAG=(\S+)")

csv_path, logs = sys.argv[1], sys.argv[2].split(",")

rows = []
for p in logs:
    arm = None
    for line in open(p):
        m = RUN.match(line)
        if m:
            arm = m.group(1)
            continue
        m = PAT.match(line)
        if m:
            ds, fam, seed, t, cv, mcw, lam = m.groups()
            rows.append(dict(arm=arm, dataset=ds, family=fam, seed=int(seed), trial=int(t), cv=float(cv),
                             lmcw=math.log10(float(mcw)), llam=math.log10(float(lam))))
tr = pd.DataFrame(rows).sort_values(["arm", "dataset", "family", "seed", "trial"])
tr["cv"] = tr["cv"].where(np.isfinite(tr["cv"]), np.inf)
key = ["arm", "family", "dataset"]
cell = key + ["seed"]

# ---------------------------------------------------------------- A. sampler behaviour
UNIFORM_SD = 6.0 / math.sqrt(12.0)  # sd of a uniform draw over six decades


def per_cell(g):
    g = g.sort_values("trial")
    best = g.cv.cummin().to_numpy()
    n = len(g)
    late = g[g.trial >= 5]
    early = g[g.trial < 5]
    # 4 x 4 grid over the six-decade square: how many of the 16 cells were visited
    gx = np.clip(((g.lmcw + 3) / 1.5).astype(int), 0, 3)
    gy = np.clip(((g.llam + 3) / 1.5).astype(int), 0, 3)
    return pd.Series(dict(
        n=n,
        best_at=int(np.argmin(g.cv.to_numpy())) + 1,
        gain_5_to_end=best[min(4, n - 1)] - best[-1],
        sd_early=(early.lmcw.std() + early.llam.std()) / 2,
        sd_late=(late.lmcw.std() + late.llam.std()) / 2,
        cells=len(set(zip(gx, gy))),
        # distance in decades from each late draw to the incumbent at that time
        near_inc=np.mean([
            math.hypot(r.lmcw - g.lmcw.iloc[int(np.argmin(g.cv.to_numpy()[:r.trial]))],
                       r.llam - g.llam.iloc[int(np.argmin(g.cv.to_numpy()[:r.trial]))])
            for r in late.itertuples()
        ]) if len(late) else np.nan,
    ))


pc = tr.groupby(cell).apply(per_cell, include_groups=False).reset_index()
A = pc.groupby(key).agg(
    seeds=("seed", "count"),
    best_at_median=("best_at", "median"),
    best_in_first5=("best_at", lambda s: f"{int((s <= 5).sum())}/{len(s)}"),
    cv_gain_after_5=("gain_5_to_end", "mean"),
    sd_draws_1to5=("sd_early", "mean"),
    sd_draws_6plus=("sd_late", "mean"),
    grid_cells_of_16=("cells", "mean"),
    decades_from_incumbent=("near_inc", "mean"),
).reset_index()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("A. sampler behaviour (cycle 13 TPE traces, 20 trials, 2 log axes over 6 decades)")
print(f"   a log-uniform sampler has sd {UNIFORM_SD:.2f} decades per axis; 20 uniform draws visit ~11.6 of 16 cells")
print("   and sit ~3.1 decades from any fixed point on average")
print(A.round(4).to_string(index=False))
print()
tot = pc.best_at
print(f"   all cells: final best found by trial 5 in {int((tot <= 5).sum())}/{len(tot)}, by trial 10 in "
      f"{int((tot <= 10).sum())}/{len(tot)}; mean sd of draws 6+ = {pc.sd_late.mean():.2f} decades")
print()

# ---------------------------------------------------------------- B. transfer
df = pd.read_csv(csv_path)
df = df[df.error.isna() | (df.error == "")]
t = df[df.tag.str.startswith("c13_", na=False)].copy()
t[["arm", "k"]] = t.tag.str.extract(r"c13_(raw|norm)_k(\d+)")
t = t.dropna(subset=["arm"])
t["k"] = t.k.astype(int)
# the last row wins when a cell was re-run
t = t.drop_duplicates(["arm", "k", "dataset", "family", "seed"], keep="last")
base = df[df.tag.isin(["c12_off", "c12_marginal"])].drop_duplicates(["tag", "dataset", "family", "seed"], keep="last")
base = base.assign(arm=base.tag.map({"c12_off": "raw", "c12_marginal": "norm"}))


def pm(d):
    d = d.dropna()
    if len(d) < 2:
        return "n/a"
    return f"{d.mean():+.4f} ± {d.std(ddof=1) / math.sqrt(len(d)):.4f} ({int((d < 0).sum())}/{len(d)})"


out = []
for (arm, fam, ds), g in t.groupby(["arm", "family", "dataset"]):
    b = base[(base.arm == arm) & (base.family == fam) & (base.dataset == ds)].set_index("seed")
    k20 = g[g.k == 20].set_index("seed")
    k5 = g[g.k == 5].set_index("seed")
    if b.empty or k20.empty:
        continue
    out.append(dict(
        arm=arm, family=fam, dataset=ds,
        tuned20_minus_untuned_valid=pm(k20.valid_nll - b.valid_nll),
        tuned20_minus_untuned_test=pm(k20.nll - b.nll),
        k20_minus_k5_test=pm(k20.nll - k5.nll),
    ))
print("B. transfer. 'untuned' = the same arm at mcw = lambda = 1 (c12_off / c12_marginal).")
print("   valid = the bench's early-stopping split (not seen by the search); mean ± s.e. (seeds better).")
print(pd.DataFrame(out).to_string(index=False))
