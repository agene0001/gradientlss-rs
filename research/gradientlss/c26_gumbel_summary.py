"""Proposal 028: Gumbel arms base / ls<k> — mean, trimmed and worst-row read-outs.

Usage: uv run --with pandas --with numpy python -I c26_gumbel_summary.py <experiments_v2.csv> <arm>=<log> ...

The CSV gives mean test NLL, CRPS, validation NLL (tags c26_gb_<arm>); the logs (BENCH_DEBUG=1
BENCH_DEBUG_WORST=1) give, per cell, the worst row and the mean without the worst 0.5 % of rows.
A blow-up cell: mean test NLL falls by more than 0.5 when its single worst row is removed.
"""
import re
import sys

import numpy as np
import pandas as pd

csv_path = sys.argv[1]
logs = dict(a.split("=", 1) for a in sys.argv[2:])
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
CELL = re.compile(r"^DEBUG (\S+) Gumbel \S+ seed (\d+): n_iter")
ROW = re.compile(r"^WORST row nll (\S+) \(share \S+ of total\) y (\S+) params \[([^,]+), ([^\]]+)\]")
SUM = re.compile(r"^WORST median row nll (\S+), mean (\S+), mean without top row (\S+), trim0.5 (\S+) crps (\S+)")
rows = []
for arm, path in logs.items():
    cur, top = None, None
    for line in open(path):
        m = CELL.match(line)
        if m:
            cur, top = (m.group(1), int(m.group(2))), None
            continue
        m = ROW.match(line)
        if m and top is None:
            nll, y, loc, s = map(float, m.groups())
            top = dict(top_nll=nll, top_z=(y - loc) / s)
            continue
        m = SUM.match(line)
        if m and cur:
            med, mean, wo1, trim, crps = map(float, m.groups())
            rows.append(dict(arm=arm, dataset=cur[0], seed=cur[1], med_row=med, mean=mean, wo_top=wo1, trim=trim, **(top or {})))
L = pd.DataFrame(rows).drop_duplicates(["arm", "dataset", "seed"], keep="last")
L["blowup"] = (L["mean"] - L.wo_top) > 0.5

df = pd.read_csv(csv_path, low_memory=False)
df = df[(df.family == "gumbel") & df.tag.str.startswith("c26_gb_")].drop_duplicates(["tag", "dataset", "seed"], keep="last")
df["arm"] = df.tag.str.replace("c26_gb_", "")
J = df.merge(L, on=["arm", "dataset", "seed"], how="left")
chk = (J.nll - J["mean"]).abs().max()
print(f"cells: {len(J)}; log mean vs CSV nll, largest difference {chk:.2e}; non-finite rows {int(J.nonfinite_rows.sum())}")
order = [a for a in ["base", "ls1", "ls2", "ls4", "ls10", "ls30", "ls100"] if a in set(J.arm)]

g = J.groupby(["dataset", "arm"]).agg(nll=("nll", "mean"), nll_med=("nll", "median"), trim=("trim", "mean"), crps=("crps", "mean"),
                                      valid=("valid_nll", "mean"), best=("best_iter", "median"), blowups=("blowup", "sum"),
                                      min_top_z=("top_z", "min"), oracle=("oracle_nll", "mean"), scale=("mean_scale", "mean"))
print()
print("per dataset and arm, mean over 5 seeds (nll_med = median over seeds; blowups = cells where one row moves the mean by > 0.5)")
print(g.reindex(order, level=1).round(4).to_string())

# choice of k by validation rank on the development sets
dev = J[~J.dataset.str.startswith("c_") & (J.arm != "base")]
v = dev.groupby(["dataset", "arm"]).valid_nll.mean().unstack()
if v.shape[1] > 1:
    print()
    print("development sets: mean validation NLL by arm, and its rank")
    print(v.round(4).to_string())
    r = v.rank(axis=1)
    print("mean rank:", r.mean().round(2).to_dict(), " → chosen:", r.mean().idxmin())

# the rule against base
b = J[J.arm == "base"].set_index(["dataset", "seed"])
for a in order[1:]:
    x = J[J.arm == a].set_index(["dataset", "seed"])
    d = pd.DataFrame({"d_nll": x.nll - b.nll, "d_trim": x.trim - b.trim, "crps_rel": x.crps / b.crps - 1}).dropna()
    s = d.groupby(level=0).mean()
    print()
    print(f"{a} − base, mean over seeds (rule: ≤ +0.005 NLL; CRPS ≤ +1 %)")
    print(s.round(4).to_string())
    print(f"   mean NLL rule fails on {int((s.d_nll > 0.005).sum())}/{len(s)}; trimmed NLL fails on {int((s.d_trim > 0.005).sum())}/{len(s)}; "
          f"CRPS worse by > 1 % on {int((s.crps_rel > 0.01).sum())}/{len(s)}")

bl = J[J.blowup == True]
print()
print(f"blow-up cells: {len(bl)}; by arm {bl.groupby('arm').size().to_dict()}; worst-row z: max {bl.top_z.max() if len(bl) else float('nan'):.2f} "
      f"(all below −5: {bool((bl.top_z < -5).all())})")
if len(bl):
    print(bl[["arm", "dataset", "seed", "nll", "wo_top", "trim", "top_nll", "top_z", "crps"]].round(3).to_string(index=False))
sg = J[J.dataset == "synth_gumbel"]
if len(sg):
    sg = sg.assign(gap=sg.nll - sg.oracle_nll)
    print()
    print("synth_gumbel: test NLL − oracle NLL by arm (mean, max over seeds)")
    print(sg.groupby("arm").gap.agg(["mean", "max"]).reindex(order).round(4).to_string())
