"""Proposal 030: gradient-regularised Newton boosting. Tags c26_e<eta>_<van|grn|grnx>, eta in 0.05, 0.3, 1.

Usage: uv run --with pandas --with numpy python -I c26_grn_summary.py <experiments_v2.csv>

A failed cell has a non-finite row or a test NLL more than 1 above the same pair and seed at vanilla eta 0.05.
"""
import math
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
df = df[df.tag.str.match(r"^c26_e[0-9.]+_(van|grn|grnx)$", na=False)].drop_duplicates(["tag", "family", "dataset", "seed"], keep="last")
df[["eta_s", "arm"]] = df.tag.str.extract(r"^c26_e([0-9.]+)_(\w+)$")
key = ["family", "dataset", "seed"]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
ref = df[(df.eta_s == "0.05") & (df.arm == "van")].set_index(key).nll
df["ref"] = ref.reindex(pd.MultiIndex.from_frame(df[key])).values
df["failed"] = (df.nonfinite_rows > 0) | df.nll.isna() | (df.nll > df.ref + 1)
print("cells per tag:", df.groupby("tag").size().to_dict())
se = lambda s: s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")

M = df.pivot_table(index=["family", "dataset"], columns=["eta_s", "arm"], values="nll", aggfunc="mean")
print()
print("mean test NLL over 5 seeds")
print(M.round(4).to_string())
R = df.pivot_table(index=["family", "dataset"], columns=["eta_s", "arm"], values="best_iter", aggfunc="median")
print()
print("median best round")
print(R.round(0).to_string())
F = df.pivot_table(index=["eta_s"], columns="arm", values="failed", aggfunc="sum")
print()
print("failed cells (of 60 per arm and eta)")
print(F.to_string())
fc = df[df.failed]
if len(fc):
    print(fc[["tag", "family", "dataset", "seed", "nll", "ref", "nonfinite_rows", "best_iter"]].round(3).to_string(index=False))

P = df.pivot_table(index=key, columns=["eta_s", "arm"], values="nll")
n = len(M)


def cmp(a, b, label, thr=0.005):
    d = (P[a] - P[b]).groupby(level=[0, 1]).mean()
    print(f"{label}: better by > {thr} on {int((d < -thr).sum())}/{n}, worse by > {thr} on {int((d > thr).sum())}/{n}, "
          f"within on {int((d.abs() <= thr).sum())}/{n}; mean {d.mean():+.4f}, median {d.median():+.4f}")
    return d


print()
d03 = cmp(("0.3", "van"), ("0.05", "van"), "R1  vanilla eta 0.3 − vanilla eta 0.05", 0.02)
d1 = cmp(("1", "van"), ("0.05", "van"), "    vanilla eta 1 − vanilla eta 0.05", 0.02)
out = {}
for eta in ("0.05", "0.3", "1"):
    for arm in ("grn", "grnx"):
        if (eta, arm) in P:
            out[(eta, arm)] = cmp((eta, arm), (eta, "van"), f"    {arm} − vanilla at eta {eta}")
print()
for eta in ("0.05", "0.3", "1"):
    for arm in ("grn", "grnx"):
        if (eta, arm) in P:
            cmp((eta, arm), ("0.05", "van"), f"R5  {arm} at eta {eta} − vanilla at eta 0.05")
print()
print("per pair: arm − vanilla at the same eta (mean over seeds)")
print(pd.DataFrame({f"{a}@{e}": v for (e, a), v in out.items()}).round(4).to_string())
print()
print("per pair: vanilla at eta 0.3 and 1 minus vanilla at eta 0.05")
print(pd.DataFrame({"eta0.3": d03, "eta1": d1}).round(4).to_string())
# amendment: half-step controls — is grn at eta more than vanilla at eta / 2?
print()
for eta, half in (("0.3", "0.15"), ("1", "0.5")):
    if (eta, "grn") in P and (half, "van") in P:
        d = cmp((eta, "grn"), (half, "van"), f"control  grn at eta {eta} − vanilla at eta {half}")
        print("        ", d.round(4).to_dict())
