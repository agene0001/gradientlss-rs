"""Proposal 018: family selection by marginal fit (dist_select) against held-out conditional score.

Usage: uv run --with pandas python -I c14f_family_selection.py <experiments_v2.csv> <c14f.log>

Per (dataset, seed): the family picked by (1) the unconditional training NLL (DSEL lines), (2) the
trained model's validation NLL, (3) test NLL (oracle). Regret = test NLL of the pick − oracle's.
"""
import math
import re
import sys

import pandas as pd

csv_path, log_path = sys.argv[1], sys.argv[2]
PAT = re.compile(r"^DSEL (\S+) (\d+) (\S+) n (\d+) nll (\S+)")
rows = []
for line in open(log_path):
    m = PAT.match(line)
    if m:
        ds, seed, fam, n, nll = m.groups()
        rows.append(dict(dataset=ds, seed=int(seed), family=fam, marg=float(nll) / int(n)))
marg = pd.DataFrame(rows)

d = pd.read_csv(csv_path)
d = d[(d.tag == "c14f")].drop_duplicates(["dataset", "family", "seed"], keep="last")
bad = d[d.error.notna() & (d.error != "")]
if len(bad):
    print("cells with errors (kept out of the selection):")
    print(bad[["dataset", "family", "seed", "error"]].to_string(index=False))
d = d[d.error.isna() | (d.error == "")]
m = d.merge(marg, on=["dataset", "family", "seed"], how="left")

pd.set_option("display.width", 250)
print("mean over seeds: marginal training NLL per row, validation NLL, test NLL, CRPS")
print(m.groupby(["dataset", "family"]).agg(marg=("marg", "mean"), valid=("valid_nll", "mean"), test=("nll", "mean"),
                                           crps=("crps", "mean"), nonfinite=("nonfinite_rows", "sum"),
                                           seeds=("seed", "count")).round(4).to_string())
print()

out = []
for (ds, seed), g in m.groupby(["dataset", "seed"]):
    g = g.set_index("family")
    pick = dict(marginal=g.marg.idxmin(), valid=g.valid_nll.idxmin(), oracle=g.nll.idxmin())
    best = g.nll.min()
    out.append(dict(dataset=ds, seed=seed, **pick,
                    regret_marginal=g.nll[pick["marginal"]] - best,
                    regret_valid=g.nll[pick["valid"]] - best))
o = pd.DataFrame(out)


def mode(s):
    c = s.value_counts()
    return f"{c.index[0]} ({c.iloc[0]}/{len(s)})"


summ = o.groupby("dataset").agg(
    marginal_pick=("marginal", mode), valid_pick=("valid", mode), oracle=("oracle", mode),
    marginal_eq_oracle=("seed", lambda s: int((o.loc[s.index, "marginal"] == o.loc[s.index, "oracle"]).sum())),
    valid_eq_oracle=("seed", lambda s: int((o.loc[s.index, "valid"] == o.loc[s.index, "oracle"]).sum())),
    regret_marginal=("regret_marginal", "mean"), regret_valid=("regret_valid", "mean"),
).round(4)
print("picks (most frequent over seeds) and mean regret in test NLL")
print(summ.to_string())
n = len(o)
print(f"\nall cells: marginal = oracle in {int((o.marginal == o.oracle).sum())}/{n}, "
      f"valid = oracle in {int((o.valid == o.oracle).sum())}/{n}; "
      f"mean regret marginal {o.regret_marginal.mean():.4f}, valid {o.regret_valid.mean():.4f}")
