"""Old recipe vs new recipe (proposals 019, 020), paired by seed.

Usage: uv run --with pandas python -I c16_recipe_transfer.py <experiments_v2.csv> <prefix>[,<prefix>...]
  e.g. c16_xgb,c16_lgb,c17     (tags <prefix>_old and <prefix>_new)

Families are grouped across the re-parametrisations (gamma / gamma_ms -> gamma, nb / nb_md -> nb).
A cell with an error, a NaN score or any non-finite test row counts as failed for that arm; the mean
difference is taken over seeds where both arms are finite, and failures are listed beside it.
"""
import math
import sys

import pandas as pd

csv_path, prefixes = sys.argv[1], sys.argv[2].split(",")
GROUP = {"gamma_ms": "gamma", "nb_md": "nb"}
df = pd.read_csv(csv_path)
df["group"] = df.family.map(lambda f: GROUP.get(f, f))
df["failed"] = df.error.notna() & (df.error != "") | df.nll.isna() | (df.nonfinite_rows > 0)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

summary = {}
for pre in prefixes:
    a = df[df.tag == f"{pre}_old"].drop_duplicates(["dataset", "group", "seed"], keep="last")
    b = df[df.tag == f"{pre}_new"].drop_duplicates(["dataset", "group", "seed"], keep="last")
    m = a.merge(b, on=["dataset", "group", "seed"], suffixes=("_old", "_new"))
    rows = []
    for (g, ds), c in m.groupby(["group", "dataset"]):
        ok = c[~c.failed_old & ~c.failed_new]
        d = ok.nll_new - ok.nll_old
        n = len(d)
        se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        rows.append(dict(
            family=g, dataset=ds, seeds=len(c),
            nll_old=round(ok.nll_old.mean(), 4) if n else float("nan"),
            nll_new=round(ok.nll_new.mean(), 4) if n else float("nan"),
            delta=round(d.mean(), 4) if n else float("nan"),
            se=round(se, 4), new_better=f"{int((d < 0).sum())}/{n}",
            fail_old=int(c.failed_old.sum()), fail_new=int(c.failed_new.sum()),
            iters=f"{c.best_iter_old.mean():.0f}->{c.best_iter_new.mean():.0f}",
            cov90=f"{ok.cov90_old.mean():.3f}->{ok.cov90_new.mean():.3f}" if n else "",
        ))
    t = pd.DataFrame(rows)
    summary[pre] = t.set_index(["family", "dataset"])
    print(f"== {pre}: new − old test NLL (negative = today's defaults better); {len(m)} paired cells")
    print(t.to_string(index=False))
    if len(t):
        worse = t[(t.delta > 0.005) | (t.fail_new > 0)]
        print(f"   pairs breaking the +0.005 rule or with failed `new` cells: {len(worse)}/{len(t)}"
              + ("" if worse.empty else "  -> " + ", ".join(f"{r.family}/{r.dataset}" for r in worse.itertuples())))
        print("   mean delta by family: " + ", ".join(f"{k} {v:+.3f}" for k, v in t.groupby("family").delta.mean().items()))
    print()

if "c16_xgb" in summary and "c16_lgb" in summary:
    j = summary["c16_xgb"][["delta"]].join(summary["c16_lgb"][["delta"]], lsuffix="_xgb", rsuffix="_lgb").dropna()
    big = j[j.delta_xgb.abs() > 0.01]
    agree = (big.delta_xgb * big.delta_lgb > 0).sum()
    print(f"backend agreement: of {len(big)} pairs with |delta| > 0.01 on XGBoost, the sign agrees on LightGBM in {agree}")
    print(f"mean delta over common pairs: XGBoost {j.delta_xgb.mean():+.3f}, LightGBM {j.delta_lgb.mean():+.3f}")
