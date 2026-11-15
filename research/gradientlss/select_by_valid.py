"""Equal-budget comparison: per (dataset, family, mode, seed) pick the η whose
*validation* NLL is lowest among the swept tags, then compare the chosen
cells' TEST scores, Fisher − Observed, paired by seed. The test split is
never used for selection.

Usage: uv run --with pandas python -I select_by_valid.py experiments_v2.csv --prefix c1b_eta
"""
import sys

import pandas as pd

path = next((a for a in sys.argv[1:] if a.endswith(".csv")), "experiments_v2.csv")
prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "c1b_eta"

df = pd.read_csv(path)
df = df[df["tag"].str.startswith(prefix) & (df["error"].isna() | (df["error"] == ""))]
df = df[df["valid_nll"].notna() & df["nll"].notna()]

keys = ["dataset", "family", "hessian_mode", "seed"]
best = df.sort_values("valid_nll").groupby(keys, as_index=False).first()

rows = []
for (ds, fam), g in best.groupby(["dataset", "family"]):
    w = g.pivot(index="seed", columns="hessian_mode", values=["nll", "crps", "cov90", "rmse", "eta", "valid_nll", "scale_max"])
    w = w.dropna(subset=[("nll", "fisher"), ("nll", "observed")])
    n = len(w)
    if n == 0:
        continue
    d = w[("nll", "fisher")] - w[("nll", "observed")]
    dc = w[("crps", "fisher")] - w[("crps", "observed")]
    se = d.std(ddof=1) / n ** 0.5 if n > 1 else float("nan")
    rows.append(dict(
        dataset=ds, family=fam, n=n,
        d_nll=f"{d.mean():+.4f} ± {se:.4f}", z=f"{d.mean() / se:+.1f}" if se == se and se > 0 else "",
        d_crps=f"{dc.mean():+.4f}", wins=f"{int((d < 0).sum())}/{n}",
        obs_nll=f"{w[('nll', 'observed')].mean():.4f}", fis_nll=f"{w[('nll', 'fisher')].mean():.4f}",
        eta_obs=",".join(f"{e:g}" for e in w[("eta", "observed")]),
        eta_fis=",".join(f"{e:g}" for e in w[("eta", "fisher")]),
        obs_valid=f"{w[('valid_nll', 'observed')].mean():.4f}", fis_valid=f"{w[('valid_nll', 'fisher')].mean():.4f}",
    ))

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).to_string(index=False))
