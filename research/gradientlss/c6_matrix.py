"""Cycle-6 matrix: ΔNLL vs the family's 'none' arm, mean (worst seed), per arm × dataset.

Usage: uv run --with pandas python -I c6_matrix.py experiments_v2.csv
"""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
df = df[df["error"].isna() | (df["error"] == "")]
GROUPS = {
    "gaussian (init0)": ("gaussian", "c5d_pes_0", {
        "freeze": "c5d_pes_20", "ann.5": "c6a_g_20_1_0.5", "ann.25": "c6d_g_20_1_0.25"}),
    "gamma_ms (init0)": ("gamma_ms", "c5f_pes_0", {
        "freeze": "c5f_pes_20_1", "ann.5": "c6a_gm_20_1_0.5", "ann.25": "c6d_gm_20_1_0.25"}),
    "nb_md (init0)": ("nb_md", "c5f_pes_0", {
        "freeze": "c5f_pes_20_1", "ann.5": "c6a_nb_20_1_0.5", "ann.25": "c6d_nb_20_1_0.25"}),
    "studentt (init0)": ("studentt", "c6b_t_0", {
        "s.freeze": "c6b_t_20_2", "df.freeze": "c6b_t_20_0", "both.freeze": "c6b_t_20_0_2",
        "s.ann.5": "c6b_t_20_2_0.5", "s.ann.25": "c6d_t_20_2_0.25", "both.ann.5": "c6b_t_20_0_2_0.5"}),
    "studentt (init1)": ("studentt", "c6e_ti_0", {
        "s.freeze": "c6e_ti_20_2", "s.ann.5": "c6e_ti_20_2_0.5", "s.ann.25": "c6e_ti_20_2_0.25",
        "both.ann.5": "c6e_ti_20_0_2_0.5"}),
    "studentt (warm api)": ("studentt", "c6c_tw_0", {
        "s.freeze": "c6c_tw_20_2", "both.freeze": "c6c_tw_20_0_2", "s.ann.5": "c6c_tw_20_2_0.5",
        "both.ann.5": "c6c_tw_20_0_2_0.5"}),
}
pd.set_option("display.width", 250)
for name, (fam, base, arms) in GROUPS.items():
    d = df[df.family == fam].drop_duplicates(["tag", "dataset", "seed"], keep="last")
    b = d[d.tag == base].set_index(["dataset", "seed"]).nll
    out = {}
    for arm, tag in arms.items():
        a = d[d.tag == tag].set_index(["dataset", "seed"]).nll
        delta = (a - b).dropna()
        if delta.empty:
            continue
        g = delta.groupby(level=0)
        out[arm] = g.apply(lambda s: f"{s.mean():+.3f} ({s.max():+.3f}){'' if len(s) == 5 else f' n{len(s)}'}")
    if out:
        print(f"\n## {name}: ΔNLL vs none — mean (worst seed)")
        print(pd.DataFrame(out).to_string())
