#!/bin/zsh
# Proposal 021, held-out step: base vs the recipe chosen on the development sets by validation rank,
# run once on the five unseen continuous sets of proposal 020.
# usage: run_c18.sh <arm: base|port>        (stderr is the log)
# Chosen (c15_port_summary.py, 2026-11-13): cauchy d_loc, gumbel d_raw, laplace d_ls, logistic d_ls, lognormal d_loc.
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c15c
A=$1
C=c_airfoil,c_parkinsons,c_gt_nox,c_gt_co,c_supercond
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_PES=auto BENCH_TAG=c18_$A BENCH_DATASETS=$C
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
if [ $A = base ]; then
  run BENCH_MODES=damped BENCH_FAMILIES=gaussian,cauchy,logistic,gumbel,laplace,lognormal
else
  run BENCH_LS_PORT=1 BENCH_MODES=damped BENCH_LS_START=loc BENCH_FAMILIES=cauchy,lognormal
  run BENCH_LS_PORT=1 BENCH_MODES=damped BENCH_LS_START=raw BENCH_FAMILIES=gumbel
  run BENCH_LS_PORT=1 BENCH_MODES=damped BENCH_LS_START=locscale BENCH_FAMILIES=laplace,logistic
fi
echo DONE >&2
