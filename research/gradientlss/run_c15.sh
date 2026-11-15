#!/bin/zsh
# Proposal 021: curvature / start arms for the five location-scale families the earlier cycles skipped.
# usage: run_c15.sh <arm: d_raw|d_loc|d_ls|o_ls>        (stderr is the log)
# (c15_base and c15_init were run by hand with the port off: BENCH_INIT=0 / 1.)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c15c
A=$1
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_PES=auto BENCH_LS_PORT=1 BENCH_TAG=c15_$A
case $A in
  d_raw) export BENCH_MODES=damped BENCH_LS_START=raw ;;
  d_loc) export BENCH_MODES=damped BENCH_LS_START=loc ;;
  d_ls)  export BENCH_MODES=damped BENCH_LS_START=locscale ;;
  o_ls)  export BENCH_MODES=observed BENCH_LS_START=locscale ;;
  *) echo "unknown arm $A" >&2; exit 2 ;;
esac
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
run BENCH_FAMILIES=cauchy,logistic,gumbel,laplace BENCH_DATASETS=kin8nm,power_plant,naval,protein,gaussian_sim
run BENCH_FAMILIES=lognormal BENCH_DATASETS=kin8nm,power_plant,naval
echo DONE >&2
