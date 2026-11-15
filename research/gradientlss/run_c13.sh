#!/bin/zsh
# Proposal 016: Hessian normalisation as a tuning axis (equal-budget hyper_opt over mcw, lambda)
# usage: run_c13.sh <arm: off|marginal> <group: a|b>
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c13
H=$1; G=$2
export OMP_NUM_THREADS=3 BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash
export BENCH_HPO=20 BENCH_HPO_CHECK=5,10,20 BENCH_HNORM=$H
A=$([ $H = off ] && echo raw || echo norm)
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
if [ $G = a ]; then
  run BENCH_TAG=c13_$A BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm,power_plant,protein,naval,gaussian_sim
else
  run BENCH_TAG=c13_$A BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt,synth_pois
  run BENCH_TAG=c13_$A BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt
  run BENCH_TAG=c13_$A BENCH_FAMILIES=gamma_ms BENCH_DATASETS=naval,kin8nm
fi
echo DONE >&2
