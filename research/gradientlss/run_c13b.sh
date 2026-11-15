#!/bin/zsh
# Proposal 016 Addendum N: normalised axis with mcw, lambda in [1e-3, 10]
# usage: run_c13b.sh <stream: 1|2|3|4>
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c13b
export OMP_NUM_THREADS=3 BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash
export BENCH_HPO=20 BENCH_HPO_CHECK=5,10,20 BENCH_HNORM=marginal BENCH_HPO_HIGH=10 BENCH_TAG=c13_norm10
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
case $1 in
  1) run BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm,power_plant,protein ;;
  2) run BENCH_FAMILIES=gaussian BENCH_DATASETS=naval,gaussian_sim ;;
  3) run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt,synth_pois
     run BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt ;;
  4) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=naval,kin8nm ;;
esac
echo DONE >&2
