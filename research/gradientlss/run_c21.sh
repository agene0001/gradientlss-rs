#!/bin/zsh
# Proposal 024: wide search space (mcw, lambda, max_depth, subsample, colsample_bytree) at 20 trials.
# usage: run_c21.sh <group: 1..10>      (stderr is the HPO log)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c20
G=$1
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_HNORM=off
export BENCH_HPO=20 BENCH_HPO_CHECK=10,20 BENCH_HPO_SAMPLER=tpe BENCH_HPO_SPACE=wide BENCH_TAG=c21_wide
[ -n "$RAYON1" ] && export RAYON_NUM_THREADS=1
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
case $G in
  1) run BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm ;;
  2) run BENCH_FAMILIES=gaussian BENCH_DATASETS=power_plant ;;
  3) run BENCH_FAMILIES=gaussian BENCH_DATASETS=protein ;;
  4) run BENCH_FAMILIES=gaussian BENCH_DATASETS=naval ;;
  5) run BENCH_FAMILIES=gaussian BENCH_DATASETS=gaussian_sim
     run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=synth_pois ;;
  6) run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt ;;
  7) run BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt ;;
  8) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=naval ;;
  9) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm ;;
esac
echo DONE >&2
