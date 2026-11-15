#!/bin/zsh
# Proposal 032: the subsampling axes on fresh datasets (c_n_*). Arms: def (untuned), 2d (mcw, lambda), subs (+ subsample, colsample_bytree).
# usage: run_c27.sh <arm: def|2d|subs> <group: 1..8>      (stderr is the HPO log)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c25
A=$1; G=$2
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive RAYON_NUM_THREADS=1 BENCH_OUT=/tmp/c27_${A}_$G.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_HNORM=off BENCH_TAG=c27_$A
case $A in
  def) ;;
  2d)   export BENCH_HPO=20 BENCH_HPO_CHECK=10,20 BENCH_HPO_SAMPLER=tpe ;;
  subs) export BENCH_HPO=20 BENCH_HPO_CHECK=10,20 BENCH_HPO_SAMPLER=tpe BENCH_HPO_SPACE=subs ;;
  *) echo "unknown arm $A" >&2; exit 2 ;;
esac
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
case $G in
  1) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_n_concrete,c_n_wine ;;
  2) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_n_appliances ;;
  3) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_n_sgemm,c_n_gridstab ;;
  4) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_n_concrete,c_n_wine ;;
  5) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_n_appliances ;;
  6) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_n_sgemm ;;
  7) run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=c_n_metro
     run BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=c_n_metro ;;
  8) run BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=c_n_news ;;
  *) echo "unknown group $G" >&2; exit 2 ;;
esac
echo DONE >&2
