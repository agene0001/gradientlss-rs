#!/bin/zsh
# Proposal 025: wide search space on the unseen sets (confirmation of 024). Reference: c19_tpe (two axes).
# usage: run_c22.sh <group: 1..10>      (stderr is the HPO log)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c20
G=$1
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive RAYON_NUM_THREADS=1 BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_HNORM=off
export BENCH_HPO=20 BENCH_HPO_CHECK=10,20 BENCH_HPO_SAMPLER=tpe BENCH_HPO_SPACE=wide BENCH_TAG=c22_wide
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
case $G in
  1) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_airfoil,c_parkinsons ;;
  2) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_gt_nox ;;
  3) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_gt_co ;;
  4) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_supercond ;;
  5) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_airfoil,c_parkinsons ;;
  6) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_gt_nox ;;
  7) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_gt_co ;;
  8) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_supercond ;;
  9) run BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=c_abalone,c_seoulbike ;;
  10) run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=c_abalone,c_seoulbike ;;
esac
echo DONE >&2
