#!/bin/zsh
# Proposal 022: validation guard for tuned models, on the unseen sets of proposal 020 (+ a 5-fold mechanism arm).
# usage: run_c19.sh <arm: tpe|dtpe|def|tpe5> <group: 1|2|3|4>      (stderr is the HPO log)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c15c
A=$1; G=$2
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_HNORM=off
case $A in
  tpe)  export BENCH_HPO=20 BENCH_HPO_CHECK=5,10,20 BENCH_HPO_SAMPLER=tpe ;;
  dtpe) export BENCH_HPO=20 BENCH_HPO_CHECK=5,10,20 BENCH_HPO_SAMPLER=tpe BENCH_HPO_INIT=default ;;
  tpe5) export BENCH_HPO=20 BENCH_HPO_CHECK=5,10,20 BENCH_HPO_SAMPLER=tpe BENCH_HPO_FOLDS=5 ;;
  def)  ;;
  *) echo "unknown arm $A" >&2; exit 2 ;;
esac
export BENCH_TAG=c19_$A
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
if [ $A = tpe5 ]; then
  case $G in
    1) run BENCH_FAMILIES=gaussian BENCH_DATASETS=naval ;;
    2) run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt ;;
  esac
else
  case $G in
    1) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_airfoil,c_parkinsons,c_gt_nox,c_gt_co ;;
    2) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_supercond ;;
    3) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_airfoil,c_parkinsons,c_gt_nox,c_gt_co,c_supercond ;;
    5) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_supercond ;;   # group 3 was cut at the 2 h task limit after seed 1
    4) run BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=c_abalone,c_seoulbike
       run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=c_abalone,c_seoulbike ;;
  esac
fi
echo DONE >&2
