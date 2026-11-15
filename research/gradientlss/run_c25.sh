#!/bin/zsh
# Proposal 027: which axis carries the wide-space gain. Arms: depth (mcw, lambda, max_depth),
# subs (mcw, lambda, subsample, colsample_bytree), d6 (untuned default at max_depth 6).
# usage: run_c25.sh <arm: depth|subs|d6> <set: dev|unseen> <group>      (stderr is the HPO log)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c25
A=$1; S=$2; G=$3
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive RAYON_NUM_THREADS=1 BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_HNORM=off
case $A in
  depth|subs) export BENCH_HPO=20 BENCH_HPO_CHECK=10,20 BENCH_HPO_SAMPLER=tpe BENCH_HPO_SPACE=$A BENCH_TAG=c25_$A ;;
  d6) export BENCH_DEPTH=6 BENCH_TAG=c25_d6 ;;
  *) echo "unknown arm $A" >&2; exit 2 ;;
esac
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
P="BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1"
N="BENCH_FAMILIES=nb_md BENCH_LOC_START=1"
case $S-$G in
  dev-1) run BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm ;;
  dev-2) run BENCH_FAMILIES=gaussian BENCH_DATASETS=power_plant ;;
  dev-3) run BENCH_FAMILIES=gaussian BENCH_DATASETS=protein ;;
  dev-4) run BENCH_FAMILIES=gaussian BENCH_DATASETS=naval ;;
  dev-5) run BENCH_FAMILIES=gaussian BENCH_DATASETS=gaussian_sim
         run ${=P} BENCH_DATASETS=synth_pois ;;
  dev-6) run ${=P} BENCH_DATASETS=bike_cnt ;;
  dev-7) run ${=N} BENCH_DATASETS=bike_cnt ;;
  dev-8) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=naval ;;
  dev-9) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm ;;
  unseen-1) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_airfoil,c_parkinsons ;;
  unseen-2) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_gt_nox ;;
  unseen-3) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_gt_co ;;
  unseen-4) run BENCH_FAMILIES=gaussian BENCH_DATASETS=c_supercond ;;
  unseen-5) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_airfoil,c_parkinsons ;;
  unseen-6) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_gt_nox ;;
  unseen-7) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_gt_co ;;
  unseen-8) run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=c_supercond ;;
  unseen-9) run ${=N} BENCH_DATASETS=c_abalone,c_seoulbike ;;
  unseen-10) run ${=P} BENCH_DATASETS=c_abalone,c_seoulbike ;;
  # d6: every pair in one stream (untuned fits only)
  all-1) run BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm,power_plant,protein,naval,gaussian_sim,c_airfoil,c_parkinsons,c_gt_nox,c_gt_co,c_supercond
         run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=naval,kin8nm,c_airfoil,c_parkinsons,c_gt_nox,c_gt_co,c_supercond
         run ${=P} BENCH_DATASETS=synth_pois,bike_cnt,c_abalone,c_seoulbike
         run ${=N} BENCH_DATASETS=bike_cnt,c_abalone,c_seoulbike ;;
  *) echo "unknown set/group $S-$G" >&2; exit 2 ;;
esac
echo DONE >&2
