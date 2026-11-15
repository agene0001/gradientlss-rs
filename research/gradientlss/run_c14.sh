#!/bin/zsh
# Proposal 017: sampler control and defaults-first (raw axis, cycle 13's protocol otherwise)
# usage: run_c14.sh <arm: rand|dtpe|ds10> <group: a|b>      (stderr is the HPO log)
# One XGBoost thread per stream: trial lines are bit-identical to the 3-thread cycle 13 run
# (checked on synth_pois and kin8nm), and under load it is faster.
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c14
A=$1; G=$2
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash
export BENCH_HPO=20 BENCH_HPO_CHECK=5,10,20 BENCH_HNORM=off
case $A in
  rand) export BENCH_HPO_SAMPLER=random ;;
  dtpe) export BENCH_HPO_SAMPLER=tpe BENCH_HPO_INIT=default ;;
  ds10) export BENCH_HPO_SAMPLER=startup:10 BENCH_HPO_INIT=default ;;
  *) echo "unknown arm $A" >&2; exit 2 ;;
esac
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
if [ $G = a ]; then
  run BENCH_TAG=c14_$A BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm,power_plant,protein,naval,gaussian_sim
else
  run BENCH_TAG=c14_$A BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt,synth_pois
  run BENCH_TAG=c14_$A BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt
  run BENCH_TAG=c14_$A BENCH_FAMILIES=gamma_ms BENCH_DATASETS=naval,kin8nm
fi
echo DONE >&2
