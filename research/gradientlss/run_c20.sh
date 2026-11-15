#!/bin/zsh
# Proposal 023: CRPS training vs NLL training for Poisson and NB.   usage: run_c20.sh <arm: nll|crps> <fam: poisson|nb>
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c20
A=$1; F=$2
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive RAYON_NUM_THREADS=1 BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_LOC_START=1 BENCH_TAG=c20_$A BENCH_LOSS=$A
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
if [ $F = poisson ]; then
  run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_DATASETS=synth_pois,c_abalone,bike_casual,bike_cnt,c_seoulbike
else
  run BENCH_FAMILIES=nb_md BENCH_DATASETS=synth_nb_lo,c_abalone,bike_casual,bike_cnt,c_seoulbike
fi
echo DONE >&2
