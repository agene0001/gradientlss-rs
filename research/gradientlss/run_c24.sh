#!/bin/zsh
# Proposal 026: CRPS training after the large-count fix (CRPS arm of run_c20.sh, new binary).
# usage: run_c24.sh <poisson|nb_small|nb_bike|nb_seoul>      (stderr is the log)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c24
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_LOC_START=1 BENCH_TAG=c24_crps BENCH_LOSS=crps
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
case $1 in
  poisson)  run RAYON_NUM_THREADS=1 BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_DATASETS=synth_pois,c_abalone,bike_casual,bike_cnt,c_seoulbike ;;
  nb_small) run RAYON_NUM_THREADS=1 BENCH_FAMILIES=nb_md BENCH_DATASETS=synth_nb_lo,c_abalone,bike_casual ;;
  nb_bike)  run RAYON_NUM_THREADS=4 BENCH_FAMILIES=nb_md BENCH_DATASETS=bike_cnt ;;
  nb_seoul) run RAYON_NUM_THREADS=4 BENCH_FAMILIES=nb_md BENCH_DATASETS=c_seoulbike ;;
  *) echo "unknown group $1" >&2; exit 2 ;;
esac
echo DONE >&2
