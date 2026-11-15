#!/bin/zsh
# Proposal 014 P (Poisson) and N (NB); hashed discrete PIT throughout
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c11
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_PIT_U=hash BENCH_PES=auto
C=synth_pois,synth_nb_ps,synth_nb_lo,bike_cnt,bike_casual
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
for L in relu exp softplus; do
  for LS in 0 1; do
    run BENCH_TAG=c11p_${L}_ls$LS BENCH_FAMILIES=poisson BENCH_MODES=observed,damped BENCH_POIS_LINK=$L BENCH_LOC_START=$LS BENCH_DATASETS=$C
  done
done
run BENCH_TAG=c11n_pes BENCH_FAMILIES=nb_md BENCH_MODES=damped BENCH_LOC_START=1 BENCH_ORACLE_COV=1 BENCH_DATASETS=$C
run BENCH_TAG=c11n_nopes BENCH_FAMILIES=nb_md BENCH_MODES=damped BENCH_LOC_START=1 BENCH_PES=0 BENCH_DATASETS=$C
echo DONE >&2
