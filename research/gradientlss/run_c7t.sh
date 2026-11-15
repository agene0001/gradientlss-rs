#!/bin/zsh
# StudentT arms of H and N rerun with the median start (the c7h_t / c7n studentt rows used median+MAD).
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c7b
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5
G=kin8nm,power_plant,protein,naval,gaussian_sim,synth_t3,studentt_sim
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
run BENCH_TAG=c7h_tmed BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_T_START=median BENCH_PES="20:0,2@0.5" BENCH_PES_HOLD=1
run BENCH_TAG=c7n_none_tmed BENCH_FAMILIES=studentt BENCH_DATASETS=naval BENCH_ROUNDS=6000 BENCH_T_START=median BENCH_PES=0
run BENCH_TAG=c7n_hold_tmed BENCH_FAMILIES=studentt BENCH_DATASETS=naval BENCH_ROUNDS=6000 BENCH_T_START=median BENCH_PES="20:0,2@0.5" BENCH_PES_HOLD=1
echo DONE >&2
