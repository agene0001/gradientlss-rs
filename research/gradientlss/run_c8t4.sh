#!/bin/zsh
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c8b
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5 BENCH_FAMILIES=studentt BENCH_DATASETS=gaussian_sim BENCH_PES=0
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
for sh in -9.5 20; do
  run BENCH_TAG=c8t4_raw_$sh BENCH_Y_SHIFT=$sh BENCH_T_START=raw
  run BENCH_TAG=c8t4_med_$sh BENCH_Y_SHIFT=$sh BENCH_T_START=median
done
echo DONE >&2
