#!/bin/zsh
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c9
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_FAMILIES=weibull BENCH_PES="20:1@0.5" BENCH_PES_HOLD=1
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
for K in 50 200 1000; do
  run BENCH_TAG=c9c_cap$K BENCH_SEEDS=5 BENCH_DATASETS=synth_weibull,kin8nm,naval BENCH_WB_CAP=$K
  run BENCH_TAG=c9c10_cap$K BENCH_SEEDS=10 BENCH_DATASETS=power_plant BENCH_WB_CAP=$K
done
echo DONE >&2
