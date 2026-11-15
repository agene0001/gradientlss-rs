#!/bin/zsh
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c8
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5 BENCH_FAMILIES=studentt BENCH_T_START=median
G=kin8nm,power_plant,protein,naval,gaussian_sim,synth_t3,studentt_sim
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
run BENCH_TAG=c8t_df10_0 BENCH_DATASETS=$G BENCH_T_START_DF=10 BENCH_PES=0
run BENCH_TAG=c8t_dfx2_0 BENCH_DATASETS=$G BENCH_PARAM_SCALES=2,1,1 BENCH_PES=0
run BENCH_TAG=c8t_df10_pes BENCH_DATASETS=$G BENCH_T_START_DF=10 BENCH_PES="20:0,2@0.5" BENCH_PES_HOLD=1
echo DONE >&2
