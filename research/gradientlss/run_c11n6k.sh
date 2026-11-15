#!/bin/zsh
# Proposal 014 Addendum G″: naval at 6000 rounds — native raw / standardized / native with B's start
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c11c
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=gaussian BENCH_DATASETS=naval BENCH_ROUNDS=6000
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
run BENCH_TAG=c11h_A_native
run BENCH_TAG=c11h_B_std BENCH_Y_STD=1
run BENCH_TAG=c11h_D_native_equiv BENCH_PRESET=g_std_equiv
echo DONE >&2
