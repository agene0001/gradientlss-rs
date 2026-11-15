#!/bin/zsh
# Proposal 014 Addendum G‴: standardized curvature, raw start held (preset g_raw_in_std)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c11d
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=gaussian BENCH_Y_STD=1 BENCH_PRESET=g_raw_in_std
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
run BENCH_TAG=c11c_C BENCH_DATASETS=kin8nm,power_plant,protein,naval,gaussian_sim
run BENCH_TAG=c11c_C6k BENCH_DATASETS=naval BENCH_ROUNDS=6000
echo DONE >&2
