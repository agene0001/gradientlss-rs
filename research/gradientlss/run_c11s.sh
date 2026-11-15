#!/bin/zsh
# Proposal 014 Addendum G′: standardized labels (Gaussian, default start), c = 1 and c = 100 (power_plant)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c11b
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=gaussian BENCH_Y_STD=1
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
run BENCH_TAG=c11s_std_c1 BENCH_DATASETS=kin8nm,power_plant,protein,naval,gaussian_sim
run BENCH_TAG=c11s_std_c100 BENCH_Y_SCALE=100 BENCH_DATASETS=power_plant
echo DONE >&2
