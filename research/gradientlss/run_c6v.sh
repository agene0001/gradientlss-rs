#!/bin/zsh
# Verify that the Auto default reproduces the explicit cycle-6 arms bit-for-bit.
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c6v
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5 BENCH_PES=auto
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
run BENCH_TAG=c6v_auto BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm,protein
run BENCH_TAG=c6v_auto BENCH_FAMILIES=gamma_ms BENCH_DATASETS=naval
run BENCH_TAG=c6v_auto BENCH_FAMILIES=nb_md BENCH_DATASETS=bike_casual
run BENCH_TAG=c6v_auto BENCH_FAMILIES=studentt BENCH_DATASETS=kin8nm,protein
echo DONE >&2
