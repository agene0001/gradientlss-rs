#!/bin/zsh
# Proposal 014 G (Gaussian start units)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c11
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=gaussian
D=kin8nm,power_plant,protein,naval,gaussian_sim
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
for C in 1 100; do
  run BENCH_TAG=c11g_raw_c$C BENCH_Y_SCALE=$C BENCH_DATASETS=$D
  run BENCH_TAG=c11g_sd3_c$C BENCH_Y_SCALE=$C BENCH_DATASETS=$D BENCH_PRESET=g_sd:3
  run BENCH_TAG=c11g_sd1_c$C BENCH_Y_SCALE=$C BENCH_DATASETS=$D BENCH_PRESET=g_sd:1
done
echo DONE >&2
