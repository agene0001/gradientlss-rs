#!/bin/zsh
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c9
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5
G=kin8nm,power_plant,protein,naval,gaussian_sim,synth_t3,studentt_sim
GM=kin8nm,power_plant,naval,synth_gamma
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
# G check: preset of the fitted start ≡ initialize=true (c7h_gm)
run BENCH_TAG=c9_check_gmfit BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm BENCH_PRESET=gm_fit_shift:0 BENCH_PES="20:1@0.5" BENCH_PES_HOLD=1
# W: StudentT (median start default), PES off
run BENCH_TAG=c9w_wu20 BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_PES=0 BENCH_WARMUP=20
run BENCH_TAG=c9w_wu50 BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_PES=0 BENCH_WARMUP=50
run BENCH_TAG=c9w_s3 BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_PES=0 BENCH_PRESET=t_scale:3
# G: Gamma MS start anatomy, default PES
run BENCH_TAG=c9g_rmfs BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$GM BENCH_PRESET=gm_rawmean_fitshape BENCH_PES="20:1@0.5" BENCH_PES_HOLD=1
run BENCH_TAG=c9g_fs46 BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$GM BENCH_PRESET=gm_fit_shift:-4.6 BENCH_PES="20:1@0.5" BENCH_PES_HOLD=1
run BENCH_TAG=c9g_fitwu BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$GM BENCH_PRESET=gm_fit_shift:0 BENCH_WARMUP=50 BENCH_PES="20:1@0.5" BENCH_PES_HOLD=1
echo DONE >&2
