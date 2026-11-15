#!/bin/zsh
# Proposal 013 A2 (new sims, StudentT median vs 3·sd, PES off) and G (Gamma MS start units)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c10
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
NS=synth_g_sin,synth_g_exp,synth_t4_flat
run BENCH_TAG=c10a_med BENCH_FAMILIES=studentt BENCH_DATASETS=$NS BENCH_PES=0
run BENCH_TAG=c10a_s3 BENCH_FAMILIES=studentt BENCH_DATASETS=$NS BENCH_PES=0 BENCH_PRESET=t_scale:3
GM=kin8nm,power_plant,naval,synth_gamma
for C in 1 0.01 100; do
  run BENCH_TAG=c10g_raw_c$C BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$GM BENCH_PES=auto BENCH_Y_SCALE=$C
  run BENCH_TAG=c10g_wb_c$C BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$GM BENCH_PES=auto BENCH_Y_SCALE=$C BENCH_PRESET=gm_wide:0.5/0.5
done
run BENCH_TAG=c10g_wa_c1 BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$GM BENCH_PES=auto BENCH_PRESET=gm_wide:0/0
run BENCH_TAG=c10g_wm_c1 BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$GM BENCH_PES=auto BENCH_PRESET=gm_wide:-0.5/0.5
echo DONE >&2
