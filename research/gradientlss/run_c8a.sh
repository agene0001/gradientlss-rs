#!/bin/zsh
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c8
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5
GM=kin8nm,power_plant,naval,synth_gamma
NB=bike_cnt,bike_casual,synth_nb_lo,synth_nb_hi,synth_nb_ps
WB=synth_weibull,kin8nm,power_plant,naval
BT=synth_beta,synth_beta_hi,bike_share
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
PES="20:1@0.5"
# checks: INIT=1 through `new` must equal c7h_gm / c7h_nb
run BENCH_TAG=c8_check_gm BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm BENCH_INIT=1 BENCH_PES=$PES BENCH_PES_HOLD=1
run BENCH_TAG=c8_check_nb BENCH_FAMILIES=nb_md BENCH_DATASETS=bike_casual BENCH_INIT=1 BENCH_PES=$PES BENCH_PES_HOLD=1
# L: location starts
run BENCH_TAG=c8l_raw BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$GM BENCH_PES=$PES BENCH_PES_HOLD=1
run BENCH_TAG=c8l_loc BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$GM BENCH_PES=$PES BENCH_PES_HOLD=1 BENCH_LOC_START=1
run BENCH_TAG=c8l_raw BENCH_FAMILIES=nb_md BENCH_DATASETS=$NB BENCH_PES=$PES BENCH_PES_HOLD=1
run BENCH_TAG=c8l_loc BENCH_FAMILIES=nb_md BENCH_DATASETS=$NB BENCH_PES=$PES BENCH_PES_HOLD=1 BENCH_LOC_START=1
run BENCH_TAG=c8l_loc BENCH_FAMILIES=weibull BENCH_DATASETS=$WB BENCH_PES=0 BENCH_LOC_START=1
run BENCH_TAG=c8l_fit BENCH_FAMILIES=weibull BENCH_DATASETS=$WB BENCH_PES=0 BENCH_INIT=1
# P: per-parameter early stopping for Beta MP (log φ) and Weibull (log k)
run BENCH_TAG=c8p BENCH_FAMILIES=beta_mp BENCH_DATASETS=$BT BENCH_PES=$PES BENCH_PES_HOLD=1
run BENCH_TAG=c8p BENCH_FAMILIES=weibull BENCH_DATASETS=$WB BENCH_PES=$PES BENCH_PES_HOLD=1
echo DONE >&2
