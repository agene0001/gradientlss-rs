#!/bin/zsh
# Proposal 019: do the cycle 1-13 defaults transfer to LightGBM? old vs new recipe on both backends.
# usage: run_c16.sh <backend: xgb|lgb>        (stderr is the log)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
K=$1
B=$([ $K = lgb ] && echo /tmp/rb_c16_lgbm || echo /tmp/rb_c15)
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_PIT_U=hash
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
G=kin8nm,power_plant,protein,naval,gaussian_sim
# old: the recipe before cycle 1
T=c16_${K}_old
run BENCH_TAG=$T BENCH_MODES=observed BENCH_FAMILIES=gaussian BENCH_DATASETS=$G
run BENCH_TAG=$T BENCH_MODES=observed BENCH_T_START=raw BENCH_FAMILIES=studentt BENCH_DATASETS=$G,studentt_sim,synth_t3
run BENCH_TAG=$T BENCH_MODES=observed BENCH_FAMILIES=gamma BENCH_DATASETS=kin8nm,naval,power_plant,synth_gamma
run BENCH_TAG=$T BENCH_MODES=observed BENCH_FAMILIES=nb BENCH_DATASETS=bike_cnt,bike_casual,synth_nb_lo
run BENCH_TAG=$T BENCH_MODES=observed BENCH_FAMILIES=poisson BENCH_DATASETS=bike_cnt,synth_pois
# new: today's defaults
T=c16_${K}_new
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=gaussian BENCH_DATASETS=$G
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=studentt BENCH_DATASETS=$G,studentt_sim,synth_t3
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm,naval,power_plant,synth_gamma
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_LOC_START=1 BENCH_FAMILIES=nb_md BENCH_DATASETS=bike_cnt,bike_casual,synth_nb_lo
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_FAMILIES=poisson BENCH_DATASETS=bike_cnt,synth_pois
echo DONE >&2
