#!/bin/zsh
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c7
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5
G=kin8nm,power_plant,protein,naval,gaussian_sim,synth_t3,studentt_sim
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
# sanity: raw start reproduces c6b_t_0
run BENCH_TAG=c7_check_raw BENCH_FAMILIES=studentt BENCH_DATASETS=kin8nm BENCH_T_RAW_START=1 BENCH_PES=0
# S: median/MAD start
run BENCH_TAG=c7s_t_ls_0 BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_PES=0
run BENCH_TAG=c7s_t_ls_ann BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_PES="20:0,2@0.5"
# H: hold on (compare with c6a_* / c7s_t_ls_ann)
run BENCH_TAG=c7h_g BENCH_FAMILIES=gaussian BENCH_DATASETS=$G BENCH_PES="20:1@0.5" BENCH_PES_HOLD=1
run BENCH_TAG=c7h_gm BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm,power_plant,naval,synth_gamma BENCH_PES="20:1@0.5" BENCH_PES_HOLD=1
run BENCH_TAG=c7h_nb BENCH_FAMILIES=nb_md BENCH_DATASETS=bike_cnt,bike_casual,synth_nb_lo,synth_nb_hi,synth_nb_ps BENCH_PES="20:1@0.5" BENCH_PES_HOLD=1
run BENCH_TAG=c7h_t BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_PES="20:0,2@0.5" BENCH_PES_HOLD=1
# N: naval at 6000 rounds
for fam in gaussian gamma_ms studentt; do
  j=1; [[ $fam == studentt ]] && j=0,2
  run BENCH_TAG=c7n_none BENCH_FAMILIES=$fam BENCH_DATASETS=naval BENCH_ROUNDS=6000 BENCH_PES=0
  run BENCH_TAG=c7n_ann BENCH_FAMILIES=$fam BENCH_DATASETS=naval BENCH_ROUNDS=6000 BENCH_PES="20:$j@0.5"
  run BENCH_TAG=c7n_hold BENCH_FAMILIES=$fam BENCH_DATASETS=naval BENCH_ROUNDS=6000 BENCH_PES="20:$j@0.5" BENCH_PES_HOLD=1
done
echo DONE >&2
