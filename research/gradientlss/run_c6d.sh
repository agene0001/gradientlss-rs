#!/bin/zsh
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c6
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5
G=kin8nm,power_plant,protein,naval,gaussian_sim,synth_t3,studentt_sim
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
run BENCH_TAG=c6d_g_20_1_0.25 BENCH_FAMILIES=gaussian BENCH_DATASETS=$G BENCH_PES="20:1@0.25"
run BENCH_TAG=c6d_gm_20_1_0.25 BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm,power_plant,naval,synth_gamma BENCH_PES="20:1@0.25"
run BENCH_TAG=c6d_nb_20_1_0.25 BENCH_FAMILIES=nb_md BENCH_DATASETS=bike_cnt,bike_casual,synth_nb_lo,synth_nb_hi,synth_nb_ps BENCH_PES="20:1@0.25"
run BENCH_TAG=c6d_t_20_2_0.25 BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_PES="20:2@0.25"
for spec in 0 "20:2" "20:2@0.5" "20:2@0.25" "20:0,2@0.5"; do
  t=${spec//[:@,]/_}
  run BENCH_TAG=c6e_ti_$t BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_INIT=1 BENCH_PES=$spec
done
echo DONE >&2
