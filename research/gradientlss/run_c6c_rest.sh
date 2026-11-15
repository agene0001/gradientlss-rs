#!/bin/zsh
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c6
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5
G=kin8nm,power_plant,protein,naval,gaussian_sim,synth_t3,studentt_sim
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
for spec in "20:0,2" "20:2@0.5" "20:0,2@0.5"; do
  t=${spec//[:@,]/_}
  run BENCH_TAG=c6c_tw_$t BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_WARM_START=api BENCH_PES=$spec
done
echo DONE_C >&2
