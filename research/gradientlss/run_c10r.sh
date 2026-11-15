#!/bin/zsh
# Proposal 013 R: tail-robust stopping metric, default recipe (BENCH_PES=auto)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c10
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_PES=auto
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
for K in none 10 25; do
  if [ $K = none ]; then X=(); else X=(BENCH_STOP_CAP=$K); fi
  run BENCH_TAG=c10r_$K BENCH_SEEDS=5 BENCH_FAMILIES=weibull BENCH_DATASETS=synth_weibull,kin8nm,naval $X
  run BENCH_TAG=c10r_$K BENCH_SEEDS=10 BENCH_FAMILIES=weibull BENCH_DATASETS=power_plant $X
  run BENCH_TAG=c10r_$K BENCH_SEEDS=5 BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm,power_plant,protein,naval,gaussian_sim $X
  run BENCH_TAG=c10r_$K BENCH_SEEDS=5 BENCH_FAMILIES=studentt BENCH_DATASETS=power_plant,protein,synth_t3 $X
  if [ $K != none ]; then
    run BENCH_TAG=c10r_$K BENCH_SEEDS=5 BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm,power_plant,naval,synth_gamma $X
  fi
done
echo DONE >&2
