#!/bin/zsh
# Cycle 7: Beta / Weibull curvature and the Beta MeanPrecision head (proposal 010).
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c7d
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_PES=0
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
# run BENCH_TAG=c7b BENCH_FAMILIES=beta,beta_mp BENCH_MODES=observed,damped BENCH_DATASETS=synth_beta,synth_beta_hi,bike_share
run BENCH_TAG=c7w2 BENCH_FAMILIES=weibull BENCH_MODES=observed,damped BENCH_DATASETS=synth_weibull,kin8nm,power_plant,naval
echo DONE >&2
