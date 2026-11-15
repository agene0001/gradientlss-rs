#!/bin/zsh
# Proposal 018: family selection — marginal fit (dist_select) against held-out conditional score.
# usage: run_c14f.sh        (stderr: DSEL lines from the marginal pass, then the fits' logs)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c14f
export OMP_NUM_THREADS=3 BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_TAG=c14f
POS="BENCH_FAMILIES=gaussian,studentt,gamma_ms,weibull BENCH_DATASETS=kin8nm,power_plant,naval"
ANY="BENCH_FAMILIES=gaussian,studentt BENCH_DATASETS=protein,gaussian_sim,studentt_sim,synth_t3"
P="BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt,bike_casual"
N="BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt,bike_casual"
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
# 1. marginal fits (no training)
for S in $POS $ANY $P $N; do :; done
run BENCH_DIST_SELECT=1 ${=POS}
run BENCH_DIST_SELECT=1 ${=ANY}
run BENCH_DIST_SELECT=1 ${=P}
run BENCH_DIST_SELECT=1 ${=N}
# 2. the trained models, default recipe
run ${=POS}
run ${=ANY}
run ${=P}
run ${=N}
echo DONE >&2
