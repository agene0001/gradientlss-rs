#!/bin/zsh
# Proposal 015: scale-relative tree regularisation (Hessian normalisation)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c12
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
C=synth_pois,synth_nb_ps,synth_nb_lo,bike_cnt,bike_casual
for H in off start marginal; do
  run BENCH_TAG=c12_$H BENCH_HNORM=$H BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm,power_plant,protein,naval,gaussian_sim
  run BENCH_TAG=c12_$H BENCH_HNORM=$H BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=$C
  run BENCH_TAG=c12_$H BENCH_HNORM=$H BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=$C
  run BENCH_TAG=c12_$H BENCH_HNORM=$H BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm,power_plant,naval,synth_gamma
done
# R0 invariance: unit-free start + marginal, c = 1 vs c = 100
for C2 in 1 100; do
  run BENCH_TAG=c12_inv_marg_c$C2 BENCH_HNORM=marginal BENCH_FAMILIES=gaussian BENCH_PRESET=g_std_equiv BENCH_Y_SCALE=$C2 BENCH_DATASETS=power_plant,kin8nm
done
echo DONE >&2
