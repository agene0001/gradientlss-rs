#!/bin/zsh
# Proposal 026: verification of the applied decisions.
# usage: run_c23.sh <default_unseen|default_dev|dsel>      (stderr is the log)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c23
A=$1
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive RAYON_NUM_THREADS=1 BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_PES=auto BENCH_MODES=damped
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
case $A in
  default_unseen)
    export BENCH_TAG=c23_default BENCH_LS_PORT=2
    run BENCH_FAMILIES=cauchy,logistic,laplace,lognormal BENCH_DATASETS=c_airfoil,c_parkinsons,c_gt_nox,c_gt_co,c_supercond ;;
  default_dev)
    export BENCH_TAG=c23_default BENCH_LS_PORT=2
    run BENCH_FAMILIES=cauchy,logistic,laplace BENCH_DATASETS=kin8nm,power_plant,naval,protein,gaussian_sim
    run BENCH_FAMILIES=lognormal BENCH_DATASETS=kin8nm,power_plant,naval ;;
  dsel)
    # the marginal pass of run_c14f.sh (no training); DSEL lines on stderr
    export BENCH_TAG=c23_dsel BENCH_DIST_SELECT=1 BENCH_PIT_U=hash
    run BENCH_FAMILIES=gaussian,studentt,gamma_ms,weibull BENCH_DATASETS=kin8nm,power_plant,naval
    run BENCH_FAMILIES=gaussian,studentt BENCH_DATASETS=protein,gaussian_sim,studentt_sim,synth_t3
    run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt,bike_casual
    run BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt,bike_casual ;;
  *) echo "unknown arm $A" >&2; exit 2 ;;
esac
echo DONE >&2
