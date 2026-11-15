#!/bin/zsh
# Proposal 031: identity checks for the applied decisions (new defaults = the tested recipes).
# usage: run_c28.sh <gb|nb|nb_tcp>      each stream writes its own CSV /tmp/c28_<name>.csv
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c28
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive BENCH_OUT=/tmp/c28_$1.csv BENCH_SEEDS=5 BENCH_PES=auto BENCH_PIT_U=hash
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
case $1 in
  gb) # Gumbel::new as is
    export RAYON_NUM_THREADS=1 BENCH_FAMILIES=gumbel BENCH_MODES=damped BENCH_LS_PORT=2 BENCH_TAG=c28_gb_default
    run BENCH_DATASETS=power_plant,kin8nm,naval,protein,gaussian_sim,synth_gumbel,c_airfoil,c_parkinsons,c_gt_nox,c_gt_co,c_supercond ;;
  nb|nb_tcp) # NB under the CRPS loss: the default head, and the opt-out
    export RAYON_NUM_THREADS=4 BENCH_MODES=damped BENCH_LOC_START=1 BENCH_LOSS=crps BENCH_FAMILIES=nb_md
    if [ $1 = nb_tcp ]; then export BENCH_NB_CRPS_TCP=1 BENCH_TAG=c28_nb_crps_tcp; else export BENCH_TAG=c28_nb_crps; fi
    run BENCH_DATASETS=synth_nb_lo,c_abalone,bike_casual,bike_cnt,c_seoulbike ;;
  *) echo "unknown $1" >&2; exit 2 ;;
esac
echo DONE >&2
