#!/bin/zsh
# Proposals 028 (Gumbel), 029 (NB CRPS in mean/dispersion coordinates), 030 (gradient-regularised Newton).
# usage: run_c26.sh gb <base|ls1|ls2|ls4|ls10> <dev|unseen>
#        run_c26.sh nbmd
#        run_c26.sh grn <eta: 0.05|0.3|1> <arm: van|grn|grnx>
# stderr is the log (WORST / DEBUG lines for gb).
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c26
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive RAYON_NUM_THREADS=1 BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_PES=auto BENCH_PIT_U=hash
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
case $1 in
  gb)
    export BENCH_FAMILIES=gumbel BENCH_DEBUG=1 BENCH_DEBUG_WORST=1 BENCH_TAG=c26_gb_$2
    case $2 in
      base) export BENCH_MODES=observed ;;                       # port off: Observed, Raw (today's default)
      ls*)  export BENCH_MODES=damped BENCH_LS_PORT=1 BENCH_LS_START=locscale BENCH_LS_SCALE_K=${2#ls} ;;
      *) echo "unknown arm $2" >&2; exit 2 ;;
    esac
    case $3 in
      dev)    run BENCH_DATASETS=power_plant,kin8nm,naval,protein,gaussian_sim,synth_gumbel ;;
      unseen) run BENCH_DATASETS=c_airfoil,c_parkinsons,c_gt_nox,c_gt_co,c_supercond ;;
      *) echo "unknown set $3" >&2; exit 2 ;;
    esac ;;
  nbmd)
    export BENCH_MODES=damped BENCH_LOC_START=1 BENCH_LOSS=crps BENCH_NB_CRPS_MD=1 BENCH_TAG=c26_nbmd
    run BENCH_FAMILIES=nb_md BENCH_DATASETS=synth_nb_lo,c_abalone,bike_casual,bike_cnt,c_seoulbike ;;
  grn)
    E=$2; A=$3
    export BENCH_MODES=damped BENCH_HNORM=off BENCH_ETA=$E BENCH_TAG=c26_e${E}_$A
    case $A in
      van) ;;
      grn) export BENCH_GRN=1 ;;
      grnx) export BENCH_GRN=x:1 ;;
      *) echo "unknown arm $A" >&2; exit 2 ;;
    esac
    run BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm,power_plant,protein,naval,gaussian_sim
    run BENCH_FAMILIES=gamma_ms BENCH_DATASETS=naval,kin8nm
    run BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_DATASETS=synth_pois,bike_cnt
    run BENCH_FAMILIES=nb_md BENCH_LOC_START=1 BENCH_DATASETS=bike_cnt
    run BENCH_FAMILIES=studentt BENCH_DATASETS=synth_t3,protein ;;
  *) echo "unknown experiment $1" >&2; exit 2 ;;
esac
echo DONE >&2
