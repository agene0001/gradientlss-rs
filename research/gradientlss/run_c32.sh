#!/bin/zsh
# Proposal 036: LightGBM's own parameters, one search axis at a time (LightGBM build of the bench).
# usage: run_c32.sh wave <streams> [minutes] [arm ...]   run every unfinished job (of the named arms); stop starting new ones after <minutes> (default 70)
#        run_c32.sh job <arm> <fam> <dataset> <seed>
#        run_c32.sh list [arm ...]                       jobs, one per line
# One job = one (arm, family, dataset, seed); output /tmp/c32/<arm>_<fam>_<dataset>_<seed>.{csv,log,done}
D=/tmp/c32; B=/tmp/rb_c32_lgbm; ME=${0:A}
ARMS=(l_def l_2d lleaves lmdl lff lbag)
G=(kin8nm power_plant protein naval gaussian_sim c_airfoil c_parkinsons c_gt_nox c_gt_co c_supercond c_n_concrete c_n_wine c_n_appliances c_n_sgemm c_n_gridstab)
M=(naval kin8nm c_airfoil c_parkinsons c_gt_nox c_gt_co c_supercond c_n_concrete c_n_wine c_n_appliances c_n_sgemm)
P=(synth_pois bike_cnt c_abalone c_seoulbike c_n_metro)
N=(bike_cnt c_abalone c_seoulbike c_n_metro c_n_news)
case $1 in
  list)
    shift; [[ $# -gt 0 ]] || set -- $ARMS
    for a in "$@"; do for s in 0 1 2 3 4; do
      for d in $G; do echo $a gaussian $d $s; done
      for d in $M; do echo $a gamma $d $s; done
      for d in $P; do echo $a poisson $d $s; done
      for d in $N; do echo $a nb $d $s; done
    done; done ;;
  job)
    A=$2; F=$3; DS=$4; S=$5; O=$D/${A}_${F}_${DS}_$S
    [[ -e $O.done ]] && exit 0
    [[ -e $D/deadline && $(date +%s) -gt $(<$D/deadline) ]] && exit 0
    rm -f $O.csv
    # The data lives on an external drive that can drop out: wait for it rather than fail.
    until [[ -d "/Volumes/Crucial X9/cs_projects/gradientlss-rs/research/gradientlss/data_confirm" ]]; do /bin/sleep 30; done
    cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
    export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive RAYON_NUM_THREADS=1 BENCH_OUT=$O.csv BENCH_SEED_START=$S BENCH_SEEDS=$((S + 1)) BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_HNORM=off BENCH_HPO_SAMPLER=tpe BENCH_DATASETS=$DS
    NR=3
    case $A in
      l_def) export BENCH_TAG=c32_$A; NR=2 ;;
      l_2d) export BENCH_HPO=20 BENCH_HPO_CHECK=10,20 BENCH_TAG=c32_$A ;;
      lleaves|lmdl|lff|lbag) export BENCH_HPO=20 BENCH_HPO_CHECK=10,20 BENCH_HPO_SPACE=$A BENCH_TAG=c32_$A ;;
      *) echo "unknown arm $A" >&2; exit 2 ;;
    esac
    case $F in
      gaussian) export BENCH_FAMILIES=gaussian ;;
      gamma) export BENCH_FAMILIES=gamma_ms ;;
      poisson) export BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 ;;
      nb) export BENCH_FAMILIES=nb_md BENCH_LOC_START=1 ;;
      *) echo "unknown family $F" >&2; exit 2 ;;
    esac
    $B > /dev/null 2> $O.log && [[ $(wc -l < $O.csv) -eq $NR ]] && touch $O.done ;;
  wave)
    P=$2; MIN=${3:-70}; shift 3 2>/dev/null || shift $#
    mkdir -p $D; echo $(( $(date +%s) + MIN * 60 )) > $D/deadline
    cp $ME $D/run.sh; chmod +x $D/run.sh  # run the jobs from a copy off the external drive, so a disconnect does not end the wave
    $D/run.sh list "$@" | xargs -P $P -L 1 $D/run.sh job
    for a in ${@:-$ARMS}; do echo "$a: $(/bin/ls $D | grep -c "^${a}_.*\.done$") / 180"; done ;;
esac
