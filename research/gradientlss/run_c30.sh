#!/bin/zsh
# Proposal 034: max_bin at 50 trials (2d50, bin50) and the two-value choice (bin2).
# usage: run_c30.sh wave <streams> [minutes]    run every unfinished job; stop starting new ones after <minutes> (default 70)
#        run_c30.sh job <arm> <fam> <dataset> <seed>
#        run_c30.sh list                        all jobs, one per line
# One job = one (arm, family, dataset, seed); output /tmp/c30/<arm>_<fam>_<dataset>_<seed>.{csv,log,done}
D=/tmp/c30; B=/tmp/rb_c30; ME=${0:A}
G=(kin8nm power_plant protein naval gaussian_sim c_airfoil c_parkinsons c_gt_nox c_gt_co c_supercond c_n_concrete c_n_wine c_n_appliances c_n_sgemm c_n_gridstab)
M=(naval kin8nm c_airfoil c_parkinsons c_gt_nox c_gt_co c_supercond c_n_concrete c_n_wine c_n_appliances c_n_sgemm)
P=(synth_pois bike_cnt c_abalone c_seoulbike c_n_metro)
N=(bike_cnt c_abalone c_seoulbike c_n_metro c_n_news)
case $1 in
  list)
    for s in 0 1 2 3 4; do for a in bin2 2d50 bin50; do
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
    cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
    export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive RAYON_NUM_THREADS=1 BENCH_OUT=$O.csv BENCH_SEED_START=$S BENCH_SEEDS=$((S + 1)) BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_HNORM=off BENCH_HPO_SAMPLER=tpe BENCH_DATASETS=$DS
    case $A in
      2d50) export BENCH_HPO=50 BENCH_HPO_CHECK=20,50 BENCH_TAG=c30_2d ;;
      bin50) export BENCH_HPO=50 BENCH_HPO_CHECK=20,50 BENCH_HPO_SPACE=bin BENCH_TAG=c30_bin ;;
      bin2) export BENCH_HPO=20 BENCH_HPO_CHECK=10,20 BENCH_HPO_SPACE=bin2 BENCH_TAG=c30_bin2 ;;
      *) echo "unknown arm $A" >&2; exit 2 ;;
    esac
    case $F in
      gaussian) export BENCH_FAMILIES=gaussian ;;
      gamma) export BENCH_FAMILIES=gamma_ms ;;
      poisson) export BENCH_FAMILIES=poisson BENCH_POIS_LINK=exp BENCH_LOC_START=1 ;;
      nb) export BENCH_FAMILIES=nb_md BENCH_LOC_START=1 ;;
      *) echo "unknown family $F" >&2; exit 2 ;;
    esac
    $B > /dev/null 2> $O.log && [[ $(wc -l < $O.csv) -eq 3 ]] && touch $O.done ;;
  wave)
    echo $(( $(date +%s) + ${3:-70} * 60 )) > $D/deadline
    $ME list | xargs -P $2 -L 1 $ME job
    echo "done: $(/bin/ls $D | grep -c '\.done$') / $($ME list | wc -l | tr -d ' ')" ;;
esac
