#!/bin/zsh
# Proposal 038: max_bin at 50 trials on the third batch of fresh datasets (c_m_*).
# usage: run_c34.sh wave <streams> [minutes] [arm ...]   run every unfinished job (of the named arms); stop starting new ones after <minutes> (default 70)
#        run_c34.sh job <arm> <fam> <dataset> <seed>
#        run_c34.sh list [arm ...]                       jobs, one per line
# One job = one (arm, family, dataset, seed); output /tmp/c34/<arm>_<fam>_<dataset>_<seed>.{csv,log,done}
D=/tmp/c34; B=/tmp/rb_c30; ME=${0:A}
ARMS=(def 2d50 bin50)
G=(c_m_energy c_m_steel c_m_fishtox c_m_tempbias c_m_video)
M=(c_m_energy c_m_steel c_m_tempbias c_m_video)
P=(c_m_pm25)
N=(c_m_pm25 c_m_fbcomments)
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
    cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
    export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive RAYON_NUM_THREADS=1 BENCH_OUT=$O.csv BENCH_SEED_START=$S BENCH_SEEDS=$((S + 1)) BENCH_MODES=damped BENCH_PES=auto BENCH_PIT_U=hash BENCH_HNORM=off BENCH_HPO_SAMPLER=tpe BENCH_DATASETS=$DS
    NR=3
    case $A in
      def) export BENCH_TAG=c34_def; NR=2 ;;
      2d50) export BENCH_HPO=50 BENCH_HPO_CHECK=20,50 BENCH_TAG=c34_2d ;;
      bin50) export BENCH_HPO=50 BENCH_HPO_CHECK=20,50 BENCH_HPO_SPACE=bin BENCH_TAG=c34_bin ;;
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
    $ME list "$@" | xargs -P $P -L 1 $ME job
    for a in ${@:-$ARMS}; do echo "$a: $(/bin/ls $D | grep -c "^${a}_.*\.done$") / 60"; done ;;
esac
