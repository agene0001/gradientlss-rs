#!/bin/zsh
# Proposal 020: the old and new recipes (proposal 019) on datasets no earlier cycle used.
# usage: run_c17.sh        (stderr is the log)
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c17
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_SEEDS=5 BENCH_PIT_U=hash
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
C=c_airfoil,c_parkinsons,c_gt_nox,c_gt_co,c_supercond
N=c_abalone,c_seoulbike
T=c17_old
run BENCH_TAG=$T BENCH_MODES=observed BENCH_FAMILIES=gaussian BENCH_DATASETS=$C
run BENCH_TAG=$T BENCH_MODES=observed BENCH_T_START=raw BENCH_FAMILIES=studentt BENCH_DATASETS=$C
run BENCH_TAG=$T BENCH_MODES=observed BENCH_FAMILIES=gamma BENCH_DATASETS=$C
run BENCH_TAG=$T BENCH_MODES=observed BENCH_FAMILIES=nb BENCH_DATASETS=$N
run BENCH_TAG=$T BENCH_MODES=observed BENCH_FAMILIES=poisson BENCH_DATASETS=$N
T=c17_new
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=gaussian BENCH_DATASETS=$C
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=studentt BENCH_DATASETS=$C
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_FAMILIES=gamma_ms BENCH_DATASETS=$C
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_LOC_START=1 BENCH_FAMILIES=nb_md BENCH_DATASETS=$N
run BENCH_TAG=$T BENCH_MODES=damped BENCH_PES=auto BENCH_POIS_LINK=exp BENCH_LOC_START=1 BENCH_FAMILIES=poisson BENCH_DATASETS=$N
echo DONE >&2
