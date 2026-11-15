#!/bin/zsh
# Cycle 6 batch A: anneal vs freeze (Gaussian, Gamma MS, NB MD) and StudentT PES. Uses a frozen
# copy of the bench binary so src/ can be edited meanwhile.
cd "/Volumes/Crucial X9/cs_projects/gradientlss-rs"
B=/tmp/rb_c6
export DYLD_LIBRARY_PATH=/tmp/rb_c6_lib
export BENCH_OUT=research/gradientlss/experiments_v2.csv BENCH_MODES=damped BENCH_SEEDS=5
G=kin8nm,power_plant,protein,naval,gaussian_sim,synth_t3,studentt_sim
run() { echo "== $*" >&2; env "$@" $B > /dev/null; }
# sanity: anneal=0 reproduces c5d_pes_20 bit-for-bit
run BENCH_TAG=c6_check_pes20 BENCH_FAMILIES=gaussian BENCH_DATASETS=kin8nm BENCH_PES=20:1
for spec in "20:1@0.5" "50:1@0.5"; do
  t=${spec//[:@,]/_}
  run BENCH_TAG=c6a_g_$t BENCH_FAMILIES=gaussian BENCH_DATASETS=$G BENCH_PES=$spec
  run BENCH_TAG=c6a_gm_$t BENCH_FAMILIES=gamma_ms BENCH_DATASETS=kin8nm,power_plant,naval,synth_gamma BENCH_PES=$spec
  run BENCH_TAG=c6a_nb_$t BENCH_FAMILIES=nb_md BENCH_DATASETS=bike_cnt,bike_casual,synth_nb_lo,synth_nb_hi,synth_nb_ps BENCH_PES=$spec
done
# StudentT (df, loc, scale): none / scale / df / both, freeze and anneal
for spec in 0 "20:2" "20:0" "20:0,2" "20:2@0.5" "20:0,2@0.5"; do
  t=${spec//[:@,]/_}
  run BENCH_TAG=c6b_t_$t BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_PES=$spec
done
# StudentT with the residual warm start (crate API)
for spec in 0 "20:2" "20:0,2" "20:2@0.5"; do
  t=${spec//[:@,]/_}
  run BENCH_TAG=c6c_tw_$t BENCH_FAMILIES=studentt BENCH_DATASETS=$G BENCH_WARM_START=api BENCH_PES=$spec
done
echo DONE >&2
