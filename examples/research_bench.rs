//! Out-of-sample probabilistic benchmark harness for algorithm research.
//!
//! Trains GradientLSS (XGBoost backend) on local CSV datasets, holding out a
//! test split, early-stopping on a separate validation split, and reports
//! proper scores (mean NLL, closed-form CRPS), PIT-based calibration
//! (central-interval coverage, KS distance to uniform), point error, and
//! failure counts. Every (dataset, family, hessian-mode, seed) cell is one
//! CSV row appended to `research/gradientlss/experiments.csv`; a paired
//! summary (Fisher − Observed per seed) is printed at the end.
//!
//! Run (release, xgboost):
//!   cargo run --release --features xgboost --example research_bench
//! Knobs (env): BENCH_DATASETS=studentt_sim,gaussian_sim,synth_t3,kin8nm,power_plant,naval,protein
//!   BENCH_FAMILIES=studentt,gaussian  BENCH_MODES=observed,fisher  BENCH_SEEDS=5  BENCH_SEED_START=0
//!   BENCH_ROUNDS=2000 BENCH_ES=50 (0 = no early stopping) BENCH_ETA=0.05 BENCH_DEPTH=4 BENCH_MCW=1
//!   BENCH_STAB=none|mad|l2  BENCH_INIT=0|1 (L-BFGS start values)
//!   BENCH_PROTEIN_N=12000 (subsample), BENCH_OUT=<csv path>, BENCH_SELFTEST=1
//!   BENCH_DATASETS may name held-out confirmation sets as c_<name> (research/gradientlss/data_confirm/<name>.csv):
//!     c_airfoil, c_supercond, c_parkinsons, c_gt_nox, c_gt_co (continuous, positive), c_abalone, c_seoulbike (counts)
//!   BENCH_TAG=<free text recorded in the CSV>
//!   BENCH_PARAM_SCALES=c0,c1[,c2]  per-parameter gradient multipliers (effective per-parameter η)
//!   BENCH_BETA_NLL=β  β-NLL row weighting;  BENCH_FREEZE=j:round[,...]  per-parameter stopping rounds
//!   BENCH_T_START=raw|median|mad  StudentT start rule without initialize (default: the crate's); BENCH_T_START_DF=ν
//!   BENCH_WARMUP=n location-only warm-up rounds; BENCH_PRESET=t_scale:c|gm_fit_shift:δ|gm_rawmean_fitshape
//!   BENCH_WB_CAP=K  Weibull shape cap;  BENCH_DEBUG_WORST=1  print the worst test rows
//!   BENCH_Y_SHIFT=c  add c to every label (translation probe); BENCH_Y_SCALE=c multiply every label by c
//!     (oracle dropped; compare NLL after subtracting ln c)
//!   BENCH_STOP_CAP=K  tail-robust stopping metric (GradientLSS::with_stopping_row_cap)
//!   BENCH_PRESET=gm_wide:dm/ls  Gamma MS [ln ȳ + dm, ls]
//!   BENCH_LOC_START=1  gamma_ms / nb_md / weibull: location-only start without initialize
//!   BENCH_POIS_LINK=relu|exp|softplus  poisson family link; BENCH_PIT_U=hash  hashed discrete-PIT uniform
//!   BENCH_ORACLE_COV=1  print the oracle's coverage (NB oracles);  BENCH_PRESET=g_sd:c  Gaussian [ȳ, ln(c·sd)]; g_std_equiv  [ȳ+0.5·sd, ln(1.65·sd)]; g_raw_in_std (with BENCH_Y_STD) the raw start in std units
//!   BENCH_Y_STD=1  standardize labels by the training-split mean/sd (prints YSTD …; add ln_sd to NLL)
//!   BENCH_HNORM=start|marginal  scale-relative tree regularisation (GradientLSS::with_hessian_normalization)
//!   BENCH_HPO=n  tune min_child_weight & lambda (log [1e-3, 1e3], TPE, 3-fold CV on train, n trials) per cell,
//!     then refit the best-so-far params at BENCH_HPO_CHECK=5,10,20 trials (BENCH_HPO_HIGH = range top); rows tagged <tag>_k<k>; HPO lines on stderr
//!   BENCH_HPO_SAMPLER=tpe|random|startup:<n>  hyper_opt sampler (default tpe);  BENCH_HPO_INIT=default  score the
//!     untuned (mcw = BENCH_MCW, lambda = 1) configuration as trial 0;  BENCH_HPO_CHECK= (empty) searches without refits
//!   BENCH_GM_RAW=1  gamma_ms: the pre-cycle-10 raw [0.5, 0.5] start (default: unit-free [ln ȳ+0.5, 0.5])
//!   BENCH_PES_HOLD=1  explicit BENCH_PES configs hold the global early stopping on each trigger
//!   BENCH_PES=window[:j,j][/patience][@anneal] | auto  per-parameter early stopping (freeze log on stderr as PES_FREEZE)
//!   BENCH_FAMILIES may include gamma, gamma_ms, nb, nb_md, beta, beta_mp, weibull,
//!     cauchy, logistic, gumbel, laplace, lognormal (observed curvature, raw start: the defaults before
//!     2026-11-13; Cauchy has no CRPS). BENCH_LS_PORT=1 makes BENCH_MODES apply to them and reads
//!     BENCH_LS_START=raw|loc|locscale (proposal 021); BENCH_LS_PORT=2 builds them with `new` as is
//!     (beta*: synth_beta, synth_beta_hi, bike_share; weibull: positive targets, synth_weibull)
//!   BENCH_WARM_START=1|api (StudentT only): two-stage residual warm start (api = crate method) — a Gaussian fit supplies
//!     per-row location offsets, a t-fit on its training residuals supplies (ν₀, σ₀); the StudentT
//!     is then trained/predicted with those offsets (BENCH_T_START_DF_CAP caps ν₀ as usual;
//!     BENCH_WARM_RESID=valid fits (ν₀, σ₀) on validation instead of training residuals).

use gradientlss::backend::{BackendDataset, BackendParams, CallbackList, ParamValue};
use gradientlss::distributions::{
    Beta as BetaDist, BetaParameterization, Distribution, Gamma, GammaParameterization, Gaussian, HessianMode,
    NbParameterization, NegativeBinomial, StudentT, Weibull as WeibullDist,
};
use rand_distr::Poisson as RandPoisson;
use rand_distr::Gamma as RandGammaD;
use gradientlss::prelude::*;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as _, StudentT as RandT, Uniform};
use statrs::distribution::{
    Beta as StatBeta, Continuous, ContinuousCDF, Gamma as StatGamma, Normal, StudentsT, Weibull as StatWeibull,
};
use statrs::function::gamma::ln_gamma;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// The backend under test. `--features xgboost` (every cycle so far) trains with XGBoost; building with
// `--features lightgbm` alone swaps in LightGBM with the same booster parameter names (the LightGBM
// backend accepts eta / max_depth / min_child_weight / lambda as aliases).
#[cfg(feature = "xgboost")]
type Bk = gradientlss::backend::XGBoostBackend;
#[cfg(all(feature = "lightgbm", not(feature = "xgboost")))]
type Bk = gradientlss::backend::LightGBMBackend;

// ---------------------------------------------------------------- config

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Family {
    Gaussian,
    StudentT,
    Gamma,
    /// Gamma boosted in the mean/shape (log µ, log α) coordinates.
    GammaMs,
    /// NegativeBinomial, default (total_count, probs) head.
    Nb,
    /// NegativeBinomial, MeanDispersion (log µ, log θ) head.
    NbMd,
    /// Beta, (concentration1, concentration0) head.
    Beta,
    /// Beta, MeanPrecision (logit µ, log φ) head.
    BetaMp,
    Weibull,
    /// Poisson (rate); link from BENCH_POIS_LINK=relu|exp|softplus (default relu, the crate's).
    Poisson,
    /// Location-scale families the first 13 cycles never measured (loc Identity, scale Exp).
    Cauchy,
    Logistic,
    Gumbel,
    Laplace,
    /// LogNormal (loc, scale of ln y); positive targets only.
    LogNormal,
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Family::Gaussian => "gaussian",
            Family::StudentT => "studentt",
            Family::Gamma => "gamma",
            Family::GammaMs => "gamma_ms",
            Family::Nb => "nb",
            Family::NbMd => "nb_md",
            Family::Beta => "beta",
            Family::BetaMp => "beta_mp",
            Family::Weibull => "weibull",
            Family::Poisson => "poisson",
            Family::Cauchy => "cauchy",
            Family::Logistic => "logistic",
            Family::Gumbel => "gumbel",
            Family::Laplace => "laplace",
            Family::LogNormal => "lognormal",
        }
    }
}

struct Config {
    datasets: Vec<String>,
    families: Vec<Family>,
    modes: Vec<HessianMode>,
    seeds: u64,
    rounds: usize,
    es: usize,
    eta: f64,
    depth: i64,
    mcw: f64,
    stab: Stabilization,
    init: bool,
    protein_n: usize,
    out: PathBuf,
    tag: String,
}

fn config() -> Config {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("research/gradientlss");
    let families = env_list("BENCH_FAMILIES", "studentt,gaussian")
        .into_iter()
        .map(|f| match f.as_str() {
            "gaussian" => Family::Gaussian,
            "studentt" | "student_t" | "t" => Family::StudentT,
            "gamma" => Family::Gamma,
            "gamma_ms" => Family::GammaMs,
            "nb" => Family::Nb,
            "nb_md" => Family::NbMd,
            "beta" => Family::Beta,
            "beta_mp" => Family::BetaMp,
            "weibull" => Family::Weibull,
            "poisson" => Family::Poisson,
            "cauchy" => Family::Cauchy,
            "logistic" => Family::Logistic,
            "gumbel" => Family::Gumbel,
            "laplace" => Family::Laplace,
            "lognormal" | "log_normal" => Family::LogNormal,
            other => panic!("unknown family {other}"),
        })
        .collect();
    let modes = env_list("BENCH_MODES", "observed,fisher")
        .into_iter()
        .map(|m| match m.as_str() {
            "observed" => HessianMode::Observed,
            "fisher" => HessianMode::Fisher,
            "damped" => HessianMode::Damped,
            other => panic!("unknown hessian mode {other}"),
        })
        .collect();
    let stab = match env_list("BENCH_STAB", "none")[0].as_str() {
        "none" => Stabilization::None,
        "mad" => Stabilization::Mad,
        "l2" => Stabilization::L2,
        other => panic!("unknown stabilization {other}"),
    };
    Config {
        datasets: env_list(
            "BENCH_DATASETS",
            "studentt_sim,gaussian_sim,synth_t3,kin8nm,power_plant,naval,protein",
        ),
        families,
        modes,
        seeds: env_or("BENCH_SEEDS", 5u64),
        rounds: env_or("BENCH_ROUNDS", 2000usize),
        es: env_or("BENCH_ES", 50usize),
        eta: env_or("BENCH_ETA", 0.05f64),
        depth: env_or("BENCH_DEPTH", 4i64),
        mcw: env_or("BENCH_MCW", 1.0f64),
        stab,
        init: env_or("BENCH_INIT", 0u8) == 1,
        protein_n: env_or("BENCH_PROTEIN_N", 12_000usize),
        out: std::env::var("BENCH_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("experiments_v2.csv")),
        tag: std::env::var("BENCH_TAG").unwrap_or_default(),
    }
}

// ---------------------------------------------------------------- data

/// A dataset with features, target, and (for simulated data) the oracle
/// per-row parameters so the irreducible NLL/CRPS can be reported.
struct Data {
    x: Array2<f64>,
    y: Array1<f64>,
    /// Oracle (loc, scale, Option<df>) per row, when the DGP is known.
    oracle: Option<Vec<(f64, f64, Option<f64>)>>,
}

struct Split {
    train: Data,
    /// `None` when the dataset ships with a fixed test file; then the seed
    /// only shuffles train/valid.
    test: Option<Data>,
}

fn parse_table(path: &PathBuf, sep: Option<char>) -> (Vec<String>, Vec<Vec<f64>>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let split = |l: &str| -> Vec<String> {
        match sep {
            Some(c) => l
                .split(c)
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect(),
            None => l
                .split_whitespace()
                .map(|s| s.trim_matches('"').to_string())
                .collect(),
        }
    };
    let first = lines.next().expect("empty file");
    let first_cells = split(first);
    let has_header = first_cells.iter().any(|c| c.parse::<f64>().is_err());
    let header = if has_header {
        first_cells
    } else {
        (0..first_cells.len()).map(|i| format!("c{i}")).collect()
    };
    let mut rows = Vec::new();
    let mut push = |cells: Vec<String>| {
        let parsed: Option<Vec<f64>> = cells.iter().map(|c| c.parse::<f64>().ok()).collect();
        if let Some(r) = parsed
            && r.len() == header.len()
        {
            rows.push(r);
        }
    };
    if !has_header {
        push(split(first));
    }
    for l in lines {
        push(split(l));
    }
    (header, rows)
}

fn table_to_data(
    header: &[String],
    rows: &[Vec<f64>],
    target: &str,
    features: &[&str],
    oracle: Option<(&str, &str, Option<&str>)>,
) -> Data {
    let idx = |name: &str| {
        header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("column {name} not in {header:?}"))
    };
    let ti = idx(target);
    let fi: Vec<usize> = features.iter().map(|f| idx(f)).collect();
    let n = rows.len();
    let mut x = Array2::zeros((n, fi.len()));
    let mut y = Array1::zeros(n);
    for (r, row) in rows.iter().enumerate() {
        y[r] = row[ti];
        for (c, &f) in fi.iter().enumerate() {
            x[[r, c]] = row[f];
        }
    }
    let oracle = oracle.map(|(l, s, d)| {
        let (li, si, di) = (idx(l), idx(s), d.map(idx));
        rows.iter()
            .map(|row| (row[li], row[si], di.map(|i| row[i])))
            .collect()
    });
    Data { x, y, oracle }
}

/// BENCH_Y_SHIFT=c adds c to every label (and to the oracle location of
/// Gaussian / Student-t oracles) — a translation-equivariance probe.
fn load(name: &str, cfg: &Config, seed: u64) -> Split {
    let mut sp = load_raw(name, cfg, seed);
    let c: f64 = env_or("BENCH_Y_SHIFT", 0.0);
    if c != 0.0 {
        let shift = |d: &mut Data| {
            d.y.mapv_inplace(|v| v + c);
            if let Some(o) = d.oracle.as_mut() {
                for r in o.iter_mut() {
                    if r.2.is_none() || r.2.is_some_and(|nu| nu > 0.0) {
                        r.0 += c;
                    }
                }
            }
        };
        shift(&mut sp.train);
        if let Some(t) = sp.test.as_mut() {
            shift(t);
        }
    }
    let c: f64 = env_or("BENCH_Y_SCALE", 1.0);
    if c != 1.0 {
        let scale = |d: &mut Data| {
            d.y.mapv_inplace(|v| v * c);
            d.oracle = None;
        };
        scale(&mut sp.train);
        if let Some(t) = sp.test.as_mut() {
            scale(t);
        }
    }
    sp
}

fn load_raw(name: &str, cfg: &Config, seed: u64) -> Split {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("research/gradientlss/data");
    // Held-out confirmation sets (`c_<name>`, built by prep_confirm_data.py): a numeric CSV whose
    // target column is `y` and whose other columns are all features. Never used before cycle 14.
    if let Some(rest) = name.strip_prefix("c_") {
        let path = dir.with_file_name("data_confirm").join(format!("{rest}.csv"));
        let (h, r) = parse_table(&path, Some(','));
        let feats: Vec<&str> = h.iter().filter(|c| c.as_str() != "y").map(|c| c.as_str()).collect();
        return Split { train: table_to_data(&h, &r, "y", &feats, None), test: None };
    }
    match name {
        "studentt_sim" => {
            let feats = ["x", "x2"];
            let (h, r) = parse_table(&dir.join("studentT_train_sim.csv"), Some(','));
            let train = table_to_data(&h, &r, "y", &feats, Some(("loc", "scale", Some("df"))));
            let (h, r) = parse_table(&dir.join("studentT_test_sim.csv"), Some(','));
            let test = table_to_data(&h, &r, "y", &feats, Some(("loc", "scale", Some("df"))));
            Split {
                train,
                test: Some(test),
            }
        }
        "gaussian_sim" => {
            let feats: Vec<String> = std::iter::once("x_true".to_string())
                .chain((1..=10).map(|i| format!("x_noise{i}")))
                .collect();
            let feats: Vec<&str> = feats.iter().map(|s| s.as_str()).collect();
            let (h, r) = parse_table(&dir.join("gaussian_train_sim.csv"), Some(','));
            let train = table_to_data(&h, &r, "y", &feats, Some(("loc", "scale", None)));
            let (h, r) = parse_table(&dir.join("gaussian_test_sim.csv"), Some(','));
            let test = table_to_data(&h, &r, "y", &feats, Some(("loc", "scale", None)));
            Split {
                train,
                test: Some(test),
            }
        }
        "synth_nb_lo" => Split {
            train: synth_nb(8000, seed, 2.0, 1.2, 1.0),
            test: None,
        },
        // near-Poisson corner: µ ≈ 1–8, θ ≈ 12–245 (θ ≫ µ)
        "synth_nb_ps" => Split {
            train: synth_nb(8000, seed, 1.0, 0.8, 4.0),
            test: None,
        },
        "synth_nb_hi" => Split {
            train: synth_nb(8000, seed, 6.5, 1.5, 2.5),
            test: None,
        },
        "synth_gamma" => Split {
            train: synth_gamma(8000, seed),
            test: None,
        },
        "synth_beta" => Split {
            train: synth_beta(8000, seed, 2.5),
            test: None,
        },
        "synth_beta_hi" => Split {
            train: synth_beta(8000, seed, 5.5),
            test: None,
        },
        "synth_weibull" => Split {
            train: synth_weibull(8000, seed),
            test: None,
        },
        "synth_gumbel" => Split {
            train: synth_gumbel(8000, seed),
            test: None,
        },
        // Share of casual riders, casual / cnt, on hours with 0 < casual < cnt.
        "bike_share" => {
            let (h, r) = parse_table(&dir.join("bike_hour.csv"), Some(','));
            let ci = h.iter().position(|c| c == "casual").unwrap();
            let ni = h.iter().position(|c| c == "cnt").unwrap();
            let mut h2 = h.clone();
            h2.push("share".to_string());
            let rows: Vec<Vec<f64>> = r
                .into_iter()
                .filter(|row| row[ci] > 0.0 && row[ci] < row[ni])
                .map(|mut row| {
                    let sh = row[ci] / row[ni];
                    row.push(sh);
                    row
                })
                .collect();
            let feats = [
                "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp",
                "atemp", "hum", "windspeed",
            ];
            Split {
                train: table_to_data(&h2, &rows, "share", &feats, None),
                test: None,
            }
        }
        "synth_t3" => Split {
            train: synth_t3(8000, seed),
            test: None,
        },
        "synth_pois" => Split {
            train: synth_pois(8000, seed),
            test: None,
        },
        "synth_g_sin" => Split {
            train: synth_ls(8000, seed, 0, None),
            test: None,
        },
        "synth_g_exp" => Split {
            train: synth_ls(8000, seed, 1, None),
            test: None,
        },
        "synth_t4_flat" => Split {
            train: synth_ls(8000, seed, 2, Some(4.0)),
            test: None,
        },
        "kin8nm" => {
            let feats: Vec<String> = (1..=8).map(|i| format!("theta{i}")).collect();
            let feats: Vec<&str> = feats.iter().map(|s| s.as_str()).collect();
            let (h, r) = parse_table(&dir.join("kin8nm.csv"), Some(','));
            Split {
                train: table_to_data(&h, &r, "y", &feats, None),
                test: None,
            }
        }
        // UCI Bike Sharing, hourly (17 379 rows; dteday dropped). Counts:
        // `cnt` (mean 189, var/mean ≈ 175) and `casual` (mean 36, var/mean ≈ 66).
        // casual + registered = cnt, so neither is a feature.
        "bike_cnt" | "bike_casual" => {
            let (h, r) = parse_table(&dir.join("bike_hour.csv"), Some(','));
            let feats = [
                "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp",
                "atemp", "hum", "windspeed",
            ];
            let target = if name == "bike_cnt" { "cnt" } else { "casual" };
            Split {
                train: table_to_data(&h, &r, target, &feats, None),
                test: None,
            }
        }
        "power_plant" => {
            let (h, r) = parse_table(&dir.join("power-plant.csv"), Some(','));
            Split {
                train: table_to_data(&h, &r, "PE", &["AT", "V", "AP", "RH"], None),
                test: None,
            }
        }
        "naval" => {
            // 16 sensor columns, then compressor decay (c16) and turbine decay (c17).
            let feats: Vec<String> = (0..16).map(|i| format!("c{i}")).collect();
            let feats: Vec<&str> = feats.iter().map(|s| s.as_str()).collect();
            let (h, r) = parse_table(&dir.join("naval-propulsion.txt"), None);
            Split {
                train: table_to_data(&h, &r, "c17", &feats, None),
                test: None,
            }
        }
        "protein" => {
            let feats: Vec<String> = (1..=9).map(|i| format!("F{i}")).collect();
            let feats: Vec<&str> = feats.iter().map(|s| s.as_str()).collect();
            let (h, mut r) = parse_table(&dir.join("protein.csv"), Some(','));
            // fixed subsample (seed-independent) so every seed sees the same pool
            let mut rng = ChaCha8Rng::seed_from_u64(2026);
            r.shuffle(&mut rng);
            r.truncate(cfg.protein_n);
            Split {
                train: table_to_data(&h, &r, "RMSD", &feats, None),
                test: None,
            }
        }
        other => panic!("unknown dataset {other}"),
    }
}

/// NB2 counts via Gamma–Poisson: ln µ = a + b·sin(πx₁) + 0.5x₂,
/// ln θ = c + 1.5x₃; five U(−1,1) features (x₄, x₅ noise).
/// `_lo` (a=2, b=1.2, c=1): µ ≈ 1.5…60, θ ≈ 0.6…12.
/// `_hi` (a=6.5, b=1.5, c=2.5): µ ≈ 100…4000, θ ≈ 2.7…55 — beyond the
/// MeanDispersion µ ≤ e⁶ ≈ 403 clamp for most rows.
fn synth_nb(n: usize, seed: u64, a: f64, b: f64, c: f64) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(0x7B ^ seed ^ (a.to_bits() >> 40));
    let u = Uniform::new(-1.0, 1.0).unwrap();
    let mut x = Array2::zeros((n, 5));
    let mut y = Array1::zeros(n);
    let mut oracle = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..5 {
            x[[i, j]] = u.sample(&mut rng);
        }
        let mu = (a + b * (std::f64::consts::PI * x[[i, 0]]).sin() + 0.5 * x[[i, 1]]).exp();
        let th = (c + 1.5 * x[[i, 2]]).exp();
        let lam = RandGammaD::new(th, mu / th).unwrap().sample(&mut rng);
        y[i] = if lam > 0.0 { RandPoisson::new(lam).unwrap().sample(&mut rng) } else { 0.0 };
        oracle.push((mu, th, Some(-2.0)));
    }
    Data { x, y, oracle: Some(oracle) }
}

/// Gamma DGP with a feature-dependent MEAN and SHAPE spanning ~2…2000:
///   ln µ = 3 + sin(πx₁) + 0.5x₂,  ln α = 4.2 + 3.4·x₃ (α ∈ [2.2, 2000]),
///   y ~ Gamma(α, rate α/µ). Five U(−1,1) features; x₄, x₅ are noise.
fn synth_gamma(n: usize, seed: u64) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(0x6A ^ seed);
    let u = Uniform::new(-1.0, 1.0).unwrap();
    let mut x = Array2::zeros((n, 5));
    let mut y = Array1::zeros(n);
    let mut oracle = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..5 {
            x[[i, j]] = u.sample(&mut rng);
        }
        let mu = (3.0 + (std::f64::consts::PI * x[[i, 0]]).sin() + 0.5 * x[[i, 1]]).exp();
        let alpha = (4.2 + 3.4 * x[[i, 2]]).exp();
        y[i] = RandGammaD::new(alpha, mu / alpha).unwrap().sample(&mut rng);
        oracle.push((alpha, alpha / mu, Some(-1.0)));
    }
    Data { x, y, oracle: Some(oracle) }
}

/// Beta DGP: logit µ = 0.8 sin(πx₁) + 0.5x₂, log φ = c + 1.5x₃ (c = 2.5: φ ≈ 2.7–55;
/// c = 5.5: φ ≈ 55–1100), a = µφ, b = (1−µ)φ. Five U(−1,1) features; x₄, x₅ noise.
fn synth_beta(n: usize, seed: u64, c: f64) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(0xBE7A ^ seed ^ (c.to_bits() >> 40));
    let u = Uniform::new(-1.0, 1.0).unwrap();
    let mut x = Array2::zeros((n, 5));
    let mut y = Array1::zeros(n);
    let mut oracle = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..5 {
            x[[i, j]] = u.sample(&mut rng);
        }
        let eta = 0.8 * (std::f64::consts::PI * x[[i, 0]]).sin() + 0.5 * x[[i, 1]];
        let mu = 1.0 / (1.0 + (-eta).exp());
        let phi = (c + 1.5 * x[[i, 2]]).exp();
        let (a, b) = (mu * phi, (1.0 - mu) * phi);
        let v: f64 = rand_distr::Beta::new(a, b).unwrap().sample(&mut rng);
        y[i] = v.clamp(1e-12, 1.0 - 1e-12);
        oracle.push((a, b, Some(-3.0)));
    }
    Data { x, y, oracle: Some(oracle) }
}

/// Weibull DGP: log λ = 1 + 0.6 sin(πx₁) + 0.3x₂, log k = 0.7 + 0.9x₃ (k ≈ 0.8–5).
fn synth_weibull(n: usize, seed: u64) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(0x3E1B ^ seed);
    let u = Uniform::new(-1.0, 1.0).unwrap();
    let mut x = Array2::zeros((n, 5));
    let mut y = Array1::zeros(n);
    let mut oracle = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..5 {
            x[[i, j]] = u.sample(&mut rng);
        }
        let lam = (1.0 + 0.6 * (std::f64::consts::PI * x[[i, 0]]).sin() + 0.3 * x[[i, 1]]).exp();
        let k = (0.7 + 0.9 * x[[i, 2]]).exp();
        y[i] = rand_distr::Weibull::new(lam, k).unwrap().sample(&mut rng);
        oracle.push((lam, k, Some(-4.0)));
    }
    Data { x, y, oracle: Some(oracle) }
}

/// Gumbel DGP (proposal 028), location far from zero: μ = 120 + 25 sin(πx₁) + 10x₂,
/// log s = 1 + 0.8x₃ (s ≈ 1.2–6), y = μ − s·ln(−ln U).
fn synth_gumbel(n: usize, seed: u64) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(0x6B5E ^ seed);
    let u = Uniform::new(-1.0, 1.0).unwrap();
    let u01 = Uniform::new(1e-12, 1.0).unwrap();
    let mut x = Array2::zeros((n, 5));
    let mut y = Array1::zeros(n);
    let mut oracle = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..5 {
            x[[i, j]] = u.sample(&mut rng);
        }
        let mu = 120.0 + 25.0 * (std::f64::consts::PI * x[[i, 0]]).sin() + 10.0 * x[[i, 1]];
        let s = (1.0 + 0.8 * x[[i, 2]]).exp();
        let p: f64 = u01.sample(&mut rng);
        y[i] = mu - s * (-(p.ln())).ln();
        oracle.push((mu, s, Some(-5.0)));
    }
    Data { x, y, oracle: Some(oracle) }
}

/// CRPS by midpoint quadrature of (F(t) − 1{t ≥ y})² over [lo, hi] ∋ y,
/// [lo, hi] covering all but ~1e-9 of the mass at each end.
fn crps_quad(y: f64, lo: f64, hi: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    let (lo, hi) = (lo.min(y), hi.max(y));
    let m = 800;
    let h = (hi - lo) / m as f64;
    (0..m)
        .map(|i| {
            let t = lo + (i as f64 + 0.5) * h;
            let ind = if t >= y { 1.0 } else { 0.0 };
            (cdf(t) - ind).powi(2)
        })
        .sum::<f64>()
        * h
}

fn crps_beta(y: f64, a: f64, b: f64) -> f64 {
    match StatBeta::new(a, b) {
        Ok(d) => crps_quad(y, d.inverse_cdf(1e-9), d.inverse_cdf(1.0 - 1e-9), |t| d.cdf(t)),
        Err(_) => f64::NAN,
    }
}

fn crps_weibull(y: f64, lam: f64, k: f64) -> f64 {
    // closed-form quantiles: λ(−ln(1−p))^{1/k}
    let q = |p: f64| lam * (-(1.0 - p).ln()).powf(1.0 / k);
    crps_quad(y, q(1e-9), q(1.0 - 1e-9), |t| if t <= 0.0 { 0.0 } else { 1.0 - (-(t / lam).powf(k)).exp() })
}

/// Heavy-tailed heteroscedastic DGP with every parameter feature-dependent:
///   µ = sin(πx₁) + x₂²,  log σ = 0.5·x₃ − 0.3,  ν = 2.5 + 3·sigmoid(3x₄),
///   y = µ + σ·t_ν. Five U(−1,1) features; x₅ is pure noise.
fn synth_t3(n: usize, seed: u64) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(0xD6 ^ seed);
    let u = Uniform::new(-1.0, 1.0).unwrap();
    let mut x = Array2::zeros((n, 5));
    let mut y = Array1::zeros(n);
    let mut oracle = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..5 {
            x[[i, j]] = u.sample(&mut rng);
        }
        let mu = (std::f64::consts::PI * x[[i, 0]]).sin() + x[[i, 1]] * x[[i, 1]];
        let sigma = (0.5 * x[[i, 2]] - 0.3).exp();
        let nu = 2.5 + 3.0 / (1.0 + (-3.0 * x[[i, 3]]).exp());
        y[i] = mu + sigma * RandT::new(nu).unwrap().sample(&mut rng);
        oracle.push((mu, sigma, Some(nu)));
    }
    Data {
        x,
        y,
        oracle: Some(oracle),
    }
}

/// True Poisson counts (proposal 014): log µ = 1.5 + 2.2·sin(πx₀) + 0.8·x₁,
/// so µ spans ≈ 0.05–180 with many zeros at the low end. The oracle is
/// stored as an NB with θ = 1e8 (sentinel −2), i.e. Poisson to < 1e-6 nats.
fn synth_pois(n: usize, seed: u64) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(0x501 ^ seed);
    let u = Uniform::new(-1.0, 1.0).unwrap();
    let mut x = Array2::zeros((n, 5));
    let mut y = Array1::zeros(n);
    let mut oracle = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..5 {
            x[[i, j]] = u.sample(&mut rng);
        }
        let mu = (1.5 + 2.2 * (std::f64::consts::PI * x[[i, 0]]).sin() + 0.8 * x[[i, 1]]).exp();
        y[i] = RandPoisson::new(mu).unwrap().sample(&mut rng);
        oracle.push((mu, 1e8, Some(-2.0)));
    }
    Data {
        x,
        y,
        oracle: Some(oracle),
    }
}

/// Held-out location–scale simulations (proposal 013). kind 0: sin mean
/// (light marginal); 1: exp(2x₀) mean (kurtotic marginal); 2: small signal.
/// Noise: Gaussian (`nu = None`) or t_ν. Heteroscedastic σ(x).
fn synth_ls(n: usize, seed: u64, kind: u8, nu: Option<f64>) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(0x13A0 ^ ((kind as u64) << 8) ^ seed);
    let u = Uniform::new(-1.0, 1.0).unwrap();
    let mut x = Array2::zeros((n, 5));
    let mut y = Array1::zeros(n);
    let mut oracle = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..5 {
            x[[i, j]] = u.sample(&mut rng);
        }
        let mu = match kind {
            0 => 2.0 * (std::f64::consts::PI * x[[i, 0]]).sin() + x[[i, 1]],
            1 => (2.0 * x[[i, 0]]).exp() + 0.5 * x[[i, 1]],
            _ => 0.3 * x[[i, 0]],
        };
        let sigma = (0.4 * x[[i, 2]] - 0.5).exp();
        let e = match nu {
            None => rand_distr::StandardNormal.sample(&mut rng),
            Some(v) => RandT::new(v).unwrap().sample(&mut rng),
        };
        let e: f64 = e;
        y[i] = mu + sigma * e;
        oracle.push((mu, sigma, nu));
    }
    Data {
        x,
        y,
        oracle: Some(oracle),
    }
}

fn take(d: &Data, idx: &[usize]) -> Data {
    Data {
        x: d.x.select(Axis(0), idx),
        y: d.y.select(Axis(0), idx),
        oracle: d
            .oracle
            .as_ref()
            .map(|o| idx.iter().map(|&i| o[i]).collect()),
    }
}

/// Seeded split. Fixed-test datasets: 85/15 train/valid. Otherwise
/// 65/15/20 train/valid/test.
fn split(split: Split, seed: u64) -> (Data, Data, Data) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = split.train.y.len();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);
    match split.test {
        Some(test) => {
            let n_valid = n * 15 / 100;
            let (v, t) = perm.split_at(n_valid);
            (take(&split.train, t), take(&split.train, v), test)
        }
        None => {
            let n_test = n / 5;
            let n_valid = n * 15 / 100;
            let (te, rest) = perm.split_at(n_test);
            let (v, t) = rest.split_at(n_valid);
            (
                take(&split.train, t),
                take(&split.train, v),
                take(&split.train, te),
            )
        }
    }
}

// ---------------------------------------------------------------- scores

fn crps_gaussian(y: f64, mu: f64, sigma: f64) -> f64 {
    let z = (y - mu) / sigma;
    let n = Normal::new(0.0, 1.0).unwrap();
    sigma
        * (z * (2.0 * n.cdf(z) - 1.0) + 2.0 * n.pdf(z)
            - std::f64::consts::FRAC_1_SQRT_2 / std::f64::consts::PI.sqrt()
                * std::f64::consts::SQRT_2)
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Closed-form CRPS of the location–scale t (Gneiting & Raftery 2007, Jordan
/// et al. 2019 `scoringRules::crps_t`); requires ν > 1.
fn crps_student_t(y: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if nu <= 1.0 {
        return f64::NAN;
    }
    // Beyond ν ≈ 1e6 the ln-beta differences lose all precision (lnΓ(ν) ~ ν ln ν);
    // the t is Gaussian to ~1/ν there.
    if nu > 1e6 {
        return crps_gaussian(y, mu, sigma);
    }
    let z = (y - mu) / sigma;
    let t = StudentsT::new(0.0, 1.0, nu).unwrap();
    let term = z * (2.0 * t.cdf(z) - 1.0) + 2.0 * t.pdf(z) * (nu + z * z) / (nu - 1.0)
        - 2.0 * nu.sqrt() * (ln_beta(0.5, nu - 0.5) - 2.0 * ln_beta(0.5, 0.5 * nu)).exp()
            / (nu - 1.0);
    sigma * term
}

/// Closed-form CRPS of Gamma(shape a, rate b) (Gneiting & Raftery 2007;
/// `scoringRules::crps_gamma`):
/// y(2F(y;a,b) − 1) − (a/b)(2F(y;a+1,b) − 1) − 1/(b·B(½, a)).
fn crps_gamma(y: f64, a: f64, b: f64) -> f64 {
    if !(a > 0.0 && b > 0.0) {
        return f64::NAN;
    }
    if y <= 0.0 {
        return a / b - (-(ln_gamma(0.5) + ln_gamma(a) - ln_gamma(a + 0.5))).exp() / b;
    }
    let f_a = StatGamma::new(a, b).map(|d| d.cdf(y)).unwrap_or(f64::NAN);
    let f_a1 = StatGamma::new(a + 1.0, b).map(|d| d.cdf(y)).unwrap_or(f64::NAN);
    let inv_beta = (-(ln_gamma(0.5) + ln_gamma(a) - ln_gamma(a + 0.5))).exp();
    y * (2.0 * f_a - 1.0) - (a / b) * (2.0 * f_a1 - 1.0) - inv_beta / b
}

/// NB in (µ, θ) from the crate's reported (total_count r, probs p) with the
/// torch convention mean = r·p/(1−p).
fn nb_mu_theta(p: &[f64]) -> (f64, f64) {
    (p[0] * p[1] / (1.0 - p[1]), p[0])
}

/// ln P(Y = k) for NB2(µ, θ).
fn nb_ln_pmf(k: f64, mu: f64, th: f64) -> f64 {
    ln_gamma(k + th) - ln_gamma(th) - ln_gamma(k + 1.0) + th * (th / (mu + th)).ln()
        + k * (mu / (mu + th)).ln()
}

/// Exact CRPS and randomized PIT of NB2(µ, θ) at y by summation over k
/// (CRPS = Σ_k (F(k) − 1{y ≤ k})², truncated where the tail mass < 1e-12).
fn nb_crps_pit(y: f64, mu: f64, th: f64, u: f64) -> (f64, f64) {
    if !(mu > 0.0 && th > 0.0) || !mu.is_finite() || !th.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let sd = (mu + mu * mu / th).sqrt();
    let kmax = (mu.max(y) + 40.0 * sd + 50.0).min(5e6) as usize;
    let (mut cdf, mut crps, mut f_below, mut p_at) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for k in 0..=kmax {
        let pk = nb_ln_pmf(k as f64, mu, th).exp();
        if (k as f64) == y {
            f_below = cdf;
            p_at = pk;
        }
        cdf += pk;
        let ind = if y <= k as f64 { 1.0 } else { 0.0 };
        crps += (cdf.min(1.0) - ind).powi(2);
        if k as f64 > y && 1.0 - cdf < 1e-12 {
            break;
        }
    }
    (crps, f_below + u * p_at)
}

/// Uniform for the randomized discrete PIT. Legacy (default): a golden-ratio
/// fraction of y, which is NOT independent of y (y = 0 always gets u ≈ 0, so
/// zero counts land in the lower tail). BENCH_PIT_U=hash: a splitmix64 hash
/// of (y, µ) bits — pseudo-uniform and unrelated to the PIT position.
fn pit_u(y: f64, mu: f64) -> f64 {
    if std::env::var("BENCH_PIT_U").as_deref() == Ok("hash") {
        let mut z = y.to_bits() ^ mu.to_bits().rotate_left(17) ^ 0x9E37_79B9_7F4A_7C15;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    } else {
        ((y * 0.618_033_988_749_895 + mu * 1e-3).fract() + 1.0).fract()
    }
}

/// Exact CRPS and randomized PIT of Poisson(λ) at y by summation over k.
fn pois_crps_pit(y: f64, lam: f64, u: f64) -> (f64, f64) {
    if !(lam > 0.0) || !lam.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let kmax = (lam.max(y) + 40.0 * lam.sqrt() + 50.0).min(5e6) as usize;
    let (mut cdf, mut crps, mut f_below, mut p_at) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for k in 0..=kmax {
        let kf = k as f64;
        let pk = (-lam + kf * lam.ln() - ln_gamma(kf + 1.0)).exp();
        if kf == y {
            f_below = cdf;
            p_at = pk;
        }
        cdf += pk;
        let ind = if y <= kf { 1.0 } else { 0.0 };
        crps += (cdf.min(1.0) - ind).powi(2);
        if kf > y && 1.0 - cdf < 1e-12 {
            break;
        }
    }
    (crps, f_below + u * p_at)
}

fn pit(y: f64, p: &[f64], fam: Family) -> f64 {
    match fam {
        Family::Gaussian => Normal::new(p[0], p[1])
            .map(|d| d.cdf(y))
            .unwrap_or(f64::NAN),
        Family::StudentT if p[0] > 1e6 => Normal::new(p[1], p[2])
            .map(|d| d.cdf(y))
            .unwrap_or(f64::NAN),
        Family::StudentT => StudentsT::new(p[1], p[2], p[0])
            .map(|d| d.cdf(y))
            .unwrap_or(f64::NAN),
        Family::Gamma | Family::GammaMs => StatGamma::new(p[0], p[1])
            .map(|d| d.cdf(y))
            .unwrap_or(f64::NAN),
        // discrete: randomized PIT with a deterministic per-value uniform
        Family::Nb | Family::NbMd => {
            let (mu, th) = nb_mu_theta(p);
            nb_crps_pit(y, mu, th, pit_u(y, mu)).1
        }
        Family::Poisson => pois_crps_pit(y, p[0], pit_u(y, p[0])).1,
        Family::Beta | Family::BetaMp => StatBeta::new(p[0], p[1]).map(|d| d.cdf(y)).unwrap_or(f64::NAN),
        // closed form: statrs' Weibull cdf returns NaN for large shapes (k ≳ 100)
        Family::Weibull => {
            if y <= 0.0 { 0.0 } else { -(-(y / p[0]).powf(p[1])).exp_m1() }
        }
        Family::Cauchy | Family::Logistic | Family::Gumbel | Family::Laplace | Family::LogNormal => ls_cdf(y, p[0], p[1], fam),
    }
}

/// CDF of the two-parameter (loc, scale) families added in cycle 14.
fn ls_cdf(y: f64, loc: f64, scale: f64, fam: Family) -> f64 {
    let z = (y - loc) / scale;
    match fam {
        Family::Cauchy => 0.5 + z.atan() / std::f64::consts::PI,
        Family::Logistic => 1.0 / (1.0 + (-z).exp()),
        // maximum-type Gumbel: F = exp(−e^{−z})
        Family::Gumbel => (-(-z).exp()).exp(),
        Family::Laplace => if z < 0.0 { 0.5 * z.exp() } else { 1.0 - 0.5 * (-z).exp() },
        Family::LogNormal => {
            if y <= 0.0 { 0.0 } else { Normal::new(loc, scale).map(|d| d.cdf(y.ln())).unwrap_or(f64::NAN) }
        }
        _ => unreachable!(),
    }
}

fn crps_row(y: f64, p: &[f64], fam: Family) -> f64 {
    match fam {
        Family::Gaussian => crps_gaussian(y, p[0], p[1]),
        Family::StudentT => crps_student_t(y, p[1], p[2], p[0]),
        Family::Gamma | Family::GammaMs => crps_gamma(y, p[0], p[1]),
        Family::Nb | Family::NbMd => {
            let (mu, th) = nb_mu_theta(p);
            nb_crps_pit(y, mu, th, 0.5).0
        }
        Family::Poisson => pois_crps_pit(y, p[0], 0.5).0,
        Family::Beta | Family::BetaMp => crps_beta(y, p[0], p[1]),
        Family::Weibull => crps_weibull(y, p[0], p[1]),
        // no finite first moment: the CRPS is infinite
        Family::Cauchy => f64::NAN,
        // closed forms (Jordan, Krüger & Lerch 2019, scoringRules)
        Family::Logistic => {
            let z = (y - p[0]) / p[1];
            p[1] * (z - 2.0 * (-(1.0 + (-z).exp()).ln()) - 1.0)
        }
        Family::Laplace => {
            let z = ((y - p[0]) / p[1]).abs();
            p[1] * (z + (-z).exp() - 0.75)
        }
        // quadrature between the 1e-9 quantiles
        Family::Gumbel => {
            let q = |u: f64| p[0] - p[1] * (-(u.ln())).ln();
            crps_quad(y, q(1e-9), q(1.0 - 1e-9), |t| ls_cdf(t, p[0], p[1], Family::Gumbel))
        }
        Family::LogNormal => {
            let (lo, hi) = ((p[0] - 6.0 * p[1]).exp(), (p[0] + 6.0 * p[1]).exp());
            crps_quad(y, lo, hi, |t| ls_cdf(t, p[0], p[1], Family::LogNormal))
        }
    }
}

fn mean_loc(p: &[f64], fam: Family) -> f64 {
    match fam {
        Family::Gaussian => p[0],
        Family::StudentT => p[1],
        Family::Gamma | Family::GammaMs => p[0] / p[1],
        Family::Nb | Family::NbMd => nb_mu_theta(p).0,
        Family::Poisson => p[0],
        Family::Beta | Family::BetaMp => p[0] / (p[0] + p[1]),
        Family::Weibull => p[0] * ln_gamma(1.0 + 1.0 / p[1]).exp(),
        // Cauchy has no mean: the median stands in for the point forecast
        Family::Cauchy | Family::Logistic | Family::Laplace => p[0],
        Family::Gumbel => p[0] + 0.577_215_664_901_532_9 * p[1],
        Family::LogNormal => (p[0] + 0.5 * p[1] * p[1]).exp(),
    }
}

fn scale_of(p: &[f64], fam: Family) -> f64 {
    match fam {
        Family::Gaussian => p[1],
        Family::StudentT => p[2],
        Family::Gamma | Family::GammaMs => p[0].sqrt() / p[1],
        Family::Nb | Family::NbMd => {
            let (mu, th) = nb_mu_theta(p);
            (mu + mu * mu / th).sqrt()
        }
        Family::Poisson => p[0].sqrt(),
        Family::Beta | Family::BetaMp => {
            let s = p[0] + p[1];
            (p[0] * p[1] / (s * s * (s + 1.0))).sqrt()
        }
        Family::Weibull => {
            let g1 = ln_gamma(1.0 + 1.0 / p[1]).exp();
            let g2 = ln_gamma(1.0 + 2.0 / p[1]).exp();
            p[0] * (g2 - g1 * g1).max(0.0).sqrt()
        }
        // Cauchy: the scale parameter itself (no variance)
        Family::Cauchy => p[1],
        Family::Logistic => p[1] * std::f64::consts::PI / 3f64.sqrt(),
        Family::Gumbel => p[1] * std::f64::consts::PI / 6f64.sqrt(),
        Family::Laplace => p[1] * std::f64::consts::SQRT_2,
        Family::LogNormal => ((p[1] * p[1]).exp_m1()).max(0.0).sqrt() * (p[0] + 0.5 * p[1] * p[1]).exp(),
    }
}

#[derive(Default, Clone)]
struct Scores {
    nll: f64,
    crps: f64,
    cov50: f64,
    cov90: f64,
    pit_ks: f64,
    rmse: f64,
    mean_scale: f64,
    nonfinite: usize,
    oracle_nll: f64,
    oracle_crps: f64,
    /// Best validation NLL (early-stopping criterion) — for selecting a
    /// setting per mode without touching the test split.
    valid_nll: f64,
    /// 5% quantile and median of the predicted df (StudentT only, else NaN).
    df_q05: f64,
    df_med: f64,
    /// Largest predicted scale on the test split (divergence detector).
    scale_max: f64,
}

fn score(dist: &dyn Distribution, params: &ArrayView2<f64>, test: &Data, fam: Family) -> Scores {
    let n = test.y.len();
    let mut s = Scores::default();
    let mut pits = Vec::with_capacity(n);
    let mut dfs = Vec::new();
    s.scale_max = f64::NEG_INFINITY;
    let (mut nll, mut crps, mut se, mut sc, mut k) = (0.0, 0.0, 0.0, 0.0, 0usize);
    let mut worst: Vec<(f64, f64, Vec<f64>)> = Vec::new();
    for i in 0..n {
        let p = params.row(i);
        let p = p.as_slice().unwrap();
        let y = test.y[i];
        if p.iter().any(|v| !v.is_finite()) {
            s.nonfinite += 1;
            continue;
        }
        let lp = dist.log_prob(p, &[y]);
        let c = crps_row(y, p, fam);
        let u = pit(y, p, fam);
        // Cauchy has no finite CRPS: its rows are judged on log score and PIT alone.
        let c_ok = c.is_finite() || fam == Family::Cauchy;
        if !lp.is_finite() || !c_ok || !u.is_finite() {
            if env_or("BENCH_DEBUG", 0u8) == 1 && s.nonfinite < 3 {
                eprintln!("DEBUG nonfinite row: y {y} params {p:?} lp {lp} crps {c} pit {u}");
            }
            s.nonfinite += 1;
            continue;
        }
        if env_or("BENCH_DEBUG_WORST", 0u8) == 1 {
            worst.push((-lp, y, p.to_vec()));
        }
        nll -= lp;
        crps += c; // NaN for Cauchy, by design
        let r = y - mean_loc(p, fam);
        se += r * r;
        sc += scale_of(p, fam);
        s.scale_max = s.scale_max.max(scale_of(p, fam));
        if fam == Family::StudentT {
            dfs.push(p[0]);
        }
        pits.push(u);
        k += 1;
    }
    let kf = k as f64;
    if !worst.is_empty() {
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        let tot: f64 = worst.iter().map(|w| w.0).sum();
        for (l, y, p) in worst.iter().take(4) {
            eprintln!("WORST row nll {l:.3} (share {:.3} of total) y {y} params {p:?}", l / tot);
        }
        let med = worst[worst.len() / 2].0;
        // proposal 028: mean without the worst 0.5 % of rows (at least one row)
        let cut = ((kf * 0.005).ceil() as usize).max(1);
        let trim = worst[cut..].iter().map(|w| w.0).sum::<f64>() / (kf - cut as f64);
        eprintln!(
            "WORST median row nll {med:.3}, mean {:.4}, mean without top row {:.4}, trim0.5 {trim:.4} crps {:.4}",
            tot / kf,
            (tot - worst[0].0) / (kf - 1.0),
            crps / kf
        );
    }
    s.nll = nll / kf;
    s.crps = crps / kf;
    s.rmse = (se / kf).sqrt();
    s.mean_scale = sc / kf;
    s.cov50 = pits.iter().filter(|&&u| (0.25..=0.75).contains(&u)).count() as f64 / kf;
    s.cov90 = pits.iter().filter(|&&u| (0.05..=0.95).contains(&u)).count() as f64 / kf;
    if dfs.is_empty() {
        s.df_q05 = f64::NAN;
        s.df_med = f64::NAN;
    } else {
        dfs.sort_by(|a, b| a.total_cmp(b));
        s.df_q05 = dfs[dfs.len() / 20];
        s.df_med = dfs[dfs.len() / 2];
    }
    pits.sort_by(|a, b| a.total_cmp(b));
    s.pit_ks = pits
        .iter()
        .enumerate()
        .map(|(i, &u)| {
            ((i + 1) as f64 / kf - u)
                .abs()
                .max((u - i as f64 / kf).abs())
        })
        .fold(0.0, f64::max);
    if let Some(o) = &test.oracle {
        let (mut on, mut oc, mut m) = (0.0, 0.0, 0usize);
        for (i, &(mu, sigma, df)) in o.iter().enumerate() {
            let y = test.y[i];
            let (lp, c) = match df {
                // sentinel: (shape, rate, Some(-1)) marks a Gamma oracle
                Some(k) if k == -2.0 => {
                    let (c, _) = nb_crps_pit(y, mu, sigma, 0.5);
                    (nb_ln_pmf(y, mu, sigma), c)
                }
                Some(k) if k == -3.0 => (StatBeta::new(mu, sigma).unwrap().ln_pdf(y), crps_beta(y, mu, sigma)),
                Some(k) if k == -4.0 => (
                    StatWeibull::new(sigma, mu).unwrap().ln_pdf(y),
                    crps_weibull(y, mu, sigma),
                ),
                Some(k) if k == -5.0 => {
                    let z = (y - mu) / sigma;
                    (-z - (-z).exp() - sigma.ln(), crps_row(y, &[mu, sigma], Family::Gumbel))
                }
                Some(k) if k == -1.0 => (
                    StatGamma::new(mu, sigma).unwrap().ln_pdf(y),
                    crps_gamma(y, mu, sigma),
                ),
                Some(nu) => (
                    StudentsT::new(mu, sigma, nu).unwrap().ln_pdf(y),
                    crps_student_t(y, mu, sigma, nu),
                ),
                None => (
                    Normal::new(mu, sigma).unwrap().ln_pdf(y),
                    crps_gaussian(y, mu, sigma),
                ),
            };
            if lp.is_finite() && c.is_finite() {
                on -= lp;
                oc += c;
                m += 1;
            }
        }
        if env_or("BENCH_ORACLE_COV", 0u8) == 1 {
            let pits: Vec<f64> = o
                .iter()
                .enumerate()
                .filter(|(_, r)| r.2 == Some(-2.0))
                .map(|(i, &(mu, th, _))| nb_crps_pit(test.y[i], mu, th, pit_u(test.y[i], mu)).1)
                .collect();
            if !pits.is_empty() {
                let k = pits.len() as f64;
                eprintln!(
                    "ORACLE_COV cov50 {:.4} cov90 {:.4}",
                    pits.iter().filter(|&&u| (0.25..=0.75).contains(&u)).count() as f64 / k,
                    pits.iter().filter(|&&u| (0.05..=0.95).contains(&u)).count() as f64 / k
                );
            }
        }
        s.oracle_nll = on / m as f64;
        s.oracle_crps = oc / m as f64;
    } else {
        s.oracle_nll = f64::NAN;
        s.oracle_crps = f64::NAN;
    }
    s
}

// ---------------------------------------------------------------- run

/// BENCH_PARAM_SCALES="c0,c1[,c2]" — per-parameter gradient multipliers in the
/// distribution's parameter order; must have exactly `n` entries for the family.
fn param_scales(n: usize) -> Option<Vec<f64>> {
    let raw = std::env::var("BENCH_PARAM_SCALES").ok()?;
    let v: Vec<f64> = raw
        .split(',')
        .map(|s| s.trim().parse().expect("BENCH_PARAM_SCALES must be numbers"))
        .collect();
    if v.len() != n {
        return None;
    }
    Some(v)
}

/// BENCH_LOSS=crps trains Poisson / NB / NB (mean, dispersion) on the CRPS (proposal 023); default NLL.
fn bench_loss() -> LossFn {
    match std::env::var("BENCH_LOSS").as_deref() {
        Ok("crps") => LossFn::Crps,
        Ok("nll") | Err(_) => LossFn::Nll,
        Ok(other) => panic!("unknown BENCH_LOSS {other}"),
    }
}

fn make_dist(fam: Family, mode: HessianMode, cfg: &Config) -> Arc<dyn Distribution> {
    match fam {
        Family::Gaussian => {
            let mut d = Gaussian::new(cfg.stab, ResponseFn::Exp, LossFn::Nll, cfg.init)
                .with_hessian_mode(mode);
            if let Some(sc) = param_scales(2) {
                d = d.with_param_gradient_scales([sc[0], sc[1]]);
            }
            if let Some(b) = beta_nll() {
                d = d.with_beta_nll(b);
            }
            Arc::new(d)
        }
        Family::Gamma => Arc::new(
            Gamma::new(cfg.stab, ResponseFn::Exp, LossFn::Nll, cfg.init)
                .with_parameterization(GammaParameterization::ConcentrationRate)
                .with_hessian_mode(mode),
        ),
        // Built like a user would (`new`): MeanShape is the Exp+NLL default.
        // BENCH_INIT=1 reproduces the pre-cycle-8 rows (new_mean_shape set
        // initialize: true); BENCH_LOC_START=1 = location-only start.
        Family::GammaMs => Arc::new(
            Gamma::new(cfg.stab, ResponseFn::Exp, LossFn::Nll, cfg.init)
                .with_hessian_mode(mode)
                .with_location_start(env_or("BENCH_LOC_START", 0u8) == 1)
                .with_wide_start(env_or("BENCH_GM_RAW", 0u8) != 1),
        ),
        // NB heads have no HessianMode; the mode axis is ignored for them.
        Family::Nb => Arc::new(NegativeBinomial::new(
            cfg.stab,
            ResponseFn::Relu,
            ResponseFn::Sigmoid,
            bench_loss(),
            cfg.init,
        )
        .with_parameterization(NbParameterization::TotalCountProbs)),
        Family::Poisson => {
            let link = match std::env::var("BENCH_POIS_LINK").as_deref() {
                Ok("exp") => ResponseFn::Exp,
                Ok("softplus") => ResponseFn::Softplus,
                _ => ResponseFn::Relu,
            };
            Arc::new(
                gradientlss::distributions::Poisson::new(cfg.stab, link, bench_loss(), cfg.init)
                    .with_hessian_mode(mode)
                    .with_location_start(env_or("BENCH_LOC_START", 0u8) == 1),
            )
        }
        Family::NbMd => {
            // CRPS trains in (log µ, log θ) too since proposal 031 (BENCH_NB_CRPS_MD is no longer read).
            // BENCH_NB_CRPS_TCP=1: the opt-out, the (total_count, probs) head CRPS used before.
            let d = if bench_loss() == LossFn::Crps && env_or("BENCH_NB_CRPS_TCP", 0u8) == 1 {
                NegativeBinomial::new(cfg.stab, ResponseFn::Softplus, ResponseFn::Sigmoid, LossFn::Crps, true)
                    .with_parameterization(NbParameterization::TotalCountProbs)
            } else {
                NegativeBinomial::new_mean_dispersion(cfg.stab, bench_loss())
            };
            Arc::new(d.with_hessian_mode(mode).with_initialize(cfg.init).with_location_start(env_or("BENCH_LOC_START", 0u8) == 1))
        }
        Family::Beta => Arc::new(
            BetaDist::new(cfg.stab, ResponseFn::Exp, LossFn::Nll, cfg.init)
                .with_parameterization(BetaParameterization::Concentrations)
                .with_hessian_mode(mode),
        ),
        Family::BetaMp => Arc::new(
            BetaDist::new(cfg.stab, ResponseFn::Exp, LossFn::Nll, cfg.init)
                .with_parameterization(BetaParameterization::MeanPrecision)
                .with_hessian_mode(mode),
        ),
        Family::Weibull => {
            let mut d = WeibullDist::new(cfg.stab, ResponseFn::Exp, ResponseFn::Exp, LossFn::Nll, cfg.init)
                .with_hessian_mode(mode)
                .with_location_start(env_or("BENCH_LOC_START", 0u8) == 1);
            if let Ok(c) = std::env::var("BENCH_WB_CAP") {
                d = d.with_max_concentration(c.parse().expect("BENCH_WB_CAP"));
            }
            Arc::new(d)
        }
        // Cycle 15 prototype (proposal 021). Without BENCH_LS_PORT these are the pre-2026-11-13
        // defaults (observed curvature, raw start) whatever BENCH_MODES says; with =1, BENCH_MODES
        // sets the curvature and BENCH_LS_START=raw|loc|locscale the start rule; =2 is `new` as is.
        Family::Cauchy | Family::Logistic | Family::Gumbel | Family::Laplace | Family::LogNormal => {
            use gradientlss::distributions::{Cauchy, Gumbel, Laplace, LocationScaleStart, LogNormal, Logistic};
            let (a, b, c, d) = (cfg.stab, ResponseFn::Exp, LossFn::Nll, cfg.init);
            // BENCH_LS_PORT=2: whatever `new` gives (the crate defaults since 2026-11-13).
            if env_or("BENCH_LS_PORT", 0u8) == 2 {
                return match fam {
                    Family::Cauchy => Arc::new(Cauchy::new(a, b, c, d)),
                    Family::Logistic => Arc::new(Logistic::new(a, b, c, d)),
                    Family::Gumbel => Arc::new(Gumbel::new(a, b, c, d)),
                    Family::Laplace => Arc::new(Laplace::new(a, b, c, d)),
                    _ => Arc::new(LogNormal::new(a, b, c, d)),
                };
            }
            let port = env_or("BENCH_LS_PORT", 0u8) == 1;
            let m = if port { mode } else { HessianMode::Observed };
            let st = match (port, std::env::var("BENCH_LS_START").as_deref()) {
                (true, Ok("loc")) => LocationScaleStart::Location,
                (true, Ok("locscale")) => LocationScaleStart::LocationScale,
                _ => LocationScaleStart::Raw,
            };
            match fam {
                Family::Cauchy => Arc::new(Cauchy::new(a, b, c, d).with_hessian_mode(m).with_start(st)),
                Family::Logistic => Arc::new(Logistic::new(a, b, c, d).with_hessian_mode(m).with_start(st)),
                // BENCH_LS_SCALE_K=c (proposal 028): Gumbel start scale = c × the moment-matched one
                Family::Gumbel => Arc::new(
                    Gumbel::new(a, b, c, d).with_hessian_mode(m).with_start(st).with_start_scale_factor(env_or("BENCH_LS_SCALE_K", 1.0f64)),
                ),
                Family::Laplace => Arc::new(Laplace::new(a, b, c, d).with_hessian_mode(m).with_start(st)),
                _ => Arc::new(LogNormal::new(a, b, c, d).with_hessian_mode(m).with_start(st)),
            }
        }
        Family::StudentT => {
            let mut d = StudentT::new(
                cfg.stab,
                ResponseFn::ExpDf,
                ResponseFn::Exp,
                LossFn::Nll,
                cfg.init,
            )
            .with_hessian_mode(mode);
            if let Ok(cap) = std::env::var("BENCH_T_START_DF_CAP") {
                d = d
                    .with_max_start_df(cap.parse().expect("BENCH_T_START_DF_CAP must be a number"));
            }
            if let Some(sc) = param_scales(3) {
                d = d.with_param_gradient_scales([sc[0], sc[1], sc[2]]);
            }
            if let Some(b) = beta_nll() {
                d = d.with_beta_nll(b);
            }
            if let Ok(df) = std::env::var("BENCH_T_START_DF") {
                d = d.with_start_df(df.parse().expect("BENCH_T_START_DF must be a number"));
            }
            match std::env::var("BENCH_T_START").as_deref() {
                Ok("raw") => d = d.with_start_rule(gradientlss::distributions::StudentTStart::Raw),
                Ok("median") => d = d.with_start_rule(gradientlss::distributions::StudentTStart::Median),
                Ok("mad") => d = d.with_start_rule(gradientlss::distributions::StudentTStart::MedianMad),
                Ok("adaptive") => d = d.with_start_rule(gradientlss::distributions::StudentTStart::Adaptive),
                _ => {}
            }
            Arc::new(d)
        }
    }
}

/// BENCH_BETA_NLL=β — β-NLL row weighting for Gaussian/StudentT.
fn beta_nll() -> Option<f64> {
    std::env::var("BENCH_BETA_NLL")
        .ok()
        .map(|v| v.parse().expect("BENCH_BETA_NLL must be a number"))
}

/// BENCH_PES="window[:j,j][/patience][@anneal]" — per-parameter early stopping
/// (all params if no list); "auto" = the crate default; unset/"0" = off.
fn param_early_stopping() -> gradientlss::model::ParamEarlyStoppingSetting {
    use gradientlss::model::ParamEarlyStoppingSetting as S;
    let Ok(raw) = std::env::var("BENCH_PES") else { return S::Off };
    let raw = raw.trim().to_string();
    if raw.is_empty() || raw == "0" {
        return S::Off;
    }
    if raw == "auto" {
        return S::Auto;
    }
    let (raw, anneal) = match raw.split_once('@') {
        Some((r, a)) => (r.to_string(), a.trim().parse().expect("BENCH_PES anneal")),
        None => (raw.clone(), 0.0),
    };
    let (raw, pat) = match raw.split_once('/') {
        Some((r, p)) => (r.to_string(), p.trim().parse().expect("BENCH_PES patience")),
        None => (raw.clone(), 1),
    };
    let (w, ps) = match raw.split_once(':') {
        Some((w, ps)) => (w, Some(ps.split(',').map(|j| j.trim().parse().expect("BENCH_PES param")).collect())),
        None => (raw.as_str(), None),
    };
    S::Custom(
        gradientlss::model::ParamEarlyStopping::new(w.trim().parse().expect("BENCH_PES window"), ps)
            .with_patience(pat)
            .with_anneal(anneal)
            .with_hold_early_stopping(env_or("BENCH_PES_HOLD", 0u8) == 1),
    )
}

/// BENCH_PRESET — research start values handed to `preset_start_values`:
///   t_scale:c          StudentT [0.5, median, ln(c·sd)]
///   gm_fit_shift:δ     Gamma MS fitted start [ln ȳ, ln α̂ + δ]
///   gm_rawmean_fitshape Gamma MS [0.5, ln α̂]
/// Training-split (mean, sd) of the current BENCH_Y_STD cell.
/// Booster params set by the BENCH_HPO refits (applied after the bench's own).
static HPO_OVERRIDE: std::sync::Mutex<Vec<(String, f64)>> = std::sync::Mutex::new(Vec::new());

/// BENCH_HPO: search min_child_weight & lambda on the training split with the
/// cell's model settings. Returns per trial (cv score, mcw, lambda).
/// BENCH_HPO_SPACE=wide (proposal 024) adds max_depth (int 2..=8), subsample and colsample_bytree
/// (uniform [0.5, 1]) to the two axes; every tuned key comes back in the override list.
/// BENCH_HPO_SPACE=depth adds max_depth only, =subs the two subsampling axes only (proposal 027);
/// =bin adds max_bin, =subsbin adds it to `subs` (proposal 033).
/// =bin2 adds max_bin as a choice between 64 and 256 (proposal 034).
/// =eta|gamma|alpha|ctree|clevel|cnode|mds|leaf: one more booster parameter each (proposal 035).
/// =pes: per-parameter early stopping window and anneal via hyper_opt's `__pes_*` axes (proposal 037).
/// =lleaves|lmdl|lff|lbag: LightGBM's num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction (proposal 036).
fn hpo_search(ds: &str, fam: Family, mode: HessianMode, seed: u64, train: &Data, cfg: &Config, n: u32) -> Vec<(f64, Vec<(String, f64)>)> {
    use gradientlss::hyper_opt::{hyper_opt_with_config, HyperOptConfig};
    let mut model = GradientLSS::<Bk>::new(make_dist(fam, mode, cfg));
    model.set_param_early_stopping_setting(param_early_stopping());
    model.set_hessian_normalization(hnorm_setting());
    let mut hp = std::collections::HashMap::new();
    hp.insert("eta".to_string(), serde_json::json!({"choices": [cfg.eta]}));
    hp.insert("max_depth".to_string(), serde_json::json!({"choices": [cfg.depth]}));
    for k in ["min_child_weight", "lambda"] {
        hp.insert(k.to_string(), serde_json::json!({"low": 1e-3, "high": env_or("BENCH_HPO_HIGH", 1e3f64), "log": true}));
    }
    let mut keys = vec!["min_child_weight", "lambda"];
    match std::env::var("BENCH_HPO_SPACE").as_deref() {
        Ok("wide") => {
            hp.insert("max_depth".to_string(), serde_json::json!({"low": 2, "high": 8, "type": "int"}));
            hp.insert("subsample".to_string(), serde_json::json!({"low": 0.5, "high": 1.0}));
            hp.insert("colsample_bytree".to_string(), serde_json::json!({"low": 0.5, "high": 1.0}));
            keys.extend(["max_depth", "subsample", "colsample_bytree"]);
        }
        // proposal 027: the two halves of the wide space, to see which one carries its gain
        Ok("depth") => {
            hp.insert("max_depth".to_string(), serde_json::json!({"low": 2, "high": 8, "type": "int"}));
            keys.push("max_depth");
        }
        Ok("subs") => {
            hp.insert("subsample".to_string(), serde_json::json!({"low": 0.5, "high": 1.0}));
            hp.insert("colsample_bytree".to_string(), serde_json::json!({"low": 0.5, "high": 1.0}));
            keys.extend(["subsample", "colsample_bytree"]);
        }
        // proposal 033: max_bin (log-uniform integer 16..=1024) on top of the two axes, or of `subs`
        Ok("bin") => {
            hp.insert("max_bin".to_string(), serde_json::json!({"low": 16, "high": 1024, "log": true, "type": "int"}));
            keys.push("max_bin");
        }
        // proposal 035: one untested booster parameter at a time on top of the two axes
        Ok("eta") => {
            hp.insert("eta".to_string(), serde_json::json!({"low": 0.02, "high": 0.3, "log": true}));
            keys.push("eta");
        }
        Ok("gamma") => {
            hp.insert("gamma".to_string(), serde_json::json!({"low": 1e-3, "high": 1e2, "log": true}));
            keys.push("gamma");
        }
        Ok("alpha") => {
            hp.insert("alpha".to_string(), serde_json::json!({"low": 1e-3, "high": 1e2, "log": true}));
            keys.push("alpha");
        }
        Ok(k @ ("ctree" | "clevel" | "cnode")) => {
            let name = match k {
                "ctree" => "colsample_bytree",
                "clevel" => "colsample_bylevel",
                _ => "colsample_bynode",
            };
            hp.insert(name.to_string(), serde_json::json!({"low": 0.5, "high": 1.0}));
            keys.push(name);
        }
        Ok("mds") => {
            hp.insert("max_delta_step".to_string(), serde_json::json!({"low": 0.1, "high": 10.0, "log": true}));
            keys.push("max_delta_step");
        }
        // leaf-wise growth: max_leaves replaces the depth limit (run_cell sets grow_policy and max_depth 0
        // whenever max_leaves is in the override list)
        Ok("leaf") => {
            hp.insert("grow_policy".to_string(), serde_json::json!({"choices": ["lossguide"]}));
            hp.insert("max_depth".to_string(), serde_json::json!({"choices": [0]}));
            hp.insert("max_leaves".to_string(), serde_json::json!({"low": 4, "high": 64, "log": true, "type": "int"}));
            keys.push("max_leaves");
        }
        // proposal 036: LightGBM's own parameters, one at a time on top of the two axes
        // (run_cell lifts the depth limit whenever num_leaves is in the override list)
        Ok("lleaves") => {
            hp.insert("max_depth".to_string(), serde_json::json!({"choices": [-1]}));
            hp.insert("num_leaves".to_string(), serde_json::json!({"low": 4, "high": 64, "log": true, "type": "int"}));
            keys.push("num_leaves");
        }
        Ok("lmdl") => {
            hp.insert("min_data_in_leaf".to_string(), serde_json::json!({"low": 1, "high": 100, "log": true, "type": "int"}));
            keys.push("min_data_in_leaf");
        }
        Ok("lff") => {
            hp.insert("feature_fraction".to_string(), serde_json::json!({"low": 0.5, "high": 1.0}));
            keys.push("feature_fraction");
        }
        Ok("lbag") => {
            hp.insert("bagging_fraction".to_string(), serde_json::json!({"low": 0.5, "high": 1.0}));
            keys.push("bagging_fraction");
        }
        // proposal 037: per-parameter early stopping as two search axes (window 0 = off)
        Ok("pes") => {
            hp.insert("__pes_window".to_string(), serde_json::json!({"choices": [0, 5, 10, 20, 40, 80]}));
            hp.insert("__pes_anneal".to_string(), serde_json::json!({"low": 0.0, "high": 0.9}));
            keys.extend(["__pes_window", "__pes_anneal"]);
        }
        // proposal 034: a two-value choice instead of the range
        Ok("bin2") => {
            hp.insert("max_bin".to_string(), serde_json::json!({"choices": [64, 256]}));
            keys.push("max_bin");
        }
        Ok("subsbin") => {
            hp.insert("subsample".to_string(), serde_json::json!({"low": 0.5, "high": 1.0}));
            hp.insert("colsample_bytree".to_string(), serde_json::json!({"low": 0.5, "high": 1.0}));
            hp.insert("max_bin".to_string(), serde_json::json!({"low": 16, "high": 1024, "log": true, "type": "int"}));
            keys.extend(["subsample", "colsample_bytree", "max_bin"]);
        }
        Ok("mcw_lambda") | Err(_) => {}
        Ok(other) => panic!("unknown BENCH_HPO_SPACE {other}"),
    }
    // BENCH_HPO_SAMPLER=tpe|random|startup:<n>;  BENCH_HPO_INIT=default evaluates the untuned
    // (mcw = BENCH_MCW, lambda = 1) configuration as trial 0 on the search's own folds.
    let sampler = match std::env::var("BENCH_HPO_SAMPLER").as_deref() {
        Ok("random") => gradientlss::hyper_opt::Sampler::Random,
        Ok(s) if s.starts_with("startup:") => gradientlss::hyper_opt::Sampler::TpeAfterRandom {
            n_startup: s["startup:".len()..].parse().expect("BENCH_HPO_SAMPLER=startup:<n>"),
        },
        Ok("tpe") | Err(_) => gradientlss::hyper_opt::Sampler::Tpe,
        Ok(other) => panic!("unknown BENCH_HPO_SAMPLER {other}"),
    };
    let initial_trials = match std::env::var("BENCH_HPO_INIT").as_deref() {
        Ok("default") => vec![
            [("min_child_weight".to_string(), serde_json::json!(cfg.mcw)), ("lambda".to_string(), serde_json::json!(1.0))]
                .into_iter()
                .collect(),
        ],
        Err(_) => vec![],
        Ok(other) => panic!("unknown BENCH_HPO_INIT {other}"),
    };
    let config = HyperOptConfig {
        n_trials: n,
        n_folds: env_or("BENCH_HPO_FOLDS", 3usize),
        sampler,
        initial_trials,
        seed,
        hp_seed: Some(seed),
        num_boost_round: cfg.rounds,
        early_stopping_rounds: if cfg.es == 0 { None } else { Some(cfg.es) },
        verbose: false,
        ..Default::default()
    };
    let t0 = Instant::now();
    let res = hyper_opt_with_config(&mut model, &train.x, &train.y, &hp, config).expect("hyper_opt");
    let out: Vec<(f64, Vec<(String, f64)>)> = res
        .trials
        .iter()
        .map(|t| {
            let g = |k: &str| t.params.get(k).and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
            (if t.pruned { f64::NAN } else { t.score }, keys.iter().map(|k| (k.to_string(), g(k))).collect())
        })
        .collect();
    for (i, (sc, kv)) in out.iter().enumerate() {
        // the first two keys are always mcw and lambda; the wide space appends the rest
        let extra: String = kv[2..].iter().map(|(k, v)| format!(" {k} {v}")).collect();
        eprintln!("HPO {ds} {} {seed} trial {i} cv {sc} mcw {} lambda {}{extra}", fam.name(), kv[0].1, kv[1].1);
    }
    eprintln!("HPO_SECS {ds} {} {seed} {:.1}", fam.name(), t0.elapsed().as_secs_f64());
    out
}

fn hnorm_setting() -> gradientlss::model::HessianNormalization {
    match std::env::var("BENCH_HNORM").as_deref() {
        Ok("start") => gradientlss::model::HessianNormalization::Start,
        Ok("marginal") => gradientlss::model::HessianNormalization::Marginal,
        _ => gradientlss::model::HessianNormalization::Off,
    }
}

static YSTD_STATS: std::sync::Mutex<Option<(f64, f64)>> = std::sync::Mutex::new(None);

fn preset_start(fam: Family, y: &Array1<f64>) -> Option<Array1<f64>> {
    let raw = std::env::var("BENCH_PRESET").ok()?;
    let (kind, arg) = raw.split_once(':').map(|(a, b)| (a.to_string(), b.parse::<f64>().ok())).unwrap_or((raw.clone(), None));
    let target = gradientlss::types::ResponseData::Univariate(&y.view());
    match (fam, kind.as_str()) {
        (Family::StudentT, "t_scale") => {
            let mut v = y.to_vec();
            v.sort_by(|a, b| a.total_cmp(b));
            let med = v[v.len() / 2];
            let sd = y.std(0.0);
            Some(Array1::from_vec(vec![0.5, med, (arg.expect("t_scale:c") * sd).ln()]))
        }
        (Family::GammaMs, k) if k == "gm_fit_shift" || k == "gm_rawmean_fitshape" => {
            let fit = Gamma::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);
            let (_, sv) = fit.calculate_start_values(&target, 50).expect("gamma start");
            Some(if k == "gm_fit_shift" {
                Array1::from_vec(vec![sv[0], sv[1] + arg.expect("gm_fit_shift:δ")])
            } else {
                Array1::from_vec(vec![0.5, sv[1]])
            })
        }
        (Family::Gaussian, "g_raw_in_std") => {
            // the native raw start [0.5, ln 1.65] mapped into standardized units
            let (m, sd) = YSTD_STATS.lock().unwrap().expect("g_raw_in_std needs BENCH_Y_STD=1");
            Some(Array1::from_vec(vec![(0.5 - m) / sd, 0.5 - sd.ln()]))
        }
        (Family::Gaussian, "g_std_equiv") => {
            // the standardized run's default start [0.5, 0.5], mapped to data units
            let sd = y.std(0.0);
            Some(Array1::from_vec(vec![y.mean().unwrap() + 0.5 * sd, (0.5f64.exp() * sd).ln()]))
        }
        (Family::Gaussian, "g_sd") => {
            let c: f64 = arg.expect("g_sd:c");
            Some(Array1::from_vec(vec![y.mean().unwrap(), (c * y.std(0.0)).ln()]))
        }
        (Family::GammaMs, "gm_wide") => {
            let a = raw.split_once(':').expect("gm_wide:dm/ls").1;
            let (dm, ls) = a.split_once('/').expect("gm_wide:dm/ls");
            let (dm, ls): (f64, f64) = (dm.parse().unwrap(), ls.parse().unwrap());
            let mean = y.mean().unwrap();
            Some(Array1::from_vec(vec![mean.ln() + dm, ls]))
        }
        _ => panic!("BENCH_PRESET {raw} not valid for {}", fam.name()),
    }
}

/// BENCH_FREEZE="j:round[,j:round]" — per-parameter stopping rounds.
fn freeze_rounds(n_params: usize) -> Option<Vec<Option<usize>>> {
    let raw = std::env::var("BENCH_FREEZE").ok()?;
    let mut v = vec![None; n_params];
    for item in raw.split(',') {
        let (j, r) = item.split_once(':').expect("BENCH_FREEZE items are j:round");
        let j: usize = j.trim().parse().unwrap();
        if j < n_params {
            v[j] = Some(r.trim().parse().unwrap());
        }
    }
    Some(v)
}

#[allow(dead_code)]
struct Row {
    dataset: String,
    fam: Family,
    mode: HessianMode,
    seed: u64,
    n_train: usize,
    n_test: usize,
    best_iter: i64,
    n_iter: usize,
    scores: Scores,
    secs: f64,
    error: String,
}

#[allow(clippy::too_many_arguments)]
fn run_cell(
    dataset: &str,
    fam: Family,
    mode: HessianMode,
    seed: u64,
    train: &Data,
    valid: &Data,
    test: &Data,
    cfg: &Config,
) -> Row {
    let dist = make_dist(fam, mode, cfg);
    let mut model = GradientLSS::<Bk>::new(dist.clone());
    let mut params = Bk::create_params(model.n_params());
    params.set("eta", ParamValue::Float(cfg.eta));
    params.set("max_depth", ParamValue::Int(cfg.depth));
    params.set("min_child_weight", ParamValue::Float(cfg.mcw));
    // BENCH_MAX_BIN=k (proposal 033): histogram bins per feature for the untuned fits (backend default 256)
    if let Ok(k) = std::env::var("BENCH_MAX_BIN") {
        params.set("max_bin", ParamValue::Int(k.parse().expect("BENCH_MAX_BIN")));
    }
    // BENCH_XGB=key=value,... (proposal 035): extra booster parameters for the untuned fits
    // (integers and floats by their spelling, anything else as a string)
    if let Ok(list) = std::env::var("BENCH_XGB") {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("BENCH_XGB=key=value,...");
            if let Ok(i) = v.parse::<i64>() {
                params.set(k, ParamValue::Int(i as _));
            } else if let Ok(f) = v.parse::<f64>() {
                params.set(k, ParamValue::Float(f));
            } else {
                params.set(k, ParamValue::String(v.to_string()));
            }
        }
    }
    for (k, v) in HPO_OVERRIDE.lock().unwrap().iter() {
        if k.starts_with("__") {
            continue; // model-level axes (`__pes_*`), applied where the model is built
        }
        if k == "max_leaves" {
            params.set("grow_policy", ParamValue::String("lossguide".to_string()));
            params.set("max_depth", ParamValue::Int(0));
        }
        if k == "num_leaves" {
            params.set("max_depth", ParamValue::Int(-1));
        }
        if k == "max_depth" || k == "max_bin" || k == "max_leaves" || k == "num_leaves" || k == "min_data_in_leaf" {
            params.set(k, ParamValue::Int(*v as _));
        } else {
            params.set(k, ParamValue::Float(*v));
        }
    }
    let config = TrainConfig {
        num_boost_round: cfg.rounds,
        // BENCH_ES=0 disables early stopping: fixed-round trajectories use every tree.
        early_stopping_rounds: if cfg.es == 0 { None } else { Some(cfg.es) },
        verbose: false,
        collect_train_metrics: false,
        seed,
        param_freeze_rounds: freeze_rounds(dist.n_params()),
    };
    let mut row = Row {
        dataset: dataset.to_string(),
        fam,
        mode,
        seed,
        n_train: train.y.len(),
        n_test: test.y.len(),
        best_iter: -1,
        n_iter: 0,
        scores: Scores::default(),
        secs: 0.0,
        error: String::new(),
    };
    let warm_mode = std::env::var("BENCH_WARM_START").unwrap_or_default();
    let warm = fam == Family::StudentT && warm_mode == "1";
    // "api": the crate's own two-stage path (validation residuals, default start cap)
    let warm_api = fam == Family::StudentT && warm_mode == "api";
    let t0 = Instant::now();
    let result = (|| -> Result<Scores> {
        let mut tr =
            <Bk as Backend>::Dataset::from_data(train.x.view(), train.y.view())?;
        let mut va =
            <Bk as Backend>::Dataset::from_data(valid.x.view(), valid.y.view())?;
        // Residual warm start: per-row raw-space offsets for [df, loc, scale].
        let offsets = if warm {
            // stage 1: Gaussian location with the same backend settings
            let gdist: Arc<dyn Distribution> =
                Arc::new(Gaussian::new(cfg.stab, ResponseFn::Exp, LossFn::Nll, cfg.init));
            let mut g = GradientLSS::<Bk>::new(gdist);
            let mut gparams = Bk::create_params(2);
            gparams.set("eta", ParamValue::Float(cfg.eta));
            gparams.set("max_depth", ParamValue::Int(cfg.depth));
            gparams.set("min_child_weight", ParamValue::Float(cfg.mcw));
            let gconfig = TrainConfig {
                num_boost_round: cfg.rounds,
                early_stopping_rounds: if cfg.es == 0 { None } else { Some(cfg.es) },
                verbose: false,
                collect_train_metrics: false,
                seed,
                param_freeze_rounds: None,
            };
            let mut gtr =
                <Bk as Backend>::Dataset::from_data(train.x.view(), train.y.view())?;
            let mut gva =
                <Bk as Backend>::Dataset::from_data(valid.x.view(), valid.y.view())?;
            g.train_with_callbacks(&mut gtr, Some(&mut gva), gparams, gconfig, None::<&mut CallbackList>)?;
            let loc_of = |x: &Array2<f64>| -> Result<Array1<f64>> {
                match g.predict(&x.view(), PredType::Parameters, 0, &[], seed)? {
                    gradientlss::backend::PredictionOutput::Parameters(p) => Ok(p.column(0).to_owned()),
                    _ => unreachable!(),
                }
            };
            let (yh_tr, yh_va, yh_te) = (loc_of(&train.x)?, loc_of(&valid.x)?, loc_of(&test.x)?);
            // stage 2: unconditional t on residuals → (ν₀, µ_r, σ₀). Training
            // residuals are in-sample (overfit ⇒ leptokurtic ⇒ ν₀ too small);
            // BENCH_WARM_RESID=valid uses the honest validation residuals.
            let resid = if std::env::var("BENCH_WARM_RESID").as_deref() == Ok("valid") {
                &valid.y - &yh_va
            } else {
                &train.y - &yh_tr
            };
            let tfit = StudentT::new(Stabilization::None, ResponseFn::ExpDf, ResponseFn::Exp, LossFn::Nll, true);
            let (_, sv) = tfit.calculate_start_values(
                &gradientlss::types::ResponseData::Univariate(&resid.view()),
                50,
            )?;
            let mut raw_df = sv[0];
            if let Ok(cap) = std::env::var("BENCH_T_START_DF_CAP") {
                let cap: f64 = cap.parse().unwrap();
                raw_df = raw_df.min((cap - 2.0).ln());
            }
            if env_or("BENCH_DEBUG", 0u8) == 1 {
                eprintln!(
                    "DEBUG warm start: ν₀ = {:.4e}, µ_r = {:.4}, σ₀ = {:.4} (resid sd {:.4}, y sd {:.4})",
                    raw_df.exp() + 2.0, sv[1], sv[2].exp(), resid.std(0.0), train.y.std(0.0)
                );
            }
            // the StudentT below uses initialize:false → start values are raw 0.5 → subtract them
            let mk = |yh: &Array1<f64>| {
                Array2::from_shape_fn((yh.len(), 3), |(i, j)| match j {
                    0 => raw_df - 0.5,
                    1 => yh[i] + sv[1] - 0.5,
                    _ => sv[2] - 0.5,
                })
            };
            Some((mk(&yh_tr), mk(&yh_va), mk(&yh_te)))
        } else {
            None
        };
        if warm {
            // rebuild the model with initialize:false so the offsets carry the whole start
            let mut d =
                StudentT::new(cfg.stab, ResponseFn::ExpDf, ResponseFn::Exp, LossFn::Nll, false)
                    .with_hessian_mode(mode)
                    .with_start_rule(gradientlss::distributions::StudentTStart::Raw);
            if let Some(sc) = param_scales(3) {
                d = d.with_param_gradient_scales([sc[0], sc[1], sc[2]]);
            }
            model = GradientLSS::<Bk>::new(Arc::new(d));
        }
        if warm_api {
            model.set_param_early_stopping_setting(param_early_stopping());
            let tres = model.train_with_residual_warm_start(
                &mut tr,
                Some(&mut va),
                params,
                config,
                gradientlss::model::ResidualWarmStart::default(),
            )?;
            row.best_iter = tres.best_iteration.map(|b| b as i64).unwrap_or(-1);
            row.n_iter = tres.n_iterations;
            let out = model.predict(&test.x.view(), PredType::Parameters, 0, &[], seed)?;
            let p = match out {
                gradientlss::backend::PredictionOutput::Parameters(p) => p,
                _ => unreachable!(),
            };
            let mut sc = score(dist.as_ref(), &p.view(), test, fam);
            sc.valid_nll = tres.best_score.unwrap_or(f64::NAN);
            return Ok(sc);
        }
        model.set_param_early_stopping_setting(param_early_stopping());
        // proposal 037: `__pes_*` values picked by a search configure the refit the way hyper_opt
        // configures its fold models (on top of the model's own setting)
        {
            let trial: std::collections::HashMap<String, serde_json::Value> = HPO_OVERRIDE
                .lock()
                .unwrap()
                .iter()
                .filter(|(k, _)| k.starts_with("__pes_"))
                .map(|(k, v)| (k.clone(), serde_json::json!(*v)))
                .collect();
            if !trial.is_empty() {
                let base = model.effective_param_early_stopping();
                let dflt = model.distribution().default_param_early_stopping_params();
                model.set_param_early_stopping(gradientlss::model::ParamEarlyStopping::from_trial_params(base.as_ref(), dflt, &trial));
            }
        }
        model.set_warmup_rounds(env_or("BENCH_WARMUP", 0usize));
        model.set_hessian_normalization(hnorm_setting());
        // BENCH_GRN=<c> | x:<c> (proposal 030): gradient-regularised Newton, x: = excess-only form
        if let Ok(v) = std::env::var("BENCH_GRN") {
            let (excess_only, c) = match v.strip_prefix("x:") {
                Some(c) => (true, c),
                None => (false, v.as_str()),
            };
            model.set_gradient_regularization(Some(gradientlss::model::GradientRegularization {
                strength: c.parse().expect("BENCH_GRN=<c> or x:<c>"),
                excess_only,
            }));
        }
        if let Ok(k) = std::env::var("BENCH_STOP_CAP") {
            model.set_stopping_row_cap(Some(k.parse().expect("BENCH_STOP_CAP")));
        }
        if fam == Family::StudentT && std::env::var("BENCH_T_START").as_deref() == Ok("adaptive") {
            let (sd, g2) = gradientlss::distributions::sd_and_excess_kurtosis(train.y.as_slice().unwrap());
            eprintln!("KURT {dataset} seed {seed} g2 {g2:.3} sd {sd:.4} inflate {}", g2 < 1.0);
        }
        if let Some(sv) = preset_start(fam, &train.y) {
            model.preset_start_values(sv);
        }
        let (_, tres) = match &offsets {
            Some((otr, ova, _)) => model.train_with_callbacks_and_offsets(
                &mut tr,
                Some(&mut va),
                params,
                config,
                None::<&mut CallbackList>,
                Some(otr.view()),
                Some(ova.view()),
            )?,
            None => model.train_with_callbacks(
                &mut tr,
                Some(&mut va),
                params,
                config,
                None::<&mut CallbackList>,
            )?,
        };
        if let Some(log) = model.param_freeze_log() {
            eprintln!(
                "PES_FREEZE {dataset} {} {seed} {log:?} triggers {:?} best_iter {:?}",
                fam.name(),
                model.param_early_stopping_triggers(),
                tres.best_iteration
            );
        }
        row.best_iter = tres.best_iteration.map(|b| b as i64).unwrap_or(-1);
        row.n_iter = tres.n_iterations;
        let out = match &offsets {
            Some((_, _, ote)) => model.predict_with_offset(
                &test.x.view(),
                Some(ote.view()),
                PredType::Parameters,
                0,
                &[],
                seed,
            )?,
            None => model.predict(&test.x.view(), PredType::Parameters, 0, &[], seed)?,
        };
        let p = match out {
            gradientlss::backend::PredictionOutput::Parameters(p) => p,
            _ => unreachable!(),
        };
        if env_or("BENCH_DEBUG", 0u8) == 1 {
            let vh: Vec<String> = tres
                .valid_history
                .iter()
                .take(8)
                .map(|v| format!("{v:.4}"))
                .collect();
            eprintln!(
                "DEBUG {dataset} {fam:?} {mode:?} seed {seed}: n_iter {} best {:?} valid_history[..8] {:?} last {:?}",
                tres.n_iterations,
                tres.best_iteration,
                vh,
                tres.valid_history.last()
            );
            eprintln!(
                "DEBUG start values (raw): {:?}",
                model
                    .start_values()
                    .map(|v| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>())
            );
            for j in 0..p.ncols() {
                let col = p.column(j);
                let finite = col.iter().filter(|v| v.is_finite()).count();
                let mn = col
                    .iter()
                    .cloned()
                    .filter(|v| v.is_finite())
                    .fold(f64::INFINITY, f64::min);
                let mx = col
                    .iter()
                    .cloned()
                    .filter(|v| v.is_finite())
                    .fold(f64::NEG_INFINITY, f64::max);
                eprintln!(
                    "DEBUG   param {j}: finite {finite}/{} min {mn:.4e} max {mx:.4e}",
                    col.len()
                );
            }
        }
        let mut sc = score(dist.as_ref(), &p.view(), test, fam);
        sc.valid_nll = tres.best_score.unwrap_or(f64::NAN);
        Ok(sc)
    })();
    row.secs = t0.elapsed().as_secs_f64();
    match result {
        Ok(s) => row.scores = s,
        Err(e) => {
            row.error = e.to_string().replace([',', '\n'], ";");
            row.scores.nll = f64::NAN;
            row.scores.crps = f64::NAN;
        }
    }
    row
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

const HEADER: &str = "commit,date,tag,dataset,family,hessian_mode,seed,eta,depth,mcw,stab,init,rounds_cap,es,n_train,n_test,best_iter,n_iter,nll,crps,cov50,cov90,pit_ks,rmse,mean_scale,oracle_nll,oracle_crps,nonfinite_rows,fit_secs,error,valid_nll,df_q05,df_med,scale_max";

fn selftest() {
    // CRPS closed forms vs Monte Carlo (E|X−y| − ½E|X−X′|) at a fixed point.
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let (mu, sigma, nu, y) = (0.4, 1.3, 4.0, 1.1);
    let m = 400_000;
    let t = RandT::new(nu).unwrap();
    let xs: Vec<f64> = (0..m).map(|_| mu + sigma * t.sample(&mut rng)).collect();
    let xs2: Vec<f64> = (0..m).map(|_| mu + sigma * t.sample(&mut rng)).collect();
    let e1 = xs.iter().map(|x| (x - y).abs()).sum::<f64>() / m as f64;
    let e2 = xs.iter().zip(&xs2).map(|(a, b)| (a - b).abs()).sum::<f64>() / m as f64;
    println!(
        "selftest t-CRPS: closed {:.5} vs MC {:.5}",
        crps_student_t(y, mu, sigma, nu),
        e1 - 0.5 * e2
    );
    let nd = Normal::new(mu, sigma).unwrap();
    let xs: Vec<f64> = (0..m).map(|_| nd.sample(&mut rng)).collect();
    let xs2: Vec<f64> = (0..m).map(|_| nd.sample(&mut rng)).collect();
    let e1 = xs.iter().map(|x| (x - y).abs()).sum::<f64>() / m as f64;
    let e2 = xs.iter().zip(&xs2).map(|(a, b)| (a - b).abs()).sum::<f64>() / m as f64;
    println!(
        "selftest N-CRPS: closed {:.5} vs MC {:.5}",
        crps_gaussian(y, mu, sigma),
        e1 - 0.5 * e2
    );
    for &(a, b, yy) in &[(2.5, 0.8, 1.7), (330.0, 330.0 / 450.0, 470.0)] {
        let g = RandGammaD::new(a, 1.0 / b).unwrap();
        let xs: Vec<f64> = (0..m).map(|_| g.sample(&mut rng)).collect();
        let xs2: Vec<f64> = (0..m).map(|_| g.sample(&mut rng)).collect();
        let e1 = xs.iter().map(|x| (x - yy).abs()).sum::<f64>() / m as f64;
        let e2 = xs.iter().zip(&xs2).map(|(p, q)| (p - q).abs()).sum::<f64>() / m as f64;
        println!(
            "selftest Γ-CRPS(a={a}): closed {:.5} vs MC {:.5}",
            crps_gamma(yy, a, b),
            e1 - 0.5 * e2
        );
    }
}

fn main() {
    let cfg = config();
    if env_or("BENCH_SELFTEST", 0u8) == 1 {
        selftest();
    }
    let commit = git_commit();
    let date = chrono_lite_now();
    let stab_name = format!("{:?}", cfg.stab).to_lowercase();

    let need_header = !cfg.out.exists();
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cfg.out)
        .unwrap();
    if need_header {
        writeln!(out, "{HEADER}").unwrap();
    }

    // dataset -> family -> mode -> seed -> (nll, crps)
    let mut paired: BTreeMap<(String, Family), BTreeMap<u64, BTreeMap<String, Scores>>> =
        BTreeMap::new();

    println!(
        "{:<13} {:<9} {:<8} {:>4} {:>7} {:>5} {:>9} {:>9} {:>6} {:>6} {:>7} {:>8} {:>8} {:>5} {:>6}",
        "dataset",
        "family",
        "mode",
        "seed",
        "ntrain",
        "iters",
        "nll",
        "crps",
        "cov50",
        "cov90",
        "pit_ks",
        "rmse",
        "oracle",
        "nonf",
        "secs"
    );
    for ds in &cfg.datasets {
        // BENCH_SEED_START=k resumes a stream that was cut after seed k − 1 (seeds k..BENCH_SEEDS)
        for seed in env_or("BENCH_SEED_START", 0u64)..cfg.seeds {
            let (mut train, mut valid, mut test) = split(load(ds, &cfg, seed), 1000 + seed);
            if env_or("BENCH_Y_STD", 0u8) == 1 {
                // training-split statistics only; ln(sd) converts the NLL back
                let (m, sd) = (train.y.mean().unwrap(), train.y.std(0.0));
                eprintln!("YSTD {ds} seed {seed} mean {m} sd {sd} ln_sd {}", sd.ln());
                *YSTD_STATS.lock().unwrap() = Some((m, sd));
                for d in [&mut train, &mut valid, &mut test] {
                    d.y.mapv_inplace(|v| (v - m) / sd);
                    d.oracle = None;
                }
            }
            if env_or("BENCH_DIST_SELECT", 0u8) == 1 {
                // The unconditional fit `dist_select` ranks families by, on the training labels
                // only. Nothing is trained; one DSEL line per family on stderr.
                let yv = train.y.view();
                for &fam in &cfg.families {
                    let d = make_dist(fam, cfg.modes[0], &cfg);
                    let n = train.y.len();
                    match d.marginal_fit(&gradientlss::types::ResponseData::Univariate(&yv), env_or("BENCH_DSEL_ITERS", 100usize)) {
                        Ok((nll, sv)) => eprintln!("DSEL {ds} {seed} {} n {n} nll {nll} start {:?}", fam.name(), sv.to_vec()),
                        Err(e) => eprintln!("DSEL {ds} {seed} {} n {n} ERROR {e}", fam.name()),
                    }
                }
                continue;
            }
            for &fam in &cfg.families {
                for &mode in &cfg.modes {
                    // (tag, booster overrides): one plain cell, or one refit per HPO budget
                    let hpo_n: u32 = env_or("BENCH_HPO", 0u32);
                    let variants: Vec<(String, Vec<(String, f64)>)> = if hpo_n == 0 {
                        vec![(cfg.tag.clone(), vec![])]
                    } else {
                        let trials = hpo_search(ds, fam, mode, seed, &train, &cfg, hpo_n);
                        env_list("BENCH_HPO_CHECK", "5,10,20")
                            .iter()
                            .map(|k| k.parse::<usize>().unwrap())
                            .filter(|&k| k <= trials.len())
                            .map(|k| {
                                let best = trials[..k]
                                    .iter()
                                    .filter(|t| t.0.is_finite())
                                    .min_by(|a, b| a.0.total_cmp(&b.0))
                                    .cloned()
                                    .unwrap_or((f64::NAN, vec![("min_child_weight".to_string(), cfg.mcw), ("lambda".to_string(), 1.0)]));
                                eprintln!("HPO_BEST {ds} {} {seed} k {k} cv {} mcw {} lambda {}", fam.name(), best.0, best.1[0].1, best.1[1].1);
                                (format!("{}_k{k}", cfg.tag), best.1)
                            })
                            .collect()
                    };
                    for (vtag, ovr) in variants {
                    *HPO_OVERRIDE.lock().unwrap() = ovr.clone();
                    let vmcw = ovr.iter().find(|(k, _)| k == "min_child_weight").map(|x| x.1).unwrap_or(cfg.mcw);
                    let row = run_cell(ds, fam, mode, seed, &train, &valid, &test, &cfg);
                    let s = &row.scores;
                    let mode_name = format!("{mode:?}").to_lowercase();
                    println!(
                        "{:<13} {:<9} {:<8} {:>4} {:>7} {:>5} {:>9.4} {:>9.4} {:>6.3} {:>6.3} {:>7.4} {:>8.4} {:>8.4} {:>5} {:>6.1}{}",
                        ds,
                        fam.name(),
                        mode_name,
                        seed,
                        row.n_train,
                        row.best_iter + 1,
                        s.nll,
                        s.crps,
                        s.cov50,
                        s.cov90,
                        s.pit_ks,
                        s.rmse,
                        s.oracle_nll,
                        s.nonfinite,
                        row.secs,
                        if row.error.is_empty() {
                            String::new()
                        } else {
                            format!("  ERROR {}", row.error)
                        }
                    );
                    writeln!(
                        out,
                        "{commit},{date},{tag},{ds},{fam},{mode},{seed},{eta},{depth},{mcw},{stab},{init},{rounds},{es},{ntr},{nte},{bi},{ni},{nll},{crps},{c50},{c90},{ks},{rmse},{ms},{onll},{ocrps},{nf},{secs:.3},{err},{vnll},{dfq},{dfm},{smax}",
                        tag = vtag, fam = fam.name(), mode = mode_name, eta = cfg.eta, depth = cfg.depth, mcw = vmcw,
                        stab = stab_name, init = cfg.init as u8, rounds = cfg.rounds, es = cfg.es, ntr = row.n_train, nte = row.n_test,
                        bi = row.best_iter, ni = row.n_iter, nll = s.nll, crps = s.crps, c50 = s.cov50, c90 = s.cov90, ks = s.pit_ks,
                        rmse = s.rmse, ms = s.mean_scale, onll = s.oracle_nll, ocrps = s.oracle_crps, nf = s.nonfinite, secs = row.secs, err = row.error,
                        vnll = s.valid_nll, dfq = s.df_q05, dfm = s.df_med, smax = s.scale_max
                    )
                    .unwrap();
                    paired
                        .entry((ds.clone(), fam))
                        .or_default()
                        .entry(seed)
                        .or_default()
                        .insert(mode_name, row.scores.clone());
                    }
                    *HPO_OVERRIDE.lock().unwrap() = Vec::new();
                }
            }
        }
    }

    // Paired summary: fisher − observed per seed.
    if cfg.modes.len() == 2 {
        println!(
            "\n== paired summary (fisher − observed), mean ± s.e. over seeds; negative = Fisher better =="
        );
        println!(
            "{:<13} {:<9} {:>5} {:>22} {:>22} {:>14} {:>12}",
            "dataset", "family", "n", "Δnll", "Δcrps", "Δcov90", "Δrmse"
        );
        for ((ds, fam), by_seed) in &paired {
            let mut d_nll = vec![];
            let mut d_crps = vec![];
            let mut d_cov = vec![];
            let mut d_rmse = vec![];
            for modes in by_seed.values() {
                if let (Some(f), Some(o)) = (modes.get("fisher"), modes.get("observed"))
                    && f.nll.is_finite()
                    && o.nll.is_finite()
                {
                    d_nll.push(f.nll - o.nll);
                    d_crps.push(f.crps - o.crps);
                    d_cov.push(f.cov90 - o.cov90);
                    d_rmse.push(f.rmse - o.rmse);
                }
            }
            let ms = |v: &[f64]| {
                let n = v.len() as f64;
                let m = v.iter().sum::<f64>() / n;
                let se = if v.len() > 1 {
                    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()
                } else {
                    f64::NAN
                };
                (m, se)
            };
            let (a, ase) = ms(&d_nll);
            let (b, bse) = ms(&d_crps);
            let (c, cse) = ms(&d_cov);
            let (d, dse) = ms(&d_rmse);
            println!(
                "{:<13} {:<9} {:>5} {:>+10.4} ± {:<9.4} {:>+10.4} ± {:<9.4} {:>+6.3} ± {:<5.3} {:>+6.3} ± {:<5.3}",
                ds,
                fam.name(),
                d_nll.len(),
                a,
                ase,
                b,
                bse,
                c,
                cse,
                d,
                dse
            );
        }
    }
    println!("\nrows appended to {}", cfg.out.display());
}

/// ISO date without pulling in chrono.
fn chrono_lite_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let days = secs / 86400;
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[allow(dead_code)]
fn _unused(_: &Array2<f64>) {
    let _ = s![..1, ..];
}
