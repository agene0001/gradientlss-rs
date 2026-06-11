//! Profiling harness for the XGBoost + NegativeBinomial training path.
//!
//! Trains on a synthetic NB-distributed dataset and reports a phase-level
//! timing breakdown: total training vs. the distribution-side work
//! (gradients/hessians, NLL metric) measured in isolation on the same data,
//! so the remainder attributes to the backend (tree building + predict).
//!
//! Run:  cargo run --release --features xgboost --example profile_nb_xgb
//! For a sampled flamegraph, run under `samply record` with symbols kept:
//!   CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=true \
//!     cargo build --release --features xgboost --example profile_nb_xgb
//!   samply record --save-only -o /tmp/nb_profile.json \
//!     ./target/release/examples/profile_nb_xgb

use gradientlss::backend::BackendDataset;
use gradientlss::distributions::{Distribution, NegativeBinomial};
use gradientlss::prelude::*;
use gradientlss::types::ResponseData;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let n: usize = std::env::var("PROFILE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let n_features: usize = 20;
    let rounds: usize = std::env::var("PROFILE_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);

    println!("n={n}, features={n_features}, rounds={rounds}");

    // Deterministic synthetic features and NB-ish counts. The exact data
    // distribution barely matters for profiling; what matters is realistic
    // integer counts with a spread of magnitudes (exercises nb_psi_diff's
    // integer fast path and the digamma fallback for larger k).
    let t0 = Instant::now();
    let mut feats = Array2::<f64>::zeros((n, n_features));
    let mut labels = Array1::<f64>::zeros(n);
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for i in 0..n {
        let mut lin = 0.0;
        for j in 0..n_features {
            let x = next();
            feats[[i, j]] = x;
            if j < 5 {
                lin += x;
            }
        }
        // counts roughly in [0, ~40], mean rising with the signal
        let mean = (0.5 * lin).exp();
        let u = next();
        labels[i] = (mean * (-(1.0 - u).ln()) * 4.0).floor();
    }
    println!("data gen:           {:>10.3?}", t0.elapsed());

    let dist = Arc::new(NegativeBinomial::default());

    // --- Isolated distribution-side timings on full-size arrays ---------
    // Skipped when PROFILE_MICRO=0 so a sampling profiler sees training only.
    let run_micro = std::env::var("PROFILE_MICRO").as_deref() != Ok("0");
    let preds = Array2::<f64>::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 { 1.5 + (i % 7) as f64 * 0.1 } else { 0.3 }
    });
    let target = ResponseData::Univariate(&labels.view());

    let mut grad_per_call = std::time::Duration::ZERO;
    let mut nll_per_call = std::time::Duration::ZERO;
    if run_micro {
    let t = Instant::now();
    let mut iters = 0u32;
    while t.elapsed().as_millis() < 1500 {
        let gh = dist
            .compute_gradients_and_hessians(&preds.view(), &target, None)
            .unwrap();
        std::hint::black_box(&gh);
        iters += 1;
    }
    grad_per_call = t.elapsed() / iters;
    println!("gradients+hessians: {:>10.3?} per call ({iters} calls)", grad_per_call);

    let transformed = dist.transform_params(&preds.view());
    let t = Instant::now();
    let mut iters = 0u32;
    while t.elapsed().as_millis() < 1500 {
        std::hint::black_box(dist.nll(&transformed.view(), &target));
        iters += 1;
    }
    nll_per_call = t.elapsed() / iters;
    println!("nll:                {:>10.3?} per call ({iters} calls)", nll_per_call);

    let t = Instant::now();
    let mut iters = 0u32;
    while t.elapsed().as_millis() < 1500 {
        std::hint::black_box(dist.transform_params(&preds.view()));
        iters += 1;
    }
    println!(
        "transform_params:   {:>10.3?} per call ({iters} calls)",
        t.elapsed() / iters
    );
    } // run_micro

    // --- End-to-end training --------------------------------------------
    let mut model = GradientLSS::<XGBoostBackend>::new(dist.clone());
    let t = Instant::now();
    let mut train_data =
        <XGBoostBackend as Backend>::Dataset::from_data(feats.view(), labels.view()).unwrap();
    println!("dataset build:      {:>10.3?}", t.elapsed());

    let params = XGBoostBackend::create_params(model.n_params());
    let config = TrainConfig {
        num_boost_round: rounds,
        early_stopping_rounds: None,
        verbose: false,
        collect_train_metrics: false,
        seed: 123,
    };

    let t = Instant::now();
    model.train(&mut train_data, None, params, config).unwrap();
    let train_time = t.elapsed();
    println!("train ({rounds} rounds): {:>10.3?}", train_time);

    let t = Instant::now();
    let p = model
        .predict(&feats.view(), PredType::Parameters, 0, &[], 123)
        .unwrap();
    std::hint::black_box(&p);
    println!("predict(Parameters):{:>10.3?}", t.elapsed());

    if run_micro {
        // Attribution: per round the objective does one gradient call and the
        // metric does one transform+nll on train predictions.
        let dist_side = (grad_per_call + nll_per_call) * rounds as u32;
        println!("\n--- attribution over {rounds} rounds ---");
        println!("distribution side (grad + nll): {:>10.3?}  ({:.1}%)",
            dist_side, 100.0 * dist_side.as_secs_f64() / train_time.as_secs_f64());
        println!("backend side (trees + predict): {:>10.3?}  ({:.1}%)",
            train_time.saturating_sub(dist_side),
            100.0 * (train_time.saturating_sub(dist_side)).as_secs_f64() / train_time.as_secs_f64());
    }
}
