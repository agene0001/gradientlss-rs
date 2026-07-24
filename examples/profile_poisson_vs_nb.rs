//! A/B harness: Poisson vs NegativeBinomial training time on identical data,
//! XGBoost backend. Answers "why is my NB fit Nx slower than my Poisson fit?"
//! by separating the structural cost (num_target=2 → 2 trees/round) from the
//! distribution-side cost (gradients + per-round NLL metric).
//!
//! Run:  cargo run --release --features xgboost --example profile_poisson_vs_nb
//! Env:  PROFILE_N (rows, default 200_000), PROFILE_ROUNDS (default 100),
//!       PROFILE_LARGE_K=1 shifts counts into the hundreds (exercises the
//!       digamma fallback in nb_psi_diff and the ln_factorial table miss),
//!       PROFILE_BACKEND=lightgbm (needs --features full),
//!       PROFILE_TREE_METHOD / PROFILE_MULTI_STRATEGY override XGBoost params
//!       (measured: multi_output_tree is ~20% SLOWER than the default
//!       one-tree-per-param here — XGBoost's multi-output trees don't pay off).

use gradientlss::backend::{BackendDataset, BackendParams};
use gradientlss::distributions::{Distribution, NegativeBinomial, Poisson};
use gradientlss::prelude::*;
use gradientlss::types::ResponseData;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let n: usize = std::env::var("PROFILE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let n_features: usize = 20;
    let rounds: usize = std::env::var("PROFILE_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let large_k = std::env::var("PROFILE_LARGE_K").as_deref() == Ok("1");
    let scale = if large_k { 40.0 } else { 4.0 };

    println!("n={n}, features={n_features}, rounds={rounds}, large_k={large_k}");

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
        let mean = (0.5 * lin).exp();
        let u = next();
        labels[i] = (mean * (-(1.0 - u).ln()) * scale).floor();
    }
    let kmax = labels.iter().cloned().fold(0.0f64, f64::max);
    let kmean = labels.sum() / n as f64;
    println!("counts: mean={kmean:.1}, max={kmax:.0}");

    let target = ResponseData::Univariate(&labels.view());

    // Per-call distribution-side costs on full-size arrays.
    let time_dist = |dist: &dyn Distribution, n_params: usize| {
        let preds = Array2::<f64>::from_shape_fn((n, n_params), |(i, j)| {
            if j == 0 {
                1.5 + (i % 7) as f64 * 0.1
            } else {
                0.3
            }
        });
        let t = Instant::now();
        let mut iters = 0u32;
        while t.elapsed().as_millis() < 1000 {
            let gh = dist
                .compute_gradients_and_hessians(&preds.view(), &target, None)
                .unwrap();
            std::hint::black_box(&gh);
            iters += 1;
        }
        let grad = t.elapsed() / iters;

        let transformed = dist.transform_params(&preds.view());
        let t = Instant::now();
        let mut iters = 0u32;
        while t.elapsed().as_millis() < 1000 {
            std::hint::black_box(dist.nll(&transformed.view(), &target));
            iters += 1;
        }
        let nll = t.elapsed() / iters;
        (grad, nll)
    };

    let run = |name: &str, dist: Arc<dyn Distribution>| -> std::time::Duration {
        let n_params = dist.n_params();
        let (grad, nll) = time_dist(dist.as_ref(), n_params);
        println!("\n[{name}] grad/call: {grad:>9.3?}   nll/call: {nll:>9.3?}");

        let config = TrainConfig {
            num_boost_round: rounds,
            early_stopping_rounds: None,
            verbose: false,
            collect_train_metrics: false,
            seed: 123,
        };

        let use_lgbm = std::env::var("PROFILE_BACKEND").as_deref() == Ok("lightgbm");
        let train_time = if use_lgbm {
            #[cfg(feature = "lightgbm")]
            {
                let mut model = GradientLSS::<LightGBMBackend>::new(dist);
                let mut train_data =
                    <LightGBMBackend as Backend>::Dataset::from_data(feats.view(), labels.view())
                        .unwrap();
                let params = LightGBMBackend::create_params(n_params);
                let t = Instant::now();
                model.train(&mut train_data, None, params, config).unwrap();
                t.elapsed()
            }
            #[cfg(not(feature = "lightgbm"))]
            panic!("rebuild with --features lightgbm for PROFILE_BACKEND=lightgbm")
        } else {
            let mut model = GradientLSS::<XGBoostBackend>::new(dist);
            let mut train_data =
                <XGBoostBackend as Backend>::Dataset::from_data(feats.view(), labels.view())
                    .unwrap();
            let mut params = XGBoostBackend::create_params(n_params);
            if let Ok(tm) = std::env::var("PROFILE_TREE_METHOD") {
                params.set(
                    "tree_method",
                    gradientlss::backend::ParamValue::from(tm.as_str()),
                );
            }
            if let Ok(ms) = std::env::var("PROFILE_MULTI_STRATEGY") {
                params.set(
                    "multi_strategy",
                    gradientlss::backend::ParamValue::from(ms.as_str()),
                );
            }
            let t = Instant::now();
            model.train(&mut train_data, None, params, config).unwrap();
            t.elapsed()
        };

        let dist_side = (grad + nll) * rounds as u32;
        println!(
            "[{name}] train {rounds} rounds: {train_time:>9.3?}   (dist-side ≈ {dist_side:.3?}, {:.1}%)",
            100.0 * dist_side.as_secs_f64() / train_time.as_secs_f64()
        );
        train_time
    };

    let t_pois = run("poisson ", Arc::new(Poisson::default()));
    let t_nb = run("negbinom", Arc::new(NegativeBinomial::default()));
    println!(
        "\nNB / Poisson ratio: {:.2}x",
        t_nb.as_secs_f64() / t_pois.as_secs_f64()
    );
}
