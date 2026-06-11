//! A/B timing for `TrainConfig::collect_train_metrics` (temporary harness).
//!
//! Trains the same NB + XGBoost model with a validation set, verbose off, and
//! no callbacks — the only configuration where the flag has any effect — and
//! alternates flag values across repetitions to cancel thermal drift. The
//! boosting trajectory is identical either way (the objective and the
//! validation metric are untouched), so the wall-time delta is exactly the
//! per-round train-metric pass.
//!
//! Run: cargo run --release --features xgboost --example bench_collect_train_metrics

use gradientlss::backend::xgboost_backend::XGBoostBackend;
use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
use gradientlss::distributions::NegativeBinomial;
use gradientlss::model::GradientLSS;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::time::Instant;

fn make_data(n: usize, n_features: usize, seed_shift: usize) -> (Array2<f64>, Array1<f64>) {
    let mut features = Array2::<f64>::zeros((n, n_features));
    let mut labels = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in 0..n_features {
            features[[i, j]] = (((i * 7 + j * 13 + seed_shift) % 101) as f64) / 101.0;
        }
        // NB-ish integer counts with a spread of magnitudes.
        let mu = 1.0 + 10.0 * features[[i, 0]] + 5.0 * features[[i, 1]];
        labels[i] = (mu + ((i * 31 + seed_shift) % 13) as f64 - 6.0).max(0.0).round();
    }
    (features, labels)
}

fn train_once(collect: bool, rounds: usize) -> f64 {
    let (train_x, train_y) = make_data(50_000, 20, 0);
    let (valid_x, valid_y) = make_data(10_000, 20, 3);

    let mut train_data =
        <XGBoostBackend as Backend>::Dataset::from_data(train_x.view(), train_y.view()).unwrap();
    let mut valid_data =
        <XGBoostBackend as Backend>::Dataset::from_data(valid_x.view(), valid_y.view()).unwrap();

    let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(NegativeBinomial::default()));
    let params = XGBoostBackend::create_params(model.n_params());
    let config = TrainConfig {
        num_boost_round: rounds,
        early_stopping_rounds: None, // identical round count both ways
        verbose: false,
        collect_train_metrics: collect,
        seed: 42,
    };

    let t = Instant::now();
    model
        .train(&mut train_data, Some(&mut valid_data), params, config)
        .unwrap();
    t.elapsed().as_secs_f64()
}

fn main() {
    let rounds = 150;
    // Warm-up (page cache, allocator, thermals).
    let _ = train_once(true, 20);

    let reps = 5;
    let mut with_metric = Vec::new();
    let mut without_metric = Vec::new();
    for r in 0..reps {
        // Alternate order each rep to cancel drift.
        if r % 2 == 0 {
            with_metric.push(train_once(true, rounds));
            without_metric.push(train_once(false, rounds));
        } else {
            without_metric.push(train_once(false, rounds));
            with_metric.push(train_once(true, rounds));
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let t_true = med(&mut with_metric);
    let t_false = med(&mut without_metric);
    println!("collect_train_metrics=true:  median {:.3}s  {:?}", t_true, with_metric);
    println!("collect_train_metrics=false: median {:.3}s  {:?}", t_false, without_metric);
    println!(
        "delta: {:.3}s ({:.1}% faster without the train metric)",
        t_true - t_false,
        100.0 * (t_true - t_false) / t_true
    );
}
