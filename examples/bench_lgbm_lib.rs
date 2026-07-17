//! A/B harness for comparing lib_lightgbm builds (Homebrew vs source-built
//! native). Times ONLY the training loop; Gaussian analytical gradients keep
//! the Rust-side cost minimal so tree building dominates. Build once per
//! linked library (LIGHTGBM_LIB_DIR env at build time selects it), copy the
//! binary aside, and run the variants alternately.
//!
//! Not registered anywhere; run with:
//!   cargo build --release --features lightgbm --example bench_lgbm_lib

use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
use gradientlss::distributions::Gaussian;
use gradientlss::model::GradientLSS;
use gradientlss::prelude::LightGBMBackend;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let n = 50_000usize;
    let f = 20usize;
    let rounds = 100usize;
    let reps = 5usize;

    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / (u32::MAX as f64)
    };

    let mut features = Array2::<f64>::zeros((n, f));
    let mut labels = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut signal = 0.0;
        for j in 0..f {
            let x = next();
            features[[i, j]] = x;
            if j < 4 {
                signal += x;
            }
        }
        labels[i] = signal + (next() - 0.5) * 2.0;
    }

    let dist = Arc::new(Gaussian::default());
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut model = GradientLSS::<LightGBMBackend>::new(dist.clone());
        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();
        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: rounds,
            early_stopping_rounds: None,
            verbose: false,
            collect_train_metrics: false,
            seed: 42,
        };
        let t = Instant::now();
        model.train(&mut train_data, None, params, config).unwrap();
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "median {:.3}s  all {:?}",
        times[times.len() / 2],
        times
            .iter()
            .map(|t| (t * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
}
