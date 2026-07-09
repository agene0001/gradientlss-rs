//! Benchmarks for model training operations.
//!
//! This module benchmarks the full model training pipeline including:
//! - XGBoost backend training (when feature enabled)
//! - LightGBM backend training (when feature enabled)
//! - Cross-validation performance
//!
//! Note: These benchmarks require the respective backend features to be enabled.

use criterion::{Criterion,criterion_group, criterion_main};
use ndarray::{Array1, Array2};
use std::hint::black_box;
/// Generate synthetic regression data for training benchmarks.

fn generate_training_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let features = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let x = ((i * 17 + j * 31) % 1000) as f64 / 1000.0;
        x * 2.0 - 1.0
    });

    let targets = Array1::from_shape_fn(n_samples, |i| {
        let sum: f64 = (0..n_features.min(3)).map(|j| features[[i, j]]).sum();
        sum / 3.0 + 0.5 + ((i * 13) % 100) as f64 / 500.0
    });

    (features, targets)
}

/// Generate positive targets for Gamma distribution.

fn generate_positive_training_data(
    n_samples: usize,
    n_features: usize,
) -> (Array2<f64>, Array1<f64>) {
    let features = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let x = ((i * 17 + j * 31) % 1000) as f64 / 1000.0;
        x * 2.0 - 1.0
    });

    let targets = Array1::from_shape_fn(n_samples, |i| {
        let sum: f64 = (0..n_features.min(3)).map(|j| features[[i, j]].abs()).sum();
        (sum / 3.0 + 0.5).max(0.1)
    });

    (features, targets)
}

// ============================================================================
// XGBoost Backend Benchmarks
// ============================================================================

#[cfg(feature = "xgboost")]
mod xgboost_benchmarks {
    use super::*;
    use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
    use gradientlss::distributions::Gaussian;
    use gradientlss::model::GradientLSS;
    use gradientlss::prelude::XGBoostBackend;
    use std::sync::Arc;
    use criterion::BenchmarkId;

    fn bench_xgboost_training_gaussian(c: &mut Criterion) {
        let mut group = c.benchmark_group("training/xgboost/gaussian");
        group.sample_size(10);

        for n_samples in [500, 1_000, 2_000] {
            let (features, targets) = generate_training_data(n_samples, 10);
            let dist = Arc::new(Gaussian::default());

            group.bench_with_input(
                BenchmarkId::from_parameter(n_samples),
                &n_samples,
                |b, _| {
                    b.iter(|| {
                        let mut model = GradientLSS::<XGBoostBackend>::new(dist.clone());
                        let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
                            features.view(),
                            targets.view(),
                        )
                        .unwrap();
                        let params = XGBoostBackend::create_params(model.n_params());
                        let config = TrainConfig {
                            num_boost_round: 50,
                            early_stopping_rounds: None,
                            verbose: false,
                            collect_train_metrics: false,
                            seed: 42,
                        };
                        model.train(
                            black_box(&mut train_data),
                            None,
                            black_box(params),
                            black_box(config),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    fn bench_xgboost_training_gamma(c: &mut Criterion) {
        let mut group = c.benchmark_group("training/xgboost/gamma");
        group.sample_size(10);

        for n_samples in [500, 1_000, 2_000] {
            let (features, targets) = generate_positive_training_data(n_samples, 10);
            let dist = Arc::new(gradientlss::distributions::Gamma::default());

            group.bench_with_input(
                BenchmarkId::from_parameter(n_samples),
                &n_samples,
                |b, _| {
                    b.iter(|| {
                        let mut model = GradientLSS::<XGBoostBackend>::new(dist.clone());
                        let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
                            features.view(),
                            targets.view(),
                        )
                        .unwrap();
                        let params = XGBoostBackend::create_params(model.n_params());
                        let config = TrainConfig {
                            num_boost_round: 50,
                            early_stopping_rounds: None,
                            verbose: false,
                            collect_train_metrics: false,
                            seed: 42,
                        };
                        model.train(
                            black_box(&mut train_data),
                            None,
                            black_box(params),
                            black_box(config),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    fn bench_xgboost_rounds_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("training/xgboost/rounds_scaling");
        group.sample_size(10);

        let n_samples = 1_000;
        let (features, targets) = generate_training_data(n_samples, 10);
        let dist = Arc::new(Gaussian::default());

        for num_rounds in [25, 50, 100, 200] {
            group.bench_with_input(
                BenchmarkId::from_parameter(num_rounds),
                &num_rounds,
                |b, &rounds| {
                    b.iter(|| {
                        let mut model = GradientLSS::<XGBoostBackend>::new(dist.clone());
                        let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
                            features.view(),
                            targets.view(),
                        )
                        .unwrap();
                        let params = XGBoostBackend::create_params(model.n_params());
                        let config = TrainConfig {
                            num_boost_round: rounds,
                            early_stopping_rounds: None,
                            verbose: false,
                            collect_train_metrics: false,
                            seed: 42,
                        };
                        model.train(
                            black_box(&mut train_data),
                            None,
                            black_box(params),
                            black_box(config),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    fn bench_xgboost_feature_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("training/xgboost/feature_scaling");
        group.sample_size(10);

        let n_samples = 1_000;

        for n_features in [5, 10, 25, 50] {
            let (features, targets) = generate_training_data(n_samples, n_features);
            let dist = Arc::new(Gaussian::default());

            group.bench_with_input(
                BenchmarkId::from_parameter(n_features),
                &n_features,
                |b, _| {
                    b.iter(|| {
                        let mut model = GradientLSS::<XGBoostBackend>::new(dist.clone());
                        let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
                            features.view(),
                            targets.view(),
                        )
                        .unwrap();
                        let params = XGBoostBackend::create_params(model.n_params());
                        let config = TrainConfig {
                            num_boost_round: 50,
                            early_stopping_rounds: None,
                            verbose: false,
                            collect_train_metrics: false,
                            seed: 42,
                        };
                        model.train(
                            black_box(&mut train_data),
                            None,
                            black_box(params),
                            black_box(config),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    criterion_group!(
        xgboost_benches,
        bench_xgboost_training_gaussian,
        bench_xgboost_training_gamma,
        bench_xgboost_rounds_scaling,
        bench_xgboost_feature_scaling,
    );
}

// ============================================================================
// LightGBM Backend Benchmarks
// ============================================================================

#[cfg(feature = "lightgbm")]
mod lightgbm_benchmarks {
    use super::*;
    use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
    use gradientlss::distributions::Gaussian;
    use gradientlss::model::GradientLSS;
    use gradientlss::prelude::LightGBMBackend;
    use std::sync::Arc;
    use criterion::BenchmarkId;

    fn bench_lightgbm_training_gaussian(c: &mut Criterion) {
        let mut group = c.benchmark_group("training/lightgbm/gaussian");
        group.sample_size(10);

        for n_samples in [500, 1_000, 2_000] {
            let (features, targets) = generate_training_data(n_samples, 10);
            let dist = Arc::new(Gaussian::default());

            group.bench_with_input(
                BenchmarkId::from_parameter(n_samples),
                &n_samples,
                |b, _| {
                    b.iter(|| {
                        let mut model = GradientLSS::<LightGBMBackend>::new(dist.clone());
                        let mut train_data = <LightGBMBackend as Backend>::Dataset::from_data(
                            features.view(),
                            targets.view(),
                        )
                        .unwrap();
                        let params = LightGBMBackend::create_params(model.n_params());
                        let config = TrainConfig {
                            num_boost_round: 50,
                            early_stopping_rounds: None,
                            verbose: false,
                            collect_train_metrics: false,
                            seed: 42,
                        };
                        model.train(
                            black_box(&mut train_data),
                            None,
                            black_box(params),
                            black_box(config),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    fn bench_lightgbm_training_gamma(c: &mut Criterion) {
        let mut group = c.benchmark_group("training/lightgbm/gamma");
        group.sample_size(10);

        for n_samples in [500, 1_000, 2_000] {
            let (features, targets) = generate_positive_training_data(n_samples, 10);
            let dist = Arc::new(gradientlss::distributions::Gamma::default());

            group.bench_with_input(
                BenchmarkId::from_parameter(n_samples),
                &n_samples,
                |b, _| {
                    b.iter(|| {
                        let mut model = GradientLSS::<LightGBMBackend>::new(dist.clone());
                        let mut train_data = <LightGBMBackend as Backend>::Dataset::from_data(
                            features.view(),
                            targets.view(),
                        )
                        .unwrap();
                        let params = LightGBMBackend::create_params(model.n_params());
                        let config = TrainConfig {
                            num_boost_round: 50,
                            early_stopping_rounds: None,
                            verbose: false,
                            collect_train_metrics: false,
                            seed: 42,
                        };
                        model.train(
                            black_box(&mut train_data),
                            None,
                            black_box(params),
                            black_box(config),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    fn bench_lightgbm_rounds_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("training/lightgbm/rounds_scaling");
        group.sample_size(10);

        let n_samples = 1_000;
        let (features, targets) = generate_training_data(n_samples, 10);
        let dist = Arc::new(Gaussian::default());

        for num_rounds in [25, 50, 100, 200] {
            group.bench_with_input(
                BenchmarkId::from_parameter(num_rounds),
                &num_rounds,
                |b, &rounds| {
                    b.iter(|| {
                        let mut model = GradientLSS::<LightGBMBackend>::new(dist.clone());
                        let mut train_data = <LightGBMBackend as Backend>::Dataset::from_data(
                            features.view(),
                            targets.view(),
                        )
                        .unwrap();
                        let params = LightGBMBackend::create_params(model.n_params());
                        let config = TrainConfig {
                            num_boost_round: rounds,
                            early_stopping_rounds: None,
                            verbose: false,
                            collect_train_metrics: false,
                            seed: 42,
                        };
                        model.train(
                            black_box(&mut train_data),
                            None,
                            black_box(params),
                            black_box(config),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    criterion_group!(
        lightgbm_benches,
        bench_lightgbm_training_gaussian,
        bench_lightgbm_training_gamma,
        bench_lightgbm_rounds_scaling,
    );
}

// ============================================================================
// Backend Comparison (when both are available)
// ============================================================================

#[cfg(all(feature = "xgboost", feature = "lightgbm"))]
mod backend_comparison {
    use super::*;
    use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
    use gradientlss::distributions::Gaussian;
    use gradientlss::model::GradientLSS;
    use gradientlss::prelude::{LightGBMBackend, XGBoostBackend};
    use std::sync::Arc;

    fn bench_backend_comparison(c: &mut Criterion) {
        let mut group = c.benchmark_group("training/backend_comparison");
        group.sample_size(10);

        let n_samples = 1_000;
        let (features, targets) = generate_training_data(n_samples, 10);

        // XGBoost
        {
            let dist = Arc::new(Gaussian::default());
            group.bench_function("xgboost_gaussian", |b| {
                b.iter(|| {
                    let mut model = GradientLSS::<XGBoostBackend>::new(dist.clone());
                    let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
                        features.view(),
                        targets.view(),
                    )
                    .unwrap();
                    let params = XGBoostBackend::create_params(model.n_params());
                    let config = TrainConfig {
                        num_boost_round: 50,
                        early_stopping_rounds: None,
                        verbose: false,
                        collect_train_metrics: false,
                        seed: 42,
                    };
                    model.train(
                        black_box(&mut train_data),
                        None,
                        black_box(params),
                        black_box(config),
                    )
                })
            });
        }

        // LightGBM
        {
            let dist = Arc::new(Gaussian::default());
            group.bench_function("lightgbm_gaussian", |b| {
                b.iter(|| {
                    let mut model = GradientLSS::<LightGBMBackend>::new(dist.clone());
                    let mut train_data = <LightGBMBackend as Backend>::Dataset::from_data(
                        features.view(),
                        targets.view(),
                    )
                    .unwrap();
                    let params = LightGBMBackend::create_params(model.n_params());
                    let config = TrainConfig {
                        num_boost_round: 50,
                        early_stopping_rounds: None,
                        verbose: false,
                        collect_train_metrics: false,
                        seed: 42,
                    };
                    model.train(
                        black_box(&mut train_data),
                        None,
                        black_box(params),
                        black_box(config),
                    )
                })
            });
        }

        group.finish();
    }

    criterion_group!(comparison_benches, bench_backend_comparison);
}

// ============================================================================
// Fallback for when no backends are enabled
// ============================================================================

#[cfg(not(any(feature = "xgboost", feature = "lightgbm")))]
mod no_backend {
    use super::*;

    fn bench_no_backend_warning(c: &mut Criterion) {
        c.bench_function("no_backend_enabled", |b| {
            b.iter(|| {
                // This benchmark exists to indicate that backend features are needed
                // Enable with: cargo bench --features xgboost
                // Or: cargo bench --features lightgbm
                // Or: cargo bench --features full
                black_box(())
            })
        });
    }

    criterion_group!(no_backend_benches, bench_no_backend_warning);
}

// ============================================================================
// Main
// ============================================================================

#[cfg(all(feature = "xgboost", not(feature = "lightgbm")))]
criterion_main!(xgboost_benchmarks::xgboost_benches);

#[cfg(all(feature = "lightgbm", not(feature = "xgboost")))]
criterion_main!(lightgbm_benchmarks::lightgbm_benches);

#[cfg(all(feature = "xgboost", feature = "lightgbm"))]
criterion_main!(
    xgboost_benchmarks::xgboost_benches,
    lightgbm_benchmarks::lightgbm_benches,
    backend_comparison::comparison_benches
);

#[cfg(not(any(feature = "xgboost", feature = "lightgbm")))]
criterion_main!(no_backend::no_backend_benches);
