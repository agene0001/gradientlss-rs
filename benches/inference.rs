//! Benchmarks for inference and prediction operations.
//!
//! This module benchmarks the prediction pipeline including:
//! - Raw predictions from trained models
//! - Parameter transformation
//! - Sampling from predicted distributions
//! - Quantile computation
//!
//! Note: These benchmarks require backend features to be enabled.

use criterion::{ Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2};
use std::hint::black_box;
/// Generate synthetic regression data.
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

/// Generate positive targets.
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
// XGBoost Inference Benchmarks
// ============================================================================

#[cfg(feature = "xgboost")]
mod xgboost_inference {
    use super::*;
    use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
    use gradientlss::distributions::{Gamma, Gaussian, StudentT};
    use gradientlss::model::{GradientLSS, PredType};
    use gradientlss::prelude::XGBoostBackend;
    use std::sync::Arc;
    use criterion::{BenchmarkId, Throughput};

    fn bench_prediction_parameters(c: &mut Criterion) {
        let mut group = c.benchmark_group("inference/xgboost/parameters");
        group.sample_size(20);

        // Train a model once
        let n_train = 1_000;
        let n_features = 10;
        let (train_features, train_targets) = generate_training_data(n_train, n_features);
        let dist = Arc::new(Gaussian::default());

        let mut model = GradientLSS::<XGBoostBackend>::new(dist);
        let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
            train_features.view(),
            train_targets.view(),
        )
        .unwrap();
        let params = XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 50,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };
        model.train(&mut train_data, None, params, config).unwrap();

        for n_test in [100, 500, 1_000, 5_000] {
            let (test_features, _) = generate_training_data(n_test, n_features);

            group.throughput(Throughput::Elements(n_test as u64));
            group.bench_with_input(BenchmarkId::from_parameter(n_test), &n_test, |b, _| {
                b.iter(|| {
                    model.predict(
                        black_box(&test_features.view()),
                        black_box(PredType::Parameters),
                        100,
                        &[],
                        42,
                    )
                })
            });
        }
        group.finish();
    }

    fn bench_prediction_samples(c: &mut Criterion) {
        let mut group = c.benchmark_group("inference/xgboost/samples");
        group.sample_size(20);

        // Train a model once
        let n_train = 1_000;
        let n_features = 10;
        let (train_features, train_targets) = generate_training_data(n_train, n_features);
        let dist = Arc::new(Gaussian::default());

        let mut model = GradientLSS::<XGBoostBackend>::new(dist);
        let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
            train_features.view(),
            train_targets.view(),
        )
        .unwrap();
        let params = XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 50,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };
        model.train(&mut train_data, None, params, config).unwrap();

        let n_test = 500;
        let (test_features, _) = generate_training_data(n_test, n_features);

        for n_samples in [100, 500, 1_000, 5_000] {
            group.throughput(Throughput::Elements((n_test * n_samples) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(n_samples),
                &n_samples,
                |b, &samples| {
                    b.iter(|| {
                        model.predict(
                            black_box(&test_features.view()),
                            black_box(PredType::Samples),
                            black_box(samples),
                            &[],
                            42,
                        )
                    })
                },
            );
        }
        group.finish();
    }

    fn bench_prediction_quantiles(c: &mut Criterion) {
        let mut group = c.benchmark_group("inference/xgboost/quantiles");
        group.sample_size(20);

        // Train a model once
        let n_train = 1_000;
        let n_features = 10;
        let (train_features, train_targets) = generate_training_data(n_train, n_features);
        let dist = Arc::new(Gaussian::default());

        let mut model = GradientLSS::<XGBoostBackend>::new(dist);
        let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
            train_features.view(),
            train_targets.view(),
        )
        .unwrap();
        let params = XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 50,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };
        model.train(&mut train_data, None, params, config).unwrap();

        let quantiles = [0.1, 0.25, 0.5, 0.75, 0.9];

        for n_test in [100, 500, 1_000, 5_000] {
            let (test_features, _) = generate_training_data(n_test, n_features);

            group.throughput(Throughput::Elements(n_test as u64));
            group.bench_with_input(BenchmarkId::from_parameter(n_test), &n_test, |b, _| {
                b.iter(|| {
                    model.predict(
                        black_box(&test_features.view()),
                        black_box(PredType::Quantiles),
                        1000,
                        black_box(&quantiles),
                        42,
                    )
                })
            });
        }
        group.finish();
    }

    fn bench_prediction_distribution_comparison(c: &mut Criterion) {
        let mut group = c.benchmark_group("inference/xgboost/distribution_comparison");
        group.sample_size(20);

        let n_train = 1_000;
        let n_features = 10;
        let n_test = 1_000;

        // Gaussian
        {
            let (train_features, train_targets) = generate_training_data(n_train, n_features);
            let (test_features, _) = generate_training_data(n_test, n_features);
            let dist = Arc::new(Gaussian::default());

            let mut model = GradientLSS::<XGBoostBackend>::new(dist);
            let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
                train_features.view(),
                train_targets.view(),
            )
            .unwrap();
            let params = XGBoostBackend::create_params(model.n_params());
            let config = TrainConfig {
                num_boost_round: 50,
                early_stopping_rounds: None,
                verbose: false,
                seed: 42,
            };
            model.train(&mut train_data, None, params, config).unwrap();

            group.bench_function("gaussian_params", |b| {
                b.iter(|| {
                    model.predict(
                        black_box(&test_features.view()),
                        black_box(PredType::Parameters),
                        100,
                        &[],
                        42,
                    )
                })
            });

            group.bench_function("gaussian_samples_1000", |b| {
                b.iter(|| {
                    model.predict(
                        black_box(&test_features.view()),
                        black_box(PredType::Samples),
                        1000,
                        &[],
                        42,
                    )
                })
            });
        }

        // Gamma
        {
            let (train_features, train_targets) =
                generate_positive_training_data(n_train, n_features);
            let (test_features, _) = generate_positive_training_data(n_test, n_features);
            let dist = Arc::new(Gamma::default());

            let mut model = GradientLSS::<XGBoostBackend>::new(dist);
            let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
                train_features.view(),
                train_targets.view(),
            )
            .unwrap();
            let params = XGBoostBackend::create_params(model.n_params());
            let config = TrainConfig {
                num_boost_round: 50,
                early_stopping_rounds: None,
                verbose: false,
                seed: 42,
            };
            model.train(&mut train_data, None, params, config).unwrap();

            group.bench_function("gamma_params", |b| {
                b.iter(|| {
                    model.predict(
                        black_box(&test_features.view()),
                        black_box(PredType::Parameters),
                        100,
                        &[],
                        42,
                    )
                })
            });

            group.bench_function("gamma_samples_1000", |b| {
                b.iter(|| {
                    model.predict(
                        black_box(&test_features.view()),
                        black_box(PredType::Samples),
                        1000,
                        &[],
                        42,
                    )
                })
            });
        }

        // StudentT
        {
            let (train_features, train_targets) = generate_training_data(n_train, n_features);
            let (test_features, _) = generate_training_data(n_test, n_features);
            let dist = Arc::new(StudentT::default());

            let mut model = GradientLSS::<XGBoostBackend>::new(dist);
            let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
                train_features.view(),
                train_targets.view(),
            )
            .unwrap();
            let params = XGBoostBackend::create_params(model.n_params());
            let config = TrainConfig {
                num_boost_round: 50,
                early_stopping_rounds: None,
                verbose: false,
                seed: 42,
            };
            model.train(&mut train_data, None, params, config).unwrap();

            group.bench_function("student_t_params", |b| {
                b.iter(|| {
                    model.predict(
                        black_box(&test_features.view()),
                        black_box(PredType::Parameters),
                        100,
                        &[],
                        42,
                    )
                })
            });

            group.bench_function("student_t_samples_1000", |b| {
                b.iter(|| {
                    model.predict(
                        black_box(&test_features.view()),
                        black_box(PredType::Samples),
                        1000,
                        &[],
                        42,
                    )
                })
            });
        }

        group.finish();
    }

    fn bench_batch_size_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("inference/xgboost/batch_size_scaling");
        group.sample_size(20);

        // Train a model once
        let n_train = 2_000;
        let n_features = 10;
        let (train_features, train_targets) = generate_training_data(n_train, n_features);
        let dist = Arc::new(Gaussian::default());

        let mut model = GradientLSS::<XGBoostBackend>::new(dist);
        let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
            train_features.view(),
            train_targets.view(),
        )
        .unwrap();
        let params = XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 100,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };
        model.train(&mut train_data, None, params, config).unwrap();

        for batch_size in [10, 50, 100, 500, 1_000, 2_000, 5_000] {
            let (test_features, _) = generate_training_data(batch_size, n_features);

            group.throughput(Throughput::Elements(batch_size as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        model.predict(
                            black_box(&test_features.view()),
                            black_box(PredType::Parameters),
                            100,
                            &[],
                            42,
                        )
                    })
                },
            );
        }
        group.finish();
    }

    criterion_group!(
        xgboost_inference_benches,
        bench_prediction_parameters,
        bench_prediction_samples,
        bench_prediction_quantiles,
        bench_prediction_distribution_comparison,
        bench_batch_size_scaling,
    );
}

// ============================================================================
// LightGBM Inference Benchmarks
// ============================================================================

#[cfg(feature = "lightgbm")]
mod lightgbm_inference {
    use super::*;
    use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
    use gradientlss::distributions::Gaussian;
    use gradientlss::model::{GradientLSS, PredType};
    use gradientlss::prelude::LightGBMBackend;
    use std::sync::Arc;
    use criterion::{BenchmarkId, Throughput};

    fn bench_lightgbm_prediction_parameters(c: &mut Criterion) {
        let mut group = c.benchmark_group("inference/lightgbm/parameters");
        group.sample_size(20);

        // Train a model once
        let n_train = 1_000;
        let n_features = 10;
        let (train_features, train_targets) = generate_training_data(n_train, n_features);
        let dist = Arc::new(Gaussian::default());

        let mut model = GradientLSS::<LightGBMBackend>::new(dist);
        let mut train_data = <LightGBMBackend as Backend>::Dataset::from_data(
            train_features.view(),
            train_targets.view(),
        )
        .unwrap();
        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 50,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };
        model.train(&mut train_data, None, params, config).unwrap();

        for n_test in [100, 500, 1_000, 5_000] {
            let (test_features, _) = generate_training_data(n_test, n_features);

            group.throughput(Throughput::Elements(n_test as u64));
            group.bench_with_input(BenchmarkId::from_parameter(n_test), &n_test, |b, _| {
                b.iter(|| {
                    model.predict(
                        black_box(&test_features.view()),
                        black_box(PredType::Parameters),
                        100,
                        &[],
                        42,
                    )
                })
            });
        }
        group.finish();
    }

    fn bench_lightgbm_prediction_samples(c: &mut Criterion) {
        let mut group = c.benchmark_group("inference/lightgbm/samples");
        group.sample_size(20);

        // Train a model once
        let n_train = 1_000;
        let n_features = 10;
        let (train_features, train_targets) = generate_training_data(n_train, n_features);
        let dist = Arc::new(Gaussian::default());

        let mut model = GradientLSS::<LightGBMBackend>::new(dist);
        let mut train_data = <LightGBMBackend as Backend>::Dataset::from_data(
            train_features.view(),
            train_targets.view(),
        )
        .unwrap();
        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 50,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };
        model.train(&mut train_data, None, params, config).unwrap();

        let n_test = 500;
        let (test_features, _) = generate_training_data(n_test, n_features);

        for n_samples in [100, 500, 1_000, 5_000] {
            group.throughput(Throughput::Elements((n_test * n_samples) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(n_samples),
                &n_samples,
                |b, &samples| {
                    b.iter(|| {
                        model.predict(
                            black_box(&test_features.view()),
                            black_box(PredType::Samples),
                            black_box(samples),
                            &[],
                            42,
                        )
                    })
                },
            );
        }
        group.finish();
    }

    criterion_group!(
        lightgbm_inference_benches,
        bench_lightgbm_prediction_parameters,
        bench_lightgbm_prediction_samples,
    );
}

// ============================================================================
// Fallback for when no backends are enabled
// ============================================================================

#[cfg(not(any(feature = "xgboost", feature = "lightgbm")))]
mod no_backend {
    use super::*;

    fn bench_no_backend_warning(c: &mut Criterion) {
        c.bench_function("inference_no_backend_enabled", |b| {
            b.iter(|| {
                // Enable with: cargo bench --features xgboost
                // Or: cargo bench --features lightgbm
                black_box(())
            })
        });
    }

    criterion_group!(no_backend_benches, bench_no_backend_warning);
}

// ============================================================================
// Main
// ============================================================================

#[cfg(feature = "xgboost")]
criterion_main!(xgboost_inference::xgboost_inference_benches);

#[cfg(all(feature = "lightgbm", not(feature = "xgboost")))]
criterion_main!(lightgbm_inference::lightgbm_inference_benches);

#[cfg(all(feature = "xgboost", feature = "lightgbm"))]
criterion_main!(
    xgboost_inference::xgboost_inference_benches,
    lightgbm_inference::lightgbm_inference_benches
);

#[cfg(not(any(feature = "xgboost", feature = "lightgbm")))]
criterion_main!(no_backend::no_backend_benches);
