//! Scaling benchmarks to analyze performance characteristics.
//!
//! This module provides comprehensive scaling benchmarks to identify bottlenecks:
//! - Data size scaling (how performance changes with more samples)
//! - Feature count scaling (how performance changes with more features)
//! - Parameter count scaling (how performance changes with distribution complexity)
//! - Memory efficiency analysis

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use gradientlss::distributions::{Beta, Distribution, Gamma, Gaussian, Poisson, StudentT};
use gradientlss::types::ResponseData;
use ndarray::{Array1, Array2};
use std::hint::black_box;
/// Generate synthetic data for benchmarking.
fn generate_data(n_samples: usize, n_params: usize) -> (Array2<f64>, Array1<f64>) {
    let predictions = Array2::from_shape_fn((n_samples, n_params), |(i, j)| {
        ((i * 7 + j * 13) % 100) as f64 / 100.0
    });
    let targets = Array1::from_shape_fn(n_samples, |i| 0.5 + (i % 10) as f64 / 10.0);
    (predictions, targets)
}

/// Generate positive targets for distributions that require them.
fn generate_positive_targets(n_samples: usize) -> Array1<f64> {
    Array1::from_shape_fn(n_samples, |i| 0.1 + (i % 10) as f64 / 5.0)
}

/// Generate training data.

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

// ============================================================================
// Data Size Scaling Benchmarks (Distribution Operations)
// ============================================================================

fn bench_data_scaling_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/data_size/gradients");
    group.sample_size(30);

    let dist = Gaussian::default();

    // Test across a wide range of data sizes
    for n_samples in [100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000] {
        let (predictions, targets) = generate_data(n_samples, 2);
        let target = ResponseData::Univariate(&targets.view());

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    dist.compute_gradients_and_hessians(
                        black_box(&predictions.view()),
                        black_box(&target),
                        None,
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_data_scaling_nll(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/data_size/nll");
    group.sample_size(30);

    let dist = Gaussian::default();

    for n_samples in [100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000] {
        let params = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                (i % 10) as f64 / 10.0
            } else {
                1.0 + (i % 5) as f64 / 10.0
            }
        });
        let targets = generate_data(n_samples, 1).1;
        let target = ResponseData::Univariate(&targets.view());

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| b.iter(|| dist.nll(black_box(&params.view()), black_box(&target))),
        );
    }
    group.finish();
}

fn bench_data_scaling_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/data_size/sampling");
    group.sample_size(30);

    let dist = Gaussian::default();
    let n_dist_samples = 100; // Fixed number of distribution samples

    for n_observations in [100, 500, 1_000, 2_500, 5_000, 10_000] {
        let params = Array2::from_shape_fn((n_observations, 2), |(i, j)| {
            if j == 0 {
                (i % 10) as f64 / 10.0
            } else {
                1.0 + (i % 5) as f64 / 10.0
            }
        });

        group.throughput(Throughput::Elements(
            (n_observations * n_dist_samples) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_observations),
            &n_observations,
            |b, _| {
                b.iter(|| {
                    dist.sample(
                        black_box(&params.view()),
                        black_box(n_dist_samples),
                        black_box(42),
                    )
                })
            },
        );
    }
    group.finish();
}

// ============================================================================
// Distribution Complexity Scaling
// ============================================================================

fn bench_distribution_complexity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/distribution_complexity");
    group.sample_size(30);

    let n_samples = 5_000;

    // Simple distributions (2 params)
    {
        let (predictions, targets) = generate_data(n_samples, 2);
        let dist = Gaussian::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("gaussian_2params", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Slightly more complex (2 params but more expensive)
    {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = generate_positive_targets(n_samples);
        let dist = Gamma::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("gamma_2params", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Medium complexity (3 params)
    {
        let (predictions, targets) = generate_data(n_samples, 3);
        let dist = StudentT::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("student_t_3params", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Beta (requires special target handling)
    {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = Array1::from_shape_fn(n_samples, |i| 0.1 + 0.8 * ((i % 10) as f64 / 10.0));
        let dist = Beta::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("beta_2params", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Poisson (discrete)
    {
        let (predictions, _) = generate_data(n_samples, 1);
        let targets = Array1::from_shape_fn(n_samples, |i| (i % 20) as f64);
        let dist = Poisson::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("poisson_1param", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    group.finish();
}

// ============================================================================
// Sample Count Scaling (for distribution sampling)
// ============================================================================

fn bench_sample_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/sample_count");
    group.sample_size(30);

    let n_observations = 1_000;
    let dist = Gaussian::default();
    let params = Array2::from_shape_fn((n_observations, 2), |(i, j)| {
        if j == 0 {
            (i % 10) as f64 / 10.0
        } else {
            1.0 + (i % 5) as f64 / 10.0
        }
    });

    for n_samples in [10, 50, 100, 500, 1_000, 2_500, 5_000, 10_000] {
        group.throughput(Throughput::Elements((n_observations * n_samples) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, &samples| {
                b.iter(|| dist.sample(black_box(&params.view()), black_box(samples), black_box(42)))
            },
        );
    }
    group.finish();
}

// ============================================================================
// Transform Parameter Scaling
// ============================================================================

fn bench_transform_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/transform_params");
    group.sample_size(30);

    let dist = Gaussian::default();

    for n_samples in [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000] {
        let predictions = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            ((i * 7 + j * 13) % 100) as f64 / 100.0 - 0.5
        });

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| b.iter(|| dist.transform_params(black_box(&predictions.view()))),
        );
    }
    group.finish();
}

// ============================================================================
// XGBoost Training Scaling Benchmarks
// ============================================================================

#[cfg(feature = "xgboost")]
mod xgboost_scaling {
    use super::*;
    use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
    use gradientlss::model::{GradientLSS, PredType};
    use gradientlss::prelude::XGBoostBackend;
    use std::sync::Arc;

    fn bench_xgboost_data_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("scaling/xgboost/data_size");
        group.sample_size(10);

        let n_features = 10;

        for n_samples in [250, 500, 1_000, 2_000, 5_000] {
            let (features, targets) = generate_training_data(n_samples, n_features);
            let dist = Arc::new(Gaussian::default());

            group.throughput(Throughput::Elements(n_samples as u64));
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

    fn bench_xgboost_feature_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("scaling/xgboost/feature_count");
        group.sample_size(10);

        let n_samples = 1_000;

        for n_features in [5, 10, 25, 50, 100] {
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

    fn bench_xgboost_rounds_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("scaling/xgboost/boosting_rounds");
        group.sample_size(10);

        let n_samples = 1_000;
        let n_features = 10;
        let (features, targets) = generate_training_data(n_samples, n_features);
        let dist = Arc::new(Gaussian::default());

        for num_rounds in [10, 25, 50, 100, 200, 500] {
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

    fn bench_xgboost_inference_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("scaling/xgboost/inference_batch");
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
            collect_train_metrics: false,
            seed: 42,
        };
        model.train(&mut train_data, None, params, config).unwrap();

        for batch_size in [10, 50, 100, 500, 1_000, 2_500, 5_000, 10_000] {
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
        xgboost_scaling_benches,
        bench_xgboost_data_scaling,
        bench_xgboost_feature_scaling,
        bench_xgboost_rounds_scaling,
        bench_xgboost_inference_scaling,
    );
}

// ============================================================================
// LightGBM Training Scaling Benchmarks
// ============================================================================

#[cfg(feature = "lightgbm")]
mod lightgbm_scaling {
    use super::*;
    use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
    use gradientlss::model::GradientLSS;
    use gradientlss::prelude::LightGBMBackend;
    use std::sync::Arc;

    fn bench_lightgbm_data_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("scaling/lightgbm/data_size");
        group.sample_size(10);

        let n_features = 10;

        for n_samples in [250, 500, 1_000, 2_000, 5_000] {
            let (features, targets) = generate_training_data(n_samples, n_features);
            let dist = Arc::new(Gaussian::default());

            group.throughput(Throughput::Elements(n_samples as u64));
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

    fn bench_lightgbm_feature_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("scaling/lightgbm/feature_count");
        group.sample_size(10);

        let n_samples = 1_000;

        for n_features in [5, 10, 25, 50, 100] {
            let (features, targets) = generate_training_data(n_samples, n_features);
            let dist = Arc::new(Gaussian::default());

            group.bench_with_input(
                BenchmarkId::from_parameter(n_features),
                &n_features,
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

    criterion_group!(
        lightgbm_scaling_benches,
        bench_lightgbm_data_scaling,
        bench_lightgbm_feature_scaling,
    );
}

// ============================================================================
// Fallback
// ============================================================================

#[cfg(not(any(feature = "xgboost", feature = "lightgbm")))]
mod no_backend {
    use super::*;

    fn bench_placeholder(c: &mut Criterion) {
        c.bench_function("scaling_no_backend", |b| b.iter(|| black_box(())));
    }

    criterion_group!(no_backend_benches, bench_placeholder);
}

// ============================================================================
// Main
// ============================================================================

criterion_group!(
    distribution_scaling_benches,
    bench_data_scaling_gradients,
    bench_data_scaling_nll,
    bench_data_scaling_sampling,
    bench_distribution_complexity_scaling,
    bench_sample_count_scaling,
    bench_transform_scaling,
);

#[cfg(all(feature = "xgboost", not(feature = "lightgbm")))]
criterion_main!(
    distribution_scaling_benches,
    xgboost_scaling::xgboost_scaling_benches
);

#[cfg(all(feature = "lightgbm", not(feature = "xgboost")))]
criterion_main!(
    distribution_scaling_benches,
    lightgbm_scaling::lightgbm_scaling_benches
);

#[cfg(all(feature = "xgboost", feature = "lightgbm"))]
criterion_main!(
    distribution_scaling_benches,
    xgboost_scaling::xgboost_scaling_benches,
    lightgbm_scaling::lightgbm_scaling_benches
);

#[cfg(not(any(feature = "xgboost", feature = "lightgbm")))]
criterion_main!(distribution_scaling_benches, no_backend::no_backend_benches);
