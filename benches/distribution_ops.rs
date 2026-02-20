//! Benchmarks for distribution operations.
//!
//! This module benchmarks the core distribution operations that are critical
//! for training performance:
//! - Gradient and Hessian computation
//! - Negative log-likelihood (NLL) calculation
//! - Sampling from distributions
//! - Parameter transformation

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gradientlss::distributions::{
    Beta, Cauchy, Distribution, Gamma, Gaussian, Gumbel, Laplace, LogNormal, Logistic,
    NegativeBinomial, Poisson, Stabilization, StudentT, Weibull,
};
use gradientlss::types::ResponseData;
use gradientlss::utils::ResponseFn;
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

/// Generate count targets for discrete distributions.
fn generate_count_targets(n_samples: usize) -> Array1<f64> {
    Array1::from_shape_fn(n_samples, |i| (i % 20) as f64)
}

/// Generate targets in (0, 1) for Beta distribution.
fn generate_unit_targets(n_samples: usize) -> Array1<f64> {
    Array1::from_shape_fn(n_samples, |i| 0.1 + 0.8 * ((i % 10) as f64 / 10.0))
}

// ============================================================================
// Gradient and Hessian Benchmarks
// ============================================================================

fn bench_gradients_gaussian(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradients/gaussian");

    for n_samples in [100, 1_000, 10_000] {
        let (predictions, targets) = generate_data(n_samples, 2);
        let dist = Gaussian::default();
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

fn bench_gradients_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradients/gamma");

    for n_samples in [100, 1_000, 10_000] {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = generate_positive_targets(n_samples);
        let dist = Gamma::default();
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

fn bench_gradients_student_t(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradients/student_t");

    for n_samples in [100, 1_000, 10_000] {
        let (predictions, _) = generate_data(n_samples, 3);
        let targets = generate_data(n_samples, 1).1;
        let dist = StudentT::default();
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

fn bench_gradients_beta(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradients/beta");

    for n_samples in [100, 1_000, 10_000] {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = generate_unit_targets(n_samples);
        let dist = Beta::default();
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

fn bench_gradients_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradients/poisson");

    for n_samples in [100, 1_000, 10_000] {
        let (predictions, _) = generate_data(n_samples, 1);
        let targets = generate_count_targets(n_samples);
        let dist = Poisson::default();
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

// ============================================================================
// NLL Benchmarks
// ============================================================================

fn bench_nll_gaussian(c: &mut Criterion) {
    let mut group = c.benchmark_group("nll/gaussian");

    for n_samples in [100, 1_000, 10_000] {
        let dist = Gaussian::default();
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

fn bench_nll_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("nll/gamma");

    for n_samples in [100, 1_000, 10_000] {
        let dist = Gamma::default();
        let params = Array2::from_shape_fn((n_samples, 2), |(i, _)| 1.0 + (i % 5) as f64 / 10.0);
        let targets = generate_positive_targets(n_samples);
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

// ============================================================================
// Sampling Benchmarks
// ============================================================================

fn bench_sample_gaussian(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample/gaussian");

    for n_samples in [100, 1_000, 10_000] {
        let dist = Gaussian::default();
        let params = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                (i % 10) as f64 / 10.0
            } else {
                1.0 + (i % 5) as f64 / 10.0
            }
        });

        group.throughput(Throughput::Elements(n_samples as u64 * 100));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| b.iter(|| dist.sample(black_box(&params.view()), black_box(100), black_box(42))),
        );
    }
    group.finish();
}

fn bench_sample_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample/gamma");

    for n_samples in [100, 1_000, 10_000] {
        let dist = Gamma::default();
        let params = Array2::from_shape_fn((n_samples, 2), |(i, _)| 1.0 + (i % 5) as f64 / 10.0);

        group.throughput(Throughput::Elements(n_samples as u64 * 100));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| b.iter(|| dist.sample(black_box(&params.view()), black_box(100), black_box(42))),
        );
    }
    group.finish();
}

fn bench_sample_student_t(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample/student_t");

    for n_samples in [100, 1_000, 10_000] {
        let dist = StudentT::default();
        let params = Array2::from_shape_fn((n_samples, 3), |(i, j)| match j {
            0 => (i % 10) as f64 / 10.0,
            1 => 1.0 + (i % 5) as f64 / 10.0,
            _ => 3.0 + (i % 3) as f64,
        });

        group.throughput(Throughput::Elements(n_samples as u64 * 100));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| b.iter(|| dist.sample(black_box(&params.view()), black_box(100), black_box(42))),
        );
    }
    group.finish();
}

// ============================================================================
// Parameter Transformation Benchmarks
// ============================================================================

fn bench_transform_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_params");

    let distributions: Vec<(&str, Box<dyn Distribution>)> = vec![
        ("gaussian", Box::new(Gaussian::default())),
        ("gamma", Box::new(Gamma::default())),
        ("beta", Box::new(Beta::default())),
        ("student_t", Box::new(StudentT::default())),
        ("laplace", Box::new(Laplace::default())),
        ("logistic", Box::new(Logistic::default())),
    ];

    for (name, dist) in distributions {
        let n_samples = 10_000;
        let n_params = dist.n_params();
        let predictions = Array2::from_shape_fn((n_samples, n_params), |(i, j)| {
            ((i * 7 + j * 13) % 100) as f64 / 100.0 - 0.5
        });

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(BenchmarkId::new("dist", name), &predictions, |b, preds| {
            b.iter(|| dist.transform_params(black_box(&preds.view())))
        });
    }
    group.finish();
}

// ============================================================================
// Distribution Comparison Benchmarks
// ============================================================================

fn bench_gradient_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradients_comparison");
    group.sample_size(50);

    let n_samples = 5_000;

    // Gaussian
    {
        let (predictions, targets) = generate_data(n_samples, 2);
        let dist = Gaussian::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("gaussian", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Gamma
    {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = generate_positive_targets(n_samples);
        let dist = Gamma::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("gamma", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Beta
    {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = generate_unit_targets(n_samples);
        let dist = Beta::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("beta", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // StudentT
    {
        let (predictions, targets) = generate_data(n_samples, 3);
        let dist = StudentT::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("student_t", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Weibull
    {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = generate_positive_targets(n_samples);
        let dist = Weibull::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("weibull", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Laplace
    {
        let (predictions, targets) = generate_data(n_samples, 2);
        let dist = Laplace::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("laplace", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Cauchy
    {
        let (predictions, targets) = generate_data(n_samples, 2);
        let dist = Cauchy::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("cauchy", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Gumbel
    {
        let (predictions, targets) = generate_data(n_samples, 2);
        let dist = Gumbel::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("gumbel", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // LogNormal
    {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = generate_positive_targets(n_samples);
        let dist = LogNormal::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("log_normal", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // NegativeBinomial
    {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = generate_count_targets(n_samples);
        let dist = NegativeBinomial::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("negative_binomial", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Logistic
    {
        let (predictions, targets) = generate_data(n_samples, 2);
        let dist = Logistic::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("logistic", |b| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    // Poisson
    {
        let (predictions, _) = generate_data(n_samples, 1);
        let targets = generate_count_targets(n_samples);
        let dist = Poisson::default();
        let target = ResponseData::Univariate(&targets.view());

        group.bench_function("poisson", |b| {
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
// Start Values Calculation Benchmarks
// ============================================================================

fn bench_start_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("start_values");
    group.sample_size(20);

    for n_samples in [100, 500, 1_000] {
        let targets = generate_data(n_samples, 1).1;
        let dist = Gaussian::new(
            Stabilization::None,
            ResponseFn::Exp,
            gradientlss::distributions::LossFn::Nll,
            true,
        );
        let target = ResponseData::Univariate(&targets.view());

        group.bench_with_input(
            BenchmarkId::new("gaussian", n_samples),
            &n_samples,
            |b, _| b.iter(|| dist.calculate_start_values(black_box(&target), black_box(50))),
        );
    }

    for n_samples in [100, 500, 1_000] {
        let targets = generate_positive_targets(n_samples);
        let dist = Gamma::new(
            Stabilization::None,
            ResponseFn::Softplus,
            gradientlss::distributions::LossFn::Nll,
            true,
        );
        let target = ResponseData::Univariate(&targets.view());

        group.bench_with_input(BenchmarkId::new("gamma", n_samples), &n_samples, |b, _| {
            b.iter(|| dist.calculate_start_values(black_box(&target), black_box(50)))
        });
    }

    group.finish();
}

fn bench_nll_spline_flow(c: &mut Criterion) {
    use gradientlss::distributions::SplineFlow;
    use gradientlss::distributions::spline_flow::{SplineOrder, TargetSupport};

    let mut group = c.benchmark_group("nll/spline_flow");

    for n_samples in [100, 1_000, 10_000] {
        let dist = SplineFlow::new(
            TargetSupport::Real,
            8,
            3.0,
            SplineOrder::Linear,
            Stabilization::None,
            gradientlss::distributions::LossFn::Nll,
            false,
        );
        let n_params = dist.n_params();
        let params = Array2::from_shape_fn((n_samples, n_params), |(i, j)| {
            ((i * 7 + j * 13) % 100) as f64 / 200.0 - 0.25
        });
        let targets = Array1::from_shape_fn(n_samples, |i| (i % 10) as f64 / 5.0 - 1.0);
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

// ============================================================================
// Parallel Scaling Benchmark
// ============================================================================

fn bench_parallel_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scaling");
    group.sample_size(30);

    for n_samples in [128, 256, 512, 1024, 4096, 16384] {
        let (predictions, targets) = generate_data(n_samples, 2);
        let dist = Gaussian::default();
        let target = ResponseData::Univariate(&targets.view());

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("gaussian", n_samples),
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

    for n_samples in [128, 256, 512, 1024, 4096, 16384] {
        let (predictions, _) = generate_data(n_samples, 2);
        let targets = generate_positive_targets(n_samples);
        let dist = Gamma::default();
        let target = ResponseData::Univariate(&targets.view());

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(BenchmarkId::new("gamma", n_samples), &n_samples, |b, _| {
            b.iter(|| {
                dist.compute_gradients_and_hessians(
                    black_box(&predictions.view()),
                    black_box(&target),
                    None,
                )
            })
        });
    }

    for n_samples in [128, 256, 512, 1024, 4096, 16384] {
        let (predictions, targets) = generate_data(n_samples, 3);
        let dist = StudentT::default();
        let target = ResponseData::Univariate(&targets.view());

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("student_t", n_samples),
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

criterion_group!(
    benches,
    bench_gradients_gaussian,
    bench_gradients_gamma,
    bench_gradients_student_t,
    bench_gradients_beta,
    bench_gradients_poisson,
    bench_nll_gaussian,
    bench_nll_gamma,
    bench_nll_spline_flow,
    bench_sample_gaussian,
    bench_sample_gamma,
    bench_sample_student_t,
    bench_transform_params,
    bench_gradient_comparison,
    bench_start_values,
    bench_parallel_scaling,
);

criterion_main!(benches);
