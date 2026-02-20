//! Benchmarks for optimization improvements.
//!
//! This module benchmarks the specific optimizations implemented:
//! - Single-pass softmax vs multi-allocation softmax
//! - apply_into (SIMD) vs apply (allocating) for response functions
//! - MVN nll parallelization (sequential vs parallel at different thresholds)
//! - Mixture nll parallelization
//! - Buffer reuse patterns

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gradientlss::distributions::{Distribution, LossFn, MVN, Mixture, Stabilization};
use gradientlss::types::ResponseData;
use gradientlss::utils::ResponseFn;
use ndarray::{Array1, Array2, ArrayView1};
use std::hint::black_box;

// ============================================================================
// Softmax Optimization Benchmarks
// ============================================================================

/// Old softmax implementation (multiple allocations)
fn softmax_old(x: &ArrayView1<f64>) -> Array1<f64> {
    // This mimics the old implementation with multiple passes and allocations

    // Pass 1: Handle NaN values (allocation 1)
    let clean: Vec<f64> = x
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0.0 })
        .collect();

    // Pass 2: Find max (no allocation but separate pass)
    let max_val = clean.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Pass 3: Compute exp (allocation 2)
    let exp_vals: Vec<f64> = clean.iter().map(|&v| (v - max_val).exp()).collect();

    // Pass 4: Compute sum
    let sum: f64 = exp_vals.iter().sum();

    // Pass 5: Normalize (allocation 3)
    Array1::from_vec(exp_vals.iter().map(|&v| v / sum).collect())
}

/// BAD: Fused softmax - kept as example of what NOT to do.
/// The loop-carried dependency (sum_exp += exp_v) prevents compiler auto-vectorization,
/// making this SLOWER than the multi-allocation version at large sizes.
fn softmax_fused(x: &ArrayView1<f64>) -> Array1<f64> {
    // Single-pass to compute mean for NaN replacement and find max
    let (sum, count, max_val) = x
        .iter()
        .fold((0.0, 0usize, f64::NEG_INFINITY), |(s, c, m), &v| {
            if v.is_finite() {
                (s + v, c + 1, m.max(v))
            } else {
                (s, c, m)
            }
        });
    let nan_replacement = if count > 0 { sum / count as f64 } else { 0.0 };
    let max_val = if max_val.is_finite() {
        max_val
    } else {
        nan_replacement
    };

    // Fused exp-sum computation with single result allocation
    // NOTE: This has a loop-carried dependency (sum_exp += exp_v) that prevents vectorization
    let mut result = Array1::zeros(x.len());
    let mut sum_exp = 0.0;

    for (i, &v) in x.iter().enumerate() {
        let v = if v.is_finite() { v } else { nan_replacement };
        let exp_v = (v - max_val).exp();
        result[i] = exp_v;
        sum_exp += exp_v;
    }

    // Normalize in-place
    if sum_exp > 0.0 {
        result.mapv_inplace(|v| v / sum_exp);
    }

    result
}

/// Optimized softmax using ndarray's vectorized operations
/// This version allows the compiler to auto-vectorize each pass
fn softmax_vectorized(x: &ArrayView1<f64>) -> Array1<f64> {
    // Use ndarray's built-in operations which are SIMD-optimized

    // Find max (vectorized reduction)
    let max_val = x.fold(
        f64::NEG_INFINITY,
        |a, &b| if b.is_finite() { a.max(b) } else { a },
    );
    let max_val = if max_val.is_finite() { max_val } else { 0.0 };

    // Compute exp(x - max) in one vectorized pass (auto-vectorizable)
    let mut result = x.mapv(|v| (v - max_val).exp());

    // Sum reduction (vectorized)
    let sum_exp: f64 = result.sum();

    // Normalize in-place (vectorized)
    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        result.mapv_inplace(|v| v * inv_sum);
    }

    result
}

fn bench_softmax_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/softmax");

    for n in [10, 100, 1_000, 10_000] {
        let input = Array1::from_shape_fn(n, |i| (i as f64 / n as f64) * 10.0 - 5.0);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("multi_alloc", n), &n, |b, _| {
            b.iter(|| softmax_old(black_box(&input.view())))
        });

        group.bench_with_input(BenchmarkId::new("fused_loop", n), &n, |b, _| {
            b.iter(|| softmax_fused(black_box(&input.view())))
        });

        group.bench_with_input(BenchmarkId::new("vectorized", n), &n, |b, _| {
            b.iter(|| softmax_vectorized(black_box(&input.view())))
        });
    }
    group.finish();
}

// ============================================================================
// Response Function apply vs apply_into Benchmarks
//
// KEY INSIGHT: apply_into has overhead from:
// 1. Computing mean for NaN replacement (iterates entire array)
// 2. Checking if any NaN/Inf exists (iterates entire array again)
// 3. SIMD dispatch logic
//
// This overhead can outweigh the allocation savings for small arrays.
// The benefit shows up when:
// - Array is large (allocation cost dominates)
// - Called in a tight loop with the same output buffer
// ============================================================================

fn bench_apply_vs_apply_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/apply_into");

    let response_fns = [
        ("exp", ResponseFn::Exp),
        ("sigmoid", ResponseFn::Sigmoid),
        ("softplus", ResponseFn::Softplus),
        ("identity", ResponseFn::Identity),
    ];

    for (name, response_fn) in response_fns {
        for n in [1_000, 10_000, 100_000] {
            let input = Array1::from_shape_fn(n, |i| (i as f64 / n as f64) * 10.0 - 5.0);
            group.throughput(Throughput::Elements(n as u64));

            // apply() allocates new array, uses nan_to_num + mapv
            group.bench_with_input(
                BenchmarkId::new(format!("{}/apply_alloc", name), n),
                &n,
                |b, _| b.iter(|| response_fn.apply(black_box(&input.view()))),
            );

            // apply_into() writes to pre-allocated buffer, zero-allocation path
            let mut output = Array1::zeros(n);
            group.bench_with_input(
                BenchmarkId::new(format!("{}/apply_into", name), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        response_fn
                            .apply_into(black_box(&input.view()), black_box(&mut output.view_mut()))
                    })
                },
            );
        }
    }
    group.finish();
}

// ============================================================================
// MVN NLL Parallelization Benchmarks
// ============================================================================

fn generate_mvn_data(n_samples: usize, n_targets: usize) -> (Array2<f64>, Array2<f64>) {
    // MVN has n_targets location params + n_targets*(n_targets+1)/2 tril params
    let n_tril = n_targets * (n_targets + 1) / 2;
    let n_params = n_targets + n_tril;

    let params = Array2::from_shape_fn((n_samples, n_params), |(i, j)| {
        if j < n_targets {
            // Location parameters
            (i % 10) as f64 / 10.0
        } else {
            // Tril parameters - need reasonable values for valid covariance
            let tril_idx = j - n_targets;
            // Diagonal elements should be positive (will be exp'd)
            // Off-diagonal can be any value
            if is_diagonal_tril_element(tril_idx, n_targets) {
                0.5 + (i % 5) as f64 / 10.0 // Results in ~1.6-2.0 after exp
            } else {
                ((i * 7 + j * 13) % 100) as f64 / 200.0 - 0.25 // Small off-diagonal
            }
        }
    });

    let targets = Array2::from_shape_fn((n_samples, n_targets), |(i, j)| {
        (i % 10) as f64 / 10.0 + (j as f64 * 0.1)
    });

    (params, targets)
}

/// Check if a tril index corresponds to a diagonal element
fn is_diagonal_tril_element(tril_idx: usize, n_targets: usize) -> bool {
    let mut idx = 0;
    for col in 0..n_targets {
        if idx == tril_idx {
            return true;
        }
        idx += n_targets - col;
    }
    false
}

fn bench_mvn_nll_parallelization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/mvn_nll");
    group.sample_size(30);

    let n_targets = 3; // 3D MVN

    // Test different sample sizes to see parallelization benefit
    for n_samples in [100, 256, 500, 1_000, 2_500, 5_000] {
        let dist = MVN::new(
            n_targets,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        );
        let (params, targets) = generate_mvn_data(n_samples, n_targets);
        let target = ResponseData::Multivariate(&targets.view());

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| b.iter(|| dist.nll(black_box(&params.view()), black_box(&target))),
        );
    }
    group.finish();
}

fn bench_mvn_sample_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/mvn_sample");
    group.sample_size(30);

    let n_targets = 3;

    for n_obs in [100, 500, 1_000, 2_500] {
        let dist = MVN::new(
            n_targets,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        );
        let (params, _) = generate_mvn_data(n_obs, n_targets);
        let n_samples = 100;

        group.throughput(Throughput::Elements((n_obs * n_samples) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n_obs), &n_obs, |b, _| {
            b.iter(|| {
                dist.sample(
                    black_box(&params.view()),
                    black_box(n_samples),
                    black_box(42),
                )
            })
        });
    }
    group.finish();
}

// ============================================================================
// Mixture NLL Parallelization Benchmarks
// ============================================================================

fn generate_mixture_data(n_samples: usize, n_components: usize) -> (Array2<f64>, Array1<f64>) {
    // Mixture has n_components mix_probs + n_components locs + n_components scales
    let n_params = 3 * n_components;

    let params = Array2::from_shape_fn((n_samples, n_params), |(i, j)| {
        if j < n_components {
            // Mix prob logits
            ((i * 7 + j * 13) % 100) as f64 / 100.0
        } else if j < 2 * n_components {
            // Locations
            (i % 10) as f64 / 5.0 - 1.0
        } else {
            // Scales (before exp)
            0.5 + (i % 5) as f64 / 10.0
        }
    });

    let targets = Array1::from_shape_fn(n_samples, |i| (i % 10) as f64 / 5.0 - 1.0);

    (params, targets)
}

fn bench_mixture_nll_parallelization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/mixture_nll");
    group.sample_size(30);

    let n_components = 3;

    // Test different sample sizes to see parallelization benefit
    for n_samples in [100, 256, 500, 1_000, 2_500, 5_000] {
        let dist = Mixture::new(n_components, 1.0, Stabilization::None, LossFn::Nll, false);
        let (params, targets) = generate_mixture_data(n_samples, n_components);
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

fn bench_mixture_sample_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/mixture_sample");
    group.sample_size(30);

    let n_components = 3;

    for n_obs in [100, 500, 1_000, 2_500] {
        let dist = Mixture::new(n_components, 1.0, Stabilization::None, LossFn::Nll, false);
        let (params, _) = generate_mixture_data(n_obs, n_components);
        let n_samples = 100;

        group.throughput(Throughput::Elements((n_obs * n_samples) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n_obs), &n_obs, |b, _| {
            b.iter(|| {
                dist.sample(
                    black_box(&params.view()),
                    black_box(n_samples),
                    black_box(42),
                )
            })
        });
    }
    group.finish();
}

// ============================================================================
// Buffer Reuse Pattern Benchmarks
//
// Compares allocation-per-iteration vs buffer reuse patterns.
// This is relevant for gradient/hessian computation where we need temporary buffers.
// ============================================================================

/// Pattern 1: Allocate a new buffer each iteration (common naive pattern)
fn buffer_alloc_each_iter(iterations: usize, buffer_size: usize) -> f32 {
    let mut total = 0.0f32;

    for i in 0..iterations {
        // Allocate fresh buffer each iteration
        let mut buffer: Vec<f32> = vec![0.0; buffer_size];

        // Do some work with the buffer
        for (j, v) in buffer.iter_mut().enumerate() {
            *v = ((i + j) as f32).sin();
        }

        // Accumulate result
        total += buffer.iter().sum::<f32>();
    }

    total
}

/// Pattern 2: Reuse a single buffer across iterations
fn buffer_reuse_pattern(iterations: usize, buffer_size: usize) -> f32 {
    let mut total = 0.0f32;
    let mut buffer: Vec<f32> = vec![0.0; buffer_size];

    for i in 0..iterations {
        // Reuse the same buffer - just overwrite contents
        for (j, v) in buffer.iter_mut().enumerate() {
            *v = ((i + j) as f32).sin();
        }

        // Accumulate result
        total += buffer.iter().sum::<f32>();
    }

    total
}

fn bench_buffer_reuse_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/buffer_reuse");

    for buffer_size in [1_000, 10_000, 100_000] {
        let iterations = 100;

        group.throughput(Throughput::Elements((buffer_size * iterations) as u64));

        group.bench_with_input(
            BenchmarkId::new("alloc_each_iter", buffer_size),
            &buffer_size,
            |b, &size| b.iter(|| buffer_alloc_each_iter(black_box(iterations), black_box(size))),
        );

        group.bench_with_input(
            BenchmarkId::new("reuse_buffer", buffer_size),
            &buffer_size,
            |b, &size| b.iter(|| buffer_reuse_pattern(black_box(iterations), black_box(size))),
        );
    }
    group.finish();
}

// ============================================================================
// Single-pass nan_to_num Benchmark
// ============================================================================

/// Old nan_to_num: multiple passes
fn nan_to_num_old(x: &ArrayView1<f64>) -> Array1<f64> {
    // Pass 1: Collect valid values (allocation)
    let valid: Vec<f64> = x.iter().filter(|v| v.is_finite()).copied().collect();

    // Pass 2: Compute mean
    let mean = if valid.is_empty() {
        0.0
    } else {
        valid.iter().sum::<f64>() / valid.len() as f64
    };

    // Pass 3: Replace NaN values (allocation)
    x.mapv(|v| if v.is_finite() { v } else { mean })
}

/// New nan_to_num: single pass
fn nan_to_num_new(x: &ArrayView1<f64>) -> Array1<f64> {
    // Single-pass mean computation - no intermediate Vec allocation
    let (sum, count) = x.iter().fold((0.0, 0usize), |(s, c), &v| {
        if v.is_finite() {
            (s + v, c + 1)
        } else {
            (s, c)
        }
    });
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };

    x.mapv(|v| if v.is_finite() { v } else { mean })
}

fn bench_nan_to_num_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/nan_to_num");

    for n in [1_000, 10_000, 100_000] {
        // Create input with ~10% NaN values
        let input = Array1::from_shape_fn(n, |i| {
            if i % 10 == 0 {
                f64::NAN
            } else {
                (i as f64 / n as f64) * 10.0 - 5.0
            }
        });

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("old_multi_pass", n), &n, |b, _| {
            b.iter(|| nan_to_num_old(black_box(&input.view())))
        });

        group.bench_with_input(BenchmarkId::new("new_single_pass", n), &n, |b, _| {
            b.iter(|| nan_to_num_new(black_box(&input.view())))
        });
    }
    group.finish();
}

// ============================================================================
// Parallelization Threshold Analysis
// ============================================================================

fn bench_parallel_threshold_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/parallel_threshold");
    group.sample_size(20);

    let n_targets = 3;

    // Fine-grained analysis around the threshold (256)
    for n_samples in [128, 192, 256, 320, 384, 512, 768, 1024] {
        let dist = MVN::new(
            n_targets,
            Stabilization::None,
            ResponseFn::Exp,
            LossFn::Nll,
            false,
        );
        let (params, targets) = generate_mvn_data(n_samples, n_targets);
        let target = ResponseData::Multivariate(&targets.view());

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("mvn_nll", n_samples),
            &n_samples,
            |b, _| b.iter(|| dist.nll(black_box(&params.view()), black_box(&target))),
        );
    }
    group.finish();
}

// ============================================================================
// Main
// ============================================================================

criterion_group!(
    optimization_benches,
    bench_softmax_optimization,
    bench_apply_vs_apply_into,
    bench_nan_to_num_optimization,
    bench_buffer_reuse_pattern,
    bench_mvn_nll_parallelization,
    bench_mvn_sample_optimization,
    bench_mixture_nll_parallelization,
    bench_mixture_sample_optimization,
    bench_parallel_threshold_analysis,
);

criterion_main!(optimization_benches);
