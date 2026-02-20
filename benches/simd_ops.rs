//! Benchmarks for SIMD operations.
//!
//! Compares SIMD-optimized response functions against scalar implementations.
//!
//! KEY INSIGHT: The `wide` crate's exp_x4 implementation calls scalar exp() 4 times,
//! which doesn't provide SIMD benefit for transcendental functions. The benefit
//! only comes from simple arithmetic operations (add, mul, sqrt, etc.)
//!
//! For true SIMD exp/sigmoid, you'd need:
//! - A polynomial approximation of exp() using SIMD operations
//! - Or use a crate like `sleef-rs` or `packed_simd` with intrinsic transcendentals

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gradientlss::simd_ops;
use std::hint::black_box;

fn generate_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 / n as f64) * 10.0 - 5.0).collect()
}

/// Scalar implementation using ndarray-style mapv (auto-vectorizable for simple ops)
fn exp_ndarray_style(input: &[f64], output: &mut [f64]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.exp() + 1e-6;
    }
}

fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/exp");

    for n in [1000, 10_000, 100_000] {
        let input = generate_input(n);
        let mut output_simd = vec![0.0; n];
        let mut output_scalar = vec![0.0; n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
            b.iter(|| {
                simd_ops::exp_simd(black_box(&input), black_box(&mut output_simd));
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                for (i, &x) in input.iter().enumerate() {
                    output_scalar[i] = x.exp() + 1e-6;
                }
                black_box(&output_scalar);
            })
        });
    }
    group.finish();
}

fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/sigmoid");

    for n in [1000, 10_000, 100_000] {
        let input = generate_input(n);
        let mut output_simd = vec![0.0; n];
        let mut output_scalar = vec![0.0; n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
            b.iter(|| {
                simd_ops::sigmoid_simd(black_box(&input), black_box(&mut output_simd));
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                for (i, &x) in input.iter().enumerate() {
                    let s = 1.0 / (1.0 + (-x).exp()) + 1e-6;
                    output_scalar[i] = s.clamp(1e-3, 1.0 - 1e-3);
                }
                black_box(&output_scalar);
            })
        });
    }
    group.finish();
}

fn bench_softplus(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/softplus");

    for n in [1000, 10_000, 100_000] {
        let input = generate_input(n);
        let mut output_simd = vec![0.0; n];
        let mut output_scalar = vec![0.0; n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
            b.iter(|| {
                simd_ops::softplus_simd(black_box(&input), black_box(&mut output_simd));
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                for (i, &x) in input.iter().enumerate() {
                    output_scalar[i] = if x > 20.0 {
                        x + 1e-6
                    } else if x < -20.0 {
                        1e-6
                    } else {
                        (1.0 + x.exp()).ln() + 1e-6
                    };
                }
                black_box(&output_scalar);
            })
        });
    }
    group.finish();
}

// NOTE: squareplus uses only arithmetic ops (*, +, sqrt) - should show SIMD benefit
fn bench_squareplus(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/squareplus");

    for n in [1000, 10_000, 100_000] {
        let input = generate_input(n);
        let mut output_simd = vec![0.0; n];
        let mut output_scalar = vec![0.0; n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
            b.iter(|| {
                simd_ops::squareplus_simd(black_box(&input), black_box(&mut output_simd));
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                for (i, &x) in input.iter().enumerate() {
                    output_scalar[i] = 0.5 * (x + (x * x + 4.0).sqrt()) + 1e-6;
                }
                black_box(&output_scalar);
            })
        });
    }
    group.finish();
}

// NOTE: relu uses only max() - should show SIMD benefit, but compiler may auto-vectorize scalar too
fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/relu");

    for n in [1000, 10_000, 100_000] {
        let input = generate_input(n);
        let mut output_simd = vec![0.0; n];
        let mut output_scalar = vec![0.0; n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
            b.iter(|| {
                simd_ops::relu_simd(black_box(&input), black_box(&mut output_simd));
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                for (i, &x) in input.iter().enumerate() {
                    output_scalar[i] = x.max(0.0) + 1e-6;
                }
                black_box(&output_scalar);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_exp,
    bench_sigmoid,
    bench_softplus,
    bench_squareplus,
    bench_relu,
);

criterion_main!(benches);
