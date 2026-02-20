//! A/B comparison benchmarks: OLD (pre-optimization) vs NEW (optimized) NLL implementations.
//!
//! For univariate distributions: OLD uses params[[i,j]] 2D indexing, NEW uses column slices.
//! For SplineFlow: OLD allocates buffers per row, NEW reuses pre-allocated buffers.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gradientlss::distributions::{
    Beta, Distribution, Gamma, Gaussian, Laplace, LossFn, Poisson, SplineFlow, SplineOrder,
    Stabilization, StudentT, TargetSupport, Weibull,
};
use gradientlss::types::ResponseData;
use gradientlss::utils::ResponseFn;
use ndarray::{Array1, Array2, ArrayView2};
use std::hint::black_box;

// ============================================================
// OLD NLL implementations: inline math with params[[i, j]] indexing
// ============================================================

fn gaussian_nll_old(params: &ArrayView2<f64>, y: &[f64]) -> f64 {
    let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let mut total = 0.0;
    for (i, &y_val) in y.iter().enumerate() {
        let loc = params[[i, 0]];
        let scale = params[[i, 1]];
        if scale <= 0.0 {
            total += 1e10;
        } else {
            let z = (y_val - loc) / scale;
            total += half_ln_2pi + scale.ln() + 0.5 * z * z;
        }
    }
    total
}

fn gamma_nll_old(params: &ArrayView2<f64>, y: &[f64]) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let mut total = 0.0;
    for (i, &y_val) in y.iter().enumerate() {
        let conc = params[[i, 0]];
        let rate = params[[i, 1]];
        if conc <= 0.0 || rate <= 0.0 || y_val <= 0.0 {
            total += 1e10;
        } else {
            total -= conc * rate.ln() + (conc - 1.0) * y_val.ln() - rate * y_val - ln_gamma(conc);
        }
    }
    total
}

fn student_t_nll_old(params: &ArrayView2<f64>, y: &[f64]) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let mut total = 0.0;
    for (i, &y_val) in y.iter().enumerate() {
        let df = params[[i, 0]];
        let loc = params[[i, 1]];
        let scale = params[[i, 2]];
        if df <= 0.0 || scale <= 0.0 {
            total += 1e10;
        } else {
            let z = (y_val - loc) / scale;
            total -= ln_gamma(0.5 * (df + 1.0))
                - ln_gamma(0.5 * df)
                - 0.5 * (df * std::f64::consts::PI).ln()
                - scale.ln()
                - 0.5 * (df + 1.0) * (1.0 + z * z / df).ln();
        }
    }
    total
}

fn beta_nll_old(params: &ArrayView2<f64>, y: &[f64]) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let mut total = 0.0;
    for (i, &y_val) in y.iter().enumerate() {
        let alpha = params[[i, 0]];
        let beta = params[[i, 1]];
        if alpha <= 0.0 || beta <= 0.0 || y_val <= 0.0 || y_val >= 1.0 {
            total += 1e10;
        } else {
            total -= ln_gamma(alpha + beta) - ln_gamma(alpha) - ln_gamma(beta)
                + (alpha - 1.0) * y_val.ln()
                + (beta - 1.0) * (1.0 - y_val).ln();
        }
    }
    total
}

fn laplace_nll_old(params: &ArrayView2<f64>, y: &[f64]) -> f64 {
    let ln_2 = 2.0_f64.ln();
    let mut total = 0.0;
    for (i, &y_val) in y.iter().enumerate() {
        let loc = params[[i, 0]];
        let scale = params[[i, 1]];
        if scale <= 0.0 {
            total += 1e10;
        } else {
            total += ln_2 + scale.ln() + (y_val - loc).abs() / scale;
        }
    }
    total
}

fn weibull_nll_old(params: &ArrayView2<f64>, y: &[f64]) -> f64 {
    let mut total = 0.0;
    for (i, &y_val) in y.iter().enumerate() {
        let shape = params[[i, 0]];
        let scale = params[[i, 1]];
        if shape <= 0.0 || scale <= 0.0 || y_val < 0.0 {
            total += 1e10;
        } else {
            total -= shape.ln() - shape * scale.ln() + (shape - 1.0) * y_val.ln()
                - (y_val / scale).powf(shape);
        }
    }
    total
}

fn poisson_nll_old(params: &ArrayView2<f64>, y: &[f64]) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let mut total = 0.0;
    for (i, &y_val) in y.iter().enumerate() {
        let rate = params[[i, 0]];
        if rate <= 0.0 {
            total += 1e10;
        } else {
            total -= y_val * rate.ln() - rate - ln_gamma(y_val + 1.0);
        }
    }
    total
}

// ============================================================
// Data preparation helpers
// ============================================================

fn prepare_gaussian(n: usize) -> (Gaussian, Array2<f64>, Array1<f64>) {
    let dist = Gaussian::new(
        Stabilization::None,
        ResponseFn::Softplus,
        LossFn::Nll,
        false,
    );
    let params = Array2::from_shape_fn((n, 2), |(i, _)| 0.5 + (i % 10) as f64 / 10.0);
    let transformed = dist.transform_params(&params.view());
    let targets = Array1::from_shape_fn(n, |i| 0.5 + (i % 10) as f64 / 10.0);
    (dist, transformed, targets)
}

fn prepare_gamma(n: usize) -> (Gamma, Array2<f64>, Array1<f64>) {
    let dist = Gamma::new(
        Stabilization::None,
        ResponseFn::Softplus,
        LossFn::Nll,
        false,
    );
    let params = Array2::from_shape_fn((n, 2), |(i, _)| 0.5 + (i % 10) as f64 / 10.0);
    let transformed = dist.transform_params(&params.view());
    let targets = Array1::from_shape_fn(n, |i| 0.1 + (i % 10) as f64 / 5.0);
    (dist, transformed, targets)
}

fn prepare_student_t(n: usize) -> (StudentT, Array2<f64>, Array1<f64>) {
    let dist = StudentT::new(
        Stabilization::None,
        ResponseFn::Softplus,
        ResponseFn::Softplus,
        LossFn::Nll,
        false,
    );
    let params = Array2::from_shape_fn((n, 3), |(i, _)| 0.5 + (i % 10) as f64 / 10.0);
    let transformed = dist.transform_params(&params.view());
    let targets = Array1::from_shape_fn(n, |i| 0.5 + (i % 10) as f64 / 10.0);
    (dist, transformed, targets)
}

fn prepare_beta(n: usize) -> (Beta, Array2<f64>, Array1<f64>) {
    let dist = Beta::new(
        Stabilization::None,
        ResponseFn::Softplus,
        LossFn::Nll,
        false,
    );
    let params = Array2::from_shape_fn((n, 2), |(i, _)| 0.5 + (i % 10) as f64 / 10.0);
    let transformed = dist.transform_params(&params.view());
    let targets = Array1::from_shape_fn(n, |i| 0.05 + (i % 9) as f64 / 10.0);
    (dist, transformed, targets)
}

fn prepare_laplace(n: usize) -> (Laplace, Array2<f64>, Array1<f64>) {
    let dist = Laplace::new(
        Stabilization::None,
        ResponseFn::Softplus,
        LossFn::Nll,
        false,
    );
    let params = Array2::from_shape_fn((n, 2), |(i, _)| 0.5 + (i % 10) as f64 / 10.0);
    let transformed = dist.transform_params(&params.view());
    let targets = Array1::from_shape_fn(n, |i| 0.5 + (i % 10) as f64 / 10.0);
    (dist, transformed, targets)
}

fn prepare_weibull(n: usize) -> (Weibull, Array2<f64>, Array1<f64>) {
    let dist = Weibull::new(
        Stabilization::None,
        ResponseFn::Softplus,
        ResponseFn::Softplus,
        LossFn::Nll,
        false,
    );
    let params = Array2::from_shape_fn((n, 2), |(i, _)| 0.5 + (i % 10) as f64 / 10.0);
    let transformed = dist.transform_params(&params.view());
    let targets = Array1::from_shape_fn(n, |i| 0.1 + (i % 10) as f64 / 5.0);
    (dist, transformed, targets)
}

fn prepare_poisson(n: usize) -> (Poisson, Array2<f64>, Array1<f64>) {
    let dist = Poisson::new(
        Stabilization::None,
        ResponseFn::Softplus,
        LossFn::Nll,
        false,
    );
    let params = Array2::from_shape_fn((n, 1), |(i, _)| 0.5 + (i % 10) as f64 / 10.0);
    let transformed = dist.transform_params(&params.view());
    let targets = Array1::from_shape_fn(n, |i| (i % 20) as f64);
    (dist, transformed, targets)
}

fn prepare_spline_flow(n: usize) -> (SplineFlow, Array2<f64>, Array1<f64>) {
    let dist = SplineFlow::new(
        TargetSupport::Real,
        8,
        3.0,
        SplineOrder::Quadratic,
        Stabilization::None,
        LossFn::Nll,
        false,
    );
    let n_params = dist.n_params();
    let params = Array2::from_shape_fn((n, n_params), |(i, j)| {
        0.1 + ((i * 7 + j * 13) % 100) as f64 / 100.0
    });
    let transformed = dist.transform_params(&params.view());
    let targets = Array1::from_shape_fn(n, |i| 0.1 + (i % 10) as f64 / 10.0);
    (dist, transformed, targets)
}

// ============================================================
// Univariate NLL: old [[i,j]] vs new column-slice
// ============================================================

fn bench_univariate_nll(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];

    macro_rules! bench_dist {
        ($group_name:expr, $prepare:ident, $old_fn:ident, $c:expr, $sizes:expr) => {
            let mut group = $c.benchmark_group($group_name);
            for &n in $sizes {
                let (dist, params, targets) = $prepare(n);
                let view = params.view();
                let y_view = targets.view();
                let target = ResponseData::Univariate(&y_view);
                let y_slice = targets.as_slice().unwrap();

                group.bench_with_input(
                    BenchmarkId::new("old_2d_index", n),
                    &n,
                    |b: &mut criterion::Bencher, _| {
                        b.iter(|| $old_fn(black_box(&view), black_box(y_slice)))
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("new_col_slice", n),
                    &n,
                    |b: &mut criterion::Bencher, _| {
                        b.iter(|| dist.nll(black_box(&view), black_box(&target)))
                    },
                );
            }
            group.finish();
        };
    }

    bench_dist!(
        "ab_gaussian_nll",
        prepare_gaussian,
        gaussian_nll_old,
        c,
        &sizes
    );
    bench_dist!("ab_gamma_nll", prepare_gamma, gamma_nll_old, c, &sizes);
    bench_dist!(
        "ab_student_t_nll",
        prepare_student_t,
        student_t_nll_old,
        c,
        &sizes
    );
    bench_dist!("ab_beta_nll", prepare_beta, beta_nll_old, c, &sizes);
    bench_dist!(
        "ab_laplace_nll",
        prepare_laplace,
        laplace_nll_old,
        c,
        &sizes
    );
    bench_dist!(
        "ab_weibull_nll",
        prepare_weibull,
        weibull_nll_old,
        c,
        &sizes
    );
    bench_dist!(
        "ab_poisson_nll",
        prepare_poisson,
        poisson_nll_old,
        c,
        &sizes
    );
}

// ============================================================
// SplineFlow: old per-row alloc vs new buffer reuse
// ============================================================

fn bench_spline_flow_nll(c: &mut Criterion) {
    let sizes = [100, 1000];
    let mut group = c.benchmark_group("ab_spline_flow_nll");
    group.sample_size(30);

    for &n in &sizes {
        let (dist, params, targets) = prepare_spline_flow(n);
        let view = params.view();
        let y_view = targets.view();
        let target = ResponseData::Univariate(&y_view);

        // OLD: per-row nll call (each 1-row nll allocates fresh buffers inside)
        group.bench_with_input(
            BenchmarkId::new("old_per_row_nll", n),
            &n,
            |b: &mut criterion::Bencher, _| {
                b.iter(|| {
                    let mut total = 0.0;
                    for (i, &y_val) in targets.iter().enumerate() {
                        let single_row = view.slice(ndarray::s![i..i + 1, ..]);
                        let y_arr = ndarray::arr1(&[y_val]);
                        let y_v = y_arr.view();
                        let t = ResponseData::Univariate(&y_v);
                        total += dist.nll(&single_row, &t);
                    }
                    total
                })
            },
        );

        // NEW: batch nll with pre-allocated buffers
        group.bench_with_input(
            BenchmarkId::new("new_buffer_reuse", n),
            &n,
            |b: &mut criterion::Bencher, _| {
                b.iter(|| dist.nll(black_box(&view), black_box(&target)))
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_univariate_nll, bench_spline_flow_nll);
criterion_main!(benches);
