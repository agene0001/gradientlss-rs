//! Hot-path benchmarks for the per-round gradient/metric passes.
//!
//! These isolate the exact functions the perf work targets, so optimizations
//! can be A/B verified against a saved criterion baseline:
//!
//! ```bash
//! cargo bench --bench hot_paths -- --save-baseline before   # pre-change
//! cargo bench --bench hot_paths -- --baseline before        # post-change
//! ```
//!
//! No backend features required — everything here is distribution-side code
//! that runs identically under XGBoost and LightGBM.

use criterion::{Criterion, criterion_group, criterion_main};
use gradientlss::distributions::{
    Distribution, Gaussian, LossFn, MVN, Mixture, NegativeBinomial, SplineFlow, Stabilization,
    spline_flow::{SplineOrder, TargetSupport},
};
use gradientlss::types::ResponseData;
use gradientlss::utils::ResponseFn;
use ndarray::{Array1, Array2};
use std::hint::black_box;

/// Deterministic pseudo-random f64 in [0, 1) — no rand dependency, stable
/// across runs so baselines compare identical work.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f64) / (u32::MAX as f64)
}

/// Raw margins in a moderate range plus matching continuous targets.
fn margins_and_targets(n: usize, n_params: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut state = seed;
    let mut preds = Array2::zeros((n, n_params));
    let mut y = Array1::zeros(n);
    for i in 0..n {
        for j in 0..n_params {
            preds[[i, j]] = (lcg(&mut state) - 0.5) * 2.0;
        }
        y[i] = (lcg(&mut state) - 0.5) * 4.0;
    }
    (preds, y)
}

fn bench_mixture_gradients(c: &mut Criterion) {
    let n = 10_000;
    let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);
    let (preds, y) = margins_and_targets(n, dist.n_params(), 0x1234_5678);

    c.bench_function("hot_paths/mixture_gradients_10k", |b| {
        b.iter(|| {
            let yv = y.view();
            let target = ResponseData::Univariate(&yv);
            black_box(
                dist.compute_gradients_and_hessians(black_box(&preds.view()), &target, None)
                    .unwrap(),
            )
        });
    });
}

fn bench_mixture_nll(c: &mut Criterion) {
    let n = 10_000;
    let dist = Mixture::new(2, 1.0, Stabilization::None, LossFn::Nll, false);
    let (preds, y) = margins_and_targets(n, dist.n_params(), 0x9abc_def0);
    let transformed = dist.transform_params(&preds.view());

    c.bench_function("hot_paths/mixture_nll_10k", |b| {
        b.iter(|| {
            let yv = y.view();
            let target = ResponseData::Univariate(&yv);
            black_box(dist.nll(black_box(&transformed.view()), &target))
        });
    });
}

fn bench_spline_flow_gradients(c: &mut Criterion) {
    let n = 2_000;
    let dist = SplineFlow::new(
        TargetSupport::Real,
        4,
        1.0,
        SplineOrder::Quadratic,
        Stabilization::None,
        LossFn::Nll,
        false,
    );
    let (preds, y) = margins_and_targets(n, dist.n_params(), 0x0dd_ba11);

    let mut group = c.benchmark_group("hot_paths");
    group.sample_size(20);
    group.bench_function("spline_flow_gradients_2k", |b| {
        b.iter(|| {
            let yv = y.view();
            let target = ResponseData::Univariate(&yv);
            black_box(
                dist.compute_gradients_and_hessians(black_box(&preds.view()), &target, None)
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_crps_gradients(c: &mut Criterion) {
    let n = 512;
    let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);
    let (preds, y) = margins_and_targets(n, dist.n_params(), 0xc0ff_ee00);

    let mut group = c.benchmark_group("hot_paths");
    group.sample_size(20);
    group.bench_function("gaussian_crps_gradients_512", |b| {
        b.iter(|| {
            let yv = y.view();
            let target = ResponseData::Univariate(&yv);
            black_box(
                dist.compute_gradients_and_hessians(black_box(&preds.view()), &target, None)
                    .unwrap(),
            )
        });
    });
    group.finish();
}

/// Start-value L-BFGS fit on count data — the `initialize: true` cost paid by
/// every plain `train` (hyper_opt caches it across trials, but the first fit
/// always runs). Cost and finite-difference gradient are full-dataset NLL
/// passes; `max_iter: 8` keeps the bench to a few L-BFGS iterations.
fn bench_start_values(c: &mut Criterion) {
    let n = 20_000;
    let dist = NegativeBinomial::new(
        Stabilization::None,
        ResponseFn::Exp,
        ResponseFn::Sigmoid,
        LossFn::Nll,
        true,
    );
    let mut state = 0xdead_beefu64;
    let y = Array1::from_iter((0..n).map(|_| (lcg(&mut state) * 10.0).floor()));

    let mut group = c.benchmark_group("hot_paths");
    group.sample_size(10);
    group.bench_function("start_values_negbinom_20k", |b| {
        b.iter(|| {
            let yv = y.view();
            let target = ResponseData::Univariate(&yv);
            black_box(dist.calculate_start_values(black_box(&target), 8).unwrap())
        });
    });
    group.finish();
}

fn bench_mvn_nll(c: &mut Criterion) {
    let n = 10_000;
    let dist = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
    let (preds, _) = margins_and_targets(n, dist.n_params(), 0x00b5_e55e);
    let transformed = dist.transform_params(&preds.view());
    let mut state = 0x00b5_e55eu64;
    let mut y = Array2::zeros((n, 2));
    for i in 0..n {
        y[[i, 0]] = (lcg(&mut state) - 0.5) * 4.0;
        y[[i, 1]] = (lcg(&mut state) - 0.5) * 4.0;
    }

    c.bench_function("hot_paths/mvn_nll_10k", |b| {
        b.iter(|| {
            let yv = y.view();
            let target = ResponseData::Multivariate(&yv);
            black_box(dist.nll(black_box(&transformed.view()), &target))
        });
    });
}

fn bench_spline_flow_nll(c: &mut Criterion) {
    let n = 10_000;
    let dist = SplineFlow::new(
        TargetSupport::Real,
        4,
        1.0,
        SplineOrder::Quadratic,
        Stabilization::None,
        LossFn::Nll,
        false,
    );
    let (preds, y) = margins_and_targets(n, dist.n_params(), 0x51ee_7bee);
    let transformed = dist.transform_params(&preds.view());

    c.bench_function("hot_paths/spline_flow_nll_10k", |b| {
        b.iter(|| {
            let yv = y.view();
            let target = ResponseData::Univariate(&yv);
            black_box(dist.nll(black_box(&transformed.view()), &target))
        });
    });
}

/// Control: an analytical-gradient distribution the perf work does not touch.
/// This should stay flat across the A/B — movement here means machine noise.
fn bench_gaussian_control(c: &mut Criterion) {
    let n = 10_000;
    let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
    let (preds, y) = margins_and_targets(n, dist.n_params(), 0x5eed_5eed);

    c.bench_function("hot_paths/gaussian_gradients_10k_control", |b| {
        b.iter(|| {
            let yv = y.view();
            let target = ResponseData::Univariate(&yv);
            black_box(
                dist.compute_gradients_and_hessians(black_box(&preds.view()), &target, None)
                    .unwrap(),
            )
        });
    });
}

criterion_group!(
    benches,
    bench_mixture_gradients,
    bench_mixture_nll,
    bench_spline_flow_gradients,
    bench_crps_gradients,
    bench_start_values,
    bench_mvn_nll,
    bench_spline_flow_nll,
    bench_gaussian_control
);
criterion_main!(benches);
