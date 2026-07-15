//! Shared helpers for distribution implementations.

use rayon::prelude::*;

/// Sample count at or above which per-sample reductions are parallelized.
///
/// Higher than the analytical-gradient paths' threshold (256): a summed NLL
/// term is a handful of libm calls per sample, so rayon's fork/join overhead
/// takes longer to amortize than in the heavier gradient passes.
pub(crate) const PAR_THRESHOLD: usize = 4096;

/// Sum `f(i)` over `0..n`, parallelizing with rayon at or above [`PAR_THRESHOLD`].
///
/// Used by the univariate `nll` metrics, which would otherwise be a sequential
/// per-sample loop of (often `ln_gamma`-heavy) `log_prob` evaluations run every
/// boosting round on both the train and validation sets.
///
/// `f` must be pure: it is invoked in arbitrary order and across threads. The
/// parallel reduction reorders the floating-point adds, so the result can differ
/// from the sequential sum in the last ~1e-12 — irrelevant for loss tracking.
#[inline]
pub(crate) fn par_sum<F>(n: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Send + Sync,
{
    if n >= PAR_THRESHOLD {
        (0..n).into_par_iter().map(f).sum()
    } else {
        (0..n).map(f).sum()
    }
}

/// Like [`par_sum`], but NaN terms contribute 0 — `torch.nansum` semantics
/// (±inf still propagates). Python's `get_params_loss` aggregates the NLL with
/// `-torch.nansum(dist.log_prob(target))`, so a single NaN log-prob (e.g. a
/// 0·inf at a support boundary) must NOT poison the whole eval metric: a NaN
/// metric never compares better than `best_loss`, which silently freezes
/// best_iteration and burns the early-stopping patience where Python trains on.
#[inline]
pub(crate) fn par_nansum<F>(n: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Send + Sync,
{
    par_sum(n, move |i| {
        let v = f(i);
        if v.is_nan() { 0.0 } else { v }
    })
}
