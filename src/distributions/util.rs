//! Shared helpers for distribution implementations.

use rayon::prelude::*;

/// Sample count at or above which per-sample reductions are parallelized.
///
/// Mirrors the threshold used by the analytical-gradient paths (e.g.
/// `NegativeBinomial::analytical_gradients`) so the metric and gradient passes
/// make the same sequential/parallel choice for a given batch size.
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
