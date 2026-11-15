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
/// `f` must be pure: it is invoked in arbitrary order and across threads.
///
/// Deterministic: the range is cut into fixed [`CHUNK`]-sized blocks, each
/// block is summed sequentially, and the block sums are added in index order
/// — so the result does not depend on the thread count or scheduling. (The
/// start-value optimiser runs on a likelihood that is flat in some directions;
/// with a scheduling-dependent sum its endpoint followed the rounding noise and
/// `calculate_start_values` was not reproducible run to run.) The association
/// still differs from a plain sequential sum by ~1e-12 relative.
#[inline]
pub(crate) fn par_sum<F>(n: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Send + Sync,
{
    if n >= PAR_THRESHOLD {
        chunked_sum(n, CHUNK, f)
    } else {
        (0..n).map(f).sum()
    }
}

/// Block size of the deterministic parallel reductions.
pub(crate) const CHUNK: usize = 1024;

/// Deterministic parallel sum: per-block sequential sums (in parallel over
/// blocks), then an in-order sum of the block results.
#[inline]
fn chunked_sum<F>(n: usize, chunk: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Send + Sync,
{
    let n_blocks = n.div_ceil(chunk);
    let block_sums: Vec<f64> = (0..n_blocks)
        .into_par_iter()
        .map(|b| {
            let start = b * chunk;
            let end = (start + chunk).min(n);
            (start..end).map(&f).sum::<f64>()
        })
        .collect();
    block_sums.iter().sum()
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

/// Row threshold for parallelizing HEAVYWEIGHT per-row reductions — per-row
/// matrix factorizations (multivariate log_probs) or spline evaluations.
/// Much lower than [`PAR_THRESHOLD`]: each row does enough work to amortize
/// rayon's fork/join far sooner than a scalar-libm NLL term does.
pub(crate) const PAR_ROW_THRESHOLD: usize = 256;

/// [`par_nansum`] with the heavyweight-row threshold ([`PAR_ROW_THRESHOLD`]).
#[inline]
pub(crate) fn par_nansum_rows<F>(n: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Send + Sync,
{
    let nan_to_zero = move |i: usize| {
        let v = f(i);
        if v.is_nan() { 0.0 } else { v }
    };
    if n >= PAR_ROW_THRESHOLD {
        // heavyweight rows: smaller blocks so a few hundred rows still spread
        chunked_sum(n, 32, nan_to_zero)
    } else {
        (0..n).map(nan_to_zero).sum()
    }
}

#[cfg(test)]
mod determinism_tests {
    use super::*;

    /// The parallel sums are bit-identical across thread pools of different
    /// sizes (and across runs), and match a chunk-ordered sequential sum.
    #[test]
    fn par_sums_are_thread_count_invariant() {
        let n = 50_000;
        let f = |i: usize| (i as f64 * 0.37).sin() * 1e3 * if i % 977 == 0 { f64::NAN } else { 1.0 };
        let g = |i: usize| (i as f64 * 1.0001).sqrt() - (i as f64).sqrt(); // tiny terms, cancellation-prone
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| (par_sum(n, g), par_nansum(n, f), par_nansum_rows(n, f)))
        };
        let (a1, b1, c1) = run(1);
        let (a4, b4, c4) = run(4);
        let (a7, b7, c7) = run(7);
        assert_eq!(a1.to_bits(), a4.to_bits());
        assert_eq!(a1.to_bits(), a7.to_bits());
        assert_eq!(b1.to_bits(), b4.to_bits());
        assert_eq!(b1.to_bits(), b7.to_bits());
        assert_eq!(c1.to_bits(), c4.to_bits());
        assert_eq!(c1.to_bits(), c7.to_bits());
        // and the value is right: chunk-ordered sequential reference
        let reference: f64 = (0..n.div_ceil(CHUNK))
            .map(|b| ((b * CHUNK)..((b + 1) * CHUNK).min(n)).map(g).sum::<f64>())
            .sum();
        assert_eq!(a1.to_bits(), reference.to_bits());
        let seq_nan: f64 = (0..n).map(|i| { let v = f(i); if v.is_nan() { 0.0 } else { v } }).sum();
        assert!((b1 - seq_nan).abs() <= 1e-9 * seq_nan.abs().max(1.0));
    }
}

/// Longest table one row of a count-family CRPS sum may use (Poisson, NB).
pub(crate) const CRPS_MAX_TERMS: usize = 200_000;

/// One warning per process when a row's CRPS sum would need more than
/// [`CRPS_MAX_TERMS`] terms and is cut to that many around the mean.
pub(crate) fn warn_crps_truncated(family: &str, mean: f64) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "gradientlss: {family} CRPS sum truncated at {CRPS_MAX_TERMS} terms (a row has mean {mean:.3e}); \
             its loss and gradients are approximate. Train on the NLL for counts this large."
        );
    });
}
