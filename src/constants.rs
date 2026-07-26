//! Pre-computed mathematical constants for performance optimization.
//!
//! These constants avoid repeated computation of values like ln(2π) which
//! are used frequently in probability density calculations.

/// ln(2π) ≈ 1.8378770664093453
pub const LOG_2PI: f64 = 1.8378770664093454;

/// 0.5 * ln(2π) ≈ 0.9189385332046727
pub const HALF_LOG_2PI: f64 = 0.9189385332046727;

/// ln(π) ≈ 1.1447298858494002
pub const LOG_PI: f64 = 1.1447298858494002;

/// 1 / √(2π) ≈ 0.3989422804014327
pub const INV_SQRT_2PI: f64 = 0.3989422804014327;

/// √(2π) ≈ 2.5066282746310002
pub const SQRT_2PI: f64 = 2.5066282746310002;

/// ln(√(2π)) = 0.5 * ln(2π) ≈ 0.9189385332046727
pub const LOG_SQRT_2PI: f64 = 0.9189385332046727;

/// 1 / π ≈ 0.3183098861837907 (aliases the std constant for full precision)
pub const INV_PI: f64 = std::f64::consts::FRAC_1_PI;

/// Epsilon used by the zero-adjusted (ZA*) distributions to clamp zero targets
/// before evaluating the continuous base density.
///
/// Python's `ZeroInflatedDistribution.log_prob` clamps with
/// `torch.finfo(value.dtype).eps`, and targets arrive as float32 from
/// xgboost/lightgbm — i.e. ~1.1920929e-7, NOT f64 machine epsilon. Using
/// `f64::EPSILON` (2.2e-16) shifted the clamped log-density by many nats for
/// e.g. Gamma with concentration < 1, diverging from the Python fits.
pub(crate) const ZERO_CLAMP_EPS: f64 = f32::EPSILON as f64;

/// Trigamma function: ψ₁(x) = d²/dx² ln(Γ(x)) = d/dx ψ(x)
///
/// Uses the recurrence ψ₁(x) = ψ₁(x+1) + 1/x² to shift x >= 6,
/// then applies the asymptotic series expansion.
pub fn trigamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    // Use recurrence to shift x to a region where the series converges well
    let mut result = 0.0;
    while x < 8.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    // Asymptotic expansion for large x using Horner's method:
    // ψ₁(x) = 1/x + 1/(2x²) * (1 + 1/(3x) * (1 - 1/(5x²) * (1 - 5/(7x²) + 5/(3x⁴))))
    // Rewritten as polynomial in w = 1/x²:
    // ψ₁(x) = 1/x + w/2 + w/(6x) - w²/(30x) + w³/(42x) - w⁴/(30x)
    // Factor as: 1/x + w * (1/2 + 1/x * (1/6 + w * (-1/30 + w * (1/42 - w/30))))
    let inv_x = 1.0 / x;
    let w = inv_x * inv_x;
    result +=
        inv_x + w * (0.5 + inv_x * (1.0 / 6.0 + w * (-1.0 / 30.0 + w * (1.0 / 42.0 - w / 30.0))));

    result
}

/// ln(k!) = ln_gamma(k+1) for integer k, served from a lazily-built table.
///
/// Every discrete distribution's log-pmf pays a `ln_gamma(k + 1)` per sample
/// per NLL evaluation, but the target `k` never changes across boosting
/// rounds and real-world counts are small integers. The table entries are
/// computed with `ln_gamma` itself, so a hit is bit-identical to the direct
/// call — this is a cache, not an approximation. Non-integer, negative, or
/// large `k` fall back to `None` (caller computes `ln_gamma(k + 1.0)`).
pub(crate) fn ln_factorial(k: f64) -> Option<f64> {
    const TABLE_SIZE: usize = 256;
    static TABLE: std::sync::OnceLock<[f64; TABLE_SIZE]> = std::sync::OnceLock::new();

    if k >= 0.0 && k < TABLE_SIZE as f64 && k == k.trunc() {
        let table = TABLE.get_or_init(|| {
            let mut t = [0.0; TABLE_SIZE];
            for (i, v) in t.iter_mut().enumerate() {
                *v = statrs::function::gamma::ln_gamma(i as f64 + 1.0);
            }
            t
        });
        Some(table[k as usize])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_trigamma() {
        // ψ₁(1) = π²/6
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!((trigamma(1.0) - expected).abs() < 1e-10);

        // ψ₁(2) = π²/6 - 1
        assert!((trigamma(2.0) - (expected - 1.0)).abs() < 1e-10);

        // ψ₁(0.5) = π²/2
        let expected_half = std::f64::consts::PI * std::f64::consts::PI / 2.0;
        assert!((trigamma(0.5) - expected_half).abs() < 1e-8);
    }

    #[test]
    fn test_ln_factorial_matches_ln_gamma() {
        use statrs::function::gamma::ln_gamma;
        for k in [0.0, 1.0, 2.0, 5.0, 64.0, 255.0] {
            assert_eq!(ln_factorial(k), Some(ln_gamma(k + 1.0)));
        }
        // Outside the table or non-integer → caller falls back to ln_gamma.
        assert_eq!(ln_factorial(256.0), None);
        assert_eq!(ln_factorial(2.5), None);
        assert_eq!(ln_factorial(-1.0), None);
    }

    #[test]
    fn test_constants_accuracy() {
        let computed_log_2pi = (2.0 * PI).ln();
        assert!((LOG_2PI - computed_log_2pi).abs() < 1e-14);

        let computed_half_log_2pi = 0.5 * (2.0 * PI).ln();
        assert!((HALF_LOG_2PI - computed_half_log_2pi).abs() < 1e-14);

        let computed_log_pi = PI.ln();
        assert!((LOG_PI - computed_log_pi).abs() < 1e-14);

        let computed_inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
        assert!((INV_SQRT_2PI - computed_inv_sqrt_2pi).abs() < 1e-14);

        let computed_sqrt_2pi = (2.0 * PI).sqrt();
        assert!((SQRT_2PI - computed_sqrt_2pi).abs() < 1e-14);
    }
}
