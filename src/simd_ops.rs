//! SIMD-optimized operations for response functions.
//!
//! This module provides SIMD-accelerated versions of common mathematical
//! operations used in distributional regression. Uses the `wide` crate
//! for portable SIMD on stable Rust.
//!
//! The functions process data in chunks of 4 f64 values (256-bit SIMD)
//! with scalar fallback for remainders.

use wide::{CmpGt, CmpLt, CmpNe, f64x4};

/// Small epsilon to ensure numerical stability
const EPSILON: f64 = 1e-6;

// Sigmoid clamping constants
const SIGMOID_MIN: f64 = 1e-3;
const SIGMOID_MAX: f64 = 1.0 - 1e-3;

/// SIMD exponential using scalar exp for accuracy.
/// While we lose some SIMD benefit for exp, we maintain accuracy.
/// The SIMD benefit comes from parallel loading/storing and other operations.
#[inline(always)]
fn exp_x4(x: f64x4) -> f64x4 {
    let arr: [f64; 4] = x.into();
    f64x4::from([arr[0].exp(), arr[1].exp(), arr[2].exp(), arr[3].exp()])
}

/// SIMD-optimized exponential function for strictly positive values.
/// Computes exp(x) + epsilon for each element.
#[inline]
pub fn exp_simd(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());

    let epsilon_x4 = f64x4::splat(EPSILON);
    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let x = f64x4::from([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);
        let result = exp_x4(x) + epsilon_x4;
        let arr: [f64; 4] = result.into();
        output[idx..idx + 4].copy_from_slice(&arr);
    }

    // Handle remainder with scalar operations
    let start = chunks * 4;
    for i in 0..remainder {
        let x = input[start + i];
        output[start + i] = x.exp() + EPSILON;
    }
}

/// SIMD-optimized sigmoid function.
/// Computes 1 / (1 + exp(-x)) with clamping.
#[inline]
pub fn sigmoid_simd(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());

    let zero = f64x4::splat(0.0);
    let one = f64x4::splat(1.0);
    let epsilon_x4 = f64x4::splat(EPSILON);
    let sig_min = f64x4::splat(SIGMOID_MIN);
    let sig_max = f64x4::splat(SIGMOID_MAX);

    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let x = f64x4::from([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);

        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_x = zero - x;
        let exp_neg_x = exp_x4(neg_x);
        let result = one / (one + exp_neg_x) + epsilon_x4;

        // Clamp to [SIGMOID_MIN, SIGMOID_MAX]
        let result = result.max(sig_min).min(sig_max);

        let arr: [f64; 4] = result.into();
        output[idx..idx + 4].copy_from_slice(&arr);
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let x = input[start + i];
        let s = 1.0 / (1.0 + (-x).exp()) + EPSILON;
        output[start + i] = s.clamp(SIGMOID_MIN, SIGMOID_MAX);
    }
}

/// SIMD-optimized softplus function.
/// Computes ln(1 + exp(x)) with numerical stability.
#[inline]
pub fn softplus_simd(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());

    let one = f64x4::splat(1.0);
    let twenty = f64x4::splat(20.0);
    let neg_twenty = f64x4::splat(-20.0);
    let epsilon_x4 = f64x4::splat(EPSILON);

    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let x = f64x4::from([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);

        // For numerical stability:
        // - If x > 20: softplus(x) ≈ x
        // - If x < -20: softplus(x) ≈ exp(x) ≈ 0
        // - Otherwise: ln(1 + exp(x))

        let high_mask = x.cmp_gt(twenty);
        let low_mask = x.cmp_lt(neg_twenty);

        // Standard computation for middle range
        let exp_x = exp_x4(x);
        // Use log1p approximation: ln(1+y) ≈ y - y²/2 + y³/3 for small y
        // For larger values, use the standard formula
        let one_plus_exp = one + exp_x;

        // Compute ln using scalar operations (wide doesn't have ln)
        let arr: [f64; 4] = one_plus_exp.into();
        let ln_arr = [arr[0].ln(), arr[1].ln(), arr[2].ln(), arr[3].ln()];
        let standard = f64x4::from(ln_arr);

        // Blend results based on masks
        // high: use x + epsilon, low: use epsilon, middle: use standard + epsilon
        let zero = f64x4::splat(0.0);
        let result = high_mask.blend(x, low_mask.blend(zero, standard));
        let result = result + epsilon_x4;

        let arr: [f64; 4] = result.into();
        output[idx..idx + 4].copy_from_slice(&arr);
    }

    // Handle remainder with accurate scalar implementation
    let start = chunks * 4;
    for i in 0..remainder {
        let x = input[start + i];
        output[start + i] = if x > 20.0 {
            x + EPSILON
        } else if x < -20.0 {
            EPSILON
        } else {
            (1.0 + x.exp()).ln() + EPSILON
        };
    }
}

/// SIMD-optimized squareplus function.
/// Computes 0.5 * (x + sqrt(x² + 4)).
#[inline]
pub fn squareplus_simd(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());

    let half = f64x4::splat(0.5);
    let four = f64x4::splat(4.0);
    let epsilon_x4 = f64x4::splat(EPSILON);

    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let x = f64x4::from([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);

        // squareplus(x) = 0.5 * (x + sqrt(x² + 4))
        let x_sq = x * x;
        let result = half * (x + (x_sq + four).sqrt()) + epsilon_x4;

        let arr: [f64; 4] = result.into();
        output[idx..idx + 4].copy_from_slice(&arr);
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let x = input[start + i];
        output[start + i] = 0.5 * (x + (x * x + 4.0).sqrt()) + EPSILON;
    }
}

/// SIMD-optimized ReLU function.
/// Computes max(x, 0) + epsilon.
#[inline]
pub fn relu_simd(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());

    let zero = f64x4::splat(0.0);
    let epsilon_x4 = f64x4::splat(EPSILON);

    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let x = f64x4::from([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);
        let result = x.max(zero) + epsilon_x4;

        let arr: [f64; 4] = result.into();
        output[idx..idx + 4].copy_from_slice(&arr);
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        output[start + i] = input[start + i].max(0.0) + EPSILON;
    }
}

/// SIMD-optimized identity function with NaN handling.
/// Replaces NaN/Inf with the provided replacement value.
#[inline]
pub fn identity_simd(input: &[f64], output: &mut [f64], nan_replacement: f64) {
    debug_assert_eq!(input.len(), output.len());

    let replacement = f64x4::splat(nan_replacement);
    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let x = f64x4::from([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);

        // Check for NaN: a value is NaN if it doesn't equal itself
        let is_nan = x.cmp_ne(x);
        // Check for +Inf and -Inf: values beyond f64::MAX magnitude
        let max_val = f64x4::splat(f64::MAX);
        let neg_max_val = f64x4::splat(-f64::MAX);
        let is_pos_inf = x.cmp_gt(max_val);
        let is_neg_inf = x.cmp_lt(neg_max_val);
        // Combine: is_nan | is_pos_inf | is_neg_inf
        // wide doesn't have bitwise OR for masks, so blend sequentially
        let result = is_nan.blend(replacement, x);
        let result = is_pos_inf.blend(replacement, result);
        let result = is_neg_inf.blend(replacement, result);

        let arr: [f64; 4] = result.into();
        output[idx..idx + 4].copy_from_slice(&arr);
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let x = input[start + i];
        output[start + i] = if x.is_finite() { x } else { nan_replacement };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_exp_simd() {
        let input = [0.0, 1.0, 2.0, -1.0, 0.5, -0.5, 3.0, -3.0];
        let mut output = [0.0; 8];

        exp_simd(&input, &mut output);

        for (i, &x) in input.iter().enumerate() {
            let expected = x.exp() + EPSILON;
            assert_relative_eq!(output[i], expected, epsilon = 0.01);
        }
    }

    #[test]
    fn test_sigmoid_simd() {
        let input = [0.0, 1.0, -1.0, 5.0, -5.0, 0.5, -0.5, 2.0];
        let mut output = [0.0; 8];

        sigmoid_simd(&input, &mut output);

        for (i, &x) in input.iter().enumerate() {
            let expected = (1.0 / (1.0 + (-x).exp()) + EPSILON).clamp(SIGMOID_MIN, SIGMOID_MAX);
            assert_relative_eq!(output[i], expected, epsilon = 0.01);
        }
    }

    #[test]
    fn test_softplus_simd() {
        let input = [0.0, 1.0, -1.0, 5.0, -5.0, 25.0, -25.0, 2.0];
        let mut output = [0.0; 8];

        softplus_simd(&input, &mut output);

        // Check that all outputs are positive
        for &v in &output {
            assert!(v > 0.0);
        }

        // Check specific values
        assert_relative_eq!(output[0], (1.0_f64 + 1.0).ln() + EPSILON, epsilon = 0.1);
    }

    #[test]
    fn test_squareplus_simd() {
        let input = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let mut output = [0.0; 8];

        squareplus_simd(&input, &mut output);

        for (i, &x) in input.iter().enumerate() {
            let expected = 0.5 * (x + (x * x + 4.0).sqrt()) + EPSILON;
            assert_relative_eq!(output[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_relu_simd() {
        let input = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let mut output = [0.0; 8];

        relu_simd(&input, &mut output);

        for (i, &x) in input.iter().enumerate() {
            let expected = x.max(0.0) + EPSILON;
            assert_relative_eq!(output[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_remainder_handling() {
        // Test with non-multiple-of-4 length
        let input = [0.0, 1.0, 2.0, 3.0, 4.0]; // 5 elements
        let mut output = [0.0; 5];

        exp_simd(&input, &mut output);

        for (i, &x) in input.iter().enumerate() {
            let expected = x.exp() + EPSILON;
            assert_relative_eq!(output[i], expected, epsilon = 0.01);
        }
    }

    #[test]
    fn test_identity_simd_nan() {
        let input = [
            1.0,
            f64::NAN,
            3.0,
            f64::INFINITY,
            5.0,
            f64::NEG_INFINITY,
            7.0,
            8.0,
        ];
        let mut output = [0.0; 8];

        identity_simd(&input, &mut output, 999.0);

        assert_relative_eq!(output[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(output[1], 999.0, epsilon = 1e-10); // NaN replaced
        assert_relative_eq!(output[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(output[3], 999.0, epsilon = 1e-10); // +Infinity replaced
        assert_relative_eq!(output[4], 5.0, epsilon = 1e-10);
        assert_relative_eq!(output[5], 999.0, epsilon = 1e-10); // -Infinity replaced
        assert_relative_eq!(output[6], 7.0, epsilon = 1e-10);
        assert_relative_eq!(output[7], 8.0, epsilon = 1e-10);
    }
}
