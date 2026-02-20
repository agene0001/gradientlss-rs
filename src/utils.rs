//! Utility functions for parameter transformations.
//!
//! These response functions transform predicted values to the appropriate
//! parameter space for each distributional parameter.

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

/// Response function types for transforming distributional parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ResponseFn {
    /// Identity transformation (no change).
    Identity,
    /// Exponential transformation for strictly positive values.
    Exp,
    /// Exponential transformation for degrees of freedom (adds 2).
    ExpDf,
    /// Softplus transformation for strictly positive values.
    Softplus,
    /// Softplus transformation for degrees of freedom (adds 2).
    SoftplusDf,
    /// Squareplus transformation for strictly positive values.
    Squareplus,
    /// Squareplus transformation for degrees of freedom (adds 2).
    SquareplusDf,
    /// Sigmoid transformation for values in (0, 1).
    Sigmoid,
    /// ReLU transformation for non-negative values.
    Relu,
    /// ReLU transformation for degrees of freedom (adds 2).
    ReluDf,
}

impl ResponseFn {
    /// Apply the response function to an array of values.
    pub fn apply(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            ResponseFn::Identity => identity_fn(x),
            ResponseFn::Exp => exp_fn(x),
            ResponseFn::ExpDf => exp_fn_df(x),
            ResponseFn::Softplus => softplus_fn(x),
            ResponseFn::SoftplusDf => softplus_fn_df(x),
            ResponseFn::Squareplus => squareplus_fn(x),
            ResponseFn::SquareplusDf => squareplus_fn_df(x),
            ResponseFn::Sigmoid => sigmoid_fn(x),
            ResponseFn::Relu => relu_fn(x),
            ResponseFn::ReluDf => relu_fn_df(x),
        }
    }

    /// Apply the response function element-wise, writing into a pre-allocated output.
    /// Uses a fast path when no NaN/Inf values are present (the common case),
    /// avoiding the overhead of computing a NaN replacement mean.
    pub fn apply_into(&self, x: &ArrayView1<f64>, out: &mut ndarray::ArrayViewMut1<f64>) {
        // Fast check: are there any non-finite values?
        let has_non_finite = x.iter().any(|v| !v.is_finite());

        if has_non_finite {
            // Slow path: compute mean of finite values for NaN replacement
            let (sum, count) = x.iter().fold((0.0, 0usize), |(s, c), &v| {
                if v.is_finite() {
                    (s + v, c + 1)
                } else {
                    (s, c)
                }
            });
            let nan_replacement = if count == 0 { 0.0 } else { sum / count as f64 };

            // Write nan-cleaned values into output
            for (o, &v) in out.iter_mut().zip(x.iter()) {
                *o = if v.is_finite() { v } else { nan_replacement };
            }
        } else {
            // Fast path: direct copy, no NaN handling needed
            out.assign(x);
        }

        // Transform in-place — mapv_inplace allows auto-vectorization of the inner loop
        match self {
            ResponseFn::Identity => {} // already written
            ResponseFn::Exp => out.mapv_inplace(|v| v.exp() + EPSILON),
            ResponseFn::ExpDf => out.mapv_inplace(|v| v.exp() + EPSILON + 2.0),
            ResponseFn::Softplus => out.mapv_inplace(|v| softplus_scalar(v)),
            ResponseFn::SoftplusDf => out.mapv_inplace(|v| softplus_scalar(v) + 2.0),
            ResponseFn::Squareplus => out.mapv_inplace(|v| squareplus_scalar(v)),
            ResponseFn::SquareplusDf => out.mapv_inplace(|v| squareplus_scalar(v) + 2.0),
            ResponseFn::Sigmoid => out.mapv_inplace(|v| {
                let s = 1.0 / (1.0 + (-v).exp()) + EPSILON;
                s.clamp(SIGMOID_CLAMP_MIN, SIGMOID_CLAMP_MAX)
            }),
            ResponseFn::Relu => out.mapv_inplace(|v| v.max(0.0) + EPSILON),
            ResponseFn::ReluDf => out.mapv_inplace(|v| v.max(0.0) + EPSILON + 2.0),
        }
    }

    /// Apply the response function to a single value.
    pub fn apply_scalar(&self, x: f64) -> f64 {
        self.apply_scalar_with_nan_replacement(x, 0.0)
    }

    /// Apply the response function to a single value, using `nan_replacement`
    /// instead of 0.0 when the input is NaN/Inf. This matches Python's
    /// `nan_to_num(predt)` which uses `torch.nanmean(predt)` as the replacement.
    pub fn apply_scalar_with_nan_replacement(&self, x: f64, nan_replacement: f64) -> f64 {
        match self {
            ResponseFn::Identity => nan_to_num_scalar(x, nan_replacement),
            ResponseFn::Exp => nan_to_num_scalar(x, nan_replacement).exp() + EPSILON,
            ResponseFn::ExpDf => nan_to_num_scalar(x, nan_replacement).exp() + EPSILON + 2.0,
            ResponseFn::Softplus => {
                let v = nan_to_num_scalar(x, nan_replacement);
                if v > 20.0 {
                    v + EPSILON
                } else if v < -20.0 {
                    EPSILON
                } else {
                    (1.0 + v.exp()).ln() + EPSILON
                }
            }
            ResponseFn::SoftplusDf => {
                let v = nan_to_num_scalar(x, nan_replacement);
                let sp = if v > 20.0 {
                    v + EPSILON
                } else if v < -20.0 {
                    EPSILON
                } else {
                    (1.0 + v.exp()).ln() + EPSILON
                };
                sp + 2.0
            }
            ResponseFn::Squareplus => {
                let v = nan_to_num_scalar(x, nan_replacement);
                0.5 * (v + (v * v + 4.0).sqrt()) + EPSILON
            }
            ResponseFn::SquareplusDf => {
                let v = nan_to_num_scalar(x, nan_replacement);
                0.5 * (v + (v * v + 4.0).sqrt()) + EPSILON + 2.0
            }
            ResponseFn::Sigmoid => {
                let v = nan_to_num_scalar(x, nan_replacement);
                let s = 1.0 / (1.0 + (-v).exp()) + EPSILON;
                s.clamp(SIGMOID_CLAMP_MIN, SIGMOID_CLAMP_MAX)
            }
            ResponseFn::Relu => nan_to_num_scalar(x, nan_replacement).max(0.0) + EPSILON,
            ResponseFn::ReluDf => nan_to_num_scalar(x, nan_replacement).max(0.0) + EPSILON + 2.0,
        }
    }
}

const EPSILON: f64 = 1e-6;
const SIGMOID_CLAMP_MIN: f64 = 1e-3;
const SIGMOID_CLAMP_MAX: f64 = 1.0 - 1e-3;

/// Replace NaN and infinity values with the mean of the array.
pub fn nan_to_num(x: &ArrayView1<f64>) -> Array1<f64> {
    // Single-pass mean computation avoids intermediate Vec allocation
    let (sum, count) = x.iter().fold((0.0, 0usize), |(s, c), &v| {
        if v.is_finite() {
            (s + v, c + 1)
        } else {
            (s, c)
        }
    });
    let mean = if count == 0 { 0.0 } else { sum / count as f64 };

    x.mapv(|v| if v.is_finite() { v } else { mean })
}

/// Replace NaN and infinity values with a replacement value for a single scalar.
fn nan_to_num_scalar(x: f64, replacement: f64) -> f64 {
    if x.is_finite() { x } else { replacement }
}

// ============================================================================
// Array functions
// ============================================================================

/// Identity mapping.
pub fn identity_fn(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x)
}

/// Exponential function for strictly positive values.
pub fn exp_fn(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x).mapv(|v| v.exp() + EPSILON)
}

/// Exponential function for degrees of freedom (Student-T, etc.).
pub fn exp_fn_df(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x).mapv(|v| v.exp() + EPSILON + 2.0)
}

/// Softplus function for strictly positive values.
/// softplus(x) = ln(1 + exp(x))
pub fn softplus_fn(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x).mapv(|v| softplus_scalar(v))
}

/// Softplus function for degrees of freedom.
pub fn softplus_fn_df(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x).mapv(|v| softplus_scalar(v) + 2.0)
}

/// Squareplus function for strictly positive values.
/// squareplus(x) = 0.5 * (x + sqrt(x^2 + 4))
pub fn squareplus_fn(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x).mapv(|v| squareplus_scalar(v))
}

/// Squareplus function for degrees of freedom.
pub fn squareplus_fn_df(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x).mapv(|v| squareplus_scalar(v) + 2.0)
}

/// Sigmoid function for values in (0, 1).
pub fn sigmoid_fn(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x).mapv(|v| {
        let s = 1.0 / (1.0 + (-v).exp()) + EPSILON;
        s.clamp(SIGMOID_CLAMP_MIN, SIGMOID_CLAMP_MAX)
    })
}

/// ReLU function for non-negative values.
pub fn relu_fn(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x).mapv(|v| v.max(0.0) + EPSILON)
}

/// ReLU function for degrees of freedom.
pub fn relu_fn_df(x: &ArrayView1<f64>) -> Array1<f64> {
    nan_to_num(x).mapv(|v| v.max(0.0) + EPSILON + 2.0)
}

/// Softmax function to ensure values sum to 1.
/// Matches Python's torch.nn.functional.softmax behavior.
pub fn softmax_fn(x: &ArrayView1<f64>) -> Array1<f64> {
    let x = nan_to_num(x);
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Array1<f64> = x.mapv(|v| (v - max_val).exp());
    let sum_exp = exp_x.sum();
    if sum_exp > 0.0 {
        exp_x / sum_exp
    } else {
        Array1::from_elem(x.len(), 1.0 / x.len() as f64)
    }
}

/// Softmax function for 2D arrays (row-wise softmax).
/// Each row will sum to 1.
pub fn softmax_fn_2d(x: &ndarray::ArrayView2<f64>) -> ndarray::Array2<f64> {
    let mut result = ndarray::Array2::zeros(x.dim());
    for (i, row) in x.rows().into_iter().enumerate() {
        let softmax_row = softmax_fn(&row);
        result.row_mut(i).assign(&softmax_row);
    }
    result
}

/// Gumbel-Softmax function for differentiable categorical sampling.
///
/// The Gumbel-softmax distribution is a continuous distribution over the simplex,
/// which can be thought of as a "soft" version of a categorical distribution.
///
/// # Arguments
/// * `x` - Input logits
/// * `tau` - Temperature parameter. As tau -> 0, output becomes more discrete.
///           As tau -> inf, output becomes more uniform.
/// * `seed` - Random seed for Gumbel noise
///
/// # Reference
/// Jang, E., Gu, Shixiang and Poole, B. "Categorical Reparameterization with Gumbel-Softmax", ICLR, 2017.
pub fn gumbel_softmax_fn(x: &ArrayView1<f64>, tau: f64, seed: u64) -> Array1<f64> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let x = nan_to_num(x);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Sample Gumbel noise: g = -log(-log(u)) where u ~ Uniform(0, 1)
    let gumbel_noise: Array1<f64> = Array1::from_iter((0..x.len()).map(|_| {
        let u: f64 = rng.random_range(1e-10..1.0 - 1e-10);
        -(-u.ln()).ln()
    }));

    // Add Gumbel noise and apply temperature
    let logits_with_noise = (&x + &gumbel_noise) / tau;

    // Apply softmax
    softmax_fn(&logits_with_noise.view())
}

/// Gumbel-Softmax for 2D arrays (row-wise).
pub fn gumbel_softmax_fn_2d(
    x: &ndarray::ArrayView2<f64>,
    tau: f64,
    seed: u64,
) -> ndarray::Array2<f64> {
    let mut result = ndarray::Array2::zeros(x.dim());
    for (i, row) in x.rows().into_iter().enumerate() {
        // Use different seed for each row to ensure different noise
        let row_seed = seed.wrapping_add(i as u64);
        let gumbel_row = gumbel_softmax_fn(&row, tau, row_seed);
        result.row_mut(i).assign(&gumbel_row);
    }
    result
}

// ============================================================================
// Scalar functions (used by array functions that call nan_to_num separately)
// ============================================================================

fn softplus_scalar(x: f64) -> f64 {
    // Numerically stable softplus (assumes input already nan-cleaned)
    if x > 20.0 {
        x + EPSILON
    } else if x < -20.0 {
        EPSILON
    } else {
        (1.0 + x.exp()).ln() + EPSILON
    }
}

fn squareplus_scalar(x: f64) -> f64 {
    // Assumes input already nan-cleaned
    0.5 * (x + (x * x + 4.0).sqrt()) + EPSILON
}

// ============================================================================
// Derivative functions for gradient computation
// ============================================================================

impl ResponseFn {
    /// Compute the derivative of the response function at x.
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ResponseFn::Identity => 1.0,
            ResponseFn::Exp | ResponseFn::ExpDf => x.exp(),
            ResponseFn::Softplus | ResponseFn::SoftplusDf => {
                // d/dx softplus(x) = sigmoid(x)
                1.0 / (1.0 + (-x).exp())
            }
            ResponseFn::Squareplus | ResponseFn::SquareplusDf => {
                // d/dx squareplus(x) = 0.5 * (1 + x / sqrt(x^2 + 4))
                0.5 * (1.0 + x / (x * x + 4.0).sqrt())
            }
            ResponseFn::Sigmoid => {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ResponseFn::Relu | ResponseFn::ReluDf => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute the second derivative of the response function at x.
    /// Needed for exact Hessian computation via chain rule:
    /// d²L/dpred² = (d²L/dθ²)*(dθ/dpred)² + (dL/dθ)*(d²θ/dpred²)
    pub fn second_derivative(&self, x: f64) -> f64 {
        match self {
            ResponseFn::Identity => 0.0,
            ResponseFn::Exp | ResponseFn::ExpDf => x.exp(),
            ResponseFn::Softplus | ResponseFn::SoftplusDf => {
                // d²/dx² softplus(x) = d/dx sigmoid(x) = sigmoid(x)*(1-sigmoid(x))
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ResponseFn::Squareplus | ResponseFn::SquareplusDf => {
                // d²/dx² squareplus(x) = 2 / (x² + 4)^(3/2)
                let denom = (x * x + 4.0).powf(1.5);
                2.0 / denom
            }
            ResponseFn::Sigmoid => {
                // d²/dx² sigmoid(x) = sigmoid(x)*(1-sigmoid(x))*(1-2*sigmoid(x))
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s) * (1.0 - 2.0 * s)
            }
            ResponseFn::Relu | ResponseFn::ReluDf => 0.0,
        }
    }

    /// Compute the derivative for an entire column of predictions.
    /// Uses mapv which LLVM can auto-vectorize for simple arithmetic.
    pub fn derivative_batch(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            ResponseFn::Identity => Array1::ones(x.len()),
            ResponseFn::Exp | ResponseFn::ExpDf => x.mapv(|v| v.exp()),
            ResponseFn::Softplus | ResponseFn::SoftplusDf => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ResponseFn::Squareplus | ResponseFn::SquareplusDf => {
                x.mapv(|v| 0.5 * (1.0 + v / (v * v + 4.0).sqrt()))
            }
            ResponseFn::Sigmoid => x.mapv(|v| {
                let s = 1.0 / (1.0 + (-v).exp());
                s * (1.0 - s)
            }),
            ResponseFn::Relu | ResponseFn::ReluDf => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
        }
    }

    /// Compute the second derivative for an entire column of predictions.
    pub fn second_derivative_batch(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            ResponseFn::Identity => Array1::zeros(x.len()),
            ResponseFn::Exp | ResponseFn::ExpDf => x.mapv(|v| v.exp()),
            ResponseFn::Softplus | ResponseFn::SoftplusDf => x.mapv(|v| {
                let s = 1.0 / (1.0 + (-v).exp());
                s * (1.0 - s)
            }),
            ResponseFn::Squareplus | ResponseFn::SquareplusDf => {
                x.mapv(|v| 2.0 / (v * v + 4.0).powf(1.5))
            }
            ResponseFn::Sigmoid => x.mapv(|v| {
                let s = 1.0 / (1.0 + (-v).exp());
                s * (1.0 - s) * (1.0 - 2.0 * s)
            }),
            ResponseFn::Relu | ResponseFn::ReluDf => Array1::zeros(x.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_identity_fn() {
        let x = array![1.0, 2.0, 3.0];
        let result = identity_fn(&x.view());
        assert_eq!(result, array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_exp_fn() {
        let x = array![0.0, 1.0];
        let result = exp_fn(&x.view());
        assert_relative_eq!(result[0], 1.0 + EPSILON, epsilon = 1e-10);
        assert_relative_eq!(result[1], std::f64::consts::E + EPSILON, epsilon = 1e-10);
    }

    #[test]
    fn test_softplus_fn() {
        let x = array![0.0];
        let result = softplus_fn(&x.view());
        // softplus(0) = ln(2) ≈ 0.693
        assert_relative_eq!(result[0], 2.0_f64.ln() + EPSILON, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_fn() {
        let x = array![0.0];
        let result = sigmoid_fn(&x.view());
        // sigmoid(0) = 0.5
        assert_relative_eq!(result[0], 0.5 + EPSILON, epsilon = 1e-6);
    }

    #[test]
    fn test_nan_handling() {
        let x = array![1.0, f64::NAN, 3.0];
        let result = nan_to_num(&x.view());
        // Mean of valid values is 2.0
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_response_fn_apply() {
        let x = array![0.0, 1.0, 2.0];
        let result = ResponseFn::Exp.apply(&x.view());
        assert!(result.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_softmax_fn() {
        let x = array![1.0, 2.0, 3.0];
        let result = softmax_fn(&x.view());

        // Check that values sum to 1
        assert_relative_eq!(result.sum(), 1.0, epsilon = 1e-10);

        // Check that all values are positive
        assert!(result.iter().all(|&v| v > 0.0));

        // Check that larger inputs give larger outputs
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_softmax_fn_equal_inputs() {
        let x = array![1.0, 1.0, 1.0];
        let result = softmax_fn(&x.view());

        // All outputs should be equal (1/3)
        assert_relative_eq!(result[0], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gumbel_softmax_fn() {
        let x = array![1.0, 2.0, 3.0];
        let result = gumbel_softmax_fn(&x.view(), 1.0, 123);

        // Check that values sum to 1
        assert_relative_eq!(result.sum(), 1.0, epsilon = 1e-10);

        // Check that all values are positive
        assert!(result.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_gumbel_softmax_temperature() {
        let x = array![1.0, 5.0, 1.0];

        // Low temperature should give more peaked distribution
        let result_low_temp = gumbel_softmax_fn(&x.view(), 0.1, 42);
        // High temperature should give more uniform distribution
        let result_high_temp = gumbel_softmax_fn(&x.view(), 10.0, 42);

        // Both should sum to 1
        assert_relative_eq!(result_low_temp.sum(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result_high_temp.sum(), 1.0, epsilon = 1e-10);

        // Low temp should have higher max value (more peaked)
        let max_low = result_low_temp.iter().cloned().fold(0.0, f64::max);
        let max_high = result_high_temp.iter().cloned().fold(0.0, f64::max);
        assert!(max_low > max_high);
    }

    #[test]
    fn test_derivative_batch_matches_scalar() {
        let x = array![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let variants = [
            ResponseFn::Identity,
            ResponseFn::Exp,
            ResponseFn::ExpDf,
            ResponseFn::Softplus,
            ResponseFn::SoftplusDf,
            ResponseFn::Squareplus,
            ResponseFn::SquareplusDf,
            ResponseFn::Sigmoid,
            ResponseFn::Relu,
            ResponseFn::ReluDf,
        ];
        for rf in &variants {
            let batch = rf.derivative_batch(&x.view());
            for (i, &v) in x.iter().enumerate() {
                assert_relative_eq!(batch[i], rf.derivative(v), epsilon = 1e-12,);
            }
        }
    }

    #[test]
    fn test_second_derivative_batch_matches_scalar() {
        let x = array![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let variants = [
            ResponseFn::Identity,
            ResponseFn::Exp,
            ResponseFn::ExpDf,
            ResponseFn::Softplus,
            ResponseFn::SoftplusDf,
            ResponseFn::Squareplus,
            ResponseFn::SquareplusDf,
            ResponseFn::Sigmoid,
            ResponseFn::Relu,
            ResponseFn::ReluDf,
        ];
        for rf in &variants {
            let batch = rf.second_derivative_batch(&x.view());
            for (i, &v) in x.iter().enumerate() {
                assert_relative_eq!(batch[i], rf.second_derivative(v), epsilon = 1e-12,);
            }
        }
    }
}
