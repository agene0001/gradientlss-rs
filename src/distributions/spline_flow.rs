//! Spline Flow distribution implementation.
//!
//! A normalizing flow based on element-wise rational spline bijections of linear and quadratic
//! order (Durkan et al., 2019; Dolatabadi et al., 2020). Rational splines are functions comprised
//! of segments that are the ratio of two polynomials, offering excellent functional flexibility
//! whilst maintaining a numerically stable inverse.
//!
//! References:
//! - Durkan, C., Bekasov, A., Murray, I. and Papamakarios, G. Neural Spline Flows. NeurIPS 2019.
//! - Dolatabadi, H. M., Erfani, S. and Leckie, C., Invertible Generative Modeling using Linear
//!   Rational Splines. AISTATS 2020.

use super::base::{Distribution, DistributionParam, LossFn, Stabilization};
use crate::constants::LOG_2PI;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Target support options for the spline flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetSupport {
    /// Real line: (-∞, +∞)
    Real,
    /// Positive reals: [0, +∞)
    Positive,
    /// Positive integers: {0, 1, 2, 3, ...}
    PositiveInteger,
    /// Unit interval: [0, 1]
    UnitInterval,
}

impl TargetSupport {
    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "real" => Some(TargetSupport::Real),
            "positive" => Some(TargetSupport::Positive),
            "positive_integer" => Some(TargetSupport::PositiveInteger),
            "unit_interval" => Some(TargetSupport::UnitInterval),
            _ => None,
        }
    }

    /// Whether this support implies discrete values.
    pub fn is_discrete(&self) -> bool {
        matches!(self, TargetSupport::PositiveInteger)
    }
}

/// Spline order options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplineOrder {
    /// Linear rational spline (Dolatabadi et al., 2020).
    Linear,
    /// Quadratic rational spline (Durkan et al., 2019).
    Quadratic,
}

impl SplineOrder {
    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "linear" => Some(SplineOrder::Linear),
            "quadratic" => Some(SplineOrder::Quadratic),
            _ => None,
        }
    }

    /// Calculate number of parameters for this spline order.
    pub fn n_params(&self, count_bins: usize) -> usize {
        match self {
            // Quadratic: widths (K) + heights (K) + derivatives (K-1)
            SplineOrder::Quadratic => 2 * count_bins + (count_bins - 1),
            // Linear: widths (K) + heights (K) + derivatives (K-1) + lambdas (K)
            SplineOrder::Linear => 3 * count_bins + (count_bins - 1),
        }
    }
}

/// Spline Flow distribution.
///
/// A normalizing flow based on rational spline bijections that transforms a standard
/// normal base distribution through a learned piecewise spline function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplineFlow {
    /// Distribution parameters (spline knots and derivatives).
    params: Vec<DistributionParam>,
    /// Number of spline bins/segments.
    count_bins: usize,
    /// Bounding box size [-bound, bound].
    bound: f64,
    /// Spline order (linear or quadratic).
    order: SplineOrder,
    /// Target support transformation.
    target_support: TargetSupport,
    /// Stabilization method.
    stabilization: Stabilization,
    /// Loss function.
    loss_fn: LossFn,
    /// Whether to initialize parameters.
    initialize: bool,
}

impl SplineFlow {
    /// Create a new SplineFlow distribution.
    ///
    /// # Arguments
    /// * `target_support` - The target support (real, positive, positive_integer, unit_interval)
    /// * `count_bins` - Number of segments in the spline (default: 8)
    /// * `bound` - Bounding box size K, defining [-K, K] x [-K, K] (default: 3.0)
    /// * `order` - Spline order: linear or quadratic (default: linear)
    /// * `stabilization` - Stabilization method for gradients
    /// * `loss_fn` - Loss function (NLL or CRPS)
    /// * `initialize` - Whether to initialize with start values
    pub fn new(
        target_support: TargetSupport,
        count_bins: usize,
        bound: f64,
        order: SplineOrder,
        stabilization: Stabilization,
        loss_fn: LossFn,
        initialize: bool,
    ) -> Self {
        let n_params = order.n_params(count_bins);

        // All spline parameters use identity response function
        // (the spline transform handles the nonlinearity)
        let params: Vec<DistributionParam> = (0..n_params)
            .map(|i| DistributionParam::new(format!("param_{}", i + 1), ResponseFn::Identity))
            .collect();

        Self {
            params,
            count_bins,
            bound,
            order,
            target_support,
            stabilization,
            loss_fn,
            initialize,
        }
    }

    /// Create with default settings.
    pub fn default() -> Self {
        Self::new(
            TargetSupport::Real,
            8,
            3.0,
            SplineOrder::Linear,
            Stabilization::None,
            LossFn::Nll,
            false,
        )
    }

    /// Split parameters into widths, heights, derivatives, and optionally lambdas.
    /// Returns borrowed slices to avoid allocation.
    fn split_params<'a>(&self, params: &'a [f64]) -> SplineParams<'a> {
        match self.order {
            SplineOrder::Quadratic => {
                let widths = &params[0..self.count_bins];
                let heights = &params[self.count_bins..2 * self.count_bins];
                let derivatives = &params[2 * self.count_bins..];
                SplineParams {
                    widths,
                    heights,
                    derivatives,
                    lambdas: None,
                }
            }
            SplineOrder::Linear => {
                let widths = &params[0..self.count_bins];
                let heights = &params[self.count_bins..2 * self.count_bins];
                let derivatives = &params[2 * self.count_bins..3 * self.count_bins - 1];
                let lambdas = &params[3 * self.count_bins - 1..];
                SplineParams {
                    widths,
                    heights,
                    derivatives,
                    lambdas: Some(lambdas),
                }
            }
        }
    }

    /// Apply the spline transform and compute log probability.
    ///
    /// The flow computes: p(y) = p_base(f^{-1}(y)) * |det(df^{-1}/dy)|
    /// log p(y) = log p_base(z) - log |det(df/dz)| where z = f^{-1}(y)
    fn log_prob_flow(&self, params: &[f64], target: f64) -> f64 {
        // Apply target transform inverse first
        let y = self.inverse_target_transform(target);

        // Check if y is in valid range
        if !y.is_finite() {
            return f64::NEG_INFINITY;
        }

        // Apply spline inverse transform to get z (base distribution sample)
        let spline_params = self.split_params(params);
        let (z, log_det_inverse) = match self.order {
            SplineOrder::Quadratic => self.rational_quadratic_spline_inverse(y, &spline_params),
            SplineOrder::Linear => self.linear_rational_spline_inverse(y, &spline_params),
        };

        if !z.is_finite() || !log_det_inverse.is_finite() {
            return f64::NEG_INFINITY;
        }

        // Base distribution log probability (standard normal)
        let log_prob_base = -0.5 * LOG_2PI - 0.5 * z * z;

        // Add log determinant of target transform inverse if needed
        let log_det_target = self.log_det_target_transform_inverse(target);

        // Total log probability
        log_prob_base + log_det_inverse + log_det_target
    }

    /// Like log_prob_flow but uses pre-allocated buffers to avoid heap allocations.
    fn log_prob_flow_with_buffers(
        &self,
        params: &[f64],
        target: f64,
        widths_buf: &mut [f64],
        heights_buf: &mut [f64],
        derivatives_buf: &mut [f64],
        lambdas_buf: &mut [f64],
    ) -> f64 {
        let y = self.inverse_target_transform(target);

        if !y.is_finite() {
            return f64::NEG_INFINITY;
        }

        let spline_params = self.split_params(params);
        let (z, log_det_inverse) = self.spline_inverse_with_buffers(
            y,
            &spline_params,
            widths_buf,
            heights_buf,
            derivatives_buf,
            lambdas_buf,
        );

        if !z.is_finite() || !log_det_inverse.is_finite() {
            return f64::NEG_INFINITY;
        }

        let log_prob_base = -0.5 * LOG_2PI - 0.5 * z * z;
        let log_det_target = self.log_det_target_transform_inverse(target);

        log_prob_base + log_det_inverse + log_det_target
    }

    /// Apply target transform inverse (from target space to real line).
    fn inverse_target_transform(&self, y: f64) -> f64 {
        match self.target_support {
            TargetSupport::Real => y,
            TargetSupport::Positive | TargetSupport::PositiveInteger => {
                // Inverse of softplus: y = ln(1 + exp(x)) => x = ln(exp(y) - 1)
                if y <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    inverse_softplus(y)
                }
            }
            TargetSupport::UnitInterval => {
                // Inverse of sigmoid (logit): x = ln(y / (1-y))
                if y <= 0.0 || y >= 1.0 {
                    if y <= 0.0 {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                } else {
                    (y / (1.0 - y)).ln()
                }
            }
        }
    }

    /// Log determinant of target transform inverse.
    fn log_det_target_transform_inverse(&self, y: f64) -> f64 {
        match self.target_support {
            TargetSupport::Real => 0.0,
            TargetSupport::Positive | TargetSupport::PositiveInteger => {
                // d/dy inverse_softplus(y) = 1 / (1 - exp(-y))
                if y <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    -(1.0 - (-y).exp()).ln()
                }
            }
            TargetSupport::UnitInterval => {
                // d/dy logit(y) = 1/(y*(1-y))
                if y <= 0.0 || y >= 1.0 {
                    f64::NEG_INFINITY
                } else {
                    -(y * (1.0 - y)).ln()
                }
            }
        }
    }

    /// Apply the forward spline transform (for sampling).
    fn forward_transform(&self, z: f64, params: &[f64]) -> f64 {
        let spline_params = self.split_params(params);
        let y_spline = match self.order {
            SplineOrder::Quadratic => self.rational_quadratic_spline_forward(z, &spline_params),
            SplineOrder::Linear => self.linear_rational_spline_forward(z, &spline_params),
        };

        // Apply target transform
        self.target_transform(y_spline)
    }

    /// Apply target transform (from real line to target space).
    fn target_transform(&self, x: f64) -> f64 {
        match self.target_support {
            TargetSupport::Real => x,
            TargetSupport::Positive | TargetSupport::PositiveInteger => softplus(x),
            TargetSupport::UnitInterval => sigmoid(x),
        }
    }

    /// Rational quadratic spline forward transform.
    ///
    /// Implementation based on Durkan et al., 2019 "Neural Spline Flows"
    fn rational_quadratic_spline_forward(&self, x: f64, params: &SplineParams) -> f64 {
        let (widths, heights, derivatives) = self.compute_spline_knots(params);

        // Handle values outside the bounding box with identity
        if x <= -self.bound {
            return x;
        }
        if x >= self.bound {
            return x;
        }

        // Find the bin
        let (bin_idx, xi) = self.find_bin(x, &widths);

        let w_k = widths[bin_idx];
        let h_k = heights[bin_idx];
        let d_k = derivatives[bin_idx];
        let d_k1 = derivatives[bin_idx + 1];
        let y_k = self.cumsum_heights(&heights, bin_idx);

        // Rational quadratic transform
        let s_k = h_k / w_k;
        let xi_sq = xi * xi;

        let numerator = h_k * (s_k * xi_sq + d_k * xi * (1.0 - xi));
        let denominator = s_k + (d_k + d_k1 - 2.0 * s_k) * xi * (1.0 - xi);

        y_k + numerator / denominator
    }

    /// Rational quadratic spline inverse transform.
    fn rational_quadratic_spline_inverse(&self, y: f64, params: &SplineParams) -> (f64, f64) {
        let (widths, heights, derivatives) = self.compute_spline_knots(params);

        // Handle values outside the bounding box with identity
        if y <= -self.bound {
            return (y, 0.0);
        }
        if y >= self.bound {
            return (y, 0.0);
        }

        // Find the bin based on y
        let (bin_idx, _) = self.find_bin_y(y, &heights);

        let w_k = widths[bin_idx];
        let h_k = heights[bin_idx];
        let d_k = derivatives[bin_idx];
        let d_k1 = derivatives[bin_idx + 1];
        let x_k = self.cumsum_widths(&widths, bin_idx);
        let y_k = self.cumsum_heights(&heights, bin_idx);

        let s_k = h_k / w_k;

        // Solve quadratic for xi
        let y_rel = y - y_k;

        let a = h_k * (s_k - d_k) + y_rel * (d_k + d_k1 - 2.0 * s_k);
        let b = h_k * d_k - y_rel * (d_k + d_k1 - 2.0 * s_k);
        let c = -s_k * y_rel;

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return (f64::NAN, f64::NEG_INFINITY);
        }

        let xi = if a.abs() < 1e-10 {
            // Linear case
            -c / b
        } else {
            // Quadratic formula - choose the root in [0, 1]
            let sqrt_disc = discriminant.sqrt();
            let xi1 = (-b + sqrt_disc) / (2.0 * a);
            let xi2 = (-b - sqrt_disc) / (2.0 * a);

            if xi1 >= 0.0 && xi1 <= 1.0 { xi1 } else { xi2 }
        };

        let x = x_k + xi * w_k;

        // Compute log determinant (negative because we're going inverse direction)
        let xi_sq = xi * xi;
        let one_minus_xi = 1.0 - xi;
        let denom = s_k + (d_k + d_k1 - 2.0 * s_k) * xi * one_minus_xi;
        let denom_sq = denom * denom;

        let numerator_deriv = s_k
            * s_k
            * (d_k1 * xi_sq + 2.0 * s_k * xi * one_minus_xi + d_k * one_minus_xi * one_minus_xi);

        let dy_dx = numerator_deriv / denom_sq;
        let log_det = -dy_dx.ln(); // Negative because inverse

        (x, log_det)
    }

    /// Linear rational spline forward transform.
    ///
    /// Implementation based on Dolatabadi et al., 2020 "Invertible Generative Modeling
    /// using Linear Rational Splines", matching pyro's `_monotonic_rational_spline`:
    /// each bin is a *two-piece* Möbius (linear rational) function split at θ = λ,
    /// built from weights wa = 1, wb = √(d_k/d_{k+1}), and a middle weight/value
    /// (wc, yc) chosen so the map is C¹, hits both knots, and has slopes d_k/d_{k+1}
    /// at the knots.
    fn linear_rational_spline_forward(&self, x: f64, params: &SplineParams) -> f64 {
        let (widths, heights, derivatives) = self.compute_spline_knots(params);
        let lambdas = self.compute_lambdas(params);

        // Handle values outside the bounding box with identity
        if x <= -self.bound {
            return x;
        }
        if x >= self.bound {
            return x;
        }

        // Find the bin
        let (bin_idx, theta) = self.find_bin(x, &widths);

        let bin = LrsBin::new(
            widths[bin_idx],
            heights[bin_idx],
            derivatives[bin_idx],
            derivatives[bin_idx + 1],
            lambdas[bin_idx],
            self.cumsum_heights(&heights, bin_idx),
        );
        bin.forward(theta)
    }

    /// Linear rational spline inverse transform.
    fn linear_rational_spline_inverse(&self, y: f64, params: &SplineParams) -> (f64, f64) {
        let (widths, heights, derivatives) = self.compute_spline_knots(params);
        let lambdas = self.compute_lambdas(params);
        self.linear_rational_spline_inverse_buf(y, &widths, &heights, &derivatives, &lambdas)
    }

    /// Compute normalized widths, heights, and derivatives from parameters.
    fn compute_spline_knots(&self, params: &SplineParams) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n_deriv = params.derivatives.len() + 2;
        let mut widths = vec![0.0; self.count_bins];
        let mut heights = vec![0.0; self.count_bins];
        let mut derivatives = vec![0.0; n_deriv];
        self.compute_spline_knots_into(params, &mut widths, &mut heights, &mut derivatives);
        (widths, heights, derivatives)
    }

    /// Compute normalized widths, heights, and derivatives into pre-allocated buffers.
    /// Avoids heap allocation in hot loops.
    ///
    /// Matches pyro's `_monotonic_rational_spline` normalization: softmax'd bin
    /// fractions are floored at `MIN_BIN_WIDTH`/`MIN_BIN_HEIGHT` (so no bin can
    /// collapse and slopes stay finite), and the inner softplus'd derivatives are
    /// floored at `MIN_DERIVATIVE`. Boundary derivatives are exactly 1 so the
    /// spline meets the identity tails smoothly.
    fn compute_spline_knots_into(
        &self,
        params: &SplineParams,
        widths: &mut [f64],
        heights: &mut [f64],
        derivatives: &mut [f64],
    ) {
        let k = self.count_bins as f64;
        let scale = 2.0 * self.bound;

        // widths = (min_w + (1 - min_w*K) * softmax(w)) * 2*bound
        softmax_into(params.widths, widths);
        for w in widths.iter_mut() {
            *w = (MIN_BIN_WIDTH + (1.0 - MIN_BIN_WIDTH * k) * *w) * scale;
        }

        softmax_into(params.heights, heights);
        for h in heights.iter_mut() {
            *h = (MIN_BIN_HEIGHT + (1.0 - MIN_BIN_HEIGHT * k) * *h) * scale;
        }

        // Inner derivatives: min_derivative + softplus(d); boundaries exactly 1.
        derivatives[0] = 1.0;
        for (i, &d) in params.derivatives.iter().enumerate() {
            derivatives[i + 1] = MIN_DERIVATIVE + softplus(d);
        }
        derivatives[params.derivatives.len() + 1] = 1.0;
    }

    /// Compute lambda parameters for linear rational splines.
    fn compute_lambdas(&self, params: &SplineParams) -> Vec<f64> {
        let mut out = vec![0.0; self.count_bins];
        self.compute_lambdas_into(params, &mut out);
        out
    }

    /// Compute lambda parameters into pre-allocated buffer. Zero-allocation.
    /// λ = min_λ + (1 - 2·min_λ)·sigmoid(l) ∈ [0.025, 0.975], as in pyro.
    fn compute_lambdas_into(&self, params: &SplineParams, out: &mut [f64]) {
        match params.lambdas {
            Some(lambdas) => {
                for (o, &l) in out.iter_mut().zip(lambdas.iter()) {
                    *o = MIN_LAMBDA + (1.0 - 2.0 * MIN_LAMBDA) * sigmoid(l);
                }
            }
            None => {
                for o in out.iter_mut() {
                    *o = 0.5;
                }
            }
        }
    }

    /// Apply spline inverse with pre-allocated knot buffers.
    /// Returns (z, log_det_inverse).
    fn spline_inverse_with_buffers(
        &self,
        y: f64,
        params: &SplineParams,
        widths_buf: &mut [f64],
        heights_buf: &mut [f64],
        derivatives_buf: &mut [f64],
        lambdas_buf: &mut [f64],
    ) -> (f64, f64) {
        self.compute_spline_knots_into(params, widths_buf, heights_buf, derivatives_buf);

        match self.order {
            SplineOrder::Quadratic => self.rational_quadratic_spline_inverse_buf(
                y,
                widths_buf,
                heights_buf,
                derivatives_buf,
            ),
            SplineOrder::Linear => {
                self.compute_lambdas_into(params, lambdas_buf);
                self.linear_rational_spline_inverse_buf(
                    y,
                    widths_buf,
                    heights_buf,
                    derivatives_buf,
                    lambdas_buf,
                )
            }
        }
    }

    /// Rational quadratic spline inverse using pre-computed buffers.
    fn rational_quadratic_spline_inverse_buf(
        &self,
        y: f64,
        widths: &[f64],
        heights: &[f64],
        derivatives: &[f64],
    ) -> (f64, f64) {
        if y <= -self.bound {
            return (y, 0.0);
        }
        if y >= self.bound {
            return (y, 0.0);
        }

        let (bin_idx, _) = self.find_bin_y(y, heights);

        let w_k = widths[bin_idx];
        let h_k = heights[bin_idx];
        let d_k = derivatives[bin_idx];
        let d_k1 = derivatives[bin_idx + 1];
        let x_k = self.cumsum_widths(widths, bin_idx);
        let y_k = self.cumsum_heights(heights, bin_idx);

        let s_k = h_k / w_k;
        let y_rel = y - y_k;

        let a = h_k * (s_k - d_k) + y_rel * (d_k + d_k1 - 2.0 * s_k);
        let b = h_k * d_k - y_rel * (d_k + d_k1 - 2.0 * s_k);
        let c = -s_k * y_rel;

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return (f64::NAN, f64::NEG_INFINITY);
        }

        let xi = if a.abs() < 1e-10 {
            -c / b
        } else {
            let sqrt_disc = discriminant.sqrt();
            let xi1 = (-b + sqrt_disc) / (2.0 * a);
            let xi2 = (-b - sqrt_disc) / (2.0 * a);
            if xi1 >= 0.0 && xi1 <= 1.0 { xi1 } else { xi2 }
        };

        let x = x_k + xi * w_k;

        let xi_sq = xi * xi;
        let one_minus_xi = 1.0 - xi;
        let denom = s_k + (d_k + d_k1 - 2.0 * s_k) * xi * one_minus_xi;
        let denom_sq = denom * denom;

        let numerator_deriv = s_k
            * s_k
            * (d_k1 * xi_sq + 2.0 * s_k * xi * one_minus_xi + d_k * one_minus_xi * one_minus_xi);
        let dy_dx = numerator_deriv / denom_sq;
        let log_det = -dy_dx.ln();

        (x, log_det)
    }

    /// Linear rational spline inverse using pre-computed buffers.
    /// Returns (x, log|det dx/dy|) = (x, -ln(dy/dx)). The inverse of each Möbius
    /// piece is closed-form, so no iterative solver is needed.
    fn linear_rational_spline_inverse_buf(
        &self,
        y: f64,
        widths: &[f64],
        heights: &[f64],
        derivatives: &[f64],
        lambdas: &[f64],
    ) -> (f64, f64) {
        if y <= -self.bound {
            return (y, 0.0);
        }
        if y >= self.bound {
            return (y, 0.0);
        }

        let (bin_idx, _) = self.find_bin_y(y, heights);

        let bin = LrsBin::new(
            widths[bin_idx],
            heights[bin_idx],
            derivatives[bin_idx],
            derivatives[bin_idx + 1],
            lambdas[bin_idx],
            self.cumsum_heights(heights, bin_idx),
        );
        let x_k = self.cumsum_widths(widths, bin_idx);

        let theta = bin.inverse(y);
        let x = x_k + theta * bin.w;
        let dy_dx = bin.derivative(theta) / bin.w;
        let log_det = -dy_dx.ln();

        (x, log_det)
    }

    /// Find which bin x falls into and compute local coordinate xi.
    fn find_bin(&self, x: f64, widths: &[f64]) -> (usize, f64) {
        let mut cumsum = -self.bound;
        for (i, &w) in widths.iter().enumerate() {
            if x < cumsum + w {
                let xi = (x - cumsum) / w;
                return (i, xi.clamp(0.0, 1.0));
            }
            cumsum += w;
        }
        // Return last bin if at boundary
        (widths.len() - 1, 1.0)
    }

    /// Find which bin y falls into (based on heights).
    fn find_bin_y(&self, y: f64, heights: &[f64]) -> (usize, f64) {
        let mut cumsum = -self.bound;
        for (i, &h) in heights.iter().enumerate() {
            if y < cumsum + h {
                let yi = (y - cumsum) / h;
                return (i, yi.clamp(0.0, 1.0));
            }
            cumsum += h;
        }
        (heights.len() - 1, 1.0)
    }

    /// Compute cumulative sum of widths up to (but not including) bin_idx.
    fn cumsum_widths(&self, widths: &[f64], bin_idx: usize) -> f64 {
        -self.bound + widths[..bin_idx].iter().sum::<f64>()
    }

    /// Compute cumulative sum of heights up to (but not including) bin_idx.
    fn cumsum_heights(&self, heights: &[f64], bin_idx: usize) -> f64 {
        -self.bound + heights[..bin_idx].iter().sum::<f64>()
    }
}

/// Internal struct to hold split spline parameters (borrows from the params slice).
struct SplineParams<'a> {
    widths: &'a [f64],
    heights: &'a [f64],
    derivatives: &'a [f64],
    lambdas: Option<&'a [f64]>,
}

/// One bin of a linear rational spline (Dolatabadi et al., 2020), in the
/// two-piece Möbius form used by pyro's `_monotonic_rational_spline`.
///
/// Within the bin, in local coordinate θ = (x - x_k)/w ∈ [0, 1], the map is a
/// rational linear (Möbius) function on [0, λ] and another on [λ, 1], glued at
/// the middle value yc with weights chosen so that the map interpolates
/// (0, y_k) → (1, y_k + h), has derivative d_k at θ=0 and d_{k+1} at θ=1, and
/// is C¹ at θ=λ:
///   wa = 1,  wb = √(d_k / d_{k+1}),
///   wc = (λ·wa·d_k + (1-λ)·wb·d_{k+1}) / s,   with s = h/w,
///   yc = ((1-λ)·wa·y_k + λ·wb·y_{k+1}) / ((1-λ)·wa + λ·wb).
struct LrsBin {
    w: f64,
    y_k: f64,
    y_k1: f64,
    lambda: f64,
    wa: f64,
    wb: f64,
    wc: f64,
    yc: f64,
}

impl LrsBin {
    fn new(w: f64, h: f64, d_k: f64, d_k1: f64, lambda: f64, y_k: f64) -> Self {
        let s = h / w;
        let wa = 1.0;
        let wb = (d_k / d_k1).sqrt() * wa;
        let wc = (lambda * wa * d_k + (1.0 - lambda) * wb * d_k1) / s;
        let y_k1 = y_k + h;
        let yc =
            ((1.0 - lambda) * wa * y_k + lambda * wb * y_k1) / ((1.0 - lambda) * wa + lambda * wb);
        Self {
            w,
            y_k,
            y_k1,
            lambda,
            wa,
            wb,
            wc,
            yc,
        }
    }

    /// Forward map θ ∈ [0,1] → y.
    fn forward(&self, theta: f64) -> f64 {
        if theta <= self.lambda {
            let num = self.wa * self.y_k * (self.lambda - theta) + self.wc * self.yc * theta;
            let den = self.wa * (self.lambda - theta) + self.wc * theta;
            num / den
        } else {
            let num =
                self.wc * self.yc * (1.0 - theta) + self.wb * self.y_k1 * (theta - self.lambda);
            let den = self.wc * (1.0 - theta) + self.wb * (theta - self.lambda);
            num / den
        }
    }

    /// dy/dθ at θ (divide by the bin width for dy/dx). Strictly positive.
    fn derivative(&self, theta: f64) -> f64 {
        if theta <= self.lambda {
            let den = self.wa * (self.lambda - theta) + self.wc * theta;
            self.lambda * self.wa * self.wc * (self.yc - self.y_k) / (den * den)
        } else {
            let den = self.wc * (1.0 - theta) + self.wb * (theta - self.lambda);
            (1.0 - self.lambda) * self.wb * self.wc * (self.y_k1 - self.yc) / (den * den)
        }
    }

    /// Closed-form inverse: y in [y_k, y_{k+1}] → θ ∈ [0,1]. Each Möbius piece
    /// inverts exactly; the branch is chosen by comparing y with yc.
    fn inverse(&self, y: f64) -> f64 {
        let theta = if y <= self.yc {
            self.lambda * self.wa * (self.y_k - y)
                / ((self.wc - self.wa) * y + self.wa * self.y_k - self.wc * self.yc)
        } else {
            ((self.wc - self.lambda * self.wb) * y - self.wc * self.yc
                + self.lambda * self.wb * self.y_k1)
                / ((self.wc - self.wb) * y - self.wc * self.yc + self.wb * self.y_k1)
        };
        theta.clamp(0.0, 1.0)
    }
}

#[typetag::serde]
impl Distribution for SplineFlow {
    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "SplineFlow"
    }

    fn n_params(&self) -> usize {
        self.params.len()
    }

    fn params(&self) -> &[DistributionParam] {
        &self.params
    }

    fn loss_fn(&self) -> LossFn {
        self.loss_fn
    }

    fn stabilization(&self) -> Stabilization {
        self.stabilization
    }

    fn should_initialize(&self) -> bool {
        self.initialize
    }

    fn is_discrete(&self) -> bool {
        self.target_support.is_discrete()
    }

    fn log_prob(&self, params: &[f64], target: &[f64]) -> f64 {
        self.log_prob_flow(params, target[0])
    }

    fn nll(&self, params: &ArrayView2<f64>, target: &ResponseData) -> f64 {
        match target {
            ResponseData::Univariate(y) => {
                // Reusable per-thread buffers for spline knot computation
                let n_deriv = match self.order {
                    SplineOrder::Quadratic => self.count_bins + 1,
                    SplineOrder::Linear => self.count_bins + 1,
                };
                let n_params = self.n_params();
                let n_samples = params.nrows();

                // Scratch: (widths, heights, derivatives, lambdas, row fallback).
                // NaN log-probs contribute 0 (torch.nansum parity).
                type Scratch = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
                let term = |scratch: &mut Scratch, i: usize| -> f64 {
                    let (widths_buf, heights_buf, derivatives_buf, lambdas_buf, params_buf) =
                        scratch;
                    let row = params.row(i);
                    let p: &[f64] = match row.as_slice() {
                        Some(s) => s,
                        None => {
                            for (k, &v) in row.iter().enumerate() {
                                params_buf[k] = v;
                            }
                            &params_buf[..n_params]
                        }
                    };
                    let lp = self.log_prob_flow_with_buffers(
                        p,
                        y[i],
                        widths_buf,
                        heights_buf,
                        derivatives_buf,
                        lambdas_buf,
                    );
                    if lp.is_nan() { 0.0 } else { -lp }
                };
                let make_scratch = || -> Scratch {
                    (
                        vec![0.0f64; self.count_bins],
                        vec![0.0f64; self.count_bins],
                        vec![0.0f64; n_deriv],
                        vec![0.0f64; self.count_bins],
                        vec![0.0f64; n_params],
                    )
                };

                // Each row evaluates the full spline transform — heavyweight
                // enough to parallelize at the row threshold.
                if n_samples >= crate::distributions::util::PAR_ROW_THRESHOLD {
                    (0..n_samples)
                        .into_par_iter()
                        .map_init(make_scratch, term)
                        .sum()
                } else {
                    let mut scratch = make_scratch();
                    (0..n_samples).map(|i| term(&mut scratch, i)).sum()
                }
            }
            ResponseData::Multivariate(_) => {
                panic!("SplineFlow is a univariate distribution.")
            }
        }
    }

    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> {
        let n_obs = params.nrows();
        let mut result = Array2::zeros((n_samples, n_obs));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let standard_normal = Normal::new(0.0, 1.0).unwrap();

        // Pre-allocate a fallback buffer for non-contiguous rows
        let n_params = self.n_params();
        let mut params_buf = vec![0.0f64; n_params];

        for j in 0..n_obs {
            let row = params.row(j);
            let obs_params: &[f64] = match row.as_slice() {
                Some(s) => s,
                None => {
                    for (k, &v) in row.iter().enumerate() {
                        params_buf[k] = v;
                    }
                    &params_buf[..n_params]
                }
            };

            for i in 0..n_samples {
                // Sample from base distribution (standard normal)
                let z: f64 = standard_normal.sample(&mut rng);

                // Apply forward transform
                let y = self.forward_transform(z, obs_params);

                // Round if discrete
                let y = if self.is_discrete() {
                    y.round().max(0.0)
                } else {
                    y
                };

                result[[i, j]] = y;
            }
        }

        result
    }

    fn calculate_start_values(
        &self,
        target: &ResponseData,
        max_iter: usize,
    ) -> crate::error::Result<(f64, Array1<f64>)> {
        use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient, State};
        use argmin::solver::linesearch::MoreThuenteLineSearch;
        use argmin::solver::quasinewton::LBFGS;

        let n_params = self.n_params();

        let targets: Vec<f64> = match target {
            ResponseData::Univariate(y) => y.iter().copied().collect(),
            ResponseData::Multivariate(_) => {
                return Err(crate::error::GradientLSSError::InvalidInput(
                    "SplineFlow requires univariate target".into(),
                ));
            }
        };

        // Clone self data needed for the optimization problem
        let count_bins = self.count_bins;
        let bound = self.bound;
        let order = self.order;
        let target_support = self.target_support;

        // Create L-BFGS optimization problem
        struct SplineFlowOptProblem {
            targets: Vec<f64>,
            count_bins: usize,
            bound: f64,
            order: SplineOrder,
            target_support: TargetSupport,
            n_params: usize,
        }

        impl SplineFlowOptProblem {
            fn compute_loss(&self, params: &[f64]) -> f64 {
                // Create a temporary SplineFlow to compute log_prob
                let dist = SplineFlow::new(
                    self.target_support,
                    self.count_bins,
                    self.bound,
                    self.order,
                    Stabilization::None,
                    LossFn::Nll,
                    false,
                );

                self.targets
                    .iter()
                    .map(|&y| -dist.log_prob_flow(params, y))
                    .sum()
            }
        }

        impl CostFunction for SplineFlowOptProblem {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, params: &Self::Param) -> std::result::Result<Self::Output, ArgminError> {
                let loss = self.compute_loss(params);
                if loss.is_finite() {
                    Ok(loss)
                } else {
                    Ok(f64::MAX)
                }
            }
        }

        impl Gradient for SplineFlowOptProblem {
            type Param = Vec<f64>;
            type Gradient = Vec<f64>;

            fn gradient(
                &self,
                params: &Self::Param,
            ) -> std::result::Result<Self::Gradient, ArgminError> {
                let eps = 1e-5;
                let mut grad = vec![0.0; self.n_params];
                let base_cost = self.compute_loss(params);

                for i in 0..self.n_params {
                    let mut params_plus = params.clone();
                    params_plus[i] += eps;
                    let cost_plus = self.compute_loss(&params_plus);
                    grad[i] = (cost_plus - base_cost) / eps;

                    // Clip gradient to prevent instability
                    if !grad[i].is_finite() {
                        grad[i] = 0.0;
                    } else {
                        grad[i] = grad[i].clamp(-100.0, 100.0);
                    }
                }

                Ok(grad)
            }
        }

        let problem = SplineFlowOptProblem {
            targets,
            count_bins,
            bound,
            order,
            target_support,
            n_params,
        };

        // Initial guess: small random values near zero for splines
        let init_params: Vec<f64> = vec![0.0; n_params];

        // Set up L-BFGS with More-Thuente line search (similar to PyTorch's strong_wolfe)
        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 7); // 7 is the default L-BFGS memory

        // Run the optimizer with adaptive iteration limit
        let actual_max_iter = max_iter.max(50); // At least 50 iterations for spline flows

        let result = Executor::new(problem, solver)
            .configure(|state| {
                state
                    .param(init_params.clone())
                    .max_iters(actual_max_iter as u64)
                    .target_cost(0.0)
            })
            .run();

        match result {
            Ok(res) => {
                let best_params: Vec<f64> = res
                    .state()
                    .get_best_param()
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; n_params]);
                let best_cost = res.state().get_best_cost();

                // Convert to Array1 and replace any NaNs
                let mut params_arr = Array1::from_vec(best_params);
                for v in params_arr.iter_mut() {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                }

                Ok((best_cost, params_arr))
            }
            Err(_) => {
                // Fall back to zero initialization if L-BFGS fails
                let params_arr = Array1::from_elem(n_params, 0.0);

                // Compute loss at zero params
                let loss: f64 = match target {
                    ResponseData::Univariate(y) => y
                        .iter()
                        .map(|&yi| -self.log_prob_flow(&vec![0.0; n_params], yi))
                        .sum(),
                    _ => f64::INFINITY,
                };

                Ok((loss, params_arr))
            }
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Knot-normalization floors, matching pyro's `_monotonic_rational_spline`
/// defaults (DEFAULT_MIN_BIN_WIDTH / HEIGHT / DERIVATIVE / LAMBDA). They keep
/// bins from collapsing under softmax and slopes/λ away from degenerate values.
const MIN_BIN_WIDTH: f64 = 1e-3;
const MIN_BIN_HEIGHT: f64 = 1e-3;
const MIN_DERIVATIVE: f64 = 1e-3;
const MIN_LAMBDA: f64 = 0.025;

/// Softplus function: ln(1 + exp(x))
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        1e-6
    } else {
        (1.0 + x.exp()).ln().max(1e-6)
    }
}

/// Inverse softplus: ln(exp(y) - 1)
fn inverse_softplus(y: f64) -> f64 {
    if y > 20.0 {
        y
    } else if y < 1e-6 {
        -20.0
    } else {
        (y.exp() - 1.0).ln()
    }
}

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    if x > 20.0 {
        1.0 - 1e-6
    } else if x < -20.0 {
        1e-6
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Softmax function for a slice.
fn softmax(x: &[f64]) -> Vec<f64> {
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Vec<f64> = x.iter().map(|&v| (v - max_x).exp()).collect();
    let sum_exp: f64 = exp_x.iter().sum();
    exp_x.iter().map(|&v| v / sum_exp).collect()
}

/// Softmax into pre-allocated output buffer. Zero-allocation.
fn softmax_into(x: &[f64], out: &mut [f64]) {
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum_exp = 0.0;
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        let e = (v - max_x).exp();
        *o = e;
        sum_exp += e;
    }
    let inv_sum = 1.0 / sum_exp;
    for o in out.iter_mut() {
        *o *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spline_flow_creation() {
        let dist = SplineFlow::default();
        assert_eq!(dist.count_bins, 8);
        assert_eq!(dist.order, SplineOrder::Linear);
        // Linear: 3*8 + (8-1) = 31 parameters
        assert_eq!(dist.n_params(), 31);
    }

    #[test]
    fn test_spline_flow_quadratic_creation() {
        let dist = SplineFlow::new(
            TargetSupport::Real,
            8,
            3.0,
            SplineOrder::Quadratic,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        // Quadratic: 2*8 + (8-1) = 23 parameters
        assert_eq!(dist.n_params(), 23);
    }

    #[test]
    fn test_spline_order_n_params() {
        assert_eq!(SplineOrder::Quadratic.n_params(8), 23);
        assert_eq!(SplineOrder::Linear.n_params(8), 31);
        assert_eq!(SplineOrder::Quadratic.n_params(4), 11);
        assert_eq!(SplineOrder::Linear.n_params(4), 15);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);
        let sum: f64 = result.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_softplus() {
        assert!(softplus(0.0) > 0.0);
        assert_relative_eq!(softplus(0.0), 2.0_f64.ln(), epsilon = 1e-6);
        assert!(softplus(-100.0) > 0.0);
        assert_relative_eq!(softplus(100.0), 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
        assert!(sigmoid(-100.0) > 0.0);
        assert!(sigmoid(100.0) < 1.0);
    }

    #[test]
    fn test_target_support() {
        assert!(TargetSupport::PositiveInteger.is_discrete());
        assert!(!TargetSupport::Real.is_discrete());
        assert!(!TargetSupport::Positive.is_discrete());
        assert!(!TargetSupport::UnitInterval.is_discrete());
    }

    #[test]
    fn test_spline_flow_log_prob_finite() {
        let dist = SplineFlow::new(
            TargetSupport::Real,
            4,
            3.0,
            SplineOrder::Quadratic,
            Stabilization::None,
            LossFn::Nll,
            false,
        );

        // Use zero parameters (which will give uniform-ish spline)
        let params = vec![0.0; dist.n_params()];
        let log_p = dist.log_prob_flow(&params, 0.0);

        // Log prob should be finite for reasonable inputs
        assert!(log_p.is_finite(), "log_prob was not finite: {}", log_p);
    }

    #[test]
    fn test_spline_flow_sampling() {
        let dist = SplineFlow::new(
            TargetSupport::Real,
            4,
            3.0,
            SplineOrder::Quadratic,
            Stabilization::None,
            LossFn::Nll,
            false,
        );

        let n_params = dist.n_params();
        let params = Array2::zeros((2, n_params));
        let samples = dist.sample(&params.view(), 100, 42);

        assert_eq!(samples.dim(), (100, 2));
        // Samples should be finite
        assert!(samples.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_spline_flow_positive_support() {
        let dist = SplineFlow::new(
            TargetSupport::Positive,
            4,
            3.0,
            SplineOrder::Linear,
            Stabilization::None,
            LossFn::Nll,
            false,
        );

        let n_params = dist.n_params();
        let params = Array2::zeros((1, n_params));
        let samples = dist.sample(&params.view(), 100, 42);

        // All samples should be non-negative for positive support
        assert!(samples.iter().all(|&x| x >= 0.0));
    }

    /// Pseudo-random but deterministic parameter vector exercising uneven bins,
    /// varied derivatives and asymmetric lambdas.
    fn varied_params(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| ((i as f64 * 0.7).sin() * 1.3) + ((i % 3) as f64 - 1.0) * 0.4)
            .collect()
    }

    #[test]
    fn test_linear_spline_forward_interpolates_knots() {
        // The forward map must be continuous across bins: approaching a knot from
        // the left and right must agree (the old single-piece formula violated
        // this whenever s_k != d_{k+1}).
        let dist = SplineFlow::new(
            TargetSupport::Real,
            6,
            3.0,
            SplineOrder::Linear,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        let params = varied_params(dist.n_params());
        let sp = dist.split_params(&params);
        let (widths, _, _) = dist.compute_spline_knots(&sp);

        let mut knot = -dist.bound;
        for w in widths.iter().take(widths.len() - 1) {
            knot += w;
            let eps = 1e-7;
            let left = dist.forward_transform(knot - eps, &params);
            let right = dist.forward_transform(knot + eps, &params);
            assert!(
                (left - right).abs() < 1e-4,
                "forward discontinuous at knot {}: {} vs {}",
                knot,
                left,
                right
            );
        }

        // Identity tails: at the boundary the spline must meet y = x.
        let at_bound = dist.forward_transform(dist.bound - 1e-9, &params);
        assert!(
            (at_bound - dist.bound).abs() < 1e-5,
            "spline does not meet identity tail at +bound: {}",
            at_bound
        );
        let at_lower = dist.forward_transform(-dist.bound + 1e-9, &params);
        assert!(
            (at_lower + dist.bound).abs() < 1e-5,
            "spline does not meet identity tail at -bound: {}",
            at_lower
        );
    }

    #[test]
    fn test_linear_spline_roundtrip_and_log_det() {
        let dist = SplineFlow::new(
            TargetSupport::Real,
            6,
            3.0,
            SplineOrder::Linear,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        let params = varied_params(dist.n_params());
        let sp = dist.split_params(&params);
        let (widths, heights, derivatives) = dist.compute_spline_knots(&sp);
        let lambdas = dist.compute_lambdas(&sp);

        let mut prev_x = f64::NEG_INFINITY;
        for i in 0..200 {
            let y = -2.95 + 5.9 * (i as f64) / 199.0;
            let (x, log_det) = dist.linear_rational_spline_inverse_buf(
                y,
                &widths,
                &heights,
                &derivatives,
                &lambdas,
            );
            assert!(x.is_finite() && log_det.is_finite());
            // Monotone inverse
            assert!(x > prev_x, "inverse not monotone at y={}", y);
            prev_x = x;

            // Round-trip: forward(inverse(y)) == y
            let y_rt = dist.forward_transform(x, &params);
            assert!(
                (y_rt - y).abs() < 1e-8,
                "round-trip failed: y={}, forward(inverse(y))={}",
                y,
                y_rt
            );

            // log_det must equal -ln(dy/dx) of the actual forward map
            let h = 1e-6;
            let dy_dx_fd = (dist.forward_transform(x + h, &params)
                - dist.forward_transform(x - h, &params))
                / (2.0 * h);
            assert!(
                ((-dy_dx_fd.ln()) - log_det).abs() < 1e-4,
                "log_det mismatch at y={}: analytic={}, fd={}",
                y,
                log_det,
                -dy_dx_fd.ln()
            );
        }
    }

    #[test]
    fn test_quadratic_spline_roundtrip_and_log_det() {
        let dist = SplineFlow::new(
            TargetSupport::Real,
            6,
            3.0,
            SplineOrder::Quadratic,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        let params = varied_params(dist.n_params());

        for i in 0..200 {
            let y = -2.95 + 5.9 * (i as f64) / 199.0;
            let sp = dist.split_params(&params);
            let (x, log_det) = dist.rational_quadratic_spline_inverse(y, &sp);
            assert!(x.is_finite() && log_det.is_finite());

            let y_rt = dist.forward_transform(x, &params);
            assert!(
                (y_rt - y).abs() < 1e-7,
                "round-trip failed: y={}, forward(inverse(y))={}",
                y,
                y_rt
            );

            let h = 1e-6;
            let dy_dx_fd = (dist.forward_transform(x + h, &params)
                - dist.forward_transform(x - h, &params))
                / (2.0 * h);
            assert!(
                ((-dy_dx_fd.ln()) - log_det).abs() < 1e-4,
                "log_det mismatch at y={}: analytic={}, fd={}",
                y,
                log_det,
                -dy_dx_fd.ln()
            );
        }
    }

    #[test]
    fn test_lrs_bin_c1_at_lambda() {
        // The two Möbius pieces must agree in value and derivative at θ = λ.
        let bin = LrsBin::new(0.8, 1.3, 0.4, 2.1, 0.3, -0.5);
        let eps = 1e-9;
        let v_left = bin.forward(bin.lambda - eps);
        let v_right = bin.forward(bin.lambda + eps);
        assert!((v_left - v_right).abs() < 1e-6);
        let d_left = bin.derivative(bin.lambda - eps);
        let d_right = bin.derivative(bin.lambda + eps);
        assert!(
            (d_left - d_right).abs() / d_left < 1e-5,
            "derivative not continuous at lambda: {} vs {}",
            d_left,
            d_right
        );
        // Knot interpolation and end slopes (defining properties of the LRS).
        assert!((bin.forward(0.0) - bin.y_k).abs() < 1e-12);
        assert!((bin.forward(1.0) - bin.y_k1).abs() < 1e-12);
        assert!((bin.derivative(0.0) / bin.w - 0.4).abs() < 1e-9);
        assert!((bin.derivative(1.0) / bin.w - 2.1).abs() < 1e-9);
    }

    #[test]
    fn test_spline_flow_unit_interval_support() {
        let dist = SplineFlow::new(
            TargetSupport::UnitInterval,
            4,
            3.0,
            SplineOrder::Linear,
            Stabilization::None,
            LossFn::Nll,
            false,
        );

        let n_params = dist.n_params();
        let params = Array2::zeros((1, n_params));
        let samples = dist.sample(&params.view(), 100, 42);

        // All samples should be in [0, 1] for unit interval support
        assert!(samples.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}
