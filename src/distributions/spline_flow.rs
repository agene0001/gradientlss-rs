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
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as RandDistribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

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
    fn split_params(&self, params: &[f64]) -> SplineParams {
        match self.order {
            SplineOrder::Quadratic => {
                let widths = &params[0..self.count_bins];
                let heights = &params[self.count_bins..2 * self.count_bins];
                let derivatives = &params[2 * self.count_bins..];
                SplineParams {
                    widths: widths.to_vec(),
                    heights: heights.to_vec(),
                    derivatives: derivatives.to_vec(),
                    lambdas: None,
                }
            }
            SplineOrder::Linear => {
                let widths = &params[0..self.count_bins];
                let heights = &params[self.count_bins..2 * self.count_bins];
                let derivatives = &params[2 * self.count_bins..3 * self.count_bins - 1];
                let lambdas = &params[3 * self.count_bins - 1..];
                SplineParams {
                    widths: widths.to_vec(),
                    heights: heights.to_vec(),
                    derivatives: derivatives.to_vec(),
                    lambdas: Some(lambdas.to_vec()),
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
        let log_prob_base = -0.5 * (2.0 * PI).ln() - 0.5 * z * z;

        // Add log determinant of target transform inverse if needed
        let log_det_target = self.log_det_target_transform_inverse(target);

        // Total log probability
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
    /// using Linear Rational Splines"
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
        let (bin_idx, xi) = self.find_bin(x, &widths);

        let w_k = widths[bin_idx];
        let h_k = heights[bin_idx];
        let d_k = derivatives[bin_idx];
        let d_k1 = derivatives[bin_idx + 1];
        let lambda_k = lambdas[bin_idx];
        let y_k = self.cumsum_heights(&heights, bin_idx);

        // Linear rational spline transform
        let s_k = h_k / w_k;

        // Compute using linear rational formula
        let t = xi;
        let one_minus_t = 1.0 - t;

        let numerator =
            d_k * one_minus_t * one_minus_t + 2.0 * s_k * t * one_minus_t + d_k1 * t * t;
        let denominator = d_k * one_minus_t + lambda_k * s_k * t * one_minus_t + d_k1 * t;

        if denominator.abs() < 1e-10 {
            return y_k + h_k * t;
        }

        y_k + h_k * t * numerator
            / (denominator * (d_k * one_minus_t + d_k1 * t + lambda_k * s_k * t * one_minus_t))
    }

    /// Linear rational spline inverse transform.
    fn linear_rational_spline_inverse(&self, y: f64, params: &SplineParams) -> (f64, f64) {
        let (widths, heights, derivatives) = self.compute_spline_knots(params);
        let lambdas = self.compute_lambdas(params);

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
        let lambda_k = lambdas[bin_idx];
        let x_k = self.cumsum_widths(&widths, bin_idx);
        let y_k = self.cumsum_heights(&heights, bin_idx);

        let s_k = h_k / w_k;
        let y_rel = (y - y_k) / h_k;

        // For linear rational splines, we solve a quadratic equation
        // This is a simplified version - for numerical stability we use iterative refinement
        let t = self.solve_linear_rational_inverse(y_rel, d_k, d_k1, s_k, lambda_k);

        let x = x_k + t * w_k;

        // Compute log determinant
        let one_minus_t = 1.0 - t;

        // Derivative computation for linear rational spline
        let numerator =
            d_k * one_minus_t * one_minus_t + 2.0 * s_k * t * one_minus_t + d_k1 * t * t;
        let denom1 = d_k * one_minus_t + d_k1 * t + lambda_k * s_k * t * one_minus_t;

        let dy_dt = h_k * numerator / (denom1 * denom1);
        let dy_dx = dy_dt / w_k;

        let log_det = -dy_dx.abs().ln();

        (x, log_det)
    }

    /// Solve for t in linear rational spline inverse using Newton's method.
    fn solve_linear_rational_inverse(
        &self,
        y_rel: f64,
        d_k: f64,
        d_k1: f64,
        s_k: f64,
        lambda_k: f64,
    ) -> f64 {
        // Initial guess
        let mut t = y_rel.clamp(0.0, 1.0);

        // Newton-Raphson iterations
        for _ in 0..20 {
            let one_minus_t = 1.0 - t;

            let numerator =
                d_k * one_minus_t * one_minus_t + 2.0 * s_k * t * one_minus_t + d_k1 * t * t;
            let denom = d_k * one_minus_t + d_k1 * t + lambda_k * s_k * t * one_minus_t;

            if denom.abs() < 1e-10 {
                break;
            }

            let f_t = t * numerator / denom - y_rel;

            // Derivative of f(t)
            let df_dt = numerator / denom
                + t * ((-2.0 * d_k * one_minus_t + 2.0 * s_k * (1.0 - 2.0 * t) + 2.0 * d_k1 * t)
                    / denom
                    - numerator * (-d_k + d_k1 + lambda_k * s_k * (1.0 - 2.0 * t))
                        / (denom * denom));

            if df_dt.abs() < 1e-10 {
                break;
            }

            let t_new = t - f_t / df_dt;
            if (t_new - t).abs() < 1e-10 {
                t = t_new.clamp(0.0, 1.0);
                break;
            }
            t = t_new.clamp(0.0, 1.0);
        }

        t
    }

    /// Compute normalized widths, heights, and derivatives from parameters.
    fn compute_spline_knots(&self, params: &SplineParams) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Apply softmax to widths and heights to ensure they sum to 2*bound
        let widths = softmax(&params.widths);
        let widths: Vec<f64> = widths.iter().map(|&w| w * 2.0 * self.bound).collect();

        let heights = softmax(&params.heights);
        let heights: Vec<f64> = heights.iter().map(|&h| h * 2.0 * self.bound).collect();

        // Apply softplus to derivatives to ensure positivity, with boundary conditions
        let mut derivatives = vec![1.0]; // d_0 = 1
        for &d in &params.derivatives {
            derivatives.push(softplus(d));
        }
        derivatives.push(1.0); // d_K = 1

        (widths, heights, derivatives)
    }

    /// Compute lambda parameters for linear rational splines.
    fn compute_lambdas(&self, params: &SplineParams) -> Vec<f64> {
        match &params.lambdas {
            Some(lambdas) => {
                // Apply sigmoid to constrain lambdas to (0, 1)
                lambdas.iter().map(|&l| sigmoid(l)).collect()
            }
            None => vec![0.5; self.count_bins], // Default value
        }
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

/// Internal struct to hold split spline parameters.
struct SplineParams {
    widths: Vec<f64>,
    heights: Vec<f64>,
    derivatives: Vec<f64>,
    lambdas: Option<Vec<f64>>,
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
                let mut total = 0.0;
                for (i, &y_val) in y.iter().enumerate() {
                    let p: Vec<f64> = params.row(i).to_vec();
                    total -= self.log_prob_flow(&p, y_val);
                }
                total
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

        for j in 0..n_obs {
            let obs_params: Vec<f64> = params.row(j).to_vec();

            for i in 0..n_samples {
                // Sample from base distribution (standard normal)
                let z: f64 = standard_normal.sample(&mut rng);

                // Apply forward transform
                let y = self.forward_transform(z, &obs_params);

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
