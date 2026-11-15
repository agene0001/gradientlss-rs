//! Distribution implementations for GradientLSS.
//!
//! This module provides the core distribution trait and implementations
//! for various probability distributions used in distributional regression.

pub mod base;
mod beta;
mod cauchy;
mod dirichlet;
mod expectile;
mod gamma;
mod gaussian;
mod gumbel;
mod laplace;
mod location_scale;
mod log_normal;
mod logistic;
mod mixture;
mod mvn;
mod mvn_lora;
mod mvt;
mod negative_binomial;
mod poisson;
pub mod spline_flow;
mod student_t;
pub(crate) mod util;
mod weibull;
mod zabeta;
mod zagamma;
mod zaln;
mod zinb;
mod zipoisson;

pub use base::{
    Distribution, DistributionParam, GradientsAndHessians, HessianMode, LossFn, Stabilization,
};
pub use beta::{Beta, BetaParameterization};
pub use cauchy::Cauchy;
pub use dirichlet::Dirichlet;
pub use expectile::Expectile;
pub use gamma::{Gamma, GammaParameterization};
pub use gaussian::Gaussian;
pub use gumbel::{Gumbel, GUMBEL_START_SCALE_FACTOR};
pub use laplace::Laplace;
pub use location_scale::LocationScaleStart;
pub use log_normal::LogNormal;
pub use logistic::Logistic;
pub use mixture::Mixture;
pub use mvn::MVN;
pub use mvn_lora::MVNLoRa;
pub use mvt::MVT;
pub use negative_binomial::{NbParameterization, NegativeBinomial};
pub use poisson::Poisson;
pub use spline_flow::{SplineFlow, SplineOrder, TargetSupport};
pub use student_t::{StudentT, StudentTStart, sd_and_excess_kurtosis, ADAPTIVE_KURTOSIS_GATE};
pub use weibull::Weibull;
pub use zabeta::ZABeta;
pub use zagamma::ZAGamma;
pub use zaln::ZALN;
pub use zinb::ZINB;
pub use zipoisson::ZIPoisson;
