//! # GradientLSS
//!
//! Distributional Gradient Boosting for Location, Scale, and Shape (LSS).
//!
//! This crate provides a unified interface for probabilistic gradient boosting
//! using either XGBoost or LightGBM as the underlying engine.
//!
//! ## Features
//!
//! - `xgboost` - Enable XGBoost backend support
//! - `lightgbm` - Enable LightGBM backend support
//! - `plotting` - Enable plotting capabilities (feature importance, PDP, etc.)
//! - `full` - Enable all backends
//!
//! ## Example
//!
//! ```ignore
//! use gradientlss::prelude::*;
//! use gradientlss::distributions::Gaussian;
//!
//! // Create a Gaussian distribution with default settings
//! let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
//!
//! // Create the model with XGBoost backend
//! #[cfg(feature = "xgboost")]
//! let model = GradientLSS::<XGBoostBackend>::new(dist);
//! ```

pub mod backend;
pub mod dist_select;
pub mod distributions;
pub mod error;
pub mod hyper_opt;
pub mod interpretability;
pub mod model;
#[cfg(feature = "plotting")]
pub mod plotting;
pub mod types;
pub mod utils;

pub mod prelude {
    //! Convenient re-exports of commonly used types.
    pub use crate::backend::{
        Backend, DistributionInfo, FeatureImportance, FeatureImportanceType, PredictionOutput,
        TrainConfig, TrainingCallback,
    };
    #[cfg(feature = "plotting")]
    pub use crate::dist_select::plot_dist_select_densities;
    pub use crate::dist_select::{DistSelectResult, dist_select, dist_select_with_params};
    pub use crate::distributions::{Distribution, LossFn, Stabilization};
    pub use crate::error::{GradientLSSError, Result};
    pub use crate::hyper_opt::{HyperOptConfig, HyperOptResult, PruningStrategy};
    pub use crate::interpretability::{IcePlot, PartialDependence, ShapExportData};
    pub use crate::model::{GradientLSS, PredType};
    pub use crate::utils::ResponseFn;

    #[cfg(feature = "xgboost")]
    pub use crate::backend::XGBoostBackend;

    #[cfg(feature = "lightgbm")]
    pub use crate::backend::LightGBMBackend;

    #[cfg(feature = "plotting")]
    pub use crate::plotting::{
        ColorPalette, PlotConfig, plot_density_comparison, plot_dist_select,
        plot_feature_importance, plot_partial_dependence, plot_partial_dependence_multi,
    };
}
