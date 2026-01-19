//! Backend implementations for gradient boosting libraries.
//!
//! This module provides a unified interface for different gradient boosting
//! backends (XGBoost, LightGBM) through the `Backend` trait.

mod traits;

#[cfg(feature = "xgboost")]
pub mod xgboost_backend;

#[cfg(feature = "lightgbm")]
pub mod lightgbm_backend;

pub use traits::{
    Backend, BackendDataset, BackendModel, BackendParams, CallbackAction, CallbackList,
    DistributionInfo, EarlyStoppingCallback, FeatureImportance, FeatureImportanceType,
    HistoryCallback, LearningRateSchedule, LearningRateScheduler, ParamValue, PredictionOutput,
    PrintCallback, SimpleParams, TrainConfig, TrainingCallback, TrainingResult,
};

#[cfg(feature = "xgboost")]
pub use xgboost_backend::XGBoostBackend;

#[cfg(feature = "lightgbm")]
pub use lightgbm_backend::LightGBMBackend;
