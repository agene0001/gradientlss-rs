//! Error types for GradientLSS.

use ndarray::ShapeError;
use thiserror::Error;

/// Result type alias for GradientLSS operations.
pub type Result<T> = std::result::Result<T, GradientLSSError>;

/// Errors that can occur in GradientLSS operations.
#[derive(Error, Debug)]
pub enum GradientLSSError {
    /// An error from the backend.
    #[error("Backend error: {0}")]
    BackendError(String),
    /// An IO error.
    #[error("IO error: {0}")]
    IoError(String),
    /// A serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),
    /// The model has not been trained yet.
    #[error("Model is not trained yet")]
    ModelNotTrained,
    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// Invalid input data.
    #[error("Invalid input data: {0}")]
    InvalidInput(String),
    /// Shape mismatch in arrays.
    #[error("Shape mismatch: expected {expected_shape}, got {actual_shape}")]
    ShapeMismatch {
        expected_shape: String,
        actual_shape: String,
    },
    /// An error occurred during hyperparameter optimization.
    #[error("Hyperparameter optimization error: {0}")]
    HyperOptError(String),
    /// A generic error from the argmin crate.
    #[error("Argmin error: {0}")]
    ArgminError(String),
    /// Invalid prediction type for the requested operation.
    #[error("Invalid prediction type for the requested operation")]
    InvalidPredictionType,
    /// An error occurred during plotting.
    #[error("Plotting error: {0}")]
    PlottingError(String),
}

impl From<argmin::core::Error> for GradientLSSError {
    fn from(err: argmin::core::Error) -> Self {
        GradientLSSError::ArgminError(err.to_string())
    }
}

impl From<ShapeError> for GradientLSSError {
    fn from(err: ShapeError) -> Self {
        GradientLSSError::ShapeMismatch {
            expected_shape: "unknown".to_string(),
            actual_shape: err.to_string(),
        }
    }
}
