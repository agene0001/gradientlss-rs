//! Interpretability module for SHAP values and model explanations.
//!
//! This module provides functionality to export model predictions and data
//! in formats compatible with SHAP analysis tools (Python's shap library).

use crate::backend::{Backend, FeatureImportanceType};
use crate::error::{GradientLSSError, Result};
use crate::model::{GradientLSS, PredType};
use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SHAP export data structure.
///
/// This structure contains all the data needed to compute SHAP values
/// using external libraries like Python's shap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapExportData {
    /// Feature matrix (n_samples x n_features).
    pub features: Vec<Vec<f64>>,
    /// Predicted distributional parameters (n_samples x n_params).
    pub predictions: Vec<Vec<f64>>,
    /// Feature names (if provided).
    pub feature_names: Option<Vec<String>>,
    /// Parameter names from the distribution.
    pub param_names: Vec<String>,
    /// Feature importance scores per parameter.
    pub feature_importance: Option<HashMap<String, Vec<f64>>>,
    /// Model type (XGBoost, LightGBM, etc.).
    pub model_type: String,
    /// Number of distributional parameters.
    pub n_params: usize,
}

impl ShapExportData {
    /// Export to JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| GradientLSSError::SerializationError(e.to_string()))
    }

    /// Export to JSON file.
    pub fn to_json_file(&self, path: &str) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(|e| GradientLSSError::IoError(e.to_string()))
    }

    /// Load from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| GradientLSSError::SerializationError(e.to_string()))
    }

    /// Load from JSON file.
    pub fn from_json_file(path: &str) -> Result<Self> {
        let json =
            std::fs::read_to_string(path).map_err(|e| GradientLSSError::IoError(e.to_string()))?;
        Self::from_json(&json)
    }
}

/// Partial dependence data for a single feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialDependence {
    /// Feature name or index.
    pub feature: String,
    /// Parameter name this PDP is for.
    pub parameter: String,
    /// Feature values (x-axis).
    pub feature_values: Vec<f64>,
    /// Average predicted parameter values (y-axis).
    pub predictions: Vec<f64>,
    /// Standard deviation of predictions (for confidence bands).
    pub std_dev: Option<Vec<f64>>,
}

/// Individual Conditional Expectation (ICE) data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcePlot {
    /// Feature name or index.
    pub feature: String,
    /// Parameter name this ICE plot is for.
    pub parameter: String,
    /// Feature values (x-axis).
    pub feature_values: Vec<f64>,
    /// Predictions for each sample (n_samples x n_values).
    pub predictions: Vec<Vec<f64>>,
}

impl<B: Backend> GradientLSS<B> {
    /// Export data for SHAP analysis.
    ///
    /// This creates a data structure that can be serialized and used with
    /// Python's shap library for model interpretation.
    ///
    /// # Arguments
    /// * `features` - Feature matrix to explain
    /// * `feature_names` - Optional feature names
    ///
    /// # Returns
    /// `ShapExportData` containing predictions and metadata.
    pub fn export_for_shap(
        &self,
        features: &ArrayView2<f64>,
        feature_names: Option<Vec<String>>,
    ) -> Result<ShapExportData> {
        // Get predictions
        let pred_output = self.predict(features, PredType::Parameters, 0, &[], 0)?;
        let predictions = match pred_output {
            crate::backend::PredictionOutput::Parameters(p) => p,
            _ => return Err(GradientLSSError::InvalidPredictionType),
        };

        // Get feature importance if available
        let feature_importance = self
            .feature_importance(FeatureImportanceType::Gain, feature_names.clone())
            .ok()
            .map(|fi| {
                let mut map = HashMap::new();
                let param_names = self.param_names();
                for (param_idx, param_name) in param_names.iter().enumerate() {
                    let scores: Vec<f64> = fi
                        .scores
                        .column(param_idx.min(fi.scores.ncols() - 1))
                        .to_vec();
                    map.insert(param_name.clone(), scores);
                }
                map
            });

        // Convert features to Vec<Vec<f64>>
        let features_vec: Vec<Vec<f64>> = features
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        // Convert predictions to Vec<Vec<f64>>
        let predictions_vec: Vec<Vec<f64>> = predictions
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        Ok(ShapExportData {
            features: features_vec,
            predictions: predictions_vec,
            feature_names,
            param_names: self.param_names(),
            feature_importance,
            model_type: B::name().to_string(),
            n_params: self.n_params(),
        })
    }

    /// Compute partial dependence for a feature.
    ///
    /// # Arguments
    /// * `features` - Base feature matrix
    /// * `feature_idx` - Index of feature to compute PDP for
    /// * `param_idx` - Index of distributional parameter
    /// * `grid_size` - Number of points in the grid (default: 50)
    /// * `feature_name` - Optional name for the feature
    /// * `param_name` - Optional name for the parameter
    pub fn partial_dependence(
        &self,
        features: &ArrayView2<f64>,
        feature_idx: usize,
        param_idx: usize,
        grid_size: Option<usize>,
        feature_name: Option<String>,
        param_name: Option<String>,
    ) -> Result<PartialDependence> {
        let grid_size = grid_size.unwrap_or(50);
        let n_samples = features.nrows();

        // Get feature range
        let feature_col = features.column(feature_idx);
        let min_val = feature_col.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = feature_col
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Create grid
        let step = (max_val - min_val) / (grid_size as f64 - 1.0);
        let grid: Vec<f64> = (0..grid_size).map(|i| min_val + i as f64 * step).collect();

        // Compute average prediction for each grid value
        let mut avg_predictions = Vec::with_capacity(grid_size);
        let mut std_predictions = Vec::with_capacity(grid_size);

        for &grid_val in &grid {
            // Create modified features
            let mut modified = features.to_owned();
            for i in 0..n_samples {
                modified[[i, feature_idx]] = grid_val;
            }

            // Get predictions
            let pred_output = self.predict(&modified.view(), PredType::Parameters, 0, &[], 0)?;
            let predictions = match pred_output {
                crate::backend::PredictionOutput::Parameters(p) => p,
                _ => return Err(GradientLSSError::InvalidPredictionType),
            };

            // Compute mean and std for this parameter
            let param_preds: Vec<f64> = predictions.column(param_idx).to_vec();
            let mean = param_preds.iter().sum::<f64>() / param_preds.len() as f64;
            let variance = param_preds.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / param_preds.len() as f64;
            let std = variance.sqrt();

            avg_predictions.push(mean);
            std_predictions.push(std);
        }

        let feature_str = feature_name.unwrap_or_else(|| format!("feature_{}", feature_idx));
        let param_str = param_name.unwrap_or_else(|| {
            self.param_names()
                .get(param_idx)
                .cloned()
                .unwrap_or_else(|| format!("param_{}", param_idx))
        });

        Ok(PartialDependence {
            feature: feature_str,
            parameter: param_str,
            feature_values: grid,
            predictions: avg_predictions,
            std_dev: Some(std_predictions),
        })
    }

    /// Compute Individual Conditional Expectation (ICE) curves.
    ///
    /// Similar to partial dependence but shows individual curves for each sample.
    pub fn ice_curves(
        &self,
        features: &ArrayView2<f64>,
        feature_idx: usize,
        param_idx: usize,
        grid_size: Option<usize>,
        feature_name: Option<String>,
        param_name: Option<String>,
    ) -> Result<IcePlot> {
        let grid_size = grid_size.unwrap_or(50);
        let n_samples = features.nrows();

        // Get feature range
        let feature_col = features.column(feature_idx);
        let min_val = feature_col.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = feature_col
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Create grid
        let step = (max_val - min_val) / (grid_size as f64 - 1.0);
        let grid: Vec<f64> = (0..grid_size).map(|i| min_val + i as f64 * step).collect();

        // Compute predictions for each sample at each grid value
        let mut ice_predictions: Vec<Vec<f64>> = vec![Vec::with_capacity(grid_size); n_samples];

        for &grid_val in &grid {
            // Create modified features
            let mut modified = features.to_owned();
            for i in 0..n_samples {
                modified[[i, feature_idx]] = grid_val;
            }

            // Get predictions
            let pred_output = self.predict(&modified.view(), PredType::Parameters, 0, &[], 0)?;
            let predictions = match pred_output {
                crate::backend::PredictionOutput::Parameters(p) => p,
                _ => return Err(GradientLSSError::InvalidPredictionType),
            };

            // Store prediction for each sample
            for (i, pred) in predictions.column(param_idx).iter().enumerate() {
                ice_predictions[i].push(*pred);
            }
        }

        let feature_str = feature_name.unwrap_or_else(|| format!("feature_{}", feature_idx));
        let param_str = param_name.unwrap_or_else(|| {
            self.param_names()
                .get(param_idx)
                .cloned()
                .unwrap_or_else(|| format!("param_{}", param_idx))
        });

        Ok(IcePlot {
            feature: feature_str,
            parameter: param_str,
            feature_values: grid,
            predictions: ice_predictions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shap_export_serialization() {
        let data = ShapExportData {
            features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            predictions: vec![vec![0.5, 0.1], vec![0.6, 0.2]],
            feature_names: Some(vec!["x1".to_string(), "x2".to_string()]),
            param_names: vec!["loc".to_string(), "scale".to_string()],
            feature_importance: None,
            model_type: "LightGBM".to_string(),
            n_params: 2,
        };

        let json = data.to_json().unwrap();
        let parsed = ShapExportData::from_json(&json).unwrap();

        assert_eq!(parsed.features.len(), 2);
        assert_eq!(parsed.param_names.len(), 2);
    }
}
