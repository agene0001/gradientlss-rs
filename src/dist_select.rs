//! Distribution selection functionality.
//!
//! This module provides functions to select the best distribution from a set of
//! candidates based on how well they fit the target data.

use crate::distributions::Distribution;
use crate::error::Result;
use crate::types::ResponseData;
use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;

/// Selects the best distribution from a list of candidates based on NLL.
///
/// This function iterates through a list of candidate distributions, fits each
/// to the target data to find the optimal unconditional parameters, and then
/// ranks them based on the resulting negative log-likelihood (NLL).
///
/// # Arguments
///
/// * `target` - A 1D array of target values.
/// * `candidate_distributions` - A vector of distribution instances to evaluate.
/// * `max_iter` - The maximum number of iterations for the optimization of each distribution.
///
/// # Returns
///
/// A `Result` containing a vector of tuples, where each tuple contains the
/// distribution's name and its corresponding NLL, sorted from best (lowest NLL)
/// to worst.
///
pub fn dist_select<'a>(
    target: &'a ArrayView1<'a, f64>,
    candidate_distributions: Vec<Box<dyn Distribution>>,
    max_iter: usize,
) -> Result<Vec<(String, f64)>> {
    let response_data = ResponseData::Univariate(target);

    let results: Vec<_> = candidate_distributions
        .into_par_iter()
        .map(|dist| {
            let dist_name = dist.param_names().join("_");
            let result = dist.calculate_start_values(&response_data, max_iter);
            (dist_name, result)
        })
        .collect();

    let mut successful_fits: Vec<(String, f64)> = Vec::new();
    for (dist_name, result) in results {
        match result {
            Ok((nll, _)) => successful_fits.push((dist_name, nll)),
            Err(e) => {
                // You might want to log this error instead of just printing it
                eprintln!("Could not fit distribution {}: {}", dist_name, e);
            }
        }
    }

    // Sort by NLL, lowest to highest
    successful_fits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(successful_fits)
}

/// Extended distribution selection result with fitted parameters.
pub struct DistSelectResult {
    /// Distribution name (derived from parameter names)
    pub name: String,
    /// Negative log-likelihood
    pub nll: f64,
    /// Fitted parameter values
    pub fitted_params: Array1<f64>,
    /// The distribution (boxed)
    pub distribution: Box<dyn Distribution>,
}

impl std::fmt::Debug for DistSelectResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistSelectResult")
            .field("name", &self.name)
            .field("nll", &self.nll)
            .field("fitted_params", &self.fitted_params)
            .field("distribution", &"<Distribution>")
            .finish()
    }
}

impl Clone for DistSelectResult {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            nll: self.nll,
            fitted_params: self.fitted_params.clone(),
            distribution: self.distribution.clone_box(),
        }
    }
}

/// Selects the best distribution with full results including fitted parameters.
///
/// This is an extended version of `dist_select` that also returns the fitted
/// parameters for each distribution, enabling sampling and density comparisons.
///
/// # Arguments
///
/// * `target` - A 1D array of target values.
/// * `candidate_distributions` - A vector of distribution instances to evaluate.
/// * `max_iter` - The maximum number of iterations for the optimization.
///
/// # Returns
///
/// A `Result` containing a vector of `DistSelectResult` sorted by NLL.
pub fn dist_select_with_params<'a>(
    target: &'a ArrayView1<'a, f64>,
    candidate_distributions: Vec<Box<dyn Distribution>>,
    max_iter: usize,
) -> Result<Vec<DistSelectResult>> {
    let response_data = ResponseData::Univariate(target);

    let results: Vec<_> = candidate_distributions
        .into_par_iter()
        .map(|dist| {
            let dist_name = dist.param_names().join("_");
            let result = dist.calculate_start_values(&response_data, max_iter);
            (dist_name, result, dist)
        })
        .collect();

    let mut successful_fits: Vec<DistSelectResult> = Vec::new();
    for (dist_name, result, dist) in results {
        match result {
            Ok((nll, params)) => successful_fits.push(DistSelectResult {
                name: dist_name,
                nll,
                fitted_params: params,
                distribution: dist,
            }),
            Err(e) => {
                eprintln!("Could not fit distribution {}: {}", dist_name, e);
            }
        }
    }

    // Sort by NLL, lowest to highest
    successful_fits.sort_by(|a, b| {
        a.nll
            .partial_cmp(&b.nll)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(successful_fits)
}

/// Plot distribution selection results with density comparisons.
///
/// This function creates visualizations comparing the fitted distributions
/// against the actual data, similar to XGBoostLSS/LightGBMLSS.
///
/// # Arguments
///
/// * `target` - The target data
/// * `results` - Distribution selection results from `dist_select_with_params`
/// * `output_dir` - Directory to save the plots
/// * `n_samples` - Number of samples to draw for density estimation (default: 1000)
/// * `top_n` - Number of top distributions to plot (default: 5)
///
/// # Returns
///
/// A list of paths to the generated plot files.
#[cfg(feature = "plotting")]
pub fn plot_dist_select_densities(
    target: &ArrayView1<f64>,
    results: &[DistSelectResult],
    output_dir: &str,
    n_samples: Option<usize>,
    top_n: Option<usize>,
) -> Result<Vec<String>> {
    use crate::plotting::{PlotConfig, plot_density_comparison, plot_dist_select};
    use ndarray::Array2;
    use std::fs;

    let n_samples = n_samples.unwrap_or(1000);
    let top_n = top_n.unwrap_or(5).min(results.len());

    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir)
        .map_err(|e| crate::error::GradientLSSError::IoError(e.to_string()))?;

    let mut plot_paths = Vec::new();

    // Plot the ranking bar chart
    let ranking_path = format!("{}/dist_select_ranking.png", output_dir);
    let ranking_data: Vec<(String, f64)> = results
        .iter()
        .take(top_n)
        .map(|r| (r.name.clone(), r.nll))
        .collect();
    plot_dist_select(&ranking_data, &ranking_path, &PlotConfig::default())?;
    plot_paths.push(ranking_path);

    // Plot density comparisons for top distributions
    for (i, result) in results.iter().take(top_n).enumerate() {
        // Create parameter matrix for sampling (1 row with fitted params)
        let n_params = result.fitted_params.len();
        let mut params = Array2::zeros((1, n_params));
        for (j, &val) in result.fitted_params.iter().enumerate() {
            params[[0, j]] = val;
        }

        // Transform parameters
        let transformed = result.distribution.transform_params(&params.view());

        // Sample from the fitted distribution
        let samples = result
            .distribution
            .sample(&transformed.view(), n_samples, 42);

        // Extract samples for the first (and only) observation
        let fitted_samples: Array1<f64> = samples.column(0).to_owned();

        // Plot density comparison
        let density_path = format!("{}/density_{:02}_{}.png", output_dir, i + 1, result.name);
        let config = PlotConfig {
            title: Some(format!("{} (NLL: {:.4})", result.name, result.nll)),
            ..PlotConfig::default()
        };
        plot_density_comparison(
            target,
            &fitted_samples.view(),
            &result.name,
            &density_path,
            &config,
            Some(30),
        )?;
        plot_paths.push(density_path);
    }

    Ok(plot_paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Gaussian, StudentT};

    #[test]
    fn test_dist_select() {
        let target = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let candidates: Vec<Box<dyn Distribution>> = vec![
            Box::new(Gaussian::new(
                crate::distributions::base::Stabilization::None,
                crate::utils::ResponseFn::Exp,
                crate::distributions::base::LossFn::Nll,
                true,
            )),
            Box::new(StudentT::default()),
        ];

        let result = dist_select(&target.view(), candidates, 100).unwrap();
        assert_eq!(result.len(), 2);
        // Check that the gaussian distribution is in the results
        assert!(result.iter().any(|(name, _)| name == "loc_scale"));
    }

    #[test]
    fn test_dist_select_with_params() {
        let target = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let candidates: Vec<Box<dyn Distribution>> = vec![
            Box::new(Gaussian::new(
                crate::distributions::base::Stabilization::None,
                crate::utils::ResponseFn::Exp,
                crate::distributions::base::LossFn::Nll,
                true,
            )),
            Box::new(StudentT::default()),
        ];

        let result = dist_select_with_params(&target.view(), candidates, 100).unwrap();
        assert_eq!(result.len(), 2);

        // Check that fitted parameters are available
        for r in &result {
            assert!(!r.fitted_params.is_empty());
            assert!(r.nll.is_finite());
        }
    }
}
