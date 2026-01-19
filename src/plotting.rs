//! Plotting functionality for GradientLSS.
//!
//! This module provides visualization capabilities similar to XGBoostLSS/LightGBMLSS,
//! including:
//! - Feature importance plots
//! - Partial dependence plots
//! - Distribution selection comparison plots
//! - SHAP summary plots (via data export)
//!
//! Requires the `plotting` feature to be enabled.

use crate::backend::{Backend, FeatureImportance, FeatureImportanceType};
use crate::error::{GradientLSSError, Result};
use crate::interpretability::PartialDependence;
use crate::model::GradientLSS;
use ndarray::{ArrayView1, ArrayView2};
use plotters::prelude::*;

/// Plot configuration options.
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Width of the plot in pixels.
    pub width: u32,
    /// Height of the plot in pixels.
    pub height: u32,
    /// Title of the plot.
    pub title: Option<String>,
    /// X-axis label.
    pub x_label: Option<String>,
    /// Y-axis label.
    pub y_label: Option<String>,
    /// Font size for labels.
    pub font_size: u32,
    /// Color palette name.
    pub palette: ColorPalette,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: None,
            x_label: None,
            y_label: None,
            font_size: 16,
            palette: ColorPalette::Default,
        }
    }
}

/// Color palettes for plots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPalette {
    /// Default blue-based palette.
    Default,
    /// Viridis-like palette.
    Viridis,
    /// Colorblind-friendly palette.
    ColorBlind,
}

impl ColorPalette {
    /// Get colors from the palette.
    pub fn colors(&self, n: usize) -> Vec<RGBColor> {
        match self {
            ColorPalette::Default => {
                let base_colors = vec![
                    RGBColor(31, 119, 180),  // Blue
                    RGBColor(255, 127, 14),  // Orange
                    RGBColor(44, 160, 44),   // Green
                    RGBColor(214, 39, 40),   // Red
                    RGBColor(148, 103, 189), // Purple
                    RGBColor(140, 86, 75),   // Brown
                    RGBColor(227, 119, 194), // Pink
                    RGBColor(127, 127, 127), // Gray
                    RGBColor(188, 189, 34),  // Olive
                    RGBColor(23, 190, 207),  // Cyan
                ];
                (0..n).map(|i| base_colors[i % base_colors.len()]).collect()
            }
            ColorPalette::Viridis => {
                // Simplified viridis-like colors
                let base_colors = vec![
                    RGBColor(68, 1, 84),
                    RGBColor(72, 40, 120),
                    RGBColor(62, 74, 137),
                    RGBColor(49, 104, 142),
                    RGBColor(38, 130, 142),
                    RGBColor(31, 158, 137),
                    RGBColor(53, 183, 121),
                    RGBColor(109, 205, 89),
                    RGBColor(180, 222, 44),
                    RGBColor(253, 231, 37),
                ];
                (0..n).map(|i| base_colors[i % base_colors.len()]).collect()
            }
            ColorPalette::ColorBlind => {
                let base_colors = vec![
                    RGBColor(0, 114, 178),   // Blue
                    RGBColor(230, 159, 0),   // Orange
                    RGBColor(0, 158, 115),   // Green
                    RGBColor(204, 121, 167), // Pink
                    RGBColor(86, 180, 233),  // Sky blue
                    RGBColor(213, 94, 0),    // Vermillion
                    RGBColor(240, 228, 66),  // Yellow
                ];
                (0..n).map(|i| base_colors[i % base_colors.len()]).collect()
            }
        }
    }
}

/// Plot feature importance as a horizontal bar chart.
///
/// # Arguments
/// * `importance` - Feature importance data from the model
/// * `param_idx` - Which distributional parameter to plot (None for aggregated)
/// * `max_features` - Maximum number of features to show (default: 20)
/// * `path` - Output file path (PNG format)
/// * `config` - Plot configuration
pub fn plot_feature_importance(
    importance: &FeatureImportance,
    param_idx: Option<usize>,
    max_features: Option<usize>,
    path: &str,
    config: &PlotConfig,
) -> Result<()> {
    let max_features = max_features.unwrap_or(20);

    // Get scores for the specified parameter or aggregate
    let scores: Vec<f64> = if let Some(idx) = param_idx {
        if idx >= importance.scores.ncols() {
            return Err(GradientLSSError::InvalidParameter(format!(
                "Parameter index {} out of range",
                idx
            )));
        }
        importance.scores.column(idx).to_vec()
    } else {
        // Aggregate across all parameters (mean)
        importance
            .scores
            .rows()
            .into_iter()
            .map(|row| row.mean().unwrap_or(0.0))
            .collect()
    };

    // Get feature names
    let feature_names: Vec<String> = importance.feature_names.clone().unwrap_or_else(|| {
        (0..scores.len())
            .map(|i| format!("Feature {}", i))
            .collect()
    });

    // Sort by importance and take top N
    let mut indexed_scores: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed_scores.truncate(max_features);

    // Prepare data for plotting
    let sorted_names: Vec<&str> = indexed_scores
        .iter()
        .map(|(i, _)| feature_names[*i].as_str())
        .collect();
    let sorted_scores: Vec<f64> = indexed_scores.iter().map(|(_, s)| *s).collect();

    // Create the plot
    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let title = config.title.clone().unwrap_or_else(|| match param_idx {
        Some(idx) => format!("Feature Importance (Parameter {})", idx),
        None => "Feature Importance (Aggregated)".to_string(),
    });

    let max_score = sorted_scores.iter().cloned().fold(0.0, f64::max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", config.font_size).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(150)
        .build_cartesian_2d(0.0..max_score, (0..sorted_names.len()).into_segmented())
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    chart
        .configure_mesh()
        .disable_y_mesh()
        .x_desc(config.x_label.as_deref().unwrap_or("Importance"))
        .y_desc(config.y_label.as_deref().unwrap_or("Feature"))
        .y_label_formatter(&|y| {
            if let SegmentValue::CenterOf(idx) = y {
                sorted_names.get(*idx).copied().unwrap_or("").to_string()
            } else {
                String::new()
            }
        })
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let colors = config.palette.colors(1);

    chart
        .draw_series(sorted_scores.iter().enumerate().map(|(i, &score)| {
            let mut bar = Rectangle::new(
                [
                    (0.0, SegmentValue::Exact(i)),
                    (score, SegmentValue::Exact(i + 1)),
                ],
                colors[0].filled(),
            );
            bar.set_margin(2, 2, 0, 0);
            bar
        }))
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    root.present()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    Ok(())
}

/// Plot partial dependence for a feature.
///
/// # Arguments
/// * `pdp` - Partial dependence data
/// * `show_confidence` - Whether to show confidence bands (±1 std)
/// * `path` - Output file path (PNG format)
/// * `config` - Plot configuration
pub fn plot_partial_dependence(
    pdp: &PartialDependence,
    show_confidence: bool,
    path: &str,
    config: &PlotConfig,
) -> Result<()> {
    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let title = config
        .title
        .clone()
        .unwrap_or_else(|| format!("Partial Dependence: {} on {}", pdp.parameter, pdp.feature));

    let x_min = pdp
        .feature_values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let x_max = pdp
        .feature_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = pdp
        .predictions
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let y_max = pdp
        .predictions
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_margin = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", config.font_size).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, (y_min - y_margin)..(y_max + y_margin))
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(config.x_label.as_deref().unwrap_or(&pdp.feature))
        .y_desc(config.y_label.as_deref().unwrap_or(&pdp.parameter))
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    // Draw confidence bands if available
    if show_confidence {
        if let Some(ref std_dev) = pdp.std_dev {
            let upper: Vec<(f64, f64)> = pdp
                .feature_values
                .iter()
                .zip(pdp.predictions.iter())
                .zip(std_dev.iter())
                .map(|((&x, &y), &s)| (x, y + s))
                .collect();
            let lower: Vec<(f64, f64)> = pdp
                .feature_values
                .iter()
                .zip(pdp.predictions.iter())
                .zip(std_dev.iter())
                .map(|((&x, &y), &s)| (x, y - s))
                .collect();

            chart
                .draw_series(AreaSeries::new(
                    upper.iter().chain(lower.iter().rev()).cloned(),
                    0.0,
                    RGBColor(31, 119, 180).mix(0.2),
                ))
                .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;
        }
    }

    // Draw main line
    let points: Vec<(f64, f64)> = pdp
        .feature_values
        .iter()
        .zip(pdp.predictions.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    chart
        .draw_series(LineSeries::new(points, &RGBColor(31, 119, 180)))
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    root.present()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    Ok(())
}

/// Plot distribution selection results as a bar chart.
///
/// # Arguments
/// * `results` - Distribution selection results (name, NLL) pairs
/// * `path` - Output file path (PNG format)
/// * `config` - Plot configuration
pub fn plot_dist_select(results: &[(String, f64)], path: &str, config: &PlotConfig) -> Result<()> {
    if results.is_empty() {
        return Err(GradientLSSError::InvalidParameter(
            "No distribution results to plot".to_string(),
        ));
    }

    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let title = config
        .title
        .clone()
        .unwrap_or_else(|| "Distribution Selection (NLL)".to_string());

    let names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
    let scores: Vec<f64> = results.iter().map(|(_, s)| *s).collect();
    let max_score = scores.iter().cloned().fold(0.0, f64::max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", config.font_size).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0..max_score, (0..names.len()).into_segmented())
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    chart
        .configure_mesh()
        .disable_y_mesh()
        .x_desc(
            config
                .x_label
                .as_deref()
                .unwrap_or("Negative Log-Likelihood"),
        )
        .y_desc(config.y_label.as_deref().unwrap_or("Distribution"))
        .y_label_formatter(&|y| {
            if let SegmentValue::CenterOf(idx) = y {
                names.get(*idx).copied().unwrap_or("").to_string()
            } else {
                String::new()
            }
        })
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    // Use gradient colors from best (green) to worst (red)
    let n = scores.len();
    let colors: Vec<RGBColor> = (0..n)
        .map(|i| {
            let ratio = i as f64 / (n - 1).max(1) as f64;
            RGBColor(
                (44.0 + ratio * (214.0 - 44.0)) as u8, // Green to red
                (160.0 - ratio * (160.0 - 39.0)) as u8,
                (44.0 - ratio * (44.0 - 40.0)) as u8,
            )
        })
        .collect();

    chart
        .draw_series(scores.iter().enumerate().map(|(i, &score)| {
            let mut bar = Rectangle::new(
                [
                    (0.0, SegmentValue::Exact(i)),
                    (score, SegmentValue::Exact(i + 1)),
                ],
                colors[i].filled(),
            );
            bar.set_margin(2, 2, 0, 0);
            bar
        }))
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    root.present()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    Ok(())
}

/// Plot density comparison between actual data and fitted distribution.
///
/// # Arguments
/// * `actual` - Actual data values
/// * `fitted_samples` - Samples from the fitted distribution
/// * `dist_name` - Name of the distribution for the title
/// * `path` - Output file path (PNG format)
/// * `config` - Plot configuration
/// * `n_bins` - Number of bins for the histogram (default: 30)
pub fn plot_density_comparison(
    actual: &ArrayView1<f64>,
    fitted_samples: &ArrayView1<f64>,
    dist_name: &str,
    path: &str,
    config: &PlotConfig,
    n_bins: Option<usize>,
) -> Result<()> {
    let n_bins = n_bins.unwrap_or(30);

    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let title = config
        .title
        .clone()
        .unwrap_or_else(|| format!("Density Comparison: {}", dist_name));

    // Compute histogram for both actual and fitted
    let all_values: Vec<f64> = actual
        .iter()
        .chain(fitted_samples.iter())
        .copied()
        .collect();
    let min_val = all_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (max_val - min_val) / n_bins as f64;

    let compute_histogram = |data: &ArrayView1<f64>| -> Vec<(f64, f64)> {
        let mut counts = vec![0usize; n_bins];
        for &v in data.iter() {
            let bin = ((v - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            counts[bin] += 1;
        }
        // Normalize to density
        let total = data.len() as f64;
        counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let x = min_val + (i as f64 + 0.5) * bin_width;
                let density = c as f64 / (total * bin_width);
                (x, density)
            })
            .collect()
    };

    let actual_hist = compute_histogram(actual);
    let fitted_hist = compute_histogram(fitted_samples);

    let max_density = actual_hist
        .iter()
        .chain(fitted_hist.iter())
        .map(|(_, d)| *d)
        .fold(0.0, f64::max)
        * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", config.font_size).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min_val..max_val, 0.0..max_density)
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(config.x_label.as_deref().unwrap_or("Value"))
        .y_desc(config.y_label.as_deref().unwrap_or("Density"))
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    // Draw actual data histogram
    chart
        .draw_series(actual_hist.iter().map(|&(x, d)| {
            Rectangle::new(
                [(x - bin_width / 2.0, 0.0), (x + bin_width / 2.0, d)],
                RGBColor(31, 119, 180).mix(0.5).filled(),
            )
        }))
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?
        .label("Actual")
        .legend(|(x, y)| {
            Rectangle::new(
                [(x, y - 5), (x + 20, y + 5)],
                RGBColor(31, 119, 180).filled(),
            )
        });

    // Draw fitted distribution as a line
    chart
        .draw_series(LineSeries::new(
            fitted_hist.iter().cloned(),
            RGBColor(255, 127, 14).stroke_width(2),
        ))
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?
        .label("Fitted")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                RGBColor(255, 127, 14).stroke_width(2),
            )
        });

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    root.present()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    Ok(())
}

/// Plot multiple partial dependence curves for different parameters.
///
/// # Arguments
/// * `pdps` - Vector of partial dependence data for different parameters
/// * `path` - Output file path (PNG format)
/// * `config` - Plot configuration
pub fn plot_partial_dependence_multi(
    pdps: &[PartialDependence],
    path: &str,
    config: &PlotConfig,
) -> Result<()> {
    if pdps.is_empty() {
        return Err(GradientLSSError::InvalidParameter(
            "No partial dependence data to plot".to_string(),
        ));
    }

    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let title = config
        .title
        .clone()
        .unwrap_or_else(|| format!("Partial Dependence: {}", pdps[0].feature));

    // Find global ranges
    let x_min = pdps
        .iter()
        .flat_map(|p| p.feature_values.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let x_max = pdps
        .iter()
        .flat_map(|p| p.feature_values.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = pdps
        .iter()
        .flat_map(|p| p.predictions.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let y_max = pdps
        .iter()
        .flat_map(|p| p.predictions.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_margin = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", config.font_size).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, (y_min - y_margin)..(y_max + y_margin))
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(config.x_label.as_deref().unwrap_or(&pdps[0].feature))
        .y_desc(config.y_label.as_deref().unwrap_or("Predicted Value"))
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let colors = config.palette.colors(pdps.len());

    for (i, pdp) in pdps.iter().enumerate() {
        let points: Vec<(f64, f64)> = pdp
            .feature_values
            .iter()
            .zip(pdp.predictions.iter())
            .map(|(&x, &y)| (x, y))
            .collect();

        let color = colors[i];
        chart
            .draw_series(LineSeries::new(points, color.stroke_width(2)))
            .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?
            .label(&pdp.parameter)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    root.present()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    Ok(())
}

/// Expectile prediction data for plotting.
#[derive(Debug, Clone)]
pub struct ExpectilePrediction {
    /// Feature name.
    pub feature: String,
    /// Feature values (x-axis).
    pub feature_values: Vec<f64>,
    /// Expectile values for each tau level.
    /// Structure: expectiles[tau_idx] = predictions for that expectile level
    pub expectiles: Vec<Vec<f64>>,
    /// Expectile levels (tau values, e.g., [0.1, 0.5, 0.9]).
    pub tau_levels: Vec<f64>,
}

/// Plot expectile predictions for a feature.
///
/// This creates a visualization similar to XGBoostLSS/LightGBMLSS's `expectile_plot()`,
/// showing multiple expectile levels as separate lines across feature values.
///
/// # Arguments
/// * `expectile_data` - Expectile prediction data
/// * `path` - Output file path (PNG format)
/// * `config` - Plot configuration
/// * `show_median` - Whether to highlight the median (tau=0.5) line
pub fn plot_expectiles(
    expectile_data: &ExpectilePrediction,
    path: &str,
    config: &PlotConfig,
    show_median: bool,
) -> Result<()> {
    if expectile_data.expectiles.is_empty() || expectile_data.tau_levels.is_empty() {
        return Err(GradientLSSError::InvalidParameter(
            "No expectile data to plot".to_string(),
        ));
    }

    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let title = config
        .title
        .clone()
        .unwrap_or_else(|| format!("Expectile Plot: {}", expectile_data.feature));

    // Find ranges
    let x_min = expectile_data
        .feature_values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let x_max = expectile_data
        .feature_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = expectile_data
        .expectiles
        .iter()
        .flat_map(|e| e.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let y_max = expectile_data
        .expectiles
        .iter()
        .flat_map(|e| e.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_margin = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", config.font_size).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, (y_min - y_margin)..(y_max + y_margin))
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(config.x_label.as_deref().unwrap_or(&expectile_data.feature))
        .y_desc(config.y_label.as_deref().unwrap_or("Expectile Value"))
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let colors = config.palette.colors(expectile_data.tau_levels.len());

    // Find median index if show_median is true
    let median_idx = if show_median {
        expectile_data
            .tau_levels
            .iter()
            .position(|&t| (t - 0.5).abs() < 0.01)
    } else {
        None
    };

    for (i, (expectile_vals, &tau)) in expectile_data
        .expectiles
        .iter()
        .zip(expectile_data.tau_levels.iter())
        .enumerate()
    {
        let points: Vec<(f64, f64)> = expectile_data
            .feature_values
            .iter()
            .zip(expectile_vals.iter())
            .map(|(&x, &y)| (x, y))
            .collect();

        let color = colors[i];
        let stroke_width = if Some(i) == median_idx { 3 } else { 2 };
        let label = format!("τ = {:.2}", tau);

        chart
            .draw_series(LineSeries::new(points, color.stroke_width(stroke_width)))
            .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(stroke_width))
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    root.present()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    Ok(())
}

/// Plot expectiles with confidence bands.
///
/// This version shows the range between symmetric expectile pairs
/// (e.g., 0.1-0.9, 0.25-0.75) as shaded regions.
///
/// # Arguments
/// * `expectile_data` - Expectile prediction data
/// * `path` - Output file path (PNG format)
/// * `config` - Plot configuration
pub fn plot_expectiles_with_bands(
    expectile_data: &ExpectilePrediction,
    path: &str,
    config: &PlotConfig,
) -> Result<()> {
    if expectile_data.expectiles.is_empty() || expectile_data.tau_levels.is_empty() {
        return Err(GradientLSSError::InvalidParameter(
            "No expectile data to plot".to_string(),
        ));
    }

    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    let title = config
        .title
        .clone()
        .unwrap_or_else(|| format!("Expectile Plot: {}", expectile_data.feature));

    // Find ranges
    let x_min = expectile_data
        .feature_values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let x_max = expectile_data
        .feature_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = expectile_data
        .expectiles
        .iter()
        .flat_map(|e| e.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let y_max = expectile_data
        .expectiles
        .iter()
        .flat_map(|e| e.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_margin = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", config.font_size).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, (y_min - y_margin)..(y_max + y_margin))
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(config.x_label.as_deref().unwrap_or(&expectile_data.feature))
        .y_desc(config.y_label.as_deref().unwrap_or("Expectile Value"))
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    // Find symmetric pairs and draw bands
    let n_levels = expectile_data.tau_levels.len();
    let base_color = RGBColor(31, 119, 180);

    // Draw bands for symmetric pairs (outermost to innermost)
    for i in 0..n_levels / 2 {
        let lower_idx = i;
        let upper_idx = n_levels - 1 - i;

        if lower_idx >= upper_idx {
            break;
        }

        let lower_tau = expectile_data.tau_levels[lower_idx];
        let upper_tau = expectile_data.tau_levels[upper_idx];

        // Check if they're symmetric around 0.5
        if ((1.0 - lower_tau) - upper_tau).abs() < 0.01 {
            let alpha = 0.1 + 0.1 * (n_levels / 2 - i) as f64 / (n_levels / 2) as f64;

            let upper_points: Vec<(f64, f64)> = expectile_data
                .feature_values
                .iter()
                .zip(expectile_data.expectiles[upper_idx].iter())
                .map(|(&x, &y)| (x, y))
                .collect();

            let lower_points: Vec<(f64, f64)> = expectile_data
                .feature_values
                .iter()
                .zip(expectile_data.expectiles[lower_idx].iter())
                .map(|(&x, &y)| (x, y))
                .collect();

            // Create area between upper and lower
            chart
                .draw_series(AreaSeries::new(
                    upper_points
                        .iter()
                        .chain(lower_points.iter().rev())
                        .cloned(),
                    y_min - y_margin,
                    base_color.mix(alpha),
                ))
                .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;
        }
    }

    // Draw median line if present
    if let Some(median_idx) = expectile_data
        .tau_levels
        .iter()
        .position(|&t| (t - 0.5).abs() < 0.01)
    {
        let median_points: Vec<(f64, f64)> = expectile_data
            .feature_values
            .iter()
            .zip(expectile_data.expectiles[median_idx].iter())
            .map(|(&x, &y)| (x, y))
            .collect();

        chart
            .draw_series(LineSeries::new(median_points, base_color.stroke_width(2)))
            .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?
            .label("Median (τ = 0.5)")
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], base_color.stroke_width(2))
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    root.present()
        .map_err(|e| GradientLSSError::PlottingError(e.to_string()))?;

    Ok(())
}

/// Extension trait to add plotting methods directly to GradientLSS.
impl<B: Backend> GradientLSS<B> {
    /// Plot feature importance for the trained model.
    ///
    /// # Arguments
    /// * `path` - Output file path (PNG format)
    /// * `importance_type` - Type of importance (Gain, Split, Cover)
    /// * `param_idx` - Which parameter to plot (None for aggregated)
    /// * `feature_names` - Optional feature names
    /// * `config` - Plot configuration
    #[cfg(feature = "plotting")]
    pub fn plot_feature_importance(
        &self,
        path: &str,
        importance_type: FeatureImportanceType,
        param_idx: Option<usize>,
        feature_names: Option<Vec<String>>,
        config: Option<PlotConfig>,
    ) -> Result<()> {
        let importance = self.feature_importance(importance_type, feature_names)?;
        let config = config.unwrap_or_default();
        plot_feature_importance(&importance, param_idx, Some(20), path, &config)
    }

    /// Plot partial dependence for a feature.
    ///
    /// # Arguments
    /// * `features` - Feature matrix
    /// * `feature_idx` - Index of feature to plot
    /// * `param_idx` - Index of parameter to plot
    /// * `path` - Output file path
    /// * `config` - Plot configuration
    #[cfg(feature = "plotting")]
    pub fn plot_partial_dependence(
        &self,
        features: &ArrayView2<f64>,
        feature_idx: usize,
        param_idx: usize,
        path: &str,
        config: Option<PlotConfig>,
    ) -> Result<()> {
        let pdp = self.partial_dependence(features, feature_idx, param_idx, None, None, None)?;
        let config = config.unwrap_or_default();
        plot_partial_dependence(&pdp, true, path, &config)
    }

    /// Plot expectile predictions for an Expectile distribution model.
    ///
    /// This method creates a visualization showing how expectile predictions
    /// vary across feature values, similar to XGBoostLSS/LightGBMLSS's `expectile_plot()`.
    ///
    /// # Arguments
    /// * `features` - Feature matrix
    /// * `feature_idx` - Index of feature to plot
    /// * `path` - Output file path (PNG format)
    /// * `grid_size` - Number of points in the feature grid (default: 50)
    /// * `feature_name` - Optional name for the feature
    /// * `config` - Plot configuration
    /// * `show_bands` - If true, shows shaded bands between symmetric expectiles
    ///
    /// # Returns
    /// Result indicating success or failure.
    ///
    /// # Example
    /// ```ignore
    /// use gradientlss::distributions::Expectile;
    ///
    /// // Train an expectile model with tau levels [0.1, 0.25, 0.5, 0.75, 0.9]
    /// let dist = Expectile::new(vec![0.1, 0.25, 0.5, 0.75, 0.9], false, ...);
    /// let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(dist));
    /// model.train(...)?;
    ///
    /// // Plot expectiles for feature 0
    /// model.expectile_plot(
    ///     &features.view(),
    ///     0,
    ///     "expectiles.png",
    ///     Some(50),
    ///     Some("Age".to_string()),
    ///     None,
    ///     true,  // show bands
    /// )?;
    /// ```
    #[cfg(feature = "plotting")]
    pub fn expectile_plot(
        &self,
        features: &ArrayView2<f64>,
        feature_idx: usize,
        path: &str,
        grid_size: Option<usize>,
        feature_name: Option<String>,
        config: Option<PlotConfig>,
        show_bands: bool,
    ) -> Result<()> {
        use crate::model::PredType;

        let grid_size = grid_size.unwrap_or(50);
        let n_samples = features.nrows();
        let n_params = self.n_params();

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

        // For each grid value, compute average expectile predictions
        let mut expectile_predictions: Vec<Vec<f64>> =
            vec![Vec::with_capacity(grid_size); n_params];

        for &grid_val in &grid {
            // Create modified features
            let mut modified = features.to_owned();
            for i in 0..n_samples {
                modified[[i, feature_idx]] = grid_val;
            }

            // Get predictions (expectiles are returned as parameters for Expectile distribution)
            let pred_output = self.predict(&modified.view(), PredType::Expectiles, 0, &[], 0)?;
            let predictions = match pred_output {
                crate::backend::PredictionOutput::Expectiles(p) => p,
                crate::backend::PredictionOutput::Parameters(p) => p,
                _ => return Err(GradientLSSError::InvalidPredictionType),
            };

            // Compute mean prediction for each expectile level
            for param_idx in 0..n_params {
                let mean = predictions.column(param_idx).mean().unwrap_or(0.0);
                expectile_predictions[param_idx].push(mean);
            }
        }

        // Get tau levels from parameter names (Expectile uses names like "expectile_0.1")
        let tau_levels: Vec<f64> = self
            .param_names()
            .iter()
            .filter_map(|name| {
                name.strip_prefix("expectile_")
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| {
                        // Fallback: try to parse the whole name or use index-based defaults
                        name.parse::<f64>().ok()
                    })
            })
            .collect();

        // If we couldn't parse tau levels, use evenly spaced defaults
        let tau_levels = if tau_levels.len() == n_params {
            tau_levels
        } else {
            (0..n_params)
                .map(|i| (i as f64 + 1.0) / (n_params as f64 + 1.0))
                .collect()
        };

        let feature_str = feature_name.unwrap_or_else(|| format!("Feature {}", feature_idx));

        let expectile_data = ExpectilePrediction {
            feature: feature_str,
            feature_values: grid,
            expectiles: expectile_predictions,
            tau_levels,
        };

        let config = config.unwrap_or_default();

        if show_bands {
            plot_expectiles_with_bands(&expectile_data, path, &config)
        } else {
            plot_expectiles(&expectile_data, path, &config, true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::tempdir;

    #[test]
    fn test_color_palette() {
        let colors = ColorPalette::Default.colors(5);
        assert_eq!(colors.len(), 5);

        let colors = ColorPalette::Viridis.colors(3);
        assert_eq!(colors.len(), 3);

        let colors = ColorPalette::ColorBlind.colors(10);
        assert_eq!(colors.len(), 10);
    }

    #[test]
    fn test_plot_config_default() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert!(config.title.is_none());
    }

    #[test]
    fn test_plot_dist_select() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dist_select.png");

        let results = vec![
            ("Gaussian".to_string(), 1.5),
            ("StudentT".to_string(), 1.8),
            ("Gamma".to_string(), 2.1),
        ];

        let result = plot_dist_select(&results, path.to_str().unwrap(), &PlotConfig::default());
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_plot_partial_dependence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pdp.png");

        let pdp = PartialDependence {
            feature: "x1".to_string(),
            parameter: "loc".to_string(),
            feature_values: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            predictions: vec![1.0, 1.5, 2.0, 2.5, 3.0],
            std_dev: Some(vec![0.1, 0.15, 0.2, 0.15, 0.1]),
        };

        let result =
            plot_partial_dependence(&pdp, true, path.to_str().unwrap(), &PlotConfig::default());
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_plot_density_comparison() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("density.png");

        let actual = array![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let fitted = array![1.1, 1.4, 2.1, 2.4, 3.1, 3.4, 4.1];

        let result = plot_density_comparison(
            &actual.view(),
            &fitted.view(),
            "Gaussian",
            path.to_str().unwrap(),
            &PlotConfig::default(),
            Some(5),
        );
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_plot_expectiles() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("expectiles.png");

        let expectile_data = ExpectilePrediction {
            feature: "x1".to_string(),
            feature_values: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            expectiles: vec![
                vec![0.5, 1.0, 1.5, 2.0, 2.5], // tau = 0.1
                vec![1.0, 1.5, 2.0, 2.5, 3.0], // tau = 0.5
                vec![1.5, 2.0, 2.5, 3.0, 3.5], // tau = 0.9
            ],
            tau_levels: vec![0.1, 0.5, 0.9],
        };

        let result = plot_expectiles(
            &expectile_data,
            path.to_str().unwrap(),
            &PlotConfig::default(),
            true,
        );
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_plot_expectiles_with_bands() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("expectiles_bands.png");

        let expectile_data = ExpectilePrediction {
            feature: "age".to_string(),
            feature_values: vec![20.0, 30.0, 40.0, 50.0, 60.0],
            expectiles: vec![
                vec![10.0, 15.0, 20.0, 25.0, 30.0], // tau = 0.1
                vec![15.0, 20.0, 25.0, 30.0, 35.0], // tau = 0.25
                vec![20.0, 25.0, 30.0, 35.0, 40.0], // tau = 0.5 (median)
                vec![25.0, 30.0, 35.0, 40.0, 45.0], // tau = 0.75
                vec![30.0, 35.0, 40.0, 45.0, 50.0], // tau = 0.9
            ],
            tau_levels: vec![0.1, 0.25, 0.5, 0.75, 0.9],
        };

        let result = plot_expectiles_with_bands(
            &expectile_data,
            path.to_str().unwrap(),
            &PlotConfig::default(),
        );
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_expectile_prediction_struct() {
        let data = ExpectilePrediction {
            feature: "test".to_string(),
            feature_values: vec![1.0, 2.0, 3.0],
            expectiles: vec![vec![1.0, 2.0, 3.0]],
            tau_levels: vec![0.5],
        };

        assert_eq!(data.feature, "test");
        assert_eq!(data.tau_levels.len(), 1);
        assert_eq!(data.expectiles.len(), 1);
    }
}
