//! Example demonstrating SHAP integration with GradientLSS.
//!
//! This example shows how to export model predictions and data for use with
//! Python's SHAP library. Since SHAP values require TreeExplainer which is
//! tightly integrated with Python's XGBoost/LightGBM bindings, the recommended
//! workflow is:
//!
//! 1. Train model in Rust with GradientLSS
//! 2. Export data using `export_for_shap()`
//! 3. Load the exported JSON in Python and use the `shap` library
//!
//! This provides full feature parity with XGBoostLSS/LightGBMLSS's SHAP functionality.

use gradientlss::backend::{Backend, BackendDataset, TrainConfig};
use gradientlss::distributions::Gaussian;
use gradientlss::model::GradientLSS;
use gradientlss::prelude::LightGBMBackend;
use ndarray::{Array1, Array2};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SHAP Integration Example");
    println!("========================\n");

    // Create a Gaussian distribution model
    let dist = Gaussian::default();
    let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(dist));

    // Generate synthetic training data
    let n_samples = 200;
    let mut features_vec = Vec::with_capacity(n_samples * 3);
    let mut labels_vec = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x1 = (i as f64) / 50.0;
        let x2 = ((i as f64) * 0.1).sin();
        let x3 = ((i % 10) as f64) / 10.0;

        features_vec.push(x1);
        features_vec.push(x2);
        features_vec.push(x3);

        // Target depends on x1 and x2
        labels_vec.push(2.0 * x1 + 1.5 * x2 + 0.1 * x3);
    }

    let features = Array2::from_shape_vec((n_samples, 3), features_vec)?;
    let labels = Array1::from_vec(labels_vec);

    println!("Training data shape: {:?}", features.dim());

    // Create training dataset and train
    let mut train_data =
        <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())?;

    let params = LightGBMBackend::create_params(model.n_params());
    let config = TrainConfig {
        num_boost_round: 50,
        early_stopping_rounds: Some(10),
        verbose: false,
        seed: 42,
    };

    println!("Training model...");
    model.train(&mut train_data, None, params, config)?;
    println!("Training completed!\n");

    // Export data for SHAP analysis
    let feature_names = vec![
        "feature_1".to_string(),
        "feature_2".to_string(),
        "feature_3".to_string(),
    ];

    let shap_data = model.export_for_shap(&features.view(), Some(feature_names))?;

    // Save to JSON file
    let output_path = "shap_export.json";
    shap_data.to_json_file(output_path)?;
    println!("SHAP data exported to: {}", output_path);

    // Print summary of exported data
    println!("\nExported data summary:");
    println!("  - Model type: {}", shap_data.model_type);
    println!("  - Number of samples: {}", shap_data.features.len());
    println!("  - Number of parameters: {}", shap_data.n_params);
    println!("  - Parameter names: {:?}", shap_data.param_names);
    println!(
        "  - Feature names: {:?}",
        shap_data.feature_names.as_ref().unwrap()
    );

    if let Some(ref importance) = shap_data.feature_importance {
        println!("\nFeature importance (per parameter):");
        for (param, scores) in importance {
            println!("  {}: {:?}", param, scores);
        }
    }

    // Print Python script to use the exported data
    println!("\n{}", "=".repeat(60));
    println!("To analyze with Python's SHAP library, use this script:");
    println!("{}", "=".repeat(60));
    println!(
        r#"
import json
import numpy as np
import shap
import xgboost as xgb  # or lightgbm

# Load the exported data
with open('{}', 'r') as f:
    data = json.load(f)

# Convert to numpy arrays
X = np.array(data['features'])
predictions = np.array(data['predictions'])
feature_names = data['feature_names']
param_names = data['param_names']

print(f"Loaded {{len(X)}} samples with {{len(feature_names)}} features")
print(f"Distribution parameters: {{param_names}}")

# For SHAP analysis, you would typically:
# 1. Train an equivalent model in Python on the same data
# 2. Or use the feature importance from GradientLSS directly

# Option 1: Use GradientLSS feature importance directly
if 'feature_importance' in data and data['feature_importance']:
    print("\nFeature Importance from GradientLSS:")
    for param, importance in data['feature_importance'].items():
        print(f"  {{param}}: {{importance}}")

# Option 2: Train a surrogate model and use TreeExplainer
# This is useful if you want full SHAP value computation
#
# Example with XGBoost surrogate for the 'loc' parameter:
# model_loc = xgb.XGBRegressor()
# model_loc.fit(X, predictions[:, 0])  # Train on loc predictions
# explainer = shap.TreeExplainer(model_loc)
# shap_values = explainer.shap_values(X)
# shap.summary_plot(shap_values, X, feature_names=feature_names)

print("\nSHAP analysis setup complete!")
"#,
        output_path
    );

    // Also demonstrate partial dependence (built-in alternative to SHAP)
    println!("\n{}", "=".repeat(60));
    println!("Built-in interpretability features (no Python needed):");
    println!("{}", "=".repeat(60));

    // Compute partial dependence for feature 0
    let pdp = model.partial_dependence(&features.view(), 0, 0, Some(20), None, None)?;

    println!(
        "\nPartial Dependence for '{}' on 'loc' parameter:",
        pdp.feature
    );
    println!("  Feature values: {:?}", &pdp.feature_values[..5]);
    println!("  Predictions: {:?}", &pdp.predictions[..5]);

    // ICE curves
    let ice = model.ice_curves(&features.view(), 0, 0, Some(10), None, None)?;
    println!(
        "\nICE curves computed for {} samples",
        ice.predictions.len()
    );

    println!("\nExample completed successfully!");
    println!("\nNote: The exported JSON file can be used with Python's SHAP library");
    println!("for advanced visualizations like beeswarm plots and dependency plots.");

    Ok(())
}
