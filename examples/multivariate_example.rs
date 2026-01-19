//! Example demonstrating multivariate distributional regression with GradientLSS.
//!
//! This example shows how to use the Multivariate Normal (MVN) distribution
//! to model multiple target variables simultaneously.

use gradientlss::backend::{Backend, BackendDataset, TrainConfig}; // Added Backend to imports
use gradientlss::distributions::MVN;
use gradientlss::model::{GradientLSS, PredType};
use gradientlss::prelude::LightGBMBackend;
use gradientlss::utils::ResponseFn;
use ndarray::array;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Multivariate Distributional Regression Example");
    println!("==============================================");

    // Create a multivariate normal distribution with 2 targets
    let mvn_dist = MVN::new(
        2, // Number of targets
        gradientlss::distributions::Stabilization::None,
        ResponseFn::Exp, // Response function for scale parameters
        gradientlss::distributions::LossFn::Nll,
        false, // Don't initialize with start values
    );

    // Create the model
    let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(mvn_dist));

    println!(
        "Model created with {} parameters for {} targets",
        model.n_params(),
        model.distribution().n_targets()
    );

    // Sample data: 100 observations with 2 features each
    let n_samples = 100;

    // Fix: Replaced .repeat_axis() with concatenate of views
    let base_features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
    let repeats = n_samples / 5;
    let views = vec![base_features.view(); repeats];
    let features = ndarray::concatenate(ndarray::Axis(0), &views)?;

    // For multivariate with 2 targets, we need 2 values per observation
    // Here we create synthetic data where target1 = feature1 + noise, target2 = feature2 + noise
    let mut labels = Vec::with_capacity(n_samples * 2);
    for i in 0..n_samples {
        let f1 = features[[i, 0]];
        let f2 = features[[i, 1]];
        labels.push(f1 + 0.1); // target1
        labels.push(f2 + 0.1); // target2
    }
    let labels = ndarray::Array1::from_vec(labels);

    println!(
        "Training data: {} samples, {} features, {} targets",
        n_samples,
        features.ncols(),
        model.distribution().n_targets()
    );

    // Create training dataset
    // Fix: Use fully qualified syntax for the associated type
    let mut train_data =
        <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())?;

    // Set training parameters
    let params = LightGBMBackend::create_params(model.n_params());
    let config = TrainConfig {
        num_boost_round: 10, // Small number for quick demo
        early_stopping_rounds: None,
        verbose: true,
        seed: 123,
    };

    // Train the model
    println!("\nTraining model...");
    model.train(&mut train_data, None, params, config)?;
    println!("Training completed!");

    // Make predictions
    println!("\nMaking predictions...");
    let test_features = array![[2.5, 3.5], [3.5, 4.5],];

    // Predict distributional parameters
    let params_pred = model.predict(&test_features.view(), PredType::Parameters, 100, &[], 123)?;

    if let gradientlss::backend::PredictionOutput::Parameters(params) = params_pred {
        println!("Predicted parameters for test observations:");
        println!("Observation 1: {:?}", params.row(0));
        println!("Observation 2: {:?}", params.row(1));
    }

    // Predict samples
    let samples_pred = model.predict(
        &test_features.view(),
        PredType::Samples,
        100, // 100 samples per observation
        &[],
        123,
    )?;

    if let gradientlss::backend::PredictionOutput::Samples(samples) = samples_pred {
        println!("\nSample predictions (first 5 samples for each observation):");
        println!(
            "Observation 1, Target 1 samples: {:?}",
            samples.column(0).iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "Observation 1, Target 2 samples: {:?}",
            samples.column(1).iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "Observation 2, Target 1 samples: {:?}",
            samples.column(2).iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "Observation 2, Target 2 samples: {:?}",
            samples.column(3).iter().take(5).collect::<Vec<_>>()
        );
    }

    println!("\nExample completed successfully!");
    Ok(())
}
