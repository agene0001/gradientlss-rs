# GradientLSS-rs

Distributional Gradient Boosting for Location, Scale, and Shape (LSS) in Rust.

A Rust implementation of probabilistic gradient boosting, supporting XGBoost and LightGBM backends.

## Features

- **Unified Distribution Interface**: Common `Distribution` trait for all probability distributions
- **Multiple Backends**: Support for XGBoost and LightGBM through feature flags
- **Response Functions**: Identity, exp, softplus, sigmoid, relu, and more
- **Loss Functions**: Negative log-likelihood (NLL) and CRPS
- **Gradient Stabilization**: MAD and L2 stabilization methods
- **Multivariate Distributions**: Support for multivariate normal (MVN) and other multivariate distributions

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
gradientlss = "0.1"

# Enable backends as needed:
# gradientlss = { version = "0.1", features = ["xgboost"] }
# gradientlss = { version = "0.1", features = ["lightgbm"] }
# gradientlss = { version = "0.1", features = ["full"] }  # Both backends
```

### System Requirements for Backends

Both XGBoost and LightGBM backends require native libraries to be compiled:

1. **CMake** (required for both):
   ```bash
   # macOS
   brew install cmake
   
   # Ubuntu/Debian
   sudo apt-get install cmake
   
   # Windows
   # Download from https://cmake.org/download/
   ```

2. **Clang/LLVM** (for bindgen):
   ```bash
   # macOS - usually pre-installed with Xcode
   xcode-select --install
   
   # Ubuntu/Debian
   sudo apt-get install llvm-dev libclang-dev clang
   ```

## Usage

### Basic Example

```rust
use gradientlss::prelude::*;
use gradientlss::distributions::Gaussian;

// Create a Gaussian distribution
let dist = Gaussian::new(
    Stabilization::None,
    ResponseFn::Exp,  // For scale parameter
    LossFn::Nll,
    false,  // Don't initialize with start values
);

// With XGBoost backend
#[cfg(feature = "xgboost")]
{
    use gradientlss::backend::XGBoostBackend;
    
    let mut model = GradientLSS::<XGBoostBackend>::new(dist);
    
    // Create dataset
    let mut train_data = XGBoostDataset::from_data(
        features.view(),
        labels.view(),
    )?;
    
    // Train
    let params = XGBoostBackend::create_params(2);  // 2 params: loc, scale
    let config = TrainConfig {
        num_boost_round: 100,
        early_stopping_rounds: Some(20),
        verbose: true,
        seed: 123,
    };
    
    model.train(&mut train_data, None, params, config)?;
    
    // Predict
    let predictions = model.predict(
        &test_features.view(),
        PredType::Parameters,
        1000,  // n_samples (for sampling)
        &[0.1, 0.5, 0.9],  // quantiles
        123,  // seed
    )?;
}
```

### Available Distributions

#### Univariate Distributions
- `Gaussian` - Normal distribution with loc (mean) and scale (std) parameters
- `Gamma` - Gamma distribution
- `Beta` - Beta distribution
- `StudentT` - Student's t distribution
- `Poisson` - Poisson distribution (discrete)
- `NegativeBinomial` - Negative binomial distribution (discrete)
- `Weibull` - Weibull distribution
- `LogNormal` - Log-normal distribution
- `Cauchy` - Cauchy distribution
- `Laplace` - Laplace distribution
- `Gumbel` - Gumbel distribution
- `Logistic` - Logistic distribution
- `ZAGamma` - Zero-adjusted Gamma distribution
- `ZINB` - Zero-inflated Negative Binomial distribution
- `ZIPoisson` - Zero-inflated Poisson distribution

#### Multivariate Distributions
- `MVN` - Multivariate Normal distribution with mean vector and Cholesky-decomposed covariance matrix
- `MVT` - Multivariate Student's T distribution with degrees of freedom, mean vector, and Cholesky-decomposed covariance matrix
- `Dirichlet` - Dirichlet distribution for compositional data (proportions that sum to 1)

More distributions can be added by implementing the `Distribution` trait.

### Response Functions

| Function | Description | Use Case |
|----------|-------------|----------|
| `Identity` | No transformation | Location parameters |
| `Exp` | Exponential | Strictly positive (scale) |
| `Softplus` | ln(1 + exp(x)) | Smooth positive |
| `Sigmoid` | 1/(1 + exp(-x)) | Bounded (0, 1) |
| `ExpDf` | exp(x) + 2 | Degrees of freedom |

### Multivariate Usage

For multivariate distributions, the target data must be provided in a flattened format:

#### MVN (Multivariate Normal)

```rust
use gradientlss::distributions::MVN;
use gradientlss::backend::lightgbm_backend::LightGBMBackend;
use gradientlss::model::GradientLSS;
use ndarray::array;

// Create a 2-target MVN distribution
let mvn = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
let mut model = GradientLSS::<LightGBMBackend>::new(mvn);

// For 3 observations with 2 targets each, labels should be:
// [y1_obs1, y2_obs1, y1_obs2, y2_obs2, y1_obs3, y2_obs3]
let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let labels = array![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]; // Flattened multivariate targets

let mut train_data = LightGBMBackend::Dataset::from_data(features.view(), labels.view())?;
```

#### MVT (Multivariate Student's T)

```rust
use gradientlss::distributions::{MVT, ResponseFn};

// Create a 2-target MVT distribution with separate response functions
let mvt = MVT::new(
    2,
    Stabilization::None,
    ResponseFn::Exp,      // For scale parameters
    ResponseFn::ExpDf,    // For degrees of freedom (ensures df > 2)
    LossFn::Nll,
    false
);
```

#### Dirichlet (Compositional Data)

```rust
use gradientlss::distributions::Dirichlet;

// Create a 3-target Dirichlet distribution for compositional data
let dirichlet = Dirichlet::new(3, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

// For Dirichlet, targets must sum to 1 for each observation
// [p1_obs1, p2_obs1, p3_obs1, p1_obs2, p2_obs2, p3_obs2, ...]
let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let labels = array![0.3, 0.4, 0.3, 0.2, 0.5, 0.3, 0.1, 0.6, 0.3];
```

The model will automatically handle the reshaping and parameter estimation for each multivariate distribution type.

## Model Interpretability

GradientLSS provides multiple approaches for model interpretation, matching the functionality of XGBoostLSS/LightGBMLSS.

### Built-in Interpretability Features

```rust
// Feature importance
let importance = model.feature_importance(
    FeatureImportanceType::Gain,
    Some(vec!["age".to_string(), "income".to_string()]),
)?;

// Partial dependence plots
let pdp = model.partial_dependence(
    &features.view(),
    0,  // feature index
    0,  // parameter index (e.g., 0 for 'loc')
    Some(50),  // grid size
    None,
    None,
)?;

// ICE (Individual Conditional Expectation) curves
let ice = model.ice_curves(
    &features.view(),
    0,  // feature index
    0,  // parameter index
    Some(50),
    None,
    None,
)?;
```

### Plotting (with `plotting` feature)

```rust
// Feature importance plot
model.plot_feature_importance(
    "importance.png",
    FeatureImportanceType::Gain,
    None,  // aggregated across parameters
    Some(feature_names),
    None,
)?;

// Partial dependence plot
model.plot_partial_dependence(
    &features.view(),
    0,  // feature index
    0,  // parameter index
    "pdp.png",
    None,
)?;

// Expectile plot (for Expectile distribution models)
model.expectile_plot(
    &features.view(),
    0,  // feature index
    "expectiles.png",
    Some(50),  // grid size
    Some("Age".to_string()),
    None,
    true,  // show confidence bands
)?;
```

### SHAP Integration

For advanced SHAP-based visualizations (beeswarm plots, dependency plots), GradientLSS provides a data export workflow compatible with Python's `shap` library:

```rust
// Export data for SHAP analysis
let shap_data = model.export_for_shap(
    &features.view(),
    Some(vec!["feature1".to_string(), "feature2".to_string()]),
)?;

// Save to JSON for Python consumption
shap_data.to_json_file("shap_export.json")?;
```

Then in Python:

```python
import json
import numpy as np
import shap
import xgboost as xgb

# Load the exported data
with open('shap_export.json', 'r') as f:
    data = json.load(f)

X = np.array(data['features'])
predictions = np.array(data['predictions'])
feature_names = data['feature_names']
param_names = data['param_names']

# Use feature importance directly from GradientLSS
if data['feature_importance']:
    for param, importance in data['feature_importance'].items():
        print(f"{param}: {importance}")

# Or train a surrogate model for full SHAP values
# (useful for beeswarm plots and SHAP dependency plots)
model_loc = xgb.XGBRegressor()
model_loc.fit(X, predictions[:, 0])  # Train on 'loc' predictions

explainer = shap.TreeExplainer(model_loc)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, feature_names=feature_names)
```

See `examples/shap_integration.rs` for a complete example.

## Backend Differences

### XGBoost (`--features xgboost`)
- Full support for custom objective functions
- Uses `update_custom()` for distributional gradient updates
- Row-major (C-order) gradient layout

### LightGBM (`--features lightgbm`)
- **Limited**: Current Rust bindings don't expose custom objective API
- Uses built-in objectives only
- Column-major (Fortran-order) gradient layout
- Consider using XGBoost for full distributional regression support

## Architecture

```
gradientlss/
├── distributions/     # Distribution implementations
│   ├── base.rs       # Distribution trait
│   └── gaussian.rs   # Gaussian distribution
├── backend/          # Gradient boosting backends
│   ├── traits.rs     # Backend trait definitions
│   ├── xgboost_backend.rs
│   └── lightgbm_backend.rs
├── utils.rs          # Response functions
├── model.rs          # GradientLSS model wrapper
└── error.rs          # Error types
```

## Adding New Distributions

1. Create a new file in `src/distributions/`
2. Implement the `Distribution` trait:

```rust
impl Distribution for MyDistribution {
    fn n_params(&self) -> usize { /* ... */ }
    fn params(&self) -> &[DistributionParam] { /* ... */ }
    fn log_prob(&self, params: &[f64], target: f64) -> f64 { /* ... */ }
    fn nll(&self, params: &ArrayView2<f64>, target: &ArrayView1<f64>) -> f64 { /* ... */ }
    fn sample(&self, params: &ArrayView2<f64>, n_samples: usize, seed: u64) -> Array2<f64> { /* ... */ }
    // ... other required methods
}
```

3. Export from `src/distributions/mod.rs`

## License

MIT
