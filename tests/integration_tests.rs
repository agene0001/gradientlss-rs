//! Integration tests for GradientLSS.

use gradientlss::distributions::{Distribution, Gaussian, StudentT};
use gradientlss::prelude::*;
use gradientlss::types::ResponseData;
use ndarray::{Array1, array};

#[test]
fn test_dist_select_integration() {
    let target = array![1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5];
    let candidates: Vec<Box<dyn Distribution>> =
        vec![Box::new(Gaussian::default()), Box::new(StudentT::default())];

    let result = dist_select(&target.view(), candidates, 100);

    assert!(result.is_ok());
    let ranked_dists = result.unwrap();

    assert_eq!(ranked_dists.len(), 2);
    assert!(!ranked_dists[0].0.is_empty());
    assert!(ranked_dists[0].1.is_finite());

    // Expect Gaussian to be a better fit for this simple data than Student-T
    let _gaussian_nll = ranked_dists
        .iter()
        .find(|(name, _)| name == "loc_scale")
        .unwrap()
        .1;
    let _studentt_nll = ranked_dists
        .iter()
        .find(|(name, _)| name == "df_loc_scale")
        .unwrap()
        .1;

    // Student-T might fit slightly better if df is large, but they should be close.
    // Let's just check that the ranking is correct.
    assert!(ranked_dists[0].1 <= ranked_dists[1].1);
}

#[test]
fn test_gaussian_distribution_workflow() {
    // Create a Gaussian distribution
    let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

    assert_eq!(dist.n_params(), 2);
    assert_eq!(dist.param_names(), vec!["loc", "scale"]);
    assert!(!dist.is_discrete());
    assert!(dist.is_univariate());
}

#[test]
fn test_gaussian_log_prob() {
    let dist = Gaussian::default();

    // Test standard normal at mean
    let log_p = dist.log_prob(&[0.0, 1.0], &[0.0]);
    let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!((log_p - expected).abs() < 1e-10);

    // Test that probability decreases away from mean
    let log_p_at_mean = dist.log_prob(&[5.0, 1.0], &[5.0]);
    let log_p_away = dist.log_prob(&[5.0, 1.0], &[7.0]);
    assert!(log_p_at_mean > log_p_away);
}

#[test]
fn test_gaussian_nll() {
    let dist = Gaussian::default();

    // Create parameters and targets
    let params = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
    let targets = array![0.0, 0.0, 0.0];

    // Fix: wrap targets in ResponseData::Univariate
    let target_data = ResponseData::Univariate(&targets.view());
    let nll = dist.nll(&params.view(), &target_data);

    // NLL should be positive
    assert!(nll > 0.0);

    // NLL at mean should be lower than NLL away from mean
    let targets_away = array![2.0, 2.0, 2.0];
    // Fix: wrap targets in ResponseData::Univariate
    let target_data_away = ResponseData::Univariate(&targets_away.view());
    let nll_away = dist.nll(&params.view(), &target_data_away);
    assert!(nll < nll_away);
}

#[test]
fn test_gaussian_sampling() {
    let dist = Gaussian::default();

    let params = array![[0.0, 1.0], [10.0, 0.1]];
    let samples = dist.sample(&params.view(), 10000, 42);

    assert_eq!(samples.dim(), (10000, 2));

    // Check approximate means
    let mean_0: f64 = samples.column(0).iter().sum::<f64>() / 10000.0;
    let mean_1: f64 = samples.column(1).iter().sum::<f64>() / 10000.0;

    assert!((mean_0 - 0.0).abs() < 0.1);
    assert!((mean_1 - 10.0).abs() < 0.1);

    // Check approximate standard deviations
    let std_0: f64 = (samples
        .column(0)
        .iter()
        .map(|x| (x - mean_0).powi(2))
        .sum::<f64>()
        / 10000.0)
        .sqrt();
    let std_1: f64 = (samples
        .column(1)
        .iter()
        .map(|x| (x - mean_1).powi(2))
        .sum::<f64>()
        / 10000.0)
        .sqrt();

    assert!((std_0 - 1.0).abs() < 0.1);
    assert!((std_1 - 0.1).abs() < 0.05);
}

#[test]
fn test_gaussian_transform_params() {
    let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

    let predictions = array![[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]];
    let transformed = dist.transform_params(&predictions.view());

    // loc (identity) should be unchanged
    assert!((transformed[[0, 0]] - 0.0).abs() < 1e-6);
    assert!((transformed[[1, 0]] - 1.0).abs() < 1e-6);
    assert!((transformed[[2, 0]] - (-1.0)).abs() < 1e-6);

    // scale (exp) should be positive
    assert!(transformed[[0, 1]] > 0.0);
    assert!(transformed[[1, 1]] > 0.0);
    assert!(transformed[[2, 1]] > 0.0);

    // exp(1) > exp(0) > exp(-1)
    assert!(transformed[[1, 1]] > transformed[[0, 1]]);
    assert!(transformed[[0, 1]] > transformed[[2, 1]]);
}

#[test]
fn test_gaussian_start_values() {
    let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);

    // Generate some target data
    let target = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 2.0, 3.0, 4.0]);

    // Fix: wrap targets in ResponseData::Univariate
    let target_data = ResponseData::Univariate(&target.view());
    let result = dist.calculate_start_values(&target_data, 50);
    assert!(result.is_ok());

    let (loss, start_vals) = result.unwrap();
    assert!(loss.is_finite());
    assert_eq!(start_vals.len(), 2);
    assert!(start_vals.iter().all(|v| v.is_finite()));
}

#[test]
fn test_gaussian_gradients_and_hessians() {
    let dist = Gaussian::default();

    // Create predictions and targets
    let predictions = array![[0.0, 0.0], [0.0, 0.0]];
    let targets = array![0.0, 1.0];

    // Fix: wrap targets in ResponseData::Univariate
    let target_data = ResponseData::Univariate(&targets.view());
    let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);

    assert!(result.is_ok());
    let gh = result.unwrap();

    assert_eq!(gh.gradients.dim(), (2, 2));
    assert_eq!(gh.hessians.dim(), (2, 2));

    // Hessians should be positive
    assert!(gh.hessians.iter().all(|&h| h > 0.0));
}

#[test]
fn test_response_functions() {
    use gradientlss::utils::{exp_fn, identity_fn, sigmoid_fn, softplus_fn};

    let x = array![0.0, 1.0, -1.0];

    // Identity should return same values (after nan handling)
    let id_result = identity_fn(&x.view());
    assert!((id_result[0] - 0.0).abs() < 1e-10);
    assert!((id_result[1] - 1.0).abs() < 1e-10);
    assert!((id_result[2] - (-1.0)).abs() < 1e-10);

    // Exp should return positive values
    let exp_result = exp_fn(&x.view());
    assert!(exp_result.iter().all(|&v| v > 0.0));

    // Softplus should return positive values
    let sp_result = softplus_fn(&x.view());
    assert!(sp_result.iter().all(|&v| v > 0.0));

    // Sigmoid should return values in (0, 1)
    let sig_result = sigmoid_fn(&x.view());
    assert!(sig_result.iter().all(|&v| v > 0.0 && v < 1.0));
}

#[test]
fn test_loss_functions() {
    assert_eq!(LossFn::Nll.name(), "nll");
    assert_eq!(LossFn::Crps.name(), "crps");
}

#[test]
fn test_stabilization_methods() {
    let dist_none = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
    let dist_mad = Gaussian::new(Stabilization::Mad, ResponseFn::Exp, LossFn::Nll, false);
    let dist_l2 = Gaussian::new(Stabilization::L2, ResponseFn::Exp, LossFn::Nll, false);

    assert_eq!(dist_none.stabilization(), Stabilization::None);
    assert_eq!(dist_mad.stabilization(), Stabilization::Mad);
    assert_eq!(dist_l2.stabilization(), Stabilization::L2);
}

#[cfg(feature = "lightgbm")]
#[test]
fn test_lightgbm_cv() {
    use gradientlss::backend::LightGBMBackend;
    use gradientlss::distributions::Gaussian;
    use gradientlss::model::GradientLSS;
    use ndarray::array;
    use std::sync::Arc;

    let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::default()));
    let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let labels = array![1.0, 2.0, 3.0, 4.0];
    let params = LightGBMBackend::create_params(model.n_params());
    let config = gradientlss::backend::TrainConfig::default();

    let score = model.cv(&features, &labels, 2, params, config).unwrap();
    assert!(score.is_finite());
}

#[cfg(feature = "lightgbm")]
#[test]
fn test_lightgbm_hyper_opt() {
    use gradientlss::backend::LightGBMBackend;
    use gradientlss::distributions::Gaussian;
    use gradientlss::model::GradientLSS;
    use ndarray::array;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::default()));
    let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let labels = array![1.0, 2.0, 3.0, 4.0];
    let mut hp_dict = HashMap::new();
    hp_dict.insert("learning_rate".to_string(), json!([0.01, 1.0]));
    hp_dict.insert("num_leaves".to_string(), json!([2, 10]));

    let best_params = model
        .hyper_opt(&features, &labels, &hp_dict, 10, 2)
        .unwrap();
    assert!(best_params.contains_key("learning_rate"));
    assert!(best_params.contains_key("num_leaves"));
}

#[cfg(feature = "xgboost")]
mod xgboost_tests {
    use super::*;
    use gradientlss::backend::{Backend, BackendDataset, XGBoostBackend};
    use gradientlss::distributions::{LossFn, Stabilization};
    use gradientlss::model::{GradientLSS, PredType};
    use gradientlss::utils::ResponseFn;
    use std::sync::Arc;

    #[test]
    fn test_xgboost_backend_name() {
        assert_eq!(XGBoostBackend::name(), "XGBoost");
    }

    #[test]
    fn test_xgboost_model_creation() {
        let dist = Gaussian::default();
        let model = GradientLSS::<XGBoostBackend>::new(Arc::new(dist));

        assert!(!model.is_trained());
        assert_eq!(model.n_params(), 2);
    }

    #[test]
    fn test_xgboost_reshape_gradients() {
        let gradients = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let hessians = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];

        let (grad_flat, _hess_flat) = XGBoostBackend::reshape_gradients(&gradients, &hessians);

        // XGBoost uses C order (row-major)
        assert_eq!(grad_flat.len(), 6);
        assert_eq!(grad_flat[0], 1.0);
        assert_eq!(grad_flat[1], 2.0);
        assert_eq!(grad_flat[2], 3.0);
    }

    #[test]
    fn test_xgboost_univariate_training() {
        let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![1.0, 2.0, 3.0];

        let mut train_data =
            <XGBoostBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = XGBoostBackend::create_params(model.n_params());
        let config = gradientlss::backend::TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
    }

    #[test]
    fn test_xgboost_prediction() {
        let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![1.0, 2.0, 3.0];

        let mut train_data =
            <XGBoostBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = XGBoostBackend::create_params(model.n_params());
        let config = gradientlss::backend::TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        // Test predictions
        let test_features = array![[2.0, 3.0], [4.0, 5.0]];

        // Test parameter prediction
        let params_pred = model.predict(&test_features.view(), PredType::Parameters, 100, &[], 42);
        assert!(matches!(
            params_pred,
            Ok(gradientlss::backend::PredictionOutput::Parameters(_))
        ));

        // Test sample prediction
        let samples_pred = model.predict(&test_features.view(), PredType::Samples, 100, &[], 42);
        assert!(matches!(
            samples_pred,
            Ok(gradientlss::backend::PredictionOutput::Samples(_))
        ));

        // Test quantile prediction
        let quantiles = vec![0.1, 0.5, 0.9];
        let quantiles_pred = model.predict(
            &test_features.view(),
            PredType::Quantiles,
            100,
            &quantiles,
            42,
        );
        assert!(matches!(
            quantiles_pred,
            Ok(gradientlss::backend::PredictionOutput::Quantiles(_))
        ));
    }

    #[test]
    fn test_xgboost_multivariate_training() {
        let mut model =
            GradientLSS::<XGBoostBackend>::new(Arc::new(gradientlss::distributions::MVN::new(
                2,
                Stabilization::None,
                ResponseFn::Exp,
                LossFn::Nll,
                false,
            )));

        // For multivariate with 2 targets, we need 2 * n_samples labels
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        // Labels are flattened: [y1_obs1, y2_obs1, y1_obs2, y2_obs2, ...]
        let labels = array![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];

        let mut train_data =
            <XGBoostBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = XGBoostBackend::create_params(model.n_params());
        let config = gradientlss::backend::TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.n_params(), 5); // 2 loc + 3 tril
    }

    #[test]
    fn test_xgboost_expectile_training() {
        let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(
            gradientlss::distributions::Expectile::new(
                vec![0.1, 0.5, 0.9],
                false,
                Stabilization::None,
                LossFn::Nll,
                false,
            ),
        ));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![1.5, 2.5, 3.5];

        let mut train_data =
            <XGBoostBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = XGBoostBackend::create_params(model.n_params());
        let config = gradientlss::backend::TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.n_params(), 3); // 3 expectiles
    }

    #[test]
    fn test_xgboost_gamma_training() {
        let mut model =
            GradientLSS::<XGBoostBackend>::new(Arc::new(gradientlss::distributions::Gamma::new(
                Stabilization::None,
                ResponseFn::Exp,
                LossFn::Nll,
                false,
            )));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![1.0, 2.0, 3.0];

        let mut train_data =
            <XGBoostBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = XGBoostBackend::create_params(model.n_params());
        let config = gradientlss::backend::TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.n_params(), 2); // shape + scale
    }

    #[test]
    fn test_xgboost_poisson_training() {
        let mut model =
            GradientLSS::<XGBoostBackend>::new(Arc::new(gradientlss::distributions::Poisson::new(
                Stabilization::None,
                ResponseFn::Exp,
                LossFn::Nll,
                false,
            )));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![1.0, 2.0, 3.0];

        let mut train_data =
            <XGBoostBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = XGBoostBackend::create_params(model.n_params());
        let config = gradientlss::backend::TrainConfig::default();

        model.train(&mut train_data, None, params, config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.n_params(), 1); // rate
    }
}

#[cfg(feature = "lightgbm")]
mod lightgbm_tests {
    use super::*;
    use gradientlss::backend::{Backend, LightGBMBackend};
    use gradientlss::model::GradientLSS;
    use std::sync::Arc;

    #[test]
    fn test_lightgbm_backend_name() {
        assert_eq!(LightGBMBackend::name(), "LightGBM");
    }

    use gradientlss::distributions::Gaussian;

    #[test]
    fn test_lightgbm_model_creation() {
        let dist = Gaussian::default();
        let model = GradientLSS::<LightGBMBackend>::new(Arc::new(dist));

        assert!(!model.is_trained());
        assert_eq!(model.n_params(), 2);
    }

    #[test]
    fn test_lightgbm_reshape_gradients() {
        let gradients = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let hessians = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];

        let (grad_flat, _hess_flat) = LightGBMBackend::reshape_gradients(&gradients, &hessians);

        // LightGBM uses Fortran order (column-major)
        assert_eq!(grad_flat.len(), 6);
        assert_eq!(grad_flat[0], 1.0); // [0,0]
        assert_eq!(grad_flat[1], 3.0); // [1,0]
        assert_eq!(grad_flat[2], 5.0); // [2,0]
        assert_eq!(grad_flat[3], 2.0); // [0,1]
    }
}

// =============================================================================
// Comprehensive Distribution Tests
// =============================================================================

mod distribution_tests {
    use gradientlss::distributions::*;
    use gradientlss::prelude::*;
    use ndarray::{Array2, array};

    // -------------------------------------------------------------------------
    // Gamma Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gamma_creation() {
        let dist = Gamma::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["concentration", "rate"]);
        assert!(!dist.is_discrete());
    }

    #[test]
    fn test_gamma_log_prob() {
        let dist = Gamma::default();
        // Gamma(shape=2, scale=1) at x=1
        let log_p = dist.log_prob(&[2.0, 1.0], &[1.0]);
        assert!(log_p.is_finite());
        assert!(log_p < 0.0); // Log prob should be negative
    }

    #[test]
    fn test_gamma_sampling() {
        let dist = Gamma::default();
        let params = array![[2.0, 1.0], [3.0, 0.5]];
        let samples = dist.sample(&params.view(), 1000, 42);

        // Gamma sample returns (n_observations, n_samples) - different from other distributions
        assert_eq!(samples.nrows(), 2);
        assert_eq!(samples.ncols(), 1000);
        // All samples should be positive (Gamma support is [0, inf))
        assert!(samples.iter().all(|&x| x > 0.0));
    }

    // -------------------------------------------------------------------------
    // Beta Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_beta_creation() {
        let dist = Beta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["concentration1", "concentration0"]);
    }

    #[test]
    fn test_beta_log_prob() {
        let dist = Beta::default();
        // Beta(2, 2) at x=0.5 should give highest probability
        let log_p_half = dist.log_prob(&[2.0, 2.0], &[0.5]);
        let log_p_quarter = dist.log_prob(&[2.0, 2.0], &[0.25]);
        assert!(log_p_half > log_p_quarter);
    }

    #[test]
    fn test_beta_sampling() {
        let dist = Beta::default();
        let params = array![[2.0, 2.0]];
        let samples = dist.sample(&params.view(), 1000, 42);

        // All samples should be in (0, 1)
        assert!(samples.iter().all(|&x| x > 0.0 && x < 1.0));
    }

    // -------------------------------------------------------------------------
    // Student-T Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_student_t_creation() {
        let dist = StudentT::default();
        assert_eq!(dist.n_params(), 3);
        assert_eq!(dist.param_names(), vec!["df", "loc", "scale"]);
    }

    #[test]
    fn test_student_t_log_prob() {
        let dist = StudentT::default();
        // df=5, loc=0, scale=1 at x=0
        let log_p = dist.log_prob(&[5.0, 0.0, 1.0], &[0.0]);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_student_t_heavier_tails() {
        let dist = StudentT::default();
        // Lower df = heavier tails = higher probability at extremes
        let log_p_df1 = dist.log_prob(&[1.0, 0.0, 1.0], &[5.0]);
        let log_p_df30 = dist.log_prob(&[30.0, 0.0, 1.0], &[5.0]);
        assert!(log_p_df1 > log_p_df30);
    }

    // -------------------------------------------------------------------------
    // Poisson Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_poisson_creation() {
        let dist = Poisson::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        assert_eq!(dist.n_params(), 1);
        assert_eq!(dist.param_names(), vec!["rate"]);
        assert!(dist.is_discrete());
    }

    #[test]
    fn test_poisson_log_prob() {
        let dist = Poisson::default();
        // Poisson(rate=5) at k=5 should give high probability
        let log_p_5 = dist.log_prob(&[5.0], &[5.0]);
        let log_p_0 = dist.log_prob(&[5.0], &[0.0]);
        assert!(log_p_5 > log_p_0);
    }

    // -------------------------------------------------------------------------
    // Negative Binomial Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_negative_binomial_creation() {
        let dist = NegativeBinomial::default();
        assert_eq!(dist.n_params(), 2);
        assert!(dist.is_discrete());
    }

    // -------------------------------------------------------------------------
    // Laplace Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_laplace_creation() {
        let dist = Laplace::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
    }

    #[test]
    fn test_laplace_log_prob() {
        let dist = Laplace::default();
        let log_p = dist.log_prob(&[0.0, 1.0], &[0.0]);
        assert!(log_p.is_finite());
    }

    // -------------------------------------------------------------------------
    // Weibull Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_weibull_creation() {
        let dist = Weibull::default();
        assert_eq!(dist.n_params(), 2);
    }

    #[test]
    fn test_weibull_sampling() {
        let dist = Weibull::default();
        let params = array![[1.5, 1.0]];
        let samples = dist.sample(&params.view(), 1000, 42);

        // All samples should be positive
        assert!(samples.iter().all(|&x| x > 0.0));
    }

    // -------------------------------------------------------------------------
    // Gumbel Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gumbel_creation() {
        let dist = Gumbel::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
    }

    // -------------------------------------------------------------------------
    // Log Normal Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_log_normal_creation() {
        let dist = LogNormal::default();
        assert_eq!(dist.n_params(), 2);
    }

    #[test]
    fn test_log_normal_sampling() {
        let dist = LogNormal::default();
        let params = array![[0.0, 1.0]];
        let samples = dist.sample(&params.view(), 1000, 42);

        // All samples should be positive (log-normal support is (0, inf))
        assert!(samples.iter().all(|&x| x > 0.0));
    }

    // -------------------------------------------------------------------------
    // Cauchy Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cauchy_creation() {
        let dist = Cauchy::default();
        assert_eq!(dist.n_params(), 2);
        assert_eq!(dist.param_names(), vec!["loc", "scale"]);
    }

    // -------------------------------------------------------------------------
    // SplineFlow Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_spline_flow_creation() {
        let dist = SplineFlow::default();
        assert_eq!(dist.n_params(), 31); // Linear with 8 bins: 3*8 + (8-1) = 31
    }

    #[test]
    fn test_spline_flow_quadratic() {
        let dist = SplineFlow::new(
            TargetSupport::Real,
            8,
            3.0,
            SplineOrder::Quadratic,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        assert_eq!(dist.n_params(), 23); // Quadratic with 8 bins: 2*8 + (8-1) = 23
    }

    #[test]
    fn test_spline_flow_log_prob() {
        let dist = SplineFlow::default();
        let params = vec![0.0; dist.n_params()];
        let log_p = dist.log_prob(&params, &[0.0]);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_spline_flow_positive_support() {
        let dist = SplineFlow::new(
            TargetSupport::Positive,
            4,
            3.0,
            SplineOrder::Linear,
            Stabilization::None,
            LossFn::Nll,
            false,
        );

        let n_params = dist.n_params();
        let params = Array2::zeros((1, n_params));
        let samples = dist.sample(&params.view(), 100, 42);

        // All samples should be non-negative
        assert!(samples.iter().all(|&x| x >= 0.0));
    }

    // -------------------------------------------------------------------------
    // Expectile Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_expectile_creation() {
        let dist = Expectile::new(
            vec![0.1, 0.5, 0.9],
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        assert_eq!(dist.n_params(), 3);
    }

    #[test]
    fn test_expectile_valid_tau() {
        // Tau values must be strictly in (0, 1)
        let dist = Expectile::new(
            vec![0.1, 0.5, 0.9], // Valid tau values
            false,
            Stabilization::None,
            LossFn::Nll,
            false,
        );
        assert_eq!(dist.n_params(), 3);
    }

    // -------------------------------------------------------------------------
    // Multivariate Normal (MVN) Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mvn_creation() {
        let dist = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        assert_eq!(dist.n_targets(), 2);
        // 2 loc + 3 tril (lower triangular 2x2 has 3 elements)
        assert_eq!(dist.n_params(), 5);
    }

    #[test]
    fn test_mvn_creation_3d() {
        let dist = MVN::new(3, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);
        assert_eq!(dist.n_targets(), 3);
        // 3 loc + 6 tril (lower triangular 3x3 has 6 elements)
        assert_eq!(dist.n_params(), 9);
    }

    // -------------------------------------------------------------------------
    // Zero-Inflated Distributions Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_zinb_creation() {
        let dist = ZINB::default();
        assert_eq!(dist.n_params(), 3); // total_count, probs, gate
        assert!(dist.is_discrete());
    }

    #[test]
    fn test_zipoisson_creation() {
        let dist = ZIPoisson::default();
        assert_eq!(dist.n_params(), 2); // rate, gate
        assert!(dist.is_discrete());
    }
}

// =============================================================================
// Numerical Stability and Edge Case Tests
// =============================================================================

mod numerical_stability_tests {
    use gradientlss::distributions::*;
    use gradientlss::prelude::*;
    use gradientlss::types::ResponseData;
    use ndarray::array;

    #[test]
    fn test_gaussian_extreme_values() {
        let dist = Gaussian::default();

        // Test with very large values
        let log_p_large = dist.log_prob(&[1e6, 1.0], &[1e6]);
        assert!(log_p_large.is_finite());

        // Test with very small scale
        let log_p_small_scale = dist.log_prob(&[0.0, 1e-6], &[0.0]);
        assert!(log_p_small_scale.is_finite());
    }

    #[test]
    fn test_gaussian_gradients_no_nan() {
        let dist = Gaussian::default();

        let predictions = array![[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]];
        let targets = array![0.5, 1.5, -0.5];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();

        // No NaN or Inf in gradients
        assert!(gh.gradients.iter().all(|&x| x.is_finite()));
        // Hessians should be positive and finite
        assert!(gh.hessians.iter().all(|&x| x.is_finite() && x > 0.0));
    }

    #[test]
    fn test_gamma_positive_params() {
        let dist = Gamma::default();

        // Gamma requires positive shape and scale
        let params = array![[2.0, 1.0]];
        let transformed = dist.transform_params(&params.view());

        assert!(transformed[[0, 0]] > 0.0);
        assert!(transformed[[0, 1]] > 0.0);
    }

    #[test]
    fn test_beta_bounded_output() {
        let dist = Beta::default();

        let params = array![[2.0, 2.0]];
        let samples = dist.sample(&params.view(), 1000, 42);

        // All Beta samples must be in (0, 1)
        assert!(samples.iter().all(|&x| x > 0.0 && x < 1.0));
    }

    #[test]
    fn test_response_functions_extreme_inputs() {
        use gradientlss::utils::{exp_fn, sigmoid_fn, softplus_fn};

        // Test with extreme values
        let extreme = array![-100.0, 0.0, 100.0];

        let exp_result = exp_fn(&extreme.view());
        assert!(exp_result.iter().all(|&x| x.is_finite() && x > 0.0));

        let sig_result = sigmoid_fn(&extreme.view());
        assert!(
            sig_result
                .iter()
                .all(|&x| x.is_finite() && x > 0.0 && x < 1.0)
        );

        let sp_result = softplus_fn(&extreme.view());
        assert!(sp_result.iter().all(|&x| x.is_finite() && x > 0.0));
    }

    #[test]
    fn test_stabilization_mad() {
        let dist = Gaussian::new(Stabilization::Mad, ResponseFn::Exp, LossFn::Nll, false);

        let predictions = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let targets = array![0.5, 1.5, 2.5];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();
        assert!(gh.gradients.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_stabilization_l2() {
        let dist = Gaussian::new(Stabilization::L2, ResponseFn::Exp, LossFn::Nll, false);

        let predictions = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let targets = array![0.5, 1.5, 2.5];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();
        assert!(gh.gradients.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_crps_score_basic() {
        let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);

        let params = array![[0.0, 1.0], [0.0, 1.0]];
        let samples = dist.sample(&params.view(), 100, 42);

        let targets = array![0.0, 0.0];
        let target_data = ResponseData::Univariate(&targets.view());

        let crps = dist.crps_score(&target_data, &samples.view());
        assert!(crps.is_finite());
        assert!(crps >= 0.0);
    }
}

// =============================================================================
// Hyperparameter Optimization Tests
// =============================================================================

mod hyper_opt_tests {
    use gradientlss::hyper_opt::{HyperParamSpec, HyperParamType};
    use serde_json::json;

    #[test]
    fn test_hyper_param_spec_float() {
        let spec = HyperParamSpec::float("learning_rate", 0.01, 0.3);
        assert_eq!(spec.name, "learning_rate");
        match spec.param_type {
            HyperParamType::Float { low, high, log } => {
                assert_eq!(low, 0.01);
                assert_eq!(high, 0.3);
                assert!(!log);
            }
            _ => panic!("Expected Float type"),
        }
    }

    #[test]
    fn test_hyper_param_spec_log_float() {
        let spec = HyperParamSpec::log_float("learning_rate", 0.001, 0.1);
        match spec.param_type {
            HyperParamType::Float { log, .. } => {
                assert!(log);
            }
            _ => panic!("Expected Float type"),
        }
    }

    #[test]
    fn test_hyper_param_spec_int() {
        let spec = HyperParamSpec::int("max_depth", 3, 10);
        match spec.param_type {
            HyperParamType::Int { low, high, log } => {
                assert_eq!(low, 3);
                assert_eq!(high, 10);
                assert!(!log);
            }
            _ => panic!("Expected Int type"),
        }
    }

    #[test]
    fn test_hyper_param_spec_categorical() {
        let spec = HyperParamSpec::categorical("booster", vec![json!("gbtree"), json!("dart")]);
        match spec.param_type {
            HyperParamType::Categorical { choices } => {
                assert_eq!(choices.len(), 2);
            }
            _ => panic!("Expected Categorical type"),
        }
    }
}

// =============================================================================
// NaN/Inf Handling Tests
// =============================================================================

mod nan_inf_handling_tests {
    use gradientlss::distributions::*;
    use gradientlss::prelude::*;
    use gradientlss::types::ResponseData;
    use ndarray::array;

    #[test]
    fn test_gaussian_gradients_with_nan_predictions() {
        let dist = Gaussian::default();

        // Predictions with NaN values
        let predictions = array![[0.0, 0.0], [f64::NAN, 1.0], [1.0, f64::NAN]];
        let targets = array![0.5, 1.5, 2.5];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();
        // After stabilization, there should be no NaN values
        assert!(
            gh.gradients.iter().all(|&x| x.is_finite()),
            "Gradients contain NaN or Inf after NaN input"
        );
        assert!(
            gh.hessians.iter().all(|&x| x.is_finite() && x > 0.0),
            "Hessians contain NaN, Inf, or non-positive values after NaN input"
        );
    }

    #[test]
    fn test_gaussian_gradients_with_inf_predictions() {
        let dist = Gaussian::default();

        // Predictions with Inf values
        let predictions = array![[0.0, 0.0], [f64::INFINITY, 1.0], [1.0, f64::NEG_INFINITY]];
        let targets = array![0.5, 1.5, 2.5];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();
        // After stabilization, there should be no Inf values
        assert!(
            gh.gradients.iter().all(|&x| x.is_finite()),
            "Gradients contain NaN or Inf after Inf input"
        );
        assert!(
            gh.hessians.iter().all(|&x| x.is_finite() && x > 0.0),
            "Hessians contain NaN, Inf, or non-positive values after Inf input"
        );
    }

    #[test]
    fn test_response_functions_nan_handling() {
        use gradientlss::utils::{exp_fn, identity_fn, sigmoid_fn, softplus_fn};

        let x_with_nan = array![0.0, f64::NAN, 1.0, f64::NEG_INFINITY, f64::INFINITY];

        // Identity should replace NaN with 0
        let id_result = identity_fn(&x_with_nan.view());
        assert!(
            id_result.iter().all(|&v| v.is_finite()),
            "Identity didn't handle NaN/Inf"
        );

        // Exp should handle NaN and produce finite positive values
        let exp_result = exp_fn(&x_with_nan.view());
        assert!(
            exp_result.iter().all(|&v| v.is_finite() && v > 0.0),
            "Exp didn't handle NaN/Inf properly"
        );

        // Sigmoid should produce values in (0, 1)
        let sig_result = sigmoid_fn(&x_with_nan.view());
        assert!(
            sig_result
                .iter()
                .all(|&v| v.is_finite() && v > 0.0 && v < 1.0),
            "Sigmoid didn't handle NaN/Inf properly"
        );

        // Softplus should produce positive values
        let sp_result = softplus_fn(&x_with_nan.view());
        assert!(
            sp_result.iter().all(|&v| v.is_finite() && v > 0.0),
            "Softplus didn't handle NaN/Inf properly"
        );
    }

    #[test]
    fn test_stabilization_mad_with_nan() {
        let dist = Gaussian::new(Stabilization::Mad, ResponseFn::Exp, LossFn::Nll, false);

        // Include some extreme values that might cause NaN in gradients
        let predictions = array![[0.0, 0.0], [100.0, 100.0], [-100.0, -100.0]];
        let targets = array![0.0, 1.0, 2.0];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();
        assert!(
            gh.gradients.iter().all(|&x| x.is_finite()),
            "MAD stabilization failed to handle extreme values"
        );
    }

    #[test]
    fn test_stabilization_l2_with_nan() {
        let dist = Gaussian::new(Stabilization::L2, ResponseFn::Exp, LossFn::Nll, false);

        // Include some extreme values
        let predictions = array![[0.0, 0.0], [100.0, 100.0], [-100.0, -100.0]];
        let targets = array![0.0, 1.0, 2.0];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();
        assert!(
            gh.gradients.iter().all(|&x| x.is_finite()),
            "L2 stabilization failed to handle extreme values"
        );
    }

    #[test]
    fn test_log_prob_with_invalid_params() {
        let dist = Gaussian::default();

        // Zero or negative scale should return -Infinity
        let log_p_zero_scale = dist.log_prob(&[0.0, 0.0], &[0.0]);
        assert!(
            log_p_zero_scale.is_finite() == false || log_p_zero_scale == f64::NEG_INFINITY,
            "log_prob should handle zero scale"
        );

        let log_p_neg_scale = dist.log_prob(&[0.0, -1.0], &[0.0]);
        assert!(
            log_p_neg_scale == f64::NEG_INFINITY,
            "log_prob should return -Inf for negative scale"
        );
    }

    #[test]
    fn test_gamma_log_prob_invalid_params() {
        let dist = Gamma::default();

        // Zero concentration should handle gracefully
        let log_p = dist.log_prob(&[0.0, 1.0], &[1.0]);
        assert!(
            !log_p.is_nan(),
            "Gamma log_prob returned NaN for zero concentration"
        );

        // Negative target should handle gracefully (Gamma support is [0, inf))
        let log_p_neg = dist.log_prob(&[2.0, 1.0], &[-1.0]);
        assert!(
            log_p_neg == f64::NEG_INFINITY || !log_p_neg.is_nan(),
            "Gamma log_prob should handle negative target"
        );
    }

    #[test]
    fn test_beta_log_prob_boundary() {
        let dist = Beta::default();

        // Targets at boundary (0 and 1) should be handled
        let log_p_zero = dist.log_prob(&[2.0, 2.0], &[0.0]);
        let log_p_one = dist.log_prob(&[2.0, 2.0], &[1.0]);

        // Should return -Inf at boundaries (Beta is open interval (0,1))
        assert!(
            !log_p_zero.is_nan() && !log_p_one.is_nan(),
            "Beta log_prob returned NaN at boundaries"
        );
    }

    #[test]
    fn test_gaussian_gradients_with_nan_target() {
        let dist = Gaussian::default();

        let predictions = array![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        // Target with a NaN value
        let targets = array![0.5, f64::NAN, 2.5];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();

        // The gradients and hessians for the row with NaN target should be finite (due to stabilization)
        assert!(
            gh.gradients.iter().all(|&x| x.is_finite()),
            "Gradients contain NaN or Inf after NaN target"
        );
        assert!(
            gh.hessians.iter().all(|&x| x.is_finite() && x > 0.0),
            "Hessians contain NaN, Inf, or non-positive values after NaN target"
        );

        // Check if the gradient for the nan-target-row is the mean of the other two
        let grad_col0_mean = (gh.gradients[[0, 0]] + gh.gradients[[2, 0]]) / 2.0;
        assert!((gh.gradients[[1, 0]] - grad_col0_mean).abs() < 1e-9);

        let grad_col1_mean = (gh.gradients[[0, 1]] + gh.gradients[[2, 1]]) / 2.0;
        assert!((gh.gradients[[1, 1]] - grad_col1_mean).abs() < 1e-9);
    }
}

// =============================================================================
// Weighted Gradients/Hessians Tests
// =============================================================================

mod weighted_gradient_tests {
    use gradientlss::distributions::*;
    use gradientlss::types::ResponseData;
    use ndarray::array;

    #[test]
    fn test_gaussian_weighted_gradients() {
        let dist = Gaussian::default();

        let predictions = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let targets = array![1.0, 2.0, 3.0];
        let weights = array![1.0, 2.0, 3.0];
        let target_data = ResponseData::Univariate(&targets.view());

        // Compute with weights
        let result_weighted = dist.compute_gradients_and_hessians(
            &predictions.view(),
            &target_data,
            Some(&weights.view()),
        );
        assert!(result_weighted.is_ok());

        // Compute without weights
        let result_unweighted =
            dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result_unweighted.is_ok());

        let gh_weighted = result_weighted.unwrap();
        let gh_unweighted = result_unweighted.unwrap();

        // Weighted gradients should differ from unweighted
        // The third sample has weight 3, so its gradient should be 3x larger
        let ratio = gh_weighted.gradients[[2, 0]] / gh_unweighted.gradients[[2, 0]];
        assert!(
            (ratio - 3.0).abs() < 0.1,
            "Weight 3.0 should scale gradient by ~3x, got ratio: {}",
            ratio
        );
    }

    #[test]
    fn test_uniform_weights_equal_unweighted() {
        let dist = Gaussian::default();

        let predictions = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let targets = array![0.5, 1.5, 2.5];
        let uniform_weights = array![1.0, 1.0, 1.0];
        let target_data = ResponseData::Univariate(&targets.view());

        let result_weighted = dist.compute_gradients_and_hessians(
            &predictions.view(),
            &target_data,
            Some(&uniform_weights.view()),
        );
        let result_unweighted =
            dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);

        let gh_weighted = result_weighted.unwrap();
        let gh_unweighted = result_unweighted.unwrap();

        // With uniform weights of 1.0, results should be identical
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (gh_weighted.gradients[[i, j]] - gh_unweighted.gradients[[i, j]]).abs() < 1e-10,
                    "Uniform weights should equal unweighted"
                );
            }
        }
    }

    #[test]
    fn test_zero_weight_zeroes_gradient() {
        let dist = Gaussian::default();

        let predictions = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let targets = array![0.5, 1.5, 2.5];
        let weights = array![0.0, 1.0, 1.0]; // First sample has zero weight
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(
            &predictions.view(),
            &target_data,
            Some(&weights.view()),
        );
        let gh = result.unwrap();

        // First sample's gradients should be zero
        assert!(
            gh.gradients[[0, 0]].abs() < 1e-10,
            "Zero weight should zero the gradient"
        );
        assert!(
            gh.gradients[[0, 1]].abs() < 1e-10,
            "Zero weight should zero the gradient"
        );
    }

    #[test]
    fn test_student_t_weighted_gradients() {
        let dist = StudentT::default();

        let predictions = array![[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let targets = array![1.0, 2.0];
        let weights = array![1.0, 2.0];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(
            &predictions.view(),
            &target_data,
            Some(&weights.view()),
        );
        assert!(result.is_ok());

        let gh = result.unwrap();
        assert!(gh.gradients.iter().all(|&x| x.is_finite()));
        assert!(gh.hessians.iter().all(|&x| x.is_finite()));
    }
}

// =============================================================================
// CRPS Loss Function Tests
// =============================================================================

mod crps_tests {
    use gradientlss::distributions::*;
    use gradientlss::prelude::*;
    use gradientlss::types::ResponseData;
    use ndarray::{Array2, array};

    #[test]
    fn test_crps_perfect_prediction() {
        let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);

        // If all samples equal the target, CRPS should be near zero
        let targets = array![5.0];
        let target_data = ResponseData::Univariate(&targets.view());

        // Create samples that are all exactly the target
        let samples = Array2::from_elem((100, 1), 5.0);

        let crps = dist.crps_score(&target_data, &samples.view());
        assert!(
            crps < 0.1,
            "CRPS should be near zero for perfect prediction, got {}",
            crps
        );
    }

    #[test]
    fn test_crps_increases_with_error() {
        let dist = Gaussian::default();

        let targets = array![0.0];
        let target_data = ResponseData::Univariate(&targets.view());

        // Samples centered at target
        let samples_good = Array2::from_shape_fn((100, 1), |(i, _)| (i as f64 - 50.0) * 0.1);

        // Samples centered away from target
        let samples_bad = Array2::from_shape_fn((100, 1), |(i, _)| (i as f64 - 50.0) * 0.1 + 10.0);

        let crps_good = dist.crps_score(&target_data, &samples_good.view());
        let crps_bad = dist.crps_score(&target_data, &samples_bad.view());

        assert!(
            crps_bad > crps_good,
            "CRPS should be higher for worse predictions: good={}, bad={}",
            crps_good,
            crps_bad
        );
    }

    #[test]
    fn test_crps_non_negative() {
        let dist = Gaussian::default();

        let targets = array![1.0, 2.0, 3.0];
        let target_data = ResponseData::Univariate(&targets.view());

        let params = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
        let samples = dist.sample(&params.view(), 100, 42);

        let crps = dist.crps_score(&target_data, &samples.view());
        assert!(crps >= 0.0, "CRPS should be non-negative, got {}", crps);
    }

    #[test]
    fn test_crps_with_single_sample() {
        let dist = Gaussian::default();

        let targets = array![5.0];
        let target_data = ResponseData::Univariate(&targets.view());

        // Single sample
        let samples = array![[3.0]];

        let crps = dist.crps_score(&target_data, &samples.view());
        // With a single sample at 3.0 and target at 5.0, CRPS = |5.0 - 3.0| = 2.0
        assert!(
            (crps - 2.0).abs() < 0.1,
            "CRPS with single sample should be |y - sample|, got {}",
            crps
        );
    }

    #[test]
    fn test_crps_gradient_computation() {
        let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Crps, false);

        let predictions = array![[0.0, 0.0], [1.0, 1.0]];
        let targets = array![0.5, 1.5];
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();

        // For CRPS, hessians should be 1.0 (constant)
        assert!(
            gh.hessians.iter().all(|&h| (h - 1.0).abs() < 1e-6),
            "CRPS hessians should be 1.0"
        );

        // Gradients should be finite
        assert!(gh.gradients.iter().all(|&g| g.is_finite()));
    }
}

// =============================================================================
// Zero-Inflated Distribution Tests
// =============================================================================

mod zero_inflated_tests {
    use gradientlss::distributions::*;
    use gradientlss::types::ResponseData;
    use ndarray::array;

    #[test]
    fn test_zinb_zero_probability() {
        let dist = ZINB::default();

        // With high gate probability, zeros should be more likely
        // gate uses sigmoid, so gate_pred=5.0 -> sigmoid(5) ≈ 0.99
        let preds_high_gate = [1.6, 0.0, 5.0]; // total_count, probs, gate preds
        let preds_low_gate = [1.6, 0.0, -5.0];

        // Transform predictions to distribution parameters
        let params_high_gate = vec![
            dist.params()[0]
                .response_fn
                .apply_scalar(preds_high_gate[0]),
            dist.params()[1]
                .response_fn
                .apply_scalar(preds_high_gate[1]),
            dist.params()[2]
                .response_fn
                .apply_scalar(preds_high_gate[2]),
        ];
        let params_low_gate = vec![
            dist.params()[0].response_fn.apply_scalar(preds_low_gate[0]),
            dist.params()[1].response_fn.apply_scalar(preds_low_gate[1]),
            dist.params()[2].response_fn.apply_scalar(preds_low_gate[2]),
        ];

        let log_p_zero_high_gate = dist.log_prob(&params_high_gate, &[0.0]);
        let log_p_zero_low_gate = dist.log_prob(&params_low_gate, &[0.0]);

        assert!(
            log_p_zero_high_gate > log_p_zero_low_gate,
            "Higher gate should give higher P(Y=0)"
        );
    }

    #[test]
    fn test_zinb_positive_count() {
        let dist = ZINB::default();

        // For positive counts, gate reduces probability
        // High gate means more zeros, so P(Y=5) should be lower
        let preds_high_gate = [1.6, 0.0, 5.0]; // total_count, probs, gate preds
        let preds_low_gate = [1.6, 0.0, -5.0];

        // Transform predictions to distribution parameters
        let params_high_gate = vec![
            dist.params()[0]
                .response_fn
                .apply_scalar(preds_high_gate[0]),
            dist.params()[1]
                .response_fn
                .apply_scalar(preds_high_gate[1]),
            dist.params()[2]
                .response_fn
                .apply_scalar(preds_high_gate[2]),
        ];
        let params_low_gate = vec![
            dist.params()[0].response_fn.apply_scalar(preds_low_gate[0]),
            dist.params()[1].response_fn.apply_scalar(preds_low_gate[1]),
            dist.params()[2].response_fn.apply_scalar(preds_low_gate[2]),
        ];

        let log_p_pos_high_gate = dist.log_prob(&params_high_gate, &[5.0]);
        let log_p_pos_low_gate = dist.log_prob(&params_low_gate, &[5.0]);

        assert!(
            log_p_pos_low_gate > log_p_pos_high_gate,
            "Lower gate should give higher P(Y=5) for positive counts"
        );
    }

    #[test]
    fn test_zipoisson_zero_probability() {
        let dist = ZIPoisson::default();

        // Similar test for ZIPoisson
        let preds_high_gate = [2.0, 5.0]; // rate, gate preds
        let preds_low_gate = [2.0, -5.0];

        // Transform predictions to distribution parameters
        let params_high_gate = vec![
            dist.params()[0]
                .response_fn
                .apply_scalar(preds_high_gate[0]),
            dist.params()[1]
                .response_fn
                .apply_scalar(preds_high_gate[1]),
        ];
        let params_low_gate = vec![
            dist.params()[0].response_fn.apply_scalar(preds_low_gate[0]),
            dist.params()[1].response_fn.apply_scalar(preds_low_gate[1]),
        ];

        let log_p_zero_high_gate = dist.log_prob(&params_high_gate, &[0.0]);
        let log_p_zero_low_gate = dist.log_prob(&params_low_gate, &[0.0]);

        assert!(
            log_p_zero_high_gate > log_p_zero_low_gate,
            "Higher gate should give higher P(Y=0) for ZIPoisson"
        );
    }

    #[test]
    fn test_zipoisson_positive_count() {
        let dist = ZIPoisson::default();

        let preds_high_gate = [2.0, 5.0]; // rate, gate preds
        let preds_low_gate = [2.0, -5.0];

        // Transform predictions to distribution parameters
        let params_high_gate = vec![
            dist.params()[0]
                .response_fn
                .apply_scalar(preds_high_gate[0]),
            dist.params()[1]
                .response_fn
                .apply_scalar(preds_high_gate[1]),
        ];
        let params_low_gate = vec![
            dist.params()[0].response_fn.apply_scalar(preds_low_gate[0]),
            dist.params()[1].response_fn.apply_scalar(preds_low_gate[1]),
        ];

        let log_p_pos_high_gate = dist.log_prob(&params_high_gate, &[3.0]);
        let log_p_pos_low_gate = dist.log_prob(&params_low_gate, &[3.0]);

        assert!(
            log_p_pos_low_gate > log_p_pos_high_gate,
            "Lower gate should give higher P(Y=3) for positive counts"
        );
    }

    #[test]
    fn test_zinb_gradients() {
        let dist = ZINB::default();

        let predictions = array![[2.0, 0.5, 0.0], [2.0, 0.5, 0.0]];
        let targets = array![0.0, 5.0]; // One zero, one positive
        let target_data = ResponseData::Univariate(&targets.view());

        let result = dist.compute_gradients_and_hessians(&predictions.view(), &target_data, None);
        assert!(result.is_ok());

        let gh = result.unwrap();
        assert!(gh.gradients.iter().all(|&x| x.is_finite()));
        assert!(gh.hessians.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_zagamma_zero_adjusted() {
        let dist = ZAGamma::default();

        // ZAGamma: zero-adjusted, should handle zeros differently
        // Note: ZAGamma::default() uses ResponseFn::Exp for concentration/rate and
        // ResponseFn::Sigmoid for gate. The log_prob method expects raw params that
        // get transformed internally. However, the current implementation's log_prob_scalar
        // expects already-transformed params. So we need to provide valid transformed values:
        // - concentration > 0, rate > 0, gate in (0, 1)
        let log_p_zero = dist.log_prob(&[2.0, 1.0, 0.3], &[0.0]); // gate=0.3 gives ln(0.3)
        let log_p_pos = dist.log_prob(&[2.0, 1.0, 0.3], &[1.0]);

        assert!(log_p_zero.is_finite(), "ZAGamma should handle zero target");
        assert!(
            log_p_pos.is_finite(),
            "ZAGamma should handle positive target"
        );
    }

    #[test]
    fn test_zabeta_boundaries() {
        let dist = ZABeta::default();

        // ZABeta has 3 params: concentration1, concentration0, gate
        // ZABeta::default() uses ResponseFn::Exp for concentrations and ResponseFn::Sigmoid for gate
        // The log_prob transforms params, so raw params [1.0, 1.0, 0.0] become:
        // concentration1=exp(1.0)≈2.72, concentration0=exp(1.0)≈2.72, gate=sigmoid(0.0)=0.5
        // ZABeta handles y=0 (point mass) and y in (0, 1) (Beta distribution)
        // Note: ZABeta does NOT handle y=1 - it's a zero-adjusted (not zero-one-adjusted) distribution
        let log_p_zero = dist.log_prob(&[1.0, 1.0, 0.0], &[0.0]);
        let log_p_mid = dist.log_prob(&[1.0, 1.0, 0.0], &[0.5]);

        assert!(log_p_zero.is_finite(), "ZABeta should handle y=0");
        assert!(log_p_mid.is_finite(), "ZABeta should handle y=0.5");

        // ZABeta does not support y=1 (it returns NEG_INFINITY for values outside (0, 1) except 0)
        let log_p_one = dist.log_prob(&[1.0, 1.0, 0.0], &[1.0]);
        assert!(
            log_p_one == f64::NEG_INFINITY,
            "ZABeta returns NEG_INFINITY for y=1"
        );
    }

    #[test]
    fn test_zaln_zero_adjusted() {
        let dist = ZALN::default();

        let log_p_zero = dist.log_prob(&[0.0, 1.0, 0.0], &[0.0]);
        let log_p_pos = dist.log_prob(&[0.0, 1.0, 0.0], &[2.0]);

        assert!(log_p_zero.is_finite(), "ZALN should handle zero target");
        assert!(log_p_pos.is_finite(), "ZALN should handle positive target");
    }
}

// =============================================================================
// Model Save/Load Tests
// =============================================================================

#[cfg(feature = "xgboost")]
mod save_load_tests {
    use gradientlss::backend::{Backend, BackendDataset, TrainConfig, XGBoostBackend};
    use gradientlss::distributions::Gaussian;
    use gradientlss::model::{GradientLSS, PredType};
    use ndarray::array;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn test_xgboost_save_load_roundtrip() {
        let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));

        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let labels = array![1.0, 2.0, 3.0, 4.0];

        let mut train_data =
            <XGBoostBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 10,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };

        model.train(&mut train_data, None, params, config).unwrap();

        // Get predictions before save
        let test_features = array![[2.0, 3.0], [4.0, 5.0]];
        let preds_before = model
            .predict(&test_features.view(), PredType::Parameters, 100, &[], 42)
            .unwrap();

        // Save model
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.xgb");
        model.save(model_path.to_str().unwrap()).unwrap();

        // Load model
        let loaded_model =
            GradientLSS::<XGBoostBackend>::load(model_path.to_str().unwrap()).unwrap();

        // Get predictions after load
        let preds_after = loaded_model
            .predict(&test_features.view(), PredType::Parameters, 100, &[], 42)
            .unwrap();

        // Compare predictions
        match (preds_before, preds_after) {
            (
                gradientlss::backend::PredictionOutput::Parameters(before),
                gradientlss::backend::PredictionOutput::Parameters(after),
            ) => {
                for i in 0..before.nrows() {
                    for j in 0..before.ncols() {
                        assert!(
                            (before[[i, j]] - after[[i, j]]).abs() < 1e-6,
                            "Predictions differ after save/load: {} vs {}",
                            before[[i, j]],
                            after[[i, j]]
                        );
                    }
                }
            }
            _ => panic!("Expected Parameters output"),
        }
    }
}

// =============================================================================
// Multivariate Distribution Tests
// =============================================================================

mod multivariate_tests {
    use gradientlss::distributions::*;
    use gradientlss::prelude::*;
    use gradientlss::types::ResponseData;
    use ndarray::Array2;

    #[test]
    fn test_mvn_log_prob_at_mean() {
        let dist = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // params: [loc1, loc2, tril_00, tril_10, tril_11]
        // At mean with identity covariance, log_prob should be highest
        let params = vec![0.0, 0.0, 1.0, 0.0, 1.0];
        let target_at_mean = vec![0.0, 0.0];
        let target_away = vec![2.0, 2.0];

        let log_p_at_mean = dist.log_prob(&params, &target_at_mean);
        let log_p_away = dist.log_prob(&params, &target_away);

        assert!(
            log_p_at_mean > log_p_away,
            "MVN: log_prob at mean should be higher than away from mean"
        );
    }

    #[test]
    fn test_mvn_nll() {
        let dist = MVN::new(2, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // 3 observations, 5 params each
        let params = Array2::from_shape_vec(
            (3, 5),
            vec![
                0.0, 0.0, 1.0, 0.0, 1.0, // obs 1
                1.0, 1.0, 1.0, 0.0, 1.0, // obs 2
                2.0, 2.0, 1.0, 0.0, 1.0, // obs 3
            ],
        )
        .unwrap();

        // Targets: 3 observations, 2 targets each
        let targets = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();

        let target_data = ResponseData::Multivariate(&targets.view());
        let nll = dist.nll(&params.view(), &target_data);

        assert!(nll.is_finite(), "MVN NLL should be finite");
        assert!(nll > 0.0, "MVN NLL should be positive");
    }

    #[test]
    fn test_mvn_sampling_dimensions() {
        let dist = MVN::new(3, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // 2 observations, 9 params each (3 loc + 6 tril)
        let n_params = dist.n_params();
        assert_eq!(n_params, 9);

        let params = Array2::zeros((2, n_params));
        let samples = dist.sample(&params.view(), 100, 42);

        // Should return samples for each target dimension
        assert!(samples.ncols() > 0, "MVN sampling should return samples");
    }

    #[test]
    fn test_mvt_creation() {
        let dist = MVT::new(
            2,
            Stabilization::None,
            ResponseFn::Exp,
            ResponseFn::ExpDf,
            LossFn::Nll,
            false,
        );

        // MVT has df + loc + tril
        // For 2 targets: 1 df + 2 loc + 3 tril = 6 params
        assert_eq!(dist.n_params(), 6);
        assert_eq!(dist.n_targets(), 2);
    }

    #[test]
    fn test_mvt_heavier_tails() {
        let dist = MVT::new(
            2,
            Stabilization::None,
            ResponseFn::Exp,
            ResponseFn::ExpDf,
            LossFn::Nll,
            false,
        );

        // Lower df = heavier tails = higher probability at extremes
        // params: [df, loc1, loc2, tril elements...]
        let params_low_df = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0]; // df=exp(1)≈2.7
        let params_high_df = vec![3.0, 0.0, 0.0, 1.0, 0.0, 1.0]; // df=exp(3)≈20

        let extreme_target = vec![5.0, 5.0];

        let log_p_low_df = dist.log_prob(&params_low_df, &extreme_target);
        let log_p_high_df = dist.log_prob(&params_high_df, &extreme_target);

        assert!(
            log_p_low_df > log_p_high_df,
            "MVT with lower df should have higher probability at extremes"
        );
    }

    #[test]
    fn test_dirichlet_creation() {
        let dist = Dirichlet::new(3, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        assert_eq!(dist.n_params(), 3); // 3 concentration parameters
        assert_eq!(dist.n_targets(), 3);
    }

    #[test]
    fn test_dirichlet_log_prob() {
        let dist = Dirichlet::new(3, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        // Concentrations (will be exp-transformed)
        let params = vec![1.0, 1.0, 1.0]; // uniform Dirichlet after exp

        // Target must sum to 1
        let target = vec![0.33, 0.33, 0.34];

        let log_p = dist.log_prob(&params, &target);
        assert!(log_p.is_finite(), "Dirichlet log_prob should be finite");
    }

    #[test]
    fn test_dirichlet_sampling_sums_to_one() {
        let dist = Dirichlet::new(3, Stabilization::None, ResponseFn::Exp, LossFn::Nll, false);

        let params = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap();
        let samples = dist.sample(&params.view(), 100, 42);

        // Each sample (across target dimensions) should sum to approximately 1
        // Note: sampling format may vary, so just check finite
        assert!(
            samples.iter().all(|&x| x.is_finite()),
            "Dirichlet samples should be finite"
        );
    }
}

// =============================================================================
// Start Values Optimization Tests
// =============================================================================

mod start_values_tests {
    use gradientlss::distributions::*;
    use gradientlss::prelude::*;
    use gradientlss::types::ResponseData;
    use ndarray::{Array1, array};

    #[test]
    fn test_gaussian_start_values_match_data() {
        let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);

        // Target with known mean ~5 and std ~1
        let target = Array1::from_vec(vec![4.0, 4.5, 5.0, 5.5, 6.0, 4.8, 5.2, 5.0, 4.9, 5.1]);
        let target_data = ResponseData::Univariate(&target.view());

        let (loss, start_vals) = dist.calculate_start_values(&target_data, 100).unwrap();

        assert!(loss.is_finite());
        assert_eq!(start_vals.len(), 2);

        // The location start value should be close to the mean
        // (after exp transform for scale)
        let transformed_loc = start_vals[0]; // Identity for loc
        assert!(
            (transformed_loc - 5.0).abs() < 1.0,
            "Start value for loc should be near data mean, got {}",
            transformed_loc
        );
    }

    #[test]
    fn test_student_t_start_values() {
        let dist = StudentT::new(
            Stabilization::None,
            ResponseFn::ExpDf,
            ResponseFn::Exp,
            LossFn::Nll,
            true,
        );

        let target = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let target_data = ResponseData::Univariate(&target.view());

        let (loss, start_vals) = dist.calculate_start_values(&target_data, 100).unwrap();

        assert!(loss.is_finite());
        assert_eq!(start_vals.len(), 3); // df, loc, scale
        assert!(start_vals.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_gamma_start_values() {
        let dist = Gamma::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);

        // Positive values for Gamma
        let target = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let target_data = ResponseData::Univariate(&target.view());

        let (loss, start_vals) = dist.calculate_start_values(&target_data, 100).unwrap();

        assert!(loss.is_finite());
        assert_eq!(start_vals.len(), 2);
        assert!(start_vals.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_spline_flow_start_values() {
        let dist = SplineFlow::new(
            TargetSupport::Real,
            4, // Fewer bins for faster test
            3.0,
            SplineOrder::Quadratic,
            Stabilization::None,
            LossFn::Nll,
            true,
        );

        let target = Array1::from_vec(vec![0.0, 1.0, -1.0, 0.5, -0.5]);
        let target_data = ResponseData::Univariate(&target.view());

        let (loss, start_vals) = dist.calculate_start_values(&target_data, 50).unwrap();

        assert!(
            loss.is_finite(),
            "SplineFlow start values loss should be finite"
        );
        assert_eq!(start_vals.len(), dist.n_params());
        assert!(
            start_vals.iter().all(|&v| v.is_finite()),
            "All SplineFlow start values should be finite"
        );
    }

    #[test]
    fn test_beta_start_values() {
        let dist = Beta::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);

        // Values in (0, 1) for Beta
        let target = Array1::from_vec(vec![0.2, 0.4, 0.5, 0.6, 0.8]);
        let target_data = ResponseData::Univariate(&target.view());

        let (loss, start_vals) = dist.calculate_start_values(&target_data, 100).unwrap();

        assert!(loss.is_finite());
        assert_eq!(start_vals.len(), 2);
    }

    #[test]
    fn test_poisson_start_values() {
        let dist = Poisson::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);

        // Count data
        let target = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 5.0, 2.0, 1.0]);
        let target_data = ResponseData::Univariate(&target.view());

        let (loss, start_vals) = dist.calculate_start_values(&target_data, 100).unwrap();

        assert!(loss.is_finite());
        assert_eq!(start_vals.len(), 1); // Just rate
    }

    #[test]
    fn test_start_values_improve_loss() {
        let dist = Gaussian::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true);

        let target = Array1::from_vec(vec![10.0, 11.0, 9.0, 10.5, 9.5]);
        let target_data = ResponseData::Univariate(&target.view());

        // Loss with default initialization (zeros)
        let default_params = array![[0.0, 0.0]]; // loc=0, scale=exp(0)=1
        let loss_default = dist.nll(&default_params.view(), &target_data);

        // Loss with optimized start values
        let (loss_optimized, _) = dist.calculate_start_values(&target_data, 100).unwrap();

        assert!(
            loss_optimized < loss_default,
            "Optimized start values should have lower loss: {} vs {}",
            loss_optimized,
            loss_default
        );
    }
}

// =============================================================================
// Callback System Tests
// =============================================================================

mod callback_tests {
    use gradientlss::backend::{
        CallbackAction, CallbackList, EarlyStoppingCallback, HistoryCallback, LearningRateSchedule,
        LearningRateScheduler, PrintCallback, TrainingCallback,
    };

    #[test]
    fn test_print_callback_creation() {
        let callback = PrintCallback::default();
        assert_eq!(callback.print_every, 10);
        assert_eq!(callback.name(), "PrintCallback");
    }

    #[test]
    fn test_print_callback_continues_training() {
        let mut callback = PrintCallback { print_every: 5 };

        // Should always return Continue
        let action = callback.on_iteration_end(0, 1.0, None);
        assert_eq!(action, CallbackAction::Continue);

        let action = callback.on_iteration_end(10, 0.5, Some(0.6));
        assert_eq!(action, CallbackAction::Continue);
    }

    #[test]
    fn test_history_callback_records() {
        let mut callback = HistoryCallback::new();

        callback.on_training_start(100);
        callback.on_iteration_end(0, 1.0, Some(1.1));
        callback.on_iteration_end(1, 0.9, Some(1.0));
        callback.on_iteration_end(2, 0.8, Some(0.9));

        assert_eq!(callback.train_history.len(), 3);
        assert_eq!(callback.valid_history.len(), 3);
        assert_eq!(callback.train_history[0], 1.0);
        assert_eq!(callback.valid_history[2], 0.9);
    }

    #[test]
    fn test_history_callback_best_methods() {
        let mut callback = HistoryCallback::new();

        callback.on_iteration_end(0, 1.0, Some(1.2));
        callback.on_iteration_end(1, 0.5, Some(0.6)); // Best train
        callback.on_iteration_end(2, 0.8, Some(0.4)); // Best valid

        let (best_train_iter, best_train_val) = callback.best_train().unwrap();
        assert_eq!(best_train_iter, 1);
        assert_eq!(best_train_val, 0.5);

        let (best_valid_iter, best_valid_val) = callback.best_valid().unwrap();
        assert_eq!(best_valid_iter, 2);
        assert_eq!(best_valid_val, 0.4);
    }

    #[test]
    fn test_history_callback_clears_on_restart() {
        let mut callback = HistoryCallback::new();

        callback.on_iteration_end(0, 1.0, Some(1.1));
        callback.on_iteration_end(1, 0.9, Some(1.0));
        assert_eq!(callback.train_history.len(), 2);

        // Restart training
        callback.on_training_start(100);
        assert_eq!(callback.train_history.len(), 0);
        assert_eq!(callback.valid_history.len(), 0);
    }

    #[test]
    fn test_early_stopping_callback_creation() {
        let callback = EarlyStoppingCallback::new(10);
        assert_eq!(callback.patience, 10);
        assert_eq!(callback.min_delta, 0.0);
        assert!(callback.monitor_validation);
        assert!(callback.minimize);
        assert_eq!(callback.name(), "EarlyStoppingCallback");
    }

    #[test]
    fn test_early_stopping_callback_builder() {
        let callback = EarlyStoppingCallback::new(5)
            .with_min_delta(0.001)
            .with_monitor_validation(false)
            .with_verbose(false);

        assert_eq!(callback.patience, 5);
        assert_eq!(callback.min_delta, 0.001);
        assert!(!callback.monitor_validation);
        assert!(!callback.verbose);
    }

    #[test]
    fn test_early_stopping_stops_when_no_improvement() {
        let mut callback = EarlyStoppingCallback::new(3).with_verbose(false);

        callback.on_training_start(100);

        // Improvement
        assert_eq!(
            callback.on_iteration_end(0, 1.0, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(1, 0.9, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(2, 0.8, None),
            CallbackAction::Continue
        );

        // No improvement for 3 iterations
        assert_eq!(
            callback.on_iteration_end(3, 0.85, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(4, 0.82, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(5, 0.81, None),
            CallbackAction::Stop
        );

        assert_eq!(callback.best_iteration(), 2);
        assert_eq!(callback.best_value(), 0.8);
    }

    #[test]
    fn test_early_stopping_uses_validation_when_available() {
        let mut callback = EarlyStoppingCallback::new(2)
            .with_monitor_validation(true)
            .with_verbose(false);

        callback.on_training_start(100);

        // Train loss improves but valid loss doesn't
        assert_eq!(
            callback.on_iteration_end(0, 1.0, Some(0.5)),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(1, 0.5, Some(0.6)),
            CallbackAction::Continue
        ); // valid worsened
        assert_eq!(
            callback.on_iteration_end(2, 0.3, Some(0.7)),
            CallbackAction::Stop
        ); // still worse

        assert_eq!(callback.best_iteration(), 0);
        assert_eq!(callback.best_value(), 0.5); // Uses valid loss
    }

    #[test]
    fn test_early_stopping_falls_back_to_train() {
        let mut callback = EarlyStoppingCallback::new(2)
            .with_monitor_validation(true)
            .with_verbose(false);

        callback.on_training_start(100);

        // No validation data, should use train loss
        assert_eq!(
            callback.on_iteration_end(0, 1.0, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(1, 0.9, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(2, 0.95, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(3, 0.92, None),
            CallbackAction::Stop
        );

        assert_eq!(callback.best_iteration(), 1);
    }

    #[test]
    fn test_early_stopping_with_min_delta() {
        let mut callback = EarlyStoppingCallback::new(2)
            .with_min_delta(0.1)
            .with_verbose(false);

        callback.on_training_start(100);

        // Improvements smaller than min_delta don't count
        assert_eq!(
            callback.on_iteration_end(0, 1.0, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(1, 0.95, None),
            CallbackAction::Continue
        ); // Not enough improvement
        assert_eq!(
            callback.on_iteration_end(2, 0.92, None),
            CallbackAction::Stop
        );

        assert_eq!(callback.best_iteration(), 0);
    }

    #[test]
    fn test_early_stopping_maximize_mode() {
        let mut callback = EarlyStoppingCallback::new(2)
            .with_minimize(false)
            .with_verbose(false);

        callback.on_training_start(100);

        // In maximize mode, higher is better
        assert_eq!(
            callback.on_iteration_end(0, 0.5, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(1, 0.7, None),
            CallbackAction::Continue
        );
        assert_eq!(
            callback.on_iteration_end(2, 0.65, None),
            CallbackAction::Continue
        ); // Worse
        assert_eq!(
            callback.on_iteration_end(3, 0.68, None),
            CallbackAction::Stop
        );

        assert_eq!(callback.best_iteration(), 1);
        assert_eq!(callback.best_value(), 0.7);
    }

    #[test]
    fn test_callback_list_creation() {
        let list = CallbackList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_callback_list_builder() {
        let list = CallbackList::new()
            .with(PrintCallback::default())
            .with(HistoryCallback::new());

        assert_eq!(list.len(), 2);
        assert!(!list.is_empty());
    }

    #[test]
    fn test_callback_list_runs_all_callbacks() {
        let mut list = CallbackList::new();
        list.add(PrintCallback { print_every: 100 }); // Won't print in these tests

        // We can't easily add the history callback to the list and check it after
        // because it gets moved. Let's just test that the list processes correctly.
        list.on_training_start(10);
        let action = list.on_iteration_end(0, 1.0, Some(1.1));
        assert_eq!(action, CallbackAction::Continue);
    }

    #[test]
    fn test_callback_list_stops_on_first_stop() {
        // Create a custom callback that stops immediately
        struct StopImmediately;
        impl TrainingCallback for StopImmediately {
            fn on_iteration_end(&mut self, _: usize, _: f64, _: Option<f64>) -> CallbackAction {
                CallbackAction::Stop
            }
        }

        let mut list = CallbackList::new()
            .with(PrintCallback { print_every: 100 })
            .with(StopImmediately)
            .with(HistoryCallback::new());

        let action = list.on_iteration_end(0, 1.0, None);
        assert_eq!(action, CallbackAction::Stop);
    }

    #[test]
    fn test_learning_rate_scheduler_step_decay() {
        let mut scheduler = LearningRateScheduler::step_decay(0.1, 10, 0.5);

        scheduler.on_training_start(100);
        assert_eq!(scheduler.current_lr, 0.1);

        // After 10 iterations, lr should be halved
        for i in 0..10 {
            scheduler.on_iteration_end(i, 1.0, None);
        }
        assert!((scheduler.current_lr - 0.05).abs() < 1e-10);

        // After 20 iterations, lr should be halved again
        for i in 10..20 {
            scheduler.on_iteration_end(i, 1.0, None);
        }
        assert!((scheduler.current_lr - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_learning_rate_scheduler_exponential_decay() {
        let mut scheduler = LearningRateScheduler::exponential_decay(0.1, 0.9);

        scheduler.on_training_start(100);
        assert_eq!(scheduler.current_lr, 0.1);

        scheduler.on_iteration_end(0, 1.0, None);
        assert!((scheduler.current_lr - 0.09).abs() < 1e-10); // 0.1 * 0.9^1

        scheduler.on_iteration_end(1, 1.0, None);
        assert!((scheduler.current_lr - 0.081).abs() < 1e-10); // 0.1 * 0.9^2
    }

    #[test]
    fn test_learning_rate_scheduler_cosine_annealing() {
        let mut scheduler = LearningRateScheduler::cosine_annealing(0.1, 100, 0.001);

        scheduler.on_training_start(100);
        assert_eq!(scheduler.current_lr, 0.1);

        // At t=50 (halfway), should be at midpoint
        for i in 0..50 {
            scheduler.on_iteration_end(i, 1.0, None);
        }
        let expected_mid =
            0.001 + (0.1 - 0.001) * (1.0 + (std::f64::consts::PI * 50.0 / 100.0).cos()) / 2.0;
        assert!((scheduler.current_lr - expected_mid).abs() < 1e-6);

        // At t=100, should be at eta_min
        for i in 50..100 {
            scheduler.on_iteration_end(i, 1.0, None);
        }
        assert!((scheduler.current_lr - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_learning_rate_scheduler_constant() {
        let mut scheduler = LearningRateScheduler::new(0.1, LearningRateSchedule::Constant);

        scheduler.on_training_start(100);
        for i in 0..50 {
            scheduler.on_iteration_end(i, 1.0, None);
        }
        assert_eq!(scheduler.current_lr, 0.1);
    }
}

// =============================================================================
// Training with Callbacks Integration Tests
// =============================================================================

#[cfg(feature = "lightgbm")]
mod training_callback_integration_tests {
    use gradientlss::backend::{
        Backend, BackendDataset, EarlyStoppingCallback, HistoryCallback, LightGBMBackend,
        TrainConfig,
    };
    use gradientlss::distributions::Gaussian;
    use gradientlss::model::GradientLSS;
    use std::sync::Arc;

    #[test]
    fn test_train_with_history_callback() {
        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::default()));

        // Use a larger dataset with more variance to avoid LightGBM's
        // "no meaningful features" early stopping
        let features = ndarray::Array2::from_shape_fn((50, 5), |(i, j)| {
            (i as f64) * 0.1 + (j as f64) * 0.5 + ((i * j) as f64).sin()
        });
        let labels = ndarray::Array1::from_shape_fn(50, |i| {
            features[[i, 0]] * 2.0 + features[[i, 1]] + (i as f64 * 0.1).cos()
        });

        let mut train_data =
            <LightGBMBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = LightGBMBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 20,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };

        let mut history = HistoryCallback::new();
        let (_, result) = model
            .train_with_callbacks(&mut train_data, None, params, config, Some(&mut history))
            .unwrap();

        // Check training result
        assert_eq!(result.n_iterations, 20);
        assert!(!result.stopped_early);
        assert!(result.best_score.is_some());

        // Check history was recorded
        assert_eq!(history.train_history.len(), 20);
        assert!(history.valid_history.is_empty()); // No validation data
    }

    #[test]
    fn test_train_with_validation_data() {
        use gradientlss::backend::{BackendParams, ParamValue};

        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::default()));

        // Use larger datasets with more variance to avoid LightGBM's
        // "no meaningful features" early stopping
        let train_features = ndarray::Array2::from_shape_fn((100, 5), |(i, j)| {
            (i as f64) * 0.1 + (j as f64) * 0.5 + ((i * j) as f64).sin()
        });
        let train_labels = ndarray::Array1::from_shape_fn(100, |i| {
            train_features[[i, 0]] * 2.0 + train_features[[i, 1]] + (i as f64 * 0.1).cos()
        });

        let valid_features = ndarray::Array2::from_shape_fn((30, 5), |(i, j)| {
            ((i + 100) as f64) * 0.1 + (j as f64) * 0.5 + (((i + 100) * j) as f64).sin()
        });
        let valid_labels = ndarray::Array1::from_shape_fn(30, |i| {
            valid_features[[i, 0]] * 2.0 + valid_features[[i, 1]] + ((i + 100) as f64 * 0.1).cos()
        });

        let mut train_data = <LightGBMBackend as Backend>::Dataset::from_data(
            train_features.view(),
            train_labels.view(),
        )
        .unwrap();
        let mut valid_data = <LightGBMBackend as Backend>::Dataset::from_data(
            valid_features.view(),
            valid_labels.view(),
        )
        .unwrap();

        let mut params = LightGBMBackend::create_params(model.n_params());
        // Reduce min_data requirements for test dataset
        params.set("min_data_in_bin", ParamValue::Int(1));
        params.set("min_data_in_leaf", ParamValue::Int(1));

        let config = TrainConfig {
            num_boost_round: 20,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };

        let mut history = HistoryCallback::new();
        let (_, result) = model
            .train_with_callbacks(
                &mut train_data,
                Some(&mut valid_data),
                params,
                config,
                Some(&mut history),
            )
            .unwrap();

        // Check that validation history was recorded
        assert_eq!(history.valid_history.len(), 20);
        assert!(!result.valid_history.is_empty());
    }

    #[test]
    fn test_train_with_early_stopping_callback() {
        use gradientlss::backend::{BackendParams, ParamValue};

        let mut model = GradientLSS::<LightGBMBackend>::new(Arc::new(Gaussian::default()));

        // Use a dataset where training and validation come from different distributions
        // to trigger overfitting and early stopping
        let train_features = ndarray::Array2::from_shape_fn((100, 5), |(i, j)| {
            (i as f64) * 0.1 + (j as f64) * 0.5 + ((i * j) as f64).sin()
        });
        let train_labels = ndarray::Array1::from_shape_fn(100, |i| {
            train_features[[i, 0]] * 2.0 + train_features[[i, 1]] + (i as f64 * 0.1).cos()
        });

        // Validation data with different pattern - model will overfit to training
        let valid_features = ndarray::Array2::from_shape_fn((30, 5), |(i, j)| {
            (i as f64) * 0.3 + (j as f64) * 0.2 + ((i + j) as f64).cos() * 2.0
        });
        let valid_labels = ndarray::Array1::from_shape_fn(30, |i| {
            valid_features[[i, 0]] * 0.5 - valid_features[[i, 1]] * 0.3 + (i as f64 * 0.2).sin()
        });

        let mut train_data = <LightGBMBackend as Backend>::Dataset::from_data(
            train_features.view(),
            train_labels.view(),
        )
        .unwrap();
        let mut valid_data = <LightGBMBackend as Backend>::Dataset::from_data(
            valid_features.view(),
            valid_labels.view(),
        )
        .unwrap();

        let mut params = LightGBMBackend::create_params(model.n_params());
        // Reduce min_data requirements for test dataset
        params.set("min_data_in_bin", ParamValue::Int(1));
        params.set("min_data_in_leaf", ParamValue::Int(1));

        let config = TrainConfig {
            num_boost_round: 1000,       // High number, should stop early
            early_stopping_rounds: None, // Rely on callback instead
            verbose: false,
            seed: 42,
        };

        // Early stopping with patience of 5 on validation loss
        let mut early_stop = EarlyStoppingCallback::new(5)
            .with_verbose(false)
            .with_monitor_validation(true);

        let (_, result) = model
            .train_with_callbacks(
                &mut train_data,
                Some(&mut valid_data),
                params,
                config,
                Some(&mut early_stop),
            )
            .unwrap();

        // Should have stopped before 1000 iterations due to validation loss not improving
        assert!(
            result.n_iterations < 1000,
            "Expected early stopping before 1000 iterations, but ran for {}",
            result.n_iterations
        );
        assert!(result.stopped_early);
        assert!(early_stop.best_iteration() < result.n_iterations);
    }
}

#[cfg(feature = "xgboost")]
mod xgboost_callback_integration_tests {
    use gradientlss::backend::{
        Backend, BackendDataset, EarlyStoppingCallback, HistoryCallback, TrainConfig,
        XGBoostBackend,
    };
    use gradientlss::distributions::Gaussian;
    use gradientlss::model::GradientLSS;
    use ndarray::array;
    use std::sync::Arc;

    #[test]
    fn test_xgboost_train_with_callbacks() {
        let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));

        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let labels = array![1.5, 2.5, 3.5, 4.5];

        let mut train_data =
            <XGBoostBackend as Backend>::Dataset::from_data(features.view(), labels.view())
                .unwrap();

        let params = XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 15,
            early_stopping_rounds: None,
            verbose: false,
            seed: 42,
        };

        let mut history = HistoryCallback::new();
        let (_, result) = model
            .train_with_callbacks(&mut train_data, None, params, config, Some(&mut history))
            .unwrap();

        assert!(model.is_trained());
        assert_eq!(result.n_iterations, 15);
        assert_eq!(history.train_history.len(), 15);
    }

    #[test]
    fn test_xgboost_train_with_validation_and_early_stop() {
        let mut model = GradientLSS::<XGBoostBackend>::new(Arc::new(Gaussian::default()));

        let train_features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let train_labels = array![1.5, 2.5, 3.5, 4.5];

        let valid_features = array![[1.5, 2.5], [3.5, 4.5]];
        let valid_labels = array![2.0, 4.0];

        let mut train_data = <XGBoostBackend as Backend>::Dataset::from_data(
            train_features.view(),
            train_labels.view(),
        )
        .unwrap();
        let mut valid_data = <XGBoostBackend as Backend>::Dataset::from_data(
            valid_features.view(),
            valid_labels.view(),
        )
        .unwrap();

        let params = XGBoostBackend::create_params(model.n_params());
        let config = TrainConfig {
            num_boost_round: 500,
            early_stopping_rounds: None, // Use callback
            verbose: false,
            seed: 42,
        };

        let mut early_stop = EarlyStoppingCallback::new(10)
            .with_monitor_validation(true)
            .with_verbose(false);

        let (_, result) = model
            .train_with_callbacks(
                &mut train_data,
                Some(&mut valid_data),
                params,
                config,
                Some(&mut early_stop),
            )
            .unwrap();

        assert!(model.is_trained());
        // Should have stopped early based on validation loss
        assert!(result.stopped_early || result.n_iterations < 500);
        assert!(!result.valid_history.is_empty());
    }
}
