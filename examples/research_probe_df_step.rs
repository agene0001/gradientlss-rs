//! Per-mode df-column gradient/Hessian sums at the fitted start values, and
//! the implied first-round root-leaf step −η·Σg/(Σh+λ) (XGBoost λ = 1,
//! η = 0.05). Run: cargo run --release --features xgboost --example research_probe_df_step
use gradientlss::distributions::{Distribution, HessianMode, StudentT};
use gradientlss::prelude::*;
use gradientlss::types::ResponseData;
use ndarray::{Array1, Array2};

fn load(file: &str, col: &str) -> Vec<f64> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("research/gradientlss/data");
    let text = std::fs::read_to_string(dir.join(file)).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').map(|s| s.trim_matches('"')).collect();
    let ci = header.iter().position(|h| *h == col).unwrap();
    lines.filter_map(|l| l.split(',').nth(ci).and_then(|v| v.trim().parse().ok())).collect()
}

fn main() {
    let eta = 0.05;
    let lambda = 1.0;
    for (file, col) in [("protein.csv", "RMSD"), ("power-plant.csv", "PE"), ("kin8nm.csv", "y")] {
        let mut y = load(file, col);
        y.truncate(12_000);
        let y = Array1::from(y);
        let n = y.len();
        let target = ResponseData::Univariate(&y.view());
        let base = StudentT::new(Stabilization::None, ResponseFn::ExpDf, ResponseFn::Exp, LossFn::Nll, true);
        let (_, sv) = base.calculate_start_values(&target, 50).unwrap();
        println!("\n== {file}: n={n}, fitted start raw {:?} -> nu0 = {:.4e}", sv.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>(), (sv[0]).exp() + 2.0);
        for &(label, raw_df) in &[("fitted", sv[0]), ("cap nu0=100", (98.0f64).ln()), ("nu0=10", (8.0f64).ln()), ("nu0=3", (1.0f64).ln())] {
            let preds = Array2::from_shape_fn((n, 3), |(_, j)| if j == 0 { raw_df } else { sv[j] });
            let tr = base.transform_params(&preds.view());
            print!("  {label:<13} ν={:<10.4e}", tr[[0, 0]]);
            for mode in [HessianMode::Observed, HessianMode::Fisher, HessianMode::Damped] {
                let d = base.clone().with_hessian_mode(mode);
                let (g, h) = d.analytical_gradients(&preds.view(), &tr.view(), &target).unwrap();
                let sg: f64 = g.column(0).sum();
                let sh: f64 = h.column(0).sum();
                let step = -eta * sg / (sh + lambda);
                let nu_after = (raw_df + step).exp() + 2.0;
                print!(" | {:?}: Σg={sg:+.3e} Σh={sh:+.3e} step={step:+.3e} ν₁={nu_after:.4e}", mode);
            }
            println!();
        }
    }
}
