use gradientlss::distributions::{Distribution, Gamma, Gaussian, StudentT};
use gradientlss::prelude::*;
use gradientlss::types::ResponseData;
use ndarray::Array1;
fn main() {
    let dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("research/gradientlss/data");
    for (file, col) in [
        ("power-plant.csv", "PE"),
        ("gaussian_train_sim.csv", "y"),
        ("kin8nm.csv", "y"),
    ] {
        let text = std::fs::read_to_string(dir.join(file)).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines
            .next()
            .unwrap()
            .split(',')
            .map(|s| s.trim_matches('"'))
            .collect();
        let ci = header.iter().position(|h| *h == col).unwrap();
        let y: Vec<f64> = lines
            .filter_map(|l| l.split(',').nth(ci).and_then(|v| v.trim().parse().ok()))
            .collect();
        let y = Array1::from(y);
        let mean = y.mean().unwrap();
        let sd = y.std(0.0);
        for (name, d) in [
            (
                "Gaussian",
                Box::new(Gaussian::new(
                    Stabilization::None,
                    ResponseFn::Exp,
                    LossFn::Nll,
                    true,
                )) as Box<dyn Distribution>,
            ),
            (
                "StudentT",
                Box::new(StudentT::new(
                    Stabilization::None,
                    ResponseFn::ExpDf,
                    ResponseFn::Exp,
                    LossFn::Nll,
                    true,
                )),
            ),
            (
                "Gamma",
                Box::new(Gamma::new(Stabilization::None, ResponseFn::Exp, LossFn::Nll, true)),
            ),
        ] {
            let sv = d.calculate_start_values(&ResponseData::Univariate(&y.view()), 50);
            println!(
                "{file:<24} y mean {mean:>9.3} sd {sd:>7.3}  {name:<9} start(raw) = {:?}",
                sv.map(|(loss, a)| (
                    format!("loss {loss:.3}"),
                    a.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>()
                ))
            );
        }
    }
}
