//! Reproduce the start-value L-BFGS failure on an unscaled target with a
//! standalone Gaussian NLL (loc Identity, scale Exp), same solver setup as
//! `calculate_start_values_default`.
use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::linesearch::{BacktrackingLineSearch, MoreThuenteLineSearch};
use argmin::solver::quasinewton::LBFGS;

struct P {
    y: Vec<f64>,
    scale_cost: f64,
}
impl P {
    fn nll(&self, p: &[f64]) -> f64 {
        let mu = p[0];
        let s = p[1].exp() + 1e-6;
        let t: f64 = self
            .y
            .iter()
            .map(|&y| {
                let z = (y - mu) / s;
                0.5 * z * z + s.ln() + 0.9189385332046727
            })
            .sum();
        if t.is_finite() {
            t * self.scale_cost
        } else {
            f64::MAX
        }
    }
}
impl CostFunction for P {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, p: &Vec<f64>) -> Result<f64, Error> {
        Ok(self.nll(p))
    }
}
impl Gradient for P {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        let eps = 1e-5;
        let mut g = vec![0.0; 2];
        for i in 0..2 {
            let mut a = p.clone();
            let mut b = p.clone();
            a[i] += eps;
            b[i] -= eps;
            g[i] = (self.nll(&a) - self.nll(&b)) / (2.0 * eps);
            if !g[i].is_finite() {
                g[i] = 0.0;
            }
        }
        Ok(g)
    }
}

fn run(label: &str, y: Vec<f64>, scale_cost: f64, init: Vec<f64>) {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let sd = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    let p = P { y, scale_cost };
    let c0 = p.nll(&init);
    let g0 = p.gradient(&init).unwrap();
    let res = Executor::new(p, LBFGS::new(MoreThuenteLineSearch::new(), 7))
        .configure(|st| st.param(init.clone()).max_iters(600))
        .run();
    match res {
        Ok(r) => println!(
            "{label:<40} y~{mean:.2}±{sd:.2}  cost0 {c0:.3e} |g0| {:.2e}  -> OK iters {} best {:?} cost {:.4e}  (target loc {mean:.3}, log sd {:.3}) term {:?}",
            (g0[0].powi(2) + g0[1].powi(2)).sqrt(),
            r.state().get_iter(),
            r.state()
                .get_best_param()
                .map(|v| v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()),
            r.state().get_best_cost(),
            sd.ln(),
            r.state().get_termination_status()
        ),
        Err(e) => println!(
            "{label:<40} y~{mean:.2}±{sd:.2}  cost0 {c0:.3e} |g0| {:.2e}  -> ERR {}",
            (g0[0].powi(2) + g0[1].powi(2)).sqrt(),
            e.to_string().lines().next().unwrap_or("")
        ),
    }
}

fn run_bt(label: &str, y: Vec<f64>, scale_cost: f64, init: Vec<f64>) {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let sd = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    let p = P { y, scale_cost };
    let ls = BacktrackingLineSearch::new(ArmijoCondition::new(1e-4).unwrap())
        .rho(0.5)
        .unwrap();
    let res = Executor::new(p, LBFGS::new(ls, 7))
        .configure(|st| st.param(init.clone()).max_iters(600))
        .run();
    match res {
        Ok(r) => println!(
            "[backtracking] {label:<30} -> iters {} best {:?} cost {:.4e} (target {mean:.3}, {:.3}) term {:?}",
            r.state().get_iter(),
            r.state()
                .get_best_param()
                .map(|v| v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()),
            r.state().get_best_cost(),
            sd.ln(),
            r.state().get_termination_status()
        ),
        Err(e) => println!(
            "[backtracking] {label:<30} -> ERR {}",
            e.to_string().lines().next().unwrap_or("")
        ),
    }
}

/// Damped diagonal-Newton pre-phase (FD gradient + FD diagonal second
/// difference per coordinate, Armijo backtracking on the full step), then
/// L-BFGS + MoreThuente from the resulting point.
fn run_dn(label: &str, y: Vec<f64>, init: Vec<f64>) {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let sd = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    let p = P { y, scale_cost: 1.0 };
    let mut x = init.clone();
    let mut c = p.nll(&x);
    let eps = 1e-4;
    let mut pre_iters = 0;
    for _ in 0..30 {
        let g = p.gradient(&x).unwrap();
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < 1e-6 {
            break;
        }
        // diagonal second differences
        let mut d = vec![0.0; x.len()];
        for i in 0..x.len() {
            let mut a = x.clone();
            let mut b = x.clone();
            a[i] += eps;
            b[i] -= eps;
            let h = (p.nll(&a) - 2.0 * c + p.nll(&b)) / (eps * eps);
            let h = if h.is_finite() && h > 0.0 {
                h
            } else {
                h.abs().max(1e-8)
            };
            d[i] = -g[i] / (h + 1e-8);
        }
        // Armijo backtracking on the Newton step
        let slope: f64 = g.iter().zip(&d).map(|(g, d)| g * d).sum();
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..40 {
            let xn: Vec<f64> = x.iter().zip(&d).map(|(x, d)| x + alpha * d).collect();
            let cn = p.nll(&xn);
            if cn.is_finite() && cn <= c + 1e-4 * alpha * slope {
                x = xn;
                c = cn;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        pre_iters += 1;
        if !accepted {
            break;
        }
    }
    let pre_x = x.clone();
    let res = Executor::new(p, LBFGS::new(MoreThuenteLineSearch::new(), 7))
        .configure(|st| st.param(x).max_iters(600))
        .run();
    match res {
        Ok(r) => println!(
            "[diag-newton] {label:<30} pre {pre_iters} iters -> {:?} cost {c:.4e}; LBFGS iters {} best {:?} cost {:.4e} (target {mean:.3}, {:.3})",
            pre_x.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>(),
            r.state().get_iter(),
            r.state()
                .get_best_param()
                .map(|v| v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()),
            r.state().get_best_cost(),
            sd.ln()
        ),
        Err(e) => println!(
            "[diag-newton] {label:<30} -> ERR {}",
            e.to_string().lines().next().unwrap_or("")
        ),
    }
}

fn main() {
    let dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("research/gradientlss/data");
    let load = |file: &str, col: &str| -> Vec<f64> {
        let text = std::fs::read_to_string(dir.join(file)).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines
            .next()
            .unwrap()
            .split(',')
            .map(|s| s.trim_matches('"'))
            .collect();
        let ci = header.iter().position(|h| *h == col).unwrap();
        lines
            .filter_map(|l| l.split(',').nth(ci).and_then(|v| v.trim().parse().ok()))
            .collect()
    };
    let pp = load("power-plant.csv", "PE");
    let gs = load("gaussian_train_sim.csv", "y");
    let kn = load("kin8nm.csv", "y");
    run(
        "power_plant sum-NLL from 0.5",
        pp.clone(),
        1.0,
        vec![0.5, 0.5],
    );
    run(
        "gaussian_sim sum-NLL from 0.5",
        gs.clone(),
        1.0,
        vec![0.5, 0.5],
    );
    run("kin8nm sum-NLL from 0.5", kn.clone(), 1.0, vec![0.5, 0.5]);
    let n = pp.len() as f64;
    run(
        "power_plant MEAN-NLL from 0.5",
        pp.clone(),
        1.0 / n,
        vec![0.5, 0.5],
    );
    run(
        "gaussian_sim MEAN-NLL from 0.5",
        gs.clone(),
        1.0 / gs.len() as f64,
        vec![0.5, 0.5],
    );
    let (m, s) = {
        let m = pp.iter().sum::<f64>() / n;
        (
            m,
            (pp.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt(),
        )
    };
    run_bt("power_plant sum", pp.clone(), 1.0, vec![0.5, 0.5]);
    run_bt("gaussian_sim sum", gs.clone(), 1.0, vec![0.5, 0.5]);
    run_bt("kin8nm sum", kn.clone(), 1.0, vec![0.5, 0.5]);
    run_bt("power_plant mean", pp.clone(), 1.0 / n, vec![0.5, 0.5]);
    run_dn("power_plant", pp.clone(), vec![0.5, 0.5]);
    run_dn("gaussian_sim", gs.clone(), vec![0.5, 0.5]);
    run_dn("kin8nm", kn.clone(), vec![0.5, 0.5]);
    run_dn(
        "power_plant x1000",
        pp.iter().map(|v| v * 1000.0).collect(),
        vec![0.5, 0.5],
    );
    run_dn(
        "kin8nm /1000",
        kn.iter().map(|v| v / 1000.0).collect(),
        vec![0.5, 0.5],
    );
    run(
        "power_plant sum-NLL from moments",
        pp.clone(),
        1.0,
        vec![m, s.ln()],
    );
    run(
        "power_plant sum-NLL from (mean y, 0.5)",
        pp.clone(),
        1.0,
        vec![m, 0.5],
    );
}
