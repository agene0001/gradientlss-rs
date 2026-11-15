//! Curvature and start rules for the two-parameter (loc, scale) families
//! that research cycles 1–13 never measured: Cauchy, Logistic, Gumbel,
//! Laplace and LogNormal (research/gradientlss, proposal 021).
//!
//! Defaults since 2026-11-13: Cauchy and LogNormal use `Damped` curvature with
//! a `Location` start, Logistic and Laplace `Damped` with `LocationScale`.
//! Since 2026-11-14 Gumbel uses `Damped` with `LocationScale` and a start
//! scale 100 × the matched one (proposal 028). `with_hessian_mode(Observed)`
//! and `with_start(Raw)` restore the old behaviour.
//!
//! Why these families need it (measured, cycle 15 baseline): from the raw
//! start a location Hessian that decays with |z| (Logistic ~ e^{−|z|},
//! Gumbel e^{−z}) or turns negative (Cauchy for |z| > 1) leaves the location
//! tree with no usable curvature, so on a target far from 0.5 the model never
//! moves (power_plant: NLL 7.4–12.9 against 2.7 for Gaussian).

use super::base::HessianMode;
use super::student_t::invert_response;
use crate::types::ResponseData;
use crate::utils::ResponseFn;
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

/// Start values used without `initialize` by the location-scale families
/// listed in the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LocationScaleStart {
    /// Raw 0.5 for both parameters: the historical behaviour,
    /// and what a model saved before this option existed loads as.
    #[default]
    Raw,
    /// Location from the labels (the median, shifted for Gumbel; of `ln y`
    /// for LogNormal); scale left at its raw 0.5.
    Location,
    /// Location and scale from the labels by closed-form moment / quantile
    /// matching (no optimiser).
    LocationScale,
}

/// Serde default for the curvature of these families: what they did before
/// the option existed.
pub(crate) fn legacy_hessian_mode() -> HessianMode {
    HessianMode::Observed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LsFamily {
    Cauchy,
    Logistic,
    Gumbel,
    Laplace,
    LogNormal,
}

impl LsFamily {
    /// `(c_loc, c_scale)` with Fisher information `I_μμ = c_loc / s²` and
    /// `I_ss = c_scale / s²` (for LogNormal: μ, σ of `ln y`). Standard
    /// results; checked by Monte Carlo in the tests below.
    pub(crate) fn fisher(self) -> (f64, f64) {
        match self {
            LsFamily::Cauchy => (0.5, 0.5),
            LsFamily::Logistic => (1.0 / 3.0, (std::f64::consts::PI.powi(2) + 3.0) / 9.0),
            // π²/6 + (1 − γ)²
            LsFamily::Gumbel => (1.0, std::f64::consts::PI.powi(2) / 6.0 + (1.0 - EULER_GAMMA).powi(2)),
            LsFamily::Laplace => (1.0, 1.0),
            LsFamily::LogNormal => (1.0, 2.0),
        }
    }
}

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Apply `mode` to the per-row Hessians of a (loc, scale) family, in the
/// predictor space: `Observed` leaves them alone, `Fisher` replaces them with
/// the expected information, `Damped` takes the row-wise maximum (and the
/// Fisher value where the observed one is not finite).
pub(crate) fn apply_hessian_mode(
    mode: HessianMode,
    fam: LsFamily,
    hessians: &mut Array2<f64>,
    p_scale: &ArrayView1<f64>,
    t_scale: &ArrayView1<f64>,
    scale_fn: &ResponseFn,
) {
    if mode == HessianMode::Observed {
        return;
    }
    let (c_loc, c_scale) = fam.fisher();
    // ds/dη of the scale link; I_ηη = I_ss · (ds/dη)²
    let (rd, _) = scale_fn.derivative_batches_from_transformed(p_scale, t_scale);
    for i in 0..hessians.nrows() {
        let s = t_scale[i].max(1e-6);
        let f_loc = c_loc / (s * s);
        let r = rd[i] / s;
        let f_scale = c_scale * r * r;
        for (j, f) in [(0usize, f_loc), (1usize, f_scale)] {
            let h = hessians[[i, j]];
            hessians[[i, j]] = match mode {
                HessianMode::Fisher => f,
                _ if h.is_finite() => h.max(f),
                _ => f,
            };
        }
    }
}

/// Closed-form start values in predictor space, or `None` for the raw start.
pub(crate) fn start_values(
    rule: LocationScaleStart,
    fam: LsFamily,
    target: &ResponseData,
    scale_fn: &ResponseFn,
) -> Option<Array1<f64>> {
    if rule == LocationScaleStart::Raw {
        return None;
    }
    let ResponseData::Univariate(y) = target else { return None };
    let mut v: Vec<f64> = if fam == LsFamily::LogNormal {
        y.iter().copied().filter(|t| t.is_finite() && *t > 0.0).map(f64::ln).collect()
    } else {
        y.iter().copied().filter(|t| t.is_finite()).collect()
    };
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let q = |p: f64| {
        let pos = p * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
    };
    let med = q(0.5);
    let mean = v.iter().sum::<f64>() / n as f64;
    let sd = (v.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let full = rule == LocationScaleStart::LocationScale;
    // scale at the raw start, for the location-only Gumbel shift
    let s0 = scale_fn.apply_scalar(0.5);
    let (loc, scale): (f64, Option<f64>) = match fam {
        LsFamily::Logistic => (med, full.then(|| sd * 3f64.sqrt() / std::f64::consts::PI)),
        LsFamily::Cauchy => (med, full.then(|| 0.5 * (q(0.75) - q(0.25)))),
        LsFamily::Laplace => (med, full.then(|| v.iter().map(|t| (t - med).abs()).sum::<f64>() / n as f64)),
        LsFamily::Gumbel => {
            if full {
                let s = sd * 6f64.sqrt() / std::f64::consts::PI;
                (mean - EULER_GAMMA * s, Some(s))
            } else {
                // median = μ − s·ln(ln 2)
                (med + s0 * std::f64::consts::LN_2.ln(), None)
            }
        }
        LsFamily::LogNormal => {
            if full { (mean, Some(sd)) } else { (med, None) }
        }
    };
    let raw_scale = match scale {
        Some(s) if s.is_finite() && s > 0.0 => invert_response(scale_fn, s),
        _ => 0.5,
    };
    Some(Array1::from_vec(vec![loc, raw_scale]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Cauchy, Distribution, Gumbel, Laplace, LogNormal, Logistic};
    use ndarray::Array2;

    /// Deterministic uniforms in (0, 1).
    fn uniforms(n: usize) -> Vec<f64> {
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((state >> 11) as f64) + 0.5) / (1u64 << 53) as f64
            })
            .collect()
    }

    fn normal(u1: f64, u2: f64) -> f64 {
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn sample(fam: LsFamily, loc: f64, s: f64, n: usize) -> Array1<f64> {
        let u = uniforms(2 * n);
        Array1::from_iter((0..n).map(|i| {
            let p = u[i];
            match fam {
                LsFamily::Logistic => loc + s * (p / (1.0 - p)).ln(),
                LsFamily::Gumbel => loc - s * (-(p.ln())).ln(),
                LsFamily::Cauchy => loc + s * (std::f64::consts::PI * (p - 0.5)).tan(),
                LsFamily::Laplace => {
                    let c = p - 0.5;
                    loc - s * c.signum() * (1.0 - 2.0 * c.abs()).ln()
                }
                LsFamily::LogNormal => (loc + s * normal(p, u[n + i])).exp(),
            }
        }))
    }

    fn dist(fam: LsFamily, mode: HessianMode, start: LocationScaleStart) -> Box<dyn Distribution> {
        match fam {
            LsFamily::Cauchy => Box::new(Cauchy::default().with_hessian_mode(mode).with_start(start)),
            LsFamily::Logistic => Box::new(Logistic::default().with_hessian_mode(mode).with_start(start)),
            LsFamily::Gumbel => {
                Box::new(Gumbel::default().with_hessian_mode(mode).with_start(start).with_start_scale_factor(1.0))
            }
            LsFamily::Laplace => Box::new(Laplace::default().with_hessian_mode(mode).with_start(start)),
            LsFamily::LogNormal => Box::new(LogNormal::default().with_hessian_mode(mode).with_start(start)),
        }
    }

    const ALL: [LsFamily; 5] =
        [LsFamily::Cauchy, LsFamily::Logistic, LsFamily::Gumbel, LsFamily::Laplace, LsFamily::LogNormal];

    fn hessians(d: &dyn Distribution, y: &Array1<f64>, loc: f64, s: f64) -> Array2<f64> {
        let n = y.len();
        let mut raw = Array2::zeros((n, 2));
        raw.column_mut(0).fill(loc);
        raw.column_mut(1).fill(s.ln());
        let mut tr = raw.clone();
        tr.column_mut(1).fill(s);
        let yv = y.view();
        d.analytical_gradients(&raw.view(), &tr.view(), &ResponseData::Univariate(&yv)).expect("analytic").1
    }

    /// The mean observed Hessian at the true parameters equals the Fisher
    /// information the constants claim (Exp link, so I_ηη = c_scale).
    #[test]
    fn fisher_constants_match_the_mean_observed_information() {
        let (loc, s, n) = (0.3, 1.7, 400_000);
        for fam in ALL {
            let y = sample(fam, loc, s, n);
            let obs = hessians(dist(fam, HessianMode::Observed, LocationScaleStart::Raw).as_ref(), &y, loc, s);
            let fis = hessians(dist(fam, HessianMode::Fisher, LocationScaleStart::Raw).as_ref(), &y, loc, s);
            for j in 0..2 {
                let (o, f) = (obs.column(j).mean().unwrap(), fis.column(j).mean().unwrap());
                // Laplace's observed location curvature is 0 a.e.; the family already hands back 1/b².
                assert!((o - f).abs() <= 0.02 * f.abs(), "{fam:?} param {j}: observed {o} vs Fisher {f}");
                assert!(fis.column(j).iter().all(|h| *h > 0.0 && h.is_finite()));
            }
        }
    }

    /// Damped is the row-wise maximum, hence positive and never below either.
    #[test]
    fn damped_is_the_rowwise_maximum() {
        let (loc, s, n) = (-2.0, 0.6, 5_000);
        for fam in ALL {
            let y = sample(fam, loc, s, n);
            let obs = hessians(dist(fam, HessianMode::Observed, LocationScaleStart::Raw).as_ref(), &y, loc, s);
            let fis = hessians(dist(fam, HessianMode::Fisher, LocationScaleStart::Raw).as_ref(), &y, loc, s);
            let dam = hessians(dist(fam, HessianMode::Damped, LocationScaleStart::Raw).as_ref(), &y, loc, s);
            for i in 0..n {
                for j in 0..2 {
                    let want = if obs[[i, j]].is_finite() { obs[[i, j]].max(fis[[i, j]]) } else { fis[[i, j]] };
                    assert_eq!(dam[[i, j]].to_bits(), want.to_bits(), "{fam:?} row {i} param {j}");
                    assert!(dam[[i, j]] > 0.0);
                }
            }
        }
    }

    /// Defaults (2026-11-13): Damped + Location for Cauchy and LogNormal, Damped +
    /// LocationScale for Logistic and Laplace; Gumbel (2026-11-14) Damped +
    /// LocationScale with a start scale 100 × the matched one. A model
    /// saved before the fields existed loads as Observed / Raw.
    #[test]
    fn defaults_and_legacy_loading() {
        let y = sample(LsFamily::Logistic, 450.0, 17.0, 2_000);
        let yv = y.view();
        let t = ResponseData::Univariate(&yv);
        assert_eq!((Cauchy::default().hessian_mode(), Cauchy::default().start()), (HessianMode::Damped, LocationScaleStart::Location));
        assert_eq!((LogNormal::default().hessian_mode(), LogNormal::default().start()), (HessianMode::Damped, LocationScaleStart::Location));
        assert_eq!((Logistic::default().hessian_mode(), Logistic::default().start()), (HessianMode::Damped, LocationScaleStart::LocationScale));
        assert_eq!((Laplace::default().hessian_mode(), Laplace::default().start()), (HessianMode::Damped, LocationScaleStart::LocationScale));
        assert_eq!((Gumbel::default().hessian_mode(), Gumbel::default().start()), (HessianMode::Damped, LocationScaleStart::LocationScale));
        // the default starts sit at the labels, not at 0.5
        for d in [&Cauchy::default() as &dyn Distribution, &Logistic::default(), &Laplace::default()] {
            let sv = d.default_start_values(&t).expect("data start");
            assert!((sv[0] - 450.0).abs() < 5.0, "{} start {sv:?}", d.name());
        }
        let sv = LogNormal::default().default_start_values(&t).expect("data start");
        assert!((sv[0] - 450f64.ln()).abs() < 0.05, "LogNormal start {sv:?}");
        // Gumbel (2026-11-14): location from the labels, scale 100 × the matched one
        let (g1, g100) = (
            Gumbel::default().with_start_scale_factor(1.0).default_start_values(&t).expect("data start"),
            Gumbel::default().default_start_values(&t).expect("data start"),
        );
        assert!((g100[0] - g1[0]).abs() < 1e-9 && (g100[0] - 450.0).abs() < 40.0, "Gumbel start {g100:?}");
        assert!((g100[1] - g1[1] - 100f64.ln()).abs() < 1e-6);
        assert!(Gumbel::default().with_start(LocationScaleStart::Raw).default_start_values(&t).is_none());
        let mut v = serde_json::to_value(Gumbel::default()).unwrap();
        for k in ["hessian_mode", "start", "start_scale_factor"] {
            v.as_object_mut().unwrap().remove(k);
        }
        let old: Gumbel = serde_json::from_value(v).unwrap();
        assert_eq!((old.hessian_mode(), old.start(), old.start_scale_factor()), (HessianMode::Observed, LocationScaleStart::Raw, 1.0));
        // opting out restores the raw start
        assert!(Logistic::default().with_start(LocationScaleStart::Raw).default_start_values(&t).is_none());
        // serde: a model saved before the fields existed loads as Observed / Raw
        for name in ["Cauchy", "Logistic", "Laplace", "LogNormal"] {
            let mut v = match name {
                "Cauchy" => serde_json::to_value(Cauchy::default()).unwrap(),
                "Logistic" => serde_json::to_value(Logistic::default()).unwrap(),
                "Laplace" => serde_json::to_value(Laplace::default()).unwrap(),
                _ => serde_json::to_value(LogNormal::default()).unwrap(),
            };
            v.as_object_mut().unwrap().remove("hessian_mode");
            v.as_object_mut().unwrap().remove("start");
            let (m, st) = match name {
                "Cauchy" => { let b: Cauchy = serde_json::from_value(v).unwrap(); (b.hessian_mode(), b.start()) }
                "Logistic" => { let b: Logistic = serde_json::from_value(v).unwrap(); (b.hessian_mode(), b.start()) }
                "Laplace" => { let b: Laplace = serde_json::from_value(v).unwrap(); (b.hessian_mode(), b.start()) }
                _ => { let b: LogNormal = serde_json::from_value(v).unwrap(); (b.hessian_mode(), b.start()) }
            };
            assert_eq!((m, st), (HessianMode::Observed, LocationScaleStart::Raw), "{name}");
        }
    }

    /// LogNormal rows outside the support carry a zero gradient and the same token
    /// curvature on the serial (n < 4096) and the parallel path, in every mode.
    #[test]
    fn lognormal_rows_outside_the_support_are_inert_on_both_paths() {
        for n in [100usize, 5_000] {
            for mode in [HessianMode::Observed, HessianMode::Fisher, HessianMode::Damped] {
                let mut y = Array1::from_elem(n, 2.0);
                y[3] = 0.0;
                y[7] = -1.5;
                let d = LogNormal::default().with_hessian_mode(mode);
                let raw = Array2::from_elem((n, 2), 0.1);
                let mut tr = raw.clone();
                tr.column_mut(1).fill(0.1f64.exp());
                let yv = y.view();
                let (g, h) = d.analytical_gradients(&raw.view(), &tr.view(), &ResponseData::Univariate(&yv)).expect("analytic");
                for i in [3usize, 7] {
                    assert_eq!((g[[i, 0]], g[[i, 1]]), (0.0, 0.0), "n {n} {mode:?}");
                    assert_eq!((h[[i, 0]], h[[i, 1]]), (1e-6, 1e-6), "n {n} {mode:?}");
                }
                assert!(h[[0, 0]] > 1e-3 && g[[0, 0]] != 0.0);
            }
        }
    }

    /// The closed-form starts recover the generating parameters and move
    /// with the labels (shift and scale equivariance).
    #[test]
    fn starts_recover_the_parameters_and_follow_the_units() {
        let (loc, s, n) = (450.0, 17.0, 200_000);
        for fam in ALL {
            let (l0, s0) = if fam == LsFamily::LogNormal { (1.2, 0.4) } else { (loc, s) };
            let y = sample(fam, l0, s0, n);
            let yv = y.view();
            let d = dist(fam, HessianMode::Observed, LocationScaleStart::LocationScale);
            let sv = d.default_start_values(&ResponseData::Univariate(&yv)).unwrap();
            assert!((sv[0] - l0).abs() < 0.05 * s0.max(1.0), "{fam:?} loc {} vs {l0}", sv[0]);
            if fam != LsFamily::Cauchy {
                assert!((sv[1].exp() - s0).abs() < 0.05 * s0, "{fam:?} scale {} vs {s0}", sv[1].exp());
            } else {
                // half the IQR of a Cauchy is its scale
                assert!((sv[1].exp() - s0).abs() < 0.05 * s0, "Cauchy scale {} vs {s0}", sv[1].exp());
            }
            let d = dist(fam, HessianMode::Observed, LocationScaleStart::Location);
            let sv = d.default_start_values(&ResponseData::Univariate(&yv)).unwrap();
            assert_eq!(sv[1], 0.5);
            if fam != LsFamily::LogNormal && fam != LsFamily::Gumbel {
                // shift equivariance of the location-only start
                let y2 = y.mapv(|t| t + 1000.0);
                let y2v = y2.view();
                let sv2 = d.default_start_values(&ResponseData::Univariate(&y2v)).unwrap();
                assert!((sv2[0] - sv[0] - 1000.0).abs() < 1e-9);
            }
        }
    }
}
