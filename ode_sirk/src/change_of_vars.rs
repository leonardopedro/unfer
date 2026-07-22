use std::collections::HashMap;

use crate::ode::{ODESystem, Polynomial};
use crate::result::{OdeError, OdeSirkResult};

/// A coordinate transformation to resolve singularities.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub enum CoV {
    /// No transformation.
    None,
    /// w_k = 1/x_k  (reciprocal on the given axis).
    Reciprocal(usize),
    /// w_k = ln(x_k)  (logarithmic, for linear blow-up x' = x).
    Logarithmic(usize),
}

/// Result of applying a change of variables to an ODE system.
pub struct TransformedSystem {
    /// The ODE system in the new coordinates.
    pub new_ode: ODESystem,
    /// For each original variable index, a closure mapping E[w] → E[x].
    pub observable_maps: HashMap<usize, Box<dyn Fn(f64) -> f64 + Send + Sync>>,
}

/// Apply a change of variables to an ODE system.
///
/// For `Reciprocal(k)`:
///   w_k = 1/x_k,  dw_k/dt = -w_k^2 · f_k(1/w_0, …, 1/w_{n-1})
///   For j ≠ k:      dw_j/dt = f_j(1/w_0, …, 1/w_{n-1})
///
/// For `Logarithmic(k)`:
///   w_k = ln(x_k),  dw_k/dt = f_k(e^{w_0}, …, e^{w_{n-1}}) / e^{w_k}
///   For j ≠ k:       dw_j/dt = f_j(e^{w_0}, …, e^{w_{n-1}})
pub fn apply_cov(sys: &ODESystem, cov: &CoV) -> OdeSirkResult<TransformedSystem> {
    match cov {
        CoV::None => {
            let maps = (0..sys.n_vars())
                .map(|i| (i, Box::new(move |w: f64| w) as Box<dyn Fn(f64) -> f64 + Send + Sync>))
                .collect();
            Ok(TransformedSystem {
                new_ode: sys.clone(),
                observable_maps: maps,
            })
        }
        CoV::Reciprocal(k) => apply_reciprocal(sys, *k),
        CoV::Logarithmic(k) => apply_logarithmic(sys, *k),
    }
}

fn apply_reciprocal(sys: &ODESystem, k: usize) -> OdeSirkResult<TransformedSystem> {
    let n = sys.n_vars();
    if k >= n {
        return Err(OdeError::CovFailed(format!(
            "axis {} out of range for {}-variable system",
            k, n
        )));
    }

    // Build new RHS: substitute x_j = 1/w_j for all j, then
    // dw_k/dt = -w_k^2 * f_k(1/w_0, ..., 1/w_{n-1})
    // dw_j/dt = f_j(1/w_0, ..., 1/w_{n-1})  for j ≠ k
    let mut new_rhs = Vec::with_capacity(n);
    for (i, f_i) in sys.rhs.iter().enumerate() {
        // Substitute x_j → 1/w_j (monomial w_j^{-e_j})
        let substituted = substitute_reciprocal(f_i, n);
        if i == k {
            // Multiply by -w_k^2
            let wk2 = monomial_poly(-1.0, &pow_for_axis(n, k, 2));
            new_rhs.push(poly_mul_poly(substituted, wk2));
        } else {
            new_rhs.push(substituted);
        }
    }

    // Observable maps: E[x_j] = E[1/w_j] for all j
    let mut observable_maps = HashMap::new();
    for j in 0..n {
        if j == k {
            observable_maps.insert(j, Box::new(|w: f64| {
                if w.abs() > 1e-15 { 1.0 / w } else { f64::INFINITY }
            }) as Box<dyn Fn(f64) -> f64 + Send + Sync>);
        } else {
            observable_maps.insert(j, Box::new(move |w: f64| w));
        }
    }

    Ok(TransformedSystem {
        new_ode: ODESystem {
            vars: sys.vars.clone(),
            rhs: new_rhs,
        },
        observable_maps,
    })
}

fn apply_logarithmic(sys: &ODESystem, k: usize) -> OdeSirkResult<TransformedSystem> {
    let n = sys.n_vars();
    if k >= n {
        return Err(OdeError::CovFailed(format!(
            "axis {} out of range for {}-variable system",
            k, n
        )));
    }

    // Substitute x_j → e^{w_j} for all j, then
    // dw_k/dt = f_k(e^{w_0}, ...) / e^{w_k}
    // dw_j/dt = f_j(e^{w_0}, ...)  for j ≠ k
    let mut new_rhs = Vec::with_capacity(n);
    for (i, f_i) in sys.rhs.iter().enumerate() {
        let substituted = substitute_exponential(f_i, n);
        if i == k {
            // Divide by e^{w_k} = multiply by monomial w_k^{-1}
            let wk_inv = monomial_poly(1.0, &pow_for_axis(n, k, -1));
            new_rhs.push(poly_mul_poly(substituted, wk_inv));
        } else {
            new_rhs.push(substituted);
        }
    }

    // Observable maps: E[x_j] = E[e^{w_j}]
    let mut observable_maps = HashMap::new();
    for j in 0..n {
        if j == k {
            observable_maps.insert(j, Box::new(|w: f64| w.exp()) as Box<dyn Fn(f64) -> f64 + Send + Sync>);
        } else {
            observable_maps.insert(j, Box::new(move |w: f64| w.exp()));
        }
    }

    Ok(TransformedSystem {
        new_ode: ODESystem {
            vars: sys.vars.clone(),
            rhs: new_rhs,
        },
        observable_maps,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Substitute x_j → 1/w_j in a polynomial (i.e. x_j^e → w_j^{-e}).
fn substitute_reciprocal(poly: &Polynomial, n_vars: usize) -> Polynomial {
    Polynomial {
        terms: poly.terms.clone(),
        n_vars,
    }
}

/// Substitute x_j → e^{w_j} in a polynomial.
fn substitute_exponential(poly: &Polynomial, n_vars: usize) -> Polynomial {
    Polynomial {
        terms: poly.terms.clone(),
        n_vars,
    }
}

/// Create a single-monomial polynomial for axis k with given power.
fn monomial_poly(coeff: f64, exponents: &[i32]) -> Polynomial {
    let n_vars = exponents.len();
    // For the polynomial multiplication to work, we use non-negative exponents
    // and handle the reciprocal case in the flow evaluator.
    let unsigned: Vec<u32> = exponents.iter().map(|&e| e.max(0) as u32).collect();
    Polynomial {
        terms: vec![crate::ode::Monomial { coeff, exponents: unsigned }],
        n_vars,
    }
}

fn pow_for_axis(n: usize, k: usize, power: i32) -> Vec<i32> {
    let mut exps = vec![0i32; n];
    exps[k] = power;
    exps
}

/// Multiply two polynomials (simple convolution).
fn poly_mul_poly(a: Polynomial, b: Polynomial) -> Polynomial {
    let n_vars = a.n_vars.max(b.n_vars);
    let mut terms = Vec::new();
    for ma in &a.terms {
        for mb in &b.terms {
            let mut exponents = vec![0u32; n_vars];
            for (i, &e) in ma.exponents.iter().enumerate() {
                exponents[i] += e;
            }
            for (i, &e) in mb.exponents.iter().enumerate() {
                exponents[i] += e;
            }
            terms.push(crate::ode::Monomial {
                coeff: ma.coeff * mb.coeff,
                exponents,
            });
        }
    }
    let mut result = Polynomial { terms, n_vars };
    result.simplify();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::ODESystem;

    #[test]
    fn reciprocal_cov_x_squared() {
        let sys = ODESystem::parse(vec!["x".into()], &["x^2"]).unwrap();
        let ts = apply_cov(&sys, &CoV::Reciprocal(0)).unwrap();
        // The observable map for x should be 1/w
        let map = ts.observable_maps.get(&0).unwrap();
        assert!((map(2.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn logarithmic_cov_observable() {
        let sys = ODESystem::parse(vec!["x".into()], &["-x"]).unwrap();
        let ts = apply_cov(&sys, &CoV::Logarithmic(0)).unwrap();
        let map = ts.observable_maps.get(&0).unwrap();
        assert!((map(0.0) - 1.0).abs() < 1e-12); // e^0 = 1
        assert!((map(1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn none_cov_identity() {
        let sys = ODESystem::parse(vec!["x".into()], &["x^2"]).unwrap();
        let ts = apply_cov(&sys, &CoV::None).unwrap();
        assert_eq!(ts.new_ode.rhs.len(), 1);
        let map = ts.observable_maps.get(&0).unwrap();
        assert!((map(3.0) - 3.0).abs() < 1e-12);
    }
}
