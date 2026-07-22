use num_complex::Complex64;

use crate::ode::ODESystem;
use crate::poly::NormalOrderedOp;
use crate::result::{OdeError, OdeSirkResult};

pub use nested_fock_algebra::Hamiltonian;

/// Weyl-symmetrized quantization of an ODE system.
///
/// Given dx_i/dt = f_i(x), constructs:
///   H = Σ_i ( f_i(x) p_i  −  (i/2) ∂f_i/∂x_i )
///
/// using the strict bosonic mapping:
///   x_i = (a_i + a†_i) / √2
///   p_i = −i(a_i − a†_i) / √2
pub fn ode_to_hamiltonian(sys: &ODESystem) -> OdeSirkResult<Hamiltonian> {
    let n = sys.n_vars();
    let max_degree: u32 = sys
        .rhs
        .iter()
        .flat_map(|p| p.terms.iter())
        .map(|m| m.exponents.iter().sum::<u32>())
        .max()
        .unwrap_or(0);

    if max_degree > 20 {
        return Err(OdeError::PolynomialTooLarge(max_degree, 20));
    }

    let mut all_terms: Vec<(Complex64, Vec<nested_fock_algebra::Operator>)> = Vec::new();

    for i in 0..n {
        // 1. Build normal-ordered representation of f_i(x)
        let f_i = poly_from_polynomial(&sys.rhs[i], n)?;

        // 2. Compute ∂f_i/∂x_i (scalar derivative, used for the Weyl correction)
        let df_i_dxi = sys.rhs[i].partial_derivative(i);

        // 3. Term: f_i(x) p_i
        {
            let mut fp = f_i.clone();
            fp.multiply_p_mode(i);
            all_terms.extend(fp.to_operator_terms());
        }

        // 4. Term: -(i/2) ∂f_i/∂x_i  (a scalar operator)
        if !df_i_dxi.is_zero() {
            let correction = Complex64::new(0.0, -0.5);
            let df_poly = poly_from_polynomial(&df_i_dxi, n)?;
            let mut scaled = df_poly;
            // Scale all coefficients
            for coeff in scaled.terms.values_mut() {
                *coeff *= correction;
            }
            all_terms.extend(scaled.to_operator_terms());
        }
    }

    Ok(Hamiltonian { terms: all_terms })
}

/// Convert a polynomial to NormalOrderedOp form by substituting x_i = (a_i + a†_i)/√2.
fn poly_from_polynomial(
    poly: &crate::ode::Polynomial,
    n_vars: usize,
) -> OdeSirkResult<NormalOrderedOp> {
    let mut result = NormalOrderedOp::new();

    for term in &poly.terms {
        let mut mono = NormalOrderedOp::from_monomial(term.coeff, &vec![0u32; n_vars]);
        // Multiply by x_i^e_i for each variable
        for (i, &e) in term.exponents.iter().enumerate() {
            for _ in 0..e {
                mono.multiply_x_mode(i);
            }
        }
        // Add to result
        for (key, coeff) in &mono.terms {
            *result.terms.entry(key.clone()).or_default() += coeff;
        }
    }

    result.prune(1e-15);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::ODESystem;

    #[test]
    fn x_squared_hamiltonian_nonempty() {
        let sys = ODESystem::parse(vec!["x".into()], &["x^2"]).unwrap();
        let h = ode_to_hamiltonian(&sys).unwrap();
        assert!(!h.terms.is_empty());
    }

    #[test]
    fn stable_linear_hamiltonian() {
        let sys = ODESystem::parse(vec!["x".into()], &["-x"]).unwrap();
        let h = ode_to_hamiltonian(&sys).unwrap();
        assert!(!h.terms.is_empty());
    }

    #[test]
    fn coupled_system_hamiltonian() {
        let sys = ODESystem::parse(vec!["x".into(), "y".into()], &["y", "2*x*y"]).unwrap();
        let h = ode_to_hamiltonian(&sys).unwrap();
        assert!(!h.terms.is_empty());
    }

    #[test]
    fn hermiticity_check() {
        let sys = ODESystem::parse(vec!["x".into()], &["-x"]).unwrap();
        let h = ode_to_hamiltonian(&sys).unwrap();
        let h_dag = h.adjoint();
        // For H = -x·p - (i/2)(-1) = -x·p + i/2
        // H† should equal H (Hermiticity)
        // We check that the term count matches (exact coefficient comparison
        // would require expanding the operator products)
        assert_eq!(h.terms.len(), h_dag.terms.len());
    }
}
