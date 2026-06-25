//! Proper BRST projection onto the physical (gauge-invariant) subspace.
//!
//! The original solver enforced the gauge constraint by *subtracting* `Q w`
//! from `w`, which is not a projection (not idempotent, not self-adjoint, and
//! leaves a gauge residue). Here we implement the true orthogonal projector onto
//! `ker Q`:
//!
//! ```text
//!   P w = w − Q† z,   where  (Q Q†) z = Q w
//! ```
//!
//! `Q Q†` is Hermitian positive-semidefinite, so we solve for `z` matrix-free
//! with the conjugate-gradient method, never materializing a dense operator.

use nested_fock_algebra::{Hamiltonian, QuantumState};
use num_complex::Complex64;

use crate::linalg::SirkError;

fn norm(s: &QuantumState) -> f64 {
    QuantumState::inner_product(s, s).re.max(0.0).sqrt()
}

/// Project `w` onto the physical subspace `ker Q` along `im Q†`.
///
/// `q` is the BRST charge `Q`. Returns [`SirkError::BrstNotConverged`] if the
/// inner CG solve does not reach `tol` within `max_iter` iterations.
pub fn project_physical(
    w: &QuantumState,
    q: &Hamiltonian,
    tol: f64,
    max_iter: usize,
) -> Result<QuantumState, SirkError> {
    let q_dag = q.adjoint();

    // The normal operator A = Q Q† (Hermitian PSD), applied matrix-free.
    let apply_a = |x: &QuantumState| -> QuantumState { q.apply(&q_dag.apply(x)) };

    // Right-hand side b = Q w.
    let b = q.apply(w);
    if norm(&b) < tol {
        // w already (numerically) physical: nothing to project out.
        return Ok(w.clone());
    }

    // Conjugate gradient solve of A z = b, starting from z = 0.
    let mut z = QuantumState::zero();
    let mut r = b.clone(); // r = b - A z = b
    let mut p = r.clone();
    let mut rs_old = QuantumState::inner_product(&r, &r).re;

    let mut converged = false;
    let mut residual = rs_old.sqrt();
    for _ in 0..max_iter {
        let ap = apply_a(&p);
        let p_ap = QuantumState::inner_product(&p, &ap).re;
        if p_ap.abs() < 1e-30 {
            // Breakdown: p lies in ker A; remaining residual is unreachable.
            break;
        }
        let alpha = rs_old / p_ap;
        z.scale_and_add(&p, Complex64::new(alpha, 0.0));
        r.scale_and_add(&ap, Complex64::new(-alpha, 0.0));

        let rs_new = QuantumState::inner_product(&r, &r).re;
        residual = rs_new.sqrt();
        if residual < tol {
            converged = true;
            break;
        }
        let beta = rs_new / rs_old;
        let mut new_p = r.clone();
        new_p.scale_and_add(&p, Complex64::new(beta, 0.0));
        p = new_p;
        rs_old = rs_new;
    }

    if !converged {
        return Err(SirkError::BrstNotConverged { residual });
    }

    // P w = w − Q† z.
    let mut projected = w.clone();
    let correction = q_dag.apply(&z);
    projected.scale_and_add(&correction, Complex64::new(-1.0, 0.0));
    Ok(projected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nested_fock_algebra::{InnerFermionicState, Operator, QuantumState};

    /// Build the ghost-lowering BRST charge Q = C_φ (outer fermion annihilation)
    /// plus the two basis states |0> (no ghost) and |1> (one ghost universe).
    fn ghost_setup() -> (Hamiltonian, QuantumState, QuantumState) {
        let phi = InnerFermionicState::vacuum();
        let q = Hamiltonian {
            terms: vec![(
                Complex64::new(1.0, 0.0),
                vec![Operator::OuterFermionAnnihilate(phi.clone())],
            )],
        };
        let s0 = QuantumState::vacuum(); // |0>: no fermionic universe
        let s1 = QuantumState::vacuum().apply(&Operator::OuterFermionCreate(phi)); // |1>
        (q, s0, s1)
    }

    fn combine(a: &QuantumState, ca: Complex64, b: &QuantumState, cb: Complex64) -> QuantumState {
        let mut s = QuantumState::zero();
        s.scale_and_add(a, ca);
        s.scale_and_add(b, cb);
        s
    }

    #[test]
    fn projection_kills_gauge_part() {
        let (q, s0, s1) = ghost_setup();
        let w = combine(&s0, Complex64::new(0.6, 0.1), &s1, Complex64::new(0.0, 0.8));
        let pw = project_physical(&w, &q, 1e-12, 100).unwrap();
        // Q(Pw) must vanish.
        assert!(norm(&q.apply(&pw)) < 1e-8, "Q(Pw) not annihilated");
        // Pw should equal the physical part 0.6+0.1i times |0>.
        let phys = combine(&s0, Complex64::new(0.6, 0.1), &s1, Complex64::new(0.0, 0.0));
        let mut diff = pw.clone();
        diff.scale_and_add(&phys, Complex64::new(-1.0, 0.0));
        assert!(norm(&diff) < 1e-8, "Pw != physical part");
    }

    #[test]
    fn projection_is_idempotent() {
        let (q, s0, s1) = ghost_setup();
        let w = combine(
            &s0,
            Complex64::new(0.3, -0.2),
            &s1,
            Complex64::new(0.7, 0.4),
        );
        let pw = project_physical(&w, &q, 1e-12, 100).unwrap();
        let ppw = project_physical(&pw, &q, 1e-12, 100).unwrap();
        let mut diff = ppw.clone();
        diff.scale_and_add(&pw, Complex64::new(-1.0, 0.0));
        assert!(norm(&diff) < 1e-10, "P is not idempotent");
    }

    #[test]
    fn projection_is_self_adjoint() {
        let (q, s0, s1) = ghost_setup();
        let v = combine(&s0, Complex64::new(0.5, 0.0), &s1, Complex64::new(0.2, 0.3));
        let w = combine(
            &s0,
            Complex64::new(0.1, -0.4),
            &s1,
            Complex64::new(0.9, 0.0),
        );
        let pv = project_physical(&v, &q, 1e-12, 100).unwrap();
        let pw = project_physical(&w, &q, 1e-12, 100).unwrap();
        // <v, Pw> == <Pv, w>
        let lhs = QuantumState::inner_product(&v, &pw);
        let rhs = QuantumState::inner_product(&pv, &w);
        assert!(
            (lhs - rhs).norm() < 1e-9,
            "P is not self-adjoint: {lhs} vs {rhs}"
        );
    }
}
