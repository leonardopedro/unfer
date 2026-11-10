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
//!
//! # Conditioning (measured 2026-11-10)
//!
//! CG runs on the normal equations, so its iteration count tracks `κ(Q)²` and
//! its residual is *exactly* the projection's physical residual:
//!
//! ```text
//!   QQ†z − Qw = −Q(w − Q†z) = −Q(Pw),
//! ```
//!
//! i.e. the `tol` the caller passes bounds `‖Q(Pw)‖` directly. A charge that
//! fails to close is **not** singular — `b = Qw ∈ im Q = range(QQ†)` always —
//! it is merely badly conditioned. Measured on the QG probe that used
//! to fail (`Ω = (a†₂ + a₂)c₉`, `qg3d_full_hamiltonian(3)`): the CG–Lanczos
//! tridiagonal at 60 steps has `λ_min = 1.648430476329554e-1`,
//! `λ_max = 4.444237534453834e2` (`κ(QQ†) ≥ 2.696e3`), and the residual decays
//! as CG theory predicts for that κ — `0.390` after 50 iterations, `9.12e-2`
//! after 60. LSQR on `min ‖Q†z − w‖` (same `z`, no squaring of the condition
//! number) reaches `5.27e-2` in the same 60 steps: a constant-factor gain, not
//! a cure. The *promoted* charge `Ω = X·c₉`, `X = Σ x[ie][je]C†(e_i)A(e_j)`,
//! converges on the same problem in **one** iteration, so the budget failure
//! is a property of the charge supplied rather than of the solver: `max_iter`
//! must be sized to the sequence's `κ`, not held fixed at 50.

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
    use nested_fock_algebra::{InnerBosonicState, InnerFermionicState, Operator, QuantumState};

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

    /// Off `ker Q` with a *structured* charge — a second-quantized position
    /// times a ghost, `Q = X·c` with `X = (|0><1| + |1><0|)/√2` on a one-mode
    /// outer boson truncation, not the rank-one `C_φ` of the tests above. This
    /// is the shape of the QG charge; it converges in a couple of CG steps,
    /// which is the point: the budget failure recorded for the QG probe came
    /// from a badly conditioned (non-closing) charge, not from this solver.
    #[test]
    fn projection_converges_off_kernel_for_a_second_quantized_charge() {
        let mut bos = InnerBosonicState::vacuum();
        bos.modes.insert(0, 1);
        let x = Hamiltonian {
            terms: vec![
                (
                    Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                    vec![
                        Operator::OuterBosonCreate(bos.clone()),
                        Operator::OuterBosonAnnihilate(InnerBosonicState::vacuum()),
                    ],
                ),
                (
                    Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                    vec![
                        Operator::OuterBosonCreate(InnerBosonicState::vacuum()),
                        Operator::OuterBosonAnnihilate(bos.clone()),
                    ],
                ),
            ],
        };
        let q = Hamiltonian {
            terms: x
                .terms
                .iter()
                .map(|(c, ops)| {
                    let mut o = ops.clone();
                    o.push(Operator::OuterFermionAnnihilate(
                        InnerFermionicState::vacuum(),
                    ));
                    (*c, o)
                })
                .collect(),
        };

        let physical = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(bos.clone()));
        let ghostly = physical
            .clone()
            .apply(&Operator::OuterFermionCreate(InnerFermionicState::vacuum()));
        let w = combine(
            &physical,
            Complex64::new(0.6, 0.0),
            &ghostly,
            Complex64::new(0.8, 0.0),
        );
        assert!(
            norm(&q.apply(&w)) > 1e-3,
            "the probe must be off ker Q: ‖Qw‖ = {:.3e}",
            norm(&q.apply(&w))
        );

        let pw = project_physical(&w, &q, 1e-12, 100).expect("CG must converge here");
        assert!(norm(&q.apply(&pw)) < 1e-8, "Q(Pw) not annihilated");
        let phys = combine(
            &physical,
            Complex64::new(0.6, 0.0),
            &ghostly,
            Complex64::new(0.0, 0.0),
        );
        let mut diff = pw.clone();
        diff.scale_and_add(&phys, Complex64::new(-1.0, 0.0));
        assert!(
            norm(&diff) < 1e-8,
            "Pw must be the physical component: residual {:.3e}",
            norm(&diff)
        );
        assert!(
            norm(&q.apply(&q.apply(&w))) < 1e-12,
            "the structured charge must be nilpotent: Q²w = {:.3e}",
            norm(&q.apply(&q.apply(&w)))
        );
    }
}
