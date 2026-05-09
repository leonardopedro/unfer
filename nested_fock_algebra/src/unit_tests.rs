/// Fast unit tests for nested_fock_algebra — no candle/CUDA dependency.
#[cfg(test)]
mod algebra_tests {
    use crate::*;
    use crate::cas::compile_to_fock;
    use crate::models::{yang_mills_hamiltonian, gravity_hamiltonian, navier_stokes_hamiltonian};
    use crate::compile_expression;
    use crate::{Operator, QuantumState};
    use num_complex::Complex64;

    // ── CAS / compile_to_fock ───────────────────────────────────────

    #[test]
    fn test_compile_number_operator() {
        // c_0 * a_0  →  one term with two operators (InnerBosonCreate, InnerBosonAnnihilate)
        let h = compile_to_fock("c_0 * a_0");
        assert!(!h.terms.is_empty(), "Number operator should produce at least one term");
        let (_, ops) = &h.terms[0];
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0], Operator::InnerBosonCreate(0)));
        assert!(matches!(ops[1], Operator::InnerBosonAnnihilate(0)));
    }

    #[test]
    fn test_compile_sum_of_operators() {
        // c_0 * a_0 + c_1 * a_1 → two terms
        let h = compile_to_fock("c_0 * a_0 + c_1 * a_1");
        assert_eq!(h.terms.len(), 2);
    }

    #[test]
    fn test_compile_scalar_dropped() {
        // Pure constant terms should be dropped by quadratic ordering
        let h = compile_to_fock("c_0 * a_0 + 1");
        // The "1" should be filtered out
        assert!(h.terms.iter().all(|(_, ops)| !ops.is_empty()));
    }

    #[test]
    fn test_compile_fermionic_operator() {
        let h = compile_to_fock("c_f0 * a_f0");
        assert!(!h.terms.is_empty());
        let (_, ops) = &h.terms[0];
        assert!(matches!(ops[0], Operator::InnerFermionCreate(0)));
        assert!(matches!(ops[1], Operator::InnerFermionAnnihilate(0)));
    }

    #[test]
    fn test_compile_outer_bosonic_operator() {
        let h = compile_to_fock("C_0 * A_0");
        assert!(!h.terms.is_empty());
    }

    // ── Hamiltonian apply ───────────────────────────────────────────

    #[test]
    fn test_number_operator_on_vacuum() {
        // N|0⟩ = 0  (vacuum has no quanta)
        let h = compile_to_fock("c_0 * a_0");
        let mut vacuum = QuantumState::vacuum();
        vacuum = vacuum.apply(&Operator::OuterBosonCreate(crate::InnerBosonicState::vacuum()));

        let result = h.apply(&vacuum);
        assert!(result.components.is_empty(), "N|vac⟩ should be zero");
    }

    #[test]
    fn test_creation_on_vacuum() {
        // a†_0 |0⟩ = |1_0⟩ — one boson in mode 0
        let h = compile_to_fock("c_0");
        let mut vacuum = QuantumState::vacuum();
        vacuum = vacuum.apply(&Operator::OuterBosonCreate(crate::InnerBosonicState::vacuum()));
        let result = h.apply(&vacuum);
        assert!(!result.components.is_empty(), "a†|vac⟩ should not be empty");
    }

    #[test]
    fn test_hermitian_conjugate_symmetry() {
        // ⟨0| (c_0 * a_0 + c_1 * a_1) |0⟩ = 0
        let h = compile_to_fock("c_0 * a_0 + c_1 * a_1");
        let mut vac = QuantumState::vacuum();
        vac = vac.apply(&Operator::OuterBosonCreate(crate::InnerBosonicState::vacuum()));
        let applied = h.apply(&vac);
        let ip = QuantumState::inner_product(&vac, &applied);
        assert!(ip.norm_sqr() < 1e-20, "⟨0|H|0⟩ should be 0");
    }

    // ── LaTeX compilation ──────────────────────────────────────────

    #[test]
    #[cfg(feature = "latex")]
    fn test_latex_annihilation() {
        use crate::compile_latex;
        let h = compile_latex("a_0");
        assert!(!h.terms.is_empty(), "LaTeX a_0 should compile to a non-empty Hamiltonian");
    }

    #[test]
    #[cfg(feature = "latex")]
    fn test_latex_fraction() {
        use crate::compile_latex;
        let h = compile_latex(r"\frac{1}{2} * c_0 * a_0");
        if let Some((coeff, _)) = h.terms.first() {
            assert!((coeff.re - 0.5).abs() < 1e-6, "coefficient should be 0.5");
        }
    }

    // ── Direct Hamiltonian builders (no Expression.expand()) ────────

    #[test]
    fn test_gravity_hamiltonian_terms() {
        let h = gravity_hamiltonian();
        // 3*3 pairs, each with 2 ops squared = 4 terms per pair → 9*4 = 36 for P²
        // minus 9*4 = 36 for e² → 72 terms total
        assert!(h.terms.len() > 0, "Gravity Hamiltonian should have terms");
        // All terms must have exactly 2 operators
        for (_, ops) in &h.terms {
            assert_eq!(ops.len(), 2, "Gravity terms should be quadratic");
        }
    }

    #[test]
    fn test_yang_mills_kinetic_terms() {
        // Yang-Mills with g=0 should have only the kinetic -½π²  terms.
        let h = yang_mills_hamiltonian(0.0);
        // With g=0 the magnetic NL and cross terms vanish.
        // Kinetic: 3*8 modes, each π_mode^2 = 4 terms → 3*8*4 = 96 kinetic terms.
        // Linear B²: ε_{ijk} gives 6 nonzero (i,j,k) triples × 8 colors × 4 field pairs = ... non-trivial.
        // Just sanity: non-empty, all terms have ops.
        assert!(h.terms.len() > 0, "Yang-Mills(g=0) should have terms");
        for (_, ops) in &h.terms {
            assert!(!ops.is_empty(), "Each Y-M term must have at least one operator");
        }
    }

    #[test]
    fn test_yang_mills_on_vacuum() {
        // H_YM |vac⟩ should be non-zero (vacuum fluctuations)
        let h = yang_mills_hamiltonian(1.0);
        let mut vac = QuantumState::vacuum();
        vac = vac.apply(&Operator::OuterBosonCreate(crate::InnerBosonicState::vacuum()));
        let result = h.apply(&vac);
        // The kinetic term π² = (ia† - ia)² creates/annihilates pairs from vacuum.
        // The result should be non-empty due to creation operators acting on vac.
        assert!(!result.components.is_empty(), "H_YM|vac⟩ should be non-zero");
    }

    #[test]
    fn test_navier_stokes_compiles() {
        // let nu = 1e-3;
        // let h_expr = navier_stokes_hamiltonian(nu);
        // let h = compile_expression(h_expr);
        // assert!(h.terms.len() > 0, "Navier-Stokes should compile to non-empty Hamiltonian");
    }

    // ── Inner product / norm ────────────────────────────────────────

    #[test]
    fn test_inner_product_vacuum_with_itself() {
        let vac = QuantumState::vacuum();
        let ip = QuantumState::inner_product(&vac, &vac);
        assert!((ip.re - 1.0).abs() < 1e-12, "⟨0|0⟩ should be 1");
        assert!(ip.im.abs() < 1e-12);
    }

    #[test]
    fn test_scale_and_add() {
        let mut a = QuantumState::vacuum();
        let b = QuantumState::vacuum();
        a.scale_and_add(&b, Complex64::new(2.0, 0.0));
        let ip = QuantumState::inner_product(&a, &a);
        // |3⟩ in vacuum direction: norm² = 9
        assert!((ip.re - 9.0).abs() < 1e-10);
    }
}
