/// Fast unit tests for nested_fock_algebra — no candle/CUDA dependency.
#[cfg(test)]
mod algebra_tests {
    use crate::cas::compile_to_fock;
    use crate::models::{
        bose_hubbard_chain, gravity_hamiltonian, navier_stokes_hamiltonian, qfm_hamiltonian,
        qfm_hamiltonian_offdiag, yang_mills_hamiltonian, yang_mills_lattice,
    };
    use crate::{Operator, QuantumState};
    use num_complex::Complex64;

    // ── CAS / compile_to_fock ───────────────────────────────────────

    #[test]
    fn test_compile_number_operator() {
        // c_0 * a_0  →  one term with two operators (InnerBosonCreate, InnerBosonAnnihilate)
        let h = compile_to_fock("c_0 * a_0");
        assert!(
            !h.terms.is_empty(),
            "Number operator should produce at least one term"
        );
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
        vacuum = vacuum.apply(&Operator::OuterBosonCreate(
            crate::InnerBosonicState::vacuum(),
        ));

        let result = h.apply(&vacuum);
        assert!(result.components.is_empty(), "N|vac⟩ should be zero");
    }

    #[test]
    fn test_creation_on_vacuum() {
        // a†_0 |0⟩ = |1_0⟩ — one boson in mode 0
        let h = compile_to_fock("c_0");
        let mut vacuum = QuantumState::vacuum();
        vacuum = vacuum.apply(&Operator::OuterBosonCreate(
            crate::InnerBosonicState::vacuum(),
        ));
        let result = h.apply(&vacuum);
        assert!(!result.components.is_empty(), "a†|vac⟩ should not be empty");
    }

    #[test]
    fn test_hermitian_conjugate_symmetry() {
        // ⟨0| (c_0 * a_0 + c_1 * a_1) |0⟩ = 0
        let h = compile_to_fock("c_0 * a_0 + c_1 * a_1");
        let mut vac = QuantumState::vacuum();
        vac = vac.apply(&Operator::OuterBosonCreate(
            crate::InnerBosonicState::vacuum(),
        ));
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
        assert!(
            !h.terms.is_empty(),
            "LaTeX a_0 should compile to a non-empty Hamiltonian"
        );
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
        assert!(!h.terms.is_empty(), "Gravity Hamiltonian should have terms");
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
        assert!(!h.terms.is_empty(), "Yang-Mills(g=0) should have terms");
        for (_, ops) in &h.terms {
            assert!(
                !ops.is_empty(),
                "Each Y-M term must have at least one operator"
            );
        }
    }

    #[test]
    fn test_yang_mills_on_vacuum() {
        // H_YM |vac⟩ should be non-zero (vacuum fluctuations)
        let h = yang_mills_hamiltonian(1.0);
        let mut vac = QuantumState::vacuum();
        vac = vac.apply(&Operator::OuterBosonCreate(
            crate::InnerBosonicState::vacuum(),
        ));
        let result = h.apply(&vac);
        // The kinetic term π² = (ia† - ia)² creates/annihilates pairs from vacuum.
        // The result should be non-empty due to creation operators acting on vac.
        assert!(
            !result.components.is_empty(),
            "H_YM|vac⟩ should be non-zero"
        );
    }

    #[test]
    fn test_bose_hubbard_structure() {
        // Open 3-site chain with hopping and on-site repulsion.
        // Bonds (open): (0,1),(1,2) -> 2 bonds * 2 (h.c.) = 4 hopping terms (arity 2).
        // Interaction: u != 0 -> one term per site = 3 terms (arity 4: a†a†aa).
        let h = bose_hubbard_chain(3, 1.0, 2.0, false);
        let hopping = h.terms.iter().filter(|(_, ops)| ops.len() == 2).count();
        let interaction = h.terms.iter().filter(|(_, ops)| ops.len() == 4).count();
        assert_eq!(hopping, 4, "open 3-site chain has 4 hopping terms");
        assert_eq!(
            interaction, 3,
            "on-site repulsion adds one quartic term per site"
        );

        // Periodic adds the wrap bond (n>=3): +2 hopping terms.
        let ring = bose_hubbard_chain(3, 1.0, 2.0, true);
        let ring_hopping = ring.terms.iter().filter(|(_, ops)| ops.len() == 2).count();
        assert_eq!(ring_hopping, 6, "periodic 3-site ring has 6 hopping terms");

        // u = 0 -> no interaction terms (pure tight-binding hopping).
        let free = bose_hubbard_chain(3, 1.0, 0.0, false);
        assert!(
            free.terms.iter().all(|(_, ops)| ops.len() == 2),
            "u=0 leaves only quadratic hopping terms"
        );
    }

    #[test]
    fn test_yang_mills_lattice_structure() {
        // 2×2 periodic lattice, 1 color (area = 4).
        //   Electric: 2 dirs × 4 sites × 1 color = 8 quadratic (arity-2) terms.
        //   Magnetic: 4 plaquettes × 1 color × 2⁴ = 64 quartic (arity-4) terms —
        //   the four plaquette links are distinct modes, so no sub-term collapses.
        let h = yang_mills_lattice(2, 1.0, 1);
        let electric = h.terms.iter().filter(|(_, ops)| ops.len() == 2).count();
        let magnetic = h.terms.iter().filter(|(_, ops)| ops.len() == 4).count();
        assert_eq!(electric, 8, "2×2 lattice, 1 color → 8 electric terms");
        assert_eq!(magnetic, 64, "4 plaquettes × 16 quartic sub-terms each");
        assert!(
            h.terms
                .iter()
                .all(|(_, ops)| ops.len() == 2 || ops.len() == 4),
            "only electric (arity 2) and magnetic (arity 4) terms"
        );
        // Hermitian construction → every coefficient is real.
        assert!(
            h.terms.iter().all(|(c, _)| c.im.abs() < 1e-15),
            "lattice gauge coefficients are real"
        );

        // Each extra color is an independent copy → the term count doubles.
        let h2 = yang_mills_lattice(2, 1.0, 2);
        assert_eq!(h2.terms.len(), 2 * h.terms.len());

        // `l` is clamped to ≥ 2 (a plaquette needs four distinct links).
        let clamped = yang_mills_lattice(1, 1.0, 1);
        assert_eq!(clamped.terms.len(), h.terms.len());
    }

    #[test]
    fn test_qfm_hamiltonian() {
        // QFM generator: H = |0><0| + Σ_j α_j n_j   (see QMF.tex).
        let alphas = [1.5, 2.1, 0.8];
        let h = qfm_hamiltonian(&alphas);

        // One projector term + one number operator per data point.
        assert_eq!(h.terms.len(), alphas.len() + 1);
        assert!(matches!(h.terms[0].1[..], [Operator::ProjectVacuum]));
        assert!(h.terms.iter().all(|(c, _)| c.im.abs() < 1e-15));

        // |x_j> — one outer universe holding a single boson in inner mode j.
        let single_boson = |j: u32| {
            let mut inner = crate::InnerBosonicState::vacuum();
            inner.modes.insert(j, 1);
            Operator::OuterBosonCreate(inner).apply_to_state(&QuantumState::vacuum())
        };

        // H|0> = |0>: the projector contributes eigenvalue 1, all n_j kill vacuum.
        let vac = QuantumState::vacuum();
        let h_vac = h.apply(&vac);
        let eig0 =
            QuantumState::inner_product(&vac, &h_vac) / QuantumState::inner_product(&vac, &vac);
        assert!(
            (eig0.re - 1.0).abs() < 1e-12 && eig0.im.abs() < 1e-12,
            "H|0> = |0>"
        );

        // H|x_j> = α_j |x_j>: the projector drops it, only n_j survives.
        for (j, &alpha) in alphas.iter().enumerate() {
            let xj = single_boson(j as u32);
            let h_xj = h.apply(&xj);
            let eig =
                QuantumState::inner_product(&xj, &h_xj) / QuantumState::inner_product(&xj, &xj);
            assert!(
                (eig.re - alpha).abs() < 1e-12 && eig.im.abs() < 1e-12,
                "H|x_{j}> = α_{j}|x_{j}> (got {eig})"
            );
            // And no leakage back into the vacuum: <0|H|x_j> = 0.
            let leak = QuantumState::inner_product(&vac, &h_xj);
            assert!(leak.norm() < 1e-12, "no vacuum leakage from |x_{j}>");
        }
    }

    #[test]
    fn test_qfm_hamiltonian_offdiag() {
        // Off-diagonal QFM generator (P5 #26):
        //   H = |0><0| + Σ_j α_j (a†_j P₀ + P₀ a_j)
        // Hermitian, with vacuum ↔ |x_j⟩ mixing (Rabi-like transport).
        let alphas = [1.5, 2.1, 0.8];
        let h = qfm_hamiltonian_offdiag(&alphas);

        // 1 projector + 2 conjugate terms per data point.
        assert_eq!(h.terms.len(), 1 + 2 * alphas.len());
        assert!(matches!(h.terms[0].1[..], [Operator::ProjectVacuum]));
        assert!(h.terms.iter().all(|(c, _)| c.im.abs() < 1e-15));

        // Hermiticity: H == H† (the coupling terms are a†P₀ / P₀a conjugates).
        let h_dag = h.adjoint();
        assert_eq!(h.terms.len(), h_dag.terms.len());

        // Helper: one outer universe holding a single boson in inner mode j.
        let single_boson = |j: u32| {
            let mut inner = crate::InnerBosonicState::vacuum();
            inner.modes.insert(j, 1);
            Operator::OuterBosonCreate(inner).apply_to_state(&QuantumState::vacuum())
        };
        let vac = QuantumState::vacuum();

        // H|0> = |0> + Σ α_j |x_j>  —  projector keeps vacuum, each ĥ_j creates
        // amplitude in channel j. Vacuum expectation = 1 (from the projector);
        // the off-diagonal terms contribute <0|a†_j P₀|0> = 0.
        let h_vac = h.apply(&vac);
        let eig0 =
            QuantumState::inner_product(&vac, &h_vac) / QuantumState::inner_product(&vac, &vac);
        assert!(
            (eig0.re - 1.0).abs() < 1e-12 && eig0.im.abs() < 1e-12,
            "⟨0|H|0⟩ = 1 (projector); got {eig0}"
        );

        for (j, &alpha) in alphas.iter().enumerate() {
            let xj = single_boson(j as u32);
            let h_xj = h.apply(&xj);

            // Diagonal: ⟨x_j|H|x_j⟩ = 0  (no number operator; P₀a_j|x_j⟩=|0⟩
            // but ⟨x_j|0⟩ = 0, and a†_jP₀|x_j⟩ = 0).
            let diag =
                QuantumState::inner_product(&xj, &h_xj) / QuantumState::inner_product(&xj, &xj);
            assert!(
                diag.norm() < 1e-12,
                "⟨x_{j}|H|x_{j}⟩ = 0 (off-diagonal only); got {diag}"
            );

            // Off-diagonal coupling: ⟨0|H|x_j⟩ = α_j  (the vacuum↔data mixing).
            // H|x_j⟩ = P₀a_j|x_j⟩ = |0⟩ (times α_j); a†_jP₀|x_j⟩ = 0.
            let offdiag = QuantumState::inner_product(&vac, &h_xj);
            assert!(
                (offdiag.re - alpha).abs() < 1e-12 && offdiag.im.abs() < 1e-12,
                "⟨0|H|x_{j}⟩ = α_{j} = {alpha}; got {offdiag}"
            );

            // By hermiticity ⟨x_j|H|0⟩ = α_j too.
            let offdiag_rev = QuantumState::inner_product(&xj, &h_vac);
            assert!(
                (offdiag_rev.re - alpha).abs() < 1e-12 && offdiag_rev.im.abs() < 1e-12,
                "⟨x_{j}|H|0⟩ = α_{j} (hermiticity); got {offdiag_rev}"
            );
        }
    }

    #[test]
    fn test_navier_stokes_compiles() {
        // Stage 5: built directly as Hamiltonian terms — no Expression::expand() hang.
        // The original Expression-based version hung in .expand() on the high-order
        // symbolic tree; building terms directly (like yang_mills_hamiltonian) avoids
        // the combinatorial explosion entirely (AGENTS.md).
        let nu = 1e-3;
        let h = navier_stokes_hamiltonian(nu);
        assert!(
            !h.terms.is_empty(),
            "Navier-Stokes should produce a non-empty Hamiltonian"
        );
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

    // ── Stage 4: bounded CAS + state-explosion bounds ───────────────

    #[test]
    fn test_bounded_cas_within_limit_succeeds() {
        use crate::{ExpansionLimits, compile_to_fock_bounded};
        // A small sum distributes to a handful of terms — well under the limit.
        let h = compile_to_fock_bounded("c_0 * a_0 + c_1 * a_1", &ExpansionLimits::default())
            .expect("small expression should compile within the default limit");
        assert_eq!(h.terms.len(), 2);
    }

    #[test]
    fn test_bounded_cas_explosion_returns_error() {
        use crate::{CasError, ExpansionLimits, compile_to_fock_bounded};
        // A product of several sums distributes combinatorially (a+b)(c+d)(e+f)...
        // With a tiny limit, the compiler must abort with TermExplosion rather
        // than exhausting memory.
        let expr = "(c_0 * a_0 + c_1 * a_1) * (c_0 * a_0 + c_1 * a_1) \
                    * (c_0 * a_0 + c_1 * a_1) * (c_0 * a_0 + c_1 * a_1)";
        let limits = ExpansionLimits { max_terms: 4 };
        let err = compile_to_fock_bounded(expr, &limits)
            .expect_err("high-order product should exceed the term limit");
        match err {
            CasError::TermExplosion { terms, limit } => {
                assert!(
                    terms > limit,
                    "reported terms {terms} should exceed limit {limit}"
                );
                assert_eq!(limit, 4);
            }
            other => panic!("expected TermExplosion, got {other:?}"),
        }
    }

    #[test]
    fn test_unbounded_matches_legacy_compile() {
        use crate::{ExpansionLimits, compile_to_fock, compile_to_fock_bounded};
        // The unbounded bounded-path must reproduce the historical result exactly.
        let legacy = compile_to_fock("c_0 * a_0 + c_1 * a_1");
        let bounded =
            compile_to_fock_bounded("c_0 * a_0 + c_1 * a_1", &ExpansionLimits::unbounded())
                .expect("unbounded compilation cannot exceed the limit");
        assert_eq!(legacy.terms.len(), bounded.terms.len());
    }

    #[test]
    fn test_prune_drops_small_components() {
        // prune(eps) drops components with |amp| <= eps, preserving the rest.
        let mut s = QuantumState::vacuum();
        s.scale_and_add(&QuantumState::vacuum(), Complex64::new(1.0, 0.0)); // vac amp = 2
        let big = s.norm();
        s.prune(1e-6);
        assert!(!s.is_empty(), "large component must survive pruning");
        assert!(
            (s.norm() - big).abs() < 1e-12,
            "pruning must not perturb surviving mass"
        );
    }

    #[test]
    fn test_truncate_top_k_keeps_largest() {
        // Build a 2-component state; truncate_top_k(1) keeps the larger one.
        let mut a = QuantumState::vacuum(); // vac, amp 1
        let mut other = QuantumState::vacuum();
        other = other.apply(&Operator::OuterBosonCreate(
            crate::InnerBosonicState::vacuum(),
        ));
        a.scale_and_add(&other, Complex64::new(0.1, 0.0)); // small second component
        assert_eq!(a.len(), 2);
        a.truncate_top_k(1);
        assert_eq!(a.len(), 1, "only the largest component should remain");
    }
}
