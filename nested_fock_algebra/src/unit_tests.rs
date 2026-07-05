/// Fast unit tests for nested_fock_algebra — no candle/CUDA dependency.
#[cfg(test)]
mod algebra_tests {
    use crate::cas::compile_to_fock;
    use crate::models::{
        QFM_DEFAULT_QUANTIZATION_SCALE, bose_hubbard_chain, gravity_hamiltonian,
        mehler_channel_overlap, navier_stokes_hamiltonian, point_to_inner_state, qfm_hamiltonian,
        qfm_hamiltonian_localized, qfm_hamiltonian_mehler_projector,
        qfm_hamiltonian_mehler_projector_localized, yang_mills_hamiltonian, yang_mills_lattice,
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
    fn test_yang_mills_lattice_l4_term_count() {
        // 4×4 periodic lattice, 1 color.
        //   Modes (link variables): 2 dirs × 16 sites × 1 color = 32 links.
        //   Electric: 32 quadratic (arity-2) terms — one number operator per link.
        //   Magnetic: 16 plaquettes × 2⁴ = 256 quartic (arity-4) sub-terms.
        //   Total: 288 terms; all coefficients must be real (no imaginary drift).
        let h = yang_mills_lattice(4, 1.0, 1);
        let electric = h.terms.iter().filter(|(_, ops)| ops.len() == 2).count();
        let magnetic = h.terms.iter().filter(|(_, ops)| ops.len() == 4).count();
        assert_eq!(electric, 32, "4×4 lattice, 1 color → 32 electric terms");
        assert_eq!(magnetic, 256, "16 plaquettes × 16 quartic sub-terms each");
        assert_eq!(h.terms.len(), 288, "total 288 terms");
        assert!(
            h.terms.iter().all(|(c, _)| c.im.abs() < 1e-15),
            "lattice gauge coefficients are real"
        );
        // Scaling: two colors doubles everything uniformly.
        let h2c = yang_mills_lattice(4, 1.0, 2);
        assert_eq!(h2c.terms.len(), 2 * h.terms.len());
    }

    #[test]
    fn test_qfm_hamiltonian() {
        // QFM generator: H = |0><0| + Σ_j α_j n_j   (see QFM.tex).
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
    fn test_qfm_hamiltonian_no_cross_channel_leakage_with_two_channels_excited() {
        // Regression test for a real bug: `qfm_hamiltonian`'s number-operator
        // term used to be built from `InnerBosonCreate`/`InnerBosonAnnihilate`
        // (operators that act on an *already-existing* universe's own inner
        // mode occupation), instead of the *outer* ladder operators that
        // actually define `|x_j> = B_j^dagger|0>` (QFM.tex §framework). On a
        // state with a single active data channel the two choices happen to
        // agree (confirmed by `test_qfm_hamiltonian` above), which is why the
        // bug went unnoticed — but on a state with *two or more* channels
        // simultaneously excited, the inner-operator version de-excites one
        // universe down to the inner vacuum and re-excites *another* universe
        // into a spurious basis state carrying both channels' inner modes at
        // once, leaking amplitude into a state that encodes no real data
        // point and breaking the zero-data-loss disjointness
        // (`QFM.tex` eq. (disjoint)) the whole encoding relies on.
        let alphas = [1.5, 2.1];
        let h = qfm_hamiltonian(&alphas);

        // |x_0, x_1>: two outer universes, one in inner mode 0, one in mode 1.
        let mut inner0 = crate::InnerBosonicState::vacuum();
        inner0.modes.insert(0, 1);
        let mut inner1 = crate::InnerBosonicState::vacuum();
        inner1.modes.insert(1, 1);
        let state = QuantumState::vacuum()
            .apply(&Operator::OuterBosonCreate(inner0))
            .apply(&Operator::OuterBosonCreate(inner1));

        let h_state = h.apply(&state);

        // H|x_0,x_1> = (α_0+α_1)|x_0,x_1>, an eigenstate with no leakage into
        // any other basis state.
        assert_eq!(
            h_state.len(),
            1,
            "H|x_0,x_1> must have exactly one component, not leak into a \
             spurious cross-channel basis state; got {:?}",
            h_state.components.keys().collect::<Vec<_>>()
        );
        let amp = h_state.components.get(&state.components.keys().next().unwrap().clone())
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0));
        let expected = alphas[0] + alphas[1];
        assert!(
            (amp.re - expected).abs() < 1e-12 && amp.im.abs() < 1e-12,
            "H|x_0,x_1> = (α_0+α_1)|x_0,x_1>; got amplitude {amp}"
        );
    }

    // ── Localized (D-coordinate) QFM encoding ───────────────────────
    // `QFM.tex`, "The data-channel wave-function on the hypersphere:
    // finitely many localized coordinates, the rest uniform": a data point
    // x ∈ R^D localizes exactly D of the (infinitely many) hyperspherical
    // coordinates, the rest staying at the uniform circle measure.
    // `point_to_inner_state`/`qfm_hamiltonian_localized` are the direct
    // computational realization: D occupied inner modes per point (one per
    // real coordinate), everything else left at zero occupation.

    #[test]
    fn point_to_inner_state_distinguishes_different_points() {
        let a = point_to_inner_state(&[1.0, 2.0, 3.0], QFM_DEFAULT_QUANTIZATION_SCALE);
        let b = point_to_inner_state(&[1.0, 2.0, 3.5], QFM_DEFAULT_QUANTIZATION_SCALE);
        assert_ne!(a, b, "points differing in one coordinate must differ");
        assert_eq!(a.modes.len(), 3, "three nonzero coordinates -> three occupied modes");
    }

    #[test]
    fn point_to_inner_state_distinguishes_sign() {
        // A naive `abs()`-based quantization would collide +v and -v onto
        // the same occupation number, silently merging two distinct points
        // into one non-orthogonal Fock state. The zigzag encoding must not
        // do that.
        let pos = point_to_inner_state(&[1.5], QFM_DEFAULT_QUANTIZATION_SCALE);
        let neg = point_to_inner_state(&[-1.5], QFM_DEFAULT_QUANTIZATION_SCALE);
        assert_ne!(pos, neg, "+1.5 and -1.5 must map to different inner states");
    }

    #[test]
    fn point_to_inner_state_zero_coordinate_leaves_mode_unoccupied() {
        // A coordinate that quantizes to exactly zero carries no
        // information (matches the vacuum in that mode) and so must not be
        // inserted into the mode map at all.
        let state = point_to_inner_state(&[0.0, 2.0], QFM_DEFAULT_QUANTIZATION_SCALE);
        assert!(!state.modes.contains_key(&0), "zero coordinate must stay unoccupied");
        assert!(state.modes.contains_key(&1));
    }

    #[test]
    fn test_qfm_hamiltonian_localized_eigenstates() {
        let points = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 2.0]];
        let alphas = [1.5, 2.1, 0.8];
        let h = qfm_hamiltonian_localized(&points, &alphas, QFM_DEFAULT_QUANTIZATION_SCALE);

        let vac = QuantumState::vacuum();
        let h_vac = h.apply(&vac);
        let eig0 =
            QuantumState::inner_product(&vac, &h_vac) / QuantumState::inner_product(&vac, &vac);
        assert!((eig0.re - 1.0).abs() < 1e-12 && eig0.im.abs() < 1e-12, "H|0> = |0>");

        for (point, &alpha) in points.iter().zip(alphas.iter()) {
            let inner = point_to_inner_state(point, QFM_DEFAULT_QUANTIZATION_SCALE);
            let xj = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner));
            let h_xj = h.apply(&xj);
            let eig =
                QuantumState::inner_product(&xj, &h_xj) / QuantumState::inner_product(&xj, &xj);
            assert!(
                (eig.re - alpha).abs() < 1e-12 && eig.im.abs() < 1e-12,
                "H|x_j> = α_j|x_j> for point {point:?}; got {eig}"
            );
        }
    }

    #[test]
    fn test_qfm_hamiltonian_localized_no_cross_channel_leakage() {
        // Same regression as the index-based encoding above, but for the
        // localized (D-mode-per-point) encoding: two simultaneously-excited
        // data channels must not leak into a spurious basis state.
        let points = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let alphas = [1.5, 2.1];
        let h = qfm_hamiltonian_localized(&points, &alphas, QFM_DEFAULT_QUANTIZATION_SCALE);

        let inner0 = point_to_inner_state(&points[0], QFM_DEFAULT_QUANTIZATION_SCALE);
        let inner1 = point_to_inner_state(&points[1], QFM_DEFAULT_QUANTIZATION_SCALE);
        let state = QuantumState::vacuum()
            .apply(&Operator::OuterBosonCreate(inner0))
            .apply(&Operator::OuterBosonCreate(inner1));

        let h_state = h.apply(&state);
        assert_eq!(
            h_state.len(),
            1,
            "must have exactly one component, no leakage; got {:?}",
            h_state.components.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_qfm_hamiltonian_mehler_projector_localized_couples_vacuum_to_data_channels() {
        // Exact off-diagonal generator with the literal data-channel encoding:
        // H = |0̃><0̃| with |0̃> = c₀|vac> + Σ ε_j|x_j>, |x_j> the localized
        // channel. ⟨x_j|H|vac⟩ = c₀ε_j and ⟨x_i|H|x_j⟩ = ε_iε_j — coupling
        // with NO explicit coupling terms and no truncation.
        let points = vec![vec![1.0, 0.5], vec![-2.0, 3.0]];
        let eps = [0.3, 0.4];
        let c0 = (1.0f64 - 0.09 - 0.16).sqrt();
        let h =
            qfm_hamiltonian_mehler_projector_localized(&points, &eps, QFM_DEFAULT_QUANTIZATION_SCALE);

        // Single rank-1 term, self-adjoint.
        assert_eq!(h.terms.len(), 1);
        assert_eq!(h.adjoint().terms.len(), 1);

        let vac = QuantumState::vacuum();
        let h_vac = h.apply(&vac);
        let x: Vec<QuantumState> = points
            .iter()
            .map(|p| {
                let inner = point_to_inner_state(p, QFM_DEFAULT_QUANTIZATION_SCALE);
                QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
            })
            .collect();
        for (i, xi) in x.iter().enumerate() {
            let amp = QuantumState::inner_product(xi, &h_vac);
            let want = c0 * eps[i];
            assert!(
                (amp.re - want).abs() < 1e-12 && amp.im.abs() < 1e-12,
                "⟨x_{i}|H|vac⟩ = c₀ε_{i} = {want}; got {amp}"
            );
            for (j, xj) in x.iter().enumerate() {
                let elem = QuantumState::inner_product(xi, &h.apply(xj));
                let want = eps[i] * eps[j];
                assert!(
                    (elem.re - want).abs() < 1e-12 && elem.im.abs() < 1e-12,
                    "⟨x_{i}|H|x_{j}⟩ = ε_iε_j = {want}; got {elem}"
                );
            }
        }

        // Exactly a projector on arbitrary probes: H² = H.
        let mut probe = QuantumState::zero();
        probe.scale_and_add(&vac, Complex64::new(0.5, 0.1));
        probe.scale_and_add(&x[0], Complex64::new(-0.3, 0.2));
        probe.scale_and_add(&x[1], Complex64::new(0.7, 0.0));
        let hp = h.apply(&probe);
        let hhp = h.apply(&hp);
        assert!(
            state_diff_norm(&hhp, &hp) < 1e-12,
            "localized exact generator must satisfy H² = H"
        );
    }

    // ── Exact Mehler-projector QFM generator ────────────────────────
    // `QFM.tex`, "The exact off-diagonal generator is just the vacuum
    // projector": the uniform Mehler vacuum is NOT orthogonal to the data
    // channels (<0|x_j> = ε_j > 0, since a channel localizes only finitely
    // many hyperspherical coordinates), so H = |0><0| is by itself the
    // off-diagonal generator. In the orthonormal OuterState frame the
    // Mehler vacuum is the dressed superposition
    //   |0> = c₀|vac>_F + Σ_j ε_j B†_j|vac>_F,  c₀ = sqrt(1 − Σ ε²).

    /// Diff-norm helper: ‖a − b‖.
    fn state_diff_norm(a: &QuantumState, b: &QuantumState) -> f64 {
        let mut d = a.clone();
        d.scale_and_add(b, Complex64::new(-1.0, 0.0));
        d.norm()
    }

    /// The single-boson channel state |x_j> = B†_j|vac>_F.
    fn channel_state(j: u32) -> QuantumState {
        let mut inner = crate::InnerBosonicState::vacuum();
        inner.modes.insert(j, 1);
        QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
    }

    /// The dressed Mehler vacuum c₀|vac>_F + Σ ε_j|x_j>.
    fn dressed_vacuum(epsilons: &[f64]) -> QuantumState {
        let sum_sq: f64 = epsilons.iter().map(|e| e * e).sum();
        let c0 = (1.0 - sum_sq).sqrt();
        let mut psi = QuantumState::zero();
        psi.scale_and_add(&QuantumState::vacuum(), Complex64::new(c0, 0.0));
        for (j, &e) in epsilons.iter().enumerate() {
            psi.scale_and_add(&channel_state(j as u32), Complex64::new(e, 0.0));
        }
        psi
    }

    #[test]
    fn test_mehler_channel_overlap_formula() {
        let two_pi = 2.0 * std::f64::consts::PI;
        // A full-circle "arc" is no localization: factor 1.
        assert!((mehler_channel_overlap(&[two_pi]) - 1.0).abs() < 1e-12);
        // Per-coordinate factor sqrt(w/2π): two arcs of width π/2 give
        // sqrt(1/4)·sqrt(1/4) = 1/4.
        let e = mehler_channel_overlap(&[two_pi / 4.0, two_pi / 4.0]);
        assert!((e - 0.25).abs() < 1e-12, "got {e}");
        // No localized coordinates at all: the channel IS the vacuum.
        assert!((mehler_channel_overlap(&[]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mehler_projector_matrix_elements() {
        // <vac|H|vac> = c₀², <x_i|H|x_j> = ε_iε_j, <vac|H|x_j> = c₀ε_j —
        // the off-diagonal channel↔channel coupling exists with NO explicit
        // coupling terms beyond the projector itself.
        let eps = [0.3, 0.4];
        let c0 = (1.0f64 - 0.09 - 0.16).sqrt(); // sqrt(0.75)
        let h = qfm_hamiltonian_mehler_projector(&eps);

        let vac = QuantumState::vacuum();
        let x: Vec<QuantumState> = (0..2).map(|j| channel_state(j)).collect();

        let vv = QuantumState::inner_product(&vac, &h.apply(&vac));
        assert!((vv.re - c0 * c0).abs() < 1e-12 && vv.im.abs() < 1e-12);

        for i in 0..2 {
            for j in 0..2 {
                let elem = QuantumState::inner_product(&x[i], &h.apply(&x[j]));
                let want = eps[i] * eps[j];
                assert!(
                    (elem.re - want).abs() < 1e-12 && elem.im.abs() < 1e-12,
                    "<x_{i}|H|x_{j}> = {elem}, want {want}"
                );
            }
            let cross = QuantumState::inner_product(&vac, &h.apply(&x[i]));
            assert!(
                (cross.re - c0 * eps[i]).abs() < 1e-12,
                "<vac|H|x_{i}> = {cross}, want {}",
                c0 * eps[i]
            );
        }
    }

    #[test]
    fn test_mehler_projector_is_exactly_a_projector() {
        // H = |0><0| is rank-1 and idempotent: H(H|s>) = H|s> for any |s>.
        // Idempotence is the signature of the exact generator — any
        // truncation of the projector would fail this.
        let eps = [0.3, 0.4];
        let h = qfm_hamiltonian_mehler_projector(&eps);

        // Probe with several states: frame vacuum, each channel, a mixed
        // superposition, and a two-particle state (annihilated by H).
        let mut probe = QuantumState::zero();
        probe.scale_and_add(&QuantumState::vacuum(), Complex64::new(0.5, 0.1));
        probe.scale_and_add(&channel_state(0), Complex64::new(-0.3, 0.2));
        probe.scale_and_add(&channel_state(1), Complex64::new(0.7, 0.0));
        let two_particle = {
            let mut i0 = crate::InnerBosonicState::vacuum();
            i0.modes.insert(0, 1);
            let mut i1 = crate::InnerBosonicState::vacuum();
            i1.modes.insert(1, 1);
            QuantumState::vacuum()
                .apply(&Operator::OuterBosonCreate(i0))
                .apply(&Operator::OuterBosonCreate(i1))
        };
        for s in [
            QuantumState::vacuum(),
            channel_state(0),
            channel_state(1),
            probe,
            two_particle.clone(),
        ] {
            let hs = h.apply(&s);
            let hhs = h.apply(&hs);
            assert!(
                state_diff_norm(&hhs, &hs) < 1e-12,
                "H must be idempotent (H² = H) on every state"
            );
        }
        // The two-particle state lies outside span{|0>}: H annihilates it.
        assert!(
            h.apply(&two_particle).norm() < 1e-12,
            "H = |0><0| must annihilate states orthogonal to the dressed vacuum"
        );
    }

    #[test]
    fn test_mehler_projector_dressed_vacuum_is_the_unit_eigenvector() {
        // H|0> = |0>: the dressed Mehler vacuum is the (only) eigenvalue-1
        // eigenvector of its own projector.
        let eps = [0.3, 0.4];
        let h = qfm_hamiltonian_mehler_projector(&eps);
        let psi0 = dressed_vacuum(&eps);
        let h_psi0 = h.apply(&psi0);
        assert!(
            state_diff_norm(&h_psi0, &psi0) < 1e-12,
            "H|0> must equal |0> exactly"
        );
    }

    #[test]
    #[should_panic(expected = "Σ ε_j² ≤ 1")]
    fn test_mehler_projector_rejects_overweight_overlaps() {
        // Σ ε² > 1 is physically impossible (the ε² are uniform-measure
        // masses of disjoint packet supports) and must be rejected.
        let _ = qfm_hamiltonian_mehler_projector(&[0.9, 0.9]);
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
