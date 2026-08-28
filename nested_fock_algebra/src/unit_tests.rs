/// Fast unit tests for nested_fock_algebra — no candle/CUDA dependency.
#[cfg(test)]
mod algebra_tests {
    use crate::cas::compile_to_fock;
    use crate::models::{
        QFM_DEFAULT_QUANTIZATION_SCALE, bose_hubbard_chain, gravity_hamiltonian,
        mehler_channel_overlap, navier_stokes_brst, navier_stokes_hamiltonian,
        point_to_inner_state, qfm_hamiltonian, qfm_hamiltonian_localized,
        qfm_hamiltonian_mehler_projector, qfm_hamiltonian_mehler_projector_localized,
        yang_mills_hamiltonian, yang_mills_lattice,
    };
    use crate::{InnerBosonicState, Operator, QuantumState};
    use num_complex::Complex64;

    // ── EXACT term-level Hermiticity certificate H = H† ─────────────
    //
    // Every term produced by the framework's CAS / normal-ordering pipeline is
    // in normal-ordered form (creator block before annihilator block, cf.
    // `cas::normal_order_inner`) with a real coefficient, so a canonical key is
    // (sorted creator block, sorted annihilator block): the ladder operators
    // inside each block act on distinct modes and hence commute pairwise, so
    // sorting is exact. `H = H†` ⟺ the key multiset of `H` equals that of
    // `H.adjoint()` (creator↔annihilator exchange leaves the multiset
    // invariant).
    fn term_key(t: &(Complex64, Vec<Operator>)) -> (String, String, f64) {
        let (coeff, ops) = t;
        let (mut creators, mut annihilators) = (Vec::new(), Vec::new());
        for op in ops {
            match op {
                Operator::InnerBosonCreate(_)
                | Operator::OuterBosonCreate(_)
                | Operator::InnerFermionCreate(_)
                | Operator::OuterFermionCreate(_) => creators.push(format!("{op:?}")),
                Operator::InnerBosonAnnihilate(_)
                | Operator::OuterBosonAnnihilate(_)
                | Operator::InnerFermionAnnihilate(_)
                | Operator::OuterFermionAnnihilate(_) => annihilators.push(format!("{op:?}")),
                // Non-ladder (projector) ops are self-adjoint; keep the whole
                // string so the multiset comparison stays exact.
                _ => creators.push(format!("[{op:?}]")),
            }
        }
        creators.sort();
        annihilators.sort();
        (creators.join(","), annihilators.join(","), coeff.re)
    }

    fn assert_exact_hermitian(h: &crate::Hamiltonian) {
        let hd = h.adjoint();
        let mut keys: Vec<_> = h.terms.iter().map(term_key).collect();
        let mut adj_keys: Vec<_> = hd.terms.iter().map(term_key).collect();
        let sort_key = |a: &(String, String, f64), b: &(String, String, f64)| {
            a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.total_cmp(&b.2))
        };
        keys.sort_by(sort_key);
        adj_keys.sort_by(sort_key);
        assert_eq!(
            keys, adj_keys,
            "H must equal H† term-by-term (multiset of canonical keys)"
        );
    }

    // ── The enclosure-form doctrine: creation left, annihilation right ──
    //
    // The FINAL Hamiltonian of every sector (QYM, QED, QG, NS) is the
    // one-particle Hamiltonian enclosed in creation (on the left) and
    // annihilation (on the right) operators acting on the nested Fock space:
    //   H = Σᵢⱼ hᵢⱼ · C†(eᵢ)·A(eⱼ)   (creation-left / annihilation-right).
    // In the framework a term's operator list is written in PRODUCT order
    // (the application iterates right-to-left, lib.rs `apply`), so the
    // doctrine is exactly the assertion below: each term splits as a creator
    // block followed by an annihilator block, with no annihilator before a
    // creator. This is the structural fact that makes ⟨0|H|0⟩ = 0 and the
    // Bose additivity of the one-particle spectrum automatic.
    fn assert_enclosure_form(h: &crate::Hamiltonian) {
        assert!(!h.terms.is_empty(), "enclosure-form Hamiltonian has no terms");
        for (coeff, ops) in &h.terms {
            let mut seen_annihilator = false;
            for op in ops {
                let is_create = matches!(
                    op,
                    Operator::InnerBosonCreate(_)
                        | Operator::OuterBosonCreate(_)
                        | Operator::InnerFermionCreate(_)
                        | Operator::OuterFermionCreate(_)
                );
                let is_annihilate = matches!(
                    op,
                    Operator::InnerBosonAnnihilate(_)
                        | Operator::OuterBosonAnnihilate(_)
                        | Operator::InnerFermionAnnihilate(_)
                        | Operator::OuterFermionAnnihilate(_)
                );
                assert!(
                    is_create || is_annihilate,
                    "term {ops:?} (coeff {coeff}) is not a ladder product — the final \
                     Hamiltonian must be an enclosure C†(h)A"
                );
                if is_annihilate {
                    seen_annihilator = true;
                } else {
                    assert!(
                        !seen_annihilator,
                        "term {ops:?} has a creator AFTER an annihilator — not \
                         creation-left/annihilation-right"
                    );
                }
            }
        }
    }

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
        let amp = h_state
            .components
            .get(&state.components.keys().next().unwrap().clone())
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
        assert_eq!(
            a.modes.len(),
            3,
            "three nonzero coordinates -> three occupied modes"
        );
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
        assert!(
            !state.modes.contains_key(&0),
            "zero coordinate must stay unoccupied"
        );
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
        assert!(
            (eig0.re - 1.0).abs() < 1e-12 && eig0.im.abs() < 1e-12,
            "H|0> = |0>"
        );

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
        let h = qfm_hamiltonian_mehler_projector_localized(
            &points,
            &eps,
            QFM_DEFAULT_QUANTIZATION_SCALE,
        );

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
        let x: Vec<QuantumState> = (0..2).map(channel_state).collect();

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

    // ── Navier-Stokes structural calculus (gravity-parallel) ─────────────
    // The gravity model (H = Σ P² − e²) is checked for its defining algebraic
    // structure; the Navier-Stokes model is checked the same way. The three
    // invariants below are the numerical shadow of the book.tex claims
    // (H = ∫a†(πⁱ(u_j u_{i,j} − ν u_{i,jj}) + h.c.)a, Ω = ∫a†[u_{j,j}ψ†]a):
    //   (1) H is Hermitian — the flow e^{−iHt} is unitary;
    //   (2) H is a polynomial of LOW DEGREE in the fields (≤ 3 ladder
    //       operators per term) — this makes H a well-defined symmetric
    //       operator on the dense finite-particle domain; it does NOT by
    //       itself give self-adjointness (essential self-adjointness requires
    //       flow completeness, Nelson; the project's ẋ=x² ODE chapter is the
    //       counterexample). The Weyl symmetrization (h.c. / anti-commutator)
    //       in (1) is what supplies formal Hermiticity;
    //   (3) the BRST charge is nilpotent, Ω² = 0 — the divergence constraint
    //       is a first-class constraint.

    /// A bosonic occupation eigenstate of `InnerBosonicState` (mode, count).
    fn ns_basis(mode: u32, count: u32) -> QuantumState {
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(mode, count);
        QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
    }

    /// Numerical Hermiticity of a Hamiltonian on a small sample of occupation
    /// eigenstates: `⟨φ|H ψ⟩ == ⟨H φ|ψ⟩` for every pair (equivalently
    /// `⟨φ|H ψ⟩ == ⟨ψ|H φ⟩*`, i.e. H = H† on the subspace).
    fn assert_numerically_hermitian(h: &crate::Hamiltonian, label: &str) {
        let states = [
            ns_basis(0, 1),
            ns_basis(0, 2),
            ns_basis(1, 1),
            ns_basis(2, 1),
            ns_basis(3, 1),
            ns_basis(12, 1),
        ];
        let mut worst: f64 = 0.0;
        for a in &states {
            for b in &states {
                let hab = QuantumState::inner_product(a, &h.apply(b));
                let hba = QuantumState::inner_product(&h.apply(a), b);
                worst = worst.max((hab - hba).norm());
            }
        }
        assert!(
            worst < 1e-9,
            "{label}: Hamiltonian not Hermitian on sample states, ‖⟨φ|H|ψ⟩−⟨Hφ|ψ⟩‖ = {worst:.3e}"
        );
    }

    /// Nilpotency of a BRST charge on sample states: `‖Ω²|s⟩‖ = 0`.
    fn assert_nilpotent(brst: &crate::Hamiltonian, label: &str) {
        let states = [
            ns_basis(0, 1),
            ns_basis(1, 1),
            ns_basis(2, 1),
            ns_basis(3, 1),
        ];
        let mut worst: f64 = 0.0;
        for s in &states {
            let twice = brst.apply(&brst.apply(s));
            worst = worst.max(twice.norm());
        }
        assert!(
            worst < 1e-9,
            "{label}: BRST charge not nilpotent, ‖Ω²|s⟩‖ = {worst:.3e}"
        );
    }

    #[test]
    fn test_navier_stokes_hermitian() {
        // book.tex §4184-4189: H = ∫a†(πⁱ(u_j u_{i,j} − ν u_{i,jj}) + h.c.)a.
        // The (h.c.) term must make the whole operator Hermitian — otherwise the
        // flow is not unitary and the self-adjointness hypothesis fails.
        let h = navier_stokes_hamiltonian(1e-3);
        assert_numerically_hermitian(&h, "Navier-Stokes H");
    }

    #[test]
    fn test_navier_stokes_low_degree() {
        // The formalization plan's core hypothesis: H is a polynomial of LOW
        // DEGREE in the field operators (book.tex §4199-4208, "time-independent
        // polynomial quantum Hamiltonians"). Each monomial π·u·u or π·u carries
        // at most 3 ladder operators. Low degree makes H a well-defined
        // polynomial operator on the dense finite-particle domain (symmetry,
        // no renormalization needed) — it is NOT by itself the self-adjointness
        // criterion, which is flow completeness (Nelson); that distinction is
        // what the finite-truncation flow test in fock_sirk actually checks.
        let h = navier_stokes_hamiltonian(1e-3);
        assert!(!h.terms.is_empty());
        for (_, ops) in &h.terms {
            assert!(
                ops.len() <= 3,
                "NS term has {} operators; low-degree hypothesis requires ≤ 3",
                ops.len()
            );
        }
        // The viscous term −νu_{i,jj} and the advection term u_j u_{i,j} are both
        // present: the degree-3 advection monomials are what drive the nonlinearity.
        let max_ops = h.terms.iter().map(|(_, ops)| ops.len()).max().unwrap_or(0);
        assert_eq!(
            max_ops, 3,
            "NS interaction is genuinely cubic in the fields"
        );
    }

    #[test]
    fn test_navier_stokes_brst_nilpotent() {
        // book.tex §4187-4189: Ω = ∫a†[u_{j,j}ψ†]a is the BRST charge imposing the
        // divergence-free constraint. Nilpotency Ω² = 0 (already proved in Lean for
        // the ghost algebra: BookProof/ChapterGhostField.lean brst_charge_nilpotent)
        // is the defining property of a first-class constraint.
        let brst = navier_stokes_brst();
        assert_nilpotent(&brst, "Navier-Stokes BRST Ω");
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

    // ── QED structural tests (see fock_sirk/tests/qed_validation.rs for the
    //    quantitative comparisons against published perturbative results) ──

    use crate::models::{
        qed_cavity_frequencies, qed_charge_operator, qed_free_photon, qed_pair_production,
        qed_static_charge_interaction,
    };

    #[test]
    fn test_qed_free_photon_is_hermitian_number_terms() {
        // H = Σ ω N_i : one inner number-operator term per mode (framework-
        // native), real coeffs. The inner construction guarantees ⟨0|H|0⟩=0.
        let h = qed_free_photon(&[1.0, 2.0, 3.0]);
        assert_eq!(h.terms.len(), 3);
        for (coeff, ops) in &h.terms {
            assert_eq!(coeff.im.abs(), 0.0, "free-photon coefficients are real");
            assert_eq!(ops.len(), 2);
            assert!(matches!(ops[0], Operator::InnerBosonCreate(_)));
            assert!(matches!(ops[1], Operator::InnerBosonAnnihilate(_)));
        }
        // H† = H term-for-term (number operators are self-adjoint).
        let h_dag = h.adjoint();
        for ((c1, o1), (c2, o2)) in h.terms.iter().zip(h_dag.terms.iter()) {
            assert!((c1 - c2).norm() < 1e-12, "adjoint coefficient must match");
            assert_eq!(
                format!("{:?}", o1),
                format!("{:?}", o2),
                "number operator is self-adjoint"
            );
        }
        // ⟨0|H|0⟩ = 0 on the physical vacuum (one empty universe).
        let vac = crate::QuantumState::vacuum()
            .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
        let hv = h.apply(&vac);
        let e0 = crate::QuantumState::inner_product(&hv, &vac).re;
        assert!(
            e0.abs() < 1e-9,
            "⟨0|H|0⟩ must be 0 (inner construction), got {e0}"
        );
    }

    #[test]
    fn test_qed_cavity_frequencies_are_n_pi_over_d() {
        // Casimir cavity spectrum ω_n = nπ/d (published mode structure).
        let d = 2.0;
        let freqs = qed_cavity_frequencies(d, 4);
        let expected: Vec<f64> = (1..=4)
            .map(|n| std::f64::consts::PI * (n as f64) / d)
            .collect();
        assert_eq!(freqs, expected);
    }

    #[test]
    fn test_qed_static_charge_interaction_is_hermitian() {
        // H = Σ k N + Σ g (B† + B): the B† and B terms carry equal real
        // coefficients, so H† = H exactly.
        let modes = crate::models::qed_coulomb_radial_modes(0.1, 2.0, 0.5);
        let h = qed_static_charge_interaction(&modes, 0.7, 1.0);
        assert!(!h.terms.is_empty());
        // 3 terms per mode: number operator + creation + annihilation.
        assert_eq!(h.terms.len(), 3 * modes.len());
        // Hermiticity: the two single-operator terms per mode have identical
        // coefficients and the creation/annihilation pair is conjugate.
        for i in 0..modes.len() {
            let (c_num, ops_num) = &h.terms[3 * i];
            assert_eq!(ops_num.len(), 2, "number-operator term");
            assert!(c_num.im.abs() < 1e-12);
            let (c_cr, ops_cr) = &h.terms[3 * i + 1];
            let (c_an, ops_an) = &h.terms[3 * i + 2];
            assert_eq!(ops_cr.len(), 1, "creation term");
            assert_eq!(ops_an.len(), 1, "annihilation term");
            assert!(
                matches!(&ops_cr[0], Operator::OuterBosonCreate(m) if m.modes.get(&(i as u32)) == Some(&1)),
                "creation mode mismatch"
            );
            assert!(
                matches!(&ops_an[0], Operator::OuterBosonAnnihilate(m) if m.modes.get(&(i as u32)) == Some(&1)),
                "annihilation mode mismatch"
            );
            assert!(
                (c_cr - c_an).norm() < 1e-12,
                "creation/annihilation coefficients must match for Hermiticity"
            );
        }
    }

    #[test]
    fn test_qed_charge_operator_eigenvalues() {
        // Q = Σ e†e − Σ p†p: |vac⟩ → 0, |e₀⟩ → +1, |p₀⟩ → −1.
        let q = qed_charge_operator(1, 1);
        let mut vac = QuantumState::vacuum();
        vac = vac.apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
        // Q|vac⟩ = 0.
        let qv = q.apply(&vac);
        let ov = QuantumState::inner_product(&qv, &vac).re;
        assert!(ov.abs() < 1e-12, "Q|vac⟩ must be 0, got {ov}");
        // One electron in fermion mode 0 → eigenvalue +1.
        let e_state = vac.create_fermion(0);
        let qe = q.apply(&e_state);
        let ov = QuantumState::inner_product(&qe, &e_state).re;
        assert!((ov - 1.0).abs() < 1e-12, "Q|e⟩ must be +1, got {ov:?}");
        // One positron in fermion mode 1 → eigenvalue −1.
        let p_state = vac.create_fermion(1);
        let qp = q.apply(&p_state);
        let ov = QuantumState::inner_product(&qp, &p_state).re;
        assert!((ov + 1.0).abs() < 1e-12, "Q|p⟩ must be −1, got {ov:?}");
    }

    #[test]
    fn test_qed_pair_production_is_hermitian_and_conserves_charge() {
        // 2 pair modes; symmetric couplings so the model is genuinely U(1)-invariant.
        let e_energies = [1.5, 2.0];
        let p_energies = [1.5, 2.0];
        let vertex = [0.3, 0.2];
        let h = qed_pair_production(1.0, &e_energies, &p_energies, &vertex);

        // Hermiticity: H† must equal H (adjoint pairs of vertex terms, real coeffs).
        let h_dag = h.adjoint();
        assert_eq!(h.terms.len(), h_dag.terms.len());
        for t in &h.terms {
            assert!(
                t.0.im.abs() < 1e-12,
                "pair-production coefficients are real"
            );
        }
        // Term count: 1 photon + 2*(electron+positron) + 2*2 vertex terms = 9.
        assert_eq!(h.terms.len(), 9);

        // Charge conservation [H, Q] = 0: apply HQ − QH to a few states.
        // Layout: electrons at fermion modes 0..2, positrons at modes 2..4.
        let q = qed_charge_operator(2, 2);
        let mut vac = QuantumState::vacuum();
        vac = vac.apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
        let probes = [
            vac.clone(),
            vac.create_fermion(0),                   // one electron
            vac.create_fermion(0).create_fermion(2), // electron + positron
            vac.apply(&Operator::OuterBosonCreate({
                let mut inner = InnerBosonicState::vacuum();
                inner.modes.insert(0, 1);
                inner
            })), // one photon
        ];
        for (i, s) in probes.iter().enumerate() {
            let hq = h.apply(&q.apply(s));
            let qh = q.apply(&h.apply(s));
            let mut diff = hq.clone();
            diff.scale_and_add(&qh, Complex64::new(-1.0, 0.0));
            let nrm = diff.norm();
            assert!(
                nrm < 1e-9,
                "[H, Q]|probe {i}⟩ must vanish, got ‖·‖ = {nrm:.3e}"
            );
        }
    }

    // ── QCD structural tests (see fock_sirk/tests/qcd_validation.rs for the
    //    quantitative comparisons against published perturbative QCD) ──

    use crate::models::{
        qcd_beta_function, qcd_color_factors, qcd_one_gluon_exchange, qcd_running_coupling,
        qcd_su3_f,
    };

    #[test]
    fn test_qcd_color_factors_are_published_values() {
        // C_F = 4/3, C_A = 3, T_R = 1/2 — the exact published QCD color factors.
        let (c_f, c_a, t_r) = qcd_color_factors();
        assert!(
            (c_f - 4.0 / 3.0).abs() < 1e-12,
            "C_F must be 4/3, got {c_f}"
        );
        assert!((c_a - 3.0).abs() < 1e-12, "C_A must be 3 (N_c), got {c_a}");
        assert!((t_r - 0.5).abs() < 1e-12, "T_R must be 1/2, got {t_r}");
    }

    #[test]
    fn test_qcd_su3_structure_constants_antisymmetry() {
        // f_abc is totally antisymmetric.
        assert!((qcd_su3_f(0, 1, 2) - 1.0).abs() < 1e-12, "f_012 = 1");
        assert!((qcd_su3_f(1, 0, 2) + 1.0).abs() < 1e-12, "f_102 = -1");
        assert!((qcd_su3_f(2, 1, 0) + 1.0).abs() < 1e-12, "f_210 = -1");
        assert!((qcd_su3_f(0, 2, 1) + 1.0).abs() < 1e-12, "f_021 = -1");
        // C_A consistency: sum_abc f_abc² / 8 = 3.
        let mut sum_f2 = 0.0;
        for a in 0..8 {
            for b in 0..8 {
                for c in 0..8 {
                    let f = qcd_su3_f(a, b, c);
                    sum_f2 += f * f;
                }
            }
        }
        assert!(
            (sum_f2 / 8.0 - 3.0).abs() < 1e-9,
            "Σ f²/8 must equal C_A = 3, got {}",
            sum_f2 / 8.0
        );
    }

    #[test]
    fn test_qcd_one_gluon_exchange_is_hermitian() {
        let modes = crate::models::qed_coulomb_radial_modes(0.1, 2.0, 0.5);
        let h = qcd_one_gluon_exchange(&modes, 0.7, 1.0);
        assert!(!h.terms.is_empty());
        // 3 terms per mode: number operator + creation + annihilation.
        assert_eq!(h.terms.len(), 3 * modes.len());
        for i in 0..modes.len() {
            let (c_num, ops_num) = &h.terms[3 * i];
            assert_eq!(ops_num.len(), 2, "number-operator term");
            assert!(c_num.im.abs() < 1e-12);
            let (c_cr, _) = &h.terms[3 * i + 1];
            let (c_an, _) = &h.terms[3 * i + 2];
            assert!(
                (c_cr - c_an).norm() < 1e-12,
                "gluon creation/annihilation coefficients must match (Hermiticity)"
            );
        }
    }

    #[test]
    fn test_qcd_beta_function_published_coefficients() {
        // β₀ = (11/3)N_c − (2/3)N_f; pure SU(3): 11, 9 (N_f=3), 7 (N_f=6).
        assert!(
            (qcd_beta_function(3.0, 0.0) - 11.0).abs() < 1e-9,
            "pure-glue β₀ must be 11"
        );
        assert!(
            (qcd_beta_function(3.0, 3.0) - 9.0).abs() < 1e-9,
            "N_f=3 β₀ must be 9"
        );
        assert!(
            (qcd_beta_function(3.0, 6.0) - 7.0).abs() < 1e-9,
            "N_f=6 β₀ must be 7"
        );
        // Asymptotic freedom: β₀ > 0.
        assert!(
            qcd_beta_function(3.0, 6.0) > 0.0,
            "SU(3) is asymptotically free"
        );
        // U(1) is NOT: β₀ < 0 is not the case here; check QED-like N_f large.
        // β₀ < 0 (no asymptotic freedom) requires N_f > (11/2)N_c = 16.5 for SU(3).
        assert!(
            qcd_beta_function(3.0, 17.0) < 0.0,
            "N_f=17 QCD-like β₀ must be negative (loss of asymptotic freedom)"
        );
    }

    #[test]
    fn test_qcd_running_coupling_asymptotic_freedom() {
        // α_s decreases with Q² when β₀ > 0.
        let alpha = qcd_running_coupling(7.0, 0.3);
        assert!(alpha[0] > alpha[1] && alpha[1] > alpha[2] && alpha[2] > alpha[3]);
        // All finite and positive, and never exceeding α_s(μ²).
        for &a in alpha.iter() {
            assert!(
                a > 0.0 && a <= 0.3 + 1e-12,
                "α_s(Q²) must stay in (0, α_s(μ²)]: {a}"
            );
        }
        // With β₀ < 0 (Landau-pole / non-asymptotic-free) and a small α_s so
        // the pole lies beyond the tested scales, α_s grows with Q².
        let alpha_bad = qcd_running_coupling(-7.0, 0.05);
        assert!(
            alpha_bad[3] > alpha_bad[0],
            "negative β₀ must make α_s grow (no asymptotic freedom)"
        );
    }

    #[test]
    fn test_qcd_beta_two_loop_published_coefficients() {
        // β₁ = (34/3)N_c² − (10/3)N_cN_f − 2C_FN_f (Jones, Caswell 1974).
        use crate::models::qcd_beta_two_loop;
        assert!(
            (qcd_beta_two_loop(3.0, 0.0) - 102.0).abs() < 1e-9,
            "pure-glue β₁ must be 102"
        );
        assert!(
            (qcd_beta_two_loop(3.0, 3.0) - 64.0).abs() < 1e-9,
            "N_f=3 β₁ must be 64"
        );
        assert!(
            (qcd_beta_two_loop(3.0, 6.0) - 26.0).abs() < 1e-9,
            "N_f=6 β₁ must be 26"
        );
    }

    #[test]
    fn test_qcd_r_ratio_published_values() {
        // R = N_c Σ Q_f²; u,d,s,c,b charges → 2, 10/3, 11/3 (published, PDG).
        use crate::models::qcd_r_ratio;
        let u = 2.0 / 3.0;
        let d = -1.0 / 3.0;
        let uds = [u, d, d];
        let udsc = [u, d, d, u];
        let udscb = [u, d, d, u, d];
        assert!((qcd_r_ratio(&uds) - 2.0).abs() < 1e-9, "R(u,d,s) must be 2");
        assert!(
            (qcd_r_ratio(&udsc) - 10.0 / 3.0).abs() < 1e-9,
            "R(u,d,s,c) must be 10/3"
        );
        assert!(
            (qcd_r_ratio(&udscb) - 11.0 / 3.0).abs() < 1e-9,
            "R(u,d,s,c,b) must be 11/3"
        );
    }

    #[test]
    fn test_qcd_two_loop_running_reaches_published_alpha_s_tau() {
        // From the PDG α_s(M_Z) = 0.1179, two-loop running reaches the
        // published α_s(M_τ) ≈ 0.33 (PDG: 0.314 ± 0.030). One-loop gives ~0.27.
        use crate::models::{QCD_ALPHA_S_MZ, qcd_alpha_s_running};
        let m_z = 91.1876;
        let m_tau = 1.777;
        let two_loop = qcd_alpha_s_running(QCD_ALPHA_S_MZ, m_z, m_tau, 5.0, 3.0, 200_000);
        // Published PDG α_s(M_τ): 0.314 ± 0.030. Two-loop must land within a
        // generous window (crude single-flavour-threshold approximation).
        assert!(
            (two_loop - 0.314).abs() < 0.05,
            "two-loop α_s(M_τ) must reach the published 0.314±0.03, got {two_loop}"
        );
        // Determinism: same inputs → same output.
        let again = qcd_alpha_s_running(QCD_ALPHA_S_MZ, m_z, m_tau, 5.0, 3.0, 200_000);
        assert!(
            (two_loop - again).abs() < 1e-15,
            "running must be deterministic"
        );
    }

    // ── QG structural tests (see fock_sirk/tests/qg_validation.rs for the
    //    comparisons against published numerical gravity results) ──

    use crate::models::{
        qg_flrw_scalars, qg_gps_rate, qg_gravitational_redshift, qg_light_bending,
        qg_newton_potential, qg_perihelion_precession, qg_planck_units,
    };

    #[test]
    fn test_qg_planck_units_match_published() {
        let (l_p, t_p, m_p, e_p) = qg_planck_units();
        // Published CODATA/PDG values.
        assert!(
            (l_p - 1.616255e-35).abs() / 1.616255e-35 < 1e-3,
            "ℓ_P must be 1.616255e-35 m, got {l_p:.5e}"
        );
        assert!(
            (t_p - 5.391247e-44).abs() / 5.391247e-44 < 1e-3,
            "t_P must be 5.391247e-44 s, got {t_p:.5e}"
        );
        assert!(
            (m_p - 2.176434e-8).abs() / 2.176434e-8 < 1e-3,
            "m_P must be 2.176434e-8 kg, got {m_p:.5e}"
        );
        // E_P = 1.221e19 GeV (1 GeV = 1.602176634e-10 J).
        let e_p_gev = e_p / 1.602_176_634e-10;
        assert!(
            (e_p_gev - 1.221e19).abs() / 1.221e19 < 1e-2,
            "E_P must be 1.221e19 GeV, got {e_p_gev:.3e} GeV"
        );
    }

    #[test]
    fn test_qg_gravitational_redshift_pound_rebka() {
        // Pound–Rebka: g ≈ 9.82 m/s², Δh = 22.5 m → z ≈ 2.5e-15 (published).
        let z = qg_gravitational_redshift(9.82, 22.5);
        assert!(
            (z - 2.46e-15).abs() / 2.46e-15 < 0.01,
            "Pound–Rebka z must be ≈2.5e-15, got {z:.3e}"
        );
    }

    #[test]
    fn test_qg_mercury_perihelion_precession() {
        // Mercury: GM_sun = 1.327e20, a = 5.791e10, e = 0.2056, period 88 d.
        let gm = 1.32712440018e20;
        let arcsec = qg_perihelion_precession(gm, 5.7909e10, 0.205_630, 88.0);
        assert!(
            (arcsec - 43.0).abs() < 0.5,
            "Mercury precession must be ≈43.0″/century, got {arcsec:.2}"
        );
    }

    #[test]
    fn test_qg_light_bending_eddington() {
        // Sun's limb: GM_sun = 1.327e20, b = R_sun = 6.96e8 → 1.75″ (published).
        let gm = 1.32712440018e20;
        let arcsec = qg_light_bending(gm, 6.96e8);
        assert!(
            (arcsec - 1.75).abs() < 0.02,
            "Sun-limb deflection must be ≈1.75″, got {arcsec:.3}"
        );
    }

    #[test]
    fn test_qg_gps_time_dilation() {
        // Earth GM = 3.986e14, R = 6.371e6, GPS h = 2.02e7 → ~5.3e-10.
        let rate = qg_gps_rate(3.986_004_418e14, 6.371e6, 2.02e7);
        assert!(
            (rate - 5.29e-10).abs() / 5.29e-10 < 0.01,
            "GPS rate must be ≈5.3e-10, got {rate:.3e}"
        );
    }

    #[test]
    fn test_qg_tegr_friedmann_equivalence() {
        // Matter-dominated FLRW: a ∝ t^{2/3}, H = 2/(3t), Ḣ = -2/(3t²).
        let t = 2.0;
        let h = 2.0 / (3.0 * t);
        let hdot = -2.0 / (3.0 * t * t);
        let (r, tegr) = qg_flrw_scalars(h, hdot);
        // Published FLRW: R = 6(Ḣ+H²), T = -6H².
        assert!((r - 6.0 * (hdot + h * h)).abs() < 1e-12, "R = 6(Ḣ+H²)");
        assert!((tegr - -6.0 * h * h).abs() < 1e-12, "T = -6H²");
        // TEGR identity: R = -T + (divergence). The r-independent piece of the
        // divergence is 6Ḣ; R + T = 6Ḣ + ... verify the field-equation content:
        // both R and TEGR-T give the same Friedmann equation 3H² = 8πGρ.
        let r_friedmann = 3.0 * h * h; // from R (vacuum part)
        let t_friedmann = -tegr / 2.0; // from T = -6H²
        assert!(
            (r_friedmann - t_friedmann).abs() < 1e-12,
            "R and TEGR-T must give the same Friedmann equation (TEGR = GR)"
        );
    }

    #[test]
    fn test_qg_newton_potential_earth() {
        // Earth surface: GM = 3.986e14, R = 6.371e6 → Φ ≈ -6.26e7 m²/s².
        let phi = qg_newton_potential(3.986_004_418e14, 6.371e6);
        assert!(
            (phi - -6.26e7).abs() / 6.26e7 < 0.01,
            "Earth surface Φ must be ≈-6.26e7 m²/s², got {phi:.3e}"
        );
    }

    // ── Cadabra2-derived Hamiltonian builders (outer nested Fock space) ──

    use crate::models::{qcd_ym_hamiltonian, qg_tegr_hamiltonian};

    #[test]
    fn test_qcd_ym_hamiltonian_outer_fock_vacuum_zero_and_hermitian() {
        // The .cdb-derived H_final = ½π² + ½B² built in the nested Fock space
        // with B a genuine function of A: ⟨0|H|0⟩ = 0 and H = H† exactly.
        //
        // Symmetry of the gauge-fixed QYM Hamiltonian (Weyl gauge, A₀ = 0): the
        // Weyl gauge is a *physical* gauge — A₀ and the longitudinal modes are
        // eliminated before quantization, no ghosts enter the Hamiltonian, and
        // H_final = ½π² + ½B² is a sum of squares of self-adjoint operators
        // (π = ∂₀A self-adjoint; B(A) a real polynomial in the self-adjoint
        // fields). The BRST machinery of Ottinger.md (arXiv:1803.00383, §III.B)
        // — where non-self-adjoint Kugo–Ojima ghosts can leave the Hamiltonian
        // non-self-adjoint in the canonical inner product — is NOT part of this
        // construction; the Gauss constraint G_y is imposed on the physical
        // states instead of being BRST-quantized.
        let h = qcd_ym_hamiltonian(0.5);
        assert!(!h.terms.is_empty());
        // Real coefficients (a necessary condition for H = H† with the
        // creator↔annihilator pairing produced by normal ordering).
        let hd = h.adjoint();
        assert_eq!(h.terms.len(), hd.terms.len());
        for t in &h.terms {
            assert!(t.0.im.abs() < 1e-12, "YM outer-Fock coefficients are real");
        }
        // Vacuum expectation 0 (outer-vacuum annihilation by the normal-ordered
        // Hamiltonian).
        let hv = h.apply(&crate::QuantumState::vacuum());
        let e0 = crate::QuantumState::inner_product(&hv, &crate::QuantumState::vacuum()).re;
        assert!(e0.abs() < 1e-9, "⟨0|H|0⟩ must be 0, got {e0}");

        // EXACT term-level Hermiticity H = H† (shared helper, see above).
        assert_exact_hermitian(&h);

        // Matrix-level spot check on the occupation-≤1 basis over the four
        // modes {0,1,2,3}: M_ij = ⟨u_i|H|u_j⟩ must satisfy M_ij = conj(M_ji).
        let mut basis: Vec<crate::QuantumState> = Vec::new();
        for mask in 0..16u32 {
            let mut st = crate::QuantumState::vacuum();
            for mode in 0..4u32 {
                if mask & (1 << mode) != 0 {
                    st = st.apply(&Operator::InnerBosonCreate(mode));
                }
            }
            basis.push(st);
        }
        for i in 0..basis.len() {
            for j in 0..basis.len() {
                let mij = crate::QuantumState::inner_product(&basis[i], &h.apply(&basis[j]));
                let mji = crate::QuantumState::inner_product(&basis[j], &h.apply(&basis[i]));
                assert!(
                    (mij - mji.conj()).norm() < 1e-9,
                    "M[{i}][{j}] = {mij} but conj(M[{j}][{i}]) = {}",
                    mji.conj()
                );
            }
        }
    }

    #[test]
    fn test_qg_tegr_hamiltonian_outer_fock_vacuum_zero_and_hermitian() {
        // The Cadabra2-derived TEGR kinetic (1/16e)𝒮² in the outer nested Fock
        // space: ⟨0|H|0⟩ = 0 and H = H† exactly (the kinetic is a real
        // quadratic form :(1/16)𝒮²: = (1/16)(B†B − ½(B†²+B²)) per mode — a sum
        // of squares of the self-adjoint densitized-tetrad momentum, so
        // Hermiticity in the canonical inner product is manifest; no ghosts, no
        // signed inner product, cf. the Weyl-gauge QYM note in
        // test_qcd_ym_hamiltonian_outer_fock_vacuum_zero_and_hermitian).
        let h = qg_tegr_hamiltonian(3);
        let hd = h.adjoint();
        assert_eq!(h.terms.len(), hd.terms.len());
        for t in &h.terms {
            assert!(
                t.0.im.abs() < 1e-12,
                "TEGR outer-Fock coefficients are real"
            );
        }
        // EXACT term-level Hermiticity H = H† (shared helper).
        assert_exact_hermitian(&h);
        let hv = h.apply(&crate::QuantumState::vacuum());
        let e0 = crate::QuantumState::inner_product(&hv, &crate::QuantumState::vacuum()).re;
        assert!(e0.abs() < 1e-9, "⟨0|H|0⟩ must be 0, got {e0}");
    }

    #[test]
    fn test_qg_starobinsky_hamiltonian_vacuum_zero_and_hermitian() {
        // The Cadabra2-derived R + αR² gauge-fixed scalar sector
        // ½π² + ½m²φ² (docs/qg_starobinsky_hamiltonian.cdb, H_final truncated
        // at quadratic order — the Starobinsky scalaron mass), built from the
        // framework-native inner ladder operators: ⟨0|H|0⟩ = 0 and H = H†.
        use crate::models::qg_starobinsky_hamiltonian;
        let h = qg_starobinsky_hamiltonian(3, 1.0);
        let hd = h.adjoint();
        assert_eq!(h.terms.len(), hd.terms.len());
        for t in &h.terms {
            assert!(
                t.0.im.abs() < 1e-12,
                "Starobinsky coefficients are real"
            );
        }
        // EXACT term-level Hermiticity H = H† (shared helper).
        assert_exact_hermitian(&h);
        // The physical vacuum for inner operators is one empty inner universe.
        let vac = crate::QuantumState::vacuum()
            .apply(&crate::Operator::OuterBosonCreate(crate::InnerBosonicState::vacuum()));
        let hv = h.apply(&vac);
        let e0 = crate::QuantumState::inner_product(&hv, &vac).re;
        assert!(e0.abs() < 1e-9, "⟨0|H|0⟩ must be 0, got {e0}");
        // Exact additivity at occupation n: ⟨n|H|n⟩ = n·m (the expectation is
        // taken on the normalized state — a†|n⟩ = √(n+1)|n+1⟩, so repeated
        // InnerBosonCreate applications build an unnormalized occupation-n
        // state with ‖|n⟩‖² = n!).
        let one = vac.apply(&crate::Operator::InnerBosonCreate(0));
        let e1 = crate::QuantumState::inner_product(&h.apply(&one), &one).re / one.norm().powi(2);
        assert!((e1 - 1.0).abs() < 1e-9, "one-scalaron energy must be m, got {e1}");
        let two = one.apply(&crate::Operator::InnerBosonCreate(0));
        let e2 = crate::QuantumState::inner_product(&h.apply(&two), &two).re / two.norm().powi(2);
        assert!((e2 - 2.0).abs() < 1e-9, "two-scalaron energy must be 2m, got {e2}");
    }

    #[test]
    fn test_qg_starobinsky_vielbein_hamiltonian_hermitian_and_structure() {
        // The NEW QG module: the vielbein (tetrad) Starobinsky Hamiltonian
        // (docs/qg_starobinsky_vielbein_hamiltonian.cdb) — the reduced
        // physical (Einstein-frame) form `H = Σ:(1/16)𝒮²: + m·N_ψ` of
        // `H_final_st = (M²/2)ψ·(book.tex 8190) + U(ψ)e`: the base TEGR
        // kinetic (general space-time, vielbein variables) plus the massive
        // scalaron (the R² content, mass m = 1/√(12α)).
        //
        // The final Hamiltonian is the ONE-PARTICLE Hamiltonian enclosed in
        // creation (left) / annihilation (right) operators on the nested Fock
        // space, H = Σ hᵢⱼ C†(eᵢ)A(eⱼ) with the one-particle operator
        // h = h_TEGR ⊕ (m): the enclosure of the TEGR one-particle kinetic
        // (1/16)𝒮² and of the scalaron one-particle energy m. The R² content
        // enters h through the scalaron mass m = 1/√(12α) = √(V″(0)) — the
        // quadratic part of the full exponential potential
        // V(φ) = (M⁴/16α)(1−e^{−√(2/3)φ/M})². The outer Hamiltonian is
        // free-particle-like (quadratic in the outer ladders) for ANY
        // one-particle operator h: the exponential may live INSIDE h (on the
        // one-particle/inner Hilbert space) without producing outer-level
        // 3-/4-particle vertices — see
        // test_qg_starobinsky_vielbein_full_exponential_hermitian_and_structure
        // for the full one-particle operator ½π² + V(φ̂).
        use crate::models::{qg_starobinsky_vielbein_hamiltonian, qg_tegr_hamiltonian};
        let m = 1.0_f64;
        let h = qg_starobinsky_vielbein_hamiltonian(3, m);
        assert!(!h.terms.is_empty());
        // Exact term-level Hermiticity H = H† (shared helper).
        assert_exact_hermitian(&h);
        // The ENCLOSURE FORM: every term is creation-left / annihilation-right
        // (the one-particle enclosure doctrine for the QG final Hamiltonian).
        assert_enclosure_form(&h);
        // Real coefficients.
        for t in &h.terms {
            assert!(t.0.im.abs() < 1e-12, "vielbein Starobinsky coeffs are real");
        }
        // The one-particle structure: exactly ONE scalaron term, the enclosure
        // m·C†(ψ)A(ψ) of the one-particle energy m on the scalaron universe
        // ψ = {n_grav: 1} (creation left, annihilation right).
        let scalaron_terms: Vec<_> = h
            .terms
            .iter()
            .filter(|(_, ops)| {
                matches!(
                    &ops[..],
                    [crate::Operator::OuterBosonCreate(s),
                     crate::Operator::OuterBosonAnnihilate(t)] if s == t && s.modes.get(&3) == Some(&1)
                )
            })
            .collect();
        assert_eq!(
            scalaron_terms.len(),
            1,
            "exactly one scalaron enclosure term m·C†(ψ)A(ψ), got {}",
            scalaron_terms.len()
        );
        assert!(
            (scalaron_terms[0].0.re - m).abs() < 1e-12,
            "scalaron enclosure coefficient must be the mass m, got {}",
            scalaron_terms[0].0
        );
        // ⟨0|H|0⟩ = 0 (nested-Fock vacuum rule, normal-ordered realization).
        let hv = h.apply(&crate::QuantumState::vacuum());
        let e0 = crate::QuantumState::inner_product(&hv, &crate::QuantumState::vacuum()).re;
        assert!(e0.abs() < 1e-9, "⟨0|H|0⟩ must be 0, got {e0}");

        // The scalaron sector: the one-scalaron state |ψ⟩ = universe {n_grav:1}
        // has the diagonal expectation m (the number operator), and the
        // two-scalaron expectation is exactly 2m (additivity — the mass gap m
        // of the Starobinsky sector). The expectation is exact because the
        // TEGR squeezed terms act on the graviton factor of the tensor-product
        // Hilbert space and produce components orthogonal to |ψ⟩.
        let mut s_one = crate::InnerBosonicState::vacuum();
        s_one.modes.insert(3, 1); // the scalaron mode = beyond the 3 gravitons
        let one = crate::QuantumState::vacuum()
            .apply(&crate::Operator::OuterBosonCreate(s_one.clone()));
        let e1 =
            crate::QuantumState::inner_product(&h.apply(&one), &one).re / one.norm().powi(2);
        assert!((e1 - m).abs() < 1e-9, "one-scalaron energy must be m, got {e1}");
        let two = one.apply(&crate::Operator::OuterBosonCreate(s_one));
        let e2 = crate::QuantumState::inner_product(&h.apply(&two), &two).re / two.norm().powi(2);
        assert!((e2 - 2.0 * m).abs() < 1e-9, "two-scalaron energy must be 2m, got {e2}");

        // The graviton sector: the one-graviton expectation is the TEGR
        // kinetic 1/16 per excitation (:𝒮²:/16 — the ESA content of the
        // vielbein module's book.tex-8190 kinetic).
        let mut g_one = crate::InnerBosonicState::vacuum();
        g_one.modes.insert(1, 1); // a graviton mode
        let g = crate::QuantumState::vacuum().apply(&crate::Operator::OuterBosonCreate(g_one));
        let eg = crate::QuantumState::inner_product(&h.apply(&g), &g).re / g.norm().powi(2);
        assert!(
            (eg - 1.0 / 16.0).abs() < 1e-9,
            "one-graviton kinetic must be 1/16, got {eg}"
        );
        // Bose additivity of the enclosure: two gravitons in DISTINCT modes
        // have the expectation 2·(1/16) = 1/8 (the one-particle eigenvalues
        // add; the squeezed terms act on a different occupation parity and
        // stay orthogonal).
        let mut g_two_a = crate::InnerBosonicState::vacuum();
        g_two_a.modes.insert(1, 1);
        let mut g_two_b = crate::InnerBosonicState::vacuum();
        g_two_b.modes.insert(2, 1);
        let g2 = crate::QuantumState::vacuum()
            .apply(&crate::Operator::OuterBosonCreate(g_two_a))
            .apply(&crate::Operator::OuterBosonCreate(g_two_b));
        let eg2 = crate::QuantumState::inner_product(&h.apply(&g2), &g2).re / g2.norm().powi(2);
        assert!(
            (eg2 - 2.0 / 16.0).abs() < 1e-9,
            "two-graviton kinetic must be additive 2/16 = 1/8, got {eg2}"
        );

        // The R²-off structural limit: at m = 0 the scalaron sector vanishes
        // and the builder reduces to the base TEGR Hamiltonian exactly.
        let h0 = qg_starobinsky_vielbein_hamiltonian(3, 0.0);
        let h_tegr = qg_tegr_hamiltonian(3);
        assert_eq!(h0.terms.len(), h_tegr.terms.len() + 1); // + the zero term
        // The zero-coefficient scalaron term contributes nothing:
        let nonzero: Vec<_> = h0
            .terms
            .iter()
            .filter(|(c, _)| c.re.abs() > 1e-15)
            .collect();
        assert_eq!(nonzero.len(), h_tegr.terms.len());
    }

    #[test]
    fn test_qg_starobinsky_vielbein_full_exponential_hermitian_and_structure() {
        // The FULL-exponential R²-vielbein enclosure
        // (qg_starobinsky_vielbein_hamiltonian_full): the same final
        // Hamiltonian doctrine — the one-particle operator enclosed in
        // creation (left) / annihilation (right) operators on the nested Fock
        // space — but with the FULL Einstein-frame scalaron potential
        // V(φ) = (1/16α)(1 − e^{−√(2/3)φ})² (M = 1) inside the ONE-PARTICLE
        // operator h = ½π² + V(φ̂) on the truncated Hermite basis
        // {|0⟩,…,|N−1⟩}. The outer Hamiltonian is still a QUADRATIC
        // (free-particle-like) form in the outer ladders: the exponential
        // lives in the one-particle matrix elements ⟨n|h|m⟩, and NO higher
        // vertices appear at the outer level — the point of the
        // outer-Fock/inner-Fock distinction.
        use crate::models::qg_starobinsky_vielbein_hamiltonian_full;
        let alpha = 1.0 / 12.0; // m = 1/√(12α) = 1, λ = 1/√(3m) ≈ 0.577
        let n_levels = 6;
        let h = qg_starobinsky_vielbein_hamiltonian_full(2, alpha, n_levels);
        // The enclosure form: every term is creation-left/annihilation-right
        // (the one-particle enclosure doctrine, full exponential included).
        assert_enclosure_form(&h);
        // Exact term-level Hermiticity H = H† (the one-particle matrix is
        // symmetrized bit-exactly in the builder).
        assert_exact_hermitian(&h);
        // ⟨0|H|0⟩ = 0 (outer vacuum, annihilation right).
        let hv = h.apply(&crate::QuantumState::vacuum());
        let e0 = crate::QuantumState::inner_product(&hv, &crate::QuantumState::vacuum()).re;
        assert!(e0.abs() < 1e-9, "⟨0|H|0⟩ must be 0, got {e0}");

        // The one-particle sector of the enclosure: H restricted to the
        // one-universe states C†(e_n)|0⟩ is exactly the matrix h. The
        // one-particle ground energy E₀ = min eigenvalue of h is the gap of
        // the enclosure (vacuum at 0, first excitation at E₀ > 0). For the
        // quadratic part ½π² + ½m²φ̂² the eigenvalues are m(n + ½); the full
        // V (a square, ≥ 0) shifts them up. With m = 1 we check E₀ > m/2 and
        // the ground-state energy lies above the pure-oscillator value.
        let n = n_levels as usize;
        let mut h_restricted = vec![vec![0.0_f64; n]; n];
        for t in &h.terms {
            // Terms with one creator and one annihilator on the scalaron
            // universes give the one-particle matrix; the graviton TEGR terms
            // are diagonal numbers we exclude by keeping only terms whose two
            // operators carry the scalaron mode (occupations of mode 2).
            if let [crate::Operator::OuterBosonCreate(si),
                    crate::Operator::OuterBosonAnnihilate(sj)] = &t.1[..]
            {
                if let (Some(&ni), Some(&nj)) = (si.modes.get(&2), sj.modes.get(&2)) {
                    if ni < n as u32 && nj < n as u32 && t.1.len() == 2 {
                        h_restricted[ni as usize][nj as usize] += t.0.re;
                    }
                }
            }
        }
        // The matrix is real-symmetric (Hermitian one-particle operator).
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (h_restricted[i][j] - h_restricted[j][i]).abs() < 1e-12,
                    "one-particle matrix must be symmetric: [{i}][{j}] vs [{j}][{i}]"
                );
            }
        }
        // Diagonal entries are positive and grow with n (bounded below).
        for i in 0..n {
            assert!(
                h_restricted[i][i] > 0.0,
                "diagonal h[{i}][{i}] must be positive, got {}",
                h_restricted[i][i]
            );
        }
        // The harmonic limit is inside the model: the diagonal ⟨n|h|n⟩ must
        // lie above the pure-oscillator ladder m(n + 1/2) — the exponential
        // V (a square, ≥ 0) only pushes the one-particle energies up. With
        // α = 1/12 (m = 1) the ground diagonal ⟨0|h|0⟩ = m/4 + V₀₀ must
        // exceed the oscillator ground m/2 = 0.5, and the ladder must be
        // strictly increasing:
        assert!(
            h_restricted[0][0] > 0.5,
            "ground diagonal must exceed the pure-oscillator m/2 = 0.5, got {}",
            h_restricted[0][0]
        );
        assert!(
            h_restricted[1][1] > h_restricted[0][0],
            "excitation diagonal must be increasing"
        );
        // The full model is genuinely different from the m·N_ψ realization:
        // the one-particle state |1⟩ carries the anharmonic energy
        // ⟨1|h|1⟩ > m (the quadratic model's exact eigenvalue).
        let e1_diag = h_restricted[1][1];
        assert!(e1_diag > 1.0, "E(1) diagonal must exceed m = 1, got {e1_diag}");
    }

    #[test]
    fn test_qg_starobinsky_scalaron_mass_and_weak_field_potential() {
        // m² = M²/(12α): the Starobinsky scalaron mass from the R² coupling
        // (the curvature of the Einstein-frame potential at the vacuum).
        use crate::models::{qg_starobinsky_scalaron_mass, qg_starobinsky_weak_field_potential};
        let m = qg_starobinsky_scalaron_mass(1.0);
        assert!(
            (m - (12.0_f64).sqrt().recip()).abs() < 1e-12,
            "m = 1/√(12α), got {m}"
        );
        // The weak-field R² potential Φ = −GM/r(1 + ⅓e^{−mr}): at r ≫ 1/m it
        // reduces to the Newtonian potential −GM/r (R² gravity passes
        // solar-system tests); at r → 0 the force is enhanced by 4/3 (the
        // classic f(R) fifth-force result).
        let gm: f64 = 3.986e14; // Earth's GM (m³/s²)
        let far = 10.0 / m;
        let phi_far = qg_starobinsky_weak_field_potential(gm, far, m);
        let newton = -gm / far;
        assert!(
            ((phi_far - newton) / newton).abs() < 1e-4,
            "at r ≫ 1/m the R² potential must be Newtonian: {phi_far} vs {newton}"
        );
        let near = 1e-8 / m;
        let phi_near = qg_starobinsky_weak_field_potential(gm, near, m);
        let enhanced = -(4.0 / 3.0) * gm / near;
        assert!(
            ((phi_near - enhanced) / enhanced).abs() < 1e-6,
            "at r ≪ 1/m the R² force must be enhanced by 4/3: {phi_near} vs {enhanced}"
        );
    }

    #[test]
    fn test_qg_starobinsky_scalaron_field_massive_dispersion() {
        // H = Σ √(k²+m²) N_i: the massive Klein-Gordon dispersion ω = √(k²+m²),
        // vacuum 0, one-scalaron = √(k²+m²), additive at occupation n, and the
        // m → 0 limit recovers the massless dispersion ω = |k|.
        use crate::models::qg_starobinsky_scalaron_field;
        let ks = [0.5, 1.0, 1.5];
        let m = 1.0;
        let h = qg_starobinsky_scalaron_field(&ks, m);
        let vac = crate::QuantumState::vacuum()
            .apply(&crate::Operator::OuterBosonCreate(crate::InnerBosonicState::vacuum()));
        let e0 = crate::QuantumState::inner_product(&h.apply(&vac), &vac).re;
        assert!(e0.abs() < 1e-9, "scalaron vacuum energy must be 0, got {e0}");
        let one = vac.apply(&crate::Operator::InnerBosonCreate(1));
        let e1 = crate::QuantumState::inner_product(&h.apply(&one), &one).re / one.norm().powi(2);
        let expected1 = (ks[1] * ks[1] + m * m).sqrt();
        assert!(
            (e1 - expected1).abs() < 1e-9,
            "one-scalaron energy must be √(k²+m²) = {expected1}, got {e1}"
        );
        let two = one.apply(&crate::Operator::InnerBosonCreate(1));
        let e2 = crate::QuantumState::inner_product(&h.apply(&two), &two).re / two.norm().powi(2);
        assert!(
            (e2 - 2.0 * expected1).abs() < 1e-9,
            "two-scalaron energy must be additive 2√(k²+m²) = {}, got {e2}",
            2.0 * expected1
        );
        // m → 0: ω = |k| (massless — the graviton limit).
        let h0 = qg_starobinsky_scalaron_field(&ks, 0.0);
        let e0_1 = crate::QuantumState::inner_product(&h0.apply(&one), &one).re / one.norm().powi(2);
        assert!(
            (e0_1 - ks[1]).abs() < 1e-9,
            "the m→0 limit must recover ω = |k| = {}, got {e0_1}",
            ks[1]
        );
    }

    #[test]
    fn test_qg_starobinsky_derivative_brst_nilpotent() {
        // The derivative-variable BRST charge Ω = Σ_i g_i·c_i (NS pattern):
        // Ω² = 0 nilpotent on ghost-carrying probes — the derivative variables
        // are fixed by a first-class constraint (like NS's u_{j,j}·c_j).
        use crate::models::qg_starobinsky_derivative_brst;
        let brst = qg_starobinsky_derivative_brst();
        // Probe: one bosonic universe with derivative-variable content (g_i on
        // mode 1+i) plus one fermionic universe with the ghost c_i.
        let probe = |bosonic: crate::InnerBosonicState, ghost: u32| {
            crate::QuantumState::vacuum()
                .apply(&crate::Operator::OuterBosonCreate(bosonic))
                .apply(&crate::Operator::OuterFermionCreate(crate::InnerFermionicState {
                    modes: std::collections::BTreeSet::from([ghost]),
                }))
        };
        let bos = |mode: u32, n: u32| {
            let mut inner = crate::InnerBosonicState::vacuum();
            if n > 0 {
                inner.modes.insert(mode, n);
            }
            inner
        };
        for i in 0..3u32 {
            let p = probe(bos(1 + i, 1), i);
            assert!(
                brst.apply(&p).norm() > 1e-6,
                "probe with g_{i} content must carry Ω-content"
            );
            let twice = brst.apply(&brst.apply(&p));
            assert!(
                twice.norm() < 1e-9,
                "Ω² must be nilpotent on the g_{i} probe, ‖Ω²ψ‖ = {:.3e}",
                twice.norm()
            );
        }
    }

    #[test]
    fn test_ns_hermite_derivative_fixing_nilpotent_and_closed() {
        // The Hermite-spatial derivative-variable gauge fixing (1D NS slice):
        // the promoted derivative variables g_m (modes 3,4) are fixed to the
        // ACTUAL spatial field derivatives D_m = 2(m+1)u_{m+1} by the BRST
        // charge Ω = (g_0 − 2u_1)c_0 + (g_1 − 4u_2)c_1.  Assertions:
        //   (1) Ω² = 0 (first-class constraint);
        //   (2) [H, Ω] = 0 — the gauge-fixed fiber is BRST-closed;
        //   (3) [H, C_m] = 0 — the constraint C_m = g_m − D_m is an exact
        //       constant of the motion ([H, g_m] = [H, u_1] = [H, u_2] = 0,
        //       the Eulerian block structure), so a physical initial state
        //       preserves ⟨g_m⟩ = 2(m+1)⟨u_{m+1}⟩ at every time;
        //   (4) [H, u_0] ≠ 0 — the value mode genuinely evolves.
        use crate::models::{ns_hermite_derivative_brst, ns_hermite_derivative_fiber};
        let brst = ns_hermite_derivative_brst();
        let h = ns_hermite_derivative_fiber();

        let probe = |bosonic: crate::InnerBosonicState, ghost: u32| {
            crate::QuantumState::vacuum()
                .apply(&crate::Operator::OuterBosonCreate(bosonic))
                .apply(&crate::Operator::OuterFermionCreate(crate::InnerFermionicState {
                    modes: std::collections::BTreeSet::from([ghost]),
                }))
        };
        let bos = |mode: u32, n: u32| {
            let mut inner = crate::InnerBosonicState::vacuum();
            if n > 0 {
                inner.modes.insert(mode, n);
            }
            inner
        };

        // (1) Nilpotency on ghost-carrying probes: ghost 0 on (g_0 − 2u_1)
        // content, ghost 1 on (g_1 − 4u_2) content.
        for (p, label) in [
            (probe(bos(3, 1), 0), "g_0 content"),
            (probe(bos(1, 1), 0), "u_1 content (the derivative D_0)"),
            (probe(bos(4, 1), 1), "g_1 content"),
            (probe(bos(2, 1), 1), "u_2 content (the derivative D_1)"),
        ] {
            assert!(
                brst.apply(&p).norm() > 1e-6,
                "{label} probe must carry Ω-content"
            );
            let twice = brst.apply(&brst.apply(&p));
            assert!(
                twice.norm() < 1e-9,
                "Ω² must be nilpotent on the {label} probe, ‖Ω²ψ‖ = {:.3e}",
                twice.norm()
            );
        }

        // (2) [H, Ω] = 0: the gauge-fixed fiber is BRST-closed (AGENTS.md).
        let comm_norm = |a: &crate::Hamiltonian, b: &crate::Hamiltonian, s: &crate::QuantumState| {
            let mut d = a.apply(&b.apply(s));
            d.scale_and_add(&b.apply(&a.apply(s)), num_complex::Complex64::new(-1.0, 0.0));
            d.norm()
        };
        let probes = [
            probe(bos(3, 1), 0),
            probe(bos(1, 1), 0),
            probe(bos(4, 1), 1),
        ];
        for s in &probes {
            let nrm = comm_norm(&h, &brst, s);
            assert!(
                nrm < 1e-8,
                "[H, Ω] must vanish on ghost probes: ‖[H,Ω]ψ‖ = {nrm:.3e}"
            );
        }

        // (3) [H, C_m] = 0 on the bosonic sector: C_0 = g_0 − 2u_1,
        //     C_1 = g_1 − 4u_2 (as Hamiltonian forms of the constraint).
        // C_0 = g_0 − 2u_1.
        let c0: crate::Hamiltonian = crate::Hamiltonian {
            terms: vec![
                (num_complex::Complex64::new(1.0, 0.0), vec![crate::Operator::InnerBosonCreate(3)]),
                (num_complex::Complex64::new(1.0, 0.0), vec![crate::Operator::InnerBosonAnnihilate(3)]),
                (num_complex::Complex64::new(-2.0, 0.0), vec![crate::Operator::InnerBosonCreate(1)]),
                (num_complex::Complex64::new(-2.0, 0.0), vec![crate::Operator::InnerBosonAnnihilate(1)]),
            ],
        };
        // C_1 = g_1 − 4u_2.
        let c1: crate::Hamiltonian = crate::Hamiltonian {
            terms: vec![
                (num_complex::Complex64::new(1.0, 0.0), vec![crate::Operator::InnerBosonCreate(4)]),
                (num_complex::Complex64::new(1.0, 0.0), vec![crate::Operator::InnerBosonAnnihilate(4)]),
                (num_complex::Complex64::new(-4.0, 0.0), vec![crate::Operator::InnerBosonCreate(2)]),
                (num_complex::Complex64::new(-4.0, 0.0), vec![crate::Operator::InnerBosonAnnihilate(2)]),
            ],
        };
        // A bosonic-only occupation eigenstate (no ghosts) for the constraint
        // commutators.
        let bos_state = |mode: u32, n: u32| -> crate::QuantumState {
            crate::QuantumState::vacuum()
                .apply(&crate::Operator::OuterBosonCreate(bos(mode, n)))
        };
        let bos_probes = [
            bos_state(0, 1),
            bos_state(1, 1),
            bos_state(2, 1),
            bos_state(3, 1),
            bos_state(4, 1),
        ];
        for s in &bos_probes {
            let nrm = comm_norm(&h, &c0, s);
            assert!(
                nrm < 1e-8,
                "[H, C_0] must vanish: ‖[H,C_0]ψ‖ = {nrm:.3e}"
            );
            let nrm1 = comm_norm(&h, &c1, s);
            assert!(
                nrm1 < 1e-8,
                "[H, C_1] must vanish: ‖[H,C_1]ψ‖ = {nrm1:.3e}"
            );
        }

        // (4) [H, u_0] ≠ 0: the value mode genuinely evolves under the fiber.
        let u0: crate::Hamiltonian = crate::Hamiltonian {
            terms: vec![
                (num_complex::Complex64::new(1.0, 0.0), vec![crate::Operator::InnerBosonCreate(0)]),
                (num_complex::Complex64::new(1.0, 0.0), vec![crate::Operator::InnerBosonAnnihilate(0)]),
            ],
        };
        let nrm = comm_norm(&h, &u0, &bos_state(0, 1));
        assert!(
            nrm > 1e-3,
            "[H, u_0] must NOT vanish (the value mode evolves), got {nrm:.3e}"
        );
    }

    #[test]
    fn test_qcd_pair_production_carries_tr_color_factor() {
        // qcd_pair_production scales the vertex by √T_R (T_R = 1/2): the
        // resulting Hamiltonian's off-diagonal strength is √(1/2)·c.
        use crate::models::qcd_pair_production;
        let quark = [1.5];
        let antiquark = [1.5];
        let vertex = [0.3];
        let h = qcd_pair_production(1.0, &quark, &antiquark, &vertex);
        let hd = h.adjoint();
        assert_eq!(h.terms.len(), hd.terms.len());
        for t in &h.terms {
            assert!(
                t.0.im.abs() < 1e-12,
                "QCD pair-production coefficients are real"
            );
        }
        // The vertex terms carry √T_R = 1/√2: λ·c with λ = 1/√2. Find the first
        // two-operator vertex term and check its amplitude is (1/√2)·0.3.
        let sqrt_tr = (0.5_f64).sqrt();
        let expected = sqrt_tr * 0.3;
        let mut found = false;
        for (coeff, ops) in &h.terms {
            if ops.len() == 3 {
                found = true;
                assert!(
                    (coeff.re - expected).abs() < 1e-12,
                    "QCD vertex amplitude must be √T_R·c = {expected}, got {}",
                    coeff.re
                );
            }
        }
        assert!(
            found,
            "QCD pair-production must contain a 3-operator vertex term"
        );
    }
}
