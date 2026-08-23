//! The QED Hamiltonian vs. the full QYM, abelian reduction, and more QED tests.
//!
//! Quantum Electrodynamics is the abelian (U(1)) specialization of Quantum
//! Yang–Mills: one color, vanishing structure constants `f_{abc}`.  The full
//! Weyl-gauge QYM Hamiltonian (`yang_mills_hamiltonian_with_colors`) is
//!
//!   H = −½ Σᵢₐ πᵢₐ² − ½ Σᵢₐ Bᵢₐ² ,   Bᵢₐ = ε_{ijk}(∂_j A_k^a + ½g f_{abc} A_j^b A_k^c) ,
//!
//! and the U(1) case (`n_colors = 1`, `g = 0`) kills every non-abelian term,
//! leaving the free-Maxwell operator
//!
//!   H = −½ Σᵢ πᵢ² − ½ Σᵢ Bᵢ² ,   Bᵢ = ε_{ijk} ∂_j A_k = (∇×A)ᵢ .
//!
//! On a transverse photon mode of frequency `ω`, the sector `½π² + ½ω²A²` is
//! the harmonic oscillator; its second quantization with `[a,a†] = 1` is
//! `ω(a†a + ½)`, whose normal-ordered form (vacuum energy stripped) is exactly
//! what the QED tests evolve with (`qed_free_photon`).  The tests here make
//! that reduction precise:
//!
//!   1. `qed_free_photon_is_normal_ordered_abelian_qym_free_sector` — the
//!      free-photon Hamiltonian used in the QED tests is the normal-ordered
//!      quantization of the abelian QYM free sector: exact operator identities
//!      `2·qed_free_photon(1) + I = ½π²+½A²` (unit frequency, zero-point
//!      stripped) and the general-`ω` identity
//!      `½π²+½ω²A² = ((1+ω²)/ω)·qed_free_photon(ω) + ½(1+ω²)I + ½(ω²−1)(a²+a†²)`,
//!      plus the photon ladder `{0, ω, 2ω}`.
//!   2. `abelian_specialization_of_full_qym_is_free_maxwell` — the *same*
//!      QYM builder with `n_colors = 1`, `g = 0` produces only quadratic terms
//!      (no A³/A⁴) and equals an independently built Maxwell Hamiltonian
//!      `−½Σπ² − ½Σ(∇×A)²` exactly, on the truncated Fock space.
//!   3. `qed_free_photon_coherent_phase_rotation` — free-photon dynamics:
//!      each Fock component rotates by `e^{−iωnt}`; the overlap
//!      `|⟨ψ₀|ψ(t)⟩|² = |1 + e^{−iωt} + e^{−2iωt}|²/9` is checked at the
//!      half-period, the anti-period and the revival.
//!   4. `qed_static_charge_displaced_oscillator_exact_ground` — the QED
//!      matter coupling (the abelian analogue of the QYM interaction) is the
//!      linear `A·J` term `g(B†+B)`; the model is an exactly solvable
//!      displaced oscillator with ground-state energy `−g²/ω`, reproduced by
//!      SIRK to high precision.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
use nalgebra::DMatrix;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, QuantumState, qed_free_photon,
    qed_static_charge_interaction, yang_mills_hamiltonian, yang_mills_hamiltonian_with_colors,
};
use num_complex::Complex64;

// ─────────────────────────────────────────────
// Local mirrors of the QYM builder helpers (nested_fock_algebra/src/models.rs).
// These are reimplemented here so that the reduction check is an independent
// verification, not a re-call of the same code.
// ─────────────────────────────────────────────

/// A hermitian field `φ_mode = a†_mode + a_mode` as (coeff, op) pairs.
fn field_ops(mode: u32) -> Vec<(Complex64, Operator)> {
    vec![
        (Complex64::new(1.0, 0.0), Operator::InnerBosonCreate(mode)),
        (Complex64::new(1.0, 0.0), Operator::InnerBosonAnnihilate(mode)),
    ]
}

/// The conjugate momentum `π_mode = i(a†_mode − a_mode)` as (coeff, op) pairs.
fn momentum_ops(mode: u32) -> Vec<(Complex64, Operator)> {
    vec![
        (Complex64::new(0.0, 1.0), Operator::InnerBosonCreate(mode)),
        (Complex64::new(0.0, -1.0), Operator::InnerBosonAnnihilate(mode)),
    ]
}

/// The Levi-Civita symbol `ε_{ijk}` (0 unless a permutation of (0,1,2), then
/// the permutation parity).
fn epsilon3(i: usize, j: usize, k: usize) -> f64 {
    let v = [i as i64, j as i64, k as i64];
    let mut s = v;
    s.sort_unstable();
    if s != [0, 1, 2] {
        return 0.0;
    }
    let mut inversions = 0i64;
    for a in 0..3 {
        for b in (a + 1)..3 {
            if v[a] > v[b] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Add `coeff · A · A` terms for an operator list `A = Σ (c, op)` pairs.
fn add_quadratic(
    terms: &mut Vec<(Complex64, Vec<Operator>)>,
    coeff: f64,
    ops: &[(Complex64, Operator)],
) {
    for (ca, oa) in ops {
        for (cb, ob) in ops {
            let c = Complex64::new(coeff, 0.0) * ca * cb;
            if c.norm_sqr() > 1e-30 {
                terms.push((c, vec![oa.clone(), ob.clone()]));
            }
        }
    }
}

/// Independent build of the abelian (U(1), `g = 0`, `n_colors = 1`)
/// specialization of the QYM Hamiltonian:
///
///   `H = −½ Σᵢ πᵢ² − ½ Σᵢ Bᵢ²`,  `Bᵢ = Σ_{jk} ε_{ijk} ∂_j A_k`,
///
/// with the A-modes `0..3` and the derivative modes `3..12` (`∂_j A_k` ↦ mode
/// `3 + (i*3+j)`, where `k` is the third index of the `ε_{ijk}` triple) — the
/// same indexing the full QYM builder uses with `n_colors = 1`.
fn abelian_maxwell_direct() -> Hamiltonian {
    let n_colors = 1usize;
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    // Kinetic: −½ πᵢ² on the three A modes.
    for i in 0..3 {
        add_quadratic(&mut terms, -0.5, &momentum_ops(i as u32));
    }
    // Magnetic: −½ Bᵢ², Bᵢ the curl of the derivative modes.
    for i in 0..3 {
        let mut b_i: Vec<(Complex64, Operator)> = Vec::new();
        for j in 0..3 {
            for k in 0..3 {
                let eps = epsilon3(i, j, k);
                if eps == 0.0 {
                    continue;
                }
                let da_mode = (3 * n_colors + (i * 3 + j) * n_colors) as u32;
                for (c, op) in field_ops(da_mode) {
                    b_i.push((c * eps, op));
                }
            }
        }
        add_quadratic(&mut terms, -0.5, &b_i);
    }
    Hamiltonian { terms }
}

// ─────────────────────────────────────────────
// Fock-space state and matrix helpers
// ─────────────────────────────────────────────

/// The physical vacuum for **inner** operators: one empty inner universe.
fn inner_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// A Fock state with the given (mode, occupation) pairs (bosonic inner modes),
/// normalized: the framework's ladder algebra builds `|n⟩` with amplitude
/// `√(n!)` for `n > 1`, so multi-creation states are rescaled to unit norm
/// (the basis used for exact matrix checks must be orthonormal).
fn fock_state(occ: &[(u32, u32)]) -> QuantumState {
    let mut s = inner_vacuum();
    for &(m, n) in occ {
        for _ in 0..n {
            s = s.apply(&Operator::InnerBosonCreate(m));
        }
    }
    let norm = s.norm();
    if (norm - 1.0).abs() > 1e-12 {
        s.scale_and_add(&s.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));
    }
    s
}

/// All Fock states with total occupation ≤ 2 over `n_modes` bosonic modes.
fn truncated_basis(n_modes: u32) -> Vec<QuantumState> {
    let mut basis = vec![fock_state(&[])];
    for m in 0..n_modes {
        basis.push(fock_state(&[(m, 1)]));
    }
    for m1 in 0..n_modes {
        for m2 in m1..n_modes {
            basis.push(fock_state(&[(m1, 1), (m2, 1)]));
        }
    }
    basis
}

/// The exact matrix of `h` on the (orthonormal) basis:
/// `M[i,j] = ⟨basis_i | h | basis_j⟩`.
fn matrix_of(h: &Hamiltonian, basis: &[QuantumState]) -> DMatrix<Complex64> {
    let n = basis.len();
    let mut m = DMatrix::<Complex64>::zeros(n, n);
    for (j, s) in basis.iter().enumerate() {
        let hs = h.apply(s);
        for (i, t) in basis.iter().enumerate() {
            m[(i, j)] = QuantumState::inner_product(t, &hs);
        }
    }
    m
}

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn sirk_ground(h: &Hamiltonian, v0: &QuantumState, m: usize) -> f64 {
    let opts = SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
    };
    let res = solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts)
        .expect("SIRK solve must complete");
    res.ground_state_energy().expect("ground-state Ritz value")
}

// ─────────────────────────────────────────────
// 1. qed_free_photon = normal-ordered abelian QYM free sector
// ─────────────────────────────────────────────

#[test]
fn qed_free_photon_is_normal_ordered_abelian_qym_free_sector() {
    // The Weyl-gauge QYM is ½Σπ² + ½ΣB².  In the abelian (U(1)) case, on a
    // transverse photon mode of frequency ω, B² = ω²A², so the free sector per
    // mode is the oscillator H_ω = ½π² + ½ω²A².  With the framework's inner
    // ladder operators (π = i(a†−a), A = a†+a) this expands exactly to
    //
    //   H_ω = ½(ω²−1)(a†²+a²) + (1+ω²)(a†a + ½I) ,           (*)
    //
    // while the QED test Hamiltonian is the *normal-ordered* photon energy
    // `qed_free_photon(ω) = ω·a†a` (vacuum energy stripped — the framework
    // vacuum rule ⟨0|H|0⟩ = 0).  We verify both the ladder spectrum and the
    // exact operator identity linking the two.
    for &omega in &[0.5, 1.0, 2.0, 3.7] {
        // H_ω = ½π² + ½ω²A² via the same ladder machinery the QYM builder uses.
        let mut terms = Vec::new();
        add_quadratic(&mut terms, 0.5, &momentum_ops(0));
        add_quadratic(&mut terms, 0.5 * omega * omega, &field_ops(0));
        let h_osc = Hamiltonian { terms };

        // The squeezing part S_ω = ½(ω²−1)(a†²+a²) of (*).
        let mut sq = Vec::new();
        let c_sq = Complex64::new(0.5 * (omega * omega - 1.0), 0.0);
        sq.push((c_sq, vec![Operator::InnerBosonCreate(0), Operator::InnerBosonCreate(0)]));
        sq.push((c_sq, vec![Operator::InnerBosonAnnihilate(0), Operator::InnerBosonAnnihilate(0)]));
        let h_sq = Hamiltonian { terms: sq };

        let basis = vec![fock_state(&[]), fock_state(&[(0, 1)]), fock_state(&[(0, 2)])];
        let m_osc = matrix_of(&h_osc, &basis);
        let m_n = matrix_of(&qed_free_photon(&[omega]), &basis);
        let m_sq = matrix_of(&h_sq, &basis);
        let m_id = DMatrix::<Complex64>::identity(3, 3);

        // (a) The normal-ordered photon ladder {0, ω, 2ω} — what the QED tests
        //     actually evolve with (vacuum energy exactly 0).
        let vals_n = m_n.clone().symmetric_eigen().eigenvalues;
        let expected_n = [0.0, omega, 2.0 * omega];
        for (v, e) in vals_n.iter().zip(expected_n.iter()) {
            assert!(
                (v - e).abs() < 1e-9,
                "ω={omega}: normal-ordered eigenvalue {v} must be {e}"
            );
        }

        // (b) The general-ω operator identity (*), as an exact matrix equality:
        //     H_ω = ((1+ω²)/ω)·qed_free_photon(ω) + ½(1+ω²)·I + ½(ω²−1)(a†²+a²).
        let rhs = m_n.clone() * Complex64::new((1.0 + omega * omega) / omega, 0.0)
            + m_id.clone() * Complex64::new(0.5 * (1.0 + omega * omega), 0.0)
            + m_sq;
        let diff = (m_osc.clone() - rhs).norm();
        assert!(
            diff < 1e-9,
            "ω={omega}: H_ω must satisfy the operator identity (*), ‖Δ‖={diff}"
        );

        // (c) Unit frequency: the squeezing vanishes, and (*) reduces to the
        //     clean normal-ordering statement ½π² + ½A² = 2·qed_free_photon(1)
        //     + I — the QED Hamiltonian is the abelian QYM free sector with the
        //     constant term stripped.  (The constant is the full `I`, not `½I`:
        //     the framework's inner `π = i(a†−a)`, `A = a†+a` are the
        //     unit-mass conventions, so ½π²+½A² = a†a + aa† = 2N + I.)
        if (omega - 1.0).abs() < 1e-12 {
            let rhs1 = m_n.clone() * Complex64::new(2.0, 0.0) + m_id.clone();
            let diff1 = (m_osc.clone() - rhs1).norm();
            assert!(
                diff1 < 1e-9,
                "ω=1: ½π²+½A² must be 2·qed_free_photon(1)+I, ‖Δ‖={diff1}"
            );
            let zero_point = m_id.clone();
            let diff2 = ((m_osc.clone() - zero_point) - m_n.clone() * Complex64::new(2.0, 0.0))
                .norm();
            assert!(
                diff2 < 1e-9,
                "ω=1: qed_free_photon(1) must be (½π²+½A²−I)/2, ‖Δ‖={diff2}"
            );
        }
    }
}

// ─────────────────────────────────────────────
// 2. The abelian specialization of the full QYM builder is free Maxwell
// ─────────────────────────────────────────────

#[test]
fn abelian_specialization_of_full_qym_is_free_maxwell() {
    // The U(1) specialization built by the *same* QYM machinery: one color and
    // no coupling.  The structure constants vanish (`f_{000} = 0`), so every
    // term must be quadratic (the non-abelian A³/A⁴ pieces of B² drop out
    // identically).
    let u1 = yang_mills_hamiltonian_with_colors(0.0, 1);
    assert!(!u1.terms.is_empty(), "U(1) QYM must be non-empty");
    for (c, ops) in &u1.terms {
        assert!(
            ops.len() == 2,
            "U(1) QYM must be purely quadratic, got a {}-operator term (coeff {c})",
            ops.len()
        );
    }
    // Kinetic: 3 A-modes × 4 products = 12; magnetic: each Bᵢ = (∇×A)ᵢ is a
    // 2-term curl, so Bᵢ² = 4² = 16 products; 3 components × 16 = 48.  Total 60.
    assert_eq!(
        u1.terms.len(),
        60,
        "U(1) QYM must have 60 quadratic terms (12 kinetic + 48 magnetic), got {}",
        u1.terms.len()
    );

    // The same property holds for the full 8-color model at g = 0 (the coupling
    // is what activates the f-terms; at g = 0 the model is just 8 decoupled
    // copies of the U(1) one: 8 × 60 = 480 quadratic terms).
    let g0_8 = yang_mills_hamiltonian(0.0);
    for (c, ops) in &g0_8.terms {
        assert!(
            ops.len() == 2,
            "QYM(g=0, 8 colors) must be purely quadratic, got a {}-operator term (coeff {c})",
            ops.len()
        );
    }
    assert_eq!(
        g0_8.terms.len(),
        480,
        "QYM(g=0, 8 colors) must be 8 × 60 = 480 quadratic terms, got {}",
        g0_8.terms.len()
    );

    // (c) The U(1) specialization equals, exactly, an independently built
    //     Maxwell Hamiltonian −½Σπ² − ½Σ(∇×A)² on the same 12-mode set.
    let direct = abelian_maxwell_direct();
    assert_eq!(
        u1.terms.len(),
        direct.terms.len(),
        "direct Maxwell build must have the same term count"
    );
    let basis = truncated_basis(12);
    let m_u1 = matrix_of(&u1, &basis);
    let m_dir = matrix_of(&direct, &basis);
    let diff = (m_u1.clone() - m_dir).norm();
    assert!(
        diff < 1e-9,
        "abelian QYM specialization must equal the direct Maxwell build on the truncated \
         Fock space, ‖Δ‖={diff}"
    );

    // Hermiticity of the abelian QYM Hamiltonian.
    let hdiff = (m_u1.clone() - m_u1.adjoint()).norm();
    assert!(
        hdiff < 1e-9,
        "abelian QYM Hamiltonian matrix must be Hermitian, ‖H−H†‖={hdiff}"
    );

    // The vacuum expectation of the *raw* quadratic Maxwell is the sum of the
    // (unrenormalized) oscillator zero-points: each of the 3 kinetic modes
    // contributes −½ and each of the 3 curl components −1, giving exactly
    // −4.5.  (The normal-ordered photon Hamiltonian of test 1 is the one with
    // ⟨0|H|0⟩ = 0.)
    let vac = fock_state(&[]);
    let hv = u1.apply(&vac);
    let ev = QuantumState::inner_product(&vac, &hv);
    assert!(
        (ev.re + 4.5).abs() < 1e-9 && ev.im.abs() < 1e-9,
        "U(1) QYM vacuum expectation must be the zero-point sum −4.5, got {ev}"
    );
}

// ─────────────────────────────────────────────
// 3. More QED tests: free-photon coherent phase rotation
// ─────────────────────────────────────────────

#[test]
fn qed_free_photon_coherent_phase_rotation() {
    // Free-photon evolution is exactly the phase rotation |n⟩ ↦ e^{−iωnt}|n⟩.
    // For the normalized superposition ψ₀ = (|0⟩ + |1⟩ + |2⟩)/√3,
    //   |⟨ψ₀|ψ(t)⟩|² = |1 + e^{−iωt} + e^{−2iωt}|² / 9,
    // which is 1/9 at t = π/ω (the odd component flips) and 1 at t = 2π/ω
    // (full revival).
    let omega = 1.0;
    let h = qed_free_photon(&[omega]);

    let mut v0 = fock_state(&[]);
    v0.scale_and_add(&fock_state(&[(0, 1)]), Complex64::new(1.0, 0.0));
    v0.scale_and_add(&fock_state(&[(0, 2)]), Complex64::new(1.0, 0.0));
    let norm = v0.norm();
    assert!((norm - 3.0f64.sqrt()).abs() < 1e-12, "seed norm must be √3");
    v0.scale_and_add(&v0.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));
    assert!((v0.norm() - 1.0).abs() < 1e-12, "seed must be normalized");

    let opts = SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
    };
    let res = solve_forward_sirk_with_opts(&h, &v0, &shifts(8), &best_device(), None, &opts)
        .expect("SIRK solve must complete");

    // The whitened Krylov basis is orthonormal, so the overlap of the evolved
    // state with the initial state is ⟨b(0) | b(t)⟩ in whitened coordinates.
    let b0 = res.time_evolve(0.0);
    let ov0 = b0.conjugate().dot(&b0).norm_sqr();
    assert!(
        (ov0 - 1.0).abs() < 1e-8,
        "overlap at t=0 must be 1, got {ov0}"
    );

    // At ωt = π/2 the phases are 1, −i, −1 so the overlap is (1 − i − 1)/3 = −i/3
    // (|·|² = 1/9); at ωt = π they are 1, −1, +1 (overlap (1 − 1 + 1)/3 = 1/3,
    // |·|² = 1/9); at ωt = 2π all three realign (|·|² = 1, full revival).
    for &(t, expected_sq) in &[
        (std::f64::consts::PI / (2.0 * omega), 1.0 / 9.0),
        (std::f64::consts::PI / omega, 1.0 / 9.0),
        (2.0 * std::f64::consts::PI / omega, 1.0),
    ] {
        let bt = res.time_evolve(t);
        let ov = b0.conjugate().dot(&bt);
        let sq = ov.norm_sqr();
        assert!(
            (sq - expected_sq).abs() < 1e-6,
            "t = {t}: |⟨ψ₀|ψ(t)⟩|² must be {expected_sq}, got {sq}"
        );
    }

    // Explicitly: at t = π/ω the three components |0⟩,|1⟩,|2⟩ pick up phases
    // 1, −1, +1 so ⟨ψ₀|ψ(π/ω)⟩ = (1 − 1 + 1)/3 = 1/3 → |·|² = 1/9.
    let bpi = res.time_evolve(std::f64::consts::PI / omega);
    let ov_pi = b0.conjugate().dot(&bpi);
    assert!(
        (ov_pi.re - 1.0 / 3.0).abs() < 1e-6 && ov_pi.im.abs() < 1e-6,
        "⟨ψ₀|ψ(π/ω)⟩ must be 1/3 (real), got {ov_pi}"
    );
}

// ─────────────────────────────────────────────
// 4. More QED tests: the A·J coupling is an exactly solvable displaced
//    oscillator — the abelian analogue of the QYM interaction
// ─────────────────────────────────────────────

#[test]
fn qed_static_charge_displaced_oscillator_exact_ground() {
    // In QED the interaction with a static charge is the linear coupling
    // A·J ∝ g(B† + B) — the abelian analogue of the QYM cubic interaction,
    // which vanishes for U(1) (no A³ self-coupling).  The single-mode model
    //   H = ω N + g(B† + B)
    // is a displaced oscillator with the exact ground-state energy −g²/ω.
    // Verify SIRK reproduces it to high precision across couplings.
    let two_pi_sq = 2.0 * std::f64::consts::PI * std::f64::consts::PI;

    // (a) The interaction terms are linear in the photon field (1-operator),
    //     and the free part is the 2-operator number term — the abelian
    //     structure of the QED Hamiltonian.
    let h2 = qed_static_charge_interaction(&[(1.0, 0.01)], 1.0, 0.3);
    let n_linear = h2.terms.iter().filter(|(_, ops)| ops.len() == 1).count();
    let n_free = h2.terms.iter().filter(|(_, ops)| ops.len() == 2).count();
    assert_eq!(n_linear, 2, "A·J coupling must give 2 linear terms (B† and B)");
    assert_eq!(n_free, 1, "the free photon must be the single number term");

    for &(k, dk, r, e) in &[
        (1.0, 0.01, 1.0, 0.3),
        (0.5, 0.05, 2.0, 0.9),
        (1.5, 0.1, 3.0, 1.2),
    ] {
        let h = qed_static_charge_interaction(&[(k, dk)], r, e);
        let kr = k * r;
        let g = (e * e * dk * k * (1.0 + kr.sin() / kr) / two_pi_sq).sqrt();
        let exact = -g * g / k;
        let e_gs = sirk_ground(&h, &QuantumState::vacuum(), 10);
        let err = (e_gs - exact).abs();
        assert!(
            err < 1e-9,
            "(k={k}, e={e}): displaced-oscillator ground energy must be −g²/ω = {exact:.12e}, \
             SIRK gave {e_gs:.12e} (err {err:.2e})"
        );
    }
}
