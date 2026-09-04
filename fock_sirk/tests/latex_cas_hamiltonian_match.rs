//! The numerical Hamiltonians, compiled **through the symbolic engine** —
//! `compile_latex` (LaTeX input, the documented mathhook route) and
//! `compile_to_fock` (the `c_0 * a_0` CAS dialect that
//! `prob_kernel::symbolic::normalize_to_cas_dialect` produces from Cadabra2
//! output) — match the Cadabra2-derived forms for QED/QYM/QG/NS.
//!
//! The user directive: the term-matching tests must flow through the
//! LaTeX→Fock compiler rather than only re-checking the builders against
//! hard-coded expansions. This file does exactly that:
//!
//!   • **LaTeX dagger mapping** (the AGENTS.md maintenance item fixed in
//!     `nested_fock_algebra/src/latex.rs`): `a_0^{dagger}` compiles to the
//!     creation operator, so `a†a` is the number operator — never zero
//!     (the pre-fix mathhook power-misparse) and never `a·a` (the pre-fix
//!     `map_to_annihilation` bug). Pinned by `qym_number_operator_latex_*`
//!     and the cross-mode pair test.
//!   • **QYM/QED** (`docs/yang_mills_hamiltonian.cdb`): the Weyl-gauge
//!     Legendre transform `H_final = ½π² + ½B²`, `B = A₀ − A₁` (U(1)
//!     lattice difference), compiled through the CAS dialect — Hermitian,
//!     normal-ordered (⟨0|H|0⟩ = 0) — and compared term-for-term against
//!     the `qcd_ym_hamiltonian(0)` builder (which is itself a
//!     `compile_to_fock` call, so this is compiler-vs-compiler).
//!   • **QG TEGR / densitized** (`docs/qg_gauge_fixed_hamiltonian.cdb`,
//!     book.tex 8190): the `(1/16)𝒮²` and `(1/16)Δ_𝒮 − (1/24)∂²_y` CAS
//!     compiles are checked structurally (4 raw quadratic terms per mode,
//!     the exact 1/16 and −1/24 coefficients, no cubic/quartic) — the
//!     builder realizations are outer-operator normal-ordered forms, so the
//!     equivalence is spectral (checked in `cdb_hamiltonian_match.rs`), not
//!     termwise.
//!   • **NS** (book.tex §4159-4197): the Euler fiber `{π_i, A_i}`,
//!     `A_i = Σ_k A_ik u_k + c_i` via the CAS dialect — term-for-term equal
//!     to the `ns_eulerian_fiber` builder on the shared modes.

use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, QuantumState, compile_latex, compile_to_fock,
};
use num_complex::Complex64;

/// The physical vacuum for **inner** operators: one empty inner universe.
fn inner_vac() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn vac_energy(h: &Hamiltonian) -> f64 {
    let vac = inner_vac();
    QuantumState::inner_product(&vac, &h.apply(&vac)).re
}

fn mode_of(op: &Operator) -> u32 {
    match op {
        Operator::InnerBosonCreate(m) | Operator::InnerBosonAnnihilate(m) => *m,
        Operator::OuterBosonCreate(s) | Operator::OuterBosonAnnihilate(s) => {
            *s.modes.keys().next().expect("outer mode state")
        }
        _ => panic!("expected a boson ladder operator, got {op:?}"),
    }
}

/// Operator equality without `PartialEq`: same kind, same mode.
fn op_eq(a: &Operator, b: &Operator) -> bool {
    matches!(
        (a, b),
        (Operator::InnerBosonCreate(x), Operator::InnerBosonCreate(y)) if x == y
    ) || matches!(
        (a, b),
        (Operator::InnerBosonAnnihilate(x), Operator::InnerBosonAnnihilate(y)) if x == y
    ) || matches!(
        (a, b),
        (Operator::OuterBosonCreate(x), Operator::OuterBosonCreate(y)) if x.modes == y.modes
    ) || matches!(
        (a, b),
        (Operator::OuterBosonAnnihilate(x), Operator::OuterBosonAnnihilate(y)) if x.modes == y.modes
    )
}

/// Is `needle` present in `hay` (coefficient and operator list)?
fn has_term(h: &Hamiltonian, coeff: Complex64, ops: &[Operator]) -> bool {
    h.terms.iter().any(|(c, o)| {
        (c - coeff).norm() < 1e-9
            && o.len() == ops.len()
            && o.iter().zip(ops.iter()).all(|(x, y)| op_eq(x, y))
    })
}

// ─────────────────────────────────────────────
// LaTeX dagger mapping (the latex.rs fix)
// ─────────────────────────────────────────────

#[test]
fn qym_number_operator_latex_dagger_is_exact() {
    // The AGENTS.md maintenance item: `a_i† a_i` must compile to the number
    // operator, NOT to zero (the pre-fix bug: same-mode dagger products
    // vanished because mathhook mis-parsed `a_0^{dagger}` as a power) and
    // NOT to `a_0 * a_0` (the pre-fix `map_to_annihilation` bug: `c` was
    // treated as annihilation).
    let h = compile_latex(r"a_0^{dagger} * a_0");
    assert_eq!(h.terms.len(), 1, "a†a must be exactly one term");
    let (c, ops) = &h.terms[0];
    assert!(
        (c.re - 1.0).abs() < 1e-12 && c.im.abs() < 1e-12,
        "coeff must be 1, got {c}"
    );
    assert!(
        matches!(ops[0], Operator::InnerBosonCreate(0)),
        "first op must be creation on mode 0: {ops:?}"
    );
    assert!(
        matches!(ops[1], Operator::InnerBosonAnnihilate(0)),
        "second op must be annihilation on mode 0: {ops:?}"
    );
    // Acting on the one-quantum state gives |1⟩ with amplitude 1.
    let vac = inner_vac();
    let one = vac.apply(&Operator::InnerBosonCreate(0));
    let h_one = h.apply(&one);
    let key = one.components.keys().next().cloned().unwrap();
    let amp = h_one
        .components
        .get(&key)
        .copied()
        .unwrap_or(Complex64::new(0.0, 0.0));
    assert!(
        (amp - Complex64::new(1.0, 0.0)).norm() < 1e-9,
        "N|1⟩ must be |1⟩, got amplitude {amp}"
    );
}

#[test]
fn cross_mode_latex_dagger_pairs() {
    // Cross-mode dagger products: a†_1 a_2 (creation on 1, annihilation on 2).
    let h = compile_latex(r"a_1^{dagger} * a_2");
    assert_eq!(h.terms.len(), 1);
    let (c, ops) = &h.terms[0];
    assert!((c.re - 1.0).abs() < 1e-12, "coeff must be 1, got {c}");
    assert!(matches!(ops[0], Operator::InnerBosonCreate(1)), "{ops:?}");
    assert!(
        matches!(ops[1], Operator::InnerBosonAnnihilate(2)),
        "{ops:?}"
    );
}

#[test]
fn double_creation_latex_dagger() {
    // a†_0 a†_1 — creation on both modes.
    let h = compile_latex(r"a_0^{dagger} * a_1^{dagger}");
    assert_eq!(h.terms.len(), 1);
    let (c, ops) = &h.terms[0];
    assert!((c.re - 1.0).abs() < 1e-12, "coeff must be 1, got {c}");
    assert!(matches!(ops[0], Operator::InnerBosonCreate(0)), "{ops:?}");
    assert!(matches!(ops[1], Operator::InnerBosonCreate(1)), "{ops:?}");
}

// ─────────────────────────────────────────────
// QYM/QED: the abelian B² via the CAS dialect == qcd_ym_hamiltonian(0)
// ─────────────────────────────────────────────

/// Mechanical CAS→LaTeX-dagger translation: `c_<i>` → `a_<i>^{dagger}`,
/// `a_<i>` unchanged. This is the exact inverse of `rewrite_daggers`
/// (nested_fock_algebra::latex), so a LaTeX fixture produced by this
/// function compiles back to the same operator string the CAS route sees.
fn cas_to_latex_dagger(cas: &str) -> String {
    let b = cas.as_bytes();
    let mut out = String::with_capacity(cas.len() + 16);
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'c' && i + 1 < b.len() && (b[i + 1] == b'_' || b[i + 1].is_ascii_digit()) {
            // Creation token: copy the subscript verbatim, then dagger it.
            out.push_str("a_");
            i += 1;
            if i < b.len() && b[i] == b'_' {
                i += 1;
            }
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'{') {
                let open = b[i] == b'{';
                out.push(b[i] as char);
                i += 1;
                if open {
                    while i < b.len() && b[i] != b'}' {
                        out.push(b[i] as char);
                        i += 1;
                    }
                    if i < b.len() {
                        out.push('}');
                        i += 1;
                    }
                    break;
                }
            }
            out.push_str("^{dagger}");
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// The full B²+kinetic CAS compile is heavy (the distribution over the
/// doubled lattice fields takes minutes): keep it for the slow runs —
/// `cargo test --test latex_cas_hamiltonian_match -- --ignored`.
///
/// The fixture is DERIVED, not transcribed: the expression is
/// [`nested_fock_algebra::qcd_ym_expression`] itself — the single source of
/// truth for the `.cdb`-derived `H_final = ½π² + ½B²`
/// (`docs/yang_mills_hamiltonian.cdb`). Compiler-vs-compiler on the same
/// expression cannot drift.
#[test]
#[ignore = "heavy CAS→Fock compile of the full B²+kinetic (minutes); run with --ignored"]
fn qym_abelian_b2_cas_matches_builder() {
    let cas = nested_fock_algebra::qcd_ym_expression(0.0);
    let h_cas = compile_to_fock(&cas);
    let h_builder = nested_fock_algebra::qcd_ym_hamiltonian(0.0);

    assert!(!h_cas.terms.is_empty());
    assert_eq!(
        h_cas.terms.len(),
        h_cas.adjoint().terms.len(),
        "H must equal H†"
    );
    assert!(
        vac_energy(&h_cas).abs() < 1e-9,
        "⟨0|H|0⟩ = 0 (normal ordered)"
    );
    assert_eq!(
        h_cas.terms.len(),
        h_builder.terms.len(),
        "CAS B²+kinetic must have the same term count as qcd_ym_hamiltonian(0)"
    );
    for (c, ops) in &h_cas.terms {
        assert!(
            has_term(&h_builder, *c, ops),
            "CAS term {c} {ops:?} must appear in qcd_ym_hamiltonian(0)"
        );
    }
    // Purely quadratic (the U(1) limit): no cubic/quartic.
    for (_, ops) in &h_cas.terms {
        assert_eq!(ops.len(), 2, "U(1) B² must be purely quadratic: {ops:?}");
    }
}

/// The full LaTeX dagger compile of the same B² + kinetic is slow (the
/// mathhook→expand path with the double dagger product): keep it for the
/// slow runs — `cargo test --test latex_cas_hamiltonian_match -- --ignored`.
///
/// The LaTeX input is GENERATED from the single-source CAS expression via the
/// mechanical `cas_to_latex_dagger` translation — never hand-maintained.
/// A hand-transcribed LaTeX fixture once drifted from the derived Hamiltonian
/// (it lagged the mode-3 kinetic block and failed 19-vs-22 terms); deriving
/// the fixture makes that failure mode impossible by construction.
#[test]
#[ignore = "heavy LaTeX→Fock compile of the full B²+kinetic (>60 s); run with --ignored"]
fn qym_abelian_b2_latex_dagger_structure() {
    let latex = cas_to_latex_dagger(&nested_fock_algebra::qcd_ym_expression(0.0));
    let h = compile_latex(&latex);
    assert!(!h.terms.is_empty());
    assert_eq!(h.terms.len(), h.adjoint().terms.len(), "H must equal H†");
    assert!(vac_energy(&h).abs() < 1e-9, "⟨0|H|0⟩ = 0 (normal ordered)");
    for (_, ops) in &h.terms {
        assert_eq!(ops.len(), 2, "U(1) B² must be purely quadratic: {ops:?}");
    }
    // Same term count as the builder (the equivalence of the two routes).
    let h_builder = nested_fock_algebra::qcd_ym_hamiltonian(0.0);
    assert_eq!(h.terms.len(), h_builder.terms.len());
}

// ─────────────────────────────────────────────
// QG: TEGR kinetic and densitized kinetic via the CAS dialect
// ─────────────────────────────────────────────

#[test]
fn qg_tegr_kinetic_cas_structure() {
    // (1/16)𝒮² with 𝒮 = 𝒮† + 𝒮: the CAS compile is the RAW expansion
    // (1/16)(a†² + a†a + aa† + aa) — 4 quadratic terms per mode, each with
    // the exact 1/16 coefficient (the framework strips the [a,a†]=1 zero
    // point so ⟨0|H|0⟩ = 0). The builder qg_tegr_hamiltonian realizes the
    // same operator with OUTER ladder operators in the normal-ordered form
    // (3 terms); the spectral equivalence is checked in cdb_hamiltonian_match.
    let cas = "(1/16) * (c_0 + a_0) * (c_0 + a_0)";
    let h = compile_to_fock(cas);
    assert_eq!(h.terms.len(), 4, "raw (a†+a)² expansion: 4 quadratic terms");
    let mut coeffs = vec![];
    for (c, ops) in &h.terms {
        assert_eq!(ops.len(), 2);
        assert_eq!(mode_of(&ops[0]), 0);
        assert_eq!(mode_of(&ops[1]), 0);
        assert!(
            (c.re - 1.0 / 16.0).abs() < 1e-12,
            "coeff must be 1/16, got {c}"
        );
        coeffs.push(*c);
    }
    // The pair structure: (a†+a)² = a†² + a†a + aa† + aa, and the framework
    // normal-orders aa† → a†a + 1 (stripping the zero point), so the compile
    // is a†² + **2**·a†a + a²: one c†c† term, two number terms, one aa term.
    let n_num = h
        .terms
        .iter()
        .filter(|(_, ops)| {
            matches!(ops[0], Operator::InnerBosonCreate(0))
                && matches!(ops[1], Operator::InnerBosonAnnihilate(0))
        })
        .count();
    let n_pair = h
        .terms
        .iter()
        .filter(|(_, ops)| {
            matches!(ops[0], Operator::InnerBosonAnnihilate(0))
                && matches!(ops[1], Operator::InnerBosonCreate(0))
        })
        .count();
    let n_cc = h
        .terms
        .iter()
        .filter(|(_, ops)| {
            matches!(ops[0], Operator::InnerBosonCreate(0))
                && matches!(ops[1], Operator::InnerBosonCreate(0))
        })
        .count();
    let n_aa = h
        .terms
        .iter()
        .filter(|(_, ops)| {
            matches!(ops[0], Operator::InnerBosonAnnihilate(0))
                && matches!(ops[1], Operator::InnerBosonAnnihilate(0))
        })
        .count();
    assert_eq!(
        (n_cc, n_num, n_pair, n_aa),
        (1, 2, 0, 1),
        "normal-ordered expansion must be a†² + 2·a†a + a²"
    );
    let _ = coeffs;
    assert!(vac_energy(&h).abs() < 1e-9, "⟨0|H|0⟩ = 0 (normal ordered)");
    assert_eq!(h.terms.len(), h.adjoint().terms.len(), "H must equal H†");
}

#[test]
fn qg_densitized_kinetic_cas_structure() {
    // H0 = (1/16)Δ_𝒮 − (1/24)∂²_y: two 𝒮 modes at +1/16, one conformal mode
    // at −1/24. Raw expansion: 4 quadratic terms per mode.
    let cas = "(1/16) * (c_0 + a_0) * (c_0 + a_0) - (1/24) * (c_1 + a_1) * (c_1 + a_1)";
    let h = compile_to_fock(cas);
    assert_eq!(h.terms.len(), 8, "2 modes × 4 raw quadratic terms");
    let mut n_plus = 0usize;
    let mut n_minus = 0usize;
    for (c, ops) in &h.terms {
        assert_eq!(ops.len(), 2);
        let m = mode_of(&ops[0]);
        assert_eq!(m, mode_of(&ops[1]));
        if m == 0 {
            assert!(
                (c.re - 1.0 / 16.0).abs() < 1e-12,
                "𝒮 coeff must be 1/16, got {c}"
            );
            n_plus += 1;
        } else {
            assert!(
                (c.re + 1.0 / 24.0).abs() < 1e-12,
                "𝒫 coeff must be −1/24, got {c}"
            );
            n_minus += 1;
        }
        assert!(c.im.abs() < 1e-12, "kinetic coefficients are real");
    }
    assert_eq!((n_plus, n_minus), (4, 4), "4 terms per mode");
    assert!(vac_energy(&h).abs() < 1e-9);
    assert_eq!(h.terms.len(), h.adjoint().terms.len(), "H must equal H†");
}

#[test]
fn qg_scalaron_mass_term_cas() {
    // The scalaron realization H = Σ m·N_i: m·(a†a). compile_to_fock treats
    // bare identifiers symbolically, so substitute the mass numerically.
    let m = 0.5;
    let cas_num = format!("({m}) * (c_0 * a_0)");
    let h = compile_to_fock(&cas_num);
    assert_eq!(h.terms.len(), 1);
    let (c, ops) = &h.terms[0];
    assert!((c.re - m).abs() < 1e-12, "coeff must be m = {m}, got {c}");
    assert!(
        matches!(ops[0], Operator::InnerBosonCreate(0))
            && matches!(ops[1], Operator::InnerBosonAnnihilate(0)),
        "must be a†_0 a_0: {ops:?}"
    );
    // One-scalaron energy: ⟨1|H|1⟩ = m.
    let vac = inner_vac();
    let one = vac.apply(&Operator::InnerBosonCreate(0));
    let hv = h.apply(&one);
    let key = one.components.keys().next().cloned().unwrap();
    let amp = hv
        .components
        .get(&key)
        .copied()
        .unwrap_or(Complex64::new(0.0, 0.0));
    assert!(
        (amp.re - m).abs() < 1e-9 && amp.im.abs() < 1e-9,
        "one-scalaron energy must be m = {m}, got {amp}"
    );
}

// ─────────────────────────────────────────────
// NS: the Euler fiber via the CAS dialect == ns_eulerian_fiber builder
// (book.tex §4159-4197)
// ─────────────────────────────────────────────

#[test]
fn ns_euler_fiber_cas_matches_builder() {
    // The affine Euler fiber: H = Σ_i {π_i, V_i} with V_i = Σ_k A_ik u_k + c_i,
    // π = i(a†−a), u = a†+a — the anti-commutator WITHOUT a ½ (the builder
    // emits π·V + V·π). In the CAS dialect (the framework's imaginary unit
    // is `I`):
    //   I*(c_0 − a_0)*(A01*(c_1 + a_1)) + I*(A01*(c_1 + a_1))*(c_0 − a_0).
    let a01 = 0.5f64;
    let cas = format!(
        "(I) * (c_0 - a_0) * ({a01} * (c_1 + a_1)) + (I) * ({a01} * (c_1 + a_1)) * (c_0 - a_0)"
    );
    let h_cas = compile_to_fock(&cas);
    assert_eq!(
        h_cas.terms.len(),
        8,
        "2 orderings × 4 products = 8 raw terms"
    );

    // The builder for the same 2×2 off-diagonal A (one hopping, no affine c),
    // filtered to the shared modes {0,1}.
    let a: [[f64; 3]; 3] = [[0.0, a01, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let c = [0.0f64, 0.0, 0.0];
    let h_builder = nested_fock_algebra::ns_eulerian_fiber(&a, &c);
    let filtered: Vec<(Complex64, Vec<Operator>)> = h_builder
        .terms
        .iter()
        .filter(|(_, ops)| ops.iter().all(|o| mode_of(o) < 2))
        .cloned()
        .collect();
    assert_eq!(
        filtered.len(),
        8,
        "builder must also emit 8 terms on modes {{0,1}}"
    );

    // Hermitian both.
    assert_eq!(
        h_cas.terms.len(),
        h_cas.adjoint().terms.len(),
        "CAS fiber must be Hermitian"
    );

    // Every CAS term appears in the builder with the same coefficient.
    for (c_cas, ops_cas) in &h_cas.terms {
        assert!(
            has_term(
                &Hamiltonian {
                    terms: filtered.clone()
                },
                *c_cas,
                ops_cas
            ),
            "CAS fiber term {c_cas} {ops_cas:?} must appear in ns_eulerian_fiber"
        );
    }
    // The coefficients are ±i·A01 — the Euler advection amplitude.
    for (c, _) in &h_cas.terms {
        assert!(
            (c.im.abs() - a01).abs() < 1e-9,
            "fiber hopping coefficient must be ±i·A01 = ±{a01}i, got {c}"
        );
    }
}
