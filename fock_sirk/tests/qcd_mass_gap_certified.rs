//! Certified mass gap on the 3D gauge-fixed QYM Hamiltonian (the T6
//! instantiation in the nested Fock space).
//!
//! The plan: `CONSOLIDATED_PLAN.md` §13 / `MASS_GAP_CERTIFIED.md` §3.4–§3.5,
//! §4.4, instantiated on the Cadabra-derived gauge-fixed Hamiltonian
//! `qcd_ym_hamiltonian(g)` (`docs/yang_mills_hamiltonian.cdb`:
//! `H_final = ½π² + ½B²`) instead of the lattice. Two reflection-sector SIRK
//! solves (R-even = vacuum, R-odd = (|1₀⟩+|1₁⟩)/√2 — the exact `Z₂`
//! symmetry `R: (A₀,A₁) → (−A₁,−A₀)` of the gauge-fixed H, verified in
//! `qym_mass_gap.rs`) produce certified Ritz intervals `[θ − δ, θ + δ]`
//! with the §4.4 width, and the T6 assembly
//! (`fock_sirk::certified_mass_gap`) gives the certified enclosure of the
//! sector-ground difference.
//!
//! 1. `qcd_mass_gap_certified_enclosure` — the certified interval encloses
//!    the EXACT truncated spectral gap `E₁ − E₀` of the gauge-fixed H
//!    (cross-checked against the exact `N ≤ 8` diagonalization), and that
//!    gap is positive — the truncated Hamiltonian is gapped.
//! 2. `qcd_mass_gap_certified_window_contains_exact_gap` — the certified
//!    window contains the exact truncated gap across the truncation family
//!    (the honest "window contains the physics" statement; the gauge-fixed
//!    gap is `≈ 0.09` at `g = 1` — NOT the lattice's `g²/2`, which is an
//!    electric-lattice statement the continuum gauge-fixed H does not make).
//! 3. `qcd_mass_gap_certificate_ndjson` — the emitter output is well-formed
//!    NDJSON with the sector labels, θ, δ consumed by the Lean4 T6 instance
//!    (`ChapterSirkCertifiedGap`); the `certified_positive` flag truthfully
//!    reflects `lo > 0` of the emitted numbers.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{
    SirkOpts, certified_mass_gap, certified_ritz_values, emit_gap_certificate_ndjson,
    solve_forward_sirk_with_opts,
};
use nalgebra::DMatrix;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, QuantumState, qcd_ym_hamiltonian,
};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
        // Unit-norm frame: the gauge-fixed H's deep squeezing spans a wide
        // spectral window (raw frame's Gram wall caps usable m).
        unit_norm_steps: true,
    }
}

fn assert_hermitian(h_proj: &DMatrix<Complex64>, label: &str) {
    let dag = h_proj.adjoint();
    let diff = (h_proj - &dag).norm();
    // The strong-coupling h_proj has ‖H‖ ~ 10², so the absolute roundoff is
    // ~1e-6 — tiny relative to the entries. 1e-4 absolute is still ~1e-6
    // relative Hermiticity.
    assert!(
        diff < 1e-4,
        "{label}: H_proj must be Hermitian, ‖H−H†‖={diff}"
    );
}

fn empty_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn fock_state(occ: &[(u32, u32)]) -> QuantumState {
    let mut s = empty_vacuum();
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

/// The R-odd one-quantum start `(|1₀⟩ + |1₁⟩)/√2`.
fn r_odd_start() -> QuantumState {
    let s0 = fock_state(&[(0, 1)]);
    let s1 = fock_state(&[(1, 1)]);
    let mut s = s0;
    s.scale_and_add(&s1, Complex64::new(1.0, 0.0));
    let inv = 1.0 / 2.0f64.sqrt();
    s.scale_and_add(&s.clone(), Complex64::new(inv - 1.0, 0.0));
    s
}

fn solve_sector(
    h: &nested_fock_algebra::Hamiltonian,
    v0: &QuantumState,
    m: usize,
) -> fock_sirk::ForwardSirkResult {
    let res = solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve must complete");
    assert_hermitian(&res.h_proj, "SIRK projected Hamiltonian");
    res
}

/// Exact low window `(E₀, E₁, E₁−E₀)` of the truncated gauge-fixed H.
fn exact_gap(h: &Hamiltonian, n_max: u32) -> (f64, f64, f64) {
    let mut basis = Vec::new();
    for n0 in 0..=n_max {
        for n1 in 0..=n_max - n0 {
            for n2 in 0..=n_max - n0 - n1 {
                for n3 in 0..=n_max - n0 - n1 - n2 {
                    let mut occ = Vec::new();
                    if n0 > 0 {
                        occ.push((0, n0));
                    }
                    if n1 > 0 {
                        occ.push((1, n1));
                    }
                    if n2 > 0 {
                        occ.push((2, n2));
                    }
                    if n3 > 0 {
                        occ.push((3, n3));
                    }
                    basis.push(fock_state(&occ));
                }
            }
        }
    }
    let n = basis.len();
    let mut m = DMatrix::<Complex64>::zeros(n, n);
    for (j, s) in basis.iter().enumerate() {
        let hs = h.apply(s);
        for (i, t) in basis.iter().enumerate() {
            m[(i, j)] = QuantumState::inner_product(t, &hs);
        }
    }
    let mut vals: Vec<f64> = m.symmetric_eigen().eigenvalues.iter().cloned().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (vals[0], vals[1], vals[1] - vals[0])
}

#[test]
fn qcd_mass_gap_certified_enclosure() {
    let g = 1.0;
    let h = qcd_ym_hamiltonian(g);
    let v_even = empty_vacuum();
    let v_odd = r_odd_start();

    let res_even = solve_sector(&h, &v_even, 12);
    let res_odd = solve_sector(&h, &v_odd, 12);

    // Per-sector certified widths: residual + roundoff + enclosure (§4.4).
    let ce = certified_ritz_values(&res_even);
    let co = certified_ritz_values(&res_odd);
    assert!(!ce.is_empty() && !co.is_empty(), "Ritz pairs must exist");

    // T6 assembly.
    let gap = certified_mass_gap(&res_even, &res_odd).expect("certified mass gap");
    println!(
        "certified mass gap (m=12): θ_o−θ_e = {:.6}, δ_o = {:.3e}, δ_e = {:.3e}, \
         certified [lo, hi] = [{:.6}, {:.6}]",
        gap.gap,
        gap.odd.delta(),
        gap.even.delta(),
        gap.lo,
        gap.hi
    );

    // The certified interval encloses the exact truncated spectral gap of the
    // gauge-fixed H (cross-checked against the exact N ≤ 8 diagonalization),
    // and that gap is positive — the truncated Hamiltonian is gapped.
    let (_, _, exact_gap) = exact_gap(&h, 8);
    assert!(
        gap.lo - 1e-9 <= exact_gap && exact_gap <= gap.hi + 1e-9,
        "certified interval must enclose the exact truncated gap E₁−E₀ = {exact_gap:.6}: \
         [{:.6}, {:.6}]",
        gap.lo,
        gap.hi
    );
    assert!(
        exact_gap > 0.05,
        "the truncated gauge-fixed H must be gapped at g=1: E₁−E₀ = {exact_gap:.6}"
    );
    // The measured gap lies inside its own certified interval.
    assert!(
        gap.contains_measured(),
        "measured gap must lie inside its certified interval"
    );
}

#[test]
fn qcd_mass_gap_certified_window_contains_exact_gap() {
    // The certified window from the SIRK sector solves contains the exact
    // truncated spectral gap E₁ − E₀ across the truncation family (N ≤ 6 and
    // N ≤ 8), for g ∈ {1, 2}. This replaces the lattice's "window contains
    // g²/2": the gauge-fixed H's gap is its own (≈ 0.09 at g = 1, growing
    // with g), not the electric-lattice value.
    for &g in &[1.0_f64, 2.0] {
        let h = qcd_ym_hamiltonian(g);
        let res_even = solve_sector(&h, &empty_vacuum(), 12);
        let res_odd = solve_sector(&h, &r_odd_start(), 12);
        let gap = certified_mass_gap(&res_even, &res_odd).expect("certified mass gap");
        for n in [6u32, 8] {
            let (_, _, exact_gap) = exact_gap(&h, n);
            println!(
                "g = {g}, N ≤ {n}: exact gap E₁−E₀ = {exact_gap:.6}, certified window \
                 [{:.6}, {:.6}]",
                gap.lo, gap.hi
            );
            assert!(
                gap.lo - 1e-9 <= exact_gap && exact_gap <= gap.hi + 1e-9,
                "g={g}, N≤{n}: certified window [{:.6}, {:.6}] must contain E₁−E₀ = \
                 {exact_gap:.6}",
                gap.lo,
                gap.hi
            );
        }
        assert!(gap.contains_measured(), "measured gap inside its interval");
    }
}

#[test]
fn qcd_mass_gap_certificate_ndjson() {
    let g = 1.0;
    let h = qcd_ym_hamiltonian(g);
    let res_even = solve_sector(&h, &empty_vacuum(), 12);
    let res_odd = solve_sector(&h, &r_odd_start(), 12);
    let gap = certified_mass_gap(&res_even, &res_odd).expect("certified mass gap");

    let ndjson = emit_gap_certificate_ndjson(&gap);
    let lines: Vec<&str> = ndjson.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 3, "one line per sector + assembly: {ndjson}");

    // Every line is valid JSON with the expected kind.
    let mut kinds = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let v: serde_json::Value =
            serde_json::from_str(line).expect("certificate line must be valid JSON");
        kinds.push(
            v.get("kind")
                .and_then(|k| k.as_str())
                .unwrap_or("")
                .to_string(),
        );
        // Ritz lines carry θ and δ; the assembly line carries gap and δ.
        if i < 2 {
            assert!(
                v.get("theta").is_some() && v.get("delta").is_some(),
                "ritz certificate must carry θ and δ: {line}"
            );
        } else {
            assert!(
                v.get("gap").is_some() && v.get("delta").is_some(),
                "assembly must carry gap and δ: {line}"
            );
        }
    }
    assert_eq!(kinds[0], "ritz_certificate", "line 1 = R-even sector");
    assert_eq!(kinds[1], "ritz_certificate", "line 2 = R-odd sector");
    assert_eq!(kinds[2], "certified_mass_gap", "line 3 = T6 assembly");

    // The assembly line reports the certified positivity truthfully: it must
    // equal `lo > 0` of the emitted numbers (the gauge-fixed H's slow Krylov
    // convergence at the solved m honestly reports `false` here — the
    // certified *enclosure* of the exact gap is the statement, asserted in
    // the two tests above).
    let assembly: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
    assert_eq!(
        assembly["certified_positive"].as_bool(),
        Some(gap.lo > 0.0),
        "T6 certified_positive must truthfully reflect lo > 0"
    );
    eprintln!("qcd_mass_gap_certificate_ndjson: emitted\n{ndjson}");
}
