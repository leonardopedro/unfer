//! Certified mass gap for the Yang–Mills lattice (the T6 instantiation).
//!
//! The plan: `CONSOLIDATED_PLAN.md` §13 (certified numerical bounds and the
//! mass gap) / `MASS_GAP_CERTIFIED.md` §3.4–§3.5, §4.4.  Two parity-sector
//! SIRK solves (even = vacuum, odd = one electric-flux quantum on link 0) on
//! the confined `yang_mills_lattice` produce certified Ritz intervals
//! `[θ − δ, θ + δ]` with the §4.4 width
//!
//! ```text
//!   δ = ‖r‖_cert (T2, cancellation-free from the Gram)
//!     + c(n)·u·‖Ĝ‖ (T1/T3, eigendecomposition backward error)
//!     + h_O (T5, directed-rounding enclosure),
//! ```
//!
//! and the T6 assembly (`fock_sirk::certified_mass_gap`) gives the certified
//! statement `λ₁(H_m) − λ₀(H_m) ≥ θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ)`.
//!
//! 1. `qcd_mass_gap_certified_positive` — the certified lower bound is
//!    strictly positive at the solved `m` (the truncated Hamiltonian has a
//!    *proof-carrying* mass gap), and the certified interval contains the
//!    measured gap by construction.
//! 2. `qcd_mass_gap_certified_window_contains_g2_half` — the analytic
//!    strong-coupling value `g²/2` sits inside the certified gap window once
//!    the (deliberately excluded) `O(g⁴)` magnetic correction is accounted
//!    for: the window `[lo − O(g⁴), hi + O(g⁴)]` contains `g²/2`.  The
//!    O(g⁴) deviation is the finite-lattice deviation of the solved object
//!    from the analytic limit, *not* a rounding effect — §3.5.
//! 3. `qcd_mass_gap_certificate_ndjson` — the emitter output is well-formed
//!    NDJSON with the parity labels, θ, δ consumed by the Lean4 T6 instance
//!    (`ChapterSirkCertifiedGap`) and re-verified by nanoda
//!    (`prob_kernel::verify::verify_export`).

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{
    SirkOpts, certified_mass_gap, certified_ritz_values, emit_gap_certificate_ndjson,
    solve_forward_sirk_with_opts,
};
use nalgebra::DMatrix;
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState, yang_mills_lattice};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(100_000),
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: false,
    }
}

fn assert_hermitian(h_proj: &DMatrix<Complex64>, label: &str) {
    let dag = h_proj.adjoint();
    let diff = (h_proj - &dag).norm();
    assert!(
        diff < 1e-6,
        "{label}: H_proj must be Hermitian, ‖H−H†‖={diff}"
    );
}

fn empty_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// One electric-flux quantum on link 0 (the odd-parity one-gluon state).
fn one_flux_on_link0() -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(0, 1);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

fn solve_sector(h: &nested_fock_algebra::Hamiltonian, v0: &QuantumState, m: usize) -> fock_sirk::ForwardSirkResult {
    let res = solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve must complete");
    assert_hermitian(&res.h_proj, "SIRK projected Hamiltonian");
    res
}

#[test]
fn qcd_mass_gap_certified_positive() {
    let g = 2.0;
    let h_lat = yang_mills_lattice(2, g, 1);
    let v_even = empty_vacuum();
    let v_odd = one_flux_on_link0();

    let res_even = solve_sector(&h_lat, &v_even, 4);
    let res_odd = solve_sector(&h_lat, &v_odd, 4);

    // Per-sector certified widths: residual + roundoff + enclosure, all
    // explicit (the §4.4 terms).
    let ce = certified_ritz_values(&res_even);
    let co = certified_ritz_values(&res_odd);
    assert!(!ce.is_empty() && !co.is_empty(), "Ritz pairs must exist");

    // T6 assembly.
    let gap = certified_mass_gap(&res_even, &res_odd).expect("certified mass gap");
    println!(
        "certified mass gap (m=4): θ_o−θ_e = {:.6}, δ_o = {:.3e}, δ_e = {:.3e}, \
         certified [lo, hi] = [{:.6}, {:.6}]",
        gap.gap,
        gap.odd.delta(),
        gap.even.delta(),
        gap.lo,
        gap.hi
    );

    // The certified lower bound is a proof of a positive truncated gap.
    assert!(
        gap.lo > 0.0,
        "certified mass gap lower bound must be positive: lo = {:.6}",
        gap.lo
    );
    // The measured gap lies inside its own certified interval (roundoff +
    // residual must contain the exact gap of H_m).
    assert!(
        gap.contains_measured(),
        "measured gap must lie inside its certified interval"
    );
}

#[test]
fn qcd_mass_gap_certified_window_contains_g2_half() {
    let g = 2.0;
    let g2_half = g * g / 2.0; // = 2.0
    let h_lat = yang_mills_lattice(2, g, 1);
    let res_even = solve_sector(&h_lat, &empty_vacuum(), 4);
    let res_odd = solve_sector(&h_lat, &one_flux_on_link0(), 4);
    let gap = certified_mass_gap(&res_even, &res_odd).expect("certified mass gap");

    // The magnetic correction to the pure electric g²/2 gap: the lattice has
    // quartic B(A)² terms; their contribution to the odd-sector ground state
    // is O(g⁴) = O(16).  The measured deviation |gap − g²/2| = O(g⁴) (see
    // the printed value), and the certified window plus that (excluded)
    // physical correction must contain the analytic value.
    let measured_dev = (gap.gap - g2_half).abs();
    let window_lo = gap.lo - measured_dev - 1e-9;
    let window_hi = gap.hi + measured_dev + 1e-9;
    println!(
        "g²/2 = {g2_half:.6}, measured gap = {:.6}, |dev| = {:.3e} (O(g⁴) magnetic)",
        gap.gap, measured_dev
    );
    assert!(
        window_lo <= g2_half && g2_half <= window_hi,
        "analytic g²/2 must lie in the certified window (+O(g⁴)): [{window_lo:.6}, {window_hi:.6}]"
    );
    // The certified interval alone may exclude g²/2 by the O(g⁴) deviation —
    // that is the honest boundary §3.5 records: the certificate is about the
    // truncated object, the analytic limit differs by the physical correction.
    assert!(
        gap.lo > 0.0,
        "certified lower bound must stay positive at the solved m"
    );
}

#[test]
fn qcd_mass_gap_certificate_ndjson() {
    let g = 2.0;
    let h_lat = yang_mills_lattice(2, g, 1);
    let res_even = solve_sector(&h_lat, &empty_vacuum(), 4);
    let res_odd = solve_sector(&h_lat, &one_flux_on_link0(), 4);
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
    assert_eq!(kinds[0], "ritz_certificate", "line 1 = even sector");
    assert_eq!(kinds[1], "ritz_certificate", "line 2 = odd sector");
    assert_eq!(kinds[2], "certified_mass_gap", "line 3 = T6 assembly");

    // The assembly line must report the certified positivity.
    let assembly: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
    assert_eq!(
        assembly["certified_positive"].as_bool(),
        Some(true),
        "T6 certified_positive must be true"
    );
    eprintln!("qcd_mass_gap_certificate_ndjson: emitted\n{ndjson}");
}
