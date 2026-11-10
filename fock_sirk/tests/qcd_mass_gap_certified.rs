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
//! ## What the certificate does and does not establish (2026-11-10 audit)
//!
//! The T6 interval at `m = 12` is **7.438 wide at `g = 1` and 50.8 wide at
//! `g = 2`** (`δᵒ = 4.088`, `δᵉ = 3.350`), against gaps of `0.0912` and
//! `1.244`. Two consequences the earlier version of this file did not state:
//!
//! * `lo > 0` — the actual mass-gap statement — **is not reachable here**
//!   (`lo = −8.009` at `g = 1`), which is why the emitter's own
//!   `"certified_positive":false` is truthful. The suite asserts the
//!   *enclosure* (a valid bound in both directions, which can fail if the
//!   assembly is wrong) and explicitly pins the non-resolution, so it cannot
//!   silently read as a certified gap. `qym_mass_gap.rs` and
//!   `mass_gap_campaign.rs` cover the same object with the same honesty; the
//!   campaign is where "which rows actually certify" is recorded.
//! * "the certified window contains the exact gap" and its sibling "the
//!   measured gap lies inside its own interval" (`contains_measured`) are
//!   statements that cannot fail: a 7.4-wide window contains everything near
//!   zero, and `contains(value)` follows from `δ ≥ 0` by construction (the
//!   library says so itself). They are replaced below by the assembly
//!   identity, by the non-resolution pin, and by the dense-diagonalization
//!   claim the window was meant to support.
//!
//! 1. `qcd_mass_gap_certified_enclosure` — the certified interval encloses
//!    the EXACT truncated spectral gap `E₁ − E₀` of the gauge-fixed H
//!    (cross-checked against the exact `N ≤ 8` diagonalization), the gap is
//!    positive, and the certificate does **not** resolve it (width ≫ gap).
//! 2. `qcd_mass_gap_window_cannot_resolve_the_gauge_fixed_gap` — the window
//!    contains the exact gap *and* the lattice value `g²/2`, i.e. it cannot
//!    distinguish them; the departure is asserted from the dense
//!    diagonalization instead (the gauge-fixed gap is `≈ 0.09` at `g = 1`, NOT
//!    `g²/2 = 0.5`).
//! 3. `qcd_mass_gap_certificate_ndjson` — the emitter output is well-formed
//!    NDJSON with the sector labels, θ, δ consumed by the Lean4 T6 instance
//!    (`ChapterSirkCertifiedGap`); the `certified_positive` flag truthfully
//!    reflects `lo > 0` of the emitted numbers.
//!
//! ## Measured runtimes (M4, 2026-10-07)
//!
//! Debug (`cargo test`, dev profile): **3 passed / 882.50s**; release
//! (`cargo test -p fock_sirk --release`, the CI profile): **3 passed /
//! 13.53s** (~65×). The debug numbers are recorded so the cost of running
//! the default profile is known before anyone starts one.

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
    // The assembly identity — a real check of `certified_mass_gap`, unlike
    // `contains_measured()` (true by construction: the width IS the sum of the
    // sector widths). A dropped term, a swapped sector or a sign slip fails here.
    let width = gap.odd.delta() + gap.even.delta();
    assert!(
        ((gap.hi - gap.lo) - 2.0 * width).abs() <= 1e-12,
        "T6 interval must be twice the two sector widths: hi−lo = {:.6}, 2(δᵒ+δᵉ) = {:.6}",
        gap.hi - gap.lo,
        2.0 * width
    );
    assert!(
        (gap.gap - (gap.odd.value - gap.even.value)).abs() <= 1e-12,
        "the assembled gap must be θᵒ − θᵉ: {} vs {}",
        gap.gap,
        gap.odd.value - gap.even.value
    );
    // Non-resolution, pinned: the certificate is an enclosure, not a mass-gap
    // proof. `lo > 0` stays out of reach while δ ≫ gap (measured 7.438 vs
    // 0.0912). If this assertion ever fails the certificate has become sharp
    // enough to certify positivity, and the test must assert `gap.lo > 0`
    // instead of recording its absence.
    assert!(
        width > exact_gap,
        "the m=12 certificate must still be wider than the gap it encloses \
         (δ = {width:.4}, gap = {exact_gap:.4}); if it now resolves the gap, \
         assert lo > 0 — do not relax this line"
    );
    assert!(
        gap.lo <= 0.0,
        "with δ = {width:.4} against a gap of {exact_gap:.4} the T6 lower bound cannot be \
         positive; got lo = {:.6}",
        gap.lo
    );
}

#[test]
fn qcd_mass_gap_window_cannot_resolve_the_gauge_fixed_gap() {
    // The certified window from the SIRK sector solves encloses the exact
    // truncated spectral gap E₁ − E₀ across the truncation family (N ≤ 6 and
    // N ≤ 8), for g ∈ {1, 2} — but it is far too wide to *decide* anything:
    // it also contains the lattice-era value g²/2, so "the window contains the
    // gauge-fixed gap" cannot be the evidence that the gauge-fixed gap is not
    // g²/2. That departure is asserted from the dense diagonalization, which
    // is where it actually lives.
    for &g in &[1.0_f64, 2.0] {
        let h = qcd_ym_hamiltonian(g);
        let res_even = solve_sector(&h, &empty_vacuum(), 12);
        let res_odd = solve_sector(&h, &r_odd_start(), 12);
        let gap = certified_mass_gap(&res_even, &res_odd).expect("certified mass gap");
        let width = gap.odd.delta() + gap.even.delta();
        let g_sq_over_2 = 0.5 * g * g;
        for n in [6u32, 8] {
            let (_, _, exact_gap) = exact_gap(&h, n);
            println!(
                "g = {g}, N ≤ {n}: exact gap E₁−E₀ = {exact_gap:.6}, certified window \
                 [{:.6}, {:.6}] (width {width:.4}), lattice g²/2 = {g_sq_over_2}",
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
        // The window does NOT resolve the lattice value either — pinned so the
        // claim cannot be overstated. If this ever fails, the certificate has
        // sharpened enough to *separate* the two models, which is a stronger
        // and different result: assert the exclusion instead.
        assert!(
            gap.lo <= g_sq_over_2 && g_sq_over_2 <= gap.hi,
            "g={g}: the certified window [{:.6}, {:.6}] must still contain the \
             lattice value g²/2 = {g_sq_over_2} — with δ = {width:.4} it cannot \
             exclude it, so the window cannot evidence the departure",
            gap.lo,
            gap.hi
        );
        // The departure itself, from the dense diagonalization: real, and
        // falsifiable (gauge-fixed gap 0.091 vs 0.5 at g=1; 1.244 vs 2 at g=2).
        let (_, _, exact_gap_g) = exact_gap(&h, 8);
        assert!(
            (exact_gap_g - g_sq_over_2).abs() > 0.3,
            "g={g}: the gauge-fixed gap {exact_gap_g:.6} must depart from the \
             lattice value g²/2 = {g_sq_over_2} — the electric-lattice reading "
        );
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
