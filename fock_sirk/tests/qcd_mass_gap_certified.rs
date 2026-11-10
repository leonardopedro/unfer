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
//! 4. `the_certified_gap_lower_bound_needs_the_sector_ground` — SIRK driven on
//!    the *bounded* `P_N H P_N` (via `solve_forward_sirk_truncated`),
//!    cross-checked against the
//!    exact parity-resolved spectrum. Establishes two things the T6 assembly
//!    depends on: `lo > 0` is reachable and is a property of the finite basis
//!    (it holds at the gapless control `g = 0`), and `lo` can exceed the true
//!    gap when a sector solve has not resolved that sector's ground, with no
//!    term of the certificate able to see the difference (a residual measures
//!    "converged", not "lowest").
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
    SirkOpts, certified_mass_gap, certified_ritz_values, certify_spectrum,
    emit_gap_certificate_ndjson, solve_forward_sirk_truncated, solve_forward_sirk_with_opts,
    truncated_matvec,
};
use nalgebra::DMatrix;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, OuterState, QuantumState, qcd_ym_hamiltonian,
};
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

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

// ---------------------------------------------------------------------------
// Truncation control, and what the T6 step actually needs
// ---------------------------------------------------------------------------
//
// The campaign's `N` axis is a dense-diagonalization cross-check, and the
// certified gap it assembles comes from two *unbounded*-operator solves.
// `solve_forward_sirk_truncated` is the supported truncation-controlled entry
// point: it applies `P_N H P_N` on both sides, and since the start and every
// iterate live in `V_N`, `P_N` acts as the identity on the state SIRK subtracts,
// so the operator the loop steps with is exactly the bounded `H_N`. That makes
// the experiments below possible on a *bounded* operator, where an exact
// parity-resolved spectrum exists to compare against.

/// The QYM reflection `R|n₀,n₁,n₂,n₃⟩ = (−1)^(n₀+n₁)|n₁,n₀,n₂,n₃⟩`.
fn apply_reflection(s: &QuantumState) -> QuantumState {
    let mut out = QuantumState::zero();
    for (outer, amp) in &s.components {
        for (inner, mult) in &outer.bosonic {
            let n0 = inner.modes.get(&0).copied().unwrap_or(0);
            let n1 = inner.modes.get(&1).copied().unwrap_or(0);
            let mut flipped = InnerBosonicState::vacuum();
            for (mode, n) in &inner.modes {
                let target = match *mode {
                    0 => 1,
                    1 => 0,
                    other => other,
                };
                if *n > 0 {
                    flipped.modes.insert(target, *n);
                }
            }
            let phase = if (n0 + n1) * mult % 2 == 0 { 1.0 } else { -1.0 };
            out.components.insert(
                OuterState {
                    bosonic: BTreeMap::from([(flipped, *mult)]),
                    fermionic: BTreeSet::new(),
                },
                *amp * Complex64::new(phase, 0.0),
            );
        }
    }
    out
}

/// Ascending `(λ, R-parity, eigenvector)` of the truncated `H_N`, parity read
/// off exactly: `R` maps each (single-component, unit-amplitude) basis state to
/// `±` another, so `⟨ψ|Rψ⟩ = ±1` for a non-degenerate eigenvector.
fn truncated_spectrum(h: &Hamiltonian, n_max: u32) -> Vec<(f64, f64, QuantumState)> {
    let mut basis = Vec::new();
    for n0 in 0..=n_max {
        for n1 in 0..=n_max - n0 {
            for n2 in 0..=n_max - n0 - n1 {
                for n3 in 0..=n_max - n0 - n1 - n2 {
                    let mut occ = Vec::new();
                    for (mode, count) in [(0, n0), (1, n1), (2, n2), (3, n3)] {
                        if count > 0 {
                            occ.push((mode, count));
                        }
                    }
                    basis.push(fock_state(&occ));
                }
            }
        }
    }
    let n = basis.len();
    let mut mat = DMatrix::<Complex64>::zeros(n, n);
    for (j, s) in basis.iter().enumerate() {
        let hs = h.apply(s);
        for (i, t) in basis.iter().enumerate() {
            mat[(i, j)] = QuantumState::inner_product(t, &hs);
        }
    }
    let mut index = std::collections::HashMap::new();
    for (i, s) in basis.iter().enumerate() {
        index.insert(s.components.keys().next().unwrap().clone(), i);
    }
    let mut perm = Vec::with_capacity(n);
    for s in basis.iter() {
        let rs = apply_reflection(s);
        let (key, amp) = rs
            .components
            .iter()
            .next()
            .expect("R image is a basis state");
        perm.push((index[key], amp.re));
    }
    let eig = mat.symmetric_eigen();
    let mut rows = Vec::new();
    for idx in 0..n {
        let mut parity = Complex64::new(0.0, 0.0);
        for (j, &(sigma, sign)) in perm.iter().enumerate() {
            parity += eig.eigenvectors[(sigma, idx)].conj()
                * eig.eigenvectors[(j, idx)]
                * Complex64::new(sign, 0.0);
        }
        let mut vec = QuantumState::zero();
        for (j, s) in basis.iter().enumerate() {
            let c = eig.eigenvectors[(j, idx)];
            if c.norm_sqr() > 0.0 {
                vec.scale_and_add(s, c);
            }
        }
        rows.push((eig.eigenvalues[idx], parity.re.signum(), vec));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    rows
}

/// `P_N H P_N` must be self-adjoint as an operator — the property the
/// projector construction rests on, checked on states rather than through the
/// whitened matrix (which carries the collapsed-chain asymmetry noted in
/// `solve_truncated`).
fn assert_operator_self_adjoint(h: &Hamiltonian, n: u32, states: &[QuantumState]) {
    let compress = truncated_matvec(h, n);
    let applied: Vec<QuantumState> = states.iter().map(&compress).collect();
    for (j, a) in applied.iter().enumerate() {
        for (k, b) in states.iter().enumerate() {
            let lhs = QuantumState::inner_product(b, a);
            let rhs = QuantumState::inner_product(&states[j], &applied[k]).conj();
            assert!(
                (lhs - rhs).norm() < 1e-12,
                "P_N H P_N must be self-adjoint: ⟨{k}|H|{j}⟩ = {lhs} vs ⟨{j}|H|{k}⟩* = {rhs}"
            );
        }
    }
}

/// The `(n₀+n₁)`-parity block of `H_N` as an explicit real matrix — **valid only
/// at `g = 0`**.
///
/// At `g = 0` this block *is* a reflection sector: `R` commutes with
/// `H = ½π² + ½(A₀−A₁)²`, which is quadratic, so both the reflection parity and
/// the `(n₀+n₁)` parity are preserved, and the block is the operator itself
/// restricted to an invariant subspace — no leakage to project away.
///
/// **At `g > 0` that is false.** `B²` then carries the three-operator products
/// `A₀²A₁` and `A₀A₁²`, which change `(n₀+n₁)` parity, so a `(n₀+n₁)%2` filter is
/// neither an `R` sector nor an `H`-invariant subspace, and every eigenvalue read
/// off such a block would be meaningless — a `λ₂` read from it is not a witness,
/// and using it would turn a refused certificate into a wrong one (measured in
/// `mass_gap_exclusion_campaign.rs`: at `g = 2, N ≤ 6` the parity-filtered
/// "sector" certified `0.322254` against a true gap of `0.044450`).
///
/// So the coupling is an argument rather than an assumption. `g != 0` panics
/// instead of returning a silently wrong block, and the `H`-invariance of the
/// result is *measured* below, so the guard is a property of the built matrix
/// and not merely a caller convention.
fn truncated_parity_block(h: &Hamiltonian, n_max: u32, even: bool, g: f64) -> DMatrix<f64> {
    assert!(
        g == 0.0,
        "truncated_parity_block is only valid at g = 0: at g > 0 the (n₀+n₁) parity filter is \
         not a reflection sector (B²'s A₀²A₁ / A₀A₁² flip that parity), so the block is not \
         H-invariant and its spectrum is not a λ₂ witness — got g = {g}"
    );
    let mut basis = Vec::new();
    for n0 in 0..=n_max {
        for n1 in 0..=n_max - n0 {
            for n2 in 0..=n_max - n0 - n1 {
                for n3 in 0..=n_max - n0 - n1 - n2 {
                    let parity = (n0 + n1) % 2 == 0;
                    if parity != even {
                        continue;
                    }
                    let mut occ = Vec::new();
                    for (mode, count) in [(0, n0), (1, n1), (2, n2), (3, n3)] {
                        if count > 0 {
                            occ.push((mode, count));
                        }
                    }
                    basis.push(fock_state(&occ));
                }
            }
        }
    }
    let n = basis.len();
    let mut mat = DMatrix::<f64>::zeros(n, n);
    let mut leak = 0.0_f64;
    for (j, s) in basis.iter().enumerate() {
        let hs = h.apply(s);
        for (i, t) in basis.iter().enumerate() {
            mat[(i, j)] = QuantumState::inner_product(t, &hs).re;
        }
        // The invariance is measured, not assumed: every component of `H s` whose
        // `(n₀+n₁)` parity is the other one is weight the block above silently
        // dropped, and any such weight would make its eigenvalues the wrong
        // operator's. `dΓ` acts on the `mult`-fold symmetric power, so the
        // occupation's parity is `(n₀+n₁)·mult`.
        for (outer, amp) in hs.components.iter() {
            for (inner, mult) in outer.bosonic.iter() {
                let n0 = inner.modes.get(&0).copied().unwrap_or(0);
                let n1 = inner.modes.get(&1).copied().unwrap_or(0);
                let is_even = ((n0 + n1) * mult) % 2 == 0;
                if is_even != even {
                    leak += amp.norm_sqr() * (*mult as f64);
                }
            }
        }
    }
    assert!(
        leak < 1e-10,
        "the (n₀+n₁) parity block must be H-invariant at g = 0: measured ‖(1−P) H P‖_F = \
         {leak:e} — a nonzero leak means this is not a reflection sector, so its λ₂ is not a \
         witness"
    );
    mat
}

/// One SIRK solve of the bounded `P_N H P_N` at Krylov order `m`.
fn solve_truncated(
    h: &Hamiltonian,
    start: &QuantumState,
    m: usize,
    n: u32,
) -> fock_sirk::ForwardSirkResult {
    let res = solve_forward_sirk_truncated(h, start, &shifts(m), &best_device(), None, &opts(), n)
        .expect("truncated solve must complete");
    // NOTE: `assert_hermitian` is deliberately not applied here. `h_proj` is
    // assembled from the step identity `⟨ŵ_j|H|ŵ_k⟩ = (σ_{k+1}/σ_k)G_{j,k+1} +
    // z_k G_{j,k}` rather than from explicit `H` applications, so when the
    // truncated chain collapses (rank 10 of `m+1 = 15` at `m = 14`, `g = 0`)
    // that matrix departs from Hermitian by a measured `2.5e-4`. That is a
    // property of the collapsed-chain assembly, not of `P_N H P_N`:
    // `assert_operator_self_adjoint` checks the operator itself, exactly, on
    // the states the test uses.
    res
}

/// The two facts the T6 assembly needs, measured on a bounded operator.
///
/// (1) **The knob is the truncation, not the coupling.** With `P_N H P_N` the
///     T6 expression is *positive at `g = 0`* — `lo = +0.189913` against a true
///     gap of `0.189928`. The gapless control certifies positivity because a
///     finite occupation basis has a discrete spectrum: `lo > 0` is therefore a
///     statement about `H_N`, never about the theory. That is the answer to
///     "find a coupling large enough" — no coupling is needed, and `g = 0`
///     already does it, which is what shows the lever was the wrong one.
///
/// (2) **`lo > 0` can be wrong.** T6 needs the odd solve to have resolved the
///     odd sector's *ground*, and no term of the certificate witnesses that.
///     Start the odd chain orthogonal to that ground (overlap `7.5e-16`): the
///     whole space stays orthogonal to it, the lowest Ritz value converges to
///     the odd *second* level with a residual `1.2e-7`, and the assembly
///     returns `lo = +0.526380` against a true gap of `0.189928`. Preconditions
///     as documented (lowest Ritz value per sector, disjoint sectors, residual
///     widths) are all satisfied, so the excess is not a bug in the arithmetic:
///     the residual measures "converged", not "lowest".
#[test]
fn the_certified_gap_lower_bound_needs_the_sector_ground() {
    let g = 0.0;
    let n = 6u32;
    let h = qcd_ym_hamiltonian(g);
    let spectrum = truncated_spectrum(&h, n);

    let lambda_even = spectrum
        .iter()
        .find(|(_, parity, _)| *parity > 0.0)
        .map(|(value, _, _)| *value)
        .expect("an R-even level");
    let odd: Vec<&(f64, f64, QuantumState)> = spectrum
        .iter()
        .filter(|(_, parity, _)| *parity < 0.0)
        .collect();
    let lambda_odd = odd[0].0;
    let second_odd = odd[1].0;
    let exact_gap = lambda_odd - lambda_even;
    assert!(
        exact_gap > 0.1,
        "the truncated theory is gapped at g = 0 by construction (finite basis): {exact_gap}"
    );

    // The projector is applied on both sides; check the bounded operator is
    // self-adjoint before trusting a Krylov spectrum of it.
    assert_operator_self_adjoint(&h, n, &[empty_vacuum(), r_odd_start()]);

    // (1) The in-sector start: the hypothesis holds and T6 is a bound.
    let res_even = solve_truncated(&h, &empty_vacuum(), 14, n);
    let res_odd = solve_truncated(&h, &r_odd_start(), 14, n);
    let ce = certified_ritz_values(&res_even);
    let co = certified_ritz_values(&res_odd);
    let in_sector = certified_mass_gap(&res_even, &res_odd).expect("assembly");
    assert!(
        (co[0].value - lambda_odd).abs() <= co[0].delta(),
        "the in-sector odd solve must isolate the odd ground: θᵒ₀ = {}, λ_odd = {lambda_odd}, δᵒ = {}",
        co[0].value,
        co[0].delta()
    );
    // Tolerance: the collapsed-chain `h_proj` asymmetry measured above (2.5e-4)
    // is the accuracy of θ on this solve, so a bound check at 1e-3 is the
    // honest resolution of this comparison (the expression is in fact tight to
    // 1.5e-5 here).
    assert!(
        in_sector.lo <= exact_gap + 1e-3,
        "with the sector ground resolved the T6 expression is a bound: lo = {} must not \
         exceed the exact gap {exact_gap}",
        in_sector.lo
    );
    assert!(
        in_sector.lo > 0.0,
        "on the bounded H_N the T6 expression is positive at the GAPLESS control g = 0 \
         (lo = {}); positivity here is a property of the finite basis, not of the theory",
        in_sector.lo
    );

    // (2) A start orthogonal to the odd ground: still "converged", wrong level.
    let overlap = QuantumState::inner_product(&odd[0].2, &odd[1].2);
    let mut start = odd[1].2.clone();
    start.scale_and_add(&odd[0].2, -overlap);
    let norm = start.norm();
    start.scale_and_add(&start.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));
    assert!(
        QuantumState::inner_product(&odd[0].2, &start).norm() < 1e-12,
        "the start must be orthogonal to the odd ground"
    );

    let res_stuck = solve_truncated(&h, &start, 14, n);
    let cs = certified_ritz_values(&res_stuck);
    let stuck = certified_mass_gap(&res_even, &res_stuck).expect("assembly");
    assert!(
        cs[0].delta() < 1e-4,
        "the stuck solve is CONVERGED in the residual sense (δᵒ = {})",
        cs[0].delta()
    );
    assert!(
        (cs[0].value - second_odd).abs() < 1e-4,
        "it converges to the odd SECOND level: θᵒ₀ = {}, λ₂(odd) = {second_odd}",
        cs[0].value
    );
    assert!(
        (second_odd - lambda_odd).abs() > 0.3,
        "the level it found must be far above the ground: λ₂ = {second_odd}, λ₁ = {lambda_odd}"
    );
    assert!(
        stuck.lo > exact_gap + 0.3,
        "the T6 expression then EXCEEDS the true gap: lo = {} vs gap = {exact_gap} — `lo > 0` \
         is not a proof, and no residual term can detect this",
        stuck.lo
    );

    // (3) With an exclusion witness the bound becomes valid: Kato–Temple.
    //
    // The witness is `λ₂` for the odd sector, obtained from a certified
    // spectrum of the explicit odd block: every computed eigenpair's Parlett
    // interval is grouped into disjoint clusters, and since an n x n matrix has
    // n eigenvalues and every interval holds at least one, each cluster holds
    // exactly its own count. That turns the second cluster's lower edge into a
    // rigorous λ₂ bound — the thing a residual cannot produce.
    let odd_block = truncated_parity_block(&h, n, false, g);
    let certified_spectrum = certify_spectrum(&odd_block).expect("the odd block is certifiable");
    let beta = certified_spectrum
        .second_eigenvalue_lower_bound()
        .expect("λ₂ lower bound from the certificate");
    assert!(
        beta > lambda_odd,
        "the λ₂ witness must sit above the odd ground: β = {beta}, λ₁ = {lambda_odd}"
    );

    let temple = fock_sirk::mass_gap_spec::certified_gap_lower_bound_with_exclusion(
        co[0].value,
        co[0].residual,
        beta,
        ce[0].value,
    )
    .expect("Temple's hypothesis must hold for the in-sector solve");
    assert!(temple > 0.0, "the certified gap must be positive: {temple}");
    assert!(
        temple <= exact_gap + 1e-9,
        "Temple's bound must stay below the exact gap: {temple} vs {exact_gap}"
    );
    assert!(
        temple >= in_sector.lo,
        "r²/(β−θ) must sharpen the Parlett-width assembly: Temple {temple} vs lo {}",
        in_sector.lo
    );
    // And the same witness refuses the stuck solve: its θ sits at λ₂, so
    // θ + r < β ≤ λ₂ fails and Temple's inequality does not apply. Without the
    // guard this is exactly the case that would return a "bound" above λ₁.
    assert!(
        fock_sirk::mass_gap_spec::certified_gap_lower_bound_with_exclusion(
            cs[0].value,
            cs[0].residual,
            beta,
            ce[0].value
        )
        .is_none(),
        "the exclusion form must decline when the residual ball reaches λ₂"
    );
    eprintln!(
        "certified gap (Temple, g=0, N<=6, m=14): {temple:.9} vs exact {exact_gap:.9}; \
         Parlett-width lo {:.9}; θᵒ₀ = {:.9} rᵒ = {:.3e} β = {beta:.9}",
        in_sector.lo, co[0].value, co[0].residual
    );
}

/// The `g > 0` guard on `truncated_parity_block` is real: a caller that reaches
/// for the parity block as if it were a reflection sector at nonzero coupling is
/// refused rather than handed a block whose spectrum is not a `λ₂` witness.
#[test]
#[should_panic(expected = "only valid at g = 0")]
fn the_parity_block_refuses_a_nonzero_coupling() {
    let g = 1.0;
    let h = qcd_ym_hamiltonian(g);
    let _ = truncated_parity_block(&h, 6, false, g);
}
