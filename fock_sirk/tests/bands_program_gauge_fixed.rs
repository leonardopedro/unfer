//! CERTIFIED NUMERICS for the program-sector gauge-fixed Hamiltonians.
//!
//! Two tiers are kept apart in every test (S43):
//!
//! * **envelope** — Theorem 4.1 (`hashimoto_support::BandParams`), the a-priori
//!   SIRK-error ceiling. Rigorous, but on these models its half-widths are
//!   1.1e4 … 5.5e15, so nothing in reach can fall outside it. It is *printed*
//!   in every test as the ceiling and never carries an assertion.
//! * **sharp** — the Rayleigh–Ritz residual certificate
//!   (`fock_sirk::certified_ritz_values`): `[θ − δ, θ + δ]`, `δ = ‖Hψ − θψ‖ +
//!   c(n)u‖Ĝ‖ + h_O`, every term computed cancellation-free from the stored
//!   Gram. By Parlett's bound each interval contains an eigenvalue of `H`, so
//!   containment and separation are *falsifiable* at this tier.
//!
//! 1. `qg_scalaron_gap_certified_contains_analytic` — the k→0 scalaron gap:
//!    analytic m(α) inside the sharp gap interval at every depth (widths
//!    2.7e-15 and 7.3e-8), and successive intervals intersect.
//! 2. `qym_g0_bottom_is_the_proven_square_sum_not_a_rung` — abelian QYM at
//!    `g = 0` has **no** lowest eigenvalue: `H + 2 = ½X² + ½π₂² + ½π₃²` exactly
//!    (checked on 81 product states, deviation 0), so the bottom is `−2`, not
//!    attained, and the Ritz values descend towards it (−1.354, −1.546, −1.649
//!    at m = 6, 9, 12) while their residual widths stay O(1).
//! 3. `qym_gpairs_symmetry_certified_overlap` — the A¹→−A¹ symmetry, as an
//!    **exact operator identity** `H(−g) = Π H(g) Π` at the operator level
//!    (deviation 0), with the sharp spectra at ±g overlapping to the
//!    certificate width as the spectral corollary.
//! 4. `ns_decay_amplitude_certified_interval` — NS laminar amplitude after one
//!    e-folding: measured solver-accuracy bar (5.45e-2 at m=10, 2.31e-2 at
//!    m=12, 9.95e-3 at m=14 — there is no sharp *state*-error certificate, so
//!    this one is an accuracy bar, printed with its measured values).
//! 5. `qg_graviton_scalaron_certified_disjoint` — same-k graviton vs scalaron
//!    levels: DISJOINT sharp intervals — the massive/massless speed split
//!    established by the residual bound alone.
//! 6. `qed_casimir_cavity_band_dispersion` — Casimir cavity ω_n = nπ/d: each
//!    exact level inside the sharp interval of its nearest Ritz value, plus
//!    the 1e-6 reproduction of `nπ/d`.
//!
//! ## Measured runtimes (M4, 2026-10-07)
//!
//! Debug (`cargo test`, dev profile): **6 passed / 2148.28s**; release
//! (`cargo test -p fock_sirk --release`, the CI profile): **6 passed /
//! 3.57s** (~602×). The debug numbers are recorded so the cost of running the
//! default profile is known before anyone starts one — ~36 min in debug.
//!
//! ## Audit note (2026-11-10, revised)
//!
//! The first pass of this audit found that tests 1–4 asserted containment /
//! overlap against the Theorem 4.1 envelope (half-widths **1.3e5** QYM g=0,
//! **5.5e15** QYM ±g, **2.0e4** QG scalaron gap, **1.1e4** NS decay, **1.1e4**
//! Casimir), which cannot fail. All of them now assert on the sharp residual
//! tier, from `ForwardSirkResult::ritz_abs_residuals` /
//! `certified_ritz_values` — no new accessor was needed; the earlier note that
//! `‖Hψ‖` had to be exposed per Ritz vector was wrong, because the residual is
//! available cancellation-free from the Gram matrix.
//!
//! Two claims did **not** survive the move to the sharp tier and are corrected
//! here rather than asserted:
//!
//! * "certified intervals nest with depth" (QG scalaron): the sharp widths are
//!   2.7e-15 at m=6 and 7.3e-8 at m=9 — the zero rung's residual grows with
//!   depth, so the m=9 interval is *wider*. Only intersection is asserted.
//! * the QYM `g = 0` "three lowest rungs" (2026-11-10, second pass): they are
//!   not eigenvalues. `H + 2` is a sum of three squares of unbounded
//!   self-adjoint operators, so its spectrum is `[0, ∞)` and the bottom is a
//!   limit point; the widths 1.104→0.929→0.819 shrink but can never resolve a
//!   level, and "the same rung at every depth" is not a well-defined object.
//!   That test is replaced by `qym_g0_bottom_is_the_proven_square_sum_not_a_rung`,
//!   which proves the bottom symbolically and asserts the descent.
//! * Casimir sharp containment: the residual certificate bounds `|θ − λ|` for
//!   the *truncated* operator the solve applied. Several rungs are exact
//!   eigenpairs of it (residual 0, δ = 1.0e-13) while their Ritz value still
//!   differs from `nπ/d` by 3e-9 — that residue is the model truncation, so
//!   Parlett's bound cannot cover the analytic level. What is asserted is the
//!   reproduction distance (2.99e-9 at m=6, 1.40e-10 at m=8, against a 1e-6
//!   tolerance) with the certificate width printed beside it.
//! * "the ±g intervals overlap, certifying the symmetry" (QYM): at the sharp
//!   tier the widths are 1.6–9 while the ±g rung differences are 0.02–0.3, i.e.
//!   the certificate is 10²–10³× too wide to resolve the asymmetry it was
//!   meant to certify. The symmetry is asserted exactly instead, as the
//!   operator identity `H(−g) = Π H(g) Π`, and the overlap is kept only as its
//!   numerical corollary with the width printed alongside.

mod hashimoto_support;

use fock_sirk::device::best_device;
use fock_sirk::{
    Certificate, ForwardSirkResult, SirkOpts, certified_ritz_values, solve_forward_sirk_with_opts,
};
use hashimoto_support::{BandParams, print_certified, sirk_paper_shifts};
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, QuantumState, ns_eulerian_fiber, qcd_ym_hamiltonian,
    qed_cavity_frequencies, qg_free_graviton, qg_starobinsky_scalaron_field,
};
use num_complex::Complex64;

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(400_000),
        brst_tol: 1e-10,
        // QYM g=0 quadratic reach can blow component counts; adaptive is
        // licensed inert-at-budget by guard_justification_study Study C.
        adaptive: true,
        unit_norm_steps: true,
    }
}

fn empty_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// One paper-shift SIRK solve, returning the full result (Ritz values,
/// residuals and the sequence needed for reconstruction).
trait Solve {
    fn solve(&self, psi: &QuantumState, bp: &BandParams, m: usize) -> ForwardSirkResult;
}

impl Solve for Hamiltonian {
    fn solve(&self, psi: &QuantumState, bp: &BandParams, m: usize) -> ForwardSirkResult {
        let shifts: Vec<Complex64> = sirk_paper_shifts(m, bp.big_n, bp.h)
            .iter()
            .map(|g| Complex64::new(0.0, g.re / bp.t))
            .collect();
        solve_forward_sirk_with_opts(self, psi, &shifts, &best_device(), None, &opts())
            .expect("paper-shift SIRK solve")
    }
}

/// The a-priori Theorem 4.1 ceiling for a solve, from twice the largest |θ|.
/// Reported only — it is too loose on these models to assert anything with.
fn envelope_ceiling(res: &ForwardSirkResult, bp: &BandParams, m: usize, v_norm: f64) -> f64 {
    let lam = res
        .ritz_values()
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        * 2.0
        + 1.0;
    bp.band(lam, m, v_norm, 0, 240, 40).hi
}

fn print_cert(label: &str, c: &Certificate) {
    print_certified(label, c.value, c.lo, c.hi);
}

#[test]
fn qg_scalaron_gap_certified_contains_analytic() {
    let mass = 0.8_f64;
    let h = qg_starobinsky_scalaron_field(&[0.0], mass);
    let bp = BandParams {
        big_n: 8.0,
        h: 0.5,
        t: 0.8,
    };
    let analytic_gap = mass; // k→0 gap IS the scalaron mass

    let mut psi0 = empty_vacuum();
    psi0.scale_and_add(
        &empty_vacuum().apply(&Operator::InnerBosonCreate(0)),
        Complex64::new(1.0, 0.0),
    );
    let v_norm = psi0.norm();

    println!("— QG scalaron k→0 gap, sharp residual tier: analytic {analytic_gap:.6}");
    let mut prev_interval: Option<(f64, f64)> = None;
    for m in [6_usize, 9] {
        let res = h.solve(&psi0, &bp, m);
        let c = certified_ritz_values(&res);
        assert!(c.len() >= 2, "two rungs needed, got {}", c.len());
        // The gap of the *exact* spectrum is bracketed by propagating both
        // rung enclosures through the difference.
        let gap_meas = c[1].value - c[0].value;
        let width = c[0].delta() + c[1].delta();
        let (lo, hi) = (gap_meas - width, gap_meas + width);
        print_certified(&format!("gap (m={m}) [res-certified]"), gap_meas, lo, hi);
        println!(
            "    rung widths {:.3e} + {:.3e}; Theorem 4.1 ceiling {:.3e}",
            c[0].delta(),
            c[1].delta(),
            envelope_ceiling(&res, &bp, m, v_norm)
        );
        assert!(
            lo <= analytic_gap && analytic_gap <= hi,
            "analytic gap {analytic_gap} outside the residual-certified interval \
             [{lo:.9},{hi:.9}] at m={m} (width {width:.3e})"
        );
        if let Some((plo, phi)) = prev_interval {
            // Intersection, not nesting: the sharp width is NOT monotone in m
            // (7.3e-8 at m=9 vs 2.7e-15 at m=6), so "nesting" would be false.
            assert!(
                lo <= phi && plo <= hi,
                "successive sharp intervals must intersect: [{plo:.9},{phi:.9}] vs [{lo:.9},{hi:.9}]"
            );
        }
        prev_interval = Some((lo, hi));
    }
}

#[test]
fn qed_casimir_cavity_band_dispersion() {
    let d = 1.7_f64;
    let omegas = qed_cavity_frequencies(d, 4); // ω_n = nπ/d, n=1..=4
    let ham = nested_fock_algebra::qed_free_photon(&omegas);
    let bp = BandParams {
        big_n: 8.0,
        h: 0.5,
        t: 0.8,
    };

    let mut psi0 = empty_vacuum();
    for i in 0..omegas.len() as u32 {
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(i, 1);
        psi0.scale_and_add(
            &QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner)),
            Complex64::new(1.0 + i as f64 * 0.3, 0.0),
        );
    }

    println!("— QED Casimir cavity dispersion (d={d}): ω_n = nπ/d, sharp tier");
    println!("  m   worst|ω−θ|   worst certified δ   Theorem 4.1 ceiling");
    for m in [6_usize, 8] {
        let res = ham.solve(&psi0, &bp, m);
        let c = certified_ritz_values(&res);
        let mut worst = 0.0_f64;
        let mut worst_delta = 0.0_f64;
        for &w in &omegas {
            let nearest = c
                .iter()
                .min_by(|a, b| {
                    (a.value - w)
                        .abs()
                        .partial_cmp(&(b.value - w).abs())
                        .unwrap()
                })
                .expect("non-empty Ritz set");
            let dist = (nearest.value - w).abs();
            worst = worst.max(dist);
            worst_delta = worst_delta.max(nearest.delta());
            // NOT asserted: `dist <= nearest.delta()`. The residual
            // certificate is a statement about the *truncated* operator that
            // the solve actually applied, and several Casimir rungs are exact
            // eigenpairs of it (residual 0, δ = 1.0e-13) while their Ritz
            // value still differs from nπ/d by 3e-9 — that residue is the
            // model's truncation, not the solver's error. Parlett's bound
            // cannot cover it, so the assertion below is the reproduction
            // distance (measured 1.8e-9 against a 1e-6 tolerance) and the
            // certificate width is printed beside it.
            assert!(
                dist < 1e-6,
                "Casimir level {w:.9} must be reproduced by the Ritz values to 1e-6; \
                 nearest is {:.9} (distance {dist:.3e}) at m={m}",
                nearest.value
            );
        }
        println!(
            "  {m:<3} {worst:.3e}    {worst_delta:.3e}          {:.3e}",
            envelope_ceiling(&res, &bp, m, psi0.norm())
        );
    }
}

/// The sum-of-squares form of the abelian `g = 0` Hamiltonian:
/// `H + 2 = ½X² + ½π₂² + ½π₃²`, with `X = A₀ − A₁` and `π_m = i(a†_m − a_m)`,
/// built in the model's own word convention (`vec![left, right]` is
/// `left·right`, the right factor applied first).
fn g0_square_sum() -> Hamiltonian {
    use Operator::{InnerBosonAnnihilate as A, InnerBosonCreate as C};
    // X = c₀ + a₀ − c₁ − a₁.
    let x = [(1.0, C(0u32)), (1.0, A(0)), (-1.0, C(1)), (-1.0, A(1))];
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for (ci, li) in &x {
        for (cj, rj) in &x {
            terms.push((
                Complex64::new(0.5 * ci * cj, 0.0),
                vec![li.clone(), rj.clone()],
            ));
        }
    }
    // ½π² = ½(−c² + ca + ac − a²) — the `ac` word is what carries the `+½`
    // zero-point, which is why no explicit constant appears.
    for m in [2u32, 3] {
        terms.push((Complex64::new(-0.5, 0.0), vec![C(m), C(m)]));
        terms.push((Complex64::new(0.5, 0.0), vec![C(m), A(m)]));
        terms.push((Complex64::new(0.5, 0.0), vec![A(m), C(m)]));
        terms.push((Complex64::new(-0.5, 0.0), vec![A(m), A(m)]));
    }
    Hamiltonian { terms }
}

/// The abelian `g = 0` spectrum has **no lowest eigenvalue**: its bottom is
/// `−2`, proved by `H + 2 = ½X² + ½π₂² + ½π₃²`.
///
/// This test replaced a `qym_g0_spectrum_certified_nesting` claim that asserted
/// three "rungs", their overlap across depths and their shrinking residual
/// widths. Those rungs are not eigenvalues of anything: `½X²` and `½π_m²` are
/// squares of unbounded self-adjoint operators with spectrum `[0, ∞)`, so the
/// spectrum of `H + 2` is `[0, ∞)` with the bottom *unattained*, and a Ritz
/// value can approach it forever without any residual going to zero. The
/// residual certificates at this tier were therefore intervals containing *a*
/// spectral point (true, and unfalsifiable on a dense spectrum) rather than
/// brackets on a level.
///
/// What is asserted instead is stronger and can fail:
/// * the square-sum identity itself, on every product state with occupations
///   `≤ 2` per mode — a *complete* check for a degree-2 polynomial, since the
///   coefficients of every degree-≤2 word are pinned by those matrix elements;
/// * `½X² + ½π₂² + ½π₃²` has no zero eigenvector on that family (measured
///   `min ‖Sψ‖ = 2.0`), so the bottom `−2` is not attained — `X` and each `π_m`
///   are injective (a nonzero Hermitian degree-1 form has no `L²` kernel);
/// * every certified Ritz value stays above `−2`, and the lowest one descends
///   monotonically towards it with depth (measured `−1.354`, `−1.546`, `−1.649`
///   at `m = 6, 9, 12`).
#[test]
fn qym_g0_bottom_is_the_proven_square_sum_not_a_rung() {
    let h = qcd_ym_hamiltonian(0.0);
    let sq = g0_square_sum();

    // ── The identity, on a spanning family ──
    let mut states = Vec::new();
    for n0 in 0..=2u32 {
        for n1 in 0..=2u32 {
            for n2 in 0..=2u32 {
                for n3 in 0..=2u32 {
                    let mut inner = InnerBosonicState::vacuum();
                    for (mode, n) in [(0u32, n0), (1, n1), (2, n2), (3, n3)] {
                        if n > 0 {
                            inner.modes.insert(mode, n);
                        }
                    }
                    states.push(QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner)));
                }
            }
        }
    }
    println!("— QYM abelian g=0: the bottom is −2, from H + 2 = ½X² + ½π₂² + ½π₃²");
    let mut worst_identity = 0.0_f64;
    let mut min_s = f64::INFINITY;
    for psi in &states {
        let mut lhs = h.apply(psi);
        lhs.scale_and_add(psi, Complex64::new(2.0, 0.0));
        let rhs = sq.apply(psi);
        let mut diff = lhs;
        diff.scale_and_add(&rhs, Complex64::new(-1.0, 0.0));
        worst_identity = worst_identity.max(diff.norm());
        min_s = min_s.min(QuantumState::inner_product(psi, &sq.apply(psi)).re);
    }
    println!(
        "    max ‖(H+2)ψ − Sψ‖ = {worst_identity:.3e} over {} product states; \
         min ‖Sψ‖ = {:.3e}",
        states.len(),
        min_s
    );
    assert!(
        worst_identity < 1e-12,
        "H + 2 must equal ½X² + ½π₂² + ½π₃² exactly; worst deviation {worst_identity:.3e}"
    );
    assert!(
        min_s > 0.0,
        "the bottom of ½X² + ½π₂² + ½π₃² must not be attained on the family, got {min_s:.3e}"
    );

    // ── The numerical side: approach, never an eigenvalue ──
    let bp = BandParams {
        big_n: 8.0,
        h: 0.5,
        t: 0.8,
    };
    let mut psi0 = empty_vacuum();
    psi0.scale_and_add(
        &empty_vacuum().apply(&Operator::InnerBosonCreate(0)),
        Complex64::new(1.0, 0.0),
    );
    let mut prev_theta0: Option<f64> = None;
    for m in [6_usize, 9, 12] {
        let res = h.solve(&psi0, &bp, m);
        let c = certified_ritz_values(&res);
        assert!(
            !c.is_empty(),
            "a non-empty certified set is needed at m={m}"
        );
        for x in c.iter() {
            assert!(
                x.value >= -2.0 - 1e-6,
                "every Ritz value must lie above the proven bottom −2: θ = {:.9} at m={m}",
                x.value
            );
        }
        let theta0 = c[0].value;
        println!(
            "  m={m:<3} θ₀ = {theta0:.9}   δ₀ = {:.3e}   next θ = {:.6}   \
             Theorem 4.1 ceiling {:.3e}",
            c[0].delta(),
            c.get(1).map(|x| x.value).unwrap_or(f64::NAN),
            envelope_ceiling(&res, &bp, m, psi0.norm())
        );
        if let Some(pt) = prev_theta0 {
            assert!(
                theta0 < pt,
                "the lowest Ritz value must descend towards the bottom: {pt:.9} (m−3) vs \
                 {theta0:.9}"
            );
        }
        prev_theta0 = Some(theta0);
    }
    println!(
        "    bottom −2 is a limit point, not an eigenvalue: the δ₀ column stays O(1) \
         because there is no level to converge to"
    );
}

/// `Π = (−1)^{n₀+n₁}` — the A¹ → −A¹ parity on the two magnetic modes of
/// `qcd_ym_hamiltonian` (models 0, 1). Diagonal in the Fock basis.
fn parity(s: &QuantumState) -> QuantumState {
    let mut out = QuantumState::zero();
    for (state, amp) in &s.components {
        let mut odd = 0u32;
        for (inner, count) in &state.bosonic {
            let n = inner.modes.get(&0).copied().unwrap_or(0)
                + inner.modes.get(&1).copied().unwrap_or(0);
            odd += n * count;
        }
        let sign = if odd % 2 == 0 { 1.0 } else { -1.0 };
        out.components.insert(state.clone(), *amp * sign);
    }
    out
}

#[test]
fn qym_gpairs_symmetry_certified_overlap() {
    let g_abs = 0.35_f64;
    let bp = BandParams {
        big_n: 8.0,
        h: 0.5,
        t: 0.8,
    };
    let hp = qcd_ym_hamiltonian(g_abs);
    let hm = qcd_ym_hamiltonian(-g_abs);

    // ── The symmetry itself, as an exact operator identity ──
    // B = (A₀ − A₁) + (g/2)A₀A₁, and ΠA_iΠ = −A_i on the magnetic modes, so
    // Π H(g) Π = H(−g) with no approximation: assert it on a spanning set of
    // product states (occupations 0..=2 on the magnetic and momentum modes).
    println!("— QYM A¹→−A¹ symmetry: exact operator identity H(−g) = Π H(g) Π");
    let mut worst = 0.0_f64;
    for n0 in 0..=2u32 {
        for n1 in 0..=2u32 {
            for n2 in 0..=1u32 {
                let mut inner = InnerBosonicState::vacuum();
                for (m, n) in [(0u32, n0), (1u32, n1), (2u32, n2)] {
                    if n > 0 {
                        inner.modes.insert(m, n);
                    }
                }
                let psi = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner));
                // H(−g) Π ψ  vs  Π H(g) ψ
                let lhs = hm.apply(&parity(&psi));
                let rhs = parity(&hp.apply(&psi));
                let mut diff = lhs.clone();
                diff.scale_and_add(&rhs, Complex64::new(-1.0, 0.0));
                worst = worst.max(diff.norm());
            }
        }
    }
    println!("    max ‖H(−g)Πψ − ΠH(g)ψ‖ = {worst:.3e} over 18 product states");
    assert!(
        worst < 1e-12,
        "the ±g Hamiltonians must be unitarily equivalent: max deviation {worst:.3e}"
    );

    // ── Spectral corollary at the sharp tier ──
    // Kept, but with its honest strength: the certificate width is O(1) and the
    // ±g rung differences are O(0.1), so this only says the spectra agree to
    // within the residual width. The exact identity above is the real claim.
    let mut psi0 = empty_vacuum();
    psi0.scale_and_add(
        &empty_vacuum().apply(&Operator::InnerBosonCreate(0)),
        Complex64::new(1.0, 0.0),
    );
    println!("— QYM ±g equal spectrum, sharp residual tier (width printed)");
    let rp = hp.solve(&psi0, &bp, 9);
    let rm = hm.solve(&psi0, &bp, 9);
    let cp = certified_ritz_values(&rp);
    let cm = certified_ritz_values(&rm);
    for i in 0..3 {
        let (p, q) = (cp[i], cm[i]);
        print_cert(&format!("θ_{i}(+g)"), &p);
        print_cert(&format!("θ_{i}(−g)"), &q);
        println!(
            "    |θ₊−θ₋| = {:.3e} vs certificate width {:.3e} → {}",
            (p.value - q.value).abs(),
            p.delta() + q.delta(),
            if (p.value - q.value).abs() <= p.delta() + q.delta() {
                "resolved within the certificate"
            } else {
                "NOT resolved"
            }
        );
        assert!(
            p.lo <= q.hi && q.lo <= p.hi,
            "rung {i}: ±g sharp intervals must overlap — [{:.6},{:.6}] vs [{:.6},{:.6}]",
            p.lo,
            p.hi,
            q.lo,
            q.hi
        );
    }
    println!(
        "    Theorem 4.1 ceilings: +g {:.3e}, −g {:.3e} (reported, not asserted)",
        envelope_ceiling(&rp, &bp, 9, psi0.norm()),
        envelope_ceiling(&rm, &bp, 9, psi0.norm())
    );
}

#[test]
fn ns_decay_amplitude_certified_interval() {
    // Nondimensional rate ≡ 1 (S40 stiffness note).
    let h = ns_eulerian_fiber(
        &[[-0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        &[0.0, 0.0, 0.0],
    );
    let u_op = Hamiltonian {
        terms: vec![
            (
                Complex64::new(1.0, 0.0),
                vec![Operator::InnerBosonCreate(0)],
            ),
            (
                Complex64::new(1.0, 0.0),
                vec![Operator::InnerBosonAnnihilate(0)],
            ),
        ],
    };
    let bp = BandParams {
        big_n: 8.0,
        h: 0.5,
        t: 1.0,
    };

    let mut psi0 = empty_vacuum();
    psi0.scale_and_add(
        &empty_vacuum().apply(&Operator::InnerBosonCreate(0)),
        Complex64::new(1.0, 0.0),
    );
    let u0 = QuantumState::inner_product(&psi0, &u_op.apply(&psi0)).re;
    let analytic = u0 * (-1.0_f64).exp();

    println!("— NS laminar amplitude after one e-folding: analytic {analytic:.6}");
    // There is no sharp *state*-error certificate (the Rayleigh–Ritz bound is
    // about eigenvalues, not about the propagated vector), so this test is an
    // accuracy bar with its measured values recorded — 5.453e-2 at m=10,
    // 2.313e-2 at m=12, 9.954e-3 at m=14 — and the Theorem 4.1 envelope is
    // printed as the (loose) ceiling. The 6e-2 tolerance is what the
    // measurement supports; it is not a certificate. `m < N/h = 16` is the
    // paper ladder's admissibility bound (γ_j = N − h j > 0).
    //
    // A defect (leakage) bound was tried instead of the bar and **measured to be
    // vacuous**: `‖ψ(t) − v(t)‖ ≤ η|t|‖ψ₀‖` with `η = ‖(1−WW†)HW‖_F` gives
    // `η = 1.023e1`, `‖ψ₀‖ = 1.414` at m=10, i.e. a bound of `1.4e1` against a
    // measured `|Δ| = 5.45e-2`. That is structural, not an implementation
    // shortfall: this generator is a dilation/squeeze (continuous spectrum), so
    // no Krylov span is nearly invariant — every basis vector leaks O(1) — and
    // the 3-digit accuracy of the amplitude comes from cancellation between the
    // leaks, which no invariance-based bound can see. Recorded here so nobody
    // re-derives it.
    let bar = 6e-2;
    let mut prev_err: Option<f64> = None;
    for m in [10_usize, 12, 14] {
        let res = h.solve(&psi0, &bp, m);
        let coeffs = res.time_evolve(bp.t);
        let psi_t = res.reconstruct(&coeffs);
        let ut = QuantumState::inner_product(&psi_t, &u_op.apply(&psi_t)).re;
        println!(
            "  ⟨u⟩(t=e-fold, m={m}) = {ut:.9}   |Δ from analytic| = {:.3e}   \
             Theorem 4.1 ceiling {:.3e}",
            (ut - analytic).abs(),
            envelope_ceiling(&res, &bp, m, psi0.norm())
        );
        let err = (ut - analytic).abs();
        assert!(
            err <= bar,
            "NS decay amplitude at m={m} is {ut:.9}, {err:.3e} from the analytic \
             {analytic:.9} — outside the measured accuracy bar {bar:.1e}"
        );
        if let Some(pe) = prev_err {
            assert!(
                err < pe,
                "the laminar error must shrink with depth: {pe:.3e} (m−2) vs {err:.3e}"
            );
        }
        prev_err = Some(err);
    }
}

#[test]
fn qg_graviton_scalaron_certified_disjoint() {
    let ks = [0.9_f64];
    let m_scal = 0.55_f64;
    let hg = qg_free_graviton(&ks);
    let hs = qg_starobinsky_scalaron_field(&ks, m_scal);
    let bp = BandParams {
        big_n: 8.0,
        h: 0.5,
        t: 0.8,
    };

    // Superposition starts: a pure number eigenstate collapses the Krylov
    // rank of a diagonal Hamiltonian.
    let mk_super = || {
        let mut s = empty_vacuum();
        s.scale_and_add(
            &empty_vacuum().apply(&Operator::InnerBosonCreate(0)),
            Complex64::new(1.0, 0.0),
        );
        s
    };
    let psi_g = mk_super();
    let psi_s = mk_super();

    println!("— Certified speed split at k=0.9 (massless graviton vs scalaron)");
    println!("  SHARP certificate: Rayleigh–Ritz residual bound |θ−λ| ≤ ‖r‖");
    println!("  (Parlett); the Theorem 4.1 envelope is printed alongside as the");
    println!("  a-priori ceiling — its constants are too loose for close-level");
    println!("  separation on unitary problems.");
    for m in [6_usize, 9] {
        let rg = hg.solve(&psi_g, &bp, m);
        let rs = hs.solve(&psi_s, &bp, m);
        // One-quanton sectors: lowest pair straddles {0, ω}; take the
        // SECOND rung (= ω) with its own residual certificate.
        let pick = |c: &[Certificate]| -> Certificate {
            *c.iter()
                .filter(|x| x.value > 1e-6)
                .min_by(|a, b| a.value.partial_cmp(&b.value).unwrap())
                .expect("ω rung")
        };
        let cg = pick(&certified_ritz_values(&rg));
        let cs = pick(&certified_ritz_values(&rs));
        print_cert(&format!("ω_grav (m={m}) [res-certified]"), &cg);
        print_cert(&format!("ω_scal (m={m}) [res-certified]"), &cs);

        println!(
            "      Theorem 4.1 ceiling for ω_scal: {:.3e}",
            envelope_ceiling(&rs, &bp, m, psi_s.norm())
        );

        assert!(cs.value > cg.value, "massive branch ordering");
        assert!(
            cg.hi < cs.lo,
            "RESIDUAL-certified intervals must be disjoint: \
             [{:.6},{:.6}] vs [{:.6},{:.6}]",
            cg.lo,
            cg.hi,
            cs.lo,
            cs.hi
        );
    }
}
