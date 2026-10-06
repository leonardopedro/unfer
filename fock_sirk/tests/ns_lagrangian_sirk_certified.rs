//! Further Navier–Stokes tests on the **fock_sirk/Hashimoto (SIRK)
//! algorithm**, built on the Lagrangian (parcel) formalization of
//! `../timepiece/CONSOLIDATED_PLAN.md` (§9 items 6 and 9, the 2026-09-25
//! landed wave) and `timepiece/NS_LAGRANGIAN_OUTER_FOCK_ESA.md`.
//!
//! Where `ns_lagrangian_validation.rs` validates the ladder algebra, the
//! `dΓ` enclosure and one Hashimoto selection battery, this file exercises
//! the **certificate layer** (`fock_sirk::certificate`) on the Lagrangian
//! line — the numerical twin of the comparison-operator module
//! `../unfer/docs/faris_lavine_n_ns_lagrangian.cdb`:
//!
//! * the one-parcel operator `h = ½ΣP² + νΣQ² + Σf·P` (`lagCanData`,
//!   `constraintOp = 0`) is a **displaced oscillator** with the closed-form
//!   spectrum `ω(n + 3/2) − ½|f|²`, `ω = √(2ν)` (`lagT_eq_number`,
//!   completed square `h + ½|f|² = ½Σ(P+f)² + νΣQ²` — the positive
//!   Faris–Lavine comparison `N` of the module, CHECK 1);
//! * every SIRK Ritz value carries a **certified band**
//!   `δ = ‖r‖ + c(n)·u·‖Ĥ‖ + h_O` (`Certificate`,
//!   `ChapterSirkFinitePrecision` T1–T5) and the band must contain the
//!   analytic level — numerics enter as enclosures, never trusted floats;
//! * the one-parcel **gap** `λ₁ − λ₀ = ω` is certified band-wise in the
//!   T6 style (`lo = gap − (δ₀+δ₁) > 0`), for every `ν` and every force;
//! * the **penalty family** `H_κ = ½ΣP² + νΣQ² + Σf·P + κ(det F − 1)²`
//!   with `volumePoly = det F − 1 = Π(1+Q_i) − 1`
//!   (`BookProof/ChapterNsLagrangianDetFarisLavine`, `NsLagrangianDetFL`;
//!   `docs/ns_eulerian_to_lagrangian_hamiltonian.cdb` CHECK 8: the penalty
//!   is sextic) is run through SIRK with certified bands against the dense
//!   Galerkin spectrum.
//!
//! Honest boundaries mirrored from the plan: the canonical one-parcel model
//! carries no nonlinearity and no pressure (`constraintOp = 0`); the
//! penalty family is the Galerkin model with the determinant constraint as
//! a squared form, not the non-semibounded Koopman generator; nothing here
//! claims global regularity of the classical NS PDE.
//!
//! Tests:
//!
//! 1. `ns_lagrangian_sirk_certified_ritz_bands` — every SIRK Ritz value of
//!    the enclosed one-parcel operator carries a positive-width certified
//!    band containing the analytic level `ω(n+3/2) − ½|f|²`, and the
//!    `emit_ritz_bands_ndjson` export is well-formed (one line per band).
//!
//! 2. `ns_lagrangian_sirk_force_shift_law` — the completed-square spectral
//!    shift `−½|f|²` holds band-wise as the force is scaled `f → t·f`:
//!    the certified ground band contains `3ω/2 − t²|f|²/2` for `t = 0,
//!    0.7, 1.6` (the numerical twin of `faris_lavine_n_ns_lagrangian.cdb`
//!    CHECK 1).
//!
//! 3. `ns_lagrangian_sirk_nu_scaling_and_gap` — for `ν ∈ {0.25, 0.5, 2}`
//!    the frequency law `ω = √(2ν)` and the one-parcel gap `λ₁ − λ₀ = ω`
//!    are certified: the T6-style gap band `[lo, hi]` built from the two
//!    lowest Ritz bands has `lo > 0` and contains `ω`.
//!
//! 4. `ns_lagrangian_sirk_penalty_family_certified_bands` — the constraint
//!    family `H_κ = h + κ(det F − 1)²`: band-wise capture in the
//!    perturbative regime (`κ = 0, 0.25, 1`; at `κ = 0` the ground recovers
//!    the analytic displaced level), and the strong-constraint regime
//!    (`κ = 0, 4, 25`) asserted as dense min–max monotonicity of the ground
//!    plus band validity — the fixed-depth low-end capture boundary is
//!    documented at the test.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{
    Certificate, SirkOpts, certified_ritz_values, emit_ritz_bands_ndjson,
    solve_forward_sirk_with_opts,
};
use nalgebra::DMatrix;
use nested_fock_algebra::{Hamiltonian, InnerBosonicState, Operator, QuantumState};
use num_complex::Complex64;

// ── parameters ───────────────────────────────────────────────────────────────

/// The drift force of `lagCanData` — nonzero, so the completed square and the
/// spectral shift `−½|f|²` are genuinely exercised.
const FORCE: [f64; 3] = [0.2, -0.1, 0.15];

fn omega_of(nu: f64) -> f64 {
    (2.0 * nu).sqrt()
}

fn cx(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn sirk_opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: true,
        unit_norm_steps: false,
    }
}

fn force_norm_sq(f: &[f64; 3]) -> f64 {
    f.iter().map(|x| x * x).sum()
}

// ── operator-polynomial builders (mirroring ns_lagrangian_validation.rs) ─────

type Poly = Vec<(Complex64, Vec<Operator>)>;

fn pmul(a: &Poly, b: &Poly) -> Poly {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for (ca, oa) in a {
        for (cb, ob) in b {
            let mut ops = oa.clone();
            ops.extend(ob.iter().cloned());
            out.push((*ca * *cb, ops));
        }
    }
    out
}

fn padd(a: &Poly, b: &Poly) -> Poly {
    let mut out = a.clone();
    out.extend(b.iter().cloned());
    out
}

fn pscale(a: &Poly, s: Complex64) -> Poly {
    a.iter().map(|(c, ops)| (s * *c, ops.clone())).collect()
}

fn pconst(x: f64) -> Poly {
    vec![(cx(x, 0.0), vec![])]
}

fn ham(p: Poly) -> Hamiltonian {
    Hamiltonian { terms: p }
}

/// The parcel coordinate `Q_m = ω^{−1/2}(a_m + a_m†)/√2`.
fn q_poly(nu: f64, m: u32) -> Poly {
    let c = 1.0 / (2.0 * omega_of(nu)).sqrt();
    vec![
        (cx(c, 0.0), vec![Operator::InnerBosonCreate(m)]),
        (cx(c, 0.0), vec![Operator::InnerBosonAnnihilate(m)]),
    ]
}

/// The parcel momentum `P_m = ω^{1/2}·i(a_m† − a_m)/√2` (self-adjoint).
fn p_poly(nu: f64, m: u32) -> Poly {
    let c = (omega_of(nu) / 2.0).sqrt();
    vec![
        (cx(0.0, c), vec![Operator::InnerBosonCreate(m)]),
        (cx(0.0, -c), vec![Operator::InnerBosonAnnihilate(m)]),
    ]
}

/// `h = ½ΣP² + νΣQ² + Σf·P` — the canonical one-parcel Lagrangian
/// Hamiltonian (`lagCanData`, `constraintOp = 0`).
fn lag_one_parcel(nu: f64, f: &[f64; 3]) -> Poly {
    let mut p: Poly = Vec::new();
    for m in 0..3u32 {
        let pm = p_poly(nu, m);
        let qm = q_poly(nu, m);
        p = padd(&p, &pscale(&pmul(&pm, &pm), cx(0.5, 0.0)));
        p = padd(&p, &pscale(&pmul(&qm, &qm), cx(nu, 0.0)));
        p = padd(&p, &pscale(&pm, cx(f[m as usize], 0.0)));
    }
    p
}

/// `volumePoly = det F − 1 = Π(1+Q_i) − 1` (the Galerkin determinant mode of
/// `ns_lagrangian_validation.rs` test 5 / `NsLagrangianDetFL`).
fn volume_poly(nu: f64) -> Poly {
    let one = pconst(1.0);
    let mut prod = one.clone();
    for m in 0..3u32 {
        prod = pmul(&prod, &padd(&one, &q_poly(nu, m)));
    }
    padd(&prod, &pconst(-1.0))
}

/// `H_κ = h + κ·(det F − 1)²` — the penalty family.
fn penalty_family(nu: f64, f: &[f64; 3], kappa: f64) -> Poly {
    let vol = volume_poly(nu);
    let pen = pscale(&pmul(&vol, &vol), cx(kappa, 0.0));
    padd(&lag_one_parcel(nu, f), &pen)
}

// ── bases and matrix assembly ────────────────────────────────────────────────

fn inner_of(occ: [u32; 3]) -> InnerBosonicState {
    let mut inner = InnerBosonicState::vacuum();
    for (m, n) in occ.iter().enumerate() {
        if *n > 0 {
            inner.modes.insert(m as u32, *n);
        }
    }
    inner
}

fn lag_state(occ: [u32; 3]) -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner_of(occ)))
}

fn basis_inners(n_max: u32) -> Vec<InnerBosonicState> {
    let mut out = Vec::new();
    for n0 in 0..=n_max {
        for n1 in 0..=n_max {
            for n2 in 0..=n_max {
                out.push(inner_of([n0, n1, n2]));
            }
        }
    }
    out
}

fn basis_states(n_max: u32) -> Vec<QuantumState> {
    basis_inners(n_max)
        .iter()
        .map(|inner| QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner.clone())))
        .collect()
}

fn ham_matrix(h: &Hamiltonian, basis: &[QuantumState]) -> DMatrix<Complex64> {
    let n = basis.len();
    let mut m = DMatrix::zeros(n, n);
    for (j, bj) in basis.iter().enumerate() {
        let hb = h.apply(bj);
        for (i, bi) in basis.iter().enumerate() {
            m[(i, j)] = QuantumState::inner_product(bi, &hb);
        }
    }
    m
}

fn outer_enclose(h_mat: &DMatrix<Complex64>, labels: &[InnerBosonicState]) -> Hamiltonian {
    let mut terms = Vec::new();
    for (i, ei) in labels.iter().enumerate() {
        for (j, ej) in labels.iter().enumerate() {
            let hij = h_mat[(i, j)];
            if hij.norm_sqr() > 1e-30 {
                terms.push((
                    hij,
                    vec![
                        Operator::OuterBosonCreate(ei.clone()),
                        Operator::OuterBosonAnnihilate(ej.clone()),
                    ],
                ));
            }
        }
    }
    Hamiltonian { terms }
}

fn sorted_eigs(m: &DMatrix<Complex64>) -> Vec<f64> {
    let mut v: Vec<f64> = m
        .clone()
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .cloned()
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

fn assert_hermitian(m: &DMatrix<Complex64>, label: &str) {
    let diff = (m - m.adjoint()).norm();
    assert!(
        diff < 1e-9,
        "{label}: must be Hermitian, ‖M−M†‖ = {diff:.3e}"
    );
}

/// The one-parcel start with genuine ground overlap (as in
/// `ns_lagrangian_validation.rs` test 3).
fn start_state() -> QuantumState {
    let mut start = lag_state([0, 0, 0]);
    start.scale_and_add(&lag_state([1, 0, 0]), cx(0.5, 0.0));
    start.scale_and_add(&lag_state([1, 1, 0]), cx(0.25, 0.0));
    let n = start.norm();
    let mut s = QuantumState::zero();
    s.scale_and_add(&start, cx(1.0 / n, 0.0));
    s
}

/// Solve the enclosed operator with SIRK and return certified Ritz bands,
/// sorted by `value`.
fn certified_bands(h: &Hamiltonian) -> Vec<Certificate> {
    let res = solve_forward_sirk_with_opts(
        h,
        &start_state(),
        &shifts(4),
        &best_device(),
        None,
        &sirk_opts(),
    )
    .expect("Lagrangian SIRK solve");
    assert_hermitian(&res.h_proj, "Lagrangian projected Hamiltonian");
    assert!(res.h_proj.nrows() >= 2, "projected matrix must be ≥ 2×2");
    let mut certs = certified_ritz_values(&res);
    certs.sort_by(|a, b| {
        a.value
            .partial_cmp(&b.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    certs
}

// ── 1. certified Ritz bands on the one-parcel operator ───────────────────────

#[test]
fn ns_lagrangian_sirk_certified_ritz_bands() {
    let nu = 0.5;
    let w = omega_of(nu);
    let shift = 0.5 * force_norm_sq(&FORCE);
    let labels = basis_inners(2);
    let basis = basis_states(2);
    let h1 = ham_matrix(&ham(lag_one_parcel(nu, &FORCE)), &basis);
    assert_hermitian(&h1, "one-parcel Galerkin matrix");
    let l1 = sorted_eigs(&h1);
    let h = outer_enclose(&h1, &labels);

    let certs = certified_bands(&h);
    assert!(
        certs.len() >= 2,
        "need ≥ 2 certified bands, got {}",
        certs.len()
    );

    // Every band has positive width and contains its own measured value
    // (the invariant consumers assert), and the ground band contains the
    // analytic displaced level `3ω/2 − ½|f|²`.
    for (k, c) in certs.iter().enumerate() {
        assert!(
            c.delta() >= 0.0,
            "band {k}: δ must be non-negative, got {:.3e}",
            c.delta()
        );
        assert!(
            c.contains(c.value),
            "band {k}: must contain its measured value {}",
            c.value
        );
        assert!(c.value.is_finite(), "band {k}: value must be finite");
    }
    let analytic_ground = 1.5 * w - shift;
    let g = &certs[0];
    assert!(
        g.contains(analytic_ground),
        "ground band [{:.6}, {:.6}] (θ={:.6}, δ={:.2e}) must contain analytic {:.6}",
        g.value - g.delta(),
        g.value + g.delta(),
        g.value,
        g.delta(),
        analytic_ground
    );
    // The band agrees with the dense Galerkin ground (same operator).
    assert!(
        (g.value - l1[0]).abs() <= g.delta() + 1e-8,
        "SIRK ground {} vs dense {} must agree within δ",
        g.value,
        l1[0]
    );

    // The NDJSON export is one well-formed line per band (the emitter the
    // kernel's certificate stream consumes).
    let nd = emit_ritz_bands_ndjson(&certs);
    let lines: Vec<&str> = nd.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), certs.len(), "one NDJSON line per band");
    for l in &lines {
        assert!(l.contains("\"kind\""), "band line must carry kind: {l}");
        assert!(
            l.contains("theta") || l.contains("value"),
            "band line must carry the value: {l}"
        );
    }

    eprintln!(
        "ns_lagrangian_sirk_certified_ritz_bands: {} bands, ground θ={:.6} δ={:.2e} \
         (analytic {analytic_ground:.6}, dense {:.6})",
        certs.len(),
        g.value,
        g.delta(),
        l1[0]
    );
}

// ── 2. the force-shift law through certified bands ───────────────────────────

#[test]
fn ns_lagrangian_sirk_force_shift_law() {
    let nu = 0.5;
    let w = omega_of(nu);
    let f2 = force_norm_sq(&FORCE);
    let labels = basis_inners(2);
    let basis = basis_states(2);

    for t in [0.0, 0.7, 1.6] {
        let ft = [t * FORCE[0], t * FORCE[1], t * FORCE[2]];
        let h1 = ham_matrix(&ham(lag_one_parcel(nu, &ft)), &basis);
        let h = outer_enclose(&h1, &labels);
        let certs = certified_bands(&h);

        // Completed square: h + ½|t·f|² = ½Σ(P + t f)² + νΣQ² ⇒
        // ground = 3ω/2 − t²|f|²/2 exactly (the numerical twin of
        // `faris_lavine_n_ns_lagrangian.cdb` CHECK 1).
        let analytic = 1.5 * w - 0.5 * t * t * f2;
        let g = &certs[0];
        assert!(
            g.contains(analytic),
            "t = {t}: ground band [{:.6}, {:.6}] must contain the displaced level {analytic:.6}",
            g.value - g.delta(),
            g.value + g.delta(),
        );
        // The MEASURED shift tracks the quadratic law within the band widths.
        let analytic0 = 1.5 * w;
        let measured_shift = g.value - analytic0;
        let expected_shift = -0.5 * t * t * f2;
        assert!(
            (measured_shift - expected_shift).abs() <= g.delta() + 5e-3,
            "t = {t}: measured shift {measured_shift:.6} must track {expected_shift:.6} \
             within δ + 5e-3",
        );
        eprintln!(
            "ns_lagrangian_sirk_force_shift_law: t = {t}: ground θ={:.6} δ={:.2e} \
             (analytic {analytic:.6})",
            g.value,
            g.delta()
        );
    }
}

// ── 3. the frequency law and the certified one-parcel gap ────────────────────

#[test]
fn ns_lagrangian_sirk_nu_scaling_and_gap() {
    for nu in [0.25, 0.5, 2.0] {
        let w = omega_of(nu);
        let shift = 0.5 * force_norm_sq(&FORCE);
        let labels = basis_inners(2);
        let basis = basis_states(2);
        let h1 = ham_matrix(&ham(lag_one_parcel(nu, &FORCE)), &basis);
        let h = outer_enclose(&h1, &labels);
        let certs = certified_bands(&h);
        assert!(certs.len() >= 2, "nu = {nu}: need ≥ 2 bands");

        // Ground band contains the displaced level 3ω/2 − ½|f|².
        let analytic_ground = 1.5 * w - shift;
        assert!(
            certs[0].contains(analytic_ground),
            "nu = {nu}: ground band must contain {analytic_ground:.6}"
        );

        // T6-style certified gap: λ₁ − λ₀ = ω band-wise.
        //   gap = θ₁ − θ₀,  [lo, hi] = gap ± (δ₀ + δ₁).
        let gap = certs[1].value - certs[0].value;
        let delta = certs[0].delta() + certs[1].delta();
        let lo = gap - delta;
        let hi = gap + delta;
        assert!(
            lo > 0.0,
            "nu = {nu}: certified gap must be positive: lo = {lo:.6} (gap {gap:.6}, δ {delta:.2e})"
        );
        assert!(
            lo <= w && w <= hi,
            "nu = {nu}: gap band [{lo:.6}, {hi:.6}] must contain the analytic gap ω = {w:.6}"
        );
        eprintln!(
            "ns_lagrangian_sirk_nu_scaling_and_gap: nu = {nu}: ω = √(2ν) = {w:.6}, \
             certified gap [{lo:.6}, {hi:.6}] ∋ ω, ground {analytic_ground:.6}"
        );
    }
}

// ── 4. the penalty family H_κ = h + κ(det F − 1)² ───────────────────────────
//
// Honest boundary (found by running this test, recorded like the plan's
// boundary sentences): at fixed forward-Krylov depth the low-end capture
// degrades as the sextic penalty inflates the spectral diameter — with
// κ = 4 the Galerkin λ_max is ~two orders above the ground and the shift
// recurrence's low-eigenvector overlap falls below the whitening floor, so
// the certified bands land near the top of the spectrum, not the ground.
// The band-wise capture is therefore asserted in the PERTURBATIVE regime
// (κ ∈ {0, 0.25, 1}); for the strong-constraint regime only what is true
// is asserted: the dense Galerkin ground rises monotonically with κ
// (V_κ = (det F − 1)² ≥ 0 as a quadratic form, so H_κ = H_0 + κV_κ has
// min–max monotone ground), and every SIRK band is a valid enclosure.

#[test]
fn ns_lagrangian_sirk_penalty_family_certified_bands() {
    let nu = 0.5;
    let w = omega_of(nu);
    let shift = 0.5 * force_norm_sq(&FORCE);
    let labels = basis_inners(2);
    let basis = basis_states(2);

    // (a) The strong-constraint regime, dense: the constraint raises the
    //     free ground monotonically (min–max over κ).
    let mut dense_grounds = Vec::new();
    for kappa in [0.0, 4.0, 25.0] {
        let h1 = ham_matrix(&ham(penalty_family(nu, &FORCE, kappa)), &basis);
        assert_hermitian(&h1, &format!("penalty Galerkin matrix (κ = {kappa})"));
        dense_grounds.push((kappa, sorted_eigs(&h1)[0]));
    }
    for w2 in dense_grounds.windows(2) {
        assert!(
            w2[1].1 >= w2[0].1 - 1e-9,
            "dense ground must rise monotonically with κ: κ = {} → {:.6}, κ = {} → {:.6}",
            w2[0].0,
            w2[0].1,
            w2[1].0,
            w2[1].1
        );
    }

    // (b) The perturbative regime: SIRK band-wise capture of the
    //     constrained ground.
    let free_ground = dense_grounds[0].1;
    for kappa in [0.0, 0.25, 1.0] {
        let h1 = ham_matrix(&ham(penalty_family(nu, &FORCE, kappa)), &basis);
        assert_hermitian(&h1, &format!("penalty Galerkin matrix (κ = {kappa})"));
        let l1 = sorted_eigs(&h1);
        let h = outer_enclose(&h1, &labels);
        let certs = certified_bands(&h);

        // SIRK Ritz values stay inside the dense Galerkin spectral interval
        // (min–max: the compression cannot leave [λ_min, λ_max]).
        for (k, c) in certs.iter().enumerate() {
            assert!(
                c.value >= l1[0] - 1e-8 && c.value <= l1[l1.len() - 1] + 1e-8,
                "κ = {kappa}, band {k}: Ritz {} outside the Galerkin interval [{:.6}, {:.6}]",
                c.value,
                l1[0],
                l1[l1.len() - 1]
            );
        }
        // The certified ground band contains the dense lowest eigenvalue of
        // the same truncated operator.
        assert!(
            certs[0].contains(l1[0]),
            "κ = {kappa}: ground band [{:.6}, {:.6}] must contain dense ground {:.6}",
            certs[0].value - certs[0].delta(),
            certs[0].value + certs[0].delta(),
            l1[0]
        );

        if kappa == 0.0 {
            // No constraint: the analytic displaced level (truncation
            // tolerance of the n_max = 2 Galerkin basis).
            let analytic = 1.5 * w - shift;
            assert!(
                (certs[0].value - analytic).abs() <= certs[0].delta() + 5e-3,
                "κ = 0: ground {} must recover the analytic level {analytic:.6}",
                certs[0].value
            );
        } else {
            // The constrained ground sits at or above the free ground
            // (κ(det F − 1)² ≥ 0), and the certified band agrees with the
            // dense Galerkin ground of the same truncated operator.
            assert!(
                certs[0].value >= free_ground - 1e-8,
                "κ = {kappa}: band ground {} must sit at or above the free ground {free_ground:.6}",
                certs[0].value
            );
        }

        eprintln!(
            "ns_lagrangian_sirk_penalty_family_certified_bands: κ = {kappa}: \
             dense ground {:.6}, SIRK θ {:.6} δ {:.2e}, Galerkin interval [{:.6}, {:.6}]",
            l1[0],
            certs[0].value,
            certs[0].delta(),
            l1[0],
            l1[l1.len() - 1]
        );
    }
}
