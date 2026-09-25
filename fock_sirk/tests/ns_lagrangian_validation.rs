//! Navier–Stokes validation in **Lagrangian (trajectory/parcel) variables** —
//! the numerical twin of the formalization in `../timepiece/CONSOLIDATED_PLAN.md`
//! (§9 items 6 and 9, and the 2026-09-25 landed wave), complementing the
//! Eulerian suites (`ns_validation.rs`, `ns_derivative_variable_fixing.rs`,
//! `ns_further_validation.rs`) with the parallel picture the plan calls "the
//! deliberately non-priority parallel route".
//!
//! **What the Lagrangian formalization is** (CONSOLIDATED_PLAN.md §9 item 6,
//! `BookProof/ChapterNavierStokesLagrangianCanonical.lean`,
//! `BookProof/ChapterNavierStokesFullLagrangianFock.lean`, and
//! `timepiece/NS_LAGRANGIAN_OUTER_FOCK_ESA.md`):
//!
//! * The fluid is read as **parcels**: trajectories `X(a,t)` with velocity
//!   `v = ∂_t X`, deformation gradient `F = ∇_a X`, velocity gradient `V`,
//!   viscous coordinate `S`.  The incompressibility constraint
//!   `div u = 0` transports to **volume preservation `det F = 1`**
//!   (`volumePoly = detPoly − 1`, Cauchy's Jacobian theorem) — the exact
//!   constraint in Lagrangian variables, imposed in the interacting model by
//!   the penalty `V_κ` (`BookProof/ChapterNsLagrangianDetFarisLavine.lean`,
//!   `NsLagrangianDetFL`) riding as a squared form.
//! * The **one-parcel canonical realization** (`lagCanData`, proved ESA by
//!   `lagCan_esa` — the only unconditional Lagrangian ESA on a proper core):
//!
//!     `h = ½ Σᵢ Pᵢ² + ν Σᵢ Qᵢ² + Σᵢ fᵢ Pᵢ + constraintOp`,   `constraintOp = 0`,
//!
//!   with `ω = √(2ν)`, `Qᵢ = ω^{−1/2}(aᵢ + aᵢ†)/√2`,
//!   `Pᵢ = ω^{1/2}·i(aᵢ† − aᵢ)/√2`, `[Pᵢ,Qⱼ] = −i δᵢⱼ`
//!   (`comm_lagP_lagQ`), and the second-order part the 3D oscillator
//!   `½ΣP² + νΣQ² = ω(N + 3/2)` (`lagT_eq_number` / `lagCan_secondOrder_eq`).
//!   This is the operator of `nsLagrangian_timeIndependent_singleTime`
//!   (autonomous flow, every `ν > 0`, every force `f`), and Kato–Rellich
//!   (`hFull_essentiallySelfAdjointOn_of_drive_eq_P`) discharges the drift.
//! * The **final Hamiltonian is the enclosure**: `H = dΓ(h)` — the one-parcel
//!   operator enclosed in outer creation (left) / annihilation (right),
//!   `Σ_{ij} h_{ij} C†(e_i) A(e_j)` — proved ESA on the finite-parcel domain
//!   by `lagOne_dGamma_esa`
//!   (`BookProof/ChapterNsLagrangianOuterFockEsa.lean`, the direct instance of
//!   `dGamma_essentiallySelfAdjointOn_of_esa` over `lagCan_esa`), and the
//!   interacting auxiliary sum-of-squares `½Σπ² + ½Σ(form)²`
//!   (`lagFullFockHam`, sectors `lagSectorHam n` = `weylPoly`) by
//!   `lagFullFockHam_esa` / `lagSectorHam_esa`
//!   (`BookProof/ChapterNsFullLagrangianFockEsa.lean`) — per-sector Weyl sums
//!   of squares of real polynomials with injective momentum coordinates
//!   (`lagIdx_injective`), glued by `dsOp_essentiallySelfAdjointOn`.
//! * Products-with-derivatives are disposed of by the **momentum-space
//!   convolution** (NS and QG only); on the material side the spatial transform
//!   is taken on the *reference* coordinate `a`, where the substitution is
//!   `σ(F) = iℓ⊗ξ`, `σ(V_{ij}) = i ℓ_j v_i`, `σ(S_i) = −|ℓ|² v_i` — and the
//!   symbolic companion module `../unfer/docs/ns_lagrangian_hamiltonian.cdb`
//!   derives the Lagrangian Hamiltonian from the Eulerian one, constraint
//!   included, and certifies why `det F` must ride as an independent mode
//!   (the rank-one degeneracy: `cof(F) = 0`, `volumePoly ↦ −1`).
//!
//! Honest boundaries mirrored from the plan: the canonical one-parcel model
//! carries **no nonlinearity, no pressure and no interaction**
//! (`constraintOp = 0`, one parcel); the interacting statements are about the
//! **auxiliary** positive sum of squares, not the non-semibounded Koopman
//! generator; nothing here claims global regularity of the classical NS PDE.
//!
//! The Lagrangian route has one structural property the Eulerian suites never
//! exercise: the one-parcel operator is a ** displaced harmonic oscillator**,
//! so every test below can be checked against the *closed-form* spectrum
//! `ω(n + 3/2) − ½|f|²` (complete-the-square), while the outer enclosure is
//! exactly number-conserving with the n-parcel spectrum the n-fold sums of the
//! one-particle levels — the sector structure `lagOne_dGamma_esa` asserts.
//!
//! Tests:
//!
//! 1. `ns_lagrangian_canonical_ladder_realization` — the ladder algebra of
//!    `lagCanData`: `[P_i,Q_j] = −i δ_ij` and `½ΣP² + νΣQ² = ω(N + 3/2)` as
//!    exact operator identities on probe states; the completed square
//!    `h + ½|f|² = ½Σ(P_i+f_i)² + νΣQ²` (the Kato–Rellich input and the
//!    source of the spectral shift); the truncated matrices Hermitian, with
//!    the `f = 0` spectrum *exactly* the oscillator levels `ω(n + 3/2)` and
//!    the `f ≠ 0` low spectrum the displaced levels `ω(n + 3/2) − ½|f|²`.
//!
//! 2. `ns_lagrangian_dGamma_enclosure_parcel_sectors` — the final-Hamiltonian
//!    enclosure `H = Σ h_ij C†(e_i) A(e_j)`: every term creator-left /
//!    annihilator-right (doctrine), `H|Ω⟩ = 0` identically, the one-parcel
//!    sector action *is* the one-particle matrix, universe number exactly
//!    conserved, and the two-parcel sector spectrum the pairwise sums
//!    `λ_a + λ_b` (the non-interacting content of `lagOne_dGamma_esa`) —
//!    hence, `h ≻ 0`, the outer vacuum is the ground with gap `λ_min(h)`.
//!
//! 3. `ns_lagrangian_sirk_hashimoto_selection` — the Hashimoto shift-invert
//!    selection (`ChapterHashimotoComplexShifts`,
//!    `ChapterSirkSingleTimeShift`) on the enclosed Lagrangian operator: the
//!    SIRK projection is Hermitian, its Ritz values lie in the one-parcel
//!    spectral interval and capture the analytic ground; the resolvent
//!    `(γI−A)⁻¹` exists for every non-real shift, obeys the identity and the
//!    `1/|Im γ|` bound, and `λ_j = γ − 1/μ_j` recovers the spectrum.
//!
//! 4. `ns_lagrangian_ehrenfest_unitary_flow` — the autonomous flow of
//!    `nsLagrangian_timeIndependent_singleTime`: the exact Heisenberg
//!    equations `d⟨Q_i⟩/dt = ⟨P_i⟩ + f_i`, `d⟨P_i⟩/dt = −2ν⟨Q_i⟩` (hence
//!    oscillation at `ω = √(2ν)`), checked both as commutator identities and
//!    by a finite difference along the SIRK-restarted flow, which conserves
//!    norm, energy and the parcel-number sector weights.
//!
//! 5. `ns_lagrangian_volume_constraint_penalty` — the constraint term: in the
//!    diagonal-stretching ansatz `F = diag(1 + Q_i)` (exact for that ansatz)
//!    `volumePoly = det F − 1 = Π(1+Q_i) − 1`, and the penalty Hamiltonian
//!    `H_κ = ½ΣP² + νΣQ² + κ·volumePoly²` — the squared-form structure of
//!    `lagSectorHam` / the `V_κ` of `NsLagrangianDetFL`: a positive
//!    sum-of-squares finite section (PSD spectrum, the shadow of
//!    `weylPoly_esa`), monotone ground energy in `κ`, the ground state
//!    localizing onto `volumePoly = 0` as `κ` grows (the exact-energy
//!    decomposition `E = ⟨osc⟩ + κ⟨volumePoly²⟩`), and unitary SIRK evolution
//!    of the constrained Hamiltonian.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, evolve_restarted, solve_forward_sirk_with_opts};
use nalgebra::DMatrix;
use nested_fock_algebra::{Hamiltonian, InnerBosonicState, Operator, QuantumState};
use num_complex::Complex64;

// ── shared parameters ────────────────────────────────────────────────────────

/// Viscosity giving the package frequency `ω = √(2ν) = 1`.
const NU: f64 = 0.5;
/// The drift force of `lagCanData` — nonzero, so the completed square and the
/// spectral shift `−½|f|²` are genuinely exercised.
const FORCE: [f64; 3] = [0.2, -0.1, 0.15];

fn omega() -> f64 {
    (2.0 * NU).sqrt()
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

fn assert_hermitian(m: &DMatrix<Complex64>, label: &str) {
    let diff = (m - m.adjoint()).norm();
    assert!(diff < 1e-9, "{label}: must be Hermitian, ‖M−M†‖ = {diff:.3e}");
}

// ── operator-polynomial builders (mirror the private helpers of models.rs) ───

/// A Hamiltonian as a list of `(coeff, ops)` terms with `ops` in algebraic
/// left-to-right order (rightmost acts first); an empty `ops` list is a
/// constant (the framework's scalar term).
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
fn q_poly(m: u32) -> Poly {
    let c = 1.0 / (2.0 * omega()).sqrt();
    vec![
        (cx(c, 0.0), vec![Operator::InnerBosonCreate(m)]),
        (cx(c, 0.0), vec![Operator::InnerBosonAnnihilate(m)]),
    ]
}

/// The parcel momentum `P_m = ω^{1/2}·i(a_m† − a_m)/√2` (self-adjoint).
fn p_poly(m: u32) -> Poly {
    let c = (omega() / 2.0).sqrt();
    vec![
        (cx(0.0, c), vec![Operator::InnerBosonCreate(m)]),
        (cx(0.0, -c), vec![Operator::InnerBosonAnnihilate(m)]),
    ]
}

/// `½ Σ_m P_m² + ν Σ_m Q_m²` — the one-parcel second-order part.
fn lag_osc_poly() -> Poly {
    let mut p: Poly = Vec::new();
    for m in 0..3u32 {
        let pm = p_poly(m);
        let qm = q_poly(m);
        p = padd(&p, &pscale(&pmul(&pm, &pm), cx(0.5, 0.0)));
        p = padd(&p, &pscale(&pmul(&qm, &qm), cx(NU, 0.0)));
    }
    p
}

/// `Σ_m f_m P_m` — the constant-force drift of `lagCanData`.
fn lag_drift_poly(f: &[f64; 3]) -> Poly {
    let mut p: Poly = Vec::new();
    for m in 0..3u32 {
        p = padd(&p, &pscale(&p_poly(m), cx(f[m], 0.0)));
    }
    p
}

/// The canonical one-parcel Lagrangian Hamiltonian
/// `h = ½ΣP² + νΣQ² + Σf·P` (the `constraintOp = 0` instance of `lagCanData`).
fn lag_one_parcel(f: &[f64; 3]) -> Hamiltonian {
    ham(padd(&lag_osc_poly(), &lag_drift_poly(f)))
}

/// The three-mode number operator `N = Σ a†a` on modes 0..2.
fn number_op() -> Hamiltonian {
    let mut terms = Vec::new();
    for m in 0..3u32 {
        terms.push((
            cx(1.0, 0.0),
            vec![
                Operator::InnerBosonCreate(m),
                Operator::InnerBosonAnnihilate(m),
            ],
        ));
    }
    Hamiltonian { terms }
}

// ── truncated bases and matrix assembly ──────────────────────────────────────

fn inner_of(occ: [u32; 3]) -> InnerBosonicState {
    let mut inner = InnerBosonicState::vacuum();
    for (m, n) in occ.iter().enumerate() {
        if *n > 0 {
            inner.modes.insert(m as u32, *n);
        }
    }
    inner
}

/// One universe whose inner configuration is `occ`.
fn lag_state(occ: [u32; 3]) -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner_of(occ)))
}

/// All occupation triples with each mode `≤ n_max`, lexicographic order.
/// The triple `(n0,n1,n2)` sits at index `(n0·S + n1)·S + n2`, `S = n_max+1`.
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

/// The exact matrix of `h` on the truncated inner basis:
/// `M_ij = ⟨e_i|h|e_j⟩`.
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

/// The final-Hamiltonian enclosure `H = Σ_{ij} h_ij C†(e_i) A(e_j)` —
/// creation on the left, annihilation on the right (doctrine clause 1).
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

/// `i⟨ψ|[H,O]|ψ⟩` — the Heisenberg right-hand side `d⟨O⟩/dt`.
fn heisenberg(h: &Hamiltonian, o: &Hamiltonian, psi: &QuantumState) -> Complex64 {
    let ho = h.apply(&o.apply(psi));
    let oh = o.apply(&h.apply(psi));
    let mut d = ho;
    d.scale_and_add(&oh, cx(-1.0, 0.0));
    cx(0.0, 1.0) * QuantumState::inner_product(psi, &d)
}

fn expect(psi: &QuantumState, o: &Hamiltonian) -> Complex64 {
    QuantumState::inner_product(psi, &o.apply(psi))
}

fn energy(h: &Hamiltonian, psi: &QuantumState) -> f64 {
    QuantumState::inner_product(psi, &h.apply(psi)).re
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

/// Analytic spectrum of `½ΣP² + νΣQ² + Σf·P` over all triples from a list:
/// `ω(n + 3/2) − ½|f|²` (complete the square; the shift is unitary).
fn analytic_levels(triples: &[[u32; 3]]) -> Vec<f64> {
    let shift = 0.5 * FORCE.iter().map(|x| x * x).sum::<f64>();
    let mut v: Vec<f64> = triples
        .iter()
        .map(|t| omega() * ((t[0] + t[1] + t[2]) as f64 + 1.5) - shift)
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

fn triples_up_to(n_max: u32) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    for n0 in 0..=n_max {
        for n1 in 0..=n_max {
            for n2 in 0..=n_max {
                out.push([n0, n1, n2]);
            }
        }
    }
    out
}

// ── 1. the canonical one-parcel ladder realization ──────────────────────────

#[test]
fn ns_lagrangian_canonical_ladder_realization() {
    let w = omega();

    let probes = vec![
        lag_state([0, 0, 0]),
        lag_state([1, 0, 0]),
        lag_state([0, 1, 0]),
        lag_state([0, 0, 1]),
        lag_state([1, 1, 0]),
        lag_state([2, 0, 0]),
        lag_state([1, 1, 1]),
    ];

    // (a) The CCR `[P_i, Q_j] = −i δ_ij` (`comm_lagP_lagQ`) — exact on every
    //     probe, diagonal and off-diagonal.
    for i in 0..3u32 {
        for j in 0..3u32 {
            let pq = ham(pmul(&p_poly(i), &q_poly(j)));
            let qp = ham(pmul(&q_poly(j), &p_poly(i)));
            for (k, psi) in probes.iter().enumerate() {
                let mut d = pq.apply(psi);
                d.scale_and_add(&qp.apply(psi), cx(-1.0, 0.0));
                if i == j {
                    // [P,Q] = −i  ⇒  d + i·ψ = 0
                    d.scale_and_add(psi, cx(0.0, 1.0));
                    let n = d.norm();
                    assert!(
                        n < 1e-10,
                        "[P_{i},Q_{i}] = −i must hold exactly on probe {k}: residual {n:.3e}"
                    );
                } else {
                    let n = d.norm();
                    assert!(
                        n < 1e-10,
                        "[P_{i},Q_{j}] = 0 must hold exactly on probe {k}: residual {n:.3e}"
                    );
                }
            }
        }
    }

    // (b) The oscillator identity `½ΣP² + νΣQ² = ω(N + 3/2)`
    //     (`lagT_eq_number` / `lagCan_secondOrder_eq`) — exact on probes.
    let osc = ham(lag_osc_poly());
    let numb = number_op();
    let konst = ham(pconst(1.5 * w));
    for (k, psi) in probes.iter().enumerate() {
        let lhs = osc.apply(psi);
        let mut rhs = numb.apply(psi);
        rhs.scale_and_add(&konst.apply(psi), cx(1.0, 0.0));
        let mut scaled = QuantumState::zero();
        scaled.scale_and_add(&rhs, cx(w, 0.0));
        let mut d = lhs;
        d.scale_and_add(&scaled, cx(-1.0, 0.0));
        let n = d.norm();
        assert!(
            n < 1e-10,
            "½ΣP²+νΣQ² = ω(N+3/2) must hold on probe {k}: residual {n:.3e}"
        );
    }

    // (c) The completed square (the Kato–Rellich input of `hFull_..._of_drive_eq_P`
    //     and the source of the spectral shift `−½|f|²`):
    //     `h + ½|f|² = ½Σ(P_m+f_m)² + νΣQ²`.
    let full = padd(&lag_osc_poly(), &lag_drift_poly(&FORCE));
    let f2: f64 = FORCE.iter().map(|x| x * x).sum();
    let lhs_poly = padd(&full, &pconst(0.5 * f2));
    let mut square: Poly = Vec::new();
    for m in 0..3u32 {
        let pm = p_poly(m);
        square = padd(&square, &pmul(&pm, &pm));
        square = padd(&square, &pscale(&pm, cx(2.0 * FORCE[m], 0.0)));
        square = padd(&square, &pconst(FORCE[m] * FORCE[m]));
    }
    let rhs_poly = padd(&pscale(&square, cx(0.5, 0.0)), &lag_osc_poly());
    for (k, psi) in probes.iter().enumerate() {
        let mut d = ham(lhs_poly.clone()).apply(psi);
        d.scale_and_add(&ham(rhs_poly.clone()).apply(psi), cx(-1.0, 0.0));
        let n = d.norm();
        assert!(
            n < 1e-10,
            "h + ½|f|² = ½Σ(P+f)² + νΣQ² must hold on probe {k}: residual {n:.3e}"
        );
    }

    // (d) The truncated one-parcel matrix: Hermitian (finite-section shadow of
    //     `lagCan_esa`), and its spectrum is the closed-form displaced one.
    let n_max = 5u32;
    let basis = basis_states(n_max);
    assert_eq!(basis.len(), ((n_max + 1) as usize).pow(3));

    // f = 0: the matrix is diagonal in the occupation basis with EXACTLY the
    // oscillator levels — every eigenvalue matches `ω(n + 3/2)` (no truncation
    // error: the identity is operator-level, checked in (b)).
    let h0 = ham(lag_osc_poly());
    let m0 = ham_matrix(&h0, &basis);
    assert_hermitian(&m0, "one-parcel ½ΣP²+νΣQ²");
    let ev0 = sorted_eigs(&m0);
    let mut expected0 = analytic_levels(&triples_up_to(n_max));
    let shift_only = 0.5 * FORCE.iter().map(|x| x * x).sum::<f64>();
    for e in expected0.iter_mut() {
        *e += shift_only; // undo the force shift: this run has f = 0
    }
    let err0 = ev0
        .iter()
        .zip(expected0.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err0 < 1e-9,
        "f = 0 spectrum must be exactly {{ω(n+3/2)}}: max error {err0:.3e}"
    );

    // f ≠ 0: the low spectrum is the displaced set `ω(n+3/2) − ½|f|²`
    // (complete the square, check (c)); compare the lowest 10 levels.
    let hf = lag_one_parcel(&FORCE);
    let mf = ham_matrix(&hf, &basis);
    assert_hermitian(&mf, "one-parcel h = ½ΣP²+νΣQ²+Σf·P");
    let evf = sorted_eigs(&mf);
    let expectedf = analytic_levels(&triples_up_to(n_max));
    let low = 10.min(evf.len());
    let errf = evf[..low]
        .iter()
        .zip(expectedf[..low].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        errf < 2e-3,
        "displaced spectrum must be ω(n+3/2)−½|f|² on the lowest {low} levels: \
         max error {errf:.3e}"
    );
    // Min-max: the truncated spectrum never dips below the exact ground.
    assert!(
        evf[0] >= expectedf[0] - 1e-9,
        "truncated ground {} must be ≥ exact ground {}",
        evf[0],
        expectedf[0]
    );
    // Bounded below and real (Hermitian truncation).
    assert!(evf.iter().all(|e| e.is_finite()));

    eprintln!(
        "ns_lagrangian_canonical_ladder_realization: [P,Q]=−i, ω(N+3/2) exact; \
         f=0 max spectral error {err0:.2e}, f≠0 lowest-10 max error {errf:.2e}, \
         ground {:.6} (analytic {:.6})",
        evf[0],
        expectedf[0]
    );
}

// ── 2. the outer-Fock enclosure and the parcel sectors ──────────────────────

#[test]
fn ns_lagrangian_dGamma_enclosure_parcel_sectors() {
    let w = omega();
    let n_max = 2u32;
    let labels = basis_inners(n_max);
    let basis = basis_states(n_max);
    let h1 = ham_matrix(&lag_one_parcel(&FORCE), &basis);
    assert_hermitian(&h1, "one-parcel h (n_max=2)");
    let h = outer_enclose(&h1, &labels);

    // (a) Doctrine: every term is `C†(e_i) A(e_j)` — outer creator leftmost,
    //     outer annihilator rightmost, nonzero coefficient.
    for (c, ops) in &h.terms {
        assert_eq!(ops.len(), 2, "enclosure terms are exactly two operators");
        assert!(
            matches!(ops[0], Operator::OuterBosonCreate(_)),
            "leftmost op must be an outer creator"
        );
        assert!(
            matches!(ops[1], Operator::OuterBosonAnnihilate(_)),
            "rightmost op must be an outer annihilator"
        );
        assert!(c.norm_sqr() > 0.0, "enclosure drops zero coefficients");
    }

    // (b) `H|Ω⟩ = 0` identically — the annihilation on the right kills the
    //     outer vacuum exactly (doctrine clause 1: the FULL Hamiltonian, not a
    //     normal-ordered part of it, annihilates the vacuum).
    let hv = h.apply(&QuantumState::vacuum());
    assert!(
        hv.norm() < 1e-12,
        "H|Ω⟩ must vanish identically: ‖H|Ω⟩‖ = {:.3e}",
        hv.norm()
    );

    // (c) On the one-parcel sector the enclosure acts as the one-particle
    //     matrix: `H|e_j⟩ = Σ_i h_ij |e_i⟩`, and the sector is preserved.
    for j in [0usize, 5, 13, 26] {
        let out = h.apply(&basis[j]);
        let mut expected = QuantumState::zero();
        for (i, bi) in basis.iter().enumerate() {
            let hij = h1[(i, j)];
            if hij.norm_sqr() > 1e-30 {
                expected.scale_and_add(bi, hij);
            }
        }
        let mut d = out.clone();
        d.scale_and_add(&expected, cx(-1.0, 0.0));
        let n = d.norm();
        assert!(
            n < 1e-10,
            "one-parcel action must equal the matrix h (j={j}): residual {n:.3e}"
        );
        for (st, amp) in &out.components {
            let universes: u32 = st.bosonic.values().sum();
            assert_eq!(universes, 1, "one-parcel sector must be preserved");
            assert!(amp.norm() > 0.0);
        }
    }

    // (d) The two-parcel sector: the n-fold sums `λ_a + λ_b` — the
    //     non-interacting content of `lagOne_dGamma_esa` (parcels do not
    //     interact in the `dΓ(h)` extension).
    let l1 = sorted_eigs(&h1);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for a in 0..labels.len() {
        for b in a..labels.len() {
            pairs.push((a, b));
        }
    }
    let pair_states: Vec<QuantumState> = pairs
        .iter()
        .map(|&(a, b)| {
            let mut s = QuantumState::vacuum()
                .apply(&Operator::OuterBosonCreate(labels[a].clone()))
                .apply(&Operator::OuterBosonCreate(labels[b].clone()));
            if a == b {
                // C†(e)²|Ω⟩ has norm √2 — normalize.
                let mut nrm = QuantumState::zero();
                nrm.scale_and_add(&s, cx(1.0 / 2.0f64.sqrt(), 0.0));
                s = nrm;
            }
            let nn = QuantumState::inner_product(&s, &s).re;
            assert!(
                (nn - 1.0).abs() < 1e-12,
                "two-parcel basis state ({a},{b}) must be normalized, got {nn}"
            );
            s
        })
        .collect();

    let k = pairs.len();
    let mut m2 = DMatrix::<Complex64>::zeros(k, k);
    for (j, pj) in pair_states.iter().enumerate() {
        let hpj = h.apply(pj);
        // Sector purity: two universes in, two universes out.
        for st in hpj.components.keys() {
            let universes: u32 = st.bosonic.values().sum();
            assert_eq!(universes, 2, "two-parcel sector must be preserved");
        }
        for (i, pi) in pair_states.iter().enumerate() {
            m2[(i, j)] = QuantumState::inner_product(pi, &hpj);
        }
    }
    assert_hermitian(&m2, "two-parcel sector matrix");

    let ev2 = sorted_eigs(&m2);
    let mut expected2: Vec<f64> = Vec::new();
    for a in 0..l1.len() {
        for b in a..l1.len() {
            expected2.push(l1[a] + l1[b]);
        }
    }
    expected2.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(ev2.len(), expected2.len());
    let err2 = ev2
        .iter()
        .zip(expected2.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err2 < 1e-8,
        "two-parcel spectrum must be the pairwise sums λ_a+λ_b: max error {err2:.3e}"
    );

    // (e) Ground-state doctrine: `h ≻ 0` (the analytic ground
    //     `ω(3/2) − ½|f|² > 0`), the vacuum energy is exactly 0, so the outer
    //     vacuum is the ground of every sector and the mass gap is `λ_min(h)`.
    let analytic_ground = 1.5 * w - 0.5 * FORCE.iter().map(|x| x * x).sum::<f64>();
    assert!(
        (l1[0] - analytic_ground).abs() < 1e-3,
        "one-parcel ground {} must ≈ analytic {analytic_ground}",
        l1[0]
    );
    assert!(l1[0] > 0.0, "h must be positive so the vacuum is the ground");
    assert!(ev2[0] >= 2.0 * l1[0] - 1e-8, "two-parcel ground = 2λ_min");

    eprintln!(
        "ns_lagrangian_dGamma_enclosure: vacuum annihilation exact, one-parcel λ₀={:.6} \
         (analytic {analytic_ground:.6}), two-parcel {k}×{k} sector = pairwise sums \
         (max err {err2:.2e})",
        l1[0]
    );
}

// ── 3. Hashimoto shift-invert selection on the Lagrangian operator ──────────

#[test]
fn ns_lagrangian_sirk_hashimoto_selection() {
    let n_max = 2u32;
    let labels = basis_inners(n_max);
    let basis = basis_states(n_max);
    let h1 = ham_matrix(&lag_one_parcel(&FORCE), &basis);
    let l1 = sorted_eigs(&h1);
    let h = outer_enclose(&h1, &labels);

    // Start in the one-parcel sector with genuine overlap on the ground.
    let mut start = lag_state([0, 0, 0]);
    start.scale_and_add(&lag_state([1, 0, 0]), cx(0.5, 0.0));
    start.scale_and_add(&lag_state([1, 1, 0]), cx(0.25, 0.0));
    let start = {
        let n = start.norm();
        let mut s = QuantumState::zero();
        s.scale_and_add(&start, cx(1.0 / n, 0.0));
        s
    };

    let res = solve_forward_sirk_with_opts(&h, &start, &shifts(4), &best_device(), None, &sirk_opts())
        .expect("Lagrangian SIRK solve");
    assert_hermitian(&res.h_proj, "Lagrangian projected Hamiltonian");
    let n = res.h_proj.nrows();
    assert!(n >= 2, "projected Lagrangian matrix must be ≥ 2×2, got {n}");

    // Ritz values: finite, within the one-parcel spectral interval (min-max:
    // the compression cannot leave `[λ_min, λ_max]` of the sector it lives
    // in — number conservation keeps the Krylov space in sector 1), and the
    // shift-invert Krylov captures the analytic ground.
    let ritz = res.ritz_values();
    assert!(ritz.iter().all(|r| r.is_finite()), "Ritz values must be finite");
    assert!(
        ritz[0] >= l1[0] - 1e-8,
        "Ritz values cannot dip below λ_min(h) = {}: got {}",
        l1[0],
        ritz[0]
    );
    assert!(
        (ritz[0] - l1[0]).abs() < 1e-3,
        "shift-invert Krylov must capture the ground {}: got {ritz0}",
        l1[0],
        ritz0 = ritz[0]
    );
    assert!(
        ritz[ritz.len() - 1] <= l1[l1.len() - 1] + 1e-8,
        "Ritz values must stay ≤ λ_max(h)"
    );

    // The Hashimoto selection battery (`ChapterHashimotoComplexShifts`):
    // for non-real γ the resolvent exists, obeys the identity and the
    // `1/|Im γ|` bound, and `λ = γ − 1/μ` recovers the spectrum.
    let a = res.h_proj.clone();
    let eig = a.clone().symmetric_eigen();
    let lambdas: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    let eigvecs = eig.eigenvectors;

    for s in [0.5, 1.0, 2.0] {
        let gamma = Complex64::new(0.0, s);
        let gi = DMatrix::<Complex64>::identity(n, n) * gamma;
        let r = (gi.clone() - a.clone())
            .try_inverse()
            .expect("(γI−A) must be invertible for Im γ ≠ 0");

        let check = (gi.clone() - a.clone()) * r.clone();
        let mut id = DMatrix::<Complex64>::identity(n, n);
        id -= &check;
        assert!(
            id.norm() < 1e-10,
            "resolvent identity must hold: ‖(γI−A)R − I‖ = {:.2e}",
            id.norm()
        );

        let sv = r.clone().svd(true, true).singular_values;
        let norm_r = sv[0];
        assert!(
            norm_r <= 1.0 / s + 1e-9,
            "‖(γI−A)⁻¹‖ ≤ 1/|Im γ| = {} must hold: got {norm_r:.6}",
            1.0 / s
        );
        let expected_max = 1.0
            / lambdas
                .iter()
                .map(|&l| (l * l + s * s).sqrt())
                .fold(f64::INFINITY, f64::min);
        assert!(
            (norm_r - expected_max).abs() < 1e-6,
            "‖(γI−A)⁻¹‖ = {norm_r:.8} must equal 1/min|γ−λ_j| = {expected_max:.8}"
        );

        for (j, vj_col) in eigvecs.column_iter().enumerate() {
            let vj = vj_col.into_owned();
            let rvj = &r * &vj;
            let mu = 1.0 / (gamma - Complex64::new(lambdas[j], 0.0));
            let mut residual = rvj;
            residual -= &(vj * mu);
            assert!(
                residual.norm() < 1e-8,
                "R v_j = 1/(γ−λ_j) v_j must hold: ‖·‖ = {:.2e}",
                residual.norm()
            );
            let recovered = gamma - (1.0 / mu);
            assert!(
                (recovered.re - lambdas[j]).abs() < 1e-8,
                "λ_j = γ − 1/μ_j must recover the projected spectrum: {:.8} vs {:.8}",
                recovered.re,
                lambdas[j]
            );
            // Recovered spectrum lives in the one-parcel interval.
            assert!(
                recovered.re >= l1[0] - 1e-8 && recovered.re <= l1[l1.len() - 1] + 1e-8,
                "recovered λ {} must lie in [{}, {}]",
                recovered.re,
                l1[0],
                l1[l1.len() - 1]
            );
        }
    }

    // SIRK's own shifts are non-real — the shifts for which the bound holds.
    let sirk_shifts = shifts(4);
    assert!(
        sirk_shifts.iter().all(|g| g.re == 0.0 && g.im != 0.0),
        "SIRK shifts must be non-real, got {sirk_shifts:?}"
    );

    eprintln!(
        "ns_lagrangian_sirk_hashimoto: projected {}×{} Hermitian, Ritz ground {:.6} \
         vs λ_min {:.6}; resolvent bound/identity/recovery all hold at γ = i·0.5, i·1, i·2",
        n,
        n,
        ritz[0],
        l1[0]
    );
}

// ── 4. the autonomous Lagrangian flow: Ehrenfest + conservation ─────────────

#[test]
fn ns_lagrangian_ehrenfest_unitary_flow() {
    let n_max = 2u32;
    let labels = basis_inners(n_max);
    let basis = basis_states(n_max);
    let h1 = ham_matrix(&lag_one_parcel(&FORCE), &basis);
    let h = outer_enclose(&h1, &labels);

    // A genuine one-parcel superposition (normalized).
    let mut psi0 = lag_state([1, 0, 0]);
    psi0.scale_and_add(&lag_state([0, 1, 0]), cx(0.5, 0.0));
    psi0.scale_and_add(&lag_state([0, 0, 1]), cx(0.25, 0.0));
    psi0.scale_and_add(&lag_state([1, 1, 0]), cx(0.125, 0.0));
    {
        let n = psi0.norm();
        let mut s = QuantumState::zero();
        s.scale_and_add(&psi0, cx(1.0 / n, 0.0));
        psi0 = s;
    }

    // (a) Exact Heisenberg identities (operator-level, as in the ladder test
    //     but through the ENCLOSURE): on the one-parcel sector `dΓ(h)` acts
    //     as `h`, so
    //       d⟨Q_i⟩/dt = i⟨[H,Q_i]⟩ = ⟨P_i⟩ + f_i,
    //       d⟨P_i⟩/dt = i⟨[H,P_i]⟩ = −2ν⟨Q_i⟩,
    //     the harmonic-oscillator equations at frequency ω = √(2ν).
    for i in 0..3u32 {
        let qi = ham(q_poly(i));
        let pi = ham(p_poly(i));
        let dq = heisenberg(&h, &qi, &psi0);
        let p_exp = expect(&psi0, &pi);
        let target = p_exp + cx(FORCE[i], 0.0);
        assert!(
            (dq - target).norm() < 1e-9,
            "d⟨Q_{i}⟩/dt = ⟨P_{i}⟩ + f_{i}: got {dq:.6}, want {target:.6}"
        );
        assert!(dq.im.abs() < 1e-9, "d⟨Q_{i}⟩/dt must be real");

        let dp = heisenberg(&h, &pi, &psi0);
        let q_exp = expect(&psi0, &qi);
        let target_p = -2.0 * NU * q_exp;
        assert!(
            (dp - target_p).norm() < 1e-9,
            "d⟨P_{i}⟩/dt = −2ν⟨Q_{i}⟩: got {dp:.6}, want {target_p:.6}"
        );
        assert!(dp.im.abs() < 1e-9, "d⟨P_{i}⟩/dt must be real");
    }

    // (b) The SIRK-restarted flow of the ENCLOSED operator: norm, energy and
    //     the parcel-sector weights conserved (number conservation of dΓ).
    let n0 = psi0.norm();
    let e0 = energy(&h, &psi0);
    assert!((n0 - 1.0).abs() < 1e-12, "start must be normalized");

    let opts = sirk_opts();
    let psi_t = evolve_restarted(&h, &psi0, 0.05, 2, 6, &best_device(), None, &opts)
        .expect("Lagrangian SIRK evolution");
    let n_t = psi_t.norm();
    let e_t = energy(&h, &psi_t);
    assert!(
        (n_t - n0).abs() < 1e-8,
        "norm must be conserved: {:.2e}",
        (n_t - n0).abs()
    );
    assert!(
        (e_t - e0).abs() < 1e-6,
        "energy must be conserved: {:.2e}",
        (e_t - e0).abs()
    );
    let mut w0 = 0.0;
    let mut w1 = 0.0;
    let mut w2 = 0.0;
    for (st, amp) in &psi_t.components {
        let universes: u32 = st.bosonic.values().sum();
        let p = amp.norm_sqr();
        match universes {
            0 => w0 += p,
            1 => w1 += p,
            2 => w2 += p,
            _ => panic!("parcel-number sector {universes} must not be populated"),
        }
    }
    assert!(w0 < 1e-10, "vacuum sector must stay empty: weight {w0:e}");
    assert!(w2 < 1e-10, "two-parcel sector must stay empty: weight {w2:e}");
    assert!((w1 - 1.0).abs() < 1e-8, "one-parcel sector weight: {w1}");

    // (c) Ehrenfest along the flow: a short-time finite difference of ⟨Q_0⟩
    //     against ⟨P_0⟩ + f_0 at t = 0 (forward difference, O(dt) error).
    let dt = 0.005;
    let psi_dt = evolve_restarted(&h, &psi0, dt, 1, 8, &best_device(), None, &opts)
        .expect("short Lagrangian SIRK step");
    let q0 = ham(q_poly(0));
    let p0 = ham(p_poly(0));
    let q_start = expect(&psi0, &q0).re;
    let q_end = expect(&psi_dt, &q0).re;
    let qdot_fd = (q_end - q_start) / dt;
    let qdot_exact = heisenberg(&h, &q0, &psi0).re;
    let target = expect(&psi0, &p0).re + FORCE[0];
    assert!(
        (qdot_exact - target).abs() < 1e-9,
        "commutator Ehrenfest must hold: {qdot_exact:.6} vs {target:.6}"
    );
    assert!(
        (qdot_fd - target).abs() < 5e-3,
        "finite-difference d⟨Q_0⟩/dt must ≈ ⟨P_0⟩+f_0: fd {qdot_fd:.6} vs {target:.6}"
    );

    eprintln!(
        "ns_lagrangian_ehrenfest: Heisenberg identities exact to 1e-9; SIRK flow: \
         ‖ψ‖ {n0:.6}→{n_t:.6}, ⟨H⟩ {e0:.6}→{e_t:.6}, sector weights (0,1,2)=({w0:.1e},\
         {w1:.6},{w2:.1e}); fd d⟨Q₀⟩/dt={qdot_fd:.6} vs ⟨P₀⟩+f₀={target:.6}"
    );
}

// ── 5. the volume constraint and its penalty ────────────────────────────────

/// `volumePoly = det F − 1` in the diagonal-stretching ansatz
/// `F = diag(1 + Q_i)` — exact for that ansatz, since all off-diagonal
/// entries vanish: `det F = Π_i (1 + Q_i)`.
fn volume_poly_chain() -> Poly {
    let mut prod = padd(&pconst(1.0), &q_poly(0));
    prod = pmul(&prod, &padd(&pconst(1.0), &q_poly(1)));
    prod = pmul(&prod, &padd(&pconst(1.0), &q_poly(2)));
    padd(&prod, &pconst(-1.0))
}

/// The same polynomial by the hand expansion
/// `ΣQ_i + Σ_{i<j} Q_iQ_j + Q_0Q_1Q_2` (independent construction).
fn volume_poly_expanded() -> Poly {
    let q: Vec<Poly> = (0..3u32).map(q_poly).collect();
    let mut p = padd(&padd(&q[0], &q[1]), &q[2]);
    p = padd(&p, &pmul(&q[0], &q[1]));
    p = padd(&p, &pmul(&q[1], &q[2]));
    p = padd(&p, &pmul(&q[0], &q[2]));
    p = padd(&p, &pmul(&pmul(&q[0], &q[1]), &q[2]));
    p
}

#[test]
fn ns_lagrangian_volume_constraint_penalty() {
    let w = omega();
    let vol = volume_poly_chain();
    let vol_h = ham(vol.clone());
    let vol_sq_h = ham(pmul(&vol, &vol));

    let probes = vec![
        lag_state([0, 0, 0]),
        lag_state([1, 0, 0]),
        lag_state([0, 1, 1]),
        lag_state([1, 2, 0]),
        lag_state([2, 1, 1]),
    ];

    // (a) The two independent constructions of `det F − 1` agree exactly.
    let vol_exp_h = ham(volume_poly_expanded());
    for (k, psi) in probes.iter().enumerate() {
        let mut d = vol_h.apply(psi);
        d.scale_and_add(&vol_exp_h.apply(psi), cx(-1.0, 0.0));
        let n = d.norm();
        assert!(
            n < 1e-12,
            "Π(1+Q_i)−1 must equal ΣQ+ΣQQ+Q₀Q₁Q₂ on probe {k}: residual {n:.3e}"
        );
    }

    // The constraint surface passes through the origin: the parity-even
    // Gaussian sees ⟨volumePoly⟩ = 0 at F = I (odd parts integrate out, the
    // quadratic cross terms factor into ⟨Q⟩⟨Q⟩ = 0).
    let origin_exp = expect(&lag_state([0, 0, 0]), &vol_h);
    assert!(
        origin_exp.norm() < 1e-12,
        "⟨volumePoly⟩ must vanish at the identity deformation: {:.3e}",
        origin_exp.norm()
    );

    // (b) The penalty family `H_κ = ½ΣP² + νΣQ² + κ·volumePoly²` — the
    //     squared-form (Weyl sum-of-squares) structure of `lagSectorHam`
    //     (`weylPoly_esa`) / the penalty `V_κ` of `NsLagrangianDetFL`:
    //     Hermitian, PSD (finite-section shadow of the positivity theorem),
    //     ground energy monotone in κ, and the ground state localizing onto
    //     `volumePoly = 0` as κ grows.
    let n_max = 4u32;
    let basis = basis_states(n_max);
    let osc_h = ham(lag_osc_poly());

    let mut ground_e: Vec<f64> = Vec::new();
    let mut ground_vol2: Vec<f64> = Vec::new();
    for &kap in &[0.0, 1.0, 100.0] {
        let mut p = lag_osc_poly();
        if kap != 0.0 {
            p = padd(&p, &pscale(&pmul(&vol, &vol), cx(kap, 0.0)));
        }
        let hk = ham(p);
        let mk = ham_matrix(&hk, &basis);
        assert_hermitian(&mk, &format!("H_κ (κ={kap})"));
        let eig = mk.clone().symmetric_eigen();
        let (mut imin, mut emin) = (0usize, f64::INFINITY);
        for (i, val) in eig.eigenvalues.iter().enumerate() {
            if *val < emin {
                emin = *val;
                imin = i;
            }
        }
        assert!(
            emin >= -1e-9,
            "H_κ must be a positive sum of squares (κ={kap}): E₀ = {emin:.3e}"
        );

        // Rebuild the ground vector and measure ⟨volumePoly²⟩ (vol is
        // self-adjoint, so ⟨vol²⟩ = ‖vol ψ‖²).
        let col = eig.eigenvectors.column(imin).into_owned();
        let mut gs = QuantumState::zero();
        for (i, amp) in col.iter().enumerate() {
            if amp.norm() > 1e-14 {
                gs.scale_and_add(&basis[i], *amp);
            }
        }
        let v2 = QuantumState::inner_product(&gs, &vol_sq_h.apply(&gs)).re;

        // Exact energy decomposition: E₀ = ⟨osc⟩ + κ⟨vol²⟩.
        let e_check = energy(&osc_h, &gs) + kap * v2;
        assert!(
            (e_check - emin).abs() < 1e-8,
            "E₀ = ⟨osc⟩ + κ⟨vol²⟩ decomposition must hold (κ={kap}): \
             {e_check:.9} vs {emin:.9}"
        );
        assert!((v2 - 0.0).max(0.0) < 1.0, "⟨vol²⟩ must be finite");
        ground_e.push(emin);
        ground_vol2.push(v2);
    }
    assert!(
        ground_e[0] <= ground_e[1] + 1e-9 && ground_e[1] <= ground_e[2] + 1e-9,
        "ground energy must be monotone in κ: {ground_e:?}"
    );
    assert!(
        ground_vol2[2] < 0.3 * ground_vol2[0],
        "the κ = 100 ground must localize on volumePoly = 0: ⟨vol²⟩ {:.6} → {:.6}",
        ground_vol2[0],
        ground_vol2[2]
    );

    // (c) SIRK/Hashimoto on the constrained Hamiltonian: Hermitian projection,
    //     PSD Ritz values, and energy conservation of the flow.
    let kap = 100.0;
    let mut p = lag_osc_poly();
    p = padd(&p, &pscale(&pmul(&vol, &vol), cx(kap, 0.0)));
    let hk = ham(p);
    let mut start = lag_state([0, 0, 0]);
    start.scale_and_add(&lag_state([1, 0, 0]), cx(0.5, 0.0));
    {
        let n = start.norm();
        let mut s = QuantumState::zero();
        s.scale_and_add(&start, cx(1.0 / n, 0.0));
        start = s;
    }
    let res =
        solve_forward_sirk_with_opts(&hk, &start, &shifts(4), &best_device(), None, &sirk_opts())
            .expect("constrained Lagrangian SIRK solve");
    assert_hermitian(&res.h_proj, "constrained projected Hamiltonian");
    let ritz = res.ritz_values();
    assert!(ritz.iter().all(|r| r.is_finite()), "ritz must be finite");
    assert!(
        ritz[0] >= -1e-6,
        "PSD Hamiltonian must have PSD Ritz values: ritz[0] = {}",
        ritz[0]
    );

    let e0 = energy(&hk, &start);
    let psi_t = evolve_restarted(&hk, &start, 0.05, 2, 6, &best_device(), None, &sirk_opts())
        .expect("constrained SIRK evolution");
    let e_t = energy(&hk, &psi_t);
    let n_t = psi_t.norm();
    assert!((n_t - 1.0).abs() < 1e-8, "norm must be conserved: {n_t}");
    assert!(
        (e_t - e0).abs() < 1e-6,
        "constrained energy must be conserved: {:.2e}",
        (e_t - e0).abs()
    );

    eprintln!(
        "ns_lagrangian_volume_constraint_penalty: identity/expanded volumePoly agree; \
         E₀(κ) = {ground_e:?} monotone, ⟨vol²⟩₀ {:.6} → {:.6} (κ=100), \
         decomposition and SIRK conservation hold",
        ground_vol2[0], ground_vol2[2]
    );
}
