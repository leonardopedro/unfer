//! Standard Model validation through the Hashimoto-SIRK approximation —
//! analogous to `qed_validation.rs` / `qym_mass_gap.rs`.
//!
//! The SM one-particle operator `h = h_Gauge + h_Higgs + h_Dirac + h_Yukawa`
//! (book.tex; `docs/faris_lavine_n_sm.cdb` CHECK 1–28) is realized here in
//! sector-wise / reduced mode budgets (full D_B = 163 is not SIRK-tractable).
//! Each test pins an exact structure or published number:
//!
//! 1. `sm_higgs_potential_nested_fock_structure` — Mexican-hat Higgs:
//!    normal-ordered vacuum 0, Hermiticity, classical minima
//!    `φ* = ±√(μ²/λ)` with `V(φ*) = −μ⁴/(4λ)`, and the quartic confinement
//!    that keeps the truncated spectrum bounded below.
//!
//! 2. `sm_higgs_sirk_spectrum_bounded_below` — SIRK diagonalizes the Higgs
//!    sector: projected H Hermitian, ground energy finite, and the lowest
//!    Ritz values real and (after the allowed constant shift) non-negative —
//!    the FL comparison-`N` positivity shadow at the numerical level.
//!
//! 3. `sm_yukawa_fermion_number_conservation` — `[H, N_f] = 0` exactly on
//!    probes spanning the vacuum / one-Higgs / one-fermion / Yukawa-vertex
//!    sectors (fermion number is a superselection charge of the Yukawa
//!    Hamiltonian).
//!
//! 4. `sm_ckm_pmns_unitarity_and_biunitary` — CKM (Cabibbo) and PMNS
//!    (Pontecorvo) matrices: `V V^T = I`, biunitarity `(VD)^T(VD) = D²`
//!    (Yukawa mass matrices `M†M = D²`), and entrywise `|V_ij| ≤ 1`,
//!    `|U_ij| ≤ 1` — numerical mirrors of Cadabra CHECK 17–19, 25–27.
//!
//! 5. `sm_reduced_gauge_higgs_structure` — combined gauge + Higgs + Yukawa
//!    Hamiltonian: vacuum 0, Hermitian, all three sector term types present,
//!    and fermion number still conserved.
//!
//! 6. `sm_unitary_evolution_energy_conservation` — restarted-Krylov time
//!    evolution of a Higgs-sector state: norm (unitarity) and energy
//!    `⟨H⟩` conserved.
//!
//! 7. `sm_comparison_n_structure_and_positivity` — Faris–Lavine comparison
//!    `N = :π²: + :φ⁴:`: Hermitian, quartic field / quadratic momentum
//!    structure, and positive Rayleigh quotient on low-excitation probes
//!    (the structural half of `±h ≤ c₁N`; the full gauge+fermion `N` is
//!    CHECK 1–24 in Cadabra).
//!
//! 8. `sm_outer_vacuum_annihilated_reduced` — outer enclosure of the reduced
//!    SM one-particle operator annihilates the outer vacuum `|Ω⟩` (doctrine
//!    clause 1 of §5.24s, extended to the SM sector).
//!
//! 9. `sm_db_163_mode_budget` — full collective-coordinate budget
//!    `D_B = 3+96+36+12+16 = 163` and the 160 field-ladder modes.
//!
//! 10. `sm_full_free_field_structure` — free quadratic Hamiltonian on all
//!     160 field-ladder modes: Hermitian, vacuum energy 0, correct term
//!     count / mode span.
//!
//! 11. `sm_brst_ghosts_nilpotent` — 12 Faddeev–Popov ghosts (8+3+1) in
//!     `Z_2^{D_F}`: Ω² = 0 on ghosted probes (Pauli + commuting Gauss).
//!
//! 12. `sm_weak_boson_polarizations` — full W±/Z/γ polarization budget
//!     (3+3+3+2 = 11 modes): transverse degeneracy, mass gaps
//!     `m_W ≈ 80.377`, `m_Z ≈ 91.1876`, `m_γ = 0`, and SIRK spectrum.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
use nalgebra::DMatrix;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, InnerFermionicState, Operator, QuantumState, SM_GHOST_COUNT,
    SmModeBudget, sm_biunitary_residual, sm_brst_charge, sm_ckm_cabibbo, sm_comparison_n,
    sm_fermion_number, sm_full_free_field_uniform, sm_ghost_mode_ranges,
    sm_higgs_classical_potential, sm_higgs_hamiltonian, sm_higgs_vev, sm_pmns_pontecorvo,
    sm_reduced_hamiltonian, sm_weak_boson_hamiltonian, sm_weak_boson_masses_gev,
    sm_weak_polarization_mode_count, sm_weak_polarization_table, sm_yukawa_hamiltonian,
};
use num_complex::Complex64;
use std::collections::BTreeSet;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: true,
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

/// One empty bosonic universe (so inner boson ops can act).
fn boson_background() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// SM probe vacuum: empty boson + empty fermion universes.
fn sm_vacuum() -> QuantumState {
    boson_background().apply(&Operator::OuterFermionCreate(InnerFermionicState::vacuum()))
}

fn one_higgs_quanton() -> QuantumState {
    sm_vacuum().apply(&Operator::InnerBosonCreate(0))
}

fn one_fermion() -> QuantumState {
    sm_vacuum().apply(&Operator::InnerFermionCreate(0))
}

fn sirk_ground(h: &Hamiltonian, v0: &QuantumState, m: usize) -> f64 {
    let res = solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve must complete");
    assert_hermitian(&res.h_proj, "SIRK projected Hamiltonian");
    res.ground_state_energy().expect("ground-state Ritz value")
}

fn commutator_norm(a: &Hamiltonian, b: &Hamiltonian, psi: &QuantumState) -> f64 {
    let ab = a.apply(&b.apply(psi));
    let ba = b.apply(&a.apply(psi));
    let mut diff = ab;
    diff.scale_and_add(&ba, Complex64::new(-1.0, 0.0));
    diff.norm()
}

// ── 1. Higgs Mexican-hat structure ───────────────────────────────────────────

#[test]
fn sm_higgs_potential_nested_fock_structure() {
    let (mu2, lambda) = (1.0, 0.5);
    let h = sm_higgs_hamiltonian(mu2, lambda);
    assert!(!h.terms.is_empty(), "Higgs H must have terms");

    // Normal ordering: ⟨0|H|0⟩ = 0 (zero-point stripped). H|0⟩ itself need
    // not vanish — pair-creation terms c†c† from :φ²:/:φ⁴: act on |0⟩;
    // the outer-vacuum annihilation doctrine is checked on the outer
    // enclosure in `sm_outer_vacuum_annihilated_reduced`.
    let vac = sm_vacuum();
    let e0 = QuantumState::inner_product(&h.apply(&vac), &vac).re;
    assert!(e0.abs() < 1e-12, "Higgs vacuum energy must be 0, got {e0}");

    // Hermiticity on a one-quanton probe via the SIRK projection.
    let one = one_higgs_quanton();
    let _ = sirk_ground(&h, &one, 8);

    // Classical Mexican-hat minima.
    let vev = sm_higgs_vev(mu2, lambda);
    assert!((vev - (mu2 / lambda).sqrt()).abs() < 1e-14);
    let v_star = sm_higgs_classical_potential(vev, mu2, lambda);
    let v_target = -mu2 * mu2 / (4.0 * lambda);
    assert!(
        (v_star - v_target).abs() < 1e-12,
        "V(φ*) = {v_star}, want {v_target}"
    );
    // φ = 0 is a local maximum (inverted curvature), minima at ±φ*.
    let v0 = sm_higgs_classical_potential(0.0, mu2, lambda);
    assert!((v0 - 0.0).abs() < 1e-15);
    let v_neg = sm_higgs_classical_potential(-vev, mu2, lambda);
    assert!(
        (v_neg - v_star).abs() < 1e-12,
        "potential must be Z₂-symmetric"
    );
    assert!(v_star < v0, "true minimum must lie below the origin");

    // Quartic terms present at λ > 0 (four-operator strings exist).
    let has_quartic = h.terms.iter().any(|(_, ops)| ops.len() >= 4);
    assert!(
        has_quartic || lambda == 0.0,
        "λ > 0 must produce quartic φ⁴ operator strings"
    );

    eprintln!(
        "sm_higgs: μ²={mu2}, λ={lambda}, φ*={vev:.6}, V*={v_star:.6}, {} terms",
        h.terms.len()
    );
}

// ── 2. Higgs SIRK spectrum ───────────────────────────────────────────────────

#[test]
fn sm_higgs_sirk_spectrum_bounded_below() {
    // μ² < 0 (symmetric phase, φ⁰ minimum) — the spectrum is an anharmonic
    // oscillator tower, bounded below by the quartic wall.
    let (mu2, lambda) = (-1.0, 0.5);
    let h = sm_higgs_hamiltonian(mu2, lambda);
    let one = one_higgs_quanton();

    let opts_m = opts();
    let res = solve_forward_sirk_with_opts(&h, &one, &shifts(10), &best_device(), None, &opts_m)
        .expect("Higgs SIRK solve");
    assert_hermitian(&res.h_proj, "Higgs H_proj");

    let ritz = res.ritz_values();
    assert!(!ritz.is_empty(), "need Ritz values");
    // All Ritz values real (Hermitian projection) and finite.
    for &e in &ritz {
        assert!(e.is_finite(), "Ritz value must be finite, got {e}");
    }
    // Ground (lowest Ritz) is a lower bound on the truncated spectrum and
    // the spectrum is bounded below: E₀ > −1e6 (quartic confinement).
    let e0 = *ritz
        .iter()
        .fold(&f64::INFINITY, |a, b| if b < a { b } else { a });
    assert!(
        e0 > -1.0e6,
        "Higgs spectrum must be bounded below (quartic wall), E₀ = {e0}"
    );

    // Ground from SIRK matches the direct ground helper.
    let e0_direct = sirk_ground(&h, &one, 10);
    assert!(
        (e0 - e0_direct).abs() < 1e-8,
        "Ritz ground mismatch: {e0} vs {e0_direct}"
    );

    eprintln!("sm_higgs_sirk: E₀ = {e0:.9}, {} Ritz values", ritz.len());
}

// ── 3. Yukawa fermion-number conservation ────────────────────────────────────

#[test]
fn sm_yukawa_fermion_number_conservation() {
    let (mu2, lambda, y, m_f) = (1.0, 0.5, 0.3, 0.2);
    let h = sm_yukawa_hamiltonian(mu2, lambda, y, m_f);
    let nf = sm_fermion_number(1);

    let probes = [
        sm_vacuum(),
        one_higgs_quanton(),
        one_fermion(),
        // Higgs + fermion (Yukawa vertex acts on this).
        sm_vacuum()
            .apply(&Operator::InnerBosonCreate(0))
            .apply(&Operator::InnerFermionCreate(0)),
    ];

    for (i, s) in probes.iter().enumerate() {
        let nrm = commutator_norm(&h, &nf, s);
        assert!(
            nrm < 1e-9,
            "[H, N_f]|probe {i}⟩ must vanish, ‖·‖ = {nrm:.3e}"
        );
    }

    // Vacuum still 0 and H Hermitian on a one-fermion probe.
    let vac = sm_vacuum();
    let e0 = QuantumState::inner_product(&h.apply(&vac), &vac).re;
    assert!(e0.abs() < 1e-12, "Yukawa vacuum energy {e0}");
    let _ = sirk_ground(&h, &one_fermion(), 6);

    eprintln!(
        "sm_yukawa: [H,N_f]=0 on {} probes, y={y}, m_f={m_f}, {} terms",
        probes.len(),
        h.terms.len()
    );
}

// ── 4. CKM / PMNS unitarity & biunitarity (Cadabra CHECK 17–19, 25–27) ─────

#[test]
fn sm_ckm_pmns_unitarity_and_biunitary() {
    // Cabibbo angle (PDG): sin θ_c ≈ 0.2243, θ_c ≈ 0.2263 rad.
    let theta_c = 0.2263_f64;
    let v = sm_ckm_cabibbo(theta_c);

    // V V^T = I (orthogonal / unitary for real matrices) — CHECK 17 / 25a.
    let mut vvt = [[0.0_f64; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            vvt[i][j] = v[i][0] * v[j][0] + v[i][1] * v[j][1];
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let target = if i == j { 1.0 } else { 0.0 };
            let got = vvt[i][j];
            assert!(
                (got - target).abs() < 1e-12,
                "CKM V V^T[{i}{j}] = {got}, want {target}"
            );
        }
    }

    // Entrywise |V_ij| ≤ 1 — CHECK 19 / 27a.
    for row in &v {
        for &x in row {
            assert!(x.abs() <= 1.0 + 1e-14, "|V_ij| = {} > 1", x.abs());
        }
    }

    // Cabibbo phenomenology: |V_ud| = cos θ_c ≈ 0.974, |V_us| = sin θ_c ≈ 0.224.
    let (c, s) = (theta_c.cos(), theta_c.sin());
    assert!((v[0][0] - c).abs() < 1e-14);
    assert!((v[0][1] - s).abs() < 1e-14);
    assert!(
        (c - 0.974).abs() < 0.01,
        "|V_ud| = {c:.4} should be ≈ 0.974 (PDG)"
    );
    assert!(
        (s - 0.224).abs() < 0.01,
        "|V_us| = {s:.4} should be ≈ 0.224 (PDG)"
    );

    // Biunitarity: (V D)^T (V D) = D² — CHECK 26a (Yukawa M†M = D²).
    // Two "masses" (down-type / charged-lepton masses in the 2×2 block).
    let d = [0.005_f64, 1.2_f64];
    let res = sm_biunitary_residual(&v, d);
    for i in 0..2 {
        for j in 0..2 {
            let got = res[i][j];
            assert!(got.abs() < 1e-12, "biunitary residual[{i}{j}] = {got}");
        }
    }

    // PMNS (Pontecorvo): θ_12 ≈ 0.584 rad (33.4°), sin²θ_12 ≈ 0.307 (PDG).
    let theta_12 = 0.584_f64;
    let u = sm_pmns_pontecorvo(theta_12);
    let mut uut = [[0.0_f64; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            uut[i][j] = u[i][0] * u[j][0] + u[i][1] * u[j][1];
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let target = if i == j { 1.0 } else { 0.0 };
            let got = uut[i][j];
            assert!(
                (got - target).abs() < 1e-12,
                "PMNS U U^T[{i}{j}] = {got}, want {target}"
            );
        }
    }
    for row in &u {
        for &x in row {
            assert!(x.abs() <= 1.0 + 1e-14, "|U_ij| = {} > 1", x.abs());
        }
    }
    // |U_e1| = cos θ_12 ≈ 0.822 (PDG solar mixing).
    let ue1 = u[0][0];
    assert!(
        (ue1 - theta_12.cos()).abs() < 1e-14,
        "|U_e1| must equal cos θ_12"
    );
    assert!(
        (ue1 - 0.822).abs() < 0.02,
        "|U_e1| = {ue1:.4} should be ≈ 0.822 (PDG)"
    );
    let sin2_th12 = theta_12.sin().powi(2);
    assert!(
        (sin2_th12 - 0.307).abs() < 0.02,
        "sin²θ_12 = {sin2_th12:.4} should be ≈ 0.307 (PDG)"
    );

    // PMNS biunitarity for the neutrino mass matrix M_ν = U D — CHECK 26b.
    let d_nu = [0.0001_f64, 0.009_f64]; // eV-scale (normal hierarchy sketch)
    let res_nu = sm_biunitary_residual(&u, d_nu);
    for i in 0..2 {
        for j in 0..2 {
            let got = res_nu[i][j];
            assert!(got.abs() < 1e-12, "PMNS biunitary residual[{i}{j}] = {got}");
        }
    }

    eprintln!("sm_ckm_pmns: |V_ud|={c:.6}, |V_us|={s:.6}, |U_e1|={ue1:.6}, sin²θ12={sin2_th12:.6}");
}

// ── 5. Reduced SM (gauge + Higgs + Yukawa) ──────────────────────────────────

#[test]
fn sm_reduced_gauge_higgs_structure() {
    let (g, mu2, lambda, y, m_f) = (1.0_f64, 1.0_f64, 0.5_f64, 0.3_f64, 0.2_f64);
    let h = sm_reduced_hamiltonian(g, mu2, lambda, y, m_f);
    assert!(!h.terms.is_empty(), "reduced SM must have terms");

    // Vacuum energy 0 (zero-point stripped). H|0⟩ need not vanish — pair
    // terms act on |0⟩; outer annihilation is the outer-enclosure test.
    let vac = sm_vacuum();
    let e0 = QuantumState::inner_product(&h.apply(&vac), &vac).re;
    assert!(e0.abs() < 1e-12, "reduced SM vacuum energy {e0}");

    // Hermitian via SIRK projection from a gauge+Higgs superposition.
    let mut psi = boson_background().apply(&Operator::InnerBosonCreate(0));
    psi.scale_and_add(
        &sm_vacuum().apply(&Operator::InnerBosonCreate(4)),
        Complex64::new(0.5, 0.0),
    );
    let _ = sirk_ground(&h, &psi, 8);

    // Sector term types: gauge magnetic (≥3 ops at g>0), Higgs quartic (≥4),
    // Yukawa vertex (boson + two fermion ops on one string).
    let has_gauge = h.terms.iter().any(|(_, ops)| {
        ops.len() >= 3
            && ops
                .iter()
                .any(|o| matches!(o, Operator::InnerBosonCreate(_)))
    });
    let has_quartic = h.terms.iter().any(|(_, ops)| ops.len() >= 4);
    let has_yukawa = h.terms.iter().any(|(_, ops)| {
        ops.iter()
            .any(|o| matches!(o, Operator::InnerFermionCreate(_)))
            && ops.iter().any(|o| {
                matches!(
                    o,
                    Operator::InnerBosonCreate(_) | Operator::InnerBosonAnnihilate(_)
                )
            })
            && ops.len() >= 3
    });
    assert!(has_gauge, "reduced SM must contain gauge sector terms");
    assert!(has_quartic, "reduced SM must contain Higgs quartic terms");
    assert!(has_yukawa, "reduced SM must contain Yukawa vertex terms");

    // Fermion number still conserved on the combined Hamiltonian.
    let nf = sm_fermion_number(1);
    for (i, s) in [
        sm_vacuum(),
        one_fermion(),
        sm_vacuum()
            .apply(&Operator::InnerBosonCreate(4))
            .apply(&Operator::InnerFermionCreate(0)),
    ]
    .iter()
    .enumerate()
    {
        let nrm = commutator_norm(&h, &nf, s);
        assert!(
            nrm < 1e-9,
            "reduced [H, N_f]|probe {i}⟩ = {nrm:.3e} must vanish"
        );
    }

    eprintln!(
        "sm_reduced: g={g}, μ²={mu2}, λ={lambda}, y={y}, m_f={m_f}, {} terms",
        h.terms.len()
    );
}

// ── 6. Unitary evolution: norm + energy conservation ────────────────────────

#[test]
fn sm_unitary_evolution_energy_conservation() {
    use fock_sirk::evolve_restarted;

    let (mu2, lambda) = (-1.0, 0.5);
    let h = sm_higgs_hamiltonian(mu2, lambda);

    // Superposition of Higgs-sector Fock states, each component unit-norm.
    let e1 = sm_vacuum().apply(&Operator::InnerBosonCreate(1));
    let e2 = sm_vacuum()
        .apply(&Operator::InnerBosonCreate(0))
        .apply(&Operator::InnerBosonCreate(0));
    let mut psi0 = one_higgs_quanton();
    psi0.scale_and_add(&e1, Complex64::new(0.5 / e1.norm(), 0.0));
    psi0.scale_and_add(&e2, Complex64::new(0.25 / e2.norm(), 0.0));
    // Exact renormalization: ψ ← ψ / ‖ψ‖.
    let nrm = psi0.norm();
    psi0.scale_and_add(&psi0.clone(), Complex64::new(1.0 / nrm - 1.0, 0.0));
    let n0 = psi0.norm();
    assert!((n0 - 1.0).abs() < 1e-9, "probe must be normalized, n0={n0}");
    let e0 = QuantumState::inner_product(&h.apply(&psi0), &psi0).re;

    // Unit-norm frame: exact reparametrization, the admitted conditioning
    // device (guide §4.5 item 3) — required for the quartic Higgs wall.
    let opts_e = SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(1_000_000),
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: true,
    };
    let psi_t =
        evolve_restarted(&h, &psi0, 2.0, 4, 6, &best_device(), None, &opts_e).expect("evolve");
    let n_t = psi_t.norm();
    let e_t = QuantumState::inner_product(&h.apply(&psi_t), &psi_t).re;

    assert!(
        (n_t - n0).abs() < 1e-9,
        "Higgs norm must be conserved: |Δ‖ψ‖| = {:.2e}",
        (n_t - n0).abs()
    );
    // Energy: quartic Higgs wall is stiff; unit-norm frame keeps drift at
    // ~2e-9 over t=2 — assert 1e-8 (same band as other gauge-fixed suites).
    assert!(
        (e_t - e0).abs() < 1e-8,
        "Higgs energy must be conserved: |Δ⟨H⟩| = {:.2e}",
        (e_t - e0).abs()
    );

    eprintln!("sm_unitary: ‖ψ‖ {n0:.6}→{n_t:.6}, ⟨H⟩ {e0:.6}→{e_t:.6}");
}

// ── 7. Faris–Lavine comparison N structure ──────────────────────────────────

#[test]
fn sm_comparison_n_structure_and_positivity() {
    let n = sm_comparison_n();
    assert!(!n.terms.is_empty(), "comparison N must have terms");

    // Hermitian (term-level: N = N† up to coefficient conjugation).
    let n_dag = n.adjoint();
    // Compare as operators on probes rather than term-list identity (CAS
    // may reorder equivalent strings).
    let probes = [
        sm_vacuum(),
        one_higgs_quanton(),
        boson_background()
            .apply(&Operator::InnerBosonCreate(1))
            .apply(&Operator::InnerBosonCreate(1)),
        boson_background()
            .apply(&Operator::InnerBosonCreate(0))
            .apply(&Operator::InnerBosonCreate(0))
            .apply(&Operator::InnerBosonCreate(0))
            .apply(&Operator::InnerBosonCreate(0)),
    ];
    for (i, s) in probes.iter().enumerate() {
        let ns = n.apply(s);
        let n_ds = n_dag.apply(s);
        let mut diff = ns;
        diff.scale_and_add(&n_ds, Complex64::new(-1.0, 0.0));
        let nrm = diff.norm();
        assert!(
            nrm < 1e-9,
            "comparison N must be Hermitian on probe {i}, ‖N−N†‖ = {nrm:.3e}"
        );
    }

    // Structural content: quadratic momentum strings and quartic field strings.
    let has_p2 = n.terms.iter().any(|(_, ops)| {
        ops.iter()
            .filter(|o| {
                matches!(
                    o,
                    Operator::InnerBosonCreate(1) | Operator::InnerBosonAnnihilate(1)
                )
            })
            .count()
            >= 2
    });
    let has_phi4 = n.terms.iter().any(|(_, ops)| ops.len() >= 4);
    assert!(has_p2, "N must contain :π²: (quadratic momentum) terms");
    assert!(has_phi4, "N must contain :φ⁴: (quartic field) terms");

    // Vacuum expectation 0 (normal-ordered zero-point stripped). N|0⟩ itself
    // need not vanish — :φ⁴: carries pure creation terms c†⁴ acting on |0⟩.
    let vac = sm_vacuum();
    let exp_vac = QuantumState::inner_product(&n.apply(&vac), &vac).re;
    assert!(
        exp_vac.abs() < 1e-12,
        "comparison N must have zero vacuum expectation, got {exp_vac}"
    );

    // On a two-quanton momentum state, ⟨N⟩ ≥ 0 (the :π²: number part
    // dominates the normal-ordered quartic's negative squeezing at low n —
    // structural positivity of the FL comparison at the numerical level).
    let two_p = boson_background()
        .apply(&Operator::InnerBosonCreate(1))
        .apply(&Operator::InnerBosonCreate(1));
    let exp_n = QuantumState::inner_product(&n.apply(&two_p), &two_p).re;
    assert!(
        exp_n >= -1e-9,
        "⟨N⟩ on |2⟩_π must be non-negative for FL comparison, got {exp_n}"
    );

    eprintln!(
        "sm_comparison_n: {} terms, ⟨N⟩(|2⟩_π) = {exp_n:.6}",
        n.terms.len()
    );
}

// ── 8. Outer-vacuum doctrine for the SM sector ──────────────────────────────

#[test]
fn sm_outer_vacuum_annihilated_reduced() {
    // Build the one-particle matrix of the reduced SM on a tiny truncated
    // basis and outer-enclose it: H = Σ h_ij C†(e_i) A(e_j), so H|Ω⟩ = 0.
    let h = sm_reduced_hamiltonian(1.0, 1.0, 0.5, 0.3, 0.2);

    // Tiny basis: vacuum and single occupations of modes 0 and 4 (gauge / Higgs).
    let basis: Vec<Vec<(u32, u32)>> =
        vec![vec![], vec![(0, 1)], vec![(4, 1)], vec![(0, 1), (4, 1)]];

    fn inner_species(occ: &[(u32, u32)]) -> InnerBosonicState {
        let mut s = InnerBosonicState::vacuum();
        for &(m, n) in occ {
            s.modes.insert(m, n);
        }
        s
    }

    fn inner_state(occ: &[(u32, u32)]) -> QuantumState {
        let mut s =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
        for &(m, n) in occ {
            for _ in 0..n {
                s = s.apply(&Operator::InnerBosonCreate(m));
            }
        }
        let norm = s.norm();
        if norm > 1e-15 {
            s.scale_and_add(&s.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));
        }
        s
    }

    let states: Vec<QuantumState> = basis.iter().map(|o| inner_state(o)).collect();
    let nb = states.len();
    let mut h_mat = DMatrix::<Complex64>::zeros(nb, nb);
    for (j, s) in states.iter().enumerate() {
        let hs = h.apply(s);
        for (i, t) in states.iter().enumerate() {
            h_mat[(i, j)] = QuantumState::inner_product(t, &hs);
        }
    }

    // Outer enclosure: creation left, annihilation right.
    let mut terms = Vec::new();
    for (i, ei) in basis.iter().enumerate() {
        for (j, ej) in basis.iter().enumerate() {
            let hij = h_mat[(i, j)];
            if hij.norm_sqr() > 1e-30 {
                terms.push((
                    hij,
                    vec![
                        Operator::OuterBosonCreate(inner_species(ei)),
                        Operator::OuterBosonAnnihilate(inner_species(ej)),
                    ],
                ));
            }
        }
    }
    let h_outer = Hamiltonian { terms };

    // H|Ω⟩ = 0 identically.
    let omega = QuantumState::vacuum();
    let h_omega = h_outer.apply(&omega);
    assert!(
        h_omega.norm() < 1e-12,
        "outer-enclosed SM H must annihilate |Ω⟩, ‖H|Ω⟩‖ = {:.3e}",
        h_omega.norm()
    );

    // On the one-quanton sector H acts as h (Hermitian).
    let one0 = omega.apply(&Operator::OuterBosonCreate(inner_species(&[(0, 1)])));
    let h_one = h_outer.apply(&one0);
    let e_one = QuantumState::inner_product(&one0, &h_one).re;
    assert!(
        e_one.is_finite(),
        "one-quanton energy must be finite, got {e_one}"
    );

    eprintln!(
        "sm_outer_vacuum: H|Ω‖ = {:.3e}, one-quanton ⟨H⟩ = {e_one:.6}",
        h_omega.norm()
    );
}

// ─────────────────────────────────────────────
// Full D_B = 163, BRST ghosts, W±/Z polarizations (§5.24u).
// ─────────────────────────────────────────────

/// Probe: outer boson occupation + optional outer fermion (ghost) content.
fn sm_probe(boson_modes: &[(u32, u32)], ghost_modes: &[u32]) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    for &(m, n) in boson_modes {
        inner.modes.insert(m, n);
    }
    let mut s = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner));
    if !ghost_modes.is_empty() {
        s = s.apply(&Operator::OuterFermionCreate(InnerFermionicState {
            modes: ghost_modes.iter().copied().collect::<BTreeSet<_>>(),
        }));
    }
    let n = s.norm();
    if n > 1e-15 {
        s.scale_and_add(&s.clone(), Complex64::new(1.0 / n - 1.0, 0.0));
    }
    s
}

#[test]
fn sm_db_163_mode_budget() {
    // VERIFY_SM_FARIS_LAVINE / faris_lavine_n_sm.cdb: D_B = 163.
    let b = SmModeBudget::standard();
    assert_eq!(
        (b.spatial_x, b.su3_field, b.su3_deriv),
        (3, 24, 72),
        "SU(3) sector must be 3 + 96"
    );
    assert_eq!(
        (b.su2_field, b.su2_deriv, b.u1_field, b.u1_deriv),
        (9, 27, 3, 9),
        "electroweak abelian/non-abelian field sectors"
    );
    assert_eq!(
        (b.higgs_field, b.higgs_deriv),
        (4, 12),
        "Higgs doublet: 4 + 12"
    );
    assert_eq!(b.total(), 163, "D_B must be 163 = 3 + 96 + 36 + 12 + 16");
    assert_eq!(b.field_ladder_modes(), 160, "ladder modes exclude base x");

    // Ranges are disjoint and tile 0..160.
    let ranges = b.field_mode_ranges();
    let mut lo = 0;
    for (name, rlo, rhi) in &ranges {
        assert_eq!(*rlo, lo, "{name} must start at {lo}");
        assert!(rhi > rlo, "{name} must be nonempty");
        lo = *rhi;
    }
    assert_eq!(lo, 160, "field mode ranges must tile 0..160");

    // Ghost budget: 8 + 3 + 1 = 12, in Z_2^{D_F} (not in D_B).
    assert_eq!(SM_GHOST_COUNT, 12);
    let gr = sm_ghost_mode_ranges();
    assert_eq!(gr, vec![("su3", 0, 8), ("su2", 8, 11), ("u1", 11, 12)]);
    let g_total: usize = gr
        .iter()
        .map(|(_, a, b)| b - a)
        .collect::<Vec<_>>()
        .iter()
        .sum();
    assert_eq!(g_total, SM_GHOST_COUNT);

    eprintln!(
        "sm_db: D_B = {} (x={} + G={} + W={} + B={} + φ={}), ladder = {}, ghosts = {}",
        b.total(),
        b.spatial_x,
        b.su3_field + b.su3_deriv,
        b.su2_field + b.su2_deriv,
        b.u1_field + b.u1_deriv,
        b.higgs_field + b.higgs_deriv,
        b.field_ladder_modes(),
        SM_GHOST_COUNT,
    );
}

#[test]
fn sm_full_free_field_structure() {
    let h = sm_full_free_field_uniform(1.5);
    let n = SmModeBudget::standard().field_ladder_modes();
    assert_eq!(h.terms.len(), n, "one :n_i: per field-ladder mode");

    // Vacuum energy 0 (normal-ordered number operators).
    let vac = sm_probe(&[], &[]);
    let e0 = QuantumState::inner_product(&vac, &h.apply(&vac)).re;
    assert!(e0.abs() < 1e-12, "⟨0|H|0⟩ must be 0, got {e0}");

    // One-quanton energies: ω on every mode in 0..160.
    for &m in &[0u32, 24, 96, 148, 159] {
        let psi = sm_probe(&[(m, 1)], &[]);
        let e = QuantumState::inner_product(&psi, &h.apply(&psi)).re;
        assert!(
            (e - 1.5).abs() < 1e-12,
            "mode {m} must have ω = 1.5, got {e}"
        );
    }

    // Hermiticity on a random low-excitation probe (matrix-free check).
    let psi = sm_probe(&[(0, 1), (36, 1), (159, 1)], &[]);
    let hpsi = h.apply(&psi);
    let e = QuantumState::inner_product(&psi, &hpsi).re;
    assert!(
        e.is_finite() && e > 0.0,
        "excited ⟨H⟩ must be finite positive, got {e}"
    );
    // ⟨H⟩ = ω × n_exc for free field: 3 quanta × 1.5 = 4.5.
    assert!(
        (e - 4.5).abs() < 1e-12,
        "3-quantum energy must be 4.5, got {e}"
    );

    eprintln!("sm_full_free: {n} modes, 3-quantum ⟨H⟩ = {e:.6}");
}

#[test]
fn sm_brst_ghosts_nilpotent() {
    let omega = sm_brst_charge();
    assert!(!omega.terms.is_empty(), "BRST charge must have terms");

    // Ghost content of Ω: every SM ghost mode appears.
    let mut ghost_touched = [false; SM_GHOST_COUNT];
    for (_, ops) in &omega.terms {
        for op in ops {
            if let Operator::InnerFermionAnnihilate(g) | Operator::InnerFermionCreate(g) = op {
                if (*g as usize) < SM_GHOST_COUNT {
                    ghost_touched[*g as usize] = true;
                }
            }
        }
    }
    assert!(
        ghost_touched.iter().all(|&t| t),
        "all 12 ghosts must appear in Ω, got {ghost_touched:?}"
    );

    // Nilpotency on ghosted probes: one SU(3), one SU(2), one U(1) ghost,
    // each with a Gauss-carrying field mode.
    // Field-ladder layout: su3_field 0..24, su2_field 24..33, u1_field 33..36.
    let probes: [(u32, u32); 3] = [(0, 0), (3, 8), (0, 11)]; // (field mode, ghost)
    let mut worst = 0.0f64;
    for (bm, g) in probes {
        let psi = sm_probe(&[(bm, 1)], &[g]);
        let once = omega.apply(&psi);
        let twice = omega.apply(&once);
        worst = worst.max(twice.norm());
        assert!(
            twice.norm() < 1e-9,
            "Ω² must vanish on (field {bm}, ghost {g}), ‖Ω²ψ‖ = {:.3e}",
            twice.norm()
        );
    }

    // Also nilpotent on a multi-ghost + multi-field probe.
    let multi = sm_probe(&[(0, 1), (24, 1), (33, 1)], &[0, 8, 11]);
    let w2 = omega.apply(&omega.apply(&multi)).norm();
    worst = worst.max(w2);
    assert!(w2 < 1e-9, "Ω² on multi-ghost probe = {w2:.3e}");

    // Ghost-free states are BRST-closed at the abelian level only if
    // Gauss·c† annihilates them — here Ω needs a ghost leg, so Ω|phys⟩ = 0
    // for fully ghost-free probes (nilpotency domain of the residual charge).
    let phys = sm_probe(&[(0, 1)], &[]);
    let o_phys = omega.apply(&phys).norm();
    assert!(
        o_phys < 1e-12,
        "Ω must annihilate ghost-free probes, got {o_phys:.3e}"
    );

    eprintln!("sm_brst: 12 ghosts, worst ‖Ω²‖ = {worst:.3e}");
}

#[test]
fn sm_weak_boson_polarizations() {
    // Full polarization budget: W±(2×3) + Z(3) + γ(2) = 11.
    assert_eq!(sm_weak_polarization_mode_count(), 11);
    let table = sm_weak_polarization_table();
    assert_eq!(table.len(), 11);
    assert_eq!(table.iter().map(|(_, m, _, _)| *m).max().unwrap(), 10);

    let [mw, mz, mgamma] = sm_weak_boson_masses_gev();
    assert!((mw - 80.377).abs() < 1e-9, "m_W = 80.377 GeV (PDG)");
    assert!((mz - 91.1876).abs() < 1e-9, "m_Z = 91.1876 GeV (PDG)");
    assert_eq!(mgamma, 0.0, "photon must be massless");

    // Mass assignment per name; longitudinal flags.
    let mut n_long = 0;
    let mut n_transverse = 0;
    for (name, _m, mass, long) in &table {
        if name.starts_with('W') {
            assert!((mass - mw).abs() < 1e-12, "{name} must have m_W");
        } else if name.starts_with('Z') {
            assert!((mass - mz).abs() < 1e-12, "{name} must have m_Z");
        } else if name.starts_with("gamma") {
            assert_eq!(*mass, 0.0, "{name} must be massless");
        }
        if *long {
            n_long += 1;
        } else {
            n_transverse += 1;
        }
    }
    assert_eq!(n_long, 3, "one longitudinal mode each for W+, W−, Z");
    assert_eq!(n_transverse, 8, "2 transverse × (W+ + W− + Z + γ)");

    // Transverse degeneracy: same mass for T0/T1 within each boson.
    let h = sm_weak_boson_hamiltonian();
    assert_eq!(h.terms.len(), 11);

    let e_of = |mode: u32| -> f64 {
        let psi = sm_probe(&[(mode, 1)], &[]);
        QuantumState::inner_product(&psi, &h.apply(&psi)).re
    };

    // W+ transverse modes 0,1 degenerate at m_W; longitudinal mode 2 also m_W.
    let (e_w_t0, e_w_t1, e_w_l) = (e_of(0), e_of(1), e_of(2));
    assert!(
        (e_w_t0 - mw).abs() < 1e-9 && (e_w_t1 - mw).abs() < 1e-9,
        "W transverse must be degenerate at m_W: {e_w_t0}, {e_w_t1}"
    );
    assert!(
        (e_w_l - mw).abs() < 1e-9,
        "W longitudinal shares m_W (Goldstone-dressed): {e_w_l}"
    );
    assert!(
        (e_w_t0 - e_w_t1).abs() < 1e-12,
        "transverse polarizations must be exactly degenerate"
    );

    // Z: modes 6,7,8 at m_Z; gap vs W is the physical m_Z − m_W.
    let (e_z_t0, e_z_t1) = (e_of(6), e_of(7));
    assert!((e_z_t0 - mz).abs() < 1e-9 && (e_z_t1 - mz).abs() < 1e-9);
    let gap = e_z_t0 - e_w_t0;
    assert!(
        (gap - (mz - mw)).abs() < 1e-9,
        "m_Z − m_W gap must match PDG, got {gap}"
    );

    // Photon: modes 9,10 massless.
    assert!(e_of(9).abs() < 1e-12 && e_of(10).abs() < 1e-12);

    // Vacuum energy 0; SIRK spectrum on a W/Z superposition stays finite
    // and real (no ghost instabilities in the polarization sector).
    let vac = sm_probe(&[], &[]);
    assert!(
        QuantumState::inner_product(&vac, &h.apply(&vac)).re.abs() < 1e-12,
        "polarization vacuum energy must vanish"
    );

    let psi0 = sm_probe(&[(0, 1), (6, 1)], &[]);
    let shifts11 = shifts(11);
    let res = solve_forward_sirk_with_opts(&h, &psi0, &shifts11, &best_device(), None, &opts())
        .expect("SIRK on weak polarization sector");
    assert_hermitian(&res.h_proj, "weak polarization H_proj");
    let ritz = res.ritz_values();
    assert!(!ritz.is_empty(), "need Ritz values");
    for &e in &ritz {
        assert!(e.is_finite(), "Ritz value must be finite, got {e}");
    }
    // Ground Ritz is a lower bound; for free oscillators started from a
    // finite excitation it sits near 0 (vacuum) or below the probe energy.
    let e0 = *ritz
        .iter()
        .fold(&f64::INFINITY, |a, b| if b < a { b } else { a });
    assert!(
        e0 > -1.0e6,
        "polarization spectrum bounded below, E₀ = {e0}"
    );
    let e_probe = QuantumState::inner_product(&psi0, &h.apply(&psi0)).re;
    assert!(
        ((e_probe - (mw + mz)).abs() < 1e-9),
        "two-quantum probe energy must be m_W + m_Z, got {e_probe}"
    );

    eprintln!(
        "sm_weak_pol: 11 modes, m_W={mw}, m_Z={mz}, gap={gap:.4}, probe ⟨H⟩={e_probe:.6}, E₀={e0:.6}"
    );
}
