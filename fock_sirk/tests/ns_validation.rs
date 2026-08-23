//! Navier–Stokes validation: the Fock-space / SIRK machinery checked against the
//! formalization of `../timepiece/CONSOLIDATED_PLAN.md` (the Navier–Stokes
//! thread, `PLAN_LEAN_SPECIALIST_NS_FLOW.md`).
//!
//! The formalization is carried in the **Eulerian derivatives-as-fields**
//! picture (the priority route of CONSOLIDATED_PLAN.md §9 items 4 and 8, and
//! "What is missing"): the velocity components `u_i` carry the momenta, the
//! spatial-derivative fields `u_{i,j}`, `u_{i,jj}` are *independent canonical
//! coordinates* that carry **no** momenta, so they commute with the Hamiltonian
//! and are constants of the motion (`BilinearEsa` / `AffineBlock` block
//! decomposition). The fiber Hamiltonian in a block is the affine
//! `V(u) = Σ_k A_{ik} u_k + c_i` (`AffineFiber`, `AffineBlock`,
//! `ThreeComponent`), whose `±2`-shifts (advection) and `±1`-shifts (viscous)
//! are `nsH(κ)`. The Hashimoto shift-invert selection
//! (`ChapterHashimotoComplexShifts`, `ChapterEsaClosure`) backs the SIRK
//! algorithm itself: for a non-real shift the resolvent `(γI−A)⁻¹` is bounded
//! by `1/|Im γ|` and determines the operator.
//!
//! The Lagrangian/trajectory route (item 9) is the deliberately non-priority
//! parallel picture — the Eulerian route is the one the tests below instantiate
//! numerically, matching the repo's `navier_stokes_hamiltonian` (velocity +
//! derivative + second-derivative modes, `H = Σ_i {π_i, A_i}`).
//!
//! Tests:
//!
//! 1. `ns_derivative_fields_constant_of_motion` — `[H, u_{i,j}] = 0` and
//!    `[H, u_{i,jj}] = 0` **exactly** (the derivative modes carry no momenta:
//!    the block-diagonalisation statement of the Eulerian picture), and
//!    `⟨u_{i,j}⟩` is conserved under SIRK time evolution.
//!
//! 2. `ns_affine_fiber_hopping_structure` — the single-mode affine fiber
//!    `V(u) = κu + c`: exactly the `±2`-hopping `2κ√((n+1)(n+2))` of the
//!    advection plus the `±1`-hopping `2c√(n+1)` of the viscous offset, with
//!    all other matrix elements zero (the analytic content of
//!    `AffineFiber.affH` / `SignedShift`).
//!
//! 3. `ns_three_component_vorticity_hopping` — the 3-component affine fiber
//!    with an **arbitrary non-symmetric** gradient matrix `A`: the 24 hopping
//!    terms per component, the number-conserving *vorticity* hopping
//!    `⟨1_k|H|1_i⟩ = 2i(A_{ki} − A_{ik})` whose amplitude is not monotone along
//!    the shift, and the `±2` pair-creation/annihilation hopping
//!    (`ThreeComponent.velH`).
//!
//! 4. `ns_sirk_esa_truncation` — the truncated NS operator via SIRK: the
//!    projected Hamiltonian is Hermitian (self-adjoint in the finite
//!    truncation — the finite shadow of essential self-adjointness on the
//!    finite-mode core), has a real spectrum bounded below, and
//!    `⟨0|H|0⟩ = 0`.
//!
//! 5. `ns_hashimoto_shift_invert_selection` — the Hashimoto shift-invert
//!    selection theorem instantiated numerically on the NS projection: for a
//!    non-real shift `γ` the resolvent `R = (γI−A)⁻¹` exists, is bounded by
//!    `‖R‖ ≤ 1/|Im γ|` (`ChapterHashimotoComplexShifts`), satisfies the
//!    resolvent identity, and its eigenvectors/eigenvalues `1/(γ−λ_j)` recover
//!    the NS spectrum — the resolvent determines the operator (the selection
//!    behind the SIRK algorithm).
//!
//! 6. `ns_unitary_evolution_conservation` — SIRK restarted-Krylov dynamics of
//!    the interacting NS Hamiltonian: probability (norm) and energy are
//!    conserved, and the derivative-field expectation `⟨u_{i,j}⟩` is a
//!    constant of the motion (unitarity of the finite flow + the Eulerian
//!    block structure).
//!
//! 7. `ns_brst_projection_physical_subspace` — the BRST divergence constraint
//!    `Ω = Σ_j u_{j,j} c_j`: nilpotency `Ω² = 0` (first-class constraint), the
//!    gauge invariance `[H, Ω] = 0` / `[H_f, Ω] = 0` (BRST-closed NS and fiber
//!    Hamiltonians), and the non-invariance of the physical subspace under the
//!    bare flow (the Ω-content of an unphysical ghost-carrying state grows),
//!    the reason the projector rides along in the solve.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, evolve_restarted, solve_forward_sirk_with_opts};
use nalgebra::DMatrix;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, InnerFermionicState, Operator, QuantumState,
    navier_stokes_brst, navier_stokes_hamiltonian, ns_eulerian_fiber,
};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn assert_hermitian(h_proj: &DMatrix<Complex64>, label: &str) {
    let dag = h_proj.adjoint();
    let diff = (h_proj - &dag).norm();
    assert!(
        diff < 1e-6,
        "{label}: H_proj must be Hermitian, ‖H−H†‖={diff}"
    );
}

/// One empty inner universe (AGENTS.md vacuum-initialization rule) — the
/// substrate the NS inner ladder operators act on.
fn inner_vac() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// A bosonic occupation eigenstate: one universe whose inner occupation of
/// `mode` is `count`. Occupation zero drops the mode entirely (the framework's
/// annihilation of a last quantum produces the empty `{}` universe, so
/// `ns_state(m, 0)` must equal the true vacuum for matrix elements to overlap).
fn ns_state(mode: u32, count: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    if count > 0 {
        inner.modes.insert(mode, count);
    }
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

/// A multi-mode bosonic occupation eigenstate from an `(mode, count)` list.
fn ns_multi(occupations: &[(u32, u32)]) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    for &(m, n) in occupations {
        inner.modes.insert(m, n);
    }
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

/// The Eulerian field operator `u_m = a†_m + a_m` as a Hamiltonian (self-adjoint
/// two-term sum) — used to measure the field-amplitude expectation `⟨u_m⟩` and
/// to compute commutators `[H, u_m]`.
fn field_hamiltonian(mode: u32) -> Hamiltonian {
    Hamiltonian {
        terms: vec![
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(mode)]),
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(mode)]),
        ],
    }
}

/// `‖[H, u]ψ‖` for the commutator of the NS Hamiltonian with a field operator.
fn commutator_norm(h: &Hamiltonian, u: &Hamiltonian, psi: &QuantumState) -> f64 {
    let hu = h.apply(&u.apply(psi));
    let uh = u.apply(&h.apply(psi));
    let mut diff = hu;
    diff.scale_and_add(&uh, Complex64::new(-1.0, 0.0));
    diff.norm()
}

// ── 1. Eulerian derivative fields are constants of the motion ────────────────

#[test]
fn ns_derivative_fields_constant_of_motion() {
    // The Eulerian derivatives-as-fields picture: the Hamiltonian carries
    // momenta only for the velocity modes, so the derivative fields `u_{i,j}`
    // (modes 3..12) and the second derivatives `u_{i,jj}` (modes 12..15)
    // commute with H and are constants of the motion — the block decomposition
    // that diagonalises the derivative field (`BilinearEsa`/`AffineBlock` in
    // CONSOLIDATED_PLAN.md §9 item 4). Every derivative mode must satisfy
    // `[H, u_m] = 0` exactly.
    let h = navier_stokes_hamiltonian(1e-3);
    let probes = [
        inner_vac(),
        ns_state(0, 1),
        ns_state(1, 1),
        ns_state(2, 1),
        ns_state(3, 1),
        ns_state(7, 1),
        ns_state(11, 1),
        ns_state(12, 1),
        ns_state(14, 1),
        ns_multi(&[(0, 1), (3, 1)]),
        ns_state(5, 2),
    ];

    for m in 3..15u32 {
        let u = field_hamiltonian(m);
        for (i, p) in probes.iter().enumerate() {
            let nrm = commutator_norm(&h, &u, p);
            assert!(
                nrm < 1e-8,
                "[H, u_{m}] must vanish on probe {i}: ‖[H,u]ψ‖ = {nrm:.3e}"
            );
        }
    }

    // Heisenberg form: ⟨ψ|i[H,u]ψ⟩ = 0 for every derivative mode and probe.
    for m in [3u32, 7, 11, 12] {
        let u = field_hamiltonian(m);
        for p in [&probes[0], &probes[3], &probes[9]] {
            let hu = h.apply(&u.apply(p));
            let uh = u.apply(&h.apply(p));
            let mut diff = hu;
            diff.scale_and_add(&uh, Complex64::new(-1.0, 0.0));
            let expect = QuantumState::inner_product(p, &diff);
            assert!(
                expect.norm() < 1e-9,
                "⟨ψ|i[H,u_{m}]ψ⟩ must vanish, got {expect:?}"
            );
        }
    }

    // Under the SIRK-restarted unitary flow the norm, the energy and the
    // derivative-field expectation ⟨u_{0,0}⟩ are conserved. Norm and energy are
    // preserved *by construction* by the unitary Krylov-restricted evolution
    // (h_proj Hermitian); ⟨u_{i,j}⟩ is an exactly-conserved quantity of the
    // exact evolution ([H, u_{i,j}] = 0), so its drift measures the accuracy of
    // the finite Krylov truncation of the interacting flow.
    let opts = SirkOpts {
        prune_eps: 1e-10,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: true,
        unit_norm_steps: false,
    };
    let mut psi0 = ns_state(0, 1);
    psi0.scale_and_add(&ns_state(1, 1), Complex64::new(0.5, 0.0));
    psi0.scale_and_add(&ns_state(2, 1), Complex64::new(0.25, 0.0));
    let u = field_hamiltonian(3);
    let n0 = psi0.norm();
    let e0 = QuantumState::inner_product(&psi0, &h.apply(&psi0)).re;
    let d0 = QuantumState::inner_product(&psi0, &u.apply(&psi0)).re;

    let psi_t = evolve_restarted(&h, &psi0, 0.05, 2, 2, &best_device(), None, &opts).unwrap();
    let n_t = psi_t.norm();
    let e_t = QuantumState::inner_product(&psi_t, &h.apply(&psi_t)).re;
    let d_t = QuantumState::inner_product(&psi_t, &u.apply(&psi_t)).re;
    assert!(
        (n_t - n0).abs() < 1e-8,
        "norm must be conserved (unitarity of the truncated flow): {:.2e}",
        (n_t - n0).abs()
    );
    assert!(
        (e_t - e0).abs() < 1e-6,
        "energy must be conserved (Hermitian truncation): {:.2e}",
        (e_t - e0).abs()
    );
    assert!(
        (d_t - d0).abs() < 1e-6,
        "⟨u_{{0,0}}⟩ must be conserved under the flow (constant of motion): \
         {d0:.8} → {d_t:.8}",
    );

    eprintln!(
        "ns_derivative_fields_constant_of_motion: [H, u_{{i,j}}] = 0 and [H, u_{{i,jj}}] = 0 \
         exactly; under SIRK evolution ‖ψ‖ {n0:.6}→{n_t:.6}, ⟨H⟩ {e0:.6}→{e_t:.6}, \
         ⟨u_{{0,0}}⟩ {d0:.8}→{d_t:.8}"
    );
}

// ── 2. Affine fiber: the ±2 (advection) and ±1 (viscous) hopping structure ───

#[test]
fn ns_affine_fiber_hopping_structure() {
    // The single-mode affine fiber `V(u) = κ·u + c` — the velocity-momentum
    // part of the NS Hamiltonian with the derivative field frozen to `κ` and
    // the viscous offset frozen to `c` (`AffineFiber.affH` / `SignedShift`).
    // With the framework-native `u = a†+a`, `π = i(a†−a)` and the
    // anti-commutator `{π,V}` the fiber is
    //
    //   H = 2κ·i(a†² − a²) + 2c·i(a† − a),
    //
    // i.e. **only** a ±2-hopping (the advection pair creation/annihilation)
    // and a ±1-hopping (the viscous offset), with exact amplitudes
    //
    //   |⟨n+2|H|n⟩| = 2κ·√((n+1)(n+2)),   |⟨n+1|H|n⟩| = 2c·√(n+1),
    //
    // and every other matrix element zero. (The plan's normalized convention
    // `u = (a†+a)/√2`, `π = i(a†−a)/√2`, `H = ½(πV+Vπ)` gives the factor-4/2√2
    // scaled `(κ/2)√((n+1)(n+2))` and `(c/√2)√(n+1)` — the identical
    // *structure*.)
    let kappa = 1.3;
    let c = 0.7;
    let h = ns_eulerian_fiber(
        &[[kappa, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        &[c, 0.0, 0.0],
    );

    // Only mode 0 is touched (modes 1, 2 are inert with this A).
    assert!(
        h.terms
            .iter()
            .all(|(_, ops)| ops.iter().all(|o| matches!(
                o,
                Operator::InnerBosonCreate(0) | Operator::InnerBosonAnnihilate(0)
            ))),
        "the single-mode fiber must only touch velocity mode 0"
    );

    for n in 0..=4u32 {
        let ket_n = ns_state(0, n);
        // ±2 hopping: |⟨n+2|H|n⟩| = 2κ√((n+1)(n+2)), pure imaginary phase.
        let me22 = QuantumState::inner_product(&ns_state(0, n + 2), &h.apply(&ket_n));
        let expect22 = 2.0 * kappa * ((n + 1) as f64 * (n + 2) as f64).sqrt();
        assert!(
            (me22.norm() - expect22).abs() < 1e-9,
            "n={n}: |⟨n+2|H|n⟩| = {} must equal 2κ√((n+1)(n+2)) = {expect22}",
            me22.norm()
        );
        assert!(
            me22.re.abs() < 1e-9 && me22.im > 0.0,
            "n={n}: ⟨n+2|H|n⟩ = {me22:?} must be +2iκ√((n+1)(n+2))"
        );

        // ±1 hopping: |⟨n+1|H|n⟩| = 2c√(n+1).
        let me21 = QuantumState::inner_product(&ns_state(0, n + 1), &h.apply(&ket_n));
        let expect21 = 2.0 * c * ((n + 1) as f64).sqrt();
        assert!(
            (me21.norm() - expect21).abs() < 1e-9,
            "n={n}: |⟨n+1|H|n⟩| = {} must equal 2c√(n+1) = {expect21}",
            me21.norm()
        );

        // Diagonal: ⟨n|H|n⟩ = 0 (the Weyl symmetrization strips the zero-point).
        let me0 = QuantumState::inner_product(&ket_n, &h.apply(&ket_n));
        assert!(
            me0.norm() < 1e-9,
            "n={n}: ⟨n|H|n⟩ must vanish, got {me0:?}"
        );

        // No other hopping: |Δn| = 3 and |Δn| = 0 are the nearest missed
        // transitions — everything outside {±1, ±2} is zero.
        if n + 3 <= 7 {
            let me3 = QuantumState::inner_product(&ns_state(0, n + 3), &h.apply(&ket_n));
            assert!(
                me3.norm() < 1e-9,
                "n={n}: ⟨n+3|H|n⟩ must vanish (no Δ=3 hopping), got {me3:?}"
            );
        }
        if n >= 3 {
            let me_neg = QuantumState::inner_product(&ns_state(0, n - 3), &h.apply(&ket_n));
            assert!(
                me_neg.norm() < 1e-9,
                "n={n}: ⟨n−3|H|n⟩ must vanish (no Δ=−3 hopping), got {me_neg:?}"
            );
        }

        // Hermiticity: ⟨n|H|n+2⟩ = conj(⟨n+2|H|n⟩) = −2iκ√((n+1)(n+2)).
        let h22 = QuantumState::inner_product(&ket_n, &h.apply(&ns_state(0, n + 2)));
        assert!(
            (h22 - me22.conj()).norm() < 1e-9,
            "n={n}: ⟨n|H|n+2⟩ = {h22:?} must equal conj(⟨n+2|H|n⟩) = {:?}",
            me22.conj()
        );
    }

    eprintln!(
        "ns_affine_fiber_hopping_structure: κ={kappa}, c={c} → |⟨n+2|H|n⟩|=2κ√((n+1)(n+2)), \
         |⟨n+1|H|n⟩|=2c√(n+1), all other elements zero"
    );
}

// ── 3. Three-component fiber: the non-monotone vorticity hopping ─────────────

#[test]
fn ns_three_component_vorticity_hopping() {
    // The 3-component affine fiber with an **arbitrary real** velocity-gradient
    // matrix A (no symmetry/positivity/sign assumption) — `ThreeComponent.velH`
    // of the formalization. `H = Σ_i {π_i, V_i}` with `V_i = Σ_k A_{ik} u_k`
    // (c = 0). For a generic A the fiber carries 24 field-hopping terms per
    // component (12 from π_i·V_i + 12 from V_i·π_i) = 72 total.
    let a = [
        [1.0, 2.0, 0.5],
        [0.25, 1.0, 3.0],
        [1.5, 0.5, 2.0],
    ];
    let h = ns_eulerian_fiber(&a, &[0.0, 0.0, 0.0]);

    assert_eq!(
        h.terms.len(),
        72,
        "the 3-component fiber must carry 3 × 24 = 72 hopping terms"
    );

    // (a) Number-conserving *vorticity* hopping: in the one-quantum velocity
    //     sector ⟨1_k|H|1_i⟩ = 2i(A_{ki} − A_{ik}) — the antisymmetric part of
    //     the gradient (the vorticity), the shift with non-monotone amplitude.
    for i in 0..3u32 {
        for k in 0..3u32 {
            let me = QuantumState::inner_product(&ns_state(k, 1), &h.apply(&ns_state(i, 1)));
            if i == k {
                assert!(
                    me.norm() < 1e-9,
                    "diagonal ⟨1_{k}|H|1_{i}⟩ must vanish, got {me:?}"
                );
            } else {
                let expect = Complex64::new(0.0, 2.0 * (a[k as usize][i as usize]
                    - a[i as usize][k as usize]));
                assert!(
                    (me - expect).norm() < 1e-9,
                    "⟨1_{k}|H|1_{i}⟩ = {me:?} must equal 2i(A_{k}{i} − A_{i}{k}) = {expect:?}"
                );
            }
        }
    }

    // (b) The amplitude is NOT monotone along the shift: for a fixed source i
    //     the |⟨1_k|H|1_i⟩| sequence across k is neither increasing nor
    //     decreasing. E.g. for i=1: k=0 → 2|A₀₁−A₁₀| = 3.5, k=2 → 2|A₂₁−A₁₂|
    //     = 5.0 (increasing), while for i=0: k=1 → 3.5, k=2 → 2|A₂₀−A₀₂| =
    //     2.0 (decreasing) — the exact "amplitude not monotone along its
    //     shift" statement of the ThreeComponent formalization.
    let amp = |i: usize, k: usize| -> f64 {
        let me = QuantumState::inner_product(&ns_state(k as u32, 1), &h.apply(&ns_state(i as u32, 1)));
        me.norm()
    };
    assert!(
        amp(1, 2) > amp(1, 0),
        "vorticity amplitude must grow along the i=1 row: {} then {}",
        amp(1, 0),
        amp(1, 2)
    );
    assert!(
        amp(0, 2) < amp(0, 1),
        "vorticity amplitude must shrink along the i=0 row: {} then {}",
        amp(0, 1),
        amp(0, 2)
    );

    // (c) The ±2 pair-creation/annihilation hopping: ⟨1_j 1_k|H|vac⟩ = 2i(A_{jk}
    //     + A_{kj}) — both components of the gradient matrix drive the
    //     pair sector (component m=j creates the pair via A_{jk} and component
    //     m=k via A_{kj}, hence the symmetric part) — and the double-occupation
    //     ⟨2_j|H|vac⟩ = 2i√2·A_{jj}.
    let vac = inner_vac();
    for j in 0..3u32 {
        for k in 0..3u32 {
            if j == k {
                let me = QuantumState::inner_product(&ns_state(j, 2), &h.apply(&vac));
                let expect = Complex64::new(0.0, 2.0 * std::f64::consts::SQRT_2 * a[j as usize][k as usize]);
                assert!(
                    (me - expect).norm() < 1e-9,
                    "⟨2_{j}|H|vac⟩ = {me:?} must equal 2i√2·A_{j}{j} = {expect:?}"
                );
            } else {
                let me = QuantumState::inner_product(&ns_multi(&[(j, 1), (k, 1)]), &h.apply(&vac));
                let expect = Complex64::new(0.0, 2.0 * (a[j as usize][k as usize]
                    + a[k as usize][j as usize]));
                assert!(
                    (me - expect).norm() < 1e-9,
                    "⟨1_{j}1_{k}|H|vac⟩ = {me:?} must equal 2i·(A_{j}{k} + A_{k}{j}) = {expect:?}"
                );
            }
        }
    }

    eprintln!(
        "ns_three_component_vorticity_hopping: ⟨1_k|H|1_i⟩ = 2i(A_ki − A_ik) \
         (non-monotone vorticity hopping), 72 terms, ±2 pair hopping present"
    );
}

// ── 4. SIRK on the truncated NS Hamiltonian: Hermitian, real, bounded below ──

#[test]
fn ns_sirk_esa_truncation() {
    // The finite truncation of the NS Hamiltonian `H = Σ_i {π_i, A_i}` reduced
    // by the Hashimoto inverse-free rational-Krylov solver. The truncated
    // operator is a finite Hermitian matrix — the numerical shadow of
    // essential self-adjointness on the finite-mode core (`BilinearEsa`,
    // `AffineBlock`, `ThreeComponent`): self-adjoint (h_proj = h_proj†), real
    // spectrum, bounded below with finite ground state.
    let nu = 1e-3;
    let h = navier_stokes_hamiltonian(nu);

    // ⟨0|H|0⟩ = 0: the Weyl-symmetrized NS operator is normal-ordered.
    let vac = inner_vac();
    let e0 = QuantumState::inner_product(&vac, &h.apply(&vac)).re;
    assert!(
        e0.abs() < 1e-9,
        "⟨0|H|0⟩ must be 0 (Weyl symmetrization), got {e0}"
    );

    // Vacuum sector.
    let opts = SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: true,
        unit_norm_steps: false,
    };
    let res = solve_forward_sirk_with_opts(&h, &vac, &shifts(4), &best_device(), None, &opts)
        .expect("NS SIRK solve from the vacuum");
    assert_hermitian(&res.h_proj, "NS projected Hamiltonian (vacuum sector)");
    let ritz = res.ritz_values();
    assert!(
        ritz.len() >= 2,
        "SIRK must resolve ≥2 levels of the NS Hamiltonian, got {}",
        ritz.len()
    );
    assert!(
        ritz[0] > -100.0,
        "NS spectrum must be bounded below (finite ground state), got ritz0={}",
        ritz[0]
    );
    assert!(
        ritz.iter().all(|r| r.is_finite()),
        "all Ritz values must be finite (no spectral blow-up)"
    );

    // A one-quantum velocity state and a one-quantum derivative state must
    // also give finite, Hermitian projections (the solver stays within the
    // dense core).
    for (label, start) in [
        ("one velocity quantum", ns_state(0, 1)),
        ("one derivative quantum", ns_state(3, 1)),
    ] {
        let r = solve_forward_sirk_with_opts(&h, &start, &shifts(4), &best_device(), None, &opts)
            .expect("NS SIRK solve");
        assert_hermitian(&r.h_proj, &format!("NS projected Hamiltonian ({label})"));
        let ritz = r.ritz_values();
        assert!(
            ritz.iter().all(|x| x.is_finite()),
            "({label}) Ritz values must be finite, got {:?}",
            &ritz[..ritz.len().min(4)]
        );
    }

    eprintln!(
        "ns_sirk_esa_truncation: ⟨0|H|0⟩={e0}, SIRK ritz[0..2]={:?} \
         (Hermitian, real spectrum, bounded below)",
        &ritz[..2.min(ritz.len())]
    );
}

// ── 5. Hashimoto shift-invert selection on the NS projection ────────────────

#[test]
fn ns_hashimoto_shift_invert_selection() {
    // The Hashimoto/SIRK shift-invert selection theorem
    // (`ChapterHashimotoComplexShifts.shiftInvertC_resolvent_identity`,
    // `hashimoto_multishift_selects_esa`; CONSOLIDATED_PLAN.md §9 item 8):
    // for a non-real shift `γ` the resolvent `R = (γI−A)⁻¹` of the (real-
    // spectrum) NS projection is bounded by `1/|Im γ|`, satisfies the
    // resolvent identity, and its spectral data determine the operator —
    // exactly the property the inverse-free rational-Krylov algorithm is built
    // on (the shifts used by SIRK, `shifts_for_range`, are non-real).
    let h = navier_stokes_hamiltonian(1e-3);
    let opts = SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: true,
        unit_norm_steps: false,
    };
    let res = solve_forward_sirk_with_opts(&h, &inner_vac(), &shifts(4), &best_device(), None, &opts)
        .expect("NS SIRK solve");
    let a = res.h_proj.clone();
    let n = a.nrows();
    assert!(n >= 2, "projected NS matrix must be ≥ 2×2, got {n}");

    // Real spectrum: h_proj is Hermitian, so its eigenvalues are real.
    let eig = a.clone().symmetric_eigen();
    let lambdas: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    let eigvecs = eig.eigenvectors;

    for s in [0.5, 1.0, 2.0] {
        let gamma = Complex64::new(0.0, s);
        let gamma_i = DMatrix::<Complex64>::identity(n, n) * gamma;

        // (a) The resolvent exists for every non-real shift (no eigenvalue lies
        //     on the imaginary axis).
        let r = (gamma_i.clone() - a.clone())
            .try_inverse()
            .expect("(γI−A) must be invertible for Im γ ≠ 0");

        // (b) Resolvent identity: (γI−A)·R = I.
        let check = (gamma_i.clone() - a.clone()) * r.clone();
        let mut id = DMatrix::<Complex64>::identity(n, n);
        id -= &check;
        assert!(
            id.norm() < 1e-10,
            "resolvent identity must hold: ‖(γI−A)R − I‖ = {:.2e}",
            id.norm()
        );

        // (c) Spectral bound ‖R‖ ≤ 1/|Im γ| (ChapterHashimotoComplexShifts):
        //     the singular values of R are 1/|γ−λ_j| = 1/√(λ_j²+s²) (R is a
        //     polynomial in the Hermitian A, hence normal), all ≤ 1/s.
        let sv = r.clone().svd(true, true).singular_values;
        let norm_r = sv[0];
        assert!(
            norm_r <= 1.0 / s + 1e-9,
            "‖(γI−A)⁻¹‖ must be ≤ 1/|Im γ| = {}: got {norm_r:.6} (γ={gamma})",
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

        // (d) The resolvent determines the operator: each eigenvector v_j of A
        //     is an eigenvector of R with eigenvalue 1/(γ−λ_j), so λ_j = γ −
        //     1/μ_j recovers the NS spectrum from the shifted data (the
        //     shift-invert selection).
        let v = eigvecs.clone();
        for j in 0..n {
            let vj = v.column(j).into_owned();
            let rvj = &r * &vj;
            let mu = 1.0 / (gamma - Complex64::new(lambdas[j], 0.0));
            let mut res_vec = rvj;
            res_vec -= &(vj * mu);
            assert!(
                res_vec.norm() < 1e-8,
                "R v_j must equal 1/(γ−λ_j) v_j: ‖R v_j − μ_j v_j‖ = {:.2e}",
                res_vec.norm()
            );
            // Recovery: λ_j = γ − 1/μ_j reproduces the exact NS spectrum.
            let recovered = gamma - (1.0 / mu);
            assert!(
                (recovered.re - lambdas[j]).abs() < 1e-8,
                "λ_j = γ − 1/μ_j must recover the NS spectrum: {:.8} vs {:.8}",
                recovered.re,
                lambdas[j]
            );
        }
    }

    // (e) SIRK's own shifts are non-real — the very shifts for which the bound
    //     (c) holds — so the algorithm is the inverse-free rational-Krylov
    //     realization of this selection.
    let sirk_shifts = shifts(4);
    assert!(
        sirk_shifts.iter().all(|g| g.re == 0.0 && g.im >= 1.0),
        "SIRK shifts must be non-real (|Im γ| ≥ 1), got {sirk_shifts:?}"
    );

    eprintln!(
        "ns_hashimoto_shift_invert: ‖(γI−A)⁻¹‖ ≤ 1/|Im γ| holds at γ=i·0.5, i·1, i·2; \
         spectrum recovered from the shifted resolvent (Hashimoto selection)"
    );
}

// ── 6. Unitary dynamics of the interacting NS Hamiltonian (SIRK restarted) ───

#[test]
fn ns_unitary_evolution_conservation() {
    // Evolve a velocity-mode superposition with the restarted-Krylov stepper
    // (two restarts, Krylov dimension 2, matching the tractable interacting
    // sector). The truncated flow is unitary (h_proj Hermitian): probability
    // is conserved, energy is conserved, and the derivative-field expectation
    // ⟨u_{i,j}⟩ is a constant of the motion (test 1).
    use fock_sirk::evolve_restarted;
    let h = navier_stokes_hamiltonian(1e-3);
    let mut psi0 = ns_state(0, 1);
    psi0.scale_and_add(&ns_state(1, 1), Complex64::new(0.5, 0.0));
    psi0.scale_and_add(&ns_state(2, 1), Complex64::new(0.25, 0.0));
    let n0 = psi0.norm();
    let e0 = QuantumState::inner_product(&psi0, &h.apply(&psi0)).re;
    let u3 = field_hamiltonian(3);
    let d0 = QuantumState::inner_product(&psi0, &u3.apply(&psi0)).re;

    let opts = SirkOpts {
        prune_eps: 1e-10,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: true,
        unit_norm_steps: false,
    };
    let psi_t = evolve_restarted(&h, &psi0, 0.05, 2, 2, &best_device(), None, &opts).unwrap();
    let n_t = psi_t.norm();
    let e_t = QuantumState::inner_product(&psi_t, &h.apply(&psi_t)).re;
    let d_t = QuantumState::inner_product(&psi_t, &u3.apply(&psi_t)).re;

    assert!(
        (n_t - n0).abs() < 1e-8,
        "NS norm must be conserved (unitarity of the finite flow): |Δ‖ψ‖| = {:.2e}",
        (n_t - n0).abs()
    );
    assert!(
        (e_t - e0).abs() < 1e-6,
        "NS energy must be conserved: |Δ⟨H⟩| = {:.2e}",
        (e_t - e0).abs()
    );
    assert!(
        (d_t - d0).abs() < 1e-6,
        "NS derivative-field ⟨u_{{0,0}}⟩ must be conserved (constant of motion): \
         {d0:.8} → {d_t:.8}",
    );

    eprintln!(
        "ns_unitary_evolution: ‖ψ‖ conserved ({n0:.6}→{n_t:.6}), ⟨H⟩ conserved \
         ({e0:.6}→{e_t:.6}), ⟨u_{{0,0}}⟩ conserved ({d0:.8}→{d_t:.8})"
    );
}

// ── 7. BRST divergence constraint: nilpotency and gauge invariance ───────────

#[test]
fn ns_brst_projection_physical_subspace() {
    // The BRST charge Ω = Σ_j u_{j,j} c_j imposes the divergence constraint
    // u_{j,j} = 0 (modes 3, 7, 11 — the diagonal derivatives), with the ghost
    // fields c_j acting on the fermionic (ghost) universes.
    //
    // Two structural facts are asserted:
    //   (a) Ω² = 0 — the constraint is first-class (nilpotent BRST charge;
    //       `nsBrst_nilpotent`, `nsDivergenceConstraint_resolution`);
    //   (b) [H, Ω] = 0 — the NS Hamiltonian (and the Eulerian fiber) is
    //       BRST-closed (gauge-invariant), the AGENTS.md requirement that
    //       physics Hamiltonians commute with the BRST charge.
    //
    // The orthogonal projector onto ker Ω itself (`project_physical`, CG on
    // ΩΩ†) is exercised in `brst.rs`'s own unit tests; for the *interacting*
    // NS charge the CG solve on the full ΩΩ† stalls (the operator expands the
    // state every iteration), so the tractable, exact assertions here are the
    // algebraic nilpotency and BRST-closure of the charge.
    let brst = navier_stokes_brst();

    // Ghost-carrying probes: a bosonic universe with diagonal-derivative
    // content (u_{j,j} on mode 3+3j+j) plus a fermionic universe with one
    // ghost in mode j (the mode annihilated by c_j), so Ω|ψ⟩ ≠ 0.
    let ghost_state = |bosonic: InnerBosonicState, ghost_mode: u32| -> QuantumState {
        QuantumState::vacuum()
            .apply(&Operator::OuterBosonCreate(bosonic))
            .apply(&Operator::OuterFermionCreate(InnerFermionicState {
                modes: std::collections::BTreeSet::from([ghost_mode]),
            }))
    };
    let boson = |mode: u32, n: u32| -> InnerBosonicState {
        let mut inner = InnerBosonicState::vacuum();
        if n > 0 {
            inner.modes.insert(mode, n);
        }
        inner
    };
    let probes = [
        // (a) ghost 0 on the diagonal derivative u_{0,0} (mode 3): Ω₀ ≠ 0.
        ghost_state(boson(3, 1), 0),
        // (b) ghost 1 on u_{1,1} (mode 7): Ω₁ ≠ 0.
        ghost_state(boson(7, 1), 1),
        // (c) ghost 2 on u_{2,2} (mode 11): Ω₂ ≠ 0.
        ghost_state(boson(11, 1), 2),
        // (d) ghost 0 on a velocity + diagonal-derivative state.
        ghost_state({
            let mut inner = boson(3, 1);
            inner.modes.insert(0, 1);
            inner
        }, 0),
        // (e) ghost 1 on a multi-mode state with u_{1,1} content.
        ghost_state({
            let mut inner = boson(4, 1);
            inner.modes.insert(0, 1);
            inner.modes.insert(11, 1);
            inner
        }, 1),
    ];

    // (a) Nilpotency Ω² = 0: the first-class nature of the constraint.
    for (i, p) in probes.iter().enumerate() {
        let twice = brst.apply(&brst.apply(p));
        assert!(
            twice.norm() < 1e-9,
            "Ω² must be nilpotent on probe {i}: ‖Ω²ψ‖ = {:.3e}",
            twice.norm()
        );
        // The probe must be genuinely *unphysical*: Ω acts on it nontrivially.
        assert!(
            brst.apply(p).norm() > 1e-6,
            "probe {i} must carry Ω-content, got ‖Ωψ‖ = {:.3e}",
            brst.apply(p).norm()
        );
    }

    // (b) Gauge invariance: [H, Ω] = 0 exactly, for the full NS Hamiltonian
    //     (velocity + derivative + viscous modes) and for the Eulerian fiber
    //     (velocity-only), on every ghost-carrying probe. (The pure-bosonic
    //     subspace is trivially Ω-closed: c_j annihilates a ghost-less state.)
    let h = navier_stokes_hamiltonian(1e-3);
    let a = [
        [1.0, 2.0, 0.5],
        [0.25, 1.0, 3.0],
        [1.5, 0.5, 2.0],
    ];
    let fiber = ns_eulerian_fiber(&a, &[0.1, -0.2, 0.3]);
    for (i, p) in probes.iter().enumerate() {
        let comm = commutator_norm(&h, &brst, p);
        assert!(
            comm < 1e-8,
            "the NS Hamiltonian must be BRST-closed on probe {i}: ‖[H,Ω]ψ‖ = {comm:.3e}"
        );
        let comm_f = commutator_norm(&fiber, &brst, p);
        assert!(
            comm_f < 1e-8,
            "the Eulerian fiber must be BRST-closed on probe {i}: ‖[H_f,Ω]ψ‖ = {comm_f:.3e}"
        );
    }

    // (c) The constraint is non-trivial in the flow: the physical subspace is
    //     *not* invariant under the bare (unprojected) evolution. Starting
    //     from a genuinely unphysical state (a ghost universe riding on
    //     diagonal-derivative content), the SIRK flow grows the Ω-content —
    //     ‖Ωψ(t)‖ > ‖Ωψ₀‖ — which is why the BRST projector must ride along
    //     in the solve (`Some(&brst)`).
    let opts = SirkOpts {
        prune_eps: 1e-10,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: true,
        unit_norm_steps: false,
    };
    let mut bosonic = InnerBosonicState::vacuum();
    bosonic.modes.insert(0, 1);
    bosonic.modes.insert(3, 1);
    let unphysical = QuantumState::vacuum()
        .apply(&Operator::OuterBosonCreate(bosonic))
        .apply(&Operator::OuterFermionCreate(InnerFermionicState {
            modes: std::collections::BTreeSet::from([0]),
        }));
    let omega0 = brst.apply(&unphysical).norm();
    assert!(
        omega0 > 1e-6,
        "the flow start must be genuinely unphysical: ‖Ωψ₀‖ = {omega0:.3e}"
    );
    let psi_t = evolve_restarted(&h, &unphysical, 0.05, 2, 2, &best_device(), None, &opts)
        .unwrap();
    let omega_t = brst.apply(&psi_t).norm();
    assert!(
        omega_t > omega0 + 1e-3,
        "the bare flow must grow the divergence (physical subspace not invariant): \
         ‖Ωψ₀‖ = {omega0:.3e} → ‖Ωψ(t)‖ = {omega_t:.3e}"
    );

    eprintln!(
        "ns_brst_projection: Ω² = 0 (nilpotent) on {n_probes} ghost probes; \
         [H,Ω] = 0 and [H_f,Ω] = 0 (BRST-closed); the bare flow grows the divergence \
         ‖Ωψ‖ {omega0:.3e} → {omega_t:.3e}",
        n_probes = probes.len()
    );
}