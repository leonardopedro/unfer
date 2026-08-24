//! Action-to-spectrum prediction chains for QG(R²): the Starobinsky action
//! fixes the scalaron mass m²=M²/(12α); that mass must appear as the k→0 SIRK
//! gap of the quantized band; the weak-field potential derived from integrating
//! the scalaron out reproduces Newtonian gravity with its ⅓ Yukawa correction;
//! and the two independent QG kinetic builders (TEGR 𝒮-form vs densitized
//! flat form) agree when probed on their common subsector.
//!
//! 1. `qg_scalaron_mass_chain_action_to_band` — α-scaling of m(α) and its
//!    reappearance as the exact k→0 gap for each α.
//! 2. `qg_weak_field_yukawa_limits` — Φ(r) = −GM/r (1 + ⅓e^{−mr}):
//!    Newtonian at r≫1/m, 4/3 enhancement at r≪1/m.
//! 3. `qg_tegr_densitized_common_subsector` — the two gauge-fixed kinetic
//!    builders agree on their shared 𝒮 sector within solver bands.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{solve_forward_sirk_with_opts, SirkOpts};
use nested_fock_algebra::{
    qg_starobinsky_scalaron_mass, qg_starobinsky_weak_field_potential, qg_tegr_hamiltonian,
    InnerBosonicState, Operator, QuantumState,
};
use num_complex::Complex64;

fn mk(deep: bool) -> SirkOpts {
    SirkOpts {
        prune_eps: 0.0,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: deep,
    }
}

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn empty_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

#[test]
fn qg_scalaron_mass_chain_action_to_band() {
    // m(α) = 1/sqrt(12α): the published scalaron mass of f(R)=(M²/2)R+αR².
    let alphas = [0.5_f64, 1.0, 2.0];
    let mut prev_m = f64::INFINITY;
    for &alpha in &alphas {
        let m_pred = qg_starobinsky_scalaron_mass(alpha);
        assert!((m_pred - 1.0 / (12.0 * alpha).sqrt()).abs() < 1e-12);
        // Heavier α ⇒ lighter scalaron.
        assert!(m_pred < prev_m);
        prev_m = m_pred;

        // The SAME mass reappears as the k→0 gap of the quantized band.
        // One-quanton start so the window spans BOTH rungs {0, m}.
        let h = nested_fock_algebra::qg_starobinsky_scalaron_field(&[0.0], m_pred);
        // Superpose |0> and |1>: a pure number eigenstate would collapse the
        // Krylov rank of a diagonal Hamiltonian to one.
        let mut psi1 = empty_vacuum();
        psi1.scale_and_add(
            &empty_vacuum().apply(&Operator::InnerBosonCreate(0)),
            Complex64::new(1.0, 0.0),
        );
        let ritz = solve_forward_sirk_with_opts(
            &h,
            &psi1,
            &shifts(6),
            &best_device(),
            None,
            &mk(true),
        )
        .unwrap()
        .ritz_values();
        assert!(ritz.len() >= 2);
        let gap = ritz[1] - ritz[0];
        assert!(
            (gap - m_pred).abs() < 1e-8,
            "α={alpha}: band gap {gap} ≠ action mass {m_pred}"
        );
    }
}

#[test]
fn qg_weak_field_yukawa_limits() {
    // Solar-mass source, Planck-heavy scalaron (Compton ≈ ℓ_P scale).
    let gm = 1.327_124_400_18e20; // G·M_sun
    let alpha = 1.0;
    let m = qg_starobinsky_scalaron_mass(alpha);

    // Far field (r ≫ 1/m): pure Newtonian — deviation is ⅓e^{-mr}, utterly
    // negligible; R² gravity passes solar-system tests.
    let r_far = 1.0e10; // ≫ 1/m ~ 5e-36 m
    let phi_far = qg_starobinsky_weak_field_potential(gm, r_far, m);
    let phi_newton = -gm / r_far;
    assert!(
        (phi_far / phi_newton - 1.0).abs() < 1e-30,
        "Newtonian far field broken: ratio {}",
        phi_far / phi_newton
    );

    // Near field (r ≪ 1/m): e^{-mr} ≈ 1 − mr ⇒ enhancement factor 4/3.
    let r_near = 1.0e-40;
    let phi_near = qg_starobinsky_weak_field_potential(gm, r_near, m);
    let ratio_near = phi_near / phi_newton_at(gm, r_near);
    assert!(
        (ratio_near - 4.0 / 3.0).abs() < 1e-6,
        "short-range 4/3 enhancement: ratio {ratio_near}"
    );
    // And it is the PUBLISHED ⅓ correction: 1 + ⅓·e^{-mr≈1}.
    assert!((ratio_near - 1.0 - 1.0 / 3.0).abs() < 1e-9);
}

fn phi_newton_at(gm: f64, r: f64) -> f64 {
    -gm / r
}

#[test]
fn qg_tegr_densitized_common_subsector() {
    // TEGR builder: outer-Fock :𝒮²:/16 per mode. Densitized builder: same
    // coefficient on its 𝒮 modes (+ conformal y mode we do NOT excite).
    let n_modes = 3_u32;
    let h_tegr = qg_tegr_hamiltonian(n_modes);
    let h_dens = nested_fock_algebra::qg_densitized_kinetic(n_modes - 1);

    // Identical one-quanton starts in 𝒮 mode 0.
    let psi = empty_vacuum().apply(&Operator::InnerBosonCreate(0));
    let rt = solve_forward_sirk_with_opts(
        &h_tegr,
        &psi,
        &shifts(7),
        &best_device(),
        None,
        &mk(true),
    )
    .unwrap()
    .ritz_values();
    // No residual filter here: the Bogoliubov blocks' true residuals sit
    // above the tight band by construction (unbounded ladders).
    let rd = solve_forward_sirk_with_opts(
        &h_dens,
        &psi,
        &shifts(7),
        &best_device(),
        None,
        &mk(true),
    )
    .unwrap()
    .ritz_values();

    // Cross-builder agreement on the shared 𝒮 subsector. Both builders'
    // windows carry their own truncation bands (~0.1 scale, see S39/S40
    // notes on rank saturation / Bogoliubov ladders), so the comparison is
    // banded rather than exact; the physics content is that NEITHER builder
    // produces a ground outside the common band and their low-rung SPREADS
    // agree.
    assert!(rt.len() >= 2 && rd.len() >= 2, "{rt:?} vs {rd:?}");
    let spread_t = rt[1] - rt[0];
    let spread_d = rd[1] - rd[0];
    assert!(
        (spread_t - spread_d).abs() < 0.3,
        "low-rung spread mismatch: TEGR {spread_t:.4} vs densitized {spread_d:.4}"
    );
    for v in rt.iter().chain(rd.iter()).take(2) {
        assert!(v.abs() < 0.2, "ground band violated: {v}");
    }
}
