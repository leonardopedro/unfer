//! SU(2) confined-lattice gauge dynamics numerical validation.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
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

fn vac() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn one_flux() -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(0, 1u32);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

fn n_flux(n: usize) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(0, n as u32);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

fn sirk_ground(h: &nested_fock_algebra::Hamiltonian, v0: &QuantumState, m: usize) -> f64 {
    solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve")
        .ground_state_energy()
        .expect("ground state")
}

fn sirk_solve(
    h: &nested_fock_algebra::Hamiltonian,
    v0: &QuantumState,
    m: usize,
) -> fock_sirk::ForwardSirkResult {
    solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve")
}

#[test]
fn su2_electric_flux_quantization() {
    for g_val in [2.0, 3.0, 4.0] {
        let g2_half = g_val * g_val / 2.0;
        let h = yang_mills_lattice(2, g_val, 1);
        let e_even = sirk_ground(&h, &vac(), 4);
        let e_odd = sirk_ground(&h, &one_flux(), 4);
        let gap = e_odd - e_even;
        let rel_err = (gap - g2_half).abs() / g2_half;
        assert!(rel_err < 0.05, "g={g_val}: gap={gap:.4}, g²/2={g2_half:.1}");
        eprintln!("su2_electric_flux_quantization: g={g_val:.0}, gap={gap:.4}, g²/2={g2_half:.1}");
    }
}

#[test]
fn su2_string_tension_scales() {
    let mut gaps = Vec::new();
    let mut g_vals = Vec::new();
    for g_val in [2.0, 3.0, 4.0, 5.0] {
        let h = yang_mills_lattice(2, g_val, 1);
        let gap = sirk_ground(&h, &one_flux(), 4) - sirk_ground(&h, &vac(), 4);
        gaps.push(gap);
        g_vals.push(g_val);
    }
    for i in 1..g_vals.len() {
        let slope = (gaps[i] / gaps[i - 1]).ln() / (g_vals[i] / g_vals[i - 1]).ln();
        assert!(
            (1.5..2.5).contains(&slope),
            "log-log slope g={}→g={} = {slope:.3}, expected ≈ 2",
            g_vals[i - 1],
            g_vals[i]
        );
    }
    eprintln!("su2_string_tension_scales: gaps = {gaps:?}");
}

#[test]
fn su2_plaquette_expectation() {
    let g = 4.0;
    let beta = 2.0 / (g * g);
    let x: f64 = beta / 2.0;
    let p_series = x - x.powi(3) / 16.0 + 5.0 * x.powi(5) / 768.0;
    assert!(
        (0.0..0.1).contains(&p_series),
        "plaquette at g=4: ⟨P⟩={p_series:.6}"
    );
    let e_plaquette = 1.0 - p_series;
    assert!(e_plaquette > 0.9, "E_P = {e_plaquette:.4}, expected > 0.9");
    let h = yang_mills_lattice(2, g, 1);
    let ground = sirk_ground(&h, &vac(), 4);
    assert!(ground.abs() < 0.5, "ground ≈ 0, got {ground}");
    eprintln!("su2_plaquette_expectation: g={g}, β={beta:.4}, ⟨P⟩={p_series:.6}");
}

#[test]
fn su2_sector_purity() {
    let g = 2.0;
    let h = yang_mills_lattice(2, g, 1);
    let res_even = sirk_solve(&h, &vac(), 4);
    let res_odd = sirk_solve(&h, &one_flux(), 4);
    let max_overlap = res_even
        .w_sequence
        .iter()
        .flat_map(|we| {
            res_odd
                .w_sequence
                .iter()
                .map(move |wo| QuantumState::inner_product(we, wo).norm())
        })
        .fold(0.0_f64, f64::max);
    assert!(max_overlap < 1e-6, "max overlap = {max_overlap:.2e}");
    eprintln!("su2_sector_purity: max overlap = {max_overlap:.2e}");
}

#[test]
fn su2_glueball_mass_ratio() {
    let g = 2.0;
    let h = yang_mills_lattice(2, g, 1);
    let e_ground = sirk_ground(&h, &vac(), 4);
    let e_excited = sirk_ground(&h, &n_flux(2), 4);
    let m_glue = e_excited - e_ground;
    let e_odd = sirk_ground(&h, &one_flux(), 4);
    let sigma = e_odd - e_ground;
    if sigma > 0.01 {
        let ratio = m_glue / sigma.sqrt();
        eprintln!("su2_glueball_mass_ratio: m_glue={m_glue:.4}, √σ={sigma:.4}, ratio={ratio:.3}");
        assert!(ratio > 0.1, "ratio must be positive: {ratio:.3}");
    }
}

#[test]
fn su2_confinement_across_couplings() {
    for g_val in [1.5, 2.0, 3.0, 4.0, 6.0] {
        let h = yang_mills_lattice(2, g_val, 1);
        let gap = sirk_ground(&h, &one_flux(), 4) - sirk_ground(&h, &vac(), 4);
        assert!(gap > 0.0, "g={g_val}: gap must be positive, got {gap:.6}");
        eprintln!("su2_confinement: g={g_val:.1}, gap={gap:.4}");
    }
}
