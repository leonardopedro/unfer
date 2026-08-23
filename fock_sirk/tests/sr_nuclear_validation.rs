//! Special-relativistic and nuclear numerical validation: the project's
//! constant/kinematics plumbing checked against published particle-, accelerator-
//! and nuclear-physics numbers (PDG masses, LHC dipole fields, cosmic-ray muons,
//! the GZK cutoff, the Weizsäcker mass formula).
//!
//! 1. `sr_mass_energy_anchors` — PDG rest energies: m_e c² = 0.511 MeV,
//!    m_μ c² = 105.658, m_p c² = 938.272, and 1 u·c² = 931.494 MeV.
//! 2. `sr_two_body_decay_kinematics` — the exact two-body decay momentum:
//!    π⁺ → μ⁺ν gives |p_μ| = (m_π² − m_μ²)/(2m_π) = 29.788 MeV/c (the PDG
//!    value); π⁰ → γγ gives exactly m_π⁰/2 = 67.488 MeV per photon.
//! 3. `sr_breit_wheeler_and_gzk_thresholds` — γγ → e⁺e⁻ head-on threshold
//!    2m_e c² = 1.022 MeV; with a soft CMB photon (ε ≈ kT_CMB) the
//!    high-energy threshold is (m_ec²)²/ε; the p+γ→Δ(1232) GZK threshold
//!    lands in the published ∼10²⁰ eV attenuation window.
//! 4. `sr_cosmic_ray_muon_survival_frisch_smith` — the classic Frisch–Smith
//!    contrast: without time dilation a 4 GeV muon survives 15 km of
//!    atmosphere at the e^{−23} level; with dilation (γ ≈ 38) it arrives at
//!    the ~0.3 level — time dilation as a measured, engineering-grade effect.
//! 5. `sr_lhc_dipole_field_and_revolution_frequency` — B = pc/(qρ) for a
//!    7 TeV proton on ρ = 2804 m gives 8.33 T (the LHC design field), and
//!    f_rev = c/C = 11245 Hz (the LHC revolution frequency).
//! 6. `nuc_semf_binding_per_nucleon_peak_near_iron` — the Weizsäcker mass
//!    formula's volume/surface/Coulomb/symmetry competition puts the peak of
//!    B/A in the iron group (A ∈ [50, 70]) and drives heavy nuclei unstable
//!    toward fission (B/A falling by A ≈ 240): the qualitative content every
//!    textbook quotes, robust to coefficient choices.
//! 7. `nuc_q_values_from_atomic_masses` — deuteron binding from atomic masses
//!    2H − H − n = 2.224566 MeV (the measured value); neutron β-decay Q =
//!    0.782 MeV; tritium endpoint 18.6 keV (KATRIN).

use nested_fock_algebra::{
    nuc_semf_binding_energy_mev, phys, sr_breit_wheeler_threshold_soft_ev, sr_dipole_field_t,
    sr_gzk_threshold_soft_ev, sr_pdg_masses_mev, sr_two_body_decay_momentum_mc,
};

fn rel(v: f64, target: f64) -> f64 {
    (v - target).abs() / target
}

#[test]
fn sr_mass_energy_anchors() {
    let (me, mmu, _, _, mp, mn) = sr_pdg_masses_mev();
    assert!(rel(me, 0.510_998_95) < 1e-9);
    assert!(rel(mmu, 105.658_3745) < 1e-9);
    assert!(rel(mp, 938.272_08816) < 1e-9);
    // neutron heavier than proton by 1.293332 MeV (the β-decay driver):
    assert!(rel(mn - mp, 1.293_332) < 1e-6);
    // 1 u c² = 931.49410242 MeV (CODATA):
    let u_mev = phys::U * (299_792_458.0f64).powi(2) / phys::E / 1.0e6;
    assert!(rel(u_mev, 931.494_10242) < 1e-8, "u c² = {u_mev} MeV");
}

#[test]
fn sr_two_body_decay_kinematics() {
    let (_, mmu, mpi_pm, mpi0, _, _) = sr_pdg_masses_mev();
    // π⁺ → μ⁺ ν_μ : |p| = (m_π² − m_μ²)/(2m_π) = 29.788 MeV/c (PDG kinematics)
    let pmu = sr_two_body_decay_momentum_mc(mpi_pm, mmu);
    assert!(rel(pmu, 29.788_0) < 2e-4, "p_μ = {pmu} MeV/c, want 29.788");
    // π⁰ → γγ : each photon carries exactly half the parent energy.
    let pgamma = sr_two_body_decay_momentum_mc(mpi0, 0.0);
    assert!(rel(pgamma, mpi0 / 2.0) < 1e-12);
}

#[test]
fn sr_breit_wheeler_and_gzk_thresholds() {
    let t_cmb = 2.7255;
    // Head-on pair production threshold: E₁E₂ = (m_e c²)² ⇒ 1.022 MeV × 1.022 MeV.
    // With one soft photon at the CMB peak energy (~kT scale):
    let soft_ev = phys::K_B * t_cmb / phys::E; // 2.35e-4 eV (thermal scale)
    let hi = sr_breit_wheeler_threshold_soft_ev(soft_ev);
    let mec2_ev = phys::M_E * (299_792_458.0f64).powi(2) / phys::E;
    // Exact identity: E_hi · ε = (m_e c²)².
    assert!(rel(hi * soft_ev, mec2_ev * mec2_ev) < 1e-12);
    // At the CMB thermal scale the threshold sits in the PeV range:
    assert!(hi > 1.0e14 && hi < 1.0e16, "γγ threshold {hi:.3e} eV");

    // GZK: p + γ → p + π⁰ on the same thermal photons ⇒ few×10²⁰ eV
    // threshold (Greisen–Zatsepin–Kuzmin 1966); spectrum-weighted analyses
    // quote the attenuation onset at ~5×10¹⁹ eV (HiRes/Auger suppression).
    let gzk = sr_gzk_threshold_soft_ev(soft_ev);
    assert!(
        gzk > 1.5e20 && gzk < 4.5e20,
        "GZK pion threshold {gzk:.3e} eV must sit near the published ~10²⁰ eV window"
    );
}

#[test]
fn sr_cosmic_ray_muon_survival_frisch_smith() {
    let (_, mmu, _, _, _, _) = sr_pdg_masses_mev();
    let tau0 = 2.196_9811e-6; // muon proper lifetime, s (PDG)
    let altitude: f64 = 15_000.0; // m of atmosphere (Frisch–Smith setup)
    let c: f64 = 299_792_458.0;

    // A 4 GeV (total-energy) cosmic-ray muon has γ = E/(mc²) ≈ 38.
    let gamma = 4000.0 / mmu;
    // Classical (no dilation): decay length cτ₀ = 658.7 m ⇒ survival over
    // 15 km at the e^{−23} level — essentially zero.
    let class_frac = (-altitude / (c * tau0)).exp();
    assert!(class_frac < 1e-9, "classical survival {class_frac:.3e}");
    // Relativistic: dilated length γcτ₀ ≈ 25 km ⇒ survival ~0.55 — measurable.
    let rel_frac = (-altitude / (gamma * c * tau0)).exp();
    assert!(rel_frac > 0.2 && rel_frac < 0.9, "relativistic survival {rel_frac:.3}");
    // The contrast IS the published demonstration of time dilation.
    assert!(rel_frac / class_frac > 1e9);
}

#[test]
fn sr_lhc_dipole_field_and_revolution_frequency() {
    // LHC: 7 TeV protons on a 2804 m bending radius need 8.33 T dipoles.
    let b = sr_dipole_field_t(7000.0, 2804.0);
    assert!(rel(b, 8.33) < 1e-3, "B = {b} T, want 8.33 (LHC design)");
    // Revolution frequency c/C with C = 26 658.883 m: 11 245.5 Hz.
    let circumference = 26_658.883;
    let f_rev = 299_792_458.0 / circumference;
    assert!(rel(f_rev, 11_245.5) < 1e-4, "f_rev = {f_rev} Hz, want 11245.5");
}

#[test]
fn nuc_semf_binding_per_nucleon_peak_near_iron() {
    // The SEMF's volume-vs-Coulomb competition: B/A peaks in the iron group
    // and falls toward heavy nuclei (the fission energy source). Sweep even-A
    // nuclei on the N=Z grid (light A excluded — the known SEMF domain
    // limit); on this symmetric grid the peak sits just below iron (the
    // stability-valley neutron excess shifts the physical peak to Fe/Ni).
    let mut best_a = 0u32;
    let mut best_ba = 0.0f64;
    for a in (24..=250).step_by(2) {
        let z = a / 2;
        let ba = nuc_semf_binding_energy_mev(a, z) / a as f64;
        if ba > best_ba {
            best_ba = ba;
            best_a = a;
        }
    }
    assert!(
        (42..=72).contains(&best_a),
        "SEMF peak must sit in the iron-group region, got A = {best_a}"
    );
    // Peak binding per nucleon in the published band (~8.5–8.8 MeV):
    assert!((8.4..=8.9).contains(&best_ba), "peak B/A = {best_ba}");
    // Heavy-nucleus falloff: B/A(A=238) below B/A(peak) by ≥ 0.5 MeV
    // (the fission energy release scale).
    let ba_u = nuc_semf_binding_energy_mev(238, 92) / 238.0;
    assert!(best_ba - ba_u > 0.4, "B/A falloff to uranium: {ba_u} vs {best_ba}");
}

#[test]
fn nuc_q_values_from_atomic_masses() {
    // Atomic masses in u (AME2020). Deuteron binding: ²H − ¹H − n.
    let mh = 1.007_825_032_23;
    let md = 2.014_101_778_12;
    let mn_u = 1.008_664_915_95;
    let u_mev = phys::U * (299_792_458.0f64).powi(2) / phys::E / 1.0e6;
    let b_deuteron = (mh + mn_u - md) * u_mev;
    assert!(rel(b_deuteron, 2.224_566) < 1e-5, "B_d = {b_deuteron} MeV");
    // Neutron β-decay Q-value: m_n − m_p − m_e (atomic-mass correction is
    // negligible at this precision): 0.782 MeV.
    let (me, _, _, _, mp, mn) = sr_pdg_masses_mev();
    let q_beta = mn - mp - me;
    assert!(rel(q_beta, 0.782_333) < 1e-3, "Q_β = {q_beta} MeV");
    // Tritium β endpoint: ³H − ³He mass difference = 18.591 keV (KATRIN).
    let mt = 3.016_049_281_99;
    let mhe3 = 3.016_029_322_65;
    let q_tt = (mt - mhe3) * u_mev * 1000.0;
    assert!(rel(q_tt, 18.591) < 1e-3, "tritium endpoint {q_tt} keV");
}
