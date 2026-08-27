//! Cavity-QED Kerr photon blockade — numerical validation.
//!
//! The single-mode Kerr cavity `H = ω a†a + χ a†a†aa` ([`qed_kerr_cavity`])
//! has the closed-form anharmonic ladder
//!
//!     E_n = ωn + χ n(n−1),   E_{n+1} − E_n = ω + 2χn,
//!
//! and `[H, N] = 0` (photon number conserved). The physics verified here —
//! all exact quantum-optics predictions, observed experimentally in
//! cavity/circuit QED:
//!
//! 1. `qed_kerr_anharmonic_ladder_exact` — the SIRK Ritz value in each
//!    photon-number sector |n⟩ lands on the closed-form `E_n` to solver
//!    precision. Because `[H, N] = 0` the sectors are exactly the number
//!    sectors, so each solve is 1-dimensional and exact.
//! 2. `qed_kerr_photon_blockade_detuning` — the first transition is resonant
//!    at `ω` while the second is detuned by `2χ`:
//!    `(E₂ − E₁) − (E₁ − E₀) = 2χ`. A drive at `ω` therefore cannot absorb a
//!    second photon — the **photon blockade** (Imamoğlu–Schmidt–Woods–Deutsch
//!    1997): the cavity emits one photon at a time and acts as a single-photon
//!    source.
//! 3. `qed_kerr_photon_number_conservation` — the χ term preserves `N`:
//!    under real-time SIRK evolution from |2⟩, `⟨N⟩ = 2` and the norm and
//!    energy `E₂ = 2ω + 2χ` are conserved at every time.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{evolve_restarted, solve_forward_sirk_with_opts, SirkOpts};
use nested_fock_algebra::{
    InnerBosonicState, Operator, QuantumState, qed_free_photon, qed_kerr_cavity,
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
        unit_norm_steps: true,
    }
}

fn empty_universe_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// The `n`-photon Fock state |n⟩ in mode 0, normalized. (The ladder
/// construction leaves amplitude √(n!), so it is normalized here.)
fn n_photons(n: u32) -> QuantumState {
    let mut s = empty_universe_vacuum();
    for _ in 0..n {
        s = s.apply(&Operator::InnerBosonCreate(0));
    }
    let norm = QuantumState::norm(&s);
    let factor = Complex64::new(1.0 / norm - 1.0, 0.0);
    s.scale_and_add(&s.clone(), factor);
    s
}

/// The photon-number operator N = a†a on mode 0.
fn number_operator() -> nested_fock_algebra::Hamiltonian {
    nested_fock_algebra::Hamiltonian {
        terms: vec![(
            Complex64::new(1.0, 0.0),
            vec![
                Operator::InnerBosonCreate(0),
                Operator::InnerBosonAnnihilate(0),
            ],
        )],
    }
}

/// SIRK ground (only) Ritz value of the n-photon sector of `h`.
fn sector_energy(h: &nested_fock_algebra::Hamiltonian, n: u32) -> f64 {
    let res =
        solve_forward_sirk_with_opts(h, &n_photons(n), &shifts(6), &best_device(), None, &opts())
            .unwrap();
    let ritz = res.ritz_values();
    assert_eq!(ritz.len(), 1, "|{n}⟩ is 1-dimensional (N conserved): got {ritz:?}");
    ritz[0]
}

#[test]
fn qed_kerr_anharmonic_ladder_exact() {
    for (omega, chi) in [(1.0_f64, 0.1_f64), (1.3_f64, 0.2_f64), (0.7_f64, 0.35_f64)] {
        let h = qed_kerr_cavity(omega, chi);
        for n in 0..=4u32 {
            let e_n = omega * n as f64 + chi * (n as f64) * (n as f64 - 1.0);
            let got = sector_energy(&h, n);
            assert!(
                (got - e_n).abs() < 1e-7,
                "ω={omega}, χ={chi}, n={n}: E = {got}, closed form {e_n}"
            );
            eprintln!("qed_kerr_ladder: ω={omega}, χ={chi}, E_{n} = {got:.9} (exact {e_n:.9})");
        }
    }
}

#[test]
fn qed_kerr_chi_zero_is_free_photon_sector() {
    // At χ = 0 the Kerr cavity is the abelian gauge-fixed QED sector itself:
    // the normal-ordered free photon `qed_free_photon(ω)` — the U(1)
    // reduction of the Cadabra-derived H_final = ½π² + ½B²
    // (`docs/yang_mills_hamiltonian.cdb`). Both builders must give the
    // identical photon ladder {nω}.
    let omega = 1.4_f64;
    let h_kerr = qed_kerr_cavity(omega, 0.0);
    let h_photon = qed_free_photon(&[omega]);
    for n in 0..=3u32 {
        let e_kerr = sector_energy(&h_kerr, n);
        let e_photon = sector_energy(&h_photon, n);
        assert!(
            (e_kerr - e_photon).abs() < 1e-9,
            "n = {n}: Kerr(χ=0) gives {e_kerr}, qed_free_photon gives {e_photon}"
        );
        assert!(
            (e_kerr - omega * n as f64).abs() < 1e-7,
            "n = {n}: E = {e_kerr}, free-photon ladder nω = {}",
            omega * n as f64
        );
    }
    eprintln!(
        "qed_kerr_chi_zero: χ=0 Kerr ≡ qed_free_photon(ω = {omega}) on the ladder {{0, ω, 2ω, 3ω}}"
    );
}

#[test]
fn qed_kerr_photon_blockade_detuning() {
    let (omega, chi) = (1.0_f64, 0.3_f64);
    let h = qed_kerr_cavity(omega, chi);
    let e0 = sector_energy(&h, 0);
    let e1 = sector_energy(&h, 1);
    let e2 = sector_energy(&h, 2);
    let e3 = sector_energy(&h, 3);

    // First transition resonant at ω; each subsequent step up by 2χ.
    let d1 = e1 - e0;
    let d2 = e2 - e1;
    let d3 = e3 - e2;
    assert!(
        (d1 - omega).abs() < 1e-7,
        "|0⟩→|1⟩ must be resonant at ω: Δ₁ = {d1}, ω = {omega}"
    );
    assert!(
        (d2 - (omega + 2.0 * chi)).abs() < 1e-7,
        "|1⟩→|2⟩ detuned by 2χ: Δ₂ = {d2}, want ω+2χ = {}",
        omega + 2.0 * chi
    );
    assert!(
        (d3 - (omega + 4.0 * chi)).abs() < 1e-7,
        "|2⟩→|3⟩ detuned by 4χ: Δ₃ = {d3}, want ω+4χ = {}",
        omega + 4.0 * chi
    );
    // The blockade: the second transition is off-resonance from the first.
    let blockade = (d2 - d1).abs();
    assert!(
        (blockade - 2.0 * chi).abs() < 1e-7,
        "two-photon detuning must be 2χ = {}, got {blockade}",
        2.0 * chi
    );
    assert!(
        blockade > 1e-3,
        "blockade detuning must be resolvable, got {blockade}"
    );
    eprintln!(
        "qed_kerr_blockade: Δ₁ = {d1:.9} (ω = {omega}), Δ₂ = {d2:.9} (ω+2χ = {}), \
         Δ₃ = {d3:.9} (ω+4χ = {}); blockade 2χ = {blockade:.9}",
        omega + 2.0 * chi,
        omega + 4.0 * chi
    );
}

#[test]
fn qed_kerr_photon_number_conservation() {
    let (omega, chi) = (1.0_f64, 0.2_f64);
    let h = qed_kerr_cavity(omega, chi);
    let n_op = number_operator();
    let e2_exact = 2.0 * omega + 2.0 * chi; // E₂ = ω·2 + χ·2·1
    let psi0 = n_photons(2);

    for &t in &[0.0_f64, 1.0, 5.0, 17.3] {
        let psi_t = evolve_restarted(&h, &psi0, t, 60, 8, &best_device(), None, &opts()).unwrap();
        let norm = QuantumState::inner_product(&psi_t, &psi_t).re;
        assert!(
            (norm - 1.0).abs() < 1e-8,
            "t = {t}: norm = {norm}, must stay 1"
        );
        let n_expect = QuantumState::inner_product(&psi_t, &n_op.apply(&psi_t)).re;
        assert!(
            (n_expect - 2.0).abs() < 1e-8,
            "t = {t}: ⟨N⟩ = {n_expect}, must stay 2 (Kerr term is number-conserving)"
        );
        let e_expect = QuantumState::inner_product(&psi_t, &h.apply(&psi_t)).re;
        assert!(
            (e_expect - e2_exact).abs() < 1e-7,
            "t = {t}: ⟨H⟩ = {e_expect}, must stay E₂ = {e2_exact}"
        );
        eprintln!(
            "qed_kerr_number: t = {t}: ⟨N⟩ = {n_expect:.12}, ⟨H⟩ = {e_expect:.12} (E₂ = {e2_exact})"
        );
    }
}
