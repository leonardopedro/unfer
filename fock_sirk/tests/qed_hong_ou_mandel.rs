//! Hong–Ou–Mandel bunching and beamsplitter statistics — numerical validation.
//!
//! The two-mode beamsplitter `H = J(a†₀a₁ + a†₁a₀)` ([`oscillator_beamsplitter`]
//! with ω = 0) is twice the Schwinger SU(2) generator `J_x`; evolution by `t`
//! rotates by `θ = 2Jt` about the x axis. `θ = π/2` (i.e. `t = π/(4J)`) is the
//! ideal 50:50 beamsplitter. The predictions verified here are exact
//! quantum-optics results (measured with high precision in photonics
//! experiments):
//!
//! 1. `qed_hong_ou_mandel_bunching` — two indistinguishable photons in the
//!    |1,1⟩ coincidence state exit a 50:50 beamsplitter *bunched*:
//!    `|1,1⟩ → (|2,0⟩ − |0,2⟩)/√2` (the spin-1 rotation `d¹_{00}(π/2) = 0`),
//!    so the coincidence probability `P₁₁ = 0` while `P₂₀ = P₀₂ = ½`. The
//!    dip of the coincidence rate to zero is the **Hong–Ou–Mandel effect**
//!    (1987), the experimental signature of bosonic indistinguishability.
//! 2. `qed_beamsplitter_balanced_splitting` — a single photon |1,0⟩ splits
//!    evenly at a 50:50 beamsplitter: `P₁₀ = P₀₁ = ½` (the spin-½ rotation).
//! 3. `qed_beamsplitter_unitarity_energy` — the restarted-SIRK evolution is
//!    unitary (norm conserved) and conserves the energy `⟨H⟩` exactly.
//! 4. `qed_hom_coincidence_curve_cos2_theta` — the full coincidence curve:
//!    `P₁₁(θ) = cos²(θ)` (`d¹₀₀(θ) = cos θ` of the spin-1 rotation), the
//!    Hong–Ou–Mandel dip at θ = π/2 flanked by full coincidence at θ = 0, π.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{evolve_restarted, solve_forward_sirk_with_opts, SirkOpts};
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, QuantumState, oscillator_beamsplitter,
    qcd_ym_hamiltonian,
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

/// Fock state with `n0` photons in mode 0 and `n1` photons in mode 1.
fn fock(n0: u32, n1: u32) -> QuantumState {
    let mut s = empty_universe_vacuum();
    for _ in 0..n0 {
        s = s.apply(&Operator::InnerBosonCreate(0));
    }
    for _ in 0..n1 {
        s = s.apply(&Operator::InnerBosonCreate(1));
    }
    s
}

/// 50:50 beamsplitter time from the coupling `j` (rotation angle θ = 2jt = π/2).
fn bs_time(j: f64) -> f64 {
    std::f64::consts::FRAC_PI_2 / (2.0 * j)
}

/// Evolve `psi0` under `h` to time `t` with restarted SIRK.
fn evolve(h: &nested_fock_algebra::Hamiltonian, psi0: &QuantumState, t: f64) -> QuantumState {
    evolve_restarted(h, psi0, t, 40, 8, &best_device(), None, &opts()).unwrap()
}

/// Population of the Fock state `target` inside `psi`: |⟨target|ψ⟩|².
/// `target` is built via ladder operators (amplitude √(n!)), so it is
/// normalized here before projecting.
fn population(psi: &QuantumState, target: &QuantumState) -> f64 {
    let norm_t = QuantumState::inner_product(target, target).re.sqrt();
    QuantumState::inner_product(target, psi).norm().powi(2) / norm_t.powi(2)
}

#[test]
fn qed_hong_ou_mandel_bunching() {
    let j = 1.0_f64;
    let h = oscillator_beamsplitter(0.0, j);
    let psi11 = fock(1, 1);
    let psi20 = fock(2, 0);
    let psi02 = fock(0, 2);

    // |1,1⟩ through a 50:50 beamsplitter: the coincidence path cancels
    // destructively, leaving only the bunched outputs (±|2,0⟩ ± |0,2⟩)/√2.
    let psi_t = evolve(&h, &psi11, bs_time(j));

    let p11 = population(&psi_t, &psi11);
    let p20 = population(&psi_t, &psi20);
    let p02 = population(&psi_t, &psi02);
    let p_bunched = p20 + p02;

    assert!(
        p11 < 1e-5,
        "HOM dip: coincidence P₁₁ = {p11}, must vanish at a 50:50 beamsplitter"
    );
    assert!(
        (p20 - 0.5).abs() < 1e-5 && (p02 - 0.5).abs() < 1e-5,
        "bunched outputs must share the weight: P₂₀ = {p20}, P₀₂ = {p02}"
    );
    assert!(
        (p_bunched - 1.0).abs() < 1e-5,
        "total bunching probability P₂₀+P₀₂ = {p_bunched}, must be 1"
    );
    let norm = QuantumState::inner_product(&psi_t, &psi_t).re;
    assert!((norm - 1.0).abs() < 1e-9, "norm = {norm}");

    eprintln!(
        "qed_hong_ou_mandel: |1,1⟩ → P₁₁ = {p11:.3e} (dip), P₂₀ = {p20:.6}, P₀₂ = {p02:.6}"
    );
}

#[test]
fn qed_beamsplitter_balanced_splitting() {
    let j = 0.8_f64;
    let h = oscillator_beamsplitter(0.0, j);
    let psi10 = fock(1, 0);
    let psi01 = fock(0, 1);

    // A single photon |1,0⟩ is split evenly: |ψ⟩ = (|1,0⟩ − i|0,1⟩)/√2.
    let psi_t = evolve(&h, &psi10, bs_time(j));

    let p10 = population(&psi_t, &psi10);
    let p01 = population(&psi_t, &psi01);
    assert!(
        (p10 - 0.5).abs() < 1e-5 && (p01 - 0.5).abs() < 1e-5,
        "balanced splitting: P₁₀ = {p10}, P₀₁ = {p01}, both must be ½"
    );
    assert!(
        (p10 + p01 - 1.0).abs() < 1e-5,
        "single photon must stay in the one-photon sector: P₁₀+P₀₁ = {}",
        p10 + p01
    );

    // Same statement through the mode-1 population expectation ⟨N₁⟩ = ½.
    let n1 = nested_fock_algebra::Hamiltonian {
        terms: vec![(
            Complex64::new(1.0, 0.0),
            vec![
                Operator::InnerBosonCreate(1),
                Operator::InnerBosonAnnihilate(1),
            ],
        )],
    };
    let n1_expect = QuantumState::inner_product(&psi_t, &n1.apply(&psi_t)).re;
    assert!(
        (n1_expect - 0.5).abs() < 1e-5,
        "⟨N₁⟩ = {n1_expect}, must be ½ after the 50:50 beamsplitter"
    );
    eprintln!(
        "qed_beamsplitter_balanced: P₁₀ = {p10:.6}, P₀₁ = {p01:.6}, ⟨N₁⟩ = {n1_expect:.6}"
    );
}

#[test]
fn qed_hom_coincidence_curve_cos2_theta() {
    // The coincidence probability of |1,1⟩ under the rotation θ = 2Jt is the
    // spin-1 Wigner d-function squared: P₁₁(θ) = |d¹₀₀(θ)|² = cos²(θ). Full
    // coincidence at θ = 0, π; the HOM dip (zero) at θ = π/2; halfway at
    // θ = π/4, 3π/4. This is the experimentally measured coincidence curve of
    // the HOM interferometer as the beamsplitter reflectivity is varied.
    let j = 1.0_f64;
    let h = oscillator_beamsplitter(0.0, j);
    let psi11 = fock(1, 1);

    for &theta in &[
        0.0_f64,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_2,
        2.0 * std::f64::consts::FRAC_PI_3,
        3.0 * std::f64::consts::FRAC_PI_4,
        std::f64::consts::PI,
    ] {
        let psi_t = evolve(&h, &psi11, theta / (2.0 * j));
        let p11 = population(&psi_t, &psi11);
        let expected = theta.cos().powi(2);
        assert!(
            (p11 - expected).abs() < 1e-5,
            "θ = {theta:.4}: P₁₁ = {p11:.8}, cos²(θ) = {expected:.8}"
        );
        eprintln!("qed_hom_coincidence_curve: θ = {theta:.4}, P₁₁ = {p11:.8} (cos²θ = {expected:.8})");
    }
}

#[test]
fn qed_beamsplitter_unitarity_energy() {
    let j = 1.0_f64;
    let h = oscillator_beamsplitter(0.0, j);
    let psi11 = fock(1, 1);
    // ⟨H⟩₀ for |1,1⟩: H|1,1⟩ = J√2(|2,0⟩ + |0,2⟩) (computed in the test below
    // via the exact two-level sector instead of hard-coding).
    let res =
        solve_forward_sirk_with_opts(&h, &psi11, &shifts(8), &best_device(), None, &opts())
            .unwrap();
    let ritz = res.ritz_values();
    // The Krylov space from |1,1⟩ is the *symmetric* N = 2 subspace
    // span{|1,1⟩, (|2,0⟩+|0,2⟩)/√2}, where H = [[0, 2J],[2J, 0]]: the exact
    // spectrum is {−2J, +2J}. (The antisymmetric state (|2,0⟩−|0,2⟩)/√2 is
    // the E = 0 partner; it is orthogonal to |1,1⟩ and never entered.)
    assert_eq!(ritz.len(), 2, "symmetric N = 2 sector has two levels: {ritz:?}");
    assert!(
        (ritz[0] + 2.0 * j).abs() < 1e-8 && (ritz[1] - 2.0 * j).abs() < 1e-8,
        "symmetric N = 2 sector spectrum must be {{−2J, +2J}}: {ritz:?}"
    );

    // Unitarity + energy conservation along the full rotation.
    let e0 = QuantumState::inner_product(&psi11, &h.apply(&psi11)).re; // = 0 exactly
    for &t in &[0.0_f64, bs_time(j) / 3.0, bs_time(j), 2.0 * bs_time(j)] {
        let psi_t = evolve(&h, &psi11, t);
        let norm = QuantumState::inner_product(&psi_t, &psi_t).re;
        let e_t = QuantumState::inner_product(&psi_t, &h.apply(&psi_t)).re;
        assert!((norm - 1.0).abs() < 1e-8, "t = {t}: norm = {norm}");
        assert!(
            (e_t - e0).abs() < 1e-7,
            "t = {t}: ⟨H⟩ = {e_t}, must stay {e0}"
        );
        eprintln!("qed_beamsplitter_unitarity: t = {t:.4}, ‖ψ‖ = {norm:.12}, ⟨H⟩ = {e_t:.3e}");
    }
}

#[test]
fn qed_hom_bunching_from_abelian_gauge_fixed_hopping() {
    // The beamsplitter generator is a SECTOR of the Cadabra-derived abelian
    // gauge-fixed Hamiltonian `qcd_ym_hamiltonian(0)` (`docs/yang_mills_
    // hamiltonian.cdb`: H_final = ½π² + ½B² with B = A₀ − A₁): the B²
    // cross-term −:A₀A₁: carries the number-conserving hopping
    // −(a†₀a₁ + a†₁a₀) alongside the pair and squeezing terms of the full
    // Hamiltonian. Filtering the builder's terms down to the cross-mode
    // hopping therefore yields exactly the 50:50 beamsplitter generator at
    // J = −1, so SIRK–Hashimoto on this sector must reproduce the full HOM
    // coincidence curve P₁₁(θ) = cos²θ and the bunching dip — the
    // beamsplitter kinematics predicted from the gauge-fixed Hamiltonian
    // itself, not from a separate model builder.
    let h_gf = qcd_ym_hamiltonian(0.0);

    // Identify the hopping terms of the abelian gauge-fixed H: one creation
    // and one annihilation on the two FIELD modes (cross-mode, so it
    // conserves photon number). Sum their coefficients per direction.
    let mut c01 = Complex64::new(0.0, 0.0); // coefficient of a†₀a₁
    let mut c10 = Complex64::new(0.0, 0.0); // coefficient of a†₁a₀
    let mut hop_terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for (coeff, ops) in &h_gf.terms {
        if ops.len() != 2 {
            continue;
        }
        match (&ops[0], &ops[1]) {
            (Operator::InnerBosonCreate(0), Operator::InnerBosonAnnihilate(1)) => {
                c01 += coeff;
                hop_terms.push((*coeff, ops.clone()));
            }
            (Operator::InnerBosonCreate(1), Operator::InnerBosonAnnihilate(0)) => {
                c10 += coeff;
                hop_terms.push((*coeff, ops.clone()));
            }
            _ => {}
        }
    }
    // The B² cross term −:A₀A₁: carries exactly one hopping per direction
    // (coefficient −1 each — the beamsplitter generator at J = −1).
    assert!(
        (c01.re + 1.0).abs() < 1e-9 && (c10.re + 1.0).abs() < 1e-9,
        "gauge-fixed B² must carry −(a†₀a₁ + a†₁a₀), got c01 = {c01}, c10 = {c10}"
    );
    let h_hop = Hamiltonian { terms: hop_terms };

    // SIRK–Hashimoto on the hopping sector: P₁₁(θ) = cos²θ for the rotation
    // θ = 2t (|J| = 1), the HOM dip at θ = π/2 with the bunched outputs
    // sharing the remaining weight, norm conserved. (One solve + time_evolve
    // — the exact dynamics of the sector, resolved in a single Krylov
    // window.)
    let psi11 = fock(1, 1);
    let res =
        solve_forward_sirk_with_opts(&h_hop, &psi11, &shifts(10), &best_device(), None, &opts())
            .unwrap();
    for &theta in &[
        0.0_f64,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_2,
        2.0 * std::f64::consts::FRAC_PI_3,
        3.0 * std::f64::consts::FRAC_PI_4,
        std::f64::consts::PI,
    ] {
        let psi_t = res.reconstruct(&res.time_evolve(theta / 2.0));
        let p11 = population(&psi_t, &psi11);
        let expected = theta.cos().powi(2);
        assert!(
            (p11 - expected).abs() < 1e-8,
            "θ = {theta:.4}: P₁₁ = {p11:.8}, cos²(θ) = {expected:.8}"
        );
        let norm = QuantumState::inner_product(&psi_t, &psi_t).re;
        assert!((norm - 1.0).abs() < 1e-8, "θ = {theta:.4}: norm = {norm}");
        eprintln!(
            "qed_hom_gauge_fixed_hopping: θ = {theta:.4}, P₁₁ = {p11:.8} (cos²θ = {expected:.8})"
        );
    }
    // The dip: P₂₀ = P₀₂ = ½ (bunching into the |2,0⟩/|0,2⟩ pair).
    let psi_dip = res.reconstruct(&res.time_evolve(std::f64::consts::FRAC_PI_4));
    let p20 = population(&psi_dip, &fock(2, 0));
    let p02 = population(&psi_dip, &fock(0, 2));
    assert!(
        (p20 - 0.5).abs() < 1e-8 && (p02 - 0.5).abs() < 1e-8,
        "bunched outputs must share the weight: P₂₀ = {p20}, P₀₂ = {p02}"
    );
    eprintln!(
        "qed_hom_gauge_fixed_hopping: dip P₁₁ = 0, P₂₀ = {p20:.8}, P₀₂ = {p02:.8}"
    );
}
