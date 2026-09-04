//! Photon statistics of the driven QED gauge-fixed sector — SIRK-resolved.
//!
//! The QED sector of the abelian Yang–Mills gauge-fixed Hamiltonian from
//! `docs/yang_mills_hamiltonian.cdb` (`H_final = ½π² + ½B²`, U(1) reduction)
//! plus the static-charge `A·J` coupling is the exactly-solvable displaced
//! oscillator `H = ωN + g(B† + B)` ([`qed_static_charge_interaction`]): its
//! ground state is the coherent state `|−g/ω⟩` with **Poissonian** photon
//! statistics — `⟨N⟩ = Var(N) = (g/ω)²`, Fano factor 1. Adding the cavity-Kerr
//! nonlinearity χ (the single-photon-source model, [`qed_kerr_cavity_driven`])
//! detunes the second photon by `2χ`, so the same drive produces
//! **sub-Poissonian** (antibunched) light — Fano factor < 1, `g⁽²⁾(0) < 1` —
//! the photon blockade. All statistics below are extracted from the SIRK–
//! Hashimoto ground eigenvector (the lowest Ritz pair of the projected
//! Hamiltonian, reconstructed to a [`QuantumState`]):
//!
//! 1. `qed_static_charge_ground_is_coherent_poissonian` — the driven QED
//!    gauge-fixed vacuum is exactly a coherent state: Fano factor 1 and
//!    `⟨N⟩ = Var(N) = (g/ω)²` to solver precision, across couplings.
//! 2. `qed_kerr_blockade_sub_poissonian_antibunched` — at χ > 0 the same
//!    drive produces sub-Poissonian statistics (Fano < 1, `g⁽²⁾(0) < 1`),
//!    increasingly so as χ grows (stronger blockade).
//! 3. `qed_kerr_statistics_return_to_poissonian_as_chi_vanishes` — the
//!    Fano factor returns monotonically to 1 as χ → 0 (continuity to the
//!    abelian gauge-fixed sector).

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, QuantumState, qed_kerr_cavity_driven,
    qed_static_charge_interaction,
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
        // The unit-norm frame (exact basis reparametrization, see
        // ritz_edge_study p2/p2b): the raw frame's Gram wall caps usable m
        // once the Krylov vectors span a wide spectral window.
        unit_norm_steps: true,
    }
}

/// The SIRK ground state: the lowest Ritz pair of the projected Hamiltonian,
/// reconstructed into a [`QuantumState`].
///
/// `nalgebra::symmetric_eigen` does NOT sort its eigenpairs (the solver's own
/// residual code re-sorts them), so the lowest eigenvalue is located
/// explicitly before the eigenvector is taken.
fn sirk_ground_state(h: &Hamiltonian, v0: &QuantumState, m: usize) -> QuantumState {
    let res = solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve must complete");
    let eig = res.h_proj.clone().symmetric_eigen();
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let ground_coeffs = eig.eigenvectors.column(order[0]).into_owned();
    res.reconstruct(&ground_coeffs)
}

/// The vacuum of the outer-ladder (static-charge) representation: one empty
/// universe.
fn outer_vacuum() -> QuantumState {
    QuantumState::vacuum()
}

/// The vacuum of the inner-ladder (driven-Kerr) representation.
fn inner_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// Outer number operator `N = B†B` on the mode with inner occupation `{0:1}`
/// (the ladder of `qed_static_charge_interaction`).
fn outer_number() -> Hamiltonian {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(0, 1);
    Hamiltonian {
        terms: vec![(
            Complex64::new(1.0, 0.0),
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::OuterBosonAnnihilate(inner),
            ],
        )],
    }
}

/// Inner number operator `N = a†a` on mode 0 (the driven-Kerr ladder).
fn inner_number() -> Hamiltonian {
    Hamiltonian {
        terms: vec![(
            Complex64::new(1.0, 0.0),
            vec![
                Operator::InnerBosonCreate(0),
                Operator::InnerBosonAnnihilate(0),
            ],
        )],
    }
}

/// `(⟨N⟩, Var(N), Fano = Var/⟨N⟩, g⁽²⁾(0))` of `psi` under the number operator.
fn statistics(psi: &QuantumState, n_op: &Hamiltonian) -> (f64, f64, f64, f64) {
    let n1 = QuantumState::inner_product(psi, &n_op.apply(psi)).re;
    let n2 = QuantumState::inner_product(psi, &n_op.apply(&n_op.apply(psi))).re;
    let var = n2 - n1 * n1;
    let fano = var / n1;
    // Second-order coherence g⁽²⁾(0) = ⟨N(N−1)⟩/⟨N⟩² (antibunching: < 1).
    let g2 = (n2 - n1) / (n1 * n1);
    (n1, var, fano, g2)
}

#[test]
fn qed_static_charge_ground_is_coherent_poissonian() {
    // The Cadabra-derived abelian QED sector with a static charge
    // (`docs/yang_mills_hamiltonian.cdb`, H_final = ½π² + ½B² + A·J): the
    // single-mode model H = ωN + g(B†+B) has the exact coherent ground state
    // |−g/ω⟩ with ⟨N⟩ = Var(N) = (g/ω)² and Fano factor exactly 1.
    let two_pi_sq = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
    for &(k, dk, r, e) in &[
        (1.0, 0.01, 1.0, 0.3),
        (0.5, 0.05, 2.0, 0.9),
        (1.5, 0.1, 3.0, 1.2),
    ] {
        let h = qed_static_charge_interaction(&[(k, dk)], r, e);
        let kr = k * r;
        let g = (e * e * dk * k * (1.0 + kr.sin() / kr) / two_pi_sq).sqrt();
        let n_exact = (g / k).powi(2);

        let psi0 = sirk_ground_state(&h, &outer_vacuum(), 12);
        let (n_mean, var, fano, _) = statistics(&psi0, &outer_number());

        assert!(
            (n_mean - n_exact).abs() < 1e-4,
            "(k={k}, e={e}): ⟨N⟩ = {n_mean}, coherent value (g/ω)² = {n_exact}"
        );
        assert!(
            (var - n_exact).abs() < 1e-4,
            "(k={k}, e={e}): Var(N) = {var}, must equal ⟨N⟩ = {n_exact}"
        );
        assert!(
            (fano - 1.0).abs() < 1e-4,
            "(k={k}, e={e}): Fano factor = {fano}, must be 1 (Poissonian coherent state)"
        );
        eprintln!(
            "qed_static_charge_coherent: k={k}, e={e}: ⟨N⟩ = {n_mean:.6}, \
             Var = {var:.6}, Fano = {fano:.8} (exact 1)"
        );
    }
}

#[test]
fn qed_kerr_blockade_sub_poissonian_antibunched() {
    // The same QED sector with the Kerr nonlinearity: at χ > 0 the drive at
    // the first transition cannot absorb a second photon (Δ₂−Δ₁ = 2χ), so the
    // ground state is a |0⟩+|1⟩-dominated superposition — sub-Poissonian and
    // antibunched (Fano < 1, g⁽²⁾(0) < 1), the photon-blockade single-photon
    // source. Stronger χ = stronger blockade = smaller Fano factor.
    let (omega, g) = (1.0_f64, 0.5_f64);
    let n_op = inner_number();
    let mut fano_prev = 1.0;
    for &chi in &[0.05_f64, 0.2, 0.5, 1.0] {
        let h = qed_kerr_cavity_driven(omega, chi, g);
        let psi0 = sirk_ground_state(&h, &inner_vacuum(), 12);
        let (n_mean, _var, fano, g2) = statistics(&psi0, &n_op);
        assert!(
            n_mean > 0.05,
            "χ={chi}: ⟨N⟩ = {n_mean} must be nonzero for the Fano factor to be meaningful"
        );
        assert!(
            fano < 1.0 - 1e-3,
            "χ={chi}: Fano = {fano}, must be < 1 (sub-Poissonian blockade statistics)"
        );
        assert!(
            g2 < 1.0 - 1e-3,
            "χ={chi}: g⁽²⁾(0) = {g2}, must be < 1 (antibunched single-photon source)"
        );
        assert!(
            fano < fano_prev + 1e-6,
            "χ={chi}: Fano = {fano} must decrease as the blockade strengthens (prev {fano_prev})"
        );
        fano_prev = fano;
        eprintln!(
            "qed_kerr_blockade_stats: χ = {chi}: ⟨N⟩ = {n_mean:.6}, Fano = {fano:.6}, \
             g⁽²⁾(0) = {g2:.6}"
        );
    }
}

#[test]
fn qed_kerr_statistics_return_to_poissonian_as_chi_vanishes() {
    // Continuity to the abelian gauge-fixed sector: as χ → 0 the driven Kerr
    // ground state approaches the coherent state, so Fano → 1 monotonically.
    let (omega, g) = (1.0_f64, 0.5_f64);
    let n_op = inner_number();
    let fano_0 = {
        let h = qed_kerr_cavity_driven(omega, 1e-6, g);
        let psi0 = sirk_ground_state(&h, &inner_vacuum(), 12);
        let (_, _, f, _) = statistics(&psi0, &n_op);
        f
    };
    assert!(
        (fano_0 - 1.0).abs() < 1e-3,
        "χ → 0 must recover the Poissonian sector: Fano = {fano_0}"
    );
    let mut fano_prev = fano_0;
    for &chi in &[0.005_f64, 0.02, 0.05] {
        let h = qed_kerr_cavity_driven(omega, chi, g);
        let psi0 = sirk_ground_state(&h, &inner_vacuum(), 12);
        let (_, _, fano, _) = statistics(&psi0, &n_op);
        // The loop walks χ UPWARD (0.005 → 0.05), so the blockade strengthens
        // and Fano must strictly decrease toward sub-Poissonian values.
        assert!(
            fano < fano_prev + 1e-6,
            "χ = {chi}: Fano = {fano} must be below Fano(smaller χ) = {fano_prev} \
             (monotone departure from Poissonian as χ grows)"
        );
        assert!(
            (fano - 1.0).abs() < 0.2,
            "χ = {chi}: Fano = {fano} must stay near 1 (nearly coherent)"
        );
        fano_prev = fano;
    }
    eprintln!("qed_kerr_continuity: Fano(χ→0) = {fano_0:.6} → Poissonian 1");
}
