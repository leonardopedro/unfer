//! Heavy Krylov cases split out of the unit-test suite (IMPROVEMENT_PLAN #28).
//!
//! These Yang-Mills lattice solves are the wall-time drivers of `fock_sirk`
//! (adaptive l=4 ~11s, adaptive l=5 ~41s on commodity CPU). They stay part of
//! `cargo test --workspace` (integration tests) and CI, but are now separate
//! from the fast `--lib` unit suite and filterable with
//! `cargo test -p fock_sirk --test heavy_krylov`.
//!
//! They exercise the bounded direct-construction path (`SirkOpts::adaptive`)
//! that keeps the quartic plaquette term under a fixed component budget, and
//! the l=3 mass-gap demonstration (the central empirical P10.18 deliverable).

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
use nalgebra::DMatrix;
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn assert_hermitian(h_proj: &DMatrix<Complex64>, label: &str) {
    let dag = h_proj.adjoint();
    let diff = (h_proj - &dag).norm();
    assert!(
        diff < 1e-8,
        "{label}: H_proj must be Hermitian, ‖H−H†‖={diff}"
    );
}

fn vacuum_seed() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// l=4 with adaptive mode: the quartic plaquette term previously caused
/// `StateExplosion` at 627K components (max=10K, m=8). With adaptive
/// truncation at max=50K, the solver completes and produces a Hermitian
/// H_proj with positive rank.
#[test]
fn adaptive_l4_completes_under_budget() {
    use nested_fock_algebra::models::yang_mills_lattice;

    let device = best_device();
    let h = yang_mills_lattice(4, 1.0, 1);
    assert!(h.terms.len() > 250, "l=4 lattice should have >250 terms");

    let v0 = vacuum_seed();

    let opts = SirkOpts {
        prune_eps: 1e-10,
        max_components: Some(50_000),
        brst_tol: 1e-10,
        adaptive: true,
    };
    let res = solve_forward_sirk_with_opts(&h, &v0, &shifts(4), &device, None, &opts)
        .expect("adaptive l=4 solve must complete under budget");

    assert!(res.rank > 0, "adaptive l=4 must produce positive rank");
    assert_hermitian(&res.h_proj, "adaptive l=4");

    // Time-evolve and verify the norm is reasonable (truncation introduces
    // some error, but the Gram whitening absorbs non-orthonormality).
    let coeffs = res.time_evolve(0.01);
    let psi_t = res.reconstruct(&coeffs);
    let norm = QuantumState::norm(&psi_t);
    assert!(
        norm > 0.5 && norm < 2.0,
        "adaptive l=4 norm={norm:.4} should be O(1) (truncation may shift it)"
    );
}

/// l=5 with adaptive mode: 450 terms, 25 plaquettes. Without adaptive
/// mode this would explode immediately. With adaptive truncation, the
/// solver completes under a fixed 50K-component budget.
#[test]
fn adaptive_l5_completes_under_budget() {
    use nested_fock_algebra::models::yang_mills_lattice;

    let device = best_device();
    let h = yang_mills_lattice(5, 1.0, 1);
    assert!(
        h.terms.len() > 400,
        "l=5 lattice should have >400 terms, got {}",
        h.terms.len()
    );

    let v0 = vacuum_seed();

    let opts = SirkOpts {
        prune_eps: 1e-8,
        max_components: Some(50_000),
        brst_tol: 1e-10,
        adaptive: true,
    };
    let res = solve_forward_sirk_with_opts(&h, &v0, &shifts(4), &device, None, &opts)
        .expect("adaptive l=5 solve must complete under budget");

    assert!(res.rank > 0, "adaptive l=5 must produce positive rank");
    assert_hermitian(&res.h_proj, "adaptive l=5");
}

/// Yang-Mills mass-gap demonstration on an l=3 lattice.
///
/// Extends the l=2 test (`yang_mills_lattice_mass_gap`) to a 3×3 periodic
/// lattice: 9 plaquettes, 18 link modes (2 dirs × 9 sites × 1 color), 162
/// Hamiltonian terms (18 electric + 9 × 16 magnetic sub-terms from the
/// quartic Φ⁴ expansion). This is the central empirical deliverable of
/// P10.18: a positive mass gap at l=3 in the strong-coupling regime (g=2)
/// demonstrates the same confinement mechanism the Millennium Prize problem
/// asks to prove in the continuum limit.
#[test]
fn yang_mills_l3_mass_gap_demo() {
    use nested_fock_algebra::models::yang_mills_lattice;

    let device = best_device();
    let g = 2.0;
    let g2_half = g * g / 2.0; // expected electric gap ≈ 2.0
    let h = yang_mills_lattice(3, g, 1); // 18 link modes, 9 plaquettes

    // Even sector: vacuum seed.
    let v_even = vacuum_seed();

    // Odd sector: single excitation on link mode 0 (dir=0, site(0,0), color=0).
    let mut inner_odd = InnerBosonicState::vacuum();
    inner_odd.modes.insert(0, 1);
    let v_odd = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner_odd));

    let opts = SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(200_000), // generous limit for l=3
        brst_tol: 1e-10,
        adaptive: false,
    };
    let m = 4;
    let res_even = solve_forward_sirk_with_opts(&h, &v_even, &shifts(m), &device, None, &opts)
        .expect("l=3 even-parity SIRK must complete (StateExplosion would indicate scaling wall)");
    let res_odd = solve_forward_sirk_with_opts(&h, &v_odd, &shifts(m), &device, None, &opts)
        .expect("l=3 odd-parity SIRK must complete");

    let e_even = res_even.ground_state_energy().unwrap();
    let e_odd = res_odd.ground_state_energy().unwrap();

    eprintln!(
        "yang_mills_lattice(3, g={g}, 1): \
         rank_even={}, rank_odd={}, \
         E_even={e_even:.6}, E_odd={e_odd:.6}",
        res_even.rank, res_odd.rank,
    );

    let gap = fock_sirk::mass_gap_from_sectors(&res_even, &res_odd).unwrap();
    eprintln!("l=3 mass gap = {gap:.6} (expected O(g²/2 = {g2_half}))");

    assert!(
        gap > 0.0,
        "l=3 mass gap must be positive (confinement): e_even={e_even:.4}, e_odd={e_odd:.4}, gap={gap:.4}"
    );
    assert!(
        gap > g2_half / 3.0 && gap < g2_half * 3.0,
        "l=3 mass gap {gap:.4} should be O(g²/2 = {g2_half}): \
         e_even={e_even:.4}, e_odd={e_odd:.4}"
    );
}
