//! Study: the fate of "spurious" Ritz values ABOVE the resolved spectral
//! window in the shifted (inverse-free) SIRK projection.
//!
//! Background (`coupled_oscillator_sirk.rs`): the m-shift projection of the
//! displaced oscillator carries Ritz values above the top physical level in
//! the Krylov-reachable set. This module establishes what they ARE.
//!
//! Theory. For the displaced oscillator H = ωN + g(a†+a) the Fock basis is
//! the EXACT eigenbasis (E_n = ωn − g²/ω), so for ANY normalized vector
//! ψ = Σ cₙ|n⟩ the Rayleigh quotient is the exact weighted mean
//!     θ[ψ] = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ = Σ |cₙ|² Eₙ ,
//! i.e. θ lies INSIDE the convex hull of the levels with nonzero weight.
//! The forward sequence w_k = ∏ⱼ(H − zⱼI)v₀ raises occupation by one per
//! step, so m shifts reach n ≤ m and the projected problem's extreme
//! eigenvalues are Rayleigh quotients of vectors mixed over the UNRESOLVED
//! rungs n ≈ K..m. Predictions:
//!   P1 (bracketing): every Ritz value lies in [E₀, E_m(+ε)] — none can
//!      exceed the highest reachable rung;
//!   P2 (convergence): the low end converges to E₀ < E₁ < … as m grows,
//!      monotonically from outside;
//!   P3 (tracking): the topmost Ritz value moves UP the ladder as m grows
//!      (sup_θ strictly increasing, approaching the next rung);
//!   P4 (mixture content): the reconstructed top-Ritz vector has mean
//!      occupation ⟨N⟩ well above the resolved window (mixed high-n),
//!      and its direct Rayleigh quotient reproduces θ (consistency);
//!   P5 (residual): ‖Hψ − θψ‖/‖ψ‖ is LARGE for the top pair (unconverged)
//!      and SMALL for the ground pair — they are approximate eigenvectors of
//!      quality proportional to their convergence, not noise.
//!
//! These tests pin each prediction numerically.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{solve_forward_sirk_with_opts, SirkOpts};
use nalgebra::DVector;
use nested_fock_algebra::{
    InnerBosonicState, Operator, QuantumState, oscillator_displaced,
};
use num_complex::Complex64;

const OMEGA: f64 = 1.7;
const G: f64 = 0.45;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
    }
}

fn exact_level(n: f64) -> f64 {
    OMEGA * n - G * G / OMEGA
}

fn vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn number_op() -> nested_fock_algebra::Hamiltonian {
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

/// Eigen-decompose the projected Hamiltonian; return (values ascending,
/// eigenvector columns matching).
fn eigen_pairs(
    h_proj: &nalgebra::DMatrix<Complex64>,
) -> (Vec<f64>, Vec<DVector<Complex64>>) {
    let eig = h_proj.clone().symmetric_eigen();
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap()
    });
    (
        order.iter().map(|&i| eig.eigenvalues[i]).collect(),
        order.iter().map(|&i| eig.eigenvectors.column(i).into_owned()).collect(),
    )
}

#[test]
fn p1_every_ritz_bracketed_by_the_reachable_ladder() {
    let h = oscillator_displaced(OMEGA, G);
    let m = 8;
    let res =
        solve_forward_sirk_with_opts(&h, &vacuum(), &shifts(m), &best_device(), None, &opts())
            .unwrap();
    // Occupation reachable after m applications: n ≤ m ⇒ θ ∈ [E₀, E_m].
    for &theta in &res.ritz_values() {
        assert!(
            theta >= exact_level(0.0) - 1e-7 && theta <= exact_level(m as f64) + 1e-6,
            "Ritz {theta} outside [{:.4}, {:.4}]",
            exact_level(0.0),
            exact_level(m as f64)
        );
    }
}

#[test]
fn p2_low_end_converges_and_sits_on_a_conditioning_floor() {
    let h = oscillator_displaced(OMEGA, G);
    // FINDING (measured profile): err(m=4)≈2e-7, err(6)≈1e-9 (optimal),
    // err(8)≈1.7e-6, err(10)≈1.7e-3. Beyond the optimum the Gram matrix of
    // the polynomial sequence becomes catastrophically ill-conditioned
    // (‖w_k‖ grows like ‖H‖ᵏ); whitening truncation then injects noise that
    // degrades even the LOW end. This conditioning wall is WHY the
    // architecture evolves long times by RESTARTED short windows
    // (`evolve_restarted`) instead of one deep solve.
    let profile = [(4_usize, 1e-5_f64), (6, 1e-8), (8, 1e-5), (10, 1e-2)];
    for (m, tol) in profile {
        let res =
            solve_forward_sirk_with_opts(&h, &vacuum(), &shifts(m), &best_device(), None, &opts())
                .unwrap();
        let theta0 = res.ritz_values()[0];
        let err = (theta0 - exact_level(0.0)).abs();
        assert!(err < tol, "ground Ritz band violated at m={m}: err={err:.3e} ≥ {tol:.0e}");
        if m == 6 {
            assert!(
                err < 1e-8 * 10.0,
                "m=6 is the optimal window depth for this model"
            );
        }
    }
    // The wall itself: degradation must be present (this documents reality;
    // if a future whitening fix removes it, update this test and the guide).
    let e8 = {
        let r = solve_forward_sirk_with_opts(&h, &vacuum(), &shifts(8), &best_device(), None, &opts()).unwrap();
        (r.ritz_values()[0] - exact_level(0.0)).abs()
    };
    let e10 = {
        let r = solve_forward_sirk_with_opts(&h, &vacuum(), &shifts(10), &best_device(), None, &opts()).unwrap();
        (r.ritz_values()[0] - exact_level(0.0)).abs()
    };
    assert!(e10 > 50.0 * e8.max(1e-15), "conditioning wall expected: err(10)≫err(8)");
}

#[test]
fn p3_topmost_ritz_climbs_toward_higher_rungs() {
    let h = oscillator_displaced(OMEGA, G);
    let mut prev_sup = f64::NEG_INFINITY;
    for m in [4_usize, 6, 8, 10] {
        let res =
            solve_forward_sirk_with_opts(&h, &vacuum(), &shifts(m), &best_device(), None, &opts())
                .unwrap();
        let sup = *res.ritz_values().last().unwrap();
        assert!(sup > prev_sup, "sup(Ritz) must climb with m: m={m} sup={sup}");
        // ...and never overshoots the next-unresolved rung band by much:
        assert!(sup < exact_level(m as f64) + 1e-6);
        prev_sup = sup;
    }
}

#[test]
fn p4_top_ritz_vector_is_a_high_occupation_mixture() {
    let h = oscillator_displaced(OMEGA, G);
    let m = 6;
    let res =
        solve_forward_sirk_with_opts(&h, &vacuum(), &shifts(m), &best_device(), None, &opts())
            .unwrap();
    let (vals, vecs) = eigen_pairs(&res.h_proj);
    let n_op = number_op();

    // Top pair:
    let top_theta = *vals.last().unwrap();
    let top_coeffs = vecs.last().unwrap().clone();
    let psi_top = res.reconstruct(&top_coeffs);
    let norm2 = QuantumState::inner_product(&psi_top, &psi_top).re;
    // Direct Rayleigh quotient of the RECONSTRUCTED vector must reproduce θ
    // (projection consistency: the small-basis eigenpair really represents a
    // big-space vector with that energy mean).
    let rq = QuantumState::inner_product(&psi_top, &h.apply(&psi_top)).re / norm2;
    assert!(
        (rq - top_theta).abs() / top_theta.abs().max(1.0) < 5e-4,
        "Rayleigh quotient {rq} vs Ritz {top_theta}"
    );
    // Mixture content: mean occupation well above the resolved window (n≤2).
    let n_mean = QuantumState::inner_product(&psi_top, &n_op.apply(&psi_top)).re / norm2;
    assert!(n_mean > 2.0, "top vector must be high-n mixed, ⟨N⟩={n_mean:.3}");
    // Mixture content: mean occupation well above the resolved window (n≤2).
    let n_mean = QuantumState::inner_product(&psi_top, &n_op.apply(&psi_top)).re / norm2;
    assert!(n_mean > 2.0, "top vector must be high-n mixed, ⟨N⟩={n_mean:.3}");

    // Ground pair: the EXACT displaced ground state is a coherent state with
    // mean occupation α² = (g/ω)² — the measured 0.0701 IS that physics, not
    // impurity. Assert the machinery reproduces it.
    let alpha_sq = (G / OMEGA).powi(2);
    let bot_coeffs = vecs.first().unwrap().clone();
    let psi_bot = res.reconstruct(&bot_coeffs);
    let nb = QuantumState::inner_product(&psi_bot, &n_op.apply(&psi_bot))
        .re
        / QuantumState::inner_product(&psi_bot, &psi_bot).re;
    assert!(
        (nb - alpha_sq).abs() < 0.02,
        "ground ⟨N⟩={nb:.4} must match the coherent-state value (g/ω)²={alpha_sq:.4}"
    );
}

#[test]
fn p5_residual_separates_converged_from_unconverged() {
    let h = oscillator_displaced(OMEGA, G);
    let m = 6;
    let res =
        solve_forward_sirk_with_opts(&h, &vacuum(), &shifts(m), &best_device(), None, &opts())
            .unwrap();
    let (vals, vecs) = eigen_pairs(&res.h_proj);

    let res_norm = |coeffs: &DVector<Complex64>, theta: f64| -> f64 {
        let psi = res.reconstruct(coeffs);
        let norm = QuantumState::inner_product(&psi, &psi).re.sqrt();
        let hp = h.apply(&psi);
        let mut acc = QuantumState::zero();
        acc.scale_and_add(&hp, Complex64::new(1.0, 0.0));
        acc.scale_and_add(&psi, Complex64::new(-theta, 0.0));
        QuantumState::inner_product(&acc, &acc).re.sqrt() / norm
    };

    let r_ground = res_norm(&vecs[0].clone(), vals[0]);
    let r_top = res_norm(&vecs.last().unwrap().clone(), *vals.last().unwrap());
    assert!(
        r_ground < 1e-3,
        "ground pair must be a genuine approximate eigenvector, r={r_ground:.3e}"
    );
    assert!(
        r_top > 10.0 * r_ground.max(1e-12),
        "top pair must be far less converged: r_top={r_top:.3e}"
    );
}
