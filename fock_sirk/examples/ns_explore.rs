use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, evolve_restarted, solve_forward_sirk_with_opts};
use nested_fock_algebra::models::{navier_stokes_brst, navier_stokes_hamiltonian};
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}
fn ns_state(mode: u32, count: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(mode, count);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}
fn ns_two(v: u32, vc: u32, d: u32, dc: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(v, vc);
    inner.modes.insert(d, dc);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

fn main() -> anyhow::Result<()> {
    let h = navier_stokes_hamiltonian(1e-3);
    let brst = navier_stokes_brst();
    let opts = SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: true,
        unit_norm_steps: false,
    };

    let mut psi = ns_two(0, 1, 3, 1);
    psi.scale_and_add(&ns_state(1, 1), Complex64::new(0.5, 0.0));
    let n0 = psi.norm();
    let e0 = QuantumState::inner_product(&h.apply(&psi), &psi).re;

    let t0 = std::time::Instant::now();
    let psi_t = evolve_restarted(&h, &psi, 0.2, 3, 4, &best_device(), None, &opts)?;
    let nt = psi_t.norm();
    let et = QuantumState::inner_product(&h.apply(&psi_t), &psi_t).re;
    println!(
        "EVOL unproj: n0={:.6} nt={:.6} dnorm={:.3e} e0={:.6} et={:.6} dE={:.3e} time={:.1}s",
        n0,
        nt,
        (n0 - nt).abs(),
        e0,
        et,
        (e0 - et).abs(),
        t0.elapsed().as_secs_f64()
    );

    let t0 = std::time::Instant::now();
    let psi_b = evolve_restarted(&h, &psi, 0.2, 3, 4, &best_device(), Some(&brst), &opts)?;
    let bnorm = brst.apply(&psi_b).norm();
    let bnorm_t = brst.apply(&psi_t).norm();
    println!(
        "BRST: ||Omega|psi_proj>||={:.3e}  ||Omega|psi_unproj>||={:.3e}  time={:.1}s",
        bnorm,
        bnorm_t,
        t0.elapsed().as_secs_f64()
    );

    let cnt3 = |s: &QuantumState| -> f64 {
        let as_ = s.apply(&Operator::InnerBosonAnnihilate(3));
        QuantumState::inner_product(&as_, &as_).re
    };
    println!(
        "mode-3 occ: t=0 {:.4}  unproj {:.4}  proj {:.4}",
        cnt3(&psi),
        cnt3(&psi_t),
        cnt3(&psi_b)
    );

    Ok(())
}
