use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{solve_forward_sirk_with_opts, SirkOpts};
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState, oscillator_displaced};
use num_complex::Complex64;

#[test]
fn probe_profile() {
    let (omega, g) = (1.7_f64, 0.45_f64);
    let h = oscillator_displaced(omega, g);
    let v0 = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
    let e0 = -g * g / omega;
    let mk = |un: bool| SirkOpts { prune_eps: 1e-14, max_components: Some(200_000), brst_tol: 1e-10, adaptive: false, unit_norm_steps: un };
    println!("PROFILE-START");
    for m in [4usize, 6, 8, 10, 12, 14] {
        let sh: Vec<Complex64> = shifts_for_range((0, m));
        let ec = solve_forward_sirk_with_opts(&h, &v0, &sh, &best_device(), None, &mk(false)).unwrap().ritz_values()[0];
        let eu = solve_forward_sirk_with_opts(&h, &v0, &sh, &best_device(), None, &mk(true)).unwrap().ritz_values()[0];
        println!("ROW {} {:.3e} {:.3e}", m, (ec - e0).abs(), (eu - e0).abs());
    }
    println!("PROFILE-END");
}
