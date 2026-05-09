use fock_sirk::solve_forward_sirk;
use nested_fock_algebra::QuantumState;
use nested_fock_algebra::models::yang_mills_hamiltonian;
use candle_core::Device;
use num_complex::Complex64;

fn main() -> anyhow::Result<()> {
    println!("--- Pure SU(3) Yang-Mills Field Theory ---");

    let g = 1.0;
    println!("Building non-Abelian Yang-Mills Hamiltonian (direct term construction)...");
    let hamiltonian = yang_mills_hamiltonian(g);
    println!("Hamiltonian has {} operator terms.", hamiltonian.terms.len());

    // Initial state: Vacuum (no gluons)
    let mut v_0 = QuantumState::vacuum();
    v_0 = v_0.apply(&nested_fock_algebra::Operator::OuterBosonCreate(nested_fock_algebra::InnerBosonicState::vacuum()));


    let m_dim = 4;
    let shifts: Vec<Complex64> = (0..m_dim)
        .map(|j| Complex64::new(0.5, 0.5 * (j as f64)))
        .collect();

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);

    println!("Building Krylov Subspace for gluon vacuum fluctuations...");
    let sirk_result = solve_forward_sirk(
        &hamiltonian,
        &v_0,
        &shifts,
        &device,
        None
    ).expect("Failed to solve SIRK");

    println!("Subspace built. G-matrix shape: {}x{}",
        sirk_result.g_matrix.nrows(), sirk_result.g_matrix.ncols());

    let t = 0.1;
    let coeffs = sirk_result.time_evolve(t);
    println!("Gluon vacuum fluctuation coefficients at t={}: {:?}", t, coeffs);

    // Export JSON for visualization
    let json = sirk_result.export_to_json();
    println!("JSON export size: {} bytes", json.len());

    Ok(())
}
