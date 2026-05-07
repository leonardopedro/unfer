use fock_sirk::{solve_forward_sirk};
use nested_fock_algebra::{
    compile_expression, QuantumState
};
use nested_fock_algebra::models::yang_mills_hamiltonian;
use candle_core::Device;
use num_complex::Complex64;

fn main() -> anyhow::Result<()> {
    println!("--- Pure SU(3) Yang-Mills Field Theory ---");
    
    // Coupling constant g
    let g = 1.0; 
    let h_expr = yang_mills_hamiltonian(g);
    
    println!("Compiling non-polynomial Yang-Mills Hamiltonian...");
    let hamiltonian = compile_expression(h_expr);
    
    // Initial state: Vacuum state (no gluons)
    // Because the Hamiltonian includes terms like a^\dagger a^\dagger a^\dagger a^\dagger 
    // (from the B_{ia} B_{ia} non-linear components), the vacuum will spontaneously 
    // generate pairs and quartets of virtual gluons.
    let v_0 = QuantumState::vacuum();

    // Since the state space grows exponentially due to the non-linear terms, 
    // the Rational Krylov Shift-Invert algorithm is critical here.
    let m_dim = 5; 
    let shifts: Vec<Complex64> = (0..m_dim)
        .map(|j| Complex64::new(0.5, 0.5 * (j as f64)))
        .collect();

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    println!("Building Krylov Subspace for Gluon interactions...");
    let sirk_result = solve_forward_sirk(
        &hamiltonian,
        &v_0,
        &shifts,
        &device
    ).expect("Failed to solve SIRK");

    println!("Subspace built. G-matrix shape: {}x{}", 
        sirk_result.g_matrix.nrows(), sirk_result.g_matrix.ncols());

    // Time Evolve
    let t = 0.1;
    let coeffs = sirk_result.time_evolve(t);
    println!("Gluon vacuum fluctuation coefficients at t={}: {:?}", t, coeffs);

    Ok(())
}
