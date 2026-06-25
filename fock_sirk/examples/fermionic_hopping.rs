use fock_sirk::solve_forward_sirk;
use nested_fock_algebra::{
    QuantumState, compile_expression, inner_fermion_annihilate, inner_fermion_create,
};
use num_complex::Complex64;

fn main() -> anyhow::Result<()> {
    // 1. Define Physics: Fermionic Hopping Hamiltonian
    // H = sum_i (c_{i+1}^dag c_i + c_i^dag c_{i+1})
    // Fermionic operators anticommute, so order matters.
    let mut h_expr = nested_fock_algebra::Expression::zero();
    for i in 0..4 {
        h_expr = h_expr
            + inner_fermion_create(i + 1) * inner_fermion_annihilate(i)
            + inner_fermion_create(i) * inner_fermion_annihilate(i + 1);
    }

    println!("Fermionic Hamiltonian: {}", h_expr);
    let hamiltonian = compile_expression(h_expr);

    // 2. Define Initial State: One fermion in mode 0
    // We use the new helper method
    let initial_state = QuantumState::vacuum().create_fermion(0);

    println!(
        "Initial State components: {}",
        initial_state.components.len()
    );

    // 3. Define Shifts for Rational Krylov
    let m_dim = 4;
    let shifts: Vec<Complex64> = (0..m_dim)
        .map(|j| Complex64::new(0.1 * (j as f64), 0.5))
        .collect();

    // 4. Solve using Inverse-Free Forward SIRK
    let device = fock_sirk::best_device();
    println!("Using device: {:?}", device);

    let sirk_result = solve_forward_sirk(&hamiltonian, &initial_state, &shifts, &device, None)
        .expect("Failed to solve SIRK");

    println!(
        "Krylov subspace built. Reduced matrix size: {}x{}",
        sirk_result.h_proj.nrows(),
        sirk_result.h_proj.ncols()
    );

    // 5. Time Evolution
    let t = 1.0;
    let coefficients = sirk_result.time_evolve(t);
    println!("Evolution coefficients: {:?}", coefficients);

    Ok(())
}
