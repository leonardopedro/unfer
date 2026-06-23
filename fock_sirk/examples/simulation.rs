use fock_sirk::solve_forward_sirk;
use nested_fock_algebra::{
    compile_expression, inner_boson_annihilate, inner_boson_create, Expression, InnerBosonicState,
    Operator, QuantumState,
};
use candle_core::Device;
use num_complex::Complex64;

fn main() -> anyhow::Result<()> {
    // 1. Define Physics: Hopping Hamiltonian
    // H = sum_i (a_{i+1}^dag a_i + a_i^dag a_{i+1})
    let mut h_expr = Expression::zero();
    for i in 0..4 {
        h_expr = h_expr + inner_boson_create(i+1) * inner_boson_annihilate(i) 
                        + inner_boson_create(i) * inner_boson_annihilate(i+1);
    }
    
    println!("Hamiltonian Expression: {}", h_expr);
    let hamiltonian = compile_expression(h_expr);
    
    // 2. Define Initial State: One boson in mode 0
    let mut inner_b = InnerBosonicState::vacuum();
    inner_b.modes.insert(0, 1);
    
    let initial_state = QuantumState::vacuum()
        .apply(&Operator::OuterBosonCreate(inner_b));

    println!("Initial State components: {}", initial_state.components.len());

    // 3. Define Shifts for Rational Krylov
    // We use a small number of shifts for this test
    let m_dim = 2;
    let shifts: Vec<Complex64> = (0..m_dim)
        .map(|j| Complex64::new(0.1 * (j as f64), 0.5))
        .collect();

    // 4. Solve using Inverse-Free Forward SIRK
    let device = fock_sirk::best_device();
    println!("Using device: {:?}", device);

    let sirk_result = solve_forward_sirk(
        &hamiltonian,
        &initial_state,
        &shifts,
        &device,
        None
    ).expect("Failed to solve SIRK");


    println!("Krylov subspace built. Reduced matrix size: {}x{}", 
        sirk_result.h_proj.nrows(), sirk_result.h_proj.ncols());

    // 5. Time Evolution
    let t = 1.0;
    let coefficients = sirk_result.time_evolve(t);
    println!("Evolution coefficients: {:?}", coefficients);

    Ok(())
}
