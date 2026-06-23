use fock_sirk::solve_forward_sirk;
use nested_fock_algebra::{
    compile_expression, Expression, InnerBosonicState, Operator, QuantumState,
    symengine::quantum::operators::{position_operator, momentum_operator},
};
use candle_core::Device;
use num_complex::Complex64;

fn main() -> anyhow::Result<()> {
    // 1. Define Physics: Anharmonic Oscillator
    let x_lib = position_operator();
    let p_lib = momentum_operator();

    // Map library internal symbols "a"/"a_dag" to our "a_0"/"c_0"
    let x = x_lib
        .substitute(&Expression::symbol("a"), &Expression::symbol("a_0"))
        .substitute(&Expression::symbol("a_dag"), &Expression::symbol("c_0"));
    let p = p_lib
        .substitute(&Expression::symbol("a"), &Expression::symbol("a_0"))
        .substitute(&Expression::symbol("a_dag"), &Expression::symbol("c_0"));

    let lambda = 0.1;
    let h_expr = (p.clone().pow(&Expression::int(2)) / Expression::from(2.0)) 
               + (x.clone().pow(&Expression::int(2)) / Expression::from(2.0)) 
               + (Expression::from(lambda) * x.pow(&Expression::int(4)));

    println!("Anharmonic Oscillator Hamiltonian (Symbolic): {}", h_expr);
    let hamiltonian = compile_expression(h_expr);
    
    // 2. Define Initial State
    let initial_state = QuantumState::vacuum()
        .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

    // 3. Define Shifts
    let m_dim = 6;
    let shifts: Vec<Complex64> = (0..m_dim)
        .map(|j| Complex64::new(0.0, 1.0 + (j as f64) * 0.2))
        .collect();

    // 4. Solve
    let device = fock_sirk::best_device();
    let sirk_result = solve_forward_sirk(
        &hamiltonian, 
        &initial_state, 
        &shifts,
        &device,
        None
    ).expect("Failed to solve SIRK");


    println!("Krylov subspace built. Matrix size: {}x{}", 
        sirk_result.h_proj.nrows(), sirk_result.h_proj.ncols());
    
    // 5. Time Evolution
    let t = 0.5;
    let coefficients = sirk_result.time_evolve(t);
    println!("Evolution coefficients: {:?}", coefficients);

    Ok(())
}
