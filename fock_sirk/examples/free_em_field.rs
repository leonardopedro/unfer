use fock_sirk::solve_forward_sirk;
use nested_fock_algebra::field_theory::{conjugate_momentum, hermitian_field};
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState, compile_expression};
use num_complex::Complex64;
use quantrs2_symengine_pure::Expression;

/// Generates the Hamiltonian for a 0D Free Electromagnetic Field
/// H = 1/2 \pi^2 + 1/2 B^2
/// In 0D, B corresponds to a linear combination of A (vector potential) fields.
/// We'll map A -> hermitian_field and \pi -> conjugate_momentum.
fn free_em_hamiltonian() -> Expression {
    let mut h = Expression::zero();

    // For 3 spatial components (i = 0, 1, 2)
    for i in 0..3 {
        let pi_i = conjugate_momentum(i);
        let a_i = hermitian_field(i); // Represents B field in 0D mapped abstraction

        let kinetic = Expression::from(0.5) * pi_i.clone() * pi_i;
        let magnetic = Expression::from(0.5) * a_i.clone() * a_i;

        h = h + kinetic + magnetic;
    }
    h
}

fn main() -> anyhow::Result<()> {
    println!("--- Free Electromagnetic Field Validation ---");

    let h_expr = free_em_hamiltonian();
    println!("Symbolic Hamiltonian: {}", h_expr);

    // Compile using the standard CAS engine
    let hamiltonian = compile_expression(h_expr);

    // Initial state: 1 photon in polarization mode 0
    let mut inner_b = InnerBosonicState::vacuum();
    inner_b.modes.insert(0, 1);
    let v_0 = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner_b));

    let m_dim = 6; // Small Krylov dimension since it's a quadratic Hamiltonian
    let shifts: Vec<Complex64> = (0..m_dim)
        .map(|j| Complex64::new(0.0, 0.2 * (j as f64)))
        .collect();

    let device = fock_sirk::best_device();

    let sirk_result = solve_forward_sirk(&hamiltonian, &v_0, &shifts, &device, None)
        .expect("Failed to solve SIRK");

    println!(
        "Krylov subspace built. Reduced matrix size: {}x{}",
        sirk_result.h_proj.nrows(),
        sirk_result.h_proj.ncols()
    );

    // Evaluate the unitary time-evolution: e^{-i H t}
    let t = 1.0;
    let coeffs = sirk_result.time_evolve(t);
    println!("Evolution coefficients at t={}: {:?}", t, coeffs);

    // Because the EM field is quadratic and exact, the norm should be strictly preserved.
    let norm_sq: f64 = coeffs.iter().map(|c| c.norm_sqr()).sum();
    println!("Total probability (should be exactly 1.0): {:.6}", norm_sq);

    Ok(())
}
