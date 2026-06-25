use fock_sirk::solve_forward_sirk;
use nested_fock_algebra::models::{navier_stokes_brst, navier_stokes_hamiltonian};
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState};
use num_complex::Complex64;

fn main() -> anyhow::Result<()> {
    // 1. Viscosity parameter
    let nu = 1e-3;

    // 2. Build Hamiltonians directly (bypasses Expression::expand())
    println!("Building Navier-Stokes Hamiltonian...");
    let hamiltonian = navier_stokes_hamiltonian(nu);
    let brst_charge = navier_stokes_brst();

    // 3. Define Initial State:
    // Creating a state that mimics a specific velocity configuration
    let mut inner_b = InnerBosonicState::vacuum();
    inner_b.modes.insert(0, 2); // 2 quanta in u_1
    // mode index = base 3 + i*3 + j (here i = j = 1) → u_{2,2}
    #[allow(clippy::identity_op)]
    inner_b.modes.insert(3 + 1 * 3 + 1, 1); // 1 quantum in u_{2,2}

    let v_0 = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner_b));

    // 4. Verification: Check if initial state satisfies BRST Divergence Free constraint
    // \Omega |v_0> = 0
    let brst_check = brst_charge.apply(&v_0);
    println!(
        "BRST Constraint Norm: {}",
        QuantumState::inner_product(&brst_check, &brst_check)
            .re
            .sqrt()
    );
    // If not zero, you mathematically project it here: v_0 = v_0 - projection(brst_check)

    // 5. Setup SIRK Shifting (using imaginary shifts for dissipative systems)
    let m_dim = 10;
    let shifts: Vec<Complex64> = (0..m_dim)
        .map(|j| Complex64::new(0.0, 0.1 * (j as f64)))
        .collect();

    let device = fock_sirk::best_device();

    // 6. Execute Forward SIRK over the Phase Space
    println!("Solving Navier-Stokes dynamics over probability space...");
    let sirk_result = solve_forward_sirk(&hamiltonian, &v_0, &shifts, &device, Some(&brst_charge))
        .expect("Failed to solve SIRK");

    // 7. Extract the Non-deterministic Time-Evolution
    let t = 0.05;
    let coefficients = sirk_result.time_evolve(t);
    println!(
        "Probability distribution coefficients at t={}: {:?}",
        t, coefficients
    );

    Ok(())
}
