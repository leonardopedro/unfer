use nested_fock_algebra::compile_expression;
use quantrs2_symengine_pure::quantum::operators::{commutator, position_operator, momentum_operator};
use quantrs2_symengine_pure::{Expression, parser::parse};

fn main() {
    // 1. Define symbolic base variables for our physical space.
    // We use the crate's standard definitions but then instantiate them 
    // for our specific coordinate c_0/a_0.
    
    // Position x and Momentum p builders from the library
    let x_lib = position_operator(); // returns (a + a_dag) / sqrt(2)
    let p_lib = momentum_operator(); // returns I * (a_dag - a) / sqrt(2)

    // Bridge the library's symbols ("a", "a_dag") to our project's names ("a_0", "c_0")
    // Note: a_dag is our creation operator c_0
    let x_def = x_lib
        .substitute(&Expression::symbol("a"), &Expression::symbol("a_0"))
        .substitute(&Expression::symbol("a_dag"), &Expression::symbol("c_0"));
    
    let p_def = p_lib
        .substitute(&Expression::symbol("a"), &Expression::symbol("a_0"))
        .substitute(&Expression::symbol("a_dag"), &Expression::symbol("c_0"));

    // 2. High-level builders from the library
    // Instead of parsing "x*p - p*x", use the library's commutator()
    let x = Expression::symbol("x");
    let p = Expression::symbol("p");
    let comm_formula = commutator(&x, &p); 

    let comm_sub = comm_formula
        .substitute(&x, &x_def)
        .substitute(&p, &p_def);
    
    let comm_ham = compile_expression(comm_sub);

    println!("Commutator [x, p] via library function:");
    println!("  Formula: {}", comm_formula);
    println!("  Compiled terms: {}", comm_ham.terms.len());
    if !comm_ham.terms.is_empty() {
        // [x, p] should be i * 1.0 (coefficient Complex { re: 0.0, im: 1.0 })
        println!("  Example term: {:?}", comm_ham.terms[0]);
    }

    // 3. Define the full Hamiltonian H = (p^2/2 + x^2/2 + g*x) * n_f
    let coupling = 0.1;
    let h_formula = parse(&format!(
        "C_f0 * (p^2 / 2 + 0.5 * x^2 + {} * x) * A_f0",
        coupling
    )).unwrap();

    let h_substituted = h_formula
        .substitute(&x, &x_def)
        .substitute(&p, &p_def);

    println!("\nFull Hamiltonian: (p^2/2 + 0.5*x^2 + {}*x) * n_f", coupling);
    let hamiltonian = compile_expression(h_substituted);

    println!("Compiled Hamiltonian has {} terms.", hamiltonian.terms.len());
    for (i, (coeff, ops)) in hamiltonian.terms.iter().enumerate().take(5) {
        println!("  Term {}: coeff={:?}, {:#?}", i, coeff, ops);
    }
}
