//! Emit the N10 finite constraint-dictionary bridge.

fn main() {
    let record = fock_sirk::sm_constraints_bridge::finite_constraint_bridge();
    println!("{}", serde_json::to_string_pretty(&record).unwrap());
}
