//! Emit the N10 derivative-jet finite BRST window.

fn main() {
    let record = fock_sirk::sm_brst_fock::finite_fock_brst_jet_window();
    println!("{}", serde_json::to_string_pretty(&record).unwrap());
}
