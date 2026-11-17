fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&fock_sirk::sm_brst_fock::finite_fock_brst_window()).unwrap()
    );
}
