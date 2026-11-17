//! Emit the N8 finite-map defect bridge record.
//!
//! The record reuses `sm_defect::finite_fixture`; it does not restate the
//! matrices. Run:
//!
//! ```text
//! cargo run -q -p fock_sirk --example sm_defect_bridge
//! ```
use fock_sirk::sm_defect::finite_fixture_bridge;

fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&finite_fixture_bridge(0.2)).unwrap()
    );
}
