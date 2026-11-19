//! N10 derivative-jet finite-window checks.
//!
//! The jet coordinates are independent E1 fields, not constrained spatial
//! derivatives. The window measurements are Rust-side; the bound Lean theorems
//! are conditional dimension and block-Q² implications, not a continuum result.

use fock_sirk::sm_brst_fock::finite_fock_brst_jet_window;
use serde_json::{json, Value};

const FIXTURE: &str = include_str!("../../docs/fixtures/sm_brst_jet_window.json");

#[test]
fn jet_window_is_closed_block_graded_and_nilpotent() {
    let pinned: Value = serde_json::from_str(FIXTURE).unwrap();
    let regenerated = finite_fock_brst_jet_window();
    assert_eq!(pinned, regenerated, "regenerate with sm_brst_jets");

    assert_eq!(pinned["schema"], "sm-finite-fock-brst-jets/v1");
    assert_eq!(pinned["claim_id"], "n10_finite_fock_brst_window_jets");
    assert_eq!(pinned["status"], "measured");
    assert_eq!(pinned["closed"], true);
    assert_eq!(pinned["graded"], true);
    assert_eq!(pinned["matter_graded"], true);
    assert_eq!(pinned["jet_graded"], true);
    assert_eq!(
        pinned["theorem"]["proof_status"],
        "kernel-verified conditional"
    );
    assert_eq!(
        pinned["theorem"]["payload_sha256"],
        "fcd22f8950e4066787a570dd263c3d921439b203e9098b4aa64211e9048c5fe5"
    );
    assert_eq!(pinned["theorem"]["declarations"], 22097);
    assert_eq!(
        pinned["theorem"]["matrix_q_sq"]["lean_name"],
        "BookProof.SmBrstQuotient.jetWindowCharge_sq"
    );
    assert_eq!(
        pinned["theorem"]["matrix_q_sq"]["proof_status"],
        "kernel-verified conditional"
    );

    let artifact = &pinned["artifact"];
    assert_eq!(artifact["field_level"], true);
    assert_eq!(artifact["finite_window"], true);
    assert_eq!(artifact["physical_continuum_quotient"], false);
    assert_eq!(artifact["state_count"], 36_207);
    assert_eq!(artifact["occupation_dimensions"], json!([1, 268, 35_938]));
    assert_eq!(
        artifact["jet_state_dimensions"],
        json!([11_067, 17_880, 7_260])
    );
    assert_eq!(
        artifact["matter_state_dimensions"],
        json!([15_039, 16_608, 4_560])
    );
    assert_eq!(artifact["block_ranks"], json!([0, 249]));
    assert_eq!(artifact["differential_rank"], 249);
    assert_eq!(artifact["matter_block_ranks"], json!([159, 90, 0]));
    assert_eq!(artifact["jet_block_ranks"], json!([138, 111, 0]));
    assert_eq!(artifact["cohomology_dimensions"], json!([1, 19, 35_689]));
    assert_eq!(artifact["cohomology_dimension"], 35_709);
    assert_eq!(
        artifact["cohomology_by_jet_number"],
        json!([[1, 10, 10_780], [0, 9, 17_649], [0, 0, 7_260]])
    );
    assert_eq!(
        artifact["cohomology_by_matter_number"],
        json!([[1, 13, 14_707], [0, 6, 16_422], [0, 0, 4_560]])
    );
    assert_eq!(artifact["gauge_singlet_kernel_dimension"], 6);
    assert_eq!(artifact["q_squared_residual"], 0.0);
    assert_eq!(
        artifact["harmonic_representatives"]["degree_1"]["dimension"],
        19
    );
    assert_eq!(
        artifact["harmonic_representatives"]["degree_2"]["dimension"],
        35_689
    );
    assert_eq!(
        artifact["harmonic_representatives"]["degree_2"]["pivot_columns"]
            .as_array()
            .unwrap()
            .len(),
        249
    );
    assert_eq!(
        artifact["quotient_map"]["dimensions"],
        json!([1, 19, 35_689])
    );
    assert_eq!(artifact["quotient_map"]["continuum"], false);
}
