//! N10 finite constraint-dictionary bridge checks.
//!
//! The fixture is measured data, not a kernel proof and not a physical quotient.
//! It must keep `quotient_map: null` and stay below the finite-image tolerance.

use fock_sirk::sm_bridge::{BridgeError, verify_bridge};
use fock_sirk::sm_constraints_bridge::finite_constraint_bridge;
use serde_json::{Value, json};

const FIXTURE: &str = include_str!("../../docs/fixtures/sm_constraints_bridge.json");

fn close(value: &Value, tolerance: f64) {
    assert!(
        value.as_f64().unwrap() < tolerance,
        "finite residual {value} exceeds {tolerance:e}"
    );
}

#[test]
fn fixture_binds_the_measured_finite_dictionary() {
    let pinned: Value = serde_json::from_str(FIXTURE).unwrap();
    let regenerated = finite_constraint_bridge();
    assert_eq!(pinned, regenerated, "regenerate with sm_constraints_bridge");
    verify_bridge(&pinned).unwrap();

    assert_eq!(pinned["status"], "measured");
    assert_eq!(pinned["theorem"]["proof_status"], "not-run");
    assert!(pinned["artifact"]["quotient_map"].is_null());
    assert_eq!(pinned["artifact"]["finite_dictionary"], true);

    close(
        &pinned["artifact"]["gauss_commutator_gauge_vacuum"]["worst_residual"],
        2e-9,
    );
    close(
        &pinned["artifact"]["gauss_commutator_gauge_occupied"]["worst_residual"],
        2e-9,
    );
    close(
        &pinned["artifact"]["brst_nilpotency"]["worst_residual"],
        2e-9,
    );
    close(
        &pinned["artifact"]["hamiltonian_gauss_commutator"]["worst_residual"],
        2e-9,
    );
}

#[test]
fn finite_ce_quotient_is_nilpotent_and_nontrivial() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let quotient = &fixture["artifact"]["finite_ce_quotient"];
    assert_eq!(quotient["status"], "measured");
    assert_eq!(quotient["operator"]["generators"], 12);

    let artifact = &quotient["artifact"];
    close(&artifact["q_squared_residual"], 1e-9);
    close(&artifact["representation_closure_residual"], 1e-9);
    assert_eq!(artifact["cochain_dimensions"], json!([12, 144, 792, 2640]));
    assert_eq!(artifact["differential_ranks"], json!([11, 132, 660]));
    assert_eq!(artifact["cohomology_dimensions"], json!([1, 1, 0]));
    assert_eq!(artifact["finite_quotient"], true);
    assert_eq!(artifact["field_level_quotient"], false);
    assert!(fixture["artifact"]["quotient_map"].is_null());

    let representatives = &artifact["harmonic_representatives"];
    assert_eq!(representatives["degree_0"], 1);
    assert_eq!(representatives["degree_1"], 1);
    assert_eq!(representatives["degree_2"], 0);
    assert_eq!(
        representatives["basis_sha256"].as_str().unwrap().len(),
        64,
        "the representative basis is bound by a sha256"
    );
}

#[test]
fn bridge_rejects_a_tampered_finite_quotient() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let mut tampered = fixture.clone();
    tampered["artifact"]["finite_ce_quotient"]["artifact"]["q_squared_residual"] = json!(0.0);
    assert!(matches!(
        verify_bridge(&tampered),
        Err(BridgeError::ArtifactDigest { .. })
    ));
}

#[test]
fn bridge_rejects_a_claimed_physical_quotient() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let mut promoted = fixture.clone();
    promoted["artifact"]["quotient_map"] = json!({"kind": "physical cohomology"});
    assert!(matches!(
        verify_bridge(&promoted),
        Err(BridgeError::ArtifactDigest { .. })
    ));
}
