//! Scoped fixture-to-theorem bridge records for the complete-SM campaign.
//!
//! N13 of `docs/SM_RESEARCH_PLAN.md` requires a proof-facing artifact to name
//! the operator, state, theorem and remaining hypotheses it claims to bind.
//! This module checks that shape and the artifact digest; it does **not**
//! prove the theorem, run Lean, or promote a record to `certified`.
//!
//! The only accepted promotion is a record whose `status` is `certified`,
//! whose `theorem.proof_status` is `kernel-verified`, and which carries a
//! `proof` object with a sha256 digest. Even then this module verifies only
//! the record's internal binding, not that nanoda accepted the payload; the
//! caller must run `prob_kernel::verify::verify_export` (or the future Lean
//! specialist's check) before trusting that field.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// Version marker for a proof-bridge fixture.
pub const BRIDGE_SCHEMA: &str = "sm-proof-bridge/v1";

/// Statuses a bridge record may carry. `certified` requires a kernel proof.
pub const BRIDGE_STATUSES: [&str; 6] = [
    "proposed",
    "source-present",
    "historical-pass",
    "measured",
    "conditionally-bounded",
    "certified",
];

/// Proof states a theorem obligation may carry.
pub const PROOF_STATUSES: [&str; 2] = ["not-run", "kernel-verified"];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BridgeError {
    #[error("bridge is missing `{0}`")]
    Missing(&'static str),
    #[error("unsupported bridge schema `{0}` (expected `{BRIDGE_SCHEMA}`)")]
    Schema(String),
    #[error("unknown bridge status `{0}`")]
    Status(String),
    #[error("unknown proof status `{0}`")]
    ProofStatus(String),
    #[error("`{0}` must be a 64-character lowercase sha256 hex digest")]
    DigestShape(&'static str),
    #[error("bridge artifact digest mismatch: bound {bound}, recomputed {actual}")]
    ArtifactDigest { bound: String, actual: String },
    #[error("`{0}` must be a non-empty array")]
    EmptyList(&'static str),
    #[error("status `certified` requires theorem.proof_status `kernel-verified`")]
    CertifiedWithoutProof,
    #[error("theorem.proof_status `kernel-verified` requires a proof object")]
    VerifiedWithoutPayload,
}

/// Lowercase hex sha256 of arbitrary bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Canonical JSON serialization (maps are sorted by key).
pub fn canonical_json(value: &Value) -> String {
    serde_json::to_string(value).expect("bridge values are finite JSON")
}

/// Sha256 of the canonical serialization of a JSON value.
pub fn canonical_json_sha256(value: &Value) -> String {
    sha256_hex(canonical_json(value).as_bytes())
}

/// Report a measured quantity with twelve significant decimal digits.
///
/// Fixture tests pin these records exactly, but a residual evaluated in `f64`
/// carries toolchain noise in its last digits: the same inputs produced
/// `1.4669222755248563e-13` and `1.4669222755248606e-13` under different builds,
/// which failed an exact fixture comparison although the measured content was
/// unchanged. Twelve significant digits sits far below the measurement's own
/// `1e-9` acceptance tolerance and far above that noise, so the record stays a
/// faithful measurement and becomes reproducible across toolchains. The
/// formatting is deterministic, and parsing the 12-digit decimal back gives the
/// nearest `f64` to it.
pub fn reported_residual(value: f64) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    format!("{value:.11e}")
        .parse::<f64>()
        .expect("a formatted finite float parses")
}

/// Insert `artifact_sha256` bound to every other field of `payload`.
pub fn bind_artifact(mut payload: Value) -> Value {
    let map = payload
        .as_object_mut()
        .expect("bridge payload must be a JSON object");
    map.remove("artifact_sha256");
    let digest = canonical_json_sha256(&Value::Object(map.clone()));
    map.insert("artifact_sha256".into(), Value::String(digest));
    payload
}

/// Recompute the artifact digest with `artifact_sha256` removed.
pub fn artifact_sha256(payload: &Value) -> String {
    let mut map = payload
        .as_object()
        .expect("bridge payload must be a JSON object")
        .clone();
    map.remove("artifact_sha256");
    canonical_json_sha256(&Value::Object(map))
}

fn object<'a>(
    root: &'a Map<String, Value>,
    key: &'static str,
) -> Result<&'a Map<String, Value>, BridgeError> {
    root.get(key)
        .and_then(Value::as_object)
        .ok_or(BridgeError::Missing(key))
}

fn string<'a>(root: &'a Map<String, Value>, key: &'static str) -> Result<&'a str, BridgeError> {
    root.get(key)
        .and_then(Value::as_str)
        .ok_or(BridgeError::Missing(key))
}

fn digest(root: &Map<String, Value>, key: &'static str) -> Result<(), BridgeError> {
    let value = string(root, key)?;
    if value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(BridgeError::DigestShape(key))
    }
}

fn nonempty_list(root: &Map<String, Value>, key: &'static str) -> Result<(), BridgeError> {
    match root.get(key).and_then(Value::as_array) {
        Some(values) if !values.is_empty() => Ok(()),
        _ => Err(BridgeError::EmptyList(key)),
    }
}

/// Verify a bridge record's shape, scope fields and artifact digest.
///
/// This is a binding check only. It rejects a `certified` status without a
/// claimed kernel proof, but it cannot independently confirm that proof.
pub fn verify_bridge(payload: &Value) -> Result<(), BridgeError> {
    let root = payload
        .as_object()
        .ok_or(BridgeError::Missing("bridge object"))?;

    let schema = string(root, "schema")?;
    if schema != BRIDGE_SCHEMA {
        return Err(BridgeError::Schema(schema.to_string()));
    }
    let status = string(root, "status")?;
    if !BRIDGE_STATUSES.contains(&status) {
        return Err(BridgeError::Status(status.to_string()));
    }
    if string(root, "claim_id")?.is_empty() {
        return Err(BridgeError::Missing("claim_id"));
    }
    if string(root, "endpoint")?.is_empty() {
        return Err(BridgeError::Missing("endpoint"));
    }

    let operator = object(root, "operator")?;
    if string(operator, "name")?.is_empty() {
        return Err(BridgeError::Missing("operator.name"));
    }
    digest(operator, "identity_sha256")?;

    let state = object(root, "state")?;
    nonempty_list(state, "basis")?;
    digest(state, "identity_sha256")?;

    let theorem = object(root, "theorem")?;
    if string(theorem, "lean_name")?.is_empty() {
        return Err(BridgeError::Missing("theorem.lean_name"));
    }
    if string(theorem, "statement")?.is_empty() {
        return Err(BridgeError::Missing("theorem.statement"));
    }
    let proof_status = string(theorem, "proof_status")?;
    if !PROOF_STATUSES.contains(&proof_status) {
        return Err(BridgeError::ProofStatus(proof_status.to_string()));
    }
    let proof = root.get("proof").and_then(Value::as_object);
    if status == "certified" && proof_status != "kernel-verified" {
        return Err(BridgeError::CertifiedWithoutProof);
    }
    if proof_status == "kernel-verified" {
        let proof = proof.ok_or(BridgeError::VerifiedWithoutPayload)?;
        digest(proof, "payload_sha256")?;
        if string(proof, "checker")?.is_empty() {
            return Err(BridgeError::Missing("proof.checker"));
        }
    } else if status == "certified" {
        return Err(BridgeError::CertifiedWithoutProof);
    }

    nonempty_list(root, "hypotheses")?;
    nonempty_list(root, "not_claimed")?;

    let bound = string(root, "artifact_sha256")?;
    let actual = artifact_sha256(payload);
    if bound != actual {
        return Err(BridgeError::ArtifactDigest {
            bound: bound.to_string(),
            actual,
        });
    }
    digest(root, "artifact_sha256")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record() -> Value {
        bind_artifact(json!({
            "schema": BRIDGE_SCHEMA,
            "claim_id": "one_mode_fermion_car",
            "status": "source-present",
            "endpoint": "E1-unreduced",
            "operator": {"name": "one-mode fermion ladder", "identity_sha256": "0".repeat(64)},
            "state": {"basis": ["vacuum", "occupied"], "identity_sha256": "1".repeat(64)},
            "artifact": {"matrices": {"a": [[0, 1], [0, 0]]}},
            "theorem": {
                "lean_name": "BookProof.Sm.ChapterSmCar.one_mode_anticommutator",
                "statement": "a * adag + adag * a = 1",
                "proof_status": "not-run",
            },
            "hypotheses": ["finite one-mode truncation"],
            "not_claimed": ["no continuum claim"],
        }))
    }

    #[test]
    fn accepts_scoped_unproved_record() {
        verify_bridge(&record()).unwrap();
    }

    #[test]
    fn rejects_mutation_without_rebinding() {
        let mut r = record();
        r["theorem"]["statement"] = json!("0 = 1");
        assert!(matches!(
            verify_bridge(&r),
            Err(BridgeError::ArtifactDigest { .. })
        ));
    }

    #[test]
    fn accepts_conditionally_bounded_status() {
        let mut r = record();
        r["status"] = json!("conditionally-bounded");
        let r = bind_artifact(r);
        verify_bridge(&r).unwrap();
    }

    #[test]
    fn rejects_self_certification() {
        let mut r = record();
        r["status"] = json!("certified");
        let r = bind_artifact(r);
        assert_eq!(verify_bridge(&r), Err(BridgeError::CertifiedWithoutProof));
    }

    #[test]
    fn rejects_verified_status_without_payload() {
        let mut r = record();
        r["theorem"]["proof_status"] = json!("kernel-verified");
        let r = bind_artifact(r);
        assert_eq!(verify_bridge(&r), Err(BridgeError::VerifiedWithoutPayload));
    }
}
