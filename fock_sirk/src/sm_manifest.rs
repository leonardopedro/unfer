//! Machine-reviewable result manifests for the complete-SM numeric cells.
//!
//! N6 of `docs/SM_RESEARCH_PLAN.md` separates what a run actually established
//! from what it merely inherited. This module only *binds and checks* the
//! provenance a caller supplies: source/dirty hashes, operator/state/basis
//! identities, solver settings, environment and artifact digest. It does not
//! build an operator, certify a bound or promote a finite-window measurement
//! to a physical-SM claim.
//!
//! The canonical hash is `sha256` over `serde_json`'s deterministic object
//! order (the workspace enables no `preserve_order` feature, so maps are
//! sorted by key). A manifest therefore hashes its exact emitted fields; any
//! changed number, label, hash or status changes the digest.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// Version marker for the emitted manifest shape.
pub const MANIFEST_SCHEMA: &str = "sm-e1-manifest/v1";

/// The N6 labels that must stay distinct. A result may not silently move
/// between these words; `verify_manifest` enforces the certificate pairing.
pub const STATUSES: [&str; 6] = [
    "historical-pass",
    "source-present",
    "measured",
    "conditionally-bounded",
    "certified",
    "proposed",
];

/// Endpoints this campaign distinguishes. A cell must name exactly one.
pub const ENDPOINTS: [&str; 3] = ["E1-unreduced", "E1-physical", "E2-field-endpoint"];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ManifestError {
    #[error("manifest is missing `{0}`")]
    Missing(&'static str),
    #[error("unsupported manifest schema `{0}` (expected `{MANIFEST_SCHEMA}`)")]
    Schema(String),
    #[error("unknown result status `{0}`")]
    Status(String),
    #[error("unknown endpoint `{0}`")]
    Endpoint(String),
    #[error("status `{0}` and claims.certified={1} disagree; only `certified` may certify")]
    CertificatePromotion(String, bool),
    #[error("`{0}` must be a 64-character lowercase sha256 hex digest")]
    DigestShape(&'static str),
    #[error("`{0}` must be a positive integer")]
    Positive(&'static str),
    #[error("artifact digest mismatch: bound {bound}, recomputed {actual}")]
    ArtifactDigest { bound: String, actual: String },
}

/// Lowercase hex sha256 of arbitrary bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Digest of an ordered label sequence. Each label is length-prefixed so
/// `["ab","c"]` and `["a","bc"]` cannot alias. Callers must pass the labels in
/// their canonical order (a `BTreeSet` iteration is canonical).
pub fn hash_labels(labels: impl IntoIterator<Item = String>) -> String {
    let mut hasher = Sha256::new();
    for label in labels {
        hasher.update((label.len() as u64).to_le_bytes());
        hasher.update(label.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Canonical JSON serialization. `serde_json` maps are sorted because the
/// workspace does not enable `preserve_order`.
pub fn canonical_json(value: &Value) -> String {
    serde_json::to_string(value).expect("manifest values are finite JSON")
}

/// Sha256 of the canonical serialization of a JSON value.
pub fn canonical_json_sha256(value: &Value) -> String {
    sha256_hex(canonical_json(value).as_bytes())
}

/// Insert `artifact_sha256` bound to every other field of `payload`.
///
/// If the payload already carries the field it is replaced, so this function
/// is idempotent. `verify_manifest` removes the field and recomputes.
pub fn bind_artifact(mut payload: Value) -> Value {
    let map = payload
        .as_object_mut()
        .expect("manifest payload must be a JSON object");
    map.remove("artifact_sha256");
    let digest = canonical_json_sha256(&Value::Object(map.clone()));
    map.insert("artifact_sha256".into(), Value::String(digest));
    payload
}

/// Recompute the artifact digest with `artifact_sha256` removed.
pub fn artifact_sha256(payload: &Value) -> String {
    let mut map = payload
        .as_object()
        .expect("manifest payload must be a JSON object")
        .clone();
    map.remove("artifact_sha256");
    canonical_json_sha256(&Value::Object(map))
}

fn object<'a>(
    root: &'a Map<String, Value>,
    key: &'static str,
) -> Result<&'a Map<String, Value>, ManifestError> {
    root.get(key)
        .and_then(Value::as_object)
        .ok_or(ManifestError::Missing(key))
}

fn string<'a>(root: &'a Map<String, Value>, key: &'static str) -> Result<&'a str, ManifestError> {
    root.get(key)
        .and_then(Value::as_str)
        .ok_or(ManifestError::Missing(key))
}

fn digest(root: &Map<String, Value>, key: &'static str) -> Result<(), ManifestError> {
    let value = string(root, key)?;
    if value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(ManifestError::DigestShape(key))
    }
}

fn positive(root: &Map<String, Value>, key: &'static str) -> Result<(), ManifestError> {
    match root.get(key).and_then(Value::as_u64) {
        Some(n) if n > 0 => Ok(()),
        _ => Err(ManifestError::Positive(key)),
    }
}

/// Verify a manifest's shape and provenance binding.
///
/// This is a *format and binding* check, not a physical or numerical
/// certificate. It rejects unknown statuses/endpoints, a status that promotes
/// itself to `certified` without the pairing, malformed digests, non-positive
/// dimensions/ranks and a stale artifact hash.
pub fn verify_manifest(payload: &Value) -> Result<(), ManifestError> {
    let root = payload
        .as_object()
        .ok_or(ManifestError::Missing("manifest object"))?;

    let schema = string(root, "schema")?;
    if schema != MANIFEST_SCHEMA {
        return Err(ManifestError::Schema(schema.to_string()));
    }

    let status = string(root, "status")?;
    if !STATUSES.contains(&status) {
        return Err(ManifestError::Status(status.to_string()));
    }
    let endpoint = string(root, "endpoint")?;
    if !ENDPOINTS.contains(&endpoint) {
        return Err(ManifestError::Endpoint(endpoint.to_string()));
    }

    let source = object(root, "source")?;
    if string(source, "git_revision")?.is_empty() {
        return Err(ManifestError::Missing("source.git_revision"));
    }
    string(source, "dirty_manifest_sha256")?;

    let operator = object(root, "operator")?;
    digest(operator, "identity_sha256")?;
    digest(operator, "fixture_sha256")?;

    let state = object(root, "state")?;
    digest(state, "identity_sha256")?;
    string(state, "sector")?;

    let basis = object(root, "basis")?;
    digest(basis, "label_sha256")?;
    positive(basis, "dimension")?;

    let solver = object(root, "solver")?;
    string(solver, "method")?;
    positive(solver, "rank")?;
    positive(solver, "gmres_max_iterations")?;
    let shift_count = solver
        .get("shift_count")
        .and_then(Value::as_u64)
        .ok_or(ManifestError::Missing("solver.shift_count"))?;
    if shift_count
        != solver
            .get("shifts")
            .and_then(Value::as_array)
            .map_or(0, Vec::len) as u64
    {
        return Err(ManifestError::Missing("solver.shifts length"));
    }

    object(root, "environment")?;

    let claims = object(root, "claims")?;
    let certified = claims
        .get("certified")
        .and_then(Value::as_bool)
        .ok_or(ManifestError::Missing("claims.certified"))?;
    if certified != (status == "certified") {
        return Err(ManifestError::CertificatePromotion(
            status.to_string(),
            certified,
        ));
    }

    let bound = string(root, "artifact_sha256")?;
    let actual = artifact_sha256(payload);
    if bound != actual {
        return Err(ManifestError::ArtifactDigest {
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

    fn fixture() -> Value {
        json!({
            "schema": MANIFEST_SCHEMA,
            "status": "measured",
            "endpoint": "E1-unreduced",
            "source": {"git_revision": "deadbeef", "dirty_manifest_sha256": "clean"},
            "operator": {"identity_sha256": "0".repeat(64), "fixture_sha256": "1".repeat(64)},
            "state": {"identity_sha256": "2".repeat(64), "sector": "graded outer"},
            "basis": {"label_sha256": "3".repeat(64), "dimension": 70},
            "solver": {
                "method": "shift-invert-gmres",
                "rank": 4,
                "gmres_max_iterations": 48,
                "shift_count": 2,
                "shifts": [[0.0, 2.0], [0.0, 1.0]],
            },
            "environment": {"backend": "cpu"},
            "claims": {"certified": false},
        })
    }

    #[test]
    fn label_hash_is_order_and_boundary_sensitive() {
        let a = hash_labels(["x".into(), "yz".into()]);
        let b = hash_labels(["yz".into(), "x".into()]);
        let c = hash_labels(["xy".into(), "z".into()]);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, hash_labels(["x".into(), "yz".into()]));
    }

    #[test]
    fn artifact_binding_detects_every_mutation() {
        let manifest = bind_artifact(fixture());
        verify_manifest(&manifest).expect("fixture verifies");

        let mut changed = manifest.clone();
        changed["solver"]["rank"] = json!(5);
        assert!(matches!(
            verify_manifest(&changed),
            Err(ManifestError::ArtifactDigest { .. })
        ));

        let mut restamped = changed;
        restamped = bind_artifact(restamped);
        verify_manifest(&restamped).expect("re-binding is idempotent");
    }

    #[test]
    fn certificate_promotion_and_shape_are_rejected() {
        let mut manifest = bind_artifact(fixture());
        manifest["status"] = json!("certified");
        manifest = bind_artifact(manifest);
        assert_eq!(
            verify_manifest(&manifest),
            Err(ManifestError::CertificatePromotion(
                "certified".into(),
                false
            ))
        );

        let mut bad_dimension = bind_artifact(fixture());
        bad_dimension["basis"]["dimension"] = json!(0);
        bad_dimension = bind_artifact(bad_dimension);
        assert_eq!(
            verify_manifest(&bad_dimension),
            Err(ManifestError::Positive("dimension"))
        );

        let mut bad_status = bind_artifact(fixture());
        bad_status["status"] = json!("probably-green");
        bad_status = bind_artifact(bad_status);
        assert_eq!(
            verify_manifest(&bad_status),
            Err(ManifestError::Status("probably-green".into()))
        );
    }
}
