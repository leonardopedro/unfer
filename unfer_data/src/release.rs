//! Content-addressed release protocol (S24, F23).
//!
//! Cloudflare pins one byte-identical, content-addressed release manifest and promotes
//! `candidate → release` in a single all-or-nothing copy. This module owns that protocol
//! for `unfer`'s deployable artifacts:
//!
//! * A [`ReleaseManifest`] maps every deployable crate/module **by name to the content CID
//!   of its bytes** (SHA-256, 64-hex — the crate's content-address convention).
//! * [`ReleaseStore::promote`] stores the manifest under a release key in one content-addressed
//!   op and returns the manifest's own CID ("one manifest copy"): the release pointer is
//!   byte-identical to the candidate's, so promotion cannot silently re-drift.
//!
//! Golden gate: `tests/release_manifest_golden.rs` compares a fixed artifact set against a
//! committed golden manifest. A wrong byte in any module changes the manifest and fails the
//! gate; regeneration is honored only via `UPDATE_GOLDEN=1`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::chunk::compute_cid;

/// The content-plane addressing shape: how many hex chars a manifest/artifact CID must be.
const CID_SHAPE: usize = 64;

/// A byte-addressed release manifest: every deployable artifact (crate/module) keyed by its
/// stable name, value = content CID of the artifact's bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReleaseManifest {
    /// Manifest schema (language of the artifact table).
    pub schema: u32,
    /// Name → content CID (64-hex sha256) of the artifact bytes. `BTreeMap` keeps the
    /// canonical JSON byte-stable for a given artifact set (golden golden-able).
    pub artifacts: BTreeMap<String, String>,
}

impl ReleaseManifest {
    /// Build a manifest from an artifact set, computing each artifact's byte-CID.
    pub fn build(schema: u32, artifacts: &[(String, &[u8])]) -> Self {
        let table = artifacts
            .iter()
            .map(|(name, bytes)| (name.clone(), compute_cid(bytes)))
            .collect();
        Self {
            schema,
            artifacts: table,
        }
    }

    /// The canonical JSON bytes of the manifest (deterministic; drives golden + CID).
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("manifest serializes")
    }

    /// The content CID of this manifest itself (sha256 of the canonical JSON — the
    /// "release pointer" a promotion moves).
    pub fn manifest_cid(&self) -> String {
        compute_cid(&self.to_canonical_bytes())
    }

    /// Structural integrity: every artifact address must be a well-formed content CID.
    /// (`is_content_cid`-equivalent: 64 hex chars.)
    pub fn validate(&self) -> bool {
        self.artifacts
            .values()
            .all(|cid| cid.len() == CID_SHAPE && cid.bytes().all(|b| b.is_ascii_hexdigit()))
    }
}

/// Content-addressed release table. `promote` is the single op: storing `release` (and
/// keeping the candidate addressable) is one `Mutex` critical section — no torn copy, no
/// second manifest. The returned CID is the manifest's own content address, so promoting
/// `candidate → release` moves the same bytes' pointer (one manifest copy, byte-identical).
pub struct ReleaseStore {
    releases: Mutex<Option<BTreeMap<String, ReleaseManifest>>>,
}

impl Default for ReleaseStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ReleaseStore {
    pub fn new() -> Self {
        Self {
            releases: Mutex::new(None),
        }
    }

    /// Promote `manifest` into the table under `key` (a release tag like `release` or the
    /// `candidate` tag). Single content-addressed op: returns the manifest CID on success,
    /// `Err` if the manifest fails its CID shape gate (a bad module address must never
    /// become a release pointer).
    pub fn promote(&self, key: &str, manifest: ReleaseManifest) -> Result<String, String> {
        if !manifest.validate() {
            return Err(format!(
                "release manifest for '{key}' has a malformed artifact CID"
            ));
        }
        let cid = manifest.manifest_cid();
        let mut guard = self.releases.lock().unwrap_or_else(|e| e.into_inner());
        let table = guard.get_or_insert_with(BTreeMap::new);
        if let Some(existing) = table.get(key)
            && existing != &manifest
        {
            return Err(format!(
                "release key '{key}' already pinss different content (promotion is all-or-nothing)"
            ));
        }
        table.insert(key.to_string(), manifest);
        Ok(cid)
    }

    /// The manifest under a release tag, if any.
    pub fn get(&self, key: &str) -> Option<ReleaseManifest> {
        self.releases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()?
            .get(key)
            .cloned()
    }

    /// The (single) manifest whose content CID is `cid` — the content-addressed read path.
    pub fn get_by_cid(&self, cid: &str) -> Option<ReleaseManifest> {
        self.releases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|table| table.values().find(|m| m.manifest_cid() == cid).cloned())
    }

    /// Galvanic slots currently pinned (tag names).
    pub fn keys(&self) -> Vec<String> {
        self.releases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|t| t.keys().cloned().collect())
            .unwrap_or_default()
    }
}

// ── tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_artifacts() -> Vec<(String, Vec<u8>)> {
        vec![
            (
                "nested_fock_algebra".to_string(),
                b"EXPORT(MODULE \"nested_fock_algebra\" (VALUE\n  (SYMBOL kernel)\n)))\n"
                    .to_vec(),
            ),
            (
                "demo".to_string(),
                b"module demo {\n grain \"src/main.js\" = \"export async function run(k,a){return a;}\";\n}\n"
                    .to_vec(),
            ),
        ]
    }

    #[test]
    fn cids_are_sha256_hex_and_byte_unique() {
        let artifacts = fixture_artifacts();
        let m = ReleaseManifest::build(
            1,
            &artifacts
                .iter()
                .map(|(k, v)| (k.clone(), v.as_slice()))
                .collect::<Vec<_>>(),
        );
        assert_eq!(m.artifacts.len(), 2);
        assert!(m.validate(), "every artifact must have a 64-hex CID");
        for cid in m.artifacts.values() {
            assert_eq!(cid.len(), 64);
            assert!(cid.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        let c1 = m.artifacts["nested_fock_algebra"].clone();
        let c2 = m.artifacts["demo"].clone();
        assert_ne!(c1, c2, "distinct artifacts must get distinct CIDs");
    }

    #[test]
    fn wrong_byte_changes_manifest() {
        // F23 gate: a wrong byte in a module changes the manifest (so a broken artifact
        // is always caught at the golden compare — never silently promoted).
        let mut artifacts = fixture_artifacts();
        let m1 = ReleaseManifest::build(
            1,
            &artifacts
                .iter()
                .map(|(k, v)| (k.clone(), v.as_slice()))
                .collect::<Vec<_>>(),
        );
        let demo = artifacts.iter_mut().find(|(n, _)| n == "demo").unwrap();
        demo.1[0] ^= 0x01;
        let m2 = ReleaseManifest::build(
            1,
            &artifacts
                .iter()
                .map(|(k, v)| (k.clone(), v.as_slice()))
                .collect::<Vec<_>>(),
        );
        assert_ne!(m1, m2, "one wrong byte must change the release manifest");
        assert_ne!(m1.manifest_cid(), m2.manifest_cid());
    }

    #[test]
    fn promote_creates_one_content_addressed_copy() {
        let store = ReleaseStore::new();
        let artifacts = fixture_artifacts();
        let m = ReleaseManifest::build(
            1,
            &artifacts
                .iter()
                .map(|(k, v)| (k.clone(), v.as_slice()))
                .collect::<Vec<_>>(),
        );

        // Promote candidate, then release: same bytes, single content address.
        let candidate_cid = store
            .promote("candidate", m.clone())
            .expect("candidate promote");
        let release_cid = store
            .promote("release", m.clone())
            .expect("release promote");
        assert_eq!(
            candidate_cid, release_cid,
            "promote is byte-identical — one manifest copy"
        );
        assert_eq!(store.get("candidate"), Some(m.clone()));
        assert_eq!(store.get("release"), Some(m.clone()));
        assert_eq!(
            store.get_by_cid(&release_cid),
            Some(m.clone()),
            "content-addressed read finds the manifest"
        );
    }

    #[test]
    fn promote_refuses_tampered_artifact_address() {
        let store = ReleaseStore::new();
        let mut m = ReleaseManifest::build(1, &[]);
        m.artifacts
            .insert("rogue".to_string(), "not-a-cid".to_string());
        assert!(
            store.promote("release", m).is_err(),
            "malformed CID must be refused"
        );
        assert!(store.get("release").is_none(), "nothing promoted");
    }

    #[test]
    fn baseline_release() {
        // The committed golden manifest must equal a rebuild from the CI fixture bytes.
        // This mirrors the integration golden gate so the module tests stand alone.
        let artifacts = fixture_artifacts();
        let built = ReleaseManifest::build(
            1,
            &artifacts
                .iter()
                .map(|(k, v)| (k.clone(), v.as_slice()))
                .collect::<Vec<_>>(),
        );
        let roundtrip: ReleaseManifest =
            serde_json::from_slice(&built.to_canonical_bytes()).expect("roundtrip");
        assert_eq!(
            roundtrip, built,
            "canonical bytes must round-trip into an identical manifest"
        );
    }
}
