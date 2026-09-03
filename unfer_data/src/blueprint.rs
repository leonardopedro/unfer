//! Content-plane store for `.cell` blueprints (S5, F4).
//!
//! A `.cell` archive is ordinary bytes from the data plane's point of view, so it is stored
//! exactly like any other content: chunked (SHA-256 per chunk), content-addressed (SHA-256 of
//! the chunk CIDs), addressed by magnet URI, and encrypted at rest with X25519+AES-GCM via the
//! same primitives `DataPublisher` uses. This module is the thin blueprint-specific wrapper
//! over the existing content plane; the archive format itself lives in
//! `unfer_protocol::archive` (the shared contract).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use unfer_protocol::ChunkRef;

use crate::chunk::{Chunker, verify_chunk};
use crate::crypto::{DataKeypair, decrypt_chunk, derive_aes_key, encrypt_chunk};
use crate::magnet::{build_magnet_uri, content_cid_from_chunks};

/// A stored `.cell` archive's content-plane addressing metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CellRef {
    /// Content CID (SHA-256 over the concatenated chunk CIDs).
    pub cid: String,
    /// Magnet URI addressing the content by CID.
    pub magnet_uri: String,
    pub filesize: u64,
    pub chunk_count: u32,
    pub chunk_refs: Vec<ChunkRef>,
}

/// "Store" a `.cell` archive: chunk it and derive its content-plane addresses.
pub fn store_cell(cell: &[u8]) -> CellRef {
    let chunker = Chunker::default();
    let chunk_refs = chunker.chunk_refs(cell);
    let cids: Vec<String> = chunk_refs.iter().map(|c| c.cid.clone()).collect();
    let cid = content_cid_from_chunks(&cids);
    CellRef {
        cid: cid.clone(),
        magnet_uri: build_magnet_uri(&cid, None),
        filesize: cell.len() as u64,
        chunk_count: chunk_refs.len() as u32,
        chunk_refs,
    }
}

/// Verify that `cell` hashes to `cid` (content-address integrity check).
pub fn verify_cell(cell: &[u8], cid: &str) -> bool {
    verify_chunk(cell, cid) || {
        let stored = store_cell(cell);
        stored.cid == cid
    }
}

/// Whether `cid` has the content-plane shape (64 lowercase hex chars).
///
/// This is a *structural* gate for address-like inputs (e.g. an edge `/cell/<cid>` path);
/// it does not prove the content exists anywhere.
pub fn is_content_cid(cid: &str) -> bool {
    cid.len() == 64 && cid.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Encrypt a stored cell for transport, mirroring `DataPublisher`: each chunk is AES-256-GCM
/// encrypted under a key derived from the content keypair's X25519 handshake (self-DH, the
/// crate's per-content symmetric-key convention). Returns the per-chunk ciphertexts; the
/// recipient uses the same keypair convention via [`decrypt_stored_cell`].
pub fn encrypt_stored_cell(keypair: &DataKeypair, cell: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let aes_key = derive_aes_key(&keypair.shared_secret(keypair.public_key()));
    let mut ciphertexts = Vec::new();
    for (idx, plain) in Chunker::default().chunk(cell) {
        ciphertexts.push(encrypt_chunk(&aes_key, idx, &plain)?);
    }
    Ok(ciphertexts)
}

/// Reassemble + decrypt a stored cell (the inverse of [`encrypt_stored_cell`]).
pub fn decrypt_stored_cell(
    keypair: &DataKeypair,
    ciphertexts: &[Vec<u8>],
) -> Result<Vec<u8>, String> {
    let aes_key = derive_aes_key(&keypair.shared_secret(keypair.public_key()));
    let mut out = Vec::new();
    for (i, ct) in ciphertexts.iter().enumerate() {
        out.extend_from_slice(&decrypt_chunk(&aes_key, i as u32, ct)?);
    }
    Ok(out)
}

// ── Blueprint registry (S20, F19) ─────────────────────────────────────
//
// cloudflare-os blueprints are immutable, shareable app templates: every consumer runs
// *its own* copy. The registry records immutable, content-addressed cells (blueprint_id
// == the content CID, so re-editing produces a different blueprint rather than mutating
// one) and retains the raw cell bytes so a fresh per-user session can be instantiated
// from them later.

/// An immutable, registered blueprint template (S20, F19).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlueprintRecord {
    /// Content CID of the `.cell` bytes — the blueprint's immutable identity.
    pub blueprint_id: String,
    /// Human-readable name from the archive metadata.
    pub name: String,
    /// Content CID of the stored cell body (`== blueprint_id`; explicit so the
    /// content plane has a single addressable artifact).
    pub cell_cid: String,
    /// The archive metadata (`name`/`version`/`archetype`/`entry`) as JSON.
    pub manifest_json: String,
    /// The first blueprint of this content lineage; never re-editable. A fresh import
    /// sets it equal to `blueprint_id`; a record can never change it (no in-place edits).
    pub immutable_blueprint_id: String,
    /// The principal that minted the record at the kernel chokepoint (audit tag, F6).
    pub created_by: String,
}

/// Registry of immutable blueprint templates.
#[derive(Debug, Default)]
pub struct BlueprintRegistry {
    records: HashMap<String, BlueprintRecord>,
    cells: HashMap<String, Vec<u8>>,
}

/// The process-global registry (one kernel per process), the same placement the
/// action/audit/resource registries use in `unfer_ffi::handles`.
pub fn global_registry() -> &'static Mutex<BlueprintRegistry> {
    static REGISTRY: OnceLock<Mutex<BlueprintRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BlueprintRegistry::default()))
}

impl BlueprintRegistry {
    /// Register a verified `.cell` archive. `created_by` is the minting principal.
    ///
    /// Re-importing the same bytes (same CID) is idempotent and returns the *existing*
    /// record unchanged (blueprints are immutable — nothing is re-editable). Any other
    /// cell yields a new immutable record.
    pub fn register(&mut self, cell: &[u8], created_by: &str) -> Result<BlueprintRecord, String> {
        let record = self.record_for(cell, created_by)?;
        self.commit_record(&record, cell);
        Ok(record)
    }

    /// Compute the record that registering `cell` would produce, WITHOUT mutating.
    /// Mirrors `register`'s idempotency: identical bytes yield the existing record
    /// (first mint preserved), any other cell yields a fresh record. The FFI uses
    /// this so a buffer probe or too-small buffer can learn the record and its size
    /// without committing a registration ("consume only on a complete copy").
    pub fn record_for(&self, cell: &[u8], created_by: &str) -> Result<BlueprintRecord, String> {
        let stored = store_cell(cell);
        let cid = stored.cid.clone();
        if let Some(existing) = self.records.get(&cid) {
            // Content-addressed idempotency: identical bytes are the same immutable
            // blueprint; keep the first mint (created_by is preserved).
            return Ok(existing.clone());
        }
        let parsed = unfer_protocol::Cell::parse(cell).map_err(|e| e.to_string())?;
        let metadata = parsed.metadata();
        Ok(BlueprintRecord {
            blueprint_id: cid.clone(),
            name: metadata.name.clone(),
            cell_cid: cid.clone(),
            manifest_json: serde_json::to_string(metadata).map_err(|e| e.to_string())?,
            immutable_blueprint_id: cid.clone(),
            created_by: created_by.to_string(),
        })
    }

    /// Commit a record previously computed by [`record_for`]: make the cell
    /// addressable and the record visible. Idempotent — re-committing the same
    /// record is a no-op (the first mint is preserved).
    pub fn commit_record(&mut self, record: &BlueprintRecord, cell: &[u8]) {
        let cid = record.blueprint_id.clone();
        self.records
            .entry(cid.clone())
            .or_insert_with(|| record.clone());
        self.cells.entry(cid).or_insert_with(|| cell.to_vec());
    }

    pub fn get(&self, blueprint_id: &str) -> Option<&BlueprintRecord> {
        self.records.get(blueprint_id)
    }

    pub fn cell_bytes(&self, blueprint_id: &str) -> Option<&[u8]> {
        self.cells.get(blueprint_id).map(|b| b.as_slice())
    }

    /// `None` when the caller lacks the id; the content itself is address-public.
    pub fn is_registered(&self, blueprint_id: &str) -> bool {
        self.records.contains_key(blueprint_id)
    }

    pub fn list(&self) -> Vec<BlueprintRecord> {
        let mut records: Vec<_> = self.records.values().cloned().collect();
        records.sort_by(|a, b| a.blueprint_id.cmp(&b.blueprint_id));
        records
    }

    /// Drop every registered blueprint (console/QA use; re-imports re-mint).
    pub fn clear(&mut self) {
        self.records.clear();
        self.cells.clear();
    }
}

/// Process-global registry maintenance (QA/console): empty the registry.
pub fn clear_global_registry() {
    global_registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::compute_cid;
    use unfer_protocol::CellBuilder;

    fn sample_cell() -> Vec<u8> {
        let mut b = CellBuilder::new("demo");
        b.set_archetype("ecmascript");
        b.set_entry("src/main.js");
        b.add_file("module.toml", b"[module]\nname = \"demo\"\n")
            .unwrap();
        b.add_file("src/main.js", b"export async function run(k,a){return a;}")
            .unwrap();
        b.set_session(br#"{"t_now":0.25}"#);
        b.build().unwrap()
    }

    #[test]
    fn store_cell_chunks_and_addresses() {
        let cell = sample_cell();
        let stored = store_cell(&cell);

        assert_eq!(stored.filesize, cell.len() as u64);
        assert_eq!(stored.chunk_refs.len() as u64, stored.chunk_count as u64);
        assert!(stored.magnet_uri.starts_with("magnet:?xt=urn:btih:"));
        assert_eq!(stored.cid.len(), 64);

        // Each chunk's CID verifies against the chunk bytes.
        let chunker = Chunker::default();
        let chunks = chunker.chunk(&cell);
        for (i, (_, plain)) in chunks.iter().enumerate() {
            assert_eq!(stored.chunk_refs[i].cid, compute_cid(plain));
        }
    }

    #[test]
    fn verify_cell_matches_content_cid() {
        let cell = sample_cell();
        let stored = store_cell(&cell);
        assert!(verify_cell(&cell, &stored.cid));
        assert!(!verify_cell(&cell, &format!("{}x", &stored.cid[..63])));
    }

    #[test]
    fn content_cid_shape_gate() {
        let stored = store_cell(&sample_cell());
        assert!(is_content_cid(&stored.cid));
        assert!(!is_content_cid("not-a-cid"));
        assert!(!is_content_cid(&stored.cid[..63])); // 63 chars
        assert!(!is_content_cid(&format!("{}G", &stored.cid[..63]))); // non-hex char
    }

    #[test]
    fn cell_content_id_is_deterministic() {
        let cell = sample_cell();
        assert_eq!(store_cell(&cell).cid, store_cell(&cell).cid);
    }

    #[test]
    fn blueprint_record_register_get_list_idempotent_import() {
        // S20 (F19): a verified cell registers an immutable record addressed by its
        // content CID; re-importing identical bytes is idempotent (never re-edited).
        let mut registry = BlueprintRegistry::default();
        let cell = sample_cell();
        let stored = store_cell(&cell);
        let cid = stored.cid.clone();

        let rec = registry.register(&cell, "publisher").expect("register");
        assert_eq!(rec.blueprint_id, cid);
        assert_eq!(rec.cell_cid, cid);
        assert_eq!(
            rec.immutable_blueprint_id, cid,
            "fresh blueprint is its own lineage root"
        );
        assert_eq!(rec.name, "demo");
        assert_eq!(rec.created_by, "publisher");
        assert!(rec.manifest_json.contains("\"archetype\":\"ecmascript\""));

        // Idempotent re-import of identical bytes returns the same record and keeps the
        // original minter — the immutable_blueprint_id never moves.
        let rec2 = registry.register(&cell, "intruder").expect("register");
        assert_eq!(rec2.blueprint_id, cid);
        assert_eq!(
            rec2.created_by, "publisher",
            "an idempotent import cannot re-mint"
        );

        assert_eq!(registry.list().len(), 1);
        assert!(registry.is_registered(&cid));
        assert_eq!(
            registry.cell_bytes(&cid).map(|b| b.to_vec()).as_deref(),
            Some(&cell[..])
        );
        assert_eq!(registry.get(&cid).map(|r| r.name.as_str()), Some("demo"));
        assert!(!registry.is_registered(&"0".repeat(64)));
    }

    #[test]
    fn blueprint_immutable_itness_content_addressed() {
        // Saving a *different* body yields a different blueprint id (no in-place edit).
        let cell = sample_cell();
        let mut registry = BlueprintRegistry::default();
        let id1 = registry.register(&cell, "alice").expect("register");
        let id2 = registry
            .register(&cell1(b"edited"), "alice")
            .expect("register");
        assert_ne!(id1.blueprint_id, id2.blueprint_id);
        assert_eq!(id1.immutable_blueprint_id, id1.blueprint_id);
        assert_eq!(id2.immutable_blueprint_id, id2.blueprint_id);
    }

    fn cell1(bytes: &[u8]) -> Vec<u8> {
        let mut b = CellBuilder::new("demo");
        b.add_file("module.toml", bytes).unwrap();
        b.build().unwrap()
    }

    #[test]
    fn register_rejects_corrupt_cell() {
        let mut registry = BlueprintRegistry::default();
        assert!(registry.register(b"not-a-cell", "alice").is_err());
    }

    #[test]
    fn cell_encrypt_decrypt_roundtrip_through_content_plane() {
        let cell = sample_cell();
        let recipient = DataKeypair::generate();

        let ciphertexts = encrypt_stored_cell(&recipient, &cell).unwrap();
        assert_ne!(
            ciphertexts.iter().flatten().cloned().collect::<Vec<u8>>(),
            cell
        );

        let plain = decrypt_stored_cell(&recipient, &ciphertexts).unwrap();
        assert_eq!(plain, cell);

        // The reassembled plaintext is still addressable by the same CID.
        let restored = store_cell(&plain);
        assert_eq!(restored.cid, store_cell(&cell).cid);
    }
}

#[cfg(test)]
mod content_proptests {
    // S17 (F16): arbitrary cell contents must round-trip through the content plane.
    use super::{is_content_cid, store_cell, verify_cell};
    use proptest::prelude::*;

    proptest! {
        fn store_and_verify_roundtrip(cell in proptest::collection::vec(any::<u8>(), 0..4000)) {
            let stored = store_cell(&cell);
            let cid = stored.cid.clone();
            // Data → CID → verify holds for arbitrary payloads.
            prop_assert!(verify_cell(&cell, &cid));
            prop_assert!(is_content_cid(&cid));
            // Deterministic content addressing.
            prop_assert_eq!(store_cell(&cell).cid, cid);
            prop_assert_eq!(stored.filesize, cell.len() as u64);
        }
    }
}
