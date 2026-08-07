//! Content-plane store for `.cell` blueprints (S5, F4).
//!
//! A `.cell` archive is ordinary bytes from the data plane's point of view, so it is stored
//! exactly like any other content: chunked (SHA-256 per chunk), content-addressed (SHA-256 of
//! the chunk CIDs), addressed by magnet URI, and encrypted at rest with X25519+AES-GCM via the
//! same primitives `DataPublisher` uses. This module is the thin blueprint-specific wrapper
//! over the existing content plane; the archive format itself lives in
//! `unfer_protocol::archive` (the shared contract).

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

/// Encrypt a stored cell for transport, mirroring `DataPublisher`: each chunk is AES-256-GCM
/// encrypted under a key derived from the content keypair's X25519 handshake (self-DH, the
/// crate's per-content symmetric-key convention). Returns the per-chunk ciphertexts; the
/// recipient uses the same keypair convention via [`decrypt_stored_cell`].
pub fn encrypt_stored_cell(
    keypair: &DataKeypair,
    cell: &[u8],
) -> Result<Vec<Vec<u8>>, String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::compute_cid;
    use unfer_protocol::CellBuilder;

    fn sample_cell() -> Vec<u8> {
        let mut b = CellBuilder::new("demo");
        b.set_archetype("ecmascript");
        b.set_entry("src/main.js");
        b.add_file("module.toml", b"[module]\nname = \"demo\"\n").unwrap();
        b.add_file("src/main.js", b"export async function run(k,a){return a;}").unwrap();
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
    fn cell_content_id_is_deterministic() {
        let cell = sample_cell();
        assert_eq!(store_cell(&cell).cid, store_cell(&cell).cid);
    }

    #[test]
    fn cell_encrypt_decrypt_roundtrip_through_content_plane() {
        let cell = sample_cell();
        let recipient = DataKeypair::generate();

        let ciphertexts = encrypt_stored_cell(&recipient, &cell).unwrap();
        assert_ne!(ciphertexts.iter().flatten().cloned().collect::<Vec<u8>>(), cell);

        let plain = decrypt_stored_cell(&recipient, &ciphertexts).unwrap();
        assert_eq!(plain, cell);

        // The reassembled plaintext is still addressable by the same CID.
        let restored = store_cell(&plain);
        assert_eq!(restored.cid, store_cell(&cell).cid);
    }
}
