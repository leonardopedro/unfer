# Federation Layer — Consensus, Identity, Content

> QuePaxa-inspired consensus engine, DID identity lifecycle, and signed
> content publishing for the unfer kernel.

## Crate Layout

```
unfer_protocol/     shared wire types (ConsensusTransaction, IdentityOp,
                    ContentRef, ChunkRef, DidEntry, DidDocument, Code)
       ↓
unfer_consensus/    ConsensusEngine trait, LocalConsensus, ConsensusNode,
                    IdentityRegistry, Keypair (Ed25519), sign/verify
       ↓
  ┌────┴────┐
  ↓         ↓
unfer_identity/   unfer_data/
DidManager        Chunker, DataKeypair (X25519),
(create/update/   AES-256-GCM encrypt/decrypt,
 revoke/resolve)  magnet URIs, DataPublisher
```

## `unfer_consensus`

### `ConsensusEngine` trait

```rust
pub trait ConsensusEngine: Send + Sync {
    fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic>;
    fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)>;
    fn current_seq(&self) -> u64;
}
```

`LocalConsensus`: single-process `Arc<RwLock<Vec>>` append-only log.
Monotonic sequence numbers. Cloneable (shares underlying log via `Arc`).

Optional `network` feature gates `rust-quepaxa` + `tokio` for a future
distributed engine behind the same trait.

### `ConsensusNode`

State machine that replays the consensus log:

- `submit(tx)` — verifies Ed25519 signature, then delegates to engine.
- `sync()` — pulls entries after `applied_seq`, replays via
  `apply_transaction`. Returns count of newly applied transactions.
- Dispatches on transaction variant:
  - `IdentityOp` → `IdentityRegistry::apply_identity_op`
  - `SessionOp` → creates `prob_kernel::Session` (currently `create_model` only)
  - `ContentOp` → inserts into content HashMap keyed by CID.

### `IdentityRegistry`

In-memory DID ledger (`HashMap<String, DidEntry>`):

| Operation | Semantics |
|-----------|-----------|
| Create (seq=1) | Registers DID + Ed25519 pubkey + optional service endpoint. Rejects duplicates. |
| Update (seq+1) | Changes service endpoint. Rejects stale seq, unknown/revoked DIDs. |
| Revoke (seq+1) | Marks revoked. `resolve()` returns `None`; `contains()` returns `true` (tombstone). |

`resolve_document(did)` produces a W3C DID Document
(`@context: "https://www.w3.org/ns/did/v1"`, `Ed25519VerificationKey2020`).

### Signing

- `Keypair`: wraps `ed25519_dalek::SigningKey`. `generate()` (OsRng),
  `did()` → `"did:unfer:<64 hex>"`.
- `canonical_bytes(tx)`: zeroes signature field, JSON-serializes, SHA-256.
- `sign_transaction(tx, keypair)`: signs canonical bytes, writes signature.
- `verify_transaction(tx)`: extracts pubkey, verifies Ed25519 signature.

## `unfer_identity`

### `DidManager<'a>` (borrows `&'a mut ConsensusNode`)

```rust
impl DidManager<'_> {
    fn create_did(&mut self, kp: &Keypair, endpoint: Option<String>) -> Result<String, Diagnostic>;
    fn update_did(&mut self, kp: &Keypair, endpoint: Option<String>) -> Result<(), Diagnostic>;
    fn revoke_did(&mut self, kp: &Keypair) -> Result<(), Diagnostic>;
    fn resolve(&self, did: &str) -> Option<DidDocument>;
    fn resolve_json(&self, did: &str) -> Option<String>;
}
```

Each operation: build signed transaction → submit → sync → return result.

Free functions: `did_from_pubkey(&[u8; 32]) -> String`,
`pubkey_from_did(&str) -> Option<[u8; 32]>`.

## `unfer_data`

### Chunking

`Chunker { chunk_size }` (default 256 KiB):
- `chunk(data) -> Vec<(u32, Vec<u8>)>` — index + bytes.
- `chunk_refs(data) -> Vec<ChunkRef>` — index + CID (SHA-256) + size.
- `compute_cid(data) -> String` — SHA-256 hex.
- `verify_chunk(data, expected_cid) -> bool`.

### Encryption

`DataKeypair` wraps `x25519_dalek::StaticSecret`:
- `shared_secret(peer_pub) -> [u8; 32]` — X25519 DH + SHA-256.
- `derive_aes_key(shared) -> [u8; 32]` — SHA-256 of shared secret.
- `encrypt_chunk(aes_key, chunk_index, plaintext) -> Vec<u8>` — AES-256-GCM,
  nonce = `[0u8; 8] || chunk_index.to_le_bytes()` (binds ciphertext to position).
- `decrypt_chunk(aes_key, chunk_index, ciphertext) -> Vec<u8>`.

### Magnet URIs

- `build_magnet_uri(cid, name) -> "magnet:?xt=urn:btih:<cid>&dn=<name>"`.
- `content_cid_from_chunks(chunk_cids) -> String` — SHA-256 of concatenated
  chunk CIDs (order-sensitive Merkle-like root).
- `parse_magnet_cid(uri) -> Option<String>`.

### `DataPublisher<'a>` (borrows `&'a mut ConsensusNode`)

```rust
publisher.publish(did_keypair, data, mime_type, name) -> Result<ContentRef, Diagnostic>
```

Pipeline: generate X25519 keypair → derive AES key → chunk → encrypt each
chunk → compute per-chunk CIDs → aggregate root CID → build magnet URI →
construct `ContentRef` → sign as `ContentOp` → submit → sync.

`resolve(cid) -> Option<&ContentRef>`.

## UK Error Codes (6xxx range)

| Code | Name | Trigger |
|------|------|---------|
| UK-6001 | ConsensusNotReady | State machine not initialized |
| UK-6002 | DuplicateTransaction | Same seq already applied |
| UK-6003 | InvalidSignature | Ed25519 verification failed |
| UK-6004 | UnknownDid | DID not found or revoked |
| UK-6005 | RelayNotConnected | Relay transport not connected |

## Agent Ops

| Op | Description |
|----|-------------|
| `did_create` | Create DID with fresh Ed25519 keypair |
| `did_resolve` | Resolve DID → W3C DID Document |
| `did_update` | Update service endpoint |
| `did_revoke` | Revoke DID |
| `content_publish` | Publish CID + metadata, signed by DID |
| `content_resolve` | Resolve CID → ContentRef |
| `consensus_sync` | Advance state machine (apply pending txs) |
| `consensus_status` | Query current state without advancing |

See `docs/PROTOCOL.md` for request/response shapes.

## Tests

- `unfer_consensus`: 14 tests (engine 3, identity 7, signing 5, node 5).
- `unfer_identity`: 6 tests (create/resolve, update, revoke, JSON, roundtrip,
  duplicate).
- `unfer_data`: 22 tests (chunk 8, crypto 8, magnet 8, publisher 5).
