# Data Plane — Chunking, Encryption, Magnet URIs

> Encrypted chunked content storage with X25519 key exchange, AES-256-GCM
> per-chunk encryption, SHA-256 content addressing, and magnet URI
> distribution.

## Architecture

```
plaintext bytes
  │
  ▼
Chunker::chunk(data)  ──→  [(0, chunk₀), (1, chunk₁), ...]
  │                          default chunk_size = 256 KiB
  ▼  (per chunk)
encrypt_chunk(aes_key, index, plaintext)
  │  AES-256-GCM, nonce = [0u8; 8] ‖ index.to_le_bytes()
  ▼
ciphertext
  │
  ▼
compute_cid(ciphertext)  ──→  per-chunk SHA-256 CID
  │
  ▼
content_cid_from_chunks([cid₀, cid₁, ...])  ──→  root CID
  │  SHA-256 of concatenated chunk CIDs (order-sensitive)
  ▼
build_magnet_uri(root_cid, name)  ──→  "magnet:?xt=urn:btih:<cid>&dn=<name>"
  │
  ▼
ContentRef { cid, magnet_uri, encryption_key, filesize, mime_type, chunks }
  │
  ▼
ConsensusTransaction::ContentOp  ──→  signed, submitted, synced
```

## Chunking (`chunk.rs`)

```rust
pub struct Chunker { chunk_size: usize }  // default 256 KiB, min 1

impl Chunker {
    fn chunk(&self, data: &[u8]) -> Vec<(u32, Vec<u8>)>;
    fn chunk_refs(&self, data: &[u8]) -> Vec<ChunkRef>;
}

pub fn compute_cid(data: &[u8]) -> String;        // SHA-256 hex (64 chars)
pub fn verify_chunk(data: &[u8], cid: &str) -> bool;
```

`ChunkRef { index: u32, cid: String, size: usize }`.

## Encryption (`crypto.rs`)

### Key exchange

```rust
pub struct DataKeypair { secret: StaticSecret, public: PublicKey }

impl DataKeypair {
    fn generate() -> Self;                              // OsRng
    fn shared_secret(&self, peer: &PublicKey) -> [u8; 32]; // X25519 DH + SHA-256
}

pub fn derive_aes_key(shared: &[u8; 32]) -> [u8; 32];  // SHA-256
```

### Per-chunk encryption

```rust
pub fn encrypt_chunk(key: &[u8; 32], index: u32, pt: &[u8]) -> Result<Vec<u8>, String>;
pub fn decrypt_chunk(key: &[u8; 32], index: u32, ct: &[u8]) -> Result<Vec<u8>, String>;
```

AES-256-GCM with deterministic 12-byte nonce: bytes `[0..8]` zero, bytes
`[8..12]` = chunk index (little-endian). This binds each ciphertext to its
chunk position — reordering chunks causes authentication failure.

### Forward secrecy

`generate_ephemeral() -> (EphemeralSecret, PublicKey)` for ephemeral key
exchange (not yet used by the publisher).

## Magnet URIs (`magnet.rs`)

```rust
pub fn build_magnet_uri(cid: &str, name: Option<&str>) -> String;
// "magnet:?xt=urn:btih:<cid>&dn=<percent-encoded name>"

pub fn content_cid_from_chunks(chunk_cids: &[String]) -> String;
// SHA-256 over concatenation of all chunk CID strings

pub fn parse_magnet_cid(uri: &str) -> Option<String>;
// extracts CID from xt=urn:btih: parameter
```

Percent-encoding: spaces → `+`; non-alphanumeric/non-`-_.~` → `%XX`.

## Publisher (`publisher.rs`)

```rust
pub struct DataPublisher<'a> { node: &'a mut ConsensusNode, chunker: Chunker }

impl DataPublisher<'_> {
    fn new(node: &mut ConsensusNode) -> Self;
    fn with_chunk_size(node: &mut ConsensusNode, size: usize) -> Self;

    fn publish(
        &mut self,
        did_keypair: &Keypair,      // Ed25519 signing key
        data: &[u8],
        mime_type: &str,
        display_name: Option<&str>,
    ) -> Result<ContentRef, Diagnostic>;

    fn resolve(&self, cid: &str) -> Option<&ContentRef>;
}
```

`publish` pipeline:
1. Generate fresh X25519 `DataKeypair` for this content object.
2. Derive AES key via self-DH (`shared_secret(own_public_key)`).
3. Chunk data, encrypt each chunk (nonce = chunk index).
4. Compute per-chunk CIDs (SHA-256 of ciphertext) and aggregate root CID.
5. Build magnet URI.
6. Construct `ContentRef { cid, magnet_uri, encryption_key: "x25519:<hex>",
   filesize, mime_type, chunks }`.
7. Wrap in `ConsensusTransaction::ContentOp`, sign with DID keypair,
   submit, sync.

The encryption key is stored in the `ContentRef` as `x25519:<hex pubkey>`.
Decryption requires the corresponding X25519 private key (held by the
publisher).

## ContentRef wire type

```rust
pub struct ContentRef {
    pub cid: String,                    // root CID (SHA-256 of chunk CIDs)
    pub magnet_uri: String,             // magnet:?xt=urn:btih:<cid>
    pub encryption_key: String,         // "x25519:<hex pubkey>"
    pub filesize: u64,                  // original plaintext size
    pub mime_type: String,
    pub chunks: Vec<ChunkRef>,          // per-chunk { index, cid, size }
}
```

## Security properties

- **Confidentiality**: AES-256-GCM per chunk; key derived from X25519 DH.
- **Integrity**: GCM authentication tag per chunk; wrong key or reordered
  chunks fail decryption.
- **Content addressing**: SHA-256 CIDs for each chunk and the aggregate;
  tampering changes the CID.
- **Non-repudiation**: ContentOp signed with Ed25519 DID keypair; consensus
  log provides ordering.

## Tests

22 tests across 4 modules:
- `chunk.rs` (8): splitting, single chunk, empty input, CID determinism,
  verification, reassembly.
- `crypto.rs` (8): keypair uniqueness, DH symmetry, encrypt/decrypt
  roundtrip, wrong key/index rejection, empty/large chunks.
- `magnet.rs` (8): URI construction, special char encoding, CID
  determinism/order-sensitivity, parse roundtrip.
- `publisher.rs` (5): end-to-end publish+resolve, multi-chunk, empty data,
  unknown CID, unique chunk CIDs.
