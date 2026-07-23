# QuePaxa Federation Plan — Adapted for unfer/australVM/velysterm

> Adapted from `docs/quepaxa.md`. The original plan targets a generic
> atproto social network. This version maps every phase onto the existing
> unfer kernel architecture: `unfer_protocol` types, `prob_kernel::Session`,
> `unfer_ffi` C ABI, `unfer_edge` Pingora gateway, `kernel_client` NDJSON
> agent, and the australVM JIT module system.

## Key Adaptation Decisions

1. **Protocol is `unfer_protocol`, not atproto XRPC.** All wire types are
   `AgentRequest`/`AgentResponse` with `Diagnostic`/`RepairHint` error
   handling. New federation ops get new UK-#### codes (6xxx range).

2. **The state machine IS `prob_kernel::Session`.** Consensus sequences
   Session operations (`create_model`, `evolve`, `condition`, etc.).
   Determinism is guaranteed: same op sequence → same state.

3. **Modules are Austral cells in the JIT.** New federation functionality
   is exposed as `uk_*` symbols, registered in `cranelift_init()`, and
   callable from `.aui`/`.aum` modules with manifest grants.

4. **The relay extends `unfer_edge`, not a new server.** The existing
   Pingora gateway already does op-allowlisting, secret masking, and
   rate limiting. Federation ops are added to the allowlist.

5. **Identity is `did:unfer`, not `did:plc`.** A new DID method backed
   by the QuePaxa consensus log, using ed25519-dalek signatures.

6. **Data plane uses `unfer_nixvm` isolation.** Content nodes run inside
   the cloud-hypervisor VM guest, not as bare processes.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │         QuePaxa Consensus Cluster       │
                        │         (20-50 vetted nodes, CFT)       │
                        │                                         │
                        │  ┌─────────┐  ┌─────────┐  ┌────────┐  │
                        │  │ Node A  │  │ Node B  │  │ Node C │  │
                        │  │ Session │  │ Session │  │Session │  │
                        │  │ Store   │  │ Store   │  │ Store  │  │
                        │  └────┬────┘  └────┬────┘  └───┬────┘  │
                        │       │            │           │       │
                        │       └────────────┼───────────┘       │
                        │                    │                   │
                        │         Consensus Log (ordered)        │
                        └────────────────────┬───────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
              ┌─────▼─────┐           ┌──────▼──────┐          ┌─────▼─────┐
              │ unfer_edge │           │ unfer_edge  │          │ unfer_edge│
              │ (relay)    │           │ (relay)     │          │ (relay)   │
              │ + firehose │           │ + firehose  │          │ + firehose│
              └─────┬──────┘           └──────┬──────┘          └─────┬─────┘
                    │                         │                       │
              ┌─────▼─────┐           ┌──────▼──────┐          ┌─────▼─────┐
              │kernel_client│          │kernel_client│          │kernel_client│
              │(velysterm) │          │(AI agent)   │          │(Austral mod)│
              └────────────┘          └─────────────┘          └───────────┘
```

## Phase 1: Consensus Layer — `unfer_consensus` crate

**Goal:** Transaction types + consensus trait + in-process node for testing.

**New crate:** `unfer/unfer_consensus/`

**Dependencies:** `unfer_protocol`, `prob_kernel`, `serde`, `serde_json`,
`sha2`, `ed25519-dalek`, `tokio` (optional, for network feature).

**Types (in `unfer_protocol`):**

```rust
// New UK codes: 6xxx = federation errors
UK-6001  ConsensusNotReady     // node not yet synced
UK-6002  DuplicateTransaction  // tx already in the log
UK-6003  InvalidSignature      // ed25519 verification failed
UK-6004  UnknownDid            // DID not in the registry
UK-6005  RelayNotConnected     // no upstream relay available

// Transaction types
enum ConsensusTransaction {
    IdentityOp(IdentityOp),     // DID create/update/revoke
    SessionOp(SessionOp),       // kernel Session operation
    ContentOp(ContentOp),       // content reference publish
}

struct IdentityOp {
    did: String,                // "did:unfer:<hex-pubkey>"
    op_kind: IdentityOpKind,    // Create | Update | Revoke
    signing_key: [u8; 32],     // ed25519 public key
    signature: [u8; 64],       // ed25519 signature over the op
    seq: u64,                   // monotonic sequence per DID
}

struct SessionOp {
    did: String,                // author
    model_id: u64,             // Session handle
    op: AgentRequest,          // the kernel op to apply
    signature: [u8; 64],
}

struct ContentOp {
    did: String,
    content_ref: ContentRef,   // magnet_uri + encryption_key + size
    signature: [u8; 64],
}
```

**Consensus trait:**

```rust
trait ConsensusEngine: Send + Sync {
    fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic>;
    fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)>;
    fn current_seq(&self) -> u64;
}
```

**In-process implementation** (`LocalConsensus`): single-node, in-memory
log. Used for testing and single-node deployments. The `network` feature
gates the `rust-quepaxa` backend.

**Node:**

```rust
struct ConsensusNode {
    engine: Box<dyn ConsensusEngine>,
    sessions: HashMap<u64, Session>,
    identity: IdentityRegistry,
    next_model_id: u64,
}
```

The node applies consensus-ordered transactions to its local state:
- `IdentityOp::Create` → register DID + pubkey
- `SessionOp` → dispatch to the local Session (create/evolve/condition)
- `ContentOp` → store content reference

**Tests:** submit → order → apply → verify state. Determinism: two nodes
applying the same log produce identical state.

## Phase 2: Identity Registry — `unfer_identity` crate

**Goal:** `did:unfer` method backed by the consensus log.

**New crate:** `unfer/unfer_identity/`

**Dependencies:** `unfer_protocol`, `unfer_consensus`, `ed25519-dalek`,
`sha2`, `serde`.

**DID format:** `did:unfer:<hex(ed25519-pubkey)>`

**DID Document:**

```json
{
  "@context": "https://www.w3.org/ns/did/v1",
  "id": "did:unfer:abcd1234...",
  "verificationMethod": [{
    "id": "did:unfer:abcd1234...#key-1",
    "type": "Ed25519VerificationKey2020",
    "publicKeyMultibase": "z..."
  }],
  "authentication": ["did:unfer:abcd1234...#key-1"],
  "service": [{
    "id": "did:unfer:abcd1234...#unfer",
    "type": "UnferKernelEndpoint",
    "serviceEndpoint": "https://node.example.com"
  }]
}
```

**Registry:**

```rust
struct IdentityRegistry {
    entries: HashMap<String, DidEntry>,  // DID → entry
}

struct DidEntry {
    did: String,
    pubkey: [u8; 32],
    seq: u64,
    created_at: u64,
    revoked: bool,
    service_endpoint: Option<String>,
}
```

**Operations:**
- `create_did(keypair) → DidDocument` — generates a new DID, signs the
  creation op, submits to consensus.
- `update_did(did, keypair, changes) → DidDocument` — updates service
  endpoint or rotates key.
- `revoke_did(did, keypair)` — marks the DID as revoked.
- `resolve_did(did) → DidDocument` — reads from the local registry
  (populated by consensus log replay).

**HTTP endpoint** (added to `unfer_edge`):
- `GET /.well-known/did.json?did=did:unfer:...` → DID document JSON.

**New `uk_*` symbols:**
- `uk_did_create(keypair_json, len) → handle`
- `uk_did_resolve(did_ptr, len, buf, cap) → needed`
- `uk_did_sign(did_handle, msg_ptr, len, sig_buf, cap) → needed`

## Phase 3: Relay & Firehose — extend `unfer_edge`

**Goal:** Stream consensus-ordered events to subscribers.

**Changes to `unfer_edge`:**
- Add `subscribe_firehose` to `ALLOWED_OPS`.
- New WebSocket endpoint: `GET /firehose` — streams `ConsensusTransaction`
  events as NDJSON as they are committed to the consensus log.
- New HTTP endpoint: `GET /log?from=<seq>` — returns the consensus log
  from a given sequence number (for catch-up sync).

**Changes to `kernel_client`:**
- New ops: `did_create`, `did_resolve`, `subscribe_firehose`.
- The `unfer_agent` binary gets a `--relay` flag that connects to a
  relay and replays the consensus log into a local `ConsensusNode`.

**Firehose event format:**

```json
{"seq": 42, "tx": {"type": "session_op", "did": "did:unfer:...", "op": {...}}}
```

**Catch-up protocol:**
1. Client connects to relay.
2. Client sends `GET /log?from=<last_known_seq>`.
3. Relay returns all transactions since that seq.
4. Client applies them to its local `ConsensusNode`.
5. Client switches to WebSocket firehose for real-time updates.

## Phase 4: Data Plane — `unfer_data` crate

**Goal:** Content references with P2P delivery metadata.

**New crate:** `unfer/unfer_data/`

**Types (in `unfer_protocol`):**

```rust
struct ContentRef {
    cid: String,           // SHA-256 content hash (hex)
    magnet_uri: String,    // BitTorrent magnet link
    encryption_key: String, // X25519 public key for chunk encryption
    filesize: u64,
    mime_type: String,
    chunks: Vec<ChunkRef>,
}

struct ChunkRef {
    index: u32,
    cid: String,
    size: u64,
}
```

**Data Node module** (Austral cell):
- `data_publish.aui` — takes a file path, chunks it, computes CIDs,
  encrypts chunk keys, publishes a `ContentOp` to consensus.
- `data_resolve.aui` — takes a CID, looks up the `ContentRef` from
  the consensus log, returns the magnet URI.

**New `uk_*` symbols:**
- `uk_content_publish(ref_json, len) → seq`
- `uk_content_resolve(cid_ptr, len, buf, cap) → needed`

## Phase 5: Client Integration — extend velysterm

**Goal:** The mathed editor and AI agent can use federation features.

**Changes to `kernel_client`:**
- New ops in `unfer_agent`: `did_create`, `did_resolve`, `firehose_subscribe`,
  `content_publish`, `content_resolve`.
- The worker thread manages a `ConsensusNode` alongside the `Session` map.

**Changes to `mathed_core`:**
- New `PropKind::Did` — a DID reference in a math document.
- New `PropKind::ContentRef` — a content reference (video, dataset).

**Changes to `mathed`:**
- `kernel_sys.rs` handles the new ops.
- Overlay renders DID resolution status (green = resolved, red = UK-6004).

## UK Code Allocation (6xxx = federation)

| Code   | Name                  | Severity | Description                                    |
|--------|-----------------------|----------|------------------------------------------------|
| UK-6001 | ConsensusNotReady    | Error    | The consensus node has not yet synced.         |
| UK-6002 | DuplicateTransaction | Error    | The transaction is already in the consensus log.|
| UK-6003 | InvalidSignature     | Error    | Ed25519 signature verification failed.         |
| UK-6004 | UnknownDid           | Error    | The DID is not in the identity registry.       |
| UK-6005 | RelayNotConnected    | Error    | No upstream relay is available.                |

## Execution Order

1. **Phase 1** (this session): `unfer_consensus` crate + protocol types + tests.
2. **Phase 2**: `unfer_identity` crate + DID registry + tests.
3. **Phase 3**: `unfer_edge` relay extensions + firehose + tests.
4. **Phase 4**: `unfer_data` crate + content refs + tests.
5. **Phase 5**: velysterm integration + end-to-end tests.

Each phase is independently testable. Phase 1-2 are pure library crates
(no network). Phase 3 adds HTTP. Phase 4-5 add P2P metadata and UI.
