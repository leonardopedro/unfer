# Alternative-to-Git: Zenodo + Loro CRDT incremental persistence

## Architecture overview

The `zenodo_store_module` (P11.24) adapts the incremental snapshot/delta
pattern from [altgit.md in the test directory] to the unfer ecosystem. It
provides **persistent, versioned, deduplicated storage for Loro CRDT
documents** — specifically the velysterm math editor document — using
**Zenodo** (a CERN open-science repository) as the backend.

The key insight from the original pattern: instead of uploading the full
document on each save (which would re-upload all unchanged history),
we maintain an **append-only sequence of binary files** on Zenodo:

```
Zenodo v1: snapshot_0.bin  + manifest.json
Zenodo v2: snapshot_0.bin  + delta_1.bin   + manifest.json
Zenodo v3: snapshot_0.bin  + delta_1.bin   + delta_2.bin + manifest.json
...
Zenodo v51: snapshot_1.bin + manifest.json  (squash every 50 deltas)
```

Each Zenodo version carries all prior files forward automatically (Zenodo
versions are append-only; the backend handles deduplication). Only the new
binary (`delta_N.bin`) and the updated `manifest.json` are uploaded on each
save — O(delta_size) network traffic per save, not O(document_size).

## Adapted for unfer

### What changes from the original JavaScript pattern

The original (`../test/software/altgit.md`) describes a JavaScript
implementation using the Loro WASM bindings. The unfer adaptation:

| Aspect | Original (JS) | Adapted (Rust/Austral) |
|--------|---------------|------------------------|
| Language | JavaScript | Rust FFI (`uz_*` symbols) + Austral module |
| Loro API | `doc.export({mode:"update",from:frontier})` | Caller (velysterm) generates bytes; `uz_push(bytes, frontier)` stores them |
| HTTP client | `fetch` | `ureq` (sync, no async runtime) |
| Auth | Personal access token | Same — passed in config JSON |
| Manifest | In-memory JS object | `ZenodoManifest` struct, serialized to `manifest.json` on Zenodo |
| Squash trigger | `file_sequence.length > 50` | `delta_count >= squash_after` (configurable, default 50) |

### Frontier bytes

The **frontier** (Loro's version vector) is an opaque byte sequence that
Loro uses to compute the delta since the last save:
```javascript
// velysterm side (JS/WASM)
const frontier = doc.frontiers(); // Uint8Array
const delta = doc.export({mode: "update", from: frontier});
```

The unfer adapter is agnostic to the frontier format — it stores it
base64-encoded in `manifest.json` and returns it to the caller on pull.

### Integration with velysterm

The velysterm math editor calls the `uz_*` symbols via the australVM JIT
(the Austral `ZenodoStore` bindings in `australVM/examples/zenodo/`):

```
velysterm editor → ZenodoStore.aum → uz_push(loro_delta, frontier) → Zenodo
velysterm loader → ZenodoStore.aum → uz_pull(buf, cap) → [all files merged]
                                   → uz_manifest_json → last_frontier
```

The pull returns concatenated bytes in `file_sequence` order; velysterm
feeds them to `doc.import(bytes)` sequentially. Because Loro's `import` is
idempotent for CRDT operations, the concatenation is safe.

## File layout

```
$ROOT/
├── unfer/
│   ├── unfer_ffi/
│   │   └── src/zenodo.rs       # uz_* Rust implementation (ureq HTTP)
│   └── zenodo_store_module/
│       ├── module.toml         # archetype: actor; grants: uz_* + uk_version
│       ├── src/
│       │   ├── ZenodoStoreDemo.aui
│       │   └── ZenodoStoreDemo.aum
│       └── run_demo.sh
└── australVM/
    └── examples/zenodo/
        ├── ZenodoStore.aui     # Typed Austral bindings for uz_*
        └── ZenodoStore.aum
```

## C ABI (`uz_*` symbols)

All functions follow the same buffer protocol as `uk_*`:
- `>= 0`: success (bytes written / 0 for void ops)
- `< 0`: `-code` (call `uz_last_error` for the message)

Buffer protocol for output functions: probe with `buf=null, cap=0` to get
total bytes needed; allocate; call again to fill.

| Symbol | Semantics |
|--------|-----------|
| `uz_init(cfg_json, len) -> i64` | Configure: `{"api_key":"TOKEN","sandbox":true,"record_id":null,"squash_after":50}` |
| `uz_push(data, data_len, frontier, frontier_len) -> i64` | Upload snapshot (first call) or delta + update manifest + publish |
| `uz_pull(buf, cap) -> i64` | Download all `file_sequence` files concatenated |
| `uz_manifest_json(buf, cap) -> i64` | Return in-memory manifest JSON (no HTTP) |
| `uz_last_error(buf, cap) -> i64` | Last error string (thread-local) |

## Zenodo API used

The `uz_*` implementation uses the **Zenodo v2 deposit REST API**:

- Create deposition: `POST /api/deposit/depositions`
- Upload file via bucket: `PUT /api/files/{bucket_id}/{filename}`
- Publish: `POST /api/deposit/depositions/{id}/actions/publish`
- New version (for deltas): `POST /api/deposit/depositions/{record_id}/actions/newversion`
- File listing for pull: `GET /api/records/{record_id}`

Sandbox: `https://sandbox.zenodo.org/api`  
Production: `https://zenodo.org/api`

### Getting a Zenodo token

1. Register at https://sandbox.zenodo.org (for testing) or https://zenodo.org
2. Settings → Applications → Personal access tokens → New token
3. Scopes needed: `deposit:write`, `deposit:actions`
4. Pass as `"api_key"` in the config JSON

## Squash heuristic

Every `squash_after` deltas (default 50), `uz_push` creates a new full
snapshot instead of a delta:

1. Uploads `snapshot_N.bin` (N = snapshot counter, incremented)
2. Clears `manifest.file_sequence` (starting fresh from the new snapshot)
3. Old deltas stay accessible on prior Zenodo versions (immutable history)

The squash threshold is configurable per-session via `uz_init`.

## Running the demo

```bash
# Local demo (no network — tests init + manifest probe):
cd unfer/zenodo_store_module
bash run_demo.sh

# Full Zenodo sandbox round-trip:
ZENODO_API_KEY=your-sandbox-token bash run_demo.sh
```

## Authorization gate

The `uz_*` symbols are subject to the same australVM manifest authorization
as `uk_*` (UK-4001 CallDenied). A module must declare `uz_push` (and other
`uz_*` it uses) in its `[grants]` → `zenodo` list. Removing a symbol from
grants causes the JIT to deny the call with UK-4001.

## Connection to P11

This module is part of the P11 external-module integration roadmap:

- **P11.19** (`pattern_unfer`) plans to use Loro for persistent kernel session
  memory. The Zenodo adapter provides an internet-accessible backend for that
  Loro document without requiring a custom server — Zenodo's immutable
  versioning + deduplication does what a Git remote would, but for CRDT
  binary data.
- **P11.20** (`arctic_authority`) — a future extension could add Arctic
  threshold-signature authentication to the Zenodo upload: the frontier
  bytes (or the delta itself) could be signed by a threshold authority before
  upload, making the history cryptographically attested.
- **P11.22** (`unfer_edge`) — the Pingora proxy could front the Zenodo API
  calls, adding rate limiting and credential rotation without changing the
  `uz_*` interface.
