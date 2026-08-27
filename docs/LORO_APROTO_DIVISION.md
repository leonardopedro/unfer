# Loro × AT Protocol — Division of Labor (Normative)

> The single reference for **which mechanism owns which bytes** across the
> five sibling repos (`unfer`, `australVM`, `velysterm`, `dynamic-arctic`,
> `timepiece`). It exists to prevent the two technologies from drifting into
> duplicated features: Loro and the AT Protocol side overlap in *reach*
> (both can carry state between machines), so every surface must pick
> exactly one — the better one — and say so here.

**The one-sentence rule.**

> **Loro is *state*; the AT Protocol side is *identity-bound publication*.**
> Loro owns documents, their convergence, their presence, and durable event
> streams. The AT Protocol side owns the collective cryptographic authority,
> delegation certificates, and public identity-bound records. The same bytes
> must never live in both.

---

## 1. The two technologies, as actually deployed

| | **Loro** (CRDT) | **AT Protocol side** |
|---|---|---|
| What it is | Conflict-free replicated datatypes: document model, delta sync, ephemeral presence, version vectors/frontiers | Collective threshold authority (`did:web`), Arctic/Shine threshold Schnorr, delegation certificates; the atproto record/relay model |
| Where it lives | `velysterm/crates/mathed_core` (doc model, `sync.rs` deltas + `PresenceStore`), `unfer/unfer_ffi/src/durable/loro.rs` (kernel event streams), `unfer/unfer_ffi/src/zenodo.rs` (Loro snapshot/delta persistence), `../pattern` `pattern_unfer` (session memory, external repo) | `dynamic-arctic` (authority server + `arctic_core`/`shine_core` library), `australVM/arctic_authority` (verification/authorization engine + optional `arctic-auth` JIT feature), `unfer/unfer_identity` (`did:unfer` registry, consensus-sequenced), `unfer/unfer_consensus` (threshold mint gate, firehose/relay plan in `docs/QUEPAXA_PLAN.md`) |
| Strength | Merging concurrent, offline-capable edits without a coordinator; cheap ephemeral gossip | Single global order; cryptographic attribution; threshold *n*-of-*t* authority with no single point of trust |
| Weakness | No global order; no identity semantics | Requires coordination (consensus/authority); useless for concurrent offline edits |

The overlap to police: *both* can carry state between machines. The division
below assigns every cross-machine byte-stream to exactly one of them.

---

## 2. Identity chain (the only bridge)

```
dynamic-arctic (did:web authority)          unfer_identity (did:unfer registry)
  Arctic/Shine threshold Schnorr              consensus-sequenced per-participant
  serves /.well-known/did.json                identities in the QuePaxa log
        │                                              ▲
        │ DelegationCertificate (hot key, expiry,      │ records the authority's
        │ capabilities — e.g. "atproto-signing")       │ group key as an entry
        ▼                                              │
australVM/arctic_authority ── verifies cert ───────────┘
(+ safestos/cranelift "arctic-auth" feature)
        │
        ▼
unfer_consensus CertificateLedger
  MintAuthority::Threshold { t, n, pubkey } — verify_arctic_threshold
```

**Rules.**

1. **Exactly one authority per DID method.** `dynamic-arctic` is the *only*
   `did:web` authority; it never grows a `did:unfer` registry.
   `unfer_identity` is the only `did:unfer` registry; it never serves a
   second `did:web` document endpoint.
2. **Delegation certificates are the only bridge** between the two systems.
   A `did:unfer` participant proves a capability granted by the collective
   authority by presenting an Arctic-signed `DelegationCertificate`, verified
   by `arctic_authority` (australVM) — never by re-implementing verification
   inside `unfer_identity`.
3. **The registry records, it does not judge.** `unfer_identity` stores the
   authority's group public key and the certificate's *effects*; whether a
   certificate is valid is `arctic_core::verify`'s job, reached only through
   `australVM/arctic_authority` (P11.20).
4. **Threshold signing is shared, identity is not.** The 64-byte Arctic
   aggregate signature `(RistrettoPoint, Scalar)` (`arctic_core::
   signature_to_bytes`) is the shared wire format used by both the AT
   Protocol ceremony and the `unfer_consensus` mint gate. That sharing is
   *intended reuse of cryptography*, not feature duplication.

---

## 3. Persistence homes (one home per byte stream)

A stream has **exactly one** durable home. Backends are alternatives, never
mirrors (already enforced by `unfer_ffi/src/durable/mod.rs`: "a given stream
lives in exactly one store").

| Byte stream | Home | Why this is the better solution |
|---|---|---|
| Kernel event streams (audit, owner log, actions, config, session) | `unfer_ffi` durable store — **Loro preferred** (`LoroDurableStore`), JSONL/SQLite as operator-chosen alternatives | Frontier/fork semantics come free; append-only lists; coalesced flush |
| Editor document history (the velysterm math doc) | Loro doc in the editor; long-term persistence via `zenodo_store_module` (`uz_*` snapshot/delta chain) | Incremental O(delta) saves with immutable versioning — a Git remote for CRDT bytes |
| AI-being session memory | `../pattern` `pattern_unfer` (external repo) wrapping `SessionBlob` in a Loro `StructuredDocument`, reached only over the `unfer_agent` NDJSON boundary | Keeps the AGPL/MPL copyleft surface out of the permissive kernel |
| Heavy content (video, datasets) | `unfer_data` `ContentRef` (magnet URI + encryption key) published as a consensus `ContentOp` | Large bytes never enter Loro, the consensus log, or an atproto record — only the *reference* does |
| Public identity-bound records | AT Protocol records / (planned) QuePaxa-backed relay firehose | Publication is identity-attributed and globally ordered — exactly what CRDTs do not provide |

**Rules.**

5. **The AT Protocol side is never a durable backend.** No `DurableStore`
   implementation may be backed by a PDS/relay; `durable::Backend` stays
   `{Loro, Jsonl, Sqlite}`. Publishing a *record* about a stream is allowed;
   storing the stream in it is not.
6. **Loro never stores identity claims.** Document content and durable
   streams carry no signing keys and no capability grants; those live in
   `unfer_identity`/`arctic_authority`.
7. **Heavy content is a reference, twice at most.** A `ContentOp` (consensus)
   may be mirrored into a public atproto record; the *bytes behind the
   magnet* are fetched from the data plane, never re-hosted by Loro or atproto.

---

## 4. Sync, presence, transport

| Concern | Owner | Non-owner |
|---|---|---|
| Real-time convergence of concurrent edits | Loro delta exchange (`mathed_core::sync::export_delta`/`import_delta`) | atproto (no offline merge model) |
| Live presence (who is here, carets) | Loro `EphemeralStore` (`PresenceStore`) — gossip, never persisted, never identity | consensus log (too heavy, wrong lifetime) |
| Global ordering of committed effects | QuePaxa consensus log (`unfer_consensus`) | Loro (CRDT order is per-document, not global) |
| Public broadcast of committed records | (planned) `unfer_edge` firehose / atproto relay | Loro sync channels |

**Rules.**

8. **Presence is not identity.** `PresenceStore` peer ids and display names
   are cosmetic transport labels. Authenticated attribution comes only from
   signed operations (delegation certificate, `did:unfer` signature over a
   consensus op). Nothing in the presence layer may gate authority.
9. **Transport boundaries are explicit.** Loro sync rides the editor's
   live channel (TCP presence transport, C13). The consensus log is reached
   through `ConsensusNode` / `unfer_edge`. An atproto firehose, when built,
   is a *mirror* of committed log entries — never a second path into kernel
   state.

---

## 5. Quick decision table

| You want to… | Use | Do not use |
|---|---|---|
| Merge two people's offline edits | Loro CRDT | consensus (no offline model), atproto |
| Show live cursors | `PresenceStore` (ephemeral) | persisted CRDT content, consensus, atproto |
| Persist kernel audit/action streams | `LoroDurableStore` (or JSONL/SQLite) | Zenodo (that is editor history), PDS, consensus log |
| Persist the editor document long-term | Zenodo `uz_*` chain | kernel durable store, PDS |
| Attribute an effect to *someone* | `did:unfer` op + signature in consensus | Loro metadata, presence |
| Prove a capability was granted collectively | Arctic `DelegationCertificate` via `arctic_authority` | Loro, editor-side trust |
| Mint/authorize ledger ops without a single key | `MintAuthority::Threshold` + `verify_arctic_threshold` | a second signing scheme |
| Publish "this dataset exists" | `ContentOp` (+ optional atproto record mirror) | Loro document text, presence |
| Globally order events | QuePaxa consensus | Loro frontiers (per-doc only) |

---

## 6. Conformance checklist (what a reviewer checks)

- [ ] No `DurableStore` implementation other than Loro/JSONL/SQLite exists.
- [ ] No second `did:web` document endpoint outside `dynamic-arctic`; no
      `did:unfer` registry outside `unfer_identity`.
- [ ] Certificate verification happens only in `australVM/arctic_authority`
      (and the `arctic-auth` JIT feature), not re-implemented elsewhere.
- [ ] Presence never reaches durable storage and never gates authority.
- [ ] Heavy content is referenced by CID/magnet; its bytes appear in no CRDT
      document and no atproto record body.
- [ ] Every new cross-machine feature states in its docs which column of the
      decision table it belongs to.

---

## 7. Deliberately delayed (timepiece-only follow-ups)

Per maintainer instruction, `timepiece/` changes are specified but not
executed in this pass:

- `timepiece/AGENTS.md`: add the division rule so Lean-specialist sessions
  do not invent persistence/identity mechanisms in the proof tooling.
- `timepiece/GPU_FEDERATION_PLAN.md`: cite this doc for where layout-claim
  certificates and federation session state live.
