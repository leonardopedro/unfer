# unfer Protocol — Agent Machine Interface

> **Status:** normative for Stage 17+. All kernel clients (Austral modules,
> velysterm UI, AI agents) speak this protocol.

## Transport

NDJSON over stdin/stdout. One JSON object per line.

```
request  →  {"id":"<string>","op":"<string>","params":{...}}
response ←  {"id":"<string>","ok":true,"result":{...},"error":null}
response ←  {"id":"<string>","ok":false,"result":null,"error":{...}}
```

Every request carries a client-chosen `id`; the response echoes it.
Requests are processed sequentially (no pipelining in v1).

## Envelope types

```rust
struct AgentRequest {
    id: String,
    op: String,
    params: serde_json::Value,
}

struct AgentResponse {
    id: String,
    ok: bool,
    result: Option<Value>,
    error: Option<Diagnostic>,
}
```

## Ops

### `version`

Returns the kernel version string.

**Request:**
```json
{"id":"1","op":"version","params":{}}
```
**Response:**
```json
{"id":"1","ok":true,"result":{"version":"0.1.0"},"error":null}
```

### `create_model`

Creates a `Session` from a `ModelSpec`. Returns a numeric `model_id`
used by subsequent ops.

**Request params:** `ModelSpec` (full spec: hamiltonian + prior + solver).

```json
{
  "id":"2",
  "op":"create_model",
  "params":{
    "hamiltonian":{"kind":"builtin","name":"harmonic_chain","params":{"n_modes":1,"omega":1.0}},
    "prior":{"kind":"vacuum"},
    "solver":{"krylov_dim":8,"prune_eps":1e-12,"max_components":null,"restarts":1,"device":{"kind":"cpu"}}
  }
}
```
**Response:**
```json
{"id":"2","ok":true,"result":{"model_id":1},"error":null}
```

### `set_prior`

Replaces the prior state of a model. Resets evolution time to 0.

**Request params:** `{"model_id": <u64>, "prior": <PriorSpec>}`

### `evolve`

Time-evolve the model by `t` seconds.

**Request params:** `{"model_id": <u64>, "t": <f64>}`

**Response result:** `EvolveReport { t, norm, components }`

### `probability`

Query the Born-rule probability of an event.

**Request params:** `{"model_id": <u64>, "event": <EventPredicate>}`

**Response result:** `{"probability": <f64>}`

### `condition`

Condition the state on an event (project + renormalize). Returns the
prior probability of the event.

**Request params:** `{"model_id": <u64>, "event": <EventPredicate>}`

**Response result:** `{"prior_probability": <f64>}`

### `snapshot`

Return the top-k state components by probability mass.

**Request params:** `{"model_id": <u64>, "top_k": <usize>}`

**Response result:** `StateSummary { norm, components, top: [StateEntry] }`

### `save_session`

Serialize the full model state (prior, hamiltonian, solver config, current
time-evolved state) to a portable JSON blob. Restorable via `restore_session`.

**Request params:** `{"model_id": <u64>}`

**Response result:** `{"blob": <SessionBlob JSON>}`

**Error codes:** UK-1004 (bad handle).

### `restore_session`

Reconstruct a model from a previously saved blob. Returns a new `model_id`
(the old handle is not reused).

**Request params:** `{"blob": <SessionBlob JSON>}`

**Response result:** `{"model_id": <u64>}`

**Error codes:** UK-1001 (malformed blob).

### `poll_events`

Read pending kernel events (status changes, error notifications) from the
model's bounded event queue (64 entries max). Non-destructively returns all
currently queued events; oldest events are dropped when the queue overflows.

**Request params:** `{"model_id": <u64>}`

**Response result:** `{"events": [<KernelEvent>, ...]}`

`KernelEvent` shape:
```json
{
  "event_id": <u64>,
  "event_type": "evolve_done" | "condition_applied" | "error" | "subscribe_match",
  "payload": { ... }
}
```

**Error codes:** UK-1004 (bad handle).

### `list_codes`

Dump all UK-#### error codes for self-documentation.

**Request:** `{"id":"9","op":"list_codes","params":{}}`

**Response:**
```json
{"id":"9","ok":true,"result":{"codes":[{"code":1001,"name":"BadJson",...},...]},"error":null}
```

### `close_model`

Free a model session and its event queue. Subsequent ops on the same
`model_id` return UK-1004.

**Request params:** `{"model_id": <u64>}`

**Response result:** `{"ok": true}`

**Error codes:** UK-1004 (bad handle).

### `bayesian_update`

Run Hamiltonian Monte Carlo (HMC) sampling to compute the posterior
distribution over the QFM tomographic state given a set of observations.

**Eligibility:** requires a QFM tomographic model
(`{ "kind": "qfm_tomography", ... }`). Non-QFM models return UK-5000.

**Request params:**

```json
{
  "model_id": 1,
  "observations": [[1.0, 0.0], [0.0, 1.0]],
  "hmc_opts": {
    "leapfrog_steps": 10,
    "step_size": 0.01,
    "n_iterations": 1000,
    "burn_in": 100,
    "seed": 42
  }
}
```

- `model_id` (u64) — model handle from `create_model`.
- `observations` (Vec<Vec<f64>>) — each inner vec is a single observation.
- `hmc_opts` (optional) — HMC hyperparameters. Defaults match
  `HmcOptsSpec::default()`.

**Response result:**

| Field | Type | Description |
|---|---|---|
| `log_posterior` | f64 | HMC log-posterior at the final sample |
| `mean_likelihood` | f64 | Geometric mean of likelihoods; `-1.0` if no observations |
| `image` | Vec<f64> | Phase 5 reconstruction of the representative draw |
| `posterior_mean_image` | Vec<f64> | Karcher mean reconstruction; empty if no post-burn-in samples |
| `n_samples` | usize | Number of samples averaged into `posterior_mean_image` |
| `n_observations` | usize | Number of observations |
| `solve_ms` | u64 | Wall-clock time for HMC + decode |

On success, a `bayesian_updated` event is pushed to the model's event queue.

**Error codes:** UK-1004 (bad handle), UK-1003 (invalid observations),
UK-5000 (non-QFM model).

### `belief_propagation`

Run chain belief propagation for a fast posterior point estimate without
HMC sampling cost. Returns the MAP (marginal mode) point estimate and the
decoded full-resolution image.

**Eligibility:** requires a QFM tomographic model. Non-QFM models return
UK-5000.

**Request params:**

```json
{
  "model_id": 1,
  "observations": [[1.0, 0.0]],
  "opts": {
    "belief_propagation_rounds": 10
  }
}
```

- `model_id` (u64) — model handle from `create_model`.
- `observations` (Vec<Vec<f64>>) — measurement observations.
- `opts` (optional) — BP hyperparameters. Defaults match
  `BeliefPropagationOptsSpec::default()`.

**Response result:**

| Field | Type | Description |
|---|---|---|
| `image` | Vec<f64> | Phase 5 reconstructed image of the MAP |
| `log_posterior` | f64 | Log-posterior at the MAP (up to a constant) |
| `n_observations` | usize | Number of observations |
| `n_sweeps` | usize | Cumulative-product sweeps (always 1) |
| `solve_ms` | u64 | Wall-clock time for BP + decode |

On success, a `belief_propagated` event is pushed to the model's event queue.

**Error codes:** UK-1004, UK-1003, UK-5000 (same as `bayesian_update`).

### `did_create`

Create a new decentralized identifier (DID) with a fresh Ed25519 keypair.
The keypair is held in the agent session for signing subsequent operations.

**Request params:** `{"service_endpoint": <string|null>}` (optional)

**Response result:** `{"did": "<did:string>"}`

### `did_resolve`

Resolve a DID to its document (public key, service endpoint, status).

**Request params:** `{"did": "<did:string>"}`

**Response result:** the DID document (JSON).

**Error codes:** UK-6004 (unknown DID).

### `did_update`

Update the service endpoint of a DID. Requires the keypair created by
`did_create` to be present in the session.

**Request params:** `{"did": "<did:string>", "service_endpoint": <string|null>}`

**Response result:** `{"ok": true}`

**Error codes:** UK-6004 (unknown DID / no keypair).

### `did_revoke`

Revoke a DID. Removes the keypair from the session.

**Request params:** `{"did": "<did:string>"}`

**Response result:** `{"ok": true}`

**Error codes:** UK-6004 (unknown DID / no keypair).

### `content_publish`

Publish a content reference (CID + metadata) to the consensus log, signed
by the given DID's keypair.

**Request params:** `ContentRef` fields plus `"did": "<did:string>"`.

**Response result:** `{"seq": <u64>, "cid": "<string>"}`

**Error codes:** UK-6004 (unknown DID / no keypair), UK-1001 (invalid
ContentRef JSON).

### `content_resolve`

Resolve a CID to its content reference from the consensus state.

**Request params:** `{"cid": "<string>"}`

**Response result:** the `ContentRef` (JSON).

**Error codes:** UK-1001 (content not found).

### `consensus_sync`

Advance the consensus state machine by applying pending transactions.

**Request params:** `{}`

**Response result:** `{"applied": <usize>, "current_seq": <u64>}`

### `consensus_status`

Query the current consensus state without advancing it.

**Request params:** `{}`

**Response result:**
```json
{"applied_seq": <u64>, "current_seq": <u64>, "synced": <bool>}
```

### `cert_set_authority`

Configure the carbon-certificate mint authority (Plan R). Empty `did` disables
minting (the safe default).

**Request params:** `{"did": "<did:unfer:...>"}`

**Response result:** `{"ok": true}`

### `cert_mint`

Issue `amount` carbon certificates to `owner` as `actor` (must be the
configured mint authority). `blinding` is 32-byte hex.

**Request params:**
```json
{"actor": "<did>", "amount": <u64>, "owner": "<did>", "blinding": "<hex32>", "source": "<provenance>"}
```

**Response result:** `{"ok": true, "root": "<hex>", "total_supply": <u64>}`
The new coin_id is `commit_coin(amount, owner, blinding)`.

### `cert_transfer`

Spend `inputs` and create `outputs`, conserving total value. `actor` must own
every input.

**Request params:**
```json
{
  "actor": "<did>",
  "inputs": [{"coin_id": "<hex32>", "amount": <u64>, "owner": "<did>"}],
  "outputs": [{"amount": <u64>, "owner": "<did>"}]
}
```

**Response result:** `{"ok": true, "root": "<hex>", "total_supply": <u64>}`

### `cert_burn`

Retire `inputs` (owner-only), removing their value from circulation.

**Request params:**
```json
{"actor": "<did>", "inputs": [{"coin_id": "<hex32>", "amount": <u64>, "owner": "<did>"}]}
```

**Response result:** `{"ok": true, "root": "<hex>", "total_supply": <u64>}`

### `cert_status`

Query the certificate ledger state without mutating it.

**Request params:** `{}`

**Response result:**
```json
{"root": "<hex>", "unspent_count": <usize>, "total_supply": <u64>}
```

### `cert_root`

Return just the committed sparse-Merkle root.

**Request params:** `{}`

**Response result:** `{"root": "<hex32>"}`

### `observe`

Record an observation against the current model state (side-effecting:
normally queued for approval unless the caller holds an observe annotation).

**Request params:** `{"event": <EventSpec>, "sensitive": <bool|optional>}`

**Response result:** `{"observed": <bool>, "latched": <bool|optional>}`

### `auction_open`

Open an auction lot (Prebid-model: carbon credits or publicity inventory).

**Request params:** `{"market": "carbon"|"publicity", "lot_id": <string>,
"price_per_unit": <u64>, "qty": <u64>}`

**Response result:** `{"lot_id": <string>, "escrow_did": <string>}`

### `auction_bid`

Submit a bid on an open lot (payment escrowed as an e-coin).

**Request params:** `{"lot_id": <string>, "price_per_unit": <u64>,
"qty": <u64>, "coin": <e-coin spec>}`

**Response result:** `{"accepted": <bool>, "seq": <u64>}`

### `auction_close`

Close a lot and settle it to the winning bid (highest `price_per_unit`, ties
break to the earliest `seq`).

**Request params:** `{"lot_id": <string>}`

**Response result:** `{"winner": <string>, "price": <u64>, "settled": <bool>}`

### `auction_report`

Read-only report of the auction ledger state.

**Request params:** `{}`

**Response result:** `{"lots": [...], "settled": <u64>}`

### `logos_compile`

Compile a controlled-natural-language (CNL) sentence with an embedded lexicon
to a CoreIR interaction-net unique normal form (S31).

**Request params:** `{"sentence": <string>}`

**Response result:** `{"unf": <string>, "unf_hash": <hex>, "verified": <bool>}`

### `ode_to_hamiltonian`

Compile an ODE system specification into a Hamiltonian operator.

**Request params:** `{"ode": <spec>}`

**Response result:** `{"hamiltonian": <spec>, "terms": <usize>}`

### `export_html`

Export the current session transcript as standalone HTML.

**Request params:** `{"path": <string|null>}`

**Response result:** `{"bytes": <usize>, "written": <bool>}`

### `export_tex`

Export the current session transcript as LaTeX.

**Request params:** `{"path": <string|null>}`

**Response result:** `{"bytes": <usize>, "written": <bool>}`

## Error codes

| Code  | Name                      | Severity | Description                                                          |
|-------|---------------------------|----------|----------------------------------------------------------------------|
| UK-1001 | BadJson                  | Error    | Input JSON could not be parsed or did not match the expected schema. |
| UK-1002 | UnknownBuiltinModel      | Error    | The requested builtin model name is not recognized by the kernel.     |
| UK-1003 | BadEventPredicate        | Error    | The event predicate is malformed or references an unknown mode.      |
| UK-1004 | BadHandle                | Error    | The referenced model handle is invalid or has been freed.             |
| UK-1005 | BufferTooSmall            | Error    | The caller-provided buffer was too small.                            |
| UK-1006 | SessionLogVersion        | Error    | The session event-log format version is unsupported, or the log is malformed and cannot be replayed. |
| UK-1007 | SessionCompactionOrphaned| Error    | A compaction lock bracket was left open (a crash between start and end); the derived history is unusable until resolved. |
| UK-1008 | SessionCompactionBusy    | Error    | Session compaction refused: the session is not idle (open compaction lock, or a boundary splitting an unanswered action_apply/evolve dependency). |
| UK-1009 | SessionForkRange         | Error    | Session fork refused: the requested log boundary is out of range or falls inside an open compaction bracket. |
| UK-1010 | UnknownOutcome           | Error    | A side-effecting call (e.g. `uk_action_apply`) was interrupted at its durable checkpoint: the in-flight marker is durable but no resolved record exists. The external outcome is UNKNOWN — retry only after manual verification (the kernel refuses to re-run the effect automatically). |
| UK-2001 | GramDegenerate            | Error    | The Krylov Gram matrix is rank-deficient.                            |
| UK-2002 | StateExplosion            | Error    | The state vector exceeded the configured component limit.            |
| UK-2003 | ZeroProbabilityCondition  | Error    | Conditioning on an event with zero prior probability.                |
| UK-2004 | BrstNotConverged         | Error    | The BRST physical-state projection failed to converge.               |
| UK-2005 | CasTermExplosion         | Error    | Symbolic expansion exceeded the term budget.                         |
| UK-3001 | CudaUnavailable          | Error    | A CUDA device was requested but is not available at runtime.         |
| UK-3002 | OutOfMemoryBudget        | Error    | The kernel exceeded its configured memory budget.                    |
| UK-4001 | CallDenied               | Error    | The authorization engine denied the caller permission.               |
| UK-4002 | ActionRequiresApproval   | Error    | The action is a mutation and must be approved by the gatekeeper.     |
| UK-4003 | ActionRejected          | Error    | The queued action was rejected by the gatekeeper.                    |
| UK-4004 | ActionNotFound          | Error    | The referenced action handle does not exist or is not pending.       |
| UK-4005 | ActionAlreadyResolved   | Error    | The action was already approved or rejected.                         |
| UK-4100 | BlueprintInvalid        | Error    | The blueprint payload failed validation.                             |
| UK-4101 | BlueprintNoSession      | Error    | A model session is required but none is open.                        |
| UK-4102 | BlueprintNotFound       | Error    | The referenced blueprint does not exist.                             |
| UK-4200 | AuditInvalid            | Error    | The audit query/entry failed validation.                             |
| UK-4201 | AgentNotFound           | Error    | The referenced agent session does not exist.                         |
| UK-4202 | AgentGrantEscalation    | Error    | The grant set would escalate the caller's capabilities.             |
| UK-4203 | AgentStateInvalid       | Error    | The agent session state is inconsistent.                             |
| UK-4401 | ResourceUnintroduced    | Error    | The resource was not introduced to this session.                     |
| UK-4402 | ResourceAlreadyIntroduced | Error  | The resource was already introduced.                                |
| UK-4403 | ResourceNotFound        | Error    | The referenced resource does not exist.                             |
| UK-4501 | ConsoleOnly             | Error    | The operation is reserved for the operator console.                  |
| UK-4601 | RateLimited             | Error    | The caller exceeded the windowed rate limit (UTC-day meter).         |
| UK-4602 | BudgetExceeded          | Error    | The caller exceeded its metered budget.                             |
| UK-4603 | ToolTimeout             | Error    | The dispatch exceeded its declared cooperative deadline at the loopback guard (the backend keeps running; its late result is discarded). |
| UK-4701 | SensitiveLatched        | Error    | The caller observed sensitive data and is latched until cleared.     |
| UK-4801 | ProofVerifyFailed       | Error    | The Lean4 proof failed verification (strict mode).                   |
| UK-4802 | ProofExportInvalid      | Error    | The lean4export payload was malformed or oversize.                   |
| UK-4803 | LogosCompileFailed      | Error    | The CNL sentence could not be compiled to a UNF.                     |
| UK-4901 | SymbolicEngineUnavailable | Error  | The Cadabra2 subprocess engine was not available.                    |
| UK-4902 | SymbolicExpressionInvalid | Error  | The symbolic expression failed validation.                          |
| UK-5000 | Internal                 | Fatal    | An internal invariant was violated; this is a bug.                    |
| UK-6001 | ConsensusNotReady        | Error    | The consensus state machine is not initialized.                      |
| UK-6002 | DuplicateTransaction     | Error    | A transaction with the same sequence number was already applied.     |
| UK-6003 | InvalidSignature         | Error    | The Ed25519 signature on a transaction failed verification.          |
| UK-6004 | UnknownDid               | Error    | The referenced DID does not exist or has been revoked.               |
| UK-6005 | RelayNotConnected        | Error    | The consensus relay transport is not connected.                      |
| UK-7001 | CertMintNotAuthorized    | Error    | The certificate mint was signed by a non-authority DID.             |
| UK-7002 | CertAmountMismatch       | Error    | Conservation violation: transfer inputs ≠ outputs.                  |
| UK-7003 | CertNonexistentInput     | Error    | A certificate input is not an unspent certificate.                  |
| UK-7004 | CertDoubleSpend          | Error    | A certificate nullifier was already consumed.                       |
| UK-7005 | CertOwnerMismatch        | Error    | The signer is not the owner of every input certificate.             |
| UK-7006 | CertLedgerSeq            | Error    | The certificate op sequence is stale or duplicated.                 |
| UK-7007 | CertOracleRejected       | Error    | A mint/oracle endorsement was rejected.                             |
| UK-7101 | TalerUnknownReserve      | Error    | The referenced Taler reserve does not exist.                        |
| UK-7102 | TalerInsufficientBalance | Error    | The reserve balance is insufficient for the operation.              |
| UK-7103 | TalerUnconfirmedWire     | Error    | The wire transfer has not confirmed.                                |
| UK-7104 | TalerDenomUnsupported    | Error    | The requested denomination is not supported.                        |
| UK-7105 | TalerCoinAlreadyDeposited | Error  | The e-coin was already deposited.                                   |
| UK-7106 | TalerRefreshNotEligible  | Error    | The refresh request is not eligible.                                |
| UK-7107 | TalerUnknownECoin        | Error    | The referenced e-coin does not exist.                               |
| UK-7201 | EscrowUnknown            | Error    | The referenced escrow does not exist.                               |
| UK-7202 | EscrowNotHolding         | Error    | The escrow is not holding the expected asset.                       |
| UK-7203 | EscrowAlreadySettled     | Error    | The escrow was already settled.                                     |
| UK-7301 | AuctionUnknownLot        | Error    | The referenced auction lot does not exist.                          |
| UK-7302 | AuctionLotClosed         | Error    | The auction lot is already closed.                                  |
| UK-7303 | AuctionBidBelowFloor     | Error    | The bid price is below the lot floor.                               |
| UK-7304 | AuctionSelfBid           | Error    | The seller cannot bid on its own lot.                               |
| UK-7305 | AuctionNotSeller         | Error    | The caller is not the lot seller.                                   |
| UK-7306 | AuctionLotExists         | Error    | A lot with this id already exists.                                  |
| UK-7307 | AuctionQtyMismatch       | Error    | The bid quantity does not match the lot units.                      |
| UK-7308 | AuctionNoBids            | Error    | The lot closed with no qualifying bids.                             |

## Diagnostic structure

```rust
struct Diagnostic {
    code: Code,          // e.g. Code(2003)
    name: String,       // e.g. "ZeroProbabilityCondition"
    message: String,    // human-readable detail
    severity: Severity, // info | warning | error | fatal
    hints: Vec<RepairHint>,
    data: Value,       // optional structured payload
}

struct RepairHint {
    kind: HintKind,      // replace_value | set_param | reduce_scope | increase_limit | use_alternative_op
    target: String,      // which field/param to change
    suggestion: String,  // what to change it to
}
```

## Repair-hint semantics

| HintKind           | Meaning                                   | Example target             |
|--------------------|-------------------------------------------|----------------------------|
| `replace_value`    | Replace the value of a field              | `"op"`, `"mode"`           |
| `set_param`        | Set a solver/model parameter              | `"solver.krylov_dim"`      |
| `reduce_scope`     | Reduce the problem size                    | `"solver.krylov_dim"`      |
| `increase_limit`   | Raise a budget/limit                       | `"solver.max_components"`  |
| `use_alternative_op` | Use a different op or model              | `"builtin:harmonic_chain"` |

## Allocating new codes

1. **1xxx** — validation errors (bad input from the caller).
2. **2xxx** — solver errors (numerical failures inside the kernel).
3. **3xxx** — resource errors (CUDA, memory).
4. **4xxx** — authorization errors (module permission denials).
5. **5xxx** — internal invariant violations (bugs).
6. **6xxx** — consensus / identity errors (DID, content, sync).

New codes must be:
- Added to `unfer_protocol/src/codes.rs` `Code` consts + `all()` registry.
- Mapped in `prob_kernel/src/error.rs` `KernelError::to_diagnostic()`.
- Documented in the table above.
