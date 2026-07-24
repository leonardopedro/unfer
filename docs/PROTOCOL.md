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

## Error codes

| Code  | Name                      | Severity | Description                                                          |
|-------|---------------------------|----------|----------------------------------------------------------------------|
| UK-1001 | BadJson                  | Error    | Input JSON could not be parsed or did not match the expected schema. |
| UK-1002 | UnknownBuiltinModel      | Error    | The requested builtin model name is not recognized by the kernel.     |
| UK-1003 | BadEventPredicate        | Error    | The event predicate is malformed or references an unknown mode.      |
| UK-1004 | BadHandle                | Error    | The referenced model handle is invalid or has been freed.             |
| UK-1005 | BufferTooSmall            | Error    | The caller-provided buffer was too small.                            |
| UK-2001 | GramDegenerate            | Error    | The Krylov Gram matrix is rank-deficient.                            |
| UK-2002 | StateExplosion            | Error    | The state vector exceeded the configured component limit.            |
| UK-2003 | ZeroProbabilityCondition  | Error    | Conditioning on an event with zero prior probability.                |
| UK-2004 | BrstNotConverged         | Error    | The BRST physical-state projection failed to converge.               |
| UK-2005 | CasTermExplosion         | Error    | Symbolic expansion exceeded the term budget.                         |
| UK-3001 | CudaUnavailable          | Error    | A CUDA device was requested but is not available at runtime.         |
| UK-3002 | OutOfMemoryBudget        | Error    | The kernel exceeded its configured memory budget.                    |
| UK-4001 | CallDenied               | Error    | The authorization engine denied the caller permission.               |
| UK-5000 | Internal                 | Fatal    | An internal invariant was violated; this is a bug.                    |
| UK-6001 | ConsensusNotReady        | Error    | The consensus state machine is not initialized.                      |
| UK-6002 | DuplicateTransaction     | Error    | A transaction with the same sequence number was already applied.     |
| UK-6003 | InvalidSignature         | Error    | The Ed25519 signature on a transaction failed verification.          |
| UK-6004 | UnknownDid               | Error    | The referenced DID does not exist or has been revoked.               |
| UK-6005 | RelayNotConnected        | Error    | The consensus relay transport is not connected.                      |

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
