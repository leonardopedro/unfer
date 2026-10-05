# unfer servers — runbook

Operational surface for the two long-running processes in this repo: the
`unfer_edge` gateway (Pingora) and the `unfer_agent` NDJSON agent. Covers start,
verify, configuration precedence, rotate, and backup.

`dynamic-arctic` has a sibling runbook at `../dynamic-arctic/docs/RUNBOOK.md` and
the two are deliberately the same shape: same precedence ladder, same
"startup prints where each value came from" rule, same division between "there is
nothing to back up" and "keep the configuration". One dialect, two servers.

---

## Start

### `unfer_edge` — the gateway

```sh
cargo run -p unfer_edge                                   # 0.0.0.0:3000 → 127.0.0.1:3001
cargo run -p unfer_edge -- init                           # write ./unfer_edge.json
cargo run -p unfer_edge -- init --config /etc/unfer/edge.json
cargo run -p unfer_edge -- --config /etc/unfer/edge.json --listen 127.0.0.1:8080
```

`init` writes a **strictly valid JSON** starter file with the documentation
carried in `"//"` keys. Redirect it wherever you keep deployment config. The
output is deterministic — the same answers always produce the same bytes — so it
can be diffed in review rather than eyeballed.

It is JSON and not JSON-with-comment-lines on purpose: a `// comment` on its own
line is not valid JSON, so a wizard that emitted one would write a file the
server cannot read, and report success while doing it.

### `unfer_agent` — the NDJSON agent

```sh
unfer_agent < requests.ndjson > responses.ndjson
```

`unfer_agent` is a **pipe protocol**, not a server: it reads one JSON request per
line on stdin and writes one JSON response per line on stdout. That is the
degenerate single-capability mode the S28 design requires, and it needs **no
configuration and no flags** — the binary has no argument parser at all. An agent
that needs a socket puts one in front of it; that front door is `unfer_edge`.

```sh
# drive it by hand: one request per line
echo '{"id":"1","op":"version"}' | unfer_agent
```

So the two-process arrangement in this repo is: `unfer_agent` on a pipe, and
`unfer_edge` in front of it speaking HTTP. `unfer_edge`'s `backend` setting points
at whatever HTTP wrapper is hosting the agent — it is **not** a flag on
`unfer_agent` itself, and there is no `unfer_agent --listen`.

---

## Endpoints

`unfer_edge`:

| endpoint | method | purpose |
|---|---|---|
| `/healthz` | GET | Liveness. Answers `{"status":"ok"}`. |
| `/version` | GET | Build identity **and the effective configuration with the layer each key came from**. |
| `/metrics` | GET | Counters; add `?format=prometheus` for the text exposition. |
| `/api/cap/*` | POST | Capability RPC (requires a minted capability). |
| everything else | — | Forwarded upstream to `unfer_agent`. |

`unfer_agent` (via `unfer_edge` or directly):

| endpoint | method | purpose |
|---|---|---|
| `/healthz` | GET | Liveness. |
| `/version` | GET | Build identity. |

### `/healthz`

```sh
curl -sf http://localhost:3000/healthz     # {"status":"ok"}
```

Reads no backend state. A probe that depends on the thing it is probing stops
answering exactly when it is most needed, so `/healthz` stays useful while the
backend is down or misconfigured. Use `/version` for readiness.

### `/version`

```sh
curl -s http://localhost:3000/version
```

```json
{
  "service": "unfer_edge",
  "version": "0.1.0",
  "config": {
    "listen": "0.0.0.0:8080",
    "backend": "127.0.0.1:3001",
    "layers": { "listen": "file", "backend": "env" }
  }
}
```

The `layers` map is the point. `/healthz` says the process is alive and `version`
says which build it is; neither says **which of four configuration layers
actually won**, which is the question an operator has when a port is not what they
expected. Reading it here turns a precedence order from something you have to
trust into something you can see.

`/healthz` and `/version` are GET-only. A known self-service path reached with
the wrong method answers **405** with an `Allow: GET` header rather than being
forwarded upstream — forwarding `POST /healthz` would return a confusing upstream
404 instead of a client error.

---

## Configuration

`unfer_edge` precedence, lowest to highest:

1. built-in defaults
2. configuration file (`--config`)
3. environment
4. command-line flags

| key | env | default |
|---|---|---|
| `listen` | `UNFER_LISTEN` | `0.0.0.0:3000` |
| `backend` | `UNFER_BACKEND` | `127.0.0.1:3001` |

Values are read once at startup; there is no hot reload.

```sh
unfer_edge --config /etc/unfer/edge.json
unfer_edge --config ./edge.json --listen 127.0.0.1:8080
unfer_edge --listen=127.0.0.1:8080
unfer_edge init
unfer_edge --help
```

Both `--flag value` and `--flag=value` are accepted, because an operator who types
one of them should not get an error from the other.

**A malformed value falls through to the layer below instead of aborting startup,
and `GET /version` shows that it did.** A typo in an env var should not take the
gateway down; and a value that was silently discarded is worse than one that was
never offered, because it looks like it took effect. A `--config` path that does
not exist, or is not valid JSON, *does* abort startup with a message naming the
file — a config layer that is silently ignored is worse than one that is absent.

Addresses are validated as `host:port` before Pingora sees them. A value that
passes validation and then fails inside `add_tcp` produces a startup panic about
the listener rather than about the setting, and the operator has no way to tell
which layer produced the bad address.

### Runtime admin config is a different thing

The S22 admin console (`--features audit`, `UNFER_ADMIN_PRINCIPAL`) mutates
**runtime** state over the loopback. `grants`, `auth`, `storage` and `backend` are
**hard keys**: never patchable, and a refusal leaves the soft config
byte-identical. The startup config above is a separate surface and must not widen
that refuse list — `unfer_edge`'s config module keeps `HARD_PATCH_KEYS` as data so
adding a startup key forces the question to be asked.

---

## Rotate and revoke

**Secrets do not live in configuration.** They go through `uk_secret_put` /
`uk_secret_get` / `uk_secret_revoke` (the S27 credential vault): opaque,
grant-gated, encrypted at rest under the S15 `KeyRing`, and never serialized into
a `SessionBlob` snapshot or a `.cell` blueprint. A secret in an env var or a JSON
config file is a secret in a process listing and in a backup.

Revoking a capability is immediate and does not require a restart: `uk_gate_approve`
re-checks a returned stub against the original caller, and revoked ids are refused.
Sensitive observations latch (S26) until an operator clears them through the S22
admin seam — that one *is* a deliberate operator action, not an automatic
timeout.

The admin principal is minted **once** at startup from `UNFER_ADMIN_PRINCIPAL`
(default `operator`). Changing it is a restart.

---

## Backup

**Nothing runtime needs backing up.** `unfer_edge` holds no durable state; it is a
proxy. A backup procedure for it would be a sign that something had acquired state
it should not have — the same reasoning `dynamic-arctic`'s runbook gives for its
authority.

What *is* worth keeping:

| artefact | why |
|---|---|
| the config file | deployment configuration: listen address, backend address |
| `unfer_data::release` manifests | byte→sha256 map of deployable artefacts (S24) |
| `qfm_text_runs/*/metrics.ndjson` | pinned experimental evidence cited by tests — deliberately tracked, unlike other build output |
| `logs/heavy_tests_*.log` | the cited all-green evidence log — whitelisted in `.gitignore` on purpose |

Run `scripts/tree-size` to see what is deliberately kept versus what is build
detritus. It fails if a build directory is tracked, and it refuses to report
"clean" when it inspected zero repositories — a gate that finds nothing must not
read as a repo with nothing wrong.

---

## Verify

```sh
cargo test -p unfer_edge                                  # config + routing table
cargo test --workspace --exclude fock_sirk                # the kernel surface
bash scripts/verify-invariants                            # H1 maintenance gate
bash scripts/tree-size ; bash scripts/check-book-sync ; bash scripts/check-status
```

The routing table (`route`) and the config resolver are pure functions, so the
decision logic — which path, which method, which status, which layer wins — is
tested without binding a port or constructing a Pingora `Session`. That matters:
`request_filter` takes a `Session`, which cannot be built without a real
connection, so testing the handler directly would mean testing nothing.

`scripts/check-book-sync` needs `timepiece` checked out as a sibling; both repos'
CI run it, because `timepiece` is the source of record for `book.tex`/`ODE.tex` and
`unfer` carries a synced copy.

---

## Related

- [`../deploy/DEPLOY.md`](../deploy/DEPLOY.md) — **booting `unfer_edge` +
  `arctic` together on a fresh machine**: systemd units, the full env-var
  reference with defaults, secret handling, the container, and 24/7 operation.
  Start here for a deployment; this file for one process.
- `docs/PROTOCOL.md` — the NDJSON agent protocol and the op registry.
- `docs/DEPLOYMENTS.md` — how the layers stack (gateway, VM, Nix packaging).
- `docs/ARCHITECTURE.md` — the crate map.
- `../dynamic-arctic/docs/RUNBOOK.md` — the sibling server, same structure.