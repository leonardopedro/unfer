# Deploying `unfer_edge` + `arctic`

C8. The acceptance for this document is that **a fresh machine can boot both
servers from this file alone**, without reading the source.

Per-server operational detail stays in the two runbooks, which are authoritative
for their own process and are maintained with it:

| Server | Runbook |
|---|---|
| `unfer_edge` (gateway) | [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md) |
| `arctic` (threshold authority) | [`../../dynamic-arctic/docs/RUNBOOK.md`](../../dynamic-arctic/docs/RUNBOOK.md) |

This file covers what only makes sense across both: ordering, secrets, the
container, and 24/7 operation.

---

## The two processes, and why there are two

`arctic` mints **threshold delegations**: a credential split across N nodes such
that any t of them can reconstruct it. That split is the entire point — a single
compiles signing key is a single point of failure and a single thing to steal.

`unfer_edge` is the gateway in front of everything else: health, metrics, the
capability RPC chokepoint, and the board/ingest surfaces.

The consequence worth internalising before you deploy: **`arctic` is useless
without its quorum, and it fails in a way that looks like a key problem.** Start
it before the nodes exist and every legitimate delegation is answered
"insufficient shares". The unit ordering exists for this reason.

```
        ┌──────────────┐
        │   clients    │
        └──────┬───────┘
               │
        ┌──────▼───────┐   /healthz /metrics /version
        │  unfer_edge  │   /api/cap/*  (S28 chokepoint)
        │   (gateway)  │   /api/v1/ingest  (C4 normalized messages)
        └──────┬───────┘
               │
        ┌──────▼───────┐   /api/v1/delegate  (threshold cert mint)
        │    arctic    │   /api/v1/messages  (C4 normalized messages)
        │  (authority) │
        └──────┬───────┘
               │  broadcasts each round to every node
     ┌─────────┼─────────┐
     ▼         ▼         ▼
   node 1    node 2    node 3 ... node N
```

Both servers accept the **same** normalized message type at different paths
(`/api/v1/ingest` and `/api/v1/messages`). That is C4: one schema, so a sender
does not branch on which server it reached. Both call one handler function, so
they cannot drift.

---

## Quick start (systemd)

### 1. Build

```sh
git clone <repo> /usr/src/unfer
cd /usr/src/unfer
# CPU-only. The `cuda` feature is additive; leave it off unless the host has a
# driver and a toolkit, or the build fails on a machine that does not.
cargo build --release -p unfer_edge -p unfer_ffi
install -Dm755 target/release/unfer_edge /opt/unfer/bin/unfer_edge
```

`arctic` builds from its own repository:

```sh
git clone <repo> /usr/src/arctic
cd /usr/src/arctic
cargo build --release
install -Dm755 target/release/arctic /opt/arctic/bin/arctic
```

### 2. Service users

Both run unprivileged, and neither needs a home directory or a shell.

```sh
useradd --system --no-create-home --shell /usr/sbin/nologin unfer
useradd --system --no-create-home --shell /usr/sbin/nologin arctic
install -d -o unfer -g unfer -m 0750 /var/lib/unfer
install -d -o arctic -g arctic -m 0750 /var/lib/arctic
```

### 3. Configuration

```sh
install -d -o root -g root -m 0755 /etc/unfer /etc/arctic
install -o root -g root -m 0600 deploy/env/edge.env.example   /etc/unfer/edge.env
install -o root -g root -m 0600 deploy/env/arctic.env.example /etc/arctic/arctic.env
$EDITOR /etc/unfer/edge.env /etc/arctic/arctic.env
```

Both `.env.example` files are **fully commented out**. That is deliberate and is
the "omit to disable" convention: every key is optional, `unfer_edge` and `arctic`
both start with none of them set, and `GET /version` reports which configuration
layer won for each key. An unset key is a supported configuration, not a missing
one.

Now write the JSON config each unit points at:

```sh
/opt/unfer/bin/unfer_edge init --config /etc/unfer/edge.json
$EDITOR /etc/unfer/edge.json
```

### 4. Units

```sh
install -Dm644 deploy/systemd/unfer-edge.service       /etc/systemd/system/unfer-edge.service
install -Dm644 deploy/systemd/arctic.service           /etc/systemd/system/arctic.service
install -Dm644 deploy/systemd/unfer-nodes.target       /etc/systemd/system/unfer-nodes.target
systemctl daemon-reload
systemctl enable --now unfer-edge
systemctl enable --now arctic
```

`arctic` is ordered `After=unfer-nodes.target`. If your nodes run somewhere else,
drop that line and say so in your own unit rather than leaving a dependency on a
target that does not exist — an `After=` on a non-existent unit orders against
nothing while looking like it orders against something.

### 5. Verify

```sh
systemctl status unfer-edge arctic --no-pager
curl -fsS localhost:3000/healthz
curl -fsS localhost:3000/version | jq .    # shows effective config AND provenance
```

`/version` is the one to read when a port is not what you expected: it reports
not just the value but **which of the four configuration layers produced it**.

```json
{
  "service": "unfer_edge",
  "config": { "listen": "0.0.0.0:3000" },
  "provenance": { "layers": { "listen": "env", "backend": "default" } }
}
```

---

## Environment variable reference

Every variable below is optional. Defaults are in the runbooks.

### `unfer_edge`

| Variable | Default | Omit to… | Notes |
|---|---|---|---|
| `UNFER_LISTEN` | `0.0.0.0:3000` | bind `0.0.0.0:3000` | Set `127.0.0.1` when a reverse proxy fronts it |
| `UNFER_BACKEND` | `127.0.0.1:3001` | proxy to `127.0.0.1:3001` | With no reachable upstream, non-self-service routes 502 |
| `UNFER_ADMIN_PRINCIPAL` | `operator` | run the S22 console as `operator` | **Minted once at startup.** Only unset deliberately |
| `UNFER_PRESETS_DIR` | *(none)* | use inline grants only | Where the `GrantSet` role roster is discovered |
| `RUST_LOG` | *(none)* | default filtering | Not a config key; the loader ignores unknown names |

### `arctic`

| Variable | Default | Omit to… | Notes |
|---|---|---|---|
| `ARCTIC_BIND_ADDR` | `0.0.0.0:3000` | bind the default | Plain HTTP — no TLS of its own |
| `ARCTIC_DOMAIN` | `authority.yourdomain.com` | use the placeholder | **Do not**: this mints a DID for a domain you do not control |
| `ARCTIC_THRESHOLD` | `3` | require 3 shares | Above `TOTAL_NODES` every delegation is unfillable; `1` defeats the split |
| `ARCTIC_TOTAL_NODES` | `7` | 7 nodes | Must exceed `THRESHOLD` |

---

## Secrets

**Never in a unit file, never in an `.env` file on disk, never in git.**

A value in `Environment=` or a world-readable env file is in the process table,
in `systemctl show`, and in every core dump. None of those are access-controlled
the way the vault is, and all three are collected routinely by monitoring.

Secrets go through the S27 credential vault (`uk_secret_put` / `uk_secret_get` /
`uk_secret_revoke`), which is grant-gated and encrypted at rest through the S15
`KeyRing`. `uk_snapshot` and `uk_blueprint_export` **refuse** to package a live
secret, so a snapshot cannot become the leak path.

The one credential these two servers want at boot is the *master public key*, and
that is not secret — it is a public key. What must never appear is the **signing
key** corresponding to it.

For a first deployment the honest sequence is:

1. Generate the signing key on the host, into the vault. Not in a shell history,
   not in a file you will `tar` up later.
2. Put only the **public** half in `/etc/arctic/arctic.env`.
3. `chmod 600`, owned by `root`, readable by the service user.

---

## Container

`deploy/Dockerfile` builds both servers from the workspace's own `Cargo.lock`.

> **Not built or run.** No container runtime is available in the environment this
> was written in, so the Dockerfile has never been executed. Treat it as reviewed
> source, not as a tested artifact — the first `docker build` may well need
> fixes. The systemd path above *is* the supported one, and its steps are the ones
> that were checked against the built binaries.
>
> The `HEALTHCHECK` is `unfer_edge --help` rather than a `/healthz` request. That
> is deliberate and it is weaker than it looks: it verifies the binary starts and
> that `libunfer_ffi.so` resolves, which is the most common container failure, but
> it does **not** verify the listener is accepting connections. Wiring the real
> probe needs `curl` (or a busybox) in the image, which is a weight/scope
> decision rather than something to decide silently in a Dockerfile.

It is a plain `debian:bookworm-slim` image rather than a Nix one because the
nixpkgs pin this repo needs (CUDA 12.6 from `nixpkgs-unstable`) is a large
download for an image that, deployed to a normal host, runs CPU-only.

If you want the flake instead — `nix build .#unfer-ffi` produces a reproducible
store path from the same lockfile — use that output as the `COPY` source and skip
the cargo stages.

```sh
docker build -f deploy/Dockerfile -t unfer:local .
docker run --rm -p 3000:3000 unfer:local
```

Two things the image deliberately does **not** do:

- It does not `ENV` a secret. There is no default credential baked into a layer,
  where it would outlive the container in the image cache.
- It does not run as root. `USER unfer` is set, so a container escape is not
  immediately root on the host.

Volumes: `/var/lib/unfer` and `/var/lib/arctic` are the only paths that must
persist. Neither holds secrets (see above), so a volume is cache, not key
material.

---

## 24/7 operation

Both units set `Restart=on-failure` with `RestartSec`, `KillSignal=SIGTERM` and a
20-second `TimeoutStopSec`. `SIGTERM` rather than the default `SIGKILL` matters:
it lets an in-flight `/api/v1/ingest` finish, and killing it mid-response gives a
sender no answer at all, which they will retry.

Both set `StartLimitBurst=3` over 60 seconds. This is the crash-loop guard, and
it is the reason there is no `ExecStartPre` probe: a malformed `--config` aborts
startup by design, and without the burst cap that single typo becomes an infinite
restart loop that hides the error message you needed to see.

### What to alert on

| Signal | Means |
|---|---|
| `unfer_edge` `/healthz` failing | process up, but not serving. Dependency-free on purpose, so this is unambiguous |
| `arctic` returning "insufficient shares" to valid requests | the quorum is not reachable — **not** a key problem |
| non-zero `unknown_sources` on `/api/v1/ingest` acks | a webhook is posting under a source name nobody registered |
| `truncated: true` on an ingest ack | the request body exceeded the envelope cap and was refused whole |

### Rolling a restart

`arctic` cannot be rolled with zero downtime: a threshold authority mid-rotation
cannot answer delegations. Restart the nodes first, confirm the quorum, then the
authority. `unfer_edge` has no such constraint.

---

## Related

- [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md) — `unfer_edge` operational detail
- [`../../dynamic-arctic/docs/RUNBOOK.md`](../../dynamic-arctic/docs/RUNBOOK.md) — `arctic`
- [`../docs/PROTOCOL.md`](../docs/PROTOCOL.md) — the wire contract both servers speak