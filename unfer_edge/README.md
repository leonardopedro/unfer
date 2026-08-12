# unfer_edge

Pingora-based security-first edge proxy fronting the `unfer_agent` NDJSON
loop over HTTP (P11.22). Validates every op against `ALLOWED_OPS`
(`UK-4001` on deny), enforces grants/backend policy, and proxies to the
backend (`--backend`, `UNFER_BACKEND` env, or `127.0.0.1:3001`).

Under `--features audit` it also serves the S22 admin console (`admin.rs`),
the audit trail (`audit.rs`), the S26 sensitive-forward latch, blueprint
and `.cell` packaging routes (`gate.rs`, `blueprint.rs`, `cells.rs`), and
the S28 capability RPC (`caprpc.rs` — `/api/cap/{mint,promise,resolve,
revoke,invoke}`).

Most security surface is audit-feature-gated, so the audit-feature CI job
is the one exercising it.