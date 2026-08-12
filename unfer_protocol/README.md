# unfer_protocol

The shared serde contract for the unfer kernel: types, `UK-####` codes, and
repair hints. This is the cross-crate language of the system — `prob_kernel`,
`unfer_ffi`, `unfer_edge`, `unfer_consensus`, and the velysterm agent all
speak it.

- `codes` — the `UK-####` diagnostic catalogue (severity + repair hint).
- `ops` — the shared op-name registry.
- `types` — wire types (model specs, sessions, kernel events, consensus
  transactions).
- `archive` — the deprecated-op archive.

Also hosts `KERNEL_VERSION`, the single source of truth for `uk_version`.