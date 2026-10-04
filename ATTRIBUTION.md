# Attribution

Borrowed patterns and components that concern *this* repo. The authoritative,
cross-repo table — including what was deliberately **not** adopted — is
[`../ATTRIBUTION.md`](../ATTRIBUTION.md).

| source | licence | what was adapted | where it landed |
|---|---|---|---|
| Cadabra2 (`cadabra2-cli`) | GPL-3.0 | field-theory CAS, subprocess only — never linked | `prob_kernel/src/symbolic.rs`, `docs/*.cdb` |
| Why3 + alt-ergo | LGPL | proof and extraction, subprocess only | `prob_kernel/src/whyml.rs` |
| nanoda_lib | see upstream | independent re-verification of exported Lean proofs | `prob_kernel/src/verify.rs` |
| lean4export (official leanprover/lean4export) | Apache-2.0 | proof export format; the legacy exporter emits a format nanoda rejects | `prob_kernel/tests/fixtures/*.ndjson` |
| dynamic-arctic (sibling, path dep) | MIT | Arctic/Shine threshold Schnorr for `MintAuthority::Threshold` | `unfer_consensus` (with `default-features = false`) |
| deepseek-harness | see PLAN_HARNESS.md | agent-harness layout and invariant-gate discipline | `PLAN_HARNESS.md`, `scripts/verify-invariants` |
| typos | Apache-2.0 | checked-index-against-fresh-rebuild discipline | `scripts/doc_index.py` + CI gate |

## Notes

Note the name collision: `docs/ATTRIBUTION.md` in this repo documents the
**attribution-credits product feature** (Open Badges + Taler), not borrowed
attribution. This file is the latter.

The Cadabra2 and Why3 rows are the load-bearing ones for licensing. Both are
copyleft and both are reached as **external processes**; the Rust binary never
links them, so the repo's Apache-2.0 licence is unaffected. Keep it that way —
reaching for a library API instead of a subprocess would change the answer.
