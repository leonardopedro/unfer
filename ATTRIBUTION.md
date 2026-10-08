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
| deepseek-harness (`deepseek-ai/deepseek-harness`) | MIT — verified from the clone at `../deepseek-harness/` (2026-10-07); see `PLAN_HARNESS.md` | agent-harness layout and invariant-gate discipline | `PLAN_HARNESS.md`, `scripts/verify-invariants` |
| typos | Apache-2.0 | checked-index-against-fresh-rebuild discipline | `scripts/doc_index.py` + CI gate |
| cloudflare-os | Apache-2.0 — verified from the clone at `../cloudflare-os/` (2026-10-07); also recorded in `docs/PLAN_cloudflare_os_adaptation.md` | gatekeeper approval queue + "nothing is ambient" resource introduction, reimplemented | `unfer_ffi/src/lib.rs` (`uk_gate_*`, `uk_resource_*`), `unfer_data/src/blueprint.rs`, `unfer_edge/src/admin.rs` |
| zentinel | Apache-2.0 — verified from the clone at `../zentinel/` (2026-10-07) | edge design cues only — no code vendored, zentinel itself not integrated | `unfer_edge/` |
| pattern + pattern-main | two checkouts of the `leonardopedro/pattern` fork (owner layout 2026-10-08): `../pattern/` = branch `rewrite-v3` = **MPL-2.0** (working tree read 2026-10-08; tip 208bf142, relicense commit 2fe18484), `../pattern-main/` = branch `main` = **AGPL-3.0** (fresh single-branch clone read 2026-10-08; tip 50241994). The plan's split (`../pattern` MPL-2.0 / `../pattern-main` AGPL-3.0) matches this layout, with `pattern_memory` and the rev-28 bridge on `rewrite-v3` (see `docs/IMPLEMENTATION_PLAN.md` §P11.19) | session-memory/shared-context design; **copyleft code stays inside `../pattern`, never vendored into unfer** | `../pattern/crates/pattern_unfer` — present in the `../pattern/` checkout (branch `rewrite-v3`); absent from `../pattern-main/` on `main` |
| claurst | GPL-3.0 — verified from the clone at `../claurst/` (2026-10-07) | inspiration only for the sandbox layering; never vendored | `unfer_nixvm/` (design influence) |
| cloud-hypervisor-build | Apache-2.0/BSD (see `docs/IMPLEMENTATION_PLAN.md` §P11.23; sibling **retired 2026-10-07** — GitHub API 404 + SSH not-found, no local copy, external mount gone; licence doc-sourced, unverified) | NixOS-on-a-VM sandbox composed, not edited (one launch-script path fix in place) | `unfer_nixvm/` |
| Azuro | see upstream public documentation | vAMM + NegRisk market framing, plan-stage only | `docs/MATHBOND.md`, `docs/PLAN_REFI_EXCHANGE.md` |
| qm | MIT — verified from the clone at `../qm/` (`github.com/yc-software/qm`, LICENSE read 2026-10-08; the owner cloned it locally, superseding the 2026-10-07 *retired* verdict); see `PLAN_HARNESS.md` for the adapted patterns | harness patterns (durable-by-default, idempotency + leader lease, claim slots, posture classifier) | `PLAN_HARNESS.md` |

## Notes

Note the name collision: `docs/ATTRIBUTION.md` in this repo documents the
**attribution-credits product feature** (Open Badges + Taler), not borrowed
attribution. This file is the latter.

The Cadabra2 and Why3 rows are the load-bearing ones for licensing. Both are
copyleft and both are reached as **external processes**; the Rust binary never
links them, so the repo's Apache-2.0 licence is unaffected. Keep it that way —
reaching for a library API instead of a subprocess would change the answer.
