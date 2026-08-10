# Plan R — Privacy-Preserving ReFi Exchange (adapted for unfer/australVM/velysterm)

> Adapted from the 6-phase "Privacy-Preserving ReFi Exchange" plan onto the
> existing three-repo system. The original plan targets a greenfield build
> (RISC Zero UTXO + bare QuePaxa nodes + TLSNotary + PayPal + GNU Taler). This
> version maps every phase onto code that already exists here, reusing the
> **Meerkat/QuePaxa consensus layer**, the **unfer_agent protocol**, the
> **Australia JIT module system**, and the **velysterm client**.
>
> Repos: `unfer/` (kernel + consensus), `australVM/` (JIT module runtime),
> `velysterm/` (editor/agent frontend). They are siblings under `$ROOT` and
> path-depend on each other's working trees.

## Status

- **Phase 1 (ZK ledger core) + Phase 2 (consensus state machine): DONE.**
  The certificate/UTXO ledger (`unfer_consensus::certs`) is implemented and
  testable offline. See [Implemented core](#implemented-core).
- **FFI + agent + edge surface: DONE.** `uk_cert_*` C symbols, the `cert_*`
  NDJSON agent ops, the australVM loopback marshaling, and the edge allowlist
  are all wired and tested. See [Implemented surface](#implemented-surface).
- Phases 3–6 are documented external integrations (zk-TLS, fiat Rails, Taler,
  audit) with concrete mapping but no code yet.

---

## Key adaptation decisions

| Original plan | This project |
|---|---|
| RISC Zero zkVM guest circuit | `unfer_consensus::certs` state-transition engine (transparent core; zkVM is an additive later layer) |
| UTXO: `Hash(amount, pk, blinding)` | `commit_coin` (sha256) producing a `CertId` |
| Sparse Merkle tree of UTXOs | `SparseMerkle` (depth-256 binary SMT) in `certs.rs` |
| Nullifier `Hash(priv, utxo)` | `nullifier_for(coin_id)` — deterministic (transparent); a confidential run uses `Hash(spend_key, commitment)` |
| QuePaxa node network | `unfer_consensus` (already present): `LocalConsensus` + `ConsensusNode` replaying a signed log |
| "Rapid validation rule" | `CertificateLedger::apply_op` — rejects before an op is sequenced |
| zk-TLS oracle | `CertificateOpKind::Mint { source: Option<String> }` provenance field; TLSNotary prover is a client-side integration |
| Payment/escrow web app | `unfer_edge` (Pingora) + `unfer_agent` NDJSON ops |
| GNU Taler Wire-Gateway | New `unfer_taler` adapter crate (out of scope here) |
| FFI surface | New additive `uk_cert_*` symbols (frozen contract permits additive-only) |

---

## Implemented core (Phases 1–2)

### Files
- `unfer_protocol/src/codes.rs` — new UK-7xxx certificate codes.
- `unfer_protocol/src/types.rs` — `CertId`, `Nullifier`, `CoinRef`,
  `CertificateOpKind` (`Mint`/`Transfer`/`Burn`), `CertificateOp`, and the
  `ConsensusTransaction::CertificateOp` variant.
- `unfer_consensus/src/certs.rs` — `SparseMerkle`, `Coin`, `MintAuthority`,
  `CertificateLedger` (the state-transition engine).
- `unfer_consensus/src/node.rs` — `ConsensusNode` carries a `CertificateLedger`,
  applies `CertificateOp`s during `sync()`, exposes `certs()`.
- `unfer_consensus/src/lib.rs` — re-exports the new surface.

### Guarantees enforced by `apply_op`
- **Mint authority**: only the configured `MintAuthority` DID can mint (default
  `None` = minting disabled). UK-7001.
- **Existence**: every transfer/burn input must be an unspent UTXO. UK-7003.
- **Conservation**: `Sum(inputs) == Sum(outputs)` on a transfer. UK-7002.
- **Double-spend**: a nullifier is consumed exactly once. UK-7004.
- **Ownership**: the signer must own every input (cross-checked against stored
  amount). UK-7005.
- **Uniqueness**: an input nullifier and an output commitment may each appear at
  most once within a single op (duplicate-input / duplicate-output guards).
- **Seq hygiene**: reserved UK-7006 for stale/duplicate seq.

### Determinism
Every node that replays the same signed log reaches the identical sparse-Merkle
root (`CertificateLedger::root()`). Two nodes converge; retiring the final UTXO
returns the root to the empty-tree hash.

### Tests (`cargo test -p unfer_consensus`)
`mint_requires_authority`, `mint_is_idempotent_per_commitment`,
`transfer_conserves_value`, `transfer_split_outputs`,
`conservation_violation_rejected`, `double_spend_rejected`,
`duplicate_input_rejected`, `duplicate_output_rejected`,
`owner_mismatch_rejected`, `burn_retires_value`, `root_changes_on_apply`,
`smt_insert_remove_roundtrip`, plus the node-level
`certificate_ledger_roundtrip_via_consensus`,
`five_nodes_converge_on_certificate_root` (5 shared-log validator nodes replay
interleaved mint/transfer/burn/identity ops and converge on one root; a
late-joining node catches up to the same root) and
`invalid_certificate_op_rejected_before_log`, and the fuzzed property
`fuzz_transfers_never_break_conservation_or_double_spend` (multi-input /
multi-output transfer fuzzing: conservation, no-double-spend, uniqueness, and
supply invariants over random op sequences).

## Implemented surface

The ledger is reachable end-to-end through every consumer:

- **FFI** (`unfer_ffi`): `uk_cert_set_authority`, `uk_cert_root`,
  `uk_cert_status`, `uk_cert_mint`, `uk_cert_transfer`, `uk_cert_burn`
  (registered in `EXPECTED_SYMBOLS.txt`). Mutating ops take a single JSON op
  arg and return `0`/`-UK-####`; reads use the buffer protocol. Tested in
  `tests::cert_ffi_*`.
- **australVM JIT** (`safestos/cranelift/src/ecma.rs`): the six `uk_cert_*`
  symbols are marshaled onto the in-process FFI; the mutating four are added to
  `SENSITIVE_BLOCKED_SYMBOLS` so a sensitive-latched caller cannot mint/transfer.
- **velysterm agent** (`kernel_client/src/bin/unfer_agent.rs`):
  `cert_set_authority`, `cert_mint`, `cert_transfer`, `cert_burn`,
  `cert_status`, `cert_root` NDJSON ops drive the in-process `ConsensusNode`
  (signing each op with the actor's keypair). Tested in
  `tests::cert_ledger_roundtrip_via_ops` and `cert_mint_refuses_non_authority`.
- **edge** (`unfer_edge/src/filter.rs`): the `cert_*` ops are added to the
  `ALLOWED_OPS` allowlist.

---

## UK code allocation (7xxx = certificate ledger)

| Code | Name | Meaning |
|---|---|---|
| UK-7001 | CertMintNotAuthorized | signer is not the configured mint authority |
| UK-7002 | CertAmountMismatch | conservation violation on a transfer |
| UK-7003 | CertNonexistentInput | input coin_id is not an unspent certificate |
| UK-7004 | CertDoubleSpend | nullifier already consumed |
| UK-7005 | CertOwnerMismatch | signer is not the owner of every input |
| UK-7006 | CertLedgerSeq | stale/duplicate op sequence |

---

## Full phase-by-phase mapping

### Phase 1 — Core ZK ledger (UTXO state machine) — DONE (transparent core)

Original: RISC Zero guest verifying signature, Merkle membership, nullifier
correctness, `Sum(in)==Sum(out)`.

Adaptation: the equivalent checks are `CertificateLedger` (indicative of the
transparent core). The sparse-Merkle membership + nullifier + conservation
logic is the exact set a RISC-Zero guest would re-prove. Recommended follow-ups
(not yet wired, additive):
1. Add a `risc0`-gated `unfer_cert_zk` crate whose guest program mirrors
   `apply_op` invariants and emits a receipt; the node verifies the receipt when
   the op carries one.
2. Keep the transparent `{amount, owner, blinding}` fields for tooling; the
   confidential path pushes only `commitment` + `nullifier` onto the wire.

### Phase 2 — Decentralized consensus (QuePaxa/Meerkat) — DONE (core)

Original: 5–7 validator nodes, state-transition engine, Merkle updater, public
RPC.

Adaptation: `unfer_consensus` already implements the ordered consensus log
(`LocalConsensus`), the signed transaction format, the onboarding DID registry,
and deterministic replay. The certificate ledger plugs straight in as a new
`ConsensusTransaction` variant, so nodes agree on the certificate root exactly
as they agree on identity/content state. The `network` feature (`rust-quepaxa`,
tokio) is the path to real 5–7 node operation; the in-process engine drives all
tests.

### Phase 3 — UNFCCC oracle bridge (zk-TLS) — DOCUMENTED

Original: TLSNotary proof of the UNFCCC receipt page → mint circuit.

The **public anchor is real**: the UN platform
`https://offset.climateneutralnow.org/vchistory` publishes every voluntary
cancellation as a page `/vchistory/details?orderId=N` carrying a free-form
**"Reason for cancellation"** field (UN states the reason is "provided by the
cancellor"), plus `Reference VC#/YEAR`, `Presented to`, and a
`Start/End serial number` range that pins the exact tonnage. That reason field
is the carrier for the `did:unfer` public key — a user buys and cancels CERs
with their key written into the reason text, giving a public, UN-hosted
key↔certificate binding. The platform is behind Incapsula bot protection, so
server-side scrapers are refused; only a browser-driven TLS session can fetch
it, which is precisely the zk-TLS scenario.

Adaptation: the mint path already exists — `CertificateOpKind::Mint` carries a
`source` provenance string (e.g. `unfccc:vc:<orderId>`). The zk-TLS prover is a
client-side tool that:
1. Connects to `https://unfccc.int` / `offset.climateneutralnow.org`,
   captures the cancellation-record page over TLSNotary.
2. Verifies the "Reason for cancellation" field embeds the user's `did:unfer`
   public key and that the serial-number range `[SN_start, SN_end]` matches the
   cancelled tonnage $m$.
3. Produces a `MintRequest` (additive FFI op) that the mint authority signs and
   submits as a `CertificateOp::Mint` with `source = "unfccc:vc:<orderId>"`.
The `source` field records the backing proof reference for auditability; a
verifier can re-derive the same proof from the public page.

### Phase 4 — Secondary fiat marketplace — DOCUMENTED

Original: PayPal/Stripe escrow web app.

Adaptation: the escrow backend maps to `unfer_edge` (operator-only routes like
`/api/gate/*`, `/api/cap/invoke` already exist) plus the `unfer_agent` NDJSON
loop. The buyer/seller transfer is a `CertificateOp::Transfer` on the consensus
log; the edge listens for the transfer receipt and releases escrow. Wallets live
in `velysterm/mathed` (keys never leave the client). No code in this crate.

### Phase 5 — GNU Taler mint — DOCUMENTED

Original: peg UTXOs to Taler e-coins for retail.

Adaptation: a new `unfer_taler` adapter crate (C/Python/PostgreSQL exchange)
talks to the `unfer_agent` `cert_*` ops. Peg-in = `CertificateOp::Transfer` to
the exchange treasury DID; peg-out = the exchange treasury signs a transfer to
the merchant. The `CertificateLedger` gives the exchange a single deterministic
root to audit against.

### Phase 6 — Security, audit, mainnet — DOCUMENTED

- **Circuit audit**: the invariants in `certs.rs` are the audit surface; a
  property test (proptest) that "no sequence of applied ops can violate
  conservation or double-spend" is the natural first audit deliverable (done:
  `fuzz_transfers_never_break_conservation_or_double_spend`).
- **Testnet**: in-process multi-node convergence is demonstrated by
  `five_nodes_converge_on_certificate_root`; the real-network step is running
  the same `ConsensusNode` on the `network` feature (`rust-quepaxa`, tokio)
  with `MintAuthority::Only(test)` and test-fiat/carbon.
- **Legal**: unchanged from the original plan (VASP, commodity-backed e-coins).
- **Mainnet**: genesis == configuring the real mint authority DID.

---

## Suggested team composition (unchanged from original, mapped to crates)

- **ZK/crypto engineer** → `unfer_consensus::certs` + future `unfer_cert_zk`.
- **Distributed systems engineer** → `unfer_consensus` `network` feature
  (`rust-quepaxa` node operation).
- **Full-stack web3 dev** → `unfer_edge` escrow routes + `velysterm` wallet.
- **FinTech/GNU Taler integrator** → `unfer_taler` adapter.

---

## Immediate next steps (Days 1–14)

1. Re-run `cargo test -p unfer_consensus` (green) as the Phase-1 baseline.
2. ~~Add additive `uk_cert_mint` / `uk_cert_transfer` / `uk_cert_burn` /
   `uk_cert_root` FFI symbols + `cranelift_init` registration + `EXPECTED_SYMBOLS`~~ — **DONE** (see [Implemented surface](#implemented-surface)).
3. ~~Add `cert_*` NDJSON ops to `kernel_client/src/bin/unfer_agent.rs`~~ — **DONE**.
4. ~~Add proptest invariants for conservation/double-spend on `CertificateLedger`~~ — **DONE** (`certs::proptests::fuzz_transfers_never_break_conservation_or_double_spend`).
5. ~~Draft the white-paper math flow UN → QuePaxa → Taler.~~ — **DONE**
   ([`docs/WHITEPAPER_REFI_MATH.md`](WHITEPAPER_REFI_MATH.md)). The UN→QuePaxa→
   Taler flow is written as a precise state-transition sequence, every primitive
   (`commit_coin`, `nullifier_for`, sparse-Merkle root, `apply_mint`/`apply_transfer`)
   tied to the implementation, with a proof outline matching the proptested invariants.