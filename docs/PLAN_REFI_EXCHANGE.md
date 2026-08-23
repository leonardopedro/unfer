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
- Phases 3–6 are external integrations (zk-TLS, fiat Rails, Taler, audit).
  Phase 3's shared contract (`MintRequest` + `uk_cert_mint_request`, UK-7007)
  is implemented; the TLSNotary prover itself is client-side and out of scope.
  Phase 5's exchange adapter (`unfer_taler`, UK-7101..7107) is implemented and
  proptested.   Phase 6's testnet is implemented (`unfer_consensus::net`, a
  2-live-of-3 mTLS rust-quepaxa cluster replaying into `ConsensusNode`, with
  durable node state and full-cluster restart via `NetworkCluster::resume`).
  Phase 4's escrow state machine is implemented and on-chain-tested
  (`unfer_consensus::escrow`, UK-7201..7203, including a live-testnet escrow
  that replays onto every node's certificate root); the PayPal/Stripe web-app
  coating over it remains a documented mapping.
- **Unified auction (Prebid-model): DONE.** The deterministic clearing engine
  (`unfer_consensus::auction`, UK-7301..7308) and the operator-side settlement
  (`unfer_taler::auction::AuctionService`) serve both markets — carbon-credit
  lots (escrowed credits delivered to the winner) and publicity inventory
  (AdSense alternative, payment-only). Wired end-to-end: FFI `uk_auction_*`,
  australVM loopback dispatch, and velysterm `auction_*` NDJSON ops.
- **Math catastrophe bond (SPV with nanoda trigger): DONE.** The deterministic
  math-bond ledger (`unfer_consensus::mathbond`, UK-7401..7407) implements a
  Special Purpose Vehicle whose trigger is a purely mathematical proof verified
  by nanoda (`prob_kernel::verify::verify_export`) running deterministically
  inside `apply_op`. The sponsor locks collateral (e-coins), investors buy the
  bond for a coupon, and if nanoda verifies a Lean4-export proof, the collateral
  is paid out as a bounty to the researcher plus a catastrophe payment to the
  sponsor. No human oracle, no external dependency. Bond probability trading
  uses the vAMM + NegRisk market below.
- **Math bond probability market (vAMM + NegRisk): DONE.** The deterministic
  market engine (`unfer_consensus::mathbond_market`, UK-7411..7419) prices
  trigger probabilities via a constant-product vAMM inspired by Azuro's
  Liquidity Tree: LPs deposit e-coins into a singleton pool, and the protocol
  mathematically prices the odds without needing a direct buyer for every seller.
  The NegRisk CTF Adapter allows mutually-exclusive conditional outcomes (e.g.
  "triggered by 2025" vs "triggered by 2026" vs "never") to share a single pool,
  preventing liquidity fragmentation. When one outcome resolves, the others become
  worthless. Settlement through the certificate ledger (Taler e-coins).

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
| GNU Taler Wire-Gateway | `unfer_taler` adapter crate: reserves, two-phase wire gateway, denominations, fiat↔e-coin conservation audit (UK-7101..7107) |
| FFI surface | New additive `uk_cert_*` symbols (frozen contract permits additive-only) |
| Ethereum SPV smart contract | `unfer_consensus::mathbond` (`MathBondLedger`) — deterministic math-bond state machine with nanoda trigger engine |
| Gnosis CTF / Polymarket | `unfer_consensus::mathbond_market` (`MarketLedger`) — vAMM + NegRisk probability market for trigger odds |
| RISC Zero zkVM (trigger) | `prob_kernel::verify::verify_export` (nanoda_lib) — deterministic Lean4-export verification inside the consensus node |
| Human oracle (Clay Institute) | None — the trigger engine is purely mathematical, no human judgment required |

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
  `uk_cert_status`, `uk_cert_mint`, `uk_cert_mint_request`, `uk_cert_transfer`,
  `uk_cert_burn` (registered in `EXPECTED_SYMBOLS.txt`). Mutating ops take a
  single JSON op arg and return `0`/`-UK-####`; reads use the buffer protocol.
  Tested in `tests::cert_ffi_*`.
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
- **Unified auction** (`unfer_consensus::auction` + `unfer_taler::auction`):
  `uk_auction_open/bid/close/report` FFI; australVM `ecma.rs` dispatch +
  `SENSITIVE_BLOCKED_SYMBOLS`; velysterm `auction_*` NDJSON ops through the
  in-process `ConsensusNode`. Tested in `unfer_consensus::auction`,
  `unfer_taler::auction`, and `unfer_ffi::tests::auction_ffi_*`.

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
| UK-7007 | CertOracleRejected | mint `source` is not a valid `unfccc:vc:<orderId>` reference |
| UK-7101 | TalerUnknownReserve | reserve id is not known to the exchange |
| UK-7102 | TalerInsufficientBalance | reserve/merchant balance shortfall on withdraw/peg-out |
| UK-7103 | TalerUnconfirmedWire | peg-in/peg-out references an unconfirmed wire transfer |
| UK-7104 | TalerDenomUnsupported | no live denomination matches the requested e-coin value |
| UK-7105 | TalerCoinAlreadyDeposited | double deposit of the same e-coin refused |
| UK-7106 | TalerRefreshNotEligible | refresh only after the denomination has expired |
| UK-7107 | TalerUnknownECoin | deposit of an e-coin this exchange never minted |
| UK-7301 | AuctionUnknownLot | the auction lot_id was never opened (or is closed) |
| UK-7302 | AuctionLotClosed | bid/close references an already-closed lot |
| UK-7303 | AuctionBidBelowFloor | bid is below the lot's floor price |
| UK-7304 | AuctionSelfBid | the seller bid on their own lot |
| UK-7305 | AuctionNotSeller | only the lot's seller may open/close |
| UK-7306 | AuctionLotExists | duplicate lot open |
| UK-7307 | AuctionQtyMismatch | bid quantity exceeds the carbon lot amount (or payment e-coin face value ≠ bid total) |
| UK-7308 | AuctionNoBids | a close landed with no eligible bids (no winner) |
| UK-7401 | MathBondUnknown | the referenced math bond id does not exist |
| UK-7402 | MathBondWrongState | the bond is not in the expected state |
| UK-7403 | MathBondNotResearcher | the submitter is not the designated researcher |
| UK-7404 | MathBondProofRejected | nanoda rejected the proof (trigger did not fire) |
| UK-7405 | MathBondOverfunded | investment exceeds the bond's remaining capacity |
| UK-7406 | MathBondProofOversize | proof payload exceeds the bond's max export size |
| UK-7407 | MathBondAlreadyTriggered | the bond has already been triggered |
| UK-7411 | MarketUnknownPool | the referenced pool id does not exist |
| UK-7412 | MarketPoolResolved | the pool is already resolved |
| UK-7413 | MarketUnknownOutcome | the outcome id is not a member of this pool |
| UK-7414 | MarketInsufficientTokens | insufficient outcome tokens to sell |
| UK-7415 | MarketInsufficientShares | insufficient LP shares to withdraw |
| UK-7416 | MarketNoLiquidity | the pool has no liquidity |
| UK-7417 | MarketPriceUnderflow | NegRisk: an outcome's price would go negative (also: malformed pool params — duplicate outcome ids, missing terminal "never" outcome, `fee_bps` > 10000) |
| UK-7418 | MarketPoolExists | the pool already exists for this bond |
| UK-7419 | MarketNotResolved | the pool is not resolved — nothing to claim (or the bond has neither triggered nor matured) |

---

## Unified auction (Prebid-model)

The deterministic auction engine lives in `unfer_consensus::auction`
(`AuctionLedger`); settlement lives in `unfer_taler::auction` (`AuctionService`).
`AuctionOp`s (Open/Bid/Close) are a `ConsensusTransaction` variant applied by
`ConsensusNode::sync`, exactly like `CertificateOp`s. The clearing rule is a pure
function of the recorded bids — highest `price_per_unit` wins, ties break to the
earliest `seq` — so every node replays the same log and converges on the same
winner.

Two markets, one mechanism (mirroring Prebid.org's unified auction):

- **Carbon credits**: a seller opens a lot and escrows the credit certificate
  into a deterministic lot DID. Bidders escrow a payment e-coin of exactly the
  bid total (Taler denomination model). On close the winner's payment releases
  to the seller, the losers' payments refund, and the escrowed credits transfer
  to the winner. An unsold lot refunds every payment and returns the credits.
- **Publicity inventory** (AdSense alternative): a publisher opens a slot; the
  winner pays the publisher. No ledger asset is delivered.

All escrow DIDs/keys are derived deterministically from `(operator pubkey,
lot_id, party, coin)` so only the marketplace operator can settle. The auction
never creates or destroys value — `total_supply` is conserved.

Surface: `uk_auction_open/bid/close/report` (C ABI), registered in
`EXPECTED_SYMBOLS.txt`, the generated C header, australVM `UNFER_SYMBOLS` +
`ecma.rs` dispatch + `SENSITIVE_BLOCKED_SYMBOLS`. `uk_auction_close` mutates and
writes, so its marshaling is a single fixed-buffer call (never probe-then-copy).

---

## Math catastrophe bond (SPV with nanoda trigger)

The deterministic math-bond ledger lives in `unfer_consensus::mathbond`
(`MathBondLedger`). `MathBondOp`s (Issue/Invest/SubmitProof/Settle) are a
`ConsensusTransaction` variant applied by `ConsensusNode::sync`, exactly like
`CertificateOp`s and `AuctionOp`s.

The trigger engine is `prob_kernel::verify::verify_export` (nanoda_lib) running
deterministically inside `apply_op` — no human oracle, no external dependency,
no zkVM. The sponsor locks collateral (e-coins), investors buy the bond for a
coupon, and if nanoda verifies a Lean4-export proof of the specified theorem,
the collateral is paid out as a bounty to the researcher plus a catastrophe
payment to the sponsor. If the proof never arrives before maturity, investors
recover their principal plus coupon.

Lifecycle: `Issue → Invest → { SubmitProof (nanoda) → Triggered | Mature →
Matured } → Settle`. A `Mature` op records that the consensus log reached
`maturity_seq` without a trigger (the seq check is the enforcement); `Settle`
is refused while the bond is live — only a `Triggered` (payout) or `Matured`
(refund) bond finalizes. The bond id commits the full issue parameters
(trigger, sponsor, principal, coupon, maturity, researcher), so two bonds with
the same theorem but different terms do not collide.

Bond probability trading uses the vAMM + NegRisk market below.

**Settlement** (`unfer_taler::bondmarket::BondMarketService`, the analogue of
`AuctionService`): the operator rows every e-coin with ordinary, conserving
`CertificateOp`s into/out of deterministic DIDs derived from its master key
(bond collateral DID, per-investor investment DIDs, per-pool cash DID). On
**trigger**, the invested e-coins are paid out as the **researcher bounty** and
the sponsor keeps its collateral as the **catastrophe payment** (investors are
wiped out — the point of a cat bond). On **maturity**, every investor recovers
principal plus a `coupon_rate_bps` coupon (paid from the collateral, which
keeps the remainder). All produced ops carry a single global seq equal to their
consensus-log position, so replaying `ops()` into a `ConsensusNode` converges
on the identical certificate root and bond state.

Surface: `MathBondOp` in `ConsensusTransaction`, idempotency-guarded via
`mathbond_key`, signed via the standard Ed25519 path. Error codes:
UK-7401..7407.

---

## Math bond probability market (vAMM + NegRisk)

The deterministic market engine lives in `unfer_consensus::mathbond_market`
(`MarketLedger`). `MarketOp`s (OpenNegRisk/AddLiquidity/RemoveLiquidity/
BuyOutcome/SellOutcome/Resolve/Claim) are a `ConsensusTransaction` variant.

The pricing model is a constant-product vAMM inspired by **Azuro's Liquidity
Tree**: LPs deposit e-coins into a singleton concentrated-liquidity pool, and
the protocol mathematically prices the odds of the trigger firing without
needing a direct buyer for every seller. The pool acts as counterparty to all
trades.

The **NegRisk CTF Adapter** (Gnosis-style) allows mutually-exclusive conditional
outcomes (e.g. "triggered by 2025" vs "triggered by 2026" vs "never") to share
a single pool, preventing liquidity fragmentation. When one outcome resolves,
the others become worthless — the NegRisk adapter ensures outcome prices sum
to 1.

Pricing: `P(outcome_i) = reserve_i / total_reserve`. When a trader buys
outcome tokens, they deposit e-coins into the pool's reserve and receive
tokens at the POST-trade marginal price (`tokens = net / P'`), so a
buy-then-sell round trip is neutral modulo fees and the pool cannot be drained
by price-jumping cycles. A fee (`fee_bps`) is deducted from each trade and
accrues to a separate LP-owned `lp_fees` accumulator; sell redemptions are
capped by the outcome's own reserve (solvency).

Resolution is **not a caller choice**: the winning outcome is a pure function
of the bond's trigger signal and the outcome maturity windows — the outcome
whose window contains `trigger_seq` wins; `None` (the bond matured without a
trigger) selects the terminal "never" outcome (`maturity_seq == u64::MAX`,
required at pool open). The consensus node validates the signal against the
bond ledger before the op applies, so a forged resolution is refused
identically on every node. When the pool resolves, the winning outcome gets
the entire pool reserve, all other outcomes go to zero, and a `Claim` op
redeems winning tokens pro-rata plus the LP's accrued fees.

**Settlement** (same `BondMarketService`): all pool cash lives in the
deterministic per-pool DID. LP deposits and buy proceeds are rowed in; sell
redemptions, liquidity withdrawals and post-resolution `Claim`s are rowed out
by spending the pool's coins (single-owner multi-input transfers, change coins
re-inserted). The pool is the counterparty: at resolution the winning token
holders redeem pro-rata against the pool reserve and the LPs collect the
accrued trading fees. `MarketLedger::apply_op` returns the exact amounts to
row (`RemoveLiquidity` cash, `SellOutcome` net payout, `Claim` payout), so the
service never guesses.

Surface: `MarketOp` in `ConsensusTransaction`, idempotency-guarded via
`market_key`, signed via the standard Ed25519 path. Error codes:
UK-7411..7419.

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

### Phase 3 — UNFCCC oracle bridge (zk-TLS) — PARTIAL (contract done)

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

Adaptation: the **shared contract is implemented** —
`unfer_protocol::MintRequest` (owner, amount, `source = "unfccc:vc:<orderId>"`,
optional blinding with a deterministic source-derived default) plus
`validate_source` (UK-7007 `CERT_ORACLE_REJECTED`) and `to_mint_kind()`, and the
additive FFI op `uk_cert_mint_request` (actor = current caller principal; a
non-authority caller is refused UK-7001 even with a valid oracle source).
Registered in `EXPECTED_SYMBOLS.txt` (61 symbols). The zk-TLS **prover** remains
client-side:
1. Connects to `https://unfccc.int` / `offset.climateneutralnow.org`,
   captures the cancellation-record page over TLSNotary.
2. Verifies the "Reason for cancellation" field embeds the user's `did:unfer`
   public key and that the serial-number range `[SN_start, SN_end]` matches the
   cancelled tonnage $m$.
3. Submits a `MintRequest` (via `uk_cert_mint_request`) that the mint authority
   signs as a `CertificateOp::Mint`.
The `source` field records the backing proof reference for auditability; a
verifier can re-derive the same proof from the public page.

### Phase 4 — Secondary fiat marketplace — DONE (escrow state machine)

Original: PayPal/Stripe escrow web app.

Adaptation: `unfer_consensus::escrow` is the on-chain escrow state machine. A
marketplace operator (the escrow agent) rows a certificate into a deterministic
intermediate DID between buyer and seller (`EscrowService::hold`), then
delivers (`release`) or returns (`refund`) it — each transition a
`CertificateOp::Transfer` on the consensus log, so any peer replays the market
onto the identical certificate root. Escrow keys are derived deterministically
from the operator's master key plus the (buyer, seller, origin-coin) triple;
an escrow settles exactly once (UK-7201 unknown coin, UK-7202 not holding,
UK-7203 already settled). The live testnet proves a full buy/sell lifecycle:
`phase6_secondary_market_escrow_lands_on_the_consensus_log` submits the market
ops through the rust-quepaxa ring and asserts every node's ledger root equals
the marketplace mirror's root. The PayPal/Stripe web-app coating maps to
`unfer_edge` (operator-only routes like `/api/gate/*`, `/api/cap/invoke`) plus
the `unfer_agent` NDJSON loop; wallets live in `velysterm/mathed` (keys never
leave the client).

### Phase 5 — GNU Taler mint — DONE (adapter)

Original: peg UTXOs to Taler e-coins for retail.

Adaptation: the `unfer_taler` adapter crate implements the exchange side of
the Taler flow over the certificate ledger. The exchange owns two views of the
same value flows:

- **On-ledger** — e-coins are ordinary certificates. `withdraw` has the
  treasury (the ledger's configured mint authority) mint a certificate owned by
  the reserve's customer; `deposit` *burns* e-coins (Taler honors e-coins it is
  given) and credits the merchant's fiat balance.
- **Private exchange state** — customer reserves, merchant balances, the
  two-phase wire gateway, and coin provenance, exactly like a real exchange's
  database.

The money-conservation identity across the seam is
`fiat_in - fiat_out = reserves + merchant_balances + funded_e_coins_outstanding`,
checked by `TalerExchange::audit()` after every op and proptested. Every op the
exchange emits is recorded so a `ConsensusNode` (configured with the same mint
authority) can replay the log and converge to the identical certificate root.
Peg-in requires a *confirmed* wire via the two-phase `WireGateway` (UK-7103);
withdraw refuses shortfalls (UK-7102) and non-live denominations (UK-7104);
deposit refuses foreign e-coins (UK-7107) and double deposits (UK-7105);
refresh of expired denominations is rejected (UK-7106). Anonymity is out of
scope (the transparent-core decision), so e-coins are keyed by the customer's
`did:unfer`. Codes UK-7101..7107.

### Phase 6 — Security, audit, mainnet — PARTIAL (testnet lite)

- **Circuit audit**: the invariants in `certs.rs` are the audit surface; a
  property test (proptest) that "no sequence of applied ops can violate
  conservation or double-spend" is the natural first audit deliverable (done:
  `fuzz_transfers_never_break_conservation_or_double_spend`). The Taler seam
  has its own conservation proptest on top (`fiat_conservation_never_breaks`).
- **Testnet**: the `network` feature (rust-quepaxa, tokio) now backs
  `ConsensusNode` for real: `unfer_consensus::net` brings up a 2-live-of-3
  mutually-authenticated (mTLS) loopback cluster. Value IDs are agreed by
  QuePaxa and applied by `LedgerStateMachine` into a shared committed log; each
  live node replays that log through its own `ConsensusNode` and converges on
  the identical certificate root. Node state is durable: recorder snapshots,
  runtime snapshots, the submission journal, and the replicated ledger live
  under a `<state_dir>`. `NetworkCluster::resume` restarts the whole cluster on
  the same TLS identities and socket addresses, reloads the committed log, and
  keeps proposing fresh slots. A value id that is decided but not yet applied is
  never lost: pending payloads survive any single node's death
  (`LedgerStateMachine` moves each id into the bank atomically, so the surviving
  peer or a restart re-applies it exactly once — covered by
  `decided_but_unapplied_value_survives_until_one_replica_applies` and
  `ledger_file_round_trips_committed_and_pending_state`). Demonstrated by
  `tests/phase6_testnet.rs` (run `cargo test -p unfer_consensus --features network`).
  The GNU Taler exchange is now a member of the same cluster: a full
  peg-in/withdraw/deposit/peg-out lifecycle emits signed ops that the cluster
  commits and every node replays onto the exchange's mirror certificate root,
  with the conservation audit holding throughout
  (`unfer_taler/tests/taler_testnet.rs`). A UNFCCC oracle client joins as a
  member too: it mints one certificate per verified VC through the cluster
  (Phase 3 `MintRequest` contract) and then audits the replicated log,
  flagging any mint whose backing VC it never verified
  (`phase6_oracle_client_audits_provenance_from_the_cluster`). All testnet
  members — live nodes, the Taler exchange, and the oracle client — now read the
  same replicated ledger.
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