# Consensus ledgers — invariants

Extracted from `AGENTS.md` (project review item **X3**). Five ledgers in
`unfer_consensus`, each settled by `unfer_taler`. What follows is the
**invariant** for each — the part that must not change. The prose they came from
is archived verbatim in `CHECKLIST_DETAIL_ARCHIVE.md`.

The invariant every one of these shares: **the ledger is a pure function of the
log.** Same ops in the same order, same state and same root, on every node. No
wall-clock, no randomness, no iteration over a `HashMap` without sorting. A
consensus engine that depends on any of those does not converge; it agrees with
itself on the run you tested and diverges on the one that matters.

---

## Certificate ledger — `unfer_consensus::certs`

`CertificateLedger` + `SparseMerkle`. Ops `Mint` / `Transfer` / `Burn`, applied by
`ConsensusNode::sync`. Minting is disabled unless a `MintAuthority` is configured.

| invariant | code |
|---|---|
| conservation on transfer | UK-7002 |
| no double-spend | UK-7004 |
| owner-only spend | UK-7005 |
| mint-authority check | UK-7001 |

**Threshold authority.** `MintAuthority::Threshold { threshold, total, pubkey }` —
reusing the sibling `dynamic-arctic` crate (path dep, `default-features = false`) —
turns minting into a t-of-n Arctic/Shine aggregate-signature check over the op's
64-byte `(RistrettoPoint, Scalar)`. `ConsensusNode::submit` and replay both route
CertificateOps through `CertificateLedger::verify_threshold_mint`, so the verdict
is identical on submit and on replay. Covered by `certs::proptests::threshold_*`
and `node::tests::threshold_*`.

See `docs/PLAN_REFI_EXCHANGE.md`.

## Unified auction — `unfer_consensus::auction`

`AuctionLedger`; settlement in `unfer_taler::auction::AuctionService` (needs both
`TalerExchange` and `CertificateLedger`). Ops `Open` / `Bid` / `Close`.

- **Clearing is a pure function of the recorded bids**: highest `price_per_unit`
  wins, ties break to the earliest `seq`. Every node replays the same log and
  converges on the same winner.
- **Two markets, one mechanism**: carbon credits (seller escrows the certificate,
  winner gets credits, loser refunded) and publicity inventory (payment only, no
  ledger asset).
- Payments are escrowed e-coins whose **face value equals the bid total**
  (denomination model, mirroring Taler). The operator derives each escrow DID/key
  from `(operator pubkey, lot_id, party, coin)`, so only it can settle.
- **Escrow is conserving**: same `total_supply` before and after.
- `uk_auction_close` both mutates and writes, so its marshalling uses a **single
  fixed-buffer call**, not the probe-then-copy `buf_out` protocol.

New `uk_auction_*` symbols follow the S29 checklist (including
`SENSITIVE_BLOCKED_SYMBOLS`).

## Math catastrophe bond — `unfer_consensus::mathbond`

`MathBondLedger`. Ops `Issue` / `Invest` / `SubmitProof` / `Mature` / `Settle`.

```
Issue → Invest → { SubmitProof (nanoda) → Triggered | Mature → Matured } → Settle
```

- A live `Issued`/`Funded` bond **cannot settle early**.
- The bond id **commits the full terms** (trigger, sponsor, principal, coupon,
  maturity, researcher).
- The trigger engine is `prob_kernel::verify::verify_export` (`nanoda_lib`) running
  **deterministically inside `apply_op`** — no human oracle, no external
  dependency. That is the whole point: the catastrophe trigger is a proof, checked
  by an independent checker.
- Settlement (`unfer_taler::bondmarket::BondMarketService`) rows every e-coin with
  conserving `CertificateOp`s into/out of deterministic operator-derived DIDs. On
  trigger the invested money becomes the researcher bounty (investors are wiped
  out); on maturity investors recover principal + coupon from the collateral.

Error codes UK-7401..7407.

## Bond probability market — `unfer_consensus::mathbond_market`

`MarketLedger`. Ops `OpenNegRisk` / `AddLiquidity` / `RemoveLiquidity` /
`BuyOutcome` / `SellOutcome` / `Resolve` / `Claim`. Codes UK-7411..7419.

- **Ratio vAMM** (Azuro-inspired): buys mint at the **post-trade** marginal price
  (`tokens = net / P'`), so a round trip is neutral modulo fees and there is no
  price-jumping drain. Sells redeem at the current price **capped by the outcome
  reserve** (solvency). Fees accrue to an LP-owned `lp_fees` accumulator.
- The NegRisk CTF adapter lets mutually-exclusive conditional outcomes share one
  pool. A terminal `"never"` outcome (`maturity_seq == u64::MAX`) is **required at
  open**.
- **Resolution is not a caller choice.** The winner is a pure function of the
  bond's trigger signal and the outcome windows, validated against the bond ledger
  by `ConsensusNode`; a forged resolve is refused (UK-7419 / UK-7413).
- `MarketLedger::apply_op` returns the exact row amounts, so settlement spends the
  right coins; every emitted op carries its **consensus-log** position as its seq.

## Attribution carbon credits — `unfer_consensus::attribution`

`AttributionLedger`. Ops `RegisterItem` / `OfferAttribution` / `Approve` / `Revoke` /
`IssueBadge`. Codes UK-7501..7512.

**The ledger receives the CONSENSUS log seq, not the submitter-set op seq**, so a
replayed `approve_seq`/`revoke_seq` — and therefore the badge date — matches.

- Lifecycle `Offered → Approved → Revoked`.
- Items are content-addressed: the same hash by a different author is refused
  (UK-7504). An offer needs both items registered (UK-7505/7508) by **different**
  authors (UK-7506) with a positive fee (UK-7507).
- Identical terms by the same pair collide (UK-7509). A live exclusive credit
  blocks a second offer against the same original — the Adidas/Yeezy sole-claim
  case.
- Only Author B approves (UK-7503), and only from `Offered` (UK-7502). Revocation
  keeps historical badges but mints no new one (UK-7510/7511).
- Settlement (`unfer_taler::attribution::AttributionService`) escrows Author A's fee
  **before** the signed offer is emitted (UK-7512), pays B on approval, and refunds
  A on refusal — nothing is stranded.
- Badges are deterministic Open Badges 3.0 assertions (W3C VC 2.0 JSON-LD), public
  or exclusive to an anonymous viewer identified by the SHA-256 of a random key
  their browser generates — the per-visualization badge; the operator never sees
  the key. The assertion bytes and Edyx19 proof are a pure function of
  `(credit, recipient, operator key, approval seq)`, so they are byte-identical
  across services.

**Keep the consensus-seq convention** (not the op seq) whenever a new ledger field
records a log position.
