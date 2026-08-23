# Math Catastrophe Bond + vAMM/NegRisk Probability Market

> Architecture, design rationale, and test coverage for the math bond SPV and
> its probability market, implemented in `unfer_consensus::mathbond` and
> `unfer_consensus::mathbond_market`.

## Overview

A **math catastrophe bond** (MCB) is a catastrophe bond whose trigger is a
purely mathematical proof. Unlike traditional cat bonds (hurricanes, earthquakes)
or the proposed Ethereum-based math bonds (which require zkVMs and smart
contracts), this implementation:

- Uses **nanoda** (the project's Lean4-export verifier) as the deterministic
  trigger engine — no human oracle, no external dependency
- Runs entirely within the **QuePaxa consensus node** — every node that replays
  the same log converges on the identical trigger verdict
- Settles through the existing **certificate ledger** (Taler e-coins) — the
  same UTXO model used for carbon credits and auction settlement
- Trades trigger probabilities via a **vAMM + NegRisk market** inspired by
  Azuro's Liquidity Tree and Gnosis's Conditional Tokens Framework

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Sponsor                                                           │
│  Issues bond → locks collateral (e-coins) → specifies trigger      │
│  theorem, coupon rate, maturity, designated researcher              │
└────────────────────────┬────────────────────────────────────────────┘
                         │ MathBondOp::Issue
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MathBondLedger (unfer_consensus::mathbond)                        │
│                                                                     │
│  BondState: Issued → Funded → Triggered ──► Settled                │
│                       └───────► Matured ──► Settled                │
│                                                                     │
│  Operations:                                                        │
│  • Issue: create bond with trigger spec + collateral               │
│  • Invest: investors fund the bond (e-coins escrowed)              │
│  • SubmitProof: researcher submits Lean4 export → nanoda verifies  │
│  • Mature: record that maturity_seq passed without a trigger      │
│  • Settle: distribute collateral (only when Triggered or Matured)  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐  ┌──────────────────────────────────────┐
│  nanoda (verify)    │  │  MarketLedger (mathbond_market)      │
│                     │  │                                      │
│  verify_export(     │  │  vAMM: constant-product pricing     │
│    export_bytes,    │  │  P(outcome_i) = reserve_i / total   │
│    LeanVerifySpec)  │  │                                      │
│                     │  │  NegRisk: mutually-exclusive outcomes│
│  → ProofReport {    │  │  share one pool; when one resolves,  │
│    verified: bool,  │  │  others become worthless             │
│    ... }            │  │                                      │
└─────────────────────┘  └──────────────────────────────────────┘
```

## Design Rationale

### Why nanoda instead of zkVM?

The original proposal (Ethereum-based math bonds) uses RISC Zero zkVM to verify
proofs on-chain. This project already has nanoda (`prob_kernel::verify`) which
type-checks Lean4 exports deterministically. Using nanoda:

- **No external dependency**: the trigger runs inside the consensus node
- **Deterministic**: every node produces the same verdict
- **No gas costs**: no blockchain transaction fees
- **No trust assumptions**: nanoda is an independent checker, not trusting Lean's
  kernel

For future use (e.g., verifying proofs from other systems), RISC Zero zkVM is
documented as an additive layer — the `LeanVerifySpec` and `MathBondTrigger`
types are agnostic to the verification backend.

### Why vAMM + NegRisk instead of order book?

The original proposal uses Gnosis CTF for prediction market trading. This
project implements:

1. **Azuro-style vAMM**: LPs deposit e-coins into a singleton pool; the
   protocol mathematically prices odds without needing a direct buyer for every
   seller. The pool acts as counterparty.

2. **NegRisk CTF Adapter**: mutually-exclusive outcomes (e.g., "triggered by
   2025" vs "triggered by 2026" vs "never") share a single pool, preventing
   liquidity fragmentation. When one outcome resolves, the others become
   worthless.

The vAMM is simpler than an order book (no matching engine needed) and the
NegRisk adapter ensures outcome prices always sum to 1 (the "negated risk"
invariant).

## Protocol Types

### MathBondTrigger

```rust
pub struct MathBondTrigger {
    pub theorem: String,           // e.g. "P_eq_NP"
    pub spec_hash: String,         // SHA-256 of the Lean4 spec
    pub max_export_bytes: usize,   // max proof size
    pub permitted_axioms: Vec<String>,  // nanoda axiom allowlist
    pub strict: bool,              // UK-7401 on rejection
    pub nat_extension: bool,       // nanoda Nat kernel extension
    pub string_extension: bool,    // nanoda String kernel extension
}
```

### MathBondState

```
Issued → Funded → Triggered → Settled   (trigger payout)
                   Matured   → Settled   (maturity refund)
```

The `Mature` op records that the consensus log reached `maturity_seq` without
a successful trigger (anyone may record the passage of time; the ledger
enforces the seq). A live `Issued`/`Funded` bond cannot be settled — its
trigger window is still open.

### LiquidityPool (vAMM)

```rust
pub struct LiquidityPool {
    pub pool_id: PoolId,
    pub bond_id: MathBondId,
    pub outcome_reserves: Vec<(OutcomeId, u64)>,  // per-outcome e-coins
    pub total_reserve: u64,                        // total e-coins
    pub lp_shares: Vec<(String, u64)>,            // DID → LP shares
    pub total_shares: u64,
    pub fee_bps: u64,                              // trading fee
    pub resolved: bool,
    pub winner: Option<OutcomeId>,
}
```

## Pricing Model

### Ratio vAMM with post-trade marginal pricing

For a NegRisk pool with N outcomes:

- **Price formula**: `P(outcome_i) = reserve_i / total_reserve`
- **Initial state**: all outcomes have equal reserve (uniform 1/N prices)
- **Buy outcome i with `net` e-coins** (deposit minus `fee_bps`): the reserve
  moves to `reserve_i' = reserve_i + net`, `total' = total + net`, and the
  trader receives `tokens = net / P'` where `P'` is the POST-trade price
  (`tokens = net * total' / reserve_i'`). Buying at the post-trade marginal
  price (the constant-product convention) makes a buy-then-sell round trip
  exactly neutral modulo fees, so the pool cannot be drained by price-jumping
  cycles.
- **Sell `amount` tokens of outcome i**: redemption at the current price,
  `payout = amount * P`, capped by the outcome's own reserve so the pool can
  never go insolvent. The tokens are burned (removed from the outstanding
  count).
- **Invariant**: `Σ P(outcome_i) = 1` and `total_reserve == Σ reserve_i`
  exactly (integer-exact, deterministic across nodes).

### Fees

A configurable `fee_bps` (basis points) is deducted from each trade. The fee
accrues to a separate LP-owned `lp_fees` accumulator (the LP yield), and is
paid out pro-rata on `RemoveLiquidity` or `Claim`.

### Resolution (deterministic — never a caller choice)

When the math bond's trigger fires (nanoda verifies a proof) or the bond
matures without one:

1. The winning outcome is a pure function of the bond's trigger signal and the
   outcome maturity windows: the outcome whose window contains `trigger_seq`
   wins; `None` (matured without a trigger) selects the terminal "never"
   outcome (`maturity_seq == u64::MAX` — required at pool open). The consensus
   node validates the op's signal against the bond ledger before applying.
2. The winning outcome gets the entire pool reserve (price 1.0); all other
   outcomes go to zero.
3. `Claim` redeems winning outcome tokens pro-rata against the pool reserve
   (`payout = tokens * total / tokens_outstanding`), plus the LP's share of
   accrued fees — and of the whole reserve when nobody held winning tokens.
   Claims are idempotent: a second claim pays nothing.

## Error Codes

| Code | Name | Meaning |
|---|---|---|
| UK-7401 | MathBondUnknown | bond id not found |
| UK-7402 | MathBondWrongState | bond not in expected state |
| UK-7403 | MathBondNotResearcher | submitter is not designated researcher |
| UK-7404 | MathBondProofRejected | nanoda rejected the proof |
| UK-7405 | MathBondOverfunded | investment exceeds remaining capacity |
| UK-7406 | MathBondProofOversize | proof exceeds max export size |
| UK-7407 | MathBondAlreadyTriggered | bond already triggered |
| UK-7411 | MarketUnknownPool | pool id not found |
| UK-7412 | MarketPoolResolved | pool already resolved |
| UK-7413 | MarketUnknownOutcome | outcome not in pool |
| UK-7414 | MarketInsufficientTokens | not enough outcome tokens to sell |
| UK-7415 | MarketInsufficientShares | not enough LP shares to withdraw |
| UK-7416 | MarketNoLiquidity | pool has no liquidity |
| UK-7417 | MarketPriceUnderflow | NegRisk: price would go negative |
| UK-7418 | MarketPoolExists | pool already exists for this bond |
| UK-7419 | MarketNotResolved | pool not resolved — nothing to claim (or the bond has neither triggered nor matured) |

## Consensus Integration

Both `MathBondOp` and `MarketOp` are `ConsensusTransaction` variants, wired
into the full consensus pipeline:

- **Signing**: canonical bytes (zeroed signature) → SHA-256 → Ed25519
- **Idempotency**: `mathbond_key` / `market_key` content-addressed keys
- **Node dispatch**: `ConsensusNode::apply_transaction` routes to the
  appropriate ledger's `apply_op`
- **Deterministic replay**: every node that replays the same log converges on
  the identical bond/market state

## Settlement (`unfer_taler::bondmarket::BondMarketService`)

The ledgers decide state; the settlement service turns it into value — the
exact analogue of `unfer_taler::auction::AuctionService`. Every e-coin is an
ordinary `CertificateOp` rowed into/out of DIDs derived deterministically from
the operator's master key (bond collateral DID, per-investor investment DIDs,
per-pool cash DID), so a peer `ConsensusNode` replaying the same signed log
lands on the identical certificate root.

**Bond economics** (conserving): at settlement the pool holds `principal`
(collateral) + `invested`.

- **Triggered**: the invested e-coins are paid out as the **researcher
  bounty**; the sponsor keeps its collateral as the **catastrophe payment**
  (investors are wiped out — the point of a catastrophe bond).
- **Matured**: every investor recovers principal plus a `coupon_rate_bps`
  coupon, paid from the collateral (which keeps the remainder).

**Market mechanics**: all pool cash lives in the per-pool DID. LP deposits and
buy proceeds are rowed in; sell redemptions, liquidity withdrawals and
post-resolution claims are rowed out by spending the pool's coins
(single-owner multi-input transfers; change coins re-inserted). The pool is
the counterparty: at resolution winning token holders redeem pro-rata against
the pool reserve, LPs collect the accrued fees. `MarketLedger::apply_op`
returns the exact amounts to row (`RemoveLiquidity` cash, `SellOutcome` net
payout, `Claim` payout), so the service never guesses.

**Sequencing**: every emitted op carries a single global monotonic seq equal
to its position in `ops()` (its consensus-log position) — mirroring
`LocalConsensus` exactly, which the bond's absolute `maturity_seq` check and
the market's `trigger_seq` signal depend on.

## Test Coverage

### Math Bond Tests (`unfer_consensus::mathbond::tests`)

| Test | What it verifies |
|---|---|
| `issue_and_invest_lifecycle` | Issue → partial invest → fully funded |
| `overfunding_rejected` | UK-7405: investment > principal |
| `unknown_bond_rejected` | UK-7401: nonexistent bond id |
| `non_researcher_proof_rejected` | UK-7403: wrong submitter |
| `early_settle_rejected` | UK-7402: live bond cannot settle |
| `premature_mature_rejected` | UK-7402: Mature before maturity_seq |
| `maturity_refund_lifecycle` | Fund → Mature → Settle; no proof after maturity |
| `triggered_bond_settles` | Trigger payout settles; double-settle rejected |
| `bond_ids_distinguish_terms` | Same theorem, different terms → different ids |
| `issue_validates_terms` | Coupon > 100% / empty researcher / zero principal rejected |
| `report_matches_state` | Read-only report consistency |
| `replay_is_deterministic` | Two independent ledgers converge |
| `proof_oversize_rejected` | UK-7406: proof too large |
| `valid_proof_triggers_bond` | Real nanoda verification → trigger fires, trigger_seq recorded |
| `invalid_proof_rejected_bond_stays_funded` | Garbage proof rejected, bond stays |

### Market Tests (`unfer_consensus::mathbond_market::tests`)

| Test | What it verifies |
|---|---|
| `open_and_add_liquidity` | Open pool + LP deposit + uniform prices sum to 1 |
| `buy_moves_price` | Buying increases price, prices sum to 1 |
| `buy_grants_price_adjusted_tokens_and_roundtrip_conserves` | Tokens minted at post-trade price; round trip never drains the pool |
| `sell_cannot_drain_the_pool` | Redemption capped by the outcome reserve |
| `resolve_makes_loser_worthless` | Winner=100%, loser=0% (winner derived from trigger signal) |
| `never_outcome_wins_on_maturity` | Signal `None` → terminal never outcome |
| `trigger_signal_selects_the_window` | Trigger seq picks the containing maturity window |
| `double_resolve_rejected` | UK-7412: already resolved |
| `unknown_pool_rejected` | UK-7411: nonexistent pool |
| `open_pool_requires_terminal_never_outcome` | UK-7417: duplicate ids / missing never outcome |
| `claim_before_resolve_rejected` | UK-7419: nothing to claim |
| `claim_pays_winning_tokens_and_lp_fees` | Winners get the reserve, LPs get accrued fees, double claim pays 0 |
| `lp_gets_reserve_when_no_one_held_winning_tokens` | LPs keep the reserve when nobody bet the winner |
| `remove_liquidity_pays_fees_and_keeps_invariant` | Fee payout + exact reserve invariant |
| `replay_is_deterministic` | Two independent ledgers converge, including claims |
| `mathbond_market_consensus_roundtrip` | Full bond→market lifecycle through `ConsensusNode` |
| `market_resolve_forgery_rejected` | Forged resolutions refused (UK-7419 / UK-7413) |

### Settlement Tests (`unfer_taler::bondmarket::tests`)

| Test | What it verifies |
|---|---|
| `matured_bond_settles_investors_with_coupon` | Collateral escrowed at issue; maturity refund pays principal + coupon, sponsor keeps the remainder, supply conserved |
| `triggered_bond_pays_bounty_and_catastrophe` | Real nanoda trigger → researcher gets the invested bounty, sponsor keeps collateral, investors wiped out |
| `market_lifecycle_settles_claims` | LP + buy cash in the pool DID; winner's claim drains the reserve, LP collects fees, supply conserved |
| `overfunding_and_foreign_funding_rejected` | Wrong-owner funding (UK-7005) and overfunding (UK-7405) refused before any coin moves |
| `replay_converges_the_consensus_node` | `ops()` replayed into a `ConsensusNode` → identical certs root, bond state, market winner |

### Numerical Physics Tests

The derivative-variable gauge fixing (the Navier-Stokes pattern applied to
QG and NS) is verified in:

- `fock_sirk/tests/qg_starobinsky_derivative_variable.rs` — 3 tests:
  physical observables (1D), higher Hermite modes, unphysical data detection
- `fock_sirk/tests/ns_derivative_variable_fixing.rs` — 4 tests:
  1D consistency, 2D consistency, unphysical data, higher Hermite modes

These tests verify that the promoted spatial-gradient variables genuinely
represent spatial field derivatives in the nested Fock space, and that the
remaining physical observables (energy, Ehrenfest equations, composite
operators) are consistent and calculable while the gauge condition holds.

## License

The math bond and market implementations are part of the unfer kernel
(Apache-2.0). The nanoda verifier (`nanoda_lib`) is a dependency used as a
library (not a subprocess like Cadabra2/Why3). The vAMM pricing model is
inspired by Azuro's public documentation; the NegRisk adapter is inspired by
Gnosis's CTF framework. No code is copied — the designs are adapted to this
project's certificate-ledger and consensus architecture.
