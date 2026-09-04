//! GNU Taler exchange adapter (Plan R Phase 5).
//!
//! Adapts the real GNU Taler exchange→bank→merchant flow to the unfer
//! certificate ledger. The **consensus-visible** part of an e-coin's life is
//! ordinary `CertificateOp`s on the shared log:
//!
//! - **peg-in** fiat is confirmed by the two-phase wire gateway and credited
//!   to a customer **reserve** (private, exchange-side bookkeeping).
//! - **withdraw** turns reserve credit into an e-coin: the treasury (the mint
//!   authority) mints a certificate owned by the customer DID and debits the
//!   reserve. The mint is visible on the ledger with `source =
//!   "taler:reserve:<reserve_id>"`.
//! - **deposit** pays a merchant: the customer signs a conserving
//!   `CertificateOp::Transfer` to the merchant DID; the exchange credits the
//!   merchant's fiat balance.
//! - **peg-out** redeems merchant fiat: the exchange debits the merchant and
//!   prepares a wire transfer to their bank account (no ledger effect).
//!
//! The exchange keeps a private fiat-side mirror (`reserves`,
//! `merchant_balances`, `deposited` set) exactly as a real Taler exchange owns
//! its database. Every emitted op is recorded so a `ConsensusNode` can replay
//! the identical log and converge to the same certificate root.
//!
//! Anonymity is out of scope: this is the transparent core (the project-wide
//! decision), so an e-coin is keyed by the customer's `did:unfer` rather than
//! by an untracked coin private key.
//!
//! Error codes: UK-7101..UK-7106 (see `unfer_protocol::codes`).

pub mod attribution;
pub mod auction;
pub mod bondmarket;
pub mod denom;
pub mod exchange;
pub mod wire;

pub use attribution::{AttributionEscrowState, AttributionService, FeeHold, OpenBadgeAssertion};
pub use auction::{AuctionEscrowState, AuctionService, CreditHold, PaymentEscrow};
pub use bondmarket::{BondMarketService, CollateralHold, InvestmentHold, PoolCoin};
pub use denom::{Denomination, DenominationBook};
pub use exchange::{PegOut, ReserveId, TalerExchange};
pub use wire::{SimulatedWireGateway, WireGateway, WireRef, WireStatus};
