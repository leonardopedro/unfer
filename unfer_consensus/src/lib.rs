//! QuePaxa-backed consensus and federation for the unfer protocol layer.
//!
//! [`engine`] is the consensus engine (`ConsensusEngine`, `LocalConsensus`),
//! [`node`] the signing `ConsensusNode` that applies transactions deterministically,
//! `net` (feature `network`) the TLS/relay transport, [`signing`] the
//! Ed25519 keypairs, [`identity`] the DID registry, [`escrow`] the
//! two-phase escrow service, and [`certs`] the UTXO/carbon-certificate
//! ledger with its `SparseMerkle` (Plan R).

pub mod attribution;
pub mod auction;
pub mod certs;
pub mod engine;
pub mod escrow;
pub mod idempotency;
pub mod identity;
pub mod jobs;
pub mod lease;
pub mod mathbond;
pub mod mathbond_market;
#[cfg(feature = "network")]
pub mod net;
pub mod node;
pub mod signing;
/// D12: the deterministic fault scheduler. Test infrastructure, published so an
/// integration test can drive it; nothing in the runtime depends on it.
pub mod sim;
pub mod storage;

pub use attribution::{AttributionLedger, CreditState, badge_id, credit_id};
pub use auction::AuctionLedger;
pub use certs::{CertificateLedger, Coin, MintAuthority, SparseMerkle};
pub use engine::{ConsensusEngine, LocalConsensus};
pub use escrow::{Escrow, EscrowService, EscrowState};
pub use idempotency::IdempotencyStore;
pub use identity::IdentityRegistry;
pub use jobs::{
    ClaimState, Fence, FenceTable, FencedSink, JobClaim, JobCommand, JobQueue, JobState,
    OrderedMockSink, JOB_COMMAND_SCHEMA,
};
pub use lease::LeaderLease;
pub use mathbond::MathBondLedger;
pub use mathbond_market::MarketLedger;
pub use node::ConsensusNode;
pub use signing::{sign_transaction, verify_transaction, Keypair};
pub use storage::{
    JournalRecord, Recovery, RequestRecord, StorageError, TransitionLog, TransitionRecord,
};
