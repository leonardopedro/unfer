//! QuePaxa-backed consensus and federation for the unfer protocol layer.
//!
//! [`engine`] is the consensus engine (`ConsensusEngine`, `LocalConsensus`),
//! [`node`] the signing `ConsensusNode` that applies transactions deterministically,
//! `net` (feature `network`) the TLS/relay transport, [`signing`] the
//! Ed25519 keypairs, [`identity`] the DID registry, [`escrow`] the
//! two-phase escrow service, and [`certs`] the UTXO/carbon-certificate
//! ledger with its `SparseMerkle` (Plan R).

pub mod auction;
pub mod certs;
pub mod engine;
pub mod escrow;
pub mod idempotency;
pub mod identity;
#[cfg(feature = "network")]
pub mod net;
pub mod jobs;
pub mod lease;
pub mod node;
pub mod signing;

pub use auction::AuctionLedger;
pub use certs::{CertificateLedger, Coin, MintAuthority, SparseMerkle};
pub use engine::{ConsensusEngine, LocalConsensus};
pub use escrow::{Escrow, EscrowService, EscrowState};
pub use idempotency::IdempotencyStore;
pub use identity::IdentityRegistry;
pub use jobs::{JobClaim, JobQueue, JobState};
pub use lease::LeaderLease;
pub use node::ConsensusNode;
pub use signing::{Keypair, sign_transaction, verify_transaction};
