//! D19 (H14): the scheduler-supplied engine view as a checked contract.
//!
//! The simulator is not a transport and not a snapshot: it stands in for the
//! consensus engine a node reads its decided log from. `Sim` builds an
//! [`EngineView`](../unfer_consensus/index.html) per replica and hands it to
//! `ConsensusNode::open`, so every log read, sequence query and receipt lookup
//! the node makes goes through a view the harness controls.
//!
//! That substitution is only sound if the view keeps four premises, and each is
//! a field here rather than a comment in the harness:
//!
//! * **one view per replica** — the node and a later incarnation of the same
//!   replica must read the same decided log, or a restart would fabricate
//!   history from a second cluster;
//! * **the horizon hides a suffix only** — a partition may stop delivery at a
//!   prefix, but it may not rewrite the prefix already served;
//! * **a receipt exists only for a submission the view accepted** — a receipt
//!   for a request that was never submitted would let a client believe in an
//!   outcome the log does not contain;
//! * **`current_seq` ends the log the view returns** — the node's `is_synced`
//!   is exactly `applied_seq == current_seq`, so a sequence that runs past the
//!   returned log reports "caught up" while entries are still missing.
//!
//! [`EngineViewContract::scheduler_view`] is the marker: it is produced only by
//! a contract with all four premises. The marker is Rust-side evidence, not a
//! proof. It says the harness *intends* a suffix-only view; the behavioural
//! test in `sim.rs` shows the live `EngineView` behaves that way on a
//! partition/heal cycle, and it does not claim the real network engine has the
//! same shape.

use crate::channel_contracts::{CapWitness, ContractError, Location, Repo, SchemaId, UsageCap};

/// How a scheduler may change what a replica is told it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizonRule {
    /// A partition lowers the horizon, which hides a suffix of the decided log;
    /// healing raises it again. An already served slot keeps its value.
    SuffixOnly,
    /// A horizon change may replace the value of a slot already served. A
    /// restore then reads a history that never happened.
    RewritePrefix,
}

/// Which submissions may have a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptRule {
    /// Only a submission this view accepted has a receipt; anything else
    /// resolves to `Unknown(Unsubmitted)`.
    OnlyAccepted,
    /// A receipt may name any submission, including one the log never held.
    AnySubmission,
}

/// What `current_seq` promises about `get_log`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceRule {
    /// The last slot `get_log` returns is `current_seq`, so `applied ==
    /// current_seq` means the replica has the whole visible prefix.
    EndsAtLogEnd,
    /// `current_seq` may run past the returned log, so a caught-up replica can
    /// still be missing entries.
    BeyondLogEnd,
}

/// The marker produced only by a contract with all four premises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerView(());

/// One simulator engine-view boundary. Every premise is a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineViewContract {
    supplier: Location,
    reader: Location,
    schema: SchemaId,
    horizon: HorizonRule,
    receipts: ReceiptRule,
    sequence: SequenceRule,
    views: UsageCap,
}

impl EngineViewContract {
    pub fn new(
        supplier: Location,
        reader: Location,
        schema: SchemaId,
        horizon: HorizonRule,
        receipts: ReceiptRule,
        sequence: SequenceRule,
        views: UsageCap,
    ) -> Result<Self, ContractError> {
        if views.max() != 1 {
            return Err(ContractError::NotSingleView { views: views.max() });
        }
        Ok(Self {
            supplier,
            reader,
            schema,
            horizon,
            receipts,
            sequence,
            views,
        })
    }

    pub fn supplier(&self) -> &Location {
        &self.supplier
    }

    pub fn reader(&self) -> &Location {
        &self.reader
    }

    pub fn schema(&self) -> SchemaId {
        self.schema
    }

    pub fn horizon(&self) -> HorizonRule {
        self.horizon
    }

    pub fn receipts(&self) -> ReceiptRule {
        self.receipts
    }

    pub fn sequence(&self) -> SequenceRule {
        self.sequence
    }

    pub fn views(&self) -> &UsageCap {
        &self.views
    }

    /// The marker the node may rely on: one view, suffix-only horizon, receipts
    /// only for accepted submissions, and a sequence that ends the log.
    pub fn scheduler_view(&self) -> Result<SchedulerView, ContractError> {
        if self.horizon == HorizonRule::RewritePrefix {
            return Err(ContractError::SchedulerRewritesPrefix);
        }
        if self.receipts == ReceiptRule::AnySubmission {
            return Err(ContractError::ReceiptWithoutSubmission);
        }
        if self.sequence == SequenceRule::BeyondLogEnd {
            return Err(ContractError::SequenceBeyondLog);
        }
        Ok(SchedulerView(()))
    }
}

/// H14: the engine view `Sim` hands to each replica.
///
/// `Sim::new` builds one `EngineView` per replica and `Sim::restore` reuses it,
/// so the restart path reads the same decided log. A partition stores a horizon
/// of zero and a closed proposal path, and healing restores both; `get_log`
/// filters slots past the horizon, which can only remove a suffix. Receipts are
/// inserted by `submit` alone, and `current_seq` is `cluster.len()` bounded by
/// the same horizon `get_log` filters on, so it ends exactly the returned log.
pub fn simulator_engine_view_contract() -> Result<EngineViewContract, ContractError> {
    let supplier = Location::new(Repo::Unfer, "unfer_consensus/src/sim.rs", "Sim::new")?;
    let reader = Location::new(
        Repo::Unfer,
        "unfer_consensus/src/sim.rs",
        "SharedEngine::get_log",
    )?;
    EngineViewContract::new(
        supplier,
        reader,
        SchemaId::new("unfer.consensus.scheduler-engine-view", 1)?,
        HorizonRule::SuffixOnly,
        ReceiptRule::OnlyAccepted,
        SequenceRule::EndsAtLogEnd,
        UsageCap::new("engine views per replica", 1, CapWitness::StaticLayout)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view_contract(
        horizon: HorizonRule,
        receipts: ReceiptRule,
        sequence: SequenceRule,
    ) -> EngineViewContract {
        let supplier = Location::new(Repo::Unfer, "sim.rs", "Sim::new").unwrap();
        let reader = Location::new(Repo::Unfer, "sim.rs", "get_log").unwrap();
        EngineViewContract::new(
            supplier,
            reader,
            SchemaId::new("test.engine-view", 1).unwrap(),
            horizon,
            receipts,
            sequence,
            UsageCap::new("views", 1, CapWitness::StaticLayout).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn two_views_for_one_replica_are_refused() {
        let supplier = Location::new(Repo::Unfer, "sim.rs", "Sim::new").unwrap();
        let reader = Location::new(Repo::Unfer, "sim.rs", "get_log").unwrap();
        let error = EngineViewContract::new(
            supplier,
            reader,
            SchemaId::new("test.engine-view", 1).unwrap(),
            HorizonRule::SuffixOnly,
            ReceiptRule::OnlyAccepted,
            SequenceRule::EndsAtLogEnd,
            UsageCap::new("views", 2, CapWitness::CallerEstablished).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error, ContractError::NotSingleView { views: 2 });
    }

    #[test]
    fn a_horizon_that_rewrites_its_prefix_yields_no_marker() {
        let checked = view_contract(
            HorizonRule::RewritePrefix,
            ReceiptRule::OnlyAccepted,
            SequenceRule::EndsAtLogEnd,
        );
        assert_eq!(
            checked.scheduler_view(),
            Err(ContractError::SchedulerRewritesPrefix)
        );
    }

    #[test]
    fn a_receipt_for_an_unsubmitted_request_yields_no_marker() {
        let checked = view_contract(
            HorizonRule::SuffixOnly,
            ReceiptRule::AnySubmission,
            SequenceRule::EndsAtLogEnd,
        );
        assert_eq!(
            checked.scheduler_view(),
            Err(ContractError::ReceiptWithoutSubmission)
        );
    }

    #[test]
    fn a_sequence_past_the_log_yields_no_marker() {
        let checked = view_contract(
            HorizonRule::SuffixOnly,
            ReceiptRule::OnlyAccepted,
            SequenceRule::BeyondLogEnd,
        );
        assert_eq!(
            checked.scheduler_view(),
            Err(ContractError::SequenceBeyondLog)
        );
    }

    #[test]
    fn the_live_view_yields_the_marker() {
        let contract = simulator_engine_view_contract().unwrap();
        assert!(contract.scheduler_view().is_ok());
        assert_eq!(contract.views().max(), 1);
        assert_eq!(contract.supplier().path(), "unfer_consensus/src/sim.rs");
    }
}
