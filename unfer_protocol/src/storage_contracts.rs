//! D19 (H05–H07, H10): checked contracts for the storage boundaries a restart
//! reads back through.
//!
//! A channel contract describes a message crossing between two processes. The
//! four remaining consensus hooks are not messages: a replica writes a snapshot
//! (runtime state, submission journal, recorder state, replicated ledger) and a
//! later incarnation of the same replica reads it back. The premise that makes
//! those writes usable is nevertheless the same kind of object — a claim with
//! fields — because "the file is written atomically" and "damage is refused
//! rather than defaulted" are exactly what a restart relies on and what a
//! comment at the call site cannot enforce.
//!
//! A [`StorageContract`] therefore records:
//!
//! * the writer and the reader (both locations, not one process);
//! * the schema the bytes are read under;
//! * the publish protocol, including on which side of the sync the name becomes
//!   visible;
//! * the recovery rule: what a missing file means and what a damaged one does;
//! * the number of writers (a single one, because two writers on one path have
//!   no publish order);
//! * whether the writer refuses to publish a sequence older than the one it
//!   already published;
//! * the byte bound, when the store enforces one.
//!
//! [`StorageContract::atomic_snapshot`] is the marker: it is produced only by a
//! contract that syncs before publishing and that refuses damage instead of
//! resetting to a default. The marker is Rust-side evidence, not a proof: the
//! filesystem must still honour `fsync`/`rename` ordering, which is an
//! assumption on the host.
//!
//! [`StorageContract::monotone_snapshot`] is the second marker: it is produced
//! only by a contract that declares [`MonotoneRule::NonDecreasing`], and it is
//! paired with a runtime guard ([`PublishWatermark`]) the writer consults before
//! replacing the file. The watermark remembers the highest sequence it has
//! published and refuses anything lower, so a stale writer that still holds an
//! older state cannot replace a newer one. What it does not do is make a stale
//! writer impossible on its own: the guard is per open handle, the contract's
//! single-writer premise is what makes that handle the path's watermark, and an
//! equal sequence is still accepted because a publish may change only the
//! outbox and not the committed prefix.
//!
//! ```
//! use unfer_protocol::storage_contracts::{
//!     PublishProtocol, RecoveryRule, StorageContract, SyncPoint,
//! };
//! use unfer_protocol::channel_contracts::{CapWitness, Location, Repo, SchemaId, UsageCap};
//! let writer = Location::new(Repo::Unfer, "writer.rs", "save").unwrap();
//! let reader = Location::new(Repo::Unfer, "writer.rs", "load").unwrap();
//! let schema = SchemaId::new("test.snapshot", 1).unwrap();
//! let writers = UsageCap::new("concurrent writers", 1, CapWitness::StaticLayout).unwrap();
//! let contract = StorageContract::new(
//!     writer,
//!     reader,
//!     schema,
//!     PublishProtocol::AtomicReplace { sync: SyncPoint::BeforePublish },
//!     RecoveryRule::MissingIsEmptyCorruptRefused,
//!     writers,
//! )
//! .unwrap();
//! assert!(contract.atomic_snapshot().is_ok());
//! ```

use crate::channel_contracts::{
    CapWitness, ContractError, Location, MAX_SUBMISSION_JOURNAL_BYTES, Repo, SchemaId, UsageCap,
};

/// When the durable bytes are made durable relative to the name becoming
/// visible. Publishing the name before the sync is the failure this field
/// exists to name: a crash then leaves a complete-looking file with no agreed
/// contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPoint {
    /// The payload is synced first, and only then published (rename, or the
    /// append that ends the frame).
    BeforePublish,
    /// The name is published first and the payload synced afterwards.
    AfterPublish,
}

/// How a snapshot's bytes reach their path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishProtocol {
    /// Write a temporary file, sync it, rename it over the path, sync the
    /// parent directory. A reader sees the old file or the new one.
    AtomicReplace { sync: SyncPoint },
    /// Append a complete, checksummed frame and sync it before the frame is
    /// part of the log. A reader sees a prefix of complete frames.
    AppendFrames { sync: SyncPoint },
}

/// What recovery does with what it finds on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryRule {
    /// A missing file is an empty state; a damaged complete snapshot is an
    /// error the caller must report, never a default state.
    MissingIsEmptyCorruptRefused,
    /// Like the snapshot rule, and an incomplete trailing frame is discarded
    /// because it was never published.
    TornTailDiscarded,
    /// A damaged file is replaced by an empty state. Restart "succeeds" and
    /// the committed history is silently gone.
    ResetOnDamage,
}

/// The marker produced only by a contract that publishes atomically and
/// refuses damage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicSnapshot(());

/// What a storage boundary promises about the order of its publishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotoneRule {
    /// The writer refuses a sequence below the last one it published, so the
    /// durable state never moves backwards.
    NonDecreasing,
    /// The boundary makes no ordering claim; a stale writer may replace a
    /// newer state.
    Unclaimed,
}

/// The marker produced only by a contract whose writer refuses to publish an
/// older sequence than the one already published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonotoneSnapshot(());

/// The runtime half of [`MonotoneRule::NonDecreasing`]: the highest sequence a
/// writer has published, and the refusal of anything below it.
///
/// Sequence is the amount of durable state the boundary publishes (for the
/// ledger, its committed length). A publisher that changes only local pending
/// state publishes the same sequence, which is accepted; a publisher offering
/// less is a state the boundary already superseded and is refused.
#[derive(Debug, Clone, Default)]
pub struct PublishWatermark {
    last: Option<u64>,
}

impl PublishWatermark {
    pub fn new() -> Self {
        Self { last: None }
    }

    /// Start from what a load observed, so the first publish after a restart is
    /// compared against the durable state rather than against nothing.
    pub fn seed(&mut self, sequence: u64) {
        self.last = Some(self.last.map_or(sequence, |last| last.max(sequence)));
    }

    /// Admit a publish, or refuse it as older than one already published.
    pub fn admit(&mut self, sequence: u64) -> Result<(), ContractError> {
        if let Some(published) = self.last
            && sequence < published
        {
            return Err(ContractError::StaleSnapshot {
                published,
                offered: sequence,
            });
        }
        self.last = Some(self.last.map_or(sequence, |last| last.max(sequence)));
        Ok(())
    }

    pub fn last(&self) -> Option<u64> {
        self.last
    }
}

/// One durable snapshot boundary. Every premise is a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageContract {
    writer: Location,
    reader: Location,
    schema: SchemaId,
    publish: PublishProtocol,
    recovery: RecoveryRule,
    writers: UsageCap,
    monotone: MonotoneRule,
    bytes: Option<UsageCap>,
}

impl StorageContract {
    pub fn new(
        writer: Location,
        reader: Location,
        schema: SchemaId,
        publish: PublishProtocol,
        recovery: RecoveryRule,
        writers: UsageCap,
    ) -> Result<Self, ContractError> {
        if writers.max() != 1 {
            return Err(ContractError::NotSingleWriter {
                writers: writers.max(),
            });
        }
        Ok(Self {
            writer,
            reader,
            schema,
            publish,
            recovery,
            writers,
            monotone: MonotoneRule::Unclaimed,
            bytes: None,
        })
    }

    /// Record a byte bound the store enforces. Only the submission journal has
    /// one; the other three grow with the log and claim no bound rather than an
    /// invented number nothing enforces.
    pub fn bounded(mut self, bytes: UsageCap) -> Self {
        self.bytes = Some(bytes);
        self
    }

    pub fn writer(&self) -> &Location {
        &self.writer
    }

    pub fn reader(&self) -> &Location {
        &self.reader
    }

    pub fn schema(&self) -> SchemaId {
        self.schema
    }

    pub fn publish(&self) -> PublishProtocol {
        self.publish
    }

    pub fn recovery(&self) -> RecoveryRule {
        self.recovery
    }

    pub fn writers(&self) -> &UsageCap {
        &self.writers
    }

    pub fn byte_bound(&self) -> Option<&UsageCap> {
        self.bytes.as_ref()
    }

    /// Record that the writer refuses to publish a sequence older than the one
    /// it already published.
    pub fn monotone(mut self) -> Self {
        self.monotone = MonotoneRule::NonDecreasing;
        self
    }

    pub fn monotone_rule(&self) -> MonotoneRule {
        self.monotone
    }

    /// The marker a stale writer's victim may rely on: the boundary declares
    /// that publishes never go backwards. A contract that makes no such claim
    /// yields no marker.
    pub fn monotone_snapshot(&self) -> Result<MonotoneSnapshot, ContractError> {
        match self.monotone {
            MonotoneRule::NonDecreasing => Ok(MonotoneSnapshot(())),
            MonotoneRule::Unclaimed => Err(ContractError::NoMonotonePublishRule),
        }
    }

    /// The marker a restart may rely on: synced before published, and damage is
    /// refused. A contract that publishes before syncing, or that resets its
    /// state on damage, yields no marker.
    pub fn atomic_snapshot(&self) -> Result<AtomicSnapshot, ContractError> {
        match self.publish {
            PublishProtocol::AtomicReplace {
                sync: SyncPoint::BeforePublish,
            }
            | PublishProtocol::AppendFrames {
                sync: SyncPoint::BeforePublish,
            } => {}
            PublishProtocol::AtomicReplace {
                sync: SyncPoint::AfterPublish,
            }
            | PublishProtocol::AppendFrames {
                sync: SyncPoint::AfterPublish,
            } => return Err(ContractError::SnapshotPublishedBeforeSync),
        }
        match self.recovery {
            RecoveryRule::MissingIsEmptyCorruptRefused | RecoveryRule::TornTailDiscarded => {
                Ok(AtomicSnapshot(()))
            }
            RecoveryRule::ResetOnDamage => Err(ContractError::SilentResetOnDamage),
        }
    }
}

/// H05: the consensus runtime snapshot (`<state_dir>/node-N/runtime.snapshot`).
///
/// `rust-quepaxa`'s `FileRuntimeStore` writes a temporary file, syncs it,
/// renames it over the path and syncs the directory, so the snapshot a restart
/// loads is either the previous one or the new one. A missing file is the first
/// start; a file that decodes to anything else is an error, not a fresh replica.
pub fn consensus_runtime_snapshot_contract() -> Result<StorageContract, ContractError> {
    snapshot_contract(
        "FileRuntimeStore::save",
        "FileRuntimeStore::load",
        "quepaxa.runtime-snapshot",
    )
}

/// H06: the durable submission journal (`<state_dir>/node-N/submission-journal.bin`).
///
/// The journal is what binds `(client_id, request_id)` to one outcome across a
/// restart, and it is the only one of the four whose size is capped: the server
/// opens it with this contract's bound, so the file it grows and the bound its
/// exactly-once claim rests on cannot drift.
pub fn consensus_submission_journal_contract() -> Result<StorageContract, ContractError> {
    Ok(snapshot_contract(
        "FileSubmissionJournal::open",
        "FileSubmissionJournal::load",
        "quepaxa.submission-journal",
    )?
    .bounded(UsageCap::new(
        "submission journal bytes",
        MAX_SUBMISSION_JOURNAL_BYTES,
        CapWitness::StaticLayout,
    )?))
}

/// H07: the peer recorder snapshot (`<state_dir>/node-N/recorder.snapshot`).
///
/// Same atomic replace as the runtime store. What it does **not** share is the
/// application's log: quepaxa persists recorder state separately and cannot
/// commit it in one transaction with the ledger, which is why recovery is
/// "recorder first, then the ledger's own log, replay by slot".
pub fn consensus_recorder_snapshot_contract() -> Result<StorageContract, ContractError> {
    snapshot_contract(
        "FileRecorderStore::save",
        "FileRecorderStore::load",
        "quepaxa.recorder-snapshot",
    )
}

/// H10: the replicated ledger snapshot (`<state_dir>/ledger.json`).
///
/// `LedgerFile::save` writes the temporary file, syncs it, renames and syncs the
/// directory, all under the single ledger lock, so the published snapshot is a
/// state that existed. A corrupt file is surfaced to the caller instead of being
/// replaced by an empty ledger, because an empty ledger silently un-commits the
/// node's history. The file's size is not bounded here: it grows with the
/// committed log. It is also the one boundary that declares a monotone publish
/// rule: `LedgerFile::save` admits the snapshot's committed length through a
/// [`PublishWatermark`] it seeded from the last load, so a caller still holding
/// an older state is refused instead of replacing the newer ledger.
pub fn consensus_ledger_snapshot_contract() -> Result<StorageContract, ContractError> {
    let writer = Location::new(
        Repo::Unfer,
        "unfer_consensus/src/net.rs",
        "LedgerFile::save",
    )?;
    let reader = Location::new(
        Repo::Unfer,
        "unfer_consensus/src/net.rs",
        "LedgerFile::load",
    )?;
    let schema = SchemaId::new("unfer.consensus.ledger-snapshot", 1)?;
    let writers = UsageCap::new("concurrent snapshot writers", 1, CapWitness::StaticLayout)?;
    Ok(StorageContract::new(
        writer,
        reader,
        schema,
        PublishProtocol::AtomicReplace {
            sync: SyncPoint::BeforePublish,
        },
        RecoveryRule::MissingIsEmptyCorruptRefused,
        writers,
    )?
    // D19: the ledger is the one boundary a stale writer could move backwards,
    // so it is the one that declares a monotone publish rule (H10).
    .monotone())
}

fn snapshot_contract(
    writer: &'static str,
    reader: &'static str,
    schema: &'static str,
) -> Result<StorageContract, ContractError> {
    let writer = Location::new(Repo::Unfer, "unfer_consensus/src/net.rs", writer)?;
    let reader = Location::new(Repo::Unfer, "unfer_consensus/src/net.rs", reader)?;
    let writers = UsageCap::new("concurrent snapshot writers", 1, CapWitness::StaticLayout)?;
    StorageContract::new(
        writer,
        reader,
        SchemaId::new(schema, 1)?,
        PublishProtocol::AtomicReplace {
            sync: SyncPoint::BeforePublish,
        },
        RecoveryRule::MissingIsEmptyCorruptRefused,
        writers,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn storage_contract(publish: PublishProtocol, recovery: RecoveryRule) -> StorageContract {
        let writer = Location::new(Repo::Unfer, "writer.rs", "save").unwrap();
        let reader = Location::new(Repo::Unfer, "writer.rs", "load").unwrap();
        let writers = UsageCap::new("writers", 1, CapWitness::StaticLayout).unwrap();
        StorageContract::new(
            writer,
            reader,
            SchemaId::new("test.snapshot", 1).unwrap(),
            publish,
            recovery,
            writers,
        )
        .unwrap()
    }

    #[test]
    fn a_contract_with_two_writers_has_no_publish_order() {
        let writer = Location::new(Repo::Unfer, "writer.rs", "save").unwrap();
        let reader = Location::new(Repo::Unfer, "writer.rs", "load").unwrap();
        let error = StorageContract::new(
            writer,
            reader,
            SchemaId::new("test.snapshot", 1).unwrap(),
            PublishProtocol::AtomicReplace {
                sync: SyncPoint::BeforePublish,
            },
            RecoveryRule::MissingIsEmptyCorruptRefused,
            UsageCap::new("writers", 2, CapWitness::CallerEstablished).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error, ContractError::NotSingleWriter { writers: 2 });
    }

    #[test]
    fn publishing_the_name_before_the_sync_yields_no_marker() {
        let checked = storage_contract(
            PublishProtocol::AtomicReplace {
                sync: SyncPoint::AfterPublish,
            },
            RecoveryRule::MissingIsEmptyCorruptRefused,
        );
        assert_eq!(
            checked.atomic_snapshot(),
            Err(ContractError::SnapshotPublishedBeforeSync)
        );

        let appended = storage_contract(
            PublishProtocol::AppendFrames {
                sync: SyncPoint::AfterPublish,
            },
            RecoveryRule::TornTailDiscarded,
        );
        assert_eq!(
            appended.atomic_snapshot(),
            Err(ContractError::SnapshotPublishedBeforeSync)
        );
    }

    #[test]
    fn resetting_a_damaged_snapshot_yields_no_marker() {
        let checked = storage_contract(
            PublishProtocol::AtomicReplace {
                sync: SyncPoint::BeforePublish,
            },
            RecoveryRule::ResetOnDamage,
        );
        assert_eq!(
            checked.atomic_snapshot(),
            Err(ContractError::SilentResetOnDamage)
        );
    }

    #[test]
    fn a_synced_replace_and_a_torn_tail_log_both_yield_the_marker() {
        for recovery in [
            RecoveryRule::MissingIsEmptyCorruptRefused,
            RecoveryRule::TornTailDiscarded,
        ] {
            let checked = storage_contract(
                PublishProtocol::AtomicReplace {
                    sync: SyncPoint::BeforePublish,
                },
                recovery,
            );
            assert!(checked.atomic_snapshot().is_ok(), "{recovery:?}");
        }
    }

    #[test]
    fn only_the_journal_claims_a_byte_bound() {
        assert_eq!(
            consensus_submission_journal_contract()
                .unwrap()
                .byte_bound()
                .map(|cap| cap.max()),
            Some(MAX_SUBMISSION_JOURNAL_BYTES)
        );
        for contract in [
            consensus_runtime_snapshot_contract().unwrap(),
            consensus_recorder_snapshot_contract().unwrap(),
            consensus_ledger_snapshot_contract().unwrap(),
        ] {
            assert_eq!(contract.byte_bound(), None, "{}", contract.schema().name());
        }
    }

    #[test]
    fn only_the_ledger_declares_a_monotone_publish_rule() {
        assert_eq!(
            consensus_ledger_snapshot_contract()
                .unwrap()
                .monotone_rule(),
            MonotoneRule::NonDecreasing
        );
        assert!(
            consensus_ledger_snapshot_contract()
                .unwrap()
                .monotone_snapshot()
                .is_ok()
        );
        for contract in [
            consensus_runtime_snapshot_contract().unwrap(),
            consensus_submission_journal_contract().unwrap(),
            consensus_recorder_snapshot_contract().unwrap(),
        ] {
            assert_eq!(contract.monotone_rule(), MonotoneRule::Unclaimed);
            assert_eq!(
                contract.monotone_snapshot(),
                Err(ContractError::NoMonotonePublishRule),
                "{}",
                contract.schema().name()
            );
        }
    }

    #[test]
    fn a_watermark_refuses_a_sequence_below_the_one_it_published() {
        let mut watermark = PublishWatermark::new();
        assert_eq!(watermark.last(), None);
        // The first publish establishes the sequence, whatever it is.
        watermark.admit(3).expect("the first publish is admitted");
        // An equal sequence is a publish that changed only the outbox.
        watermark.admit(3).expect("an equal sequence is admitted");
        watermark.admit(5).expect("a newer sequence is admitted");
        assert_eq!(
            watermark.admit(4),
            Err(ContractError::StaleSnapshot {
                published: 5,
                offered: 4,
            })
        );
        // A refused publish does not move the watermark backwards.
        assert_eq!(watermark.last(), Some(5));
        watermark
            .admit(5)
            .expect("the refused offer changed nothing");
    }

    #[test]
    fn a_watermark_seeded_from_a_load_refuses_the_state_it_loaded_over() {
        let mut watermark = PublishWatermark::new();
        watermark.seed(7);
        assert_eq!(watermark.last(), Some(7));
        assert_eq!(
            watermark.admit(6),
            Err(ContractError::StaleSnapshot {
                published: 7,
                offered: 6,
            })
        );
        watermark.admit(8).expect("progress past the loaded state");
    }

    #[test]
    fn every_live_storage_contract_yields_the_marker() {
        for contract in [
            consensus_runtime_snapshot_contract().unwrap(),
            consensus_submission_journal_contract().unwrap(),
            consensus_recorder_snapshot_contract().unwrap(),
            consensus_ledger_snapshot_contract().unwrap(),
        ] {
            assert!(
                contract.atomic_snapshot().is_ok(),
                "{}",
                contract.schema().name()
            );
            assert_eq!(contract.writers().max(), 1);
            assert_eq!(
                contract.writer().path(),
                "unfer_consensus/src/net.rs",
                "the census resolves the marker in net.rs"
            );
        }
    }
}
