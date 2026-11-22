//! H7: consensus/auction idempotency — duplicated or replayed delivery applies
//! exactly once.
//!
//! The ledgers (`CertificateLedger`, `AuctionLedger`) are deterministic: given
//! the same ordered log every node produces the same root. In a distributed
//! setting the *delivery* of a transaction is not guaranteed to be exactly-once
//! (a scheduler may double-fire, a relay may re-deliver). [`IdempotencyStore`]
//! guards every `CertificateOp` (Mint/Transfer/Burn) and `AuctionOp`
//! (Open/Bid/Close) at application: `once(key, f)` runs `f` only when `key` has
//! not been committed, so a duplicated or replayed transaction applies exactly
//! once. The store is rebuilt by replaying the same ordered log, so it is as
//! deterministic as the ledgers it guards (same log → same committed set →
//! same root).

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use sha2::{Digest as Sha2Digest, Sha256};

use unfer_protocol::submission::{Refusal, Slot, SubmissionId};
use unfer_protocol::{Code, ConsensusTransaction, Diagnostic, Severity};

/// Exactly-once application guard for replayed/duplicated consensus delivery.
///
/// `committed` maps an op's content key to the log sequence at which it was
/// applied. Retention: `prune_before(seq)` drops entries committed at an
/// earlier sequence (scheduled garbage collection).
#[derive(Debug, Clone, Default)]
pub struct IdempotencyStore {
    committed: HashMap<String, u64>,
}

impl IdempotencyStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `key` was already applied exactly-once.
    pub fn committed(&self, key: &str) -> bool {
        self.committed.contains_key(key)
    }

    /// Apply `f` exactly once for `key`. A duplicate/replay is a no-op that
    /// still reports success (the outcome was already produced); the caller's
    /// applied-seq watermark advances either way, mirroring a settled record.
    pub fn once(
        &mut self,
        key: &str,
        seq: u64,
        f: impl FnOnce() -> Result<(), Diagnostic>,
    ) -> Result<(), Diagnostic> {
        if self.committed.contains_key(key) {
            return Ok(());
        }
        f()?;
        self.committed.insert(key.to_string(), seq);
        Ok(())
    }

    /// Retention prune on a schedule: drop entries committed before `seq`
    /// (a replayed *old* delivery no longer needs its guard — the log has
    /// already been replayed past it). Keeps the store bounded.
    pub fn prune_before(&mut self, seq: u64) {
        self.committed.retain(|_, at| *at >= seq);
    }

    /// The number of guarded keys (test/QA surface).
    pub fn len(&self) -> usize {
        self.committed.len()
    }

    /// Whether no guards are held (test/QA surface).
    pub fn is_empty(&self) -> bool {
        self.committed.is_empty()
    }
}

/// Content key for a certificate op (Plan R): the acting DID plus the op kind
/// *without* the per-delivery `seq`/`signature`, so re-submitting the same
/// logical transfer maps to the same key and applies once.
pub fn certificate_key(op: &unfer_protocol::CertificateOp) -> String {
    let kind = serde_json::to_string(&op.kind).unwrap_or_default();
    format!("cert:{}:{}", op.did, kind)
}

/// Content key for an auction op: the acting DID plus the op kind, again
/// excluding the delivery-scoped `seq`/`signature`.
pub fn auction_key(op: &unfer_protocol::AuctionOp) -> String {
    let kind = serde_json::to_string(&op.kind).unwrap_or_default();
    format!("auction:{}:{}", op.did, kind)
}

/// Content key for a math bond op: the acting DID plus the op kind.
pub fn mathbond_key(op: &unfer_protocol::MathBondOp) -> String {
    let kind = serde_json::to_string(&op.kind).unwrap_or_default();
    format!("mathbond:{}:{}", op.did, kind)
}

/// Content key for a market op: the acting DID plus the op kind.
pub fn market_key(op: &unfer_protocol::MarketOp) -> String {
    let kind = serde_json::to_string(&op.kind).unwrap_or_default();
    format!("market:{}:{}", op.did, kind)
}

/// Content key for an attribution op: the acting DID plus the op kind.
pub fn attribution_key(op: &unfer_protocol::AttributionOp) -> String {
    let kind = serde_json::to_string(&op.kind).unwrap_or_default();
    format!("attribution:{}:{}", op.did, kind)
}

/// The idempotency key for a consensus transaction, if it is a guarded op.
pub fn transaction_key(tx: &ConsensusTransaction) -> Option<String> {
    match tx {
        ConsensusTransaction::CertificateOp(op) => Some(certificate_key(op)),
        ConsensusTransaction::AuctionOp(op) => Some(auction_key(op)),
        ConsensusTransaction::MathBondOp(op) => Some(mathbond_key(op)),
        ConsensusTransaction::MarketOp(op) => Some(market_key(op)),
        ConsensusTransaction::AttributionOp(op) => Some(attribution_key(op)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// D06: durable logical request identity and safe retention
// ---------------------------------------------------------------------------
//
// `IdempotencyStore` above deduplicates *delivery*: it answers "did this content
// already apply?". That is not enough for a client, because a client asks a
// different question after a timeout or a crash: "did **my request** apply?" A
// content key answers that only if the caller can reproduce the exact content,
// and it cannot distinguish my transfer from an identical-looking transfer by
// someone else that happens to hash the same.
//
// A [`RequestId`] is the caller's own name for one logical action, durable on
// the client before it is ever transmitted, so a retry carries the same name and
// the server can answer without re-deciding. Three rules make that safe:
//
// * same request id + same bytes  -> the recorded result, replayed;
// * same request id + different bytes -> a conflict (refused, never applied);
// * different request ids + equal bytes -> two distinct actions, left to the
//   business ledger's own exactly-once guard.
//
// Retention is a **floor**, not a prune schedule: once the server forgets a
// request id it can no longer tell a first delivery from a retry, so an id at or
// below the retained floor is refused outright rather than applied a second
// time. Unacknowledged clients cannot force unbounded memory: admission is
// capped and a full book answers with backpressure instead of evicting.

/// A caller's durable name for one logical action: never reused for different
/// bytes, and stable across retries, restarts and transport failures.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RequestId {
    /// Which federation/deployment this name is scoped to.
    pub deployment: String,
    /// The acting principal (a DID for signed transactions).
    pub principal: String,
    /// Which *run* of the client this is: a restarted client picks a fresh
    /// incarnation, so a reused sequence number is not mistaken for a retry of
    /// the previous run.
    pub incarnation: u64,
    /// Monotonic per (deployment, principal, incarnation).
    pub sequence: u64,
}

impl RequestId {
    pub fn new(
        deployment: impl Into<String>,
        principal: impl Into<String>,
        incarnation: u64,
        sequence: u64,
    ) -> Self {
        Self {
            deployment: deployment.into(),
            principal: principal.into(),
            incarnation,
            sequence,
        }
    }

    /// The scope key: everything except the sequence. Two ids with the same
    /// scope belong to the same monotonic sequence space.
    fn scope(&self) -> String {
        serde_json::to_string(&(
            self.deployment.as_str(),
            self.principal.as_str(),
            self.incarnation,
        ))
        .unwrap_or_else(|_| format!("{}:{}:{}", self.deployment, self.principal, self.incarnation))
    }

    /// A short rendering for diagnostics (never the full principal list).
    pub fn label(&self) -> String {
        format!(
            "{}#{}:{}",
            self.principal, self.incarnation, self.sequence
        )
    }
}

/// The payload digest the server records with an admission. Admission binds an
/// id to *these* bytes; the same id with different bytes is equivocation.
pub fn payload_digest(tx: &ConsensusTransaction) -> Result<[u8; 32], Diagnostic> {
    // The encoded form is the canonical one used on the wire, so two nodes
    // digesting the same transaction agree byte for byte.
    let bytes = serde_json::to_vec(tx).map_err(|error| {
        Diagnostic::new(
            Code::BAD_JSON,
            format!("could not canonicalise the transaction for admission: {error}"),
            Severity::Error,
        )
    })?;
    Ok(Sha256::digest(&bytes).into())
}

/// What the server recorded for a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmittedOutcome {
    /// Submitted, commitment not yet known.
    Pending { submission: SubmissionId },
    /// Committed and applied at `slot`.
    Accepted { submission: SubmissionId, slot: Slot },
    /// In the log, applied, and refused by the domain: a stable verdict, safe
    /// to replay to a retrying client.
    Rejected(Refusal),
}

/// One recorded admission: the digest it was bound to and its outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Admission {
    pub digest: [u8; 32],
    pub outcome: AdmittedOutcome,
}

/// What an attempt to admit a request produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionVerdict {
    /// First sight of this id: submit it, then record the outcome.
    Fresh,
    /// Already admitted with the same bytes: replay this result, do not submit.
    Duplicate(AdmittedOutcome),
    /// Same id, different bytes: equivocation. Nothing is applied.
    Conflict {
        admitted: [u8; 32],
        offered: [u8; 32],
    },
    /// At or below the retained floor: the record is gone, so a retry of this id
    /// cannot be distinguished from a first delivery and is refused.
    BelowRetention { floor: u64 },
    /// The book is full and no client has acknowledged a floor: backpressure.
    AtCapacity { capacity: usize },
}

/// Per-(deployment, principal, incarnation) retention window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ClientWindow {
    /// Every sequence at or below this value has been forgotten; a request in
    /// that range is refused rather than applied a second time.
    retained_floor: u64,
    entries: BTreeMap<u64, Admission>,
}

/// Server-side admission record: which requests this node has seen, bound to the
/// bytes they carried, with a bounded retained window per client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionBook {
    clients: BTreeMap<String, ClientWindow>,
    capacity: usize,
    #[serde(skip)]
    admitted: usize,
}

/// Default admission capacity. A full book answers [`AdmissionVerdict::AtCapacity`]
/// rather than evicting a record a client may still retry.
pub const DEFAULT_ADMISSION_CAPACITY: usize = 4096;

impl Default for AdmissionBook {
    fn default() -> Self {
        Self::new(DEFAULT_ADMISSION_CAPACITY)
    }
}

impl AdmissionBook {
    pub fn new(capacity: usize) -> Self {
        Self {
            clients: BTreeMap::new(),
            capacity: capacity.max(1),
            admitted: 0,
        }
    }

    /// Decide whether this request may proceed. A `Fresh` verdict is not yet a
    /// record: the caller submits, then calls [`AdmissionBook::record`], so an
    /// ingress failure leaves the id unclaimed and retryable.
    pub fn admit(&self, id: &RequestId, digest: [u8; 32]) -> AdmissionVerdict {
        let window = self.clients.get(&id.scope());
        if let Some(window) = window {
            if id.sequence <= window.retained_floor {
                return AdmissionVerdict::BelowRetention {
                    floor: window.retained_floor,
                };
            }
            if let Some(admission) = window.entries.get(&id.sequence) {
                return if admission.digest == digest {
                    AdmissionVerdict::Duplicate(admission.outcome.clone())
                } else {
                    AdmissionVerdict::Conflict {
                        admitted: admission.digest,
                        offered: digest,
                    }
                };
            }
        }
        if self.admitted >= self.capacity {
            return AdmissionVerdict::AtCapacity {
                capacity: self.capacity,
            };
        }
        AdmissionVerdict::Fresh
    }

    /// Record a freshly submitted request as pending.
    pub fn record(&mut self, id: &RequestId, submission: SubmissionId, digest: [u8; 32]) {
        let window = self.clients.entry(id.scope()).or_default();
        if !window.entries.contains_key(&id.sequence) {
            self.admitted += 1;
        }
        window.entries.insert(
            id.sequence,
            Admission {
                digest,
                outcome: AdmittedOutcome::Pending { submission },
            },
        );
    }

    /// Record the settled outcome. A refusal is cached (it is a deterministic
    /// verdict); a pending or timed-out request stays pending.
    pub fn settle(&mut self, id: &RequestId, outcome: AdmittedOutcome) {
        if let Some(window) = self.clients.get_mut(&id.scope()) {
            if let Some(admission) = window.entries.get_mut(&id.sequence) {
                admission.outcome = outcome;
            }
        }
    }

    /// Advance a client's retention floor to `through`: every request at or
    /// below it is forgotten (a tombstone, not a schedule). Returns how many
    /// records were dropped.
    pub fn acknowledge(&mut self, id: &RequestId, through: u64) -> usize {
        let Some(window) = self.clients.get_mut(&id.scope()) else {
            return 0;
        };
        let floor = through.max(window.retained_floor);
        let before = window.entries.len();
        window.entries.retain(|sequence, _| *sequence > floor);
        window.retained_floor = floor;
        let dropped = before - window.entries.len();
        self.admitted = self.admitted.saturating_sub(dropped);
        dropped
    }

    /// The retained floor for a request's scope, if the client is known.
    pub fn floor(&self, id: &RequestId) -> Option<u64> {
        self.clients.get(&id.scope()).map(|w| w.retained_floor)
    }

    /// The recorded admission for a request, if still retained.
    pub fn admission(&self, id: &RequestId) -> Option<&Admission> {
        self.clients
            .get(&id.scope())
            .and_then(|window| window.entries.get(&id.sequence))
    }

    /// Number of retained admissions across all clients.
    pub fn len(&self) -> usize {
        self.clients.values().map(|w| w.entries.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Recompute the cached count after deserialization.
    pub fn rehydrate(&mut self) {
        self.admitted = self.len();
    }
}

/// The sequence a client keeps durable so a retry reuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientSequence {
    pub incarnation: u64,
    pub sequence: u64,
}

/// Where a client persists its incarnation/sequence. Implementations must make
/// the write durable *before* returning: a sequence that is used but not stored
/// turns an in-flight request into an unrepeatable one.
pub trait RequestSequenceStore {
    fn load(&self) -> Result<Option<ClientSequence>, String>;
    fn save(&mut self, sequence: ClientSequence) -> Result<(), String>;
}

/// A client's request numbering: every id is persisted before it is returned,
/// so a crash between "about to send" and "sent" leaves a sequence the client
/// can safely retry with the same id.
pub struct ClientRequests<S: RequestSequenceStore> {
    deployment: String,
    principal: String,
    state: ClientSequence,
    store: S,
}

impl<S: RequestSequenceStore> ClientRequests<S> {
    /// Resume the persisted identity, or start one with `incarnation` if the
    /// store is empty. The incarnation itself is persisted here, before any
    /// request exists under it.
    pub fn open(
        deployment: impl Into<String>,
        principal: impl Into<String>,
        incarnation: u64,
        mut store: S,
    ) -> Result<Self, Diagnostic> {
        let state = match store.load() {
            Ok(Some(state)) => state,
            Ok(None) => {
                let fresh = ClientSequence {
                    incarnation,
                    sequence: 0,
                };
                store.save(fresh).map_err(|error| {
                    Diagnostic::new(
                        Code::STORAGE_FAILED,
                        format!("could not persist a client incarnation: {error}"),
                        Severity::Error,
                    )
                })?;
                fresh
            }
            Err(error) => {
                return Err(Diagnostic::new(
                    Code::STORAGE_FAILED,
                    format!("could not read the client sequence: {error}"),
                    Severity::Error,
                ))
            }
        };
        Ok(Self {
            deployment: deployment.into(),
            principal: principal.into(),
            state,
            store,
        })
    }

    /// Allocate the next request id, persisted **before** it is returned. On a
    /// store failure the sequence does not advance, so the caller may retry and
    /// the id it eventually sends is the one that was made durable.
    pub fn next(&mut self) -> Result<RequestId, Diagnostic> {
        let sequence = self.state.sequence.checked_add(1).ok_or_else(|| {
            Diagnostic::new(
                Code::STORAGE_FAILED,
                "the client sequence is exhausted; start a new incarnation",
                Severity::Error,
            )
        })?;
        let next = ClientSequence {
            incarnation: self.state.incarnation,
            sequence,
        };
        self.store.save(next).map_err(|error| {
            Diagnostic::new(
                Code::STORAGE_FAILED,
                format!("could not persist request sequence {sequence}: {error}"),
                Severity::Error,
            )
        })?;
        self.state = next;
        Ok(RequestId::new(
            self.deployment.clone(),
            self.principal.clone(),
            self.state.incarnation,
            sequence,
        ))
    }

    /// The current (already-persisted) id.
    pub fn current(&self) -> RequestId {
        RequestId::new(
            self.deployment.clone(),
            self.principal.clone(),
            self.state.incarnation,
            self.state.sequence,
        )
    }
}

/// A file-backed [`RequestSequenceStore`]: one small JSON record, written to a
/// temporary file, fsynced, renamed, and the directory fsynced, so a crash
/// leaves either the old sequence or the new one and never a half-written id.
#[derive(Debug, Clone)]
pub struct FileSequenceStore {
    path: std::path::PathBuf,
}

impl FileSequenceStore {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl RequestSequenceStore for FileSequenceStore {
    fn load(&self) -> Result<Option<ClientSequence>, String> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|error| format!("corrupt client sequence {}: {error}", self.path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!(
                "could not read client sequence {}: {error}",
                self.path.display()
            )),
        }
    }

    fn save(&mut self, sequence: ClientSequence) -> Result<(), String> {
        let bytes = serde_json::to_vec(&sequence)
            .map_err(|error| format!("could not encode a client sequence: {error}"))?;
        let temporary = self.path.with_extension("tmp");
        let mut file = std::fs::File::create(&temporary)
            .map_err(|error| format!("could not create {}: {error}", temporary.display()))?;
        std::io::Write::write_all(&mut file, &bytes)
            .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
        file.sync_all()
            .map_err(|error| format!("could not fsync {}: {error}", temporary.display()))?;
        std::fs::rename(&temporary, &self.path)
            .map_err(|error| format!("could not publish {}: {error}", self.path.display()))?;
        if let Some(parent) = self.path.parent() {
            if let Ok(directory) = std::fs::File::open(parent) {
                directory.sync_all().map_err(|error| {
                    format!("could not fsync {}: {error}", parent.display())
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn once_runs_f_only_on_first_delivery() {
        let mut store = IdempotencyStore::new();
        let mut runs = 0;
        store
            .once("k", 1, || {
                runs += 1;
                Ok(())
            })
            .unwrap();
        store
            .once("k", 2, || {
                runs += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(runs, 1, "duplicate delivery must not re-run f");
        assert!(store.committed("k"));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn failed_f_is_not_committed() {
        let mut store = IdempotencyStore::new();
        let r = store.once("k", 1, || {
            Err(Diagnostic::new(
                unfer_protocol::Code::CERT_AMOUNT_MISMATCH,
                "rejected",
                unfer_protocol::Severity::Error,
            ))
        });
        assert!(r.is_err());
        assert!(
            !store.committed("k"),
            "a failed apply must stay uncommitted"
        );
    }

    #[test]
    fn prune_before_keeps_recent_guards() {
        let mut store = IdempotencyStore::new();
        store.once("old", 1, || Ok(())).unwrap();
        store.once("new", 2, || Ok(())).unwrap();
        store.prune_before(2);
        assert!(!store.committed("old"), "old guard pruned");
        assert!(store.committed("new"), "recent guard retained");
        assert_eq!(store.len(), 1);
    }

    /// D06 test double: a store whose failure is the point of the test.
    struct MemoryStore {
        state: Option<ClientSequence>,
        fail: bool,
    }

    impl MemoryStore {
        fn new() -> Self {
            Self {
                state: None,
                fail: false,
            }
        }
    }

    impl RequestSequenceStore for MemoryStore {
        fn load(&self) -> Result<Option<ClientSequence>, String> {
            Ok(self.state)
        }

        fn save(&mut self, sequence: ClientSequence) -> Result<(), String> {
            if self.fail {
                return Err("disk full".to_string());
            }
            self.state = Some(sequence);
            Ok(())
        }
    }

    fn id(sequence: u64) -> RequestId {
        RequestId::new("federation-main", "did:unfer:alice", 4, sequence)
    }

    #[test]
    fn a_retry_with_the_same_bytes_replays_the_recorded_result() {
        let mut book = AdmissionBook::default();
        let request = id(7);
        let digest = [1u8; 32];
        assert_eq!(book.admit(&request, digest), AdmissionVerdict::Fresh);
        book.record(&request, SubmissionId(11), digest);
        book.settle(
            &request,
            AdmittedOutcome::Accepted {
                submission: SubmissionId(11),
                slot: Slot(3),
            },
        );
        assert_eq!(
            book.admit(&request, digest),
            AdmissionVerdict::Duplicate(AdmittedOutcome::Accepted {
                submission: SubmissionId(11),
                slot: Slot(3),
            })
        );
    }

    #[test]
    fn a_retry_with_different_bytes_is_a_conflict() {
        let mut book = AdmissionBook::default();
        let request = id(1);
        book.record(&request, SubmissionId(1), [1u8; 32]);
        assert_eq!(
            book.admit(&request, [2u8; 32]),
            AdmissionVerdict::Conflict {
                admitted: [1u8; 32],
                offered: [2u8; 32],
            }
        );
    }

    #[test]
    fn below_the_retained_floor_a_retry_is_refused_not_reapplied() {
        let mut book = AdmissionBook::default();
        for sequence in 1..=3 {
            book.record(&id(sequence), SubmissionId(sequence), [sequence as u8; 32]);
        }
        assert_eq!(book.acknowledge(&id(9), 3), 3, "three records released");
        assert_eq!(
            book.admit(&id(2), [2u8; 32]),
            AdmissionVerdict::BelowRetention { floor: 3 },
            "a forgotten request cannot be told from a first delivery"
        );
        assert_eq!(book.admit(&id(4), [4u8; 32]), AdmissionVerdict::Fresh);
        assert_eq!(book.floor(&id(1)), Some(3));
    }

    #[test]
    fn unacknowledged_clients_hit_the_quota_instead_of_evicting() {
        let mut book = AdmissionBook::new(2);
        book.record(&id(1), SubmissionId(1), [1u8; 32]);
        book.record(&id(2), SubmissionId(2), [2u8; 32]);
        assert_eq!(
            book.admit(&id(3), [3u8; 32]),
            AdmissionVerdict::AtCapacity { capacity: 2 },
            "a full book backpressures; it never drops a record a client may retry"
        );
        // Acknowledging a floor frees room without forgetting the boundary.
        assert_eq!(book.acknowledge(&id(1), 1), 1);
        assert_eq!(book.admit(&id(3), [3u8; 32]), AdmissionVerdict::Fresh);
        // Admission is a verdict, not a reservation: capacity is held by the
        // record that follows a successful submit.
        assert_eq!(book.len(), 1);
        book.record(&id(3), SubmissionId(3), [3u8; 32]);
        assert_eq!(book.len(), 2);
    }

    #[test]
    fn independent_incarnations_are_distinct_sequence_spaces() {
        let mut book = AdmissionBook::default();
        let first = RequestId::new("f", "did:unfer:alice", 1, 5);
        let restarted = RequestId::new("f", "did:unfer:alice", 2, 5);
        book.record(&first, SubmissionId(1), [1u8; 32]);
        assert_eq!(
            book.admit(&restarted, [9u8; 32]),
            AdmissionVerdict::Fresh,
            "a reused sequence in a fresh incarnation is a different request"
        );
        assert!(book.admission(&first).is_some());
    }

    #[test]
    fn a_client_store_failure_does_not_advance_the_sequence() {
        let mut requests =
            ClientRequests::open("f", "did:unfer:alice", 7, MemoryStore::new()).unwrap();
        let first = requests.next().unwrap();
        assert_eq!(first.sequence, 1);
        // Fail the next persist: the id was never transmitted, so the sequence
        // must stay where the durable record is, not where the attempt got to.
        requests.store.fail = true;
        let error = requests.next().unwrap_err();
        assert_eq!(error.code, Code::STORAGE_FAILED);
        requests.store.fail = false;
        let retried = requests.next().unwrap();
        assert_eq!(
            retried.sequence, 2,
            "the retried id is the first one that became durable"
        );
    }

    #[test]
    fn a_restarted_client_resumes_its_persisted_sequence() {
        let mut requests =
            ClientRequests::open("f", "did:unfer:alice", 9, MemoryStore::new()).unwrap();
        requests.next().unwrap();
        let stored = requests.store.state.unwrap();
        let mut resumed = ClientRequests::open(
            "f",
            "did:unfer:alice",
            99,
            MemoryStore {
                state: Some(stored),
                fail: false,
            },
        )
        .unwrap();
        assert_eq!(resumed.current().incarnation, 9, "the stored incarnation wins");
        assert_eq!(resumed.next().unwrap().sequence, 2);
    }

    #[test]
    fn a_file_backed_client_sequence_survives_a_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "unfer-seq-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("client-request.json");
        let mut requests =
            ClientRequests::open("f", "did:unfer:alice", 3, FileSequenceStore::new(&path)).unwrap();
        assert_eq!(requests.next().unwrap().sequence, 1);
        drop(requests);
        let mut reopened =
            ClientRequests::open("f", "did:unfer:alice", 3, FileSequenceStore::new(&path)).unwrap();
        assert_eq!(
            reopened.next().unwrap().sequence,
            2,
            "the durable sequence is the one the retry reuses"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn identical_transfer_bytes_map_to_one_key() {
        use unfer_protocol::{CertificateOp, CertificateOpKind};
        let mk = || {
            ConsensusTransaction::CertificateOp(CertificateOp {
                did: "did:unfer:bob".to_string(),
                kind: CertificateOpKind::Burn {
                    inputs: vec![unfer_protocol::CoinRef {
                        coin_id: unfer_protocol::CertId([7u8; 32]),
                        amount: 100,
                        owner: "did:unfer:bob".to_string(),
                    }],
                },
                seq: 1,
                signature: [0u8; 64],
            })
        };
        let a = transaction_key(&mk()).unwrap();
        let b = transaction_key(&mk()).unwrap();
        assert_eq!(a, b, "same logical op must share one idempotency key");
    }
}
