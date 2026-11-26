use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use prob_kernel::Session;
use unfer_protocol::submission::{
    HaltReason, Receipt, ReceiptState, Refusal, Slot, SubmissionId, SyncReport, UnknownReason,
    Watermark,
};
use unfer_protocol::{
    AgentRequest, Code, ConsensusTransaction, ContentRef, Diagnostic, JobFence, JobOp, JobOpKind,
    MarketOpKind, MathBondState, ModelSpec, Severity,
};

use crate::auction::AuctionLedger;
use crate::certs::{CertificateLedger, MintAuthority};
use crate::engine::ConsensusEngine;
use crate::idempotency::{
    AdmissionBook, AdmittedOutcome, IdempotencyStore, RequestId, transaction_key,
};
use crate::identity::IdentityRegistry;
use crate::jobs::{ClaimState, Fence, FencedSink, JobClaim, JobQueue, JobState};
use crate::storage::{
    DedupUpdate, FenceRecord, JournalRecord, RecordedOutcome, RequestRecord, StorageError,
    TransitionLog, TransitionRecord,
};

/// What one committed entry did to this node's state (D05/D07).
#[derive(Debug, Clone)]
enum ApplyFailure {
    /// The entry is invalid as a matter of policy. This is a stable verdict,
    /// recorded against the slot, and the slot still advances.
    Refused(Refusal),
    /// This node failed to evaluate or durably record the entry, so no verdict
    /// exists. The slot does not advance and the node stops applying.
    Halt(HaltReason),
}

impl ApplyFailure {
    /// Classify by code family rather than by call site: an infrastructure code
    /// means "this node could not tell", so it can never be recorded as a
    /// refusal of the entry.
    fn from(seq: u64, diagnostic: Diagnostic) -> Self {
        if diagnostic.code.is_infrastructure() {
            ApplyFailure::Halt(HaltReason::Failed {
                slot: Slot(seq),
                refusal: Refusal::from(diagnostic),
            })
        } else {
            ApplyFailure::Refused(Refusal::from(diagnostic))
        }
    }
}

pub struct ConsensusNode {
    engine: Box<dyn ConsensusEngine>,
    sessions: HashMap<u64, Session>,
    identity: IdentityRegistry,
    content: HashMap<String, ContentRef>,
    certs: CertificateLedger,
    auction: AuctionLedger,
    mathbond: crate::mathbond::MathBondLedger,
    market: crate::mathbond_market::MarketLedger,
    attribution: crate::attribution::AttributionLedger,
    /// H7: exactly-once guard for replayed/duplicated certificate + auction + mathbond + market + attribution ops.
    idempotency: IdempotencyStore,
    next_model_id: u64,
    applied_seq: u64,
    /// D05: the stable outcome of every committed entry the domain refused,
    /// keyed by the slot it occupied. A refusal is part of the applied history:
    /// it advanced the watermark, so it is recorded rather than re-tried.
    refusals: BTreeMap<u64, Refusal>,
    /// D05: set when the committed history itself is unusable (a gap or a
    /// backwards slot). A halted node applies nothing further until restarted
    /// onto a coherent history, so it cannot silently skip entries.
    halted: Option<HaltReason>,
    /// D06: which durable request ids this node has admitted, bound to the
    /// bytes they carried, with a bounded retained window per client.
    admissions: AdmissionBook,
    /// D06: the request a submission was admitted under, so a settled
    /// submission can be reported back to the book.
    request_of: HashMap<u64, RequestId>,
    /// D14: scheduled jobs and the ordered fences that decide who may settle
    /// them. The committed state owns the fence, so it lives here and is
    /// rebuilt from the journal rather than held by a worker.
    jobs: JobQueue,
    /// D07: one durable log for transitions and request bindings. `None` means
    /// the node is in-memory only, which is stated rather than implied.
    journal: Option<TransitionLog>,
    /// D07: transitions recovered from the journal that the engine log has not
    /// yet been replayed over. Each is compared, never applied twice.
    replay: Vec<TransitionRecord>,
    replay_next: usize,
    /// D07: the last journal failure, so a caller can tell the state the node
    /// reports from the state it could promise.
    journal_error: Option<String>,
}

impl ConsensusNode {
    pub fn new(engine: Box<dyn ConsensusEngine>) -> Self {
        Self {
            engine,
            sessions: HashMap::new(),
            identity: IdentityRegistry::new(),
            content: HashMap::new(),
            certs: CertificateLedger::new(MintAuthority::None),
            auction: AuctionLedger::new(),
            mathbond: crate::mathbond::MathBondLedger::new(),
            market: crate::mathbond_market::MarketLedger::new(),
            attribution: crate::attribution::AttributionLedger::new(),
            idempotency: IdempotencyStore::new(),
            next_model_id: 1,
            applied_seq: 0,
            refusals: BTreeMap::new(),
            halted: None,
            admissions: AdmissionBook::default(),
            request_of: HashMap::new(),
            jobs: JobQueue::new(),
            journal: None,
            replay: Vec::new(),
            replay_next: 0,
            journal_error: None,
        }
    }

    /// D07: open a node over a durable transition log.
    ///
    /// The journal is recovered first (a torn tail is discarded, corruption is
    /// refused), request bindings are rehydrated into the admission book, and
    /// the recovered transitions become a comparison script: the caller's next
    /// `sync()` replays the engine log and every already-recorded slot must
    /// produce the outcome the journal says it did. A mismatch is a mismatched
    /// checkpoint and halts the node rather than merging two histories.
    pub fn open(
        engine: Box<dyn ConsensusEngine>,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<(Self, crate::storage::Recovery), StorageError> {
        let (log, recovery) = TransitionLog::open(path)?;
        let mut node = Self::new(engine);
        node.journal = Some(log);
        for record in &recovery.records {
            if let JournalRecord::Request(request) = record {
                node.admissions
                    .record(&request.id, request.submission, request.digest);
                node.admissions.settle(&request.id, request.outcome.clone());
                node.request_of
                    .insert(request.submission.0, request.id.clone());
            }
            // D14: the last fence record per resource is the current fence. The
            // table is rebuilt, not trusted from memory, so a worker that
            // restarted cannot invent a claim it did not durably take.
            if let JournalRecord::Fence(fence) = record {
                node.jobs.install_fence(
                    Fence::new(
                        fence.deployment.clone(),
                        fence.resource.clone(),
                        fence.epoch,
                        fence.generation,
                    ),
                    fence.holder.clone(),
                    fence.state,
                );
            }
        }
        node.replay = recovery
            .records
            .iter()
            .filter_map(|record| match record {
                JournalRecord::Transition(transition) => Some(transition.clone()),
                JournalRecord::Request(_) | JournalRecord::Fence(_) => None,
            })
            .collect();
        Ok((node, recovery))
    }

    /// D07: whether this node writes a durable transition log.
    pub fn is_durable(&self) -> bool {
        self.journal.is_some()
    }

    /// D07: the most recent journal failure, if the node had to report a result
    /// it could not make durable.
    pub fn journal_error(&self) -> Option<&str> {
        self.journal_error.as_deref()
    }

    /// Whether this node's state may be read as a statement about the committed
    /// history.
    ///
    /// A halted node applied an entry to its in-memory ledgers before it could
    /// make that application durable, so its accessors describe an evaluation,
    /// not a state (`D12`/`D13`). Readers that serve state to a caller must ask
    /// this first and pass the error on: the honest answer is never the memory.
    /// `settle` and `sync` already refuse, because they go through the watermark;
    /// this is for the plain accessors (`content`, `certs`, `identity`, …), which
    /// cannot refuse on their own without changing their signatures.
    pub fn servable(&self) -> Result<(), Diagnostic> {
        if let Some(halt) = &self.halted {
            return Err(halt.diagnostic());
        }
        if let Some(error) = &self.journal_error {
            return Err(Diagnostic::new(
                Code::STORAGE_FAILED,
                format!(
                    "the last applied slot could not be recorded durably ({error}); this \
                     node's in-memory state is an evaluation, not a durable state"
                ),
                Severity::Error,
            ));
        }
        Ok(())
    }

    /// Test seam for the deterministic scheduler (plan D12): make this node's
    /// next journal append fail at `fault`. Returns `false` when the node has no
    /// journal, because then there is no write to fail and a caller that assumed
    /// one would be testing nothing.
    #[doc(hidden)]
    pub fn inject_storage_fault(&mut self, fault: crate::storage::FaultPoint) -> bool {
        match self.journal.as_mut() {
            Some(journal) => {
                journal.inject(fault);
                true
            }
            None => false,
        }
    }

    /// Constructor with a configured certificate mint authority (ReFi exchange).
    pub fn with_mint_authority(engine: Box<dyn ConsensusEngine>, authority: MintAuthority) -> Self {
        let mut node = Self::new(engine);
        node.certs = CertificateLedger::new(authority);
        node
    }

    /// Reconfigure the certificate mint authority (ReFi exchange). Minting is
    /// disabled by default; call with `MintAuthority::Only(did)` to permit a
    /// specific authority to mint, or `MintAuthority::None` to disable it.
    pub fn set_mint_authority(&mut self, authority: MintAuthority) {
        self.certs = CertificateLedger::new(authority);
    }

    // -----------------------------------------------------------------------
    // D14: ordered job fencing
    //
    // Every command below turns into one journal record before it publishes
    // anything. A transition whose record is not durable is rolled back, so a
    // node that restarts comes up with the fences it wrote and no others — the
    // rollback is what makes "the committed state owns the fence" true after a
    // crash rather than only in memory.
    //
    // None of these read a clock. A claim is authorized by being the current
    // fence, not by being held during an interval, so two workers partitioned
    // from each other cannot both be right about who leads: only one of their
    // fences is current, and the other is refused before it reaches a sink or a
    // journal.
    // -----------------------------------------------------------------------

    /// The committed job/fence state.
    pub fn jobs(&self) -> &JobQueue {
        &self.jobs
    }

    pub fn job_state(&self, job: &str) -> Option<JobState> {
        self.jobs.state(job)
    }

    /// The current fence for a job, if one was ever issued.
    pub fn job_fence(&self, job: &str) -> Option<&Fence> {
        self.jobs.fence(job)
    }

    /// Register a scheduled job. Registration is local state, not a fence
    /// transition, so it is not journaled: the job's state is `Queued` in every
    /// replica that registers it, and a claim is what needs a record.
    pub fn enqueue_job(&mut self, job: &str) {
        self.jobs.enqueue(job);
    }

    /// The ordered claim command: `(deployment, resource, epoch, generation)` is
    /// issued here, by the committed state, and returned to the caller.
    pub fn claim_job(&mut self, job: &str, holder: &str) -> Result<JobClaim, Diagnostic> {
        let before = self.jobs.clone();
        let claim = self.jobs.claim(job, holder)?;
        if let Err(error) = self.record_fence(&claim, ClaimState::Claimed, Some(holder.to_string()))
        {
            self.jobs = before;
            return Err(error);
        }
        Ok(claim)
    }

    /// Re-arm a claim with a fresh fence (same holder, new generation).
    pub fn renew_job(&mut self, claim: &JobClaim) -> Result<JobClaim, Diagnostic> {
        let before = self.jobs.clone();
        let renewed = self.jobs.renew(claim)?;
        if let Err(error) = self.record_fence(&renewed, ClaimState::Claimed, None) {
            self.jobs = before;
            return Err(error);
        }
        Ok(renewed)
    }

    /// Return a job to the queue after a failed fire, releasing its fence.
    /// A stale claim is refused and changes nothing.
    pub fn unclaim_job(&mut self, claim: &JobClaim) -> Result<(), Diagnostic> {
        let before = self.jobs.clone();
        self.jobs.unclaim(claim)?;
        if let Err(error) = self.record_fence(claim, ClaimState::Queued, None) {
            self.jobs = before;
            return Err(error);
        }
        Ok(())
    }

    /// Mark the job fired. Only the current fence can do this: a stale worker
    /// may keep computing, but its completion is not a state change here.
    pub fn mark_job_fired(&mut self, claim: &JobClaim) -> Result<(), Diagnostic> {
        let before = self.jobs.clone();
        self.jobs.mark_fired(claim)?;
        if let Err(error) = self.record_fence(claim, ClaimState::Fired, None) {
            self.jobs = before;
            return Err(error);
        }
        Ok(())
    }

    /// Disable the job on an authorization failure, releasing its fence.
    pub fn disable_job(&mut self, claim: &JobClaim) -> Result<(), Diagnostic> {
        let before = self.jobs.clone();
        self.jobs.disable(claim)?;
        if let Err(error) = self.record_fence(claim, ClaimState::Disabled, None) {
            self.jobs = before;
            return Err(error);
        }
        Ok(())
    }

    /// Fire the job through a sink that enforces order, then commit the settle.
    ///
    /// If the sink accepts the effect but the settle cannot be recorded, the
    /// effect is *applied and unrecorded* — the uncertain outcome D15's outbox
    /// exists for. It is reported as an error naming that state, never as a
    /// clean failure, because a caller that retried blindly would repeat it.
    pub fn settle_job(
        &mut self,
        claim: &JobClaim,
        sink: &mut dyn FencedSink,
        payload: &str,
    ) -> Result<(), Diagnostic> {
        let before = self.jobs.clone();
        self.jobs.settle(claim, sink, payload)?;
        if let Err(error) = self.record_fence(claim, ClaimState::Fired, None) {
            self.jobs = before;
            return Err(Diagnostic::new(
                Code::STORAGE_FAILED,
                format!(
                    "the effect for job '{}' reached the sink but the settle could not be recorded ({error}); 
                     the result is applied and unrecorded",
                    claim.job()
                ),
                Severity::Error,
            ));
        }
        Ok(())
    }

    /// Write one fence transition. Absent a journal the node is in-memory only
    /// (visible through `is_durable`), which is a property of the node the
    /// caller chose, not a silent success.
    fn record_fence(
        &mut self,
        claim: &JobClaim,
        state: ClaimState,
        holder: Option<String>,
    ) -> Result<(), Diagnostic> {
        let fence = claim.fence();
        let record = FenceRecord {
            deployment: fence.deployment.clone(),
            resource: fence.resource.clone(),
            epoch: fence.epoch,
            generation: fence.generation,
            holder,
            state,
        };
        if let Some(journal) = self.journal.as_mut() {
            journal
                .append(&JournalRecord::Fence(record))
                .map_err(|error| {
                    Diagnostic::new(
                        Code::STORAGE_FAILED,
                        format!(
                            "the fence for job '{}' could not be recorded ({error})",
                            claim.job()
                        ),
                        Severity::Error,
                    )
                })?;
        }
        Ok(())
    }

    /// Adapter for callers that only need the receipt id (D05 keeps the local
    /// call shape). Prefer [`ConsensusNode::submit_tracked`] when the caller
    /// will ask what happened: the returned number is a submission id, and on
    /// a network engine it is not the slot the entry lands in.
    pub fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic> {
        self.submit_tracked(tx).map(|id| id.0)
    }

    pub fn submit_tracked(&self, tx: ConsensusTransaction) -> Result<SubmissionId, Diagnostic> {
        self.verify_ingress(&tx)?;
        // D19-HOOK: H12 (send: admitted request handed to the ordered engine)
        self.engine.submit(tx).map(SubmissionId)
    }

    /// D06: submit under the caller's own durable request id.
    ///
    /// This is the admission point. The id is checked against what this node
    /// recorded for it, so a retry after a timeout replays the settled result
    /// instead of submitting a second action:
    ///
    /// * first sight -> submit, and record the admission;
    /// * same id, same bytes -> the recorded submission (or its rejection);
    /// * same id, different bytes -> UK-8101, nothing applied;
    /// * at or below the retained floor -> UK-8102, since a forgotten request
    ///   cannot be distinguished from a first delivery;
    /// * book full, no acknowledged floor -> UK-8103 (backpressure).
    ///
    /// The book is in memory in D06 and is rebuilt by replaying the log; making
    /// it survive a crash in the same transition as the applied state is D07.
    pub fn submit_request(
        &mut self,
        id: &RequestId,
        tx: ConsensusTransaction,
    ) -> Result<SubmissionId, Diagnostic> {
        let digest = crate::idempotency::payload_digest(&tx)?;
        match self.admissions.admit(id, digest) {
            crate::idempotency::AdmissionVerdict::Fresh => {
                let submission = self.submit_tracked(tx)?;
                self.admissions.record(id, submission, digest);
                self.request_of.insert(submission.0, id.clone());
                Ok(submission)
            }
            crate::idempotency::AdmissionVerdict::Duplicate(AdmittedOutcome::Pending {
                submission,
            })
            | crate::idempotency::AdmissionVerdict::Duplicate(AdmittedOutcome::Accepted {
                submission,
                ..
            }) => Ok(submission),
            crate::idempotency::AdmissionVerdict::Duplicate(AdmittedOutcome::Rejected(refusal)) => {
                Err(refusal.diagnostic())
            }
            crate::idempotency::AdmissionVerdict::Conflict { admitted, offered } => {
                Err(Diagnostic::new(
                    Code::REQUEST_CONFLICT,
                    format!(
                        "request {} was already admitted for {} and cannot be retried with {}: \
                         one request id names one action",
                        id.label(),
                        hex::encode(admitted),
                        hex::encode(offered)
                    ),
                    Severity::Error,
                ))
            }
            crate::idempotency::AdmissionVerdict::BelowRetention { floor } => Err(Diagnostic::new(
                Code::REQUEST_BELOW_RETENTION,
                format!(
                    "request {} is at or below the retained floor {floor}: this node can no \
                         longer tell a retry from a first delivery, so it refuses rather than \
                         applying the action twice",
                    id.label()
                ),
                Severity::Error,
            )),
            crate::idempotency::AdmissionVerdict::AtCapacity { capacity } => Err(Diagnostic::new(
                Code::REQUEST_QUOTA,
                format!(
                    "admission is at capacity ({capacity} retained requests) and no client \
                         has acknowledged a retention floor; acknowledge settled requests and retry"
                ),
                Severity::Error,
            )),
            crate::idempotency::AdmissionVerdict::IncarnationsExhausted { limit, known } => {
                Err(Diagnostic::new(
                    Code::REQUEST_QUOTA,
                    format!(
                        "principal {} already holds {known} live incarnations (limit {limit}): \
                         a new incarnation would start a fresh retention window, so it is refused \
                         until one is retired",
                        id.principal
                    ),
                    Severity::Error,
                ))
            }
        }
    }

    /// D06 follow-up: declare an incarnation finished so its principal regains
    /// incarnation headroom. The client must never retry a request under a
    /// retired incarnation, because its floor is gone with the window.
    pub fn retire_incarnation(
        &mut self,
        deployment: &str,
        principal: &str,
        incarnation: u64,
    ) -> bool {
        let retired = self
            .admissions
            .retire_incarnation(deployment, principal, incarnation);
        if retired {
            self.request_of
                .retain(|_, id| id.incarnation != incarnation || id.principal != principal);
        }
        retired
    }

    /// D06: the book (test/QA surface for retention and quota).
    pub fn admissions(&self) -> &AdmissionBook {
        &self.admissions
    }

    /// D06: acknowledge every request of this client at or below `through`, so
    /// the retained window can shrink. Returns how many records were dropped.
    pub fn acknowledge_requests(&mut self, id: &RequestId, through: u64) -> usize {
        self.admissions.acknowledge(id, through)
    }

    /// A threshold mint authority signs with an Arctic aggregate signature
    /// (Ristretto point + scalar, 64 bytes), not the single-key Ed25519 the
    /// generic path checks — route those through the ledger's threshold
    /// verification instead. Used at ingress *and* on replay, so a committed
    /// entry is checked by every node rather than only by its submitter.
    fn verify_ingress(&self, tx: &ConsensusTransaction) -> Result<(), Diagnostic> {
        if let ConsensusTransaction::CertificateOp(_) = tx
            && self.certs.is_threshold_authority()
        {
            return self.certs.verify_threshold_mint(tx);
        }
        crate::signing::verify_transaction(tx)
    }

    /// Apply every committed entry after the watermark. Returns how many
    /// entries were applied; a domain refusal does **not** appear here as an
    /// error (see [`ConsensusNode::sync_report`]), and only a halt does.
    pub fn sync(&mut self) -> Result<u64, Diagnostic> {
        self.sync_report().map(|report| report.entries)
    }

    /// D05: apply the committed prefix, distinguishing the two failure kinds.
    ///
    /// * A **domain refusal** is a property of the entry: it is recorded as a
    ///   [`Refusal`] against its slot and the slot advances, so one bad entry
    ///   does not wedge the good entries committed behind it.
    /// * A **halt** is a property of the history: a gap or a backwards slot
    ///   means this node cannot know what it missed, so it stops, remembers why,
    ///   and refuses to apply anything else.
    pub fn sync_report(&mut self) -> Result<SyncReport, Diagnostic> {
        if let Some(halt) = &self.halted {
            return Err(halt.diagnostic());
        }
        let log = self.engine.get_log(self.applied_seq + 1);
        let mut report = SyncReport::default();
        for (seq, tx) in log {
            let expected = self.applied_seq + 1;
            let halt = if seq > expected {
                Some(HaltReason::Gap {
                    expected: Slot(expected),
                    found: Slot(seq),
                })
            } else if seq < expected {
                Some(HaltReason::OutOfOrder {
                    applied: Watermark(self.applied_seq),
                    found: Slot(seq),
                })
            } else {
                None
            };
            if let Some(halt) = halt {
                let diagnostic = halt.diagnostic();
                self.halted = Some(halt);
                return Err(diagnostic);
            }
            let refused = match self.apply_committed(seq, &tx) {
                Ok(()) => None,
                Err(ApplyFailure::Refused(refusal)) => Some(refusal),
                Err(ApplyFailure::Halt(halt)) => {
                    // D07: a failure of *this node* (storage, budget) is not a
                    // verdict on the entry. The slot was not applied, so the
                    // watermark stops before it and nothing after it applies.
                    let diagnostic = halt.diagnostic();
                    self.halted = Some(halt);
                    return Err(diagnostic);
                }
            };
            // D07: the transition is durable before the watermark moves, and a
            // slot the journal already recorded must reproduce its outcome.
            let digest = crate::idempotency::payload_digest(&tx).map_err(|diagnostic| {
                let halt = HaltReason::Failed {
                    slot: Slot(seq),
                    refusal: Refusal::from(diagnostic),
                };
                let rendered = halt.diagnostic();
                self.halted = Some(halt);
                rendered
            })?;
            let outcome = match &refused {
                None => RecordedOutcome::Applied,
                Some(refusal) => RecordedOutcome::refused(refusal.code, &refusal.message),
            };
            let dedup = match &refused {
                None => transaction_key(&tx).map(|key| DedupUpdate { key, slot: seq }),
                Some(_) => None,
            };
            if let Err(error) = self.persist_transition(seq, digest, &outcome, dedup) {
                let halt = HaltReason::Failed {
                    slot: Slot(seq),
                    refusal: Refusal::new(
                        Code::STORAGE_FAILED,
                        format!("could not record slot {seq} durably: {error}"),
                    ),
                };
                let diagnostic = halt.diagnostic();
                self.journal_error = Some(error.to_string());
                self.halted = Some(halt);
                return Err(diagnostic);
            }
            if let Some(refusal) = refused {
                report.refusals.push((Slot(seq), refusal.clone()));
                self.refusals.insert(seq, refusal);
            }
            self.applied_seq = seq;
            report.entries += 1;
        }
        report.watermark = Watermark(self.applied_seq);
        Ok(report)
    }

    /// D07: make one transition durable. During recovery replay a slot the
    /// journal already recorded is **compared** instead of written again; a
    /// mismatch (different command bytes or a different decision) is a
    /// mismatched checkpoint and is reported as corruption rather than merged.
    fn persist_transition(
        &mut self,
        seq: u64,
        digest: [u8; 32],
        outcome: &RecordedOutcome,
        dedup: Option<DedupUpdate>,
    ) -> Result<(), StorageError> {
        if let Some(record) = self.replay.get(self.replay_next) {
            if record.slot == seq {
                self.replay_next += 1;
                if record.command_digest != digest {
                    return Err(StorageError::Corrupt {
                        offset: 0,
                        reason: format!(
                            "slot {seq}: the journal recorded command {} but the engine log has {}",
                            hex::encode(record.command_digest),
                            hex::encode(digest)
                        ),
                    });
                }
                if record.outcome != *outcome {
                    return Err(StorageError::Corrupt {
                        offset: 0,
                        reason: format!(
                            "slot {seq}: the journal recorded {:?}, replay produced {outcome:?}",
                            record.outcome
                        ),
                    });
                }
                return Ok(());
            }
            if record.slot < seq {
                return Err(StorageError::Corrupt {
                    offset: 0,
                    reason: format!(
                        "slot {seq}: the journal is missing slot {}, which the engine log skipped",
                        record.slot
                    ),
                });
            }
        }
        let Some(journal) = self.journal.as_mut() else {
            return Ok(());
        };
        journal.append(&JournalRecord::Transition(TransitionRecord {
            slot: seq,
            state_version: seq,
            command_digest: digest,
            outcome: outcome.clone(),
            dedup,
        }))
    }

    /// Re-verify a committed entry and apply it. The signature check is
    /// repeated here on purpose: ingress verification only covers what *this*
    /// node submitted, and every node must independently refuse a tampered
    /// entry that arrives through the log.
    fn apply_committed(&mut self, seq: u64, tx: &ConsensusTransaction) -> Result<(), ApplyFailure> {
        if let Err(diagnostic) = self.verify_ingress(tx) {
            return Err(ApplyFailure::from(seq, diagnostic));
        }
        self.apply_transaction(seq, tx)
            .map_err(|diagnostic| ApplyFailure::from(seq, diagnostic))
    }

    /// D05: wait (bounded) for a submission to be committed **and applied**,
    /// then report the settled outcome. `Committed` means the entry is in this
    /// node's state; a refusal comes back as [`ReceiptState::Refused`], and a
    /// deadline as [`UnknownReason::TimedOut`] -- never as success.
    pub fn settle(&mut self, id: SubmissionId, timeout: Duration) -> Receipt {
        let receipt = self.resolve(id, timeout);
        // D07: the request binding is made durable *before* the caller is told
        // the outcome, because that binding is what lets a retry be answered
        // after a restart. If it cannot be recorded, the honest answer is
        // "applied, but not recorded" -- never a refusal, and never a silent
        // claim that a retry will be recognised.
        if let Err(error) = self.remember_settled(&receipt) {
            self.journal_error = Some(error.to_string());
            return Receipt::unknown(receipt.id, UnknownReason::Unrecorded);
        }
        receipt
    }

    /// D06/D07: feed a settled outcome back to the admission book and the
    /// durable log, so a retry under the same request id replays it instead of
    /// submitting again. A pending or timed-out submission stays pending: it
    /// may still commit.
    fn remember_settled(&mut self, receipt: &Receipt) -> Result<(), StorageError> {
        let Some(request) = self.request_of.get(&receipt.id.0).cloned() else {
            return Ok(());
        };
        let outcome = match &receipt.state {
            ReceiptState::Committed(slot) => AdmittedOutcome::Accepted {
                submission: receipt.id,
                slot: *slot,
            },
            ReceiptState::Refused { refusal, .. } => AdmittedOutcome::Rejected(refusal.clone()),
            ReceiptState::Pending | ReceiptState::Unknown(_) => return Ok(()),
        };
        if self.admissions.admission(&request).map(|a| &a.outcome) == Some(&outcome) {
            return Ok(()); // a repeated settle of an already-recorded request
        }
        let digest = self
            .admissions
            .admission(&request)
            .map(|admission| admission.digest)
            .unwrap_or([0u8; 32]);
        if let Some(journal) = self.journal.as_mut() {
            journal.append(&JournalRecord::Request(RequestRecord {
                id: request.clone(),
                digest,
                submission: receipt.id,
                outcome: outcome.clone(),
                slot: receipt.slot().map(|slot| slot.0),
            }))?;
        }
        self.admissions.settle(&request, outcome);
        Ok(())
    }

    fn resolve(&mut self, id: SubmissionId, timeout: Duration) -> Receipt {
        let receipt = self.engine.wait(id, timeout);
        let slot = match receipt.state {
            ReceiptState::Committed(slot) => slot,
            _ => return receipt,
        };
        if self.applied_seq < slot.0 && self.sync_through(slot).is_err() {
            return Receipt::unknown(id, UnknownReason::Halted);
        }
        if let Some(refusal) = self.refusals.get(&slot.0) {
            return Receipt {
                id,
                state: ReceiptState::Refused {
                    slot,
                    refusal: refusal.clone(),
                },
            };
        }
        if self.applied_seq >= slot.0 {
            receipt
        } else {
            Receipt::unknown(id, UnknownReason::Halted)
        }
    }

    /// Apply until `slot` is inside the watermark. Stops early when the engine
    /// has nothing more committed: a later `settle`/`sync` picks it up.
    fn sync_through(&mut self, slot: Slot) -> Result<(), Diagnostic> {
        while self.applied_seq < slot.0 {
            if let Some(halt) = &self.halted {
                return Err(halt.diagnostic());
            }
            let before = self.applied_seq;
            self.sync()?;
            if self.applied_seq == before {
                break;
            }
        }
        Ok(())
    }

    fn apply_transaction(&mut self, seq: u64, tx: &ConsensusTransaction) -> Result<(), Diagnostic> {
        match tx {
            ConsensusTransaction::IdentityOp(op) => {
                self.identity.apply_identity_op(op)?;
            }
            ConsensusTransaction::SessionOp(op) => {
                self.apply_session_op(&op.op)?;
            }
            ConsensusTransaction::ContentOp(op) => {
                self.content
                    .insert(op.content_ref.cid.clone(), op.content_ref.clone());
            }
            ConsensusTransaction::CertificateOp(op) => {
                // H7: a duplicated or replayed certificate op applies exactly once.
                // A threshold mint authority re-verifies the Arctic aggregate
                // signature on replay (deterministic: same log, same verdict),
                // then applies through the shared idempotency gate.
                if self.certs.is_threshold_authority() {
                    self.certs.verify_threshold_mint(tx)?;
                }
                let actor = op.did.clone();
                let kind = op.kind.clone();
                let key = crate::idempotency::certificate_key(op);
                self.idempotency.once(&key, seq, || {
                    self.certs.apply_op(&actor, &kind, op.seq).map(|_| ())
                })?;
            }
            ConsensusTransaction::AuctionOp(op) => {
                // H7: a duplicated or replayed auction op applies exactly once.
                let actor = op.did.clone();
                let kind = op.kind.clone();
                let key = crate::idempotency::auction_key(op);
                self.idempotency.once(&key, seq, || {
                    self.auction.apply_op(&actor, &kind, op.seq).map(|_| ())
                })?;
            }
            ConsensusTransaction::MathBondOp(op) => {
                // H7: a duplicated or replayed math bond op applies exactly once.
                // The ledger receives the CONSENSUS seq (not the op's own
                // submitter-set field) so maturity enforcement and trigger_seq
                // reflect the honest log position.
                let actor = op.did.clone();
                let kind = op.kind.clone();
                let key = crate::idempotency::mathbond_key(op);
                self.idempotency.once(&key, seq, || {
                    self.mathbond.apply_op(&actor, &kind, seq).map(|_| ())
                })?;
            }
            ConsensusTransaction::MarketOp(op) => {
                // H7: a duplicated or replayed market op applies exactly once.
                let actor = op.did.clone();
                let kind = op.kind.clone();
                // A pool resolution is not a caller's free choice: the winner
                // is a pure function of the bond's trigger state. Validate the
                // op's trigger signal against the deterministic bond ledger
                // before the op is applied, so every node rejects a forged
                // resolution identically.
                if let MarketOpKind::Resolve {
                    pool_id,
                    trigger_seq,
                } = &kind
                {
                    let pool = self.market.pool(pool_id).ok_or_else(|| {
                        Diagnostic::new(Code::MARKET_UNKNOWN_POOL, "unknown pool", Severity::Error)
                    })?;
                    let bond = self.mathbond.bond(&pool.pool.bond_id).ok_or_else(|| {
                        Diagnostic::new(
                            Code::MATHBOND_UNKNOWN,
                            "the pool's bond does not exist on the ledger",
                            Severity::Error,
                        )
                    })?;
                    let expected = match (bond.state, bond.trigger_seq) {
                        (MathBondState::Triggered, Some(t)) => Some(t),
                        (MathBondState::Settled, t) => t,
                        (MathBondState::Matured, _) => None,
                        _ => {
                            return Err(Diagnostic::new(
                                Code::MARKET_NOT_RESOLVED,
                                format!(
                                    "bond is {:?}; it must be triggered or matured before the pool can resolve",
                                    bond.state
                                ),
                                Severity::Error,
                            ));
                        }
                    };
                    if *trigger_seq != expected {
                        return Err(Diagnostic::new(
                            Code::MARKET_UNKNOWN_OUTCOME,
                            format!(
                                "trigger signal {trigger_seq:?} does not match the bond ledger {expected:?}"
                            ),
                            Severity::Error,
                        ));
                    }
                }
                let key = crate::idempotency::market_key(op);
                self.idempotency.once(&key, seq, || {
                    self.market.apply_op(&actor, &kind, op.seq).map(|_| ())
                })?;
            }
            ConsensusTransaction::AttributionOp(op) => {
                // H7: a duplicated or replayed attribution op applies exactly once.
                let actor = op.did.clone();
                let kind = op.kind.clone();
                let key = crate::idempotency::attribution_key(op);
                self.idempotency
                    .once(&key, seq, || self.attribution.apply_op(&actor, &kind, seq))?;
            }
            ConsensusTransaction::JobOp(op) => {
                // D14: the fence is issued by this replay, in committed order.
                // There is no per-node FenceRecord to install: a restart that
                // replays the log reconstructs exactly the same claim, and a
                // committed claim cannot be refused against itself.
                self.apply_job_op(op)?;
            }
        }
        Ok(())
    }

    /// Apply one signed job-lifecycle command.
    ///
    /// `expected` is a compare-and-swap precondition only: it must equal the
    /// fence the committed state already holds, and the state refuses the
    /// command otherwise. A claim carries no expected fence because the state
    /// issues it; the submitter cannot choose its own epoch or generation.
    fn apply_job_op(&mut self, op: &JobOp) -> Result<(), Diagnostic> {
        let expected = |fence: &JobFence| {
            Fence::new(
                op.deployment.clone(),
                op.job.clone(),
                fence.epoch,
                fence.generation,
            )
        };
        match &op.kind {
            JobOpKind::Register => self.jobs.enqueue(&op.job),
            JobOpKind::Claim { holder } => {
                self.jobs.claim_in(&op.deployment, &op.job, holder)?;
            }
            JobOpKind::Renew { expected: fence } => {
                self.jobs.renew_at(&op.job, &expected(fence))?;
            }
            JobOpKind::Release { expected: fence } => {
                self.jobs.unclaim_at(&op.job, &expected(fence))?;
            }
            JobOpKind::Fire { expected: fence } => {
                self.jobs.mark_fired_at(&op.job, &expected(fence))?;
            }
            JobOpKind::Disable { expected: fence } => {
                self.jobs.disable_at(&op.job, &expected(fence))?;
            }
        }
        Ok(())
    }

    /// D05: the highest slot applied contiguously to this node's state.
    pub fn watermark(&self) -> Watermark {
        Watermark(self.applied_seq)
    }

    /// D05: the stable outcome recorded for a refused slot, if any.
    pub fn refusal(&self, slot: Slot) -> Option<&Refusal> {
        self.refusals.get(&slot.0)
    }

    /// D05: why this node stopped applying, if it did.
    pub fn halted(&self) -> Option<&HaltReason> {
        self.halted.as_ref()
    }

    /// H7: retention prune on a schedule — drop idempotency guards committed
    /// before `seq` (replayed old deliveries no longer need their guard).
    pub fn prune_idempotency_before(&mut self, seq: u64) {
        self.idempotency.prune_before(seq);
    }

    /// H7: whether a transaction key was already applied exactly-once.
    pub fn idempotency_committed(&self, tx: &ConsensusTransaction) -> bool {
        match transaction_key(tx) {
            Some(key) => self.idempotency.committed(&key),
            None => false,
        }
    }

    fn apply_session_op(&mut self, req: &AgentRequest) -> Result<(), Diagnostic> {
        // Dispatch derives from the shared CONSENSUS_OPS table (the canonical
        // `unfer_protocol::ops` registry) so the consensus seam can never
        // drift from the agent/edge allowlists.
        if unfer_protocol::ops::CONSENSUS_OPS.contains(&req.op.as_str()) {
            let spec: ModelSpec = serde_json::from_value(req.params.clone())
                .map_err(|e| Diagnostic::new(Code::BAD_JSON, e.to_string(), Severity::Error))?;
            let session = Session::new(&spec).map_err(|e| e.to_diagnostic())?;
            let id = self.next_model_id;
            self.next_model_id += 1;
            self.sessions.insert(id, session);
            return Ok(());
        }
        Err(Diagnostic::new(
            Code::BAD_JSON,
            format!("unsupported session op in consensus: {}", req.op),
            Severity::Error,
        ))
    }

    pub fn identity(&self) -> &IdentityRegistry {
        &self.identity
    }

    pub fn content(&self, cid: &str) -> Option<&ContentRef> {
        self.content.get(cid)
    }

    pub fn certs(&self) -> &CertificateLedger {
        &self.certs
    }

    pub fn auction(&self) -> &AuctionLedger {
        &self.auction
    }

    pub fn mathbond(&self) -> &crate::mathbond::MathBondLedger {
        &self.mathbond
    }

    pub fn market(&self) -> &crate::mathbond_market::MarketLedger {
        &self.market
    }

    pub fn attribution(&self) -> &crate::attribution::AttributionLedger {
        &self.attribution
    }

    pub fn session(&self, id: u64) -> Option<&Session> {
        self.sessions.get(&id)
    }

    pub fn session_mut(&mut self, id: u64) -> Option<&mut Session> {
        self.sessions.get_mut(&id)
    }

    pub fn applied_seq(&self) -> u64 {
        self.applied_seq
    }

    pub fn current_seq(&self) -> u64 {
        self.engine.current_seq()
    }

    pub fn is_synced(&self) -> bool {
        self.applied_seq == self.engine.current_seq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::MintAuthority;
    use crate::engine::LocalConsensus;
    use crate::signing::Keypair;
    use unfer_protocol::{CertificateOp, CertificateOpKind, IdentityOp, IdentityOpKind};

    fn make_node() -> ConsensusNode {
        ConsensusNode::new(Box::new(LocalConsensus::new()))
    }

    /// Build a mint op signed with a valid Arctic t-of-n aggregate signature.
    /// Mirrors `certs::proptests::threshold_mint_op` at the node level.
    fn threshold_mint_tx(
        n: u32,
        t: u32,
        coalition: &[u32],
        amount: u64,
        owner: &str,
        seq: u64,
    ) -> (ConsensusTransaction, [u8; 32]) {
        let (group_pk, _, seckeys) = arctic::arctic_core::keygen(n, t);
        let mut op = CertificateOp {
            did: format!("did:unfer:{}", hex::encode(group_pk.compress().to_bytes())),
            kind: CertificateOpKind::Mint {
                amount,
                owner: owner.to_string(),
                blinding: [1u8; 32],
                source: None,
            },
            seq,
            signature: [0u8; 64],
        };
        let tx = ConsensusTransaction::CertificateOp(op.clone());
        let msg = crate::signing::canonical_bytes(&tx);
        let r1: Vec<arctic::arctic_core::R1Output> = coalition
            .iter()
            .map(|&k| arctic::arctic_core::sign1(&seckeys[(k - 1) as usize], coalition, &msg))
            .collect();
        let shares: Vec<curve25519_dalek::scalar::Scalar> = coalition
            .iter()
            .map(|&k| {
                arctic::arctic_core::sign2(
                    &group_pk,
                    &seckeys[(k - 1) as usize],
                    coalition,
                    &msg,
                    &r1,
                )
                .unwrap()
            })
            .collect();
        let sig =
            arctic::arctic_core::combine(&group_pk, t, coalition, &msg, &r1, &shares).unwrap();
        let mut sig_bytes = [0u8; 64];
        sig_bytes[..32].copy_from_slice(sig.0.compress().as_bytes());
        sig_bytes[32..].copy_from_slice(&sig.1.to_bytes());
        op.signature = sig_bytes;
        (
            ConsensusTransaction::CertificateOp(op),
            group_pk.compress().to_bytes(),
        )
    }

    #[test]
    fn threshold_mint_submit_syncs_through_the_gate() {
        let (tx, pubkey) =
            threshold_mint_tx(7, 4, &[1, 2, 3, 4, 5, 6, 7], 100, "did:unfer:alice", 1);
        let mut node = make_node();
        node.set_mint_authority(MintAuthority::Threshold {
            threshold: 4,
            total: 7,
            pubkey,
        });
        // The Arctic aggregate signature passes the node's threshold gate.
        node.submit(tx).unwrap();
        let applied = node.sync().unwrap();
        assert_eq!(applied, 1);
        let supply = node.certs().total_supply();
        assert_eq!(supply, 100);
    }

    #[test]
    fn threshold_mint_rejects_forged_sig_at_submit() {
        let (mut tx, pubkey) =
            threshold_mint_tx(7, 4, &[1, 2, 3, 4, 5, 6, 7], 100, "did:unfer:alice", 1);
        if let ConsensusTransaction::CertificateOp(op) = &mut tx {
            op.signature[0] ^= 0xff;
        }
        let mut node = make_node();
        node.set_mint_authority(MintAuthority::Threshold {
            threshold: 4,
            total: 7,
            pubkey,
        });
        let err = node.submit(tx).unwrap_err();
        assert_eq!(err.code, Code::CERT_MINT_NOT_AUTHORIZED);
    }

    #[test]
    fn submit_and_sync_identity() {
        let mut node = make_node();
        let kp = Keypair::generate();
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx, &kp);
        node.submit(tx).unwrap();
        let applied = node.sync().unwrap();
        assert_eq!(applied, 1);
        assert!(node.identity().resolve(&kp.did()).is_some());
        assert!(node.is_synced());
    }

    #[test]
    fn submit_and_sync_content() {
        let mut node = make_node();
        let kp = Keypair::generate();
        let mut tx = ConsensusTransaction::ContentOp(unfer_protocol::ContentOp {
            did: kp.did(),
            content_ref: ContentRef {
                cid: "abc123".to_string(),
                magnet_uri: "magnet:?xt=urn:btih:abc123".to_string(),
                encryption_key: "x25519:deadbeef".to_string(),
                filesize: 1024,
                mime_type: "video/mp4".to_string(),
                chunks: vec![],
            },
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut tx, &kp);
        node.submit(tx).unwrap();
        node.sync().unwrap();
        assert!(node.content("abc123").is_some());
    }

    #[test]
    fn invalid_signature_rejected() {
        let node = make_node();
        let kp = Keypair::generate();
        let tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0xFFu8; 64],
            seq: 1,
            service_endpoint: None,
        });
        assert!(node.submit(tx).is_err());
    }

    #[test]
    fn two_nodes_converge() {
        let engine = LocalConsensus::new();
        let kp = Keypair::generate();

        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx, &kp);
        engine.submit(tx).unwrap();

        let mut node_a = ConsensusNode::new(Box::new(engine.clone()));
        let mut node_b = ConsensusNode::new(Box::new(engine.clone()));

        node_a.sync().unwrap();
        node_b.sync().unwrap();

        assert!(node_a.identity().resolve(&kp.did()).is_some());
        assert!(node_b.identity().resolve(&kp.did()).is_some());
        assert_eq!(node_a.applied_seq(), node_b.applied_seq());
    }

    #[test]
    fn incremental_sync() {
        let engine = LocalConsensus::new();
        let kp = Keypair::generate();

        let mut tx1 = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx1, &kp);
        engine.submit(tx1).unwrap();

        let mut node = ConsensusNode::new(Box::new(engine.clone()));
        node.sync().unwrap();
        assert_eq!(node.applied_seq(), 1);

        let kp2 = Keypair::generate();
        let mut tx2 = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp2.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp2.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx2, &kp2);
        engine.submit(tx2).unwrap();

        let applied = node.sync().unwrap();
        assert_eq!(applied, 1);
        assert_eq!(node.applied_seq(), 2);
        assert!(node.identity().resolve(&kp2.did()).is_some());
    }

    #[test]
    fn certificate_ledger_roundtrip_via_consensus() {
        use crate::certs::{MintAuthority, commit_coin};
        use unfer_protocol::{CertificateOp, CertificateOpKind, CoinRef};

        let engine = LocalConsensus::new();
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();

        let mut node = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(authority.did()),
        );
        let empty_root = node.certs().root();
        assert_ne!(empty_root, [0u8; 32]);

        // Mint 1000 to alice (signed by the authority).
        let mut mint = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount: 1000,
                owner: alice.did(),
                blinding: [1u8; 32],
                source: Some("unfccc:vc:TEST-0001".to_string()),
            },
            seq: 1,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut mint, &authority);
        node.submit(mint).unwrap();
        node.sync().unwrap();

        let alice_coin = commit_coin(1000, &alice.did(), &[1u8; 32]);
        assert!(node.certs().utxo(&alice_coin).is_some());
        assert_eq!(node.certs().total_supply(), 1000);

        // Transfer the whole 1000 to bob (signed by alice).
        let input = CoinRef {
            coin_id: alice_coin,
            amount: 1000,
            owner: alice.did(),
        };
        let output = CoinRef {
            coin_id: alice_coin, // recomputed below; amount/owner drive the id
            amount: 1000,
            owner: bob.did(),
        };
        let mut transfer = ConsensusTransaction::CertificateOp(CertificateOp {
            did: alice.did(),
            kind: CertificateOpKind::Transfer {
                inputs: vec![input],
                outputs: vec![output],
            },
            seq: 2,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut transfer, &alice);
        node.submit(transfer).unwrap();
        node.sync().unwrap();

        let bob_coin = commit_coin(1000, &bob.did(), &[0u8; 32]);
        assert!(node.certs().utxo(&bob_coin).is_some());
        assert!(node.certs().utxo(&alice_coin).is_none());
        assert!(node.certs().is_spent(&alice_coin));
        assert_eq!(node.certs().total_supply(), 1000);

        // Burn bob's certificate.
        let burn_input = CoinRef {
            coin_id: bob_coin,
            amount: 1000,
            owner: bob.did(),
        };
        let mut burn = ConsensusTransaction::CertificateOp(CertificateOp {
            did: bob.did(),
            kind: CertificateOpKind::Burn {
                inputs: vec![burn_input],
            },
            seq: 3,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut burn, &bob);
        node.submit(burn).unwrap();
        node.sync().unwrap();

        assert!(node.certs().utxo(&bob_coin).is_none());
        assert_eq!(node.certs().total_supply(), 0);
        // Retiring the only UTXO restores the empty root.
        assert_eq!(node.certs().root(), empty_root);
    }

    #[test]
    fn refused_certificate_op_advances_the_slot_with_a_typed_outcome() {
        use unfer_protocol::{CertificateOp, CertificateOpKind};

        let engine = LocalConsensus::new();
        let alice = Keypair::generate();
        let mut node = ConsensusNode::new(Box::new(engine));
        // Minting is disabled by default (MintAuthority::None).
        let mut mint = ConsensusTransaction::CertificateOp(CertificateOp {
            did: alice.did(),
            kind: CertificateOpKind::Mint {
                amount: 100,
                owner: alice.did(),
                blinding: [9u8; 32],
                source: None,
            },
            seq: 1,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut mint, &alice);
        // Signature is valid, so it reaches the log; application refuses it with
        // UK-7001 (mint not authorized).
        let seq = node.submit(mint).unwrap();
        assert_eq!(seq, 1);
        // D05: the refusal is a stable outcome at that slot, not a sync error.
        // Reporting it as an error would stop the prefix at a domain rejection
        // and wedge every good entry committed behind it.
        let report = node.sync_report().unwrap();
        assert_eq!(report.entries, 1);
        assert_eq!(report.watermark, Watermark(1));
        assert_eq!(report.refusals.len(), 1);
        assert_eq!(report.refusals[0].0, Slot(1));
        assert_eq!(report.refusals[0].1.code, Code::CERT_MINT_NOT_AUTHORIZED);
        let refusal = node.refusal(Slot(1)).expect("a refusal is recorded");
        assert_eq!(refusal.code, Code::CERT_MINT_NOT_AUTHORIZED);
        assert_eq!(node.certs().total_supply(), 0, "the refusal minted nothing");
    }

    #[test]
    fn five_nodes_converge_on_certificate_root() {
        use crate::certs::commit_coin;
        use unfer_protocol::{CertificateOp, CertificateOpKind, CoinRef};

        // QuePaxa target is 5-7 validator nodes; all share one ordered log.
        let engine = LocalConsensus::new();
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();
        let carol = Keypair::generate();

        let mut nodes: Vec<ConsensusNode> = (0..5)
            .map(|_| {
                ConsensusNode::with_mint_authority(
                    Box::new(engine.clone()),
                    MintAuthority::Only(authority.did()),
                )
            })
            .collect();

        let sign_and_submit =
            |node: &ConsensusNode, tx: &mut ConsensusTransaction, kp: &Keypair| {
                crate::signing::sign_transaction(tx, kp);
                node.submit(tx.clone()).unwrap();
            };

        // Interleaved certificate + identity ops on the shared log.
        // Both coins are minted to bob so he can spend them together (a
        // multi-input transfer requires one signer to own every input).
        let mut mint1 = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount: 1000,
                owner: bob.did(),
                blinding: [1u8; 32],
                source: Some("unfccc:vc:MLT-0001".to_string()),
            },
            seq: 1,
            signature: [0u8; 64],
        });
        sign_and_submit(&nodes[0], &mut mint1, &authority);

        let mut mint2 = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount: 600,
                owner: bob.did(),
                blinding: [2u8; 32],
                source: Some("unfccc:vc:MLT-0002".to_string()),
            },
            seq: 2,
            signature: [0u8; 64],
        });
        sign_and_submit(&nodes[1], &mut mint2, &authority);

        // Two-input transfer: bob's 1000 + bob's 600 -> split to carol + alice.
        let bob_coin_a = commit_coin(1000, &bob.did(), &[1u8; 32]);
        let bob_coin_b = commit_coin(600, &bob.did(), &[2u8; 32]);
        let mut transfer = ConsensusTransaction::CertificateOp(CertificateOp {
            did: bob.did(),
            kind: CertificateOpKind::Transfer {
                inputs: vec![
                    CoinRef {
                        coin_id: bob_coin_a,
                        amount: 1000,
                        owner: bob.did(),
                    },
                    CoinRef {
                        coin_id: bob_coin_b,
                        amount: 600,
                        owner: bob.did(),
                    },
                ],
                outputs: vec![
                    CoinRef {
                        coin_id: bob_coin_b,
                        amount: 900,
                        owner: carol.did(),
                    },
                    CoinRef {
                        coin_id: bob_coin_b,
                        amount: 700,
                        owner: alice.did(),
                    },
                ],
            },
            seq: 3,
            signature: [0u8; 64],
        });
        sign_and_submit(&nodes[2], &mut transfer, &bob);

        // Identity op mixed in.
        let mut id_op = ConsensusTransaction::IdentityOp(IdentityOp {
            did: carol.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: carol.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        sign_and_submit(&nodes[3], &mut id_op, &carol);

        // Burn carol's 900.
        let carol_coin = commit_coin(900, &carol.did(), &[0u8; 32]);
        let mut burn = ConsensusTransaction::CertificateOp(CertificateOp {
            did: carol.did(),
            kind: CertificateOpKind::Burn {
                inputs: vec![CoinRef {
                    coin_id: carol_coin,
                    amount: 900,
                    owner: carol.did(),
                }],
            },
            seq: 4,
            signature: [0u8; 64],
        });
        sign_and_submit(&nodes[4], &mut burn, &carol);

        // Every node replays the same log.
        for node in nodes.iter_mut() {
            node.sync().unwrap();
            assert!(node.is_synced());
        }

        // All five converge to the identical state.
        let r0 = nodes[0].certs().root();
        for node in &nodes[1..] {
            assert_eq!(
                node.certs().root(),
                r0,
                "all nodes share one certificate root"
            );
            assert_eq!(node.certs().total_supply(), 700, "supply: 1600 - 900 = 700");
            assert_eq!(node.applied_seq(), 5);
        }
        assert!(nodes[0].certs().utxo(&carol_coin).is_none(), "burned");
        assert!(nodes[0].certs().utxo(&bob_coin_a).is_none(), "spent");
        assert!(nodes[0].certs().utxo(&bob_coin_b).is_none(), "spent");
        // Alice's 700 change coin is live on every node.
        let change = commit_coin(700, &alice.did(), &[0u8; 32]);
        for node in &nodes {
            assert!(node.certs().utxo(&change).is_some(), "change coin live");
        }

        // A late-joining node syncs from the full log and agrees too.
        let mut late = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(authority.did()),
        );
        late.sync().unwrap();
        assert_eq!(late.applied_seq(), 5);
        assert_eq!(late.certs().root(), r0);
        assert!(late.certs().utxo(&change).is_some());
    }

    #[test]
    fn session_op_dispatch_locks_to_shared_consensus_ops() {
        use unfer_protocol::AgentRequest;

        // `apply_session_op` must support exactly the shared CONSENSUS_OPS
        // table; any op outside it is rejected with a diagnostic.
        let req = AgentRequest {
            id: "x".into(),
            op: "create_model".into(),
            params: serde_json::to_value(unfer_protocol::ModelSpec {
                hamiltonian: unfer_protocol::HamiltonianSpec::builtin(
                    "harmonic_chain",
                    serde_json::json!({"n_modes": 2, "omega": 1.0}),
                ),
                prior: unfer_protocol::PriorSpec::Vacuum,
                solver: unfer_protocol::SolverSpec::default(),
            })
            .unwrap(),
            provenance: None,
        };
        let mut node = ConsensusNode::new(Box::new(LocalConsensus::new()));
        assert!(node.apply_session_op(&req).is_ok());

        for op in unfer_protocol::ops::CONSENSUS_OPS {
            let req = AgentRequest {
                id: "x".into(),
                op: op.to_string(),
                params: serde_json::json!({}),
                provenance: None,
            };
            // create_model with `{}` params fails parsing (BAD_JSON) but must
            // still be a *supported* op (never the "unsupported session op" arm).
            let res = node.apply_session_op(&req);
            assert!(
                res.is_err(),
                "op '{op}' reported success without model params"
            );
        }
    }

    // ── H7: distributed-delivery protections ─────────────────────────────
    // Idempotency (exactly-once replay), leader lease (single firer per tick),
    // and the job queue (claim/unclaim/markFired) around the existing ledgers.

    #[test]
    fn double_submitted_transfer_applies_once_and_conservation_holds() {
        use crate::certs::commit_coin;
        use unfer_protocol::{CertificateOp, CertificateOpKind, CoinRef};

        // H7 acceptance: a duplicated delivery of the same transfer applies
        // exactly once; conservation (UK-7002) holds because the input is spent
        // exactly once.
        let engine = LocalConsensus::new();
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();

        let mut node = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(authority.did()),
        );

        let sign_and_submit =
            |node: &ConsensusNode, tx: &mut ConsensusTransaction, kp: &Keypair| {
                crate::signing::sign_transaction(tx, kp);
                node.submit(tx.clone()).unwrap();
            };

        // Mint 1000 to alice.
        let mut mint = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount: 1000,
                owner: alice.did(),
                blinding: [1u8; 32],
                source: None,
            },
            seq: 1,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut mint, &authority);

        // The exact same transfer submitted twice (duplicated delivery).
        let coin = commit_coin(1000, &alice.did(), &[1u8; 32]);
        let mut transfer = ConsensusTransaction::CertificateOp(CertificateOp {
            did: alice.did(),
            kind: CertificateOpKind::Transfer {
                inputs: vec![CoinRef {
                    coin_id: coin,
                    amount: 1000,
                    owner: alice.did(),
                }],
                outputs: vec![CoinRef {
                    coin_id: coin,
                    amount: 1000,
                    owner: bob.did(),
                }],
            },
            seq: 2,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut transfer, &alice);
        // Duplicate delivery of the same signed transfer.
        sign_and_submit(&node, &mut transfer, &alice);

        node.sync().unwrap();

        // Applied exactly once: the input is spent once, conservation holds.
        assert_eq!(node.certs().total_supply(), 1000, "UK-7002 conservation");
        assert!(
            node.certs().utxo(&coin).is_none(),
            "input spent exactly once"
        );
        assert_eq!(node.certs().unspent_count(), 1, "one output, not two");
        // The second delivery was recognized as committed.
        assert!(
            node.idempotency_committed(&transfer),
            "duplicated delivery must be recorded as committed"
        );
    }

    #[test]
    fn two_nodes_single_leader_fires_per_tick() {
        // H7 acceptance: exactly one node is the leader per tick; the
        // non-leader does not fire.
        let leaders0: Vec<bool> = (0..4)
            .map(|t| crate::lease::LeaderLease::is_leader(t, 2, 0))
            .collect();
        let leaders1: Vec<bool> = (0..4)
            .map(|t| crate::lease::LeaderLease::is_leader(t, 2, 1))
            .collect();
        for tick in 0..4 {
            assert!(
                leaders0[tick] ^ leaders1[tick],
                "exactly one leader fires per tick"
            );
        }
        // A lost lease stops firing without corrupting state: node 0 held tick
        // 0, but node 1 takes tick 1 — node 0 must stop.
        assert!(!crate::lease::LeaderLease::tick_held(0, 1, 2, 0));
    }

    #[test]
    fn failed_fire_is_unqueued_and_retried() {
        use crate::jobs::{JobQueue, JobState};

        // H7 acceptance: a job whose fire fails is re-queued (unclaimSlot) and
        // a later claim retries it; markFired only after a successful fire.
        let mut q = JobQueue::new();
        q.enqueue("settle-auction");
        let claim = q.claim("settle-auction", "worker-0").expect("claim");
        q.unclaim(&claim).expect("unclaim"); // fire failed → re-queue
        assert_eq!(q.state("settle-auction"), Some(JobState::Queued));
        let retry = q.claim("settle-auction", "worker-1").expect("retry");
        q.mark_fired(&retry).unwrap(); // fire succeeded
        assert_eq!(q.state("settle-auction"), Some(JobState::Fired));
    }

    // ── Math bond + probability market through the consensus node ─────────

    #[test]
    fn mathbond_market_consensus_roundtrip() {
        use crate::mathbond::compute_bond_id;
        use crate::mathbond_market::compute_pool_id;
        use unfer_protocol::{
            MarketOp, MarketOpKind, MathBondOp, MathBondOpKind, MathBondTrigger, NegRiskOutcome,
            OutcomeId,
        };

        let mut node = make_node();
        let sponsor = Keypair::generate();
        let investor = Keypair::generate();
        let researcher = Keypair::generate();
        let creator = Keypair::generate();
        let lp = Keypair::generate();

        let sign_and_submit =
            |node: &ConsensusNode, tx: &mut ConsensusTransaction, kp: &Keypair| {
                crate::signing::sign_transaction(tx, kp);
                node.submit(tx.clone()).unwrap();
            };

        // Issue the bond; maturity is at consensus seq 5.
        let trigger = MathBondTrigger {
            theorem: "P_eq_NP".to_string(),
            spec_hash: "deadbeef".to_string(),
            max_export_bytes: 1024,
            permitted_axioms: vec![],
            strict: false,
            nat_extension: false,
            string_extension: false,
        };
        let mut issue = ConsensusTransaction::MathBondOp(MathBondOp {
            did: sponsor.did(),
            kind: MathBondOpKind::Issue {
                trigger: trigger.clone(),
                principal: 10000,
                coupon_rate_bps: 500,
                maturity_seq: 5,
                researcher_did: researcher.did(),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut issue, &sponsor);
        node.sync().unwrap();
        let bond_id = compute_bond_id(&trigger, &sponsor.did(), 10000, 500, 5, &researcher.did());
        assert_eq!(
            node.mathbond().bond(&bond_id).unwrap().state,
            MathBondState::Issued
        );

        // Fund it.
        let mut invest = ConsensusTransaction::MathBondOp(MathBondOp {
            did: investor.did(),
            kind: MathBondOpKind::Invest {
                bond_id,
                amount: 10000,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut invest, &investor);
        node.sync().unwrap();
        assert_eq!(
            node.mathbond().bond(&bond_id).unwrap().state,
            MathBondState::Funded
        );

        // Open the NegRisk market for the bond and seed liquidity.
        let outcome_a = OutcomeId([1u8; 32]);
        let outcome_b = OutcomeId([2u8; 32]);
        let outcome_never = OutcomeId([3u8; 32]);
        let pool_id = compute_pool_id(&bond_id);
        let mut open = ConsensusTransaction::MarketOp(MarketOp {
            did: creator.did(),
            kind: MarketOpKind::OpenNegRisk {
                bond_id,
                outcomes: vec![
                    NegRiskOutcome {
                        outcome_id: outcome_a,
                        pool_id,
                        label: "by_2025".to_string(),
                        maturity_seq: 500,
                    },
                    NegRiskOutcome {
                        outcome_id: outcome_b,
                        pool_id,
                        label: "by_2026".to_string(),
                        maturity_seq: 1000,
                    },
                    NegRiskOutcome {
                        outcome_id: outcome_never,
                        pool_id,
                        label: "never".to_string(),
                        maturity_seq: u64::MAX,
                    },
                ],
                fee_bps: 300,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut open, &creator);
        node.sync().unwrap();

        let mut add = ConsensusTransaction::MarketOp(MarketOp {
            did: lp.did(),
            kind: MarketOpKind::AddLiquidity {
                pool_id,
                amount: 12000,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut add, &lp);
        node.sync().unwrap();
        assert_eq!(
            node.market().pool(&pool_id).unwrap().pool.total_reserve,
            12000
        );

        // Record maturity (consensus seq 5 >= maturity_seq 5).
        let mut mature = ConsensusTransaction::MathBondOp(MathBondOp {
            did: investor.did(),
            kind: MathBondOpKind::Mature { bond_id },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut mature, &investor);
        node.sync().unwrap();
        assert_eq!(
            node.mathbond().bond(&bond_id).unwrap().state,
            MathBondState::Matured
        );

        // Resolve with `None` (matured without a trigger): the never outcome
        // wins deterministically — no caller-chosen outcome.
        let mut resolve = ConsensusTransaction::MarketOp(MarketOp {
            did: creator.did(),
            kind: MarketOpKind::Resolve {
                pool_id,
                trigger_seq: None,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut resolve, &creator);
        node.sync().unwrap();
        assert_eq!(
            node.market().pool(&pool_id).unwrap().pool.winner,
            Some(outcome_never)
        );

        // The LP claims the whole reserve (nobody held winning tokens).
        let mut claim = ConsensusTransaction::MarketOp(MarketOp {
            did: lp.did(),
            kind: MarketOpKind::Claim { pool_id },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut claim, &lp);
        node.sync().unwrap();
        assert_eq!(
            node.market().pool(&pool_id).unwrap().lp_map.get(&lp.did()),
            Some(&0)
        );
    }

    #[test]
    fn market_resolve_forgery_rejected() {
        use crate::mathbond::compute_bond_id;
        use crate::mathbond_market::compute_pool_id;
        use unfer_protocol::{
            MarketOp, MarketOpKind, MathBondOp, MathBondOpKind, MathBondTrigger, NegRiskOutcome,
            OutcomeId,
        };

        let sign_and_submit =
            |node: &ConsensusNode, tx: &mut ConsensusTransaction, kp: &Keypair| {
                crate::signing::sign_transaction(tx, kp);
                node.submit(tx.clone()).unwrap();
            };

        // Scenario 1: the bond is still live — a forged resolve is refused
        // before it can touch the pool.
        let mut node = make_node();
        let sponsor = Keypair::generate();
        let creator = Keypair::generate();
        let trigger = MathBondTrigger {
            theorem: "RiemannHypothesis".to_string(),
            spec_hash: "cafe".to_string(),
            max_export_bytes: 1024,
            permitted_axioms: vec![],
            strict: false,
            nat_extension: false,
            string_extension: false,
        };
        let mut issue = ConsensusTransaction::MathBondOp(MathBondOp {
            did: sponsor.did(),
            kind: MathBondOpKind::Issue {
                trigger: trigger.clone(),
                principal: 1000,
                coupon_rate_bps: 0,
                maturity_seq: 1000,
                researcher_did: sponsor.did(),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut issue, &sponsor);
        let bond_id = compute_bond_id(&trigger, &sponsor.did(), 1000, 0, 1000, &sponsor.did());
        let pool_id = compute_pool_id(&bond_id);
        let mut open = ConsensusTransaction::MarketOp(MarketOp {
            did: creator.did(),
            kind: MarketOpKind::OpenNegRisk {
                bond_id,
                outcomes: vec![NegRiskOutcome {
                    outcome_id: OutcomeId([1u8; 32]),
                    pool_id,
                    label: "never".to_string(),
                    maturity_seq: u64::MAX,
                }],
                fee_bps: 0,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut open, &creator);
        node.sync().unwrap();

        let mut forged = ConsensusTransaction::MarketOp(MarketOp {
            did: creator.did(),
            kind: MarketOpKind::Resolve {
                pool_id,
                trigger_seq: Some(1),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut forged, &creator);
        // D05: refused at its slot with a typed outcome; the slot still advances
        // so the entries behind it are not held hostage by the forgery.
        let report = node.sync_report().unwrap();
        assert_eq!(report.refusals.len(), 1);
        assert_eq!(report.refusals[0].1.code, Code::MARKET_NOT_RESOLVED);
        assert!(
            !node.market().pool(&pool_id).unwrap().pool.resolved,
            "the forged resolution must not touch the pool"
        );

        // Scenario 2: the bond matured without a trigger (expected signal
        // None) — claiming a trigger fired is refused as a mismatch.
        let mut node2 = make_node();
        let sponsor2 = Keypair::generate();
        let creator2 = Keypair::generate();
        let trigger2 = MathBondTrigger {
            theorem: "Goldbach".to_string(),
            spec_hash: "beef".to_string(),
            max_export_bytes: 1024,
            permitted_axioms: vec![],
            strict: false,
            nat_extension: false,
            string_extension: false,
        };
        let mut issue2 = ConsensusTransaction::MathBondOp(MathBondOp {
            did: sponsor2.did(),
            kind: MathBondOpKind::Issue {
                trigger: trigger2.clone(),
                principal: 1000,
                coupon_rate_bps: 0,
                maturity_seq: 3,
                researcher_did: sponsor2.did(),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node2, &mut issue2, &sponsor2);
        let bond_id2 = compute_bond_id(&trigger2, &sponsor2.did(), 1000, 0, 3, &sponsor2.did());
        let pool_id2 = compute_pool_id(&bond_id2);
        let mut open2 = ConsensusTransaction::MarketOp(MarketOp {
            did: creator2.did(),
            kind: MarketOpKind::OpenNegRisk {
                bond_id: bond_id2,
                outcomes: vec![NegRiskOutcome {
                    outcome_id: OutcomeId([1u8; 32]),
                    pool_id: pool_id2,
                    label: "never".to_string(),
                    maturity_seq: u64::MAX,
                }],
                fee_bps: 0,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node2, &mut open2, &creator2);
        let mut mature2 = ConsensusTransaction::MathBondOp(MathBondOp {
            did: sponsor2.did(),
            kind: MathBondOpKind::Mature { bond_id: bond_id2 },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node2, &mut mature2, &sponsor2);
        node2.sync().unwrap();

        let mut forged2 = ConsensusTransaction::MarketOp(MarketOp {
            did: creator2.did(),
            kind: MarketOpKind::Resolve {
                pool_id: pool_id2,
                trigger_seq: Some(1),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node2, &mut forged2, &creator2);
        let report = node2.sync_report().unwrap();
        assert_eq!(report.refusals.len(), 1);
        assert_eq!(report.refusals[0].1.code, Code::MARKET_UNKNOWN_OUTCOME);
        assert!(!node2.market().pool(&pool_id2).unwrap().pool.resolved);
    }

    // ── D05: submission / commitment / application ──────────────────────

    /// A committed log served verbatim, so a test can put entries into the
    /// history that this node never submitted: a gap, a backwards slot, or a
    /// payload whose signature was stripped.
    struct ScriptedLog {
        log: std::sync::Mutex<Vec<(u64, ConsensusTransaction)>>,
    }

    impl ScriptedLog {
        fn new() -> Self {
            Self {
                log: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn push_after(&self, tx: ConsensusTransaction) -> u64 {
            let mut log = self.log.lock().unwrap();
            let seq = log.last().map(|(s, _)| s + 1).unwrap_or(1);
            log.push((seq, tx));
            seq
        }

        /// Append at an explicit slot, leaving the slots before it missing.
        fn push_at(&self, seq: u64, tx: ConsensusTransaction) {
            self.log.lock().unwrap().push((seq, tx));
        }
    }

    impl ConsensusEngine for ScriptedLog {
        fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic> {
            Ok(self.push_after(tx))
        }

        fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)> {
            let mut log = self
                .log
                .lock()
                .unwrap()
                .iter()
                .filter(|(seq, _)| *seq >= from_seq)
                .cloned()
                .collect::<Vec<_>>();
            log.sort_by_key(|(seq, _)| *seq);
            log
        }

        fn current_seq(&self) -> u64 {
            self.log
                .lock()
                .unwrap()
                .iter()
                .map(|(seq, _)| *seq)
                .max()
                .unwrap_or(0)
        }
    }

    fn signed_mint(
        authority: &Keypair,
        owner: &str,
        amount: u64,
        blinding: [u8; 32],
        seq: u64,
    ) -> ConsensusTransaction {
        use unfer_protocol::{CertificateOp, CertificateOpKind};
        let mut mint = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount,
                owner: owner.to_string(),
                blinding,
                source: None,
            },
            seq,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut mint, authority);
        mint
    }

    #[test]
    fn a_refused_entry_between_two_good_ones_does_not_wedge_the_suffix() {
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();
        let mut node = ConsensusNode::with_mint_authority(
            Box::new(LocalConsensus::new()),
            MintAuthority::Only(authority.did()),
        );

        // 1: authorized mint of 100 to alice.
        node.submit(signed_mint(&authority, &alice.did(), 100, [1u8; 32], 1))
            .unwrap();
        // 2: bob mints to himself -- a valid signature and a refused operation.
        node.submit(signed_mint(&bob, &bob.did(), 500, [2u8; 32], 1))
            .unwrap();
        // 3: authorized mint of 50 to alice, behind the bad entry.
        node.submit(signed_mint(&authority, &alice.did(), 50, [3u8; 32], 2))
            .unwrap();

        let report = node.sync_report().unwrap();
        assert_eq!(report.entries, 3, "the refused slot still counted");
        assert_eq!(report.watermark, Watermark(3));
        assert_eq!(node.watermark(), Watermark(3));
        assert_eq!(report.refusals.len(), 1);
        assert_eq!(report.refusals[0].0, Slot(2));
        assert_eq!(report.refusals[0].1.code, Code::CERT_MINT_NOT_AUTHORIZED);

        // The good suffix applied; the refused entry left nothing behind.
        assert_eq!(node.certs().total_supply(), 150);
        assert!(
            node.certs()
                .utxo(&crate::certs::commit_coin(100, &alice.did(), &[1u8; 32]))
                .is_some()
        );
        assert!(
            node.certs()
                .utxo(&crate::certs::commit_coin(50, &alice.did(), &[3u8; 32]))
                .is_some()
        );
        assert!(
            node.certs()
                .utxo(&crate::certs::commit_coin(500, &bob.did(), &[2u8; 32]))
                .is_none()
        );
    }

    #[test]
    fn a_refused_entry_leaves_the_ledger_fingerprint_unchanged() {
        let authority = Keypair::generate();
        let bob = Keypair::generate();
        let mut node = ConsensusNode::with_mint_authority(
            Box::new(LocalConsensus::new()),
            MintAuthority::Only(authority.did()),
        );
        node.submit(signed_mint(&authority, &bob.did(), 100, [1u8; 32], 1))
            .unwrap();
        node.sync().unwrap();
        let before = format!("{:?}", node.certs());

        node.submit(signed_mint(&bob, &bob.did(), 500, [2u8; 32], 2))
            .unwrap();
        let report = node.sync_report().unwrap();
        assert_eq!(report.refusals.len(), 1);

        // Whole-ledger fingerprint: the refusal is a property of the entry, so
        // it cannot have touched the state at all (D04's comparison, reused).
        assert_eq!(format!("{:?}", node.certs()), before);
        assert_eq!(node.certs().total_supply(), 100);
    }

    #[test]
    fn a_gap_in_committed_history_halts_without_applying_anything_past_it() {
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let log = ScriptedLog::new();
        log.push_at(1, signed_mint(&authority, &alice.did(), 100, [1u8; 32], 1));
        // Slot 2 is missing entirely; the next entry was committed at 3.
        log.push_at(3, signed_mint(&authority, &alice.did(), 50, [2u8; 32], 2));

        let mut node =
            ConsensusNode::with_mint_authority(Box::new(log), MintAuthority::Only(authority.did()));
        let err = node.sync().unwrap_err();
        assert_eq!(err.code, Code::UNKNOWN_OUTCOME);
        assert_eq!(
            node.halted(),
            Some(&HaltReason::Gap {
                expected: Slot(2),
                found: Slot(3),
            })
        );
        // The coherent prefix applied; nothing past the gap did.
        assert_eq!(node.watermark(), Watermark(1));
        assert_eq!(node.certs().total_supply(), 100);

        // A halted node does not skip the gap on the next attempt either.
        assert!(node.sync().is_err());
        assert_eq!(node.watermark(), Watermark(1));
        assert_eq!(node.certs().total_supply(), 100);
    }

    #[test]
    fn a_tampered_entry_is_refused_on_replay() {
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let mut mint = signed_mint(&authority, &alice.did(), 100, [1u8; 32], 1);
        // Tamper after signing: the submitter never saw this, the log did.
        if let ConsensusTransaction::CertificateOp(op) = &mut mint {
            op.signature[0] ^= 0xff;
        }
        let log = ScriptedLog::new();
        log.push_at(1, mint);

        let mut node =
            ConsensusNode::with_mint_authority(Box::new(log), MintAuthority::Only(authority.did()));
        let report = node.sync_report().unwrap();
        assert_eq!(report.refusals.len(), 1);
        assert_eq!(report.refusals[0].1.code, Code::INVALID_SIGNATURE);
        assert_eq!(node.watermark(), Watermark(1));
        assert_eq!(node.certs().total_supply(), 0);
    }

    #[test]
    fn settle_applies_the_entry_it_reports_as_committed() {
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let mut node = ConsensusNode::with_mint_authority(
            Box::new(LocalConsensus::new()),
            MintAuthority::Only(authority.did()),
        );
        let id = node
            .submit_tracked(signed_mint(&authority, &alice.did(), 100, [1u8; 32], 1))
            .unwrap();
        // Before settlement the entry is committed but not applied by this node.
        assert_eq!(node.applied_seq(), 0);
        let receipt = node.settle(id, Duration::from_millis(50));
        assert!(receipt.is_committed());
        assert_eq!(receipt.slot(), Some(Slot(id.0)));
        assert_eq!(
            node.certs().total_supply(),
            100,
            "a Committed receipt must mean the state already reflects it"
        );
    }

    #[test]
    fn settle_reports_a_refusal_rather_than_committed() {
        let alice = Keypair::generate();
        // Minting is disabled, so this committed entry is refused at slot 1.
        let mut node = ConsensusNode::new(Box::new(LocalConsensus::new()));
        let id = node
            .submit_tracked(signed_mint(&alice, &alice.did(), 100, [1u8; 32], 1))
            .unwrap();
        let receipt = node.settle(id, Duration::from_millis(50));
        assert!(!receipt.is_committed());
        assert_eq!(
            receipt.refusal().map(|r| r.code),
            Some(Code::CERT_MINT_NOT_AUTHORIZED)
        );
        assert_eq!(receipt.slot(), Some(Slot(1)));
        assert_eq!(node.certs().total_supply(), 0);
    }

    #[test]
    fn a_receipt_bounded_wait_reports_unknown_not_success() {
        // A local engine that has no record of the id cannot claim success.
        let mut node = ConsensusNode::new(Box::new(LocalConsensus::new()));
        let receipt = node.settle(SubmissionId(7), Duration::from_millis(10));
        assert!(!receipt.is_committed());
        assert_eq!(
            receipt.failure().map(|d| d.code),
            Some(Code::UNKNOWN_OUTCOME)
        );
    }

    fn journal_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir()
            .join(format!(
                "unfer-node-{}-{}-{name}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
            .join("node.log")
    }

    #[test]
    fn a_recorded_transition_survives_a_restart_and_is_compared_not_reapplied() {
        let authority = Keypair::generate();
        let path = journal_path("restart");
        let mints = [
            signed_mint(&authority, "did:unfer:bob", 100, [1u8; 32], 1),
            signed_mint(&authority, "did:unfer:bob", 50, [2u8; 32], 2),
        ];

        {
            let (mut node, recovery) =
                ConsensusNode::open(Box::new(LocalConsensus::new()), &path).unwrap();
            assert!(recovery.records.is_empty());
            node.set_mint_authority(MintAuthority::Only(authority.did()));
            node.engine.submit(mints[0].clone()).unwrap();
            node.engine.submit(mints[1].clone()).unwrap();
            assert_eq!(node.sync().unwrap(), 2);
            assert!(node.is_durable());
        }

        // Restart: a fresh engine carries the same committed entries, and the
        // recovered journal is a comparison script rather than a re-apply plan.
        let (mut node, recovery) =
            ConsensusNode::open(Box::new(LocalConsensus::new()), &path).unwrap();
        assert_eq!(recovery.last_slot, Some(2));
        node.set_mint_authority(MintAuthority::Only(authority.did()));
        for mint in &mints {
            node.engine.submit(mint.clone()).unwrap();
        }
        assert_eq!(node.sync().unwrap(), 2);
        assert_eq!(
            node.watermark(),
            Watermark(2),
            "replay reached the recorded watermark"
        );
        assert_eq!(
            node.certs.total_supply(),
            150,
            "state was rebuilt by replay"
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_mismatched_journal_halts_instead_of_merging_two_histories() {
        let authority = Keypair::generate();
        let path = journal_path("mismatch");
        let recorded = signed_mint(&authority, "did:unfer:bob", 100, [1u8; 32], 1);
        {
            let (mut node, _) =
                ConsensusNode::open(Box::new(LocalConsensus::new()), &path).unwrap();
            node.set_mint_authority(MintAuthority::Only(authority.did()));
            node.engine.submit(recorded.clone()).unwrap();
            node.sync().unwrap();
        }

        let different = signed_mint(&authority, "did:unfer:bob", 7, [9u8; 32], 1);
        let (mut node, _) = ConsensusNode::open(Box::new(LocalConsensus::new()), &path).unwrap();
        node.set_mint_authority(MintAuthority::Only(authority.did()));
        node.engine.submit(different).unwrap();
        let error = node.sync().expect_err("a mismatched checkpoint is refused");
        assert_eq!(error.code, Code::UNKNOWN_OUTCOME);
        match node.halted() {
            Some(HaltReason::Failed { slot, refusal }) => {
                assert_eq!(*slot, Slot(1));
                assert_eq!(refusal.code, Code::STORAGE_FAILED);
                assert!(refusal.message.contains("journal recorded"));
            }
            other => panic!("a mismatched journal must halt, got {other:?}"),
        }
        assert_eq!(node.watermark(), Watermark(0));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_torn_transition_write_halts_at_that_slot() {
        let authority = Keypair::generate();
        let path = journal_path("torn");
        let (mut node, _) = ConsensusNode::open(Box::new(LocalConsensus::new()), &path).unwrap();
        node.set_mint_authority(MintAuthority::Only(authority.did()));
        node.engine
            .submit(signed_mint(&authority, "did:unfer:bob", 100, [1u8; 32], 1))
            .unwrap();
        node.journal
            .as_mut()
            .unwrap()
            .inject(crate::storage::FaultPoint::HalfWrite);
        let error = node
            .sync()
            .expect_err("an unrecorded transition is not a success");
        assert_eq!(error.code, Code::UNKNOWN_OUTCOME);
        match node.halted() {
            Some(HaltReason::Failed { slot, refusal }) => {
                assert_eq!(*slot, Slot(1));
                assert_eq!(refusal.code, Code::STORAGE_FAILED);
            }
            other => panic!("a torn transition write must halt, got {other:?}"),
        }
        assert_eq!(
            node.watermark(),
            Watermark(0),
            "the watermark must not pass an unrecorded transition"
        );
        assert!(node.journal_error().is_some());
        // The next sync also refuses: a halted node does not skip history.
        assert!(node.sync().is_err());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn an_acknowledged_request_result_survives_a_restart() {
        let authority = Keypair::generate();
        let path = journal_path("request");
        let request = RequestId::new("federation-main", "did:unfer:alice", 1, 1);
        let tx = signed_mint(&authority, "did:unfer:bob", 100, [1u8; 32], 1);

        let submission = {
            let (mut node, _) =
                ConsensusNode::open(Box::new(LocalConsensus::new()), &path).unwrap();
            node.set_mint_authority(MintAuthority::Only(authority.did()));
            let submission = node.submit_request(&request, tx.clone()).unwrap();
            let settled = node.settle(submission, Duration::from_millis(100));
            assert!(settled.is_committed());
            submission
        };

        let (mut node, recovery) =
            ConsensusNode::open(Box::new(LocalConsensus::new()), &path).unwrap();
        assert!(
            recovery
                .records
                .iter()
                .any(|record| matches!(record, JournalRecord::Request(_))),
            "the request binding is in the durable log"
        );
        node.set_mint_authority(MintAuthority::Only(authority.did()));
        node.engine.submit(tx.clone()).unwrap();
        node.sync().unwrap();

        let retried = node
            .submit_request(&request, tx)
            .expect("the recovered book answers the retry");
        assert_eq!(
            retried, submission,
            "the retry reuses the recorded submission"
        );
        let receipt = node.settle(retried, Duration::from_millis(100));
        assert_eq!(receipt.slot(), Some(Slot(1)));
        assert_eq!(
            node.watermark(),
            Watermark(1),
            "the retry did not append a second entry"
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn an_unrecordable_request_result_is_reported_as_unrecorded_not_refused() {
        let authority = Keypair::generate();
        let path = journal_path("unrecorded");
        let (mut node, _) = ConsensusNode::open(Box::new(LocalConsensus::new()), &path).unwrap();
        node.set_mint_authority(MintAuthority::Only(authority.did()));
        let request = RequestId::new("federation-main", "did:unfer:alice", 1, 1);
        let submission = node
            .submit_request(
                &request,
                signed_mint(&authority, "did:unfer:bob", 100, [1u8; 32], 1),
            )
            .unwrap();
        // Record the transition first, so the injected failure hits the
        // *request binding* write and not the application transition itself.
        assert_eq!(node.sync().unwrap(), 1);
        node.journal
            .as_mut()
            .unwrap()
            .inject(crate::storage::FaultPoint::BeforeWrite);
        let receipt = node.settle(submission, Duration::from_millis(100));
        assert_eq!(
            receipt.state,
            ReceiptState::Unknown(UnknownReason::Unrecorded),
            "an applied but unrecorded result is never reported as a refusal"
        );
        assert!(node.journal_error().is_some());
        assert!(
            node.halted().is_none(),
            "a missing client binding is not a reason to stop applying history"
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_storage_failure_is_a_halt_and_a_domain_error_is_a_refusal() {
        // D07: the two failure kinds are decided by code family, so a durable
        // write failing inside application can never be filed as a stable
        // verdict on the entry it was applying.
        match ApplyFailure::from(
            3,
            Diagnostic::new(Code::STORAGE_FAILED, "disk full", Severity::Error),
        ) {
            ApplyFailure::Halt(HaltReason::Failed { slot, refusal }) => {
                assert_eq!(slot, Slot(3));
                assert_eq!(refusal.code, Code::STORAGE_FAILED);
            }
            other => panic!("a storage failure must halt, got {other:?}"),
        }
        match ApplyFailure::from(
            3,
            Diagnostic::new(Code::CERT_AMOUNT_MISMATCH, "bad", Severity::Error),
        ) {
            ApplyFailure::Refused(refusal) => assert_eq!(refusal.code, Code::CERT_AMOUNT_MISMATCH),
            other => panic!("a domain error must be refused, got {other:?}"),
        }
    }
}
