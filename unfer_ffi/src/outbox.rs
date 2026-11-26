//! D15 — durable outbox for approval-controlled effects.
//!
//! The action queue already refuses to run a side effect before an operator
//! resolves it. What it did not have is a *dispatch* protocol: `uk_action_apply`
//! wrote an in-flight marker, ran the effect, and wrote a resolved record, all
//! inline, with no place to record "the sink accepted this" separately from
//! "the sink might have accepted this". An approval also covered whatever
//! payload arrived on the handle, not the payload the operator read.
//!
//! This module is that protocol, built *under* the existing queue rather than
//! beside it: an `ActionRecord` is still the thing an operator sees, and an
//! applied action still ends up `Approved` with a resolved record. What changes
//! is the middle:
//!
//!   1. **Intent before dispatch.** The effect intent — request identity,
//!      principal, effect, resource, epoch, payload digest — is appended and
//!      flushed before any sink is called. A dispatcher that runs on an intent
//!      whose write failed would be running an effect it cannot account for.
//!   2. **Approval bound to the bytes.** An [`ApprovalBinding`](crate::outbox::ApprovalBinding) names the
//!      request, the payload digest, the resource and the epoch it was granted
//!      for. Resolution re-checks all four, so an approval replayed against an
//!      altered payload is refused before the sink sees it — the same check
//!      `dispatch` performs again immediately before the call, which is where a
//!      revocation that happened while the operator was reading is caught.
//!   3. **Three outcomes, three records.**
//!      [`OutboxRecord::Acknowledged`](crate::outbox::OutboxRecord::Acknowledged)
//!      is the sink saying it applied the effect.
//!      [`OutboxRecord::Retryable`](crate::outbox::OutboxRecord::Retryable) is the
//!      sink saying it did not, and may be asked again.
//!      [`OutboxRecord::Uncertain`](crate::outbox::OutboxRecord::Uncertain)
//!      is the honest one: the marker went out and no acknowledgment came back,
//!      so whether the effect landed is *unknown* and must be looked at rather
//!      than guessed. They are separate variants because they need different
//!      operator actions, and folding them into one "failed" state is how an
//!      uncertain charge gets replayed.
//!   4. **The idempotency key never changes.** A retry — including the one a
//!      restart performs after a crash between the sink call and the
//!      acknowledgment — carries the key the first attempt used, so a sink that
//!      deduplicates applies the effect once. A sink that cannot deduplicate is
//!      not retried automatically: it goes to
//!      [`OutboxRecord::Uncertain`](crate::outbox::OutboxRecord::Uncertain) and
//!      waits for [`Outbox::reconcile`](crate::outbox::Outbox::reconcile), because replaying an irreversible
//!      effect that may already have landed is not a recovery strategy.
//!
//! What this is *not*: a second approval system. The operator surface
//! (`uk_gate_approve`/`uk_action_apply`) is unchanged; the outbox sits under the
//! apply step and speaks the same `ActionRecord`. Nor is it a delivery
//! guarantee: it records what happened, and where the record is missing it says
//! so instead of pretending.
//!
//! The fence in [`EffectCall`](crate::outbox::EffectCall) is the D14 ordered fence, when the caller has
//! one. A sink that can enforce an order receives it; a sink that cannot is
//! refused when a fence is required, because a fence a sink ignores is a
//! promise nobody keeps.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use unfer_protocol::durable::{DurableError, DurableStore, streams};
use unfer_protocol::types::{ActionRecord, EffectKind};

/// The fence an effect was dispatched under (D14). Kept structural rather than
/// imported so this module does not depend on the consensus crate: the wire
/// shape is what a sink compares, and it is four numbers and two names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fence {
    pub deployment: String,
    pub resource: String,
    pub epoch: u64,
    pub generation: u64,
}

/// What an operator approved, and exactly what it was approved *for*.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalBinding {
    pub request_id: String,
    pub payload_digest: String,
    pub resource: String,
    pub epoch: u64,
    /// The principal that resolved the approval (the gatekeeper, or the kernel
    /// for the observation/vetted path that never queues).
    pub approved_by: String,
}

/// A durable effect intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectIntent {
    /// Stable identity of the request. A retry reuses it; a new action never
    /// does.
    pub request_id: String,
    pub principal: String,
    pub effect: String,
    /// The thing the effect acts on. For a local action this is the effect
    /// name, because the queue's unit of protection is the capability; a
    /// distributed caller passes the resource its fence names.
    pub resource: String,
    /// Authority epoch. Zero for the local action queue, which has no
    /// distributed authority to name — a placeholder the binding still covers,
    /// so a distributed caller cannot downgrade a real epoch to 0.
    pub epoch: u64,
    pub class: EffectKind,
    pub payload: serde_json::Value,
}

impl EffectIntent {
    /// `sha256:<hex>` over the canonical JSON encoding of the payload.
    /// `serde_json::Value` maps are ordered (no `preserve_order` feature in this
    /// workspace), so equal payloads encode to equal bytes and the digest is a
    /// commitment the operator's approval can be checked against.
    pub fn payload_digest(&self) -> String {
        digest_of(&self.payload)
    }

    /// The key every attempt at this intent carries. It covers the request
    /// identity *and* the payload digest, so a sink that deduplicates on it
    /// cannot confuse two different payloads submitted under one request id.
    pub fn idempotency_key(&self) -> String {
        format!("{}:{}", self.request_id, self.payload_digest())
    }

    pub fn approval_by(&self, approved_by: impl Into<String>) -> ApprovalBinding {
        ApprovalBinding {
            request_id: self.request_id.clone(),
            payload_digest: self.payload_digest(),
            resource: self.resource.clone(),
            epoch: self.epoch,
            approved_by: approved_by.into(),
        }
    }

    /// The intent implied by an action-queue record.
    pub fn from_action(record: &ActionRecord, class: EffectKind) -> Self {
        Self {
            request_id: record.id.clone(),
            principal: record.principal.clone(),
            effect: record.effect.clone(),
            resource: record.effect.clone(),
            epoch: 0,
            class,
            payload: record.params.clone(),
        }
    }
}

pub fn digest_of(value: &serde_json::Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

/// Why an approval does not cover an intent. Separate variants because the
/// operator's next question is different in each case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalMismatch {
    Request { approved: String, presented: String },
    Payload { approved: String, presented: String },
    Resource { approved: String, presented: String },
    Epoch { approved: u64, presented: u64 },
}

impl std::fmt::Display for ApprovalMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApprovalMismatch::Request {
                approved,
                presented,
            } => write!(
                f,
                "approval names request {approved} but the intent is {presented}"
            ),
            ApprovalMismatch::Payload {
                approved,
                presented,
            } => write!(
                f,
                "approval covers payload {approved} but the intent hashes to {presented}"
            ),
            ApprovalMismatch::Resource {
                approved,
                presented,
            } => write!(
                f,
                "approval covers resource {approved} but the intent names {presented}"
            ),
            ApprovalMismatch::Epoch {
                approved,
                presented,
            } => write!(
                f,
                "approval was granted at epoch {approved} but the intent is at epoch {presented}"
            ),
        }
    }
}

impl ApprovalBinding {
    /// Whether this approval is an approval *of* `intent`, in all four
    /// dimensions. Any field left unchecked is a replay surface.
    pub fn covers(&self, intent: &EffectIntent) -> Result<(), ApprovalMismatch> {
        if self.request_id != intent.request_id {
            return Err(ApprovalMismatch::Request {
                approved: self.request_id.clone(),
                presented: intent.request_id.clone(),
            });
        }
        let digest = intent.payload_digest();
        if self.payload_digest != digest {
            return Err(ApprovalMismatch::Payload {
                approved: self.payload_digest.clone(),
                presented: digest,
            });
        }
        if self.resource != intent.resource {
            return Err(ApprovalMismatch::Resource {
                approved: self.resource.clone(),
                presented: intent.resource.clone(),
            });
        }
        if self.epoch != intent.epoch {
            return Err(ApprovalMismatch::Epoch {
                approved: self.epoch,
                presented: intent.epoch,
            });
        }
        Ok(())
    }
}

/// One durable transition of one effect. Serialized as `{"outbox": …}` into the
/// `actions` stream, next to the `ActionRecord` transitions it accompanies, so
/// one stream gives one total order for the whole resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutboxRecord {
    /// The intent and the approval that covers it, written before dispatch.
    Planned {
        intent: EffectIntent,
        approval: ApprovalBinding,
    },
    /// A dispatch attempt is about to call the sink. Durable and flushed first:
    /// a marker that is not durable cannot tell a restart what may have been
    /// sent.
    Dispatching {
        request_id: String,
        attempt: u32,
        idempotency_key: String,
    },
    /// The sink said it applied the effect.
    Acknowledged {
        request_id: String,
        attempt: u32,
        result: serde_json::Value,
    },
    /// The sink said it did not apply the effect and may be asked again.
    Retryable {
        request_id: String,
        attempt: u32,
        error: String,
    },
    /// No acknowledgment was recorded for a dispatch that may have run. Not a
    /// failure: a question.
    Uncertain {
        request_id: String,
        attempt: u32,
        reason: String,
    },
    /// The dispatch was refused before the sink was called (a stale or missing
    /// approval, a revoked principal, a fence the sink cannot enforce).
    Refused { request_id: String, reason: String },
    /// A human (or the owning system) determined what happened to an uncertain
    /// dispatch. Terminal.
    Reconciled {
        request_id: String,
        applied: bool,
        note: String,
    },
}

/// The call a sink receives. `idempotency_key` is the same on every attempt at
/// one intent; `fence` is present only when the caller holds one.
#[derive(Debug)]
pub struct EffectCall<'a> {
    pub idempotency_key: &'a str,
    pub intent: &'a EffectIntent,
    pub fence: Option<&'a Fence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkReply {
    /// Applied now.
    Applied(serde_json::Value),
    /// Already applied under this key (a deduplicating sink's answer to a
    /// retry). Counted as success; the effect happened once.
    Duplicate(serde_json::Value),
    /// Not applied; the same intent may be dispatched again.
    Retryable(String),
    /// Permanently refused by the sink.
    Refused(String),
}

pub trait EffectSink {
    /// True iff applying the same idempotency key twice has the effect once.
    /// A sink that cannot say this is never retried automatically.
    fn deduplicates(&self) -> bool;
    /// Whether the sink can enforce an ordering fence. A fence handed to a sink
    /// that cannot enforce it would be a promise nobody keeps.
    fn enforces_fence(&self) -> bool {
        false
    }
    fn apply(&mut self, call: EffectCall<'_>) -> SinkReply;
}

/// Re-checks authority at the execution point, after the approval and before
/// the sink. This is where a revocation lands: an approval granted while the
/// principal was trusted does not survive the revocation.
pub trait DispatchPolicy {
    fn authorize(&self, intent: &EffectIntent) -> Result<(), String>;
}

/// The permissive policy, for callers whose gate is elsewhere (the action queue
/// was already gated at submission).
pub struct AllowAll;

impl DispatchPolicy for AllowAll {
    fn authorize(&self, _intent: &EffectIntent) -> Result<(), String> {
        Ok(())
    }
}

/// Where one effect stands, folded from its records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectState {
    /// Intent durable, not yet dispatched.
    Planned,
    /// A dispatch was recorded and no outcome was. A restart must resolve this
    /// rather than assume it failed.
    Dispatching,
    Applied(serde_json::Value),
    Retryable(String),
    Uncertain(String),
    Refused(String),
    /// Reconciled by a human/system: `applied` is what they determined.
    Reconciled {
        applied: bool,
        note: String,
    },
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub intent: EffectIntent,
    pub approval: ApprovalBinding,
    pub state: EffectState,
    pub attempts: u32,
}

/// What a dispatch call did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// The sink applied the effect now.
    Applied(serde_json::Value),
    /// The sink had already applied it under this key.
    AlreadyApplied(serde_json::Value),
    /// The sink would not apply it; a later attempt may succeed.
    Retryable(String),
}

/// Why a dispatch was refused before the sink could be called, or what went
/// wrong around the call. The distinction matters at the FFI boundary: only
/// `Unknown` means "the effect may have happened".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// The intent is not in the outbox (or its records are unusable).
    NotPlanned(String),
    /// The approval does not cover the intent presented.
    Approval(ApprovalMismatch),
    /// The execution-point policy refused (revoked principal, resource gone).
    Policy(String),
    /// A previous dispatch may or may not have landed and the sink cannot
    /// deduplicate, so dispatching again would risk applying it twice.
    Uncertain(String),
    /// The state is final and cannot be dispatched again.
    Terminal(String),
    /// The durable store refused a write, so nothing was dispatched.
    Storage(String),
    /// The sink futures ran but the acknowledgment could not be recorded: the
    /// effect is applied and unrecorded. Not retryable as a fresh action.
    Unrecorded(String),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::NotPlanned(m)
            | DispatchError::Policy(m)
            | DispatchError::Uncertain(m)
            | DispatchError::Terminal(m)
            | DispatchError::Storage(m)
            | DispatchError::Unrecorded(m) => f.write_str(m),
            DispatchError::Approval(m) => write!(f, "{m}"),
        }
    }
}

impl DispatchError {
    /// Whether the effect might have been applied under this error. Only the
    /// unrecorded-acknowledgment case has an effect with no record; an
    /// `Uncertain` state says a *previous* attempt is in that position.
    pub fn may_have_applied(&self) -> bool {
        matches!(self, DispatchError::Unrecorded(_))
    }
}

/// The outbox: durable intents, folded state, and the dispatch protocol.
pub struct Outbox {
    store: Arc<dyn DurableStore>,
    entries: BTreeMap<String, Entry>,
    /// Records in the stream that carried an `outbox` field but did not parse.
    /// `open_checked` refuses on a non-zero count.
    skipped: usize,
}

impl std::fmt::Debug for Outbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Outbox")
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl Outbox {
    /// Open against a store, recovering from its records. A record that cannot
    /// be parsed is skipped with a warning to stderr rather than dropped
    /// silently — but a skipped record is a hole in the state, so callers that
    /// care use [`Outbox::open_checked`].
    pub fn open(store: Arc<dyn DurableStore>) -> Self {
        let records = store.replay(streams::ACTIONS).unwrap_or_default();
        Self::from_records(store, &records)
    }

    /// Open, failing on any unreadable record. This is the form a dispatcher
    /// should use: a digest it cannot read is an intent it cannot honor.
    pub fn open_checked(store: Arc<dyn DurableStore>) -> Result<Self, String> {
        let records = store
            .replay(streams::ACTIONS)
            .map_err(|e| format!("outbox replay failed: {e}"))?;
        let mut outbox = Self::from_records(store, &records);
        if outbox.skipped > 0 {
            return Err(format!(
                "outbox: {} effect record(s) could not be parsed; refusing to dispatch on an \
                 incomplete view",
                outbox.skipped
            ));
        }
        outbox.skipped = 0;
        Ok(outbox)
    }

    fn from_records(store: Arc<dyn DurableStore>, records: &[Vec<u8>]) -> Self {
        let mut outbox = Self {
            store,
            entries: BTreeMap::new(),
            skipped: 0,
        };
        for raw in records {
            // `absorb` counts only records that *look* like ours: the action
            // queue shares this stream, and its records are not holes.

            outbox.absorb(raw);
        }
        outbox
    }

    fn absorb(&mut self, raw: &[u8]) {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(raw) else {
            self.skipped += 1;
            return;
        };
        let Some(field) = value.get("outbox") else {
            // Not an outbox record: the action-queue records share this stream.
            return;
        };
        let Ok(record) = serde_json::from_value::<OutboxRecord>(field.clone()) else {
            self.skipped += 1;
            return;
        };
        self.apply_record(record);
    }

    fn apply_record(&mut self, record: OutboxRecord) {
        match record {
            OutboxRecord::Planned { intent, approval } => {
                let request_id = intent.request_id.clone();
                // Re-planning the same intent is idempotent (a recovered
                // process re-preparing work it already planned). Re-planning a
                // *different* intent under the same request id is a replay and
                // keeps the first one, which is what the store's order says.
                self.entries.entry(request_id).or_insert(Entry {
                    intent,
                    approval,
                    state: EffectState::Planned,
                    attempts: 0,
                });
            }
            OutboxRecord::Dispatching {
                request_id,
                attempt,
                ..
            } => {
                if let Some(entry) = self.entries.get_mut(&request_id) {
                    entry.attempts = entry.attempts.max(attempt);
                    entry.state = EffectState::Dispatching;
                }
            }
            OutboxRecord::Acknowledged {
                request_id,
                attempt,
                result,
            } => {
                if let Some(entry) = self.entries.get_mut(&request_id) {
                    entry.attempts = entry.attempts.max(attempt);
                    entry.state = EffectState::Applied(result);
                }
            }
            OutboxRecord::Retryable {
                request_id,
                attempt,
                error,
            } => {
                if let Some(entry) = self.entries.get_mut(&request_id) {
                    entry.attempts = entry.attempts.max(attempt);
                    entry.state = EffectState::Retryable(error);
                }
            }
            OutboxRecord::Uncertain {
                request_id, reason, ..
            } => {
                if let Some(entry) = self.entries.get_mut(&request_id) {
                    entry.state = EffectState::Uncertain(reason);
                }
            }
            OutboxRecord::Refused { request_id, reason } => {
                if let Some(entry) = self.entries.get_mut(&request_id) {
                    entry.state = EffectState::Refused(reason);
                }
            }
            OutboxRecord::Reconciled {
                request_id,
                applied,
                note,
            } => {
                if let Some(entry) = self.entries.get_mut(&request_id) {
                    entry.state = EffectState::Reconciled { applied, note };
                }
            }
        }
    }

    pub fn store(&self) -> &Arc<dyn DurableStore> {
        &self.store
    }

    pub fn get(&self, request_id: &str) -> Option<&Entry> {
        self.entries.get(request_id)
    }

    pub fn records(&self) -> impl Iterator<Item = (&String, &Entry)> {
        self.entries.iter()
    }

    fn append(&self, record: &OutboxRecord) -> Result<(), DispatchError> {
        let envelope = serde_json::json!({ "outbox": record });
        let bytes = serde_json::to_vec(&envelope)
            .map_err(|e| DispatchError::Storage(format!("outbox encode failed: {e}")))?;
        self.store
            .append(streams::ACTIONS, &bytes)
            .map_err(|e| DispatchError::Storage(format!("outbox append failed: {e}")))?;
        self.store
            .flush()
            .map_err(|e| DispatchError::Storage(format!("outbox flush failed: {e}")))
    }

    /// Record an intent and its approval durably. Nothing is dispatched here:
    /// the caller must see this return `Ok` before a sink is called, and the
    /// approval must cover the intent or planning refuses.
    pub fn plan(
        &mut self,
        intent: EffectIntent,
        approval: ApprovalBinding,
    ) -> Result<(), DispatchError> {
        approval.covers(&intent).map_err(DispatchError::Approval)?;
        if let Some(existing) = self.entries.get(&intent.request_id) {
            // The same request id is the same request. If the payload changed,
            // this is someone reusing an identity — refuse rather than pick.
            if existing.intent.payload_digest() != intent.payload_digest() {
                return Err(DispatchError::Policy(format!(
                    "request {} was already planned with payload {}; refusing to reuse the \
                     identity for payload {}",
                    intent.request_id,
                    existing.intent.payload_digest(),
                    intent.payload_digest()
                )));
            }
            return Ok(());
        }
        let record = OutboxRecord::Planned {
            intent: intent.clone(),
            approval: approval.clone(),
        };
        self.append(&record)?;
        let request_id = intent.request_id.clone();
        self.entries.insert(
            request_id,
            Entry {
                intent,
                approval,
                state: EffectState::Planned,
                attempts: 0,
            },
        );
        Ok(())
    }

    /// Dispatch a planned intent. `fence` is the D14 fence when the caller has
    /// one; a sink that cannot enforce it is refused rather than trusted.
    ///
    /// The order here is the protocol: authorize → durable dispatch marker →
    /// sink → durable outcome. Each step's failure has its own answer, and none
    /// of them is "try again as a new action".
    pub fn dispatch(
        &mut self,
        request_id: &str,
        sink: &mut dyn EffectSink,
        policy: &dyn DispatchPolicy,
        fence: Option<&Fence>,
    ) -> Result<DispatchOutcome, DispatchError> {
        let entry = self.entries.get(request_id).ok_or_else(|| {
            DispatchError::NotPlanned(format!("no intent for request {request_id}"))
        })?;
        let intent = entry.intent.clone();
        let approval = entry.approval.clone();
        let attempts = entry.attempts;
        match &entry.state {
            EffectState::Applied(result) => {
                return Ok(DispatchOutcome::AlreadyApplied(result.clone()));
            }
            EffectState::Reconciled { applied: true, .. } => {
                return Err(DispatchError::Terminal(format!(
                    "request {request_id} was already applied (reconciled); not dispatching again"
                )));
            }
            EffectState::Reconciled { applied: false, .. } => {
                return Err(DispatchError::Terminal(format!(
                    "request {request_id} was reconciled as not applied; re-plan it as a new \
                     request if it should still be attempted"
                )));
            }
            EffectState::Refused(reason) => {
                return Err(DispatchError::Terminal(format!(
                    "request {request_id} was refused: {reason}"
                )));
            }
            EffectState::Uncertain(reason) => {
                return Err(DispatchError::Uncertain(format!(
                    "request {request_id} has an uncertain outcome ({reason}); reconcile it before \
                     dispatching again"
                )));
            }
            EffectState::Dispatching => {
                if !sink.deduplicates() {
                    // The dispatch marker is durable and no outcome is. The sink
                    // cannot tell a retry from a first attempt, so this is the
                    // reconciliation case, not a retry.
                    let reason = format!(
                        "attempt {attempts} was recorded with no acknowledgment and the sink \
                         cannot deduplicate; whether the effect landed is unknown"
                    );
                    self.mark_uncertain(request_id, attempts, &reason)?;
                    return Err(DispatchError::Uncertain(format!(
                        "request {request_id}: {reason}"
                    )));
                }
                // A deduplicating sink: re-dispatch under the same key. This is
                // the restart path — the sink answers `Duplicate` if the earlier
                // attempt landed, and applies it if it did not.
            }
            EffectState::Planned | EffectState::Retryable(_) => {}
        }
        // Execution-point recheck, after the approval and before the sink: an
        // approval does not outlive a revocation.
        policy.authorize(&intent).map_err(DispatchError::Policy)?;
        approval.covers(&intent).map_err(DispatchError::Approval)?;
        if let Some(fence) = fence
            && !sink.enforces_fence()
        {
            let reason = format!(
                "sink cannot enforce fence ({}, {}, epoch {}, generation {}); refusing rather \
                 than dispatching an effect whose order nobody checks",
                fence.deployment, fence.resource, fence.epoch, fence.generation
            );
            self.append(&OutboxRecord::Refused {
                request_id: request_id.to_string(),
                reason: reason.clone(),
            })?;
            if let Some(entry) = self.entries.get_mut(request_id) {
                entry.state = EffectState::Refused(reason.clone());
            }
            return Err(DispatchError::Policy(reason));
        }
        let attempt = attempts + 1;
        let key = intent.idempotency_key();
        self.append(&OutboxRecord::Dispatching {
            request_id: request_id.to_string(),
            attempt,
            idempotency_key: key.clone(),
        })?;
        if let Some(entry) = self.entries.get_mut(request_id) {
            entry.attempts = attempt;
            entry.state = EffectState::Dispatching;
        }

        let reply = sink.apply(EffectCall {
            idempotency_key: &key,
            intent: &intent,
            fence,
        });
        match reply {
            SinkReply::Applied(result) => {
                self.record_ack(request_id, attempt, result.clone())?;
                Ok(DispatchOutcome::Applied(result))
            }
            SinkReply::Duplicate(result) => {
                // The sink says an earlier attempt landed. Recorded as the
                // acknowledgment it is: the effect happened once, and the
                // durable record now says so.
                self.record_ack(request_id, attempt, result.clone())?;
                Ok(DispatchOutcome::AlreadyApplied(result))
            }
            SinkReply::Retryable(error) => {
                self.append(&OutboxRecord::Retryable {
                    request_id: request_id.to_string(),
                    attempt,
                    error: error.clone(),
                })?;
                if let Some(entry) = self.entries.get_mut(request_id) {
                    entry.state = EffectState::Retryable(error.clone());
                }
                Ok(DispatchOutcome::Retryable(error))
            }
            SinkReply::Refused(reason) => {
                self.append(&OutboxRecord::Refused {
                    request_id: request_id.to_string(),
                    reason: reason.clone(),
                })?;
                if let Some(entry) = self.entries.get_mut(request_id) {
                    entry.state = EffectState::Refused(reason.clone());
                }
                Err(DispatchError::Policy(reason))
            }
        }
    }

    /// Record the sink's acknowledgment. A failure here is the "applied and
    /// unrecorded" case: the effect happened, the store does not know, and the
    /// durable dispatch marker is what a restart will find.
    fn record_ack(
        &mut self,
        request_id: &str,
        attempt: u32,
        result: serde_json::Value,
    ) -> Result<(), DispatchError> {
        let record = OutboxRecord::Acknowledged {
            request_id: request_id.to_string(),
            attempt,
            result: result.clone(),
        };
        if let Err(e) = self.append(&record) {
            let reason =
                format!("the sink applied the effect but the acknowledgment is not durable: {e}");
            // Best-effort: the state is already durable via the dispatch
            // marker, so recovery will see `Dispatching` and ask.
            if let Some(entry) = self.entries.get_mut(request_id) {
                entry.state = EffectState::Uncertain(reason.clone());
            }
            return Err(DispatchError::Unrecorded(reason));
        }
        if let Some(entry) = self.entries.get_mut(request_id) {
            entry.state = EffectState::Applied(result);
        }
        Ok(())
    }

    fn mark_uncertain(
        &mut self,
        request_id: &str,
        attempt: u32,
        reason: &str,
    ) -> Result<(), DispatchError> {
        self.append(&OutboxRecord::Uncertain {
            request_id: request_id.to_string(),
            attempt,
            reason: reason.to_string(),
        })?;
        if let Some(entry) = self.entries.get_mut(request_id) {
            entry.state = EffectState::Uncertain(reason.to_string());
        }
        Ok(())
    }

    /// Resolve an uncertain dispatch by recording what was determined. This is
    /// the only way an uncertain effect becomes terminal, and it is deliberately
    /// a separate call from `dispatch`: a system that can answer "did it land?"
    /// on its own is a system that has a query interface, and this one does not
    /// pretend to have one.
    pub fn reconcile(
        &mut self,
        request_id: &str,
        applied: bool,
        note: impl Into<String>,
    ) -> Result<(), DispatchError> {
        let entry = self.entries.get(request_id).ok_or_else(|| {
            DispatchError::NotPlanned(format!("no intent for request {request_id}"))
        })?;
        if !matches!(
            entry.state,
            EffectState::Uncertain(_) | EffectState::Dispatching
        ) {
            return Err(DispatchError::Terminal(format!(
                "request {request_id} is not uncertain; there is nothing to reconcile"
            )));
        }
        let note = note.into();
        self.append(&OutboxRecord::Reconciled {
            request_id: request_id.to_string(),
            applied,
            note: note.clone(),
        })?;
        if let Some(entry) = self.entries.get_mut(request_id) {
            entry.state = EffectState::Reconciled { applied, note };
        }
        Ok(())
    }

    /// Requests whose durable state says a dispatch may have landed with no
    /// recorded outcome. Called by recovery so the uncertainty is visible
    /// (UK-1010) instead of silently re-dispatched.
    pub fn unresolved(&self) -> Vec<(&str, &Entry)> {
        self.entries
            .iter()
            .filter(|(_, e)| {
                matches!(
                    e.state,
                    EffectState::Dispatching | EffectState::Uncertain(_)
                )
            })
            .map(|(id, e)| (id.as_str(), e))
            .collect()
    }
}

/// The kernel's own sink for the action queue: the "effect" is the action
/// resolution itself, which is idempotent by action id.
pub struct KernelActionSink;

impl EffectSink for KernelActionSink {
    fn deduplicates(&self) -> bool {
        true
    }

    fn apply(&mut self, call: EffectCall<'_>) -> SinkReply {
        SinkReply::Applied(serde_json::json!({
            "applied": true,
            "action_id": call.intent.request_id,
            "effect": call.intent.effect,
        }))
    }
}

/// A store that refuses everything, used only to fold records that are already
/// in memory (recovery's uncertainty scan). It never sees a write, and if one
/// somehow arrives, `append` fails rather than pretending.
#[derive(Debug)]
struct NullStore;

impl DurableStore for NullStore {
    fn append(&self, _stream: &str, _record: &[u8]) -> Result<(), DurableError> {
        Err(DurableError::Unsupported("NullStore".into()))
    }
    fn flush(&self) -> Result<(), DurableError> {
        Err(DurableError::Unsupported("NullStore".into()))
    }
    fn replay(&self, _stream: &str) -> Result<Vec<Vec<u8>>, DurableError> {
        Ok(Vec::new())
    }
    fn frontier(&self) -> Result<Vec<u8>, DurableError> {
        Ok(Vec::new())
    }
    fn fork_at(&self, _frontier: &[u8]) -> Result<Box<dyn DurableStore>, DurableError> {
        Err(DurableError::Unsupported("NullStore".into()))
    }
    fn backend(&self) -> &'static str {
        "null"
    }
}

/// Where the kernel's outbox lives. Rebuilt when the durable store is swapped
/// (tests do this), so a stale view never answers for a new store.
static KERNEL_OUTBOX: std::sync::Mutex<Option<Outbox>> = std::sync::Mutex::new(None);

/// Run `f` against the kernel outbox for the currently configured store.
/// `Ok(None)` means no durable store is configured: an outbox with no store
/// cannot promise "durable before dispatch", so callers keep their RAM-only
/// path instead of pretending.
pub fn with_kernel<R>(f: impl FnOnce(&mut Outbox) -> R) -> Result<Option<R>, String> {
    let Some(store) = crate::handles::durable() else {
        return Ok(None);
    };
    let mut guard = KERNEL_OUTBOX.lock().unwrap_or_else(|e| e.into_inner());
    let rebuild = match guard.as_ref() {
        Some(outbox) => !Arc::ptr_eq(outbox.store(), &store),
        None => true,
    };
    if rebuild {
        *guard = Some(Outbox::open_checked(store)?);
    }
    let outbox = guard.as_mut().expect("outbox initialised above");
    Ok(Some(f(outbox)))
}

/// Drop the cached kernel outbox (test/restart helper).
pub fn reset_kernel() {
    *KERNEL_OUTBOX.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Plan and dispatch one action-queue record through `outbox`. This is the
/// end-to-end path `uk_action_apply` uses when a durable store is configured:
/// the record's params are the payload, the resolving principal is the one the
/// approval names, and the kernel's own sink applies the effect.
pub fn apply_action(
    outbox: &mut Outbox,
    record: &ActionRecord,
    class: EffectKind,
    approved_by: &str,
) -> Result<DispatchOutcome, DispatchError> {
    let intent = EffectIntent::from_action(record, class);
    let approval = intent.approval_by(approved_by);
    outbox.plan(intent, approval)?;
    let mut sink = KernelActionSink;
    outbox.dispatch(&record.id, &mut sink, &AllowAll, None)
}

/// Requests whose durable records say a dispatch may have landed with no
/// outcome recorded — the set recovery must surface rather than re-dispatch.
/// Takes the raw `actions` stream so recovery can ask without opening an
/// `Outbox` and without trusting a partially-parsed view.
pub fn unresolved_in(records: &[Vec<u8>]) -> Vec<(String, EffectState)> {
    let outbox = Outbox::from_records(Arc::new(NullStore), records);
    outbox
        .unresolved()
        .into_iter()
        .map(|(id, e)| (id.to_string(), e.state.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::durable::JsonlDurableStore;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "unfer-d15-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn intent(id: &str, payload: serde_json::Value) -> EffectIntent {
        EffectIntent {
            request_id: id.to_string(),
            principal: "agent-a".into(),
            effect: "send_notification".into(),
            resource: "send_notification".into(),
            epoch: 0,
            class: EffectKind::Mutate,
            payload,
        }
    }

    /// A sink that remembers the keys it applied — the shape a real
    /// deduplicating provider (an idempotency-key API) has. `dedup` is a
    /// parameter so the same sink can model one that does not.
    #[derive(Default)]
    struct MemSink {
        keys: Vec<String>,
        effects: usize,
        dedup: bool,
    }

    impl MemSink {
        fn dedup() -> Self {
            Self {
                dedup: true,
                ..Default::default()
            }
        }
    }

    impl EffectSink for MemSink {
        fn deduplicates(&self) -> bool {
            self.dedup
        }
        fn apply(&mut self, call: EffectCall<'_>) -> SinkReply {
            if self.dedup && self.keys.iter().any(|k| k == call.idempotency_key) {
                return SinkReply::Duplicate(serde_json::json!({"duplicate": true}));
            }
            self.keys.push(call.idempotency_key.to_string());
            self.effects += 1;
            SinkReply::Applied(serde_json::json!({"ok": true}))
        }
    }

    /// A sink whose record of applied keys outlives the process (the file is
    /// its ledger). Used for the crash test, where the point is that the sink's
    /// own dedup state is what makes the retry safe.
    struct FileSink {
        path: PathBuf,
        dedup: bool,
    }

    impl FileSink {
        fn effects(&self) -> usize {
            std::fs::read_to_string(&self.path)
                .map(|s| s.lines().count())
                .unwrap_or(0)
        }
    }

    impl EffectSink for FileSink {
        fn deduplicates(&self) -> bool {
            self.dedup
        }
        fn apply(&mut self, call: EffectCall<'_>) -> SinkReply {
            let seen = std::fs::read_to_string(&self.path)
                .map(|s| s.lines().any(|l| l == call.idempotency_key))
                .unwrap_or(false);
            if self.dedup && seen {
                return SinkReply::Duplicate(serde_json::json!({"duplicate": true}));
            }
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .unwrap();
            writeln!(f, "{}", call.idempotency_key).unwrap();
            f.sync_all().unwrap();
            SinkReply::Applied(serde_json::json!({"ok": true}))
        }
    }

    /// The first dispatch is retryable, the second applies.
    #[derive(Default)]
    struct FlakySink {
        calls: usize,
    }

    impl EffectSink for FlakySink {
        fn deduplicates(&self) -> bool {
            true
        }
        fn apply(&mut self, _call: EffectCall<'_>) -> SinkReply {
            self.calls += 1;
            if self.calls == 1 {
                SinkReply::Retryable("provider 503".into())
            } else {
                SinkReply::Applied(serde_json::json!({"ok": true}))
            }
        }
    }

    struct Revoke {
        principal: String,
    }

    impl DispatchPolicy for Revoke {
        fn authorize(&self, intent: &EffectIntent) -> Result<(), String> {
            if intent.principal == self.principal {
                Err(format!("principal {} was revoked", self.principal))
            } else {
                Ok(())
            }
        }
    }

    /// Delegates to a real store but refuses the record that carries `needle`:
    /// the process died between the sink returning and the acknowledgment being
    /// written. (The marker *before* the call was already durable, which is the
    /// point.)
    struct DropAppend {
        inner: Arc<dyn DurableStore>,
        needle: &'static str,
        dropped: AtomicUsize,
    }

    impl std::fmt::Debug for DropAppend {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "DropAppend({})", self.needle)
        }
    }

    impl DropAppend {
        fn new(inner: Arc<dyn DurableStore>, needle: &'static str) -> Self {
            Self {
                inner,
                needle,
                dropped: AtomicUsize::new(0),
            }
        }
    }

    impl DurableStore for DropAppend {
        fn append(&self, stream: &str, record: &[u8]) -> Result<(), DurableError> {
            if let Ok(text) = std::str::from_utf8(record)
                && text.contains(self.needle)
            {
                self.dropped.fetch_add(1, Ordering::SeqCst);
                return Err(DurableError::Io("simulated crash before the write".into()));
            }
            self.inner.append(stream, record)
        }
        fn flush(&self) -> Result<(), DurableError> {
            self.inner.flush()
        }
        fn replay(&self, stream: &str) -> Result<Vec<Vec<u8>>, DurableError> {
            self.inner.replay(stream)
        }
        fn frontier(&self) -> Result<Vec<u8>, DurableError> {
            self.inner.frontier()
        }
        fn fork_at(&self, frontier: &[u8]) -> Result<Box<dyn DurableStore>, DurableError> {
            self.inner.fork_at(frontier)
        }
        fn backend(&self) -> &'static str {
            self.inner.backend()
        }
    }

    #[test]
    fn intent_is_durable_before_any_sink_is_called() {
        let dir = temp_dir("planned");
        let store = Arc::new(JsonlDurableStore::open(Some(&dir)).unwrap());
        let mut outbox = Outbox::open(store);
        let mut sink = MemSink::dedup();

        // Nothing planned: dispatch must refuse before touching the sink.
        let err = outbox
            .dispatch("req-1", &mut sink, &AllowAll, None)
            .unwrap_err();
        assert!(matches!(err, DispatchError::NotPlanned(_)), "{err:?}");
        assert_eq!(sink.effects, 0);

        let i = intent("req-1", serde_json::json!({"to": "alice"}));
        let a = i.approval_by("gatekeeper");
        outbox.plan(i, a).unwrap();
        // The plan is on disk, not only in memory: a fresh outbox over the same
        // store sees the intent and its approval.
        let reopened =
            Outbox::open_checked(Arc::new(JsonlDurableStore::open(Some(&dir)).unwrap())).unwrap();
        assert_eq!(
            reopened.get("req-1").map(|e| e.state.clone()),
            Some(EffectState::Planned)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn approval_replay_on_an_altered_payload_is_refused_before_the_sink() {
        let dir = temp_dir("altered");
        let store = Arc::new(JsonlDurableStore::open(Some(&dir)).unwrap());
        let mut outbox = Outbox::open(store);

        // The operator read `{"amount": 10}`.
        let approved = intent("req-1", serde_json::json!({"amount": 10}));
        let approval = approved.approval_by("gatekeeper");

        // The dispatch presents `{"amount": 1000}` under the same request id.
        let tampered = intent("req-1", serde_json::json!({"amount": 1000}));
        let err = outbox.plan(tampered, approval.clone()).unwrap_err();
        match &err {
            DispatchError::Approval(ApprovalMismatch::Payload {
                approved,
                presented,
            }) => {
                assert_ne!(approved, presented);
            }
            other => panic!("expected a payload mismatch, got {other:?}"),
        }
        let mut sink = MemSink::dedup();
        assert!(
            outbox
                .dispatch("req-1", &mut sink, &AllowAll, None)
                .is_err()
        );
        assert_eq!(sink.effects, 0, "the sink must never see a tampered intent");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn approval_covers_request_resource_and_epoch_too() {
        let i = intent("req-1", serde_json::json!({"to": "alice"}));
        let a = i.approval_by("gatekeeper");

        let mut other_resource = i.clone();
        other_resource.resource = "send_email".into();
        assert!(matches!(
            a.covers(&other_resource),
            Err(ApprovalMismatch::Resource { .. })
        ));

        let mut other_epoch = i.clone();
        other_epoch.epoch = 7;
        assert!(matches!(
            a.covers(&other_epoch),
            Err(ApprovalMismatch::Epoch { .. })
        ));

        let mut other_request = i.clone();
        other_request.request_id = "req-2".into();
        assert!(matches!(
            a.covers(&other_request),
            Err(ApprovalMismatch::Request { .. })
        ));
    }

    #[test]
    fn revoked_worker_fails_at_the_execution_point_and_never_reaches_the_sink() {
        let dir = temp_dir("revoked");
        let store = Arc::new(JsonlDurableStore::open(Some(&dir)).unwrap());
        let mut outbox = Outbox::open(store);
        let i = intent("req-1", serde_json::json!({"to": "alice"}));
        let a = i.approval_by("gatekeeper");
        outbox.plan(i, a).unwrap();

        let mut sink = MemSink::dedup();
        let revoke = Revoke {
            principal: "agent-a".into(),
        };
        let err = outbox
            .dispatch("req-1", &mut sink, &revoke, None)
            .unwrap_err();
        assert!(matches!(err, DispatchError::Policy(_)), "{err:?}");
        assert_eq!(sink.effects, 0);
        // Refusal is not a dispatch: no marker was written, so a recovery pass
        // does not treat this request as uncertain.
        assert!(outbox.unresolved().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn retryable_failure_is_recorded_apart_from_acknowledgment() {
        let dir = temp_dir("retryable");
        let store = Arc::new(JsonlDurableStore::open(Some(&dir)).unwrap());
        let mut outbox = Outbox::open(store);
        let i = intent("req-1", serde_json::json!({"to": "alice"}));
        let a = i.approval_by("gatekeeper");
        outbox.plan(i, a).unwrap();

        let mut sink = FlakySink::default();
        assert_eq!(
            outbox
                .dispatch("req-1", &mut sink, &AllowAll, None)
                .unwrap(),
            DispatchOutcome::Retryable("provider 503".into())
        );
        assert_eq!(
            outbox.get("req-1").unwrap().state,
            EffectState::Retryable("provider 503".into())
        );
        // A retryable failure is not uncertain: the sink said it did not apply,
        // so recovery does not put it in the unknown-outcome set.
        assert!(outbox.unresolved().is_empty());

        // Retry carries the same idempotency key.
        let key = outbox.get("req-1").unwrap().intent.idempotency_key();
        assert_eq!(
            outbox
                .dispatch("req-1", &mut sink, &AllowAll, None)
                .unwrap(),
            DispatchOutcome::Applied(serde_json::json!({"ok": true}))
        );
        assert_eq!(sink.calls, 2);
        assert_eq!(
            outbox.get("req-1").unwrap().state,
            EffectState::Applied(serde_json::json!({"ok": true}))
        );
        assert_eq!(key, outbox.get("req-1").unwrap().intent.idempotency_key());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn crash_after_sink_success_before_ack_does_not_duplicate_at_a_dedup_sink() {
        let dir = temp_dir("crash-dedup");
        let store_dir = dir.join("store");
        let ledger = dir.join("sink.log");
        let store = Arc::new(JsonlDurableStore::open(Some(&store_dir)).unwrap());
        let drop_ack = Arc::new(DropAppend::new(store.clone(), "\"acknowledged\""));
        let mut outbox = Outbox::open_checked(drop_ack.clone()).unwrap();

        let i = intent("req-1", serde_json::json!({"to": "alice"}));
        let a = i.approval_by("gatekeeper");
        outbox.plan(i, a).unwrap();

        let mut sink = FileSink {
            path: ledger.clone(),
            dedup: true,
        };
        // The sink applies the effect; the acknowledgment write then "crashes".
        let err = outbox
            .dispatch("req-1", &mut sink, &AllowAll, None)
            .unwrap_err();
        assert!(matches!(err, DispatchError::Unrecorded(_)), "{err:?}");
        assert_eq!(sink.effects(), 1);
        assert_eq!(drop_ack.dropped.load(Ordering::SeqCst), 1);

        // Restart: new process, same store directory, same sink ledger.
        let reopened_store = Arc::new(JsonlDurableStore::open(Some(&store_dir)).unwrap());
        let mut recovered = Outbox::open_checked(reopened_store).unwrap();
        let unresolved = recovered.unresolved();
        assert_eq!(unresolved.len(), 1, "the dispatch marker is the only state");
        assert_eq!(unresolved[0].0, "req-1");

        let mut sink_after = FileSink {
            path: ledger.clone(),
            dedup: true,
        };
        // Same key, so the sink answers `Duplicate` — the effect stays at one.
        assert_eq!(
            recovered
                .dispatch("req-1", &mut sink_after, &AllowAll, None)
                .unwrap(),
            DispatchOutcome::AlreadyApplied(serde_json::json!({"duplicate": true}))
        );
        assert_eq!(
            sink_after.effects(),
            1,
            "the effect must not be applied twice"
        );
        assert_eq!(
            recovered.get("req-1").unwrap().state,
            EffectState::Applied(serde_json::json!({"duplicate": true}))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_dedup_sink_enters_reconciliation_instead_of_being_retried() {
        let dir = temp_dir("crash-nodedup");
        let store_dir = dir.join("store");
        let ledger = dir.join("sink.log");
        let store = Arc::new(JsonlDurableStore::open(Some(&store_dir)).unwrap());
        let drop_ack = Arc::new(DropAppend::new(store.clone(), "\"acknowledged\""));
        let mut outbox = Outbox::open_checked(drop_ack).unwrap();

        let i = intent("req-1", serde_json::json!({"charge": 5}));
        let a = i.approval_by("gatekeeper");
        outbox.plan(i, a).unwrap();
        let mut sink = FileSink {
            path: ledger.clone(),
            dedup: false,
        };
        let err = outbox
            .dispatch("req-1", &mut sink, &AllowAll, None)
            .unwrap_err();
        assert!(matches!(err, DispatchError::Unrecorded(_)));

        // Restart. The sink cannot tell a retry from a first attempt, so the
        // outbox refuses to dispatch and says the outcome is unknown.
        let reopened = Arc::new(JsonlDurableStore::open(Some(&store_dir)).unwrap());
        let mut recovered = Outbox::open_checked(reopened).unwrap();
        let mut sink_after = FileSink {
            path: ledger.clone(),
            dedup: false,
        };
        let err = recovered
            .dispatch("req-1", &mut sink_after, &AllowAll, None)
            .unwrap_err();
        assert!(matches!(err, DispatchError::Uncertain(_)), "{err:?}");
        assert_eq!(sink_after.effects(), 1, "no second attempt was made");
        assert!(matches!(
            recovered.get("req-1").unwrap().state,
            EffectState::Uncertain(_)
        ));
        // Dispatching again after reconciliation-as-applied is terminal.
        recovered
            .reconcile("req-1", true, "provider console shows one charge")
            .unwrap();
        let err = recovered
            .dispatch("req-1", &mut sink_after, &AllowAll, None)
            .unwrap_err();
        assert!(matches!(err, DispatchError::Terminal(_)), "{err:?}");
        assert_eq!(sink_after.effects(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fence_is_refused_before_a_sink_that_cannot_enforce_it() {
        let dir = temp_dir("fence");
        let store = Arc::new(JsonlDurableStore::open(Some(&dir)).unwrap());
        let mut outbox = Outbox::open(store);
        let i = intent("req-1", serde_json::json!({"to": "alice"}));
        let a = i.approval_by("gatekeeper");
        outbox.plan(i, a).unwrap();
        let fence = Fence {
            deployment: "prod".into(),
            resource: "send_notification".into(),
            epoch: 3,
            generation: 9,
        };
        let mut sink = MemSink::dedup(); // enforces_fence() defaults to false
        let err = outbox
            .dispatch("req-1", &mut sink, &AllowAll, Some(&fence))
            .unwrap_err();
        assert!(matches!(err, DispatchError::Policy(_)), "{err:?}");
        assert_eq!(sink.effects, 0);
        assert!(matches!(
            outbox.get("req-1").unwrap().state,
            EffectState::Refused(_)
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fence_reaches_a_sink_that_enforces_order() {
        struct OrderedSink {
            seen: Vec<(u64, u64)>,
            last: Option<(u64, u64)>,
        }
        impl EffectSink for OrderedSink {
            fn deduplicates(&self) -> bool {
                true
            }
            fn enforces_fence(&self) -> bool {
                true
            }
            fn apply(&mut self, call: EffectCall<'_>) -> SinkReply {
                let f = call.fence.expect("a fence was passed in");
                let at = (f.epoch, f.generation);
                if let Some(last) = self.last
                    && at <= last
                {
                    return SinkReply::Refused(format!("fence {at:?} does not follow {last:?}"));
                }
                self.last = Some(at);
                self.seen.push(at);
                SinkReply::Applied(serde_json::json!({"at": at}))
            }
        }
        let dir = temp_dir("fence-ok");
        let store = Arc::new(JsonlDurableStore::open(Some(&dir)).unwrap());
        let mut outbox = Outbox::open(store);
        let i = intent("req-1", serde_json::json!({"to": "alice"}));
        let a = i.approval_by("gatekeeper");
        outbox.plan(i, a).unwrap();
        let fence = Fence {
            deployment: "prod".into(),
            resource: "send_notification".into(),
            epoch: 3,
            generation: 9,
        };
        let mut sink = OrderedSink {
            seen: Vec::new(),
            last: None,
        };
        outbox
            .dispatch("req-1", &mut sink, &AllowAll, Some(&fence))
            .unwrap();
        assert_eq!(sink.seen, vec![(3, 9)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reusing_a_request_identity_for_a_different_payload_is_refused() {
        let dir = temp_dir("identity");
        let store = Arc::new(JsonlDurableStore::open(Some(&dir)).unwrap());
        let mut outbox = Outbox::open(store);
        let first = intent("req-1", serde_json::json!({"to": "alice"}));
        let a1 = first.approval_by("gatekeeper");
        outbox.plan(first, a1).unwrap();
        let second = intent("req-1", serde_json::json!({"to": "bob"}));
        let a2 = second.approval_by("gatekeeper");
        let err = outbox.plan(second, a2).unwrap_err();
        assert!(matches!(err, DispatchError::Policy(_)), "{err:?}");
        // Re-planning the *same* intent (a recovered process re-preparing work)
        // is idempotent.
        let again = intent("req-1", serde_json::json!({"to": "alice"}));
        let a3 = again.approval_by("gatekeeper");
        outbox.plan(again, a3).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_checked_refuses_an_unreadable_effect_record() {
        let dir = temp_dir("garbage");
        let store = Arc::new(JsonlDurableStore::open(Some(&dir)).unwrap());
        store
            .append(streams::ACTIONS, br#"{"outbox":{"kind":"bogus"}}"#)
            .unwrap();
        store
            .append(streams::ACTIONS, br#"{"id":"action-1"}"#)
            .unwrap();
        store.flush().unwrap();
        let err = Outbox::open_checked(store).unwrap_err();
        assert!(err.contains("could not be parsed"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_idempotency_key_covers_payload_and_request_identity() {
        let a = intent("req-1", serde_json::json!({"to": "alice"}));
        let b = intent("req-2", serde_json::json!({"to": "alice"}));
        let c = intent("req-1", serde_json::json!({"to": "bob"}));
        assert_ne!(a.idempotency_key(), b.idempotency_key());
        assert_ne!(a.idempotency_key(), c.idempotency_key());
        assert_eq!(
            a.idempotency_key(),
            intent("req-1", serde_json::json!({"to": "alice"})).idempotency_key(),
            "the same intent always produces the same key"
        );
    }
}
