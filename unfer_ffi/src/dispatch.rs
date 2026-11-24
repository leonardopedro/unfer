//! D15: the durable effect dispatcher.
//!
//! An approval-controlled effect is not fired inline. The intent is first
//! **committed** to the durable `dispatches` stream, the approval is checked
//! against the exact thing it authorised (request identity, resource, epoch and
//! payload digest), the policy is rechecked at the execution point, and only
//! then does a sink receive the effect — under one idempotency key that every
//! retry reuses.
//!
//! The outcome is recorded as one of three distinct durable states, never
//! collapsed into "done":
//!
//! - [`DispatchState::Acknowledged`] — the sink confirmed the effect;
//! - [`DispatchState::Retryable`] — the sink explicitly did not perform it;
//! - [`DispatchState::Uncertain`] — the sink may have performed it.
//!
//! A retry is allowed from `Retryable` against any sink, and from `Uncertain`
//! **only** against a sink that deduplicates by idempotency key. An uncertain
//! irreversible action against a non-deduplicating sink moves to
//! [`DispatchState::Reconciling`] and is never replayed as a fresh action; an
//! operator has to reconcile it outside this module.
//!
//! This module introduces no parallel approval system: it consumes the record
//! the existing action queue already persists, and the approval it checks is
//! derived from that record. The queue remains the authority on who may act.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use unfer_protocol::durable::{streams, DurableStore};
use unfer_protocol::types::EffectKind;

/// A lowercase hex sha256 digest.
pub type Digest = String;

fn sha256_hex(parts: &[&[u8]]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"unfer.dispatch.v1\0");
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hex::encode(hasher.finalize())
}

/// The exact effect an approval authorises.
///
/// `payload_digest` is computed from the payload at construction, so altering
/// the payload without re-approval cannot present the same approval. The
/// identity fields (`request_id`, `resource`, `epoch`) are part of the
/// authorisation too: an approval for one request never authorises another.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectIntent {
    pub request_id: String,
    pub principal: String,
    pub resource: String,
    pub epoch: u64,
    pub effect: String,
    pub kind: EffectKind,
    pub payload: serde_json::Value,
    pub payload_digest: Digest,
}

impl EffectIntent {
    pub fn new(
        request_id: impl Into<String>,
        principal: impl Into<String>,
        resource: impl Into<String>,
        epoch: u64,
        effect: impl Into<String>,
        kind: EffectKind,
        payload: serde_json::Value,
    ) -> Self {
        let digest = payload_digest(&payload);
        Self {
            request_id: request_id.into(),
            principal: principal.into(),
            resource: resource.into(),
            epoch,
            effect: effect.into(),
            kind,
            payload,
            payload_digest: digest,
        }
    }

    /// The approval that authorises exactly this intent. A caller holding any
    /// different payload/resource/epoch/request identity cannot reuse it.
    pub fn approval(&self) -> Approval {
        Approval {
            request_id: self.request_id.clone(),
            resource: self.resource.clone(),
            epoch: self.epoch,
            payload_digest: self.payload_digest.clone(),
        }
    }

    /// The idempotency key the sink sees. Derived from the intent's identity,
    /// not from an attempt counter, so every retry presents the same key.
    pub fn idempotency_key(&self) -> String {
        sha256_hex(&[
            self.request_id.as_bytes(),
            self.resource.as_bytes(),
            &self.epoch.to_be_bytes(),
            self.payload_digest.as_bytes(),
            self.effect.as_bytes(),
        ])
    }
}

/// The digest of a payload as the dispatcher binds it.
fn payload_digest(payload: &serde_json::Value) -> Digest {
    match serde_json::to_vec(payload) {
        Ok(bytes) => sha256_hex(&[&bytes]),
        // An unserialisable payload cannot be authorised: the digest of the
        // error text is deliberately unlike any real payload's digest, and
        // `Dispatcher::commit` refuses it before a sink could see it.
        Err(error) => sha256_hex(&[b"unserialisable", error.to_string().as_bytes()]),
    }
}

/// What an operator approves: the intent's identity plus its payload digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Approval {
    pub request_id: String,
    pub resource: String,
    pub epoch: u64,
    pub payload_digest: Digest,
}

impl Approval {
    /// Whether this approval authorises `intent` exactly.
    pub fn binds(&self, intent: &EffectIntent) -> bool {
        self.request_id == intent.request_id
            && self.resource == intent.resource
            && self.epoch == intent.epoch
            && self.payload_digest == intent.payload_digest
    }
}

/// What a sink reports. There is no "maybe" beyond these three.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SinkOutcome {
    /// The effect happened; the value is the sink's result.
    Applied(serde_json::Value),
    /// The sink did not perform the effect and it is safe to try again.
    Retryable(String),
    /// The sink may or may not have performed the effect.
    Uncertain(String),
}

/// A sink that can receive an effect. `dedups` is the sink's contract, not a
/// hint: a sink that returns `true` must apply one effect per idempotency key
/// even if the same key arrives more than once.
pub trait EffectSink: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn dedups(&self) -> bool;
    fn dispatch(&self, idempotency_key: &str, intent: &EffectIntent) -> SinkOutcome;
}

/// The durable lifecycle of one intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DispatchState {
    /// Persisted; the sink has not been called yet (or the process died before
    /// recording an outcome — see [`Dispatcher::resume`]).
    Committed,
    /// The sink acknowledged the effect.
    Acknowledged,
    /// The sink refused and the effect is known not to have happened.
    Retryable,
    /// The sink's outcome is unknown, but the sink deduplicates, so the same
    /// idempotency key can safely be presented again.
    Uncertain,
    /// The sink's outcome is unknown and it does not deduplicate. Never replayed
    /// as a fresh action; an operator must reconcile it.
    Reconciling,
    /// The policy/revocation check refused the effect at the execution point.
    Refused,
}

impl DispatchState {
    fn is_terminal_for_retry(self) -> bool {
        matches!(
            self,
            DispatchState::Acknowledged
                | DispatchState::Reconciling
                | DispatchState::Refused
        )
    }
}

/// One durable transition of an intent.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DispatchRecord {
    /// Monotonic transition sequence (wall-clock free).
    seq: u64,
    request_id: String,
    state: DispatchState,
    /// Present on the `Committed` record, which is what an operator reviews and
    /// what a retry after a restart re-presents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    intent: Option<EffectIntent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    detail: Option<serde_json::Value>,
}

struct Entry {
    intent: EffectIntent,
    state: DispatchState,
    key: String,
    seq: u64,
}

/// The dispatcher: a durable log of intents and outcomes, one entry per
/// request identity.
pub struct Dispatcher {
    store: Arc<dyn DurableStore>,
    entries: Mutex<BTreeMap<String, Entry>>,
    seq: Mutex<u64>,
}

/// Why a dispatch was refused before the sink ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// No such committed request.
    UnknownRequest,
    /// The approval does not bind this intent's identity/digest/epoch.
    ApprovalMismatch,
    /// The policy (e.g. delegation, revocation) refused at the execution point.
    Refused(String),
    /// The state does not allow this operation (already applied, reconciling,
    /// or never committed).
    WrongState(DispatchState),
    /// The durable store refused the record. Nothing was dispatched.
    Storage(String),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::UnknownRequest => write!(f, "no committed effect intent for that request"),
            DispatchError::ApprovalMismatch => write!(
                f,
                "the approval does not bind this request/resource/epoch/payload digest"
            ),
            DispatchError::Refused(reason) => write!(f, "policy refused at the execution point: {reason}"),
            DispatchError::WrongState(state) => write!(f, "effect is in state {state:?}"),
            DispatchError::Storage(error) => write!(f, "durable store refused the record: {error}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl Dispatcher {
    /// Open the dispatcher over a durable store and replay the `dispatches`
    /// stream. A record that cannot be read is an error: recovery must not
    /// silently forget an intent that may have fired.
    pub fn open(store: Arc<dyn DurableStore>) -> Result<Self, String> {
        let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
        let mut seq = 0u64;
        for raw in store
            .replay(streams::DISPATCH)
            .map_err(|error| format!("could not replay {}: {error}", streams::DISPATCH))?
        {
            let record: DispatchRecord = serde_json::from_slice(&raw)
                .map_err(|error| format!("corrupt dispatch record: {error}"))?;
            seq = seq.max(record.seq + 1);
            match record.intent {
                Some(intent) => {
                    let key = intent.idempotency_key();
                    entries.insert(
                        record.request_id,
                        Entry {
                            intent,
                            state: record.state,
                            key,
                            seq: record.seq,
                        },
                    );
                }
                None => {
                    if let Some(entry) = entries.get_mut(&record.request_id) {
                        entry.state = record.state;
                        entry.seq = record.seq;
                    }
                }
            }
        }
        Ok(Self {
            store,
            entries: Mutex::new(entries),
            seq: Mutex::new(seq),
        })
    }

    /// The durable state of a request, if it was ever committed.
    pub fn state(&self, request_id: &str) -> Option<DispatchState> {
        self.entries
            .lock()
            .unwrap()
            .get(request_id)
            .map(|entry| entry.state)
    }

    /// The idempotency key a request was committed under.
    pub fn idempotency_key(&self, request_id: &str) -> Option<String> {
        self.entries
            .lock()
            .unwrap()
            .get(request_id)
            .map(|entry| entry.key.clone())
    }

    fn append(&self, record: DispatchRecord) -> Result<(), DispatchError> {
        let bytes = serde_json::to_vec(&record)
            .map_err(|error| DispatchError::Storage(error.to_string()))?;
        self.store
            .append(streams::DISPATCH, &bytes)
            .map_err(|error| DispatchError::Storage(error.to_string()))?;
        self.store
            .flush()
            .map_err(|error| DispatchError::Storage(error.to_string()))
    }

    fn next_seq(&self) -> u64 {
        let mut seq = self.seq.lock().unwrap();
        let value = *seq;
        *seq += 1;
        value
    }

    fn record_outcome(
        &self,
        request_id: &str,
        state: DispatchState,
        detail: Option<serde_json::Value>,
    ) -> Result<(), DispatchError> {
        let seq = self.next_seq();
        self.append(DispatchRecord {
            seq,
            request_id: request_id.to_string(),
            state,
            intent: None,
            detail,
        })?;
        let mut entries = self.entries.lock().unwrap();
        if let Some(entry) = entries.get_mut(request_id) {
            entry.state = state;
            entry.seq = seq;
        }
        Ok(())
    }

    /// Durably commit an intent. Must succeed before any sink sees the effect.
    ///
    /// Committing the same request identity with the same digest is idempotent
    /// (a retry of the *submission*); the same identity with a different digest
    /// is a refusal — one request identity never authorises two payloads.
    pub fn commit(&self, intent: &EffectIntent) -> Result<(), DispatchError> {
        {
            let entries = self.entries.lock().unwrap();
            if let Some(existing) = entries.get(&intent.request_id) {
                if existing.intent.payload_digest == intent.payload_digest {
                    return Ok(());
                }
                return Err(DispatchError::Refused(format!(
                    "request {} is already committed with a different payload digest",
                    intent.request_id
                )));
            }
        }
        let seq = self.next_seq();
        let key = intent.idempotency_key();
        self.append(DispatchRecord {
            seq,
            request_id: intent.request_id.clone(),
            state: DispatchState::Committed,
            intent: Some(intent.clone()),
            detail: None,
        })?;
        self.entries.lock().unwrap().insert(
            intent.request_id.clone(),
            Entry {
                intent: intent.clone(),
                state: DispatchState::Committed,
                key,
                seq,
            },
        );
        Ok(())
    }

    fn load(&self, request_id: &str) -> Result<(EffectIntent, DispatchState, String), DispatchError> {
        self.entries
            .lock()
            .unwrap()
            .get(request_id)
            .map(|entry| (entry.intent.clone(), entry.state, entry.key.clone()))
            .ok_or(DispatchError::UnknownRequest)
    }

    /// Dispatch a committed intent through `sink`.
    ///
    /// The policy is evaluated *here*, at the execution point, not at approval
    /// time, so a principal revoked between the two cannot execute.
    pub fn dispatch(
        &self,
        approval: &Approval,
        sink: &dyn EffectSink,
        policy: &dyn Fn(&EffectIntent) -> Result<(), String>,
    ) -> Result<SinkOutcome, DispatchError> {
        let (intent, state, key) = self.load(&approval.request_id)?;
        if !approval.binds(&intent) {
            return Err(DispatchError::ApprovalMismatch);
        }
        match state {
            DispatchState::Committed | DispatchState::Retryable | DispatchState::Uncertain => {}
            other => return Err(DispatchError::WrongState(other)),
        }
        if let Err(reason) = policy(&intent) {
            self.record_outcome(
                &intent.request_id,
                DispatchState::Refused,
                Some(serde_json::json!({ "refused": reason })),
            )?;
            return Err(DispatchError::Refused(reason));
        }
        let outcome = sink.dispatch(&key, &intent);
        let (next, detail) = match &outcome {
            SinkOutcome::Applied(value) => (DispatchState::Acknowledged, Some(value.clone())),
            SinkOutcome::Retryable(reason) => {
                (DispatchState::Retryable, Some(serde_json::json!({ "retryable": reason })))
            }
            SinkOutcome::Uncertain(reason) => {
                let state = if sink.dedups() {
                    DispatchState::Uncertain
                } else {
                    DispatchState::Reconciling
                };
                (state, Some(serde_json::json!({ "uncertain": reason })))
            }
        };
        self.record_outcome(&intent.request_id, next, detail)?;
        Ok(outcome)
    }

    /// Retry an effect whose outcome the durable record does not call applied.
    ///
    /// Refused for an acknowledged effect (that would be a second effect) and
    /// for a non-deduplicating sink whose delivery is uncertain (that could be a
    /// second effect the operator cannot see). Both are [`DispatchState`]
    /// reasons, not silent successes.
    pub fn retry(
        &self,
        approval: &Approval,
        sink: &dyn EffectSink,
        policy: &dyn Fn(&EffectIntent) -> Result<(), String>,
    ) -> Result<SinkOutcome, DispatchError> {
        let (_, state, _) = self.load(&approval.request_id)?;
        match state {
            DispatchState::Retryable => self.dispatch(approval, sink, policy),
            DispatchState::Uncertain if sink.dedups() => self.dispatch(approval, sink, policy),
            DispatchState::Uncertain | DispatchState::Reconciling => {
                Err(DispatchError::WrongState(DispatchState::Reconciling))
            }
            other if other.is_terminal_for_retry() => Err(DispatchError::WrongState(other)),
            other => Err(DispatchError::WrongState(other)),
        }
    }

    /// Resume an intent whose `Committed` record has no outcome after it (the
    /// process died between the commit and the sink, or between the sink and
    /// the acknowledgment).
    ///
    /// A deduplicating sink is safe to ask again under the same key. A sink
    /// without dedup is not: the effect may already have happened, so the
    /// intent moves to `Reconciling` and the caller must resolve it manually.
    pub fn resume(
        &self,
        request_id: &str,
        sink: &dyn EffectSink,
        policy: &dyn Fn(&EffectIntent) -> Result<(), String>,
    ) -> Result<Option<SinkOutcome>, DispatchError> {
        let (intent, state, _) = self.load(request_id)?;
        match state {
            DispatchState::Committed => {
                if sink.dedups() {
                    self.dispatch(&intent.approval(), sink, policy).map(Some)
                } else {
                    self.record_outcome(
                        request_id,
                        DispatchState::Reconciling,
                        Some(serde_json::json!({
                            "uncertain": "committed with no recorded outcome; the sink does not deduplicate"
                        })),
                    )?;
                    Ok(None)
                }
            }
            _ => Err(DispatchError::WrongState(state)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    struct MemStore {
        streams: Mutex<BTreeMap<String, Vec<Vec<u8>>>>,
        puts: AtomicUsize,
    }

    impl MemStore {
        fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        fn copy_into(&self, other: &MemStore) {
            let streams = self.streams.lock().unwrap();
            let mut target = other.streams.lock().unwrap();
            for (name, records) in streams.iter() {
                target.insert(name.clone(), records.clone());
            }
        }
    }

    impl DurableStore for MemStore {
        fn append(&self, stream: &str, record: &[u8]) -> Result<(), unfer_protocol::durable::DurableError> {
            self.streams
                .lock()
                .unwrap()
                .entry(stream.to_string())
                .or_default()
                .push(record.to_vec());
            Ok(())
        }

        fn flush(&self) -> Result<(), unfer_protocol::durable::DurableError> {
            self.puts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn replay(&self, stream: &str) -> Result<Vec<Vec<u8>>, unfer_protocol::durable::DurableError> {
            Ok(self
                .streams
                .lock()
                .unwrap()
                .get(stream)
                .cloned()
                .unwrap_or_default())
        }

        fn frontier(&self) -> Result<Vec<u8>, unfer_protocol::durable::DurableError> {
            Ok(Vec::new())
        }

        fn fork_at(
            &self,
            _: &[u8],
        ) -> Result<Box<dyn DurableStore>, unfer_protocol::durable::DurableError> {
            Err(unfer_protocol::durable::DurableError::Unsupported(
                "memory store has no forks".into(),
            ))
        }

        fn backend(&self) -> &'static str {
            "memory-test"
        }
    }

    /// A store whose appends fail: nothing may reach a sink through it.
    #[derive(Debug)]
    struct BrokenStore;

    impl DurableStore for BrokenStore {
        fn append(&self, _: &str, _: &[u8]) -> Result<(), unfer_protocol::durable::DurableError> {
            Err(unfer_protocol::durable::DurableError::Io("disk full".into()))
        }

        fn flush(&self) -> Result<(), unfer_protocol::durable::DurableError> {
            Ok(())
        }

        fn replay(&self, _: &str) -> Result<Vec<Vec<u8>>, unfer_protocol::durable::DurableError> {
            Ok(Vec::new())
        }

        fn frontier(&self) -> Result<Vec<u8>, unfer_protocol::durable::DurableError> {
            Ok(Vec::new())
        }

        fn fork_at(
            &self,
            _: &[u8],
        ) -> Result<Box<dyn DurableStore>, unfer_protocol::durable::DurableError> {
            Err(unfer_protocol::durable::DurableError::Unsupported(
                "broken store has no forks".into(),
            ))
        }

        fn backend(&self) -> &'static str {
            "broken-test"
        }
    }

    #[derive(Debug)]
    struct CountingSink {
        dedups: bool,
        calls: Mutex<Vec<String>>,
        outcome: Mutex<SinkOutcome>,
        seen: Mutex<std::collections::HashSet<String>>,
    }

    impl CountingSink {
        fn new(dedups: bool, outcome: SinkOutcome) -> Self {
            Self {
                dedups,
                calls: Mutex::new(Vec::new()),
                outcome: Mutex::new(outcome),
                seen: Mutex::new(Default::default()),
            }
        }

        fn set_outcome(&self, outcome: SinkOutcome) {
            *self.outcome.lock().unwrap() = outcome;
        }

        fn calls(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    impl EffectSink for CountingSink {
        fn name(&self) -> &str {
            "counting"
        }

        fn dedups(&self) -> bool {
            self.dedups
        }

        fn dispatch(&self, key: &str, _: &EffectIntent) -> SinkOutcome {
            self.calls.lock().unwrap().push(key.to_string());
            let outcome = self.outcome.lock().unwrap().clone();
            if self.dedups {
                // A deduplicating sink reports the earlier result for a repeat
                // key and does not perform the effect twice.
                let mut seen = self.seen.lock().unwrap();
                if !seen.insert(key.to_string()) {
                    return SinkOutcome::Applied(serde_json::json!({ "deduplicated": true }));
                }
            }
            outcome
        }
    }

    fn allow(_: &EffectIntent) -> Result<(), String> {
        Ok(())
    }

    fn intent(request_id: &str, payload: serde_json::Value) -> EffectIntent {
        EffectIntent::new(
            request_id,
            "module-a",
            "notify-channel",
            7,
            "send_notification",
            EffectKind::Mutate,
            payload,
        )
    }

    #[test]
    fn an_intent_is_committed_before_the_sink_sees_it() {
        let store = MemStore::new();
        let dispatcher = Dispatcher::open(store.clone()).unwrap();
        let intent = intent("req-1", serde_json::json!({ "to": "ops" }));

        dispatcher.commit(&intent).unwrap();
        // The commit is replayable before any dispatch: a fresh open sees it.
        let replayed = Dispatcher::open(store).unwrap();
        assert_eq!(replayed.state("req-1"), Some(DispatchState::Committed));
        assert_eq!(
            replayed.idempotency_key("req-1"),
            Some(intent.idempotency_key())
        );

        let sink = CountingSink::new(true, SinkOutcome::Applied(serde_json::json!({ "ok": true })));
        let outcome = dispatcher
            .dispatch(&intent.approval(), &sink, &allow)
            .unwrap();
        assert_eq!(outcome, SinkOutcome::Applied(serde_json::json!({ "ok": true })));
        assert_eq!(sink.calls(), 1);
        assert_eq!(dispatcher.state("req-1"), Some(DispatchState::Acknowledged));
    }

    #[test]
    fn dispatch_without_a_durable_commit_is_refused_and_never_reaches_a_sink() {
        let store = MemStore::new();
        let dispatcher = Dispatcher::open(store).unwrap();
        let intent = intent("req-1", serde_json::json!({}));
        let sink = CountingSink::new(true, SinkOutcome::Applied(serde_json::json!({})));
        assert_eq!(
            dispatcher.dispatch(&intent.approval(), &sink, &allow),
            Err(DispatchError::UnknownRequest)
        );
        assert_eq!(sink.calls(), 0);
    }

    #[test]
    fn a_store_that_cannot_commit_never_lets_the_effect_fire() {
        let dispatcher = Dispatcher::open(Arc::new(BrokenStore)).unwrap();
        let intent = intent("req-1", serde_json::json!({}));
        assert!(matches!(
            dispatcher.commit(&intent),
            Err(DispatchError::Storage(_))
        ));
        let sink = CountingSink::new(true, SinkOutcome::Applied(serde_json::json!({})));
        assert!(dispatcher.dispatch(&intent.approval(), &sink, &allow).is_err());
        assert_eq!(sink.calls(), 0);
    }

    #[test]
    fn approval_replay_on_an_altered_payload_is_refused() {
        let dispatcher = Dispatcher::open(MemStore::new()).unwrap();
        let approved = intent("req-1", serde_json::json!({ "to": "ops" }));
        dispatcher.commit(&approved).unwrap();
        let approval = approved.approval();

        // Same request identity, altered payload: an approval for that altered
        // payload is a different authorisation and does not bind the committed
        // intent, so presenting it cannot execute the committed effect.
        let altered = intent("req-1", serde_json::json!({ "to": "attacker" }));
        assert!(!altered.approval().binds(&approved));
        let sink = CountingSink::new(true, SinkOutcome::Applied(serde_json::json!({})));
        let refused = dispatcher.dispatch(&altered.approval(), &sink, &allow);
        assert_eq!(refused, Err(DispatchError::ApprovalMismatch));
        assert_eq!(sink.calls(), 0);
        assert_eq!(dispatcher.state("req-1"), Some(DispatchState::Committed));

        // ...and re-committing the same identity with a different payload is
        // refused too, so the queue cannot be rewritten in place.
        assert!(matches!(
            dispatcher.commit(&altered),
            Err(DispatchError::Refused(_))
        ));

        // A different resource or epoch is likewise not authorised.
        let mut other_epoch = approval.clone();
        other_epoch.epoch += 1;
        assert_eq!(
            dispatcher.dispatch(&other_epoch, &sink, &allow),
            Err(DispatchError::ApprovalMismatch)
        );
    }

    #[test]
    fn a_revoked_worker_fails_at_the_execution_point() {
        let dispatcher = Dispatcher::open(MemStore::new()).unwrap();
        let intent = intent("req-1", serde_json::json!({}));
        dispatcher.commit(&intent).unwrap();
        let sink = CountingSink::new(true, SinkOutcome::Applied(serde_json::json!({})));
        let revoked = |intent: &EffectIntent| {
            if intent.principal == "module-a" {
                Err("principal revoked".to_string())
            } else {
                Ok(())
            }
        };
        assert_eq!(
            dispatcher.dispatch(&intent.approval(), &sink, &revoked),
            Err(DispatchError::Refused("principal revoked".to_string()))
        );
        assert_eq!(sink.calls(), 0, "a refused effect never reaches the sink");
        assert_eq!(dispatcher.state("req-1"), Some(DispatchState::Refused));
    }

    #[test]
    fn a_retry_reuses_the_same_idempotency_key() {
        let dispatcher = Dispatcher::open(MemStore::new()).unwrap();
        let intent = intent("req-1", serde_json::json!({}));
        dispatcher.commit(&intent).unwrap();
        let sink = CountingSink::new(
            true,
            SinkOutcome::Retryable("sink busy".to_string()),
        );
        dispatcher.dispatch(&intent.approval(), &sink, &allow).unwrap();
        assert_eq!(dispatcher.state("req-1"), Some(DispatchState::Retryable));

        sink.set_outcome(SinkOutcome::Applied(serde_json::json!({ "ok": true })));
        dispatcher.retry(&intent.approval(), &sink, &allow).unwrap();
        assert_eq!(dispatcher.state("req-1"), Some(DispatchState::Acknowledged));

        let calls = sink.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], calls[1], "every retry presents the same key");
    }

    #[test]
    fn crash_after_sink_success_before_acknowledgment_does_not_duplicate_at_a_dedup_sink() {
        // The durable log holds only the commit: the process died after the
        // sink fired but before the acknowledgment record.
        let store = MemStore::new();
        let committed = Dispatcher::open(store.clone()).unwrap();
        let intent = intent("req-1", serde_json::json!({ "to": "ops" }));
        committed.commit(&intent).unwrap();
        drop(committed);

        let recovered = Dispatcher::open(store).unwrap();
        assert_eq!(recovered.state("req-1"), Some(DispatchState::Committed));

        let sink = CountingSink::new(true, SinkOutcome::Applied(serde_json::json!({})));
        recovered.resume("req-1", &sink, &allow).unwrap().unwrap();
        // A later resume cannot happen (the state is acknowledged), and a real
        // replay of the original call under the same key is deduplicated.
        let first_key = sink.calls.lock().unwrap()[0].clone();
        assert_eq!(
            sink.dispatch(&first_key, &intent),
            SinkOutcome::Applied(serde_json::json!({ "deduplicated": true }))
        );
        assert_eq!(sink.calls(), 2, "the sink saw the key twice");
        assert_eq!(
            sink.seen.lock().unwrap().len(),
            1,
            "the effect happened once"
        );
        assert_eq!(recovered.state("req-1"), Some(DispatchState::Acknowledged));
        assert_eq!(
            recovered.retry(&intent.approval(), &sink, &allow),
            Err(DispatchError::WrongState(DispatchState::Acknowledged)),
            "an acknowledged effect is never retried"
        );
    }

    #[test]
    fn an_uncertain_irreversible_effect_enters_reconciliation_and_is_not_replayed() {
        let dispatcher = Dispatcher::open(MemStore::new()).unwrap();
        let intent = intent("req-1", serde_json::json!({}));
        dispatcher.commit(&intent).unwrap();
        let sink = CountingSink::new(false, SinkOutcome::Uncertain("timeout".to_string()));
        dispatcher.dispatch(&intent.approval(), &sink, &allow).unwrap();
        assert_eq!(
            dispatcher.state("req-1"),
            Some(DispatchState::Reconciling),
            "unknown outcome is visible, not recorded as applied"
        );
        assert!(matches!(
            dispatcher.retry(&intent.approval(), &sink, &allow),
            Err(DispatchError::WrongState(DispatchState::Reconciling))
        ));
        assert_eq!(sink.calls(), 1, "never replayed as a fresh action");
    }

    #[test]
    fn an_uncertain_deduplicating_effect_may_be_retried() {
        let dispatcher = Dispatcher::open(MemStore::new()).unwrap();
        let intent = intent("req-1", serde_json::json!({}));
        dispatcher.commit(&intent).unwrap();
        let sink = CountingSink::new(true, SinkOutcome::Uncertain("timeout".to_string()));
        dispatcher.dispatch(&intent.approval(), &sink, &allow).unwrap();
        assert_eq!(dispatcher.state("req-1"), Some(DispatchState::Uncertain));

        sink.set_outcome(SinkOutcome::Applied(serde_json::json!({})));
        dispatcher.retry(&intent.approval(), &sink, &allow).unwrap();
        assert_eq!(dispatcher.state("req-1"), Some(DispatchState::Acknowledged));
        let calls = sink.calls.lock().unwrap().clone();
        assert_eq!(calls[0], calls[1]);
    }

    #[test]
    fn resume_of_a_committed_effect_at_a_non_dedup_sink_reconciles_instead_of_firing() {
        let store = MemStore::new();
        let dispatcher = Dispatcher::open(store.clone()).unwrap();
        let intent = intent("req-1", serde_json::json!({}));
        dispatcher.commit(&intent).unwrap();
        drop(dispatcher);

        let recovered = Dispatcher::open(store).unwrap();
        let sink = CountingSink::new(false, SinkOutcome::Applied(serde_json::json!({})));
        assert_eq!(recovered.resume("req-1", &sink, &allow).unwrap(), None);
        assert_eq!(sink.calls(), 0);
        assert_eq!(
            recovered.state("req-1"),
            Some(DispatchState::Reconciling)
        );
    }

    #[test]
    fn idempotency_keys_differ_by_request_and_payload() {
        let a = intent("req-1", serde_json::json!({ "to": "ops" }));
        let b = intent("req-1", serde_json::json!({ "to": "oncall" }));
        let c = intent("req-2", serde_json::json!({ "to": "ops" }));
        assert_ne!(a.idempotency_key(), b.idempotency_key());
        assert_ne!(a.idempotency_key(), c.idempotency_key());
    }
}
