//! Phase 6 testnet: `ConsensusNode` running on a real QuePaxa cluster.
//!
//! The `network` feature wires `rust-quepaxa`'s mutually-authenticated TLS
//! transport. The value being agreed on is a `ReplicatedValue` that carries its
//! own bytes (see [`encode_ledger_value`]), and a [`LedgerStateMachine`] applies
//! each committed decision from that value. Every live node executes the same
//! committed decisions in order, so each peer's [`ConsensusNode`] (built over a
//! [`NetConsensus`] engine) reproduces the identical certificate root.
//!
//! D11 splits the two roles the module used to serve at once. A process is
//! started from a validated [`NodeConfig`] and owns exactly one replica's state
//! ([`start_node`] / [`RunningNode`]); [`NetworkCluster`] is the in-process
//! test harness that starts two of those replicas plus a local submission pump
//! over one shared state directory.
//!
//! Node state is durable: recorder snapshots, runtime snapshots, the submission
//! journal, and the replicated ledger are written to `<state_dir>` and survive a
//! graceful full-cluster restart ([`NetworkCluster::resume`]). A node restarts
//! onto the same socket addresses and TLS identities, reloads its committed log,
//! and continues proposing fresh slots.
//!
//! [`ConsensusNode`]: crate::node::ConsensusNode

use std::collections::BTreeMap;
use std::fs;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_quepaxa::network::{
    AsyncRecorderClient, DeduplicatingSubmissionHandler, DeploymentId, FileSubmissionJournal,
    MutualTlsConfigs, NetworkConsensusHandler, NetworkMetrics, NetworkNodeServer, PeerIdentity,
    PostcardRecorderCodec, PostcardRuntimeCodec, Submission, SubmissionHandler, SubmissionOutcome,
    TlsIdentity, TlsRecorderClient, TlsSubmitClient,
};
use rust_quepaxa::{
    AllowAllAvailability, Decision, DurableRecorderCore, FileRecorderStore, FileRuntimeStore,
    LaneId, RecorderConfig, ReplicaConfig, ReplicaId, ReplicaRuntimeConfig, SlotIndex, StateMachine,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use unfer_protocol::distributed::{Command, ReplicatedValue};
use unfer_protocol::submission::{ReceiptState, Slot, SubmissionId, UnknownReason};
use unfer_protocol::{Code, ConsensusTransaction, Diagnostic, Severity};

use crate::engine::ConsensusEngine;
use crate::jobs::{JobCommand, JobFence, JobQueue, JOB_COMMAND_SCHEMA};

/// D10: the schema a replicated ledger submission is read under. A command
/// under any other schema is refused by name rather than applied as a
/// neighbour.
pub const LEDGER_TX_SCHEMA: u16 = 1;

const LIVE_NODES: usize = 2;
const MAX_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

/// How often a replica offers its log to a peer that is behind. An idle cluster
/// has no new decision to carry the offer, which is why this is a timer: a
/// rejoining replica is caught up because its peers look, not because something
/// else happened.
const CATCH_UP_INTERVAL: Duration = Duration::from_millis(200);

/// Decisions per push. Bounded so one catch-up frame stays well inside the
/// transport's frame limit, whatever the size of the log behind it.
const CATCH_UP_BATCH: u64 = 256;

/// Durable ledger snapshot: the committed bank plus this node's outbox.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LedgerSnapshot {
    bank: Vec<ConsensusTransaction>,
    /// D10: the local outbox — submissions this node has accepted and not yet
    /// seen committed. It is **not** how a committed value's payload is found:
    /// the value carries its own bytes, so a peer that never held this entry
    /// still applies the work. Renamed from `pending`; `alias` keeps reading a
    /// pre-D10 snapshot, whose entries are re-proposed on load.
    #[serde(default, alias = "pending")]
    outbox: BTreeMap<u64, ConsensusTransaction>,
    /// D05: submission id -> committed slot. This is what lets a receipt survive
    /// a restart: the id a caller received is not the position it landed at, so
    /// without this map a restarted node could not answer "what happened to my
    /// submission?". `serde(default)` reads a snapshot written before the map
    /// existed (its committed entries simply have no receipt).
    #[serde(default)]
    committed_ids: BTreeMap<u64, u64>,
    next_id: u64,
    /// Monotonic pump request counter; survives restarts so request ids never
    /// collide with entries already in a durable submission journal.
    next_request_id: u64,
    /// D14: the current job fence per resource. It lives in the committed
    /// snapshot, so the committed state — not a worker — owns which fence is
    /// current, and a restarted worker cannot re-admit the one it lost.
    #[serde(default)]
    jobs: JobQueue,
}

/// Atomic (`write` + `rename`) JSON persistence for the replicated ledger.
#[derive(Debug, Clone)]
struct LedgerFile {
    path: PathBuf,
}

impl LedgerFile {
    fn new(state_dir: &Path) -> Self {
        Self {
            path: state_dir.join("ledger.json"),
        }
    }

    /// Read the durable snapshot. A missing file means a fresh ledger (the
    /// `Ok(default)` case); an unreadable or corrupt file is surfaced as an
    /// error so the caller can fail loudly instead of silently losing the
    /// committed log.
    fn load(&self) -> Result<LedgerSnapshot, String> {
        match fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|error| {
                format!(
                    "ledger snapshot {} is corrupt: {error}",
                    self.path.display()
                )
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(LedgerSnapshot::default())
            }
            Err(error) => Err(format!(
                "could not read ledger snapshot {}: {error}",
                self.path.display()
            )),
        }
    }

    /// D07 crash protocol: write the temporary file, **fsync it**, rename, then
    /// fsync the parent directory.
    ///
    /// Without the first fsync a crash can leave the renamed file empty or
    /// truncated (the rename is durable, its contents are not); without the
    /// second the rename itself can be lost. Errors are propagated rather than
    /// swallowed: a snapshot that is not durable is a commit the caller must not
    /// report as durable.
    fn save(&self, snapshot: &LedgerSnapshot) -> Result<(), String> {
        use std::io::Write;
        let bytes = serde_json::to_vec(snapshot)
            .map_err(|error| format!("ledger snapshot encode failed: {error}"))?;
        let parent = self.path.parent().unwrap_or(Path::new("."));
        let temporary = self.path.with_extension("ledger.tmp");
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create ledger dir {}: {error}", parent.display()))?;
        let mut file = fs::File::create(&temporary)
            .map_err(|error| format!("could not create {}: {error}", temporary.display()))?;
        file.write_all(&bytes)
            .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
        file.sync_all()
            .map_err(|error| format!("could not fsync {}: {error}", temporary.display()))?;
        drop(file);
        fs::rename(&temporary, &self.path).map_err(|error| {
            format!(
                "ledger snapshot publish failed for {}: {error}",
                self.path.display()
            )
        })?;
        // The rename is a directory entry: fsync the directory so the new name
        // survives a power loss.
        if let Ok(directory) = fs::File::open(parent) {
            directory.sync_all().map_err(|error| {
                format!("could not fsync ledger dir {}: {error}", parent.display())
            })?;
        }
        Ok(())
    }
}

/// D07: everything the replicated ledger decides, behind **one** lock.
///
/// The committed bank, the outbox and the submission-id map used to be read
/// under separate locks when persisting, so a crash could publish a snapshot
/// assembled from two different instants — for example a payload already
/// removed from the outbox and not yet in `bank`, which was a lost transaction
/// that looked like a clean snapshot. One lock makes the snapshot a state that
/// actually existed.
#[derive(Debug, Clone, Default)]
struct LedgerState {
    bank: Vec<ConsensusTransaction>,
    outbox: BTreeMap<u64, ConsensusTransaction>,
    committed_ids: BTreeMap<u64, u64>,
    jobs: JobQueue,
}

impl LedgerState {
    fn snapshot(&self, next_id: u64, next_request_id: u64) -> LedgerSnapshot {
        LedgerSnapshot {
            bank: self.bank.clone(),
            outbox: self.outbox.clone(),
            committed_ids: self.committed_ids.clone(),
            next_id,
            next_request_id,
            jobs: self.jobs.clone(),
        }
    }
}

/// The replicated outcome: the ordered committed log, this node's outbox and
/// progress bookkeeping for the pump task.
#[derive(Clone)]
pub struct SharedLedger {
    state: Arc<Mutex<LedgerState>>,
    next_id: Arc<AtomicU64>,
    request_seq: Arc<AtomicU64>,
    committed_tx: watch::Sender<u64>,
    /// The local submission pump, when this ledger has one. A standalone node
    /// process has none: submissions reach it over TLS as complete values, so
    /// the ledger is only the application target (D11).
    submitted: Option<mpsc::UnboundedSender<(u64, ReplicatedValue)>>,
    file: Option<LedgerFile>,
}

/// D14: encode an ordered job command as a replicated value. It carries its
/// own payload under [`JOB_COMMAND_SCHEMA`], so a replica that never saw the
/// submission applies the same command from the decision alone.
pub fn encode_job_value(command: &JobCommand) -> Result<ReplicatedValue, String> {
    let json = serde_json::to_vec(command)
        .map_err(|error| format!("could not encode job command: {error}"))?;
    Command::new(JOB_COMMAND_SCHEMA, json)
        .map(ReplicatedValue::Command)
        .map_err(|error| format!("could not frame job command: {error}"))
}

/// Decode a job command from a decision value. `None` when the value is under a
/// different schema; undecodable bytes under the job schema are an error, so a
/// malformed command halts every replica instead of being silently skipped.
fn decode_job_command(value: &ReplicatedValue) -> Result<Option<JobCommand>, String> {
    let ReplicatedValue::Command(command) = value else {
        return Ok(None);
    };
    if command.schema() != JOB_COMMAND_SCHEMA {
        return Ok(None);
    }
    serde_json::from_slice(command.payload())
        .map(Some)
        .map_err(|error| format!("job command does not decode: {error}"))
}

/// D10: the value a submission becomes. The reservation id lives **inside**
/// the payload, ahead of the canonical transaction bytes, so a decision is
/// self-describing: the state machine needs no map to know which submission it
/// applied, and a replica that never queued the transaction can still apply it.
///
/// Payload layout: `id(u64 BE) | serde_json(ConsensusTransaction)`. The JSON is
/// the same canonical rendering `signing::canonical_bytes` hashes.
pub fn encode_ledger_value(id: u64, tx: &ConsensusTransaction) -> Result<ReplicatedValue, String> {
    let json = serde_json::to_vec(tx)
        .map_err(|error| format!("could not encode transaction {id}: {error}"))?;
    let mut payload = Vec::with_capacity(8 + json.len());
    payload.extend_from_slice(&id.to_be_bytes());
    payload.extend_from_slice(&json);
    Command::new(LEDGER_TX_SCHEMA, payload)
        .map(ReplicatedValue::Command)
        .map_err(|error| format!("could not frame transaction {id}: {error}"))
}

/// The inverse of [`encode_ledger_value`]. A no-op decision yields `None`; an
/// unknown schema, a short payload or undecodable bytes are errors — a
/// committed value this node cannot read is a refusal it must report, never a
/// slot it silently skips.
pub fn decode_ledger_value(
    value: &ReplicatedValue,
) -> Result<Option<(u64, ConsensusTransaction)>, String> {
    let ReplicatedValue::Command(command) = value else {
        return Ok(None);
    };
    if command.schema() != LEDGER_TX_SCHEMA {
        return Err(format!(
            "committed value has schema {}, not {LEDGER_TX_SCHEMA}",
            command.schema()
        ));
    }
    let payload = command.payload();
    if payload.len() < 8 {
        return Err(format!(
            "committed value payload is {} byte(s), shorter than its id",
            payload.len()
        ));
    }
    let (id_bytes, json) = payload.split_at(8);
    let mut raw = [0u8; 8];
    raw.copy_from_slice(id_bytes);
    let id = u64::from_be_bytes(raw);
    let tx = serde_json::from_slice(json)
        .map_err(|error| format!("committed value {id} does not decode: {error}"))?;
    Ok(Some((id, tx)))
}

impl std::fmt::Debug for SharedLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("SharedLedger")
            .field("committed", &state.bank.len())
            .field("outbox", &state.outbox.len())
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .field("with_file", &self.file.is_some())
            .finish()
    }
}

impl SharedLedger {
    /// D11: open a ledger that only applies committed decisions. A standalone
    /// node process submits nothing itself — its clients send whole values over
    /// TLS — so `allocate` refuses instead of queueing into a pump that is not
    /// there.
    pub fn open(state_dir: &Path) -> Result<Self, String> {
        Self::new(None, Some(LedgerFile::new(state_dir)))
    }

    fn new(
        submitted: Option<mpsc::UnboundedSender<(u64, ReplicatedValue)>>,
        file: Option<LedgerFile>,
    ) -> Result<Self, String> {
        // A missing ledger file is a fresh cluster (`Ok(default)`); a corrupt
        // file fails loudly at the call site rather than silently resetting the
        // committed log to empty.
        let loaded = match file.as_ref().map(LedgerFile::load) {
            Some(Ok(snapshot)) => snapshot,
            Some(Err(error)) => return Err(error),
            None => LedgerSnapshot::default(),
        };
        let (committed_tx, _) = watch::channel(loaded.bank.len() as u64);
        let ledger = Self {
            state: Arc::new(Mutex::new(LedgerState {
                bank: loaded.bank,
                outbox: loaded.outbox,
                committed_ids: loaded.committed_ids,
                jobs: loaded.jobs,
            })),
            next_id: Arc::new(AtomicU64::new(loaded.next_id)),
            request_seq: Arc::new(AtomicU64::new(loaded.next_request_id)),
            committed_tx,
            submitted,
            file,
        };
        // Re-enqueue anything still in the outbox so the pump re-proposes it
        // after a restart, then persist the reconciled view. A pre-D10 snapshot
        // reaches this path too (its `pending` map is read as the outbox), so
        // queued work is re-proposed rather than dropped.
        let outstanding = ledger
            .state
            .lock()
            .unwrap()
            .outbox
            .iter()
            .map(|(id, tx)| (*id, tx.clone()))
            .collect::<Vec<_>>();
        for (id, tx) in outstanding {
            if let (Some(pump), Ok(value)) = (&ledger.submitted, encode_ledger_value(id, &tx)) {
                let _ = pump.send((id, value));
            }
        }
        ledger.persist()?;
        Ok(ledger)
    }

    /// Reserve a unique id for a submitted transaction, persist it in this
    /// node's outbox, and hand the framed value to the pump task to propose to
    /// the cluster. The outbox entry is what makes the submission durable
    /// before it is acknowledged; the **value** is what makes it replicable.
    fn allocate(&self, tx: ConsensusTransaction) -> Result<u64, String> {
        let Some(pump) = &self.submitted else {
            return Err(
                "this ledger has no submission pump; submit through the node's TLS "
                    .to_owned()
                    + "endpoint instead of proposing locally",
            );
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let value = encode_ledger_value(id, &tx)?;
        {
            let mut state = self.state.lock().unwrap();
            state.outbox.insert(id, tx);
            if let Err(error) = self.persist_state(&state) {
                // A reservation the ledger could not make durable is not kept:
                // leaving it in memory would let it commit later, under an id the
                // caller was told had failed.
                state.outbox.remove(&id);
                return Err(error);
            }
        }
        let _ = pump.send((id, value));
        Ok(id)
    }

    /// Apply a batch of committed values in decision order.
    ///
    /// D10: the transaction is read **from the decision value**, not from a
    /// shared payload map, so a replica that never queued the submission still
    /// applies it and a restarted node does not depend on a map that only the
    /// submitting node had. Extending the bank and persisting happen under one
    /// lock hold, so the durable snapshot is a state that actually existed, and
    /// re-applying the same decision (a retry, or the second replica on the same
    /// ledger) is a no-op because the id is already committed.
    fn append(&self, values: Vec<ReplicatedValue>) -> Result<(), String> {
        // Decode before touching state: a malformed value must not leave the
        // outbox half-edited. D14: a job command is dispatched by schema, so
        // both kinds of decision flow through one ordered apply.
        let mut decoded = Vec::with_capacity(values.len());
        let mut job_commands = Vec::new();
        for value in &values {
            if let Some(command) = decode_job_command(value)? {
                job_commands.push(command);
                continue;
            }
            if let Some((id, tx)) = decode_ledger_value(value)? {
                decoded.push((id, tx));
            }
        }
        let committed;
        {
            let mut state = self.state.lock().unwrap();
            let bank_len = state.bank.len();
            let mut appended = Vec::new();
            let mut removed = Vec::new();
            for (id, tx) in decoded {
                // Already committed: the same decision delivered twice adds
                // nothing. Its outbox entry is gone either way.
                if let Some(previous) = state.outbox.remove(&id) {
                    removed.push((id, previous));
                }
                if !state.committed_ids.contains_key(&id) {
                    appended.push((id, tx));
                }
            }
            // A refused job command is a deterministic no-op: every replica
            // computes the same verdict, so it changes no state and is not an
            // error here. It must still be applied in decision order.
            for command in &job_commands {
                let _ = state.jobs.apply(command);
            }
            if appended.is_empty() && values.is_empty() {
                return Ok(());
            }
            let mut inserted = Vec::new();
            for (id, tx) in &appended {
                state.bank.push(tx.clone());
                // D05: record id -> slot as each payload enters the bank, so the
                // receipt answers with the position the payload landed at rather
                // than the id it was reserved under.
                let slot = state.bank.len() as u64;
                state.committed_ids.insert(*id, slot);
                inserted.push(*id);
            }
            committed = state.bank.len() as u64;
            // The decision is acknowledged only once one coherent snapshot of
            // the new state is durable. If it cannot be written, the in-memory
            // state is rolled back to the durable prefix: a slot this ledger
            // could not persist must not be served by `committed`, `receipt` or
            // `state_digest`, which is the same rule the journaled node enforces
            // by halting instead of advancing its watermark.
            if let Err(error) = self.persist_state(&state) {
                state.bank.truncate(bank_len);
                for id in inserted {
                    state.committed_ids.remove(&id);
                }
                for (id, tx) in removed {
                    state.outbox.insert(id, tx);
                }
                return Err(error);
            }
        }
        let _ = self.committed_tx.send(committed);
        Ok(())
    }

    /// D05: what this ledger knows about a submission id. Committed slots come
    /// from the durable map, still-queued ids from the pending store, and
    /// anything else is not known here -- never "refused".
    fn receipt(&self, id: u64) -> ReceiptState {
        let state = self.state.lock().unwrap();
        if let Some(slot) = state.committed_ids.get(&id) {
            return ReceiptState::Committed(Slot(*slot));
        }
        if state.outbox.contains_key(&id) {
            return ReceiptState::Pending;
        }
        ReceiptState::Unknown(UnknownReason::Unsubmitted)
    }

    fn persist(&self) -> Result<(), String> {
        let state = self.state.lock().unwrap();
        self.persist_state(&state)
    }

    /// Persist one coherent snapshot. The caller holds the state lock, so the
    /// bytes written are a state the ledger actually had, not a blend of two.
    fn persist_state(&self, state: &LedgerState) -> Result<(), String> {
        let Some(file) = &self.file else {
            return Ok(());
        };
        let snapshot = state.snapshot(
            self.next_id.load(Ordering::Relaxed),
            self.request_seq.load(Ordering::Relaxed),
        );
        file.save(&snapshot)
    }

    /// Claim the next submission request id (monotonic across restarts).
    fn next_request_id(&self) -> u64 {
        self.request_seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Number of transactions committed so far.
    pub fn committed(&self) -> u64 {
        self.state.lock().unwrap().bank.len() as u64
    }

    /// Committed prefix in `(1-based_seq, tx)` form, from `from_seq` onward.
    pub fn committed_since(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)> {
        let state = self.state.lock().unwrap();
        state
            .bank
            .iter()
            .enumerate()
            .filter(|(i, _)| *i as u64 >= from_seq.saturating_sub(1))
            .map(|(i, tx)| (i as u64 + 1, tx.clone()))
            .collect()
    }

    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.committed_tx.subscribe()
    }

    /// D14: mutate the committed job fences under the ledger lock and persist
    /// one coherent snapshot. Every transition (claim, renew, release,
    /// reclaim, settle, disable) goes through here, so the durable snapshot is
    /// always a state the cluster actually held — the same rule the bank
    /// follows.
    pub fn with_jobs<R>(&self, f: impl FnOnce(&mut JobQueue) -> R) -> Result<R, String> {
        let mut state = self.state.lock().unwrap();
        let result = f(&mut state.jobs);
        self.persist_state(&state)?;
        Ok(result)
    }

    /// D14: read the current fence without taking a mutable handle.
    pub fn current_fence(&self, resource: &str) -> Option<JobFence> {
        self.state
            .lock()
            .unwrap()
            .jobs
            .current_fence(resource)
            .cloned()
    }
}

/// A [`ConsensusEngine`] whose committed log arrives asynchronously via QuePaxa
/// consensus instead of a local append. `submit` reserves an id immediately and
/// queues it for proposal; the transaction is readable through `get_log` only
/// after the cluster commits its slot. Callers reconcile with `sync()`.
#[derive(Clone)]
pub struct NetConsensus {
    ledger: SharedLedger,
}

impl NetConsensus {
    pub fn new(ledger: SharedLedger) -> Self {
        Self { ledger }
    }
}

impl ConsensusEngine for NetConsensus {
    fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic> {
        self.ledger.allocate(tx).map_err(|error| {
            Diagnostic::new(
                Code::INTERNAL,
                format!("could not durably reserve the submission: {error}"),
                Severity::Error,
            )
        })
    }

    fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)> {
        self.ledger.committed_since(from_seq)
    }

    fn current_seq(&self) -> u64 {
        self.ledger.committed()
    }

    fn receipt(&self, id: SubmissionId) -> ReceiptState {
        self.ledger.receipt(id.0)
    }
}

/// Applies committed QuePaxa decisions to the shared ledger. Every live node
/// runs one of these against the same decision stream.
pub struct LedgerStateMachine {
    ledger: SharedLedger,
}

impl StateMachine<ReplicatedValue> for LedgerStateMachine {
    fn execute(&mut self, decision: &Decision<ReplicatedValue>) -> rust_quepaxa::Result<()> {
        // A gap-filler value is the explicit `Noop` variant, decided by the
        // runtime through `with_noop_value`; it commits a slot and changes no
        // state.
        self.ledger
            .append(decision.value_ids.clone())
            .map_err(|error| {
                rust_quepaxa::QuePaxaError::StorageError(format!(
                    "could not persist committed decision: {error}"
                ))
            })
    }
}

// ── D11: one replica per process ─────────────────────────────────────────────
//
// Everything below is what a *process* is started with. `NetworkCluster` still
// exists as a fast in-process harness, but it is now built out of these pieces
// instead of owning a private copy of the startup logic.

/// One member of a deployment: its replica id, the address it listens on and
/// the TLS identity peers authenticate it by.
#[derive(Debug, Clone)]
pub struct MemberConfig {
    pub replica: ReplicaId,
    pub address: SocketAddr,
    pub identity: TlsIdentity,
}

/// A client identity a deployment accepts submissions from. Clients never
/// listen; they present this certificate when they connect.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub client_id: [u8; 16],
    pub identity: TlsIdentity,
}

/// The deployment every process of a consensus run shares: the deployment id,
/// the lane, every replica's address and identity, and the accepted clients.
///
/// D11: this is the whole trust boundary a process is started with, and it is
/// validated before anything is bound or written. A malformed deployment — id
/// zero, a duplicate replica id or address, a certificate shared by two roles,
/// a non-consecutive member list — is refused with a named error instead of
/// half a cluster coming up.
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    pub deployment: DeploymentId,
    pub lane: LaneId,
    pub members: Vec<MemberConfig>,
    pub clients: Vec<ClientConfig>,
}

/// The JSON a `DeploymentConfig` is read from. Certificates are DER, hex
/// encoded, so a fixture is one self-contained file and two processes never
/// have to agree on a path (which is what "no shared payload files" means).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeploymentFile {
    /// The 128-bit deployment id, as a decimal string so the full width is
    /// explicit rather than left to a float.
    deployment: String,
    lane: u64,
    members: Vec<MemberFile>,
    #[serde(default)]
    clients: Vec<ClientFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemberFile {
    replica: u64,
    address: String,
    certificate_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClientFile {
    client_id_hex: String,
    certificate_hex: String,
}

/// One process's own config: which replica it is, where its private state
/// lives, and the deployment it belongs to.
#[derive(Debug, Clone)]
pub struct NodeConfig {
    pub deployment: DeploymentConfig,
    pub replica: ReplicaId,
    /// This process's private state: recorder/runtime snapshots, the submission
    /// journal and the replicated ledger. It is locked for the process's
    /// lifetime, so a second process pointed at it fails instead of silently
    /// sharing an application store.
    pub state_dir: PathBuf,
    /// This replica's TLS identity (certificate chain + PKCS#8 key).
    pub identity: TlsIdentity,
    pub rpc_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeFile {
    replica: u64,
    state_dir: String,
    certificate_hex: String,
    key_hex: String,
    #[serde(default = "default_rpc_timeout_ms")]
    rpc_timeout_ms: u64,
    deployment: DeploymentFile,
}

/// A client process's config: the deployment, which client it is, and its TLS
/// identity. Used by the `submit` command, which is a client, not a node.
#[derive(Debug, Clone)]
pub struct SubmitConfig {
    pub deployment: DeploymentConfig,
    pub client_id: [u8; 16],
    pub identity: TlsIdentity,
    pub rpc_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SubmitFile {
    client_id_hex: String,
    certificate_hex: String,
    key_hex: String,
    #[serde(default = "default_rpc_timeout_ms")]
    rpc_timeout_ms: u64,
    deployment: DeploymentFile,
}

fn default_rpc_timeout_ms() -> u64 {
    DEFAULT_RPC_TIMEOUT_MS
}

const DEFAULT_RPC_TIMEOUT_MS: u64 = 30_000;

fn hex_bytes(field: &str, value: &str) -> Result<Vec<u8>, String> {
    let bytes = hex::decode(value.trim())
        .map_err(|error| format!("{field} is not hex: {error}"))?;
    if bytes.is_empty() {
        return Err(format!("{field} is empty"));
    }
    Ok(bytes)
}

fn parse_address(field: &str, value: &str) -> Result<SocketAddr, String> {
    value
        .parse()
        .map_err(|error| format!("{field} `{value}` is not a socket address: {error}"))
}

impl DeploymentConfig {
    fn from_file(file: &DeploymentFile) -> Result<Self, String> {
        let deployment: u128 = file
            .deployment
            .trim()
            .parse()
            .map_err(|error| format!("deployment id is not a u128: {error}"))?;
        let members = file
            .members
            .iter()
            .map(|member| {
                Ok(MemberConfig {
                    replica: ReplicaId::new(member.replica),
                    address: parse_address("member address", &member.address)?,
                    identity: TlsIdentity {
                        certificate_chain_der: vec![hex_bytes(
                            "member certificate_hex",
                            &member.certificate_hex,
                        )?],
                        private_key_pkcs8_der: Vec::new(),
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let clients = file
            .clients
            .iter()
            .map(|client| {
                let raw = hex_bytes("client_id_hex", &client.client_id_hex)?;
                let client_id: [u8; 16] = raw.as_slice().try_into().map_err(|_| {
                    format!(
                        "client_id_hex is {} byte(s), not the 16 a client id is",
                        raw.len()
                    )
                })?;
                Ok(ClientConfig {
                    client_id,
                    identity: TlsIdentity {
                        certificate_chain_der: vec![hex_bytes(
                            "client certificate_hex",
                            &client.certificate_hex,
                        )?],
                        private_key_pkcs8_der: Vec::new(),
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            deployment: DeploymentId::from_u128(deployment),
            lane: LaneId::new(file.lane),
            members,
            clients,
        })
    }

    /// D11: refuse a deployment that cannot be one deployment. Every rule here
    /// exists because violating it produces a cluster that binds, answers, and
    /// then disagrees with itself.
    pub fn validate(&self) -> Result<(), String> {
        if self.deployment.0 == [0u8; 16] {
            return Err("deployment id must not be zero".to_owned());
        }
        if self.members.len() < 2 {
            return Err(format!(
                "a deployment needs at least two members, not {}",
                self.members.len()
            ));
        }
        let mut addresses = BTreeMap::new();
        let mut certificates = BTreeMap::new();
        for (index, member) in self.members.iter().enumerate() {
            // The recorder and runtime address slots by member order, so a
            // member list that is not `1..=n` duplicates or skips a slot.
            let expected = index as u64 + 1;
            if member.replica.get() != expected {
                return Err(format!(
                    "members must be replicas 1..=n in order; position {index} holds replica {}",
                    member.replica.get()
                ));
            }
            if let Some(other) = addresses.insert(member.address, member.replica) {
                return Err(format!(
                    "replicas {} and {} both listen on {}",
                    other.get(),
                    member.replica.get(),
                    member.address
                ));
            }
            let certificate = member
                .identity
                .certificate_chain_der
                .first()
                .ok_or_else(|| format!("replica {} has no certificate", member.replica.get()))?;
            if let Some(other) = certificates.insert(certificate.clone(), format!("replica {}", member.replica.get())) {
                return Err(format!(
                    "replica {} presents the same certificate as {other}",
                    member.replica.get()
                ));
            }
        }
        let mut client_ids = BTreeMap::new();
        for client in &self.clients {
            if client.client_id == [0u8; 16] {
                return Err("a client id must not be zero".to_owned());
            }
            if client.identity.certificate_chain_der.is_empty() {
                return Err("a client has no certificate".to_owned());
            }
            if let Some(other) = client_ids.insert(client.client_id, ()) {
                let _ = other;
                return Err(format!(
                    "client {} is declared twice",
                    hex::encode(client.client_id)
                ));
            }
            let certificate = client.identity.certificate_chain_der[0].clone();
            if let Some(other) = certificates.get(&certificate) {
                return Err(format!(
                    "client {} presents the certificate of {other}",
                    hex::encode(client.client_id)
                ));
            }
            certificates.insert(certificate, format!("client {}", hex::encode(client.client_id)));
        }
        Ok(())
    }

    fn member(&self, replica: ReplicaId) -> Result<&MemberConfig, String> {
        self.members
            .iter()
            .find(|member| member.replica == replica)
            .ok_or_else(|| format!("replica {} is not a member of this deployment", replica.get()))
    }

    fn index_of(&self, replica: ReplicaId) -> Result<usize, String> {
        self.members
            .iter()
            .position(|member| member.replica == replica)
            .ok_or_else(|| format!("replica {} is not a member of this deployment", replica.get()))
    }

    /// Every certificate this deployment trusts, in one deterministic order.
    fn roots(&self) -> Vec<Vec<u8>> {
        let mut roots = self
            .members
            .iter()
            .filter_map(|member| member.identity.certificate_chain_der.first().cloned())
            .collect::<Vec<_>>();
        roots.extend(
            self.clients
                .iter()
                .filter_map(|client| client.identity.certificate_chain_der.first().cloned()),
        );
        roots
    }

    /// The certificate → sender map the node server authenticates peers with.
    fn peer_registry(&self) -> BTreeMap<Vec<u8>, PeerIdentity> {
        self.members
            .iter()
            .filter_map(|member| {
                member
                    .identity
                    .certificate_chain_der
                    .first()
                    .map(|certificate| (certificate.clone(), PeerIdentity::Replica(member.replica)))
            })
            .chain(self.clients.iter().filter_map(|client| {
                client
                    .identity
                    .certificate_chain_der
                    .first()
                    .map(|certificate| (certificate.clone(), PeerIdentity::Client(client.client_id)))
            }))
            .collect()
    }

    /// The mutual-TLS configuration for **one** identity in this deployment:
    /// the server config this node presents and the client config it uses to
    /// reach the other members. Only the node's own key material is needed —
    /// peers are authenticated by the certificates `roots()` trusts, and a
    /// deployment file never carries another member's private key.
    fn node_tls(&self, identity: &TlsIdentity) -> Result<MutualTlsConfigs, String> {
        MutualTlsConfigs::new(identity, self.roots())
            .map_err(|error| format!("node TLS setup failed: {error}"))
    }
}

impl NodeConfig {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let file: NodeFile =
            serde_json::from_str(text).map_err(|error| format!("node config is not valid JSON: {error}"))?;
        let deployment = DeploymentConfig::from_file(&file.deployment)?;
        let config = Self {
            replica: ReplicaId::new(file.replica),
            state_dir: PathBuf::from(file.state_dir.trim()),
            identity: TlsIdentity {
                certificate_chain_der: vec![hex_bytes("certificate_hex", &file.certificate_hex)?],
                private_key_pkcs8_der: hex_bytes("key_hex", &file.key_hex)?,
            },
            rpc_timeout: Duration::from_millis(file.rpc_timeout_ms),
            deployment,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        Self::from_json(&text)
            .map_err(|error| format!("{} is not a usable node config: {error}", path.display()))
    }

    pub fn validate(&self) -> Result<(), String> {
        self.deployment.validate()?;
        let member = self.deployment.member(self.replica)?;
        if self.identity.certificate_chain_der.is_empty() {
            return Err("this node has no certificate".to_owned());
        }
        if self.identity.private_key_pkcs8_der.is_empty() {
            return Err("this node has no private key".to_owned());
        }
        if member.identity.certificate_chain_der[0] != self.identity.certificate_chain_der[0] {
            return Err(format!(
                "this node's certificate is not replica {}'s certificate in the deployment",
                self.replica.get()
            ));
        }
        if self.state_dir.as_os_str().is_empty() {
            return Err("state_dir is empty".to_owned());
        }
        if !self.state_dir.is_absolute() {
            return Err(format!(
                "state_dir {} must be absolute, so two processes cannot resolve it differently",
                self.state_dir.display()
            ));
        }
        if self.rpc_timeout.is_zero() {
            return Err("rpc_timeout_ms must be non-zero".to_owned());
        }
        Ok(())
    }
}

impl SubmitConfig {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let file: SubmitFile = serde_json::from_str(text)
            .map_err(|error| format!("client config is not valid JSON: {error}"))?;
        let raw = hex_bytes("client_id_hex", &file.client_id_hex)?;
        let client_id: [u8; 16] = raw.as_slice().try_into().map_err(|_| {
            format!(
                "client_id_hex is {} byte(s), not the 16 a client id is",
                raw.len()
            )
        })?;
        let config = Self {
            deployment: DeploymentConfig::from_file(&file.deployment)?,
            client_id,
            identity: TlsIdentity {
                certificate_chain_der: vec![hex_bytes("certificate_hex", &file.certificate_hex)?],
                private_key_pkcs8_der: hex_bytes("key_hex", &file.key_hex)?,
            },
            rpc_timeout: Duration::from_millis(file.rpc_timeout_ms),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        Self::from_json(&text)
            .map_err(|error| format!("{} is not a usable client config: {error}", path.display()))
    }

    /// A client is only usable against a deployment that declares it: an
    /// undeclared client is refused here rather than after a TLS handshake.
    pub fn validate(&self) -> Result<(), String> {
        self.deployment.validate()?;
        if self.identity.private_key_pkcs8_der.is_empty() {
            return Err("this client has no private key".to_owned());
        }
        let declared = self
            .deployment
            .clients
            .iter()
            .find(|client| client.client_id == self.client_id)
            .ok_or_else(|| {
                format!(
                    "client {} is not declared by this deployment",
                    hex::encode(self.client_id)
                )
            })?;
        if declared.identity.certificate_chain_der[0] != self.identity.certificate_chain_der[0] {
            return Err(format!(
                "client {} presents a certificate the deployment does not declare for it",
                hex::encode(self.client_id)
            ));
        }
        Ok(())
    }

    /// The deployment's mutual-TLS client config for this client's own identity.
    pub fn tls(&self) -> Result<MutualTlsConfigs, String> {
        MutualTlsConfigs::new(&self.identity, self.deployment.roots())
            .map_err(|error| format!("client TLS setup failed: {error}"))
    }
}

/// An exclusive claim on a node's state directory, held for the process's
/// lifetime.
///
/// D11 asks that no two nodes share an application store. A state directory is
/// the store here, so the claim is an advisory lock on `<state_dir>/node.lock`
/// (released by the kernel when the process dies, including on a crash — which
/// is why it is a lock and not a lock *file*). A second process pointed at the
/// same directory fails at startup instead of racing the first one's writes.
pub struct StateDirLock {
    directory: PathBuf,
    file: fs::File,
}

impl StateDirLock {
    pub fn acquire(directory: &Path) -> Result<Self, String> {
        fs::create_dir_all(directory).map_err(|error| {
            format!(
                "could not create state dir {}: {error}",
                directory.display()
            )
        })?;
        let path = directory.join("node.lock");
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| format!("could not open {}: {error}", path.display()))?;
        file.try_lock().map_err(|_| {
            format!(
                "state dir {} is locked by another node process; a node owns its \
                 application store exclusively",
                directory.display()
            )
        })?;
        Ok(Self {
            directory: directory.to_path_buf(),
            file,
        })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }
}

impl Drop for StateDirLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

impl std::fmt::Debug for StateDirLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateDirLock")
            .field("directory", &self.directory)
            .finish()
    }
}

/// A running replica: its TLS server task, the ledger it applies decisions to,
/// and (for a standalone process) the exclusive claim on its state directory.
pub struct RunningNode {
    pub replica_id: ReplicaId,
    /// The address the server actually bound. A config may ask for port 0 and
    /// learn the port here.
    pub address: SocketAddr,
    pub ledger: SharedLedger,
    /// `Some` for a standalone process ([`start_node`]), `None` for the
    /// in-process harness, whose replicas share one directory by construction.
    pub state_dir: Option<StateDirLock>,
    shutdown: CancellationToken,
    handle: tokio::task::JoinHandle<()>,
}

impl RunningNode {
    pub async fn shutdown(self) {
        self.shutdown.cancel();
        let _ = self.handle.await;
    }

    /// The canonical digest of the log this replica has applied.
    pub fn digest(&self) -> String {
        state_digest(&self.ledger)
    }

    pub async fn wait_committed(&self, at_least: u64) {
        wait_committed(&self.ledger, at_least).await
    }
}

/// D11: a digest two independent processes must agree on. It covers the
/// `(1-based slot, canonical transaction hash)` pairs in order, so two replicas
/// that applied the same decisions in the same order agree, and any divergence
/// — a missing commit, a reordered one, a different payload — does not.
pub fn state_digest(ledger: &SharedLedger) -> String {
    digest_of(ledger.committed_since(1).iter().map(|(slot, tx)| (*slot, tx)))
}

/// The same digest, read straight from a state directory's durable ledger.
///
/// This takes no lock and writes nothing, so it answers "what has this node
/// applied?" about a running node — a process that opened the ledger the way
/// [`SharedLedger::open`] does would re-propose from the outbox and rewrite the
/// snapshot, which is exactly the shared write D11 rules out.
pub fn state_dir_digest(directory: &Path) -> Result<String, String> {
    let snapshot = LedgerFile::new(directory).load()?;
    Ok(digest_of(
        snapshot
            .bank
            .iter()
            .enumerate()
            .map(|(index, tx)| (index as u64 + 1, tx)),
    ))
}

fn digest_of<'a>(entries: impl Iterator<Item = (u64, &'a ConsensusTransaction)>) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut count = 0u64;
    for (slot, tx) in entries {
        hasher.update(slot.to_be_bytes());
        hasher.update(crate::signing::canonical_bytes(tx));
        count = slot;
    }
    format!("{count}:{}", hex::encode(hasher.finalize()))
}

async fn wait_committed(ledger: &SharedLedger, at_least: u64) {
    let mut rx = ledger.subscribe();
    loop {
        if *rx.borrow() >= at_least {
            return;
        }
        if rx.changed().await.is_err() {
            return;
        }
    }
}

/// The submission handler the server owns, shared with the catch-up task.
///
/// The task needs the runtime's decision log to push it to a peer, and the
/// server needs the handler to accept submissions. One `Arc` serves both, so the
/// task reads the decisions the replica actually decided instead of a copy that
/// could disagree with it.
struct SharedSubmissionHandler<H>(Arc<H>);

impl<H> SubmissionHandler<ReplicatedValue> for SharedSubmissionHandler<H>
where
    H: SubmissionHandler<ReplicatedValue>,
{
    fn submit(
        &self,
        submission: Submission<ReplicatedValue>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = rust_quepaxa::Result<SubmissionOutcome<ReplicatedValue>>,
                > + Send
                + '_,
        >,
    > {
        self.0.submit(submission)
    }

    fn receive_decisions(
        &self,
        decisions: Vec<Decision<ReplicatedValue>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = rust_quepaxa::Result<()>> + Send + '_>,
    > {
        self.0.receive_decisions(decisions)
    }
}

/// Push the decisions a peer is missing, without waiting for a new one.
///
/// This is the answer to the gap D11/D13 recorded: a replica that rejoins an
/// idle cluster used to learn nothing until the cluster next decided. Its peers
/// now probe it every [`CATCH_UP_INTERVAL`], find the frontier by status, and
/// send the decisions behind it through the transport's own
/// `InformDecisions` request, which the peer's recorder and state machine apply
/// in slot order exactly as they would a live decision.
async fn catch_up_loop<V, C, S, E>(
    self_id: ReplicaId,
    peers: Vec<C>,
    runtime: Arc<NetworkConsensusHandler<V, C, S, E>>,
    ledger: SharedLedger,
    shutdown: CancellationToken,
) where
    V: Clone + Ord + Send + Sync + 'static,
    C: AsyncRecorderClient<V> + Send + Sync + 'static,
    S: rust_quepaxa::RuntimeStateStore<V> + Send + 'static,
    E: StateMachine<V> + Send + 'static,
{
    let mut known: BTreeMap<ReplicaId, u64> = BTreeMap::new();
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => return,
            _ = tokio::time::sleep(CATCH_UP_INTERVAL) => {}
        }
        let head = ledger.committed();
        if head == 0 {
            continue;
        }
        for peer in &peers {
            let peer_id = AsyncRecorderClient::id(peer);
            if peer_id == self_id {
                continue;
            }
            let floor = known.get(&peer_id).copied().unwrap_or(0);
            if floor >= head {
                continue;
            }
            let held = highest_held(peer, floor, head).await;
            if held < head && held > floor {
                known.insert(peer_id, held);
            }
            let mut from = held + 1;
            while from <= head {
                let to = (from + CATCH_UP_BATCH - 1).min(head);
                let mut batch = Vec::with_capacity((to - from + 1) as usize);
                for slot in from..=to {
                    match runtime.decision(SlotIndex::new(slot)).await {
                        Some(decision) => batch.push(decision),
                        // The peer's frontier is ahead of what this replica can
                        // produce: stop rather than send a hole.
                        None => break,
                    }
                }
                if batch.is_empty() {
                    break;
                }
                let sent = batch.len() as u64;
                if peer.inform_decisions(batch).await.is_err() {
                    // The peer is down or refusing; the next tick tries again.
                    break;
                }
                known.insert(peer_id, from + sent - 1);
                from += sent;
            }
            if from > head {
                known.insert(peer_id, head);
            }
        }
    }
}

/// The highest slot `peer` holds, between `floor` (already known) and `head`.
///
/// A status probe answers "does this recorder have this slot", so the frontier is
/// found by binary search rather than by replaying the log: a peer behind by one
/// slot costs the same probes as a peer behind by a million.
async fn highest_held<V, C>(peer: &C, floor: u64, head: u64) -> u64
where
    V: Clone + Ord + Send + Sync + 'static,
    C: AsyncRecorderClient<V> + Send + Sync + 'static,
{
    let has = |slot: u64| {
        let fut = peer.status(SlotIndex::new(slot));
        async move { matches!(fut.await, Ok(Some(_))) }
    };
    let mut lo = floor.min(head);
    let mut hi = head;
    if has(head).await {
        return head;
    }
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if mid > lo && has(mid).await {
            lo = mid;
        } else {
            hi = mid.saturating_sub(1);
        }
        if hi <= lo {
            break;
        }
    }
    lo
}

/// Start one replica of `config`'s deployment over `ledger`.
///
/// This is the single startup path: a standalone process owns `ledger` through
/// [`SharedLedger::open`] and passes a [`StateDirLock`]; the in-process harness
/// passes one ledger shared by its replicas. Nothing else differs, so the fast
/// harness exercises the same server, recorder and state machine the processes
/// run.
async fn start_replica(
    config: &NodeConfig,
    ledger: SharedLedger,
    state_dir: Option<StateDirLock>,
    metrics: Arc<NetworkMetrics>,
) -> Result<RunningNode, String> {
    let deployment = &config.deployment;
    let index = deployment.index_of(config.replica)?;
    let tls = deployment.node_tls(&config.identity)?;
    let peers = deployment.peer_registry();
    let members = deployment
        .members
        .iter()
        .map(|member| member.replica)
        .collect::<Vec<_>>();
    let node_dir = config.state_dir.join(format!("node-{}", config.replica.get()));
    fs::create_dir_all(&node_dir).map_err(|error| {
        format!(
            "could not create node state dir {}: {error}",
            node_dir.display()
        )
    })?;

    let recorder_clients = deployment
        .members
        .iter()
        .map(|recorder| {
            TlsRecorderClient::new(
                config.replica,
                recorder.replica,
                deployment.deployment,
                recorder.address,
                "localhost",
                recorder.identity.certificate_chain_der[0].clone(),
                tls.client.clone(),
                Arc::clone(&metrics),
            )
        })
        .collect::<Vec<_>>();
    let runtime_config = ReplicaRuntimeConfig::new(
        config.replica,
        deployment.lane,
        members.clone(),
        1,
        ReplicaConfig {
            pipeline_len: 4,
            ..ReplicaConfig::default()
        },
        64,
        Duration::ZERO,
    )
    .map_err(|error| format!("replica {} runtime config: {error}", config.replica.get()))?
    .with_auto_schedules()
    .map_err(|error| format!("replica {} schedules: {error}", config.replica.get()))?;
    let consensus = NetworkConsensusHandler::new(
        runtime_config,
        recorder_clients.clone(),
        FileRuntimeStore::new(
            node_dir.join("runtime.snapshot"),
            PostcardRuntimeCodec::default(),
        ),
        LedgerStateMachine {
            ledger: ledger.clone(),
        },
        Arc::clone(&metrics),
    )
    .await
    .map_err(|error| {
        format!(
            "replica {} could not start its consensus handler: {error}",
            config.replica.get()
        )
    })?
    // D10: the gap filler is a value, not a reserved integer.
    .with_noop_value(ReplicatedValue::Noop);
    // The catch-up task reads this replica's decisions from the same runtime the
    // server applies them with, so both share one handler.
    let consensus = Arc::new(consensus);
    let submissions = DeduplicatingSubmissionHandler::new(
        SharedSubmissionHandler(Arc::clone(&consensus)),
        FileSubmissionJournal::open(node_dir.join("submission-journal.bin"), MAX_JOURNAL_BYTES)
            .map_err(|error| {
                format!(
                    "replica {} submission journal: {error}",
                    config.replica.get()
                )
            })?,
    );
    let recorder = DurableRecorderCore::new(
        RecorderConfig::new(config.replica, members.clone(), 1)
            .map_err(|error| format!("replica {} recorder config: {error}", config.replica.get()))?,
        Arc::new(AllowAllAvailability),
        FileRecorderStore::new(
            node_dir.join("recorder.snapshot"),
            PostcardRecorderCodec::default(),
        ),
    )
    .map_err(|error| {
        format!(
            "replica {} recorder store: {error}",
            config.replica.get()
        )
    })?;
    let server = NetworkNodeServer::bind(
        deployment.members[index].address,
        deployment.deployment,
        tls.server.clone(),
        peers,
        recorder,
        submissions,
        Arc::clone(&metrics),
    )
    .await
    .map_err(|error| {
        format!(
            "replica {} could not bind {}: {error}",
            config.replica.get(),
            deployment.members[index].address
        )
    })?;
    let address = server
        .local_addr()
        .map_err(|error| format!("replica {} bound address: {error}", config.replica.get()))?;

    let shutdown = CancellationToken::new();
    let inner = shutdown.clone();
    let replica_label = config.replica.get();
    let handle = tokio::spawn(async move {
        if let Err(error) = server.run(inner).await {
            eprintln!("quepaxa node {replica_label} stopped: {error}");
        }
    });
    // D11 follow-up: a rejoining replica is pushed what it missed instead of
    // waiting for the cluster to decide something new.
    let catch_up = catch_up_loop(
        config.replica,
        recorder_clients,
        Arc::clone(&consensus),
        ledger.clone(),
        shutdown.clone(),
    );
    tokio::spawn(catch_up);
    Ok(RunningNode {
        replica_id: config.replica,
        address,
        ledger,
        state_dir,
        shutdown,
        handle,
    })
}

/// Start a standalone replica process from a validated [`NodeConfig`].
///
/// The process owns its state directory (through [`StateDirLock`]), its ledger
/// and its server; nothing is shared with another process except the
/// deployment description it was configured with.
pub async fn start_node(config: NodeConfig) -> Result<RunningNode, String> {
    config.validate()?;
    let lock = StateDirLock::acquire(&config.state_dir)?;
    let ledger = SharedLedger::open(&config.state_dir)?;
    let metrics = Arc::new(NetworkMetrics::default());
    start_replica(&config, ledger, Some(lock), metrics).await
}

/// A [`TlsSubmitClient`] for `config` pointed at `address`. A client is not a
/// node: it proposes complete values to a replica and holds no state.
pub fn submit_client(
    config: &SubmitConfig,
    address: SocketAddr,
) -> Result<TlsSubmitClient, String> {
    let tls = config.tls()?;
    Ok(TlsSubmitClient::new(
        config.client_id,
        config.deployment.deployment,
        address,
        "localhost",
        config
            .deployment
            .member_address_certificate(address)?,
        tls.client.clone(),
        Arc::new(NetworkMetrics::default()),
    )
    .with_rpc_timeout(config.rpc_timeout))
}

impl DeploymentConfig {
    /// The certificate a client expects from `address` before it connects: the
    /// deployment declares exactly one, so an address that does not belong to a
    /// member is refused instead of being handed some other peer's identity.
    fn member_address_certificate(&self, address: SocketAddr) -> Result<Vec<u8>, String> {
        self.members
            .iter()
            .find(|member| member.address == address)
            .and_then(|member| member.identity.certificate_chain_der.first().cloned())
            .ok_or_else(|| format!("no member of this deployment listens on {address}"))
    }
}

/// A 2-live-of-3 QuePaxa cluster on loopback addresses, plus the client pump
/// that proposes locally-submitted transactions and the shared ledger they
/// land in. All consensus state is durable under `<state_dir>`.
///
/// D11: this is the fast in-process harness, not a deployment. It starts two of
/// [`start_replica`]'s replicas over **one** shared ledger and state directory,
/// which a real deployment never does (there each process owns its own); the
/// harness keeps it because it makes the failure modes it tests reproducible in
/// a single test binary.
pub struct NetworkCluster {
    pub ledger: SharedLedger,
    pub nodes: Vec<RunningNode>,
    /// Socket addresses of all three replicas (including the offline third
    /// member), needed to resume the cluster on the same ports.
    pub addresses: Vec<SocketAddr>,
    deployment: DeploymentConfig,
    pump: tokio::task::JoinHandle<()>,
}

impl NetworkCluster {
    /// Bring up a fresh cluster, picking three ephemeral loopback ports.
    /// `identities` must carry certificates for the three replicas followed by
    /// one client: `[replica1, replica2, replica3, client]`.
    pub async fn begin(
        identities: Vec<TlsIdentity>,
        state_dir: &Path,
    ) -> rust_quepaxa::Result<Self> {
        let reservations = (0..3)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect::<Vec<_>>();
        let addresses = reservations
            .iter()
            .map(|listener| listener.local_addr().unwrap())
            .collect::<Vec<_>>();
        drop(reservations);
        Self::build(identities, state_dir, addresses).await
    }

    /// Restart a previously-running cluster on the same TLS identities, state
    /// directory, and socket addresses. The recorder/runtime/journals/ledger
    /// are reloaded from disk, so committed work survives and fresh slots are
    /// proposed on top of it.
    pub async fn resume(
        identities: Vec<TlsIdentity>,
        state_dir: &Path,
        addresses: Vec<SocketAddr>,
    ) -> rust_quepaxa::Result<Self> {
        assert_eq!(
            addresses.len(),
            3,
            "resume requires the three replica addresses from the previous run"
        );
        Self::build(identities, state_dir, addresses).await
    }

    /// The deployment the harness runs: three replicas from `identities[0..3]`
    /// and one accepted client from `identities[3]`.
    fn deployment(
        identities: Vec<TlsIdentity>,
        addresses: Vec<SocketAddr>,
    ) -> Result<DeploymentConfig, String> {
        if identities.len() != 4 || addresses.len() != 3 {
            return Err(format!(
                "the harness needs 4 identities and 3 addresses, not {} and {}",
                identities.len(),
                addresses.len()
            ));
        }
        Ok(DeploymentConfig {
            deployment: DeploymentId::from_u128(1),
            lane: LaneId::new(1),
            members: identities[..3]
                .iter()
                .zip(&addresses)
                .enumerate()
                .map(|(index, (identity, address))| MemberConfig {
                    replica: ReplicaId::new(index as u64 + 1),
                    address: *address,
                    identity: identity.clone(),
                })
                .collect(),
            clients: vec![ClientConfig {
                client_id: HARNESS_CLIENT_ID,
                identity: identities[3].clone(),
            }],
        })
    }

    async fn build(
        identities: Vec<TlsIdentity>,
        state_dir: &Path,
        addresses: Vec<SocketAddr>,
    ) -> rust_quepaxa::Result<Self> {
        let deployment =
            Self::deployment(identities.clone(), addresses.clone()).map_err(storage_error)?;
        deployment.validate().map_err(storage_error)?;
        let metrics = Arc::new(NetworkMetrics::default());
        let (submitted_tx, submitted_rx) = mpsc::unbounded_channel::<(u64, ReplicatedValue)>();
        let ledger = SharedLedger::new(Some(submitted_tx), Some(LedgerFile::new(state_dir)))
            .map_err(|error| {
                rust_quepaxa::QuePaxaError::StorageError(format!(
                    "could not load the durable ledger: {error}"
                ))
            })?;

        let mut nodes = Vec::new();
        for member in deployment.members.iter().take(LIVE_NODES) {
            let config = NodeConfig {
                deployment: deployment.clone(),
                replica: member.replica,
                state_dir: state_dir.to_path_buf(),
                identity: member.identity.clone(),
                rpc_timeout: Duration::from_millis(DEFAULT_RPC_TIMEOUT_MS),
            };
            let node = start_replica(
                &config,
                ledger.clone(),
                None,
                Arc::clone(&metrics),
            )
            .await
            .map_err(storage_error)?;
            nodes.push(node);
        }

        let client = deployment
            .clients
            .first()
            .ok_or_else(|| storage_error("the harness deployment declares no client".to_owned()))?;
        let tls = MutualTlsConfigs::new(&client.identity, deployment.roots()).map_err(|error| {
            rust_quepaxa::QuePaxaError::TransportError(format!("client TLS setup failed: {error}"))
        })?;
        let submit_client = TlsSubmitClient::new(
            client.client_id,
            deployment.deployment,
            addresses[0],
            "localhost",
            deployment.members[0].identity.certificate_chain_der[0].clone(),
            tls.client.clone(),
            Arc::clone(&metrics),
        )
        .with_rpc_timeout(Duration::from_secs(30));

        let pump_ledger = ledger.clone();
        let pump = tokio::spawn(async move {
            let mut messages = submitted_rx;
            while let Some((id, value)) = messages.recv().await {
                // A failed RPC (e.g. a slow first TLS handshake) is retried;
                // each attempt claims a fresh request id so a durable journal
                // entry can never collide, and the value already carries the
                // payload, so nothing has to survive in a local map for the
                // commit to be applicable.
                loop {
                    let request_id = pump_ledger.next_request_id();
                    match submit_client.submit(request_id, vec![value.clone()]).await {
                        Ok(_) => break,
                        Err(error) => {
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            eprintln!("quepaxa submit retried for id {id}: {error:?}");
                        }
                    }
                }
            }
        });

        Ok(Self {
            ledger,
            nodes,
            addresses,
            deployment,
            pump,
        })
    }

    /// A consensus engine backed by this cluster's replicated ledger.
    pub fn engine(&self) -> NetConsensus {
        NetConsensus::new(self.ledger.clone())
    }

    /// Block until at least `at_least` transactions are committed.
    pub async fn wait_committed(&self, at_least: u64) {
        wait_committed(&self.ledger, at_least).await
    }

    /// Convention marker: queued-but-uncommitted submissions carry a provisional
    /// seq reserved at submit time.
    pub fn pending_seq(&self) -> u64 {
        let state = self.ledger.state.lock().unwrap();
        state.bank.len() as u64 + state.outbox.len() as u64
    }

    /// The deployment the harness runs (its members, addresses and client), so
    /// a test can start an independent process against the same trust boundary.
    pub fn deployment_config(&self) -> &DeploymentConfig {
        &self.deployment
    }

    /// Gracefully stop every node and the pump. Ports are released once each
    /// node's server task has fully returned, so a resumed cluster can rebind
    /// the same addresses.
    pub async fn shutdown(mut self) {
        for node in std::mem::take(&mut self.nodes) {
            node.shutdown().await;
        }
        self.pump.abort();
    }
}

impl Drop for NetworkCluster {
    fn drop(&mut self) {
        for node in &self.nodes {
            node.shutdown.cancel();
        }
        self.pump.abort();
    }
}

fn storage_error(message: String) -> rust_quepaxa::QuePaxaError {
    rust_quepaxa::QuePaxaError::StorageError(message)
}

/// The client id the in-process harness submits under.
const HARNESS_CLIENT_ID: [u8; 16] = [42u8; 16];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signing::Keypair;
    use unfer_protocol::{IdentityOp, IdentityOpKind};

    fn tx(label: u64) -> ConsensusTransaction {
        let did = Keypair::generate();
        ConsensusTransaction::IdentityOp(IdentityOp {
            did: did.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: did.public_key(),
            signature: [0u8; 64],
            seq: label,
            service_endpoint: None,
        })
    }

    // ── The ledger serves only what it made durable ───────────────────────

    /// D12's halted-node observation, in the other applying path: a decision the
    /// ledger could not persist must not be readable through `committed`,
    /// `receipt` or `committed_since`.
    #[test]
    fn a_decision_that_cannot_be_persisted_is_not_served() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "unfer-ledger-fence-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let ledger = SharedLedger::open(&dir).unwrap();
        let value = encode_ledger_value(7, &tx(1)).unwrap();

        let original = std::fs::metadata(&dir).unwrap().permissions();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        let refused = ledger.append(vec![value.clone()]);
        let error = match refused {
            Ok(()) => {
                std::fs::set_permissions(&dir, original).unwrap();
                let _ = std::fs::remove_dir_all(&dir);
                panic!("a read-only state directory must refuse the snapshot");
            }
            Err(error) => error,
        };

        assert_eq!(ledger.committed(), 0, "the bank must not keep the slot");
        assert!(ledger.committed_since(1).is_empty());
        assert_eq!(
            ledger.receipt(7),
            ReceiptState::Unknown(UnknownReason::Unsubmitted),
            "an unwritten slot must not answer as committed"
        );
        assert!(
            !error.is_empty(),
            "the refusal must say why it could not persist"
        );

        // Given the directory back, the same decision commits exactly once.
        std::fs::set_permissions(&dir, original).unwrap();
        ledger.append(vec![value]).expect("the retry must persist");
        assert_eq!(ledger.committed(), 1);
        assert_eq!(ledger.receipt(7), ReceiptState::Committed(Slot(1)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A reservation that could not be persisted is not kept: otherwise a client
    /// that was told the submission failed could have it commit later.
    #[test]
    fn a_reservation_that_cannot_be_persisted_is_not_kept() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "unfer-ledger-reserve-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (pump, _receiver) = mpsc::unbounded_channel();
        let ledger = SharedLedger::new(Some(pump), Some(LedgerFile::new(&dir))).unwrap();
        let original = std::fs::metadata(&dir).unwrap().permissions();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        let reserved = ledger.allocate(tx(1));
        let id_was_refused = reserved.is_err();
        std::fs::set_permissions(&dir, original).unwrap();
        assert!(id_was_refused, "an unwritable state dir must refuse the reservation");
        assert_eq!(ledger.committed(), 0);
        // Nothing is left queued that a later write could pick up.
        assert_eq!(ledger.receipt(1), ReceiptState::Unknown(UnknownReason::Unsubmitted));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── D11: the configuration a process is started from ──────────────────

    /// A JSON deployment with `members` replicas, `clients` one client, and the
    /// first node's key material attached.
    fn node_json(deployment: &str, members: &[(u64, &str)], state_dir: &str) -> String {
        use rcgen::{CertifiedKey, generate_simple_self_signed};
        let mut certificates = Vec::new();
        let mut keys = Vec::new();
        for _ in members {
            let CertifiedKey { cert, key_pair } =
                generate_simple_self_signed(["localhost".to_owned()]).unwrap();
            certificates.push(hex::encode(cert.der()));
            keys.push(hex::encode(key_pair.serialize_der()));
        }
        let CertifiedKey { cert, key_pair } =
            generate_simple_self_signed(["localhost".to_owned()]).unwrap();
        let client_certificate = hex::encode(cert.der());
        let _ = key_pair;
        let members_json = members
            .iter()
            .zip(&certificates)
            .map(|((replica, address), certificate)| {
                format!(
                    "{{\"replica\":{replica},\"address\":\"{address}\",\"certificate_hex\":\"{certificate}\"}}"
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"replica\":{},\"state_dir\":\"{state_dir}\",\"certificate_hex\":\"{}\",\"key_hex\":\"{}\",\
             \"deployment\":{{\"deployment\":\"{deployment}\",\"lane\":1,\"members\":[{members_json}],\
             \"clients\":[{{\"client_id_hex\":\"{}\",\"certificate_hex\":\"{client_certificate}\"}}]}}}}",
            members.first().map(|(replica, _)| *replica).unwrap_or(1),
            certificates.first().cloned().unwrap_or_default(),
            keys.first().cloned().unwrap_or_default(),
            "07".repeat(16),
        )
    }

    #[test]
    fn a_zero_deployment_id_is_refused() {
        let text = node_json("0", &[(1, "127.0.0.1:17001"), (2, "127.0.0.1:17002")], "/tmp/d11");
        let error = NodeConfig::from_json(&text).unwrap_err();
        assert!(error.contains("deployment id must not be zero"), "{error}");
    }

    #[test]
    fn members_must_be_consecutive_replicas_from_one() {
        let text = node_json(
            "1",
            &[(1, "127.0.0.1:17001"), (3, "127.0.0.1:17002")],
            "/tmp/d11",
        );
        let error = NodeConfig::from_json(&text).unwrap_err();
        assert!(error.contains("replicas 1..=n in order"), "{error}");
    }

    #[test]
    fn two_replicas_cannot_share_an_address() {
        let text = node_json(
            "1",
            &[(1, "127.0.0.1:17001"), (2, "127.0.0.1:17001")],
            "/tmp/d11",
        );
        let error = NodeConfig::from_json(&text).unwrap_err();
        assert!(error.contains("both listen on"), "{error}");
    }

    #[test]
    fn a_state_dir_must_be_absolute_and_exclusive() {
        let relative = node_json(
            "1",
            &[(1, "127.0.0.1:17001"), (2, "127.0.0.1:17002")],
            "node-1",
        );
        let error = NodeConfig::from_json(&relative).unwrap_err();
        assert!(error.contains("must be absolute"), "{error}");

        // A directory locked by a live process cannot be claimed again, which is
        // what stops two nodes from sharing one application store.
        let dir = std::env::temp_dir().join(format!("unfer-d11-lock-{}", std::process::id()));
        let _first = StateDirLock::acquire(&dir).expect("first claim");
        let error = StateDirLock::acquire(&dir).unwrap_err();
        assert!(error.contains("locked by another node process"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_undeclared_client_is_refused() {
        let text = node_json(
            "1",
            &[(1, "127.0.0.1:17001"), (2, "127.0.0.1:17002")],
            "/tmp/d11",
        );
        let node: serde_json::Value = serde_json::from_str(&text).unwrap();
        let deployment = node.get("deployment").unwrap().clone();
        let submit = format!(
            "{{\"client_id_hex\":\"{}\",\"certificate_hex\":\"{}\",\"key_hex\":\"00\",\"deployment\":{deployment}}}",
            "08".repeat(16),
            node.get("certificate_hex").unwrap().as_str().unwrap(),
        );
        let error = SubmitConfig::from_json(&submit).unwrap_err();
        assert!(error.contains("is not declared by this deployment"), "{error}");
    }

    fn mem_ledger() -> SharedLedger {
        let (submitted, _rx) = mpsc::unbounded_channel::<(u64, ReplicatedValue)>();
        SharedLedger::new(Some(submitted), None).expect("in-memory ledger cannot fail")
    }

    #[test]
    fn append_is_idempotent_across_replicas_sharing_one_ledger() {
        let ledger = mem_ledger();
        let transaction = tx(1);
        let id = ledger.allocate(transaction.clone()).expect("durable reserve");
        let value = encode_ledger_value(id, &transaction).unwrap();

        // Both live nodes see the same decision and both apply it.
        ledger.append(vec![value.clone()]).expect("persist");
        ledger.append(vec![value]).expect("persist");

        assert_eq!(ledger.committed(), 1, "no double-apply");
        assert!(ledger.state.lock().unwrap().outbox.is_empty());
    }

    /// D10's whole point: the value carries the payload, so a ledger that never
    /// saw the submission applies it exactly once.
    #[test]
    fn a_committed_value_applies_on_a_ledger_that_never_queued_it() {
        let submitting = mem_ledger();
        let transaction = tx(7);
        let id = submitting
            .allocate(transaction.clone())
            .expect("durable reserve");
        let value = encode_ledger_value(id, &transaction).unwrap();

        // The value is the only carrier: it decodes without the outbox.
        assert_eq!(decode_ledger_value(&value).unwrap(), Some((id, transaction)));

        // A different node's ledger: no outbox entry, no payload map entry.
        let peer = mem_ledger();
        assert!(peer.state.lock().unwrap().outbox.is_empty());
        peer.append(vec![value.clone()]).expect("apply from the value alone");
        assert_eq!(peer.committed(), 1);
        assert_eq!(peer.committed_since(1).len(), 1, "the payload is in the bank");
        assert_eq!(
            peer.receipt(id),
            ReceiptState::Committed(Slot(1)),
            "the id inside the value is what the receipt answers"
        );
        // And re-delivering the same decision changes nothing.
        peer.append(vec![value]).expect("persist");
        assert_eq!(peer.committed(), 1);
    }

    /// D14 step 3: the job fence is sourced from the ledger's live committed
    /// watermark — the ordered point every replica applies decisions at — so a
    /// claim made at a committed slot is refused once a later decision moves
    /// the ledger past it.
    #[test]
    fn the_job_fence_is_owned_and_durably_held_by_the_committed_ledger() {
        use crate::jobs::JobState;

        let dir = std::env::temp_dir().join(format!(
            "unfer-d14-fence-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let claim = {
            let ledger = SharedLedger::open(&dir).expect("open committed state");
            ledger
                .with_jobs(|jobs| {
                    jobs.set_deployment("deploy-1");
                    jobs.enqueue("settle-auction");
                })
                .unwrap();
            let claim = ledger
                .with_jobs(|jobs| jobs.claim("settle-auction", "worker-a"))
                .unwrap()
                .expect("claim under the ledger lock");
            assert_eq!(
                ledger.current_fence("settle-auction").as_ref(),
                Some(claim.fence()),
                "the committed state owns the current fence"
            );
            claim
        };

        // Reopen the committed state: the fence survives, so a restarted
        // process can take over and the old worker's fence is dead.
        let reopened = SharedLedger::open(&dir).expect("reopen committed state");
        assert_eq!(
            reopened.current_fence("settle-auction").as_ref(),
            Some(claim.fence()),
            "the fence is durable, not worker state"
        );
        reopened
            .with_jobs(|jobs| jobs.reclaim("settle-auction", "worker-b"))
            .unwrap()
            .expect("the restarted process takes over");
        let stale = reopened
            .with_jobs(|jobs| jobs.settle(&claim, "worker-a"))
            .unwrap();
        assert!(stale.is_err(), "the old fence is stale after the reclaim");
        let state = reopened
            .with_jobs(|jobs| jobs.state("settle-auction"))
            .unwrap();
        assert_eq!(state, Some(JobState::Claimed), "worker-b still holds it");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// D14: a job command is an ordered decision, so two replicas that apply
    /// the same command stream converge on the same fence — and a command that
    /// is refused is a deterministic no-op that changes neither replica.
    #[test]
    fn a_job_command_converges_across_replicas_and_a_refusal_mutates_nothing() {
        use crate::jobs::{JobClaim, JobCommand, JobState};

        let a = mem_ledger();
        let b = mem_ledger();
        for ledger in [&a, &b] {
            ledger
                .with_jobs(|jobs| {
                    jobs.set_deployment("deploy-1");
                    jobs.enqueue("settle-auction");
                })
                .unwrap();
        }

        let claim = encode_job_value(&JobCommand::Claim {
            resource: "settle-auction".to_string(),
            holder: "worker-a".to_string(),
        })
        .unwrap();
        a.append(vec![claim.clone()]).unwrap();
        b.append(vec![claim]).unwrap();
        let fence = a.current_fence("settle-auction").expect("fenced");
        assert_eq!(
            Some(&fence),
            b.current_fence("settle-auction").as_ref(),
            "both replicas computed the same fence from the same command"
        );

        // A takeover moves the committed fence on both replicas.
        let takeover = encode_job_value(&JobCommand::TakeOver {
            resource: "settle-auction".to_string(),
            epoch: 2,
            holder: "worker-b".to_string(),
        })
        .unwrap();
        a.append(vec![takeover.clone()]).unwrap();
        b.append(vec![takeover]).unwrap();
        let newer = a.current_fence("settle-auction").expect("fenced");
        assert!(newer.supersedes(&fence));
        assert_eq!(Some(&newer), b.current_fence("settle-auction").as_ref());

        // The old worker's late settle is refused identically by both.
        let stale = encode_job_value(&JobCommand::Settle {
            claim: JobClaim::for_test(fence, "worker-a"),
        })
        .unwrap();
        a.append(vec![stale.clone()]).expect("a refusal is not an apply error");
        b.append(vec![stale]).expect("a refusal is not an apply error");
        for ledger in [&a, &b] {
            assert_eq!(
                Some(&newer),
                ledger.current_fence("settle-auction").as_ref(),
                "a refused command must not change the fence"
            );
            assert_eq!(
                ledger.with_jobs(|jobs| jobs.state("settle-auction")).unwrap(),
                Some(JobState::Claimed),
                "a refused command must not change the job state"
            );
        }
    }

    /// The outbox is a local durability device only: it is drained by the
    /// decision that carries the same id, and never read to apply a value.
    #[test]
    fn the_outbox_is_drained_by_the_decision_and_is_not_needed_to_apply_it() {
        let ledger = mem_ledger();
        let transaction = tx(3);
        let id = ledger.allocate(transaction.clone()).expect("durable reserve");
        assert_eq!(ledger.receipt(id), ReceiptState::Pending);

        let value = encode_ledger_value(id, &transaction).unwrap();
        // Drop the outbox entry by hand: applying must still work, because the
        // id comes from the value, not from the map.
        ledger.state.lock().unwrap().outbox.remove(&id);
        ledger.append(vec![value]).expect("persist");
        assert_eq!(ledger.committed(), 1);
        assert_eq!(ledger.receipt(id), ReceiptState::Committed(Slot(1)));
    }

    #[test]
    fn a_value_of_an_unknown_schema_or_undecodable_payload_is_refused() {
        let ledger = mem_ledger();
        let foreign = ReplicatedValue::Command(Command::new(999, vec![1, 2, 3]).unwrap());
        let error = ledger.append(vec![foreign]).expect_err("unknown schema");
        assert!(error.contains("schema 999"), "got: {error}");
        assert_eq!(ledger.committed(), 0, "a refused value commits nothing");

        let short = ReplicatedValue::Command(Command::new(LEDGER_TX_SCHEMA, vec![0, 1]).unwrap());
        let error = ledger.append(vec![short]).expect_err("short payload");
        assert!(error.contains("shorter than its id"), "got: {error}");

        let mut payload = 5u64.to_be_bytes().to_vec();
        payload.extend_from_slice(b"{ not a transaction");
        let broken = ReplicatedValue::Command(Command::new(LEDGER_TX_SCHEMA, payload).unwrap());
        let error = ledger.append(vec![broken]).expect_err("undecodable");
        assert!(error.contains("does not decode"), "got: {error}");
        assert_eq!(ledger.committed(), 0);
    }

    #[test]
    fn a_noop_decision_commits_nothing_and_is_not_an_error() {
        let ledger = mem_ledger();
        ledger.append(vec![ReplicatedValue::Noop]).expect("no-op");
        assert_eq!(ledger.committed(), 0);
        assert_eq!(encode_ledger_value(0, &tx(1)).unwrap().encode().unwrap()[6], 1);
        assert!(decode_ledger_value(&ReplicatedValue::Noop).unwrap().is_none());
    }

    #[test]
    fn ledger_file_round_trips_committed_and_queued_state() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("unfer-ledger-file-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = Some(LedgerFile::new(&dir));

        // Phase 1: one committed, one still queued in the outbox.
        let (submitted, _rx) = mpsc::unbounded_channel::<(u64, ReplicatedValue)>();
        let ledger = SharedLedger::new(Some(submitted), file.clone()).expect("fresh ledger");
        let first = tx(1);
        let second = tx(2);
        let applied = ledger.allocate(first.clone()).expect("durable reserve");
        let in_flight = ledger.allocate(second.clone()).expect("durable reserve");
        ledger
            .append(vec![encode_ledger_value(applied, &first).unwrap()])
            .expect("persist");
        drop(ledger);

        // Phase 2: "crash" and reload. Committed work is in the bank; the
        // queued submission is still in the outbox and is re-proposed.
        let (submitted, _rx) = mpsc::unbounded_channel::<(u64, ReplicatedValue)>();
        let ledger = SharedLedger::new(Some(submitted), file).expect("reload");
        assert_eq!(ledger.committed(), 1, "committed log reloaded");
        assert!(ledger.state.lock().unwrap().outbox.contains_key(&in_flight));
        ledger
            .append(vec![encode_ledger_value(in_flight, &second).unwrap()])
            .expect("persist");
        assert_eq!(ledger.committed(), 2);
        drop(ledger);

        // Phase 3: reload once more; everything is in the bank now.
        let (submitted, _rx) = mpsc::unbounded_channel::<(u64, ReplicatedValue)>();
        let ledger = SharedLedger::new(Some(submitted), Some(LedgerFile::new(&dir))).expect("reload");
        assert_eq!(ledger.committed(), 2);
        assert!(ledger.state.lock().unwrap().outbox.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_ledger_file_fails_loudly_at_load() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "unfer-ledger-corrupt-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let ledger_file = LedgerFile::new(&dir);
        assert!(
            ledger_file.load().is_ok(),
            "a missing ledger file is a fresh ledger, not an error"
        );
        std::fs::write(ledger_file.path, b"{ not valid ledger json").unwrap();

        let (submitted, _rx) = mpsc::unbounded_channel::<(u64, ReplicatedValue)>();
        // A corrupt durable ledger must propagate as an error, never silently
        // reset the committed log to empty (the old `unwrap_or_default` path).
        let error = SharedLedger::new(Some(submitted), Some(LedgerFile::new(&dir)))
            .expect_err("corrupt ledger must not load");

        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            error.contains("corrupt"),
            "the propagated error should name the corruption, got: {error}"
        );
    }
}
