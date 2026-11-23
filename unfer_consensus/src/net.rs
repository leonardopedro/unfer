//! Phase 6 testnet: `ConsensusNode` running on a real QuePaxa cluster.
//!
//! The `network` feature wires `rust-quepaxa`'s mutually-authenticated TLS
//! transport. The value being agreed on is a `u64` transaction id; the payload
//! (`ConsensusTransaction`) travels in a shared pending store and is applied by
//! a [`LedgerStateMachine`] when its id is committed. Every live node executes
//! the same committed decisions in order, so each peer's [`ConsensusNode`]
//! (built over a [`NetConsensus`] engine) reproduces the identical certificate
//! root.
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
    DeduplicatingSubmissionHandler, DeploymentId, FileSubmissionJournal, MutualTlsConfigs,
    NetworkConsensusHandler, NetworkMetrics, NetworkNodeServer, PeerIdentity,
    PostcardRecorderCodec, PostcardRuntimeCodec, TlsIdentity, TlsRecorderClient, TlsSubmitClient,
};
use rust_quepaxa::{
    AllowAllAvailability, Decision, DurableRecorderCore, FileRecorderStore, FileRuntimeStore,
    LaneId, RecorderConfig, ReplicaConfig, ReplicaId, ReplicaRuntimeConfig, StateMachine,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use unfer_protocol::distributed::{Command, ReplicatedValue};
use unfer_protocol::submission::{ReceiptState, Slot, SubmissionId, UnknownReason};
use unfer_protocol::{Code, ConsensusTransaction, Diagnostic, Severity};

use crate::engine::ConsensusEngine;

/// D10: the schema a replicated ledger submission is read under. A command
/// under any other schema is refused by name rather than applied as a
/// neighbour.
pub const LEDGER_TX_SCHEMA: u16 = 1;

const LIVE_NODES: usize = 2;
const CLIENT_NODES: usize = 1;
const MAX_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

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
}

impl LedgerState {
    fn snapshot(&self, next_id: u64, next_request_id: u64) -> LedgerSnapshot {
        LedgerSnapshot {
            bank: self.bank.clone(),
            outbox: self.outbox.clone(),
            committed_ids: self.committed_ids.clone(),
            next_id,
            next_request_id,
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
    submitted: mpsc::UnboundedSender<(u64, ReplicatedValue)>,
    file: Option<LedgerFile>,
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
    fn new(
        submitted: mpsc::UnboundedSender<(u64, ReplicatedValue)>,
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
            if let Ok(value) = encode_ledger_value(id, &tx) {
                let _ = ledger.submitted.send((id, value));
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
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let value = encode_ledger_value(id, &tx)?;
        {
            let mut state = self.state.lock().unwrap();
            state.outbox.insert(id, tx);
            self.persist_state(&state)?;
        }
        let _ = self.submitted.send((id, value));
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
        let committed;
        {
            let mut state = self.state.lock().unwrap();
            let mut appended = Vec::new();
            for value in &values {
                if let Some((id, tx)) = decode_ledger_value(value)? {
                    // Already committed: the same decision delivered twice adds
                    // nothing. Its outbox entry is gone either way.
                    state.outbox.remove(&id);
                    if !state.committed_ids.contains_key(&id) {
                        appended.push((id, tx));
                    }
                }
            }
            if appended.is_empty() && values.is_empty() {
                return Ok(());
            }
            for (id, tx) in &appended {
                state.bank.push(tx.clone());
                // D05: record id -> slot as each payload enters the bank, so the
                // receipt answers with the position the payload landed at rather
                // than the id it was reserved under.
                let slot = state.bank.len() as u64;
                state.committed_ids.insert(*id, slot);
            }
            committed = state.bank.len() as u64;
            // The decision is acknowledged only once one coherent snapshot of
            // the new state is durable.
            self.persist_state(&state)?;
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

/// A running QuePaxa node: a loopback TLS server owning one replica's recorder
/// core, runtime, and consensus handler.
pub struct ClusterNode {
    pub replica_id: ReplicaId,
    pub address: SocketAddr,
    shutdown: CancellationToken,
    handle: tokio::task::JoinHandle<()>,
}

impl ClusterNode {
    pub async fn shutdown(self) {
        self.shutdown.cancel();
        let _ = self.handle.await;
    }
}

/// A 2-live-of-3 QuePaxa cluster on loopback addresses, plus the client pump
/// that proposes locally-submitted transactions and the shared ledger they
/// land in. All consensus state is durable under `<state_dir>`.
pub struct NetworkCluster {
    pub ledger: SharedLedger,
    pub nodes: Vec<ClusterNode>,
    /// Socket addresses of all three replicas (including the offline third
    /// member), needed to resume the cluster on the same ports.
    pub addresses: Vec<SocketAddr>,
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

    async fn build(
        identities: Vec<TlsIdentity>,
        state_dir: &Path,
        addresses: Vec<SocketAddr>,
    ) -> rust_quepaxa::Result<Self> {
        let deployment = DeploymentId::from_u128(1);
        let members = (1..=3).map(ReplicaId::new).collect::<Vec<_>>();
        let client_id = [42u8; 16];

        let roots = identities
            .iter()
            .map(|identity| identity.certificate_chain_der[0].clone())
            .collect::<Vec<_>>();
        let tls = identities
            .iter()
            .map(|identity| MutualTlsConfigs::new(identity, roots.clone()))
            .collect::<rust_quepaxa::Result<Vec<_>>>()?;

        let peers = members
            .iter()
            .enumerate()
            .map(|(index, replica)| {
                (
                    identities[index].certificate_chain_der[0].clone(),
                    PeerIdentity::Replica(*replica),
                )
            })
            .chain([(
                identities[3].certificate_chain_der[0].clone(),
                PeerIdentity::Client(client_id),
            )])
            .collect::<BTreeMap<_, _>>();

        let metrics = Arc::new(NetworkMetrics::default());
        let (submitted_tx, submitted_rx) = mpsc::unbounded_channel::<(u64, ReplicatedValue)>();
        let ledger =
            SharedLedger::new(submitted_tx, Some(LedgerFile::new(state_dir))).map_err(|error| {
                rust_quepaxa::QuePaxaError::StorageError(format!(
                    "could not load the durable ledger: {error}"
                ))
            })?;

        let mut nodes = Vec::new();
        for index in 0..LIVE_NODES {
            let node_dir = state_dir.join(format!("node-{}", members[index].get()));
            fs::create_dir_all(&node_dir).map_err(|error| {
                rust_quepaxa::QuePaxaError::StorageError(format!(
                    "could not create node state dir {}: {error}",
                    node_dir.display()
                ))
            })?;

            let recorder_clients = members
                .iter()
                .enumerate()
                .map(|(target, recorder)| {
                    TlsRecorderClient::new(
                        members[index],
                        *recorder,
                        deployment,
                        addresses[target],
                        "localhost",
                        identities[target].certificate_chain_der[0].clone(),
                        Arc::clone(&tls[index].client),
                        Arc::clone(&metrics),
                    )
                })
                .collect::<Vec<_>>();
            let runtime_config = ReplicaRuntimeConfig::new(
                members[index],
                LaneId::new(1),
                members.clone(),
                1,
                ReplicaConfig {
                    pipeline_len: 4,
                    ..ReplicaConfig::default()
                },
                64,
                Duration::ZERO,
            )?
            .with_auto_schedules()?;
            let consensus = NetworkConsensusHandler::new(
                runtime_config,
                recorder_clients,
                FileRuntimeStore::new(
                    node_dir.join("runtime.snapshot"),
                    PostcardRuntimeCodec::default(),
                ),
                LedgerStateMachine {
                    ledger: ledger.clone(),
                },
                Arc::clone(&metrics),
            )
            .await?
            // D10: the gap filler is a value, not a reserved integer.
            .with_noop_value(ReplicatedValue::Noop);
            let submissions = DeduplicatingSubmissionHandler::new(
                consensus,
                FileSubmissionJournal::open(
                    node_dir.join("submission-journal.bin"),
                    MAX_JOURNAL_BYTES,
                )?,
            );
            let recorder = DurableRecorderCore::new(
                RecorderConfig::new(members[index], members.clone(), 1)?,
                Arc::new(AllowAllAvailability),
                FileRecorderStore::new(
                    node_dir.join("recorder.snapshot"),
                    PostcardRecorderCodec::default(),
                ),
            )?;
            let server = NetworkNodeServer::bind(
                addresses[index],
                deployment,
                Arc::clone(&tls[index].server),
                peers.clone(),
                recorder,
                submissions,
                Arc::clone(&metrics),
            )
            .await?;

            let shutdown = CancellationToken::new();
            let inner = shutdown.clone();
            let replica_label = members[index].get();
            let handle = tokio::spawn(async move {
                if let Err(error) = server.run(inner).await {
                    eprintln!("quepaxa node {replica_label} stopped: {error}");
                }
            });
            nodes.push(ClusterNode {
                replica_id: members[index],
                address: addresses[index],
                shutdown,
                handle,
            });
        }

        let submit_client = TlsSubmitClient::new(
            client_id,
            deployment,
            addresses[0],
            "localhost",
            identities[0].certificate_chain_der[0].clone(),
            Arc::clone(&tls[tls.len() - CLIENT_NODES].client),
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
            pump,
        })
    }

    /// A consensus engine backed by this cluster's replicated ledger.
    pub fn engine(&self) -> NetConsensus {
        NetConsensus::new(self.ledger.clone())
    }

    /// Block until at least `at_least` transactions are committed.
    pub async fn wait_committed(&self, at_least: u64) {
        let mut rx = self.ledger.subscribe();
        loop {
            if *rx.borrow() >= at_least {
                return;
            }
            if rx.changed().await.is_err() {
                return;
            }
        }
    }

    /// Convention marker: queued-but-uncommitted submissions carry a provisional
    /// seq reserved at submit time.
    pub fn pending_seq(&self) -> u64 {
        let state = self.ledger.state.lock().unwrap();
        state.bank.len() as u64 + state.outbox.len() as u64
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

    fn mem_ledger() -> SharedLedger {
        let (submitted, _rx) = mpsc::unbounded_channel::<(u64, ReplicatedValue)>();
        SharedLedger::new(submitted, None).expect("in-memory ledger cannot fail")
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
        let ledger = SharedLedger::new(submitted, file.clone()).expect("fresh ledger");
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
        let ledger = SharedLedger::new(submitted, file).expect("reload");
        assert_eq!(ledger.committed(), 1, "committed log reloaded");
        assert!(ledger.state.lock().unwrap().outbox.contains_key(&in_flight));
        ledger
            .append(vec![encode_ledger_value(in_flight, &second).unwrap()])
            .expect("persist");
        assert_eq!(ledger.committed(), 2);
        drop(ledger);

        // Phase 3: reload once more; everything is in the bank now.
        let (submitted, _rx) = mpsc::unbounded_channel::<(u64, ReplicatedValue)>();
        let ledger = SharedLedger::new(submitted, Some(LedgerFile::new(&dir))).expect("reload");
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
        let error = SharedLedger::new(submitted, Some(LedgerFile::new(&dir)))
            .expect_err("corrupt ledger must not load");

        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            error.contains("corrupt"),
            "the propagated error should name the corruption, got: {error}"
        );
    }
}
