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

use unfer_protocol::submission::{ReceiptState, Slot, SubmissionId, UnknownReason};
use unfer_protocol::{Code, ConsensusTransaction, Diagnostic, Severity};

use crate::engine::ConsensusEngine;

/// Value id QuePaxa uses to fill gaps in the committed prefix; the state
/// machine treats it as a no-op.
pub const NOOP_VALUE: u64 = u64::MAX;

const LIVE_NODES: usize = 2;
const CLIENT_NODES: usize = 1;
const MAX_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

/// Durable ledger snapshot: the committed bank plus everything still queued.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LedgerSnapshot {
    bank: Vec<ConsensusTransaction>,
    pending: BTreeMap<u64, ConsensusTransaction>,
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
/// The committed bank, the queued payloads and the submission-id map used to be
/// read under separate locks when persisting, so a crash could publish a
/// snapshot assembled from two different instants — for example a payload
/// already removed from `pending` and not yet in `bank`, which was a lost
/// transaction that looked like a clean snapshot. One lock makes the snapshot a
/// state that actually existed.
#[derive(Debug, Clone, Default)]
struct LedgerState {
    bank: Vec<ConsensusTransaction>,
    pending: BTreeMap<u64, ConsensusTransaction>,
    committed_ids: BTreeMap<u64, u64>,
}

impl LedgerState {
    fn snapshot(&self, next_id: u64, next_request_id: u64) -> LedgerSnapshot {
        LedgerSnapshot {
            bank: self.bank.clone(),
            pending: self.pending.clone(),
            committed_ids: self.committed_ids.clone(),
            next_id,
            next_request_id,
        }
    }
}

/// The replicated outcome: a durable pending store of submitted transactions,
/// the ordered committed log, and progress bookkeeping for the pump task.
#[derive(Clone)]
pub struct SharedLedger {
    state: Arc<Mutex<LedgerState>>,
    next_id: Arc<AtomicU64>,
    request_seq: Arc<AtomicU64>,
    committed_tx: watch::Sender<u64>,
    submitted: mpsc::UnboundedSender<u64>,
    file: Option<LedgerFile>,
}

impl std::fmt::Debug for SharedLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("SharedLedger")
            .field("committed", &state.bank.len())
            .field("pending", &state.pending.len())
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .field("with_file", &self.file.is_some())
            .finish()
    }
}

impl SharedLedger {
    fn new(
        submitted: mpsc::UnboundedSender<u64>,
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
                pending: loaded.pending,
                committed_ids: loaded.committed_ids,
            })),
            next_id: Arc::new(AtomicU64::new(loaded.next_id)),
            request_seq: Arc::new(AtomicU64::new(loaded.next_request_id)),
            committed_tx,
            submitted,
            file,
        };
        // Re-enqueue anything still queued so the pump re-proposes it after a
        // restart, then persist the reconciled view.
        let outstanding = ledger
            .state
            .lock()
            .unwrap()
            .pending
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for id in outstanding {
            let _ = ledger.submitted.send(id);
        }
        ledger.persist()?;
        Ok(ledger)
    }

    /// Reserve a unique id for a submitted transaction and enqueue it for the
    /// pump task to propose to the cluster. The reserved payload is persisted
    /// before it is acknowledged, so a durable submission survives a crash.
    fn allocate(&self, tx: ConsensusTransaction) -> Result<u64, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut state = self.state.lock().unwrap();
            state.pending.insert(id, tx);
            self.persist_state(&state)?;
        }
        let _ = self.submitted.send(id);
        Ok(id)
    }

    /// Apply a batch of committed value ids in decision order. Fetching from
    /// the pending store, extending the bank, and persisting happen under one
    /// lock hold, so exactly one replica moves each payload into the bank and a
    /// node that dies after reading a decision but before applying it loses
    /// nothing: the payload is still pending, and a restart (or the surviving
    /// peer) re-applies it exactly once.
    fn append(&self, ids: Vec<u64>) -> Result<(), String> {
        let committed;
        {
            let mut state = self.state.lock().unwrap();
            let mut appended = Vec::new();
            for id in ids {
                if let Some(tx) = state.pending.remove(&id) {
                    appended.push((id, tx));
                }
            }
            if appended.is_empty() {
                return Ok(());
            }
            for (id, tx) in appended {
                state.bank.push(tx);
                // D05: record id -> slot as each payload enters the bank, so the
                // receipt answers with the position the payload landed at rather
                // than the id it was reserved under.
                let slot = state.bank.len() as u64;
                state.committed_ids.insert(id, slot);
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
        if state.pending.contains_key(&id) {
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

impl StateMachine<u64> for LedgerStateMachine {
    fn execute(&mut self, decision: &Decision<u64>) -> rust_quepaxa::Result<()> {
        if decision.value_ids == [NOOP_VALUE] {
            return Ok(());
        }
        let ids = decision
            .value_ids
            .iter()
            .copied()
            .filter(|id| *id != NOOP_VALUE)
            .collect::<Vec<_>>();
        self.ledger.append(ids).map_err(|error| {
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
        let (submitted_tx, submitted_rx) = mpsc::unbounded_channel::<u64>();
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
            .with_noop_value(NOOP_VALUE);
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
            while let Some(id) = messages.recv().await {
                // A failed RPC (e.g. a slow first TLS handshake) is retried;
                // each attempt claims a fresh request id so a durable journal
                // entry can never collide, and the payload stays in the pending
                // store until it commits.
                loop {
                    let request_id = pump_ledger.next_request_id();
                    match submit_client.submit(request_id, vec![id]).await {
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
        state.bank.len() as u64 + state.pending.len() as u64
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
        let (submitted, _rx) = mpsc::unbounded_channel::<u64>();
        SharedLedger::new(submitted, None).expect("in-memory ledger cannot fail")
    }

    #[test]
    fn append_is_idempotent_across_replicas_sharing_one_ledger() {
        let ledger = mem_ledger();
        let id = ledger.allocate(tx(1)).expect("durable reserve");

        // Both live nodes see the same decision and both apply it.
        ledger.append(vec![id]).expect("persist");
        ledger.append(vec![id]).expect("persist");

        assert_eq!(ledger.committed(), 1, "no double-apply");
        assert!(ledger.state.lock().unwrap().pending.is_empty());
    }

    #[test]
    fn decided_but_unapplied_value_survives_until_one_replica_applies() {
        let ledger = mem_ledger();
        let id = ledger.allocate(tx(1)).expect("durable reserve");

        // A node reads the decision but dies before applying it; the payload
        // must still be pending so the surviving peer (or a restart) can move
        // it into the bank exactly once.
        assert!(
            ledger.state.lock().unwrap().pending.contains_key(&id),
            "decided-but-unapplied value stays pending"
        );
        ledger.append(vec![id]).expect("persist");
        assert_eq!(ledger.committed(), 1);
        ledger.append(vec![id]).expect("persist");
        assert_eq!(ledger.committed(), 1, "re-applying the same id is a no-op");
    }

    #[test]
    fn ledger_file_round_trips_committed_and_pending_state() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("unfer-ledger-file-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = Some(LedgerFile::new(&dir));

        // Phase 1: one committed, one decided-but-unapplied.
        let (submitted, _rx) = mpsc::unbounded_channel::<u64>();
        let ledger = SharedLedger::new(submitted, file.clone()).expect("fresh ledger");
        let applied = ledger.allocate(tx(1)).expect("durable reserve");
        let in_flight = ledger.allocate(tx(2)).expect("durable reserve");
        ledger.append(vec![applied]).expect("persist");
        drop(ledger);

        // Phase 2: "crash" and reload. Committed work is in the bank; the
        // in-flight value is still pending and gets re-proposed.
        let (submitted, _rx) = mpsc::unbounded_channel::<u64>();
        let ledger = SharedLedger::new(submitted, file).expect("reload");
        assert_eq!(ledger.committed(), 1, "committed log reloaded");
        assert!(ledger.state.lock().unwrap().pending.contains_key(&in_flight));
        ledger.append(vec![in_flight]).expect("persist");
        assert_eq!(ledger.committed(), 2);
        drop(ledger);

        // Phase 3: reload once more; everything is in the bank now.
        let (submitted, _rx) = mpsc::unbounded_channel::<u64>();
        let ledger = SharedLedger::new(submitted, Some(LedgerFile::new(&dir))).expect("reload");
        assert_eq!(ledger.committed(), 2);
        assert!(ledger.state.lock().unwrap().pending.is_empty());

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

        let (submitted, _rx) = mpsc::unbounded_channel::<u64>();
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
