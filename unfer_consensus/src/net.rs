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
//! [`ConsensusNode`]: crate::node::ConsensusNode

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use rust_quepaxa::network::{
    DeduplicatingSubmissionHandler, DeploymentId, InMemorySubmissionJournal, MutualTlsConfigs,
    NetworkConsensusHandler, NetworkMetrics, NetworkNodeServer, PeerIdentity, TlsIdentity,
    TlsRecorderClient, TlsSubmitClient,
};
use rust_quepaxa::{
    AllowAllAvailability, Decision, DurableRecorderCore, InMemoryRecorderStore,
    InMemoryRuntimeStore, LaneId, ReplicaConfig, ReplicaId, ReplicaRuntimeConfig, RecorderConfig,
    StateMachine,
};
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use unfer_protocol::{ConsensusTransaction, Diagnostic};

use crate::engine::ConsensusEngine;

/// Value id QuePaxa uses to fill gaps in the committed prefix; the state
/// machine treats it as a no-op.
pub const NOOP_VALUE: u64 = u64::MAX;

const LIVE_NODES: usize = 2;
const CLIENT_NODES: usize = 1;

/// The replicated outcome: a pending store of submitted transactions, the
/// ordered committed log, and progress bookkeeping for the pump task.
#[derive(Clone)]
pub struct SharedLedger {
    bank: Arc<RwLock<Vec<ConsensusTransaction>>>,
    pending: Arc<Mutex<BTreeMap<u64, ConsensusTransaction>>>,
    next_id: Arc<AtomicU64>,
    committed_tx: watch::Sender<u64>,
    submitted: mpsc::UnboundedSender<u64>,
}

impl SharedLedger {
    fn new(submitted: mpsc::UnboundedSender<u64>) -> Self {
        let (committed_tx, _) = watch::channel(0);
        Self {
            bank: Arc::new(RwLock::new(Vec::new())),
            pending: Arc::new(Mutex::new(BTreeMap::new())),
            next_id: Arc::new(AtomicU64::new(0)),
            committed_tx,
            submitted,
        }
    }

    /// Reserve a unique id for a submitted transaction and enqueue it for the
    /// pump task to propose to the cluster.
    fn allocate(&self, tx: ConsensusTransaction) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.pending.lock().unwrap().insert(id, tx);
        let _ = self.submitted.send(id);
        id
    }

    /// Retrieve and consume a proposed payload once its id is committed.
    fn take(&self, id: u64) -> Option<ConsensusTransaction> {
        self.pending.lock().unwrap().remove(&id)
    }

    /// Append a batch of committed transactions, in decision order.
    fn append(&self, txs: Vec<ConsensusTransaction>) {
        let mut bank = self.bank.write().unwrap();
        bank.extend(txs);
        let _ = self.committed_tx.send(bank.len() as u64);
    }

    /// Number of transactions committed so far.
    pub fn committed(&self) -> u64 {
        self.bank.read().unwrap().len() as u64
    }

    /// Committed prefix in `(1-based_seq, tx)` form, from `from_seq` onward.
    pub fn committed_since(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)> {
        let bank = self.bank.read().unwrap();
        bank.iter()
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
        Ok(self.ledger.allocate(tx))
    }

    fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)> {
        self.ledger.committed_since(from_seq)
    }

    fn current_seq(&self) -> u64 {
        self.ledger.committed()
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
        let txs = decision
            .value_ids
            .iter()
            .filter_map(|id| self.ledger.take(*id))
            .collect::<Vec<_>>();
        self.ledger.append(txs);
        Ok(())
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
/// land in.
pub struct NetworkCluster {
    pub ledger: SharedLedger,
    pub nodes: Vec<ClusterNode>,
    pump: tokio::task::JoinHandle<()>,
}

impl NetworkCluster {
    /// Bring up the cluster. `identities` must carry certificates for the
    /// three replicas followed by one client: `[replica1, replica2, replica3,
    /// client]`.
    pub async fn begin(identities: Vec<TlsIdentity>) -> rust_quepaxa::Result<Self> {
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

        let reservations = (0..3)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect::<Vec<_>>();
        let addresses = reservations
            .iter()
            .map(|listener| listener.local_addr().unwrap())
            .collect::<Vec<_>>();
        drop(reservations);

        let metrics = Arc::new(NetworkMetrics::default());
        let (submitted_tx, submitted_rx) = mpsc::unbounded_channel::<u64>();
        let ledger = SharedLedger::new(submitted_tx);

        let mut nodes = Vec::new();
        for index in 0..LIVE_NODES {
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
                InMemoryRuntimeStore::default(),
                LedgerStateMachine {
                    ledger: ledger.clone(),
                },
                Arc::clone(&metrics),
            )
            .await?
            .with_noop_value(NOOP_VALUE);
            let submissions =
                DeduplicatingSubmissionHandler::new(consensus, InMemorySubmissionJournal::default());
            let recorder = DurableRecorderCore::new(
                RecorderConfig::new(members[index], members.clone(), 1)?,
                Arc::new(AllowAllAvailability),
                InMemoryRecorderStore::default(),
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

        let pump = tokio::spawn(async move {
            let mut request_id = 0u64;
            let mut messages = submitted_rx;
            while let Some(id) = messages.recv().await {
                request_id += 1;
                // A failed RPC (e.g. a slow first TLS handshake) is retried;
                // the payload stays in the pending store until it commits.
                loop {
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
        self.ledger.bank.read().unwrap().len() as u64 + self.ledger.pending.lock().unwrap().len() as u64
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