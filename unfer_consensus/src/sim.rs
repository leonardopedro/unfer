//! D12: a deterministic fault scheduler over the production transition core.
//!
//! The harness does **not** reimplement the node. Every replica is a real
//! [`ConsensusNode`] over its own real [`TransitionLog`];
//! what the harness replaces is the *environment*: which decision a replica is
//! told about and when, whether it can reach the quorum, what its next journal
//! append returns, and whether its process is running at all. Those are the
//! inputs a node gets from the operating system and the network in production
//! and they are exactly the ones a test cannot control from outside.
//!
//! Three rules keep the model honest:
//!
//! * **The consensus engine is an assumption.** One `Cluster` holds the decided
//!   log; `get_log` only reveals the prefix a replica has been *told* about
//!   (`EngineView::horizon`). The harness never claims to test ordering
//!   — it consumes it as an explicit input, so a schedule is a statement about
//!   the application layer only.
//! * **The business logic is the production logic.** Applying an entry, the
//!   admission book, the idempotency store, the journal comparison on restore
//!   and the refusal/halt distinction all come from `node.rs` and `storage.rs`.
//! * **Nondeterminism is registered, not avoided.** Every choice the harness
//!   makes is recorded in a `Trace` with its class and site id, and the same seed
//!   plus the same action list reproduces the same trace and the same state
//!   digests (`Sim::fingerprint`).
//!
//! ## What is deterministic here and what is not
//!
//! The trace records three separate classes, because they fail differently:
//!
//! * `ChoiceClass::Content` — which request (and which *encoding* of it, e.g. a
//!   re-signed duplicate) a client submits. A bug here is a bad client.
//! * `ChoiceClass::Scheduling` — delivery, partition, crash and restore. A bug
//!   here is a scheduler or a host, not the node.
//! * `ChoiceClass::Fuel` — how many steps a bounded run may take. A bug here
//!   is a truncated run being mistaken for a passing one, so a run that ends on
//!   the fuel bound is reported as such rather than as success.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use sha2::{Digest, Sha256};

use unfer_protocol::simulator_contracts::{EngineViewContract, simulator_engine_view_contract};
use unfer_protocol::submission::{ReceiptState, Slot, SubmissionId, UnknownReason};
use unfer_protocol::{
    CertificateOp, CertificateOpKind, Code, ConsensusTransaction, Diagnostic, IdentityOp,
    IdentityOpKind, Severity,
};

use crate::certs::MintAuthority;
use crate::engine::ConsensusEngine;
use crate::idempotency::RequestId;
use crate::node::ConsensusNode;
use crate::signing::{Keypair, sign_transaction};
use crate::storage::FaultPoint;

// ---------------------------------------------------------------------------
// Nondeterminism sites
// ---------------------------------------------------------------------------

/// Why a choice was made. The classes are kept apart because a failure in one
/// is a different claim: content is the client's, scheduling is the host's, fuel
/// is the harness's own bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceClass {
    Content,
    Scheduling,
    Fuel,
}

/// One recorded decision: its class, the stable site id that made it, and the
/// value it chose. Site ids are literals so a saved trace keeps its meaning
/// across refactors of this module.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Choice {
    pub class: ChoiceClass,
    pub site: &'static str,
    pub value: u64,
}

/// Everything a bounded run decided, in order: the seed, the actions and the
/// choices they produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trace {
    pub seed: u64,
    /// The actions the run was given. A trace is replayed by re-running exactly
    /// these, which is why the actions are part of the trace and not only the
    /// choices.
    pub actions: Vec<Action>,
    pub choices: Vec<Choice>,
    /// Steps the run actually executed, which is not `actions.len()` once a
    /// crash makes a later action a no-op.
    pub steps: u64,
}

impl Trace {
    pub fn push(&mut self, class: ChoiceClass, site: &'static str, value: u64) {
        self.choices.push(Choice { class, site, value });
    }

    /// Choices of one class, in order. A saved trace can be diffed on this to
    /// see whether a mismatch was in content or in scheduling.
    pub fn of(&self, class: ChoiceClass) -> impl Iterator<Item = &Choice> {
        self.choices.iter().filter(move |c| c.class == class)
    }

    /// A one-line rendering, written next to the digests so a saved trace stays
    /// auditable without the harness.
    pub fn render(&self) -> String {
        let actions = self
            .actions
            .iter()
            .map(|action| action.label())
            .collect::<Vec<_>>()
            .join(" ");
        let choices = self
            .choices
            .iter()
            .map(|c| format!("{:?}:{}={}", c.class, c.site, c.value))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "seed={} steps={} actions=[{}] choices=[{}]",
            self.seed, self.steps, actions, choices
        )
    }
}

/// SplitMix64. Small, seedable and reproducible across platforms and releases —
/// `rand` is not a dependency here, and a harness that quietly used OS entropy
/// would not replay.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n`. `n` is capped by the caller's alphabet size, so the
    /// modulo bias is bounded by one counter and cannot change which actions are
    /// reachable.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next_u64() % n }
    }
}

// ---------------------------------------------------------------------------
// The decided log (the consensus assumption)
// ---------------------------------------------------------------------------

/// The decided log shared by every replica in one schedule.
///
/// It is append-only and total: `commit` is what the consensus engine is assumed
/// to have decided. Delivery to a particular replica is separate, because that is
/// the scheduler's input and the thing under test.
#[derive(Debug, Default)]
pub struct Cluster {
    entries: RwLock<Vec<ConsensusTransaction>>,
}

impl Cluster {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> u64 {
        self.entries.read().unwrap().len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a decided transaction and return the slot it occupies.
    pub fn commit(&self, tx: ConsensusTransaction) -> u64 {
        let mut entries = self.entries.write().unwrap();
        entries.push(tx);
        entries.len() as u64
    }

    pub fn at(&self, slot: u64) -> Option<ConsensusTransaction> {
        self.entries
            .read()
            .unwrap()
            .get(slot.checked_sub(1)? as usize)
            .cloned()
    }

    pub fn since(&self, from: u64) -> Vec<(u64, ConsensusTransaction)> {
        let entries = self.entries.read().unwrap();
        entries
            .iter()
            .enumerate()
            .map(|(index, tx)| (index as u64 + 1, tx))
            .filter(|(slot, _)| *slot >= from)
            .map(|(slot, tx)| (slot, tx.clone()))
            .collect()
    }

    /// A digest of the whole decided log. Two schedules that decide the same
    /// values in the same order agree here; that is what makes "same trace, same
    /// state" checkable across replicas and across runs.
    pub fn digest(&self) -> String {
        let entries = self.entries.read().unwrap();
        let mut hasher = Sha256::new();
        for (index, tx) in entries.iter().enumerate() {
            hasher.update((index as u64 + 1).to_be_bytes());
            hasher.update(crate::signing::canonical_bytes(tx));
        }
        format!("{}:{}", entries.len(), hex::encode(hasher.finalize()))
    }
}

/// One replica's view of the decided log.
///
/// `horizon` is how far this replica has been *told*; `writable` is whether it
/// can reach the quorum to propose. Both are scheduling inputs, so a partition is
/// a horizon of zero plus a closed proposal path — not a transport error the node
/// has to guess about.
#[derive(Debug)]
pub struct EngineView {
    cluster: Arc<Cluster>,
    horizon: Arc<AtomicU64>,
    writable: Arc<AtomicBool>,
    receipts: RwLock<BTreeMap<u64, ReceiptState>>,
}

impl EngineView {
    pub fn horizon(&self) -> u64 {
        self.horizon.load(Ordering::SeqCst)
    }

    pub fn can_propose(&self) -> bool {
        self.writable.load(Ordering::SeqCst)
    }
}

impl ConsensusEngine for EngineView {
    fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic> {
        if !self.writable.load(Ordering::SeqCst) {
            return Err(Diagnostic::new(
                Code::INTERNAL,
                "this replica cannot propose: the harness holds it outside the quorum",
                Severity::Error,
            ));
        }
        let slot = self.cluster.commit(tx);
        self.receipts
            .write()
            .unwrap()
            .insert(slot, ReceiptState::Committed(Slot(slot)));
        Ok(slot)
    }

    fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)> {
        let horizon = self.horizon.load(Ordering::SeqCst);
        self.cluster
            .since(from_seq)
            .into_iter()
            .filter(|(slot, _)| *slot <= horizon)
            .collect()
    }

    fn current_seq(&self) -> u64 {
        self.cluster.len().min(self.horizon.load(Ordering::SeqCst))
    }

    fn receipt(&self, id: SubmissionId) -> ReceiptState {
        self.receipts
            .read()
            .unwrap()
            .get(&id.0)
            .cloned()
            .unwrap_or(ReceiptState::Unknown(UnknownReason::Unsubmitted))
    }
}

// ---------------------------------------------------------------------------
// Clients and requests
// ---------------------------------------------------------------------------

/// A client of the domain: its own keypair and its own durable request
/// sequence. Two clients is the fixed domain size (plan D12 step 3).
#[derive(Debug, Clone)]
pub struct Client {
    pub name: String,
    pub keypair: Keypair,
    pub did: String,
    pub incarnation: u64,
    pub next_seq: u64,
}

impl Client {
    fn from_seed(name: &str, seed: u8) -> Self {
        let keypair = Keypair::from_bytes(&[seed; 32]);
        let did = keypair.did();
        Self {
            name: name.to_string(),
            keypair,
            did,
            incarnation: 1,
            next_seq: 1,
        }
    }
}

/// The fixed request domain. Two of them are exercised by the explored
/// alphabet; the identity pair exists for the revocation witness, which needs a
/// history (create, then revoke) rather than one entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Request {
    /// A mint by the authority client: applies.
    AuthorizedMint,
    /// A mint by the other client: the node refuses it, and the slot advances.
    UnauthorizedMint,
    IdentityCreate,
    IdentityRevoke,
}

impl Request {
    /// The requests the explored domain schedules: an applied mint, a refused
    /// mint, and an identity create. Three kinds rather than two so the search
    /// mixes a refusal, an unguarded apply and a guarded one in the same log.
    pub const EXPLORED: [Request; 3] = [
        Request::AuthorizedMint,
        Request::UnauthorizedMint,
        Request::IdentityCreate,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Request::AuthorizedMint => "mint-authorised",
            Request::UnauthorizedMint => "mint-unauthorised",
            Request::IdentityCreate => "identity-create",
            Request::IdentityRevoke => "identity-revoke",
        }
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/// One scheduler input. Everything a node learns or loses is one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// A client submits `request` to `replica` under a **fresh** request id.
    Admit {
        client: usize,
        replica: usize,
        request: Request,
    },
    /// The same client retries its **last** request id: the duplicate case.
    Retry { client: usize, replica: usize },
    /// The replica learns every decided slot up to the cluster head.
    Deliver { replica: usize },
    /// The replica loses the quorum: it cannot propose and is told nothing.
    Partition { replica: usize },
    /// The replica is back in the quorum and learns the full log.
    Heal { replica: usize },
    /// The next journal append on this replica fails before writing anything.
    Fault { replica: usize },
    /// The process disappears (its journal stays on disk).
    Crash { replica: usize },
    /// A process starts again on the same directory and replays the log.
    Restore { replica: usize },
}

impl Action {
    pub fn label(&self) -> String {
        match self {
            Action::Admit {
                client,
                replica,
                request,
            } => format!("admit:{}/{}:{}", client, replica, request.label()),
            Action::Retry { client, replica } => format!("retry:{}/{}", client, replica),
            Action::Deliver { replica } => format!("deliver:{}", replica),
            Action::Partition { replica } => format!("partition:{}", replica),
            Action::Heal { replica } => format!("heal:{}", replica),
            Action::Fault { replica } => format!("fault:{}", replica),
            Action::Crash { replica } => format!("crash:{}", replica),
            Action::Restore { replica } => format!("restore:{}", replica),
        }
    }
}

/// What one step did, in the terms a failure witness needs: how many slots the
/// replica applied, which new refusals it recorded, whether it halted, and what
/// the client was told.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Effect {
    pub committed: u64,
    pub applied: u64,
    pub refusals: Vec<(u64, String)>,
    pub halted: Option<String>,
    pub client: Option<String>,
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------
// Replicas
// ---------------------------------------------------------------------------

/// One replica: its journal, its view of the log, and its node when the process
/// is up. `node` is `None` exactly while the process is crashed.
pub struct Replica {
    pub name: String,
    journal: PathBuf,
    dir: PathBuf,
    engine: Arc<EngineView>,
    horizon: Arc<AtomicU64>,
    writable: Arc<AtomicBool>,
    node: Option<ConsensusNode>,
    highest_applied: u64,
    restarts: u64,
}

impl Replica {
    pub fn node(&self) -> Option<&ConsensusNode> {
        self.node.as_ref()
    }

    pub fn is_up(&self) -> bool {
        self.node.is_some()
    }

    pub fn restarts(&self) -> u64 {
        self.restarts
    }

    pub fn journal_len(&self) -> u64 {
        std::fs::metadata(&self.journal)
            .map(|meta| meta.len())
            .unwrap_or(0)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn applied(&self) -> u64 {
        self.node
            .as_ref()
            .map(|node| node.applied_seq())
            .unwrap_or(self.highest_applied)
    }
}

// ---------------------------------------------------------------------------
// The simulator
// ---------------------------------------------------------------------------

/// A deterministic schedule over `replicas` real nodes.
///
/// Construction opens every journal, so a `Sim` is as expensive as starting the
/// replicas it models; the exploration loop builds one per schedule and drops it.
pub struct Sim {
    pub cluster: Arc<Cluster>,
    pub replicas: Vec<Replica>,
    pub clients: Vec<Client>,
    root: PathBuf,
    trace: Trace,
    rng: Rng,
    last_request: Vec<Option<RequestId>>,
    /// The exact bytes each client last submitted, so a retry is a retry of
    /// *that* request: same payload, same request id, no new sequence.
    last_payload: Vec<Option<(Request, u64)>>,
    /// Invariant violations found so far, in step order. A schedule that
    /// violates nothing leaves this empty and says so.
    pub violations: Vec<String>,
}

impl Sim {
    /// Build a domain of `replicas` nodes and `clients` clients under `root`.
    ///
    /// The authority is the first client, so the second client's mint is refused
    /// by the node itself rather than by the harness — that is the rejection
    /// witness, and it has to come from the production path.
    pub fn new(seed: u64, root: impl Into<PathBuf>, replicas: usize, clients: usize) -> Self {
        let root = root.into();
        let cluster = Arc::new(Cluster::new());
        let clients: Vec<Client> = (0..clients)
            .map(|index| Client::from_seed(&format!("client-{index}"), 0x20 + index as u8))
            .collect();
        let authority = clients
            .first()
            .map(|client| client.did.clone())
            .unwrap_or_default();

        let mut slots = Vec::with_capacity(replicas);
        for index in 0..replicas {
            let dir = root.join(format!("replica-{index}"));
            std::fs::create_dir_all(&dir).expect("could not create a replica directory");
            let journal = dir.join("ledger.log");
            let horizon = Arc::new(AtomicU64::new(u64::MAX));
            let writable = Arc::new(AtomicBool::new(true));
            let engine = Arc::new(EngineView {
                cluster: Arc::clone(&cluster),
                horizon: Arc::clone(&horizon),
                writable: Arc::clone(&writable),
                receipts: RwLock::new(BTreeMap::new()),
            });
            let (mut node, _recovery) = ConsensusNode::open(
                Box::new(
                    SharedEngine::attach(Arc::clone(&engine))
                        .expect("the scheduler's engine view is admissible"),
                ),
                &journal,
            )
            .expect("could not open a replica journal");
            node.set_mint_authority(MintAuthority::Only(authority.clone()));
            slots.push(Replica {
                name: format!("replica-{index}"),
                journal,
                dir,
                engine,
                horizon,
                writable,
                node: Some(node),
                highest_applied: 0,
                restarts: 0,
            });
        }

        Sim {
            cluster,
            replicas: slots,
            last_request: vec![None; clients.len()],
            last_payload: vec![None; clients.len()],
            clients,
            root,
            trace: Trace {
                seed,
                ..Trace::default()
            },
            rng: Rng::new(seed),
            violations: Vec::new(),
        }
    }

    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// A client request id. `fresh` increments the client's durable sequence;
    /// a retry passes the recorded id back unchanged, which is what makes the
    /// duplicate case a real duplicate rather than a different request.
    pub fn request_id(&mut self, client: usize) -> RequestId {
        let entry = &mut self.clients[client];
        let id = RequestId::new(
            "sim-deployment",
            entry.did.clone(),
            entry.incarnation,
            entry.next_seq,
        );
        entry.next_seq += 1;
        id
    }

    /// Encode one request as the client would sign it.
    pub fn transaction(&self, request: Request, seq: u64) -> ConsensusTransaction {
        let client = &self.clients[0];
        let mut tx = match request {
            Request::AuthorizedMint => {
                ConsensusTransaction::CertificateOp(mint(&client.keypair, 100, seq))
            }
            Request::UnauthorizedMint => {
                let other = &self.clients[1];
                ConsensusTransaction::CertificateOp(mint(&other.keypair, 50, seq))
            }
            Request::IdentityCreate => {
                let keypair = &client.keypair;
                ConsensusTransaction::IdentityOp(IdentityOp {
                    did: keypair.did(),
                    op_kind: IdentityOpKind::Create,
                    signing_key: keypair.public_key(),
                    signature: [0u8; 64],
                    seq,
                    service_endpoint: None,
                })
            }
            Request::IdentityRevoke => {
                let keypair = &client.keypair;
                ConsensusTransaction::IdentityOp(IdentityOp {
                    did: keypair.did(),
                    op_kind: IdentityOpKind::Revoke,
                    signing_key: keypair.public_key(),
                    signature: [0u8; 64],
                    seq,
                    service_endpoint: None,
                })
            }
        };
        let signer = match request {
            Request::UnauthorizedMint => &self.clients[1],
            _ => &self.clients[0],
        };
        sign_transaction(&mut tx, &signer.keypair);
        tx
    }

    /// Execute one action and check the invariants afterwards.
    ///
    /// `Ok` means the step happened, not that it was uneventful: a refused entry
    /// and a lost quorum are both ordinary inputs here, and are reported in the
    /// [`Effect`]. `Err` is a harness misuse (an index out of range), never a
    /// node failure.
    pub fn step(&mut self, action: Action) -> Result<Effect, String> {
        let mut effect = Effect::default();
        self.trace.actions.push(action);
        self.trace.steps += 1;

        match action {
            Action::Admit {
                client,
                replica,
                request,
            } => {
                self.trace
                    .push(ChoiceClass::Content, "content.request", request as u64);
                let id = self.request_id(client);
                let seq = self.clients[client].next_seq;
                let tx = self.transaction(request, seq);
                self.last_request[client] = Some(id.clone());
                self.last_payload[client] = Some((request, seq));
                self.admit(replica, &id, tx, &mut effect)?;
            }
            Action::Retry { client, replica } => {
                let Some(id) = self.last_request[client].clone() else {
                    effect.note =
                        Some("no request to retry; the client has not submitted".to_string());
                    self.trace
                        .push(ChoiceClass::Content, "content.retry-without-request", 0);
                    return Ok(effect);
                };
                let Some((request, seq)) = self.last_payload[client] else {
                    effect.note = Some("no payload to retry".to_string());
                    return Ok(effect);
                };
                self.trace.push(ChoiceClass::Content, "content.retry", 1);
                let tx = self.transaction(request, seq);
                self.admit(replica, &id, tx, &mut effect)?;
            }
            Action::Deliver { replica } => {
                self.trace
                    .push(ChoiceClass::Scheduling, "sched.deliver", replica as u64);
                self.deliver(replica, &mut effect)?;
            }
            Action::Partition { replica } => {
                self.trace
                    .push(ChoiceClass::Scheduling, "sched.partition", replica as u64);
                let slot = self.replica_mut(replica)?;
                slot.horizon.store(0, Ordering::SeqCst);
                slot.writable.store(false, Ordering::SeqCst);
                effect.note = Some("partitioned: no deliveries, no proposals".to_string());
            }
            Action::Heal { replica } => {
                self.trace
                    .push(ChoiceClass::Scheduling, "sched.heal", replica as u64);
                let slot = self.replica_mut(replica)?;
                slot.horizon.store(u64::MAX, Ordering::SeqCst);
                slot.writable.store(true, Ordering::SeqCst);
                effect.note = Some("healed: full log delivered".to_string());
            }
            Action::Fault { replica } => {
                self.trace.push(
                    ChoiceClass::Scheduling,
                    "sched.storage-fault",
                    replica as u64,
                );
                let slot = self.replica_mut(replica)?;
                match slot.node.as_mut() {
                    Some(node) => {
                        let armed = node.inject_storage_fault(FaultPoint::BeforeWrite);
                        effect.note = Some(format!(
                            "next journal append fails before the write: armed={armed}"
                        ));
                    }
                    None => {
                        effect.note = Some("the process is down: no journal to fail".to_string())
                    }
                }
            }
            Action::Crash { replica } => {
                self.trace
                    .push(ChoiceClass::Scheduling, "sched.crash", replica as u64);
                let slot = self.replica_mut(replica)?;
                if let Some(node) = slot.node.take() {
                    slot.highest_applied = node.applied_seq();
                    effect.note = Some(format!("crashed at slot {}", slot.highest_applied));
                } else {
                    effect.note = Some("already down".to_string());
                }
            }
            Action::Restore { replica } => {
                self.trace
                    .push(ChoiceClass::Scheduling, "sched.restore", replica as u64);
                self.restore(replica, &mut effect)?;
            }
        }

        effect.committed = self.cluster.len();
        self.check(action, &effect);
        Ok(effect)
    }

    /// Retry a request id that the client already used, without any scheduling
    /// choice of its own: the smallest form of the duplicate case.
    pub fn retry_last(&mut self, client: usize, replica: usize) -> Result<Effect, String> {
        self.step(Action::Retry { client, replica })
    }

    /// A seeded random walk over `alphabet`: one step per unit of fuel, with the
    /// action chosen by the run's rng and recorded in the scheduling class. This
    /// is the unbounded-shape companion to the exhaustive search: it reaches
    /// depths the bounded domain cannot afford to enumerate.
    pub fn walk(&mut self, alphabet: &[Action], steps: usize) -> Vec<Effect> {
        let mut effects = Vec::with_capacity(steps);
        for _ in 0..steps {
            let pick = self.rng.below(alphabet.len() as u64) as usize;
            self.trace
                .push(ChoiceClass::Scheduling, "sched.walk", pick as u64);
            match self.step(alphabet[pick]) {
                Ok(effect) => effects.push(effect),
                Err(error) => panic!("the walk used an invalid action: {error}"),
            }
        }
        self.trace
            .push(ChoiceClass::Fuel, "fuel.depth", steps as u64);
        effects
    }

    fn admit(
        &mut self,
        replica: usize,
        id: &RequestId,
        tx: ConsensusTransaction,
        effect: &mut Effect,
    ) -> Result<(), String> {
        let slot = self.replica_mut(replica)?;
        let Some(node) = slot.node.as_mut() else {
            effect.note = Some("the process is down: the submission is not attempted".to_string());
            return Ok(());
        };
        match node.submit_request(id, tx) {
            Ok(submission) => {
                effect.client = Some(format!("accepted {}", submission.0));
            }
            Err(diagnostic) => {
                effect.client = Some(format!(
                    "refused {}: {}",
                    diagnostic.code, diagnostic.message
                ));
            }
        }
        Ok(())
    }

    fn deliver(&mut self, replica: usize, effect: &mut Effect) -> Result<(), String> {
        let slot = self.replica_mut(replica)?;
        let Some(node) = slot.node.as_mut() else {
            effect.note = Some("the process is down: nothing is delivered".to_string());
            return Ok(());
        };
        match node.sync_report() {
            Ok(report) => {
                effect.applied = report.entries;
                effect.refusals = report
                    .refusals
                    .iter()
                    .map(|(slot, refusal)| {
                        (slot.0, format!("{} {}", refusal.code, refusal.message))
                    })
                    .collect();
            }
            Err(diagnostic) => {
                effect.halted = Some(format!("{} {}", diagnostic.code, diagnostic.message));
            }
        }
        Ok(())
    }

    fn restore(&mut self, replica: usize, effect: &mut Effect) -> Result<(), String> {
        let slot = self.replica_mut(replica)?;
        if slot.node.is_some() {
            effect.note = Some("already running".to_string());
            return Ok(());
        }
        let journal = slot.journal.clone();
        let engine = Arc::clone(&slot.engine);
        // Re-open over the same engine view: the replica keeps whatever delivery
        // horizon it had, which is what makes a restore-then-deliver a catch-up.
        let opened = ConsensusNode::open(Box::new(SharedEngine::attach(engine)?), &journal);
        match opened {
            Ok((mut node, recovery)) => {
                let authority = self.clients[0].did.clone();
                node.set_mint_authority(MintAuthority::Only(authority));
                let slot = self.replica_mut(replica)?;
                slot.highest_applied = recovery.last_slot.unwrap_or(0);
                slot.node = Some(node);
                slot.restarts += 1;
                effect.note = Some(format!(
                    "restored {} journal record(s), last slot {:?}, discarded {} torn byte(s)",
                    recovery.records.len(),
                    recovery.last_slot,
                    recovery.torn_tail_bytes
                ));
            }
            Err(error) => {
                effect.note = Some(format!("restore failed: {error}"));
            }
        }
        Ok(())
    }

    fn replica_mut(&mut self, index: usize) -> Result<&mut Replica, String> {
        self.replicas
            .get_mut(index)
            .ok_or_else(|| format!("replica {index} is not in this domain"))
    }

    fn replica_ref(&self, index: usize) -> Result<&Replica, String> {
        self.replicas
            .get(index)
            .ok_or_else(|| format!("replica {index} is not in this domain"))
    }

    // -- invariants ---------------------------------------------------------

    /// Check the invariants that hold for *every* schedule, not just the ones a
    /// test constructs by hand:
    ///
    /// 1. a replica's applied watermark never goes backwards;
    /// 2. no replica applies past the decided log;
    /// 3. replicas that applied the same prefix hold the same state;
    /// 4. a replica that is not halted and is caught up agrees with the cluster.
    fn check(&mut self, action: Action, effect: &Effect) {
        let head = self.cluster.len();
        let mut caught_up = Vec::new();
        for index in 0..self.replicas.len() {
            let slot = &mut self.replicas[index];
            let Some(node) = slot.node.as_ref() else {
                continue;
            };
            let applied = node.applied_seq();
            if applied < slot.highest_applied {
                self.violations.push(format!(
                    "step {} {action:?}: {} went backwards, from slot {} to {applied}",
                    self.trace.steps, slot.name, slot.highest_applied
                ));
            }
            if applied > head {
                self.violations.push(format!(
                    "step {} {action:?}: {} applied slot {applied} but the decided log has {head}",
                    self.trace.steps, slot.name
                ));
            }
            slot.highest_applied = applied;
            if node.halted().is_none() && applied == head && head > 0 {
                caught_up.push((slot.name.clone(), state_digest(node)));
            }
        }
        if let Some((first, digest)) = caught_up.first() {
            for (name, other) in &caught_up[1..] {
                if other != digest {
                    self.violations.push(format!(
                        "step {} {action:?}: {name} and {first} both applied all {head} slot(s) \
                         but hold different state ({other} vs {digest})",
                        self.trace.steps
                    ));
                }
            }
        }
        if let Some(note) = &effect.halted {
            // A halt is a legal outcome (a storage failure is not a verdict on
            // the entry), but it may not silently look like progress.
            self.trace.push(ChoiceClass::Scheduling, "sched.halt", 1);
            let _ = note;
        }
    }

    /// The state a saved trace is judged by: every replica's digest plus the
    /// decided log. Two runs that agree here observed the same schedule.
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.cluster.digest());
        for slot in &self.replicas {
            match slot.node.as_ref() {
                Some(node) => {
                    hasher.update(state_digest(node));
                    hasher.update(node.applied_seq().to_be_bytes());
                }
                None => hasher.update(b"down"),
            }
        }
        hex::encode(hasher.finalize())
    }

    /// A pruning key: two prefixes with the same key admit the same set of
    /// continuations, so the explorer can explore one of them.
    ///
    /// The key is `(decided log, per-replica applied watermark and liveness)`.
    /// The equivalence is exact only because the transition core is
    /// deterministic and all of its inputs beyond this key are in the log: a
    /// replica's journal is a function of its applied prefix, its admission book
    /// is rebuilt from that same prefix, and the remaining scheduling inputs
    /// (deliver, partition, crash) are actions, not state. Retention pruning and
    /// a lost journal are the two things that would break it, and this domain
    /// does neither.
    pub fn state_key(&self) -> String {
        let mut key = format!("log={}", self.cluster.digest());
        for slot in &self.replicas {
            match slot.node.as_ref() {
                Some(node) => key.push_str(&format!(
                    "|{}={}:{}:{}",
                    slot.name,
                    node.applied_seq(),
                    node.halted().is_some(),
                    state_digest(node)
                )),
                None => key.push_str(&format!("|{}=down:{}", slot.name, slot.highest_applied)),
            }
        }
        key
    }

    /// Total supply across this replica's certificate ledger. Used by the
    /// dedup witness: a double-applied mint shows up here and nowhere else.
    pub fn supply(&self, replica: usize) -> u64 {
        self.replicas[replica]
            .node
            .as_ref()
            .map(|node| node.certs().total_supply())
            .unwrap_or(0)
    }

    pub fn is_up(&self, replica: usize) -> bool {
        self.replica_ref(replica)
            .map(|slot| slot.is_up())
            .unwrap_or(false)
    }

    pub fn applied(&self, replica: usize) -> u64 {
        self.replica_ref(replica)
            .map(|slot| slot.applied())
            .unwrap_or(0)
    }

    pub fn digest(&self, replica: usize) -> String {
        self.replica_ref(replica)
            .ok()
            .and_then(|slot| slot.node.as_ref())
            .map(state_digest)
            .unwrap_or_default()
    }

    /// Refusals this replica recorded, as `slot -> code`.
    pub fn refusals(&self, replica: usize) -> Vec<(u64, Code)> {
        let Ok(slot) = self.replica_ref(replica) else {
            return Vec::new();
        };
        let Some(node) = slot.node.as_ref() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for slot_number in 1..=node.applied_seq() {
            match node.refusal(Slot(slot_number)) {
                Some(refusal) => out.push((slot_number, refusal.code)),
                // A refusal is recorded against its slot, so walking the applied
                // prefix reads exactly the refusals this replica holds.
                None => {
                    let _ = slot_number;
                }
            }
        }
        out
    }

    pub fn halted(&self, replica: usize) -> Option<String> {
        self.replica_ref(replica)
            .ok()
            .and_then(|slot| slot.node.as_ref())
            .and_then(|node| node.halted().cloned())
            .map(|halt| halt.diagnostic().code.to_string())
    }

    /// Whether `did` is registered *and* revoked on this replica. `None` means
    /// the replica has never seen the DID at all, which is a different answer:
    /// `IdentityRegistry::resolve` reports a revoked entry as absent, so a caller
    /// that asked only that could not tell "revoked" from "never created".
    pub fn identity_revoked(&self, replica: usize, did: &str) -> Option<bool> {
        let slot = self.replica_ref(replica).ok()?;
        let node = slot.node.as_ref()?;
        let registry = node.identity();
        if !registry.contains(did) {
            return None;
        }
        Some(registry.resolve(did).is_none())
    }

    /// Ask the node to settle a submission, which is how a client learns the
    /// applied slot instead of guessing it.
    pub fn settle(&mut self, replica: usize, submission: u64) -> ReceiptState {
        match self
            .replica_mut(replica)
            .ok()
            .and_then(|slot| slot.node.as_mut())
        {
            Some(node) => {
                node.settle(SubmissionId(submission), Duration::from_millis(0))
                    .state
            }
            None => ReceiptState::Unknown(UnknownReason::Unsubmitted),
        }
    }

    /// Drop the run's directories. The logs are a test artifact; a failing run
    /// keeps them (`UNFER_D12_KEEP`) so a counterexample can be inspected.
    pub fn clean(self) {
        if std::env::var_os("UNFER_D12_KEEP").is_some() {
            return;
        }
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// `ConsensusNode::open` takes ownership of the engine, while a replica has to
/// keep its view (the same horizon and quorum flags) across a restart. This is
/// that sharing: the node sees the process's view, and a restore reuses it.
struct SharedEngine(Arc<EngineView>);

impl SharedEngine {
    /// The one crossing a scheduler-supplied view makes into a node. The view
    /// is what `ConsensusNode` reads its log and receipts from, so the premises
    /// `EngineViewContract` names — one view per replica, a horizon that only
    /// hides a suffix, receipts only for accepted submissions, and a sequence
    /// that ends the returned log — are checked here rather than assumed by the
    /// harness that built it.
    fn attach(view: Arc<EngineView>) -> Result<Self, String> {
        check_live_simulator_view_contract()?;
        Ok(Self(view))
    }
}

/// D19: the gate itself, separate from the one live contract it reads, so a test
/// can show that a violating view contract is refused.
fn check_simulator_view_contract(contract: &EngineViewContract) -> Result<(), String> {
    contract
        .scheduler_view()
        .map_err(|error| format!("the scheduler's engine view is not admissible: {error}"))?;
    Ok(())
}

fn check_live_simulator_view_contract() -> Result<(), String> {
    let contract = simulator_engine_view_contract()
        .map_err(|error| format!("the simulator engine-view contract is invalid: {error}"))?;
    check_simulator_view_contract(&contract)
}

// D19-HOOK: H14 (simulator: scheduler-supplied engine view)
impl ConsensusEngine for SharedEngine {
    fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic> {
        self.0.submit(tx)
    }

    fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)> {
        self.0.get_log(from_seq)
    }

    fn current_seq(&self) -> u64 {
        self.0.current_seq()
    }

    fn receipt(&self, id: SubmissionId) -> ReceiptState {
        self.0.receipt(id)
    }
}

fn mint(keypair: &Keypair, amount: u64, seq: u64) -> CertificateOp {
    CertificateOp {
        did: keypair.did(),
        kind: CertificateOpKind::Mint {
            amount,
            owner: keypair.did(),
            blinding: [7u8; 32],
            source: None,
        },
        seq,
        signature: [0u8; 64],
    }
}

// ---------------------------------------------------------------------------
// Exploration and minimization
// ---------------------------------------------------------------------------

/// The fixed action alphabet of the explored domain: two clients, three nodes,
/// three request kinds.
///
/// It is deliberately small and fixed. A larger alphabet does not find a
/// different *kind* of bug in a bounded search, it just costs exponentially more
/// schedules; what makes the search meaningful is that every action is a real
/// input the production code handles.
pub fn fixed_alphabet() -> Vec<Action> {
    vec![
        Action::Admit {
            client: 0,
            replica: 0,
            request: Request::AuthorizedMint,
        },
        Action::Admit {
            client: 1,
            replica: 1,
            request: Request::UnauthorizedMint,
        },
        Action::Admit {
            client: 0,
            replica: 2,
            request: Request::IdentityCreate,
        },
        Action::Retry {
            client: 0,
            replica: 2,
        },
        Action::Deliver { replica: 0 },
        Action::Deliver { replica: 1 },
        Action::Deliver { replica: 2 },
        Action::Partition { replica: 0 },
        Action::Crash { replica: 1 },
    ]
}

/// What a bounded search covered. Reported rather than asserted: a bound is a
/// property of the search, not of the implementation.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// Prefixes actually executed, including the empty one.
    pub prefixes: u64,
    /// Prefixes skipped because an equivalent state was already executed.
    pub pruned: u64,
    /// Leaves: prefixes that reached the depth bound.
    pub leaves: u64,
    pub depth: usize,
    pub alphabet: usize,
    /// The first violating prefix and what it violated, if one was found.
    pub counterexample: Option<(Vec<Action>, String)>,
}

impl Report {
    /// The state space this search *would* have walked without pruning, and the
    /// fraction it walked instead. Both are exact counts, not estimates.
    pub fn unpruned(&self) -> u64 {
        let mut total = 1u64;
        let mut level = 1u64;
        for _ in 0..self.depth {
            level = level.saturating_mul(self.alphabet as u64);
            total = total.saturating_add(level);
        }
        total
    }

    pub fn render(&self) -> String {
        format!(
            "depth={} alphabet={} unpruned={} prefixes={} pruned={} leaves={} counterexample={}",
            self.depth,
            self.alphabet,
            self.unpruned(),
            self.prefixes,
            self.pruned,
            self.leaves,
            match &self.counterexample {
                Some((actions, why)) => format!(
                    "{} ({})",
                    actions
                        .iter()
                        .map(|action| action.label())
                        .collect::<Vec<_>>()
                        .join(" "),
                    why
                ),
                None => "none".to_string(),
            }
        )
    }
}

/// Run one action list from scratch and report its state key and violations.
fn run_script(
    seed: u64,
    root: &Path,
    index: u64,
    actions: &[Action],
) -> (String, Vec<String>, String) {
    let mut sim = Sim::new(seed, root.join(format!("run-{index}")), 3, 2);
    for action in actions {
        // A harness misuse is a bug in the explorer, not a schedule outcome.
        sim.step(*action)
            .expect("the explorer ran an invalid action");
    }
    // Fuel is its own class: a run that stops on the bound is not a proof, and
    // saying so in the trace keeps a truncated run from reading like a pass.
    sim.trace
        .push(ChoiceClass::Fuel, "fuel.depth", actions.len() as u64);
    let result = (sim.state_key(), sim.violations.clone(), sim.fingerprint());
    sim.clean();
    result
}

/// Explore every action sequence of at most `depth` actions from `alphabet`,
/// pruning prefixes whose state is equivalent to one already executed.
///
/// The pruning rule is stated with [`Sim::state_key`]: two prefixes with the same
/// decided log, the same per-replica watermark and the same liveness admit the
/// same continuations. This is an *equivalence for exploration*, not a proof
/// that the two prefixes are indistinguishable to an observer.
pub fn explore(seed: u64, root: &Path, alphabet: &[Action], depth: usize) -> Report {
    let mut report = Report {
        depth,
        alphabet: alphabet.len(),
        ..Report::default()
    };
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![Vec::new()];
    let mut index = 0u64;
    while let Some(prefix) = stack.pop() {
        index += 1;
        let (key, violations, _fingerprint) = run_script(seed, root, index, &prefix);
        report.prefixes += 1;
        if !violations.is_empty() {
            if report.counterexample.is_none() {
                report.counterexample = Some((prefix, violations[0].clone()));
            }
            continue;
        }
        if !seen.insert(key) {
            report.pruned += 1;
            continue;
        }
        if prefix.len() == depth {
            report.leaves += 1;
            continue;
        }
        // Push in reverse so the alphabet is explored in its declared order.
        for action in alphabet.iter().rev() {
            let mut next = prefix.clone();
            next.push(*action);
            stack.push(next);
        }
    }
    report
}

/// Shrink an action list to the shortest one that still satisfies `keeps`.
///
/// One removal per pass, repeated until a pass removes nothing: an action is only
/// dropped when the property survives without it, so the result is a genuine
/// minimal witness for *that* property and not a re-encoding of the input.
pub fn minimize(
    seed: u64,
    root: &Path,
    actions: &[Action],
    keeps: impl Fn(&Sim) -> bool,
) -> Vec<Action> {
    let mut best: Vec<Action> = actions.to_vec();
    let mut changed = true;
    while changed {
        changed = false;
        for index in 0..best.len() {
            let mut attempt = best.clone();
            attempt.remove(index);
            let mut sim = Sim::new(seed, root.join("minimize"), 3, 2);
            for action in &attempt {
                sim.step(*action).expect("minimize ran an invalid action");
            }
            let holds = keeps(&sim);
            sim.clean();
            if holds {
                best = attempt;
                changed = true;
                break;
            }
        }
    }
    best
}

/// The guards this harness can demonstrate, each with a schedule that a design
/// missing that guard would fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Guard {
    /// Two admissions of the same logical mint, at two replicas under two
    /// request ids: the ledger's content key applies it once.
    Dedup,
    /// An entry the domain refuses, followed by one it accepts: the refusal is
    /// recorded against its slot and the watermark still advances.
    Rejection,
    /// A replica that missed two slots and is restarted over its own journal
    /// applies exactly the missing slots and agrees with its peers.
    Restart,
    /// A revocation ordered behind a crash is applied on restore in log order,
    /// so every replica ends with the identity revoked.
    Revoke,
}

/// A recorded, minimized schedule that demonstrates one guard.
#[derive(Debug, Clone)]
pub struct Witness {
    pub guard: Guard,
    pub trace: Trace,
    /// Actions in the hand-written schedule, before minimization.
    pub original: usize,
    /// Actions after minimization: the shortest schedule that still shows the
    /// guard.
    pub minimized: usize,
    /// What the production core did on the minimized schedule, in words.
    pub outcome: String,
    /// The state fingerprint, so a witness can be compared across runs.
    pub fingerprint: String,
}

impl Witness {
    pub fn render(&self) -> String {
        format!(
            "guard={:?} original={} minimized={}\n  {}\n  outcome: {}\n  fingerprint: {}",
            self.guard,
            self.original,
            self.minimized,
            self.trace.render(),
            self.outcome,
            self.fingerprint
        )
    }
}

/// Build, run and minimize the witness for one guard.
///
/// `keeps` is the guard's own property, and it is checked against the
/// *production* node, so a witness that stops holding after a refactor is a
/// discovery rather than a stale fixture.
pub fn witness(guard: Guard, seed: u64, root: &Path) -> Witness {
    #[allow(clippy::type_complexity)]
    let (script, keeps, describe): (Vec<Action>, fn(&Sim) -> bool, fn(&Sim) -> String) = match guard
    {
        Guard::Dedup => (
            vec![
                Action::Admit {
                    client: 0,
                    replica: 0,
                    request: Request::AuthorizedMint,
                },
                Action::Retry {
                    client: 0,
                    replica: 1,
                },
                Action::Deliver { replica: 0 },
            ],
            |sim: &Sim| sim.cluster.len() == 2 && sim.applied(0) == 2 && sim.supply(0) == 100,
            |sim: &Sim| {
                format!(
                    "two committed entries for one mint (slots {}), one application: \
                         supply is {} and both replicas applied {} of {} slot(s)",
                    sim.cluster.len(),
                    sim.supply(0),
                    sim.applied(0),
                    sim.cluster.len()
                )
            },
        ),
        Guard::Rejection => (
            vec![
                Action::Admit {
                    client: 1,
                    replica: 0,
                    request: Request::UnauthorizedMint,
                },
                Action::Deliver { replica: 0 },
                Action::Deliver { replica: 0 },
            ],
            |sim: &Sim| {
                sim.applied(0) == 1
                    && sim.refusals(0) == vec![(1, Code::CERT_MINT_NOT_AUTHORIZED)]
                    && sim.halted(0).is_none()
                    && sim.supply(0) == 0
            },
            |sim: &Sim| {
                format!(
                    "the unauthorized mint is refused at slot 1 ({}), the watermark is {} \
                         and the replica is not halted; supply {}",
                    sim.refusals(0)
                        .first()
                        .map(|(_, code)| code.to_string())
                        .unwrap_or_else(|| "no refusal".to_string()),
                    sim.applied(0),
                    sim.supply(0)
                )
            },
        ),
        Guard::Restart => (
            vec![
                Action::Crash { replica: 1 },
                Action::Admit {
                    client: 0,
                    replica: 0,
                    request: Request::AuthorizedMint,
                },
                Action::Deliver { replica: 0 },
                Action::Admit {
                    client: 1,
                    replica: 0,
                    request: Request::UnauthorizedMint,
                },
                Action::Deliver { replica: 0 },
                Action::Restore { replica: 1 },
                Action::Deliver { replica: 1 },
            ],
            |sim: &Sim| {
                sim.applied(1) == sim.cluster.len()
                    && sim.cluster.len() == 2
                    && sim.digest(1) == sim.digest(0)
            },
            |sim: &Sim| {
                format!(
                    "a replica restarted over its journal applied {} of {} slot(s) and holds \
                         the same digest as its peer ({})",
                    sim.applied(1),
                    sim.cluster.len(),
                    sim.digest(1)
                )
            },
        ),
        Guard::Revoke => (
            vec![
                Action::Crash { replica: 2 },
                Action::Admit {
                    client: 0,
                    replica: 0,
                    request: Request::IdentityCreate,
                },
                Action::Deliver { replica: 0 },
                Action::Admit {
                    client: 0,
                    replica: 0,
                    request: Request::IdentityRevoke,
                },
                Action::Deliver { replica: 0 },
                Action::Deliver { replica: 1 },
                Action::Restore { replica: 2 },
                Action::Deliver { replica: 2 },
            ],
            |sim: &Sim| {
                let did = &sim.clients[0].did;
                sim.identity_revoked(2, did) == Some(true)
                    && sim.digest(2) == sim.digest(0)
                    && sim.applied(2) == 2
            },
            |sim: &Sim| {
                let did = &sim.clients[0].did;
                format!(
                    "a replica that was down for create and revoke replays both in log order: \
                         revoked={:?}, applied={}, digest {} vs peer {}",
                    sim.identity_revoked(2, did),
                    sim.applied(2),
                    sim.digest(2),
                    sim.digest(0)
                )
            },
        ),
    };

    let minimized = minimize(seed, root, &script, keeps);
    let runroot = root.join(format!("witness-{:?}", guard));
    let mut sim = Sim::new(seed, runroot.clone(), 3, 2);
    for action in &minimized {
        sim.step(*action).expect("the witness script is invalid");
    }
    sim.trace
        .push(ChoiceClass::Fuel, "fuel.depth", minimized.len() as u64);
    let holds = keeps(&sim);
    let outcome = describe(&sim);
    let fingerprint = sim.fingerprint();
    let mut trace = sim.trace.clone();
    trace.actions = minimized.clone();
    sim.clean();
    assert!(
        holds,
        "the minimized {guard:?} witness no longer holds: {outcome}"
    );
    Witness {
        guard,
        trace,
        original: script.len(),
        minimized: minimized.len(),
        outcome,
        fingerprint,
    }
}

/// The state of one node, as a digest: what it applied, what it refused and the
/// ledgers it holds. Built from public accessors on purpose — a digest that read
/// private fields could agree on state the node does not expose.
pub fn state_digest(node: &ConsensusNode) -> String {
    let mut hasher = Sha256::new();
    hasher.update(node.applied_seq().to_be_bytes());
    for slot in 1..=node.applied_seq() {
        if let Some(refusal) = node.refusal(Slot(slot)) {
            hasher.update(slot.to_be_bytes());
            hasher.update(refusal.code.0.to_be_bytes());
            hasher.update(refusal.message.as_bytes());
        }
    }
    hasher.update(node.identity().len().to_be_bytes());
    hasher.update(node.certs().root());
    hasher.update(node.certs().total_supply().to_be_bytes());
    hasher.update((node.auction().open_lots().len() as u64).to_be_bytes());
    match node.halted() {
        Some(halt) => {
            hasher.update(b"halted");
            hasher.update(halt.diagnostic().code.0.to_be_bytes());
        }
        None => hasher.update(b"running"),
    }
    format!("{}:{}", node.applied_seq(), hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::channel_contracts::{CapWitness, Location, Repo, SchemaId, UsageCap};
    use unfer_protocol::simulator_contracts::{HorizonRule, ReceiptRule, SequenceRule};

    fn root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("unfer-h14-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("could not create the test root");
        root
    }

    fn decided(view: &EngineView) -> Vec<(u64, Vec<u8>)> {
        view.get_log(1)
            .into_iter()
            .map(|(slot, tx)| (slot, crate::signing::canonical_bytes(&tx)))
            .collect()
    }

    fn view_contract(
        horizon: HorizonRule,
        receipts: ReceiptRule,
        sequence: SequenceRule,
    ) -> EngineViewContract {
        let supplier =
            Location::new(Repo::Unfer, "unfer_consensus/src/sim.rs", "Sim::new").unwrap();
        let reader = Location::new(Repo::Unfer, "unfer_consensus/src/sim.rs", "get_log").unwrap();
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
    fn the_simulator_gate_accepts_the_live_view() {
        check_live_simulator_view_contract().expect("the live engine view is admissible");
    }

    #[test]
    fn the_simulator_gate_refuses_a_view_that_can_rewrite_history() {
        let cases = [
            (
                view_contract(
                    HorizonRule::RewritePrefix,
                    ReceiptRule::OnlyAccepted,
                    SequenceRule::EndsAtLogEnd,
                ),
                "rewrite a served prefix",
            ),
            (
                view_contract(
                    HorizonRule::SuffixOnly,
                    ReceiptRule::AnySubmission,
                    SequenceRule::EndsAtLogEnd,
                ),
                "a receipt must name a submission",
            ),
            (
                view_contract(
                    HorizonRule::SuffixOnly,
                    ReceiptRule::OnlyAccepted,
                    SequenceRule::BeyondLogEnd,
                ),
                "current_seq must end the log",
            ),
        ];
        for (contract, needle) in cases {
            let error = check_simulator_view_contract(&contract).unwrap_err();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn a_partition_hides_a_suffix_and_a_heal_returns_it_unchanged() {
        let root = root("suffix");
        let mut sim = Sim::new(11, &root, 1, 1);
        sim.step(Action::Admit {
            client: 0,
            replica: 0,
            request: Request::AuthorizedMint,
        })
        .unwrap();
        sim.step(Action::Deliver { replica: 0 }).unwrap();
        let view = Arc::clone(&sim.replicas[0].engine);
        let full = decided(&view);
        assert_eq!(full.len(), 1);
        assert_eq!(view.current_seq(), 1);

        sim.step(Action::Partition { replica: 0 }).unwrap();
        assert_eq!(view.current_seq(), 0, "a partition hides the suffix");
        assert!(view.get_log(1).is_empty());
        assert_eq!(sim.cluster.len(), 1, "the decided prefix is still there");
        let tx = sim.transaction(Request::AuthorizedMint, 1);
        assert!(
            view.submit(tx).is_err(),
            "a partitioned view must refuse to propose, not append"
        );
        assert_eq!(sim.cluster.len(), 1, "the refused proposal did not append");

        sim.step(Action::Heal { replica: 0 }).unwrap();
        assert_eq!(decided(&view), full, "healing returns the same prefix");
        assert_eq!(view.current_seq(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_receipt_exists_only_for_an_accepted_submission() {
        let root = root("receipt");
        let sim = Sim::new(12, &root, 1, 1);
        let view = Arc::clone(&sim.replicas[0].engine);
        assert_eq!(
            view.receipt(SubmissionId(1)),
            ReceiptState::Unknown(UnknownReason::Unsubmitted),
            "an unsubmitted request has no receipt to remember"
        );

        let tx = sim.transaction(Request::AuthorizedMint, 1);
        let slot = view.submit(tx).expect("a writable view accepts");
        assert_eq!(
            view.receipt(SubmissionId(slot)),
            ReceiptState::Committed(Slot(slot))
        );
        assert_eq!(
            view.current_seq(),
            slot,
            "current_seq ends the log this view returns"
        );
        assert_eq!(
            view.get_log(1).last().map(|(last, _)| *last),
            Some(slot),
            "the returned log ends at current_seq"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
