//! D19: checked contracts for the boundaries messages cross.
//!
//! A comment at a call site is not a premise. This module makes the premise a
//! value: a [`ChannelContract`] names the producer and consumer, the schema, the
//! ordering and retry scopes, the authentication, the fault model, a bounded
//! capacity with a witness, the delivery guarantee, and (for a nonmonotone
//! decision) the completeness evidence.
//!
//! Constructors refuse the guarantees that do not follow from the fields:
//!
//! * an unordered fold over subtraction or division
//!   ([`CommutativeFold::new`]);
//! * exactly-once delivery, or a promotion from at-least-once, without a
//!   request-identity (or global) duplicate suppressor carrying its own bound
//!   ([`Delivery::exactly_once`], [`Delivery::promote_to_exactly_once`]);
//! * a nonmonotone decision without a completeness frontier, and a local
//!   timeout offered as one ([`ChannelContract::new`],
//!   [`CompletenessGuarantee::from_local_timeout`]);
//! * a quorum counted per packet instead of per authenticated member
//!   ([`VoteTally`]).
//!
//! The opaque marker types ([`OrderedChannel`], [`ExactlyOnceChannel`],
//! [`CompletenessFrontier`]) are only produced by a contract that actually
//! carries the corresponding field. They are Rust-side evidence, not a proof:
//! the boundary still has to call the operation whose contract it claims.
//!
//! ```
//! use unfer_protocol::channel_contracts::{
//!     CapWitness, CommutativeFold, FoldOp, UsageCap,
//! };
//! let cap = UsageCap::new("in-flight submissions", 64, CapWitness::ProtocolInvariant).unwrap();
//! assert_eq!(cap.max(), 64);
//! assert!(CommutativeFold::new(FoldOp::Add).is_ok());
//! assert!(CommutativeFold::new(FoldOp::Subtract).is_err());
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Which repository a contract location resolves in. The census checker uses
/// the same split as `docs/distributed/contracts.tsv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Repo {
    AustralVm,
    Unfer,
}

/// A source location that produces or consumes a boundary message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    repo: Repo,
    path: &'static str,
    item: &'static str,
}

impl Location {
    pub fn new(repo: Repo, path: &'static str, item: &'static str) -> Result<Self, ContractError> {
        if path.is_empty() {
            return Err(ContractError::EmptyField("location path"));
        }
        if item.is_empty() {
            return Err(ContractError::EmptyField("location item"));
        }
        Ok(Self { repo, path, item })
    }

    pub fn repo(&self) -> Repo {
        self.repo
    }

    pub fn path(&self) -> &'static str {
        self.path
    }

    pub fn item(&self) -> &'static str {
        self.item
    }
}

/// The versioned schema a payload is read under. Version 0 is never a schema,
/// matching the wire codec's own refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaId {
    name: &'static str,
    version: u16,
}

impl SchemaId {
    pub fn new(name: &'static str, version: u16) -> Result<Self, ContractError> {
        if name.is_empty() {
            return Err(ContractError::EmptyField("schema name"));
        }
        if version == 0 {
            return Err(ContractError::ZeroSchemaVersion);
        }
        Ok(Self { name, version })
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn version(&self) -> u16 {
        self.version
    }
}

/// The order a scope actually promises. `PerConnection` is FIFO on one live
/// connection only; it says nothing about a retry that arrives after a
/// reconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderingScope {
    PerConnection,
    TotalOrder,
    Unordered,
}

impl OrderingScope {
    /// Whether this scope orders a retry with the given lifetime.
    pub fn covers(self, retries: RetryScope) -> bool {
        match (self, retries) {
            (OrderingScope::TotalOrder, _) => true,
            (OrderingScope::PerConnection, RetryScope::WithinConnection) => true,
            (OrderingScope::PerConnection, RetryScope::AcrossReconnects) => false,
            (OrderingScope::Unordered, _) => false,
        }
    }
}

/// Whether a logical request retry stays on the connection that carried the
/// first attempt, or may arrive after a reconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryScope {
    WithinConnection,
    AcrossReconnects,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Authentication {
    MutualTls,
    SignedFrame,
    InProcess,
}

/// Delay and loss assumptions recorded with the contract instead of inferred
/// from a working happy path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaultModel {
    loss: Loss,
    reordering: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    None,
    Possible,
}

impl FaultModel {
    pub fn new(loss: Loss, reordering: bool) -> Self {
        Self { loss, reordering }
    }

    pub fn loss(&self) -> Loss {
        self.loss
    }

    pub fn reorders(&self) -> bool {
        self.reordering
    }
}

/// What justifies a capacity number. The point is that a caller can see whether
/// the bound was proved, follows from the layout, or remains the caller's
/// obligation; it is not a free-form comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapWitness {
    OrderedFrontier,
    StaticLayout,
    ProtocolInvariant,
    CallerEstablished,
}

/// A capacity bound that a guarantee depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageCap {
    name: &'static str,
    max: u64,
    witness: CapWitness,
}

impl UsageCap {
    pub fn new(name: &'static str, max: u64, witness: CapWitness) -> Result<Self, ContractError> {
        if name.is_empty() {
            return Err(ContractError::EmptyField("capacity name"));
        }
        if max == 0 {
            return Err(ContractError::ZeroCapacity { name });
        }
        Ok(Self { name, max, witness })
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn max(&self) -> u64 {
        self.max
    }

    pub fn witness(&self) -> CapWitness {
        self.witness
    }
}

/// Where a duplicate can be recognised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateScope {
    PerConnection,
    RequestIdentity,
    Global,
}

/// Duplicate suppression plus the capacity under which it holds. A suppressor
/// with no bound cannot back an exactly-once claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateSuppression {
    scope: DuplicateScope,
    cap: UsageCap,
}

impl DuplicateSuppression {
    pub fn new(scope: DuplicateScope, cap: UsageCap) -> Self {
        Self { scope, cap }
    }

    pub fn scope(&self) -> DuplicateScope {
        self.scope
    }

    pub fn cap(&self) -> &UsageCap {
        &self.cap
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryMode {
    AtMostOnce,
    AtLeastOnce,
    ExactlyOnce,
}

/// The delivery guarantee and, for exactly-once, the suppressor it rests on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delivery {
    mode: DeliveryMode,
    suppression: Option<DuplicateSuppression>,
}

impl Delivery {
    pub fn at_most_once() -> Self {
        Self {
            mode: DeliveryMode::AtMostOnce,
            suppression: None,
        }
    }

    pub fn at_least_once() -> Self {
        Self {
            mode: DeliveryMode::AtLeastOnce,
            suppression: None,
        }
    }

    /// Exactly-once needs a duplicate scope that survives the retry path. A
    /// per-connection suppressor is refused even when it has a capacity bound,
    /// because a reconnect retry is not on that connection.
    pub fn exactly_once(suppression: DuplicateSuppression) -> Result<Self, ContractError> {
        if suppression.scope == DuplicateScope::PerConnection {
            return Err(ContractError::PerConnectionDedupCannotBeExactlyOnce);
        }
        Ok(Self {
            mode: DeliveryMode::ExactlyOnce,
            suppression: Some(suppression),
        })
    }

    /// At-least-once does not promote itself. The caller must supply the
    /// suppressor, and the same per-connection refusal applies.
    pub fn promote_to_exactly_once(
        self,
        suppression: Option<DuplicateSuppression>,
    ) -> Result<Self, ContractError> {
        if self.mode == DeliveryMode::ExactlyOnce {
            return Ok(self);
        }
        let suppression = suppression.ok_or(ContractError::PromotionNeedsDedup)?;
        Delivery::exactly_once(suppression)
    }

    pub fn mode(&self) -> DeliveryMode {
        self.mode
    }

    pub fn suppression(&self) -> Option<&DuplicateSuppression> {
        self.suppression.as_ref()
    }
}

/// A member identity established by authentication, not a connection index or
/// packet counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemberId([u8; 32]);

impl MemberId {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A quorum is a threshold over eligible authenticated members.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quorum {
    threshold: usize,
    eligible: BTreeSet<MemberId>,
}

impl Quorum {
    pub fn new(
        threshold: usize,
        eligible: impl IntoIterator<Item = MemberId>,
    ) -> Result<Self, ContractError> {
        let eligible: BTreeSet<MemberId> = eligible.into_iter().collect();
        if threshold == 0 {
            return Err(ContractError::ZeroQuorumThreshold);
        }
        if threshold > eligible.len() {
            return Err(ContractError::QuorumExceedsMembers {
                threshold,
                members: eligible.len(),
            });
        }
        Ok(Self {
            threshold,
            eligible,
        })
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }

    pub fn members(&self) -> usize {
        self.eligible.len()
    }

    pub fn is_eligible(&self, member: MemberId) -> bool {
        self.eligible.contains(&member)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vote {
    Accept,
    Reject,
}

/// Counts votes by authenticated member, never by packet. A member that repeats
/// a vote is counted once; a member that changes its vote keeps its first
/// recorded vote, so a tally cannot be inflated or rewritten by replay.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VoteTally {
    votes: BTreeMap<MemberId, Vote>,
}

impl VoteTally {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the first vote from `member`. Returns `false` when this member
    /// already voted (the packet is a duplicate, not another voter).
    pub fn record(&mut self, member: MemberId, vote: Vote) -> bool {
        if self.votes.contains_key(&member) {
            return false;
        }
        self.votes.insert(member, vote);
        true
    }

    pub fn distinct_members(&self) -> usize {
        self.votes.len()
    }

    pub fn count_for(&self, vote: Vote) -> usize {
        self.votes
            .values()
            .filter(|recorded| **recorded == vote)
            .count()
    }

    /// Whether distinct, eligible members for `vote` reach the threshold.
    pub fn has_quorum(&self, quorum: &Quorum, vote: Vote) -> bool {
        self.votes
            .iter()
            .filter(|(member, recorded)| **recorded == vote && quorum.is_eligible(**member))
            .count()
            >= quorum.threshold()
    }
}

/// What makes a nonmonotone outcome (close, absence, failure) global knowledge.
/// A local timeout is deliberately not a variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletenessKind {
    OrderedClose,
    Watermark,
    Quorum(Quorum),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletenessGuarantee {
    kind: CompletenessKind,
    cap: UsageCap,
}

impl CompletenessGuarantee {
    pub fn new(kind: CompletenessKind, cap: UsageCap) -> Result<Self, ContractError> {
        Ok(Self { kind, cap })
    }

    /// There is no constructor from a local timeout: a timeout says this caller
    /// stopped waiting, not that every other participant has spoken.
    pub fn from_local_timeout(_cap: UsageCap) -> Result<Self, ContractError> {
        Err(ContractError::LocalTimeoutIsNotCompleteness)
    }

    pub fn kind(&self) -> &CompletenessKind {
        &self.kind
    }

    pub fn cap(&self) -> &UsageCap {
        &self.cap
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionKind {
    Monotone,
    NonMonotone,
}

/// A boundary contract. Every guarantee here is a field, not an inference from
/// neighbouring prose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelContract {
    producer: Location,
    consumer: Location,
    schema: SchemaId,
    ordering: OrderingScope,
    retries: RetryScope,
    authentication: Authentication,
    faults: FaultModel,
    capacity: UsageCap,
    delivery: Delivery,
    completeness: Option<CompletenessGuarantee>,
    decision: DecisionKind,
}

impl ChannelContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer: Location,
        consumer: Location,
        schema: SchemaId,
        ordering: OrderingScope,
        retries: RetryScope,
        authentication: Authentication,
        faults: FaultModel,
        capacity: UsageCap,
        delivery: Delivery,
        completeness: Option<CompletenessGuarantee>,
        decision: DecisionKind,
    ) -> Result<Self, ContractError> {
        if decision == DecisionKind::NonMonotone && completeness.is_none() {
            return Err(ContractError::MissingCompleteness);
        }
        if delivery.mode == DeliveryMode::ExactlyOnce {
            let suppression = delivery
                .suppression
                .as_ref()
                .ok_or(ContractError::PromotionNeedsDedup)?;
            if retries == RetryScope::AcrossReconnects
                && suppression.scope == DuplicateScope::PerConnection
            {
                return Err(ContractError::PerConnectionDedupCannotBeExactlyOnce);
            }
        }
        Ok(Self {
            producer,
            consumer,
            schema,
            ordering,
            retries,
            authentication,
            faults,
            capacity,
            delivery,
            completeness,
            decision,
        })
    }

    pub fn producer(&self) -> &Location {
        &self.producer
    }

    pub fn consumer(&self) -> &Location {
        &self.consumer
    }

    pub fn schema(&self) -> SchemaId {
        self.schema
    }

    pub fn ordering(&self) -> OrderingScope {
        self.ordering
    }

    pub fn retries(&self) -> RetryScope {
        self.retries
    }

    pub fn authentication(&self) -> Authentication {
        self.authentication
    }

    pub fn faults(&self) -> FaultModel {
        self.faults
    }

    pub fn capacity(&self) -> &UsageCap {
        &self.capacity
    }

    pub fn delivery(&self) -> &Delivery {
        &self.delivery
    }

    pub fn completeness(&self) -> Option<&CompletenessGuarantee> {
        self.completeness.as_ref()
    }

    pub fn decision(&self) -> DecisionKind {
        self.decision
    }

    /// A marker only when the ordering field covers the retry lifetime. A
    /// per-connection FIFO contract therefore yields no marker for a retry that
    /// crosses a reconnect.
    pub fn ordered_channel(&self) -> Result<OrderedChannel, ContractError> {
        if self.ordering.covers(self.retries) {
            Ok(OrderedChannel(()))
        } else {
            Err(ContractError::OrderingDoesNotCoverRetries)
        }
    }

    pub fn exactly_once_channel(&self) -> Result<ExactlyOnceChannel, ContractError> {
        if self.delivery.mode == DeliveryMode::ExactlyOnce {
            Ok(ExactlyOnceChannel(()))
        } else {
            Err(ContractError::NotExactlyOnce)
        }
    }

    pub fn completeness_frontier(&self) -> Result<CompletenessFrontier, ContractError> {
        if self.completeness.is_some() {
            Ok(CompletenessFrontier(()))
        } else {
            Err(ContractError::MissingCompleteness)
        }
    }
}

/// The schema a replicated ledger command is read under. The consensus crate
/// re-exports this value instead of keeping a second copy.
pub const CONSENSUS_LEDGER_SCHEMA: u16 = 1;

/// The byte bound on the durable submission journal. Exactly-once delivery on
/// the live path rests on this bound, so it is recorded in the contract rather
/// than inferred from the file layout.
pub const MAX_SUBMISSION_JOURNAL_BYTES: u64 = 16 * 1024 * 1024;

/// The schema this replica's decisions travel under: the transport's
/// `InformDecisions` frame, read by the peer's recorder and state machine.
pub const DECISION_DELIVERY_SCHEMA: u16 = 1;

/// The most decisions one delivery frame may carry. A catch-up push is sized by
/// this bound so a single frame stays inside the transport's frame limit,
/// whatever the length of the log behind it. The contract's capacity field
/// carries the same number, and the sender reads it from the checked contract
/// rather than from a constant beside the send.
pub const MAX_DECISION_BATCH: u64 = 256;

/// The live consensus submission path as a checked contract.
///
/// The sender is `unfer_node submit`, the receiver is `start_replica`; the
/// commit order is the decided log, retries may cross reconnects, mutual TLS
/// authenticates both ends, and the server's durable journal binds
/// `(client_id, request_id)` to one outcome. Exactly-once therefore rests on
/// request identity and the journal byte bound, not on a connection.
pub fn consensus_live_submission_contract() -> Result<ChannelContract, ContractError> {
    let producer = Location::new(
        Repo::Unfer,
        "unfer_consensus/src/bin/unfer_node.rs",
        "submit",
    )?;
    let consumer = Location::new(Repo::Unfer, "unfer_consensus/src/net.rs", "start_replica")?;
    let schema = SchemaId::new("unfer.consensus.ledger-tx", CONSENSUS_LEDGER_SCHEMA)?;
    let capacity = UsageCap::new(
        "durable submission journal bytes",
        MAX_SUBMISSION_JOURNAL_BYTES,
        CapWitness::StaticLayout,
    )?;
    let suppression = DuplicateSuppression::new(DuplicateScope::RequestIdentity, capacity.clone());
    ChannelContract::new(
        producer,
        consumer,
        schema,
        OrderingScope::TotalOrder,
        RetryScope::AcrossReconnects,
        Authentication::MutualTls,
        FaultModel::new(Loss::Possible, true),
        capacity,
        Delivery::exactly_once(suppression)?,
        None,
        DecisionKind::Monotone,
    )
}

/// The peer-decision path as a checked contract.
///
/// A decision is addressed by its slot, retried after a reconnect and applied
/// by the peer's recorder and state machine in slot order. Out-of-order or
/// oversize frames are what the contract exists to refuse, and a lost
/// acknowledgement re-sends the same decision, so delivery is at-least-once:
/// the apply path is idempotent by submission id, not by frame.
pub fn consensus_decision_delivery_contract() -> Result<ChannelContract, ContractError> {
    let producer = Location::new(Repo::Unfer, "unfer_consensus/src/net.rs", "catch_up_loop")?;
    let consumer = Location::new(
        Repo::Unfer,
        "unfer_consensus/src/net.rs",
        "receive_decisions",
    )?;
    let schema = SchemaId::new("quepaxa.inform-decisions", DECISION_DELIVERY_SCHEMA)?;
    let capacity = UsageCap::new(
        "decisions per delivery frame",
        MAX_DECISION_BATCH,
        CapWitness::ProtocolInvariant,
    )?;
    ChannelContract::new(
        producer,
        consumer,
        schema,
        OrderingScope::TotalOrder,
        RetryScope::AcrossReconnects,
        Authentication::MutualTls,
        FaultModel::new(Loss::Possible, true),
        capacity,
        Delivery::at_least_once(),
        None,
        DecisionKind::Monotone,
    )
}

/// The most logical requests one admission book retains before it answers with
/// backpressure. The in-process admission contract's delivery claim is exactly
/// this bound, so the bound lives here and the book reads it rather than
/// keeping a second number that could drift.
pub const MAX_ADMISSION_ENTRIES: u64 = 4096;

/// The in-process admission point as a checked contract.
///
/// `ConsensusNode::submit_request` is the only place a client-named request is
/// admitted, and the handoff to the ordered engine follows it in the same call.
/// There is no transport here, so the authentication is `InProcess` and no
/// message can be lost or reordered between the probe and `engine.submit`; the
/// retry lifetime nevertheless crosses restarts (the request id is durable on
/// the client), so the ordering claim is the engine's total order and the
/// exactly-once claim rests on request identity plus the book's retained bound.
pub fn consensus_inprocess_admission_contract() -> Result<ChannelContract, ContractError> {
    let producer = Location::new(Repo::Unfer, "unfer_consensus/src/node.rs", "submit_request")?;
    let consumer = Location::new(Repo::Unfer, "unfer_consensus/src/node.rs", "submit_tracked")?;
    let schema = SchemaId::new("unfer.consensus.ledger-tx", CONSENSUS_LEDGER_SCHEMA)?;
    let capacity = UsageCap::new(
        "retained admission entries",
        MAX_ADMISSION_ENTRIES,
        CapWitness::ProtocolInvariant,
    )?;
    let suppression = DuplicateSuppression::new(DuplicateScope::RequestIdentity, capacity.clone());
    ChannelContract::new(
        producer,
        consumer,
        schema,
        OrderingScope::TotalOrder,
        RetryScope::AcrossReconnects,
        Authentication::InProcess,
        FaultModel::new(Loss::None, false),
        capacity,
        Delivery::exactly_once(suppression)?,
        None,
        DecisionKind::Monotone,
    )
}

/// The schema a peer recorder request is read under.
pub const RECORDER_RPC_SCHEMA: u16 = 1;

/// The peer recorder RPC as a checked contract.
///
/// A replica proposes decisions and reads a peer's frontier through the
/// certificate-pinned recorder client. Each request names its slot, so the
/// client claims no arrival order; the authentication is the premise the
/// construction actually implements (mutual TLS with the member's pinned
/// certificate), and the bound is the decision batch that the catch-up sender
/// hands to this same client.
pub fn consensus_recorder_rpc_contract() -> Result<ChannelContract, ContractError> {
    let producer = Location::new(
        Repo::Unfer,
        "unfer_consensus/src/net.rs",
        "TlsRecorderClient",
    )?;
    let consumer = Location::new(
        Repo::Unfer,
        "unfer_consensus/src/net.rs",
        "NetworkNodeServer",
    )?;
    let schema = SchemaId::new("quepaxa.recorder-rpc", RECORDER_RPC_SCHEMA)?;
    let capacity = UsageCap::new(
        "decisions per recorder request",
        MAX_DECISION_BATCH,
        CapWitness::ProtocolInvariant,
    )?;
    ChannelContract::new(
        producer,
        consumer,
        schema,
        OrderingScope::Unordered,
        RetryScope::WithinConnection,
        Authentication::MutualTls,
        FaultModel::new(Loss::Possible, true),
        capacity,
        Delivery::at_least_once(),
        None,
        DecisionKind::Monotone,
    )
}

/// Opaque evidence that a contract's ordering covers its retry lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderedChannel(());

/// Opaque evidence that a contract carries exactly-once delivery and its
/// duplicate suppressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactlyOnceChannel(());

/// Opaque evidence that a contract carries a completeness frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletenessFrontier(());

/// The operations an unordered fold is allowed to use. Addition, max, min and
/// set union are commutative; subtraction and division are not, and no amount
/// of "the current usage makes it look ordered" turns them into one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldOp {
    Add,
    Max,
    Min,
    Union,
    Subtract,
    Divide,
}

impl FoldOp {
    pub fn is_commutative(self) -> bool {
        match self {
            FoldOp::Add | FoldOp::Max | FoldOp::Min | FoldOp::Union => true,
            FoldOp::Subtract | FoldOp::Divide => false,
        }
    }
}

impl fmt::Display for FoldOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            FoldOp::Add => "add",
            FoldOp::Max => "max",
            FoldOp::Min => "min",
            FoldOp::Union => "union",
            FoldOp::Subtract => "subtract",
            FoldOp::Divide => "divide",
        };
        f.write_str(name)
    }
}

/// A marker for a fold that may consume an unordered stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommutativeFold {
    op: FoldOp,
}

impl CommutativeFold {
    pub fn new(op: FoldOp) -> Result<Self, ContractError> {
        if !op.is_commutative() {
            return Err(ContractError::NotCommutative { op });
        }
        Ok(Self { op })
    }

    pub fn op(&self) -> FoldOp {
        self.op
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    EmptyField(&'static str),
    ZeroSchemaVersion,
    ZeroCapacity {
        name: &'static str,
    },
    PerConnectionDedupCannotBeExactlyOnce,
    PromotionNeedsDedup,
    MissingCompleteness,
    LocalTimeoutIsNotCompleteness,
    NotCommutative {
        op: FoldOp,
    },
    ZeroQuorumThreshold,
    QuorumExceedsMembers {
        threshold: usize,
        members: usize,
    },
    OrderingDoesNotCoverRetries,
    NotExactlyOnce,
    /// A storage boundary claims one writer; two writers on one path have no
    /// publish order.
    NotSingleWriter {
        writers: u64,
    },
    /// A snapshot's name became visible before its bytes were synced.
    SnapshotPublishedBeforeSync,
    /// A damaged snapshot was reset to a default instead of refused.
    SilentResetOnDamage,
    /// One replica claims more than one engine view; a restart would then read
    /// a decided log the node never wrote to.
    NotSingleView {
        views: u64,
    },
    /// A scheduler horizon change may replace a slot that was already served.
    SchedulerRewritesPrefix,
    /// A receipt was returned for a submission the view never accepted.
    ReceiptWithoutSubmission,
    /// `current_seq` runs past the log `get_log` returns, so `is_synced` can be
    /// true while entries are missing.
    SequenceBeyondLog,
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractError::EmptyField(field) => write!(f, "{field} is empty"),
            ContractError::ZeroSchemaVersion => f.write_str("schema version 0 is not a version"),
            ContractError::ZeroCapacity { name } => {
                write!(f, "capacity {name} must be greater than zero")
            }
            ContractError::PerConnectionDedupCannotBeExactlyOnce => f.write_str(
                "per-connection duplicate suppression cannot back exactly-once delivery",
            ),
            ContractError::PromotionNeedsDedup => f.write_str(
                "at-least-once does not promote to exactly-once without duplicate suppression",
            ),
            ContractError::MissingCompleteness => f.write_str(
                "a nonmonotone decision needs an ordered close, watermark or quorum frontier",
            ),
            ContractError::LocalTimeoutIsNotCompleteness => {
                f.write_str("a local timeout is not evidence of global completeness")
            }
            ContractError::NotCommutative { op } => {
                write!(f, "an unordered fold cannot use noncommutative {op}")
            }
            ContractError::ZeroQuorumThreshold => {
                f.write_str("a quorum threshold must be greater than zero")
            }
            ContractError::QuorumExceedsMembers { threshold, members } => write!(
                f,
                "quorum threshold {threshold} exceeds {members} eligible member(s)"
            ),
            ContractError::OrderingDoesNotCoverRetries => {
                f.write_str("the ordering scope does not cover this retry lifetime")
            }
            ContractError::NotExactlyOnce => {
                f.write_str("the delivery guarantee is not exactly-once")
            }
            ContractError::NotSingleWriter { writers } => write!(
                f,
                "a storage boundary has one writer at a time, not {writers}"
            ),
            ContractError::SnapshotPublishedBeforeSync => f.write_str(
                "a snapshot published before its sync can be lost while its name survives",
            ),
            ContractError::SilentResetOnDamage => {
                f.write_str("a damaged snapshot must be refused, not reset to an empty state")
            }
            ContractError::NotSingleView { views } => {
                write!(f, "a replica has one engine view at a time, not {views}")
            }
            ContractError::SchedulerRewritesPrefix => f.write_str(
                "a scheduler horizon change may hide a suffix, not rewrite a served prefix",
            ),
            ContractError::ReceiptWithoutSubmission => {
                f.write_str("a receipt must name a submission the view accepted")
            }
            ContractError::SequenceBeyondLog => {
                f.write_str("current_seq must end the log get_log returns, or is_synced lies")
            }
        }
    }
}

impl std::error::Error for ContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(byte: u8) -> MemberId {
        MemberId::new([byte; 32])
    }

    fn cap() -> UsageCap {
        UsageCap::new("test bound", 8, CapWitness::StaticLayout).unwrap()
    }

    fn location(item: &'static str) -> Location {
        Location::new(Repo::Unfer, "unfer_consensus/src/net.rs", item).unwrap()
    }

    fn contract(
        ordering: OrderingScope,
        retries: RetryScope,
        delivery: Delivery,
        completeness: Option<CompletenessGuarantee>,
        decision: DecisionKind,
    ) -> Result<ChannelContract, ContractError> {
        ChannelContract::new(
            location("producer"),
            location("consumer"),
            SchemaId::new("test", 1).unwrap(),
            ordering,
            retries,
            Authentication::MutualTls,
            FaultModel::new(Loss::Possible, true),
            cap(),
            delivery,
            completeness,
            decision,
        )
    }

    #[test]
    fn a_packet_count_is_not_a_quorum() {
        let alice = member(1);
        let bob = member(2);
        let quorum = Quorum::new(2, [alice, bob]).unwrap();
        let mut tally = VoteTally::new();

        assert!(tally.record(alice, Vote::Accept));
        assert!(!tally.record(alice, Vote::Accept));
        assert!(!tally.record(alice, Vote::Accept));
        assert_eq!(tally.distinct_members(), 1);
        assert_eq!(tally.count_for(Vote::Accept), 1);
        assert!(
            !tally.has_quorum(&quorum, Vote::Accept),
            "three packets from one member never form a two-member quorum"
        );

        assert!(tally.record(bob, Vote::Accept));
        assert!(tally.has_quorum(&quorum, Vote::Accept));
    }

    #[test]
    fn an_ineligible_member_never_counts_toward_quorum() {
        let alice = member(1);
        let bob = member(2);
        let outsider = member(99);
        let quorum = Quorum::new(2, [alice, bob]).unwrap();
        let mut tally = VoteTally::new();
        tally.record(alice, Vote::Accept);
        tally.record(outsider, Vote::Accept);
        assert!(!tally.has_quorum(&quorum, Vote::Accept));
        tally.record(bob, Vote::Accept);
        assert!(tally.has_quorum(&quorum, Vote::Accept));
    }

    #[test]
    fn an_unordered_fold_refuses_subtraction() {
        assert!(CommutativeFold::new(FoldOp::Add).is_ok());
        assert_eq!(
            CommutativeFold::new(FoldOp::Subtract),
            Err(ContractError::NotCommutative {
                op: FoldOp::Subtract
            })
        );
        assert_eq!(
            CommutativeFold::new(FoldOp::Divide),
            Err(ContractError::NotCommutative { op: FoldOp::Divide })
        );
    }

    #[test]
    fn at_least_once_does_not_promote_itself() {
        let delivery = Delivery::at_least_once();
        assert_eq!(
            delivery.clone().promote_to_exactly_once(None),
            Err(ContractError::PromotionNeedsDedup)
        );
        let suppression = DuplicateSuppression::new(DuplicateScope::RequestIdentity, cap());
        let promoted = delivery
            .promote_to_exactly_once(Some(suppression))
            .expect("a request-identity suppressor is sufficient");
        assert_eq!(promoted.mode(), DeliveryMode::ExactlyOnce);
    }

    #[test]
    fn per_connection_dedup_cannot_be_exactly_once() {
        let suppression = DuplicateSuppression::new(DuplicateScope::PerConnection, cap());
        assert_eq!(
            Delivery::exactly_once(suppression),
            Err(ContractError::PerConnectionDedupCannotBeExactlyOnce)
        );
    }

    #[test]
    fn a_capacity_needs_a_positive_bound() {
        assert_eq!(
            UsageCap::new("empty", 0, CapWitness::ProtocolInvariant),
            Err(ContractError::ZeroCapacity { name: "empty" })
        );
    }

    #[test]
    fn a_local_timeout_is_not_a_completeness_frontier() {
        assert_eq!(
            CompletenessGuarantee::from_local_timeout(cap()),
            Err(ContractError::LocalTimeoutIsNotCompleteness)
        );
    }

    #[test]
    fn a_nonmonotone_decision_without_completeness_is_refused() {
        let error = contract(
            OrderingScope::TotalOrder,
            RetryScope::WithinConnection,
            Delivery::at_least_once(),
            None,
            DecisionKind::NonMonotone,
        )
        .expect_err("close/absence needs a frontier");
        assert_eq!(error, ContractError::MissingCompleteness);
    }

    #[test]
    fn a_nonmonotone_decision_with_an_ordered_close_is_accepted() {
        let completeness = CompletenessGuarantee::new(
            CompletenessKind::OrderedClose,
            UsageCap::new("close frontier", 1, CapWitness::OrderedFrontier).unwrap(),
        )
        .unwrap();
        let contract = contract(
            OrderingScope::TotalOrder,
            RetryScope::WithinConnection,
            Delivery::at_least_once(),
            Some(completeness),
            DecisionKind::NonMonotone,
        )
        .expect("an ordered close is completeness evidence");
        assert!(contract.completeness_frontier().is_ok());
    }

    #[test]
    fn per_connection_fifo_does_not_order_a_reconnect_retry() {
        let contract = contract(
            OrderingScope::PerConnection,
            RetryScope::AcrossReconnects,
            Delivery::at_least_once(),
            None,
            DecisionKind::Monotone,
        )
        .unwrap();
        assert!(!contract.ordering().covers(contract.retries()));
        assert_eq!(
            contract.ordered_channel(),
            Err(ContractError::OrderingDoesNotCoverRetries)
        );
    }

    #[test]
    fn total_order_orders_a_reconnect_retry() {
        let contract = contract(
            OrderingScope::TotalOrder,
            RetryScope::AcrossReconnects,
            Delivery::at_least_once(),
            None,
            DecisionKind::Monotone,
        )
        .unwrap();
        assert!(contract.ordered_channel().is_ok());
    }

    #[test]
    fn exactly_once_marker_needs_exactly_once_delivery() {
        let at_least_once = contract(
            OrderingScope::TotalOrder,
            RetryScope::AcrossReconnects,
            Delivery::at_least_once(),
            None,
            DecisionKind::Monotone,
        )
        .unwrap();
        assert_eq!(
            at_least_once.exactly_once_channel(),
            Err(ContractError::NotExactlyOnce)
        );

        let suppression = DuplicateSuppression::new(DuplicateScope::RequestIdentity, cap());
        let exactly_once = contract(
            OrderingScope::TotalOrder,
            RetryScope::AcrossReconnects,
            Delivery::exactly_once(suppression).unwrap(),
            None,
            DecisionKind::Monotone,
        )
        .unwrap();
        assert!(exactly_once.exactly_once_channel().is_ok());
    }

    #[test]
    fn the_live_consensus_contract_is_ordered_and_exactly_once() {
        let contract = consensus_live_submission_contract().expect("live path contract");
        assert_eq!(contract.ordering(), OrderingScope::TotalOrder);
        assert_eq!(contract.retries(), RetryScope::AcrossReconnects);
        assert!(contract.ordered_channel().is_ok());
        assert!(contract.exactly_once_channel().is_ok());
        assert_eq!(
            contract.delivery().suppression().map(|s| s.scope()),
            Some(DuplicateScope::RequestIdentity)
        );
    }

    #[test]
    fn the_consensus_decision_contract_is_ordered_and_frame_bounded() {
        let contract = consensus_decision_delivery_contract().expect("decision delivery contract");
        assert_eq!(contract.ordering(), OrderingScope::TotalOrder);
        assert_eq!(contract.retries(), RetryScope::AcrossReconnects);
        assert!(contract.ordered_channel().is_ok());
        assert_eq!(contract.capacity().max(), MAX_DECISION_BATCH);
        assert_eq!(contract.capacity().witness(), CapWitness::ProtocolInvariant);
        assert_eq!(contract.delivery().mode(), DeliveryMode::AtLeastOnce);
    }

    #[test]
    fn the_inprocess_admission_contract_is_request_identity_exactly_once() {
        let contract = consensus_inprocess_admission_contract().expect("in-process contract");
        assert_eq!(contract.authentication(), Authentication::InProcess);
        assert_eq!(contract.ordering(), OrderingScope::TotalOrder);
        assert_eq!(contract.retries(), RetryScope::AcrossReconnects);
        assert_eq!(contract.faults().loss(), Loss::None);
        assert!(!contract.faults().reorders());
        assert!(contract.ordered_channel().is_ok());
        assert!(contract.exactly_once_channel().is_ok());
        assert_eq!(
            contract.delivery().suppression().map(|s| s.scope()),
            Some(DuplicateScope::RequestIdentity)
        );
        assert_eq!(contract.capacity().max(), MAX_ADMISSION_ENTRIES);
        assert_eq!(contract.producer().item(), "submit_request");
        assert_eq!(contract.consumer().item(), "submit_tracked");
    }

    #[test]
    fn the_recorder_rpc_contract_is_pinned_and_can_carry_a_batch() {
        let contract = consensus_recorder_rpc_contract().expect("recorder RPC contract");
        assert_eq!(contract.authentication(), Authentication::MutualTls);
        assert_eq!(contract.ordering(), OrderingScope::Unordered);
        assert!(contract.ordered_channel().is_err());
        assert!(contract.capacity().max() >= MAX_DECISION_BATCH);
    }

    #[test]
    fn a_quorum_threshold_cannot_exceed_its_members() {
        assert_eq!(
            Quorum::new(2, [member(1)]),
            Err(ContractError::QuorumExceedsMembers {
                threshold: 2,
                members: 1
            })
        );
        assert_eq!(
            Quorum::new(0, [member(1)]),
            Err(ContractError::ZeroQuorumThreshold)
        );
    }
}
