use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Code(pub u32);

impl Code {
    pub const BAD_JSON: Code = Code(1001);
    pub const UNKNOWN_BUILTIN_MODEL: Code = Code(1002);
    pub const BAD_EVENT_PREDICATE: Code = Code(1003);
    pub const BAD_HANDLE: Code = Code(1004);
    pub const BUFFER_TOO_SMALL: Code = Code(1005);

    pub const GRAM_DEGENERATE: Code = Code(2001);
    pub const STATE_EXPLOSION: Code = Code(2002);
    pub const ZERO_PROBABILITY_CONDITION: Code = Code(2003);
    pub const BRST_NOT_CONVERGED: Code = Code(2004);
    pub const CAS_TERM_EXPLOSION: Code = Code(2005);

    pub const CUDA_UNAVAILABLE: Code = Code(3001);
    pub const OUT_OF_MEMORY_BUDGET: Code = Code(3002);

    pub const CALL_DENIED: Code = Code(4001);
    /// The submitted side-effecting action requires operator/gatekeeper approval;
    /// a provisional (simulated) result was returned so the caller can keep working.
    pub const ACTION_REQUIRES_APPROVAL: Code = Code(4002);
    /// The action was rejected by the operator/gatekeeper.
    pub const ACTION_REJECTED: Code = Code(4003);
    /// No action exists for the referenced handle/id.
    pub const ACTION_NOT_FOUND: Code = Code(4004);
    /// The action was already resolved (approved/rejected/reverted) and cannot be
    /// resolved again.
    pub const ACTION_ALREADY_RESOLVED: Code = Code(4005);

    /// The `.cell` blueprint archive could not be parsed (bad magic, unsupported version,
    /// corrupt gzip, malformed metadata).
    pub const BLUEPRINT_INVALID: Code = Code(4100);
    /// The `.cell` blueprint archive parsed but carries no session snapshot to restore.
    pub const BLUEPRINT_NO_SESSION: Code = Code(4101);
    /// Referencing a blueprint id never imported at the registry (F19).
    pub const BLUEPRINT_NOT_FOUND: Code = Code(4102);

    /// The audit/caller JSON was malformed (bad `CallerTag` / `AuditEntry`).
    pub const AUDIT_INVALID: Code = Code(4200);
    /// No agent exists for the referenced handle/id.
    pub const AGENT_NOT_FOUND: Code = Code(4201);
    /// `uk_agent_spawn` refused: the requested grant set is not a subset of the
    /// caller's (capability escalation is impossible — the chokepoint).
    pub const AGENT_GRANT_ESCALATION: Code = Code(4202);
    /// The operation is invalid for the agent's current state (e.g. killing a
    /// stopped agent).
    pub const AGENT_STATE_INVALID: Code = Code(4203);

    /// The caller attempted an operation on a resource it has not been introduced to:
    /// the resource is not in the caller's `[grants] resources` set (and was not minted at
    /// the kernel chokepoint for this session). Nothing is ambient (F17).
    pub const RESOURCE_UNINTRODUCED: Code = Code(4401);
    /// The resource was already introduced/minted at the kernel chokepoint.
    pub const RESOURCE_ALREADY_INTRODUCED: Code = Code(4402);
    /// Referencing a resource id that has never been minted (unknown at the chokepoint).
    pub const RESOURCE_NOT_FOUND: Code = Code(4403);

    /// F20 trust annotations: an operation reserved for the operator console was
    /// attempted by a module/agent. A module can never self-declare **vetted** status
    /// (`uk_registry_vetted` is hook-only) nor mint any other console-only capability.
    pub const CONSOLE_ONLY: Code = Code(4501);

    /// F24 metering: the caller's windowed call rate exceeded its limit at the
    /// loopback chokepoint (denied there, never a post-hoc report).
    pub const RATE_LIMITED: Code = Code(4601);
    /// F24 metering: the caller exhausted its per-principal budget for the window
    /// (denied at the loopback chokepoint with an audit entry).
    pub const BUDGET_EXCEEDED: Code = Code(4602);

    /// F25 forward policy: the caller has observed `<*sensitive*>` data, so the
    /// chokepoint latches it and refuses forward-mutating ops (egress, hand-off,
    /// blueprints, writes) until an operator clears the latch.
    pub const SENSITIVE_LATCHED: Code = Code(4701);

    /// S29: the Lean4 export file could not be type-checked — a theorem or
    /// definition's proof term did not check (type mismatch, missing
    /// declaration, or a kernel panic inside nanoda_lib).
    pub const PROOF_VERIFY_FAILED: Code = Code(4801);
    /// S29: the Lean4 export file was malformed — unparseable NDJSON, missing
    /// declarations, oversize payload, or a bad `LeanVerifySpec`.
    pub const PROOF_EXPORT_INVALID: Code = Code(4802);

    /// S30: the Cadabra2 subprocess is not available (binary not on `PATH`,
    /// no `CADABRA_CLI` override) or failed to launch.
    pub const SYMBOLIC_ENGINE_UNAVAILABLE: Code = Code(4901);
    /// S30: the symbolic expression was malformed, the requested operation is
    /// not supported, or the engine rejected the input (no canonical form
    /// produced).
    pub const SYMBOLIC_EXPRESSION_INVALID: Code = Code(4902);

    pub const CONSENSUS_NOT_READY: Code = Code(6001);
    pub const DUPLICATE_TRANSACTION: Code = Code(6002);
    pub const INVALID_SIGNATURE: Code = Code(6003);
    pub const UNKNOWN_DID: Code = Code(6004);
    pub const RELAY_NOT_CONNECTED: Code = Code(6005);

    /// Certificate/UTXO ledger (ReFi exchange, 7xxx). The state-transition
    /// engine on each QuePaxa node rejects invalid certificate ops before they
    /// enter the consensus log — the same "rapid validation rule" the exchange
    /// plan runs for RISC-Zero receipts.
    pub const CERT_MINT_NOT_AUTHORIZED: Code = Code(7001);
    /// Conservation violation: `Sum(inputs) != Sum(outputs)` on a transfer.
    pub const CERT_AMOUNT_MISMATCH: Code = Code(7002);
    /// An input coin_id does not exist (or already spent) in the ledger.
    pub const CERT_NONEXISTENT_INPUT: Code = Code(7003);
    /// A nullifier was already consumed — attempted double spend.
    pub const CERT_DOUBLE_SPEND: Code = Code(7004);
    /// The transaction signer is not the owner of every input certificate.
    pub const CERT_OWNER_MISMATCH: Code = Code(7005);
    /// The certificate op's seq is stale or a duplicate for its kind.
    pub const CERT_LEDGER_SEQ: Code = Code(7006);
    /// The mint's `source` provenance does not reference a valid UNFCCC
    /// oracle record (`unfccc:vc:<orderId>`) — the mint request is rejected.
    pub const CERT_ORACLE_REJECTED: Code = Code(7007);

    // ------------------------------------------------------------------
    // GNU Taler exchange adapter (ReFi exchange, 71xx). The exchange owns the
    // fiat-side bookkeeping (customer reserves, merchant balances, wire
    // transfers) that is invisible to the consensus log; these codes report
    // that private-sided state machine (see `unfer_taler`).
    pub const TALER_UNKNOWN_RESERVE: Code = Code(7101);
    /// Reserve / merchant-balance shortfall: withdraw or peg-out is refused.
    pub const TALER_INSUFFICIENT_BALANCE: Code = Code(7102);
    /// A peg-in references a wire transfer that has not been confirmed by the
    /// wire gateway (Taler's two-phase reserve funding).
    pub const TALER_UNCONFIRMED_WIRE: Code = Code(7103);
    /// No denomination in the book matches the requested value (or it has
    /// expired).
    pub const TALER_DENOM_UNSUPPORTED: Code = Code(7104);
    /// An e-coin was already deposited to a merchant — double deposit at the
    /// exchange's private ledger.
    pub const TALER_COIN_ALREADY_DEPOSITED: Code = Code(7105);
    /// E-coin denomination refresh requested for a coin that is still fresh
    /// (refresh is only legal once a denomination has expired).
    pub const TALER_REFRESH_NOT_ELIGIBLE: Code = Code(7106);
    /// A deposit references an e-coin this exchange never minted (it is not
    /// backed by a customer reserve, so it must not fund fiat redemption).
    pub const TALER_UNKNOWN_E_COIN: Code = Code(7107);

    // ------------------------------------------------------------------
    // Secondary-market escrow (ReFi exchange, Phase 4, 72xx). The marketplace
    // operator rows a certificate into a deterministic intermediate DID between
    // buyer and seller; these codes report the escrow state machine (see
    // `unfer_consensus::escrow`).
    /// The referenced coin_id was never placed in this marketplace's escrow.
    pub const ESCROW_UNKNOWN: Code = Code(7201);
    /// Release/refund attempted on an escrow that is not in the Holding state.
    pub const ESCROW_NOT_HOLDING: Code = Code(7202);
    /// The escrow was already settled (released or refunded); it cannot settle
    /// again — a single outcome per escrow.
    pub const ESCROW_ALREADY_SETTLED: Code = Code(7203);

    pub const INTERNAL: Code = Code(5000);

    pub fn raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for Code {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UK-{}", self.0)
    }
}

pub fn all() -> &'static [(u32, &'static str, &'static str)] {
    &[
        (
            1001,
            "BadJson",
            "Input JSON could not be parsed or did not match the expected schema.",
        ),
        (
            1002,
            "UnknownBuiltinModel",
            "The requested builtin model name is not recognized by the kernel.",
        ),
        (
            1003,
            "BadEventPredicate",
            "The event predicate is malformed or references an unknown mode.",
        ),
        (
            1004,
            "BadHandle",
            "The referenced model handle is invalid or has been freed.",
        ),
        (
            1005,
            "BufferTooSmall",
            "The caller-provided buffer was too small; the return value holds the required size.",
        ),
        (
            2001,
            "GramDegenerate",
            "The Krylov Gram matrix is rank-deficient; reduce the Krylov dimension or adjust shifts.",
        ),
        (
            2002,
            "StateExplosion",
            "The state vector exceeded the configured component limit during expansion.",
        ),
        (
            2003,
            "ZeroProbabilityCondition",
            "Conditioning on an event with zero prior probability would divide by zero.",
        ),
        (
            2004,
            "BrstNotConverged",
            "The BRST physical-state projection failed to converge within the iteration budget.",
        ),
        (
            2005,
            "CasTermExplosion",
            "Symbolic expansion exceeded the term budget without producing a Hamiltonian.",
        ),
        (
            3001,
            "CudaUnavailable",
            "A CUDA device was requested but is not available at runtime.",
        ),
        (
            3002,
            "OutOfMemoryBudget",
            "The kernel exceeded its configured memory budget.",
        ),
        (
            4001,
            "CallDenied",
            "The authorization engine denied the caller permission to invoke this kernel symbol.",
        ),
        (
            4002,
            "ActionRequiresApproval",
            "The side-effecting action requires operator/gatekeeper approval; a provisional (simulated) result was returned.",
        ),
        (
            4003,
            "ActionRejected",
            "The action was rejected by the operator/gatekeeper.",
        ),
        (
            4004,
            "ActionNotFound",
            "No action exists for the referenced handle/id.",
        ),
        (
            4005,
            "ActionAlreadyResolved",
            "The action was already resolved (approved/rejected/reverted) and cannot be resolved again.",
        ),
        (
            4100,
            "BlueprintInvalid",
            "The .cell blueprint archive could not be parsed (bad magic, unsupported version, corrupt gzip, malformed metadata).",
        ),
        (
            4101,
            "BlueprintNoSession",
            "The .cell blueprint archive parsed but carries no session snapshot to restore.",
        ),
        (
            4102,
            "BlueprintNotFound",
            "No blueprint with that id exists at the blueprint registry.",
        ),
        (
            4200,
            "AuditInvalid",
            "The audit/caller JSON was malformed (bad CallerTag / AuditEntry).",
        ),
        (
            4201,
            "AgentNotFound",
            "No agent exists for the referenced handle/id.",
        ),
        (
            4202,
            "AgentGrantEscalation",
            "uk_agent_spawn refused: the requested grant set is not a subset of the caller's (capability escalation is impossible — the chokepoint).",
        ),
        (
            4203,
            "AgentStateInvalid",
            "The operation is invalid for the agent's current state (e.g. killing a stopped agent).",
        ),
        (
            4401,
            "ResourceUnintroduced",
            "The caller has not been introduced to the resource: it is absent from its `[grants] resources` set (nothing is ambient).",
        ),
        (
            4402,
            "ResourceAlreadyIntroduced",
            "The resource is already minted at the kernel chokepoint.",
        ),
        (
            4403,
            "ResourceNotFound",
            "No such resource id exists at the kernel chokepoint.",
        ),
        (
            4501,
            "ConsoleOnly",
            "This operation is reserved for the operator console; a module or agent cannot perform it.",
        ),
        (
            4601,
            "RateLimited",
            "The caller exceeded its windowed call-rate limit at the loopback chokepoint.",
        ),
        (
            4602,
            "BudgetExceeded",
            "The caller exhausted its per-principal budget for the current window.",
        ),
        (
            4701,
            "SensitiveLatched",
            "The caller observed sensitive data and is latched from forward-mutating operations until an operator clears it.",
        ),
        (
            4801,
            "ProofVerifyFailed",
            "The Lean4 export file did not type-check: a theorem or definition's proof term was rejected by the external kernel.",
        ),
        (
            4802,
            "ProofExportInvalid",
            "The Lean4 export file was malformed or the LeanVerifySpec was invalid (unparseable NDJSON, missing declaration, oversize payload).",
        ),
        (
            4901,
            "SymbolicEngineUnavailable",
            "The Cadabra2 subprocess is not available: the binary is not on PATH and no CADABRA_CLI override is set, or it failed to launch.",
        ),
        (
            4902,
            "SymbolicExpressionInvalid",
            "The symbolic expression was malformed, the requested operation is unsupported, or Cadabra2 rejected the input (no canonical form produced).",
        ),
        (
            6001,
            "ConsensusNotReady",
            "The consensus node has not yet synced to the latest committed sequence.",
        ),
        (
            6002,
            "DuplicateTransaction",
            "The transaction is already in the consensus log.",
        ),
        (
            6003,
            "InvalidSignature",
            "Ed25519 signature verification failed for the transaction.",
        ),
        (
            6004,
            "UnknownDid",
            "The DID is not in the identity registry.",
        ),
        (
            6005,
            "RelayNotConnected",
            "No upstream relay is available for firehose subscription.",
        ),
        (
            7001,
            "CertMintNotAuthorized",
            "The certificate mint was signed by a DID that is not the configured mint authority.",
        ),
        (
            7002,
            "CertAmountMismatch",
            "Conservation violation: the sum of transfer inputs does not equal the sum of outputs.",
        ),
        (
            7003,
            "CertNonexistentInput",
            "A certificate input coin_id does not exist (or was already spent) in the ledger.",
        ),
        (
            7004,
            "CertDoubleSpend",
            "A certificate nullifier was already consumed: attempted double spend.",
        ),
        (
            7005,
            "CertOwnerMismatch",
            "The transaction signer is not the owner of every input certificate.",
        ),
        (
            7006,
            "CertLedgerSeq",
            "The certificate op's sequence is stale or duplicated for its kind.",
        ),
        (
            7007,
            "CertOracleRejected",
            "The mint's source does not reference a valid UNFCCC oracle record (`unfccc:vc:<orderId>`).",
        ),
        (
            7101,
            "TalerUnknownReserve",
            "The reserve id is not known to the GNU Taler exchange.",
        ),
        (
            7102,
            "TalerInsufficientBalance",
            "Reserve or merchant balance is too low for the requested withdraw / peg-out.",
        ),
        (
            7103,
            "TalerUnconfirmedWire",
            "A peg-in references a wire transfer that the wire gateway has not confirmed.",
        ),
        (
            7104,
            "TalerDenomUnsupported",
            "No (unexpired) denomination matches the requested e-coin value.",
        ),
        (
            7105,
            "TalerCoinAlreadyDeposited",
            "The e-coin was already deposited — double deposit refused.",
        ),
        (
            7106,
            "TalerRefreshNotEligible",
            "Refresh is only legal once the issued denomination has expired.",
        ),
        (
            7107,
            "TalerUnknownECoin",
            "The deposit references an e-coin this exchange never minted.",
        ),
        (
            7201,
            "EscrowUnknown",
            "The coin_id was never placed in this marketplace's escrow.",
        ),
        (
            7202,
            "EscrowNotHolding",
            "Release/refund requires the escrow to be in the Holding state.",
        ),
        (
            7203,
            "EscrowAlreadySettled",
            "The escrow was already released or refunded and cannot settle twice.",
        ),
        (
            5000,
            "Internal",
            "An internal invariant was violated; this is a bug, not a user error.",
        ),
    ]
}

pub fn name_of(code: u32) -> Option<&'static str> {
    all()
        .iter()
        .find(|(c, _, _)| *c == code)
        .map(|(_, n, _)| *n)
}

pub fn description_of(code: u32) -> Option<&'static str> {
    all()
        .iter()
        .find(|(c, _, _)| *c == code)
        .map(|(_, _, d)| *d)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Error,
    Fatal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HintKind {
    ReplaceValue,
    SetParam,
    ReduceScope,
    IncreaseLimit,
    UseAlternativeOp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairHint {
    pub kind: HintKind,
    pub target: String,
    pub suggestion: String,
}

impl RepairHint {
    pub fn new(kind: HintKind, target: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            kind,
            target: target.into(),
            suggestion: suggestion.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: Code,
    pub name: String,
    pub message: String,
    pub severity: Severity,
    pub hints: Vec<RepairHint>,
    pub data: serde_json::Value,
}

impl Diagnostic {
    pub fn new(code: Code, message: impl Into<String>, severity: Severity) -> Self {
        let name = name_of(code.0).unwrap_or("Unknown").to_string();
        Self {
            code,
            name,
            message: message.into(),
            severity,
            hints: Vec::new(),
            data: serde_json::Value::Null,
        }
    }

    pub fn with_hint(mut self, hint: RepairHint) -> Self {
        self.hints.push(hint);
        self
    }

    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = data;
        self
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.code, self.name, self.message)
    }
}
