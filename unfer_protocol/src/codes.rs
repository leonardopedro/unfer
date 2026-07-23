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

    pub const CONSENSUS_NOT_READY: Code = Code(6001);
    pub const DUPLICATE_TRANSACTION: Code = Code(6002);
    pub const INVALID_SIGNATURE: Code = Code(6003);
    pub const UNKNOWN_DID: Code = Code(6004);
    pub const RELAY_NOT_CONNECTED: Code = Code(6005);

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
