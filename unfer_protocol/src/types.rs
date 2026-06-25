use serde::{Deserialize, Serialize};

use crate::codes::Diagnostic;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub hamiltonian: HamiltonianSpec,
    pub prior: PriorSpec,
    pub solver: SolverSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HamiltonianSpec {
    Builtin {
        name: String,
        params: serde_json::Value,
    },
    Latex {
        latex: String,
    },
    Terms {
        terms: Vec<TermSpec>,
    },
}

impl HamiltonianSpec {
    pub fn builtin(name: impl Into<String>, params: serde_json::Value) -> Self {
        Self::Builtin {
            name: name.into(),
            params,
        }
    }

    pub fn latex(src: impl Into<String>) -> Self {
        Self::Latex { latex: src.into() }
    }

    pub fn terms(terms: Vec<TermSpec>) -> Self {
        Self::Terms { terms }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TermSpec {
    pub coeff_re: f64,
    pub coeff_im: f64,
    pub ops: Vec<OpSpec>,
}

impl TermSpec {
    pub fn new(coeff_re: f64, coeff_im: f64, ops: Vec<OpSpec>) -> Self {
        Self {
            coeff_re,
            coeff_im,
            ops,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    Create,
    Annihilate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    InnerBoson,
    InnerFermion,
    OuterBoson,
    OuterFermion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpSpec {
    pub kind: OpKind,
    pub level: Level,
    pub mode: u32,
}

impl OpSpec {
    pub fn new(kind: OpKind, level: Level, mode: u32) -> Self {
        Self { kind, level, mode }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PriorSpec {
    Vacuum,
    Bosons {
        modes: Vec<(u32, u32)>,
    },
    Fermions {
        modes: Vec<u32>,
    },
    Superposition {
        terms: Vec<SuperpositionTerm>,
    },
}

impl PriorSpec {
    pub fn bosons(modes: Vec<(u32, u32)>) -> Self {
        Self::Bosons { modes }
    }

    pub fn fermions(modes: Vec<u32>) -> Self {
        Self::Fermions { modes }
    }

    pub fn superposition(terms: Vec<SuperpositionTerm>) -> Self {
        Self::Superposition { terms }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuperpositionTerm {
    pub re: f64,
    pub im: f64,
    pub spec: PriorSpec,
}

impl SuperpositionTerm {
    pub fn new(re: f64, im: f64, spec: PriorSpec) -> Self {
        Self { re, im, spec }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cmp {
    Eq,
    Ge,
    Le,
    Gt,
    Lt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventPredicate {
    BosonModeTotal {
        mode: u32,
        cmp: Cmp,
        value: u32,
    },
    FermionModePresent {
        mode: u32,
    },
    BosonUniverseCount {
        cmp: Cmp,
        value: u32,
    },
    FermionUniverseCount {
        cmp: Cmp,
        value: u32,
    },
    Vacuum,
    And {
        parts: Vec<EventPredicate>,
    },
    Or {
        parts: Vec<EventPredicate>,
    },
    Not {
        inner: Box<EventPredicate>,
    },
}

impl EventPredicate {
    pub fn and(parts: Vec<EventPredicate>) -> Self {
        Self::And { parts }
    }

    pub fn or(parts: Vec<EventPredicate>) -> Self {
        Self::Or { parts }
    }

    // `and`/`or`/`not` form a deliberate constructor trio mirroring the predicate
    // combinators; this is not an implementation of `std::ops::Not`.
    #[allow(clippy::should_implement_trait)]
    pub fn not(inner: EventPredicate) -> Self {
        Self::Not {
            inner: Box::new(inner),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[derive(Default)]
pub enum DeviceSpec {
    #[default]
    Cpu,
    Cuda {
        device_id: u32,
    },
}


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverSpec {
    pub krylov_dim: usize,
    pub prune_eps: f64,
    pub max_components: Option<usize>,
    pub restarts: usize,
    pub device: DeviceSpec,
}

impl Default for SolverSpec {
    fn default() -> Self {
        Self {
            krylov_dim: 8,
            prune_eps: 1e-12,
            max_components: None,
            restarts: 1,
            device: DeviceSpec::Cpu,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRequest {
    pub id: String,
    pub op: String,
    pub params: serde_json::Value,
}

impl AgentRequest {
    pub fn new(id: impl Into<String>, op: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            op: op.into(),
            params,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentResponse {
    pub id: String,
    pub ok: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<Diagnostic>,
}

impl AgentResponse {
    pub fn ok(id: impl Into<String>, result: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: impl Into<String>, diagnostic: Diagnostic) -> Self {
        Self {
            id: id.into(),
            ok: false,
            result: None,
            error: Some(diagnostic),
        }
    }
}
