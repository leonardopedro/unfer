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
    /// Non-Neural QFM with Tomographic Subspace Recovery (Workstream F).
    /// The compilation spec carries the training data and the sketch/Krylov
    /// dimensions. The `prob_kernel` compiles a `QfmPipeline` from this spec
    /// and stores it in the session; `evolve` dispatches to the pipeline's
    /// 4-phase `generate` instead of the SIRK solver.
    QfmTomography {
        spec: Box<QfmTomographySpec>,
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

    pub fn qfm_tomography(spec: QfmTomographySpec) -> Self {
        Self::QfmTomography {
            spec: Box::new(spec),
        }
    }
}

/// Compilation spec for the QFM tomographic pipeline (Workstream F).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QfmTomographySpec {
    /// Training data: a list of d-dimensional points.
    pub training_data: Vec<Vec<f64>>,
    /// Level 1 sketch dimension (k, where k << d).
    pub k: usize,
    /// Level 2 sketched Hilbert space dimension (K_2 > k).
    pub k2: usize,
    /// Krylov subspace dimension (m, the reduced rank).
    pub krylov_dim: usize,
    /// PRNG seed for the Level 1 sketch.
    pub seed: u64,
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
    Bosons { modes: Vec<(u32, u32)> },
    Fermions { modes: Vec<u32> },
    Superposition { terms: Vec<SuperpositionTerm> },
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
    BosonModeTotal { mode: u32, cmp: Cmp, value: u32 },
    FermionModePresent { mode: u32 },
    BosonUniverseCount { cmp: Cmp, value: u32 },
    FermionUniverseCount { cmp: Cmp, value: u32 },
    Vacuum,
    And { parts: Vec<EventPredicate> },
    Or { parts: Vec<EventPredicate> },
    Not { inner: Box<EventPredicate> },
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
    /// When true, the SIRK solver truncates to the top-k components instead of
    /// erroring with `StateExplosion` when `max_components` is exceeded. This
    /// enables quartic-heavy models (e.g. `yang_mills_lattice` at l≥4) to run
    /// under a fixed memory budget, at the cost of approximation error.
    #[serde(default)]
    pub adaptive: bool,
}

impl Default for SolverSpec {
    fn default() -> Self {
        Self {
            krylov_dim: 8,
            prune_eps: 1e-12,
            max_components: None,
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
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
    /// Wall-clock time for the op in milliseconds (absent on very fast ops).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KernelEvent {
    PriorSet,
    HamiltonianSet,
    Evolved { t: f64, norm: f64, solve_ms: u64 },
    Conditioned { prior_probability: f64 },
    Observed { value: f64 },
    Error { diagnostic: Diagnostic },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventQuery {
    pub types: Option<Vec<String>>,
}

impl AgentResponse {
    pub fn ok(id: impl Into<String>, result: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            ok: true,
            result: Some(result),
            error: None,
            timing_ms: None,
        }
    }

    pub fn err(id: impl Into<String>, diagnostic: Diagnostic) -> Self {
        Self {
            id: id.into(),
            ok: false,
            result: None,
            error: Some(diagnostic),
            timing_ms: None,
        }
    }

    pub fn with_timing(mut self, ms: u64) -> Self {
        self.timing_ms = Some(ms);
        self
    }
}
