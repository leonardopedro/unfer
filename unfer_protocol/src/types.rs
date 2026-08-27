use serde::{Deserialize, Serialize};

use crate::codes::{Diagnostic, HintKind, RepairHint};

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
    /// Typst math input (P8.7). The compiler maps the operator-product
    /// dialect (`a^dagger a`, `\omega * c^dagger c + h.c.`) directly to
    /// the project's internal CAS string and bypasses mathhook.
    /// Requires the `latex` feature on `prob_kernel` (compile-time gate
    /// matches the existing `Latex` variant).
    Typst {
        typst: String,
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
    /// Polynomial autonomous ODE system → Weyl-symmetrized Hamiltonian.
    /// Each `rhs[i]` is a polynomial expression in the `vars` names
    /// (e.g. `["x","y"]`, `["x^2", "2*x*y"]`).
    /// `change_of_variables` may be `"none"`, `"reciprocal:0"`,
    /// `"logarithmic:0"`, etc.
    OdeSystem {
        vars: Vec<String>,
        rhs: Vec<String>,
        #[serde(default)]
        change_of_variables: Option<String>,
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

    pub fn typst(src: impl Into<String>) -> Self {
        Self::Typst { typst: src.into() }
    }

    pub fn terms(terms: Vec<TermSpec>) -> Self {
        Self::Terms { terms }
    }

    pub fn qfm_tomography(spec: QfmTomographySpec) -> Self {
        Self::QfmTomography {
            spec: Box::new(spec),
        }
    }

    pub fn ode_system(
        vars: Vec<String>,
        rhs: Vec<String>,
        change_of_variables: Option<String>,
    ) -> Self {
        Self::OdeSystem {
            vars,
            rhs,
            change_of_variables,
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
    /// H9: provenance source of external data reaching agent context
    /// (`file|web|tool_result|webhook|overheard`). Additive — absent on ops
    /// that carry no external data; the `auto` posture screens labelled
    /// payloads before they reach agent context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<crate::posture::ProvenanceSource>,
}

impl AgentRequest {
    pub fn new(id: impl Into<String>, op: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            op: op.into(),
            params,
            provenance: None,
        }
    }

    /// Attach a provenance label (H9) to an external-data op.
    pub fn with_provenance(mut self, source: crate::posture::ProvenanceSource) -> Self {
        self.provenance = Some(source);
        self
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
    Evolved {
        t: f64,
        norm: f64,
        solve_ms: u64,
    },
    Conditioned {
        prior_probability: f64,
    },
    Observed {
        value: f64,
    },
    /// A Lean4 export file was type-checked by the external kernel; `verified`
    /// is the boolean reduction of the proof(s). Broadcast to subscribers of
    /// the model handle.
    Verified {
        verified: bool,
        declarations_checked: usize,
    },
    /// A symbolic expression was processed by the external Cadabra2 engine;
    /// `verified` is the zero-detection verdict. Broadcast to subscribers of
    /// the model handle.
    Simplified {
        op: String,
        verified: bool,
    },
    /// A CNL sentence was compiled by the Logos engine to a unique normal form
    /// (UNF). `result` is the readback, `unf_hash` its content-addressable
    /// digest, and `verified` the confluence self-check (re-reducing the same
    /// input yields an identical UNF). Broadcast to subscribers of the model
    /// handle.
    LogosCompiled {
        result: String,
        unf_hash: String,
        verified: bool,
    },
    /// A WhyML program was emitted by the kernel (S36); `verified` is
    /// `Some(..)` when the external Why3 prover discharged every proof
    /// obligation (`WhymlOp::Prove`), `None` for pure emission. Broadcast to
    /// subscribers of the model handle.
    WhymlCompiled {
        module_name: String,
        whyml_len: usize,
        verified: Option<bool>,
    },
    /// An AustralVM-language source fragment was translated to a unique
    /// normal form through DeltaNets (`uk_austral_unf`); `value` is the
    /// numerical result when the term is closed (no unknowns), `None` when
    /// it stays symbolic. Broadcast to subscribers of the model handle.
    AustralUnf {
        sym_expr: String,
        value: Option<String>,
        unf_hash: String,
        verified: bool,
    },
    Error {
        diagnostic: Diagnostic,
    },
    /// A side-effecting action was submitted for approval (S4). Broadcast to all
    /// subscriptions (kernel-global approval lane), not scoped to one model handle.
    ActionPending {
        action: ActionRecord,
    },
    /// An action was resolved by the operator/gatekeeper (approved / rejected /
    /// reverted). Broadcast like [`KernelEvent::ActionPending`].
    ActionResolved {
        action: ActionRecord,
    },
}

// ── Deferred approval + local simulation (S4) ────────────────────────────
//
// Cloudflare "Gatekeeper" adaptation (PLAN_cloudflare_os_adaptation.md §F2):
// side-effecting ops are not executed inline. `uk_action_submit` queues an
// `ActionRecord` in the `{staged, pending, approved, rejected, reverted}`
// lifecycle and returns a *provisional* (simulated) result immediately so the
// agent keeps working; an operator/gatekeeper later resolves the record via
// `uk_action_apply` / `uk_action_reject` / `uk_action_revert`. Reads merge the
// provisional item back: a pending action reports its provisional result, an
// approved one reports the real applied result.

/// Lifecycle of a deferred-approval action (mirrors gatekeeper `state ∈
/// {staged, pending, approved, rejected}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionState {
    /// Record created but not yet submitted for approval.
    Staged,
    /// Submitted; awaiting operator/gatekeeper resolution. The submitter sees
    /// only the provisional (simulated) result until this resolves.
    Pending,
    /// Approved by the operator/gatekeeper; the real (applied) result is set.
    Approved,
    /// Rejected by the operator/gatekeeper; the effect will not be executed.
    Rejected,
    /// An approved action was later reverted (rollback).
    Reverted,
}

/// A side-effecting action awaiting (or having received) operator approval (S4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionRecord {
    /// Stable identifier (`action-<seq>`).
    pub id: String,
    /// The module/agent that submitted the action (audit tag, F6).
    pub principal: String,
    /// The full `GatekeeperCaller`-style tag (`{from, principal, chat_id}`) that
    /// submitted the action (S6, F6). Injected at the loopback chokepoint from the
    /// module's caller context; a worker cannot forge another identity's tag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller: Option<CallerTag>,
    /// The effect name, e.g. `"send_notification"`. The submitting module must
    /// hold the matching grant in the `effects` namespace.
    pub effect: String,
    /// Effect-specific parameters (the payload the operator reviews).
    pub params: serde_json::Value,
    /// Current lifecycle state.
    pub state: ActionState,
    /// Monotonic creation sequence (wall-clock free, stable ordering).
    pub created_at: u64,
    /// The simulated result returned immediately on submission (local simulation).
    /// Read back while the action is still pending.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provisional: Option<serde_json::Value>,
    /// The real result once `uk_action_apply` executed the effect.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied: Option<serde_json::Value>,
}

impl ActionRecord {
    pub fn new(
        id: impl Into<String>,
        principal: impl Into<String>,
        effect: impl Into<String>,
        params: serde_json::Value,
        created_at: u64,
        provisional: Option<serde_json::Value>,
    ) -> Self {
        Self {
            id: id.into(),
            principal: principal.into(),
            caller: None,
            effect: effect.into(),
            params,
            state: ActionState::Pending,
            created_at,
            provisional,
            applied: None,
        }
    }

    /// The merged result a reader sees: the applied result once approved, else the
    /// provisional (simulated) result. This is the "reads merge the provisional
    /// items back" behavior (mirror `github.ts:839`).
    pub fn merged_result(&self) -> Option<serde_json::Value> {
        match self.state {
            ActionState::Approved => self.applied.clone(),
            _ => self.provisional.clone(),
        }
    }
}

// ── Agent accountability + audit (S6) ──────────────────────────────────
//
// Cloudflare "GatekeeperCaller" adaptation (PLAN_cloudflare_os_adaptation.md
// §F6): every `uk_*` call and every `ActionRecord` carries a caller tag so the
// initiating human stays accountable. The tag is minted once at the loopback
// chokepoint (a module cannot claim another identity), the kernel appends the
// audit trail, and the `AgentSpawner` capability spawns sub-agents bounded to a
// fixed grant set.

/// Where a kernel caller came from — the `{from: agent|gadget|hook}` audit axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallerKind {
    /// A sub-agent spawned via `uk_agent_spawn` (F6 `AgentSpawner`).
    Agent,
    /// A module instance (the workerd-sidecar gadget path).
    Gadget,
    /// The operator/harness driving the kernel directly (trusted).
    Hook,
}

/// The `GatekeeperCaller`-style audit tag on a `uk_*` call or `ActionRecord`
/// (S6, F6). `{from, principal, chat_id}` — the human remains accountable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallerTag {
    pub from: CallerKind,
    /// Module name / agent id / hook label.
    pub principal: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<String>,
}

impl CallerTag {
    pub fn new(from: CallerKind, principal: impl Into<String>, chat_id: Option<String>) -> Self {
        Self {
            from,
            principal: principal.into(),
            chat_id,
        }
    }

    pub fn gadget(principal: impl Into<String>) -> Self {
        Self::new(CallerKind::Gadget, principal, None)
    }

    pub fn agent(principal: impl Into<String>) -> Self {
        Self::new(CallerKind::Agent, principal, None)
    }

    pub fn hook(principal: impl Into<String>) -> Self {
        Self::new(CallerKind::Hook, principal, None)
    }
}

impl Default for CallerTag {
    fn default() -> Self {
        Self::hook("kernel")
    }
}

/// Trust annotation for a granted effect (S21, F20). Cloudflare's `readOnlyHint`
/// makes a tool run as *observation* (no approval needed); everything else is a
/// mutation queued for the gatekeeper console. A mutation can still auto-apply
/// only through the console's *vetted* marker — never by module self-declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffectKind {
    /// Read-only: never queues, applied immediately.
    Observe,
    /// Side-effecting (default): queues for approval unless vetted.
    #[default]
    Mutate,
}

/// A granted effect with its trust annotation (S21, F20).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectGrant {
    pub name: String,
    /// The annotation carried on the console invitation or in `[grants] effects`.
    /// Absent = [`EffectKind::Mutate`] (conservative).
    #[serde(default)]
    pub effect_kind: EffectKind,
}

/// A bounded set of capability grants. `kernel` names `uk_*` symbols; `effects`
/// names side-effecting effect names (the S4 `[grants] effects` namespace);
/// `observers` names *other principals* whose records/audit entries this caller
/// may read (the F8 `[grants] observers` namespace). A caller always observes
/// its own principal; the trusted harness (`grants: None`) observes everything.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GrantSet {
    #[serde(default)]
    pub kernel: Vec<String>,
    #[serde(default)]
    pub effects: Vec<String>,
    #[serde(default)]
    pub observers: Vec<String>,
    /// S18 (F17): the `[grants] resources` namespace — resource ids introduced to *this*
    /// session (`"github.repo#denoission"`). Nothing is ambient: a caller can only exercise a
    /// resource that is granted here (and was minted at the kernel chokepoint). Participates
    /// in `is_subset_of` so no path can mint an introduction it does not hold.
    #[serde(default)]
    pub resources: Vec<String>,
    /// S21 (F20): trust annotations for granted effects. Names here must already
    /// appear in `effects`; an observe-annotated effect never queues, a plain one is
    /// a mutation (queued unless vetted). Older grants deserialize without this.
    #[serde(default)]
    pub effect_kinds: Vec<EffectGrant>,
}

impl GrantSet {
    pub fn kernel(symbols: &[&str]) -> Self {
        Self {
            kernel: symbols.iter().map(|s| s.to_string()).collect(),
            effects: Vec::new(),
            observers: Vec::new(),
            resources: Vec::new(),
            effect_kinds: Vec::new(),
        }
    }

    /// The trust annotation for a granted effect, if one exists. An effect name
    /// granted but never annotated defaults to [`EffectKind::Mutate`] (conservative:
    /// an annotation cannot turn a side-effect into an observation).
    pub fn effect_kind_of(&self, effect: &str) -> Option<EffectKind> {
        self.effect_kinds
            .iter()
            .find(|g| g.name == effect)
            .map(|g| g.effect_kind)
    }

    /// True when `self` is a subset of `other` (capability non-escalation).
    /// Observation rights count too: a caller cannot mint observer visibility it
    /// does not already hold (F8 no-read-up). Introduced resources likewise: minting an
    /// introduction the caller does not already hold is escalation (S18/F17). The
    /// trust annotations close one more escalation path: a mutation cannot be
    /// relabeled `observe` (which bypasses approval) unless the parent already
    /// holds that annotation for the same effect (F20).
    pub fn is_subset_of(&self, other: &GrantSet) -> bool {
        let in_other = |s: &String| other.kernel.contains(s);
        self.kernel.iter().all(in_other)
            && self.effects.iter().all(|e| other.effects.contains(e))
            && self.observers.iter().all(|o| other.observers.contains(o))
            && self.resources.iter().all(|r| other.resources.contains(r))
            && self
                .effect_kinds
                .iter()
                .all(|g| other.effect_kind_of(&g.name) == Some(g.effect_kind))
    }
}

#[cfg(test)]
mod grantset_proptests {
    // Property invariants for the capability lattice (S17, F16):
    //   * reflexivity — every set is its own subset;
    //   * transitivity — A ⊆ B ∧ B ⊆ C ⇒ A ⊆ C (no escalation path);
    //   * antisymmetry — A ⊆ B ∧ B ⊆ A ⇒ equal as sets (no phantom grants incl. observers).
    use super::GrantSet;
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    fn grant_any() -> impl Strategy<Value = GrantSet> {
        let ks = proptest::collection::hash_set(0u8..4, 0..6);
        let es = proptest::collection::hash_set(0u8..3, 0..6);
        let os = proptest::collection::hash_set(0u8..3, 0..6);
        let rs = proptest::collection::hash_set(0u8..4, 0..5);
        (ks, es, os, rs).prop_map(|(k, e, o, r)| GrantSet {
            kernel: k.into_iter().map(|i| format!("kernel{i:02}")).collect(),
            effects: e.into_iter().map(|i| format!("effect{i:02}")).collect(),
            observers: o.into_iter().map(|i| format!("peer{i:02}")).collect(),
            resources: r.into_iter().map(|i| format!("res{i:02}")).collect(),
            effect_kinds: Vec::new(),
        })
    }

    fn sorted(
        g: &GrantSet,
    ) -> (
        BTreeSet<&str>,
        BTreeSet<&str>,
        BTreeSet<&str>,
        BTreeSet<&str>,
    ) {
        (
            g.kernel.iter().map(String::as_str).collect(),
            g.effects.iter().map(String::as_str).collect(),
            g.observers.iter().map(String::as_str).collect(),
            g.resources.iter().map(String::as_str).collect(),
        )
    }

    proptest! {
        #[test]
        fn subset_is_reflexive(g in grant_any()) {
            prop_assert!(g.is_subset_of(&g));
        }

        #[test]
        fn subset_is_transitive(a in grant_any(), b in grant_any(), c in grant_any()) {
            if a.is_subset_of(&b) && b.is_subset_of(&c) {
                prop_assert!(a.is_subset_of(&c));
            }
        }

        #[test]
        fn subset_antisymmetry_means_equal_sets(a in grant_any(), b in grant_any()) {
            if a.is_subset_of(&b) && b.is_subset_of(&a) {
                prop_assert_eq!(sorted(&a), sorted(&b));
            }
        }

        #[test]
        fn observer_grant_cannot_read_up(a in grant_any(), b in grant_any()) {
            // An observer entry in `a` that `b` lacks must disqualify the subset.
            if a.observers.iter().any(|o| !b.observers.contains(o)) {
                prop_assert!(!a.is_subset_of(&b));
            }
        }

        #[test]
        fn resource_grant_cannot_be_minted(a in grant_any(), b in grant_any()) {
            // An introduced resource in `a` that `b` lacks must disqualify the subset:
            // a caller cannot mint an introduction it does not already hold (F17).
            if a.resources.iter().any(|r| !b.resources.contains(r)) {
                prop_assert!(!a.is_subset_of(&b));
            }
        }
    }
}

#[cfg(test)]
mod grantset_tests {
    use super::{EffectGrant, EffectKind, GrantSet};

    #[test]
    fn observers_are_serde_default() {
        // A spec with no observers/resouces deserializes to empty lists (back-compat).
        let g: GrantSet = serde_json::from_str(r#"{"kernel":["uk_version"]}"#).unwrap();
        assert_eq!(g.observers, Vec::<String>::new());
        assert_eq!(g.resources, Vec::<String>::new());
        assert_eq!(g.effect_kinds, Vec::<EffectGrant>::new());
        assert_eq!(
            serde_json::to_string(&GrantSet::kernel(&["uk_version"])).unwrap(),
            r#"{"kernel":["uk_version"],"effects":[],"observers":[],"resources":[],"effect_kinds":[]}"#
        );
    }

    #[test]
    fn is_subset_of_includes_observers() {
        let base = GrantSet {
            kernel: vec!["uk_version".into(), "uk_evolve".into()],
            effects: vec!["notify".into()],
            observers: vec!["peer_a".into()],
            resources: vec!["github.repo#denoission".into()],
            effect_kinds: vec![EffectGrant {
                name: "notify".into(),
                effect_kind: EffectKind::Observe,
            }],
        };
        // Subset of kernel+effects+observers → allowed.
        let ok = GrantSet {
            kernel: vec!["uk_version".into()],
            effects: vec![],
            observers: vec!["peer_a".into()],
            resources: vec![],
            effect_kinds: vec![],
        };
        assert!(ok.is_subset_of(&base));
        // Observing a peer the caller does not observe → escalation.
        let escalate = GrantSet {
            kernel: vec![],
            effects: vec![],
            observers: vec!["peer_secret".into()],
            resources: vec![],
            effect_kinds: vec![],
        };
        assert!(
            !escalate.is_subset_of(&base),
            "observer escalation must be refused"
        );
        // Minting a resource introduction the caller does not hold → escalation.
        let mint_resource = GrantSet {
            kernel: vec![],
            effects: vec![],
            observers: vec![],
            resources: vec!["s3.bucket#confidential".into()],
            effect_kinds: vec![],
        };
        assert!(
            !mint_resource.is_subset_of(&base),
            "resource-introduction escalation must be refused"
        );
    }

    #[test]
    fn effect_kind_annotation_cannot_escape() {
        // F20 trust annotations: a caller may carry an observe annotation only if the
        // parent already holds it. Downgrading an un-annotated (mutate) effect to
        // `observe` would bypass approval — that is escalation.
        let base = GrantSet {
            kernel: vec![],
            effects: vec!["read_metric".into(), "delete_row".into()],
            observers: vec![],
            resources: vec![],
            effect_kinds: vec![EffectGrant {
                name: "read_metric".into(),
                effect_kind: EffectKind::Observe,
            }],
        };
        // Carrying the observe annotation the parent holds → allowed.
        let ok = GrantSet {
            kernel: vec![],
            effects: vec!["read_metric".into()],
            observers: vec![],
            resources: vec![],
            effect_kinds: vec![EffectGrant {
                name: "read_metric".into(),
                effect_kind: EffectKind::Observe,
            }],
        };
        assert!(ok.is_subset_of(&base));
        // Relabeling `delete_row` (a mutation) as observe → refused.
        let escalate = GrantSet {
            kernel: vec![],
            effects: vec!["delete_row".into()],
            observers: vec![],
            resources: vec![],
            effect_kinds: vec![EffectGrant {
                name: "delete_row".into(),
                effect_kind: EffectKind::Observe,
            }],
        };
        assert!(
            !escalate.is_subset_of(&base),
            "a mutation cannot be relabeled observe (bypasses approval)"
        );
        // An un-annotated granted effect reads back as None (caller decides Mutate).
        assert_eq!(base.effect_kind_of("delete_row"), None);
        assert_eq!(
            base.effect_kind_of("read_metric"),
            Some(EffectKind::Observe)
        );
        assert_eq!(base.effect_kind_of("never_granted"), None);
    }
}

/// One immutable audit-trail entry (S6). Created at the loopback chokepoint per
/// `uk_*` call and by the kernel for action submissions/resolutions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Monotonic sequence number (stable ordering, wall-clock free).
    pub seq: u64,
    /// The calling entity's tag — cannot be forged by the callee.
    pub caller: CallerTag,
    /// The kernel symbol invoked (e.g. `"uk_evolve"`, `"uk_action_submit"`).
    pub symbol: String,
    /// Whether the call succeeded (no diagnostic raised).
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// The JSON arguments (a JSON array for dispatched calls).
    #[serde(default)]
    pub args: serde_json::Value,
    /// Dot-separated owner component reporting the entry (S23/F22, e.g. `"kernel.audit"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
    /// Per-call observability context (AsyncLocal analog: `{trace_id, ...}` fields
    /// threaded from the host into the call that produced this entry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<serde_json::Value>,
    /// F25 forward policy: this entry represents a `<*sensitive*>` observation.
    /// When set, the caller's sticky observation set latches and the loopback
    /// refuses forward-mutating ops (egress, hand-off, blueprints, writes) with
    /// [`Code::SENSITIVE_LATCHED`] until an operator clears it. Set by the
    /// gatekeeper/`uk_*` that minted the read — never by the reader.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub sensitive: bool,
}

/// Lifecycle of a spawned sub-agent (S6 `AgentSpawner`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    Running,
    Paused,
    Stopped,
}

/// A spawned sub-agent (S6). The `grants` set is fixed at spawn time — a
/// capability-minting chokepoint (`uk_agent_spawn` refuses escalation), and the
/// host loopback enforces it (default-deny).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Stable identifier (`agent-<seq>`).
    pub id: String,
    /// Human/operator label.
    pub name: String,
    /// The fixed, bounded grant set the sub-agent may exercise.
    pub grants: GrantSet,
    /// Parent agent id, if this is a sub-agent of another agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    pub state: AgentState,
    /// Monotonic creation sequence (== spawn order).
    pub created_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<String>,
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

// ── Bayesian update (QFM.tex §8 + P6 H follow-on) ──────────────────────
//
// The Quantum Bayesian Update on the TSR-evolved prior
// (`qfm::bayes::Likelihood` + `Posterior` + `sample_hmc_single` +
// `reconstruct`) is exposed over the kernel ABI as `uk_bayesian_update`.
// The protocol types below are the JSON schema for the request and
// the result; they are translated to/from the qfm crate types in
// `prob_kernel/src/session.rs` (Bayesian update on a QFM model) and
// `unfer_ffi/src/lib.rs` (`uk_bayesian_update` FFI dispatch).

/// HMC sampler configuration. Mirrors `qfm::bayes::HmcOpts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HmcOptsSpec {
    /// Number of leapfrog steps per HMC proposal.
    #[serde(default = "default_leapfrog_steps")]
    pub leapfrog_steps: usize,
    /// Step size $\epsilon$ in the leapfrog integrator.
    #[serde(default = "default_step_size")]
    pub step_size: f64,
    /// Number of HMC proposals (burn-in + sample).
    #[serde(default = "default_n_iterations")]
    pub n_iterations: usize,
    /// Number of initial proposals to discard as burn-in.
    #[serde(default = "default_burn_in")]
    pub burn_in: usize,
    /// PRNG seed (deterministic HMC).
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_leapfrog_steps() -> usize {
    20
}
fn default_step_size() -> f64 {
    0.05
}
fn default_n_iterations() -> usize {
    200
}
fn default_burn_in() -> usize {
    100
}
fn default_seed() -> u64 {
    42
}

impl Default for HmcOptsSpec {
    fn default() -> Self {
        Self {
            leapfrog_steps: default_leapfrog_steps(),
            step_size: default_step_size(),
            n_iterations: default_n_iterations(),
            burn_in: default_burn_in(),
            seed: default_seed(),
        }
    }
}

impl HmcOptsSpec {
    /// Validate the HMC options. Returns a list of `RepairHint`s, one
    /// per invalid field, in priority order. An empty list means the
    /// spec is valid and `uk_bayesian_update` can proceed.
    ///
    /// (P7 P5, rev 18: was missing; a `leapfrog_steps = 0` or
    /// `step_size = 0.0` would silently produce a broken HMC chain.
    /// The FFI now calls this and returns `UK-1001 BAD_JSON` with the
    /// hints attached when the spec is invalid.)
    pub fn validate(&self) -> Vec<RepairHint> {
        let mut hints = Vec::new();
        if self.leapfrog_steps == 0 {
            hints.push(RepairHint::new(
                HintKind::IncreaseLimit,
                "hmc_opts.leapfrog_steps",
                "leapfrog_steps must be > 0; the HMC chain has no inner-loop steps to advance the integrator",
            ));
        }
        if self.leapfrog_steps > 10_000 {
            hints.push(RepairHint::new(
                HintKind::ReduceScope,
                "hmc_opts.leapfrog_steps",
                format!(
                    "leapfrog_steps = {} is unusually large; consider <= 1000 (per-step cost is O(N * m^2))",
                    self.leapfrog_steps
                ),
            ));
        }
        if self.step_size <= 0.0 || !self.step_size.is_finite() {
            hints.push(RepairHint::new(
                HintKind::ReplaceValue,
                "hmc_opts.step_size",
                format!(
                    "step_size = {} is invalid; must be a positive finite f64 (typical: 0.01..0.1)",
                    self.step_size
                ),
            ));
        }
        if self.n_iterations == 0 {
            hints.push(RepairHint::new(
                HintKind::IncreaseLimit,
                "hmc_opts.n_iterations",
                "n_iterations must be > 0; the HMC sampler has no proposals to draw",
            ));
        }
        if self.n_iterations < self.burn_in {
            hints.push(RepairHint::new(
                HintKind::SetParam,
                "hmc_opts",
                format!(
                    "n_iterations = {} is less than burn_in = {}; after burn-in there are no samples to keep. Set n_iterations >= burn_in",
                    self.n_iterations, self.burn_in
                ),
            ));
        }
        hints
    }
}

/// Request body for `uk_bayesian_update`. A list of $N$ raw
/// observations $\{D_1, \dots, D_N\}$ (each a d-dim vector) and the
/// HMC sampler configuration.
///
/// Only QFM tomographic models (`HamiltonianSpec::QfmTomography`) are
/// eligible for Bayesian updates — the prior is the TSR-evolved vacuum
/// state. Calling `uk_bayesian_update` on a non-QFM model returns
/// UK-5000 (INTERNAL).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BayesianUpdateRequest {
    /// The list of new observations. Each observation is a d-dim
    /// vector matching the pipeline's training-data dimension.
    pub observations: Vec<Vec<f64>>,
    /// HMC sampler configuration.
    #[serde(default)]
    pub hmc_opts: HmcOptsSpec,
}

/// Result body for `uk_bayesian_update`. The single posterior sample
/// (Krylov coefficient vector, complex-magnitude per Krylov mode) and
/// the decoded full-resolution image (Phase 5 tomographic
/// reconstruction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BayesianUpdateResult {
    /// HMC diagnostics: log-posterior at the sample.
    pub log_posterior: f64,
    /// HMC diagnostics: geometric-mean of the likelihoods (one per
    /// observation; `-1` if there were no observations, i.e. posterior
    /// == prior).
    pub mean_likelihood: f64,
    /// The full-resolution image $\vec x_{\mathrm{out}} \in \Rset^d$
    /// produced by Phase 5 tomographic reconstruction of the
    /// representative (final) HMC draw.
    pub image: Vec<f64>,
    /// The full-resolution image decoded from the **posterior-mean**
    /// point estimate — the Karcher (Fréchet) mean of the post-burn-in
    /// HMC chain on the projective unit sphere of $\Cset^m$. This is the
    /// denoised estimate that integrates over the whole typical set,
    /// rather than a single stochastic draw (`image`). Empty if there
    /// were no post-burn-in samples.
    #[serde(default)]
    pub posterior_mean: Vec<f64>,
    /// The number of post-burn-in HMC samples averaged into
    /// `posterior_mean`.
    #[serde(default)]
    pub n_samples: usize,
    /// The number of observations $N$ (cached for the agent surface).
    pub n_observations: usize,
    /// Wall-clock time for the HMC + decode in milliseconds.
    pub solve_ms: u64,
}

// ---------------------------------------------------------------------------
// P8.8: belief propagation (chain exact BP on the Krylov coefficients)
// ---------------------------------------------------------------------------

/// Configuration for chain belief propagation (P8.8, qfm::bayes::
/// `belief_propagation_chain`). This is a fast alternative to HMC for
/// product-of-likelihoods posteriors; complexity is $O(\mathrm{max\_iter}
/// \cdot N \cdot m)$ instead of HMC's $O(\mathrm{leapfrog\_steps} \cdot
/// N \cdot m)$.
///
/// **Use case:** when the user wants a **point estimate** (the marginal
/// mode of the chain-posterior) without paying the HMC sampling cost.
/// The returned MAP is a gradient-ascent solution on the log posterior
/// from the prior-initialization; it is not a sample from the posterior
/// and does not estimate the typical set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefPropagationOptsSpec {
    /// Maximum number of gradient-ascent iterations. Default 200.
    #[serde(default = "default_bp_max_iter")]
    pub max_iter: usize,
    /// Step size for gradient ascent. Default 0.05.
    #[serde(default = "default_bp_step_size")]
    pub step_size: f64,
    /// Convergence tolerance on $|\log P^{(t+1)} - \log P^{(t)}|$.
    /// Default 1e-6.
    #[serde(default = "default_bp_tol")]
    pub tol: f64,
}

fn default_bp_max_iter() -> usize {
    200
}
fn default_bp_step_size() -> f64 {
    0.05
}
fn default_bp_tol() -> f64 {
    1e-6
}

impl Default for BeliefPropagationOptsSpec {
    fn default() -> Self {
        Self {
            max_iter: default_bp_max_iter(),
            step_size: default_bp_step_size(),
            tol: default_bp_tol(),
        }
    }
}

impl BeliefPropagationOptsSpec {
    /// Validate the BP options. Mirrors `HmcOptsSpec::validate` (P7.5).
    /// Returns a list of per-field `RepairHint`s suitable for surfacing
    /// via `uk_belief_propagation`'s `UK-1001` diagnostic.
    pub fn validate(&self) -> Vec<crate::codes::RepairHint> {
        use crate::codes::{HintKind, RepairHint};
        let mut hints = Vec::new();
        if self.max_iter == 0 {
            hints.push(RepairHint::new(
                HintKind::SetParam,
                "opts.max_iter",
                "max_iter must be > 0 (set to 200 for a typical chain BP run)",
            ));
        }
        if self.step_size <= 0.0 || self.step_size.is_nan() {
            hints.push(RepairHint::new(
                HintKind::SetParam,
                "opts.step_size",
                "step_size must be a positive finite number (e.g. 0.05)",
            ));
        }
        if self.tol <= 0.0 || self.tol.is_nan() {
            hints.push(RepairHint::new(
                HintKind::SetParam,
                "opts.tol",
                "tol must be a positive finite number (e.g. 1e-6)",
            ));
        }
        hints
    }
}

/// Request body for `uk_belief_propagation`. Same observation format
/// as `BayesianUpdateRequest` (d-dim vectors matching the pipeline's
/// training-data dimension).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefPropagationRequest {
    /// The list of new observations.
    pub observations: Vec<Vec<f64>>,
    /// BP configuration.
    #[serde(default)]
    pub opts: BeliefPropagationOptsSpec,
}

/// Result body for `uk_belief_propagation`. The MAP (maximum a
/// posteriori) point estimate and the log-posterior at the MAP.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefPropagationResult {
    /// The full-resolution image decoded from the MAP Krylov coefficient
    /// vector $\vec c^* \in \Cset^m$ (Phase 5 tomographic reconstruction).
    pub image: Vec<f64>,
    /// The log-posterior at the MAP (up to a constant).
    pub log_posterior: f64,
    /// The number of observations $N$ (cached for the agent surface).
    pub n_observations: usize,
    /// The number of cumulative-product sweeps used (always 1 for the
    /// exact chain case; reserved for the future loopy-BP generalization).
    pub n_sweeps: usize,
    /// Wall-clock time for the BP + decode in milliseconds.
    pub solve_ms: u64,
}

// ── Lean4 proof verification (S29, nanoda_lib) ────────────────────────
//
// The kernel mixes numerical calculation with machine-checked proof
// verification. A `LeanVerifySpec` describes how a Lean4 export file (the
// `lean4export` NDJSON format) is type-checked by the external kernel
// `nanoda_lib`; the result is reduced to a boolean verdict
// (`ProofReport.verified`) — the proof-irrelevance analogue of the
// interaction-net unique-normal-form reduction (`logos::deltanet`).

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeanVerifySpec {
    /// Axioms the export file is permitted to use. Typical Lean exports
    /// need `Quot.sound`, `Classical.choice`, `propext`, `Lean.trustCompiler`.
    #[serde(default)]
    pub permitted_axioms: Vec<String>,
    /// If true, an exported axiom outside `permitted_axioms` aborts checking.
    /// If false, unpermitted axioms are skipped (not added to the
    /// environment) and only error if actually used.
    #[serde(default = "default_true")]
    pub unpermitted_axiom_hard_error: bool,
    /// Enable nanoda's Nat kernel extension (numeric literal reduction).
    #[serde(default)]
    pub nat_extension: bool,
    /// Enable nanoda's String kernel extension.
    #[serde(default)]
    pub string_extension: bool,
    /// Maximum export size in bytes (default 16 MiB). Larger exports are
    /// rejected before any parsing work.
    #[serde(default = "default_max_export_bytes")]
    pub max_export_bytes: usize,
    /// If true, a proof that fails to type-check is a hard UK-4801 error
    /// instead of the default `verified: false` verdict. Strict mode is for
    /// callers that must *fail closed* on a rejected proof rather than
    /// inspect a boolean report.
    #[serde(default)]
    pub strict: bool,
}

fn default_true() -> bool {
    true
}

fn default_max_export_bytes() -> usize {
    16 * 1024 * 1024
}

impl Default for LeanVerifySpec {
    fn default() -> Self {
        Self {
            permitted_axioms: Vec::new(),
            unpermitted_axiom_hard_error: true,
            nat_extension: false,
            string_extension: false,
            max_export_bytes: default_max_export_bytes(),
            strict: false,
        }
    }
}

impl LeanVerifySpec {
    /// The three semi-official Lean axioms plus the compiler trust axiom,
    /// matching nanoda's documented default profile.
    pub fn standard_axioms() -> Self {
        Self {
            permitted_axioms: vec![
                "Quot.sound".to_string(),
                "Classical.choice".to_string(),
                "propext".to_string(),
                "Lean.trustCompiler".to_string(),
            ],
            ..Self::default()
        }
    }
}

/// Result of type-checking a Lean4 export file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofReport {
    /// Boolean verdict: true iff every declaration type-checked. This is the
    /// "reduce a proof to `true`/`false`" step — by proof irrelevance all
    /// proofs of a proposition are definitionally equal, so a successful
    /// kernel run collapses the export to `true`.
    pub verified: bool,
    /// Number of declarations checked by the kernel.
    pub declarations_checked: usize,
    /// Name of the theorem that failed (if a specific declaration failed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failing_theorem: Option<String>,
    /// Human-readable kernel error (empty on success).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// SHA-256 of the export payload, for content-addressed audit (the
    /// `unf_hash` analogue: a canonical digest of the checked term).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_hash: Option<String>,
}

// ── Logos CNL->UNF coupling ──────────────────────────────────────────
//
// The kernel couples the Logos controlled-natural-language compiler
// (`logos` crate) to the session model. A CNL sentence is parsed, compiled
// to a CoreIR term, reduced to an interaction-net unique normal form (UNF),
// and read back. The report carries the readback string, its SHA-256 digest,
// and a confluence self-check: the same input reduced twice must yield an
// identical UNF (the "unique normal form" guarantee), the proof-irrelevance
// analogue of `logos::deltanet`'s unique-normal-form hash.

/// Result of compiling a CNL sentence through the Logos pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogosReport {
    /// The readback of the reduced net (e.g. `"Love(john, mary)"`, `"9.5"`,
    /// `"true"`). This is the human-readable normal form.
    pub result: String,
    /// The content-addressable UNF digest (SHA-256 of the canonical net
    /// serialization).
    pub unf_hash: String,
    /// True iff reducing the same sentence twice yields the identical UNF —
    /// the confluence / unique-normal-form self-check. (Confluence is also
    /// argued in `logos/docs/LOGOS.md`; the runtime check corroborates it.)
    pub verified: bool,
    /// The CNL sentence that was compiled (echoed for audit).
    pub sentence: String,
}

/// Result of translating an AustralVM-language source fragment to a unique
/// normal form through DeltaNets (the `uk_austral_unf` symbol).
///
/// The source is lowered to CoreIR, compiled to an interaction net, reduced
/// to its unique normal form, and read back as a **symbolic expression**
/// ([`SymExpr`](logos::deltanet::symbolic::SymExpr)). Whenever the term has
/// **no unknown variables** the expression collapses to the numerical result
/// of its calculation (`value`), e.g. `ADD` of two 64-bit integers → `5`;
/// when unknowns remain (`Add64(x, 3)`) `value` is `None` and the symbolic
/// expression is the answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AustralReport {
    /// The symbolic readback of the reduced net in canonical prefix form
    /// (e.g. `"20"`, `"(Add64 x 3)"`).
    pub sym_expr: String,
    /// The arithmetic fragment in infix form (e.g. `"(x + 3)"`, `"(2 + 3) * 4"`).
    pub infix: String,
    /// The numerical result of the calculation, when the term is closed (no
    /// unknowns): `Some("5")` for `ADD(2, 3)`, `None` for open terms.
    pub value: Option<String>,
    /// The content-addressable UNF digest (SHA-256 of the canonical net
    /// serialization).
    pub unf_hash: String,
    /// The word-level TED (Phase 3): the canonical polynomial normal form of
    /// the Int64-arithmetic fragment over ℤ/2⁶⁴ (e.g. `"3*x + 6"`, `"20"`),
    /// when the term lies in that fragment; `None` otherwise.
    pub ted: Option<String>,
    /// SHA-256 of the canonical TED serialization — the content-addressable
    /// *algebraic* UNF, independent of the net encoding.
    pub ted_hash: Option<String>,
    /// True iff reducing the same source twice yields the identical UNF —
    /// the confluence / unique-normal-form self-check.
    pub verified: bool,
    /// The Austral source that was translated (echoed for audit).
    pub source: String,
}

// ── Cadabra2 symbolic coupling (S30) ──────────────────────────────────
//
// The kernel couples the existing LaTeX symbolic engine
// (`nested_fock_algebra::compile_latex` → CAS-string dialect) with the
// external field-theory CAS Cadabra2 (GPL-3.0, invoked as a **subprocess**
// `cadabra2-cli` so the Rust binary stays an independent work). Cadabra2
// consumes the same TeX-subset input the mathhook parser handles and
// returns a canonicalized expression plus a zero-detection verdict — the
// symbolic analogue of `logos::deltanet`'s unique-normal-form reduction.

/// A symbolic operation to run in Cadabra2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolicOp {
    /// Canonicalize: reduce the expression to a unique canonical form
    /// (Cadabra2 `@canonicalise`), preserving operator order (no
    /// commutativity assumptions).
    Canonicalize,
    /// Expand then canonicalize (full algebraic simplification).
    Simplify,
    /// Verify a Hermiticity identity: the expression is interpreted as
    /// `H - H†` and `verified` reports whether it canonicalizes to zero.
    VerifyHermitian,
    /// Verify a substitution identity (e.g. a constraint resolution): apply
    /// the `substitution` rule to the expression, canonicalize, and report
    /// whether the result is identically zero.
    VerifySubstitution,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicSpec {
    /// The expression, in TeX-subset (Cadabra2 `Ex(...)` input) or the
    /// CAS-string dialect (`c_0 * a_0`).
    pub expression: String,
    /// The operation to run.
    pub op: SymbolicOp,
    /// A Cadabra2 substitution rule for [`SymbolicOp::VerifySubstitution`]
    /// (e.g. `u_33 -> -(u_11 + u_22)`), applied before canonicalization.
    #[serde(default)]
    pub substitution: Option<String>,
    /// Subprocess timeout in milliseconds (default 30 s).
    #[serde(default = "default_symbolic_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_symbolic_timeout_ms() -> u64 {
    30_000
}

impl Default for SymbolicSpec {
    fn default() -> Self {
        Self {
            expression: String::new(),
            op: SymbolicOp::Canonicalize,
            substitution: None,
            timeout_ms: default_symbolic_timeout_ms(),
        }
    }
}

/// Result of a Cadabra2 symbolic run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicReport {
    /// The canonicalized expression, in Cadabra2's output notation. For
    /// [`SymbolicOp::VerifyHermitian`] this is the normal form of `H - H†`.
    pub normal_form: String,
    /// Boolean verdict: for [`SymbolicOp::VerifyHermitian`], whether the
    /// expression canonicalizes to zero (the identity holds). For
    /// [`SymbolicOp::Canonicalize`]/[`SymbolicOp::Simplify`], whether the
    /// expression is identically zero.
    pub verified: bool,
    /// SHA-256 of the canonical normal form — the `unf_hash` analogue: a
    /// canonical digest of the reduced expression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_form_hash: Option<String>,
    /// Human-readable kernel/engine error (empty on success).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Wall-clock engine time in milliseconds.
    pub engine_ms: u64,
}

// ── WhyML codegen (S36, the Why3 cycle) ──────────────────────────────
//
// The kernel produces WhyML programs that the external Why3 toolchain
// verifies (provers discharge the lemmas/postconditions) and extracts to
// OCaml modules that extend the australVM compiler (see
// `docs/WHYML_CYCLE.md`). The default emitted program is the
// **authorization gate**: the kernel's own grant-subset semantics
// (`GrantSet::is_subset_of`, S21) written as a WhyML function whose
// postcondition is the soundness+completeness statement
// `authorize grants required = true  <->  required ⊆ grants`. Why3 proves
// the subset lemmas (reflexivity/transitivity — the "no escalation path"
// property) and the postcondition; extraction is semantics-preserving, so
// the OCaml module the australVM compiler loads satisfies the property.

/// What the WhyML pipeline should do with the emitted program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhymlOp {
    /// Emit the `.mlw` program only (pure; no external engine needed).
    #[default]
    Emit,
    /// Emit and run `why3 prove` (the external Why3 toolchain, invoked as a
    /// subprocess exactly like Cadabra2 in S30) to discharge the lemmas and
    /// postconditions; `verified` reports whether every goal was proved.
    Prove,
}

/// Which WhyML program the kernel emits (S36 + GPU.md). Additive: each
/// variant is a `.mlw` the same Why3 cycle proves and extracts to an OCaml
/// module that the australVM compiler loads as a pass plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhymlProgram {
    /// The authorization gate (default): grant-subset semantics (S21) —
    /// `authorize grants required = true <-> required ⊆ grants`.
    #[default]
    AuthorizeGate,
    /// The NPU DMA gate (GPU.md): a DMA transfer into a linear NPU buffer is
    /// physically safe iff it stays inside the SRAM —
    /// `buf.offset + bytes <= MAX_NPU_SRAM`. The extracted OCaml decides
    /// `(offset, bytes)` pairs; the compiler pass rejects any module whose
    /// declared DMA transfers exceed the hardware limit.
    NpuDmaGate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhymlSpec {
    /// WhyML module name (default `"AuthorizeGate"`).
    #[serde(default = "default_whyml_module_name")]
    pub module_name: String,
    /// Name of the entrypoint `let` extracted for the compiler pass
    /// (default `"gate_verdict"`).
    #[serde(default = "default_whyml_fn_name")]
    pub function_name: String,
    /// The module manifest's granted `uk_*` symbols (the `GrantSet.kernel`
    /// namespace). Validated against the kernel's own symbol registry.
    #[serde(default)]
    pub grants: Vec<String>,
    /// The `uk_*` symbols the compiled module imports (what the gate checks).
    #[serde(default)]
    pub required: Vec<String>,
    /// Optional `uk_*` symbols the emitted WhyML declares as *external*
    /// kernel calls (`val` declarations) — the "call the probability kernel
    /// from WhyML" direction; the extracted OCaml binds them at link time.
    #[serde(default)]
    pub kernel_externals: Vec<String>,
    /// Which program to emit (default [`WhymlProgram::AuthorizeGate`]).
    #[serde(default)]
    pub program: WhymlProgram,
    /// The operation to run.
    #[serde(default)]
    pub op: WhymlOp,
    /// Subprocess timeout for `why3 prove` in milliseconds (default 30 s).
    #[serde(default = "default_whyml_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_whyml_module_name() -> String {
    "AuthorizeGate".into()
}
fn default_whyml_fn_name() -> String {
    "gate_verdict".into()
}
fn default_whyml_timeout_ms() -> u64 {
    30_000
}

impl Default for WhymlSpec {
    fn default() -> Self {
        Self {
            module_name: default_whyml_module_name(),
            function_name: default_whyml_fn_name(),
            grants: Vec::new(),
            required: Vec::new(),
            kernel_externals: Vec::new(),
            program: WhymlProgram::AuthorizeGate,
            op: WhymlOp::Emit,
            timeout_ms: default_whyml_timeout_ms(),
        }
    }
}

/// Result of the WhyML pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhymlReport {
    /// The emitted WhyML program (a `.mlw` file).
    pub whyml: String,
    /// SHA-256 of the emitted program (content-addressed audit; the OCaml
    /// extraction embeds it so a loaded plugin can be traced to the emitted
    /// source).
    pub sha256: String,
    /// Number of proof obligations the `.mlw` declares (the subset lemmas +
    /// the postconditions of `authorize` and the gate entrypoint).
    pub proof_obligations: usize,
    /// For [`WhymlOp::Prove`]: whether `why3 prove` discharged every goal.
    /// `None` for [`WhymlOp::Emit`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    /// Human-readable engine error (empty on success).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Wall-clock engine time in milliseconds (0 for `Emit`).
    pub engine_ms: u64,
}

// ── Federation types (QuePaxa plan, 6xxx codes) ──────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityOpKind {
    Create,
    Update,
    Revoke,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityOp {
    pub did: String,
    pub op_kind: IdentityOpKind,
    #[serde(with = "hex_bytes_32")]
    pub signing_key: [u8; 32],
    #[serde(with = "hex_bytes_64")]
    pub signature: [u8; 64],
    pub seq: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_endpoint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionOp {
    pub did: String,
    pub model_id: u64,
    pub op: AgentRequest,
    #[serde(with = "hex_bytes_64")]
    pub signature: [u8; 64],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkRef {
    pub index: u32,
    pub cid: String,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentRef {
    pub cid: String,
    pub magnet_uri: String,
    pub encryption_key: String,
    pub filesize: u64,
    pub mime_type: String,
    #[serde(default)]
    pub chunks: Vec<ChunkRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentOp {
    pub did: String,
    pub content_ref: ContentRef,
    #[serde(with = "hex_bytes_64")]
    pub signature: [u8; 64],
}

// ── Carbon-certificate / UTXO ledger (ReFi exchange, Plan R) ──────────
//
// A carbon certificate is held as an unspent transaction output (UTXO). The
// transparent consensus core keeps `amount`/`owner`/`blinding` explicit so the
// QuePaxa state-transition engine can run the "rapid validation rule" offline;
// a RISC-Zero layer (Plan R Phase 1) replaces the transparent fields with a
// commitment `Hash(amount, owner, blinding)` and proves conservation inside the
// zkVM, leaving only coin_ids + nullifiers on the wire.

/// A 32-byte payment/certificate identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CertId(pub [u8; 32]);

/// A spent-input marker. Double-spend protection: a nullifier may appear in at
/// most one committed transaction. In the transparent core it is derived
/// deterministically from the coin_id; a confidential layer would use
/// `Hash(spend_key, coin_commitment)` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Nullifier(pub [u8; 32]);

/// A reference to (or declaration of) a certificate input/output. The node
/// cross-checks `amount`/`owner` against the ledger's stored coin for inputs,
/// so a spender cannot lie about value or ownership.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoinRef {
    pub coin_id: CertId,
    pub amount: u64,
    pub owner: String,
}

/// The kind of certificate state transition carried by a [`CertificateOp`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CertificateOpKind {
    /// Issuance of a fresh certificate out of "thin air" (backed by an oracle
    /// proof — e.g. the UNFCCC zk-TLS bridge). Only the mint authority may mint.
    Mint {
        amount: u64,
        owner: String,
        #[serde(with = "hex_bytes_32")]
        blinding: [u8; 32],
        /// Optional provenance/backing reference (e.g. `unfccc:vc:<orderId>`
        /// pointing at `offset.climateneutralnow.org/vchistory/details?orderId=N`,
        /// where the "Reason for cancellation" field carries the user's
        /// `did:unfer` public key).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
    },
    /// A transfer: spends `inputs`, creates `outputs`, conserving total value.
    Transfer {
        inputs: Vec<CoinRef>,
        outputs: Vec<CoinRef>,
    },
    /// Retirement/destruction: spends `inputs` and removes their value from the
    /// circulating supply (conservation is intentionally *not* required here).
    Burn { inputs: Vec<CoinRef> },
}

/// A signed certificate state transition. Mirrors the other consensus ops: the
/// `did` is the acting principal (mint authority for a Mint, the spender for a
/// Transfer/Burn) and `signature` covers the op bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CertificateOp {
    pub did: String,
    pub kind: CertificateOpKind,
    pub seq: u64,
    #[serde(with = "hex_bytes_64")]
    pub signature: [u8; 64],
}

/// An oracle-backed mint request (Plan R Phase 3). This is the **shared
/// contract** between the zk-TLS client (which proves the UNFCCC cancellation
/// record) and the mint authority (which signs the resulting
/// `CertificateOp::Mint`). The `source` MUST reference a public UN oracle
/// record `unfccc:vc:<orderId>` —
/// `https://offset.climateneutralnow.org/vchistory/details?orderId=<orderId>` —
/// where the "Reason for cancellation" field carries the owner's `did:unfer`
/// public key and the serial-number range pins the tonnage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MintRequest {
    /// DID that will own the minted certificate.
    pub owner: String,
    /// Tonnage (Mg CO₂e) cancelled on the UN platform.
    pub amount: u64,
    /// UN oracle provenance: `unfccc:vc:<orderId>`.
    pub source: String,
    /// Optional blinding. When omitted, a deterministic per-request value is
    /// derived from `source` so the coin_id is reproducible by the client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blinding: Option<[u8; 32]>,
}

impl MintRequest {
    /// Validate that `source` references a well-formed UN oracle record
    /// (`unfccc:vc:<orderId>`). An empty or malformed source fails
    /// UK-7007 `CERT_ORACLE_REJECTED`.
    pub fn validate_source(&self) -> Result<(), crate::codes::Code> {
        let prefix = "unfccc:vc:";
        if !self.source.starts_with(prefix) || self.source.len() == prefix.len() {
            return Err(crate::codes::Code::CERT_ORACLE_REJECTED);
        }
        let order_id = &self.source[prefix.len()..];
        if order_id
            .chars()
            .any(|c| !c.is_ascii_alphanumeric() && c != '-')
        {
            return Err(crate::codes::Code::CERT_ORACLE_REJECTED);
        }
        Ok(())
    }

    /// The blinding for this request: the explicit value if given, else a
    /// deterministic hash of the source so both client and authority derive
    /// the same coin_id.
    pub fn effective_blinding(&self) -> [u8; 32] {
        match self.blinding {
            Some(b) => b,
            None => {
                use sha2::{Digest, Sha256};
                let mut ctx = Sha256::new();
                ctx.update(b"unfer:mint_request");
                ctx.update(self.source.as_bytes());
                ctx.update(self.owner.as_bytes());
                ctx.update(self.amount.to_le_bytes());
                ctx.finalize().into()
            }
        }
    }

    /// Build the mint `CertificateOpKind` for this request (still needs to be
    /// wrapped in a `CertificateOp` and signed by the authority).
    pub fn to_mint_kind(&self) -> CertificateOpKind {
        CertificateOpKind::Mint {
            amount: self.amount,
            owner: self.owner.clone(),
            blinding: self.effective_blinding(),
            source: Some(self.source.clone()),
        }
    }
}

/// A 32-byte auction identifier (deterministic lot commitment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AuctionId(pub [u8; 32]);

/// What is being auctioned in a unified auction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AuctionAsset {
    /// A carbon-credit lot: `amount` credits (Mg CO₂e) offered for sale. The
    /// seller's certificate for the lot transfers to the winner on settlement.
    CarbonCredits { amount: u64 },
    /// A publicity / ad-inventory slot (the AdSense alternative). `slot` is a
    /// publisher inventory id (e.g. `homepage_leaderboard_300x250`). There is
    /// no ledger asset to deliver — settlement is the winner's payment only.
    PublicitySlot {
        slot: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

/// The payment medium a winning bid is denominated in. Both rails settle as
/// ordinary certificates on the ledger: Taler e-coins are fiat-backed
/// certificates the seller can deposit for fiat; carbon credits are certificates
/// traded directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuctionCurrency {
    /// GNU Taler e-coins (fiat-backed digital cash).
    Taler,
    /// Carbon-credit certificates on the ledger.
    CarbonCredits,
}

/// A lot offered at a unified auction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuctionLot {
    pub lot_id: AuctionId,
    pub seller_did: String,
    pub asset: AuctionAsset,
    /// The payment medium the winning bid must be denominated in.
    pub currency: AuctionCurrency,
    /// Minimum price per unit (Prebid-style price floor). Bids below it are
    /// rejected at the ledger.
    pub floor: u64,
    pub opens_seq: u64,
    pub closes_seq: u64,
}

/// A single bid in a unified auction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuctionBid {
    pub lot_id: AuctionId,
    pub bidder_did: String,
    pub price_per_unit: u64,
    pub quantity: u64,
    pub seq: u64,
}

/// The deterministic winner of a closed auction. Computed from the recorded
/// bids by the unified-auction clearing rule: the highest `price_per_unit`
/// wins; ties break to the earliest `seq`. Every node replays the same log and
/// converges on the same winner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuctionWinner {
    pub lot_id: AuctionId,
    pub bidder_did: String,
    pub price_per_unit: u64,
    pub quantity: u64,
    pub total: u64,
}

/// The kind of auction state transition carried by an [`AuctionOp`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AuctionOpKind {
    /// Seller opens a lot for bidding. Rejected if the lot already exists.
    Open { lot: AuctionLot },
    /// Bidder places a price-per-unit bid. Rejected below floor, against the
    /// seller, or after the lot has closed.
    Bid {
        lot_id: AuctionId,
        price_per_unit: u64,
        quantity: u64,
    },
    /// Seller closes the lot; the deterministic winner is computed from the
    /// recorded bids. Rejected if the lot is unknown or already closed.
    Close { lot_id: AuctionId },
}

/// A signed auction state transition. Mirrors the other consensus ops: `did`
/// is the acting principal (the seller for Open/Close, the bidder for Bid)
/// and `signature` covers the op bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuctionOp {
    pub did: String,
    pub kind: AuctionOpKind,
    pub seq: u64,
    #[serde(with = "hex_bytes_64")]
    pub signature: [u8; 64],
}

/// A read-only snapshot of an auction lot for reporting/querying.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuctionReport {
    pub lot: AuctionLot,
    pub bids: Vec<AuctionBid>,
    pub closed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<AuctionWinner>,
}

// ── Math catastrophe bond (SPV with nanoda trigger, Plan R) ───────────
//
// A math bond is a catastrophe bond whose trigger is a *purely mathematical*
// proof: the sponsor locks collateral (e-coins/certificates), investors buy
// the bond for a coupon, and if nanoda verifies a Lean4-export proof of the
// specified theorem, the collateral is paid out as a bounty to the researcher
// plus a catastrophe payment to the sponsor. If the proof never arrives before
// maturity, investors recover their principal plus coupon.
//
// The trigger engine is `prob_kernel::verify::verify_export` (nanoda_lib)
// running deterministically inside the consensus node's `apply_op`.
//
// Bond probability trading uses the unified auction mechanism (Prebid-model)
// for conditional-token-like shares of the trigger probability.

/// A 32-byte math bond identifier (deterministic content commitment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MathBondId(pub [u8; 32]);

/// The mathematical trigger specification for a bond.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MathBondTrigger {
    /// Short label identifying the theorem (e.g. "P_eq_NP", "RiemannHypothesis").
    pub theorem: String,
    /// SHA-256 hash of the expected Lean4 export specification.
    pub spec_hash: String,
    /// Maximum size (bytes) of a submitted proof export.
    pub max_export_bytes: usize,
    /// Permitted Lean4 axioms for the nanoda verifier.
    pub permitted_axioms: Vec<String>,
    /// Whether strict mode is enabled (UK-7401 on rejection).
    pub strict: bool,
    /// Enable nanoda's Nat kernel extension (numeric literal reduction).
    #[serde(default)]
    pub nat_extension: bool,
    /// Enable nanoda's String kernel extension.
    #[serde(default)]
    pub string_extension: bool,
}

/// Lifecycle of a math bond.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathBondState {
    /// Bond issued but not yet fully funded.
    Issued,
    /// Bond fully funded by investors; awaiting proof or maturity.
    Funded,
    /// A proof was submitted and nanoda verified it — trigger fired.
    Triggered,
    /// Bond reached maturity without a successful trigger.
    Matured,
    /// Collateral distributed (trigger payout or maturity refund).
    Settled,
}

/// The kind of math bond state transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MathBondOpKind {
    /// Sponsor issues a new math bond. The sponsor locks `principal` e-coins
    /// as collateral and specifies the trigger theorem, coupon, and maturity.
    Issue {
        trigger: MathBondTrigger,
        /// Collateral amount locked by the sponsor (e-coins).
        principal: u64,
        /// Coupon rate in basis points (e.g. 500 = 5%).
        coupon_rate_bps: u64,
        /// Consensus-log seq at which the bond matures without trigger.
        maturity_seq: u64,
        /// DID of the researcher authorized to submit proofs.
        researcher_did: String,
    },
    /// An investor funds the bond by escrowing e-coins.
    Invest {
        bond_id: MathBondId,
        /// Amount of e-coins the investor puts in.
        amount: u64,
    },
    /// Researcher submits a proof attempt. The ledger runs nanoda verification
    /// deterministically — if the proof checks, the trigger fires.
    SubmitProof {
        bond_id: MathBondId,
        /// The raw lean4export NDJSON payload.
        export_bytes: Vec<u8>,
    },
    /// Record that the bond reached its `maturity_seq` without a successful
    /// trigger, moving it `Issued`/`Funded` → `Matured`. Anyone may submit it
    /// (recording the passage of time); the ledger enforces that the consensus
    /// log is actually at/past `maturity_seq`. A `Matured` bond can then be
    /// settled as a maturity refund.
    Mature {
        bond_id: MathBondId,
    },
    /// Finalize the bond: distribute collateral per the trigger/maturity
    /// outcome. Allowed only for a `Triggered` bond (trigger payout) or a
    /// `Matured` bond (maturity refund) — never for a live `Issued`/`Funded`
    /// bond, whose trigger window is still open.
    Settle {
        bond_id: MathBondId,
    },
}

/// A signed math bond state transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MathBondOp {
    pub did: String,
    pub kind: MathBondOpKind,
    pub seq: u64,
    #[serde(with = "hex_bytes_64")]
    pub signature: [u8; 64],
}

/// A read-only snapshot of a math bond for reporting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MathBondReport {
    pub bond_id: MathBondId,
    pub trigger: MathBondTrigger,
    pub state: MathBondState,
    pub principal: u64,
    pub invested: u64,
    pub coupon_rate_bps: u64,
    pub maturity_seq: u64,
    pub researcher_did: String,
    pub sponsor_did: String,
    /// The proof report if a proof was submitted (None = no submission yet).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_report: Option<ProofReport>,
    /// The consensus-log seq at which the trigger fired (None = no trigger yet).
    /// Market resolution reads this to pick the winning outcome deterministically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_seq: Option<u64>,
}

// ── Math bond probability market (vAMM + NegRisk, Plan R) ──────────
//
// The probability market for math bond triggers uses two complementary
// designs:
//
// 1. **Azuro vAMM** (virtual Automated Market Maker): a singleton
//    concentrated-liquidity pool where LPs deposit e-coins and the protocol
//    mathematically prices the odds of the trigger firing without needing
//    a direct buyer for every seller. The pool acts as counterparty.
//
// 2. **NegRisk CTF Adapter**: mutually-exclusive conditional outcomes
//    (e.g. "triggered by 2025" vs "triggered by 2026" vs "never") share
//    a single pool, preventing liquidity fragmentation. When one outcome
//    resolves, the others become worthless — the "negated risk" adapter
//    ensures the outcome prices sum to 1.
//
// The market settles through the certificate ledger (Taler e-coins)
// exactly like the auction + escrow pattern.

/// A 32-byte pool identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PoolId(pub [u8; 32]);

/// A 32-byte outcome identifier within a pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutcomeId(pub [u8; 32]);

/// A liquidity pool for a math bond (vAMM style).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiquidityPool {
    pub pool_id: PoolId,
    /// The bond this pool prices.
    pub bond_id: MathBondId,
    /// Per-outcome reserve (outcome_id → e-coin amount).
    pub outcome_reserves: Vec<(OutcomeId, u64)>,
    /// Total e-coins in the pool.
    pub total_reserve: u64,
    /// LP share holders (DID → share amount).
    pub lp_shares: Vec<(String, u64)>,
    /// Total LP shares outstanding.
    pub total_shares: u64,
    /// Trading fee in basis points (e.g. 300 = 3%).
    pub fee_bps: u64,
    /// Whether the pool has been resolved (one outcome triggered).
    pub resolved: bool,
    /// The winning outcome (if resolved).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<OutcomeId>,
}

/// A read-only snapshot of a pool for reporting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoolReport {
    pub pool: LiquidityPool,
    /// Current prices per outcome (0.0..1.0, summing to 1.0).
    pub prices: Vec<(OutcomeId, f64)>,
}

/// A single NegRisk outcome within a pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NegRiskOutcome {
    pub outcome_id: OutcomeId,
    pub pool_id: PoolId,
    /// Human-readable label (e.g. "P=NP proved by 2025").
    pub label: String,
    /// Consensus-log seq at which this outcome matures.
    pub maturity_seq: u64,
}

/// The kind of market state transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MarketOpKind {
    /// LP adds liquidity to the pool. Receives LP shares proportional to
    /// the deposit relative to the pool's total reserve.
    AddLiquidity {
        pool_id: PoolId,
        /// E-coin amount to deposit.
        amount: u64,
    },
    /// LP removes liquidity. Burns LP shares and withdraws proportional
    /// reserve (minus any unrealized losses from Impermanent Loss).
    RemoveLiquidity {
        pool_id: PoolId,
        /// Number of LP shares to burn.
        shares: u64,
    },
    /// Trader buys outcome tokens at the current vAMM price.
    /// NegRisk: the pool ensures outcome prices sum to 1.
    BuyOutcome {
        pool_id: PoolId,
        outcome_id: OutcomeId,
        /// E-coin amount the trader is willing to pay.
        amount: u64,
    },
    /// Trader sells outcome tokens back to the pool.
    SellOutcome {
        pool_id: PoolId,
        outcome_id: OutcomeId,
        /// Number of outcome tokens to sell.
        amount: u64,
    },
    /// Open a NegRisk pool with multiple mutually-exclusive outcomes.
    OpenNegRisk {
        bond_id: MathBondId,
        outcomes: Vec<NegRiskOutcome>,
        /// Trading fee in basis points.
        fee_bps: u64,
    },
    /// Resolve the pool: the bond's trigger fired (or the bond matured without
    /// one). The winning outcome is NOT a caller choice — it is a pure function
    /// of the bond's trigger signal and the outcome maturity windows: the
    /// outcome whose window contains `trigger_seq` wins, `None` (the bond
    /// matured without a trigger) selects the terminal "never" outcome
    /// (`maturity_seq == u64::MAX`). The consensus node validates `trigger_seq`
    /// against the deterministic bond ledger before the op is applied.
    Resolve {
        pool_id: PoolId,
        /// The bond's trigger signal: `Some(consensus seq)` when the trigger
        /// fired, `None` when the bond matured without a trigger.
        trigger_seq: Option<u64>,
    },
    /// Post-resolution withdrawal: an actor redeems their winning outcome
    /// tokens (pro-rata against the pool reserve) and their LP share (accrued
    /// fees; plus the whole reserve when nobody held winning tokens). Idempotent
    /// — a second claim pays nothing.
    Claim {
        pool_id: PoolId,
    },
}

/// A signed market state transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketOp {
    pub did: String,
    pub kind: MarketOpKind,
    pub seq: u64,
    #[serde(with = "hex_bytes_64")]
    pub signature: [u8; 64],
}

// ── Attribution Carbon Credits (Open Badges + Taler micropayments) ────
//
// The Adidas/Yeezy pattern as a tradable certificate: Author A pays Author B
// for the right to publicly claim that A's item is derived from B's item and
// that B approves that attribution — the difference between an attribution
// and a *negotiated, author-approved* attribution. The credit is minted on
// Author B's on-ledger approval (the fee is escrowed beforehand), lives on
// the certificate ledger exactly like a carbon credit, and is rendered as a
// deterministic Open Badges 3.0 assertion (public, or exclusive to an
// anonymous viewer identified by a hash of a random key their browser
// generates — the per-visualization badge of a YouTube-style context).
//
// This is complementary to copyright licensing: it works for public-domain
// and private items alike, because it is issued by the author being
// attributed, not by the law.

/// A 32-byte attribution credit identifier — the deterministic commitment of
/// the full offer terms (derived item, original item, fee, context,
/// exclusivity) plus both authors, so identical terms by the same pair of
/// authors collide (a credit is a unique fact).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttributionCreditId(pub [u8; 32]);

/// A 32-byte Open Badges assertion identifier — the deterministic commitment
/// of `(credit_id, recipient)`. The same credit issued to the same recipient
/// (public, or the same anonymous viewer hash) is the *same* badge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttributionBadgeId(pub [u8; 32]);

/// A registered work item: the derived work (owned by Author A) or the
/// original work (owned by the attributed Author B). Content-addressed so the
/// same work registered twice collides on the same `item_hash`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionItem {
    /// Content-addressable hash of the work (or of its canonical descriptor).
    #[serde(with = "hex_bytes_32")]
    pub item_hash: [u8; 32],
    /// Human-readable title (e.g. "Yeezy Boost 350 V2").
    pub title: String,
    /// Optional external reference (catalog page, DOI, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Lifecycle of an attribution credit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributionState {
    /// Author A offered the fee; awaiting Author B's approval. No badge may be
    /// issued yet — the attribution is not (yet) approved.
    Offered,
    /// Author B approved; the escrowed fee is paid out; badges may be issued.
    Approved,
    /// Author B revoked the endorsement. Already-issued badges stay valid as
    /// historical (content-addressed) records, but no new badge is minted.
    Revoked,
}

/// The negotiated terms of one attribution (Author A → Author B).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionOffer {
    /// A's derived work (registered to Author A).
    pub derived_item: AttributionItem,
    /// B's original work (registered to Author B).
    pub original_item: AttributionItem,
    /// Negotiated fee Author A pays Author B (e-coins), escrowed at offer time.
    pub fee: u64,
    /// Free-form context displayed on the badge (e.g. "Yeezy line, 2023
    /// collection").
    pub context: String,
    /// Exclusive: while this credit is live, Author B may not accept a second
    /// offer against the same original item — the Adidas/Yeezy exclusivity
    /// deal, not just an endorsement.
    #[serde(default)]
    pub exclusive: bool,
}

/// The kind of attribution state transition carried by an [`AttributionOp`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttributionOpKind {
    /// Author A registers a work item they own. The item is content-addressed
    /// (`item_hash`) and its owner recorded, so offers can be validated
    /// against real ownership.
    RegisterItem { item: AttributionItem },
    /// Author A offers Author B a fee for an attribution credit referencing
    /// B's original item. The fee is escrowed by the settlement service
    /// before the signed op is emitted; the ledger records the terms.
    OfferAttribution { offer: AttributionOffer },
    /// Author B approves an offered credit (`Offered` → `Approved`). This is
    /// the on-ledger moment the attribution becomes author-approved; the
    /// settlement service releases the escrowed fee to B at the same time.
    Approve { credit_id: AttributionCreditId },
    /// Author B revokes an approved credit (`Approved` → `Revoked`): the
    /// endorsement is withdrawn and no further badge is issued.
    Revoke { credit_id: AttributionCreditId },
    /// Mint a deterministic Open Badges 3.0 assertion for an approved credit.
    /// `viewer` is the SHA-256 of a random key generated by the end-user's
    /// browser — `None` for a public badge visible to all users, `Some` for a
    /// badge exclusive to that anonymous viewer (per-visualization badge; the
    /// operator never sees the random key itself, only its hash).
    IssueBadge {
        credit_id: AttributionCreditId,
        #[serde(default)]
        viewer: Option<[u8; 32]>,
    },
}

/// A signed attribution state transition. Mirrors the other consensus ops:
/// `did` is the acting principal (the item's author for RegisterItem, Author
/// A for OfferAttribution, Author B for Approve/Revoke) and `signature`
/// covers the op bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionOp {
    pub did: String,
    pub kind: AttributionOpKind,
    pub seq: u64,
    #[serde(with = "hex_bytes_64")]
    pub signature: [u8; 64],
}

/// A read-only snapshot of an attribution credit for reporting/querying.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionReport {
    pub credit_id: AttributionCreditId,
    pub offer: AttributionOffer,
    pub state: AttributionState,
    /// Author A (the derived item's owner, the payer).
    pub author_a: String,
    /// Author B (the original item's owner, the attributed party, the payee).
    pub author_b: String,
    /// Consensus-log seq at which B approved (badge issuance date derives from
    /// it deterministically). Absent while `Offered`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approve_seq: Option<u64>,
    /// Consensus-log seq at which B revoked. Absent while not revoked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revoke_seq: Option<u64>,
    /// Badges minted for this credit, in issuance order.
    #[serde(default)]
    pub badges: Vec<AttributionBadgeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConsensusTransaction {
    IdentityOp(IdentityOp),
    SessionOp(SessionOp),
    ContentOp(ContentOp),
    /// A carbon-certificate / UTXO state transition (ReFi exchange, Plan R).
    /// The state-transition engine on each QuePaxa node validates the op
    /// (mint authority, existence, conservation, double-spend, owner) before
    /// it is sequenced into the consensus log.
    CertificateOp(CertificateOp),
    /// A unified-auction state transition (Prebid-model unified auction for
    /// carbon credits and publicity inventory). Deterministic: every node
    /// replays the same log and converges on the same winner.
    AuctionOp(AuctionOp),
    /// A math catastrophe bond state transition (SPV with nanoda trigger).
    /// The trigger engine runs `verify_export` deterministically inside each
    /// node's apply — no human oracle, no external dependency.
    MathBondOp(MathBondOp),
    /// A math bond probability market state transition (vAMM + NegRisk).
    MarketOp(MarketOp),
    /// An attribution-credit state transition (Open Badges + Taler
    /// micropayments). Deterministic: every node replays the same log and
    /// converges on the same credits, approvals, revocations and badge ids.
    AttributionOp(AttributionOp),
}

impl ConsensusTransaction {
    pub fn did(&self) -> &str {
        match self {
            Self::IdentityOp(op) => &op.did,
            Self::SessionOp(op) => &op.did,
            Self::ContentOp(op) => &op.did,
            Self::CertificateOp(op) => &op.did,
            Self::AuctionOp(op) => &op.did,
            Self::MathBondOp(op) => &op.did,
            Self::MarketOp(op) => &op.did,
            Self::AttributionOp(op) => &op.did,
        }
    }

    pub fn signature(&self) -> &[u8; 64] {
        match self {
            Self::IdentityOp(op) => &op.signature,
            Self::SessionOp(op) => &op.signature,
            Self::ContentOp(op) => &op.signature,
            Self::CertificateOp(op) => &op.signature,
            Self::AuctionOp(op) => &op.signature,
            Self::MathBondOp(op) => &op.signature,
            Self::MarketOp(op) => &op.signature,
            Self::AttributionOp(op) => &op.signature,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DidEntry {
    pub did: String,
    #[serde(with = "hex_bytes_32")]
    pub pubkey: [u8; 32],
    pub seq: u64,
    pub created_at: u64,
    pub revoked: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_endpoint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DidDocument {
    #[serde(rename = "@context")]
    pub context: String,
    pub id: String,
    #[serde(rename = "verificationMethod")]
    pub verification_method: Vec<VerificationMethod>,
    pub authentication: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub service: Vec<DidService>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationMethod {
    pub id: String,
    #[serde(rename = "type")]
    pub method_type: String,
    #[serde(rename = "publicKeyMultibase")]
    pub public_key_multibase: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DidService {
    pub id: String,
    #[serde(rename = "type")]
    pub service_type: String,
    #[serde(rename = "serviceEndpoint")]
    pub service_endpoint: String,
}

impl DidEntry {
    pub fn to_document(&self) -> DidDocument {
        let key_id = format!("{}#key-1", self.did);
        let pk_hex = hex::encode(self.pubkey);
        let mut doc = DidDocument {
            context: "https://www.w3.org/ns/did/v1".to_string(),
            id: self.did.clone(),
            verification_method: vec![VerificationMethod {
                id: key_id.clone(),
                method_type: "Ed25519VerificationKey2020".to_string(),
                public_key_multibase: format!("z{pk_hex}"),
            }],
            authentication: vec![key_id],
            service: Vec::new(),
        };
        if let Some(ref ep) = self.service_endpoint {
            doc.service.push(DidService {
                id: format!("{}#unfer", self.did),
                service_type: "UnferKernelEndpoint".to_string(),
                service_endpoint: ep.clone(),
            });
        }
        doc
    }
}

mod hex_bytes_32 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 32 bytes"))
    }
}

mod hex_bytes_64 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 64 bytes"))
    }
}

#[cfg(test)]
mod mint_request_tests {
    // Plan R Phase 3: the oracle-backed mint contract shared by the zk-TLS
    // client and the mint authority.
    use super::*;
    use crate::codes::Code;

    fn req(source: &str) -> MintRequest {
        MintRequest {
            owner: "did:unfer:alice".to_string(),
            amount: 15,
            source: source.to_string(),
            blinding: None,
        }
    }

    #[test]
    fn valid_unfccc_source_accepted() {
        assert_eq!(req("unfccc:vc:34791").validate_source(), Ok(()));
        assert_eq!(req("unfccc:vc:MLT-0001").validate_source(), Ok(()));
    }

    #[test]
    fn malformed_source_rejected() {
        let bad = [
            "",                   // empty
            "unfccc:vc:",         // no order id
            "unfccc:cert:123",    // old format
            "https://unfccc.int", // not a vc reference
            "unfccc:vc:12 3",     // space
            "unfccc:vc:abc/def",  // slash
        ];
        for s in bad {
            assert_eq!(
                req(s).validate_source(),
                Err(Code::CERT_ORACLE_REJECTED),
                "source {:?} must be rejected",
                s
            );
        }
    }

    #[test]
    fn effective_blinding_reproducible_without_field() {
        let a = req("unfccc:vc:34791");
        let b = req("unfccc:vc:34791");
        assert_eq!(a.effective_blinding(), b.effective_blinding());
        // Different sources → different blinding (no cross-request collision).
        assert_ne!(
            a.effective_blinding(),
            req("unfccc:vc:34792").effective_blinding()
        );
        // An explicit blinding wins over the derived one.
        let explicit = MintRequest {
            blinding: Some([7u8; 32]),
            ..a
        };
        assert_eq!(explicit.effective_blinding(), [7u8; 32]);
    }

    #[test]
    fn to_mint_kind_roundtrip_serde() {
        let r = req("unfccc:vc:34791");
        let kind = r.to_mint_kind();
        match kind {
            CertificateOpKind::Mint {
                amount,
                owner,
                source,
                ..
            } => {
                assert_eq!(amount, 15);
                assert_eq!(owner, "did:unfer:alice");
                assert_eq!(source.as_deref(), Some("unfccc:vc:34791"));
            }
            other => panic!("expected Mint, got {other:?}"),
        }
        // The request itself is JSON-serializable (the wire contract).
        let json = serde_json::to_string(&r).unwrap();
        let back: MintRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }
}

#[cfg(test)]
mod kernel_event_tests {
    use super::KernelEvent;

    #[test]
    fn evolved_event_round_trips_through_kernel_event() {
        let ev = KernelEvent::Evolved {
            t: 0.25,
            norm: 1.0,
            solve_ms: 12,
        };
        let json = serde_json::to_value(&ev).unwrap();
        // Canonical wire shape (internally tagged `type = evolved`).
        assert_eq!(json["type"], "evolved");
        assert_eq!(json["t"], 0.25);
        assert_eq!(json["norm"], 1.0);
        assert_eq!(json["solve_ms"], 12);

        // The kernel_client agent historically appended a `components` extra
        // field. Deserializing must tolerate the extra field while re-serializing
        // to the canonical KernelEvent form (extra fields are dropped).
        let mut with_extra = json.clone();
        with_extra
            .as_object_mut()
            .unwrap()
            .insert("components".to_string(), serde_json::json!([[0.1], [0.2]]));
        let back: KernelEvent = serde_json::from_value(with_extra).unwrap();
        assert_eq!(back, ev);
        let re = serde_json::to_value(&back).unwrap();
        assert_eq!(re, json);
    }

    #[test]
    fn prior_set_and_conditioned_round_trip() {
        let cases = vec![
            (serde_json::to_value(KernelEvent::PriorSet).unwrap(), true),
            (
                serde_json::to_value(KernelEvent::Conditioned {
                    prior_probability: 0.7,
                })
                .unwrap(),
                true,
            ),
        ];
        for (json, _) in cases {
            let back: KernelEvent = serde_json::from_value(json.clone()).unwrap();
            assert_eq!(serde_json::to_value(&back).unwrap(), json);
        }
    }
}
