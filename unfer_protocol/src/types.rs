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
    Burn {
        inputs: Vec<CoinRef>,
    },
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
        if order_id.chars().any(|c| !c.is_ascii_alphanumeric() && c != '-') {
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
}

impl ConsensusTransaction {
    pub fn did(&self) -> &str {
        match self {
            Self::IdentityOp(op) => &op.did,
            Self::SessionOp(op) => &op.did,
            Self::ContentOp(op) => &op.did,
            Self::CertificateOp(op) => &op.did,
        }
    }

    pub fn signature(&self) -> &[u8; 64] {
        match self {
            Self::IdentityOp(op) => &op.signature,
            Self::SessionOp(op) => &op.signature,
            Self::ContentOp(op) => &op.signature,
            Self::CertificateOp(op) => &op.signature,
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
            "",                 // empty
            "unfccc:vc:",       // no order id
            "unfccc:cert:123",  // old format
            "https://unfccc.int", // not a vc reference
            "unfccc:vc:12 3",   // space
            "unfccc:vc:abc/def",// slash
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
        assert_ne!(a.effective_blinding(), req("unfccc:vc:34792").effective_blinding());
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
