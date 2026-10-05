//! H10: named GrantSet presets (dsh `agent-presets`).
//!
//! `AgentPreset` is a *named reuse* of the existing `GrantSet` and symbol
//! vocabulary — no new permission. A preset is discovered unmemoized from a
//! roster directory; resolution merges `agent → preset → global` nearest-wins
//! (mirror dsh-scope), and switching presets is valid only while the session
//! has produced nothing (a blank session). The switch is a logged event
//! reconstructable from the H3 event log.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::GrantSet;

/// A named, reusable composition of grants + tool symbols + sections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentPreset {
    pub id: String,
    /// Trust tier label (advisory; e.g. `read-only` / `interactive` /
    /// `automation`). Not a security boundary — the grants are.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust: Option<String>,
    /// The grants this preset contributes (nearest-wins merges these).
    pub grants: GrantSet,
    /// Tool symbols (`uk_*`) this preset enables.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Named sections (advisory UI grouping; e.g. `["overview","graph"]`).
    #[serde(default)]
    pub sections: Vec<String>,
}

impl AgentPreset {
    /// Load a preset from a JSON roster file. `Err` carries the human-readable
    /// reason the preset is broken (a broken preset is *listed with its reason*,
    /// never skipped silently).
    pub fn from_json(s: &str, id: &str) -> Result<Self, String> {
        let mut preset: AgentPreset =
            serde_json::from_str(s).map_err(|e| format!("preset '{id}' is not valid JSON: {e}"))?;
        if preset.id.is_empty() {
            preset.id = id.to_string();
        }
        if preset.id != id {
            return Err(format!(
                "preset '{id}' declares id '{}' (roster key mismatch)",
                preset.id
            ));
        }
        Ok(preset)
    }
}

/// A roster discovery result: one entry per candidate file, so a broken preset
/// is surfaced with its reason rather than skipped silently.
#[derive(Debug, Clone, PartialEq)]
pub struct RosterEntry {
    /// The preset id this candidate claims (file stem, or the parsed id).
    pub id: String,
    /// `Some(preset)` when the file parsed cleanly, `None` + `reason` when broken.
    pub preset: Option<AgentPreset>,
    pub reason: Option<String>,
}

/// Discover presets from a roster directory, unmemoized. Each `*.json` file is
/// parsed; a broken file yields a [`RosterEntry`] with its reason (never
/// silently skipped).
pub fn discover_roster(dir: &Path) -> Vec<RosterEntry> {
    let mut entries = Vec::new();
    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            return vec![RosterEntry {
                id: format!("<roster {dir:?}>"),
                preset: None,
                reason: Some(format!("cannot read roster dir: {e}")),
            }];
        }
    };
    let mut names: Vec<String> = read_dir
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();
    for name in names {
        let id = name.trim_end_matches(".json").to_string();
        let path = dir.join(name);
        match std::fs::read_to_string(&path) {
            Ok(body) => match AgentPreset::from_json(&body, &id) {
                Ok(p) => entries.push(RosterEntry {
                    id,
                    preset: Some(p),
                    reason: None,
                }),
                Err(reason) => entries.push(RosterEntry {
                    id,
                    preset: None,
                    reason: Some(reason),
                }),
            },
            Err(e) => entries.push(RosterEntry {
                id,
                preset: None,
                reason: Some(format!("cannot read preset: {e}")),
            }),
        }
    }
    entries
}

/// Resolve the effective grant set for a caller: `agent → preset → global`
/// nearest-wins. The nearest non-empty scope wins per field (kernel, effects,
/// observers, resources, tools). Mirrors dsh-scope.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResolvedPreset {
    pub grants: GrantSet,
    pub tools: Vec<String>,
    pub sections: Vec<String>,
    /// The preset id that supplied the winning grants, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_preset: Option<String>,
}

/// Nearest-wins resolution over `agent → preset → global`.
///
/// - `agent`: the caller's own overrides (highest precedence).
/// - `preset`: the named preset's grants.
/// - `global`: deployment-wide defaults (lowest precedence).
///
/// A field is taken from the *first* (nearest) scope that specifies it. When a
/// preset name is unknown, resolution falls through to global (nearest-wins
/// treats it as "no preset specified") rather than failing the call.
pub fn resolve_preset_chain(
    global: &GrantSet,
    preset: Option<&AgentPreset>,
    agent: Option<&GrantSet>,
) -> ResolvedPreset {
    fn pick<T: Clone>(a: Option<&Vec<T>>, p: Option<&Vec<T>>, g: &[T]) -> Vec<T> {
        a.filter(|v| !v.is_empty())
            .or_else(|| p.filter(|v| !v.is_empty()))
            .cloned()
            .unwrap_or_else(|| g.to_vec())
    }
    let mut grants = GrantSet {
        kernel: pick(
            agent.map(|g| &g.kernel),
            preset.map(|p| &p.grants.kernel),
            &global.kernel,
        ),
        effects: pick(
            agent.map(|g| &g.effects),
            preset.map(|p| &p.grants.effects),
            &global.effects,
        ),
        observers: pick(
            agent.map(|g| &g.observers),
            preset.map(|p| &p.grants.observers),
            &global.observers,
        ),
        resources: pick(
            agent.map(|g| &g.resources),
            preset.map(|p| &p.grants.resources),
            &global.resources,
        ),
        effect_kinds: pick(
            agent.map(|g| &g.effect_kinds),
            preset.map(|p| &p.grants.effect_kinds),
            &global.effect_kinds,
        ),
    };
    grants
        .effect_kinds
        .retain(|eg| grants.effects.contains(&eg.name));
    let tools = preset.map(|p| p.tools.clone()).unwrap_or_default();
    let sections = preset.map(|p| p.sections.clone()).unwrap_or_default();
    ResolvedPreset {
        grants,
        tools,
        sections,
        source_preset: preset.map(|p| p.id.clone()),
    }
}

/// Whether a preset switch is valid for a session that has produced `ops`
/// records. Valid only while the session is blank (no prior work); switching
/// mid-session would silently change the tool surface under a model that has
/// already run.
pub fn switch_valid_when_blank(produced_ops: usize) -> bool {
    produced_ops == 0
}

/// A roster as a lookup map: preset id → preset. Broken entries are excluded
/// but still reported (the caller lists them with their reason).
#[derive(Debug, Clone, Default)]
pub struct Roster {
    presets: BTreeMap<String, AgentPreset>,
    broken: Vec<RosterEntry>,
}

impl Roster {
    pub fn from_entries(entries: Vec<RosterEntry>) -> Self {
        let mut presets = BTreeMap::new();
        let mut broken = Vec::new();
        for e in entries {
            match e.preset {
                Some(p) => {
                    presets.insert(e.id.clone(), p);
                }
                None => broken.push(e),
            }
        }
        Self { presets, broken }
    }

    pub fn get(&self, id: &str) -> Option<&AgentPreset> {
        self.presets.get(id)
    }

    /// The broken presets with their reasons (never skipped silently).
    pub fn broken(&self) -> &[RosterEntry] {
        &self.broken
    }

    pub fn ids(&self) -> Vec<&str> {
        self.presets.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EffectKind;

    fn g(kernel: &[&str]) -> GrantSet {
        GrantSet {
            kernel: kernel.iter().map(|s| s.to_string()).collect(),
            effects: vec![],
            observers: vec![],
            resources: vec![],
            effect_kinds: vec![],
        }
    }

    #[test]
    fn preset_roundtrips_json() {
        let json = r#"{
            "id": "analyst",
            "trust": "read-only",
            "grants": { "kernel": ["uk_evolve", "uk_probability"] },
            "tools": ["uk_probability", "uk_condition"],
            "sections": ["overview", "graph"]
        }"#;
        let p = AgentPreset::from_json(json, "analyst").unwrap();
        assert_eq!(p.trust.as_deref(), Some("read-only"));
        assert_eq!(p.grants.kernel, vec!["uk_evolve", "uk_probability"]);
        assert_eq!(p.tools, vec!["uk_probability", "uk_condition"]);
        assert_eq!(p.sections, vec!["overview", "graph"]);
    }

    #[test]
    fn broken_preset_surfaces_reason() {
        // Malformed JSON → the roster entry carries the reason (never silent).
        let err = AgentPreset::from_json("not json", "broken").unwrap_err();
        assert!(err.contains("broken"), "reason names the preset: {err}");
        // Roster-key mismatch.
        let json = r#"{"id":"other","grants":{}}"#;
        let err = AgentPreset::from_json(json, "analyst").unwrap_err();
        assert!(err.contains("mismatch"), "{err}");
    }

    #[test]
    fn nearest_wins_agent_over_preset_over_global() {
        let global = g(&["uk_version", "uk_snapshot"]);
        let preset = AgentPreset {
            id: "analyst".into(),
            trust: None,
            grants: g(&["uk_evolve", "uk_probability", "uk_version"]),
            tools: vec!["uk_condition".into()],
            sections: vec!["graph".into()],
        };
        let agent = g(&["uk_evolve"]);

        // Agent overrides preset over global per-field.
        let r = resolve_preset_chain(&global, Some(&preset), Some(&agent));
        assert_eq!(r.grants.kernel, vec!["uk_evolve"]);
        assert_eq!(r.source_preset.as_deref(), Some("analyst"));
        assert_eq!(r.tools, vec!["uk_condition"], "tools come from the preset");

        // No agent → preset wins.
        let r = resolve_preset_chain(&global, Some(&preset), None);
        assert_eq!(
            r.grants.kernel,
            vec!["uk_evolve", "uk_probability", "uk_version"]
        );
        // No preset → global wins.
        let r = resolve_preset_chain(&global, None, None);
        assert_eq!(r.grants.kernel, vec!["uk_version", "uk_snapshot"]);
    }

    #[test]
    fn switch_valid_only_when_blank() {
        assert!(switch_valid_when_blank(0));
        assert!(!switch_valid_when_blank(1));
        assert!(!switch_valid_when_blank(7));
    }

    #[test]
    fn roster_lists_broken_entries_with_reason() {
        let dir = std::env::temp_dir().join(format!(
            "unfer-h10-roster-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("analyst.json"),
            r#"{"id":"analyst","grants":{"kernel":["uk_evolve"]}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("broken.json"), "not json").unwrap();

        let roster = Roster::from_entries(discover_roster(&dir));
        assert_eq!(roster.ids(), vec!["analyst"]);
        assert_eq!(roster.broken().len(), 1);
        assert_eq!(roster.broken()[0].id, "broken");
        assert!(
            roster.broken()[0]
                .reason
                .as_deref()
                .unwrap()
                .contains("broken"),
            "reason surfaced: {:?}",
            roster.broken()[0].reason
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn effect_kinds_pruned_to_resolved_effects() {
        let global = GrantSet {
            kernel: vec!["uk_action_submit".into()],
            effects: vec!["email".into()],
            observers: vec![],
            resources: vec![],
            effect_kinds: vec![crate::EffectGrant {
                name: "email".into(),
                effect_kind: EffectKind::Observe,
            }],
        };
        let r = resolve_preset_chain(&global, None, None);
        assert_eq!(r.grants.effects, vec!["email"]);
        assert_eq!(r.grants.effect_kinds.len(), 1);
        assert_eq!(r.grants.effect_kinds[0].effect_kind, EffectKind::Observe);
    }
}

// ---------------------------------------------------------------------------
// C5: role presets over the same GrantSet vocabulary
// ---------------------------------------------------------------------------
//
// `AgentPreset` above is the generic H10 mechanism: a named composition,
// discovered from a roster, resolved through a chain. What it does not give you
// is the *vocabulary* the multi-worker designs need -- a reviewer who cannot
// mutate, a director who has no tools at all -- expressed as data so the
// capability layer keeps enforcing it rather than convention.
//
// This is deliberately not a second permission system. A `RolePreset` is a
// `GrantSet` value, so `is_subset_of` applies unchanged and every S21 invariant
// (conservative `Mutate` default, no annotate-your-way-out) holds by
// construction. The only new thing is which grants each name carries, and the
// rule about which role may hold which preset.

/// A named least-privilege grant set for one kind of worker.
///
/// Ordered by breadth: every preset is a subset of [`RolePreset::Maintainer`],
/// and [`RolePreset::Orchestrator`] holds nothing at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RolePreset {
    /// Orchestrates, executes nothing. No kernel symbols, no effects: the
    /// director cannot act even if it decides to.
    Orchestrator,
    /// Reads status and results. No side-effecting symbol.
    Reader,
    /// Reader plus the verification and compile-to-normal-form surface.
    ProverRunner,
    /// Reader plus inspection and reporting.
    Reviewer,
    /// Reviewer plus the mutating blueprint/session surface.
    Integrator,
    /// Everything. The trusted-harness set; never a worker's default.
    Maintainer,
}

/// The role a process is running as, which bounds the presets it may hold.
///
/// This is the C5 misuse rule. A worker holding `Maintainer` is refused at the
/// loopback with an audit entry rather than trusted, because "the orchestrator
/// gave this worker broad grants" is exactly the escalation the grant lattice
/// exists to make visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentRole {
    /// Plans and delegates. May hold `Orchestrator`, and the readers it needs to
    /// see what it is delegating.
    Director,
    /// Executes. Any preset except `Maintainer`.
    Worker,
    /// Reviews output. `Reader`, `Reviewer`, `ProverRunner`.
    Reviewer,
}

impl RolePreset {
    pub const ALL: &'static [RolePreset] = &[
        RolePreset::Orchestrator,
        RolePreset::Reader,
        RolePreset::ProverRunner,
        RolePreset::Reviewer,
        RolePreset::Integrator,
        RolePreset::Maintainer,
    ];

    pub fn name(self) -> &'static str {
        match self {
            RolePreset::Orchestrator => "orchestrator",
            RolePreset::Reader => "reader",
            RolePreset::ProverRunner => "prover-runner",
            RolePreset::Reviewer => "reviewer",
            RolePreset::Integrator => "integrator",
            RolePreset::Maintainer => "maintainer",
        }
    }

    pub fn from_name(name: &str) -> Option<RolePreset> {
        RolePreset::ALL.iter().find(|p| p.name() == name).copied()
    }

    /// The grants this preset carries.
    ///
    /// No preset declares `effect_kinds`. That is the point: an unannotated
    /// effect resolves to `Mutate` and so still queues for approval, whereas an
    /// `observe` annotation would let it apply immediately. Leaving the
    /// annotations off means a preset can never hand a worker a bypass -- there
    /// is nothing in this table to relabel.
    pub fn grants(self) -> GrantSet {
        let kernel: &[&str] = match self {
            RolePreset::Orchestrator => &[],
            RolePreset::Reader => &[
                "uk_version",
                "uk_last_error",
                "uk_get_result",
                "uk_meter_status",
                "uk_registry_vetted",
                // C2: Observe-kind, so it needs no approval lane and every role
                // below Maintainer can consult what it already remembers. Note
                // `uk_memory_append` is deliberately *not* here: writing memory is
                // Mutate-kind under S21, because memory is model-visible state
                // that outlives the call. An unattended loop that wants to
                // summarise its own segments needs a vetted grant for that, which
                // is the intended posture rather than an oversight.
                "uk_memory_read",
            ],
            RolePreset::ProverRunner => &[
                "uk_version",
                "uk_last_error",
                "uk_get_result",
                "uk_meter_status",
                "uk_registry_vetted",
                "uk_memory_read",
                "uk_proof_verify",
                "uk_logos_compile",
                "uk_austral_unf",
                "uk_symbolic_simplify",
            ],
            RolePreset::Reviewer => &[
                "uk_version",
                "uk_last_error",
                "uk_get_result",
                "uk_meter_status",
                "uk_registry_vetted",
                "uk_memory_read",
                "uk_blueprint_list",
                "uk_blueprint_export",
                "uk_report_issue",
            ],
            RolePreset::Integrator => &[
                "uk_version",
                "uk_last_error",
                "uk_get_result",
                "uk_meter_status",
                "uk_registry_vetted",
                "uk_memory_read",
                "uk_blueprint_list",
                "uk_blueprint_export",
                "uk_report_issue",
                "uk_blueprint_import",
                "uk_session_fork",
            ],
            // The trusted harness: every kernel symbol the registry defines, which
            // is how `GrantSet` is built elsewhere for an unconstrained caller.
            RolePreset::Maintainer => &[],
        };
        if matches!(self, RolePreset::Maintainer) {
            return GrantSet {
                kernel: crate::symbols::SYMBOL_REGISTRY
                    .iter()
                    .map(|r| r.name.to_string())
                    .collect(),
                ..GrantSet::default()
            };
        }
        GrantSet::kernel(kernel)
    }

    /// May a process running as `role` hold this preset?
    ///
    /// The refusal is `Maintainer` for every non-trusted role. It is the one
    /// preset that is not least-privilege, so handing it to a worker is either
    /// a mistake or an escalation, and both deserve a loud refusal.
    pub fn permitted_for(self, role: AgentRole) -> bool {
        match role {
            AgentRole::Director => !matches!(self, RolePreset::Maintainer),
            AgentRole::Worker => !matches!(self, RolePreset::Maintainer),
            AgentRole::Reviewer => matches!(
                self,
                RolePreset::Orchestrator
                    | RolePreset::Reader
                    | RolePreset::Reviewer
                    | RolePreset::ProverRunner
            ),
        }
    }
}

impl AgentRole {
    pub fn name(self) -> &'static str {
        match self {
            AgentRole::Director => "director",
            AgentRole::Worker => "worker",
            AgentRole::Reviewer => "reviewer",
        }
    }
}

#[cfg(test)]
mod role_preset_tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn every_preset_is_a_subset_of_maintainer() {
        // Least privilege, stated as a property rather than as an intention: if a
        // new preset is added that reaches outside the harness set, this fails.
        let harness = RolePreset::Maintainer.grants();
        for preset in RolePreset::ALL.iter().copied() {
            assert!(
                preset.grants().is_subset_of(&harness),
                "{} is not a subset of the maintainer set, so it is not least-privilege",
                preset.name()
            );
        }
    }

    #[test]
    fn the_preset_relation_is_a_diamond_not_a_ladder() {
        // These are roles, not rungs. A reviewer and a prover-runner are both
        // readers and neither contains the other, so a "monotonically widening
        // ladder" is a false property -- asserting it was wrong, and asserting
        // it is how the shape gets misdocumented afterwards.
        //
        // What actually holds: both specialisations extend Reader, Integrator
        // extends Reviewer, and everything sits under Maintainer.
        let reader = RolePreset::Reader.grants();
        for specialisation in [RolePreset::ProverRunner, RolePreset::Reviewer] {
            assert!(
                reader.is_subset_of(&specialisation.grants()),
                "{} must extend reader",
                specialisation.name()
            );
            assert!(
                !specialisation.grants().is_subset_of(&reader),
                "{} adds nothing over reader, so it is not a distinct role",
                specialisation.name()
            );
        }
        assert!(
            RolePreset::Reviewer
                .grants()
                .is_subset_of(&RolePreset::Integrator.grants()),
            "integrator must extend reviewer"
        );
        assert!(
            !RolePreset::ProverRunner
                .grants()
                .is_subset_of(&RolePreset::Integrator.grants()),
            "a prover-runner is not an integrator; the two branches stay distinct"
        );
    }

    #[test]
    fn the_orchestrator_has_no_tools_at_all() {
        // "Orchestrates but has no tools" is only meaningful if it is enforced.
        let grants = RolePreset::Orchestrator.grants();
        assert!(
            grants.kernel.is_empty(),
            "director must hold no kernel symbols"
        );
        assert!(grants.effects.is_empty(), "director must hold no effects");
        assert!(grants.resources.is_empty());
        assert!(grants.observers.is_empty());
    }

    #[test]
    fn no_preset_declares_an_effect_annotation() {
        // S21: an `observe` annotation makes an effect apply immediately
        // instead of queueing for approval. A preset that carried one would be a
        // bypass handed out by name, so none may declare any.
        for preset in RolePreset::ALL.iter().copied() {
            assert!(
                preset.grants().effect_kinds.is_empty(),
                "{} declares effect annotations; every unannotated effect already                  defaults to Mutate and queues, so an annotation here can only                  weaken the approval lane",
                preset.name()
            );
        }
    }

    #[test]
    fn readers_and_reviewers_cannot_reach_a_mutating_symbol() {
        // The grants that change state, kept out of the read-side presets.
        let mutating = [
            "uk_blueprint_import",
            "uk_session_fork",
            "uk_session_compact",
        ];
        for preset in [
            RolePreset::Reader,
            RolePreset::Reviewer,
            RolePreset::ProverRunner,
            RolePreset::Orchestrator,
        ] {
            let held = preset.grants();
            for symbol in mutating {
                assert!(
                    !held.kernel.iter().any(|k| k == symbol),
                    "{} holds the mutating symbol {symbol}",
                    preset.name()
                );
            }
        }
        // And the integrator, which is the role that exists to hold them, does.
        assert!(
            RolePreset::Integrator
                .grants()
                .kernel
                .iter()
                .any(|k| k == "uk_blueprint_import")
        );
    }

    #[test]
    fn maintainer_covers_every_kernel_symbol_the_registry_defines() {
        let harness: BTreeSet<String> =
            RolePreset::Maintainer.grants().kernel.into_iter().collect();
        let registry: BTreeSet<String> = crate::symbols::SYMBOL_REGISTRY
            .iter()
            .map(|r| r.name.to_string())
            .collect();
        let missing: Vec<&String> = registry.difference(&harness).collect();
        assert!(
            missing.is_empty(),
            "maintainer is the trusted-harness set and must cover the registry;              missing {missing:?}"
        );
    }

    #[test]
    fn every_preset_symbol_exists_in_the_registry() {
        // A preset naming a symbol the registry does not define would fail at
        // mint time, far from the table that introduced it.
        let registry: BTreeSet<&str> = crate::symbols::SYMBOL_REGISTRY
            .iter()
            .map(|r| r.name)
            .collect();
        for preset in RolePreset::ALL.iter().copied() {
            for symbol in preset.grants().kernel {
                assert!(
                    registry.contains(symbol.as_str()),
                    "{} names {symbol}, which is not in the registry",
                    preset.name()
                );
            }
        }
    }

    #[test]
    fn names_round_trip_and_are_unique() {
        let mut seen = BTreeSet::new();
        for preset in RolePreset::ALL.iter().copied() {
            assert!(
                seen.insert(preset.name()),
                "duplicate preset name {}",
                preset.name()
            );
            assert_eq!(
                Some(preset),
                RolePreset::from_name(preset.name()),
                "{} does not round-trip through from_name",
                preset.name()
            );
        }
        assert_eq!(RolePreset::ALL.len(), seen.len());
        assert_eq!(None, RolePreset::from_name("no-such-preset"));
    }

    #[test]
    fn a_worker_or_director_cannot_hold_maintainer() {
        // The C5 misuse rule. Checked here as a pure predicate; the loopback
        // refuses on the same predicate and writes an audit entry.
        for role in [AgentRole::Worker, AgentRole::Director] {
            assert!(
                !RolePreset::Maintainer.permitted_for(role),
                "a {} was permitted to hold maintainer",
                role.name()
            );
            assert!(
                !RolePreset::Maintainer.permitted_for(AgentRole::Reviewer),
                "a reviewer was permitted to hold maintainer"
            );
        }
        // Every non-maintainer preset is available to an ordinary worker.
        for preset in RolePreset::ALL.iter().copied() {
            if preset == RolePreset::Maintainer {
                continue;
            }
            assert!(
                preset.permitted_for(AgentRole::Worker),
                "a worker was refused the ordinary preset {}",
                preset.name()
            );
        }
    }

    #[test]
    fn a_reviewer_cannot_hold_the_integrator_preset() {
        // A reviewer that can import blueprints and fork sessions is not a
        // reviewer. Read-only-by-role is the whole point of the role.
        assert!(!RolePreset::Integrator.permitted_for(AgentRole::Reviewer));
        assert!(RolePreset::Reviewer.permitted_for(AgentRole::Reviewer));
    }

    #[test]
    fn role_and_preset_names_are_distinct_within_their_kind() {
        // "reviewer" names both a preset and a role. They are separate vocabularies
        // over separate types, and the tests above rely on that; a reader of the
        // tables should not have to guess which is which.
        let presets: BTreeSet<&str> = RolePreset::ALL.iter().map(|p| p.name()).collect();
        let roles = ["director", "worker", "reviewer"];
        // `reviewer` legitimately appears in both; every other role name must not
        // collide with a preset name.
        for role in roles {
            if role == "reviewer" {
                continue;
            }
            assert!(
                !presets.contains(role),
                "role name {role} collides with a preset name"
            );
        }
    }
}
