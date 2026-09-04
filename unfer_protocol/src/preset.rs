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
