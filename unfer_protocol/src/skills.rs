//! H13: skills registry — discovery/sharing over the existing module path.
//!
//! Modules *are* the project's skills (australVM is the plugin engine,
//! `module.toml` + `modhost` are the plugin slots, `uk_*` symbols the capability
//! surface). H13 adds only what the module path lacked: **discovery and
//! sharing**. A [`Skill`] is a discoverable, shareable, scope-owned reference to
//! an existing module (or a packed cell), reusable by grant without
//! re-authoring. No second plugin-loading mechanism: a skill's pack lands as a
//! `module.toml` cell loaded by the existing modhost.

use serde::{Deserialize, Serialize};

/// A discoverable, shareable module reference (the "skill" surface).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    /// The skill id (namespaced `org/skill`, e.g. `acme/carbon-audit`).
    pub id: String,
    /// The module it references (loaded by modhost; may be empty for a
    /// cell-only pack).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub module: String,
    /// Scope that owns the skill (`org`, `team`, or `personal:<principal>`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scope: String,
    /// Human description (the catalog panel renders this).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// The module's grants (shareable by grant — a caller must hold these to
    /// invoke the skill).
    #[serde(default)]
    pub grants: Vec<String>,
    /// Pack content: a `module.toml` cell body (loaded by the existing modhost)
    /// or a git pack reference. Git-importable packs land here.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub pack: String,
    /// Admin-gated promotion flag: `true` once an operator promoted the skill
    /// from personal/team scope to org scope (S22 admin seam).
    #[serde(default)]
    pub promoted: bool,
}

impl Skill {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            module: String::new(),
            scope: String::new(),
            description: String::new(),
            grants: Vec::new(),
            pack: String::new(),
            promoted: false,
        }
    }

    /// Whether a caller holding `caller_grants` may invoke this skill: the
    /// caller must hold every grant the skill requires (default-deny).
    pub fn caller_may_invoke(&self, caller_grants: &[String]) -> bool {
        self.grants.iter().all(|g| caller_grants.contains(g))
    }
}

/// The registry: a deterministic, scope-owned list of skills.
#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    skills: Vec<Skill>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or replace) a skill. Returns `false` when a skill with the
    /// same id already exists and is admin-promoted (promoted skills are
    /// immutable outside the admin seam).
    pub fn register(&mut self, skill: Skill) -> bool {
        if let Some(existing) = self.skills.iter_mut().find(|s| s.id == skill.id) {
            if existing.promoted && !skill.promoted {
                return false;
            }
            *existing = skill;
            return true;
        }
        self.skills.push(skill);
        true
    }

    pub fn get(&self, id: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.id == id)
    }

    /// List skills visible to a caller: org-scoped skills plus any skill the
    /// caller owns (`personal:<principal>`) or that requires no grants.
    pub fn list_visible(&self, principal: &str) -> Vec<&Skill> {
        let own = format!("personal:{principal}");
        self.skills
            .iter()
            .filter(|s| s.scope == "org" || s.scope == own || s.grants.is_empty())
            .collect()
    }

    /// Admin-gated promotion (S22 seam): move a skill to org scope.
    pub fn promote(&mut self, id: &str) -> bool {
        match self.skills.iter_mut().find(|s| s.id == id) {
            Some(s) => {
                s.scope = "org".to_string();
                s.promoted = true;
                true
            }
            None => false,
        }
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_list_visible_scope() {
        let mut reg = SkillRegistry::new();
        let mut org = Skill::new("acme/carbon-audit");
        org.scope = "org".to_string();
        let mut personal = Skill::new("me/scratch");
        personal.scope = "personal:alice".to_string();
        personal.grants = vec!["uk_snapshot".to_string()];
        let mut free = Skill::new("public/hello");
        free.grants = vec![];

        reg.register(org);
        reg.register(personal);
        reg.register(free);

        // Alice sees org + her own + grant-free.
        let alice: Vec<&str> = reg
            .list_visible("alice")
            .iter()
            .map(|s| s.id.as_str())
            .collect();
        assert!(alice.contains(&"acme/carbon-audit"));
        assert!(alice.contains(&"me/scratch"));
        assert!(alice.contains(&"public/hello"));
        // Bob does not see Alice's personal skill.
        let bob: Vec<&str> = reg
            .list_visible("bob")
            .iter()
            .map(|s| s.id.as_str())
            .collect();
        assert!(!bob.contains(&"me/scratch"));
    }

    #[test]
    fn caller_must_hold_all_required_grants() {
        let mut s = Skill::new("acme/carbon-audit");
        s.grants = vec!["uk_cert_mint".to_string(), "uk_cert_transfer".to_string()];
        assert!(s.caller_may_invoke(&["uk_cert_mint".to_string(), "uk_cert_transfer".to_string()]));
        assert!(!s.caller_may_invoke(&["uk_cert_mint".to_string()]));
        assert!(!s.caller_may_invoke(&[]));
    }

    #[test]
    fn promoted_skills_are_immutable_outside_admin_seam() {
        let mut reg = SkillRegistry::new();
        let s = Skill::new("acme/x");
        reg.register(s.clone());
        assert!(reg.promote("acme/x"));
        // A non-promoted re-register of a promoted skill is refused.
        assert!(!reg.register(Skill::new("acme/x")));
        // The admin seam can replace it.
        let mut promoted = Skill::new("acme/x");
        promoted.promoted = true;
        assert!(reg.register(promoted));
    }

    #[test]
    fn pack_roundtrips() {
        let mut s = Skill::new("acme/cell");
        s.pack = "[module]\nname = \"cell\"\n".to_string();
        let json = serde_json::to_string(&s).unwrap();
        let back: Skill = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "acme/cell");
        assert!(back.pack.contains("[module]"));
    }
}
