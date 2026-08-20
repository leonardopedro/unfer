//! H8: module archetype harness registry + runtime-choice resolver.
//!
//! Consolidates `modhost`'s bespoke branching over the module archetypes
//! (Austral cells / `austral_cps`, workerd ECMAScript, Tidepool Haskell
//! effects, cap-std Rust, plus a degenerate kernelless profile that only reads
//! `snapshot`/`probability`) into one registered resolver. No fourth plugin
//! mechanism is added — each archetype registers an adapter [`HarnessProfile`]
//! and [`resolve_runtime_choice`] selects among the existing slots.

/// A module runtime harness profile: how the module is hosted, how it talks to
/// the kernel (loopback / JIT / effect-handlers), and what it can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HarnessProfile {
    pub id: &'static str,
    /// Transport the control path uses to reach the kernel (loopback chokepoint
    /// for sidecar archetypes; CPS JIT for compiled Austral binaries).
    pub control_transport: &'static str,
    /// Transport the tool/effect path uses (kernel loopback, per-effect
    /// handler stack, or none).
    pub tool_transport: &'static str,
    /// Transcript format exchanged with the kernel (`ndjson`, `snapshot`, …).
    pub transcript_format: &'static str,
    /// Capability set this profile may request (kernel `uk_*` families).
    pub capabilities: &'static [&'static str],
}

/// The registered archetype adapter table (one row per plugin slot).
pub const HARNESS_PROFILES: &[HarnessProfile] = &[
    HarnessProfile {
        id: "austral_cps",
        control_transport: "jit",
        tool_transport: "loopback",
        transcript_format: "ndjson",
        capabilities: &["kernel"],
    },
    HarnessProfile {
        id: "capstd",
        control_transport: "cranelift",
        tool_transport: "loopback",
        transcript_format: "ndjson",
        capabilities: &["kernel"],
    },
    HarnessProfile {
        id: "ecmascript",
        control_transport: "workerd",
        tool_transport: "loopback",
        transcript_format: "ndjson",
        capabilities: &["kernel"],
    },
    // Degenerate kernelless profile: only reads the existing
    // `snapshot`/`probability` (no kernel capability, no tool transport).
    HarnessProfile {
        id: "kernelless",
        control_transport: "none",
        tool_transport: "none",
        transcript_format: "snapshot",
        capabilities: &[],
    },
    HarnessProfile {
        id: "tidepool",
        control_transport: "workerd",
        tool_transport: "effect-handlers",
        transcript_format: "ndjson",
        capabilities: &["kernel", "effects"],
    },
];

/// Look up a registered profile by id.
pub fn profile(id: &str) -> Option<&'static HarnessProfile> {
    HARNESS_PROFILES.iter().find(|p| p.id == id)
}

/// The outcome of resolving which archetype runs a module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeChoice {
    /// The resolved archetype id (always one of [`HARNESS_PROFILES`]).
    Profile(&'static str),
    /// Non-retryable rejection: the requested archetype is not approved.
    Rejected { requested: String },
}

/// Resolve the runtime choice for a module (qm `harness-router.ts`).
///
/// Candidate precedence (nearest-wins): scope override → module request →
/// fallback. The **org floor** is a constraint filter over the candidate: a
/// scope/request may only tighten toward the org's approved archetype, never
/// widen below it (a scope cannot pick an archetype the org floor excludes).
///
/// `approved` is the allowlist (module.toml `archetypes` / operator config); an
/// empty allowlist means "operator config decides" and any known profile is
/// acceptable. A candidate archetype that is not approved is rejected
/// non-retryably (UK-4001 family) rather than silently falling back.
pub fn resolve_runtime_choice(
    approved: &[&str],
    org: Option<&str>,
    scope: Option<&str>,
    fallback: &str,
    requested: Option<&str>,
) -> RuntimeChoice {
    let org_allows = |candidate: &str| match org {
        None | Some("") => true,
        Some(floor) => floor == candidate,
    };
    let approved_known = |candidate: &str| profile(candidate).is_some();
    let is_approved = |candidate: &str| {
        (approved.is_empty() || approved.contains(&candidate)) && approved_known(candidate)
    };
    let effective = |candidate: &str| {
        is_approved(candidate)
            && org_allows(candidate)
    };

    // 1. Scope override (a scope may only pick an approved archetype at/above
    //    the org floor).
    if let Some(s) = scope {
        if !s.is_empty() && effective(s) {
            return RuntimeChoice::Profile(profile(s).expect("approved ⇒ known").id);
        }
        if !s.is_empty() && approved_known(s) {
            // A real archetype that is unapproved or below the org floor is
            // a hard request — reject, never silently downgrade.
            return RuntimeChoice::Rejected {
                requested: s.to_string(),
            };
        }
    }

    // 2. The module's own request. Unapproved / below-org → rejection.
    if let Some(r) = requested {
        if !r.is_empty() && effective(r) {
            return RuntimeChoice::Profile(profile(r).expect("approved ⇒ known").id);
        }
        if !r.is_empty() && approved_known(r) {
            return RuntimeChoice::Rejected {
                requested: r.to_string(),
            };
        }
    }

    // 3. Fallback when nothing above names an approved archetype.
    if effective(fallback) {
        RuntimeChoice::Profile(profile(fallback).expect("approved ⇒ known").id)
    } else {
        RuntimeChoice::Rejected {
            requested: fallback.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_profiles_are_distinct_and_known() {
        let ids: Vec<&str> = HARNESS_PROFILES.iter().map(|p| p.id).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted, "profiles must be sorted by id");
        assert_eq!(ids.len(), 5, "four archetypes + the kernelless degenerate");
        for id in ids {
            assert_eq!(profile(id).unwrap().id, id);
        }
    }

    #[test]
    fn scope_override_selects_approved_archetype() {
        // Per-scope archetype selection: the scope names a real archetype.
        let approved = ["austral_cps", "ecmascript"];
        assert_eq!(
            resolve_runtime_choice(&approved, None, Some("ecmascript"), "austral_cps", Some("austral_cps")),
            RuntimeChoice::Profile("ecmascript"),
            "scope override wins over the module request"
        );
    }

    #[test]
    fn unapproved_requested_is_rejected_non_retryably() {
        // UK-4001 family: a requested archetype not on the operator allowlist is
        // rejected, never silently downgraded to the fallback.
        let approved = ["austral_cps"];
        assert_eq!(
            resolve_runtime_choice(&approved, None, None, "austral_cps", Some("tidepool")),
            RuntimeChoice::Rejected { requested: "tidepool".to_string() }
        );
    }

    #[test]
    fn fallback_used_when_scope_names_nothing() {
        // A scope that names nothing falls back to the requested / fallback.
        let approved = ["austral_cps", "ecmascript"];
        assert_eq!(
            resolve_runtime_choice(&approved, None, None, "austral_cps", Some("ecmascript")),
            RuntimeChoice::Profile("ecmascript"),
            "requested wins when scope is silent"
        );
        assert_eq!(
            resolve_runtime_choice(&approved, None, Some(""), "austral_cps", None),
            RuntimeChoice::Profile("austral_cps"),
            "empty scope + no request → fallback"
        );
    }

    #[test]
    fn org_floor_is_an_upper_bound_on_scope() {
        // The org floor binds: a scope that names an archetype not at the org
        // floor is rejected (a scope can only tighten, never widen below org).
        let approved = ["austral_cps", "ecmascript"];
        let org = Some("austral_cps");
        assert_eq!(
            resolve_runtime_choice(&approved, org, Some("ecmascript"), "austral_cps", None),
            RuntimeChoice::Rejected { requested: "ecmascript".to_string() },
            "scope cannot widen below the org floor"
        );
        // A scope at the org floor resolves.
        assert_eq!(
            resolve_runtime_choice(&approved, org, Some("austral_cps"), "austral_cps", None),
            RuntimeChoice::Profile("austral_cps")
        );
    }

    #[test]
    fn empty_approved_means_operator_config_decides() {
        // No module.toml allowlist → any known profile is acceptable.
        let approved: [&str; 0] = [];
        assert_eq!(
            resolve_runtime_choice(&approved, None, None, "kernelless", Some("capstd")),
            RuntimeChoice::Profile("capstd")
        );
    }

    #[test]
    fn kernelless_profile_reads_snapshot_only() {
        let k = profile("kernelless").unwrap();
        assert_eq!(k.transcript_format, "snapshot");
        assert!(k.capabilities.is_empty(), "kernelless carries no kernel capability");
    }
}