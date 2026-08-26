//! Single source of truth for the `uk_*` / `uz_*` symbol census (H2).
//!
//! This table is the ONE place a symbol is registered. Every derived artifact
//! is generated from it by `scripts/gen_symbol_artifacts`:
//!
//! - `unfer_ffi/EXPECTED_SYMBOLS.txt` (kernel names)
//! - `unfer_ffi/EXPECTED_SYMBOLS_ZENODO.txt` (zenodo names)
//! - `unfer_ffi/include/unfer_kernel.h` (C header declarations)
//! - australVM's `UNFER_SYMBOLS` table (symbol_sync gate cross-checks it)
//! - the `GrantSet.kernel` namespace
//!
//! Adding a symbol means: add a row here, then run the generator. The H1
//! verify-invariants gate fails CI if the generated files drift from this
//! table. `effect_kind` is the *conservative* trust annotation for a
//! symbol-granted effect (Observe = read-only consult, Mutate = everything
//! else). `timeout_ms` is the declared cooperative deadline (H6); `None`
//! means the symbol declares no deadline and the guard never arms for it.

/// Which ABI surface a symbol belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// The core kernel surface (always linked, `uk_*`).
    Kernel,
    /// The zenodo store surface (built with `--features zenodo`, `uz_*`).
    Zenodo,
}

/// One registered symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolRecord {
    pub name: &'static str,
    pub kind: SymbolKind,
    /// Conservative trust annotation (S21). `Observe` = read-only consult
    /// (never queues for approval); `Mutate` = side-effecting (queued unless
    /// vetted). This is the DEFAULT when a grant carries no annotation.
    pub effect_kind: super::types::EffectKind,
    /// Declared cooperative deadline in ms (H6). `None` = no deadline.
    pub timeout_ms: Option<u64>,
}

/// The canonical symbol census. Sorted by name. Generated bootstrap; kept as
/// data (not code) so the generator can diff it byte-for-byte.
pub const SYMBOL_REGISTRY: &[SymbolRecord] = &[
    SymbolRecord { name: "uk_action_apply", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_action_get", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_action_list", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_action_reject", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_action_revert", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_action_submit", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_agent_grants", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_agent_kill", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_agent_list", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_agent_spawn", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_auction_bid", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_auction_close", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_auction_open", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_auction_report", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_audit_clear", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_audit_list", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_austral_unf", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_bayesian_update", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_belief_propagation", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_blueprint_cell", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_blueprint_export", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_blueprint_export_gadget", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_blueprint_get_by_id", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_blueprint_import", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_blueprint_instantiate", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_blueprint_list", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_buf_free", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
SymbolRecord { name: "uk_cert_burn", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_cert_mint", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_cert_mint_request", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_cert_root", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_cert_set_authority", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_cert_status", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_cert_transfer", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_certificate_issued", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_condition", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_durable_status", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_event_probability", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_evolve", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_gate_approve", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: Some(5000) },
    SymbolRecord { name: "uk_gate_list_pending", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_gate_reject", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_get_result", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_init", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_last_error", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_logos_compile", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_meter_status", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_model_create", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_model_free", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_observability", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_observe", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_ode_analyze", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_ode_measure_original", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_owner_clear", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_owner_list", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_owner_log", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_poll", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_posture_get", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_posture_set", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_proof_verify", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_registry_vetted", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_report_issue", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_request_resource", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_resource_forfeit", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_resource_introduce", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_resource_pending", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_resource_use", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_restore", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_secret_get", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_secret_put", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_secret_revoke", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_session_compact", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_session_fork", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_set_hamiltonian", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_set_prior", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_skill_get", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_skill_list", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_skill_pack_import", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_skill_register", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_snapshot", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_subscribe", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_symbolic_simplify", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uk_version", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uk_whyml_emit", kind: SymbolKind::Kernel, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uz_init", kind: SymbolKind::Zenodo, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uz_last_error", kind: SymbolKind::Zenodo, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uz_manifest_json", kind: SymbolKind::Zenodo, effect_kind: super::types::EffectKind::Observe, timeout_ms: None },
    SymbolRecord { name: "uz_pull", kind: SymbolKind::Zenodo, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
    SymbolRecord { name: "uz_push", kind: SymbolKind::Zenodo, effect_kind: super::types::EffectKind::Mutate, timeout_ms: None },
];

impl SymbolRecord {
    /// The kernel-namespace symbol names (`uk_*`), sorted — the
    /// `GrantSet.kernel` census and the `EXPECTED_SYMBOLS.txt` contents.
    pub fn kernel_names() -> Vec<&'static str> {
        SYMBOL_REGISTRY
            .iter()
            .filter(|r| r.kind == SymbolKind::Kernel)
            .map(|r| r.name)
            .collect()
    }

    /// The zenodo-namespace symbol names (`uz_*`), sorted.
    pub fn zenodo_names() -> Vec<&'static str> {
        SYMBOL_REGISTRY
            .iter()
            .filter(|r| r.kind == SymbolKind::Zenodo)
            .map(|r| r.name)
            .collect()
    }

    /// Look up a record by name (used by the loopback/H6 deadline guard).
    pub fn by_name(name: &str) -> Option<&'static SymbolRecord> {
        SYMBOL_REGISTRY.iter().find(|r| r.name == name)
    }

    /// The declared deadline for a symbol, if any (H6).
    pub fn timeout_ms(name: &str) -> Option<u64> {
        SYMBOL_REGISTRY
            .iter()
            .find(|r| r.name == name)
            .and_then(|r| r.timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registry_names_are_unique_and_sorted() {
        let names: Vec<_> = SYMBOL_REGISTRY.iter().map(|r| r.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "registry must be sorted by name");
        let set: HashSet<_> = names.iter().copied().collect();
        assert_eq!(set.len(), names.len(), "registry names must be unique");
    }

    #[test]
    fn kernel_and_zenodo_names_split_covers_registry() {
        let all: HashSet<_> = SYMBOL_REGISTRY.iter().map(|r| r.name).collect();
        let split: HashSet<_> = SymbolRecord::kernel_names()
            .into_iter()
            .chain(SymbolRecord::zenodo_names())
            .collect();
        assert_eq!(all, split, "kernel_names ∪ zenodo_names must cover the registry");
        let k: HashSet<_> = SymbolRecord::kernel_names().into_iter().collect();
        let z: HashSet<_> = SymbolRecord::zenodo_names().into_iter().collect();
        assert!(k.is_disjoint(&z), "kernel and zenodo names must not overlap");
    }

    #[test]
    fn kernel_names_are_all_uk_prefix() {
        for n in SymbolRecord::kernel_names() {
            assert!(n.starts_with("uk_"), "{n} must start with uk_");
        }
        for n in SymbolRecord::zenodo_names() {
            assert!(n.starts_with("uz_"), "{n} must start with uz_");
        }
    }

    #[test]
    fn by_name_and_timeout_lookup() {
        assert_eq!(SymbolRecord::by_name("uk_version").unwrap().kind, SymbolKind::Kernel);
        assert_eq!(SymbolRecord::by_name("uz_push").unwrap().kind, SymbolKind::Zenodo);
        assert!(SymbolRecord::by_name("uk_does_not_exist").is_none());
        assert_eq!(SymbolRecord::timeout_ms("uk_version"), None);
    }

    #[test]
    fn effect_kind_is_conservative() {
        // Read-only consult symbols are Observe; everything else defaults to
        // Mutate. Verify a few known-observes and known-mutators.
        assert_eq!(
            SymbolRecord::by_name("uk_meter_status").unwrap().effect_kind,
            super::super::types::EffectKind::Observe
        );
        assert_eq!(
            SymbolRecord::by_name("uk_evolve").unwrap().effect_kind,
            super::super::types::EffectKind::Mutate
        );
        assert_eq!(
            SymbolRecord::by_name("uk_snapshot").unwrap().effect_kind,
            super::super::types::EffectKind::Observe
        );
    }
}