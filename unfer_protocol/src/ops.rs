//! Shared op-name registry for the `unfer_agent` protocol seam.
//!
//! Three historical registries drifted apart: the kernel_client agent
//! (`VALID_OPS`), the edge gateway allowlist (`ALLOWED_OPS`), and the
//! consensus `apply_session_op` dispatcher. This module is the single source
//! of truth; every consumer derives its slice from here so drift is caught by
//! `tests::consistency` (and by the cross-repo `symbol_sync` style gates).
//!
//! Invariants (enforced by `tests::registry_invariants`):
//! - every name is unique across all tables;
//! - `EDGE_ALLOWED_OPS ⊆ SESSION_OPS`;
//! - `CONSENSUS_OPS ⊆ SESSION_OPS`;
//! - `AGENT_OPS ∪ EDGE_ALLOWED_OPS = SESSION_OPS` (no orphan session op).

/// The complete, canonical catalog of ops the `unfer_agent` protocol can carry.
pub const SESSION_OPS: &[&str] = &[
    // kernel session ops
    "version",
    "create_model",
    "set_prior",
    "evolve",
    "condition",
    "probability",
    "observe",
    "snapshot",
    "bayesian_update",
    "belief_propagation",
    "list_codes",
    // identity + content namespace
    "did_create",
    "did_resolve",
    "did_update",
    "did_revoke",
    "content_publish",
    "content_resolve",
    // consensus + certificate ledger
    "consensus_sync",
    "consensus_status",
    "cert_set_authority",
    "cert_mint",
    "cert_transfer",
    "cert_burn",
    "cert_status",
    "cert_root",
    // agent-local ops (dispatched by the client, never forwarded to the edge)
    "save_session",
    "restore_session",
    "poll_events",
    "close_model",
    "logos_compile",
    "ode_to_hamiltonian",
    "export_html",
    "export_tex",
];

/// Ops the edge gateway accepts and forwards to the backend.
pub const EDGE_ALLOWED_OPS: &[&str] = &[
    "version",
    "create_model",
    "set_prior",
    "evolve",
    "condition",
    "probability",
    "observe",
    "snapshot",
    "list_codes",
    "did_create",
    "did_resolve",
    "did_update",
    "did_revoke",
    "content_publish",
    "content_resolve",
    "consensus_sync",
    "consensus_status",
    "cert_set_authority",
    "cert_mint",
    "cert_transfer",
    "cert_burn",
    "cert_status",
    "cert_root",
];

/// Ops the kernel_client agent dispatches locally. Deliberately excludes
/// `observe`: the client-side `Session` API has no observe method — it is an
/// edge/backend op accepted by the gateway but not dispatched by this client.
pub const AGENT_OPS: &[&str] = &[
    // kernel session ops
    "version",
    "create_model",
    "set_prior",
    "evolve",
    "condition",
    "probability",
    "snapshot",
    "bayesian_update",
    "belief_propagation",
    "list_codes",
    // identity + content namespace
    "did_create",
    "did_resolve",
    "did_update",
    "did_revoke",
    "content_publish",
    "content_resolve",
    // consensus + certificate ledger
    "consensus_sync",
    "consensus_status",
    "cert_set_authority",
    "cert_mint",
    "cert_transfer",
    "cert_burn",
    "cert_status",
    "cert_root",
    // agent-local ops
    "save_session",
    "restore_session",
    "poll_events",
    "close_model",
    "logos_compile",
    "ode_to_hamiltonian",
    "export_html",
    "export_tex",
];

/// Session ops the consensus node applies (multi-node merge support).
pub const CONSENSUS_OPS: &[&str] = &["create_model"];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn set<'a>(s: &[&'a str]) -> HashSet<&'a str> {
        s.iter().copied().collect()
    }

    #[test]
    fn no_duplicates_across_all_tables() {
        for table in [SESSION_OPS, EDGE_ALLOWED_OPS, AGENT_OPS, CONSENSUS_OPS] {
            let s = set(table);
            assert_eq!(
                s.len(),
                table.len(),
                "duplicate op names are not allowed in {:?}",
                table
            );
        }
    }

    #[test]
    fn edge_ops_are_a_subset_of_session_ops() {
        let session = set(SESSION_OPS);
        let extra: Vec<_> = EDGE_ALLOWED_OPS
            .iter()
            .filter(|op| !session.contains(**op))
            .collect();
        assert!(
            extra.is_empty(),
            "EDGE_ALLOWED_OPS contains ops missing from SESSION_OPS: {:?}",
            extra
        );
    }

    #[test]
    fn consensus_ops_are_a_subset_of_session_ops() {
        let session = set(SESSION_OPS);
        let extra: Vec<_> = CONSENSUS_OPS
            .iter()
            .filter(|op| !session.contains(**op))
            .collect();
        assert!(
            extra.is_empty(),
            "CONSENSUS_OPS contains ops missing from SESSION_OPS: {:?}",
            extra
        );
    }

    #[test]
    fn agent_ops_are_a_subset_of_session_ops() {
        let session = set(SESSION_OPS);
        let extra: Vec<_> = AGENT_OPS
            .iter()
            .filter(|op| !session.contains(**op))
            .collect();
        assert!(
            extra.is_empty(),
            "AGENT_OPS contains ops missing from SESSION_OPS: {:?}",
            extra
        );
    }

    #[test]
    fn session_ops_are_the_union_of_agent_and_edge_ops() {
        let union: HashSet<_> = AGENT_OPS.iter().chain(EDGE_ALLOWED_OPS).copied().collect();
        let session = set(SESSION_OPS);
        let orphan: Vec<_> = session.difference(&union).collect();
        assert!(
            orphan.is_empty(),
            "SESSION_OPS has ops in no consumer table: {:?}",
            orphan
        );
    }
}
