//! H7: consensus/auction idempotency — duplicated or replayed delivery applies
//! exactly once.
//!
//! The ledgers (`CertificateLedger`, `AuctionLedger`) are deterministic: given
//! the same ordered log every node produces the same root. In a distributed
//! setting the *delivery* of a transaction is not guaranteed to be exactly-once
//! (a scheduler may double-fire, a relay may re-deliver). [`IdempotencyStore`]
//! guards every `CertificateOp` (Mint/Transfer/Burn) and `AuctionOp`
//! (Open/Bid/Close) at application: `once(key, f)` runs `f` only when `key` has
//! not been committed, so a duplicated or replayed transaction applies exactly
//! once. The store is rebuilt by replaying the same ordered log, so it is as
//! deterministic as the ledgers it guards (same log → same committed set →
//! same root).

use std::collections::HashMap;

use unfer_protocol::{Diagnostic, ConsensusTransaction};

/// Exactly-once application guard for replayed/duplicated consensus delivery.
///
/// `committed` maps an op's content key to the log sequence at which it was
/// applied. Retention: `prune_before(seq)` drops entries committed at an
/// earlier sequence (scheduled garbage collection).
#[derive(Debug, Clone, Default)]
pub struct IdempotencyStore {
    committed: HashMap<String, u64>,
}

impl IdempotencyStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `key` was already applied exactly-once.
    pub fn committed(&self, key: &str) -> bool {
        self.committed.contains_key(key)
    }

    /// Apply `f` exactly once for `key`. A duplicate/replay is a no-op that
    /// still reports success (the outcome was already produced); the caller's
    /// applied-seq watermark advances either way, mirroring a settled record.
    pub fn once(
        &mut self,
        key: &str,
        seq: u64,
        f: impl FnOnce() -> Result<(), Diagnostic>,
    ) -> Result<(), Diagnostic> {
        if self.committed.contains_key(key) {
            return Ok(());
        }
        f()?;
        self.committed.insert(key.to_string(), seq);
        Ok(())
    }

    /// Retention prune on a schedule: drop entries committed before `seq`
    /// (a replayed *old* delivery no longer needs its guard — the log has
    /// already been replayed past it). Keeps the store bounded.
    pub fn prune_before(&mut self, seq: u64) {
        self.committed.retain(|_, at| *at >= seq);
    }

    /// The number of guarded keys (test/QA surface).
    pub fn len(&self) -> usize {
        self.committed.len()
    }

    /// Whether no guards are held (test/QA surface).
    pub fn is_empty(&self) -> bool {
        self.committed.is_empty()
    }
}

/// Content key for a certificate op (Plan R): the acting DID plus the op kind
/// *without* the per-delivery `seq`/`signature`, so re-submitting the same
/// logical transfer maps to the same key and applies once.
pub fn certificate_key(op: &unfer_protocol::CertificateOp) -> String {
    let kind = serde_json::to_string(&op.kind).unwrap_or_default();
    format!("cert:{}:{}", op.did, kind)
}

/// Content key for an auction op: the acting DID plus the op kind, again
/// excluding the delivery-scoped `seq`/`signature`.
pub fn auction_key(op: &unfer_protocol::AuctionOp) -> String {
    let kind = serde_json::to_string(&op.kind).unwrap_or_default();
    format!("auction:{}:{}", op.did, kind)
}

/// The idempotency key for a consensus transaction, if it is a guarded op.
pub fn transaction_key(tx: &ConsensusTransaction) -> Option<String> {
    match tx {
        ConsensusTransaction::CertificateOp(op) => Some(certificate_key(op)),
        ConsensusTransaction::AuctionOp(op) => Some(auction_key(op)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn once_runs_f_only_on_first_delivery() {
        let mut store = IdempotencyStore::new();
        let mut runs = 0;
        store
            .once("k", 1, || {
                runs += 1;
                Ok(())
            })
            .unwrap();
        store
            .once("k", 2, || {
                runs += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(runs, 1, "duplicate delivery must not re-run f");
        assert!(store.committed("k"));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn failed_f_is_not_committed() {
        let mut store = IdempotencyStore::new();
        let r = store.once("k", 1, || {
            Err(Diagnostic::new(
                unfer_protocol::Code::CERT_AMOUNT_MISMATCH,
                "rejected",
                unfer_protocol::Severity::Error,
            ))
        });
        assert!(r.is_err());
        assert!(!store.committed("k"), "a failed apply must stay uncommitted");
    }

    #[test]
    fn prune_before_keeps_recent_guards() {
        let mut store = IdempotencyStore::new();
        store.once("old", 1, || Ok(())).unwrap();
        store.once("new", 2, || Ok(())).unwrap();
        store.prune_before(2);
        assert!(!store.committed("old"), "old guard pruned");
        assert!(store.committed("new"), "recent guard retained");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn identical_transfer_bytes_map_to_one_key() {
        use unfer_protocol::{CertificateOp, CertificateOpKind};
        let mk = || ConsensusTransaction::CertificateOp(CertificateOp {
            did: "did:unfer:bob".to_string(),
            kind: CertificateOpKind::Burn {
                inputs: vec![unfer_protocol::CoinRef {
                    coin_id: unfer_protocol::CertId([7u8; 32]),
                    amount: 100,
                    owner: "did:unfer:bob".to_string(),
                }],
            },
            seq: 1,
            signature: [0u8; 64],
        });
        let a = transaction_key(&mk()).unwrap();
        let b = transaction_key(&mk()).unwrap();
        assert_eq!(a, b, "same logical op must share one idempotency key");
    }
}