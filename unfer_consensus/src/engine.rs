use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use unfer_protocol::submission::{Receipt, ReceiptState, Slot, SubmissionId, UnknownReason};
use unfer_protocol::{Code, ConsensusTransaction, Diagnostic, Severity};

/// The ordering seam.
///
/// `submit` returns a **submission id**, not a log position: on a local engine
/// the two coincide (submission is commitment), on a network engine the id is
/// reserved before consensus runs and may never be committed. `receipt` and
/// `wait` are how a caller learns what actually happened, and `get_log` is the
/// committed prefix a node applies.
pub trait ConsensusEngine: Send + Sync {
    fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic>;
    fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)>;
    fn current_seq(&self) -> u64;

    /// What this engine knows about a submission id. The default is the safe
    /// answer for an engine that keeps no receipt store: it cannot say.
    fn receipt(&self, _id: SubmissionId) -> ReceiptState {
        ReceiptState::Unknown(UnknownReason::Unsubmitted)
    }

    /// Bounded wait for a single submission. A deadline that passes returns
    /// [`UnknownReason::TimedOut`], never success and never a refusal -- the
    /// entry may still commit after the caller stops waiting.
    ///
    /// The clock reads below are a caller-side observation deadline, not a
    /// state transition: nothing in the log depends on when the caller gave up.
    /// They carry the `h1.15-exception` marker the determinism gate allows on a
    /// line, rather than excluding this file the way `signing.rs` is excluded.
    fn wait(&self, id: SubmissionId, timeout: Duration) -> Receipt {
        let deadline = Instant::now() + timeout; // h1.15-exception: wait deadline
        loop {
            let expired = Instant::now() >= deadline; // h1.15-exception: wait deadline
            match self.receipt(id) {
                ReceiptState::Pending if !expired => {
                    std::thread::sleep(Duration::from_millis(2));
                }
                state => return Receipt { id, state },
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LocalConsensus {
    log: Arc<RwLock<Vec<(u64, ConsensusTransaction)>>>,
}

impl LocalConsensus {
    pub fn new() -> Self {
        Self {
            log: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl Default for LocalConsensus {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsensusEngine for LocalConsensus {
    fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic> {
        let mut log = self.log.write().unwrap();
        let seq = log.len() as u64 + 1;
        log.push((seq, tx));
        Ok(seq)
    }

    fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)> {
        let log = self.log.read().unwrap();
        log.iter()
            .filter(|(seq, _)| *seq >= from_seq)
            .cloned()
            .collect()
    }

    fn current_seq(&self) -> u64 {
        self.log.read().unwrap().len() as u64
    }

    /// Local submission is commitment: the id `submit` returned is the slot the
    /// entry occupies, so a receipt for it is already committed.
    fn receipt(&self, id: SubmissionId) -> ReceiptState {
        if id.0 == 0 || id.0 > self.current_seq() {
            ReceiptState::Unknown(UnknownReason::Unsubmitted)
        } else {
            ReceiptState::Committed(Slot(id.0))
        }
    }
}

pub fn duplicate_tx_diagnostic() -> Diagnostic {
    Diagnostic::new(
        Code::DUPLICATE_TRANSACTION,
        "transaction already in the consensus log",
        Severity::Error,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::{IdentityOp, IdentityOpKind};

    fn dummy_identity_op(seq: u64) -> ConsensusTransaction {
        ConsensusTransaction::IdentityOp(IdentityOp {
            did: format!("did:unfer:{:064x}", seq),
            op_kind: IdentityOpKind::Create,
            signing_key: [0u8; 32],
            signature: [0u8; 64],
            seq,
            service_endpoint: None,
        })
    }

    #[test]
    fn submit_assigns_monotonic_seq() {
        let engine = LocalConsensus::new();
        let s1 = engine.submit(dummy_identity_op(1)).unwrap();
        let s2 = engine.submit(dummy_identity_op(2)).unwrap();
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(engine.current_seq(), 2);
    }

    #[test]
    fn get_log_filters_by_seq() {
        let engine = LocalConsensus::new();
        engine.submit(dummy_identity_op(1)).unwrap();
        engine.submit(dummy_identity_op(2)).unwrap();
        engine.submit(dummy_identity_op(3)).unwrap();

        let from_2 = engine.get_log(2);
        assert_eq!(from_2.len(), 2);
        assert_eq!(from_2[0].0, 2);
        assert_eq!(from_2[1].0, 3);
    }

    #[test]
    fn local_receipt_is_its_own_slot() {
        let engine = LocalConsensus::new();
        let id = engine.submit(dummy_identity_op(1)).unwrap();
        assert_eq!(
            engine.receipt(SubmissionId(id)),
            ReceiptState::Committed(Slot(id))
        );
        assert_eq!(
            engine.receipt(SubmissionId(id + 1)),
            ReceiptState::Unknown(UnknownReason::Unsubmitted)
        );
        let receipt = engine.wait(SubmissionId(id), Duration::from_secs(1));
        assert!(receipt.is_committed());
    }

    #[test]
    fn empty_log_returns_nothing() {
        let engine = LocalConsensus::new();
        assert_eq!(engine.current_seq(), 0);
        assert!(engine.get_log(0).is_empty());
    }
}
