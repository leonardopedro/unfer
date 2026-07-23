use std::sync::{Arc, RwLock};

use unfer_protocol::{Code, ConsensusTransaction, Diagnostic, Severity};

pub trait ConsensusEngine: Send + Sync {
    fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic>;
    fn get_log(&self, from_seq: u64) -> Vec<(u64, ConsensusTransaction)>;
    fn current_seq(&self) -> u64;
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
    fn empty_log_returns_nothing() {
        let engine = LocalConsensus::new();
        assert_eq!(engine.current_seq(), 0);
        assert!(engine.get_log(0).is_empty());
    }
}
