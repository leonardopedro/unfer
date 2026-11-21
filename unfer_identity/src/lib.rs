//! `did:unfer` identity method backed by the QuePaxa consensus log.
//!
//! [`DidManager`] implements the create / resolve / update / revoke
//! lifecycle over a `ConsensusNode`, emitting signed `IdentityOp`
//! transactions. [`did_from_pubkey`] and [`pubkey_from_did`] convert
//! between the 32-byte Ed25519 public key and the `did:unfer:<hex>` form.
//!
//! [`DidManager`]: struct@DidManager
//! [`did_from_pubkey`]: fn@did_from_pubkey
//! [`pubkey_from_did`]: fn@pubkey_from_did
//!
//! ```
//! let did = unfer_identity::did_from_pubkey(&[7u8; 32]);
//! assert!(did.starts_with("did:unfer:"));
//! let pubkey = unfer_identity::pubkey_from_did(&did).unwrap();
//! assert_eq!(&pubkey[..4], &[7u8; 4]);
//! ```

use std::time::Duration;

use unfer_consensus::{ConsensusNode, Keypair};
use unfer_protocol::submission::{ReceiptState, SubmissionId};
use unfer_protocol::{ConsensusTransaction, DidDocument, Diagnostic, IdentityOp, IdentityOpKind};

/// How long a lifecycle call waits for its transaction to be committed and
/// applied before reporting an unknown outcome (D05). A local engine settles
/// immediately; a network engine may not have committed yet, and that is
/// reported, never reported as success.
pub const DID_SETTLE_TIMEOUT: Duration = Duration::from_secs(5);

pub struct DidManager<'a> {
    node: &'a mut ConsensusNode,
    settle_timeout: Duration,
}

impl<'a> DidManager<'a> {
    pub fn new(node: &'a mut ConsensusNode) -> Self {
        Self {
            node,
            settle_timeout: DID_SETTLE_TIMEOUT,
        }
    }

    /// Override the settlement wait (tests use a short deadline).
    pub fn with_settle_timeout(mut self, timeout: Duration) -> Self {
        self.settle_timeout = timeout;
        self
    }

    /// D05: a lifecycle call is complete only once its entry is committed *and*
    /// applied. A refusal is the entry's typed outcome; a deadline is an unknown
    /// outcome, not a creation and not a refusal.
    fn settle(&mut self, id: SubmissionId) -> Result<(), Diagnostic> {
        let receipt = self.node.settle(id, self.settle_timeout);
        match receipt.state {
            ReceiptState::Committed(_) => Ok(()),
            _ => Err(receipt.failure().unwrap_or_else(|| {
                Diagnostic::new(
                    unfer_protocol::Code::UNKNOWN_OUTCOME,
                    "the identity transaction did not report an outcome",
                    unfer_protocol::Severity::Error,
                )
            })),
        }
    }

    pub fn create_did(
        &mut self,
        keypair: &Keypair,
        service_endpoint: Option<String>,
    ) -> Result<String, unfer_protocol::Diagnostic> {
        let did = keypair.did();
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: did.clone(),
            op_kind: IdentityOpKind::Create,
            signing_key: keypair.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint,
        });
        unfer_consensus::sign_transaction(&mut tx, keypair);
        let id = self.node.submit_tracked(tx)?;
        self.settle(id)?;
        Ok(did)
    }

    pub fn update_did(
        &mut self,
        keypair: &Keypair,
        service_endpoint: Option<String>,
    ) -> Result<(), unfer_protocol::Diagnostic> {
        let did = keypair.did();
        let entry = self.node.identity().resolve(&did).ok_or_else(|| {
            unfer_protocol::Diagnostic::new(
                unfer_protocol::Code::UNKNOWN_DID,
                format!("DID not found: {did}"),
                unfer_protocol::Severity::Error,
            )
        })?;
        let next_seq = entry.seq + 1;
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: did.clone(),
            op_kind: IdentityOpKind::Update,
            signing_key: keypair.public_key(),
            signature: [0u8; 64],
            seq: next_seq,
            service_endpoint,
        });
        unfer_consensus::sign_transaction(&mut tx, keypair);
        let id = self.node.submit_tracked(tx)?;
        self.settle(id)
    }

    pub fn revoke_did(&mut self, keypair: &Keypair) -> Result<(), unfer_protocol::Diagnostic> {
        let did = keypair.did();
        let entry = self.node.identity().resolve(&did).ok_or_else(|| {
            unfer_protocol::Diagnostic::new(
                unfer_protocol::Code::UNKNOWN_DID,
                format!("DID not found: {did}"),
                unfer_protocol::Severity::Error,
            )
        })?;
        let next_seq = entry.seq + 1;
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: did.clone(),
            op_kind: IdentityOpKind::Revoke,
            signing_key: keypair.public_key(),
            signature: [0u8; 64],
            seq: next_seq,
            service_endpoint: None,
        });
        unfer_consensus::sign_transaction(&mut tx, keypair);
        let id = self.node.submit_tracked(tx)?;
        self.settle(id)
    }

    pub fn resolve(&self, did: &str) -> Option<DidDocument> {
        self.node.identity().resolve_document(did)
    }

    pub fn resolve_json(&self, did: &str) -> Option<String> {
        self.resolve(did)
            .map(|doc| serde_json::to_string_pretty(&doc).unwrap())
    }
}

pub fn did_from_pubkey(pubkey: &[u8; 32]) -> String {
    format!("did:unfer:{}", hex::encode(pubkey))
}

pub fn pubkey_from_did(did: &str) -> Option<[u8; 32]> {
    let hex_key = did.strip_prefix("did:unfer:")?;
    let bytes = hex::decode(hex_key).ok()?;
    bytes.try_into().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_consensus::{ConsensusNode, LocalConsensus};

    fn make_node() -> ConsensusNode {
        ConsensusNode::new(Box::new(LocalConsensus::new()))
    }

    #[test]
    fn create_and_resolve_did() {
        let mut node = make_node();
        let kp = Keypair::generate();
        let did = {
            let mut mgr = DidManager::new(&mut node);
            mgr.create_did(&kp, Some("https://node.example.com".to_string()))
                .unwrap()
        };
        assert!(did.starts_with("did:unfer:"));

        let mgr = DidManager::new(&mut node);
        let doc = mgr.resolve(&did).unwrap();
        assert_eq!(doc.id, did);
        assert_eq!(doc.service.len(), 1);
        assert_eq!(doc.service[0].service_endpoint, "https://node.example.com");
    }

    #[test]
    fn update_service_endpoint() {
        let mut node = make_node();
        let kp = Keypair::generate();
        {
            let mut mgr = DidManager::new(&mut node);
            mgr.create_did(&kp, None).unwrap();
        }
        {
            let mut mgr = DidManager::new(&mut node);
            mgr.update_did(&kp, Some("https://new.example.com".to_string()))
                .unwrap();
        }
        let mgr = DidManager::new(&mut node);
        let doc = mgr.resolve(&kp.did()).unwrap();
        assert_eq!(doc.service.len(), 1);
        assert_eq!(doc.service[0].service_endpoint, "https://new.example.com");
    }

    #[test]
    fn revoke_did() {
        let mut node = make_node();
        let kp = Keypair::generate();
        {
            let mut mgr = DidManager::new(&mut node);
            mgr.create_did(&kp, None).unwrap();
        }
        {
            let mut mgr = DidManager::new(&mut node);
            mgr.revoke_did(&kp).unwrap();
        }
        let mgr = DidManager::new(&mut node);
        assert!(mgr.resolve(&kp.did()).is_none());
    }

    #[test]
    fn resolve_json_is_valid() {
        let mut node = make_node();
        let kp = Keypair::generate();
        {
            let mut mgr = DidManager::new(&mut node);
            mgr.create_did(&kp, None).unwrap();
        }
        let mgr = DidManager::new(&mut node);
        let json = mgr.resolve_json(&kp.did()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["@context"], "https://www.w3.org/ns/did/v1");
        assert!(parsed["id"].as_str().unwrap().starts_with("did:unfer:"));
    }

    #[test]
    fn did_pubkey_roundtrip() {
        let kp = Keypair::generate();
        let did = did_from_pubkey(&kp.public_key());
        let recovered = pubkey_from_did(&did).unwrap();
        assert_eq!(kp.public_key(), recovered);
    }

    /// A network engine that accepts submissions and never commits them: the
    /// submission is durable, its outcome is simply not known yet.
    struct NeverCommits {
        next: std::sync::atomic::AtomicU64,
    }

    impl unfer_consensus::ConsensusEngine for NeverCommits {
        fn submit(
            &self,
            _tx: ConsensusTransaction,
        ) -> Result<u64, Diagnostic> {
            Ok(self
                .next
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed))
        }

        fn get_log(&self, _from_seq: u64) -> Vec<(u64, ConsensusTransaction)> {
            Vec::new()
        }

        fn current_seq(&self) -> u64 {
            0
        }

        fn receipt(&self, _id: SubmissionId) -> ReceiptState {
            ReceiptState::Pending
        }
    }

    #[test]
    fn pending_creation_is_not_reported_as_created() {
        let engine = NeverCommits {
            next: std::sync::atomic::AtomicU64::new(1),
        };
        let mut node = ConsensusNode::new(Box::new(engine));
        let kp = Keypair::generate();
        let err = {
            let mut mgr =
                DidManager::new(&mut node).with_settle_timeout(Duration::from_millis(20));
            mgr.create_did(&kp, None).unwrap_err()
        };
        // Not created, and not a refusal either: the outcome is unknown here.
        assert_eq!(err.code, unfer_protocol::Code::UNKNOWN_OUTCOME);
        assert_eq!(node.applied_seq(), 0);
        assert!(DidManager::new(&mut node).resolve(&kp.did()).is_none());
    }

    #[test]
    fn duplicate_create_fails() {
        let mut node = make_node();
        let kp = Keypair::generate();
        {
            let mut mgr = DidManager::new(&mut node);
            mgr.create_did(&kp, None).unwrap();
        }
        {
            let mut mgr = DidManager::new(&mut node);
            assert!(mgr.create_did(&kp, None).is_err());
        }
    }
}
