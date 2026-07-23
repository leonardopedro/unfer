use std::collections::HashMap;

use prob_kernel::Session;
use unfer_protocol::{
    AgentRequest, Code, ConsensusTransaction, ContentRef, Diagnostic, ModelSpec, Severity,
};

use crate::engine::ConsensusEngine;
use crate::identity::IdentityRegistry;

pub struct ConsensusNode {
    engine: Box<dyn ConsensusEngine>,
    sessions: HashMap<u64, Session>,
    identity: IdentityRegistry,
    content: HashMap<String, ContentRef>,
    next_model_id: u64,
    applied_seq: u64,
}

impl ConsensusNode {
    pub fn new(engine: Box<dyn ConsensusEngine>) -> Self {
        Self {
            engine,
            sessions: HashMap::new(),
            identity: IdentityRegistry::new(),
            content: HashMap::new(),
            next_model_id: 1,
            applied_seq: 0,
        }
    }

    pub fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic> {
        crate::signing::verify_transaction(&tx)?;
        self.engine.submit(tx)
    }

    pub fn sync(&mut self) -> Result<u64, Diagnostic> {
        let log = self.engine.get_log(self.applied_seq + 1);
        let mut applied = 0u64;
        for (seq, tx) in log {
            self.apply_transaction(seq, &tx)?;
            self.applied_seq = seq;
            applied += 1;
        }
        Ok(applied)
    }

    fn apply_transaction(
        &mut self,
        _seq: u64,
        tx: &ConsensusTransaction,
    ) -> Result<(), Diagnostic> {
        match tx {
            ConsensusTransaction::IdentityOp(op) => {
                self.identity.apply_identity_op(op)?;
            }
            ConsensusTransaction::SessionOp(op) => {
                self.apply_session_op(&op.op)?;
            }
            ConsensusTransaction::ContentOp(op) => {
                self.content
                    .insert(op.content_ref.cid.clone(), op.content_ref.clone());
            }
        }
        Ok(())
    }

    fn apply_session_op(&mut self, req: &AgentRequest) -> Result<(), Diagnostic> {
        match req.op.as_str() {
            "create_model" => {
                let spec: ModelSpec = serde_json::from_value(req.params.clone()).map_err(|e| {
                    Diagnostic::new(Code::BAD_JSON, e.to_string(), Severity::Error)
                })?;
                let session = Session::new(&spec).map_err(|e| e.to_diagnostic())?;
                let id = self.next_model_id;
                self.next_model_id += 1;
                self.sessions.insert(id, session);
            }
            _ => {
                return Err(Diagnostic::new(
                    Code::BAD_JSON,
                    format!("unsupported session op in consensus: {}", req.op),
                    Severity::Error,
                ));
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> &IdentityRegistry {
        &self.identity
    }

    pub fn content(&self, cid: &str) -> Option<&ContentRef> {
        self.content.get(cid)
    }

    pub fn session(&self, id: u64) -> Option<&Session> {
        self.sessions.get(&id)
    }

    pub fn session_mut(&mut self, id: u64) -> Option<&mut Session> {
        self.sessions.get_mut(&id)
    }

    pub fn applied_seq(&self) -> u64 {
        self.applied_seq
    }

    pub fn current_seq(&self) -> u64 {
        self.engine.current_seq()
    }

    pub fn is_synced(&self) -> bool {
        self.applied_seq == self.engine.current_seq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::LocalConsensus;
    use crate::signing::Keypair;
    use unfer_protocol::{IdentityOp, IdentityOpKind};

    fn make_node() -> ConsensusNode {
        ConsensusNode::new(Box::new(LocalConsensus::new()))
    }

    #[test]
    fn submit_and_sync_identity() {
        let mut node = make_node();
        let kp = Keypair::generate();
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx, &kp);
        node.submit(tx).unwrap();
        let applied = node.sync().unwrap();
        assert_eq!(applied, 1);
        assert!(node.identity().resolve(&kp.did()).is_some());
        assert!(node.is_synced());
    }

    #[test]
    fn submit_and_sync_content() {
        let mut node = make_node();
        let kp = Keypair::generate();
        let mut tx = ConsensusTransaction::ContentOp(unfer_protocol::ContentOp {
            did: kp.did(),
            content_ref: ContentRef {
                cid: "abc123".to_string(),
                magnet_uri: "magnet:?xt=urn:btih:abc123".to_string(),
                encryption_key: "x25519:deadbeef".to_string(),
                filesize: 1024,
                mime_type: "video/mp4".to_string(),
                chunks: vec![],
            },
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut tx, &kp);
        node.submit(tx).unwrap();
        node.sync().unwrap();
        assert!(node.content("abc123").is_some());
    }

    #[test]
    fn invalid_signature_rejected() {
        let node = make_node();
        let kp = Keypair::generate();
        let tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0xFFu8; 64],
            seq: 1,
            service_endpoint: None,
        });
        assert!(node.submit(tx).is_err());
    }

    #[test]
    fn two_nodes_converge() {
        let engine = LocalConsensus::new();
        let kp = Keypair::generate();

        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx, &kp);
        engine.submit(tx).unwrap();

        let mut node_a = ConsensusNode::new(Box::new(engine.clone()));
        let mut node_b = ConsensusNode::new(Box::new(engine.clone()));

        node_a.sync().unwrap();
        node_b.sync().unwrap();

        assert!(node_a.identity().resolve(&kp.did()).is_some());
        assert!(node_b.identity().resolve(&kp.did()).is_some());
        assert_eq!(node_a.applied_seq(), node_b.applied_seq());
    }

    #[test]
    fn incremental_sync() {
        let engine = LocalConsensus::new();
        let kp = Keypair::generate();

        let mut tx1 = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx1, &kp);
        engine.submit(tx1).unwrap();

        let mut node = ConsensusNode::new(Box::new(engine.clone()));
        node.sync().unwrap();
        assert_eq!(node.applied_seq(), 1);

        let kp2 = Keypair::generate();
        let mut tx2 = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp2.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp2.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx2, &kp2);
        engine.submit(tx2).unwrap();

        let applied = node.sync().unwrap();
        assert_eq!(applied, 1);
        assert_eq!(node.applied_seq(), 2);
        assert!(node.identity().resolve(&kp2.did()).is_some());
    }
}
