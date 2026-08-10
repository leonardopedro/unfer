use std::collections::HashMap;

use prob_kernel::Session;
use unfer_protocol::{
    AgentRequest, Code, ConsensusTransaction, ContentRef, Diagnostic, ModelSpec, Severity,
};

use crate::certs::{CertificateLedger, MintAuthority};
use crate::engine::ConsensusEngine;
use crate::identity::IdentityRegistry;

pub struct ConsensusNode {
    engine: Box<dyn ConsensusEngine>,
    sessions: HashMap<u64, Session>,
    identity: IdentityRegistry,
    content: HashMap<String, ContentRef>,
    certs: CertificateLedger,
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
            certs: CertificateLedger::new(MintAuthority::None),
            next_model_id: 1,
            applied_seq: 0,
        }
    }

    /// Constructor with a configured certificate mint authority (ReFi exchange).
    pub fn with_mint_authority(engine: Box<dyn ConsensusEngine>, authority: MintAuthority) -> Self {
        let mut node = Self::new(engine);
        node.certs = CertificateLedger::new(authority);
        node
    }

    /// Reconfigure the certificate mint authority (ReFi exchange). Minting is
    /// disabled by default; call with `MintAuthority::Only(did)` to permit a
    /// specific authority to mint, or `MintAuthority::None` to disable it.
    pub fn set_mint_authority(&mut self, authority: MintAuthority) {
        self.certs = CertificateLedger::new(authority);
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
            ConsensusTransaction::CertificateOp(op) => {
                let actor = op.did.clone();
                let kind = op.kind.clone();
                self.certs.apply_op(&actor, &kind, op.seq)?;
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

    pub fn certs(&self) -> &CertificateLedger {
        &self.certs
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

    #[test]
    fn certificate_ledger_roundtrip_via_consensus() {
        use crate::certs::{MintAuthority, commit_coin};
        use unfer_protocol::{CertificateOp, CertificateOpKind, CoinRef};

        let engine = LocalConsensus::new();
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();

        let mut node =
            ConsensusNode::with_mint_authority(Box::new(engine.clone()), MintAuthority::Only(authority.did()));
        let empty_root = node.certs().root();
        assert_ne!(empty_root, [0u8; 32]);

        // Mint 1000 to alice (signed by the authority).
        let mut mint = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount: 1000,
                owner: alice.did(),
                blinding: [1u8; 32],
                source: Some("unfccc:cert:TEST-0001".to_string()),
            },
            seq: 1,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut mint, &authority);
        node.submit(mint).unwrap();
        node.sync().unwrap();

        let alice_coin = commit_coin(1000, &alice.did(), &[1u8; 32]);
        assert!(node.certs().utxo(&alice_coin).is_some());
        assert_eq!(node.certs().total_supply(), 1000);

        // Transfer the whole 1000 to bob (signed by alice).
        let input = CoinRef {
            coin_id: alice_coin,
            amount: 1000,
            owner: alice.did(),
        };
        let output = CoinRef {
            coin_id: alice_coin, // recomputed below; amount/owner drive the id
            amount: 1000,
            owner: bob.did(),
        };
        let mut transfer = ConsensusTransaction::CertificateOp(CertificateOp {
            did: alice.did(),
            kind: CertificateOpKind::Transfer {
                inputs: vec![input],
                outputs: vec![output],
            },
            seq: 2,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut transfer, &alice);
        node.submit(transfer).unwrap();
        node.sync().unwrap();

        let bob_coin = commit_coin(1000, &bob.did(), &[0u8; 32]);
        assert!(node.certs().utxo(&bob_coin).is_some());
        assert!(node.certs().utxo(&alice_coin).is_none());
        assert!(node.certs().is_spent(&alice_coin));
        assert_eq!(node.certs().total_supply(), 1000);

        // Burn bob's certificate.
        let burn_input = CoinRef {
            coin_id: bob_coin,
            amount: 1000,
            owner: bob.did(),
        };
        let mut burn = ConsensusTransaction::CertificateOp(CertificateOp {
            did: bob.did(),
            kind: CertificateOpKind::Burn {
                inputs: vec![burn_input],
            },
            seq: 3,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut burn, &bob);
        node.submit(burn).unwrap();
        node.sync().unwrap();

        assert!(node.certs().utxo(&bob_coin).is_none());
        assert_eq!(node.certs().total_supply(), 0);
        // Retiring the only UTXO restores the empty root.
        assert_eq!(node.certs().root(), empty_root);
    }

    #[test]
    fn invalid_certificate_op_rejected_before_log() {
        use unfer_protocol::{CertificateOp, CertificateOpKind};

        let engine = LocalConsensus::new();
        let alice = Keypair::generate();
        let mut node = ConsensusNode::new(Box::new(engine));
        // Minting is disabled by default (MintAuthority::None).
        let mut mint = ConsensusTransaction::CertificateOp(CertificateOp {
            did: alice.did(),
            kind: CertificateOpKind::Mint {
                amount: 100,
                owner: alice.did(),
                blinding: [9u8; 32],
                source: None,
            },
            seq: 1,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut mint, &alice);
        // Signature is valid, so it reaches the log; application fails on sync
        // with UK-7001 (mint not authorized).
        let seq = node.submit(mint).unwrap();
        assert_eq!(seq, 1);
        let err = node.sync().unwrap_err();
        assert_eq!(err.code, Code::CERT_MINT_NOT_AUTHORIZED);
    }
}
