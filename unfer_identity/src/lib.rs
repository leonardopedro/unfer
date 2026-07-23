use unfer_consensus::{ConsensusNode, Keypair};
use unfer_protocol::{
    ConsensusTransaction, DidDocument, IdentityOp, IdentityOpKind,
};

pub struct DidManager<'a> {
    node: &'a mut ConsensusNode,
}

impl<'a> DidManager<'a> {
    pub fn new(node: &'a mut ConsensusNode) -> Self {
        Self { node }
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
        self.node.submit(tx)?;
        self.node.sync()?;
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
        self.node.submit(tx)?;
        self.node.sync()?;
        Ok(())
    }

    pub fn revoke_did(
        &mut self,
        keypair: &Keypair,
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
            op_kind: IdentityOpKind::Revoke,
            signing_key: keypair.public_key(),
            signature: [0u8; 64],
            seq: next_seq,
            service_endpoint: None,
        });
        unfer_consensus::sign_transaction(&mut tx, keypair);
        self.node.submit(tx)?;
        self.node.sync()?;
        Ok(())
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
        assert_eq!(
            doc.service[0].service_endpoint,
            "https://node.example.com"
        );
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
        assert_eq!(
            doc.service[0].service_endpoint,
            "https://new.example.com"
        );
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
