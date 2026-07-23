use std::collections::HashMap;

use unfer_protocol::{
    Code, Diagnostic, DidEntry, DidDocument, IdentityOp, IdentityOpKind, Severity,
};

#[derive(Debug, Clone, Default)]
pub struct IdentityRegistry {
    entries: HashMap<String, DidEntry>,
}

impl IdentityRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn apply_identity_op(&mut self, op: &IdentityOp) -> Result<(), Diagnostic> {
        match op.op_kind {
            IdentityOpKind::Create => {
                if self.entries.contains_key(&op.did) {
                    return Err(Diagnostic::new(
                        Code::DUPLICATE_TRANSACTION,
                        format!("DID already registered: {}", op.did),
                        Severity::Error,
                    ));
                }
                self.entries.insert(
                    op.did.clone(),
                    DidEntry {
                        did: op.did.clone(),
                        pubkey: op.signing_key,
                        seq: op.seq,
                        created_at: op.seq,
                        revoked: false,
                        service_endpoint: op.service_endpoint.clone(),
                    },
                );
                Ok(())
            }
            IdentityOpKind::Update => {
                let entry = self.entries.get_mut(&op.did).ok_or_else(|| {
                    Diagnostic::new(
                        Code::UNKNOWN_DID,
                        format!("DID not found: {}", op.did),
                        Severity::Error,
                    )
                })?;
                if entry.revoked {
                    return Err(Diagnostic::new(
                        Code::UNKNOWN_DID,
                        format!("DID is revoked: {}", op.did),
                        Severity::Error,
                    ));
                }
                if op.seq <= entry.seq {
                    return Err(Diagnostic::new(
                        Code::DUPLICATE_TRANSACTION,
                        format!(
                            "stale seq: op.seq={} <= entry.seq={}",
                            op.seq, entry.seq
                        ),
                        Severity::Error,
                    ));
                }
                entry.seq = op.seq;
                if let Some(ref ep) = op.service_endpoint {
                    entry.service_endpoint = Some(ep.clone());
                }
                Ok(())
            }
            IdentityOpKind::Revoke => {
                let entry = self.entries.get_mut(&op.did).ok_or_else(|| {
                    Diagnostic::new(
                        Code::UNKNOWN_DID,
                        format!("DID not found: {}", op.did),
                        Severity::Error,
                    )
                })?;
                entry.revoked = true;
                entry.seq = op.seq;
                Ok(())
            }
        }
    }

    pub fn resolve(&self, did: &str) -> Option<&DidEntry> {
        self.entries.get(did).filter(|e| !e.revoked)
    }

    pub fn resolve_document(&self, did: &str) -> Option<DidDocument> {
        self.resolve(did).map(|e| e.to_document())
    }

    pub fn contains(&self, did: &str) -> bool {
        self.entries.contains_key(did)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_op(did: &str, seq: u64) -> IdentityOp {
        IdentityOp {
            did: did.to_string(),
            op_kind: IdentityOpKind::Create,
            signing_key: [1u8; 32],
            signature: [0u8; 64],
            seq,
            service_endpoint: None,
        }
    }

    #[test]
    fn create_and_resolve() {
        let mut reg = IdentityRegistry::new();
        let op = create_op("did:unfer:alice", 1);
        reg.apply_identity_op(&op).unwrap();
        assert!(reg.resolve("did:unfer:alice").is_some());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn duplicate_create_fails() {
        let mut reg = IdentityRegistry::new();
        let op = create_op("did:unfer:alice", 1);
        reg.apply_identity_op(&op).unwrap();
        let err = reg.apply_identity_op(&op).unwrap_err();
        assert_eq!(err.code, Code::DUPLICATE_TRANSACTION);
    }

    #[test]
    fn update_increments_seq() {
        let mut reg = IdentityRegistry::new();
        reg.apply_identity_op(&create_op("did:unfer:alice", 1))
            .unwrap();
        let update = IdentityOp {
            did: "did:unfer:alice".to_string(),
            op_kind: IdentityOpKind::Update,
            signing_key: [1u8; 32],
            signature: [0u8; 64],
            seq: 2,
            service_endpoint: Some("https://node.example.com".to_string()),
        };
        reg.apply_identity_op(&update).unwrap();
        let entry = reg.resolve("did:unfer:alice").unwrap();
        assert_eq!(entry.seq, 2);
        assert_eq!(
            entry.service_endpoint.as_deref(),
            Some("https://node.example.com")
        );
    }

    #[test]
    fn stale_update_fails() {
        let mut reg = IdentityRegistry::new();
        reg.apply_identity_op(&create_op("did:unfer:alice", 5))
            .unwrap();
        let stale = IdentityOp {
            did: "did:unfer:alice".to_string(),
            op_kind: IdentityOpKind::Update,
            signing_key: [1u8; 32],
            signature: [0u8; 64],
            seq: 3,
            service_endpoint: None,
        };
        let err = reg.apply_identity_op(&stale).unwrap_err();
        assert_eq!(err.code, Code::DUPLICATE_TRANSACTION);
    }

    #[test]
    fn revoke_hides_from_resolve() {
        let mut reg = IdentityRegistry::new();
        reg.apply_identity_op(&create_op("did:unfer:alice", 1))
            .unwrap();
        let revoke = IdentityOp {
            did: "did:unfer:alice".to_string(),
            op_kind: IdentityOpKind::Revoke,
            signing_key: [1u8; 32],
            signature: [0u8; 64],
            seq: 2,
            service_endpoint: None,
        };
        reg.apply_identity_op(&revoke).unwrap();
        assert!(reg.resolve("did:unfer:alice").is_none());
        assert!(reg.contains("did:unfer:alice"));
    }

    #[test]
    fn resolve_document_has_correct_structure() {
        let mut reg = IdentityRegistry::new();
        reg.apply_identity_op(&create_op("did:unfer:alice", 1))
            .unwrap();
        let doc = reg.resolve_document("did:unfer:alice").unwrap();
        assert_eq!(doc.id, "did:unfer:alice");
        assert_eq!(doc.context, "https://www.w3.org/ns/did/v1");
        assert_eq!(doc.verification_method.len(), 1);
        assert_eq!(
            doc.verification_method[0].method_type,
            "Ed25519VerificationKey2020"
        );
        assert!(doc.service.is_empty());
    }

    #[test]
    fn unknown_did_returns_none() {
        let reg = IdentityRegistry::new();
        assert!(reg.resolve("did:unfer:nobody").is_none());
        assert!(reg.resolve_document("did:unfer:nobody").is_none());
    }
}
