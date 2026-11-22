use std::collections::HashMap;

use unfer_protocol::{
    Code, Diagnostic, DidDocument, DidEntry, IdentityOp, IdentityOpKind, Severity,
};

use crate::signing;

/// The `did:unfer:` name a public key owns. A DID here is self-certifying:
/// the name *is* the key, so a create that names someone else's DID with its
/// own key is a forged claim and not a rename.
fn did_for_key(pubkey: &[u8; 32]) -> String {
    format!("did:unfer:{}", hex::encode(pubkey))
}

fn refused(code: Code, message: String) -> Diagnostic {
    Diagnostic::new(code, message, Severity::Error)
}

/// Verify an op against the authority that owns the DID, not against the key
/// the op carries. A forwarded or replayed identity op has to be authentic
/// under the ordered state, so the registry is the enforcement point and
/// ingress verification is not sufficient on its own.
fn verify_under_current_authority(
    op: &IdentityOp,
    stored_pubkey: [u8; 32],
) -> Result<(), Diagnostic> {
    if op.signing_key != stored_pubkey {
        return Err(refused(
            Code::INVALID_SIGNATURE,
            format!(
                "{}: the operation carries a key that is not the DID's current authority; \
                 unsigned key rotation is not supported",
                op.did
            ),
        ));
    }
    signing::verify_identity_op_as(op, stored_pubkey)
}

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
                let expected = did_for_key(&op.signing_key);
                if op.did != expected {
                    return Err(refused(
                        Code::INVALID_SIGNATURE,
                        format!(
                            "create names {} but carries the key of {}; a DID is \
                             self-certifying and cannot claim another name",
                            op.did, expected
                        ),
                    ));
                }
                signing::verify_identity_op_as(op, op.signing_key)?;
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
                verify_under_current_authority(op, entry.pubkey)?;
                if op.seq <= entry.seq {
                    return Err(Diagnostic::new(
                        Code::DUPLICATE_TRANSACTION,
                        format!("stale seq: op.seq={} <= entry.seq={}", op.seq, entry.seq),
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
                if entry.revoked {
                    return Err(Diagnostic::new(
                        Code::DUPLICATE_TRANSACTION,
                        format!("DID is already revoked: {}", op.did),
                        Severity::Error,
                    ));
                }
                verify_under_current_authority(op, entry.pubkey)?;
                // A revoke ordered behind a later operation is a stale
                // authority claim, not a revocation: only a sequence newer than
                // what the DID has already accepted may end it.
                if op.seq <= entry.seq {
                    return Err(Diagnostic::new(
                        Code::DUPLICATE_TRANSACTION,
                        format!("stale revoke: op.seq={} <= entry.seq={}", op.seq, entry.seq),
                        Severity::Error,
                    ));
                }
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

    use crate::signing::{sign_transaction, verify_identity_op_as, Keypair};
    use unfer_protocol::ConsensusTransaction;

    /// Build an op signed by `signer`, but carrying `signing_key` — the two
    /// agree for an honest caller and disagree for an attacker presenting its
    /// own key under the victim's DID.
    fn op_signed_by(
        signer: &Keypair,
        did: &str,
        signing_key: [u8; 32],
        kind: IdentityOpKind,
        seq: u64,
        service_endpoint: Option<&str>,
    ) -> IdentityOp {
        let mut op = IdentityOp {
            did: did.to_string(),
            op_kind: kind,
            signing_key,
            signature: [0u8; 64],
            seq,
            service_endpoint: service_endpoint.map(|e| e.to_string()),
        };
        // Sign the canonical bytes of the op as submitted (the signature field
        // is zeroed inside `canonical_bytes`).
        let mut tx = ConsensusTransaction::IdentityOp(op.clone());
        sign_transaction(&mut tx, signer);
        if let ConsensusTransaction::IdentityOp(signed) = tx {
            op.signature = signed.signature;
        }
        op
    }

    fn create(kp: &Keypair, seq: u64) -> IdentityOp {
        op_signed_by(kp, &kp.did(), kp.public_key(), IdentityOpKind::Create, seq, None)
    }

    fn state_of(reg: &IdentityRegistry, did: &str) -> Option<(u64, bool, Option<String>)> {
        reg.resolve(did)
            .map(|e| (e.seq, e.revoked, e.service_endpoint.clone()))
    }

    #[test]
    fn create_and_resolve() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        reg.apply_identity_op(&create(&alice, 1)).unwrap();
        assert!(reg.resolve(&alice.did()).is_some());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn duplicate_create_fails() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        let op = create(&alice, 1);
        reg.apply_identity_op(&op).unwrap();
        let err = reg.apply_identity_op(&op).unwrap_err();
        assert_eq!(err.code, Code::DUPLICATE_TRANSACTION);
    }

    #[test]
    fn update_increments_seq() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        reg.apply_identity_op(&create(&alice, 1)).unwrap();
        let update = op_signed_by(
            &alice,
            &alice.did(),
            alice.public_key(),
            IdentityOpKind::Update,
            2,
            Some("https://node.example.com"),
        );
        reg.apply_identity_op(&update).unwrap();
        let entry = reg.resolve(&alice.did()).unwrap();
        assert_eq!(entry.seq, 2);
        assert_eq!(
            entry.service_endpoint.as_deref(),
            Some("https://node.example.com")
        );
    }

    #[test]
    fn stale_update_fails() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        reg.apply_identity_op(&create(&alice, 5)).unwrap();
        let stale = op_signed_by(
            &alice,
            &alice.did(),
            alice.public_key(),
            IdentityOpKind::Update,
            3,
            None,
        );
        let err = reg.apply_identity_op(&stale).unwrap_err();
        assert_eq!(err.code, Code::DUPLICATE_TRANSACTION);
    }

    #[test]
    fn revoke_hides_from_resolve() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        reg.apply_identity_op(&create(&alice, 1)).unwrap();
        let revoke = op_signed_by(
            &alice,
            &alice.did(),
            alice.public_key(),
            IdentityOpKind::Revoke,
            2,
            None,
        );
        reg.apply_identity_op(&revoke).unwrap();
        assert!(reg.resolve(&alice.did()).is_none());
        assert!(reg.contains(&alice.did()));
    }

    /// D01 witness: a create that names a DID its key does not own is refused,
    /// even when the attacker signed its own body correctly.
    #[test]
    fn create_cannot_claim_another_dids_name() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        let attacker = Keypair::generate();
        let forged = op_signed_by(
            &attacker,
            &alice.did(),
            attacker.public_key(),
            IdentityOpKind::Create,
            1,
            None,
        );
        let err = reg.apply_identity_op(&forged).unwrap_err();
        assert_eq!(err.code, Code::INVALID_SIGNATURE);
        assert!(!reg.contains(&alice.did()));
    }

    /// D01 witness: the op is authentic under the attacker's own key, which is
    /// exactly why ingress-only verification passes it; the registry must
    /// authenticate it under Alice's stored authority and leave her state alone.
    #[test]
    fn attacker_update_under_its_own_key_is_refused() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        let attacker = Keypair::generate();
        reg.apply_identity_op(&create(&alice, 1)).unwrap();
        let before = state_of(&reg, &alice.did());
        let forged = op_signed_by(
            &attacker,
            &alice.did(),
            attacker.public_key(),
            IdentityOpKind::Update,
            2,
            Some("https://attacker.example.net"),
        );
        assert!(verify_identity_op_as(&forged, attacker.public_key()).is_ok());
        let err = reg.apply_identity_op(&forged).unwrap_err();
        assert_eq!(err.code, Code::INVALID_SIGNATURE);
        assert_eq!(state_of(&reg, &alice.did()), before);
    }

    /// D01 witness: the same forgery as a revoke — the op carries the
    /// attacker's key, verifies under it, and must not end Alice's identity.
    #[test]
    fn attacker_revoke_under_its_own_key_is_refused() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        let attacker = Keypair::generate();
        reg.apply_identity_op(&create(&alice, 1)).unwrap();
        let before = state_of(&reg, &alice.did());
        let forged = op_signed_by(
            &attacker,
            &alice.did(),
            attacker.public_key(),
            IdentityOpKind::Revoke,
            2,
            None,
        );
        let err = reg.apply_identity_op(&forged).unwrap_err();
        assert_eq!(err.code, Code::INVALID_SIGNATURE);
        assert_eq!(state_of(&reg, &alice.did()), before);
        assert!(reg.resolve(&alice.did()).is_some());
    }

    /// D01 witness: a revoke ordered behind a later accepted sequence is a
    /// stale authority claim. It is signed by the right key and still refused.
    #[test]
    fn stale_revoke_is_refused() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        reg.apply_identity_op(&create(&alice, 1)).unwrap();
        let update = op_signed_by(
            &alice,
            &alice.did(),
            alice.public_key(),
            IdentityOpKind::Update,
            5,
            None,
        );
        reg.apply_identity_op(&update).unwrap();
        let before = state_of(&reg, &alice.did());
        let stale = op_signed_by(
            &alice,
            &alice.did(),
            alice.public_key(),
            IdentityOpKind::Revoke,
            2,
            None,
        );
        let err = reg.apply_identity_op(&stale).unwrap_err();
        assert_eq!(err.code, Code::DUPLICATE_TRANSACTION);
        assert_eq!(state_of(&reg, &alice.did()), before);
    }

    /// A body whose signature does not cover it is refused at the registry, so
    /// an unverified op replayed from a log cannot create an identity.
    #[test]
    fn unsigned_create_is_refused() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        let mut op = create(&alice, 1);
        op.service_endpoint = Some("https://tampered.example".to_string());
        let err = reg.apply_identity_op(&op).unwrap_err();
        assert_eq!(err.code, Code::INVALID_SIGNATURE);
        assert!(!reg.contains(&alice.did()));
    }

    #[test]
    fn resolve_document_has_correct_structure() {
        let mut reg = IdentityRegistry::new();
        let alice = Keypair::generate();
        reg.apply_identity_op(&create(&alice, 1)).unwrap();
        let doc = reg.resolve_document(&alice.did()).unwrap();
        assert_eq!(doc.id, alice.did());
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
