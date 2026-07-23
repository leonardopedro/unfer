use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use unfer_protocol::{Code, ConsensusTransaction, Diagnostic, Severity};

#[derive(Debug, Clone)]
pub struct Keypair {
    signing: SigningKey,
}

impl Keypair {
    pub fn generate() -> Self {
        let mut csprng = OsRng;
        Self {
            signing: SigningKey::generate(&mut csprng),
        }
    }

    pub fn from_bytes(secret: &[u8; 32]) -> Self {
        Self {
            signing: SigningKey::from_bytes(secret),
        }
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.signing.verifying_key().to_bytes()
    }

    pub fn did(&self) -> String {
        format!("did:unfer:{}", hex::encode(self.public_key()))
    }

    pub fn sign(&self, msg: &[u8]) -> [u8; 64] {
        self.signing.sign(msg).to_bytes()
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing.verifying_key()
    }
}

pub fn canonical_bytes(tx: &ConsensusTransaction) -> Vec<u8> {
    let mut unsigned = match tx {
        ConsensusTransaction::IdentityOp(op) => {
            let mut o = op.clone();
            o.signature = [0u8; 64];
            serde_json::to_vec(&ConsensusTransaction::IdentityOp(o)).unwrap()
        }
        ConsensusTransaction::SessionOp(op) => {
            let mut o = op.clone();
            o.signature = [0u8; 64];
            serde_json::to_vec(&ConsensusTransaction::SessionOp(o)).unwrap()
        }
        ConsensusTransaction::ContentOp(op) => {
            let mut o = op.clone();
            o.signature = [0u8; 64];
            serde_json::to_vec(&ConsensusTransaction::ContentOp(o)).unwrap()
        }
    };
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(&unsigned);
    unsigned = hash.to_vec();
    unsigned
}

pub fn sign_transaction(tx: &mut ConsensusTransaction, keypair: &Keypair) {
    let msg = canonical_bytes(tx);
    let sig = keypair.sign(&msg);
    match tx {
        ConsensusTransaction::IdentityOp(op) => op.signature = sig,
        ConsensusTransaction::SessionOp(op) => op.signature = sig,
        ConsensusTransaction::ContentOp(op) => op.signature = sig,
    }
}

pub fn verify_transaction(tx: &ConsensusTransaction) -> Result<(), Diagnostic> {
    let msg = canonical_bytes(tx);
    let sig = tx.signature();
    let pubkey_bytes = match tx {
        ConsensusTransaction::IdentityOp(op) => op.signing_key,
        _ => {
            let did = tx.did();
            let hex_key = did
                .strip_prefix("did:unfer:")
                .ok_or_else(|| {
                    Diagnostic::new(
                        Code::UNKNOWN_DID,
                        format!("malformed DID: {did}"),
                        Severity::Error,
                    )
                })?;
            let bytes = hex::decode(hex_key).map_err(|_| {
                Diagnostic::new(
                    Code::UNKNOWN_DID,
                    format!("invalid hex in DID: {did}"),
                    Severity::Error,
                )
            })?;
            bytes.try_into().map_err(|_| {
                Diagnostic::new(
                    Code::UNKNOWN_DID,
                    format!("DID pubkey is not 32 bytes: {did}"),
                    Severity::Error,
                )
            })?
        }
    };

    let verifying_key = VerifyingKey::from_bytes(&pubkey_bytes).map_err(|_| {
        Diagnostic::new(
            Code::INVALID_SIGNATURE,
            "invalid ed25519 public key",
            Severity::Error,
        )
    })?;

    let signature = ed25519_dalek::Signature::from_bytes(sig);
    verifying_key.verify(&msg, &signature).map_err(|_| {
        Diagnostic::new(
            Code::INVALID_SIGNATURE,
            "ed25519 signature verification failed",
            Severity::Error,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::{IdentityOp, IdentityOpKind};

    #[test]
    fn keypair_generates_unique_keys() {
        let kp1 = Keypair::generate();
        let kp2 = Keypair::generate();
        assert_ne!(kp1.public_key(), kp2.public_key());
        assert_ne!(kp1.did(), kp2.did());
    }

    #[test]
    fn did_format_is_correct() {
        let kp = Keypair::generate();
        let did = kp.did();
        assert!(did.starts_with("did:unfer:"));
        let hex_part = &did["did:unfer:".len()..];
        assert_eq!(hex_part.len(), 64);
        assert!(hex::decode(hex_part).is_ok());
    }

    #[test]
    fn sign_and_verify_identity_op() {
        let kp = Keypair::generate();
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        sign_transaction(&mut tx, &kp);
        assert!(verify_transaction(&tx).is_ok());
    }

    #[test]
    fn tampered_signature_fails_verification() {
        let kp = Keypair::generate();
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        sign_transaction(&mut tx, &kp);
        if let ConsensusTransaction::IdentityOp(ref mut op) = tx {
            op.signature[0] ^= 0xFF;
        }
        assert!(verify_transaction(&tx).is_err());
    }

    #[test]
    fn wrong_key_fails_verification() {
        let kp1 = Keypair::generate();
        let kp2 = Keypair::generate();
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp1.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp1.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        sign_transaction(&mut tx, &kp2);
        assert!(verify_transaction(&tx).is_err());
    }

    #[test]
    fn canonical_bytes_zeroes_signature() {
        let kp = Keypair::generate();
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        let before = canonical_bytes(&tx);
        sign_transaction(&mut tx, &kp);
        let after = canonical_bytes(&tx);
        assert_eq!(before, after, "canonical bytes must not depend on signature");
    }
}
