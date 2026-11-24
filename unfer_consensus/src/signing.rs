use curve25519_dalek::ristretto::CompressedRistretto;
use curve25519_dalek::scalar::Scalar;
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use unfer_protocol::distributed::{preimage, SigningDomain, SignatureVersion};
use unfer_protocol::{Code, ConsensusTransaction, Diagnostic, Severity};

/// The signing domain every new consensus-transaction signature covers. A
/// signature made here is not a signature over the same transaction bytes in
/// another deployment, audience, schema or context.
pub const CONSENSUS_SIGNING_DOMAIN: SigningDomain<'static> = SigningDomain {
    deployment: "unfer",
    audience: "unfer-consensus",
    schema: "consensus-transaction",
    context: "consensus",
};

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

/// The unsigned canonical frame of a transaction: the same deterministic JSON
/// the legacy construction hashed, before any hashing. D10 signs this frame
/// under a domain prefix; the legacy preimage is `sha256(frame)`.
pub fn canonical_frame(tx: &ConsensusTransaction) -> Vec<u8> {
    match tx {
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
        ConsensusTransaction::CertificateOp(op) => {
            let mut o = op.clone();
            o.signature = [0u8; 64];
            serde_json::to_vec(&ConsensusTransaction::CertificateOp(o)).unwrap()
        }
        ConsensusTransaction::AuctionOp(op) => {
            let mut o = op.clone();
            o.signature = [0u8; 64];
            serde_json::to_vec(&ConsensusTransaction::AuctionOp(o)).unwrap()
        }
        ConsensusTransaction::MathBondOp(op) => {
            let mut o = op.clone();
            o.signature = [0u8; 64];
            serde_json::to_vec(&ConsensusTransaction::MathBondOp(o)).unwrap()
        }
        ConsensusTransaction::MarketOp(op) => {
            let mut o = op.clone();
            o.signature = [0u8; 64];
            serde_json::to_vec(&ConsensusTransaction::MarketOp(o)).unwrap()
        }
        ConsensusTransaction::AttributionOp(op) => {
            let mut o = op.clone();
            o.signature = [0u8; 64];
            serde_json::to_vec(&ConsensusTransaction::AttributionOp(o)).unwrap()
        }
    }
}

/// The pre-D10 preimage: `sha256(canonical_frame(tx))`, no domain at all.
/// Still produced (and accepted) so already-signed history keeps verifying.
pub fn canonical_bytes(tx: &ConsensusTransaction) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    Sha256::digest(canonical_frame(tx)).to_vec()
}

/// The preimage a signature of `version` over `tx` covers under `domain`.
///
/// `Err` only when the frame or a domain field is past its bound; callers turn
/// that into a refusal rather than signing something they cannot describe.
pub fn signing_message(
    version: SignatureVersion,
    domain: &SigningDomain<'_>,
    tx: &ConsensusTransaction,
) -> Result<Vec<u8>, Diagnostic> {
    preimage(version, domain, &canonical_frame(tx)).map_err(|error| {
        Diagnostic::new(
            Code::INVALID_SIGNATURE,
            format!("transaction cannot be framed for signing: {error}"),
            Severity::Error,
        )
    })
}

/// Every preimage a valid signature over `tx` may cover, most-preferred first:
/// the domain-separated frame, then the legacy `sha256(frame)` so signatures
/// made before D10 still verify. A verifier tries them in order and a signer
/// only ever produces the first.
pub fn accepted_messages(
    domain: &SigningDomain<'_>,
    tx: &ConsensusTransaction,
) -> Vec<Vec<u8>> {
    let mut messages = Vec::with_capacity(2);
    if let Ok(message) = signing_message(SignatureVersion::DomainSeparated, domain, tx) {
        messages.push(message);
    }
    messages.push(canonical_bytes(tx));
    messages
}

/// Sign a transaction under the consensus signing domain (D10).
///
/// An unframeable transaction is a caller bug that must not circulate as a
/// zeroed signature, so it panics here instead of failing later at a verifier
/// far from the mistake.
pub fn sign_transaction(tx: &mut ConsensusTransaction, keypair: &Keypair) {
    let msg = signing_message(SignatureVersion::DomainSeparated, &CONSENSUS_SIGNING_DOMAIN, tx)
        .expect("consensus transaction exceeds the signature frame bound");
    let sig = keypair.sign(&msg);
    set_signature(tx, sig);
}

/// Sign a transaction the pre-D10 way (`sha256(frame)`, no domain). Only for
/// reproducing or witnessing legacy history; new signatures must use
/// [`sign_transaction`].
pub fn sign_transaction_legacy(tx: &mut ConsensusTransaction, keypair: &Keypair) {
    let msg = canonical_bytes(tx);
    let sig = keypair.sign(&msg);
    set_signature(tx, sig);
}

fn set_signature(tx: &mut ConsensusTransaction, sig: [u8; 64]) {
    match tx {
        ConsensusTransaction::IdentityOp(op) => op.signature = sig,
        ConsensusTransaction::SessionOp(op) => op.signature = sig,
        ConsensusTransaction::ContentOp(op) => op.signature = sig,
        ConsensusTransaction::CertificateOp(op) => op.signature = sig,
        ConsensusTransaction::AuctionOp(op) => op.signature = sig,
        ConsensusTransaction::MathBondOp(op) => op.signature = sig,
        ConsensusTransaction::MarketOp(op) => op.signature = sig,
        ConsensusTransaction::AttributionOp(op) => op.signature = sig,
    }
}

/// Verify an Arctic threshold signature over a transaction's canonical bytes.
///
/// The 64-byte `signature` is the Arctic aggregate `(RistrettoPoint, Scalar)`
/// (32 + 32 bytes — exactly the size of every op's `signature` field). The
/// `pubkey` is the compressed group Ristretto point. Reuses the sibling
/// `dynamic-arctic` crate's `arctic_core::verify` (the same determinism as
/// `verify_transaction`: identical log in, identical verdict out).
///
/// Returns `false` on any malformed input (bad point, bad scalar, wrong
/// message) — never panics.
pub fn verify_arctic_threshold(
    pubkey: [u8; 32],
    threshold: u32,
    total: u32,
    tx: &ConsensusTransaction,
    signature: &[u8; 64],
) -> bool {
    let group_pk = match CompressedRistretto::from_slice(&pubkey) {
        Ok(pk) => match pk.decompress() {
            Some(pk) => pk,
            None => return false,
        },
        Err(_) => return false,
    };
    let r_point = match CompressedRistretto::from_slice(&signature[..32]) {
        Ok(r) => match r.decompress() {
            Some(r) => r,
            None => return false,
        },
        Err(_) => return false,
    };
    // curve25519-dalek 4.x returns a CtOption; `is_some` unwraps the constant-time
    // choice without branching on secret data.
    let z_bytes: [u8; 32] = signature[32..].try_into().unwrap();
    let z = match Scalar::from_canonical_bytes(z_bytes).into() {
        Some(z) => z,
        None => return false,
    };
    // Threshold/total are informational here (the aggregate signature already
    // commits to the coalition size); the mathematical check is the Schnorr
    // verification against the group key.
    let _ = (threshold, total);
    accepted_messages(&CONSENSUS_SIGNING_DOMAIN, tx)
        .iter()
        .any(|msg| arctic::arctic_core::verify(&group_pk, msg, &(r_point, z)))
}

/// Verify an `IdentityOp` against a caller-supplied key instead of the key the
/// operation carries.
///
/// `verify_transaction` has to trust `op.signing_key` for identity ops because
/// it has no state; that is fine at ingress, where the DID is unknown, and wrong
/// at the ordered application point, where the DID already has an authority.
/// The registry calls this with the *stored* key so an update or revoke signed
/// by an attacker's key cannot be authenticated by that same key.
pub fn verify_identity_op_as(
    op: &unfer_protocol::IdentityOp,
    pubkey: [u8; 32],
) -> Result<(), Diagnostic> {
    let tx = ConsensusTransaction::IdentityOp(op.clone());
    let verifying_key = VerifyingKey::from_bytes(&pubkey).map_err(|_| {
        Diagnostic::new(
            Code::INVALID_SIGNATURE,
            "invalid ed25519 public key",
            Severity::Error,
        )
    })?;
    let signature = ed25519_dalek::Signature::from_bytes(&op.signature);
    let messages = accepted_messages(&CONSENSUS_SIGNING_DOMAIN, &tx);
    if messages
        .iter()
        .any(|msg| verifying_key.verify(msg, &signature).is_ok())
    {
        return Ok(());
    }
    Err(Diagnostic::new(
        Code::INVALID_SIGNATURE,
        "ed25519 signature verification failed",
        Severity::Error,
    ))
}

/// Verify a transaction under its domain, accepting a legacy signature too.
pub fn verify_transaction_in(
    tx: &ConsensusTransaction,
    domain: &SigningDomain<'_>,
) -> Result<(), Diagnostic> {
    let messages = accepted_messages(domain, tx);
    let sig = tx.signature();
    let pubkey_bytes = match tx {
        ConsensusTransaction::IdentityOp(op) => op.signing_key,
        _ => {
            let did = tx.did();
            let hex_key = did.strip_prefix("did:unfer:").ok_or_else(|| {
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
    if messages
        .iter()
        .any(|msg| verifying_key.verify(msg, &signature).is_ok())
    {
        return Ok(());
    }
    Err(Diagnostic::new(
        Code::INVALID_SIGNATURE,
        "ed25519 signature verification failed",
        Severity::Error,
    ))
}

/// Verify a transaction under the consensus signing domain, accepting a legacy
/// signature made before D10 so signed history keeps verifying.
pub fn verify_transaction(tx: &ConsensusTransaction) -> Result<(), Diagnostic> {
    verify_transaction_in(tx, &CONSENSUS_SIGNING_DOMAIN)
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

    fn identity_tx(kp: &Keypair) -> ConsensusTransaction {
        ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        })
    }

    /// A pre-D10 signature must keep verifying: that is the whole reason the
    /// legacy preimage is still in the accepted set.
    #[test]
    fn a_legacy_signature_still_verifies() {
        let kp = Keypair::generate();
        let mut tx = identity_tx(&kp);
        sign_transaction_legacy(&mut tx, &kp);
        assert_eq!(
            *tx.signature(),
            kp.sign(&canonical_bytes(&tx)),
            "the legacy signer covers sha256(frame), exactly as before D10"
        );
        assert!(verify_transaction(&tx).is_ok(), "legacy history still verifies");
        assert!(verify_identity_op_as(
            match &tx {
                ConsensusTransaction::IdentityOp(op) => op,
                _ => unreachable!(),
            },
            kp.public_key()
        )
        .is_ok());
    }

    /// A new signature covers the domain-separated preimage, so it is not a
    /// signature over the bare frame — and it is refused under another
    /// deployment, audience, schema or context.
    #[test]
    fn a_domain_separated_signature_is_bound_to_its_domain() {
        let kp = Keypair::generate();
        let mut tx = identity_tx(&kp);
        sign_transaction(&mut tx, &kp);

        let frame = canonical_frame(&tx);
        assert_ne!(
            *tx.signature(),
            kp.sign(&canonical_bytes(&tx)),
            "a new signature must not be the legacy sha256(frame) signature"
        );
        assert_eq!(
            *tx.signature(),
            kp.sign(
                &signing_message(SignatureVersion::DomainSeparated, &CONSENSUS_SIGNING_DOMAIN, &tx)
                    .unwrap()
            ),
            "a new signature is over the domain-separated frame"
        );
        assert!(verify_transaction(&tx).is_ok(), "its own domain verifies");

        for (label, other) in [
            (
                "deployment",
                SigningDomain {
                    deployment: "other-deployment",
                    ..CONSENSUS_SIGNING_DOMAIN
                },
            ),
            (
                "audience",
                SigningDomain {
                    audience: "another-service",
                    ..CONSENSUS_SIGNING_DOMAIN
                },
            ),
            (
                "schema",
                SigningDomain {
                    schema: "another-schema",
                    ..CONSENSUS_SIGNING_DOMAIN
                },
            ),
            (
                "context",
                SigningDomain {
                    context: "another-lane",
                    ..CONSENSUS_SIGNING_DOMAIN
                },
            ),
        ] {
            assert!(
                verify_transaction_in(&tx, &other).is_err(),
                "a signature for one {label} must not verify for another"
            );
        }
        assert!(
            !frame.is_empty(),
            "the frame exists and is what the preimage carries"
        );
    }

    /// A signature lifted onto altered bytes is refused, in either layout.
    #[test]
    fn altered_bytes_are_refused_under_both_layouts() {
        let kp = Keypair::generate();
        for legacy in [false, true] {
            let mut tx = identity_tx(&kp);
            if legacy {
                sign_transaction_legacy(&mut tx, &kp);
            } else {
                sign_transaction(&mut tx, &kp);
            }
            if let ConsensusTransaction::IdentityOp(ref mut op) = tx {
                op.seq = 2;
            }
            assert!(
                verify_transaction(&tx).is_err(),
                "tampered bytes must fail (legacy={legacy})"
            );
        }
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
        assert_eq!(
            before, after,
            "canonical bytes must not depend on signature"
        );
    }
}
