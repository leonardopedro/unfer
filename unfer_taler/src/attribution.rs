//! Attribution carbon credit settlement (Open Badges + Taler micropayments).
//!
//! The operator-side service that turns a deterministic
//! [`AttributionLedger`] approval into settled value and a verifiable badge —
//! the exchange analogue of [`AuctionService`](crate::auction::AuctionService)
//! and the Phase-4 [`EscrowService`](unfer_consensus::escrow::EscrowService).
//!
//! The Adidas/Yeezy pattern: Author A (Adidas) pays Author B (Kanye) for the
//! right to publicly claim that A's item derives from B's work and that B
//! approves that attribution. Issuing Attribution Badges is automated with
//! GNU Taler micropayments:
//!
//! - **Offer** escrows Author A's fee e-coin (face value = the negotiated fee)
//!   into a deterministic per-credit escrow DID derived from the operator's
//!   master key — a peer `ConsensusNode` replaying the log can regenerate the
//!   same DID and verify the same state.
//! - **Approve** is Author B's on-ledger endorsement; the same call releases
//!   the escrowed fee to B. No value is created or destroyed: the audit
//!   (`total_supply`) balances before and after.
//! - **Badge** mints a deterministic Open Badges 3.0 assertion (W3C
//!   Verifiable-Credential data model) for an approved credit — public, or
//!   exclusive to an anonymous viewer identified by the SHA-256 of a random
//!   key their browser generates (the per-visualization badge of a
//!   YouTube-style context: the operator only ever sees the hash). The
//!   assertion is signed by the operator (Ed25519) so viewers can verify both
//!   the author's approval (on-ledger) and the badge's provenance.
//!
//! Every produced op is an ordinary `ConsensusTransaction` (an
//! [`AttributionOp`] or a conserving [`CertificateOp`]) recorded in
//! [`ops`](AttributionService::ops). The same determinism guarantees as the
//! rest of Plan R: identical log in, identical state out.
//!
//! Error codes: UK-7501..UK-7512 (ledger) + the shared ESCROW codes (service),
//! see `unfer_protocol::codes`.

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};

use unfer_consensus::attribution::{AttributionLedger, badge_id};
use unfer_consensus::certs::{CertificateLedger, MintAuthority, commit_coin};
use unfer_consensus::signing::{Keypair, sign_transaction};
use unfer_protocol::{
    AttributionBadgeId, AttributionCreditId, AttributionItem, AttributionOffer, AttributionOp,
    AttributionOpKind, CertId, CertificateOp, CertificateOpKind, Code, CoinRef, ConsensusTransaction,
    Diagnostic, Severity,
};

const ATTRIBUTION_BLINDING: [u8; 32] = [0u8; 32];

/// Lifecycle of one escrowed attribution fee inside the service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributionEscrowState {
    /// The fee e-coin sits in the operator-derived escrow DID (offer made,
    /// awaiting Author B's approval).
    Holding,
    /// The fee moved to Author B on approval. Final.
    Released,
    /// The fee returned to Author A (offer abandoned). Final.
    Refunded,
}

/// Author A's fee, held between offer and approval.
#[derive(Debug, Clone)]
pub struct FeeHold {
    /// The escrowed coin (owned by the deterministic escrow DID).
    pub escrowed: CertId,
    /// Author A's original fee e-coin.
    pub origin: CertId,
    pub credit_id: AttributionCreditId,
    pub author_a: String,
    pub author_b: String,
    pub amount: u64,
    pub state: AttributionEscrowState,
}

/// A deterministic Open Badges 3.0 assertion for one credit.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenBadgeAssertion {
    pub badge_id: AttributionBadgeId,
    /// Canonical (deterministic) OB 3.0 / VC 2.0 JSON-LD document.
    pub json: String,
    /// Ed25519 signature (operator key) over SHA-256 of the canonical JSON.
    /// Hex-encoded; the same credit+recipient always yields the same bytes.
    pub proof_value: String,
    /// RFC 3339 issuance instant, derived deterministically from the
    /// consensus-log approval sequence (no wall-clock).
    pub created: String,
}

/// The operator-side attribution service: runs the deterministic attribution
/// engine, escrows fee e-coins, pays Author B on approval, and mints
/// deterministic Open Badges assertions.
pub struct AttributionService {
    operator: Keypair,
    certs: CertificateLedger,
    attribution: AttributionLedger,
    seqs: HashMap<String, u64>,
    fee_holds: HashMap<[u8; 32], FeeHold>,
    settled: HashSet<[u8; 32]>,
    ops: Vec<ConsensusTransaction>,
}

impl AttributionService {
    pub fn new(operator: Keypair, authority: MintAuthority) -> Self {
        Self {
            operator,
            certs: CertificateLedger::new(authority),
            attribution: AttributionLedger::new(),
            seqs: HashMap::new(),
            fee_holds: HashMap::new(),
            settled: HashSet::new(),
            ops: Vec::new(),
        }
    }

    pub fn operator_did(&self) -> String {
        self.operator.did()
    }

    pub fn attribution(&self) -> &AttributionLedger {
        &self.attribution
    }

    pub fn certs(&self) -> &CertificateLedger {
        &self.certs
    }

    pub fn fee_hold(&self, escrowed: &CertId) -> Option<&FeeHold> {
        self.fee_holds.get(&escrowed.0)
    }

    /// Every signed transaction this service produced or observed, in order.
    /// Feed these to a `ConsensusNode` to replay the identical state.
    pub fn ops(&self) -> &[ConsensusTransaction] {
        &self.ops
    }

    /// Feed an external op (e.g. the authority's mint that created Author A's
    /// fee e-coin) into the mirror ledgers.
    pub fn observe(&mut self, tx: ConsensusTransaction) -> Result<(), Diagnostic> {
        self.apply_transaction(&tx)?;
        self.ops.push(tx);
        Ok(())
    }

    /// Author A registers a work item they own (content-addressed).
    pub fn register_item(
        &mut self,
        author: &Keypair,
        item: &AttributionItem,
    ) -> Result<(), Diagnostic> {
        self.apply_attribution(author, AttributionOpKind::RegisterItem { item: item.clone() })?;
        Ok(())
    }

    /// Author A offers Author B a fee for an attribution credit. The fee
    /// e-coin `funding` (face value = `offer.fee`, already in A's hands) is
    /// rowed into the deterministic per-credit escrow DID *before* the signed
    /// offer is emitted, exactly as the attribution ledger's contract
    /// requires; a failed offer leaves A's coin untouched.
    pub fn offer(
        &mut self,
        author_a: &Keypair,
        offer: &AttributionOffer,
        funding: CertId,
    ) -> Result<AttributionCreditId, Diagnostic> {
        let original_owner = self
            .attribution
            .item_owner(&offer.original_item.item_hash)
            .ok_or_else(|| {
                Diagnostic::new(
                    Code::ATTRIBUTION_ITEM_UNKNOWN,
                    "the original item was never registered",
                    Severity::Error,
                )
            })?
            .to_string();
        let id = unfer_consensus::attribution::credit_id(offer, &author_a.did(), &original_owner);

        // The fee coin must exist and its face value must be the fee.
        let stored = self.certs.utxo(&funding).map(|c| c.amount).ok_or_else(|| {
            Diagnostic::new(
                Code::ESCROW_UNKNOWN,
                "the fee e-coin does not exist",
                Severity::Error,
            )
        })?;
        if stored != offer.fee {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_FEE_MISMATCH,
                format!(
                    "fee e-coin face value {stored} must equal the negotiated fee {}",
                    offer.fee
                ),
                Severity::Error,
            ));
        }

        // Row the fee into the escrow DID, then emit the offer op. The offer
        // itself is validated by the ledger against the recorded items.
        let escrowed = self.escrow_fee(author_a, &id, &original_owner, offer.fee, funding)?;
        let applied = self.apply_attribution(
            author_a,
            AttributionOpKind::OfferAttribution { offer: offer.clone() },
        );
        if let Err(e) = applied {
            // The offer was refused: give the fee back so nothing is stranded.
            let _ = self.refund_fee(escrowed, &author_a.did());
            return Err(e);
        }
        Ok(id)
    }

    /// Author B approves an offered credit. The on-ledger transition to
    /// `Approved` happens first; the same call releases the escrowed fee to B.
    pub fn approve(
        &mut self,
        author_b: &Keypair,
        credit_id: &AttributionCreditId,
    ) -> Result<(), Diagnostic> {
        self.apply_attribution(author_b, AttributionOpKind::Approve { credit_id: *credit_id })?;
        // Release every fee held for this credit to Author B.
        let holds: Vec<CertId> = self
            .fee_holds
            .values()
            .filter(|h| h.credit_id == *credit_id && h.state == AttributionEscrowState::Holding)
            .map(|h| h.escrowed)
            .collect();
        for escrowed in holds {
            self.release_fee(escrowed, &author_b.did())?;
        }
        Ok(())
    }

    /// Author B revokes an approved credit. The fee was already paid at
    /// approval (the endorsement was bought); revocation only withdraws the
    /// endorsement and stops new badges.
    pub fn revoke(
        &mut self,
        author_b: &Keypair,
        credit_id: &AttributionCreditId,
    ) -> Result<(), Diagnostic> {
        self.apply_attribution(author_b, AttributionOpKind::Revoke { credit_id: *credit_id })?;
        Ok(())
    }

    /// Mint a deterministic Open Badges 3.0 assertion for an approved credit.
    /// `viewer` is `None` for the public badge or `Some(sha256)` of a random
    /// key generated by the viewer's browser — the per-visualization badge,
    /// exclusive to that anonymous viewer. The assertion bytes are a pure
    /// function of (credit, recipient, operator key, approval seq).
    pub fn issue_badge(
        &mut self,
        requester: &Keypair,
        credit_id: &AttributionCreditId,
        viewer: Option<[u8; 32]>,
    ) -> Result<OpenBadgeAssertion, Diagnostic> {
        self.apply_attribution(
            requester,
            AttributionOpKind::IssueBadge {
                credit_id: *credit_id,
                viewer,
            },
        )?;
        let report = self.attribution.report(credit_id).ok_or_else(|| {
            Diagnostic::new(
                Code::ATTRIBUTION_UNKNOWN_CREDIT,
                "unknown credit id",
                Severity::Error,
            )
        })?;
        let approve_seq = report.approve_seq.unwrap_or(0);
        let bid = badge_id(credit_id, viewer);
        let created = rfc3339_from_seq(approve_seq);

        let json = build_assertion(&self.operator, &bid, credit_id, &report, viewer, &created);
        // Sign SHA-256 of the canonical document (deterministic; the
        // commitment is content-addressed — see docs/ATTRIBUTION.md).
        let digest = Sha256::digest(json.as_bytes());
        let sig = self.operator.sign(&digest);
        Ok(OpenBadgeAssertion {
            badge_id: bid,
            json,
            proof_value: hex::encode(sig),
            created,
        })
    }

    /// The deterministic escrow DID holding Author A's fee for `credit_id`:
    /// only the operator can regenerate it (master key), and only for this
    /// exact credit.
    pub fn fee_did(&self, credit_id: &AttributionCreditId) -> String {
        let mut ctx = Sha256::new();
        ctx.update(b"unfer:attribution:fee:v1");
        ctx.update(self.operator.public_key());
        ctx.update(credit_id.0);
        let key = Keypair::from_bytes(&ctx.finalize().into());
        key.did()
    }

    /// Row `origin` (A's fee e-coin) into the escrow DID for `credit_id`.
    fn escrow_fee(
        &mut self,
        author_a: &Keypair,
        credit_id: &AttributionCreditId,
        author_b: &str,
        amount: u64,
        origin: CertId,
    ) -> Result<CertId, Diagnostic> {
        if self.fee_holds.values().any(|h| h.origin == origin) {
            return Err(Diagnostic::new(
                Code::ESCROW_ALREADY_SETTLED,
                "this fee e-coin is already escrowed",
                Severity::Error,
            ));
        }
        let escrow_did = self.fee_did(credit_id);
        let escrowed = commit_coin(amount, &escrow_did, &ATTRIBUTION_BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: origin,
                amount,
                owner: author_a.did(),
            }],
            outputs: vec![CoinRef {
                coin_id: escrowed,
                amount,
                owner: escrow_did,
            }],
        };
        let tx = self.build_cert(&author_a.did(), kind);
        self.emit_cert(tx, author_a)?;
        self.fee_holds.insert(
            escrowed.0,
            FeeHold {
                escrowed,
                origin,
                credit_id: *credit_id,
                author_a: author_a.did(),
                author_b: author_b.to_string(),
                amount,
                state: AttributionEscrowState::Holding,
            },
        );
        Ok(escrowed)
    }

    /// Fee → Author B (approval payout).
    fn release_fee(&mut self, escrowed: CertId, recipient: &str) -> Result<CertId, Diagnostic> {
        let hold = self.fee_holds.get(&escrowed.0).ok_or_else(|| {
            Diagnostic::new(Code::ESCROW_UNKNOWN, "fee escrow not found", Severity::Error)
        })?;
        if self.settled.contains(&escrowed.0) || hold.state != AttributionEscrowState::Holding {
            return Err(Diagnostic::new(
                Code::ESCROW_ALREADY_SETTLED,
                "fee escrow already settled",
                Severity::Error,
            ));
        }
        let out = commit_coin(hold.amount, recipient, &ATTRIBUTION_BLINDING);
        let escrow_did = self.fee_did(&hold.credit_id);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: escrowed,
                amount: hold.amount,
                owner: escrow_did.clone(),
            }],
            outputs: vec![CoinRef {
                coin_id: out,
                amount: hold.amount,
                owner: recipient.to_string(),
            }],
        };
        let key = self.fee_key(&hold.credit_id);
        let tx = self.build_cert(&escrow_did, kind);
        self.emit_cert(tx, &key)?;
        if let Some(h) = self.fee_holds.get_mut(&escrowed.0) {
            h.state = AttributionEscrowState::Released;
        }
        self.settled.insert(escrowed.0);
        Ok(out)
    }

    /// Fee → Author A (abandoned offer refund).
    fn refund_fee(&mut self, escrowed: CertId, recipient: &str) -> Result<CertId, Diagnostic> {
        let hold = self.fee_holds.get(&escrowed.0).ok_or_else(|| {
            Diagnostic::new(Code::ESCROW_UNKNOWN, "fee escrow not found", Severity::Error)
        })?;
        if self.settled.contains(&escrowed.0) || hold.state != AttributionEscrowState::Holding {
            return Err(Diagnostic::new(
                Code::ESCROW_ALREADY_SETTLED,
                "fee escrow already settled",
                Severity::Error,
            ));
        }
        let out = commit_coin(hold.amount, recipient, &ATTRIBUTION_BLINDING);
        let escrow_did = self.fee_did(&hold.credit_id);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: escrowed,
                amount: hold.amount,
                owner: escrow_did.clone(),
            }],
            outputs: vec![CoinRef {
                coin_id: out,
                amount: hold.amount,
                owner: recipient.to_string(),
            }],
        };
        let key = self.fee_key(&hold.credit_id);
        let tx = self.build_cert(&escrow_did, kind);
        self.emit_cert(tx, &key)?;
        if let Some(h) = self.fee_holds.get_mut(&escrowed.0) {
            h.state = AttributionEscrowState::Refunded;
        }
        self.settled.insert(escrowed.0);
        Ok(out)
    }

    /// Deterministic per-credit escrow spending key: only the operator can
    /// regenerate it, and only for this exact credit.
    fn fee_key(&self, credit_id: &AttributionCreditId) -> Keypair {
        let mut ctx = Sha256::new();
        ctx.update(b"unfer:attribution:fee:v1");
        ctx.update(self.operator.public_key());
        ctx.update(credit_id.0);
        Keypair::from_bytes(&ctx.finalize().into())
    }

    fn next_seq(&mut self, did: &str) -> u64 {
        let seq = self.seqs.entry(did.to_string()).or_insert(0);
        *seq += 1;
        *seq
    }

    fn build_cert(&mut self, did: &str, kind: CertificateOpKind) -> ConsensusTransaction {
        ConsensusTransaction::CertificateOp(CertificateOp {
            did: did.to_string(),
            kind,
            seq: self.next_seq(did),
            signature: [0u8; 64],
        })
    }

    fn emit_cert(&mut self, mut tx: ConsensusTransaction, signer: &Keypair) -> Result<(), Diagnostic> {
        sign_transaction(&mut tx, signer);
        self.apply_transaction(&tx)?;
        self.ops.push(tx);
        Ok(())
    }

    fn apply_attribution(
        &mut self,
        signer: &Keypair,
        kind: AttributionOpKind,
    ) -> Result<(), Diagnostic> {
        let did = signer.did();
        // The ledger receives the CONSENSUS log position (the position this op
        // will occupy in `ops()`, which a peer `ConsensusNode` replays 1-based)
        // — exactly the seq the node passes in `apply_transaction`, so a
        // replayed `approve_seq`/`revoke_seq` (and thus the badge date) match.
        // The op's own wire `seq` stays the per-DID submitter counter.
        let consensus_seq = self.ops.len() as u64 + 1;
        let op_seq = self.next_seq(&did);
        self.attribution.apply_op(&did, &kind, consensus_seq)?;
        let mut tx = ConsensusTransaction::AttributionOp(AttributionOp {
            did,
            kind,
            seq: op_seq,
            signature: [0u8; 64],
        });
        sign_transaction(&mut tx, signer);
        self.ops.push(tx);
        Ok(())
    }

    fn apply_transaction(&mut self, tx: &ConsensusTransaction) -> Result<(), Diagnostic> {
        match tx {
            ConsensusTransaction::CertificateOp(op) => {
                self.certs.apply_op(&op.did, &op.kind, op.seq).map(|_| ())
            }
            ConsensusTransaction::AttributionOp(op) => {
                // Mirrors the node: observed attribution ops are applied at the
                // consensus position they will occupy when the log is replayed.
                self.attribution
                    .apply_op(&op.did, &op.kind, self.ops.len() as u64 + 1)
            }
            other => Err(Diagnostic::new(
                Code::INTERNAL,
                format!("attribution service only observes certificate/attribution ops, got {other:?}"),
                Severity::Error,
            )),
        }
    }
}

/// Deterministic RFC 3339 instant: `2020-01-01T00:00:00Z` plus `approve_seq`
/// seconds. No wall-clock — replaying the log yields the identical date.
fn rfc3339_from_seq(seq: u64) -> String {
    const EPOCH: i64 = 1_577_836_800; // 2020-01-01T00:00:00Z
    let unix = EPOCH + seq as i64;
    let days = unix.div_euclid(86_400);
    let secs = unix.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Howard Hinnant's `days_from_civil` inverse: day count → (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Build the canonical Open Badges 3.0 / W3C VC 2.0 assertion document.
///
/// Deterministic: every field is a pure function of ledger state and the
/// operator key; `serde_json` serializes map keys in sorted order, so the
/// same (credit, recipient) always produces byte-identical JSON.
fn build_assertion(
    operator: &Keypair,
    badge: &AttributionBadgeId,
    credit_id: &AttributionCreditId,
    report: &unfer_protocol::AttributionReport,
    viewer: Option<[u8; 32]>,
    created: &str,
) -> String {
    let offer = &report.offer;
    let json = serde_json::json!({
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://purl.imsglobal.org/spec/ob/v3p0/context-3.0.3.json"
        ],
        "id": format!("urn:unfer:attribution:badge:{}", hex::encode(badge.0)),
        "type": ["VerifiableCredential", "OpenBadgeCredential"],
        "issuer": {
            "id": operator.did(),
            "type": "Profile",
            "name": "unfer attribution operator"
        },
        "issuanceDate": created,
        "credentialSubject": {
            "id": format!("urn:unfer:attribution:credit:{}", hex::encode(credit_id.0)),
            "type": "AttributionCredit",
            "derived": {
                "id": format!("urn:unfer:item:{}", hex::encode(offer.derived_item.item_hash)),
                "name": offer.derived_item.title,
                "owner": report.author_a
            },
            "original": {
                "id": format!("urn:unfer:item:{}", hex::encode(offer.original_item.item_hash)),
                "name": offer.original_item.title,
                "owner": report.author_b
            },
            "approvedBy": report.author_b,
            "fee": offer.fee,
            "context": offer.context,
            "exclusive": offer.exclusive,
            "viewer": viewer.map(hex::encode)
        },
        "proof": {
            "type": "Ed25519Signature2020",
            "created": created,
            "verificationMethod": format!("{}#attribution", operator.did()),
            "proofPurpose": "assertionMethod",
            // Filled in by the caller after signing the canonical document.
            "proofValue": ""
        }
    });
    serde_json::to_string(&json).expect("assertion serialization cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_consensus::engine::LocalConsensus;
    use unfer_consensus::node::ConsensusNode;
    use unfer_protocol::AttributionState;

    fn authority() -> Keypair {
        Keypair::generate()
    }

    fn mint_to(
        service: &mut AttributionService,
        auth: &Keypair,
        owner: &str,
        amount: u64,
        blinding: [u8; 32],
    ) -> CertId {
        let mut tx = ConsensusTransaction::CertificateOp(CertificateOp {
            did: auth.did(),
            kind: CertificateOpKind::Mint {
                amount,
                owner: owner.to_string(),
                blinding,
                source: None,
            },
            seq: 1,
            signature: [0u8; 64],
        });
        sign_transaction(&mut tx, auth);
        service.observe(tx).unwrap();
        commit_coin(amount, owner, &blinding)
    }

    fn item(title: &str, byte: u8) -> AttributionItem {
        AttributionItem {
            item_hash: [byte; 32],
            title: title.to_string(),
            url: None,
        }
    }

    fn offer(derived: &AttributionItem, original: &AttributionItem, fee: u64, exclusive: bool) -> AttributionOffer {
        AttributionOffer {
            derived_item: derived.clone(),
            original_item: original.clone(),
            fee,
            context: "Yeezy line, 2023 collection".to_string(),
            exclusive,
        }
    }

    fn setup(
        svc: &mut AttributionService,
        auth: &Keypair,
        adidas: &Keypair,
        kanye: &Keypair,
        fee: u64,
        exclusive: bool,
    ) -> (AttributionItem, AttributionItem, AttributionCreditId, CertId) {
        let shoe = item("Yeezy Boost 350 V2", 1);
        let sketch = item("Kanye's 2015 sketch", 2);
        svc.register_item(adidas, &shoe).unwrap();
        svc.register_item(kanye, &sketch).unwrap();
        let funding = mint_to(svc, auth, &adidas.did(), fee, [5u8; 32]);
        let id = svc.offer(adidas, &offer(&shoe, &sketch, fee, exclusive), funding).unwrap();
        (shoe, sketch, id, funding)
    }

    #[test]
    fn full_lifecycle_escrow_approve_payout_badge_revoke() {
        let auth = authority();
        let adidas = Keypair::generate();
        let kanye = Keypair::generate();
        let mut svc = AttributionService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let (shoe, _sketch, id, funding) = setup(&mut svc, &auth, &adidas, &kanye, 5000, true);

        // After the offer, the fee sits in the escrow DID, not with A or B.
        assert_eq!(svc.certs().total_supply(), 5000);
        assert_eq!(svc.certs().coins_of(&adidas.did()).iter().map(|c| c.amount).sum::<u64>(), 0);
        assert_eq!(svc.certs().coins_of(&kanye.did()).iter().map(|c| c.amount).sum::<u64>(), 0);
        assert_eq!(svc.attribution().report(&id).unwrap().state, AttributionState::Offered);

        // Only B approves; approval pays the fee out to B.
        let err = svc.approve(&adidas, &id).unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_NOT_AUTHOR);
        svc.approve(&kanye, &id).unwrap();
        assert_eq!(svc.attribution().report(&id).unwrap().state, AttributionState::Approved);
        assert_eq!(svc.certs().coins_of(&kanye.did()).iter().map(|c| c.amount).sum::<u64>(), 5000);
        assert_eq!(svc.certs().coins_of(&adidas.did()).iter().map(|c| c.amount).sum::<u64>(), 0);
        // No value created or destroyed.
        assert_eq!(svc.certs().total_supply(), 5000);

        // Badges mint deterministically, public + anonymous per-view.
        let viewer = [9u8; 32];
        let pub_badge = svc.issue_badge(&adidas, &id, None).unwrap();
        let anon_badge = svc.issue_badge(&adidas, &id, Some(viewer)).unwrap();
        assert_ne!(pub_badge.badge_id, anon_badge.badge_id);
        assert_eq!(pub_badge.badge_id, badge_id(&id, None));
        assert_eq!(anon_badge.badge_id, badge_id(&id, Some(viewer)));
        assert_eq!(pub_badge.created, anon_badge.created, "same approval → same instant");
        // The anonymous badge carries the viewer hash (and only the hash).
        let json: serde_json::Value = serde_json::from_str(&anon_badge.json).unwrap();
        assert_eq!(
            json["credentialSubject"]["viewer"],
            serde_json::Value::String(hex::encode(viewer))
        );
        let pub_json: serde_json::Value = serde_json::from_str(&pub_badge.json).unwrap();
        assert_eq!(pub_json["credentialSubject"]["viewer"], serde_json::Value::Null);
        // Re-issuing the same badge is refused (exactly-once).
        let err = svc.issue_badge(&adidas, &id, None).unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_BADGE_EXISTS);

        // Revoke: only B; the fee stays paid; new badges refused.
        let err = svc.revoke(&adidas, &id).unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_NOT_AUTHOR);
        svc.revoke(&kanye, &id).unwrap();
        assert_eq!(svc.attribution().report(&id).unwrap().state, AttributionState::Revoked);
        let err = svc.issue_badge(&adidas, &id, Some(viewer)).unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_BADGE_REVOKED);

        // The derived item is still registered; the fee coin's origin is spent.
        assert_eq!(svc.attribution().item_owner(&shoe.item_hash), Some(adidas.did().as_str()));
        assert!(svc.certs().utxo(&funding).is_none());
    }

    #[test]
    fn offer_refuses_fee_mismatch_and_unknown_item() {
        let auth = authority();
        let adidas = Keypair::generate();
        let kanye = Keypair::generate();
        let mut svc = AttributionService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let shoe = item("Yeezy Boost 350 V2", 1);
        let sketch = item("Kanye's 2015 sketch", 2);
        svc.register_item(&adidas, &shoe).unwrap();
        svc.register_item(&kanye, &sketch).unwrap();

        // Unknown original item → refused at the service before any escrow.
        let ghost = item("ghost", 9);
        let funding = mint_to(&mut svc, &auth, &adidas.did(), 5000, [1u8; 32]);
        let err = svc.offer(&adidas, &offer(&shoe, &ghost, 5000, false), funding).unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_ITEM_UNKNOWN);

        // Wrong face value → refused, coin untouched.
        let funding = mint_to(&mut svc, &auth, &adidas.did(), 3000, [2u8; 32]);
        let err = svc.offer(&adidas, &offer(&shoe, &sketch, 5000, false), funding).unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_FEE_MISMATCH);
        assert_eq!(svc.certs().coins_of(&adidas.did()).iter().map(|c| c.amount).sum::<u64>(), 8000);

        // Ledger validation failures (self-attribution) refund the escrow.
        let funding = mint_to(&mut svc, &auth, &adidas.did(), 5000, [3u8; 32]);
        let err = svc.offer(&adidas, &offer(&shoe, &shoe, 5000, false), funding).unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_SELF_ATTRIBUTION);
        // The fee came back to A — nothing stranded.
        assert_eq!(svc.certs().coins_of(&adidas.did()).iter().map(|c| c.amount).sum::<u64>(), 13000);
        assert_eq!(svc.certs().total_supply(), 13000);
    }

    #[test]
    fn badge_assertions_are_deterministic_across_services() {
        let auth = authority();
        let adidas = Keypair::generate();
        let kanye = Keypair::generate();
        let operator = Keypair::generate();
        let mut svc1 = AttributionService::new(operator.clone(), MintAuthority::Only(auth.did()));
        let mut svc2 = AttributionService::new(operator, MintAuthority::Only(auth.did()));
        let (_, _, id1, _) = setup(&mut svc1, &auth, &adidas, &kanye, 700, true);
        let (_, _, id2, _) = setup(&mut svc2, &auth, &adidas, &kanye, 700, true);
        assert_eq!(id1, id2);
        svc1.approve(&kanye, &id1).unwrap();
        svc2.approve(&kanye, &id2).unwrap();
        let viewer = [3u8; 32];
        let b1 = svc1.issue_badge(&adidas, &id1, Some(viewer)).unwrap();
        let b2 = svc2.issue_badge(&adidas, &id2, Some(viewer)).unwrap();
        assert_eq!(b1.badge_id, b2.badge_id);
        assert_eq!(b1.json, b2.json, "byte-identical assertion");
        assert_eq!(b1.proof_value, b2.proof_value, "byte-identical Ed25519 proof");
        assert_eq!(b1.created, b2.created);
    }

    #[test]
    fn replay_converges_the_consensus_node() {
        let auth = authority();
        let adidas = Keypair::generate();
        let kanye = Keypair::generate();
        let viewer = [7u8; 32];
        let mut svc = AttributionService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let (_, _, id, _) = setup(&mut svc, &auth, &adidas, &kanye, 1200, true);
        svc.approve(&kanye, &id).unwrap();
        let badge = svc.issue_badge(&adidas, &id, Some(viewer)).unwrap();

        let engine = LocalConsensus::new();
        let mut node = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(auth.did()),
        );
        for tx in svc.ops() {
            node.submit(tx.clone()).unwrap();
            node.sync().unwrap();
        }
        // The replaying node converges on the identical certificate + attribution state.
        assert_eq!(svc.certs().root(), node.certs().root());
        assert_eq!(svc.certs().total_supply(), node.certs().total_supply());
        let report = node.attribution().report(&id).unwrap();
        assert_eq!(report.state, AttributionState::Approved);
        assert_eq!(report.badges, vec![badge.badge_id]);
        // Deterministic ids match the service's view.
        assert_eq!(node.attribution().root(), svc.attribution().root());
    }
}
