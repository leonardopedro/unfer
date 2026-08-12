//! Phase 4 secondary-market escrow over the certificate ledger.
//!
//! A trusted marketplace operator (the escrow agent) rows a certificate into a
//! deterministic intermediate DID between a buyer and a seller. Every
//! transition is an ordinary [`CertificateOp::Transfer`] on the consensus log:
//!
//! 1. [`EscrowService::hold`] — the buyer transfers the coin into the escrow
//!    DID, derived deterministically from the operator's master key, the two
//!    parties, and the coin. While held, neither the buyer nor the seller can
//!    spend it: only the escrow key (regenerable by the operator) can.
//! 2. [`EscrowService::release`] — on the delivery receipt condition, the
//!    operator moves the coin to the seller.
//! 3. [`EscrowService::refund`] — if the deal falls through, the operator moves
//!    it back to the buyer.
//!
//! An escrow has exactly one outcome: release or refund. Every produced op is
//! recorded in [`ops`](EscrowService::ops), so a peer `ConsensusNode`
//! replaying the log lands on the identical certificate root — the market is
//! fully auditable from the replicated ledger ([`ConsensusNode`](crate::node::ConsensusNode)).

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};

use unfer_protocol::{
    CertId, CertificateOp, CertificateOpKind, Code, CoinRef, ConsensusTransaction, Diagnostic,
    Severity,
};

use crate::certs::{CertificateLedger, MintAuthority, commit_coin};
use crate::signing::{Keypair, sign_transaction};

const ESCROW_BLINDING: [u8; 32] = [0u8; 32];

/// Lifecycle of one escrowed certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscrowState {
    /// The coin sits in the escrow DID, spendable by neither party.
    Holding,
    /// Delivered: the coin moved to the seller. Final.
    Released,
    /// Deal abandoned: the coin returned to the buyer. Final.
    Refunded,
}

/// A certificate held between a buyer and a seller.
#[derive(Debug, Clone)]
pub struct Escrow {
    /// The coin id while it lives in the escrow DID (committed from amount,
    /// escrow owner and zero blinding).
    pub coin_id: CertId,
    /// The coin id the buyer originally owned — the deterministic anchor for
    /// the escrow key and DID.
    pub origin: CertId,
    pub amount: u64,
    pub buyer: String,
    pub seller: String,
    pub state: EscrowState,
}

/// The secondary-market escrow agent. Holds a mirror of the certificate ledger
/// and the marketplace operator's master key, from which each escrow's spending
/// key is derived.
pub struct EscrowService {
    operator: Keypair,
    ledger: CertificateLedger,
    seqs: HashMap<String, u64>,
    escrows: HashMap<[u8; 32], Escrow>,
    settled: HashSet<[u8; 32]>,
    ops: Vec<ConsensusTransaction>,
}

impl EscrowService {
    pub fn new(operator: Keypair, authority: MintAuthority) -> Self {
        Self {
            operator,
            ledger: CertificateLedger::new(authority),
            seqs: HashMap::new(),
            escrows: HashMap::new(),
            settled: HashSet::new(),
            ops: Vec::new(),
        }
    }

    pub fn operator_did(&self) -> String {
        self.operator.did()
    }

    pub fn ledger(&self) -> &CertificateLedger {
        &self.ledger
    }

    pub fn escrow(&self, coin_id: &CertId) -> Option<&Escrow> {
        self.escrows.get(&coin_id.0)
    }

    /// Every signed certificate op this service has produced or observed, in
    /// order. Feed these to a `ConsensusNode` to replay the identical state.
    pub fn ops(&self) -> &[ConsensusTransaction] {
        &self.ops
    }

    /// Feed an external op (e.g. the authority's mint that created the coin)
    /// into the mirror ledger before escrow transitions.
    pub fn observe(&mut self, tx: ConsensusTransaction) -> Result<(), Diagnostic> {
        self.apply_transaction(&tx)?;
        self.ops.push(tx);
        Ok(())
    }

    /// The escrow DID that owns `coin_id` between `buyer` and `seller`.
    pub fn escrow_did(&self, buyer: &str, seller: &str, origin: CertId) -> String {
        self.escrow_key(buyer, seller, origin).did()
    }

    /// Row `coin_id` into escrow: the buyer transfers it to the escrow DID.
    /// The coin must currently be owned by `buyer`.
    pub fn hold(
        &mut self,
        buyer: &Keypair,
        seller_did: &str,
        coin_id: CertId,
        amount: u64,
    ) -> Result<CertId, Diagnostic> {
        let escrow_did = self.escrow_did(&buyer.did(), seller_did, coin_id);
        let escrowed = commit_coin(amount, &escrow_did, &ESCROW_BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id,
                amount,
                owner: buyer.did(),
            }],
            outputs: vec![CoinRef {
                coin_id: escrowed,
                amount,
                owner: escrow_did,
            }],
        };
        let tx = self.build_op(&buyer.did(), kind);
        self.emit(tx, buyer)?;
        self.escrows.insert(
            escrowed.0,
            Escrow {
                coin_id: escrowed,
                origin: coin_id,
                amount,
                buyer: buyer.did(),
                seller: seller_did.to_string(),
                state: EscrowState::Holding,
            },
        );
        Ok(escrowed)
    }

    /// Settle in the seller's favor (the receipt condition is met): move the
    /// escrowed coin from the escrow DID to the seller. Final state.
    pub fn release(&mut self, escrowed: CertId, seller_did: &str) -> Result<CertId, Diagnostic> {
        self.settle(escrowed, seller_did, EscrowState::Released)
    }

    /// Abandon the deal: return the escrowed coin to the buyer. Final state.
    pub fn refund(&mut self, escrowed: CertId, buyer_did: &str) -> Result<CertId, Diagnostic> {
        self.settle(escrowed, buyer_did, EscrowState::Refunded)
    }

    fn settle(
        &mut self,
        escrowed: CertId,
        recipient: &str,
        outcome: EscrowState,
    ) -> Result<CertId, Diagnostic> {
        let escrow = self
            .escrows
            .get(&escrowed.0)
            .ok_or_else(|| self.diag(Code::ESCROW_UNKNOWN, "coin was never placed in escrow"))?;
        if self.settled.contains(&escrowed.0) {
            return Err(self.diag(
                Code::ESCROW_ALREADY_SETTLED,
                "escrow already has an outcome (release or refund)",
            ));
        }
        if escrow.state != EscrowState::Holding {
            return Err(self.diag(
                Code::ESCROW_NOT_HOLDING,
                "escrow is not in the Holding state",
            ));
        }
        let escrow_did = self.escrow_did(&escrow.buyer, &escrow.seller, escrow.origin);
        let recipient_coin = commit_coin(escrow.amount, recipient, &ESCROW_BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: escrowed,
                amount: escrow.amount,
                owner: escrow_did.clone(),
            }],
            outputs: vec![CoinRef {
                coin_id: recipient_coin,
                amount: escrow.amount,
                owner: recipient.to_string(),
            }],
        };
        let key = self.escrow_key(&escrow.buyer, &escrow.seller, escrow.origin);
        let tx = self.build_op(&escrow_did, kind);
        self.emit(tx, &key)?;
        let missing_escrow = self.diag(Code::ESCROW_UNKNOWN, "escrow vanished during settlement");
        match self.escrows.get_mut(&escrowed.0) {
            Some(e) => e.state = outcome,
            None => return Err(missing_escrow),
        }
        self.settled.insert(escrowed.0);
        Ok(recipient_coin)
    }

    /// Deterministic per-deal spending key: only the marketplace operator can
    /// regenerate it, and only for this exact (buyer, seller, coin) triple.
    fn escrow_key(&self, buyer: &str, seller: &str, coin_id: CertId) -> Keypair {
        let mut ctx = Sha256::new();
        ctx.update(b"unfer:escrow:v1");
        ctx.update(self.operator.public_key());
        ctx.update(buyer.as_bytes());
        ctx.update(seller.as_bytes());
        ctx.update(coin_id.0);
        Keypair::from_bytes(&ctx.finalize().into())
    }

    fn build_op(&mut self, did: &str, kind: CertificateOpKind) -> ConsensusTransaction {
        let seq = self.seqs.entry(did.to_string()).or_insert(0);
        *seq += 1;
        ConsensusTransaction::CertificateOp(CertificateOp {
            did: did.to_string(),
            kind,
            seq: *seq,
            signature: [0u8; 64],
        })
    }

    fn emit(&mut self, mut tx: ConsensusTransaction, signer: &Keypair) -> Result<(), Diagnostic> {
        sign_transaction(&mut tx, signer);
        self.apply_transaction(&tx)?;
        self.ops.push(tx);
        Ok(())
    }

    fn apply_transaction(&mut self, tx: &ConsensusTransaction) -> Result<(), Diagnostic> {
        match tx {
            ConsensusTransaction::CertificateOp(op) => {
                self.ledger.apply_op(&op.did, &op.kind, op.seq)?;
            }
            other => {
                return Err(self.diag(
                    Code::INTERNAL,
                    format!("escrow only observes certificate ops, got {other:?}"),
                ));
            }
        }
        Ok(())
    }

    fn diag(&self, code: Code, msg: impl Into<String>) -> Diagnostic {
        Diagnostic::new(code, msg, Severity::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::LocalConsensus;
    use crate::node::ConsensusNode;
    use unfer_protocol::CertificateOpKind;

    fn authority() -> Keypair {
        Keypair::generate()
    }

    fn mint_to(
        service: &mut EscrowService,
        authority: &Keypair,
        owner: &str,
        amount: u64,
        blinding: [u8; 32],
    ) -> CertId {
        let mut tx = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount,
                owner: owner.to_string(),
                blinding,
                source: None,
            },
            seq: 1,
            signature: [0u8; 64],
        });
        sign_transaction(&mut tx, authority);
        service.observe(tx).unwrap();
        commit_coin(amount, owner, &blinding)
    }

    fn settles(
        auth: &Keypair,
        buyer: &Keypair,
        seller: &str,
        amount: u64,
    ) -> (EscrowService, CertId) {
        let mut service = EscrowService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let raw = mint_to(&mut service, auth, &buyer.did(), amount, [7u8; 32]);
        let held = service.hold(buyer, seller, raw, amount).unwrap();
        (service, held)
    }

    #[test]
    fn hold_interposes_until_settled() {
        let auth = authority();
        let buyer = Keypair::generate();
        let seller = Keypair::generate();
        let (service, held) = settles(&auth, &buyer, &seller.did(), 1000);

        let escrow = service.escrow(&held).unwrap();
        assert_eq!(escrow.state, EscrowState::Holding);
        // Neither party owns the coin while it is in escrow.
        assert!(service.ledger().utxo(&held).is_some());
        assert_ne!(service.ledger().utxo(&held).unwrap().owner, buyer.did());
        assert_ne!(service.ledger().utxo(&held).unwrap().owner, seller.did());
        assert_eq!(
            service.ledger().total_supply(),
            1000,
            "no value created or destroyed"
        );
    }

    #[test]
    fn release_delivers_to_the_seller() {
        let auth = authority();
        let buyer = Keypair::generate();
        let seller = Keypair::generate();
        let (mut service, held) = settles(&auth, &buyer, &seller.did(), 1000);

        let delivered = service.release(held, &seller.did()).unwrap();
        assert_eq!(service.escrow(&held).unwrap().state, EscrowState::Released);
        assert_eq!(
            service.ledger().utxo(&delivered).unwrap().owner,
            seller.did()
        );
        assert!(service.ledger().utxo(&held).is_none());
        assert_eq!(service.ledger().total_supply(), 1000);

        // A second settlement is refused: one outcome per escrow.
        let dup = service.refund(held, &buyer.did()).unwrap_err();
        assert_eq!(dup.code, Code::ESCROW_ALREADY_SETTLED);
    }

    #[test]
    fn refund_returns_to_the_buyer() {
        let auth = authority();
        let buyer = Keypair::generate();
        let seller = Keypair::generate();
        let (mut service, held) = settles(&auth, &buyer, &seller.did(), 500);

        let returned = service.refund(held, &buyer.did()).unwrap();
        assert_eq!(service.escrow(&held).unwrap().state, EscrowState::Refunded);
        assert_eq!(service.ledger().utxo(&returned).unwrap().owner, buyer.did());
        assert_eq!(service.ledger().total_supply(), 500);
    }

    #[test]
    fn replay_reproduces_the_escrow_mirror() {
        let auth = authority();
        let buyer = Keypair::generate();
        let seller = Keypair::generate();
        let mut service = EscrowService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let raw = mint_to(&mut service, &auth, &buyer.did(), 1000, [7u8; 32]);
        let held = service.hold(&buyer, &seller.did(), raw, 1000).unwrap();
        service.release(held, &seller.did()).unwrap();

        let engine = LocalConsensus::new();
        let mut node = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(auth.did()),
        );
        for tx in service.ops() {
            node.submit(tx.clone()).unwrap();
            node.sync().unwrap();
        }
        assert_eq!(service.ledger().root(), node.certs().root());
        assert_eq!(node.certs().total_supply(), 1000);
        assert!(node.certs().utxo(&held).is_none());
    }

    #[test]
    fn unknown_or_non_holding_escrow_is_refused() {
        let auth = authority();
        let buyer = Keypair::generate();
        let seller = Keypair::generate();
        let (mut service, held) = settles(&auth, &buyer, &seller.did(), 1000);

        // A coin id that was never rowed into this marketplace is rejected.
        let ghost_did = service.escrow_did(&buyer.did(), &seller.did(), CertId([9u8; 32]));
        let ghost = commit_coin(1000, &ghost_did, &ESCROW_BLINDING);
        let unknown = service.release(ghost, &seller.did()).unwrap_err();
        assert_eq!(unknown.code, Code::ESCROW_UNKNOWN);

        // After release, the dealt escrow is final — refund is refused.
        service.release(held, &seller.did()).unwrap();
        let stale = service.refund(held, &buyer.did()).unwrap_err();
        assert_eq!(stale.code, Code::ESCROW_ALREADY_SETTLED);
    }
}
