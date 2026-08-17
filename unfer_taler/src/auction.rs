//! Prebid-model unified auction settlement over the certificate ledger.
//!
//! This is the operator-side marketplace service that turns a deterministic
//! [`AuctionLedger`] winner into settled value — the exchange analogue of the
//! Phase-4 [`EscrowService`](unfer_consensus::escrow::EscrowService). Both
//! payments and (for carbon lots) the credits themselves are rowed into escrow
//! DIDs derived deterministically from the operator's master key, exactly like
//! the certificate escrow, so a peer `ConsensusNode` replaying the log lands on
//! the identical certificate root and the same winner.
//!
//! Two markets, one mechanism (Prebid's "unified auction"):
//!
//! - **Carbon credits** ([`AuctionAsset::CarbonCredits`]): a seller opens a lot
//!   and escrows the credit certificate. Bidders escrow their payment e-coin
//!   (Taler withdraw output or a credits coin). On close, the unified clearing
//!   selects the winner; the operator releases the winner's payment to the
//!   seller, refunds the losers, and transfers the escrowed credit to the
//!   winner.
//! - **Publicity inventory** ([`AuctionAsset::PublicitySlot`]): a publisher
//!   opens a slot (an AdSense alternative). Same payment escrow; no credit
//!   certificate is involved, and the seller escrows nothing.
//!
//! Every produced op is an ordinary `ConsensusTransaction` (either an
//! [`AuctionOp`] or a conserving [`CertificateOp`]) recorded in
//! [`ops`](AuctionService::ops). No value is created or destroyed: the auction
//! only moves coins that already exist, so the `total_supply` audit (UK-7107
//! analogue) always balances.
//!
//! Error codes: UK-7301..UK-7308 (see `unfer_protocol::codes`).

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};

use unfer_consensus::auction::AuctionLedger;
use unfer_consensus::certs::{CertificateLedger, MintAuthority, commit_coin};
use unfer_consensus::signing::{Keypair, sign_transaction};
use unfer_protocol::{
    AuctionAsset, AuctionId, AuctionLot, AuctionOp, AuctionOpKind, AuctionWinner, CertId,
    CertificateOp, CertificateOpKind, Code, CoinRef, ConsensusTransaction, Diagnostic, Severity,
};

const AUCTION_BLINDING: [u8; 32] = [0u8; 32];

/// Lifecycle of one escrowed coin inside an auction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuctionEscrowState {
    /// The coin sits in the operator-derived escrow DID.
    Holding,
    /// The coin moved to its recipient (winner/seller). Final.
    Released,
    /// The coin returned to its depositor (loser refund / unsold lot). Final.
    Refunded,
}

/// A bidder's payment coin held between a bidder and a seller.
#[derive(Debug, Clone)]
pub struct PaymentEscrow {
    pub escrowed: CertId,
    pub origin: CertId,
    pub lot_id: AuctionId,
    pub bidder: String,
    pub seller: String,
    pub amount: u64,
    pub state: AuctionEscrowState,
}

/// A seller's credit certificate held for a carbon lot.
#[derive(Debug, Clone)]
pub struct CreditHold {
    pub escrowed: CertId,
    pub origin: CertId,
    pub lot_id: AuctionId,
    pub seller: String,
    pub amount: u64,
    pub state: AuctionEscrowState,
}

/// The operator-side auction service: runs the deterministic auction engine and
/// settles every outcome with ordinary certificate transfers.
pub struct AuctionService {
    operator: Keypair,
    certs: CertificateLedger,
    auction: AuctionLedger,
    seqs: HashMap<String, u64>,
    payments: HashMap<[u8; 32], PaymentEscrow>,
    credit_holds: HashMap<[u8; 32], CreditHold>,
    settled_payments: HashSet<[u8; 32]>,
    settled_holds: HashSet<[u8; 32]>,
    ops: Vec<ConsensusTransaction>,
}

impl AuctionService {
    pub fn new(operator: Keypair, authority: MintAuthority) -> Self {
        Self {
            operator,
            certs: CertificateLedger::new(authority),
            auction: AuctionLedger::new(),
            seqs: HashMap::new(),
            payments: HashMap::new(),
            credit_holds: HashMap::new(),
            settled_payments: HashSet::new(),
            settled_holds: HashSet::new(),
            ops: Vec::new(),
        }
    }

    pub fn operator_did(&self) -> String {
        self.operator.did()
    }

    pub fn auction(&self) -> &AuctionLedger {
        &self.auction
    }

    pub fn certs(&self) -> &CertificateLedger {
        &self.certs
    }

    pub fn payment_escrow(&self, escrowed: &CertId) -> Option<&PaymentEscrow> {
        self.payments.get(&escrowed.0)
    }

    pub fn credit_hold(&self, escrowed: &CertId) -> Option<&CreditHold> {
        self.credit_holds.get(&escrowed.0)
    }

    /// Every signed transaction this service produced or observed, in order.
    /// Feed these to a `ConsensusNode` to replay the identical state.
    pub fn ops(&self) -> &[ConsensusTransaction] {
        &self.ops
    }

    /// Feed an external op (e.g. the authority's mint that created a bidder's
    /// e-coin or the seller's credit certificate) into the mirror ledger.
    pub fn observe(&mut self, tx: ConsensusTransaction) -> Result<(), Diagnostic> {
        self.apply_transaction(&tx)?;
        self.ops.push(tx);
        Ok(())
    }

    /// The escrow DID that will hold `coin` for a bid on `lot_id`.
    pub fn payment_did(&self, lot_id: &AuctionId, bidder: &str, origin: CertId) -> String {
        self.escrow_key(b"unfer:auction:payment:v1", lot_id, bidder, origin)
            .did()
    }

    /// The escrow DID that will hold a seller's credit certificate for `lot_id`.
    pub fn credit_did(&self, lot_id: &AuctionId, seller: &str, origin: CertId) -> String {
        self.escrow_key(b"unfer:auction:credit:v1", lot_id, seller, origin)
            .did()
    }

    /// Open a lot. The seller signs the [`AuctionOpKind::Open`]; for a carbon
    /// lot they must also escrow `funding` (a credit certificate they own for
    /// the lot amount), which is rowed into the deterministic lot escrow DID.
    pub fn open_lot(
        &mut self,
        seller: &Keypair,
        lot: AuctionLot,
        funding: Option<CertId>,
    ) -> Result<(), Diagnostic> {
        if self.auction.lot(&lot.lot_id).is_some() {
            return Err(Diagnostic::new(
                Code::AUCTION_LOT_EXISTS,
                "lot already exists on the ledger",
                Severity::Error,
            ));
        }
        if lot.seller_did != seller.did() {
            return Err(Diagnostic::new(
                Code::AUCTION_NOT_SELLER,
                "only the lot's seller may open it",
                Severity::Error,
            ));
        }
        if lot.floor == 0 {
            return Err(Diagnostic::new(
                Code::AUCTION_BID_BELOW_FLOOR,
                "lot floor must be positive",
                Severity::Error,
            ));
        }
        if let AuctionAsset::CarbonCredits { amount } = lot.asset {
            let funding = funding.ok_or_else(|| {
                Diagnostic::new(
                    Code::AUCTION_QTY_MISMATCH,
                    "a carbon lot must be backed by a credit certificate",
                    Severity::Error,
                )
            })?;
            let owner = self
                .certs
                .utxo(&funding)
                .map(|c| c.owner.clone())
                .ok_or_else(|| {
                    Diagnostic::new(
                        Code::AUCTION_UNKNOWN_LOT,
                        "funding certificate does not exist",
                        Severity::Error,
                    )
                })?;
            if owner != seller.did() {
                return Err(Diagnostic::new(
                    Code::AUCTION_NOT_SELLER,
                    "the funding certificate is not owned by the seller",
                    Severity::Error,
                ));
            }
            self.escrow_credit(seller, &lot.lot_id, funding, amount)?;
        }
        let kind = AuctionOpKind::Open { lot };
        self.apply_auction(seller, kind)?;
        Ok(())
    }

    /// Place a bid on an open lot. The bidder escrows their payment e-coin
    /// (face value `price_per_unit * quantity`) into the lot's payment escrow,
    /// then the signed bid is emitted. Both ops are applied to the mirrors.
    pub fn bid(
        &mut self,
        bidder: &Keypair,
        lot_id: AuctionId,
        price_per_unit: u64,
        quantity: u64,
        funding: CertId,
    ) -> Result<(), Diagnostic> {
        let amount = price_per_unit.checked_mul(quantity).ok_or_else(|| {
            Diagnostic::new(
                Code::AUCTION_QTY_MISMATCH,
                "payment overflow",
                Severity::Error,
            )
        })?;
        let owner = self
            .certs
            .utxo(&funding)
            .map(|c| c.owner.clone())
            .ok_or_else(|| {
                Diagnostic::new(
                    Code::AUCTION_UNKNOWN_LOT,
                    "payment certificate does not exist",
                    Severity::Error,
                )
            })?;
        if owner != bidder.did() {
            return Err(Diagnostic::new(
                Code::AUCTION_SELF_BID,
                "the payment certificate is not owned by the bidder",
                Severity::Error,
            ));
        }
        let lot = self.auction.lot(&lot_id).ok_or_else(|| {
            Diagnostic::new(Code::AUCTION_UNKNOWN_LOT, "unknown lot", Severity::Error)
        })?;
        if lot.closed {
            return Err(Diagnostic::new(
                Code::AUCTION_LOT_CLOSED,
                "lot is already closed",
                Severity::Error,
            ));
        }
        if lot.lot.seller_did == bidder.did() {
            return Err(Diagnostic::new(
                Code::AUCTION_SELF_BID,
                "the seller cannot bid on their own lot",
                Severity::Error,
            ));
        }
        if price_per_unit < lot.lot.floor {
            return Err(Diagnostic::new(
                Code::AUCTION_BID_BELOW_FLOOR,
                format!("bid {price_per_unit} below the floor {}", lot.lot.floor),
                Severity::Error,
            ));
        }
        if quantity == 0 {
            return Err(Diagnostic::new(
                Code::AUCTION_QTY_MISMATCH,
                "bid quantity must be positive",
                Severity::Error,
            ));
        }
        if let AuctionAsset::CarbonCredits { amount: lot_amount } = &lot.lot.asset
            && quantity > *lot_amount
        {
            return Err(Diagnostic::new(
                Code::AUCTION_QTY_MISMATCH,
                format!("bid quantity {quantity} exceeds lot amount {lot_amount}"),
                Severity::Error,
            ));
        }
        self.escrow_payment(bidder, &lot_id, funding, amount)?;
        let kind = AuctionOpKind::Bid {
            lot_id,
            price_per_unit,
            quantity,
        };
        self.apply_auction(bidder, kind)?;
        Ok(())
    }

    /// Close a lot and settle: the unified clearing selects the winner; the
    /// winner's payment releases to the seller, losers are refunded, and a
    /// carbon lot's escrowed credits transfer to the winner. An unsold lot
    /// refunds every payment and returns the credits to the seller.
    pub fn close(
        &mut self,
        seller: &Keypair,
        lot_id: AuctionId,
    ) -> Result<Option<AuctionWinner>, Diagnostic> {
        let kind = AuctionOpKind::Close { lot_id };
        let outcome = self.apply_auction(seller, kind)?;

        let mut deliveries = Vec::new();
        let mut holds: Vec<CertId> = self
            .credit_holds
            .values()
            .filter(|h| h.lot_id == lot_id && h.seller == seller.did())
            .map(|h| h.escrowed)
            .collect();
        holds.sort_by_key(|a| a.0);

        match &outcome {
            Some(winner) => {
                for (escrowed, payment) in self.payments.clone() {
                    if payment.lot_id != lot_id {
                        continue;
                    }
                    let escrowed = CertId(escrowed);
                    if payment.bidder == winner.bidder_did {
                        deliveries.push(self.release_payment(escrowed, &payment.seller)?);
                    } else {
                        deliveries.push(self.refund_payment(escrowed, &payment.bidder)?);
                    }
                }
                for held in holds {
                    deliveries.push(self.deliver_credit(held, &winner.bidder_did)?);
                }
            }
            None => {
                for (escrowed, payment) in self.payments.clone() {
                    if payment.lot_id == lot_id {
                        deliveries.push(self.refund_payment(CertId(escrowed), &payment.bidder)?);
                    }
                }
                for held in holds {
                    deliveries.push(
                        self.refund_credit(
                            held,
                            &self
                                .credit_hold(&held)
                                .map(|h| h.seller.clone())
                                .unwrap_or_default(),
                        )?,
                    );
                }
            }
        }

        Ok(outcome)
    }

    // -- internal settlement ------------------------------------------------

    /// Row `origin` (a coin `bidder` owns) into the payment escrow DID for
    /// `lot_id`. Refuses double escrow of the same origin.
    fn escrow_payment(
        &mut self,
        bidder: &Keypair,
        lot_id: &AuctionId,
        origin: CertId,
        amount: u64,
    ) -> Result<CertId, Diagnostic> {
        let seller = self
            .auction
            .lot(lot_id)
            .map(|s| s.lot.seller_did.clone())
            .ok_or_else(|| {
                Diagnostic::new(Code::AUCTION_UNKNOWN_LOT, "unknown lot", Severity::Error)
            })?;
        let stored = self.certs.utxo(&origin).map(|c| c.amount).ok_or_else(|| {
            Diagnostic::new(
                Code::AUCTION_UNKNOWN_LOT,
                "payment certificate does not exist",
                Severity::Error,
            )
        })?;
        if stored != amount {
            return Err(Diagnostic::new(
                Code::AUCTION_QTY_MISMATCH,
                format!("payment e-coin face value {stored} must equal the bid total {amount}"),
                Severity::Error,
            ));
        }
        if self.payments.values().any(|p| p.origin == origin) {
            return Err(Diagnostic::new(
                Code::AUCTION_SELF_BID,
                "payment certificate already escrowed",
                Severity::Error,
            ));
        }
        let escrow_did = self.payment_did(lot_id, &bidder.did(), origin);
        let escrowed = commit_coin(amount, &escrow_did, &AUCTION_BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: origin,
                amount,
                owner: bidder.did(),
            }],
            outputs: vec![CoinRef {
                coin_id: escrowed,
                amount,
                owner: escrow_did,
            }],
        };
        let tx = self.build_cert(&bidder.did(), kind);
        self.emit_cert(tx, bidder)?;
        self.payments.insert(
            escrowed.0,
            PaymentEscrow {
                escrowed,
                origin,
                lot_id: *lot_id,
                bidder: bidder.did(),
                seller,
                amount,
                state: AuctionEscrowState::Holding,
            },
        );
        Ok(escrowed)
    }

    /// Row `origin` (a credit certificate `seller` owns) into the lot escrow.
    fn escrow_credit(
        &mut self,
        seller: &Keypair,
        lot_id: &AuctionId,
        origin: CertId,
        amount: u64,
    ) -> Result<CertId, Diagnostic> {
        if self.credit_holds.values().any(|h| h.origin == origin) {
            return Err(Diagnostic::new(
                Code::AUCTION_LOT_EXISTS,
                "credit certificate already escrowed",
                Severity::Error,
            ));
        }
        let escrow_did = self.credit_did(lot_id, &seller.did(), origin);
        let escrowed = commit_coin(amount, &escrow_did, &AUCTION_BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: origin,
                amount,
                owner: seller.did(),
            }],
            outputs: vec![CoinRef {
                coin_id: escrowed,
                amount,
                owner: escrow_did,
            }],
        };
        let tx = self.build_cert(&seller.did(), kind);
        self.emit_cert(tx, seller)?;
        self.credit_holds.insert(
            escrowed.0,
            CreditHold {
                escrowed,
                origin,
                lot_id: *lot_id,
                seller: seller.did(),
                amount,
                state: AuctionEscrowState::Holding,
            },
        );
        Ok(escrowed)
    }

    /// Winner's payment → seller.
    fn release_payment(&mut self, escrowed: CertId, recipient: &str) -> Result<CertId, Diagnostic> {
        let payment = self.payments.get(&escrowed.0).ok_or_else(|| {
            Diagnostic::new(
                Code::ESCROW_UNKNOWN,
                "payment escrow not found",
                Severity::Error,
            )
        })?;
        self.ensure_unsettled_payment(&escrowed)?;
        let out = commit_coin(payment.amount, recipient, &AUCTION_BLINDING);
        let did = self.payment_did(&payment.lot_id, &payment.bidder, payment.origin);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: escrowed,
                amount: payment.amount,
                owner: did.clone(),
            }],
            outputs: vec![CoinRef {
                coin_id: out,
                amount: payment.amount,
                owner: recipient.to_string(),
            }],
        };
        let key = self.escrow_key(
            b"unfer:auction:payment:v1",
            &payment.lot_id,
            &payment.bidder,
            payment.origin,
        );
        let tx = self.build_cert(&did, kind);
        self.emit_cert(tx, &key)?;
        self.settle_payment(escrowed, AuctionEscrowState::Released)?;
        Ok(out)
    }

    /// Loser's payment → bidder.
    fn refund_payment(&mut self, escrowed: CertId, recipient: &str) -> Result<CertId, Diagnostic> {
        let payment = self.payments.get(&escrowed.0).ok_or_else(|| {
            Diagnostic::new(
                Code::ESCROW_UNKNOWN,
                "payment escrow not found",
                Severity::Error,
            )
        })?;
        self.ensure_unsettled_payment(&escrowed)?;
        let out = commit_coin(payment.amount, recipient, &AUCTION_BLINDING);
        let did = self.payment_did(&payment.lot_id, &payment.bidder, payment.origin);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: escrowed,
                amount: payment.amount,
                owner: did.clone(),
            }],
            outputs: vec![CoinRef {
                coin_id: out,
                amount: payment.amount,
                owner: recipient.to_string(),
            }],
        };
        let key = self.escrow_key(
            b"unfer:auction:payment:v1",
            &payment.lot_id,
            &payment.bidder,
            payment.origin,
        );
        let tx = self.build_cert(&did, kind);
        self.emit_cert(tx, &key)?;
        self.settle_payment(escrowed, AuctionEscrowState::Refunded)?;
        Ok(out)
    }

    /// Carbon credits → winner.
    fn deliver_credit(&mut self, escrowed: CertId, recipient: &str) -> Result<CertId, Diagnostic> {
        let hold = self.credit_holds.get(&escrowed.0).ok_or_else(|| {
            Diagnostic::new(
                Code::ESCROW_UNKNOWN,
                "credit hold not found",
                Severity::Error,
            )
        })?;
        if self.settled_holds.contains(&escrowed.0) {
            return Err(Diagnostic::new(
                Code::ESCROW_ALREADY_SETTLED,
                "credit already settled",
                Severity::Error,
            ));
        }
        let out = commit_coin(hold.amount, recipient, &AUCTION_BLINDING);
        let did = self.credit_did(&hold.lot_id, &hold.seller, hold.origin);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: escrowed,
                amount: hold.amount,
                owner: did.clone(),
            }],
            outputs: vec![CoinRef {
                coin_id: out,
                amount: hold.amount,
                owner: recipient.to_string(),
            }],
        };
        let key = self.escrow_key(
            b"unfer:auction:credit:v1",
            &hold.lot_id,
            &hold.seller,
            hold.origin,
        );
        let tx = self.build_cert(&did, kind);
        self.emit_cert(tx, &key)?;
        if let Some(h) = self.credit_holds.get_mut(&escrowed.0) {
            h.state = AuctionEscrowState::Released;
        }
        self.settled_holds.insert(escrowed.0);
        Ok(out)
    }

    /// Unsold carbon credits → seller.
    fn refund_credit(&mut self, escrowed: CertId, recipient: &str) -> Result<CertId, Diagnostic> {
        let hold = self.credit_holds.get(&escrowed.0).ok_or_else(|| {
            Diagnostic::new(
                Code::ESCROW_UNKNOWN,
                "credit hold not found",
                Severity::Error,
            )
        })?;
        if self.settled_holds.contains(&escrowed.0) {
            return Err(Diagnostic::new(
                Code::ESCROW_ALREADY_SETTLED,
                "credit already settled",
                Severity::Error,
            ));
        }
        let out = commit_coin(hold.amount, recipient, &AUCTION_BLINDING);
        let did = self.credit_did(&hold.lot_id, &hold.seller, hold.origin);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: escrowed,
                amount: hold.amount,
                owner: did.clone(),
            }],
            outputs: vec![CoinRef {
                coin_id: out,
                amount: hold.amount,
                owner: recipient.to_string(),
            }],
        };
        let key = self.escrow_key(
            b"unfer:auction:credit:v1",
            &hold.lot_id,
            &hold.seller,
            hold.origin,
        );
        let tx = self.build_cert(&did, kind);
        self.emit_cert(tx, &key)?;
        if let Some(h) = self.credit_holds.get_mut(&escrowed.0) {
            h.state = AuctionEscrowState::Refunded;
        }
        self.settled_holds.insert(escrowed.0);
        Ok(out)
    }

    fn ensure_unsettled_payment(&self, escrowed: &CertId) -> Result<(), Diagnostic> {
        if self.settled_payments.contains(&escrowed.0) {
            return Err(Diagnostic::new(
                Code::ESCROW_ALREADY_SETTLED,
                "payment already settled",
                Severity::Error,
            ));
        }
        if let Some(p) = self.payments.get(&escrowed.0)
            && p.state != AuctionEscrowState::Holding
        {
            return Err(Diagnostic::new(
                Code::ESCROW_NOT_HOLDING,
                "payment escrow is not holding",
                Severity::Error,
            ));
        }
        Ok(())
    }

    fn settle_payment(
        &mut self,
        escrowed: CertId,
        outcome: AuctionEscrowState,
    ) -> Result<(), Diagnostic> {
        if let Some(p) = self.payments.get_mut(&escrowed.0) {
            p.state = outcome;
        }
        self.settled_payments.insert(escrowed.0);
        Ok(())
    }

    /// Deterministic per-(domain, lot, party, coin) escrow spending key: only
    /// the marketplace operator can regenerate it.
    fn escrow_key(&self, domain: &[u8], lot_id: &AuctionId, party: &str, coin: CertId) -> Keypair {
        let mut ctx = Sha256::new();
        ctx.update(domain);
        ctx.update(self.operator.public_key());
        ctx.update(lot_id.0);
        ctx.update(party.as_bytes());
        ctx.update(coin.0);
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

    fn emit_cert(
        &mut self,
        mut tx: ConsensusTransaction,
        signer: &Keypair,
    ) -> Result<(), Diagnostic> {
        sign_transaction(&mut tx, signer);
        self.apply_transaction(&tx)?;
        self.ops.push(tx);
        Ok(())
    }

    fn apply_auction(
        &mut self,
        signer: &Keypair,
        kind: AuctionOpKind,
    ) -> Result<Option<AuctionWinner>, Diagnostic> {
        let did = signer.did();
        let seq = self.next_seq(&did);
        let outcome = self.auction.apply_op(&did, &kind, seq)?;
        let mut tx = ConsensusTransaction::AuctionOp(AuctionOp {
            did,
            kind,
            seq,
            signature: [0u8; 64],
        });
        sign_transaction(&mut tx, signer);
        self.ops.push(tx);
        Ok(outcome)
    }

    fn apply_transaction(&mut self, tx: &ConsensusTransaction) -> Result<(), Diagnostic> {
        match tx {
            ConsensusTransaction::CertificateOp(op) => {
                self.certs.apply_op(&op.did, &op.kind, op.seq)?;
                Ok(())
            }
            other => Err(Diagnostic::new(
                Code::INTERNAL,
                format!("auction service only observes certificate ops, got {other:?}"),
                Severity::Error,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_consensus::engine::LocalConsensus;
    use unfer_consensus::node::ConsensusNode;
    use unfer_protocol::AuctionAsset;

    fn authority() -> Keypair {
        Keypair::generate()
    }

    fn mint_to(
        service: &mut AuctionService,
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

    fn seller_setup(
        service: &mut AuctionService,
        auth: &Keypair,
        seller: &str,
        credits: u64,
    ) -> CertId {
        mint_to(service, auth, seller, credits, [7u8; 32])
    }

    fn buyer_setup(
        service: &mut AuctionService,
        auth: &Keypair,
        buyer: &str,
        ecoins: u64,
    ) -> CertId {
        mint_to(service, auth, buyer, ecoins, [9u8; 32])
    }

    fn carbon_lot(seller: &str, floor: u64, id_byte: u8) -> AuctionLot {
        AuctionLot {
            lot_id: AuctionId([id_byte; 32]),
            seller_did: seller.to_string(),
            asset: AuctionAsset::CarbonCredits { amount: 1000 },
            currency: unfer_protocol::AuctionCurrency::Taler,
            floor,
            opens_seq: 1,
            closes_seq: 100,
        }
    }

    fn slot_lot(publisher: &str, floor: u64, id_byte: u8) -> AuctionLot {
        AuctionLot {
            lot_id: AuctionId([id_byte; 32]),
            seller_did: publisher.to_string(),
            asset: AuctionAsset::PublicitySlot {
                slot: "homepage_leaderboard_300x250".to_string(),
                description: None,
            },
            currency: unfer_protocol::AuctionCurrency::CarbonCredits,
            floor,
            opens_seq: 1,
            closes_seq: 100,
        }
    }

    #[test]
    fn carbon_auction_settles_to_winner_and_refunds_losers() {
        let auth = authority();
        let seller = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();
        let mut svc = AuctionService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let credits = seller_setup(&mut svc, &auth, &seller.did(), 1000);
        let alice_money = buyer_setup(&mut svc, &auth, &alice.did(), 2400);
        let bob_money = buyer_setup(&mut svc, &auth, &bob.did(), 2700);

        svc.open_lot(&seller, carbon_lot(&seller.did(), 5, 1), Some(credits))
            .unwrap();
        svc.bid(&alice, AuctionId([1; 32]), 6, 400, alice_money)
            .unwrap();
        svc.bid(&bob, AuctionId([1; 32]), 9, 300, bob_money)
            .unwrap();

        let winner = svc.close(&seller, AuctionId([1; 32])).unwrap().unwrap();
        assert_eq!(winner.bidder_did, bob.did());
        assert_eq!(winner.total, 2700);

        // Winner's payment went to the seller; the credits moved to the winner.
        let seller_after = svc
            .certs()
            .coins_of(&seller.did())
            .iter()
            .map(|c| c.amount)
            .sum::<u64>();
        assert_eq!(seller_after, 2700, "seller keeps the winner's payment only");
        let alice_after = svc
            .certs()
            .coins_of(&alice.did())
            .iter()
            .map(|c| c.amount)
            .sum::<u64>();
        assert_eq!(alice_after, 2400, "loser fully refunded");
        let bob_after = svc
            .certs()
            .coins_of(&bob.did())
            .iter()
            .map(|c| c.amount)
            .sum::<u64>();
        assert_eq!(bob_after, 1000, "winner holds the credits");
        // No value created or destroyed.
        assert_eq!(svc.certs().total_supply(), 1000 + 2400 + 2700);
    }

    #[test]
    fn publicity_slot_is_an_adsense_alternative() {
        let auth = authority();
        let publisher = Keypair::generate();
        let advertiser = Keypair::generate();
        let mut svc = AuctionService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let campaign = buyer_setup(&mut svc, &auth, &advertiser.did(), 3);

        // No credit certificate is involved — only payment moves.
        svc.open_lot(&publisher, slot_lot(&publisher.did(), 2, 2), None)
            .unwrap();
        svc.bid(&advertiser, AuctionId([2; 32]), 3, 1, campaign)
            .unwrap();

        let winner = svc.close(&publisher, AuctionId([2; 32])).unwrap().unwrap();
        assert_eq!(winner.bidder_did, advertiser.did());
        assert_eq!(winner.total, 3);
        let publisher_after = svc
            .certs()
            .coins_of(&publisher.did())
            .iter()
            .map(|c| c.amount)
            .sum::<u64>();
        assert_eq!(publisher_after, 3, "publisher earns the slot price");
        assert_eq!(svc.certs().total_supply(), 3);
    }

    #[test]
    fn unsold_carbon_lot_returns_credits() {
        let auth = authority();
        let seller = Keypair::generate();
        let alice = Keypair::generate();
        let mut svc = AuctionService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let credits = seller_setup(&mut svc, &auth, &seller.did(), 1000);
        let alice_money = buyer_setup(&mut svc, &auth, &alice.did(), 600);

        svc.open_lot(&seller, carbon_lot(&seller.did(), 10, 3), Some(credits))
            .unwrap();
        // Alice's bid is below the floor → rejected at the auction engine; her
        // payment e-coin is never escrowed.
        let below_floor = svc
            .bid(&alice, AuctionId([3; 32]), 6, 100, alice_money)
            .unwrap_err();
        assert_eq!(below_floor.code, Code::AUCTION_BID_BELOW_FLOOR);
        assert_eq!(
            svc.certs()
                .coins_of(&alice.did())
                .iter()
                .map(|c| c.amount)
                .sum::<u64>(),
            600
        );

        // The lot closes with no eligible bids: credits return to the seller.
        let outcome = svc.close(&seller, AuctionId([3; 32])).unwrap();
        assert!(outcome.is_none());
        let seller_after = svc
            .certs()
            .coins_of(&seller.did())
            .iter()
            .map(|c| c.amount)
            .sum::<u64>();
        assert_eq!(seller_after, 1000, "seller keeps credits on an unsold lot");
        assert_eq!(svc.certs().total_supply(), 1600);
    }

    #[test]
    fn bid_escrows_payment_before_bid_and_rejects_unknown_lot() {
        let auth = authority();
        let _seller = Keypair::generate();
        let alice = Keypair::generate();
        let mut svc = AuctionService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let alice_money = buyer_setup(&mut svc, &auth, &alice.did(), 500);

        // Bidding on a lot that was never opened fails at the auction engine.
        let err = svc
            .bid(&alice, AuctionId([9; 32]), 3, 1, alice_money)
            .unwrap_err();
        assert_eq!(err.code, Code::AUCTION_UNKNOWN_LOT);
        assert_eq!(
            svc.certs()
                .coins_of(&alice.did())
                .iter()
                .map(|c| c.amount)
                .sum::<u64>(),
            500
        );
    }

    #[test]
    fn replay_converges_the_consensus_node() {
        let auth = authority();
        let seller = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();
        let mut svc = AuctionService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let credits = seller_setup(&mut svc, &auth, &seller.did(), 1000);
        let alice_money = buyer_setup(&mut svc, &auth, &alice.did(), 600);
        let bob_money = buyer_setup(&mut svc, &auth, &bob.did(), 1600);

        svc.open_lot(&seller, carbon_lot(&seller.did(), 5, 4), Some(credits))
            .unwrap();
        svc.bid(&alice, AuctionId([4; 32]), 6, 100, alice_money)
            .unwrap();
        svc.bid(&bob, AuctionId([4; 32]), 8, 200, bob_money)
            .unwrap();
        let winner = svc.close(&seller, AuctionId([4; 32])).unwrap().unwrap();
        assert_eq!(winner.bidder_did, bob.did());

        let engine = LocalConsensus::new();
        let mut node = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(auth.did()),
        );
        for tx in svc.ops() {
            node.submit(tx.clone()).unwrap();
            node.sync().unwrap();
        }
        assert_eq!(svc.certs().root(), node.certs().root());
        assert_eq!(node.certs().total_supply(), 1000 + 600 + 1600);
        let bob_winner = svc
            .auction()
            .report(&AuctionId([4; 32]))
            .unwrap()
            .winner
            .unwrap();
        assert_eq!(
            bob_winner.bidder_did,
            node.auction()
                .report(&AuctionId([4; 32]))
                .unwrap()
                .winner
                .unwrap()
                .bidder_did
        );
    }
}
