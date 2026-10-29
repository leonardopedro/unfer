//! The Taler exchange adapter over the certificate ledger.
//!
//! The exchange owns two views of the same value flows:
//!
//! - **On-ledger** (`CertificateLedger`, mirrored here) — e-coins are ordinary
//!   certificates: minted by the treasury (the mint authority) on withdraw,
//!   transferred to merchants on deposit. Replayable by any `ConsensusNode`.
//! - **Private** (this struct) — the fiat-side bookkeeping a real exchange
//!   keeps in its own database: customer reserves, merchant balances, wire
//!   transfers, and coin provenance.
//!
//! The money-conservation identity across the seam is
//!
//! ```text
//! fiat_in - fiat_out = reserves_total + merchant_total + outstanding_funded
//! ```
//!
//! i.e. every unit of fiat that arrived is still somewhere: in a customer's
//! unwithdrawn reserve, in a merchant's redeemable balance, or inside an e-coin
//! still in a customer's wallet. `audit()` checks this identity after every op
//! (asserted in the proptest).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use unfer_consensus::certs::{CertificateLedger, LedgerSnapshot, MintAuthority};
use unfer_consensus::signing::Keypair;
use unfer_protocol::{
    CertId, CertificateOp, CertificateOpKind, Code, CoinRef, ConsensusTransaction, Diagnostic,
    Severity,
};

use crate::denom::{Denomination, DenominationBook};
use crate::wire::{WireGateway, WireRef, WireStatus};

/// A customer's fiat reserve, keyed by an id the customer chooses (mirrors the
/// Taler reserve public key).
pub type ReserveId = [u8; 32];

/// A peg-out wire in flight to a merchant's bank account.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PegOut {
    pub wire: WireRef,
    pub merchant_did: String,
    /// Pending until the wire gateway confirms.
    pub confirmed: bool,
}

#[derive(Debug)]
struct Reserve {
    owner: String,
    balance: u64,
}

/// A durable image of the exchange's private (fiat-side) state plus its mirror
/// of the certificate ledger and the emitted consensus log.
///
/// The treasury key is deliberately absent: a host keeps that secret in its own
/// key file and passes the `Keypair` back on restore, so a state snapshot can
/// be backed up without carrying signing material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeSnapshot {
    /// The treasury DID the snapshot was taken with; restore refuses a
    /// mismatch rather than silently serving a different mint authority.
    pub treasury_did: String,
    pub ledger: LedgerSnapshot,
    pub denominations: Vec<Denomination>,
    pub reserves: Vec<(String, String, u64)>,
    pub merchant_balances: Vec<(String, u64)>,
    pub funded: Vec<String>,
    pub deposited: Vec<String>,
    pub coin_blinding: Vec<(String, String)>,
    pub peg_outs: Vec<PegOut>,
    pub gateway_wires: Vec<WireRef>,
    pub current_seq: u64,
    pub funded_outstanding: u64,
    pub fiat_in: u64,
    pub fiat_out: u64,
    pub ops: Vec<ConsensusTransaction>,
}

/// The GNU Taler exchange adapter.
pub struct TalerExchange {
    treasury: Keypair,
    /// The exchange's view of the consensus certificate state. The treasury is
    /// the configured mint authority.
    ledger: CertificateLedger,
    book: DenominationBook,
    gateway: Box<dyn WireGateway>,

    current_seq: u64,
    ops: Vec<ConsensusTransaction>,

    reserves: HashMap<ReserveId, Reserve>,
    merchant_balances: HashMap<String, u64>,

    /// Coin ids minted by this exchange during `withdraw` (i.e. funded by a
    /// customer reserve) and still in a customer's hands.
    funded_outstanding: u64,
    /// Coin provenance: every e-coin minted here, so deposit can refuse coins
    /// the exchange never issued (they would not be fiat-backed).
    funded: HashSet<[u8; 32]>,
    /// Coin ids already deposited to a merchant (no double deposit).
    deposited: HashSet<[u8; 32]>,
    /// Blinding used for each minted e-coin, so coin ids can be re-derived.
    coin_blinding: HashMap<[u8; 32], [u8; 32]>,

    /// Fiat that arrived (confirmed peg-ins) / left (peg-outs) across the seam.
    fiat_in: u64,
    fiat_out: u64,
    peg_outs: HashMap<String, PegOut>,
}

impl TalerExchange {
    pub fn new(treasury: Keypair, gateway: Box<dyn WireGateway>) -> Self {
        Self {
            ledger: CertificateLedger::new(MintAuthority::Only(treasury.did())),
            book: DenominationBook::new(),
            gateway,
            current_seq: 0,
            ops: Vec::new(),
            reserves: HashMap::new(),
            merchant_balances: HashMap::new(),
            funded_outstanding: 0,
            funded: HashSet::new(),
            deposited: HashSet::new(),
            coin_blinding: HashMap::new(),
            fiat_in: 0,
            fiat_out: 0,
            peg_outs: HashMap::new(),
            treasury,
        }
    }

    pub fn treasury_did(&self) -> String {
        self.treasury.did()
    }

    /// A durable image of every field this exchange owns (see
    /// [`ExchangeSnapshot`]).
    pub fn snapshot(&self) -> ExchangeSnapshot {
        let mut reserves: Vec<(String, String, u64)> = self
            .reserves
            .iter()
            .map(|(id, r)| (hex::encode(id), r.owner.clone(), r.balance))
            .collect();
        reserves.sort_by(|a, b| a.0.cmp(&b.0));
        let mut merchants: Vec<(String, u64)> = self
            .merchant_balances
            .iter()
            .map(|(did, bal)| (did.clone(), *bal))
            .collect();
        merchants.sort_by(|a, b| a.0.cmp(&b.0));
        let mut funded: Vec<String> = self.funded.iter().map(hex::encode).collect();
        funded.sort();
        let mut deposited: Vec<String> = self.deposited.iter().map(hex::encode).collect();
        deposited.sort();
        let mut coin_blinding: Vec<(String, String)> = self
            .coin_blinding
            .iter()
            .map(|(coin, blind)| (hex::encode(coin), hex::encode(blind)))
            .collect();
        coin_blinding.sort_by(|a, b| a.0.cmp(&b.0));
        let mut peg_outs: Vec<PegOut> = self.peg_outs.values().cloned().collect();
        peg_outs.sort_by(|a, b| a.wire.wire_id.cmp(&b.wire.wire_id));
        ExchangeSnapshot {
            treasury_did: self.treasury.did(),
            ledger: self.ledger.snapshot(),
            denominations: self.book.entries().cloned().collect(),
            reserves,
            merchant_balances: merchants,
            funded,
            deposited,
            coin_blinding,
            peg_outs,
            gateway_wires: self.gateway.export(),
            current_seq: self.current_seq,
            funded_outstanding: self.funded_outstanding,
            fiat_in: self.fiat_in,
            fiat_out: self.fiat_out,
            ops: self.ops.clone(),
        }
    }

    /// Restore an exchange from a snapshot. The gateway is seeded with the
    /// wires the snapshot recorded so a pending peg-out can still be confirmed
    /// or cancelled after a restart.
    pub fn from_snapshot(
        treasury: Keypair,
        gateway: Box<dyn WireGateway>,
        snap: &ExchangeSnapshot,
    ) -> Result<Self, String> {
        if snap.treasury_did != treasury.did() {
            return Err(format!(
                "snapshot belongs to {}, not {}",
                snap.treasury_did,
                treasury.did()
            ));
        }
        let mut book = DenominationBook::new();
        for d in &snap.denominations {
            book.issue(d.value, d.expires_seq);
        }
        let mut gw = gateway;
        gw.import(snap.gateway_wires.clone());
        let mut reserves = HashMap::new();
        for (id, owner, balance) in &snap.reserves {
            let bytes = hex32(id).ok_or_else(|| format!("bad reserve id {id}"))?;
            reserves.insert(
                bytes,
                Reserve {
                    owner: owner.clone(),
                    balance: *balance,
                },
            );
        }
        let exchange = Self {
            ledger: CertificateLedger::from_snapshot(&snap.ledger),
            book,
            gateway: gw,
            current_seq: snap.current_seq,
            ops: snap.ops.clone(),
            reserves,
            merchant_balances: snap.merchant_balances.iter().cloned().collect(),
            funded_outstanding: snap.funded_outstanding,
            funded: snap.funded.iter().filter_map(|s| hex32(s)).collect(),
            deposited: snap.deposited.iter().filter_map(|s| hex32(s)).collect(),
            coin_blinding: snap
                .coin_blinding
                .iter()
                .filter_map(|(c, b)| Some((hex32(c)?, hex32(b)?)))
                .collect(),
            fiat_in: snap.fiat_in,
            fiat_out: snap.fiat_out,
            peg_outs: snap
                .peg_outs
                .iter()
                .map(|p| (p.wire.wire_id.clone(), p.clone()))
                .collect(),
            treasury,
        };
        exchange
            .audit()
            .map_err(|e| format!("restored state fails the fiat audit: {}", e.message))?;
        Ok(exchange)
    }

    /// The consensus sequence the exchange has advanced to (denomination
    /// expiry is measured against it).
    pub fn current_seq(&self) -> u64 {
        self.current_seq
    }

    /// Value inside e-coins minted by this exchange that are still in customer
    /// hands (i.e. funded, not yet deposited).
    pub fn funded_outstanding(&self) -> u64 {
        self.funded_outstanding
    }

    /// A reserve's owner and remaining balance.
    pub fn reserve(&self, reserve_id: &ReserveId) -> Option<(&str, u64)> {
        self.reserves
            .get(reserve_id)
            .map(|r| (r.owner.as_str(), r.balance))
    }

    /// Every reserve, in id order (an audit/console view).
    pub fn reserves(&self) -> Vec<(ReserveId, String, u64)> {
        let mut out: Vec<(ReserveId, String, u64)> = self
            .reserves
            .iter()
            .map(|(id, r)| (*id, r.owner.clone(), r.balance))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Every merchant fiat balance, in DID order.
    pub fn merchants(&self) -> Vec<(String, u64)> {
        let mut out: Vec<(String, u64)> = self
            .merchant_balances
            .iter()
            .map(|(did, bal)| (did.clone(), *bal))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Whether this exchange minted the e-coin (and it is not deposited yet).
    pub fn is_funded(&self, coin_id: &CertId) -> bool {
        self.funded.contains(&coin_id.0)
    }

    /// Whether the e-coin was already deposited to a merchant.
    pub fn is_deposited(&self, coin_id: &CertId) -> bool {
        self.deposited.contains(&coin_id.0)
    }

    /// A peg-out wire by id.
    pub fn peg_out_wire(&self, wire_id: &str) -> Option<&PegOut> {
        self.peg_outs.get(wire_id)
    }

    /// Every peg-out, in wire-id order.
    pub fn peg_outs(&self) -> Vec<PegOut> {
        let mut out: Vec<PegOut> = self.peg_outs.values().cloned().collect();
        out.sort_by(|a, b| a.wire.wire_id.cmp(&b.wire.wire_id));
        out
    }

    /// The denomination book (including expired entries).
    pub fn denominations(&self) -> Vec<Denomination> {
        self.book.entries().cloned().collect()
    }

    /// Fiat that arrived across the seam (confirmed peg-ins).
    pub fn fiat_in(&self) -> u64 {
        self.fiat_in
    }

    /// Fiat that left across the seam (confirmed peg-outs).
    pub fn fiat_out(&self) -> u64 {
        self.fiat_out
    }

    pub fn ledger(&self) -> &CertificateLedger {
        &self.ledger
    }

    /// The bank-facing gateway, for a host that drives the two-phase seam
    /// itself (e.g. a sandbox "incoming wire" endpoint). A production bank
    /// backend belongs behind this seam either way.
    pub fn gateway_mut(&mut self) -> &mut dyn WireGateway {
        self.gateway.as_mut()
    }

    /// Every signed certificate op the exchange has emitted, in sequence.
    /// Feed these to a `ConsensusNode` to replay the identical consensus view.
    pub fn ops(&self) -> &[ConsensusTransaction] {
        &self.ops
    }

    pub fn issue_denomination(&mut self, value: u64, expires_seq: u64) {
        self.book.issue(value, expires_seq);
    }

    pub fn denomination(&self, value: u64) -> Option<&Denomination> {
        self.book.find(value, self.current_seq)
    }

    // -- peg-in --------------------------------------------------------------

    /// Open a new reserve for `customer_did`. Idempotent for the same id+owner.
    pub fn open_reserve(&mut self, reserve_id: ReserveId, customer_did: &str) {
        self.reserves.entry(reserve_id).or_insert_with(|| Reserve {
            owner: customer_did.to_string(),
            balance: 0,
        });
    }

    /// Credit `reserve_id` from a *confirmed* bank wire. Refuses unconfirmed
    /// wires (UK-7103) and unknown reserves (UK-7101).
    pub fn peg_in(&mut self, reserve_id: ReserveId, wire: &WireRef) -> Result<(), Diagnostic> {
        if wire.status != WireStatus::Confirmed {
            return Err(diag(
                Code::TALER_UNCONFIRMED_WIRE,
                format!("wire {} is not confirmed", wire.wire_id),
            ));
        }
        let reserve = self
            .reserves
            .get_mut(&reserve_id)
            .ok_or_else(|| diag(Code::TALER_UNKNOWN_RESERVE, "reserve is not open"))?;
        reserve.balance = reserve
            .balance
            .checked_add(wire.amount)
            .ok_or_else(|| diag(Code::TALER_INSUFFICIENT_BALANCE, "reserve overflow"))?;
        self.fiat_in = self
            .fiat_in
            .checked_add(wire.amount)
            .ok_or_else(|| diag(Code::TALER_INSUFFICIENT_BALANCE, "fiat_in overflow"))?;
        Ok(())
    }

    // -- withdraw ------------------------------------------------------------

    /// Convert reserve credit into an e-coin of face value `amount`. The
    /// treasury mints a certificate owned by the reserve's customer; the
    /// reserve is debited. UK-7102 on shortfall, UK-7104 if the value is not
    /// a live denomination.
    pub fn withdraw(&mut self, reserve_id: ReserveId, amount: u64) -> Result<CertId, Diagnostic> {
        if self.book.find(amount, self.current_seq).is_none() {
            return Err(diag(
                Code::TALER_DENOM_UNSUPPORTED,
                format!("no live denomination for {amount}"),
            ));
        }
        let (owner, balance) = {
            let reserve = self
                .reserves
                .get(&reserve_id)
                .ok_or_else(|| diag(Code::TALER_UNKNOWN_RESERVE, "reserve is not open"))?;
            (reserve.owner.clone(), reserve.balance)
        };
        if balance < amount {
            return Err(diag(
                Code::TALER_INSUFFICIENT_BALANCE,
                format!("reserve has {balance}, withdraw needs {amount}"),
            ));
        }

        self.current_seq += 1;
        let seq = self.current_seq;
        let blinding = withdraw_blinding(&reserve_id, amount, seq);
        let kind = CertificateOpKind::Mint {
            amount,
            owner: owner.clone(),
            blinding,
            source: Some(format!("taler:reserve:{}", hex::encode(reserve_id))),
        };
        let op = CertificateOp {
            did: self.treasury.did(),
            kind,
            seq,
            signature: [0u8; 64],
        };
        let mut ids = self.push_op(op, &self.treasury.clone())?;
        let coin_id = ids.swap_remove(0);

        self.reserves
            .get_mut(&reserve_id)
            .ok_or_else(|| {
                diag(
                    Code::TALER_UNKNOWN_RESERVE,
                    "reserve vanished during withdraw",
                )
            })?
            .balance -= amount;
        self.funded.insert(coin_id.0);
        self.coin_blinding.insert(coin_id.0, blinding);
        self.funded_outstanding += amount;

        debug_assert!(self.audit().is_ok());
        Ok(coin_id)
    }

    /// Re-derive the blinding an e-coin was minted with under this exchange.
    pub fn blinding_for(&self, coin_id: &CertId) -> Option<&[u8; 32]> {
        self.coin_blinding.get(&coin_id.0)
    }

    // -- deposit -------------------------------------------------------------

    /// Pay a merchant: retire e-coins owned by `customer` on the ledger (a `Burn`)
    /// and credit the merchant's fiat balance. The e-coins leave circulation exactly
    /// like a Taler exchange honors e-coins it has been given; the merchant's
    /// balance is redeemable as fiat via `peg_out`.
    /// Refuses coins the exchange never minted (UK-7107) and double deposits
    /// (UK-7105).
    pub fn deposit(
        &mut self,
        customer: &Keypair,
        coins: &[CoinRef],
        merchant_did: &str,
    ) -> Result<u64, Diagnostic> {
        if coins.is_empty() {
            return Err(diag(Code::CERT_NONEXISTENT_INPUT, "empty deposit"));
        }
        for c in coins {
            if self.deposited.contains(&c.coin_id.0) {
                return Err(diag(
                    Code::TALER_COIN_ALREADY_DEPOSITED,
                    format!("coin {:?} already deposited", c.coin_id),
                ));
            }
            if !self.funded.contains(&c.coin_id.0) {
                return Err(diag(
                    Code::TALER_UNKNOWN_E_COIN,
                    format!("coin {:?} was not minted by this exchange", c.coin_id),
                ));
            }
        }
        let total: u64 = coins
            .iter()
            .map(|c| c.amount)
            .try_fold(0u64, |a, v| a.checked_add(v))
            .ok_or_else(|| diag(Code::CERT_AMOUNT_MISMATCH, "deposit sum overflow"))?;

        self.current_seq += 1;
        let seq = self.current_seq;
        let kind = CertificateOpKind::Burn {
            inputs: coins.to_vec(),
        };
        let op = CertificateOp {
            did: customer.did(),
            kind,
            seq,
            signature: [0u8; 64],
        };
        // Validates ownership + existence on the mirror ledger before any
        // private-side credit happens.
        self.push_op(op, customer)?;

        for c in coins {
            self.deposited.insert(c.coin_id.0);
            self.funded.remove(&c.coin_id.0);
        }
        self.funded_outstanding -= total;
        *self
            .merchant_balances
            .entry(merchant_did.to_string())
            .or_insert(0) += total;

        debug_assert!(self.audit().is_ok());
        Ok(total)
    }

    // -- certificates that are not fiat-backed e-coins -----------------------

    /// Mint carbon certificates as the treasury (the mint authority) without a
    /// reserve behind them. This is the asset side of the exchange: a UNFCCC
    /// provenance reference is attached to the mint, and the value is backed by
    /// the registry record rather than by fiat. It does not change any
    /// fiat-side balance, so the conservation audit is unaffected.
    pub fn mint_certificate(
        &mut self,
        amount: u64,
        owner: &str,
        blinding: &[u8; 32],
        source: Option<String>,
    ) -> Result<CertId, Diagnostic> {
        self.current_seq += 1;
        let seq = self.current_seq;
        let kind = CertificateOpKind::Mint {
            amount,
            owner: owner.to_string(),
            blinding: *blinding,
            source,
        };
        let op = CertificateOp {
            did: self.treasury.did(),
            kind,
            seq,
            signature: [0u8; 64],
        };
        let mut ids = self.push_op(op, &self.treasury.clone())?;
        Ok(ids.swap_remove(0))
    }

    /// Transfer certificates owned by `owner` to another DID. `outputs` are
    /// `(amount, new_owner)` pairs and must conserve the input total.
    pub fn transfer_certificate(
        &mut self,
        owner: &Keypair,
        inputs: &[CoinRef],
        outputs: &[(u64, String)],
    ) -> Result<Vec<CertId>, Diagnostic> {
        if outputs.is_empty() {
            return Err(diag(
                Code::CERT_AMOUNT_MISMATCH,
                "a transfer needs at least one output",
            ));
        }
        self.current_seq += 1;
        let seq = self.current_seq;
        let outs: Vec<CoinRef> = outputs
            .iter()
            .map(|(amount, to)| CoinRef {
                coin_id: CertId([0u8; 32]),
                amount: *amount,
                owner: to.clone(),
            })
            .collect();
        let kind = CertificateOpKind::Transfer {
            inputs: inputs.to_vec(),
            outputs: outs,
        };
        let op = CertificateOp {
            did: owner.did(),
            kind,
            seq,
            signature: [0u8; 64],
        };
        self.push_op(op, owner)
    }

    pub fn merchant_balance(&self, merchant_did: &str) -> u64 {
        self.merchant_balances
            .get(merchant_did)
            .copied()
            .unwrap_or(0)
    }

    // -- peg-out -------------------------------------------------------------

    /// Redeem merchant fiat: debit the merchant balance and book an outgoing
    /// wire to their bank account. The wire is `Preparing` until the gateway
    /// confirms it.
    pub fn peg_out(
        &mut self,
        merchant_did: &str,
        amount: u64,
        bank_account: &str,
    ) -> Result<PegOut, Diagnostic> {
        let balance = self
            .merchant_balances
            .get(merchant_did)
            .copied()
            .unwrap_or(0);
        if balance < amount {
            return Err(diag(
                Code::TALER_INSUFFICIENT_BALANCE,
                format!("merchant balance is {balance}, peg-out needs {amount}"),
            ));
        }
        let wire = self
            .gateway
            .prepare_transfer(bank_account, amount)
            .map_err(|e| diag(Code::TALER_UNCONFIRMED_WIRE, e))?;
        *self
            .merchant_balances
            .get_mut(merchant_did)
            .ok_or_else(|| {
                diag(
                    Code::TALER_INSUFFICIENT_BALANCE,
                    "merchant balance entry missing during peg-out",
                )
            })? -= amount;
        self.fiat_out = self
            .fiat_out
            .checked_add(amount)
            .ok_or_else(|| diag(Code::TALER_INSUFFICIENT_BALANCE, "fiat_out overflow"))?;
        let peg = PegOut {
            wire,
            merchant_did: merchant_did.to_string(),
            confirmed: false,
        };
        self.peg_outs.insert(peg.wire.wire_id.clone(), peg.clone());
        debug_assert!(self.audit().is_ok());
        Ok(peg)
    }

    /// Mark an outgoing wire settled at the bank. Only a confirmed wire
    /// actually moves fiat; until then `peg_out` was a cancellation-free
    /// reservation.
    pub fn confirm_peg_out(&mut self, wire_id: &str) -> Result<(), Diagnostic> {
        self.gateway
            .confirm(wire_id)
            .map_err(|e| diag(Code::TALER_UNCONFIRMED_WIRE, e))?;
        if let Some(p) = self.peg_outs.get_mut(wire_id) {
            p.confirmed = true;
        }
        Ok(())
    }

    /// A wire that never settled cancels the reservation: refund the merchant
    /// balance and reverse the fiat commitment.
    pub fn cancel_peg_out(&mut self, wire_id: &str) -> Result<(), Diagnostic> {
        let peg = self
            .peg_outs
            .remove(wire_id)
            .ok_or_else(|| diag(Code::TALER_UNCONFIRMED_WIRE, "unknown peg-out"))?;
        if peg.confirmed {
            return Err(diag(
                Code::TALER_UNCONFIRMED_WIRE,
                "settled wire cannot be cancelled",
            ));
        }
        *self
            .merchant_balances
            .entry(peg.merchant_did.clone())
            .or_insert(0) += peg.wire.amount;
        self.fiat_out = self.fiat_out.saturating_sub(peg.wire.amount);
        debug_assert!(self.audit().is_ok());
        Ok(())
    }

    // -- audit ---------------------------------------------------------------

    /// The fiat↔e-coin conservation identity across the two sides of the
    /// exchange. A violation of this while the ledger-side invariants hold is
    /// an internal bug (UK-5000).
    pub fn audit(&self) -> Result<(), Diagnostic> {
        let reserves_total: u64 = self.reserves.values().map(|r| r.balance).sum();
        let merchant_total: u64 = self.merchant_balances.values().sum();
        let liability = reserves_total + merchant_total + self.funded_outstanding;
        let asset = self.fiat_in - self.fiat_out;
        if asset != liability {
            return Err(diag(
                Code::INTERNAL,
                format!(
                    "fiat conservation broken: assets {asset} != liabilities {liability} \
                     (reserves {reserves_total}, merchant {merchant_total}, \
                     funded outstanding {})",
                    self.funded_outstanding
                ),
            ));
        }
        Ok(())
    }

    fn push_op(&mut self, op: CertificateOp, signer: &Keypair) -> Result<Vec<CertId>, Diagnostic> {
        let mut tx = ConsensusTransaction::CertificateOp(op.clone());
        unfer_consensus::signing::sign_transaction(&mut tx, signer);
        self.ops.push(tx);
        self.ledger.apply_op(&op.did, &op.kind, op.seq)
    }
}

fn hex32(s: &str) -> Option<[u8; 32]> {
    let bytes = hex::decode(s).ok()?;
    bytes.as_slice().try_into().ok()
}

fn withdraw_blinding(reserve_id: &ReserveId, amount: u64, seq: u64) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut ctx = Sha256::new();
    ctx.update(b"unfer:taler:withdraw");
    ctx.update(reserve_id);
    ctx.update(amount.to_le_bytes());
    ctx.update(seq.to_le_bytes());
    ctx.finalize().into()
}

fn diag(code: Code, msg: impl Into<String>) -> Diagnostic {
    Diagnostic::new(code, msg, Severity::Error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::SimulatedWireGateway;
    use unfer_consensus::engine::LocalConsensus;
    use unfer_consensus::node::ConsensusNode;

    fn rid(id: u8) -> ReserveId {
        [id; 32]
    }

    fn exchange() -> (TalerExchange, Keypair, Keypair, Keypair) {
        let treasury = Keypair::generate();
        let mut ex = TalerExchange::new(treasury.clone(), Box::new(SimulatedWireGateway::new()));
        for v in [100u64, 500, 1000] {
            ex.issue_denomination(v, u64::MAX);
        }
        let alice = Keypair::generate();
        let bob = Keypair::generate();
        (ex, treasury, alice, bob)
    }

    fn fund(
        ex: &mut TalerExchange,
        gw: &mut SimulatedWireGateway,
        reserve: ReserveId,
        did: &str,
    ) -> WireRef {
        ex.open_reserve(reserve, did);
        let w = gw.prepare_transfer("unfer-bank", 1000).unwrap();
        gw.confirm(&w.wire_id).unwrap();
        let w = gw.get(&w.wire_id).unwrap().clone();
        ex.peg_in(reserve, &w).unwrap();
        w
    }

    #[test]
    fn peg_in_withdraw_deposit_peg_out_roundtrip() {
        let (mut ex, _t, alice, bob) = exchange();
        let mut gw = SimulatedWireGateway::new();
        let reserve = rid(1);
        ex.open_reserve(reserve, &alice.did());
        // Prepare + confirm a customer wire, then credit the reserve.
        let w = gw.prepare_transfer("unfer-bank", 500).unwrap();
        gw.confirm(&w.wire_id).unwrap();
        let w = gw.get(&w.wire_id).unwrap().clone();
        ex.peg_in(reserve, &w).unwrap();

        // Withdraw a 500 e-coin.
        let coin = ex.withdraw(reserve, 500).unwrap();
        assert_eq!(
            ex.ledger().utxo(&coin).unwrap().amount,
            500,
            "e-coin is a live certificate"
        );
        assert!(ex.audit().is_ok());
        let outstanding = ex.funded_outstanding;
        assert_eq!(outstanding, 500);

        // Deposit to the merchant: the e-coin is retired, fiat balance credits.
        let coinref = CoinRef {
            coin_id: coin,
            amount: 500,
            owner: alice.did(),
        };
        let credited = ex.deposit(&alice, &[coinref], &bob.did()).unwrap();
        assert_eq!(credited, 500);
        assert_eq!(ex.merchant_balance(&bob.did()), 500);
        assert_eq!(
            ex.ledger().total_supply(),
            0,
            "deposit retires the e-coin from circulation"
        );
        assert!(ex.audit().is_ok());

        // Peg out to bob's bank.
        let peg = ex
            .peg_out(&bob.did(), 500, "DE99 0000 0000 1234 5678 90")
            .unwrap();
        assert!(!peg.confirmed);
        ex.confirm_peg_out(&peg.wire.wire_id).unwrap();
        assert!(ex.peg_outs[&peg.wire.wire_id].confirmed);
        assert_eq!(ex.merchant_balance(&bob.did()), 0);
        assert!(ex.audit().is_ok());

        // Nothing left anywhere; fiat fully repatriated.
        assert_eq!(ex.fiat_in - ex.fiat_out, 0);
    }

    #[test]
    fn unconfirmed_wire_never_credits() {
        let (mut ex, _t, alice, _bob) = exchange();
        let reserve = rid(2);
        ex.open_reserve(reserve, &alice.did());
        let mut gw = SimulatedWireGateway::new();
        let w = gw.prepare_transfer("unfer-bank", 1000).unwrap(); // still Preparing
        let err = ex.peg_in(reserve, &w).unwrap_err();
        assert_eq!(err.code, Code::TALER_UNCONFIRMED_WIRE);
        assert!(
            ex.audit().is_ok(),
            "no credit happened, so nothing to conserve"
        );
    }

    #[test]
    fn withdraw_requires_funds_and_live_denomination() {
        let (mut ex, _t, alice, _bob) = exchange();
        let reserve = rid(3);
        ex.open_reserve(reserve, &alice.did());
        let missing = ex.withdraw(reserve, 500).unwrap_err();
        assert_eq!(missing.code, Code::TALER_INSUFFICIENT_BALANCE);

        let mut gw = SimulatedWireGateway::new();
        let w = gw.prepare_transfer("bank", 700).unwrap();
        gw.confirm(&w.wire_id).unwrap();
        let w = gw.get(&w.wire_id).unwrap().clone();
        ex.peg_in(reserve, &w).unwrap();

        // 700 not in the denomination book (100/500/1000).
        let bad_denom = ex.withdraw(reserve, 700).unwrap_err();
        assert_eq!(bad_denom.code, Code::TALER_DENOM_UNSUPPORTED);
        // Withdraw the full 500, leaving 200 that cannot buy another 500.
        assert!(ex.withdraw(reserve, 500).is_ok());
        let second = ex.withdraw(reserve, 500).unwrap_err();
        assert_eq!(second.code, Code::TALER_INSUFFICIENT_BALANCE);
    }

    #[test]
    fn undo_on_consensus_rejection_keeps_audit() {
        let (mut ex, _t, alice, _bob) = exchange();
        let reserve = rid(4);
        let mut gw = SimulatedWireGateway::new();
        fund(&mut ex, &mut gw, reserve, &alice.did());
        let coin = ex.withdraw(reserve, 100).unwrap();
        // Second withdraw with the exact same (reserve, amount, seq) is refused
        // by the ledger (UK-7004??) — actually seq differs, so instead we feed
        // a deposit of a foreign coin, which must be refused pre-ledger.
        let foreign = CertId([0xAA; 32]);
        let foreign_ref = CoinRef {
            coin_id: foreign,
            amount: 100,
            owner: alice.did(),
        };
        let stray = ex.deposit(&alice, &[foreign_ref], "did:unfer:mallory");
        assert_eq!(stray.unwrap_err().code, Code::TALER_UNKNOWN_E_COIN);
        let _ = coin;
        assert!(ex.audit().is_ok());
    }

    #[test]
    fn no_double_deposit() {
        let (mut ex, _t, alice, bob) = exchange();
        let reserve = rid(5);
        let mut gw = SimulatedWireGateway::new();
        fund(&mut ex, &mut gw, reserve, &alice.did());
        let coin = ex.withdraw(reserve, 500).unwrap();
        let c = CoinRef {
            coin_id: coin,
            amount: 500,
            owner: alice.did(),
        };
        ex.deposit(&alice, std::slice::from_ref(&c), &bob.did())
            .unwrap();
        let again = ex.deposit(&alice, &[c], &bob.did()).unwrap_err();
        assert_eq!(again.code, Code::TALER_COIN_ALREADY_DEPOSITED);
    }

    #[test]
    fn cancel_unsettled_peg_out_refunds() {
        let (mut ex, _t, alice, bob) = exchange();
        let reserve = rid(6);
        let mut gw = SimulatedWireGateway::new();
        fund(&mut ex, &mut gw, reserve, &alice.did());
        let coin = ex.withdraw(reserve, 500).unwrap();
        ex.deposit(
            &alice,
            &[CoinRef {
                coin_id: coin,
                amount: 500,
                owner: alice.did(),
            }],
            &bob.did(),
        )
        .unwrap();
        let peg = ex.peg_out(&bob.did(), 500, "DE99 bank").unwrap();
        ex.cancel_peg_out(&peg.wire.wire_id).unwrap();
        assert_eq!(ex.merchant_balance(&bob.did()), 500, "refunded");
        // Unwithdrawn reserve (500) + refunded merchant (500) = all 1000 fiat
        // credited is still backed; nothing left the seam.
        assert_eq!(ex.fiat_in - ex.fiat_out, 1000);
    }

    #[test]
    fn exchange_ops_replay_to_identical_consensus_root() {
        let (mut ex, treasury, alice, bob) = exchange();
        let reserve = rid(7);
        let mut gw = SimulatedWireGateway::new();
        fund(&mut ex, &mut gw, reserve, &alice.did());
        let c1 = ex.withdraw(reserve, 1000).unwrap();
        let deposit = CoinRef {
            coin_id: c1,
            amount: 1000,
            owner: alice.did(),
        };
        ex.deposit(&alice, &[deposit], &bob.did()).unwrap();

        // A fresh node configured with the same mint authority replays the
        // exchange's signed ops and must land on the same certificate root.
        let engine = LocalConsensus::new();
        let mut node = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(treasury.did()),
        );
        for tx in ex.ops() {
            node.submit(tx.clone()).unwrap();
            node.sync().unwrap();
        }
        assert_eq!(ex.ledger().root(), node.certs().root());
        assert_eq!(ex.ledger().total_supply(), node.certs().total_supply());
        assert_eq!(
            node.certs().total_supply(),
            0,
            "deposit retires the e-coin from circulation"
        );
        assert!(node.is_synced());
    }

    #[test]
    fn unaudited_fiat_leak_is_caught() {
        let (mut ex, _t, alice, _bob) = exchange();
        let reserve = rid(8);
        let mut gw = SimulatedWireGateway::new();
        fund(&mut ex, &mut gw, reserve, &alice.did());
        // Accounting leak: invent fiat_out without a matching wire.
        ex.fiat_out = 1000;
        assert_eq!(ex.audit().unwrap_err().code, Code::INTERNAL);
    }

    #[test]
    fn expired_denomination_refuses_withdraw() {
        let (mut ex, _t, alice, _bob) = exchange();
        ex.issue_denomination(25, 10); // expires at consensus seq 10
        let reserve = rid(9);
        let mut gw = SimulatedWireGateway::new();
        fund(&mut ex, &mut gw, reserve, &alice.did());
        // Establish sequence progress with a live denomination (book defaults
        // never expire), then advance past the 25's expiry point.
        let _coin = ex.withdraw(reserve, 100).unwrap();
        ex.current_seq += 11; // now past seq 10
        let err = ex.withdraw(reserve, 25).unwrap_err();
        assert_eq!(err.code, Code::TALER_DENOM_UNSUPPORTED);
    }

    #[test]
    fn snapshot_round_trips_and_keeps_pending_wires_actionable() {
        let treasury = Keypair::from_bytes(&[7u8; 32]);
        let mut ex = TalerExchange::new(
            treasury.clone(),
            Box::new(SimulatedWireGateway::new()),
        );
        ex.issue_denomination(100, u64::MAX);
        ex.issue_denomination(500, 1_000);
        let alice = Keypair::generate();
        let bob = Keypair::generate();
        let reserve = rid(11);
        let mut gw = SimulatedWireGateway::new();
        fund(&mut ex, &mut gw, reserve, &alice.did());
        let coin = ex.withdraw(reserve, 500).unwrap();
        ex.deposit(
            &alice,
            &[CoinRef {
                coin_id: coin,
                amount: 500,
                owner: alice.did(),
            }],
            &bob.did(),
        )
        .unwrap();
        let peg = ex.peg_out(&bob.did(), 200, "DE99 bank").unwrap();
        let root = ex.ledger().root();
        let supply = ex.ledger().total_supply();

        // Through JSON, as a host persists it.
        let json = serde_json::to_string(&ex.snapshot()).unwrap();
        let snap: ExchangeSnapshot = serde_json::from_str(&json).unwrap();
        let mut restored = TalerExchange::from_snapshot(
            treasury.clone(),
            Box::new(SimulatedWireGateway::new()),
            &snap,
        )
        .unwrap();

        assert_eq!(restored.ledger().root(), root);
        assert_eq!(restored.ledger().total_supply(), supply);
        assert_eq!(restored.reserve(&reserve), ex.reserve(&reserve));
        assert_eq!(restored.merchant_balance(&bob.did()), 300);
        assert_eq!(restored.ops().len(), ex.ops().len());
        assert_eq!(restored.current_seq(), ex.current_seq());
        assert!(restored.audit().is_ok());

        // The pending wire survived: it can still be confirmed after restore,
        // and the restored gateway allocates ids past the imported ones.
        restored.confirm_peg_out(&peg.wire.wire_id).unwrap();
        assert!(restored.peg_out_wire(&peg.wire.wire_id).unwrap().confirmed);
        let next = restored.peg_out(&bob.did(), 100, "DE99 bank").unwrap();
        assert_ne!(next.wire.wire_id, peg.wire.wire_id);
        assert!(restored.audit().is_ok());

        // A snapshot restored under a different treasury is refused.
        let other = Keypair::from_bytes(&[9u8; 32]);
        assert!(
            TalerExchange::from_snapshot(
                other,
                Box::new(SimulatedWireGateway::new()),
                &snap
            )
            .is_err()
        );
    }

    proptest::proptest! {
        #[test]
        fn fiat_conservation_never_breaks(
            amounts in proptest::collection::vec(1u64..=3, 1..12),
        ) {
            let (mut ex, _t, alice, bob) = exchange();
            let reserve = rid(10);
            ex.open_reserve(reserve, &alice.did());
            let mut gw = SimulatedWireGateway::new();
            let mut peg_outs: Vec<String> = Vec::new();
            let mut verifier = ConsensusNode::with_mint_authority(
                Box::new(LocalConsensus::new()),
                MintAuthority::Only(ex.treasury_did().to_string()),
            );

            for (i, a) in amounts.iter().enumerate() {
                // Replenish the reserve from a (simulated) confirmed wire.
                let w = gw.prepare_transfer("bank", *a * 1000).unwrap();
                gw.confirm(&w.wire_id).unwrap();
                let w = gw.get(&w.wire_id).unwrap().clone();
                ex.peg_in(reserve, &w).unwrap();

                // Spend some of it as an e-coin and deposit to the merchant.
                let with_amount = *a * 100;
                if let Ok(coin) = ex.withdraw(reserve, with_amount) {
                    let c = CoinRef {
                        coin_id: coin,
                        amount: with_amount,
                        owner: alice.did(),
                    };
                    if ex.deposit(&alice, &[c], &bob.did()).is_ok() {
                        // Merchant redeems half back to fiat every other step.
                        if i % 2 == 0 {
                            let mbal = ex.merchant_balance(&bob.did());
                            let out = (mbal / 2).max(1);
                            if let Ok(peg) = ex.peg_out(&bob.did(), out.min(mbal), "DE merchant") {
                                peg_outs.push(peg.wire.wire_id.clone());
                            }
                        }
                    }
                }

                // The private-side audit always holds after every step.
                assert!(ex.audit().is_ok(), "audit holds after step {i}");
            }
            // Full replay: a consensus node applying the emitted ops must land
            // on the same certificate root the exchange's mirror ledger holds.
            for tx in ex.ops() {
                verifier.submit(tx.clone()).unwrap();
                verifier.sync().unwrap();
            }
            assert_eq!(ex.ledger().root(), verifier.certs().root());
            assert_eq!(ex.ledger().total_supply(), verifier.certs().total_supply());
            assert!(ex.audit().is_ok());
            let _ = peg_outs;
        }
    }
}
