//! Math catastrophe bond + probability market settlement over the certificate
//! ledger — the operator-side analogue of [`AuctionService`](crate::auction::AuctionService).
//!
//! The deterministic engines ([`MathBondLedger`](unfer_consensus::mathbond::MathBondLedger)
//! and [`MarketLedger`](unfer_consensus::mathbond_market::MarketLedger)) only
//! decide state. This service turns those decisions into settled value with
//! ordinary, conserving `CertificateOp`s, exactly like the auction service:
//! every e-coin is rowed into (and out of) DIDs derived deterministically from
//! the operator's master key, so a peer `ConsensusNode` replaying the same
//! signed log lands on the identical certificate root and the same bond/market
//! state.
//!
//! # Math bond economics (documented, conserving)
//!
//! At settlement the pool holds `principal` (the sponsor's collateral) plus
//! `invested` (investors' e-coins):
//!
//! - **Triggered** (nanoda verified a proof): the investors' money is the
//!   catastrophe — it is paid out as the **researcher bounty**; the sponsor
//!   keeps its collateral as the **catastrophe payment**. Investors are wiped
//!   out (the point of a cat bond).
//! - **Matured** (no proof by `maturity_seq`): every investor recovers their
//!   principal plus a `coupon_rate_bps` coupon; the coupon is paid from the
//!   sponsor's collateral, which keeps the remainder.
//!
//! # Probability market (vAMM + NegRisk)
//!
//! All pool cash lives in a deterministic per-pool DID (`pool_did`). LP
//! deposits and buy proceeds are rowed in; sell redemptions, liquidity
//! withdrawals and post-resolution claims are rowed out by spending the pool's
//! coins (single-owner multi-input transfers). The pool is the counterparty:
//! when the pool resolves, winning token holders redeem pro-rata against the
//! pool reserve and LPs collect the accrued trading fees.
//!
//! # Sequencing
//!
//! Every op the service emits carries a single global monotonic seq equal to
//! its position in [`ops`](BondMarketService::ops) — the position it will
//! occupy in the consensus log. This mirrors `LocalConsensus` exactly, which
//! matters for the bond's absolute `maturity_seq` check and for the market's
//! `trigger_seq` signal (both are consensus-log positions).
//!
//! Error codes: UK-7401..UK-7419 + UK-7002/7003/7005 + UK-7107-analogue
//! (see `unfer_protocol::codes`).

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use unfer_consensus::certs::{CertificateLedger, MintAuthority, commit_coin};
use unfer_consensus::mathbond::{MathBondLedger, compute_bond_id};
use unfer_consensus::mathbond_market::{MarketLedger, compute_pool_id};
use unfer_consensus::signing::{Keypair, sign_transaction};
use unfer_protocol::{
    CertId, CertificateOp, CertificateOpKind, Code, CoinRef, ConsensusTransaction, Diagnostic,
    MarketOp, MarketOpKind, MathBondId, MathBondOp, MathBondOpKind, MathBondState, MathBondTrigger,
    NegRiskOutcome, OutcomeId, PoolId, Severity,
};

const BLINDING: [u8; 32] = [0u8; 32];

/// The sponsor's locked collateral for one bond, sitting in the deterministic
/// collateral DID.
#[derive(Debug, Clone)]
pub struct CollateralHold {
    pub bond_id: MathBondId,
    pub escrowed: CertId,
    pub origin: CertId,
    pub sponsor: String,
    pub principal: u64,
}

/// One investor's escrowed e-coin for a bond, sitting in the per-investor
/// investment DID.
#[derive(Debug, Clone)]
pub struct InvestmentHold {
    pub bond_id: MathBondId,
    pub escrowed: CertId,
    pub origin: CertId,
    pub investor: String,
    pub amount: u64,
}

/// One e-coin owned by the pool DID (the pool's cash is a set of these).
#[derive(Debug, Clone)]
pub struct PoolCoin {
    pub pool_id: PoolId,
    pub coin: CertId,
    pub amount: u64,
}

/// The operator-side bond + market service: runs the deterministic engines and
/// settles every outcome with ordinary certificate transfers.
pub struct BondMarketService {
    operator: Keypair,
    certs: CertificateLedger,
    bonds: MathBondLedger,
    market: MarketLedger,
    collateral: HashMap<[u8; 32], CollateralHold>,
    investments: HashMap<[u8; 32], InvestmentHold>,
    pool_coins: HashMap<[u8; 32], Vec<PoolCoin>>,
    ops: Vec<ConsensusTransaction>,
}

impl BondMarketService {
    pub fn new(operator: Keypair, authority: MintAuthority) -> Self {
        Self {
            operator,
            certs: CertificateLedger::new(authority),
            bonds: MathBondLedger::new(),
            market: MarketLedger::new(),
            collateral: HashMap::new(),
            investments: HashMap::new(),
            pool_coins: HashMap::new(),
            ops: Vec::new(),
        }
    }

    pub fn operator_did(&self) -> String {
        self.operator.did()
    }

    pub fn bonds(&self) -> &MathBondLedger {
        &self.bonds
    }

    pub fn market(&self) -> &MarketLedger {
        &self.market
    }

    pub fn certs(&self) -> &CertificateLedger {
        &self.certs
    }

    pub fn collateral_hold(&self, bond_id: &MathBondId) -> Option<&CollateralHold> {
        self.collateral.get(&bond_id.0)
    }

    pub fn investment_hold(&self, escrowed: &CertId) -> Option<&InvestmentHold> {
        self.investments.get(&escrowed.0)
    }

    /// Every signed transaction this service produced or observed, in order.
    /// Feed these to a `ConsensusNode` to replay the identical state.
    pub fn ops(&self) -> &[ConsensusTransaction] {
        &self.ops
    }

    /// Feed an external certificate op (e.g. the authority's mint that created
    /// a sponsor's collateral or an investor's e-coin) into the mirror ledger.
    pub fn observe(&mut self, tx: ConsensusTransaction) -> Result<(), Diagnostic> {
        match &tx {
            ConsensusTransaction::CertificateOp(_) => {}
            other => {
                return Err(Diagnostic::new(
                    Code::INTERNAL,
                    format!("bond/market service only observes certificate ops, got {other:?}"),
                    Severity::Error,
                ));
            }
        }
        self.apply_transaction(&tx)?;
        self.ops.push(tx);
        Ok(())
    }

    // ── deterministic DIDs -------------------------------------------------

    /// The DID holding a bond's sponsor collateral.
    pub fn collateral_did(&self, bond_id: &MathBondId) -> String {
        self.collateral_key(bond_id).did()
    }

    /// The DID holding one investor's e-coin for a bond.
    pub fn investment_did(&self, bond_id: &MathBondId, investor: &str, origin: CertId) -> String {
        self.investment_key(bond_id, investor, origin).did()
    }

    /// The DID holding all of a market pool's cash.
    pub fn pool_did(&self, pool_id: &PoolId) -> String {
        self.pool_key(pool_id).did()
    }

    fn collateral_key(&self, bond_id: &MathBondId) -> Keypair {
        self.derive(b"unfer:mathbond:collateral:v1", &[&bond_id.0])
    }

    fn investment_key(&self, bond_id: &MathBondId, investor: &str, origin: CertId) -> Keypair {
        self.derive(
            b"unfer:mathbond:investment:v1",
            &[&bond_id.0, investor.as_bytes(), &origin.0],
        )
    }

    fn pool_key(&self, pool_id: &PoolId) -> Keypair {
        self.derive(b"unfer:market:pool:v1", &[&pool_id.0])
    }

    /// Deterministic per-(domain, parts) spending key: only the marketplace
    /// operator can regenerate it.
    fn derive(&self, domain: &[u8], parts: &[&[u8]]) -> Keypair {
        let mut ctx = Sha256::new();
        ctx.update(domain);
        ctx.update(self.operator.public_key());
        for p in parts {
            ctx.update(p);
        }
        Keypair::from_bytes(&ctx.finalize().into())
    }

    // ── math bond ops ------------------------------------------------------

    /// Issue a bond: the sponsor's `funding` coin (face value `principal`) is
    /// rowed into the deterministic collateral DID, then the signed `Issue` is
    /// emitted.
    pub fn issue_bond(
        &mut self,
        sponsor: &Keypair,
        trigger: MathBondTrigger,
        principal: u64,
        coupon_rate_bps: u64,
        maturity_seq: u64,
        researcher_did: &str,
        funding: CertId,
    ) -> Result<MathBondId, Diagnostic> {
        let bond_id = compute_bond_id(
            &trigger,
            &sponsor.did(),
            principal,
            coupon_rate_bps,
            maturity_seq,
            researcher_did,
        );
        if self.bonds.bond(&bond_id).is_some() {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                "a bond with these terms already exists",
                Severity::Error,
            ));
        }
        self.check_funding(funding, &sponsor.did(), principal)?;

        let escrow_did = self.collateral_did(&bond_id);
        let escrowed = commit_coin(principal, &escrow_did, &BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: funding,
                amount: principal,
                owner: sponsor.did(),
            }],
            outputs: vec![CoinRef {
                coin_id: escrowed,
                amount: principal,
                owner: escrow_did.clone(),
            }],
        };
        self.emit_cert(sponsor, &sponsor.did(), kind)?;
        self.collateral.insert(
            bond_id.0,
            CollateralHold {
                bond_id,
                escrowed,
                origin: funding,
                sponsor: sponsor.did(),
                principal,
            },
        );

        let kind = MathBondOpKind::Issue {
            trigger,
            principal,
            coupon_rate_bps,
            maturity_seq,
            researcher_did: researcher_did.to_string(),
        };
        self.apply_bond(sponsor, kind)?;
        Ok(bond_id)
    }

    /// An investor funds a bond: their e-coin is rowed into the per-investor
    /// investment DID, then the signed `Invest` is emitted.
    pub fn invest(
        &mut self,
        investor: &Keypair,
        bond_id: MathBondId,
        amount: u64,
        funding: CertId,
    ) -> Result<(), Diagnostic> {
        let bond = self
            .bonds
            .bond(&bond_id)
            .ok_or_else(|| {
                Diagnostic::new(Code::MATHBOND_UNKNOWN, "unknown bond id", Severity::Error)
            })?
            .clone();
        if bond.state != MathBondState::Issued && bond.state != MathBondState::Funded {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                format!("bond is {:?}, cannot invest", bond.state),
                Severity::Error,
            ));
        }
        if amount == 0 {
            return Err(Diagnostic::new(
                Code::MATHBOND_OVERFUNDED,
                "investment amount must be positive",
                Severity::Error,
            ));
        }
        if bond.invested.saturating_add(amount) > bond.principal {
            return Err(Diagnostic::new(
                Code::MATHBOND_OVERFUNDED,
                format!(
                    "invested {} + {amount} would exceed principal {}",
                    bond.invested, bond.principal
                ),
                Severity::Error,
            ));
        }
        self.check_funding(funding, &investor.did(), amount)?;

        let escrow_did = self.investment_did(&bond_id, &investor.did(), funding);
        let escrowed = commit_coin(amount, &escrow_did, &BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: funding,
                amount,
                owner: investor.did(),
            }],
            outputs: vec![CoinRef {
                coin_id: escrowed,
                amount,
                owner: escrow_did.clone(),
            }],
        };
        self.emit_cert(investor, &investor.did(), kind)?;
        self.investments.insert(
            escrowed.0,
            InvestmentHold {
                bond_id,
                escrowed,
                origin: funding,
                investor: investor.did(),
                amount,
            },
        );

        let kind = MathBondOpKind::Invest { bond_id, amount };
        self.apply_bond(investor, kind)?;
        Ok(())
    }

    /// The designated researcher submits a proof export. The signed
    /// `SubmitProof` is emitted; nanoda's deterministic verdict runs in the
    /// mirror ledger. Returns `true` when the trigger fired.
    pub fn submit_proof(
        &mut self,
        researcher: &Keypair,
        bond_id: MathBondId,
        export_bytes: Vec<u8>,
    ) -> Result<bool, Diagnostic> {
        let kind = MathBondOpKind::SubmitProof {
            bond_id,
            export_bytes,
        };
        let outcome = self.apply_bond(researcher, kind)?;
        Ok(outcome.is_some())
    }

    /// Record that the bond reached `maturity_seq` without a trigger. Anyone
    /// may submit it; the seq check is the enforcement.
    pub fn mature(&mut self, actor: &Keypair, bond_id: MathBondId) -> Result<(), Diagnostic> {
        self.apply_bond(actor, MathBondOpKind::Mature { bond_id })?;
        Ok(())
    }

    /// Finalize a bond and row the distribution:
    ///
    /// - **Triggered**: every investment coin → the researcher (bounty);
    ///   the collateral coin → the sponsor (catastrophe payment).
    /// - **Matured**: every investment coin → its investor (principal back);
    ///   the collateral coin → sponsor change plus one coupon coin per
    ///   investor (`amount * coupon_rate_bps / 10000`, floor).
    ///
    /// The signed `Settle` is emitted last, so the deterministic engine's
    /// finality lands on the log after the money has moved.
    pub fn settle(&mut self, actor: &Keypair, bond_id: MathBondId) -> Result<(), Diagnostic> {
        let bond = self
            .bonds
            .bond(&bond_id)
            .ok_or_else(|| {
                Diagnostic::new(Code::MATHBOND_UNKNOWN, "unknown bond id", Severity::Error)
            })?
            .clone();
        let collateral = self.collateral.get(&bond_id.0).cloned().ok_or_else(|| {
            Diagnostic::new(
                Code::ESCROW_UNKNOWN,
                "collateral hold not found for this bond",
                Severity::Error,
            )
        })?;
        let mut investments: Vec<InvestmentHold> = self
            .investments
            .values()
            .filter(|h| h.bond_id == bond_id)
            .cloned()
            .collect();
        investments.sort_by_key(|h| (h.escrowed.0, h.investor.clone()));

        match bond.state {
            MathBondState::Triggered => {
                // Bounty: all invested e-coins → the researcher.
                let researcher = bond.researcher_did.clone();
                let mut bounty = 0u64;
                for h in &investments {
                    self.row_investment_out(h, &researcher)?;
                    bounty += h.amount;
                }
                // Catastrophe payment: the collateral → the sponsor.
                self.row_collateral_out(&collateral, &collateral.sponsor)?;
                let _ = bounty; // audit: bounty == bond.invested
            }
            MathBondState::Matured => {
                // Principal back: each investment coin → its investor.
                for h in &investments {
                    self.row_investment_out(h, &h.investor)?;
                }
                // Coupons from the collateral: one output per investor plus
                // the sponsor's change — a single conserving transfer.
                let mut outputs = Vec::new();
                let mut coupon_total = 0u64;
                for h in &investments {
                    let coupon = h.amount * bond.coupon_rate_bps / 10_000;
                    if coupon > 0 {
                        outputs.push(CoinRef {
                            coin_id: commit_coin(coupon, &h.investor, &BLINDING),
                            amount: coupon,
                            owner: h.investor.clone(),
                        });
                        coupon_total += coupon;
                    }
                }
                let sponsor_change = collateral.principal - coupon_total;
                outputs.push(CoinRef {
                    coin_id: commit_coin(sponsor_change, &collateral.sponsor, &BLINDING),
                    amount: sponsor_change,
                    owner: collateral.sponsor.clone(),
                });
                let col_did = self.collateral_did(&bond_id);
                let kind = CertificateOpKind::Transfer {
                    inputs: vec![CoinRef {
                        coin_id: collateral.escrowed,
                        amount: collateral.principal,
                        owner: col_did.clone(),
                    }],
                    outputs,
                };
                self.emit_cert(&self.collateral_key(&bond_id), &col_did, kind)?;
            }
            _ => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_WRONG_STATE,
                    format!(
                        "bond is {:?}; only a Triggered or Matured bond settles",
                        bond.state
                    ),
                    Severity::Error,
                ));
            }
        }

        self.apply_bond(actor, MathBondOpKind::Settle { bond_id })?;
        Ok(())
    }

    // ── market ops ---------------------------------------------------------

    /// Open the NegRisk probability market for a bond.
    pub fn open_pool(
        &mut self,
        creator: &Keypair,
        bond_id: MathBondId,
        outcomes: Vec<NegRiskOutcome>,
        fee_bps: u64,
    ) -> Result<PoolId, Diagnostic> {
        let pool_id = compute_pool_id(&bond_id);
        if self.market.pool(&pool_id).is_some() {
            return Err(Diagnostic::new(
                Code::MARKET_POOL_EXISTS,
                "pool already exists for this bond",
                Severity::Error,
            ));
        }
        let kind = MarketOpKind::OpenNegRisk {
            bond_id,
            outcomes,
            fee_bps,
        };
        self.apply_market(creator, kind)?;
        Ok(pool_id)
    }

    /// An LP seeds the pool: their e-coin is rowed into the pool DID, then the
    /// signed `AddLiquidity` is emitted.
    pub fn add_liquidity(
        &mut self,
        lp: &Keypair,
        pool_id: PoolId,
        amount: u64,
        funding: CertId,
    ) -> Result<(), Diagnostic> {
        if self.market.pool(&pool_id).is_none() {
            return Err(Diagnostic::new(
                Code::MARKET_UNKNOWN_POOL,
                "unknown pool",
                Severity::Error,
            ));
        }
        self.check_funding(funding, &lp.did(), amount)?;

        let pool_did = self.pool_did(&pool_id);
        let coin = commit_coin(amount, &pool_did, &BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: funding,
                amount,
                owner: lp.did(),
            }],
            outputs: vec![CoinRef {
                coin_id: coin,
                amount,
                owner: pool_did,
            }],
        };
        self.emit_cert(lp, &lp.did(), kind)?;
        self.pool_coins.entry(pool_id.0).or_default().push(PoolCoin {
            pool_id,
            coin,
            amount,
        });

        let kind = MarketOpKind::AddLiquidity { pool_id, amount };
        self.apply_market(lp, kind)?;
        Ok(())
    }

    /// An LP burns shares and withdraws proportional reserve + fees from the
    /// pool DID. Returns the cash rowed to the LP.
    pub fn remove_liquidity(
        &mut self,
        lp: &Keypair,
        pool_id: PoolId,
        shares: u64,
    ) -> Result<u64, Diagnostic> {
        let kind = MarketOpKind::RemoveLiquidity { pool_id, shares };
        let cash = self.apply_market(lp, kind)?.unwrap_or(0);
        if cash > 0 {
            self.spend_pool(&pool_id, cash, &lp.did())?;
        }
        Ok(cash)
    }

    /// A trader buys outcome tokens: their e-coin is rowed into the pool DID,
    /// then the signed `BuyOutcome` is emitted.
    pub fn buy_outcome(
        &mut self,
        trader: &Keypair,
        pool_id: PoolId,
        outcome_id: OutcomeId,
        amount: u64,
        funding: CertId,
    ) -> Result<(), Diagnostic> {
        let pool = self.market.pool(&pool_id).ok_or_else(|| {
            Diagnostic::new(Code::MARKET_UNKNOWN_POOL, "unknown pool", Severity::Error)
        })?;
        if pool.pool.resolved {
            return Err(Diagnostic::new(
                Code::MARKET_POOL_RESOLVED,
                "pool is resolved",
                Severity::Error,
            ));
        }
        self.check_funding(funding, &trader.did(), amount)?;

        let pool_did = self.pool_did(&pool_id);
        let coin = commit_coin(amount, &pool_did, &BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: funding,
                amount,
                owner: trader.did(),
            }],
            outputs: vec![CoinRef {
                coin_id: coin,
                amount,
                owner: pool_did,
            }],
        };
        self.emit_cert(trader, &trader.did(), kind)?;
        self.pool_coins.entry(pool_id.0).or_default().push(PoolCoin {
            pool_id,
            coin,
            amount,
        });

        let kind = MarketOpKind::BuyOutcome {
            pool_id,
            outcome_id,
            amount,
        };
        self.apply_market(trader, kind)?;
        Ok(())
    }

    /// A trader sells outcome tokens back to the pool. Returns the net payout
    /// rowed from the pool DID to the trader.
    pub fn sell_outcome(
        &mut self,
        trader: &Keypair,
        pool_id: PoolId,
        outcome_id: OutcomeId,
        amount: u64,
    ) -> Result<u64, Diagnostic> {
        let kind = MarketOpKind::SellOutcome {
            pool_id,
            outcome_id,
            amount,
        };
        let payout = self.apply_market(trader, kind)?.unwrap_or(0);
        if payout > 0 {
            self.spend_pool(&pool_id, payout, &trader.did())?;
        }
        Ok(payout)
    }

    /// Resolve the pool. The winning outcome is derived deterministically from
    /// the bond's trigger state on the mirror ledger (`trigger_seq` when the
    /// trigger fired, `None` when the bond matured without one) — the same
    /// signal the consensus node validates against its own bond ledger.
    pub fn resolve(&mut self, actor: &Keypair, pool_id: PoolId) -> Result<OutcomeId, Diagnostic> {
        let pool = self.market.pool(&pool_id).ok_or_else(|| {
            Diagnostic::new(Code::MARKET_UNKNOWN_POOL, "unknown pool", Severity::Error)
        })?;
        let bond = self
            .bonds
            .bond(&pool.pool.bond_id)
            .ok_or_else(|| {
                Diagnostic::new(
                    Code::MATHBOND_UNKNOWN,
                    "the pool's bond does not exist on the mirror ledger",
                    Severity::Error,
                )
            })?
            .clone();
        let signal = match (bond.state, bond.trigger_seq) {
            (MathBondState::Triggered, Some(t)) => Some(t),
            (MathBondState::Settled, t) => t,
            (MathBondState::Matured, _) => None,
            _ => {
                return Err(Diagnostic::new(
                    Code::MARKET_NOT_RESOLVED,
                    format!(
                        "bond is {:?}; it must be triggered or matured before the pool can resolve",
                        bond.state
                    ),
                    Severity::Error,
                ));
            }
        };
        let kind = MarketOpKind::Resolve {
            pool_id,
            trigger_seq: signal,
        };
        self.apply_market(actor, kind)?;
        Ok(self.market.pool(&pool_id).unwrap().pool.winner.unwrap())
    }

    /// Post-resolution withdrawal: redeem winning outcome tokens (pro-rata
    /// against the pool reserve) plus the LP's share of accrued fees. Returns
    /// the payout rowed from the pool DID to the claimant (0 when there is
    /// nothing to claim).
    pub fn claim(&mut self, claimant: &Keypair, pool_id: PoolId) -> Result<u64, Diagnostic> {
        let kind = MarketOpKind::Claim { pool_id };
        let payout = self.apply_market(claimant, kind)?.unwrap_or(0);
        if payout > 0 {
            self.spend_pool(&pool_id, payout, &claimant.did())?;
        }
        Ok(payout)
    }

    // ── internal rows ------------------------------------------------------

    /// Row one investment coin to `recipient` (bounty → researcher, principal
    /// refund → investor).
    fn row_investment_out(&mut self, hold: &InvestmentHold, recipient: &str) -> Result<(), Diagnostic> {
        let did = self.investment_did(&hold.bond_id, &hold.investor, hold.origin);
        let out = commit_coin(hold.amount, recipient, &BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: hold.escrowed,
                amount: hold.amount,
                owner: did.clone(),
            }],
            outputs: vec![CoinRef {
                coin_id: out,
                amount: hold.amount,
                owner: recipient.to_string(),
            }],
        };
        let key = self.investment_key(&hold.bond_id, &hold.investor, hold.origin);
        self.emit_cert(&key, &did, kind)?;
        Ok(())
    }

    /// Row the collateral coin to `recipient` (catastrophe payment → sponsor).
    fn row_collateral_out(&mut self, hold: &CollateralHold, recipient: &str) -> Result<(), Diagnostic> {
        let did = self.collateral_did(&hold.bond_id);
        let out = commit_coin(hold.principal, recipient, &BLINDING);
        let kind = CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: hold.escrowed,
                amount: hold.principal,
                owner: did.clone(),
            }],
            outputs: vec![CoinRef {
                coin_id: out,
                amount: hold.principal,
                owner: recipient.to_string(),
            }],
        };
        let key = self.collateral_key(&hold.bond_id);
        self.emit_cert(&key, &did, kind)?;
        Ok(())
    }

    /// Spend `amount` from the pool DID's coins, rowing it to `recipient`.
    /// Coins are consumed in deterministic (id) order; change coins (owned by
    /// the pool DID) are re-inserted. The pool's coin balance always covers
    /// ledger payouts (`cash == total_reserve + lp_fees`), so the spend is
    /// total and the transfer conserves exactly.
    fn spend_pool(&mut self, pool_id: &PoolId, amount: u64, recipient: &str) -> Result<(), Diagnostic> {
        let pool_did = self.pool_did(pool_id);
        let mut coins = self.pool_coins.remove(&pool_id.0).unwrap_or_default();
        coins.sort_by_key(|c| (c.coin.0, c.amount));

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut change = Vec::new();
        let mut remaining = amount;
        for c in coins {
            if remaining == 0 {
                change.push(c);
                continue;
            }
            let take = remaining.min(c.amount);
            inputs.push(CoinRef {
                coin_id: c.coin,
                amount: c.amount,
                owner: pool_did.clone(),
            });
            remaining -= take;
            if c.amount - take > 0 {
                let ch = commit_coin(c.amount - take, &pool_did, &BLINDING);
                change.push(PoolCoin {
                    pool_id: *pool_id,
                    coin: ch,
                    amount: c.amount - take,
                });
                outputs.push(CoinRef {
                    coin_id: ch,
                    amount: c.amount - take,
                    owner: pool_did.clone(),
                });
            }
        }
        if remaining > 0 {
            return Err(Diagnostic::new(
                Code::INTERNAL,
                format!("pool cash insufficient: {amount} requested, {remaining} uncovered"),
                Severity::Error,
            ));
        }
        let out = commit_coin(amount, recipient, &BLINDING);
        outputs.push(CoinRef {
            coin_id: out,
            amount,
            owner: recipient.to_string(),
        });
        let kind = CertificateOpKind::Transfer { inputs, outputs };
        let key = self.pool_key(pool_id);
        self.emit_cert(&key, &pool_did, kind)?;
        self.pool_coins.insert(pool_id.0, change);
        Ok(())
    }

    /// A funding coin must exist, be owned by `owner` and have face value
    /// `amount` (the mint authority's withdrawal output / an unspent coin).
    fn check_funding(&self, funding: CertId, owner: &str, amount: u64) -> Result<(), Diagnostic> {
        let coin = self.certs.utxo(&funding).ok_or_else(|| {
            Diagnostic::new(
                Code::CERT_NONEXISTENT_INPUT,
                "funding certificate does not exist",
                Severity::Error,
            )
        })?;
        if coin.owner != owner {
            return Err(Diagnostic::new(
                Code::CERT_OWNER_MISMATCH,
                "funding certificate is not owned by the caller",
                Severity::Error,
            ));
        }
        if coin.amount != amount {
            return Err(Diagnostic::new(
                Code::CERT_AMOUNT_MISMATCH,
                format!("funding face value {} must equal {amount}", coin.amount),
                Severity::Error,
            ));
        }
        Ok(())
    }

    // ── emission & mirror application --------------------------------------

    /// The consensus-log position this op will occupy: every emitted op
    /// carries a single global monotonic seq equal to its position in `ops`,
    /// mirroring `LocalConsensus` exactly (the bond's `maturity_seq` check and
    /// the market's `trigger_seq` are both consensus-log positions).
    fn next_seq(&self) -> u64 {
        self.ops.len() as u64 + 1
    }

    /// Row a certificate transfer: apply to the certs mirror and push the
    /// signed op. `did` is the input owner (the signer for user coins, or the
    /// derived escrow/pool DID).
    fn emit_cert(
        &mut self,
        signer: &Keypair,
        did: &str,
        kind: CertificateOpKind,
    ) -> Result<(), Diagnostic> {
        let seq = self.next_seq();
        self.certs.apply_op(did, &kind, seq)?;
        let mut tx = ConsensusTransaction::CertificateOp(CertificateOp {
            did: did.to_string(),
            kind,
            seq,
            signature: [0u8; 64],
        });
        sign_transaction(&mut tx, signer);
        self.ops.push(tx);
        Ok(())
    }

    /// Apply a signed math bond op to the mirror and push it.
    fn apply_bond(
        &mut self,
        signer: &Keypair,
        kind: MathBondOpKind,
    ) -> Result<Option<MathBondTrigger>, Diagnostic> {
        let did = signer.did();
        let seq = self.next_seq();
        let outcome = self.bonds.apply_op(&did, &kind, seq)?;
        let mut tx = ConsensusTransaction::MathBondOp(MathBondOp {
            did,
            kind,
            seq,
            signature: [0u8; 64],
        });
        sign_transaction(&mut tx, signer);
        self.ops.push(tx);
        Ok(outcome)
    }

    /// Apply a signed market op to the mirror and push it.
    fn apply_market(
        &mut self,
        signer: &Keypair,
        kind: MarketOpKind,
    ) -> Result<Option<u64>, Diagnostic> {
        let did = signer.did();
        let seq = self.next_seq();
        let outcome = self.market.apply_op(&did, &kind, seq)?;
        let mut tx = ConsensusTransaction::MarketOp(MarketOp {
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
            ConsensusTransaction::MathBondOp(op) => {
                self.bonds.apply_op(&op.did, &op.kind, op.seq)?;
                Ok(())
            }
            ConsensusTransaction::MarketOp(op) => {
                self.market.apply_op(&op.did, &op.kind, op.seq)?;
                Ok(())
            }
            other => Err(Diagnostic::new(
                Code::INTERNAL,
                format!("bond/market service cannot apply {other:?}"),
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

    fn authority() -> Keypair {
        Keypair::generate()
    }

    fn mint_to(
        service: &mut BondMarketService,
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

    fn trigger() -> MathBondTrigger {
        MathBondTrigger {
            theorem: "P_eq_NP".to_string(),
            spec_hash: "deadbeef".to_string(),
            max_export_bytes: 16 * 1024 * 1024,
            permitted_axioms: vec![],
            strict: false,
            nat_extension: false,
            string_extension: false,
        }
    }

    fn coins_of(service: &BondMarketService, did: &str) -> u64 {
        service.certs().coins_of(did).iter().map(|c| c.amount).sum()
    }

    #[test]
    fn matured_bond_settles_investors_with_coupon() {
        let auth = authority();
        let sponsor = Keypair::generate();
        let investor = Keypair::generate();
        let mut svc = BondMarketService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let collateral = mint_to(&mut svc, &auth, &sponsor.did(), 10_000, [1u8; 32]);
        let money = mint_to(&mut svc, &auth, &investor.did(), 8_000, [2u8; 32]);

        // Issue escrows the sponsor's collateral (10k) and funds the bond.
        let bond_id = svc
            .issue_bond(&sponsor, trigger(), 10_000, 500, 5, &sponsor.did(), collateral)
            .unwrap();
        assert_eq!(
            coins_of(&svc, &svc.collateral_did(&bond_id)),
            10_000,
            "collateral rowed into the bond escrow DID"
        );
        assert_eq!(coins_of(&svc, &sponsor.did()), 0, "sponsor no longer holds the collateral");

        svc.invest(&investor, bond_id, 8_000, money).unwrap();
        // Partially funded (8000 of 10000) — the bond stays Issued and can
        // still mature; investors recover their share at settlement.
        assert_eq!(svc.bonds().bond(&bond_id).unwrap().state, MathBondState::Issued);

        // Mature at consensus position 5 (>= maturity_seq 5).
        svc.mature(&investor, bond_id).unwrap();
        assert_eq!(svc.bonds().bond(&bond_id).unwrap().state, MathBondState::Matured);

        svc.settle(&sponsor, bond_id).unwrap();
        // Investor: 8000 principal + 5% coupon (400) = 8400.
        assert_eq!(coins_of(&svc, &investor.did()), 8_400);
        // Sponsor: 10000 collateral − 400 coupon = 9600.
        assert_eq!(coins_of(&svc, &sponsor.did()), 9_600);
        // No value created or destroyed.
        assert_eq!(svc.certs().total_supply(), 18_000);
        assert_eq!(svc.bonds().bond(&bond_id).unwrap().state, MathBondState::Settled);
    }

    #[test]
    fn triggered_bond_pays_bounty_and_catastrophe() {
        // Uses the real nanoda-valid confluence export from prob_kernel's
        // fixtures (the same proof the mathbond ledger test triggers on).
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../prob_kernel/tests/fixtures/confluence.ndjson"
        );
        let export_bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("confluence.ndjson fixture not available; skipping triggered_bond_pays_bounty_and_catastrophe");
                return;
            }
        };

        let auth = authority();
        let sponsor = Keypair::generate();
        let investor = Keypair::generate();
        let researcher = Keypair::generate();
        let mut svc = BondMarketService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let collateral = mint_to(&mut svc, &auth, &sponsor.did(), 10_000, [1u8; 32]);
        let money = mint_to(&mut svc, &auth, &investor.did(), 6_000, [2u8; 32]);

        let nat_trigger = MathBondTrigger {
            theorem: "confluence".to_string(),
            spec_hash: "confluence".to_string(),
            max_export_bytes: 16 * 1024 * 1024,
            permitted_axioms: vec![
                "Quot.sound".to_string(),
                "Classical.choice".to_string(),
                "propext".to_string(),
                "Lean.trustCompiler".to_string(),
            ],
            strict: false,
            nat_extension: true,
            string_extension: true,
        };
        let bond_id = svc
            .issue_bond(&sponsor, nat_trigger.clone(), 10_000, 500, 1000, &researcher.did(), collateral)
            .unwrap();
        svc.invest(&investor, bond_id, 6_000, money).unwrap();

        let triggered = svc.submit_proof(&researcher, bond_id, export_bytes).unwrap();
        assert!(triggered, "nanoda should verify the confluence proof");
        assert_eq!(svc.bonds().bond(&bond_id).unwrap().state, MathBondState::Triggered);

        svc.settle(&sponsor, bond_id).unwrap();
        // Researcher gets the 6000 invested as bounty; the sponsor keeps its
        // 10000 collateral as the catastrophe payment; the investor is wiped out.
        assert_eq!(coins_of(&svc, &researcher.did()), 6_000, "bounty = invested");
        assert_eq!(coins_of(&svc, &sponsor.did()), 10_000, "catastrophe payment = collateral");
        assert_eq!(coins_of(&svc, &investor.did()), 0, "investors are wiped out on trigger");
        assert_eq!(svc.certs().total_supply(), 16_000);
    }

    #[test]
    fn market_lifecycle_settles_claims() {
        let auth = authority();
        let sponsor = Keypair::generate();
        let investor = Keypair::generate();
        let lp = Keypair::generate();
        let trader = Keypair::generate();
        let mut svc = BondMarketService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let collateral = mint_to(&mut svc, &auth, &sponsor.did(), 10_000, [1u8; 32]);
        let money = mint_to(&mut svc, &auth, &investor.did(), 10_000, [2u8; 32]);
        let lp_money = mint_to(&mut svc, &auth, &lp.did(), 12_000, [3u8; 32]);
        let trader_money = mint_to(&mut svc, &auth, &trader.did(), 3_000, [4u8; 32]);

        // Bond: issue → invest → mature without a trigger (maturity at pos 5).
        let bond_id = svc
            .issue_bond(&sponsor, trigger(), 10_000, 500, 5, &sponsor.did(), collateral)
            .unwrap();
        svc.invest(&investor, bond_id, 10_000, money).unwrap();
        svc.mature(&investor, bond_id).unwrap();

        // Market: one terminal "never" outcome; LP seeds 12k, trader buys 3k.
        let never = OutcomeId([1u8; 32]);
        let pool_id = svc
            .open_pool(
                &sponsor,
                bond_id,
                vec![NegRiskOutcome {
                    outcome_id: never,
                    pool_id: compute_pool_id(&bond_id),
                    label: "never".to_string(),
                    maturity_seq: u64::MAX,
                }],
                300,
            )
            .unwrap();
        svc.add_liquidity(&lp, pool_id, 12_000, lp_money).unwrap();
        svc.buy_outcome(&trader, pool_id, never, 3_000, trader_money).unwrap();
        assert_eq!(coins_of(&svc, &svc.pool_did(&pool_id)), 15_000, "pool holds LP + buy cash");

        // Bond matured without a trigger → the never outcome wins.
        let winner = svc.resolve(&sponsor, pool_id).unwrap();
        assert_eq!(winner, never);
        assert_eq!(svc.market().pool(&pool_id).unwrap().pool.winner, Some(never));

        // Trader holds ALL winning tokens → claims the whole reserve (14910);
        // the LP collects the accrued 90 of fees.
        let trader_claim = svc.claim(&trader, pool_id).unwrap();
        let lp_claim = svc.claim(&lp, pool_id).unwrap();
        assert_eq!(trader_claim, 14_910);
        assert_eq!(lp_claim, 90);
        assert_eq!(coins_of(&svc, &trader.did()), 14_910);
        assert_eq!(coins_of(&svc, &lp.did()), 90);
        // Pool drained exactly; nothing created or destroyed.
        assert_eq!(coins_of(&svc, &svc.pool_did(&pool_id)), 0);
        assert_eq!(svc.certs().total_supply(), 10_000 + 10_000 + 12_000 + 3_000);
    }

    #[test]
    fn overfunding_and_foreign_funding_rejected() {
        let auth = authority();
        let sponsor = Keypair::generate();
        let investor = Keypair::generate();
        let stranger = Keypair::generate();
        let mut svc = BondMarketService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let collateral = mint_to(&mut svc, &auth, &sponsor.did(), 10_000, [1u8; 32]);
        let money = mint_to(&mut svc, &auth, &investor.did(), 6_000, [2u8; 32]);
        let stranger_money = mint_to(&mut svc, &auth, &stranger.did(), 1_000, [3u8; 32]);

        let bond_id = svc
            .issue_bond(&sponsor, trigger(), 10_000, 500, 5, &sponsor.did(), collateral)
            .unwrap();
        // Funding owned by someone else is refused before any coin moves.
        let err = svc
            .invest(&investor, bond_id, 1_000, stranger_money)
            .unwrap_err();
        assert_eq!(err.code, Code::CERT_OWNER_MISMATCH);
        assert_eq!(coins_of(&svc, &investor.did()), 6_000);
        assert_eq!(coins_of(&svc, &stranger.did()), 1_000);
        // Investing beyond capacity is refused (ledger check, mirrored here).
        svc.invest(&investor, bond_id, 6_000, money).unwrap();
        let more = mint_to(&mut svc, &auth, &investor.did(), 5_000, [4u8; 32]);
        let err = svc.invest(&investor, bond_id, 5_000, more).unwrap_err();
        assert_eq!(err.code, Code::MATHBOND_OVERFUNDED);
        assert_eq!(coins_of(&svc, &investor.did()), 5_000, "rejected investment never escrowed");
    }

    #[test]
    fn replay_converges_the_consensus_node() {
        let auth = authority();
        let sponsor = Keypair::generate();
        let investor = Keypair::generate();
        let lp = Keypair::generate();
        let trader = Keypair::generate();
        let mut svc = BondMarketService::new(Keypair::generate(), MintAuthority::Only(auth.did()));
        let collateral = mint_to(&mut svc, &auth, &sponsor.did(), 10_000, [1u8; 32]);
        let money = mint_to(&mut svc, &auth, &investor.did(), 10_000, [2u8; 32]);
        let lp_money = mint_to(&mut svc, &auth, &lp.did(), 12_000, [3u8; 32]);
        let trader_money = mint_to(&mut svc, &auth, &trader.did(), 3_000, [4u8; 32]);

        let bond_id = svc
            .issue_bond(&sponsor, trigger(), 10_000, 500, 5, &sponsor.did(), collateral)
            .unwrap();
        svc.invest(&investor, bond_id, 10_000, money).unwrap();
        svc.mature(&investor, bond_id).unwrap();
        let never = OutcomeId([1u8; 32]);
        let pool_id = svc
            .open_pool(
                &sponsor,
                bond_id,
                vec![NegRiskOutcome {
                    outcome_id: never,
                    pool_id: compute_pool_id(&bond_id),
                    label: "never".to_string(),
                    maturity_seq: u64::MAX,
                }],
                300,
            )
            .unwrap();
        svc.add_liquidity(&lp, pool_id, 12_000, lp_money).unwrap();
        svc.buy_outcome(&trader, pool_id, never, 3_000, trader_money).unwrap();
        svc.resolve(&sponsor, pool_id).unwrap();
        svc.claim(&trader, pool_id).unwrap();
        svc.claim(&lp, pool_id).unwrap();

        // Replay the whole signed log into a ConsensusNode: every node
        // converges on the identical certificate root, bond state and winner.
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
        assert_eq!(svc.certs().total_supply(), node.certs().total_supply());
        assert_eq!(
            svc.bonds().bond(&bond_id).unwrap().state,
            node.mathbond().bond(&bond_id).unwrap().state
        );
        assert_eq!(
            svc.market().pool(&pool_id).unwrap().pool.winner,
            node.market().pool(&pool_id).unwrap().pool.winner
        );
    }
}

