//! Math bond probability market (vAMM + NegRisk, Plan R).
//!
//! The probability market for math bond triggers uses two complementary
//! designs:
//!
//! 1. **Azuro vAMM** (virtual Automated Market Maker): a singleton
//!    concentrated-liquidity pool where LPs deposit e-coins and the protocol
//!    mathematically prices the odds of the trigger firing without needing
//!    a direct buyer for every seller. The pool acts as counterparty.
//!
//! 2. **NegRisk CTF Adapter**: mutually-exclusive conditional outcomes
//!    (e.g. "triggered by 2025" vs "triggered by 2026" vs "never") share
//!    a single pool, preventing liquidity fragmentation. When one outcome
//!    resolves, the others become worthless — the "negated risk" adapter
//!    ensures the outcome prices sum to 1.
//!
//! The pricing model is a constant-product vAMM: `P(outcome_i) = reserve_i /
//! total_reserve`. When a trader buys outcome tokens, they deposit e-coins
//! into the pool's reserve and the price moves along the curve. The pool
//! acts as counterparty to all trades — no order book is needed.
//!
//! Accounting invariants (all integer-exact, deterministic across nodes):
//!
//! - `total_reserve == sum(outcome_reserves)` at all times.
//! - A buy mints `net / price` tokens (price-adjusted), where `net` is the
//!   deposit minus the trading fee; the fee accrues to the LPs (`lp_fees`).
//! - A sell redeems tokens at the current price, capped by the outcome's own
//!   reserve so the pool can never go insolvent.
//! - Resolution is NOT a caller choice: the winner is a pure function of the
//!   bond's trigger signal and the outcome maturity windows (the outcome whose
//!   window contains the trigger seq wins; a terminal "never" outcome with
//!   `maturity_seq == u64::MAX` wins when the bond matured without a trigger).
//! - After resolution, `Claim` redeems winning tokens pro-rata against the
//!   pool reserve and pays LPs their accrued fees (plus the whole reserve when
//!   nobody held winning tokens).
//!
//! Error codes: UK-7411..UK-7419 (see `unfer_protocol::codes`).

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};
use unfer_protocol::{
    Code, Diagnostic, LiquidityPool, MarketOpKind, MathBondId, NegRiskOutcome, OutcomeId, PoolId,
    PoolReport, Severity,
};

/// Internal per-trader position in an outcome.
#[derive(Debug, Clone, Default)]
pub struct OutcomePosition {
    pub amount: u64,
}

/// Full internal state for one pool.
#[derive(Debug, Clone)]
pub struct PoolState {
    pub pool: LiquidityPool,
    /// The NegRisk outcome metadata (maturity windows) in stored order. The
    /// deterministic resolution rule reads the windows from here.
    pub outcomes: Vec<NegRiskOutcome>,
    /// Per-outcome reserve lookup (faster than scanning Vec).
    pub reserve_map: HashMap<[u8; 32], u64>,
    /// Per-trader outcome positions (outcome_id → DID → amount).
    pub positions: HashMap<[u8; 32], HashMap<String, u64>>,
    /// Per-trader LP shares (DID → amount).
    pub lp_map: HashMap<String, u64>,
    /// Total outcome tokens minted per outcome (decreased when tokens are
    /// sold back and burned). The resolution payout divides the reserve by
    /// this to value each winning token.
    pub total_outcome_tokens: HashMap<[u8; 32], u64>,
    /// LP-owned trading fees accrued from every trade (the LP yield).
    pub lp_fees: u64,
}

/// The deterministic market state-transition engine.
#[derive(Debug, Default)]
pub struct MarketLedger {
    pools: HashMap<[u8; 32], PoolState>,
}

impl MarketLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a pool by id.
    pub fn pool(&self, id: &PoolId) -> Option<&PoolState> {
        self.pools.get(&id.0)
    }

    /// Read-only report for a pool.
    pub fn report(&self, id: &PoolId) -> Option<PoolReport> {
        let s = self.pools.get(&id.0)?;
        let prices = self.compute_prices(s);
        Some(PoolReport {
            pool: s.pool.clone(),
            prices,
        })
    }

    /// Compute current prices for all outcomes.
    fn compute_prices(&self, state: &PoolState) -> Vec<(OutcomeId, f64)> {
        let total = state.pool.total_reserve;
        if total == 0 {
            return state
                .pool
                .outcome_reserves
                .iter()
                .map(|(oid, _)| (*oid, 0.0))
                .collect();
        }
        state
            .pool
            .outcome_reserves
            .iter()
            .map(|(oid, reserve)| (*oid, *reserve as f64 / total as f64))
            .collect()
    }

    /// Dispatch a signed market op against the ledger. `actor` is the
    /// signer's DID (already verified by the caller). `seq` is the
    /// consensus-log sequence.
    ///
    /// Returns `Some(amount)` for the ops that move e-coins out of the pool
    /// — `RemoveLiquidity` (cash withdrawn), `SellOutcome` (net payout) and
    /// `Claim` (payout) — so the settlement service knows exactly how much to
    /// row to the recipient. All other ops return `None`.
    pub fn apply_op(
        &mut self,
        actor: &str,
        kind: &MarketOpKind,
        _seq: u64,
    ) -> Result<Option<u64>, Diagnostic> {
        match kind {
            MarketOpKind::OpenNegRisk {
                bond_id,
                outcomes,
                fee_bps,
            } => {
                self.apply_open_neg_risk(actor, bond_id, outcomes, *fee_bps)?;
                Ok(None)
            }
            MarketOpKind::AddLiquidity { pool_id, amount } => {
                self.apply_add_liquidity(actor, pool_id, *amount)?;
                Ok(None)
            }
            MarketOpKind::RemoveLiquidity { pool_id, shares } => {
                let cash = self.apply_remove_liquidity(actor, pool_id, *shares)?;
                Ok(Some(cash))
            }
            MarketOpKind::BuyOutcome {
                pool_id,
                outcome_id,
                amount,
            } => {
                self.apply_buy_outcome(actor, pool_id, outcome_id, *amount)?;
                Ok(None)
            }
            MarketOpKind::SellOutcome {
                pool_id,
                outcome_id,
                amount,
            } => {
                let payout = self.apply_sell_outcome(actor, pool_id, outcome_id, *amount)?;
                Ok(Some(payout))
            }
            MarketOpKind::Resolve {
                pool_id,
                trigger_seq,
            } => {
                self.apply_resolve(actor, pool_id, *trigger_seq)?;
                Ok(None)
            }
            MarketOpKind::Claim { pool_id } => self.apply_claim(actor, pool_id),
        }
    }

    /// Open a NegRisk pool with multiple mutually-exclusive outcomes.
    fn apply_open_neg_risk(
        &mut self,
        _actor: &str,
        bond_id: &MathBondId,
        outcomes: &[NegRiskOutcome],
        fee_bps: u64,
    ) -> Result<(), Diagnostic> {
        if outcomes.is_empty() {
            return Err(Diagnostic::new(
                Code::MARKET_NO_LIQUIDITY,
                "must have at least one outcome",
                Severity::Error,
            ));
        }
        if fee_bps > 10_000 {
            return Err(Diagnostic::new(
                Code::MARKET_PRICE_UNDERFLOW,
                format!("fee_bps {fee_bps} exceeds 10000 (100%)"),
                Severity::Error,
            ));
        }
        // The NegRisk price model is ill-defined with duplicate outcome ids or
        // without a terminal "never" outcome (maturity_seq == u64::MAX) to win
        // when the bond matures without a trigger.
        let mut seen = HashSet::new();
        for o in outcomes {
            if !seen.insert(o.outcome_id) {
                return Err(Diagnostic::new(
                    Code::MARKET_PRICE_UNDERFLOW,
                    "duplicate outcome id",
                    Severity::Error,
                ));
            }
        }
        if !outcomes.iter().any(|o| o.maturity_seq == u64::MAX) {
            return Err(Diagnostic::new(
                Code::MARKET_PRICE_UNDERFLOW,
                "a terminal 'never' outcome (maturity_seq == u64::MAX) is required",
                Severity::Error,
            ));
        }

        let pool_id = compute_pool_id(bond_id);
        if self.pools.contains_key(&pool_id.0) {
            return Err(Diagnostic::new(
                Code::MARKET_POOL_EXISTS,
                "pool already exists for this bond",
                Severity::Error,
            ));
        }

        let outcome_reserves: Vec<(OutcomeId, u64)> =
            outcomes.iter().map(|o| (o.outcome_id, 0)).collect();

        // Canonicalize each outcome's pool_id to the computed id so the stored
        // state cannot drift from the deterministic pool id.
        let canonical: Vec<NegRiskOutcome> = outcomes
            .iter()
            .cloned()
            .map(|mut o| {
                o.pool_id = pool_id;
                o
            })
            .collect();

        let pool = LiquidityPool {
            pool_id,
            bond_id: *bond_id,
            outcome_reserves,
            total_reserve: 0,
            lp_shares: Vec::new(),
            total_shares: 0,
            fee_bps,
            resolved: false,
            winner: None,
        };

        self.pools.insert(
            pool_id.0,
            PoolState {
                pool,
                outcomes: canonical,
                reserve_map: HashMap::new(),
                positions: HashMap::new(),
                lp_map: HashMap::new(),
                total_outcome_tokens: HashMap::new(),
                lp_fees: 0,
            },
        );
        Ok(())
    }

    /// LP adds liquidity. If the pool is empty, the first LP sets the initial
    /// price ratio by depositing proportionally across outcomes.
    fn apply_add_liquidity(
        &mut self,
        actor: &str,
        pool_id: &PoolId,
        amount: u64,
    ) -> Result<(), Diagnostic> {
        if amount == 0 {
            return Err(Diagnostic::new(
                Code::MARKET_NO_LIQUIDITY,
                "deposit amount must be positive",
                Severity::Error,
            ));
        }
        let state = match self.pools.get_mut(&pool_id.0) {
            Some(s) => s,
            None => {
                return Err(Diagnostic::new(
                    Code::MARKET_UNKNOWN_POOL,
                    "unknown pool",
                    Severity::Error,
                ));
            }
        };
        if state.pool.resolved {
            return Err(Diagnostic::new(
                Code::MARKET_POOL_RESOLVED,
                "pool is resolved",
                Severity::Error,
            ));
        }

        let n_outcomes = state.pool.outcome_reserves.len() as u64;
        if n_outcomes == 0 {
            return Err(Diagnostic::new(
                Code::MARKET_NO_LIQUIDITY,
                "pool has no outcomes",
                Severity::Error,
            ));
        }

        // Distribute deposit equally across all outcomes (NegRisk invariant:
        // initial prices are uniform 1/N).
        let per_outcome = amount / n_outcomes;

        if state.pool.total_shares == 0 {
            // First LP: shares = deposit amount (1:1 initially).
            state.pool.total_shares = amount;
            state.lp_map.insert(actor.to_string(), amount);
            state.pool.lp_shares.push((actor.to_string(), amount));
        } else {
            // Subsequent LPs: shares proportional to their deposit relative
            // to the total reserve.
            let shares = if state.pool.total_reserve > 0 {
                (amount as u128 * state.pool.total_shares as u128
                    / state.pool.total_reserve as u128) as u64
            } else {
                amount
            };
            state.pool.total_shares += shares;
            *state.lp_map.entry(actor.to_string()).or_insert(0) += shares;
            // Update the Vec for serialization consistency.
            if let Some(entry) = state.pool.lp_shares.iter_mut().find(|(d, _)| d == actor) {
                entry.1 += shares;
            } else {
                state.pool.lp_shares.push((actor.to_string(), shares));
            }
        }

        // Add to reserves (NegRisk invariant: uniform initial distribution).
        // Distribute equally; remainder goes to the last outcome.
        let mut distributed = 0u64;
        let n = state.pool.outcome_reserves.len();
        for (i, (oid, _)) in state.pool.outcome_reserves.clone().iter().enumerate() {
            let share = if i + 1 == n {
                amount - distributed
            } else {
                per_outcome
            };
            *state.reserve_map.entry(oid.0).or_insert(0) += share;
            distributed += share;
        }
        state.pool.total_reserve += amount;

        // Update the Vec for serialization.
        for (oid, reserve) in state.pool.outcome_reserves.iter_mut() {
            *reserve = *state.reserve_map.get(&oid.0).unwrap_or(&0);
        }

        Ok(())
    }

    /// LP removes liquidity by burning shares.
    ///
    /// Withdrawal is proportional to the shares: each outcome reserve is
    /// reduced by `shares/total` of itself, plus a pro-rata slice of the
    /// accrued LP fees. The total reserve is recomputed as the sum of the
    /// per-outcome reductions so the `total_reserve == sum(reserves)`
    /// invariant stays exact; any integer rounding dust stays in the pool.
    fn apply_remove_liquidity(
        &mut self,
        actor: &str,
        pool_id: &PoolId,
        shares: u64,
    ) -> Result<u64, Diagnostic> {
        if shares == 0 {
            return Err(Diagnostic::new(
                Code::MARKET_INSUFFICIENT_SHARES,
                "shares must be positive",
                Severity::Error,
            ));
        }
        let state = match self.pools.get_mut(&pool_id.0) {
            Some(s) => s,
            None => {
                return Err(Diagnostic::new(
                    Code::MARKET_UNKNOWN_POOL,
                    "unknown pool",
                    Severity::Error,
                ));
            }
        };
        if state.pool.resolved {
            return Err(Diagnostic::new(
                Code::MARKET_POOL_RESOLVED,
                "pool is resolved; claim via the Claim op instead",
                Severity::Error,
            ));
        }
        let held = state.lp_map.get(actor).copied().unwrap_or(0);
        if held < shares {
            return Err(Diagnostic::new(
                Code::MARKET_INSUFFICIENT_SHARES,
                format!("have {held}, need {shares}"),
                Severity::Error,
            ));
        }

        let total = state.pool.total_shares;
        if total == 0 {
            return Err(Diagnostic::new(
                Code::MARKET_NO_LIQUIDITY,
                "no shares outstanding",
                Severity::Error,
            ));
        }

        // Proportional withdrawal: shares/total of each outcome reserve. The
        // LP's cash is the sum of the per-outcome reductions (deterministic;
        // the ideal `shares * total_reserve / total` differs by < n dust which
        // stays in the pool, keeping the reserve invariant exact).
        let mut withdrawn = 0u64;
        for (oid, reserve) in state.pool.outcome_reserves.iter_mut() {
            let r = state.reserve_map.get(&oid.0).copied().unwrap_or(0);
            let r_withdraw = (shares as u128 * r as u128 / total as u128) as u64;
            let new_r = r - r_withdraw;
            state.reserve_map.insert(oid.0, new_r);
            *reserve = new_r;
            withdrawn += r_withdraw;
        }
        state.pool.total_reserve -= withdrawn;

        // Pay out the LP's pro-rata slice of the accrued trading fees too.
        let fee_share = (shares as u128 * state.lp_fees as u128 / total as u128) as u64;
        state.lp_fees -= fee_share;

        state.lp_map.insert(actor.to_string(), held - shares);
        state.pool.total_shares -= shares;

        // Update the Vec and drop zero-share entries so it stays in sync with
        // the lp_map.
        if let Some(entry) = state.pool.lp_shares.iter_mut().find(|(d, _)| d == actor) {
            entry.1 -= shares;
        }
        state.pool.lp_shares.retain(|(_, s)| *s > 0);

        // The cash the LP withdraws: the reserve reduction plus the fee slice.
        Ok(withdrawn + fee_share)
    }

    /// Trader buys outcome tokens at the current vAMM price.
    ///
    /// Pricing: `P(outcome_i) = reserve_i / total_reserve`. The trader
    /// deposits `amount` e-coins; the fee (`fee_bps`) accrues to the LPs and
    /// the net deposit buys tokens at the POST-trade marginal price:
    /// `tokens = net / P' = net * (total + net) / (reserve_i + net)`, where
    /// `P'` is the price after the deposit lands in the reserve. Buying at the
    /// post-trade price (the constant-product convention) makes a buy-then-
    /// sell round trip exactly neutral modulo fees — the sell-back redeems at
    /// the same `P'` — so the pool cannot be drained by price-jumping cycles.
    fn apply_buy_outcome(
        &mut self,
        actor: &str,
        pool_id: &PoolId,
        outcome_id: &OutcomeId,
        amount: u64,
    ) -> Result<(), Diagnostic> {
        if amount == 0 {
            return Err(Diagnostic::new(
                Code::MARKET_NO_LIQUIDITY,
                "amount must be positive",
                Severity::Error,
            ));
        }
        let state = match self.pools.get_mut(&pool_id.0) {
            Some(s) => s,
            None => {
                return Err(Diagnostic::new(
                    Code::MARKET_UNKNOWN_POOL,
                    "unknown pool",
                    Severity::Error,
                ));
            }
        };
        if state.pool.resolved {
            return Err(Diagnostic::new(
                Code::MARKET_POOL_RESOLVED,
                "pool is resolved",
                Severity::Error,
            ));
        }
        if state.pool.total_reserve == 0 {
            return Err(Diagnostic::new(
                Code::MARKET_NO_LIQUIDITY,
                "pool has no liquidity",
                Severity::Error,
            ));
        }
        if !state.reserve_map.contains_key(&outcome_id.0) {
            return Err(Diagnostic::new(
                Code::MARKET_UNKNOWN_OUTCOME,
                "outcome not in pool",
                Severity::Error,
            ));
        }

        // Fee: deduct fee_bps from the deposit; the fee is LP yield, not pool
        // collateral, so it accrues separately from the pricing reserves.
        let fee = amount * state.pool.fee_bps / 10000;
        let net = amount - fee;

        let old_r = state.reserve_map.get(&outcome_id.0).copied().unwrap_or(0);
        if old_r == 0 {
            return Err(Diagnostic::new(
                Code::MARKET_NO_LIQUIDITY,
                "outcome has no reserve; its price is zero — nothing to buy at",
                Severity::Error,
            ));
        }
        let old_total = state.pool.total_reserve;
        let new_r = old_r + net;
        let new_total = old_total + net;

        // Tokens granted at the post-trade marginal price P' = new_r / new_total:
        // tokens = net / P' = net * new_total / new_r. Guard the cast against
        // overflow in extreme price skews (new_total >= new_r so tokens >= net).
        let tokens =
            u64::try_from(net as u128 * new_total as u128 / new_r as u128).map_err(|_| {
                Diagnostic::new(
                    Code::MARKET_PRICE_UNDERFLOW,
                    "token grant overflow",
                    Severity::Error,
                )
            })?;

        state.reserve_map.insert(outcome_id.0, new_r);
        state.pool.total_reserve = new_total;
        state.lp_fees = state.lp_fees.checked_add(fee).ok_or_else(|| {
            Diagnostic::new(
                Code::MARKET_PRICE_UNDERFLOW,
                "fee accumulator overflow",
                Severity::Error,
            )
        })?;
        *state.total_outcome_tokens.entry(outcome_id.0).or_insert(0) += tokens;

        // Credit the trader's outcome position.
        let trader_pos = state
            .positions
            .entry(outcome_id.0)
            .or_insert_with(HashMap::new);
        *trader_pos.entry(actor.to_string()).or_insert(0) += tokens;

        // Update the Vec.
        for (oid, reserve) in state.pool.outcome_reserves.iter_mut() {
            *reserve = *state.reserve_map.get(&oid.0).unwrap_or(&0);
        }

        Ok(())
    }

    /// Trader sells outcome tokens back to the pool.
    ///
    /// Each token redeems at the current price: `payout = amount * P`. The
    /// payout is capped by the outcome's own reserve so the pool can never pay
    /// out more e-coins than it holds for that outcome (in the ratio model the
    /// token supply is not bounded by the reserve, so the cap is what keeps
    /// the pool solvent). The fee accrues to the LPs; the burned tokens are
    /// removed from the outstanding-token count used by resolution payouts.
    fn apply_sell_outcome(
        &mut self,
        actor: &str,
        pool_id: &PoolId,
        outcome_id: &OutcomeId,
        amount: u64,
    ) -> Result<u64, Diagnostic> {
        if amount == 0 {
            return Err(Diagnostic::new(
                Code::MARKET_INSUFFICIENT_TOKENS,
                "amount must be positive",
                Severity::Error,
            ));
        }
        let state = match self.pools.get_mut(&pool_id.0) {
            Some(s) => s,
            None => {
                return Err(Diagnostic::new(
                    Code::MARKET_UNKNOWN_POOL,
                    "unknown pool",
                    Severity::Error,
                ));
            }
        };
        if state.pool.resolved {
            return Err(Diagnostic::new(
                Code::MARKET_POOL_RESOLVED,
                "pool is resolved; claim via the Claim op instead",
                Severity::Error,
            ));
        }

        let held = state
            .positions
            .get(&outcome_id.0)
            .and_then(|m| m.get(actor))
            .copied()
            .unwrap_or(0);
        if held < amount {
            return Err(Diagnostic::new(
                Code::MARKET_INSUFFICIENT_TOKENS,
                format!("have {held}, need {amount}"),
                Severity::Error,
            ));
        }

        let old_r = state.reserve_map.get(&outcome_id.0).copied().unwrap_or(0);
        let payout = if state.pool.total_reserve > 0 {
            (amount as u128 * old_r as u128 / state.pool.total_reserve as u128) as u64
        } else {
            0
        };
        // Solvency cap: never pay more than the outcome's reserve.
        let payout = payout.min(old_r);
        let fee = payout * state.pool.fee_bps / 10000;
        let net_payout = payout - fee;

        // Burn the tokens: remove the sold tokens from the trader's position
        // (dropping the entry at zero) and from the outstanding count (used by
        // resolution payouts).
        if held == amount {
            if let Some(m) = state.positions.get_mut(&outcome_id.0) {
                m.remove(actor);
            }
        } else {
            state
                .positions
                .get_mut(&outcome_id.0)
                .unwrap()
                .insert(actor.to_string(), held - amount);
        }
        let outstanding = state
            .total_outcome_tokens
            .get(&outcome_id.0)
            .copied()
            .unwrap_or(0);
        state
            .total_outcome_tokens
            .insert(outcome_id.0, outstanding.saturating_sub(amount));

        let new_r = old_r - payout;
        state.reserve_map.insert(outcome_id.0, new_r);
        state.pool.total_reserve -= payout;
        state.lp_fees = state.lp_fees.checked_add(fee).ok_or_else(|| {
            Diagnostic::new(
                Code::MARKET_PRICE_UNDERFLOW,
                "fee accumulator overflow",
                Severity::Error,
            )
        })?;

        // Update the Vec.
        for (oid, reserve) in state.pool.outcome_reserves.iter_mut() {
            *reserve = *state.reserve_map.get(&oid.0).unwrap_or(&0);
        }

        // The net payout goes to the trader as e-coins (certificate transfer
        // in the settlement service).
        Ok(net_payout)
    }

    /// Resolve the pool: the bond's trigger fired (or the bond matured without
    /// one). The winner is NOT a caller choice — it is the outcome whose
    /// maturity window contains the trigger signal, computed deterministically
    /// from `trigger_signal` (`Some(consensus seq)` when the trigger fired,
    /// `None` when the bond matured without one → the terminal "never"
    /// outcome). The consensus node validates the signal against the bond
    /// ledger before this op is applied.
    fn apply_resolve(
        &mut self,
        _actor: &str,
        pool_id: &PoolId,
        trigger_signal: Option<u64>,
    ) -> Result<(), Diagnostic> {
        // Immutable pass: existence, resolved, and the deterministic winner.
        let winner = {
            let state = match self.pools.get(&pool_id.0) {
                Some(s) => s,
                None => {
                    return Err(Diagnostic::new(
                        Code::MARKET_UNKNOWN_POOL,
                        "unknown pool",
                        Severity::Error,
                    ));
                }
            };
            if state.pool.resolved {
                return Err(Diagnostic::new(
                    Code::MARKET_POOL_RESOLVED,
                    "pool already resolved",
                    Severity::Error,
                ));
            }
            self.winner_for(state, trigger_signal)?
        };

        let state = self.pools.get_mut(&pool_id.0).unwrap();
        let total = state.pool.total_reserve;

        // NegRisk resolution: the winning outcome gets the entire pool reserve
        // (price 1.0); all other outcomes go to zero.
        for (oid, reserve) in state.pool.outcome_reserves.iter_mut() {
            if *oid == winner {
                *reserve = total;
            } else {
                *reserve = 0;
            }
            state.reserve_map.insert(oid.0, *reserve);
        }

        state.pool.resolved = true;
        state.pool.winner = Some(winner);

        Ok(())
    }

    /// The deterministic winning outcome for a trigger signal: the outcome
    /// with the smallest `maturity_seq >= t` (t = `signal` or `u64::MAX` when
    /// `None`). Windows are `(prev_maturity, maturity]`, so an outcome with
    /// `maturity_seq == u64::MAX` is the terminal "never" outcome that wins
    /// whenever no window covers the trigger. Ties break to the earliest
    /// stored order.
    fn winner_for(&self, state: &PoolState, signal: Option<u64>) -> Result<OutcomeId, Diagnostic> {
        let t = signal.unwrap_or(u64::MAX);
        let mut best: Option<(u64, OutcomeId)> = None;
        for o in &state.outcomes {
            if o.maturity_seq >= t {
                match best {
                    None => best = Some((o.maturity_seq, o.outcome_id)),
                    Some((bm, _)) if o.maturity_seq < bm => {
                        best = Some((o.maturity_seq, o.outcome_id));
                    }
                    _ => {}
                }
            }
        }
        best.map(|(_, oid)| oid).ok_or_else(|| {
            Diagnostic::new(
                Code::MARKET_UNKNOWN_OUTCOME,
                "no outcome covers the trigger signal",
                Severity::Error,
            )
        })
    }

    /// Post-resolution withdrawal for one actor: redeem winning outcome tokens
    /// pro-rata against the pool reserve, plus the LP's share of accrued fees
    /// (and of the whole reserve when nobody held winning tokens). The returned
    /// amount is the deterministic e-coin payout the settlement service rows.
    /// Idempotent by construction: claimed positions/shares are zeroed, so a
    /// second claim pays 0.
    fn apply_claim(&mut self, actor: &str, pool_id: &PoolId) -> Result<Option<u64>, Diagnostic> {
        let state = match self.pools.get_mut(&pool_id.0) {
            Some(s) => s,
            None => {
                return Err(Diagnostic::new(
                    Code::MARKET_UNKNOWN_POOL,
                    "unknown pool",
                    Severity::Error,
                ));
            }
        };
        let winner = state.pool.winner.ok_or_else(|| {
            Diagnostic::new(
                Code::MARKET_NOT_RESOLVED,
                "pool is not resolved; nothing to claim",
                Severity::Error,
            )
        })?;

        let total = state.pool.total_reserve;
        let win_tokens = state
            .total_outcome_tokens
            .get(&winner.0)
            .copied()
            .unwrap_or(0);
        let total_shares = state.pool.total_shares;
        let mut payout = 0u64;

        // Winning outcome tokens: each token is worth total / win_tokens.
        let held_tokens = state
            .positions
            .get(&winner.0)
            .and_then(|m| m.get(actor))
            .copied()
            .unwrap_or(0);
        if held_tokens > 0 && win_tokens > 0 {
            let token_payout = (held_tokens as u128 * total as u128 / win_tokens as u128) as u64;
            payout = payout.checked_add(token_payout).ok_or_else(|| {
                Diagnostic::new(
                    Code::MARKET_PRICE_UNDERFLOW,
                    "claim payout overflow",
                    Severity::Error,
                )
            })?;
            if let Some(m) = state.positions.get_mut(&winner.0) {
                m.remove(actor);
            }
        }

        // LP share: accrued fees always; plus the whole reserve when no one
        // held winning tokens (the pool's cash goes back to its underwriters).
        let shares = state.lp_map.get(actor).copied().unwrap_or(0);
        if shares > 0 && total_shares > 0 {
            let cash_share = if win_tokens == 0 {
                (shares as u128 * total as u128 / total_shares as u128) as u64
            } else {
                0
            };
            let fee_share = (shares as u128 * state.lp_fees as u128 / total_shares as u128) as u64;
            payout = payout
                .checked_add(cash_share)
                .and_then(|p| p.checked_add(fee_share))
                .ok_or_else(|| {
                    Diagnostic::new(
                        Code::MARKET_PRICE_UNDERFLOW,
                        "claim payout overflow",
                        Severity::Error,
                    )
                })?;

            state.lp_map.insert(actor.to_string(), 0);
            if let Some(entry) = state.pool.lp_shares.iter_mut().find(|(d, _)| d == actor) {
                entry.1 = 0;
            }
            state.pool.lp_shares.retain(|(_, s)| *s > 0);
            state.lp_fees -= fee_share;
        }

        Ok(Some(payout))
    }
}

/// Deterministic pool id from the bond id. Public so clients can reference a
/// pool (AddLiquidity/BuyOutcome/...) without round-tripping through a report.
pub fn compute_pool_id(bond_id: &MathBondId) -> PoolId {
    let mut ctx = Sha256::new();
    ctx.update(b"unfer:market:v1");
    ctx.update(bond_id.0);
    PoolId(ctx.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bond_id() -> MathBondId {
        MathBondId([42u8; 32])
    }

    fn outcome_a() -> OutcomeId {
        OutcomeId([1u8; 32])
    }

    fn outcome_b() -> OutcomeId {
        OutcomeId([2u8; 32])
    }

    fn outcome_never() -> OutcomeId {
        OutcomeId([3u8; 32])
    }

    fn pool_id() -> PoolId {
        compute_pool_id(&bond_id())
    }

    fn open_pool(l: &mut MarketLedger) {
        l.apply_op(
            "did:unfer:creator",
            &MarketOpKind::OpenNegRisk {
                bond_id: bond_id(),
                outcomes: vec![
                    NegRiskOutcome {
                        outcome_id: outcome_a(),
                        pool_id: pool_id(),
                        label: "triggered_by_2025".to_string(),
                        maturity_seq: 500,
                    },
                    NegRiskOutcome {
                        outcome_id: outcome_b(),
                        pool_id: pool_id(),
                        label: "triggered_by_2026".to_string(),
                        maturity_seq: 1000,
                    },
                    NegRiskOutcome {
                        outcome_id: outcome_never(),
                        pool_id: pool_id(),
                        label: "never".to_string(),
                        maturity_seq: u64::MAX,
                    },
                ],
                fee_bps: 300,
            },
            1,
        )
        .unwrap();
    }

    #[test]
    fn open_and_add_liquidity() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);

        // Add 10000 e-coins of liquidity.
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 10000,
            },
            2,
        )
        .unwrap();

        let report = l.report(&pool_id()).unwrap();
        assert_eq!(report.pool.total_reserve, 10000);
        assert_eq!(report.pool.total_shares, 10000);
        // Initial prices are uniform (1/3 for 3 outcomes).
        let price_a = report
            .prices
            .iter()
            .find(|(oid, _)| *oid == outcome_a())
            .unwrap()
            .1;
        let price_b = report
            .prices
            .iter()
            .find(|(oid, _)| *oid == outcome_b())
            .unwrap()
            .1;
        assert!((price_a - 1.0 / 3.0).abs() < 0.01, "price_a = {price_a}");
        assert!((price_b - 1.0 / 3.0).abs() < 0.01, "price_b = {price_b}");
        // Prices sum to 1 across all outcomes.
        let sum: f64 = report.prices.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 0.01, "prices sum to 1: {sum}");
    }

    #[test]
    fn buy_moves_price() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 10000,
            },
            2,
        )
        .unwrap();

        // Buy 2000 of outcome A.
        l.apply_op(
            "did:unfer:trader",
            &MarketOpKind::BuyOutcome {
                pool_id: pool_id(),
                outcome_id: outcome_a(),
                amount: 2000,
            },
            3,
        )
        .unwrap();

        let report = l.report(&pool_id()).unwrap();
        // Price of A should have increased.
        let price_a = report
            .prices
            .iter()
            .find(|(oid, _)| *oid == outcome_a())
            .unwrap()
            .1;
        let price_b = report
            .prices
            .iter()
            .find(|(oid, _)| *oid == outcome_b())
            .unwrap()
            .1;
        assert!(
            price_a > 1.0 / 3.0,
            "buying A should increase its price: {price_a}"
        );
        assert!(
            price_b < 1.0 / 3.0,
            "buying A should decrease B's price: {price_b}"
        );
        // Prices should still sum to ~1.
        let sum: f64 = report.prices.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 0.01, "prices should sum to 1: {sum}");
    }

    #[test]
    fn buy_grants_price_adjusted_tokens_and_roundtrip_conserves() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 12000,
            },
            2,
        )
        .unwrap();

        // Buy 3000 of A (3% fee → net 2910). Reserves start at 4000 each.
        l.apply_op(
            "did:unfer:trader",
            &MarketOpKind::BuyOutcome {
                pool_id: pool_id(),
                outcome_id: outcome_a(),
                amount: 3000,
            },
            3,
        )
        .unwrap();

        let state = l.pool(&pool_id()).unwrap();
        // net = 3000 - 90 = 2910; new reserve A = 4000 + 2910 = 6910;
        // total = 12000 + 2910 = 14910.
        assert_eq!(state.pool.total_reserve, 14910);
        assert_eq!(
            state.reserve_map.get(&outcome_a().0).copied().unwrap(),
            6910
        );
        // Tokens = net * new_total / new_reserve = 2910 * 14910 / 6910 = 6279.
        let tokens = state
            .total_outcome_tokens
            .get(&outcome_a().0)
            .copied()
            .unwrap();
        let expected = 2910u128 * 14910u128 / 6910u128;
        assert_eq!(
            tokens as u128, expected,
            "tokens minted at the post-trade price"
        );
        // The trader holds exactly those tokens.
        assert_eq!(
            state
                .positions
                .get(&outcome_a().0)
                .and_then(|m| m.get("did:unfer:trader"))
                .copied()
                .unwrap_or(0),
            tokens
        );
        // The LP fee accrued separately (300 bps of the deposit).
        assert_eq!(state.lp_fees, 90);

        // Round trip: selling all tokens back returns ~net (2910) minus fees —
        // never MORE than the deposit, so the pool cannot be drained.
        l.apply_op(
            "did:unfer:trader",
            &MarketOpKind::SellOutcome {
                pool_id: pool_id(),
                outcome_id: outcome_a(),
                amount: tokens,
            },
            4,
        )
        .unwrap();
        let after = l.pool(&pool_id()).unwrap();
        // The trader's position was burned.
        assert_eq!(
            after
                .positions
                .get(&outcome_a().0)
                .and_then(|m| m.get("did:unfer:trader"))
                .copied()
                .unwrap_or(0),
            0
        );
        // Pool cash grew by the two fees (buy 90 + sell fee) — never shrank.
        assert!(after.pool.total_reserve + after.lp_fees >= 12000 + 90);
        // Invariant: total_reserve == sum of reserves.
        let sum: u64 = after.reserve_map.values().sum();
        assert_eq!(after.pool.total_reserve, sum);
    }

    #[test]
    fn resolve_makes_loser_worthless() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 10000,
            },
            2,
        )
        .unwrap();

        // The trigger fired at consensus seq 300, inside A's window (<= 500).
        // The winner is derived from the signal, not chosen by the resolver.
        l.apply_op(
            "did:unfer:resolver",
            &MarketOpKind::Resolve {
                pool_id: pool_id(),
                trigger_seq: Some(300),
            },
            3,
        )
        .unwrap();

        let report = l.report(&pool_id()).unwrap();
        assert!(report.pool.resolved);
        assert_eq!(report.pool.winner, Some(outcome_a()));
        // A gets 100%, B gets 0%.
        let price_a = report
            .prices
            .iter()
            .find(|(oid, _)| *oid == outcome_a())
            .unwrap()
            .1;
        let price_b = report
            .prices
            .iter()
            .find(|(oid, _)| *oid == outcome_b())
            .unwrap()
            .1;
        assert!((price_a - 1.0).abs() < 0.01, "winner = 100%: {price_a}");
        assert!((price_b).abs() < 0.01, "loser = 0%: {price_b}");
    }

    #[test]
    fn never_outcome_wins_on_maturity() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 10000,
            },
            2,
        )
        .unwrap();

        // The bond matured without a trigger: signal None → the terminal
        // "never" outcome wins.
        l.apply_op(
            "did:unfer:resolver",
            &MarketOpKind::Resolve {
                pool_id: pool_id(),
                trigger_seq: None,
            },
            3,
        )
        .unwrap();

        let report = l.report(&pool_id()).unwrap();
        assert_eq!(report.pool.winner, Some(outcome_never()));
    }

    #[test]
    fn trigger_signal_selects_the_window() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 10000,
            },
            2,
        )
        .unwrap();

        // Trigger at seq 700 → inside B's window (500 < 700 <= 1000).
        l.apply_op(
            "did:unfer:resolver",
            &MarketOpKind::Resolve {
                pool_id: pool_id(),
                trigger_seq: Some(700),
            },
            3,
        )
        .unwrap();
        let report = l.report(&pool_id()).unwrap();
        assert_eq!(report.pool.winner, Some(outcome_b()));

        // Trigger after every window (1500) → the never outcome.
        let mut l2 = MarketLedger::new();
        open_pool(&mut l2);
        l2.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 10000,
            },
            2,
        )
        .unwrap();
        l2.apply_op(
            "did:unfer:resolver",
            &MarketOpKind::Resolve {
                pool_id: pool_id(),
                trigger_seq: Some(1500),
            },
            3,
        )
        .unwrap();
        assert_eq!(
            l2.pool(&pool_id()).unwrap().pool.winner,
            Some(outcome_never())
        );
    }

    #[test]
    fn double_resolve_rejected() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 10000,
            },
            2,
        )
        .unwrap();
        l.apply_op(
            "did:unfer:resolver",
            &MarketOpKind::Resolve {
                pool_id: pool_id(),
                trigger_seq: Some(300),
            },
            3,
        )
        .unwrap();

        let err = l.apply_op(
            "did:unfer:resolver",
            &MarketOpKind::Resolve {
                pool_id: pool_id(),
                trigger_seq: Some(700),
            },
            4,
        );
        assert_eq!(err.unwrap_err().code, Code::MARKET_POOL_RESOLVED);
    }

    #[test]
    fn open_pool_requires_terminal_never_outcome() {
        let mut l = MarketLedger::new();
        // No u64::MAX maturity → the pool cannot represent "bond matured
        // without a trigger" and is refused.
        let err = l.apply_op(
            "did:unfer:creator",
            &MarketOpKind::OpenNegRisk {
                bond_id: bond_id(),
                outcomes: vec![NegRiskOutcome {
                    outcome_id: outcome_a(),
                    pool_id: pool_id(),
                    label: "A".to_string(),
                    maturity_seq: 500,
                }],
                fee_bps: 300,
            },
            1,
        );
        assert_eq!(err.unwrap_err().code, Code::MARKET_PRICE_UNDERFLOW);

        // Duplicate outcome ids are refused.
        let err = l.apply_op(
            "did:unfer:creator",
            &MarketOpKind::OpenNegRisk {
                bond_id: bond_id(),
                outcomes: vec![
                    NegRiskOutcome {
                        outcome_id: outcome_a(),
                        pool_id: pool_id(),
                        label: "A1".to_string(),
                        maturity_seq: 500,
                    },
                    NegRiskOutcome {
                        outcome_id: outcome_a(),
                        pool_id: pool_id(),
                        label: "A2".to_string(),
                        maturity_seq: u64::MAX,
                    },
                ],
                fee_bps: 300,
            },
            1,
        );
        assert_eq!(err.unwrap_err().code, Code::MARKET_PRICE_UNDERFLOW);
    }

    #[test]
    fn claim_before_resolve_rejected() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 10000,
            },
            2,
        )
        .unwrap();
        let err = l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::Claim { pool_id: pool_id() },
            3,
        );
        assert_eq!(err.unwrap_err().code, Code::MARKET_NOT_RESOLVED);
    }

    #[test]
    fn claim_pays_winning_tokens_and_lp_fees() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 12000,
            },
            2,
        )
        .unwrap();
        // Trader buys 3000 of A (fee 90 → net 2910 → 6279 tokens).
        l.apply_op(
            "did:unfer:trader",
            &MarketOpKind::BuyOutcome {
                pool_id: pool_id(),
                outcome_id: outcome_a(),
                amount: 3000,
            },
            3,
        )
        .unwrap();
        // A wins (trigger at 300).
        l.apply_op(
            "did:unfer:resolver",
            &MarketOpKind::Resolve {
                pool_id: pool_id(),
                trigger_seq: Some(300),
            },
            4,
        )
        .unwrap();

        let state = l.pool(&pool_id()).unwrap();
        let total = state.pool.total_reserve; // 14910
        // Trader claim: they hold ALL of the winning tokens, so the claim is
        // the whole reserve.
        assert_eq!(
            state
                .total_outcome_tokens
                .get(&outcome_a().0)
                .copied()
                .unwrap(),
            6279
        );
        let trader_claim = l
            .apply_op(
                "did:unfer:trader",
                &MarketOpKind::Claim { pool_id: pool_id() },
                5,
            )
            .unwrap()
            .unwrap();
        assert_eq!(trader_claim, total);
        // LP claim: the accrued fee (90).
        let lp_claim = l
            .apply_op(
                "did:unfer:lp",
                &MarketOpKind::Claim { pool_id: pool_id() },
                6,
            )
            .unwrap()
            .unwrap();
        assert_eq!(lp_claim, 90);
        // Second claims pay nothing (idempotent).
        let again = l
            .apply_op(
                "did:unfer:trader",
                &MarketOpKind::Claim { pool_id: pool_id() },
                7,
            )
            .unwrap()
            .unwrap();
        assert_eq!(again, 0);
    }

    #[test]
    fn lp_gets_reserve_when_no_one_held_winning_tokens() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 12000,
            },
            2,
        )
        .unwrap();
        // No one bought any outcome. The never outcome wins (matured without
        // a trigger) and the LP keeps the whole reserve.
        l.apply_op(
            "did:unfer:resolver",
            &MarketOpKind::Resolve {
                pool_id: pool_id(),
                trigger_seq: None,
            },
            3,
        )
        .unwrap();
        let claim = l
            .apply_op(
                "did:unfer:lp",
                &MarketOpKind::Claim { pool_id: pool_id() },
                4,
            )
            .unwrap()
            .unwrap();
        assert_eq!(claim, 12000);
    }

    #[test]
    fn remove_liquidity_pays_fees_and_keeps_invariant() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 12000,
            },
            2,
        )
        .unwrap();
        // A trade accrues an LP fee (90).
        l.apply_op(
            "did:unfer:trader",
            &MarketOpKind::BuyOutcome {
                pool_id: pool_id(),
                outcome_id: outcome_a(),
                amount: 3000,
            },
            3,
        )
        .unwrap();

        // Remove half the LP position (6000 shares).
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::RemoveLiquidity {
                pool_id: pool_id(),
                shares: 6000,
            },
            4,
        )
        .unwrap();

        let state = l.pool(&pool_id()).unwrap();
        assert_eq!(state.pool.total_shares, 6000);
        // Invariant: total_reserve == sum of reserves.
        let sum: u64 = state.reserve_map.values().sum();
        assert_eq!(state.pool.total_reserve, sum);
        // The fee pool shrank pro-rata (half of 90) — 45 stays for the rest.
        assert_eq!(state.lp_fees, 45);
        // The Vec is in sync with the lp_map.
        assert_eq!(state.pool.lp_shares.len(), 1);
        assert_eq!(state.pool.lp_shares[0].1, 6000);
    }

    #[test]
    fn sell_cannot_drain_the_pool() {
        let mut l = MarketLedger::new();
        open_pool(&mut l);
        l.apply_op(
            "did:unfer:lp",
            &MarketOpKind::AddLiquidity {
                pool_id: pool_id(),
                amount: 12000,
            },
            2,
        )
        .unwrap();
        // Mint a huge number of A tokens via a big buy, then try to dump them.
        l.apply_op(
            "did:unfer:trader",
            &MarketOpKind::BuyOutcome {
                pool_id: pool_id(),
                outcome_id: outcome_a(),
                amount: 1_000_000,
            },
            3,
        )
        .unwrap();
        let state = l.pool(&pool_id()).unwrap();
        let tokens = state
            .total_outcome_tokens
            .get(&outcome_a().0)
            .copied()
            .unwrap();
        let total_before = state.pool.total_reserve;

        // Selling everything back pays at most the reserve — the pool's cash
        // can never go below zero.
        l.apply_op(
            "did:unfer:trader",
            &MarketOpKind::SellOutcome {
                pool_id: pool_id(),
                outcome_id: outcome_a(),
                amount: tokens,
            },
            4,
        )
        .unwrap();
        let after = l.pool(&pool_id()).unwrap();
        // The pool never pays more than the whole reserve for one outcome.
        assert!(after.pool.total_reserve + after.lp_fees <= total_before + 1_000_000);
        assert!(after.pool.total_reserve + after.lp_fees >= 12_000);
    }

    #[test]
    fn unknown_pool_rejected() {
        let mut l = MarketLedger::new();
        let err = l.apply_op(
            "did:unfer:x",
            &MarketOpKind::AddLiquidity {
                pool_id: PoolId([99u8; 32]),
                amount: 100,
            },
            1,
        );
        assert_eq!(err.unwrap_err().code, Code::MARKET_UNKNOWN_POOL);
    }

    #[test]
    fn replay_is_deterministic() {
        let ops: Vec<(&str, MarketOpKind, u64)> = vec![
            (
                "did:unfer:creator",
                MarketOpKind::OpenNegRisk {
                    bond_id: bond_id(),
                    outcomes: vec![
                        NegRiskOutcome {
                            outcome_id: outcome_a(),
                            pool_id: pool_id(),
                            label: "A".to_string(),
                            maturity_seq: 500,
                        },
                        NegRiskOutcome {
                            outcome_id: outcome_b(),
                            pool_id: pool_id(),
                            label: "B".to_string(),
                            maturity_seq: 1000,
                        },
                        NegRiskOutcome {
                            outcome_id: outcome_never(),
                            pool_id: pool_id(),
                            label: "never".to_string(),
                            maturity_seq: u64::MAX,
                        },
                    ],
                    fee_bps: 300,
                },
                1,
            ),
            (
                "did:unfer:lp",
                MarketOpKind::AddLiquidity {
                    pool_id: pool_id(),
                    amount: 10000,
                },
                2,
            ),
            (
                "did:unfer:trader",
                MarketOpKind::BuyOutcome {
                    pool_id: pool_id(),
                    outcome_id: outcome_a(),
                    amount: 2000,
                },
                3,
            ),
            (
                "did:unfer:trader",
                MarketOpKind::Resolve {
                    pool_id: pool_id(),
                    trigger_seq: Some(300),
                },
                4,
            ),
            (
                "did:unfer:trader",
                MarketOpKind::Claim { pool_id: pool_id() },
                5,
            ),
        ];

        let mut a = MarketLedger::new();
        let mut b = MarketLedger::new();
        for (actor, kind, seq) in &ops {
            let r_a = a.apply_op(actor, kind, *seq);
            let r_b = b.apply_op(actor, kind, *seq);
            assert_eq!(r_a.is_ok(), r_b.is_ok(), "ops must agree at seq {seq}");
        }
        // Both ledgers converge on the same pool state.
        let report_a = a.report(&pool_id()).unwrap();
        let report_b = b.report(&pool_id()).unwrap();
        assert_eq!(report_a.pool.total_reserve, report_b.pool.total_reserve);
        assert_eq!(report_a.pool.total_shares, report_b.pool.total_shares);
        assert_eq!(report_a.prices, report_b.prices);
        // Claim payouts are identical across replays too.
        assert_eq!(
            a.pool(&pool_id()).unwrap().positions,
            b.pool(&pool_id()).unwrap().positions
        );
    }
}
