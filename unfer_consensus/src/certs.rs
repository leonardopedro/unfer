//! Carbon-certificate / UTXO ledger (ReFi exchange, Plan R Phase 1–2).
//!
//! This is the **state-transition engine** that every QuePaxa node runs when it
//! applies a `CertificateOp` from the consensus log. It is the "rapid
//! validation rule" of the exchange plan: a node checks an op (mint authority,
//! input existence, conservation, double-spend, owner) and only then lets it
//! into the log. Because the state is fully deterministic, every node that
//! replays the same log converges to the identical sparse-Merkle root — the
//! same determinism guarantee the rest of `unfer_consensus` relies on.
//!
//! The transparent core keeps amount/owner/blinding explicit. The RISC-Zero zkVM
//! layer is a **documented future additive layer** (see `docs/PLAN_REFI_EXCHANGE.md`,
//! "RISC Zero zkVM guest circuit", originally scoped as Plan R Phase 1): it would
//! replace the explicit fields with commitments and prove conservation inside the
//! zkVM. It is deliberately not wired into the shipped ledger; the deterministic
//! transparent core below is the confidence surface that layer would plug into.

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};
use unfer_protocol::{
    CertId, CertificateOpKind, Code, CoinRef, ConsensusTransaction, Diagnostic, Nullifier, Severity,
};

/// Sparse-Merkle depth = the 256-bit key space (`[u8; 32]`), so every key bit
/// selects its own tree level and the full commitment is irreducible. Derived
/// from the hash width instead of a magic constant; changing it would change
/// every ledger root, so it must stay pinned to `8 * 32`.
const SMT_DEPTH: usize = 8 * std::mem::size_of::<[u8; 32]>();
const ZERO: [u8; 32] = [0u8; 32];

fn h2(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
    let mut ctx = Sha256::new();
    ctx.update(b"unfer:smt");
    ctx.update(a);
    ctx.update(b);
    ctx.finalize().into()
}

/// A binary sparse Merkle tree over 32-byte keys at a fixed depth. Only the
/// non-empty nodes are stored; the empty subtree at each depth has a fixed
/// default hash. This commits the full (sparse) UTXO set to a single root.
#[derive(Debug)]
pub struct SparseMerkle {
    depth: usize,
    nodes: HashMap<Vec<bool>, [u8; 32]>,
    /// `defaults[d]` = hash of an empty subtree rooted at a node of path length `d`.
    defaults: Vec<[u8; 32]>,
}

fn key_bits(key: &[u8; 32], depth: usize) -> Vec<bool> {
    let mut out = Vec::with_capacity(depth);
    for i in 0..depth {
        let byte = key[i / 8];
        let bit = (byte >> (7 - (i % 8))) & 1;
        out.push(bit == 1);
    }
    out
}

impl SparseMerkle {
    pub fn new(depth: usize) -> Self {
        let mut defaults = vec![ZERO; depth + 1];
        defaults[depth] = ZERO;
        for d in (0..depth).rev() {
            defaults[d] = h2(defaults[d + 1], defaults[d + 1]);
        }
        Self {
            depth,
            nodes: HashMap::new(),
            defaults,
        }
    }

    pub fn empty() -> Self {
        Self::new(SMT_DEPTH)
    }

    /// Insert (or overwrite) the leaf `value` under `key`, then recompute the
    /// affected ancestors up to the root.
    pub fn insert(&mut self, key: &[u8; 32], value: [u8; 32]) {
        let full = key_bits(key, self.depth);
        self.nodes.insert(full.clone(), value);
        for d in (0..self.depth).rev() {
            let mut left = full[..d].to_vec();
            left.push(false);
            let mut right = full[..d].to_vec();
            right.push(true);
            let lh = self
                .nodes
                .get(&left)
                .copied()
                .unwrap_or(self.defaults[d + 1]);
            let rh = self
                .nodes
                .get(&right)
                .copied()
                .unwrap_or(self.defaults[d + 1]);
            self.nodes.insert(full[..d].to_vec(), h2(lh, rh));
        }
    }

    /// Remove a leaf (spend). The ancestor path is recomputed against the
    /// empty-subtree defaults, so the spent UTXO stops contributing to the root.
    pub fn remove(&mut self, key: &[u8; 32]) {
        let full = key_bits(key, self.depth);
        self.nodes.remove(&full);
        for d in (0..self.depth).rev() {
            let mut left = full[..d].to_vec();
            left.push(false);
            let mut right = full[..d].to_vec();
            right.push(true);
            let lh = self
                .nodes
                .get(&left)
                .copied()
                .unwrap_or(self.defaults[d + 1]);
            let rh = self
                .nodes
                .get(&right)
                .copied()
                .unwrap_or(self.defaults[d + 1]);
            self.nodes.insert(full[..d].to_vec(), h2(lh, rh));
        }
    }

    pub fn contains(&self, key: &[u8; 32]) -> bool {
        self.nodes.contains_key(&key_bits(key, self.depth))
    }

    /// The current commitment: the root hash of the sparse tree.
    pub fn root(&self) -> [u8; 32] {
        self.nodes
            .get(&Vec::new())
            .copied()
            .unwrap_or(self.defaults[0])
    }
}

/// A certificate currently held (unspent) in the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coin {
    pub coin_id: CertId,
    pub amount: u64,
    pub owner: String,
    pub blinding: [u8; 32],
    pub minted_seq: u64,
}

/// Deterministic commitment `Hash(amount, owner, blinding)` for a new coin.
pub fn commit_coin(amount: u64, owner: &str, blinding: &[u8; 32]) -> CertId {
    let mut ctx = Sha256::new();
    ctx.update(b"unfer:coin");
    ctx.update(amount.to_le_bytes());
    ctx.update(owner.as_bytes());
    ctx.update(blinding);
    CertId(ctx.finalize().into())
}

/// Deterministic nullifier for a coin_id (transparent core). A confidential
/// layer replaces this with `Hash(spend_key, coin_commitment)`.
pub fn nullifier_for(coin_id: &CertId) -> Nullifier {
    let mut ctx = Sha256::new();
    ctx.update(b"unfer:nullifier");
    ctx.update(coin_id.0);
    Nullifier(ctx.finalize().into())
}

/// Determines which authority is allowed to mint certificates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MintAuthority {
    /// Minting is disabled (safe default).
    None,
    /// Only the given DID may mint.
    Only(String),
    /// A t-of-n threshold authority (Arctic/Shine, reusing the sibling
    /// `dynamic-arctic` crate): a valid Arctic aggregate signature over the
    /// canonical op bytes must verify against `pubkey` (the compressed group
    /// Ristretto point). The aggregate signature occupies the op's existing
    /// 64-byte `signature` field (Ristretto point + scalar).
    Threshold {
        threshold: u32,
        total: u32,
        pubkey: [u8; 32],
    },
}

/// The certificate ledger state-transition engine.
#[derive(Debug)]
pub struct CertificateLedger {
    smt: SparseMerkle,
    utxos: HashMap<[u8; 32], Coin>,
    spent: HashSet<Nullifier>,
    mint_authority: MintAuthority,
}

impl Default for CertificateLedger {
    fn default() -> Self {
        Self::new(MintAuthority::None)
    }
}

impl CertificateLedger {
    pub fn new(mint_authority: MintAuthority) -> Self {
        Self {
            smt: SparseMerkle::empty(),
            utxos: HashMap::new(),
            spent: HashSet::new(),
            mint_authority,
        }
    }

    pub fn root(&self) -> [u8; 32] {
        self.smt.root()
    }

    pub fn unspent_count(&self) -> usize {
        self.utxos.len()
    }

    pub fn utxo(&self, coin_id: &CertId) -> Option<&Coin> {
        self.utxos.get(&coin_id.0)
    }

    pub fn is_spent(&self, coin_id: &CertId) -> bool {
        self.spent.contains(&nullifier_for(coin_id))
    }

    pub fn total_supply(&self) -> u64 {
        self.utxos.values().map(|c| c.amount).sum()
    }

    /// All unspent coins currently owned by `owner_did` — an audit view of a
    /// party's balance across the consensus certificate state.
    pub fn coins_of(&self, owner_did: &str) -> Vec<Coin> {
        self.utxos
            .values()
            .filter(|c| c.owner == owner_did)
            .cloned()
            .collect()
    }

    fn mint_authorized(&self, did: &str) -> bool {
        match &self.mint_authority {
            MintAuthority::None => false,
            MintAuthority::Only(authority) => did == authority,
            // The DID for a threshold authority is the group DID derived from
            // the aggregate pubkey (hex); the cryptographic check is the Arctic
            // signature verification in `verify_threshold_mint`.
            MintAuthority::Threshold { pubkey, .. } => {
                did == format!("did:unfer:{}", hex::encode(pubkey))
            }
        }
    }

    /// Whether the ledger is configured with a threshold mint authority.
    pub fn is_threshold_authority(&self) -> bool {
        matches!(self.mint_authority, MintAuthority::Threshold { .. })
    }

    /// The configured threshold parameters, if any.
    pub fn threshold_params(&self) -> Option<(u32, u32)> {
        match self.mint_authority {
            MintAuthority::Threshold {
                threshold, total, ..
            } => Some((threshold, total)),
            _ => None,
        }
    }

    /// Verify a mint op's Arctic threshold signature against the configured
    /// threshold authority. Only meaningful for [`MintAuthority::Threshold`];
    /// any other authority refuses with `CERT_MINT_NOT_AUTHORIZED` (the
    /// single-DID path keeps the existing Ed25519 check in `verify_transaction`).
    pub fn verify_threshold_mint(&self, tx: &ConsensusTransaction) -> Result<(), Diagnostic> {
        let op = match tx {
            ConsensusTransaction::CertificateOp(op) => op,
            _ => {
                return Err(self.diag(
                    Code::CERT_MINT_NOT_AUTHORIZED,
                    "threshold mint verification requires a CertificateOp",
                ));
            }
        };
        let (threshold, total, pubkey) = match &self.mint_authority {
            MintAuthority::Threshold {
                threshold,
                total,
                pubkey,
            } => (*threshold, *total, *pubkey),
            _ => {
                return Err(self.diag(
                    Code::CERT_MINT_NOT_AUTHORIZED,
                    "ledger is not configured with a threshold mint authority",
                ));
            }
        };
        if !crate::signing::verify_arctic_threshold(pubkey, threshold, total, tx, &op.signature) {
            return Err(self.diag(
                Code::CERT_MINT_NOT_AUTHORIZED,
                "threshold signature verification failed",
            ));
        }
        Ok(())
    }

    /// Apply a mint: issue `amount` to `owner`. Returns the new coin_id.
    pub fn apply_mint(
        &mut self,
        actor: &str,
        amount: u64,
        owner: &str,
        blinding: &[u8; 32],
        seq: u64,
    ) -> Result<CertId, Diagnostic> {
        if amount == 0 {
            return Err(self.diag(Code::CERT_AMOUNT_MISMATCH, "mint amount must be positive"));
        }
        if !self.mint_authorized(actor) {
            return Err(self.diag(
                Code::CERT_MINT_NOT_AUTHORIZED,
                format!("{actor} is not the configured mint authority"),
            ));
        }
        let coin_id = commit_coin(amount, owner, blinding);
        if self.utxos.contains_key(&coin_id.0) {
            return Err(self.diag(
                Code::CERT_DOUBLE_SPEND,
                "mint collides with an existing certificate",
            ));
        }
        let coin = Coin {
            coin_id,
            amount,
            owner: owner.to_string(),
            blinding: *blinding,
            minted_seq: seq,
        };
        self.utxos.insert(coin_id.0, coin);
        self.smt.insert(&coin_id.0, coin_id.0);
        Ok(coin_id)
    }

    fn spend_input(&self, input: &CoinRef, actor: &str) -> Result<&Coin, Diagnostic> {
        let coin = self.utxos.get(&input.coin_id.0).ok_or_else(|| {
            self.diag(
                Code::CERT_NONEXISTENT_INPUT,
                format!("input {:?} is not an unspent certificate", input.coin_id),
            )
        })?;
        if coin.owner != actor {
            return Err(self.diag(
                Code::CERT_OWNER_MISMATCH,
                format!("{actor} is not the owner of {:?}", input.coin_id),
            ));
        }
        if input.amount != coin.amount {
            return Err(self.diag(
                Code::CERT_AMOUNT_MISMATCH,
                format!(
                    "declared amount {} != stored amount {} for {:?}",
                    input.amount, coin.amount, input.coin_id
                ),
            ));
        }
        Ok(coin)
    }

    /// Apply a transfer. Spends `inputs`, creates `outputs`, conserving value.
    /// Returns the new coin_ids.
    pub fn apply_transfer(
        &mut self,
        spender: &str,
        inputs: &[CoinRef],
        outputs: &[CoinRef],
        seq: u64,
    ) -> Result<Vec<CertId>, Diagnostic> {
        if inputs.is_empty() {
            return Err(self.diag(
                Code::CERT_NONEXISTENT_INPUT,
                "transfer requires at least one input",
            ));
        }
        // 1. Validate all inputs exist, are unspent, distinct, owned by spender.
        let mut in_sum: u64 = 0;
        let mut seen_inputs: HashSet<Nullifier> = HashSet::with_capacity(inputs.len());
        for input in inputs {
            let null = nullifier_for(&input.coin_id);
            if self.spent.contains(&null) {
                return Err(self.diag(
                    Code::CERT_DOUBLE_SPEND,
                    format!("nullifier {:?} already spent", null),
                ));
            }
            if !seen_inputs.insert(null) {
                return Err(self.diag(
                    Code::CERT_DOUBLE_SPEND,
                    format!("input nullifier {:?} listed twice", null),
                ));
            }
            let coin = self.spend_input(input, spender)?;
            in_sum = in_sum
                .checked_add(coin.amount)
                .ok_or_else(|| self.diag(Code::CERT_AMOUNT_MISMATCH, "input sum overflow"))?;
        }
        // 2. Validate outputs: positive amounts, unique ids, no collision.
        let mut out_sum: u64 = 0;
        let mut new_ids = Vec::with_capacity(outputs.len());
        let mut seen_outputs: HashSet<CertId> = HashSet::with_capacity(outputs.len());
        for out in outputs {
            if out.amount == 0 {
                return Err(self.diag(Code::CERT_AMOUNT_MISMATCH, "output amount must be positive"));
            }
            let id = commit_coin(out.amount, &out.owner, &[0u8; 32]);
            out_sum = out_sum
                .checked_add(out.amount)
                .ok_or_else(|| self.diag(Code::CERT_AMOUNT_MISMATCH, "output sum overflow"))?;
            if self.utxos.contains_key(&id.0) || !seen_outputs.insert(id) {
                return Err(self.diag(
                    Code::CERT_DOUBLE_SPEND,
                    format!("output {:?} collides with an unspent certificate", id),
                ));
            }
            new_ids.push(id);
        }
        // 3. Conservation: Sum(inputs) == Sum(outputs).
        if in_sum != out_sum {
            return Err(self.diag(
                Code::CERT_AMOUNT_MISMATCH,
                format!("input sum {in_sum} != output sum {out_sum}"),
            ));
        }
        // 4. Commit: spend inputs, create outputs.
        for input in inputs {
            self.spent.insert(nullifier_for(&input.coin_id));
            self.utxos.remove(&input.coin_id.0);
            self.smt.remove(&input.coin_id.0);
        }
        for (out, id) in outputs.iter().zip(new_ids.iter()) {
            let coin = Coin {
                coin_id: *id,
                amount: out.amount,
                owner: out.owner.clone(),
                blinding: [0u8; 32],
                minted_seq: seq,
            };
            self.utxos.insert(id.0, coin);
            self.smt.insert(&id.0, id.0);
        }
        Ok(new_ids)
    }

    /// Apply a burn (retirement): consumes `inputs`, removing their value from
    /// circulation. Conservation intentionally does not apply.
    pub fn apply_burn(
        &mut self,
        spender: &str,
        inputs: &[CoinRef],
        _seq: u64,
    ) -> Result<u64, Diagnostic> {
        if inputs.is_empty() {
            return Err(self.diag(Code::CERT_NONEXISTENT_INPUT, "burn requires an input"));
        }
        let mut burned: u64 = 0;
        for input in inputs {
            let null = nullifier_for(&input.coin_id);
            if self.spent.contains(&null) {
                return Err(self.diag(Code::CERT_DOUBLE_SPEND, "nullifier already spent"));
            }
            let coin = self.spend_input(input, spender)?;
            burned = burned
                .checked_add(coin.amount)
                .ok_or_else(|| self.diag(Code::CERT_AMOUNT_MISMATCH, "burn sum overflow"))?;
            self.spent.insert(null);
            self.utxos.remove(&input.coin_id.0);
            self.smt.remove(&input.coin_id.0);
        }
        Ok(burned)
    }

    /// Dispatch a signed certificate op against the ledger. `actor` is the
    /// signer's DID (from the op), already verified by the caller.
    pub fn apply_op(
        &mut self,
        actor: &str,
        kind: &CertificateOpKind,
        seq: u64,
    ) -> Result<Vec<CertId>, Diagnostic> {
        match kind {
            CertificateOpKind::Mint {
                amount,
                owner,
                blinding,
                ..
            } => Ok(vec![self.apply_mint(actor, *amount, owner, blinding, seq)?]),
            CertificateOpKind::Transfer { inputs, outputs } => {
                self.apply_transfer(actor, inputs, outputs, seq)
            }
            CertificateOpKind::Burn { inputs } => {
                self.apply_burn(actor, inputs, seq)?;
                Ok(Vec::new())
            }
        }
    }

    fn diag(&self, code: Code, msg: impl Into<String>) -> Diagnostic {
        Diagnostic::new(code, msg, Severity::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auth() -> String {
        "did:unfer:authority".to_string()
    }

    fn ledger() -> CertificateLedger {
        CertificateLedger::new(MintAuthority::Only(auth()))
    }

    fn coinref(id: CertId, amount: u64, owner: &str) -> CoinRef {
        CoinRef {
            coin_id: id,
            amount,
            owner: owner.to_string(),
        }
    }

    #[test]
    fn mint_requires_authority() {
        let mut l = ledger();
        let bad = l.apply_mint("did:unfer:nobody", 100, "did:unfer:alice", &[1u8; 32], 1);
        assert_eq!(bad.unwrap_err().code, Code::CERT_MINT_NOT_AUTHORIZED);
        let ok = l.apply_mint(&auth(), 100, "did:unfer:alice", &[1u8; 32], 1);
        assert!(ok.is_ok());
        assert_eq!(l.total_supply(), 100);
    }

    #[test]
    fn mint_is_idempotent_per_commitment() {
        let mut l = ledger();
        let id1 = l
            .apply_mint(&auth(), 100, "did:unfer:alice", &[1u8; 32], 1)
            .unwrap();
        let dup = l.apply_mint(&auth(), 100, "did:unfer:alice", &[1u8; 32], 2);
        assert_eq!(dup.unwrap_err().code, Code::CERT_DOUBLE_SPEND);
        assert_eq!(l.unspent_count(), 1);
        assert_eq!(l.utxo(&id1).unwrap().amount, 100);
    }

    #[test]
    fn transfer_conserves_value() {
        let mut l = ledger();
        let alice = "did:unfer:alice";
        let bob = "did:unfer:bob";
        let id = l.apply_mint(&auth(), 1000, alice, &[1u8; 32], 1).unwrap();
        let out = coinref(id, 1000, bob);
        let new_ids = l
            .apply_transfer(alice, &[coinref(id, 1000, alice)], &[out], 2)
            .unwrap();
        assert_eq!(new_ids.len(), 1);
        assert!(l.utxo(&new_ids[0]).is_some());
        assert!(l.utxo(&id).is_none(), "input spent");
        assert!(l.is_spent(&id));
        assert_eq!(l.total_supply(), 1000);
    }

    #[test]
    fn transfer_split_outputs() {
        let mut l = ledger();
        let alice = "did:unfer:alice";
        let bob = "did:unfer:bob";
        let id = l.apply_mint(&auth(), 1000, alice, &[1u8; 32], 1).unwrap();
        let outs = vec![coinref(id, 400, bob), coinref(id, 600, alice)];
        let ids = l
            .apply_transfer(alice, &[coinref(id, 1000, alice)], &outs, 2)
            .unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(l.total_supply(), 1000);
    }

    #[test]
    fn conservation_violation_rejected() {
        let mut l = ledger();
        let alice = "did:unfer:alice";
        let bob = "did:unfer:bob";
        let id = l.apply_mint(&auth(), 1000, alice, &[1u8; 32], 1).unwrap();
        // Creating 1100 out of 1000 → UK-7002.
        let err = l
            .apply_transfer(
                alice,
                &[coinref(id, 1000, alice)],
                &[coinref(id, 1100, bob)],
                2,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::CERT_AMOUNT_MISMATCH);
        assert!(l.utxo(&id).is_some(), "inputs untouched on reject");
        assert_eq!(l.total_supply(), 1000);
    }

    #[test]
    fn double_spend_rejected() {
        let mut l = ledger();
        let alice = "did:unfer:alice";
        let id = l.apply_mint(&auth(), 500, alice, &[1u8; 32], 1).unwrap();
        let out1 = coinref(id, 500, alice);
        l.apply_transfer(alice, &[coinref(id, 500, alice)], &[out1], 2)
            .unwrap();
        // Re-spending the (now spent) input → UK-7004.
        let err = l
            .apply_transfer(
                alice,
                &[coinref(id, 500, alice)],
                &[coinref(id, 500, alice)],
                3,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::CERT_DOUBLE_SPEND);
    }

    #[test]
    fn owner_mismatch_rejected() {
        let mut l = ledger();
        let alice = "did:unfer:alice";
        let mallory = "did:unfer:mallory";
        let id = l.apply_mint(&auth(), 500, alice, &[1u8; 32], 1).unwrap();
        let err = l
            .apply_transfer(
                mallory,
                &[coinref(id, 500, alice)],
                &[coinref(id, 500, mallory)],
                2,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::CERT_OWNER_MISMATCH);
    }

    #[test]
    fn duplicate_input_rejected() {
        let mut l = ledger();
        let alice = "did:unfer:alice";
        let bob = "did:unfer:bob";
        let id = l.apply_mint(&auth(), 500, alice, &[1u8; 32], 1).unwrap();
        // Listing the same coin twice must not let 500 turn into 1000.
        let err = l
            .apply_transfer(
                alice,
                &[coinref(id, 500, alice), coinref(id, 500, alice)],
                &[coinref(id, 1000, bob)],
                2,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::CERT_DOUBLE_SPEND);
        assert!(l.utxo(&id).is_some(), "inputs untouched on reject");
        assert_eq!(l.total_supply(), 500);
    }

    #[test]
    fn duplicate_output_rejected() {
        let mut l = ledger();
        let alice = "did:unfer:alice";
        let bob = "did:unfer:bob";
        let id = l.apply_mint(&auth(), 1000, alice, &[1u8; 32], 1).unwrap();
        // Two outputs with the same (amount, owner, blinding) collide.
        let err = l
            .apply_transfer(
                alice,
                &[coinref(id, 1000, alice)],
                &[coinref(id, 500, bob), coinref(id, 500, bob)],
                2,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::CERT_DOUBLE_SPEND);
        assert!(l.utxo(&id).is_some(), "inputs untouched on reject");
        assert_eq!(l.unspent_count(), 1);
    }

    #[test]
    fn burn_retires_value() {
        let mut l = ledger();
        let alice = "did:unfer:alice";
        let id = l.apply_mint(&auth(), 700, alice, &[1u8; 32], 1).unwrap();
        let burned = l.apply_burn(alice, &[coinref(id, 700, alice)], 2).unwrap();
        assert_eq!(burned, 700);
        assert!(l.utxo(&id).is_none());
        assert_eq!(l.total_supply(), 0);
    }

    #[test]
    fn root_changes_on_apply() {
        let mut l = ledger();
        let alice = "did:unfer:alice";
        let r0 = l.root();
        let id = l.apply_mint(&auth(), 100, alice, &[1u8; 32], 1).unwrap();
        let r1 = l.root();
        assert_ne!(r0, r1, "mint changes the root");
        l.apply_burn(alice, &[coinref(id, 100, alice)], 2).unwrap();
        let r2 = l.root();
        assert_eq!(r2, r0, "retiring the only UTXO restores the empty root");
    }

    #[test]
    fn smt_insert_remove_roundtrip() {
        let mut smt = SparseMerkle::empty();
        let empty = smt.root();
        let key = [7u8; 32];
        smt.insert(&key, key);
        assert!(smt.contains(&key));
        assert_ne!(smt.root(), empty);
        smt.remove(&key);
        assert!(!smt.contains(&key));
        assert_eq!(smt.root(), empty);
    }
}

// ── property tests (Plan R audit surface) ─────────────────────────────
//
// The certificate ledger is the audit surface of the ReFi exchange, so the
// load-bearing invariants — conservation, no-double-spend, and exact supply —
// are fuzzed as properties rather than pinned to a handful of hand-written
// cases. Each attempt is a *multi-input / multi-output* transfer so that the
// duplicate-input and duplicate-output guards are exercised too:
//   * a transfer is accepted iff inputs are fresh & distinct AND outputs are
//     fresh & distinct AND Sum(inputs) == Sum(outputs);
//   * an input whose nullifier was already consumed is always refused;
//   * `total_supply` always equals the sum of the currently-unspent coins.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use unfer_protocol::{CertificateOp, CertificateOpKind};

    const ALICE: &str = "did:unfer_pt:alice";
    const BOB: &str = "did:unfer_pt:bob";

    fn auth() -> String {
        "did:unfer_pt:authority".to_string()
    }

    /// The commitment id an output would get (zero blinding, like the ledger).
    fn out_id(amount: u64, owner: &str) -> CertId {
        commit_coin(amount, owner, &[0u8; 32])
    }

    proptest! {
        #[test]
        fn fuzz_transfers_never_break_conservation_or_double_spend(
            mint_amounts in prop::collection::vec(1u64..1_000, 1..8),
            // A sequence of multi-input/multi-output transfer attempts. Input
            // indices may repeat (duplicate-input guard) and output amounts are
            // deliberately independent of the input amounts so the fuzzer
            // produces both conserving and non-conserving transfers.
            attempts in prop::collection::vec(
                (
                    prop::collection::vec(0usize..8, 1..4),
                    prop::collection::vec(0u64..2_000, 1..4),
                ),
                0..40,
            ),
        ) {
            let alice = ALICE.to_string();
            let bob = BOB.to_string();
            let mut ledger = CertificateLedger::new(MintAuthority::Only(auth()));

            // Mint one coin per amount, all owned by ALICE. Track each coin's
            // original amount so a spent coin can still be re-attempted (the
            // ledger no longer holds it).
            let mut live: Vec<(CertId, u64)> = Vec::new();
            for (i, a) in mint_amounts.iter().enumerate() {
                let id = ledger
                    .apply_mint(&auth(), *a, &alice, &[i as u8; 32], i as u64)
                    .unwrap();
                live.push((id, *a));
            }
            let expected_supply: u64 = mint_amounts.iter().sum();

            for (in_idxs, out_amounts) in attempts.into_iter() {
                if live.is_empty() {
                    break;
                }

                let inputs: Vec<CoinRef> = in_idxs
                    .iter()
                    .map(|&i| {
                        let (coin_id, coin_amount) = live[i % live.len()];
                        CoinRef {
                            coin_id,
                            amount: coin_amount,
                            owner: alice.clone(),
                        }
                    })
                    .collect();
                let outputs: Vec<CoinRef> = out_amounts
                    .iter()
                    .map(|&a| CoinRef {
                        coin_id: CertId([0u8; 32]),
                        amount: a,
                        owner: bob.clone(),
                    })
                    .collect();

                let in_sum: u64 = inputs.iter().map(|i| i.amount).sum();
                let out_sum: u64 = outputs.iter().map(|o| o.amount).sum();

                // What the ledger *should* do:
                //  * every listed input is fresh (nullifier unconsumed);
                //  * no input coin listed twice;
                //  * every output amount is positive;
                //  * every output commitment is fresh and not repeated.
                let all_inputs_fresh = inputs.iter().all(|i| !ledger.is_spent(&i.coin_id));
                let inputs_distinct = {
                    let mut seen = std::collections::HashSet::new();
                    inputs.iter().all(|i| seen.insert(i.coin_id))
                };
                let outputs_positive = outputs.iter().all(|o| o.amount > 0);
                let outputs_distinct_fresh = {
                    let mut seen = std::collections::HashSet::new();
                    outputs.iter().all(|o| {
                        let id = out_id(o.amount, &o.owner);
                        !ledger.utxo(&id).is_some() && seen.insert(id)
                    })
                };
                let should_accept = all_inputs_fresh
                    && inputs_distinct
                    && outputs_positive
                    && outputs_distinct_fresh
                    && in_sum == out_sum;

                let before_supply = ledger.total_supply();
                let before_unspent = ledger.unspent_count();

                let res = ledger.apply_transfer(&alice, &inputs, &outputs, 1);

                match res {
                    Ok(new_ids) => {
                        prop_assert!(should_accept, "accepted an invalid transfer");
                        prop_assert_eq!(new_ids.len(), outputs.len());
                        // Exact conservation => supply unchanged.
                        prop_assert_eq!(ledger.total_supply(), before_supply);
                        // All inputs must now be spent.
                        for i in &inputs {
                            prop_assert!(ledger.is_spent(&i.coin_id));
                        }
                    }
                    Err(diag) => {
                        prop_assert!(
                            !should_accept,
                            "refused a conserving, fresh, distinct transfer: {diag}"
                        );
                        prop_assert!(
                            diag.code == Code::CERT_AMOUNT_MISMATCH
                                || diag.code == Code::CERT_DOUBLE_SPEND,
                            "unexpected refusal code: {diag}"
                        );
                        // State is untouched on refusal.
                        prop_assert_eq!(ledger.total_supply(), before_supply);
                        prop_assert_eq!(ledger.unspent_count(), before_unspent);
                    }
                }

                // Invariant: total supply always equals the sum of live coins.
                prop_assert_eq!(ledger.total_supply(), expected_supply);
            }
        }
    }

    // ── Threshold mint authority (Arctic/Shine, sibling dynamic-arctic) ──────

    /// Build a signed CertificateOp::Mint whose signature is a valid Arctic
    /// t-of-n aggregate signature over the canonical op bytes.
    fn threshold_mint_op(
        n: u32,
        t: u32,
        coalition: &[u32],
        amount: u64,
        owner: &str,
        blinding: [u8; 32],
        seq: u64,
    ) -> (ConsensusTransaction, [u8; 32]) {
        let (group_pk, _, seckeys) = arctic::arctic_core::keygen(n, t);

        let mut op = CertificateOp {
            did: format!("did:unfer:{}", hex::encode(group_pk.compress().to_bytes())),
            kind: CertificateOpKind::Mint {
                amount,
                owner: owner.to_string(),
                blinding,
                source: None,
            },
            seq,
            signature: [0u8; 64],
        };
        let tx = ConsensusTransaction::CertificateOp(op.clone());
        let real_msg = crate::signing::canonical_bytes(&tx);

        // Arctic two-round signing over the canonical bytes.
        let r1_outputs: Vec<arctic::arctic_core::R1Output> = coalition
            .iter()
            .map(|&k| arctic::arctic_core::sign1(&seckeys[(k - 1) as usize], coalition, &real_msg))
            .collect();
        let sigshares: Vec<curve25519_dalek::scalar::Scalar> = coalition
            .iter()
            .map(|&k| {
                arctic::arctic_core::sign2(
                    &group_pk,
                    &seckeys[(k - 1) as usize],
                    coalition,
                    &real_msg,
                    &r1_outputs,
                )
                .unwrap()
            })
            .collect();
        let sig = arctic::arctic_core::combine(
            &group_pk,
            t,
            coalition,
            &real_msg,
            &r1_outputs,
            &sigshares,
        )
        .expect("honest coalition must combine");

        // Serialize (RistrettoPoint, Scalar) → 32 + 32 = 64 bytes.
        let mut sig_bytes = [0u8; 64];
        sig_bytes[..32].copy_from_slice(sig.0.compress().as_bytes());
        sig_bytes[32..].copy_from_slice(&sig.1.to_bytes());
        op.signature = sig_bytes;

        let pubkey = group_pk.compress().to_bytes();
        (ConsensusTransaction::CertificateOp(op), pubkey)
    }

    #[test]
    fn threshold_mint_authority_accepts_t_of_n_sig() {
        let (tx, pubkey) = threshold_mint_op(
            7,
            4,
            &[1, 2, 3, 4, 5, 6, 7],
            100,
            "did:unfer:alice",
            [1u8; 32],
            1,
        );
        let mut l = CertificateLedger::new(MintAuthority::Threshold {
            threshold: 4,
            total: 7,
            pubkey,
        });
        assert!(l.is_threshold_authority());
        assert_eq!(l.threshold_params(), Some((4, 7)));

        // The threshold signature verifies against the group key.
        l.verify_threshold_mint(&tx)
            .expect("threshold sig must verify");

        // And the mint applies.
        let op = match &tx {
            ConsensusTransaction::CertificateOp(op) => op,
            _ => unreachable!(),
        };
        let actor = op.did.clone();
        let kind = op.kind.clone();
        l.apply_op(&actor, &kind, op.seq).expect("mint applies");
        assert_eq!(l.total_supply(), 100);
        assert_eq!(l.unspent_count(), 1);
    }

    #[test]
    fn threshold_mint_authority_rejects_small_coalition() {
        // A 2-of-7 coalition cannot produce a valid 4-of-7 aggregate signature:
        // Arctic's combine is only defined for |coalition| >= 2t-1, so the
        // signers simply cannot build a verifiable signature — the ledger must
        // refuse the best they can produce. We emulate a forged attempt by
        // taking a valid 7-of-7 signature and corrupting it, then check the
        // ledger refuses.
        let (mut tx, pubkey) = threshold_mint_op(
            7,
            4,
            &[1, 2, 3, 4, 5, 6, 7],
            100,
            "did:unfer:alice",
            [1u8; 32],
            1,
        );
        let l = CertificateLedger::new(MintAuthority::Threshold {
            threshold: 4,
            total: 7,
            pubkey,
        });
        if let ConsensusTransaction::CertificateOp(op) = &mut tx {
            op.signature[0] ^= 0xff;
        }
        // The gate (node.rs calls this before submit/replay) refuses the
        // corrupted aggregate signature. `apply_op` itself trusts the DID
        // exactly like the single-key path trusts `verify_transaction` — the
        // cryptographic check lives in the gate.
        let err = l.verify_threshold_mint(&tx).unwrap_err();
        assert_eq!(err.code, Code::CERT_MINT_NOT_AUTHORIZED);
        assert_eq!(l.total_supply(), 0);
    }

    #[test]
    fn threshold_mint_authority_rejects_wrong_message() {
        // Sign one message, submit a different op: the aggregate signature is
        // message-bound (hash3 includes the canonical bytes), so verification
        // must fail.
        let (mut tx, pubkey) = threshold_mint_op(
            7,
            4,
            &[1, 2, 3, 4, 5, 6, 7],
            100,
            "did:unfer:alice",
            [1u8; 32],
            1,
        );
        let l = CertificateLedger::new(MintAuthority::Threshold {
            threshold: 4,
            total: 7,
            pubkey,
        });
        if let ConsensusTransaction::CertificateOp(op) = &mut tx {
            op.seq = 999; // changes the canonical bytes
        }
        let err = l.verify_threshold_mint(&tx).unwrap_err();
        assert_eq!(err.code, Code::CERT_MINT_NOT_AUTHORIZED);
    }

    #[test]
    fn threshold_authority_rejects_non_threshold_ledger() {
        let (tx, pubkey) = threshold_mint_op(
            7,
            4,
            &[1, 2, 3, 4, 5, 6, 7],
            100,
            "did:unfer:alice",
            [1u8; 32],
            1,
        );
        let l = CertificateLedger::new(MintAuthority::Only("did:unfer:authority".into()));
        assert!(!l.is_threshold_authority());
        let err = l.verify_threshold_mint(&tx).unwrap_err();
        assert_eq!(err.code, Code::CERT_MINT_NOT_AUTHORIZED);
        let _ = pubkey;
    }

    #[test]
    fn threshold_mint_replay_converges() {
        // Determinism: two ledgers with the same threshold authority and the
        // same log produce the same state — the Arctic verification is a pure
        // function of (pubkey, tx), so identical log in ⇒ identical verdict out.
        let (tx, pubkey) = threshold_mint_op(
            7,
            4,
            &[1, 2, 3, 4, 5, 6, 7],
            100,
            "did:unfer:alice",
            [1u8; 32],
            1,
        );
        let mut a = CertificateLedger::new(MintAuthority::Threshold {
            threshold: 4,
            total: 7,
            pubkey,
        });
        let mut b = CertificateLedger::new(MintAuthority::Threshold {
            threshold: 4,
            total: 7,
            pubkey,
        });
        let op = match &tx {
            ConsensusTransaction::CertificateOp(op) => op,
            _ => unreachable!(),
        };
        let actor = op.did.clone();
        let kind = op.kind.clone();
        a.verify_threshold_mint(&tx).unwrap();
        b.verify_threshold_mint(&tx).unwrap();
        a.apply_op(&actor, &kind, op.seq).unwrap();
        b.apply_op(&actor, &kind, op.seq).unwrap();
        assert_eq!(a.root(), b.root());
        assert_eq!(a.total_supply(), b.total_supply());
        assert_eq!(a.unspent_count(), b.unspent_count());
    }
}
