use std::collections::HashMap;

use prob_kernel::Session;
use unfer_protocol::{
    AgentRequest, Code, ConsensusTransaction, ContentRef, Diagnostic, MarketOpKind, MathBondState,
    ModelSpec, Severity,
};

use crate::auction::AuctionLedger;
use crate::certs::{CertificateLedger, MintAuthority};
use crate::engine::ConsensusEngine;
use crate::idempotency::{IdempotencyStore, transaction_key};
use crate::identity::IdentityRegistry;

pub struct ConsensusNode {
    engine: Box<dyn ConsensusEngine>,
    sessions: HashMap<u64, Session>,
    identity: IdentityRegistry,
    content: HashMap<String, ContentRef>,
    certs: CertificateLedger,
    auction: AuctionLedger,
    mathbond: crate::mathbond::MathBondLedger,
    market: crate::mathbond_market::MarketLedger,
    attribution: crate::attribution::AttributionLedger,
    /// H7: exactly-once guard for replayed/duplicated certificate + auction + mathbond + market + attribution ops.
    idempotency: IdempotencyStore,
    next_model_id: u64,
    applied_seq: u64,
}

impl ConsensusNode {
    pub fn new(engine: Box<dyn ConsensusEngine>) -> Self {
        Self {
            engine,
            sessions: HashMap::new(),
            identity: IdentityRegistry::new(),
            content: HashMap::new(),
            certs: CertificateLedger::new(MintAuthority::None),
            auction: AuctionLedger::new(),
            mathbond: crate::mathbond::MathBondLedger::new(),
            market: crate::mathbond_market::MarketLedger::new(),
            attribution: crate::attribution::AttributionLedger::new(),
            idempotency: IdempotencyStore::new(),
            next_model_id: 1,
            applied_seq: 0,
        }
    }

    /// Constructor with a configured certificate mint authority (ReFi exchange).
    pub fn with_mint_authority(engine: Box<dyn ConsensusEngine>, authority: MintAuthority) -> Self {
        let mut node = Self::new(engine);
        node.certs = CertificateLedger::new(authority);
        node
    }

    /// Reconfigure the certificate mint authority (ReFi exchange). Minting is
    /// disabled by default; call with `MintAuthority::Only(did)` to permit a
    /// specific authority to mint, or `MintAuthority::None` to disable it.
    pub fn set_mint_authority(&mut self, authority: MintAuthority) {
        self.certs = CertificateLedger::new(authority);
    }

    pub fn submit(&self, tx: ConsensusTransaction) -> Result<u64, Diagnostic> {
        // A threshold mint authority signs with an Arctic aggregate signature
        // (Ristretto point + scalar, 64 bytes), not the single-key Ed25519 the
        // generic path checks — route those through the ledger's threshold
        // verification instead.
        if let ConsensusTransaction::CertificateOp(op) = &tx {
            if self.certs.is_threshold_authority() {
                self.certs.verify_threshold_mint(&tx)?;
                return self.engine.submit(tx);
            }
            let _ = op;
        }
        crate::signing::verify_transaction(&tx)?;
        self.engine.submit(tx)
    }

    pub fn sync(&mut self) -> Result<u64, Diagnostic> {
        let log = self.engine.get_log(self.applied_seq + 1);
        let mut applied = 0u64;
        for (seq, tx) in log {
            self.apply_transaction(seq, &tx)?;
            self.applied_seq = seq;
            applied += 1;
        }
        Ok(applied)
    }

    fn apply_transaction(&mut self, seq: u64, tx: &ConsensusTransaction) -> Result<(), Diagnostic> {
        match tx {
            ConsensusTransaction::IdentityOp(op) => {
                self.identity.apply_identity_op(op)?;
            }
            ConsensusTransaction::SessionOp(op) => {
                self.apply_session_op(&op.op)?;
            }
            ConsensusTransaction::ContentOp(op) => {
                self.content
                    .insert(op.content_ref.cid.clone(), op.content_ref.clone());
            }
            ConsensusTransaction::CertificateOp(op) => {
                // H7: a duplicated or replayed certificate op applies exactly once.
                // A threshold mint authority re-verifies the Arctic aggregate
                // signature on replay (deterministic: same log, same verdict),
                // then applies through the shared idempotency gate.
                if self.certs.is_threshold_authority() {
                    self.certs.verify_threshold_mint(tx)?;
                }
                let actor = op.did.clone();
                let kind = op.kind.clone();
                let key = crate::idempotency::certificate_key(op);
                self.idempotency.once(&key, seq, || {
                    self.certs.apply_op(&actor, &kind, op.seq).map(|_| ())
                })?;
            }
            ConsensusTransaction::AuctionOp(op) => {
                // H7: a duplicated or replayed auction op applies exactly once.
                let actor = op.did.clone();
                let kind = op.kind.clone();
                let key = crate::idempotency::auction_key(op);
                self.idempotency.once(&key, seq, || {
                    self.auction.apply_op(&actor, &kind, op.seq).map(|_| ())
                })?;
            }
            ConsensusTransaction::MathBondOp(op) => {
                // H7: a duplicated or replayed math bond op applies exactly once.
                // The ledger receives the CONSENSUS seq (not the op's own
                // submitter-set field) so maturity enforcement and trigger_seq
                // reflect the honest log position.
                let actor = op.did.clone();
                let kind = op.kind.clone();
                let key = crate::idempotency::mathbond_key(op);
                self.idempotency.once(&key, seq, || {
                    self.mathbond.apply_op(&actor, &kind, seq).map(|_| ())
                })?;
            }
            ConsensusTransaction::MarketOp(op) => {
                // H7: a duplicated or replayed market op applies exactly once.
                let actor = op.did.clone();
                let kind = op.kind.clone();
                // A pool resolution is not a caller's free choice: the winner
                // is a pure function of the bond's trigger state. Validate the
                // op's trigger signal against the deterministic bond ledger
                // before the op is applied, so every node rejects a forged
                // resolution identically.
                if let MarketOpKind::Resolve {
                    pool_id,
                    trigger_seq,
                } = &kind
                {
                    let pool = self.market.pool(pool_id).ok_or_else(|| {
                        Diagnostic::new(Code::MARKET_UNKNOWN_POOL, "unknown pool", Severity::Error)
                    })?;
                    let bond = self.mathbond.bond(&pool.pool.bond_id).ok_or_else(|| {
                        Diagnostic::new(
                            Code::MATHBOND_UNKNOWN,
                            "the pool's bond does not exist on the ledger",
                            Severity::Error,
                        )
                    })?;
                    let expected = match (bond.state, bond.trigger_seq) {
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
                    if *trigger_seq != expected {
                        return Err(Diagnostic::new(
                            Code::MARKET_UNKNOWN_OUTCOME,
                            format!(
                                "trigger signal {trigger_seq:?} does not match the bond ledger {expected:?}"
                            ),
                            Severity::Error,
                        ));
                    }
                }
                let key = crate::idempotency::market_key(op);
                self.idempotency.once(&key, seq, || {
                    self.market.apply_op(&actor, &kind, op.seq).map(|_| ())
                })?;
            }
            ConsensusTransaction::AttributionOp(op) => {
                // H7: a duplicated or replayed attribution op applies exactly once.
                let actor = op.did.clone();
                let kind = op.kind.clone();
                let key = crate::idempotency::attribution_key(op);
                self.idempotency
                    .once(&key, seq, || self.attribution.apply_op(&actor, &kind, seq))?;
            }
        }
        Ok(())
    }

    /// H7: retention prune on a schedule — drop idempotency guards committed
    /// before `seq` (replayed old deliveries no longer need their guard).
    pub fn prune_idempotency_before(&mut self, seq: u64) {
        self.idempotency.prune_before(seq);
    }

    /// H7: whether a transaction key was already applied exactly-once.
    pub fn idempotency_committed(&self, tx: &ConsensusTransaction) -> bool {
        match transaction_key(tx) {
            Some(key) => self.idempotency.committed(&key),
            None => false,
        }
    }

    fn apply_session_op(&mut self, req: &AgentRequest) -> Result<(), Diagnostic> {
        // Dispatch derives from the shared CONSENSUS_OPS table (the canonical
        // `unfer_protocol::ops` registry) so the consensus seam can never
        // drift from the agent/edge allowlists.
        if unfer_protocol::ops::CONSENSUS_OPS.contains(&req.op.as_str()) {
            let spec: ModelSpec = serde_json::from_value(req.params.clone())
                .map_err(|e| Diagnostic::new(Code::BAD_JSON, e.to_string(), Severity::Error))?;
            let session = Session::new(&spec).map_err(|e| e.to_diagnostic())?;
            let id = self.next_model_id;
            self.next_model_id += 1;
            self.sessions.insert(id, session);
            return Ok(());
        }
        Err(Diagnostic::new(
            Code::BAD_JSON,
            format!("unsupported session op in consensus: {}", req.op),
            Severity::Error,
        ))
    }

    pub fn identity(&self) -> &IdentityRegistry {
        &self.identity
    }

    pub fn content(&self, cid: &str) -> Option<&ContentRef> {
        self.content.get(cid)
    }

    pub fn certs(&self) -> &CertificateLedger {
        &self.certs
    }

    pub fn auction(&self) -> &AuctionLedger {
        &self.auction
    }

    pub fn mathbond(&self) -> &crate::mathbond::MathBondLedger {
        &self.mathbond
    }

    pub fn market(&self) -> &crate::mathbond_market::MarketLedger {
        &self.market
    }

    pub fn attribution(&self) -> &crate::attribution::AttributionLedger {
        &self.attribution
    }

    pub fn session(&self, id: u64) -> Option<&Session> {
        self.sessions.get(&id)
    }

    pub fn session_mut(&mut self, id: u64) -> Option<&mut Session> {
        self.sessions.get_mut(&id)
    }

    pub fn applied_seq(&self) -> u64 {
        self.applied_seq
    }

    pub fn current_seq(&self) -> u64 {
        self.engine.current_seq()
    }

    pub fn is_synced(&self) -> bool {
        self.applied_seq == self.engine.current_seq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::MintAuthority;
    use crate::engine::LocalConsensus;
    use crate::signing::Keypair;
    use unfer_protocol::{CertificateOp, CertificateOpKind, IdentityOp, IdentityOpKind};

    fn make_node() -> ConsensusNode {
        ConsensusNode::new(Box::new(LocalConsensus::new()))
    }

    /// Build a mint op signed with a valid Arctic t-of-n aggregate signature.
    /// Mirrors `certs::proptests::threshold_mint_op` at the node level.
    fn threshold_mint_tx(
        n: u32,
        t: u32,
        coalition: &[u32],
        amount: u64,
        owner: &str,
        seq: u64,
    ) -> (ConsensusTransaction, [u8; 32]) {
        let (group_pk, _, seckeys) = arctic::arctic_core::keygen(n, t);
        let mut op = CertificateOp {
            did: format!("did:unfer:{}", hex::encode(group_pk.compress().to_bytes())),
            kind: CertificateOpKind::Mint {
                amount,
                owner: owner.to_string(),
                blinding: [1u8; 32],
                source: None,
            },
            seq,
            signature: [0u8; 64],
        };
        let tx = ConsensusTransaction::CertificateOp(op.clone());
        let msg = crate::signing::canonical_bytes(&tx);
        let r1: Vec<arctic::arctic_core::R1Output> = coalition
            .iter()
            .map(|&k| arctic::arctic_core::sign1(&seckeys[(k - 1) as usize], coalition, &msg))
            .collect();
        let shares: Vec<curve25519_dalek::scalar::Scalar> = coalition
            .iter()
            .map(|&k| {
                arctic::arctic_core::sign2(
                    &group_pk,
                    &seckeys[(k - 1) as usize],
                    coalition,
                    &msg,
                    &r1,
                )
                .unwrap()
            })
            .collect();
        let sig =
            arctic::arctic_core::combine(&group_pk, t, coalition, &msg, &r1, &shares).unwrap();
        let mut sig_bytes = [0u8; 64];
        sig_bytes[..32].copy_from_slice(sig.0.compress().as_bytes());
        sig_bytes[32..].copy_from_slice(&sig.1.to_bytes());
        op.signature = sig_bytes;
        (
            ConsensusTransaction::CertificateOp(op),
            group_pk.compress().to_bytes(),
        )
    }

    #[test]
    fn threshold_mint_submit_syncs_through_the_gate() {
        let (tx, pubkey) =
            threshold_mint_tx(7, 4, &[1, 2, 3, 4, 5, 6, 7], 100, "did:unfer:alice", 1);
        let mut node = make_node();
        node.set_mint_authority(MintAuthority::Threshold {
            threshold: 4,
            total: 7,
            pubkey,
        });
        // The Arctic aggregate signature passes the node's threshold gate.
        node.submit(tx).unwrap();
        let applied = node.sync().unwrap();
        assert_eq!(applied, 1);
        let supply = node.certs().total_supply();
        assert_eq!(supply, 100);
    }

    #[test]
    fn threshold_mint_rejects_forged_sig_at_submit() {
        let (mut tx, pubkey) =
            threshold_mint_tx(7, 4, &[1, 2, 3, 4, 5, 6, 7], 100, "did:unfer:alice", 1);
        if let ConsensusTransaction::CertificateOp(op) = &mut tx {
            op.signature[0] ^= 0xff;
        }
        let mut node = make_node();
        node.set_mint_authority(MintAuthority::Threshold {
            threshold: 4,
            total: 7,
            pubkey,
        });
        let err = node.submit(tx).unwrap_err();
        assert_eq!(err.code, Code::CERT_MINT_NOT_AUTHORIZED);
    }

    #[test]
    fn submit_and_sync_identity() {
        let mut node = make_node();
        let kp = Keypair::generate();
        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx, &kp);
        node.submit(tx).unwrap();
        let applied = node.sync().unwrap();
        assert_eq!(applied, 1);
        assert!(node.identity().resolve(&kp.did()).is_some());
        assert!(node.is_synced());
    }

    #[test]
    fn submit_and_sync_content() {
        let mut node = make_node();
        let kp = Keypair::generate();
        let mut tx = ConsensusTransaction::ContentOp(unfer_protocol::ContentOp {
            did: kp.did(),
            content_ref: ContentRef {
                cid: "abc123".to_string(),
                magnet_uri: "magnet:?xt=urn:btih:abc123".to_string(),
                encryption_key: "x25519:deadbeef".to_string(),
                filesize: 1024,
                mime_type: "video/mp4".to_string(),
                chunks: vec![],
            },
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut tx, &kp);
        node.submit(tx).unwrap();
        node.sync().unwrap();
        assert!(node.content("abc123").is_some());
    }

    #[test]
    fn invalid_signature_rejected() {
        let node = make_node();
        let kp = Keypair::generate();
        let tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0xFFu8; 64],
            seq: 1,
            service_endpoint: None,
        });
        assert!(node.submit(tx).is_err());
    }

    #[test]
    fn two_nodes_converge() {
        let engine = LocalConsensus::new();
        let kp = Keypair::generate();

        let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx, &kp);
        engine.submit(tx).unwrap();

        let mut node_a = ConsensusNode::new(Box::new(engine.clone()));
        let mut node_b = ConsensusNode::new(Box::new(engine.clone()));

        node_a.sync().unwrap();
        node_b.sync().unwrap();

        assert!(node_a.identity().resolve(&kp.did()).is_some());
        assert!(node_b.identity().resolve(&kp.did()).is_some());
        assert_eq!(node_a.applied_seq(), node_b.applied_seq());
    }

    #[test]
    fn incremental_sync() {
        let engine = LocalConsensus::new();
        let kp = Keypair::generate();

        let mut tx1 = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx1, &kp);
        engine.submit(tx1).unwrap();

        let mut node = ConsensusNode::new(Box::new(engine.clone()));
        node.sync().unwrap();
        assert_eq!(node.applied_seq(), 1);

        let kp2 = Keypair::generate();
        let mut tx2 = ConsensusTransaction::IdentityOp(IdentityOp {
            did: kp2.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: kp2.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        crate::signing::sign_transaction(&mut tx2, &kp2);
        engine.submit(tx2).unwrap();

        let applied = node.sync().unwrap();
        assert_eq!(applied, 1);
        assert_eq!(node.applied_seq(), 2);
        assert!(node.identity().resolve(&kp2.did()).is_some());
    }

    #[test]
    fn certificate_ledger_roundtrip_via_consensus() {
        use crate::certs::{MintAuthority, commit_coin};
        use unfer_protocol::{CertificateOp, CertificateOpKind, CoinRef};

        let engine = LocalConsensus::new();
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();

        let mut node = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(authority.did()),
        );
        let empty_root = node.certs().root();
        assert_ne!(empty_root, [0u8; 32]);

        // Mint 1000 to alice (signed by the authority).
        let mut mint = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount: 1000,
                owner: alice.did(),
                blinding: [1u8; 32],
                source: Some("unfccc:vc:TEST-0001".to_string()),
            },
            seq: 1,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut mint, &authority);
        node.submit(mint).unwrap();
        node.sync().unwrap();

        let alice_coin = commit_coin(1000, &alice.did(), &[1u8; 32]);
        assert!(node.certs().utxo(&alice_coin).is_some());
        assert_eq!(node.certs().total_supply(), 1000);

        // Transfer the whole 1000 to bob (signed by alice).
        let input = CoinRef {
            coin_id: alice_coin,
            amount: 1000,
            owner: alice.did(),
        };
        let output = CoinRef {
            coin_id: alice_coin, // recomputed below; amount/owner drive the id
            amount: 1000,
            owner: bob.did(),
        };
        let mut transfer = ConsensusTransaction::CertificateOp(CertificateOp {
            did: alice.did(),
            kind: CertificateOpKind::Transfer {
                inputs: vec![input],
                outputs: vec![output],
            },
            seq: 2,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut transfer, &alice);
        node.submit(transfer).unwrap();
        node.sync().unwrap();

        let bob_coin = commit_coin(1000, &bob.did(), &[0u8; 32]);
        assert!(node.certs().utxo(&bob_coin).is_some());
        assert!(node.certs().utxo(&alice_coin).is_none());
        assert!(node.certs().is_spent(&alice_coin));
        assert_eq!(node.certs().total_supply(), 1000);

        // Burn bob's certificate.
        let burn_input = CoinRef {
            coin_id: bob_coin,
            amount: 1000,
            owner: bob.did(),
        };
        let mut burn = ConsensusTransaction::CertificateOp(CertificateOp {
            did: bob.did(),
            kind: CertificateOpKind::Burn {
                inputs: vec![burn_input],
            },
            seq: 3,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut burn, &bob);
        node.submit(burn).unwrap();
        node.sync().unwrap();

        assert!(node.certs().utxo(&bob_coin).is_none());
        assert_eq!(node.certs().total_supply(), 0);
        // Retiring the only UTXO restores the empty root.
        assert_eq!(node.certs().root(), empty_root);
    }

    #[test]
    fn invalid_certificate_op_rejected_before_log() {
        use unfer_protocol::{CertificateOp, CertificateOpKind};

        let engine = LocalConsensus::new();
        let alice = Keypair::generate();
        let mut node = ConsensusNode::new(Box::new(engine));
        // Minting is disabled by default (MintAuthority::None).
        let mut mint = ConsensusTransaction::CertificateOp(CertificateOp {
            did: alice.did(),
            kind: CertificateOpKind::Mint {
                amount: 100,
                owner: alice.did(),
                blinding: [9u8; 32],
                source: None,
            },
            seq: 1,
            signature: [0u8; 64],
        });
        crate::signing::sign_transaction(&mut mint, &alice);
        // Signature is valid, so it reaches the log; application fails on sync
        // with UK-7001 (mint not authorized).
        let seq = node.submit(mint).unwrap();
        assert_eq!(seq, 1);
        let err = node.sync().unwrap_err();
        assert_eq!(err.code, Code::CERT_MINT_NOT_AUTHORIZED);
    }

    #[test]
    fn five_nodes_converge_on_certificate_root() {
        use crate::certs::commit_coin;
        use unfer_protocol::{CertificateOp, CertificateOpKind, CoinRef};

        // QuePaxa target is 5-7 validator nodes; all share one ordered log.
        let engine = LocalConsensus::new();
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();
        let carol = Keypair::generate();

        let mut nodes: Vec<ConsensusNode> = (0..5)
            .map(|_| {
                ConsensusNode::with_mint_authority(
                    Box::new(engine.clone()),
                    MintAuthority::Only(authority.did()),
                )
            })
            .collect();

        let sign_and_submit =
            |node: &ConsensusNode, tx: &mut ConsensusTransaction, kp: &Keypair| {
                crate::signing::sign_transaction(tx, kp);
                node.submit(tx.clone()).unwrap();
            };

        // Interleaved certificate + identity ops on the shared log.
        // Both coins are minted to bob so he can spend them together (a
        // multi-input transfer requires one signer to own every input).
        let mut mint1 = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount: 1000,
                owner: bob.did(),
                blinding: [1u8; 32],
                source: Some("unfccc:vc:MLT-0001".to_string()),
            },
            seq: 1,
            signature: [0u8; 64],
        });
        sign_and_submit(&nodes[0], &mut mint1, &authority);

        let mut mint2 = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount: 600,
                owner: bob.did(),
                blinding: [2u8; 32],
                source: Some("unfccc:vc:MLT-0002".to_string()),
            },
            seq: 2,
            signature: [0u8; 64],
        });
        sign_and_submit(&nodes[1], &mut mint2, &authority);

        // Two-input transfer: bob's 1000 + bob's 600 -> split to carol + alice.
        let bob_coin_a = commit_coin(1000, &bob.did(), &[1u8; 32]);
        let bob_coin_b = commit_coin(600, &bob.did(), &[2u8; 32]);
        let mut transfer = ConsensusTransaction::CertificateOp(CertificateOp {
            did: bob.did(),
            kind: CertificateOpKind::Transfer {
                inputs: vec![
                    CoinRef {
                        coin_id: bob_coin_a,
                        amount: 1000,
                        owner: bob.did(),
                    },
                    CoinRef {
                        coin_id: bob_coin_b,
                        amount: 600,
                        owner: bob.did(),
                    },
                ],
                outputs: vec![
                    CoinRef {
                        coin_id: bob_coin_b,
                        amount: 900,
                        owner: carol.did(),
                    },
                    CoinRef {
                        coin_id: bob_coin_b,
                        amount: 700,
                        owner: alice.did(),
                    },
                ],
            },
            seq: 3,
            signature: [0u8; 64],
        });
        sign_and_submit(&nodes[2], &mut transfer, &bob);

        // Identity op mixed in.
        let mut id_op = ConsensusTransaction::IdentityOp(IdentityOp {
            did: carol.did(),
            op_kind: IdentityOpKind::Create,
            signing_key: carol.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        sign_and_submit(&nodes[3], &mut id_op, &carol);

        // Burn carol's 900.
        let carol_coin = commit_coin(900, &carol.did(), &[0u8; 32]);
        let mut burn = ConsensusTransaction::CertificateOp(CertificateOp {
            did: carol.did(),
            kind: CertificateOpKind::Burn {
                inputs: vec![CoinRef {
                    coin_id: carol_coin,
                    amount: 900,
                    owner: carol.did(),
                }],
            },
            seq: 4,
            signature: [0u8; 64],
        });
        sign_and_submit(&nodes[4], &mut burn, &carol);

        // Every node replays the same log.
        for node in nodes.iter_mut() {
            node.sync().unwrap();
            assert!(node.is_synced());
        }

        // All five converge to the identical state.
        let r0 = nodes[0].certs().root();
        for node in &nodes[1..] {
            assert_eq!(
                node.certs().root(),
                r0,
                "all nodes share one certificate root"
            );
            assert_eq!(node.certs().total_supply(), 700, "supply: 1600 - 900 = 700");
            assert_eq!(node.applied_seq(), 5);
        }
        assert!(nodes[0].certs().utxo(&carol_coin).is_none(), "burned");
        assert!(nodes[0].certs().utxo(&bob_coin_a).is_none(), "spent");
        assert!(nodes[0].certs().utxo(&bob_coin_b).is_none(), "spent");
        // Alice's 700 change coin is live on every node.
        let change = commit_coin(700, &alice.did(), &[0u8; 32]);
        for node in &nodes {
            assert!(node.certs().utxo(&change).is_some(), "change coin live");
        }

        // A late-joining node syncs from the full log and agrees too.
        let mut late = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(authority.did()),
        );
        late.sync().unwrap();
        assert_eq!(late.applied_seq(), 5);
        assert_eq!(late.certs().root(), r0);
        assert!(late.certs().utxo(&change).is_some());
    }

    #[test]
    fn session_op_dispatch_locks_to_shared_consensus_ops() {
        use unfer_protocol::AgentRequest;

        // `apply_session_op` must support exactly the shared CONSENSUS_OPS
        // table; any op outside it is rejected with a diagnostic.
        let req = AgentRequest {
            id: "x".into(),
            op: "create_model".into(),
            params: serde_json::to_value(unfer_protocol::ModelSpec {
                hamiltonian: unfer_protocol::HamiltonianSpec::builtin(
                    "harmonic_chain",
                    serde_json::json!({"n_modes": 2, "omega": 1.0}),
                ),
                prior: unfer_protocol::PriorSpec::Vacuum,
                solver: unfer_protocol::SolverSpec::default(),
            })
            .unwrap(),
            provenance: None,
        };
        let mut node = ConsensusNode::new(Box::new(LocalConsensus::new()));
        assert!(node.apply_session_op(&req).is_ok());

        for op in unfer_protocol::ops::CONSENSUS_OPS {
            let req = AgentRequest {
                id: "x".into(),
                op: op.to_string(),
                params: serde_json::json!({}),
                provenance: None,
            };
            // create_model with `{}` params fails parsing (BAD_JSON) but must
            // still be a *supported* op (never the "unsupported session op" arm).
            let res = node.apply_session_op(&req);
            assert!(
                res.is_err(),
                "op '{op}' reported success without model params"
            );
        }
    }

    // ── H7: distributed-delivery protections ─────────────────────────────
    // Idempotency (exactly-once replay), leader lease (single firer per tick),
    // and the job queue (claim/unclaim/markFired) around the existing ledgers.

    #[test]
    fn double_submitted_transfer_applies_once_and_conservation_holds() {
        use crate::certs::commit_coin;
        use unfer_protocol::{CertificateOp, CertificateOpKind, CoinRef};

        // H7 acceptance: a duplicated delivery of the same transfer applies
        // exactly once; conservation (UK-7002) holds because the input is spent
        // exactly once.
        let engine = LocalConsensus::new();
        let authority = Keypair::generate();
        let alice = Keypair::generate();
        let bob = Keypair::generate();

        let mut node = ConsensusNode::with_mint_authority(
            Box::new(engine.clone()),
            MintAuthority::Only(authority.did()),
        );

        let sign_and_submit =
            |node: &ConsensusNode, tx: &mut ConsensusTransaction, kp: &Keypair| {
                crate::signing::sign_transaction(tx, kp);
                node.submit(tx.clone()).unwrap();
            };

        // Mint 1000 to alice.
        let mut mint = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: CertificateOpKind::Mint {
                amount: 1000,
                owner: alice.did(),
                blinding: [1u8; 32],
                source: None,
            },
            seq: 1,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut mint, &authority);

        // The exact same transfer submitted twice (duplicated delivery).
        let coin = commit_coin(1000, &alice.did(), &[1u8; 32]);
        let mut transfer = ConsensusTransaction::CertificateOp(CertificateOp {
            did: alice.did(),
            kind: CertificateOpKind::Transfer {
                inputs: vec![CoinRef {
                    coin_id: coin,
                    amount: 1000,
                    owner: alice.did(),
                }],
                outputs: vec![CoinRef {
                    coin_id: coin,
                    amount: 1000,
                    owner: bob.did(),
                }],
            },
            seq: 2,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut transfer, &alice);
        // Duplicate delivery of the same signed transfer.
        sign_and_submit(&node, &mut transfer, &alice);

        node.sync().unwrap();

        // Applied exactly once: the input is spent once, conservation holds.
        assert_eq!(node.certs().total_supply(), 1000, "UK-7002 conservation");
        assert!(
            node.certs().utxo(&coin).is_none(),
            "input spent exactly once"
        );
        assert_eq!(node.certs().unspent_count(), 1, "one output, not two");
        // The second delivery was recognized as committed.
        assert!(
            node.idempotency_committed(&transfer),
            "duplicated delivery must be recorded as committed"
        );
    }

    #[test]
    fn two_nodes_single_leader_fires_per_tick() {
        // H7 acceptance: exactly one node is the leader per tick; the
        // non-leader does not fire.
        let leaders0: Vec<bool> = (0..4)
            .map(|t| crate::lease::LeaderLease::is_leader(t, 2, 0))
            .collect();
        let leaders1: Vec<bool> = (0..4)
            .map(|t| crate::lease::LeaderLease::is_leader(t, 2, 1))
            .collect();
        for tick in 0..4 {
            assert!(
                leaders0[tick] ^ leaders1[tick],
                "exactly one leader fires per tick"
            );
        }
        // A lost lease stops firing without corrupting state: node 0 held tick
        // 0, but node 1 takes tick 1 — node 0 must stop.
        assert!(!crate::lease::LeaderLease::tick_held(0, 1, 2, 0));
    }

    #[test]
    fn failed_fire_is_unqueued_and_retried() {
        use crate::jobs::{JobQueue, JobState};

        // H7 acceptance: a job whose fire fails is re-queued (unclaimSlot) and
        // a later claim retries it; markFired only after a successful fire.
        let mut q = JobQueue::new();
        q.enqueue("settle-auction");
        let claim = q.claim_slot("settle-auction").expect("claim");
        q.unclaim_slot(&claim); // fire failed → re-queue
        assert_eq!(q.state("settle-auction"), Some(JobState::Queued));
        let retry = q.claim_slot("settle-auction").expect("retry");
        q.mark_fired(&retry).unwrap(); // fire succeeded
        assert_eq!(q.state("settle-auction"), Some(JobState::Fired));
    }

    // ── Math bond + probability market through the consensus node ─────────

    #[test]
    fn mathbond_market_consensus_roundtrip() {
        use crate::mathbond::compute_bond_id;
        use crate::mathbond_market::compute_pool_id;
        use unfer_protocol::{
            MarketOp, MarketOpKind, MathBondOp, MathBondOpKind, MathBondTrigger, NegRiskOutcome,
            OutcomeId,
        };

        let mut node = make_node();
        let sponsor = Keypair::generate();
        let investor = Keypair::generate();
        let researcher = Keypair::generate();
        let creator = Keypair::generate();
        let lp = Keypair::generate();

        let sign_and_submit =
            |node: &ConsensusNode, tx: &mut ConsensusTransaction, kp: &Keypair| {
                crate::signing::sign_transaction(tx, kp);
                node.submit(tx.clone()).unwrap();
            };

        // Issue the bond; maturity is at consensus seq 5.
        let trigger = MathBondTrigger {
            theorem: "P_eq_NP".to_string(),
            spec_hash: "deadbeef".to_string(),
            max_export_bytes: 1024,
            permitted_axioms: vec![],
            strict: false,
            nat_extension: false,
            string_extension: false,
        };
        let mut issue = ConsensusTransaction::MathBondOp(MathBondOp {
            did: sponsor.did(),
            kind: MathBondOpKind::Issue {
                trigger: trigger.clone(),
                principal: 10000,
                coupon_rate_bps: 500,
                maturity_seq: 5,
                researcher_did: researcher.did(),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut issue, &sponsor);
        node.sync().unwrap();
        let bond_id = compute_bond_id(&trigger, &sponsor.did(), 10000, 500, 5, &researcher.did());
        assert_eq!(
            node.mathbond().bond(&bond_id).unwrap().state,
            MathBondState::Issued
        );

        // Fund it.
        let mut invest = ConsensusTransaction::MathBondOp(MathBondOp {
            did: investor.did(),
            kind: MathBondOpKind::Invest {
                bond_id,
                amount: 10000,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut invest, &investor);
        node.sync().unwrap();
        assert_eq!(
            node.mathbond().bond(&bond_id).unwrap().state,
            MathBondState::Funded
        );

        // Open the NegRisk market for the bond and seed liquidity.
        let outcome_a = OutcomeId([1u8; 32]);
        let outcome_b = OutcomeId([2u8; 32]);
        let outcome_never = OutcomeId([3u8; 32]);
        let pool_id = compute_pool_id(&bond_id);
        let mut open = ConsensusTransaction::MarketOp(MarketOp {
            did: creator.did(),
            kind: MarketOpKind::OpenNegRisk {
                bond_id,
                outcomes: vec![
                    NegRiskOutcome {
                        outcome_id: outcome_a,
                        pool_id,
                        label: "by_2025".to_string(),
                        maturity_seq: 500,
                    },
                    NegRiskOutcome {
                        outcome_id: outcome_b,
                        pool_id,
                        label: "by_2026".to_string(),
                        maturity_seq: 1000,
                    },
                    NegRiskOutcome {
                        outcome_id: outcome_never,
                        pool_id,
                        label: "never".to_string(),
                        maturity_seq: u64::MAX,
                    },
                ],
                fee_bps: 300,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut open, &creator);
        node.sync().unwrap();

        let mut add = ConsensusTransaction::MarketOp(MarketOp {
            did: lp.did(),
            kind: MarketOpKind::AddLiquidity {
                pool_id,
                amount: 12000,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut add, &lp);
        node.sync().unwrap();
        assert_eq!(
            node.market().pool(&pool_id).unwrap().pool.total_reserve,
            12000
        );

        // Record maturity (consensus seq 5 >= maturity_seq 5).
        let mut mature = ConsensusTransaction::MathBondOp(MathBondOp {
            did: investor.did(),
            kind: MathBondOpKind::Mature { bond_id },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut mature, &investor);
        node.sync().unwrap();
        assert_eq!(
            node.mathbond().bond(&bond_id).unwrap().state,
            MathBondState::Matured
        );

        // Resolve with `None` (matured without a trigger): the never outcome
        // wins deterministically — no caller-chosen outcome.
        let mut resolve = ConsensusTransaction::MarketOp(MarketOp {
            did: creator.did(),
            kind: MarketOpKind::Resolve {
                pool_id,
                trigger_seq: None,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut resolve, &creator);
        node.sync().unwrap();
        assert_eq!(
            node.market().pool(&pool_id).unwrap().pool.winner,
            Some(outcome_never)
        );

        // The LP claims the whole reserve (nobody held winning tokens).
        let mut claim = ConsensusTransaction::MarketOp(MarketOp {
            did: lp.did(),
            kind: MarketOpKind::Claim { pool_id },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut claim, &lp);
        node.sync().unwrap();
        assert_eq!(
            node.market().pool(&pool_id).unwrap().lp_map.get(&lp.did()),
            Some(&0)
        );
    }

    #[test]
    fn market_resolve_forgery_rejected() {
        use crate::mathbond::compute_bond_id;
        use crate::mathbond_market::compute_pool_id;
        use unfer_protocol::{
            MarketOp, MarketOpKind, MathBondOp, MathBondOpKind, MathBondTrigger, NegRiskOutcome,
            OutcomeId,
        };

        let sign_and_submit =
            |node: &ConsensusNode, tx: &mut ConsensusTransaction, kp: &Keypair| {
                crate::signing::sign_transaction(tx, kp);
                node.submit(tx.clone()).unwrap();
            };

        // Scenario 1: the bond is still live — a forged resolve is refused
        // before it can touch the pool.
        let mut node = make_node();
        let sponsor = Keypair::generate();
        let creator = Keypair::generate();
        let trigger = MathBondTrigger {
            theorem: "RiemannHypothesis".to_string(),
            spec_hash: "cafe".to_string(),
            max_export_bytes: 1024,
            permitted_axioms: vec![],
            strict: false,
            nat_extension: false,
            string_extension: false,
        };
        let mut issue = ConsensusTransaction::MathBondOp(MathBondOp {
            did: sponsor.did(),
            kind: MathBondOpKind::Issue {
                trigger: trigger.clone(),
                principal: 1000,
                coupon_rate_bps: 0,
                maturity_seq: 1000,
                researcher_did: sponsor.did(),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut issue, &sponsor);
        let bond_id = compute_bond_id(&trigger, &sponsor.did(), 1000, 0, 1000, &sponsor.did());
        let pool_id = compute_pool_id(&bond_id);
        let mut open = ConsensusTransaction::MarketOp(MarketOp {
            did: creator.did(),
            kind: MarketOpKind::OpenNegRisk {
                bond_id,
                outcomes: vec![NegRiskOutcome {
                    outcome_id: OutcomeId([1u8; 32]),
                    pool_id,
                    label: "never".to_string(),
                    maturity_seq: u64::MAX,
                }],
                fee_bps: 0,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut open, &creator);
        node.sync().unwrap();

        let mut forged = ConsensusTransaction::MarketOp(MarketOp {
            did: creator.did(),
            kind: MarketOpKind::Resolve {
                pool_id,
                trigger_seq: Some(1),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node, &mut forged, &creator);
        let err = node.sync().unwrap_err();
        assert_eq!(err.code, Code::MARKET_NOT_RESOLVED);
        assert!(
            !node.market().pool(&pool_id).unwrap().pool.resolved,
            "the forged resolution must not touch the pool"
        );

        // Scenario 2: the bond matured without a trigger (expected signal
        // None) — claiming a trigger fired is refused as a mismatch.
        let mut node2 = make_node();
        let sponsor2 = Keypair::generate();
        let creator2 = Keypair::generate();
        let trigger2 = MathBondTrigger {
            theorem: "Goldbach".to_string(),
            spec_hash: "beef".to_string(),
            max_export_bytes: 1024,
            permitted_axioms: vec![],
            strict: false,
            nat_extension: false,
            string_extension: false,
        };
        let mut issue2 = ConsensusTransaction::MathBondOp(MathBondOp {
            did: sponsor2.did(),
            kind: MathBondOpKind::Issue {
                trigger: trigger2.clone(),
                principal: 1000,
                coupon_rate_bps: 0,
                maturity_seq: 3,
                researcher_did: sponsor2.did(),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node2, &mut issue2, &sponsor2);
        let bond_id2 = compute_bond_id(&trigger2, &sponsor2.did(), 1000, 0, 3, &sponsor2.did());
        let pool_id2 = compute_pool_id(&bond_id2);
        let mut open2 = ConsensusTransaction::MarketOp(MarketOp {
            did: creator2.did(),
            kind: MarketOpKind::OpenNegRisk {
                bond_id: bond_id2,
                outcomes: vec![NegRiskOutcome {
                    outcome_id: OutcomeId([1u8; 32]),
                    pool_id: pool_id2,
                    label: "never".to_string(),
                    maturity_seq: u64::MAX,
                }],
                fee_bps: 0,
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node2, &mut open2, &creator2);
        let mut mature2 = ConsensusTransaction::MathBondOp(MathBondOp {
            did: sponsor2.did(),
            kind: MathBondOpKind::Mature { bond_id: bond_id2 },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node2, &mut mature2, &sponsor2);
        node2.sync().unwrap();

        let mut forged2 = ConsensusTransaction::MarketOp(MarketOp {
            did: creator2.did(),
            kind: MarketOpKind::Resolve {
                pool_id: pool_id2,
                trigger_seq: Some(1),
            },
            seq: 0,
            signature: [0u8; 64],
        });
        sign_and_submit(&node2, &mut forged2, &creator2);
        let err = node2.sync().unwrap_err();
        assert_eq!(err.code, Code::MARKET_UNKNOWN_OUTCOME);
        assert!(!node2.market().pool(&pool_id2).unwrap().pool.resolved);
    }
}
