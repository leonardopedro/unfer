//! Phase 6 testnet: a GNU Taler exchange's lifecycle replicated through the
//! QuePaxa cluster. Requires the `network` feature (see `[[test]]` in
//! Cargo.toml).
//!
//! The exchange drives its private seam (reserve, fiat peg-in, e-coin withdraw,
//! merchant deposit, fiat peg-out) while emitting signed certificate ops. Every
//! op is proposed to the cluster through the two live nodes; after QuePaxa
//! commits them, each node replays the replicated log and must land on the
//! exchange's own mirror certificate root — the exchange is a member of the
//! same replicated ledger, and its conservation audit holds throughout.

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rcgen::{CertifiedKey, generate_simple_self_signed};
use rust_quepaxa::network::TlsIdentity;
use tokio::time::timeout;

use unfer_consensus::certs::MintAuthority;
use unfer_consensus::net::NetworkCluster;
use unfer_consensus::node::ConsensusNode;
use unfer_consensus::signing::Keypair;
use unfer_protocol::CoinRef;
use unfer_taler::TalerExchange;
use unfer_taler::wire::{SimulatedWireGateway, WireGateway};

fn identity() -> TlsIdentity {
    let CertifiedKey { cert, key_pair } =
        generate_simple_self_signed(["localhost".to_owned()]).unwrap();
    TlsIdentity {
        certificate_chain_der: vec![cert.der().to_vec()],
        private_key_pkcs8_der: key_pair.serialize_der(),
    }
}

fn temp_state_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("unfer-{name}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn taler_exchange_lifecycle_replicates_through_the_cluster() {
    let identities = (0..4).map(|_| identity()).collect::<Vec<_>>();
    let state_dir = temp_state_dir("phase6-taler");
    let cluster = NetworkCluster::begin(identities, &state_dir).await.unwrap();

    let treasury = Keypair::generate();
    let alice = Keypair::generate();
    let bob = Keypair::generate();
    let treasury_did = treasury.did();

    // The live cluster nodes share the exchange's treasury as mint authority.
    let mut nodes: Vec<ConsensusNode> = (0..cluster.nodes.len())
        .map(|_| {
            ConsensusNode::with_mint_authority(
                Box::new(cluster.engine()),
                MintAuthority::Only(treasury_did.clone()),
            )
        })
        .collect();

    // The exchange: mirror ledger + emitted signed ops, all fiat-backed.
    let mut ex = TalerExchange::new(treasury, Box::new(SimulatedWireGateway::new()));
    for value in [100u64, 500, 1000] {
        ex.issue_denomination(value, u64::MAX);
    }

    // -- lifecycle -----------------------------------------------------------
    let reserve = [1u8; 32];
    ex.open_reserve(reserve, &alice.did());

    let mut gw = SimulatedWireGateway::new();
    let wire = gw.prepare_transfer("unfer-bank", 1000).unwrap();
    gw.confirm(&wire.wire_id).unwrap();
    let wire = gw.get(&wire.wire_id).unwrap().clone();
    ex.peg_in(reserve, &wire).unwrap();

    // Two withdraws of the two live denominations, both retired to the merchant.
    for _ in 0..2 {
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
    }

    assert_eq!(ex.ledger().total_supply(), 0, "all e-coins retired");
    assert_eq!(ex.merchant_balance(&bob.did()), 1000);
    assert!(ex.audit().is_ok());

    // Merchant redeems part of the balance back to fiat; the rest stays inside
    // the seam, fully backed.
    let peg = ex
        .peg_out(&bob.did(), 400, "DE99 0000 0000 1234 5678 90")
        .unwrap();
    ex.confirm_peg_out(&peg.wire.wire_id).unwrap();
    assert_eq!(ex.merchant_balance(&bob.did()), 600);
    assert!(ex.audit().is_ok());
    assert_eq!(
        ex.fiat_in() - ex.fiat_out(),
        600,
        "600 still backed inside the seam"
    );

    // -- replicate through the cluster ---------------------------------------
    let ops = ex.ops().to_vec();
    assert!(!ops.is_empty());
    for (i, op) in ops.iter().enumerate() {
        nodes[i % nodes.len()].submit(op.clone()).unwrap();
    }

    timeout(
        Duration::from_secs(90),
        cluster.wait_committed(ops.len() as u64),
    )
    .await
    .expect("cluster commits every exchange op in time");

    for node in nodes.iter_mut() {
        node.sync().unwrap();
    }

    // Every node reproduces the exchange's mirror certificate root exactly.
    for node in &nodes {
        assert_eq!(
            node.certs().root(),
            ex.ledger().root(),
            "cluster nodes land on the exchange's certificate root"
        );
        assert_eq!(node.certs().total_supply(), ex.ledger().total_supply());
        assert_eq!(node.applied_seq(), ops.len() as u64);
    }

    // The exchange's conservation audit still holds after full replication.
    assert!(ex.audit().is_ok());

    let _ = std::fs::remove_dir_all(&state_dir);
}
