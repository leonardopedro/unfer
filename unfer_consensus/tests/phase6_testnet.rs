//! Phase 6 testnet: `ConsensusNode` running on a live rust-quepaxa cluster
//! (2-live-of-3 mutually-authenticated loopback nodes). Requires the
//! `network` feature (see `[[test]] required-features` in Cargo.toml).
//!
//! Genesis certificate ops and an identity op are proposed to the cluster
//! through the two live nodes; after QuePaxa commits every slot, each node
//! replays the replicated log into its own application state and all peers
//! converge on the identical certificate root. Node state is durable: a
//! second test shuts the whole cluster down, restarts it on the same TLS
//! identities, state directory, and socket addresses, and verifies committed
//! work survives plus fresh ops keep committing.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rcgen::{generate_simple_self_signed, CertifiedKey};
use rust_quepaxa::network::TlsIdentity;
use tokio::time::timeout;

use unfer_consensus::certs::{commit_coin, MintAuthority};
use unfer_consensus::engine::ConsensusEngine;
use unfer_consensus::net::NetworkCluster;
use unfer_consensus::node::ConsensusNode;
use unfer_consensus::signing::{sign_transaction, Keypair};
use unfer_protocol::{
    CertificateOp, CertificateOpKind, CoinRef, ConsensusTransaction, IdentityOp, IdentityOpKind,
    MintRequest,
};

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
    let dir = std::env::temp_dir().join(format!(
        "unfer-{name}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn mint_op(authority: &Keypair, amount: u64, owner: &Keypair, blinding: [u8; 32], seq: u64, source: &str) -> ConsensusTransaction {
    let mut tx = ConsensusTransaction::CertificateOp(CertificateOp {
        did: authority.did(),
        kind: CertificateOpKind::Mint {
            amount,
            owner: owner.did(),
            blinding,
            source: Some(source.to_string()),
        },
        seq,
        signature: [0u8; 64],
    });
    sign_transaction(&mut tx, authority);
    tx
}

fn identity_op(bob: &Keypair, seq: u64) -> ConsensusTransaction {
    let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
        did: bob.did(),
        op_kind: IdentityOpKind::Create,
        signing_key: bob.public_key(),
        signature: [0u8; 64],
        seq,
        service_endpoint: None,
    });
    sign_transaction(&mut tx, bob);
    tx
}

fn transfer_op(alice: &Keypair, bob: &Keypair, blinding: [u8; 32], seq: u64) -> ConsensusTransaction {
    let alice_coin = commit_coin(1000, &alice.did(), &blinding);
    let mut tx = ConsensusTransaction::CertificateOp(CertificateOp {
        did: alice.did(),
        kind: CertificateOpKind::Transfer {
            inputs: vec![CoinRef {
                coin_id: alice_coin,
                amount: 1000,
                owner: alice.did(),
            }],
            outputs: vec![CoinRef {
                coin_id: alice_coin,
                amount: 1000,
                owner: bob.did(),
            }],
        },
        seq,
        signature: [0u8; 64],
    });
    sign_transaction(&mut tx, alice);
    tx
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase6_testnet_backs_consensus_node_with_certificate_ops() {
    let identities = (0..4).map(|_| identity()).collect::<Vec<_>>();
    let state_dir = temp_state_dir("phase6-live");
    let cluster = NetworkCluster::begin(identities, &state_dir).await.unwrap();

    let authority = Keypair::generate();
    let alice = Keypair::generate();
    let bob = Keypair::generate();

    let mut nodes: Vec<ConsensusNode> = (0..cluster.nodes.len())
        .map(|_| {
            ConsensusNode::with_mint_authority(
                Box::new(cluster.engine()),
                MintAuthority::Only(authority.did()),
            )
        })
        .collect();
    assert_eq!(nodes.len(), 2);

    // Op 1: the mint authority plates one carbon certificate to alice.
    let mint = mint_op(&authority, 1000, &alice, [1u8; 32], 1, "unfccc:vc:TESTNET-0001");
    nodes[0].submit(mint).unwrap();

    // Op 2: an identity registration, interleaved on the second live node.
    nodes[1].submit(identity_op(&bob, 1)).unwrap();

    // Op 3: alice spends her whole certificate to bob.
    nodes[1].submit(transfer_op(&alice, &bob, [1u8; 32], 2)).unwrap();

    // Let QuePaxa commit all three slots.
    timeout(Duration::from_secs(90), cluster.wait_committed(3))
        .await
        .expect("cluster commits all three operations in time");

    // Every live node replays the replicated log into its own state machine.
    for node in nodes.iter_mut() {
        node.sync().unwrap();
    }

    let root0 = nodes[0].certs().root();
    for node in &nodes[1..] {
        assert_eq!(
            node.certs().root(),
            root0,
            "all live nodes converge on one certificate root"
        );
        assert_eq!(node.certs().total_supply(), 1000);
        assert_eq!(node.applied_seq(), 3);
    }

    // Certificate state is consistent with the ops above.
    let alice_coin = commit_coin(1000, &alice.did(), &[1u8; 32]);
    assert!(nodes[0].certs().utxo(&alice_coin).is_none(), "spent");
    let bob_coin = commit_coin(1000, &bob.did(), &[0u8; 32]);
    assert!(
        nodes[0].certs().utxo(&bob_coin).is_some(),
        "bob owns the transferred certificate"
    );
    assert!(
        nodes[0].identity().resolve(&bob.did()).is_some(),
        "identity op replicated to the network log"
    );

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase6_nodes_survive_full_cluster_restart() {
    let identities = (0..4).map(|_| identity()).collect::<Vec<_>>();
    let state_dir = temp_state_dir("phase6-restart");

    let authority = Keypair::generate();
    let alice = Keypair::generate();
    let bob = Keypair::generate();

    // ---- Phase A: commit a mint + identity + transfer, then stop everything.
    let (root0, addresses) = {
        let cluster = NetworkCluster::begin(identities.clone(), &state_dir).await.unwrap();
        let addresses = cluster.addresses.clone();
        let mut nodes: Vec<ConsensusNode> = (0..cluster.nodes.len())
            .map(|_| {
                ConsensusNode::with_mint_authority(
                    Box::new(cluster.engine()),
                    MintAuthority::Only(authority.did()),
                )
            })
            .collect();

        nodes[0].submit(mint_op(&authority, 1000, &alice, [1u8; 32], 1, "unfccc:vc:TESTNET-0001")).unwrap();
        nodes[1].submit(identity_op(&bob, 1)).unwrap();
        nodes[1].submit(transfer_op(&alice, &bob, [1u8; 32], 2)).unwrap();

        timeout(Duration::from_secs(90), cluster.wait_committed(3))
            .await
            .expect("cluster commits before restart");
        for node in nodes.iter_mut() {
            node.sync().unwrap();
        }
        let root0 = nodes[0].certs().root();
        assert_eq!(nodes[0].applied_seq(), 3);
        assert_eq!(nodes[0].certs().total_supply(), 1000);

        // Graceful stop: every node task returns and releases its socket.
        cluster.shutdown().await;
        (root0, addresses)
    };

    // ---- Phase B: restart on the same identities, state dir, and addresses.
    let cluster = NetworkCluster::resume(identities, &state_dir, addresses)
        .await
        .unwrap();
    assert_eq!(
        cluster.ledger.committed(),
        3,
        "committed log reloaded from the durable ledger"
    );

    let mut nodes: Vec<ConsensusNode> = (0..cluster.nodes.len())
        .map(|_| {
            ConsensusNode::with_mint_authority(
                Box::new(cluster.engine()),
                MintAuthority::Only(authority.did()),
            )
        })
        .collect();

    // The replayed nodes converge on the same root as before the restart.
    for node in nodes.iter_mut() {
        node.sync().unwrap();
    }
    assert_eq!(nodes[0].certs().root(), root0, "root survives restart");
    assert_eq!(nodes[0].certs().total_supply(), 1000);
    assert_eq!(nodes[0].applied_seq(), 3);

    // ---- Phase C: fresh slots keep committing on top of the restarted log.
    nodes[0]
        .submit(mint_op(&authority, 500, &alice, [2u8; 32], 2, "unfccc:vc:TESTNET-0002"))
        .unwrap();
    timeout(Duration::from_secs(90), cluster.wait_committed(4))
        .await
        .expect("cluster commits after restart");

    for node in nodes.iter_mut() {
        node.sync().unwrap();
    }
    for node in &nodes[1..] {
        assert_eq!(node.certs().root(), nodes[0].certs().root());
    }
    assert_eq!(nodes[0].applied_seq(), 4, "both phases in one log");
    assert_eq!(nodes[0].certs().total_supply(), 1500);
    let alice_coin_2 = commit_coin(500, &alice.did(), &[2u8; 32]);
    assert!(
        nodes[0].certs().utxo(&alice_coin_2).is_some(),
        "post-restart mint lands in the ledger"
    );

    cluster.shutdown().await;
    let _ = std::fs::remove_dir_all(&state_dir);
}

/// A UNFCCC oracle client member: it holds the registry of verified
/// cancellation VCs it has zk-TLS proven, mints a certificate per verified VC
/// through the cluster (the Phase 3 `MintRequest` contract), and then audits
/// the replicated log for provenance — flagging any mint whose backing VC it
/// never verified.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase6_oracle_client_audits_provenance_from_the_cluster() {
    let identities = (0..4).map(|_| identity()).collect::<Vec<_>>();
    let state_dir = temp_state_dir("phase6-oracle");
    let cluster = NetworkCluster::begin(identities, &state_dir).await.unwrap();

    let authority = Keypair::generate();
    let alice = Keypair::generate();

    let mut nodes: Vec<ConsensusNode> = (0..cluster.nodes.len())
        .map(|_| {
            ConsensusNode::with_mint_authority(
                Box::new(cluster.engine()),
                MintAuthority::Only(authority.did()),
            )
        })
        .collect();

    // The VCs the oracle has proven on the UN platform.
    let verified = BTreeSet::from([
        "unfccc:vc:VC-0001".to_string(),
        "unfccc:vc:VC-0002".to_string(),
    ]);

    // Mint one certificate per verified VC (Phase 3 contract), plus a foreign
    // mint whose source the oracle never verified (VC-9999).
    let mut pending = Vec::new();
    for (i, source) in verified
        .iter()
        .chain(std::iter::once(&"unfccc:vc:VC-9999".to_string()))
        .enumerate()
    {
        let request = MintRequest {
            owner: alice.did(),
            amount: 1000,
            source: source.clone(),
            blinding: None,
        };
        assert!(request.validate_source().is_ok(), "well-formed oracle record");
        let mut tx = ConsensusTransaction::CertificateOp(CertificateOp {
            did: authority.did(),
            kind: request.to_mint_kind(),
            seq: i as u64 + 1,
            signature: [0u8; 64],
        });
        sign_transaction(&mut tx, &authority);
        pending.push(tx);
    }

    for (i, tx) in pending.iter().enumerate() {
        nodes[i % nodes.len()].submit(tx.clone()).unwrap();
    }

    timeout(Duration::from_secs(90), cluster.wait_committed(3))
        .await
        .expect("cluster commits all three mints in time");
    for node in nodes.iter_mut() {
        node.sync().unwrap();
    }
    for node in &nodes[1..] {
        assert_eq!(node.certs().root(), nodes[0].certs().root());
    }
    assert_eq!(nodes[0].certs().total_supply(), 3000);

    // The oracle member replays the replicated log and audits provenance.
    let engine = cluster.engine();
    let mut oracle_member = ConsensusNode::with_mint_authority(
        Box::new(engine.clone()),
        MintAuthority::Only(authority.did()),
    );
    oracle_member.sync().unwrap();
    assert_eq!(oracle_member.applied_seq(), 3);

    let unverified = engine
        .get_log(0)
        .iter()
        .filter_map(|(_, tx)| match tx {
            ConsensusTransaction::CertificateOp(op) => match &op.kind {
                CertificateOpKind::Mint {
                    source: Some(source), ..
                } => Some(source.clone()),
                _ => None,
            },
            _ => None,
        })
        .filter(|source| !verified.contains(source))
        .collect::<Vec<_>>();
    assert_eq!(
        unverified,
        vec!["unfccc:vc:VC-9999".to_string()],
        "oracle flags the foreign mint, verifies the genuine ones"
    );

    let _ = std::fs::remove_dir_all(&state_dir);
}
