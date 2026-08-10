//! Phase 6 testnet: `ConsensusNode` running on a live rust-quepaxa cluster
//! (2-live-of-3 mutually-authenticated loopback nodes). Requires the
//! `network` feature (see `[[test]] required-features` in Cargo.toml).
//!
//! Genesis certificate ops and an identity op are proposed to the cluster
//! through the two live nodes; after QuePaxa commits every slot, each node
//! replays the replicated log into its own application state and all peers
//! converge on the identical certificate root.

use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use rust_quepaxa::network::TlsIdentity;
use tokio::time::timeout;

use unfer_consensus::certs::{commit_coin, MintAuthority};
use unfer_consensus::net::NetworkCluster;
use unfer_consensus::node::ConsensusNode;
use unfer_consensus::signing::{sign_transaction, Keypair};
use unfer_protocol::{
    CertificateOp, CertificateOpKind, CoinRef, ConsensusTransaction, IdentityOp, IdentityOpKind,
};

fn identity() -> TlsIdentity {
    let CertifiedKey { cert, key_pair } =
        generate_simple_self_signed(["localhost".to_owned()]).unwrap();
    TlsIdentity {
        certificate_chain_der: vec![cert.der().to_vec()],
        private_key_pkcs8_der: key_pair.serialize_der(),
    }
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase6_testnet_backs_consensus_node_with_certificate_ops() {
    let identities = (0..4).map(|_| identity()).collect::<Vec<_>>();
    let cluster = NetworkCluster::begin(identities).await.unwrap();

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
    let mut identity_op = ConsensusTransaction::IdentityOp(IdentityOp {
        did: bob.did(),
        op_kind: IdentityOpKind::Create,
        signing_key: bob.public_key(),
        signature: [0u8; 64],
        seq: 1,
        service_endpoint: None,
    });
    sign_transaction(&mut identity_op, &bob);
    nodes[1].submit(identity_op).unwrap();

    // Op 3: alice spends her whole certificate to bob.
    let alice_coin = commit_coin(1000, &alice.did(), &[1u8; 32]);
    let mut transfer = ConsensusTransaction::CertificateOp(CertificateOp {
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
        seq: 2,
        signature: [0u8; 64],
    });
    sign_transaction(&mut transfer, &alice);
    nodes[1].submit(transfer).unwrap();

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
}