//! D06: durable logical request identity, at the node's admission point.
//!
//! These tests drive [`ConsensusNode::submit_request`], not the book directly:
//! the point of a request id is that a *client* retrying after a timeout gets the
//! same answer as the first attempt without the action happening twice.

use std::time::Duration;

use unfer_consensus::idempotency::{payload_digest, RequestId};
use unfer_consensus::{sign_transaction, ConsensusNode, Keypair, LocalConsensus, MintAuthority};
use unfer_protocol::submission::Watermark;
use unfer_protocol::{CertificateOp, CertificateOpKind, Code, ConsensusTransaction};

fn mint_tx(authority: &Keypair, owner: &str, amount: u64, seq: u64) -> ConsensusTransaction {
    let mut tx = ConsensusTransaction::CertificateOp(CertificateOp {
        did: authority.did(),
        kind: CertificateOpKind::Mint {
            amount,
            owner: owner.to_string(),
            blinding: [seq as u8; 32],
            source: None,
        },
        seq,
        signature: [0u8; 64],
    });
    sign_transaction(&mut tx, authority);
    tx
}

fn request(sequence: u64) -> RequestId {
    RequestId::new("federation-main", "did:unfer:alice", 1, sequence)
}

fn authorized_node(authority: &Keypair) -> ConsensusNode {
    ConsensusNode::with_mint_authority(
        Box::new(LocalConsensus::new()),
        MintAuthority::Only(authority.did()),
    )
}

#[test]
fn a_retried_request_is_submitted_once_and_lands_in_one_slot() {
    let authority = Keypair::generate();
    let mut node = authorized_node(&authority);
    let request = request(1);
    let tx = mint_tx(&authority, "did:unfer:bob", 100, 1);

    let first = node.submit_request(&request, tx.clone()).unwrap();
    let settled = node.settle(first, Duration::from_millis(100));
    assert!(settled.is_committed(), "the mint must commit locally");

    // The client times out and retries: same id, same bytes.
    let retried = node.submit_request(&request, tx.clone()).unwrap();
    assert_eq!(retried, first, "a retry must reuse the original submission");
    let again = node.settle(retried, Duration::from_millis(100));
    assert_eq!(again, settled, "and it must report the same settled result");

    assert_eq!(
        node.watermark(),
        Watermark(1),
        "the retry must not append a second entry"
    );
    assert_eq!(
        node.admissions().admission(&request).map(|a| a.digest),
        Some(payload_digest(&tx).unwrap())
    );
}

#[test]
fn a_conflicting_retry_is_refused_and_applies_nothing() {
    let authority = Keypair::generate();
    let mut node = authorized_node(&authority);
    let request = request(1);
    let original = mint_tx(&authority, "did:unfer:bob", 100, 1);
    let equivocation = mint_tx(&authority, "did:unfer:bob", 999, 2);

    let submission = node.submit_request(&request, original).unwrap();
    node.settle(submission, Duration::from_millis(100));

    let error = node
        .submit_request(&request, equivocation)
        .expect_err("one request id names one action");
    assert_eq!(error.code, Code::REQUEST_CONFLICT);
    assert_eq!(
        node.watermark(),
        Watermark(1),
        "an equivocating retry must not reach the log"
    );
}

#[test]
fn a_settled_rejection_is_replayed_to_the_retrying_client() {
    // No mint authority: the mint is a deterministic, cached rejection.
    let authority = Keypair::generate();
    let mut node = ConsensusNode::new(Box::new(LocalConsensus::new()));
    let request = request(1);
    let tx = mint_tx(&authority, "did:unfer:bob", 100, 1);

    let submission = node.submit_request(&request, tx.clone()).unwrap();
    let first = node.settle(submission, Duration::from_millis(100));
    let refusal = first.refusal().expect("unauthorized mint is refused");
    assert_eq!(refusal.code, Code::CERT_MINT_NOT_AUTHORIZED);

    let retried = node
        .submit_request(&request, tx)
        .expect_err("a retried rejection is reported, not re-submitted");
    assert_eq!(retried.code, Code::CERT_MINT_NOT_AUTHORIZED);
    assert_eq!(
        node.watermark(),
        Watermark(1),
        "the retry applied nothing new"
    );
}

#[test]
fn different_request_ids_with_identical_bytes_are_distinct_actions() {
    let authority = Keypair::generate();
    let mut node = authorized_node(&authority);
    let tx = mint_tx(&authority, "did:unfer:bob", 100, 1);

    let a = node.submit_request(&request(1), tx.clone()).unwrap();
    let b = node.submit_request(&request(2), tx).unwrap();
    assert_ne!(a, b, "two ids are two actions, even with equal bytes");
    node.settle(a, Duration::from_millis(100));
    node.settle(b, Duration::from_millis(100));
    assert_eq!(
        node.watermark(),
        Watermark(2),
        "both actions are in the log"
    );
}

#[test]
fn a_request_below_the_acknowledged_floor_is_refused_after_restart_of_retention() {
    let authority = Keypair::generate();
    let mut node = authorized_node(&authority);
    for sequence in 1..=3 {
        let tx = mint_tx(&authority, "did:unfer:bob", 10, sequence);
        let submission = node.submit_request(&request(sequence), tx).unwrap();
        node.settle(submission, Duration::from_millis(100));
    }
    let dropped = node.acknowledge_requests(&request(3), 2);
    assert_eq!(dropped, 2, "the client acknowledged sequences 1 and 2");

    let old = mint_tx(&authority, "did:unfer:bob", 10, 1);
    let error = node
        .submit_request(&request(1), old)
        .expect_err("a forgotten request cannot be replayed safely");
    assert_eq!(error.code, Code::REQUEST_BELOW_RETENTION);
    assert_eq!(
        node.watermark(),
        Watermark(3),
        "the below-retention retry applied nothing"
    );
    // The retained window still answers for a request it remembers.
    assert!(node.admissions().admission(&request(3)).is_some());
}
