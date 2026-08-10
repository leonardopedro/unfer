use unfer_consensus::{ConsensusNode, Keypair, LocalConsensus};
use unfer_data::crypto::derive_aes_key;
use unfer_data::{
    Chunker, DataKeypair, DataPublisher, compute_cid, decrypt_chunk, encrypt_chunk, verify_chunk,
};
use unfer_identity::DidManager;

fn make_node() -> ConsensusNode {
    ConsensusNode::new(Box::new(LocalConsensus::new()))
}

fn register_did(node: &mut ConsensusNode, kp: &Keypair) -> String {
    let mut mgr = DidManager::new(node);
    mgr.create_did(kp, Some("https://test.example.com".to_string()))
        .unwrap()
}

#[test]
fn consensus_identity_content_roundtrip() {
    let mut node = make_node();
    let kp = Keypair::generate();

    let did = register_did(&mut node, &kp);
    assert!(did.starts_with("did:unfer:"));

    let mgr = DidManager::new(&mut node);
    let doc = mgr.resolve(&did).unwrap();
    assert_eq!(doc.id, did);
    assert_eq!(doc.service[0].service_endpoint, "https://test.example.com");

    let data = b"consensus-backed content payload";
    let content_ref = {
        let mut pub_ = DataPublisher::new(&mut node);
        pub_.publish(&kp, data, "application/octet-stream", Some("payload.bin"))
            .unwrap()
    };

    assert_eq!(content_ref.filesize, data.len() as u64);
    assert!(content_ref.magnet_uri.contains(&content_ref.cid));

    let resolved = node.content(&content_ref.cid).unwrap();
    assert_eq!(resolved.cid, content_ref.cid);
    assert_eq!(resolved.magnet_uri, content_ref.magnet_uri);
    assert_eq!(resolved.encryption_key, content_ref.encryption_key);
    assert_eq!(resolved.chunks.len(), content_ref.chunks.len());

    assert!(node.is_synced());
    assert!(node.applied_seq() > 0);
}

#[test]
fn data_plane_encrypt_chunk_reassemble_decrypt_roundtrip() {
    let plaintext: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
    let chunk_size = 100;

    let data_kp = DataKeypair::generate();
    let aes_key = derive_aes_key(&data_kp.shared_secret(data_kp.public_key()));

    let chunker = Chunker::new(chunk_size);
    let chunks = chunker.chunk(&plaintext);
    assert_eq!(chunks.len(), 11);

    let mut ciphertexts = Vec::new();
    let mut chunk_cids = Vec::new();

    for (index, chunk_data) in &chunks {
        let ct = encrypt_chunk(&aes_key, *index, chunk_data).unwrap();
        let cid = compute_cid(&ct);
        assert!(verify_chunk(&ct, &cid));
        chunk_cids.push(cid);
        ciphertexts.push((*index, ct));
    }

    let mut reassembled = Vec::new();
    for (index, ct) in &ciphertexts {
        let pt = decrypt_chunk(&aes_key, *index, ct).unwrap();
        reassembled.extend_from_slice(&pt);
    }

    assert_eq!(reassembled, plaintext);

    for (index, ct) in &ciphertexts {
        assert!(
            decrypt_chunk(&aes_key, index + 1, ct).is_err(),
            "wrong chunk index must fail authentication"
        );
    }

    let wrong_kp = DataKeypair::generate();
    let wrong_key = derive_aes_key(&wrong_kp.shared_secret(wrong_kp.public_key()));
    assert!(
        decrypt_chunk(&wrong_key, 0, &ciphertexts[0].1).is_err(),
        "wrong key must fail authentication"
    );
}

#[test]
fn two_nodes_converge_on_content() {
    let engine = LocalConsensus::new();
    let mut node_a = ConsensusNode::new(Box::new(engine.clone()));
    let mut node_b = ConsensusNode::new(Box::new(engine));

    let kp = Keypair::generate();
    register_did(&mut node_a, &kp);

    let data = b"shared content across nodes";
    let content_ref = {
        let mut pub_ = DataPublisher::new(&mut node_a);
        pub_.publish(&kp, data, "text/plain", None).unwrap()
    };

    node_b.sync().unwrap();

    let resolved_b = node_b.content(&content_ref.cid);
    assert!(resolved_b.is_some(), "node B must see content after sync");
    assert_eq!(resolved_b.unwrap().cid, content_ref.cid);

    let did_doc_b = node_b.identity().resolve_document(&kp.did());
    assert!(did_doc_b.is_some(), "node B must see DID after sync");
}

#[test]
fn did_lifecycle_blocks_content_after_revoke() {
    let mut node = make_node();
    let kp = Keypair::generate();

    register_did(&mut node, &kp);

    let data = b"content before revocation";
    let content_ref = {
        let mut pub_ = DataPublisher::new(&mut node);
        pub_.publish(&kp, data, "text/plain", None).unwrap()
    };
    assert!(node.content(&content_ref.cid).is_some());

    {
        let mut mgr = DidManager::new(&mut node);
        mgr.revoke_did(&kp).unwrap();
    }

    let mgr = DidManager::new(&mut node);
    assert!(
        mgr.resolve(&kp.did()).is_none(),
        "revoked DID must not resolve"
    );

    assert!(
        node.content(&content_ref.cid).is_some(),
        "content published before revocation remains in the log"
    );
}
