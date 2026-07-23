use unfer_consensus::{ConsensusNode, Keypair};
use unfer_protocol::{ChunkRef, ConsensusTransaction, ContentOp, ContentRef, Diagnostic};

use crate::chunk::{Chunker, compute_cid};
use crate::crypto::{DataKeypair, derive_aes_key, encrypt_chunk};
use crate::magnet::{build_magnet_uri, content_cid_from_chunks};

pub struct DataPublisher<'a> {
    node: &'a mut ConsensusNode,
    chunker: Chunker,
}

impl<'a> DataPublisher<'a> {
    pub fn new(node: &'a mut ConsensusNode) -> Self {
        Self {
            node,
            chunker: Chunker::default(),
        }
    }

    pub fn with_chunk_size(node: &'a mut ConsensusNode, chunk_size: usize) -> Self {
        Self {
            node,
            chunker: Chunker::new(chunk_size),
        }
    }

    pub fn publish(
        &mut self,
        did_keypair: &Keypair,
        data: &[u8],
        mime_type: &str,
        display_name: Option<&str>,
    ) -> Result<ContentRef, Diagnostic> {
        let data_kp = DataKeypair::generate();
        let aes_key = derive_aes_key(&data_kp.shared_secret(data_kp.public_key()));

        let chunks = self.chunker.chunk(data);
        let mut chunk_refs = Vec::with_capacity(chunks.len());
        let mut chunk_cids = Vec::with_capacity(chunks.len());

        for (index, plaintext) in &chunks {
            let ciphertext = encrypt_chunk(&aes_key, *index, plaintext)
                .map_err(|e| {
                    Diagnostic::new(
                        unfer_protocol::Code::INTERNAL,
                        format!("chunk encryption failed: {e}"),
                        unfer_protocol::Severity::Error,
                    )
                })?;
            let cid = compute_cid(&ciphertext);
            chunk_cids.push(cid.clone());
            chunk_refs.push(ChunkRef {
                index: *index,
                cid,
                size: ciphertext.len() as u64,
            });
        }

        let content_cid = content_cid_from_chunks(&chunk_cids);
        let magnet_uri = build_magnet_uri(&content_cid, display_name);

        let content_ref = ContentRef {
            cid: content_cid,
            magnet_uri,
            encryption_key: format!("x25519:{}", data_kp.public_key_hex()),
            filesize: data.len() as u64,
            mime_type: mime_type.to_string(),
            chunks: chunk_refs,
        };

        let mut tx = ConsensusTransaction::ContentOp(ContentOp {
            did: did_keypair.did(),
            content_ref: content_ref.clone(),
            signature: [0u8; 64],
        });
        unfer_consensus::sign_transaction(&mut tx, did_keypair);
        self.node.submit(tx)?;
        self.node.sync()?;

        Ok(content_ref)
    }

    pub fn resolve(&self, cid: &str) -> Option<&ContentRef> {
        self.node.content(cid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_consensus::LocalConsensus;

    fn make_node() -> ConsensusNode {
        ConsensusNode::new(Box::new(LocalConsensus::new()))
    }

    #[test]
    fn publish_and_resolve() {
        let mut node = make_node();
        let kp = Keypair::generate();

        // Register the DID first.
        let mut id_tx = ConsensusTransaction::IdentityOp(unfer_protocol::IdentityOp {
            did: kp.did(),
            op_kind: unfer_protocol::IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        unfer_consensus::sign_transaction(&mut id_tx, &kp);
        node.submit(id_tx).unwrap();
        node.sync().unwrap();

        let data = b"hello, data plane!";
        let content_ref = {
            let mut pub_ = DataPublisher::new(&mut node);
            pub_
                .publish(&kp, data, "text/plain", Some("greeting.txt"))
                .unwrap()
        };

        assert_eq!(content_ref.filesize, data.len() as u64);
        assert_eq!(content_ref.mime_type, "text/plain");
        assert!(content_ref.encryption_key.starts_with("x25519:"));
        assert!(content_ref.magnet_uri.starts_with("magnet:?xt=urn:btih:"));
        assert!(!content_ref.chunks.is_empty());
        assert_eq!(content_ref.cid.len(), 64);

        let resolved = node.content(&content_ref.cid);
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().magnet_uri, content_ref.magnet_uri);
    }

    #[test]
    fn publish_large_data_multiple_chunks() {
        let mut node = make_node();
        let kp = Keypair::generate();

        let mut id_tx = ConsensusTransaction::IdentityOp(unfer_protocol::IdentityOp {
            did: kp.did(),
            op_kind: unfer_protocol::IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        unfer_consensus::sign_transaction(&mut id_tx, &kp);
        node.submit(id_tx).unwrap();
        node.sync().unwrap();

        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let content_ref = {
            let mut pub_ = DataPublisher::with_chunk_size(&mut node, 100);
            pub_
                .publish(&kp, &data, "application/octet-stream", None)
                .unwrap()
        };

        assert_eq!(content_ref.chunks.len(), 10);
        for (i, chunk) in content_ref.chunks.iter().enumerate() {
            assert_eq!(chunk.index, i as u32);
            assert_eq!(chunk.cid.len(), 64);
        }
    }

    #[test]
    fn publish_empty_data() {
        let mut node = make_node();
        let kp = Keypair::generate();

        let mut id_tx = ConsensusTransaction::IdentityOp(unfer_protocol::IdentityOp {
            did: kp.did(),
            op_kind: unfer_protocol::IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        unfer_consensus::sign_transaction(&mut id_tx, &kp);
        node.submit(id_tx).unwrap();
        node.sync().unwrap();

        let content_ref = {
            let mut pub_ = DataPublisher::new(&mut node);
            pub_.publish(&kp, b"", "text/plain", None).unwrap()
        };
        assert_eq!(content_ref.filesize, 0);
        assert!(content_ref.chunks.is_empty());
    }

    #[test]
    fn resolve_unknown_cid_returns_none() {
        let mut node = make_node();
        let pub_ = DataPublisher::new(&mut node);
        assert!(pub_.resolve("nonexistent").is_none());
    }

    #[test]
    fn chunk_cids_are_unique_per_chunk() {
        let mut node = make_node();
        let kp = Keypair::generate();

        let mut id_tx = ConsensusTransaction::IdentityOp(unfer_protocol::IdentityOp {
            did: kp.did(),
            op_kind: unfer_protocol::IdentityOpKind::Create,
            signing_key: kp.public_key(),
            signature: [0u8; 64],
            seq: 1,
            service_endpoint: None,
        });
        unfer_consensus::sign_transaction(&mut id_tx, &kp);
        node.submit(id_tx).unwrap();
        node.sync().unwrap();

        let data = b"aaaaaabbbbbbcccccc";
        let content_ref = {
            let mut pub_ = DataPublisher::with_chunk_size(&mut node, 6);
            pub_.publish(&kp, data, "text/plain", None).unwrap()
        };

        let cids: Vec<&str> = content_ref.chunks.iter().map(|c| c.cid.as_str()).collect();
        let unique: std::collections::HashSet<&str> = cids.iter().copied().collect();
        assert_eq!(cids.len(), unique.len(), "chunk CIDs must be unique");
    }
}
