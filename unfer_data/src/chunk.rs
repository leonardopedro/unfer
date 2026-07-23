use sha2::{Digest, Sha256};
use unfer_protocol::ChunkRef;

pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;

pub struct Chunker {
    chunk_size: usize,
}

impl Chunker {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
        }
    }

    pub fn chunk(&self, data: &[u8]) -> Vec<(u32, Vec<u8>)> {
        data.chunks(self.chunk_size)
            .enumerate()
            .map(|(i, c)| (i as u32, c.to_vec()))
            .collect()
    }

    pub fn chunk_refs(&self, data: &[u8]) -> Vec<ChunkRef> {
        data.chunks(self.chunk_size)
            .enumerate()
            .map(|(i, c)| ChunkRef {
                index: i as u32,
                cid: compute_cid(c),
                size: c.len() as u64,
            })
            .collect()
    }
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(DEFAULT_CHUNK_SIZE)
    }
}

pub fn compute_cid(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    hex::encode(hash)
}

pub fn verify_chunk(data: &[u8], expected_cid: &str) -> bool {
    compute_cid(data) == expected_cid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_splits_correctly() {
        let chunker = Chunker::new(4);
        let data = b"hello world!";
        let chunks = chunker.chunk(data);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], (0, b"hell".to_vec()));
        assert_eq!(chunks[1], (1, b"o wo".to_vec()));
        assert_eq!(chunks[2], (2, b"rld!".to_vec()));
    }

    #[test]
    fn chunker_single_chunk() {
        let chunker = Chunker::new(1024);
        let data = b"small";
        let chunks = chunker.chunk(data);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (0, b"small".to_vec()));
    }

    #[test]
    fn chunker_empty_input() {
        let chunker = Chunker::new(4);
        let chunks = chunker.chunk(b"");
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_refs_have_correct_cids() {
        let chunker = Chunker::new(4);
        let data = b"abcdefgh";
        let refs = chunker.chunk_refs(data);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].index, 0);
        assert_eq!(refs[0].size, 4);
        assert_eq!(refs[0].cid, compute_cid(b"abcd"));
        assert_eq!(refs[1].index, 1);
        assert_eq!(refs[1].size, 4);
        assert_eq!(refs[1].cid, compute_cid(b"efgh"));
    }

    #[test]
    fn compute_cid_is_deterministic() {
        let cid1 = compute_cid(b"test data");
        let cid2 = compute_cid(b"test data");
        assert_eq!(cid1, cid2);
        assert_eq!(cid1.len(), 64);
    }

    #[test]
    fn compute_cid_differs_for_different_data() {
        assert_ne!(compute_cid(b"aaa"), compute_cid(b"bbb"));
    }

    #[test]
    fn verify_chunk_passes_for_correct_cid() {
        let data = b"chunk data";
        let cid = compute_cid(data);
        assert!(verify_chunk(data, &cid));
    }

    #[test]
    fn verify_chunk_fails_for_wrong_cid() {
        assert!(!verify_chunk(b"chunk data", "0000"));
    }

    #[test]
    fn reassembled_chunks_match_original() {
        let chunker = Chunker::new(3);
        let original = b"the quick brown fox jumps over the lazy dog";
        let chunks = chunker.chunk(original);
        let reassembled: Vec<u8> = chunks.iter().flat_map(|(_, c)| c.iter().copied()).collect();
        assert_eq!(reassembled, original);
    }
}
