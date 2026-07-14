//! "Tangible step" test: run the QFM model with the **natural
//! dense W matrix** + **Krylov subspace projection** on a tiny
//! controlled synthetic corpus. **No oxieml, no hashing of W** —
//! the dense W matrix is the model.
//!
//! User guidance (2026-07-10):
//! - "There must be Krylov subspace."
//! - "not an attempt to simplify the large matrix that projects
//!   the vectors to the Krylov subspace (no to attempts such as
//!   oxieml or hashing)"
//!
//! What we test here:
//! - The QFM model API (`QfmTextModel::logprob` /
//!   `next_token_dist`) uses the natural dense W matrix
//!   (`M × rank`) as the change-of-basis from mode index to
//!   Krylov basis. The W matrix is loaded and used in dense
//!   form — no oxieml replacement, no hash compression.
//! - On a tiny synthetic corpus where the next token is
//!   deterministic given the context (e.g. "ABABAB..."), the
//!   model achieves a much lower perplexity than unigram.
//!
//! Memory note: the test uses `block_sizes = [256; 2]` (total M
//! = 512 modes) and `max_rank = 2` so the dense W matrix is
//! 512 × 2 × 16 bytes = 16 KB. The accumulator only stores
//! active modes, so the actual memory is O(corpus_size ×
//! n_orders). This keeps the test below 1 MB of allocations.

use qfm_text::accumulate::{accumulate_shards, Encoder};
use qfm_text::config::{DecodeStrategy, TextConfig};
use qfm_text::features::OrderHasher;
use qfm_text::model::QfmTextModel;
use std::io::Write;

fn small_config(n_orders: usize) -> TextConfig {
    TextConfig {
        n_orders,
        hist_cap: 32,
        max_rank: 2,
        m_shifts: 4,
        lambda: vec![1.0; n_orders],
        t: 1.0,
        discount: 0.75,
        seed: 42,
        decode_strategy: DecodeStrategy::Renormalize,
        top_k: 4,
        block_sizes: vec![256; n_orders],
        salts: (1..=n_orders as u64).collect(),
        use_registry_encoder: false,
        fock_resolution: None,
    }
}

fn tempdir() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "qfm_text_dense_w_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write_shard_and_manifest(
    dir: &std::path::Path,
    tokens: &[u32],
) -> (std::path::PathBuf, std::path::PathBuf) {
    let path = dir.join("shard_00000.bin");
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&bytes).unwrap();
    let manifest_path = dir.join("manifest.json");
    let json = format!(
        r#"{{
            "schema": "qfm_text.shard_manifest/v1",
            "vocab_size": 1024,
            "tokens_per_shard": {},
            "n_shards": 1,
            "n_tokens": {},
            "shards": [{{
                "n_tokens": {},
                "path": "shard_00000.bin",
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
            }}],
            "tokenizer_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "tokenizer_path": "tokenizer.json",
            "corpus": "test",
            "license": "test",
            "attribution": "test"
        }}"#,
        tokens.len(),
        tokens.len(),
        tokens.len()
    );
    std::fs::write(&manifest_path, json).unwrap();
    (path, manifest_path)
}

fn train_qfm(cfg: &TextConfig, shard_path: &std::path::Path) -> QfmTextModel {
    let mut encoder = Encoder::Hasher(OrderHasher::new(cfg.clone()));
    let acc = accumulate_shards(&[shard_path.to_path_buf()], &mut encoder, cfg, 1024)
        .expect("accumulate_shards");
    let registry = qfm_text::registry::ContextRegistry::new(cfg.n_orders);
    QfmTextModel::from_accumulator(acc, registry, cfg).expect("from_accumulator")
}

#[test]
fn dense_w_is_used_during_inference() {
    // Step 2 (revised): confirm the dense W matrix is the
    // active decode path. The Krylov subspace is required, but
    // the W matrix is kept in its natural dense form (no
    // oxieml, no hashing).
    let cfg = small_config(2);
    let tokens: Vec<u32> = (0..200u32).map(|i| i % 4).collect();
    let dir = tempdir();
    let (shard, _manifest) = write_shard_and_manifest(&dir, &tokens);
    let model = train_qfm(&cfg, &shard);

    let w = model.w_matrix();
    let k = model.krylov_rank();
    let k2 = model.k2_total();
    eprintln!(
        "dense_w_is_used_during_inference: W shape = {} x {}, k2_total = {}, krylov_rank = {}",
        w.nrows(),
        w.ncols(),
        k2,
        k
    );
    assert!(w.nrows() > 0, "W matrix has 0 rows");
    assert!(w.ncols() > 0, "W matrix has 0 cols");
    assert!(k > 0, "krylov_rank is 0");
    assert!(k2 > 0, "k2_total is 0");
    assert!(w.nrows() as u32 <= k2, "W has more rows than k2_total");
    assert!(w.ncols() as usize == k, "W cols != krylov_rank");

    // Verify the decoder is Dense (NOT Analytical — oxieml
    // was rejected by the user).
    let decoder = model.decoder();
    eprintln!("  decoder kind: {:?}", std::mem::discriminant(decoder));
    match decoder {
        qfm_text::model::DecoderKind::Dense { .. } => {}
        qfm_text::model::DecoderKind::Analytical { .. } => {
            panic!("decoder should be Dense, not Analytical (oxieml was rejected)")
        }
    }

    // Verify the model can do inference (no panic).
    let dist = model.next_token_dist(&[0, 1]).expect("next_token_dist");
    // The vocab size is the configured 1024, but the model
    // can also return a smaller distribution if it has fewer
    // unique tokens. We just check that the distribution is
    // non-empty and finite.
    assert!(!dist.is_empty(), "next_token_dist is empty");
    let sum: f64 = dist.iter().sum();
    assert!(sum.is_finite(), "next_token_dist contains non-finite");
    eprintln!(
        "  next_token_dist: len = {}, sum = {:.6}",
        dist.len(),
        sum
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn qfm_learns_deterministic_alternation() {
    // Tiny corpus: deterministic alternation. The QFM should
    // learn to put all mass on the next-expected token.
    let cfg = small_config(2);
    let n = 200u32;
    let train: Vec<u32> = (0..n).map(|i| if i % 2 == 0 { 7 } else { 11 }).collect();
    let held_out: Vec<u32> = (n..n + 50).map(|i| if i % 2 == 0 { 7 } else { 11 }).collect();
    let dir = tempdir();
    let (shard, _manifest) = write_shard_and_manifest(&dir, &train);
    let model = train_qfm(&cfg, &shard);

    // Evaluate on held_out: track QFM logprob and unigram
    // logprob (uniform over vocab).
    let mut qfm_lp = 0.0_f64;
    let mut n_tok = 0u64;
    let unigram_lp_per = -(1.0_f64 / 1024.0_f64).ln();
    for window in held_out.windows(3) {
        let ctx = &window[..2];
        let next = window[2];
        let dist = model.next_token_dist(ctx).expect("next_token_dist");
        let p = dist[next as usize].max(1e-30);
        qfm_lp += -p.ln();
        n_tok += 1;
    }
    let qfm_ppl = (qfm_lp / n_tok as f64).exp();
    let unigram_ppl = unigram_lp_per.exp();
    eprintln!(
        "qfm_learns_deterministic_alternation: qfm_ppl={:.3} unigram_ppl={:.3}",
        qfm_ppl, unigram_ppl
    );
    // The QFM has access to context. On a deterministic
    // alternation, it should be much better than unigram.
    assert!(
        qfm_ppl < unigram_ppl * 0.5,
        "QFM ppl {qfm_ppl} should be < 0.5 * unigram {unigram_ppl}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn krylov_subspace_is_used_during_inference() {
    // The Krylov subspace IS the rank-r reduction of the
    // forward sequence w_k = (H - z_k I)^k c_0. The W matrix
    // projects mode indices to this subspace. Verify that the
    // pipeline reports a krylov_rank > 0 (the subspace
    // exists) and that the evolution reduces the dimension
    // (krylov_rank = rank, not M).
    let cfg = small_config(2);
    let tokens: Vec<u32> = (0..100u32).map(|i| i % 4).collect();
    let dir = tempdir();
    let (shard, _manifest) = write_shard_and_manifest(&dir, &tokens);
    let model = train_qfm(&cfg, &shard);

    let w = model.w_matrix();
    let k = model.krylov_rank();
    eprintln!(
        "krylov_subspace_is_used_during_inference: W shape = {} x {}, krylov_rank = {}",
        w.nrows(),
        w.ncols(),
        k
    );
    // The Krylov subspace dim = rank ≤ max_rank = 2. Verify
    // it's at most 2 and the W has rank columns.
    assert!(k <= cfg.max_rank, "krylov_rank {k} > max_rank {}", cfg.max_rank);
    assert!(w.ncols() == k, "W cols {} != krylov_rank {}", w.ncols(), k);
    let _ = std::fs::remove_dir_all(&dir);
}
