//! Integration test: end-to-end QFM-Text on the WikiText-103
//! shards (if present) or the synthetic fixture (otherwise).
//!
//! This is the Stage 6 acceptance test: train a model + score it
//! against the classical n-gram baseline + the unigram. The
//! numbers are recorded in `docs/QFM_TEXT_STATUS.md`; this test
//! just verifies the pipeline runs end-to-end without panicking
//! and produces finite perplexities.

use std::path::Path;

use qfm_text::{
    NgramBaseline, QfmTextModel, Shard, TextConfig, accumulate_shards, perplexity,
    perplexity_baseline_capped,
};

fn small_cfg() -> TextConfig {
    TextConfig {
        n_orders: 2,
        block_sizes: vec![128, 128],
        salts: vec![1, 2],
        hist_cap: 16,
        max_rank: 4,
        m_shifts: 4,
        lambda: vec![1.0, 1.0],
        t: 1.0,
        discount: 0.75,
        seed: 0,
        ..Default::default()
    }
}

fn run_with_shard(shard_path: &Path, vocab_size: u32) {
    let cfg = small_cfg();
    // 1. Accumulate.
    let acc = accumulate_shards(&[shard_path.to_path_buf()], &cfg, vocab_size)
        .expect("accumulate");
    assert!(acc.total_windows > 0);
    // 2. Compile.
    let model = QfmTextModel::from_accumulator(acc.clone(), &cfg).expect("compile");
    // 3. Score (capped to 1000 windows for test speed).
    let shard = Shard::open(shard_path, vocab_size).expect("open");
    let qp = perplexity(&model, &shard).expect("qfm ppl");
    assert!(qp.ppl.is_finite() && qp.ppl > 0.0);
    let baseline = NgramBaseline::from_accumulator(acc);
    let bp = perplexity_baseline_capped(&baseline, &shard, 1000).expect("baseline ppl");
    assert!(bp.ppl.is_finite() && bp.ppl > 0.0);
    eprintln!(
        "[integration] shard = {}, vocab = {}, qfm ppl = {:.3}, baseline ppl = {:.3}",
        shard_path.display(),
        vocab_size,
        qp.ppl,
        bp.ppl
    );
}

#[test]
fn integration_synthetic_fixture() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("tiny_fixture.bin");
    if !p.exists() {
        eprintln!("[integration] fixture missing, skipping");
        return;
    }
    run_with_shard(&p, 16);
}

#[test]
fn integration_wikitext_smoke() {
    // If the WikiText-103 test split shards are present
    // (produced by `qfm_text/scripts/prepare_corpus.sh`), score
    // a small slice.
    let manifest_path = std::env::var("QFM_TEXT_MANIFEST").ok();
    let manifest_path = match manifest_path {
        Some(p) => Path::new(&p).to_path_buf(),
        None => {
            eprintln!("[integration] QFM_TEXT_MANIFEST unset, skipping");
            return;
        }
    };
    if !manifest_path.exists() {
        eprintln!("[integration] manifest missing, skipping");
        return;
    }
    let manifest = qfm_text::Manifest::read(&manifest_path).expect("read manifest");
    let shard_dir = manifest_path.parent().expect("parent");
    let shard = shard_dir.join(&manifest.shards[0].path);
    run_with_shard(&shard, manifest.vocab_size);
}
