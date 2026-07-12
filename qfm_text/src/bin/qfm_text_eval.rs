//! `qfm_text_eval` — Stage 6 of the QFM-Text plan.
//!
//! The HRM-Text `evaluation.main` analog: load a checkpoint, score
//! it (and the classical n-gram baseline + unigram) on a held-out
//! split, optionally run a hyperparameter sweep over `t`, `lambda`,
//! and `discount` to find the best config on the validation split.
//!
//! Usage:
//!   qfm_text_eval --checkpoint ./out/checkpoint_epoch0.qfm
//!                  --manifest manifest.json
//!                  [--split test]
//!                  [--sweep]
//!                  [--diagnose]
//!                  [--prompts 5]
//!                  [--encode super|model_avg]
//!                  [--decode dense|renormalize|top_k|order_prior]
//!                  [--top-k N]

use std::path::PathBuf;

use anyhow::Context;
use qfm_text::{
    DecodeStrategy, NgramBaseline, PerplexityReport, QfmTextModel, Shard, accumulate_shards,
    perplexity, perplexity_baseline, perplexity_baseline_capped, perplexity_capped,
    perplexity_model_avg, perplexity_model_avg_capped, sample_text,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncodeKind {
    Superposition,
    ModelAvg,
}

fn main() -> anyhow::Result<()> {
    let mut ckpt: Option<PathBuf> = None;
    let mut manifest_path: Option<PathBuf> = None;
    let mut eval_manifest_path: Option<PathBuf> = None;
    let mut split = "test".to_string();
    let mut sweep = false;
    let mut diagnose = false;
    let mut baseline_from_checkpoint = false;
    let mut n_prompts: usize = 5;
    let mut max_tokens: usize = 0;
    let mut encode_kind = EncodeKind::Superposition;
    let mut decode_strategy = DecodeStrategy::default();
    let mut top_k: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--checkpoint" => {
                ckpt = Some(PathBuf::from(args.next().context("--checkpoint needs value")?));
            }
            "--manifest" => {
                manifest_path =
                    Some(PathBuf::from(args.next().context("--manifest needs value")?));
            }
            "--eval-manifest" => {
                eval_manifest_path = Some(PathBuf::from(
                    args.next().context("--eval-manifest needs value")?,
                ));
            }
            "--split" => {
                split = args.next().context("--split needs value")?;
            }
            "--sweep" => {
                sweep = true;
            }
            "--diagnose" => {
                diagnose = true;
            }
            "--baseline-from-checkpoint" => {
                baseline_from_checkpoint = true;
            }
            "--prompts" => {
                n_prompts = args
                    .next()
                    .context("--prompts needs value")?
                    .parse()
                    .context("--prompts must be a number")?;
            }
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .context("--max-tokens needs value")?
                    .parse()
                    .context("--max-tokens must be a number")?;
            }
            "--encode" => {
                let kind = args.next().context("--encode needs value")?;
                encode_kind = match kind.as_str() {
                    "super" | "superposition" => EncodeKind::Superposition,
                    "model_avg" | "model-avg" | "avg" => EncodeKind::ModelAvg,
                    other => anyhow::bail!("unknown --encode kind: {other}"),
                };
            }
            "--decode" => {
                let s = args.next().context("--decode needs value")?;
                decode_strategy = match s.as_str() {
                    "dense" => DecodeStrategy::Dense,
                    "renormalize" | "renorm" => DecodeStrategy::Renormalize,
                    "top_k" | "top-k" | "topk" => DecodeStrategy::TopK,
                    "order_prior" | "order-prior" | "prior" => DecodeStrategy::OrderPrior,
                    other => anyhow::bail!("unknown --decode strategy: {other}"),
                };
            }
            "--top-k" => {
                let v = args
                    .next()
                    .context("--top-k needs value")?
                    .parse()
                    .context("--top-k must be a number")?;
                top_k = Some(v);
            }
            "--help" | "-h" => {
                eprintln!(
                    "qfm_text_eval --checkpoint <ckpt.qfm> --manifest <train_manifest.json>\n\
                     [--eval-manifest <test_manifest.json>] [--split test] [--sweep] [--prompts 5]\n\
                     [--max-tokens N] [--encode super|model_avg]\n\
                     [--decode dense|renormalize|top_k|order_prior] [--top-k N]\n\n\
                     --manifest is the manifest used to accumulate the baseline\n\
                     (i.e. the data the baseline was *trained* on).\n\
                     --eval-manifest is the manifest used to score the model\n\
                     (i.e. the held-out test data). If --eval-manifest is\n\
                     omitted, the model and baseline are both scored on the\n\
                     --manifest shards (a training-data fit)."
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }
    let ckpt = ckpt.context("--checkpoint is required")?;
    let manifest_path = manifest_path.context("--manifest is required")?;
    let manifest = qfm_text::Manifest::read(&manifest_path)?;
    let shard_dir = manifest_path.parent().context("manifest_path needs parent")?;
    // If --eval-manifest is provided, score on those shards; otherwise
    // score on the --manifest shards (training-data fit).
    let (eval_manifest, eval_label) = if let Some(ep) = &eval_manifest_path {
        let m = qfm_text::Manifest::read(ep)?;
        (m, format!("held-out: {}", ep.display()))
    } else {
        (
            manifest.clone(),
            format!("in-sample: {}", manifest_path.display()),
        )
    };
    let eval_shard_dir = if let Some(ep) = &eval_manifest_path {
        ep.parent().context("eval-manifest path needs parent")?
    } else {
        shard_dir
    };
    eprintln!(
        "[qfm_text_eval] train_manifest = {}, eval = {eval_label}, checkpoint = {}, sweep = {sweep}, diagnose = {diagnose}, baseline_from_checkpoint = {baseline_from_checkpoint}, n_prompts = {n_prompts}, encode = {encode_kind:?}, decode = {decode_strategy:?}, top_k = {top_k:?}",
        manifest_path.display(),
        ckpt.display(),
    );
    // The baseline is accumulated from the *training* manifest
    // (the data the model was trained on); the model is scored on
    // the *eval* manifest (held-out data). This is the proper
    // held-out evaluation.
    let train_shard_paths: Vec<PathBuf> = (0..manifest.shards.len())
        .map(|i| manifest.shard_path(shard_dir, i))
        .collect();
    let eval_shard_paths: Vec<PathBuf> = (0..eval_manifest.shards.len())
        .map(|i| eval_manifest.shard_path(eval_shard_dir, i))
        .collect();
    // Open the checkpoint.
    eprintln!("[qfm_text_eval] loading checkpoint from {}", ckpt.display());
    let mut model = QfmTextModel::load(&ckpt)?;
    eprintln!("[qfm_text_eval] model loaded");
    model.cfg.decode_strategy = decode_strategy;
    if let Some(k) = top_k {
        model.cfg.top_k = k;
    }
    // Re-derive a baseline from the *training* shards (not the
    // eval shards) so the comparison is fair: both the model and
    // the baseline see the same training data, and both are scored
    // on the same held-out data.
    let cfg = model.cfg.clone();
    // (The model's own registry is used at inference time and is
    // also the one handed to the baseline when the user passes
    // `--baseline-from-checkpoint`. The `else` branch below
    // re-derives both the accumulator and the registry from the
    // train shards.)
    // Two paths to the baseline:
    //   - default: re-derive from the train shards (slow: ~7 min
    //     for the full WikiText-103 train corpus, but the user has
    //     the same data the model was trained on)
    //   - --baseline-from-checkpoint: use the model's stored
    //     mode_hists + unigram directly (fast: <1 s, no corpus
    //     pass). The unigram is reconstructed from the normalized
    //     `unigram: Vec<f64>` + `unigram_total: f64` (approximate
    //     but the baseline only needs the distribution shape).
    let baseline = if baseline_from_checkpoint {
        eprintln!("[qfm_text_eval] building baseline from checkpoint (clone mode_hists)...");
        let baseline_acc = model.as_accumulator();
        let baseline_reg = model.registry_clone();
        eprintln!("[qfm_text_eval] baseline from checkpoint built");
        NgramBaseline::from_accumulator(baseline_acc, baseline_reg)
    } else {
        eprintln!("[qfm_text_eval] accumulating baseline from train shards ({} shards)...", train_shard_paths.len());
        let mut baseline_reg = qfm_text::ContextRegistry::new(cfg.n_orders);
        let mut baseline_enc = qfm_text::Encoder::Registry(baseline_reg.clone());
        let baseline_acc = accumulate_shards(&train_shard_paths, &mut baseline_enc, &cfg, manifest.vocab_size)?;
        if let Some(r) = baseline_enc.as_registry() {
            baseline_reg.clone_from(r);
        }
        eprintln!("[qfm_text_eval] baseline accumulated");
        NgramBaseline::from_accumulator(baseline_acc, baseline_reg)
    };
    // Diagnostic: SVD ranks of W and H_m, and how well the QFM
    // predicted distribution matches the empirical next-token
    // distribution for a few contexts. This tells us whether the
    // model is underfit (rank saturated by H) or simply has a
    // good/bad match on individual contexts.
    if diagnose {
        diagnose_pipeline(&model, &baseline, &train_shard_paths[0], manifest.vocab_size)?;
    }
    // `cfg` is no longer needed (we used it to build the baseline
    // and to override decode_strategy on the model above); drop it
    // explicitly so the borrow checker is happy with the model
    // clone in the sweep below.
    drop(cfg);
    // Score the model + baseline + unigram on each eval shard.
    let mut qfm_total = PerplexityReport {
        n_tokens: 0,
        nll_nats_per_token: 0.0,
        ppl: 0.0,
    };
    let mut base_total = qfm_total.clone();
    let mut uni_total = qfm_total.clone();
    for sp in &eval_shard_paths {
        let shard = Shard::open(sp, eval_manifest.vocab_size).with_context(|| format!("open {sp:?}"))?;
        let qp = match (encode_kind, max_tokens) {
            (EncodeKind::Superposition, 0) => perplexity(&model, &shard)?,
            (EncodeKind::Superposition, n) => perplexity_capped(&model, &shard, n)?,
            (EncodeKind::ModelAvg, 0) => perplexity_model_avg(&model, &shard)?,
            (EncodeKind::ModelAvg, n) => perplexity_model_avg_capped(&model, &shard, n)?,
        };
        let bp = if max_tokens > 0 {
            perplexity_baseline_capped(&baseline, &shard, max_tokens)?
        } else {
            perplexity_baseline(&baseline, &shard)?
        };
        let up = if max_tokens > 0 {
            NgramBaseline::unigram_ppl_capped(&shard, max_tokens)?
        } else {
            NgramBaseline::unigram_ppl(&shard)?
        };
        qfm_total.n_tokens += qp.n_tokens;
        qfm_total.nll_nats_per_token += qp.nll_nats_per_token * qp.n_tokens as f64;
        base_total.n_tokens += bp.n_tokens;
        base_total.nll_nats_per_token += bp.nll_nats_per_token * bp.n_tokens as f64;
        uni_total.n_tokens += up.n_tokens;
        uni_total.nll_nats_per_token += up.nll_nats_per_token * up.n_tokens as f64;
        eprintln!(
            "[shard {}] qfm ppl = {:.3}, baseline ppl = {:.3}, unigram ppl = {:.3}",
            sp.display(),
            qp.ppl,
            bp.ppl,
            up.ppl,
        );
        if max_tokens > 0 && qfm_total.n_tokens >= max_tokens as u64 {
            break;
        }
    }
    let n = qfm_total.n_tokens.max(1) as f64;
    qfm_total.nll_nats_per_token /= n;
    qfm_total.ppl = qfm_total.nll_nats_per_token.exp();
    let n = base_total.n_tokens.max(1) as f64;
    base_total.nll_nats_per_token /= n;
    base_total.ppl = base_total.nll_nats_per_token.exp();
    let n = uni_total.n_tokens.max(1) as f64;
    uni_total.nll_nats_per_token /= n;
    uni_total.ppl = uni_total.nll_nats_per_token.exp();
    eprintln!(
        "=== summary (split = {split}, encode = {:?}, decode = {:?}, top_k = {:?}) ===",
        encode_kind, decode_strategy, model.cfg.top_k
    );
    eprintln!("  QFM ppl       = {:.3}", qfm_total.ppl);
    eprintln!("  n-gram ppl    = {:.3}", base_total.ppl);
    eprintln!("  unigram ppl   = {:.3}", uni_total.ppl);
    if sweep {
        let ts = [0.25_f64, 0.5, 1.0, 1.5, 2.0];
        let discounts = [0.5_f64, 0.75];
        let mut best: Option<(f64, f64, f64, f64)> = None;
        for &t in &ts {
            for &d in &discounts {
                let mut m = model.clone();
                m.cfg.t = t;
                m.cfg.discount = d;
                let mut nll = 0.0;
                let mut nt = 0u64;
                for sp in &eval_shard_paths {
                    let shard = Shard::open(sp, eval_manifest.vocab_size)?;
                    let p = match (encode_kind, max_tokens) {
                        (EncodeKind::Superposition, 0) => perplexity(&m, &shard)?,
                        (EncodeKind::Superposition, n) => perplexity_capped(&m, &shard, n)?,
                        (EncodeKind::ModelAvg, 0) => perplexity_model_avg(&m, &shard)?,
                        (EncodeKind::ModelAvg, n) => perplexity_model_avg_capped(&m, &shard, n)?,
                    };
                    nll += p.nll_nats_per_token * p.n_tokens as f64;
                    nt += p.n_tokens;
                    if max_tokens > 0 && nt >= max_tokens as u64 {
                        break;
                    }
                }
                let ppl = (nll / nt.max(1) as f64).exp();
                if best.map_or(true, |(_, _, _, bpp)| ppl < bpp) {
                    best = Some((t, d, nll / nt.max(1) as f64, ppl));
                }
            }
        }
        if let Some((t, d, nll, ppl)) = best {
            eprintln!(
                "[sweep] best t = {t}, discount = {d}, ppl = {ppl:.3} (nll = {nll:.4})"
            );
        }
    }
    // Sample a few continuations.
    let prompts: Vec<Vec<u32>> = vec![
        vec![0, 1],
        vec![1, 2],
        vec![2, 3],
        vec![3, 0],
        vec![0, 0, 1],
    ];
    for (i, prompt) in prompts.iter().take(n_prompts).enumerate() {
        let s = sample_text(&model, prompt, 16, 1.0, 42);
        eprintln!("[sample {i}] prompt = {prompt:?}, out = {s:?}");
    }
    Ok(())
}

/// SVD-based numerical rank of a complex matrix. Returns the count
/// of singular values above the absolute threshold
/// `max_dim * max_sv * 1e-10` (a relative-to-largest-sv cutoff that
/// is robust to scale).
fn svd_rank(matrix: &nalgebra::DMatrix<num_complex::Complex64>, label: &str) -> usize {
    // The singular values of a complex M are the sqrt of the eigenvalues
    // of M†M (Hermitian PSD). nalgebra's `SymmetricEigen` actually
    // implements Hermitian eigendecomposition for complex matrices.
    let (nrows, ncols) = (matrix.nrows(), matrix.ncols());
    let mt: nalgebra::DMatrix<num_complex::Complex64> = matrix.adjoint();
    let mtm = &mt * matrix;
    let svd = nalgebra::linalg::SymmetricEigen::new(mtm);
    let mut sv: Vec<f64> = svd
        .eigenvalues
        .iter()
        .map(|&c: &f64| c.max(0.0))
        .collect();
    sv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let max_sv = sv.first().copied().unwrap_or(0.0);
    let dim = (nrows.max(ncols)) as f64;
    let thr = (dim * max_sv * 1e-10).max(1e-300);
    let rank = sv.iter().filter(|&&s| s > thr).count();
    eprintln!(
        "[diagnose] {label}: shape = ({nrows}, {ncols}), top 5 singular values = {:?}, numerical rank (> {thr:.2e}) = {rank}",
        &sv[..sv.len().min(5)]
    );
    rank
}

/// Per-context fit diagnostic. For a handful of contexts drawn from
/// the training shard, build the empirical next-token histogram and
/// the QFM predicted distribution, then report the KL divergence,
/// cosine similarity, and the top-1 / top-5 overlap. This tells us
/// whether the model is approximating the empirical distribution
/// (a good fit) or producing a near-uniform distribution that misses
/// the actual distribution (an underfit).
fn diagnose_pipeline(
    model: &QfmTextModel,
    baseline: &NgramBaseline,
    train_shard_path: &std::path::Path,
    vocab_size: u32,
) -> anyhow::Result<()> {
    eprintln!("[diagnose] ============== PIPELINE DIAGNOSTIC ==============");
    let w = model.pipeline.w();
    let h_m = model.pipeline.h_m();
    let w_rank = model.pipeline.rank();
    let w_svd = svd_rank(w, "W");
    let h_svd = svd_rank(h_m, "H_m (projected Hamiltonian)");
    eprintln!(
        "[diagnose] pipeline.w().nrows() = {}, pipeline.w().ncols() = rank = {w_rank}, pipeline.k2_dim() = {}",
        w.nrows(),
        model.pipeline.k2_dim()
    );
    eprintln!(
        "[diagnose] SVD ranks: W = {w_svd}, H_m = {h_svd} (vs nominal rank = {w_rank})"
    );
    // Diagonalise H_m to see how many distinct eigenvalues it has.
    {
        let m = h_m.clone();
        let mt = m.adjoint();
        let mtm = &mt * &m;
        let svd = nalgebra::linalg::SymmetricEigen::new(mtm);
        let mut eigs: Vec<f64> = svd
            .eigenvalues
            .iter()
            .map(|&c: &f64| c.max(0.0).sqrt())
            .collect();
        eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "[diagnose] |H_m| singular values (top 8): {:?}",
            &eigs[..eigs.len().min(8)]
        );
    }
    // Per-context fit. Build a context -> Vec<next-token> map in a
    // single O(n) pass instead of the previous O(n_candidates * n)
    // brute-force re-scan (50,000 candidates x 200,000 tokens = 10^10
    // comparisons, which is what made --diagnose take a very long
    // time on the real corpus). One pass fills the map; each
    // candidate context then looks up its occurrences in O(1)
    // amortized.
    let shard = qfm_text::Shard::open(train_shard_path, vocab_size)?;
    let tokens: Vec<u32> = shard.iter().take(200_000).collect();
    eprintln!(
        "[diagnose] per-context fit on first {} tokens of {}",
        tokens.len(),
        train_shard_path.display()
    );
    let n_max = model.cfg.n_orders;
    let mut ctx_map: rustc_hash::FxHashMap<Vec<u32>, Vec<u32>> = rustc_hash::FxHashMap::default();
    for i in n_max..tokens.len() {
        let ctx = tokens[i - n_max..i].to_vec();
        ctx_map.entry(ctx).or_default().push(tokens[i]);
    }
    // For each distinct context with >= 3 occurrences, compute QFM
    // dist, baseline dist, empirical hist, and metrics for both.
    let mut kls_qfm = Vec::new();
    let mut coss_qfm = Vec::new();
    let mut kls_base = Vec::new();
    let mut coss_base = Vec::new();
    let mut top1_qfm = 0u32;
    let mut top1_base = 0u32;
    let mut n_ctx = 0u32;
    let mut last_log = 0u32;
    // Miss attribution: when QFM's top-1 differs from the true
    // (independently, exact-match-counted) empirical top-1, is the
    // true token even *present* in any of this context's active
    // mode histograms? If it's absent, no weighting/mixing scheme
    // could have found it — that's a shared hashing/hist_cap
    // capacity artifact (would equally corrupt the baseline, since
    // it reads the same histograms), not a QFM-specific defect. If
    // it's present but QFM still didn't pick it, that's a genuine
    // weighting/smoothing miss.
    let hasher = model.registry.clone();
    let mut miss_absent = 0u32;
    let mut miss_present = 0u32;
    let mut miss_detail_logged = 0u32;
    let metrics = |emp: &[f64], dist: &[f64], vocab_size: usize| -> (f64, f64, usize) {
        let mut kl = 0.0_f64;
        for j in 0..vocab_size {
            if emp[j] > 0.0 && dist[j] > 0.0 {
                kl += emp[j] * (emp[j] / dist[j]).ln();
            }
        }
        let dot: f64 = (0..vocab_size).map(|j| emp[j] * dist[j]).sum();
        let ne: f64 = (0..vocab_size).map(|j| emp[j] * emp[j]).sum::<f64>().sqrt();
        let nd: f64 = (0..vocab_size).map(|j| dist[j] * dist[j]).sum::<f64>().sqrt();
        let cos = if ne > 0.0 && nd > 0.0 { dot / (ne * nd) } else { 0.0 };
        let mut argmax = 0usize;
        for (j, &p) in dist.iter().enumerate() {
            if p > dist[argmax] {
                argmax = j;
            }
        }
        (kl, cos, argmax)
    };
    for (ctx, nexts) in ctx_map.iter() {
        let total = nexts.len() as u32;
        if total < 3 {
            continue;
        }
        let mut emp = vec![0.0_f64; vocab_size as usize];
        for &t in nexts {
            emp[t as usize] += 1.0;
        }
        for x in emp.iter_mut() {
            *x /= total as f64;
        }
        let emp_argmax = {
            let mut best = 0usize;
            for (j, &p) in emp.iter().enumerate() {
                if p > emp[best] {
                    best = j;
                }
            }
            best
        };
        let qfm_dist = model.next_token_dist(ctx)?;
        let base_dist = baseline.next_token_dist(ctx);
        let (kl_q, cos_q, argmax_q) = metrics(&emp, &qfm_dist, vocab_size as usize);
        let (kl_b, cos_b, argmax_b) = metrics(&emp, &base_dist, vocab_size as usize);
        if argmax_q == emp_argmax {
            top1_qfm += 1;
        } else {
            // Miss: check whether the true token survives in any
            // active mode's histogram at all. For rev 36
            // (registry-based), an "unseen" context returns
            // [VACUUM_MODE] = [0], whose histogram is the unigram —
            // so the true token is always present in the vacuum's
            // histogram if it's anywhere in the vocabulary. This
            // is a structural change from the rev-35 hashed design
            // (where a context could map to a mode whose histogram
            // was already capped out). The miss is now always
            // "present but not chosen" (a genuine weighting/smoothing
            // miss).
            let active_modes = hasher.encode_modes(ctx);
            let present = active_modes.iter().any(|m| {
                model
                    .mode_hists
                    .get(m)
                    .map(|s| s.hist.iter().any(|&(tok, _)| tok as usize == emp_argmax))
                    .unwrap_or(false)
            });
            if present {
                miss_present += 1;
            } else {
                miss_absent += 1;
            }
            if miss_detail_logged < 8 {
                eprintln!(
                    "[diagnose]   MISS ctx={:?} true={} present_in_any_active_mode_hist={}",
                    &ctx[..ctx.len().min(4)],
                    emp_argmax,
                    present,
                );
                miss_detail_logged += 1;
            }
        }
        if argmax_b == emp_argmax {
            top1_base += 1;
        }
        kls_qfm.push(kl_q);
        coss_qfm.push(cos_q);
        kls_base.push(kl_b);
        coss_base.push(cos_b);
        n_ctx += 1;
        if n_ctx <= 5 {
            eprintln!(
                "[diagnose] ctx = {:?} (n={total}): emp top-1 = {} ({:.4}) | qfm: KL={kl_q:.4} cos={cos_q:.4} top1={} ({:.4}) | base: KL={kl_b:.4} cos={cos_b:.4} top1={} ({:.4})",
                &ctx[..ctx.len().min(4)],
                emp_argmax, emp[emp_argmax],
                argmax_q, qfm_dist[argmax_q],
                argmax_b, base_dist[argmax_b],
            );
        }
        if n_ctx - last_log >= 1000 {
            let mk_q: f64 = kls_qfm.iter().sum::<f64>() / kls_qfm.len() as f64;
            let mc_q: f64 = coss_qfm.iter().sum::<f64>() / coss_qfm.len() as f64;
            let mk_b: f64 = kls_base.iter().sum::<f64>() / kls_base.len() as f64;
            let mc_b: f64 = coss_base.iter().sum::<f64>() / coss_base.len() as f64;
            eprintln!(
                "[diagnose]   {n_ctx} contexts: qfm mean KL={mk_q:.4} cos={mc_q:.4} top1={}/{}={:.3} | base mean KL={mk_b:.4} cos={mc_b:.4} top1={}/{}={:.3}",
                top1_qfm, n_ctx, top1_qfm as f64 / n_ctx as f64,
                top1_base, n_ctx, top1_base as f64 / n_ctx as f64,
            );
            last_log = n_ctx;
        }
    }
    let mk_q: f64 = kls_qfm.iter().sum::<f64>() / kls_qfm.len().max(1) as f64;
    let mc_q: f64 = coss_qfm.iter().sum::<f64>() / coss_qfm.len().max(1) as f64;
    let mk_b: f64 = kls_base.iter().sum::<f64>() / kls_base.len().max(1) as f64;
    let mc_b: f64 = coss_base.iter().sum::<f64>() / coss_base.len().max(1) as f64;
    eprintln!(
        "[diagnose] FINAL ({n_ctx} distinct contexts with >= 3 occurrences):"
    );
    eprintln!(
        "[diagnose]   QFM:      mean KL = {mk_q:.4}, mean cos = {mc_q:.4}, top-1 hit = {top1_qfm}/{n_ctx} = {:.3}",
        top1_qfm as f64 / n_ctx.max(1) as f64,
    );
    eprintln!(
        "[diagnose]   Baseline: mean KL = {mk_b:.4}, mean cos = {mc_b:.4}, top-1 hit = {top1_base}/{n_ctx} = {:.3}",
        top1_base as f64 / n_ctx.max(1) as f64,
    );
    let n_miss = miss_absent + miss_present;
    eprintln!(
        "[diagnose]   QFM miss attribution ({n_miss} misses): true token absent from every active mode's histogram (registry/vacuum capacity limit, not QFM-specific) = {miss_absent}/{n_miss} = {:.3}; present but not chosen (genuine weighting/smoothing miss) = {miss_present}/{n_miss} = {:.3}",
        miss_absent as f64 / n_miss.max(1) as f64,
        miss_present as f64 / n_miss.max(1) as f64,
    );
    eprintln!("[diagnose] =================================================");
    Ok(())
}
