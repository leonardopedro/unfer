//! `qfm_text_train` — Stage 6 of the QFM-Text plan.
//!
//! The HRM-Text `pretrain.py` analog: per-epoch accumulation +
//! compile + checkpoint + NDJSON metrics. Reads a `TextConfig` +
//! manifest from `--config`, runs the streaming pass over each
//! epoch's shard set, and emits one checkpoint per epoch + one
//! NDJSON line per epoch (the W&B analog).
//!
//! Usage:
//!   qfm_text_train --config train.toml
//!
//! The `train.toml` is the TextConfig (see `cargo run
//! --bin qfm_text_train -- --help` for the keys; defaults match
//! `TextConfig::default()`).

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Context;
use qfm_text::{
    ContextRegistry, Encoder, OrderHasher, QfmTextModel, TextConfig, accumulate_shards,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TrainConfig {
    /// The text model config (the TOML inner object).
    #[serde(flatten)]
    text: TextConfig,
    /// Path to the shard manifest.
    manifest_path: PathBuf,
    /// Output directory (created if missing).
    out_dir: PathBuf,
    /// Number of epochs to run (default: 1, mirroring the rev-36
    /// single-pass design: the registry is the side product of
    /// training and a per-epoch fresh registry is the simplest way
    /// to keep the registry consistent with the per-epoch
    /// accumulator's stats).
    #[serde(default = "default_epochs")]
    epochs: usize,
    /// Number of threads (default: 0 = num_cpus). Reserved for
    /// future use; current implementation uses rayon's default
    /// thread pool.
    #[serde(default)]
    #[allow(dead_code)]
    threads: usize,
}

fn default_epochs() -> usize {
    1
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut config_path: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => {
                config_path = Some(PathBuf::from(args.next().context("--config needs value")?));
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }
    let config_path = config_path.context("--config is required")?;
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let cfg: TrainConfig = toml::from_str(&raw).context("parse TOML")?;
    let text = cfg.text;
    let out = cfg.out_dir;
    std::fs::create_dir_all(&out).with_context(|| format!("create {}", out.display()))?;
    let manifest =
        qfm_text::Manifest::read(&cfg.manifest_path).context("read shard manifest")?;
    let shard_dir = cfg
        .manifest_path
        .parent()
        .context("manifest_path must have a parent directory")?;
    let mut per_epoch_shards: Vec<Vec<PathBuf>> = Vec::with_capacity(cfg.epochs);
    for epoch in 0..cfg.epochs {
        // HRM-Text's stratified sampling produces different shard
        // content per epoch. Our shard manifest is the same per
        // epoch (the underlying corpus is one file); we rotate the
        // shard set deterministically per epoch to give the
        // "different data per epoch" property.
        let n = manifest.shards.len();
        let shift = epoch % n;
        let rotated: Vec<PathBuf> = (0..n)
            .map(|i| manifest.shard_path(shard_dir, (i + shift) % n))
            .collect();
        per_epoch_shards.push(rotated);
    }
    // Open metrics.ndjson in append mode.
    let metrics_path = out.join("metrics.ndjson");
    let mut metrics = File::create(&metrics_path)
        .with_context(|| format!("create {}", metrics_path.display()))?;
    // Per-epoch fresh registry: the registry is a side product of
    // training, and the per-epoch accumulator's stats are keyed by
    // the registry's mode indices. A single shared registry across
    // epochs would mean epoch-N's stats are keyed by mode indices
    // assigned at epoch-0's first pass — the freshest registry per
    // epoch is the simplest invariant. The model is then built from
    // the (per-epoch) accumulator + (per-epoch) registry.
    let mut registry_per_epoch: Option<ContextRegistry> = None;
    for (epoch, shards) in per_epoch_shards.iter().enumerate() {
        let wall_s = std::time::Instant::now();
        let mut registry = registry_per_epoch
            .take()
            .unwrap_or_else(|| ContextRegistry::new(text.n_orders));
        let mut encoder = if text.use_registry_encoder {
            Encoder::Registry(registry.clone())
        } else {
            Encoder::Hasher(OrderHasher::new(text.clone()))
        };
        let epoch_acc = accumulate_shards(shards, &mut encoder, &text, manifest.vocab_size)?;
        // Sync the (possibly grown) registry back from the encoder
        // if we used the `Registry` variant. (The `Hasher` variant
        // is read-only and needs no sync.)
        if let Some(r) = encoder.as_registry() {
            registry.clone_from(r);
        }
        let epoch_windows = epoch_acc.total_windows;
        let n_active = epoch_acc.n_active_modes();
        let total_windows = epoch_acc.total_windows;
        // Build the model by consuming the per-epoch accumulator
        // and the matching registry. The model now owns
        // `epoch_acc.stats` and `epoch_acc.unigram` and the
        // registry's mode-to-context map.
        let model = QfmTextModel::from_accumulator(epoch_acc, registry, &text)?;
        let ckpt_path = out.join(format!("checkpoint_epoch{epoch}.qfm"));
        model.save(&ckpt_path)?;
        let wall_s = wall_s.elapsed().as_secs_f64();
        let line = serde_json::json!({
            "epoch": epoch,
            "wall_s": wall_s,
            "n_windows": epoch_windows,
            "n_active_modes": n_active,
            "total_windows": total_windows,
            "out_dir": out.display().to_string(),
        });
        writeln!(metrics, "{line}")?;
        eprintln!("[epoch {epoch}] wall_s = {wall_s:.2}, n_windows = {epoch_windows}, n_active = {n_active}, total = {total_windows}");
        // The model is dropped at end of scope. The next epoch
        // starts with a fresh registry.
    }
    eprintln!("done. metrics -> {}", metrics_path.display());
    Ok(())
}

fn print_help() {
    eprintln!(
        "qfm_text_train --config <train.toml> [--help]\n\n\
         The TOML config is:\n  \
           manifest_path = \"...\"\n  \
           out_dir = \"...\"\n  \
           epochs = 1  # optional (rev 36 default; one pass, fresh registry per epoch)\n  \
           threads = 0 # optional, 0 = num_cpus\n  \
           # All TextConfig fields are also valid at the top level:\n  \
           n_orders = 4\n  \
           # ... etc.\n\n\
         Rev 37: encoder is selected by `use_registry_encoder` in\n  \
         the TextConfig (default `false` = rev 35 `OrderHasher` with\n  \
         `block_sizes` and `salts`; `true` = rev 36 `ContextRegistry`).\n  \
         Decoder is the rev 35 dense W matrix; the optional oxieml\n  \
         SymRegEngine decoder is being added in rev 37."
    );
}
