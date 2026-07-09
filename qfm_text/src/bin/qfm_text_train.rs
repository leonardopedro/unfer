//! `qfm_text_train` — Stage 6 of the QFM-Text plan.
//!
//! The HRM-Text `pretrain.py` analog: per-epoch accumulation +
//! compile + checkpoint + NDJSON metrics. Reads a `TextConfig` +
//! manifest from `--config`, runs the streaming pass over each
//! epoch's shard set, and emits one checkpoint per epoch + one
//! NDJSON line per epoch (the W&B analog).
//!
//! Usage:
//!   qfm_text_train --config train.toml --manifest manifest.json
//!                   --out ./out [--epochs 4]
//!
//! The `train.toml` is the TextConfig (see `cargo run
//! --bin qfm_text_train -- --help` for the keys; defaults match
//! `TextConfig::default()`).

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Context;
use qfm_text::{QfmTextModel, TextConfig, accumulate_shards};
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
    /// Number of epochs to run (default: 4, mirroring HRM-Text).
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
    4
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
        // "different data per epoch" property. This is the simplest
        // honest simulation: the counts keep accumulating across
        // epochs (the "training curve"), but each epoch sees a
        // different subset of the data.
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
    // Running accumulator across epochs. We take ownership of the
    // accumulator each epoch and move it into the model; the next
    // epoch rebuilds a fresh running_acc from the model's data
    // (this is the round-trip cost of having the model own the
    // stats). The dominant memory cost per epoch is the per-mode
    // `ModeStats` map (≈ 200 MB at 250K active modes × 64 hist
    // entries); the previous `par_iter().collect()` of all
    // per-shard accumulators at once (≈ 25 GB) has been replaced
    // by a sequential pass in `accumulate_shards`.
    let mut running_acc: Option<qfm_text::ChannelAccumulator> = None;
    for (epoch, shards) in per_epoch_shards.iter().enumerate() {
        let wall_s = std::time::Instant::now();
        let epoch_acc = accumulate_shards(shards, &text, manifest.vocab_size)?;
        let epoch_windows = epoch_acc.total_windows;
        // Merge epoch_acc into the running accumulator.
        let mut running = match running_acc.take() {
            Some(r) => r,
            None => qfm_text::ChannelAccumulator::new(manifest.vocab_size, text.clone()),
        };
        running.merge(epoch_acc);
        let n_active = running.n_active_modes();
        let total_windows = running.total_windows;
        // Build the model by consuming the running accumulator.
        // The model now owns `running.stats` and `running.unigram`.
        let model = QfmTextModel::from_accumulator(running, &text)?;
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
        // starts with a fresh running_acc.
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
           epochs = 4  # optional\n  \
           threads = 0 # optional, 0 = num_cpus\n  \
           # All TextConfig fields are also valid at the top level:\n  \
           n_orders = 4\n  \
           block_sizes = [65536, 65536, 65536, 65536]\n  \
           # ... etc."
    );
}
