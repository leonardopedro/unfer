//! R0.6 — the run manifest: the identity a table needs before anything in it
//! can be re-checked.
//!
//! Saving draws is not resumption. A `.safetensors` trajectory fixes the
//! numbers but not the run that made them: two runs at different code
//! revisions, datasets, temperatures or schedules can write the same tensor
//! names. This module writes the missing half beside the draws
//! (`{stem}.manifest.json`), in the plan's order: code revision and dirty-diff
//! identity, model/normalisation version, task and data hashes, anchor hash,
//! seeds, target and temperature, sampler and optimiser configuration, the
//! *evaluated* schedule, the metric/basis record, device/dtype, the warm-up and
//! thinning window, evaluation counts, wall time, peak memory and non-finite
//! events.
//!
//! Two things this module deliberately does **not** claim:
//!
//! * **Exact resumption.** RNG state, loader position and the adapt-warm-up
//!   state are not serialised, so a manifest plus a trajectory lets you audit a
//!   run, not continue it.
//! * **Clean-tree certainty.** A dirty hash identifies the diff that was
//!   *present*, not the binary that *ran*: `cargo` may have used a cached build
//!   from a different tree. The revision is what `git` reports from the crate
//!   directory at capture time.
//!
//! The JSON is written by hand (one flat object, stable key order) so the
//! crate keeps its dependency set and two runs of one configuration produce
//! byte-identical output apart from `created_unix`.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// FNV-1a 64, incremental — the identity function here, not a security
/// primitive. `new()`/`write()`/`finish()` so a caller can hash a sequence of
/// tensors without concatenating them.
pub struct Fnv(u64);

impl Fnv {
    pub const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    pub const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub fn new() -> Self {
        Fnv(Self::OFFSET)
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot FNV-1a 64.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = Fnv::new();
    h.write(bytes);
    h.finish()
}

/// Which source tree produced the numbers.
///
/// `revision` is `git rev-parse --short=12 HEAD`; `dirty_hash` hashes the
/// `git diff HEAD --binary` output **plus** `git status --porcelain`, so an
/// untracked file changes the identity even though it appears in no diff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeId {
    pub revision: String,
    pub dirty_hash: Option<u64>,
    pub dirty_files: usize,
}

impl CodeId {
    /// Capture from `root`. Never fails: outside a repository (or without
    /// `git`) the revision is the literal `"unknown"`, which is a recorded
    /// fact about the run rather than a silently empty field.
    pub fn from_git(root: &Path) -> Self {
        let rev = git(root, &["rev-parse", "--short=12", "HEAD"]);
        let status = git(root, &["status", "--porcelain"]);
        let diff = git(root, &["diff", "HEAD", "--binary"]);
        match (rev, status, diff) {
            (Some(rev), Some(status), Some(diff)) => {
                let dirty_files = status.lines().filter(|l| !l.trim().is_empty()).count();
                let mut bytes = diff.into_bytes();
                bytes.extend_from_slice(status.as_bytes());
                Self {
                    revision: rev,
                    dirty_hash: (dirty_files > 0).then(|| fnv1a(&bytes)),
                    dirty_files,
                }
            }
            _ => Self {
                revision: "unknown".into(),
                dirty_hash: None,
                dirty_files: 0,
            },
        }
    }

    /// Capture for this crate's own directory — the value a run should record.
    pub fn capture() -> Self {
        Self::from_git(Path::new(env!("CARGO_MANIFEST_DIR")))
    }
}

/// Run `git -C root <args>`; `None` when the command is missing, fails, or
/// prints nothing useful.
fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim_end().to_string();
    if s.is_empty() && args.first() != Some(&"diff") {
        return None;
    }
    Some(s)
}

/// The model the manifest describes.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelId {
    pub variant: String,
    pub p: usize,
    pub dtype: String,
    pub device: String,
    /// Which normalisation derivative the forward/backward used —
    /// `NormGrad::Stopped` is a labelled surrogate, not the exact map.
    pub normalization: String,
}

/// Task and data identity. `hash` is a hash of the *materialised* tensors, so
/// a change in sampler draw order shows up even when `seed` does not.
#[derive(Clone, Debug, PartialEq)]
pub struct DataId {
    pub task: String,
    pub k: usize,
    pub n_train: usize,
    pub n_test: usize,
    pub seed: u64,
    pub hash: u64,
    /// Hash of the frozen anchor `θ_A` the adaptation started from.
    pub anchor_hash: Option<u64>,
}

/// The target, stated separately from the kernel. `lambda = 1` is the
/// likelihood target; any other value is a *different* distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetId {
    pub lambda: f64,
    pub prior: String,
    pub anneal_steps: usize,
}

/// Kernel configuration. `beta_floor` is [`crate::sampler::SgPcnlScheduler::BETA_FLOOR`],
/// recorded because it decides which schedule the mobility claim is about.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerId {
    pub kernel: String,
    pub beta_0: f64,
    pub gamma: f64,
    pub kappa: f64,
    pub beta_floor: f64,
    pub sigma: f64,
    pub steps: usize,
    pub burn: usize,
    pub thin: usize,
    pub chains: usize,
    pub batch_size: usize,
    /// 0 = Robbins–Monro; > 0 = cyclical cosine period.
    pub cyclical_period: usize,
}

/// R0.5 — one stretch of a trace, with its own target and step size.
///
/// `stationary` is the flag that decides whether an ESS/R̂ computed on this
/// stretch is a stationary-chain diagnostic. It is true only when the step
/// size is *identical* at both ends and the likelihood ramp finished before the
/// stretch began; a decaying `β` over the kept window makes it false even when
/// the decay is visually flat.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase {
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub lambda: f64,
    pub beta_start: f64,
    pub beta_end: f64,
    pub stationary: bool,
}

/// R0.5 — where adapt-then-freeze leaves the metric, and which metric it is.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricRecord {
    /// `"isotropic"`, `"diagonal-secant"`, `"installed-diagonal"`, …
    pub kind: String,
    /// Step at which adaptation stopped; `None` when it never started or has
    /// not finished (then `adapting` is true).
    pub freeze_step: Option<usize>,
    pub adapting: bool,
    pub dim: usize,
    pub hash: Option<u64>,
}

impl MetricRecord {
    /// No metric: the isotropic proposal.
    pub fn isotropic() -> Self {
        Self {
            kind: "isotropic".into(),
            freeze_step: None,
            adapting: false,
            dim: 0,
            hash: None,
        }
    }
}

/// Cost and health of the run: what a "per evaluation" or "per second" claim
/// is divided by, and whether anything went non-finite.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkId {
    pub forward_evals: u64,
    pub backward_evals: u64,
    pub wall_seconds: f64,
    /// Peak resident set size (Linux `VmHWM`); `None` where unavailable.
    pub peak_mem_mb: Option<f64>,
    pub nonfinite_events: u64,
}

impl WorkId {
    /// Peak RSS of the current process in MiB, from `/proc/self/status`.
    ///
    /// This is the *process* high-water mark, not the run's: a binary that
    /// runs several stages reports the max over all of them.
    pub fn peak_rss_mb() -> Option<f64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb / 1024.0)
    }
}

/// The manifest itself.
#[derive(Clone, Debug, PartialEq)]
pub struct RunManifest {
    pub tool: String,
    pub version: String,
    pub created_unix: u64,
    pub code: CodeId,
    pub model: ModelId,
    pub data: DataId,
    pub target: TargetId,
    pub sampler: SamplerId,
    pub phases: Vec<Phase>,
    pub metric: MetricRecord,
    pub work: WorkId,
}

impl RunManifest {
    /// Stamp a manifest with now.
    pub fn new(code: CodeId, model: ModelId, data: DataId) -> Self {
        let created_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            tool: "pcnl".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            created_unix,
            code,
            model,
            data,
            target: TargetId {
                lambda: 1.0,
                prior: "N(0, sigma^2 I)".into(),
                anneal_steps: 0,
            },
            sampler: SamplerId {
                kernel: "sg-pcnl-unadjusted".into(),
                beta_0: 0.0,
                gamma: 0.0,
                kappa: 0.5,
                beta_floor: crate::sampler::SgPcnlScheduler::BETA_FLOOR,
                sigma: 1.0,
                steps: 0,
                burn: 0,
                thin: 1,
                chains: 0,
                batch_size: 0,
                cyclical_period: 0,
            },
            phases: Vec::new(),
            metric: MetricRecord::isotropic(),
            work: WorkId {
                forward_evals: 0,
                backward_evals: 0,
                wall_seconds: 0.0,
                peak_mem_mb: WorkId::peak_rss_mb(),
                nonfinite_events: 0,
            },
        }
    }

    /// One flat JSON object, stable key order, non-finite reals as `null`.
    pub fn to_json(&self) -> String {
        let phases: Vec<String> = self
            .phases
            .iter()
            .map(|ph| {
                obj(&[
                    ("name", q(&ph.name)),
                    ("start", ph.start.to_string()),
                    ("end", ph.end.to_string()),
                    ("lambda", num(ph.lambda)),
                    ("beta_start", num(ph.beta_start)),
                    ("beta_end", num(ph.beta_end)),
                    ("stationary", ph.stationary.to_string()),
                ])
            })
            .collect();
        obj(&[
            ("tool", q(&self.tool)),
            ("version", q(&self.version)),
            ("created_unix", self.created_unix.to_string()),
            (
                "code",
                obj(&[
                    ("revision", q(&self.code.revision)),
                    ("dirty_hash", opt_u64(self.code.dirty_hash)),
                    ("dirty_files", self.code.dirty_files.to_string()),
                ]),
            ),
            (
                "model",
                obj(&[
                    ("variant", q(&self.model.variant)),
                    ("p", self.model.p.to_string()),
                    ("dtype", q(&self.model.dtype)),
                    ("device", q(&self.model.device)),
                    ("normalization", q(&self.model.normalization)),
                ]),
            ),
            (
                "data",
                obj(&[
                    ("task", q(&self.data.task)),
                    ("k", self.data.k.to_string()),
                    ("n_train", self.data.n_train.to_string()),
                    ("n_test", self.data.n_test.to_string()),
                    ("seed", self.data.seed.to_string()),
                    ("hash", self.data.hash.to_string()),
                    ("anchor_hash", opt_u64(self.data.anchor_hash)),
                ]),
            ),
            (
                "target",
                obj(&[
                    ("lambda", num(self.target.lambda)),
                    ("prior", q(&self.target.prior)),
                    ("anneal_steps", self.target.anneal_steps.to_string()),
                ]),
            ),
            (
                "sampler",
                obj(&[
                    ("kernel", q(&self.sampler.kernel)),
                    ("beta_0", num(self.sampler.beta_0)),
                    ("gamma", num(self.sampler.gamma)),
                    ("kappa", num(self.sampler.kappa)),
                    ("beta_floor", num(self.sampler.beta_floor)),
                    ("sigma", num(self.sampler.sigma)),
                    ("steps", self.sampler.steps.to_string()),
                    ("burn", self.sampler.burn.to_string()),
                    ("thin", self.sampler.thin.to_string()),
                    ("chains", self.sampler.chains.to_string()),
                    ("batch_size", self.sampler.batch_size.to_string()),
                    ("cyclical_period", self.sampler.cyclical_period.to_string()),
                ]),
            ),
            ("phases", arr(&phases)),
            (
                "metric",
                obj(&[
                    ("kind", q(&self.metric.kind)),
                    ("freeze_step", opt_usize(self.metric.freeze_step)),
                    ("adapting", self.metric.adapting.to_string()),
                    ("dim", self.metric.dim.to_string()),
                    ("hash", opt_u64(self.metric.hash)),
                ]),
            ),
            (
                "work",
                obj(&[
                    ("forward_evals", self.work.forward_evals.to_string()),
                    ("backward_evals", self.work.backward_evals.to_string()),
                    ("wall_seconds", num(self.work.wall_seconds)),
                    ("peak_mem_mb", opt_f64(self.work.peak_mem_mb)),
                    ("nonfinite_events", self.work.nonfinite_events.to_string()),
                ]),
            ),
        ])
    }

    /// Write `{dir}/{stem}.manifest.json`, creating `dir`. Returns the path.
    pub fn write(&self, dir: &Path, stem: &str) -> io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{stem}.manifest.json"));
        std::fs::write(&path, format!("{}\n", self.to_json()))?;
        Ok(path)
    }
}

fn obj(fields: &[(&str, String)]) -> String {
    let body: Vec<String> = fields
        .iter()
        .map(|(k, v)| format!("{}: {v}", q(k)))
        .collect();
    format!("{{{}}}", body.join(", "))
}

fn arr(items: &[String]) -> String {
    format!("[{}]", items.join(", "))
}

/// A JSON string with the two escapes a path or task name can actually carry,
/// plus control characters so the output is always parseable.
fn q(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// JSON has no NaN or infinity: a diagnostic that failed is recorded as
/// `null`, never as a number a reader would average.
fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:?}")
    } else {
        "null".into()
    }
}

fn opt_f64(v: Option<f64>) -> String {
    v.map(num).unwrap_or_else(|| "null".into())
}

fn opt_u64(v: Option<u64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "null".into())
}

fn opt_usize(v: Option<usize>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "null".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RunManifest {
        let mut m = RunManifest::new(
            CodeId {
                revision: "abc123def456".into(),
                dirty_hash: Some(7),
                dirty_files: 2,
            },
            ModelId {
                variant: "cosine/nGPT-lite".into(),
                p: 584,
                dtype: "f32".into(),
                device: "cpu".into(),
                normalization: "fro-stopped/v1".into(),
            },
            DataId {
                task: "modular {\"p\":8}".into(),
                k: 16,
                n_train: 1024,
                n_test: 256,
                seed: 20261020,
                hash: u64::MAX,
                anchor_hash: Some(3),
            },
        );
        m.phases = vec![
            Phase {
                name: "warmup".into(),
                start: 0,
                end: 5000,
                lambda: 1.0,
                beta_start: 0.1,
                beta_end: 0.0707,
                stationary: false,
            },
            Phase {
                name: "kept".into(),
                start: 5000,
                end: 10000,
                lambda: 1.0,
                beta_start: 0.0707,
                beta_end: 0.05,
                stationary: false,
            },
        ];
        m.metric = MetricRecord {
            kind: "diagonal-secant".into(),
            freeze_step: Some(200),
            adapting: false,
            dim: 584,
            hash: Some(9),
        };
        m.work.wall_seconds = 12.5;
        m.work.nonfinite_events = 1;
        m
    }

    #[test]
    fn fnv1a_matches_the_reference_vectors() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x85944171f73967e8);
        // incremental == one-shot
        let mut h = Fnv::new();
        h.write(b"foo");
        h.write(b"bar");
        assert_eq!(h.finish(), fnv1a(b"foobar"));
    }

    #[test]
    fn json_is_one_flat_object_with_every_required_field() {
        let j = sample().to_json();
        assert!(j.starts_with('{') && j.ends_with('}'), "{j}");
        // balance: the hand writer's one structural invariant
        assert_eq!(
            j.chars().filter(|c| *c == '{').count(),
            j.chars().filter(|c| *c == '}').count()
        );
        for needle in [
            "\"code\"",
            "\"revision\"",
            "\"dirty_hash\": 7",
            "\"dirty_files\": 2",
            "\"normalization\": \"fro-stopped/v1\"",
            "\"anchor_hash\": 3",
            "\"hash\": 18446744073709551615",
            "\"phases\"",
            "\"freeze_step\": 200",
            "\"peak_mem_mb\"",
            "\"nonfinite_events\": 1",
            "\"wall_seconds\": 12.5",
        ] {
            assert!(j.contains(needle), "missing {needle} in {j}");
        }
        // key order is fixed, so a two-run diff only shows values
        let code = j.find("\"code\"").unwrap();
        let model = j.find("\"model\"").unwrap();
        let data = j.find("\"data\"").unwrap();
        let target = j.find("\"target\"").unwrap();
        let sampler = j.find("\"sampler\"").unwrap();
        let phases = j.find("\"phases\"").unwrap();
        let metric = j.find("\"metric\"").unwrap();
        let work = j.find("\"work\"").unwrap();
        assert!(code < model && model < data && data < target);
        assert!(target < sampler && sampler < phases && phases < metric && metric < work);
    }

    #[test]
    fn doubles_are_escaped_and_nonfinite_becomes_null() {
        let mut m = sample();
        m.data.task = "say \"hi\"\tnow".into();
        m.work.wall_seconds = f64::NAN;
        m.work.peak_mem_mb = Some(f64::INFINITY);
        m.code.dirty_hash = None;
        let j = m.to_json();
        assert!(j.contains(r#""task": "say \"hi\"\tnow""#), "{j}");
        assert!(j.contains("\"wall_seconds\": null"), "{j}");
        assert!(j.contains("\"peak_mem_mb\": null"), "{j}");
        assert!(j.contains("\"dirty_hash\": null"), "{j}");
        assert!(!j.contains("NaN") && !j.contains("inf"), "{j}");
    }

    #[test]
    fn write_puts_the_manifest_next_to_the_draws() {
        let dir = std::env::temp_dir().join(format!("pcnl-manifest-{}", std::process::id()));
        let m = sample();
        let path = m.write(&dir, "cosine_nGPT-lite_chain0").unwrap();
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            "cosine_nGPT-lite_chain0.manifest.json"
        );
        let back = std::fs::read_to_string(&path).unwrap();
        assert_eq!(back.trim_end(), m.to_json());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn capture_degrades_to_unknown_outside_a_repository() {
        // A path that does not exist cannot be a repository and cannot be
        // walked up from, so `git` fails and the manifest records its absence.
        let id = CodeId::from_git(Path::new("/nonexistent/pcnl-manifest-probe"));
        assert_eq!(id.revision, "unknown");
        assert_eq!(id.dirty_hash, None);
        assert_eq!(id.dirty_files, 0);
    }

    #[test]
    fn capture_in_this_repository_records_a_revision() {
        let id = CodeId::capture();
        assert_ne!(id.revision, "unknown", "crate dir should be a git repo");
        assert!(
            id.revision.len() >= 7 && id.revision.chars().all(|c| c.is_ascii_hexdigit()),
            "revision {} is not a short hash",
            id.revision
        );
        // A clean or dirty tree are both fine; the flag must agree with the hash.
        assert_eq!(id.dirty_hash.is_some(), id.dirty_files > 0);
    }
}
