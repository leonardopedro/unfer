//! The publish worker — `brief-ideas`' `ZenodoWorker`, ported.
//!
//! Publishing used to happen inside the HTTP request: the browser waited on a
//! Zenodo round-trip that can take tens of seconds, and a failure mid-push was
//! the client's problem. Here the request hands the job to a **bounded queue**
//! and answers `202`; one worker thread performs the push, retries with
//! backoff, and records the outcome in the audit log. The queue bound matters:
//! open submission plus an unbounded queue is just a slower denial of service,
//! so a full queue is refused rather than buffered.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread;
use std::time::Duration;

use serde_json::json;

use crate::audit::Audit;
use crate::store::{Index, STATE_PUBLISH_FAILED, STATE_PUBLISHED, StoreConfig, Zenodo};

/// Jobs waiting to be pushed. Small on purpose: the worker is one thread and
/// Zenodo is the bottleneck, so a deep queue only hides backpressure.
pub const QUEUE_CAPACITY: usize = 32;

/// Attempts per job before the proof is marked `publish-failed`. Zenodo is
/// usually briefly unavailable rather than wrong, so retrying is right; retrying
/// forever is not.
pub const MAX_ATTEMPTS: u32 = 5;

/// First retry delay; doubles each attempt, capped at [`BACKOFF_CAP`].
const BACKOFF_BASE: Duration = Duration::from_millis(500);
const BACKOFF_CAP: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct PublishJob {
    pub project: String,
    pub proof: String,
    pub attempt: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum QueueError {
    /// The queue is full — the caller should shed load, not wait.
    Full,
    /// The worker is gone; publishing is unavailable in this process.
    Stopped,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueueError::Full => write!(f, "publish queue is full"),
            QueueError::Stopped => write!(f, "publish worker is not running"),
        }
    }
}

/// Handle to the queue. `Clone` is cheap (`SyncSender` behind an `Arc`-free
/// clone), so the web config can carry one.
#[derive(Debug)]
pub struct Publisher {
    tx: SyncSender<PublishJob>,
}

impl Publisher {
    /// Start the worker thread. The thread lives for the process's lifetime.
    pub fn spawn(store: StoreConfig, audit: Arc<Audit>) -> Self {
        let (tx, rx) = sync_channel(QUEUE_CAPACITY);
        thread::Builder::new()
            .name("mm0web-publisher".into())
            .spawn(move || worker(rx, store, audit))
            .expect("spawn publish worker");
        Self { tx }
    }

    /// Hand a job over without blocking. `Full` means the caller should answer
    /// `503`/`429` rather than wait — the proof is verified and stored either
    /// way, so shedding is safe.
    pub fn enqueue(&self, project: &str, proof: &str) -> Result<(), QueueError> {
        let job = PublishJob {
            project: project.to_string(),
            proof: proof.to_string(),
            attempt: 0,
        };
        match self.tx.try_send(job) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(QueueError::Full),
            Err(TrySendError::Disconnected(_)) => Err(QueueError::Stopped),
        }
    }
}

fn worker(rx: Receiver<PublishJob>, store: StoreConfig, audit: Arc<Audit>) {
    while let Ok(job) = rx.recv() {
        let mut job = job;
        loop {
            match attempt(&store, &job, &audit) {
                Ok(()) => break,
                Err(e) => {
                    let _ = audit.record(
                        "publish-failed",
                        &job.project,
                        &job.proof,
                        json!({"error": e, "attempt": job.attempt + 1}),
                    );
                    job.attempt += 1;
                    if job.attempt >= MAX_ATTEMPTS {
                        mark_failed(&store, &job);
                        break;
                    }
                    thread::sleep(backoff(job.attempt));
                }
            }
        }
    }
}

fn backoff(attempt: u32) -> Duration {
    BACKOFF_BASE
        .saturating_mul(1u32 << attempt.min(16))
        .min(BACKOFF_CAP)
}

/// One push attempt. Idempotent: a proof already `published` is a no-op, which
/// is what makes a duplicate job (crash before the queue ack, or a re-queue)
/// harmless.
fn attempt(store: &StoreConfig, job: &PublishJob, audit: &Audit) -> Result<(), String> {
    let mut index = Index::load(&store.state_dir, &job.project);
    let proof = index
        .proofs
        .get(&job.proof)
        .ok_or_else(|| format!("proof {} is not in the index", job.proof))?;
    if proof.state == STATE_PUBLISHED && proof.record.is_some() {
        return Ok(());
    }

    let zenodo = Zenodo::new(store.zenodo.clone(), store.state_dir.clone());
    let outcome = zenodo.push(&index)?;

    if let Some(p) = index.proofs.get_mut(&job.proof) {
        p.record = Some(outcome.record_id);
        p.doi = outcome.doi.clone();
        p.state = STATE_PUBLISHED.to_string();
    }
    index
        .save(&store.state_dir)
        .map_err(|e| format!("cannot save index after push: {e}"))?;

    audit.record(
        "published",
        &job.project,
        &job.proof,
        json!({
            "record": outcome.record_id,
            "doi": outcome.doi,
            "url": outcome.url,
            "kind": outcome.kind,
            "file": outcome.filename,
        }),
    )?;
    Ok(())
}

/// Give up: record the terminal state so the status page stops refreshing and
/// says what happened.
fn mark_failed(store: &StoreConfig, job: &PublishJob) {
    let mut index = Index::load(&store.state_dir, &job.project);
    let Some(p) = index.proofs.get_mut(&job.proof) else {
        return;
    };
    p.state = STATE_PUBLISH_FAILED.to_string();
    let _ = index.save(&store.state_dir);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_is_capped() {
        assert_eq!(backoff(0), BACKOFF_BASE);
        assert_eq!(backoff(1), BACKOFF_BASE * 2);
        assert!(backoff(3) <= BACKOFF_CAP);
        assert_eq!(
            backoff(64),
            BACKOFF_CAP,
            "absurd attempt counts stay bounded"
        );
    }

    #[test]
    fn a_full_queue_is_refused_not_buffered() {
        // No worker attached: the channel fills after its first send.
        let (tx, _rx) = sync_channel::<PublishJob>(1);
        let p = Publisher { tx };
        assert!(p.enqueue("unfer", "a").is_ok());
        assert_eq!(p.enqueue("unfer", "b"), Err(QueueError::Full));
    }

    #[test]
    fn queue_errors_explain_themselves() {
        assert!(QueueError::Full.to_string().contains("full"));
        assert!(QueueError::Stopped.to_string().contains("not running"));
    }

    #[test]
    fn the_worker_marks_a_proof_failed_after_the_last_attempt() {
        let dir = std::env::temp_dir().join(format!(
            "mm0web-pubfail-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut index = Index::load(&dir, "unfer");
        let id = index.upsert("t", "sort Prop\n", "");
        index.save(&dir).unwrap();

        let job = PublishJob {
            project: "unfer".into(),
            proof: id.clone(),
            attempt: MAX_ATTEMPTS,
        };
        mark_failed(
            &StoreConfig {
                state_dir: dir.clone(),
                zenodo: Default::default(),
            },
            &job,
        );

        let back = Index::load(&dir, "unfer");
        assert_eq!(back.proofs[&id].state, STATE_PUBLISH_FAILED);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn queued_is_a_state_the_web_layer_can_set() {
        // the vocabulary the publisher relies on is the store's, not a copy
        assert_eq!(crate::store::STATE_QUEUED, "queued");
        assert_eq!(STATE_PUBLISHED, "published");
        assert_eq!(STATE_PUBLISH_FAILED, "publish-failed");
    }
}
