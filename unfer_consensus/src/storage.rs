//! D07: one durable application transition, with an explicit crash protocol.
//!
//! The network backend used to persist its state as a bundle of values sampled
//! independently (`bank` under one lock, `pending` under another, counters from
//! atomics). A crash between two of those reads produced a snapshot that never
//! existed, and a crash that lost the rename produced no snapshot at all.
//!
//! This module replaces that with one append-only log of framed, checksummed
//! records and one writer:
//!
//! ```text
//! | magic (4) | version (1) | length (4) | crc32 (4) | payload (length) |
//! ```
//!
//! The protocol for every record is **write the whole frame, then `sync_data`,
//! and only then is the transition published**. A record is published when its
//! frame is complete and its checksum matches; anything else at the tail is a
//! torn write and is truncated, because a crash during `write_all` may leave a
//! partial frame. A *complete* frame with a bad magic, version or checksum is
//! not a torn tail: it is corruption, and recovery refuses rather than guessing.
//!
//! Two record kinds share the log, because they are two different facts with the
//! same writer:
//!
//! * [`TransitionRecord`] — what applying one committed slot did (command
//!   digest, decision/result, exactly-once guard, resulting state version);
//! * [`RequestRecord`] — the binding from a client's durable request id to the
//!   submission and settled result it produced. It is written when the outcome
//!   is settled and *before* the client is told, which is what makes an
//!   acknowledged result survive a restart.
//!
//! Recovery order: transitions are replayed in slot order (a slot gap or a
//! repeated slot in the log is corruption), then request bindings are applied to
//! the admission book. A request record that names a slot the log never recorded
//! is corruption too.
//!
//! What this does **not** provide: atomicity with the consensus library's own
//! runtime store. `rust-quepaxa` persists its recorder/runtime state separately
//! and cannot commit it in the same transaction as application state, so the
//! recovery order is "runtime store first, then this log, replay by slot", and a
//! runtime store that is ahead of this log is handled by replay (execution is
//! idempotent by slot). A checkpoint whose slot disagrees with this log is
//! rejected rather than merged.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use unfer_protocol::Code;
use unfer_protocol::submission::SubmissionId;

use crate::idempotency::{AdmittedOutcome, RequestId};

const MAGIC: u32 = 0x55_4E_46_45; // "UNFE"
const VERSION: u8 = 1;
/// magic + version + length + crc32
const HEADER_LEN: usize = 4 + 1 + 4 + 4;
/// A record larger than this is not a record we wrote; refuse to read it.
const MAX_PAYLOAD: u32 = 16 * 1024 * 1024;

/// CRC-32 (IEEE, reflected, poly 0xEDB88320), computed without a table: this is
/// a torn-write detector, not a cryptographic checksum.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// What applying a committed entry produced. A refusal is a stable verdict and
/// is part of the applied history; a halt is never recorded here, because a
/// halted node has no transition to record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum RecordedOutcome {
    Applied,
    Refused { code: u32, message: String },
}

impl RecordedOutcome {
    pub fn refused(code: Code, message: impl Into<String>) -> Self {
        RecordedOutcome::Refused {
            code: code.raw(),
            message: message.into(),
        }
    }
}

/// The exactly-once guard a transition installed (H7 content key + its slot).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupUpdate {
    pub key: String,
    pub slot: u64,
}

/// One durable application transition: everything needed to say, after a crash,
/// what the node decided at a slot and what version of its state that produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionRecord {
    pub slot: u64,
    /// The resulting state version. Equal to `slot` for a contiguous local
    /// application, kept separate so an independent version can be recorded
    /// without changing the wire shape.
    pub state_version: u64,
    /// SHA-256 of the canonical command bytes.
    pub command_digest: [u8; 32],
    pub outcome: RecordedOutcome,
    pub dedup: Option<DedupUpdate>,
}

/// The durable binding from a client request id to its submission and settled
/// result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestRecord {
    pub id: RequestId,
    pub digest: [u8; 32],
    pub submission: SubmissionId,
    pub outcome: AdmittedOutcome,
    /// The slot the submission settled at, when it committed.
    pub slot: Option<u64>,
}

/// D14: one job-fence transition, durable before it is published.
///
/// The record names the *resulting* fence rather than the transition that made
/// it, so recovery is "install the last record per resource" and needs no replay
/// rules. A claim, a renewal, a release, a fire and a disable are distinguished
/// by `state`, which is also what the job itself recovers to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FenceRecord {
    pub deployment: String,
    pub resource: String,
    pub epoch: u64,
    pub generation: u64,
    pub holder: Option<String>,
    /// What the transition left the job in. Recovery needs this and not only the
    /// fence: a released job is retried by the next claim, a fired one is not,
    /// and the fence alone cannot tell them apart.
    pub state: crate::jobs::ClaimState,
}

/// A record in the log. Both kinds are written by the same writer in one order,
/// so "the request record was durable" and "the transition was durable" have a
/// well-defined sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum JournalRecord {
    Transition(TransitionRecord),
    Request(RequestRecord),
    Fence(FenceRecord),
}

/// Why a write to the log failed. The distinction matters: an `Io` failure
/// before the frame was written changed nothing, while an `Uncertain` failure
/// (the frame was written but the sync failed) leaves the record's durability
/// unknown, so the node must stop rather than report a verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    /// The transition was not written; the caller may retry or halt with a
    /// definite "not recorded".
    Io(String),
    /// The frame reached the file but `sync_data` failed: it may or may not be
    /// durable. Never report this as "definitely not recorded".
    Uncertain(String),
    /// The existing log is not a log this module wrote.
    Corrupt { offset: u64, reason: String },
    /// The record could not be encoded.
    Encode(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::Io(reason) => write!(f, "storage write failed: {reason}"),
            StorageError::Uncertain(reason) => {
                write!(
                    f,
                    "storage sync failed after the write (durability unknown): {reason}"
                )
            }
            StorageError::Corrupt { offset, reason } => {
                write!(f, "corrupt log at offset {offset}: {reason}")
            }
            StorageError::Encode(reason) => write!(f, "could not encode the record: {reason}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl StorageError {
    pub fn code(&self) -> Code {
        Code::STORAGE_FAILED
    }
}

/// What recovery found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Recovery {
    pub records: Vec<JournalRecord>,
    /// Bytes of an incomplete trailing frame that were discarded. Non-zero means
    /// a crash interrupted a write, not that the log is damaged.
    pub torn_tail_bytes: u64,
    /// The last transition slot found.
    pub last_slot: Option<u64>,
}

/// A single-writer, append-only transition log.
///
/// `append` takes `&mut self` and the type is not `Clone`, so two threads cannot
/// hold a writer: the durability order is the caller's order.
#[derive(Debug)]
pub struct TransitionLog {
    path: PathBuf,
    file: File,
    len: u64,
    fault: Option<FaultPoint>,
}

/// Fault injection for the crash tests (plan D07 acceptance): each boundary of
/// the crash protocol can be made to fail.
///
/// This is a test seam, not a runtime switch: nothing outside a test ever calls
/// [`TransitionLog::inject`], and an uninjected log takes the same path it always
/// did. It is compiled unconditionally because the deterministic scheduler
/// (plan D12) lives in the library and must reach it from an integration test.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultPoint {
    /// Fail before any byte is written.
    BeforeWrite,
    /// Write half of the frame, then fail (a torn tail on recovery).
    HalfWrite,
    /// Write the whole frame, then fail the sync (uncertain durability).
    AfterWrite,
}

impl TransitionLog {
    /// Open (or create) the log for appending, recovering first. A torn tail is
    /// truncated and reported; corruption is an error and nothing is truncated.
    pub fn open(path: impl Into<PathBuf>) -> Result<(Self, Recovery), StorageError> {
        let path = path.into();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|error| {
                StorageError::Io(format!("could not create {}: {error}", parent.display()))
            })?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| {
                StorageError::Io(format!("could not open {}: {error}", path.display()))
            })?;
        let mut log = Self {
            path,
            file,
            len: 0,
            fault: None,
        };
        let recovery = log.recover()?;
        Ok((log, recovery))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Scan the log from the start.
    ///
    /// * A short or empty frame at EOF is a torn tail: the file is truncated back
    ///   to the last complete frame, since a crash during `write_all` may have
    ///   left a partial record and that record was never published.
    /// * A complete frame with a bad magic, version, length or checksum is
    ///   corruption: the file is left untouched and recovery fails.
    fn recover(&mut self) -> Result<Recovery, StorageError> {
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|error| StorageError::Io(format!("could not seek: {error}")))?;
        let mut bytes = Vec::new();
        self.file
            .read_to_end(&mut bytes)
            .map_err(|error| StorageError::Io(format!("could not read the log: {error}")))?;

        let mut records = Vec::new();
        let mut offset = 0usize;
        let mut last_slot = None;
        while offset < bytes.len() {
            let remaining = &bytes[offset..];
            if remaining.len() < HEADER_LEN {
                break; // torn header
            }
            let magic = u32::from_le_bytes(remaining[0..4].try_into().unwrap());
            let version = remaining[4];
            let length = u32::from_le_bytes(remaining[5..9].try_into().unwrap()) as usize;
            let checksum = u32::from_le_bytes(remaining[9..13].try_into().unwrap());
            if magic != MAGIC || version != VERSION || length > MAX_PAYLOAD as usize {
                return Err(StorageError::Corrupt {
                    offset: offset as u64,
                    reason: format!(
                        "frame header is not a v{VERSION} UNFE record (magic {magic:#x}, length {length})"
                    ),
                });
            }
            if remaining.len() < HEADER_LEN + length {
                break; // torn payload
            }
            let payload = &remaining[HEADER_LEN..HEADER_LEN + length];
            if crc32(payload) != checksum {
                return Err(StorageError::Corrupt {
                    offset: offset as u64,
                    reason: "record checksum does not match its payload".to_string(),
                });
            }
            let record: JournalRecord =
                serde_json::from_slice(payload).map_err(|error| StorageError::Corrupt {
                    offset: offset as u64,
                    reason: format!("record payload does not decode: {error}"),
                })?;
            if let JournalRecord::Transition(t) = &record {
                match last_slot {
                    Some(previous) if t.slot <= previous => {
                        return Err(StorageError::Corrupt {
                            offset: offset as u64,
                            reason: format!(
                                "transition at slot {} follows slot {previous}: slots must increase",
                                t.slot
                            ),
                        });
                    }
                    _ => last_slot = Some(t.slot),
                }
            }
            if let JournalRecord::Request(request) = &record
                && let Some(slot) = request.slot
            {
                let known = last_slot.map(|last| slot <= last).unwrap_or(false);
                if !known {
                    return Err(StorageError::Corrupt {
                        offset: offset as u64,
                        reason: format!(
                            "request record names slot {slot}, but the log's last transition is {last_slot:?}"
                        ),
                    });
                }
            }
            records.push(record);
            offset += HEADER_LEN + length;
        }

        let torn_tail_bytes = (bytes.len() - offset) as u64;
        if torn_tail_bytes > 0 {
            self.file
                .set_len(offset as u64)
                .and_then(|()| self.file.sync_all())
                .map_err(|error| {
                    StorageError::Io(format!("could not truncate the torn tail: {error}"))
                })?;
        }
        self.len = offset as u64;
        self.file
            .seek(SeekFrom::Start(self.len))
            .map_err(|error| StorageError::Io(format!("could not seek to the end: {error}")))?;
        Ok(Recovery {
            records,
            torn_tail_bytes,
            last_slot,
        })
    }

    /// Append one record: write the complete frame, synchronize, publish.
    /// Nothing is published until the sync returns, so a reader never sees a
    /// record that is not durable.
    pub fn append(&mut self, record: &JournalRecord) -> Result<(), StorageError> {
        let payload =
            serde_json::to_vec(record).map_err(|error| StorageError::Encode(error.to_string()))?;
        if payload.len() > MAX_PAYLOAD as usize {
            return Err(StorageError::Encode(format!(
                "record of {} bytes exceeds the {MAX_PAYLOAD}-byte limit",
                payload.len()
            )));
        }
        let mut frame = Vec::with_capacity(HEADER_LEN + payload.len());
        frame.extend_from_slice(&MAGIC.to_le_bytes());
        frame.push(VERSION);
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&crc32(&payload).to_le_bytes());
        frame.extend_from_slice(&payload);

        if let Some(fault) = self.fault.take() {
            match fault {
                FaultPoint::BeforeWrite => {
                    return Err(StorageError::Io("injected: before write".to_string()));
                }
                FaultPoint::HalfWrite => {
                    let half = frame.len() / 2;
                    let _ = self.file.write_all(&frame[..half]);
                    let _ = self.file.sync_all();
                    return Err(StorageError::Io("injected: torn write".to_string()));
                }
                FaultPoint::AfterWrite => {
                    let _ = self.file.write_all(&frame);
                    return Err(StorageError::Uncertain(
                        "injected: sync failed after write".to_string(),
                    ));
                }
            }
        }

        if let Err(error) = self.file.write_all(&frame) {
            // Best effort: drop whatever part of the frame reached the file so
            // the next append starts on a frame boundary. A torn tail is
            // recoverable either way, but leaving it makes the next reader pay
            // for it.
            let _ = self.file.set_len(self.len);
            let _ = self.file.seek(SeekFrom::Start(self.len));
            return Err(StorageError::Io(error.to_string()));
        }
        if let Err(error) = self.file.sync_data() {
            return Err(StorageError::Uncertain(error.to_string()));
        }
        self.len += frame.len() as u64;
        Ok(())
    }

    /// Inject a failure at the next boundary (tests only).
    #[doc(hidden)]
    pub fn inject(&mut self, fault: FaultPoint) {
        self.fault = Some(fault);
    }

    /// Bytes currently published.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "unfer-storage-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        dir.join("transitions.log")
    }

    fn transition(slot: u64) -> JournalRecord {
        JournalRecord::Transition(TransitionRecord {
            slot,
            state_version: slot,
            command_digest: [slot as u8; 32],
            outcome: RecordedOutcome::Applied,
            dedup: None,
        })
    }

    fn request(slot: u64) -> JournalRecord {
        JournalRecord::Request(RequestRecord {
            id: RequestId::new("f", "did:unfer:alice", 1, slot),
            digest: [slot as u8; 32],
            submission: SubmissionId(slot),
            outcome: AdmittedOutcome::Accepted {
                submission: SubmissionId(slot),
                slot: unfer_protocol::submission::Slot(slot),
            },
            slot: Some(slot),
        })
    }

    #[test]
    fn records_survive_a_reopen() {
        let path = temp_path("reopen");
        let (mut log, recovery) = TransitionLog::open(&path).unwrap();
        assert!(recovery.records.is_empty());
        log.append(&transition(1)).unwrap();
        log.append(&request(1)).unwrap();
        log.append(&transition(2)).unwrap();
        drop(log);

        let (_, recovery) = TransitionLog::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 3);
        assert_eq!(recovery.last_slot, Some(2));
        assert_eq!(recovery.torn_tail_bytes, 0);
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_torn_tail_is_discarded_and_the_log_stays_usable() {
        let path = temp_path("torn");
        let (mut log, _) = TransitionLog::open(&path).unwrap();
        log.append(&transition(1)).unwrap();
        log.inject(FaultPoint::HalfWrite);
        assert!(log.append(&transition(2)).is_err());
        drop(log);

        let (mut log, recovery) = TransitionLog::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 1, "the published record survives");
        assert!(
            recovery.torn_tail_bytes > 0,
            "the interrupted frame is a torn tail, not corruption"
        );
        log.append(&transition(3)).unwrap();
        drop(log);
        let (_, recovery) = TransitionLog::open(&path).unwrap();
        assert_eq!(
            recovery.records.len(),
            2,
            "the log continues on a frame boundary"
        );
        assert_eq!(recovery.last_slot, Some(3));
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_complete_frame_with_a_bad_checksum_is_corruption_not_a_torn_tail() {
        let path = temp_path("corrupt");
        let (mut log, _) = TransitionLog::open(&path).unwrap();
        log.append(&transition(1)).unwrap();
        drop(log);

        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();

        match TransitionLog::open(&path) {
            Err(StorageError::Corrupt { .. }) => {}
            other => panic!("a damaged complete frame must be reported, got {other:?}"),
        }
        assert_eq!(
            fs::read(&path).unwrap(),
            bytes,
            "recovery must not truncate a damaged record as if it were torn"
        );
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn an_uncertain_write_is_reported_as_uncertain() {
        let path = temp_path("uncertain");
        let (mut log, _) = TransitionLog::open(&path).unwrap();
        log.inject(FaultPoint::AfterWrite);
        let error = log.append(&transition(1)).unwrap_err();
        assert!(
            matches!(error, StorageError::Uncertain(_)),
            "a failed sync after a complete write must not be reported as a definite failure"
        );
        // The frame is complete, so recovery publishes it: the write did land.
        drop(log);
        let (_, recovery) = TransitionLog::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 1);
        assert_eq!(recovery.torn_tail_bytes, 0);
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_write_failure_before_the_frame_leaves_nothing() {
        let path = temp_path("before");
        let (mut log, _) = TransitionLog::open(&path).unwrap();
        log.append(&transition(1)).unwrap();
        log.inject(FaultPoint::BeforeWrite);
        assert!(log.append(&transition(2)).is_err());
        drop(log);
        let (_, recovery) = TransitionLog::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 1);
        assert_eq!(recovery.torn_tail_bytes, 0, "nothing was written");
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn slots_must_increase_and_request_slots_must_be_known() {
        let path = temp_path("order");
        let (mut log, _) = TransitionLog::open(&path).unwrap();
        log.append(&transition(2)).unwrap();
        log.append(&transition(2)).unwrap();
        drop(log);
        assert!(matches!(
            TransitionLog::open(&path),
            Err(StorageError::Corrupt { .. })
        ));

        let path = temp_path("request-ahead");
        let (mut log, _) = TransitionLog::open(&path).unwrap();
        log.append(&transition(1)).unwrap();
        log.append(&request(5)).unwrap();
        drop(log);
        assert!(
            matches!(
                TransitionLog::open(&path),
                Err(StorageError::Corrupt { .. })
            ),
            "a request cannot settle at a slot the log never recorded"
        );
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_partial_header_at_eof_is_torn() {
        let path = temp_path("partial-header");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, [0x45, 0x46]).unwrap();
        let (_, recovery) = TransitionLog::open(&path).unwrap();
        assert!(recovery.records.is_empty());
        assert_eq!(recovery.torn_tail_bytes, 2);
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }
}
