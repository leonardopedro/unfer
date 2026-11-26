//! D05: submission, commitment and application are three different events, and
//! the types here keep them apart.
//!
//! A [`SubmissionId`] is the *receipt* a caller gets the moment an engine has
//! durably accepted a transaction. It is not a position in the log: on a
//! network engine the id is allocated before consensus even runs, and the
//! transaction may never be committed. A [`Slot`] is the committed position,
//! assigned by the log, and a [`Watermark`] is the highest slot a node has
//! actually applied to its state.
//!
//! [`ReceiptState`] is the whole answer to "what happened to my submission?".
//! In particular [`UnknownReason`] exists so a bounded wait cannot be reported
//! as a refusal or as success: a timeout means the outcome is not known here,
//! and [`Code::UNKNOWN_OUTCOME`] says so to the caller.
//!
//! ```
//! use unfer_protocol::submission::{Receipt, ReceiptState, Slot, SubmissionId};
//! let receipt = Receipt {
//!     id: SubmissionId(1),
//!     state: ReceiptState::Committed(Slot(1)),
//! };
//! assert_eq!(receipt.slot(), Some(Slot(1)));
//! assert!(receipt.is_settled());
//! ```

use serde::{Deserialize, Serialize};

use crate::{Code, Diagnostic, Severity};

/// The receipt identity returned by `submit`. Unique per engine lifetime (and
/// durable across a restart on the network engine), never a log position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SubmissionId(pub u64);

/// A 1-based position in the committed log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Slot(pub u64);

impl Slot {
    /// The slot a node applies next, given the last one it applied.
    pub fn next_after(watermark: Watermark) -> Slot {
        Slot(watermark.0.saturating_add(1))
    }
}

/// The highest slot a node has applied, contiguously, to its state. Everything
/// at or below it is reflected in the node; nothing above it may be assumed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct Watermark(pub u64);

/// A domain rejection of a committed entry, in a form every node derives
/// identically from the same entry: the diagnostic code and the message the
/// ledger produced. Equality is (code, message), so "the same entry was
/// refused the same way" is testable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refusal {
    pub code: Code,
    pub message: String,
}

impl Refusal {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Rebuild the caller-facing diagnostic. Deliberately includes the code's
    /// canonical name so a receipt and a live error read the same.
    pub fn diagnostic(&self) -> Diagnostic {
        Diagnostic::new(self.code, self.message.clone(), Severity::Error)
    }
}

impl From<Diagnostic> for Refusal {
    fn from(diagnostic: Diagnostic) -> Self {
        Self {
            code: diagnostic.code,
            message: diagnostic.message,
        }
    }
}

/// Why a submission's outcome is not known at the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownReason {
    /// The bounded wait elapsed before the engine committed the entry. The
    /// transaction may still commit later; this is not a refusal.
    TimedOut,
    /// The engine has no record of this submission id.
    Unsubmitted,
    /// The node stopped applying (a gap or corrupt history) before reaching
    /// this submission, so its outcome cannot be read from this node.
    Halted,
    /// The node applied the submission but could not make the result durable
    /// (D07). The effect happened; what is missing is the memory a retry needs,
    /// so the caller must verify rather than assume it did not happen.
    Unrecorded,
}

impl UnknownReason {
    pub fn diagnostic(self) -> Diagnostic {
        let text = match self {
            UnknownReason::TimedOut => {
                "the submission was accepted but its commitment is not known yet; \
                 the transaction may still commit"
            }
            UnknownReason::Unsubmitted => "the engine has no record of this submission id",
            UnknownReason::Halted => {
                "this node halted on an unusable committed history and cannot report an outcome"
            }
            UnknownReason::Unrecorded => {
                "the submission was applied but its result could not be recorded durably; \
                 verify the effect before retrying it under the same request id"
            }
        };
        Diagnostic::new(Code::UNKNOWN_OUTCOME, text, Severity::Error)
    }
}

/// Everything a caller can learn about one submission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum ReceiptState {
    /// Durably accepted, not yet known to be committed.
    Pending,
    /// Committed at `slot` and applied to the node's state.
    Committed(Slot),
    /// Committed at `slot`, and the domain refused it. The slot still advanced:
    /// one bad entry does not wedge the good entries behind it.
    Refused { slot: Slot, refusal: Refusal },
    /// Not known at this caller; see the reason.
    Unknown(UnknownReason),
}

/// A submission id paired with what the engine can say about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    pub id: SubmissionId,
    pub state: ReceiptState,
}

impl Receipt {
    pub fn pending(id: SubmissionId) -> Self {
        Self {
            id,
            state: ReceiptState::Pending,
        }
    }

    pub fn unknown(id: SubmissionId, reason: UnknownReason) -> Self {
        Self {
            id,
            state: ReceiptState::Unknown(reason),
        }
    }

    /// The committed slot, if the entry has one (refused entries do).
    pub fn slot(&self) -> Option<Slot> {
        match self.state {
            ReceiptState::Committed(slot) => Some(slot),
            ReceiptState::Refused { slot, .. } => Some(slot),
            ReceiptState::Pending | ReceiptState::Unknown(_) => None,
        }
    }

    /// Whether the caller may stop waiting: committed, refused, or explicitly
    /// unknown. `Pending` is the only unsettled state.
    pub fn is_settled(&self) -> bool {
        !matches!(self.state, ReceiptState::Pending)
    }

    pub fn is_committed(&self) -> bool {
        matches!(self.state, ReceiptState::Committed(_))
    }

    pub fn refusal(&self) -> Option<&Refusal> {
        match &self.state {
            ReceiptState::Refused { refusal, .. } => Some(refusal),
            _ => None,
        }
    }

    /// The typed error a caller returns when it must not report success.
    pub fn failure(&self) -> Option<Diagnostic> {
        match &self.state {
            ReceiptState::Refused { refusal, .. } => Some(refusal.diagnostic()),
            ReceiptState::Unknown(reason) => Some(reason.diagnostic()),
            ReceiptState::Pending => Some(UnknownReason::TimedOut.diagnostic()),
            ReceiptState::Committed(_) => None,
        }
    }
}

/// Why a node stopped applying. Unlike a [`Refusal`], this is not a property of
/// the entry: the committed history itself is unusable here, so nothing after
/// it may be applied or assumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "halt")]
pub enum HaltReason {
    /// The log skipped slots: the entry at `found` arrived while `expected` was
    /// still missing. Applying it would silently drop the missing entries.
    Gap { expected: Slot, found: Slot },
    /// The log went backwards or repeated a slot after the watermark had moved
    /// past it; either way the local view is inconsistent.
    OutOfOrder { applied: Watermark, found: Slot },
    /// The entry was neither applied nor refused: this node could not evaluate
    /// or durably record it (storage failure, exhausted budget, interrupted
    /// checkpoint — [`Code::is_infrastructure`]). The state may not match the
    /// committed prefix, so nothing after this slot is applied either; the
    /// refusal text is kept only to say *why* the node stopped.
    Failed { slot: Slot, refusal: Refusal },
}

impl HaltReason {
    pub fn diagnostic(&self) -> Diagnostic {
        let text = match self {
            HaltReason::Gap { expected, found } => format!(
                "committed history has a gap: expected slot {}, found slot {}",
                expected.0, found.0
            ),
            HaltReason::OutOfOrder { applied, found } => format!(
                "committed history went backwards: watermark is {}, found slot {}",
                applied.0, found.0
            ),
            HaltReason::Failed { slot, refusal } => format!(
                "could not apply slot {} ({}: {}); the node halted so its state \
                 cannot drift from the committed prefix",
                slot.0, refusal.code, refusal.message
            ),
        };
        Diagnostic::new(Code::UNKNOWN_OUTCOME, text, Severity::Fatal)
    }
}

/// What one `sync` pass did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyncReport {
    /// Newly applied (or newly refused-but-advanced) slots.
    pub entries: u64,
    pub watermark: Watermark,
    /// Domain refusals produced by this pass, in slot order.
    pub refusals: Vec<(Slot, Refusal)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_is_the_same_refusal_from_the_same_diagnostic() {
        let a = Refusal::from(Diagnostic::new(
            Code::CERT_MINT_NOT_AUTHORIZED,
            "mint authority does not permit this mint",
            Severity::Error,
        ));
        let b = Refusal::from(Diagnostic::new(
            Code::CERT_MINT_NOT_AUTHORIZED,
            "mint authority does not permit this mint",
            Severity::Error,
        ));
        assert_eq!(a, b);
        assert_eq!(a.diagnostic().code, Code::CERT_MINT_NOT_AUTHORIZED);
    }

    #[test]
    fn refused_and_unknown_receipts_are_not_success() {
        let committed = Receipt {
            id: SubmissionId(1),
            state: ReceiptState::Committed(Slot(3)),
        };
        assert!(committed.is_committed());
        assert!(committed.failure().is_none());
        assert_eq!(committed.slot(), Some(Slot(3)));

        let refused = Receipt {
            id: SubmissionId(2),
            state: ReceiptState::Refused {
                slot: Slot(4),
                refusal: Refusal::new(Code::CERT_MINT_NOT_AUTHORIZED, "no"),
            },
        };
        assert!(!refused.is_committed());
        assert_eq!(refused.slot(), Some(Slot(4)));
        assert_eq!(
            refused.failure().unwrap().code,
            Code::CERT_MINT_NOT_AUTHORIZED
        );

        // An unrecorded outcome is not a refusal either: the effect happened.
        let unrecorded = Receipt::unknown(SubmissionId(9), UnknownReason::Unrecorded);
        assert_eq!(unrecorded.failure().unwrap().code, Code::UNKNOWN_OUTCOME);

        // A timeout is not a refusal: the transaction may still commit.
        let unknown = Receipt::unknown(SubmissionId(3), UnknownReason::TimedOut);
        assert!(!unknown.is_committed());
        assert_eq!(unknown.slot(), None);
        assert_eq!(unknown.failure().unwrap().code, Code::UNKNOWN_OUTCOME);
        assert!(unknown.is_settled());
        assert!(!Receipt::pending(SubmissionId(4)).is_settled());
    }

    #[test]
    fn a_failed_halt_says_what_stopped_the_node() {
        let halt = HaltReason::Failed {
            slot: Slot(7),
            refusal: Refusal::new(Code::STORAGE_FAILED, "disk full"),
        };
        let diagnostic = halt.diagnostic();
        assert_eq!(diagnostic.code, Code::UNKNOWN_OUTCOME);
        assert_eq!(diagnostic.severity, Severity::Fatal);
        assert!(
            diagnostic.message.contains("slot 7") && diagnostic.message.contains("UK-5011"),
            "the halt must name the slot and the cause: {}",
            diagnostic.message
        );
        // A failed halt is not a refusal of the entry, and it is not a gap.
        assert_ne!(
            halt,
            HaltReason::Gap {
                expected: Slot(7),
                found: Slot(7)
            }
        );
    }

    #[test]
    fn watermark_advances_one_slot_at_a_time() {
        assert_eq!(Slot::next_after(Watermark(0)), Slot(1));
        assert_eq!(Slot::next_after(Watermark(41)), Slot(42));
        assert_eq!(Slot::next_after(Watermark(u64::MAX)), Slot(u64::MAX));
    }
}
