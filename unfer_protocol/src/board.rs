//! G1: the shared context board — a typed, append-only, sanitized view of what
//! a group of agents is doing.
//!
//! ## What this is, and what it deliberately is not
//!
//! It is **not a new database**. It is a typed projection over the append-only
//! event stream the agent binary already keeps, with the same bounded,
//! drop-oldest discipline as the per-model event queue. An agent writing
//! `FACT` and a human reading the audit trail are looking at the same history
//! through two lenses, which is the property that makes the board worth having:
//! the shared context and the audit record cannot disagree.
//!
//! ## The entry kinds
//!
//! | kind | meaning | effect kind |
//! |---|---|---|
//! | `Observed` | something noticed, no conclusion drawn | `Observe` |
//! | `Fact` | something established, with its evidence | `Observe` |
//! | `Fail` | an approach that did not work, and why | `Observe` |
//! | `Claim` | "I am working on this scope" | `Mutate` |
//! | `PatchSummary` | "here is what I changed, and here is the gate output" | `Mutate` |
//!
//! The split is the trust model, not a taxonomy. `Claim` and `PatchSummary`
//! assert something about the world that other agents will act on — a peer
//! stops duplicating work because of a `Claim`, and a human approves a merge on
//! the strength of a `PatchSummary` — so they are [`EffectKind::Mutate`] and
//! queue for approval like any other mutating effect (S21). `Observed`,
//! `Fact` and `Fail` only add text, so they are [`EffectKind::Observe`] and
//! apply immediately. `Fail` is `Observe` rather than `Mutate` despite being the
//! most valuable thing on the board, precisely because it *removes* work rather
//! than committing it.
//!
//! ## Sizing
//!
//! Entries are capped in both directions, because an append-only log that only
//! grows is a log nobody reads:
//!
//! - `MAX_TEXT` / `MAX_DETAIL` bound a single entry, so one agent cannot fill the
//!   board with a pasted stack trace;
//! - `CAPACITY` bounds the board, dropping oldest-first, and `dropped` records
//!   how many — so a reader can tell a short history from a truncated one.
//!
//! `detail` is optional and separate from `text` precisely so the common case
//! stays cheap: `text` is the one-line claim that goes on the board, `detail` is
//! the evidence a reader unfolds when they need it.
//!
//! ## Query syntax (`board_grep`)
//!
//! `,` is OR and `&` is AND, case-insensitively: `fail,refuted` matches either
//! term; `claim&faris` matches both. AND binds tighter, so `a&b,c` is
//! `(a&b) | c`. Deliberately not a regex language — every regex engine is a
//! denial-of-service surface on an unauthenticated input, and the two operators
//! above are what the actual use cases need (find anything that is a failure;
//! find claims mentioning a file).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::EffectKind;

/// Largest accepted `text`, in characters. Enough for a sentence, short enough
/// that a dumped log cannot dominate the board.
pub const MAX_TEXT: usize = 280;

/// Largest accepted `detail`, in characters.
pub const MAX_DETAIL: usize = 4096;

/// Entries retained. Oldest are dropped first.
pub const CAPACITY: usize = 512;

/// What kind of statement an entry is.
///
/// Serialized in the **upper case** the Agensh-style literature uses
/// (`OBSERVED` / `FACT` / `FAIL` / `CLAIM` / `PATCH_SUMMARY`) because these
/// strings appear verbatim in work orders and board greps, and matching the
/// spelling people write saves a translation layer in every consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BoardKind {
    /// Something noticed. No conclusion, no commitment.
    Observed,
    /// Something established, ideally with evidence in `detail`.
    Fact,
    /// An approach that did not work, and why. The highest-value entry kind:
    /// its whole purpose is to stop a peer re-deriving a dead end.
    Fail,
    /// "I am working on this scope." See [`BoardEntry::effect_kind`].
    Claim,
    /// "Here is what changed and here is the machine-generated gate output."
    PatchSummary,
}

impl BoardKind {
    pub const ALL: &'static [BoardKind] = &[
        BoardKind::Observed,
        BoardKind::Fact,
        BoardKind::Fail,
        BoardKind::Claim,
        BoardKind::PatchSummary,
    ];

    /// The wire spelling, matching the serde representation.
    pub fn as_str(self) -> &'static str {
        match self {
            BoardKind::Observed => "OBSERVED",
            BoardKind::Fact => "FACT",
            BoardKind::Fail => "FAIL",
            BoardKind::Claim => "CLAIM",
            BoardKind::PatchSummary => "PATCH_SUMMARY",
        }
    }

    /// Parse a wire spelling. Case-insensitive, because a caller writing
    /// `fact` should not be punished for it — the grep side is case-insensitive
    /// too, and a registry that is strict in one place and lenient in another
    /// teaches callers which one to trust.
    pub fn parse(s: &str) -> Option<BoardKind> {
        match s.trim().to_ascii_uppercase().as_str() {
            "OBSERVED" => Some(BoardKind::Observed),
            "FACT" => Some(BoardKind::Fact),
            "FAIL" => Some(BoardKind::Fail),
            "CLAIM" => Some(BoardKind::Claim),
            "PATCH_SUMMARY" | "PATCHSUMMARY" => Some(BoardKind::PatchSummary),
            _ => None,
        }
    }

    /// The S21 trust annotation for writing this kind.
    ///
    /// `Claim` and `PatchSummary` are `Mutate`: they are assertions other agents
    /// and humans act on. The rest only append text and are `Observe`.
    pub fn effect_kind(self) -> EffectKind {
        match self {
            BoardKind::Claim | BoardKind::PatchSummary => EffectKind::Mutate,
            BoardKind::Observed | BoardKind::Fact | BoardKind::Fail => EffectKind::Observe,
        }
    }
}

/// One board entry.
///
/// `cursor` reuses the C1 event cursor (a process-global monotonic counter), so
/// a board read and an `events_poll` read address the same ordered history and a
/// consumer can hold one checkpoint covering both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardEntry {
    /// Monotonic position in the shared history. Assigned on write.
    pub cursor: u64,
    pub kind: BoardKind,
    /// Who wrote it. A worker id, not an identity claim — see the note in
    /// [`Board::write`] about why this is not authenticated.
    pub worker: String,
    /// The one-line statement. Capped at [`MAX_TEXT`].
    pub text: String,
    /// Optional evidence, unfolded on demand. Capped at [`MAX_DETAIL`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl BoardEntry {
    /// Case-insensitive substring match over the entry's **kind**, its text, and
    /// its detail.
    ///
    /// The kind is searchable because "show me every failure" and "show me every
    /// claim" are the two queries the board exists to answer, and an entry whose
    /// kind is not findable makes both of them impossible to write. Note the
    /// consequence: a prose entry that happens to contain the word "fact" is
    /// matched by `board_grep {expr: "FACT"}`. That is the same
    /// substring-match semantics as every other term, and being consistent about
    /// it is better than a query language where the kind field behaves
    /// differently from the text.
    pub fn matches(&self, needle: &str) -> bool {
        let n = needle.to_lowercase();
        if self.kind.as_str().to_lowercase().contains(&n) {
            return true;
        }
        if self.text.to_lowercase().contains(&n) {
            return true;
        }
        self.detail
            .as_ref()
            .is_some_and(|d| d.to_lowercase().contains(&n))
    }
}

/// A parsed `board_grep` expression: a disjunction of conjunctions.
///
/// `,` (OR) binds looser than `&` (AND), so `a&b,c` means `(a AND b) OR c`.
/// An empty term matches everything, which makes `board_grep {expr: ""}` a
/// synonym for `board_read` rather than a query that returns nothing — the
/// forgiving reading, and the one that makes "show me everything matching my
/// filter, where my filter might be empty" do the obvious thing.
#[derive(Debug, Clone, PartialEq)]
pub struct GrepExpr {
    terms: Vec<Vec<String>>,
}

impl GrepExpr {
    pub fn parse(expr: &str) -> GrepExpr {
        let terms = expr
            .split(',')
            .map(|conj| {
                conj.split('&')
                    .map(|t| t.trim().to_lowercase())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|conj| !conj.is_empty())
            .collect();
        GrepExpr { terms }
    }

    /// True when `expr` selects nothing at all — an expression made only of
    /// separators. Distinct from an expression that matches everything.
    pub fn is_empty_selection(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn matches(&self, entry: &BoardEntry) -> bool {
        if self.terms.is_empty() {
            return true;
        }
        self.terms
            .iter()
            .any(|conj| conj.iter().all(|t| entry.matches(t)))
    }
}

/// The board itself: a bounded, append-only, in-order log.
#[derive(Debug, Clone, Default)]
pub struct Board {
    entries: VecDeque<BoardEntry>,
    next_cursor: u64,
    dropped: u64,
}

impl Board {
    pub fn new() -> Board {
        Board {
            entries: VecDeque::new(),
            // Cursor 1 is the first entry, matching the C1 event log.
            next_cursor: 1,
            dropped: 0,
        }
    }

    /// The cursor the next entry would get, without consuming it.
    ///
    /// Public because cursors are a **shared** ordering, not a property of the
    /// entry log: a recorded gate run takes one too (G4), and freshness is
    /// compared across both. If a non-entry event predicted its own cursor as
    /// `latest + 1` instead of taking it from here, a run and the next entry
    /// would share a cursor and the staleness comparison would be off by one in
    /// whichever direction happened to matter.
    pub fn peek_cursor(&self) -> u64 {
        self.next_cursor
    }

    /// Take the next cursor without writing an entry.
    ///
    /// Used by anything that needs to position itself in the shared ordering
    /// without being an entry — currently the G4 gate-run registry.
    pub fn reserve_cursor(&mut self) -> u64 {
        let c = self.next_cursor;
        self.next_cursor += 1;
        c
    }

    /// Append an entry, assigning it a cursor.
    ///
    /// `text` and `detail` pass through [`redact_secrets`] **before** the cap is
    /// applied, so a secret is removed rather than truncated into
    /// unrecognisability — and so redaction cannot be used to smuggle a secret
    /// past the cap.
    ///
    /// `worker` is recorded but **not authenticated**: nothing here proves the
    /// writer is who they say they are. That is deliberate and is why the board
    /// carries no authority of its own — a `Claim` is a statement to be
    /// negotiated (G3), and a `PatchSummary` is evidence for a human approval
    /// that the trust model already gates separately (S21/S28). Making the
    /// worker id meaningful is the grant system's job, not the board's, and
    /// pretending otherwise would be a security claim this type cannot support.
    ///
    /// Returns the stored entry, with redaction and truncation already applied,
    /// so a caller can see what was actually kept.
    pub fn write(
        &mut self,
        kind: BoardKind,
        worker: &str,
        text: &str,
        detail: Option<&str>,
    ) -> BoardEntry {
        let cursor = self.next_cursor;
        self.next_cursor += 1;
        let entry = BoardEntry {
            cursor,
            kind,
            worker: worker.to_string(),
            text: truncate(&redact_secrets(text), MAX_TEXT),
            detail: detail.map(|d| truncate(&redact_secrets(d), MAX_DETAIL)),
        };
        self.entries.push_back(entry.clone());
        while self.entries.len() > CAPACITY {
            self.entries.pop_front();
            self.dropped += 1;
        }
        entry
    }

    /// The most recent `limit` entries, oldest first.
    pub fn tail(&self, limit: usize) -> Vec<&BoardEntry> {
        let skip = self.entries.len().saturating_sub(limit);
        self.entries.iter().skip(skip).collect()
    }

    /// Every retained entry, oldest first.
    pub fn all(&self) -> Vec<&BoardEntry> {
        self.entries.iter().collect()
    }

    /// Matching entries, oldest first.
    pub fn grep(&self, expr: &GrepExpr) -> Vec<&BoardEntry> {
        self.entries.iter().filter(|e| expr.matches(e)).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many entries have aged out. A non-zero value means the board is a
    /// window, not the whole history — a reader that treats it as complete is
    /// wrong, so the number travels with every read.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Cursor of the newest entry (0 when empty).
    pub fn latest_cursor(&self) -> u64 {
        self.entries.back().map(|e| e.cursor).unwrap_or(0)
    }

    /// Cursor of the oldest retained entry (`latest + 1` when empty).
    pub fn oldest_available(&self) -> u64 {
        self.entries.front().map(|e| e.cursor).unwrap_or(self.latest_cursor() + 1)
    }

    /// Whether a consumer resuming from `since_cursor` has missed entries that
    /// have since aged out. Same semantics as `uk_events_poll`'s `gap`, so one
    /// reader can handle both without special-casing.
    pub fn has_gap(&self, since_cursor: u64) -> bool {
        match self.entries.front() {
            Some(front) => since_cursor.saturating_add(1) < front.cursor,
            None => false,
        }
    }

    /// The cursor of the newest entry `worker` wrote that is **not** a
    /// `PATCH_SUMMARY`, i.e. the last thing they actually changed.
    ///
    /// This is what makes merge evidence checkable without trusting the worker.
    /// A `PATCH_SUMMARY` asserts "these files, this idea, and here is the gate
    /// output"; the gate output is only meaningful if it was produced *after* the
    /// last change it vouches for. The board already knows that ordering, so the
    /// check reads it rather than taking the worker's word.
    ///
    /// Returns `None` when the worker has written nothing, or when every entry
    /// they wrote has aged out of the bounded board. Callers must treat `None` as
    /// *cannot establish freshness* and fail closed — see
    /// [`crate::evidence`].
    pub fn last_change_cursor(&self, worker: &str) -> Option<u64> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.worker == worker && e.kind != BoardKind::PatchSummary)
            .map(|e| e.cursor)
    }
}

/// Truncate on a char boundary, marking that it happened.
///
/// Cutting mid-codepoint would produce invalid UTF-8 and a serde failure on the
/// way out — an entry limit that breaks the encoding is worse than no limit. The
/// ellipsis is inside the cap so the result is always `<= max` chars.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

// ── secret redaction ──────────────────────────────────────────────────────
//
// ## Why this is here and not a call to `sanitize_sensitive`
//
// The plan for G1 says "every write passes `sanitize_sensitive`". That function
// exists (`unfer_ffi::handles::sanitize_sensitive`) and is the right thing for
// the S23 audit path — but it redacts **by key name**: it walks a JSON object
// and rewrites the value of any field whose name contains `token`, `secret`,
// `api_key` and so on. A board entry is free text. There is no key to match, so
// calling it would be theatre: it would return the input unchanged and the
// requirement would appear satisfied.
//
// The gap is real rather than theoretical. The board is the one surface where
// many workers paste command output, and the project's own S23 note already
// records the failure mode this creates elsewhere — `audit_trail_never_persists_
// fixture_secret` exists because a secret reached a log through a path nobody was
// watching. A worker that pastes `Authorization: Bearer sk-live-…` into a `FAIL`
// entry has published it to every other worker and to any reader of the board.
//
// So this is a **value-level** scrubber, complementing the key-level one rather
// than replacing it. It is deliberately conservative: it fires on a sensitive
// key followed by a value, on `Bearer`/known-prefix token shapes, and on nothing
// else. A bare 64-character hex run is left alone, because board entries
// legitimately contain commit hashes, digests and fixture names, and a board that
// redacts every hash is a board nobody reads. Under-redacting a novel token
// format is recoverable; a board full of `***REDACTED***` is not.

/// Replacement written in place of a detected secret.
pub const REDACTED: &str = "***REDACTED***";

/// Key fragments that mark the *following* value as sensitive. Same vocabulary
/// as `unfer_ffi::handles::SENSITIVE_FRAGMENTS`, kept in step deliberately: two
/// sanitizer vocabularies that disagree is how a secret survives the one that is
/// less careful.
const SENSITIVE_KEY_FRAGMENTS: &[&str] = &[
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "authorization",
    "credential",
    "private_key",
];

/// Literal token shapes, matched case-sensitively because these prefixes are
/// themselves case-sensitive by convention (`sk-` is lowercase, `AKIA` is
/// uppercase). Listing them beats a generic "long random-looking run" rule,
/// which would eat the commit hashes and digests this board legitimately carries.
const TOKEN_PREFIXES: &[&str] = &[
    "sk-",         // OpenAI-style
    "sk_live_",    // Stripe-style
    "ghp_",        // GitHub PAT
    "gho_",
    "github_pat_", // fine-grained GitHub PAT
    "xoxb-",       // Slack bot token
    "xoxp-",       // Slack user token
    "AKIA",        // AWS access key id
    "AIza",        // Google API key
    "eyJ",         // JWT header
];

/// Rewrite secrets found in free text.
///
/// Returns the text unchanged when nothing matches, so the common case allocates
/// nothing beyond the scan.
pub fn redact_secrets(text: &str) -> String {
    let mut out = redact_bearer(text);
    for prefix in TOKEN_PREFIXES {
        out = redact_token_prefix(&out, prefix);
    }
    redact_keyed_values(&out)
}

/// `Authorization: Bearer <token>` / `bearer <token>`, whatever follows.
fn redact_bearer(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let needle = "bearer ";
    let mut from = 0usize;
    while let Some(rel) = lower[from..].find(needle) {
        let start = from + rel;
        let vstart = start + needle.len();
        let vend = text[vstart..]
            .find(|c: char| c.is_whitespace() || c == '"' || c == '\'')
            .map(|i| vstart + i)
            .unwrap_or(text.len());
        if vend > vstart {
            out.push_str(&text[last..vstart]);
            out.push_str(REDACTED);
            last = vend;
        }
        from = vstart.max(start + needle.len());
        if from >= text.len() {
            break;
        }
    }
    if last == 0 {
        return text.to_string();
    }
    out.push_str(&text[last..]);
    out
}

/// Any run starting with `prefix`, up to the next delimiter.
fn redact_token_prefix(text: &str, prefix: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut from = 0usize;
    while let Some(rel) = text[from..].find(prefix) {
        let start = from + rel;
        // Must be at a token boundary, so `mask` in `task` does not match
        // `sk-` inside a longer word.
        let boundary = start == 0
            || !text[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric());
        if boundary {
            let vend = text[start + prefix.len()..]
                .find(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == ',')
                .map(|i| start + prefix.len() + i)
                .unwrap_or(text.len());
            out.push_str(&text[last..start]);
            out.push_str(REDACTED);
            last = vend;
        }
        from = start + prefix.len();
        if from >= text.len() {
            break;
        }
    }
    if last == 0 {
        return text.to_string();
    }
    out.push_str(&text[last..]);
    out
}

/// `<sensitive key><separator><value>` → redact the value.
///
/// The value runs to the next whitespace, comma or quote, or to end of line.
fn redact_keyed_values(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;

    for (idx, _) in text.match_indices(|c: char| c == '=' || c == ':') {
        // Walk back over the key.
        //
        // Quotes are stripped *first*, because a quoted key ends at the separator
        // with its own closing quote: `{"token": …}` has `kb == {"token"`, and
        // searching that for a delimiter finds the quote at the end and yields an
        // empty key. Stripping it gives `{"token`, whose last delimiter is the
        // `{`, so the key comes out as `token`.
        let kb = text[..idx].trim_end().trim_end_matches('"');
        let kstart = kb
            .rfind(|c: char| c.is_whitespace() || c == ',' || c == '{' || c == '"')
            .map(|i| i + 1)
            .unwrap_or(0);
        let key = kb[kstart..].trim_matches('"');
        if key.is_empty() {
            continue;
        }
        let klower = key.to_ascii_lowercase();
        if !SENSITIVE_KEY_FRAGMENTS
            .iter()
            .any(|f| klower.contains(f))
        {
            continue;
        }
        // Skip `=`, optional spaces, an opening quote, then the value.
        let mut vstart = idx + 1;
        let bytes = text.as_bytes();
        while vstart < text.len() && (bytes[vstart] as char).is_whitespace() {
            vstart += 1;
        }
        let quoted = matches!(bytes.get(vstart), Some(b'"') | Some(b'\''));
        if quoted {
            vstart += 1;
        }
        let vend = text[vstart..]
            .find(|c: char| {
                c.is_whitespace() || c == ',' || c == '"' || c == '\'' || c == '}'
            })
            .map(|i| vstart + i)
            .unwrap_or(text.len());
        if vend <= vstart {
            continue;
        }
        out.push_str(&text[last..vstart]);
        out.push_str(REDACTED);
        last = vend;
    }
    if last == 0 {
        return text.to_string();
    }
    out.push_str(&text[last..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board_with(n: usize) -> Board {
        let mut b = Board::new();
        for i in 0..n {
            b.write(BoardKind::Observed, "w", &format!("entry {i}"), None);
        }
        b
    }

    // ---- the trust split ----------------------------------------------------

    #[test]
    fn claim_and_patch_summary_are_mutating_and_the_rest_are_not() {
        assert_eq!(BoardKind::Claim.effect_kind(), EffectKind::Mutate);
        assert_eq!(BoardKind::PatchSummary.effect_kind(), EffectKind::Mutate);
        for k in [BoardKind::Observed, BoardKind::Fact, BoardKind::Fail] {
            assert_eq!(k.effect_kind(), EffectKind::Observe, "{k:?}");
        }
    }

    #[test]
    fn every_kind_has_exactly_one_effect_kind_and_the_map_is_total() {
        // If a sixth kind is added, this forces a decision about its trust
        // annotation instead of inheriting `Mutate` by default.
        for k in BoardKind::ALL {
            let _ = k.effect_kind();
            assert_eq!(BoardKind::parse(k.as_str()), Some(*k));
        }
    }

    #[test]
    fn the_mutating_kinds_are_exactly_the_ones_that_commit() {
        // The rule stated as a property, so it cannot rot as kinds are added.
        let mutating: Vec<_> = BoardKind::ALL
            .iter()
            .filter(|k| k.effect_kind() == EffectKind::Mutate)
            .collect();
        assert_eq!(mutating.len(), 2);
        assert!(mutating.contains(&&BoardKind::Claim));
        assert!(mutating.contains(&&BoardKind::PatchSummary));
    }

    // ---- cursors ------------------------------------------------------------

    #[test]
    fn cursors_start_at_one_and_increase_by_one() {
        let b = board_with(5);
        let cs: Vec<u64> = b.all().iter().map(|e| e.cursor).collect();
        assert_eq!(cs, vec![1, 2, 3, 4, 5]);
        assert_eq!(b.latest_cursor(), 5);
        assert_eq!(b.oldest_available(), 1);
    }

    #[test]
    fn a_reserved_cursor_is_consumed_and_never_reissued() {
        // A gate run takes a cursor too (G4). If a reserved cursor could be
        // handed out again, a run and an entry would share one and "newer than"
        // would be ambiguous exactly where it is relied on.
        let mut b = Board::new();
        assert_eq!(b.peek_cursor(), 1);
        assert_eq!(b.reserve_cursor(), 1);
        assert_eq!(b.peek_cursor(), 2);
        // Peeking does not consume.
        assert_eq!(b.peek_cursor(), 2);
        assert_eq!(b.reserve_cursor(), 2);
        // ...and the next entry continues the same sequence.
        assert_eq!(b.write(BoardKind::Observed, "w", "after", None).cursor, 3);
    }

    #[test]
    fn a_reserved_cursor_leaves_no_entry_but_still_ages_the_history() {
        let mut b = Board::new();
        b.write(BoardKind::Observed, "w", "one", None);
        b.reserve_cursor();
        b.write(BoardKind::Observed, "w", "two", None);
        let cs: Vec<u64> = b.all().iter().map(|e| e.cursor).collect();
        assert_eq!(cs, vec![1, 3], "the gap is the reserved cursor");
        assert_eq!(b.latest_cursor(), 3);
    }

    #[test]
    fn an_empty_board_reports_no_gap_however_old_the_cursor() {
        let b = Board::new();
        assert!(b.is_empty());
        assert_eq!(b.latest_cursor(), 0);
        assert!(!b.has_gap(9999));
    }

    #[test]
    fn a_caught_up_reader_is_never_told_it_has_a_gap() {
        let b = board_with(3);
        assert!(!b.has_gap(3));
        assert!(!b.has_gap(1));
    }

    // ---- capacity -----------------------------------------------------------

    #[test]
    fn the_board_is_bounded_and_says_how_much_it_dropped() {
        let b = board_with(CAPACITY + 25);
        assert_eq!(b.len(), CAPACITY);
        assert_eq!(b.dropped(), 25);
        assert_eq!(b.oldest_available(), 26);
        assert!(
            b.has_gap(0),
            "a reader starting from 0 must be told the beginning is gone"
        );
        assert!(!b.has_gap(25), "a reader at the oldest retained cursor is whole");
    }

    #[test]
    fn dropping_is_oldest_first_and_preserves_order() {
        let b = board_with(CAPACITY + 3);
        let cs: Vec<u64> = b.all().iter().map(|e| e.cursor).collect();
        assert_eq!(cs.len(), CAPACITY);
        assert!(cs.windows(2).all(|w| w[0] < w[1]), "order must hold");
        assert_eq!(*cs.last().unwrap(), CAPACITY as u64 + 3);
    }

    // ---- entry caps ---------------------------------------------------------

    #[test]
    fn oversized_text_is_truncated_and_the_result_stays_within_the_cap() {
        let mut b = Board::new();
        let e = b.write(BoardKind::Fact, "w", &"x".repeat(MAX_TEXT * 3), None);
        assert_eq!(e.text.chars().count(), MAX_TEXT);
        assert!(e.text.ends_with('…'));
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        // Cutting mid-codepoint yields invalid UTF-8 and a serde failure on the
        // way out; an entry cap must never break the encoding.
        let mut b = Board::new();
        let s = "π".repeat(MAX_TEXT + 50);
        let e = b.write(BoardKind::Fact, "w", &s, None);
        assert_eq!(e.text.chars().count(), MAX_TEXT);
        assert!(std::str::from_utf8(e.text.as_bytes()).is_ok());
        let round: BoardEntry =
            serde_json::from_str(&serde_json::to_string(&e).unwrap()).expect("round-trips");
        assert_eq!(round, e);
    }

    #[test]
    fn text_at_exactly_the_cap_is_untouched() {
        let mut b = Board::new();
        let s = "y".repeat(MAX_TEXT);
        let e = b.write(BoardKind::Fact, "w", &s, None);
        assert_eq!(e.text, s);
        assert!(!e.text.ends_with('…'));
    }

    #[test]
    fn detail_is_capped_independently_of_text() {
        let mut b = Board::new();
        let e = b.write(
            BoardKind::PatchSummary,
            "w",
            "short",
            Some(&"d".repeat(MAX_DETAIL * 2)),
        );
        assert_eq!(e.text, "short");
        assert_eq!(e.detail.as_ref().unwrap().chars().count(), MAX_DETAIL);
    }

    #[test]
    fn an_absent_detail_stays_absent_rather_than_becoming_empty() {
        let mut b = Board::new();
        let e = b.write(BoardKind::Observed, "w", "t", None);
        assert_eq!(e.detail, None);
    }

    // ---- grep ---------------------------------------------------------------

    fn grep_board() -> Board {
        let mut b = Board::new();
        b.write(BoardKind::Fail, "w1", "Faris-Lavine route refuted", None);
        b.write(BoardKind::Fact, "w2", "nanoda re-verifies the export", None);
        b.write(BoardKind::Claim, "w1", "claiming the QYM form gap", None);
        b.write(
            BoardKind::Fail,
            "w2",
            "unitarity trick",
            Some("collapsed the sum to one term"),
        );
        b
    }

    #[test]
    fn grep_is_case_insensitive() {
        let b = grep_board();
        assert_eq!(GrepExpr::parse("REFUTED").matches_all(&b), 1);
        assert_eq!(GrepExpr::parse("refuted").matches_all(&b), 1);
    }

    #[test]
    fn a_kind_is_searchable_so_failures_and_claims_are_listable() {
        // The two queries the board exists to answer. An entry's kind is metadata,
        // not prose, so if it is not searchable neither of these can be written.
        let b = grep_board();
        assert_eq!(GrepExpr::parse("FAIL").matches_all(&b), 2);
        assert_eq!(GrepExpr::parse("CLAIM").matches_all(&b), 1);
        assert_eq!(GrepExpr::parse("FACT").matches_all(&b), 1);
        assert_eq!(GrepExpr::parse("OBSERVED").matches_all(&b), 0);
        assert_eq!(GrepExpr::parse("PATCH_SUMMARY").matches_all(&b), 0);
    }

    #[test]
    fn a_kind_term_combines_with_a_text_term() {
        let b = grep_board();
        // Every FAIL ...
        assert_eq!(GrepExpr::parse("FAIL").matches_all(&b), 2);
        // ... and only the one that mentions Faris-Lavine.
        assert_eq!(GrepExpr::parse("FAIL&lavine").matches_all(&b), 1);
        assert_eq!(GrepExpr::parse("FAIL&nanoda").matches_all(&b), 0);
    }

    #[test]
    fn comma_is_or() {
        let b = grep_board();
        let e = GrepExpr::parse("nanoda,claiming");
        assert_eq!(e.matches_all(&b), 2);
    }

    #[test]
    fn ampersand_is_and() {
        let b = grep_board();
        assert_eq!(GrepExpr::parse("unitarity&collapsed").matches_all(&b), 1);
        assert_eq!(GrepExpr::parse("unitarity&nanoda").matches_all(&b), 0);
    }

    #[test]
    fn and_binds_tighter_than_or() {
        // `a&b,c` is `(a AND b) OR c`, not `a AND (b OR c)`.
        let b = grep_board();
        let e = GrepExpr::parse("nanoda&re-verifies,claiming");
        assert_eq!(e.matches_all(&b), 2);
    }

    #[test]
    fn grep_searches_detail_as_well_as_text() {
        let b = grep_board();
        // "collapsed" appears only in detail.
        assert_eq!(GrepExpr::parse("collapsed").matches_all(&b), 1);
    }

    #[test]
    fn an_empty_expression_matches_everything() {
        // Forgiving on purpose: an empty filter should read as "no filter", not
        // as "nothing matches".
        let b = grep_board();
        for expr in ["", "   ", ",", ",,&"] {
            assert_eq!(
                GrepExpr::parse(expr).matches_all(&b),
                4,
                "expr {expr:?} should select everything"
            );
        }
    }

    #[test]
    fn whitespace_around_operators_is_ignored() {
        let b = grep_board();
        assert_eq!(GrepExpr::parse("  nanoda ,  claiming  ").matches_all(&b), 2);
        assert_eq!(GrepExpr::parse(" unitarity & collapsed ").matches_all(&b), 1);
    }

    #[test]
    fn grep_results_come_back_oldest_first() {
        let b = grep_board();
        let hits = b.grep(&GrepExpr::parse("refuted,claiming"));
        let cs: Vec<u64> = hits.iter().map(|e| e.cursor).collect();
        assert_eq!(cs, vec![1, 3]);
    }

    // ---- read shapes --------------------------------------------------------

    #[test]
    fn tail_returns_the_newest_n_in_order() {
        let b = board_with(10);
        let t = b.tail(3);
        assert_eq!(t.len(), 3);
        assert_eq!(
            t.iter().map(|e| e.cursor).collect::<Vec<_>>(),
            vec![8, 9, 10]
        );
    }

    #[test]
    fn tail_beyond_the_ending_is_the_whole_board_not_an_error() {
        let b = board_with(4);
        assert_eq!(b.tail(100).len(), 4);
    }

    #[test]
    fn tail_zero_is_empty_not_a_panic() {
        let b = board_with(4);
        assert!(b.tail(0).is_empty());
    }

    // ---- wire format --------------------------------------------------------

    #[test]
    fn kinds_serialize_as_the_documented_upper_case() {
        // These strings are what appear in work orders and greps.
        assert_eq!(
            serde_json::to_string(&BoardKind::PatchSummary).unwrap(),
            "\"PATCH_SUMMARY\""
        );
        assert_eq!(serde_json::to_string(&BoardKind::Fact).unwrap(), "\"FACT\"");
        let back: BoardKind =
            serde_json::from_str("\"PATCH_SUMMARY\"").expect("round-trips");
        assert_eq!(back, BoardKind::PatchSummary);
    }

    #[test]
    fn kind_parsing_is_forgiving_about_case_and_the_patch_spelling() {
        assert_eq!(BoardKind::parse("fail"), Some(BoardKind::Fail));
        assert_eq!(BoardKind::parse(" FAIL "), Some(BoardKind::Fail));
        assert_eq!(BoardKind::parse("patch_summary"), Some(BoardKind::PatchSummary));
        assert_eq!(BoardKind::parse("PATCHSUMMARY"), Some(BoardKind::PatchSummary));
        assert_eq!(BoardKind::parse("nonsense"), None);
    }

    #[test]
    fn an_entry_round_trips_through_json() {
        let mut b = Board::new();
        let e = b.write(
            BoardKind::PatchSummary,
            "integrator",
            "merged the gauge fix",
            Some("invariants: 41/41 green"),
        );
        let text = serde_json::to_string(&e).unwrap();
        assert_eq!(serde_json::from_str::<BoardEntry>(&text).unwrap(), e);
        assert!(text.contains("\"cursor\":1"), "{text}");
    }

    #[test]
    fn an_entry_without_detail_omits_the_field() {
        let mut b = Board::new();
        let e = b.write(BoardKind::Observed, "w", "t", None);
        let text = serde_json::to_string(&e).unwrap();
        assert!(!text.contains("detail"), "{text}");
    }

    // ---- redaction ----------------------------------------------------------

    #[test]
    fn a_bearer_token_in_free_text_is_redacted() {
        let got = redact_secrets("failed: Authorization: Bearer sk-live-abc123");
        assert!(!got.contains("abc123"), "{got}");
        assert!(got.contains(REDACTED), "{got}");
        // The surrounding words survive, so the entry is still readable.
        assert!(got.starts_with("failed: Authorization:"), "{got}");
    }

    #[test]
    fn known_token_prefixes_are_redacted_wherever_they_appear() {
        for (prefix, sample) in [
            ("sk-", "sk-abc123"),
            ("ghp_", "ghp_abc123"),
            ("github_pat_", "github_pat_11ABC"),
            ("xoxb-", "xoxb-123-456"),
            ("AKIA", "AKIAIOSFODNN7EXAMPLE"),
            ("eyJ", "eyJhbGciOiJIUzI1NiJ9"),
        ] {
            let text = format!("curl failed with {sample} attached");
            let got = redact_secrets(&text);
            assert!(!got.contains(sample), "{prefix}: {got}");
        }
    }

    #[test]
    fn a_token_prefix_inside_a_longer_word_is_not_a_token() {
        // `task` contains `sk-`; redacting it would mangle ordinary prose.
        let got = redact_secrets("the task-skeptic disagreed");
        assert_eq!(got, "the task-skeptic disagreed");
    }

    #[test]
    fn a_sensitive_key_followed_by_a_value_redacts_the_value() {
        for text in [
            "api_key=abc123",
            "api_key: abc123",
            "MY_SECRET = abc123",
            "{\"token\": \"abc123\"}",
            "password=abc123",
        ] {
            let got = redact_secrets(text);
            assert!(!got.contains("abc123"), "{text} -> {got}");
        }
    }

    #[test]
    fn the_key_word_itself_survives_so_the_entry_stays_readable() {
        let got = redact_secrets("api_key=abc123");
        assert!(got.starts_with("api_key"), "{got}");
    }

    #[test]
    fn redaction_keeps_the_rest_of_the_sentence() {
        let got = redact_secrets("zenodo upload failed with api_key=abc123 after 3 retries");
        assert!(got.contains("zenodo upload failed"), "{got}");
        assert!(got.contains("after 3 retries"), "{got}");
        assert!(!got.contains("abc123"), "{got}");
    }

    #[test]
    fn ordinary_board_content_is_never_redacted() {
        // The false-positive guard. A board entry legitimately contains hashes,
        // cursors, paths and theorem names; redacting those makes it unreadable
        // and trains peers to ignore it.
        for text in [
            "the Faris-Lavine route is refuted (see ChapterSirkReliability.lean)",
            "cursor 41, dropped_total 0, gap false",
            "commit 7fa2b6c added Orientation sections to 48 chapters",
            "gap certificate sha256 = 5f13e80606a26aae for book.tex",
            "unfer_consensus::certs::CertificateLedger::mint",
            "VERIFI no sorry in BookProof/ChapterStarobinskyPotential.lean",
        ] {
            assert_eq!(redact_secrets(text), text, "false positive on {text:?}");
        }
    }

    #[test]
    fn a_bare_long_hex_run_is_left_alone() {
        // Deliberate: hashes and digests are first-class board content, and the
        // under-redaction risk is recoverable while a board of redactions is not.
        let digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let text = format!("fixture digest {digest}");
        assert_eq!(redact_secrets(&text), text);
    }

    #[test]
    fn redaction_is_case_insensitive_for_the_bearer_scheme() {
        for text in ["BEARER abc123", "bearer abc123", "Bearer abc123"] {
            assert!(!redact_secrets(text).contains("abc123"), "{text}");
        }
    }

    #[test]
    fn redaction_happens_before_truncation() {
        // Otherwise a secret past the cap is merely truncated, which is not
        // removal, and redaction could be used to push a secret out of view.
        let long = format!("{} api_key=supersecretvalue", "filler ".repeat(80));
        let mut b = Board::new();
        let e = b.write(BoardKind::Fail, "w", &long, None);
        assert!(
            !e.text.contains("supersecretvalue"),
            "secret survived: {:?}",
            &e.text[..e.text.len().min(120)]
        );
    }

    #[test]
    fn detail_is_redacted_too() {
        let mut b = Board::new();
        let e = b.write(
            BoardKind::PatchSummary,
            "w",
            "merged",
            Some("used token=abc123 to push".to_string().as_str()),
        );
        assert!(!e.detail.as_ref().unwrap().contains("abc123"));
    }

    #[test]
    fn a_grep_still_finds_the_entry_after_redaction() {
        // Redaction must not destroy the entry's usefulness: the *fact* is still
        // on the board, only the credential is gone.
        let mut b = Board::new();
        b.write(BoardKind::Fail, "w1", "zenodo push failed with api_key=abc123", None);
        assert_eq!(GrepExpr::parse("zenodo").matches_all(&b), 1);
        assert_eq!(GrepExpr::parse("abc123").matches_all(&b), 0);
    }

    #[test]
    fn redact_is_idempotent() {
        // A consumer that re-sanitizes on read must not double-redact.
        let once = redact_secrets("api_key=abc123");
        assert_eq!(redact_secrets(&once), once);
    }

    impl GrepExpr {
/// Test helper: how many retained entries this expression selects.
    ///
    /// `pub` so sibling modules' tests can use it; it exists only under
    /// `#[cfg(test)]` and is not part of the crate's surface.
    pub fn matches_all(&self, b: &Board) -> usize {
        b.grep(self).len()
    }
    }
}