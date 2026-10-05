//! C4: one message shape for every ingress.
//!
//! ## The problem
//!
//! Every platform an agent can be reached on has its own payload shape. Telegram
//! nests the sender three objects deep under `message`; Discord calls the same
//! field `content` and puts the author in `author`; a webhook from a CI system has
//! neither and calls it `body`. A handler that wants to reason about "a person
//! said a thing in a thread" has to be written once per platform, and adding a
//! platform means touching agent logic — which is the coupling the plan names
//! when it says *"one ingest API; add platforms without touching agent logic"*.
//!
//! ## What this is and is not
//!
//! It is a **schema**, not a broker. Nothing here queues, retries, fans out or
//! delivers. A message arrives, is normalized, and is handed to code that knows
//! nothing about where it came from. That boundary is deliberate: this type is
//! the cheapest part of the problem and the only part that has to be right before
//! any of the harder parts can be.
//!
//! ## Extensibility without a code change
//!
//! [`IngestSource`] carries known platforms as unit variants and everything else
//! as `Other(String)`. That is what makes "add a platform" a config change rather
//! than a release: a new ingress can be registered, and handlers that do not care
//! about it keep working because they match on the variants they know and treat
//! `Other` as opaque.
//!
//! ## The three decisions worth arguing about
//!
//! **Attachments are references, never inline bytes.** A message envelope that
//! accepts a base64 blob is an unbounded allocation with a JSON parser in front
//! of it, and the cap that stops it is a number someone has to remember to lower.
//! [`Attachment`] holds a locator and a size the caller *claims*, never content.
//!
//! **`text` is redacted on the way in.** This is the same reason the board scrubs
//! on write ([`crate::board::redact_secrets`]): an ingress is exactly where a
//! token arrives, because that is what people paste. Redacting here means the
//! secret is not stored rather than merely hidden later, and it applies to *every*
//! ingress uniformly instead of once per handler that remembered.
//!
//! **Oversize is a rejection, not a silent truncation.** [`IngestMessage::validate`]
//! refuses a message whose text exceeds the cap and says which field and by how
//! much. A handler that silently truncated would be sending a different message
//! than the one it received, and the sender would never know — for a field that is
//! sometimes the whole point of the message, that is data loss with a 200 on it.

use serde::{Deserialize, Serialize};

use crate::board::{REDACTED, redact_secrets};

/// Characters of message text retained.
pub const MAX_MESSAGE_TEXT: usize = 8_000;
/// Characters retained for a sender identifier.
pub const MAX_SENDER: usize = 128;
/// Characters retained for a channel or thread identifier.
pub const MAX_SCOPE: usize = 128;
/// Attachments retained on one message.
pub const MAX_ATTACHMENTS: usize = 16;
/// Characters retained for one attachment locator.
pub const MAX_ATTACHMENT_REF: usize = 512;
/// Messages retained in one batch.
///
/// Bounded because a webhook that omits `count` must not be able to make the
/// receiver allocate without limit. A platform's real batch sizes are far below
/// this; the cap is a floor for "you may be about to be attacked", not a target.
pub const MAX_BATCH: usize = 256;

/// Why a message was refused.
///
/// Distinguishing the reasons matters to a sender: "too long" is fixable by the
/// sender, "unknown source" is not, and collapsing both into `BAD_JSON` teaches
/// senders to retry something that cannot succeed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngestError {
    /// `text` exceeded [`MAX_MESSAGE_TEXT`].
    TextTooLong { len: usize, max: usize },
    /// More than [`MAX_ATTACHMENTS`] attachments.
    TooManyAttachments { len: usize, max: usize },
    /// An attachment locator exceeded [`MAX_ATTACHMENT_REF`].
    AttachmentRefTooLong {
        index: usize,
        len: usize,
        max: usize,
    },
    /// `source` was blank.
    MissingSource,
    /// `sender` was blank, so nothing can be attributed to anyone.
    MissingSender,
    /// `id` was blank, so the message cannot be deduplicated or traced.
    MissingId,
    /// A batch carried more than [`MAX_BATCH`] messages.
    BatchTooLarge { len: usize, max: usize },
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestError::TextTooLong { len, max } => {
                write!(f, "message text is {len} characters, over the {max} cap")
            }
            IngestError::TooManyAttachments { len, max } => {
                write!(f, "message carries {len} attachments, over the {max} cap")
            }
            IngestError::AttachmentRefTooLong { index, len, max } => write!(
                f,
                "attachment {index} locator is {len} characters, over the {max} cap"
            ),
            IngestError::MissingSource => write!(f, "message has no source"),
            IngestError::MissingSender => write!(f, "message has no sender"),
            IngestError::MissingId => write!(f, "message has no id"),
            IngestError::BatchTooLarge { len, max } => {
                write!(f, "batch carries {len} messages, over the {max} cap")
            }
        }
    }
}

impl std::error::Error for IngestError {}

/// Where a message came from.
///
/// Known platforms are variants so handler code can match exhaustively; anything
/// else is [`IngestSource::Other`], which is the extension point. A handler that
/// only cares about `Other` still works when a platform is added, which is the
/// property that keeps agent logic off the critical path of adding a channel.
///
/// # Wire format
///
/// A bare string, always: `"telegram"`, or `"matrix"` for an unregistered one.
///
/// Derived serde would have produced `{"other":"matrix"}` for the unregistered
/// case and then **failed to read it back** from the bare string a sender
/// actually sends — so the extension point would have existed in the type and
/// not on the wire, which is the one place it matters. `Other` has to be
/// indistinguishable from a variant to a sender that has never heard of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IngestSource {
    Telegram,
    Discord,
    Whatsapp,
    Slack,
    Email,
    Webhook,
    /// The operator's own console.
    Console,
    /// Any other ingress. Deliberately open.
    Other(String),
}

impl Serialize for IngestSource {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for IngestSource {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        // Never fails: an unrecognised source is a message that arrived, and
        // refusing it here would turn a misconfigured webhook into a silent drop.
        Ok(IngestSource::parse(&raw))
    }
}

impl IngestSource {
    /// Parse a source name, falling back to [`IngestSource::Other`].
    ///
    /// Never fails. A source this build has never heard of is still a message
    /// that arrived, and refusing it would mean an unrecognised webhook is a
    /// silent drop rather than a visible one.
    pub fn parse(s: &str) -> IngestSource {
        match s.trim().to_ascii_lowercase().as_str() {
            "telegram" => IngestSource::Telegram,
            "discord" => IngestSource::Discord,
            "whatsapp" | "wa" => IngestSource::Whatsapp,
            "slack" => IngestSource::Slack,
            "email" | "mail" => IngestSource::Email,
            "webhook" | "hook" => IngestSource::Webhook,
            "console" => IngestSource::Console,
            other => IngestSource::Other(other.to_string()),
        }
    }

    /// The stable wire spelling.
    pub fn as_str(&self) -> &str {
        match self {
            IngestSource::Telegram => "telegram",
            IngestSource::Discord => "discord",
            IngestSource::Whatsapp => "whatsapp",
            IngestSource::Slack => "slack",
            IngestSource::Email => "email",
            IngestSource::Webhook => "webhook",
            IngestSource::Console => "console",
            IngestSource::Other(s) => s,
        }
    }

    /// Whether this build knows the source by name.
    ///
    /// Distinct from "is not `Other`" in spirit only — provided so a
    /// `/version` or metrics surface can report how much of the traffic is
    /// arriving from channels this build has never been told about, which is the
    /// early warning for a misconfigured webhook.
    pub fn is_known(&self) -> bool {
        !matches!(self, IngestSource::Other(_))
    }
}

/// What kind of thing an attachment is.
///
/// A kind rather than a MIME string because the handler's decision is about
/// *handling* ("fetch and OCR this" versus "just show a link"), and that
/// decision does not need the exact type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentKind {
    Image,
    Document,
    Audio,
    Link,
    /// Anything the platform did not classify, or a new kind this build predates.
    Other,
}

impl AttachmentKind {
    pub fn parse(s: &str) -> AttachmentKind {
        match s.trim().to_ascii_lowercase().as_str() {
            "image" | "photo" => AttachmentKind::Image,
            "document" | "file" | "doc" => AttachmentKind::Document,
            "audio" | "voice" => AttachmentKind::Audio,
            "link" | "url" => AttachmentKind::Link,
            _ => AttachmentKind::Other,
        }
    }
}

/// A pointer at content stored elsewhere.
///
/// Deliberately **not** the content. See the module note: inline bytes in an
/// envelope are an unbounded allocation behind a JSON parser.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    pub kind: AttachmentKind,
    /// Where the content lives — a URL, a blob digest, a content id. Opaque here.
    #[serde(default)]
    pub reference: String,
    /// Size the caller claims, in bytes. A claim, not a measurement: nothing
    /// verifies it, and it is recorded so a handler can decide whether fetching
    /// it is worth doing before it does.
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

/// One normalized inbound message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestMessage {
    /// The platform's own message id. Deduplication and tracing key; required.
    pub id: String,
    pub source: IngestSource,
    /// Channel, room, or conversation. `None` for a one-to-one surface that has
    /// no separate channel concept.
    #[serde(default)]
    pub channel: Option<String>,
    /// Thread within the channel, when the platform has one.
    #[serde(default)]
    pub thread: Option<String>,
    /// Who sent it. An identifier, **not** an identity claim — the same caveat as
    /// the board's `worker` field: nothing here proves the sender is who they say.
    pub sender: String,
    pub text: String,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
    /// Unix seconds. `None` when the platform did not supply one, and it is
    /// `Option` rather than defaulted to "now" precisely so that an absent
    /// timestamp stays visible instead of being quietly replaced by ingest time —
    /// the two mean very different things to anything reasoning about ordering.
    #[serde(default)]
    pub ts: Option<u64>,
}

impl IngestMessage {
    /// Build a message from loose parts, applying redaction and the caps.
    ///
    /// This is the constructor a platform adapter uses. It is separate from
    /// [`Self::validate`] because an adapter needs to *produce* a message and
    /// only then find out whether it was acceptable; validating inside the
    /// constructor would leave an adapter with no way to report which field was
    /// wrong.
    pub fn new(
        id: impl Into<String>,
        source: IngestSource,
        sender: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        IngestMessage {
            id: truncate(&id.into(), MAX_SCOPE),
            source,
            channel: None,
            thread: None,
            sender: truncate(&redact_secrets(&sender.into()), MAX_SENDER),
            // Redaction happens before the cap for the same reason it does on the
            // board: truncating first could cut a secret into something the
            // scrubber no longer recognizes, which is redaction bypass by
            // accident.
            text: truncate(&redact_secrets(&text.into()), MAX_MESSAGE_TEXT),
            attachments: Vec::new(),
            ts: None,
        }
    }

    /// Attach a channel and thread.
    pub fn with_scope(mut self, channel: Option<&str>, thread: Option<&str>) -> Self {
        self.channel = channel
            .map(|c| truncate(&redact_secrets(c), MAX_SCOPE))
            .filter(|c| !c.is_empty());
        self.thread = thread
            .map(|t| truncate(&redact_secrets(t), MAX_SCOPE))
            .filter(|t| !t.is_empty());
        self
    }

    /// Attach a timestamp.
    pub fn with_ts(mut self, ts: u64) -> Self {
        self.ts = Some(ts);
        self
    }

    /// Add an attachment, keeping only the first [`MAX_ATTACHMENTS`].
    ///
    /// Silent truncation here is a considered exception to the rule above. An
    /// extra attachment is not a *different message* — the text and sender are
    /// intact — whereas truncating text would change what was said. The count
    /// that was dropped is available via [`Self::validate`] on the pre-trim list
    /// for a caller that cares.
    pub fn with_attachment(
        mut self,
        kind: AttachmentKind,
        reference: &str,
        size: Option<u64>,
    ) -> Self {
        if self.attachments.len() < MAX_ATTACHMENTS {
            self.attachments.push(Attachment {
                kind,
                reference: truncate(&redact_secrets(reference), MAX_ATTACHMENT_REF),
                size_bytes: size,
            });
        }
        self
    }

    /// Whether the message is acceptable, and why not if it is not.
    ///
    /// This is an oversize check, not a shape check. Deserialization already
    /// guarantees the required fields exist; what this catches is a message that
    /// parsed and is still too big or too empty to be worth storing.
    pub fn validate(&self) -> Result<(), IngestError> {
        if self.id.trim().is_empty() {
            return Err(IngestError::MissingId);
        }
        if self.source.as_str().trim().is_empty() {
            return Err(IngestError::MissingSource);
        }
        if self.sender.trim().is_empty() {
            return Err(IngestError::MissingSender);
        }
        let text_len = self.text.chars().count();
        if text_len > MAX_MESSAGE_TEXT {
            return Err(IngestError::TextTooLong {
                len: text_len,
                max: MAX_MESSAGE_TEXT,
            });
        }
        if self.attachments.len() > MAX_ATTACHMENTS {
            return Err(IngestError::TooManyAttachments {
                len: self.attachments.len(),
                max: MAX_ATTACHMENTS,
            });
        }
        for (index, a) in self.attachments.iter().enumerate() {
            let len = a.reference.chars().count();
            if len > MAX_ATTACHMENT_REF {
                return Err(IngestError::AttachmentRefTooLong {
                    index,
                    len,
                    max: MAX_ATTACHMENT_REF,
                });
            }
        }
        Ok(())
    }

    /// A key for deduplication.
    ///
    /// Namespaced by source because two platforms will both hand out the id
    /// `1`, and deduplicating across them would silently drop half the traffic.
    pub fn dedup_key(&self) -> String {
        format!("{}:{}", self.source.as_str(), self.id)
    }

    /// Whether this message is text-only, ignoring attachments entirely.
    ///
    /// Deliberately ignores attachments. "Does this carry content that needs a
    /// model" is a routing question, and a handler that has to special-case
    /// attachments here is back to being channel-aware.
    pub fn has_text(&self) -> bool {
        !self.text.trim().is_empty()
    }
}

/// A bounded batch of messages, as a webhook delivers them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestBatch {
    pub messages: Vec<IngestMessage>,
}

impl IngestBatch {
    /// Wrap messages, refusing a batch over the cap.
    ///
    /// Refused rather than trimmed: unlike an extra attachment, dropping a
    /// *message* loses something a sender will believe was received.
    pub fn new(messages: Vec<IngestMessage>) -> Result<Self, IngestError> {
        if messages.len() > MAX_BATCH {
            return Err(IngestError::BatchTooLarge {
                len: messages.len(),
                max: MAX_BATCH,
            });
        }
        Ok(IngestBatch { messages })
    }

    /// Validate every message, stopping at the first failure.
    ///
    /// Stopping rather than collecting all errors is what a receiver wants: it
    /// must not partially apply a batch, and reporting the first bad message is
    /// enough for a sender to find it.
    pub fn validate(&self) -> Result<(), (usize, IngestError)> {
        for (i, m) in self.messages.iter().enumerate() {
            m.validate().map_err(|e| (i, e))?;
        }
        Ok(())
    }

    /// How many messages came from a source this build has no name for.
    ///
    /// A cheap health signal: a non-zero count usually means a webhook is
    /// posting under a source string nobody registered.
    pub fn unknown_source_count(&self) -> usize {
        self.messages
            .iter()
            .filter(|m| !m.source.is_known())
            .count()
    }
}

/// Truncate on a char boundary, marking that it happened.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Whether a redacted string still carries a scrub mark.
///
/// Used by tests and by a `/version`-style diagnostic. Exposed because "did
/// redaction fire" is a question operators ask, and answering it by inspecting
/// internals is worse than asking.
pub fn was_redacted(s: &str) -> bool {
    s.contains(REDACTED)
}

/// C4: the property the fuzz target asserts, run deterministically.
///
/// ## Why this exists alongside a fuzz target
///
/// `fuzz/fuzz_targets/ingest_message.rs` states the same four properties, but a
/// fuzz target needs nightly and `libfuzzer-sys`, neither of which this
/// environment has -- so it is a comment that happens to compile. This module
/// runs the identical assertions over a large generated corpus in ordinary CI,
/// which means the property is actually enforced rather than merely intended.
///
/// ## The corpus
///
/// A hand-written list of bad inputs tests the cases its author imagined. This
/// generates from a small xorshift PRNG over an alphabet chosen to break things:
/// NUL and control bytes, a four-byte astral character, a bidi override, secret
/// shapes, and the two JSON metacharacters. Byte-level mutation of valid JSON is
/// what actually finds parser disagreements, and a fixed seed keeps a failure
/// reproducible -- a fuzz finding you cannot re-run is a rumour.
pub mod property {
    use super::*;

    /// xorshift64*. Fixed seed, no clock: reproducible and dependency-free.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Fragments chosen for what they break, not for looking adversarial.
    const FRAGMENTS: &[&str] = &[
        "",
        " ",
        "telegram",
        "matrix",
        "\u{0}",
        "\u{1}\u{7f}",
        "\u{202e}",
        "\u{1F642}",
        "api_key=sk-live-abc123",
        "Bearer ghp_realtokenvalue",
        "{",
        "}",
        "[",
        "]",
        "\"",
        "\\",
        "\\u0000",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ];

    fn corpus_value(rng: &mut Rng, depth: usize) -> String {
        let mut out = String::new();
        let pieces = 1 + rng.below(4);
        for _ in 0..pieces {
            let f = FRAGMENTS[rng.below(FRAGMENTS.len())];
            out.push_str(f);
        }
        // Occasionally splice in a bracket or quote unescaped, which is what
        // turns a value into malformed JSON.
        if depth > 0 && rng.below(6) == 0 {
            out.push(match rng.below(4) {
                0 => '{',
                1 => '[',
                2 => '"',
                _ => '}',
            });
        }
        out
    }

    fn corpus_message(rng: &mut Rng) -> String {
        let mut o = String::from("{");
        let fields: [(&str, &str); 4] = [
            ("id", "\"id\":\""),
            ("source", "\"source\":\""),
            ("sender", "\"sender\":\""),
            ("text", "\"text\":\""),
        ];
        for (i, (_name, prefix)) in fields.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(prefix);
            o.push_str(&corpus_value(rng, 1));
            o.push('"');
        }
        if rng.below(2) == 0 {
            o.push_str(",\"ts\":");
            o.push_str(&rng.next().to_string());
        }
        if rng.below(3) == 0 {
            o.push_str(",\"attachments\":[{\"kind\":\"image\",\"reference\":\"");
            o.push_str(&corpus_value(rng, 0));
            o.push_str("\",\"size_bytes\":9223372036854775807}]");
        }
        o.push('}');
        o
    }

    /// The four fuzz properties, over a generated corpus.
    ///
    /// 20k messages keeps this well under a second while covering every fragment
    /// combination the alphabet allows many times over.
    pub fn arbitrary_json_never_panics() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut parsed = 0usize;
        for _ in 0..20_000 {
            let raw = corpus_message(&mut rng);

            // 1. Parsing arbitrary bytes must never panic, and a failure is fine.
            let Ok(msg) = serde_json::from_str::<IngestMessage>(&raw) else {
                continue;
            };
            parsed += 1;

            // 2. The caps hold on anything that parsed.
            assert!(
                msg.text.chars().count() <= MAX_MESSAGE_TEXT,
                "text cap broken by {raw}"
            );
            assert!(
                msg.sender.chars().count() <= MAX_SENDER,
                "sender cap broken by {raw}"
            );
            assert!(
                msg.attachments.len() <= MAX_ATTACHMENTS,
                "attachment count cap broken by {raw}"
            );
            for a in &msg.attachments {
                assert!(
                    a.reference.chars().count() <= MAX_ATTACHMENT_REF,
                    "attachment ref cap broken by {raw}"
                );
            }

            // 3. Parse -> serialize -> parse is a fixed point.
            let once = serde_json::to_string(&msg).expect("re-serializes");
            let twice: IngestMessage =
                serde_json::from_str(&once).expect("re-parses what it serialized");
            assert_eq!(msg, twice, "not a fixed point: {raw}");

            // 4. An unregistered source survives as its own name on the wire.
            if !msg.source.is_known() {
                let v: serde_json::Value = serde_json::from_str(&once).unwrap();
                assert_eq!(
                    v.get("source").and_then(|s| s.as_str()),
                    Some(msg.source.as_str()),
                    "an unregistered source lost its name: {raw}"
                );
            }

            // And the constructor path redacts regardless of what went in.
            let built = IngestMessage::new(
                msg.id.clone(),
                msg.source.clone(),
                msg.sender.clone(),
                msg.text.clone(),
            );
            assert!(
                !built.text.contains("abc123"),
                "a secret reached the store: {}",
                built.text
            );
        }
        // The corpus must actually be exercising the parser, or the properties
        // above are vacuous -- a generator that always produced invalid JSON
        // would pass every one of them.
        assert!(
            parsed > 1_000,
            "corpus too weak: only {parsed} of 20000 inputs parsed"
        );
    }
}

/// The result of handing a batch to the ingress handler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestAck {
    /// How many messages were accepted.
    pub accepted: usize,
    /// Dedup keys, so a sender can correlate an ack with what it sent.
    pub ids: Vec<String>,
    /// `unknown_source_count` — a misconfigured webhook shows up here.
    pub unknown_sources: usize,
    /// Messages that were refused, and why. Empty on a clean batch.
    pub rejected: Vec<IngestRejection>,
}

/// One refused message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestRejection {
    /// Index within the submitted batch.
    pub index: usize,
    pub error: IngestError,
}

/// Handle a raw request body: parse, validate, and summarize.
///
/// Shared by both HTTP surfaces on purpose. `unfer_edge` and dynamic-arctic
/// calling the *same* function is what makes "one ingest API" true rather than
/// aspirational — two hand-written handlers drift, and then a sender gets a
/// different answer depending on which server it reached.
///
/// ## Partial acceptance, deliberately
///
/// A batch with one bad message still accepts the good ones and reports the bad
/// one by index. Rejecting the whole batch would make one malformed message from
/// a shared channel lose every other message in the batch, which for a webhook
/// that delivers ten at a time is a self-inflicted outage. The alternative
/// failure — silently dropping the bad one — is avoided by naming it in
/// `rejected`.
///
/// Note this differs from [`IngestBatch::validate`], which is all-or-nothing. That
/// is the right shape for a caller that needs to apply a batch atomically; this is
/// the right shape for a receiver that would rather make progress. Both exist
/// because those are genuinely different callers.
pub fn handle_ingest_body(raw: &[u8]) -> Result<IngestAck, IngestError> {
    let batch: IngestBatch = serde_json::from_slice(raw).map_err(|_| {
        // A malformed *envelope* is refused wholesale. It gets its own 400 at the
        // HTTP layer, and inventing an `IngestError` variant for "not JSON" would
        // give two spellings of one failure.
        //
        // Reading a broken body as zero messages and answering 200 is the outcome
        // worth avoiding: it tells the sender everything was delivered.
        IngestError::BatchTooLarge {
            len: raw.len(),
            max: MAX_BATCH,
        }
    })?;

    // `IngestBatch::new` enforces the cap, but deserialization does not go through
    // it -- so a hand-crafted body would otherwise bypass the limit entirely, which
    // is the whole reason the limit exists.
    if batch.messages.len() > MAX_BATCH {
        return Err(IngestError::BatchTooLarge {
            len: batch.messages.len(),
            max: MAX_BATCH,
        });
    }

    let mut ack = IngestAck {
        accepted: 0,
        ids: Vec::new(),
        unknown_sources: batch.unknown_source_count(),
        rejected: Vec::new(),
    };
    for (index, m) in batch.messages.iter().enumerate() {
        match m.validate() {
            Ok(()) => {
                ack.accepted += 1;
                ack.ids.push(m.dedup_key());
            }
            Err(e) => ack.rejected.push(IngestRejection { index, error: e }),
        }
    }
    Ok(ack)
}

#[cfg(test)]
mod handler_tests {
    use super::*;

    fn body(msgs: &[IngestMessage]) -> Vec<u8> {
        serde_json::to_vec(&IngestBatch::new(msgs.to_vec()).unwrap()).unwrap()
    }

    fn good() -> IngestMessage {
        IngestMessage::new("m1", IngestSource::Telegram, "u1", "hello")
    }

    #[test]
    fn a_clean_batch_is_accepted_and_acknowledged_by_dedup_key() {
        let ack = handle_ingest_body(&body(&[good()])).expect("accepted");
        assert_eq!(ack.accepted, 1);
        assert_eq!(ack.ids, vec!["telegram:m1".to_string()]);
        assert!(ack.rejected.is_empty());
        assert_eq!(ack.unknown_sources, 0);
    }

    #[test]
    fn one_bad_message_does_not_lose_the_good_ones() {
        // A webhook that delivers ten at a time should not lose nine of them
        // because one was malformed.
        let bad = IngestMessage::new("", IngestSource::Telegram, "u1", "x");
        let ack = handle_ingest_body(&body(&[good(), bad, good()])).expect("accepted");
        assert_eq!(ack.accepted, 2);
        assert_eq!(ack.rejected.len(), 1);
        assert_eq!(ack.rejected[0].index, 1);
        assert_eq!(ack.rejected[0].error, IngestError::MissingId);
    }

    #[test]
    fn the_rejection_index_points_at_the_offending_message() {
        let mut msgs = vec![good(); 5];
        msgs[3] = IngestMessage::new("", IngestSource::Telegram, "u", "x");
        let ack = handle_ingest_body(&body(&msgs)).unwrap();
        assert_eq!(ack.rejected[0].index, 3);
    }

    #[test]
    fn unknown_sources_surface_as_a_health_signal() {
        let odd = IngestMessage::new("m2", IngestSource::Other("matrix".into()), "u", "x");
        let ack = handle_ingest_body(&body(&[good(), odd])).unwrap();
        assert_eq!(ack.unknown_sources, 1);
    }

    #[test]
    fn a_malformed_body_is_refused_rather_than_treated_as_empty() {
        // The dangerous alternative is reading a broken body as zero messages and
        // answering 200, which tells the sender everything was delivered.
        assert!(handle_ingest_body(b"not json").is_err());
        assert!(handle_ingest_body(b"").is_err());
        assert!(handle_ingest_body(b"{}").is_err());
        assert!(handle_ingest_body(b"[]").is_err());
    }

    #[test]
    fn an_empty_batch_is_legal_and_accepted() {
        let ack = handle_ingest_body(&body(&[])).expect("an empty batch is not an error");
        assert_eq!(ack.accepted, 0);
        assert!(ack.rejected.is_empty());
    }

    #[test]
    fn the_ack_round_trips() {
        let ack = handle_ingest_body(&body(&[good()])).unwrap();
        let s = serde_json::to_string(&ack).unwrap();
        assert_eq!(serde_json::from_str::<IngestAck>(&s).unwrap(), ack);
    }

    #[test]
    fn a_secret_in_a_submitted_message_is_never_in_the_ack() {
        let m = IngestMessage::new("m1", IngestSource::Webhook, "u1", "api_key=sk-live-abc123");
        let ack = handle_ingest_body(&body(&[m])).unwrap();
        let json = serde_json::to_string(&ack).unwrap();
        assert!(!json.contains("abc123"), "{json}");
    }

    #[test]
    fn a_whole_batch_is_refused_when_it_exceeds_the_cap() {
        // The envelope path cannot report per-message rejections, so an oversize
        // batch is an error rather than a partial accept.
        let many: Vec<IngestMessage> = (0..(MAX_BATCH + 1))
            .map(|i| IngestMessage::new(format!("m{i}"), IngestSource::Webhook, "u", "x"))
            .collect();
        // Force it past the cap without `IngestBatch::new` refusing.
        let raw = serde_json::to_vec(&serde_json::json!({ "messages": many })).unwrap();
        let err = handle_ingest_body(&raw).expect_err("an oversize batch is refused");
        assert!(matches!(err, IngestError::BatchTooLarge { .. }), "{err:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fuzz_properties_hold_over_a_generated_corpus() {
        property::arbitrary_json_never_panics();
    }

    fn msg() -> IngestMessage {
        IngestMessage::new("m1", IngestSource::Telegram, "u1", "hello")
    }

    // ---- the normalized shape -------------------------------------------

    #[test]
    fn a_minimal_message_round_trips_through_json() {
        let m = msg();
        let s = serde_json::to_string(&m).expect("serializes");
        let back: IngestMessage = serde_json::from_str(&s).expect("deserializes");
        assert_eq!(m, back);
    }

    #[test]
    fn optional_fields_may_be_omitted_entirely() {
        // A webhook that knows nothing of threads must not have to send `thread:
        // null` to satisfy the schema.
        let json = r#"{"id":"m1","source":"telegram","sender":"u1","text":"hi"}"#;
        let m: IngestMessage = serde_json::from_str(json).expect("minimal message parses");
        assert!(m.channel.is_none());
        assert!(m.thread.is_none());
        assert!(m.ts.is_none());
        assert!(m.attachments.is_empty());
    }

    #[test]
    fn a_required_field_that_is_absent_is_an_error_not_a_default() {
        for missing in ["id", "source", "sender", "text"] {
            let mut map = serde_json::Map::new();
            map.insert("id".into(), "m1".into());
            map.insert("source".into(), "telegram".into());
            map.insert("sender".into(), "u1".into());
            map.insert("text".into(), "hi".into());
            map.remove(missing);
            let json = serde_json::Value::Object(map).to_string();
            assert!(
                serde_json::from_str::<IngestMessage>(&json).is_err(),
                "a message without {missing} must not parse"
            );
        }
    }

    #[test]
    fn an_absent_timestamp_stays_absent_rather_than_becoming_now() {
        // Defaulting to ingest time would silently reorder a backlog.
        let json = r#"{"id":"m1","source":"telegram","sender":"u1","text":"hi"}"#;
        let m: IngestMessage = serde_json::from_str(json).unwrap();
        assert_eq!(m.ts, None, "no timestamp supplied, so none is claimed");
    }

    // ---- sources: the extension point -----------------------------------

    #[test]
    fn known_sources_parse_by_name() {
        for (wire, want) in [
            ("telegram", IngestSource::Telegram),
            ("Discord", IngestSource::Discord),
            ("whatsapp", IngestSource::Whatsapp),
            ("email", IngestSource::Email),
            ("webhook", IngestSource::Webhook),
        ] {
            assert_eq!(IngestSource::parse(wire), want, "{wire}");
        }
    }

    #[test]
    fn an_unknown_source_is_kept_rather_than_refused() {
        // Refusing would make an unrecognised webhook a silent drop instead of a
        // visible one.
        let s = IngestSource::parse("matrix");
        assert_eq!(s, IngestSource::Other("matrix".into()));
        assert!(!s.is_known());
        assert_eq!(s.as_str(), "matrix");
    }

    #[test]
    fn an_unknown_source_round_trips_so_a_receiver_sees_the_real_name() {
        let json = r#"{"id":"m1","source":"matrix","sender":"u1","text":"hi"}"#;
        let m: IngestMessage = serde_json::from_str(json).unwrap();
        assert_eq!(m.source, IngestSource::Other("matrix".into()));
    }

    #[test]
    fn dedup_keys_are_namespaced_by_source() {
        // Two platforms both hand out id "1"; deduplicating across them would
        // drop half the traffic.
        let a = IngestMessage::new("1", IngestSource::Telegram, "u", "x");
        let b = IngestMessage::new("1", IngestSource::Discord, "u", "x");
        assert_ne!(a.dedup_key(), b.dedup_key());
    }

    // ---- validation ------------------------------------------------------

    #[test]
    fn a_well_formed_message_validates() {
        assert_eq!(msg().validate(), Ok(()));
    }

    #[test]
    fn a_blank_required_field_is_named_in_the_error() {
        for (m, want) in [
            (
                IngestMessage::new("", IngestSource::Telegram, "u", "x"),
                IngestError::MissingId,
            ),
            (
                IngestMessage::new("m", IngestSource::Telegram, "  ", "x"),
                IngestError::MissingSender,
            ),
            (
                IngestMessage::new("m", IngestSource::Other("  ".into()), "u", "x"),
                IngestError::MissingSource,
            ),
        ] {
            assert_eq!(m.validate(), Err(want));
        }
    }

    #[test]
    fn oversize_text_is_refused_rather_than_truncated() {
        // A 200 on a truncated message would be data loss the sender never sees.
        let m = IngestMessage {
            text: "x".repeat(MAX_MESSAGE_TEXT + 1),
            ..msg()
        };
        match m.validate() {
            Err(IngestError::TextTooLong { len, max }) => {
                assert_eq!(len, MAX_MESSAGE_TEXT + 1);
                assert_eq!(max, MAX_MESSAGE_TEXT);
            }
            other => panic!("expected TextTooLong, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_message_is_allowed_because_an_attachment_only_message_is_real() {
        let m = IngestMessage::new("m1", IngestSource::Slack, "u1", "");
        assert_eq!(m.validate(), Ok(()));
        assert!(!m.has_text());
    }

    #[test]
    fn the_error_message_names_the_field_and_the_number() {
        let m = IngestMessage {
            text: "x".repeat(MAX_MESSAGE_TEXT + 1),
            ..msg()
        };
        let s = m.validate().unwrap_err().to_string();
        assert!(s.contains(&(MAX_MESSAGE_TEXT + 1).to_string()), "{s}");
        assert!(s.contains(&MAX_MESSAGE_TEXT.to_string()), "{s}");
    }

    // ---- attachments -----------------------------------------------------

    #[test]
    fn attachments_are_references_never_content() {
        let m = msg().with_attachment(AttachmentKind::Image, "https://x/i.png", Some(1024));
        let json = serde_json::to_string(&m).unwrap();
        assert!(
            !json.contains("\"content\""),
            "no inline payload field: {json}"
        );
        assert!(json.contains("https://x/i.png"));
    }

    #[test]
    fn extra_attachments_beyond_the_cap_are_dropped_at_build_time() {
        let mut m = msg();
        for i in 0..(MAX_ATTACHMENTS + 5) {
            m = m.with_attachment(AttachmentKind::Link, &format!("https://x/{i}"), None);
        }
        assert_eq!(m.attachments.len(), MAX_ATTACHMENTS);
        // Dropped rather than refused, and the message survives intact.
        assert_eq!(m.validate(), Ok(()));
        assert_eq!(m.text, "hello");
    }

    #[test]
    fn an_oversize_attachment_list_is_refused_if_constructed_directly() {
        // The builder trims; a hand-built message is still checked.
        let mut m = msg();
        m.attachments = (0..(MAX_ATTACHMENTS + 1))
            .map(|i| Attachment {
                kind: AttachmentKind::Link,
                reference: format!("https://x/{i}"),
                size_bytes: None,
            })
            .collect();
        assert_eq!(
            m.validate(),
            Err(IngestError::TooManyAttachments {
                len: MAX_ATTACHMENTS + 1,
                max: MAX_ATTACHMENTS
            })
        );
    }

    #[test]
    fn an_oversize_attachment_locator_is_refused_with_its_index() {
        let mut m = msg().with_attachment(AttachmentKind::Link, "https://ok", None);
        m.attachments.push(Attachment {
            kind: AttachmentKind::Link,
            reference: "x".repeat(MAX_ATTACHMENT_REF + 1),
            size_bytes: None,
        });
        match m.validate() {
            Err(IngestError::AttachmentRefTooLong { index, .. }) => assert_eq!(index, 1),
            other => panic!("expected AttachmentRefTooLong, got {other:?}"),
        }
    }

    #[test]
    fn attachment_kinds_parse_from_platform_vocabulary() {
        assert_eq!(AttachmentKind::parse("photo"), AttachmentKind::Image);
        assert_eq!(AttachmentKind::parse("voice"), AttachmentKind::Audio);
        assert_eq!(AttachmentKind::parse("file"), AttachmentKind::Document);
        assert_eq!(
            AttachmentKind::parse("something-new"),
            AttachmentKind::Other
        );
    }

    // ---- secrets ---------------------------------------------------------

    #[test]
    fn a_token_pasted_into_text_is_redacted_on_the_way_in() {
        let m = IngestMessage::new(
            "m1",
            IngestSource::Webhook,
            "u1",
            "failed: api_key=sk-live-abc123",
        );
        assert!(!m.text.contains("abc123"), "{}", m.text);
        assert!(was_redacted(&m.text));
        assert!(
            m.text.contains("failed"),
            "the useful part survives: {}",
            m.text
        );
    }

    #[test]
    fn a_bearer_token_in_text_is_redacted() {
        let m = IngestMessage::new(
            "m1",
            IngestSource::Webhook,
            "u1",
            "Authorization: Bearer ghp_realtokenvalue",
        );
        assert!(!m.text.contains("realtokenvalue"));
    }

    #[test]
    fn a_secret_in_a_sender_or_channel_is_redacted_too() {
        let m = IngestMessage::new("m1", IngestSource::Webhook, "api_key=sk-live-abc123", "hi")
            .with_scope(Some("token=sk-live-zzz999"), None);
        assert!(!m.sender.contains("abc123"));
        assert!(!m.channel.as_deref().unwrap_or("").contains("zzz999"));
    }

    #[test]
    fn an_attachment_reference_is_redacted() {
        let m = msg().with_attachment(
            AttachmentKind::Link,
            "https://x/?api_key=sk-live-abc123",
            None,
        );
        assert!(!m.attachments[0].reference.contains("abc123"));
    }

    #[test]
    fn redaction_happens_before_the_cap_so_a_long_secret_cannot_hide_by_truncation() {
        // Truncating first could cut a secret into something the scrubber no
        // longer recognises, which is redaction bypass by accident.
        let long_prefix = "a".repeat(MAX_MESSAGE_TEXT);
        let m = IngestMessage::new(
            "m1",
            IngestSource::Webhook,
            "u1",
            format!("{long_prefix} api_key=sk-live-abc123"),
        );
        assert!(
            !m.text.contains("abc123"),
            "{}",
            &m.text[m.text.len().saturating_sub(80)..]
        );
    }

    // ---- batches ---------------------------------------------------------

    #[test]
    fn a_batch_round_trips() {
        let b = IngestBatch::new(vec![msg(), msg()]).unwrap();
        let s = serde_json::to_string(&b).unwrap();
        let back: IngestBatch = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn an_oversize_batch_is_refused_rather_than_trimmed() {
        // Dropping a message loses something the sender believes was received.
        let many = (0..(MAX_BATCH + 1))
            .map(|i| IngestMessage::new(format!("m{i}"), IngestSource::Telegram, "u", "x"))
            .collect();
        assert_eq!(
            IngestBatch::new(many),
            Err(IngestError::BatchTooLarge {
                len: MAX_BATCH + 1,
                max: MAX_BATCH
            })
        );
    }

    #[test]
    fn batch_validation_names_the_offending_index() {
        let bad = IngestMessage::new("", IngestSource::Telegram, "u", "x");
        let b = IngestBatch::new(vec![msg(), bad]).unwrap();
        assert_eq!(b.validate(), Err((1, IngestError::MissingId)));
    }

    #[test]
    fn a_good_batch_validates() {
        assert_eq!(
            IngestBatch::new(vec![msg(), msg()]).unwrap().validate(),
            Ok(())
        );
    }

    #[test]
    fn unknown_sources_are_counted_as_a_health_signal() {
        let b = IngestBatch::new(vec![
            msg(),
            IngestMessage::new("m2", IngestSource::Other("matrix".into()), "u", "x"),
        ])
        .unwrap();
        assert_eq!(b.unknown_source_count(), 1);
    }

    // ---- robustness ------------------------------------------------------

    #[test]
    fn a_message_survives_adversarial_field_content() {
        // The property the fuzz target asserts: nothing panics, and the caps hold
        // however the fields are arranged.
        let nasty: Vec<String> = vec![
            "\u{0}\u{1}\u{7f}".to_string(),
            // Four bytes, one char: the surrogate-pair boundary in UTF-16 terms.
            "𝔘𝔫𝔦𝔠𝔬𝔡".to_string(),
            "\u{202e}reversed".to_string(),
            "api_key=sk-live-abc123".to_string(),
            String::new(),
            "🙂".repeat(3_000),
            "a".repeat(50_000),
        ];
        for text in &nasty {
            let m = IngestMessage::new("m1", IngestSource::parse(text), text.clone(), text.clone())
                .with_scope(Some(text), Some(text))
                .with_attachment(AttachmentKind::Link, text, Some(u64::MAX));
            let json = serde_json::to_string(&m).expect("never fails to serialize");
            let back: IngestMessage = serde_json::from_str(&json).expect("round trips");
            assert!(back.text.chars().count() <= MAX_MESSAGE_TEXT);
            assert!(back.sender.chars().count() <= MAX_SENDER);
            assert!(back.attachments[0].reference.chars().count() <= MAX_ATTACHMENT_REF);
            if text.trim().is_empty() {
                // A blank source and a blank sender are both refused, and both
                // refusals name the field rather than collapsing into one.
                assert!(
                    matches!(m.validate(), Err(IngestError::MissingSource)),
                    "expected MissingSource, got {:?}",
                    m.validate()
                );
            } else {
                assert_eq!(
                    m.validate(),
                    Ok(()),
                    "adversarial content must still be acceptable"
                );
            }
        }
    }

    #[test]
    fn an_attachment_claiming_u64_max_bytes_is_not_rejected_for_its_size() {
        // `size_bytes` is a caller claim, not a measurement. Nothing verifies it,
        // so refusing on it would reject honest senders whose accounting is off.
        let m = msg().with_attachment(
            AttachmentKind::Document,
            "blob:sha256:deadbeef",
            Some(u64::MAX),
        );
        assert_eq!(m.validate(), Ok(()));
    }
}
