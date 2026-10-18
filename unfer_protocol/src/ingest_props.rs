//! The ingest properties, stated exactly once.
//!
//! ## Why this module exists
//!
//! C4 says an `IngestMessage` must hold up under arbitrary input. That claim is
//! checked in **two** places, which is the correct amount — a fast deterministic
//! test that runs in ordinary CI, and a coverage-guided fuzzer that explores
//! inputs nobody thought to write. Two *assertions* of the same property in two
//! places is one too many: the second copy is free to drift, and a drifted copy
//! either fails on something the first one already covers (noise) or, worse,
//! quietly stops checking what it was written to check.
//!
//! So the properties live here, and both callers invoke *these* functions. There
//! is deliberately no second copy of any assertion below; if you find yourself
//! adding one, add it here instead.
//!
//! ## The properties
//!
//! - **P1 — parsing never panics.** Any `&str` either parses or is rejected.
//!   A rejection is not a failure: an ingress must survive hostile bytes.
//! - **P2 — the caps hold on anything that parsed.** Length and count limits are
//!   enforced *by parsing*, not by callers remembering to apply them, so there is
//!   no path that builds an over-long message.
//! - **P3 — parse → serialize → parse is a fixed point.** A normalized message
//!   is still a valid message, and re-serializing it does not drift. This is what
//!   makes `board_write`/`memory_append` idempotent on replay.
//! - **P4 — an unregistered source survives as its own name.** `IngestSource::Other`
//!   must not be flattened into a placeholder, or adding a platform becomes a
//!   data-loss event instead of a config change.
//! - **P5 — scrubbing converges in one pass.** [`redaction_is_idempotent`].
//!   Parsing re-redacts every string field, so P3 holds only because
//!   re-redacting already-redacted text is a no-op; when it is not, a
//!   secret-shaped field drifts a little further on every serialize→parse hop
//!   and the "fixed point" is a slow erosion. The fuzzer found exactly that on
//!   2026-10-18 — the first day the fuzz crate could compile at all.
//!
//! ## A check that is not a property
//!
//! The constructor redaction ([`IngestMessage::new`]) is verified separately, in
//! `ingest.rs`'s own tests. It is a statement about *construction*, not about
//! parsing, and it cannot be reached from the fuzzer — which is precisely why it
//! is asserted where it is and not here. See [`redacted_on_construction`].

use crate::board::redact_secrets;
use crate::ingest::{
    IngestMessage, MAX_ATTACHMENT_REF, MAX_ATTACHMENTS, MAX_MESSAGE_TEXT, MAX_SENDER,
};

/// Run P5: scrubbing an already-scrubbed string changes nothing.
///
/// Split out like P2/P3 so a caller with a scrubber question does not have to
/// build a message to ask it. The fuzzer reaches this on every byte string it
/// tries — including strings that are not JSON at all, which is where P3 cannot
/// look and the only reason this is a property in its own right.
pub fn redaction_is_idempotent(raw: &str) {
    let once = redact_secrets(raw);
    let twice = redact_secrets(&once);
    assert_eq!(once, twice, "redaction is not idempotent");
}

/// Run P2 and P3 against a message that is known to have parsed.
///
/// Split out so that P3 — the round-trip fixed point — can be reused on its own:
/// the round-trip is the property that matters for replay, and a caller checking
/// only idempotence should not have to also assert the caps to get it.
pub fn caps_hold_and_round_trip_is_a_fixed_point(msg: &IngestMessage) {
    // P2. The caps are properties of parsing, so a message that exists at all
    // already satisfies them.
    assert!(
        msg.text.chars().count() <= MAX_MESSAGE_TEXT,
        "text cap broken: {} chars",
        msg.text.chars().count()
    );
    assert!(
        msg.sender.chars().count() <= MAX_SENDER,
        "sender cap broken: {} chars",
        msg.sender.chars().count()
    );
    assert!(
        msg.attachments.len() <= MAX_ATTACHMENTS,
        "attachment count cap broken: {}",
        msg.attachments.len()
    );
    for a in &msg.attachments {
        assert!(
            a.reference.chars().count() <= MAX_ATTACHMENT_REF,
            "attachment ref cap broken: {} chars",
            a.reference.chars().count()
        );
    }

    // P3. Round-trip is a fixed point.
    let once = serde_json::to_string(msg).expect("a parsed message re-serializes");
    let twice: IngestMessage =
        serde_json::from_str(&once).expect("re-parses what it just serialized");
    assert_eq!(*msg, twice, "parse/serialize is not a fixed point");
}

/// Run P4: a source this build does not know still round-trips by name.
pub fn an_unregistered_source_survives(msg: &IngestMessage) {
    if msg.source.is_known() {
        return;
    }
    let once = serde_json::to_string(msg).expect("a parsed message re-serializes");
    let seen = serde_json::from_str::<serde_json::Value>(&once)
        .ok()
        .and_then(|v| v.get("source").and_then(|s| s.as_str()).map(str::to_string));
    assert_eq!(
        seen,
        Some(msg.source.as_str().to_string()),
        "an unregistered source must survive as its own name, not a placeholder"
    );
}

/// Run P2, P3 and P4 on one already-parsed message.
///
/// This is the entry point both callers use for a message that parsed. P1 is not
/// checkable here — it is a statement about the *absence* of a panic, which only
/// the caller's `from_str` call can witness.
pub fn check_parsed(msg: &IngestMessage) {
    caps_hold_and_round_trip_is_a_fixed_point(msg);
    an_unregistered_source_survives(msg);
}

/// Check one raw input end to end: P1 and P5 on the raw bytes, then P2–P4 if
/// it parsed.
///
/// Returns `true` when the input parsed, so a caller can assert its corpus is
/// actually exercising the parser rather than passing vacuously — a generator
/// that always produced invalid JSON would satisfy P1..P4 trivially. P5 runs
/// either way: scrubbing is defined on arbitrary text, not only on JSON.
pub fn check_raw(raw: &str) -> bool {
    // P5. Scrubbing must converge in one pass, and it runs on raw text before
    // anything below can look at it.
    redaction_is_idempotent(raw);

    // P1. Any failure here is a *result*, not an error: the point is that the
    // parser returns instead of unwinding.
    let Ok(msg) = serde_json::from_str::<IngestMessage>(raw) else {
        return false;
    };
    check_parsed(&msg);
    true
}

/// The constructor's redaction duty, for callers that can construct.
///
/// Not a parsing property — see the module docs — but it belongs next to the
/// others because it is the fourth thing an ingress must never get wrong, and
/// having it here means a fuzzer harness that *can* reach the constructor uses
/// the same assertion the CI test uses.
pub fn redacted_on_construction(raw: &str, secret_marker: &str) -> IngestMessage {
    let Ok(parsed) = serde_json::from_str::<IngestMessage>(raw) else {
        // Nothing to construct from. Return an empty message rather than
        // panicking: a fuzzer that hits this must not be reported as a finding.
        return IngestMessage::new(
            "fuzz",
            crate::ingest::IngestSource::Other("fuzz".into()),
            "f",
            "",
        );
    };
    let built = IngestMessage::new(
        parsed.id.clone(),
        parsed.source.clone(),
        parsed.sender.clone(),
        parsed.text.clone(),
    );
    assert!(
        !built.text.contains(secret_marker),
        "a secret reached the store: {}",
        built.text
    );
    built
}
#[cfg(test)]
mod tests {
    use super::*;

    fn raw_with_text(n: usize) -> String {
        format!(
            r#"{{"id":"m1","source":"webhook","sender":"u1","text":"{}"}}"#,
            "x".repeat(n)
        )
    }

    /// P2 has teeth only if an over-long field is capped by *parsing*. This test
    /// is the evidence for that claim, and it is written so that if the claim is
    /// ever quietly dropped, this fails rather than the property going vacuous.
    #[test]
    fn parsing_caps_an_over_long_text_field() {
        let raw = raw_with_text(MAX_MESSAGE_TEXT * 2);
        let parsed = check_raw(&raw);
        assert!(
            parsed,
            "an over-long message should still parse, just be capped"
        );
    }

    #[test]
    fn garbage_does_not_parse_and_saying_so_is_the_point() {
        assert!(!check_raw(""));
        assert!(!check_raw("not json"));
        assert!(!check_raw("{\"id\":"));
    }

    /// The counter-case to the test above. If this one starts failing, P2 became
    /// vacuous in the other direction: the cap is applied by rejecting, not by
    /// truncating, and the fixed-point property would then be untested on long
    /// messages.
    #[test]
    fn a_long_message_is_truncated_not_rejected() {
        let raw = raw_with_text(MAX_MESSAGE_TEXT * 2);
        let msg: IngestMessage = serde_json::from_str(&raw).expect("parses");
        assert_eq!(
            msg.text.chars().count(),
            MAX_MESSAGE_TEXT,
            "an over-long text must be truncated to the cap, not kept whole"
        );
    }

    /// P2 was vacuous *per field*, not just for `text`: the 20k corpus generated
    /// short values for everything, so an uncapped field anywhere would have gone
    /// unnoticed. These assert each capped field is capped **by parsing**, which
    /// is the only form in which P2 says anything.
    #[test]
    fn every_capped_field_is_capped_by_parsing() {
        // Longer than the largest cap in the type (MAX_MESSAGE_TEXT = 8000), so
        // every capped field is genuinely over its limit here. A shorter filler
        // would make this test pass vacuously — which is precisely the bug it
        // exists to catch.
        let big = "y".repeat(9_000);
        let raw = format!(
            r#"{{"id":"{big}","source":"webhook","sender":"{big}","text":"{big}","attachments":[{}]}}"#,
            (0..MAX_ATTACHMENTS + 4)
                .map(|_| format!(r#"{{"kind":"image","reference":"{big}"}}"#))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(check_raw(&raw), "the message parses, just capped");

        let msg: IngestMessage = serde_json::from_str(&raw).expect("parses");
        assert_eq!(msg.id.chars().count(), 128, "id cap: MAX_SCOPE");
        assert_eq!(msg.sender.chars().count(), MAX_SENDER, "sender cap");
        assert_eq!(msg.text.chars().count(), MAX_MESSAGE_TEXT, "text cap");
        assert_eq!(
            msg.attachments.len(),
            MAX_ATTACHMENTS,
            "attachment count cap"
        );
        for a in &msg.attachments {
            assert_eq!(
                a.reference.chars().count(),
                MAX_ATTACHMENT_REF,
                "attachment reference cap"
            );
        }
    }

    /// Parsing redacts too, not just caps. This is the half of the ingress duty
    /// that a cap cannot provide: a secret arrives, and it must not be *stored*
    /// just because it happened to be short.
    #[test]
    fn parsing_redacts_before_capping() {
        let raw = r#"{"id":"m1","source":"webhook","sender":"u1",
            "text":"api_key=sk-live-abc123"}"#;
        let msg: IngestMessage = serde_json::from_str(raw).expect("parses");
        assert!(
            !msg.text.contains("abc123"),
            "a secret survived parsing: {}",
            msg.text
        );
        check_parsed(&msg);
    }
}
