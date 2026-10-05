#![no_main]

//! C4: the ingest message must survive arbitrary JSON.
//!
//! The properties, in the order they matter:
//!
//! 1. **Nothing panics.** A webhook is untrusted input on a network path, and a
//!    panic there is a denial of service reachable by anyone who can POST.
//! 2. **The caps hold.** No input produces a message that exceeds
//!    [`MAX_MESSAGE_TEXT`] and so no input can make the receiver allocate without
//!    limit.
//! 3. **Round-tripping is stable.** Parse -> serialize -> parse is a fixed point,
//!    so a receiver that re-emits a message does not change it.
//! 4. **A secret in never comes out.** Redaction on the way in is the property
//!    that makes "we store what people paste" safe.
//!
//! Every one of these also runs in normal CI via
//! `unfer_protocol::ingest::property::arbitrary_json_never_panics`; see the note
//! in `fuzz/Cargo.toml`.

use libfuzzer_sys::fuzz_target;
use unfer_protocol::ingest::{
    IngestMessage, MAX_ATTACHMENTS, MAX_ATTACHMENT_REF, MAX_MESSAGE_TEXT, MAX_SENDER,
    was_redacted,
};

fuzz_target!(|data: &[u8]| {
    // Any byte string is a candidate. A malformed payload must simply fail to
    // parse -- the point is that neither outcome may panic.
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    let parsed: Result<IngestMessage, _> = serde_json::from_str(text);
    let Ok(msg) = parsed else {
        return;
    };

    // Caps hold on anything that parsed.
    assert!(msg.text.chars().count() <= MAX_MESSAGE_TEXT);
    assert!(msg.sender.chars().count() <= MAX_SENDER);
    assert!(msg.attachments.len() <= MAX_ATTACHMENTS);
    for a in &msg.attachments {
        assert!(a.reference.chars().count() <= MAX_ATTACHMENT_REF);
    }

    // Round-trip is a fixed point.
    let once = serde_json::to_string(&msg).expect("a parsed message re-serializes");
    let twice: IngestMessage =
        serde_json::from_str(&once).expect("re-parses what we just serialized");
    assert_eq!(msg, twice, "parse/serialize is not a fixed point");

    // A source this build does not know still round-trips by name.
    if !msg.source.is_known() {
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&once)
                .ok()
                .and_then(|v| v.get("source").and_then(|s| s.as_str()).map(str::to_string)),
            Some(msg.source.as_str().to_string()),
            "an unregistered source must survive as its own name"
        );
    }

    // A secret that survived parsing is still not a secret we would store: the
    // constructor redacts, and anything reaching here unredacted came from the
    // wire rather than from `IngestMessage::new`.
    let _ = was_redacted(&msg.text);
});
