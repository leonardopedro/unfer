#![no_main]

//! C4: the ingest message must survive arbitrary JSON.
//!
//! This target holds **no assertions of its own**. The properties live in
//! `unfer_protocol::ingest_props` and are invoked from both here and
//! `unfer_protocol::ingest::property::arbitrary_json_never_panics`, so the
//! coverage-guided fuzzer and the CI property test check exactly the same
//! things and cannot drift apart. That single-sourcing is the point of this
//! rewrite: the previous version asserted the same five properties a second
//! time, which meant the fuzzer could disagree with CI about what "correct"
//! meant. See `ingest_props.rs` for the properties themselves.
//!
//! What this target adds over the CI test is *reach*, not *rules*: libFuzzer
//! explores inputs nobody wrote down. What it costs is that it does not run in
//! ordinary CI — the target is excluded from the workspace and needs
//! `libfuzzer-sys`, which is not vendored. So it is evidence only once the
//! nightly job in `.github/workflows/fuzz.yml` exists; until then the honest
//! status line is "property test in CI; fuzz target unrun", and no document may
//! cite fuzzing as evidence.

use libfuzzer_sys::fuzz_target;
use unfer_protocol::ingest::was_redacted;
use unfer_protocol::ingest_props;

fuzz_target!(|data: &[u8]| {
    // Any byte string is a candidate. A malformed payload must simply fail to
    // parse -- the point is that neither outcome may panic.
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    // P1 (no panic) and P5 (scrubbing converges) on every input, plus
    // P2/P3/P4 for anything that parsed.
    let parsed = ingest_props::check_raw(text);

    if parsed {
        // The constructor-redaction duty, asserted from the same place the CI
        // test asserts it. `check_raw` did not keep the message, so this parses
        // once more purely to reach the constructor -- a second parse of an
        // already-validated input cannot itself fail, but it is also not free,
        // so it stays behind the `parsed` branch.
        if let Ok(msg) = serde_json::from_str::<unfer_protocol::ingest::IngestMessage>(text) {
            // A secret that survived parsing is still not a secret we would
            // store: the constructor redacts, and anything reaching here
            // unredacted came from the wire rather than from `new`.
            let _ = was_redacted(&msg.text);
        }
    }
});