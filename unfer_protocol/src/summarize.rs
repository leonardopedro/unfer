//! C6: summarize agent output for a human surface.
//!
//! ## What this is for
//!
//! An agent's raw output is written for a machine that will read all of it. A
//! human surface — an editor overlay, a docs draft, a console — needs the
//! opposite: a few lines that say what changed and whether to believe it. The
//! full trace stays in the session log; this produces the other thing.
//!
//! ## The rule that makes this safe
//!
//! **A summary may not assert more than its trace supports.** That is the whole
//! design constraint, and it is why [`Summary`] carries [`SummaryProvenance]
//! rather than being a bare `String`:
//!
//! - `derived_from` says which records went in, so a reader can ask for them.
//! - `lossy` says whether anything was dropped. A summary that dropped something
//!   and does not say so is the exact failure this module exists to prevent: a
//!   confident one-paragraph answer over a trace that contained a caveat.
//!
//! A caller that cannot tell a complete digest from a truncated one will draw the
//! wrong conclusion from it — the same reasoning as the board's `truncated` flag
//! and the ingest ack's counts.
//!
//! ## Why extractive, not abstractive
//!
//! [`summarize`] selects sentences; it does not write new ones. An LLM
//! summarizer invents prose that can assert something the trace never said, and
//! the failure is invisible because the output *reads* like a summary. Selection
//! cannot do that: every sentence in the output is a sentence that was in the
//! input.
//!
//! [`Summarizer`] is a trait so a model-backed implementation can be dropped in
//! — that is the C7 local-model path — and it must uphold the same contract:
//! populate `derived_from` honestly and set `lossy`. A summarizer that returns
//! provenance-free text is not a faster summarizer, it is an unauditable one.
//!
//! ## It is an observe-kind effect
//!
//! Summarization reads the session and returns text. It changes nothing, so
//! under S21 it is `observe` and never queues for approval. Were it `mutate`, the
//! one part of the loop an agent could not run unattended would be the part that
//! reads its own history.

use serde::{Deserialize, Serialize};

/// Which human surface the text is for.
///
/// A channel is not a formatting preference — it decides how much is dropped, and
/// therefore whether the reader is being misled. `Console` keeps more than
/// `Docs` does, and both report `lossy` when they drop anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    /// A terminal. More room, and an operator who can ask for the trace.
    Console,
    /// A docs page. Least room; least trust that anything was left out.
    Docs,
    /// A document editor overlay, on top of someone's actual work.
    Editor,
}

impl Channel {
    /// Character budget for this surface.
    pub fn budget(self) -> usize {
        match self {
            // A terminal is scrollable and the operator has the trace, so it can
            // afford to be generous.
            Channel::Console => 1_200,
            // An overlay sits on top of someone's document. Anything long enough
            // to need scrolling stops being an overlay.
            Channel::Editor => 320,
            // A docs page is read once, by someone who will not ask for the trace.
            Channel::Docs => 600,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Channel::Console => "console",
            Channel::Docs => "docs",
            Channel::Editor => "editor",
        }
    }

    pub fn parse(s: &str) -> Option<Channel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "console" | "terminal" | "tty" => Some(Channel::Console),
            "docs" | "doc" | "page" => Some(Channel::Docs),
            "editor" | "overlay" | "edit" => Some(Channel::Editor),
            _ => None,
        }
    }
}

/// Where a summary's content came from.
///
/// Cheap to produce and impossible to fake by accident: it is filled in by
/// [`summarize`] from the selection it actually made, not by the caller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummaryProvenance {
    /// 1-based positions of the selected sentences within the source text.
    pub sentences: Vec<usize>,
    /// Source characters considered.
    pub source_chars: usize,
    /// Source sentences considered.
    pub source_sentences: usize,
}

/// The summarized text plus what it is standing in for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    /// The text a human reads.
    pub text: String,
    pub channel: Channel,
    /// Which sentences went in.
    pub derived_from: SummaryProvenance,
    /// True when anything was dropped to fit.
    pub lossy: bool,
    /// Characters of summary produced.
    pub chars: usize,
    /// The channel's budget, so a reader can see the ceiling they are under.
    pub budget: usize,
}

impl Summary {
    /// A one-line account for a log or a status line.
    pub fn explain(&self) -> String {
        format!(
            "{}/{} chars from {}/{} sentences for {} ({}).",
            self.chars,
            self.budget,
            self.derived_from.sentences.len(),
            self.derived_from.source_sentences,
            self.channel.as_str(),
            if self.lossy { "lossy" } else { "complete" }
        )
    }
}

/// Something that can turn a trace into a summary.
///
/// The trait exists so a model-backed summarizer (the C7 path) is a drop-in. It
/// carries an obligation, not just a signature: an implementation must fill in
/// [`Summary::derived_from`] from what it actually used and set
/// [`Summary::lossy`] truthfully. A summarizer that returns confident
/// provenance-free text has not saved effort, it has moved the audit somewhere
/// nobody will look.
pub trait Summarizer {
    fn summarize(&self, trace: &str, channel: Channel) -> Summary;
}

/// Split text into sentences, returning `(text, 1-based position)`.
///
/// Deliberately not a regex and not sentence-*embedding*: splitting on terminal
/// punctuation is predictable, has no dependency, and cannot be defeated by
/// adversarial input into quadratic behaviour. Abbreviations are not special-
/// cased, which costs a little fidelity on "e.g." and buys not having a table of
/// them go stale.
fn sentences(text: &str) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = Vec::new();
    let mut pos = 0usize;
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Only ASCII terminators count, so a multi-byte character's continuation
        // bytes can never be mistaken for one.
        if bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?' || bytes[i] == b'\n' {
            // A `.` is a sentence end only when whitespace or end-of-text follows.
            // Without this, `board.rs` splits into `board.` and `rs.`, `1.5` into
            // `1.` and `5`, and the second fragment scores as a one-word fragment
            // and gets dropped -- which silently loses half of every sentence
            // containing a filename or a version number. That is not a cosmetic
            // bug: agent traces mention paths constantly.
            let is_boundary = if bytes[i] == b'.' {
                // Whitespace, end of text, or a closing quote/bracket. The quote
                // case is what keeps `He said "no." Then he left.` at two
                // sentences instead of one; the whitespace case is what keeps
                // `board.rs` at one instead of two.
                i + 1 >= bytes.len()
                    || (bytes[i + 1] as char).is_whitespace()
                    || matches!(bytes[i + 1], b'"' | b'\'' | b')' | b']' | b'}')
            } else {
                true
            };
            if !is_boundary {
                i += 1;
                continue;
            }
            let mut end = i + 1;
            // Swallow trailing quotes/brackets: `...end."` is one sentence.
            while end < bytes.len() && matches!(bytes[end], b'"' | b'\'' | b')' | b']' | b'}') {
                end += 1;
            }
            let piece = text[start..end].trim();
            if !piece.is_empty() {
                pos += 1;
                out.push((piece.to_string(), pos));
            }
            start = end;
            i = end;
            continue;
        }
        i += 1;
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        pos += 1;
        out.push((tail.to_string(), pos));
    }
    out
}

/// Score a sentence for inclusion.
///
/// Signal strength is deliberately simple and legible: a sentence carrying a
/// number, a file path, a diagnostic code or a verdict word is a sentence
/// carrying *information*, and prose that does none of those is usually
/// connective tissue. This is a heuristic and it is labelled as one; the point is
/// that a reader can look at a selection and see roughly why.
fn score(sentence: &str) -> f64 {
    let mut s = 0.0;
    // Information-bearing markers, in rough order of usefulness.
    for marker in [
        "error", "failed", "fail", "pass", "passed", "ok", "warn", "warning", "refuted", "proved",
        "sorry", "gate", "test", "tests", "commit", "cursor", "claim",
    ] {
        if contains_word(sentence, marker) {
            s += 2.0;
        }
    }
    // A digit means something was measured. Two digits read as a version, an
    // offset, or a count.
    let digits = sentence.chars().filter(|c| c.is_ascii_digit()).count();
    s += (digits.min(4)) as f64 * 0.75;
    // Paths and identifiers.
    if sentence.contains('/') || sentence.contains(".rs") || sentence.contains(".md") {
        s += 1.5;
    }
    // Code-ish punctuation suggests structure worth keeping.
    for marker in ["UK-", "::", "--", "`"] {
        if sentence.contains(marker) {
            s += 1.0;
        }
    }
    // Length: a fragment is rarely worth a line, a paragraph rarely fits.
    let words = sentence.split_whitespace().count();
    if words < 3 {
        s -= 3.0;
    }
    if words > 60 {
        s -= 1.0;
    }
    s
}

/// Substring match on word boundaries, case-insensitive.
fn contains_word(haystack: &str, needle: &str) -> bool {
    let h = haystack.to_lowercase();
    let mut from = 0usize;
    while let Some(rel) = h[from..].find(needle) {
        let start = from + rel;
        let end = start + needle.len();
        let before_ok = start == 0
            || !h[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric());
        let after_ok =
            end >= h.len() || !h[end..].chars().next().is_some_and(|c| c.is_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// Extractively summarize `trace` for `channel`.
pub fn summarize(trace: &str, channel: Channel) -> Summary {
    let budget = channel.budget();
    let sents = sentences(trace);
    let source_chars = trace.chars().count();
    let source_sentences = sents.len();

    if sents.is_empty() {
        return Summary {
            text: String::new(),
            channel,
            derived_from: SummaryProvenance {
                sentences: Vec::new(),
                source_chars,
                source_sentences: 0,
            },
            lossy: source_chars > 0,
            chars: 0,
            budget,
        };
    }

    // Rank, then restore document order for presentation.
    let mut ranked: Vec<(f64, usize, &str)> = sents
        .iter()
        .enumerate()
        .map(|(i, (text, _))| (score(text), i, text.as_str()))
        .collect();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let mut chosen: Vec<usize> = Vec::new();
    let mut used = 0usize;
    // Two spaces of separation, matching how the sentences are re-joined.
    for (rank, i, text) in &ranked {
        // A sentence with no marker, no number, no path and no structure is
        // connective tissue — "Started the run.", "End of run." Filling a budget
        // with those is extraction, not summarization: the reader gets lines that
        // carry no information and still has to read them.
        //
        // So only sentences that scored above zero are eligible. The cost is that
        // an important sentence using none of the vocabulary is dropped, which is
        // why `lossy` exists and why this is a heuristic and not a claim.
        if *rank <= 0.0 {
            continue;
        }
        let cost = text.chars().count() + if chosen.is_empty() { 0 } else { 2 };
        if used + cost > budget {
            // Keep looking: a shorter later sentence may still fit, and stopping
            // at the first overflow would make the result depend on rank order
            // rather than on what fits.
            continue;
        }
        used += cost;
        chosen.push(*i);
    }
    chosen.sort_unstable();

    let text = chosen
        .iter()
        .map(|i| sents[*i].0.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let chars = text.chars().count();

    Summary {
        derived_from: SummaryProvenance {
            sentences: chosen.iter().map(|i| sents[*i].1).collect(),
            source_chars,
            source_sentences,
        },
        // Lossy means "a reader is not seeing everything", which is true whenever
        // a sentence was dropped OR the budget cut one mid-way.
        lossy: chosen.len() < source_sentences,
        chars,
        budget,
        channel,
        text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A realistic trace: mostly connective prose, a few information-bearing
    /// sentences. Deliberately longer than every channel budget, because a fixture
    /// that fits makes "was anything dropped?" untestable — the first version of this
    /// was 260 chars against a 320-char editor budget, so the selection logic was
    /// never actually exercised.
    const TRACE: &str = "Started the run. \
The gate failed: cargo test reported 3 failures in mathed_core. \
I looked at the first failure for a while. \
Commit 5e7b2c6 touched unfer_protocol/src/board.rs. \
Then I read some documentation. \
The redaction fix landed in unfer_protocol/src/board.rs and it closed 3 panics. \
I considered a summarizer model but it is not necessary. \
The board snapshot is now readable across processes. \
Commit a9dd8c3 added uk_memory_read and uk_memory_append to the symbol census. \
There were some warnings during the build. \
tests::a_summary_never_exceeds_its_channel_budget passed. \
I also updated the README counts. \
The unfer_edge ingest route now refuses an oversize batch with UK-4130. \
End of run. \
Nothing else was noteworthy in this run. \
The run took about four minutes in total.";

    // ---- the core property ----------------------------------------------

    #[test]
    fn every_summary_sentence_appears_verbatim_in_the_trace() {
        // The extractive guarantee: selection cannot invent an assertion. A
        // summary whose sentences are not in the input could assert something the
        // trace never said, and the failure is invisible because it reads like a
        // summary.
        for ch in [Channel::Console, Channel::Docs, Channel::Editor] {
            let s = summarize(TRACE, ch);
            for sentence in s.text.split(". ").filter(|x| !x.trim().is_empty()) {
                assert!(
                    TRACE.contains(sentence.trim()),
                    "{ch:?} produced a sentence not in the trace: {sentence:?}"
                );
            }
        }
    }

    #[test]
    fn the_trace_is_never_modified() {
        let before = TRACE.to_string();
        let _ = summarize(TRACE, Channel::Editor);
        assert_eq!(TRACE, before);
    }

    // ---- provenance -------------------------------------------------------

    #[test]
    fn provenance_names_the_sentences_that_were_used() {
        let s = summarize(TRACE, Channel::Editor);
        assert!(
            !s.derived_from.sentences.is_empty(),
            "a summary must say where it came from"
        );
        for p in &s.derived_from.sentences {
            assert!(
                *p >= 1 && *p <= s.derived_from.source_sentences,
                "sentence position {p} is outside 1..={}",
                s.derived_from.source_sentences
            );
        }
    }

    #[test]
    fn provenance_records_the_full_source_not_just_the_selection() {
        let s = summarize(TRACE, Channel::Editor);
        assert_eq!(s.derived_from.source_chars, TRACE.chars().count());
        assert_eq!(s.derived_from.source_sentences, sentences(TRACE).len());
        // And the selection is a strict subset when lossy.
        if s.lossy {
            assert!(s.derived_from.sentences.len() < s.derived_from.source_sentences);
        }
    }

    #[test]
    fn lossy_means_a_reader_is_not_seeing_everything() {
        // The property that stops a truncated digest reading as a complete one.
        let tight = summarize(TRACE, Channel::Editor);
        let loose = summarize(TRACE, Channel::Console);
        assert!(
            tight.lossy,
            "a 320-char budget cannot hold this trace whole"
        );
        assert!(
            loose.chars >= tight.chars,
            "a bigger budget should not summarize less: editor {} vs console {}",
            tight.chars,
            loose.chars
        );
    }

    #[test]
    fn a_trace_that_fits_is_not_lossy() {
        let short = "Gate passed: 12 tests, commit abc1234 touched board.rs.";
        let s = summarize(short, Channel::Console);
        assert!(!s.lossy);
        assert_eq!(
            s.derived_from.sentences.len(),
            s.derived_from.source_sentences
        );
    }

    // ---- selection quality -------------------------------------------------

    #[test]
    fn an_informative_sentence_beats_connective_prose() {
        let s = summarize(TRACE, Channel::Editor);
        assert!(
            s.text.contains("3 failures") || s.text.contains("5e7b2c6"),
            "expected the informative sentence to survive: {:?}",
            s.text
        );
        assert!(
            !s.text.contains("Started the run"),
            "connective prose should not win a tight budget: {:?}",
            s.text
        );
    }

    #[test]
    fn selection_is_deterministic() {
        // A retried summary must not read differently; an operator comparing two
        // attempts would be comparing nothing.
        for ch in [Channel::Console, Channel::Docs, Channel::Editor] {
            assert_eq!(summarize(TRACE, ch), summarize(TRACE, ch));
        }
    }

    #[test]
    fn sentences_come_back_in_document_order_however_they_were_ranked() {
        // Selection is by score; presentation is chronological. Rank order would
        // read as a causal sequence that did not happen.
        let s = summarize(TRACE, Channel::Console);
        let positions = &s.derived_from.sentences;
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "not in document order: {positions:?}"
        );
    }

    // ---- budgets -----------------------------------------------------------

    #[test]
    fn a_summary_never_exceeds_its_channel_budget() {
        let long = "Sentence with a number 42 and a path src/x.rs and a code UK-4601. ".repeat(200);
        for ch in [Channel::Console, Channel::Docs, Channel::Editor] {
            let s = summarize(&long, ch);
            assert!(s.chars <= ch.budget(), "{ch:?} produced {} chars", s.chars);
        }
    }

    #[test]
    fn a_bigger_budget_never_produces_less() {
        let long = TRACE.repeat(4);
        let e = summarize(&long, Channel::Editor).chars;
        let d = summarize(&long, Channel::Docs).chars;
        let c = summarize(&long, Channel::Console).chars;
        assert!(e <= d && d <= c, "editor {e} <= docs {d} <= console {c}");
    }

    #[test]
    fn an_oversized_single_sentence_is_dropped_rather_than_cut_midway() {
        // Cutting mid-sentence would violate the extractive guarantee above, so
        // an over-long sentence has to be left out entirely.
        let huge = "x".repeat(Channel::Editor.budget() + 100);
        let s = summarize(&huge, Channel::Editor);
        assert!(
            s.text.is_empty(),
            "an unfitting sentence must not be truncated"
        );
        assert!(s.lossy);
        assert_eq!(s.chars, 0);
    }

    // ---- edges -------------------------------------------------------------

    #[test]
    fn an_empty_trace_summarizes_to_nothing_and_says_nothing_was_dropped() {
        let s = summarize("", Channel::Console);
        assert!(s.text.is_empty());
        assert!(!s.lossy, "there was nothing to lose");
        assert_eq!(s.derived_from.source_sentences, 0);
    }

    #[test]
    fn whitespace_only_input_is_an_empty_summary_not_a_fragment() {
        let s = summarize("   \n\t  \n ", Channel::Console);
        assert!(s.text.is_empty());
        assert!(s.text.trim().is_empty());
    }

    #[test]
    fn trailing_quotes_stay_with_their_sentence() {
        let t = r#"He said "no." Then he left. It was over."#;
        let sents = sentences(t);
        assert_eq!(sents.len(), 3, "{sents:?}");
        assert!(sents[0].0.ends_with('"'), "{:?}", sents[0].0);
    }

    #[test]
    fn a_dot_inside_a_word_is_not_a_sentence_boundary() {
        // `board.rs` must stay one sentence. Splitting it produced the fragment
        // `rs.`, which scored as a one-word fragment and was dropped -- silently
        // losing the tail of every sentence mentioning a path or a version.
        let s = sentences("Gate passed: commit abc1234 touched board.rs.");
        assert_eq!(s.len(), 1, "{s:?}");
        assert!(s[0].0.ends_with("board.rs."), "{:?}", s[0].0);

        for t in [
            "version 1.5 shipped.",
            "see src/board.rs now.",
            "gate: 3 of 12 tests passed.",
        ] {
            assert_eq!(sentences(t).len(), 1, "{t:?} -> {:?}", sentences(t));
        }
        // Two sentences is correct here: the final `.` IS followed by a space.
        assert_eq!(sentences("a.b.c.d. done.").len(), 2);
    }

    #[test]
    fn a_dot_followed_by_a_quote_still_ends_the_sentence() {
        let s = sentences(r#"He said "no." Then he left."#);
        assert_eq!(s.len(), 2, "{s:?}");
    }

    #[test]
    fn a_multi_byte_character_is_never_split_as_a_terminator() {
        // The scanner works on bytes, so it must only treat ASCII terminators as
        // sentence ends or it will slice a character in half.
        let t = "Emoji 🙂 here. Next sentence with 7 items. Third one.";
        let sents = sentences(t);
        assert_eq!(sents.len(), 3, "{sents:?}");
        for (text, _) in &sents {
            assert!(std::str::from_utf8(text.as_bytes()).is_ok());
        }
    }

    #[test]
    fn adversarial_input_does_not_panic_or_run_away() {
        let nasty: Vec<String> = vec![
            "\u{0}\u{1}\u{7f}".to_string(),
            "🙂".repeat(5_000),
            "a".repeat(50_000),
            ".....".repeat(2_000),
            "?!".repeat(2_000),
            "api_key=sk-live-abc123 and Bearer ghp_realtoken".to_string(),
        ];
        for text in &nasty {
            let s = summarize(text, Channel::Editor);
            assert!(s.chars <= Channel::Editor.budget());
            assert!(s.text.chars().count() == s.chars);
        }
    }

    // ---- channels ----------------------------------------------------------

    #[test]
    fn channels_parse_and_round_trip() {
        for (wire, want) in [
            ("console", Channel::Console),
            ("terminal", Channel::Console),
            ("docs", Channel::Docs),
            ("overlay", Channel::Editor),
            ("EDITOR", Channel::Editor),
        ] {
            assert_eq!(Channel::parse(wire), Some(want), "{wire}");
        }
        assert_eq!(Channel::parse("nonsense"), None);
        for ch in [Channel::Console, Channel::Docs, Channel::Editor] {
            assert_eq!(Channel::parse(ch.as_str()), Some(ch));
        }
    }

    #[test]
    fn channels_are_ordered_by_how_much_they_can_hold() {
        assert!(Channel::Editor.budget() < Channel::Docs.budget());
        assert!(Channel::Docs.budget() < Channel::Console.budget());
    }

    // ---- reporting ---------------------------------------------------------

    #[test]
    fn a_summary_explains_itself_for_a_log_line() {
        let s = summarize(TRACE, Channel::Editor);
        let e = s.explain();
        assert!(e.contains("editor"), "{e}");
        assert!(e.contains("lossy"), "{e}");
        assert!(e.contains(&s.chars.to_string()), "{e}");
    }

    #[test]
    fn a_summary_round_trips_through_json() {
        let s = summarize(TRACE, Channel::Console);
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Summary>(&j).unwrap(), s);
    }

    #[test]
    fn the_function_is_a_summarizer_so_a_model_backed_one_is_a_drop_in() {
        // The trait is the C7 extension point; if this stopped compiling, adding a
        // local-model summarizer would mean changing every call site.
        let s: Summary = Extractive.summarize(TRACE, Channel::Console);
        assert_eq!(s, summarize(TRACE, Channel::Console));
    }

    /// The default implementation, named so the trait has a referent.
    struct Extractive;

    impl Summarizer for Extractive {
        fn summarize(&self, trace: &str, channel: Channel) -> Summary {
            summarize(trace, channel)
        }
    }
}
