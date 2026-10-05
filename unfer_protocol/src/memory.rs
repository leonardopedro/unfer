//! C2: bounded agent memory with retrieval.
//!
//! ## The shape of the problem
//!
//! A long agent session accumulates more than fits in a prompt. The two obvious
//! answers are both bad: truncate (and the agent forgets the thing that mattered)
//! or grow (and the prompt stops being readable, or the context window runs out).
//!
//! So: a bounded store, and retrieval when it overflows. Under the cap you get
//! everything, because there is nothing to choose. Over it you get the chunks
//! that match what you asked for, which is the only reason to keep a history in
//! the first place.
//!
//! ## Determinism is the requirement, not a nicety
//!
//! Retrieval here is BM25 over the records. Two properties are load-bearing:
//!
//! - **Same query, same answer.** An agent that re-reads its memory gets the same
//!   chunks, so a retry does not change what it concluded. This is why ties break
//!   on record id rather than on hash order.
//! - **No embeddings, no model.** A pure lexical score is inspectable: you can
//!   read why a chunk was returned. That matters more here than semantic quality,
//!   because the caller is an agent whose next action depends on the result and
//!   whose operator has to be able to audit it.
//!
//! ## Secrets
//!
//! Every record passes [`crate::board::redact_secrets`] on the way in. The board
//! needs this because it is many workers pasting output; memory needs it because
//! it is the *same* text kept for longer. The S27 rule holds: vault material is
//! opaque and non-serializable, and never reaches here in the first place.
//! Redaction here is the second line, not the first.
//!
//! ## What "full" means
//!
//! Under the cap, [`Memory::read`] returns every record, oldest first, and says
//! `truncated: false`. Over it, `truncated: true` and the result is whatever fit.
//! The counts travel with every read, because a caller that cannot tell a short
//! history from a pruned one will draw the wrong conclusion from it.

use serde::{Deserialize, Serialize};

/// Characters retained across all records.
pub const MEMORY_CAP: usize = 8_000;

/// Characters per record. Generous for a summary, small enough that one runaway
/// record cannot evict everything else.
pub const RECORD_CAP: usize = 600;

/// Records retained. The character cap is the real bound; this keeps the
/// per-record scan cheap.
pub const RECORD_LIMIT: usize = 512;

/// BM25 term-saturation parameter. The conventional 1.2.
const K1: f32 = 1.2;
/// BM25 length-normalisation parameter. The conventional 0.75.
const B: f32 = 0.75;

/// One remembered thing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Monotonic, assigned on append. The tie-break key, and what makes
    /// retrieval reproducible.
    pub id: u64,
    /// Who wrote it.
    pub worker: String,
    /// The text, redacted and capped.
    pub text: String,
}

impl MemoryRecord {
    pub fn chars(&self) -> usize {
        self.text.chars().count()
    }

    /// Lowercased terms, for scoring.
    fn terms(&self) -> Vec<String> {
        tokenize(&self.text)
    }
}

/// Split text into lowercase terms.
///
/// Deliberately not a regex and not a stop-word list: a stop-word list is a
/// judgement about language that this corpus does not justify, and the common
/// words in a worker's memory ("the", "is") are already discounted by IDF.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// A bounded, retrievable memory.
#[derive(Debug, Clone, Default)]
pub struct Memory {
    records: Vec<MemoryRecord>,
    next_id: u64,
    /// Total characters currently held.
    total: usize,
    /// How many records were evicted by the cap. Reported, never silent.
    evicted: u64,
}

impl Memory {
    pub fn new() -> Memory {
        Memory {
            records: Vec::new(),
            // Ids start at 1, matching the board cursor convention.
            next_id: 1,
            total: 0,
            evicted: 0,
        }
    }

    /// Append a record, redacting and capping it.
    pub fn append(&mut self, worker: &str, text: &str) -> MemoryRecord {
        let id = self.next_id;
        self.next_id += 1;
        let rec = MemoryRecord {
            id,
            worker: worker.to_string(),
            text: cap_chars(&crate::board::redact_secrets(text), RECORD_CAP),
        };
        self.total += rec.chars();
        self.records.push(rec.clone());
        self.evict();
        rec
    }

    /// Drop oldest-first until both caps are satisfied.
    ///
    /// Oldest-first because recent context is what an agent is working from, and
    /// because dropping the newest record would discard the thing the agent just
    /// learned.
    fn evict(&mut self) {
        while self.records.len() > RECORD_LIMIT || self.total > MEMORY_CAP {
            let Some(old) = self.records.first().cloned() else {
                break;
            };
            self.total -= old.chars();
            self.records.remove(0);
            self.evicted += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn total_chars(&self) -> usize {
        self.total
    }

    pub fn evicted(&self) -> u64 {
        self.evicted
    }

    pub fn records(&self) -> &[MemoryRecord] {
        &self.records
    }

    /// Whether history has been lost.
    ///
    /// Note this is *not* `total > MEMORY_CAP`: [`Memory::append`] evicts to stay
    /// under the cap, so the literal condition is never observable from outside.
    /// What matters is whether records have actually been dropped, and that is
    /// what decides whether a read has to choose.
    pub fn history_truncated(&self) -> bool {
        self.evicted > 0
    }

    /// Read memory, retrieving when over the cap.
    ///
    /// Under the cap this is everything. Over it, the highest-scoring records
    /// that fit in `budget`, most relevant first. An empty or whitespace query
    /// has no relevance signal, so it falls back to **most recent first** — the
    /// one ordering that is right when you do not know what you are looking for.
    pub fn read(&self, query: &str, budget: usize) -> MemoryRead {
        let query_terms = tokenize(query);
        // Nothing was ever dropped and it fits: there is no choice to make.
        let complete = !self.history_truncated() && self.total <= budget;

        if complete {
            return MemoryRead {
                records: self.records.clone(),
                truncated: false,
                total_chars: self.total,
                returned_chars: self.total,
                evicted: self.evicted,
                matched: self.records.len(),
                retrieved: false,
            };
        }

        let mut chosen: Vec<MemoryRecord> = Vec::new();
        let mut used = 0usize;
        let matched: usize = if query_terms.is_empty() {
            // No relevance signal. The one ordering that is right when you do not
            // know what you are looking for is *most recent first*, so the budget
            // is spent on the newest records rather than the oldest.
            //
            // This deliberately does not go through the score sort below: with
            // every score equal, the id tie-break sorts ascending and silently
            // hands back the oldest records instead.
            let n = self.records.len();
            for r in self.records.iter().rev() {
                let c = r.chars();
                if used + c > budget {
                    continue;
                }
                chosen.push(r.clone());
                used += c;
            }
            n
        } else {
            let mut ranked: Vec<(f32, &MemoryRecord)> = self
                .records
                .iter()
                .map(|r| (score(r, &query_terms, &self.records), r))
                .filter(|(s, _)| *s > 0.0)
                .collect();
            // Ties break on id, so the order never depends on iteration order.
            ranked.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.id.cmp(&b.1.id))
            });
            let n = ranked.len();
            for (_, r) in &ranked {
                let c = r.chars();
                if used + c > budget {
                    continue;
                }
                chosen.push((*r).clone());
                used += c;
            }
            n
        };
        // Selection was by relevance; *presentation* is chronological. A reader
        // looking at several records needs them in the order they happened, and
        // returning them in rank order would read as a causal sequence that did
        // not occur.
        chosen.sort_by_key(|r| r.id);
        MemoryRead {
            records: chosen,
            truncated: matched < self.records.len() || used < self.total,
            total_chars: self.total,
            returned_chars: used,
            evicted: self.evicted,
            matched,
            retrieved: true,
        }
    }
}

/// What a read produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRead {
    /// Oldest first within the result, so a reader sees them in the order they
    /// happened even though they were *selected* by relevance.
    pub records: Vec<MemoryRecord>,
    /// True when anything was left out.
    pub truncated: bool,
    pub total_chars: usize,
    pub returned_chars: usize,
    /// Records evicted from the store entirely, over time.
    pub evicted: u64,
    /// How many records matched before the budget was applied.
    pub matched: usize,
    /// Whether retrieval ran at all (false = you got everything).
    pub retrieved: bool,
}

impl MemoryRead {
    /// A sentence a caller can act on, for an editor overlay.
    pub fn explain(&self) -> String {
        if !self.retrieved {
            return format!(
                "{} record(s), {} chars, complete",
                self.records.len(),
                self.returned_chars
            );
        }
        format!(
            "retrieved {} of {} matching record(s) ({} of {} chars); {} evicted from \
             the store entirely",
            self.records.len(),
            self.matched,
            self.returned_chars,
            self.total_chars,
            self.evicted
        )
    }
}

/// BM25 score for one record against the query terms.
///
/// Scored against *all* records for the IDF, including the one being scored --
/// that is what BM25 specifies, and it keeps the function pure.
fn score(rec: &MemoryRecord, query_terms: &[String], corpus: &[MemoryRecord]) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let n = corpus.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let terms = rec.terms();
    let len = terms.len() as f32;
    if len == 0.0 {
        return 0.0;
    }
    let avg_len = corpus.iter().map(|r| r.terms().len() as f32).sum::<f32>() / n;

    let mut total = 0.0f32;
    for q in query_terms {
        let tf = terms.iter().filter(|t| *t == q).count() as f32;
        if tf == 0.0 {
            continue;
        }
        // Documents containing the term, for IDF.
        let df = corpus
            .iter()
            .filter(|r| r.terms().iter().any(|t| t == q))
            .count() as f32;
        // BM25's IDF with the +1 that keeps it non-negative for very common terms.
        let idf = (((n - df + 0.5) / (df + 0.5)) + 1.0).ln();
        let denom = tf + K1 * (1.0 - B + B * (len / avg_len.max(1.0)));
        total += idf * (tf * (K1 + 1.0)) / denom.max(f32::EPSILON);
    }
    total
}

/// Truncate on a char boundary to `max`, marking that it happened.
fn cap_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(entries: &[(&str, &str)]) -> Memory {
        let mut m = Memory::new();
        for (w, t) in entries {
            m.append(w, t);
        }
        m
    }

    /// Fill past the character cap with distinct, findable records.
    ///
    /// The padding matters: an earlier version of this used 40 short records,
    /// which is 2.7k characters against an 8k cap, so it never overflowed and
    /// the "over the cap" tests were quietly testing the under-cap path.
    fn overflowing() -> Memory {
        let mut m = Memory::new();
        for i in 0..40 {
            m.append(
                "w1",
                &format!(
                    "routine step {i} of the ordinary pipeline {}",
                    "padding ".repeat(25)
                ),
            );
        }
        m
    }

    // ---- under the cap ----------------------------------------------------

    #[test]
    fn under_the_cap_a_read_returns_everything() {
        let m = mem(&[
            ("w1", "the Faris-Lavine route is refuted"),
            ("w1", "nanoda re-verifies the export"),
        ]);
        let r = m.read("anything", MEMORY_CAP);
        assert_eq!(r.records.len(), 2);
        assert!(!r.truncated);
        assert!(!r.retrieved, "no choice was needed");
        assert_eq!(r.total_chars, r.returned_chars);
    }

    #[test]
    fn a_read_explains_itself() {
        let m = mem(&[("w1", "a fact worth keeping")]);
        assert!(m.read("x", MEMORY_CAP).explain().contains("complete"));
        assert!(overflowing().read("x", 400).explain().contains("retrieved"));
    }

    // ---- over the cap -----------------------------------------------------

    #[test]
    fn an_empty_memory_reads_empty_without_error() {
        let m = Memory::new();
        let r = m.read("anything", MEMORY_CAP);
        assert!(r.records.is_empty());
        assert!(!r.truncated);
    }

    #[test]
    fn over_the_cap_retrieval_runs_and_says_so() {
        let m = overflowing();
        assert!(
            m.history_truncated(),
            "the filler must actually force eviction"
        );
        let r = m.read("pipeline", 500);
        assert!(r.retrieved);
        assert!(r.records.len() < m.len(), "not everything fits");
    }

    #[test]
    fn the_relevant_chunks_are_the_ones_returned() {
        let mut m = overflowing();
        m.append(
            "w1",
            "the QYM one-particle form gap is the outstanding input",
        );
        let r = m.read("form gap QYM", 400);
        assert!(r.retrieved);
        let texts: Vec<&str> = r.records.iter().map(|x| x.text.as_str()).collect();
        assert!(
            texts.iter().any(|t| t.contains("form gap")),
            "the matching record should be retrieved: {texts:?}"
        );
    }

    #[test]
    fn retrieval_is_deterministic() {
        // An agent that re-reads must get the same chunks, or a retry changes
        // what it concluded.
        let m = overflowing();
        let a = m.read("routine pipeline step", 400);
        let b = m.read("routine pipeline step", 400);
        let c = m.read("routine pipeline step", 400);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn retrieval_respects_the_budget() {
        let m = overflowing();
        let r = m.read("routine", 300);
        assert!(r.returned_chars <= 300, "{}", r.returned_chars);
        assert!(r.truncated);
    }

    #[test]
    fn results_come_back_oldest_first_even_though_they_were_selected_by_rank() {
        let m = overflowing();
        let r = m.read("routine step 3", 300);
        let ids: Vec<u64> = r.records.iter().map(|x| x.id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(
            ids, sorted,
            "a reader should see them in the order they happened"
        );
    }

    #[test]
    fn an_empty_query_falls_back_to_the_most_recent() {
        // No signal about what is wanted, so the one ordering that is right when
        // you do not know is most-recent-first.
        let m = overflowing();
        let r = m.read("   ", 300);
        assert!(r.retrieved);
        assert!(!r.records.is_empty());
        let newest = r.records.last().expect("some records").id;
        assert_eq!(
            newest,
            m.records().last().unwrap().id,
            "the newest record should be included"
        );
    }

    #[test]
    fn a_query_matching_nothing_returns_nothing_rather_than_everything() {
        let m = overflowing();
        let r = m.read("zzzznonexistentterm", 400);
        assert!(r.records.is_empty());
        assert!(
            r.truncated,
            "and it must not pretend it returned everything"
        );
    }

    #[test]
    fn a_tiny_budget_returns_nothing_but_still_reports_cleanly() {
        let m = overflowing();
        let r = m.read("routine", 1);
        assert!(r.records.is_empty());
        assert!(r.truncated);
        assert!(r.total_chars > 0, "the store is not empty; the budget was");
    }

    #[test]
    fn a_record_too_large_for_the_budget_is_skipped_not_truncated_midway() {
        let mut m = overflowing();
        m.append("w1", &"x".repeat(RECORD_CAP));
        let r = m.read("routine", 100);
        // Every returned record must be whole.
        for rec in &r.records {
            assert!(rec.chars() <= 100, "a record was cut: {}", rec.chars());
        }
    }

    // ---- bounds -----------------------------------------------------------

    #[test]
    fn the_store_is_bounded_by_characters() {
        let m = overflowing();
        assert!(m.total_chars() <= MEMORY_CAP, "{}", m.total_chars());
        assert!(m.len() <= RECORD_LIMIT);
    }

    #[test]
    fn eviction_drops_oldest_first_and_counts_what_it_lost() {
        let mut m = Memory::new();
        for i in 0..(RECORD_LIMIT + 10) {
            m.append("w1", &format!("entry {i}"));
        }
        assert_eq!(m.len(), RECORD_LIMIT);
        assert_eq!(m.evicted(), 10);
        // The survivors are the newest.
        assert_eq!(m.records()[0].id, 11);
        assert_eq!(m.records().last().unwrap().id, RECORD_LIMIT as u64 + 10);
    }

    #[test]
    fn one_oversized_record_cannot_evict_everything_else() {
        let mut m = mem(&[("w1", "an older fact"), ("w1", "another older fact")]);
        m.append("w1", &"y".repeat(MEMORY_CAP * 2));
        // The huge record is capped on the way in, so it cannot dominate.
        assert!(m.total_chars() <= MEMORY_CAP + RECORD_CAP);
        assert!(m.len() >= 1);
    }

    #[test]
    fn a_single_record_is_capped_on_the_way_in() {
        let mut m = Memory::new();
        let r = m.append("w1", &"z".repeat(RECORD_CAP * 3));
        assert_eq!(r.text.chars().count(), RECORD_CAP);
        assert!(r.text.ends_with('…'));
    }

    #[test]
    fn truncation_keeps_valid_utf8() {
        let m0 = Memory::new();
        let mut m = m0;
        let r = m.append("w1", &"π".repeat(RECORD_CAP + 50));
        assert_eq!(r.text.chars().count(), RECORD_CAP);
        assert!(std::str::from_utf8(r.text.as_bytes()).is_ok());
    }

    #[test]
    fn ids_are_unique_and_monotonic() {
        let m = mem(&[("w1", "a"), ("w1", "b"), ("w1", "c")]);
        let ids: Vec<u64> = m.records().iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    // ---- secrets ----------------------------------------------------------

    #[test]
    fn a_secret_is_redacted_before_it_is_remembered() {
        let mut m = Memory::new();
        m.append("w1", "the zenodo push failed; api_key=sk-live-abc123");
        let stored = &m.records()[0].text;
        assert!(!stored.contains("abc123"), "{stored}");
        assert!(
            stored.contains("zenodo"),
            "the useful part survives: {stored}"
        );

        // Retrieval cannot resurface it either. A complete read deliberately does
        // no filtering -- it returns everything -- so the meaningful check is that
        // nothing in the store *contains* the secret, and that a targeted query
        // for it scores zero against every record.
        let mut big = overflowing();
        big.append("w1", "the zenodo push failed; api_key=sk-live-abc123");
        for r in big.records() {
            assert!(!r.text.contains("abc123"), "{}", r.text);
        }
        assert!(big.read("abc123", 400).records.is_empty());
    }

    #[test]
    fn a_bearer_token_never_reaches_the_store() {
        let mut m = Memory::new();
        m.append("w1", "Authorization: Bearer ghp_realtokenvalue123");
        assert!(!m.records()[0].text.contains("realtokenvalue123"));
    }

    #[test]
    fn ordinary_content_is_never_redacted() {
        // A memory full of redactions is a memory nobody can use.
        let text = "commit 7fa2b6c touched ChapterFoo.lean; no sorry remains";
        let mut m = Memory::new();
        let r = m.append("w1", text);
        assert_eq!(r.text, text);
    }

    // ---- scoring ----------------------------------------------------------

    #[test]
    fn a_rare_term_outranks_a_common_one() {
        let mut m = Memory::new();
        for _ in 0..20 {
            m.append("w1", "the ordinary pipeline advanced");
        }
        m.append("w1", "a quixotic anachronism appeared");
        let r = m.read("quixotic", 400);
        assert!(
            r.records[0].text.contains("quixotic"),
            "the distinctive record should rank first: {:?}",
            r.records[0].text
        );
    }

    #[test]
    fn tokenization_splits_on_punctuation_but_keeps_paths() {
        assert_eq!(tokenize("a, b; c"), vec!["a", "b", "c"]);
        assert_eq!(tokenize("src/board.rs"), vec!["src", "board.rs"]);
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn scoring_is_case_insensitive() {
        let rec = MemoryRecord {
            id: 1,
            worker: "w".into(),
            text: "The Gap Certificate".into(),
        };
        let corpus = vec![rec.clone()];
        assert!(score(&rec, &tokenize("GAP"), &corpus) > 0.0);
        assert_eq!(
            score(&rec, &tokenize("GAP"), &corpus),
            score(&rec, &tokenize("gap"), &corpus)
        );
    }

    #[test]
    fn an_empty_record_scores_zero_rather_than_dividing_by_zero() {
        let rec = MemoryRecord {
            id: 1,
            worker: "w".into(),
            text: String::new(),
        };
        let corpus = vec![rec.clone()];
        assert_eq!(score(&rec, &tokenize("anything"), &corpus), 0.0);
    }

    #[test]
    fn a_read_round_trips_through_json() {
        let m = mem(&[("w1", "a remembered thing")]);
        let r = m.read("remembered", MEMORY_CAP);
        let s = serde_json::to_string(&r).expect("serializes");
        assert_eq!(
            serde_json::from_str::<MemoryRead>(&s).expect("deserializes"),
            r
        );
    }
}
