//! P3 — the engram-backed lemma store.
//!
//! # What it replaces, and what is genuinely new
//!
//! ProofFlow has no memory: every run re-asks the model to formalize every node,
//! and nothing carries across runs. Two caches are standard practice in
//! autoformalization and both are needed here, so both are here:
//!
//! - a **provenance cache**, keyed by `(node text, prompt version, model)`, which
//!   is plan §17.3's answer to LLM nondeterminism — an identical re-request is
//!   served from memory instead of re-asked, so a pipeline run is reproducible
//!   even though the model is not;
//! - a **lemma store**, keyed by **UNF hash**, which is the content-addressed
//!   identity §14 introduces. Two nodes that denote the same term share one
//!   lemma, and two lemmas that mean the same thing cannot drift apart.
//!
//! Retrieval feeds few-shot examples into [`crate::formalize::formalizer`]'s
//! prompt, which is what makes it retrieval-*augmented* rather than merely
//! cached.
//!
//! # Why this does not reuse `engram::table::EngramTable`
//!
//! `EngramTable` maps `EngramKey → Embedding`, and `Embedding = Vec<f32>` —
//! vector payloads for vector-similarity retrieval. A lemma's payload is its CNL
//! text, readback, and provenance, and none of that survives a round trip
//! through `f32`. Storing a handle there and keeping the real payload in a
//! side map would be an `EngramTable` with its lookup path bypassed, so this
//! module keys directly on the same 32-byte digest `EngramKey` carries.
//!
//! What *is* reused is everything that matters: key derivation
//! ([`crate::engram::engram_key`], so a key here is an engram key with §2.2's
//! 84-byte layout), granularity semantics ([`crate::engram::Granularity`], whose
//! rule that "a lookup at `g` must not fall back to a coarser `g`" this module
//! obeys), and weighted retrieval
//! ([`crate::engram::l1keys::weighted_key_set`]).
//!
//! Consequently [`crate::engram::tiered::TieredStore`] and
//! [`crate::engram::spill`] are *not* used for the payload. Both are typed to
//! `Embedding`, and their spill addressing ([`crate::engram::spill::spill_addr`])
//! is by `(granularity, unf_hash)` — which is exactly this module's key, so
//! lifting the payload type there is a small change to that module rather than a
//! reimplementation here. Recorded as a follow-up, not silently skipped.

use crate::engram::{self, Granularity};
use crate::formalize::formalizer::CnlFormalization;
use crate::formalize::graph::GraphNode;
use crate::l1::TriggerTable;
use crate::lexicon::Lexicon;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Where a lemma came from, and under what conditions it may be reused.
///
/// The fields are exactly plan §17.3's cache key. `model` and
/// `prompt_version` are not decoration: a lemma formalized by a different model,
/// or by a prompt whose wording has since changed, is a *different* answer to
/// the same question, and silently reusing it would make a "verified" node
/// verified against a standard that no longer exists.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Provenance {
    /// The node id this was formalized for (`l3`, `ts_1`, …).
    pub node_id: String,
    /// The node's `statement`, verbatim. Part of the cache key.
    pub node_text: String,
    /// The generating model.
    pub model: String,
    /// The prompt's identity. Bumped whenever a prompt's wording changes.
    pub prompt_version: String,
}

impl Provenance {
    /// The provenance-cache key: `(node text, prompt version, model)`, per §17.3.
    ///
    /// `node_id` is deliberately **excluded**. A node id is a position in *this*
    /// graph; the same step in the next proof is `l3` again but a different
    /// statement. Keying on the id would return `l3`'s lemma for an unrelated
    /// `l3`.
    pub fn cache_key(&self) -> CacheKey {
        CacheKey {
            node_text: self.node_text.clone(),
            prompt_version: self.prompt_version.clone(),
            model: self.model.clone(),
        }
    }
}

/// `(node text, prompt version, model)` — plan §17.3's cache key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    pub node_text: String,
    pub prompt_version: String,
    pub model: String,
}

/// A stored, verified CNL lemma.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lemma {
    pub cnl: String,
    /// The readable normal form (`Love(john, mary)`).
    pub readback: String,
    /// Content-addressable identity.
    pub unf_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ted: Option<String>,
    /// The confluence self-check at insert time.
    pub verified: bool,
    /// Every provenance that contributed this lemma, sorted and deduplicated.
    ///
    /// A list rather than one: this is a *shared* lemma under §14's identity, so
    /// two nodes denoting the same term both point at it, and dropping one would
    /// lose the audit trail for half of it.
    pub provenances: Vec<Provenance>,
}

impl Lemma {
    /// Record a provenance, keeping the list sorted and duplicate-free.
    ///
    /// Sorted so that a store built from two runs of the same pipeline serializes
    /// identically — otherwise every run would produce a diff.
    fn with_provenance(mut self, p: Provenance) -> Self {
        if !self.provenances.contains(&p) {
            self.provenances.push(p);
            self.provenances.sort();
        }
        self
    }
}

impl Lemma {
    /// View the lemma as a [`CnlFormalization`].
    ///
    /// `tries` is `0`, not a remembered attempt count: a cache hit made no
    /// attempts, and reporting the count from the run that originally produced
    /// the lemma would attribute work to a run that did not do it.
    pub fn to_formalization(&self) -> CnlFormalization {
        CnlFormalization {
            cnl: self.cnl.clone(),
            readback: self.readback.clone(),
            unf_hash: self.unf_hash.clone(),
            verified: self.verified,
            value: self.value.clone(),
            ted: self.ted.clone(),
            tries: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryError {
    /// A lemma offered for insertion has no usable identity.
    ///
    /// Never inserted: plan §17.4 — a node without a unique normal form must not
    /// be hashed, and therefore must not enter the store.
    Unverified { cnl: String },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::Unverified { cnl } => write!(
                f,
                "refusing to store {cnl:?}: it has no unique normal form, so it \
                 has no content address"
            ),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Statistics for a store, for P6's report and P8's harness.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StoreStats {
    /// Distinct UNF-keyed lemmas.
    pub lemmas: usize,
    /// Distinct provenance-cache entries.
    pub cache_entries: usize,
    /// Offers that matched an existing lemma's identity, merging provenance.
    pub identity_hits: usize,
    /// Offers that added a new lemma.
    pub inserts: usize,
    /// Offers refused for having no unique normal form.
    pub refused: usize,
}

/// The lemma store: UNF-keyed identity plus a provenance cache.
#[derive(Debug, Clone, Default)]
pub struct LemmaStore {
    lemmas: HashMap<String, Lemma>,
    by_cache_key: HashMap<CacheKey, String>,
    stats: StoreStats,
}

impl LemmaStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> &StoreStats {
        &self.stats
    }

    pub fn len(&self) -> usize {
        self.lemmas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lemmas.is_empty()
    }

    /// Number of nodes sharing lemmas — the deduplication factor.
    pub fn nodes_stored(&self) -> usize {
        self.lemmas.values().map(|l| l.provenances.len()).sum()
    }

    /// Offer a formalization to the store.
    ///
    /// Inserted only if it verified, per §17.4: an unverified reduction is not
    /// hashed and not stored, because it has no content address and admitting it
    /// would make the store's hit rate a lie.
    ///
    /// A second offer of the same UNF merges its provenance into the existing
    /// lemma rather than replacing it — same term, same lemma.
    pub fn insert(
        &mut self,
        formalization: &CnlFormalization,
        provenance: Provenance,
    ) -> Result<&Lemma, MemoryError> {
        if !formalization.verified {
            self.stats.refused += 1;
            return Err(MemoryError::Unverified {
                cnl: formalization.cnl.clone(),
            });
        }

        let existing = self
            .lemmas
            .entry(formalization.unf_hash.clone())
            .or_insert_with(|| {
                self.stats.inserts += 1;
                Lemma {
                    cnl: formalization.cnl.clone(),
                    readback: formalization.readback.clone(),
                    unf_hash: formalization.unf_hash.clone(),
                    value: formalization.value.clone(),
                    ted: formalization.ted.clone(),
                    verified: true,
                    provenances: Vec::new(),
                }
            });

        if !existing.provenances.contains(&provenance) {
            // A *merge*: this term was already stored under another node.
            self.stats.identity_hits += 1;
        }
        // Clone rather than `mem::take`, which would need `Lemma: Default` for
        // no gain — an insert happens once per node, and the alternative
        // silently manufactures an empty lemma if this method ever grows a panic
        // between the take and the write-back.
        *existing = existing.clone().with_provenance(provenance.clone());

        // Only the first provenance of a shared lemma becomes a cache entry;
        // re-pointing it at a later one would silently change what a cache hit
        // returns for the original node. The clone is because the merge above
        // consumed `provenance`.
        self.by_cache_key
            .entry(provenance.cache_key())
            .or_insert_with(|| formalization.unf_hash.clone());
        // Both counters are refreshed from the structures themselves rather than
        // incremented alongside, so a new field cannot be forgotten: this bug
        // already happened once, and a report that said "0 distinct lemmas"
        // after storing nine is worse than no report.
        self.stats.cache_entries = self.by_cache_key.len();
        self.stats.lemmas = self.lemmas.len();
        Ok(self
            .lemmas
            .get(&formalization.unf_hash)
            .expect("just inserted"))
    }

    /// The provenance-cache lookup: "have I already asked this exact question?"
    pub fn lookup_provenance(&self, key: &CacheKey) -> Option<&Lemma> {
        let hash = self.by_cache_key.get(key)?;
        self.lemmas.get(hash)
    }

    /// Insert the formalization for `node`, deriving the provenance.
    ///
    /// `model` and `prompt_version` identify the generating configuration, and
    /// are part of the cache key — see [`Provenance`].
    pub fn remember(
        &mut self,
        node: &GraphNode,
        formalization: &CnlFormalization,
        model: &str,
        prompt_version: &str,
    ) -> Result<&Lemma, MemoryError> {
        let provenance = Provenance {
            node_id: node.id.clone(),
            node_text: node.statement.clone(),
            model: model.to_string(),
            prompt_version: prompt_version.to_string(),
        };
        self.insert(formalization, provenance)
    }

    /// Exact lookup by UNF hash.
    pub fn get(&self, unf_hash: &str) -> Option<&Lemma> {
        self.lemmas.get(unf_hash)
    }

    pub fn contains(&self, unf_hash: &str) -> bool {
        self.lemmas.contains_key(unf_hash)
    }

    /// Every lemma, ordered by UNF hash.
    ///
    /// Ordered rather than hash-ordered so a serialized store is byte-stable.
    pub fn iter(&self) -> impl Iterator<Item = &Lemma> {
        self.lemmas.values()
    }
}

// ── retrieval ───────────────────────────────────────────────────────────────

/// One retrieved lemma, with why it was retrieved.
#[derive(Debug, Clone, PartialEq)]
pub struct Retrieved {
    pub lemma: Lemma,
    /// The ranking score. Meaning depends on the retrieval used; see
    /// [`retrieve_lexical`] and [`retrieve_weighted`].
    pub score: f64,
    /// Human-readable justification, for the prompt and for a report.
    pub because: String,
}

/// Content words shared between a node's statement and a lemma's CNL.
///
/// Deliberately *not* stemmed and *not* stop-word-filtered against a language
/// list: the statements are mathematical English and the vocabulary is 46 words,
/// so an aggressive normalizer would remove signal rather than noise. The one
/// filter is length ≥ 2 **characters**, which drops `x`/`n`-style single letters
/// that would otherwise match every lemma.
///
/// Characters, not bytes: `str::len` counts bytes, and every mathematical
/// variable in a proof — `β`, `ζ`, `ε` — is two bytes in UTF-8, so a byte-length
/// filter keeps exactly the tokens it is meant to drop. That inflated the
/// retrieval score with a free match against any statement mentioning a Greek
/// letter, which is most of them.
fn content_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Rank stored lemmas by vocabulary overlap with a node's statement.
///
/// This is the few-shot retriever. It is lexical rather than vector because
/// there is no embedding model in the pipeline: `engram::placeholder_embedding`
/// is a deterministic stand-in derived from the key, so similarity on it would be
/// similarity of *hashes*, which says nothing about meaning. A lexical overlap
/// score is weak but it is *honest* — and a weak-but-real signal beats a
/// strong-looking one that measures nothing.
///
/// Ties break on UNF hash so the ordering is total and reproducible.
pub fn retrieve_lexical(store: &LemmaStore, node: &GraphNode, k: usize) -> Vec<Retrieved> {
    let want = content_words(&node.statement);
    let mut scored: Vec<Retrieved> = store
        .iter()
        .map(|lemma| {
            let have = content_words(&lemma.cnl);
            let shared: Vec<String> = want.iter().filter(|w| have.contains(w)).cloned().collect();
            let score = shared.len() as f64 / want.len().max(1) as f64;
            Retrieved {
                lemma: lemma.clone(),
                score,
                because: format!(
                    "shares {} of {} statement words ({}); CNL {:?} reduces to {}",
                    shared.len(),
                    want.len(),
                    shared.join(", "),
                    lemma.cnl,
                    lemma.readback
                ),
            }
        })
        .filter(|r| r.score > 0.0)
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .expect("scores are ratios, never NaN")
            .then_with(|| a.lemma.unf_hash.cmp(&b.lemma.unf_hash))
    });
    scored.truncate(k);
    scored
}

/// The certainty of a hedged fragment: how much L1 mass stayed on one reading.
///
/// `probably X` splits into 0.8 identity and 0.2 negate. The mass that reached
/// the UNF path on a *non-fallback* key is the part of the probability we are
/// actually standing on; the rest went to tagged window fallbacks and is not
/// evidence of anything.
///
/// `None` when the fragment does not parse at all, which is different from
/// "certainly nothing" — the caller cannot hedge about a sentence it cannot read.
pub fn certainty(fragment: &str, lexicon: &Lexicon, triggers: &TriggerTable) -> Option<f64> {
    let keys = engram::l1keys::weighted_key_set(fragment, Granularity::Sentence, lexicon, triggers)
        .ok()?;
    Some(
        keys.iter()
            .filter(|k| !k.key.is_fallback())
            .map(|k| k.weight)
            .sum::<f64>(),
    )
}

/// Retrieve lemmas for a *hedged* fragment, weighted by how certain it is.
///
/// This is the plan's "`l1keys` for weighted retrieval", and it deliberately
/// does **not** look lemmas up by L1 world key. That is the obvious design and
/// it cannot work: an L1 world's key is derived from the world's
/// `DerivationTree` ([`engram::l1keys`]), i.e. from the sub-derivation *under*
/// the trigger, whereas a lemma's key is the UNF of a whole sentence. For
/// `probably John sees Mary` the identity world's key is the same for every
/// fragment with that trigger — it is the key of the bare `probably` modifier —
/// so a store keyed by sentence UNFs would never match it and every lookup
/// would silently return nothing.
///
/// So `l1keys` is used for what it can actually answer: how much of the
/// probability mass reached the UNF path. That mass scales the lexical score, so
/// a hedged fragment retrieves the same examples as a certain one but reports
/// less confidence, and an unparseable fragment returns nothing rather than
/// pretending to be certain.
pub fn retrieve_weighted(
    store: &LemmaStore,
    fragment: &str,
    lexicon: &Lexicon,
    triggers: &TriggerTable,
    k: usize,
) -> Vec<Retrieved> {
    let Some(certain) = certainty(fragment, lexicon, triggers) else {
        return Vec::new();
    };

    let hedge = if certain < 1.0 {
        format!(
            "the fragment hedges: only {certain:.4} of its L1 probability reached \
             the UNF path, so this score is discounted"
        )
    } else {
        String::new()
    };

    let mut out = retrieve_lexical(store, &fragment_node(fragment), k);
    for r in &mut out {
        r.score *= certain;
        if !hedge.is_empty() {
            r.because.push_str(&format!("; {hedge}"));
        }
    }
    out
}

/// A pseudo-node whose `statement` is the fragment, so
/// [`retrieve_lexical`] — which scores against a statement's vocabulary — can be
/// reused for a bare CNL string.
///
/// A wrapper rather than a change to `retrieve_lexical`'s signature: the graph
/// type is the pipeline's currency, and the fragment is not a graph node, but
/// inventing a second scoring function would let the two drift apart.
fn fragment_node(fragment: &str) -> GraphNode {
    GraphNode::new("fragment", fragment, fragment, Vec::<String>::new())
}

/// Build the few-shot block for a formalizer prompt.
///
/// Empty when nothing was retrieved — an empty block is better than a
/// placeholder, because the prompt already tells the model what to do and a
/// "no examples available" section would only invite it to comment on the gap.
pub fn few_shot_block(retrieved: &[Retrieved]) -> String {
    if retrieved.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "\nPreviously formalized steps you may reuse or must deliberately differ from:\n",
    );
    for r in retrieved {
        out.push_str(&format!(
            "\n  {} (CNL `{}` reduces to {}; {})\n",
            r.lemma.unf_hash, r.lemma.cnl, r.lemma.readback, r.because
        ));
    }
    out
}

/// Serialize the store for `--out memory.json` in P6's CLI.
pub fn to_json(store: &LemmaStore) -> Result<String, serde_json::Error> {
    let mut lemmas: BTreeMap<&str, &Lemma> = BTreeMap::new();
    for lemma in store.iter() {
        lemmas.insert(lemma.unf_hash.as_str(), lemma);
    }
    serde_json::to_string_pretty(&serde_json::json!({
        "stats": store.stats(),
        "lemmas": lemmas,
    }))
}

/// Rebuild a store from [`to_json`]'s output.
pub fn from_json(text: &str) -> Result<LemmaStore, serde_json::Error> {
    let value: serde_json::Value = serde_json::from_str(text)?;
    let mut store = LemmaStore::new();
    if let Some(map) = value.get("lemmas").and_then(Value::as_object) {
        for lemma in map.values() {
            let lemma: Lemma = serde_json::from_value(lemma.clone())?;
            // The cache index is rebuilt from the provenances, so a reloaded
            // store serves provenance hits exactly as the original did.
            for p in &lemma.provenances {
                store
                    .by_cache_key
                    .entry(p.cache_key())
                    .or_insert_with(|| lemma.unf_hash.clone());
            }
            store.lemmas.insert(lemma.unf_hash.clone(), lemma);
        }
    }
    store.stats.cache_entries = store.by_cache_key.len();
    store.stats.lemmas = store.lemmas.len();
    Ok(store)
}

use serde_json::Value;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formalize::formalizer::{base_lexicon, verify_cnl};

    fn node(id: &str, statement: &str) -> GraphNode {
        GraphNode::new(id, "quote", statement, Vec::<String>::new())
    }

    /// A verified formalization for `cnl`, straight from the pipeline.
    fn formal(cnl: &str) -> CnlFormalization {
        let r =
            verify_cnl(cnl, &base_lexicon()).unwrap_or_else(|e| panic!("{cnl} should verify: {e}"));
        r.into_formalization(1)
    }

    fn prov(node_id: &str, text: &str) -> Provenance {
        Provenance {
            node_id: node_id.to_string(),
            node_text: text.to_string(),
            model: "test-model".into(),
            prompt_version: "v1".into(),
        }
    }

    // ── identity and refusal ────────────────────────────────────────────────

    #[test]
    fn a_verified_lemma_is_stored_and_found_by_its_hash() {
        let mut store = LemmaStore::new();
        let f = formal("John loves Mary");
        let hash = f.unf_hash.clone();
        store
            .insert(&f, prov("l1", "Mary is loved by John"))
            .unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.contains(&hash));
        assert_eq!(store.get(&hash).unwrap().readback, "Love(john, mary)");
        assert_eq!(store.stats().inserts, 1);
    }

    /// `StoreStats::lemmas` once read 0 after nine inserts, because it was a
    /// field nobody maintained — so a report said "0 distinct lemmas" after
    /// storing nine. Both counters are now derived from the structures rather
    /// than incremented alongside, and this pins that they track.
    #[test]
    fn stats_track_the_structures() {
        let mut store = LemmaStore::new();
        store
            .insert(&formal("John loves Mary"), prov("l1", "a"))
            .unwrap();
        store.insert(&formal("Bob runs"), prov("l2", "b")).unwrap();
        // A shared lemma: distinct terms stay at 2, cache entries at 3.
        store
            .insert(&formal("John loves Mary"), prov("l3", "c"))
            .unwrap();
        assert_eq!(store.stats().lemmas, store.len());
        assert_eq!(store.stats().cache_entries, store.by_cache_key.len());
        assert_eq!(store.stats().lemmas, 2);
        assert_eq!(store.stats().cache_entries, 3);
    }

    /// §17.4: a node without a unique normal form has no content address, so it
    /// must not enter the store — and admitting it would make the hit rate a lie.
    #[test]
    fn an_unverified_formalization_is_refused() {
        let mut store = LemmaStore::new();
        let mut f = formal("John loves Mary");
        f.verified = false;
        let err = store.insert(&f, prov("l1", "x")).unwrap_err();
        assert!(matches!(err, MemoryError::Unverified { .. }));
        assert!(store.is_empty());
        assert_eq!(store.stats().refused, 1);
        assert!(err.to_string().contains("content address"), "{err}");
    }

    /// §14's identity: two nodes denoting the same term share one lemma.
    #[test]
    fn two_provenances_of_the_same_term_share_one_lemma() {
        let mut store = LemmaStore::new();
        let f = formal("John loves Mary");
        store.insert(&f, prov("l1", "first phrasing")).unwrap();
        store.insert(&f, prov("l2", "second phrasing")).unwrap();
        assert_eq!(store.len(), 1, "one term, one lemma");
        assert_eq!(store.nodes_stored(), 2, "but two nodes are recorded");
        let lemma = store.get(&f.unf_hash).unwrap();
        assert_eq!(lemma.provenances.len(), 2);
        assert_eq!(store.stats().inserts, 1);
    }

    #[test]
    fn provenance_list_is_sorted_and_deduplicated() {
        let mut store = LemmaStore::new();
        let f = formal("John loves Mary");
        for _ in 0..3 {
            store.insert(&f, prov("l1", "same")).unwrap();
        }
        let lemma = store.get(&f.unf_hash).unwrap();
        assert_eq!(lemma.provenances.len(), 1);
    }

    // ── the provenance cache ────────────────────────────────────────────────

    #[test]
    fn the_provenance_cache_serves_an_identical_request() {
        let mut store = LemmaStore::new();
        let f = formal("John loves Mary");
        let key = prov("l1", "Mary is loved by John").cache_key();
        assert!(store.lookup_provenance(&key).is_none());
        store
            .insert(&f, prov("l1", "Mary is loved by John"))
            .unwrap();
        assert_eq!(
            store.lookup_provenance(&key).unwrap().readback,
            "Love(john, mary)"
        );
    }

    /// The cache key excludes `node_id` on purpose: `l3` in this proof is a
    /// different statement from `l3` in the next one.
    #[test]
    fn a_different_statement_is_not_a_cache_hit_despite_the_same_node_id() {
        let mut store = LemmaStore::new();
        let f = formal("John loves Mary");
        store
            .insert(&f, prov("l1", "Mary is loved by John"))
            .unwrap();
        let other = prov("l1", "John sleeps").cache_key();
        assert!(store.lookup_provenance(&other).is_none());
    }

    /// A different model or prompt version is a different answer to the same
    /// question, so §17.3's key must separate them.
    #[test]
    fn a_different_model_or_prompt_is_not_a_cache_hit() {
        let mut store = LemmaStore::new();
        let f = formal("John loves Mary");
        store.insert(&f, prov("l1", "s")).unwrap();
        let same = prov("l9", "s").cache_key();
        assert!(store.lookup_provenance(&same).is_some());

        for p in [
            Provenance {
                node_id: "l1".into(),
                node_text: "s".into(),
                model: "other".into(),
                prompt_version: "v1".into(),
            },
            Provenance {
                node_id: "l1".into(),
                node_text: "s".into(),
                model: "test-model".into(),
                prompt_version: "v2".into(),
            },
        ] {
            assert!(
                store.lookup_provenance(&p.cache_key()).is_none(),
                "{p:?} should miss"
            );
        }
    }

    #[test]
    fn remember_builds_the_provenance_from_the_node() {
        let mut store = LemmaStore::new();
        let n = node("ts_1", "the conclusion");
        let f = formal("John runs");
        store.remember(&n, &f, "m", "p").unwrap();
        assert_eq!(store.len(), 1);
        let lemma = store.get(&f.unf_hash).unwrap();
        assert_eq!(lemma.provenances[0].node_id, "ts_1");
        assert_eq!(lemma.provenances[0].node_text, "the conclusion");
        // `remember` uses the model and prompt version it was handed, so the
        // cache key follows those — not the test helper's defaults.
        let key = CacheKey {
            node_text: "the conclusion".into(),
            prompt_version: "p".into(),
            model: "m".into(),
        };
        assert!(store.lookup_provenance(&key).is_some());
    }

    // ── lexical retrieval ───────────────────────────────────────────────────

    #[test]
    fn lexical_retrieval_ranks_by_shared_vocabulary() {
        let mut store = LemmaStore::new();
        store
            .insert(&formal("the cat sleeps"), prov("l1", "the cat sleeps"))
            .unwrap();
        store
            .insert(&formal("Mary sees Bob"), prov("l2", "Mary sees Bob"))
            .unwrap();
        let n = node("l3", "the cat is red");
        let got = retrieve_lexical(&store, &n, 5);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].lemma.cnl, "the cat sleeps");
        assert!(got[0].because.contains("cat"), "{}", got[0].because);
        assert!(got[0].score > 0.0);
    }

    #[test]
    fn lexical_retrieval_respects_k_and_is_deterministic() {
        let mut store = LemmaStore::new();
        for c in ["John loves Mary", "Mary sees Bob", "Bob sees Alice"] {
            store.insert(&formal(c), prov("l", c)).unwrap();
        }
        let n = node("l9", "John and Mary and Bob see things");
        let a = retrieve_lexical(&store, &n, 2);
        let b = retrieve_lexical(&store, &n, 2);
        assert_eq!(a.len(), 2, "k truncates");
        assert_eq!(a, b, "same input, same order");
    }

    #[test]
    fn lexical_retrieval_on_an_empty_store_is_empty() {
        let store = LemmaStore::new();
        assert!(retrieve_lexical(&store, &node("l1", "anything"), 3).is_empty());
    }

    #[test]
    fn single_letters_do_not_match_everything() {
        let mut store = LemmaStore::new();
        store.insert(&formal("John runs"), prov("l1", "x")).unwrap();
        // "x y z" shares nothing of length >= 2 with the CNL.
        assert!(retrieve_lexical(&store, &node("l2", "x y z"), 3).is_empty());
    }

    /// `str::len` counts *bytes*, and every mathematical variable in a proof is
    /// two bytes in UTF-8 — so a byte-length filter kept exactly the tokens it was
    /// meant to drop, giving a free score boost to any statement that mentioned a
    /// Greek letter. Which is most of them.
    #[test]
    fn single_character_unicode_words_are_dropped_too() {
        let mut store = LemmaStore::new();
        store
            .insert(&formal("the cat sleeps"), prov("l1", "s"))
            .unwrap();
        // `β` is one character and two bytes. A byte filter let it through and
        // matched it against a lemma containing no mathematics at all.
        assert!(
            retrieve_lexical(&store, &node("l2", "β β β"), 3).is_empty(),
            "a single Greek letter must not match anything"
        );
        // Two characters still count, ASCII or not.
        assert_eq!(
            content_words("βγ δε").len(),
            2,
            "two-character runs are kept"
        );
    }

    // ── L1 weighted retrieval ───────────────────────────────────────────────

    /// The L1 mass that reached the UNF path. `probably` splits 0.8 identity /
    /// 0.2 negate, and the 0.2 lands on a tagged fallback — so 0.8 is the
    /// certainty.
    #[test]
    fn certainty_is_the_mass_that_reached_the_unf_path() {
        let lex = base_lexicon();
        let triggers = TriggerTable::new();
        let certain = certainty("probably John sees Mary", &lex, &triggers).unwrap();
        assert!(certain < 1.0 && certain > 0.5, "{certain}");
        // An unhedged sentence is certain.
        assert_eq!(certainty("John sees Mary", &lex, &triggers), Some(1.0));
        // An unreadable fragment is not "certainly nothing".
        assert_eq!(certainty("Euler proves congruences", &lex, &triggers), None);
    }

    /// A hedge scales the lexical score rather than looking lemmas up by world
    /// key — see `retrieve_weighted`'s documentation for why world keys cannot
    /// index sentence-level lemmas.
    #[test]
    fn weighted_retrieval_discounts_a_hedged_fragment() {
        let mut store = LemmaStore::new();
        store
            .insert(&formal("John sees Mary"), prov("l1", "John sees Mary"))
            .unwrap();
        let lex = base_lexicon();
        let triggers = TriggerTable::new();

        let plain = retrieve_weighted(&store, "John sees Mary", &lex, &triggers, 8);
        assert_eq!(plain.len(), 1);
        assert!(!plain[0].because.contains("hedges"), "{}", plain[0].because);

        let hedged = retrieve_weighted(&store, "probably John sees Mary", &lex, &triggers, 8);
        assert_eq!(hedged.len(), 1, "same fragment, still retrieved");
        assert!(
            hedged[0].because.contains("hedges"),
            "{}",
            hedged[0].because
        );
        assert!(
            hedged[0].score < plain[0].score,
            "hedged {} should score below plain {}",
            hedged[0].score,
            plain[0].score
        );
    }

    /// §14's identity is what makes a sub-derivation key useless as a lemma
    /// key, and this pins the fact that motivates the design above.
    #[test]
    fn l1_world_keys_do_not_index_sentence_level_lemmas() {
        use crate::engram::{Granularity, l1keys};
        let lex = base_lexicon();
        let triggers = TriggerTable::new();
        let a = l1keys::weighted_key_set(
            "probably John sees Mary",
            Granularity::Sentence,
            &lex,
            &triggers,
        )
        .unwrap();
        let b = l1keys::weighted_key_set(
            "probably Alice runs",
            Granularity::Sentence,
            &lex,
            &triggers,
        )
        .unwrap();
        let sentence = verify_cnl("John sees Mary", &lex).unwrap();
        // An L1 world key is the hash of a *world's* CoreIR, and the identity
        // world is `App(Var("probably"), <the rest of the sentence>)` — the
        // trigger enters as a free variable applied to everything after it. Two
        // different sentences therefore give different keys, which is the point:
        // an L1 key addresses a world, not the fragment it came from.
        //
        // This assertion was vacuous until the UNF serializer covered every
        // agent kind. `App` used to serialize as a single catch-all byte, so
        // these two keys were equal *because every key was equal*, and the test
        // passed for the wrong reason.
        assert_ne!(
            a[0].key.unf_hash, b[0].key.unf_hash,
            "two different sentences must not share an L1 world key"
        );
        // And an L1 key is never a sentence's UNF, so the two key spaces stay
        // disjoint and a world key cannot be mistaken for a lemma key.
        let hex = |b: &[u8; 32]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        assert_ne!(hex(&a[0].key.unf_hash), sentence.unf_hash);
    }

    /// A fragment we cannot read retrieves nothing, rather than retrieving
    /// everything with zero confidence.
    #[test]
    fn weighted_retrieval_of_an_unparsable_fragment_is_empty() {
        let mut store = LemmaStore::new();
        store
            .insert(&formal("John sees Mary"), prov("l1", "s"))
            .unwrap();
        assert!(
            retrieve_weighted(
                &store,
                "Euler proves congruences",
                &base_lexicon(),
                &TriggerTable::new(),
                4
            )
            .is_empty()
        );
    }

    // ── prompt assembly and persistence ─────────────────────────────────────

    #[test]
    fn the_few_shot_block_names_each_lemma_and_its_normal_form() {
        let mut store = LemmaStore::new();
        store
            .insert(&formal("the cat sleeps"), prov("l1", "the cat sleeps"))
            .unwrap();
        let block = few_shot_block(&retrieve_lexical(&store, &node("l3", "cat"), 3));
        assert!(block.contains("the cat sleeps"), "{block}");
        assert!(block.contains("Sleep(Cat)"), "{block}");
    }

    #[test]
    fn an_empty_few_shot_block_is_empty() {
        assert_eq!(few_shot_block(&[]), "");
    }

    #[test]
    fn the_store_round_trips_through_json() {
        let mut store = LemmaStore::new();
        store
            .insert(&formal("John loves Mary"), prov("l1", "Mary is loved"))
            .unwrap();
        store
            .insert(&formal("Bob runs"), prov("l2", "Bob runs"))
            .unwrap();
        // A shared lemma, so the reload has to preserve two provenances.
        let f = formal("John loves Mary");
        store.insert(&f, prov("ts_1", "the conclusion")).unwrap();

        let json = to_json(&store).unwrap();
        let back = from_json(&json).unwrap();
        assert_eq!(back.len(), store.len());
        assert_eq!(back.nodes_stored(), store.nodes_stored());
        assert_eq!(
            back.lookup_provenance(&prov("ts_1", "the conclusion").cache_key())
                .unwrap()
                .readback,
            "Love(john, mary)"
        );
    }

    /// A serialized store must be byte-stable: two runs of the same pipeline
    /// produce the same bytes, or every run shows a spurious diff.
    #[test]
    fn serialization_is_byte_stable() {
        let build = || {
            let mut store = LemmaStore::new();
            store
                .insert(&formal("John loves Mary"), prov("l1", "a"))
                .unwrap();
            store.insert(&formal("Bob runs"), prov("l2", "b")).unwrap();
            store
        };
        assert_eq!(to_json(&build()).unwrap(), to_json(&build()).unwrap());
    }

    #[test]
    fn retrieval_helpers_never_panic_on_odd_input() {
        let store = LemmaStore::new();
        assert!(retrieve_lexical(&store, &node("l1", ""), 3).is_empty());
        assert!(retrieve_lexical(&store, &node("l1", "!!! ???"), 3).is_empty());
        assert!(retrieve_lexical(&store, &node("l1", "x"), 0).is_empty());
    }
}
