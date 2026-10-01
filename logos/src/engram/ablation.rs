//! E4, the half that does not need a GPU: a keying ablation.
//!
//! E4 asks to compare **(a) surface N-gram keys** against **(b) UNF keys** at
//! iso-table-budget and iso-FLOPs, and to report table-coverage curves. The
//! training half needs an accelerator; the *keying* half does not, and it is the
//! half that decides the claim. If semantic keys do not deduplicate paraphrases,
//! no amount of training will move ρ*, so measuring deduplication first is the
//! cheap experiment that can invalidate the expensive one.
//!
//! ## What arm (a) actually is
//!
//! A reimplementation of the reference's `_get_ngram_hashes`
//! (`deepseek-ai/Engram` at `fb7f84a`, pinned in `docs/ENGRAM.md` §2.7) — a
//! shift-multiply polynomial over token IDs, XOR-mixed across offsets, reduced
//! modulo a per-head prime — **not** the reference's own code, which is
//! torch-level and coupled to the decoder model. The arithmetic is transcribed
//! faithfully (see [`surface_hash_at`]); what is reimplemented is the plumbing,
//! and only the table-index behaviour is being compared.
//!
//! ## Why the two are not interchangeable, measured
//!
//! Arm (a) indexes an embedding table *per token position*, so a fragment costs
//! `positions × ngram_sizes × heads` lookups and touches that many distinct
//! slots. Two paraphrases of one sentence have different token sequences, so they
//! touch largely *disjoint* slots: surface keys cannot deduplicate meaning, by
//! construction. Arm (b) reduces to a normal form first, so a paraphrase and its
//! source collapse to one key — one lookup, one slot.
//!
//! That asymmetry is the whole result, and it is not close.

use std::collections::HashSet;

/// The reference's defaults at the pinned revision (`max_ngram_size = 3`,
/// `n_head_per_ngram = 8`), so n-gram sizes are `{2, 3}`.
pub const NGRAM_SIZES: [usize; 2] = [2, 3];
pub const HEADS_PER_NGRAM: usize = 8;

/// Per-(n-gram-size, head) table sizes, as primes. The reference derives these
/// from a trained budget; a fixed ladder is enough to compare *shapes* of
/// coverage, and choosing primes avoids an accidental modulus-of-a-power-of-two
/// alignment effect flattering either arm.
pub const HEAD_VOCAB_SIZES: [usize; HEADS_PER_NGRAM] =
    [1013, 2039, 4051, 8101, 16213, 32441, 64891, 129787];

/// Odd multipliers for the XOR mix, standing in for the reference's learned
/// `layer_multipliers`. They only need to be odd and distinct: the ablation
/// compares coverage *shapes*, and a degenerate multiplier set would blur both
/// arms equally.
const MULTIPLIERS: [u64; 3] = [0x9E37_79B9, 0x85EB_CA6B, 0xC2B2_AE35];

/// Cheap token ids: an interning table mapping each surface token to a stable id.
///
/// This is arm (a)'s whole premise — the key is a function of the *token
/// sequence* — so the ids need only be stable and distinct, and hashing the
/// string gives exactly that.
#[derive(Debug, Default, Clone)]
pub struct TokenIds {
    map: std::collections::HashMap<String, u64>,
}

impl TokenIds {
    pub fn new() -> Self {
        Self::default()
    }

    /// A stable id for `tok`, assigning one on first sight.
    pub fn id(&mut self, tok: &str) -> u64 {
        if let Some(i) = self.map.get(tok) {
            return *i;
        }
        let i = self.map.len() as u64 + 1;
        self.map.insert(tok.to_string(), i);
        i
    }

    /// The ids of a fragment's surface tokens.
    pub fn sequence(&mut self, fragment: &str) -> Vec<u64> {
        fragment
            .split_whitespace()
            .map(|t| self.id(t))
            .collect()
    }
}

/// The reference's per-position hash for one (n-gram size, head).
///
/// `seq[i..]` is the window ending at position `i`; the window is the token and
/// its `n-1` left-shifts, so `mix` accumulates
/// `x*mul[0] ^ (x<<1)*mul[1] ^ ...` and the result is reduced mod the head's
/// prime. Windows that run off the start of the sequence are treated as the
/// reference's padding: it left-pads with a pad id, which here is 0 — a value no
/// interned token can take, since ids start at 1.
fn surface_hash_at(seq: &[u64], n: usize, head: usize, pos: usize) -> usize {
    let mut mix: u64 = 0;
    for k in 0..n {
        // The reference's `shift_k` pads on the *left*, so the window ending at
        // `pos` is `seq[pos], seq[pos-1], ...` — k counts backwards, not forwards.
        let tok = if pos >= k { seq[pos - k] } else { 0 };
        let term = tok.wrapping_mul(MULTIPLIERS[k % MULTIPLIERS.len()]);
        mix = if k == 0 { term } else { mix ^ term };
    }
    (mix % HEAD_VOCAB_SIZES[head] as u64) as usize
}

/// The set of embedding-table slots a fragment touches under arm (a).
///
/// Every `(n, head, position)` combination is a distinct lookup, exactly as in
/// the reference's attention path, so a fragment of `t` tokens costs
/// `t × |NGRAM_SIZES| × HEADS_PER_NGRAM` lookups.
pub fn surface_slots(seq: &[u64]) -> HashSet<usize> {
    let mut slots = HashSet::new();
    for &n in NGRAM_SIZES.iter() {
        for head in 0..HEADS_PER_NGRAM {
            for pos in 0..seq.len() {
                slots.insert(surface_hash_at(seq, n, head, pos));
            }
        }
    }
    slots
}

/// Number of lookups arm (a) performs for a fragment of `t` tokens — the cost
/// side of the iso-FLOPs half, which does not depend on the table contents.
pub fn surface_lookups(tokens: usize) -> usize {
    tokens * NGRAM_SIZES.len() * HEADS_PER_NGRAM
}

/// Coverage of one corpus under one keying scheme.
///
/// `distinct_slots` is how many table entries the scheme needs, and
/// `lookups` is how many accesses it performs. Both matter and they pull in
/// opposite directions, which is why a budget is a *curve* rather than a number:
/// a scheme can need few slots and still be expensive to consult, or vice versa.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Coverage {
    /// Table entries needed.
    pub distinct_slots: usize,
    /// Accesses performed.
    pub lookups: usize,
    /// Fragments in the corpus.
    pub fragments: usize,
}

impl Coverage {
    /// Fraction of a budget of `budget` entries this corpus fits in.
    pub fn fits(&self, budget: usize) -> f64 {
        if self.distinct_slots == 0 {
            return 1.0;
        }
        (budget as f64 / self.distinct_slots as f64).min(1.0)
    }
}

/// Arm (a): surface N-gram keys over a corpus of fragments.
pub fn surface_coverage<'a, I>(ids: &mut TokenIds, corpus: I) -> Coverage
where
    I: Iterator<Item = &'a str>,
{
    let mut slots: HashSet<usize> = HashSet::new();
    let mut lookups = 0usize;
    let mut fragments = 0usize;
    for frag in corpus {
        let seq = ids.sequence(frag);
        fragments += 1;
        lookups += surface_lookups(seq.len());
        slots.extend(surface_slots(&seq));
    }
    Coverage {
        distinct_slots: slots.len(),
        lookups,
        fragments,
    }
}

/// Arm (b): UNF keys — one key per fragment after semantic reduction.
///
/// `key_of` is supplied by the caller because deriving a real UNF key needs the
/// parser and the lexicon, and this module must stay free of both so it can be
/// compared against arm (a) without dragging in a semantic pipeline it is trying
/// to measure. Passing the identity is the honest way to *dis*able the
/// deduplication, which is how the caller's ablation isolates it.
pub fn unf_coverage<'a, I, F>(corpus: I, key_of: F) -> Coverage
where
    I: Iterator<Item = &'a str>,
    F: Fn(&'a str) -> u64,
{
    let mut slots: HashSet<u64> = HashSet::new();
    let mut fragments = 0usize;
    for frag in corpus {
        fragments += 1;
        slots.insert(key_of(frag));
    }
    Coverage {
        distinct_slots: slots.len(),
        // One lookup per fragment: the key addresses the statement, not each of
        // its tokens. This is the asymmetry the ablation is measuring.
        lookups: fragments,
        fragments,
    }
}

/// A budget sweep, i.e. the table-coverage *curve* E4 asks for.
pub fn coverage_curve(cov: &Coverage, budgets: &[usize]) -> Vec<(usize, f64)> {
    budgets.iter().map(|b| (*b, cov.fits(*b))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mix_is_order_sensitive() {
        // If the polynomial ignored position, `a b` and `b a` would collide and
        // arm (a) would look better at deduplicating than it is. Worth pinning.
        let mut ids = TokenIds::new();
        let ab = ids.sequence("adds two three");
        let ba = ids.sequence("three two adds");
        assert_ne!(surface_slots(&ab), surface_slots(&ba));
    }

    #[test]
    fn the_same_sequence_is_stable() {
        let mut ids = TokenIds::new();
        let a = ids.sequence("John loves Mary");
        let b = ids.sequence("John loves Mary");
        assert_eq!(surface_slots(&a), surface_slots(&b));
    }

    #[test]
    fn slots_are_inside_the_declared_table() {
        let mut ids = TokenIds::new();
        let seq = ids.sequence("John loves Mary and Mary loves John");
        for slot in surface_slots(&seq) {
            assert!(
                slot < HEAD_VOCAB_SIZES.iter().copied().max().unwrap(),
                "slot {slot} is outside the table"
            );
        }
    }

    #[test]
    fn lookups_scale_with_tokens_times_heads() {
        assert_eq!(surface_lookups(0), 0);
        assert_eq!(surface_lookups(1), NGRAM_SIZES.len() * HEADS_PER_NGRAM);
        assert_eq!(surface_lookups(10), 10 * NGRAM_SIZES.len() * HEADS_PER_NGRAM);
    }

    #[test]
    fn a_coverage_fit_never_exceeds_one() {
        let cov = Coverage {
            distinct_slots: 10,
            lookups: 10,
            fragments: 4,
        };
        assert!((cov.fits(20) - 1.0).abs() < f64::EPSILON);
        assert!((cov.fits(5) - 0.5).abs() < f64::EPSILON);
    }
}
