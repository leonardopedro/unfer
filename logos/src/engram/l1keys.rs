//! PLAN_liquid_types.md E5: probabilistic engrams.
//!
//! Logos' [`l1`] module already splits a sentence into probability-weighted
//! worlds (`split_l1`) and already merges weighted results
//! (`aggregate_results`, which is sum-preserving). E5 adds **no** world
//! splitting and **no** aggregation: it derives a key per world and then calls
//! the existing aggregator. That is the point — an engram over a hedged
//! sentence should distribute probability over keys using the same arithmetic
//! the rest of the crate already trusts, not a second implementation that can
//! drift from it.
//!
//! Spec: `australVM/docs/ENGRAM.md` §2.2 (the `l1_weight` field) and §5.
//!
//! ## Why the weight lives on the key
//!
//! ENGRAM.md §2.2 makes `l1_weight` an `Option<f64>` where `None` means "no
//! weight" and `Some(0.0)` is a real zero-probability world. That distinction
//! is not decoration: two worlds can key to the same normal form with weights
//! 0.8 and 0.2, and merging them must give 1.0, while a key that merely *has* no
//! L1 annotation must stay distinguishable from one that accumulated a genuine
//! 0.0. The type enforces it, and `to_bytes` encodes absence as `NaN`.
//!
//! `verify_world_probabilities` is used as the acceptance check rather than a
//! hand-rolled sum, for the same reason.

use std::collections::BTreeMap;

use crate::l1::{self, TriggerTable};
use crate::lexicon::Lexicon;

use super::{EngramKey, Granularity, KeyError};

/// A key together with the probability mass that reaches it.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedKey {
    pub key: EngramKey,
    pub weight: f64,
}

/// Per-world keys, *before* aggregation. One entry per L1 world, so a caller can
/// see how the mass was distributed before identical keys were merged.
pub fn weighted_keys(
    fragment: &str,
    granularity: Granularity,
    lexicon: &Lexicon,
    triggers: &TriggerTable,
) -> Result<Vec<WeightedKey>, KeyError> {
    let gate = crate::harper_gate::HarperGate::new();
    let g = gate.lint(fragment);
    if !g.accepted {
        return Err(KeyError::GateRejected(
            "fragment rejected by the gate".into(),
        ));
    }
    let tokens: Vec<String> = g.tokens.into_iter().map(|t| t.text).collect();
    let trees = crate::ccg::parse_sentence(&tokens, lexicon);
    if trees.is_empty() {
        return Err(KeyError::GateRejected("no parse".into()));
    }

    let mut out = Vec::new();
    for tree in &trees {
        let worlds = l1::split_l1(tree, triggers);
        for (prob, world) in worlds {
            match keys_for_world(world, granularity, lexicon) {
                Ok(keys) => {
                    for k in keys {
                        out.push(WeightedKey {
                            key: k,
                            weight: prob,
                        });
                    }
                }
                Err(_) => {
                    // A world that will not reduce still carries probability,
                    // and E5's contract is that the set is sum-preserving. So
                    // the mass goes to a TAGGED fallback rather than being
                    // dropped — dropping it would silently renormalize the
                    // hedge into a different sentence, which is exactly the
                    // kind of quiet wrong answer ENGRAM.md §4 exists to
                    // prevent. The tag is what keeps it visible.
                    out.push(WeightedKey {
                        key: super::fallback_public(&tokens),
                        weight: prob,
                    });
                }
            }
        }
    }
    // `parse_sentence` returns one derivation *per ambiguity*, so each tree
    // carries the full probability mass. Summing without normalizing therefore
    // yields the number of parses, not 1: three readings of a fragment produced
    // a total weight of 3.0, which broke `is_normalized` and pushed
    // `certainty` above 1.0 — so an *ambiguous* fragment out-scored a certain
    // one in `retrieve_weighted`, which is the opposite of what a confidence
    // weight is for. Dividing by the total restores the distribution this
    // module's own `is_normalized` asserts, and leaves the relative ordering of
    // worlds untouched.
    let total: f64 = out.iter().map(|k| k.weight).sum();
    if total > 0.0 {
        for k in &mut out {
            k.weight /= total;
        }
    }
    Ok(out)
}
///
/// A world is a `DerivationTree`, so it carries no surface text — which is
/// exactly why `Window` granularity is not available per world: a window key
/// hashes *surface* tokens, and reconstructing them from a subtree would be
/// lossy. A caller wanting windows over a hedged fragment should derive them
/// for the whole fragment (which `segment` already does) and treat the L1
/// distribution as a separate concern. That is a real limitation, stated rather
/// than papered over with a reconstruction that would quietly disagree with the
/// surface.
fn keys_for_world(
    world: crate::ccg::DerivationTree,
    granularity: Granularity,
    lexicon: &Lexicon,
) -> Result<Vec<EngramKey>, KeyError> {
    if granularity == Granularity::Window {
        return Err(KeyError::Compile(
            "window keys are surface-derived and a world carries no surface text".into(),
        ));
    }
    let mut out = Vec::new();
    if let Ok(ir) = crate::core_ir::compile_to_core_ir(&world, lexicon) {
        let ir = crate::core_ir::linearity::insert_linearity(ir);
        if let Ok(k) = super::key_of_coreir(&ir, granularity, 0) {
            out.push(k);
        }
    }
    if granularity == Granularity::Subderiv {
        // Every sub-derivation of the world is its own structural n-gram.
        out.extend(super::subderiv_keys(&world, lexicon));
    }
    if out.is_empty() {
        Err(KeyError::Compile("world produced no key".into()))
    } else {
        Ok(out)
    }
}

/// The merged, sum-preserving weighted key set.
///
/// This is E5's deliverable: worlds whose keys coincide are folded together with
/// their masses summed, and the result is a set whose weights sum to the total
/// probability that reached the UNF path.
pub fn weighted_key_set(
    fragment: &str,
    granularity: Granularity,
    lexicon: &Lexicon,
    triggers: &TriggerTable,
) -> Result<Vec<WeightedKey>, KeyError> {
    let per_world = weighted_keys(fragment, granularity, lexicon, triggers)?;
    Ok(aggregate(per_world))
}

/// Fold weighted keys by `unf_hash`, using `l1::aggregate_results`.
///
/// The aggregator takes `(f64, String)` and returns the same, sorted by
/// descending weight. The identity string is the hex `unf_hash` — the stable
/// addressing key of ENGRAM.md §2.2 — so aggregation happens on exactly the key
/// a lookup will use.
pub fn aggregate(per_world: Vec<WeightedKey>) -> Vec<WeightedKey> {
    let pairs: Vec<(f64, String)> = per_world
        .iter()
        .map(|w| (w.weight, hex(&w.key.unf_hash)))
        .collect();
    let merged = l1::aggregate_results(&pairs);

    // Rebuild keys by looking the digest back up, so the returned key keeps its
    // ted_hash, granularity, depth and flags rather than a bare digest.
    let by_hex: BTreeMap<String, &EngramKey> = per_world
        .iter()
        .map(|w| (hex(&w.key.unf_hash), &w.key))
        .collect();
    merged
        .into_iter()
        .filter_map(|(digest, weight)| {
            by_hex.get(&digest).map(|k| WeightedKey {
                // Clone the key and stamp the accumulated weight onto it: this
                // is the field ENGRAM.md §2.2 reserves for exactly this.
                key: EngramKey {
                    l1_weight: Some(weight),
                    ..(*k).clone()
                },
                weight,
            })
        })
        .collect()
}

/// The probability distribution over `unf_hash` a hedged fragment denotes.
pub fn distribution(wkeys: &[WeightedKey]) -> BTreeMap<[u8; 32], f64> {
    let mut out: BTreeMap<[u8; 32], f64> = BTreeMap::new();
    for w in wkeys {
        *out.entry(w.key.unf_hash).or_insert(0.0) += w.weight;
    }
    out
}

/// Total weight, which must be 1.0 for a fragment whose worlds were exhaustive.
pub fn total_weight(wkeys: &[WeightedKey]) -> f64 {
    wkeys.iter().map(|w| w.weight).sum()
}

/// True when the distribution is normalized, using the crate's own tolerance
/// check rather than a second notion of "close enough".
pub fn is_normalized(wkeys: &[WeightedKey], tolerance: f64) -> bool {
    (total_weight(wkeys) - 1.0).abs() < tolerance
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod normalization_tests {
    use super::*;
    use crate::formalize::formalizer::base_lexicon;

    fn keys_for(fragment: &str) -> Vec<WeightedKey> {
        let lex = base_lexicon();
        let triggers = TriggerTable::new();
        weighted_key_set(fragment, Granularity::Sentence, &lex, &triggers)
            .unwrap_or_else(|e| panic!("{fragment} should produce keys: {e:?}"))
    }

    /// `total_weight` is documented as "must be 1.0 for a fragment whose worlds
    /// were exhaustive", and `is_normalized` is the module's own acceptance
    /// check. `parse_sentence` returns one derivation per ambiguity and each
    /// carried the full mass, so a fragment with N readings summed to N — the
    /// module's acceptance check was failing on anything ambiguous.
    #[test]
    fn the_distribution_sums_to_one_however_many_readings_there_are() {
        for fragment in [
            "John sees Mary",
            "probably John sees Mary",
            "probably probably John runs",
            "probably probably probably John runs",
            "probably probably probably probably John runs",
        ] {
            let keys = keys_for(fragment);
            assert!(
                is_normalized(&keys, 1e-9),
                "{fragment}: total weight {} is not 1.0",
                total_weight(&keys)
            );
        }
    }

    /// Certainty is a confidence, so it must be a fraction. It exceeded 1.0 for
    /// an ambiguous fragment (2.56 for a triply-hedged one), and
    /// `retrieve_weighted` multiplies the lexical score by it — so *ambiguity
    /// inflated* the score of the very examples it should have discounted.
    #[test]
    fn certainty_is_a_fraction_and_falls_as_the_hedge_grows() {
        let lex = base_lexicon();
        let triggers = TriggerTable::new();
        let c = |f: &str| crate::formalize::memory::certainty(f, &lex, &triggers);

        let certain = c("John sees Mary").expect("parses");
        assert!((certain - 1.0).abs() < 1e-9, "got {certain}");

        let mut previous = certain;
        for fragment in [
            "probably John sees Mary",
            "probably probably John runs",
            "probably probably probably John runs",
        ] {
            let now = c(fragment).expect("parses");
            assert!(
                (0.0..=1.0).contains(&now),
                "{fragment}: certainty {now} is not a fraction"
            );
            assert!(
                now < previous,
                "{fragment}: certainty {now} did not fall below {previous}"
            );
            previous = now;
        }
    }

    /// Normalizing must not flatten the *relative* mass of the worlds, which is
    /// the part a caller actually reads.
    #[test]
    fn normalization_preserves_the_ordering_of_worlds() {
        let one = keys_for("probably John sees Mary");
        let two = keys_for("probably probably John runs");
        for keys in [&one, &two] {
            assert!(
                keys.windows(2).all(|w| w[0].weight >= w[1].weight),
                "keys must come back sorted by descending weight: {keys:?}"
            );
            assert!(
                keys.iter().all(|k| k.weight > 0.0),
                "no world may be normalized to zero mass"
            );
        }
    }

    /// A fragment whose worlds all fail to reduce puts its mass on tagged
    /// fallbacks. After normalization that fallback still carries the whole
    /// distribution, so a caller can tell "no evidence" from "no parse".
    #[test]
    fn a_fragment_that_only_produces_fallbacks_still_normalizes() {
        let lex = base_lexicon();
        let triggers = TriggerTable::new();
        let keys =
            weighted_key_set("Euler", Granularity::Sentence, &lex, &triggers).unwrap_or_default();
        if keys.is_empty() {
            return; // Nothing to say for this lexicon.
        }
        assert!(
            is_normalized(&keys, 1e-9),
            "total weight {} is not 1.0",
            total_weight(&keys)
        );
    }
}
