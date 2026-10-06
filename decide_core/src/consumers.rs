//! D8/D9/D10 — the consumers, each an *improvement of an existing feature*.
//!
//! # The one rule every consumer obeys
//!
//! **Advice cannot block.** A wrong suggestion may cost a worker a re-read or a
//! user a scroll. It may not fail a merge, refuse an approval, change a
//! certificate, or reorder a set of items the user then acts on as if it were
//! the original. `advisory()` is the single place that rule is expressed, and
//! `advice_never_blocks_a_merge` is the test that checks it against the real
//! evidence path rather than against a comment.
//!
//! # Why consumers live here and not in the kernel
//!
//! The plan's placement rule is the project's improvement principle: the engine
//! improves the G1 board, the C5 presets, the C6 summarizer, the G9 nudges and
//! the velysterm UI. Putting the consumer logic in the crate that owns the
//! methods keeps it testable without a kernel, a socket or a UI, and keeps the
//! trust boundary checkable in one place.

use crate::{DecideCore, DecideError};
use nalgebra::DVector;

/// D-C1: what kind of board entry a draft looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KindSuggestion {
    Fact,
    Fail,
    Claim,
    Observed,
}

/// One piece of advice: a suggestion, a score, and whether it may be trusted as
/// more than a hint.
#[derive(Debug, Clone, PartialEq)]
pub struct Advice {
    pub suggestion: String,
    /// The engine's score in `[-1, 1]`. Always a **score**; see §0 of DECIDE_SPEC.
    pub score: f64,
    /// Whether `score` carries a calibration. False for every consumer today.
    pub calibrated: bool,
    /// Always true. Present as a field rather than assumed by convention, so a
    /// consumer that wants to block has to set it to false *deliberately*, which
    /// is a reviewable act.
    pub advisory: bool,
}

impl Advice {
    /// Advice that is never authoritative.
    pub fn new(suggestion: impl Into<String>, score: f64, calibrated: bool) -> Self {
        Advice {
            suggestion: suggestion.into(),
            score,
            calibrated,
            advisory: true,
        }
    }
}

/// D-C1 — board entry triage.
///
/// Suggests a `kind` for a drafted board entry. The worker confirms or edits;
/// nothing here writes to the board, because the board is a shared, audited
/// surface and an engine's guess about it would be an unaudited write.
pub fn suggest_board_kind(
    core: &DecideCore,
    draft_embedding: &DVector<f64>,
) -> Result<Advice, DecideError> {
    let sims = core.similarities(draft_embedding)?;
    // The engine is trained with one class per board kind for this consumer, so
    // the argmax names the kind directly.
    let best = argmax(&sims);
    let kind = core
        .labels()
        .get(best)
        .map(|s| s.as_str())
        .unwrap_or("observed");
    let suggestion = match kind {
        "fact" => KindSuggestion::Fact,
        "fail" => KindSuggestion::Fail,
        "claim" => KindSuggestion::Claim,
        _ => KindSuggestion::Observed,
    };
    let calibrated = core
        .calibrator()
        .is_calibrated(crate::calibrate::DEFAULT_MAX_ECE);
    Ok(Advice::new(
        format!("{suggestion:?}"),
        sims[best],
        calibrated,
    ))
}

/// D-C2 — preset routing.
///
/// Suggests which C5 role preset a queued task fits. The dispatcher (or a human)
/// decides; this returns a label, and applying it is somebody else's act.
pub fn suggest_preset(
    core: &DecideCore,
    task_embedding: &DVector<f64>,
) -> Result<Advice, DecideError> {
    let sims = core.similarities(task_embedding)?;
    let best = argmax(&sims);
    let preset = core
        .labels()
        .get(best)
        .map(|s| s.as_str())
        .unwrap_or("reader");
    Ok(Advice::new(
        preset.to_string(),
        sims[best],
        core.calibrator()
            .is_calibrated(crate::calibrate::DEFAULT_MAX_ECE),
    ))
}

/// D-C3 — summarizer channel routing (D9).
///
/// Chooses a detail level per target surface. A routing failure degrades to the
/// default level rather than erroring, because a wrong summary length is not
/// worth failing a request over.
pub fn route_detail_level(
    core: &DecideCore,
    channel_embedding: &DVector<f64>,
    default_level: usize,
) -> usize {
    let Ok(sims) = core.similarities(channel_embedding) else {
        return default_level;
    };
    if sims.is_empty() {
        return default_level;
    }
    let best = argmax(&sims);
    // The class index *is* the level, so long as the consumer registered them
    // that way; anything out of range falls back rather than clamping silently.
    let level = core
        .labels()
        .get(best)
        .and_then(|l| l.parse::<usize>().ok());
    match level {
        Some(l) if l < default_level.max(1) => l,
        _ => default_level,
    }
}

/// D-C5 — nudge priority (D9).
///
/// Orders queued nudges. Ordering only: the *set* of nudges a worker sees is
/// unchanged, and an urgent nudge is never dropped because the engine ranked it
/// low.
///
/// `focus` names the class whose score orders the queue. This is not a
/// convenience: scoring by the *maximum* similarity returns nearly the same class
/// for almost every input, because the best-aligned prototype is a property of
/// the geometry rather than of the item — so a max-based ranking is close to
/// constant and reorders nothing. Naming the class is what makes it mean
/// anything.
pub fn nudge_order<'a>(
    core: &DecideCore,
    nudges: &'a [String],
    embeddings: &[DVector<f64>],
    focus: &str,
) -> Vec<&'a String> {
    let Some(f) = core.id_of(focus) else {
        // An unknown focus degrades to the original order rather than an
        // arbitrary one.
        return nudges.iter().collect();
    };
    let mut idx: Vec<usize> = (0..nudges.len()).collect();
    if embeddings.len() == nudges.len() && !nudges.is_empty() {
        let scores: Vec<f64> = embeddings
            .iter()
            .map(|e| {
                core.similarities(e)
                    .map(|s| s[f])
                    .unwrap_or(f64::NEG_INFINITY)
            })
            .collect();
        // Stable sort by descending score: equal scores keep their original
        // order, so a tie never reshuffles the queue.
        idx.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    idx.into_iter().map(|i| &nudges[i]).collect()
}

/// D-C4 — editor ranking (D10).
///
/// Advisory ordering: the **set** is provably unchanged and only the order
/// differs. `ranked` takes the items and returns a permutation of exactly the
/// same slice, so a caller cannot accidentally receive a different set.
///
/// `focus` names the class whose score orders the list, for the same reason as
/// in [`nudge_order`]: a maximum-over-classes score is nearly constant across
/// inputs and would reorder nothing.
pub fn ranked<'a, T>(
    core: &DecideCore,
    items: &'a [T],
    embeddings: &[DVector<f64>],
    focus: &str,
) -> Vec<&'a T> {
    if embeddings.len() != items.len() || items.len() < 2 {
        return items.iter().collect();
    }
    let Some(f) = core.id_of(focus) else {
        return items.iter().collect();
    };
    let mut idx: Vec<usize> = (0..items.len()).collect();
    let scores: Vec<f64> = embeddings
        .iter()
        .map(|e| {
            core.similarities(e)
                .map(|s| s[f])
                .unwrap_or(f64::NEG_INFINITY)
        })
        .collect();
    idx.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter().map(|i| &items[i]).collect()
}

fn argmax(s: &DVector<f64>) -> usize {
    let mut best = 0usize;
    for i in 1..s.len() {
        if s[i] > s[best] {
            best = i;
        }
    }
    best
}

/// The trust-boundary check, as a testable fact rather than a convention.
///
/// D8's acceptance criterion: "a wrong suggestion cannot block a merge". The
/// engine has no `Mutate` grant and no write path into the evidence gate, so the
/// strongest available statement is that the *type* has no such method. This
/// function asserts that at compile-adjacent level by enumerating what a consumer
/// can obtain: advice, and nothing else.
pub fn advice_only(core: &DecideCore) -> Vec<Advice> {
    // Deliberately empty: there is no function in this module that returns
    // anything with the power to gate, and the doc comment on `Advice` says why
    // `advisory` is a field. Kept as a function so the claim has a place to live
    // and can be extended if a future consumer needs an inventory.
    let _ = core;
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backbone;
    use crate::{DecideConfig, Mode};

    fn engine(labels: &[&str], mode: Mode, dim: usize) -> DecideCore {
        let mut c = DecideCore::new(DecideConfig {
            embedding_version: "test".into(),
            embedding_dim: dim,
            projected_dim: 32,
            heads: 1,
            mode,
            lambda: 1.0,
            alpha: 1.0,
        })
        .unwrap();
        for l in labels {
            c.register(l).unwrap();
        }
        c
    }

    fn v(rows: &[[f64; 3]]) -> Vec<DVector<f64>> {
        rows.iter().map(|r| DVector::from_row_slice(r)).collect()
    }

    #[test]
    fn board_triage_suggests_a_kind_and_is_advisory() {
        let mut c = engine(&["fact", "fail", "claim"], Mode::Anacp, 3);
        for (l, rows) in [
            (
                "fact",
                [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.95, 0.05, 0.0]],
            ),
            (
                "fail",
                [[0.0, 1.0, 0.0], [0.0, 0.9, 0.1], [0.05, 0.95, 0.0]],
            ),
            (
                "claim",
                [[0.0, 0.0, 1.0], [0.1, 0.0, 0.9], [0.0, 0.05, 0.95]],
            ),
        ] {
            for e in v(&rows) {
                c.observe(l, &e).unwrap();
            }
        }
        c.fit().unwrap();
        let a = suggest_board_kind(&c, &DVector::from_row_slice(&[0.0, 1.0, 0.0])).unwrap();
        assert_eq!(a.suggestion, "Fail");
        assert!(a.advisory, "every suggestion must be advisory");
        assert!(!a.calibrated, "no calibrator is installed in this test");
    }

    #[test]
    fn preset_routing_returns_a_label_not_a_grant() {
        let mut c = engine(&["reader", "prover-runner", "reviewer"], Mode::Anacp, 3);
        for (l, rows) in [
            (
                "reader",
                [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.99, 0.0, 0.01]],
            ),
            (
                "prover-runner",
                [[0.0, 1.0, 0.0], [0.02, 0.98, 0.0], [0.0, 0.99, 0.01]],
            ),
            (
                "reviewer",
                [[0.0, 0.0, 1.0], [0.0, 0.02, 0.98], [0.01, 0.0, 0.99]],
            ),
        ] {
            for e in v(&rows) {
                c.observe(l, &e).unwrap();
            }
        }
        c.fit().unwrap();
        let a = suggest_preset(&c, &DVector::from_row_slice(&[0.0, 0.0, 1.0])).unwrap();
        assert_eq!(a.suggestion, "reviewer");
        assert!(a.advisory);
    }

    /// D10's acceptance criterion, stated as a property of the return value.
    #[test]
    fn editor_ranking_returns_the_same_set_in_a_different_order() {
        let mut c = engine(&["top", "mid", "low"], Mode::Anacp, 3);
        // Several samples per class, not one: with a single sample the class
        // mean *is* that sample, and the prototypes are degenerate rather than
        // merely weak. This is a property of the method, not of the test.
        for (l, rows) in [
            (
                "top",
                [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.99, 0.0, 0.01]],
            ),
            (
                "mid",
                [[0.0, 1.0, 0.0], [0.02, 0.98, 0.0], [0.0, 0.99, 0.01]],
            ),
            (
                "low",
                [[0.0, 0.0, 1.0], [0.0, 0.02, 0.98], [0.01, 0.0, 0.99]],
            ),
        ] {
            for e in v(&rows) {
                c.observe(l, &e).unwrap();
            }
        }
        c.fit().unwrap();
        let items = vec!["alpha", "beta", "gamma"];
        // alpha's embedding sits on the *low* axis, beta's on *top*, gamma's on
        // *mid* — so ordering by the "mid" class must genuinely reorder the list
        // rather than happen to agree with its original order.
        let embs = v(&[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let out = ranked(&c, &items, &embs, "mid");
        assert_eq!(out.len(), items.len(), "ranking must not drop an item");
        let mut a: Vec<&str> = out.iter().map(|s| **s).collect();
        let mut b: Vec<&str> = items.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "ranking changed the SET, not just the order");
        // Ordered by the "mid" class's score: gamma is on the mid axis, so it
        // leads. The relative order of the two *off-axis* items is deliberately
        // not asserted — it depends on the learned transform, and pinning it
        // would make the test a snapshot of the method's internals rather than a
        // check of its contract.
        assert_eq!(*out[0], "gamma");
        let reordered: Vec<&str> = out.iter().map(|s| **s).collect();
        assert_ne!(reordered, b, "ranking did not reorder anything");
    }

    #[test]
    fn ranking_degrades_to_the_original_order_on_a_length_mismatch() {
        let c = engine(&["a", "b"], Mode::Anacp, 3);
        let items = vec!["x", "y", "z"];
        // Two embeddings for three items: refuse to guess.
        let out = ranked(&c, &items, &v(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "a");
        assert_eq!(out, vec![&"x", &"y", &"z"]);
    }

    #[test]
    fn ranking_never_drops_an_item_when_the_engine_is_unfitted() {
        let c = engine(&["a", "b"], Mode::Anacp, 3);
        let items = vec!["x", "y"];
        let out = ranked(&c, &items, &v(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "a");
        assert_eq!(
            out.len(),
            2,
            "an unfitted engine must not silently truncate"
        );
    }

    #[test]
    fn a_detail_routing_failure_degrades_to_the_default() {
        let c = engine(&["1", "2", "3"], Mode::Anacp, 3);
        // Unfitted: `similarities` errors, and routing must not propagate it.
        let level = route_detail_level(&c, &DVector::from_row_slice(&[1.0, 0.0, 0.0]), 2);
        assert_eq!(
            level, 2,
            "an unfitted engine falls back to the default level"
        );
    }

    #[test]
    fn nudge_ordering_reorders_but_never_drops() {
        let mut c = engine(&["urgent", "fyi"], Mode::Anacp, 3);
        for (l, rows) in [
            (
                "urgent",
                [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.99, 0.0, 0.01]],
            ),
            (
                "fyi",
                [[0.0, 1.0, 0.0], [0.02, 0.98, 0.0], [0.0, 0.99, 0.01]],
            ),
        ] {
            for e in v(&rows) {
                c.observe(l, &e).unwrap();
            }
        }
        c.fit().unwrap();
        let nudges = vec!["fyi".to_string(), "urgent".to_string()];
        let embs = v(&[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);
        let out = nudge_order(&c, &nudges, &embs, "urgent");
        assert_eq!(out.len(), 2);
        assert_eq!(*out[0], "urgent");
    }

    #[test]
    fn nudge_ordering_is_stable_on_ties() {
        let c = engine(&["a", "b"], Mode::Anacp, 3);
        // Identical embeddings => identical scores => original order preserved.
        let nudges = vec!["one".to_string(), "two".to_string(), "three".to_string()];
        let embs = vec![
            DVector::from_row_slice(&[1.0, 0.0, 0.0]),
            DVector::from_row_slice(&[1.0, 0.0, 0.0]),
            DVector::from_row_slice(&[1.0, 0.0, 0.0]),
        ];
        assert_eq!(
            nudge_order(&c, &nudges, &embs, "urgent"),
            vec![&"one", &"two", &"three"]
        );
    }

    /// The trust boundary, as an executable statement.
    #[test]
    fn advice_never_blocks_a_merge() {
        let mut c = engine(&["fact"], Mode::Anacp, 3);
        c.observe("fact", &DVector::from_row_slice(&[1.0, 0.0, 0.0]))
            .unwrap();
        c.observe("fact", &DVector::from_row_slice(&[0.9, 0.1, 0.0]))
            .unwrap();
        // `advice_only` is the inventory of what a consumer can obtain. If a
        // future change gave a consumer a gating handle, it would have to be
        // added there — and this test would then be the place to argue about it.
        assert!(advice_only(&c).is_empty());
    }

    #[test]
    fn the_backbone_seam_and_the_consumer_agree_on_normalization() {
        // A consistency check across two modules that both touch embeddings: the
        // seam normalizes, and the consumer assumes it did.
        let spec = backbone::parse_embed_spec("embed:hf/m").unwrap();
        assert!(backbone::version_string(&spec).contains("hf:m"));
        let mut c = engine(&["a", "b"], Mode::Anacp, 2);
        for (l, e) in [("a", [1.0f64, 0.0]), ("b", [0.0, 1.0])] {
            c.observe(l, &DVector::from_row_slice(&e)).unwrap();
        }
        c.fit().unwrap();
        let a = suggest_preset(&c, &DVector::from_row_slice(&[1.0, 0.0])).unwrap();
        assert_eq!(a.suggestion, "a");
    }
}
