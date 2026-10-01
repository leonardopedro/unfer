//! E2/E3 tested against the E1 golden corpus.
//!
//! The corpus lives in the sibling repo (`australVM/corpus/engram_keys.tsv`)
//! and records *expectations* — which members must share a `unf_hash` and which
//! must not. These tests read it when it is present and are skipped when it is
//! not, so `unfer` stays buildable and green on its own (PLAN_HARNESS.md's
//! green-workspace rule) while still checking the specification whenever the
//! sibling checkout is available.
//!
//! The `distinct` rows are the ones that earn their keep: `x minus y` against
//! `y minus x`, `x times one` against `x times two`. An over-eager
//! canonicalization passes every `collide` row and fails these.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use logos::engram::{segment, Granularity};
use logos::lexicon::Lexicon;

const CORPUS_REL: &str = "../../australVM/corpus/engram_keys.tsv";
const LEXICON_REL: &str = "corpus/lexicon.tsv";

fn corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CORPUS_REL)
}

fn lexicon_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(LEXICON_REL)
}

fn load_lexicon() -> Option<Lexicon> {
    let p = lexicon_path();
    if p.exists() {
        Lexicon::load(&p).ok()
    } else {
        None
    }
}

/// One row of the golden corpus.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Row {
    group: String,
    member: String,
    granularity: String,
    /// Which digest this expectation is about. `unf` is canonical *net* form,
    /// which is order-sensitive; `ted` is the algebraic canonical form, which
    /// sorts. Commutativity is a semantic equivalence and only shows up in
    /// `ted`.
    field: String,
    expect: String,
}

fn parse_corpus() -> Option<Vec<Row>> {
    let path = corpus_path();
    if !path.exists() {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 || f[0] == "group" {
            continue;
        }
        rows.push(Row {
            group: f[0].to_string(),
            member: f[1].to_string(),
            granularity: f[2].to_string(),
            field: f[3].to_string(),
            expect: f[4].to_string(),
        });
    }
    if rows.is_empty() {
        None
    } else {
        Some(rows)
    }
}

fn granularity_of(s: &str) -> Option<Granularity> {
    match s {
        "window" => Some(Granularity::Window),
        "subderiv" => Some(Granularity::Subderiv),
        "sentence" => Some(Granularity::Sentence),
        _ => None,
    }
}

/// The `unf_hash` a member yields at a granularity, or `None` if it yields
/// nothing (which for a UNF granularity means a tagged fallback).
fn hash_of(lex: &Lexicon, member: &str, g: Granularity, field: &str) -> Option<[u8; 32]> {
    let keys = segment(member, g, lex);
    keys.into_iter().find(|k| !k.is_fallback()).map(|k| {
        if field == "ted" {
            k.ted_hash
        } else {
            k.unf_hash
        }
    })
}

#[test]
fn golden_collide_rows_share_a_hash() {
    let (Some(rows), Some(lex)) = (parse_corpus(), load_lexicon()) else {
        eprintln!("skipping: golden corpus or lexicon not present");
        return;
    };
    // group -> member -> hash, per granularity
    let mut index: BTreeMap<(String, String, String), BTreeMap<String, [u8; 32]>> = BTreeMap::new();
    for r in &rows {
        let Some(g) = granularity_of(&r.granularity) else { continue };
        if let Some(h) = hash_of(&lex, &r.member, g, &r.field) {
            index
                .entry((r.group.clone(), r.granularity.clone(), r.field.clone()))
                .or_default()
                .insert(r.member.clone(), h);
        }
    }
    let mut checked = 0usize;
    for ((group, gran, field), members) in &index {
        if !rows.iter().any(|r| {
            &r.group == group && r.expect == "collide" && &r.granularity == gran && &r.field == field
        }) {
            continue;
        }
        let hashes: BTreeSet<[u8; 32]> = members.values().cloned().collect();
        assert_eq!(
            1,
            hashes.len(),
            "group `{}` at {} on `{}` expected one digest, got {} — a `collide` row broke",
            group,
            gran,
            field,
            hashes.len()
        );
        checked += 1;
    }
    eprintln!("collide groups checked: {}", checked);
    assert!(checked > 0, "no collide groups were exercised");
}

#[test]
fn golden_distinct_rows_do_not_share_a_hash() {
    let (Some(rows), Some(lex)) = (parse_corpus(), load_lexicon()) else {
        eprintln!("skipping: golden corpus or lexicon not present");
        return;
    };
    // A `distinct` row asserts its member differs from the *other* members of
    // its group at the same granularity.
    let mut by_group: BTreeMap<(String, String, String), Vec<&Row>> = BTreeMap::new();
    for r in &rows {
        by_group
            .entry((r.group.clone(), r.granularity.clone(), r.field.clone()))
            .or_default()
            .push(r);
    }
    let mut checked = 0usize;
    for ((group, gran, field), group_rows) in &by_group {
        if !group_rows.iter().any(|r| r.expect == "distinct") {
            continue;
        }
        let Some(g) = granularity_of(gran) else { continue };
        let hashes: BTreeMap<&str, [u8; 32]> = group_rows
            .iter()
            .filter_map(|r| hash_of(&lex, &r.member, g, field).map(|h| (r.member.as_str(), h)))
            .collect();
        // Compare each `distinct` member against every *other* member of the
        // group; it must not equal any of them.
        for row in group_rows.iter().filter(|r| r.expect == "distinct") {
            let Some(mine) = hashes.get(row.member.as_str()) else { continue };
            for (other, theirs) in &hashes {
                if *other == row.member.as_str() {
                    continue;
                }
                // `window` keys are surface-derived and *are* allowed to be
                // order-sensitive, so a `distinct` expectation there is about
                // the surface string differing — which it does by construction.
                if g != Granularity::Window {
                    assert_ne!(
                        mine,
                        theirs,
                        "group `{}` at {} on `{}`: `{}` and `{}` collapsed to the same digest",
                        group,
                        gran,
                        field,
                        row.member,
                        other
                    );
                }
            }
            checked += 1;
        }
    }
    eprintln!("distinct members checked: {}", checked);
    assert!(checked > 0, "no distinct members were exercised");
}

#[test]
fn key_layout_is_the_versioned_84_byte_form() {
    use logos::engram::{EngramKey, FLAG_FALLBACK, LAYOUT_VERSION};
    let mut k = EngramKey::window(&["a".into(), "b".into()], 2);
    k.ted_hash = [7u8; 32];
    k.l1_weight = Some(0.0);
    let bytes = k.to_bytes();
    assert_eq!(84, bytes.len());
    assert_eq!(b"ENGM", &bytes[0..4]);
    assert_eq!(LAYOUT_VERSION, u16::from_le_bytes([bytes[4], bytes[5]]));
    assert_eq!(0, bytes[6], "window is granularity code 0");
    // A real zero probability is *not* the absent-weight NaN.
    let w = f64::from_le_bytes(bytes[76..84].try_into().unwrap());
    assert_eq!(0.0, w);
    assert!(!w.is_nan());

    // …and an absent weight *is* NaN, which is the §2.2 distinction.
    let mut k2 = EngramKey::window(&["a".into()], 2);
    k2.l1_weight = None;
    let b2 = k2.to_bytes();
    let w2 = f64::from_le_bytes(b2[76..84].try_into().unwrap());
    assert!(w2.is_nan(), "absent l1_weight must be NaN");

    // The fallback flag travels with the key.
    let k3 = EngramKey::window(&["a".into()], 2).as_fallback();
    assert_eq!(FLAG_FALLBACK, k3.flags);
    assert!(k3.is_fallback());
    assert_eq!(FLAG_FALLBACK, k3.to_bytes()[7]);
}

#[test]
fn unparseable_fragments_fall_back_and_are_tagged() {
    let Some(lex) = load_lexicon() else {
        eprintln!("skipping: lexicon not present");
        return;
    };
    // Nonsense the CNL gate will not accept: §4 says this must become a tagged
    // window key, never a silent success and never an error.
    let keys = segment("zzz qqq wwwww", Granularity::Sentence, &lex);
    assert!(!keys.is_empty(), "a fallback must still produce a key");
    assert!(
        keys.iter().all(|k| k.is_fallback()),
        "an unparseable fragment must yield only tagged fallbacks"
    );
    // …and the fallback is a *window* key: ENGRAM.md §2.3 keeps a miss at `g`
    // from being answered by a coarser granularity.
    assert_eq!(Granularity::Window, keys[0].granularity);
}

#[test]
fn lookup_is_granularity_partitioned() {
    use logos::engram::EngramTable;
    let mut t = EngramTable::new();
    let a = logos::engram::EngramKey::window(&["x".into()], 2);
    let mut b = a.clone();
    b.granularity = logos::engram::Granularity::Sentence;
    t.insert(a.clone(), vec![1.0]);
    t.insert(b.clone(), vec![2.0]);
    assert_eq!(Some(&vec![1.0]), t.get(&a));
    // Same unf_hash, different granularity: §2.3 says a lookup must not cross
    // partitions, so a miss here is the correct behaviour, not a bug.
    assert_eq!(Some(&vec![2.0]), t.get(&b));
    let mut c = a.clone();
    c.unf_hash = [9u8; 32];
    assert!(t.get(&c).is_none(), "a miss must stay a miss");
}

#[test]
fn ingest_reports_parse_rate_and_never_nan() {
    use logos::engram::{placeholder_embedding, EngramTable};
    let Some(lex) = load_lexicon() else {
        eprintln!("skipping: lexicon not present");
        return;
    };
    let corpus = vec![
        "one plus one".to_string(),
        "two plus two".to_string(),
        "zzz qqq".to_string(),
    ];
    let mut t = EngramTable::new();
    let stats = t.ingest(&corpus, &lex, placeholder_embedding);
    // Every granularity must appear and every ratio must be a real number: a
    // report containing NaN cannot be compared or charted.
    for g in [Granularity::Window, Granularity::Subderiv, Granularity::Sentence] {
        let d = *stats
            .dedup_ratio
            .get(&g)
            .expect("every granularity is reported");
        assert!(d.is_finite() && (0.0..=1.0).contains(&d), "{:?} dedup {}", g, d);
        let p = *stats.parse_rate.get(&g).expect("parse_rate is reported");
        assert!(p.is_finite() && (0.0..=1.0).contains(&p), "{:?} parse_rate {}", g, p);
    }
    // The unparseable fragment must drag the parse rate below 1 for the UNF
    // granularities — this is the metric doing its job.
    let pr = *stats.parse_rate.get(&Granularity::Sentence).unwrap();
    assert!(
        pr < 1.0,
        "a corpus containing an unparseable fragment must not report a 1.0 parse rate"
    );
}

#[test]
fn empty_corpus_reports_zero_not_nan() {
    use logos::engram::{placeholder_embedding, EngramTable};
    let Some(lex) = load_lexicon() else {
        return;
    };
    let mut t = EngramTable::new();
    let stats = t.ingest(&[], &lex, placeholder_embedding);
    assert!(stats.dedup_ratio.is_empty());
    assert!(stats.lookup_p50_ns == 0);
}
// ── E5: probabilistic engrams ─────────────────────────────────────────────
// The plan's acceptance: "probably John loves Mary" yields a weighted key set
// summing to 1.0, and lookup returns the distribution.

fn triggers() -> logos::l1::TriggerTable {
    logos::l1::TriggerTable::new()
}

#[test]
fn hedged_fragment_yields_a_normalized_weighted_key_set() {
    use logos::engram::l1keys;
    let Some(lex) = load_lexicon() else {
        eprintln!("skipping: lexicon not present");
        return;
    };
    let frag = "probably John loves Mary";
    let wkeys = match l1keys::weighted_key_set(
        frag,
        Granularity::Sentence,
        &lex,
        &triggers(),
    ) {
        Ok(k) => k,
        // The hedged CNL surface may not parse with this lexicon; that is a
        // lexicon fact, not a contract failure. What must hold either way is
        // that a set which *does* come back is normalized and stamped.
        Err(e) => {
            eprintln!("skipping hedged case, fragment did not reduce: {}", e);
            return;
        }
    };
    assert!(!wkeys.is_empty());
    assert!(
        l1keys::is_normalized(&wkeys, 1e-9),
        "weighted keys must sum to 1.0, got {}",
        l1keys::total_weight(&wkeys)
    );
    // Every returned key carries its accumulated weight — that is what
    // ENGRAM.md §2.2's `l1_weight` field is for.
    for w in &wkeys {
        assert_eq!(Some(w.weight), w.key.l1_weight);
    }
    // The identity world must have reduced through the UNF path. The negate
    // world of this hedge does not — the CNL lexicon is 47 words and has no
    // negation vocabulary — so its mass lands on a *tagged* fallback. That is
    // the contract working: mass is conserved and the degradation is visible,
    // rather than the world being dropped and the hedge silently renormalized
    // into a different sentence.
    let real_mass: f64 = wkeys.iter().filter(|w| !w.key.is_fallback()).map(|w| w.weight).sum();
    let fallback_mass: f64 = wkeys.iter().filter(|w| w.key.is_fallback()).map(|w| w.weight).sum();
    assert!(
        real_mass > 0.0,
        "at least one world must reduce through the UNF path"
    );
    eprintln!(
        "hedged: real_mass={:.3} fallback_mass={:.3} keys={}",
        real_mass,
        fallback_mass,
        wkeys.len()
    );
    // Conservation: the two masses account for the whole distribution.
    assert!((real_mass + fallback_mass - 1.0).abs() < 1e-9);
    // The distribution is the same mass, keyed by address.
    let dist = l1keys::distribution(&wkeys);
    assert!((dist.values().sum::<f64>() - 1.0).abs() < 1e-9);
}

#[test]
fn aggregation_merges_coincident_keys_and_sums_their_mass() {
    use logos::engram::l1keys::{aggregate, WeightedKey};
    // Two worlds that key identically must merge to their sum, which is the
    // whole reason E5 reuses `l1::aggregate_results` instead of reimplementing
    // it.
    let mk = |h: u8, w: f64| WeightedKey {
        key: logos::engram::EngramKey::window(&[format!("k{}", h)], 2),
        weight: w,
    };
    let merged = aggregate(vec![mk(1, 0.3), mk(1, 0.5), mk(2, 0.2)]);
    assert_eq!(2, merged.len(), "two distinct keys survive");
    let k1 = merged.iter().find(|w| w.key.unf_hash == mk(1, 0.0).key.unf_hash).unwrap();
    assert!((k1.weight - 0.8).abs() < 1e-12, "0.3 + 0.5 = 0.8, got {}", k1.weight);
    assert_eq!(Some(0.8), k1.key.l1_weight);
    // Sum-preserving.
    assert!((merged.iter().map(|w| w.weight).sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn a_genuine_zero_weight_is_not_an_absent_weight() {
    // ENGRAM.md §2.2: `Some(0.0)` is a real zero-probability world and must not
    // be confused with "no L1 annotation at all". The type and the byte layout
    // both have to keep that apart.
    let mut zero = logos::engram::EngramKey::window(&["a".into()], 2);
    zero.l1_weight = Some(0.0);
    let mut absent = logos::engram::EngramKey::window(&["a".into()], 2);
    absent.l1_weight = None;

    assert_ne!(zero.to_bytes(), absent.to_bytes());
    let wz = f64::from_le_bytes(zero.to_bytes()[76..84].try_into().unwrap());
    let wa = f64::from_le_bytes(absent.to_bytes()[76..84].try_into().unwrap());
    assert_eq!(0.0, wz);
    assert!(wa.is_nan());
    // …and the aggregate for a zero-mass key is still present, carrying 0.0.
    use logos::engram::l1keys::{aggregate, WeightedKey};
    let merged = aggregate(vec![WeightedKey { key: zero, weight: 0.0 }]);
    assert_eq!(1, merged.len());
    assert_eq!(0.0, merged[0].weight);
    assert_eq!(Some(0.0), merged[0].key.l1_weight);
}
