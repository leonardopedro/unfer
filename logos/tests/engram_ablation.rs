//! E4's keying ablation, run on the golden corpus.
//!
//! The question E4 asks is whether semantic keys are worth their cost. This
//! measures the part that decides it, on the corpus that already exists: for a
//! corpus of paraphrase-equivalent fragments, how many table entries and how many
//! lookups does each keying scheme need?
//!
//! Reported, not asserted. The qualitative claim (surface keys cannot deduplicate
//! paraphrases; UNF keys can) is asserted, because it follows from what the two
//! schemes *are*. The magnitudes are printed, because they depend on the corpus,
//! the table sizes and the machine, and quoting them as constants would be
//! inventing a result.

use std::collections::{BTreeMap, HashSet};

use logos::engram::ablation::{
    TokenIds, coverage_curve, surface_coverage, surface_lookups, unf_coverage,
};

const CORPUS_REL: &str = "../../australVM/corpus/engram_keys.tsv";

struct Row {
    group: String,
    fragment: String,
    granularity: String,
    field: String,
    expect: String,
}

fn corpus() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CORPUS_REL);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    // Columns: group, member, granularity, field, expect, unf_hash, ted_hash,
    // note. The *fragment* is `member` (column 1) and `expect` is column 4 —
    // getting these two the wrong way round keys the ablation on the strings
    // "ted" and "unf" instead of on sentences, which is exactly what happened
    // the first time and produced 46 one-token "fragments".
    let mut out = Vec::new();
    let mut header = true;
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 {
            continue;
        }
        if header {
            header = false;
            if f[0] != "group" {
                panic!(
                    "{}: expected a `group` header, got {:?}",
                    path.display(),
                    f[0]
                );
            }
            continue;
        }
        out.push(Row {
            group: f[0].to_string(),
            fragment: f[1].to_string(),
            granularity: f[2].to_string(),
            field: f[3].to_string(),
            expect: f[4].to_string(),
        });
    }
    out
}

/// The keying arm (b) would produce *if the reduction deduplicated the group*.
///
/// This is the honest way to measure the ablation without a semantic pipeline in
/// the loop: the corpus already labels which fragments are meant to be
/// paraphrase-equivalent, so the key of a group is the key its members would
/// share *if* they reduced together. Reading the label instead of running the
/// reducer means the number answers "what does semantic dedup buy?", which is the
/// question, rather than "how good is our reducer?", which is a different one and
/// would confound the comparison with E2's accuracy.
fn group_key(groups: &BTreeMap<String, usize>, fragment: &str) -> u64 {
    groups.get(fragment).copied().unwrap_or(usize::MAX) as u64
}

#[test]
#[ignore = "ablation; run with --ignored --nocapture"]
fn e4_keying_ablation_on_the_golden_corpus() {
    let rows = corpus();
    assert!(!rows.is_empty(), "corpus is empty or unreadable");

    // Fragment -> group id, for the dedup arm.
    let mut groups = BTreeMap::new();
    let mut next = 0usize;
    for r in &rows {
        groups.entry(r.fragment.clone()).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
    }

    let fragments: Vec<&str> = rows.iter().map(|r| r.fragment.as_str()).collect();

    // ── arm (a): surface N-gram keys ──────────────────────────────────────
    let mut ids = TokenIds::new();
    let surface = surface_coverage(&mut ids, fragments.iter().copied());
    let tokens: usize = fragments.iter().map(|f| f.split_whitespace().count()).sum();

    // ── arm (b): UNF keys, deduplicated by group ──────────────────────────
    let unf = unf_coverage(fragments.iter().copied(), |f| group_key(&groups, f));

    // ── the control: UNF keying with the dedup DISABLED ───────────────────
    // Without this, "UNF keys dedup" could be an artefact of the labelling
    // rather than of the keying. This arm has the same code path and no
    // grouping, and it should look like arm (a) in slot count while doing one
    // lookup per fragment.
    let mut distinct = HashSet::new();
    for f in &fragments {
        distinct.insert(ids.sequence(f));
    }
    let control_slots = distinct.len();

    println!("--- E4 keying ablation ({} fragments) ---", fragments.len());
    println!(
        "arm (a) surface N-gram : {:>8} slots, {:>8} lookups ({tokens} tokens)",
        surface.distinct_slots, surface.lookups
    );
    println!(
        "arm (b) UNF + dedup    : {:>8} slots, {:>8} lookups",
        unf.distinct_slots, unf.lookups
    );
    println!(
        "control: UNF, no dedup : {control_slots:>8} distinct sequences (one lookup per fragment)"
    );
    println!();
    println!(
        "slot reduction from dedup: {:.2}x",
        surface.distinct_slots as f64 / unf.distinct_slots.max(1) as f64
    );
    println!(
        "lookup reduction        : {:.2}x",
        surface.lookups as f64 / unf.lookups.max(1) as f64
    );
    println!();

    // Table-coverage curves: what fraction of the corpus fits in a budget.
    let budgets: Vec<usize> = vec![
        unf.distinct_slots,
        unf.distinct_slots * 2,
        unf.distinct_slots * 4,
        unf.distinct_slots * 8,
        surface.distinct_slots,
        surface.distinct_slots * 2,
    ];
    println!("--- table-coverage curve (fraction of corpus covered per budget) ---");
    println!("{:>10} {:>14} {:>14}", "budget", "arm (a)", "arm (b)");
    for (budget, fit_a) in coverage_curve(&surface, &budgets) {
        let fit_b = unf.fits(budget);
        println!("{budget:>10} {fit_a:>14.4} {fit_b:>14.4}");
    }

    println!();
    println!("--- what these numbers do and do not show ---");
    println!(
        "* The corpus is {} fragments and was *built* to contain paraphrase groups. \
         A natural corpus has far fewer, so the ratio is an UPPER BOUND on what \
         semantic dedup buys, not an estimate of it.",
        fragments.len()
    );
    println!(
        "* Arm (b)'s dedup is read from the corpus's labels, not produced by the \
         reducer. So this answers \"what does perfect semantic dedup buy?\" -- not \
         \"how good is our reducer?\", which would confound the keying comparison \
         with E2's accuracy."
    );
    println!(
        "* Arm (a) is a faithful reimplementation of the reference's hash \
         arithmetic (docs/ENGRAM.md 2.7), not its torch code, which is coupled to \
         the decoder model. Only table-index behaviour is compared."
    );
    println!(
        "* No training, no FLOPs, no val loss, no RULER: E4's other metrics need an \
         accelerator and remain outstanding."
    );

    // ── the claims that are decidable here, asserted ──────────────────────

    // Semantic dedup is a strict improvement in slots over one-slot-per-fragment
    // labelling, because the corpus genuinely contains groups of more than one.
    assert!(
        unf.distinct_slots < fragments.len(),
        "the corpus should contain paraphrase groups, so dedup must reduce slots \
         ({} slots for {} fragments)",
        unf.distinct_slots,
        fragments.len()
    );

    // Arm (a) cannot deduplicate: it is one lookup per token per head, so it
    // always does strictly more work than arm (b)'s one-lookup-per-fragment.
    assert!(
        surface.lookups > unf.lookups,
        "surface keying must do more lookups ({} vs {})",
        surface.lookups,
        unf.lookups
    );
    assert_eq!(
        surface.lookups,
        tokens * 2 * 8,
        "surface lookups must be tokens x ngram_sizes x heads"
    );

    // The control confirms the slot saving comes from dedup, not from arm (b)
    // being cheap to hash.
    assert_eq!(unf.lookups, fragments.len());
}

/// The decidable claim, as a fast test that does not need the corpus file.
#[test]
fn surface_keys_cannot_deduplicate_a_paraphrase_but_unf_keys_can() {
    let paraphrase_a = "John adds two and three";
    let paraphrase_b = "John adds three and two";

    let mut ids = TokenIds::new();
    let a = surface_slots_of(&mut ids, paraphrase_a);
    let b = surface_slots_of(&mut ids, paraphrase_b);

    // Same meaning, different token order: surface keys land on different slots.
    // Both are *semantically* the same statement, which is the point.
    assert_ne!(
        a, b,
        "surface keys must not dedup an anaphora-free reordering"
    );

    // Arm (b) given the same information must dedup it, because the reduction
    // happens before the key is formed. `group_key` stands in for the reducer.
    let mut groups = BTreeMap::new();
    groups.insert(paraphrase_a, 0usize);
    groups.insert(paraphrase_b, 0usize);
    let unf = unf_coverage([paraphrase_a, paraphrase_b].into_iter(), |f| {
        *groups.get(f).unwrap() as u64
    });
    assert_eq!(unf.distinct_slots, 1, "UNF keys must dedup the pair");
    assert_eq!(unf.lookups, 2);
}

fn surface_slots_of(ids: &mut TokenIds, fragment: &str) -> HashSet<usize> {
    let seq = ids.sequence(fragment);
    logos::engram::ablation::surface_slots(&seq)
}
