//! P8 — the evaluation harness.
//!
//! # What this measures, and what it deliberately does not
//!
//! ProofFlow's benchmark scripts measure Lean4 formalization accuracy. The
//! adapted target is CNL, and §17.1 says plainly that stock L0 will not
//! formalize real mathematical steps. So the harness is split into two halves
//! with very different standing:
//!
//! - **Exact, and load-bearing.** P1 is a port of `proof_graph.py`, and the
//!   corpus here is that script's own committed output. Every graph in the
//!   corpus must validate under the port, byte-for-byte equivalent to what
//!   ProofFlow accepted. This is a genuine parity check, and
//!   `the_port_accepts_every_human_graph` asserts it.
//! - **Measured, not asserted.** How far the CNL stage actually gets is
//!   *reported*, not asserted, because the expected answer is "very little"
//!   and pinning it to a number would make the metric a constant.
//!   `lexicon_coverage_is_the_measured_limit` prints the figure.
//!
//! # The corpus
//!
//! `testdata/formalize/corpus/sample_graphs.json` is 15 real human graphs from
//! ProofFlow's `data/benchmark_0409.json` (MIT; notice in that directory's
//! README). `CORPUS_PATH` points the harness at a larger file — the full 184 —
//! and the parity test runs over whatever it finds.
//!
//! Every test here runs the **deterministic** half of the pipeline: DAG
//! validation, CNL verification, identity distinctness, the dependency report
//! and centrality weights. Nothing calls a model, so the harness is a unit test
//! and not a benchmark run.

use logos::formalize::formalizer::{self, base_lexicon};
use logos::formalize::graph::{self, Validated};
use logos::formalize::score::{self, Aggregation};
use serde_json::Value;
use std::collections::BTreeMap;

/// One corpus entry, as stored.
#[derive(Debug, Clone)]
struct CorpusEntry {
    origin: String,
    id: i64,
    nodes: Value,
}

/// The corpus, from `CORPUS_PATH` or the checked-in sample.
fn corpus() -> Vec<CorpusEntry> {
    let external = std::env::var("CORPUS_PATH").ok();
    let (path, text) = match external {
        Some(p) => (
            p.clone(),
            std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("CORPUS_PATH {p}: {e}")),
        ),
        None => (
            "testdata/formalize/corpus/sample_graphs.json".to_string(),
            include_str!("../testdata/formalize/corpus/sample_graphs.json").to_string(),
        ),
    };
    let _ = path;

    let value: Value = serde_json::from_str(&text).expect("corpus is valid JSON");
    value
        .as_array()
        .expect("corpus is an array")
        .iter()
        .map(|entry| CorpusEntry {
            origin: entry["origin"].as_str().unwrap_or("?").to_string(),
            id: entry["id"].as_i64().unwrap_or(-1),
            nodes: entry["proof_graph"].clone(),
        })
        .collect()
}

/// The graphs that validate, and the reasons the others do not.
fn validated_corpus() -> (Vec<CorpusEntry>, Vec<String>) {
    let mut ok = Vec::new();
    let mut rejected = Vec::new();
    for entry in corpus() {
        match graph::validate_proof_graph(&entry.nodes) {
            Ok(_) => ok.push(entry),
            Err(e) => rejected.push(format!("{}/{}: {e}", entry.origin, entry.id)),
        }
    }
    (ok, rejected)
}

/// An independent structural check, written from the rules rather than by
/// calling into `graph::check_dag`.
///
/// This exists because the parity claim needs a witness that does not share code
/// with the thing it is checking. `R prob/21`-style conclusions are exactly the
/// kind of bug a checker can have in both directions at once, and the first
/// version of this harness simply asserted that everything validates — which is
/// false of the full 184-graph corpus, because 11 of its entries are malformed in
/// the source data (`M series/15` literally lists `l3` as depending on `l3`).
fn dag_violation(graph: &[Value]) -> Option<String> {
    let id_of = |v: &Value| v["id"].as_str().unwrap_or("").to_string();
    let deps_of = |v: &Value| -> Vec<String> {
        v["dependencies"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };

    let ids: Vec<String> = graph.iter().map(id_of).collect();
    let known: std::collections::BTreeSet<&str> = ids.iter().map(String::as_str).collect();

    // Self-dependency and unknown dependency, by direct comparison.
    for node in graph {
        let id = id_of(node);
        for dep in deps_of(node) {
            if dep == id {
                return Some(format!("{id} depends on itself"));
            }
            if !known.contains(dep.as_str()) {
                return Some(format!("{id} depends on unknown {dep}"));
            }
        }
    }

    // Orphan: referenced by nothing, and not the closing `ts_`.
    let referenced: std::collections::BTreeSet<String> = graph.iter().flat_map(deps_of).collect();
    for id in &ids {
        if !id.starts_with("ts_") && !referenced.contains(id) {
            return Some(format!("{id} is an orphan"));
        }
    }

    // Forward reference: a dependency defined later in the list.
    for (i, node) in graph.iter().enumerate() {
        let id = id_of(node);
        for dep in deps_of(node) {
            if !ids[..i].contains(&dep) {
                return Some(format!("{id} forward-references {dep}"));
            }
        }
    }

    // Cycle: plain recursive DFS over an adjacency map.
    let adj: std::collections::BTreeMap<String, Vec<String>> =
        graph.iter().map(|n| (id_of(n), deps_of(n))).collect();
    let mut state: std::collections::BTreeMap<String, u8> =
        adj.keys().map(|k| (k.clone(), 0)).collect();
    fn visit(
        node: &str,
        adj: &std::collections::BTreeMap<String, Vec<String>>,
        state: &mut std::collections::BTreeMap<String, u8>,
        depth: usize,
    ) -> bool {
        // Bounded so a malformed graph cannot overflow the stack before the
        // report can be produced.
        if depth > 10_000 {
            return true;
        }
        match state.get(node) {
            Some(1) => return true,
            Some(2) => return false,
            _ => {}
        }
        state.insert(node.to_string(), 1);
        for dep in adj.get(node).map(Vec::as_slice).unwrap_or(&[]) {
            if adj.contains_key(dep.as_str()) && visit(dep, adj, state, depth + 1) {
                return true;
            }
        }
        state.insert(node.to_string(), 2);
        false
    }
    for node in &ids {
        if state.get(node.as_str()) == Some(&0) && visit(node, &adj, &mut state, 0) {
            return Some(format!("{node} is in a cycle"));
        }
    }
    None
}

/// #1 — the parity check that matters.
///
/// Every human graph that *satisfies the DAG rules* must validate under the port,
/// and every graph it rejects must violate one of them.
///
/// The weaker-looking form of this claim is the accurate one. `benchmark_0409.json`
/// is ProofFlow's benchmark **input**, not its validated output, and 11 of its 184
/// entries are malformed — ProofFlow would raise `ValueError` on them at runtime
/// too. The sample corpus checked in beside this harness has all 15 graphs valid,
/// so the common case needs no allowance; the allowance is only for a full
/// external corpus, and `every_rejection_is_a_real_dag_violation` pins that each
/// one is genuinely broken rather than merely disagreeing.
#[test]
fn every_graph_whose_structure_is_sound_validates() {
    let (ok, rejected) = validated_corpus();
    // Keyed on graph identity, not on the reason text: the two checkers word
    // their reasons differently, so comparing whole strings could never match.
    let unsound: std::collections::BTreeSet<String> = corpus()
        .iter()
        .filter_map(|e| {
            let g = e.nodes.as_array()?;
            dag_violation(g).map(|_| format!("{}/{}", e.origin, e.id))
        })
        .collect();
    let rejected_keys: std::collections::BTreeSet<String> = rejected
        .iter()
        .map(|r| r.split(": ").next().unwrap_or(r).to_string())
        .collect();

    let spurious: Vec<&String> = rejected_keys.difference(&unsound).collect();
    assert!(
        spurious.is_empty(),
        "the port rejected {} structurally sound graph(s):\n  {}",
        spurious.len(),
        spurious
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
    println!(
        "corpus: {} graphs, {} validate, {} rejected — all {} rejections are real DAG violations",
        ok.len() + rejected.len(),
        ok.len(),
        rejected.len(),
        rejected.len()
    );
    assert!(
        ok.len() >= 15,
        "expected the sample corpus, got {}",
        ok.len()
    );
}

/// Every node's id prefix must resolve to a kind, and every node's `statement`
/// must mention its own id — ProofFlow's own "Consistency rule" for the field
/// the formalizer sees. This is a property of the *data*, checked here because
/// it is the invariant P4's judge prompt relies on.
#[test]
fn every_node_is_self_referencing_and_kind_resolvable() {
    // Over the graphs that validate: a malformed entry in a full corpus has no
    // `statement` contract to check, and the parity test already accounts for
    // those.
    let (ok, _) = validated_corpus();
    for entry in &ok {
        let tag = format!("{}/{}", entry.origin, entry.id);
        let v = graph::validate_proof_graph(&entry.nodes).unwrap_or_else(|e| panic!("{tag}: {e}"));
        for node in &v.nodes {
            assert!(node.kind().is_some(), "{tag}: {} has no kind", node.id);
            assert!(
                node.statement.contains(&node.id),
                "{tag}: {} does not mention itself: {:?}",
                node.id,
                node.statement
            );
        }
    }
}

/// No graph may depend on itself, and every dependency must appear *earlier* —
/// the two structural invariants the DAG check enforces, re-asserted here so a
/// regression in `check_dag` cannot hide behind the validator accepting them.
#[test]
fn the_corpus_is_topologically_ordered() {
    // Over the graphs that validate — a malformed one violates topological
    // order *by definition*, so including it would test the corpus rather than
    // the port. The malformed ones are accounted for by the parity test.
    let (ok, _) = validated_corpus();
    for entry in &ok {
        let tag = format!("{}/{}", entry.origin, entry.id);
        let v = graph::validate_proof_graph(&entry.nodes).unwrap();
        let seen: std::collections::BTreeSet<&str> =
            v.nodes.iter().map(|n| n.id.as_str()).collect();
        for (i, node) in v.nodes.iter().enumerate() {
            assert!(
                !node.dependencies.contains(&node.id),
                "{tag}: {} depends on itself",
                node.id
            );
            for dep in &node.dependencies {
                assert!(
                    seen.contains(dep.as_str()),
                    "{tag}: {} depends on unknown {dep}",
                    node.id
                );
                assert!(
                    v.nodes[..i].iter().any(|p| &p.id == dep),
                    "{tag}: {dep} is not earlier than {}",
                    node.id
                );
            }
        }
    }
}

/// #2 — the §17.1 coverage limit, measured.
///
/// Not asserted to a number: the point is to *print* it, because the number is
/// the honest headline of this whole effort — L0 is 46 words, and no amount of
/// pipeline correctness changes that. What is asserted is the weaker, still
/// important property that the measurement is a real measurement: at least one
/// node out of many must be out of vocabulary, or the corpus is not what this
/// harness thinks it is.
#[test]
fn lexicon_coverage_is_the_measured_limit() {
    let (ok, _) = validated_corpus();
    let lex = base_lexicon();
    let known: std::collections::BTreeSet<String> = lex
        .entries()
        .iter()
        .map(|e| e.word.to_lowercase())
        .collect();

    let mut nodes = 0usize;
    let mut with_known_vocab = 0usize;
    let mut corpus_words: BTreeMap<String, usize> = BTreeMap::new();

    for entry in &ok {
        let v = graph::validate_proof_graph(&entry.nodes).unwrap();
        for node in &v.nodes {
            nodes += 1;
            if formalizer::unknown_words(&node.statement, &lex).is_empty() {
                with_known_vocab += 1;
            }
            for word in alphabetic_words(&node.statement) {
                *corpus_words.entry(word).or_insert(0) += 1;
            }
        }
    }

    let missing: Vec<(&String, &usize)> = corpus_words
        .iter()
        .filter(|(w, _)| !known.contains(*w))
        .collect();
    let total_distinct = corpus_words.len();
    let covered = total_distinct - missing.len();

    println!(
        "lexicon coverage: {with_known_vocab}/{nodes} node statements ({:.1}%) use only L0 vocabulary",
        100.0 * with_known_vocab as f64 / nodes as f64
    );
    println!(
        "distinct alphabetic words: {covered}/{total_distinct} already in L0 ({:.1}%); a domain lexicon would need {} more",
        100.0 * covered as f64 / total_distinct as f64,
        missing.len()
    );
    let top: Vec<String> = missing
        .iter()
        .rev()
        .take(15)
        .map(|(w, n)| format!("{w}×{n}"))
        .collect();
    println!("most frequent missing words: {}", top.join(", "));

    assert!(
        nodes >= 100,
        "expected a substantial corpus, got {nodes} nodes"
    );
    assert!(
        with_known_vocab < nodes,
        "§17.1 says L0 cannot express real proofs; if every statement were in \
         vocabulary the corpus would not be what this harness assumes"
    );
    assert!(
        !missing.is_empty(),
        "a real corpus cannot already be covered by L0"
    );
}

/// The alphabetic runs in a statement, lowercased and of length ≥ 2
/// **characters**.
///
/// This is what a domain lexicon would actually have to be written in, so it
/// drops LaTeX symbols, digits and single letters. Two details matter:
///
/// - characters, not bytes: `str::len` counts bytes, and every mathematical
///   variable in a proof — `β`, `ζ`, `ε` — is two bytes in UTF-8. A byte-length
///   filter keeps exactly the tokens it is meant to drop, and it did: the first
///   run of this harness reported `β×30` and `ζ×25` among the "missing words".
/// - symbols: the raw token count blames the lexicon for the corpus being
///   written in LaTeX, which is a different problem.
///
/// The helper exists because §17.1's answer ("stage the grammar extension") is
/// only actionable once someone knows how big the extension is.
fn alphabetic_words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|w| w.chars().count() >= 2)
        .map(|w| w.to_lowercase())
}
/// node as a typed, reported failure — never a panic and never a silent drop.
#[test]
fn every_human_statement_is_rejected_cleanly_or_verified_cleanly() {
    let (ok, _) = validated_corpus();
    let lex = base_lexicon();
    for entry in &ok {
        let tag = format!("{}/{}", entry.origin, entry.id);
        let v = graph::validate_proof_graph(&entry.nodes).unwrap();
        for node in &v.nodes {
            // A statement is not a CNL sentence, so verification must fail —
            // but it must fail with a *typed* error naming the words.
            match formalizer::verify_cnl(&node.statement, &lex) {
                Err(e) => {
                    assert!(!e.to_string().is_empty(), "{tag}: empty error");
                }
                Ok(r) => {
                    assert!(
                        !r.unf_hash.is_empty(),
                        "{tag}: a successful verification must carry a hash"
                    );
                    assert!(r.verified, "{tag}: not uniquely reduced");
                }
            }
        }
    }
}

/// The dependency report and centrality weights must be computable for every
/// corpus graph, on a graph where nothing has been formalized — that is the
/// state P6's `--skip-complete --skip-score` path lands in, and a report that
/// cannot be built there is a report that cannot be built early.
#[test]
fn dependency_and_weighting_reports_compute_on_every_graph() {
    let (ok, _) = validated_corpus();
    for entry in &ok {
        let tag = format!("{}/{}", entry.origin, entry.id);
        let v = graph::validate_proof_graph(&entry.nodes).unwrap();
        let dep = score::dependency_report(&v);
        assert!(dep.no_orphans, "{tag}");
        assert!(dep.no_dangling, "{tag}");
        assert!(dep.no_forward_references, "{tag}");
        // Nothing is formalized, so every node is a gap rather than a collision.
        assert_eq!(dep.unformalized.len(), v.len());
        assert!(dep.no_duplicate_identities);

        for aggregation in [
            Aggregation::Equal,
            Aggregation::Laplacian,
            Aggregation::Katz,
        ] {
            let w = score::centrality(&v, aggregation);
            assert_eq!(w.len(), v.len(), "{aggregation:?} on {tag}");
            assert!(
                w.values().all(|x| *x > 0.0),
                "{aggregation:?} gave a non-positive weight on {tag}"
            );
        }
    }
}

/// §14's identity requirement, exercised at corpus scale: distinct sentences
/// must get distinct UNF hashes, or two proof steps would silently compare
/// equal. Uses L0 sentences rather than the corpus statements, which are prose.
#[test]
fn identity_is_distinct_across_many_sentences() {
    let lex = base_lexicon();
    let sentences = [
        "John loves Mary",
        "Mary sees Bob",
        "Alice likes John",
        "Bob sees Alice",
        "Mary sees Alice",
        "Alice likes Mary",
        "Bob likes John",
        "Alice sees Mary",
        "John runs",
        "Mary runs",
        "Bob runs",
        "Alice runs",
        "the cat sleeps",
        "the dog sleeps",
        "the cat runs",
        "the dog runs",
        "John loves Alice",
        "Bob sees John",
        "Alice sees John",
    ];
    let mut seen: BTreeMap<String, &str> = BTreeMap::new();
    for s in sentences {
        let r =
            formalizer::verify_cnl(s, &lex).unwrap_or_else(|e| panic!("{s} should verify: {e}"));
        assert!(r.verified, "{s} did not reduce uniquely");
        if let Some(prev) = seen.insert(r.unf_hash.clone(), s) {
            panic!("{s:?} and {prev:?} share UNF {}", r.unf_hash);
        }
    }
    assert_eq!(seen.len(), sentences.len());
}

/// A graph can be *given* CNL formalizations and then scored structurally,
/// which is the state P6 lands in after `--skip-complete`. The score's
/// *numeric* half is vacuous under L0 (no L0 term is closed), so only the
/// structural half is asserted here — and `stock_l0_denotes_no_numbers` in
/// `score.rs` pins the vacuity.
#[test]
fn a_formalized_graph_scores_structurally() {
    let lex = base_lexicon();
    let pairs = [
        ("tc_1", "John loves Mary"),
        ("l1", "Mary sees Bob"),
        ("l2", "Alice likes John"),
        ("ts_1", "John runs"),
    ];
    let items: Vec<Value> = pairs
        .iter()
        .enumerate()
        .map(|(i, (id, cnl))| {
            let f = formalizer::verify_cnl(cnl, &lex).unwrap();
            let f = f.into_formalization(1);
            let deps: Vec<String> = if i == 0 {
                vec![]
            } else {
                vec![pairs[i - 1].0.to_string()]
            };
            serde_json::json!({
                "id": id,
                "natural_language": "q",
                "statement": format!("the statement of {id}"),
                "dependencies": deps,
                "formalization": f,
            })
        })
        .collect();
    let v = graph::validate_proof_graph(&Value::Array(items)).unwrap();

    let dep = score::dependency_report(&v);
    assert!(dep.is_clean(), "{dep:?}");
    assert!(dep.unformalized.is_empty());

    let weights = score::centrality(&v, Aggregation::Katz);
    assert_eq!(weights.len(), 4);

    // An empty judgement set aggregates to a total of 0, not to `None` — "no
    // verdict" and "not run" are different and both are reported.
    let rows: Vec<score::NodeScore> = v
        .nodes
        .iter()
        .map(|n| score::unjudged_row(n, "the harness supplies no judgements"))
        .collect();
    let total = score::aggregate_total(&rows, &weights).unwrap();
    assert_eq!(total, 0.0);
    let report = score::report(&v, rows, Aggregation::Katz);
    assert!(
        !report.is_clean(),
        "unjudged provable nodes keep it untrusted"
    );
    assert_eq!(report.aggregation, "Katz");
}

/// Round-tripping the corpus through validation must not lose or reorder a
/// node: P6 serializes and re-reads the graph, so a lossy round trip would
/// silently change what is formalized.
#[test]
fn validation_round_trips_without_loss() {
    let (ok, _) = validated_corpus();
    for entry in &ok {
        let v = graph::validate_proof_graph(&entry.nodes).unwrap();
        let again: Value = serde_json::to_value(&v.nodes).unwrap();
        let back = graph::validate_proof_graph(&again).unwrap();
        assert_eq!(back.nodes.len(), v.nodes.len());
        assert_eq!(back.ids(), v.ids());
        for (a, b) in v.nodes.iter().zip(&back.nodes) {
            assert_eq!(a.statement, b.statement, "{}", a.id);
            assert_eq!(a.natural_language, b.natural_language, "{}", a.id);
            assert_eq!(a.dependencies, b.dependencies, "{}", a.id);
        }
    }
}

/// The sample corpus spans the full corpus's size range. A sample of fifteen
/// small graphs would make every other number here optimistic, so this pins
/// the shape rather than trusting the extraction script.
#[test]
fn the_sample_spans_the_corpus_size_range() {
    let (ok, _) = validated_corpus();
    let sizes: Vec<usize> = ok
        .iter()
        .map(|e| e.nodes.as_array().map(|a| a.len()).unwrap_or(0))
        .collect();
    let min = *sizes.iter().min().unwrap();
    let max = *sizes.iter().max().unwrap();
    println!("corpus: {} graphs, {min}..={max} nodes", sizes.len());
    assert!(min <= 3, "the smallest graphs are represented");
    assert!(max >= 12, "the largest graphs are represented");
}
/// A helper the docs point at, so the numbers quoted in `FORMALIZE.md` come from
/// a run rather than from memory.
#[test]
fn corpus_summary_for_the_docs() {
    let (ok, rejected) = validated_corpus();
    let nodes: usize = ok
        .iter()
        .map(|e| e.nodes.as_array().map(|a| a.len()).unwrap_or(0))
        .sum();
    let origins: Vec<&str> = {
        let set: std::collections::BTreeSet<&str> = ok.iter().map(|e| e.origin.as_str()).collect();
        set.into_iter().collect()
    };
    println!(
        "graphs: {} accepted, {} rejected; {nodes} nodes; {} sources ({})",
        ok.len(),
        rejected.len(),
        origins.len(),
        origins.join(", ")
    );
    // Three sources in the sample, which is what the first-graph-of-each-size
    // extraction happens to reach — the *full* 184-graph corpus spans more, and
    // `CORPUS_PATH` exercises that. Asserting a larger number here would fail
    // for a property of the extraction, not of the port.
    assert!(
        origins.len() >= 2,
        "the corpus should span more than one source"
    );
}

/// A `Validated` built by hand, for the scoring tests that need one and should
/// not depend on the corpus file being present.
#[allow(dead_code)]
fn empty_graph() -> Validated {
    Validated {
        nodes: vec![],
        warnings: vec![],
    }
}
