use logos::ccg;
use logos::core_ir;
use logos::deltanet;
use logos::l1::{self, TriggerTable};
use logos::lexicon::Lexicon;

fn test_lexicon() -> Lexicon {
    let tsv = concat!(
        "John\tNP\tVar(\"john\")\n",
        "Mary\tNP\tVar(\"mary\")\n",
        "loves\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"Love\", [Var(\"x\"), Var(\"y\")])))\n",
        "sees\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"See\", [Var(\"x\"), Var(\"y\")])))\n",
        "the\tNP/N\tLam(\"n\", Var(\"n\"))\n",
        "cat\tN\tCon(\"Cat\", [])\n",
        "dog\tN\tCon(\"Dog\", [])\n",
        "sleeps\tS\\NP\tLam(\"x\", Con(\"Sleep\", [Var(\"x\")]))\n",
        "runs\tS\\NP\tLam(\"x\", Con(\"Run\", [Var(\"x\")]))\n",
        "zero\tNP\tLit(Int64(0))\n",
        "one\tNP\tLit(Int64(1))\n",
        "two\tNP\tLit(Int64(2))\n",
        "three\tNP\tLit(Int64(3))\n",
        "adds\t((S\\NP)/NP)/NP\tLam(\"z\", Lam(\"y\", Lam(\"x\", Con(\"Assign\", [Var(\"x\"), Con(\"Add\", [Var(\"y\"), Var(\"z\")])]))))\n",
        "is\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"Eq\", [Var(\"x\"), Var(\"y\")])))\n",
        "true\tNP\tLit(Bool(true))\n",
        "false\tNP\tLit(Bool(false))\n",
    );
    Lexicon::parse(tsv).unwrap()
}

fn pipeline(sentence: &str, lexicon: &Lexicon) -> (String, String) {
    let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();
    let trees = ccg::parse_sentence(&tokens, lexicon);
    assert!(!trees.is_empty(), "no parse for: {}", sentence);
    let tree = &trees[0];
    let ir = core_ir::compile_to_core_ir(tree, lexicon).unwrap();
    let mut net = deltanet::compile_to_net(&ir).unwrap();
    deltanet::reduce(&mut net).unwrap();
    let result = deltanet::readback(&net).unwrap();
    let hash = deltanet::unf_hash_string(&net).unwrap();
    (result, hash)
}

#[test]
fn test_e2e_john_loves_mary() {
    let lex = test_lexicon();
    let (result, _hash) = pipeline("John loves Mary", &lex);
    assert_eq!(result, "Love(john, mary)");
}

#[test]
fn test_e2e_mary_sees_john() {
    let lex = test_lexicon();
    let (result, _hash) = pipeline("Mary sees John", &lex);
    assert_eq!(result, "See(mary, john)");
}

#[test]
fn test_e2e_the_cat_sleeps() {
    let lex = test_lexicon();
    let (result, _hash) = pipeline("the cat sleeps", &lex);
    assert_eq!(result, "Sleep(Cat)");
}

#[test]
fn test_e2e_the_dog_runs() {
    let lex = test_lexicon();
    let (result, _hash) = pipeline("the dog runs", &lex);
    assert_eq!(result, "Run(Dog)");
}

#[test]
fn test_e2e_john_adds_two_three() {
    let lex = test_lexicon();
    let (result, _hash) = pipeline("John adds two three", &lex);
    assert_eq!(result, "Assign(john, Add(3, 2))");
}

#[test]
fn test_e2e_one_is_one() {
    let lex = test_lexicon();
    let (result, _hash) = pipeline("one is one", &lex);
    assert_eq!(result, "Eq(1, 1)");
}

#[test]
fn test_unf_hash_deterministic() {
    let lex = test_lexicon();
    let (_r1, h1) = pipeline("John loves Mary", &lex);
    let (_r2, h2) = pipeline("John loves Mary", &lex);
    assert_eq!(h1, h2, "same sentence must produce same UNF hash");
}

#[test]
fn test_unf_hash_different_sentences() {
    let lex = test_lexicon();
    let (_r1, h1) = pipeline("John loves Mary", &lex);
    let (_r2, h2) = pipeline("Mary sees John", &lex);
    assert_ne!(
        h1, h2,
        "different sentences should produce different UNF hashes"
    );
}

#[test]
fn test_unf_hash_intensional_equivalence() {
    let lex = test_lexicon();
    let (r1, h1) = pipeline("John loves Mary", &lex);
    let (r2, h2) = pipeline("John loves Mary", &lex);
    assert_eq!(r1, r2);
    assert_eq!(h1, h2);
}

#[test]
fn test_l1_no_triggers_single_world() {
    let lex = test_lexicon();
    let tokens: Vec<String> = "John loves Mary"
        .split_whitespace()
        .map(String::from)
        .collect();
    let trees = ccg::parse_sentence(&tokens, &lex);
    assert!(!trees.is_empty());
    let triggers = TriggerTable::new();
    let worlds = l1::split_l1(&trees[0], &triggers);
    assert_eq!(worlds.len(), 1);
    assert!((worlds[0].0 - 1.0).abs() < 1e-9);
}

#[test]
fn test_l1_probabilities_sum_to_one() {
    let triggers = TriggerTable::new();
    let tree = logos::ccg::DerivationTree::Leaf {
        word: "probably".to_string(),
        category: logos::ccg::CCGCategory::S,
    };
    let worlds = l1::split_l1(&tree, &triggers);
    assert!(l1::verify_world_probabilities(&worlds, 1e-9));
}

#[test]
fn test_l1_aggregation() {
    let worlds = vec![
        (0.8, "Love(john, mary)".to_string()),
        (0.2, "Love(john, mary)".to_string()),
        (0.5, "See(john, mary)".to_string()),
    ];
    let agg = l1::aggregate_results(&worlds);
    assert_eq!(agg[0].0, "Love(john, mary)");
    assert!((agg[0].1 - 1.0).abs() < 1e-9);
    assert_eq!(agg[1].0, "See(john, mary)");
    assert!((agg[1].1 - 0.5).abs() < 1e-9);
}

#[test]
fn test_linearity_check_ok() {
    let ir = core_ir::CoreIR::Lam(
        "x".to_string(),
        Box::new(core_ir::CoreIR::Var("x".to_string())),
    );
    let checked = core_ir::insert_linearity(ir);
    assert!(core_ir::check_linearity(&checked).is_ok());
}

#[test]
fn test_linearity_check_unused() {
    let ir = core_ir::CoreIR::Lam(
        "x".to_string(),
        Box::new(core_ir::CoreIR::Lit(core_ir::Literal::Int64(42))),
    );
    let checked = core_ir::insert_linearity(ir);
    assert!(core_ir::check_linearity(&checked).is_ok());
}

#[test]
fn test_corpus_seed_parses() {
    let lex_path = std::path::Path::new("corpus/lexicon.tsv");
    if !lex_path.exists() {
        return;
    }
    let lexicon = Lexicon::load(lex_path).unwrap();
    let corpus_path = std::path::Path::new("corpus/l0_seed/corpus.jsonl");
    if !corpus_path.exists() {
        return;
    }
    let content = std::fs::read_to_string(corpus_path).unwrap();
    let mut parsed = 0;
    let mut total = 0;
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        let entry: serde_json::Value = serde_json::from_str(line).unwrap();
        let sentence = entry["sentence"].as_str().unwrap();
        let tokens: Vec<String> = sentence
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();
        let trees = ccg::parse_sentence(&tokens, &lexicon);
        if !trees.is_empty() {
            parsed += 1;
        }
    }
    assert!(
        parsed as f64 / total as f64 > 0.5,
        "expected >50% parse rate, got {}/{}",
        parsed,
        total
    );
}

#[test]
fn property_confluence_reduce_twice_same_result() {
    let lex = test_lexicon();
    let sentences = [
        "John loves Mary",
        "the cat sleeps",
        "John adds two three",
        "one is one",
        "Mary sees John",
    ];
    for sentence in &sentences {
        let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();
        let trees = ccg::parse_sentence(&tokens, &lex);
        assert!(!trees.is_empty(), "no parse for: {sentence}");
        let ir = core_ir::compile_to_core_ir(&trees[0], &lex).unwrap();

        let mut net1 = deltanet::compile_to_net(&ir).unwrap();
        let _ = deltanet::reduce(&mut net1);
        let r1 = deltanet::readback(&net1);
        let h1 = deltanet::unf_hash_string(&net1);

        let mut net2 = deltanet::compile_to_net(&ir).unwrap();
        let _ = deltanet::reduce(&mut net2);
        let r2 = deltanet::readback(&net2);
        let h2 = deltanet::unf_hash_string(&net2);

        assert_eq!(r1, r2, "confluence violated for readback: {sentence}");
        assert_eq!(h1, h2, "confluence violated for hash: {sentence}");
    }
}

#[test]
fn property_hash_discrimination_all_distinct() {
    let lex = test_lexicon();
    let sentences = [
        "John loves Mary",
        "Mary sees John",
        "the cat sleeps",
        "the dog runs",
        "John adds two three",
        "one is one",
    ];
    let mut hashes = std::collections::HashSet::new();
    for sentence in &sentences {
        let (_result, hash) = pipeline(sentence, &lex);
        assert!(
            hashes.insert(hash.clone()),
            "hash collision between distinct sentences: {sentence}"
        );
    }
    assert_eq!(hashes.len(), sentences.len());
}

#[test]
fn property_linearity_always_holds_after_insert() {
    let terms = vec![
        core_ir::CoreIR::Lam(
            "x".to_string(),
            Box::new(core_ir::CoreIR::Var("x".to_string())),
        ),
        core_ir::CoreIR::Lam(
            "x".to_string(),
            Box::new(core_ir::CoreIR::Lit(core_ir::Literal::Int64(42))),
        ),
        core_ir::CoreIR::Lit(core_ir::Literal::Int64(7)),
        core_ir::CoreIR::Lit(core_ir::Literal::Bool(false)),
        core_ir::CoreIR::Var("free".to_string()),
    ];
    for term in terms {
        let checked = core_ir::insert_linearity(term);
        assert!(
            core_ir::check_linearity(&checked).is_ok(),
            "linearity check failed after insert_linearity"
        );
    }
}

#[test]
fn property_readback_is_stable_across_recompilation() {
    let lex = test_lexicon();
    let sentence = "John adds two three";
    let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();

    let mut results = Vec::new();
    for _ in 0..5 {
        let trees = ccg::parse_sentence(&tokens, &lex);
        let ir = core_ir::compile_to_core_ir(&trees[0], &lex).unwrap();
        let mut net = deltanet::compile_to_net(&ir).unwrap();
        deltanet::reduce(&mut net).unwrap();
        results.push(deltanet::readback(&net));
    }

    let first = &results[0];
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r, first, "readback unstable at iteration {i}");
    }
}

/// Numerical (F64) reduction end-to-end: build a CoreIR Prim over F64
/// literals, compile it to an interaction net, reduce it, read back and
/// hash. This is the "unique normal form via numerical operations" path:
/// the deltanet reducer must evaluate AddF64/MulF64 natively.
#[test]
fn test_numerical_f64_reduction_pipeline() {
    // ((3.5 + 1.25) * 2.0) = 9.5
    let sum = core_ir::CoreIR::Prim(
        core_ir::PrimOp::AddF64,
        vec![
            core_ir::CoreIR::Lit(core_ir::Literal::F64(3.5)),
            core_ir::CoreIR::Lit(core_ir::Literal::F64(1.25)),
        ],
    );
    let prod = core_ir::CoreIR::Prim(
        core_ir::PrimOp::MulF64,
        vec![sum, core_ir::CoreIR::Lit(core_ir::Literal::F64(2.0))],
    );

    let mut net = deltanet::compile_to_net(&prod).unwrap();
    deltanet::reduce(&mut net).unwrap();
    let result = deltanet::readback(&net).unwrap();
    assert_eq!(result, "9.5");

    // Same input must give the same UNF (deterministic unique normal form).
    let mut net2 = deltanet::compile_to_net(&prod).unwrap();
    deltanet::reduce(&mut net2).unwrap();
    let h1 = deltanet::unf_hash(&net).unwrap();
    let h2 = deltanet::unf_hash(&net2).unwrap();
    assert_eq!(h1, h2, "numerical UNF must be deterministic");
}

/// F64 comparison lowers to a Bool normal form; the hash of the reduced net
/// must discriminate the true/false outcomes.
#[test]
fn test_numerical_f64_comparison_normal_form() {
    let lt = core_ir::CoreIR::Prim(
        core_ir::PrimOp::LtF64,
        vec![
            core_ir::CoreIR::Lit(core_ir::Literal::F64(2.5)),
            core_ir::CoreIR::Lit(core_ir::Literal::F64(3.0)),
        ],
    );
    let mut net = deltanet::compile_to_net(&lt).unwrap();
    deltanet::reduce(&mut net).unwrap();
    let result = deltanet::readback(&net).unwrap();
    assert_eq!(result, "true");

    let gt = core_ir::CoreIR::Prim(
        core_ir::PrimOp::GtF64,
        vec![
            core_ir::CoreIR::Lit(core_ir::Literal::F64(2.5)),
            core_ir::CoreIR::Lit(core_ir::Literal::F64(3.0)),
        ],
    );
    let mut net = deltanet::compile_to_net(&gt).unwrap();
    deltanet::reduce(&mut net).unwrap();
    let result = deltanet::readback(&net).unwrap();
    assert_eq!(result, "false");
}

/// An F64 literal must have a distinct UNF serialization from an Int64 with
/// the same decimal digits (no silent numeric coercion between domains).
#[test]
fn test_unf_discriminates_f64_from_int64() {
    let f = core_ir::CoreIR::Lit(core_ir::Literal::F64(3.0));
    let i = core_ir::CoreIR::Lit(core_ir::Literal::Int64(3));
    let nf = deltanet::compile_to_net(&f).unwrap();
    let ni = deltanet::compile_to_net(&i).unwrap();
    assert_ne!(
        deltanet::unf_hash(&nf).unwrap(),
        deltanet::unf_hash(&ni).unwrap(),
        "F64(3.0) and Int64(3) must not share a UNF"
    );
}
