use logos::ccg;
use logos::core_ir;
use logos::deltanet;
use logos::lexicon::Lexicon;
use logos::l1::{self, TriggerTable};

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
    let ir = core_ir::compile_to_core_ir(tree, lexicon);
    let mut net = deltanet::compile_to_net(&ir);
    deltanet::reduce(&mut net);
    let result = deltanet::readback(&net);
    let hash = deltanet::unf_hash_string(&net);
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
    assert_ne!(h1, h2, "different sentences should produce different UNF hashes");
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
    let tokens: Vec<String> = "John loves Mary".split_whitespace().map(String::from).collect();
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
