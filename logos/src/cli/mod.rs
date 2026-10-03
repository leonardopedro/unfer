use crate::deltanet;
use crate::deltanet::unf_hash_string;
use crate::l1::{self, TriggerTable};
use crate::lexicon::Lexicon;
use std::process;

pub mod formalize;

fn compile_or_die(tree: &crate::ccg::DerivationTree, lexicon: &Lexicon) -> crate::core_ir::CoreIR {
    match crate::core_ir::compile_to_core_ir(tree, lexicon) {
        Ok(ir) => ir,
        Err(msg) => {
            eprintln!("Compile error: {}", msg);
            process::exit(1);
        }
    }
}

fn compile_to_reduced_net(ir: &crate::core_ir::CoreIR) -> deltanet::Net {
    let mut net = match deltanet::compile_to_net(ir) {
        Ok(net) => net,
        Err(msg) => {
            eprintln!("Compile error: {}", msg);
            process::exit(1);
        }
    };
    deltanet::reduce(&mut net).unwrap_or_else(|msg| {
        eprintln!("Reduction error: {}", msg);
        process::exit(1);
    });
    net
}

fn readback_or_die(net: &deltanet::Net) -> String {
    match deltanet::readback(net) {
        Ok(result) => result,
        Err(msg) => {
            eprintln!("Readback error: {}", msg);
            process::exit(1);
        }
    }
}

pub fn run_cli(args: Vec<String>) {
    if args.len() < 2 {
        eprintln!("Usage: logos <subcommand> [args]");
        eprintln!("Subcommands: parse, run, verify, hash, l1, unf, formalize");
        process::exit(1);
    }

    match args[1].as_str() {
        "parse" => cmd_parse(&args[2..]),
        "run" => cmd_run(&args[2..]),
        "verify" => cmd_verify(&args[2..]),
        "hash" => cmd_hash(&args[2..]),
        "l1" => cmd_l1(&args[2..]),
        "unf" => cmd_unf(&args[2..]),
        "formalize" => process::exit(formalize::run(&args[2..])),
        _ => {
            eprintln!("Unknown subcommand: {}", args[1]);
            eprintln!("Subcommands: parse, run, verify, hash, l1, unf, formalize");
            process::exit(1);
        }
    }
}

fn cmd_parse(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: logos parse <sentence>");
        process::exit(1);
    }
    let sentence = args.join(" ");
    let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();

    let lex_path = find_lexicon();
    let lexicon = Lexicon::load(&lex_path).expect("failed to load lexicon");

    let trees = crate::ccg::parse_sentence(&tokens, &lexicon);
    if trees.is_empty() {
        eprintln!("No parse found for: {}", sentence);
        process::exit(1);
    }

    println!("Found {} parse(s):", trees.len());
    for (i, tree) in trees.iter().enumerate() {
        println!("  Parse {}: {}", i + 1, tree_to_string(tree));
        match crate::core_ir::compile_to_core_ir(tree, &lexicon) {
            Ok(ir) => println!("    Core IR: {}", ir),
            Err(msg) => {
                eprintln!("    Compile error: {}", msg);
                process::exit(1);
            }
        }
    }
}

fn cmd_run(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: logos run <sentence>");
        process::exit(1);
    }
    let sentence = args.join(" ");
    let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();

    let lex_path = find_lexicon();
    let lexicon = Lexicon::load(&lex_path).expect("failed to load lexicon");

    let trees = crate::ccg::parse_sentence(&tokens, &lexicon);
    if trees.is_empty() {
        eprintln!("No parse found for: {}", sentence);
        process::exit(1);
    }

    let tree = &trees[0];
    let ir = compile_or_die(tree, &lexicon);
    let result = readback_or_die(&compile_to_reduced_net(&ir));
    println!("{}", result);
}

fn cmd_verify(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: logos verify <sentence> <expected>");
        process::exit(1);
    }
    let sentence = args[0].clone();
    let expected = args[1].clone();
    let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();

    let lex_path = find_lexicon();
    let lexicon = Lexicon::load(&lex_path).expect("failed to load lexicon");

    let trees = crate::ccg::parse_sentence(&tokens, &lexicon);
    if trees.is_empty() {
        eprintln!("FAIL: No parse found");
        process::exit(1);
    }

    let tree = &trees[0];
    let ir = compile_or_die(tree, &lexicon);
    let net = compile_to_reduced_net(&ir);
    let result = readback_or_die(&net);

    if result == expected {
        println!("PASS: {} → {}", sentence, result);
    } else {
        eprintln!("FAIL: expected '{}', got '{}'", expected, result);
        process::exit(1);
    }
}

fn cmd_hash(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: logos hash <sentence>");
        process::exit(1);
    }
    let sentence = args.join(" ");
    let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();

    let lex_path = find_lexicon();
    let lexicon = Lexicon::load(&lex_path).expect("failed to load lexicon");

    let trees = crate::ccg::parse_sentence(&tokens, &lexicon);
    if trees.is_empty() {
        eprintln!("No parse found for: {}", sentence);
        process::exit(1);
    }

    let tree = &trees[0];
    let ir = compile_or_die(tree, &lexicon);
    let net = compile_to_reduced_net(&ir);
    let hash = unf_hash_string(&net).unwrap_or_else(|msg| {
        eprintln!("Hash error: {}", msg);
        process::exit(1);
    });
    println!("{}", hash);
}

fn cmd_l1(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: logos l1 <sentence>");
        process::exit(1);
    }
    let sentence = args.join(" ");
    let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();

    let lex_path = find_lexicon();
    let lexicon = Lexicon::load(&lex_path).expect("failed to load lexicon");

    let trees = crate::ccg::parse_sentence(&tokens, &lexicon);
    if trees.is_empty() {
        eprintln!("No parse found for: {}", sentence);
        process::exit(1);
    }

    let tree = &trees[0];
    let triggers = TriggerTable::new();
    let worlds = l1::split_l1(tree, &triggers);

    println!("{} worlds:", worlds.len());
    for (prob, world_tree) in &worlds {
        let ir = compile_or_die(world_tree, &lexicon);
        let net = compile_to_reduced_net(&ir);
        let result = readback_or_die(&net);
        let hash = unf_hash_string(&net).unwrap_or_else(|msg| {
            eprintln!("Hash error: {}", msg);
            process::exit(1);
        });
        println!("  p={:.4} result={} hash={}", prob, result, hash);
    }
}

/// `logos unf <sentence> [--json]` — one CNL sentence to its unique normal form.
///
/// This is the kernel surface the Austral VM plugin calls
/// (`australVM/lib/formalize_plugin.ml`, P5 of the rewrite plan): the
/// `uk_logos_compile` protocol op is one call into the kernel process, which is
/// not reachable from an OCaml compiler plugin, so the same operation is exposed
/// as a subprocess whose stdout is a `LogosReport`-shaped object.
///
/// It prints `prob_kernel::logos::logos_compile`'s four fields — `result`,
/// `unf_hash`, `verified`, `sentence` — and exits non-zero on any failure, so a
/// caller can distinguish "did not reduce" from "ran and disagreed" without
/// parsing prose.
/// The body of [`cmd_unf`], without the `process::exit`.
///
/// Returns the line to print and the exit code. Split out because every other
/// subcommand here exits on failure, and this one is the surface an external
/// plugin parses — so its output and its exit codes are a contract, and a
/// contract should be testable without a subprocess.
fn unf_report(
    sentence: &str,
    lexicon: &Lexicon,
    json: bool,
) -> Result<(String, i32), (String, i32)> {
    match crate::formalize::formalizer::verify_cnl(sentence, lexicon) {
        Ok(r) => {
            let line = if json {
                // `LogosReport`'s shape, so a caller can parse this with the same
                // schema the kernel protocol op uses.
                let report = serde_json::json!({
                    "result": r.readback,
                    "unf_hash": r.unf_hash,
                    "verified": r.verified,
                    "sentence": r.cnl,
                });
                serde_json::to_string(&report).expect("a report of four strings serializes")
            } else {
                // Tab-separated rather than space-separated: `result` contains
                // spaces (`Love(john, mary)`), so a caller splitting on
                // whitespace would get the wrong fields.
                format!("{}\t{}\t{}", r.readback, r.unf_hash, r.verified)
            };
            // Compiled but not *uniquely* reduced: still a success as a
            // compilation, so code 2 rather than 1. The Austral plugin
            // distinguishes them — a 1 is "the kernel rejected this", a 2 is "the
            // kernel accepted it but cannot vouch for its identity".
            //
            // `--json` does not buy a pass here. The documented contract, three
            // lines up, is that this "exits non-zero on any failure, so a caller
            // can distinguish 'did not reduce' from 'ran and disagreed'"; folding `json` in
            // made code 2 unreachable for exactly the callers who asked for
            // machine-readable output and are best placed to act on it.
            Ok((line, if r.verified { 0 } else { 2 }))
        }
        Err(e) => Err((format!("error: {e}"), 1)),
    }
}

fn cmd_unf(args: &[String]) {
    let mut json = false;
    let mut lexicon_path: Option<std::path::PathBuf> = None;
    let mut words: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => json = true,
            "--lexicon" => {
                i += 1;
                match args.get(i) {
                    Some(p) => lexicon_path = Some(std::path::PathBuf::from(p)),
                    None => {
                        eprintln!("error: --lexicon needs a value");
                        process::exit(1);
                    }
                }
            }
            "-h" | "--help" => {
                println!("Usage: logos unf <sentence> [--json] [--lexicon <tsv>]");
                return;
            }
            other => words.push(other.to_string()),
        }
        i += 1;
    }

    if words.is_empty() {
        eprintln!("Usage: logos unf <sentence> [--json] [--lexicon <tsv>]");
        process::exit(1);
    }
    let sentence = words.join(" ");

    // The embedded lexicon by default, so this works from any directory — the
    // plugin runs with whatever CWD the compiler has. `find_lexicon`'s
    // CWD-relative search would only find it inside an unfer checkout.
    let lex_path = lexicon_path.unwrap_or_else(|| {
        let p = std::path::PathBuf::from("corpus/lexicon.tsv");
        if p.exists() {
            p
        } else {
            std::path::PathBuf::new()
        }
    });
    let lexicon = if lex_path.as_os_str().is_empty() {
        crate::formalize::formalizer::base_lexicon()
    } else {
        match Lexicon::load(&lex_path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error: {}: {e}", lex_path.display());
                process::exit(1);
            }
        }
    };

    match unf_report(&sentence, &lexicon, json) {
        Ok((line, 0)) => println!("{line}"),
        Ok((line, code)) => {
            println!("{line}");
            process::exit(code);
        }
        Err((msg, code)) => {
            eprintln!("{msg}");
            process::exit(code);
        }
    }
}

fn find_lexicon() -> std::path::PathBuf {
    let candidates = [
        "corpus/lexicon.tsv",
        "../corpus/lexicon.tsv",
        "../../corpus/lexicon.tsv",
    ];
    for c in &candidates {
        let p = std::path::Path::new(c);
        if p.exists() {
            return p.to_path_buf();
        }
    }
    eprintln!("Error: lexicon.tsv not found");
    process::exit(1);
}

fn tree_to_string(tree: &crate::ccg::DerivationTree) -> String {
    match tree {
        crate::ccg::DerivationTree::Leaf { word, category } => {
            format!("{}:{}", word, category)
        }
        crate::ccg::DerivationTree::Application {
            left,
            right,
            result_category,
            ..
        } => {
            format!(
                "({} {}):{}",
                tree_to_string(left),
                tree_to_string(right),
                result_category
            )
        }
        crate::ccg::DerivationTree::Composition {
            left,
            right,
            result_category,
            ..
        } => {
            format!(
                "({} >B {}):{}",
                tree_to_string(left),
                tree_to_string(right),
                result_category
            )
        }
    }
}

#[cfg(test)]
mod unf_tests {
    use super::unf_report;
    use crate::formalize::formalizer::base_lexicon;

    fn lex() -> crate::lexicon::Lexicon {
        base_lexicon()
    }

    /// The line an external plugin parses, so its shape is a contract.
    #[test]
    fn the_plain_line_is_tab_separated() {
        let (line, code) = unf_report("John loves Mary", &lex(), false).unwrap();
        assert_eq!(code, 0);
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 3, "{line}");
        // `result` contains spaces, which is exactly why the separator is a tab.
        assert_eq!(fields[0], "Love(john, mary)");
        assert_eq!(fields[1].len(), 64, "a hex sha256: {}", fields[1]);
        assert_eq!(fields[2], "true");
    }

    /// `--json` matches the kernel's `LogosReport`, so one parser serves both the
    /// in-process op and this subprocess.
    #[test]
    fn the_json_line_is_a_logos_report() {
        let (line, code) = unf_report("John loves Mary", &lex(), true).unwrap();
        assert_eq!(code, 0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["result"], "Love(john, mary)");
        assert_eq!(v["verified"], true);
        assert_eq!(v["sentence"], "John loves Mary");
        assert_eq!(v["unf_hash"].as_str().unwrap().len(), 64);
    }

    /// A failed reduction is code 1 with the reason on stderr — never a 0 with an
    /// empty body, which a caller would read as "no result but no problem".
    #[test]
    fn a_failed_reduction_is_exit_one_with_a_reason() {
        let (msg, code) = unf_report("Euler proves congruences", &lex(), true).unwrap_err();
        assert_eq!(code, 1);
        assert!(msg.starts_with("error:"), "{msg}");
        assert!(
            msg.contains("Euler"),
            "the offending words are named: {msg}"
        );
    }

    #[test]
    fn an_empty_sentence_is_rejected() {
        assert!(unf_report("", &lex(), false).is_err());
    }
}
