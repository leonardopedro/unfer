use crate::deltanet;
use crate::deltanet::unf_hash_string;
use crate::l1::{self, TriggerTable};
use crate::lexicon::Lexicon;
use std::process;

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
        eprintln!("Subcommands: parse, run, verify, hash, l1");
        process::exit(1);
    }

    match args[1].as_str() {
        "parse" => cmd_parse(&args[2..]),
        "run" => cmd_run(&args[2..]),
        "verify" => cmd_verify(&args[2..]),
        "hash" => cmd_hash(&args[2..]),
        "l1" => cmd_l1(&args[2..]),
        _ => {
            eprintln!("Unknown subcommand: {}", args[1]);
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
