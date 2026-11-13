//! `logos mm0` — the math fragment of the CNL, certified by MM0.
//!
//! The pipeline is the project's existing one, plus a target theory:
//!
//! ```text
//! sentence ──cnl──► Claim ──core_ir/deltanet──► Net ──reduce──► value
//!                     │                                        │
//!                     └──────────proof/emit────────────────────┘
//!                                     │
//!                              logos_math.mm0 + .mm1 ──► mm0-rs ──► mm0-c
//! ```
//!
//! * [`cnl`] is the surface syntax: a subset whose sentences have one reading.
//! * [`bridge`] runs the claim through `logos`' real lowering (CoreIR → DeltaNet
//!   → reducer) and uses the result to corroborate the number the certificate is
//!   built for, and to produce the isomorphism witness when two reduced nets are
//!   the same net.
//! * [`proof`] encodes expressions as the net they compile to and builds the
//!   MM0 proof terms.
//! * [`emit`] assembles the certificate (`mm0/logos_math.mm0` statements + `.mm1`
//!   proofs) and runs the tools.
//!
//! The answer to a question is the *shape* of the certificate: proved
//! `teq L R` means yes, `teq L R ⟹ tfals` means no, and no certificate means the
//! pipeline does not know — never a guess. A quantified claim is closed by
//! rewriting under the recursion laws when they reach both sides, and otherwise
//! only by a candidate proof term that mm0's elaborator accepts (`--hint`,
//! `--llm`): a proposal enters the artifact through mm0 or not at all.

pub mod bridge;
pub mod cnl;
pub mod dk;
pub mod emit;
pub mod kernel;
pub mod kop;
pub mod proof;

pub use cnl::Claim;
pub use emit::{Certificate, Verdict};
pub use kop::KClaim;
pub use proof::Expr;

use std::path::{Path, PathBuf};

/// Exit codes, the same contract `logos unf`/`logos iso` already use: 0 for yes,
/// 2 for no, 1 for "did not run / could not decide".
const YES: i32 = 0;
const NO: i32 = 2;
const UNKNOWN: i32 = 1;

pub fn run(args: &[String]) -> i32 {
    let Some(command) = args.first().map(String::as_str) else {
        eprintln!("{USAGE}");
        return UNKNOWN;
    };
    match command {
        "theory" => cmd_theory(&args[1..]),
        "ask" => cmd_ask(&args[1..]),
        "claims" => cmd_claims(&args[1..]),
        "check" => cmd_check(&args[1..]),
        "dk" => cmd_dk(&args[1..]),
        other => {
            eprintln!("unknown `logos mm0` command: {other}\n\n{USAGE}");
            UNKNOWN
        }
    }
}

const USAGE: &str = "\
usage:
  logos mm0 theory [--out FILE] [--kopperman] [--mehler]  dump the target theory
  logos mm0 ask <sentence> [options]            certify one sentence
  logos mm0 claims <FILE> [options]             certify a fixture, check expectations
  logos mm0 check <FILE.mm0 | DIR>              run mm0-rs and mm0-c on artifacts
  logos mm0 dk <FILE.dk> [--out DIR]           translate a Dedukti/Lambdapi file, then
                                                compile it with mm0-rs and re-check it
                                                with mm0-c (always: that is the point)

ask options:
  --out DIR          write <theorem>.mm0 / <theorem>.mm1
  --verify           compile and re-check the artifact with mm0-rs + mm0-c
  --json             one JSON object instead of prose
  --hint TERM        try TERM (an mm0 proof term) for an open claim; repeatable
  --llm              ask the LLM for hint terms (needs the `llm-http` feature)
  --mehler           use the Kopperman/Mehler tail-determination theory
                     (theory: mehler sections in a claims file do the same)
  --kernel-certificate FILE.mm0
                     decide `theory: kernel` claim lines by citing that kernel
                     certificate (--kernel-obligations FILE.json overrides the
                     sidecar its `-- Source:` header pins)

exit codes: 0 yes (true / proved), 2 no (refuted), 1 unproved or error";

/// `logos mm0 dk <FILE.dk>`: translate a Dedukti/Lambdapi presentation into the
/// target theory and have mm0 judge it.
///
/// Verification is not behind a flag here. A front end is only worth something
/// if its output is checked, so the artifacts are always compiled by `mm0-rs`
/// and re-checked by `mm0-c`, and a rejection is an error rather than a note —
/// the same contract `scripts/logos2mm0` holds the CNL path to.
fn cmd_dk(args: &[String]) -> i32 {
    let file = match args.first().filter(|a| !a.starts_with('-')) {
        Some(f) => f.clone(),
        None => {
            eprintln!("usage: logos mm0 dk <FILE.dk> [--out DIR]\n\n{USAGE}");
            return UNKNOWN;
        }
    };
    let out = flag_value(args, "--out");
    match dk::run(Path::new(&file), out.as_deref().map(Path::new)) {
        Ok((translated, verified)) => {
            println!("dk:      {file}");
            println!(
                "rules:   {} re-proved in the theory (each one is its own mm0 theorem)",
                translated.rules.len()
            );
            for rule in &translated.rules {
                println!("  {:<12} {}  = '{} ", rule.name, rule.statement, rule.proof);
            }
            println!("thm:     {} translated", translated.theorems.len());
            for thm in &translated.theorems {
                println!(
                    "  {:<20} {}  = '{} ",
                    thm.name, thm.statement, thm.proof
                );
            }
            println!(
                "mm0:     ok{} ({})",
                verified
                    .sorts
                    .as_ref()
                    .map(|s| format!(", {s}"))
                    .unwrap_or_default(),
                verified.mmb.display()
            );
            match verified.mm0_c_exit {
                Some(code) => println!("mm0-c:   exit {code} (independent checker over the .mmb)"),
                None => println!("mm0-c:   not found; the elaborator's verdict stands alone"),
            }
            YES
        }
        Err(e) => {
            eprintln!("{e}");
            UNKNOWN
        }
    }
}

fn cmd_theory(args: &[String]) -> i32 {
    // `--kopperman` dumps the full Hilbert-space target theory (the agent
    // layer plus the overlay); `--mehler` adds the tail-determination
    // extension on top of it; the default stays the arithmetic one.
    let theory = if has_flag(args, "--mehler") {
        kop::Theory::Mehler.text()
    } else if has_flag(args, "--kopperman") {
        kop::Theory::Base.text()
    } else {
        emit::THEORY
    };
    let out = flag_value(args, "--out");
    match out {
        Some(path) => match std::fs::write(&path, theory) {
            Ok(()) => {
                println!("{}", path);
                YES
            }
            Err(e) => {
                eprintln!("{path}: {e}");
                UNKNOWN
            }
        },
        None => {
            print!("{theory}");
            YES
        }
    }
}

/// A parsed sentence: the arithmetic fragment's claim, or the Kopperman
/// fragment's. The arithmetic parser is tried first — its sentences are a
/// subset of what the Kopperman parser accepts, and keeping the old path
/// first keeps every existing fixture reading exactly as before.
enum Parsed {
    Arith(Claim),
    Kop(KClaim),
}

/// Parse a sentence in either math fragment.
///
/// When neither accepts it, the diagnostic has to come from *some* parser,
/// and the two disagree about what is wrong: a sentence with the Kopperman
/// vocabulary gets the Kopperman parser's complaint (the arithmetic one
/// would only say where it stopped, which for `the inner product of …` is
/// always at the first word). Anything else keeps the arithmetic one, whose
/// diagnostics the existing tests pin.
fn parse_claim(text: &str) -> Result<Parsed, String> {
    match Claim::parse(text) {
        Ok(c) => Ok(Parsed::Arith(c)),
        Err(arith_err) => match KClaim::parse(text) {
            Ok(k) => Ok(Parsed::Kop(k)),
            Err(kop_err) => Err(if mentions_kopperman(text) {
                kop_err
            } else {
                arith_err
            }),
        },
    }
}

/// Whether the sentence speaks the Kopperman vocabulary — used only to pick
/// which parser's diagnostic to report when neither accepts the sentence.
fn mentions_kopperman(text: &str) -> bool {
    [
        "inner product",
        "vector",
        "coordinate",
        "the sum of",
        "the scalar",
    ]
    .iter()
    .any(|k| text.contains(k))
}

fn cmd_ask(args: &[String]) -> i32 {
    let text = words(args);
    if text.is_empty() {
        eprintln!("usage: logos mm0 ask <sentence>\n\n{USAGE}");
        return UNKNOWN;
    }

    let mut hints = flag_values(args, "--hint");
    let cert = match parse_claim(&text) {
        Ok(Parsed::Arith(claim)) => {
            if has_flag(args, "--llm") {
                match llm_hints(&emit::goal_of(&claim)) {
                    Ok(more) => hints.extend(more),
                    Err(e) => eprintln!("llm: {e}"),
                }
            }
            emit::certify(&claim, &hints)
        }
        Ok(Parsed::Kop(claim)) => {
            let theory = if has_flag(args, "--mehler") {
                kop::Theory::Mehler
            } else {
                kop::Theory::Base
            };
            if has_flag(args, "--llm") {
                match llm_hints_with(theory.llm_system(), &claim.goal_of()) {
                    Ok(more) => hints.extend(more),
                    Err(e) => eprintln!("llm: {e}"),
                }
            }
            kop::certify_in(&claim, &hints, theory)
        }
        Err(e) => {
            eprintln!("{e}");
            return UNKNOWN;
        }
    };
    let cert = match cert {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return UNKNOWN;
        }
    };

    let mut verified = None;
    if let Some(dir) = flag_value(args, "--out") {
        match write_certificate(&dir, &cert) {
            Ok(paths) => {
                if let (true, Some((mm0, _))) = (has_flag(args, "--verify"), &paths) {
                    match emit::verify_mm0(mm0) {
                        Ok(v) => verified = Some(v),
                        Err(e) => {
                            eprintln!("{e}");
                            return UNKNOWN;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("{e}");
                return UNKNOWN;
            }
        }
    } else if has_flag(args, "--verify") {
        eprintln!("--verify needs --out: there is nothing on disk to check");
        return UNKNOWN;
    }

    if has_flag(args, "--json") {
        let mut json = cert.to_json();
        if let Some(v) = &verified {
            json["verified"] = serde_json::json!({
                "mmb": v.mmb.display().to_string(),
                "sorts": v.sorts,
                "mm0_c": v.mm0_c_exit,
            });
        }
        println!("{json}");
    } else {
        report(&cert, verified.as_ref());
    }

    match cert.verdict {
        Verdict::True | Verdict::Proved => YES,
        Verdict::False => NO,
        Verdict::Unproved(_) => UNKNOWN,
    }
}

fn report(cert: &Certificate, verified: Option<&emit::Verified>) {
    let answer = if cert.question { "answer" } else { "claim" };
    println!("{answer}:  {}", cert.claim);
    match &cert.verdict {
        Verdict::True => println!("verdict: true  (teq L R is provable)"),
        Verdict::False => println!("verdict: false (teq L R implies tfals)"),
        Verdict::Proved => println!("verdict: proved (for all instantiations)"),
        Verdict::Unproved(why) => println!("verdict: unproved\n  {why}"),
    }
    if !cert.proof.is_empty() {
        println!("goal:    $ {} $", cert.goal);
        println!("theorem: {}", cert.name);
    }
    if let Some(pairs) = &cert.net_witness {
        println!(
            "nets:    same reduced net ({} nodes, isomorphism witness checked)",
            pairs.len()
        );
    }
    if let Some(v) = verified {
        println!(
            "mm0:     ok{} ({})",
            v.sorts
                .as_ref()
                .map(|s| format!(", {s}"))
                .unwrap_or_default(),
            v.mmb.display()
        );
    } else if matches!(
        cert.verdict,
        Verdict::True | Verdict::False | Verdict::Proved
    ) {
        println!("mm0:     not run (pass --verify to compile and re-check)");
    }
}

/// Write `<dir>/<theorem>.mm0` and `.mm1` — the pairing `mm0-rs` and `mm0-c`
/// expect — and return the paths, or `None` when there is no certificate.
///
/// An unproved claim deliberately gets no `.mm0`: a statement without a proof is
/// a `sorry`, `mm0-c` rejects those (correctly), and calling the file a
/// certificate would misrepresent it. It becomes `<theorem>.goal` instead —
/// extensionless, so `check` skips it — which is exactly the artifact an LLM (or
/// a person) needs to answer with `--hint`.
fn write_certificate(dir: &str, cert: &Certificate) -> Result<Option<(PathBuf, PathBuf)>, String> {
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if matches!(cert.verdict, Verdict::Unproved(_)) {
        let goal = dir.join(format!("{}.goal", cert.name));
        std::fs::write(&goal, &cert.mm0).map_err(|e| format!("{}: {e}", goal.display()))?;
        return Ok(None);
    }
    let stem = dir.join(&cert.name);
    let mm0 = stem.with_extension("mm0");
    let mm1 = stem.with_extension("mm1");
    std::fs::write(&mm0, &cert.mm0).map_err(|e| format!("{}: {e}", mm0.display()))?;
    std::fs::write(&mm1, &cert.mm1).map_err(|e| format!("{}: {e}", mm1.display()))?;
    Ok(Some((mm0, mm1)))
}

// ── the fixture runner ──────────────────────────────────────────────────────

/// Which fragment a `theory:` directive selected for the lines below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TheorySel {
    Base,
    Mehler,
    /// Sentences about a symbolic kernel certificate: what the `theory: kernel`
    /// lines in `claims.txt` state. Decided by citation, so they need a
    /// certificate to cite (`--kernel-certificate`), and are skipped with a note
    /// when there is none.
    Kernel,
}

/// One line of a claims file: `true: …`, `false: …`, `proved: …`, or a bare
/// sentence (which must not be refuted — the default expectation is "yes").
/// `theory` is the `theory:` directive in force at that line: `None` means
/// no directive has been seen (the `--mehler` flag decides), `Some(sel)` the
/// directive has fixed it.
struct Expected {
    line: usize,
    sentence: String,
    want: Option<Verdict>,
    theory: Option<TheorySel>,
}

fn parse_fixture(text: &str) -> Result<Vec<Expected>, String> {
    let mut out = Vec::new();
    let mut current: Option<TheorySel> = None;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (want, sentence, theory) = match line.split_once(':') {
            Some((tag, rest)) if matches!(tag.trim(), "true" | "false" | "proved" | "unproved") => {
                let want = match tag.trim() {
                    "true" => Verdict::True,
                    "false" => Verdict::False,
                    "unproved" => Verdict::Unproved(String::new()),
                    _ => Verdict::Proved,
                };
                (Some(want), rest.trim().to_string(), current)
            }
            // `theory: base` / `theory: mehler` / `theory: kernel` switches the
            // target theory for every claim below it — the fixture says which
            // theory each claim expects, so one file (and one gate run) covers
            // all three.
            Some((tag, rest)) if tag.trim() == "theory" => match rest.trim() {
                "base" => {
                    current = Some(TheorySel::Base);
                    continue;
                }
                "mehler" => {
                    current = Some(TheorySel::Mehler);
                    continue;
                }
                "kernel" => {
                    current = Some(TheorySel::Kernel);
                    continue;
                }
                other => {
                    return Err(format!(
                        "line {}: unknown theory `{other}` (base | mehler | kernel)",
                        i + 1
                    ));
                }
            },
            // `for all x: …` has a colon of its own, so a colon that is not one
            // of the tags is part of the sentence.
            _ => (None, line.to_string(), current),
        };
        out.push(Expected {
            line: i + 1,
            sentence,
            want,
            theory,
        });
    }
    Ok(out)
}

/// What one line of a corpus produced: a certificate of one of the proof-
/// producing fragments, or a citation from a kernel certificate.
///
/// The kernel case is deliberately not an `emit::Certificate`: it has no proof
/// term of its own to show, and dressing a citation up as one would say more
/// than the artifact does.
enum Outcome {
    Math(emit::Certificate),
    Kernel(kernel::Citation),
}

impl Outcome {
    fn verdict(&self) -> &Verdict {
        match self {
            Outcome::Math(c) => &c.verdict,
            Outcome::Kernel(c) => &c.verdict,
        }
    }

    fn json(&self) -> serde_json::Value {
        match self {
            Outcome::Math(c) => c.to_json(),
            Outcome::Kernel(c) => c.to_json(),
        }
    }
}

fn cmd_claims(args: &[String]) -> i32 {
    let Some(file) = args
        .iter()
        .find(|a| !a.starts_with("--") && !is_flag_value(args, a))
    else {
        eprintln!(
            "usage: logos mm0 claims <FILE> [--out DIR] [--verify] [--json] [--mehler]\n\
             \x20      [--kernel-certificate FILE.mm0 [--kernel-obligations FILE.json]]\n\n{USAGE}"
        );
        return UNKNOWN;
    };
    let text = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{file}: {e}");
            return UNKNOWN;
        }
    };
    let cases = match parse_fixture(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{file}: {e}");
            return UNKNOWN;
        }
    };

    let out_dir = flag_value(args, "--out");
    let verify = has_flag(args, "--verify");
    let as_json = has_flag(args, "--json");
    // The theory for claims no `theory:` directive fixes.
    let default_mehler = has_flag(args, "--mehler");

    // A kernel certificate, when the run was given one. Kernel claims are
    // decided by citing it, so without it they are skipped with a note rather
    // than failed: the corpus lives in this repo, the certificate lives in the
    // kernel's repo, and a checkout that has only the former is not wrong.
    let kernel_theory = match flag_value(args, "--kernel-certificate") {
        Some(path) => {
            let obl = flag_value(args, "--kernel-obligations");
            match kernel::Theory::load(Path::new(&path), obl.as_deref().map(Path::new)) {
                Ok(t) => Some(t),
                Err(e) => {
                    eprintln!("--kernel-certificate: {e}");
                    return UNKNOWN;
                }
            }
        }
        None => None,
    };
    let mut skipped_kernel = 0usize;

    let mut failures = 0usize;
    let mut results = Vec::new();
    for case in cases {
        let outcome = match case.theory {
            Some(TheorySel::Kernel) => match &kernel_theory {
                Some(t) => match t.decide(&case.sentence) {
                    Ok(c) => Outcome::Kernel(c),
                    Err(e) => {
                        eprintln!("line {}: {e}", case.line);
                        failures += 1;
                        continue;
                    }
                },
                None => {
                    skipped_kernel += 1;
                    continue;
                }
            },
            sel => {
                let cert = match parse_claim(&case.sentence) {
                    Ok(Parsed::Arith(claim)) => emit::certify(&claim, &[]),
                    Ok(Parsed::Kop(claim)) => {
                        let theory = match sel {
                            Some(TheorySel::Mehler) => kop::Theory::Mehler,
                            Some(TheorySel::Base) => kop::Theory::Base,
                            _ => {
                                if default_mehler {
                                    kop::Theory::Mehler
                                } else {
                                    kop::Theory::Base
                                }
                            }
                        };
                        kop::certify_in(&claim, &[], theory)
                    }
                    Err(e) => {
                        eprintln!("line {}: {e}", case.line);
                        failures += 1;
                        continue;
                    }
                };
                match cert {
                    Ok(c) => Outcome::Math(c),
                    Err(e) => {
                        eprintln!("line {}: {e}", case.line);
                        failures += 1;
                        continue;
                    }
                }
            }
        };
        let written = match (&out_dir, &outcome) {
            (Some(dir), Outcome::Math(cert)) => match write_certificate(dir, cert) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("line {}: {e}", case.line);
                    failures += 1;
                    continue;
                }
            },
            _ => None,
        };
        // Only a real certificate is checked: an unproved claim wrote a `.goal`
        // file, which is a request for a proof, not a proof.
        let verified = match (&written, verify) {
            (Some((mm0, _)), true) => match emit::verify_mm0(mm0) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("line {}: {e}", case.line);
                    failures += 1;
                    continue;
                }
            },
            _ => None,
        };

        let got = outcome.verdict().clone();
        let ok = match &case.want {
            Some(Verdict::True) => matches!(got, Verdict::True | Verdict::Proved),
            Some(Verdict::Proved) => matches!(got, Verdict::Proved),
            Some(Verdict::False) => matches!(got, Verdict::False),
            Some(Verdict::Unproved(_)) => matches!(got, Verdict::Unproved(_)),
            None => !matches!(got, Verdict::False | Verdict::Unproved(_)),
        };
        if !ok {
            failures += 1;
            eprintln!(
                "line {}: expected {}, got {}: {}",
                case.line,
                case.want.as_ref().map(|v| v.label()).unwrap_or("yes"),
                got.label(),
                case.sentence
            );
        }
        results.push((case.line, case.sentence, outcome, verified));
    }
    if skipped_kernel > 0 {
        eprintln!(
            "{skipped_kernel} kernel claim(s) skipped: pass --kernel-certificate \
             <mm0/<stem>_certificate.mm0> to decide them"
        );
    }

    if as_json {
        let rows: Vec<serde_json::Value> = results
            .iter()
            .map(|(line, sentence, outcome, verified)| {
                let mut v = outcome.json();
                v["line"] = serde_json::json!(line);
                v["sentence"] = serde_json::json!(sentence);
                v["verified"] = serde_json::json!(verified.is_some());
                v
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({ "claims": rows, "failures": failures })
        );
    } else {
        for (_, sentence, outcome, verified) in &results {
            let detail = match outcome {
                Outcome::Math(_) => {
                    if verified.is_some() {
                        "mm0 ok".to_string()
                    } else {
                        "-".to_string()
                    }
                }
                Outcome::Kernel(c) => match &c.cited {
                    Some(name) => format!("cited: {name}"),
                    None => "-".to_string(),
                },
            };
            println!(
                "{:<8} {}  [{}]",
                outcome.verdict().label(),
                sentence,
                detail
            );
            // A kernel sentence that is not proved says why: identities the
            // certificate does not state and sentences it cannot state are the
            // two things a reader of this corpus needs told apart.
            if let Outcome::Kernel(k) = outcome {
                if let Verdict::Unproved(why) = &k.verdict {
                    println!("         {why}");
                }
            }
        }
        println!(
            "\n{} claim(s), {failures} failure(s)",
            results.len() + failures
        );
    }

    if failures == 0 { YES } else { UNKNOWN }
}

fn cmd_check(args: &[String]) -> i32 {
    let Some(target) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!("usage: logos mm0 check <FILE.mm0 | DIR>\n\n{USAGE}");
        return UNKNOWN;
    };
    let path = PathBuf::from(target);
    let files: Vec<PathBuf> = if path.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&path)
            .map(|rd| {
                rd.filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "mm0"))
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    } else {
        vec![path]
    };
    if files.is_empty() {
        eprintln!("no .mm0 files in {target}");
        return UNKNOWN;
    }

    let mut failures = 0;
    for f in &files {
        match emit::verify_mm0(f) {
            Ok(v) => println!(
                "ok    {}{}",
                f.display(),
                v.sorts.map(|s| format!("  ({s})")).unwrap_or_default()
            ),
            Err(e) => {
                failures += 1;
                eprintln!("FAIL  {}\n{e}", f.display());
            }
        }
    }
    if failures == 0 { YES } else { UNKNOWN }
}

// ── argument plumbing ───────────────────────────────────────────────────────

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn flag_values(args: &[String], flag: &str) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter(|(_, a)| *a == flag)
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect()
}

/// Whether `arg` is the value of a preceding flag (so it is not the sentence).
fn is_flag_value(args: &[String], arg: &str) -> bool {
    args.iter()
        .position(|a| a == arg)
        .is_some_and(|i| i > 0 && args[i - 1].starts_with("--"))
}

/// The sentence: every argument that is not a flag or a flag's value.
fn words(args: &[String]) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
            continue;
        }
        if a.starts_with("--") {
            // Exactly the flags that take a value; everything else is a boolean.
            skip = matches!(a.as_str(), "--out" | "--hint");
            continue;
        }
        out.push(a);
    }
    out.join(" ")
}

// ── the LLM seam ────────────────────────────────────────────────────────────

/// Ask the model for candidate proof terms for an open goal.
///
/// The model is given the theory's rules by name and the goal, and asked for
/// mm0 proof terms (the `'(…)` bodies). Whatever comes back is *only* a
/// candidate list: `emit::certify_open` hands each one to mm0-rs, and a
/// hallucinated rule simply fails to elaborate. That is the intended division of
/// labour — the LLM proposes, MM0 disposes.
#[cfg(feature = "llm-http")]
fn llm_hints(goal: &str) -> Result<Vec<String>, String> {
    const SYSTEM: &str = "\
You are helping close a goal in the MM0 theory `logos_math`. \
Rules available: prim_add_denote, prim_mul_denote, eq_refl, eq_sym, eq_trans, eq_ts, \
eq_app, eq_add_l, eq_add_r, eq_mul_l, eq_mul_r, add_z, add_s, mul_z, mul_s, s_inj, \
z_ne_s, fals_elim. Terms: tz (zero), ts (successor), tadd, tmul, tapp, tprim_add, \
tprim_mul, teq. Reply with ONLY a JSON array of MM0 proof terms (strings), most \
promising first. A proof term looks like `(add_z)` or `(eq_trans (add_s) (eq_ts (add_z)))`.";
    llm_hints_with(SYSTEM, goal)
}

/// The LLM seam, parameterised by the theory's rule catalogue: the
/// arithmetic fragment and the Kopperman fragment offer different vocabular
/// to the model, but the contract is the same — whatever comes back is only
/// ever a candidate list, and mm0-rs is the judge (see [`llm_hints`]).
#[cfg(feature = "llm-http")]
fn llm_hints_with(system: &str, goal: &str) -> Result<Vec<String>, String> {
    use crate::formalize::graph::Turn;
    use crate::formalize::llm::{HttpTransport, LlmClient, LlmConfig};

    let mut client = LlmClient::new(HttpTransport::new(), LlmConfig::default());
    let ask = Turn::user(&format!("goal: $ {goal} $"));
    let reply = client.chat(system, &[ask]).map_err(|e| e.to_string())?;
    let parsed: serde_json::Value = serde_json::from_str(reply.trim())
        .map_err(|e| format!("model did not return a JSON array: {e}"))?;
    Ok(parsed
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default())
}

#[cfg(not(feature = "llm-http"))]
fn llm_hints(_goal: &str) -> Result<Vec<String>, String> {
    Err("this build has no `llm-http` feature; pass --hint instead".into())
}

#[cfg(not(feature = "llm-http"))]
fn llm_hints_with(_system: &str, _goal: &str) -> Result<Vec<String>, String> {
    Err("this build has no `llm-http` feature; pass --hint instead".into())
}
