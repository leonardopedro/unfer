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
pub mod emit;
pub mod proof;

pub use cnl::Claim;
pub use emit::{Certificate, Verdict};
pub use proof::Expr;

use std::path::PathBuf;

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
        other => {
            eprintln!("unknown `logos mm0` command: {other}\n\n{USAGE}");
            UNKNOWN
        }
    }
}

const USAGE: &str = "\
usage:
  logos mm0 theory [--out FILE]                 dump the target theory
  logos mm0 ask <sentence> [options]            certify one sentence
  logos mm0 claims <FILE> [options]             certify a fixture, check expectations
  logos mm0 check <FILE.mm0 | DIR>              run mm0-rs and mm0-c on artifacts

ask options:
  --out DIR          write <theorem>.mm0 / <theorem>.mm1
  --verify           compile and re-check the artifact with mm0-rs + mm0-c
  --json             one JSON object instead of prose
  --hint TERM        try TERM (an mm0 proof term) for an open claim; repeatable
  --llm              ask the LLM for hint terms (needs the `llm-http` feature)

exit codes: 0 yes (true / proved), 2 no (refuted), 1 unproved or error";

fn cmd_theory(args: &[String]) -> i32 {
    let out = flag_value(args, "--out");
    match out {
        Some(path) => match std::fs::write(&path, emit::THEORY) {
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
            print!("{}", emit::THEORY);
            YES
        }
    }
}

fn cmd_ask(args: &[String]) -> i32 {
    let text = words(args);
    if text.is_empty() {
        eprintln!("usage: logos mm0 ask <sentence>\n\n{USAGE}");
        return UNKNOWN;
    }
    let claim = match Claim::parse(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return UNKNOWN;
        }
    };

    let mut hints = flag_values(args, "--hint");
    if has_flag(args, "--llm") {
        match llm_hints(&emit::goal_of(&claim)) {
            Ok(more) => hints.extend(more),
            Err(e) => eprintln!("llm: {e}"),
        }
    }

    let cert = match emit::certify(&claim, &hints) {
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

/// One line of a claims file: `true: …`, `false: …`, `proved: …`, or a bare
/// sentence (which must not be refuted — the default expectation is "yes").
struct Expected {
    line: usize,
    sentence: String,
    want: Option<Verdict>,
}

fn parse_fixture(text: &str) -> Result<Vec<Expected>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (want, sentence) = match line.split_once(':') {
            Some((tag, rest)) if matches!(tag.trim(), "true" | "false" | "proved" | "unproved") => {
                let want = match tag.trim() {
                    "true" => Verdict::True,
                    "false" => Verdict::False,
                    "unproved" => Verdict::Unproved(String::new()),
                    _ => Verdict::Proved,
                };
                (Some(want), rest.trim().to_string())
            }
            // `for all x: …` has a colon of its own, so a colon that is not one
            // of the tags is part of the sentence.
            _ => (None, line.to_string()),
        };
        out.push(Expected {
            line: i + 1,
            sentence,
            want,
        });
    }
    Ok(out)
}

fn cmd_claims(args: &[String]) -> i32 {
    let Some(file) = args
        .iter()
        .find(|a| !a.starts_with("--") && !is_flag_value(args, a))
    else {
        eprintln!("usage: logos mm0 claims <FILE> [--out DIR] [--verify] [--json]\n\n{USAGE}");
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

    let mut failures = 0usize;
    let mut results = Vec::new();
    for case in cases {
        let claim = match Claim::parse(&case.sentence) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("line {}: {e}", case.line);
                failures += 1;
                continue;
            }
        };
        let cert = match emit::certify(&claim, &[]) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("line {}: {e}", case.line);
                failures += 1;
                continue;
            }
        };
        let written = match &out_dir {
            Some(dir) => match write_certificate(dir, &cert) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("line {}: {e}", case.line);
                    failures += 1;
                    continue;
                }
            },
            None => None,
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

        let got = cert.verdict.clone();
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
        results.push((case.line, case.sentence, cert, verified));
    }

    if as_json {
        let rows: Vec<serde_json::Value> = results
            .iter()
            .map(|(line, sentence, cert, verified)| {
                let mut v = cert.to_json();
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
        for (_, sentence, cert, verified) in &results {
            println!(
                "{:<8} {}  [{}]",
                cert.verdict.label(),
                sentence,
                if verified.is_some() { "mm0 ok" } else { "-" }
            );
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
    use crate::formalize::graph::Turn;
    use crate::formalize::llm::{HttpTransport, LlmClient, LlmConfig};

    let mut client = LlmClient::new(HttpTransport::new(), LlmConfig::default());
    let system = "You are helping close a goal in the MM0 theory `logos_math`. \
Rules available: prim_add_denote, prim_mul_denote, eq_refl, eq_sym, eq_trans, eq_ts, \
eq_app, eq_add_l, eq_add_r, eq_mul_l, eq_mul_r, add_z, add_s, mul_z, mul_s, s_inj, \
z_ne_s, fals_elim. Terms: tz (zero), ts (successor), tadd, tmul, tapp, tprim_add, \
tprim_mul, teq. Reply with ONLY a JSON array of MM0 proof terms (strings), most \
promising first. A proof term looks like `(add_z)` or `(eq_trans (add_s) (eq_ts (add_z)))`.";
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
