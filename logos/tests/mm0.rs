//! Tests for `logos mm0`: the CNL fragment, the two encodings, the DeltaNet
//! cross-check, and — when the toolchain is present — mm0-rs/mm0-c on real
//! certificates, including a tampered one that must be rejected.
//!
//! The toolchain tests are not optional in spirit: the whole point of the module
//! is that an answer rests on mm0 accepting a proof, so "mm0-rs is not installed"
//! is a *skipped check reported as skipped*, not a pass. `tests/mm0_corpus.sh`
//! (run by `australVM/scripts/logos2mm0`) is the gate that requires the tools.

use logos::mm0::bridge;
use logos::mm0::cnl::Claim;
use logos::mm0::emit::{self, Verdict};
use logos::mm0::proof::Expr;

fn num(n: u64) -> Expr {
    Expr::Num(n)
}
fn add(a: Expr, b: Expr) -> Expr {
    Expr::Add(Box::new(a), Box::new(b))
}
fn var(v: &str) -> Expr {
    Expr::Var(v.to_string())
}

/// Are the mm0 tools available? If not, say so out loud.
fn tools() -> bool {
    if emit::have_elaborator() {
        true
    } else {
        eprintln!("skipping: `mm0-rs` is not on PATH (set MM0_RS); mm0 checks not run");
        false
    }
}

// ── the surface syntax ──────────────────────────────────────────────────────

#[test]
fn cnl_reads_words_and_digits() {
    let c = Claim::parse("two plus three equals five").expect("parses");
    assert_eq!(c.lhs, add(num(2), num(3)));
    assert_eq!(c.rhs, num(5));
    assert!(!c.question);

    let c = Claim::parse("30 + 12 = 42").expect("parses");
    assert_eq!(c.lhs, add(num(30), num(12)));
    assert_eq!(c.rhs, num(42));

    let c = Claim::parse("five times five equals twenty five").expect("parses");
    assert_eq!(c.lhs, Expr::Mul(Box::new(num(5)), Box::new(num(5))));
    assert_eq!(c.rhs, num(25));
}

#[test]
fn cnl_binds_times_tighter_than_plus() {
    let c = Claim::parse("two plus three times four equals fourteen").expect("parses");
    assert_eq!(
        c.lhs,
        add(num(2), Expr::Mul(Box::new(num(3)), Box::new(num(4))))
    );

    let c = Claim::parse("(two plus three) times four equals twenty").expect("parses");
    assert_eq!(
        c.lhs,
        Expr::Mul(Box::new(add(num(2), num(3))), Box::new(num(4)))
    );
}

#[test]
fn cnl_recognizes_questions() {
    for sentence in [
        "is two plus three five?",
        "is two plus three equal to five?",
        "does two plus three equal five?",
        "it is true that two plus three equals five",
    ] {
        let c = Claim::parse(sentence).unwrap_or_else(|e| panic!("{sentence}: {e}"));
        assert_eq!(c.lhs, add(num(2), num(3)), "{sentence}");
        assert_eq!(c.rhs, num(5), "{sentence}");
    }
    assert!(Claim::parse("is two plus three five?").unwrap().question);
    // A bare claim is not a question, and a trailing `?` makes it one.
    assert!(!Claim::parse("two plus three is five").unwrap().question);
    assert!(
        Claim::parse("two plus three equals five?")
            .unwrap()
            .question
    );
}

#[test]
fn cnl_quantifier_names_the_variables_the_claim_uses() {
    let c = Claim::parse("for all x, y: x plus y equals y plus x").expect("parses");
    assert_eq!(c.lhs.vars(), vec!["x".to_string(), "y".to_string()]);

    // A `for all` that quantifies nothing the body uses is refused, rather than
    // silently accepted as a claim about no variables.
    let e = Claim::parse("for all x: one plus one equals two").unwrap_err();
    assert!(e.contains("the claim uses no variables"), "{e}");

    let e = Claim::parse("for every x and y: x plus zero equals x").unwrap_err();
    assert!(e.contains("names x, y"), "{e}");

    // A name listed twice is refused too: deduplicating it silently would turn
    // `for all a, a:` — a slip no reader intends — into the same certificate as
    // `for all a:`, hiding the mistake the check exists to report.
    let e = Claim::parse("for all a, a: a plus zero equals a").unwrap_err();
    assert!(e.contains("names `a` twice"), "{e}");

    let e = Claim::parse("for all a, b, a: a plus b equals b plus a").unwrap_err();
    assert!(e.contains("names `a` twice"), "{e}");
}

#[test]
fn cnl_refuses_what_it_cannot_place() {
    let e = Claim::parse("two plus three").unwrap_err();
    assert!(e.contains("expected `equals`"), "{e}");

    let e = Claim::parse("two plus three equals").unwrap_err();
    assert!(e.contains("expected a number or a variable"), "{e}");

    // `and` is an addition word, so this reads `five + then` and then chokes on
    // the leftover; the point is that the extra words are named, not swallowed.
    let e = Claim::parse("two plus three equals five and then some").unwrap_err();
    assert!(e.contains("unexpected"), "{e}");
    assert!(e.contains("some") || e.contains("then"), "{e}");
}

#[test]
fn cnl_reads_the_successor() {
    let a = var("a");
    let expected = Claim {
        lhs: add(Expr::Suc(Box::new(a.clone())), num(0)),
        rhs: a.clone(),
        question: false,
    };
    // The three spellings are one claim: same AST, so same goal, so the same
    // certificate under the same content-addressed name.
    for sentence in [
        "for all a: suc a plus zero equals a",
        "for all a: the successor of a plus zero equals a",
        "for all a: successor of a plus zero equals a",
    ] {
        let c = Claim::parse(sentence).unwrap_or_else(|e| panic!("{sentence}: {e}"));
        assert_eq!(c, expected, "{sentence}");
    }

    // Like a numeral, `suc` takes an atom: `suc a plus b` is `(suc a) + b`, and
    // the parenthesised form is how one says `suc (a + b)`.
    let c = Claim::parse("for all a, b: suc a plus b equals suc (a plus b)").unwrap();
    assert_eq!(
        c.lhs,
        add(Expr::Suc(Box::new(var("a"))), var("b")),
        "suc binds one atom"
    );
    assert_eq!(
        c.rhs,
        Expr::Suc(Box::new(add(var("a"), var("b")))),
        "parentheses group the operand"
    );
}

// ── the encodings ───────────────────────────────────────────────────────────

#[test]
fn terms_encode_as_the_net_and_as_peano() {
    assert_eq!(Expr::numeral(0), "tz");
    assert_eq!(Expr::numeral(3), "(ts (ts (ts tz)))");

    let two_plus_three = add(num(2), num(3));
    assert_eq!(
        two_plus_three.net(),
        "(tapp (tapp tprim_add (ts (ts tz))) (ts (ts (ts tz))))"
    );
    assert_eq!(
        two_plus_three.peano(),
        "(tadd (ts (ts tz)) (ts (ts (ts tz))))"
    );

    // A variable is a term binder with a reserved prefix, so an axiom's own
    // binder of the same name cannot capture it.
    let c = Claim::parse("for all a: a plus zero equals a").unwrap();
    assert_eq!(c.lhs.vars(), vec!["a".to_string()]);
    assert_eq!(Expr::binders(&[&c.lhs]), vec!["(v_a: Term)".to_string()]);
}

#[test]
fn the_successor_is_plus_one_at_the_net_and_ts_at_peano() {
    let s = Expr::Suc(Box::new(var("a")));
    // DeltaNet has no successor agent, so it lowers to the agent that computes
    // the same thing — `Add64(a, 1)` — which is why `lower`/`corroborate` see a
    // real net and a ground `suc 4` folds to 5.
    assert_eq!(s.net(), "(tapp (tapp tprim_add v_a) (ts tz))");
    assert_eq!(s.peano(), "(ts v_a)");
    assert_eq!(Expr::Suc(Box::new(num(2))).value(), Some(3));
    assert_eq!(s.value(), None);

    assert_eq!(
        bridge::corroborate(&Expr::Suc(Box::new(num(4))), 5)
            .expect("suc 4 is 5")
            .normal_form,
        "5"
    );
}

#[test]
fn values_and_refusals_of_open_expressions() {
    assert_eq!(add(num(2), num(3)).value(), Some(5));
    assert_eq!(
        Expr::Mul(Box::new(add(num(2), num(3))), Box::new(num(4))).value(),
        Some(20)
    );
    assert_eq!(add(var("x"), num(1)).value(), None);
    assert!(add(var("x"), num(1)).evaluates().is_err());

    // A refutation of a claim that holds is refused: there is nothing to refute.
    let e = add(num(2), num(3)).refutes(&num(5), "h").unwrap_err();
    assert!(e.contains("nothing to refute"), "{e}");
}

// ── the upstream corpus ─────────────────────────────────────────────────────

// ── the candidate seam (`--hint`, `--llm`) ────────────────────────

#[test]
fn hints_are_judged_not_believed() {
    if !tools() {
        return;
    }
    // The proposal seam (`--hint`, `--llm`): every candidate — the theory's
    // own rules first, then whatever was proposed — is *elaborated*, never
    // believed. A hallucinated term must be rejected by mm0-rs and named in
    // the unproved reason, so a caller can see what was tried.
    let claim = Claim::parse("for all x, y: x plus y equals y plus x").expect("parses");
    let cert = emit::certify(&claim, &["(bogus_one v_x)".into()])
        .expect("certifies (as unproved)");
    match &cert.verdict {
        Verdict::Unproved(reason) => {
            assert!(reason.contains("(bogus_one v_x)"), "hint not reported: {reason}");
            // the theory's own rules were offered first
            assert!(reason.contains("(add_s)"), "rules not offered: {reason}");
        }
        v => panic!("a bogus hint must not prove anything, got {v:?}"),
    }

    // A well-formed hint does not change that verdict either: this theory
    // has no induction rule, and a candidate that typechecks would have to
    // actually prove the goal — not merely be well-formed.
    let cert = emit::certify(&claim, &["(eq_refl)".into()]).expect("certifies");
    assert!(
        matches!(cert.verdict, Verdict::Unproved(_)),
        "eq_refl cannot prove commutativity: {:?}",
        cert.verdict
    );
}

/// The `--llm` seam against a *deferred* endpoint, end to end.
///
/// `scripts/llm_proxy` receives the pipeline's real prompt, parks it on disk,
/// and answers only once an out-of-band model writes `ans.json`. The test
/// itself plays that model here — the same files let an agent (or a person)
/// answer by hand, which is how this seam gets run live, with a real model
/// reasoning about the actual goal. What is pinned is the contract the loop
/// rests on: the goal really goes out over the OpenAI transport, the model's
/// proposals really come back through it, and their fate is mm0-rs's judgment
/// — named in the verdict either way — never a claim made by the transport.
#[cfg(feature = "llm-http")]
#[test]
fn the_deferred_proxy_carries_model_proposals_into_judgment() {
    use std::process::{Child, Command, Stdio};
    use std::time::Duration;

    if !tools() {
        return;
    }
    if Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping: `python3` is not on PATH; deferred-proxy seam not run");
        return;
    }

    // Kill whatever is still running if an assertion trips mid-flight.
    struct Kills(Vec<Child>);
    impl Drop for Kills {
        fn drop(&mut self) {
            for c in &mut self.0 {
                let _ = c.kill();
                let _ = c.wait();
            }
        }
    }
    let state = tempfile::tempdir().expect("scratch state dir");
    let slurp = |name: &str| {
        std::fs::read_to_string(state.path().join(name)).unwrap_or_default()
    };
    // Claim a loopback port and release it: unique per run, so parallel tests
    // cannot collide on the default port.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .expect("a free loopback port")
        .local_addr()
        .unwrap()
        .port();

    let proxy_log = std::fs::File::create(state.path().join("proxy.log")).expect("log");
    let proxy = Command::new("python3")
        .arg(format!(
            "{}/../scripts/llm_proxy",
            env!("CARGO_MANIFEST_DIR")
        ))
        .arg(port.to_string())
        .env("LLM_PROXY_STATE", state.path())
        .stderr(Stdio::from(proxy_log))
        .spawn()
        .expect("the proxy runs");
    let mut kids = Kills(vec![proxy]);

    let mut listening = false;
    for _ in 0..50 {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            listening = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        listening,
        "the proxy never listened: {}",
        slurp("proxy.log")
    );

    let ask_out = std::fs::File::create(state.path().join("ask.out")).expect("log");
    let ask_err = std::fs::File::create(state.path().join("ask.err")).expect("log");
    kids.0.push(
        Command::new(env!("CARGO_BIN_EXE_logos"))
            .args([
                "mm0", "ask", "for all x, y: x times y equals y times x", "--llm",
            ])
            .env("OPENAI_BASE_URL", format!("http://127.0.0.1:{port}/v1"))
            .env("OPENAI_API_KEY", "deferred-proxy-test")
            .env("OPENAI_MODEL", "deferred-model")
            .stdout(Stdio::from(ask_out))
            .stderr(Stdio::from(ask_err))
            .spawn()
            .expect("logos runs"),
    );

    // The seam's first half: the *real* prompt lands on disk.
    for _ in 0..150 {
        if state.path().join("req.ready").exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let request = std::fs::read_to_string(state.path().join("req.json")).unwrap_or_else(|e| {
        panic!(
            "no request reached the proxy: {e}\nproxy: {}\nask: {}",
            slurp("proxy.log"),
            slurp("ask.err")
        )
    });
    assert!(
        request.contains("You are helping close a goal")
            && request.contains("logos_math"),
        "the system prompt went out: {request}"
    );
    assert!(
        request.contains(
            "teq (tapp (tapp tprim_mul v_x) v_y) (tapp (tapp tprim_mul v_y) v_x)"
        ),
        "the goal went out: {request}"
    );

    // The model's turn: two proposals, most promising first.
    let reply = serde_json::json!({
        "id": "chatcmpl-deferred",
        "object": "chat.completion",
        "created": 0,
        "model": "deferred-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "[\"(eq_sym (mul_s))\", \"(bogus_mul_comm v_x v_y)\"]"
            },
            "finish_reason": "stop"
        }],
    });
    std::fs::write(state.path().join("ans.json"), reply.to_string())
        .expect("the answer is written");

    // …and the seam's second half: judgment, not belief.
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(s) = kids.0[1].try_wait().expect("logos status") {
            break s;
        }
        if std::time::Instant::now() > deadline {
            panic!("`mm0 ask --llm` never finished\nask: {}", slurp("ask.err"));
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    assert_eq!(status.code(), Some(1), "unproved exits 1, got {status}");

    let mut verdict = slurp("ask.out");
    verdict.push_str(&slurp("ask.err"));
    assert!(verdict.contains("verdict: unproved"), "{verdict}");
    // Both model-supplied terms were carried into the judged candidate list
    // and named after failing elaboration — the transport vouches for nothing.
    for term in ["(eq_sym (mul_s))", "(bogus_mul_comm v_x v_y)"] {
        assert!(
            verdict.contains(term),
            "proposal not judged and named: {term}\n{verdict}"
        );
    }
    assert!(verdict.contains("no candidate typechecked"), "{verdict}");
}

#[test]
fn the_upstream_corpus_parses() {
    // `unfer/logos/mm0/upstream.txt` rewrites theorems from MM0's Peano
    // development (`peano.mm1`) into the CNL. Its *verdicts* are checked by
    // `australVM/scripts/logos2mm0`; this keeps the corpus from drifting out of
    // the grammar, which is cheap to check and needs no toolchain.
    let path = format!("{}/mm0/upstream.txt", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = 0;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let sentence = match line.split_once(':') {
            Some((tag, rest)) if matches!(tag.trim(), "true" | "false" | "proved" | "unproved") => {
                rest.trim()
            }
            // `for all a: …` has a colon of its own.
            _ => line,
        };
        Claim::parse(sentence).unwrap_or_else(|e| panic!("{path}: `{sentence}`: {e}"));
        lines += 1;
    }
    assert!(lines >= 60, "the corpus shrank to {lines} lines");
}

// ── the DeltaNet half ───────────────────────────────────────────────────────

#[test]
fn the_reducer_corroborates_the_arithmetic() {
    let e = Expr::Mul(Box::new(add(num(2), num(3))), Box::new(num(4)));
    let lowered = bridge::corroborate(&e, 20).expect("the net folds to 20");
    assert_eq!(lowered.value, Some(20));
    assert_eq!(lowered.normal_form, "20");
    assert_eq!(lowered.unf_hash.len(), 64, "a sha256 digest");

    // The check has teeth: a wrong claim is a disagreement, not a warning.
    let e = bridge::corroborate(&e, 21).unwrap_err();
    assert!(e.contains("disagree"), "{e}");

    // And an expression the reducer cannot fold is refused rather than guessed.
    let e = bridge::corroborate(&var("x"), 0).unwrap_err();
    assert!(e.contains("did not reduce to a literal"), "{e}");
}

#[test]
fn equal_values_are_the_same_reduced_net() {
    // `2 + 3` and `1 + 4` are different nets that reduce to the same literal, so
    // the witness is between the *reduced* nets — which is what the certificate
    // records.
    let l = bridge::corroborate(&add(num(2), num(3)), 5).expect("lhs folds");
    let r = bridge::corroborate(&add(num(1), num(4)), 5).expect("rhs folds");
    let pairs = bridge::witness(&l, &r).expect("witness runs");
    assert!(pairs.is_some(), "equal normal forms are isomorphic");
    assert_eq!(l.unf_hash, r.unf_hash, "and hash the same");
}

// ── certificates ────────────────────────────────────────────────────────────

#[test]
fn a_true_claim_is_a_proof_about_nets() {
    let c = Claim::parse("two plus three equals five").unwrap();
    let cert = emit::certify(&c, &[]).expect("certifies");
    assert_eq!(cert.verdict, Verdict::True);
    assert!(cert.mm1.contains("pub theorem"), "{}", cert.mm1);
    assert!(cert.mm1.contains("prim_add_denote"), "{}", cert.mm1);
    assert!(cert.mm1.contains("add_z"), "{}", cert.mm1);
    assert!(cert.proof.contains("prim_add_denote"));
    // The net side is carried in the certificate, not just used to check.
    assert!(cert.net_witness.as_ref().is_some_and(|p| !p.is_empty()));
    assert!(cert.unf_hashes.is_some());
}

#[test]
fn a_false_claim_is_a_refutation_under_a_hypothesis() {
    let c = Claim::parse("is two plus three six?").unwrap();
    let cert = emit::certify(&c, &[]).expect("certifies");
    assert_eq!(cert.verdict, Verdict::False);
    assert!(cert.theorem.contains("(h: $"), "{}", cert.theorem);
    assert!(cert.theorem.contains("tfals"), "{}", cert.theorem);
    assert!(cert.proof.contains("z_ne_s"), "{}", cert.proof);
    assert!(cert.proof.contains("s_inj"), "{}", cert.proof);
    assert!(cert.net_witness.is_none(), "a refutation has no witness");
}

#[test]
fn the_recursion_laws_close_open_claims_without_a_tool() {
    // Rewriting is done by the emitter, so these are decided without mm0-rs on
    // the machine — the certificate is then checked by `--verify`/the gate.
    for (sentence, rule) in [
        ("for all x: x plus zero equals x", "add_z"),
        ("for all x: x times zero equals zero", "mul_z"),
        ("for all a, b: a plus suc b equals suc (a plus b)", "add_s"),
        (
            "for all a, b: a times suc b equals a times b plus a",
            "mul_s",
        ),
        ("for all a: a plus one equals suc a", "add_s"),
        ("for all a: a plus two equals suc (suc a)", "add_s"),
        // Congruence: the zero is rewritten under the outer operator.
        (
            "for all a, b: (a plus zero) plus (b plus zero) equals a plus b",
            "eq_add_l",
        ),
        (
            "for all a, b: a times (b times zero) equals zero",
            "eq_mul_r",
        ),
    ] {
        let c = Claim::parse(sentence).unwrap_or_else(|e| panic!("{sentence}: {e}"));
        let cert = emit::certify(&c, &[]).unwrap_or_else(|e| panic!("{sentence}: {e}"));
        assert_eq!(cert.verdict, Verdict::Proved, "{sentence}");
        assert!(
            cert.proof.contains(rule),
            "{sentence}: expected {rule} in {}",
            cert.proof
        );
    }

    // Normalising both sides is what decides it: `(a + 0)` is `a`, and the two
    // sides meet at the same term.
    let (nf, proof) = add(var("a"), num(0)).normalize();
    assert_eq!(nf, var("a"));
    assert_eq!(
        proof, "(add_z)",
        "congruence steps that rewrite nothing are dropped"
    );

    // `suc` normalises at the rewriting level as well: `a + suc b` is `add_s`
    // once, with the inner `a + b` already stuck.
    let (nf, proof) = add(var("a"), Expr::Suc(Box::new(var("b")))).normalize();
    assert_eq!(nf, Expr::Suc(Box::new(add(var("a"), var("b")))));
    assert_eq!(proof, "(add_s)");
}

#[test]
fn induction_only_claims_stay_unproved() {
    if !tools() {
        return;
    }
    // Each of these is true in the standard model but not provable in the
    // equational fragment: `teq` has a model where `tz + a` is stuck. The
    // pipeline must say "unproved", name both normal forms, and say what is
    // missing — that honesty is what makes a `proved` verdict worth anything.
    for sentence in [
        "for all x, y: x plus y equals y plus x",
        "for all a: zero plus a equals a",
        "for all a: a times one equals a",
        "for all a, b, c: (a plus b) plus c equals a plus (b plus c)",
    ] {
        let c = Claim::parse(sentence).unwrap();
        let cert = emit::certify(&c, &[]).expect("runs");
        match &cert.verdict {
            Verdict::Unproved(why) => {
                assert!(why.contains("rewrites to"), "{sentence}: {why}");
                assert!(why.contains("induction"), "{sentence}: {why}");
            }
            other => panic!("{sentence}: expected unproved, got {other:?}"),
        }
        // An unproved claim is a goal, and it says so: no `.mm0` certificate.
        assert!(cert.proof.is_empty());
    }
}

// ── mm0 itself ──────────────────────────────────────────────────────────────

/// Write a certificate to a scratch directory and run the tools on it.
fn verify(cert: &emit::Certificate) -> Result<emit::Verified, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let stem = dir.path().join(&cert.name);
    std::fs::write(stem.with_extension("mm0"), &cert.mm0).map_err(|e| e.to_string())?;
    std::fs::write(stem.with_extension("mm1"), &cert.mm1).map_err(|e| e.to_string())?;
    emit::verify_mm0(&stem.with_extension("mm0"))
}

#[test]
fn mm0_accepts_a_proof_and_a_refutation() {
    if !tools() {
        return;
    }
    for (sentence, want) in [
        ("two plus three equals five", Verdict::True),
        ("is two plus three six?", Verdict::False),
        ("five times five equals twenty five", Verdict::True),
        ("30 + 12 = 42", Verdict::True),
    ] {
        let cert = emit::certify(&Claim::parse(sentence).unwrap(), &[]).expect("certifies");
        assert_eq!(cert.verdict, want, "{sentence}");
        let v = verify(&cert).unwrap_or_else(|e| panic!("{sentence}: {e}"));
        assert!(v.sorts.is_some(), "{sentence}: mm0-rs reported no counts");
    }
}

#[test]
fn mm0_accepts_rewritten_open_certificates() {
    if !tools() {
        return;
    }
    // The rewriting path's proofs are longer than a single rule application
    // (they are a chain of congruence and recursion steps), so they need their
    // own check: every one of these must survive mm0-rs *and* mm0-c.
    for sentence in [
        "for all a: a plus zero equals a",
        "for all a: a plus one equals suc a",
        "for all a: a plus two equals suc (suc a)",
        "for all a, b: a plus suc b equals suc (a plus b)",
        "for all a, b: a times suc b equals a times b plus a",
        "for all a, b: (a plus zero) plus (b plus zero) equals a plus b",
        "for all a: (a plus zero) plus zero equals a",
        "for all tz: tz plus zero equals tz",
    ] {
        let cert = emit::certify(&Claim::parse(sentence).unwrap(), &[]).expect("certifies");
        assert_eq!(cert.verdict, Verdict::Proved, "{sentence}");
        verify(&cert).unwrap_or_else(|e| panic!("{sentence}: {e}"));
    }
}

#[test]
fn mm0_rejects_a_tampered_certificate() {
    if !tools() {
        return;
    }
    // The anchor: the same claim with the *right* side changed to a different
    // numeral — the proof no longer proves the statement it claims, and mm0 has
    // to refuse it. Without this the "verified" result would be unfalsifiable.
    let cert = emit::certify(&Claim::parse("two plus three equals five").unwrap(), &[]).unwrap();
    let mut tampered = cert.clone();
    tampered.mm0 = cert
        .mm0
        .replace("(ts (ts (ts (ts (ts tz)))))", "(ts (ts (ts (ts tz))))");
    tampered.mm1 = cert
        .mm1
        .replace("(ts (ts (ts (ts (ts tz)))))", "(ts (ts (ts (ts tz))))");
    assert_ne!(
        tampered.mm0, cert.mm0,
        "the tamper must change the artifact"
    );

    let err = verify(&tampered).unwrap_err();
    assert!(
        err.contains("rejected") || err.contains("error"),
        "expected a rejection, got: {err}"
    );
}
