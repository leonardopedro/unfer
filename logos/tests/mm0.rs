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
