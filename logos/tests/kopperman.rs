//! Tests for the Kopperman fragment: the CNL's formation checks (soft types),
//! the normaliser's verdicts, the bounded obvious/calculational layer, the
//! lexicon's coverage of the theory — and, when the toolchain is present,
//! mm0-rs/mm0-c on real Hilbert-space certificates.
//!
//! Same contract as `tests/mm0.rs`: a missing `mm0-rs` is a *skipped check
//! reported as skipped*, not a pass. `tests/mm0_corpus.sh` (run by
//! `australVM/scripts/logos2mm0`) is the gate that requires the tools.

use logos::mm0::Certificate;
use logos::mm0::emit::{self, Verdict};
use logos::mm0::kop::{self, KClaim, LEXICON};

/// Are the mm0 tools available? If not, say so out loud.
fn tools() -> bool {
    if emit::have_elaborator() {
        true
    } else {
        eprintln!("skipping: `mm0-rs` is not on PATH (set MM0_RS); mm0 checks not run");
        false
    }
}

fn certify(sentence: &str) -> Certificate {
    let claim = KClaim::parse(sentence).unwrap_or_else(|e| panic!("{sentence}: {e}"));
    kop::certify(&claim, &[]).unwrap_or_else(|e| panic!("{sentence}: {e}"))
}

// ── the surface syntax ──────────────────────────────────────────────────────

#[test]
fn cnl_reads_inner_products_and_heads() {
    let c =
        KClaim::parse("the inner product of the vector of 1, 2 and the vector of 3, 4 equals 11")
            .expect("parses");
    assert!(!c.question);
    assert_eq!(
        c.goal_of(),
        "teq (tinpr (tvcons (ts tz) (tvcons (ts (ts tz)) tvnil)) \
         (tvcons (ts (ts (ts tz))) (tvcons (ts (ts (ts (ts tz)))) tvnil))) \
         (ts (ts (ts (ts (ts (ts (ts (ts (ts (ts (ts tz)))))))))))"
    );

    let c = KClaim::parse(
        "for all u, v: the inner product of u and v equals the inner product of v and u",
    )
    .expect("parses");
    assert_eq!(c.goal_of(), "teq (tinpr v_u v_v) (tinpr v_v v_u)");

    let c = KClaim::parse("the coordinate 1 of the vector of 3, 4 equals 4").expect("parses");
    assert_eq!(
        c.goal_of(),
        "teq (tcoord (tvcons (ts (ts (ts tz))) (tvcons (ts (ts (ts (ts tz)))) tvnil)) (ts tz)) \
         (ts (ts (ts (ts tz))))"
    );
}

#[test]
fn cnl_recognizes_questions_and_quantifiers() {
    let c = KClaim::parse(
        "is the inner product of the vector of 1, 1 and the vector of 1, 1 equal to two?",
    )
    .expect("parses");
    assert!(c.question);

    // A `for all` that names nothing the claim uses is refused, like the
    // arithmetic fragment's.
    let e = KClaim::parse("for all x: the inner product of u and u equals zero").unwrap_err();
    assert!(e.contains("the claim uses"), "{e}");

    let e = KClaim::parse("for all a, a: the inner product of a and a equals zero").unwrap_err();
    assert!(e.contains("names `a` twice"), "{e}");
}

#[test]
fn cnl_accepts_and_separated_sums() {
    let c = KClaim::parse("for all u: the sum of the zero vector and u equals u").expect("parses");
    assert_eq!(c.goal_of(), "teq (tvadd tvnil v_u) v_u");

    let c = KClaim::parse(
        "the sum of the vector of 1, 2 and the vector of 3, 4 equals the vector of 4, 6",
    )
    .expect("parses");
    assert_eq!(
        c.goal_of(),
        "teq (tvadd (tvcons (ts tz) (tvcons (ts (ts tz)) tvnil)) \
             (tvcons (ts (ts (ts tz))) (tvcons (ts (ts (ts (ts tz)))) tvnil))) \
             (tvcons (ts (ts (ts (ts tz)))) \
              (tvcons (ts (ts (ts (ts (ts (ts tz)))))) tvnil))"
    );
}

// ── formation checking: the soft types ──────────────────────────────────────

#[test]
fn formation_refuses_sort_errors() {
    // A number where a vector belongs: `two` is a scalar, and cannot be an
    // inner product's argument.
    let e = KClaim::parse("the inner product of two and the zero vector equals zero").unwrap_err();
    assert!(e.contains("a number is a scalar"), "{e}");

    // A vector phrase in scalar position.
    let e = KClaim::parse("one plus the vector of 1, 2 equals the vector of 1, 2").unwrap_err();
    assert!(!e.is_empty());

    // Two sides of different sorts.
    let e = KClaim::parse("for all u: the inner product of u and u equals the zero vector")
        .unwrap_err();
    assert!(e.contains("different sorts"), "{e}");

    // One name, two sorts — the formation check that makes the fragment's
    // soft types checkable at parse time, since MM0's `Term` is one sort:
    // `u` is an inner product's vector argument and a coordinate's scalar
    // index in the same sentence.
    let e = KClaim::parse(
        "for all u: the inner product of u and u equals the coordinate u of the vector of 1, 2",
    )
    .unwrap_err();
    assert!(e.contains("two sorts"), "{e}");

    // A stray `the` is a phrase opener, not a variable: refuse it where the
    // phrase is not one this grammar knows.
    let e = KClaim::parse("the cat equals the cat").unwrap_err();
    assert!(!e.is_empty());
}

// ── verdicts: the normaliser decides the fragment ───────────────────────────

#[test]
fn ground_claims_are_decided_and_cross_checked() {
    let t = certify("the inner product of the vector of 1, 2 and the vector of 3, 4 equals 11");
    assert_eq!(t.verdict, Verdict::True);
    // The DeltaNet side of a true equality: the isomorphism witness between
    // the two expansions' reduced nets.
    assert!(t.net_witness.is_some(), "witness for a ground equality");
    assert!(t.unf_hashes.is_some());

    let f = certify("the inner product of the vector of 1, 2 and the vector of 3, 4 equals 12");
    assert_eq!(f.verdict, Verdict::False);
    assert!(
        f.theorem.contains("(h: $"),
        "refutation assumes the goal: {}",
        f.theorem
    );
    assert!(f.proof.contains("z_ne_s"), "{}", f.proof);

    // Coordinates compute through the Peano layer.
    let c = certify("the coordinate 1 of the vector of 3, 4 equals 4");
    assert_eq!(c.verdict, Verdict::True);

    // Vector addition merges spines coordinatewise.
    let v =
        certify("the sum of the vector of 1, 2 and the vector of 3, 4 equals the vector of 4, 6");
    assert_eq!(v.verdict, Verdict::Proved); // ground, but not a numeral claim
}

#[test]
fn open_claims_close_by_rewriting() {
    // `inpr_sym` moves the computing argument forward, then the recursion
    // computes: the flip is part of the normaliser, not the search.
    let c = certify("for all u: the inner product of u and the zero vector equals zero");
    assert_eq!(c.verdict, Verdict::Proved);

    // An open claim whose normal form computes is *proved*, not "true":
    // `true` is reserved for what evaluation decides about a closed claim.
    let c = certify("for all u: the sum of the zero vector and u equals u");
    assert_eq!(c.verdict, Verdict::Proved);

    // The zero vector computes from either side of an inner product.
    let c = certify("for all u: the inner product of the zero vector and u equals zero");
    assert_eq!(c.verdict, Verdict::Proved);
}

// ── the bounded obvious layer and calculational composition ─────────────────

#[test]
fn obvious_laws_close_what_the_recursion_cannot() {
    // `inpr_sym` at the root: the normal forms differ, the catalogue joins
    // them in one step, and mm0-rs replays the chain.
    let c =
        certify("for all u, v: the inner product of u and v equals the inner product of v and u");
    assert_eq!(c.verdict, Verdict::Proved);

    // Bilinearity: `inpr_add_r` at the root.
    let c = certify(
        "for all u, v, w: the inner product of u and the sum of v and w equals the inner product of u and v plus the inner product of u and w",
    );
    assert_eq!(c.verdict, Verdict::Proved);

    // `inpr_scal_l` at the root.
    let c = certify(
        "for all u, v: the inner product of the scalar two times u and v equals two times the inner product of u and v",
    );
    assert_eq!(c.verdict, Verdict::Proved);

    // Commutativity — the equality every instance of which is true and which
    // the recursion laws alone do not generate (the arithmetic fragment's
    // documented boundary). The Kopperman fragment's catalogue closes it,
    // replayed by mm0-rs.
    let claim = KClaim::parse("x plus five equals five plus x").expect("parses");
    let c = kop::certify(&claim, &[]).expect("certifies");
    assert_eq!(c.verdict, Verdict::Proved);
}

#[test]
fn calculational_chains_compose_laws() {
    // Two steps and a re-normalisation between them: `mul_comm` turns
    // `2 * X` into `X * 2`, re-normalising peels it into `X + X`, and the
    // sides meet. One obvious step is not enough; the bounded search is.
    let c = certify(
        "for all u: two times the inner product of u and u equals \
         the inner product of u and u plus the inner product of u and u",
    );
    assert_eq!(c.verdict, Verdict::Proved, "{:?}", c.verdict);
    assert!(!c.proof.is_empty());
}

#[test]
fn unproved_claims_stay_goals() {
    // Positive definiteness is an *implication* (`inpr_pos` takes a
    // hypothesis binder), so `u·u = 0` is not in the fragment's reach.
    let c = certify("for all u: the inner product of u and u equals zero");
    assert!(matches!(c.verdict, Verdict::Unproved(_)), "{:?}", c.verdict);
    let why = match &c.verdict {
        Verdict::Unproved(why) => why.clone(),
        _ => unreachable!(),
    };
    assert!(why.contains("tinpr v_u v_u"), "{why}");
    assert!(why.contains("tz"), "{why}");
    // An unproved claim carries no proof, and its `.mm0` is the goal file's
    // body — a statement without a proof is a `sorry`, and `mm0-c` rejects
    // those (correctly).
    assert!(c.proof.is_empty());
    assert!(c.mm0.contains("theorem"));

    // Vector commutativity is not in the overlay's axioms at all: honestly
    // unproved, never guessed.
    let c = certify("for all u, v: the sum of u and v equals the sum of v and u");
    assert!(matches!(c.verdict, Verdict::Unproved(_)), "{:?}", c.verdict);

    // The boundary of the decidable fragment, spelled out: expanding an
    // inner product whose second argument is a variable reads `tcoord (ttail
    // u)`, and relating that to `tcoord u (s u)` needs a case split on `u`
    // (`u` is `tvnil` or a `tvcons`) that this theory has no rule for. The
    // claim is true in the standard model and unreachable here — so it is a
    // goal, not a certificate.
    let c = certify(
        "for all u: the inner product of the vector of 2, 3 and u equals \
         two times the coordinate 0 of u plus three times the coordinate 1 of u",
    );
    match &c.verdict {
        Verdict::Unproved(why) => {
            assert!(why.contains("ttail"), "{why}");
        }
        other => panic!("expected the ttail boundary to stay a goal, got {other:?}"),
    }
}

// ── the lexicon ─────────────────────────────────────────────────────────────

#[test]
fn lexicon_covers_the_overlay_both_ways() {
    let overlay = kop::overlay_terms();
    assert!(overlay.contains(&"tinpr".to_string()));
    assert!(overlay.contains(&"tvcons".to_string()));

    // Every theory symbol is either spoken by the CNL or sealed on purpose.
    for term in &overlay {
        assert!(
            LEXICON.iter().any(|(s, _)| s == term),
            "`{term}` is declared by kopperman.mm0 but missing from the lexicon: \
             either give the CNL a phrase for it or seal it"
        );
    }
    // And the lexicon names no symbol the theory does not declare — drift
    // the other way would mean a phrase reading a symbol that does not exist.
    for (symbol, _) in LEXICON {
        assert!(
            overlay.iter().any(|t| t == symbol),
            "the lexicon names `{symbol}`, which kopperman.mm0 does not declare"
        );
    }
    // The sealed set is exactly the K3/limit layer and the order predicates —
    // Kopperman's Warning: no sentence may name a limit.
    let sealed: Vec<&str> = LEXICON
        .iter()
        .filter(|(_, p)| p.is_none())
        .map(|(s, _)| *s)
        .collect();
    assert!(sealed.contains(&"tlimit"));
    assert!(sealed.contains(&"tsup"));
    assert!(!sealed.contains(&"tinpr"));
}

// ── determinism ─────────────────────────────────────────────────────────────

#[test]
fn certification_is_deterministic() {
    let a = certify("the inner product of the vector of 1, 2 and the vector of 3, 4 equals 11");
    let b = certify("the inner product of the vector of 1, 2 and the vector of 3, 4 equals 11");
    assert_eq!(a.name, b.name, "content-addressed theorem names");
    assert_eq!(a.mm0, b.mm0);
    assert_eq!(a.mm1, b.mm1);

    // A different spelling of the same claim gets the same goal — and the
    // same theorem name.
    let c = certify("the inner product of the vector of 1, 2 and the vector of 3, 4 is 11?");
    assert_eq!(a.name, c.name);
    assert!(c.question);
}

// ── the exit-code contract, through `logos mm0 ask` ─────────────────────────

#[test]
fn ask_reports_verdicts_as_exit_codes() {
    assert_eq!(
        logos::mm0::run(&[
            "ask".into(),
            "the".into(),
            "inner".into(),
            "product".into(),
            "of".into(),
            "the".into(),
            "vector".into(),
            "of".into(),
            "1,".into(),
            "2".into(),
            "and".into(),
            "the".into(),
            "vector".into(),
            "of".into(),
            "3,".into(),
            "4".into(),
            "equals".into(),
            "11".into(),
        ]),
        0,
        "yes"
    );
    assert_eq!(
        logos::mm0::run(&[
            "ask".into(),
            "the inner product of two and the zero vector equals zero".into()
        ]),
        1,
        "a formation error is not an answer"
    );
}

// ── the toolchain: real certificates, both checkers ─────────────────────────

#[test]
fn certificates_verify_with_mm0_rs_and_mm0_c() {
    if !tools() {
        return;
    }
    let dir = tempfile::tempdir().expect("scratch dir");
    for sentence in [
        // a ground equality, its refutation, an open claim closed by
        // rewriting, one closed by the obvious layer, and one by a
        // calculational chain
        "the inner product of the vector of 1, 2 and the vector of 3, 4 equals 11",
        "the inner product of the vector of 1, 2 and the vector of 3, 4 equals 12",
        "for all u: the inner product of u and the zero vector equals zero",
        "for all u, v: the inner product of u and v equals the inner product of v and u",
        "for all u: two times the inner product of u and u equals the inner product of u and u plus the inner product of u and u",
    ] {
        let cert = certify(sentence);
        assert!(
            !matches!(cert.verdict, Verdict::Unproved(_)),
            "{sentence}: {}",
            cert.verdict.label()
        );
        let mm0 = dir.path().join(format!("{}.mm0", cert.name));
        let mm1 = dir.path().join(format!("{}.mm1", cert.name));
        std::fs::write(&mm0, &cert.mm0).expect("write .mm0");
        std::fs::write(&mm1, &cert.mm1).expect("write .mm1");
        let v = emit::verify_mm0(&mm0).unwrap_or_else(|e| panic!("{sentence}: {e}"));
        assert!(v.sorts.is_some(), "{sentence}: expected a sorts line");
    }
}

#[test]
fn a_tampered_certificate_is_rejected() {
    if !tools() {
        return;
    }
    let cert =
        certify("for all u, v: the inner product of u and v equals the inner product of v and u");
    // Change the theorem's *statement* to something the proof does not prove:
    // the same shape of tampering `tests/mm0.rs` performs on an arithmetic
    // certificate, and mm0-c must refuse it here too.
    let tampered = cert.mm1.replace(
        "teq (tinpr v_u v_v) (tinpr v_v v_u)",
        "teq (tinpr v_u v_v) (tinpr v_u v_u)",
    );
    assert_ne!(
        tampered, cert.mm1,
        "the tamper must actually change the file"
    );
    let dir = tempfile::tempdir().expect("scratch dir");
    let mm0 = dir.path().join("tampered.mm0");
    let mm1 = dir.path().join("tampered.mm1");
    std::fs::write(
        &mm0,
        cert.mm0.replace(
            "teq (tinpr v_u v_v) (tinpr v_v v_u)",
            "teq (tinpr v_u v_v) (tinpr v_u v_u)",
        ),
    )
    .expect("write .mm0");
    std::fs::write(&mm1, tampered).expect("write .mm1");
    assert!(
        emit::verify_mm0(&mm0).is_err(),
        "mm0-rs/mm0-c must reject a certificate whose statement was altered"
    );
}
