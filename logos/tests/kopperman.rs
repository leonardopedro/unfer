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

/// Certify against the Kopperman/Mehler theory (`--mehler`): the
/// tail-determination axioms are in the target, so may be cited.
fn certify_mehler(sentence: &str) -> Certificate {
    let claim = KClaim::parse(sentence).unwrap_or_else(|e| panic!("{sentence}: {e}"));
    kop::certify_in(&claim, &[], kop::Theory::Mehler).unwrap_or_else(|e| panic!("{sentence}: {e}"))
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

// ── the Kopperman/Mehler overlay: the random tail, fully determined ─────────

/// The `--mehler` extension, end to end at the verdict level: the case split
/// the base fragment cannot touch closes under the determination axioms (and
/// the proof *cites* the axiom it used), the coordinatewise laws each close
/// with their own axiom, and the vector-equality boundary does not move —
/// determination is not induction.
#[test]
fn mehler_determines_the_random_tail() {
    let split = "for all u: the inner product of the vector of 2, 3 and u equals two times the coordinate 0 of u plus three times the coordinate 1 of u";
    assert!(
        matches!(certify(split).verdict, Verdict::Unproved(_)),
        "the base fragment must still leave the case split as a goal"
    );
    let cert = certify_mehler(split);
    assert!(
        matches!(cert.verdict, Verdict::Proved),
        "{}",
        cert.verdict.label()
    );
    assert!(
        cert.mm1.contains("(coord_tail)"),
        "the proof must cite coord_tail: {}",
        cert.mm1
    );

    let sum = "for all u, v, i: the coordinate i of the sum of u, v equals the coordinate i of u plus the coordinate i of v";
    assert!(matches!(certify(sum).verdict, Verdict::Unproved(_)));
    let cert = certify_mehler(sum);
    assert!(
        matches!(cert.verdict, Verdict::Proved),
        "{}",
        cert.verdict.label()
    );
    assert!(cert.mm1.contains("(vadd_coord)"), "{}", cert.mm1);

    let scal = "for all c, v, i: the coordinate i of the scalar c times v equals c times the coordinate i of v";
    let cert = certify_mehler(scal);
    assert!(
        matches!(cert.verdict, Verdict::Proved),
        "{}",
        cert.verdict.label()
    );
    assert!(cert.mm1.contains("(vscal_coord)"), "{}", cert.mm1);

    // The boundary: vector commutativity needs extensionality, not
    // determination — still a goal under `--mehler`.
    assert!(matches!(
        certify_mehler("for all u, v: the sum of u and v equals the sum of v and u").verdict,
        Verdict::Unproved(_)
    ));

    // The two target theories differ by exactly the overlay file.
    assert!(
        kop::MEHLER_THEORY.contains("axiom coord_tail"),
        "the mehler theory must carry the determination axioms"
    );
    assert!(
        !kop::KTHEORY.contains("axiom coord_tail"),
        "the base theory must not carry them — `--mehler` is optional"
    );
}

/// The `--mehler` rule catalogue, checked against the overlay the same way
/// the base catalogue is checked against `kopperman.mm0`: every offered rule
/// exists, every overlay axiom is offered (all four are unconditional
/// equalities, so none may be silently dropped), the mehler catalogue keeps
/// every base rule, and the goal format the prompt promises is the one
/// `goal_of` produces.
#[test]
fn llm_mehler_catalogue_tracks_the_overlay() {
    fn rules_of<'a>(system: &'a str, name: &str) -> Vec<&'a str> {
        system
            .split_once("Rules available: ")
            .map(|(_, rest)| rest.split('.').next().unwrap_or_default())
            .unwrap_or_else(|| panic!("{name} lost its rule catalogue"))
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    }
    let offered = rules_of(kop::LLM_SYSTEM_MEHLER, "LLM_SYSTEM_MEHLER");
    let base = rules_of(kop::LLM_SYSTEM, "LLM_SYSTEM");
    assert!(offered.len() > 30, "catalogue too thin: {}", offered.len());

    // A `--mehler` session must not lose a base rule.
    for rule in &base {
        assert!(
            offered.contains(rule),
            "`{rule}` is in the base catalogue but missing from the mehler one"
        );
    }

    // Every offered rule is a declaration of the mehler target theory.
    let declared: Vec<String> = kop::MEHLER_THEORY
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let rest = line
                .strip_prefix("axiom ")
                .or_else(|| line.strip_prefix("theorem "))?;
            rest.split_whitespace()
                .next()
                .map(|n| n.trim_end_matches(['(', ':']).to_string())
        })
        .collect();
    for rule in &offered {
        assert!(
            declared.iter().any(|d| d == rule),
            "`{rule}` is offered to the model but is not an axiom or theorem of the mehler theory"
        );
    }

    // Every axiom of the overlay file is offered: all four are unconditional
    // equalities that can close a bare `teq`.
    let mut overlay = 0usize;
    for line in include_str!("../mm0/mehler.mm0").lines() {
        let Some(rest) = line.trim_start().strip_prefix("axiom ") else {
            continue;
        };
        overlay += 1;
        let name = rest.split_whitespace().next().unwrap_or_default();
        let name = name.trim_end_matches(['(', ':']);
        assert!(
            offered.contains(&name),
            "the mehler axiom `{name}` is never offered to the model"
        );
    }
    assert_eq!(overlay, 4, "mehler.mm0 should carry exactly four axioms");

    // And the promised goal format is the one `goal_of` produces.
    let goal = KClaim::parse(
        "for all u, v, i: the coordinate i of the sum of u, v equals the coordinate i of u plus the coordinate i of v",
    )
    .expect("parses")
    .goal_of();
    assert!(goal.starts_with("teq "), "goal shape: {goal}");
    assert!(
        kop::LLM_SYSTEM_MEHLER.contains("goal: $ teq L R $"),
        "the prompt must describe the goal line the seam sends"
    );
}

/// The `--mehler` certificates are real: mm0-rs elaborates them and mm0-c
/// re-checks them, over the extended theory that carries the axioms they
/// cite.
#[test]
fn mehler_certificates_verify_with_mm0_rs_and_mm0_c() {
    if !tools() {
        return;
    }
    let dir = tempfile::tempdir().expect("scratch dir");
    for sentence in [
        "for all u: the inner product of the vector of 2, 3 and u equals two times the coordinate 0 of u plus three times the coordinate 1 of u",
        "for all u, v, i: the coordinate i of the sum of u, v equals the coordinate i of u plus the coordinate i of v",
        "for all c, v, i: the coordinate i of the scalar c times v equals c times the coordinate i of v",
    ] {
        let cert = certify_mehler(sentence);
        assert!(
            matches!(cert.verdict, Verdict::Proved),
            "{}",
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

// ── the LLM seam: the catalogue, the goal format, the judgment ──────────────

/// The `--llm` catalogue is data *about* the theory, checked against it the
/// same way [`LEXICON`] is: every rule the prompt offers must be a real
/// declaration of the target theory (a hallucinated name in the prompt is a
/// hallucination waiting to happen), every axiom of the overlay must be
/// offered — with the two that cannot close an equality goal named as
/// exceptions rather than silently dropped — and the goal format the prompt
/// promises must be the one [`KClaim::goal_of`] actually produces.
#[test]
fn llm_catalogue_tracks_the_theory_and_goal_format() {
    // The "Rules available:" list, up to its terminating period.
    let offered: Vec<&str> = kop::LLM_SYSTEM
        .split_once("Rules available: ")
        .map(|(_, rest)| rest.split('.').next().unwrap_or_default())
        .unwrap_or_else(|| panic!("LLM_SYSTEM lost its rule catalogue"))
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    assert!(
        offered.len() > 30,
        "the catalogue looks too thin: {} rules",
        offered.len()
    );

    // Every offered rule is a declaration of the full target theory.
    let declared: Vec<String> = kop::KTHEORY
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let rest = line
                .strip_prefix("axiom ")
                .or_else(|| line.strip_prefix("theorem "))?;
            rest.split_whitespace()
                .next()
                .map(|name| name.trim_end_matches(['(', ':']).to_string())
        })
        .collect();
    for rule in &offered {
        assert!(
            declared.iter().any(|d| d == rule),
            "`{rule}` is offered to the model but is not an axiom or theorem of the Kopperman theory"
        );
    }

    // Every axiom of the overlay itself is offered. The exceptions are
    // named, not dropped — each is an axiom that cannot close a bare `teq`:
    // Prop-valued conclusions (`tle`/`tlt` atoms), hypothesis binders, or
    // statements over the sealed K1/K3 vocabulary no CNL sentence can reach.
    let cannot_close_a_teq = [
        // inpr_pos is an implication; eq_tapply mentions the sealed K3
        // sequence layer.
        "inpr_pos",
        "eq_tapply",
        // K1 order: conclusions are `tle`/`tlt` atoms or refute to tfals.
        "le_refl",
        "le_antisym",
        "le_trans",
        "lt_le",
        "lt_irrefl",
        "lt_asym",
        "le_add_r",
        "le_mul_nonneg",
        // K1 roots: hypothesis-bound, and `tsqrt`/`tcbrt` are sealed in the
        // CNL — no goal can even mention them.
        "rcf_sqrt",
        "rcf_cbrt",
        // K1 completeness: hypothesis-bound over the sealed sequence layer.
        "sup_upper",
        "sup_approx",
        // K3: Prop-valued, or over sealed vocabulary.
        "cauchy_complete",
        "conv_limit",
        "riesz_fischer",
        "rat_approx",
    ];
    let mut overlay = 0usize;
    for line in include_str!("../mm0/kopperman.mm0").lines() {
        let Some(rest) = line.trim_start().strip_prefix("axiom ") else {
            continue;
        };
        overlay += 1;
        let name = rest.split_whitespace().next().unwrap_or_default();
        let name = name.trim_end_matches(['(', ':']);
        if cannot_close_a_teq.contains(&name) {
            continue;
        }
        assert!(
            offered.contains(&name),
            "the overlay's `{name}` is never offered to the model"
        );
    }
    assert!(
        overlay > 20,
        "the overlay axiom parse looks wrong: {overlay}"
    );

    // The goal format the prompt promises is the goal `goal_of` produces:
    // a `teq` line whose variables carry the `v_` prefix.
    let goal = KClaim::parse(
        "for all u, v: the inner product of u and v equals the inner product of v and u",
    )
    .expect("parses")
    .goal_of();
    assert!(goal.starts_with("teq "), "goal shape: {goal}");
    assert!(
        goal.contains("v_u") && goal.contains("v_v"),
        "the `v_` prefix the prompt names: {goal}"
    );
    assert!(
        kop::LLM_SYSTEM.contains("goal: $ teq L R $"),
        "the prompt must describe the goal line the seam sends"
    );
}

/// The `--llm` seam on a *Kopperman* goal, against the same deferred endpoint
/// `tests/mm0.rs` uses: the Kopperman catalogue (not the arithmetic one) must
/// go out with the Kopperman goal, and the model's proposals must come back
/// into mm0-rs's judgment — named in the verdict when they fail, exactly like
/// the arithmetic fragment's `the_deferred_proxy_carries_model_proposals_into_judgment`.
#[cfg(feature = "llm-http")]
#[test]
fn the_deferred_proxy_carries_kopperman_proposals_into_judgment() {
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
    let slurp = |name: &str| std::fs::read_to_string(state.path().join(name)).unwrap_or_default();
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

    // A goal the obvious layer cannot close: vector commutativity needs
    // induction, so without a hint this claim stays a goal.
    let ask_out = std::fs::File::create(state.path().join("ask.out")).expect("log");
    let ask_err = std::fs::File::create(state.path().join("ask.err")).expect("log");
    kids.0.push(
        Command::new(env!("CARGO_BIN_EXE_logos"))
            .args([
                "mm0",
                "ask",
                "for all u, v: the sum of u and v equals the sum of v and u",
                "--llm",
            ])
            .env("OPENAI_BASE_URL", format!("http://127.0.0.1:{port}/v1"))
            .env("OPENAI_API_KEY", "deferred-proxy-test")
            .env("OPENAI_MODEL", "deferred-model")
            .stdout(Stdio::from(ask_out))
            .stderr(Stdio::from(ask_err))
            .spawn()
            .expect("logos runs"),
    );

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
    // The Kopperman catalogue went out — not the arithmetic fragment's: a
    // K2 rule is named, and the agent-spine vocabulary it does not offer is
    // absent.
    assert!(
        request.contains("theory `kopperman`")
            && request.contains("inpr_cons")
            && request.contains("eq_tvadd"),
        "the Kopperman catalogue went out: {request}"
    );
    assert!(
        !request.contains("tprim_mul"),
        "the arithmetic catalogue leaked into a Kopperman goal: {request}"
    );
    // …with the Kopperman goal in the format the catalogue promises.
    assert!(
        request.contains("teq (tvadd v_u v_v) (tvadd v_v v_u)"),
        "the Kopperman goal went out: {request}"
    );

    // The model's turn: one real rule that does not apply, one hallucination.
    let reply = serde_json::json!({
        "id": "chatcmpl-deferred",
        "object": "chat.completion",
        "created": 0,
        "model": "deferred-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "[\"(inpr_sym)\", \"(bogus_vector_comm v_u v_v)\"]"
            },
            "finish_reason": "stop"
        }],
    });
    std::fs::write(state.path().join("ans.json"), reply.to_string())
        .expect("the answer is written");

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
    // Both proposals were carried into the judged candidate list, replayed
    // by mm0-rs, and named in the reason — the transport vouches for nothing.
    assert!(
        verdict.contains("(inpr_sym)") && verdict.contains("(bogus_vector_comm v_u v_v)"),
        "the model's proposals were not named in the verdict: {verdict}"
    );
}
