//! Term encoding and proof construction for the math fragment.
//!
//! Everything here is *text generation* for MM0, in the style the rest of the
//! repo's emitters already use (`australVM/scripts/lean2mm0`, and the theory in
//! `australVM/docs/cic.mm0`): a certificate is a `.mm0` file of statements plus
//! a `.mm1` file of the same statements carrying proof terms. MM0's elaborator
//! does the argument inference, so an emitted proof term only ever names the
//! rules it uses and the hypotheses it consumes — a `(teq_trans …)` with no
//! type arguments, exactly as `lean2mm0` writes `(teq_app (teq_app teq_refl
//! teq_refl) teq_refl)`.
//!
//! The three encodings of one CNL expression, and why all three exist:
//!
//! * [`Expr::net`] — the expression as the DeltaNet agent graph it compiles to:
//!   `2 + 3` is `(tapp (tapp tprim_add (ts (ts tz))) (ts (ts (ts tz))))`, the
//!   `Prim(Add64)` agent with its two literal ports. This is what the theory's
//!   bridge axioms talk about, so the certificate is about the *net*, not about
//!   an arithmetic notation invented for the certificate.
//! * [`Expr::peano`] — the same expression in Peano form (`(tadd …)`), which is
//!   what the recursion axioms (`add_z`/`add_s`/`mul_z`/`mul_s`) rewrite.
//! * the numeral of a value — `ts^n tz`, the normal form both chains meet at.
//!
//! [`Expr::bridge`] is the proof that the first equals the second, one bridge
//! axiom per primitive agent.

use std::collections::BTreeSet;

/// An expression of the math fragment: the CNL's arithmetic, after parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Num(u64),
    Var(String),
    /// `suc a` / `the successor of a`.
    ///
    /// DeltaNet has no successor agent, so this is not a primitive of the
    /// encoding: it lowers to the `Add64` agent with a literal `1` on its second
    /// port ([`Expr::net`]), which is the same reduction the reducer performs and
    /// exactly Peano's `addS` from the other side. Keeping it in the language is
    /// what lets the fragment state the theorems an MM0 Peano development states
    /// in `suc` (`add12`, `add22`, `addSass`), rather than only their `+ 1`
    /// paraphrases.
    Suc(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
}

/// The MM0 spelling of a CNL variable.
///
/// Prefixed and sanitized because the theory's own names (`a`, `b`, `tz`, …)
/// are in scope inside every formula: a CNL variable called `a` must not be
/// captured by an axiom binder of the same name in the *statement*, and
/// `v_a` proves that at a glance.
pub fn param(name: &str) -> String {
    let mut out = String::from("v_");
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    out
}

impl Expr {
    /// A decimal numeral as a Peano term: `0` is `tz`, `n+1` is `ts (…)`.
    pub fn numeral(n: u64) -> String {
        let mut s = String::from("tz");
        for _ in 0..n {
            s = format!("(ts {s})");
        }
        s
    }

    /// The expression as the DeltaNet agent graph it compiles to.
    ///
    /// A `Prim(Add64)`/`Prim(Mul64)` node with a literal on each auxiliary port
    /// is the application spine `tapp (tapp tprim_add a) b` — the shape
    /// `deltanet::unf::canonical_serialize` walks for the same net.
    ///
    /// `suc e` is `e + 1`: `(tapp (tapp tprim_add e) (ts tz))`. That is the net
    /// `bridge::lower` builds for it (CoreIR `Prim(Add64, [e, 1])`), so the
    /// certificate is about what the reducer actually folds.
    pub fn net(&self) -> String {
        match self {
            Expr::Num(n) => Expr::numeral(*n),
            Expr::Var(v) => param(v),
            Expr::Suc(e) => format!("(tapp (tapp tprim_add {}) (ts tz))", e.net()),
            Expr::Add(a, b) => format!("(tapp (tapp tprim_add {}) {})", a.net(), b.net()),
            Expr::Mul(a, b) => format!("(tapp (tapp tprim_mul {}) {})", a.net(), b.net()),
        }
    }

    /// The expression in Peano form, the shape the recursion axioms rewrite.
    pub fn peano(&self) -> String {
        match self {
            Expr::Num(n) => Expr::numeral(*n),
            Expr::Var(v) => param(v),
            // At the Peano level `suc` is the numeral constructor the theory
            // already has: `ts`.
            Expr::Suc(e) => format!("(ts {})", e.peano()),
            Expr::Add(a, b) => format!("(tadd {} {})", a.peano(), b.peano()),
            Expr::Mul(a, b) => format!("(tmul {} {})", a.peano(), b.peano()),
        }
    }

    /// The free variables, deduplicated and sorted (so emission is stable).
    pub fn vars(&self) -> Vec<String> {
        let mut set = BTreeSet::new();
        self.collect_vars(&mut set);
        set.into_iter().collect()
    }

    fn collect_vars(&self, out: &mut BTreeSet<String>) {
        match self {
            Expr::Num(_) => {}
            Expr::Var(v) => {
                out.insert(v.clone());
            }
            Expr::Suc(e) => e.collect_vars(out),
            Expr::Add(a, b) | Expr::Mul(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
        }
    }

    /// The value of the expression, when it has one.
    ///
    /// `None` for an expression with a free variable, or one whose value does
    /// not fit the fragment's `u64` substrate. This is *not* the trusted step:
    /// it only aims the proof. The kernel is mm0, and a wrong value here is
    /// caught by mm0-rs refusing the certificate (see the tests' negative
    /// anchor).
    pub fn value(&self) -> Option<u64> {
        match self {
            Expr::Num(n) => Some(*n),
            Expr::Var(_) => None,
            Expr::Suc(e) => Some(e.value()?.checked_add(1)?),
            Expr::Add(a, b) => Some(a.value()?.checked_add(b.value()?)?),
            Expr::Mul(a, b) => Some(a.value()?.checked_mul(b.value()?)?),
        }
    }

    /// The MM0 term binders of a claim over these expressions.
    ///
    /// Sorted and deduplicated so that two spellings of one claim — and a
    /// question asked twice — emit byte-identical certificates.
    pub fn binders(exprs: &[&Expr]) -> Vec<String> {
        let mut set = BTreeSet::new();
        for e in exprs {
            for v in e.vars() {
                set.insert(v);
            }
        }
        set.into_iter()
            .map(|v| format!("({}: Term)", param(&v)))
            .collect()
    }

    // ── bridge proofs ───────────────────────────────────────────────────────

    /// A proof that the net encoding equals the Peano encoding.
    ///
    /// One bridge axiom per primitive agent, congruence for the application
    /// spine. This works for *open* expressions too — the axiom is about the
    /// agents, not about the values — which is what lets a universally
    /// quantified claim be stated at all.
    pub fn bridge(&self) -> String {
        match self {
            Expr::Num(_) | Expr::Var(_) => "(eq_refl)".to_string(),
            // At the net level `suc e` *is* `e + 1`; at the Peano level it is
            // `ts e`. So: lift the operand bridge, denote the `Add64`, and use
            // `add_s`/`add_z` once to see that `e + 1` is `ts e`.
            Expr::Suc(e) => chain(
                &chain(&app_spine(&e.bridge(), "(eq_refl)"), "(prim_add_denote)"),
                &add_one_is_suc(),
            ),
            Expr::Add(a, b) => chain(&app_spine(&a.bridge(), &b.bridge()), "(prim_add_denote)"),
            Expr::Mul(a, b) => chain(&app_spine(&a.bridge(), &b.bridge()), "(prim_mul_denote)"),
        }
    }

    /// A proof that the net encoding evaluates to the numeral of its value.
    ///
    /// `Err` when the expression is open: a variable has no value, so there is
    /// nothing to evaluate to and this certificate shape does not apply (an
    /// open claim is proved by rewriting, see [`crate::mm0::emit`]).
    pub fn evaluates(&self) -> Result<String, String> {
        let v = self.value().ok_or_else(|| {
            format!(
                "`{}` is open, so it has no value to evaluate to",
                self.pretty()
            )
        })?;
        Ok(self.prove_value(v))
    }

    /// A human-readable infix form, for diagnostics only.
    pub fn pretty(&self) -> String {
        match self {
            Expr::Num(n) => n.to_string(),
            Expr::Var(v) => v.clone(),
            Expr::Suc(e) => format!("suc {}", e.pretty()),
            Expr::Add(a, b) => format!("({} + {})", a.pretty(), b.pretty()),
            Expr::Mul(a, b) => format!("({} * {})", a.pretty(), b.pretty()),
        }
    }

    /// The proof of `teq (net self) (num v)`, for a closed `self`.
    ///
    /// Operand values are recomputed rather than threaded through the caller so
    /// that the recursion cannot be handed a value that disagrees with the
    /// subexpression it is proving something about.
    fn prove_value(&self, v: u64) -> String {
        match self {
            // A numeral's net encoding *is* its numeral, so the reflexive step
            // is the whole proof.
            Expr::Num(_) | Expr::Var(_) => "(eq_refl)".to_string(),
            Expr::Suc(e) => {
                let inner = closed_value(e);
                let spine = app_spine(&e.prove_value(inner), "(eq_refl)");
                let proved = chain(&chain(&spine, "(prim_add_denote)"), &add_numerals(inner, 1));
                debug_assert_eq!(inner + 1, v);
                proved
            }
            Expr::Add(a, b) => {
                let (va, vb) = (closed_value(a), closed_value(b));
                let spine = app_spine(&a.prove_value(va), &b.prove_value(vb));
                let proved = chain(&chain(&spine, "(prim_add_denote)"), &add_numerals(va, vb));
                debug_assert_eq!(va + vb, v);
                proved
            }
            Expr::Mul(a, b) => {
                let (va, vb) = (closed_value(a), closed_value(b));
                let spine = app_spine(&a.prove_value(va), &b.prove_value(vb));
                let proved = chain(&chain(&spine, "(prim_mul_denote)"), &mul_numerals(va, vb));
                debug_assert_eq!(va * vb, v);
                proved
            }
        }
    }

    /// A refutation of the claim `self = other`, to be given as the proof of
    /// `$ tfals $` under the hypothesis `h: $ teq (net self) (net other) $`.
    ///
    /// The shape is: normalize both sides (`evaluates`, which is already a proof
    /// about *nets*), carry the hypothesis across the equality of numerals, peel
    /// the shared `ts` prefix with `s_inj`, and let `z_ne_s` refuse the residue.
    /// Both sides must be closed and must differ; a true claim has no refutation.
    pub fn refutes(&self, other: &Expr, hyp: &str) -> Result<String, String> {
        let (vl, vr) = match (self.value(), other.value()) {
            (Some(l), Some(r)) => (l, r),
            _ => return Err("only closed claims can be refuted".into()),
        };
        if vl == vr {
            return Err(format!(
                "`{}` and `{}` have the same value ({vl}); there is nothing to refute",
                self.pretty(),
                other.pretty()
            ));
        }
        let left = self.evaluates()?;
        let right = other.evaluates()?;
        // num vl = net self (`left`, reversed), then = net other, then = num vr.
        let through_hyp = chain(&format!("(eq_sym {left})"), hyp);
        let into_numerals = chain(&through_hyp, &right);
        // After `lo` peelings both sides are numerals and the smaller one is
        // `tz`, which is exactly what `z_ne_s` refuses.
        let lo = vl.min(vr);
        let mut p = if vl < vr {
            into_numerals
        } else {
            format!("(eq_sym {into_numerals})")
        };
        for _ in 0..lo {
            p = format!("(s_inj {p})");
        }
        Ok(format!("(z_ne_s {p})"))
    }
}

/// The value of a closed expression, or a panic naming the fragment invariant
/// that was violated.
fn closed_value(e: &Expr) -> u64 {
    e.value()
        .unwrap_or_else(|| panic!("`{}` was closed when its parent was certified", e.pretty()))
}

/// `(eq_trans a b)`: transitivity, the only glue the chains need.
///
/// `eq_refl` is the identity of this composition, and dropping it keeps the
/// certificates readable: a claim like `a + 0 = a` then reads as
/// `(eq_trans (prim_add_denote) (add_z))` rather than as the same chain padded
/// with three reflexive steps.
fn chain(first: &str, second: &str) -> String {
    let identity = |p: &str| p.is_empty() || p == "(eq_refl)";
    if identity(first) {
        second.to_string()
    } else if identity(second) {
        first.to_string()
    } else {
        format!("(eq_trans {first} {second})")
    }
}

/// `(rule proof)`, or the empty chain when `proof` rewrites nothing.
///
/// A congruence step whose premise is reflexive is not emitted: `eq_add_l
/// (eq_refl)` proves `a + c = a + c`, and MM0 will accept it, but a certificate
/// full of steps that rewrite nothing is harder to read than the rewriting it is
/// supposed to show.
fn lift(rule: &str, proof: &str) -> String {
    if proof == "(eq_refl)" || proof.is_empty() {
        String::new()
    } else {
        format!("({rule} {proof})")
    }
}

/// Turn an empty chain into the reflexive proof, which every caller may hand to
/// `eq_sym`/`eq_trans`.
fn or_refl(proof: String) -> String {
    if proof.is_empty() {
        "(eq_refl)".to_string()
    } else {
        proof
    }
}

/// `(eq_sym proof)`, or the proof itself when it is already reflexive.
fn sym(proof: &str) -> String {
    if proof == "(eq_refl)" || proof.is_empty() {
        "(eq_refl)".to_string()
    } else {
        format!("(eq_sym {proof})")
    }
}

// ── open claims: normalisation by the recursion laws ────────────────────────
//
// A closed claim is decided by evaluation (`prove_value`). An *open* claim has
// no value, so the only way to prove it is to rewrite: `add_z`/`add_s` are
// directed at the second operand, and a term is brought to a normal form by
// rewriting its subterms first and then the operator itself. Two sides with the
// same normal form are equal by `eq_trans`.
//
// The rules are left-linear and non-overlapping, and they terminate (each step
// consumes a constructor of the second operand, and `mul_s` descends under
// `tadd`/`tmul`), so the system is convergent: a normal form exists, is unique,
// and two terms are equal in the congruence the laws generate exactly when their
// normal forms coincide. So this *decides* the fragment's equalities.
//
// What it does not decide is a truth the laws do not generate: `x + y = y + x`
// holds in the standard model (every numeral instance is true) yet fails in the
// congruence — there is a model of the recursion laws where it fails — which is
// why it comes back unproved. The gap is induction, a rule this theory does not
// have; a claim that lands in it is reported as a goal, never as a guess.

/// Both sides of an open claim, normalised, plus the proof when they meet.
#[derive(Debug, Clone)]
pub struct Rewritten {
    pub lhs_nf: Expr,
    pub rhs_nf: Expr,
    /// `teq (net lhs) (net rhs)`, when the normal forms coincide.
    pub proof: Option<String>,
}

/// Rewrite both sides of `lhs = rhs` to normal form and close the claim if they
/// meet. Always returns the normal forms, so the caller can say *why* a claim
/// was not closed.
pub fn rewrite_claim(lhs: &Expr, rhs: &Expr) -> Rewritten {
    let (lhs_nf, lhs_proof) = lhs.normalize();
    let (rhs_nf, rhs_proof) = rhs.normalize();
    let proof = if lhs_nf == rhs_nf {
        // net L → peano L → nf, then back out to net R through nf = nf.
        Some(join(
            &lhs.bridge(),
            &chain(&lhs_proof, &sym(&rhs_proof)),
            &rhs.bridge(),
        ))
    } else {
        None
    };
    Rewritten {
        lhs_nf,
        rhs_nf,
        proof,
    }
}

/// `teq (net lhs) (net rhs)` from a Peano-level step: bridge down to Peano
/// form, take `middle`, and bridge back up on the other side.
///
/// The two halves are the same shape everywhere in this module, which is why it
/// is one function: a certificate is always "the net is its Peano form, the
/// Peano form is normalised, and the other side is the same normal form".
pub fn join(lhs_bridge: &str, middle: &str, rhs_bridge: &str) -> String {
    chain(lhs_bridge, &chain(middle, &sym(rhs_bridge)))
}

impl Expr {
    /// Rewrite to normal form with the theory's recursion laws, returning the
    /// normal form and a proof of `teq (peano self) (peano nf)`.
    ///
    /// The proof is stated at the *Peano* level because that is the level the
    /// recursion laws talk about; [`rewrite_claim`] composes it with the net ↔
    /// Peano bridges on both sides.
    pub fn normalize(&self) -> (Expr, String) {
        match self {
            Expr::Num(_) | Expr::Var(_) => (self.clone(), "(eq_refl)".to_string()),
            Expr::Suc(e) => {
                let (nf, p) = e.normalize();
                (Expr::Suc(Box::new(nf)), or_refl(lift("eq_ts", &p)))
            }
            Expr::Add(a, b) => {
                let (na, pa) = a.normalize();
                let (nb, pb) = b.normalize();
                let step = chain(&lift("eq_add_r", &pb), &lift("eq_add_l", &pa));
                let (nf, p) = add_nf(&na, &nb);
                (nf, or_refl(chain(&step, &p)))
            }
            Expr::Mul(a, b) => {
                let (na, pa) = a.normalize();
                let (nb, pb) = b.normalize();
                let step = chain(&lift("eq_mul_r", &pb), &lift("eq_mul_l", &pa));
                let (nf, p) = mul_nf(&na, &nb);
                (nf, or_refl(chain(&step, &p)))
            }
        }
    }
}

/// The predecessor of a normal form, when it has one: `tz` has none, `ts x` and
/// a decimal numeral both do. This is the case split `add_z`/`add_s` perform.
fn pred(e: &Expr) -> Option<Expr> {
    match e {
        Expr::Num(0) => None,
        Expr::Num(n) => Some(Expr::Num(n - 1)),
        Expr::Suc(c) => Some((**c).clone()),
        _ => None,
    }
}

/// `teq (tadd a b) nf`, for normal forms `a`, `b`, by recursion on `b` — the
/// operand `add_z`/`add_s` consume.
fn add_nf(a: &Expr, b: &Expr) -> (Expr, String) {
    match pred(b) {
        None if b == &Expr::Num(0) => (a.clone(), "(add_z)".to_string()),
        Some(c) => {
            let (nf, p) = add_nf(a, &c);
            (
                Expr::Suc(Box::new(nf)),
                chain("(add_s)", &lift("eq_ts", &p)),
            )
        }
        // A variable, or a sum stuck on a variable, on the right: no rule
        // matches, and `tadd a b` is the normal form.
        None => (
            Expr::Add(Box::new(a.clone()), Box::new(b.clone())),
            "(eq_refl)".to_string(),
        ),
    }
}

/// `teq (tmul a b) nf`, likewise: `mul_s` unfolds one successor of `b` into
/// `tadd (tmul a c) a`, which is then normalised as a sum.
fn mul_nf(a: &Expr, b: &Expr) -> (Expr, String) {
    match pred(b) {
        None if b == &Expr::Num(0) => (Expr::Num(0), "(mul_z)".to_string()),
        Some(c) => {
            let (product, p) = mul_nf(a, &c);
            let (nf, p_add) = add_nf(&product, a);
            (nf, chain("(mul_s)", &chain(&lift("eq_add_l", &p), &p_add)))
        }
        None => (
            Expr::Mul(Box::new(a.clone()), Box::new(b.clone())),
            "(eq_refl)".to_string(),
        ),
    }
}

/// The congruence part of a bridge: both operand bridges lifted through the
/// application spine (`eq_app` on the head, then on the argument).
fn app_spine(a_bridge: &str, b_bridge: &str) -> String {
    format!("(eq_app (eq_app (eq_refl) {a_bridge}) {b_bridge})")
}

/// `teq (tadd a (ts tz)) (ts a)`: `add_s` once, then `add_z` under `ts`.
///
/// The two-place recursion read as "successor": this is the step that makes a
/// net `Add64` with literal `1` — which is how `suc` lowers — equal to the
/// numeral constructor it denotes.
fn add_one_is_suc() -> String {
    chain("(add_s)", "(eq_ts (add_z))")
}

/// `teq (tadd (num a) (num b)) (num (a + b))`, by Peano recursion.
///
/// Successors are peeled off the *right* operand because `add_s` is stated that
/// way: `b` unfoldings of `add_s` end exactly at `add_z`. Note that `a` does not
/// appear in the term: MM0 infers it from the goal, which is why an emitted
/// chain is as long as the smaller operand's unfoldings and no longer.
fn add_numerals(a: u64, b: u64) -> String {
    let _ = a;
    let mut p = "(add_z)".to_string();
    for _ in 0..b {
        p = chain("(add_s)", &format!("(eq_ts {p})"));
    }
    p
}

/// `teq (tmul (num a) (num b)) (num (a * b))`.
///
/// Each `mul_s` unfolds one successor of the right operand into
/// `tadd (tmul …) a`, so the `k`-th step needs the sum `a * k + a`.
fn mul_numerals(a: u64, b: u64) -> String {
    let mut p = "(mul_z)".to_string();
    for k in 0..b {
        p = chain(
            &chain("(mul_s)", &format!("(eq_add_l {p})")),
            &add_numerals(a * k, a),
        );
    }
    p
}
