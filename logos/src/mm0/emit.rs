//! Certification: turn a CNL claim into an MM0 theorem, a `.mm1` proof, and a
//! verdict.
//!
//! Two certificate shapes, matching the two kinds of sentence the fragment
//! accepts:
//!
//! * A **closed** claim is decided by evaluation. `2 + 3 = 5` becomes
//!   `theorem … : $ teq (net 2+3) (net 5) $` proved by normalizing both sides —
//!   each normalization is a proof about *nets* (`prim_add_denote`, then
//!   `add_z`/`add_s`), glued with `eq_trans`/`eq_sym`. A false closed claim
//!   cannot be proved that way, so it is certified in the other direction:
//!   `theorem … (h: $ teq (net L) (net R) $): $ tfals $` — assuming it yields
//!   absurdity, by peeling the common successors with `s_inj` and letting
//!   `z_ne_s` refuse the residue.
//!
//! * An **open** claim (`for all x: x + 0 = x`) has no value, so there is
//!   nothing to evaluate. It is stated at the level of the Peano form and closed
//!   by *rewriting both sides to normal form* with the recursion laws
//!   (`add_z`/`add_s`/`mul_z`/`mul_s`) under congruence, then joining the two
//!   normal forms (see [`crate::mm0::proof::rewrite_claim`]). A claim the
//!   rewrite system cannot join is then offered to the theory's own
//!   implication-free rules, one candidate at a time, with MM0's elaborator as
//!   the judge: mm0-rs answers whether the instantiation typechecks, and the
//!   first that does is kept. That is the seam the LLM drives — it proposes
//!   intermediate facts as candidate rules or as proof terms, and mm0 disposes
//!   of them. Nothing an LLM says enters the certificate unverified, which is
//!   the whole reason the answer is worth anything.
//!
//! Theorem names are content-addressed (a digest of the goal), so a corpus is
//! stable under reordering and re-emission, and two differently spelled claims
//! with the same goal share a name.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use super::cnl::Claim;
use super::proof::Expr;

/// The target theory, as verified by `mm0-rs` and `mm0-c` in the tests and in
/// `australVM/scripts/logos2mm0`.
pub const THEORY: &str = include_str!("../../mm0/logos_math.mm0");

/// The rules a rewritten open claim may still be closed by, in the order they
/// are offered.
///
/// Only implication-free *equality* rules: a rule with a premise needs that
/// premise constructed, which is exactly the intermediate lemma an LLM (or a
/// `--hint`) has to supply. `eq_refl` is last because a rule that rewrites
/// something is more informative than one that rewrites nothing.
const RULES: &[&str] = &[
    "prim_add_denote",
    "prim_mul_denote",
    "add_z",
    "add_s",
    "mul_z",
    "mul_s",
    "eq_refl",
];

/// What the certificate establishes about the sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// A closed claim that holds: `teq L R` is proved.
    True,
    /// A closed claim that fails: `teq L R` implies absurdity.
    False,
    /// An open claim closed by rewriting and/or a rule instantiation (the chain
    /// is in [`Certificate::proof`]).
    Proved,
    /// Nothing was proved. The reason is kept: for an open claim it names the
    /// goal, which is what a caller (or an LLM's next turn) needs.
    Unproved(String),
}

impl Verdict {
    pub fn label(&self) -> &'static str {
        match self {
            Verdict::True => "true",
            Verdict::False => "false",
            Verdict::Proved => "proved",
            Verdict::Unproved(_) => "unproved",
        }
    }
}

/// One certified sentence.
#[derive(Debug, Clone)]
pub struct Certificate {
    /// The sentence, canonically spelled (`(2 + 3) = 5`).
    pub claim: String,
    /// The theorem's public name (also the artifact's file stem).
    pub name: String,
    /// The sentence was interrogative.
    pub question: bool,
    pub verdict: Verdict,
    /// The MM0 formula proved (or, for a refutation, the hypothesis it refutes).
    pub goal: String,
    /// The MM0 rule chain, as emitted.
    pub proof: String,
    pub theorem: String,
    /// The full `.mm0` (statements) and `.mm1` (proofs) file bodies.
    pub mm0: String,
    pub mm1: String,
    /// The DeltaNet side of a true equality: the node permutation π between the
    /// two reduced nets, and the two `unf_hash` digests. `None` for a refutation
    /// (the nets differ, so there is no witness) and for open claims (nothing
    /// reduced to a literal).
    pub net_witness: Option<Vec<(u32, u32)>>,
    pub unf_hashes: Option<(String, String)>,
}

impl Certificate {
    /// A one-line, machine-readable summary.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "claim": self.claim,
            "question": self.question,
            "verdict": self.verdict.label(),
            "reason": match &self.verdict {
                Verdict::Unproved(why) => Some(why.clone()),
                _ => None,
            },
            "goal": self.goal,
            "name": self.name,
            "theorem": self.theorem,
            "proof": self.proof,
            "net_witness": self.net_witness.as_ref().map(|pairs| {
                pairs
                    .iter()
                    .map(|(a, b)| serde_json::json!([a, b]))
                    .collect::<Vec<_>>()
            }),
            "unf_hashes": self.unf_hashes,
        })
    }
}

/// A public name for a goal: the goal's digest, so that the same claim always
/// gets the same theorem name and a corpus diff shows only real changes.
fn theorem_name(goal: &str, refutation: bool) -> String {
    let mut h = Sha256::new();
    h.update(if refutation {
        "refute:".as_bytes()
    } else {
        "prove:".as_bytes()
    });
    h.update(goal.as_bytes());
    format!("logos_{}", &hex(&h.finalize())[..16])
}

/// Lowercase hex. The crate has no `hex` dependency — `deltanet::unf` carries
/// the same six lines for the same reason — and a digest prefix does not justify
/// growing the dependency list.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The claim's goal formula at the net level: `teq (net L) (net R)`.
fn goal_formula(claim: &Claim) -> String {
    format!("teq {} {}", claim.lhs.net(), claim.rhs.net())
}

/// The goal as a caller sees it — what an LLM is asked about when an open claim
/// survives rewriting and has no rule instantiation yet.
pub fn goal_of(claim: &Claim) -> String {
    goal_formula(claim)
}

/// Can this claim be decided by evaluation?
///
/// Only if both sides are closed. A claim with a variable is *universally*
/// quantified (the emitted theorem binds it), and universal claims are never
/// decided by evaluation — `2 * x = x + x` is not a fact about a number.
pub fn is_ground(claim: &Claim) -> bool {
    claim.lhs.value().is_some() && claim.rhs.value().is_some()
}

/// Certify a closed claim by evaluation, cross-checked against the net reducer.
///
/// The returned certificate has not been through mm0 yet: [`verify`] is what
/// turns the artifact into an answer. Keeping those apart matters, because the
/// *emitter* is allowed to know arithmetic (it aims the proof) while the
/// *answer* rests only on what mm0 accepts.
pub fn certify_ground(claim: &Claim) -> Result<Certificate, String> {
    let (vl, vr) = match (claim.lhs.value(), claim.rhs.value()) {
        (Some(l), Some(r)) => (l, r),
        _ => {
            return Err(format!(
                "`{} = {}` is open: it has no value to evaluate, so it must be closed by rewriting or a rule instantiation (see `certify_open`)",
                claim.lhs.pretty(),
                claim.rhs.pretty()
            ));
        }
    };

    // Both sides go through the real logos lowering, and must agree with the
    // value the proof is built for. See `bridge::corroborate`.
    let left_net = super::bridge::corroborate(&claim.lhs, vl)?;
    let right_net = super::bridge::corroborate(&claim.rhs, vr)?;

    // For a true equality the net side has its own certificate: the reduced nets
    // are the same net, and `isomorphism_witness` hands back the permutation π
    // that says so (re-checked by `IsomorphismWitness::verifies`). Two sides with
    // the same value but different normal forms would be a contradiction between
    // the reducer and arithmetic, so it is an error, not a note.
    let (net_witness, unf_hashes) = if vl != vr {
        (None, None)
    } else {
        let pairs = super::bridge::witness(&left_net, &right_net)?.ok_or_else(|| {
            format!(
                "`{}` and `{}` both equal {vl}, but their reduced nets differ — the reducer and the arithmetic disagree",
                claim.lhs.pretty(),
                claim.rhs.pretty()
            )
        })?;
        (
            Some(pairs),
            Some((left_net.unf_hash.clone(), right_net.unf_hash.clone())),
        )
    };

    let goal = goal_formula(claim);
    let refutation = vl != vr;
    let name = theorem_name(&goal, refutation);

    let proof = if refutation {
        claim.lhs.refutes(&claim.rhs, "h")?
    } else {
        // teq (net L) (num v) then teq (net R) (num v), reversed.
        let left = claim.lhs.evaluates()?;
        let right = claim.rhs.evaluates()?;
        format!("(eq_trans {left} (eq_sym {right}))")
    };

    let theorem = if refutation {
        format!("theorem {name} (h: $ {goal} $): $ tfals $")
    } else {
        format!("theorem {name}: $ {goal} $")
    };

    Ok(Certificate {
        claim: format!("{} = {}", claim.lhs.pretty(), claim.rhs.pretty()),
        name,
        question: claim.question,
        verdict: if refutation {
            Verdict::False
        } else {
            Verdict::True
        },
        mm1: file_text(&proved(&theorem, &proof)),
        mm0: file_text(&declared(&theorem)),
        proof,
        theorem,
        goal,
        net_witness,
        unf_hashes,
    })
}

/// A theorem's declaration in a `.mm0` file: the statement, terminated.
///
/// The `;` matters: `mm0-rs` tolerates its absence when elaborating the `.mm1`,
/// but `mm0-c` reads the `.mm0` as a statement stream and runs past the end of
/// the claim without it.
fn declared(theorem: &str) -> String {
    format!("{theorem};")
}

/// A proved theorem's declaration in the `.mm1`.
///
/// The `pub` is load-bearing and was found the hard way: a private theorem's
/// proof stream reaches the `.mmb` but its *declaration* does not, and `mm0-c`
/// then walks the `.mm0` past the end of the compiled file and reports
/// `expecting a theorem`. `australVM/mm0/gap_certificate.mm1` publishes its
/// theorems for the same reason; the `.mm0` keeps them unmarked because that is
/// the shape `mm0-c` checks.
fn proved(theorem: &str, proof: &str) -> String {
    format!("pub {theorem} = '{proof};")
}

/// The `.mm0`/`.mm1` body: the theory, then one claim.
fn file_text(statement: &str) -> String {
    format!(
        "{THEORY}\n-- ── claim ──────────────────────────────────────────────────────────────\n{statement}\n"
    )
}

/// Certify an open claim.
///
/// Two ways to close one, in order:
///
/// 1. **rewriting**: both sides are normalised with the recursion laws under
///    congruence. This is the normal case, needs no external tool to *build*,
///    and covers every claim the equations themselves justify (`a + 0 = a`,
///    `a + suc b = suc (a + b)`, `a * 0 = 0`, and anything with a zero in a
///    nested operand).
/// 2. **candidate rules**: [`RULES`], then `hints` in the order given — an
///    LLM's proposals come last so they cannot shadow the theory's own rules.
///    Every candidate is tried by *elaborating it*, not by believing it: mm0-rs
///    infers the arguments and either accepts the instantiation or does not.
///    This path requires `mm0-rs`: without an elaborator there is no verdict,
///    only a guess, and a guess is not a certificate.
///
/// When neither closes the claim, the certificate says what each side rewrote to
/// — which is the goal an LLM (or a person) has to answer, and the honest
/// boundary of the fragment: nothing joins them because the theory has no
/// induction rule.
pub fn certify_open(claim: &Claim, hints: &[String]) -> Result<Certificate, String> {
    let goal = goal_formula(claim);
    let name = theorem_name(&goal, false);
    let binders = Expr::binders(&[&claim.lhs, &claim.rhs]);
    // `Expr::binders` yields whole binder *groups* (`(v_x: Term)`), because the
    // quantifier prefix can name several variables at once; joining them with a
    // space gives `(v_x: Term) (v_y: Term)`, which MM0 accepts as one binder
    // list. Wrapping the join again would emit `((v_x: Term))`.
    let theorem = if binders.is_empty() {
        format!("theorem {name}: $ {goal} $")
    } else {
        format!("theorem {name} {}: $ {goal} $", binders.join(" "))
    };

    // 1. the recursion laws
    let rewritten = super::proof::rewrite_claim(&claim.lhs, &claim.rhs);
    if let Some(proof) = &rewritten.proof {
        let mm1 = file_text(&proved(&theorem, proof));
        return Ok(Certificate {
            claim: format!("{} = {}", claim.lhs.pretty(), claim.rhs.pretty()),
            name,
            question: claim.question,
            verdict: Verdict::Proved,
            goal,
            proof: proof.clone(),
            theorem: theorem.clone(),
            mm0: file_text(&declared(&theorem)),
            mm1,
            net_witness: None,
            unf_hashes: None,
        });
    }

    // 2. candidate rule instantiations. Every candidate is used the same way:
    // bridge the net goal down to Peano form, apply the candidate there, and
    // bridge back up. `(eq_refl)` is a candidate like any other, and closes
    // exactly the claims whose two Peano forms are already the same term.
    let bridge_l = claim.lhs.bridge();
    let bridge_r = claim.rhs.bridge();
    let mut candidates = Vec::new();
    for rule in RULES {
        candidates.push(format!("({rule})"));
    }
    candidates.extend(hints.iter().cloned());

    let mut failures = Vec::new();
    for cand in candidates {
        let proof = super::proof::join(&bridge_l, &cand, &bridge_r);
        let mm1 = file_text(&proved(&theorem, &proof));
        match elaborates(&file_text(&declared(&theorem)), &mm1)? {
            true => {
                return Ok(Certificate {
                    claim: format!("{} = {}", claim.lhs.pretty(), claim.rhs.pretty()),
                    name: name.clone(),
                    question: claim.question,
                    verdict: Verdict::Proved,
                    goal,
                    proof,
                    theorem: theorem.clone(),
                    mm0: file_text(&declared(&theorem)),
                    mm1,
                    net_witness: None,
                    unf_hashes: None,
                });
            }
            false => failures.push(cand),
        }
    }

    let reason = format!(
        "the recursion laws do not close it: the left side rewrites to `{}`, the right to `{}` ({}). \
         Closing that gap needs a rule this theory does not have — induction is the usual one — \
         so the claim is left as a goal rather than guessed at",
        rewritten.lhs_nf.peano(),
        rewritten.rhs_nf.peano(),
        if failures.is_empty() {
            "no candidate instantiation was offered".to_string()
        } else {
            format!("no candidate typechecked: {}", failures.join(", "))
        }
    );

    Ok(Certificate {
        claim: format!("{} = {}", claim.lhs.pretty(), claim.rhs.pretty()),
        name,
        question: claim.question,
        verdict: Verdict::Unproved(reason),
        goal,
        proof: String::new(),
        mm0: file_text(&declared(&theorem)),
        mm1: file_text(&declared(&format!("pub {theorem}"))),
        theorem,
        net_witness: None,
        unf_hashes: None,
    })
}

/// Certify one claim, choosing the shape by whether it has a value.
pub fn certify(claim: &Claim, hints: &[String]) -> Result<Certificate, String> {
    if is_ground(claim) {
        certify_ground(claim)
    } else {
        certify_open(claim, hints)
    }
}

// ── running the tools ───────────────────────────────────────────────────────

/// Where the mm0 tools live: `MM0_RS`/`MM0_C` if set, else `PATH`.
fn tool(var: &str, default: &str) -> String {
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Whether `mm0-rs` is available at all.
pub fn have_elaborator() -> bool {
    Command::new(tool("MM0_RS", "mm0-rs"))
        .arg("--version")
        .output()
        .is_ok()
}

/// Build a command for an mm0 tool that gives the child a big stack.
///
/// The checkers recurse over term structure, so a numeral like `ts^9801 tz`
/// (the value of `99 * 99`) is thousands of frames deep and overflows the
/// default 8MB stack: `ask "99 times 99 equals 9801" --verify` used to abort
/// with `thread 'main' has overflowed its stack` before any verdict, while the
/// very same artifact verifies once the checker has room (and `mm0-c`, which
/// is iterative over the proof stream, was always fine with it). The tool is
/// `exec`'d through a shell that raises `ulimit -s` first; argv, stdin and the
/// exit code are unchanged because `exec` replaces the shell.
///
/// The wrapper is used only when the tool can already be spawned directly:
/// a missing `mm0-rs` must stay a spawn error ("could not run") rather than
/// becoming a 127 exit that reads as "the kernel said no" — that distinction
/// is what [`elaborates`] and [`verify_mm0`] promise their callers. Where `sh`
/// or the rlimit cannot cooperate, the tool runs with the default stack, i.e.
/// exactly the previous behavior.
fn tool_cmd(var: &str, default: &str, args: &[&OsStr]) -> Command {
    let tool = tool(var, default);
    let available = Command::new(&tool).arg("--version").output().is_ok();
    if available && cfg!(unix) {
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(
                "ulimit -s 524288 2>/dev/null || ulimit -s unlimited \
                 2>/dev/null; exec \"$0\" \"$@\"",
            )
            .arg(&tool);
        for a in args {
            cmd.arg(a);
        }
        cmd
    } else {
        let mut cmd = Command::new(&tool);
        for a in args {
            cmd.arg(a);
        }
        cmd
    }
}

/// Ask mm0 whether the proof in `mm1` checks against the statements in `mm0`.
///
/// Returns `Ok(false)` for a rejected proof and `Err` when the tool could not be
/// run at all — a distinction the caller must preserve, since "the kernel said
/// no" and "there is no kernel here" are different facts about a claim.
fn elaborates(mm0: &str, mm1: &str) -> Result<bool, String> {
    let dir = temp_dir("elab")?;
    let mm1_path = dir.join("claim.mm1");
    std::fs::write(&mm1_path, mm1).map_err(|e| format!("{}: {e}", mm1_path.display()))?;
    // The `.mm0` is written alongside for humans and for `mm0-c`; `mm0-rs`
    // compiles the `.mm1`, which carries the same statements plus proofs.
    let mm0_path = dir.join("claim.mm0");
    std::fs::write(&mm0_path, mm0).map_err(|e| format!("{}: {e}", mm0_path.display()))?;

    let mmb_path = dir.join("claim.mmb");
    let out = tool_cmd(
        "MM0_RS",
        "mm0-rs",
        &[
            OsStr::new("compile"),
            mm1_path.as_os_str(),
            mmb_path.as_os_str(),
        ],
    )
    .output()
    .map_err(|e| format!("could not run `mm0-rs`: {e} (set MM0_RS)"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let _ = std::fs::remove_dir_all(&dir);
    let text = format!("{stdout}{stderr}");
    Ok(out.status.success() && !text.lines().any(|l| l.starts_with("error")))
}

/// A fresh scratch directory. Deliberately not `crate::engram`'s store: this is
/// a scratch space for one compiler invocation, not content to keep.
fn temp_dir(tag: &str) -> Result<PathBuf, String> {
    let mut h = Sha256::new();
    h.update(std::process::id().to_le_bytes());
    h.update(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
            .to_le_bytes(),
    );
    let dir = std::env::temp_dir().join(format!("logos-mm0-{tag}-{}", &hex(&h.finalize())[..12]));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir)
}

/// What `mm0-rs` and `mm0-c` said about a written certificate.
#[derive(Debug, Clone)]
pub struct Verified {
    pub mmb: PathBuf,
    pub sorts: Option<String>,
    /// `mm0-c`'s verdict on the `.mm0` text against the compiled `.mmb`.
    pub mm0_c_exit: Option<i32>,
}

/// Compile `certificate.mm1` and re-check the `.mm0` with the C verifier.
///
/// Two tools on purpose, exactly as `australVM/scripts/lean2mm0` does it:
/// `mm0-rs` is the elaborator that fills in the proof's implicit arguments,
/// `mm0-c` is the small independent checker over the compiled file. A claim is
/// only "verified" when both are happy, and both exits are reported so a gate
/// can tell a timeout from a rejection.
pub fn verify_mm0(mm0: &Path) -> Result<Verified, String> {
    let mm1 = mm0.with_extension("mm1");
    if !mm1.exists() {
        return Err(format!("no {} beside {}", mm1.display(), mm0.display()));
    }
    let mmb = mm0.with_extension("mmb");
    let out = tool_cmd(
        "MM0_RS",
        "mm0-rs",
        &[OsStr::new("compile"), mm1.as_os_str(), mmb.as_os_str()],
    )
    .output()
    .map_err(|e| format!("could not run `mm0-rs`: {e} (set MM0_RS)"))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() || text.lines().any(|l| l.starts_with("error")) {
        return Err(format!("mm0-rs rejected {}:\n{text}", mm0.display()));
    }
    let sorts = text
        .lines()
        .find(|l| l.contains("sorts,"))
        .map(|l| l.trim().to_string());

    // `mm0-c` reads the `.mm0` on stdin: `mm0-c MMB < MM0`.
    let source = std::fs::File::open(mm0).map_err(|e| format!("{}: {e}", mm0.display()))?;
    let check = tool_cmd("MM0_C", "mm0-c", &[mmb.as_os_str()])
        .stdin(source)
        .output();
    let mm0_c_exit = match check {
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            if !o.status.success() {
                return Err(format!("mm0-c rejected {}:\n{err}", mm0.display()));
            }
            o.status.code()
        }
        Err(_) => None, // no mm0-c here: the elaborator's word stands alone
    };

    Ok(Verified {
        mmb,
        sorts,
        mm0_c_exit,
    })
}
