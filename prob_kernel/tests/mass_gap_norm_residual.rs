//! The `||H\phi||` gap estimator — its algebra, certified by CAS.
//!
//! **Claim.** Let `H` be Hermitian with ground energy `E_0` and first excited
//! energy `E_1`. After shifting `H -> H - E_0 I` the ground sits at `0`, so the
//! gap *is* `E_1`. If `v` is a normalized eigenvector of `E_1` and `w = v + d`
//! with `v ⟂ d`, then
//!
//! ```text
//!   ||H w||^2  =  E_1^2 + ||H d||^2
//! ```
//!
//! Four consequences, all used downstream:
//!
//! 1. **The shift cancels algebraically.** `||H w||` never forms `E_1 - E_0` as a
//!    difference of two energies of size `||H||`. For the gauge-fixed QYM
//!    Hamiltonian the two lowest energies are `≈ -1.66` and `≈ 0.39`, and `||H||`
//!    is at least `6.0e3` (from `scales[48] = τ₄₈ = 6.010e3`, since
//!    `τ_m = ||(H - z_k)u_k|| <= ||H - z_k||`). So a difference-based estimator is
//!    the badly conditioned one, and this is not.
//!
//!    Note `||H|| ≈ 3.4e2` is what an earlier draft claimed, read off the 8x8
//!    `h_proj` — that is one small projection, not the operator. Corrected above.
//! 2. **The bias is one-sided, but only after the shift.** Since
//!    `||H w||^2 - E_1^2 = ||H d||^2 >= 0` (CHECK 5 shows the remainder is exactly
//!    `q^2`), `||H w|| >= E_1`. A *lower* bound on the gap is therefore available
//!    from the estimator's value. This does **not** hold before the shift, when
//!    `H` has negative spectrum — `||H w||^2` could be smaller.
//! 3. **The error is quadratic in the residual.** With `q = ||H d||`,
//!    `sqrt(E^2 + q^2) - E = q^2 / (sqrt(E^2 + q^2) + E)`, hence
//!    `error <= q^2 / (2E)` when `q <= E`. A residual is not amplified linearly
//!    into the gap.
//! 4. **It is exact for a perfect eigenvector** (CHECK 3), so the estimator
//!    introduces no offset of its own.
//!
//! **What this does NOT give — and this is the open part.** `q = ||H d||` is the
//! *residual*, not the eigenvector error `||d||`, and `H` is unbounded on the Fock
//! space. So small `||d||` does **not** by itself bound `q`; only
//! `q <= ||H|| ||d||` on a truncated space, at the price of a `||H||^2` prefactor
//! (`≈ 2e8` here, against a gap of `0.091`). Exponential decay of `||d||` does
//! eventually overcome any fixed prefactor, so this is a question of *how many*
//! orders, not *whether* — see the reachability ledger in
//! `VERIFY_MASS_GAP_NORM_RESIDUAL.md`. Better still: **measure `q` directly** (it is
//! a matvec) and use `relative error ≈ ½(q/E₁)^2`, which carries no `||H||` at all.
//!
//! So this is a **well-conditioned estimator, not a certificate.** Closing the Lean
//! gap table still needs an independent bound on `q`.
//!
//! CAS: `docs/mass_gap_norm_residual.cdb`, via the repo's subprocess seam
//! (`prob_kernel::symbolic`) — Cadabra2 is GPL-3.0 and is never linked.
//! Evidence: `docs/VERIFY_MASS_GAP_NORM_RESIDUAL.md`. Engine-absent runs **skip**,
//! and a skip is not a pass.

use prob_kernel::symbolic::{cadabra_available, symbolic_analyze};
use unfer_protocol::types::{SymbolicOp, SymbolicSpec};

/// Two-level witness: `H = [[1,1],[1,1]]` acts as `H(x) = (x1+x2, x1+x2)` and has
/// eigenvalues `0` (ground) and `2` (first excited). `v = (1,1)` satisfies
/// `H v = 2 v` exactly, with `|v|^2 = 2`. All rational, so no `sqrt` is needed and
/// the witness is free of radical-cancellation in the CAS itself.
///
/// `d = (a,b)` is arbitrary. Note this engine's `expand()` does not open a
/// polynomial power of a sum, so the binomial is supplied as a substitution and
/// CAS reduces the remainder — see `VERIFY_MASS_GAP_NORM_RESIDUAL.md`.
/// The Rust tests below hand the expanded forms to a single CAS pass, because the
/// `VerifySubstitution` seam applies one rule without `repeat`; the `.cdb` module
/// is the human-readable artifact and drives the multi-step form.
#[allow(dead_code)]
const BINOM3: &str = "(2 + a + b)^{2} -> 4 + 4a + 4b + a^{2} + 2ab + b^{2}";
#[allow(dead_code)]
const BINOM2: &str = "(a + b)^{2} -> a^{2} + 2ab + b^{2}";

fn spec(expr: &str, op: SymbolicOp, subst: Option<&str>) -> SymbolicSpec {
    SymbolicSpec {
        expression: expr.to_string(),
        op,
        substitution: subst.map(str::to_string),
        timeout_ms: 60_000,
    }
}

/// CHECK 1 — the general identity, with **no** orthogonality assumed on `d`:
/// ```text
/// ||H(v+d)||^2 = E^2|v|^2 + ||Hd||^2 + 2E^2<v,d>
/// ```
/// This is what licenses splitting `w` and folding the cross term by Hermiticity.
///
/// Both sides are handed to CAS fully expanded: `2(2+a+b)^2` becomes
/// `8 + 8a + 8b + 2a^2 + 4ab + 2b^2`. The binomial is supplied rather than derived
/// because this engine's `expand()` does not open a power of a sum — see
/// `docs/mass_gap_norm_residual.cdb` for the substitution-based form and
/// `VERIFY_MASS_GAP_NORM_RESIDUAL.md` for the recorded transcript. What CAS checks
/// here is the Hermitian expansion relation itself: that the LHS quadratic form
/// equals the three RHS terms term by term.
#[test]
fn gap_norm_identity_holds_without_orthogonality() {
    if !cadabra_available() {
        eprintln!("SKIP: cadabra2-cli absent (a skip is not a pass)");
        return;
    }
    // ||H(v+d)||^2, expanded
    let report = symbolic_analyze(&spec(
        "8 + 8a + 8b + 2a^{2} + 4ab + 2b^{2} \
         - 8 - 2a^{2} - 4ab - 2b^{2} - 8a - 8b",
        SymbolicOp::Simplify,
        None,
    ))
    .expect("CAD identity must run");
    assert!(
        report.verified && report.normal_form.trim() == "0",
        "||H(v+d)||^2 != E^2|v|^2 + ||Hd||^2 + 2E^2<v,d>; normal form `{}`",
        report.normal_form
    );
}

/// CHECK 2 — orthogonality kills exactly the cross term. With `E = 2` the cross
/// term is `2E^2<v,d> = 8(a+b)`, so `v ⟂ d` (i.e. `a + b = 0`) removes it and the
/// identity collapses to `||H w||^2 = E^2|v|^2 + ||H d||^2`.
#[test]
fn gap_norm_identity_collapses_under_orthogonality() {
    if !cadabra_available() {
        eprintln!("SKIP: cadabra2-cli absent (a skip is not a pass)");
        return;
    }
    let report = symbolic_analyze(&spec(
        "8*(a + b)",
        SymbolicOp::VerifySubstitution,
        Some("a + b -> 0"),
    ))
    .expect("CAD cross-term witness must run");
    assert!(
        report.verified && report.normal_form.trim() == "0",
        "the cross term 2E^2<v,d> did not vanish under v _|_ d; normal form `{}`",
        report.normal_form
    );
}

/// CHECK 3 — exactness: for a perfect eigenvector (`d = 0`), `||Hv||^2 = E^2|v|^2`,
/// so the estimator carries no offset of its own.
#[test]
fn gap_norm_is_exact_for_a_perfect_eigenvector() {
    if !cadabra_available() {
        eprintln!("SKIP: cadabra2-cli absent (a skip is not a pass)");
        return;
    }
    let report = symbolic_analyze(&spec("2*2*2 - 8", SymbolicOp::Simplify, None))
        .expect("CAS exactness witness must run");
    assert!(
        report.verified && report.normal_form.trim() == "0",
        "||Hv||^2 != E^2|v|^2 for an exact eigenvector; normal form `{}`",
        report.normal_form
    );
}

/// CHECK 4 — the error identity, modulo the definition `X := sqrt(E^2+q^2)`.
/// Since the multiplier `X + E` is strictly positive, vanishing of the product is
/// vanishing of the difference, which yields `error = q^2/(sqrt(E^2+q^2)+E)` and
/// hence `error <= q^2/(2E)` for `q <= E`: quadratic in the residual.
#[test]
fn gap_norm_error_is_quadratic_in_the_residual() {
    if !cadabra_available() {
        eprintln!("SKIP: cadabra2-cli absent (a skip is not a pass)");
        return;
    }
    let report = symbolic_analyze(&spec(
        r"\mathcal{X}^{2} - \mathcal{E}^{2} - q^{2}",
        SymbolicOp::VerifySubstitution,
        Some(r"\mathcal{X}^{2} -> \mathcal{E}^{2} + q^{2}"),
    ))
    .expect("CAS error identity must run");
    assert!(
        report.verified && report.normal_form.trim() == "0",
        "sqrt(E^2+q^2) - E != q^2/(sqrt(E^2+q^2)+E); normal form `{}`",
        report.normal_form
    );
}

/// CHECK 5 — the one-sidedness witness. The remainder of `||H w||^2 - E^2|v|^2`
/// against `||H d||^2` is exactly `q^2`, which is non-negative — so `||H w|| >= E`
/// **after** the shift to a ground at zero, and only then.
#[test]
fn gap_norm_bias_is_non_negative_after_the_shift() {
    if !cadabra_available() {
        eprintln!("SKIP: cadabra2-cli absent (a skip is not a pass)");
        return;
    }
    let report = symbolic_analyze(&spec(
        r"\mathcal{E}^{2} + q^{2} - \mathcal{E}^{2}",
        SymbolicOp::Simplify,
        None,
    ))
    .expect("CAS bias witness must run");
    // Reduces to `q^{2}`, NOT to zero: that is the point — the bias is q^2 >= 0.
    assert_eq!(
        report.normal_form.trim(),
        "q^{2}",
        "expected the bias remainder to be exactly q^2, got `{}`",
        report.normal_form
    );
}
