//! M5 — Kani harnesses for the pure mass-gap core (`mass_gap_spec`).
//!
//! `MASS_GAP_CERTIFIED.md` §5.3 prescribes a Kani complement for the f64
//! paths. `tests/mass_gap_properties.rs` holds the proptest half; this module
//! is the model-checking half: the same twelve properties (plus the three
//! witness predicates the proptests do not sample — non-negativity of the
//! Parlett bound, and the disjointness/vacuum witness predicates), each
//! checked over **every** finite `f64` in the stated range instead of a
//! random sample.
//!
//! ## Why the harnesses live in the lib, not in `tests/`
//!
//! `cargo kani` 0.68 verifies only lib and bin targets — its CLI has no
//! `--tests` option, and a harness file under `tests/` is silently not built
//! ("No proof harnesses were found"). So the module is declared in `lib.rs`
//! behind `#[cfg(kani)]` (and gated again below): `cargo kani` sets that cfg,
//! a plain `cargo build`/`cargo test` compiles an empty module, and no
//! `#[test]` lives here either way, so suite counts do not move.
//!
//! ## What a harness here proves, and what it does not
//!
//! CBMC checks the compiled IEEE-754 operations of these pure functions: for
//! all inputs satisfying the harness's `kani::assume` bounds, the asserted
//! postcondition holds **bit-for-bit** — no `+ tolerance` fudge. The proptests
//! needed tolerances because sampled floats are noisy; the exact inequalities
//! are theorems (rounding to nearest is monotone, and every term here is a
//! monotone composition of additions, multiplications and `max` in
//! non-overflowing ranges).
//!
//! It does **not** prove the certificate. The residual measurement (T2 on a
//! real Krylov chain), the eigendecomposition enclosure (T1/T3), and
//! `nalgebra`'s `h_proj_spectral_norm` live outside this seam; the runtime
//! trusts the *enclosures* (T5), not any of these numbers.
//!
//! ## Running
//!
//! ```text
//! cargo kani -Z unstable-options -p fock_sirk --ignore-global-asm
//! cargo kani -Z unstable-options -p fock_sirk --ignore-global-asm --harness <name>
//! ```
//!
//! `--ignore-global-asm` is required because `pulp` — a transitive SIMD
//! dependency of `gemm`/`scirs2` — declares `global_asm!`, which Kani refuses
//! to compile by default. The flag carries Kani's own caveat (it "may impact
//! the soundness of the analysis"); none of these harnesses can reach `pulp`:
//! `mass_gap_spec` is plain f64 arithmetic with no dependencies.
//!
//! ## Why nothing here calls `powi`
//!
//! CBMC's library model of `__builtin_powi` (shipped inside the `cbmc`
//! 6.11.0 binary that Kani 0.68 drives) is **not** the exact integer power:
//! it is the Ankerl "optimized pow" bit-hack whose exponent correction
//! `exp_c` is a `__VERIFIER_nondet_int32_t()` assumed to lie in
//! `[-90253, 1]` — a nondeterministic approximation, fresh on every call.
//! On hardware `x.powi(3)` lowers to two `mulsd`s, bit-identical to
//! `x * x * x` (checked over 5 M random bit patterns), but under Kani any
//! assertion that pins or orders a `powi` result is unprovable. The first
//! run of this module demonstrated exactly that: `the_unit_roundoff_
//! constant_is_pinned`, `backward_error_is_monotone_in_the_rank`,
//! `the_width_is_monotone_in_the_residual` and
//! `the_width_is_monotone_in_the_operator_norm` each FAILED with a
//! counterexample while every non-`powi` check in the same harness passed —
//! the artifact, not a bug. So the spec cubes the rank by explicit
//! multiplication (bit-identical, see `backward_error_const`) and the
//! roundoff pin compares `to_bits()` against `0x3CA0_0000_0000_0000`.

#![cfg(kani)]

use crate::mass_gap_spec as spec;

/// The width is monotone in the residual (proptest
/// `the_width_is_monotone_in_the_residual`), exactly: `r_lo ≤ r_hi` implies
/// `w(r_lo) ≤ w(r_hi)` for all inputs in the proptest range.
#[kani::proof]
fn the_width_is_monotone_in_the_residual() {
    let r1 = kani::any::<f64>();
    let r2 = kani::any::<f64>();
    let g = kani::any::<f64>();
    let n = kani::any::<usize>();
    let th = kani::any::<f64>();
    kani::assume((0.0..=1e6).contains(&r1));
    kani::assume((0.0..=1e6).contains(&r2));
    kani::assume((0.0..=1e6).contains(&g));
    kani::assume((1..=64).contains(&n));
    kani::assume((-1e3..=1e3).contains(&th));
    let (r_lo, r_hi) = if r1 <= r2 { (r1, r2) } else { (r2, r1) };
    let w_lo = spec::certified_width(r_lo, n, g, th);
    let w_hi = spec::certified_width(r_hi, n, g, th);
    assert!(
        w_lo <= w_hi,
        "certified_width is not monotone in the residual: w({r_lo}) = {w_lo} > w({r_hi}) = {w_hi}"
    );
}

/// The width is monotone in the operator norm (proptest
/// `the_width_is_monotone_in_the_operator_norm`), exactly: `g_norm.max(1.0)`
/// and the subsequent scaling are monotone, so no tolerance is needed.
#[kani::proof]
fn the_width_is_monotone_in_the_operator_norm() {
    let r = kani::any::<f64>();
    let g1 = kani::any::<f64>();
    let g2 = kani::any::<f64>();
    let n = kani::any::<usize>();
    let th = kani::any::<f64>();
    kani::assume((0.0..=1e6).contains(&r));
    kani::assume((0.0..=1e6).contains(&g1));
    kani::assume((0.0..=1e6).contains(&g2));
    kani::assume((1..=64).contains(&n));
    kani::assume((-1e3..=1e3).contains(&th));
    let (g_lo, g_hi) = if g1 <= g2 { (g1, g2) } else { (g2, g1) };
    let w_lo = spec::certified_width(r, n, g_lo, th);
    let w_hi = spec::certified_width(r, n, g_hi, th);
    assert!(
        w_lo <= w_hi,
        "certified_width is not monotone in the operator norm: w({g_lo}) = {w_lo} > w({g_hi}) = {w_hi}"
    );
}

/// The width is never negative and always finite (proptest
/// `the_width_is_never_negative`): a sum of non-negative, bounded terms.
#[kani::proof]
fn the_width_is_never_negative() {
    let r = kani::any::<f64>();
    let g = kani::any::<f64>();
    let n = kani::any::<usize>();
    let th = kani::any::<f64>();
    kani::assume((0.0..=1e6).contains(&r));
    kani::assume((0.0..=1e6).contains(&g));
    kani::assume((1..=64).contains(&n));
    kani::assume((-1e3..=1e3).contains(&th));
    let w = spec::certified_width(r, n, g, th);
    assert!(w >= 0.0, "negative certified width {w}");
    assert!(w.is_finite(), "non-finite certified width {w}");
}

/// Interval containment is reflexive (proptest
/// `interval_containment_is_reflexive`), and containment is exact at the
/// upper endpoint `value + delta` (unit test
/// `gap_assembly_and_interval_contracts`), both for all inputs in range.
#[kani::proof]
fn interval_containment_is_reflexive() {
    let x = kani::any::<f64>();
    let d = kani::any::<f64>();
    kani::assume((-1e6..=1e6).contains(&x));
    kani::assume((0.0..=1e3).contains(&d));
    assert!(
        spec::interval_contains(x, d, x),
        "a value must lie inside its own interval"
    );
    assert!(
        spec::interval_contains(x, d, x + d),
        "the upper endpoint must be contained exactly"
    );
}

/// The gap interval is ordered and finite (proptest
/// `the_gap_interval_is_ordered`): `lo ≤ hi` for all inputs in range, exactly.
#[kani::proof]
fn the_gap_interval_is_ordered() {
    let t_o = kani::any::<f64>();
    let t_e = kani::any::<f64>();
    let d_o = kani::any::<f64>();
    let d_e = kani::any::<f64>();
    kani::assume((-1e3..=1e3).contains(&t_o));
    kani::assume((-1e3..=1e3).contains(&t_e));
    kani::assume((0.0..=1e3).contains(&d_o));
    kani::assume((0.0..=1e3).contains(&d_e));
    let (lo, hi) = spec::gap_interval(t_o, t_e, d_o, d_e);
    assert!(
        lo <= hi,
        "gap_interval returned an inverted interval: lo={lo} > hi={hi}"
    );
    assert!(
        lo.is_finite() && hi.is_finite(),
        "gap_interval produced a non-finite bound"
    );
}

/// The gap interval endpoints *are* the T6 assembly (proptest
/// `the_gap_interval_endpoints_match_the_t6_assembly`), bitwise: `gap_interval`
/// and `certified_gap_lower_bound` evaluate the same expression with the same
/// associativity, so CBMC proves exact `==` rather than the proptest's 1e-9
/// tolerance.
#[kani::proof]
fn the_gap_interval_endpoints_match_the_t6_assembly() {
    let t_o = kani::any::<f64>();
    let t_e = kani::any::<f64>();
    let d_o = kani::any::<f64>();
    let d_e = kani::any::<f64>();
    kani::assume((-1e3..=1e3).contains(&t_o));
    kani::assume((-1e3..=1e3).contains(&t_e));
    kani::assume((0.0..=1e3).contains(&d_o));
    kani::assume((0.0..=1e3).contains(&d_e));
    let (lo, hi) = spec::gap_interval(t_o, t_e, d_o, d_e);
    let mid = t_o - t_e;
    let half = d_o + d_e;
    let lb = spec::certified_gap_lower_bound(t_o, t_e, d_o, d_e);
    assert!(
        lo == mid - half,
        "lo {lo} is not bitwise the T6 lower bound {}",
        mid - half
    );
    assert!(
        hi == mid + half,
        "hi {hi} is not bitwise the T6 upper bound {}",
        mid + half
    );
    assert!(
        lo == lb,
        "certified_gap_lower_bound {lb} disagrees with gap_interval lo {lo}"
    );
}

/// The certified lower bound lies inside the gap interval (proptest
/// `the_certified_lower_bound_lies_inside_the_gap_interval`) — exactly: it
/// *equals* `lo`, and `lo ≤ hi` whenever both widths are non-negative.
#[kani::proof]
fn the_certified_lower_bound_lies_inside_the_gap_interval() {
    let t_o = kani::any::<f64>();
    let t_e = kani::any::<f64>();
    let d_o = kani::any::<f64>();
    let d_e = kani::any::<f64>();
    kani::assume((-1e3..=1e3).contains(&t_o));
    kani::assume((-1e3..=1e3).contains(&t_e));
    kani::assume((0.0..=1e3).contains(&d_o));
    kani::assume((0.0..=1e3).contains(&d_e));
    let lb = spec::certified_gap_lower_bound(t_o, t_e, d_o, d_e);
    let (lo, hi) = spec::gap_interval(t_o, t_e, d_o, d_e);
    assert!(
        lb == lo,
        "certified_gap_lower_bound {lb} is not the interval's own lo {lo}"
    );
    assert!(
        lb <= hi,
        "certified_gap_lower_bound {lb} exceeds the interval's hi {hi}"
    );
}

/// The stopping rule follows `lo` (proptest `certified_positive_follows_lo`):
/// the predicate is the raw `lo > 0.0` comparison, for every interval the
/// assembly can produce.
#[kani::proof]
fn certified_positive_follows_lo() {
    let t_o = kani::any::<f64>();
    let t_e = kani::any::<f64>();
    let d_o = kani::any::<f64>();
    let d_e = kani::any::<f64>();
    kani::assume((-1e3..=1e3).contains(&t_o));
    kani::assume((-1e3..=1e3).contains(&t_e));
    kani::assume((0.0..=1e3).contains(&d_o));
    kani::assume((0.0..=1e3).contains(&d_e));
    let lo = spec::gap_interval(t_o, t_e, d_o, d_e).0;
    assert!(
        spec::gap_certified_positive(lo) == (lo > 0.0),
        "certified_positive did not follow lo = {lo}"
    );
}

/// The backward-error constant is monotone in the rank (proptest
/// `backward_error_is_monotone_in_the_rank`) and finite — exactly, since
/// `n³` for `n ≤ 64` is representable without rounding.
#[kani::proof]
fn backward_error_is_monotone_in_the_rank() {
    let n1 = kani::any::<usize>();
    let n2 = kani::any::<usize>();
    kani::assume((1..=64).contains(&n1));
    kani::assume((1..=64).contains(&n2));
    let (n_lo, n_hi) = if n1 <= n2 { (n1, n2) } else { (n2, n1) };
    let c_lo = spec::backward_error_const(n_lo);
    let c_hi = spec::backward_error_const(n_hi);
    assert!(
        c_lo <= c_hi,
        "backward_error_const is not monotone: c({n_lo}) = {c_lo} > c({n_hi}) = {c_hi}"
    );
    assert!(
        c_lo.is_finite() && c_hi.is_finite(),
        "backward error constant is not finite"
    );
}

/// `UNIT_ROUNDOFF` is exactly `2⁻⁵³` (proptest
/// `the_unit_roundoff_constant_is_pinned`). If it changed, every pinned
/// certificate would have been computed against a different number. Checked
/// as the IEEE-754 bit pattern of `2⁻⁵³` (exponent field `0x3CA`, zero
/// mantissa) and against `f64::EPSILON` (which is `2⁻⁵²`). The proptest
/// spells the power `2f64.powi(-53)`; that cannot be used here because
/// CBMC models `__builtin_powi` approximately — see the module header.
#[kani::proof]
fn the_unit_roundoff_constant_is_pinned() {
    assert!(
        spec::UNIT_ROUNDOFF.to_bits() == 0x3CA0_0000_0000_0000,
        "UNIT_ROUNDOFF must be exactly 2^-53"
    );
    assert!(
        spec::UNIT_ROUNDOFF * 2.0 == f64::EPSILON,
        "UNIT_ROUNDOFF must be half f64::EPSILON"
    );
}

/// The Parlett bound is monotone in the residual (proptest
/// `the_parlett_bound_is_monotone_in_the_residual`), exactly: division by the
/// same positive `psi_norm` is monotone in the numerator.
#[kani::proof]
fn the_parlett_bound_is_monotone_in_the_residual() {
    let r1 = kani::any::<f64>();
    let r2 = kani::any::<f64>();
    let psi = kani::any::<f64>();
    kani::assume((0.0..=1e6).contains(&r1));
    kani::assume((0.0..=1e6).contains(&r2));
    kani::assume((0.1..=1e3).contains(&psi));
    let (r_lo, r_hi) = if r1 <= r2 { (r1, r2) } else { (r2, r1) };
    let b_lo = spec::parlett_bound(r_lo, psi);
    let b_hi = spec::parlett_bound(r_hi, psi);
    assert!(
        b_lo <= b_hi,
        "parlett_bound is not monotone in the residual: {b_lo} > {b_hi}"
    );
}

/// The Rayleigh quotient is finite on the proptest domain (proptest
/// `the_rayleigh_quotient_is_finite`). `psi_psi` is bounded away from zero on
/// purpose: the quotient *panics* on a zero vector by design, so the
/// precondition is stated rather than violated. `|h / x| ≤ 1e6 / 1e-9 = 1e15`
/// cannot overflow.
#[kani::proof]
fn the_rayleigh_quotient_is_finite() {
    let x = kani::any::<f64>();
    let h = kani::any::<f64>();
    kani::assume((1e-9..=1e3).contains(&x));
    kani::assume((-1e6..=1e6).contains(&h));
    let q = spec::rayleigh_quotient(h, x);
    assert!(q.is_finite(), "rayleigh_quotient({h}, {x}) is not finite");
}

/// Spec postcondition of [`spec::parlett_bound`]: the returned value is
/// non-negative (the proptests only checked monotonicity).
#[kani::proof]
fn the_parlett_bound_is_non_negative() {
    let r = kani::any::<f64>();
    let psi = kani::any::<f64>();
    kani::assume((0.0..=1e6).contains(&r));
    kani::assume((0.1..=1e3).contains(&psi));
    let b = spec::parlett_bound(r, psi);
    assert!(b >= 0.0, "negative Parlett bound {b}");
    assert!(b.is_finite(), "non-finite Parlett bound {b}");
}

/// The sector-disjointness witness predicate is exactly its comparison
/// (spec contract of [`spec::parities_disjoint`]); the assumptions satisfy
/// the `debug_assert` precondition (`overlap ≥ 0`, `tol ≥ 0`).
#[kani::proof]
fn the_disjointness_predicate_follows_its_witness() {
    let overlap = kani::any::<f64>();
    let tol = kani::any::<f64>();
    kani::assume((0.0..=1e3).contains(&overlap));
    kani::assume((0.0..=1e3).contains(&tol));
    assert!(
        spec::parities_disjoint(overlap, tol) == (overlap < tol),
        "parities_disjoint({overlap}, {tol}) did not follow the raw comparison"
    );
}

/// The legacy vacuum predicate is exactly its comparison (spec contract of
/// [`spec::even_sector_is_vacuum`]).
#[kani::proof]
fn the_vacuum_predicate_follows_its_tolerance() {
    let even_ground = kani::any::<f64>();
    let tol = kani::any::<f64>();
    kani::assume((-1e6..=1e6).contains(&even_ground));
    kani::assume((0.0..=1e3).contains(&tol));
    assert!(
        spec::even_sector_is_vacuum(even_ground, tol) == (even_ground.abs() < tol),
        "even_sector_is_vacuum({even_ground}, {tol}) did not follow the raw comparison"
    );
}
