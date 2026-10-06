//! M5 — property tests for the pure mass-gap functions.
//!
//! **Improves** the trust story of PLAN_M §1. The certified bands enclose
//! rounding, but nothing checked the *code paths* for the ways f64 can be wrong
//! in a way that is not rounding: monotonicity that should hold, orderings that
//! should not cross, containment that should be reflexive.
//!
//! ## Scope, honestly
//!
//! These are **`proptest` properties, not Kani harnesses**. `MASS_GAP_CERTIFIED.md`
//! §5.3 prescribes a Kani complement for the f64 paths, and that is **not done**:
//! `kani` is not a workspace dependency, no `#[kani::proof]` harness exists, and this
//! environment cannot install the toolchain. The properties below are the half of
//! M5 that is genuinely done; the Kani half is recorded as outstanding rather
//! than claimed.
//!
//! What a property test can and cannot establish is worth stating plainly,
//! because this project is particular about it. These check that the functions
//! behave on the generated domain. They do **not** prove the compiler emits
//! correct code, and a passing property test is not a certificate — which is
//! why the certificate path consumes *enclosures* (T5) rather than trusting any
//! of these numbers.
//!
//! ## Two `proptest!` gotchas, both found the hard way
//!
//! 1. A test with **no generator arguments** does not parse; the macro reports
//!    the error at the *first* function in the block, which sends you looking in
//!    the wrong place. `the_unit_roundoff_constant_is_pinned` carries a throwaway
//!    `_unused in 0u8..1` generator for this reason.
//! 2. Doc comments (`///`) inside the `proptest!` block are not accepted in the
//!    attribute position, so the per-property notes live in the function names
//!    and in this header rather than above each `#[test]`.

use fock_sirk::mass_gap_spec as spec;
use proptest::prelude::*;

proptest! {
    #[test]
    fn the_width_is_monotone_in_the_residual((r1, g, n, th) in (0.0f64..1e6, 0.0f64..1e6, 1usize..64, -1e3f64..1e3), (r2, _, _, _) in (0.0f64..1e6, 0.0f64..1e6, 1usize..64, -1e3f64..1e3)) {
        let r_lo = r1.min(r2);
                let r_hi = r1.max(r2);
                let w_lo = spec::certified_width(r_lo, n, g, th);
                let w_hi = spec::certified_width(r_hi, n, g, th);
                prop_assert!(
                    w_lo <= w_hi + 1e-9 * w_lo.abs().max(1.0),
                    "certified_width is not monotone in the residual: w({r_lo}) = {w_lo} > w({r_hi}) = {w_hi}"
                );
    }

    #[test]
    fn the_width_is_monotone_in_the_operator_norm((r, g1, n, th) in (0.0f64..1e6, 0.0f64..1e6, 1usize..64, -1e3f64..1e3), (g2, _, _, _) in (0.0f64..1e6, 0.0f64..1e6, 1usize..64, -1e3f64..1e3)) {
        let g_lo = g1.min(g2);
                let g_hi = g1.max(g2);
                let w_lo = spec::certified_width(r, n, g_lo, th);
                let w_hi = spec::certified_width(r, n, g_hi, th);
                prop_assert!(
                    w_lo <= w_hi + 1e-9 * w_lo.abs().max(1.0),
                    "certified_width is not monotone in the operator norm"
                );
    }

    #[test]
    fn the_width_is_never_negative((r, g, n, th) in (0.0f64..1e6, 0.0f64..1e6, 1usize..64, -1e3f64..1e3)) {
        let w = spec::certified_width(r, n, g, th);
                prop_assert!(w >= 0.0, "negative certified width {w}");
                prop_assert!(w.is_finite(), "non-finite certified width {w}");
    }

    #[test]
    fn interval_containment_is_reflexive(x in -1e6f64..1e6, d in 0.0f64..1e3) {
        prop_assert!(spec::interval_contains(x, d, x), "a value must lie inside its own interval");
    }

    #[test]
    fn the_gap_interval_is_ordered((t_o, t_e) in (-1e3f64..1e3, -1e3f64..1e3), (d_o, d_e) in (0.0f64..1e3, 0.0f64..1e3)) {
        let (lo, hi) = spec::gap_interval(t_o, t_e, d_o, d_e);
                prop_assert!(lo <= hi + 1e-12, "gap_interval returned an inverted interval: lo={lo} > hi={hi}");
                prop_assert!(lo.is_finite() && hi.is_finite(), "gap_interval produced a non-finite bound");
    }

    #[test]
    fn the_gap_interval_endpoints_match_the_t6_assembly((t_o, t_e) in (-1e3f64..1e3, -1e3f64..1e3), (d_o, d_e) in (0.0f64..1e3, 0.0f64..1e3)) {
        let (lo, hi) = spec::gap_interval(t_o, t_e, d_o, d_e);
                let mid = t_o - t_e;
                let half = d_o + d_e;
                prop_assert!((lo - (mid - half)).abs() <= 1e-9 * half.abs().max(1.0), "lo is not the T6 lower bound");
                prop_assert!((hi - (mid + half)).abs() <= 1e-9 * half.abs().max(1.0), "hi is not the T6 upper bound");
    }

    #[test]
    fn the_certified_lower_bound_lies_inside_the_gap_interval((t_o, t_e) in (-1e3f64..1e3, -1e3f64..1e3), (d_o, d_e) in (0.0f64..1e3, 0.0f64..1e3)) {
        let lb = spec::certified_gap_lower_bound(t_o, t_e, d_o, d_e);
                let (lo, hi) = spec::gap_interval(t_o, t_e, d_o, d_e);
                prop_assert!(
                    lb >= lo - 1e-9 && lb <= hi + 1e-9,
                    "certified_gap_lower_bound {lb} is outside the gap interval [{lo}, {hi}]"
                );
    }

    #[test]
    fn certified_positive_follows_lo((t_o, t_e) in (-1e3f64..1e3, -1e3f64..1e3), (d_o, d_e) in (0.0f64..1e3, 0.0f64..1e3)) {
        let iv = spec::gap_interval(t_o, t_e, d_o, d_e);
        let lo = iv.0;
                prop_assert!(
                    spec::gap_certified_positive(lo) == (lo > 0.0),
                    "certified_positive did not follow lo = {lo}"
                );
    }

    #[test]
    fn backward_error_is_monotone_in_the_rank(n1 in 1u32..64, n2 in 1u32..64) {
        let lo = n1.min(n2);
                let hi = n1.max(n2);
                let c_lo = spec::backward_error_const(lo as usize);
                let c_hi = spec::backward_error_const(hi as usize);
                prop_assert!(
                    c_lo <= c_hi + 1e-12,
                    "backward_error_const is not monotone in the rank: c({lo}) = {c_lo} > c({hi}) = {c_hi}"
                );
                prop_assert!(c_lo.is_finite() && c_hi.is_finite(), "backward error constant is not finite");
    }

    #[test]
    fn the_unit_roundoff_constant_is_pinned(_unused in 0u8..1) {
        prop_assert_eq!(
                    spec::UNIT_ROUNDOFF,
                    2f64.powi(-53),
                    "UNIT_ROUNDOFF must be exactly 2^-53; if it changed, every pinned certificate was computed against a different number"
                );
    }

    #[test]
    fn the_parlett_bound_is_monotone_in_the_residual(r1 in 0.0f64..1e6, r2 in 0.0f64..1e6, psi in 0.1f64..1e3) {
        let lo_r = r1.min(r2);
                let hi_r = r1.max(r2);
                let b_lo = spec::parlett_bound(lo_r, psi);
                let b_hi = spec::parlett_bound(hi_r, psi);
                prop_assert!(b_lo <= b_hi + 1e-9, "parlett_bound is not monotone in the residual");
    }

    #[test]
fn the_rayleigh_quotient_is_finite(x in 1e-9f64..1e3f64, h in -1e6f64..1e6) {
        // `x` is bounded away from zero on purpose: `rayleigh_quotient` *panics*
        // on a zero vector, which is the right behaviour (the quotient is
        // undefined), so the property has to state the precondition rather than
        // call the function outside its domain. An earlier version of this
        // generator included 0.0 and produced a spurious failure.
        let q = spec::rayleigh_quotient(h, x);
        prop_assert!(q.is_finite(), "rayleigh_quotient({h}, {x}) is not finite");
    }

}
