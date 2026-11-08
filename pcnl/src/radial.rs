//! The radial law `p(d) ∝ e^{−N·L(d)}·(sin d)^{P−2}` on `S^{P−1}`.
//!
//! Haar measure in geodesic polar coordinates about an anchor contributes
//! `(sin d)^{P−2}`, whose log-derivative is `(P−2)·cot d`, so
//!
//! ```text
//! d ln p/dd = (P − 2)·cot d − N·L′(d)          (exact)
//! mode:       N·L′(d)·tan d = P − 2            (the same equation, d < π/2)
//! ```
//!
//! `docs/PCNL_STATUS.md` §5 quotes the second form as a *measured* inequality:
//! `basin_posterior`'s sweep prints `N·L′(d)·tan d`, and comparing that column
//! with `P − 2` decides whether the density is still rising at a radius.  For
//! `P = 16 672` the answer is that it is rising everywhere measured — the
//! largest value in the informative band is 888 against 16 670 — which is the
//! project's "no interior mode, the predictive is the base rate" claim.
//!
//! Two details the functions here keep that the `d ≪ 1` shorthand does not:
//! `tan` rather than `d` (the profile is measured out to the pole, where they
//! differ by a factor `π/2`), and the sign of `tan` past `π/2`, which is what
//! turns the pole-adjacent row of a grid into a spurious negative residual
//! instead of a mode.

/// `(P − 2)·cot d`, the log-derivative of Haar's radial weight at `d`.
///
/// At `P = 2` the sphere is a circle and this vanishes: there is no volume
/// growth to pay for, so the density just follows the likelihood.
pub fn log_volume_slope(p: usize, d: f64) -> f64 {
    (p as f64 - 2.0) / d.tan()
}

/// `d ln p/dd`; positive means the density is still rising away from the anchor.
pub fn log_density_slope(p: usize, n: f64, dl_dd: f64, d: f64) -> f64 {
    log_volume_slope(p, d) - n * dl_dd
}

/// `N·L′(d)·tan d`, the left side of the exact mode equation.
///
/// `L′` is the derivative of the *mean* loss along the radius; a mode needs this
/// to reach `P − 2`.  Only defined for `d ∈ (0, π/2)`: past the pole `tan` is
/// negative, and a negative residual is an artefact of the coordinate, not a
/// statement about the target.
pub fn mode_residual(n: f64, dl_dd: f64, d: f64) -> f64 {
    n * dl_dd * d.tan()
}

/// Whether the mode equation is *reachable* at `d`: the residual has grown to
/// `P − 2`.  Because the residual is `0` at `d = 0` and increases while the loss
/// is convex, the largest residual on a measured grid decides existence: if it
/// falls short of `P − 2`, the density is still rising at the edge of the grid
/// and there is no mode on it (§5's claim at `P = 16 672`).
///
/// `P = 2` is excluded on purpose: the volume term vanishes, the criterion is
/// satisfied by any `N·L′·tan d > 0` however small, and the density is in fact
/// falling (see `p_two_has_no_volume_term`).
pub fn has_mode(p: usize, n: f64, dl_dd: f64, d: f64) -> bool {
    p > 2 && mode_residual(n, dl_dd, d) >= p as f64 - 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two forms are the same equation inside the hemisphere: multiplying
    /// `d ln p/dd` by `tan d` gives `(P − 2) − N·L′·tan d`, i.e. *minus* the
    /// residual's excess.  The density rises exactly while the residual is)
    /// short of `P − 2` — the reading §5 depends on.
    #[test]
    fn residual_and_slope_agree_inside_the_hemisphere() {
        let p = 16_672;
        for (n, dl, d) in [
            (512.0, 1.0, 1e-3),
            (512.0, 0.7, 4e-2),
            (512.0, 0.2, 0.3),
            (1.0, 5.0, 1.2),
        ] {
            let slope = log_density_slope(p, n, dl, d);
            let excess = mode_residual(n, dl, d) - (p as f64 - 2.0);
            assert!(
                (slope * d.tan() + excess).abs() < 1e-9 * excess.abs().max(1.0),
                "d = {d}: slope·tan = {}, excess = {}",
                slope * d.tan(),
                excess
            );
            assert_eq!(
                slope > 0.0,
                excess < 0.0,
                "d = {d}: rising density must mean a residual short of P − 2"
            );
        }
    }

    /// A quadratic loss has a closed-form mode to check against: with
    /// `L(d) = ½c·d²` the residual is `N·c·d·tan d`, and `d*` is where it crosses
    /// `P − 2`.  The density rises before `d*` and falls after — so the sign of
    /// `log_density_slope` flips exactly there.
    #[test]
    fn quadratic_loss_has_a_mode_and_the_slope_changes_sign_there() {
        let p = 8;
        let (n, c) = (64.0, 0.5);
        // Bisect the mode equation on (1e-6, π/2 − 1e-3): the residual is
        // increasing in d on that interval.
        let (mut lo, mut hi) = (1e-6_f64, std::f64::consts::FRAC_PI_2 - 1e-3);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mode_residual(n, c * mid, mid) < (p as f64 - 2.0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // `hi` is the first end whose residual has reached `P − 2`.
        let d_star = 0.5 * (lo + hi);
        assert!(
            has_mode(p, n, c * hi, hi),
            "hi = {hi} should have reached the mode"
        );
        assert!(
            !has_mode(p, n, c * lo, lo),
            "lo = {lo} should still be short"
        );
        assert!(
            log_density_slope(p, n, c * d_star, d_star).abs() < 1e-6,
            "slope at the mode should vanish, got {}",
            log_density_slope(p, n, c * d_star, d_star)
        );
        assert!(log_density_slope(p, n, c * (0.9 * d_star), 0.9 * d_star) > 0.0);
        assert!(log_density_slope(p, n, c * (1.1 * d_star), 1.1 * d_star) < 0.0);
    }

    /// The measured case: a loss that is flat in the radius (`L′ = 0`, which is
    /// what the profile shows past 0.2 rad at `P = 16 672`) can never balance the
    /// volume term, so no mode exists and the density rises all the way to the
    /// equator — the qualitative form of §5's finding.
    #[test]
    fn flat_loss_has_no_mode_anywhere_and_the_density_rises() {
        let p = 16_672;
        for d in [1e-4, 1e-2, 0.1, 0.5, 1.0, 1.5] {
            assert!(!has_mode(p, 512.0, 0.0, d), "d = {d} reported a mode");
            assert!(
                log_density_slope(p, 512.0, 0.0, d) > 0.0,
                "d = {d}: density should still be rising"
            );
            // And any plausible slope is far short of what the half-sphere needs.
            assert!(mode_residual(512.0, 0.0, d) < 1e-12);
        }
    }

    /// `P = 2` is the boundary of the family: the sphere is a circle, Haar has no
    /// radial weight, and the mode equation asks for `L′ = 0` exactly.  Any
    /// strictly rising loss therefore has no interior mode.
    #[test]
    fn p_two_has_no_volume_term() {
        for d in [1e-3, 0.2, 1.0] {
            assert_eq!(log_volume_slope(2, d), 0.0);
            assert!(log_density_slope(2, 512.0, 1e-9, d) < 0.0);
            assert!(!has_mode(2, 512.0, 1e-9, d));
        }
        // With a flat loss the radial density is flat instead: slope zero
        // everywhere, and `has_mode` declines to call that a mode.
        assert_eq!(log_density_slope(2, 512.0, 0.0, 1.0), 0.0);
        assert!(!has_mode(2, 512.0, 0.0, 1.0));
    }

    /// At the pole both `tan` and `cot` change sign, so the residual goes
    /// negative and the *volume* weight starts shrinking: the density peaks at
    /// `π/2` and falls past it.  This is the trap the sweep's grid hit — a
    /// `1.5708` row prints `−3.2 × 10⁵` and means nothing, because the radius is
    /// no longer a coordinate the mode equation describes.
    #[test]
    fn pole_flips_both_terms() {
        let p = 16_672;
        let just_inside = std::f64::consts::FRAC_PI_2 - 1e-3;
        let just_past = std::f64::consts::FRAC_PI_2 + 1e-3;
        assert!(mode_residual(512.0, 0.1, just_inside) > 0.0);
        assert!(mode_residual(512.0, 0.1, just_past) < 0.0);
        // The 1.5708 rad row of the profile grid is past the pole.
        assert!(mode_residual(512.0, 0.1, 1.5708) < 0.0);
        // Inside, the volume weight still grows — though only at rate
        // `16670·cot(π/2 − 1e-3) ≈ 16.7`, which a steep loss can already
        // outweigh — so a small likelihood slope leaves the density rising;
        // outside, the weight shrinks and the density falls for any positive
        // slope.  Both terms change sign at the pole, which is why a grid row
        // there carries no information.
        assert!(log_density_slope(p, 512.0, 1e-3, just_inside) > 0.0);
        assert!(log_density_slope(p, 512.0, 0.1, just_inside) < 0.0);
        assert!(log_density_slope(p, 512.0, 1e-3, just_past) < 0.0);
        // `cot` itself flips at the pole: the volume term is maximal there.
        assert!(log_volume_slope(p, just_inside) > 0.0);
        assert!(log_volume_slope(p, just_past) < 0.0);
    }
}
