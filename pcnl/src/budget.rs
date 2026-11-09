//! The information–volume budget: **how many directions can a finite data set
//! localize?**
//!
//! **Why this exists.**  `docs/PCNL_STATUS.md` (2026-11-06) established that the
//! tempered sphere target `π(u) ∝ exp(−N·L(u))` against Haar measure on
//! `S^{P−1}` is *prior-dominated* at `P = 16 672`: every Haar draw lands 1.0–1.6
//! rad from the fine-tune, so `π`'s predictive is the base rate and every
//! 46–51 % number in the study is `π` **conditioned on a cone**.  The two sides
//! of that trade were priced with a proxy — a cap of angular radius `d` holds
//! `≈ (sin(d/2))^{P−1}` of the sphere — which is exact only at `P = 3` (it is
//! `sin²(d/2)` on `S²`).  That error is irrelevant to the conclusion (`examples/
//! dimension_budget` measures it as `(P−1)·ln 2 = 11 547` nats, i.e. 5 014
//! decades, against a six-figure margin) but it is **not** irrelevant to the
//! crossover, which lives at small `m`, where a factor of `2^m` is the answer.
//!
//! **The exact law.**  The Haar mass of a geodesic cap of radius `d` around a
//! point of `S^{m−1}` is
//!
//! ```text
//! f_m(d) = ½ · I_{sin² d}((m−1)/2, ½),      d ≤ π/2,
//! ```
//!
//! with `I` the regularized incomplete beta and `f_m(d) = 1 − f_m(π−d)` beyond
//! `π/2`.  The three small cases are closed forms and are the module's unit
//! tests: `d/π` at `m = 2`, `sin²(d/2)` at `m = 3`, `(d − sin d cos d)/π` at
//! `m = 4`.  For large `m` and small `d` the leading order is
//!
//! ```text
//! −ln f_m(d) ≈ (m−1)·ln(1/sin d) + ½·ln(2π(m−1)),
//! ```
//!
//! i.e. **a linear price in the number of directions**, which is the whole
//! point: the likelihood can pay at most `N·(L_max − L_min)` nats, so the
//! number of directions it can localize is `m* = N·ΔL / ln(1/sin d)` and the
//! remaining `P − m*` directions are a free random walk.  That is the statement
//! the sampler measurements were groping at: the fixed-radius chain's growth
//! exponent of 0.509 and 89.7° pairwise angles are what `P − m* ≫ m*` looks
//! like, and the radial relaxation time `≈ P/(K r²)` is the same arithmetic
//! read from the other end.
//!
//! **What the budget is not.**  It is a statement about a *uniform* (Haar)
//! prior; a centred prior moves the counting to a ball and changes the price
//! per direction, and an optimisation does not sample at all.  It prices
//! conditioning, not samplers.

use std::f64::consts::{FRAC_PI_2, LN_10, PI};

/// Lanczos `g = 7, n = 9` coefficients (the usual double-precision set).
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// `ln Γ(x)` for `x > 0`, by Lanczos with the reflection formula below `0.5`.
///
/// Accuracy is ~1e-13 relative, which is all the budget needs: every use here
/// differs two large `ln Γ` values separated by a half-integer, so the absolute
/// error is ~1e-12 against prices of order `10³–10⁵`.
pub fn ln_gamma(x: f64) -> f64 {
    assert!(x > 0.0, "ln_gamma needs a positive argument, got {x}");
    if x < 0.5 {
        return PI.ln() - (PI * x).sin().ln() - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let t = z + 7.5;
    let mut a = LANCZOS[0];
    for (i, c) in LANCZOS.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

/// Modified Lentz continued fraction for the incomplete beta (Numerical
/// Recipes 6.4), evaluated in plain `f64`: it is `O(1)` in the regime used here
/// because `x` is small, and the *magnitude* `bt·cf/a` is what underflows —
/// which is why the caller takes logarithms.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 400;
    const EPS: f64 = 3e-16;
    const FPMIN: f64 = f64::MIN_POSITIVE / EPS;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        // even step
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // odd step
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            return h;
        }
    }
    h
}

/// `ln I_x(a, b)`, the logarithm of the regularized incomplete beta.
///
/// Computed as `bt·betacf/a` (the series branch, taken for
/// `x < (a+1)/(a+b+2)`) or as `1 − bt·betacf(b,a,1−x)/b`, with `bt` folded into
/// log space so that the `x^a` underflow that makes the plain formula useless
/// at `m = 16 672, d = 1e-3` never happens.
fn ln_beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return 0.0;
    }
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_bt - a.ln() + betacf(a, b, x).ln()
    } else {
        // I_x(a,b) = 1 − I_{1−x}(b,a); the second term is the small one here.
        let tail = ln_bt - b.ln() + betacf(b, a, 1.0 - x).ln();
        (-tail.exp()).ln_1p()
    }
}

/// `ln` of the Haar mass of a geodesic cap of angular radius `d` on `S^{m−1}`.
///
/// `m` is the **ambient** dimension: `m = 2` is the circle, `m = 3` the ordinary
/// sphere.  `d` is in radians and clamped to `[0, π]`.
///
/// ```text
/// f_m(d) = ½ I_{sin²d}((m−1)/2, ½)          d ≤ π/2
/// f_m(d) = 1 − f_m(π − d)                   d > π/2
/// ```
pub fn ln_cap_fraction(m: usize, d: f64) -> f64 {
    assert!(m >= 2, "a cap needs an ambient dimension of at least 2");
    if d <= 0.0 || !d.is_finite() {
        return f64::NEG_INFINITY;
    }
    if d >= PI {
        return 0.0;
    }
    let near = d.min(PI - d);
    let a = (m as f64 - 1.0) / 2.0;
    let ln_half = 0.5f64.ln() + ln_beta_inc(a, 0.5, near.sin().powi(2));
    if d <= FRAC_PI_2 {
        ln_half
    } else {
        // f = 1 − exp(ln_half); ln_1p keeps the digits when the cap is nearly
        // the whole sphere.
        (-ln_half.exp()).ln_1p()
    }
}

/// The cap's mass as a fraction of the sphere, in decades (`log₁₀`).
pub fn cap_decades(m: usize, d: f64) -> f64 {
    -ln_cap_fraction(m, d) / LN_10
}

/// Nats a likelihood must pay to give a `d`-cap most of the posterior mass
/// against a uniform prior on `S^{m−1}`: `−ln f_m(d)`.
pub fn cap_cost(m: usize, d: f64) -> f64 {
    -ln_cap_fraction(m, d)
}

/// The asymptotic price of one direction, `−ln sin d` nats (`≈ −ln d` while
/// `d ≪ 1`): the exchange rate between nats of likelihood and dimensions of
/// conditioning.
pub fn cost_per_direction(d: f64) -> f64 {
    assert!(d > 0.0 && d < PI, "a radius in (0, π) rad is required");
    -(d.sin().ln())
}

/// The largest ambient dimension `m` for which `available` nats of likelihood
/// buy a `d`-cap most of the mass — the **dimension budget** `m*`.
///
/// Solved by bisection on the exact `cap_cost` (monotone increasing in `m`
/// while `d ≤ π/2`), so the answer is not the linear approximation; it agrees
/// with `1 + available / cost_per_direction(d)` to a few percent wherever the
/// leading order is valid.  Returns `2.0` when even the circle cannot be
/// localized (the budget is below the smallest meaningful dimension), and caps
/// at `M_MAX` — a budget that buys more than ten million directions is, for
/// this problem, unbounded.
pub fn budget_dimension(available: f64, d: f64) -> f64 {
    const M_MAX: f64 = 1e7;
    assert!(d > 0.0 && d < PI, "a radius in (0, π) rad is required");
    if available.is_nan() || available <= 0.0 {
        return 2.0;
    }
    if cap_cost(2, d) >= available {
        return 2.0;
    }
    let (mut lo, mut hi) = (2.0f64, M_MAX);
    if cap_cost(M_MAX as usize, d) <= available {
        return M_MAX;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cap_cost(mid.round().max(2.0) as usize, d) <= available {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-6 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// The radius at which the cap holds a fraction `1/draws` of the sphere: the
/// nearest a set of `draws` Haar directions is expected to come to a given
/// point.  This is the check that validates the whole formula against a
/// measured *distance distribution* rather than a moment.
pub fn nearest_expected_radius(m: usize, draws: usize) -> f64 {
    assert!(draws >= 2);
    let target = -(draws as f64).ln();
    let (mut lo, mut hi) = (1e-12f64, FRAC_PI_2);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if ln_cap_fraction(m, mid) <= target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-15 {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    /// `S¹`: the cap is an arc, so the fraction is exactly `d/π`.
    #[test]
    fn circle_cap_is_the_arc_length() {
        for d in [1e-6, 0.01, 0.5, 1.0, 1.5, 2.0, 3.0] {
            let f = ln_cap_fraction(2, d).exp();
            assert!(close(f, d / PI, 1e-12), "m=2 d={d}: {f} vs {}", d / PI);
        }
    }

    /// `S²`: the cap is exactly `sin²(d/2)` — this is the one case the status
    /// doc's `(sin(d/2))^{P−1}` proxy gets right, which is why the proxy looked
    /// safe.
    #[test]
    fn s2_cap_is_sin_squared_half_angle() {
        for d in [1e-5, 0.02, 0.3, 1.0, 1.5, 2.0, 3.0] {
            let f = ln_cap_fraction(3, d).exp();
            let want = (d / 2.0).sin().powi(2);
            assert!(close(f, want, 1e-12), "m=3 d={d}: {f} vs {want}");
        }
    }

    /// `S³`: the closed form is `(d − sin d cos d)/π`, the first case where the
    /// proxy is wrong (it says `sin³(d/2)`).  Radii stay above 0.05 rad because
    /// the *reference* `d − sin d cos d` cancels catastrophically below that;
    /// small caps are covered by the Monte-Carlo test.
    #[test]
    fn s3_cap_matches_its_closed_form() {
        for d in [0.05, 0.2, 0.7, 1.2, 1.5] {
            let f = ln_cap_fraction(4, d).exp();
            let want = (d - d.sin() * d.cos()) / PI;
            assert!(close(f, want, 1e-11), "m=4 d={d}: {f} vs {want}");
        }
    }

    /// The formula against the law it claims to describe: sample Haar
    /// directions and count how many land inside a cap.  This is the only test
    /// that would catch a wrong *exponent*, which no closed form at `m ≤ 4`
    /// can see.
    #[test]
    fn cap_mass_matches_monte_carlo() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let draws = 60_000usize;
        for m in [5usize, 9, 23] {
            for d in [0.35f64, 0.8] {
                let mut inside = 0usize;
                for _ in 0..draws {
                    let u: Vec<f64> = (0..m)
                        .map(|_| rng.sample::<f64, _>(StandardNormal))
                        .collect();
                    let n = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                    // the first axis is the cap's centre
                    if (u[0] / n).acos() <= d {
                        inside += 1;
                    }
                }
                let hit = inside as f64 / draws as f64;
                let want = ln_cap_fraction(m, d).exp();
                let se = (want * (1.0 - want) / draws as f64).sqrt();
                assert!(
                    (hit - want).abs() < 4.0 * se.max(1e-12),
                    "m={m} d={d}: MC {hit} vs formula {want} (4 se = {})",
                    4.0 * se
                );
            }
        }
    }

    /// Monotone the way the geometry is: smaller caps, and more dimensions, both
    /// cost mass.
    #[test]
    fn cap_mass_is_monotone_in_dimension_and_radius() {
        let mut prev = 0.0;
        for m in 2..=60 {
            let f = ln_cap_fraction(m, 0.05);
            assert!(f < prev, "m={m} is not cheaper than m={}", m - 1);
            prev = f;
        }
        let mut prev = f64::NEG_INFINITY;
        for k in 1..=200 {
            let d = k as f64 * 0.01;
            let f = ln_cap_fraction(9, d);
            assert!(f > prev, "d={d} is not cheaper than the radius below it");
            prev = f;
        }
    }

    /// The published configuration: `P = 16 672`, a 1 mrad cap.  The cost is
    /// `≈ 1.15 × 10⁵` nats — three orders above anything a 512-example data set
    /// can pay (`N·ΔL ≲ 2 500` nats).
    #[test]
    fn published_configuration_costs_five_figures_of_nats() {
        let cost = cap_cost(16_672, 1e-3);
        assert!(
            (1.0e5..1.2e5).contains(&cost),
            "1 mrad at P=16672 should cost ~1.15e5 nats, got {cost}"
        );
        // and the leading order is right to a fraction of a percent
        let lead = 16_671.0 * (1.0 / 1e-3f64.sin()).ln() + 0.5 * (2.0 * PI * 16_671.0).ln();
        assert!(close(cost, lead, 1e-3), "exact {cost} vs leading {lead}");
    }

    /// `budget_dimension` inverts `cap_cost`, which is the property every
    /// caller relies on.  The bracket is between integers, so the tolerance is
    /// one dimension's worth of price.
    #[test]
    fn budget_dimension_inverts_the_cap_cost() {
        for (nats, d) in [(50.0, 0.04), (500.0, 0.02), (1_990.0, 0.03), (40.0, 1e-3)] {
            let m = budget_dimension(nats, d);
            let back = cap_cost(m.round().max(2.0) as usize, d);
            assert!(
                (back - nats).abs() < cost_per_direction(d) + 1.0,
                "nats={nats} d={d}: m*={m} costs {back}"
            );
            // and the linear exchange rate is within a factor of two
            let linear = 1.0 + nats / cost_per_direction(d);
            assert!(
                (0.5..2.0).contains(&(m / linear)),
                "nats={nats} d={d}: m*={m} vs linear {linear}"
            );
        }
    }

    /// The nearest-`M` radius is nearly independent of `M` (it goes as
    /// `√(ln M / P)`), which is why a 256-draw measurement can validate a
    /// 2 048-draw one.  At `P = 16 672` it sits at ~1.55 rad — the nearest of
    /// 2 048 Haar directions is still **~88° away** from a given point, so the
    /// status doc's "1.0–1.6 rad" row is a bucket label and the real statement
    /// is sharper than it reads.
    #[test]
    fn nearest_expected_radius_barely_moves_with_the_sample_size() {
        let a = nearest_expected_radius(16_672, 256);
        let b = nearest_expected_radius(16_672, 2_048);
        assert!((1.50..1.57).contains(&a), "256 draws: {a}");
        assert!((1.50..1.57).contains(&b), "2 048 draws: {b}");
        assert!(b < a, "more draws must get closer, {b} < {a}");
    }

    /// A vanishing cap has no mass; the whole sphere has all of it.  Both ends
    /// are where a log-space implementation is most likely to produce `NaN`.
    #[test]
    fn degenerate_radii_are_not_panic() {
        assert_eq!(ln_cap_fraction(7, 0.0), f64::NEG_INFINITY);
        assert_eq!(ln_cap_fraction(7, PI), 0.0);
        assert!(cap_cost(16_672, 1e-300) > 0.0);
        assert!(cap_cost(16_672, PI * 0.999).abs() < 1e-12);
        assert!((ln_gamma(0.5) - PI.sqrt().ln()).abs() < 1e-12);
        // ln Γ(12) = ln 11! = ln(39 916 800)
        assert!((ln_gamma(12.0) - 39_916_800f64.ln()).abs() < 1e-12);
    }
}
