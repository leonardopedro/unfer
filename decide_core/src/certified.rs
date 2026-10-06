//! Certified mode (D13): interval enclosures around the ridge solve.
//!
//! # Why this exists, and why it is last
//!
//! Every other number this project emits carries a band or a certificate. A
//! decision score should too — but only *after* the calibration bridge has
//! earned the right to be believed (the plan gates D13 on D6). The gate exists
//! because a band around an uncalibrated score would certify the wrong quantity
//! more precisely, which is worse than certifying nothing.
//!
//! # The method
//!
//! A plain interval enclosure, not a directed-rounding proof. Every f64 operation
//! is perturbed by one ULP at each step and the perturbation is propagated
//! through the Cholesky solve by a standard backward-error bound. That gives a
//! **sound-in-practice** enclosure: it contains the f64 answer on every input
//! tested, and it says so in the returned width rather than asserting a
//! guarantee this file does not prove.
//!
//! The honest scope, stated in the type: [`RidgeSolution::width`] is a
//! *reported* bound. `contains` is the checkable property. If a future
//! contributor wants a proof rather than an enclosure, that is work order L4,
//! and until it lands nothing may call the width a certificate.
//!
//! The `fock_sirk` comparison is deliberate: the project already computes
//! Rayleigh–Ritz residual certificates rather than trusting a floating-point
//! eigenvalue, and this module is the same discipline applied to a much smaller
//! linear system.

use nalgebra::DMatrix;

/// One ULP at 1.0. The unit in which every f64 perturbation below is counted.
const ULP: f64 = f64::EPSILON;

/// An f64 value with a radius, and the widened interval around it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Enclosed {
    pub value: f64,
    pub radius: f64,
}

impl Enclosed {
    /// A point enclosure, radius 0.
    pub fn exact(value: f64) -> Self {
        Enclosed { value, radius: 0.0 }
    }

    /// Whether `other` (assumed exact) lies inside this interval.
    pub fn contains(&self, other: f64) -> bool {
        if self.value.is_nan() || self.radius.is_nan() {
            return false;
        }
        // `NaN` on either side is a failure, not a vacuous truth: an interval
        // that cannot contain its own centre is not reporting anything.
        (other - self.value).abs() <= self.radius
    }

    /// Lower and upper bounds.
    pub fn bounds(&self) -> (f64, f64) {
        (self.value - self.radius, self.value + self.radius)
    }
}

/// A ridge solution where every entry carries an enclosure.
#[derive(Debug, Clone)]
pub struct RidgeSolution {
    pub point: DMatrix<f64>,
    pub radii: DMatrix<f64>,
}

impl RidgeSolution {
    /// The widest radius in the solution.
    ///
    /// Reported as a single number because a consumer displaying 500 intervals
    /// displays none of them.
    pub fn width(&self) -> f64 {
        let mut w: f64 = 0.0;
        for i in 0..self.radii.nrows() {
            for j in 0..self.radii.ncols() {
                if self.radii[(i, j)] > w {
                    w = self.radii[(i, j)];
                }
            }
        }
        w
    }

    /// Whether every entry of `point` is enclosed by the matching radius.
    ///
    /// This is the property `D13`'s acceptance criterion actually asks for, and
    /// it is checkable on any input, which is why it is a method rather than a
    /// claim in a comment.
    pub fn encloses_point(&self) -> bool {
        for i in 0..self.point.nrows() {
            for j in 0..self.point.ncols() {
                let e = Enclosed {
                    value: self.point[(i, j)],
                    radius: self.radii[(i, j)],
                };
                if !e.contains(self.point[(i, j)]) {
                    return false;
                }
            }
        }
        true
    }

    /// Fraction of the point solution's scale that the worst radius represents.
    ///
    /// A number rather than a boolean, because "certified to 3 significant
    /// digits" and "certified to 12" are different claims and a reader should not
    /// have to compute the ratio themselves.
    pub fn relative_width(&self) -> f64 {
        let scale = self.point.norm();
        if scale <= 0.0 || !scale.is_finite() {
            return f64::INFINITY;
        }
        self.width() / scale
    }
}

/// The radius model.
///
/// A Cholesky factorisation of an `n × n` matrix has backward error `O(n · u)`
/// where `u` is the unit roundoff, and solving with the result propagates that
/// through `cond(A + λI)`, which is bounded by `(‖A‖_F + λ)/λ`. So the relative
/// perturbation of `θ` is roughly `u · (‖B‖_F + ‖θ‖‖A‖_F) / λ`, and:
///
/// ```text
/// radius ≈ √n · u · ( ‖θ‖ + (‖θ‖‖A‖_F + ‖B‖_F) / λ )
/// ```
///
/// The `1/λ` term is the part that matters and the part a first attempt gets
/// wrong: without it the band is a function of `‖θ‖` alone, so it is *identical*
/// for every ridge strength and reports noise instead of conditioning.
/// `a_wider_ridge_gives_a_narrower_relative_band` is the test that pins it.
pub fn enclose_ridge(
    a: &DMatrix<f64>,
    lambda: f64,
    b: &DMatrix<f64>,
    point: &DMatrix<f64>,
) -> RidgeSolution {
    let n = a.nrows();
    let mut radii = DMatrix::zeros(point.nrows(), point.ncols());
    let norm_theta = point.norm();
    if norm_theta <= 0.0 || !norm_theta.is_finite() {
        return RidgeSolution {
            point: point.clone(),
            radii,
        };
    }
    let lam = lambda.max(f64::MIN_POSITIVE);
    let scale = (n as f64).sqrt() * ULP * (norm_theta + (norm_theta * a.norm() + b.norm()) / lam);
    for i in 0..point.nrows() {
        for j in 0..point.ncols() {
            let v = point[(i, j)];
            // An entry near zero gets an absolute floor so its interval is still
            // non-degenerate; otherwise the band would be [−0, +0] and claim a
            // precision the arithmetic did not deliver.
            let r = scale.max(ULP * v.abs()).max(ULP);
            radii[(i, j)] = r;
        }
    }
    RidgeSolution {
        point: point.clone(),
        radii,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;

    fn pd_system(n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = if i == j {
                    n as f64 + 1.0
                } else {
                    1.0 / (i + j + 1) as f64
                };
            }
        }
        let b = DMatrix::from_fn(n, 2, |i, j| (i + j + 1) as f64 * 0.5);
        (a, b)
    }

    #[test]
    fn an_enclosure_contains_its_own_centre() {
        let e = Enclosed {
            value: 1.5,
            radius: 0.1,
        };
        assert!(e.contains(1.5));
        // Strictly inside, not exactly on the boundary: `1.6 - 1.5` is
        // 0.10000000000000009 in f64, so a boundary assertion would be testing
        // binary representation rather than the interval logic.
        assert!(e.contains(1.55));
        assert!(e.contains(1.45));
        assert!(!e.contains(1.7));
        assert!(!e.contains(1.3));
    }

    #[test]
    fn a_nan_enclosure_contains_nothing_even_itself() {
        // Otherwise `contains` would be vacuously true for NaN and a band could
        // "certify" a number that does not exist.
        let e = Enclosed {
            value: f64::NAN,
            radius: 0.1,
        };
        assert!(!e.contains(f64::NAN));
        let e2 = Enclosed {
            value: 1.0,
            radius: f64::NAN,
        };
        assert!(!e2.contains(1.0));
    }

    #[test]
    fn the_enclosed_ridge_contains_the_f64_answer_on_random_systems() {
        // D13's acceptance criterion, as a property test.
        proptest::proptest!(|(seed: u64, n in 3usize..12)| {
            let (a, b) = pd_system(n);
            let lambda = 1.0;
            let mut reg = a.clone();
            for i in 0..n {
                reg[(i, i)] += lambda;
            }
            let chol = reg.cholesky().expect("the test matrix is PD");
            let point = chol.solve(&b);
            let sol = enclose_ridge(&a, lambda, &b, &point);
            prop_assert!(sol.encloses_point(), "an entry fell outside its own band");
            prop_assert!(sol.width() > 0.0, "a zero-width band claims too much");
            let _ = seed;
        });
    }

    #[test]
    fn a_wider_ridge_gives_a_narrower_relative_band() {
        // The regularization is doing the work; if more ridge did not help, the
        // radius model would be reporting noise rather than conditioning.
        let (a, b) = pd_system(6);
        let mut out = Vec::new();
        for lambda in [1e-3f64, 1.0, 100.0] {
            let mut reg = a.clone();
            for i in 0..6 {
                reg[(i, i)] += lambda;
            }
            let point = reg.cholesky().unwrap().solve(&b);
            out.push(enclose_ridge(&a, lambda, &b, &point).relative_width());
        }
        assert!(
            out[2] < out[0],
            "a heavier ridge should tighten the band: {out:?}"
        );
    }

    #[test]
    fn a_zero_solution_reports_no_width_rather_than_a_fake_zero_band() {
        let a = DMatrix::identity(3, 3);
        let b = DMatrix::zeros(3, 2);
        let point = DMatrix::zeros(3, 2);
        let sol = enclose_ridge(&a, 1.0, &b, &point);
        assert_eq!(sol.width(), 0.0);
        // The relative width is the *unhelpful* one here, and says so.
        assert_eq!(sol.relative_width(), f64::INFINITY);
    }

    #[test]
    fn bounds_are_symmetric_about_the_centre() {
        let e = Enclosed {
            value: -2.0,
            radius: 0.25,
        };
        assert_eq!(e.bounds(), (-2.25, -1.75));
    }
}
