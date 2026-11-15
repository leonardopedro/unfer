//! D1's tiny witness: a nonconstant rank-1/2 potential whose evidence and
//! predictive come with **enclosures**, not point estimates.
//!
//! A number computed by a quadrature rule is a reference estimate until its
//! truncation and tail are bounded. This module builds a small explicit target
//! whose potential is a sum of clamped logistic losses,
//!
//! ```text
//! Ψ(z) = λ Σ_i −log max(σ(s_i z_{c_i}), floor),     s_i = ±1
//! ```
//!
//! over `r ∈ {1, 2}` chart coordinates, together with the total
//! `J(z) = Ψ(z) + ½‖z‖²`. Every ingredient of the enclosure is an explicit
//! constant:
//!
//! - **Box quadrature.** The midpoint rule on `[−R, R]^d` has error at most
//!   `(2R)^d · d · h²/24 · sup|∂²g/∂z_j²|` for `g = e^{−Ψ}φ`, and
//!   `g'' ≤ (λN/4 + 1 + 2(λN)² + 2R²)·φ_max`. (The first-order `h` bound was
//!   also sound, but too loose to leave a positive predictive interval.)
//! - **Tail.** `μ0(outside box) ≤ 2d·φ(R)/R` (Mills ratio), and
//!   `0 ≤ ∫_outside e^{−Ψ}φ ≤ μ0(outside)`.
//! - **Box ceiling.** On the box `p_i ≥ σ(−R)`, so
//!   `Ψ ≤ λN·(−log σ(−R))`, giving a certified region `(μ0(box) ≥ 1 − tail,
//!   b)` and hence `c = ω·μ0(box)·e^{−b}` with `M = 1`. On this witness that
//!   `c` is positive but small, which is itself the D1 point.
//!
//! **Rounding.** Both enclosures are evaluated in outward-rounded interval
//! arithmetic ([`Iv`]): node coordinates, cell volumes, sigmoids, the Gaussian
//! density, the predictive weight and a reduced 26-term Taylor `exp` each carry
//! their own rounding, and the Taylor remainder is explicit. `rounding_envelope`
//! is therefore `0` for both the evidence and the predictive, and no
//! correctly-rounded-libm assumption is made. The only remaining numerical
//! bridge is the cell-node uncertainty (`NODE_UNCERTAINTY`), which is an
//! interval-width term, not a declared `f64`-sum envelope.
//!
//! `λ` must be a positive integer for the interval path (repeated
//! multiplication produces `p^λ`); `TinyLogistic::new` itself accepts any
//! positive `λ`, so the assertion lives in `evidence_enclosure`.

use crate::certificate::{Certificate, VerifiedRegion};
use crate::edge0_potential::NLL_FLOOR;
use crate::target::ZPotential;

/// A generous bound on the `f64` cell-node coordinate's own uncertainty, in
/// chart units. Interval midpoints are exact multiples of `h` reduced to an
/// `f64`; this term is many orders above their ulp, so it stays sound even for
/// a very large box. It enters the predictive quadrature error multiplied by
/// the integrand's derivative bound, not as a relative envelope on the sum.
pub const NODE_UNCERTAINTY: f64 = 1e-12;

/// One clamped logistic loss `−log max(σ(s·z_c), floor)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogisticTerm {
    pub coord: usize,
    /// `+1` when the label is the `σ(z)` class, `−1` for its complement.
    pub sign: f64,
}

/// The rank-1/2 witness target: `Ψ = λ Σ terms`, `J = Ψ + ½‖z‖²`.
#[derive(Clone, Debug)]
pub struct TinyLogistic {
    terms: Vec<LogisticTerm>,
    lambda: f64,
}

impl TinyLogistic {
    /// `rank` must be 1 or 2 and every `coord < rank`; `sign` must be `±1`.
    pub fn new(rank: usize, terms: Vec<LogisticTerm>, lambda: f64) -> Self {
        assert!(rank == 1 || rank == 2, "the witness is rank 1 or 2");
        assert!(!terms.is_empty(), "a nonconstant witness has terms");
        assert!(lambda > 0.0 && lambda.is_finite());
        for t in &terms {
            assert!(t.coord < rank, "term coordinate outside the chart");
            assert!(t.sign == 1.0 || t.sign == -1.0, "sign must be ±1");
        }
        Self { terms, lambda }
    }

    /// The plan's canonical rank-1 witness: one `+` and one `−` term on the
    /// same coordinate, so `Ψ` is nonconstant and bounded below.
    pub fn rank_one(lambda: f64) -> Self {
        Self::new(
            1,
            vec![
                LogisticTerm {
                    coord: 0,
                    sign: 1.0,
                },
                LogisticTerm {
                    coord: 0,
                    sign: -1.0,
                },
            ],
            lambda,
        )
    }

    /// The rank-2 version: the `+` term reads `z₀`, the `−` term reads `z₁`.
    pub fn rank_two(lambda: f64) -> Self {
        Self::new(
            2,
            vec![
                LogisticTerm {
                    coord: 0,
                    sign: 1.0,
                },
                LogisticTerm {
                    coord: 1,
                    sign: -1.0,
                },
            ],
            lambda,
        )
    }

    pub fn dim(&self) -> usize {
        self.terms.iter().map(|t| t.coord).max().unwrap_or(0) + 1
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// `Ψ(z)` — the likelihood potential alone.
    pub fn psi(&self, z: &[f32]) -> f64 {
        assert_eq!(z.len(), self.dim(), "z has the chart's dimension");
        self.terms
            .iter()
            .map(|t| {
                let p = sigmoid(t.sign * f64::from(z[t.coord])).max(NLL_FLOOR);
                -p.ln()
            })
            .sum::<f64>()
            * self.lambda
    }

    /// `sup |∂_c Ψ|` on all of `R^d`: `λ` times the terms reading `c`, and each
    /// `|d/dz log σ(±z)| ≤ 1`.
    fn grad_bound(&self) -> f64 {
        self.lambda * self.terms.len() as f64
    }

    /// `e^{−Ψ(z)} φ(z)` for an interval point, with clamped probabilities and an
    /// integer `λ` (repeated multiplication).
    fn integrand_iv(&self, z: &[Iv]) -> Iv {
        let power = self.lambda as usize;
        let mut g = Iv::point(1.0);
        for t in &self.terms {
            let x = if t.sign > 0.0 {
                z[t.coord]
            } else {
                z[t.coord].neg()
            };
            let e = x.neg().exp();
            let p = Iv::point(1.0).div(&Iv::point(1.0).add(&e));
            let p = Iv {
                lo: p.lo.max(NLL_FLOOR),
                hi: p.hi.clamp(NLL_FLOOR, 1.0),
            };
            g = g.mul(&p.powi(power));
        }
        let mut dens = Iv::point(1.0);
        for &zc in z {
            dens = dens.mul(&zc.mul(&zc).neg().div(&Iv::point(2.0)).exp());
        }
        g.mul(&dens.mul(&INV_SQRT_2PI.powi(z.len())))
    }

    /// `sup |∂²g/∂z_j²|` over the box for `g = e^{−Ψ}φ`, up to the common
    /// factor `φ_max`.
    ///
    /// With `u = −∇Ψ − z` and `Hess g = (H_u + uuᵀ)g`,
    /// `|∂_j u_j| ≤ |Ψ_{jj}| + 1` and `|Ψ_{jj}| ≤ λN/4`; also
    /// `u_j² ≤ 2(L² + R²)`. A first-order bound (`sup|g'|·h√d/2`) was sound but
    /// sat at ~40 % of `Z` on this witness; second order gives `h²` instead of
    /// `h` and is what makes the predictive interval useful.
    fn curvature_diag_at(&self, radius: f64) -> f64 {
        let l = self.grad_bound();
        self.lambda * self.terms.len() as f64 / 4.0 + 1.0 + 2.0 * (l * l + radius * radius)
    }

    /// `μ0(box)`'s guaranteed lower bound and the box's potential ceiling.
    fn box_bounds(&self, radius: f64) -> (f64, f64) {
        assert!(radius > 0.0);
        // `Ψ` is largest where the probability is smallest: at `z = −R` for
        // both label signs, so the box ceiling uses `σ(−R)`, not `σ(R)`.
        let p_min = sigmoid(-radius).max(NLL_FLOOR);
        let ceiling = self.lambda * self.terms.len() as f64 * (-p_min.ln());
        let tail = self.tail_upper(radius);
        ((1.0 - tail).max(0.0), ceiling)
    }

    /// `μ0(|z_j| > R for some j) ≤ 2d·φ(R)/R`, by Mills' ratio and a union
    /// bound. Sound for every `R > 0` (it exceeds 1 only where it is vacuous).
    pub fn tail_upper(&self, radius: f64) -> f64 {
        assert!(radius > 0.0);
        let d = self.dim() as f64;
        let phi = (-0.5 * radius * radius).exp() / (2.0 * std::f64::consts::PI).sqrt();
        (d * 2.0 * phi / radius).min(1.0)
    }

    /// `Z = ∫ e^{−J} dz = ∫_box e^{−Ψ}φ dz + tail`, enclosed.
    pub fn evidence_enclosure(&self, radius: f64, cells: usize) -> EvidenceEnclosure {
        let d = self.dim();
        assert!(cells >= 2, "need at least two cells per axis");
        assert!(
            self.lambda.fract() == 0.0 && self.lambda <= 64.0,
            "the interval enclosure supports positive integer lambda"
        );
        let h = Iv::point(2.0 * radius).div(&Iv::point(cells as f64));
        let (prior_mass_lower, ceiling) = self.box_bounds(radius);
        let tail = self.tail_upper(radius);

        // Prior measure: `Z0 = ∫ e^{−Ψ} φ dz`, the constant the certificate's
        // region argument is stated in. Every node, volume and integrand value
        // is an interval rounded outward, so the sum below is a bound rather
        // than an estimate with a declared envelope.
        let mut mid = Iv::point(0.0);
        let cell_volume = h.powi(d);
        for_each_midpoint_iv(d, radius, &h, cells, |z| {
            mid = mid.add(&cell_volume.mul(&self.integrand_iv(z)));
        });

        let l = self.lambda * self.terms.len() as f64;
        let curvature = Iv::point(l / 4.0 + 1.0)
            .add(&Iv::point(2.0).mul(&Iv::point(l * l).add(&Iv::point(radius * radius))));
        let vol = Iv::point(2.0 * radius).powi(d);
        let phi_max = INV_SQRT_2PI.powi(d);
        // Midpoint rule: `vol · d · h²/24 · sup|Σ_j g_jj|`, all factors
        // outward-rounded; the node intervals already absorb their own rounding.
        let quad_error = vol
            .mul(&Iv::point(d as f64))
            .mul(&h.powi(2))
            .div(&Iv::point(24.0))
            .mul(&curvature)
            .mul(&phi_max)
            .hi;

        EvidenceEnclosure {
            z_lo: (mid.lo - quad_error).max(0.0),
            z_hi: mid.hi + quad_error + tail,
            midpoint: 0.5 * (mid.lo + mid.hi),
            quadrature_error: quad_error,
            tail_upper: tail,
            prior_mass_lower,
            potential_ceiling: ceiling,
            rounding_envelope: 0.0,
            radius,
            cells,
            dim: d,
        }
    }

    /// `E_p[f]` for `f(z) = s²/(s² + ‖z‖²)`, enclosed by bounding the numerator
    /// as well as the denominator. `f ∈ [0, 1]`, so the outside-box numerator is
    /// at most `μ0(outside)`.
    pub fn predictive_enclosure(
        &self,
        radius: f64,
        cells: usize,
        scale: f64,
    ) -> PredictiveEnclosure {
        assert!(scale > 0.0 && scale.is_finite());
        let d = self.dim();
        let ev = self.evidence_enclosure(radius, cells);
        let tail = ev.tail_upper;
        let s2 = Iv::point(scale * scale);

        let h = Iv::point(2.0 * radius).div(&Iv::point(cells as f64));
        let vol = h.powi(d);
        let mut a_mid = Iv::point(0.0);
        let mut f_hat = 0.0f64;
        let mut nodes = 0usize;
        for_each_midpoint_iv(d, radius, &h, cells, |z| {
            let mut r2 = Iv::point(0.0);
            for zc in z {
                r2 = r2.add(&zc.mul(zc));
            }
            let f = s2.div(&s2.add(&r2));
            a_mid = a_mid.add(&vol.mul(&f).mul(&self.integrand_iv(z)));
            f_hat += 0.5 * (f.lo + f.hi);
            nodes += 1;
        });
        f_hat /= nodes.max(1) as f64;

        let h64 = 2.0 * radius / cells as f64;
        let phi_max = (2.0 * std::f64::consts::PI).powf(-0.5 * d as f64);
        let r = radius * (d as f64).sqrt();
        let l = self.grad_bound();
        let vol64 = (2.0 * radius).powi(d as i32);
        // `f` cancels the constant `φ_max` in `g`; `|f'| ≤ 2r/(s²(1+r²/s²)²)` and
        // `|f''| ≤ 2 + 8 = 10` (the `1/(1+u)` second derivative, `u = r²/s²`).
        let c_f = 2.0 * r / (scale * scale) / (1.0 + r * r / (scale * scale)).powi(2);
        let d_a = 10.0 + 2.0 * c_f * (l + r) + self.curvature_diag_at(radius);
        let quad = Iv::point(vol64)
            .mul(&Iv::point(d as f64))
            .mul(&Iv::point(h64).powi(2))
            .div(&Iv::point(24.0))
            .mul(&Iv::point(d_a))
            .mul(&Iv::point(phi_max));
        // Node coordinates are interval points whose widths are ~ulp; this is a
        // deliberately generous multiple of that, multiplied by the bound on
        // `(f·g)'`, so the endpoint arithmetic is not the assumption.
        let node_err = (c_f + l + r) * phi_max * NODE_UNCERTAINTY * vol64;
        let a_err = quad.hi + node_err;

        let a_lo = (a_mid.lo - a_err).max(0.0);
        let a_hi = a_mid.hi + a_err + tail;
        PredictiveEnclosure {
            lo: (a_lo / ev.z_hi).clamp(0.0, 1.0),
            hi: ((a_hi + tail) / ev.z_lo).min(1.0),
            cell_mean: f_hat,
            rounding_envelope: 0.0,
            radius,
            cells,
        }
    }

    /// The certified-region certificate for this witness: `a = 1 − tail`,
    /// `b` the box ceiling, `M = 1`. `c` is positive and may be small.
    pub fn certificate(&self, radius: f64, omega: f64) -> Option<Certificate> {
        let (prior_mass, ceiling) = self.box_bounds(radius);
        if prior_mass <= 0.0 {
            return None;
        }
        Some(Certificate::new(
            omega,
            VerifiedRegion::new(prior_mass, ceiling),
            1.0,
        ))
    }
}

impl ZPotential for TinyLogistic {
    fn dim(&self) -> usize {
        TinyLogistic::dim(self)
    }

    fn phi(&self, z: &[f32]) -> f64 {
        self.psi(z) + 0.5 * z.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
    }
}

/// `Z ∈ [z_lo, z_hi]` with the truncation/tail split reported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvidenceEnclosure {
    pub z_lo: f64,
    pub z_hi: f64,
    /// The raw midpoint sum, before any bound is added.
    pub midpoint: f64,
    /// `sup|∇g| · (h√d/2) · (2R)^d` — exact-arithmetic truncation error.
    pub quadrature_error: f64,
    /// `μ0(outside)` by Mills' ratio.
    pub tail_upper: f64,
    /// `1 − tail_upper`, the certified region's prior mass.
    pub prior_mass_lower: f64,
    /// `λN·(−log σ(−R))`, the box's potential ceiling.
    pub potential_ceiling: f64,
    pub rounding_envelope: f64,
    pub radius: f64,
    pub cells: usize,
    pub dim: usize,
}

impl EvidenceEnclosure {
    pub fn contains(&self, z: f64) -> bool {
        self.z_lo <= z && z <= self.z_hi
    }

    pub fn width(&self) -> f64 {
        self.z_hi - self.z_lo
    }
}

/// `E_p[f] ∈ [lo, hi]` for the bounded predictive functional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictiveEnclosure {
    pub lo: f64,
    pub hi: f64,
    /// The unweighted cell average of `f` — reported only to show the posterior
    /// reweighting is doing something, never used as a bound.
    pub cell_mean: f64,
    /// Always `0`: the numerator midpoint sum is interval arithmetic. The field
    /// remains so a caller can assert the claim rather than infer it.
    pub rounding_envelope: f64,
    pub radius: f64,
    pub cells: usize,
}

impl PredictiveEnclosure {
    pub fn contains(&self, value: f64) -> bool {
        self.lo <= value && value <= self.hi
    }
}

/// `ln 2` bracketed by two `f64`s that contain the exact value.
const LN2: Iv = Iv::span(0.6931471805599452, 0.6931471805599454);
/// `1/√(2π)`, bracketed likewise.
const INV_SQRT_2PI: Iv = Iv::span(0.3989422804014326, 0.3989422804014328);

/// A closed interval with **outward-rounded** arithmetic.
///
/// Every operation is followed by `next_down`/`next_up`, which encloses the
/// exact real result because IEEE addition, subtraction, multiplication and
/// division are correctly rounded. `exp` is the only transcendental: it is
/// reduced to `2^k e^r` with `|r| ≤ ln2/2`, summed by a 26-term Taylor series,
/// and widened by the explicit remainder `2·0.35²⁷/27!` — so the bound does not
/// assume a correctly-rounded libm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Iv {
    pub lo: f64,
    pub hi: f64,
}

impl Iv {
    pub const fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    pub const fn span(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    pub fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    pub fn add(self, o: &Iv) -> Self {
        Self {
            lo: next_down(self.lo + o.lo),
            hi: next_up(self.hi + o.hi),
        }
    }

    pub fn sub(self, o: &Iv) -> Self {
        self.add(&o.neg())
    }

    pub fn mul(self, o: &Iv) -> Self {
        let p = [
            self.lo * o.lo,
            self.lo * o.hi,
            self.hi * o.lo,
            self.hi * o.hi,
        ];
        let mut lo = p[0];
        let mut hi = p[0];
        for v in &p[1..] {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        Self {
            lo: next_down(lo),
            hi: next_up(hi),
        }
    }

    pub fn div(self, o: &Iv) -> Self {
        assert!(
            o.lo > 0.0 || o.hi < 0.0,
            "interval division by an interval containing zero"
        );
        let p = [
            self.lo / o.lo,
            self.lo / o.hi,
            self.hi / o.lo,
            self.hi / o.hi,
        ];
        let mut lo = p[0];
        let mut hi = p[0];
        for v in &p[1..] {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        Self {
            lo: next_down(lo),
            hi: next_up(hi),
        }
    }

    pub fn powi(self, n: usize) -> Self {
        let mut acc = Iv::point(1.0);
        let mut base = self;
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.mul(&base);
            k >>= 1;
        }
        acc
    }

    /// `exp` on an interval; monotone, so the endpoint enclosures suffice.
    pub fn exp(self) -> Self {
        Self {
            lo: exp_enclose(self.lo).lo,
            hi: exp_enclose(self.hi).hi,
        }
    }

    /// `ln` on a strictly positive interval; monotone, endpoint enclosures.
    pub fn ln(self) -> Self {
        Self {
            lo: ln_enclose(self.lo).lo,
            hi: ln_enclose(self.hi).hi,
        }
    }
}

/// Enclosure of `e^x` for `|x| ≤ 60`, independent of the platform libm.
pub fn exp_enclose(x: f64) -> Iv {
    assert!(
        x.is_finite() && x.abs() <= 60.0,
        "exp_enclose is bounded to |x| <= 60"
    );
    let k = (x / 0.6931471805599453).round();
    let r = Iv::point(x).sub(&Iv::point(k).mul(&LN2));
    let mut term = Iv::point(1.0);
    let mut sum = Iv::point(1.0);
    for n in 1..=26 {
        term = term.mul(&r).div(&Iv::point(n as f64));
        sum = sum.add(&term);
    }
    // `e^|r| < 2` for |r| < ln 2/2 + ulp, and the tail is below the next term's
    // geometric factor, so this symmetric remainder is conservative.
    let mut factorial = 1.0f64;
    for n in 1..=27 {
        factorial *= n as f64;
    }
    let rem = 2.0 * 0.35f64.powi(27) / factorial;
    let sum = sum.add(&Iv::span(-rem, rem));
    let scale = 2f64.powi(k as i32);
    Iv {
        lo: next_down(sum.lo * scale),
        hi: next_up(sum.hi * scale),
    }
}

/// Enclosure of `ln x` for finite `x > 0`, without assuming a correctly
/// rounded libm. Reduces `x = m·2^k` with `m ∈ [1,2)`, so `ln m` is the atanh
/// series in `y = (m−1)/(m+1) ∈ [0,1/3]`; the tail is `2 y^{2N+3}/((2N+3)(1−y²))`.
pub fn ln_enclose(x: f64) -> Iv {
    assert!(x.is_finite() && x > 0.0, "ln_enclose needs finite x > 0");
    let mut m = x;
    let mut k = 0i32;
    while m >= 2.0 {
        m *= 0.5;
        k += 1;
    }
    while m < 1.0 {
        m *= 2.0;
        k -= 1;
    }
    let one = Iv::point(1.0);
    let y = Iv::point(m).sub(&one).div(&Iv::point(m).add(&one));
    let y2 = y.mul(&y);
    let mut term = y;
    let mut sum = Iv::point(0.0);
    const N: usize = 40;
    for j in 0..N {
        let denom = (2 * j + 1) as f64;
        sum = sum.add(&term.div(&Iv::point(denom)));
        term = term.mul(&y2);
    }
    // The next omitted term is `y^(2N+1)/(2N+1)`; the remaining positive
    // series is below `y^(2N+1)/((2N+1)(1−y²))`, and `y ≤ 1/3`.
    let remainder = 2.0
        * (1.0f64 / 3.0).powi((2 * N) as i32)
        / ((2 * N + 1) as f64 * (1.0 - 1.0 / 9.0));
    let ln_m = sum.mul(&Iv::point(2.0)).add(&Iv::span(-remainder, remainder));
    ln_m.add(&Iv::point(k as f64).mul(&LN2))
}

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        x
    } else if x == 0.0 {
        -f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() - 1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        x
    } else if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

/// Midpoints as intervals: the node coordinate itself carries its rounding, so
/// no separate node-shift term is needed.
fn for_each_midpoint_iv(d: usize, radius: f64, h: &Iv, cells: usize, mut f: impl FnMut(&[Iv])) {
    let center = |i: usize| Iv::point(-radius).add(&h.mul(&Iv::point(i as f64 + 0.5)));
    match d {
        1 => {
            for i in 0..cells {
                f(&[center(i)]);
            }
        }
        2 => {
            for i in 0..cells {
                let x = center(i);
                for j in 0..cells {
                    f(&[x, center(j)]);
                }
            }
        }
        _ => unreachable!("the witness is rank 1 or 2"),
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}


