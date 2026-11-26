//! N8: finite-map defect certificate for the compressed rational flow.
//!
//! For an isometric basis `V`, reduced generator `G` and full columns `HV`,
//! define `D = i V G - H V`. Then
//!
//! ```text
//! ||exp(-itH) psi - V exp(tG) V* psi||
//!   <= ||psi - V V* psi|| + integral_0^t ||D exp(sG) V* psi|| ds
//!   <= leakage + t ||D|| ||V* psi|| exp(omega_+ t),
//! ```
//!
//! where `omega_+ = max(0, lambda_max(Re G))` covers the returned inexact
//! generator. The `f64` path is a measured estimate. [`interval_terms`] repeats
//! it with directed `f64` endpoints, encloses `G = V* H V` from the declared
//! `H` and `V`, and applies a Gershgorin bound to the enclosed Hermitian part.
//! Its `hi` endpoint is therefore a directed bound on the declared finite
//! matrices, but still not a certified enclosure of a physical reduction. The
//! separately measured map error uses nalgebra's Padé exponential and is only a
//! diagnostic witness. Neither path is an unconditional physical-SM certificate:
//! the self-adjoint operator, the physical reduction, and the matrix-specific
//! defect bound remain assumptions.

use crate::sm_bridge::{bind_artifact, canonical_json_sha256, reported_residual, sha256_hex};
use nalgebra::{DMatrix, DVector};
use nested_fock_algebra::exact_inner::NestedVector;
use num_complex::Complex64 as C;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

pub const DEFECT_SCHEMA: &str = "sm-defect-certificate/v1";

const SM_DEFECT_PROOF: &[u8] = include_bytes!("../../docs/fixtures/sm_defect_proof.ndjson");

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

/// Round down, but keep an exact zero exact. Without the zero case, summing
/// nonnegative terms would make a lower endpoint `-denorm_min`, which reads as a
/// negative leakage or norm.
fn down(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.next_down() }
}

fn up(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.next_up() }
}

fn add_down(a: f64, b: f64) -> f64 {
    down(a + b)
}

fn add_up(a: f64, b: f64) -> f64 {
    up(a + b)
}

fn mul_down(a: f64, b: f64) -> f64 {
    down(a * b)
}

fn mul_up(a: f64, b: f64) -> f64 {
    up(a * b)
}

// An enclosed numeric carrier: these methods take `self` by value (the type is
// `Copy`) and mirror the scalar operators, so they stay inherent.
#[allow(clippy::should_implement_trait)]
impl Interval {
    pub fn point(x: f64) -> Self {
        assert!(x.is_finite(), "interval endpoints must be finite");
        Self { lo: x, hi: x }
    }
    pub fn add(self, other: Self) -> Self {
        Self {
            lo: add_down(self.lo, other.lo),
            hi: add_up(self.hi, other.hi),
        }
    }
    pub fn sub(self, other: Self) -> Self {
        Self {
            lo: down(self.lo - other.hi),
            hi: up(self.hi - other.lo),
        }
    }
    pub fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    pub fn mul(self, other: Self) -> Self {
        let lower = [
            mul_down(self.lo, other.lo),
            mul_down(self.lo, other.hi),
            mul_down(self.hi, other.lo),
            mul_down(self.hi, other.hi),
        ];
        let upper = [
            mul_up(self.lo, other.lo),
            mul_up(self.lo, other.hi),
            mul_up(self.hi, other.lo),
            mul_up(self.hi, other.hi),
        ];
        Self {
            lo: lower.iter().copied().fold(f64::INFINITY, f64::min),
            hi: upper.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
    pub fn square_upper(self) -> f64 {
        mul_up(self.lo, self.lo).max(mul_up(self.hi, self.hi))
    }
    pub fn sqrt_upper(self) -> f64 {
        up(self.hi.max(0.0).sqrt())
    }
    pub fn exp_upper(self) -> f64 {
        up(self.hi.exp())
    }
    pub fn max0(self) -> f64 {
        self.hi.max(0.0)
    }
    /// Outward-rounded `max(0, lambda_max(sym part))` is supplied by the caller.
    pub fn nonnegative(self) -> Option<Self> {
        (self.lo.is_finite() && self.hi.is_finite() && self.lo >= 0.0).then_some(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexInterval {
    pub re: Interval,
    pub im: Interval,
}

#[allow(clippy::should_implement_trait)]
impl ComplexInterval {
    pub fn point(x: C) -> Self {
        Self {
            re: Interval::point(x.re),
            im: Interval::point(x.im),
        }
    }
    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re.add(other.re),
            im: self.im.add(other.im),
        }
    }
    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re.mul(other.re).sub(self.im.mul(other.im)),
            im: self.re.mul(other.im).add(self.im.mul(other.re)),
        }
    }
    pub fn sub(self, other: Self) -> Self {
        self.add(ComplexInterval {
            re: other.re.neg(),
            im: other.im.neg(),
        })
    }
    pub fn scale(self, factor: C) -> Self {
        self.mul(Self::point(factor))
    }
    pub fn norm_square_upper(self) -> f64 {
        add_up(self.re.square_upper(), self.im.square_upper())
    }
}

/// One term per error component. `evolution_bound` is the Duhamel state bound
/// `leakage + time * defect_action_envelope`; the remaining fields stay
/// separate so a reader can see what was priced.
#[derive(Clone, Debug)]
pub struct DefectTerms {
    pub initial_leakage: f64,
    pub defect_frobenius: f64,
    /// `||D c||` for the projected seed coefficients, rather than the
    /// worst-case full-column Frobenius norm. This is a diagnostic; the
    /// certified bound still uses `defect_frobenius`.
    pub defect_seed_action: f64,
    /// Resolution-converged estimate of `sup_{0<=s<=time} ||D exp(sG) c||`,
    /// sampled along the returned trajectory rather than derived from
    /// `||D G||_F ||c|| exp(omega_+ time)`. The analytic product is deliberately
    /// **not** used: every entry of a SIRK generator `G = gamma I - B^-1`
    /// carries the shift while `exp(sG)` stays `O(1)`, so the product prices
    /// the shift (measured `||D G||_F ~ 1.5e4` against a trajectory maximum of
    /// `3.6`). Still a measured estimate, not an enclosure.
    pub defect_action_envelope: f64,
    /// Retained for the record: identical to `evolution_bound`, whose formula
    /// is now this state-dependent bound.
    pub state_weighted_bound: f64,
    pub seed_norm: f64,
    pub reduced_growth: f64,
    pub flow_growth: f64,
    pub time: f64,
    /// `leakage + time * defect_action_envelope`: bounds `||e(t)||` through the
    /// Duhamel representation with the sampled defect action.
    pub evolution_bound: f64,
    /// `leakage + time * defect_frobenius * seed_norm * flow_growth`: the
    /// conservative uniform-column bound. It is the shifted analytic product,
    /// so it prices the generator's scale as well as the defect; it is kept
    /// under its own name now that `evolution_bound` samples the trajectory,
    /// because a record that only carries the tighter estimate cannot show what
    /// the worst-case column argument gives.
    pub frobenius_product_bound: f64,
}

impl DefectTerms {
    /// The measured fields at their reported precision.
    ///
    /// `to_json` and `certificate_json` both build records from this map, so the
    /// two cannot report different digits for the same measurement. `time` is an
    /// input and stays exact; the interval endpoints elsewhere in the record are
    /// directed and are never rounded.
    fn reported_fields(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("initial_leakage", reported_residual(self.initial_leakage)),
            ("defect_frobenius", reported_residual(self.defect_frobenius)),
            (
                "defect_seed_action",
                reported_residual(self.defect_seed_action),
            ),
            (
                "defect_action_envelope",
                reported_residual(self.defect_action_envelope),
            ),
            (
                "state_weighted_bound",
                reported_residual(self.state_weighted_bound),
            ),
            ("seed_norm", reported_residual(self.seed_norm)),
            ("reduced_growth", reported_residual(self.reduced_growth)),
            ("flow_growth", reported_residual(self.flow_growth)),
            ("evolution_bound", reported_residual(self.evolution_bound)),
            (
                "frobenius_product_bound",
                reported_residual(self.frobenius_product_bound),
            ),
        ]
    }

    pub fn to_json(&self, extra: Value) -> Value {
        let mut record = Map::new();
        for (key, value) in self.reported_fields() {
            record.insert(key.into(), json!(value));
        }
        record.insert("time".into(), json!(self.time));
        record.insert("extra".into(), extra);
        Value::Object(record)
    }
}

fn add_scaled(out: &mut NestedVector, state: &NestedVector, factor: C) {
    for (basis, amplitude) in state {
        let value = out.entry(basis.clone()).or_default();
        *value += *amplitude * factor;
        if *value == C::new(0.0, 0.0) {
            out.remove(basis);
        }
    }
}

fn joint_norm(state: &NestedVector) -> f64 {
    state
        .values()
        .map(|amplitude| amplitude.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn inner(a: &NestedVector, b: &NestedVector) -> C {
    a.iter()
        .filter_map(|(basis, x)| b.get(basis).map(|y| x.conj() * y))
        .sum()
}

/// Apply `D = i V G - H V` to a reduced coefficient vector, returning the
/// full-space defect vector. This is the state-dependent object behind the
/// integral bound; [`defect_frobenius`] is its worst case over the reduced
/// coordinate directions.
pub fn defect_action(
    basis: &[NestedVector],
    full_images: &[NestedVector],
    generator: &DMatrix<C>,
    coefficients: &[C],
) -> NestedVector {
    let rank = basis.len();
    assert_eq!(full_images.len(), rank);
    assert_eq!(coefficients.len(), rank);
    assert_eq!(generator.nrows(), rank);
    assert_eq!(generator.ncols(), rank);
    let mut d = NestedVector::new();
    for column in 0..rank {
        for row in 0..rank {
            let factor = C::i() * generator[(row, column)] * coefficients[column];
            add_scaled(&mut d, &basis[row], factor);
        }
        add_scaled(&mut d, &full_images[column], -coefficients[column]);
    }
    d
}

/// `D = i V G - H V` column by column, with the full (unprojected) `H V`.
///
/// Resolution-converged estimate of `sup_{0<=s<=time} ||D exp(sG) c||`.
///
/// The estimator samples the defect trajectory and refines the grid until the
/// sampled maximum stops moving, then adds the observed last-step change as a
/// remainder. It deliberately does **not** use the analytic Lipschitz product
/// `||D G||_F ||c|| exp(omega_+ time)`: for a SIRK generator `G = gamma I - B^-1`
/// every entry of `G` carries the shift `gamma`, while `exp(sG)` is `O(1)` on
/// the interval, so that product is larger than the defect it bounds by the
/// square of the shift (measured on the phi4 continuous-momentum operator:
/// `||DG||_F ~ 1.5e4` against a sampled maximum of `3.6`). The sampled route is
/// basis-covariant, shift-free, and — unlike the product — it is bounded by
/// what the returned flow actually does.
///
/// This remains a *measured* estimate, not an enclosure, exactly as the module
/// documents for the `f64` path.
fn defect_action_envelope(
    basis: &[NestedVector],
    full_images: &[NestedVector],
    generator: &DMatrix<C>,
    coefficients: &[C],
    time: f64,
) -> f64 {
    let rank = coefficients.len();
    let seed = DVector::from_iterator(rank, coefficients.iter().copied());
    let sample = |s: f64| -> f64 {
        let evolved = (generator * C::new(s, 0.0)).exp() * &seed;
        joint_norm(&defect_action(
            basis,
            full_images,
            generator,
            evolved.as_slice(),
        ))
    };
    // Small reduced systems (the phi4 cells are rank 4) can afford a fine grid;
    // larger ones stop earlier so the certificate stays a small multiple of the
    // solve it prices.
    let max_steps = if rank <= 8 { 4096 } else { 256 };
    sampled_sup(&sample, time, max_steps)
}

/// Resolution-converged estimate of `sup_{0<=s<=time} f(s)`: double the grid
/// until the sampled maximum stops moving, then add the observed last-step
/// change as the remainder. Shared by the SM cell path and the finite fixture
/// so the two cannot drift apart.
fn sampled_sup(sample: &impl Fn(f64) -> f64, time: f64, max_steps: usize) -> f64 {
    if time <= 0.0 || time.is_nan() {
        return sample(0.0);
    }
    let tolerance = 1e-3;
    let mut steps = 8usize;
    let mut previous = 0.0f64;
    loop {
        let mut maximum = 0.0f64;
        for index in 0..=steps {
            maximum = maximum.max(sample(time * index as f64 / steps as f64));
        }
        let change = (maximum - previous).abs();
        if steps >= max_steps || (steps > 8 && change <= tolerance * (1.0 + maximum)) {
            return maximum + change;
        }
        previous = maximum;
        steps *= 2;
    }
}

pub fn defect_frobenius(
    basis: &[NestedVector],
    full_images: &[NestedVector],
    generator: &DMatrix<C>,
) -> f64 {
    let rank = basis.len();
    assert_eq!(full_images.len(), rank);
    assert_eq!(generator.nrows(), rank);
    assert_eq!(generator.ncols(), rank);
    let mut total = 0.0;
    for column in 0..rank {
        let mut d = NestedVector::new();
        for row in 0..rank {
            add_scaled(&mut d, &basis[row], C::i() * generator[(row, column)]);
        }
        add_scaled(&mut d, &full_images[column], C::new(-1.0, 0.0));
        total += d.values().map(|a| a.norm_sqr()).sum::<f64>();
    }
    total.sqrt()
}

/// Largest eigenvalue of the Hermitian part of `G` (nonzero only for the
/// returned inexact generator).
pub fn reduced_growth(generator: &DMatrix<C>) -> f64 {
    let symmetric = (generator + generator.adjoint()) * C::new(0.5, 0.0);
    symmetric
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Point (non-enclosed) defect terms for the actual returned solve.
pub fn defect_terms(
    basis: &[NestedVector],
    full_images: &[NestedVector],
    generator: &DMatrix<C>,
    state: &NestedVector,
    time: f64,
) -> DefectTerms {
    let coefficients: DVector<C> =
        DVector::from_iterator(basis.len(), basis.iter().map(|vector| inner(vector, state)));
    let mut projected = NestedVector::new();
    for (vector, coefficient) in basis.iter().zip(coefficients.iter()) {
        add_scaled(&mut projected, vector, *coefficient);
    }
    let mut residual = state.clone();
    for (basis, amplitude) in &projected {
        let value = residual.entry(basis.clone()).or_default();
        *value -= *amplitude;
    }
    residual.retain(|_, amplitude| *amplitude != C::new(0.0, 0.0));
    let leakage = joint_norm(&residual);
    let seed = coefficients
        .iter()
        .map(|c| c.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let defect = defect_frobenius(basis, full_images, generator);
    let seed_action = joint_norm(&defect_action(
        basis,
        full_images,
        generator,
        coefficients.as_slice(),
    ));
    let action_envelope =
        defect_action_envelope(basis, full_images, generator, coefficients.as_slice(), time);
    let growth = reduced_growth(generator).max(0.0);
    let flow = (growth * time).exp();
    DefectTerms {
        initial_leakage: leakage,
        defect_frobenius: defect,
        defect_seed_action: seed_action,
        defect_action_envelope: action_envelope,
        state_weighted_bound: leakage + time * action_envelope,
        seed_norm: seed,
        reduced_growth: growth,
        flow_growth: flow,
        time,
        // State norm: Duhamel gives `||e(t)|| <= leakage + int_0^t ||D exp(sG) c|| ds`,
        // which is the sampled defect action integrated over the interval. The
        // Frobenius product `defect * seed * flow` stays reported (and feeds the
        // interval enclosure), but it prices the worst reduced direction rather
        // than the trajectory and carries the shift scale.
        evolution_bound: leakage + time * action_envelope,
        frobenius_product_bound: leakage + time * defect * seed * flow,
    }
}

// ---------------------------------------------------------------------------
// Amplitude-overlap bound
// ---------------------------------------------------------------------------

/// Amplitude (not state-norm) defect bound for a return probability.
///
/// A probability is quadratic in an amplitude, so
/// `| |a|^2 - |b|^2 | <= 2|a - b| + |a - b|^2` for `|a|,|b| <= 1`: pricing the
/// amplitude difference `A = <psi|psi(t)> - <psi|w(t)>` directly, instead of
/// `||e(t)||`, removes the near-orthogonality penalty that makes the state
/// bound unusable for probability responses.
///
/// Variation of constants gives `A = <psi|e(t)>` with
/// `e(t) = e^{-itH}ell - int_0^t e^{-i(t-s)H} r(s) ds` and
/// `r(s) = D exp(sG) c`. Splitting the exact evolved seed `u(tau) = e^{-itau H}psi`
/// into the computable reduced flow `w(tau) = V exp(tau G) c` plus an error of
/// norm `||e(tau)|| <= int_0^tau ||r||` yields
///
/// ```text
/// |A| <= ||ell|| + int_0^t |<w(t-s)|r(s)>| ds + (int_0^t ||r(s)|| ds)^2 / 2
/// ```
///
/// * `overlap` is **computed**, not majorised: it is the one place where the
///   defect's oscillation can cancel against the propagated seed, and a
///   Cauchy-Schwarz bound here is what made the state route blind.
/// * `replacement` is the price of substituting the reduced flow for the exact
///   one; since the double integral over `sigma + s <= t` is half the square, it
///   is `(int ||r||)^2 / 2`. This term is what remains after the overlap, and it
///   is the honest floor of a defect-only certificate.
///
/// Point f64 estimate: exact `exp`/`G` and floating arithmetic remain
/// assumptions, and the quadrature is converged, not enclosed.
#[derive(Clone, Debug)]
pub struct AmplitudeBound {
    /// `||ell||`: how much of the seed lies outside the Krylov basis.
    pub initial_leakage: f64,
    /// `int_0^t ||r(s)|| ds`.
    pub defect_action_integral: f64,
    /// `int_0^t |<w(t-s)|r(s)>| ds`, the computed overlap.
    pub overlap_integral: f64,
    /// `int_0^t (int_0^{t-s} ||r||) ||r(s)|| ds`, the price of substituting the
    /// reduced flow for the exact one. Bounded above by `(int ||r||)^2 / 2`
    /// (the whole square is twice the triangle), and computed rather than
    /// majorised: the defect grows with `s` here, so the accumulated norm is
    /// smallest exactly where `||r(s)||` is largest.
    pub replacement_integral: f64,
    /// The symmetric majorant `(int_0^t ||r|| ds)^2 / 2`, kept for comparison.
    pub replacement_majorant: f64,
    /// `leakage + overlap + replacement`.
    pub amplitude_bound: f64,
    /// `2 * amplitude_bound + amplitude_bound^2`.
    pub probability_bound: f64,
    /// Quadrature panels used for the converged estimate.
    pub panels: usize,
    pub time: f64,
}

impl AmplitudeBound {
    /// The measured fields at their reported precision, matching
    /// [`DefectTerms`]. `panels` is a count and `time` is an input, so both stay
    /// exact.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "initial_leakage": reported_residual(self.initial_leakage),
            "defect_action_integral": reported_residual(self.defect_action_integral),
            "overlap_integral": reported_residual(self.overlap_integral),
            "replacement_integral": reported_residual(self.replacement_integral),
            "replacement_majorant": reported_residual(self.replacement_majorant),
            "amplitude_bound": reported_residual(self.amplitude_bound),
            "probability_bound": reported_residual(self.probability_bound),
            "panels": self.panels,
            "time": self.time,
        })
    }
}

/// See [`AmplitudeBound`]. `basis`, `full_images`, `generator` and `state` are
/// exactly the arguments of [`defect_terms`], so both bounds can be formed from
/// one solve.
pub fn amplitude_terms(
    basis: &[NestedVector],
    full_images: &[NestedVector],
    generator: &DMatrix<C>,
    state: &NestedVector,
    time: f64,
) -> AmplitudeBound {
    let rank = basis.len();
    let coefficients = DVector::from_iterator(rank, basis.iter().map(|v| inner(v, state)));
    let mut projected = NestedVector::new();
    for (vector, coefficient) in basis.iter().zip(coefficients.iter()) {
        add_scaled(&mut projected, vector, *coefficient);
    }
    let mut residual = state.clone();
    for (label, amplitude) in &projected {
        let value = residual.entry(label.clone()).or_default();
        *value -= *amplitude;
    }
    residual.retain(|_, amplitude| *amplitude != C::new(0.0, 0.0));
    let leakage = joint_norm(&residual);
    let seed = DVector::from_iterator(rank, coefficients.iter().copied());
    let flow =
        |tau: f64, seed: &DVector<C>| -> DVector<C> { (generator * C::new(tau, 0.0)).exp() * seed };
    let state_at = |coef: &DVector<C>| -> NestedVector {
        let mut out = NestedVector::new();
        for (vector, c) in basis.iter().zip(coef.iter()) {
            add_scaled(&mut out, vector, *c);
        }
        out
    };
    // Integrands: `||r(s)||` and `|<w(t-s)|r(s)>|`.
    let norms = |s: f64| -> (f64, f64) {
        let r = defect_action(basis, full_images, generator, flow(s, &seed).as_slice());
        let w = state_at(&flow(time - s, &seed));
        (joint_norm(&r), inner(&w, &r).norm())
    };
    let max_panels = if rank <= 8 { 4096 } else { 256 };
    let tolerance = 1e-6;
    let mut panels = 16usize;
    let mut previous = (0.0f64, 0.0f64);
    loop {
        // Composite trapezoid: the integrands are analytic on the interval, so
        // this converges geometrically in the panel count.
        let h = time / panels as f64;
        let weights: Vec<f64> = (0..=panels)
            .map(|index| {
                if index == 0 || index == panels {
                    0.5 * h
                } else {
                    h
                }
            })
            .collect();
        let sampled = norms_grid(&norms, time, panels);
        let mut total = (0.0f64, 0.0f64);
        for (index, (norm, overlap)) in sampled.iter().enumerate() {
            total.0 += weights[index] * norm;
            total.1 += weights[index] * overlap;
        }
        let change = ((total.0 - previous.0).abs(), (total.1 - previous.1).abs());
        let converged =
            change.0 <= tolerance * (1.0 + total.0) && change.1 <= tolerance * (1e-3 + total.1);
        if panels >= max_panels || (panels > 16 && converged) {
            let integral = total.0 + change.0;
            let overlap = total.1 + change.1;
            // Replacement: `int_0^t F(t-s) ||r(s)|| ds` with
            // `F(t-s) = int_0^{t-s} ||r||` from a cumulative trapezoid on the
            // same grid, so no extra kernel evaluations are needed.
            let mut cumulative = vec![0.0f64; panels + 1];
            for index in 1..=panels {
                cumulative[index] =
                    cumulative[index - 1] + 0.5 * h * (sampled[index - 1].0 + sampled[index].0);
            }
            let mut replacement = 0.0f64;
            for index in 0..=panels {
                replacement += weights[index] * cumulative[panels - index] * sampled[index].0;
            }
            let majorant = 0.5 * integral * integral;
            let replacement = replacement.min(majorant);
            let amplitude = leakage + overlap + replacement;
            return AmplitudeBound {
                initial_leakage: leakage,
                defect_action_integral: integral,
                overlap_integral: overlap,
                replacement_integral: replacement,
                replacement_majorant: majorant,
                amplitude_bound: amplitude,
                probability_bound: 2.0 * amplitude + amplitude * amplitude,
                panels,
                time,
            };
        }
        previous = total;
        panels *= 2;
    }
}

fn norms_grid(norms: &impl Fn(f64) -> (f64, f64), time: f64, panels: usize) -> Vec<(f64, f64)> {
    (0..=panels)
        .map(|index| norms(time * index as f64 / panels as f64))
        .collect()
}

// ---------------------------------------------------------------------------
// Interval fixture
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct IntervalTerms {
    pub initial_leakage: Interval,
    pub defect_frobenius: Interval,
    pub seed_norm: Interval,
    pub spectral_growth: Interval,
    pub flow_growth: Interval,
    pub time: Interval,
    pub evolution_bound: Interval,
}

fn abs_upper(interval: Interval) -> f64 {
    interval.lo.abs().max(interval.hi.abs())
}

fn zero_interval() -> ComplexInterval {
    ComplexInterval::point(C::new(0.0, 0.0))
}

fn complex_conj(value: ComplexInterval) -> ComplexInterval {
    ComplexInterval {
        re: value.re,
        im: value.im.neg(),
    }
}

fn interval_matmul(
    left: &[Vec<ComplexInterval>],
    right: &[Vec<ComplexInterval>],
) -> Vec<Vec<ComplexInterval>> {
    assert!(!left.is_empty());
    assert_eq!(left[0].len(), right.len());
    let columns = right.first().map_or(0, Vec::len);
    let mut product = vec![vec![zero_interval(); columns]; left.len()];
    for (row, output) in product.iter_mut().enumerate() {
        for (column, entry) in output.iter_mut().enumerate() {
            let mut sum = zero_interval();
            for (inner, factor) in left[row].iter().enumerate() {
                sum = sum.add(factor.mul(right[inner][column]));
            }
            *entry = sum;
        }
    }
    product
}

fn interval_dense(matrix: &DMatrix<C>) -> Vec<Vec<ComplexInterval>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| ComplexInterval::point(matrix[(row, column)]))
                .collect()
        })
        .collect()
}

fn interval_adjoint(matrix: &[Vec<ComplexInterval>]) -> Vec<Vec<ComplexInterval>> {
    (0..matrix[0].len())
        .map(|row| {
            (0..matrix.len())
                .map(|column| complex_conj(matrix[column][row]))
                .collect()
        })
        .collect()
}

/// Enclose the columns of `H V` from the exact declared `f64` matrices, in the
/// same column-per-basis-vector layout as [`interval_terms`]. Every add and
/// multiply rounds outward, so each endpoint bounds the exact real-arithmetic
/// product for the declared input data.
pub fn full_images_enclosure(h: &DMatrix<C>, v: &DMatrix<C>) -> Vec<Vec<ComplexInterval>> {
    let product = interval_matmul(&interval_dense(h), &interval_dense(v));
    (0..product[0].len())
        .map(|column| (0..product.len()).map(|row| product[row][column]).collect())
        .collect()
}

/// Enclose `G = V* H V` from the exact declared `f64` matrices. Every add and
/// multiply rounds outward, so each endpoint bounds the exact real-arithmetic
/// product for the declared input data.
pub fn reduced_generator_enclosure(h: &DMatrix<C>, v: &DMatrix<C>) -> Vec<Vec<ComplexInterval>> {
    let v = interval_dense(v);
    let hv = interval_matmul(&interval_dense(h), &v);
    interval_matmul(&interval_adjoint(&v), &hv)
}

/// Gershgorin upper bound on `lambda_max(Re G)` for an enclosed `G`. Each row
/// disc uses directed complex magnitudes; the maximum row upper endpoint bounds
/// every eigenvalue of every matrix in the interval family.
fn hermitian_growth_upper(generator: &[Vec<ComplexInterval>]) -> Interval {
    let half = Interval::point(0.5);
    let mut bound = 0.0_f64;
    for (row, generator_row) in generator.iter().enumerate() {
        let mut radius = 0.0_f64;
        for (column, cell) in generator_row.iter().enumerate() {
            if row == column {
                continue;
            }
            // Re G[row][column] = (G[row][column] + conj G[column][row]) / 2.
            let re = cell.re.add(generator[column][row].re).mul(half);
            let im = cell.im.sub(generator[column][row].im).mul(half);
            let re_abs = abs_upper(re);
            let im_abs = abs_upper(im);
            let modulus = add_up(mul_up(re_abs, re_abs), mul_up(im_abs, im_abs)).sqrt();
            radius = add_up(radius, up(modulus));
        }
        // The diagonal of the Hermitian part is Re G[row][row].
        bound = bound.max(add_up(generator[row][row].re.hi, radius));
    }
    Interval {
        lo: 0.0,
        hi: bound.max(0.0),
    }
}

fn ci_norm_square(sum: &mut Interval, z: ComplexInterval) {
    *sum = sum.add(Interval {
        lo: 0.0,
        hi: z.norm_square_upper(),
    });
}

/// `f64` directed-endpoint repetition of [`defect_terms`].
///
/// Addition, subtraction, multiplication, square and sqrt round outward with
/// `next_down`/`next_up`; `exp` also rounds its upper endpoint outward. The
/// basis and state are exact `f64` data; `full_images` and `generator` must
/// enclose `H V` and `V* H V` from the declared `H` and `V` (see
/// [`full_images_enclosure`] and [`reduced_generator_enclosure`]). The spectral
/// growth is a directed Gershgorin bound on the enclosed Hermitian part. The
/// result is a directed bound for the declared finite matrices, not a physical
/// reduction certificate. The measured map error separately uses Padé without
/// an error enclosure. The caller owns the theorem hypotheses.
pub fn interval_terms(
    basis: &[Vec<C>],
    full_images: &[Vec<ComplexInterval>],
    generator: &[Vec<ComplexInterval>],
    state: &[C],
    time: f64,
) -> IntervalTerms {
    let rank = basis.len();
    assert_eq!(full_images.len(), rank);
    assert_eq!(generator.len(), rank);
    let to_vec = |slice: &[C]| -> Vec<ComplexInterval> {
        slice.iter().copied().map(ComplexInterval::point).collect()
    };
    let basis: Vec<Vec<ComplexInterval>> = basis.iter().map(|v| to_vec(v)).collect();
    let images = full_images.to_vec();
    let state: Vec<ComplexInterval> = to_vec(state);
    let zero = ComplexInterval::point(C::new(0.0, 0.0));
    // coefficients c_j = <v_j, psi>
    let mut coefficients = Vec::with_capacity(rank);
    for v in &basis {
        let mut acc = zero;
        for (a, b) in v.iter().zip(&state) {
            // conj(a) * b
            let conj = ComplexInterval {
                re: a.re,
                im: a.im.neg(),
            };
            acc = acc.add(conj.mul(*b));
        }
        coefficients.push(acc);
    }
    let mut seed_square = Interval::point(0.0);
    for c in &coefficients {
        ci_norm_square(&mut seed_square, *c);
    }
    let seed_norm = Interval::point(seed_square.sqrt_upper());
    // residual = psi - sum c_j v_j
    let mut residual: Vec<ComplexInterval> = state.clone();
    for (c, v) in coefficients.iter().zip(&basis) {
        for (r, x) in residual.iter_mut().zip(v) {
            *r = r.sub(c.mul(*x));
        }
    }
    let mut leakage_square = Interval::point(0.0);
    for r in &residual {
        ci_norm_square(&mut leakage_square, *r);
    }
    let initial_leakage = Interval::point(leakage_square.sqrt_upper());
    let mut defect_square = Interval::point(0.0);
    for column in 0..rank {
        let mut d: Vec<ComplexInterval> = vec![zero; state.len()];
        for (row, v) in basis.iter().enumerate() {
            let value = generator[row][column];
            let factor = ComplexInterval {
                re: value.im.neg(),
                im: value.re,
            };
            for (out, x) in d.iter_mut().zip(v) {
                *out = out.add(x.mul(factor));
            }
        }
        for (out, x) in d.iter_mut().zip(&images[column]) {
            *out = out.sub(*x);
        }
        for value in d {
            ci_norm_square(&mut defect_square, value);
        }
    }
    let defect_frobenius = Interval::point(defect_square.sqrt_upper());
    let spectral_growth = hermitian_growth_upper(generator);
    let flow_growth = Interval::point(spectral_growth.mul(Interval::point(time)).exp_upper());
    let evolution_bound = initial_leakage.add(
        Interval::point(time)
            .mul(defect_frobenius)
            .mul(seed_norm)
            .mul(flow_growth),
    );
    IntervalTerms {
        initial_leakage,
        defect_frobenius,
        seed_norm,
        spectral_growth,
        flow_growth,
        time: Interval::point(time),
        evolution_bound,
    }
}

/// The actual dense inputs of the finite N8 fixture, so a bridge record can
/// bind their hashes instead of restating duplicate matrices.
pub fn finite_fixture_inputs() -> (DMatrix<C>, DMatrix<C>, DVector<C>) {
    let h = DMatrix::from_row_slice(
        4,
        4,
        &[
            C::new(0.7, 0.0),
            C::new(0.0, 0.3),
            C::new(0.2, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, -0.3),
            C::new(1.1, 0.0),
            C::new(0.0, 0.0),
            C::new(0.15, 0.0),
            C::new(0.2, 0.0),
            C::new(0.0, 0.0),
            C::new(-0.4, 0.0),
            C::new(0.0, -0.25),
            C::new(0.0, 0.0),
            C::new(0.15, 0.0),
            C::new(0.0, 0.25),
            C::new(0.9, 0.0),
        ],
    );
    let v = DMatrix::from_row_slice(
        4,
        2,
        &[
            C::new(1.0, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(1.0, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
        ],
    );
    let psi = DVector::from_vec(vec![
        C::new(0.5, 0.1),
        C::new(0.2, -0.4),
        C::new(0.6, 0.0),
        C::new(0.0, 0.3),
    ]);
    let psi = psi.map(|x| x / psi.norm());
    (h, v, psi)
}

/// A small independent finite fixture: a 4x4 Hermitian H, a 4x2 isometry and
/// `G = V* H V`. Returns point terms, interval terms and the directly measured
/// map error (the direct value is a check, not an enclosure).
pub fn finite_fixture(time: f64) -> (DefectTerms, IntervalTerms, f64) {
    let (h, v, psi) = finite_fixture_inputs();
    let generator = v.adjoint() * &h * &v;
    let basis: Vec<Vec<C>> = (0..2)
        .map(|column| (0..4).map(|row| v[(row, column)]).collect())
        .collect();
    let state: Vec<C> = psi.iter().copied().collect();
    let images_enclosure = full_images_enclosure(&h, &v);
    let generator_enclosure = reduced_generator_enclosure(&h, &v);
    let interval = interval_terms(
        &basis,
        &images_enclosure,
        &generator_enclosure,
        &state,
        time,
    );
    // Dense point terms for the same fixture (the SM cell uses `defect_terms`).
    let coefficients = v.adjoint() * &psi;
    let projected = &v * &coefficients;
    let leakage = (&psi - &projected).norm();
    let seed = coefficients.norm();
    let mut defect_square = 0.0;
    let mut action = DVector::<C>::zeros(4);
    for column in 0..2 {
        let mut d = DVector::<C>::zeros(4);
        for row in 0..2 {
            d += v.column(row) * (C::i() * generator[(row, column)]);
        }
        d -= (&h * &v).column(column).into_owned();
        defect_square += d.norm_squared();
        action += d * coefficients[column];
    }
    let defect = defect_square.sqrt();
    let seed_action = action.norm();
    let growth = reduced_growth(&generator).max(0.0);
    let flow = (growth * time).exp();
    // Same quantity and same estimator as `defect_action_envelope`: the defect
    // action sampled along the trajectory. The old `||D||_F seed e^{omega t}`
    // product is *smaller* here only because this fixture's generator is
    // ordinary (no rational shift scale); sharing the helper keeps the fixture
    // and the shipped SM path on one formula.
    let envelope = sampled_sup(
        &|s: f64| {
            let evolved = (&generator * C::new(s, 0.0)).exp() * &coefficients;
            let mut d = &v * (&generator * &evolved) * C::i();
            d -= &h * (&v * &evolved);
            d.norm()
        },
        time,
        64,
    );
    let terms = DefectTerms {
        initial_leakage: leakage,
        defect_frobenius: defect,
        defect_seed_action: seed_action,
        defect_action_envelope: envelope,
        state_weighted_bound: leakage + time * envelope,
        seed_norm: seed,
        reduced_growth: growth,
        flow_growth: flow,
        time,
        evolution_bound: leakage + time * envelope,
        frobenius_product_bound: leakage + time * defect * seed * flow,
    };
    let exact = (&h * (-C::i() * time)).exp() * &psi;
    let compressed = &v * (&generator * C::new(time, 0.0)).exp() * (&v.adjoint() * &psi);
    let measured = (&exact - &compressed).norm();
    (terms, interval, measured)
}

fn dense_json(matrix: &DMatrix<C>) -> Value {
    Value::Array(
        (0..matrix.nrows())
            .map(|row| {
                Value::Array(
                    (0..matrix.ncols())
                        .map(|column| {
                            let value = matrix[(row, column)];
                            json!([value.re, value.im])
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

fn vector_json(vector: &DVector<C>) -> Value {
    Value::Array(
        vector
            .iter()
            .map(|value| json!([value.re, value.im]))
            .collect(),
    )
}

fn interval_json(interval: Interval) -> Value {
    json!({"lo": interval.lo, "hi": interval.hi})
}

fn interval_matrix_json(matrix: &[Vec<ComplexInterval>]) -> Value {
    Value::Array(
        matrix
            .iter()
            .map(|row| {
                Value::Array(
                    row.iter()
                        .map(|value| {
                            json!({
                                "re": interval_json(value.re),
                                "im": interval_json(value.im),
                            })
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

/// N8 proof-facing record: binds the actual finite fixture, the certificate
/// reader's terms and the measured map error. The record is deliberately
/// `conditionally-bounded`; interval arithmetic encloses `HV` and `G`, the
/// spectral bound is directed, and nanoda checks the finite endpoint
/// arithmetic. The Duhamel representation, its integral bound and the
/// skew-adjoint flow bound are kernel-verified. `SmDefectSkew.lean` proves the
/// declared finite fixture is Hermitian and `A = iH` is skew-adjoint, with its
/// exported payload independently replayed by nanoda; the physical reduction
/// remains open.
pub fn finite_fixture_bridge(time: f64) -> Value {
    let (h, v, psi) = finite_fixture_inputs();
    let full_images_enclosure = full_images_enclosure(&h, &v);
    let generator_enclosure = reduced_generator_enclosure(&h, &v);
    let (terms, interval, measured) = finite_fixture(time);
    let hashes = CertificateHashes {
        operator: canonical_json_sha256(&json!({"h": dense_json(&h)})),
        state: canonical_json_sha256(&json!({"psi": vector_json(&psi)})),
        basis: canonical_json_sha256(&json!({"v": dense_json(&v)})),
    };
    let certificate = certificate_json(&terms, &hashes);
    bind_artifact(json!({
        "schema": crate::sm_bridge::BRIDGE_SCHEMA,
        "claim_id": "n8_finite_map_defect",
        "status": "conditionally-bounded",
        "endpoint": "N8-finite-map",
        "operator": {
            "name": "finite 4x4 Hermitian H with 4x2 isometric V",
            "identity_sha256": hashes.operator,
            "source": "fock_sirk::sm_defect::finite_fixture_inputs",
        },
        "state": {
            "basis": ["v_0", "v_1"],
            "identity_sha256": hashes.state,
            "note": "psi is the normalized declared state vector; the isometry hash is bound under the embedded certificate hashes.basis",
        },
        "artifact": {
            "time": time,
            "certificate": certificate,
            "full_images_enclosure": interval_matrix_json(&full_images_enclosure),
            "generator_enclosure": interval_matrix_json(&generator_enclosure),
            "interval_terms": {
                "initial_leakage": interval_json(interval.initial_leakage),
                "defect_frobenius": interval_json(interval.defect_frobenius),
                "seed_norm": interval_json(interval.seed_norm),
                "spectral_growth": interval_json(interval.spectral_growth),
                "flow_growth": interval_json(interval.flow_growth),
                "time": interval_json(interval.time),
                "evolution_bound": interval_json(interval.evolution_bound),
            },
            "measured_map_error": reported_residual(measured),
            "measured_within_interval_hi": measured <= interval.evolution_bound.hi,
            "rounding": {
                "mode": "f64-directed-endpoints",
                "directed": true,
                "scope": "add/sub/mul/square/sqrt/exp in interval_terms; enclosed HV and V*HV; Gershgorin row sums",
                "spectral_growth_directed": true,
                "spectral_growth_scope": "Gershgorin upper bound on the enclosed Hermitian part of V*HV",
                "spectral_input_enclosed": true,
                "full_images_enclosed": true,
                "pade_exponential_directed": false,
                "note": "endpoint arithmetic rounds outward, HV and V*HV are enclosed from the declared H and V, and lambda_max(Re G) is Gershgorin-bounded; the Duhamel representation, integral bound and skew-adjoint flow bound are kernel-verified, and A = iH is kernel-verified for the declared finite fixture by SmDefectSkew; the physical reduction remains open, so evolution_bound.hi is a directed bound for the finite fixture, not a physical certificate; the measured map error separately uses Pade without an error enclosure",
            },
        },
        "theorem": {
            "lean_name": "Analytic.SmDefect.finite_map",
            "statement": "||exp(-itH) psi - V exp(tG) V* psi|| <= leakage + t ||D|| ||V* psi|| exp(omega_+ t)",
            "proof_status": "not-run",
            "remaining": "The Duhamel representation and the N8 bound are kernel-verified for a skew-adjoint generator A, including the flow bound ||exp((-r)A)|| = 1. `BookProof.Sm.ChapterSmDefectSkew.A_skew` proves A = iH is skew-adjoint for the declared finite fixture and its exported payload is nanoda-verified; the physical reduction remains open, and the measured map error separately uses nalgebra's Pade exponential without an error enclosure.",
        },
        "finite_endpoint_proof": {
            "lean_name": "BookProof.Sm.ChapterSmDefect.finite_map_endpoint",
            "statement": "5*10^32*measured <= 5*10^32*leakage + defect*seed*flow for the declared 16-decimal N8 endpoint rationals",
            "proof_status": "kernel-verified",
            "checker": "nanoda_lib via prob_kernel::verify::verify_export",
            "payload": "docs/fixtures/sm_defect_proof.ndjson",
            "payload_sha256": sha256_hex(SM_DEFECT_PROOF),
            "declarations": 273,
            "scope": "finite endpoint arithmetic only; the Duhamel/flow theorem is separate",
        },
        "analytic_defect_proof": {
            "lean_name": "BookProof.Sm.ChapterSmDefectAnalytic.norm_duhamel_representation_le_of_skew",
            "statement": "for u' = -A u + b with star A = -A, u t = exp(-t A) u 0 + integral exp(-(t-s)A) b s, and ||u t|| <= leakage + t*defect*seed*exp(omega*t) when ||exp(-t A) u 0|| <= leakage and ||b s|| <= defect*seed*exp(omega*t)",
            "proof_status": "kernel-verified",
            "checker": "nanoda_lib via prob_kernel::verify::verify_export",
            "payload": "docs/fixtures/sm_defect_analytic_proof.ndjson.gz",
            "compressed_payload_sha256": "ccddf3b7c83edfed0d44aa6d9b9ae81080bff7fc04b04ded4829f65028f96445",
            "payload_sha256": "83e3571ed458dff72b086934d57724be3b4763f47d58a07439b401c0d714ed31",
            "declarations": 38336,
            "scope": "Duhamel representation and skew-adjoint flow bound formalized; the physical reduction is open",
        },
        "matrix_skew_proof": {
            "lean_name": "BookProof.Sm.ChapterSmDefectSkew.A_skew",
            "hermitian_name": "BookProof.Sm.ChapterSmDefectSkew.H_hermitian",
            "statement": "the declared exact-rational 4x4 fixture H is Hermitian and A = i*H satisfies A^H = -A",
            "proof_status": "kernel-verified",
            "checker": "nanoda_lib via prob_kernel::verify::verify_export",
            "payload": "docs/fixtures/sm_defect_skew_proof.ndjson.gz",
            "compressed_payload_sha256": "eeb92d5b0fc588851ee4600b692c092a3d8e0a982b509b1947c38bc32c7dd9aa",
            "payload_sha256": "498ae454792caafc7fb4150fb1cfb7a9ecf336d6a3b84d2ee9dcf9f80db528cc",
            "declarations": 7540,
            "source": "fock_sirk/lean/SmDefectSkew.lean",
            "source_sha256": "acc3dfc49fad2760020f33617e2ff72e3d74b999a4643046ae6efd2d149afc53",
            "fixture_provenance": {
                "check": "fock_sirk/tests/sm_defect_bridge.rs::lean_matrix_matches_rust_fixture_provenance",
                "status": "test-checked",
                "note": "the H cells are parsed from the Lean source as exact rationals and compared entrywise with finite_fixture_inputs",
            },
            "scope": "the declared finite fixture only; the physical reduction remains open",
        },
        "hypotheses": [
            "H is the declared finite 4x4 Hermitian matrix; its hash is bound under operator.identity_sha256.",
            "V is the declared 4x2 isometry and G = V* H V; its hash is bound under state.identity_sha256.",
            "The bound's exponential is exact; the measured map error uses nalgebra's Pade exp and is a check, not an enclosure.",
            "interval_terms uses directed f64 endpoints, encloses HV and G = V*HV from the declared H and V, and applies a Gershgorin spectral bound; its hi endpoint is a directed bound for the finite fixture, not a physical reduction certificate.",
            "The endpoint Lean proof checks only the finite arithmetic; the analytic Lean proof derives the Duhamel representation, the integral bound, and the skew-adjoint flow bound; the nanoda-verified `SmDefectSkew` payload identifies A = iH for the declared finite matrix and leaves the physical reduction open.",
        ],
        "not_claimed": [
            "No SM cell, physical-sector reduction, cutoff or continuum claim.",
            "No certified f64 enclosure and no Padé-error enclosure.",
            "No resolution of the N11 response and no promotion of N12.",
        ],
    }))
}

// ---------------------------------------------------------------------------
// Certificate reader
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertificateHashes {
    pub operator: String,
    pub state: String,
    pub basis: String,
}

impl std::fmt::Display for DefectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DefectError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DefectError {
    Missing(&'static str),
    Shape(&'static str),
    NegativeOrNonfinite(&'static str),
    BoundBelowLeakage,
    HashMismatch(&'static str),
    Status,
}

pub fn certificate_json(terms: &DefectTerms, hashes: &CertificateHashes) -> Value {
    let mut fields = Map::new();
    for (key, value) in terms.reported_fields() {
        // `defect_seed_action` is a diagnostic of the term record; the
        // certificate has always carried only the bound's inputs.
        if key == "defect_seed_action" {
            continue;
        }
        fields.insert(key.into(), json!(value));
    }
    fields.insert("time".into(), json!(terms.time));
    json!({
        "schema": DEFECT_SCHEMA,
        "status": "conditionally-bounded",
        "theorem": "||exp(-itH)psi - V exp(tG)V*psi|| <= leakage + t ||D|| ||V*psi|| exp(omega_+ t)",
        "fields": fields,
        "hashes": {
            "operator": hashes.operator,
            "state": hashes.state,
            "basis": hashes.basis,
        },
        "assumptions": [
            "V is isometric in the declared joint inner product",
            "H is the identified self-adjoint realization (SM3/SM5 open)",
            "G is the returned reduced generator; omega_+ covers its Hermitian part",
            "the finite exponential is approximate; Padé/rounding are not enclosed here",
        ],
    })
}

fn number(map: &Map<String, Value>, key: &'static str) -> Result<f64, DefectError> {
    map.get(key)
        .and_then(Value::as_f64)
        .ok_or(DefectError::Missing(key))
}

pub fn verify_certificate(root: &Value, expected: &CertificateHashes) -> Result<(), DefectError> {
    if root.get("schema").and_then(Value::as_str) != Some(DEFECT_SCHEMA) {
        return Err(DefectError::Shape("schema"));
    }
    if root.get("status").and_then(Value::as_str) != Some("conditionally-bounded") {
        return Err(DefectError::Status);
    }
    let fields = root
        .get("fields")
        .and_then(Value::as_object)
        .ok_or(DefectError::Missing("fields"))?;
    for key in [
        "initial_leakage",
        "defect_frobenius",
        "seed_norm",
        "reduced_growth",
        "flow_growth",
        "time",
        "evolution_bound",
    ] {
        let value = number(fields, key)?;
        if !value.is_finite() || value < 0.0 || (key == "time" && value == 0.0) {
            return Err(DefectError::NegativeOrNonfinite(key));
        }
    }
    if number(fields, "evolution_bound")? < number(fields, "initial_leakage")? {
        return Err(DefectError::BoundBelowLeakage);
    }
    let hashes = root
        .get("hashes")
        .and_then(Value::as_object)
        .ok_or(DefectError::Missing("hashes"))?;
    for (key, expected) in [
        ("operator", &expected.operator),
        ("state", &expected.state),
        ("basis", &expected.basis),
    ] {
        let value = hashes
            .get(key)
            .and_then(Value::as_str)
            .ok_or(DefectError::Missing("hashes"))?;
        if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(DefectError::Shape("hashes"));
        }
        if value != expected {
            return Err(DefectError::HashMismatch(key));
        }
    }
    Ok(())
}

/// Convenience for JSON-producing drivers: a map from label to certificate.
pub fn certificates_json(
    terms: &[DefectTerms],
    hashes: &CertificateHashes,
) -> BTreeMap<String, Value> {
    terms
        .iter()
        .enumerate()
        .map(|(step, terms)| (format!("step_{step}"), certificate_json(terms, hashes)))
        .collect()
}
