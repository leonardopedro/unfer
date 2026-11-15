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
//! generator. The `f64` path is a measured estimate. [`interval_certificate`]
//! repeats it with outward-rounded intervals for a finite fixture. Neither is an
//! unconditional physical-SM certificate: the self-adjoint operator, the
//! physical reduction and the GMRES/Padé/rounding bridge remain assumptions.

use nalgebra::{DMatrix, DVector};
use nested_fock_algebra::exact_inner::NestedVector;
use num_complex::Complex64 as C;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

pub const DEFECT_SCHEMA: &str = "sm-defect-certificate/v1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn point(x: f64) -> Self {
        assert!(x.is_finite(), "interval endpoints must be finite");
        Self { lo: x, hi: x }
    }
    pub fn add(self, other: Self) -> Self {
        Self {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
        }
    }
    pub fn sub(self, other: Self) -> Self {
        Self {
            lo: self.lo - other.hi,
            hi: self.hi - other.lo,
        }
    }
    pub fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    pub fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        Self {
            lo: products.iter().copied().fold(f64::INFINITY, f64::min),
            hi: products.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
    pub fn square_upper(self) -> f64 {
        (self.lo * self.lo).max(self.hi * self.hi)
    }
    pub fn sqrt_upper(self) -> f64 {
        self.hi.max(0.0).sqrt()
    }
    pub fn exp_upper(self) -> f64 {
        self.hi.exp()
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
        self.re.square_upper() + self.im.square_upper()
    }
}

/// One term per error component. `evolution_bound` is the sum below; the
/// remaining fields stay separate so a reader can see what was priced.
#[derive(Clone, Debug)]
pub struct DefectTerms {
    pub initial_leakage: f64,
    pub defect_frobenius: f64,
    pub seed_norm: f64,
    pub reduced_growth: f64,
    pub flow_growth: f64,
    pub time: f64,
    pub evolution_bound: f64,
}

impl DefectTerms {
    pub fn to_json(&self, extra: Value) -> Value {
        json!({
            "initial_leakage": self.initial_leakage,
            "defect_frobenius": self.defect_frobenius,
            "seed_norm": self.seed_norm,
            "reduced_growth": self.reduced_growth,
            "flow_growth": self.flow_growth,
            "time": self.time,
            "evolution_bound": self.evolution_bound,
            "extra": extra,
        })
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

/// `D = i V G - H V` column by column, with the full (unprojected) `H V`.
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
    let growth = reduced_growth(generator).max(0.0);
    let flow = (growth * time).exp();
    DefectTerms {
        initial_leakage: leakage,
        defect_frobenius: defect,
        seed_norm: seed,
        reduced_growth: growth,
        flow_growth: flow,
        time,
        evolution_bound: leakage + time * defect * seed * flow,
    }
}

// ---------------------------------------------------------------------------
// Interval fixture
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct IntervalTerms {
    pub initial_leakage: Interval,
    pub defect_frobenius: Interval,
    pub seed_norm: Interval,
    pub flow_growth: Interval,
    pub time: Interval,
    pub evolution_bound: Interval,
}

fn ci_norm_square(sum: &mut Interval, z: ComplexInterval) {
    *sum = sum.add(Interval::point(z.norm_square_upper()));
}

/// Outward-rounded repetition of [`defect_terms`] with point f64 inputs.
/// Every operation rounds outward; transcendental `exp`/`sqrt` use the upper
/// endpoint. The caller still owns the theorem hypotheses.
pub fn interval_terms(
    basis: &[Vec<C>],
    full_images: &[Vec<C>],
    generator: &[Vec<C>],
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
    let images: Vec<Vec<ComplexInterval>> = full_images.iter().map(|v| to_vec(v)).collect();
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
            let factor = ComplexInterval::point(C::i() * generator[row][column]);
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
    // Hermitian part eigenvalues in f64; its upper endpoint is the enclosure.
    let dense = DMatrix::from_fn(rank, rank, |r, c| generator[r][c]);
    let symmetric = (dense.clone() + dense.adjoint()) * C::new(0.5, 0.0);
    let growth = symmetric
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    let flow_value = (growth * time).exp();
    let flow_growth = Interval::point(flow_value);
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
        flow_growth,
        time: Interval::point(time),
        evolution_bound,
    }
}

/// A small independent finite fixture: a 4x4 Hermitian H, a 4x2 isometry and
/// `G = V* H V`. Returns point terms, interval terms and the directly measured
/// map error (the direct value is a check, not an enclosure).
pub fn finite_fixture(time: f64) -> (DefectTerms, IntervalTerms, f64) {
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
    let generator = v.adjoint() * &h * &v;
    let psi = DVector::from_vec(vec![
        C::new(0.5, 0.1),
        C::new(0.2, -0.4),
        C::new(0.6, 0.0),
        C::new(0.0, 0.3),
    ]);
    let psi = psi.map(|x| x / psi.norm());
    let basis: Vec<Vec<C>> = (0..2)
        .map(|column| (0..4).map(|row| v[(row, column)]).collect())
        .collect();
    let images: Vec<Vec<C>> = (0..2)
        .map(|column| (0..4).map(|row| (&h * &v).column(column)[row]).collect())
        .collect();
    let state: Vec<C> = psi.iter().copied().collect();
    let generator_columns: Vec<Vec<C>> = (0..2)
        .map(|column| (0..2).map(|row| generator[(row, column)]).collect())
        .collect();
    let interval = interval_terms(&basis, &images, &generator_columns, &state, time);
    // Dense point terms for the same fixture (the SM cell uses `defect_terms`).
    let coefficients = v.adjoint() * &psi;
    let projected = &v * &coefficients;
    let leakage = (&psi - &projected).norm();
    let seed = coefficients.norm();
    let mut defect_square = 0.0;
    for column in 0..2 {
        let mut d = DVector::<C>::zeros(4);
        for row in 0..2 {
            d += &v.column(row) * (C::i() * generator[(row, column)]);
        }
        d -= (&h * &v).column(column).into_owned();
        defect_square += d.norm_squared();
    }
    let defect = defect_square.sqrt();
    let growth = reduced_growth(&generator).max(0.0);
    let flow = (growth * time).exp();
    let terms = DefectTerms {
        initial_leakage: leakage,
        defect_frobenius: defect,
        seed_norm: seed,
        reduced_growth: growth,
        flow_growth: flow,
        time,
        evolution_bound: leakage + time * defect * seed * flow,
    };
    let exact = (&h * (-C::i() * time)).exp() * &psi;
    let compressed = &v * (&generator * C::new(time, 0.0)).exp() * (&v.adjoint() * &psi);
    let measured = (&exact - &compressed).norm();
    (terms, interval, measured)
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
    fields.insert("initial_leakage".into(), json!(terms.initial_leakage));
    fields.insert("defect_frobenius".into(), json!(terms.defect_frobenius));
    fields.insert("seed_norm".into(), json!(terms.seed_norm));
    fields.insert("reduced_growth".into(), json!(terms.reduced_growth));
    fields.insert("flow_growth".into(), json!(terms.flow_growth));
    fields.insert("time".into(), json!(terms.time));
    fields.insert("evolution_bound".into(), json!(terms.evolution_bound));
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

fn number<'a>(map: &'a Map<String, Value>, key: &'static str) -> Result<f64, DefectError> {
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
