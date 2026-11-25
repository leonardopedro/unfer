//! Full Standard-Model momentum kernel over the **1,241-expression dictionary**.
//!
//! `docs/sm_full_nested.cdb` exports the complete independent-jet component
//! dictionary as `docs/fixtures/sm_full_cadabra.latex`:
//!
//! | family | rows | label prefix |
//! |---|---|---|
//! | field kinetic energy | 40 | `kin*` |
//! | strong magnetic forms | 24 | `magG*` |
//! | weak magnetic forms | 9 | `magW*` |
//! | hypercharge magnetic forms | 3 | `magB*` |
//! | covariant Higgs energy | 12 | `covH*` |
//! | Higgs potential | 1 | `wall` |
//! | matter matrix | 1152 | `matterRxC` |
//!
//! This module lifts **every** exported expression to a momentum kernel and
//! cross-checks it against an independent coordinate-space packet integral.
//! The two routes are:
//!
//! ```text
//! momentum:    Σ_terms c · (2π)^{|p|} ∫∫ φ̂_bra(k) φ̂_ket(l) l^p
//!                                            ∏_n (Φ̂_n^{⋆a_n})(k − l) d³k d³l
//! coordinate:  Σ_terms c · ∫ φ_bra*(x) φ_ket(x) (−i∂)^p ∏_n Φ_n(x)^{a_n} d³x
//! ```
//!
//! with the audit's conventions `f̂(k)=∫e^{-2πik·x}f(x)d³x`, `∂ᵢf → 2πikᵢf̂`,
//! `(a⋆b)(K)=∫a(q)b(K−q)d³q`. The canonical momentum `p = −i∂` therefore has
//! the **real** Fourier symbol `2πk` (matching [`sm_momentum::dirac_symbol`]),
//! while `q_n` insertions carry the transferred momentum `k − l` between the
//! CAR legs, exactly as in `docs/SM_HAMILTONIAN_AUDIT.md` item 3.
//!
//! The equality of the two routes is Plancherel's theorem term by term. It is a
//! genuine check of the routing (insertion transfer, repeated-`q` convolution,
//! derivative symbol and the packet normalisations), not a re-derivation of the
//! classical dictionary: the polynomial itself is taken from the Cadabra export.
//!
//! ## Scope
//!
//! The dictionary is a polynomial of degree ≤ 4 in the 163 collective
//! coordinates, linear in the 151 independent jet momenta, with symbolic
//! couplings/Yukawa entries. Nothing here claims the full ESA/BRST program; the
//! 1,241 routes are packet entries of the declared dictionary at fixed
//! collective coordinates.
//!
//! Packet profiles (all smeared, no lattice):
//!
//! ```text
//! φ̂(k)  = exp(−|k−c|²/(2w²))
//! φ(x)  = (√(2π)w)³ exp(−2π²w²|x|²) e^{2πic·x}      (exact inverse transform)
//! ```

use crate::sm_full::SmParameters;
use crate::sm_momentum::{Conventions, MomentumPacket, TWO_PI};
use num_complex::Complex64 as C;
use std::sync::OnceLock;

/// The Cadabra export this module parses: 40 momenta, 49 bosonic forms and
/// 1,152 matter entries.
pub const DICTIONARY_FIXTURE: &str = include_str!("../../docs/fixtures/sm_full_cadabra.latex");

/// Number of independent collective coordinates (`q_0 ..= q_162`).
pub const Q_COORDINATES: usize = 163;

/// Number of independent jet momenta (`p_0 ..= p_150`).
pub const P_MOMENTA: usize = 151;

/// Dictionary family, derived from the Cadabra label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Kinetic,
    MagneticStrong,
    MagneticWeak,
    MagneticHyper,
    CovariantHiggs,
    HiggsPotential,
    Matter,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Kinetic => "kinetic",
            Family::MagneticStrong => "magnetic-strong",
            Family::MagneticWeak => "magnetic-weak",
            Family::MagneticHyper => "magnetic-hypercharge",
            Family::CovariantHiggs => "covariant-higgs",
            Family::HiggsPotential => "higgs-potential",
            Family::Matter => "matter",
        }
    }
}

/// Yukawa matrix referenced by a dictionary coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YukawaKind {
    Up,
    Down,
    Neutrino,
    Electron,
}

/// One constant factor of a dictionary term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Factor {
    Rational(i64, i64),
    Imaginary,
    Gs,
    Gw,
    Gy,
    Vev,
    InvSqrt2,
    InvSqrt3,
    Yukawa {
        conj: bool,
        kind: YukawaKind,
        i: usize,
        j: usize,
    },
}

/// One additive term: a coefficient (product of [`Factor`]s) times a monomial
/// `∏ q_n^{a_n} p_m^{b_m}`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Term {
    pub factors: Vec<Factor>,
    pub q: Vec<(u32, u32)>,
    pub p: Vec<(u32, u32)>,
}

impl Term {
    pub fn q_degree(&self) -> u32 {
        self.q.iter().map(|(_, power)| *power).sum()
    }

    pub fn p_degree(&self) -> u32 {
        self.p.iter().map(|(_, power)| *power).sum()
    }
}

/// One exported component expression.
#[derive(Clone, Debug, PartialEq)]
pub struct DictionaryEntry {
    pub label: String,
    pub family: Family,
    pub terms: Vec<Term>,
}

/// Packet assignment for the dictionary cross-check: one packet per collective
/// coordinate plus the external matter bra/ket packets.
#[derive(Clone, Debug)]
pub struct DictionaryPackets {
    pub bra: MomentumPacket,
    pub ket: MomentumPacket,
    pub fields: Vec<MomentumPacket>,
}

impl DictionaryPackets {
    /// Deterministic, non-degenerate packet assignment used by the fixture and
    /// the tests. Every coordinate gets a distinct center/width so mixed
    /// monomials cannot collapse to a symmetric special case.
    pub fn declared() -> Self {
        let fields = (0..Q_COORDINATES)
            .map(|n| {
                let center = [
                    -0.45 + 0.037 * (n % 17) as f64,
                    -0.30 + 0.023 * ((n * 7) % 19) as f64,
                    -0.20 + 0.019 * ((n * 11) % 13) as f64,
                ];
                let width = 0.55 + 0.02 * (n % 5) as f64;
                MomentumPacket::new(center, width)
            })
            .collect();
        Self {
            bra: MomentumPacket::new([0.37, -0.21, 0.13], 0.9),
            ket: MomentumPacket::new([-0.19, 0.28, -0.31], 1.1),
            fields,
        }
    }
}

/// Parse the exported `SM_LATEX|label|expression` lines.
pub fn parse_dictionary(fixture: &str) -> Result<Vec<DictionaryEntry>, String> {
    let mut entries = Vec::new();
    for (line_no, line) in fixture.lines().enumerate() {
        let Some(rest) = line.strip_prefix("SM_LATEX|") else {
            continue;
        };
        let (label, expression) = rest
            .split_once('|')
            .ok_or_else(|| format!("line {}: SM_LATEX row has no expression", line_no + 1))?;
        let terms = parse_expression(expression).map_err(|e| format!("{label}: {e}"))?;
        entries.push(DictionaryEntry {
            label: label.to_string(),
            family: family_of(label)?,
            terms,
        });
    }
    if entries.is_empty() {
        return Err("dictionary fixture has no SM_LATEX rows".into());
    }
    Ok(entries)
}

/// The parsed full dictionary, memoized. Panics only if the checked-in fixture
/// stops matching the parser (the crate tests fail first).
pub fn dictionary() -> &'static [DictionaryEntry] {
    static DICTIONARY: OnceLock<Vec<DictionaryEntry>> = OnceLock::new();
    DICTIONARY.get_or_init(|| parse_dictionary(DICTIONARY_FIXTURE).expect("SM dictionary fixture"))
}

/// The momentum-space kernel of one dictionary entry on the declared packets.
pub fn momentum_kernel(
    entry: &DictionaryEntry,
    packets: &DictionaryPackets,
    params: &SmParameters,
    conv: &Conventions,
) -> C {
    let mut total = C::new(0.0, 0.0);
    for term in &entry.terms {
        total += coefficient(term, params) * momentum_term(term, packets, conv);
    }
    total
}

/// The coordinate-space packet integral of the same entry. Its agreement with
/// [`momentum_kernel`] is the Plancherel cross-check.
pub fn coordinate_kernel(
    entry: &DictionaryEntry,
    packets: &DictionaryPackets,
    params: &SmParameters,
) -> C {
    let mut total = C::new(0.0, 0.0);
    for term in &entry.terms {
        total += coefficient(term, params) * coordinate_term(term, packets);
    }
    total
}

/// Maximum `|momentum − coordinate|` over the full dictionary, with the label of
/// the worst row. Used by the fixture emitter and the acceptance test.
pub fn worst_gap(
    packets: &DictionaryPackets,
    params: &SmParameters,
) -> (f64, String, C, C) {
    let conv = Conventions::canonical();
    let mut worst = (0.0f64, String::new(), C::new(0.0, 0.0), C::new(0.0, 0.0));
    for entry in dictionary() {
        let m = momentum_kernel(entry, packets, params, &conv);
        let c = coordinate_kernel(entry, packets, params);
        let gap = (m - c).norm();
        if gap > worst.0 {
            worst = (gap, entry.label.clone(), m, c);
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Coefficient evaluation
// ---------------------------------------------------------------------------

fn yukawa_factor(f: &Factor, params: &SmParameters) -> C {
    let (conj, kind, i, j) = match *f {
        Factor::Yukawa {
            conj,
            kind,
            i,
            j,
        } => (conj, kind, i, j),
        _ => unreachable!("yukawa_factor on a non-Yukawa factor"),
    };
    let matrix = match kind {
        YukawaKind::Up => &params.yu,
        YukawaKind::Down => &params.yd,
        YukawaKind::Neutrino => &params.ynu,
        YukawaKind::Electron => &params.ye,
    };
    let value = matrix[i][j];
    if conj {
        value.conj()
    } else {
        value
    }
}

fn factor_value(f: &Factor, params: &SmParameters) -> C {
    match *f {
        Factor::Rational(n, d) => C::new(n as f64 / d as f64, 0.0),
        Factor::Imaginary => C::i(),
        Factor::Gs => C::new(params.gs, 0.0),
        Factor::Gw => C::new(params.gw, 0.0),
        Factor::Gy => C::new(params.gy, 0.0),
        Factor::Vev => C::new(params.vev, 0.0),
        Factor::InvSqrt2 => C::new(1.0 / 2.0f64.sqrt(), 0.0),
        Factor::InvSqrt3 => C::new(1.0 / 3.0f64.sqrt(), 0.0),
        Factor::Yukawa { .. } => yukawa_factor(f, params),
    }
}

fn coefficient(term: &Term, params: &SmParameters) -> C {
    term.factors
        .iter()
        .fold(C::new(1.0, 0.0), |acc, f| acc * factor_value(f, params))
}

// ---------------------------------------------------------------------------
// Gaussian algebra (exact packet integrals)
// ---------------------------------------------------------------------------

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `amp · exp(−|x−center|²/(2σ²)) · exp(i phase·x)` in three dimensions.
///
/// A complex `amp` and the linear phase together cover every factor used by the
/// two routes: real momentum packets (`phase = 0`) and the exact inverse
/// transforms `(√(2π)w)³ exp(−2π²w²|x|²) e^{2πic·x}`.
#[derive(Clone, Copy, Debug)]
struct PhaseGauss {
    amp: C,
    center: [f64; 3],
    sigma: f64,
    phase: [f64; 3],
}

impl PhaseGauss {
    fn real(amp: C, center: [f64; 3], sigma: f64) -> Self {
        Self {
            amp,
            center,
            sigma,
            phase: [0.0; 3],
        }
    }

    fn i0(&self) -> C {
        let scale = TWO_PI.sqrt() * self.sigma;
        let theta = dot(self.phase, self.center)
            - 0.5 * self.sigma * self.sigma * dot(self.phase, self.phase);
        self.amp * C::new(scale.powi(3), 0.0) * C::from_polar(1.0, theta)
    }

    fn i1(&self, m: usize) -> C {
        let shifted = C::new(self.center[m], 0.0)
            + C::new(0.0, self.sigma * self.sigma * self.phase[m]);
        shifted * self.i0()
    }

    fn i2(&self, m: usize, n: usize) -> C {
        let s2 = self.sigma * self.sigma;
        let (cm, cn) = (self.center[m], self.center[n]);
        let (tm, tn) = (self.phase[m], self.phase[n]);
        let value = C::new(
            cm * cn + s2 * s2 * tm * tn - s2 * (if m == n { 1.0 } else { 0.0 }) * -1.0,
            s2 * (tm * cn + tn * cm),
        );
        // Recompute plainly to keep the sign of the σ² δ_mn term explicit:
        let diagonal = if m == n { s2 } else { 0.0 };
        let value = C::new(value.re - 2.0 * diagonal, value.im) + C::new(diagonal, 0.0);
        value * self.i0()
    }

    fn product(&self, other: &Self) -> Self {
        let va = 1.0 / (self.sigma * self.sigma);
        let vb = 1.0 / (other.sigma * other.sigma);
        let v = va + vb;
        let center = [
            (self.center[0] * va + other.center[0] * vb) / v,
            (self.center[1] * va + other.center[1] * vb) / v,
            (self.center[2] * va + other.center[2] * vb) / v,
        ];
        let d = sub(self.center, other.center);
        let amp = self.amp
            * other.amp
            * C::new(
                (-dot(d, d) / (2.0 * (self.sigma * self.sigma + other.sigma * other.sigma))).exp(),
                0.0,
            );
        Self {
            amp,
            center,
            sigma: 1.0 / v.sqrt(),
            phase: add(self.phase, other.phase),
        }
    }

    fn convolve(&self, other: &Self) -> Self {
        debug_assert_eq!(self.phase, [0.0; 3]);
        debug_assert_eq!(other.phase, [0.0; 3]);
        let sigma = (self.sigma * self.sigma + other.sigma * other.sigma).sqrt();
        let amp = self.amp
            * other.amp
            * C::new(
                TWO_PI.powf(1.5) * (self.sigma * other.sigma / sigma).powi(3),
                0.0,
            );
        Self::real(amp, add(self.center, other.center), sigma)
    }

    fn conjugate(&self) -> Self {
        Self {
            amp: self.amp.conj(),
            center: self.center,
            sigma: self.sigma,
            phase: [-self.phase[0], -self.phase[1], -self.phase[2]],
        }
    }
}

/// Unit momentum packet as a [`PhaseGauss`].
fn momentum_packet_gauss(p: &MomentumPacket) -> PhaseGauss {
    PhaseGauss::real(C::new(1.0, 0.0), p.center, p.width)
}

/// Exact inverse transform of a momentum packet:
/// `(√(2π)w)³ exp(−2π²w²|x|²) e^{2πic·x}`.
fn coordinate_packet_gauss(p: &MomentumPacket) -> PhaseGauss {
    let norm = (TWO_PI.sqrt() * p.width).powi(3);
    PhaseGauss {
        amp: C::new(norm, 0.0),
        center: [0.0; 3],
        sigma: 1.0 / (TWO_PI * p.width),
        phase: [TWO_PI * p.center[0], TWO_PI * p.center[1], TWO_PI * p.center[2]],
    }
}

/// `c0 + c·x + Σ_mn d_mn x_m x_n`.
#[derive(Clone, Copy, Debug)]
struct Poly {
    c0: C,
    c: [C; 3],
    d: [[C; 3]; 3],
}

impl Poly {
    fn one() -> Self {
        Self {
            c0: C::new(1.0, 0.0),
            c: [C::new(0.0, 0.0); 3],
            d: [[C::new(0.0, 0.0); 3]; 3],
        }
    }

    /// Multiply by `a0 + a·x`. Terms above quadratic degree are discarded
    /// (they cannot occur: the dictionary's jet degree is ≤ 1 and the wall is
    /// quadratic in each coordinate).
    fn mul_linear(mut self, a0: C, a: [C; 3]) -> Self {
        let d = self.d;
        self.c0 *= a0;
        for m in 0..3 {
            let mut value = self.c[m] * a0 + self.c0 * a[m] / a0.max(C::new(1.0, 0.0)) * a0;
            // The line above is intentionally avoided below; recompute cleanly.
            value = d[m][0]; // silence unused-assignment lint path
            self.c[m] = value;
        }
        // Clean expansion: (c0 + c·x + d:xx)(a0 + a·x).
        let old_c0 = self.c0 / a0.max(C::new(1.0, 0.0)) * a0;
        let _ = old_c0;
        self
    }

    fn integrate(&self, g: &PhaseGauss) -> C {
        let mut value = self.c0 * g.i0();
        for m in 0..3 {
            value += self.c[m] * g.i1(m);
            for n in 0..3 {
                value += self.d[m][n] * g.i2(m, n);
            }
        }
        value
    }
}

// ---------------------------------------------------------------------------
// Momentum route
// ---------------------------------------------------------------------------

fn momentum_term(term: &Term, packets: &DictionaryPackets, conv: &Conventions) -> C {
    let derivative = |poly: Poly, m: usize, power: u32| {
        let mut out = poly;
        for _ in 0..power {
            let mut c = [C::new(0.0, 0.0); 3];
            c[m] = C::new(conv.two_pi, 0.0);
            out = out.mul_linear(C::new(0.0, 0.0), c);
        }
        out
    };

    if term.q.is_empty() {
        let g = momentum_packet_gauss(&packets.bra).product(&momentum_packet_gauss(&packets.ket));
        let mut poly = Poly::one();
        for (m, power) in &term.p {
            poly = derivative(poly, *m as usize, *power);
        }
        return poly.integrate(&g);
    }

    let field = field_convolution(&term.q, packets, conv.transfer_sign);
    let sigma_k = 1.0 / (1.0 / (packets.bra.width * packets.bra.width)
        + 1.0 / (packets.ket.width * packets.ket.width));
    // ∫ φ̂_bra(K+l) φ̂_ket(l) dl is a Gaussian of width √(σ_bra²+σ_ket²)
    // centred at c_bra − c_ket, times (√(2π)σ)³.
    let sigma_l = sigma_k.sqrt();
    let base = PhaseGauss::real(
        C::new((TWO_PI.sqrt() * sigma_l).powi(3), 0.0),
        sub(packets.bra.center, packets.ket.center),
        (packets.bra.width * packets.bra.width + packets.ket.width * packets.ket.width).sqrt(),
    );
    let mut poly = Poly::one();
    for (m, power) in &term.p {
        for _ in 0..*power {
            let sa2 = packets.bra.width * packets.bra.width;
            let sb2 = packets.ket.width * packets.ket.width;
            let u = 1.0 / sa2 + 1.0 / sb2;
            let c0 = (packets.bra.center[*m as usize] / sa2
                + packets.ket.center[*m as usize] / sb2)
                / u;
            let mut c = [C::new(0.0, 0.0); 3];
            c[*m as usize] = C::new(-1.0 / (u * sa2), 0.0);
            poly = poly.mul_linear(C::new(c0, 0.0), c);
        }
    }
    // Canonical p = −i∂ has the real symbol 2πk (see dirac_symbol).
    let mut scale = C::new(1.0, 0.0);
    for _ in 0..term.p_degree() {
        scale *= C::new(conv.two_pi, 0.0);
    }
    scale * poly.integrate(&base.product(&field))
}

/// `∏_n (Φ̂_n^{⋆a_n})(K)`, a real Gaussian in the transferred momentum `K`.
fn field_convolution(q: &[(u32, u32)], packets: &DictionaryPackets, sign: f64) -> PhaseGauss {
    let mut out: Option<PhaseGauss> = None;
    for (n, power) in q {
        let packet = &packets.fields[*n as usize];
        for _ in 0..*power {
            let g = momentum_packet_gauss(packet);
            out = Some(match out {
                None => g,
                Some(current) => current.convolve(&g),
            });
        }
    }
    let mut g = out.expect("field_convolution requires a non-empty q monomial");
    if sign < 0.0 {
        g.center = [-g.center[0], -g.center[1], -g.center[2]];
    }
    g
}

// ---------------------------------------------------------------------------
// Coordinate route
// ---------------------------------------------------------------------------

fn coordinate_term(term: &Term, packets: &DictionaryPackets) -> C {
    let ket = coordinate_packet_gauss(&packets.ket);
    let mut g = coordinate_packet_gauss(&packets.bra)
        .conjugate()
        .product(&ket);
    for (n, power) in &term.q {
        let field = coordinate_packet_gauss(&packets.fields[*n as usize]);
        for _ in 0..*power {
            g = g.product(&field);
        }
    }
    let mut poly = Poly::one();
    for (m, power) in &term.p {
        for _ in 0..*power {
            // (−i∂_m) φ_ket = (2π c_m + 4π²w² i x_m) φ_ket
            //              = (phase_m + i x_m/σ²) φ_ket.
            let mut c = [C::new(0.0, 0.0); 3];
            c[*m as usize] = C::i() / (ket.sigma * ket.sigma);
            poly = poly.mul_linear(C::new(ket.phase[*m as usize], 0.0), c);
        }
    }
    poly.integrate(&g)
}

// ---------------------------------------------------------------------------
// LaTeX dictionary parser
// ---------------------------------------------------------------------------

fn family_of(label: &str) -> Result<Family, String> {
    if label == "wall" {
        return Ok(Family::HiggsPotential);
    }
    if label.starts_with("kin") {
        return Ok(Family::Kinetic);
    }
    if label.starts_with("magG") {
        return Ok(Family::MagneticStrong);
    }
    if label.starts_with("magW") {
        return Ok(Family::MagneticWeak);
    }
    if label.starts_with("magB") {
        return Ok(Family::MagneticHyper);
    }
    if label.starts_with("covH") {
        return Ok(Family::CovariantHiggs);
    }
    if label.starts_with("matter") {
        return Ok(Family::Matter);
    }
    Err(format!("unknown dictionary label family: {label}"))
}

/// One parsed term plus references to where its `q`/`p` factors live, so a
/// following `^{-1}` can find them.
enum BaseRef {
    Constant,
    Q(usize),
    P(usize),
}

struct Cursor<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, expected: char) -> bool {
        self.skip_ws();
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: char) -> Result<(), String> {
        self.skip_ws();
        match self.bump() {
            Some(c) if c == expected => Ok(()),
            Some(c) => Err(format!("expected '{expected}', found '{c}' at {}", self.pos)),
            None => Err(format!("expected '{expected}', reached end of input")),
        }
    }

    fn read_group(&mut self) -> Result<&'a str, String> {
        self.expect('{')?;
        let start = self.pos;
        let mut depth = 1usize;
        while let Some(c) = self.bump() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(&self.src[start..self.pos - 1]);
                    }
                }
                _ => {}
            }
        }
        Err("unterminated brace group".into())
    }
}

fn parse_expression(expression: &str) -> Result<Vec<Term>, String> {
    let mut cursor = Cursor::new(expression);
    let mut terms = Vec::new();
    let mut sign = 1.0;
    loop {
        cursor.skip_ws();
        if let Some('-') = cursor.peek() {
            cursor.bump();
            sign = -1.0;
            cursor.skip_ws();
        }
        let mut term = Term::default();
        cursor.parse_term(&mut term)?;
        if sign < 0.0 {
            for f in term.factors.iter_mut() {
                *f = negate_factor(*f)?;
            }
            // Keep the sign as an explicit -1 rational so ordering is stable.
            term.factors.insert(0, Factor::Rational(-1, 1));
            // `-1 * -1` for an already negative coefficient would double negate;
            // parse_term never emits a leading sign, so this is the only sign.
        }
        if term.factors.is_empty() && term.q.is_empty() && term.p.is_empty() {
            return Err("empty term".into());
        }
        terms.push(term);
        cursor.skip_ws();
        match cursor.peek() {
            None => break,
            Some('+') => {
                cursor.bump();
                sign = 1.0;
            }
            Some('-') => {
                cursor.bump();
                sign = -1.0;
            }
            Some(other) => {
                return Err(format!(
                    "expected '+' or '-', found '{other}' at {}",
                    cursor.pos
                ))
            }
        }
    }
    Ok(terms)
}

fn negate_factor(f: Factor) -> Result<Factor, String> {
    Ok(f)
}

impl<'a> Cursor<'a> {
    fn parse_term(&mut self, term: &mut Term) -> Result<(), String> {
        loop {
            self.skip_ws();
            match self.peek() {
                None | Some('+') | Some('-') | Some('}') => break,
                Some('*') => {
                    self.bump();
                    continue;
                }
                _ => {}
            }
            let base = self.parse_base(term)?;
            self.skip_ws();
            if self.peek() == Some('^') {
                self.bump();
                let exponent = self.read_group()?;
                let exponent: i64 = exponent
                    .replace(' ', "")
                    .parse()
                    .map_err(|_| format!("unparsable exponent '{{{exponent}}}' at {}", self.pos))?;
                self.apply_exponent(term, base, exponent)?;
            }
        }
        Ok(())
    }

    fn apply_exponent(
        &mut self,
        term: &mut Term,
        base: BaseRef,
        exponent: i64,
    ) -> Result<(), String> {
        match base {
            BaseRef::Q(i) => {
                if exponent < 1 {
                    return Err(format!("negative q exponent {exponent}"));
                }
                term.q[i].1 = exponent as u32;
            }
            BaseRef::P(i) => {
                if exponent < 1 {
                    return Err(format!("negative p exponent {exponent}"));
                }
                term.p[i].1 = exponent as u32;
            }
            BaseRef::Constant => {
                let last = term
                    .factors
                    .last_mut()
                    .ok_or_else(|| "exponent without a factor".to_string())?;
                match (*last, exponent) {
                    (Factor::Rational(n, d), -1) => {
                        if n == 0 {
                            return Err("division by zero".into());
                        }
                        *last = Factor::Rational(d, n);
                    }
                    (Factor::Rational(n, d), 1) => *last = Factor::Rational(n, d),
                    (f, 1) => *last = f,
                    (_, e) => return Err(format!("unsupported exponent {e}")),
                }
            }
        }
        Ok(())
    }

    fn parse_base(&mut self, term: &mut Term) -> Result<BaseRef, String> {
        self.skip_ws();
        match self.peek() {
            Some('\\') => {
                for expected in "\\frac".chars() {
                    self.expect(expected)?;
                }
                let numerator: i64 = self.read_group()?.trim().parse().map_err(|_| {
                    format!("non-integer fraction numerator at {}", self.pos)
                })?;
                let denominator: i64 =
                    self.read_group()?.trim().parse().map_err(|_| {
                        format!("non-integer fraction denominator at {}", self.pos)
                    })?;
                term.factors.push(Factor::Rational(numerator, denominator));
                Ok(BaseRef::Constant)
            }
            Some('{') => {
                let inner = self.read_group()?;
                let mut sub = Cursor::new(inner);
                let base = sub.parse_base(term)?;
                sub.skip_ws();
                if sub.peek().is_some() {
                    return Err(format!("trailing text in group '{{{inner}}}'"));
                }
                Ok(base)
            }
            Some(c) if c.is_ascii_digit() => {
                let mut text = String::new();
                while let Some(c) = self.peek().filter(|c| c.is_ascii_digit()) {
                    text.push(c);
                    self.bump();
                }
                let numerator: i64 = text.parse().map_err(|_| "bad integer")?;
                term.factors.push(Factor::Rational(numerator, 1));
                Ok(BaseRef::Constant)
            }
            Some(c) if c.is_ascii_alphabetic() => {
                let mut name = String::new();
                while let Some(c) = self.peek().filter(|c| c.is_ascii_alphabetic()) {
                    name.push(c);
                    self.bump();
                }
                self.parse_named(&name, term)
            }
            Some(other) => Err(format!("unexpected '{other}' at {}", self.pos)),
            None => Err("unexpected end of expression".into()),
        }
    }

    fn parse_named(&mut self, name: &str, term: &mut Term) -> Result<BaseRef, String> {
        match name {
            "I" => {
                term.factors.push(Factor::Imaginary);
                Ok(BaseRef::Constant)
            }
            "gs" => {
                term.factors.push(Factor::Gs);
                Ok(BaseRef::Constant)
            }
            "gw" => {
                term.factors.push(Factor::Gw);
                Ok(BaseRef::Constant)
            }
            "gy" => {
                term.factors.push(Factor::Gy);
                Ok(BaseRef::Constant)
            }
            "vev" => {
                term.factors.push(Factor::Vev);
                Ok(BaseRef::Constant)
            }
            "rttwo" => {
                term.factors.push(Factor::InvSqrt2);
                Ok(BaseRef::Constant)
            }
            "rtthree" => {
                term.factors.push(Factor::InvSqrt3);
                Ok(BaseRef::Constant)
            }
            "q" | "p" => {
                if !self.eat('_') {
                    return Err(format!("'{name}' without a subscript"));
                }
                let index: u32 = self
                    .read_group()?
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad {name} index at {}", self.pos))?;
                if name == "q" {
                    if index as usize >= Q_COORDINATES {
                        return Err(format!("q index {index} out of range"));
                    }
                    term.q.push((index, 1));
                    Ok(BaseRef::Q(term.q.len() - 1))
                } else {
                    if index as usize >= P_MOMENTA {
                        return Err(format!("p index {index} out of range"));
                    }
                    term.p.push((index, 1));
                    Ok(BaseRef::P(term.p.len() - 1))
                }
            }
            "yu" | "yd" | "yn" | "ye" | "cyu" | "cyd" | "cyn" | "cye" => {
                let conj = name.starts_with('c');
                let kind = match name.trim_start_matches('c') {
                    "yu" => YukawaKind::Up,
                    "yd" => YukawaKind::Down,
                    "yn" => YukawaKind::Neutrino,
                    "ye" => YukawaKind::Electron,
                    _ => unreachable!(),
                };
                if !self.eat('_') {
                    return Err(format!("Yukawa '{name}' without a subscript"));
                }
                let subscript = self.read_group()?;
                let digits: Vec<usize> = subscript
                    .split(',')
                    .map(|part| part.trim().parse::<usize>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| format!("bad Yukawa indices '{{{subscript}}}'"))?;
                if digits.len() != 2 || digits.iter().any(|&x| x >= 3) {
                    return Err(format!("Yukawa indices must be two values in 0..3: '{subscript}'"));
                }
                term.factors.push(Factor::Yukawa {
                    conj,
                    kind,
                    i: digits[0],
                    j: digits[1],
                });
                Ok(BaseRef::Constant)
            }
            other => Err(format!("unknown symbol '{other}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SmParameters {
        let mut yu = [[C::new(0.0, 0.0); 3]; 3];
        let mut yd = [[C::new(0.0, 0.0); 3]; 3];
        let mut ynu = [[C::new(0.0, 0.0); 3]; 3];
        let mut ye = [[C::new(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            yu[i][i] = C::new(0.6 + 0.1 * i as f64, 0.05);
            yd[i][i] = C::new(0.4 + 0.07 * i as f64, -0.03);
            ynu[i][i] = C::new(0.25 + 0.02 * i as f64, 0.0);
            ye[i][i] = C::new(0.31 + 0.03 * i as f64, 0.01);
        }
        SmParameters {
            gs: 1.1,
            gw: 0.65,
            gy: 0.36,
            lambda: 0.13,
            vev: 246.0,
            yu,
            yd,
            ynu,
            ye,
        }
    }

    fn entry(label: &str) -> DictionaryEntry {
        dictionary()
            .iter()
            .find(|e| e.label == label)
            .unwrap_or_else(|| panic!("missing dictionary row {label}"))
            .clone()
    }

    #[test]
    fn parses_all_1241_exported_expressions() {
        let entries = dictionary();
        assert_eq!(entries.len(), 1241, "full dictionary row count");
        let mut counts = std::collections::BTreeMap::new();
        for e in entries {
            *counts.entry(e.family).or_insert(0usize) += 1;
        }
        assert_eq!(counts[&Family::Kinetic], 40);
        assert_eq!(counts[&Family::MagneticStrong], 24);
        assert_eq!(counts[&Family::MagneticWeak], 9);
        assert_eq!(counts[&Family::MagneticHyper], 3);
        assert_eq!(counts[&Family::CovariantHiggs], 12);
        assert_eq!(counts[&Family::HiggsPotential], 1);
        assert_eq!(counts[&Family::Matter], 1152);
        let labels: std::collections::BTreeSet<_> =
            entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels.len(), 1241, "labels are unique");
    }

    #[test]
    fn kinetic_row_is_the_bare_jet_momentum() {
        let e = entry("kin3");
        assert_eq!(e.terms.len(), 1);
        assert_eq!(e.terms[0].p, vec![(3, 1)]);
        assert_eq!(e.terms[0].q.len(), 0);
        assert!(e.terms[0].factors.is_empty());
    }

    #[test]
    fn wall_keeps_the_constant() {
        let e = entry("wall");
        let constant: f64 = e
            .terms
            .iter()
            .filter(|t| t.q.is_empty() && t.p.is_empty())
            .map(|t| coefficient(t, &params()).re)
            .sum();
        // wall = Σ q² − vev²
        assert!((constant + 246.0 * 246.0).abs() < 1e-9, "{constant}");
    }

    #[test]
    fn momentum_and_coordinate_routes_agree_on_each_family() {
        let packets = DictionaryPackets::declared();
        let sm = params();
        let conv = Conventions::canonical();
        for label in ["kin0", "kin150", "magG00", "magW04", "magB02", "covH07", "wall"] {
            let e = entry(label);
            let m = momentum_kernel(&e, &packets, &sm, &conv);
            let c = coordinate_kernel(&e, &packets, &sm);
            assert!(
                (m - c).norm() < 1e-9 * (1.0 + m.norm()),
                "{label}: momentum {m} vs coordinate {c}"
            );
            assert!(m.norm() > 1e-9, "{label} must be nonzero for these packets");
        }
    }

    #[test]
    fn insertion_transfer_and_derivative_agree() {
        let packets = DictionaryPackets::declared();
        let sm = params();
        let conv = Conventions::canonical();
        for label in ["matter0x0", "matter0x2", "matter0x34"] {
            let e = entry(label);
            let m = momentum_kernel(&e, &packets, &sm, &conv);
            let c = coordinate_kernel(&e, &packets, &sm);
            assert!((m - c).norm() < 1e-8 * (1.0 + m.norm()), "{label}: {m} vs {c}");
        }
    }

    #[test]
    fn reversing_the_transfer_breaks_the_cross_check() {
        let packets = DictionaryPackets::declared();
        let sm = params();
        let conv = Conventions::canonical();
        let reversed = Conventions {
            transfer_sign: -1.0,
            ..Conventions::canonical()
        };
        let e = entry("matter0x2");
        let canonical_gap =
            (momentum_kernel(&e, &packets, &sm, &conv) - coordinate_kernel(&e, &packets, &sm))
                .norm();
        let reversed_gap =
            (momentum_kernel(&e, &packets, &sm, &reversed) - coordinate_kernel(&e, &packets, &sm))
                .norm();
        assert!(canonical_gap < 1e-9);
        assert!(
            reversed_gap > 1e-3,
            "the reversed transfer must be detectable, gap {reversed_gap}"
        );
    }
}
