//! Full Standard-Model kernel over the **1,241-expression dictionary**.
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
//! Every row is lifted to a kernel on smeared packet states and evaluated by
//! **two independent routes** whose agreement is Plancherel's theorem term by
//! term:
//!
//! ```text
//! coordinate:  ⟨φ_b| P(q̂, p̂) |φ_k⟩ = ∫ φ_b*(q) P(q, −i∂_q) φ_k(q) d^163q
//! momentum:    ∫ φ̂_b*(k) P((i/2π)∂_k, 2πk) φ̂_k(k) d^163k
//! ```
//!
//! ## What `q` and `p` mean here
//!
//! This is the **E1 collective-coordinate operator**, not a 3D field kernel.
//! `sm_full.rs` defines 163 independent bosonic collective coordinates
//! (`x; G,dG; W,dW; B,dB; phi,dphi`) and treats `p_m` as the conjugate momentum
//! of coordinate `m`, so `p̂_m = −i∂/∂q_m` and `q̂_n` is multiplication. The
//! dictionary is at most quadratic in `q` and linear in `p`, and **no term mixes
//! `q` and `p`** (verified over the fixture), so normal ordering is never
//! ambiguous. `p_0..p_150` are exactly the momenta of the active coordinates;
//! the twelve `dphi` jets have none, matching `sm_full`'s `1/2 Σ_active p²`.
//!
//! This deliberately does *not* reuse the 3D packet/convolution routing of
//! [`crate::sm_momentum`]. That module kernels the hand-written SM matter and
//! gauge entries, where `p_j` is literally the spatial Dirac momentum; here the
//! dictionary's `p_m` is the canonical momentum of a named collective
//! coordinate, which is a spatial derivative only for `m < 3`. `SM_HAMILTONIAN_AUDIT.md`
//! states the same boundary: "spatial Dirac derivatives act at fixed independent
//! collective coordinates; NS/QG jet elimination is not used to change this SM
//! realization."
//!
//! ## Per-mode 3D kernel
//!
//! [`three_d_kernel`] evaluates the same expressions as a **3D field-mode**
//! kernel (the routing of [`crate::sm_momentum`]): every `q_n` inserts a 3D
//! packet and products transfer momentum `k − l`, while `p_m` acts as the
//! spatial derivative `−i∂_{axis(m)}` on the ket. The axis is the named
//! layout's own: for `x`, `G`, `W`, `B` and their jets the block index is
//! `m mod 3` (true for every block base `0, 27, 99, 108, 135, 138`), and the
//! four Higgs momenta `p_147..150` — scalars, which have no spatial axis — get
//! the declared profile axis `m mod 3`. So `axis_of(m) = m mod 3` throughout.
//!
//! The two kernels are the **same operator only on the derivative terms**. A
//! `q`-free term is `p_m` acting on the ket in both, so with `O_j` the term's
//! one-axis overlap on residue `j` (no derivative factor) the two values are
//! related exactly, because every packet is separable and its per-axis
//! parameters depend on the axis only through `a mod 3` and the derivative
//! factor is the same on `m` and `m mod 3`:
//!
//! ```text
//! collective = three_d · O_0^54 · O_1^53 · O_2^53     (axes 3..162 contribute
//!                                                     54/53/53 spectators)
//! ```
//!
//! That is checked term by term over the 232 p-bearing rows, so the 3D axis map
//! and derivative routing are validated against the collective kernel, not
//! against a re-typed formula. A `q`-bearing term is **not** comparable: `q` is
//! a coordinate multiplication in the collective kernel and a field insertion
//! with momentum transfer in the 3D kernel (a different operator).
//!
//! The `q`-bearing rows are therefore cross-checked inside each route. Every
//! closed-form value is compared with a trapezoid evaluation of the same
//! integrand ([`coordinate_term_quadrature`], [`coordinate_term_3d_quadrature`]),
//! and for a `p`-free term the integrand is assembled from the term and the
//! packet layout instead of from the closed-form polynomial
//! ([`coordinate_term_q_only_quadrature`],
//! [`coordinate_term_3d_q_only_quadrature`]) — so the check is sensitive to
//! which mode is inserted, on which axis, and with which weight. The 3D
//! insertion is a profile along **every** axis; `axis_of` routes `p`, not `q`.
//!
//! ## Packets
//!
//! Momentum packets are L²-normalised Gaussians in `D = 163` dimensions,
//!
//! ```text
//! φ̂_n(k) = (π w_n²)^{−D/4} exp(−|k − c_n|²/(2 w_n²)),
//! ```
//!
//! and their exact inverse transforms are used on the coordinate side,
//!
//! ```text
//! φ_n(q) = (4π w_n²)^{D/4} exp(−2π² w_n² |q|²) e^{2πi c_n·q}.
//! ```
//!
//! Both routes are evaluated in closed form: a Gaussian times a polynomial of
//! degree ≤ 2 is integrated exactly from its moments, so there is no quadrature
//! error and the tolerance below measures floating-point roundoff only.
//!
//! ## Scope
//!
//! This certifies that all 1,241 exported expressions parse, that their
//! coefficients and monomial structure evaluate identically in the position and
//! momentum representations, and that the `p`/`q` routing and Fourier constants
//! are consistent. It does not re-derive the classical dictionary (that is the
//! Cadabra module's job) and it does not certify ESA or BRST.

use crate::sm_full::SmParameters;
use crate::sm_momentum::{Conventions, TWO_PI};
use num_complex::Complex64 as C;
#[cfg(test)]
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The Cadabra export this module parses: 40 momenta, 49 bosonic forms and
/// 1,152 matter entries.
pub const DICTIONARY_FIXTURE: &str = include_str!("../../docs/fixtures/sm_full_cadabra.latex");

/// Number of independent collective coordinates (`q_0 ..= q_162`).
pub const Q_COORDINATES: usize = 163;

/// Number of independent jet momenta (`p_0 ..= p_150`).
pub const P_MOMENTA: usize = 151;

/// Dictionary family, derived from the Cadabra label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// Reciprocal of [`Factor::InvSqrt2`]; produced by `{rttwo}^{-1}`.
    Sqrt2,
    /// Reciprocal of [`Factor::InvSqrt3`]; produced by `{rtthree}^{-1}`.
    Sqrt3,
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

/// A centred, isotropic Gaussian packet in `D` collective dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    pub center: Vec<f64>,
    pub width: f64,
}

/// Packet assignment for the dictionary cross-check: one packet per collective
/// coordinate plus the external bra/ket packets.
#[derive(Clone, Debug)]
pub struct DictionaryPackets {
    pub bra: Packet,
    pub ket: Packet,
    pub fields: Vec<Packet>,
}

impl DictionaryPackets {
    /// Deterministic packet assignment used by the fixture and the tests.
    ///
    /// Centres are kept within a few tenths of a unit on every axis so that the
    /// 163-dimensional overlaps stay O(1) instead of underflowing, while still
    /// giving every coordinate a distinct centre and one of five widths.
    pub fn declared() -> Self {
        let d = Q_COORDINATES;
        // Per-axis parameters depend on the axis only through `a mod 3`. That
        // is what makes the collective kernel an exact product of its three
        // spatial residues and lets the 3D mode kernel be cross-checked against
        // it (`collective = three_d * O_0^54 O_1^53 O_2^53`).
        let fields = (0..d)
            .map(|n| Packet {
                center: (0..d)
                    .map(|a| -0.045 + 0.004 * (((n * 7 + (a % 3) * 11 + 3) % 23) as f64))
                    .collect(),
                width: 0.55 + 0.02 * (n % 5) as f64,
            })
            .collect();
        // The external packets are offset in opposite directions with a small
        // per-axis jitter. Their width-weighted sum `M_b c_b + M_k c_k` is then
        // strictly negative on every axis, so no kinetic row can have a zero
        // momentum expectation by accident (a coincidence that would hide a
        // wrong Fourier symbol on that row).
        Self {
            bra: Packet {
                center: (0..d)
                    .map(|a| -0.0281 + 0.0007 * ((a % 3) as f64))
                    .collect(),
                width: 0.9,
            },
            ket: Packet {
                center: (0..d).map(|a| 0.0137 - 0.0004 * ((a % 3) as f64)).collect(),
                width: 1.1,
            },
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

/// Deterministic, **fully populated** coupling/Yukawa set used by the fixture
/// and the acceptance tests. Every one of the nine entries of each matrix is
/// nonzero so that no dictionary row is exercised with a vanishing coefficient
/// by accident.
pub fn fixture_parameters() -> SmParameters {
    let mut yu = [[C::new(0.0, 0.0); 3]; 3];
    let mut yd = [[C::new(0.0, 0.0); 3]; 3];
    let mut ynu = [[C::new(0.0, 0.0); 3]; 3];
    let mut ye = [[C::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (i, j) = (i as f64, j as f64);
            yu[i as usize][j as usize] = C::new(0.31 + 0.07 * i + 0.03 * j, 0.05 + 0.01 * (i + j));
            yd[i as usize][j as usize] = C::new(0.22 + 0.05 * i - 0.02 * j, -0.03 + 0.01 * i * j);
            ynu[i as usize][j as usize] = C::new(0.15 + 0.04 * j, 0.02 * i);
            ye[i as usize][j as usize] = C::new(0.18 + 0.03 * i + 0.01 * j, -0.02);
        }
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

/// The parsed full dictionary, memoized. Panics only if the checked-in fixture
/// stops matching the parser (the crate tests fail first).
pub fn dictionary() -> &'static [DictionaryEntry] {
    static DICTIONARY: OnceLock<Vec<DictionaryEntry>> = OnceLock::new();
    DICTIONARY.get_or_init(|| parse_dictionary(DICTIONARY_FIXTURE).expect("SM dictionary fixture"))
}

/// The momentum-representation kernel of one dictionary entry.
///
/// Only [`Conventions::two_pi`] is read: it is the Fourier symbol of `p`.
/// `transfer_sign`/`nonabelian_sign` belong to the 3D field kernel in
/// [`crate::sm_momentum`] and have no meaning on collective coordinates.
pub fn momentum_kernel(
    entry: &DictionaryEntry,
    packets: &DictionaryPackets,
    params: &SmParameters,
    conv: &Conventions,
) -> C {
    entry
        .terms
        .iter()
        .map(|term| coefficient(term, params) * momentum_term(term, packets, conv))
        .sum()
}

/// The coordinate-representation kernel of the same entry. Its agreement with
/// [`momentum_kernel`] is the Plancherel cross-check.
pub fn coordinate_kernel(
    entry: &DictionaryEntry,
    packets: &DictionaryPackets,
    params: &SmParameters,
) -> C {
    entry
        .terms
        .iter()
        .map(|term| coefficient(term, params) * coordinate_term(term, packets))
        .sum()
}

/// The largest relative deviation `|momentum − coordinate| / (1 + |coordinate|)`
/// over the full dictionary, with the label of the worst row.
pub fn worst_gap(packets: &DictionaryPackets, params: &SmParameters) -> GapReport {
    let conv = Conventions::canonical();
    let mut report = GapReport {
        label: String::new(),
        relative_gap: 0.0,
        momentum: C::new(0.0, 0.0),
        coordinate: C::new(0.0, 0.0),
        rows: 0,
    };
    for entry in dictionary() {
        let m = momentum_kernel(entry, packets, params, &conv);
        let c = coordinate_kernel(entry, packets, params);
        let gap = (m - c).norm() / (1.0 + c.norm());
        report.rows += 1;
        if gap > report.relative_gap {
            report.label = entry.label.clone();
            report.relative_gap = gap;
            report.momentum = m;
            report.coordinate = c;
        }
    }
    report
}

/// Worst-row summary of the dictionary cross-check.
#[derive(Clone, Debug)]
pub struct GapReport {
    pub label: String,
    pub relative_gap: f64,
    pub momentum: C,
    pub coordinate: C,
    pub rows: usize,
}

// ---------------------------------------------------------------------------
// Coefficients
// ---------------------------------------------------------------------------

fn yukawa_factor(f: &Factor, params: &SmParameters) -> C {
    let (conj, kind, i, j) = match *f {
        Factor::Yukawa { conj, kind, i, j } => (conj, kind, i, j),
        _ => unreachable!("yukawa_factor on a non-Yukawa factor"),
    };
    let matrix = match kind {
        YukawaKind::Up => &params.yu,
        YukawaKind::Down => &params.yd,
        YukawaKind::Neutrino => &params.ynu,
        YukawaKind::Electron => &params.ye,
    };
    let value = matrix[i][j];
    if conj { value.conj() } else { value }
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
        Factor::Sqrt2 => C::new(2.0f64.sqrt(), 0.0),
        Factor::Sqrt3 => C::new(3.0f64.sqrt(), 0.0),
        Factor::Yukawa { .. } => yukawa_factor(f, params),
    }
}

fn coefficient(term: &Term, params: &SmParameters) -> C {
    term.factors
        .iter()
        .fold(C::new(1.0, 0.0), |acc, f| acc * factor_value(f, params))
}

// ---------------------------------------------------------------------------
// Closed-form Gaussian moments
// ---------------------------------------------------------------------------

/// `exp(log_amp − ½ Σ_i M_i x_i² + Σ_i b_i x_i)` on `R^D`.
///
/// A single complex `b_i` per axis covers both representations: momentum
/// packets have real `b = c` and zero phase, coordinate packets have
/// `b = 2πi c`. Keeping the centre implicit in `b` is what makes the product of
/// a phased packet with its conjugate exact — a real-centre-plus-phase form
/// would drop the phase-induced shift.
#[derive(Clone, Debug)]
struct Gauss {
    log_amp: C,
    inv_var: Vec<f64>,
    b: Vec<C>,
}

impl Gauss {
    fn i0(&self) -> C {
        let mut log = self.log_amp;
        for (m, b) in self.inv_var.iter().zip(&self.b) {
            // ∫exp(−½Mx² + bx)dx = √(2π/M)·exp(b²/(2M)).
            log += C::new(0.5 * (TWO_PI / m).ln(), 0.0) + *b * *b / C::new(2.0 * m, 0.0);
        }
        log.exp()
    }

    fn moment1(&self, axis: usize) -> C {
        self.b[axis] / C::new(self.inv_var[axis], 0.0)
    }

    /// Evaluate the function itself (used by the quadrature cross-checks).
    fn value(&self, x: &[f64]) -> C {
        let mut log = self.log_amp;
        for (i, &xi) in x.iter().enumerate() {
            log += C::new(-0.5 * self.inv_var[i] * xi * xi, 0.0) + self.b[i] * C::new(xi, 0.0);
        }
        log.exp()
    }

    fn moment2(&self, m: usize, n: usize) -> C {
        let mut value = self.b[m] * self.b[n] / C::new(self.inv_var[m] * self.inv_var[n], 0.0);
        if m == n {
            value += C::new(1.0 / self.inv_var[m], 0.0);
        }
        value
    }

    fn product(&self, other: &Self) -> Self {
        debug_assert_eq!(self.inv_var.len(), other.inv_var.len());
        Self {
            log_amp: self.log_amp + other.log_amp,
            inv_var: self
                .inv_var
                .iter()
                .zip(&other.inv_var)
                .map(|(a, b)| a + b)
                .collect(),
            b: self.b.iter().zip(&other.b).map(|(a, b)| a + b).collect(),
        }
    }

    fn conj(&self) -> Self {
        Self {
            log_amp: self.log_amp.conj(),
            inv_var: self.inv_var.clone(),
            b: self.b.iter().map(|x| x.conj()).collect(),
        }
    }

    /// `amp · exp(−|x−center|²/(2σ²))` in `center.len()` dimensions.
    fn from_center(amp: C, center: &[f64], sigma: f64) -> Self {
        let m = 1.0 / (sigma * sigma);
        let shift = 0.5 * m * center.iter().map(|c| c * c).sum::<f64>();
        Self {
            log_amp: amp.ln() - C::new(shift, 0.0),
            inv_var: vec![m; center.len()],
            b: center.iter().map(|&c| C::new(m * c, 0.0)).collect(),
        }
    }

    /// `(f ⋆ g)(K) = ∫ f(q) g(K−q) dq` for real, centre-carrying factors.
    fn convolve(&self, other: &Self) -> Self {
        debug_assert_eq!(self.b.len(), other.b.len());
        let mut log_amp = self.log_amp + other.log_amp;
        let mut inv_var = Vec::with_capacity(self.b.len());
        let mut b = Vec::with_capacity(self.b.len());
        for i in 0..self.b.len() {
            let ma = self.inv_var[i];
            let mb = other.inv_var[i];
            let m = ma * mb / (ma + mb);
            let ba = self.b[i];
            let bb = other.b[i];
            // ∫ f(q)g(K−q)dq in this representation:
            //   √(2π/M) exp((b_a−b_b)²/(2M)) exp(−½m|K−(c_a+c_b)|²),
            // with the two centre offsets already carried by b = M·c.
            let d = ba - bb;
            log_amp +=
                C::new(0.5 * (TWO_PI / (ma + mb)).ln(), 0.0) + d * d / C::new(2.0 * (ma + mb), 0.0);
            inv_var.push(m);
            b.push(m * (ba / ma + bb / mb));
        }
        Self {
            log_amp,
            inv_var,
            b,
        }
    }
}

/// L²-normalised momentum packet `(πw²)^{−D/4} exp(−|k−c|²/(2w²))`.
fn momentum_packet_gauss(p: &Packet) -> Gauss {
    let d = p.center.len() as f64;
    let inv_var = 1.0 / (p.width * p.width);
    // exp(−(k−c)²/(2w²)) = exp(−½Mk² + (Mc)k − ½Mc²), so the canonical linear
    // coefficient is M·c and the constant is folded into the amplitude.
    let shift = 0.5 * inv_var * p.center.iter().map(|c| c * c).sum::<f64>();
    let log_amp = -(d / 4.0) * (std::f64::consts::PI * p.width * p.width).ln() - shift;
    Gauss {
        log_amp: C::new(log_amp, 0.0),
        inv_var: vec![inv_var; p.center.len()],
        b: p.center.iter().map(|&c| C::new(inv_var * c, 0.0)).collect(),
    }
}

/// Exact inverse transform of [`momentum_packet_gauss`].
fn coordinate_packet_gauss(p: &Packet) -> Gauss {
    let d = p.center.len() as f64;
    // (2π)^{−D/4} w^{−D/2} · (√(2π) w)^D = (4π w²)^{D/4}
    let log_amp = (d / 4.0) * (4.0 * std::f64::consts::PI * p.width * p.width).ln();
    Gauss {
        log_amp: C::new(log_amp, 0.0),
        inv_var: vec![4.0 * std::f64::consts::PI.powi(2) * p.width * p.width; p.center.len()],
        b: p.center.iter().map(|&c| C::new(0.0, TWO_PI * c)).collect(),
    }
}

/// A Gaussian times a polynomial of degree ≤ 2, kept as a sparse monomial list.
#[derive(Clone, Debug)]
struct Poly {
    g: Gauss,
    terms: Vec<(Vec<(usize, u32)>, C)>,
}

impl Poly {
    fn new(g: Gauss) -> Self {
        Self {
            g,
            terms: vec![(Vec::new(), C::new(1.0, 0.0))],
        }
    }

    /// Multiply by `c0 + Σ spikes`.
    fn mul_linear(&mut self, c0: C, spikes: &[(usize, C)]) {
        let mut out = Vec::with_capacity(self.terms.len() * (1 + spikes.len()));
        for (mono, coef) in &self.terms {
            if c0 != C::new(0.0, 0.0) {
                out.push((mono.clone(), *coef * c0));
            }
            for (axis, c) in spikes {
                let mut m = mono.clone();
                match m.iter_mut().find(|(a, _)| a == axis) {
                    Some(entry) => entry.1 += 1,
                    None => m.push((*axis, 1)),
                }
                out.push((m, *coef * *c));
            }
        }
        self.terms = out;
    }

    /// Multiply by `scale · ∂_{axis}`. For a [`Gauss`],
    /// `∂_m f = (−M_m x_m + b_m) f`.
    fn apply_derivative(&mut self, axis: usize, scale: C) {
        let m = self.g.inv_var[axis];
        let b = self.g.b[axis];
        // ∂_m (x^k f) = k x^{k−1} f + x^k (−M_m x_m + b_m) f. The first term is
        // what distinguishes a second derivative from squaring: it only appears
        // for the quadratic rows (`wall`), where it must not be dropped.
        let mut from_power = Vec::new();
        for (mono, coef) in &self.terms {
            if let Some(pos) = mono.iter().position(|(a, _)| *a == axis) {
                let power = mono[pos].1;
                let mut reduced = mono.clone();
                if power == 1 {
                    reduced.remove(pos);
                } else {
                    reduced[pos].1 -= 1;
                }
                from_power.push((reduced, *coef * scale * C::new(power as f64, 0.0)));
            }
        }
        self.mul_linear(scale * b, &[(axis, -scale * C::new(m, 0.0))]);
        self.terms.extend(from_power);
    }

    fn multiply_by(&mut self, other: &Gauss) {
        self.g = self.g.product(other);
    }

    fn integrate(&self) -> C {
        let i0 = self.g.i0();
        let mut total = C::new(0.0, 0.0);
        for (mono, coef) in &self.terms {
            let mut value = *coef;
            for &(axis, power) in mono {
                value *= match power {
                    1 => self.g.moment1(axis),
                    2 => self.g.moment2(axis, axis),
                    other => unreachable!("dictionary monomial power {other} exceeds 2"),
                };
            }
            total += value;
        }
        i0 * total
    }
}

// ---------------------------------------------------------------------------
// The two routes
// ---------------------------------------------------------------------------

/// Momentum representation: `q̂ → (i/2π)∂_k`, `p̂ → 2πk`.
///
/// The `q` map uses the fixed Fourier constant [`TWO_PI`]; only the `p` symbol
/// tracks [`Conventions::two_pi`], so mutating it is a real convention error.
fn momentum_term(term: &Term, packets: &DictionaryPackets, conv: &Conventions) -> C {
    let mut poly = Poly::new(momentum_packet_gauss(&packets.ket));
    for (n, power) in &term.q {
        for _ in 0..*power {
            poly.apply_derivative(*n as usize, C::i() / TWO_PI);
        }
    }
    for (m, power) in &term.p {
        for _ in 0..*power {
            poly.mul_linear(C::new(0.0, 0.0), &[(*m as usize, C::new(conv.two_pi, 0.0))]);
        }
    }
    poly.multiply_by(&momentum_packet_gauss(&packets.bra).conj());
    poly.integrate()
}

/// The collective-coordinate integrand of one term: `q̂ → x`, `p̂ → −i∂_x`,
/// times the conjugate bra packet.
fn coordinate_poly(term: &Term, packets: &DictionaryPackets) -> Poly {
    let mut poly = Poly::new(coordinate_packet_gauss(&packets.ket));
    for (n, power) in &term.q {
        for _ in 0..*power {
            poly.mul_linear(C::new(0.0, 0.0), &[(*n as usize, C::new(1.0, 0.0))]);
        }
    }
    for (m, power) in &term.p {
        for _ in 0..*power {
            poly.apply_derivative(*m as usize, -C::i());
        }
    }
    poly.multiply_by(&coordinate_packet_gauss(&packets.bra).conj());
    poly
}

/// Coordinate representation: `q̂ → x`, `p̂ → −i∂_x`.
fn coordinate_term(term: &Term, packets: &DictionaryPackets) -> C {
    coordinate_poly(term, packets).integrate()
}

// ---------------------------------------------------------------------------
// Per-mode 3D field kernel
// ---------------------------------------------------------------------------

/// The spatial axis of a momentum index in the named layout.
///
/// The blocks `x; G,dG; W,dW; B,dB` all start at a base divisible by three, so
/// their spatial index is exactly `m mod 3`; the four Higgs momenta are scalar
/// amplitudes and get the declared profile axis `m mod 3`.
pub fn axis_of(momentum: u32) -> usize {
    momentum as usize % 3
}

fn packet3(p: &Packet) -> Packet {
    Packet {
        center: p.center[..3].to_vec(),
        width: p.width,
    }
}

fn packet_axis(p: &Packet, residue: usize) -> Packet {
    Packet {
        center: vec![p.center[residue]],
        width: p.width,
    }
}

/// `∏_n Φ̂_n^{⋆a_n}` in three dimensions, a real Gaussian in the transferred
/// momentum.
fn field_convolution_3d(term: &Term, packets: &DictionaryPackets, sign: f64) -> Gauss {
    let mut out: Option<Gauss> = None;
    for (n, power) in &term.q {
        let field = momentum_packet_gauss(&packet3(&packets.fields[*n as usize]));
        for _ in 0..*power {
            out = Some(match out {
                None => field.clone(),
                Some(current) => current.convolve(&field),
            });
        }
    }
    let mut g = out.expect("field convolution requires a non-empty q monomial");
    if sign < 0.0 {
        g.b = g.b.iter().map(|b| -*b).collect();
    }
    g
}

/// `∫ φ̂_bra(K+l) φ̂_ket(l) l^p dl` as a Gaussian in `K` times a linear factor
/// (with the `(2π)^p` symbol of `p` already folded in).
fn ket_side_momentum(bra: &Packet, ket: &Packet, p: &[(u32, u32)], conv: &Conventions) -> Poly {
    let mb = 1.0 / (bra.width * bra.width);
    let mk = 1.0 / (ket.width * ket.width);
    let m = mb + mk;
    let s = (bra.width * bra.width + ket.width * ket.width).sqrt();
    // ∫φ̂_b(K+l)φ̂_k(l)dl
    //   = amp_b amp_k (2π)^{3/2} w_b³w_k³/S³ exp(−|K−(c_b−c_k)|²/(2S²)),
    // with the packets' own L² normalisation amp = (πw²)^{−3/4}.
    let norm = |w: f64| -(3.0 / 4.0) * (std::f64::consts::PI * w * w).ln();
    let amp = (norm(bra.width) + norm(ket.width)).exp()
        * (TWO_PI.sqrt() * s).powi(3)
        * (bra.width * ket.width / (s * s)).powi(3);
    let base = Gauss::from_center(
        C::new(amp, 0.0),
        &sub3(bra.center.as_slice(), ket.center.as_slice()),
        s,
    );
    let mut poly = Poly::new(base);
    let mut scale = C::new(1.0, 0.0);
    for (m_index, power) in p {
        let axis = axis_of(*m_index);
        for _ in 0..*power {
            // l_axis = (M_b c_b + M_k c_k)/M − (M_b/M) K_axis.
            let c0 = (mb * bra.center[axis] + mk * ket.center[axis]) / m;
            poly.mul_linear(C::new(c0, 0.0), &[(axis, C::new(-mb / m, 0.0))]);
            scale *= C::new(conv.two_pi, 0.0);
        }
    }
    for (_, coef) in poly.terms.iter_mut() {
        *coef *= scale;
    }
    poly
}

fn sub3(a: &[f64], b: &[f64]) -> Vec<f64> {
    a[..3].iter().zip(&b[..3]).map(|(x, y)| x - y).collect()
}

fn momentum_term_3d(term: &Term, packets: &DictionaryPackets, conv: &Conventions) -> C {
    let bra = packet3(&packets.bra);
    let ket = packet3(&packets.ket);
    if term.q.is_empty() {
        let mut poly = Poly::new(momentum_packet_gauss(&bra).product(&momentum_packet_gauss(&ket)));
        for (m, power) in &term.p {
            for _ in 0..*power {
                poly.mul_linear(C::new(0.0, 0.0), &[(axis_of(*m), C::new(conv.two_pi, 0.0))]);
            }
        }
        return poly.integrate();
    }
    let mut poly = ket_side_momentum(&bra, &ket, &term.p, conv);
    let field = field_convolution_3d(term, packets, conv.transfer_sign);
    poly.multiply_by(&field);
    poly.integrate()
}

/// The 3D field-mode coordinate integrand of one term.
fn coordinate_poly_3d(term: &Term, packets: &DictionaryPackets) -> Poly {
    let ket = packet3(&packets.ket);
    let xket = coordinate_packet_gauss(&ket);
    let mut g = coordinate_packet_gauss(&packet3(&packets.bra))
        .conj()
        .product(&xket);
    for (n, power) in &term.q {
        let field = coordinate_packet_gauss(&packet3(&packets.fields[*n as usize]));
        for _ in 0..*power {
            g = g.product(&field);
        }
    }
    let mut poly = Poly::new(g);
    for (m, power) in &term.p {
        let axis = axis_of(*m);
        for _ in 0..*power {
            // −i∂_axis f = i M x_axis − i b_axis for the Gauss form.
            let inv_var = xket.inv_var[axis];
            let b = xket.b[axis];
            poly.mul_linear(-C::i() * b, &[(axis, C::i() * C::new(inv_var, 0.0))]);
        }
    }
    poly
}

fn coordinate_term_3d(term: &Term, packets: &DictionaryPackets) -> C {
    coordinate_poly_3d(term, packets).integrate()
}

// ---------------------------------------------------------------------------
// Independent quadrature evaluation
// ---------------------------------------------------------------------------

/// Trapezoid quadrature of `∫ x^k exp(−½ M x² + b x) dx` on one axis.
///
/// The window follows the Gaussian envelope (`±12σ` around the shifted centre)
/// and the rule converges super-exponentially there for a Gaussian times a
/// low-degree polynomial, so this shares no step with the closed-form moments in
/// [`Poly::integrate`]: it is an independent evaluation of the same integrand.
fn axis_quadrature(g: &Gauss, max_power: u32, nodes: usize) -> Vec<C> {
    debug_assert_eq!(g.inv_var.len(), 1, "one axis at a time");
    let inv_var = g.inv_var[0];
    let sigma = (1.0 / inv_var).sqrt();
    let center = g.b[0].re / inv_var;
    let half = 12.0 * sigma + center.abs();
    let step = 2.0 * half / nodes as f64;
    let mut sums = vec![C::new(0.0, 0.0); max_power as usize + 1];
    for i in 0..=nodes {
        let x = center - half + step * i as f64;
        let weight = if i == 0 || i == nodes { 0.5 } else { 1.0 };
        // The wavefunction itself, sampled — no moment formula is involved.
        let envelope = g.value(&[x]) * C::new(weight * step, 0.0);
        let mut power = C::new(1.0, 0.0);
        for sum in sums.iter_mut() {
            *sum += envelope * power;
            power *= C::new(x, 0.0);
        }
    }
    sums
}

/// One-axis view of a separable [`Gauss`], with the amplitude left out so the
/// per-axis integrals stay multiplicative.
fn axis_view(g: &Gauss, axis: usize) -> Gauss {
    Gauss {
        log_amp: C::new(0.0, 0.0),
        inv_var: vec![g.inv_var[axis]],
        b: vec![g.b[axis]],
    }
}

/// Quadrature evaluation of a [`Poly`], axis by axis.
///
/// A Gaussian times a polynomial is separable, so the multi-dimensional
/// integral is the product of the per-axis integrals `∫ x^k exp(−½M_a x² + b_a x)
/// dx` — no tensor grid is needed, which is what makes the 163-dimensional
/// collective coordinate integral cheap enough to evaluate by quadrature.
fn quadrature_value(poly: &Poly, nodes: usize) -> C {
    let dims = poly.g.inv_var.len();
    let max_power = poly
        .terms
        .iter()
        .flat_map(|(mono, _)| mono.iter().map(|(_, power)| *power))
        .max()
        .unwrap_or(0);
    let per_axis: Vec<Vec<C>> = (0..dims)
        .map(|axis| axis_quadrature(&axis_view(&poly.g, axis), max_power, nodes))
        .collect();
    let mut total = C::new(0.0, 0.0);
    for (mono, coef) in &poly.terms {
        let mut value = *coef;
        let mut powers = vec![0u32; dims];
        for &(axis, power) in mono {
            powers[axis] = power;
        }
        for (axis, power) in powers.iter().enumerate() {
            value *= per_axis[axis][*power as usize];
        }
        total += value;
    }
    poly.g.log_amp.exp() * total
}

/// Quadrature evaluation of a term's collective-coordinate value.
pub fn coordinate_term_quadrature(term: &Term, packets: &DictionaryPackets, nodes: usize) -> C {
    quadrature_value(&coordinate_poly(term, packets), nodes)
}

/// Quadrature evaluation of a term's 3D field-mode coordinate value.
pub fn coordinate_term_3d_quadrature(term: &Term, packets: &DictionaryPackets, nodes: usize) -> C {
    quadrature_value(&coordinate_poly_3d(term, packets), nodes)
}

/// One-dimensional packet at `axis`, as the route reads it.
fn axis_packet(p: &Packet, axis: usize) -> Packet {
    Packet {
        center: vec![p.center[axis]],
        width: p.width,
    }
}

/// Per-axis integrand of a `p`-free term, built from the term and the packet
/// layout rather than from [`Poly`].
///
/// `three_d` selects the route's own insertion: a `q_n` becomes `x_n` (the
/// coordinate multiplication of the collective kernel, power on axis `n`) or the
/// profile of field `n` (the per-mode insertion, a factor on axis `n mod 3`).
/// Nothing here reads the closed-form polynomial, so a wrong mode, axis or
/// grouping shows up as a disagreement with it.
fn q_only_axis_factors(
    term: &Term,
    packets: &DictionaryPackets,
    three_d: bool,
) -> Vec<(Gauss, u32)> {
    let axes = if three_d { 3 } else { Q_COORDINATES };
    let mut factors = Vec::with_capacity(axes);
    for axis in 0..axes {
        let mut g = coordinate_packet_gauss(&axis_packet(&packets.bra, axis))
            .conj()
            .product(&coordinate_packet_gauss(&axis_packet(&packets.ket, axis)));
        let mut power = 0u32;
        for (n, insertions) in &term.q {
            for _ in 0..*insertions {
                if three_d {
                    // A 3D field insertion is the mode's profile along every
                    // axis; `axis_of` routes `p`, not `q`.
                    g = g.product(&coordinate_packet_gauss(&axis_packet(
                        &packets.fields[*n as usize],
                        axis,
                    )));
                } else if *n as usize == axis {
                    power += 1;
                }
            }
        }
        factors.push((g, power));
    }
    factors
}

fn q_only_quadrature(term: &Term, packets: &DictionaryPackets, three_d: bool, nodes: usize) -> C {
    let mut total = C::new(1.0, 0.0);
    for (g, power) in q_only_axis_factors(term, packets, three_d) {
        let parts = axis_quadrature(&axis_view(&g, 0), power, nodes);
        total *= g.log_amp.exp() * parts[power as usize];
    }
    total
}

/// Quadrature evaluation of a `p`-free term's collective-coordinate value,
/// assembled from the packet layout instead of from the closed-form integrand.
pub fn coordinate_term_q_only_quadrature(
    term: &Term,
    packets: &DictionaryPackets,
    nodes: usize,
) -> C {
    q_only_quadrature(term, packets, false, nodes)
}

/// Quadrature evaluation of a `p`-free term's 3D field-mode value, assembled
/// from the packet layout instead of from the closed-form integrand.
pub fn coordinate_term_3d_q_only_quadrature(
    term: &Term,
    packets: &DictionaryPackets,
    nodes: usize,
) -> C {
    q_only_quadrature(term, packets, true, nodes)
}

/// Measured agreement between the closed-form `q`-bearing values and an
/// independent quadrature evaluation of the same integrand.
#[derive(Clone, Debug)]
pub struct QBearingQuadrature {
    pub terms: usize,
    /// Terms whose integrand was rebuilt from the packet layout, so the check is
    /// sensitive to the insertion routing and not only to the integration.
    pub routing_sensitive_terms: usize,
    pub worst_collective_gap: f64,
    pub worst_collective_row: String,
    pub worst_three_d_gap: f64,
    pub worst_three_d_row: String,
}

/// Cross-check **every** `q`-bearing term of the dictionary against the
/// quadrature evaluation, in both routes.
pub fn q_bearing_quadrature(packets: &DictionaryPackets, nodes: usize) -> QBearingQuadrature {
    let mut report = QBearingQuadrature {
        terms: 0,
        routing_sensitive_terms: 0,
        worst_collective_gap: 0.0,
        worst_collective_row: String::new(),
        worst_three_d_gap: 0.0,
        worst_three_d_row: String::new(),
    };
    for entry in dictionary() {
        for term in &entry.terms {
            if term.q_degree() == 0 {
                continue;
            }
            let p_free = term.p_degree() == 0;
            let collective_quad = if p_free {
                coordinate_term_q_only_quadrature(term, packets, nodes)
            } else {
                coordinate_term_quadrature(term, packets, nodes)
            };
            let three_d_quad = if p_free {
                coordinate_term_3d_q_only_quadrature(term, packets, nodes)
            } else {
                coordinate_term_3d_quadrature(term, packets, nodes)
            };
            let collective = coordinate_term(term, packets);
            let three_d = coordinate_term_3d(term, packets);
            let collective_gap =
                (collective - collective_quad).norm() / (1.0 + collective_quad.norm());
            let three_d_gap = (three_d - three_d_quad).norm() / (1.0 + three_d_quad.norm());
            if collective_gap > report.worst_collective_gap {
                report.worst_collective_gap = collective_gap;
                report.worst_collective_row = entry.label.clone();
            }
            if three_d_gap > report.worst_three_d_gap {
                report.worst_three_d_gap = three_d_gap;
                report.worst_three_d_row = entry.label.clone();
            }
            report.terms += 1;
            report.routing_sensitive_terms += usize::from(p_free);
        }
    }
    report
}

/// The 3D field-mode kernel of one dictionary entry (momentum representation).
pub fn three_d_kernel(
    entry: &DictionaryEntry,
    packets: &DictionaryPackets,
    params: &SmParameters,
    conv: &Conventions,
) -> C {
    entry
        .terms
        .iter()
        .map(|term| coefficient(term, params) * momentum_term_3d(term, packets, conv))
        .sum()
}

/// The 3D field-mode kernel of one dictionary entry (coordinate representation).
pub fn three_d_coordinate_kernel(
    entry: &DictionaryEntry,
    packets: &DictionaryPackets,
    params: &SmParameters,
) -> C {
    entry
        .terms
        .iter()
        .map(|term| coefficient(term, params) * coordinate_term_3d(term, packets))
        .sum()
}

/// The one-axis overlap of a term's fields on spatial residue `residue`, with no
/// derivative factor. This is the spectator factor in
/// `collective = three_d * O_0^54 O_1^53 O_2^53`.
pub fn axis_overlap(term: &Term, packets: &DictionaryPackets, residue: usize) -> C {
    let mut g = momentum_packet_gauss(&packet_axis(&packets.bra, residue))
        .product(&momentum_packet_gauss(&packet_axis(&packets.ket, residue)));
    for (n, power) in &term.q {
        let field = momentum_packet_gauss(&packet_axis(&packets.fields[*n as usize], residue));
        for _ in 0..*power {
            g = g.product(&field);
        }
    }
    Poly::new(g).integrate()
}

/// Per-term values of all four kernels, for the cross-check tests.
#[derive(Clone, Debug)]
pub struct TermKernels {
    pub label: String,
    pub q_degree: u32,
    pub coefficient: C,
    pub collective_momentum: C,
    pub collective_coordinate: C,
    pub three_d_momentum: C,
    pub three_d_coordinate: C,
    pub axis_overlap: [C; 3],
}

/// Per-term breakdown of the whole dictionary.
pub fn term_kernels(
    packets: &DictionaryPackets,
    params: &SmParameters,
    conv: &Conventions,
) -> Vec<TermKernels> {
    let mut out = Vec::new();
    for entry in dictionary() {
        for term in &entry.terms {
            out.push(TermKernels {
                label: entry.label.clone(),
                q_degree: term.q_degree(),
                coefficient: coefficient(term, params),
                collective_momentum: momentum_term(term, packets, conv),
                collective_coordinate: coordinate_term(term, packets),
                three_d_momentum: momentum_term_3d(term, packets, conv),
                three_d_coordinate: coordinate_term_3d(term, packets),
                axis_overlap: [
                    axis_overlap(term, packets, 0),
                    axis_overlap(term, packets, 1),
                    axis_overlap(term, packets, 2),
                ],
            });
        }
    }
    out
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

/// Where a just-parsed base sits in the term, so a following `^{-1}` can find it.
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
            Some(c) => Err(format!(
                "expected '{expected}', found '{c}' at {}",
                self.pos
            )),
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
        if term.factors.is_empty() && term.q.is_empty() && term.p.is_empty() {
            return Err("empty term".into());
        }
        if sign < 0.0 {
            // parse_term never consumes a leading sign, so this -1 is the only
            // sign on the term.
            term.factors.insert(0, Factor::Rational(-1, 1));
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
                ));
            }
        }
    }
    Ok(terms)
}

impl Cursor<'_> {
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
                    (Factor::InvSqrt2, -1) => *last = Factor::Sqrt2,
                    (Factor::InvSqrt3, -1) => *last = Factor::Sqrt3,
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
                let numerator: i64 = self
                    .read_group()?
                    .trim()
                    .parse()
                    .map_err(|_| format!("non-integer fraction numerator at {}", self.pos))?;
                let denominator: i64 = self
                    .read_group()?
                    .trim()
                    .parse()
                    .map_err(|_| format!("non-integer fraction denominator at {}", self.pos))?;
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
                let subscript = self.read_group()?.trim().to_string();
                // The Cadabra export writes `yu_{00}` (two bare digits) while
                // hand-written forms use `yu_{0,0}`; accept both.
                let digits: Vec<usize> = if subscript.contains(',') {
                    subscript
                        .split(',')
                        .map(|part| part.trim().parse::<usize>())
                        .collect::<Result<_, _>>()
                        .map_err(|_| format!("bad Yukawa indices '{{{subscript}}}'"))?
                } else {
                    subscript
                        .chars()
                        .map(|c| {
                            c.to_digit(10)
                                .map(|d| d as usize)
                                .ok_or_else(|| format!("bad Yukawa index '{c}'"))
                        })
                        .collect::<Result<_, _>>()?
                };
                if digits.len() != 2 || digits.iter().any(|&x| x >= 3) {
                    return Err(format!(
                        "Yukawa indices must be two values in 0..3: '{subscript}'"
                    ));
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
        fixture_parameters()
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
    fn the_dictionary_never_mixes_q_and_p() {
        for e in dictionary() {
            for term in &e.terms {
                assert!(
                    term.q.is_empty() || term.p.is_empty(),
                    "{} mixes q and p in one term",
                    e.label
                );
                assert!(term.q_degree() <= 2, "{} has q-degree > 2", e.label);
                assert!(term.p_degree() <= 1, "{} has p-degree > 1", e.label);
            }
        }
    }

    #[test]
    fn kinetic_row_is_the_bare_jet_momentum() {
        let e = entry("kin3");
        assert_eq!(e.terms.len(), 1);
        assert_eq!(e.terms[0].p, vec![(3, 1)]);
        assert!(e.terms[0].q.is_empty());
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
    fn momentum_space_q_is_a_derivative_not_a_shift() {
        // kin3 = p_3 has the real symbol 2πk_3; the value must equal 2π times
        // the k_3 moment of the bra/ket overlap.
        let packets = DictionaryPackets::declared();
        let sm = params();
        let conv = Conventions::canonical();
        let e = entry("kin3");
        let value = momentum_kernel(&e, &packets, &sm, &conv);
        let overlap =
            momentum_packet_gauss(&packets.bra).product(&momentum_packet_gauss(&packets.ket));
        let expected = C::new(TWO_PI, 0.0) * overlap.moment1(3) * overlap.i0();
        assert!(
            (value - expected).norm() < 1e-12 * (1.0 + expected.norm()),
            "{value} vs {expected}"
        );
    }

    #[test]
    fn momentum_and_coordinate_routes_agree_on_each_family() {
        let packets = DictionaryPackets::declared();
        let sm = params();
        let conv = Conventions::canonical();
        for label in [
            "kin3", "kin150", "magG00", "magW22", "magB02", "covH10", "wall",
        ] {
            let e = entry(label);
            let m = momentum_kernel(&e, &packets, &sm, &conv);
            let c = coordinate_kernel(&e, &packets, &sm);
            assert!(
                (m - c).norm() < 1e-9 * (1.0 + c.norm()),
                "{label}: momentum {m} vs coordinate {c}"
            );
        }
    }

    #[test]
    fn insertion_and_derivative_rows_agree() {
        let packets = DictionaryPackets::declared();
        let sm = params();
        let conv = Conventions::canonical();
        for label in ["matter0x0", "matter0x2", "matter0x34"] {
            let e = entry(label);
            let m = momentum_kernel(&e, &packets, &sm, &conv);
            let c = coordinate_kernel(&e, &packets, &sm);
            assert!(
                (m - c).norm() < 1e-9 * (1.0 + c.norm()),
                "{label}: {m} vs {c}"
            );
        }
    }

    #[test]
    fn the_whole_dictionary_cross_checks() {
        let packets = DictionaryPackets::declared();
        let sm = params();
        let report = worst_gap(&packets, &sm);
        assert_eq!(report.rows, 1241);
        assert!(
            report.relative_gap < 1e-9,
            "worst row {} gap {:e} (momentum {}, coordinate {})",
            report.label,
            report.relative_gap,
            report.momentum,
            report.coordinate
        );
    }

    #[test]
    fn the_3d_mode_kernel_is_plancherel_consistent_on_every_row() {
        let packets = DictionaryPackets::declared();
        let sm = params();
        let conv = Conventions::canonical();
        let mut worst = (0.0f64, String::new());
        for entry in dictionary() {
            let m = three_d_kernel(entry, &packets, &sm, &conv);
            let c = three_d_coordinate_kernel(entry, &packets, &sm);
            let gap = (m - c).norm() / (1.0 + c.norm());
            assert!(
                gap < 1e-9,
                "{}: 3D momentum {m} vs coordinate {c}",
                entry.label
            );
            if gap > worst.0 {
                worst = (gap, entry.label.clone());
            }
        }
        assert!(
            worst.0 < 1e-11,
            "worst 3D row {} gap {:e}",
            worst.1,
            worst.0
        );
    }

    #[test]
    fn the_derivative_terms_are_the_collective_symbol_on_their_axis() {
        // A `q`-free term is the same object in both kernels: `p_m` acting on the
        // ket. The collective value is therefore the 3D value times the 160
        // spectator overlaps (54/53/53 by residue), and the derivative factor
        // cancels because it lives on the same residue in both routes.
        //
        // `q`-bearing terms are NOT compared this way: `q` is a coordinate
        // multiplication in the collective kernel and a field insertion in the
        // 3D kernel, i.e. a different operator. Their 3D routing is checked by
        // `the_3d_mode_kernel_is_plancherel_consistent_on_every_row` instead.
        let packets = DictionaryPackets::declared();
        let sm = params();
        let conv = Conventions::canonical();
        let mut checked = 0usize;
        for term in term_kernels(&packets, &sm, &conv) {
            if term.q_degree != 0 {
                continue;
            }
            let expected = term.three_d_momentum
                * term.axis_overlap[0].powi(54)
                * term.axis_overlap[1].powi(53)
                * term.axis_overlap[2].powi(53);
            let scale = 1.0 + term.collective_momentum.norm();
            assert!(
                (term.collective_momentum - expected).norm() < 1e-9 * scale,
                "{}: collective {} vs 3D*spectators {}",
                term.label,
                term.collective_momentum,
                expected
            );
            checked += 1;
        }
        // 40 kinetic rows plus every derivative term of the 232 p-bearing rows.
        assert!(
            checked >= 232,
            "only {checked} derivative terms matched the collective symbol"
        );
    }

    #[test]
    fn the_q_bearing_rows_agree_with_independent_quadrature() {
        // `q` is not the same operator in the two routes — a coordinate
        // multiplication in the collective kernel, a per-mode field insertion in
        // the 3D kernel — so the spectator identity above cannot compare them
        // here. What can be checked is that each route's `q` insertion is the
        // one the dictionary asks for: both closed forms are compared with a
        // trapezoid evaluation of their own integrand, which contains no share
        // of the moment formulas and would move if the insertion picked the
        // wrong mode, the wrong axis or the wrong Fourier weight.
        let report = q_bearing_quadrature(&DictionaryPackets::declared(), 400);
        assert!(
            report.terms >= 20,
            "only {} q-bearing terms checked",
            report.terms
        );
        assert!(
            report.routing_sensitive_terms >= 8,
            "only {} q-bearing terms were routing-sensitive",
            report.routing_sensitive_terms
        );
        assert!(
            report.worst_collective_gap < 1e-11,
            "worst collective row {} gap {:e}",
            report.worst_collective_row,
            report.worst_collective_gap
        );
        assert!(
            report.worst_three_d_gap < 1e-11,
            "worst 3D row {} gap {:e}",
            report.worst_three_d_row,
            report.worst_three_d_gap
        );
        // No dictionary term mixes `q` and `p`, so every q-bearing term is
        // compared with an integrand rebuilt from the term and the layout.
        assert_eq!(
            report.routing_sensitive_terms, report.terms,
            "every q-bearing term should be routing-sensitive"
        );
    }

    #[test]
    fn a_mutated_field_axis_map_is_caught_by_the_q_bearing_check() {
        // The check reads each field's profile from the packet layout along the
        // spatial axes. Permuting which packet coordinate feeds which axis must
        // therefore show up as a large gap against the true layout: if the
        // check were insensitive to the axis assignment, a wrong field routing
        // could pass unnoticed.
        let packets = DictionaryPackets::declared();
        let mut per_family: BTreeMap<Family, usize> = BTreeMap::new();
        let mut checked = 0usize;
        let mut caught = 0usize;
        let mut weakest = (f64::INFINITY, String::new());
        for permutation in [[1usize, 0, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1]] {
            let mut mutated = packets.clone();
            for field in &mut mutated.fields {
                // 163 = 54*3 + 1, so the last block is ragged: an axis whose
                // permuted partner does not exist keeps its own coordinate.
                let permuted: Vec<f64> = (0..field.center.len())
                    .map(|axis| {
                        let base = axis - axis % 3;
                        let shifted = base + permutation[axis % 3];
                        if shifted < field.center.len() {
                            field.center[shifted]
                        } else {
                            field.center[axis]
                        }
                    })
                    .collect();
                field.center = permuted;
            }
            per_family.clear();
            for entry in dictionary() {
                let seen = per_family.entry(entry.family).or_insert(0);
                if *seen >= 4 {
                    continue;
                }
                let mut used = false;
                for term in &entry.terms {
                    if term.q_degree() == 0 {
                        continue;
                    }
                    used = true;
                    let truth = coordinate_term_3d(term, &packets);
                    let mutated_value = coordinate_term_3d(term, &mutated);
                    let gap = (mutated_value - truth).norm() / (1.0 + truth.norm());
                    checked += 1;
                    if gap > 1e-6 {
                        caught += 1;
                    } else if gap < weakest.0 {
                        weakest = (gap, format!("{:?} {}", permutation, entry.label));
                    }
                }
                if used {
                    *seen += 1;
                }
            }
        }
        assert!(checked >= 80, "only {checked} mutated terms checked");
        assert_eq!(
            caught, checked,
            "the q-bearing check missed a permuted axis map (weakest {} gap {:e})",
            weakest.1, weakest.0
        );
    }

    #[test]
    fn a_mutated_collective_mode_index_is_caught_by_the_q_bearing_check() {
        // The axis-map mutation above exercises the 3D route, where a `q_n` is
        // a profile on every spatial axis. The collective route reads the same
        // `q_n` as a multiplication by coordinate `n`; if the check were
        // insensitive to *which* collective mode the dictionary inserts, the
        // quadrature could be fed a shifted mode and still agree with the
        // closed form. Shift each single-mode insertion by one coordinate and
        // require the disagreement to be large.
        let packets = DictionaryPackets::declared();
        let mut per_family: BTreeMap<Family, usize> = BTreeMap::new();
        let mut checked = 0usize;
        let mut caught = 0usize;
        let mut weakest = (f64::INFINITY, String::new());
        for entry in dictionary() {
            let seen = per_family.entry(entry.family).or_insert(0);
            if *seen >= 4 {
                continue;
            }
            let mut used = false;
            for term in &entry.terms {
                // One insertion keeps the mutation a pure mode shift: the
                // closed form has `x_n`, the mutated quadrature has `x_{n+1}`.
                if term.q_degree() != 1 || term.p_degree() != 0 {
                    continue;
                }
                used = true;
                let truth = coordinate_term(term, &packets);
                for (mode, power) in &term.q {
                    let mut mutated = term.clone();
                    mutated.q = vec![((mode + 1) % Q_COORDINATES as u32, *power)];
                    let mutated_value = coordinate_term_q_only_quadrature(&mutated, &packets, 400);
                    let gap = (mutated_value - truth).norm() / (1.0 + truth.norm());
                    checked += 1;
                    if gap > 1e-6 {
                        caught += 1;
                    } else if gap < weakest.0 {
                        weakest = (gap, entry.label.clone());
                    }
                }
            }
            if used {
                *seen += 1;
            }
        }
        assert!(
            checked >= 40,
            "only {checked} mutated collective mode indices checked"
        );
        assert_eq!(
            caught, checked,
            "the q-bearing check missed a shifted collective mode (weakest {} gap {:e})",
            weakest.1, weakest.0
        );
    }

    #[test]
    fn the_3d_axis_map_matches_the_named_layout() {
        use crate::sm_full::coord;
        for a in 0..8u32 {
            for i in 0..3u32 {
                assert_eq!(axis_of(coord::g(a, i)), i as usize);
                for j in 0..3u32 {
                    assert_eq!(axis_of(coord::dg(a, i, j)), j as usize);
                }
            }
        }
        for a in 0..3u32 {
            for i in 0..3u32 {
                assert_eq!(axis_of(coord::w(a, i)), i as usize);
                for j in 0..3u32 {
                    assert_eq!(axis_of(coord::dw(a, i, j)), j as usize);
                }
            }
            assert_eq!(axis_of(coord::b(a)), a as usize);
            for j in 0..3u32 {
                assert_eq!(axis_of(coord::db(a, j)), j as usize);
            }
        }
        for i in 0..3u32 {
            assert_eq!(axis_of(coord::x(i)), i as usize);
        }
        // The four Higgs momenta are scalar amplitudes: no spatial axis exists,
        // so the kernel uses the declared profile axis `m mod 3`. Assert the
        // declaration rather than pretending it is a layout fact.
        for a in 0..4u32 {
            assert_eq!(axis_of(coord::phi(a)), a as usize % 3);
        }
    }

    #[test]
    fn convolution_matches_quadrature_in_one_dimension() {
        let a = momentum_packet_gauss(&Packet {
            center: vec![0.3],
            width: 0.7,
        });
        let b = momentum_packet_gauss(&Packet {
            center: vec![-0.2],
            width: 1.1,
        });
        let c = a.convolve(&b);
        let n = 400_000usize;
        let (lo, hi) = (-8.0f64, 8.0f64);
        let h = (hi - lo) / n as f64;
        for k in [-0.8f64, 0.0, 0.35] {
            let mut sum = 0.0;
            for i in 0..n {
                let q = lo + (i as f64 + 0.5) * h;
                sum += (a.value(&[q]) * b.value(&[k - q])).re;
            }
            let numeric = sum * h;
            let closed = c.value(&[k]).re;
            assert!(
                (closed - numeric).abs() < 1e-6 * (1.0 + numeric.abs()),
                "k={k}: closed {closed} vs quadrature {numeric}"
            );
        }
    }

    #[test]
    fn a_wrong_fourier_symbol_breaks_the_cross_check() {
        let packets = DictionaryPackets::declared();
        let sm = params();
        let wrong = Conventions {
            two_pi: 1.0,
            ..Conventions::canonical()
        };
        let e = entry("kin7");
        let canonical = (momentum_kernel(&e, &packets, &sm, &Conventions::canonical())
            - coordinate_kernel(&e, &packets, &sm))
        .norm();
        let mutated = (momentum_kernel(&e, &packets, &sm, &wrong)
            - coordinate_kernel(&e, &packets, &sm))
        .norm();
        assert!(canonical < 1e-12);
        assert!(mutated > 1e-3, "mutation must be detectable, gap {mutated}");
    }
}
