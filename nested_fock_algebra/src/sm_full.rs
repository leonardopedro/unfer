//! Complete finite-internal, independent-collective-coordinate SM operator.
//!
//! Reference: book.tex, "Majorana spinors in the Standard Model", and
//! BookProof/Sm/ChapterSmHamiltonian. All 163 bosonic coordinates, 40 field
//! momenta, spatial Dirac derivatives and 96 independent complex CAR
//! coordinates (192 real Majorana coordinates) are retained. This is E1, not
//! an ultraviolet construction of an infinite inner matter-mode theory.
//! The final Hamiltonian is the single outer enclosure of this full operator.

use crate::exact_inner::{InnerExpr as E, InnerLadder as L};
use num_complex::Complex64 as C;
use std::collections::BTreeMap;

pub const BOSON_MODES: u32 = 163;
pub const MATTER_MODES: u32 = 96;
pub type YukawaMatrix = [[C; 3]; 3];

#[derive(Clone, Debug)]
pub struct SmParameters {
    pub gs: f64,
    pub gw: f64,
    pub gy: f64,
    pub lambda: f64,
    pub vev: f64,
    pub yu: YukawaMatrix,
    pub yd: YukawaMatrix,
    pub ynu: YukawaMatrix,
    pub ye: YukawaMatrix,
}

impl SmParameters {
    pub fn validate(&self) -> Result<(), String> {
        if [self.gs, self.gw, self.gy, self.lambda, self.vev]
            .iter()
            .any(|x| !x.is_finite())
            || self.lambda < 0.0
        {
            return Err("SM parameters must be finite and lambda nonnegative".into());
        }
        for matrix in [&self.yu, &self.yd, &self.ynu, &self.ye] {
            if matrix
                .iter()
                .flatten()
                .any(|x| !x.re.is_finite() || !x.im.is_finite())
            {
                return Err("nonfinite Yukawa entry".into());
            }
        }
        Ok(())
    }

    pub fn symbolic_bindings(&self) -> BTreeMap<String, C> {
        let mut out = BTreeMap::new();
        for (name, x) in [
            ("gs", self.gs),
            ("gw", self.gw),
            ("gy", self.gy),
            ("lam", self.lambda),
            ("vev", self.vev),
            ("rtthree", 3f64.sqrt()),
            ("rttwo", 2f64.sqrt()),
        ] {
            out.insert(name.into(), C::new(x, 0.0));
        }
        for (name, matrix) in [
            ("yu", &self.yu),
            ("yd", &self.yd),
            ("yn", &self.ynu),
            ("ye", &self.ye),
        ] {
            for (i, row) in matrix.iter().enumerate() {
                for (j, &x) in row.iter().enumerate() {
                    out.insert(format!("{name}{i}{j}"), x);
                    out.insert(format!("c{name}{i}{j}"), x.conj());
                }
            }
        }
        out
    }
}

/// Explicit named-coordinate bijection. Lean's noncomputable `smIdx` must be
/// compared through these names, not presumed to use these numeric labels.
pub mod coord {
    pub fn x(i: u32) -> u32 {
        assert!(i < 3);
        i
    }
    pub fn g(a: u32, i: u32) -> u32 {
        assert!(a < 8 && i < 3);
        3 + 3 * a + i
    }
    pub fn dg(a: u32, i: u32, j: u32) -> u32 {
        assert!(a < 8 && i < 3 && j < 3);
        27 + 9 * a + 3 * i + j
    }
    pub fn w(a: u32, i: u32) -> u32 {
        assert!(a < 3 && i < 3);
        99 + 3 * a + i
    }
    pub fn dw(a: u32, i: u32, j: u32) -> u32 {
        assert!(a < 3 && i < 3 && j < 3);
        108 + 9 * a + 3 * i + j
    }
    pub fn b(i: u32) -> u32 {
        assert!(i < 3);
        135 + i
    }
    pub fn db(i: u32, j: u32) -> u32 {
        assert!(i < 3 && j < 3);
        138 + 3 * i + j
    }
    pub fn phi(a: u32) -> u32 {
        assert!(a < 4);
        147 + a
    }
    pub fn dphi(a: u32, i: u32) -> u32 {
        assert!(a < 4 && i < 3);
        151 + 3 * a + i
    }
    pub fn active() -> Vec<u32> {
        (0..8)
            .flat_map(|a| (0..3).map(move |i| g(a, i)))
            .chain((0..3).flat_map(|a| (0..3).map(move |i| w(a, i))))
            .chain((0..3).map(b))
            .chain((0..4).map(phi))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Species {
    Up,
    Down,
    Neutrino,
    Electron,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Chirality {
    Left,
    Right,
}

impl Species {
    pub fn colors(self) -> u32 {
        if matches!(self, Self::Up | Self::Down) {
            3
        } else {
            1
        }
    }
    pub fn hypercharge(self, chirality: Chirality) -> f64 {
        match (self, chirality) {
            (Self::Up | Self::Down, Chirality::Left) => 1.0 / 6.0,
            (Self::Neutrino | Self::Electron, Chirality::Left) => -0.5,
            (Self::Up, Chirality::Right) => 2.0 / 3.0,
            (Self::Down, Chirality::Right) => -1.0 / 3.0,
            (Self::Electron, Chirality::Right) => -1.0,
            (Self::Neutrino, Chirality::Right) => 0.0,
        }
    }
}

/// Independent complex chiral spinor coordinates; no additional, duplicate
/// antiparticle copy is appended. The book's conjugate singlets are transported
/// to physical right-handed coordinates (see the representation dictionary).
pub fn matter_mode(s: Species, generation: u32, color: u32, ch: Chirality, spin: u32) -> u32 {
    assert!(generation < 3 && color < s.colors() && spin < 2);
    let offset = match s {
        Species::Up => 0,
        Species::Down => 12,
        Species::Neutrino => 24,
        Species::Electron => 28,
    };
    32 * generation + offset + 4 * color + if ch == Chirality::Right { 2 } else { 0 } + spin
}

pub fn pauli(a: u32) -> [[C; 2]; 2] {
    let z = C::new(0.0, 0.0);
    let o = C::new(1.0, 0.0);
    let i = C::new(0.0, 1.0);
    match a {
        0 => [[z, o], [o, z]],
        1 => [[z, -i], [i, z]],
        2 => [[o, z], [z, -o]],
        _ => panic!("Pauli index"),
    }
}

/// Right-handed kinetic matrices in the manuscript's actual Majorana spin
/// frame. Left-handed matrices are their negatives. They follow from
/// KinZ_j=-(i gamma^j)(i gamma^0), with J=i gamma5 and the gamma0 singlet map;
/// see the independent fixed-integer-matrix bridge regression.
pub fn book_kinetic_spin(j: u32) -> [[C; 2]; 2] {
    match j {
        0 => pauli(0),
        1 => pauli(2).map(|row| row.map(|x| -x)),
        2 => pauli(1).map(|row| row.map(|x| -x)),
        _ => panic!("spatial spin index"),
    }
}

pub fn su3_generators() -> [[[C; 3]; 3]; 8] {
    let mut t = [[[C::new(0.0, 0.0); 3]; 3]; 8];
    for (a, r, s) in [(0, 0, 1), (3, 0, 2), (5, 1, 2)] {
        t[a][r][s] = C::new(0.5, 0.0);
        t[a][s][r] = C::new(0.5, 0.0);
    }
    for (a, r, s) in [(1, 0, 1), (4, 0, 2), (6, 1, 2)] {
        t[a][r][s] = C::new(0.0, -0.5);
        t[a][s][r] = C::new(0.0, 0.5);
    }
    t[2][0][0] = C::new(0.5, 0.0);
    t[2][1][1] = C::new(-0.5, 0.0);
    let d = 0.5 / 3f64.sqrt();
    t[7][0][0] = C::new(d, 0.0);
    t[7][1][1] = C::new(d, 0.0);
    t[7][2][2] = C::new(-2.0 * d, 0.0);
    t
}

/// Computed from [T_a,T_b]=i f_abc T_c, not a separate hard-coded f table.
pub fn su3_structure() -> [[[f64; 8]; 8]; 8] {
    let t = su3_generators();
    let mut f = [[[0.0; 8]; 8]; 8];
    for a in 0..8 {
        for b in 0..8 {
            for k in 0..8 {
                let mut trace = C::new(0.0, 0.0);
                for r in 0..3 {
                    for s in 0..3 {
                        for u in 0..3 {
                            trace +=
                                (t[a][r][u] * t[b][u][s] - t[b][r][u] * t[a][u][s]) * t[k][s][r];
                        }
                    }
                }
                f[a][b][k] = (C::new(0.0, -2.0) * trace).re;
            }
        }
    }
    f
}

pub fn epsilon(a: u32, b: u32, c: u32) -> f64 {
    match (a, b, c) {
        (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
        (0, 2, 1) | (2, 1, 0) | (1, 0, 2) => -1.0,
        _ => 0.0,
    }
}

/// Realification of i*T, on (Re H_0,Im H_0,Re H_1,Im H_1).
pub fn higgs_real_generator(weak: Option<u32>) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    let matrix = if let Some(a) = weak {
        pauli(a).map(|r| r.map(|x| C::i() * x * 0.5))
    } else {
        [
            [C::new(0.0, 0.5), C::new(0.0, 0.0)],
            [C::new(0.0, 0.0), C::new(0.0, 0.5)],
        ]
    };
    for r in 0..2 {
        for s in 0..2 {
            let x = matrix[r][s];
            out[2 * r][2 * s] = x.re;
            out[2 * r][2 * s + 1] = -x.im;
            out[2 * r + 1][2 * s] = x.im;
            out[2 * r + 1][2 * s + 1] = x.re;
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Variable {
    Q(u32),
    P(u32),
}

/// Commuting polynomial used only where coordinates/momenta commute. Mixed
/// q_m*p_m monomials are rejected by the LaTeX import path, never reordered.
#[derive(Clone, Debug, Default)]
pub struct SmPolynomial {
    pub terms: BTreeMap<Vec<Variable>, C>,
}

impl SmPolynomial {
    pub fn add(&mut self, mut word: Vec<Variable>, coefficient: C) {
        if coefficient == C::new(0.0, 0.0) {
            return;
        }
        word.sort();
        let value = self.terms.entry(word.clone()).or_default();
        *value += coefficient;
        if *value == C::new(0.0, 0.0) {
            self.terms.remove(&word);
        }
    }
    pub fn expression(&self) -> E {
        E::Sum(
            self.terms
                .iter()
                .map(|(word, &c)| {
                    E::Product(
                        std::iter::once(E::Scalar(c))
                            .chain(word.iter().map(|v| match v {
                                Variable::Q(m) => E::coordinate(*m),
                                Variable::P(m) => E::momentum(*m),
                            }))
                            .collect(),
                    )
                })
                .collect(),
        )
    }
    pub fn value(&self, q: &[f64; 163], p: &[f64; 163]) -> C {
        self.terms
            .iter()
            .map(|(word, c)| {
                word.iter().fold(*c, |x, v| {
                    x * match v {
                        Variable::Q(m) => q[*m as usize],
                        Variable::P(m) => p[*m as usize],
                    }
                })
            })
            .sum()
    }
}

pub type MatterMatrix = BTreeMap<(u32, u32), C>;
fn matrix_add(m: &mut MatterMatrix, r: u32, s: u32, c: C) {
    if c == C::new(0.0, 0.0) {
        return;
    }
    let value = m.entry((r, s)).or_default();
    *value += c;
    if *value == C::new(0.0, 0.0) {
        m.remove(&(r, s));
    }
}
pub fn matrix_bilinear(m: &MatterMatrix) -> E {
    E::Sum(
        m.iter()
            .map(|(&(r, s), &c)| {
                E::Product(vec![
                    E::Scalar(c),
                    E::Ladder(L::FermionCreate(r)),
                    E::Ladder(L::FermionAnnihilate(s)),
                ])
            })
            .collect(),
    )
}

#[derive(Clone, Debug)]
pub struct WeightedForm {
    pub name: String,
    pub weight: f64,
    pub polynomial: SmPolynomial,
}

/// All terms, with inspectable symbolic forms and matter coefficient matrices.
#[derive(Clone, Debug)]
pub struct SmHamiltonian {
    pub parameters: SmParameters,
    pub forms: Vec<WeightedForm>,
    pub matter: BTreeMap<Variable, MatterMatrix>,
}

impl SmHamiltonian {
    pub fn new(parameters: SmParameters) -> Result<Self, String> {
        parameters.validate()?;
        let mut model = Self {
            parameters,
            forms: Vec::new(),
            matter: BTreeMap::new(),
        };
        model.build_bosons();
        model.build_matter();
        Ok(model)
    }

    fn build_bosons(&mut self) {
        use Variable::Q;
        let f = su3_structure();
        for (label, colors, coupling) in [
            ("G", 8, self.parameters.gs),
            ("W", 3, self.parameters.gw),
            ("B", 1, 0.0),
        ] {
            for a in 0..colors {
                for i in 0..3 {
                    let mut poly = SmPolynomial::default();
                    for j in 0..3 {
                        for k in 0..3 {
                            let e = 0.5 * epsilon(i, j, k);
                            if e == 0.0 {
                                continue;
                            }
                            let derivative = |r, s| match label {
                                "G" => coord::dg(a, r, s),
                                "W" => coord::dw(a, r, s),
                                _ => coord::db(r, s),
                            };
                            poly.add(vec![Q(derivative(k, j))], C::new(e, 0.0));
                            poly.add(vec![Q(derivative(j, k))], C::new(-e, 0.0));
                            if label != "B" {
                                for b in 0..colors {
                                    for d in 0..colors {
                                        // D=d+i*g*T*A gives F=dA-dA-g*f*A*A (book's weak sign).
                                        let structure = if label == "G" {
                                            f[a as usize][b as usize][d as usize]
                                        } else {
                                            epsilon(a, b, d)
                                        };
                                        let field = |x, r| {
                                            if label == "G" {
                                                coord::g(x, r)
                                            } else {
                                                coord::w(x, r)
                                            }
                                        };
                                        poly.add(
                                            vec![Q(field(b, j)), Q(field(d, k))],
                                            C::new(-e * coupling * structure, 0.0),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    self.forms.push(WeightedForm {
                        name: format!("mag{label}{a}{i}"),
                        weight: 0.5,
                        polynomial: poly,
                    });
                }
            }
        }
        for a in 0..4 {
            for i in 0..3 {
                let mut poly = SmPolynomial::default();
                poly.add(vec![Q(coord::dphi(a, i))], C::new(1.0, 0.0));
                for b in 0..4 {
                    for j in 0..3 {
                        poly.add(
                            vec![Q(coord::w(j, i)), Q(coord::phi(b))],
                            C::new(
                                self.parameters.gw
                                    * higgs_real_generator(Some(j))[a as usize][b as usize],
                                0.0,
                            ),
                        );
                    }
                    poly.add(
                        vec![Q(coord::b(i)), Q(coord::phi(b))],
                        C::new(
                            self.parameters.gy * higgs_real_generator(None)[a as usize][b as usize],
                            0.0,
                        ),
                    );
                }
                self.forms.push(WeightedForm {
                    name: format!("covH{a}{i}"),
                    weight: 0.5,
                    polynomial: poly,
                });
            }
        }
        let mut wall = SmPolynomial::default();
        for a in 0..4 {
            wall.add(vec![Q(coord::phi(a)); 2], C::new(1.0, 0.0));
        }
        wall.add(vec![], C::new(-self.parameters.vev.powi(2), 0.0));
        self.forms.push(WeightedForm {
            name: "wall".into(),
            weight: self.parameters.lambda / 4.0,
            polynomial: wall,
        });
    }

    fn build_matter(&mut self) {
        use Chirality::*;
        use Species::*;
        use Variable::*;
        let su3 = su3_generators();
        // -i alpha·D, at fixed independent collective coordinates; no fixed
        // plane-wave parameter replaces p_x. All species, chirality and spins.
        for species in [Up, Down, Neutrino, Electron] {
            for generation in 0..3 {
                for color in 0..species.colors() {
                    for ch in [Left, Right] {
                        for j in 0..3 {
                            for spin in 0..2 {
                                for other_spin in 0..2 {
                                    let alpha = book_kinetic_spin(j)[spin as usize]
                                        [other_spin as usize]
                                        * if ch == Left { -1.0 } else { 1.0 };
                                    let row = matter_mode(species, generation, color, ch, spin);
                                    let col =
                                        matter_mode(species, generation, color, ch, other_spin);
                                    matrix_add(
                                        self.matter.entry(P(coord::x(j))).or_default(),
                                        row,
                                        col,
                                        alpha,
                                    );
                                    matrix_add(
                                        self.matter.entry(Q(coord::b(j))).or_default(),
                                        row,
                                        col,
                                        alpha * self.parameters.gy * species.hypercharge(ch),
                                    );
                                    if species.colors() == 3 {
                                        for a in 0..8 {
                                            for other_color in 0..3 {
                                                matrix_add(
                                                    self.matter
                                                        .entry(Q(coord::g(a, j)))
                                                        .or_default(),
                                                    row,
                                                    matter_mode(
                                                        species,
                                                        generation,
                                                        other_color,
                                                        ch,
                                                        other_spin,
                                                    ),
                                                    alpha
                                                        * self.parameters.gs
                                                        * su3[a as usize][color as usize]
                                                            [other_color as usize],
                                                );
                                            }
                                        }
                                    }
                                    if ch == Left {
                                        let (doublet, index) = if matches!(species, Up | Down) {
                                            ([Up, Down], if species == Up { 0 } else { 1 })
                                        } else {
                                            (
                                                [Neutrino, Electron],
                                                if species == Neutrino { 0 } else { 1 },
                                            )
                                        };
                                        for a in 0..3 {
                                            for (k, target) in doublet.iter().copied().enumerate() {
                                                matrix_add(
                                                    self.matter
                                                        .entry(Q(coord::w(a, j)))
                                                        .or_default(),
                                                    row,
                                                    matter_mode(
                                                        target, generation, color, Left, other_spin,
                                                    ),
                                                    alpha
                                                        * self.parameters.gw
                                                        * pauli(a)[index][k]
                                                        * 0.5,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Qbar H d_R + Qbar (i sigma_2 H*) u_R, and lepton analogues,
        // with all complex generation entries plus Hermitian conjugates.
        for (right, doublet, matrix, up_type) in [
            (Up, [Up, Down], &self.parameters.yu, true),
            (Down, [Up, Down], &self.parameters.yd, false),
            (Neutrino, [Neutrino, Electron], &self.parameters.ynu, true),
            (Electron, [Neutrino, Electron], &self.parameters.ye, false),
        ] {
            for gl in 0..3 {
                for gr in 0..3 {
                    for color in 0..right.colors() {
                        for spin in 0..2 {
                            for (weak, left) in doublet.iter().copied().enumerate() {
                                for a in 0..4 {
                                    let z = higgs_yukawa_component(up_type, weak, a);
                                    let coupling = matrix[gl as usize][gr as usize] * z;
                                    let row = matter_mode(left, gl, color, Left, spin);
                                    let col = matter_mode(right, gr, color, Right, spin);
                                    let target =
                                        self.matter.entry(Q(coord::phi(a as u32))).or_default();
                                    matrix_add(target, row, col, coupling);
                                    matrix_add(target, col, row, coupling.conj());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn bosonic_expression(&self) -> E {
        E::Sum(
            coord::active()
                .into_iter()
                .map(|m| E::Product(vec![E::momentum(m), E::momentum(m)]).scaled(C::new(0.5, 0.0)))
                .chain(self.forms.iter().map(|form| {
                    let x = form.polynomial.expression();
                    E::Product(vec![x.clone(), x]).scaled(C::new(form.weight, 0.0))
                }))
                .collect(),
        )
    }

    pub fn expression(&self) -> E {
        E::Sum(
            std::iter::once(self.bosonic_expression())
                .chain(self.matter.iter().map(|(&v, m)| {
                    let coefficient = match v {
                        Variable::Q(i) => E::coordinate(i),
                        Variable::P(i) => E::momentum(i),
                    };
                    E::Product(vec![coefficient, matrix_bilinear(m)])
                }))
                .collect(),
        )
    }

    /// Actual scalar comparison candidate, not an added physical confinement.
    pub fn comparison_expression(&self, c: f64) -> Result<E, String> {
        if !c.is_finite() || c < 1.0 {
            return Err("comparison shift must be finite and >=1".into());
        }
        let mut terms = vec![
            self.bosonic_expression().scaled(C::new(2.0, 0.0)),
            E::Scalar(C::new(c, 0.0)),
        ];
        for m in coord::active().into_iter().chain(0..3) {
            terms.push(E::Product(vec![E::coordinate(m), E::coordinate(m)]));
        }
        for m in 0..3 {
            terms.push(E::Product(vec![E::momentum(m), E::momentum(m)]));
        }
        Ok(E::Sum(terms))
    }

    pub fn matter_entry_polynomials(&self) -> BTreeMap<(u32, u32), SmPolynomial> {
        let mut out = BTreeMap::<(u32, u32), SmPolynomial>::new();
        for (&variable, matrix) in &self.matter {
            for (&pair, &coefficient) in matrix {
                out.entry(pair)
                    .or_default()
                    .add(vec![variable], coefficient);
            }
        }
        out
    }
}

pub fn higgs_yukawa_component(up_type: bool, weak: usize, component: usize) -> C {
    let r = 1.0 / 2f64.sqrt();
    match (up_type, weak, component) {
        (false, 0, 0) | (false, 1, 2) | (true, 0, 2) => C::new(r, 0.0),
        (false, 0, 1) | (false, 1, 3) | (true, 1, 1) => C::new(0.0, r),
        (true, 0, 3) => C::new(0.0, -r),
        (true, 1, 0) => C::new(-r, 0.0),
        _ => C::new(0.0, 0.0),
    }
}

#[cfg(feature = "latex")]
pub mod cadabra_import {
    use super::*;
    use mathhook::prelude::{Expression, MathConstant, Parser, ParserConfig};

    fn canonical_symbol(s: &str) -> String {
        s.chars()
            .filter(|c| !matches!(c, '_' | '{' | '}' | ' ' | '\\'))
            .collect()
    }
    fn scalar(p: &SmPolynomial) -> Result<C, String> {
        if p.terms.keys().any(|word| !word.is_empty()) {
            return Err("non-scalar denominator/exponent".into());
        }
        Ok(p.terms.get(&vec![]).copied().unwrap_or_default())
    }
    fn constant(c: C) -> SmPolynomial {
        let mut p = SmPolynomial::default();
        p.add(vec![], c);
        p
    }
    fn multiply(a: &SmPolynomial, b: &SmPolynomial) -> Result<SmPolynomial, String> {
        if a.terms
            .len()
            .checked_mul(b.terms.len())
            .is_none_or(|n| n > 100_000)
        {
            return Err("symbolic polynomial expansion budget exceeded".into());
        }
        let mut p = SmPolynomial::default();
        for (x, c) in &a.terms {
            for (y, d) in &b.terms {
                let mut word = x.clone();
                word.extend(y);
                if word.len() > 4 {
                    return Err("SM coefficient polynomial degree exceeds four".into());
                }
                if word
                    .iter()
                    .any(|v| matches!(v,Variable::Q(m) if word.contains(&Variable::P(*m))))
                {
                    return Err(
                        "noncommuting same-mode q*p cannot cross the scalar LaTeX bridge".into(),
                    );
                }
                p.add(word, c * d);
            }
        }
        Ok(p)
    }
    fn walk(
        expr: &Expression,
        bindings: &BTreeMap<String, C>,
        depth: usize,
    ) -> Result<SmPolynomial, String> {
        if depth > 128 {
            return Err("LaTeX expression depth limit".into());
        }
        match expr {
            Expression::Number(n) => {
                let text = n.to_string();
                let parts: Vec<_> = text.split('/').collect();
                let x = match parts.as_slice() {
                    [x] => x.parse::<f64>().map_err(|_| "invalid number")?,
                    [x, y] => {
                        x.parse::<f64>().map_err(|_| "invalid numerator")?
                            / y.parse::<f64>().map_err(|_| "invalid denominator")?
                    }
                    _ => return Err("unsupported numeric representation".into()),
                };
                if !x.is_finite() {
                    return Err("nonfinite imported number".into());
                }
                Ok(constant(C::new(x, 0.0)))
            }
            Expression::Symbol(symbol) => {
                let name = canonical_symbol(&symbol.name);
                if let Some(c) = bindings.get(&name) {
                    return Ok(constant(*c));
                }
                if name == "I" || name == "i" {
                    return Ok(constant(C::i()));
                }
                if let Some((prefix, index)) = name.get(..1).zip(name.get(1..)) {
                    if (prefix == "q" || prefix == "p") && index.chars().all(|x| x.is_ascii_digit())
                    {
                        let m = index
                            .parse::<u32>()
                            .map_err(|_| "invalid coordinate index")?;
                        if m >= BOSON_MODES {
                            return Err("SM coordinate index outside 163".into());
                        }
                        let mut p = SmPolynomial::default();
                        p.add(
                            vec![if prefix == "q" {
                                Variable::Q(m)
                            } else {
                                Variable::P(m)
                            }],
                            C::new(1.0, 0.0),
                        );
                        return Ok(p);
                    }
                }
                Err(format!(
                    "unknown Cadabra scalar/coordinate: {}",
                    symbol.name
                ))
            }
            Expression::Constant(MathConstant::I) => Ok(constant(C::i())),
            Expression::Add(terms) => {
                let mut p = SmPolynomial::default();
                for t in terms.iter() {
                    for (word, c) in walk(t, bindings, depth + 1)?.terms {
                        p.add(word, c);
                    }
                }
                Ok(p)
            }
            Expression::Mul(factors) => {
                let mut p = constant(C::new(1.0, 0.0));
                for f in factors.iter() {
                    p = multiply(&p, &walk(f, bindings, depth + 1)?)?;
                }
                Ok(p)
            }
            Expression::Pow(base, exponent) => {
                let x = walk(base, bindings, depth + 1)?;
                let n = scalar(&walk(exponent, bindings, depth + 1)?)?;
                if n.im != 0.0 || !n.re.is_finite() {
                    return Err("non-real exponent".into());
                }
                if x.terms.keys().all(|word| word.is_empty()) {
                    let value = scalar(&x)?.powf(n.re);
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err("nonfinite scalar power".into());
                    }
                    return Ok(constant(value));
                }
                if n.re < 0.0 || n.re > 4.0 || n.re.fract() != 0.0 {
                    return Err("non-polynomial coordinate power".into());
                }
                let mut out = constant(C::new(1.0, 0.0));
                for _ in 0..n.re as usize {
                    out = multiply(&out, &x)?;
                }
                Ok(out)
            }
            Expression::Function { name, args } if name.as_ref() == "frac" && args.len() == 2 => {
                let numerator = walk(&args[0], bindings, depth + 1)?;
                let denominator = scalar(&walk(&args[1], bindings, depth + 1)?)?;
                if denominator == C::new(0.0, 0.0) {
                    return Err("zero denominator".into());
                }
                multiply(&numerator, &constant(C::new(1.0, 0.0) / denominator))
            }
            _ => Err(format!("unsupported Cadabra LaTeX AST: {expr:?}")),
        }
    }

    /// Imports commuting coefficient polynomials through the existing mathhook
    /// calculus crate. No legacy scalar stripping or noncommuting sorting path.
    pub fn polynomial_from_latex(
        latex: &str,
        bindings: &BTreeMap<String, C>,
    ) -> Result<SmPolynomial, String> {
        let mut text = latex.to_owned();
        let mut mapped = BTreeMap::new();
        // Cadabra renders yu00 as {yu_{00}}. Alphabetic placeholders avoid
        // parsers numerically collapsing its two independent generation labels.
        for (name, &value) in bindings {
            let digits: String = name.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() == 2 {
                let prefix: String = name.chars().filter(|c| !c.is_ascii_digit()).collect();
                let encoded = format!(
                    "{}{}{}",
                    prefix,
                    (b'a' + digits.as_bytes()[0] - b'0') as char,
                    (b'a' + digits.as_bytes()[1] - b'0') as char
                );
                text = text.replace(&format!("{{{prefix}_{{{digits}}}}}"), &encoded);
                text = text.replace(&format!("{prefix}_{{{digits}}}"), &encoded);
                mapped.insert(encoded, value);
            } else {
                mapped.insert(name.clone(), value);
            }
        }
        for m in 0..BOSON_MODES {
            text = text.replace(&format!("q_{{{m}}}"), &format!("q_{m}"));
            text = text.replace(&format!("p_{{{m}}}"), &format!("p_{m}"));
        }
        text = normalize_tex_groups(&text, 0)?;
        // Bare multi-letter symbols are split by the implicit-product lexer.
        // Scalar indexed aliases preserve their meaning in the existing parser.
        let aliases: BTreeMap<_, _> = mapped
            .iter()
            .enumerate()
            .map(|(i, (name, &value))| (name.clone(), (format!("z_{i}"), value)))
            .collect();
        let mut rewritten = String::new();
        let mut pos = 0;
        while pos < text.len() {
            let start = pos;
            if text.as_bytes()[pos].is_ascii_alphabetic() {
                while pos < text.len()
                    && (text.as_bytes()[pos].is_ascii_alphanumeric()
                        || text.as_bytes()[pos] == b'_')
                {
                    pos += 1;
                }
                let word = &text[start..pos];
                rewritten.push_str(aliases.get(word).map(|x| x.0.as_str()).unwrap_or(word));
            } else {
                rewritten.push(text.as_bytes()[pos] as char);
                pos += 1;
            }
        }
        text = rewritten;
        // Cadabra suppresses some multiplication signs next to grouped powers
        // and I. Insert those lexical products before the calculus parser.
        // Functions have already been normalized/rejected by the TeX adapter.
        let mut tokens = Vec::<String>::new();
        let mut cursor = 0;
        while cursor < text.len() {
            let start = cursor;
            let ch = text.as_bytes()[cursor];
            if ch.is_ascii_whitespace() {
                cursor += 1;
                continue;
            }
            if ch.is_ascii_alphabetic() {
                cursor += 1;
                while cursor < text.len()
                    && (text.as_bytes()[cursor].is_ascii_alphanumeric()
                        || text.as_bytes()[cursor] == b'_')
                {
                    cursor += 1;
                }
            } else if ch.is_ascii_digit() || ch == b'.' {
                cursor += 1;
                while cursor < text.len()
                    && (text.as_bytes()[cursor].is_ascii_digit() || text.as_bytes()[cursor] == b'.')
                {
                    cursor += 1;
                }
            } else {
                cursor += 1;
            }
            tokens.push(text[start..cursor].to_owned());
        }
        text = String::new();
        let mut previous_atom = false;
        for token in tokens {
            let initial = token.as_bytes()[0];
            let starts_atom = initial.is_ascii_alphanumeric() || initial == b'.' || initial == b'(';
            if previous_atom && starts_atom {
                text.push_str(" * ");
            }
            text.push_str(&token);
            previous_atom = initial.is_ascii_alphanumeric() || initial == b'.' || token == ")";
        }
        mapped = aliases
            .into_values()
            .map(|(name, value)| (canonical_symbol(&name), value))
            .collect();
        let parser = Parser::new(&ParserConfig::default());
        let ast = parser
            .parse(&text)
            .map_err(|e| format!("Cadabra LaTeX parse: {e:?}: {text}"))?;
        let p = walk(&ast, &mapped, 0)?;
        if p.terms
            .values()
            .any(|x| !x.re.is_finite() || !x.im.is_finite())
        {
            return Err("nonfinite imported polynomial".into());
        }
        Ok(p)
    }

    // mathhook 0.2 accepts a_n but not a_{n}, and ordinary braced grouping
    // is not accepted in every grammar position. Normalize only this display
    // syntax; expression parsing and its AST remain the existing calculus crate.
    fn normalize_tex_groups(input: &str, depth: usize) -> Result<String, String> {
        if depth > 128 {
            return Err("TeX group depth limit".into());
        }
        fn group<'a>(input: &'a str, pos: &mut usize) -> Result<&'a str, String> {
            while input.as_bytes().get(*pos) == Some(&b' ') {
                *pos += 1;
            }
            if input.as_bytes().get(*pos) != Some(&b'{') {
                return Err("expected TeX group".into());
            }
            *pos += 1;
            let start = *pos;
            let mut nesting = 1;
            while *pos < input.len() {
                match input.as_bytes()[*pos] {
                    b'{' => nesting += 1,
                    b'}' => nesting -= 1,
                    _ => {}
                }
                if nesting == 0 {
                    let end = *pos;
                    *pos += 1;
                    return Ok(&input[start..end]);
                }
                *pos += 1;
            }
            Err("unclosed TeX group".into())
        }
        if !input.is_ascii() {
            return Err("unsupported non-ASCII coefficient display".into());
        }
        let mut out = String::new();
        let mut pos = 0;
        while pos < input.len() {
            if input[pos..].starts_with("\\frac") {
                pos += 5;
                let a = group(input, &mut pos)?;
                let b = group(input, &mut pos)?;
                out.push_str(&format!(
                    "(({})/({}))",
                    normalize_tex_groups(a, depth + 1)?,
                    normalize_tex_groups(b, depth + 1)?
                ));
            } else if input.as_bytes()[pos] == b'{' {
                let a = group(input, &mut pos)?;
                out.push('(');
                out.push_str(&normalize_tex_groups(a, depth + 1)?);
                out.push(')');
            } else if input.as_bytes()[pos] == b'\\' {
                return Err("unsupported TeX coefficient command".into());
            } else {
                out.push(input.as_bytes()[pos] as char);
                pos += 1;
            }
        }
        Ok(out)
    }

    fn equal_polynomial(a: &SmPolynomial, b: &SmPolynomial) -> bool {
        a.terms.keys().chain(b.terms.keys()).all(|word| {
            let x = a.terms.get(word).copied().unwrap_or_default();
            let y = b.terms.get(word).copied().unwrap_or_default();
            (x - y).norm() <= 1e-12 * (1.0 + x.norm() + y.norm())
        })
    }

    /// Build the numerical operator from actual Cadabra LaTeX exports, checked
    /// termwise against the independent Rust dictionary. Missing, extra or
    /// mutated components fail rather than silently producing a smaller model.
    pub fn from_cadabra_latex(
        parameters: SmParameters,
        export: &str,
    ) -> Result<SmHamiltonian, String> {
        if !export.lines().any(|l| l == "SM_COMPLETE|163|40|49|96|1152") {
            return Err("missing complete Cadabra pipeline trailer".into());
        }
        let bindings = parameters.symbolic_bindings();
        let mut records = BTreeMap::new();
        for line in export.lines().filter(|l| l.starts_with("SM_LATEX|")) {
            let (_, rest) = line.split_once('|').unwrap();
            let (name, latex) = rest.split_once('|').ok_or("malformed Cadabra export")?;
            if records
                .insert(
                    name.to_owned(),
                    polynomial_from_latex(latex, &bindings).map_err(|e| format!("{name}: {e}"))?,
                )
                .is_some()
            {
                return Err("duplicate Cadabra component".into());
            }
        }
        let mut model = SmHamiltonian::new(parameters)?;
        for m in coord::active() {
            let actual = records
                .remove(&format!("kin{m}"))
                .ok_or("missing field momentum export")?;
            let mut expected = SmPolynomial::default();
            expected.add(vec![Variable::P(m)], C::new(1.0, 0.0));
            if !equal_polynomial(&actual, &expected) {
                return Err("wrong field momentum".into());
            }
        }
        for form in &mut model.forms {
            let actual = records
                .remove(&form.name)
                .ok_or_else(|| format!("missing form {}", form.name))?;
            if !equal_polynomial(&actual, &form.polynomial) {
                return Err(format!("Cadabra/Rust form mismatch: {}", form.name));
            }
            form.polynomial = actual;
        }
        let expected = model.matter_entry_polynomials();
        let mut matter = BTreeMap::<Variable, MatterMatrix>::new();
        // The symbolic matrix has 1,152 entries even when chosen parameter
        // values make individual entries zero after binding.
        for r in 0..MATTER_MODES {
            for s in 0..MATTER_MODES {
                if let Some(actual) = records.remove(&format!("matter{r}x{s}")) {
                    let zero = SmPolynomial::default();
                    let reference = expected.get(&(r, s)).unwrap_or(&zero);
                    if !equal_polynomial(&actual, reference) {
                        return Err(format!("Cadabra/Rust matter mismatch ({r},{s})"));
                    }
                    for (word, c) in actual.terms {
                        if word.len() != 1 {
                            return Err("matter coefficient not linear in q/p".into());
                        }
                        matrix_add(matter.entry(word[0]).or_default(), r, s, c);
                    }
                } else if expected.contains_key(&(r, s)) {
                    return Err(format!("missing matter entry ({r},{s})"));
                }
            }
        }
        if !records.is_empty() {
            return Err("unexpected Cadabra component".into());
        }
        model.matter = matter;
        Ok(model)
    }
}
