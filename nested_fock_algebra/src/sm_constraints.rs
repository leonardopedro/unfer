//! Joint SU(3)xSU(2)xU(1) Gauss/BRST dictionary for the complete E1 operator.
//!
//! This is a finite-representation construction, not a proof of the physical
//! BRST cohomology. The gauge generators use the 163 named coordinates, the 40
//! active field momenta, the 96 complex CAR coordinates and the four real Higgs
//! coordinates already used by [`crate::sm_full`]. Ghosts are twelve additional
//! fermionic modes at and above `MATTER_MODES`; the antighost of a ghost mode is
//! its conjugate Fock ladder, so `{c_a,b_b}=delta_ab` holds in the dictionary.
//!
//! The electric term is the symmetrized nonabelian `A.E` density. Matter and
//! Higgs currents are normal-ordered (creation left of annihilation). No
//! gauge-fixing, auxiliary field, anomaly or infinite-mode completion is added.

use crate::exact_inner::{InnerExpr as E, InnerLadder as L};
use crate::sm_full::{
    Chirality, MATTER_MODES, MatterMatrix, SmParameters, Species, coord, higgs_real_generator,
    matrix_bilinear, matter_mode, su3_generators, su3_structure,
};
use num_complex::Complex64 as C;

/// Number of gauge generators: eight SU(3), three SU(2), one U(1).
pub const GENERATORS: usize = 12;
/// First ghost/antighost Fock mode. Modes `GHOST_BASE..GHOST_BASE+12` are used.
pub const GHOST_BASE: u32 = MATTER_MODES;

/// Which simple or abelian factor a generator belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sector {
    Su3(u32),
    Su2(u32),
    U1,
}

/// Stable generator order and labels.
pub fn generators() -> Vec<(Sector, &'static str)> {
    let mut out = Vec::with_capacity(GENERATORS);
    for a in 0..8 {
        out.push((
            Sector::Su3(a),
            [
                "su3_0", "su3_1", "su3_2", "su3_3", "su3_4", "su3_5", "su3_6", "su3_7",
            ][a as usize],
        ));
    }
    for a in 0..3 {
        out.push((Sector::Su2(a), ["su2_0", "su2_1", "su2_2"][a as usize]));
    }
    out.push((Sector::U1, "u1_y"));
    out
}

fn su2_structure(a: u32, b: u32, c: u32) -> f64 {
    crate::sm_full::epsilon(a, b, c)
}

/// `f_abc` for the generator order above. The abelian factor has zero.
pub fn structure(a: usize, b: usize, c: usize) -> f64 {
    if a < 8 && b < 8 && c < 8 {
        su3_structure()[a][b][c]
    } else if (8..11).contains(&a) && (8..11).contains(&b) && (8..11).contains(&c) {
        su2_structure(a as u32 - 8, b as u32 - 8, c as u32 - 8)
    } else {
        0.0
    }
}

/// Symmetrized `q^T t p` on four real Higgs coordinates.
pub fn higgs_charge(t: &[[f64; 4]; 4]) -> E {
    let mut terms = Vec::new();
    for (r, row) in t.iter().enumerate() {
        for (s, c) in row.iter().enumerate() {
            let c = *c;
            if c != 0.0 {
                let qr = E::coordinate(coord::phi(r as u32));
                let ps = E::momentum(coord::phi(s as u32));
                terms.push(E::Product(vec![
                    E::Scalar(C::new(0.5 * c, 0.0)),
                    qr.clone(),
                    ps.clone(),
                ]));
                terms.push(E::Product(vec![E::Scalar(C::new(0.5 * c, 0.0)), ps, qr]));
            }
        }
    }
    E::Sum(terms)
}

/// Symmetrized nonabelian electric density `f^abc sum_i A_i^b E_i^c`.
///
/// The generator uses the **bare** structure constants, not `g f`: with the
/// matter current also bare, the joint charge is one Lie-algebra element and
/// `[G_a,G_b]=i f_abc G_c`. The coupling belongs to the Hamiltonian's magnetic
/// and electric energy, not to the Gauss generator. All eight SU(3) colours are
/// included; an earlier three-colour loop was the measured N10 obstruction.
///
/// This is the homogeneous (zero-derivative) part. The independent jet
/// coordinates carry the derivative directions; [`adjoint_jet`] rotates them.
/// On the tested gauge-occupied images the homogeneous+jet generator closes
/// without a separate `d_i E_i` trace; adding that trace with an independent
/// jet momentum would not change the algebra because it commutes with `A E`.
pub fn electric(_parameters: &SmParameters, sector: Sector) -> E {
    let mut terms = Vec::new();
    let (a, n) = match sector {
        Sector::Su3(a) => (a, 8u32),
        Sector::Su2(a) => (a, 3u32),
        Sector::U1 => return E::Sum(Vec::new()),
    };
    for b in 0..n {
        for c in 0..n {
            let f = if n == 3 {
                su2_structure(a, b, c)
            } else {
                su3_structure()[a as usize][b as usize][c as usize]
            };
            if f == 0.0 {
                continue;
            }
            for i in 0..3 {
                let a_mode = match sector {
                    Sector::Su3(_) => coord::g(b, i),
                    Sector::Su2(_) => coord::w(b, i),
                    Sector::U1 => unreachable!(),
                };
                let e_mode = match sector {
                    Sector::Su3(_) => coord::g(c, i),
                    Sector::Su2(_) => coord::w(c, i),
                    Sector::U1 => unreachable!(),
                };
                let qa = E::coordinate(a_mode);
                let pe = E::momentum(e_mode);
                let coeff = C::new(0.5 * f, 0.0);
                terms.push(E::Product(vec![E::Scalar(coeff), qa.clone(), pe.clone()]));
                terms.push(E::Product(vec![E::Scalar(coeff), pe, qa]));
            }
        }
    }
    E::Sum(terms)
}

/// Normal-ordered matter current for one factor.
pub fn matter_current(sector: Sector) -> E {
    let mut m = MatterMatrix::new();
    match sector {
        Sector::Su3(a) => {
            let t = su3_generators()[a as usize];
            for species in [Species::Up, Species::Down] {
                for generation in 0..3 {
                    for chirality in [Chirality::Left, Chirality::Right] {
                        for spin in 0..2 {
                            for r in 0..3 {
                                for s in 0..3 {
                                    let c = t[r as usize][s as usize];
                                    if c != C::new(0.0, 0.0) {
                                        let create =
                                            matter_mode(species, generation, r, chirality, spin);
                                        let destroy =
                                            matter_mode(species, generation, s, chirality, spin);
                                        *m.entry((create, destroy)).or_default() += c;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Sector::Su2(a) => {
            let t = crate::sm_full::pauli(a).map(|row| row.map(|x| x * 0.5));
            for (up, down) in [
                (Species::Up, Species::Down),
                (Species::Neutrino, Species::Electron),
            ] {
                for generation in 0..3 {
                    for color in 0..up.colors() {
                        for spin in 0..2 {
                            for (r, row) in t.iter().enumerate() {
                                for (s, c) in row.iter().enumerate() {
                                    let c = *c;
                                    if c != C::new(0.0, 0.0) {
                                        let species = if r == 0 { up } else { down };
                                        let other = if s == 0 { up } else { down };
                                        let create = matter_mode(
                                            species,
                                            generation,
                                            color,
                                            Chirality::Left,
                                            spin,
                                        );
                                        let destroy = matter_mode(
                                            other,
                                            generation,
                                            color,
                                            Chirality::Left,
                                            spin,
                                        );
                                        *m.entry((create, destroy)).or_default() += c;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Sector::U1 => {
            for species in [
                Species::Up,
                Species::Down,
                Species::Neutrino,
                Species::Electron,
            ] {
                for generation in 0..3 {
                    for color in 0..species.colors() {
                        for chirality in [Chirality::Left, Chirality::Right] {
                            for spin in 0..2 {
                                let mode = matter_mode(species, generation, color, chirality, spin);
                                let y = species.hypercharge(chirality);
                                if y != 0.0 {
                                    *m.entry((mode, mode)).or_default() += C::new(y, 0.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    matrix_bilinear(&m)
}

/// Adjoint action of a homogeneous gauge rotation on the derivative
/// coordinates of a gauge field, `f^abc dA^b p_dA^c`.
///
/// This is the jet counterpart of [`electric`]: it rotates the independent
/// derivative coordinates exactly as `electric` rotates the homogeneous
/// coordinates, with the same bare structure constants. Without it the magnetic
/// term is not gauge invariant on the E1 jet space.
pub fn adjoint_jet(sector: Sector, _coupling: f64) -> E {
    let (colors, base) = match sector {
        Sector::Su3(_) => (8u32, 1u32),
        Sector::Su2(_) => (3u32, 2u32),
        Sector::U1 => return E::Sum(Vec::new()),
    };
    let a = match sector {
        Sector::Su3(a) | Sector::Su2(a) => a,
        Sector::U1 => unreachable!(),
    };
    let mut terms = Vec::new();
    for b in 0..colors {
        for c in 0..colors {
            let f = if colors == 3 {
                su2_structure(a, b, c)
            } else {
                su3_structure()[a as usize][b as usize][c as usize]
            };
            if f == 0.0 {
                continue;
            }
            for i in 0..3 {
                for j in 0..3 {
                    let (qb, pc) = if base == 1 {
                        (coord::dg(b, i, j), coord::dg(c, i, j))
                    } else {
                        (coord::dw(b, i, j), coord::dw(c, i, j))
                    };
                    let q = E::coordinate(qb);
                    let p = E::momentum(pc);
                    let coeff = C::new(0.5 * f, 0.0);
                    terms.push(E::Product(vec![E::Scalar(coeff), q.clone(), p.clone()]));
                    terms.push(E::Product(vec![E::Scalar(coeff), p, q]));
                }
            }
        }
    }
    E::Sum(terms)
}

/// Adjoint action on the derivative coordinates of the Higgs field.
pub fn higgs_jet_charge(sector: Sector) -> E {
    let (t, coupling) = match sector {
        Sector::Su2(a) => (higgs_real_generator(Some(a)), 1.0),
        Sector::U1 => (higgs_real_generator(None), 1.0),
        Sector::Su3(_) => return E::Sum(Vec::new()),
    };
    let mut terms = Vec::new();
    for i in 0..3 {
        for (r, row) in t.iter().enumerate() {
            for (s, c) in row.iter().enumerate() {
                let c = *c;
                if c == 0.0 {
                    continue;
                }
                let q = E::coordinate(coord::dphi(r as u32, i));
                let p = E::momentum(coord::dphi(s as u32, i));
                let coeff = C::new(0.5 * coupling * c, 0.0);
                terms.push(E::Product(vec![E::Scalar(coeff), q.clone(), p.clone()]));
                terms.push(E::Product(vec![E::Scalar(coeff), p, q]));
            }
        }
    }
    E::Sum(terms)
}

/// Full homogeneous+jet Gauss generator.
///
/// The jet terms carry the same adjoint/fundamental representation as the
/// homogeneous terms, so `sum_i q_i T p_i` is formed for every independent
/// derivative direction. This finite generator closes on both gauge-vacuum and
/// gauge-occupied images with bare structure constants. It is still not a proof
/// of the physical cohomology: the `d_i E_i` trace is absent, and the jet
/// coordinates are independent rather than constrained to `d_i A`; see the N10
/// record.
pub fn gauge_generator(parameters: &SmParameters, sector: Sector) -> E {
    let higgs = match sector {
        Sector::Su3(_) => E::Sum(Vec::new()),
        Sector::Su2(a) => higgs_charge(&higgs_real_generator(Some(a))),
        Sector::U1 => higgs_charge(&higgs_real_generator(None)),
    };
    E::Sum(vec![
        electric(parameters, sector),
        matter_current(sector),
        higgs,
        adjoint_jet(
            sector,
            match sector {
                Sector::Su3(_) => parameters.gs,
                Sector::Su2(_) => parameters.gw,
                Sector::U1 => 0.0,
            },
        ),
        higgs_jet_charge(sector),
    ])
}

/// All twelve generators in the stable [`generators`] order.
pub fn gauge_dictionary(parameters: &SmParameters) -> Vec<E> {
    generators()
        .into_iter()
        .map(|(sector, _)| gauge_generator(parameters, sector))
        .collect()
}

/// Joint BRST charge
/// `Q = sum_a c_a G_a - (1/2) sum_abc f_abc c_a c_b b_c`,
/// with `c_a` the ghost annihilator and `b_a` its conjugate creator on mode
/// `GHOST_BASE+a`.
pub fn brst_charge(parameters: &SmParameters) -> E {
    let g = gauge_dictionary(parameters);
    let mut terms = Vec::new();
    for (a, ga) in g.into_iter().enumerate() {
        terms.push(E::Product(vec![
            ga,
            E::Ladder(L::FermionAnnihilate(GHOST_BASE + a as u32)),
        ]));
    }
    for a in 0..GENERATORS {
        for b in 0..GENERATORS {
            for c in 0..GENERATORS {
                let f = structure(a, b, c);
                if f == 0.0 {
                    continue;
                }
                terms.push(E::Product(vec![
                    E::Scalar(C::new(0.0, -0.5 * f)),
                    E::Ladder(L::FermionAnnihilate(GHOST_BASE + a as u32)),
                    E::Ladder(L::FermionAnnihilate(GHOST_BASE + b as u32)),
                    E::Ladder(L::FermionCreate(GHOST_BASE + c as u32)),
                ]));
            }
        }
    }
    E::Sum(terms)
}
