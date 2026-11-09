#![cfg(feature = "latex")]
//! Full E1 SM dictionary and Cadabra-to-exact-action contract checks.
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr as E, InnerLadder as L, InnerVector, JointInnerBasis,
};
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

fn parameters() -> SmParameters {
    let matrix = |scale: f64| {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                C::new(
                    scale * (1 + i + 2 * j) as f64,
                    scale * 0.17 * (1 + 2 * i + j) as f64,
                )
            })
        })
    };
    SmParameters {
        gs: 0.3,
        gw: 0.2,
        gy: 0.15,
        lambda: 0.1,
        vev: 0.7,
        yu: matrix(0.02),
        yd: matrix(0.013),
        ynu: matrix(0.007),
        ye: matrix(0.019),
    }
}
fn model() -> &'static SmHamiltonian {
    static MODEL: OnceLock<SmHamiltonian> = OnceLock::new();
    MODEL.get_or_init(|| {
        cadabra_import::from_cadabra_latex(
            parameters(),
            include_str!("../../docs/fixtures/sm_full_cadabra.latex"),
        )
        .unwrap()
    })
}
fn close(a: C, b: C) {
    assert!((a - b).norm() < 1e-10, "{a} != {b}");
}
fn matrix_close(a: &MatterMatrix, b: &MatterMatrix) {
    for k in a.keys().chain(b.keys()) {
        close(
            a.get(k).copied().unwrap_or_default(),
            b.get(k).copied().unwrap_or_default(),
        );
    }
}
fn add(m: &mut MatterMatrix, r: u32, s: u32, c: C) {
    *m.entry((r, s)).or_default() += c;
}
fn commutator(a: &MatterMatrix, b: &MatterMatrix) -> MatterMatrix {
    let mut out = MatterMatrix::new();
    for (&(i, j), &x) in a {
        for (&(r, s), &y) in b {
            if j == r {
                add(&mut out, i, s, C::i() * x * y);
            }
            if s == i {
                add(&mut out, r, j, -C::i() * y * x);
            }
        }
    }
    out
}

#[test]
fn complete_coordinate_and_majorana_component_inventory() {
    let mut coords = BTreeSet::new();
    for i in 0..3 {
        assert!(coords.insert(coord::x(i)));
    }
    for a in 0..8 {
        for i in 0..3 {
            assert!(coords.insert(coord::g(a, i)));
            for j in 0..3 {
                assert!(coords.insert(coord::dg(a, i, j)));
            }
        }
    }
    for a in 0..3 {
        for i in 0..3 {
            assert!(coords.insert(coord::w(a, i)));
            for j in 0..3 {
                assert!(coords.insert(coord::dw(a, i, j)));
            }
        }
    }
    for i in 0..3 {
        assert!(coords.insert(coord::b(i)));
        for j in 0..3 {
            assert!(coords.insert(coord::db(i, j)));
        }
    }
    for a in 0..4 {
        assert!(coords.insert(coord::phi(a)));
        for i in 0..3 {
            assert!(coords.insert(coord::dphi(a, i)));
        }
    }
    assert_eq!(coords, (0..163).collect());
    assert_eq!(coord::active().len(), 40);
    let mut matter = BTreeSet::new();
    let mut real_counts = [0u32; 6];
    for s in [
        Species::Up,
        Species::Down,
        Species::Neutrino,
        Species::Electron,
    ] {
        for g in 0..3 {
            for c in 0..s.colors() {
                for ch in [Chirality::Left, Chirality::Right] {
                    for spin in 0..2 {
                        assert!(matter.insert(matter_mode(s, g, c, ch, spin)));
                        let index = match (s, ch) {
                            (Species::Up | Species::Down, Chirality::Left) => 0,
                            (Species::Up, Chirality::Right) => 1,
                            (Species::Down, Chirality::Right) => 2,
                            (Species::Neutrino | Species::Electron, Chirality::Left) => 3,
                            (Species::Neutrino, Chirality::Right) => 4,
                            (Species::Electron, Chirality::Right) => 5,
                        };
                        real_counts[index] += 2;
                    }
                }
            }
        }
    }
    assert_eq!(matter, (0..96).collect());
    assert_eq!(real_counts, [72, 36, 36, 24, 12, 12]);
    assert_eq!(model().forms.len(), 49);
    assert_eq!(model().matter.len(), 43);
    assert_eq!(model().matter_entry_polynomials().len(), 1152);
}

#[test]
fn cadabra_export_matches_all_forms_and_complete_matter_matrix() {
    let full = model();
    let direct = SmHamiltonian::new(parameters()).unwrap();
    let q: [f64; 163] = std::array::from_fn(|i| 0.013 * (i as f64 % 13.0 - 5.0));
    let p = std::array::from_fn(|i| 0.02 * (i as f64 % 7.0 - 3.0));
    for (a, b) in full.forms.iter().zip(&direct.forms) {
        assert_eq!(a.name, b.name);
        close(a.polynomial.value(&q, &p), b.polynomial.value(&q, &p));
    }
    for (&variable, m) in &full.matter {
        matrix_close(m, &direct.matter[&variable]);
        for (&(r, s), &x) in m {
            close(x, m.get(&(s, r)).copied().unwrap_or_default().conj());
        }
    }
    // The Higgs wall constant is retained before enclosure.
    let wall = full.forms.iter().find(|f| f.name == "wall").unwrap();
    let zero = [0.0; 163];
    close(wall.polynomial.value(&zero, &zero), C::new(-0.49, 0.0));
    close(
        C::new(wall.weight, 0.0) * wall.polynomial.value(&zero, &zero).powi(2),
        C::new(0.1 * 0.7f64.powi(4) / 4.0, 0.0),
    );
}

#[test]
fn all_four_yukawa_components_obey_full_gauge_covariance() {
    use Chirality::*;
    use Species::*;
    for weak in [None, Some(0), Some(1), Some(2)] {
        let mut generator = MatterMatrix::new();
        for s in [Up, Down, Neutrino, Electron] {
            for g in 0..3 {
                for c in 0..s.colors() {
                    for spin in 0..2 {
                        if let Some(a) = weak {
                            let doublet = if s.colors() == 3 {
                                [Up, Down]
                            } else {
                                [Neutrino, Electron]
                            };
                            let r = doublet.iter().position(|x| *x == s).unwrap();
                            for (v, &t) in doublet.iter().enumerate() {
                                add(
                                    &mut generator,
                                    matter_mode(s, g, c, Left, spin),
                                    matter_mode(t, g, c, Left, spin),
                                    pauli(a)[r][v] * 0.5,
                                );
                            }
                        } else {
                            for ch in [Left, Right] {
                                let m = matter_mode(s, g, c, ch, spin);
                                add(&mut generator, m, m, C::new(s.hypercharge(ch), 0.0));
                            }
                        }
                    }
                }
            }
        }
        let real = higgs_real_generator(weak);
        for b in 0..4 {
            let mut derivative = MatterMatrix::new();
            for a in 0..4 {
                for (&(r, s), &x) in &model().matter[&Variable::Q(coord::phi(a as u32))] {
                    add(&mut derivative, r, s, x * real[a][b]);
                }
            }
            matrix_close(
                &derivative,
                &commutator(
                    &generator,
                    &model().matter[&Variable::Q(coord::phi(b as u32))],
                ),
            );
        }
    }
    // Color singlet Yukawa contractions for every generation and spin.
    for t in su3_generators() {
        let mut generator = MatterMatrix::new();
        for s in [Up, Down] {
            for g in 0..3 {
                for ch in [Left, Right] {
                    for spin in 0..2 {
                        for c in 0..3 {
                            for d in 0..3 {
                                add(
                                    &mut generator,
                                    matter_mode(s, g, c, ch, spin),
                                    matter_mode(s, g, d, ch, spin),
                                    t[c as usize][d as usize],
                                );
                            }
                        }
                    }
                }
            }
        }
        for b in 0..4 {
            matrix_close(
                &commutator(&generator, &model().matter[&Variable::Q(coord::phi(b))]),
                &MatterMatrix::new(),
            );
        }
    }
}

#[test]
fn real_majorana_quadratic_form_matches_complex_car_with_constant_retained() {
    let q: [f64; 163] = std::array::from_fn(|i| 0.01 * (1 + i % 11) as f64);
    let p: [f64; 163] = std::array::from_fn(|i| 0.03 * (1 + i % 3) as f64);
    let mut k = MatterMatrix::new();
    for (&v, m) in &model().matter {
        for (&(r, s), &x) in m {
            add(
                &mut k,
                r,
                s,
                x * match v {
                    Variable::Q(m) => q[m as usize],
                    Variable::P(m) => p[m as usize],
                },
            );
        }
    }
    // R_xx=Im K, R_yy=Im K, R_xy=Re K, R_yx=-Re K.
    let mut real = BTreeMap::<(u32, u32), f64>::new();
    let mut trace = C::new(0.0, 0.0);
    for (&(r, s), &x) in &k {
        real.insert((2 * r, 2 * s), x.im);
        real.insert((2 * r + 1, 2 * s + 1), x.im);
        real.insert((2 * r, 2 * s + 1), x.re);
        real.insert((2 * r + 1, 2 * s), -x.re);
        if r == s {
            trace += x;
        }
    }
    for (&(r, s), &x) in &real {
        assert!((x + real.get(&(s, r)).copied().unwrap_or(0.0)).abs() < 1e-12);
    }
    let gamma = |a: u32| {
        let m = a / 2;
        let create = E::Ladder(L::FermionCreate(m));
        let annihilate = E::Ladder(L::FermionAnnihilate(m));
        if a % 2 == 0 {
            E::Sum(vec![create, annihilate])
        } else {
            E::Sum(vec![create.scaled(C::i()), annihilate.scaled(-C::i())])
        }
    };
    let majorana =
        E::Sum(
            std::iter::once(E::Scalar(trace * 0.5))
                .chain(real.iter().filter(|((r, s), x)| r < s && **x != 0.0).map(
                    |(&(r, s), &x)| {
                        E::Product(vec![gamma(r), gamma(s)]).scaled(C::new(0.0, x * 0.5))
                    },
                ))
                .collect(),
        );
    let limits = ActionLimits {
        max_components: 100_000,
        max_work: 100_000_000,
    };
    for occupied in [vec![], vec![0], vec![24], vec![0, 24], vec![31, 64]] {
        let v = InnerVector::from([(
            JointInnerBasis::new([], occupied).unwrap(),
            C::new(1.0, 0.0),
        )]);
        let a = majorana.apply(&v, limits).unwrap();
        let b = matrix_bilinear(&k).apply(&v, limits).unwrap();
        for basis in a.keys().chain(b.keys()) {
            close(
                a.get(basis).copied().unwrap_or_default(),
                b.get(basis).copied().unwrap_or_default(),
            );
        }
    }
}

#[test]
fn latex_bridge_fails_on_missing_mutated_or_noncommuting_terms() {
    let source = include_str!("../../docs/fixtures/sm_full_cadabra.latex");
    let missing = source
        .lines()
        .filter(|l| !l.starts_with("SM_LATEX|magG00|"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(cadabra_import::from_cadabra_latex(parameters(), &missing).is_err());
    let missing_matter = source
        .lines()
        .filter(|l| !l.starts_with("SM_LATEX|matter0x1|"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(cadabra_import::from_cadabra_latex(parameters(), &missing_matter).is_err());
    let altered = source.replace("SM_LATEX|kin3|p_{3}", "SM_LATEX|kin3|2 * p_{3}");
    assert!(cadabra_import::from_cadabra_latex(parameters(), &altered).is_err());
    assert!(cadabra_import::polynomial_from_latex("q_{3} * p_{3}", &BTreeMap::new()).is_err());
    assert!(cadabra_import::polynomial_from_latex("unrecognized", &BTreeMap::new()).is_err());
}

#[test]
fn fresh_cadabra_export_reproduces_checked_fixture() {
    let Some(cli) = std::env::var_os("CADABRA_CLI") else {
        eprintln!("SKIP: set CADABRA_CLI to run the engine/fixture equality check");
        return;
    };
    let module =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/sm_full_nested.cdb");
    let output = std::process::Command::new(cli)
        .args(["-q", "-n"])
        .arg(module)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        stdout.trim(),
        include_str!("../../docs/fixtures/sm_full_cadabra.latex").trim()
    );
}

#[test]
fn manuscript_majorana_spin_frame_matches_all_spatial_kinetic_blocks() {
    // Fixed source data: BookProof/Textbook/ChapterA3 and ChapterCPTHamiltonian.
    // This is independent of the constructor's Pauli formula.
    let mgamma = [
        [[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]],
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]],
    ];
    let j = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]];
    let multiply4 = |a: [[i32; 4]; 4], b: [[i32; 4]; 4]| {
        std::array::from_fn::<_, 4, _>(|r| {
            std::array::from_fn::<_, 4, _>(|s| (0..4).map(|k| a[r][k] * b[k][s]).sum::<i32>())
        })
    };
    let multiply2 = |a: [[C; 2]; 2], b: [[C; 2]; 2]| {
        std::array::from_fn::<_, 2, _>(|r| {
            std::array::from_fn::<_, 2, _>(|s| (0..2).map(|k| a[r][k] * b[k][s]).sum::<C>())
        })
    };
    let to_complex = |real: [[i32; 4]; 4]| {
        std::array::from_fn::<_, 2, _>(|r| {
            std::array::from_fn::<_, 2, _>(|s| {
                C::new(
                    real[2 * r][2 * s] as f64,
                    (if r == 0 { 1 } else { -1 }) as f64 * real[2 * r + 1][2 * s] as f64,
                )
            })
        })
    };
    let eps = [
        [C::new(0.0, 0.0), C::new(1.0, 0.0)],
        [C::new(-1.0, 0.0), C::new(0.0, 0.0)],
    ];
    let eps_adjoint = std::array::from_fn(|r| std::array::from_fn(|s| eps[s][r].conj()));
    let commuted = multiply4(mgamma[0], j);
    let reversed = multiply4(j, mgamma[0]);
    for r in 0..4 {
        for s in 0..4 {
            assert_eq!(commuted[r][s], -reversed[r][s]);
        }
    }
    for axis in 0..3 {
        let kinetic = multiply4(mgamma[axis + 1], mgamma[0]).map(|row| row.map(|x| -x));
        assert_eq!(multiply4(kinetic, j), multiply4(j, kinetic));
        let left = to_complex(kinetic);
        let right = multiply2(
            multiply2(eps, left.map(|row| row.map(|x| x.conj()))),
            eps_adjoint,
        );
        for r in 0..2 {
            for s in 0..2 {
                close(left[r][s], -book_kinetic_spin(axis as u32)[r][s]);
                close(right[r][s], book_kinetic_spin(axis as u32)[r][s]);
            }
        }
    }
    // i gamma0 maps the opposite-complex-structure right singlet to the left;
    // after the right spin-basis change, the Yukawa spin contraction is identity.
    let mass = multiply2(to_complex(mgamma[0]), eps_adjoint);
    for r in 0..2 {
        for s in 0..2 {
            close(mass[r][s], C::new(if r == s { 1.0 } else { 0.0 }, 0.0));
        }
    }
}
