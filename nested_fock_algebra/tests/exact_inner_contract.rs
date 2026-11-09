//! Construction regressions for SM work packets N1/N3. These exercise the new
//! ordered joint action, not a reduced model advertised as full-SM validation.
use nested_fock_algebra::exact_inner::{
    ActionError, ActionLimits, InnerExpr as E, InnerLadder as L, InnerVector, JointInnerBasis as B,
    JointOuterBasis as O, NestedVector,
};
use num_complex::Complex64 as C;
use std::collections::BTreeMap;

fn c(x: f64) -> C {
    C::new(x, 0.0)
}
fn basis(n: u32, fermions: &[u32]) -> B {
    B::new([(0, n)], fermions.iter().copied()).unwrap()
}
fn ket(b: B) -> InnerVector {
    InnerVector::from([(b, c(1.0))])
}
fn nested(b: O) -> NestedVector {
    NestedVector::from([(b, c(1.0))])
}
fn product(ls: &[L]) -> E {
    E::Product(ls.iter().copied().map(E::Ladder).collect())
}
fn apply(e: &E, v: &InnerVector) -> InnerVector {
    e.apply(v, ActionLimits::default()).unwrap()
}
fn lift(e: &E, v: &NestedVector) -> NestedVector {
    e.d_gamma(v, ActionLimits::default()).unwrap()
}
fn power(e: E, n: usize) -> E {
    E::Product(vec![e; n])
}
fn close(a: C, b: C) {
    assert!((a - b).norm() < 2e-10, "{a} != {b}");
}
fn vectors_close<K: Ord + std::fmt::Debug>(a: &BTreeMap<K, C>, b: &BTreeMap<K, C>) {
    for k in a.keys().chain(b.keys()) {
        close(
            a.get(k).copied().unwrap_or_default(),
            b.get(k).copied().unwrap_or_default(),
        );
    }
}

// Independent dense tensor-ladder/Jordan-Wigner reference. No sparse transition,
// joint-action function, normal ordering or production compiler is used here.
type Matrix = Vec<Vec<C>>;
fn eye(n: usize) -> Matrix {
    (0..n)
        .map(|i| (0..n).map(|j| c(f64::from(i == j))).collect())
        .collect()
}
fn matvec(a: &Matrix, x: &[C]) -> Vec<C> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}
fn mm(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..n).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}
fn dense_ladder(l: L, boson_sizes: &[usize], nf: usize) -> Matrix {
    let nb = boson_sizes.len();
    let sizes: Vec<_> = boson_sizes.iter().copied().chain(vec![2; nf]).collect();
    let dim: usize = sizes.iter().product();
    let digit = |mut index: usize, mode: usize| {
        for radix in &sizes[..mode] {
            index /= radix;
        }
        index % sizes[mode]
    };
    let (mode, create, fermion) = match l {
        L::BosonCreate(m) => (m as usize, true, false),
        L::BosonAnnihilate(m) => (m as usize, false, false),
        L::FermionCreate(m) => (m as usize, true, true),
        L::FermionAnnihilate(m) => (m as usize, false, true),
    };
    let target = if fermion { nb + mode } else { mode };
    (0..dim)
        .map(|row| {
            (0..dim)
                .map(|col| {
                    let mut value = 1.0;
                    for (k, _) in sizes.iter().enumerate() {
                        let r = digit(row, k);
                        let s = digit(col, k);
                        let local = if k == target {
                            if fermion {
                                f64::from(if create {
                                    r == 1 && s == 0
                                } else {
                                    r == 0 && s == 1
                                })
                            } else if create {
                                if r == s + 1 {
                                    ((s + 1) as f64).sqrt()
                                } else {
                                    0.0
                                }
                            } else if s == r + 1 {
                                (s as f64).sqrt()
                            } else {
                                0.0
                            }
                        } else if r != s {
                            0.0
                        } else if fermion && k >= nb && k < target && s == 1 {
                            -1.0
                        } else {
                            1.0
                        };
                        value *= local;
                    }
                    c(value)
                })
                .collect()
        })
        .collect()
}
fn dense_expr(e: &E, sizes: &[usize], nf: usize) -> Matrix {
    let dim: usize = sizes.iter().product::<usize>() * (1usize << nf);
    match e {
        E::Scalar(s) => eye(dim)
            .iter()
            .map(|r| r.iter().map(|x| x * s).collect())
            .collect(),
        E::Ladder(l) => dense_ladder(*l, sizes, nf),
        E::Sum(terms) => {
            let mut out = vec![vec![c(0.0); dim]; dim];
            for t in terms {
                let a = dense_expr(t, sizes, nf);
                for i in 0..dim {
                    for j in 0..dim {
                        out[i][j] += a[i][j];
                    }
                }
            }
            out
        }
        E::Product(factors) => factors
            .iter()
            .fold(eye(dim), |a, b| mm(&a, &dense_expr(b, sizes, nf))),
    }
}
fn inner_index(b: &B, sizes: &[usize]) -> usize {
    let mut index = 0;
    let mut stride = 1;
    for (m, size) in sizes.iter().enumerate() {
        let n = b.bosons().get(&(m as u32)).copied().unwrap_or(0) as usize;
        assert!(n < *size, "reference padding was insufficient");
        index += stride * n;
        stride *= size;
    }
    for &m in b.fermions() {
        index += stride * (1usize << m);
    }
    index
}
fn compare_column(e: &E, initial: &B, sizes: &[usize], nf: usize) {
    let matrix = dense_expr(e, sizes, nf);
    let col = inner_index(initial, sizes);
    let image = apply(e, &ket(initial.clone()));
    let mut actual = vec![c(0.0); matrix.len()];
    for (b, a) in image {
        actual[inner_index(&b, sizes)] += a;
    }
    for (i, a) in actual.into_iter().enumerate() {
        close(a, matrix[i][col]);
    }
}

#[test]
fn ordered_words_retain_ccr_contractions_and_same_mode_momentum() {
    let ac = product(&[L::BosonAnnihilate(0), L::BosonCreate(0)]);
    for n in 0..5 {
        let v = ket(basis(n, &[]));
        close(apply(&ac, &v)[&basis(n, &[])], c((n + 1) as f64));
        let comm = E::Sum(vec![
            E::Product(vec![E::coordinate(0), E::momentum(0)]),
            E::Product(vec![E::momentum(0), E::coordinate(0)]).scaled(c(-1.0)),
        ]);
        let expected = v.iter().map(|(b, a)| (b.clone(), a * C::i())).collect();
        vectors_close(&apply(&comm, &v), &expected);
        let independent = E::Sum(vec![
            E::Product(vec![E::coordinate(0), E::momentum(1)]),
            E::Product(vec![E::momentum(1), E::coordinate(0)]).scaled(c(-1.0)),
        ]);
        vectors_close(&apply(&independent, &v), &InnerVector::new());
    }
    compare_column(
        &product(&[
            L::BosonAnnihilate(0),
            L::BosonCreate(0),
            L::BosonAnnihilate(0),
        ]),
        &basis(2, &[]),
        &[8],
        0,
    );
}

#[test]
fn quartic_preserves_operator_contractions_and_apply_then_project() {
    let q4 = power(E::coordinate(0), 4);
    let v = ket(B::vacuum());
    let image = apply(&q4, &v);
    close(image[&basis(0, &[])], c(3.0));
    close(image[&basis(2, &[])], c(6.0 * 2f64.sqrt()));
    close(image[&basis(4, &[])], c(24f64.sqrt()));
    compare_column(&q4, &B::vacuum(), &[8], 0);
    let uncut = apply(&q4, &ket(basis(1, &[])));
    close(uncut[&basis(1, &[])], c(15.0));
    let qcut = dense_expr(&E::coordinate(0), &[3], 0);
    let wrong = mm(&mm(&qcut, &qcut), &mm(&qcut, &qcut));
    close(wrong[1][1], c(9.0));
    // A finite compression of the completed action retains 15, not 9.
    assert!((uncut[&basis(1, &[])] - wrong[1][1]).norm() > 5.0);
}

#[test]
fn all_short_two_mode_car_words_match_jordan_wigner() {
    let alphabet = [
        L::FermionCreate(0),
        L::FermionAnnihilate(0),
        L::FermionCreate(1),
        L::FermionAnnihilate(1),
    ];
    for length in 0..=5u32 {
        for mut code in 0..4usize.pow(length) {
            let mut word = Vec::new();
            for _ in 0..length {
                word.push(alphabet[code % 4]);
                code /= 4;
            }
            let e = product(&word);
            let matrix = dense_expr(&e, &[], 2);
            for bits in 0..4 {
                let b = B::new([], (0..2).filter(|m| bits & (1 << m) != 0)).unwrap();
                let mut out = vec![c(0.0); 4];
                for (b, a) in apply(&e, &ket(b)) {
                    out[inner_index(&b, &[])] += a;
                }
                for row in 0..4 {
                    close(out[row], matrix[row][bits]);
                }
            }
        }
    }
}

#[test]
fn mixed_factorized_action_and_adjoint_match_independent_tensor_matrices() {
    let vertex = E::Product(vec![
        E::coordinate(0),
        product(&[L::FermionCreate(0), L::FermionAnnihilate(1)]),
    ])
    .scaled(C::new(0.3, 0.7));
    let e = E::Sum(vec![
        power(E::momentum(0), 2),
        power(E::coordinate(0), 4).scaled(c(0.2)),
        vertex.clone(),
        vertex.adjoint(),
        E::Scalar(c(0.75)),
    ]);
    let reference = dense_expr(&e, &[8], 2);
    for i in 0..reference.len() {
        for j in 0..reference.len() {
            close(reference[i][j], reference[j][i].conj());
        }
    }
    for n in 0..2 {
        for f in [vec![], vec![0], vec![1], vec![0, 1]] {
            compare_column(&e, &basis(n, &f), &[8], 2);
            compare_column(&vertex.adjoint(), &basis(n, &f), &[8], 2);
        }
    }
}

#[test]
fn joint_inner_coupling_is_lifted_once_on_one_particle() {
    let e = E::Product(vec![
        E::coordinate(0),
        product(&[L::FermionCreate(1), L::FermionAnnihilate(0)]),
    ]);
    let input = basis(1, &[0]);
    let image = apply(&e, &ket(input.clone()));
    assert_eq!(image.len(), 2);
    for fermionic in [false, true] {
        let outer = |b| {
            if fermionic {
                O::fermionic([b]).unwrap()
            } else {
                O::bosonic([(b, 1)]).unwrap()
            }
        };
        let actual = lift(&e, &nested(outer(input.clone())));
        let expected = image.iter().map(|(b, a)| (outer(b.clone()), *a)).collect();
        vectors_close(&actual, &expected);
    }
}

#[test]
fn inner_identity_is_outer_number_and_outer_vacuum_is_annihilated() {
    let constant = E::Scalar(C::new(1.2, -0.3));
    let states = [
        O::bosonic([]).unwrap(),
        O::fermionic([]).unwrap(),
        O::bosonic([(basis(0, &[0]), 3), (basis(2, &[1]), 2)]).unwrap(),
        O::fermionic([basis(0, &[0]), basis(2, &[1])]).unwrap(),
    ];
    for b in states {
        let result = lift(&constant, &nested(b.clone()));
        if b.particle_number() == 0 {
            assert!(result.is_empty());
        } else {
            close(result[&b], C::new(1.2, -0.3) * (b.particle_number() as f64));
        }
    }
    let arbitrary = E::Sum(vec![
        power(E::coordinate(0), 4),
        E::Scalar(c(8.0)),
        E::Ladder(L::FermionCreate(0)),
    ]);
    assert!(lift(&arbitrary, &nested(O::bosonic([]).unwrap())).is_empty());
}

#[test]
fn multiparticle_lift_of_product_differs_from_product_of_lifts() {
    let input = nested(O::bosonic([(B::vacuum(), 2)]).unwrap());
    let q = E::coordinate(0);
    let once = lift(&power(q.clone(), 2), &input);
    let twice = lift(&q, &lift(&q, &input));
    let mut difference = twice;
    for (b, a) in &once {
        *difference.entry(b.clone()).or_default() -= a;
    }
    let expected = nested(O::bosonic([(basis(1, &[]), 2)]).unwrap())
        .into_iter()
        .map(|(b, a)| (b, a * 2.0))
        .collect();
    vectors_close(&difference, &expected);
    close(once[&O::bosonic([(basis(0, &[]), 2)]).unwrap()], c(2.0));
    close(
        once[&O::bosonic([(basis(0, &[]), 1), (basis(2, &[]), 1)]).unwrap()],
        c(2.0),
    );
}

#[test]
fn two_outer_particle_actions_match_symmetric_and_exterior_tensor_sums() {
    // Inner one-fermion subspace: |0>, |1>, |2>; complex hopping 0 <-> 2.
    // Ordering spectators tests the exterior sign, not just Pauli exclusion.
    let t = product(&[L::FermionCreate(2), L::FermionAnnihilate(0)]).scaled(C::new(0.2, 0.6));
    let e = E::Sum(vec![t.clone(), t.adjoint(), E::Scalar(c(0.4))]);
    let bs: Vec<_> = (0..3).map(|m| basis(0, &[m])).collect();
    for fermionic in [false, true] {
        let mut pairs = Vec::new();
        for i in 0..3 {
            for j in i..3 {
                if !fermionic || i != j {
                    pairs.push((i, j));
                }
            }
        }
        for &(i, j) in &pairs {
            let make = |i: usize, j: usize| {
                if fermionic {
                    O::fermionic([bs[i].clone(), bs[j].clone()]).unwrap()
                } else {
                    O::bosonic([(bs[i].clone(), 1), (bs[j].clone(), 1)]).unwrap()
                }
            };
            let image = lift(&e, &nested(make(i, j)));
            // Independently embed a normalized symmetric/wedge vector in K⊗K.
            let embed = |i: usize, j: usize| {
                let mut v = vec![c(0.0); 9];
                if i == j {
                    v[3 * i + j] = c(1.0);
                } else {
                    v[3 * i + j] = c(1.0 / 2f64.sqrt());
                    v[3 * j + i] = c(if fermionic {
                        -1.0 / 2f64.sqrt()
                    } else {
                        1.0 / 2f64.sqrt()
                    });
                }
                v
            };
            let mut h = vec![vec![c(0.0); 3]; 3];
            for (k, row) in h.iter_mut().enumerate() {
                row[k] = c(0.4);
            }
            h[2][0] = C::new(0.2, 0.6);
            h[0][2] = C::new(0.2, -0.6);
            let mut tensor = vec![vec![c(0.0); 9]; 9];
            for a in 0..3 {
                for b in 0..3 {
                    for r in 0..3 {
                        for s in 0..3 {
                            tensor[3 * a + b][3 * r + s] =
                                h[a][r] * f64::from(b == s) + h[b][s] * f64::from(a == r);
                        }
                    }
                }
            }
            let out = matvec(&tensor, &embed(i, j));
            for &(r, s) in &pairs {
                let expected: C = embed(r, s)
                    .iter()
                    .zip(&out)
                    .map(|(a, b)| a.conj() * b)
                    .sum();
                close(
                    image.get(&make(r, s)).copied().unwrap_or_default(),
                    expected,
                );
            }
            assert!(
                image
                    .keys()
                    .all(|b| pairs.iter().any(|&(r, s)| make(r, s) == *b))
            );
        }
    }
}

#[test]
fn limits_fail_instead_of_pruning_and_small_amplitudes_are_preserved() {
    let v = ket(basis(1, &[]));
    assert_eq!(
        E::coordinate(0).apply(
            &v,
            ActionLimits {
                max_components: 1,
                max_work: 100
            }
        ),
        Err(ActionError::ComponentLimit { limit: 1 })
    );
    assert_eq!(
        E::coordinate(0).apply(
            &v,
            ActionLimits {
                max_components: 100,
                max_work: 1
            }
        ),
        Err(ActionError::WorkLimit { limit: 1 })
    );
    let tiny = E::Scalar(c(1e-30));
    assert_eq!(apply(&tiny, &v)[&basis(1, &[])], c(1e-30));
    assert_eq!(
        E::Scalar(c(f64::NAN)).apply(&v, ActionLimits::default()),
        Err(ActionError::NonFiniteAmplitude)
    );
    let huge = ket(B::new([(0, u32::MAX)], []).unwrap());
    assert_eq!(
        E::Ladder(L::BosonCreate(0)).apply(&huge, ActionLimits::default()),
        Err(ActionError::OccupationOverflow)
    );
    assert_eq!(B::new([], [0, 0]), Err(ActionError::PauliOccupation));
    assert_eq!(
        O::fermionic([B::vacuum(), B::vacuum()]),
        Err(ActionError::PauliOccupation)
    );
}

#[test]
fn legacy_ladder_products_include_cross_particle_terms_and_outer_scalars() {
    use nested_fock_algebra::{Hamiltonian, InnerBosonicState, Operator, OuterState, QuantumState};
    let vacuum = InnerBosonicState::vacuum();
    let mut outer = OuterState::vacuum();
    outer.bosonic.insert(vacuum, 2);
    let input = QuantumState {
        components: [(outer.clone(), c(1.0))].into_iter().collect(),
    };
    let letters = [
        Operator::InnerBosonCreate(0),
        Operator::InnerBosonAnnihilate(0),
    ];
    let legacy_square = Hamiltonian {
        terms: letters
            .iter()
            .flat_map(|a| {
                letters
                    .iter()
                    .map(move |b| (c(1.0), vec![a.clone(), b.clone()]))
            })
            .collect(),
    };
    let legacy_image = legacy_square.apply(&input);
    let mut excited = InnerBosonicState::vacuum();
    excited.modes.insert(0, 1);
    let mut two_excited = OuterState::vacuum();
    two_excited.bosonic.insert(excited, 2);
    close(legacy_image.components[&two_excited], c(2.0));
    // Completed inner q^2, then enclosed once, has no such |1>⊗|1> term.
    let correct = lift(
        &power(E::coordinate(0), 2),
        &nested(O::bosonic([(B::vacuum(), 2)]).unwrap()),
    );
    assert!(!correct.contains_key(&O::bosonic([(basis(1, &[]), 2)]).unwrap()));
    let legacy_scalar = Hamiltonian {
        terms: vec![(c(1.0), vec![])],
    };
    close(
        legacy_scalar.apply(&QuantumState::vacuum()).components[&OuterState::vacuum()],
        c(1.0),
    );
    assert!(lift(&E::Scalar(c(1.0)), &nested(O::bosonic([]).unwrap())).is_empty());
}

#[test]
fn graded_outer_statistics_reject_odd_duplicates_and_keep_even_multiplicity() {
    let even = basis(0, &[]);
    let odd = basis(0, &[0]);
    assert_eq!(
        O::graded([(odd.clone(), 2)]),
        Err(ActionError::PauliOccupation)
    );
    assert_eq!(
        O::graded([(odd.clone(), 1), (odd, 1)]),
        Err(ActionError::PauliOccupation)
    );
    let mixed = O::graded([(even.clone(), 2), (basis(0, &[1]), 1)]).unwrap();
    assert_eq!(mixed.particle_number(), 3);
    let out = lift(&E::Scalar(c(0.7)), &nested(mixed.clone()));
    close(out[&mixed], c(2.1));
    assert!(lift(&power(E::coordinate(0), 4), &nested(O::graded([]).unwrap())).is_empty());
}

#[test]
fn graded_mixed_sector_has_the_odd_spectator_sign() {
    let e = basis(0, &[]);
    let a = basis(0, &[0]);
    let b = basis(0, &[1]);
    let d = basis(0, &[2]);
    let source = O::graded([(e.clone(), 2), (a, 1), (b.clone(), 1)]).unwrap();
    let hop = product(&[L::FermionCreate(2), L::FermionAnnihilate(0)]);
    let target = O::graded([(e, 2), (b, 1), (d, 1)]).unwrap();
    let image = lift(&hop, &nested(source));
    assert_eq!(image.len(), 1);
    close(image[&target], c(-1.0));
}

#[test]
fn graded_one_particle_fast_path_preserves_its_statistics_tag() {
    let even = basis(0, &[]);
    let odd = basis(0, &[0]);
    let create = E::Ladder(L::FermionCreate(0));
    let from = O::graded([(even.clone(), 1)]).unwrap();
    let to = O::graded([(odd.clone(), 1)]).unwrap();
    vectors_close(&lift(&create, &nested(from)), &nested(to));
    let pair = O::graded([(even.clone(), 2)]).unwrap();
    let mixed = O::graded([(even, 1), (odd, 1)]).unwrap();
    let out = lift(&create, &nested(pair));
    assert_eq!(out.len(), 1);
    close(out[&mixed], c(2f64.sqrt()));
}
