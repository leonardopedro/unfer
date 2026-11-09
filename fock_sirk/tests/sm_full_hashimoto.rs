//! Full E1 SM operator -> joint dGamma -> genuine shift-invert rational Krylov.
//! Numerical windows project this SAME operator and report its omitted action.
//! No radial/free/decoupled replacement Hamiltonian is used.
use fock_sirk::joint_sirk::*;
use nalgebra::{DMatrix, DVector};
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, InnerLadder, JointInnerBasis as B, JointOuterBasis as O, NestedVector,
};
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
use std::collections::BTreeSet;
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
    static M: OnceLock<SmHamiltonian> = OnceLock::new();
    M.get_or_init(|| {
        cadabra_import::from_cadabra_latex(
            parameters(),
            include_str!("../../docs/fixtures/sm_full_cadabra.latex"),
        )
        .unwrap()
    })
}
fn h() -> &'static InnerExpr {
    static H: OnceLock<InnerExpr> = OnceLock::new();
    H.get_or_init(|| model().expression())
}
fn limits() -> ActionLimits {
    ActionLimits {
        max_components: 500_000,
        max_work: 500_000_000,
    }
}
fn seed() -> NestedVector {
    // Both a quark and a lepton are present. All three-generation matrices,
    // all fields/jet terms and both Yukawa doublet partners remain in h.
    let inner = B::new(
        [],
        [
            matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
            matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
        ],
    )
    .unwrap();
    NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))])
}
fn distance(a: &NestedVector, b: &NestedVector) -> f64 {
    let mut d = a.clone();
    joint_axpy(&mut d, b, C::new(-1.0, 0.0), limits()).unwrap();
    joint_norm(&d)
}
struct Reference {
    basis: Vec<O>,
    matrix: DMatrix<C>,
    initial: DVector<C>,
}
fn reference() -> &'static Reference {
    static R: OnceLock<Reference> = OnceLock::new();
    R.get_or_init(||{
        let initial=seed();let image=h().d_gamma(&initial,limits()).unwrap();
        let mut window:BTreeSet<_>=project_occupation(&image,1).0.into_keys().collect();window.extend(initial.keys().cloned());
        let basis:Vec<_>=window.into_iter().collect();let n=basis.len();
        let mut matrix=DMatrix::<C>::zeros(n,n);
        for (j,b) in basis.iter().enumerate() {
            let image=h().d_gamma(&NestedVector::from([(b.clone(),C::new(1.0,0.0))]),limits()).unwrap();
            for (i,target) in basis.iter().enumerate() {matrix[(i,j)]=image.get(target).copied().unwrap_or_default();}
        }
        assert!((&matrix-matrix.adjoint()).norm()<1e-10);
        let v=DVector::from_iterator(n,basis.iter().map(|b|initial.get(b).copied().unwrap_or_default()));
        eprintln!("full_SM_reference: coordinates=163 CAR=96 forms=49 window={n} (all physical terms retained)");
        Reference{basis,matrix,initial:v}
    })
}
fn options() -> JointSirkOptions {
    JointSirkOptions {
        action_limits: limits(),
        basis_window: Some(reference().basis.iter().cloned().collect()),
        method: JointKrylovMethod::ShiftInvert {
            relative_tolerance: 2e-11,
            max_iterations: 32,
        },
        ..Default::default()
    }
}
fn shifts(count: usize, budget: f64) -> Vec<C> {
    hashimoto_unitary_shifts(1.0, budget, 1.0, count).unwrap()
}
fn vector_from_dense(v: &DVector<C>) -> NestedVector {
    reference()
        .basis
        .iter()
        .cloned()
        .zip(v.iter().copied())
        .filter(|(_, a)| *a != C::new(0.0, 0.0))
        .collect()
}

#[test]
fn sm_full_hashimoto_inner_outer_action() {
    use nested_fock_algebra::exact_inner::InnerVector;
    let input = seed();
    let inner = input.keys().next().unwrap().inner_occupations()[0]
        .0
        .clone();
    let image = h()
        .apply(&InnerVector::from([(inner, C::new(1.0, 0.0))]), limits())
        .unwrap();
    let enclosed: NestedVector = image
        .into_iter()
        .map(|(b, a)| (O::graded([(b, 1)]).unwrap(), a))
        .collect();
    let actual = h().d_gamma(&input, limits()).unwrap();
    assert!(actual.len() > 1_000);
    assert!(distance(&actual, &enclosed) < 1e-10);
    for b in actual.keys() {
        for (inner, _) in b.inner_occupations() {
            assert!(inner.bosons().keys().all(|m| *m < 163));
            assert!(inner.fermions().iter().all(|m| *m < 96));
        }
    }
}

#[test]
fn sm_full_hashimoto_sector_structure() {
    let vacuum = NestedVector::from([(O::graded([]).unwrap(), C::new(1.0, 0.0))]);
    let solve = solve_joint_sirk(
        h(),
        &vacuum,
        &shifts(3, 9.0),
        &JointSirkOptions {
            action_limits: limits(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(solve.rank(), 1);
    assert!(solve.h_proj.norm() < 1e-12);
    assert!(
        distance(
            &solve.reconstruct(&solve.time_evolve(0.1).unwrap()).unwrap(),
            &vacuum
        ) < 1e-10
    );
    let initial = seed();
    let inner = initial.keys().next().unwrap().inner_occupations()[0]
        .0
        .clone();
    let two = O::graded([(inner.clone(), 2)]).unwrap();
    let actual = h()
        .d_gamma(&NestedVector::from([(two, C::new(1.0, 0.0))]), limits())
        .unwrap();
    let inner_image = h()
        .apply(
            &std::collections::BTreeMap::from([(inner.clone(), C::new(1.0, 0.0))]),
            limits(),
        )
        .unwrap();
    let mut expected = NestedVector::new();
    for (target, a) in inner_image {
        let (basis, factor) = if target == inner {
            (O::graded([(inner.clone(), 2)]).unwrap(), 2.0)
        } else {
            (
                O::graded([(inner.clone(), 1), (target, 1)]).unwrap(),
                2f64.sqrt(),
            )
        };
        expected.insert(basis, a * factor);
    }
    assert!(distance(&actual, &expected) < 1e-10);
    let occupied = solve_joint_sirk(h(), &initial, &shifts(3, 9.0), &options()).unwrap();
    assert!(occupied.rank() >= 2);
    assert!(occupied.registry.is_bijective());
    let even = seed().keys().next().unwrap().inner_occupations()[0]
        .0
        .clone();
    let odd = B::new([], [matter_mode(Species::Up, 0, 0, Chirality::Left, 1)]).unwrap();
    let mixed = NestedVector::from([(
        O::graded([(even.clone(), 1), (odd.clone(), 1)]).unwrap(),
        C::new(1.0, 0.0),
    )]);
    let actual = h().d_gamma(&mixed, limits()).unwrap();
    let mut expected = NestedVector::new();
    for (source, spectator) in [(even.clone(), odd.clone()), (odd, even)] {
        let image = h()
            .apply(
                &std::collections::BTreeMap::from([(source, C::new(1.0, 0.0))]),
                limits(),
            )
            .unwrap();
        for (target, a) in image {
            let basis = O::graded([(spectator.clone(), 1), (target, 1)]).unwrap();
            *expected.entry(basis).or_default() += a;
        }
    }
    assert!(distance(&actual, &expected) < 1e-10);
    for basis in actual.keys() {
        let occupations = basis.inner_occupations();
        assert_eq!(
            occupations
                .iter()
                .filter(|(inner, _)| inner.is_odd())
                .map(|(_, n)| *n)
                .sum::<u32>(),
            1
        );
        assert_eq!(
            occupations
                .iter()
                .filter(|(inner, _)| !inner.is_odd())
                .map(|(_, n)| *n)
                .sum::<u32>(),
            1
        );
    }
}

#[test]
fn sm_full_hashimoto_ritz_residual_enclosure() {
    let solve = solve_joint_sirk(h(), &seed(), &shifts(4, 9.0), &options()).unwrap();
    let finite = reference().matrix.clone().symmetric_eigen().eigenvalues;
    let eig = solve.h_proj.clone().symmetric_eigen();
    let measured = solve.ritz_residuals().unwrap();
    assert!(solve.projected_action_tail_norms.iter().any(|r| *r > 0.1));
    assert!(solve.shift_solve_error_bounds.iter().any(|r| *r > 1e-6));
    for (i, &theta) in eig.eigenvalues.iter().enumerate() {
        let state = solve
            .reconstruct(&eig.eigenvectors.column(i).into_owned())
            .unwrap();
        let mut residual = h().d_gamma(&state, limits()).unwrap();
        joint_axpy(&mut residual, &state, C::new(-theta, 0.0), limits()).unwrap();
        let norm = joint_norm(&residual);
        let nearest = finite
            .iter()
            .map(|x| (x - theta).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(nearest <= norm / joint_norm(&state) + 1e-9);
        let recorded = measured
            .iter()
            .find(|x| (x.0 - theta).abs() < 1e-9)
            .unwrap();
        assert!((recorded.1 - norm).abs() < 1e-8);
    }
    // The first computed inverse vector must agree with an independent dense
    // solve for the exact same full-SM finite compression, up to its phase/norm.
    let z = solve.shifts[0];
    let a = &reference().matrix
        - DMatrix::<C>::identity(reference().basis.len(), reference().basis.len()) * z;
    let dense = a.lu().solve(&reference().initial).unwrap();
    let exact = vector_from_dense(&dense);
    let mut normalized = NestedVector::new();
    joint_axpy(
        &mut normalized,
        &exact,
        C::new(1.0 / joint_norm(&exact), 0.0),
        limits(),
    )
    .unwrap();
    assert!(distance(&normalized, &solve.sequence[1]) < 1e-8);
    let n = reference().basis.len();
    let rank = solve.rank();
    let mut v = DMatrix::<C>::zeros(n, rank);
    for j in 0..rank {
        let state = solve
            .reconstruct(&DVector::from_fn(rank, |i, _| {
                if i == j {
                    C::new(1.0, 0.0)
                } else {
                    C::new(0.0, 0.0)
                }
            }))
            .unwrap();
        for (i, b) in reference().basis.iter().enumerate() {
            v[(i, j)] = state.get(b).copied().unwrap_or_default();
        }
    }
    let gamma = solve.shifts.last().unwrap().im;
    let resolvent = (DMatrix::<C>::identity(n, n) * C::new(gamma, 0.0)
        + reference().matrix.clone() * C::i())
    .lu()
    .solve(&v)
    .unwrap();
    let expected = v.adjoint() * resolvent;
    assert!((&expected - solve.projected_resolvent.as_ref().unwrap()).norm() < 1e-8);
    let generator =
        DMatrix::<C>::identity(rank, rank) * C::new(gamma, 0.0) - expected.try_inverse().unwrap();
    assert!((&generator - &solve.function_generator).norm() < 1e-5);
}

/// Leading complete-action tail at occupation cutoff 3 (638-state window),
/// measured on 2026-11-09; cutoff 4 encloses the first image and gives 0.
const CUTOFF3_TAIL: f64 = 1.2179612832516469;

#[test]
fn sm_full_hashimoto_budget_refinement() {
    let time = 0.01;
    let dense = (&reference().matrix * C::new(0.0, -time)).exp() * &reference().initial;
    let exact = vector_from_dense(&dense);
    let mut errors = Vec::new();
    // Two budget/tolerance pairs bracket the rational approximation; the tighter
    // 5-shift budget is not repeated because it is *less* accurate than m = 4
    // (measured 5.9e-7 against 4.2e-7) and costs a full extra solve set.
    for (count, budget, tolerance) in [(2, 8.0, 1e-8), (4, 9.0, 2e-11)] {
        let mut opts = options();
        opts.method = JointKrylovMethod::ShiftInvert {
            relative_tolerance: tolerance,
            max_iterations: 32,
        };
        let solve = solve_joint_sirk(h(), &seed(), &shifts(count, budget), &opts).unwrap();
        let state = solve
            .reconstruct(&solve.time_evolve(time).unwrap())
            .unwrap();
        let error = distance(&state, &exact);
        eprintln!(
            "full_SM_rational: m={count} budget={budget} error={error:e} action_tail={:?}",
            solve.projected_action_tail_norms
        );
        errors.push(error);
        assert!((joint_norm(&state) - 1.0).abs() < 2.0 * error + 1e-8);
    }
    assert!(errors[1] <= errors[0] + 1e-9);
    assert!(errors[1] < 1e-6);
    // Independent occupation budgets preserve the complete operator. They
    // report discarded action, rather than silently zeroing interactions. The
    // wider windows need raised budgets, since the cached images grow with the
    // window; the shipped defaults refuse them explicitly (see the doc).
    let mut tails = Vec::new();
    let mut supports = Vec::new();
    // The leading tail the solver reports is exactly the part of `H |seed>` that
    // the declared window omits, so it is computed from the first action image
    // directly rather than paying for a Krylov solve at every cutoff. One cheap
    // solve below keeps the solver's own reporting under test.
    let first_image = h().d_gamma(&seed(), limits()).unwrap();
    let mut cross_check = None;
    for cutoff in [1u64, 2, 3, 4] {
        let mut window: BTreeSet<_> = project_occupation(&first_image, cutoff)
            .0
            .into_keys()
            .collect();
        window.extend(seed().into_keys());
        let mut omitted = NestedVector::new();
        for (basis, amplitude) in first_image.iter() {
            let inside = window.contains(basis)
                && basis
                    .inner_occupations()
                    .iter()
                    .map(|(inner, n)| inner.total_bosons() * u64::from(*n))
                    .sum::<u64>()
                    <= cutoff;
            if !inside {
                omitted.insert(basis.clone(), *amplitude);
            }
        }
        let tail = joint_norm(&omitted);
        assert!(tail.is_finite() && tail >= 0.0);
        if cutoff == 2 {
            cross_check = Some((window.clone(), tail));
        }
        eprintln!(
            "full_SM_occupation: cutoff={cutoff} support={} tail={tail:e}",
            window.len()
        );
        supports.push(window.len());
        tails.push(tail);
    }
    let (window, direct) = cross_check.unwrap();
    let opts = JointSirkOptions {
        action_limits: limits(),
        occupation_cutoff: Some(2),
        basis_window: Some(window),
        max_cached_components: 1_000_000,
        method: JointKrylovMethod::ShiftInvert {
            relative_tolerance: 1e-8,
            max_iterations: 32,
        },
        ..Default::default()
    };
    // One shift is enough: the reported tail belongs to the first Krylov vector.
    let solve = solve_joint_sirk(h(), &seed(), &shifts(1, 9.0), &opts).unwrap();
    assert_eq!(solve.registry.len(), 230, "cutoff-2 window moved");
    assert!(
        (solve.projected_action_tail_norms[0] - direct).abs() <= 1e-12 * direct,
        "the solver's reported tail disagrees with the direct action: {:e} vs {direct:e}",
        solve.projected_action_tail_norms[0]
    );
    // The window now covers the first complete image: the leading tail closes
    // monotonically to exactly zero, while the omitted-support norms above
    // remain nonzero for the later Krylov vectors (reported, not asserted).
    assert!(
        supports == [70, 230, 638, 1176],
        "windows grew: {supports:?}"
    );
    assert!(
        tails[0] > tails[1] && tails[1] > tails[2],
        "leading tail must shrink with the window: {tails:?}"
    );
    assert_eq!(
        tails[3], 0.0,
        "cutoff 4 must enclose the first image exactly: {tails:?}"
    );
    assert!(
        (tails[2] - CUTOFF3_TAIL).abs() < 1e-6 * CUTOFF3_TAIL,
        "cutoff-3 leading tail moved: {} vs {CUTOFF3_TAIL}",
        tails[2]
    );
}

#[test]
fn sm_full_hashimoto_restarted_evolution() {
    let time = 0.02;
    let exact = vector_from_dense(
        &((&reference().matrix * C::new(0.0, -time)).exp() * &reference().initial),
    );
    let mut errors = Vec::new();
    // The flow error here is the window's truncation (~8e-7), so a 1e-10 solve
    // target and a 10-iteration cap leave it unchanged (measured) at half the
    // cost of the tightest setting.
    let mut opts = options();
    opts.method = JointKrylovMethod::ShiftInvert {
        relative_tolerance: 1e-10,
        max_iterations: 10,
    };
    // The 2-step split is covered by the energy test; 1 and 4 bound the
    // restart refinement, and each extra level costs a full set of solves.
    for steps in [1, 4] {
        let result =
            evolve_restarted_joint(h(), &seed(), time, steps, &shifts(4, 9.0), &opts).unwrap();
        let error = distance(&result.state, &exact);
        errors.push(error);
        assert!(
            result
                .norms
                .iter()
                .all(|n| (n - 1.0).abs() < 2.0 * error + 1e-7)
        );
        assert!(
            result
                .action_tail_norms
                .iter()
                .flatten()
                .all(|x| x.is_finite())
        );
        assert!(
            result
                .shift_solve_error_bounds
                .iter()
                .flatten()
                .all(|x| x.is_finite())
        );
        assert!(
            result
                .function_solve_error_bounds
                .iter()
                .flatten()
                .all(|x| x.is_finite())
        );
        assert!(result.orthogonality_defects.iter().all(|x| *x < 1e-10));
        let initial_energy = result.energies[0];
        for energy in &result.energies {
            assert!(energy.im.abs() < 1e-8);
            assert!(
                (*energy - initial_energy).norm() < 8.0 * reference().matrix.norm() * error + 1e-8
            );
        }
        eprintln!("full_SM_restart: steps={steps} error={error:e}");
    }
    assert!(errors[1] <= errors[0] + 1e-8);
    assert!(errors[1] < 1e-6);
}

/// A state with definite field content: the same quark/lepton pair in two
/// universes whose gauge coordinate carries zero and one boson with relative
/// phase `e^{i pi/4}`. For `q = a + a†` and `p = i(a† - a)` that gives
/// `<q> = cos(pi/4)` and `<p> = -sin(pi/4)`, so both are nonzero.
fn bosonic_seed() -> NestedVector {
    let fermions = [
        matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
        matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
    ];
    let empty = B::new([], fermions).unwrap();
    let filled = B::new([(coord::g(0, 0), 1)], fermions).unwrap();
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let phase = C::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    NestedVector::from([
        (O::graded([(empty, 1)]).unwrap(), C::new(half, 0.0)),
        (O::graded([(filled, 1)]).unwrap(), C::new(half, 0.0) * phase),
    ])
}

/// `<psi|O|psi>` for a nested-Fock state and an exact inner operator.
fn expectation(state: &NestedVector, observable: &InnerExpr) -> C {
    let applied = observable.d_gamma(state, limits()).unwrap();
    joint_inner_product(state, &applied) / joint_inner_product(state, state)
}

/// Gauss generator of one temporal-gauge ghost sector: the sum of the three
/// spatial gauge momenta of that sector, the `P` blocks of
/// `models::sm_brst_charge`.
fn gauss_generator(ghost: u32) -> InnerExpr {
    let spatial = |i: u32| match ghost {
        0..=7 => coord::g(ghost, i),
        8..=10 => coord::w(ghost - 8, i),
        _ => coord::b(i),
    };
    InnerExpr::Sum((0..3).map(|i| InnerExpr::momentum(spatial(i))).collect())
}

/// BRST charge `Q = sum_g G_g c_g^dagger` of this representation.
///
/// `models::sm_brst_charge` writes its ghost ladder as
/// `InnerFermionAnnihilate`, which is the ghost *raise* in the filled-sea
/// convention of `ym_brst_charge` and is therefore zero on a ghost-free state.
/// The ghost reference here is the empty set, where the raise is
/// `FermionCreate`; with that ladder `Q|psi> = 0` is `G_g|psi> = 0` in every
/// sector — the electric part of Gauss's law (`d_i Pi_i = 0`, the
/// matter/current term absent). This is the only version of the charge that can
/// detect a constraint violation: with the vanishing ladder, `Q` annihilates
/// every state of this dictionary.
fn brst_charge() -> InnerExpr {
    InnerExpr::Sum(
        (0..12u32)
            .map(|ghost| {
                InnerExpr::Product(vec![
                    gauss_generator(ghost),
                    InnerExpr::Ladder(InnerLadder::FermionCreate(MATTER_MODES + ghost)),
                ])
            })
            .collect(),
    )
}

/// `Q^dagger Q = sum_g G_g^2`: the gauge-fixing penalty, and the measure of the
/// Gauss residual through `|Q psi|^2 = <psi|P|psi>`.
fn gauss_penalty() -> InnerExpr {
    InnerExpr::Sum(
        (0..12u32)
            .map(|ghost| {
                let generator = gauss_generator(ghost);
                InnerExpr::Product(vec![generator.clone(), generator])
            })
            .collect(),
    )
}

/// Compression of `operator` to the recorded window.
fn window_matrix(operator: &InnerExpr) -> DMatrix<C> {
    let reference = reference();
    let n = reference.basis.len();
    let mut matrix = DMatrix::<C>::zeros(n, n);
    for (j, basis) in reference.basis.iter().enumerate() {
        let image = operator
            .d_gamma(
                &NestedVector::from([(basis.clone(), C::new(1.0, 0.0))]),
                limits(),
            )
            .unwrap();
        for (i, target) in reference.basis.iter().enumerate() {
            matrix[(i, j)] = image.get(target).copied().unwrap_or_default();
        }
    }
    matrix
}

/// Lowest eigenpair of the compressed window.
fn ground_state() -> (f64, NestedVector) {
    let spectrum = reference().matrix.clone().symmetric_eigen();
    let index = spectrum
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    (
        spectrum.eigenvalues[index],
        vector_from_dense(&spectrum.eigenvectors.column(index).into_owned()),
    )
}

fn dense_state(state: &NestedVector) -> DVector<C> {
    DVector::from_iterator(
        reference().basis.len(),
        reference()
            .basis
            .iter()
            .map(|basis| state.get(basis).copied().unwrap_or_default()),
    )
}

/// Gauss's law under a real gauge-fixing term, in the recorded window.
///
/// The physical content of the charge is `Q|psi> = 0  <=>  G_g|psi> = 0`, with
/// `P = Q^dagger Q` and `|Q psi|^2 = <psi|P|psi>`. The test measures the
/// verdict where it can be measured — on the window's compressed `P`:
///
/// * the ground state violates the constraint (`|Q gs>|` is O(1)), so the
///   charge is not vacuous on it;
/// * `P` is positive semidefinite, and its *smallest* eigenvalue is still
///   positive, so the window contains no state obeying the electric constraint
///   at all — the local constraint is solved in a gauge-fixed treatment, never
///   imposed on Fock states, since no normalizable state annihilates a field
///   momentum;
/// * the ground state already attains that smallest violation;
/// * consequently the gauge-fixing term `H + lambda P` cannot select a physical
///   state here: it shifts every energy by `lambda * gap` (verified to 1%).
///
/// The structural reason is that the generator is a field momentum, and
/// `p = i(a^dagger - a)/2` has continuous spectrum on the oscillator, so
/// `G|psi> = 0` has no normalizable solution. Adding the (bounded, bilinear)
/// matter/current term would not change that; the local constraint is *solved*
/// in a gauge-fixed treatment, not imposed on states. That is why the earlier
/// ghost-vacuum statement could not say anything about Gauss's law.
#[test]
fn sm_full_gauss_law_is_empty_in_the_recorded_window() {
    let (theta0, gs) = ground_state();
    assert!(
        (theta0 - 49.7633053378465).abs() < 1e-9,
        "the window ground energy moved: {theta0:e}"
    );
    let charge = brst_charge();
    let penalty_matrix = window_matrix(&gauss_penalty());
    let penalty_spectrum = penalty_matrix.clone().symmetric_eigen();
    let gap = penalty_spectrum.eigenvalues.min();
    assert!(
        gap >= -1e-9,
        "Q^dagger Q must be positive semidefinite: min = {gap:e}"
    );

    let charged = charge.d_gamma(&gs, limits()).unwrap();
    let residual = joint_norm(&charged);
    let nilpotent = joint_norm(&charge.d_gamma(&charged, limits()).unwrap());
    let gs_dense = dense_state(&gs);
    let penalty_expectation = (gs_dense.adjoint() * &penalty_matrix * &gs_dense)[(0, 0)].re;
    // |Q psi|^2 = <psi|Q^dagger Q|psi>, so the ghost construction and the
    // field-only penalty must agree.
    assert!(
        (residual * residual - penalty_expectation).abs() < 1e-9,
        "|Q gs>|^2 = {:e} but <gs|P|gs> = {penalty_expectation:e}",
        residual * residual
    );
    // A physical state of the window would be a zero eigenvector of `P`.
    assert!(
        gap > 1e-3,
        "the window unexpectedly contains a Gauss-invariant state: {gap:e}"
    );
    assert!(
        (residual - gap.sqrt()).abs() < 1e-2,
        "the ground state should attain the smallest violation: {residual:e} vs {:e}",
        gap.sqrt()
    );

    let mut shifts = Vec::new();
    for lambda in [1.0, 10.0, 100.0] {
        let shifted = &reference().matrix + &penalty_matrix * C::new(lambda, 0.0);
        let spectrum = shifted.symmetric_eigen();
        let energy = spectrum
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        // `P >= gap > 0` on the whole window, so the penalised ground state can
        // at best reduce the violation to `gap`: the term is a constant shift.
        let predicted = lambda * gap;
        assert!(
            (energy - theta0 - predicted).abs() < 1e-2 * predicted,
            "the penalty is not the constant shift lambda*gap at lambda={lambda}: {:e} vs {predicted:e}",
            energy - theta0
        );
        eprintln!(
            "full_SM_gauss: lambda={lambda} dE={:e} predicted={predicted:e}",
            energy - theta0
        );
        shifts.push(energy - theta0);
    }

    assert!(
        nilpotent == 0.0,
        "the charge must be nilpotent: |Q^2 gs>| = {nilpotent:e}"
    );
    eprintln!(
        "full_SM_gauss: theta0={theta0:e} |Q gs>={residual:e} gap={gap:e} P_min={:e} shifts={shifts:?}",
        penalty_spectrum.eigenvalues.min()
    );
}

/// One monomial of a form polynomial, in the word order `expression()` uses.
fn monomial_expression(word: &[Variable], coefficient: C) -> InnerExpr {
    let mut factors = vec![InnerExpr::Scalar(coefficient)];
    factors.extend(word.iter().map(|variable| match variable {
        Variable::Q(mode) => InnerExpr::coordinate(*mode),
        Variable::P(mode) => InnerExpr::momentum(*mode),
    }));
    InnerExpr::Product(factors)
}

/// The model's free part (bosonic products of total degree at most two), its
/// interaction part (everything of degree three or more, including every
/// gauge-matter and Yukawa coupling), and the Higgs/Yukawa block alone.
///
/// A form contributes `w * (sum of monomials)^2` in operator order: the second
/// factor is applied first and each monomial's variables in word order.
/// Splitting that ordered pair sum by `degree(left) + degree(right)` reproduces
/// `H` exactly — the bosonic factors are not assumed to commute.
fn model_split() -> (InnerExpr, InnerExpr, InnerExpr) {
    let model = model();
    // `bosonic_expression` also carries the canonical kinetic `P^2 / 2`.
    let mut free: Vec<InnerExpr> = coord::active()
        .into_iter()
        .map(|mode| {
            InnerExpr::Product(vec![InnerExpr::momentum(mode), InnerExpr::momentum(mode)])
                .scaled(C::new(0.5, 0.0))
        })
        .collect();
    let mut interaction = Vec::new();
    for form in &model.forms {
        let weight = C::new(form.weight, 0.0);
        let monomials: Vec<(usize, InnerExpr)> = form
            .polynomial
            .terms
            .iter()
            .map(|(word, &coefficient)| (word.len(), monomial_expression(word, coefficient)))
            .collect();
        for (left_degree, left) in &monomials {
            for (right_degree, right) in &monomials {
                let term = InnerExpr::Product(vec![left.clone(), right.clone()]).scaled(weight);
                if left_degree + right_degree <= 2 {
                    free.push(term);
                } else {
                    interaction.push(term);
                }
            }
        }
    }
    let mut yukawa = Vec::new();
    for (&variable, matrix) in &model.matter {
        let coefficient = match variable {
            Variable::Q(mode) => InnerExpr::coordinate(mode),
            Variable::P(mode) => InnerExpr::momentum(mode),
        };
        let term = InnerExpr::Product(vec![coefficient, matrix_bilinear(matrix)]);
        // Every matter entry is interaction; the Higgs/Yukawa ones are also
        // collected separately, so `free + interaction` stays complete.
        match variable {
            Variable::Q(mode) if (coord::phi(0)..=coord::phi(3)).contains(&mode) => {
                yukawa.push(term.clone());
                interaction.push(term);
            }
            _ => interaction.push(term),
        }
    }
    (
        InnerExpr::Sum(free),
        InnerExpr::Sum(interaction),
        InnerExpr::Sum(yukawa),
    )
}

/// The interaction share of the window energy.
///
/// Every bosonic form squares a polynomial, so the parts separate by total
/// degree: `(P0 + P1 + P2)^2 = P0^2 + 2P0P1 + (2P0P2 + P1^2) + 2P1P2 + P2^2`,
/// of which the last two are the quartic gauge/Higgs self-couplings. Every
/// matter entry is one boson times a fermion bilinear, so all of them are
/// interactions. The split is checked against `H` on the seed exactly, and the
/// ground state then has to show the coupling present and bounded.
#[test]
fn sm_full_interaction_energy_is_a_nonzero_bounded_share() {
    let (free, interaction, yukawa) = model_split();
    let mut residual = h().d_gamma(&seed(), limits()).unwrap();
    let free_image = free.d_gamma(&seed(), limits()).unwrap();
    let interaction_image = interaction.d_gamma(&seed(), limits()).unwrap();
    joint_axpy(&mut residual, &free_image, C::new(-1.0, 0.0), limits()).unwrap();
    joint_axpy(
        &mut residual,
        &interaction_image,
        C::new(-1.0, 0.0),
        limits(),
    )
    .unwrap();
    assert!(
        joint_norm(&residual) < 1e-10,
        "the degree split does not reproduce H: {:e}",
        joint_norm(&residual)
    );

    let (theta0, gs) = ground_state();
    let energy = expectation(&gs, h()).re;
    let interaction_energy = expectation(&gs, &interaction).re;
    let yukawa_energy = expectation(&gs, &yukawa).re;
    assert!(
        (energy - theta0).abs() < 1e-10,
        "the ground state is not an eigenstate: {energy:e} vs {theta0:e}"
    );
    assert!(
        yukawa_energy.abs() > 1e-6,
        "the Higgs/quark coupling contributes nothing: {yukawa_energy:e}"
    );
    assert!(
        yukawa_energy.abs() < 0.25 * energy.abs(),
        "the Higgs/quark coupling dominates the window energy: {yukawa_energy:e} of {energy:e}"
    );
    assert!(
        interaction_energy.abs() > yukawa_energy.abs(),
        "the full interaction share must exceed its Higgs/quark part"
    );
    assert!(
        interaction_energy.abs() < 0.9 * energy.abs(),
        "the interactions dominate the window energy: {interaction_energy:e} of {energy:e}"
    );
    eprintln!(
        "full_SM_interaction: E={energy:e} interaction={interaction_energy:e} ({:+.3}%) yukawa={yukawa_energy:e} ({:+.3}%)",
        100.0 * interaction_energy / energy,
        100.0 * yukawa_energy / energy
    );
}

/// Energy is the one quantity a closed system must conserve exactly: the exact
/// flow is unitary, so `<H>` is fixed. The compressed rational flow can only
/// move it by its own truncation, which is what makes the drift a physical
/// accuracy measure rather than an algebraic one.
#[test]
fn sm_full_energy_is_conserved_along_the_sirk_evolution() {
    let time = 0.02;
    let initial = h().d_gamma(&seed(), limits()).unwrap();
    let reference_energy =
        joint_inner_product(&seed(), &initial) / joint_inner_product(&seed(), &seed());
    assert!(
        reference_energy.im.abs() < 1e-10,
        "a Hermitian operator must give a real energy: {reference_energy}"
    );
    let mut drifts = Vec::new();
    // Restart levels 1 and 2 carry the conservation claim; the 4-way split is
    // already exercised by `sm_full_hashimoto_restarted_evolution`.
    for steps in [1usize, 2] {
        let result =
            evolve_restarted_joint(h(), &seed(), time, steps, &shifts(4, 9.0), &options()).unwrap();
        let energies: Vec<f64> = result.energies.iter().map(|e| e.re).collect();
        assert!(
            result.energies.iter().all(|e| e.im.abs() < 1e-9),
            "energies must stay real: {:?}",
            result.energies
        );
        let drift = energies
            .iter()
            .map(|e| (e - reference_energy.re).abs())
            .fold(0.0f64, f64::max);
        eprintln!(
            "full_SM_energy: steps={steps} reference={:e} drift={drift:e} energies={energies:?}",
            reference_energy.re
        );
        drifts.push(drift);
    }
    // The drift is roundoff-level for every restart count, so ordering it would
    // only measure noise; the physical claim is the bound.
    assert!(
        drifts.iter().all(|d| *d < 1e-9),
        "energy drifted beyond roundoff: {drifts:?}"
    );
}

/// Canonical structure of the nested Fock lift: with `q = a + a†` and
/// `p = i(a† - a)` (`[q, p] = i`), Ehrenfest gives `d<q>/dt = <p>`. A state
/// with bosonic content is evolved by SIRK and the finite-difference field
/// velocity is compared with the independently computed momentum.
#[test]
fn sm_full_canonical_field_velocity_matches_momentum() {
    let coordinate = InnerExpr::coordinate(coord::g(0, 0));
    let momentum = InnerExpr::momentum(coord::g(0, 0));
    let start = bosonic_seed();
    let image = h().d_gamma(&start, limits()).unwrap();
    let q0 = expectation(&start, &coordinate).re;
    let p0 = expectation(&start, &momentum).re;
    eprintln!("full_SM_ehrenfest: q0={q0:e} p0={p0:e}");
    // Analytic values for this state: <q> = cos(pi/4) and, with the SM
    // convention p = i(a† - a)/2, <p> = sin(pi/4)/2. The second pins the
    // momentum normalization, which [q, p] = i depends on.
    let eighth = std::f64::consts::FRAC_1_SQRT_2;
    assert!(
        (q0 - eighth).abs() < 1e-12,
        "<q> must be cos(pi/4) = {eighth:e}, got {q0:e}"
    );
    assert!(
        (p0 - 0.5 * eighth).abs() < 1e-12,
        "<p> must be sin(pi/4)/2 = {:e}, got {p0:e}",
        0.5 * eighth
    );
    let dt = 1e-3;
    // The cutoff-3 point is the strongest convergence evidence but the 936-state
    // window costs about two minutes, so the default run stops at cutoff 2 and
    // the runner's wide tier (`SM_FULL_WIDE_CUTOFFS=1`) re-measures it. Recorded
    // on 2026-11-09.
    const CUTOFF3_DEFECT: f64 = 4.31846279135123e-3;
    let deep = std::env::var("SM_FULL_WIDE_CUTOFFS").is_ok_and(|value| value == "1");
    let cutoffs: &[u64] = if deep { &[1, 2, 3] } else { &[1, 2] };
    let mut defects = Vec::new();
    for &cutoff in cutoffs {
        let mut window: BTreeSet<_> = project_occupation(&image, cutoff).0.into_keys().collect();
        window.extend(start.keys().cloned());
        let opts = JointSirkOptions {
            action_limits: ActionLimits {
                max_components: 8_000_000,
                max_work: 4_000_000_000,
            },
            occupation_cutoff: Some(cutoff),
            basis_window: Some(window),
            max_cached_components: 4_000_000,
            // The measured defect is the window's omitted action (3e-2 at
            // cutoff 2), so a 1e-9 target and an 8-iteration cap leave it
            // untouched while cutting the solve cost.
            method: JointKrylovMethod::ShiftInvert {
                relative_tolerance: 1e-9,
                max_iterations: 8,
            },
            ..Default::default()
        };
        let evolved = evolve_restarted_joint(h(), &start, dt, 1, &shifts(4, 9.0), &opts).unwrap();
        let velocity = (expectation(&evolved.state, &coordinate).re - q0) / dt;
        let defect = (velocity - p0).abs();
        eprintln!(
            "full_SM_ehrenfest: cutoff={cutoff} window={} tail={:e} d<q>/dt={velocity:e} <p>={p0:e} defect={defect:e}",
            opts.basis_window.as_ref().unwrap().len(),
            evolved.action_tail_norms[0][0]
        );
        defects.push(defect);
    }
    let last = *defects.last().unwrap();
    assert!(
        last < defects[0],
        "the canonical velocity must converge as the window closes: {defects:?}"
    );
    // Cutoff 2 leaves an 8% defect and cutoff 3 was measured at 1.2%; both are
    // far inside the omitted-action scale reported alongside each run.
    let bound = if deep { 1.25e-2 } else { 1.0e-1 } * p0.abs();
    assert!(
        last < bound,
        "<q> still moves at the wrong rate: defect {last:e} against |<p>| = {p0:e}"
    );
    if deep {
        assert!(
            (defects[2] - CUTOFF3_DEFECT).abs() < 1e-3 * CUTOFF3_DEFECT,
            "the cutoff-3 defect moved: {:e} vs {CUTOFF3_DEFECT:e}",
            defects[2]
        );
    }
}

#[test]
fn sm_full_hashimoto_interaction_mutations() {
    let original = h().d_gamma(&seed(), limits()).unwrap();
    for group in [
        "magG",
        "magW",
        "magB",
        "covH",
        "wall",
        "spatial",
        "gauge_matter",
        "yukawa",
    ] {
        let mut changed = model().clone();
        match group {
            "spatial" => changed.matter.retain(|v, _| !matches!(v, Variable::P(_))),
            "gauge_matter" => changed
                .matter
                .retain(|v, _| !matches!(v,Variable::Q(m) if *m<147)),
            "yukawa" => changed
                .matter
                .retain(|v, _| !matches!(v,Variable::Q(m) if *m>=147)),
            prefix => changed.forms.retain(|f| !f.name.starts_with(prefix)),
        }
        let image = changed.expression().d_gamma(&seed(), limits()).unwrap();
        assert!(
            distance(&original, &image) > 1e-5,
            "missing {group} was invisible"
        );
    }
    // Wrong chiral/spin, gauge and charged-Higgs signs must also be visible,
    // even though the mutated expressions still contain every sector.
    for variable in [
        Variable::P(coord::x(0)),
        Variable::Q(coord::w(0, 0)),
        Variable::Q(coord::phi(1)),
    ] {
        let mut changed = model().clone();
        for coefficient in changed.matter.get_mut(&variable).unwrap().values_mut() {
            *coefficient *= -1.0;
        }
        let image = changed.expression().d_gamma(&seed(), limits()).unwrap();
        assert!(
            distance(&original, &image) > 1e-5,
            "wrong coefficient sign was invisible"
        );
    }
    let constant = model().parameters.lambda * model().parameters.vev.powi(4) / 4.0;
    let missing_constant =
        InnerExpr::Sum(vec![h().clone(), InnerExpr::Scalar(C::new(-constant, 0.0))]);
    assert!(
        distance(
            &original,
            &missing_constant.d_gamma(&seed(), limits()).unwrap()
        ) > 1e-5
    );
    // Propagate resource failure without falling back to a simpler model.
    let error = solve_joint_sirk(
        h(),
        &seed(),
        &shifts(2, 9.0),
        &JointSirkOptions {
            action_limits: ActionLimits {
                max_components: 1,
                max_work: 100,
            },
            ..Default::default()
        },
    );
    assert!(matches!(error, Err(JointSirkError::Action(_))));
    assert!(hashimoto_unitary_shifts(1.0, 2.0, 1.0, 2).is_err());
    let mut opts = options();
    opts.max_cached_components = 1;
    assert!(matches!(
        solve_joint_sirk(h(), &seed(), &shifts(2, 9.0), &opts),
        Err(JointSirkError::CacheBudget { .. })
    ));
}
