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
/// sector — the *electric* part of Gauss's law (`d_i Pi_i = 0`, the matter
/// density absent). This is the only version of the charge that can detect a
/// constraint violation: with the vanishing ladder, `Q` annihilates every state
/// of this dictionary. The matter half is added by `brst_charge_with_matter`,
/// and it changes what the penalty measures —
/// `sm_full_gauss_with_matter_charge_is_content_sensitive`.
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

/// The Gauss charge of this representation, and what its square actually
/// measures.
///
/// The physical content would be `Q|psi> = 0  <=>  G_g|psi> = 0`, with
/// `P = Q^dagger Q` and `|Q psi|^2 = <psi|P|psi>`. Measured on the compressed
/// window, `P` is the free-field electric zero-point plus a counting operator:
///
/// * `P`'s spectrum takes exactly three values, `9`, `9.5`, `10`, with
///   multiplicities 44/12/14 — the gauge vacuum at `12 * (3 * 1/4) = 9` and
///   half a quantum per excited gauge component;
/// * the floor `9` is therefore a zero-point constant, not a constraint gap:
///   `H + lambda P` shifts every energy by `lambda * 9` and selects no state,
///   with the residual excess shrinking as `lambda` grows;
/// * `|Q gs>|^2 = <gs|P|gs> = 9.0306` is the vacuum fluctuation of
///   `sum_i Pi_i`, not a charge violation — the generator carries no matter
///   term at all, so it cannot see one;
/// * `Q^2 = 0` holds exactly (`Q^2 gs = 0`), which is a real property of the
///   charge but says nothing about Gauss's law.
///
/// The analytic reason a normalizable `G|psi> = 0` does not exist is separate
/// and unmeasured here: `p = i(a^dagger - a)/2` has continuous spectrum, so the
/// constraint is *solved* in a gauge-fixed treatment (Coulomb gauge, where the
/// longitudinal momentum is fixed by the charge) rather than imposed on Fock
/// states. `sm_full_coulomb_gauge_eliminates_the_longitudinal_zero_point`
/// measures the eliminated mode; `sm_full_fermion_number_superselected` measures
/// the integrated statement that does survive.
#[test]
fn sm_full_gauss_law_measures_the_electric_zero_point() {
    let (theta0, gs) = ground_state();
    assert!(
        (theta0 - 49.7633053378465).abs() < 1e-9,
        "the window ground energy moved: {theta0:e}"
    );
    let charge = brst_charge();
    let penalty_matrix = window_matrix(&gauss_penalty());
    let penalty_spectrum = penalty_matrix.clone().symmetric_eigen();
    let gap = penalty_spectrum.eigenvalues.min();
    let mut buckets = std::collections::BTreeMap::<i64, usize>::new();
    for value in penalty_spectrum.eigenvalues.iter() {
        *buckets.entry((value * 1e6).round() as i64).or_default() += 1;
    }
    // 9 = 12 sectors x 3 components x 1/4 (the free oscillator's <p^2>), i.e.
    // the electric zero-point of the gauge vacuum, not a constraint gap.
    assert!(
        (gap - 9.0).abs() < 1e-9,
        "P's floor must be the gauge-vacuum zero-point 12*3/4 = 9: {gap:e}"
    );
    // The spectrum is 9, 9.5, 10 only: half a quantum per excited component.
    assert_eq!(
        buckets.keys().copied().collect::<Vec<_>>(),
        vec![9_000_000, 9_500_000, 10_000_000],
        "P must count gauge quanta on top of the zero-point: {buckets:?}"
    );
    let counts: Vec<usize> = buckets.values().copied().collect();
    assert_eq!(
        counts,
        vec![44, 12, 14],
        "occupancy classes moved: {buckets:?}"
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
    // The violation the ground state shows is slightly *above* the zero-point
    // floor, by its small population of excited gauge components.
    assert!(
        penalty_expectation > 9.0 && penalty_expectation < 9.5,
        "the ground state should sit just above the zero-point: {penalty_expectation:e}"
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
        let predicted = lambda * 9.0;
        assert!(
            (energy - theta0 - predicted).abs() < 1e-2 * predicted,
            "the penalty is not the zero-point shift lambda*9 at lambda={lambda}: {:e} vs {predicted:e}",
            energy - theta0
        );
        eprintln!(
            "full_SM_gauss: lambda={lambda} dE={:e} predicted={predicted:e}",
            energy - theta0
        );
        shifts.push((energy - theta0) / predicted - 1.0);
    }
    // The relative excess over the zero-point is the truncation's, and it
    // shrinks by roughly an order of magnitude per decade of lambda.
    assert!(
        shifts[1] < shifts[0] && shifts[2] < shifts[1],
        "the relative excess over lambda*9 must shrink with lambda: {shifts:?}"
    );

    assert!(
        nilpotent == 0.0,
        "the charge must be nilpotent: |Q^2 gs>| = {nilpotent:e}"
    );
    eprintln!(
        "full_SM_gauss: theta0={theta0:e} |Q gs>={residual:e} floor={gap:e} rel_excess={shifts:?}"
    );
}

/// Matter charge density `g_a rho^a = psi^dagger (g_a T^a) psi` of one gauge
/// sector, with the same generator matrices and coupling the covariant
/// derivative of `SmHamiltonian::build_matter` uses:
///
/// * colour `a < 8`: `gs * T^a` on the triplet modes (`T^a = lambda^a/2`);
/// * weak `8..=10`: `gw * sigma^a/2` on the left doublets only;
/// * hypercharge `11`: `gy * Y` on every mode.
///
/// Gauss's law is `(D_i E_i)^a = g_a rho^a`, so this is the term the electric
/// generator `sum_i Pi_i^a` is missing.
fn matter_charge(ghost: u32) -> InnerExpr {
    let parameters = &model().parameters;
    let mut pairs: Vec<(u32, u32, C)> = Vec::new();
    match ghost {
        0..=7 => {
            let generators = su3_generators();
            let t = generators[ghost as usize];
            for species in [Species::Up, Species::Down] {
                for generation in 0..3 {
                    for ch in [Chirality::Left, Chirality::Right] {
                        for spin in 0..2 {
                            for row in 0..3u32 {
                                for col in 0..3u32 {
                                    pairs.push((
                                        matter_mode(species, generation, row, ch, spin),
                                        matter_mode(species, generation, col, ch, spin),
                                        t[row as usize][col as usize] * C::new(parameters.gs, 0.0),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        8..=10 => {
            let a = ghost - 8;
            for (doublet, colors) in [
                ([Species::Up, Species::Down], 3u32),
                ([Species::Neutrino, Species::Electron], 1),
            ] {
                for generation in 0..3 {
                    for color in 0..colors {
                        for spin in 0..2 {
                            for (index, row) in doublet.iter().copied().enumerate() {
                                for (k, col) in doublet.iter().copied().enumerate() {
                                    pairs.push((
                                        matter_mode(row, generation, color, Chirality::Left, spin),
                                        matter_mode(col, generation, color, Chirality::Left, spin),
                                        pauli(a)[index][k] * C::new(0.5 * parameters.gw, 0.0),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {
            for species in [
                Species::Up,
                Species::Down,
                Species::Neutrino,
                Species::Electron,
            ] {
                for generation in 0..3 {
                    for ch in [Chirality::Left, Chirality::Right] {
                        for color in 0..species.colors() {
                            for spin in 0..2 {
                                let mode = matter_mode(species, generation, color, ch, spin);
                                pairs.push((
                                    mode,
                                    mode,
                                    C::new(parameters.gy * species.hypercharge(ch), 0.0),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    InnerExpr::Sum(
        pairs
            .into_iter()
            .filter(|(_, _, c)| *c != C::new(0.0, 0.0))
            .map(|(row, col, c)| {
                InnerExpr::Product(vec![
                    InnerExpr::Scalar(c),
                    InnerExpr::Ladder(InnerLadder::FermionCreate(row)),
                    InnerExpr::Ladder(InnerLadder::FermionAnnihilate(col)),
                ])
            })
            .collect(),
    )
}

/// Full Gauss generator `G_g = sum_i Pi_i^g - g_a rho^a`.
fn gauss_generator_with_matter(ghost: u32) -> InnerExpr {
    InnerExpr::Sum(vec![
        gauss_generator(ghost),
        matter_charge(ghost).scaled(C::new(-1.0, 0.0)),
    ])
}

/// `Q^dagger Q` for the charge built from `gauss_generator_with_matter`.
fn gauss_penalty_with_matter() -> InnerExpr {
    InnerExpr::Sum(
        (0..12u32)
            .map(|ghost| {
                let generator = gauss_generator_with_matter(ghost);
                InnerExpr::Product(vec![generator.clone(), generator])
            })
            .collect(),
    )
}

/// The charge `Q = sum_g G_g c_g^dagger` with the matter density included.
fn brst_charge_with_matter() -> InnerExpr {
    InnerExpr::Sum(
        (0..12u32)
            .map(|ghost| {
                InnerExpr::Product(vec![
                    gauss_generator_with_matter(ghost),
                    InnerExpr::Ladder(InnerLadder::FermionCreate(MATTER_MODES + ghost)),
                ])
            })
            .collect(),
    )
}

/// Total matter fermion number `N = sum_{i < MATTER_MODES} a_i^dagger a_i`.
fn fermion_number() -> InnerExpr {
    InnerExpr::Sum(
        (0..MATTER_MODES)
            .map(|mode| {
                InnerExpr::Product(vec![
                    InnerExpr::Ladder(InnerLadder::FermionCreate(mode)),
                    InnerExpr::Ladder(InnerLadder::FermionAnnihilate(mode)),
                ])
            })
            .collect(),
    )
}

/// The longitudinal part of the electric energy,
/// `(1/2) sum_g (Pi_L^g)^2` with `Pi_L^g = sum_i Pi_i^g / sqrt(3)`.
///
/// Since `G_g = sqrt(3) Pi_L^g`, this is `Q^dagger Q / 6`: the gauge-fixing
/// penalty is the electric energy of the three-component zero mode, scaled by
/// the `6 = 3 * 2` that `(sum_i Pi_i)^2` and the `1/2` contribute.
fn longitudinal_energy() -> InnerExpr {
    InnerExpr::Sum(
        (0..12u32)
            .map(|ghost| {
                let generator = gauss_generator(ghost);
                InnerExpr::Product(vec![generator.clone(), generator])
                    .scaled(C::new(1.0 / 6.0, 0.0))
            })
            .collect(),
    )
}

fn window_expectation(state: &NestedVector, operator: &InnerExpr) -> f64 {
    let dense = dense_state(state);
    let matrix = window_matrix(operator);
    (dense.adjoint() * matrix * &dense)[(0, 0)].re
}

/// Gauss's law with the matter charge density in it.
///
/// The electric generator `sum_i Pi_i^g` of the test above is one half of
/// `G_g = (D_i E_i)^g - g_a rho^a`; the window carries matter, so the missing
/// half is not a constant. Adding it (`matter_charge`, built from the same
/// generator matrices and couplings as `SmHamiltonian::build_matter`) is
/// only a small change in the operator and a large one in what the constraint
/// measures: it is a charge-sensitive penalty, not the zero-point counter.
///
/// Measured on the recorded window:
///
/// * `rho^a` is diagonal in the window basis in all twelve sectors, and
///   non-vanishing only for the four that act diagonally on definite matter
///   content — the two colour Cartan generators, the weak `T^3` and
///   hypercharge. The eight off-diagonal generators compress to exactly zero:
///   the window is not closed under the gauge rotations, so it cannot see a
///   colour/isospin coherence it does not contain;
/// * the surviving values are the dictionary's own: `g_s {1/2, -1/2, 0}`,
///   `g_s {1/(2 sqrt 3), -1/sqrt 3}`, `g_w {0, 1/2, 1}`, `g_y {1/6, -1/3, -5/6}`,
///   so the coupling and the `T^a = lambda^a/2`, `sigma^a/2`, `Y` normalisation
///   are the ones the Hamiltonian's covariant derivative uses;
/// * the penalty's floor drops from `9` (the electric zero-point) to `8.9377`,
///   below it, and its spectrum spreads from three values to ten: the minimum
///   has moved to a different matter content, so the penalty now selects;
/// * the ground state's violation nonetheless *rises*, from `9.0306` to
///   `9.2326` (`|Q gs>|` from `3.005` to `3.0385`): the two facts are not
///   contradictory — a lower floor with a higher ground state is exactly what a
///   content-sensitive functional looks like;
/// * `|Q gs>|^2 = <gs|P|gs>` still holds to `1e-12` when `P` is built at
///   operator level, so the ghost construction and the field penalty agree;
/// * the gauge algebra closes on the density: `[rho^0, rho^1] = i g_s^2 T^3`
///   and `[rho^8, rho^9] = i g_w^2 T^3` on the seed, the off-diagonal pairs
///   reflecting the coefficients the dictionary assigns them.
///
/// What this does *not* establish: nilpotency of `Q`. With one ghost mode per
/// sector and no antighost or cubic ghost term, `Q^2` is not zero here
/// (`|Q^2 gs>| = 0.139`); the matter term makes that visible instead of hiding
/// it behind an abelianised generator, and the missing piece is the ghost
/// cubic that cancels `[G_g, G_h]`. The penalty, which is `[Q, Q^dagger]/2`-like,
/// is unaffected by that gap.
#[test]
fn sm_full_gauss_with_matter_charge_is_content_sensitive() {
    let window = &reference().basis;
    let (theta0, gs_state) = ground_state();
    let parameters = &model().parameters;

    // The density, sector by sector: diagonal, and diagonal *values*.
    let mut charged = Vec::new();
    for ghost in 0..12u32 {
        let matrix = window_matrix(&matter_charge(ghost));
        let off_diagonal = &matrix - DMatrix::from_diagonal(&matrix.diagonal().into());
        assert!(
            off_diagonal.norm() < 1e-12,
            "rho^{ghost} is not diagonal on the window: {:e}",
            off_diagonal.norm()
        );
        let mut values: Vec<f64> = matrix.diagonal().iter().map(|c| c.re).collect();
        values.sort_by(f64::total_cmp);
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        eprintln!("full_SM_matter: ghost={ghost} values={values:?}");
        if values.iter().any(|v| v.abs() > 1e-12) {
            charged.push((ghost, values));
        }
    }
    // Only the four generators that are diagonal on definite matter content
    // survive the compression: T^3 and T^8 of colour, T^3 of weak isospin, Y.
    assert_eq!(
        charged.iter().map(|(g, _)| *g).collect::<Vec<_>>(),
        vec![2, 7, 10, 11],
        "the window's charged sectors moved: {charged:?}"
    );
    // The other eight are not merely traceless on the window: they vanish on
    // it, i.e. it contains no colour or isospin coherence for them to rotate.
    for ghost in [0u32, 1, 3, 4, 5, 6, 8, 9] {
        assert!(
            window_matrix(&matter_charge(ghost)).norm() < 1e-12,
            "rho^{ghost} must vanish on a window with definite matter content"
        );
    }
    let close = |values: &[f64], want: &[f64]| {
        values.len() == want.len() && values.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12)
    };
    let third = 1.0 / 3f64.sqrt();
    assert!(
        close(
            &charged[0].1,
            &[-0.5 * parameters.gs, 0.0, 0.5 * parameters.gs]
        ),
        "colour T^3 values: {:?}",
        charged[0].1
    );
    assert!(
        close(
            &charged[1].1,
            &[-third * parameters.gs, 0.5 * third * parameters.gs]
        ),
        "colour T^8 values: {:?}",
        charged[1].1
    );
    assert!(
        close(&charged[2].1, &[0.0, 0.5 * parameters.gw, parameters.gw]),
        "weak T^3 values: {:?}",
        charged[2].1
    );
    assert!(
        close(
            &charged[3].1,
            &[
                -5.0 * parameters.gy / 6.0,
                -parameters.gy / 3.0,
                parameters.gy / 6.0,
            ]
        ),
        "hypercharge values: {:?}",
        charged[3].1
    );

    // The penalty at operator level: the square must be built from the full
    // generator, not by squaring a compression.
    let penalty = window_matrix(&gauss_penalty_with_matter());
    let spectrum = penalty.clone().symmetric_eigen();
    let floor = spectrum.eigenvalues.min();
    let mut buckets = std::collections::BTreeMap::<i64, usize>::new();
    for value in spectrum.eigenvalues.iter() {
        *buckets.entry((value * 1e4).round() as i64).or_default() += 1;
    }
    assert!(
        buckets.values().sum::<usize>() == window.len(),
        "the penalty must stay on the window: {buckets:?}"
    );
    // 8.9377, below the electric zero-point of 9: the least-violating window
    // state is no longer the gauge vacuum.
    assert!(
        (floor - 8.937_718_492_951).abs() < 1e-9,
        "the matter-sensitive floor moved: {floor:.12}"
    );
    assert_eq!(
        buckets.len(),
        10,
        "the matter-sensitive penalty must spread the spectrum: {buckets:?}"
    );
    let mut counts: Vec<usize> = buckets.values().copied().collect();
    counts.sort_unstable();
    assert_eq!(
        counts,
        vec![1, 1, 2, 4, 4, 9, 10, 12, 12, 15],
        "the penalty's occupancy classes moved: {buckets:?}"
    );
    let dense = dense_state(&gs_state);
    let with_matter = (dense.adjoint() * &penalty * &dense)[(0, 0)].re;
    let electric = window_expectation(&gs_state, &gauss_penalty());
    assert!(
        (electric - 9.030_620_800_228).abs() < 1e-9,
        "the electric ground-state penalty moved: {electric:.12}"
    );
    assert!(
        (with_matter - 9.232_585_715_293).abs() < 1e-9,
        "the matter ground-state penalty moved: {with_matter:.12}"
    );
    assert!(
        with_matter > electric,
        "the ground state's penalty must rise with the matter term: {with_matter:e} vs {electric:e}"
    );

    // The charge built from the same generators: the ghost construction and the
    // field penalty must still agree.
    let charge = brst_charge_with_matter();
    let applied = charge.d_gamma(&gs_state, limits()).unwrap();
    let residual = joint_norm(&applied);
    assert!(
        (residual * residual - with_matter).abs() < 1e-9,
        "|Q gs>|^2 = {:e} but <gs|P|gs> = {with_matter:e}",
        residual * residual
    );
    // And it is not nilpotent: one ghost mode per sector cannot cancel
    // [G_g, G_h] without the cubic ghost term.
    let nilpotent = joint_norm(&charge.d_gamma(&applied, limits()).unwrap());
    assert!(
        nilpotent > 1e-6,
        "this charge has no cubic term, so Q^2 must not vanish: {nilpotent:e}"
    );

    // The gauge algebra closes on the density itself: the coefficient is the
    // coupling squared, times 1/2 for the two doublets that carry it.
    let inner = B::new(
        [],
        [
            matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
            matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
        ],
    )
    .unwrap();
    let probe = NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))]);
    let commutator = |a: u32, b: u32| {
        // Composition is right to left, so `forward` is rho^a after rho^b.
        let forward = matter_charge(a).d_gamma(
            &matter_charge(b).d_gamma(&probe, limits()).unwrap(),
            limits(),
        );
        let backward = matter_charge(b).d_gamma(
            &matter_charge(a).d_gamma(&probe, limits()).unwrap(),
            limits(),
        );
        let mut difference = forward.unwrap();
        joint_axpy(
            &mut difference,
            &backward.unwrap(),
            C::new(-1.0, 0.0),
            limits(),
        )
        .unwrap();
        joint_inner_product(&probe, &difference) / joint_inner_product(&probe, &probe)
    };
    let colour_algebra = commutator(0, 1);
    let weak_algebra = commutator(8, 9);
    eprintln!("full_SM_matter: [rho0,rho1]={colour_algebra:?} [rho8,rho9]={weak_algebra:?}");
    assert!(
        (colour_algebra - C::new(0.0, parameters.gs * parameters.gs * 0.5)).norm() < 1e-12,
        "colour algebra: {colour_algebra:?}"
    );
    assert!(
        (weak_algebra - C::new(0.0, parameters.gw * parameters.gw)).norm() < 1e-12,
        "weak algebra: {weak_algebra:?}"
    );
    // The two Cartan directions commute, so no mixed sector can contribute.
    assert!(
        commutator(2, 10).norm() < 1e-12,
        "T^3 of colour and of weak isospin must commute"
    );

    eprintln!(
        "full_SM_matter: theta0={theta0:.12} floor={floor:.12} with_matter={with_matter:.12} electric={electric:.12} residual={residual:.12} nilpotent={nilpotent:.12}"
    );
}

/// Coulomb gauge: the longitudinal mode is solved for, not imposed.
///
/// Gauss's law `G_g = sum_i Pi_i^g + rho_g = 0` determines the longitudinal
/// momentum; with the matter charge absent from this dictionary (`rho = 0`) the
/// determination is `Pi_L = 0`, and Coulomb gauge removes the mode from the
/// Hilbert space instead of asking a state to annihilate it. The energy it
/// contributes is `(1/2) sum_g (Pi_L^g)^2`, so the Coulomb-gauge Hamiltonian is
/// `H_C = H - (1/2) sum_g (Pi_L^g)^2`. Measured:
///
/// * `P = 6 * (1/2) sum_g (Pi_L^g)^2` — the gauge-fixing penalty *is* the
///   longitudinal electric energy up to that factor, so its floor `9` is a
///   zero-point and not a constraint functional;
/// * the electric zero-point splits exactly as `12 * 1/8` longitudinal +
///   `12 * 2 * 1/4 * 1/2` transverse + `4 * 1/8` Higgs `= 1.5 + 3 + 0.5 = 5`;
/// * on the 44-dimensional longitudinal-vacuum sector (the `P = 9` eigenspace)
///   the elimination is *exactly* the constant `1.5`, so `H_C` reproduces those
///   window energies shifted by the longitudinal zero-point;
/// * off that sector it is not a constant: the ground state sits at
///   `1.5051 = 1.5 + 0.0051`, so the elimination moves its energy by more than
///   the zero-point, and no identification of `H_C` with `H` holds for states
///   the gauge-fixed theory does not contain.
#[test]
fn sm_full_coulomb_gauge_eliminates_the_longitudinal_zero_point() {
    let seed_state = seed();
    let n = reference().basis.len();
    let penalty_matrix = window_matrix(&gauss_penalty());
    let longitudinal = window_matrix(&longitudinal_energy());

    // G_g = sqrt(3) Pi_L^g, so the penalty is the longitudinal energy times 6.
    assert!(
        (&penalty_matrix - &longitudinal * C::new(6.0, 0.0)).norm() < 1e-12,
        "the Gauss penalty must be 6x the longitudinal electric energy"
    );

    // Zero-point accounting on the gauge vacuum.
    let mut gauge = DMatrix::<C>::zeros(n, n);
    let mut higgs = DMatrix::<C>::zeros(n, n);
    for mode in coord::active() {
        let momentum = InnerExpr::momentum(mode);
        let square = InnerExpr::Product(vec![momentum.clone(), momentum]);
        let term = window_matrix(&square) * C::new(0.5, 0.0);
        if (147..151).contains(&mode) {
            higgs += &term;
        } else {
            gauge += &term;
        }
    }
    let transverse = &gauge - &longitudinal;
    let seed_dense = dense_state(&seed_state);
    let long_seed = window_expectation(&seed_state, &longitudinal_energy());
    let trans_seed = (seed_dense.adjoint() * &transverse * &seed_dense)[(0, 0)].re;
    let higgs_seed = (seed_dense.adjoint() * &higgs * &seed_dense)[(0, 0)].re;
    eprintln!(
        "full_SM_coulomb: long={long_seed} trans={trans_seed} higgs={higgs_seed} total={}",
        long_seed + trans_seed + higgs_seed
    );
    for (measured, want, label) in [
        (long_seed, 1.5, "longitudinal 12*(1/8)"),
        (trans_seed, 3.0, "transverse 12*2*(1/8)"),
        (higgs_seed, 0.5, "Higgs 4*(1/8)"),
    ] {
        assert!(
            (measured - want).abs() < 1e-12,
            "electric zero-point {label} must be {want}: {measured}"
        );
    }

    // The 44-dimensional longitudinal-vacuum sector: elimination is a constant.
    let spectrum = penalty_matrix.clone().symmetric_eigen();
    let vacuum: Vec<usize> = spectrum
        .eigenvalues
        .iter()
        .enumerate()
        .filter(|(_, value)| (**value - 9.0).abs() < 1e-9)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(vacuum.len(), 44, "longitudinal-vacuum sector moved");
    let mut basis = DMatrix::<C>::zeros(n, vacuum.len());
    for (column, index) in vacuum.iter().enumerate() {
        basis.set_column(column, &spectrum.eigenvectors.column(*index));
    }
    let restricted_long = basis.adjoint() * &longitudinal * &basis;
    let want = DMatrix::<C>::identity(vacuum.len(), vacuum.len()) * C::new(1.5, 0.0);
    assert!(
        (&restricted_long - &want).norm() < 1e-12,
        "on the longitudinal vacuum the eliminated energy must be the constant 1.5: {:e}",
        (&restricted_long - &want).norm()
    );
    // Hence C-gauge energies on that sector are the window energies minus 1.5.
    let restricted_h = basis.adjoint() * &reference().matrix * &basis;
    let restricted_coulomb = basis.adjoint() * (&reference().matrix - &longitudinal) * &basis;
    let eliminated = &restricted_h - &restricted_coulomb;
    assert!(
        (&eliminated - &want).norm() < 1e-12,
        "H - H_C must be exactly 1.5 on the longitudinal vacuum: {:e}",
        (&eliminated - &want).norm()
    );
    let lowest = restricted_h.clone().symmetric_eigen().eigenvalues.min();
    let lowest_coulomb = restricted_coulomb.symmetric_eigen().eigenvalues.min();
    assert!(
        (lowest - lowest_coulomb - 1.5).abs() < 1e-12,
        "the Coulomb-gauge spectrum must be the window spectrum shifted by 1.5: {lowest:e} vs {lowest_coulomb:e}"
    );
    eprintln!(
        "full_SM_coulomb: gauge_vacuum_dim={} E={lowest:e} E_coulomb={lowest_coulomb:e} shift={:e}",
        vacuum.len(),
        lowest - lowest_coulomb
    );

    // Off that sector the elimination is not a constant.
    let (theta0, gs) = ground_state();
    let gs_long = window_expectation(&gs, &longitudinal_energy());
    assert!(
        gs_long > 1.5 + 1e-6 && gs_long < 1.5 + 1e-2,
        "the ground state carries a small longitudinal excess: {gs_long:e}"
    );
    eprintln!(
        "full_SM_coulomb: gs_long={gs_long:e} theta0={theta0:e} coulomb_energy={:e}",
        theta0 - gs_long
    );
    assert!(
        (theta0 - gs_long) < theta0 - 1.5 + 0.01,
        "the elimination must lower the window energy by more than the zero-point here"
    );
}

/// Charge superselection: `[H, N] = 0` and no matrix element between sectors.
///
/// The recorded window is a *single* charge sector — every one of its 70 states
/// carries two matter fermions, so `N = 2 * I` there and no charge-variance
/// statement can be made inside it. The anchor is therefore the operator
/// statement, measured on probes in five sectors `N = 0..4`:
///
/// * `[H, N]` vanishes to rounding (relative defect `9.1e-16`);
/// * every matrix element of `H` between different sectors vanishes exactly
///   (20 ordered pairs, worst `0.0`), which is the superselection statement
///   proper;
/// * `N` is diagonal and constant on the window, so the fixed-charge
///   truncation `project_occupation` cannot leak charge.
#[test]
fn sm_full_fermion_number_superselected() {
    let number = fermion_number();
    let number_matrix = window_matrix(&number);
    let number_spectrum = number_matrix.clone().symmetric_eigen();
    let mut histogram = std::collections::BTreeMap::<i64, usize>::new();
    for value in number_spectrum.eigenvalues.iter() {
        *histogram.entry(value.round() as i64).or_default() += 1;
    }
    assert_eq!(
        histogram,
        [(2, 70)].into_iter().collect(),
        "the window must be one charge sector: {histogram:?}"
    );
    let commutation =
        (&reference().matrix * &number_matrix - &number_matrix * &reference().matrix).norm();
    assert!(
        commutation < 1e-12,
        "[H, N] must vanish on the window: {commutation:e}"
    );

    let pool: Vec<u32> = (0..6u32)
        .map(|j| matter_mode(Species::Down, 0, j % 3, Chirality::Left, j / 3))
        .collect();
    let mut states = Vec::new();
    let mut images = Vec::new();
    for k in 0..5usize {
        let mut modes: Vec<u32> = pool.iter().copied().take(k).collect();
        modes.sort_unstable();
        modes.dedup();
        let inner = B::new([], modes.clone()).unwrap();
        let state = NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))]);
        let applied = number.d_gamma(&state, limits()).unwrap();
        let charge =
            joint_inner_product(&state, &applied).re / joint_inner_product(&state, &state).re;
        assert!(
            (charge - modes.len() as f64).abs() < 1e-12,
            "probe {k} should have N = {}: {charge}",
            modes.len()
        );
        let image = h().d_gamma(&state, limits()).unwrap();
        let left = h().d_gamma(&applied, limits()).unwrap();
        let right = number.d_gamma(&image, limits()).unwrap();
        let mut defect = left;
        joint_axpy(&mut defect, &right, C::new(-1.0, 0.0), limits()).unwrap();
        let relative = joint_norm(&defect) / joint_norm(&image);
        assert!(
            relative < 1e-12,
            "[H, N] must vanish on the N = {} probe: {relative:e}",
            modes.len()
        );
        states.push(state);
        images.push((image, charge));
    }
    let mut worst = 0.0f64;
    let mut pairs = 0usize;
    for (i, (image, charge)) in images.iter().enumerate() {
        for other in states.iter().enumerate() {
            let (j, other) = other;
            if i == j {
                continue;
            }
            if (images[j].1 - charge).abs() < 1e-12 {
                continue;
            }
            let element =
                joint_inner_product(other, image).norm() / (joint_norm(image) * joint_norm(other));
            worst = worst.max(element);
            pairs += 1;
        }
    }
    assert_eq!(pairs, 20, "cross-sector pair count moved");
    assert!(
        worst < 1e-12,
        "H must not connect different charge sectors: worst {worst:e}"
    );
    eprintln!(
        "full_SM_charge: sectors=0..4 window_N=2I comm={commutation:e} cross={worst:e} pairs={pairs}"
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
