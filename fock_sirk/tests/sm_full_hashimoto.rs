//! Full E1 SM operator -> joint dGamma -> genuine shift-invert rational Krylov.
//! Numerical windows project this SAME operator and report its omitted action.
//! No radial/free/decoupled replacement Hamiltonian is used.
use fock_sirk::joint_sirk::*;
use nalgebra::{DMatrix, DVector};
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, JointInnerBasis as B, JointOuterBasis as O, NestedVector,
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
    for (count, budget, tolerance) in [(2, 8.0, 1e-8), (4, 9.0, 2e-11), (5, 12.0, 2e-11)] {
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
    assert!(errors[2] < 1e-6);
    // Independent occupation budgets preserve the complete operator. They
    // report discarded action, rather than silently zeroing interactions. The
    // wider windows need raised budgets, since the cached images grow with the
    // window; the shipped defaults refuse them explicitly (see the doc).
    let mut tails = Vec::new();
    let mut supports = Vec::new();
    for (cutoff, cached, components, work) in [
        (1u64, 1_000_000usize, 500_000usize, 500_000_000usize),
        (2, 1_000_000, 500_000, 500_000_000),
        (3, 4_000_000, 8_000_000, 4_000_000_000),
        (4, 4_000_000, 8_000_000, 4_000_000_000),
    ] {
        let first_image = h().d_gamma(&seed(), limits()).unwrap();
        let mut window: BTreeSet<_> = project_occupation(&first_image, cutoff)
            .0
            .into_keys()
            .collect();
        window.extend(seed().into_keys());
        let opts = JointSirkOptions {
            action_limits: ActionLimits {
                max_components: components,
                max_work: work,
            },
            occupation_cutoff: Some(cutoff),
            basis_window: Some(window),
            max_cached_components: cached,
            method: JointKrylovMethod::ShiftInvert {
                relative_tolerance: 1e-8,
                max_iterations: 32,
            },
            ..Default::default()
        };
        let solve = solve_joint_sirk(h(), &seed(), &shifts(2, 9.0), &opts).unwrap();
        let tail = solve.projected_action_tail_norms[0];
        assert!(tail.is_finite() && tail >= 0.0);
        eprintln!(
            "full_SM_occupation: cutoff={cutoff} support={} tail={:?}",
            solve.registry.len(),
            solve.projected_action_tail_norms
        );
        supports.push(solve.registry.len());
        tails.push(tail);
    }
    // The window now covers the first complete image: the leading tail closes
    // monotonically to exactly zero, while the omitted-support norms above
    // remain nonzero for the later Krylov vectors (reported, not asserted).
    assert!(supports == [70, 230, 638, 1176], "windows grew: {supports:?}");
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
    for steps in [1, 2, 4] {
        let result =
            evolve_restarted_joint(h(), &seed(), time, steps, &shifts(4, 9.0), &options()).unwrap();
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
    assert!(errors[2] <= errors[0] + 1e-8);
    assert!(errors[2] < 1e-6);
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
