//! N10 joint dictionary: full (unprojected) Gauss algebra, BRST nilpotency and
//! the Hamiltonian commutator, on the complete E1 operator and its 96 CAR modes.
//! These are finite-representation identities; they are not a physical-cohomology
//! or anomaly theorem. The joint charge closes on gauge-occupied images after the
//! SU(3) electric term was corrected to all eight colors and the generator was
//! normalized with the bare structure constants (not `g f`).
use fock_sirk::joint_sirk::*;
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, JointInnerBasis as B, JointOuterBasis as O, NestedVector,
};
use nested_fock_algebra::sm_constraints::*;
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
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
        max_components: 2_000_000,
        max_work: 20_000_000_000,
    }
}
fn seed() -> NestedVector {
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
fn with_gauge() -> NestedVector {
    let inner = B::new(
        [(coord::g(1, 0), 1), (coord::g(2, 0), 1)],
        [
            matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
            matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
        ],
    )
    .unwrap();
    NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))])
}
fn with_ghosts(ghosts: &[u32]) -> NestedVector {
    let mut fermions = vec![
        matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
        matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
    ];
    fermions.extend(ghosts.iter().map(|g| GHOST_BASE + g));
    let inner = B::new([], fermions).unwrap();
    NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))])
}
fn norm(v: &NestedVector) -> f64 {
    joint_norm(v)
}
fn diff_norm(a: &NestedVector, b: &NestedVector) -> f64 {
    let mut d = a.clone();
    joint_axpy(&mut d, b, C::new(-1.0, 0.0), limits()).unwrap();
    norm(&d)
}
fn apply(op: &InnerExpr, v: &NestedVector) -> NestedVector {
    op.d_gamma(v, limits()).unwrap()
}

/// The E1 jet coordinates: dG, dW, dB and dphi. They have no conjugate momenta
/// in this representation, so a homogeneous Gauss charge cannot shift them.
fn is_jet(inner: &B) -> bool {
    inner
        .bosons()
        .keys()
        .any(|&m| matches!(m, 27..=98 | 108..=134 | 138..=146 | 151..=162))
}
fn split_norm(v: &NestedVector) -> (f64, f64) {
    let (mut jet, mut rest) = (NestedVector::new(), NestedVector::new());
    for (outer, a) in v {
        let has_jet = outer.inner_occupations().iter().any(|(b, _)| is_jet(b));
        if has_jet {
            jet.insert(outer.clone(), *a);
        } else {
            rest.insert(outer.clone(), *a);
        }
    }
    (norm(&jet), norm(&rest))
}

#[test]
fn gauss_generators_close_on_gauge_vacuum_images() {
    let g = gauge_dictionary(&parameters());
    assert_eq!(g.len(), GENERATORS);
    let state = seed();
    let mut worst = 0.0f64;
    let mut worst_pair = (0usize, 0usize);
    let mut checked = 0usize;
    for a in 0..GENERATORS {
        for b in 0..GENERATORS {
            let mut lhs = apply(&g[a], &apply(&g[b], &state));
            joint_axpy(
                &mut lhs,
                &apply(&g[b], &apply(&g[a], &state)),
                C::new(-1.0, 0.0),
                limits(),
            )
            .unwrap();
            let mut rhs = NestedVector::new();
            for (c, gc) in g.iter().enumerate() {
                let f = structure(a, b, c);
                if f != 0.0 {
                    let image = apply(gc, &state);
                    for (k, v) in image {
                        *rhs.entry(k).or_default() += C::new(0.0, f) * v;
                    }
                }
            }
            let residual = diff_norm(&lhs, &rhs);
            if residual > worst {
                worst = residual;
                worst_pair = (a, b);
            }
            if residual > 1e-9 {
                eprintln!("joint_gauss: pair ({a},{b}) residual {residual:e}");
            }
            checked += 1;
        }
    }
    eprintln!(
        "joint_gauss: {checked} commutators, worst pair {:?} residual {worst:e}",
        worst_pair
    );
    assert!(
        worst < 2e-9,
        "gauge-vacuum Gauss algebra residual {worst:e} exceeds the finite-representation tolerance"
    );
}

/// The electric term is invisible on the gauge-vacuum seed above. On a state
/// with two gauge quanta the quantized adjoint bilinear closes on the bare
/// structure constants once all eight SU(3) colors are present.
#[test]
fn gauss_algebra_closes_on_gauge_occupied_images() {
    let g = gauge_dictionary(&parameters());
    let state = with_gauge();
    let mut worst = 0.0f64;
    let mut worst_pair = (0usize, 0usize);
    for a in 0..8 {
        for b in 0..8 {
            let mut lhs = apply(&g[a], &apply(&g[b], &state));
            joint_axpy(
                &mut lhs,
                &apply(&g[b], &apply(&g[a], &state)),
                C::new(-1.0, 0.0),
                limits(),
            )
            .unwrap();
            let mut rhs = NestedVector::new();
            for (c, gc) in g.iter().enumerate() {
                let f = structure(a, b, c);
                if f != 0.0 {
                    for (k, v) in apply(gc, &state) {
                        *rhs.entry(k).or_default() += C::new(0.0, f) * v;
                    }
                }
            }
            let residual = diff_norm(&lhs, &rhs);
            if residual > worst {
                worst = residual;
                worst_pair = (a, b);
            }
        }
    }
    eprintln!(
        "joint_gauss (gauge-occupied): worst pair {:?} residual {worst:e}",
        worst_pair
    );
    assert!(
        worst < 2e-9,
        "gauge-occupied Gauss algebra residual {worst:e} exceeds tolerance"
    );
}

#[test]
fn electric_sector_algebra_closes_with_bare_structure_constants() {
    let p = parameters();
    let state = with_gauge();
    let e: Vec<_> = (0..8).map(|a| electric(&p, Sector::Su3(a))).collect();
    let mut worst = 0.0f64;
    let mut pair = (0, 0);
    for a in 0..8 {
        for b in 0..8 {
            let mut lhs = apply(&e[a], &apply(&e[b], &state));
            joint_axpy(
                &mut lhs,
                &apply(&e[b], &apply(&e[a], &state)),
                C::new(-1.0, 0.0),
                limits(),
            )
            .unwrap();
            let mut rhs = NestedVector::new();
            for (c, ec) in e.iter().enumerate() {
                let f = structure(a, b, c);
                if f != 0.0 {
                    for (k, v) in apply(ec, &state) {
                        *rhs.entry(k).or_default() += C::new(0.0, f) * v;
                    }
                }
            }
            let r = diff_norm(&lhs, &rhs);
            if r > worst {
                worst = r;
                pair = (a, b);
            }
        }
    }
    eprintln!("electric_diag: worst pair {pair:?} residual {worst:e}");
    assert!(worst < 2e-9, "electric SU(3) algebra residual {worst:e}");
}

#[test]
fn joint_brst_charge_is_nilpotent_on_ghost_images() {
    let q = brst_charge(&parameters());
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    for g in [0u32, 1, 2, 3, 7, 8, 9, 10, 11] {
        for state in [with_ghosts(&[g]), with_ghosts(&[g, (g + 5) % 12])] {
            let image = apply(&q, &apply(&q, &state));
            let residual = norm(&image);
            eprintln!("joint_brst: ghosts {:?} residual {residual:e}", state.len());
            worst = worst.max(residual);
            checked += 1;
        }
    }
    eprintln!("joint_brst: {checked} ghost images, worst |Q^2 psi| {worst:e}");
    assert!(worst < 2e-9, "Q_joint^2 residual {worst:e}");
}

#[test]
fn joint_charge_commutes_on_ghost_images() {
    let q = brst_charge(&parameters());
    let mut su_worst = 0.0f64;
    let mut u1 = f64::INFINITY;
    let mut total_worst = 0.0f64;
    let mut checked = 0usize;
    for g in [0u32, 2, 8, 11] {
        let state = with_ghosts(&[g]);
        let hq = apply(&q, &apply(h(), &state));
        let qh = apply(h(), &apply(&q, &state));
        let mut residual = hq.clone();
        joint_axpy(&mut residual, &qh, C::new(-1.0, 0.0), limits()).unwrap();
        let (jet, rest) = split_norm(&residual);
        let total = norm(&residual);
        eprintln!("joint_brst: [H,Q] on ghost {g}: total {total:e}, jet {jet:e}, non-jet {rest:e}");
        total_worst = total_worst.max(total);
        if g == 11 {
            u1 = rest;
        } else {
            su_worst = su_worst.max(rest);
        }
        checked += 1;
    }
    assert_eq!(checked, 4);
    // All twelve charges commute with H on these full ghost images. The
    // nonabelian closure required all eight SU(3) colors in the electric term
    // and the bare structure constants in the generator.
    assert!(
        u1 < 2e-9,
        "U(1)_Y non-jet commutator should vanish, got {u1:e}"
    );
    assert!(
        su_worst < 2e-9,
        "nonabelian [H,Q] residual {su_worst:e} exceeds tolerance"
    );
    assert!(
        total_worst < 2e-9,
        "total [H,Q] residual {total_worst:e} exceeds tolerance"
    );
}
