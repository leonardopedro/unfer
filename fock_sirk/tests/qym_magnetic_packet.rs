//! Continuous three-profile Weyl-gauge form, before residual Gauss reduction.
use fock_sirk::joint_sirk::{
    JointKrylovMethod, JointSirkOptions, hashimoto_unitary_shifts, joint_inner_product,
    solve_joint_sirk,
};
use nalgebra::DMatrix;
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, JointInnerBasis, JointOuterBasis, NestedVector,
};
use num_complex::Complex64 as C;
use std::collections::BTreeSet;

struct Packet {
    entry: f64,
    packet_action: Coefficients,
}

struct Coefficients {
    profile_metric: [f64; 3],
    curl_norm_squared: [f64; 3],
    wall_12_13: [f64; 2],
}

impl Packet {
    fn hamiltonian(&self, g: f64) -> InnerExpr {
        let coeff = &self.packet_action;
        let mut terms = Vec::new();
        for mode in 0..3 {
            let q = InnerExpr::coordinate(mode as u32);
            let p = InnerExpr::momentum(mode as u32);
            terms.push(
                InnerExpr::Product(vec![p.clone(), p])
                    .scaled(C::from(0.5 / coeff.profile_metric[mode])),
            );
            terms.push(
                InnerExpr::Product(vec![q.clone(), q])
                    .scaled(C::from(0.5 * coeff.curl_norm_squared[mode])),
            );
        }
        let triple = (0..3).map(InnerExpr::coordinate).collect();
        let mixed = self.entry;
        terms.push(InnerExpr::Product(triple).scaled(C::from(g * mixed)));
        for (index, wall) in coeff.wall_12_13.iter().enumerate() {
            let other = (index + 1) as u32;
            let words = [0, 0, other, other].map(InnerExpr::coordinate);
            terms.push(InnerExpr::Product(words.to_vec()).scaled(C::from(0.5 * g * g * wall)));
        }
        InnerExpr::Sum(terms)
    }
}

fn label(x: u32, y: u32, z: u32) -> JointOuterBasis {
    let inner = JointInnerBasis::new([(0, x), (1, y), (2, z)], []).unwrap();
    JointOuterBasis::bosonic([(inner, 1)]).unwrap()
}

fn seed(label: JointOuterBasis) -> NestedVector {
    NestedVector::from([(label, C::from(1.0))])
}

fn window(cutoff: u32) -> BTreeSet<JointOuterBasis> {
    let mut result = BTreeSet::new();
    for x in 0..=cutoff {
        for y in 0..=cutoff - x {
            for z in 0..=cutoff - x - y {
                result.insert(label(x, y, z));
            }
        }
    }
    result
}

fn reference_packet() -> Packet {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../docs/fixtures/qym_magnetic_packet.json")).unwrap();
    let coeff = &fixture["packet_action"];
    Packet {
        entry: fixture["entry_decimal"].as_str().unwrap().parse().unwrap(),
        packet_action: Coefficients {
            profile_metric: serde_json::from_value(coeff["profile_metric"].clone()).unwrap(),
            curl_norm_squared: serde_json::from_value(coeff["curl_norm_squared"].clone()).unwrap(),
            wall_12_13: serde_json::from_value(coeff["wall_12_13"].clone()).unwrap(),
        },
    }
}

fn occupied_matrix(h: &InnerExpr, labels: &[JointOuterBasis]) -> DMatrix<C> {
    let limits = ActionLimits::default();
    let mut dense = DMatrix::<C>::zeros(labels.len(), labels.len());
    for (j, input) in labels.iter().enumerate() {
        let image = h.d_gamma(&seed(input.clone()), limits).unwrap();
        for (i, output) in labels.iter().enumerate() {
            dense[(i, j)] = image.get(output).copied().unwrap_or_default();
        }
    }
    dense
}

#[test]
fn mixed_entry_and_occupied_packet_flow() {
    let packet = reference_packet();
    let initial = seed(label(0, 0, 0));
    let final_state = seed(label(1, 1, 1));
    let limits = ActionLimits::default();
    let empty = seed(JointOuterBasis::bosonic([]).unwrap());
    let coefficient = packet.entry;
    for g in [0.0, 1.0, -1.0, 4.0] {
        let h = packet.hamiltonian(g);
        let forward = joint_inner_product(&final_state, &h.d_gamma(&initial, limits).unwrap());
        let reverse = joint_inner_product(&initial, &h.d_gamma(&final_state, limits).unwrap());
        assert!((forward - g * coefficient).norm() < 1e-14);
        assert!((reverse - forward.conj()).norm() < 1e-14);
        assert!(h.d_gamma(&empty, limits).unwrap().is_empty());
    }
    let h = packet.hamiltonian(1.0);
    let t = 0.01;
    for cutoff in [3, 4] {
        let basis = window(cutoff);
        let labels: Vec<_> = basis.iter().cloned().collect();
        let dense = occupied_matrix(&h, &labels);
        assert!((&dense - dense.adjoint()).norm() < 1e-11);
        let reference = (dense * C::new(0.0, -t)).exp();
        let i = labels.iter().position(|v| *v == label(0, 0, 0)).unwrap();
        let p = reference[(i, i)].norm_sqr();
        for rank in [2, 3] {
            let options = JointSirkOptions {
                method: JointKrylovMethod::ShiftInvert {
                    relative_tolerance: 1e-13,
                    max_iterations: 100,
                },
                basis_window: Some(basis.clone()),
                ..JointSirkOptions::default()
            };
            let shifts = hashimoto_unitary_shifts(t, 4.0, 0.5, rank).unwrap();
            let result = solve_joint_sirk(&h, &initial, &shifts, &options).unwrap();
            assert_eq!(result.rank(), rank);
            let flow = result.reconstruct(&result.time_evolve(t).unwrap()).unwrap();
            let probability = joint_inner_product(&initial, &flow).norm_sqr();
            let max_tail = result
                .projected_action_tail_norms
                .iter()
                .copied()
                .fold(0.0, f64::max);
            assert!(max_tail > 0.0 && max_tail.is_finite());
            assert!((probability - p).abs() < 0.01);
            println!(
                "C={cutoff} m={rank} return={probability:.10} dense={p:.10} max_column_tail={max_tail:.8}"
            );
        }
    }
}

#[test]
fn global_colour_singlet_cubic_entry() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../docs/fixtures/qym_magnetic_packet.json")).unwrap();
    let k: f64 = fixture["entry_decimal"].as_str().unwrap().parse().unwrap();
    let expected: f64 = fixture["global_colour_packet"]["singlet_entry_at_g1"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    let mut singlet = NestedVector::new();
    let mut words = Vec::new();
    // For each cyclic choice of a, the two orders of b,c carry opposite signs.
    for a in 0..3 {
        let b = (a + 1) % 3;
        let c = (a + 2) % 3;
        for (j, l, sign) in [(b, c, 1.0), (c, b, -1.0)] {
            let modes = [a, 3 + j, 6 + l];
            let occupation = JointInnerBasis::new(modes.map(|mode| (mode, 1)), []).unwrap();
            let outer = JointOuterBasis::bosonic([(occupation, 1)]).unwrap();
            singlet.insert(outer, C::from(sign / 6_f64.sqrt()));
            words.push(
                InnerExpr::Product(modes.map(InnerExpr::coordinate).to_vec())
                    .scaled(C::from(k * sign)),
            );
        }
    }
    let initial = seed(label(0, 0, 0));
    let cubic = InnerExpr::Sum(words);
    let image = cubic.d_gamma(&initial, ActionLimits::default()).unwrap();
    let adjoint = cubic.d_gamma(&singlet, ActionLimits::default()).unwrap();
    assert_eq!(image.len(), 6);
    assert!((joint_inner_product(&singlet, &singlet) - 1.0).norm() < 1e-14);
    assert!((joint_inner_product(&singlet, &image) - expected).norm() < 1e-14);
    assert!((joint_inner_product(&initial, &adjoint) - expected).norm() < 1e-14);
}

/// The bounded target is kept fixed across ranks; the basis comes from the rational solver.
#[test]
fn fixed_shift_whole_window_resolvent_errors() {
    let h = reference_packet().hamiltonian(2.0);
    let basis = window(3);
    let labels: Vec<_> = basis.iter().cloned().collect();
    let dense = occupied_matrix(&h, &labels);
    let identity = DMatrix::<C>::identity(labels.len(), labels.len());
    let gamma = 4.0;
    let shifted = &dense + &identity * C::from(gamma);
    assert!(shifted.clone().cholesky().is_some());
    let positive = shifted.clone().try_inverse().unwrap();
    let complex = (&identity * C::from(gamma) + &dense * C::i())
        .try_inverse()
        .unwrap();
    let options = JointSirkOptions {
        basis_window: Some(basis),
        method: JointKrylovMethod::ShiftInvert {
            relative_tolerance: 1e-13,
            max_iterations: 100,
        },
        ..JointSirkOptions::default()
    };
    let operator_norm = |a: DMatrix<C>| a.singular_values().max();
    let mut initial_tails = None;
    let mut previous_projection: Option<DMatrix<C>> = None;
    let mut previous_error = f64::INFINITY;
    let mut initial_positive_error = None;
    for rank in 1..=3 {
        let result = solve_joint_sirk(
            &h,
            &seed(label(0, 0, 0)),
            &vec![C::i() * gamma; rank],
            &options,
        )
        .unwrap();
        assert_eq!(result.rank(), rank);
        let v = DMatrix::from_fn(labels.len(), rank, |i, j| {
            result.orthonormal_basis[j]
                .get(&labels[i])
                .copied()
                .unwrap_or_default()
        });
        let p = &v * v.adjoint();
        let q = &identity - &p;
        if let Some(previous) = &previous_projection {
            assert!(operator_norm(&q * previous) < 1e-10);
        }
        let rhs = v.adjoint();
        let galerkin = &v * (&rhs * &shifted * &v).cholesky().unwrap().solve(&rhs);
        let estimated_edge = operator_norm(galerkin.clone());
        let positive_error = operator_norm(&positive - galerkin);
        let positive_budget = *initial_positive_error.get_or_insert(positive_error);
        let mass_upper = 1.0 / estimated_edge - gamma;
        let mass_lower = 1.0 / (estimated_edge + positive_budget) - gamma;
        let exact_mass = dense.clone().symmetric_eigen().eigenvalues.min();
        assert!(mass_lower > 0.0 && mass_lower <= exact_mass + 1e-10);
        assert!(exact_mass <= mass_upper + 1e-10);
        println!(
            "absolute finite mass={exact_mass:.8} rank={rank} bracket=[{mass_lower:.8},{mass_upper:.8}]"
        );
        assert!(positive_error <= previous_error + 1e-11);
        previous_error = positive_error;
        previous_projection = Some(p);

        let head = result.projected_resolvent.as_ref().unwrap();
        let head_error = operator_norm(head - &rhs * &complex * &v);
        let complex_error = operator_norm(&complex - &v * head * rhs);
        let tail_sum = operator_norm(&q * &complex) + operator_norm(&complex * &q);
        let initial = *initial_tails.get_or_insert(tail_sum);
        assert!(head_error < 1e-10);
        assert!(tail_sum <= initial + 1e-10);
        assert!(complex_error <= initial + head_error + 1e-10);
        println!(
            "g=2 cutoff=3 rank={rank} positive={positive_error:.9} complex={complex_error:.9} initial_tails={initial:.9} head={head_error:.2e}"
        );
    }
}

/// Two transverse, colour-complete profiles in the chart standardized to unit variance.
#[test]
fn transverse_gauss_trial_and_longitudinal_extension() {
    use std::f64::consts::PI;
    let initial = seed(label(0, 0, 0));
    let empty = seed(JointOuterBasis::bosonic([]).unwrap());
    let limits = ActionLimits::default();
    let metric = 1.0 / (2.0 * PI * 2_f64.powf(1.5));
    let curl = 5.0 * PI * metric;
    let wall = 5.0 / (512.0 * PI * PI);
    for length in [1_f64, 2.0, 4.0, 8.0] {
        let q = |j| InnerExpr::coordinate(j).scaled(C::from(1.0 / length));
        let p = |j| InnerExpr::momentum(j).scaled(C::from(length));
        let sq = |v: InnerExpr| InnerExpr::Product(vec![v.clone(), v]);
        for g in [0.0, 2.0] {
            let mut terms = Vec::new();
            for j in 0..6 {
                terms.push(sq(p(j)).scaled(C::from(0.5 / (metric * length.powi(3)))));
                terms.push(sq(q(j)).scaled(C::from(0.5 * curl * length)));
            }
            for a in 0..3 {
                let b = (a + 1) % 3;
                let c = (a + 2) % 3;
                let wedge = InnerExpr::Sum(vec![
                    InnerExpr::Product(vec![q(b), q(3 + c)]),
                    InnerExpr::Product(vec![q(c), q(3 + b)]).scaled(C::from(-1.0)),
                ]);
                terms.push(sq(wedge).scaled(C::from(0.5 * g * g * wall * length.powi(3))));
            }
            let h = InnerExpr::Sum(terms);
            let image = h.d_gamma(&initial, limits).unwrap();
            let energy = joint_inner_product(&initial, &image);
            let expected = (6.0 / (8.0 * metric) + 3.0 * curl + 3.0 * g * g * wall) / length;
            assert!((energy - expected).norm() < 1e-12);
            assert!(h.d_gamma(&empty, limits).unwrap().is_empty());
        }

        // For the gauge parameter w T¹ both transverse colour rows rotate with this overlap.
        let mut rotations = Vec::new();
        for base in [0, 3] {
            rotations.push(InnerExpr::Product(vec![q(base + 2), p(base + 1)]));
            rotations
                .push(InnerExpr::Product(vec![q(base + 1), p(base + 2)]).scaled(C::from(-1.0)));
        }
        let rotation = InnerExpr::Sum(rotations).scaled(C::from(2.0 * (2_f64 / 3.0).powf(2.5)));
        let projected = rotation.d_gamma(&initial, limits).unwrap();
        assert!(joint_inner_product(&projected, &projected).norm() < 1e-24);
        // The added profile is (grad w)(x/L) T¹. Its raw coefficient translates by 1/L.
        let enlarged = InnerExpr::Sum(vec![rotation, p(6).scaled(C::from(1.0 / length))]);
        let image = enlarged.d_gamma(&initial, limits).unwrap();
        assert!((joint_inner_product(&image, &image) - 0.25).norm() < 1e-14);
    }
}
