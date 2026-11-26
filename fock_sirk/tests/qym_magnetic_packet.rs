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

#[test]
fn mixed_entry_and_occupied_packet_flow() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../docs/fixtures/qym_magnetic_packet.json")).unwrap();
    let coeff = &fixture["packet_action"];
    let packet = Packet {
        entry: fixture["entry_decimal"].as_str().unwrap().parse().unwrap(),
        packet_action: Coefficients {
            profile_metric: serde_json::from_value(coeff["profile_metric"].clone()).unwrap(),
            curl_norm_squared: serde_json::from_value(coeff["curl_norm_squared"].clone()).unwrap(),
            wall_12_13: serde_json::from_value(coeff["wall_12_13"].clone()).unwrap(),
        },
    };
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
        let mut dense = DMatrix::<C>::zeros(labels.len(), labels.len());
        for (j, input) in labels.iter().enumerate() {
            let image = h.d_gamma(&seed(input.clone()), limits).unwrap();
            for (i, output) in labels.iter().enumerate() {
                dense[(i, j)] = image.get(output).copied().unwrap_or_default();
            }
        }
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
