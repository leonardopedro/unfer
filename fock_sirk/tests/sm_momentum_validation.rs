//! Validation for the continuous-momentum SM kernel (2026-11-24 work order).
//!
//! These tests bind the live module to the recorded fixture
//! `docs/fixtures/sm_momentum_kernel.json` and check the properties the
//! work order names: a nonzero mixed matrix element between two distinct
//! smooth momentum packets, the `2π` transfer convention, the nonabelian sign,
//! the gauge–Higgs cross term, the Yukawa adjoint, and a finite
//! one-outer-particle evolution with a measured error diagnostic.
//!
//! Scope: these are algebraically exact packet entries and a measured
//! packet-basis evolution. They are **not** a continuum/ESA/gauge-reduction
//! result and not a `joint_sirk` run on the complete inner operator.

use nalgebra::DMatrix;
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, JointInnerBasis, JointOuterBasis, NestedVector,
};
use nested_fock_algebra::sm_full::SmParameters;
use nested_fock_algebra::sm_momentum::{
    Conventions, GaugeMode, HiggsMode, MomentumPacket, TWO_PI, dirac_packet_element,
    gauge_magnetic_cross, gauge_magnetic_free, higgs_gauge_cross, higgs_potential_constant,
    scalar_gradient_symbol, transfer_integral, transverse_curl_projector, yukawa_transfer,
};
use num_complex::Complex64 as C;
use serde_json::Value;
use std::path::Path;

fn parameters() -> SmParameters {
    let mut yu = [[C::new(0.0, 0.0); 3]; 3];
    yu[0][0] = C::new(0.7, 0.2);
    yu[1][1] = C::new(0.5, -0.1);
    yu[2][2] = C::new(0.9, 0.0);
    SmParameters {
        gs: 1.1,
        gw: 0.65,
        gy: 0.36,
        lambda: 0.13,
        vev: 246.0,
        yu,
        yd: [[C::new(0.0, 0.0); 3]; 3],
        ynu: [[C::new(0.0, 0.0); 3]; 3],
        ye: [[C::new(0.0, 0.0); 3]; 3],
    }
}

fn packet(center: [f64; 3], width: f64) -> MomentumPacket {
    MomentumPacket::new(center, width)
}

fn fixture() -> Value {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/fixtures/sm_momentum_kernel.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture json")
}

fn cplx(v: &Value) -> C {
    C::new(v["re"].as_f64().expect("re"), v["im"].as_f64().expect("im"))
}

fn close(a: C, b: C, tol: f64, label: &str) {
    assert!(
        (a - b).norm() <= tol,
        "{label}: live {a} vs fixture {b} (tol {tol})"
    );
}

fn live_elements() -> (C, C, C, f64) {
    let sm = parameters();
    let conv = Conventions::canonical();
    let packets = [
        packet([0.40, 0.00, 0.00], 0.80),
        packet([-0.20, 0.30, 0.00], 0.90),
        packet([0.10, 0.10, -0.20], 0.60),
    ];
    let bra = GaugeMode::new(1, 2, packets[0]);
    let ket = GaugeMode::new(5, 1, packets[1]);
    let bg = GaugeMode::new(3, 2, packets[2]);
    let magnetic = gauge_magnetic_cross(&bra, &ket, &bg, &sm, &conv);
    let weak_bg = GaugeMode::new(1, 0, packets[2]);
    let higgs_bra = HiggsMode::new(0, packets[0]);
    let higgs_ket = HiggsMode::new(2, packets[1]);
    let higgs = higgs_gauge_cross(&higgs_bra, &higgs_ket, &weak_bg, false, &sm, &conv);
    let yukawa = yukawa_transfer(sm.yu[0][0], &packets[2], &packets[0], &packets[1], &conv);
    let transfer = transfer_integral(&packets[0], &packets[2], &packets[1], &conv);
    (magnetic, higgs, yukawa, transfer)
}

#[test]
fn fixture_matches_the_live_kernel_entry_by_entry() {
    let fixture = fixture();
    let (magnetic, higgs, yukawa, transfer) = live_elements();
    assert_eq!(fixture["schema"], "sm-momentum-kernel/v1");
    close(
        cplx(&fixture["elements"]["magnetic_cross"]),
        magnetic,
        1e-10,
        "magnetic",
    );
    close(
        cplx(&fixture["elements"]["gauge_higgs_cross"]),
        higgs,
        1e-10,
        "higgs",
    );
    close(
        cplx(&fixture["elements"]["yukawa_transfer"]),
        yukawa,
        1e-10,
        "yukawa",
    );
    assert!((fixture["elements"]["transfer_integral"].as_f64().unwrap() - transfer).abs() < 1e-10);
    assert!(
        fixture["elements"]["higgs_potential_constant"]
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[test]
fn fixture_records_the_spatial_and_lift_checks() {
    let fixture = fixture();
    let conv = Conventions::canonical();
    let a = packet([0.40, 0.00, 0.00], 0.80);
    let b = packet([-0.20, 0.30, 0.00], 0.90);
    let right = dirac_packet_element(&a, 0, &b, 1, true, &conv);
    let left = dirac_packet_element(&a, 0, &b, 1, false, &conv);
    close(
        cplx(&fixture["elements"]["dirac_right"]),
        right,
        1e-10,
        "dirac right",
    );
    close(
        cplx(&fixture["elements"]["dirac_left"]),
        left,
        1e-10,
        "dirac left",
    );
    assert!((right + left).norm() < 1e-12, "chiral signs are opposite");
    assert!(
        right.norm() > 1e-8,
        "distinct packets give a nonzero Dirac entry"
    );

    let k = [0.7, -0.2, 0.4];
    let e1 = [1.0, 0.0, 0.0];
    assert!(transverse_curl_projector(k, k, k).abs() < 1e-12);
    assert!(
        (fixture["elements"]["longitudinal_curl"].as_f64().unwrap()
            - transverse_curl_projector(k, k, k))
        .abs()
            < 1e-12
    );
    assert!(
        (fixture["elements"]["transverse_curl"].as_f64().unwrap()
            - transverse_curl_projector(k, e1, e1))
        .abs()
            < 1e-12
    );
    assert!(
        (fixture["elements"]["scalar_gradient"].as_f64().unwrap() - scalar_gradient_symbol(k))
            .abs()
            < 1e-12
    );

    let lift = &fixture["outer_lift"];
    let c = cplx(&lift["inner_constant"]);
    assert_eq!(lift["vacuum_terms"].as_u64().unwrap(), 0);
    close(cplx(&lift["one_quantum"]), c, 1e-12, "one quantum -> c");
    close(
        cplx(&lift["two_quanta"]),
        2.0 * c,
        1e-12,
        "two quanta -> 2c",
    );
    assert!(lift["ordered_product_difference"].as_f64().unwrap() > 1e-8);
}

#[test]
fn two_distinct_packets_give_a_nonzero_mixed_element() {
    let sm = parameters();
    let conv = Conventions::canonical();
    let a = packet([0.40, 0.00, 0.00], 0.80);
    let b = packet([-0.20, 0.30, 0.00], 0.90);
    assert_ne!(a.center, b.center, "the two packets must be distinct");
    let bg = GaugeMode::new(3, 2, packet([0.10, 0.10, -0.20], 0.60));
    let value = gauge_magnetic_cross(
        &GaugeMode::new(1, 2, a),
        &GaugeMode::new(5, 1, b),
        &bg,
        &sm,
        &conv,
    );
    assert!(value.norm() > 1e-8, "mixed SM element is nonzero: {value}");
    // Zero coupling kills the nonabelian transfer.
    let mut free = sm.clone();
    free.gs = 0.0;
    let zero = gauge_magnetic_cross(
        &GaugeMode::new(1, 2, a),
        &GaugeMode::new(5, 1, b),
        &bg,
        &free,
        &conv,
    );
    assert!(zero.norm() < 1e-14, "abelian/zero-coupling limit vanishes");
}

#[test]
fn mutations_are_detected_by_the_fixture() {
    let fixture = fixture();
    let sm = parameters();
    let conv = Conventions::canonical();
    let packets = [
        packet([0.40, 0.00, 0.00], 0.80),
        packet([-0.20, 0.30, 0.00], 0.90),
        packet([0.10, 0.10, -0.20], 0.60),
    ];
    let bra = GaugeMode::new(1, 2, packets[0]);
    let ket = GaugeMode::new(5, 1, packets[1]);
    let bg = GaugeMode::new(3, 2, packets[2]);
    let base = gauge_magnetic_cross(&bra, &ket, &bg, &sm, &conv);

    let no_two_pi = cplx(&fixture["mutations"]["drop_two_pi"]);
    assert!((no_two_pi - base).norm() > 1e-8, "2π mutation must differ");

    let flipped = cplx(&fixture["mutations"]["flip_nonabelian_sign"]);
    assert!(
        (flipped + base).norm() < 1e-10,
        "sign mutation must negate the element"
    );

    let removed = cplx(&fixture["mutations"]["remove_higgs_cross"]);
    assert_eq!(removed, C::new(0.0, 0.0));

    let reversed = fixture["mutations"]["reverse_transfer"].as_f64().unwrap();
    let forward = transfer_integral(&packets[0], &packets[2], &packets[1], &conv);
    assert!(
        (reversed - forward).abs() > 1e-9,
        "transfer reversal must differ"
    );

    let adjoint = cplx(&fixture["mutations"]["drop_yukawa_adjoint"]);
    let y = sm.yu[0][0];
    close(
        adjoint,
        yukawa_transfer(
            y,
            &packets[2],
            &packets[0],
            &packets[1],
            &Conventions {
                yukawa_adjoint: true,
                ..conv
            },
        ),
        1e-12,
        "adjoint mutation",
    );

    let kept = fixture["mutations"]["drop_higgs_constant"]["kept"]
        .as_f64()
        .unwrap();
    assert!((kept - higgs_potential_constant(&sm)).abs() < 1e-6);
    assert!(kept > 0.0, "the λv⁴/4 constant is retained");
}

#[test]
fn conventions_are_the_declared_fourier_ones() {
    let conv = Conventions::canonical();
    assert_eq!(conv.two_pi, TWO_PI);
    assert_eq!(conv.transfer_sign, 1.0);
    assert_eq!(conv.nonabelian_sign, 1.0);
    assert!(conv.include_higgs_cross);
    assert!(!conv.yukawa_adjoint);
}

#[test]
fn outer_shift_lifts_to_zero_one_and_two_quanta() {
    // The work order's `0, c, 2c` check: an inner constant `c I` lifts to
    // `dΓ(cI) = c N_out`, not to `c I_out`.
    let c = C::new(1.75, -0.5);
    let h = InnerExpr::Scalar(c);
    let inner = JointInnerBasis::vacuum();
    let limits = ActionLimits::default();

    let vacuum = NestedVector::new();
    assert!(
        h.d_gamma(&vacuum, limits)
            .expect("vacuum action")
            .is_empty()
    );

    let mut one = NestedVector::new();
    let one_label = JointOuterBasis::bosonic([(inner.clone(), 1)]).expect("one label");
    one.insert(one_label.clone(), C::new(1.0, 0.0));
    let one_image = h.d_gamma(&one, limits).expect("one-quantum action");
    assert_eq!(one_image.len(), 1);
    assert!(
        (one_image[&one_label] - c).norm() < 1e-12,
        "one quantum -> c"
    );

    let mut two = NestedVector::new();
    let two_label = JointOuterBasis::bosonic([(inner, 2)]).expect("two label");
    two.insert(two_label.clone(), C::new(1.0, 0.0));
    let two_image = h.d_gamma(&two, limits).expect("two-quantum action");
    assert!(
        (two_image[&two_label] - 2.0 * c).norm() < 1e-12,
        "two quanta -> 2c"
    );
}

#[test]
fn ordered_inner_product_differs_from_the_lifted_product() {
    // `dΓ(B²) ≠ dΓ(B)²` for occupation ≥ 2: the inner product must be formed
    // before the single outer lift.
    let limits = ActionLimits::default();
    let b = InnerExpr::coordinate(0);
    let b_squared = InnerExpr::Product(vec![b.clone(), b.clone()]);
    let inner = JointInnerBasis::vacuum();
    let mut seed = NestedVector::new();
    seed.insert(
        JointOuterBasis::bosonic([(inner, 2)]).expect("two-quantum label"),
        C::new(1.0, 0.0),
    );
    let lifted_product = b_squared.d_gamma(&seed, limits).expect("dΓ(B²)");
    let lifted_twice = b
        .d_gamma(&b.d_gamma(&seed, limits).expect("dΓ(B)"), limits)
        .expect("dΓ(B)²");
    let mut difference = 0.0f64;
    for (label, amplitude) in &lifted_product {
        let other = lifted_twice.get(label).copied().unwrap_or_default();
        difference = difference.max((amplitude - other).norm());
    }
    for (label, amplitude) in &lifted_twice {
        if !lifted_product.contains_key(label) {
            difference = difference.max(amplitude.norm());
        }
    }
    assert!(
        difference > 1e-8,
        "dΓ(B²) must differ from dΓ(B)² on a two-quantum state, got {difference}"
    );
}

#[test]
fn one_outer_particle_evolution_has_a_measured_error_budget() {
    let sm = parameters();
    let conv = Conventions::canonical();
    let packets = [
        packet([0.40, 0.00, 0.00], 0.80),
        packet([-0.20, 0.30, 0.00], 0.90),
        packet([0.10, 0.10, -0.20], 0.60),
    ];
    let bg = GaugeMode::new(3, 2, packets[2]);
    let n = packets.len();
    let mut gram = DMatrix::<f64>::zeros(n, n);
    let mut kernel = DMatrix::<C>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            gram[(i, j)] = MomentumPacket::overlap(&packets[i], &packets[j]);
            let free = gauge_magnetic_free(
                &GaugeMode::new(1, i % 3, packets[i]),
                &GaugeMode::new(1, j % 3, packets[j]),
                &conv,
            );
            let cross = gauge_magnetic_cross(
                &GaugeMode::new(1, i % 3, packets[i]),
                &GaugeMode::new(5, j % 3, packets[j]),
                &bg,
                &sm,
                &conv,
            );
            kernel[(i, j)] = free + cross;
        }
    }
    let hermitian = (&kernel + kernel.adjoint()) * C::new(0.5, 0.0);
    assert!((&hermitian - hermitian.adjoint()).norm() < 1e-12);

    let eigen = gram.clone().symmetric_eigen();
    let mut inv = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        assert!(eigen.eigenvalues[i] > 0.0, "Gram positive definite");
        inv[(i, i)] = 1.0 / eigen.eigenvalues[i].sqrt();
    }
    let s_inv =
        (&eigen.eigenvectors * inv * eigen.eigenvectors.transpose()).map(|x| C::new(x, 0.0));
    let h = &s_inv * &hermitian * &s_inv;

    let time = 0.01;
    let full = (h.clone() * C::new(0.0, -time)).exp();
    let basis = |i: usize| {
        nalgebra::DVector::from_iterator(
            n,
            (0..n).map(|j| C::new(if j == i { 1.0 } else { 0.0 }, 0.0)),
        )
    };
    let seed = basis(0);
    let evolved = &full * &seed;
    let return_full = evolved[0].norm_sqr();

    let v = DMatrix::<C>::from_columns(&[basis(0), basis(1)]);
    let g_red = v.adjoint() * &h * &v;
    let reduced = (g_red.clone() * C::new(0.0, -time)).exp();
    let compressed = &v * (&reduced * (&v.adjoint() * &seed));
    let return_compressed = compressed[0].norm_sqr();

    let defect = (&v * (&g_red * C::i()) - &h * &v).norm();
    let leakage = (&seed - &v * (&v.adjoint() * &seed)).norm();
    let measured_gap = (evolved - compressed).norm();
    let bound = leakage + time * defect * (&v.adjoint() * &seed).norm();

    assert!((return_full - return_compressed).abs() > 0.0);
    assert!(
        bound >= measured_gap,
        "measured first-order bound {bound} must cover the map gap {measured_gap}"
    );
    // The fixture records the same cell; check the recorded numbers are live.
    let fixture = fixture();
    let recorded = &fixture["evolution"];
    assert!((recorded["return_full"].as_f64().unwrap() - return_full).abs() < 1e-10);
    assert!((recorded["return_compressed"].as_f64().unwrap() - return_compressed).abs() < 1e-10);
    assert!(
        (recorded["first_order_bound"].as_f64().unwrap() - bound).abs() < 1e-10,
        "fixture bound {} vs live {bound}",
        recorded["first_order_bound"]
    );
}
