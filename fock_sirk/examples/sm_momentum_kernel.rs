//! Emit the continuous-momentum SM kernel fixture
//! (`docs/fixtures/sm_momentum_kernel.json`) for the 2026-11-24 work order.
//!
//! Run with:
//!
//! ```text
//! cargo run -p fock_sirk --example sm_momentum_kernel
//! ```
//!
//! The fixture records the declared Fourier conventions, the packet data, the
//! packet matrix elements, the five mutation controls, and one finite
//! packet-basis one-outer-particle evolution cell. It is a *measured* record:
//! the error column is a measured defect/leakage diagnostic, not a verified
//! enclosure. See `docs/SM_FULL_IMPLEMENTATION.md` for the ledger entry.

use fock_sirk::sm_bridge::reported_residual;
use nalgebra::{DMatrix, DVector};
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, JointInnerBasis, JointOuterBasis, NestedVector,
};
use nested_fock_algebra::sm_full::SmParameters;
use nested_fock_algebra::sm_momentum::{
    Conventions, GaugeMode, HiggsMode, MomentumPacket, dirac_packet_element, gauge_magnetic_cross,
    gauge_magnetic_free, higgs_gauge_cross, higgs_potential_constant, scalar_gradient_symbol,
    transfer_integral, transverse_curl_projector, yukawa_transfer,
};
use num_complex::Complex64 as C;
use serde_json::{Value, json};

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

fn complex_json(x: C) -> Value {
    json!({"re": x.re, "im": x.im})
}

/// A measured complex element at twelve significant digits per component.
///
/// Declared inputs (`parameters`, `packets`, `conventions.two_pi`) and the
/// `inner_constant` keep full precision: they are quoted, not measured. Every
/// evaluator output goes through `reported_residual` so a rebuild on another
/// toolchain does not churn the record in its last digits.
fn measured_complex_json(x: C) -> Value {
    json!({
        "re": reported_residual(x.re),
        "im": reported_residual(x.im),
    })
}

/// A measured real scalar at twelve significant digits.
fn measured(x: f64) -> f64 {
    reported_residual(x)
}

/// `S^{-1/2}` for a Hermitian positive-definite Gram matrix, via eigendecomposition
/// (the same inverse-free route the rational solver uses).
fn inverse_sqrt(m: &DMatrix<f64>) -> DMatrix<f64> {
    let eigen = m.clone().symmetric_eigen();
    let mut inv = DMatrix::<f64>::zeros(m.nrows(), m.ncols());
    for i in 0..m.nrows() {
        assert!(eigen.eigenvalues[i] > 0.0, "Gram must be positive definite");
        inv[(i, i)] = 1.0 / eigen.eigenvalues[i].sqrt();
    }
    &eigen.eigenvectors * inv * eigen.eigenvectors.transpose()
}

fn main() {
    let sm = parameters();
    let conv = Conventions::canonical();

    // Two distinct smooth packets and a third used as the background.
    let packets = [
        packet([0.40, 0.00, 0.00], 0.80),
        packet([-0.20, 0.30, 0.00], 0.90),
        packet([0.10, 0.10, -0.20], 0.60),
    ];

    // Nonabelian magnetic cross term: colours 1,5 with background colour 3,
    // chosen so both structure constants are nonzero.
    let bra = GaugeMode::new(1, 2, packets[0]);
    let ket = GaugeMode::new(5, 1, packets[1]);
    let bg = GaugeMode::new(3, 2, packets[2]);
    let magnetic = gauge_magnetic_cross(&bra, &ket, &bg, &sm, &conv);

    // Gauge–Higgs mixing (weak isospin, so a weak adjoint index 0..2) between
    // components 0 and 2.
    let weak_bg = GaugeMode::new(1, 0, packets[2]);
    let higgs_bra = HiggsMode::new(0, packets[0]);
    let higgs_ket = HiggsMode::new(2, packets[1]);
    let higgs = higgs_gauge_cross(&higgs_bra, &higgs_ket, &weak_bg, false, &sm, &conv);

    // Yukawa transfer between the two smooth packets with a background.
    let yukawa = yukawa_transfer(sm.yu[0][0], &packets[2], &packets[0], &packets[1], &conv);
    let transfer = transfer_integral(&packets[0], &packets[2], &packets[1], &conv);

    // Mutation controls: each flips exactly one declared convention.
    let mutations = json!({
        "drop_two_pi": measured_complex_json(gauge_magnetic_cross(
            &bra, &ket, &bg, &sm,
            &Conventions { two_pi: 1.0, ..conv },
        )),
        "flip_nonabelian_sign": measured_complex_json(gauge_magnetic_cross(
            &bra, &ket, &bg, &sm,
            &Conventions { nonabelian_sign: -1.0, ..conv },
        )),
        "remove_higgs_cross": measured_complex_json(higgs_gauge_cross(
            &higgs_bra, &higgs_ket, &weak_bg, false, &sm,
            &Conventions { include_higgs_cross: false, ..conv },
        )),
        "reverse_transfer": measured(transfer_integral(
            &packets[0], &packets[2], &packets[1],
            &Conventions { transfer_sign: -1.0, ..conv },
        )),
        // Raw, not `measured_complex_json`: `sm_momentum_validation` pins this
        // control at 1e-12 absolute, which is stricter than twelve significant
        // digits (rounding moves it by ~1.6e-12), so rounding it would weaken
        // a check rather than normalize a record.
        "drop_yukawa_adjoint": complex_json(yukawa_transfer(
            sm.yu[0][0], &packets[2], &packets[0], &packets[1],
            &Conventions { yukawa_adjoint: true, ..conv },
        )),
        "drop_higgs_constant": {
            "kept": measured(higgs_potential_constant(&sm)),
            "dropped": 0.0,
        },
    });

    // Independent spatial checks: the Dirac symbol/chiral signs, the
    // transverse curl projector, the scalar-gradient normalization.
    let dirac_right = dirac_packet_element(&packets[0], 0, &packets[1], 1, true, &conv);
    let dirac_left = dirac_packet_element(&packets[0], 0, &packets[1], 1, false, &conv);
    let k = [0.7, -0.2, 0.4];
    let e1 = [1.0, 0.0, 0.0];
    let transverse = transverse_curl_projector(k, e1, e1);
    let longitudinal = transverse_curl_projector(k, k, k);

    // Outer shift lift: an inner constant cI becomes c N_out (0, c, 2c).
    let shift_c = C::new(1.75, -0.5);
    let shift_expr = InnerExpr::Scalar(shift_c);
    let limits = ActionLimits::default();
    let inner = JointInnerBasis::vacuum();
    let shift_action = |n: u32| {
        let mut state = NestedVector::new();
        state.insert(
            JointOuterBasis::bosonic([(inner.clone(), n)]).expect("outer label"),
            C::new(1.0, 0.0),
        );
        shift_expr.d_gamma(&state, limits).expect("shift lift")
    };
    let one_coefficient = shift_action(1).values().next().copied().unwrap_or_default();
    let two_coefficient = shift_action(2).values().next().copied().unwrap_or_default();
    let vacuum_image = shift_expr
        .d_gamma(&NestedVector::new(), limits)
        .expect("vacuum")
        .len();

    // Ordered product: dΓ(B²) vs dΓ(B)² on a two-quantum state.
    let b = InnerExpr::coordinate(0);
    let b_squared = InnerExpr::Product(vec![b.clone(), b.clone()]);
    let mut ordered_seed = NestedVector::new();
    ordered_seed.insert(
        JointOuterBasis::bosonic([(JointInnerBasis::vacuum(), 2)]).expect("two quanta"),
        C::new(1.0, 0.0),
    );
    let lifted_product = b_squared.d_gamma(&ordered_seed, limits).expect("dΓ(B²)");
    let lifted_twice = b
        .d_gamma(&b.d_gamma(&ordered_seed, limits).expect("dΓ(B)"), limits)
        .expect("dΓ(B)²");
    let mut ordered_difference = 0.0f64;
    for (label, amplitude) in &lifted_product {
        let other = lifted_twice.get(label).copied().unwrap_or_default();
        ordered_difference = ordered_difference.max((amplitude - other).norm());
    }
    for (label, amplitude) in &lifted_twice {
        if !lifted_product.contains_key(label) {
            ordered_difference = ordered_difference.max(amplitude.norm());
        }
    }

    // One-outer-particle evolution on the finite packet basis.
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
    // Hermitian part of the induced kernel.
    let hermitian = (&kernel + kernel.adjoint()) * C::new(0.5, 0.0);
    let hermiticity_defect = (&hermitian - hermitian.adjoint()).norm();

    // Orthonormal packet basis: H' = S^{-1/2} H S^{-1/2}.
    let s_inv = inverse_sqrt(&gram).map(|x| C::new(x, 0.0));
    let h_orth = &s_inv * &hermitian * &s_inv;
    let time = 0.01;
    let full = (h_orth.clone() * C::new(0.0, -time)).exp();
    let unit = |i: usize| C::new(if i == 0 { 1.0 } else { 0.0 }, 0.0);
    let seed = DVector::from_iterator(n, (0..n).map(unit));
    let evolved = &full * &seed;
    let return_full = evolved[0].norm_sqr();

    // A rank-2 window (first two orthonormal packet directions).
    let basis = |i: usize| {
        DVector::from_iterator(
            n,
            (0..n).map(|j| C::new(if j == i { 1.0 } else { 0.0 }, 0.0)),
        )
    };
    let v = DMatrix::<C>::from_columns(&[basis(0), basis(1)]);
    let g_red = v.adjoint() * &h_orth * &v;
    let reduced_flow = (g_red.clone() * C::new(0.0, -time)).exp();
    let coefficients = v.adjoint() * &seed;
    let compressed = &v * (&reduced_flow * &coefficients);
    let return_compressed = compressed[0].norm_sqr();
    // D = i V G - H V, first order map bound on the window.
    let defect = {
        let term = &v * (&g_red * C::i()) - &h_orth * &v;
        term.norm()
    };
    let leakage = (&seed - &v * (&v.adjoint() * &seed)).norm();
    let measured_gap = (evolved - &compressed).norm();

    let fixture = json!({
        "schema": "sm-momentum-kernel/v1",
        "source": "nested_fock_algebra::sm_momentum (2026-11-24 work order)",
        "status": "measured",
        "conventions": {
            "transform": "f_hat(k) = \\int e^{-2\\pi i k.x} f(x) d^3x",
            "derivative": "d_i f -> 2\\pi i k_i f_hat(k)",
            "convolution": "(a*b)(K) = \\int a(q) b(K-q) d^3q",
            "adjoint": "f_hat_dagger(q) = f_hat(-q)_dagger",
            "two_pi": conv.two_pi,
            "transfer": "k-l",
        },
        "parameters": {
            "gs": sm.gs, "gw": sm.gw, "gy": sm.gy,
            "lambda": sm.lambda, "vev": sm.vev,
            "yu00": complex_json(sm.yu[0][0]),
        },
        "packets": packets.iter().map(|p| json!({
            "center": p.center, "width": p.width,
        })).collect::<Vec<_>>(),
        "elements": {
            "magnetic_cross": measured_complex_json(magnetic),
            "gauge_higgs_cross": measured_complex_json(higgs),
            "yukawa_transfer": measured_complex_json(yukawa),
            "transfer_integral": measured(transfer),
            "higgs_potential_constant": measured(higgs_potential_constant(&sm)),
            "dirac_right": measured_complex_json(dirac_right),
            "dirac_left": measured_complex_json(dirac_left),
            "transverse_curl": measured(transverse),
            "longitudinal_curl": measured(longitudinal),
            "scalar_gradient": measured(scalar_gradient_symbol(k)),
        },
        "outer_lift": {
            "inner_constant": complex_json(shift_c),
            "vacuum_terms": vacuum_image,
            "one_quantum": measured_complex_json(one_coefficient),
            "two_quanta": measured_complex_json(two_coefficient),
            "ordered_product_difference": measured(ordered_difference),
        },
        "mutations": mutations,
        "evolution": {
            "time": time,
            "window_dim": 2,
            "full_dim": n,
            "gram_frobenius": measured(gram.norm()),
            "hermiticity_defect": measured(hermiticity_defect),
            "return_full": measured(return_full),
            "return_compressed": measured(return_compressed),
            "return_gap": measured((return_full - return_compressed).abs()),
            "leakage": measured(leakage),
            "defect_frobenius": measured(defect),
            "first_order_bound": measured(leakage + time * defect * coefficients.norm()),
            "measured_map_gap": measured(measured_gap),
            "label": "measured finite packet-basis diagnostic; not a certified enclosure",
        },
        "non_claims": [
            "no continuum or gauge-reduction claim",
            "no smearing removal or self-adjointness proof",
            "no joint_sirk run on the complete inner operator",
            "the cadabra2 symbolic reference has not been executed on this host",
        ],
    });

    let path = std::path::Path::new("docs/fixtures/sm_momentum_kernel.json");
    std::fs::write(path, serde_json::to_string_pretty(&fixture).unwrap() + "\n")
        .expect("write fixture");
    eprintln!("wrote {}", path.display());
}
