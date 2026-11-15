//! N8 finite fixture and certificate-reader checks.

use fock_sirk::sm_defect::*;

#[test]
fn finite_fixture_encloses_the_direct_map_error() {
    let (point, interval, measured) = finite_fixture(0.2);
    assert!(point.initial_leakage.is_finite());
    assert!(point.defect_frobenius > 0.0);
    assert!(point.seed_norm > 0.0);
    assert!(point.evolution_bound >= point.initial_leakage);
    assert!(
        measured <= point.evolution_bound + 1e-12,
        "measured {measured:.6e} exceeds point bound {:.6e}",
        point.evolution_bound
    );
    assert!(
        measured <= interval.evolution_bound.hi + 1e-12,
        "measured {measured:.6e} exceeds interval bound {:.6e}",
        interval.evolution_bound.hi
    );
    for value in [
        interval.initial_leakage,
        interval.defect_frobenius,
        interval.seed_norm,
        interval.time,
    ] {
        assert!(value.lo.is_finite() && value.lo <= value.hi);
    }
    assert!(interval.evolution_bound.lo <= interval.evolution_bound.hi);
    // The interval path is outward: it must not be tighter than the point path.
    assert!(interval.evolution_bound.hi + 1e-15 >= point.evolution_bound);
    // At t=0 the compressed map is the initial projection, so the direct error
    // is exactly the initial leakage.
    let (at_zero, _, measured_at_zero) = finite_fixture(0.0);
    assert!((measured_at_zero - at_zero.initial_leakage).abs() < 1e-12);
}

#[test]
fn certificate_reader_rejects_missing_negative_nonfinite_and_mismatched_hashes() {
    let (terms, _, _) = finite_fixture(0.2);
    let hashes = CertificateHashes {
        operator: "a".repeat(64),
        state: "b".repeat(64),
        basis: "c".repeat(64),
    };
    let certificate = certificate_json(&terms, &hashes);
    verify_certificate(&certificate, &hashes).expect("well-formed certificate");

    let mut missing = certificate.clone();
    missing["fields"]
        .as_object_mut()
        .unwrap()
        .remove("evolution_bound");
    assert!(matches!(
        verify_certificate(&missing, &hashes),
        Err(DefectError::Missing("evolution_bound"))
    ));

    let mut negative = certificate.clone();
    negative["fields"]["defect_frobenius"] = serde_json::json!(-1.0);
    assert!(matches!(
        verify_certificate(&negative, &hashes),
        Err(DefectError::NegativeOrNonfinite("defect_frobenius"))
    ));

    let mut nonfinite = certificate.clone();
    nonfinite["fields"]["seed_norm"] = serde_json::json!(f64::INFINITY);
    assert!(matches!(
        verify_certificate(&nonfinite, &hashes),
        Err(DefectError::Missing("seed_norm"))
    ));

    let mut zero_time = certificate.clone();
    zero_time["fields"]["time"] = serde_json::json!(0.0);
    assert!(matches!(
        verify_certificate(&zero_time, &hashes),
        Err(DefectError::NegativeOrNonfinite("time"))
    ));

    let mut below = certificate.clone();
    below["fields"]["evolution_bound"] = serde_json::json!(0.0);
    assert!(matches!(
        verify_certificate(&below, &hashes),
        Err(DefectError::BoundBelowLeakage)
    ));

    let mut moved = hashes.clone();
    moved.operator = "d".repeat(64);
    assert!(matches!(
        verify_certificate(&certificate, &moved),
        Err(DefectError::HashMismatch("operator"))
    ));

    let mut promoted = certificate;
    promoted["status"] = serde_json::json!("certified");
    assert!(matches!(
        verify_certificate(&promoted, &hashes),
        Err(DefectError::Status)
    ));
}
