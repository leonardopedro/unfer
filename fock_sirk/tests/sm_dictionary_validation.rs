//! Acceptance test for the full Standard-Model dictionary kernel.
//!
//! Covers all **1,241** exported component expressions: every row is evaluated
//! in the momentum and coordinate representations of the E1
//! collective-coordinate operator and must agree to roundoff (Plancherel). The
//! checked-in fixture `docs/fixtures/sm_dictionary_kernel.json` is regenerated
//! by `cargo run -p fock_sirk --example sm_dictionary_kernel`.

use nested_fock_algebra::sm_dictionary::{
    DictionaryPackets, Q_COORDINATES, coordinate_kernel, dictionary, fixture_parameters,
    momentum_kernel, worst_gap,
};
use nested_fock_algebra::sm_momentum::Conventions;
use serde_json::Value;

const FIXTURE: &str = include_str!("../../docs/fixtures/sm_dictionary_kernel.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("dictionary kernel fixture must be JSON")
}

fn pair(value: &Value) -> (f64, f64) {
    (
        value[0].as_f64().unwrap(),
        value[1].as_f64().unwrap(),
    )
}

#[test]
fn fixture_declares_the_full_dictionary() {
    let doc = fixture();
    assert_eq!(doc["schema"], "sm-dictionary-kernel/v1");
    assert_eq!(doc["rows"], 1241);
    assert_eq!(doc["collective_dimensions"], Q_COORDINATES);
    let families = &doc["families"];
    for (name, count) in [
        ("kinetic", 40),
        ("magnetic-strong", 24),
        ("magnetic-weak", 9),
        ("magnetic-hypercharge", 3),
        ("covariant-higgs", 12),
        ("higgs-potential", 1),
        ("matter", 1152),
    ] {
        assert_eq!(families[name], count, "family {name}");
    }
    assert_eq!(doc["entries"].as_array().unwrap().len(), 1241);
    // The fixture binds fully populated Yukawa matrices, so an accidentally
    // zero row would mean a parser/coefficient regression, not a zero coupling.
    assert!(
        doc["nonzero_momentum_rows"].as_u64().unwrap() >= 1240,
        "only {} nonzero rows",
        doc["nonzero_momentum_rows"]
    );
}

#[test]
fn fixture_rows_match_the_parsed_dictionary_in_order() {
    let doc = fixture();
    let entries = dictionary();
    let labels: Vec<String> = entries.iter().map(|e| e.label.clone()).collect();
    let fixture_labels: Vec<String> = doc["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["label"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(labels, fixture_labels);
    for (entry, row) in entries.iter().zip(doc["entries"].as_array().unwrap()) {
        assert_eq!(row["family"], entry.family.as_str(), "{}", entry.label);
        assert_eq!(
            row["terms"].as_u64().unwrap() as usize,
            entry.terms.len(),
            "{}",
            entry.label
        );
    }
}

#[test]
fn every_row_agrees_between_the_two_representations() {
    let doc = fixture();
    let packets = DictionaryPackets::declared();
    let params = fixture_parameters();
    let conv = Conventions::canonical();
    let entries = dictionary();
    let mut worst: (f64, &str) = (0.0, "");
    for (entry, row) in entries.iter().zip(doc["entries"].as_array().unwrap()) {
        let momentum = momentum_kernel(entry, &packets, &params, &conv);
        let coordinate = coordinate_kernel(entry, &packets, &params);
        let (mr, mi) = pair(&row["momentum"]);
        let (cr, ci) = pair(&row["coordinate"]);
        let scale = 1.0 + coordinate.norm();
        assert!(
            (momentum.re - mr).abs() < 1e-12 * scale && (momentum.im - mi).abs() < 1e-12 * scale,
            "{}: fixture momentum ({mr}, {mi}) vs recomputed {momentum}",
            entry.label
        );
        assert!(
            (coordinate.re - cr).abs() < 1e-12 * scale
                && (coordinate.im - ci).abs() < 1e-12 * scale,
            "{}: fixture coordinate ({cr}, {ci}) vs recomputed {coordinate}",
            entry.label
        );
        let gap = (momentum - coordinate).norm() / scale;
        assert!(
            gap < 1e-9,
            "{}: Plancherel gap {gap:e} (momentum {momentum}, coordinate {coordinate})",
            entry.label
        );
        if gap > worst.0 {
            worst = (gap, &entry.label);
        }
    }
    assert!(worst.0 < 1e-11, "worst row {} gap {:e}", worst.1, worst.0);
}

#[test]
fn fixture_worst_row_is_reproduced() {
    let doc = fixture();
    let report = worst_gap(&DictionaryPackets::declared(), &fixture_parameters());
    assert_eq!(report.rows, 1241);
    assert_eq!(doc["worst"]["label"], report.label);
    let stored = doc["worst"]["relative_gap"].as_f64().unwrap();
    assert!(
        (stored - report.relative_gap).abs() <= 1e-12 * (1.0 + report.relative_gap),
        "stored {stored} vs recomputed {}",
        report.relative_gap
    );
    assert!(stored < 1e-12, "worst gap {stored:e}");
}

#[test]
fn a_wrong_fourier_symbol_changes_exactly_the_p_bearing_rows() {
    let packets = DictionaryPackets::declared();
    let params = fixture_parameters();
    let canonical = Conventions::canonical();
    let wrong = Conventions {
        two_pi: 1.0,
        ..canonical
    };
    let mut changed = 0usize;
    let mut with_p = 0usize;
    for entry in dictionary() {
        let has_p = entry.terms.iter().any(|t| t.p_degree() > 0);
        if has_p {
            with_p += 1;
        }
        let a = momentum_kernel(entry, &packets, &params, &canonical);
        let b = momentum_kernel(entry, &packets, &params, &wrong);
        let differs = (a - b).norm() > 1e-12 * (1.0 + a.norm());
        assert_eq!(
            differs, has_p,
            "{}: p-bearing={has_p} but changed={differs}",
            entry.label
        );
        if differs {
            changed += 1;
        }
    }
    assert_eq!(changed, with_p);
    // 40 kinetic rows plus the 192 matter rows that carry the spatial Dirac
    // derivative; the remaining 960 matter rows are pure gauge/Higgs insertions.
    assert_eq!(with_p, 232);
}
