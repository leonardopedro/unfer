//! Emit the full-dictionary SM kernel fixture
//! (`docs/fixtures/sm_dictionary_kernel.json`).
//!
//! Run with:
//!
//! ```text
//! cargo run -p fock_sirk --example sm_dictionary_kernel
//! ```
//!
//! Every one of the 1,241 exported component expressions is evaluated in the
//! momentum and coordinate representations of the E1 collective-coordinate
//! operator; `relative_gap` is the measured Plancherel defect for that row. See
//! `nested_fock_algebra::sm_dictionary` for the conventions and
//! `docs/SM_FULL_IMPLEMENTATION.md` for the ledger entry.

use nested_fock_algebra::sm_dictionary::{
    DictionaryPackets, Q_COORDINATES, coordinate_kernel, dictionary, fixture_parameters,
    momentum_kernel, three_d_coordinate_kernel, three_d_kernel, worst_gap,
};
use nested_fock_algebra::sm_momentum::Conventions;
use num_complex::Complex64 as C;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn pair(x: C) -> Value {
    json!([x.re, x.im])
}

fn main() {
    let packets = DictionaryPackets::declared();
    let params = fixture_parameters();
    let conv = Conventions::canonical();
    let entries = dictionary();

    let mut families: BTreeMap<&str, usize> = BTreeMap::new();
    let mut rows = Vec::with_capacity(entries.len());
    let mut nonzero = 0usize;
    let mut worst_3d = (0.0f64, String::new());
    for entry in entries {
        let m = momentum_kernel(entry, &packets, &params, &conv);
        let c = coordinate_kernel(entry, &packets, &params);
        let gap = (m - c).norm() / (1.0 + c.norm());
        let m3 = three_d_kernel(entry, &packets, &params, &conv);
        let c3 = three_d_coordinate_kernel(entry, &packets, &params);
        let gap3 = (m3 - c3).norm() / (1.0 + c3.norm());
        if gap3 > worst_3d.0 {
            worst_3d = (gap3, entry.label.clone());
        }
        if m.norm() > 1e-12 {
            nonzero += 1;
        }
        *families.entry(entry.family.as_str()).or_default() += 1;
        rows.push(json!({
            "label": entry.label,
            "family": entry.family.as_str(),
            "terms": entry.terms.len(),
            "momentum": pair(m),
            "coordinate": pair(c),
            "relative_gap": gap,
        }));
    }

    let worst = worst_gap(&packets, &params);
    assert_eq!(worst.rows, entries.len());

    let document = json!({
        "schema": "sm-dictionary-kernel/v1",
        "source": "docs/fixtures/sm_full_cadabra.latex",
        "rows": entries.len(),
        "collective_dimensions": Q_COORDINATES,
        "families": families,
        "nonzero_momentum_rows": nonzero,
        "packets": {
            "rule": "DictionaryPackets::declared()",
            "note": "centres are deterministic functions of the index; widths are 0.55..0.63 per coordinate, 0.9 (bra), 1.1 (ket)",
        },
        "conventions": {
            "fourier": "f_hat(k) = integral exp(-2*pi*i*k.x) f(x) d^163x",
            "coordinate_route": "P(q, -i d/dq) on the ket packet, then <bra|.>",
            "momentum_route": "P((i/2pi) d/dk, 2*pi*k) on the ket packet",
            "p_symbol": "2*pi*k_m (canonical p = -i d/dq_m)",
            "note": "collective-coordinate E1 kernel; not the 3D field-mode kernel of sm_momentum",
        },
        "worst": {
            "label": worst.label,
            "relative_gap": worst.relative_gap,
            "momentum": pair(worst.momentum),
            "coordinate": pair(worst.coordinate),
        },
        "three_d": {
            "rows": entries.len(),
            "worst_label": worst_3d.1,
            "worst_relative_gap": worst_3d.0,
            "note": "per-mode 3D kernel (field insertions, spatial derivative on the ket); validated against its own coordinate route, and its derivative terms against the collective kernel",
        },
        "parameters": {
            "gs": params.gs,
            "gw": params.gw,
            "gy": params.gy,
            "lambda": params.lambda,
            "vev": params.vev,
            "yukawa": "fully populated complex 3x3 matrices, see sm_dictionary::fixture_parameters()",
        },
        "entries": rows,
    });

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/fixtures/sm_dictionary_kernel.json");
    std::fs::write(&path, serde_json::to_string_pretty(&document).unwrap()).unwrap();
    println!(
        "wrote {} ({} rows, {} families, worst {} gap {:.3e})",
        path.display(),
        entries.len(),
        families.len(),
        worst.label,
        worst.relative_gap
    );
    println!("nonzero momentum rows: {nonzero}/{}", entries.len());
    println!("worst 3D row: {} gap {:.3e}", worst_3d.1, worst_3d.0);
}
