//! The committed UltraBERT embedding fixture is part of the experiment's
//! identity: a shape or split change would silently change every reported
//! number.  These checks are cheap and do not run the encoder.

use pcnl::text_head::{EmbeddingTable, Split};
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/fixtures/ultrabert_agnews_embeddings.safetensors")
}

#[test]
fn ultrabert_fixture_has_the_declared_shape_and_split() {
    let t = EmbeddingTable::load(&fixture()).expect("load UltraBERT fixture");
    assert_eq!(t.dim(), 768);
    assert_eq!(t.classes(), 4);
    assert_eq!(t.n_train(), 240);
    assert_eq!(t.n_test(), 120);
    assert_eq!(t.y(Split::Train).len(), 240);
    assert_eq!(t.y(Split::Test).len(), 120);
    for &y in t.y(Split::Train).iter().chain(t.y(Split::Test)) {
        assert!((y as usize) < t.classes(), "label {y}");
    }
}

#[test]
fn the_fixture_embeddings_are_unit_norm() {
    let t = EmbeddingTable::load(&fixture()).expect("load UltraBERT fixture");
    for split in [Split::Train, Split::Test] {
        let x = t.x(split);
        let n = t.y(split).len();
        for row in x.chunks(t.dim()).take(n) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 2e-3, "row norm {norm}");
        }
    }
}
