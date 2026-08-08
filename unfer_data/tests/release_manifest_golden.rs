//! F23 golden gate: a byte-identical release manifest must survive a rebuild of the
//! module artifacts. Regenerate with `UPDATE_GOLDEN=1 cargo test -p unfer_data -t
//! release_manifest_golden` — a wrong byte in any module changes the manifest and this
//! test fails (the CI gate), so a golden bump is explicit and reviewed.

use unfer_data::release::ReleaseManifest;

fn fixture_artifacts() -> Vec<(String, Vec<u8>)> {
    vec![
        (
            "nested_fock_algebra".to_string(),
            b"EXPORT(MODULE \"nested_fock_algebra\" (VALUE\n  (SYMBOL kernel)\n)))\n".to_vec(),
        ),
        (
            "demo".to_string(),
            b"module demo {\n grain \"src/main.js\" = \"export async function run(k,a){return a;}\";\n}\n"
                .to_vec(),
        ),
    ]
}

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("release_manifest.json")
}

fn built_manifest() -> ReleaseManifest {
    let artifacts = fixture_artifacts();
    let refs: Vec<(String, &[u8])> = artifacts
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();
    ReleaseManifest::build(1, &refs)
}

#[test]
fn release_manifest_matches_committed_golden() {
    let built = built_manifest();
    assert!(
        built.validate(),
        "all artifact CIDs must be well-formed content addresses"
    );

    let mut canonical = built.to_canonical_bytes();
    canonical.push(b'\n'); // editor-friendly trailing newline for the golden file

    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        let dir = golden_path().parent().expect("golden dir").to_path_buf();
        std::fs::create_dir_all(&dir).expect("golden dir");
        std::fs::write(&golden_path(), &canonical).expect("write golden");
        eprintln!("UPDATE_GOLDEN=1: regenerated {}", golden_path().display());
        return;
    }

    let golden = std::fs::read_to_string(&golden_path()).unwrap_or_else(|_| {
        panic!(
            "missing golden manifest {} — run with UPDATE_GOLDEN=1 to generate",
            golden_path().display()
        )
    });
    let actual = String::from_utf8(canonical).expect("canonical utf8");
    assert_eq!(
        actual.trim_end(),
        golden.trim_end(),
        "release manifest drifted from the committed golden file. Regenerate only with \
         UPDATE_GOLDEN=1 after an intentional artifact change (F23 gate)."
    );
}
