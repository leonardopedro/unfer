use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_rotation")
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/fixtures/ultrabert_agnews_embeddings.safetensors")
}

fn state_path() -> PathBuf {
    std::env::temp_dir().join(format!("pcnl_rotation_cli_{}.json", std::process::id()))
}

fn run(args: &[&str]) -> String {
    let output = Command::new(bin())
        .args(args)
        .output()
        .expect("run rotation CLI");
    assert!(
        output.status.success(),
        "rotation {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 CLI output")
}

#[test]
fn cli_init_update_predict_show_restarts_from_the_same_checkpoint() {
    let path = state_path();
    std::fs::remove_file(&path).ok();
    let state = path.to_str().unwrap();
    let fixture = fixture();
    let fixture = fixture.to_str().unwrap();

    let init = run(&[
        "init", "--state", state, "--hidden", "4", "--chains", "1", "--steps", "30", "--burn", "5",
        "--thin", "2", "--draws", "4",
    ]);
    assert!(init.contains("initialized"));
    assert!(path.is_file());

    let update = run(&[
        "update",
        "--state",
        state,
        "--fixture",
        fixture,
        "--per-class",
        "1",
    ]);
    assert!(update.contains("checkpoint=4"));
    assert!(update.contains("retained=4"));

    let predict = run(&[
        "predict",
        "--state",
        state,
        "--fixture",
        fixture,
        "--row",
        "0",
    ]);
    assert!(predict.contains("checkpoint=4"));
    assert!(predict.contains("p=["));

    let show = run(&["show", "--state", state]);
    assert!(show.contains("checkpoint=4"));
    assert!(show.contains("updates=1"));
    assert!(show.contains("retained=4"));

    // Reopening must not create or depend on a stale temporary checkpoint.
    let stale = path.with_extension("tmp.999999");
    assert!(!stale.exists());
    std::fs::remove_file(&path).unwrap();
}
