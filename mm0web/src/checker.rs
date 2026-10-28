//! Verification: the same two-stage protocol `australVM/scripts/lean2mm0`
//! runs in CI (`verify`, §5 of `australVM/docs/lean4mm0.md`).
//!
//! ```text
//! mm0-rs compile proof.mm1 proof.mmb     → rc 0, and no "warning" in the output
//! mm0-c proof.mmb   (stdin: proof.mm0)   → rc 0
//! ```
//!
//! A warning from `mm0-rs` is treated as a **proof hole**, exactly as the CI
//! does ("a warning is a proof hole here"), so acceptance is rc 0 *and* a clean
//! log. Both stages are configurable by path, because the toolchain is built
//! separately (`cargo install --path mm0/mm0-rs`, `gcc mm0/mm0-c/main.c`) and
//! is not vendored in this repo.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Where the two checker binaries live, and how long each stage may run.
#[derive(Clone, Debug)]
pub struct CheckerConfig {
    pub mm0_rs: PathBuf,
    pub mm0_c: PathBuf,
    pub timeout: Duration,
}

impl Default for CheckerConfig {
    fn default() -> Self {
        Self {
            mm0_rs: PathBuf::from("mm0-rs"),
            mm0_c: PathBuf::from("mm0-c"),
            timeout: Duration::from_secs(60),
        }
    }
}

/// The verdict plus the log a user needs to act on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckOutcome {
    pub accepted: bool,
    /// `"mm0-rs"`, `"mm0-c"`, `"timeout"`, or `"missing-toolchain"`.
    pub stage: &'static str,
    pub log: String,
}

impl CheckOutcome {
    fn reject(stage: &'static str, log: impl Into<String>) -> Self {
        Self {
            accepted: false,
            stage,
            log: log.into(),
        }
    }
}

/// Verify one submission. `workdir` is where the two files and the `.mmb` are
/// written; the caller owns it (the service uses a per-request temp dir), so
/// nothing is deleted behind a concurrent reader's back.
pub fn check(cfg: &CheckerConfig, mm0_src: &str, mm1_src: &str, workdir: &Path) -> CheckOutcome {
    if let Err(e) = std::fs::create_dir_all(workdir) {
        return CheckOutcome::reject("io", format!("cannot create workdir: {e}"));
    }
    let mm0_path = workdir.join("proof.mm0");
    let mm1_path = workdir.join("proof.mm1");
    let mmb_path = workdir.join("proof.mmb");
    if let Err(e) = std::fs::write(&mm0_path, mm0_src) {
        return CheckOutcome::reject("io", format!("cannot write proof.mm0: {e}"));
    }
    if let Err(e) = std::fs::write(&mm1_path, mm1_src) {
        return CheckOutcome::reject("io", format!("cannot write proof.mm1: {e}"));
    }

    // Stage 1 — mm0-rs compile <in.mm1> <out.mmb>
    let mut log = String::new();
    let stage1 = run(
        &cfg.mm0_rs,
        &["compile", &path_str(&mm1_path), &path_str(&mmb_path)],
        None,
        cfg.timeout,
    );
    match stage1 {
        Ok((code, out)) => {
            log.push_str(&format!(
                "$ {} compile proof.mm1 proof.mmb\n",
                cfg.mm0_rs.display()
            ));
            log.push_str(&out);
            if code != 0 {
                return CheckOutcome::reject("mm0-rs", log);
            }
            // A warning is a proof hole (the CI rule), not a note.
            if out.to_lowercase().contains("warning") {
                log.push_str("\nrejected: mm0-rs emitted a warning — a warning is a proof hole\n");
                return CheckOutcome::reject("mm0-rs", log);
            }
        }
        Err(e) => return CheckOutcome::reject("missing-toolchain", e),
    }

    // Stage 2 — mm0-c <in.mmb>, with the mm0 source on stdin.
    match run(
        &cfg.mm0_c,
        &[&path_str(&mmb_path)],
        Some(mm0_src),
        cfg.timeout,
    ) {
        Ok((code, out)) => {
            log.push_str(&format!(
                "\n$ {} proof.mmb < proof.mm0\n",
                cfg.mm0_c.display()
            ));
            log.push_str(&out);
            if code != 0 {
                return CheckOutcome::reject("mm0-c", log);
            }
        }
        Err(e) => return CheckOutcome::reject("missing-toolchain", e),
    }

    log.push_str("\naccepted: mm0-rs elaborated the proofs and mm0-c re-verified the .mmb\n");
    CheckOutcome {
        accepted: true,
        stage: "ok",
        log,
    }
}

fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Run a stage, capturing stdout+stderr, with a wall-clock deadline. `Ok` is
/// `(exit_code, combined_output)`; `Err` is the "could not run at all" case
/// (missing binary), which is reported differently from a rejection.
fn run(
    bin: &Path,
    args: &[&str],
    stdin_data: Option<&str>,
    timeout: Duration,
) -> Result<(i32, String), String> {
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(if stdin_data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return Err(format!(
                "cannot run {}: {e} (build it: `cargo install --path mm0/mm0-rs` and \
                 `gcc mm0/mm0-c/main.c -O2 -o mm0-c`, then put both on PATH or point \
                 MM0WEB_MM0_RS / MM0WEB_MM0_C at them)",
                bin.display()
            ));
        }
    };
    if let Some(data) = stdin_data
        && let Some(mut si) = child.stdin.take()
    {
        // A checker that exits before reading stdin gives EPIPE; that is its
        // own verdict, not an error of ours, so the write failure is reported
        // in the log rather than raised.
        if si.write_all(data.as_bytes()).is_err() {
            // ignored on purpose — see comment above
        }
    }
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => break,
            Ok(None) => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok((-1, format!("timed out after {:?}\n", timeout)));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(format!("waiting for {}: {e}", bin.display())),
        }
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let code = out.status.code().unwrap_or(-1);
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    if code == -1 {
        // killed by the deadline above
        text.push_str("(killed: deadline exceeded)\n");
    }
    Ok((code, text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn stub(dir: &Path, name: &str, body: &str) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let p = dir.join(name);
        fs::write(&p, body).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = fs::metadata(&p).unwrap().permissions();
            perm.set_mode(0o755);
            fs::set_permissions(&p, perm).unwrap();
        }
        p
    }

    #[test]
    fn accepts_when_both_stages_are_clean() {
        let d = std::env::temp_dir().join(format!("mm0web-check-ok-{}", std::process::id()));
        let mm0rs = stub(&d, "mm0-rs", "#!/bin/sh\necho 'elab ok'\ntouch \"$3\"\n");
        let mm0c = stub(&d, "mm0-c", "#!/bin/sh\ncat >/dev/null\nexit 0\n");
        let cfg = CheckerConfig {
            mm0_rs: mm0rs,
            mm0_c: mm0c,
            timeout: Duration::from_secs(10),
        };
        let out = check(&cfg, "sort Prop\n", "theorem t: $ Prop;\n", &d.join("w"));
        assert!(out.accepted, "{:?}", out);
        assert_eq!(out.stage, "ok");
        assert!(out.log.contains("elab ok"));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn rejects_on_warning_even_with_exit_zero() {
        let d = std::env::temp_dir().join(format!("mm0web-check-warn-{}", std::process::id()));
        let mm0rs = stub(
            &d,
            "mm0-rs",
            "#!/bin/sh\necho 'warning: hole in proof'\ntouch \"$3\"\nexit 0\n",
        );
        let mm0c = stub(&d, "mm0-c", "#!/bin/sh\nexit 0\n");
        let cfg = CheckerConfig {
            mm0_rs: mm0rs,
            mm0_c: mm0c,
            timeout: Duration::from_secs(10),
        };
        let out = check(&cfg, "s\n", "p\n", &d.join("w"));
        assert!(!out.accepted);
        assert_eq!(out.stage, "mm0-rs");
        assert!(out.log.contains("proof hole"), "{}", out.log);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn rejects_when_mm0_c_rejects_and_reports_missing_toolchain() {
        let d = std::env::temp_dir().join(format!("mm0web-check-rej-{}", std::process::id()));
        let ok_rs = stub(&d, "mm0-rs", "#!/bin/sh\ntouch \"$3\"\nexit 0\n");
        let bad_c = stub(&d, "mm0-c", "#!/bin/sh\necho 'no such theorem'\nexit 3\n");
        let cfg = CheckerConfig {
            mm0_rs: ok_rs,
            mm0_c: bad_c,
            timeout: Duration::from_secs(10),
        };
        let out = check(&cfg, "s\n", "p\n", &d.join("w"));
        assert!(!out.accepted);
        assert_eq!(out.stage, "mm0-c");

        let missing = CheckerConfig {
            mm0_rs: d.join("does-not-exist"),
            ..cfg
        };
        let out = check(&missing, "s\n", "p\n", &d.join("w2"));
        assert_eq!(out.stage, "missing-toolchain");
        assert!(
            out.log.contains("cargo install --path mm0/mm0-rs"),
            "{}",
            out.log
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_hanging_stage_times_out_instead_of_hanging_the_server() {
        let d = std::env::temp_dir().join(format!("mm0web-check-slow-{}", std::process::id()));
        let slow = stub(&d, "mm0-rs", "#!/bin/sh\nsleep 30\n");
        let cfg = CheckerConfig {
            mm0_rs: slow,
            mm0_c: PathBuf::from("mm0-c"),
            timeout: Duration::from_millis(300),
        };
        let started = Instant::now();
        let out = check(&cfg, "s\n", "p\n", &d.join("w"));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "took {:?}",
            started.elapsed()
        );
        assert!(!out.accepted);
        assert!(out.log.contains("timed out"), "{}", out.log);
        let _ = fs::remove_dir_all(&d);
    }
}
