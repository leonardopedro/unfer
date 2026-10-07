//! `cell_tool` — the `.cell` blueprint round-trip CLI (N2 module reconstruction).
//!
//! The N2 benchmark task `module-reconstruction` rebuilds a module's behaviour
//! from a pinned `.cell` blueprint and holds the reconstruction to a golden
//! manifest before the project's own gate (`tools/module_builder`) runs against
//! the reconstructed directory. That needs three verbs the kernel FFI does not
//! expose as a command:
//!
//! ```text
//!   cell_tool pack    <module_dir> <out.cell>   # source dir -> .cell blueprint
//!   cell_tool unpack  <in.cell> <out_dir>       # .cell blueprint -> materialized dir
//!   cell_tool manifest <module_dir>             # golden manifest lines (sha256sum -c format)
//!   cell_tool verify  <dir> <manifest_file>     # dir matches manifest (extra/missing/hashed)
//! ```
//!
//! `pack` and `unpack` are the archive's own pair (`CellBuilder`/`Cell::parse`);
//! `manifest`/`verify` give the benchmark a golden input it did not just compute
//! at run time — a reconstruction gate that regenerates its own expectations
//! cannot fail on drift, which is the one failure it exists to catch.
//!
//! Deliberate exclusions: symlinks and `target/` / `.git/` directories are never
//! packed (build output is not module behaviour; a symlink target cannot be
//! represented in the archive format).

use std::collections::BTreeMap;
use std::path::{Component, Path};
use std::process::ExitCode;

use sha2::{Digest, Sha256};
use unfer_protocol::archive::{Cell, CellBuilder};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("pack") => pack(&args[1..]),
        Some("unpack") => unpack(&args[1..]),
        Some("manifest") => manifest_cmd(&args[1..]),
        Some("verify") => verify(&args[1..]),
        _ => Err(usage()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("cell_tool: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn usage() -> String {
    "usage: cell_tool pack <module_dir> <out.cell> | unpack <in.cell> <out_dir> | \
     manifest <module_dir> | verify <dir> <manifest_file>"
        .to_string()
}

/// First `key = "value"` line of a `module.toml`, mirroring how
/// `tools/module_builder` greps it — the CLI must not disagree with the gate.
fn toml_string_field(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        let rest = rest.trim_start();
        let mut chars = rest.chars();
        if chars.next() != Some('"') {
            continue;
        }
        let mut out = String::new();
        let mut escaped = false;
        for c in chars {
            if escaped {
                out.push(c);
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                return Some(out);
            } else {
                out.push(c);
            }
        }
    }
    None
}

/// Every regular file under `dir` as `(relative/slash/path, bytes)`, sorted, with
/// `target/` and `.git/` subtrees and symlinks excluded (see the module docs).
fn collect_files(dir: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::new();
    walk(dir, dir, &mut out)?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    if out.is_empty() {
        return Err(format!("{} contains no files to pack", dir.display()));
    }
    Ok(out)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let kind = entry
            .file_type()
            .map_err(|e| format!("stat {}: {e}", path.display()))?;
        if kind.is_dir() {
            if name == "target" || name == ".git" {
                continue;
            }
            walk(root, &path, out)?;
        } else if kind.is_symlink() {
            // Not representable in the archive: recorded as an exclusion, never
            // silently dereferenced (a symlink to ../.. would escape the module).
            eprintln!(
                "cell_tool: skipping symlink {} (the archive format has no symlink)",
                path.display()
            );
        } else if kind.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(|e| format!("strip_prefix {}: {e}", path.display()))?
                .to_string_lossy()
                .replace('\\', "/");
            let bytes =
                std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            out.push((rel, bytes));
        }
    }
    Ok(())
}

fn pack(args: &[String]) -> Result<(), String> {
    let [dir, out] = args else {
        return Err(usage());
    };
    let dir = Path::new(dir);
    let toml_path = dir.join("module.toml");
    let toml_text = std::fs::read_to_string(&toml_path).map_err(|e| {
        format!(
            "{}: {e} (a module blueprint needs its module.toml)",
            toml_path.display()
        )
    })?;
    let name = toml_string_field(&toml_text, "name")
        .ok_or_else(|| "module.toml lacks name = \"…\"".to_string())?;
    let entry = toml_string_field(&toml_text, "entry")
        .ok_or_else(|| "module.toml lacks entry = \"…\"".to_string())?;

    let mut builder = CellBuilder::new(&name);
    builder.set_entry(&entry);
    for (rel, bytes) in collect_files(dir)? {
        builder
            .add_file(&rel, &bytes)
            .map_err(|e| format!("add {rel}: {e}"))?;
    }
    let cell = builder.build().map_err(|e| format!("build: {e}"))?;
    std::fs::write(out, &cell).map_err(|e| format!("write {out}: {e}"))?;
    println!(
        "packed {} files from {} into {out} ({} bytes)",
        builder.files().len(),
        dir.display(),
        cell.len()
    );
    Ok(())
}

/// Reject anything that would escape the output directory. The parser keeps
/// paths verbatim, so this is the reconstruction's traversal guard (the same
/// rule `ModuleHost::instantiate_from_blueprint` applies).
fn safe_relpath(rel: &Path) -> Result<(), String> {
    for comp in rel.components() {
        match comp {
            Component::Normal(_) => {}
            Component::CurDir => {}
            _ => {
                return Err(format!(
                    "refusing to write outside the output directory: {}",
                    rel.display()
                ));
            }
        }
    }
    if rel.as_os_str().is_empty() {
        return Err("empty file path in blueprint".to_string());
    }
    Ok(())
}

fn unpack(args: &[String]) -> Result<(), String> {
    let [cell_path, out_dir] = args else {
        return Err(usage());
    };
    let bytes = std::fs::read(cell_path).map_err(|e| format!("read {cell_path}: {e}"))?;
    let cell = Cell::parse(&bytes).map_err(|e| format!("parse {cell_path}: {e}"))?;
    if cell.file("module.toml").is_none() {
        return Err(format!(
            "{cell_path} has no module.toml — it is not an instantiable module blueprint"
        ));
    }
    let out_dir = Path::new(out_dir);
    for (rel, content) in cell.files() {
        safe_relpath(rel)?;
        let dest = out_dir.join(rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        std::fs::write(&dest, content).map_err(|e| format!("write {}: {e}", dest.display()))?;
    }
    println!(
        "unpacked {} files from {cell_path} into {}",
        cell.files().len(),
        out_dir.display()
    );
    Ok(())
}

fn manifest_cmd(args: &[String]) -> Result<(), String> {
    let [dir] = args else {
        return Err(usage());
    };
    for (rel, bytes) in collect_files(Path::new(dir))? {
        println!("{}  {rel}", sha256_hex(&bytes));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Parse one `sha256sum -c` line: `<64 hex><two spaces><path>`.
fn parse_manifest_line(line: &str) -> Result<(String, String), String> {
    let line = line.trim_end_matches(['\r', '\n']);
    if line.len() < 67 {
        return Err(format!("malformed manifest line: {line:?}"));
    }
    let (hash, path) = line.split_at(64);
    let path = path.strip_prefix("  ").ok_or_else(|| {
        format!("malformed manifest line (expected two spaces after the hash): {line:?}")
    })?;
    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("malformed manifest hash: {line:?}"));
    }
    Ok((hash.to_ascii_lowercase(), path.to_string()))
}

fn verify(args: &[String]) -> Result<(), String> {
    let [dir, manifest_path] = args else {
        return Err(usage());
    };
    let manifest_text =
        std::fs::read_to_string(manifest_path).map_err(|e| format!("read {manifest_path}: {e}"))?;
    let mut expected: BTreeMap<String, String> = BTreeMap::new();
    for (i, line) in manifest_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (hash, path) =
            parse_manifest_line(line).map_err(|e| format!("{}:{}: {e}", manifest_path, i + 1))?;
        expected.insert(path, hash);
    }
    if expected.is_empty() {
        return Err(format!("{manifest_path} lists no files"));
    }

    let actual: BTreeMap<String, String> = collect_files(Path::new(dir))?
        .into_iter()
        .map(|(rel, bytes)| (rel, sha256_hex(&bytes)))
        .collect();

    let mut problems = Vec::new();
    for (path, hash) in &expected {
        match actual.get(path) {
            None => problems.push(format!("missing: {path}")),
            Some(a) if a != hash => problems.push(format!("content drifted: {path}")),
            Some(_) => {}
        }
    }
    for path in actual.keys() {
        if !expected.contains_key(path) {
            problems.push(format!("unexpected extra file: {path}"));
        }
    }
    if problems.is_empty() {
        println!("verified {} files against {manifest_path}", expected.len());
        return Ok(());
    }
    for p in &problems {
        eprintln!("cell_tool: {p}");
    }
    Err(format!(
        "{} problem(s): {dir} does not match {manifest_path}",
        problems.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A unique scratch directory (no `tempfile` in this workspace's deps).
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cell_tool_{tag}_{}_{n}",
            std::process::id(),
            n = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, body: &str) {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).unwrap();
        }
        std::fs::write(path, body).unwrap();
    }

    fn sample_module(tag: &str) -> PathBuf {
        let dir = scratch(tag);
        write(
            &dir.join("module.toml"),
            "name = \"demo\"\nentry = \"src/DemoModule\"\n",
        );
        write(&dir.join("src/DemoModule.aui"), "module DemoModule;\n");
        write(&dir.join("src/DemoModule.aum"), "body DemoModule;\n");
        dir
    }

    #[test]
    fn pack_unpack_verify_round_trips_byte_exactly() {
        let src = sample_module("roundtrip");
        let cell_dir = scratch("roundtrip_cell");
        let cell_path = cell_dir.join("module.cell");
        let out = scratch("roundtrip_out");

        pack(&[src.display().to_string(), cell_path.display().to_string()]).unwrap();
        unpack(&[cell_path.display().to_string(), out.display().to_string()]).unwrap();

        // The golden manifest of the source is exactly what the reconstruction
        // must satisfy — generated once, checked every run.
        let mut golden = String::new();
        for (rel, bytes) in collect_files(&src).unwrap() {
            golden.push_str(&format!("{}  {rel}\n", sha256_hex(&bytes)));
        }
        let manifest = src.join("golden.manifest");
        std::fs::write(&manifest, &golden).unwrap();
        verify(&[out.display().to_string(), manifest.display().to_string()]).unwrap();

        // Drift in the reconstruction must be caught.
        write(
            &out.join("src/DemoModule.aum"),
            "body DemoModule; // edited\n",
        );
        assert!(
            verify(&[out.display().to_string(), manifest.display().to_string()]).is_err(),
            "a changed file must fail the golden manifest"
        );
        // An extra file must be caught too (a reconstruction that invents files
        // is not the blueprint's module).
        write(&out.join("src/Extra.aum"), "body Extra;\n");
        write(&out.join("src/DemoModule.aum"), "body DemoModule;\n");
        assert!(
            verify(&[out.display().to_string(), manifest.display().to_string()]).is_err(),
            "an extra file must fail the golden manifest"
        );
    }

    #[test]
    fn unpack_refuses_paths_that_escape_the_output_directory() {
        let mut b = CellBuilder::new("evil");
        b.add_file("../evil.aum", b"pwn").unwrap();
        b.add_file("module.toml", b"name = \"evil\"\n").unwrap();
        let bytes = b.build().unwrap();
        let cell_path =
            std::env::temp_dir().join(format!("cell_tool_escape_{}.cell", std::process::id()));
        std::fs::write(&cell_path, &bytes).unwrap();
        let out = scratch("escape_out");
        let err =
            unpack(&[cell_path.display().to_string(), out.display().to_string()]).unwrap_err();
        assert!(err.contains("outside the output directory"), "{err}");
        assert!(!out.parent().unwrap().join("evil.aum").exists());
    }

    #[test]
    fn pack_requires_a_module_toml() {
        let dir = scratch("no_toml");
        write(&dir.join("src/Main.aum"), "body Main;\n");
        let out = dir.join("x.cell");
        let err = pack(&[dir.display().to_string(), out.display().to_string()]).unwrap_err();
        assert!(err.contains("module.toml"), "{err}");
    }

    #[test]
    fn unpack_refuses_a_blueprint_without_module_toml() {
        let mut b = CellBuilder::new("naked");
        b.add_file("src/Main.aum", b"body Main;\n").unwrap();
        let bytes = b.build().unwrap();
        let cell_path =
            std::env::temp_dir().join(format!("cell_tool_naked_{}.cell", std::process::id()));
        std::fs::write(&cell_path, &bytes).unwrap();
        let out = scratch("naked_out");
        let err =
            unpack(&[cell_path.display().to_string(), out.display().to_string()]).unwrap_err();
        assert!(err.contains("module.toml"), "{err}");
    }
}
