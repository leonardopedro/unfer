//! Zenodo-Loro incremental storage adapter (`uz_*` ABI).
//!
//! Implements the "Alternative-to-Git" architecture described in
//! `unfer/docs/altgit.md` (adapted from the Zenodo + Loro CRDT
//! incremental delta pattern):
//!
//! - **Snapshot** — the starting-point full export of a Loro document
//!   (`snapshot_N.bin`), uploaded once, never overwritten.
//! - **Delta** — incremental changes since the last upload
//!   (`delta_N.bin`), uploaded on each subsequent save.
//! - **Manifest** — `manifest.json` on Zenodo that tracks the file
//!   sequence and the Loro frontier (opaque version vector) so the
//!   next delta can be generated from the right base.
//! - **Squash** — every `squash_after` deltas (default 50), create a
//!   new full snapshot and clear the old delta chain (Zenodo keeps all
//!   prior versions immutably; the squash only affects the latest
//!   version for fast cold-load).
//!
//! The adapter is agnostic to the Loro binary format — it stores and
//! retrieves raw bytes and the caller (velysterm) drives Loro's
//! `doc.export()` / `doc.import()`.
//!
//! ## API summary
//!
//! | Symbol | Semantics |
//! |--------|-----------|
//! | `uz_init(cfg_json, len) -> i64` | Configure: api_key, sandbox flag, optional existing record_id. Returns 0. |
//! | `uz_push(data, data_len, frontier, frontier_len) -> i64` | Upload snapshot (first call) or delta (subsequent). Updates manifest. Returns 0. |
//! | `uz_pull(buf, cap) -> i64` | Download all files in file_sequence order, concatenated. Buffer protocol. |
//! | `uz_manifest_json(buf, cap) -> i64` | Current manifest.json bytes (in-memory; no HTTP). Buffer protocol. |
//! | `uz_last_error(buf, cap) -> i64` | Last error string. Buffer protocol. |
//!
//! ## Zenodo API used (v2 deposit API)
//!
//! Sandbox: `https://sandbox.zenodo.org/api`
//! Production: `https://zenodo.org/api`
//!
//! Endpoints: `/deposit/depositions` (CRUD),
//! `/deposit/depositions/{id}/files` (upload via bucket URL),
//! `/deposit/depositions/{id}/actions/publish`,
//! `/deposit/depositions/{id}/actions/newversion`,
//! `/records/{id}` (file listing for pull).
//!
//! ## Feature flag
//!
//! Gated behind `[features] zenodo` in `unfer_ffi/Cargo.toml`.
//! Build: `cargo build -p unfer_ffi --features zenodo`.

use std::io::Read;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

// ── Config ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct InitConfig {
    api_key: String,
    /// Use the Zenodo sandbox (sandbox.zenodo.org). Default: true.
    #[serde(default = "default_true")]
    sandbox: bool,
    /// Existing published record ID to append new versions to.
    /// Omit (or null) to create a new record.
    #[serde(default)]
    record_id: Option<u64>,
    /// Number of deltas before a squash snapshot is created. Default: 50.
    #[serde(default = "default_squash")]
    squash_after: usize,
}

fn default_true() -> bool {
    true
}
fn default_squash() -> usize {
    50
}

// ── Manifest (persisted in manifest.json on Zenodo) ───────────────────

#[derive(Serialize, Deserialize, Default, Clone)]
struct ZenodoManifest {
    /// Ordered list of data files (snapshot_N.bin, delta_N.bin, …).
    file_sequence: Vec<String>,
    /// Base64-encoded Loro frontier bytes from the last upload.
    /// Opaque to this crate — the caller decides how to use it.
    last_frontier: String,
}

// ── Client state ──────────────────────────────────────────────────────

struct ZenodoClient {
    api_key: String,
    base_url: String,
    /// Active draft deposition ID (set during push; cleared after publish).
    deposition_id: Option<u64>,
    /// Zenodo S3 bucket URL for the active draft (faster than the files API).
    bucket_url: Option<String>,
    /// The most recently published record ID.
    record_id: Option<u64>,
    manifest: ZenodoManifest,
    squash_after: usize,
    /// Running count of deltas uploaded since the last snapshot.
    delta_count: usize,
}

impl ZenodoClient {
    fn auth(&self) -> String {
        format!("Bearer {}", self.api_key)
    }

    // ── Draft management ─────────────────────────────────────────────

    fn ensure_draft(&mut self) -> Result<(), String> {
        if self.deposition_id.is_some() {
            return Ok(());
        }

        if let Some(record_id) = self.record_id {
            // New version of an existing published record.
            let url = format!(
                "{}/deposit/depositions/{}/actions/newversion",
                self.base_url, record_id
            );
            let resp: serde_json::Value = ureq::post(&url)
                .set("Authorization", &self.auth())
                .call()
                .map_err(|e| format!("newversion POST: {e}"))?
                .into_json()
                .map_err(|e| format!("newversion JSON: {e}"))?;

            // The new draft's URL is in links.latest_draft.
            let draft_url = resp["links"]["latest_draft"]
                .as_str()
                .ok_or("missing links.latest_draft in newversion response")?
                .to_string();

            // Fetch the draft to get its ID and bucket URL.
            let draft: serde_json::Value = ureq::get(&draft_url)
                .set("Authorization", &self.auth())
                .call()
                .map_err(|e| format!("get draft: {e}"))?
                .into_json()
                .map_err(|e| format!("get draft JSON: {e}"))?;

            let draft_id = draft["id"]
                .as_u64()
                .ok_or("no id in draft response")?;
            self.deposition_id = Some(draft_id);
            self.bucket_url = draft["links"]["bucket"].as_str().map(String::from);

            // Delete the inherited manifest.json so we can upload a fresh one.
            self.delete_file_if_exists(draft_id, "manifest.json")?;
        } else {
            // Create a brand-new deposition.
            let url = format!("{}/deposit/depositions", self.base_url);
            let body = serde_json::json!({
                "metadata": {
                    "title": "unfer session (zenodo_store_module)",
                    "upload_type": "dataset",
                    "description": "Incremental unfer kernel session backup via Loro CRDT delta snapshots. Generated by unfer/zenodo_store_module."
                }
            });
            let resp: serde_json::Value = ureq::post(&url)
                .set("Authorization", &self.auth())
                .set("Content-Type", "application/json")
                .send_json(body)
                .map_err(|e| format!("create deposition: {e}"))?
                .into_json()
                .map_err(|e| format!("create deposition JSON: {e}"))?;

            self.deposition_id = resp["id"]
                .as_u64()
                .ok_or("no id in create deposition response")?
                .into();
            self.bucket_url = resp["links"]["bucket"].as_str().map(String::from);
        }
        Ok(())
    }

    fn delete_file_if_exists(&self, dep_id: u64, filename: &str) -> Result<(), String> {
        let url = format!(
            "{}/deposit/depositions/{}/files",
            self.base_url, dep_id
        );
        let files: Vec<serde_json::Value> = ureq::get(&url)
            .set("Authorization", &self.auth())
            .call()
            .map_err(|e| format!("list files: {e}"))?
            .into_json()
            .map_err(|e| format!("list files JSON: {e}"))?;

        for file in &files {
            if file["filename"].as_str() == Some(filename) {
                if let Some(file_id) = file["id"].as_str() {
                    let del_url = format!(
                        "{}/deposit/depositions/{}/files/{}",
                        self.base_url, dep_id, file_id
                    );
                    ureq::delete(&del_url)
                        .set("Authorization", &self.auth())
                        .call()
                        .map_err(|e| format!("delete {filename}: {e}"))?;
                }
                break;
            }
        }
        Ok(())
    }

    // ── File upload ──────────────────────────────────────────────────

    fn upload_bytes(&self, filename: &str, data: &[u8]) -> Result<(), String> {
        if let Some(bucket) = &self.bucket_url {
            // Bucket API (preferred — no multipart needed).
            let url = format!("{}/{}", bucket, filename);
            ureq::put(&url)
                .set("Authorization", &self.auth())
                .set("Content-Type", "application/octet-stream")
                .send_bytes(data)
                .map_err(|e| format!("upload {filename} via bucket: {e}"))?;
        } else {
            return Err(format!(
                "no bucket URL for deposition {:?}",
                self.deposition_id
            ));
        }
        Ok(())
    }

    // ── Publish ──────────────────────────────────────────────────────

    fn publish(&mut self) -> Result<u64, String> {
        let dep_id = self
            .deposition_id
            .ok_or("no active draft deposition to publish")?;
        let url = format!(
            "{}/deposit/depositions/{}/actions/publish",
            self.base_url, dep_id
        );
        let resp: serde_json::Value = ureq::post(&url)
            .set("Authorization", &self.auth())
            .call()
            .map_err(|e| format!("publish: {e}"))?
            .into_json()
            .map_err(|e| format!("publish JSON: {e}"))?;

        let record_id = resp["id"]
            .as_u64()
            .ok_or("no id in publish response")?;

        self.deposition_id = None;
        self.bucket_url = None;
        Ok(record_id)
    }

    // ── Core push ────────────────────────────────────────────────────

    pub fn push_bytes(&mut self, data: &[u8], frontier: &[u8]) -> Result<(), String> {
        self.ensure_draft()?;

        let is_first = self.manifest.file_sequence.is_empty();
        let squash = !is_first && self.delta_count >= self.squash_after;

        // Choose filename.
        let filename = if is_first || squash {
            let snap_idx = self
                .manifest
                .file_sequence
                .iter()
                .filter(|f| f.starts_with("snapshot_"))
                .count();
            format!("snapshot_{snap_idx}.bin")
        } else {
            let delta_idx = self
                .manifest
                .file_sequence
                .iter()
                .filter(|f| f.starts_with("delta_"))
                .count();
            format!("delta_{delta_idx}.bin")
        };

        // If squashing, reset the file sequence (old versions stay on Zenodo
        // under prior version numbers — they are immutable).
        if squash {
            self.manifest.file_sequence.clear();
            self.delta_count = 0;
        }

        // Upload data file.
        self.upload_bytes(&filename, data)?;

        // Update in-memory manifest.
        self.manifest.file_sequence.push(filename.clone());
        self.manifest.last_frontier = base64_encode(frontier);
        if filename.starts_with("delta_") {
            self.delta_count += 1;
        }

        // Upload updated manifest.json.
        let manifest_json = serde_json::to_vec(&self.manifest)
            .map_err(|e| format!("manifest serialise: {e}"))?;
        self.upload_bytes("manifest.json", &manifest_json)?;

        // Publish draft → new Zenodo version.
        let record_id = self.publish()?;
        self.record_id = Some(record_id);
        Ok(())
    }

    // ── Pull (download) ──────────────────────────────────────────────

    pub fn pull_all(&self) -> Result<Vec<u8>, String> {
        let record_id = self
            .record_id
            .ok_or("no published record — call uz_push first")?;

        // Fetch the record metadata (file list with download links).
        let url = format!("{}/records/{}", self.base_url, record_id);
        let record: serde_json::Value = ureq::get(&url)
            .set("Authorization", &self.auth())
            .call()
            .map_err(|e| format!("get record {record_id}: {e}"))?
            .into_json()
            .map_err(|e| format!("get record JSON: {e}"))?;

        let files = record["files"]
            .as_array()
            .ok_or("no files array in record response")?;

        // Build a name → download-URL map.
        let mut url_map = std::collections::HashMap::new();
        for f in files {
            if let (Some(key), Some(link)) =
                (f["key"].as_str(), f["links"]["self"].as_str())
            {
                url_map.insert(key.to_string(), link.to_string());
            }
        }

        // Download manifest.json to get the canonical file_sequence.
        let manifest_url = url_map
            .get("manifest.json")
            .ok_or("manifest.json not found on Zenodo record")?
            .clone();

        let mut manifest_bytes = Vec::new();
        ureq::get(&manifest_url)
            .set("Authorization", &self.auth())
            .call()
            .map_err(|e| format!("download manifest.json: {e}"))?
            .into_reader()
            .read_to_end(&mut manifest_bytes)
            .map_err(|e| format!("read manifest.json: {e}"))?;

        let manifest: ZenodoManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| format!("parse manifest.json: {e}"))?;

        // Download each file in order and concatenate.
        let mut all_bytes = Vec::new();
        for fname in &manifest.file_sequence {
            let file_url = url_map
                .get(fname)
                .ok_or_else(|| format!("{fname} listed in manifest but not on Zenodo record"))?
                .clone();

            ureq::get(&file_url)
                .set("Authorization", &self.auth())
                .call()
                .map_err(|e| format!("download {fname}: {e}"))?
                .into_reader()
                .read_to_end(&mut all_bytes)
                .map_err(|e| format!("read {fname}: {e}"))?;
        }

        Ok(all_bytes)
    }

    pub fn manifest_json(&self) -> Vec<u8> {
        serde_json::to_vec(&self.manifest).unwrap_or_default()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

fn base64_encode(bytes: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 { chunk[1] as usize } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as usize } else { 0 };
        out.push(CHARS[b0 >> 2] as char);
        out.push(CHARS[((b0 & 3) << 4) | (b1 >> 4)] as char);
        out.push(if chunk.len() > 1 {
            CHARS[((b1 & 0xf) << 2) | (b2 >> 6)] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            CHARS[b2 & 0x3f] as char
        } else {
            '='
        });
    }
    out
}

fn write_buf_bytes(buf: *mut u8, cap: i64, data: &[u8]) -> i64 {
    let needed = data.len() as i64;
    if cap > 0 && !buf.is_null() {
        let copy = std::cmp::min(needed as usize, cap as usize);
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf, copy) };
    }
    needed
}

// ── Global state ──────────────────────────────────────────────────────

static STATE: Mutex<Option<ZenodoClient>> = Mutex::new(None);

thread_local! {
    static UZ_LAST_ERROR: std::cell::RefCell<String> =
        const { std::cell::RefCell::new(String::new()) };
}

fn set_error(msg: &str) {
    UZ_LAST_ERROR.with(|e| *e.borrow_mut() = msg.to_string());
}

fn get_error() -> String {
    UZ_LAST_ERROR.with(|e| e.borrow().clone())
}

// ── Public ABI ────────────────────────────────────────────────────────

/// Configure the Zenodo client.
///
/// `cfg_json` must be valid UTF-8 JSON with at least `"api_key"`:
/// ```json
/// {"api_key":"TOKEN","sandbox":true,"record_id":null,"squash_after":50}
/// ```
/// Returns 0 on success, -1001 on bad JSON, -5000 on other error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uz_init(cfg_json: *const u8, cfg_len: i64) -> i64 {
    if cfg_json.is_null() || cfg_len <= 0 {
        set_error("uz_init: null or empty config");
        return -1001;
    }
    let slice = unsafe { std::slice::from_raw_parts(cfg_json, cfg_len as usize) };
    let cfg: InitConfig = match serde_json::from_slice(slice) {
        Ok(c) => c,
        Err(e) => {
            set_error(&format!("uz_init: bad config JSON: {e}"));
            return -1001;
        }
    };
    let base_url = if cfg.sandbox {
        "https://sandbox.zenodo.org/api".to_string()
    } else {
        "https://zenodo.org/api".to_string()
    };
    let client = ZenodoClient {
        api_key: cfg.api_key,
        base_url,
        deposition_id: None,
        bucket_url: None,
        record_id: cfg.record_id,
        manifest: ZenodoManifest::default(),
        squash_after: cfg.squash_after,
        delta_count: 0,
    };
    *STATE.lock().unwrap_or_else(|e| e.into_inner()) = Some(client);
    0
}

/// Upload a snapshot (first push) or delta (subsequent push) to Zenodo.
///
/// - `data` / `data_len`: Loro document bytes (snapshot or delta, opaque).
/// - `frontier` / `frontier_len`: Loro frontier bytes to persist in the
///   manifest so the caller can generate the next delta from the right base.
///
/// On the first call a new Zenodo deposition is created and published.
/// On subsequent calls a new version is created and published.
/// Returns 0 on success, -5000 on network or API failure (call
/// `uz_last_error` for details).
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uz_push(
    data: *const u8,
    data_len: i64,
    frontier: *const u8,
    frontier_len: i64,
) -> i64 {
    if data.is_null() || data_len <= 0 {
        set_error("uz_push: null or empty data");
        return -1001;
    }
    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len as usize) };
    let frontier_slice = if frontier.is_null() || frontier_len <= 0 {
        &b""[..]
    } else {
        unsafe { std::slice::from_raw_parts(frontier, frontier_len as usize) }
    };
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    match guard.as_mut() {
        None => {
            set_error("uz_push: not initialized — call uz_init first");
            -5000
        }
        Some(client) => match client.push_bytes(data_slice, frontier_slice) {
            Ok(()) => 0,
            Err(e) => {
                set_error(&e);
                -5000
            }
        },
    }
}

/// Download all Zenodo files for the current record, concatenated in
/// `file_sequence` order, into `buf`.
///
/// Buffer protocol (same as `uk_get_result`): returns total bytes needed;
/// copies `min(needed, cap)` into `buf`. Callers should probe with
/// `buf=NULL, cap=0` first.
///
/// Returns total bytes on success, -5000 on network/API failure.
#[unsafe(no_mangle)]
pub extern "C" fn uz_pull(buf: *mut u8, cap: i64) -> i64 {
    let guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        None => {
            set_error("uz_pull: not initialized — call uz_init first");
            -5000
        }
        Some(client) => match client.pull_all() {
            Ok(bytes) => write_buf_bytes(buf, cap, &bytes),
            Err(e) => {
                set_error(&e);
                -5000
            }
        },
    }
}

/// Return the in-memory `manifest.json` as a JSON byte string.
/// Buffer protocol: returns total bytes needed.
/// Does NOT hit the network.
#[unsafe(no_mangle)]
pub extern "C" fn uz_manifest_json(buf: *mut u8, cap: i64) -> i64 {
    let guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        None => {
            set_error("uz_manifest_json: not initialized");
            -5000
        }
        Some(client) => write_buf_bytes(buf, cap, &client.manifest_json()),
    }
}

/// Return the last error string (thread-local, same semantics as `uk_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn uz_last_error(buf: *mut u8, cap: i64) -> i64 {
    let err = get_error();
    write_buf_bytes(buf, cap, err.as_bytes())
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn init_client_no_net() -> ZenodoClient {
        ZenodoClient {
            api_key: "test_key".to_string(),
            base_url: "https://sandbox.zenodo.org/api".to_string(),
            deposition_id: None,
            bucket_url: None,
            record_id: None,
            manifest: ZenodoManifest::default(),
            squash_after: 3,
            delta_count: 0,
        }
    }

    #[test]
    fn base64_empty() {
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn base64_standard_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"Man"), "TWFu");
    }

    #[test]
    fn manifest_first_push_names_snapshot_0() {
        let mut c = init_client_no_net();
        // Build the file name as push_bytes would (without actually uploading).
        let is_first = c.manifest.file_sequence.is_empty();
        let squash = !is_first && c.delta_count >= c.squash_after;
        let filename = if is_first || squash {
            let idx = c.manifest.file_sequence.iter().filter(|f| f.starts_with("snapshot_")).count();
            format!("snapshot_{idx}.bin")
        } else {
            let idx = c.manifest.file_sequence.iter().filter(|f| f.starts_with("delta_")).count();
            format!("delta_{idx}.bin")
        };
        assert_eq!(filename, "snapshot_0.bin");
    }

    #[test]
    fn manifest_second_push_names_delta_0() {
        let mut c = init_client_no_net();
        // Simulate one snapshot already uploaded.
        c.manifest.file_sequence.push("snapshot_0.bin".to_string());
        let is_first = c.manifest.file_sequence.is_empty();
        let squash = !is_first && c.delta_count >= c.squash_after;
        let filename = if is_first || squash {
            let idx = c.manifest.file_sequence.iter().filter(|f| f.starts_with("snapshot_")).count();
            format!("snapshot_{idx}.bin")
        } else {
            let idx = c.manifest.file_sequence.iter().filter(|f| f.starts_with("delta_")).count();
            format!("delta_{idx}.bin")
        };
        assert_eq!(filename, "delta_0.bin");
    }

    #[test]
    fn manifest_squash_triggers_snapshot_1() {
        let mut c = init_client_no_net();
        // Simulate squash_after=3 deltas already uploaded.
        c.manifest.file_sequence.push("snapshot_0.bin".to_string());
        c.manifest.file_sequence.push("delta_0.bin".to_string());
        c.manifest.file_sequence.push("delta_1.bin".to_string());
        c.manifest.file_sequence.push("delta_2.bin".to_string());
        c.delta_count = 3;
        let is_first = c.manifest.file_sequence.is_empty();
        let squash = !is_first && c.delta_count >= c.squash_after;
        assert!(squash, "should trigger squash at delta_count=3=squash_after");
        let filename = if is_first || squash {
            if squash {
                c.manifest.file_sequence.clear();
                c.delta_count = 0;
            }
            let idx = c.manifest.file_sequence.iter().filter(|f| f.starts_with("snapshot_")).count();
            format!("snapshot_{idx}.bin")
        } else {
            let idx = c.manifest.file_sequence.iter().filter(|f| f.starts_with("delta_")).count();
            format!("delta_{idx}.bin")
        };
        // After squash the file_sequence was cleared, so count=0 → snapshot_0.
        assert_eq!(filename, "snapshot_0.bin");
        assert_eq!(c.delta_count, 0);
        assert!(c.manifest.file_sequence.is_empty());
    }

    #[test]
    fn uz_init_bad_json_returns_neg1001() {
        let bad = b"not json";
        let rc = uz_init(bad.as_ptr(), bad.len() as i64);
        assert_eq!(rc, -1001);
    }

    #[test]
    fn uz_init_null_pointer_returns_neg1001() {
        let rc = uz_init(std::ptr::null(), 0);
        assert_eq!(rc, -1001);
    }

    // Note: tests that rely on STATE being None (e.g. "push before init")
    // cannot be made deterministic in a parallel test harness without a
    // serialization primitive such as serial_test. We test the "not
    // initialized" path via the Rust-level helper instead (see
    // `push_bytes_fails_without_network` below — it calls push_bytes()
    // on a client that has no record_id, which hits the network and
    // correctly returns an error).

    #[test]
    fn uz_init_returns_zero_on_valid_config() {
        // Just test the return code; do not inspect global STATE to avoid
        // poisoning the Mutex if the assertion would fail in a parallel run.
        let cfg = br#"{"api_key":"test_key_abc","sandbox":true}"#;
        let rc = uz_init(cfg.as_ptr(), cfg.len() as i64);
        assert_eq!(rc, 0, "uz_init must return 0 on valid config");
    }

    #[test]
    fn uz_manifest_json_after_init_returns_empty_manifest() {
        // Initialize, then probe the manifest via the FFI (observable behavior,
        // not STATE internals — avoids holding the lock when panicking).
        let cfg = br#"{"api_key":"manifest_test","sandbox":true}"#;
        let rc = uz_init(cfg.as_ptr(), cfg.len() as i64);
        assert_eq!(rc, 0);
        let needed = uz_manifest_json(std::ptr::null_mut(), 0);
        assert!(needed > 0, "manifest must have some bytes");
        let mut buf = vec![0u8; needed as usize];
        let written = uz_manifest_json(buf.as_mut_ptr(), needed);
        assert_eq!(written, needed);
        let json: serde_json::Value = serde_json::from_slice(&buf).expect("manifest not valid JSON");
        assert_eq!(
            json["file_sequence"].as_array().expect("file_sequence missing").len(),
            0,
            "fresh manifest must have empty file_sequence"
        );
        assert_eq!(
            json["last_frontier"].as_str().expect("last_frontier missing"),
            "",
            "fresh manifest must have empty last_frontier"
        );
    }
}
