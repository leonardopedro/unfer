//! `.cell` blueprint archive format (S5, F4).
//!
//! Mirrors the Cloudflare blueprint archive (`blueprint-archive.ts`): a magic + version +
//! metadata header followed by a gzip-compressed snapshot of the module's files and an
//! optional session blob. An archive produced by [`CellBuilder::build`] round-trips exactly
//! through [`Cell::parse`].
//!
//! Layout (little-endian scalar encoding):
//! ```text
//! [8]  magic         "UNFERCL1"
//! [4]  version       u32 LE (=1)
//! [8]  metadata_len  u64 LE
//! [..] metadata      UTF-8 JSON (`CellMetadata`)
//! [8]  body_len      u64 LE
//! [..] body          gzip(UTF-8 JSON: {"files":[[path,base64],...],"session":base64|null})
//! ```
//!
//! The session payload is opaque bytes (a `SessionBlob` JSON string at the kernel layer); the
//! protocol crate deliberately does not interpret it so the archive stays decoupled from the
//! solver types.

use std::io::{Read, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// 8-byte magic identifying a `.cell` archive.
pub const CELL_MAGIC: [u8; 8] = *b"UNFERCL1";

/// Current archive version. Bumped only for breaking header/body changes.
pub const CELL_VERSION: u32 = 1;

/// Human-readable blueprint metadata, stored in the archive header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellMetadata {
    pub name: String,
    pub version: String,
    pub archetype: String,
    pub entry: String,
    /// Unix epoch seconds at build time.
    pub created_at: i64,
    /// Whether the archive body carries a session snapshot. Cached in the header so an
    /// instantiator can decide to error (or not) without decompressing the body.
    pub session_present: bool,
}

impl CellMetadata {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: "0.1.0".to_string(),
            archetype: "kernel".to_string(),
            entry: String::new(),
            created_at: now_epoch(),
            session_present: false,
        }
    }
}

/// Errors produced while building or parsing a `.cell` archive.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ArchiveError {
    #[error("not a .cell archive: bad magic {0:?}")]
    BadMagic([u8; 8]),
    #[error("unsupported .cell version {0} (expected {CELL_VERSION})")]
    UnsupportedVersion(u32),
    #[error("truncated .cell header")]
    TruncatedHeader,
    #[error("invalid .cell metadata: {0}")]
    BadMetadata(String),
    #[error("invalid .cell body: {0}")]
    BadBody(String),
    #[error("archive I/O error: {0}")]
    Io(String),
}

impl From<std::io::Error> for ArchiveError {
    fn from(e: std::io::Error) -> Self {
        ArchiveError::Io(e.to_string())
    }
}

/// Accumulates a `.cell` archive from metadata, module files and an optional session.
#[derive(Debug, Clone)]
pub struct CellBuilder {
    metadata: CellMetadata,
    files: Vec<(PathBuf, Vec<u8>)>,
    session: Option<Vec<u8>>,
}

impl CellBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            metadata: CellMetadata::new(name),
            files: Vec::new(),
            session: None,
        }
    }

    pub fn metadata(&self) -> &CellMetadata {
        &self.metadata
    }

    pub fn set_metadata(&mut self, metadata: CellMetadata) -> &mut Self {
        self.metadata = metadata;
        self.metadata.session_present = self.session.is_some();
        self
    }

    pub fn set_version(&mut self, version: impl Into<String>) -> &mut Self {
        self.metadata.version = version.into();
        self
    }

    pub fn set_archetype(&mut self, archetype: impl Into<String>) -> &mut Self {
        self.metadata.archetype = archetype.into();
        self
    }

    pub fn set_entry(&mut self, entry: impl Into<String>) -> &mut Self {
        self.metadata.entry = entry.into();
        self
    }

    /// Add a module file. `path` is relative to the module directory (e.g. `src/main.js`);
    /// absolute paths are rejected.
    pub fn add_file(&mut self, path: impl AsRef<std::path::Path>, bytes: &[u8]) -> Result<&mut Self, ArchiveError> {
        let p = path.as_ref();
        if p.is_absolute() {
            return Err(ArchiveError::BadBody(format!(
                "absolute file path not allowed in archive: {}",
                p.display()
            )));
        }
        self.files.push((p.to_path_buf(), bytes.to_vec()));
        Ok(self)
    }

    pub fn files(&self) -> &[(PathBuf, Vec<u8>)] {
        &self.files
    }

    /// Attach the opaque session snapshot (a `SessionBlob` JSON string at the kernel layer).
    pub fn set_session(&mut self, bytes: &[u8]) -> &mut Self {
        self.session = Some(bytes.to_vec());
        self.metadata.session_present = true;
        self
    }

    pub fn session(&self) -> Option<&[u8]> {
        self.session.as_deref()
    }

    pub fn has_session(&self) -> bool {
        self.session.is_some()
    }

    /// Serialize to the on-disk `.cell` format.
    pub fn build(&self) -> Result<Vec<u8>, ArchiveError> {
        let metadata_json = serde_json::to_vec(&self.metadata)
            .map_err(|e| ArchiveError::BadMetadata(e.to_string()))?;

        let files: Vec<(String, String)> = self
            .files
            .iter()
            .map(|(p, b)| {
                let path = p.to_string_lossy().replace('\\', "/");
                (path, hex::encode(b))
            })
            .collect();
        let body = serde_json::json!({
            "files": files,
            "session": self.session.as_ref().map(|s| hex::encode(s)),
        });
        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| ArchiveError::BadBody(e.to_string()))?;

        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(&body_bytes)
            .map_err(|e| ArchiveError::Io(e.to_string()))?;
        let compressed = gz
            .finish()
            .map_err(|e| ArchiveError::Io(e.to_string()))?;

        let mut out = Vec::with_capacity(
            8 + 4 + 8 + metadata_json.len() + 8 + compressed.len(),
        );
        out.extend_from_slice(&CELL_MAGIC);
        out.extend_from_slice(&CELL_VERSION.to_le_bytes());
        out.extend_from_slice(&(metadata_json.len() as u64).to_le_bytes());
        out.extend_from_slice(&metadata_json);
        out.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        out.extend_from_slice(&compressed);
        Ok(out)
    }
}

/// A parsed `.cell` archive.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    metadata: CellMetadata,
    files: Vec<(PathBuf, Vec<u8>)>,
    session: Option<Vec<u8>>,
}

impl Cell {
    pub fn parse(bytes: &[u8]) -> Result<Self, ArchiveError> {
        if bytes.len() < 8 + 4 + 8 + 8 {
            return Err(ArchiveError::TruncatedHeader);
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[..8]);
        if magic != CELL_MAGIC {
            return Err(ArchiveError::BadMagic(magic));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != CELL_VERSION {
            return Err(ArchiveError::UnsupportedVersion(version));
        }
        let meta_len = u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize;
        let meta_start = 20;
        let body_len_at = meta_start + meta_len;
        if bytes.len() < body_len_at + 8 {
            return Err(ArchiveError::TruncatedHeader);
        }
        let metadata_json = &bytes[meta_start..body_len_at];
        let metadata: CellMetadata = serde_json::from_slice(metadata_json)
            .map_err(|e| ArchiveError::BadMetadata(e.to_string()))?;
        let body_len = u64::from_le_bytes(bytes[body_len_at..body_len_at + 8].try_into().unwrap()) as usize;
        let body_start = body_len_at + 8;
        if bytes.len() < body_start + body_len {
            return Err(ArchiveError::TruncatedHeader);
        }
        let body = parse_body(&bytes[body_start..body_start + body_len])?;
        Ok(Self {
            metadata,
            files: body.files,
            session: body.session,
        })
    }

    pub fn metadata(&self) -> &CellMetadata {
        &self.metadata
    }

    pub fn files(&self) -> &[(PathBuf, Vec<u8>)] {
        &self.files
    }

    pub fn file(&self, rel: &str) -> Option<&[u8]> {
        self.files
            .iter()
            .find(|(p, _)| p.to_string_lossy().replace('\\', "/") == rel)
            .map(|(_, b)| b.as_slice())
    }

    pub fn session(&self) -> Option<&[u8]> {
        self.session.as_deref()
    }

    pub fn has_session(&self) -> bool {
        self.session.is_some()
    }
}

struct Body {
    files: Vec<(PathBuf, Vec<u8>)>,
    session: Option<Vec<u8>>,
}

fn parse_body(compressed: &[u8]) -> Result<Body, ArchiveError> {
    let mut gz = flate2::read::GzDecoder::new(compressed);
    let mut buf = Vec::new();
    gz.read_to_end(&mut buf)
        .map_err(|e| ArchiveError::BadBody(format!("gzip: {e}")))?;
    let v: serde_json::Value = serde_json::from_slice(&buf)
        .map_err(|e| ArchiveError::BadBody(format!("json: {e}")))?;

    let files_val = v
        .get("files")
        .and_then(|f| f.as_array())
        .ok_or_else(|| ArchiveError::BadBody("missing \"files\" array".into()))?;
    let mut files = Vec::with_capacity(files_val.len());
    for item in files_val {
        let pair = item
            .as_array()
            .ok_or_else(|| ArchiveError::BadBody("file entry must be [path, bytes]".into()))?;
        let path = pair
            .first()
            .and_then(|p| p.as_str())
            .ok_or_else(|| ArchiveError::BadBody("file path must be a string".into()))?;
        let encoded = pair
            .get(1)
            .and_then(|b| b.as_str())
            .ok_or_else(|| ArchiveError::BadBody("file bytes must be a string".into()))?;
        let bytes = hex::decode(encoded)
            .map_err(|e| ArchiveError::BadBody(format!("bad hex file bytes: {e}")))?;
        files.push((PathBuf::from(path), bytes));
    }

    let session = match v.get("session") {
        Some(serde_json::Value::String(s)) => Some(
            hex::decode(s).map_err(|e| ArchiveError::BadBody(format!("bad hex session: {e}")))?,
        ),
        Some(serde_json::Value::Null) | None => None,
        Some(_) => {
            return Err(ArchiveError::BadBody(
                "session must be a hex string or null".into(),
            ))
        }
    };

    Ok(Body { files, session })
}

fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_build_parse_roundtrip() {
        let mut b = CellBuilder::new("demo");
        b.set_version("0.3.1");
        b.set_archetype("ecmascript");
        b.set_entry("src/main.js");
        b.add_file("module.toml", b"[module]\nname = \"demo\"\n").unwrap();
        b.add_file("src/main.js", b"export async function run(k, a) { return a; }").unwrap();
        b.add_file("data/blob.bin", &[0u8, 1, 2, 3, 255]).unwrap();
        b.set_session(b"{\"t_now\": 0.5}");

        let bytes = b.build().unwrap();
        let cell = Cell::parse(&bytes).unwrap();

        assert_eq!(cell.metadata().name, "demo");
        assert_eq!(cell.metadata().version, "0.3.1");
        assert_eq!(cell.metadata().archetype, "ecmascript");
        assert_eq!(cell.metadata().entry, "src/main.js");
        assert!(cell.metadata().session_present);
        assert_eq!(cell.files().len(), 3);
        assert_eq!(cell.file("src/main.js").unwrap(), b"export async function run(k, a) { return a; }");
        assert_eq!(cell.file("data/blob.bin").unwrap(), &[0u8, 1, 2, 3, 255]);
        assert_eq!(cell.session().unwrap(), b"{\"t_now\": 0.5}");
    }

    #[test]
    fn cell_without_session_reports_none() {
        let mut b = CellBuilder::new("dry");
        b.add_file("module.toml", b"[module]\nname = \"dry\"\n").unwrap();
        let bytes = b.build().unwrap();
        let cell = Cell::parse(&bytes).unwrap();
        assert!(!cell.has_session());
        assert!(!cell.metadata().session_present);
        assert_eq!(cell.session(), None);
    }

    #[test]
    fn cell_roundtrip_is_lossless_for_empty_module() {
        let b = CellBuilder::new("empty");
        let bytes = b.build().unwrap();
        let cell = Cell::parse(&bytes).unwrap();
        assert_eq!(cell.metadata().name, "empty");
        assert!(cell.files().is_empty());
        assert!(!cell.has_session());
    }

    #[test]
    fn cell_parse_rejects_bad_magic() {
        let mut b = CellBuilder::new("x");
        b.add_file("f", b"data").unwrap();
        let mut bytes = b.build().unwrap();
        bytes[0] = b'N';
        let err = Cell::parse(&bytes).unwrap_err();
        assert!(matches!(err, ArchiveError::BadMagic(m) if m[0] == b'N'));
    }

    #[test]
    fn cell_parse_rejects_unsupported_version() {
        let mut b = CellBuilder::new("x");
        let mut bytes = b.build().unwrap();
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        let err = Cell::parse(&bytes).unwrap_err();
        assert!(matches!(err, ArchiveError::UnsupportedVersion(99)));
    }

    #[test]
    fn cell_parse_rejects_truncated_input() {
        let b = CellBuilder::new("x");
        let bytes = b.build().unwrap();
        assert!(matches!(
            Cell::parse(&bytes[..12]).unwrap_err(),
            ArchiveError::TruncatedHeader
        ));
        assert!(matches!(
            Cell::parse(&bytes[..bytes.len() - 3]).unwrap_err(),
            ArchiveError::TruncatedHeader
        ));
    }

    #[test]
    fn cell_parse_rejects_corrupted_gzip() {
        let mut b = CellBuilder::new("x");
        b.add_file("f", b"data").unwrap();
        let mut bytes = b.build().unwrap();
        // Corrupt the gzip trailer (last bytes of the archive are the compressed body).
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let err = Cell::parse(&bytes).unwrap_err();
        assert!(matches!(err, ArchiveError::BadBody(_)), "got {err:?}");
    }

    #[test]
    fn cell_parse_rejects_absolute_path() {
        let mut b = CellBuilder::new("x");
        let err = b.add_file("/etc/passwd", b"root").unwrap_err();
        assert!(matches!(err, ArchiveError::BadBody(_)));
    }
}
