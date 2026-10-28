//! The store: a per-project index on disk, published to Zenodo with the
//! snapshot + delta convention of `unfer_ffi/src/zenodo.rs` (`docs/altgit.md`,
//! P11.24) — `snapshot_N.bin` once, `delta_N.bin` per later push, a refreshed
//! `manifest.json`, and a squash to a fresh snapshot every `squash_after`
//! deltas. Zenodo keeps every published version immutably; the manifest tells
//! the next reader which files to replay in order.
//!
//! The store's contract is deliberately byte/format-agnostic, exactly like the
//! adapter it mirrors: it uploads *opaque bytes* plus an *opaque frontier*. The
//! document here is the project index as JSON, and a frontier is
//! `{"seq": n}`; swapping in a Loro export (as `unfer_ffi`'s caller does) means
//! changing [`Index::encode_snapshot`] and nothing else.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::http;

#[derive(Clone, Debug)]
pub struct ZenodoConfig {
    /// Token, from `ZENODO_API_KEY`. Never logged.
    pub api_key: String,
    /// `https://sandbox.zenodo.org/api` (default) or `https://zenodo.org/api`.
    pub base_url: String,
    /// Pushes between squash snapshots.
    pub squash_after: usize,
    /// Deposit creator name, `MM0WEB_CREATOR`.
    pub creator: String,
    /// Creator ORCID, `MM0WEB_CREATOR_ORCID` (empty = omit the field).
    pub creator_orcid: String,
    /// The address this service is reachable at, `MM0WEB_PUBLIC_URL`. Deposits
    /// point back at it via `related_identifiers`, which is how a Zenodo reader
    /// gets from the archived artifact to the live project page.
    pub public_url: String,
}

impl Default for ZenodoConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://sandbox.zenodo.org/api".to_string(),
            squash_after: 50,
            creator: "mm0web".to_string(),
            creator_orcid: String::new(),
            public_url: "http://127.0.0.1:8080".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub state_dir: PathBuf,
    pub zenodo: ZenodoConfig,
}

/// A proof's publish state — the fixed vocabulary an audit log and a reader
/// can both rely on (`brief-ideas` keeps a similarly closed set of actions).
///
/// `verified` → `queued` → `published` | `publish-failed`, and a revision of a
/// published proof drops back to `verified` until its push lands.
pub const STATE_VERIFIED: &str = "verified";
pub const STATE_QUEUED: &str = "queued";
pub const STATE_PUBLISHED: &str = "published";
pub const STATE_PUBLISH_FAILED: &str = "publish-failed";

/// One verified proof, as stored locally and published.
#[derive(Clone, Debug)]
pub struct Proof {
    pub id: String,
    pub title: String,
    /// Content fingerprint of the mm0+mm1 bytes. **Not** a cryptographic
    /// hash: it is a local identity for dedup and for the delta chain.
    pub fingerprint: String,
    pub seq: u64,
    pub verified: bool,
    pub mm0: String,
    pub mm1: String,
    pub record: Option<u64>,
    /// One of the `STATE_*` constants above.
    pub state: String,
    /// DOI of the deposit this proof's revision is archived in, once known.
    pub doi: Option<String>,
    /// First-seen time (unix seconds) — what the daily submission cap counts.
    pub created_at: u64,
}

impl Proof {
    fn from_json(id: &str, p: &Value) -> Self {
        Self {
            id: id.to_string(),
            title: p["title"].as_str().unwrap_or("").to_string(),
            fingerprint: p["fingerprint"].as_str().unwrap_or("").to_string(),
            seq: p["seq"].as_u64().unwrap_or(0),
            verified: p["verified"].as_bool().unwrap_or(false),
            mm0: p["mm0"].as_str().unwrap_or("").to_string(),
            mm1: p["mm1"].as_str().unwrap_or("").to_string(),
            record: p["record"].as_u64(),
            state: p["state"].as_str().unwrap_or(STATE_VERIFIED).to_string(),
            doi: p["doi"].as_str().map(str::to_string),
            created_at: p["created_at"].as_u64().unwrap_or(0),
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "title": self.title,
            "fingerprint": self.fingerprint,
            "seq": self.seq,
            "verified": self.verified,
            "record": self.record,
            "state": self.state,
            "doi": self.doi,
            "mm0": self.mm0,
            "mm1": self.mm1,
            "created_at": self.created_at,
        })
    }

    /// The wire form inside a delta: no sources, everything a reader needs to
    /// know about the proof's identity and publication state.
    fn to_delta_json(&self) -> Value {
        json!({
            "title": self.title,
            "fingerprint": self.fingerprint,
            "seq": self.seq,
            "verified": self.verified,
            "record": self.record,
            "state": self.state,
            "doi": self.doi,
            "created_at": self.created_at,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Index {
    pub project: String,
    pub frontier: u64,
    pub proofs: BTreeMap<String, Proof>,
}

impl Index {
    pub fn load(state_dir: &Path, project: &str) -> Self {
        let path = index_path(state_dir, project);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self {
                project: project.to_string(),
                ..Default::default()
            };
        };
        let Ok(v) = serde_json::from_str::<Value>(&text) else {
            return Self {
                project: project.to_string(),
                ..Default::default()
            };
        };
        let mut idx = Index {
            project: project.to_string(),
            frontier: v["frontier"].as_u64().unwrap_or(0),
            proofs: BTreeMap::new(),
        };
        if let Some(map) = v["proofs"].as_object() {
            for (id, p) in map {
                idx.proofs.insert(id.clone(), Proof::from_json(id, p));
            }
        }
        idx
    }

    pub fn save(&self, state_dir: &Path) -> std::io::Result<()> {
        let dir = state_dir.join(slug(&self.project));
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("index.json"), self.to_json().to_string())
    }

    pub fn to_json(&self) -> Value {
        let proofs: serde_json::Map<String, Value> = self
            .proofs
            .iter()
            .map(|(id, p)| (id.clone(), p.to_json()))
            .collect();
        json!({"project": self.project, "frontier": self.frontier, "proofs": proofs})
    }

    /// Insert or revise a proof, bumping the frontier. Returns the identifier.
    /// A *new* proof starts `verified` and gets a `created_at`; a revision of a
    /// known one keeps its identity, its publication record and its clock, but
    /// drops back to `verified` — the archive does not yet hold this content.
    pub fn upsert(&mut self, title: &str, mm0: &str, mm1: &str) -> String {
        let id = proof_id(title, mm0, mm1);
        let fp = fingerprint(mm0, mm1);
        self.frontier += 1;
        let prev = self.proofs.get(&id);
        let record = prev.and_then(|p| p.record);
        let doi = prev.and_then(|p| p.doi.clone());
        let created_at = match prev {
            Some(p) if p.created_at != 0 => p.created_at,
            _ => now_unix(),
        };
        self.proofs.insert(
            id.clone(),
            Proof {
                id: id.clone(),
                title: title.to_string(),
                fingerprint: fp,
                seq: self.frontier,
                verified: true,
                mm0: mm0.to_string(),
                mm1: mm1.to_string(),
                record,
                state: STATE_VERIFIED.to_string(),
                doi,
                created_at,
            },
        );
        id
    }

    /// The full document (snapshot payload).
    pub fn encode_snapshot(&self) -> Vec<u8> {
        self.to_json().to_string().into_bytes()
    }

    /// The document delta since `base_seq` (delta payload).
    pub fn encode_delta(&self, base_seq: u64) -> Vec<u8> {
        let changed: serde_json::Map<String, Value> = self
            .proofs
            .iter()
            .filter(|(_, p)| p.seq > base_seq)
            .map(|(id, p)| (id.clone(), p.to_delta_json()))
            .collect();
        json!({"base_seq": base_seq, "frontier": self.frontier, "proofs": changed})
            .to_string()
            .into_bytes()
    }

    /// The frontier as the store's opaque version marker.
    pub fn encode_frontier(&self) -> Vec<u8> {
        json!({"seq": self.frontier}).to_string().into_bytes()
    }

    /// Replay a snapshot + deltas into an index (what a reader of the deposit
    /// does; the inverse of the two encoders above).
    pub fn replay(project: &str, files: &[Vec<u8>]) -> Result<Index, String> {
        let mut idx = Index {
            project: project.to_string(),
            ..Default::default()
        };
        for (i, bytes) in files.iter().enumerate() {
            let v: Value =
                serde_json::from_slice(bytes).map_err(|e| format!("file {i} is not JSON: {e}"))?;
            if let Some(map) = v["proofs"].as_object() {
                for (id, p) in map {
                    let seq = p["seq"].as_u64().unwrap_or(0);
                    if idx.frontier < seq {
                        idx.frontier = seq;
                    }
                    let mut proof = Proof::from_json(id, p);
                    proof.seq = seq; // the delta's own seq wins, as before
                    idx.proofs.insert(id.clone(), proof);
                }
            }
        }
        Ok(idx)
    }
}

pub fn index_path(state_dir: &Path, project: &str) -> PathBuf {
    state_dir.join(slug(project)).join("index.json")
}

pub fn slug(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    out = out.trim_matches('-').to_string();
    if out.is_empty() {
        out = "project".to_string();
    }
    out.chars().take(64).collect()
}

/// The stable identifier of a submission: `slug(title)-<fingerprint head>`.
/// Content-addressed, so the same proof submitted twice is the same proof —
/// which is what lets the daily cap, the delta chain and the dedup all agree
/// on what "new" means.
pub fn proof_id(title: &str, mm0: &str, mm1: &str) -> String {
    let fp = fingerprint(mm0, mm1);
    format!("{}-{}", slug(title), &fp[..8.min(fp.len())])
}

/// FNV-1a over both sources — a content fingerprint, not a cryptographic hash
/// (collisions are not an attack surface here: the deposit itself is the record
/// of truth, and the fingerprint only names a revision).
pub fn fingerprint(mm0: &str, mm1: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in mm0.as_bytes().iter().chain(mm1.as_bytes()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

/// What a push did, for the response page.
#[derive(Clone, Debug)]
pub struct PushOutcome {
    pub record_id: u64,
    pub url: String,
    pub filename: String,
    pub kind: &'static str,
    pub files: Vec<String>,
    /// DOI of the record, once Zenodo hands one back.
    pub doi: Option<String>,
}

/// The Zenodo client: deposit lifecycle + the snapshot/delta push.
pub struct Zenodo {
    pub cfg: ZenodoConfig,
    state_dir: PathBuf,
}

/// A draft deposition that exists on Zenodo but has not been published yet.
///
/// This is the `zenodo_id` pattern from `brief-ideas`: the id is written to
/// disk *before* any file is uploaded, so a process that dies mid-push leaves
/// a deposit that the next attempt finds and resumes, instead of an orphan.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Draft {
    id: u64,
    bucket: String,
}

/// The payload a push has committed to but may not have finished writing.
/// Re-deriving the filename on resume is what keeps a crash between "chose
/// `delta_3.bin`" and "uploaded it" from producing two half-written deltas.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Inflight {
    filename: String,
    kind: String,
    base_seq: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RemoteState {
    record_id: Option<u64>,
    /// Set as soon as the draft exists, cleared the moment it is published.
    draft: Option<Draft>,
    /// Set as soon as a payload has been named, cleared once it is recorded.
    inflight: Option<Inflight>,
    file_sequence: Vec<String>,
    delta_count: usize,
    /// The index frontier at the last successful push — the base a delta is
    /// encoded against. (Not the delta count: a squash resets that count while
    /// the document keeps moving.)
    last_seq: u64,
}

impl Zenodo {
    pub fn new(cfg: ZenodoConfig, state_dir: PathBuf) -> Self {
        Self { cfg, state_dir }
    }

    pub fn configured(&self) -> bool {
        !self.cfg.api_key.is_empty()
    }

    pub fn record_url(&self, record_id: u64) -> String {
        // records live one path above the API root: /api/deposit/... → /records/…
        let root = self
            .cfg
            .base_url
            .trim_end_matches("/api")
            .trim_end_matches('/')
            .to_string();
        format!("{root}/records/{record_id}")
    }

    fn auth(&self) -> (String, String) {
        (
            "Authorization".to_string(),
            format!("Bearer {}", self.cfg.api_key),
        )
    }

    fn remote_path(&self, project: &str) -> PathBuf {
        self.state_dir.join(slug(project)).join("remote.json")
    }

    fn load_remote(&self, project: &str) -> RemoteState {
        let Ok(text) = std::fs::read_to_string(self.remote_path(project)) else {
            return RemoteState::default();
        };
        let Ok(v) = serde_json::from_str::<Value>(&text) else {
            return RemoteState::default();
        };
        RemoteState {
            record_id: v["record_id"].as_u64(),
            draft: v["draft"].as_object().and_then(|d| {
                Some(Draft {
                    id: d["id"].as_u64()?,
                    bucket: d["bucket"].as_str()?.to_string(),
                })
            }),
            inflight: v["inflight"].as_object().and_then(|f| {
                Some(Inflight {
                    filename: f["filename"].as_str()?.to_string(),
                    kind: f["kind"].as_str()?.to_string(),
                    base_seq: f["base_seq"].as_u64()?,
                })
            }),
            file_sequence: v["file_sequence"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            delta_count: v["delta_count"].as_u64().unwrap_or(0) as usize,
            last_seq: v["last_seq"].as_u64().unwrap_or(0),
        }
    }

    fn save_remote(&self, project: &str, st: &RemoteState) -> Result<(), String> {
        let dir = self.state_dir.join(slug(project));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let v = json!({
            "record_id": st.record_id,
            "draft": st.draft.as_ref().map(|d| json!({"id": d.id, "bucket": d.bucket})),
            "inflight": st.inflight.as_ref().map(|f| json!({
                "filename": f.filename, "kind": f.kind, "base_seq": f.base_seq
            })),
            "file_sequence": st.file_sequence,
            "delta_count": st.delta_count,
            "last_seq": st.last_seq,
        });
        std::fs::write(dir.join("remote.json"), v.to_string()).map_err(|e| e.to_string())
    }

    /// The deposit metadata — a small part of what `brief-ideas` sends with
    /// each Zenodo deposit: a named creator (with ORCID when configured),
    /// keywords, a license, and a `related_identifiers` link from the archived
    /// artifact back to the live project page here.
    fn deposit_metadata(&self, project: &str) -> Value {
        let mut creator = json!({"name": self.cfg.creator});
        if !self.cfg.creator_orcid.is_empty() {
            creator["orcid"] = json!(self.cfg.creator_orcid);
        }
        json!({
            "title": format!("mm0 proofs — {project}"),
            "upload_type": "dataset",
            "description": format!(
                "mm0 proof artifacts published by the mm0web service for the project \
                 `{project}`: a cumulative snapshot plus incremental deltas, ordered by \
                 manifest.json. Verification is the mm0-rs + mm0-c two-stage root of \
                 trust, where a warning from the elaborator is a proof hole."
            ),
            "creators": [creator],
            "keywords": ["mm0", "proof", project],
            "license": "cc-by-4.0",
            "access_right": "open",
            "related_identifiers": [{
                "relation": "isIdenticalTo",
                "identifier": format!(
                    "{}/proofs?project={}",
                    self.cfg.public_url.trim_end_matches('/'),
                    project
                ),
            }],
        })
    }

    fn create_deposit(&self, project: &str) -> Result<(u64, String), String> {
        let url = format!("{}/deposit/depositions", self.cfg.base_url);
        let body = json!({"metadata": self.deposit_metadata(project)}).to_string();
        let (code, text) = http::curl("POST", &url, &[(self.auth())], Some(body.as_bytes()), None)?;
        if !(200..300).contains(&code) {
            return Err(format!("create deposit failed ({code}): {text}"));
        }
        let v: Value =
            serde_json::from_str(&text).map_err(|e| format!("create deposit JSON: {e}"))?;
        let id = v["id"].as_u64().ok_or("no id in create response")?;
        let bucket = v["links"]["bucket"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        Ok((id, bucket))
    }

    /// Fetch a draft we checkpointed earlier to learn whether it still exists
    /// and where its bucket is. `Err` means the checkpoint is stale.
    fn fetch_draft(&self, id: u64) -> Result<String, String> {
        let (code, text) = http::curl(
            "GET",
            &format!("{}/deposit/depositions/{id}", self.cfg.base_url),
            &[(self.auth())],
            None,
            None,
        )?;
        if !(200..300).contains(&code) {
            return Err(format!("draft {id} unavailable ({code}): {text}"));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("draft JSON: {e}"))?;
        v["links"]["bucket"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| "no bucket in draft".to_string())
    }

    fn new_version(&self, record_id: u64) -> Result<(u64, String), String> {
        let url = format!(
            "{}/deposit/depositions/{record_id}/actions/newversion",
            self.cfg.base_url
        );
        let (code, text) = http::curl("POST", &url, &[(self.auth())], None, None)?;
        if !(200..300).contains(&code) {
            return Err(format!("new version failed ({code}): {text}"));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("newversion JSON: {e}"))?;
        // The response carries the *new draft's* links; take its id, or the
        // latest draft id, and fetch the draft to learn the bucket.
        let draft_id = v["links"]["latest_draft"]
            .as_str()
            .and_then(|u| u.rsplit('/').next())
            .and_then(|s| s.parse::<u64>().ok())
            .or_else(|| v["id"].as_u64())
            .ok_or("no draft id in newversion response")?;
        let (code, text) = http::curl(
            "GET",
            &format!("{}/deposit/depositions/{draft_id}", self.cfg.base_url),
            &[(self.auth())],
            None,
            None,
        )?;
        if !(200..300).contains(&code) {
            return Err(format!("draft fetch failed ({code}): {text}"));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("draft JSON: {e}"))?;
        let bucket = v["links"]["bucket"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        Ok((draft_id, bucket))
    }

    fn upload(&self, bucket: &str, name: &str, bytes: &[u8]) -> Result<(), String> {
        if bucket.is_empty() {
            return Err("no bucket URL in the draft — cannot upload".to_string());
        }
        let url = format!("{}/{}", bucket.trim_end_matches('/'), name);
        let (code, text) = http::curl(
            "PUT",
            &url,
            &[
                (self.auth()),
                (
                    "Content-Type".to_string(),
                    "application/octet-stream".to_string(),
                ),
            ],
            Some(bytes),
            None,
        )?;
        if !(200..300).contains(&code) {
            return Err(format!("upload {name} failed ({code}): {text}"));
        }
        Ok(())
    }

    fn publish(&self, deposition_id: u64) -> Result<(u64, Option<String>), String> {
        let url = format!(
            "{}/deposit/depositions/{deposition_id}/actions/publish",
            self.cfg.base_url
        );
        let (code, text) = http::curl("POST", &url, &[(self.auth())], None, None)?;
        if !(200..300).contains(&code) {
            return Err(format!("publish failed ({code}): {text}"));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("publish JSON: {e}"))?;
        let id = v["id"]
            .as_u64()
            .ok_or_else(|| "no id in publish response".to_string())?;
        // Zenodo hands back `doi` and `doi_url`; keep whichever arrives and
        // normalize the URL form so the stored value is always a bare DOI.
        let doi = v["doi"].as_str().map(str::to_string).or_else(|| {
            v["doi_url"].as_str().map(|u| {
                u.trim_start_matches("https://doi.org/")
                    .trim_start_matches("http://dx.doi.org/")
                    .to_string()
            })
        });
        Ok((id, doi))
    }

    /// Publish the project's current index. Snapshot on the first push and
    /// after each squash; delta otherwise. Returns what was written.
    ///
    /// Every step checkpoints `remote.json` before the next one starts, so a
    /// push interrupted anywhere resumes where it stopped instead of starting
    /// a second deposit: draft id → payload name → uploaded sequence → publish.
    pub fn push(&self, index: &Index) -> Result<PushOutcome, String> {
        if !self.configured() {
            return Err(
                "ZENODO_API_KEY is not set — verification works without it, publishing does not"
                    .to_string(),
            );
        }
        let project = &index.project;
        let mut st = self.load_remote(project);

        // 1 — the draft. Reuse a checkpointed one when it is still there.
        let (dep_id, bucket) = match st.draft.clone() {
            Some(d) => match self.fetch_draft(d.id) {
                Ok(bucket) => (d.id, bucket),
                Err(_) => {
                    // Stale checkpoint (published elsewhere, or gone): rebuild
                    // from what we do know rather than trusting a dead id.
                    st.draft = None;
                    self.make_draft(&mut st, project)?
                }
            },
            None => self.make_draft(&mut st, project)?,
        };
        st.draft = Some(Draft {
            id: dep_id,
            bucket: bucket.clone(),
        });
        self.save_remote(project, &st)?;

        // 2 — the payload. Either resume the one already named, or choose it
        // and freeze it (including its delta base) before a byte is uploaded.
        let inflight = match st.inflight.clone() {
            Some(inf) => inf,
            None => {
                let is_first = st.file_sequence.is_empty();
                let squash = !is_first && st.delta_count >= self.cfg.squash_after;
                let kind = if is_first || squash {
                    "snapshot"
                } else {
                    "delta"
                };
                let n = st
                    .file_sequence
                    .iter()
                    .filter(|f| f.starts_with(&format!("{kind}_")))
                    .count();
                let inf = Inflight {
                    filename: format!("{kind}_{n}.bin"),
                    kind: kind.to_string(),
                    base_seq: st.last_seq,
                };
                if squash {
                    st.file_sequence.clear();
                    st.delta_count = 0;
                }
                inf
            }
        };
        let filename = inflight.filename.clone();
        let kind: &'static str = if inflight.kind == "delta" {
            "delta"
        } else {
            "snapshot"
        };
        let payload = if kind == "delta" {
            index.encode_delta(inflight.base_seq)
        } else {
            index.encode_snapshot()
        };
        st.inflight = Some(inflight);
        self.save_remote(project, &st)?;

        // 3 — write the payload, then record it (exactly once: on a resumed
        // push the sequence already ends in this filename).
        self.upload(&bucket, &filename, &payload)?;
        if st.file_sequence.last().map(String::as_str) != Some(filename.as_str()) {
            st.file_sequence.push(filename.clone());
            if kind == "delta" {
                st.delta_count += 1;
            }
        }
        self.save_remote(project, &st)?;

        // 4 — manifest, then publish, then retire the checkpoints together.
        let manifest = json!({
            "project": project,
            "file_sequence": st.file_sequence,
            "last_frontier": base64(&index.encode_frontier()),
            "pushed_at": now_unix(),
        });
        self.upload(&bucket, "manifest.json", manifest.to_string().as_bytes())?;

        let (record_id, doi) = self.publish(dep_id)?;
        st.inflight = None;
        st.draft = None;
        st.record_id = Some(record_id);
        st.last_seq = index.frontier;
        self.save_remote(project, &st)?;

        Ok(PushOutcome {
            record_id,
            url: self.record_url(record_id),
            filename,
            kind,
            files: st.file_sequence.clone(),
            doi,
        })
    }

    /// Open a draft for this project — a new version of the existing record,
    /// or the project's first deposit.
    fn make_draft(&self, st: &mut RemoteState, project: &str) -> Result<(u64, String), String> {
        match st.record_id {
            Some(record) => self.new_version(record),
            None => self.create_deposit(project),
        }
    }

    /// Read a published record back: download every file in manifest order and
    /// replay it into an index. This is the reader half of the deposit, and
    /// the reason the format is worth pinning.
    pub fn pull(&self, record_id: u64, project: &str) -> Result<Index, String> {
        let url = format!("{}/records/{record_id}", self.cfg.base_url);
        let (code, text) = http::curl("GET", &url, &[(self.auth())], None, None)?;
        if !(200..300).contains(&code) {
            return Err(format!("record {record_id} fetch failed ({code}): {text}"));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("record JSON: {e}"))?;
        let mut links = std::collections::HashMap::new();
        if let Some(files) = v["files"].as_array() {
            for f in files {
                if let (Some(k), Some(l)) = (f["key"].as_str(), f["links"]["self"].as_str()) {
                    links.insert(k.to_string(), l.to_string());
                }
            }
        }
        let manifest_url = links
            .get("manifest.json")
            .ok_or("manifest.json is not in the record")?
            .clone();
        let manifest: Value =
            serde_json::from_slice(&http::curl_download(&manifest_url, &[(self.auth())])?)
                .map_err(|e| format!("manifest JSON: {e}"))?;
        let mut files = Vec::new();
        for name in manifest["file_sequence"]
            .as_array()
            .ok_or("manifest has no file_sequence")?
        {
            let name = name.as_str().unwrap_or_default();
            let link = links
                .get(name)
                .ok_or_else(|| format!("{name} listed in manifest but missing from the record"))?;
            files.push(http::curl_download(link, &[(self.auth())])?);
        }
        Index::replay(project, &files)
    }
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// base64 (standard alphabet, padded) — the same opaque encoding the store
/// adapter uses for the frontier.
pub fn base64(bytes: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(CHARS[((n >> 18) & 63) as usize] as char);
        out.push(CHARS[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            CHARS[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            CHARS[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_round_trips_through_snapshot_and_delta() {
        let mut idx = Index {
            project: "unfer".into(),
            ..Default::default()
        };
        let a = idx.upsert("gap certificate", "sort Prop\n", "theorem t: $ Prop;\n");
        let snapshot = idx.encode_snapshot();
        let base = idx.frontier;
        let b = idx.upsert("layout certificate", "sort Prop\n", "theorem u: $ Prop;\n");
        let delta = idx.encode_delta(base);

        let replayed = Index::replay("unfer", &[snapshot, delta.clone()]).unwrap();
        assert_eq!(replayed.frontier, idx.frontier);
        assert_eq!(replayed.proofs.len(), 2);
        assert!(replayed.proofs.contains_key(&a));
        assert!(replayed.proofs.contains_key(&b));
        assert_eq!(replayed.proofs[&b].title, "layout certificate");

        // the delta carries only what changed, and names its base
        let d: Value = serde_json::from_slice(&delta).unwrap();
        assert_eq!(d["base_seq"].as_u64().unwrap(), base);
        assert_eq!(d["proofs"].as_object().unwrap().len(), 1);
    }

    #[test]
    fn revisions_keep_the_identity_and_the_record() {
        let mut idx = Index::default();
        let id = idx.upsert("t", "a", "b");
        idx.proofs.get_mut(&id).unwrap().record = Some(7);
        let again = idx.upsert("t", "a", "b"); // identical content
        assert_eq!(id, again, "the same content keeps the same id");
        assert_eq!(idx.proofs[&id].record, Some(7), "and the published record");
        assert_eq!(idx.frontier, 2, "but the frontier advances");
        let other = idx.upsert("t", "a", "c");
        assert_ne!(id, other, "different content is a different revision");
    }

    #[test]
    fn fingerprints_are_stable_and_content_addressed() {
        assert_eq!(fingerprint("a", "b"), fingerprint("a", "b"));
        assert_ne!(fingerprint("a", "b"), fingerprint("b", "a"));
        assert_ne!(fingerprint("a", "b"), fingerprint("a", "b "));
        assert_eq!(fingerprint("a", "b").len(), 16);
    }

    #[test]
    fn slugs_and_record_urls_are_safe() {
        assert_eq!(slug("unfer proofs/2026"), "unfer-proofs-2026");
        assert_eq!(slug("  "), "project");
        let z = Zenodo::new(ZenodoConfig::default(), PathBuf::from("/tmp"));
        assert_eq!(z.record_url(42), "https://sandbox.zenodo.org/records/42");
        assert!(!z.configured());
        assert!(
            z.push(&Index::default()).is_err(),
            "unconfigured store must refuse"
        );
    }

    #[test]
    fn proof_state_and_doi_survive_a_round_trip() {
        let mut idx = Index::default();
        let id = idx.upsert("t", "a", "b");
        {
            let p = idx.proofs.get_mut(&id).unwrap();
            p.state = STATE_QUEUED.to_string();
            p.doi = Some("10.5281/zenodo.1".to_string());
            p.record = Some(11);
        }
        let v = idx.to_json();
        let re = Proof::from_json(&id, &v["proofs"][id.as_str()]);
        assert_eq!(re.state, STATE_QUEUED);
        assert_eq!(re.doi.as_deref(), Some("10.5281/zenodo.1"));
        assert_eq!(re.record, Some(11));
        assert_eq!(re.created_at, idx.proofs[&id].created_at);
        // an index written to disk reads back the same way
        let dir = temp_state_dir("save-rt");
        idx.save(&dir).unwrap();
        let loaded = Index::load(&dir, "");
        assert_eq!(loaded.proofs[&id].state, STATE_QUEUED);
        assert_eq!(loaded.proofs[&id].doi.as_deref(), Some("10.5281/zenodo.1"));
        assert!(loaded.proofs[&id].created_at > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_revision_keeps_its_identity_clock_and_record_but_re_verifies() {
        let mut idx = Index::default();
        let id = idx.upsert("t", "a", "b");
        idx.proofs.get_mut(&id).unwrap().state = STATE_PUBLISHED.to_string();
        idx.proofs.get_mut(&id).unwrap().record = Some(5);
        let created = idx.proofs[&id].created_at;

        let same = idx.upsert("t", "a", "b");
        assert_eq!(same, id);
        assert_eq!(
            idx.proofs[&id].created_at, created,
            "first-seen time sticks"
        );
        assert_eq!(idx.proofs[&id].record, Some(5));
        assert_eq!(idx.proofs[&id].state, STATE_VERIFIED);
    }

    #[test]
    fn remote_checkpoints_round_trip() {
        let dir = temp_state_dir("remote");
        let z = Zenodo::new(ZenodoConfig::default(), dir.clone());
        let mut st = RemoteState {
            record_id: Some(3),
            draft: Some(Draft {
                id: 4,
                bucket: "https://b/4".into(),
            }),
            inflight: Some(Inflight {
                filename: "delta_2.bin".into(),
                kind: "delta".into(),
                base_seq: 7,
            }),
            file_sequence: vec!["snapshot_0.bin".into(), "delta_1.bin".into()],
            delta_count: 2,
            last_seq: 7,
        };
        z.save_remote("p", &st).unwrap();
        let back = z.load_remote("p");
        assert_eq!(back, st);

        st.inflight = None;
        st.draft = None;
        z.save_remote("p", &st).unwrap();
        let cleared = z.load_remote("p");
        assert!(cleared.draft.is_none() && cleared.inflight.is_none());
        assert_eq!(cleared.record_id, Some(3));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deposit_metadata_names_a_creator_and_points_back_at_the_service() {
        let cfg = ZenodoConfig {
            creator: "Ada Lovelace".into(),
            creator_orcid: "0000-0002-1825-0097".into(),
            public_url: "https://proofs.example.org/".into(),
            ..Default::default()
        };
        let z = Zenodo::new(cfg, PathBuf::from("/tmp"));
        let m = z.deposit_metadata("unfer");
        assert_eq!(m["creators"][0]["name"], "Ada Lovelace");
        assert_eq!(m["creators"][0]["orcid"], "0000-0002-1825-0097");
        assert_eq!(m["keywords"][2], "unfer");
        assert_eq!(m["license"], "cc-by-4.0");
        assert_eq!(
            m["related_identifiers"][0]["identifier"],
            "https://proofs.example.org/proofs?project=unfer"
        );
        assert_eq!(m["related_identifiers"][0]["relation"], "isIdenticalTo");

        // no ORCID configured → the field is omitted, not sent empty
        let bare = Zenodo::new(ZenodoConfig::default(), PathBuf::from("/tmp"));
        let m = bare.deposit_metadata("p");
        assert!(m["creators"][0].get("orcid").is_none());
    }

    fn temp_state_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mm0web-store-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn base64_matches_the_standard_alphabet() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"hello world"), "aGVsbG8gd29ybGQ=");
    }
}
