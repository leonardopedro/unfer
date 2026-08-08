//! Edge content-plane read route — `GET /cell/<cid>` (S15, F14).
//!
//! Resolves a content-address against the process-local [`CellStore`] (S20 will seed
//! blueprints through it). If the cell is present it returns its stored metadata;
//! otherwise a well-formed CID returns a 404 and a malformed one returns 400, so the
//! gateway never guesses at content by address shape alone.

use unfer_data::blueprint::is_content_cid;
use unfer_data::CellStore;

/// Extract the CID segment from a `/cell/<cid>` path (`""` when not a cell path).
pub fn cid_from_path(path: &str) -> Option<String> {
    let rest = path.strip_prefix("/cell/")?;
    if rest.is_empty() {
        return None;
    }
    Some(rest.split('/').next().unwrap_or_default().to_string())
}

/// Resolve a `/cell/<cid>` request against a store.
///
/// Returns `(status, json_body)`.
pub fn resolve_cell(store: &CellStore, path: &str) -> (u16, Vec<u8>) {
    let Some(cid) = cid_from_path(path) else {
        return (400, body_json(&serde_json::json!({ "error": "not a /cell/<cid> path" })));
    };
    if !is_content_cid(&cid) {
        return (400, body_json(&serde_json::json!({ "error": "malformed content CID", "cid": cid })));
    }
    match store.get(&cid) {
        Some(rec) => {
            let meta = serde_json::json!({
                "cid": cid,
                "present": true,
                "pins": rec.pins,
                "filesize": rec.cell_ref.filesize,
                "magnet": rec.cell_ref.magnet_uri,
            });
            (200, body_json(&meta))
        }
        None => (
            404,
            body_json(&serde_json::json!({ "error": "cell not found", "cid": cid })),
        ),
    }
}

fn body_json(v: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_else(|_| b"{\"error\":\"serialize\"}".to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid_of(cell: &[u8]) -> String {
        unfer_data::blueprint::store_cell(cell).cid
    }

    #[test]
    fn cid_path_parsing() {
        assert_eq!(cid_from_path("/cell/abc").as_deref(), Some("abc"));
        assert_eq!(cid_from_path("/cell/abc/").as_deref(), Some("abc"));
        let with_query = "/cell/abc?verify=true";
        let no_query = with_query.split('?').next().unwrap_or(with_query);
        assert_eq!(cid_from_path(no_query).as_deref(), Some("abc"));
        assert_eq!(cid_from_path("/cell/"), None);
        assert_eq!(cid_from_path("/audit"), None);
    }

    #[test]
    fn resolve_present_and_absent_cells() {
        let cell = b"blueprint-cell-bytes";
        let mut store = CellStore::new();
        let stored = store.store(cell);
        let cid = stored.cid.clone();

        let (status, body) = resolve_cell(&store, &format!("/cell/{cid}"));
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["present"], true);
        assert_eq!(v["cid"], cid);
        assert_eq!(v["filesize"], cell.len() as u64);

        // An un-stored but well-formed CID is a 404, never a synthesized 200.
        let other = cid_of(b"other");
        assert_ne!(other, cid);
        let (status404, _) = resolve_cell(&store, &format!("/cell/{other}"));
        assert_eq!(status404, 404);

        // Malformed → 400.
        let (bad_status, bad_body) = resolve_cell(&store, "/cell/zzz");
        assert_eq!(bad_status, 400);
        let v: serde_json::Value = serde_json::from_slice(&bad_body).unwrap();
        assert_eq!(v["error"], "malformed content CID");
    }

    #[test]
    fn resolve_non_cell_path_is_400() {
        let store = CellStore::new();
        let (status, _) = resolve_cell(&store, "/audit");
        assert_eq!(status, 400);
    }
}