//! Blueprint content plane (S20, F19): `POST /api/blueprint/import` publishes a
//! verified `.cell` archive into the kernel blueprint registry and seeds the edge
//! `/cell/<cid>` content route, so registered blueprints surface through the content
//! gateway (S15). Requires `--features audit` (links the kernel FFI in-process).
//!
//! * `POST /api/blueprint/import` body `{"cell_hex": "<hex of the .cell>"}` — mint an
//!   immutable, content-addressed `BlueprintRecord` under the publisher principal and
//!   seed the store backing `/cell/<blueprint_id>`.

use serde_json::Value;

/// The publishing principal that mints blueprint records at the kernel chokepoint.
const PUBLISHER: &str = r#"{"from":"gadget","principal":"app_publisher"}"#;

/// True when `path` belongs to the blueprint console. The caller dispatches on the
/// exact route (method + suffix).
pub fn is_blueprint_path(path: &str) -> bool {
    path.starts_with("/api/blueprint/")
}

/// Decode a lowercase-hex string (the transport encoding for binary cells).
pub fn from_hex(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err("cell_hex must have an even number of digits".to_string());
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    for pair in hex.as_bytes().chunks_exact(2) {
        let hi = (pair[0] as char)
            .to_digit(16)
            .ok_or("cell_hex is not hex")?;
        let lo = (pair[1] as char)
            .to_digit(16)
            .ok_or("cell_hex is not hex")?;
        out.push((hi * 16 + lo) as u8);
    }
    Ok(out)
}

/// Mint a blueprint from the raw `.cell` bytes.
///
/// Registers the archive at the kernel (content-CID addressed, immutable) and seeds
/// the edge content store at the same address so `GET /cell/<cid>` resolves. On
/// success returns `(200, record_json)`; a malformed archive or kernel refusal
/// returns `(400|500, {"error": ..., "code": N})`.
pub fn import_record(cell: &[u8]) -> Result<Vec<u8>, String> {
    // The publisher principal mints the record at the kernel chokepoint.
    unfer_ffi::uk_set_caller(PUBLISHER).map_err(|e| e.to_string())?;
    // Two-call buffer protocol: probe for the record size, then receive it. The
    // registration itself is content-addressed idempotent, so the probe call and the
    // receive call seal the same immutable record.
    let needed =
        unfer_ffi::uk_blueprint_import(cell.as_ptr(), cell.len() as i64, std::ptr::null_mut(), 0);
    if needed < 0 {
        unfer_ffi::uk_clear_caller();
        return Err(format!("blueprint import refused ({needed})"));
    }
    let mut buf = vec![0u8; needed as usize];
    let n =
        unfer_ffi::uk_blueprint_import(cell.as_ptr(), cell.len() as i64, buf.as_mut_ptr(), needed);
    unfer_ffi::uk_clear_caller();
    if n < 0 {
        return Err(format!("blueprint import receive failed ({n})"));
    }
    let record: Value = serde_json::from_slice(&buf).map_err(|e| e.to_string())?;
    let cid = record["blueprint_id"]
        .as_str()
        .ok_or_else(|| "record lacks blueprint_id".to_string())?
        .to_string();

    // Seed the content gateway at the blueprint's content address.
    crate::cell_store()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .store(cell);
    serde_json::to_vec(&serde_json::json!({ "ok": true, "record": record, "cid": cid }))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::CellBuilder;

    fn sample_cell() -> Vec<u8> {
        let mut b = CellBuilder::new("demo");
        b.set_archetype("ecmascript");
        b.set_entry("src/main.js");
        b.add_file("module.toml", b"[module]\nname = \"demo\"\n")
            .unwrap();
        b.add_file("src/main.js", b"export async function run(k,a){return a;}")
            .unwrap();
        b.set_session(br#"{"t_now":0.25}"#);
        b.build().unwrap()
    }

    #[test]
    fn hex_transport_roundtrip() {
        let cell = sample_cell();
        let hex: String = cell.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(from_hex(&hex).unwrap(), cell);
        assert!(from_hex("abc").is_err(), "odd hex length must be refused");
        assert!(
            from_hex("zz").is_err(),
            "non-hex characters must be refused"
        );
    }

    #[test]
    fn import_mints_and_seeds_content_gateway() {
        let _lock = crate::gate::tests::store_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unfer_ffi::uk_audit_clear();
        unfer_ffi::uk_clear_caller();
        unfer_data::blueprint::clear_global_registry();

        let cell = sample_cell();
        let body = import_record(&cell).expect("import must mint");
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["ok"], true, "{v}");
        let cid = v["cid"].as_str().unwrap().to_string();
        assert_eq!(cid.len(), 64);
        assert_eq!(v["record"]["created_by"], "app_publisher");
        assert_eq!(v["record"]["immutable_blueprint_id"], cid);

        // The minted address is immediately resolvable through the content gateway.
        let (status, res) = crate::cells::resolve_cell(
            &crate::cell_store()
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
            &format!("/cell/{cid}"),
        );
        assert_eq!(status, 200, "{cid}: {res:?}");
        let meta: Value = serde_json::from_slice(&res).unwrap();
        assert_eq!(meta["present"], true);
        assert_eq!(meta["filesize"], cell.len() as u64);
    }

    #[test]
    fn import_refuses_malformed_archive() {
        let _lock = crate::gate::tests::store_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unfer_ffi::uk_clear_caller();
        let err = import_record(b"NOTACEL1-definitely-not-a-cell-archive").unwrap_err();
        assert!(
            err.contains("refused"),
            "tampered archive must be refused: {err}"
        );
    }
}
