mod handles;

use std::panic::{AssertUnwindSafe, catch_unwind};

use prob_kernel::{Session, SessionBlob};
use unfer_protocol::{
    Code, Diagnostic, EventPredicate, HamiltonianSpec, ModelSpec, PriorSpec, Severity,
};

pub use unfer_protocol;

const VERSION: i64 = 1;

// ── helpers ──────────────────────────────────────────────────────────

fn fail(diag: Diagnostic) -> i64 {
    handles::set_last_error(&diag);
    -(diag.code.raw() as i64)
}

fn fail_code(code: Code, msg: impl Into<String>) -> i64 {
    fail(Diagnostic::new(code, msg, Severity::Error))
}

fn bad_handle(handle: i64) -> Diagnostic {
    Diagnostic::new(
        Code::BAD_HANDLE,
        format!("invalid model handle: {handle}"),
        Severity::Error,
    )
}

fn ffi_entry(func_name: &str, f: impl FnOnce() -> Result<i64, Diagnostic>) -> i64 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(val)) => val,
        Ok(Err(diag)) => fail(diag),
        Err(_) => fail_code(Code::INTERNAL, format!("panic in {func_name}")),
    }
}

fn parse_json<T: serde::de::DeserializeOwned>(ptr: *const u8, len: i64) -> Result<T, Diagnostic> {
    if len < 0 {
        return Err(Diagnostic::new(
            Code::BAD_JSON,
            "negative length",
            Severity::Error,
        ));
    }
    if ptr.is_null() && len > 0 {
        return Err(Diagnostic::new(
            Code::BAD_JSON,
            "null pointer with non-zero length",
            Severity::Error,
        ));
    }
    let slice = if len == 0 {
        &b""[..]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len as usize) }
    };
    let json_str = match std::str::from_utf8(slice) {
        Ok(s) => s,
        Err(e) => {
            return Err(Diagnostic::new(
                Code::BAD_JSON,
                format!("invalid UTF-8: {e}"),
                Severity::Error,
            ));
        }
    };
    serde_json::from_str(json_str)
        .map_err(|e| Diagnostic::new(Code::BAD_JSON, e.to_string(), Severity::Error))
}

fn write_buf(buf: *mut u8, cap: i64, data: &str) -> i64 {
    let needed = data.len() as i64;
    if cap <= 0 || buf.is_null() {
        return needed;
    }
    let copy_len = std::cmp::min(needed, cap) as usize;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf, copy_len);
    }
    needed
}

// ── ABI functions ─────────────────────────────────────────────────────

/// Return the ABI version (currently 1).
#[unsafe(no_mangle)]
pub extern "C" fn uk_version() -> i64 {
    VERSION
}

/// Initialize the kernel. `cfg_json` is optional (`{}` is accepted).
/// Returns 0 on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_init(_cfg_json: *const u8, _len: i64) -> i64 {
    ffi_entry("uk_init", || {
        handles::ensure_init();
        Ok(0)
    })
}

/// Create a model session from a `ModelSpec` JSON.
/// Returns a positive handle on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_model_create(spec_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_model_create", || {
        let spec: ModelSpec = parse_json(spec_json, len)?;
        let session = Session::new(&spec).map_err(|e| e.to_diagnostic())?;
        Ok(handles::store_session(session))
    })
}

/// Free a model session. Returns 0 on success, -1004 if the handle is invalid.
#[unsafe(no_mangle)]
pub extern "C" fn uk_model_free(model: i64) -> i64 {
    if handles::free_session(model) {
        0
    } else {
        fail(bad_handle(model))
    }
}

/// Replace the prior state. `json` is a `PriorSpec` JSON.
/// Returns 0 on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_set_prior(model: i64, json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_set_prior", || {
        let prior: PriorSpec = parse_json(json, len)?;
        let result = handles::with_session_mut(model, |s| s.set_prior(&prior));
        result
            .ok_or_else(|| bad_handle(model))?
            .map_err(|e| e.to_diagnostic())?;
        Ok(0)
    })
}

/// Replace the Hamiltonian. `json` is a `HamiltonianSpec` JSON.
/// Returns 0 on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_set_hamiltonian(model: i64, json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_set_hamiltonian", || {
        let ham: HamiltonianSpec = parse_json(json, len)?;
        let result = handles::with_session_mut(model, |s| s.set_hamiltonian(&ham));
        result
            .ok_or_else(|| bad_handle(model))?
            .map_err(|e| e.to_diagnostic())?;
        Ok(0)
    })
}

/// Evolve the state forward. `opts_json` is `{"t": <seconds>}`.
/// Result JSON (an `EvolveReport`) is retrievable via `uk_get_result`.
/// Returns 0 on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_evolve(model: i64, opts_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_evolve", || {
        let opts: serde_json::Value = parse_json(opts_json, len)?;
        let t = opts.get("t").and_then(|v| v.as_f64()).ok_or_else(|| {
            Diagnostic::new(
                Code::BAD_JSON,
                "missing or invalid 't' in evolve opts",
                Severity::Error,
            )
        })?;
        let report = handles::with_session_mut(model, |s| s.evolve(t))
            .ok_or_else(|| bad_handle(model))?
            .map_err(|e| e.to_diagnostic())?;
        let json = serde_json::to_string(&report).unwrap_or_else(|_| "{}".to_string());
        handles::set_last_result(model, json);
        Ok(0)
    })
}

/// Condition the state on an event (Bayesian update).
/// `event_json` is an `EventPredicate` JSON.
/// Result JSON `{"prior_probability": <f64>}` is retrievable via `uk_get_result`.
/// Returns 0 on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_condition(model: i64, event_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_condition", || {
        let event: EventPredicate = parse_json(event_json, len)?;
        let prior_p = handles::with_session_mut(model, |s| s.condition(&event))
            .ok_or_else(|| bad_handle(model))?
            .map_err(|e| e.to_diagnostic())?;
        let json = serde_json::json!({"prior_probability": prior_p}).to_string();
        handles::set_last_result(model, json);
        Ok(0)
    })
}

/// Compute the Born-rule probability of an event without modifying the state.
/// `event_json` is an `EventPredicate` JSON.
/// Result JSON `{"probability": <f64>}` is retrievable via `uk_get_result`.
/// Returns 0 on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_event_probability(model: i64, event_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_event_probability", || {
        let event: EventPredicate = parse_json(event_json, len)?;
        let prob = handles::with_session_mut(model, |s| s.probability(&event))
            .ok_or_else(|| bad_handle(model))?
            .map_err(|e| e.to_diagnostic())?;
        let json = serde_json::json!({"probability": prob}).to_string();
        handles::set_last_result(model, json);
        Ok(0)
    })
}

/// Observe an event (v1: alias for `uk_condition`).
/// `obs_json` is an `EventPredicate` JSON.
/// Returns 0 on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_observe(model: i64, obs_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_observe", || {
        let event: EventPredicate = parse_json(obs_json, len)?;
        let prior_p = handles::with_session_mut(model, |s| s.condition(&event))
            .ok_or_else(|| bad_handle(model))?
            .map_err(|e| e.to_diagnostic())?;
        let json = serde_json::json!({"prior_probability": prior_p}).to_string();
        handles::set_last_result(model, json);
        Ok(0)
    })
}

/// Retrieve the JSON result of the last operation (evolve / condition /
/// probability).  Buffer protocol: returns total bytes needed; copies
/// `min(needed, cap)` into `buf`.  Returns <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_get_result(model: i64, buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_get_result", || match handles::get_last_result(model) {
        Some(json) if !json.is_empty() => Ok(write_buf(buf, cap, &json)),
        Some(_) => Ok(0),
        None => Err(bad_handle(model)),
    })
}

/// Retrieve the last error as a `Diagnostic` JSON string.
/// Buffer protocol: returns total bytes needed.
#[unsafe(no_mangle)]
pub extern "C" fn uk_last_error(buf: *mut u8, cap: i64) -> i64 {
    let error = handles::get_last_error();
    if error.is_empty() {
        return 0;
    }
    write_buf(buf, cap, &error)
}

/// Serialize the session to a `SessionBlob` JSON string.
/// Buffer protocol: returns total bytes needed; copies min(needed, cap) into buf.
/// Returns <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_snapshot(model: i64, buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_snapshot", || {
        let blob =
            handles::with_session_mut(model, |s| s.save()).ok_or_else(|| bad_handle(model))?;
        let json = serde_json::to_string(&blob)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// Create a new session from a `SessionBlob` JSON string (produced by `uk_snapshot`).
/// Returns a positive handle on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_restore(blob_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_restore", || {
        let blob: SessionBlob = parse_json(blob_json, len)?;
        let session = Session::restore(blob).map_err(|e| e.to_diagnostic())?;
        Ok(handles::store_session(session))
    })
}
