mod handles;

use std::panic::{AssertUnwindSafe, catch_unwind};

use prob_kernel::{Session, SessionBlob};
use unfer_protocol::{
    BayesianUpdateRequest, BayesianUpdateResult, Code, Diagnostic, EventPredicate, EventQuery,
    HamiltonianSpec, ModelSpec, PriorSpec, Severity,
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
        let event = unfer_protocol::KernelEvent::PriorSet;
        handles::push_event(model, event);
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
        handles::push_event(model, unfer_protocol::KernelEvent::HamiltonianSet);
        Ok(0)
    })
}

/// Evolve the state forward. `opts_json` is `{"t": <seconds>, "query": [<f64; d>]?}`.
/// The optional `query` field is required for QFM tomographic models
/// (Workstream F) and must be a d-dim vector matching the training data
/// dimension. Result JSON (an `EvolveReport`) is retrievable via
/// `uk_get_result`. Also enqueues an `evolved` event for `uk_poll`
/// subscribers. Returns 0 on success, <0 (-code) on error.
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
        // Optional query for QFM tomographic models.
        let query: Option<Vec<f64>> = opts
            .get("query")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect());
        let query_slice: Option<Vec<f64>> = query;
        let report =
            handles::with_session_mut(model, |s| s.evolve_with_query(t, query_slice.as_deref()))
                .ok_or_else(|| bad_handle(model))?
                .map_err(|e| e.to_diagnostic())?;
        let result_json = serde_json::to_string(&report).unwrap_or_else(|_| "{}".to_string());
        let event = unfer_protocol::KernelEvent::Evolved {
            t: report.t,
            norm: report.norm,
            solve_ms: report.solve_ms,
        };
        handles::set_last_result(model, result_json);
        handles::push_event(model, event);
        Ok(0)
    })
}

/// Condition the state on an event (Bayesian update).
/// `event_json` is an `EventPredicate` JSON.
/// Result JSON `{"prior_probability": <f64>}` is retrievable via `uk_get_result`.
/// Also enqueues a `conditioned` event for `uk_poll` subscribers.
/// Returns 0 on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_condition(model: i64, event_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_condition", || {
        let event: EventPredicate = parse_json(event_json, len)?;
        let prior_p = handles::with_session_mut(model, |s| s.condition(&event))
            .ok_or_else(|| bad_handle(model))?
            .map_err(|e| e.to_diagnostic())?;
        let result_json = serde_json::json!({"prior_probability": prior_p}).to_string();
        let event = unfer_protocol::KernelEvent::Conditioned {
            prior_probability: prior_p,
        };
        handles::set_last_result(model, result_json);
        handles::push_event(model, event);
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
/// Also enqueues an `observed` event for `uk_poll` subscribers.
/// Returns 0 on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_observe(model: i64, obs_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_observe", || {
        let event: EventPredicate = parse_json(obs_json, len)?;
        let prior_p = handles::with_session_mut(model, |s| s.condition(&event))
            .ok_or_else(|| bad_handle(model))?
            .map_err(|e| e.to_diagnostic())?;
        let result_json = serde_json::json!({"prior_probability": prior_p}).to_string();
        let event = unfer_protocol::KernelEvent::Observed { value: prior_p };
        handles::set_last_result(model, result_json);
        handles::push_event(model, event);
        Ok(0)
    })
}

/// Quantum Bayesian Update on the TSR-evolved prior
/// (QMF.tex §8, P6 H follow-on).
///
/// `req_json` is a `BayesianUpdateRequest` JSON:
///   `{"observations": [[f64; d], ...], "hmc_opts": {...}}`
///
/// Returns 0 on success (the result is retrievable via `uk_get_result`).
/// Also enqueues a `conditioned` event for `uk_poll` subscribers
/// (the Bayesian update is morally a conditioning op, just on a
/// TSR-prior posterior rather than the SIRK state). Returns
/// `UK-1001` for malformed JSON, `UK-1004` for an invalid model
/// handle, `UK-5000` for non-QFM models.
#[unsafe(no_mangle)]
pub extern "C" fn uk_bayesian_update(model: i64, req_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_bayesian_update", || {
        let req: BayesianUpdateRequest = parse_json(req_json, len)?;
        let report = handles::with_session_mut(model, |s| {
            s.bayesian_update(&req.observations, &req.hmc_opts)
        })
        .ok_or_else(|| bad_handle(model))?
        .map_err(|e| e.to_diagnostic())?;
        let result = BayesianUpdateResult {
            log_posterior: report.log_posterior,
            mean_likelihood: report.mean_likelihood,
            image: report.image,
            n_observations: report.n_observations,
            solve_ms: report.solve_ms,
        };
        let result_json = serde_json::to_string(&result)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        handles::set_last_result(model, result_json);
        // Use the existing 'conditioned' event vocabulary: a Bayesian
        // update is morally a conditioning op (just on a TSR-prior
        // posterior rather than the SIRK state). The mean_likelihood
        // is reported as a probability-like value (clamped to [0, 1]).
        let prior_p = report.mean_likelihood.clamp(0.0, 1.0);
        let event = unfer_protocol::KernelEvent::Conditioned {
            prior_probability: prior_p,
        };
        handles::push_event(model, event);
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

/// Register interest in a model's event stream.
/// `query_json` is an `EventQuery` JSON (`{}` accepts all event types).
/// Returns a positive subscription handle on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_subscribe(model: i64, query_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_subscribe", || {
        let query: EventQuery = parse_json(query_json, len)?;
        handles::create_subscription(model, query).map_err(|_| bad_handle(model))
    })
}

/// Poll the next pending event from a subscription (returned by `uk_subscribe`).
///
/// Buffer protocol: peek at the event and return its byte length; if `buf` is
/// non-null and `cap` > 0, also pop the event and copy `min(needed, cap)` bytes.
/// Callers should first probe with `buf=NULL, cap=0` to learn the size, allocate,
/// then call again with a real buffer — the event stays in the queue until the
/// second call. Returns 0 if no events are pending, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_poll(sub: i64, buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_poll", || match handles::peek_subscription(sub) {
        None => Err(bad_handle(sub)),
        Some(None) => Ok(0),
        Some(Some(event_json)) => {
            let needed = event_json.len() as i64;
            if cap > 0 && !buf.is_null() {
                handles::poll_subscription(sub); // consume
                let copy_len = std::cmp::min(needed, cap) as usize;
                unsafe {
                    let src = event_json.as_bytes().as_ptr();
                    std::ptr::copy_nonoverlapping(src, buf, copy_len);
                }
            }
            Ok(needed)
        }
    })
}

// ── tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn json_ptr(s: &str) -> (*const u8, i64) {
        (s.as_ptr(), s.len() as i64)
    }

    fn read_buf(f: impl Fn(*mut u8, i64) -> i64) -> String {
        let needed = f(std::ptr::null_mut(), 0);
        assert!(needed >= 0, "unexpected error probing buffer size");
        let mut buf = vec![0u8; needed as usize];
        f(buf.as_mut_ptr(), needed);
        String::from_utf8(buf).unwrap()
    }

    fn create_harmonic_model() -> i64 {
        let spec = r#"{"hamiltonian":{"kind":"builtin","name":"harmonic_chain","params":{"n_modes":2,"omega":1.0}},"prior":{"kind":"vacuum"},"solver":{"krylov_dim":4,"prune_eps":1e-12,"max_components":null,"restarts":1,"device":{"kind":"cpu"},"adaptive":false}}"#;
        let (ptr, len) = json_ptr(spec);
        uk_model_create(ptr, len)
    }

    #[test]
    fn version_returns_one() {
        assert_eq!(uk_version(), 1);
    }

    #[test]
    fn create_free_happy_path() {
        let h = create_harmonic_model();
        assert!(h > 0, "expected positive handle, got {h}");
        assert_eq!(uk_model_free(h), 0);
        assert!(uk_model_free(h) < 0, "double-free must fail");
    }

    #[test]
    fn bad_handle_returns_neg1004() {
        assert_eq!(uk_model_free(99999), -1004);
    }

    #[test]
    fn bad_json_returns_neg1001() {
        let (ptr, len) = json_ptr("not json");
        assert_eq!(uk_model_create(ptr, len), -1001);
    }

    fn subscribe(h: i64) -> i64 {
        let (ptr, len) = json_ptr("{}");
        let sub = uk_subscribe(h, ptr, len);
        assert!(sub > 0, "subscription handle must be positive, got {sub}");
        sub
    }

    #[test]
    fn evolve_enqueues_event() {
        let h = create_harmonic_model();
        assert!(h > 0);

        let sub = subscribe(h);

        // No events yet.
        let mut buf = [0u8; 256];
        assert_eq!(
            uk_poll(sub, buf.as_mut_ptr(), 256),
            0,
            "queue must be empty before any op"
        );

        // Evolve.
        let opts = r#"{"t":0.01}"#;
        let (ptr, len) = json_ptr(opts);
        assert_eq!(uk_evolve(h, ptr, len), 0);

        // Poll the event.
        let event_json = read_buf(|b, c| uk_poll(sub, b, c));
        let event: serde_json::Value = serde_json::from_str(&event_json).unwrap();
        assert_eq!(event["type"], "evolved");
        assert!(event["t"].as_f64().unwrap() > 0.0);
        assert!(event["norm"].as_f64().unwrap() > 0.99);
        assert!(event["solve_ms"].as_u64().is_some());

        // Queue empty again.
        assert_eq!(uk_poll(sub, buf.as_mut_ptr(), 256), 0);

        uk_model_free(h);
    }

    #[test]
    fn condition_enqueues_event() {
        let h = create_harmonic_model();
        let sub = subscribe(h);
        // Condition on the vacuum (should succeed — vacuum prior has mass 1).
        let event = r#"{"kind":"vacuum"}"#;
        let (ptr, len) = json_ptr(event);
        assert_eq!(uk_condition(h, ptr, len), 0);

        let evt_json = read_buf(|b, c| uk_poll(sub, b, c));
        let evt: serde_json::Value = serde_json::from_str(&evt_json).unwrap();
        assert_eq!(evt["type"], "conditioned");
        assert!((evt["prior_probability"].as_f64().unwrap() - 1.0).abs() < 1e-6);

        uk_model_free(h);
    }

    #[test]
    fn set_prior_enqueues_event() {
        let h = create_harmonic_model();
        let sub = subscribe(h);
        let prior = r#"{"kind":"vacuum"}"#;
        let (ptr, len) = json_ptr(prior);
        assert_eq!(uk_set_prior(h, ptr, len), 0);

        let evt_json = read_buf(|b, c| uk_poll(sub, b, c));
        let evt: serde_json::Value = serde_json::from_str(&evt_json).unwrap();
        assert_eq!(evt["type"], "prior_set");

        uk_model_free(h);
    }

    #[test]
    fn queue_drops_oldest_when_full() {
        let h = create_harmonic_model();
        let sub = subscribe(h);
        // Push exactly CAPACITY+1 set_prior events; the first must be dropped.
        let prior = r#"{"kind":"vacuum"}"#;
        let (ptr, len) = json_ptr(prior);
        for _ in 0..=handles::EVENT_QUEUE_CAPACITY {
            uk_set_prior(h, ptr, len);
        }
        // Drain queue.
        let mut count = 0usize;
        let mut buf = vec![0u8; 64];
        while uk_poll(sub, buf.as_mut_ptr(), buf.len() as i64) > 0 {
            count += 1;
        }
        assert_eq!(
            count,
            handles::EVENT_QUEUE_CAPACITY,
            "queue must hold exactly CAPACITY events (oldest dropped)"
        );

        uk_model_free(h);
    }

    #[test]
    fn poll_bad_handle_returns_neg1004() {
        let mut buf = [0u8; 64];
        assert_eq!(uk_poll(99999, buf.as_mut_ptr(), 64), -1004);
    }

    #[test]
    fn subscribe_bad_handle_returns_neg1004() {
        let (ptr, len) = json_ptr("{}");
        assert_eq!(uk_subscribe(99999, ptr, len), -1004);
    }

    #[test]
    fn subscribe_filters_by_event_type() {
        let h = create_harmonic_model();
        // Subscribe to only "evolved" events.
        let (qptr, qlen) = json_ptr(r#"{"types":["evolved"]}"#);
        let sub = uk_subscribe(h, qptr, qlen);
        assert!(sub > 0);

        // Push a prior_set event — must be filtered out.
        let (ptr, len) = json_ptr(r#"{"kind":"vacuum"}"#);
        assert_eq!(uk_set_prior(h, ptr, len), 0);

        let mut buf = [0u8; 256];
        assert_eq!(
            uk_poll(sub, buf.as_mut_ptr(), 256),
            0,
            "prior_set must be filtered out by evolved-only query"
        );

        // Evolve — this event must pass the filter.
        let (eptr, elen) = json_ptr(r#"{"t":0.01}"#);
        assert_eq!(uk_evolve(h, eptr, elen), 0);
        let evt_json = read_buf(|b, c| uk_poll(sub, b, c));
        let evt: serde_json::Value = serde_json::from_str(&evt_json).unwrap();
        assert_eq!(evt["type"], "evolved");

        uk_model_free(h);
    }

    #[test]
    fn snapshot_restore_roundtrip() {
        let h = create_harmonic_model();
        let blob_json = read_buf(|b, c| uk_snapshot(h, b, c));
        assert!(!blob_json.is_empty());

        let (ptr, len) = json_ptr(&blob_json);
        let h2 = uk_restore(ptr, len);
        assert!(h2 > 0 && h2 != h);

        uk_model_free(h);
        uk_model_free(h2);
    }
}
