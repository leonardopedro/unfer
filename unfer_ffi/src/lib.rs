mod handles;
#[cfg(feature = "zenodo")]
pub mod zenodo;

use std::panic::{AssertUnwindSafe, catch_unwind};

use prob_kernel::{Session, SessionBlob};
use unfer_protocol::{
    ActionRecord, ActionState, AgentInfo, AgentState, AuditEntry, BayesianUpdateRequest,
    BayesianUpdateResult, BeliefPropagationRequest, BeliefPropagationResult, CallerKind, CallerTag,
    Code, Diagnostic, EventPredicate, EventQuery, GrantSet, HamiltonianSpec, KernelEvent, ModelSpec,
    PriorSpec, Severity,
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
    write_bytes(buf, cap, data.as_bytes())
}

fn write_bytes(buf: *mut u8, cap: i64, data: &[u8]) -> i64 {
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

fn read_bytes(ptr: *const u8, len: i64) -> Result<Vec<u8>, Diagnostic> {
    if len < 0 {
        return Err(Diagnostic::new(
            Code::BAD_JSON,
            "negative length",
            Severity::Error,
        ));
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(Diagnostic::new(
            Code::BAD_JSON,
            "null pointer with non-zero length",
            Severity::Error,
        ));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec())
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
/// (QFM.tex §8, P6 H follow-on).
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
        // P7 P5: validate the HMC options. A leapfrog_steps=0 or
        // step_size=0 would silently produce a broken HMC chain. Surface
        // as UK-1001 with a per-field RepairHint.
        let hints = req.hmc_opts.validate();
        if !hints.is_empty() {
            let mut diag = Diagnostic::new(
                Code::BAD_JSON,
                format!("invalid HmcOptsSpec: {} field(s) out of range", hints.len()),
                Severity::Error,
            );
            for hint in hints {
                diag = diag.with_hint(hint);
            }
            return Err(diag);
        }
        let report = handles::with_session_mut(model, |s| {
            s.bayesian_update(&req.observations, &req.hmc_opts)
        })
        .ok_or_else(|| bad_handle(model))?
        .map_err(|e| e.to_diagnostic())?;
        let result = BayesianUpdateResult {
            log_posterior: report.log_posterior,
            mean_likelihood: report.mean_likelihood,
            image: report.image,
            posterior_mean: report.posterior_mean_image,
            n_samples: report.n_samples,
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

/// Run chain belief propagation on the TSR-evolved prior (P8.8).
///
/// Like `uk_bayesian_update`, this is morally a conditioning op, but
/// uses **chain exact BP** (`qfm::bayes::belief_propagation_chain`) —
/// the marginal mode via gradient ascent on the log posterior — instead
/// of HMC sampling. Complexity is $O(\mathrm{max\_iter} \cdot N \cdot
/// m)$ instead of HMC's $O(\mathrm{leapfrog\_steps} \cdot N \cdot m)$,
/// so it is the documented fast path when the user only needs a posterior
/// **point estimate** (not a sample from the typical set).
///
/// **Only QFM tomographic models are eligible.** The request body is
/// `{"observations": [[...], ...], "opts": {"max_iter": ..., "step_size":
/// ..., "tol": ...}}`. All `opts` fields are optional with sensible
/// defaults.
///
/// Returns 0 on success (the result is retrievable via `uk_get_result`).
/// Also enqueues a `conditioned` event for `uk_poll` subscribers. Returns
/// `UK-1001` for malformed JSON or invalid `opts` (with per-field
/// `RepairHint`s), `UK-1004` for an invalid model handle, `UK-5000` for
/// non-QFM models.
#[unsafe(no_mangle)]
pub extern "C" fn uk_belief_propagation(model: i64, req_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_belief_propagation", || {
        let req: BeliefPropagationRequest = parse_json(req_json, len)?;
        // P8.8 (mirroring the P7.5 HMC validation): validate the BP
        // options. A `max_iter=0` or non-positive `step_size` would
        // silently produce a no-op BP. Surface as UK-1001 with
        // per-field RepairHint.
        let hints = req.opts.validate();
        if !hints.is_empty() {
            let mut diag = Diagnostic::new(
                Code::BAD_JSON,
                format!(
                    "invalid BeliefPropagationOptsSpec: {} field(s) out of range",
                    hints.len()
                ),
                Severity::Error,
            );
            for hint in hints {
                diag = diag.with_hint(hint);
            }
            return Err(diag);
        }
        let report = handles::with_session_mut(model, |s| {
            s.belief_propagation(&req.observations, &req.opts)
        })
        .ok_or_else(|| bad_handle(model))?
        .map_err(|e| e.to_diagnostic())?;
        let result = BeliefPropagationResult {
            image: report.image,
            log_posterior: report.log_posterior,
            n_observations: report.n_observations,
            n_sweeps: report.n_sweeps,
            solve_ms: report.solve_ms,
        };
        let result_json = serde_json::to_string(&result)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        handles::set_last_result(model, result_json);
        // Use the 'conditioned' event vocabulary (same as uk_bayesian_update).
        let event = unfer_protocol::KernelEvent::Conditioned {
            prior_probability: 1.0,
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

/// Package a session snapshot into a `.cell` blueprint archive (S5, F4). The archive body
/// carries the `SessionBlob` JSON produced by `uk_snapshot`; module files are added by the
/// host (`ModuleHost::instantiate_from_blueprint`), which owns the module directory.
/// Buffer protocol: returns total bytes needed; copies `min(needed, cap)` into `buf`.
/// Returns <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_blueprint_export(model: i64, buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_blueprint_export", || {
        let blob = handles::with_session_mut(model, |s| s.save()).ok_or_else(|| bad_handle(model))?;
        let session_json = serde_json::to_string(&blob)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;

        let mut builder = unfer_protocol::CellBuilder::new("unfer-session");
        builder.set_archetype("kernel");
        builder.set_session(session_json.as_bytes());

        let cell = builder.build().map_err(|e| {
            Diagnostic::new(Code::BLUEPRINT_INVALID, e.to_string(), Severity::Error)
        })?;
        Ok(write_bytes(buf, cap, &cell))
    })
}

/// Instantiate a session from a `.cell` blueprint archive (S5, F4). The archive must carry a
/// session snapshot (a `SessionBlob` JSON string) — the `initialize_from_blueprint` path.
/// Returns a positive session handle on success, <0 (-code) on error
/// (UK-4100 invalid archive, UK-4101 no session in archive).
#[unsafe(no_mangle)]
pub extern "C" fn uk_blueprint_instantiate(cell: *const u8, len: i64) -> i64 {
    ffi_entry("uk_blueprint_instantiate", || {
        let bytes = read_bytes(cell, len)?;
        let parsed = unfer_protocol::Cell::parse(&bytes).map_err(|e| {
            Diagnostic::new(Code::BLUEPRINT_INVALID, e.to_string(), Severity::Error)
        })?;
        let session = parsed.session().ok_or_else(|| {
            Diagnostic::new(
                Code::BLUEPRINT_NO_SESSION,
                "blueprint archive carries no session snapshot",
                Severity::Error,
            )
        })?;
        let blob: SessionBlob = serde_json::from_slice(session).map_err(|e| {
            Diagnostic::new(Code::BAD_JSON, e.to_string(), Severity::Error)
        })?;
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

/// Analyze an ODE system for essential self-adjointness.
/// `json` is a JSON object: `{"vars":["x"],"rhs":["x^2"],"cov":"reciprocal:0","t_max":100.0}`.
/// Returns a serialized `OdeReport` JSON on success, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_ode_analyze(json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_ode_analyze", || {
        #[derive(serde::Deserialize)]
        struct OdeAnalyzeReq {
            vars: Vec<String>,
            rhs: Vec<String>,
            #[serde(default)]
            cov: Option<String>,
            #[serde(default = "default_t_max")]
            t_max: f64,
        }
        fn default_t_max() -> f64 {
            100.0
        }

        let req: OdeAnalyzeReq = parse_json(json, len)?;
        let samples: Vec<Vec<f64>> = (1..=3).map(|i| vec![i as f64; req.vars.len()]).collect();
        let (report, _) = ode_sirk::protocol::analyze_ode_system(
            req.vars,
            &req.rhs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            req.cov.as_deref(),
            req.t_max,
            &samples,
        )
        .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        let report_json = serde_json::to_string(&report)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        // Write into a thread-local buffer and return the handle.
        // For simplicity, return via a new handle in the session store.
        // Actually, we need to return the JSON string. Use a static buffer approach:
        // store in a global and return a pointer-like integer. For now, we leak
        // the string and return its pointer as an i64 (caller must free via uk_buf_free).
        let boxed = report_json.into_bytes().into_boxed_slice();
        let len = boxed.len() as i64;
        let ptr = Box::into_raw(boxed) as *mut u8;
        // Pack ptr and len into a single i64? No — we return ptr as the handle,
        // and the caller uses uk_buf_free. For ABI simplicity, store in handles.
        let h = handles::store_buffer(ptr, len);
        Ok(h)
    })
}

/// Measure an ODE observable in the original coordinate system.
/// `json` is a JSON object: `{"model":<handle>,"var":"x"}`.
/// Returns the expectation value as a double, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_ode_measure_original(model: i64, var_json: *const u8, len: i64) -> f64 {
    let result = ffi_entry("uk_ode_measure_original", || {
        #[derive(serde::Deserialize)]
        struct MeasureReq {
            var: String,
        }
        let req: MeasureReq = parse_json(var_json, len)?;
        let val = handles::with_session(model, |s| s.measure_ode_observable(&req.var))
            .ok_or_else(|| bad_handle(model))?
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        // Encode f64 as i64 bits for the return value.
        Ok(val.to_bits() as i64)
    });
    f64::from_bits(result as u64)
}

/// Free a buffer returned by `uk_ode_analyze`. Returns 0 on success.
#[unsafe(no_mangle)]
pub extern "C" fn uk_buf_free(handle: i64) -> i64 {
    if handles::free_buffer(handle) {
        0
    } else {
        fail(bad_handle(handle))
    }
}

// ── deferred approval + local simulation (S4) ─────────────────────────────
//
// Side-effecting ops are never executed inline. `uk_action_submit` queues a
// Pending `ActionRecord` and returns a provisional (simulated) result immediately
// (the agent keeps working); an operator/gatekeeper resolves it later with
// `uk_action_apply` / `uk_action_reject` / `uk_action_revert`. Reads merge the
// provisional item back: pending → provisional result, approved → applied result.
// The `effects` grant namespace (host-side `auth::check(principal, "Effect", name)`,
// enforced by the cranelift loopback) is the gate on `uk_action_submit`.

/// Submit a side-effecting action for approval.
/// `req_json` is `{"principal":"<module>","effect":"<name>","params":{...},
/// "provisional":{...optional simulated result...}}`.
/// Creates a Pending `ActionRecord`, returns the action handle (positive), queues an
/// `action_pending` event on every subscription (kernel-global approval lane). The
/// caller reads the provisional result via `uk_action_get`. Returns <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_action_submit(req_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_action_submit", || {
        #[derive(serde::Deserialize)]
        struct ActionSubmitReq {
            #[serde(default)]
            principal: String,
            effect: String,
            #[serde(default)]
            params: serde_json::Value,
            #[serde(default)]
            provisional: Option<serde_json::Value>,
        }
        let req: ActionSubmitReq = parse_json(req_json, len)?;
        let seq = next_action_seq();
        let mut record = ActionRecord::new(
            format!("action-{seq}"),
            req.principal.clone(),
            req.effect.clone(),
            req.params,
            seq,
            Some(req.provisional.unwrap_or_else(|| {
                serde_json::json!({ "simulated": true, "effect": req.effect })
            })),
        );
        // S6 (F6): tag the record with the current caller. When the host loopback
        // dispatched this call it set the thread-local caller to the module's
        // identity, so `principal` (injected by the loopback) and the caller tag
        // agree. A direct (untagged) caller falls back to a gadget tag built from
        // the request's `principal`, if any.
        let ctx = handles::current_caller();
        record.caller = Some(if ctx.is_explicit() {
            ctx.tag.clone()
        } else if req.principal.is_empty() {
            CallerTag::default()
        } else {
            CallerTag::gadget(&req.principal)
        });
        let handle = handles::store_action(record.clone());
        handles::push_action_event(KernelEvent::ActionPending { action: record });
        Ok(handle)
    })
}

/// Approve (and execute) a pending action. Returns 0 on success; UK-4005 if already
/// resolved; UK-4004 if the handle is unknown. Queues an `action_resolved` event.
#[unsafe(no_mangle)]
pub extern "C" fn uk_action_apply(action_handle: i64) -> i64 {
    ffi_entry("uk_action_apply", || {
        let approved = handles::with_action_mut(action_handle, |record| {
            if record.state != ActionState::Pending {
                return false;
            }
            record.state = ActionState::Approved;
            record.applied = Some(serde_json::json!({
                "applied": true,
                "action_id": record.id,
                "effect": record.effect,
            }));
            true
        });
        match approved {
            None => Err(fail_action_not_found(action_handle)),
            Some(true) => {
                push_resolved(action_handle)?;
                Ok(0)
            }
            Some(false) => Err(action_not_pending(action_handle)),
        }
    })
}

/// Reject a pending action. Returns 0 on success; UK-4005 if already resolved;
/// UK-4004 if the handle is unknown. Queues an `action_resolved` event.
#[unsafe(no_mangle)]
pub extern "C" fn uk_action_reject(action_handle: i64) -> i64 {
    ffi_entry("uk_action_reject", || {
        let rejected = handles::with_action_mut(action_handle, |record| {
            if record.state != ActionState::Pending {
                return false;
            }
            record.state = ActionState::Rejected;
            record.applied = None;
            true
        });
        match rejected {
            None => Err(fail_action_not_found(action_handle)),
            Some(true) => {
                push_resolved(action_handle)?;
                Ok(0)
            }
            Some(false) => Err(action_not_pending(action_handle)),
        }
    })
}

/// Revert an approved action (rollback). Returns 0 on success; UK-4005 unless the
/// action is currently `Approved`; UK-4004 if the handle is unknown. Queues an
/// `action_resolved` event.
#[unsafe(no_mangle)]
pub extern "C" fn uk_action_revert(action_handle: i64) -> i64 {
    ffi_entry("uk_action_revert", || {
        let reverted = handles::with_action_mut(action_handle, |record| {
            if record.state != ActionState::Approved {
                return false;
            }
            record.state = ActionState::Reverted;
            record.applied = None;
            true
        });
        match reverted {
            None => Err(fail_action_not_found(action_handle)),
            Some(true) => {
                push_resolved(action_handle)?;
                Ok(0)
            }
            Some(false) => Err(action_not_pending(action_handle)),
        }
    })
}

/// Read an action with the merged (provisional→applied) result.
/// Buffer-out protocol: probe with `buf=NULL,cap=0` for the length, then copy.
/// Returns the byte length on success, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_action_get(action_handle: i64, buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_action_get", || {
        let record = handles::with_action(action_handle, |r| r.clone())
            .ok_or_else(|| fail_action_not_found(action_handle))?;
        // F8 observer re-check: a bounded caller may only read its own records
        // and those of its declared `observers`. An un-observable record is
        // indistinguishable from a missing one (no existence oracle).
        if !handles::current_caller().may_observe(&record.principal) {
            return Err(fail_action_not_found(action_handle));
        }
        let value = action_record_json(&record, Some(action_handle))?;
        let json = serde_json::to_string(&value)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// List all actions in the queue (gatekeeper scan surface), oldest first.
/// `out_json` is a JSON array of `ActionRecord`s (with merged results and the
/// numeric `handle` a gatekeeper needs to call `uk_action_apply`/`reject`/`revert`).
/// F8: a bounded caller only sees records it may observe (its own principal plus
/// its `observers`); the trusted harness sees all. Buffer-out protocol; returns
/// the byte length on success, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_action_list(buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_action_list", || {
        let ctx = handles::current_caller();
        let items = handles::list_actions();
        let mut records = Vec::with_capacity(items.len());
        for (handle, record) in items {
            if !ctx.may_observe(&record.principal) {
                continue;
            }
            records.push(action_record_json(&record, Some(handle))?);
        }
        let json = serde_json::to_string(&records)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

fn fail_action_not_found(handle: i64) -> Diagnostic {
    Diagnostic::new(
        Code::ACTION_NOT_FOUND,
        format!("no action with handle {handle}"),
        Severity::Error,
    )
}

fn action_not_pending(handle: i64) -> Diagnostic {
    Diagnostic::new(
        Code::ACTION_ALREADY_RESOLVED,
        format!("action {handle} is not pending"),
        Severity::Error,
    )
}

fn next_action_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);
    NEXT_SEQ.fetch_add(1, Ordering::SeqCst)
}

fn next_agent_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_AGENT_SEQ: AtomicU64 = AtomicU64::new(1);
    NEXT_AGENT_SEQ.fetch_add(1, Ordering::SeqCst)
}

fn action_record_json(
    record: &ActionRecord,
    handle: Option<i64>,
) -> Result<serde_json::Value, Diagnostic> {
    let mut value = serde_json::to_value(record)
        .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "result".to_string(),
            record.merged_result().unwrap_or(serde_json::Value::Null),
        );
        if let Some(h) = handle {
            obj.insert("handle".to_string(), serde_json::json!(h));
        }
    }
    Ok(value)
}

/// Emit the `action_resolved` event for a resolved action (approval lane).
fn push_resolved(action_handle: i64) -> Result<(), Diagnostic> {
    let record = handles::with_action(action_handle, |r| r.clone())
        .ok_or_else(|| fail_action_not_found(action_handle))?;
    handles::push_action_event(KernelEvent::ActionResolved { action: record });
    Ok(())
}

// ── agent accountability + audit (S6) ────────────────────────────────────
//
// Cloudflare "GatekeeperCaller" adaptation (F6). Three pieces:
//
// 1. **Caller context** (`uk_set_caller`/`uk_clear_caller`, Rust-ABI host-internal):
//    the loopback chokepoint tags the current thread's identity before dispatching
//    any `uk_*` call. These are NOT part of the C ABI surface (a worker cannot call
//    them — the host owns the identity), so they are plain `pub fn`, not `extern "C"`.
// 2. **Audit trail** (`uk_audit_append` host-internal; `uk_audit_list`/`uk_audit_clear`
//    C-ABI): every dispatched call appends an immutable `AuditEntry` tagged with the
//    caller. An operator/gatekeeper lists it; only an operator clears it.
// 3. **AgentSpawner** (`uk_agent_spawn`/`uk_agent_list`/`uk_agent_kill`/
//    `uk_agent_grants`, C-ABI): spawns sub-agents **bounded to a fixed grant set**,
//    minted once at the chokepoint. Escalation is refused (UK-4202); the host loopback
//    enforces the bounded set on every call attributed to the agent (default-deny).

/// Set the current thread's caller context. `caller_json` is
/// `{"from":"agent|gadget|hook","principal":"...","chat_id":?,"grants":?}` where
/// `grants` is `{"kernel":[...],"effects":[...]}` (absent = unrestricted).
/// Host-internal (Rust-ABI): not callable over the loopback. Returns 0 on success.
pub fn uk_set_caller(caller_json: &str) -> Result<(), Diagnostic> {
    #[derive(serde::Deserialize)]
    struct CallerReq {
        #[serde(default)]
        from: Option<CallerKind>,
        #[serde(default)]
        principal: Option<String>,
        #[serde(default)]
        chat_id: Option<String>,
        #[serde(default)]
        grants: Option<GrantSet>,
    }
    let req: CallerReq = serde_json::from_str(caller_json).map_err(|e| {
        Diagnostic::new(Code::AUDIT_INVALID, e.to_string(), Severity::Error)
    })?;
    let tag = CallerTag::new(
        req.from.unwrap_or(CallerKind::Hook),
        req.principal.unwrap_or_else(|| "kernel".to_string()),
        req.chat_id,
    );
    handles::set_caller(tag, req.grants);
    Ok(())
}

/// Reset the current thread's caller context to the default (trusted harness).
/// Host-internal (Rust-ABI). Returns 0.
pub fn uk_clear_caller() {
    handles::clear_caller();
}

/// Append an audit entry for the current thread's caller. `entry_json` is
/// `{"symbol":"...","args":?,"ok":bool,"detail":?}` — the caller tag is read from
/// the thread-local caller context (the loopback set it at dispatch start).
/// Host-internal (Rust-ABI): only the host loopback appends. Returns the assigned
/// sequence number (0 on malformed input).
pub fn uk_audit_append(entry_json: &str) -> i64 {
    #[derive(serde::Deserialize)]
    struct AuditAppendReq {
        symbol: String,
        #[serde(default)]
        args: serde_json::Value,
        #[serde(default)]
        ok: bool,
        #[serde(default)]
        detail: Option<String>,
    }
    let req: AuditAppendReq = match serde_json::from_str(entry_json) {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let ctx = handles::current_caller();
    let entry = AuditEntry {
        seq: 0, // assigned by the store
        caller: ctx.tag,
        symbol: req.symbol,
        ok: req.ok,
        detail: req.detail,
        args: if req.args.is_null() {
            serde_json::json!([])
        } else {
            req.args
        },
    };
    handles::store_audit(entry) as i64
}

/// List the audit trail, newest first. Buffer-out protocol (probe, then copy).
/// F8: a bounded caller only sees entries it may observe (its own principal plus
/// its `observers`); the trusted harness sees the full trail. Returns the byte
/// length on success, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_audit_list(buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_audit_list", || {
        let ctx = handles::current_caller();
        let entries: Vec<AuditEntry> = handles::list_audit()
            .into_iter()
            .filter(|e| ctx.may_observe(&e.caller.principal))
            .collect();
        let json = serde_json::to_string(&entries)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// Clear the audit trail (an operator action). Returns the number of entries
/// removed, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_audit_clear() -> i64 {
    ffi_entry("uk_audit_clear", || Ok(handles::clear_audit() as i64))
}

/// Spawn a sub-agent bounded to a fixed grant set (S6 `AgentSpawner`).
/// `spec_json` is `{"name":"...","grants":{"kernel":[...],"effects":[...]},
/// "parent":?,"chat_id":?}`.
///
/// **Capability non-escalation:** when the caller holds a bounded grant set
/// (thread-local caller context), the requested grants must be a subset of it
/// (UK-4202 `AGENT_GRANT_ESCALATION` otherwise). A caller with no bound (the
/// trusted harness) may mint any set. Returns the positive agent handle on
/// success, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_agent_spawn(spec_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_agent_spawn", || {
        #[derive(serde::Deserialize)]
        struct AgentSpawnReq {
            name: String,
            grants: GrantSet,
            #[serde(default)]
            parent: Option<String>,
            #[serde(default)]
            chat_id: Option<String>,
        }
        let req: AgentSpawnReq = parse_json(spec_json, len)?;
        let ctx = handles::current_caller();
        if let Some(caller_grants) = ctx.grants.as_ref()
            && !req.grants.is_subset_of(caller_grants)
        {
            return Err(Diagnostic::new(
                Code::AGENT_GRANT_ESCALATION,
                format!(
                    "grant escalation refused: requested {:?} is not a subset of caller grants",
                    req.grants
                ),
                Severity::Error,
            ));
        }
        let seq = next_agent_seq();
        let agent = AgentInfo {
            id: format!("agent-{seq}"),
            name: req.name,
            grants: req.grants,
            parent: req.parent,
            state: AgentState::Running,
            created_at: seq,
            chat_id: req.chat_id,
        };
        Ok(handles::store_agent(agent))
    })
}

/// List all spawned sub-agents (registry scan surface), oldest first.
/// Buffer-out protocol; returns the byte length on success, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_agent_list(buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_agent_list", || {
        let agents = handles::list_agents();
        let mut values = Vec::with_capacity(agents.len());
        for (handle, agent) in agents {
            let mut value = serde_json::to_value(&agent).map_err(|e| {
                Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error)
            })?;
            if let Some(obj) = value.as_object_mut() {
                obj.insert("handle".to_string(), serde_json::json!(handle));
            }
            values.push(value);
        }
        let json = serde_json::to_string(&values)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// Stop a sub-agent. Running/Paused → Stopped. Returns 0 on success;
/// UK-4203 if already stopped; UK-4201 if the handle is unknown.
#[unsafe(no_mangle)]
pub extern "C" fn uk_agent_kill(handle: i64) -> i64 {
    ffi_entry("uk_agent_kill", || {
        let killed = handles::with_agent_mut(handle, |agent| {
            if agent.state == AgentState::Stopped {
                return false;
            }
            agent.state = AgentState::Stopped;
            true
        });
        match killed {
            None => Err(Diagnostic::new(
                Code::AGENT_NOT_FOUND,
                format!("no agent with handle {handle}"),
                Severity::Error,
            )),
            Some(true) => Ok(0),
            Some(false) => Err(Diagnostic::new(
                Code::AGENT_STATE_INVALID,
                format!("agent {handle} is already stopped"),
                Severity::Error,
            )),
        }
    })
}

/// Read a sub-agent's **fixed** grant set (the enforcement surface the host
/// loopback checks every attributed call against). Buffer-out protocol; returns
/// the byte length on success, UK-4201 if the handle is unknown.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_agent_grants(handle: i64, buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_agent_grants", || {
        let grants = handles::with_agent(handle, |a| a.grants.clone()).ok_or_else(|| {
            Diagnostic::new(
                Code::AGENT_NOT_FOUND,
                format!("no agent with handle {handle}"),
                Severity::Error,
            )
        })?;
        let json = serde_json::to_string(&grants)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// Read a sub-agent's stable id (`agent-<seq>`). Host-internal (Rust-ABI): the
/// loopback uses it as the agent's caller principal. Returns `None` if the
/// handle is unknown or the agent is stopped.
pub fn uk_agent_id(handle: i64) -> Option<String> {
    handles::with_agent(handle, |a| {
        if a.state == AgentState::Stopped {
            None
        } else {
            Some(a.id.clone())
        }
    })
    .flatten()
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

    // ── S5: .cell blueprint archives (instance isolation + blueprints) ──────

    fn read_raw(f: impl Fn(*mut u8, i64) -> i64) -> Vec<u8> {
        let needed = f(std::ptr::null_mut(), 0);
        assert!(needed >= 0, "unexpected error probing buffer size");
        let mut buf = vec![0u8; needed as usize];
        f(buf.as_mut_ptr(), needed);
        buf
    }

    #[test]
    fn blueprint_export_instantiate_roundtrip() {
        let h = create_harmonic_model();
        assert!(h > 0);

        // Evolve so the state differs from the vacuum prior.
        let (eptr, elen) = json_ptr(r#"{"t":0.05}"#);
        assert_eq!(uk_evolve(h, eptr, elen), 0);

        // Package the session into a .cell and instantiate a fresh handle from it.
        let cell = read_raw(|b, c| uk_blueprint_export(h, b, c));
        assert!(!cell.is_empty());
        let h2 = uk_blueprint_instantiate(cell.as_ptr(), cell.len() as i64);
        assert!(h2 > 0 && h2 != h, "expected fresh handle, got {h2}");

        // The restored session must reproduce the same probability as the original.
        let (pptr, plen) = json_ptr(r#"{"kind":"vacuum"}"#);
        assert_eq!(uk_event_probability(h, pptr, plen), 0);
        let p1 = read_buf(|b, c| uk_get_result(h, b, c));
        assert_eq!(uk_event_probability(h2, pptr, plen), 0);
        let p2 = read_buf(|b, c| uk_get_result(h2, b, c));
        assert_eq!(p1, p2, "restored session must reproduce the original state");

        uk_model_free(h);
        uk_model_free(h2);
    }

    #[test]
    fn blueprint_instantiate_rejects_bad_magic() {
        let garbage = b"NOTACEL1-not-a-cell-archive";
        let ret = uk_blueprint_instantiate(garbage.as_ptr(), garbage.len() as i64);
        assert_eq!(ret, -4100, "bad magic must yield UK-4100, got {ret}");
    }

    #[test]
    fn blueprint_instantiate_rejects_cell_without_session() {
        let mut builder = unfer_protocol::CellBuilder::new("dry");
        builder.add_file("module.toml", b"[module]\nname = \"dry\"\n").unwrap();
        let cell = builder.build().unwrap();
        let ret = uk_blueprint_instantiate(cell.as_ptr(), cell.len() as i64);
        assert_eq!(ret, -4101, "session-less cell must yield UK-4101, got {ret}");
    }

    #[test]
    fn blueprint_export_bad_handle() {
        let ret = uk_blueprint_export(99999, std::ptr::null_mut(), 0);
        assert_eq!(ret, -1004);
    }

    // ── S4: deferred approval + local simulation ─────────────────────────
    //
    // The action store is kernel-global (shared FFI statics), so the action tests
    // serialize on ACTION_TESTS_LOCK: they must not run concurrently or they would
    // interfere through the shared queue (counts and buffer sizes).

    static ACTION_TESTS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn submit_action(effect: &str, params: &str) -> i64 {
        let req = format!(
            r#"{{"principal":"test_module","effect":"{effect}","params":{params}}}"#
        );
        let (ptr, len) = json_ptr(&req);
        uk_action_submit(ptr, len)
    }

    fn action_get(handle: i64) -> serde_json::Value {
        let json = read_buf(|b, c| uk_action_get(handle, b, c));
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn action_submit_queues_pending_record_with_provisional_result() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let handle = submit_action("send_notification", r#"{"to":"alice"}"#);
        assert!(handle > 0, "expected positive action handle, got {handle}");

        let record = action_get(handle);
        assert_eq!(record["effect"], "send_notification");
        assert_eq!(record["state"], "pending");
        assert_eq!(record["principal"], "test_module");
        // Local simulation: the merged result while pending is the provisional one.
        assert_eq!(record["result"]["simulated"], true);
        assert_eq!(record["result"]["effect"], "send_notification");
    }

    #[test]
    fn action_apply_resolves_and_merges_applied_result() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let handle = submit_action("send_notification", r#"{"to":"alice"}"#);
        assert_eq!(uk_action_apply(handle), 0);

        let record = action_get(handle);
        assert_eq!(record["state"], "approved");
        assert_eq!(record["result"]["applied"], true);
        assert_eq!(record["result"]["action_id"], record["id"]);
    }

    #[test]
    fn action_reject_marks_rejected_and_blocks_reapply() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let handle = submit_action("send_notification", r#"{"to":"bob"}"#);
        assert_eq!(uk_action_reject(handle), 0);
        assert_eq!(action_get(handle)["state"], "rejected");
        // Re-resolving a resolved action is UK-4005.
        assert_eq!(uk_action_apply(handle), -4005);
        assert_eq!(uk_action_reject(handle), -4005);
    }

    #[test]
    fn action_revert_requires_approved() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let handle = submit_action("send_notification", r#"{"to":"carol"}"#);
        // Reverting a pending action is UK-4005 (only approved actions can roll back).
        assert_eq!(uk_action_revert(handle), -4005);
        assert_eq!(uk_action_apply(handle), 0);
        assert_eq!(uk_action_revert(handle), 0);
        assert_eq!(action_get(handle)["state"], "reverted");
    }

    #[test]
    fn action_unknown_handle_returns_neg4004() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(uk_action_apply(99999), -4004);
        assert_eq!(uk_action_reject(99999), -4004);
        assert_eq!(uk_action_revert(99999), -4004);
    }

    #[test]
    fn action_list_returns_all_records() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let a = submit_action("op_a", r#"{"n":1}"#);
        let b = submit_action("op_b", r#"{"n":2}"#);
        assert_eq!(uk_action_reject(b), 0);

        let json = read_buf(|buf, cap| uk_action_list(buf, cap));
        let records: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = records.as_array().expect("action list must be an array");
        // The action store is kernel-global and persists across tests; filter to this
        // test's own records (the lock serializes action tests, but earlier tests' records
        // remain in the shared queue).
        let mine: Vec<&serde_json::Value> = arr
            .iter()
            .filter(|r| {
                matches!(r["effect"].as_str(), Some("op_a") | Some("op_b"))
            })
            .collect();
        assert_eq!(mine.len(), 2, "expected op_a + op_b in list: {json}");
        assert_eq!(mine[0]["effect"], "op_a");
        assert_eq!(mine[0]["state"], "pending");
        assert_eq!(mine[0]["result"]["simulated"], true);
        assert_eq!(mine[1]["effect"], "op_b");
        assert_eq!(mine[1]["state"], "rejected");

        let _ = a;
    }

    #[test]
    fn action_submit_bad_json_returns_neg1001() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (ptr, len) = json_ptr("not json");
        assert_eq!(uk_action_submit(ptr, len), -1001);
        // Missing required `effect` field.
        let (ptr, len) = json_ptr(r#"{"principal":"x","params":{}}"#);
        assert_eq!(uk_action_submit(ptr, len), -1001);
    }

    #[test]
    fn action_pending_event_broadcasts_to_subscriptions() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let h = create_harmonic_model();
        let (qptr, qlen) = json_ptr(r#"{"types":["action_pending","action_resolved"]}"#);
        let sub = uk_subscribe(h, qptr, qlen);
        assert!(sub > 0);

        let handle = submit_action("op_x", r#"{"k":"v"}"#);
        let evt_json = read_buf(|b, c| uk_poll(sub, b, c));
        let evt: serde_json::Value = serde_json::from_str(&evt_json).unwrap();
        assert_eq!(evt["type"], "action_pending");
        assert_eq!(evt["action"]["effect"], "op_x");
        assert_eq!(evt["action"]["id"], record_id_of(handle));

        assert_eq!(uk_action_apply(handle), 0);
        let evt_json = read_buf(|b, c| uk_poll(sub, b, c));
        let evt: serde_json::Value = serde_json::from_str(&evt_json).unwrap();
        assert_eq!(evt["type"], "action_resolved");
        assert_eq!(evt["action"]["state"], "approved");

        uk_model_free(h);
    }

    #[test]
    fn action_pending_event_respects_type_filter() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let h = create_harmonic_model();
        // Subscribe to only "evolved" — action events must NOT arrive.
        let (qptr, qlen) = json_ptr(r#"{"types":["evolved"]}"#);
        let sub = uk_subscribe(h, qptr, qlen);

        let _handle = submit_action("op_y", r#"{}"#);
        let mut buf = [0u8; 256];
        assert_eq!(
            uk_poll(sub, buf.as_mut_ptr(), 256),
            0,
            "action_pending must be filtered out by evolved-only query"
        );

        uk_model_free(h);
    }

    fn record_id_of(handle: i64) -> String {
        action_get(handle)["id"].as_str().unwrap().to_string()
    }

    // ── S6: agent accountability + audit ──────────────────────────────────
    //
    // The audit trail and agent registry are kernel-global (shared FFI statics),
    // so these tests serialize on AUDIT_AGENT_TESTS_LOCK (like the action store)
    // and clear the stores first.

    static AUDIT_AGENT_TESTS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn audit_list() -> serde_json::Value {
        let json = read_buf(|b, c| uk_audit_list(b, c));
        serde_json::from_str(&json).unwrap()
    }

    fn set_caller_gadget(principal: &str) {
        let caller = format!(r#"{{"from":"gadget","principal":"{principal}","chat_id":"c-42"}}"#);
        uk_set_caller(&caller).expect("caller json must parse");
    }

    #[test]
    fn audit_append_list_clear_roundtrip() {
        let _lock = AUDIT_AGENT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_audit_clear();
        set_caller_gadget("mod_a");
        let seq1 = uk_audit_append(r#"{"symbol":"uk_evolve","args":[{"t":0.1}],"ok":true}"#);
        let seq2 = uk_audit_append(
            r#"{"symbol":"uk_action_submit","args":[{"effect":"x"}],"ok":true}"#,
        );
        assert!(seq1 > 0 && seq2 > seq1);

        let entries = audit_list();
        let arr = entries.as_array().unwrap();
        // Newest first; clear() removed prior entries so these two are all we see.
        assert_eq!(arr.len(), 2, "expected 2 audit entries, got {entries}");
        assert_eq!(arr[0]["symbol"], "uk_action_submit");
        assert_eq!(arr[0]["caller"]["from"], "gadget");
        assert_eq!(arr[0]["caller"]["principal"], "mod_a");
        assert_eq!(arr[0]["caller"]["chat_id"], "c-42");
        assert_eq!(arr[0]["ok"], true);
        assert_eq!(arr[0]["args"][0]["effect"], "x");
        assert_eq!(arr[1]["symbol"], "uk_evolve");
        assert_eq!(arr[1]["caller"]["principal"], "mod_a");

        assert!(uk_audit_clear() >= 2);
        assert_eq!(audit_list().as_array().unwrap().len(), 0);
        uk_clear_caller();
    }

    #[test]
    fn audit_default_caller_is_hook() {
        let _lock = AUDIT_AGENT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_audit_clear();
        // No explicit caller: the default hook/kernel tag applies.
        let seq = uk_audit_append(r#"{"symbol":"uk_version","ok":true}"#);
        assert!(seq > 0);
        let entries = audit_list();
        let arr = entries.as_array().unwrap();
        assert_eq!(arr[0]["caller"]["from"], "hook");
        assert_eq!(arr[0]["caller"]["principal"], "kernel");
        uk_audit_clear();
    }

    #[test]
    fn audit_append_rejects_malformed_entry() {
        let _lock = AUDIT_AGENT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Missing required `symbol` → seq 0 (no entry appended).
        assert_eq!(uk_audit_append(r#"{"ok":true}"#), 0);
    }

    #[test]
    fn agent_spawn_bounded_and_list() {
        let _lock = AUDIT_AGENT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_caller();
        // A gadget holding {uk_evolve, uk_action_submit} may spawn an agent bounded
        // to a subset, but not one with a superset (escalation is refused).
        let caller = r#"{"from":"gadget","principal":"parent_mod","grants":{"kernel":["uk_evolve","uk_action_submit"],"effects":["send_notification"]}}"#;
        uk_set_caller(caller).unwrap();

        let spec =
            r#"{"name":"analyst","grants":{"kernel":["uk_evolve"],"effects":["send_notification"]}}"#;
        let (ptr, len) = json_ptr(spec);
        let handle = uk_agent_spawn(ptr, len);
        assert!(handle > 0, "subset spawn must succeed, got {handle}");

        // Escalation: requesting a symbol the parent does not hold → UK-4202.
        let bad =
            r#"{"name":"sneaky","grants":{"kernel":["uk_evolve","uk_model_create"]}}"#;
        let (ptr, len) = json_ptr(bad);
        assert_eq!(uk_agent_spawn(ptr, len), -4202);
        // Escalation via the effects namespace too.
        let bad_effect = r#"{"name":"sneaky","grants":{"effects":["delete_all"]}}"#;
        let (ptr, len) = json_ptr(bad_effect);
        assert_eq!(uk_agent_spawn(ptr, len), -4202);

        // The registry records the fixed (bounded) set.
        let agents = read_buf(|b, c| uk_agent_list(b, c));
        let arr: serde_json::Value = serde_json::from_str(&agents).unwrap();
        let mine = arr
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["name"] == "analyst")
            .expect("analyst agent must be listed");
        assert_eq!(mine["state"], "running");
        assert_eq!(mine["grants"]["kernel"][0], "uk_evolve");
        assert_eq!(mine["grants"]["effects"][0], "send_notification");
        uk_clear_caller();
    }

    #[test]
    fn agent_kill_lifecycle() {
        let _lock = AUDIT_AGENT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_caller();
        let spec = r#"{"name":"worker","grants":{"kernel":["uk_version"]}}"#;
        let (ptr, len) = json_ptr(spec);
        let handle = uk_agent_spawn(ptr, len);
        assert!(handle > 0);
        assert_eq!(uk_agent_kill(handle), 0);
        // Killing an already-stopped agent → UK-4203.
        assert_eq!(uk_agent_kill(handle), -4203);
        // Unknown handle → UK-4201.
        assert_eq!(uk_agent_kill(99999), -4201);
    }

    #[test]
    fn agent_grants_returns_bounded_set() {
        let _lock = AUDIT_AGENT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_caller();
        let spec = r#"{"name":"bounded","grants":{"kernel":["uk_evolve"],"effects":["notify"]}}"#;
        let (ptr, len) = json_ptr(spec);
        let handle = uk_agent_spawn(ptr, len);
        assert!(handle > 0);
        let grants_json = read_buf(|b, c| uk_agent_grants(handle, b, c));
        let grants: serde_json::Value = serde_json::from_str(&grants_json).unwrap();
        assert_eq!(grants["kernel"][0], "uk_evolve");
        assert_eq!(grants["effects"][0], "notify");
        // Unknown handle → UK-4201.
        assert_eq!(uk_agent_grants(99999, std::ptr::null_mut(), 0), -4201);
    }

    #[test]
    fn action_record_carries_caller_tag() {
        let _lock = AUDIT_AGENT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_caller();
        set_caller_gadget("mod_tagged");
        // The loopback injects the request principal to match the caller identity.
        let req = r#"{"principal":"mod_tagged","effect":"send_notification","params":{"to":"dave"}}"#;
        let (ptr, len) = json_ptr(req);
        let handle = uk_action_submit(ptr, len);
        assert!(handle > 0);
        let record = action_get(handle);
        assert_eq!(record["principal"], "mod_tagged");
        assert_eq!(record["caller"]["from"], "gadget");
        assert_eq!(record["caller"]["principal"], "mod_tagged");
        assert_eq!(record["caller"]["chat_id"], "c-42");
        uk_clear_caller();
    }

    #[test]
    fn set_caller_rejects_malformed_json() {
        let _lock = AUDIT_AGENT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let err = uk_set_caller("not json");
        assert!(err.is_err(), "malformed caller json must fail");
    }

    // ── F8: observer re-check on shared reads ─────────────────────────────
    //
    // A bounded caller may read only records/audit entries for its own principal
    // and any principal in its `observers` grant. The trusted harness sees all.

    fn set_caller_bounded(principal: &str, observers: &[&str]) {
        let obs: Vec<String> = observers.iter().map(|s| format!("\"{s}\"")).collect();
        let caller = format!(
            r#"{{"from":"gadget","principal":"{principal}","grants":{{"kernel":["uk_action_list","uk_action_get","uk_audit_list"],"effects":[],"observers":[{}]}}}}"#,
            obs.join(",")
        );
        uk_set_caller(&caller).expect("caller json must parse");
    }

    #[test]
    fn action_list_get_filtered_by_observer_grants() {
        // Serializes on ACTION_TESTS_LOCK: uk_action_submit broadcasts a kernel-global
        // action_pending event that concurrent subscription/poll tests would receive.
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_caller();
        set_caller_gadget("f8_owner");
        let req = r#"{"principal":"f8_owner","effect":"send_notification","params":{"to":"eve"}}"#;
        let (ptr, len) = json_ptr(req);
        let handle = uk_action_submit(ptr, len);
        assert!(handle > 0, "submit must succeed");

        // Reader with NO observer grant: f8_owner's record is invisible, and reading
        // its handle is indistinguishable from a missing record (UK-4004).
        set_caller_bounded("f8_reader", &[]);
        let json = read_buf(|b, c| uk_action_list(b, c));
        let records: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = records.as_array().expect("action list must be an array");
        assert!(
            arr.iter().all(|r| r["principal"] != "f8_owner"),
            "reader without observer grant must not see f8_owner: {json}"
        );
        assert_eq!(
            uk_action_get(handle, std::ptr::null_mut(), 0),
            -4004,
            "reading an un-observable record is UK-4004 (no existence oracle)"
        );

        // Reader WITH the observer grant CAN see it.
        set_caller_bounded("f8_peer", &["f8_owner"]);
        let json = read_buf(|b, c| uk_action_list(b, c));
        let records: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = records.as_array().expect("action list must be an array");
        assert!(
            arr.iter().any(|r| r["principal"] == "f8_owner"),
            "peer with observer grant must see f8_owner: {json}"
        );
        let record = action_get(handle);
        assert_eq!(record["principal"], "f8_owner");
        uk_clear_caller();
    }

    #[test]
    fn audit_list_filtered_by_observer_grants() {
        let _lock = AUDIT_AGENT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_audit_clear();
        uk_clear_caller();
        set_caller_gadget("f8_audit_owner");
        uk_audit_append(r#"{"symbol":"uk_evolve","args":[{"t":0.1}],"ok":true}"#);

        // Reader without the observer grant sees no f8_audit_owner entries.
        set_caller_bounded("f8_audit_reader", &[]);
        let json = read_buf(|b, c| uk_audit_list(b, c));
        let entries: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = entries.as_array().expect("audit list must be an array");
        assert!(
            arr.iter().all(|e| e["caller"]["principal"] != "f8_audit_owner"),
            "reader without observer grant must not see f8_audit_owner: {json}"
        );

        // Reader WITH the observer grant sees it.
        set_caller_bounded("f8_audit_peer", &["f8_audit_owner"]);
        let json = read_buf(|b, c| uk_audit_list(b, c));
        let entries: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = entries.as_array().expect("audit list must be an array");
        assert!(
            arr.iter().any(|e| e["caller"]["principal"] == "f8_audit_owner"),
            "peer with observer grant must see f8_audit_owner: {json}"
        );
        uk_clear_caller();
    }
}
