mod handles;
#[cfg(feature = "zenodo")]
pub mod zenodo;

use std::panic::{AssertUnwindSafe, catch_unwind};

use prob_kernel::{Session, SessionBlob};
use unfer_protocol::{
    ActionRecord, ActionState, AgentInfo, AgentState, AuditEntry, BayesianUpdateRequest,
    BayesianUpdateResult, BeliefPropagationRequest, BeliefPropagationResult, CallerKind, CallerTag,
    Code, Diagnostic, EffectKind, EventPredicate, EventQuery, GrantSet, HamiltonianSpec,
    KernelEvent, ModelSpec, PriorSpec, Severity,
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

/// S27: a live credential in the vault cannot be packaged into a snapshot or a
/// `.cell` blueprint — the secret must never serialize out of the host.
fn snapshot_refuses_live_secret() -> Diagnostic {
    Diagnostic::new(
        Code::INTERNAL,
        "refusing to package: a live credential is held in the vault and must not serialize",
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

/// Read a NUL-free UTF-8 string from the C ABI (`ptr` + `len`).
fn read_utf8(ptr: *const u8, len: i64) -> Result<String, Diagnostic> {
    let bytes = read_bytes(ptr, len)?;
    String::from_utf8(bytes).map_err(|e| {
        Diagnostic::new(
            Code::BAD_JSON,
            format!("invalid UTF-8: {e}"),
            Severity::Error,
        )
    })
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
        // S27: a live secret must never serialize into a SessionBlob snapshot.
        if handles::vault_has_live_secrets() {
            return Err(snapshot_refuses_live_secret());
        }
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
        // S27: a live secret must never serialize into a `.cell` blueprint.
        if handles::vault_has_live_secrets() {
            return Err(snapshot_refuses_live_secret());
        }
        let blob =
            handles::with_session_mut(model, |s| s.save()).ok_or_else(|| bad_handle(model))?;
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
        let blob: SessionBlob = serde_json::from_slice(session)
            .map_err(|e| Diagnostic::new(Code::BAD_JSON, e.to_string(), Severity::Error))?;
        let session = Session::restore(blob).map_err(|e| e.to_diagnostic())?;
        Ok(handles::store_session(session))
    })
}

/// Import a verified `.cell` blueprint archive into the blueprint registry (S20, F19).
///
/// The archive is content-addressed (its `blueprint_id` == the content CID, so the record
/// is immutable — an edit would be a *different* blueprint) and registered with the current
/// caller's principal as `created_by`. Re-importing identical bytes is idempotent and
/// preserves the original minter. Buffer protocol: writes the `BlueprintRecord` JSON;
/// returns total bytes needed, or <0 (-code) on error (UK-4100 invalid archive).
#[unsafe(no_mangle)]
pub extern "C" fn uk_blueprint_import(cell: *const u8, len: i64, buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_blueprint_import", || {
        let bytes = read_bytes(cell, len)?;
        let principal = handles::current_caller().tag.principal.clone();
        let record = unfer_data::blueprint::global_registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(&bytes, &principal)
            .map_err(|e| Diagnostic::new(Code::BLUEPRINT_INVALID, e, Severity::Error))?;
        let json = serde_json::to_string(&record)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// List every registered blueprint (S20, F19), address-sorted.
#[unsafe(no_mangle)]
pub extern "C" fn uk_blueprint_list(buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_blueprint_list", || {
        // Always serve the list (blueprints are the operator's content plane); empty is [].
        let records = unfer_data::blueprint::global_registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .list();
        let json = serde_json::to_string(&records)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// Fetch one `BlueprintRecord` by its content CID (S20, F19). Buffer protocol; UK-4102.
#[unsafe(no_mangle)]
pub extern "C" fn uk_blueprint_get_by_id(
    id: *const u8,
    id_len: i64,
    buf: *mut u8,
    cap: i64,
) -> i64 {
    ffi_entry("uk_blueprint_get_by_id", || {
        let id_str = read_utf8(id, id_len)?;
        let registry = unfer_data::blueprint::global_registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(record) = registry.get(&id_str) else {
            return Err(Diagnostic::new(
                Code::BLUEPRINT_NOT_FOUND,
                format!("no blueprint '{id_str}'"),
                Severity::Error,
            ));
        };
        let json = serde_json::to_string(record)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// The raw `.cell` bytes registered under `blueprint_id` (S20, F19). Serves the edge
/// `/cell/<cid>` seed so a registered blueprint can be surfaced through the content
/// gateway. Buffer protocol; the content is address-public.
#[unsafe(no_mangle)]
pub extern "C" fn uk_blueprint_cell(id: *const u8, id_len: i64, buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_blueprint_cell", || {
        let id_str = read_utf8(id, id_len)?;
        let registry = unfer_data::blueprint::global_registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(bytes) = registry.cell_bytes(&id_str).map(|b| b.to_vec()) else {
            return Err(Diagnostic::new(
                Code::BLUEPRINT_NOT_FOUND,
                format!("no blueprint '{id_str}'"),
                Severity::Error,
            ));
        };
        Ok(write_bytes(buf, cap, &bytes))
    })
}

/// Spawn a **fresh per-user copy** of a registered blueprint (S20, F19): every consumer
/// instantiates its own session from the immutable template (never sharing a mutable
/// instance). The archive must carry a session snapshot. Returns the new session handle
/// as a JSON int. UK-4100 invalid, UK-4101 no session, UK-4102 unknown blueprint.
#[unsafe(no_mangle)]
pub extern "C" fn uk_blueprint_export_gadget(
    id: *const u8,
    id_len: i64,
    buf: *mut u8,
    cap: i64,
) -> i64 {
    ffi_entry("uk_blueprint_export_gadget", || {
        let id_str = read_utf8(id, id_len)?;
        let bytes = {
            let registry = unfer_data::blueprint::global_registry()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            match registry.cell_bytes(&id_str) {
                Some(b) => b.to_vec(),
                None => {
                    return Err(Diagnostic::new(
                        Code::BLUEPRINT_NOT_FOUND,
                        format!("no blueprint '{id_str}'"),
                        Severity::Error,
                    ));
                }
            }
        };
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
        let blob: SessionBlob = serde_json::from_slice(session)
            .map_err(|e| Diagnostic::new(Code::BAD_JSON, e.to_string(), Severity::Error))?;
        let session = Session::restore(blob).map_err(|e| e.to_diagnostic())?;
        let handle = handles::store_session(session);
        let json = serde_json::to_string(&serde_json::json!({ "handle": handle }))
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
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
///
/// F20 trust annotations gate the lane: an `observe`-kind effect (readOnlyHint) never
/// queues — the action is applied immediately (state `Approved`). A `mutate`-kind
/// effect queues unless the operator console has *vetted* the caller's principal
/// (`uk_registry_vetted`), in which case it also applies immediately. A `mutate`-kind
/// effect by an un-vetted caller is therefore never applied without a pending approval.
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
        let mut record =
            ActionRecord::new(
                format!("action-{seq}"),
                req.principal.clone(),
                req.effect.clone(),
                req.params,
                seq,
                Some(req.provisional.unwrap_or_else(
                    || serde_json::json!({ "simulated": true, "effect": req.effect }),
                )),
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
        // F20: an observation (or a console-vetted mutation) applies immediately —
        // it never occupies the approval lane. Everything else lands Pending and
        // requires `uk_gate_approve` before it can apply.
        let kind = ctx.effect_kind_of(&req.effect);
        let vetted = handles::is_vetted(&ctx.tag.principal);
        if kind == EffectKind::Observe || vetted {
            handles::with_action_mut(handle, |r| {
                r.state = ActionState::Approved;
                r.applied = Some(serde_json::json!({
                    "applied": true,
                    "action_id": r.id,
                    "effect": r.effect,
                }));
            });
            push_resolved(handle)?;
        } else {
            handles::push_action_event(KernelEvent::ActionPending { action: record });
        }
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

// ── gatekeeper console (S19/F18: human-mediated resolution) ─────────────
//
// Adapted from cloudflare-os gatekeepers: a side-effecting export queued for human approval
// returns "pending approval + simulated outcome" (UK-4002 with the provisional forecast)
// instead of erroring; a human operator then resolves the queue with `uk_gate_approve` /
// `uk_gate_reject`, each resolution written to the audit trail with the human's principal.
// These are the console-facing counterparts of the gadget-facing `uk_action_apply`/reject
// — both mutate the same `ActionRecord`, but the resolve path spells its intent in the audit
// stream.

/// List the gatekeeper's approval-queue (Pending actions only), oldest first. A gatekeeper
/// console scans this and resolves entries with `uk_gate_approve`/`uk_gate_reject`.
/// Buffer-out protocol; returns the byte length on success, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_gate_list_pending(buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_gate_list_pending", || {
        let items: Vec<(i64, serde_json::Value)> = handles::list_actions()
            .into_iter()
            .filter(|(_, r)| r.state == ActionState::Pending)
            .map(|(h, r)| Ok((h, action_record_json(&r, Some(h))?)))
            .collect::<Result<_, Diagnostic>>()?;
        let json = serde_json::to_string(&items)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// Approve a pending action (the human console resolution). The gatekeeper's simulated
/// outcome (provisional forecast) becomes the applied result and the resolution is audited
/// with the resolving principal. Returns 0 on success; UK-4004/UK-4005 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn uk_gate_approve(action_handle: i64) -> i64 {
    ffi_entry("uk_gate_approve", || {
        let approved = handles::with_action_mut(action_handle, |record| {
            if record.state != ActionState::Pending {
                return false;
            }
            record.state = ActionState::Approved;
            record.applied = Some(match record.provisional.clone() {
                Some(sim) => serde_json::json!({
                    "applied": true,
                    "action_id": record.id,
                    "effect": record.effect,
                    "forecast": sim,
                }),
                None => serde_json::json!({
                    "applied": true,
                    "action_id": record.id,
                    "effect": record.effect,
                }),
            });
            true
        });
        match approved {
            None => Err(fail_action_not_found(action_handle)),
            Some(true) => {
                let ctx = handles::current_caller();
                let detail = format!(
                    "gatekeeper approve handle={action_handle} by='{}'",
                    ctx.tag.principal
                );
                let entry = AuditEntry {
                    seq: 0, // store assigns
                    caller: ctx.tag,
                    symbol: "uk_gate_approve".to_string(),
                    ok: true,
                    detail: Some(detail),
                    args: serde_json::json!([action_handle]),
                    component: handles::current_component(),
                    context: handles::current_observability(),
                    sensitive: false,
                };
                handles::store_audit(entry);
                push_resolved(action_handle)?;
                Ok(0)
            }
            Some(false) => Err(action_not_pending(action_handle)),
        }
    })
}

/// Reject a pending action (the human console resolution). The simulated outcome never applies;
/// the rejection is audited with the resolving principal. Returns 0 on success; UK-4004/4005.
#[unsafe(no_mangle)]
pub extern "C" fn uk_gate_reject(action_handle: i64) -> i64 {
    ffi_entry("uk_gate_reject", || {
        let rejected = handles::with_action_mut(action_handle, |record| {
            if record.state != ActionState::Pending {
                return false;
            }
            record.state = ActionState::Rejected;
            true
        });
        match rejected {
            None => Err(fail_action_not_found(action_handle)),
            Some(true) => {
                let ctx = handles::current_caller();
                let action = format!(
                    "gatekeeper reject handle={action_handle} by='{}'",
                    ctx.tag.principal
                );
                let entry = AuditEntry {
                    seq: 0, // store assigns
                    caller: ctx.tag,
                    symbol: "uk_gate_reject".to_string(),
                    ok: true,
                    detail: Some(action),
                    args: serde_json::json!([action_handle]),
                    component: handles::current_component(),
                    context: handles::current_observability(),
                    sensitive: false,
                };
                handles::store_audit(entry);
                push_resolved(action_handle)?;
                Ok(0)
            }
            Some(false) => Err(action_not_pending(action_handle)),
        }
    })
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
    let req: CallerReq = serde_json::from_str(caller_json)
        .map_err(|e| Diagnostic::new(Code::AUDIT_INVALID, e.to_string(), Severity::Error))?;
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

/// Mint (or clear) the operator console's **vetted** marker on `principal`
/// (S21/F20). Vetted principals may auto-apply `mutate`-kind effects without a
/// pending approval. A module can never self-declare vetted status: only the
/// trusted operator harness (`{from:"hook", grants:null}`) may call this;
/// anything else is refused with UK-4501. Never touches the approval queue —
/// clearing the flag for a principal leaves every pending action intact.
#[unsafe(no_mangle)]
pub extern "C" fn uk_registry_vetted(principal: *const u8, len: i64, vetted: i64) -> i64 {
    ffi_entry("uk_registry_vetted", || {
        let caller = handles::current_caller();
        let console_principal = caller.tag.from == CallerKind::Hook && caller.grants.is_none();
        if !console_principal {
            return Err(Diagnostic::new(
                Code::CONSOLE_ONLY,
                "vetted status is minted by the operator console only (UK-4501)",
                Severity::Error,
            ));
        }
        let principal = read_utf8(principal, len)?;
        handles::mark_vetted(&principal, vetted != 0);
        Ok(0)
    })
}

/// C-ABI host-internal marker maintenance used by QA/console reset paths.
pub fn uk_clear_vetted() {
    handles::clear_vetted();
}

/// Read the current windowed meter status for `principal` without consuming a
/// metered call (mirrors Cloudflare's `checkDailyLlmCount`). `budget_json` is
/// `{"budget":N,"rate_limit":M}`; the returned handle encodes a JSON
/// `MeterStatus` blob via the buffer protocol. Read-only — never consumes.
pub extern "C" fn uk_meter_status(
    principal: *const u8,
    len: i64,
    budget_json: *const u8,
    blade: i64,
    buf: *mut u8,
    cap: i64,
) -> i64 {
    ffi_entry("uk_meter_status", || {
        let principal = read_utf8(principal, len)?;
        #[derive(serde::Deserialize)]
        #[serde(default)]
        struct Limits {
            budget: u64,
            rate_limit: u64,
        }
        impl Default for Limits {
            fn default() -> Self {
                Self {
                    budget: 0,
                    rate_limit: 0,
                }
            }
        }
        let limits: Limits = if blade != 0 {
            let json = read_utf8(budget_json, blade)?;
            serde_json::from_str(&json).map_err(|e| {
                Diagnostic::new(Code::BAD_JSON, format!("bad limits: {e}"), Severity::Error)
            })?
        } else {
            Limits::default()
        };
        let status = handles::meter_status(&principal, limits.budget.max(1));
        let json = serde_json::to_string(&status).map_err(|e| {
            Diagnostic::new(Code::INTERNAL, format!("serialize: {e}"), Severity::Error)
        })?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// C-ABI QA/console reset of the windowed meter.
pub fn uk_clear_meter() {
    handles::clear_meter();
}

/// Whether `principal` is sensitive-latched (S26/F25). Host-loopback gate: the
/// chokepoint consults this before dispatching a forward-mutating symbol.
pub fn uk_is_sensitive_latched(principal: &str) -> bool {
    handles::is_sensitive_latched(principal)
}

/// Set or clear the sensitive latch for `principal`. Console-only (the S22
/// operator); returns the new latched state.
pub fn uk_set_sensitive_latch(principal: &str, latched: bool) -> bool {
    handles::set_sensitive_latch(principal, latched)
}

/// Clear every sensitive latch (QA/console reset). Approval queue untouched.
pub fn uk_clear_sensitive_latches() {
    handles::clear_sensitive_latches();
}

/// Store a secret under the current caller (S27/F26). `value_bytes` is the raw
/// secret; the host encrypts it at rest (S15 KeyRing) and returns an opaque
/// handle. The raw value is never returned from the vault — only the handle.
/// Returns the secret handle (>0) on success, <0 (-code) on error.
pub extern "C" fn uk_secret_put(
    owner: *const u8,
    owner_len: i64,
    value: *const u8,
    value_len: i64,
) -> i64 {
    ffi_entry("uk_secret_put", || {
        let owner = read_utf8(owner, owner_len)?;
        let value = read_bytes(value, value_len)?;
        handles::vault_put_secret(&owner, &value)
            .map(|h| h as i64)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e, Severity::Error))
    })
}

/// Dereference a secret handle for the current caller (S27/F26). Only the owner
/// may read it; the host dereferences at call time (grant-checked) so the raw
/// value reaches the dereferencing call, never gadget code. Buffer protocol.
/// <0 (-code) on error (unknown handle / not owner).
pub extern "C" fn uk_secret_get(
    handle: i64,
    owner: *const u8,
    owner_len: i64,
    buf: *mut u8,
    cap: i64,
) -> i64 {
    ffi_entry("uk_secret_get", || {
        let owner = read_utf8(owner, owner_len)?;
        let value = handles::vault_get_secret(handle as u64, &owner)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e, Severity::Error))?;
        Ok(write_bytes(buf, cap, &value))
    })
}

/// Revoke a secret handle (S27/F26), invalidating it. Returns 0 on success
/// (or -code if the handle was already unknown).
pub extern "C" fn uk_secret_revoke(handle: i64) -> i64 {
    ffi_entry("uk_secret_revoke", || {
        if handles::vault_revoke_secret(handle as u64) {
            Ok(0)
        } else {
            Err(Diagnostic::new(
                Code::BAD_HANDLE,
                format!("secret handle {handle} is unknown or already revoked"),
                Severity::Error,
            ))
        }
    })
}

/// Whether any live secret exists in the vault (S27). `uk_snapshot`/`uk_blueprint_export`
/// consult this to refuse packaging a live secret into a snapshot or `.cell` blueprint.
pub fn uk_vault_has_live_secrets() -> bool {
    handles::vault_has_live_secrets()
}

/// Drop every secret (QA/console reset).
pub fn uk_vault_clear() {
    handles::vault_clear();
}

/// Host-loopback gate: consume one unit of the caller's windowed budget for a
/// *metered* symbol. Returns the [`handles::MeterDecision`] tag (`0` = allowed,
/// `1` = rate-limited, `2` = budget-exceeded). This is the single denial point
/// for cost governance — the loopback calls it before dispatching a metered
/// `uk_*`, never after the fact.
pub fn uk_meter_consume(principal: &str, budget: u64, rate_limit: u64) -> i64 {
    match handles::meter_consume(principal, budget, rate_limit) {
        handles::MeterDecision::Allowed => 0,
        handles::MeterDecision::RateLimited => 1,
        handles::MeterDecision::BudgetExceeded => 2,
    }
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
        #[serde(default)]
        sensitive: bool,
    }
    let req: AuditAppendReq = match serde_json::from_str(entry_json) {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let ctx = handles::current_caller();
    // Sanitize the args before they are stored: the audit trail never persists a
    // secret/prompt/key even if a module's args carried one (F22 discipline).
    let mut args = if req.args.is_null() {
        serde_json::json!([])
    } else {
        req.args
    };
    handles::sanitize_sensitive(&mut args);
    let entry = AuditEntry {
        seq: 0, // assigned by the store
        caller: ctx.tag,
        symbol: req.symbol,
        ok: req.ok,
        detail: req.detail,
        args,
        component: handles::current_component().or_else(|| Some("kernel.audit".to_string())),
        context: handles::current_observability(),
        sensitive: req.sensitive,
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

// ── S23 (F22): observability context + dot-separated owner logger ─────
//
// The `unfer_agent` host seeds a per-call observability context (AsyncLocal
// analog) before dispatch and clears it after; the kernel threads its fields
// into every audit entry (`context.trace_id`, `component`). The owner logger
// writes dot-separated `(component)` lines into a ring sink the operator drains.

/// Host-internal (Rust-ABI): seed the current thread's per-call observability
/// context. `value_json` is `{"trace_id": "...", "component": "kernel.audit", ...}`.
/// Used by the kernel loopback at dispatch start; cleared after the call.
pub fn uk_observability_set(value_json: &str) -> Result<(), String> {
    let value: serde_json::Value =
        serde_json::from_str(value_json).map_err(|e| format!("observability json: {e}"))?;
    handles::set_observability(value);
    Ok(())
}

/// Host-internal (Rust-ABI): drop the current thread's per-call context.
pub fn uk_observability_clear() {
    handles::clear_observability();
}

/// C-ABI entry for the observability context (parity with the Rust host helper).
#[unsafe(no_mangle)]
pub extern "C" fn uk_observability(ptr: *const u8, len: i64) -> i64 {
    ffi_entry("uk_observability", || {
        let json = read_utf8(ptr, len)?;
        let value: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| Diagnostic::new(Code::BAD_JSON, e.to_string(), Severity::Error))?;
        handles::set_observability(value);
        Ok(0)
    })
}

/// C-internal observer for `uk_report_issue`: no-op (0) unless an
/// `ERROR_REPORT_BINDING` is provisioned. When bound, the sanitized payload is
/// written as a dot-separated owner line (so a fixture's secret never lands).
#[unsafe(no_mangle)]
pub extern "C" fn uk_report_issue(ptr: *const u8, len: i64) -> i64 {
    ffi_entry("uk_report_issue", || {
        if std::env::var_os("ERROR_REPORT_BINDING").is_none() {
            return Ok(0);
        }
        let json = read_utf8(ptr, len)?;
        let mut value: serde_json::Value = match serde_json::from_str(&json) {
            Ok(v) => v,
            Err(_) => serde_json::Value::String(json),
        };
        handles::sanitize_sensitive(&mut value);
        handles::owner_log("kernel.report_issue", &value.to_string());
        Ok(1)
    })
}

/// Write a dot-separated owner component line into the owner sink.
/// Host-internal (Rust-ABI): only host components call the sink; a gadget cannot
/// log on a foreign owner line.
pub fn owner_log(component: &str, message: &str) {
    handles::owner_log(component, message);
}

/// C-ABI entry of the owner logger (edge → kernel link).
#[unsafe(no_mangle)]
pub extern "C" fn uk_owner_log(ptr: *const u8, len: i64, msg_ptr: *const u8, msg_len: i64) -> i64 {
    ffi_entry("uk_owner_log", || {
        let component = read_utf8(ptr, len)?;
        let message = read_utf8(msg_ptr, msg_len)?;
        handles::owner_log(&component, &message);
        Ok(0)
    })
}

/// List the dot-separated owner log, newest first. Buffer-out protocol.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_owner_list(buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_owner_list", || {
        let lines = handles::list_owner_log();
        let json = serde_json::to_string(&lines)
            .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
        Ok(write_buf(buf, cap, &json))
    })
}

/// Clear the owner sink (operator action). Returns lines removed.
#[unsafe(no_mangle)]
pub extern "C" fn uk_owner_clear() -> i64 {
    ffi_entry("uk_owner_clear", || Ok(handles::clear_owner_log() as i64))
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
            let mut value = serde_json::to_value(&agent)
                .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
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

// ── resource introductions (S18/F17) ───────────────────────────────────
//
// Adapted from cloudflare-os "nothing is ambient": a resource id is introduced at a single
// kernel chokepoint (`uk_resource_introduce`), revoked with `uk_resource_forfeit`, and can
// only be *used* by a session whose caller `GrantSet.resources` includes the id
// (`uk_resource_use` → UK-4401 `RESOURCE_UNINTRODUCED` otherwise). Agents may request an
// introduction (`uk_request_resource`) which lands an approval-pending audit entry and a
// queued `PendingResourceRequest` for the human console (resolved by the F18 `uk_gate_*`).

/// Mint a resource at the kernel chokepoint. `resource_json` is a JSON string id
/// (e.g. `"github.repo#denoission"`). Returns 0 on success; <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_resource_introduce(resource_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_resource_introduce", || {
        let resource_id: String = parse_json(resource_json, len)?;
        let ctx = handles::current_caller();
        handles::resource_introduce(&resource_id, &ctx.tag.principal).map_err(|code| {
            Diagnostic::new(
                code,
                format!("resource '{resource_id}' is already introduced"),
                Severity::Error,
            )
        })?;
        Ok(0)
    })
}

/// Revoke a minted resource. Returns 0 on success; UK-4403 if unknown.
#[unsafe(no_mangle)]
pub extern "C" fn uk_resource_forfeit(resource_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_resource_forfeit", || {
        let resource_id: String = parse_json(resource_json, len)?;
        handles::resource_forfeit(&resource_id).map_err(|code| {
            Diagnostic::new(
                code,
                format!("resource '{resource_id}' was never introduced"),
                Severity::Error,
            )
        })?;
        Ok(0)
    })
}

/// Exercise a resource (the F17 gate surface). Returns 0 when the current caller holds
/// the introduction; UK-4401 `RESOURCE_UNINTRODUCED` for a bounded caller without it in
/// `grants.resources`; UK-4003 for a never-minted id observed by the trusted harness.
#[unsafe(no_mangle)]
pub extern "C" fn uk_resource_use(resource_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_resource_use", || {
        let resource_id: String = parse_json(resource_json, len)?;
        let ctx = handles::current_caller();
        handles::resource_authorized(&resource_id, &ctx).map_err(|code| {
            Diagnostic::new(
                code,
                format!("resource '{resource_id}' is not available to this caller"),
                Severity::Error,
            )
        })?;
        Ok(0)
    })
}

/// Request an introduction for a resource (nothing ambient). Lands an
/// `approval_pending` audit entry and queues a `PendingResourceRequest`. Returns the
/// positive request handle; <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_request_resource(resource_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_request_resource", || {
        let resource_id: String = parse_json(resource_json, len)?;
        let ctx = handles::current_caller();
        let handle = handles::queue_resource_request(&resource_id, ctx.tag.clone());
        let entry = AuditEntry {
            seq: 0, // assigned by the store
            caller: ctx.tag,
            symbol: "uk_request_resource".to_string(),
            ok: true,
            detail: Some(format!(
                "approval_pending request={handle} resource='{resource_id}'"
            )),
            args: serde_json::json!([resource_id]),
            component: handles::current_component(),
            context: handles::current_observability(),
            sensitive: false,
        };
        handles::store_audit(entry);
        Ok(handle)
    })
}

/// List approval-pending resource requests, oldest first (the human review order).
/// Buffer-out protocol; returns the byte length on success, <0 (-code) on error.
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn uk_resource_pending(buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_resource_pending", || {
        let pending = handles::list_pending_resource_requests();
        let mut values = Vec::with_capacity(pending.len());
        for (handle, req) in pending {
            let mut value = serde_json::to_value(&req)
                .map_err(|e| Diagnostic::new(Code::INTERNAL, e.to_string(), Severity::Error))?;
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

// ── certificate ledger (Plan R: carbon-certificate / UTXO state machine) ──
//
// `uk_cert_*` exposes the process-global `CertificateLedger` (the same
// state-transition engine a QuePaxa node applies a `CertificateOp` with) over
// the C ABI. Op JSON schemas:
//
//   uk_cert_mint     {"actor":DID,"amount":u64,"owner":DID,"blinding":"hex32","source?":"str"}
//   uk_cert_mint_request
//                    {"owner":DID,"amount":u64,"source":"unfccc:vc:<orderId>","blinding?":"hex32"}
//                    — oracle-backed mint; source validated (UK-7007)
//   uk_cert_transfer {"actor":DID,
//                     "inputs":[{"coin_id":"hex32","amount":u64,"owner":DID}],
//                     "outputs":[{"amount":u64,"owner":DID}]}
//   uk_cert_burn     {"actor":DID,"inputs":[{"coin_id":"hex32","amount":u64,"owner":DID}]}
//
// `coin_id`/`blinding` are 32-byte hex. Buffer-protocol reads: `uk_cert_root`
// returns raw 32 bytes; `uk_cert_status` returns a JSON blob.

/// Configure the certificate mint authority. `did_utf8` empty (len 0) disables
/// minting (the safe default). Returns 0 on success.
#[unsafe(no_mangle)]
pub extern "C" fn uk_cert_set_authority(did_utf8: *const u8, len: i64) -> i64 {
    ffi_entry("uk_cert_set_authority", || {
        let did = if len > 0 {
            Some(read_utf8(did_utf8, len)?)
        } else {
            None
        };
        handles::cert_set_authority(did);
        Ok(0)
    })
}

/// Read the current committed sparse-Merkle root (32 raw bytes via the buffer
/// protocol). Returns needed size on success, <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_cert_root(buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_cert_root", || {
        let root = handles::cert_root();
        Ok(write_bytes(buf, cap, &root))
    })
}

/// Read a JSON snapshot of the ledger state: `{root, unspent_count, total_supply}`.
#[unsafe(no_mangle)]
pub extern "C" fn uk_cert_status(buf: *mut u8, cap: i64) -> i64 {
    ffi_entry("uk_cert_status", || {
        let json = serde_json::to_string(&handles::cert_status()).map_err(|e| {
            Diagnostic::new(Code::INTERNAL, format!("serialize: {e}"), Severity::Error)
        })?;
        Ok(write_buf(buf, cap, &json))
    })
}

fn parse_hex32(s: &str, field: &str) -> Result<[u8; 32], Diagnostic> {
    let bytes = hex::decode(s).map_err(|e| {
        Diagnostic::new(
            Code::BAD_JSON,
            format!("{field}: invalid hex: {e}"),
            Severity::Error,
        )
    })?;
    bytes.try_into().map_err(|_| {
        Diagnostic::new(
            Code::BAD_JSON,
            format!("{field}: expected 32 bytes"),
            Severity::Error,
        )
    })
}

fn parse_coinrefs(v: &serde_json::Value, field: &str) -> Result<Vec<unfer_protocol::CoinRef>, Diagnostic> {
    let arr = v.as_array().ok_or_else(|| {
        Diagnostic::new(Code::BAD_JSON, format!("{field}: expected an array"), Severity::Error)
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        #[derive(serde::Deserialize)]
        struct CRefJson {
            #[serde(default)]
            coin_id: String,
            amount: u64,
            owner: String,
        }
        let c: CRefJson = serde_json::from_value(item.clone()).map_err(|e| {
            Diagnostic::new(
                Code::BAD_JSON,
                format!("{field}: bad coin ref: {e}"),
                Severity::Error,
            )
        })?;
        let coin_id = if c.coin_id.is_empty() {
            unfer_protocol::CertId([0u8; 32])
        } else {
            unfer_protocol::CertId(parse_hex32(&c.coin_id, "coin_id")?)
        };
        out.push(unfer_protocol::CoinRef {
            coin_id,
            amount: c.amount,
            owner: c.owner,
        });
    }
    Ok(out)
}

/// Mint `amount` carbon certificates to `owner` as `actor` (must be the
/// configured mint authority). `blinding` is hex32. Returns 0 on success,
/// <0 (-code) on error. The new coin_id is `commit_coin(amount, owner,
/// blinding)` — the caller can derive it to chain a transfer.
#[unsafe(no_mangle)]
pub extern "C" fn uk_cert_mint(op_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_cert_mint", || {
        #[derive(serde::Deserialize)]
        struct MintJson {
            actor: String,
            amount: u64,
            owner: String,
            #[serde(default)]
            blinding: String,
            #[serde(default)]
            source: Option<String>,
        }
        let m: MintJson = parse_json(op_json, len)?;
        let blinding = parse_hex32(&m.blinding, "blinding")?;
        let kind = unfer_protocol::CertificateOpKind::Mint {
            amount: m.amount,
            owner: m.owner,
            blinding,
            source: m.source,
        };
        handles::cert_apply(&m.actor, &kind)?;
        Ok(0)
    })
}

/// Submit an oracle-backed mint (Plan R Phase 3). Takes a `MintRequest` JSON:
///
///   {"owner":DID,"amount":u64,"source":"unfccc:vc:<orderId>","blinding?":"hex32"}
///
/// The `source` MUST reference a public UN cancellation record
/// (`unfccc:vc:<orderId>`) — UK-7007 `CERT_ORACLE_REJECTED` otherwise. The
/// `actor` field is not part of the request; the caller's current principal
/// (the configured mint authority) signs the mint. Returns 0 on success,
/// <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_cert_mint_request(req_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_cert_mint_request", || {
        let req: unfer_protocol::MintRequest = parse_json(req_json, len)?;
        req.validate_source().map_err(|code| {
            Diagnostic::new(
                code,
                format!(
                    "source '{}' must reference a UN oracle record (unfccc:vc:<orderId>)",
                    req.source
                ),
                Severity::Error,
            )
        })?;
        let kind = req.to_mint_kind();
        // The request must be submitted by the configured mint authority; the
        // actor is the current caller's principal (set by the loopback).
        let actor = handles::current_caller().tag.principal;
        handles::cert_apply(&actor, &kind)?;
        Ok(0)
    })
}

/// Transfer certificates as `actor` (spender). Returns 0 on success,
/// <0 (-code) on error. New output coin_ids are `commit_coin(amount, owner,
/// [0;32])`; the caller can derive them to spend later.
#[unsafe(no_mangle)]
pub extern "C" fn uk_cert_transfer(op_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_cert_transfer", || {
        #[derive(serde::Deserialize)]
        struct TransferJson {
            actor: String,
            inputs: serde_json::Value,
            outputs: serde_json::Value,
        }
        let t: TransferJson = parse_json(op_json, len)?;
        let inputs = parse_coinrefs(&t.inputs, "inputs")?;
        let outputs = parse_coinrefs(&t.outputs, "outputs")?;
        let kind = unfer_protocol::CertificateOpKind::Transfer { inputs, outputs };
        let _ = handles::cert_apply(&t.actor, &kind)?;
        Ok(0)
    })
}

/// Burn (retire) certificates as `actor` (owner). Returns 0 on success,
/// <0 (-code) on error.
#[unsafe(no_mangle)]
pub extern "C" fn uk_cert_burn(op_json: *const u8, len: i64) -> i64 {
    ffi_entry("uk_cert_burn", || {
        #[derive(serde::Deserialize)]
        struct BurnJson {
            actor: String,
            inputs: serde_json::Value,
        }
        let b: BurnJson = parse_json(op_json, len)?;
        let inputs = parse_coinrefs(&b.inputs, "inputs")?;
        let kind = unfer_protocol::CertificateOpKind::Burn { inputs };
        let _ = handles::cert_apply(&b.actor, &kind)?;
        Ok(0)
    })
}

/// Reset the certificate ledger (QA/console reset). Minting returns to None.
pub fn uk_cert_clear() {
    handles::cert_clear();
}

// ── tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn json_ptr(s: &str) -> (*const u8, i64) {
        (s.as_ptr(), s.len() as i64)
    }

    fn read_buf(f: impl Fn(*mut u8, i64) -> i64) -> String {
        // Two-call buffer protocol, retry-safe: the kernel-global stores can grow
        // between the size probe and the copy, so re-probe when the copy reports a
        // larger needed size (otherwise a concurrent append would truncate the JSON).
        let mut cap = f(std::ptr::null_mut(), 0);
        assert!(cap >= 0, "unexpected error probing buffer size");
        loop {
            let mut buf = vec![0u8; cap as usize];
            let n = f(buf.as_mut_ptr(), cap);
            assert!(n >= 0, "unexpected error receiving buffer");
            if n <= cap {
                buf.truncate(n as usize);
                return String::from_utf8(buf).unwrap();
            }
            cap = n;
        }
    }

    fn read_raw(f: impl Fn(*mut u8, i64) -> i64) -> Vec<u8> {
        let mut cap = f(std::ptr::null_mut(), 0);
        assert!(cap >= 0, "unexpected error probing buffer size");
        loop {
            let mut buf = vec![0u8; cap as usize];
            let n = f(buf.as_mut_ptr(), cap);
            assert!(n >= 0, "unexpected error receiving buffer");
            if n <= cap {
                buf.truncate(n as usize);
                return buf;
            }
            cap = n;
        }
    }

// ── certificate ledger FFI (Plan R) ────────────────────────────────
    // The ledger is a process-global store shared with `handles::cert_ledger_tests`,
    // so these serialize on the single crate-wide `CERT_TESTS_LOCK` and reset it.

    fn cert_root_hex() -> String {
        let raw = read_raw(|b, c| uk_cert_root(b, c));
        assert_eq!(raw.len(), 32);
        hex::encode(&raw)
    }

    fn cert_status_json() -> serde_json::Value {
        serde_json::from_str(&read_buf(|b, c| uk_cert_status(b, c))).unwrap()
    }

    #[test]
    fn cert_ffi_mint_transfer_burn_roundtrip() {
        let _lock = handles::CERT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_cert_clear();
        assert_ne!(cert_root_hex(), "00".repeat(32));
        let (auth_ptr, auth_len) = json_ptr("did:unfer:authority");
        assert_eq!(uk_cert_set_authority(auth_ptr, auth_len), 0);

        // Mint 1000 to alice (single mutating call — never probe/receive).
        let mint = r#"{"actor":"did:unfer:authority","amount":1000,"owner":"did:unfer:alice","blinding":"0101010101010101010101010101010101010101010101010101010101010101","source":"unfccc:cert:TEST"}"#;
        let (p, l) = json_ptr(mint);
        assert_eq!(uk_cert_mint(p, l), 0);
        assert_eq!(cert_status_json()["total_supply"], 1000);
        let alice_coin =
            unfer_consensus::certs::commit_coin(1000, "did:unfer:alice", &[1u8; 32]);

        // Transfer the whole thing to bob.
        let input = format!(
            r#"{{"coin_id":"{}","amount":1000,"owner":"did:unfer:alice"}}"#,
            hex::encode(alice_coin.0)
        );
        let transfer = format!(
            r#"{{"actor":"did:unfer:alice","inputs":[{input}],"outputs":[{{"amount":1000,"owner":"did:unfer:bob"}}]}}"#
        );
        let (p, l) = json_ptr(&transfer);
        assert_eq!(uk_cert_transfer(p, l), 0);
        assert_eq!(cert_status_json()["total_supply"], 1000);
        let bob_coin = unfer_consensus::certs::commit_coin(1000, "did:unfer:bob", &[0u8; 32]);

        // Burn bob's certificate.
        let burn = format!(
            r#"{{"actor":"did:unfer:bob","inputs":[{{"coin_id":"{}","amount":1000,"owner":"did:unfer:bob"}}]}}"#,
            hex::encode(bob_coin.0)
        );
        let (p, l) = json_ptr(&burn);
        assert_eq!(uk_cert_burn(p, l), 0);
        assert_eq!(cert_status_json()["total_supply"], 0);
        uk_cert_clear();
    }

    #[test]
    fn cert_ffi_mint_refuses_non_authority() {
        let _lock = handles::CERT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_cert_clear();
        let (auth_ptr, auth_len) = json_ptr("did:unfer:authority");
        uk_cert_set_authority(auth_ptr, auth_len);
        let mint = r#"{"actor":"did:unfer:nobody","amount":100,"owner":"did:unfer:alice","blinding":"0202020202020202020202020202020202020202020202020202020202020202"}"#;
        let (p, l) = json_ptr(mint);
        let rc = uk_cert_mint(p, l);
        assert_eq!(rc, -7001); // -UK-7001 CertMintNotAuthorized
        uk_cert_clear();
    }

    #[test]
    fn cert_ffi_mint_request_oracle_anchor() {
        let _lock = handles::CERT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_cert_clear();
        let (auth_ptr, auth_len) = json_ptr("did:unfer:authority");
        uk_cert_set_authority(auth_ptr, auth_len);
        let req = r#"{"owner":"did:unfer:alice","amount":15,"source":"unfccc:vc:34791"}"#;

        // The actor of a mint request is the current caller's principal, so
        // tag the thread as the mint authority (matching the consensus flow).
        uk_set_caller(r#"{"from":"hook","principal":"did:unfer:authority"}"#).unwrap();
        let (p, l) = json_ptr(req);
        assert_eq!(uk_cert_mint_request(p, l), 0);
        assert_eq!(cert_status_json()["total_supply"], 15);
        assert_eq!(cert_status_json()["unspent_count"], 1);

        // A source that does not reference a UN oracle record → -UK-7007.
        let bad = r#"{"owner":"did:unfer:alice","amount":15,"source":"unfccc:cert:999"}"#;
        let (p, l) = json_ptr(bad);
        assert_eq!(uk_cert_mint_request(p, l), -7007); // CertOracleRejected
        assert_eq!(cert_status_json()["total_supply"], 15, "rejected mint is a no-op");

        // Without the caller being the authority, a valid oracle source is still
        // refused (UK-7001) — the caller context, not the request, decides.
        uk_set_caller(r#"{"from":"hook","principal":"did:unfer:nobody"}"#).unwrap();
        let (p, l) = json_ptr(req);
        assert_eq!(uk_cert_mint_request(p, l), -7001);
        uk_clear_caller();
        uk_cert_clear();
    }

    #[test]
    fn cert_ffi_double_spend_rejected() {
        let _lock = handles::CERT_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_cert_clear();
        let (a, al) = json_ptr("did:unfer:authority");
        uk_cert_set_authority(a, al);
        let mint = r#"{"actor":"did:unfer:authority","amount":500,"owner":"did:unfer:alice","blinding":"0303030303030303030303030303030303030303030303030303030303030303"}"#;
        let (p, l) = json_ptr(mint);
        assert_eq!(uk_cert_mint(p, l), 0);
        let alice_coin = unfer_consensus::certs::commit_coin(500, "did:unfer:alice", &[3u8; 32]);
        let input = format!(r#"{{"coin_id":"{}","amount":500,"owner":"did:unfer:alice"}}"#, hex::encode(alice_coin.0));
        let out = r#"{"amount":500,"owner":"did:unfer:alice"}"#;
        let once = format!(r#"{{"actor":"did:unfer:alice","inputs":[{input}],"outputs":[{out}]}}"#);
        let (p, l) = json_ptr(&once);
        assert_eq!(uk_cert_transfer(p, l), 0);
        // Re-spending the now-spent input → -UK-7004.
        let (p, l) = json_ptr(&once);
        let rc = uk_cert_transfer(p, l);
        assert_eq!(rc, -7004); // -UK-7004 CertDoubleSpend
        uk_cert_clear();
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
        builder
            .add_file("module.toml", b"[module]\nname = \"dry\"\n")
            .unwrap();
        let cell = builder.build().unwrap();
        let ret = uk_blueprint_instantiate(cell.as_ptr(), cell.len() as i64);
        assert_eq!(
            ret, -4101,
            "session-less cell must yield UK-4101, got {ret}"
        );
    }

    #[test]
    fn blueprint_export_bad_handle() {
        let ret = uk_blueprint_export(99999, std::ptr::null_mut(), 0);
        assert_eq!(ret, -1004);
    }

    // ── S20 (F19): blueprint templates + per-user instantiation ──────────
    //
    // The blueprint registry is process-global; serialize like the action tests.

    fn blueprint_cell_from_session(h: i64) -> Vec<u8> {
        read_raw(|b, c| uk_blueprint_export(h, b, c))
    }

    fn blueprint_import(cell: &[u8]) -> i64 {
        let mut buf = [0u8; 4096];
        let n = uk_blueprint_import(cell.as_ptr(), cell.len() as i64, buf.as_mut_ptr(), 4096);
        assert!(n >= 0, "import must succeed, got {n}");
        n
    }

    #[test]
    fn blueprint_import_register_list_export_gadget_two_sessions() {
        let _lock = BLUEPRINT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unfer_data::blueprint::clear_global_registry();
        uk_clear_caller();

        let h = create_harmonic_model();
        let (vptr, vlen) = json_ptr(r#"{"t":0.05}"#);
        assert_eq!(uk_evolve(h, vptr, vlen), 0);
        let cell = blueprint_cell_from_session(h);
        assert!(!cell.is_empty());

        set_caller_gadget("blueprint_publisher");
        let needed = blueprint_import(&cell);
        assert!(needed > 0);

        // The registry carries the immutable record addressed by the content CID.
        let list_json = read_buf(|b, c| uk_blueprint_list(b, c));
        let list: serde_json::Value = serde_json::from_str(&list_json).unwrap();
        assert_eq!(
            list.as_array().map(|a| a.len()).unwrap_or(0),
            1,
            "one import: {list}"
        );
        let record = &list[0];
        let cid = record["blueprint_id"].as_str().unwrap().to_string();
        assert_eq!(record["created_by"], "blueprint_publisher");
        assert_eq!(record["immutable_blueprint_id"], cid);
        assert_eq!(cid.len(), 64);

        // Every consumer runs its own copy: two export_gadget calls give two distinct
        // handles whose restored sessions behave identically to the source.
        let g1 = export_gadget(&cid);
        let g2 = export_gadget(&cid);
        assert!(
            g1 > 0 && g2 > 0 && g1 != g2,
            "per-user copies must be distinct sessions"
        );

        // The per-user copies must reproduce identical Born-rule behavior.
        let (pptr, plen) = json_ptr(r#"{"kind":"vacuum"}"#);
        assert_eq!(
            uk_event_probability(g1, pptr, plen),
            0,
            "gadget 1 must answer"
        );
        let p1 = read_buf(|b, c| uk_get_result(g1, b, c));
        assert_eq!(
            uk_event_probability(g2, pptr, plen),
            0,
            "gadget 2 must answer"
        );
        let p2 = read_buf(|b, c| uk_get_result(g2, b, c));
        assert_eq!(
            p1, p2,
            "two per-user copies must reproduce identical behavior"
        );

        // Idempotent import: same bytes → same immutable id, original minter kept.
        set_caller_gadget("second_publisher");
        let needed2 = blueprint_import(&cell);
        assert!(needed2 > 0);
        let list_json = read_buf(|b, c| uk_blueprint_list(b, c));
        let list: serde_json::Value = serde_json::from_str(&list_json).unwrap();
        assert_eq!(list.as_array().map(|a| a.len()).unwrap_or(0), 1);
        assert_eq!(
            list[0]["created_by"], "blueprint_publisher",
            "immutable: no re-mint"
        );

        uk_model_free(h);
        uk_model_free(g1);
        uk_model_free(g2);
    }

    fn export_gadget(cid: &str) -> i64 {
        let out = read_raw(|b, c| uk_blueprint_export_gadget(cid.as_ptr(), cid.len() as i64, b, c));
        let v: serde_json::Value = serde_json::from_slice(&out).expect("gadget json");
        v["handle"]
            .as_i64()
            .expect("gadget returns a session handle")
    }

    #[test]
    fn blueprint_import_rejects_tampered_cell() {
        let _lock = BLUEPRINT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut builder = unfer_protocol::CellBuilder::new("dry");
        builder
            .add_file("module.toml", b"[module]\nname=\"dry\"\n")
            .unwrap();
        let mut cell = builder.build().unwrap();
        cell[10] ^= 0xff; // flip a body byte — content address no longer matches
        let ret = uk_blueprint_import(cell.as_ptr(), cell.len() as i64, std::ptr::null_mut(), 0);
        assert_eq!(
            ret, -4100,
            "tampered archive must fail verification (UK-4100), got {ret}"
        );

        // Bad magic entirely.
        let ret = uk_blueprint_import(b"NOTACEL1".as_ptr(), 8, std::ptr::null_mut(), 0);
        assert_eq!(ret, -4100, "bad magic must yield UK-4100, got {ret}");
    }

    #[test]
    fn blueprint_get_and_cell_and_unknown_id() {
        let _lock = BLUEPRINT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unfer_data::blueprint::clear_global_registry();
        uk_clear_caller();
        let h = create_harmonic_model();
        let (pptr, plen) = json_ptr(r#"{"t":0.02}"#);
        assert_eq!(uk_evolve(h, pptr, plen), 0);
        let cell = blueprint_cell_from_session(h);
        blueprint_import(&cell);
        let cid: String = {
            let list = read_buf(|b, c| uk_blueprint_list(b, c));
            serde_json::from_str::<serde_json::Value>(&list).unwrap()[0]["blueprint_id"]
                .as_str()
                .unwrap()
                .to_string()
        };

        // uk_blueprint_get_by_id round-trips the record; uk_blueprint_cell returns the raw
        // archive bytes the edge `/cell/<cid>` route seeds from.
        let rec = read_buf(|b, c| uk_blueprint_get_by_id(cid.as_ptr(), cid.len() as i64, b, c));
        let rec: serde_json::Value = serde_json::from_str(&rec).unwrap();
        assert_eq!(rec["blueprint_id"], cid);

        let bytes = read_raw(|b, c| uk_blueprint_cell(cid.as_ptr(), cid.len() as i64, b, c));
        assert_eq!(
            bytes, cell,
            "uk_blueprint_cell must return the exact registered bytes"
        );

        // Unknown ids read UK-4102 on both.
        let unknown = "0".repeat(64);
        let r1 = uk_blueprint_get_by_id(
            unknown.as_ptr(),
            unknown.len() as i64,
            std::ptr::null_mut(),
            0,
        );
        assert_eq!(r1, -4102);
        let r2 = uk_blueprint_cell(
            unknown.as_ptr(),
            unknown.len() as i64,
            std::ptr::null_mut(),
            0,
        );
        assert_eq!(r2, -4102);
        let r3 = uk_blueprint_export_gadget(
            unknown.as_ptr(),
            unknown.len() as i64,
            std::ptr::null_mut(),
            0,
        );
        assert_eq!(r3, -4102);

        uk_model_free(h);
    }

    static BLUEPRINT_TESTS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // ── S4: deferred approval + local simulation ─────────────────────────
    //
    // The action store is kernel-global (shared FFI statics), so the action tests
    // serialize on ACTION_TESTS_LOCK: they must not run concurrently or they would
    // interfere through the shared queue (counts and buffer sizes).

    static ACTION_TESTS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn submit_action(effect: &str, params: &str) -> i64 {
        let req = format!(r#"{{"principal":"test_module","effect":"{effect}","params":{params}}}"#);
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
            .filter(|r| matches!(r["effect"].as_str(), Some("op_a") | Some("op_b")))
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
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        uk_audit_clear();
        set_caller_gadget("mod_a");
        let seq1 = uk_audit_append(r#"{"symbol":"uk_evolve","args":[{"t":0.1}],"ok":true}"#);
        let seq2 =
            uk_audit_append(r#"{"symbol":"uk_action_submit","args":[{"effect":"x"}],"ok":true}"#);
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
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Missing required `symbol` → seq 0 (no entry appended).
        assert_eq!(uk_audit_append(r#"{"ok":true}"#), 0);
    }

    #[test]
    fn agent_spawn_bounded_and_list() {
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        uk_clear_caller();
        // A gadget holding {uk_evolve, uk_action_submit} may spawn an agent bounded
        // to a subset, but not one with a superset (escalation is refused).
        let caller = r#"{"from":"gadget","principal":"parent_mod","grants":{"kernel":["uk_evolve","uk_action_submit"],"effects":["send_notification"]}}"#;
        uk_set_caller(caller).unwrap();

        let spec = r#"{"name":"analyst","grants":{"kernel":["uk_evolve"],"effects":["send_notification"]}}"#;
        let (ptr, len) = json_ptr(spec);
        let handle = uk_agent_spawn(ptr, len);
        assert!(handle > 0, "subset spawn must succeed, got {handle}");

        // Escalation: requesting a symbol the parent does not hold → UK-4202.
        let bad = r#"{"name":"sneaky","grants":{"kernel":["uk_evolve","uk_model_create"]}}"#;
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
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        uk_clear_caller();
        set_caller_gadget("mod_tagged");
        // The loopback injects the request principal to match the caller identity.
        let req =
            r#"{"principal":"mod_tagged","effect":"send_notification","params":{"to":"dave"}}"#;
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
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
            arr.iter()
                .all(|e| e["caller"]["principal"] != "f8_audit_owner"),
            "reader without observer grant must not see f8_audit_owner: {json}"
        );

        // Reader WITH the observer grant sees it.
        set_caller_bounded("f8_audit_peer", &["f8_audit_owner"]);
        let json = read_buf(|b, c| uk_audit_list(b, c));
        let entries: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = entries.as_array().expect("audit list must be an array");
        assert!(
            arr.iter()
                .any(|e| e["caller"]["principal"] == "f8_audit_owner"),
            "peer with observer grant must see f8_audit_owner: {json}"
        );
        uk_clear_caller();
    }

    // ── S18 (F17): resource introductions ────────────────────────────────

    /// The FFI surface: `uk_resource_introduce` / `uk_request_resource` / `uk_resource_use`
    /// / `uk_resource_forfeit` operate on a JSON-string resource id.
    fn ffi_str(f: extern "C" fn(*const u8, i64) -> i64, s: &str) -> i64 {
        let (ptr, len) = json_ptr(s);
        f(ptr, len)
    }

    #[test]
    fn introduce_grants_resource_and_use_is_direct() {
        uk_clear_caller();
        assert_eq!(ffi_str(uk_resource_introduce, r#""s3.bucket#demo""#), 0);
        // The trusted harness (no bounded grants) may exercise a minted resource.
        assert_eq!(ffi_str(uk_resource_use, r#""s3.bucket#demo""#), 0);
        // Re-introduction at the single-mint chokepoint is refused (UK-4402).
        assert_eq!(ffi_str(uk_resource_introduce, r#""s3.bucket#demo""#), -4402);
        // Forfeit revokes; afterwards the id is no longer minted (UK-4403).
        assert_eq!(ffi_str(uk_resource_forfeit, r#""s3.bucket#demo""#), 0);
        assert_eq!(ffi_str(uk_resource_use, r#""s3.bucket#demo""#), -4403);
    }

    #[test]
    fn un_introduced_resource_call_is_4401() {
        uk_clear_caller();
        assert_eq!(ffi_str(uk_resource_introduce, r#""github.repo#secret""#), 0);
        // A bounded caller with an empty `resources` grant has NOT been introduced to the
        // session: the call is denied with UK-4401 RESOURCE_UNINTRODUCED.
        set_caller_bounded("f17_agent_bnd", &[]);
        assert_eq!(ffi_str(uk_resource_use, r#""github.repo#secret""#), -4401);
        uk_clear_caller();
        assert_eq!(ffi_str(uk_resource_forfeit, r#""github.repo#secret""#), 0);
    }

    #[test]
    fn request_queues_for_approval_and_audits() {
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        uk_audit_clear();
        uk_clear_caller();
        set_caller_gadget("f17_requester");
        let h = ffi_str(uk_request_resource, r#""github.repo#demo""#);
        assert!(h > 0, "resource request must queue, got {h}");

        // The approval queue is visible to the human console.
        let pending = read_buf(|b, c| uk_resource_pending(b, c));
        let pending: serde_json::Value = serde_json::from_str(&pending).unwrap();
        let arr = pending.as_array().expect("pending must be an array");
        assert!(
            arr.iter().any(|e| {
                e["resource_id"] == "github.repo#demo"
                    && e["requested_by"]["principal"] == "f17_requester"
            }),
            "queue must carry the requested id + requester: {pending}"
        );

        // An approval_pending audit row was written for the human.
        let entries = audit_list();
        let arr = entries.as_array().expect("audit must be an array");
        assert!(
            arr.iter().any(|e| {
                e["symbol"] == "uk_request_resource"
                    && e["detail"]
                        .as_str()
                        .unwrap_or("")
                        .contains("approval_pending")
            }),
            "approval_pending must be audited: {entries}"
        );
        uk_clear_caller();
    }

    // ── S19 (F18): gatekeeper approval console ────────────────────────────

    #[test]
    fn gatekeeper_approve_resolves_pending_and_lands_in_stream() {
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        uk_audit_clear();
        uk_clear_caller();

        // A side-effecting export submits a Pending action with a simulated outcome.
        let req = r#"{"principal":"scan-agent","effect":"notify_admins","params":{"msg":"new scan"},"provisional":{"forecast":true,"threat":"none"}}"#;
        let (p, l) = json_ptr(req);
        let handle = uk_action_submit(p, l);
        assert!(handle > 0, "sidecar submit must succeed, got {handle}");

        // While pending, the action sits in the gatekeeper console queue carrying the forecast.
        let pending_json = read_buf(|b, c| uk_gate_list_pending(b, c));
        let pending: serde_json::Value = serde_json::from_str(&pending_json).unwrap();
        let pending = pending.as_array().expect("pending must be an array");
        let found = pending
            .iter()
            .any(|v| v[0] == serde_json::json!(handle) && v[1]["effect"] == "notify_admins");
        assert!(
            found,
            "pending console must list the scan action: {pending:?}"
        );

        // The human operator approves through uk_gate_approve (audited with their principal).
        set_caller_gadget("scan_operator");
        assert_eq!(uk_gate_approve(handle), 0);

        // It is no longer pending...
        let pending_json = read_buf(|b, c| uk_gate_list_pending(b, c));
        let pending: serde_json::Value = serde_json::from_str(&pending_json).unwrap();
        assert!(
            !pending
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v[0] == serde_json::json!(handle)),
            "approved action leaves the approval queue: {pending}"
        );
        // ...and the simulated forecast became the applied result.
        let rec_json = read_buf(|b, c| uk_action_get(handle, b, c));
        let record: serde_json::Value = serde_json::from_str(&rec_json).unwrap();
        assert_eq!(record["state"], "approved");
        assert_eq!(record["applied"]["forecast"]["threat"], "none");

        // The approval is in the audit trail with the human principal.
        let entries = audit_list();
        let arr = entries.as_array().unwrap();
        assert!(
            arr.iter().any(|e| {
                e["symbol"] == "uk_gate_approve"
                    && e["caller"]["principal"] == "scan_operator"
                    && e["detail"]
                        .as_str()
                        .unwrap_or("")
                        .contains("by='scan_operator'")
            }),
            "human approval must be audited: {entries}"
        );
        uk_clear_caller();
    }

    #[test]
    fn gatekeeper_reject_keeps_pending_out_of_applied() {
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        uk_audit_clear();
        uk_clear_caller();
        let req = r#"{"principal":"scan-agent","effect":"notify_admins","params":{"msg":"x"},"provisional":{"forecast":true}}"#;
        let (p, l) = json_ptr(req);
        let handle = uk_action_submit(p, l);
        assert!(handle > 0);

        set_caller_gadget("scan_operator");
        assert_eq!(uk_gate_reject(handle), 0);

        let rec_json = read_buf(|b, c| uk_action_get(handle, b, c));
        let record: serde_json::Value = serde_json::from_str(&rec_json).unwrap();
        assert_eq!(record["state"], "rejected");
        assert!(
            record["applied"].is_null(),
            "rejected action has no applied result"
        );

        // A gate approve over a resolved action is refused (UK-4005).
        assert_eq!(uk_gate_approve(handle), -4005);
        uk_clear_caller();
    }

    // ── S21 (F20): trust annotations (observe vs mutate) + console-vetted ──
    //
    // The vetted registry and the action lane are both kernel-global, so these
    // serialize on ACTION_TESTS_LOCK and reset both stores first.

    fn pending_list() -> serde_json::Value {
        let json = read_buf(|b, c| uk_gate_list_pending(b, c));
        serde_json::from_str(&json).unwrap()
    }

    /// A gadget whose grant carries a F20 trust annotation for one effect.
    fn set_caller_annotated(principal: &str, effect_kinds: &str) {
        let caller = format!(
            r#"{{"from":"gadget","principal":"{principal}","grants":{{"kernel":[],"effects":["{principal}_eff"],"observers":[],"resources":[],"effect_kinds":{effect_kinds}}}}}"#
        );
        uk_set_caller(&caller).expect("annotated caller json must parse");
    }

    #[test]
    fn observe_kind_effect_never_queues() {
        // F20 readOnlyHint: an observe-annotated effect applies immediately — it never
        // occupies the approval lane, so no pending approval is needed.
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_vetted();
        uk_clear_caller();
        set_caller_annotated(
            "obs_probe",
            r#"[{"name":"obs_probe_eff","effect_kind":"observe"}]"#,
        );

        let req = r#"{"principal":"obs_probe","effect":"obs_probe_eff","params":{"metric":"qps"}}"#;
        let (p, l) = json_ptr(req);
        let handle = uk_action_submit(p, l);
        assert!(handle > 0);

        let record: serde_json::Value =
            serde_json::from_str(&read_buf(|b, c| uk_action_get(handle, b, c))).unwrap();
        assert_eq!(
            record["state"], "approved",
            "observe-kind applies immediately: {record}"
        );
        assert_eq!(record["applied"]["applied"], true);

        assert!(
            !pending_list()
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v[0] == serde_json::json!(handle)),
            "observe-kind never enters the approval queue"
        );
        uk_clear_caller();
    }

    #[test]
    fn mutate_kind_effect_requires_pending_approval() {
        // A mutate-kind effect by an un-vetted caller can never apply without a pending
        // approval: the submission queues, stays provisional, and only a gate approval
        // promotes it to applied.
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_vetted();
        uk_clear_caller();
        set_caller_gadget("mut_mod");
        let req = r#"{"principal":"mut_mod","effect":"delete_row","params":{"row":7},"provisional":{"rows":1}}"#;
        let (p, l) = json_ptr(req);
        let handle = uk_action_submit(p, l);
        assert!(handle > 0);

        let record: serde_json::Value =
            serde_json::from_str(&read_buf(|b, c| uk_action_get(handle, b, c))).unwrap();
        assert_eq!(record["state"], "pending", "mutate-kind queues: {record}");
        assert!(
            record["applied"].is_null(),
            "no applied result without approval"
        );
        assert_eq!(
            record["result"]["rows"], 1,
            "provisional is served meanwhile"
        );

        // The pending lane carries exactly this action; approving it promotes applied.
        let pending = pending_list();
        let arr = pending.as_array().unwrap();
        assert!(arr.iter().any(|v| v[0] == serde_json::json!(handle)));
        uk_clear_caller();
        assert_eq!(uk_gate_approve(handle), 0);
        let record: serde_json::Value =
            serde_json::from_str(&read_buf(|b, c| uk_action_get(handle, b, c))).unwrap();
        assert_eq!(record["state"], "approved");
        assert_eq!(record["applied"]["forecast"]["rows"], 1);
        uk_clear_caller();
    }

    #[test]
    fn vetted_mutate_effect_auto_applies() {
        // A console-vetted principal may auto-apply a mutate-kind effect without a
        // pending approval (the vetted marker is the console's invitation).
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_vetted();
        uk_clear_caller();

        // The operator console mints the vetted marker for the principal.
        let (p, l) = json_ptr("vet_ops");
        assert_eq!(uk_registry_vetted(p, l, 1), 0, "console mints vetted");

        set_caller_gadget("vet_ops");
        let req = r#"{"principal":"vet_ops","effect":"rotate_keys","params":{"scope":"prod"}}"#;
        let (p, l) = json_ptr(req);
        let handle = uk_action_submit(p, l);
        assert!(handle > 0);
        let record: serde_json::Value =
            serde_json::from_str(&read_buf(|b, c| uk_action_get(handle, b, c))).unwrap();
        assert_eq!(
            record["state"], "approved",
            "vetted mutate auto-applies: {record}"
        );
        assert!(
            !pending_list()
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v[0] == serde_json::json!(handle)),
            "vetted action skips the approval lane"
        );
        uk_clear_caller();
    }

    #[test]
    fn unvetted_console_clearing_flag_leaves_queue_intact() {
        // Clearing the vetted flag never touches the approval queue: a mutation that
        // already queued stays pending for the human to resolve.
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_vetted();
        uk_clear_caller();

        set_caller_gadget("later_unvetted");
        let req = r#"{"principal":"later_unvetted","effect":"remove_user","params":{"uid":9}}"#;
        let (p, l) = json_ptr(req);
        let handle = uk_action_submit(p, l);
        assert!(handle > 0);

        // Mint, then clear, the vetted flag from the console.
        uk_clear_caller();
        let (p, l) = json_ptr("later_unvetted");
        assert_eq!(uk_registry_vetted(p, l, 1), 0);
        assert_eq!(uk_registry_vetted(p, l, 0), 0);

        // The queue still carries the same pending action: un-vetting did not resolve
        // or drop anything. (Assert the handle persists rather than whole-queue
        // equality — the lane is a shared store, other tests may append concurrently.)
        let after = pending_list();
        assert!(
            after
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v[0] == serde_json::json!(handle)),
            "approval queue survives flag clearing: {after:?}"
        );
        let record: serde_json::Value =
            serde_json::from_str(&read_buf(|b, c| uk_action_get(handle, b, c))).unwrap();
        assert_eq!(record["state"], "pending");
        uk_clear_caller();
    }

    #[test]
    fn registry_vetted_is_console_only() {
        // A module can never self-declare vetted status (UK-4501); the marker comes
        // only from the operator console (hook, untrusted-bounds absent).
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        uk_clear_vetted();
        uk_clear_caller();

        set_caller_gadget("wannabe");
        let (p, l) = json_ptr("wannabe");
        assert_eq!(
            uk_registry_vetted(p, l, 1),
            -4501,
            "gadget cannot self-mint vetted"
        );

        // The operator harness can.
        uk_clear_caller();
        assert_eq!(uk_registry_vetted(p, l, 1), 0);
        assert_eq!(uk_registry_vetted(p, l, 0), 0);
        uk_clear_caller();
    }

    // ── S23 (F22): observability context + owner logger + secret discipline ──
    //
    // The owner sink and observability thread-local are kernel-global per process,
    // so these serialize on ACTION_TESTS_LOCK and reset their stores first.

    fn owner_lines() -> Vec<String> {
        let json = read_buf(|b, c| uk_owner_list(b, c));
        serde_json::from_str(&json).unwrap_or_default()
    }

    #[test]
    fn owner_logger_writes_dot_component_lines() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(uk_owner_clear() >= 0);
        owner_log("kernel.audit", "entry persisted");
        owner_log("kernel.action", "approval promoted");
        let lines = owner_lines();
        assert!(
            lines.iter().any(|l| l == "(kernel.audit) entry persisted"),
            "{lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l == "(kernel.action) approval promoted"),
            "{lines:?}"
        );
        assert!(uk_owner_clear() >= 2, "clear removes the lines");
        assert!(owner_lines().is_empty());
    }

    #[test]
    fn observability_context_threads_into_audit_entries() {
        // The audit trail is shared; serialize with the other audit-appending tests.
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        uk_audit_clear();
        uk_clear_caller();

        // Host seeds a per-call context (AsyncLocal analog) before the call.
        uk_observability_set(r#"{"trace_id":"f22-42","component":"kernel.audit"}"#)
            .expect("context json parses");
        let caller = r#"{"from":"gadget","principal":"obs_probe"}"#;
        uk_set_caller(caller).expect("caller json parses");
        let _ = uk_audit_append(r#"{"symbol":"uk_evolve","args":[{"t":0.1}],"ok":true}"#);
        uk_clear_caller();
        uk_observability_clear();

        let json = read_buf(|b, c| uk_audit_list(b, c));
        let entries: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = entries.as_array().unwrap();
        let mine = arr
            .iter()
            .find(|e| e["symbol"] == "uk_evolve" && e["caller"]["principal"] == "obs_probe")
            .expect("seeded call must be audited: {json}");
        assert_eq!(
            mine["context"]["trace_id"], "f22-42",
            "trace id threads per call"
        );
        assert_eq!(mine["component"], "kernel.audit");

        // A call *after* clear carries no context.
        let caller = r#"{"from":"gadget","principal":"obs_clear"}"#;
        uk_set_caller(caller).expect("caller json parses");
        let _ = uk_audit_append(r#"{"symbol":"uk_version","args":[],"ok":true}"#);
        uk_clear_caller();
        let json = read_buf(|b, c| uk_audit_list(b, c));
        let entries: serde_json::Value = serde_json::from_str(&json).unwrap();
        let mine = entries
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["symbol"] == "uk_version" && e["caller"]["principal"] == "obs_clear")
            .expect("clear call must be audited");
        assert_eq!(mine["context"], serde_json::Value::Null, "context cleared");
    }

    #[test]
    fn report_issue_is_noop_without_error_report_binding() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var("ERROR_REPORT_BINDING");
        }
        uk_owner_clear();
        let (p, l) = json_ptr(r#"{"message":"transient encode failure"}"#);
        // No binding → the report is a no-op and nothing lands in any log.
        assert_eq!(uk_report_issue(p, l), 0);
        let lines = owner_lines();
        assert!(
            lines.is_empty(),
            "no binding must mean no report: {lines:?}"
        );
    }

    #[test]
    fn report_issue_bound_appends_sanitized_line() {
        let _lock = ACTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("ERROR_REPORT_BINDING", "test://error-queue");
        }
        uk_owner_clear();
        let (p, l) = json_ptr(r#"{"message":"encode failure","api_key":"sec-F22-SECRET-xyz"}"#);
        assert_eq!(uk_report_issue(p, l), 1, "bound report is recorded");
        let lines = owner_lines();
        assert!(
            !lines.is_empty(),
            "bound report must land in the owner sink"
        );
        let joined = lines.join("\n");
        assert!(
            !joined.contains("sec-F22-SECRET-xyz"),
            "reported secret must never be logged raw: {joined}"
        );
        assert!(
            joined.contains("***REDACTED***"),
            "the sensitive field is redacted: {joined}"
        );
        unsafe {
            std::env::remove_var("ERROR_REPORT_BINDING");
        }
    }

    #[test]
    fn audit_trail_never_persists_fixture_secret() {
        // F22 discipline gate: feed a known token through the logging surface (the
        // host loopback's `uk_audit_append` — where a leaked credential-keyed param
        // would land) and scan audit + owner logs for it.
        let _lock = AUDIT_AGENT_TESTS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        uk_audit_clear();
        uk_owner_clear();
        uk_clear_caller();
        let token = "SEC-UNFER-9f2b-4e6c";

        // The loopback tags the caller, then appends one audit entry with the full
        // args it received — including a credential-keyed param that must be scrubbed.
        let caller = r#"{"from":"gadget","principal":"fixture_leaker"}"#;
        uk_set_caller(caller).expect("caller json parses");
        let entry = format!(
            r#"{{"symbol":"uk_action_submit","args":[{{"principal":"fixture_leaker","effect":"send_notification","params":{{"api_key":"{token}"}}}}],"ok":true}}"#
        );
        assert!(
            uk_audit_append(&entry) > 0,
            "audit append must assign a seq"
        );
        uk_clear_caller();

        // The audit trail (its serialized JSON) + owner sink never contain the token.
        let audit_json = read_buf(|b, c| uk_audit_list(b, c));
        assert!(
            !audit_json.contains(token),
            "audit trail must never contain the secret token: {audit_json}"
        );
        assert!(
            audit_json.contains("***REDACTED***"),
            "the sensitive key's value must be redacted in the trail"
        );
        let lines = owner_lines();
        assert!(
            lines.iter().all(|l| !l.contains(token)),
            "owner logs must never contain the token"
        );
    }
}
