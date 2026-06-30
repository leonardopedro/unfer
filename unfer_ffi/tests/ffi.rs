use unfer_ffi::*;
use unfer_protocol::Code;

const HARMONIC_SPEC: &str = r#"{
  "hamiltonian": {
    "kind": "builtin",
    "name": "harmonic_chain",
    "params": {"n_modes": 2, "omega": 1.0}
  },
  "prior": {
    "kind": "superposition",
    "terms": [
      {"re": 0.5, "im": 0.0, "spec": {"kind": "vacuum"}},
      {"re": 0.5, "im": 0.0, "spec": {"kind": "bosons", "modes": [[0, 1]]}}
    ]
  },
  "solver": {
    "krylov_dim": 4,
    "prune_eps": 1e-12,
    "max_components": 50000,
    "restarts": 1,
    "device": {"kind": "cpu"}
  }
}"#;

const EVENT_VACUUM: &str = r#"{"kind": "vacuum"}"#;
const EVENT_BOSON_GE1: &str = r#"{"kind": "boson_mode_total", "mode": 0, "cmp": "ge", "value": 1}"#;

fn json_ptr(s: &[u8]) -> (*const u8, i64) {
    (s.as_ptr(), s.len() as i64)
}

fn read_result(model: i64) -> String {
    let needed = uk_get_result(model, std::ptr::null_mut(), 0);
    assert!(needed > 0, "uk_get_result returned {needed}");
    let mut buf = vec![0u8; needed as usize + 1];
    let written = uk_get_result(model, buf.as_mut_ptr(), buf.len() as i64);
    assert_eq!(written, needed);
    buf.truncate(needed as usize);
    String::from_utf8(buf).expect("valid UTF-8")
}

fn read_error() -> String {
    let needed = uk_last_error(std::ptr::null_mut(), 0);
    if needed == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; needed as usize + 1];
    let written = uk_last_error(buf.as_mut_ptr(), buf.len() as i64);
    assert_eq!(written, needed);
    buf.truncate(needed as usize);
    String::from_utf8(buf).expect("valid UTF-8")
}

#[test]
fn version_returns_positive() {
    assert!(uk_version() > 0, "uk_version should be positive");
}

#[test]
fn init_succeeds() {
    let (ptr, len) = json_ptr(b"{}");
    let r = uk_init(ptr, len);
    assert_eq!(r, 0, "uk_init with {{}} should return 0");
}

#[test]
fn happy_path_create_evolve_probability_free() {
    let (ptr, len) = json_ptr(HARMONIC_SPEC.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0, "uk_model_create should return positive handle");

    let (ptr, len) = json_ptr(br#"{"t": 0.05}"#);
    let r = uk_evolve(model, ptr, len);
    assert_eq!(r, 0, "uk_evolve should succeed");

    let evolve_json = read_result(model);
    let evolve_val: serde_json::Value =
        serde_json::from_str(&evolve_json).expect("evolve result is JSON");
    assert!(evolve_val["norm"].as_f64().unwrap() > 0.99, "norm ~1");

    let (ptr, len) = json_ptr(EVENT_VACUUM.as_bytes());
    let r = uk_event_probability(model, ptr, len);
    assert_eq!(r, 0, "uk_event_probability should succeed");

    let prob_json = read_result(model);
    let prob_val: serde_json::Value =
        serde_json::from_str(&prob_json).expect("probability result is JSON");
    let p = prob_val["probability"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&p), "probability in [0,1], got {p}");

    assert_eq!(uk_model_free(model), 0, "uk_model_free should return 0");
}

#[test]
fn bad_handle_returns_1004() {
    let (ptr, len) = json_ptr(EVENT_VACUUM.as_bytes());
    let r = uk_event_probability(99999, ptr, len);
    assert_eq!(r, -(Code::BAD_HANDLE.raw() as i64), "expected -1004");

    let err_json = read_error();
    let diag: serde_json::Value = serde_json::from_str(&err_json).expect("error is JSON");
    assert_eq!(diag["code"].as_u64(), Some(Code::BAD_HANDLE.raw() as u64));
    assert!(!diag["message"].as_str().unwrap().is_empty());
}

#[test]
fn bad_json_returns_1001() {
    let (ptr, len) = json_ptr(b"not valid json {{{");
    let r = uk_model_create(ptr, len);
    assert_eq!(r, -(Code::BAD_JSON.raw() as i64), "expected -1001");

    let err_json = read_error();
    let diag: serde_json::Value = serde_json::from_str(&err_json).expect("error is JSON");
    assert_eq!(diag["code"].as_u64(), Some(Code::BAD_JSON.raw() as u64));
}

#[test]
fn bad_handle_on_free_returns_1004() {
    let r = uk_model_free(88888);
    assert_eq!(r, -(Code::BAD_HANDLE.raw() as i64));
}

#[test]
fn condition_then_probability_is_one() {
    let (ptr, len) = json_ptr(HARMONIC_SPEC.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    let (ptr, len) = json_ptr(EVENT_BOSON_GE1.as_bytes());
    let r = uk_condition(model, ptr, len);
    assert_eq!(r, 0, "uk_condition should succeed");

    let cond_json = read_result(model);
    let cond_val: serde_json::Value =
        serde_json::from_str(&cond_json).expect("condition result is JSON");
    let prior_p = cond_val["prior_probability"].as_f64().unwrap();
    assert!(
        (prior_p - 0.5).abs() < 1e-10,
        "prior P(E) ~ 0.5, got {prior_p}"
    );

    let (ptr, len) = json_ptr(EVENT_BOSON_GE1.as_bytes());
    let r = uk_event_probability(model, ptr, len);
    assert_eq!(r, 0);
    let prob_json = read_result(model);
    let prob_val: serde_json::Value =
        serde_json::from_str(&prob_json).expect("probability result is JSON");
    let post_p = prob_val["probability"].as_f64().unwrap();
    assert!(
        (post_p - 1.0).abs() < 1e-10,
        "post P(E) ~ 1.0, got {post_p}"
    );

    uk_model_free(model);
}

#[test]
fn zero_probability_condition_returns_2003() {
    let vacuum_spec = r#"{
      "hamiltonian": {
        "kind": "builtin",
        "name": "harmonic_chain",
        "params": {"n_modes": 2, "omega": 1.0}
      },
      "prior": {"kind": "vacuum"},
      "solver": {
        "krylov_dim": 4, "prune_eps": 1e-12,
        "max_components": 50000, "restarts": 1,
        "device": {"kind": "cpu"}
      }
    }"#;
    let (ptr, len) = json_ptr(vacuum_spec.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    let (ptr, len) = json_ptr(EVENT_BOSON_GE1.as_bytes());
    let r = uk_condition(model, ptr, len);
    assert_eq!(
        r,
        -(Code::ZERO_PROBABILITY_CONDITION.raw() as i64),
        "expected -2003"
    );

    let err_json = read_error();
    let diag: serde_json::Value = serde_json::from_str(&err_json).expect("error is JSON");
    assert_eq!(
        diag["code"].as_u64(),
        Some(Code::ZERO_PROBABILITY_CONDITION.raw() as u64)
    );

    uk_model_free(model);
}

#[test]
fn buffer_protocol_returns_needed_size() {
    let (ptr, len) = json_ptr(HARMONIC_SPEC.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    let (ptr, len) = json_ptr(EVENT_VACUUM.as_bytes());
    uk_event_probability(model, ptr, len);

    let needed = uk_get_result(model, std::ptr::null_mut(), 0);
    assert!(needed > 0, "should need > 0 bytes");

    let mut small = [0u8; 3];
    let written = uk_get_result(model, small.as_mut_ptr(), small.len() as i64);
    assert_eq!(written, needed, "written == needed regardless of cap");
    assert_eq!(&small, b"{\"p");

    let mut full = vec![0u8; needed as usize];
    let written2 = uk_get_result(model, full.as_mut_ptr(), full.len() as i64);
    assert_eq!(written2, needed);

    uk_model_free(model);
}

#[test]
fn set_prior_resets_state() {
    let (ptr, len) = json_ptr(HARMONIC_SPEC.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    let new_prior = r#"{"kind": "bosons", "modes": [[1, 2]]}"#;
    let (ptr, len) = json_ptr(new_prior.as_bytes());
    let r = uk_set_prior(model, ptr, len);
    assert_eq!(r, 0, "uk_set_prior should succeed");

    let (ptr, len) = json_ptr(EVENT_VACUUM.as_bytes());
    uk_event_probability(model, ptr, len);
    let prob_json = read_result(model);
    let prob_val: serde_json::Value =
        serde_json::from_str(&prob_json).expect("probability result is JSON");
    let p = prob_val["probability"].as_f64().unwrap();
    assert!(
        p < 1e-10,
        "P(vacuum) should be ~0 after setting boson prior"
    );

    uk_model_free(model);
}

#[test]
fn double_free_returns_1004() {
    let (ptr, len) = json_ptr(HARMONIC_SPEC.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    assert_eq!(uk_model_free(model), 0);
    let r = uk_model_free(model);
    assert_eq!(r, -(Code::BAD_HANDLE.raw() as i64), "second free -> -1004");
}

#[test]
fn unknown_builtin_returns_1002() {
    let spec = r#"{
      "hamiltonian": {"kind": "builtin", "name": "nonexistent", "params": {}},
      "prior": {"kind": "vacuum"},
      "solver": {"krylov_dim": 4, "prune_eps": 1e-12, "max_components": 50000, "restarts": 1, "device": {"kind": "cpu"}}
    }"#;
    let (ptr, len) = json_ptr(spec.as_bytes());
    let r = uk_model_create(ptr, len);
    assert_eq!(r, -(Code::UNKNOWN_BUILTIN_MODEL.raw() as i64));

    let err_json = read_error();
    let diag: serde_json::Value = serde_json::from_str(&err_json).expect("error is JSON");
    assert_eq!(
        diag["code"].as_u64(),
        Some(Code::UNKNOWN_BUILTIN_MODEL.raw() as u64)
    );
}

#[test]
fn null_pointer_with_nonzero_len_returns_1001() {
    let r = uk_model_create(std::ptr::null(), 5);
    assert_eq!(r, -(Code::BAD_JSON.raw() as i64));
}

#[test]
fn negative_len_returns_1001() {
    let (ptr, _) = json_ptr(b"{}");
    let r = uk_model_create(ptr, -1);
    assert_eq!(r, -(Code::BAD_JSON.raw() as i64));
}

#[test]
fn evolve_missing_t_returns_1001() {
    let (ptr, len) = json_ptr(HARMONIC_SPEC.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    let (ptr, len) = json_ptr(b"{}");
    let r = uk_evolve(model, ptr, len);
    assert_eq!(r, -(Code::BAD_JSON.raw() as i64));

    uk_model_free(model);
}

#[test]
fn qfm_tomo_via_ffi() {
    // P6 E #14: FFI integration test for the QFM tomographic pipeline.
    // Build a 4-point training set in d=4, k=2, K_2=4, krylov_dim=2,
    // evolve with a query, and assert qfm_output is present in the
    // returned EvolveReport JSON.
    let spec = r#"{
      "hamiltonian": {
        "kind": "qfm_tomography",
        "spec": {
          "training_data": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
          ],
          "k": 2,
          "k2": 4,
          "krylov_dim": 2,
          "seed": 42
        }
      },
      "prior": {"kind": "vacuum"},
      "solver": {
        "krylov_dim": 2,
        "prune_eps": 1e-12,
        "max_components": 50000,
        "restarts": 1,
        "device": {"kind": "cpu"}
      }
    }"#;
    let (ptr, len) = json_ptr(spec.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(
        model > 0,
        "uk_model_create should succeed for qfm_tomography"
    );

    // Evolve with a query (the first training point).
    let evolve_opts = r#"{"t": 1.0, "query": [1.0, 0.0, 0.0, 0.0]}"#;
    let (ptr, len) = json_ptr(evolve_opts.as_bytes());
    let r = uk_evolve(model, ptr, len);
    assert_eq!(
        r, 0,
        "uk_evolve should succeed for qfm_tomography with query"
    );

    // Drain the result and check for qfm_output.
    let report_json = read_result(model);
    let report: serde_json::Value =
        serde_json::from_str(&report_json).expect("report is valid JSON");
    let qfm_output = report
        .get("qfm_output")
        .and_then(|v| v.as_array())
        .expect("qfm_output should be a JSON array");
    assert_eq!(qfm_output.len(), 4, "qfm_output should have d=4 elements");
    for v in qfm_output {
        let f = v.as_f64().expect("qfm_output elements are f64");
        assert!(
            f.is_finite(),
            "qfm_output elements should be finite, got {f}"
        );
    }

    // Evolving WITHOUT a query on a QFM model must fail: the pipeline
    // requires a raw input to drive the 4-phase generate. The error is
    // mapped to UK-5000 (INTERNAL) — see `evolve_with_query`'s QFM branch.
    let evolve_opts_no_query = r#"{"t": 1.0}"#;
    let (ptr, len) = json_ptr(evolve_opts_no_query.as_bytes());
    let r = uk_evolve(model, ptr, len);
    assert_eq!(
        r,
        -(Code::INTERNAL.raw() as i64),
        "evolve on QFM model without query should return UK-5000"
    );

    uk_model_free(model);
}

#[test]
fn bayesian_update_via_ffi() {
    // P6 H follow-on: end-to-end test of the Quantum Bayesian Update
    // on the TSR-evolved prior via the C ABI. Create a QFM
    // tomographic model, call uk_bayesian_update with a single
    // observation, drain the result via uk_get_result, and verify
    // the schema.
    let spec = r#"{
      "hamiltonian": {
        "kind": "qfm_tomography",
        "spec": {
          "training_data": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
          ],
          "k": 2, "k2": 4, "krylov_dim": 4, "seed": 42
        }
      },
      "prior": {"kind": "vacuum"},
      "solver": {
        "krylov_dim": 4, "prune_eps": 1e-12,
        "max_components": 50000, "restarts": 1,
        "device": {"kind": "cpu"}
      }
    }"#;
    let (ptr, len) = json_ptr(spec.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0, "uk_model_create for qfm_tomography");

    // Bayesian update with a single observation at training point 0
    // and the default HMC options.
    let req = r#"{"observations": [[1.0, 0.0, 0.0, 0.0]]}"#;
    let (ptr, len) = json_ptr(req.as_bytes());
    let r = uk_bayesian_update(model, ptr, len);
    assert_eq!(r, 0, "uk_bayesian_update should succeed");

    // Drain the result and check the schema.
    let result_json = read_result(model);
    let result: serde_json::Value =
        serde_json::from_str(&result_json).expect("BayesianUpdateResult is valid JSON");
    assert_eq!(result["n_observations"].as_u64(), Some(1));
    assert!(result["log_posterior"].as_f64().unwrap().is_finite());
    let ml = result["mean_likelihood"].as_f64().unwrap();
    assert!(ml > 0.0 && ml <= 1.0, "mean_likelihood in (0, 1], got {ml}");
    let image = result["image"].as_array().expect("image is an array");
    assert_eq!(image.len(), 4, "image has d=4 elements");
    for v in image {
        let f = v.as_f64().expect("image elements are f64");
        assert!(f.is_finite(), "image element should be finite, got {f}");
    }
    assert!(result["solve_ms"].as_u64().is_some());

    uk_model_free(model);
}

#[test]
fn bayesian_update_via_ffi_zero_observations() {
    // Zero-observation Bayesian update: posterior = prior. The
    // result should have n_observations=0 and mean_likelihood=-1.
    let spec = r#"{
      "hamiltonian": {
        "kind": "qfm_tomography",
        "spec": {
          "training_data": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
          "k": 2, "k2": 4, "krylov_dim": 4, "seed": 42
        }
      },
      "prior": {"kind": "vacuum"},
      "solver": {
        "krylov_dim": 4, "prune_eps": 1e-12,
        "max_components": 50000, "restarts": 1,
        "device": {"kind": "cpu"}
      }
    }"#;
    let (ptr, len) = json_ptr(spec.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    let req = r#"{"observations": []}"#;
    let (ptr, len) = json_ptr(req.as_bytes());
    assert_eq!(uk_bayesian_update(model, ptr, len), 0);

    let result_json = read_result(model);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();
    assert_eq!(result["n_observations"].as_u64(), Some(0));
    assert!(
        (result["mean_likelihood"].as_f64().unwrap() + 1.0).abs() < 1e-12,
        "mean_likelihood should be -1 for prior-only, got {:?}",
        result["mean_likelihood"]
    );

    uk_model_free(model);
}

#[test]
fn bayesian_update_via_ffi_on_non_qfm_returns_5000() {
    // The Bayesian update is QFM-only; calling it on a non-QFM model
    // should return UK-5000 (INTERNAL) with an "Internal" diagnostic.
    let (ptr, len) = json_ptr(HARMONIC_SPEC.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    let req = r#"{"observations": [[0.0, 0.0]]}"#;
    let (ptr, len) = json_ptr(req.as_bytes());
    let r = uk_bayesian_update(model, ptr, len);
    assert_eq!(
        r,
        -(Code::INTERNAL.raw() as i64),
        "non-QFM model should return UK-5000, got {r}"
    );

    uk_model_free(model);
}

#[test]
fn bayesian_update_via_ffi_bad_obs_dim_returns_1001() {
    // Observation with wrong dimension should return UK-1001 (BAD_JSON,
    // surfaced via the Qfm DimensionMismatch -> to_diagnostic mapping).
    let spec = r#"{
      "hamiltonian": {
        "kind": "qfm_tomography",
        "spec": {
          "training_data": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
          "k": 2, "k2": 4, "krylov_dim": 4, "seed": 42
        }
      },
      "prior": {"kind": "vacuum"},
      "solver": {
        "krylov_dim": 4, "prune_eps": 1e-12,
        "max_components": 50000, "restarts": 1,
        "device": {"kind": "cpu"}
      }
    }"#;
    let (ptr, len) = json_ptr(spec.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    // Wrong-dim observation (2 instead of 4).
    let req = r#"{"observations": [[1.0, 0.0]]}"#;
    let (ptr, len) = json_ptr(req.as_bytes());
    let r = uk_bayesian_update(model, ptr, len);
    assert_eq!(
        r,
        -(Code::BAD_JSON.raw() as i64),
        "expected -1001 for dim mismatch, got {r}"
    );

    uk_model_free(model);
}

#[test]
fn qfm_tomo_via_ffi_bad_query_dim_returns_1001() {
    // A qfm_tomography model expects the query to have d elements; a
    // query of the wrong dimension must surface as BAD_JSON with a
    // DimensionMismatch-derived message.
    let spec = r#"{
      "hamiltonian": {
        "kind": "qfm_tomography",
        "spec": {
          "training_data": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
          "k": 2, "k2": 4, "krylov_dim": 2, "seed": 42
        }
      },
      "prior": {"kind": "vacuum"},
      "solver": {
        "krylov_dim": 2, "prune_eps": 1e-12,
        "max_components": 50000, "restarts": 1,
        "device": {"kind": "cpu"}
      }
    }"#;
    let (ptr, len) = json_ptr(spec.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    // Query of dimension 2 (should be 4).
    let evolve_opts = r#"{"t": 1.0, "query": [1.0, 0.0]}"#;
    let (ptr, len) = json_ptr(evolve_opts.as_bytes());
    let r = uk_evolve(model, ptr, len);
    assert_eq!(
        r,
        -(Code::BAD_JSON.raw() as i64),
        "expected -1001 for dim mismatch"
    );

    let err_json = read_error();
    let diag: serde_json::Value = serde_json::from_str(&err_json).expect("error is JSON");
    assert_eq!(diag["code"].as_u64(), Some(Code::BAD_JSON.raw() as u64));
    // The message should mention the expected and got dimensions.
    let msg = diag["message"].as_str().unwrap();
    assert!(
        msg.contains("4"),
        "message should mention expected d=4, got: {msg}"
    );
    assert!(
        msg.contains("2"),
        "message should mention got d=2, got: {msg}"
    );

    uk_model_free(model);
}
