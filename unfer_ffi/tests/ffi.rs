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
fn subscribe_and_poll() {
    let (ptr, len) = json_ptr(HARMONIC_SPEC.as_bytes());
    let model = uk_model_create(ptr, len);
    assert!(model > 0);

    let (ptr, len) = json_ptr(EVENT_VACUUM.as_bytes());
    let sub = uk_subscribe(model, ptr, len);
    assert!(sub > 0, "uk_subscribe should return positive sub handle");

    let mut buf = vec![0u8; 256];
    let needed = uk_poll(sub, buf.as_mut_ptr(), buf.len() as i64);
    assert!(needed > 0);

    buf.truncate(needed as usize);
    let val: serde_json::Value = serde_json::from_slice(&buf).expect("poll result is JSON");
    let p = val["probability"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&p));

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
