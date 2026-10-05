//! C3 — configuration for the edge gateway: one vocabulary, one precedence.
//!
//! Precedence, lowest to highest:
//!
//!     defaults  <  config file  <  environment  <  flags
//!
//! `dynamic-arctic` already implements exactly this (see its `src/config.rs`), and
//! the point of putting it here is that there is now **one** dialect for the two
//! servers rather than one per server. The three design rules below are copied
//! from that implementation because they are the ones that matter, not because
//! the shape is exotic:
//!
//! 1. **Nothing here reads the process environment or the filesystem.** `resolve`
//!    takes the environment as a lookup closure and the file as a parsed value, so
//!    the precedence order is a property of a pure function a test can drive by
//!    passing three values instead of one. Mutating the real environment in a test
//!    is how suites become order-dependent.
//! 2. **A malformed value falls back to the layer below** rather than aborting
//!    startup, and the provenance map records the rejection so it is visible.
//!    A typo in an env var should not take the gateway down -- and a silently
//!    discarded value is worse than no value, because it looks like it worked.
//! 3. **Provenance is returned, not just the value**, so an operator can see
//!    *why* a setting is what it is instead of trusting a precedence order.
//!
//! The S22 admin console keys (`grants`, `auth`, `storage`, `backend` are hard;
//! the rest soft) are unaffected: those are *runtime* admin mutations over the
//! loopback, while this module is *startup* configuration. What this module must
//! not do is widen what the admin refuse list protects -- see `HARD_PATCH_KEYS`
//! below and the test that pins it.

use serde_json::{json, Value};

/// Keys as they appear in the config file and as `UNFER_*` environment
/// variables. One table, so the two spellings cannot drift apart.
pub const KEYS: &[(&str, &str, &str)] = &[
    // (key, env var, default)
    ("listen", "UNFER_LISTEN", "0.0.0.0:3000"),
    ("backend", "UNFER_BACKEND", "127.0.0.1:3001"),
];

/// Startup keys that the S22 admin console also treats as **hard** (never
/// patchable at runtime).
///
/// This is *not* a copy of `admin::HARD_KEYS` and must not become one — the
/// admin list is the authority and it lives behind the `audit` feature, so it
/// cannot be imported unconditionally from here. This is the startup-side subset:
/// the keys in `KEYS` that the admin console refuses to patch. Duplicating the
/// full list would create a second authority that drifts silently; the subset is
/// small enough to check, and
/// `hard_patch_keys_agree_with_the_admin_refuse_list` (under `--features audit`)
/// asserts the agreement so a new key cannot be added without noticing.
/// `backend` is here because it is startup configuration *and* a hard admin key;
/// `grants`, `auth` and `storage` are admin-only and have no startup spelling.
pub const HARD_PATCH_KEYS: &[&str] = &["backend"];

/// The effective configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub listen: String,
    pub backend: String,
}

/// Command-line overrides. Absent means "not supplied", which is what keeps flags
/// above env: an unset flag must not shadow anything.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Flags {
    pub listen: Option<String>,
    pub backend: Option<String>,
}

/// Which layer supplied a value, so `GET /version` can report the effective
/// configuration *and* where each part of it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    Default,
    File,
    Env,
    Flag,
}

impl Layer {
    pub fn as_str(self) -> &'static str {
        match self {
            Layer::Default => "default",
            Layer::File => "file",
            Layer::Env => "env",
            Layer::Flag => "flag",
        }
    }
}

/// key -> layer that won.
pub type Provenance = std::collections::BTreeMap<String, Layer>;

impl Config {
    pub fn defaults() -> Config {
        Config {
            listen: "0.0.0.0:3000".to_string(),
            backend: "127.0.0.1:3001".to_string(),
        }
    }

    /// Resolve the four layers.
    pub fn resolve(
        file: Option<&Value>,
        env: &dyn Fn(&str) -> Option<String>,
        flags: &Flags,
    ) -> (Config, Provenance) {
        let mut out = Config::defaults();
        let mut prov = Provenance::default();
        for key in KEYS.iter().map(|(k, _, _)| *k) {
            prov.insert(key.to_string(), Layer::Default);
        }

        if let Some(obj) = file {
            for (key, _, _) in KEYS {
                // Accept a JSON string, number or bool. Reading only `as_str()`
                // would mean a config written the obvious way is silently ignored
                // in favour of the default -- a file layer that quietly discards
                // what you wrote looks exactly like a file layer that is working.
                if let Some(raw) = obj.get(*key).and_then(scalar_to_string) {
                    if apply(&mut out, key, &raw) {
                        prov.insert((*key).to_string(), Layer::File);
                    }
                }
            }
        }
        for (key, env_name, _) in KEYS {
            if let Some(raw) = env(env_name) {
                if apply(&mut out, key, &raw) {
                    prov.insert((*key).to_string(), Layer::Env);
                }
            }
        }
        if let Some(v) = &flags.listen {
            apply(&mut out, "listen", v);
            prov.insert("listen".to_string(), Layer::Flag);
        }
        if let Some(v) = &flags.backend {
            apply(&mut out, "backend", v);
            prov.insert("backend".to_string(), Layer::Flag);
        }
        (out, prov)
    }

    /// The effective configuration as JSON, with provenance per key.
    ///
    /// This is what `/version` reports beyond the crate version: an operator
    /// debugging "why is it bound to the wrong port" gets the answer from the
    /// endpoint they already probe.
    pub fn to_json(&self, prov: &Provenance) -> Value {
        let layers: serde_json::Map<String, Value> = prov
            .iter()
            .map(|(k, layer)| (k.clone(), json!(layer.as_str())))
            .collect();
        json!({
            "listen": self.listen,
            "backend": self.backend,
            "layers": layers,
        })
    }
}

/// Render a JSON scalar as the string form the setters validate.
///
/// Objects and arrays return `None`: they are a configuration error, and treating
/// them as absent lets the layer below win, which is the documented behaviour for
/// a malformed value.
fn scalar_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Apply one value, validating it. Returns false if rejected.
fn apply(cfg: &mut Config, key: &str, raw: &str) -> bool {
    let v = raw.trim();
    if v.is_empty() {
        return false;
    }
    match key {
        "listen" => {
            if !valid_addr(v) {
                return false;
            }
            cfg.listen = v.to_string();
        }
        "backend" => {
            if !valid_addr(v) {
                return false;
            }
            cfg.backend = v.to_string();
        }
        _ => return false,
    }
    true
}

/// `host:port`, the only address form Pingora's `add_tcp` takes here.
///
/// Deliberately strict. A value that passes this and then fails inside Pingora
/// produces a startup panic with a message about the listener rather than about
/// the setting, and the operator has no way to tell which config layer produced
/// the bad address.
fn valid_addr(v: &str) -> bool {
    match v.rsplit_once(':') {
        Some((host, port)) => {
            !port.is_empty()
                && port.chars().all(|c| c.is_ascii_digit())
                && port.parse::<u16>().is_ok()
                && !host.is_empty()
                && !host.contains(char::is_whitespace)
        }
        None => false,
    }
}

/// What the command line asked for.
#[derive(Debug, Clone, PartialEq)]
pub enum Invocation {
    /// `unfer_edge init [--config FILE]` -- write a starter config. Carries the
    /// path because `init` is not necessarily the first argument: `--config`
    /// after it must still be honoured. Returning early on seeing `init` would
    /// silently drop it and write to the default path instead, which is a bug
    /// that looks like the flag was ignored.
    Init { file: Option<std::path::PathBuf> },
    Serve {
        file: Option<std::path::PathBuf>,
        flags: Flags,
    },
}

impl Flags {
    fn set_raw(&mut self, key: &str, raw: &str) -> Result<(), String> {
        match key {
            "listen" => self.listen = Some(raw.to_string()),
            "backend" => self.backend = Some(raw.to_string()),
            other => return Err(format!("unknown flag --{other}")),
        }
        Ok(())
    }
}

/// Parse the command line.
///
/// Hand-rolled, matching dynamic-arctic: this is two options and a subcommand, and
/// the crate carries no CLI dependency. More to the point it is a pure function,
/// so precedence is testable directly rather than by running the binary.
///
/// Both `--flag value` and `--flag=value` are accepted, because an operator who
/// types one of them should not get an error from the other.
pub fn parse_args(args: &[String]) -> Result<Invocation, String> {
    let mut it = args.iter().skip(1).peekable();
    let mut file = None;
    let mut flags = Flags::default();
    let mut init = false;

    while let Some(arg) = it.next() {
        if arg == "init" {
            // Recorded, not returned: the loop must keep going so a `--config`
            // that follows `init` is still seen.
            init = true;
            continue;
        }
        let Some(rest) = arg.strip_prefix("--") else {
            return Err(format!("unexpected argument {arg:?}"));
        };
        let (key, inline) = match rest.split_once('=') {
            Some((k, v)) => (k, Some(v.to_string())),
            None => (rest, None),
        };
        if key == "config" {
            file = Some(match inline {
                Some(v) => std::path::PathBuf::from(v),
                None => std::path::PathBuf::from(
                    it.next()
                        .ok_or_else(|| "--config expects a file path".to_string())?,
                ),
            });
            continue;
        }
        if key == "help" || key == "h" {
            return Err(usage());
        }
        let value = match inline {
            Some(v) => v,
            None => it
                .next()
                .ok_or_else(|| format!("--{key} expects a value"))?
                .to_string(),
        };
        flags.set_raw(key, &value)?;
    }

    Ok(if init {
        Invocation::Init { file }
    } else {
        Invocation::Serve { file, flags }
    })
}

pub fn usage() -> String {
    "usage: unfer_edge init [--config FILE]\n       unfer_edge [--config FILE] [--listen ADDR] [--backend ADDR]"
        .to_string()
}

/// Read a config file, treating "no path given" as "no file layer".
pub fn load_file(path: Option<&std::path::Path>) -> Result<Option<Value>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read config {}: {e}", path.display()))?;
    serde_json::from_str(&raw)
        .map(Some)
        .map_err(|e| format!("config {} is not valid JSON: {e}", path.display()))
}

/// The onboarding half: write a starter config.
///
/// Deterministic by construction -- same answers, same bytes -- so it can be
/// asserted on rather than eyeballed.
///
/// **Strictly valid JSON, comments as `//` keys.** The obvious thing to write
/// here is `// a comment` on its own line, and it is a trap: `serde_json` rejects
/// it, so the wizard would emit a file the server cannot read -- an onboarding
/// step that produces a broken config and reports success. Carrying the prose in
/// `"//"` keys keeps the file parseable by the resolver and by every other JSON
/// tool, and the resolver already ignores unknown keys.
pub fn starter_config(answers: &[(String, String)]) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"//\": \"unfer_edge configuration. Delete any key you do not set.\",\n");
    out.push_str("  \"//precedence\": \"defaults < this file < environment < command-line flags. A value set in several places takes the highest one; a malformed value falls back to the layer below. GET /version reports which layer won for each key, so if a setting is not what you expect, that endpoint says why.\",\n");
    for (key, env_name, default) in KEYS {
        out.push_str(&format!("\n  \"//{key}\": \"overridden by {env_name}\",\n"));
        let value = answers
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or(default);
        out.push_str(&format!("  \"{key}\": \"{value}\",\n"));
    }
    if answers.is_empty() {
        out.push_str("\n  \"//note\": \"Nothing was answered at init time, so every value is the default.\",\n");
    }
    // Trim the trailing comma left by the last emitted pair, then close the object.
    while out.trim_end().ends_with(',') {
        let trimmed = out.trim_end().trim_end_matches(',').to_string();
        out = format!("{trimmed}\n");
    }
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An environment stand-in. Owns its pairs so the closure needs no lifetime
    /// relationship to the caller's slice -- which also means a test cannot
    /// accidentally observe a mutation of the environment it did not make.
    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let owned: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |k| {
            owned
                .iter()
                .find(|(ek, _)| ek == k)
                .map(|(_, v)| v.clone())
        }
    }

    fn flags(listen: Option<&str>, backend: Option<&str>) -> Flags {
        Flags {
            listen: listen.map(str::to_string),
            backend: backend.map(str::to_string),
        }
    }

    fn layer_of(prov: &Provenance, key: &str) -> Layer {
        *prov.get(key).expect("every key has provenance")
    }

    // ---- precedence ---------------------------------------------------------

    #[test]
    fn defaults_apply_when_nothing_else_is_supplied() {
        let (cfg, prov) = Config::resolve(None, &env_of(&[]), &flags(None, None));
        assert_eq!(cfg, Config::defaults());
        assert_eq!(layer_of(&prov, "listen"), Layer::Default);
        assert_eq!(layer_of(&prov, "backend"), Layer::Default);
    }

    #[test]
    fn file_beats_default() {
        let file = json!({"listen": "10.0.0.1:8080"});
        let (cfg, prov) = Config::resolve(Some(&file), &env_of(&[]), &flags(None, None));
        assert_eq!(cfg.listen, "10.0.0.1:8080");
        assert_eq!(layer_of(&prov, "listen"), Layer::File);
    }

    #[test]
    fn env_beats_file() {
        let file = json!({"listen": "10.0.0.1:8080"});
        let env = [("UNFER_LISTEN", "10.0.0.2:9090")];
        let (cfg, prov) = Config::resolve(Some(&file), &env_of(&env), &flags(None, None));
        assert_eq!(cfg.listen, "10.0.0.2:9090");
        assert_eq!(layer_of(&prov, "listen"), Layer::Env);
    }

    #[test]
    fn flag_beats_everything() {
        let file = json!({"listen": "10.0.0.1:8080"});
        let env = [("UNFER_LISTEN", "10.0.0.2:9090")];
        let (cfg, prov) = Config::resolve(Some(&file), &env_of(&env), &flags(Some("10.0.0.3:7070"), None));
        assert_eq!(cfg.listen, "10.0.0.3:7070");
        assert_eq!(layer_of(&prov, "listen"), Layer::Flag);
    }

    #[test]
    fn the_full_ladder_is_ordered_on_one_key() {
        let file = json!({"backend": "1.1.1.1:1"});
        let env = [("UNFER_BACKEND", "2.2.2.2:2")];
        let f = flags(None, Some("3.3.3.3:3"));

        let (a, _) = Config::resolve(None, &env_of(&[]), &flags(None, None));
        assert_eq!(a.backend, Config::defaults().backend);
        let (b, _) = Config::resolve(Some(&file), &env_of(&[]), &flags(None, None));
        assert_eq!(b.backend, "1.1.1.1:1");
        let (c, _) = Config::resolve(Some(&file), &env_of(&env), &flags(None, None));
        assert_eq!(c.backend, "2.2.2.2:2");
        let (d, _) = Config::resolve(Some(&file), &env_of(&env), &f);
        assert_eq!(d.backend, "3.3.3.3:3");
    }

    // ---- the failure modes that make a config layer look like it works -------

    #[test]
    fn a_json_scalar_is_read_even_when_it_is_not_a_string() {
        // Reading only `as_str()` would make a config written the obvious way look
        // like it worked while being ignored. `scalar_to_string` is what prevents
        // that, so it is tested directly -- for *these* keys a bare number is
        // still rejected, because `listen` is a `host:port`, not a port. That is
        // validation doing its job, not the coercion failing: the next test pins
        // the fallback.
        assert_eq!(scalar_to_string(&serde_json::json!("s")), Some("s".into()));
        assert_eq!(scalar_to_string(&serde_json::json!(8080)), Some("8080".into()));
        assert_eq!(scalar_to_string(&serde_json::json!(true)), Some("true".into()));
        assert_eq!(scalar_to_string(&serde_json::json!({"a": 1})), None);
        assert_eq!(scalar_to_string(&serde_json::json!([1])), None);
    }

    #[test]
    fn a_bare_port_number_is_not_a_valid_listen_address() {
        // `{"listen": 8080}` is the intuitive mistake. It cannot become a
        // `host:port`, so it is rejected and the default stands -- visibly, via
        // provenance, rather than by half-applying it.
        let file = json!({"listen": 8080});
        let (cfg, prov) = Config::resolve(Some(&file), &env_of(&[]), &flags(None, None));
        assert_eq!(cfg.listen, Config::defaults().listen);
        assert_eq!(layer_of(&prov, "listen"), Layer::Default);
    }

    #[test]
    fn a_malformed_value_falls_back_and_leaves_provenance_at_the_layer_below() {
        let file = json!({"listen": "not-an-address"});
        let (cfg, prov) = Config::resolve(Some(&file), &env_of(&[]), &flags(None, None));
        assert_eq!(cfg.listen, Config::defaults().listen);
        // Not File: the file layer was tried and lost.
        assert_eq!(layer_of(&prov, "listen"), Layer::Default);
    }

    #[test]
    fn a_malformed_env_value_falls_back_to_the_file_layer() {
        let file = json!({"listen": "10.0.0.1:8080"});
        let env = [("UNFER_LISTEN", "garbage")];
        let (cfg, prov) = Config::resolve(Some(&file), &env_of(&env), &flags(None, None));
        assert_eq!(cfg.listen, "10.0.0.1:8080");
        assert_eq!(layer_of(&prov, "listen"), Layer::File);
    }

    #[test]
    fn a_json_object_is_treated_as_absent_not_as_a_string() {
        let file = json!({"listen": {"port": 80}});
        let (cfg, _) = Config::resolve(Some(&file), &env_of(&[]), &flags(None, None));
        assert_eq!(cfg.listen, Config::defaults().listen);
    }

    #[test]
    fn an_unknown_key_in_the_file_is_ignored_rather_than_fatal() {
        let file = json!({"listen": "10.0.0.1:8080", "nonsense": "x"});
        let (cfg, _) = Config::resolve(Some(&file), &env_of(&[]), &flags(None, None));
        assert_eq!(cfg.listen, "10.0.0.1:8080");
    }

    #[test]
    fn addresses_are_validated_before_pingora_sees_them() {
        for bad in ["", "  ", "host", "host:", ":8080", "host:port", "host:99999", "ho st:80"] {
            assert!(!valid_addr(bad), "{bad:?} should be rejected");
        }
        for good in ["0.0.0.0:3000", "127.0.0.1:3001", "[::1]:8080"] {
            assert!(valid_addr(good), "{good:?} should be accepted");
        }
    }

    #[test]
    fn every_key_has_provenance_even_when_unset() {
        let (_, prov) = Config::resolve(None, &env_of(&[]), &flags(None, None));
        for (key, _, _) in KEYS {
            assert!(prov.contains_key(*key), "{key} missing from provenance");
        }
    }

    // ---- arg parsing --------------------------------------------------------

    fn args(v: &[&str]) -> Vec<String> {
        std::iter::once("unfer_edge".to_string())
            .chain(v.iter().map(|s| s.to_string()))
            .collect()
    }

    #[test]
    fn both_flag_spellings_parse() {
        let a = parse_args(&args(&["--listen", "1.2.3.4:80"])).expect("space form");
        let b = parse_args(&args(&["--listen=1.2.3.4:80"])).expect("equals form");
        assert_eq!(a, b);
        let Invocation::Serve { flags, .. } = a else {
            panic!("expected Serve")
        };
        assert_eq!(flags.listen.as_deref(), Some("1.2.3.4:80"));
    }

    #[test]
    fn a_config_path_parses_in_both_spellings() {
        let a = parse_args(&args(&["--config", "/tmp/x.json"])).expect("space");
        let b = parse_args(&args(&["--config=/tmp/x.json"])).expect("equals");
        let (Invocation::Serve { file: fa, .. }, Invocation::Serve { file: fb, .. }) = (a, b)
        else {
            panic!("expected Serve")
        };
        assert_eq!(fa, fb);
        assert_eq!(fa.unwrap().to_str(), Some("/tmp/x.json"));
    }

    #[test]
    fn an_unknown_flag_names_itself() {
        let e = parse_args(&args(&["--nonsense", "1"])).expect_err("should refuse");
        assert!(e.contains("--nonsense"), "{e}");
    }

    #[test]
    fn a_valueless_flag_is_an_error_not_a_silent_default() {
        assert!(parse_args(&args(&["--listen"])).is_err());
    }

    #[test]
    fn help_returns_usage_as_an_error_so_the_process_can_exit_non_zero() {
        assert!(parse_args(&args(&["--help"]))
            .expect_err("help is Err")
            .contains("usage:"));
    }

    #[test]
    fn init_is_its_own_invocation() {
        assert_eq!(
            parse_args(&args(&["init"])).expect("init"),
            Invocation::Init { file: None }
        );
    }

    #[test]
    fn a_config_path_after_init_is_still_honoured() {
        // `init` must not return early: returning on the subcommand and discarding
        // the rest of argv is how `--config` after it gets silently ignored, and
        // the wizard then writes to the default path while reporting success.
        for form in [vec!["init", "--config", "/tmp/x.json"], vec!["init", "--config=/tmp/x.json"]] {
            let got = parse_args(&args(&form)).expect("init with config");
            assert_eq!(
                got,
                Invocation::Init {
                    file: Some(std::path::PathBuf::from("/tmp/x.json"))
                },
                "form {form:?} lost its --config path"
            );
        }
    }

    #[test]
    fn a_config_path_before_init_is_also_honoured() {
        assert_eq!(
            parse_args(&args(&["--config", "/tmp/y.json", "init"])).expect("init"),
            Invocation::Init {
                file: Some(std::path::PathBuf::from("/tmp/y.json"))
            }
        );
    }

    #[test]
    fn a_bad_flag_still_fails_even_in_init_mode() {
        assert!(parse_args(&args(&["init", "--nonsense", "1"])).is_err());
    }

    // ---- starter config (C3 onboarding) -------------------------------------

    #[test]
    fn the_starter_config_is_byte_identical_for_the_same_answers() {
        let a = starter_config(&[("listen".into(), "1.2.3.4:80".into())]);
        let b = starter_config(&[("listen".into(), "1.2.3.4:80".into())]);
        assert_eq!(a, b, "onboarding output must be deterministic");
    }

    #[test]
    fn the_starter_config_round_trips_through_the_resolver() {
        // The point of the wizard: what it writes is what the server reads. If the
        // file parses but the resolver ignores a key, the wizard has taught the
        // operator a lie.
        let text = starter_config(&[
            ("listen".into(), "10.1.1.1:1111".into()),
            ("backend".into(), "10.2.2.2:2222".into()),
        ]);
        let value: Value = serde_json::from_str(&text).expect("starter config is valid JSON");
        let (cfg, prov) = Config::resolve(Some(&value), &env_of(&[]), &flags(None, None));
        assert_eq!(cfg.listen, "10.1.1.1:1111");
        assert_eq!(cfg.backend, "10.2.2.2:2222");
        assert_eq!(layer_of(&prov, "listen"), Layer::File);
    }

    #[test]
    fn the_starter_config_mentions_every_key_and_its_env_var() {
        let text = starter_config(&[]);
        for (key, env_name, _) in KEYS {
            assert!(text.contains(&format!("\"{key}\"")), "{key} missing");
            assert!(text.contains(env_name), "{env_name} missing");
        }
        assert!(text.contains("defaults"), "precedence is not documented");
    }

    // ---- the S22 boundary this module must not widen -------------------------

    #[test]
    fn every_startup_key_is_either_soft_or_on_the_admin_refuse_list() {
        // `listen` is startup-only: the admin console has no opinion about it.
        // `backend` is both startup configuration and a hard admin key. A key that
        // is startup state but which the admin layer treats as soft runtime state
        // would be mutable from two places, so this forces the decision to be
        // written down rather than defaulted.
        for (key, _, _) in KEYS {
            if *key == "listen" {
                continue;
            }
            assert!(
                HARD_PATCH_KEYS.contains(key),
                "{key} is startup configuration but is not listed as a hard admin key"
            );
        }
    }

    #[cfg(feature = "audit")]
    #[test]
    fn hard_patch_keys_agree_with_the_admin_refuse_list() {
        // `admin::HARD_KEYS` is the authority. This table is the startup-side
        // subset, so every entry here must be in the admin list -- and every admin
        // key that also has a startup spelling must be in this one. The second
        // direction is what stops a startup key from being added as soft while the
        // admin console already treats the concept as hard.
        for key in HARD_PATCH_KEYS {
            assert!(
                super::super::admin::HARD_KEYS.contains(key),
                "{key} is claimed hard here but is not on admin::HARD_KEYS"
            );
        }
        for (key, _, _) in KEYS {
            if super::super::admin::HARD_KEYS.contains(key) {
                assert!(
                    HARD_PATCH_KEYS.contains(key),
                    "{key} is hard in the admin console but missing from HARD_PATCH_KEYS"
                );
            }
        }
    }

    #[test]
    fn version_json_reports_provenance_per_key() {
        let file = json!({"listen": "10.0.0.1:8080"});
        let env = [("UNFER_BACKEND", "10.0.0.9:9000")];
        let (cfg, prov) = Config::resolve(Some(&file), &env_of(&env), &flags(None, None));
        let v = cfg.to_json(&prov);
        assert_eq!(v["listen"], "10.0.0.1:8080");
        assert_eq!(v["backend"], "10.0.0.9:9000");
        assert_eq!(v["layers"]["listen"], "file");
        assert_eq!(v["layers"]["backend"], "env");
    }
}