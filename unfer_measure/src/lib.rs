//! `svcmux-bundle-v1` — the canonical per-service measurement.
//!
//! The confidential guest runs three webservices inside one TDX domain. A
//! single TD quote covers the whole boot chain, so *independent* attestation
//! per webservice must come from a digest each service computes for itself
//! and anyone can recompute from outside:
//!
//! ```text
//! bundle = "svcmux-bundle-v1" ++ NUL
//!        ++ service_id               ++ NUL
//!        ++ hex(sha256(exe bytes))   ++ NUL
//!        ++ hex(sha256(canonical_config)) ++ NUL
//!        ++ state_root               ++ NUL
//!        ++ listen
//! canonical_config = {"args":[argv[1..]],"env":{selected keys}}
//! bundle_digest    = hex(sha256(bundle))
//! binding          = hex(sha256(utf8(bundle_digest) ++ NUL ++ utf8(nonce)))
//! ```
//!
//! Three implementations exist on purpose — `svcmux` in the guest, this
//! crate for the services, `rs_service_bundle` in australVM — and each pins
//! the same test vectors. A service cannot misreport its own measurement
//! without disagreeing with the supervisor's manifest and with the build's
//! `service-pins.json`, and a tampered binary cannot match either.
//!
//! Under svcmux the selected env keys are exactly the ones the supervisor
//! declared (`SVCMUX_ENV_KEYS`); running standalone, a service falls back to
//! its own name prefixes, which is deterministic but only comparable with a
//! manifest built from the same view.

use serde::Serialize;
use sha2::{Digest, Sha256};

/// The spec tag. Changing the canonical form means a new tag, never a
/// silent edit: old and new attestors would otherwise both say "v1".
pub const BUNDLE_SPEC: &str = "svcmux-bundle-v1";

/// Identity a service supplies when running unsupervised; under svcmux the
/// `SVCMUX_*` bookkeeping variables take precedence.
pub struct Identity<'a> {
    pub id: &'a str,
    pub state_root: &'a str,
    pub listen: &'a str,
    /// Env-name prefixes used to pick the config view when svcmux did not
    /// declare the keys (e.g. `["REFI_", "AUSTRAL_BIN"]`).
    pub env_prefixes: &'a [&'a str],
}

/// One service's measurement, in the shape the supervisor's manifest uses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Bundle {
    pub spec: String,
    pub id: String,
    pub binary_sha256: String,
    pub config_sha256: String,
    pub state_root: String,
    pub listen: String,
    pub bundle: String,
    pub path: String,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

/// `{"args":[…],"env":{…}}`, compact, env keys sorted (BTreeMap order).
pub fn canonical_config(args: &[String], env: &std::collections::BTreeMap<String, String>) -> Vec<u8> {
    let mut out = String::from("{\"args\":[");
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(a).expect("string serialises"));
    }
    out.push_str("],\"env\":{");
    for (i, (k, v)) in env.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(k).expect("key serialises"));
        out.push(':');
        out.push_str(&serde_json::to_string(v).expect("value serialises"));
    }
    out.push_str("}}");
    out.into_bytes()
}

/// The bundle digest (see the module docs for the exact byte sequence).
pub fn bundle_digest(
    id: &str,
    binary_sha256: &str,
    config_sha256: &str,
    state_root: &str,
    listen: &str,
) -> String {
    let mut v: Vec<u8> = Vec::with_capacity(256);
    for (i, part) in [
        BUNDLE_SPEC,
        id,
        binary_sha256,
        config_sha256,
        state_root,
        listen,
    ]
    .iter()
    .enumerate()
    {
        if i > 0 {
            v.push(0);
        }
        v.extend_from_slice(part.as_bytes());
    }
    sha256_hex(&v)
}

/// A nonce-bound statement of this bundle: the verifier sends a fresh
/// nonce, the service answers with `binding`, and replaying an old answer
/// to a new challenge fails.
pub fn attestation_binding(bundle_digest_hex: &str, nonce: &str) -> String {
    let mut v = Vec::with_capacity(bundle_digest_hex.len() + nonce.len() + 1);
    v.extend_from_slice(bundle_digest_hex.as_bytes());
    v.push(0);
    v.extend_from_slice(nonce.as_bytes());
    sha256_hex(&v)
}

fn selected_env(identity: &Identity) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut env = std::collections::BTreeMap::new();
    match std::env::var_os("SVCMUX_ENV_KEYS") {
        Some(keys) => {
            // Supervised: exactly the supervisor's declared map. A declared
            // key missing from our environment means our view of the config
            // differs from the manifest's — say so rather than hash a
            // silently different blob.
            for k in keys.to_string_lossy().split(',') {
                if k.is_empty() {
                    continue;
                }
                match std::env::var(k) {
                    Ok(v) => {
                        env.insert(k.to_string(), v);
                    }
                    Err(_) => {
                        return Err(format!("declared config key {k} is not in this process's environment"))
                    }
                }
            }
        }
        None => {
            // Standalone: our own prefixes, deterministic but only comparable
            // with a manifest built from the same view.
            for (k, v) in std::env::vars() {
                if identity.env_prefixes.iter().any(|p| k.starts_with(p)) {
                    env.insert(k, v);
                }
            }
        }
    }
    Ok(env)
}

impl Bundle {
    /// Measure the running process: its own executable bytes, its argv, and
    /// the config view described above. Under svcmux every non-binary field
    /// comes from the supervisor's bookkeeping; standalone, from `identity`.
    pub fn for_service(identity: &Identity) -> Result<Bundle, String> {
        let exe = std::fs::read("/proc/self/exe").map_err(|e| format!("read /proc/self/exe: {e}"))?;
        let path = std::fs::read_link("/proc/self/exe")
            .map_err(|e| format!("readlink /proc/self/exe: {e}"))?
            .to_string_lossy()
            .into_owned();
        let args: Vec<String> = std::env::args().skip(1).collect();
        let env = selected_env(identity)?;
        let config_sha256 = sha256_hex(&canonical_config(&args, &env));

        let pick = |key: &str, fallback: &str| -> String {
            std::env::var(key).unwrap_or_else(|_| fallback.to_string())
        };
        let id = pick("SVCMUX_SERVICE_ID", identity.id);
        let state_root = pick("SVCMUX_STATE_ROOT", identity.state_root);
        let listen = pick("SVCMUX_LISTEN", identity.listen);
        let binary_sha256 = sha256_hex(&exe);
        let bundle = bundle_digest(&id, &binary_sha256, &config_sha256, &state_root, &listen);

        Ok(Bundle {
            spec: BUNDLE_SPEC.to_string(),
            id,
            binary_sha256,
            config_sha256,
            state_root,
            listen,
            bundle,
            path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of(pairs: &[(&str, &str)]) -> std::collections::BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn canonical_config_is_compact_and_key_sorted() {
        let args = vec!["--state".to_string(), "dir".to_string()];
        let env = env_of(&[("MM0WEB_LISTEN", "127.0.0.1:8081")]);
        assert_eq!(
            String::from_utf8(canonical_config(&args, &env)).unwrap(),
            r#"{"args":["--state","dir"],"env":{"MM0WEB_LISTEN":"127.0.0.1:8081"}}"#
        );
    }

    /// The vector svcmux pins too — same bytes, other implementation.
    #[test]
    fn config_digest_matches_the_pinned_vector() {
        let args = vec!["--state".to_string(), "dir".to_string()];
        let env = env_of(&[("MM0WEB_LISTEN", "127.0.0.1:8081")]);
        assert_eq!(
            sha256_hex(&canonical_config(&args, &env)),
            "e5ed602fdff1c6824c8407a713bcd90a99f8037921c4c9ca54d8f24a01858958"
        );
    }

    #[test]
    fn bundle_digest_matches_the_pinned_vector() {
        assert_eq!(
            bundle_digest(
                "svc-a",
                &"0123456789abcdef".repeat(4),
                &"fedcba9876543210".repeat(4),
                "/run/svc-a",
                "127.0.0.1:8080",
            ),
            "6820fb2ab8939587369b84d2a7e6756a468f7532af459a8ef76c7a8a86c31901"
        );
    }

    #[test]
    fn binding_matches_the_pinned_vector() {
        assert_eq!(
            attestation_binding(
                "6820fb2ab8939587369b84d2a7e6756a468f7532af459a8ef76c7a8a86c31901",
                "nonce-abc",
            ),
            "5f703fe1f4ddedae452f38a662f24d1b3edea28d1362b22a305b0bc967f3f28f"
        );
    }
}
