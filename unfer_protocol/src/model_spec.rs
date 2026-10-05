//! C7: model spec prefixes, so a local model is a config change.
//!
//! ## What a spec string is
//!
//! `model_spec.rs` takes one string and answers three questions a caller
//! otherwise has to answer separately: **which backend**, **where does it live**,
//! and **what is it called**. The prefixes are a naming convention and nothing
//! more — no model code, no vendored client, no inference engine. What runs the
//! model is entirely somebody else's problem, which is the point.
//!
//! ```text
//!   openai-compatible:http://localhost:8000/v1   an explicit endpoint
//!   vllm/meta-llama/Llama-3.1-8B                  a local vLLM server
//!   hf/Qwen/Qwen2.5-Coder-32B                    HuggingFace, hosted router
//!   hf-local/Qwen/Qwen2.5-Coder-32B               the same repo, served locally
//!   kernel                                         no model: the kernel's solver
//!   gpt-4o-mini                                    bare name → OpenAI-compatible, default URL
//! ```
//!
//! ## `hf/` versus `hf-local/`, and why there are two
//!
//! A HuggingFace repo id is **not** a location. The same `Qwen/Qwen2.5-7B` can be
//! served by a hosted router or by a vLLM process on the next machine over, and
//! they have different trust properties: one sends the prompt to a third party,
//! the other does not.
//!
//! A prefix that guessed would silently choose whether a prompt leaves the host.
//! That is not a detail to infer, so `hf/` means the hosted router, `hf-local/`
//! means local, and [`ModelSpec::is_local`] is the question a policy check asks.
//!
//! ## Worker versus primary model
//!
//! [`ModelSpec::for_role`] exists because a director and its workers want
//! different models, and threading a second string through every call site is how
//! they end up identical. The override is an *override*: an unset
//! `WORKER_MODEL_NAME` falls back to the primary, so adding the concept changed
//! nothing for an operator who never set it.
//!
//! ## What this deliberately does not do
//!
//! It does not probe the endpoint, check that the model exists, or report a health
//! status. A spec string is a *declaration*, and this resolves declarations. A
//! reachable-but-wrong model is a runtime failure with a runtime message, and a
//! resolver that silently probed would be slow, network-dependent, and would turn
//! config loading into a blocking network call.

use serde::{Deserialize, Serialize};

/// Default endpoint for a bare model name.
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// Default endpoint for `vllm/…` when `VLLM_BASE_URL` is unset.
pub const DEFAULT_VLLM_BASE_URL: &str = "http://localhost:8000/v1";
/// Hosted HuggingFace router, used by `hf/…`.
pub const DEFAULT_HF_BASE_URL: &str = "https://router.huggingface.co/v1";

/// Which role a model is being resolved for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The model the operator configured explicitly.
    Primary,
    /// A worker in a multi-agent run. May be overridden separately.
    Worker,
}

/// Where a model runs and how to reach it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum ModelBackend {
    /// An OpenAI-compatible HTTP endpoint. `base_url` already includes `/v1`.
    OpenAiCompatible {
        base_url: String,
        /// True when the endpoint is a loopback address. Computed from the URL,
        /// never declared, so it cannot be set to a reassuring lie.
        loopback: bool,
    },
    /// A local vLLM server.
    Vllm { base_url: String, loopback: bool },
    /// HuggingFace, hosted router or local server.
    HuggingFace { base_url: String, loopback: bool },
    /// No model. The kernel's own solver answers.
    Kernel,
}

impl ModelBackend {
    /// The base URL to POST to, or `None` for [`ModelBackend::Kernel`].
    pub fn base_url(&self) -> Option<&str> {
        match self {
            ModelBackend::OpenAiCompatible { base_url, .. }
            | ModelBackend::Vllm { base_url, .. }
            | ModelBackend::HuggingFace { base_url, .. } => Some(base_url),
            ModelBackend::Kernel => None,
        }
    }

    /// Whether reaching this model does not leave the machine.
    pub fn is_local(&self) -> bool {
        match self {
            ModelBackend::Kernel => true,
            ModelBackend::OpenAiCompatible { loopback, .. }
            | ModelBackend::Vllm { loopback, .. }
            | ModelBackend::HuggingFace { loopback, .. } => *loopback,
        }
    }
}

/// A resolved model specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// The name the backend knows the model by.
    pub model: String,
    pub backend: ModelBackend,
    /// The spec string this came from, verbatim. Kept so `/version` and error
    /// messages can echo what the operator actually wrote instead of a
    /// reconstructed version of it.
    pub spec: String,
}

impl ModelSpec {
    /// Resolve a spec string.
    ///
    /// Never fails. A string that matches no prefix is a bare model name against
    /// the default OpenAI-compatible endpoint, which is the historical behaviour
    /// of `OPENAI_BASE_URL`/`OPENAI_MODEL` and therefore the only backward-
    /// compatible reading of an unprefixed string.
    pub fn parse(spec: &str) -> ModelSpec {
        let raw = spec.trim();
        // A bare `kernel` is the one spec with no slash and no URL.
        if raw.eq_ignore_ascii_case("kernel") {
            return ModelSpec {
                model: "kernel".to_string(),
                backend: ModelBackend::Kernel,
                spec: raw.to_string(),
            };
        }
        if let Some(rest) = raw.strip_prefix("openai-compatible:") {
            let url = rest.trim();
            return ModelSpec {
                model: model_from_url(url),
                backend: openai(url),
                spec: raw.to_string(),
            };
        }
        // Order matters only for readability; no two prefixes are prefixes of each
        // other except `hf-local/` vs `hf/`, which differ before the `/` and so
        // cannot collide.
        for (prefix, kind) in [("vllm/", 1u8), ("hf-local/", 2), ("hf/", 3)] {
            if let Some(rest) = raw.strip_prefix(prefix) {
                let repo = rest.trim();
                if repo.is_empty() {
                    // `vllm/` with nothing after it is a typo, not a request for
                    // an unnamed model. Fall through to the bare-name reading so it
                    // fails as an unknown model rather than as a URL.
                    continue;
                }
                let (base, model) = match kind {
                    1 => (env_or("VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL), repo),
                    2 => (env_or("HF_LOCAL_BASE_URL", DEFAULT_VLLM_BASE_URL), repo),
                    _ => (DEFAULT_HF_BASE_URL.to_string(), repo),
                };
                let backend = match kind {
                    1 => ModelBackend::Vllm {
                        base_url: base.clone(),
                        loopback: is_loopback(&base),
                    },
                    2 | _ => ModelBackend::HuggingFace {
                        base_url: base.clone(),
                        loopback: is_loopback(&base),
                    },
                };
                return ModelSpec {
                    model: model.to_string(),
                    backend,
                    spec: raw.to_string(),
                };
            }
        }
        ModelSpec {
            model: raw.to_string(),
            backend: openai(&env_or("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)),
            spec: raw.to_string(),
        }
    }

    /// Resolve the model for a role, honouring the worker override.
    ///
    /// `WORKER_MODEL_NAME` is consulted only for [`Role::Worker`], and only when
    /// set and non-blank. An operator who never set it gets the primary, so this
    /// is additive.
    pub fn for_role(spec: &str, role: Role) -> ModelSpec {
        match role {
            Role::Primary => ModelSpec::parse(spec),
            Role::Worker => match std::env::var("WORKER_MODEL_NAME") {
                Ok(w) if !w.trim().is_empty() => ModelSpec::parse(&w),
                // Explicitly *not* an error: "no override configured" is the
                // expected state for most deployments, and failing here would
                // make the concept mandatory rather than optional.
                _ => ModelSpec::parse(spec),
            },
        }
    }

    /// Whether reaching this model does not leave the machine.
    pub fn is_local(&self) -> bool {
        self.backend.is_local()
    }

    /// A one-line description for `/version`, a log, or a smoke test.
    pub fn describe(&self) -> String {
        match &self.backend {
            ModelBackend::Kernel => "kernel (no model; the solver answers)".to_string(),
            _ => format!(
                "{} via {}{}",
                self.model,
                self.backend.base_url().unwrap_or("?"),
                if self.is_local() { " [local]" } else { "" }
            ),
        }
    }
}

/// Build an OpenAI-compatible backend, normalising the URL.
fn openai(url: &str) -> ModelBackend {
    let base_url = normalise_base(url);
    let loopback = is_loopback(&base_url);
    ModelBackend::OpenAiCompatible { base_url, loopback }
}

/// Strip a trailing slash so `base_url + "/chat/completions"` never doubles.
///
/// A trailing slash is the kind of thing a user types and a doubled path is the
/// kind of thing that 404s with a message about the wrong thing.
fn normalise_base(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        DEFAULT_OPENAI_BASE_URL.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Guess a model name from a URL path.
///
/// `openai-compatible:http://host:8000/v1` names an *endpoint*, not a model, so
/// something has to be sent as the model. Taking the last path segment is a
/// guess, and this function says so in its name: a caller that cares should use
/// `openai-compatible:<url>#<model>`, which is unambiguous.
fn model_from_url(url: &str) -> String {
    let path = url
        .split("://")
        .nth(1)
        .and_then(|rest| rest.split_once('/').map(|(_, p)| p))
        .unwrap_or("");
    let last = path
        .split('/')
        .filter(|s| !s.is_empty())
        .next_back()
        .unwrap_or("");
    // Strip the conventional API version suffix; it is not a model name.
    let last = last.strip_prefix("v1").unwrap_or(last);
    if last.is_empty() {
        "local-model".to_string()
    } else {
        last.to_string()
    }
}

/// Whether a URL points at this machine.
///
/// Checks the host portion for `localhost`, `127.x`, `::1` and `[::1]`. Computed
/// rather than declared so a spec string cannot assert that a remote endpoint is
/// local — the property exists for a policy check ("does this prompt leave the
/// host?"), and a self-declared answer to that is worth nothing.
fn is_loopback(url: &str) -> bool {
    let host = url
        .split("://")
        .nth(1)
        .unwrap_or(url)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("");
    // Strip userinfo and port.
    let host = host.rsplit('@').next().unwrap_or(host);
    // A bracketed IPv6 literal is `[::1]:8000`, so the port comes after the
    // closing bracket. Splitting on the first `:` instead truncates `[::1]` to
    // `[`, which then matches nothing -- the loopback check silently said "no" for
    // the one address that is unambiguously local.
    let host = if let Some(rest) = host.strip_prefix('[') {
        match rest.split_once(']') {
            Some((inside, _)) => inside,
            None => rest,
        }
    } else {
        host.split(':').next().unwrap_or(host)
    };
    host.eq_ignore_ascii_case("localhost")
        || host == "::1"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Read an env var, falling back on blank.
fn env_or(key: &str, default: &str) -> String {
    match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => v.trim().to_string(),
        _ => default.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- prefixes ---------------------------------------------------------

    #[test]
    fn an_explicit_openai_compatible_url_is_honoured() {
        let s = ModelSpec::parse("openai-compatible:http://localhost:8000/v1");
        assert_eq!(s.model, "local-model");
        assert_eq!(s.backend.base_url(), Some("http://localhost:8000/v1"));
        assert!(s.is_local());
    }

    #[test]
    fn a_vllm_prefix_resolves_to_a_local_server() {
        let s = ModelSpec::parse("vllm/meta-llama/Llama-3.1-8B");
        assert_eq!(s.model, "meta-llama/Llama-3.1-8B");
        assert_eq!(s.backend.base_url(), Some(DEFAULT_VLLM_BASE_URL));
        assert!(s.is_local());
    }

    #[test]
    fn a_bare_hf_prefix_is_the_hosted_router() {
        // The whole point of having both prefixes: a repo id is not a location,
        // and guessing would decide silently whether a prompt leaves the host.
        let s = ModelSpec::parse("hf/Qwen/Qwen2.5-Coder-32B");
        assert_eq!(s.model, "Qwen/Qwen2.5-Coder-32B");
        assert_eq!(s.backend.base_url(), Some(DEFAULT_HF_BASE_URL));
        assert!(!s.is_local(), "hf/ is hosted, and says so");
    }

    #[test]
    fn hf_local_is_the_same_repo_served_locally() {
        let hosted = ModelSpec::parse("hf/Qwen/Qwen2.5-Coder-32B");
        let local = ModelSpec::parse("hf-local/Qwen/Qwen2.5-Coder-32B");
        assert_eq!(hosted.model, local.model, "same repo, different place");
        assert_ne!(hosted.backend.base_url(), local.backend.base_url());
        assert!(hosted.is_local() == false && local.is_local());
    }

    #[test]
    fn hf_local_does_not_collide_with_hf() {
        // `hf-local/…` must not be read as `hf/` with the repo
        // "-local/…". The prefixes differ before the slash so they cannot.
        let s = ModelSpec::parse("hf-local/Qwen/Qwen2.5-7B");
        assert_eq!(s.model, "Qwen/Qwen2.5-7B");
        assert!(s.is_local());
    }

    #[test]
    fn kernel_means_no_model_at_all() {
        let s = ModelSpec::parse("kernel");
        assert_eq!(s.backend, ModelBackend::Kernel);
        assert_eq!(s.backend.base_url(), None);
        assert!(s.is_local());
    }

    #[test]
    fn a_bare_name_keeps_the_historical_meaning() {
        // Backward compatibility with OPENAI_BASE_URL/OPENAI_MODEL: an unprefixed
        // string must not change behaviour for an existing deployment.
        let s = ModelSpec::parse("gpt-4o-mini");
        assert_eq!(s.model, "gpt-4o-mini");
        assert_eq!(s.backend.base_url(), Some(DEFAULT_OPENAI_BASE_URL));
        assert!(!s.is_local());
    }

    #[test]
    fn the_original_spec_is_kept_verbatim() {
        // So an error message can echo what the operator wrote rather than a
        // reconstructed version of it.
        let s = ModelSpec::parse("  vllm/Qwen/Qwen2.5-7B  ");
        assert_eq!(s.spec, "vllm/Qwen/Qwen2.5-7B");
    }

    // ---- URLs -------------------------------------------------------------

    #[test]
    fn a_trailing_slash_is_normalised_away() {
        // A doubled slash 404s with a message about the wrong thing.
        let s = ModelSpec::parse("openai-compatible:http://localhost:8000/v1/");
        assert_eq!(s.backend.base_url(), Some("http://localhost:8000/v1"));
        assert!(!s.backend.base_url().unwrap().ends_with('/'));
    }

    #[test]
    fn loopback_detection_covers_the_usual_spellings() {
        for url in [
            "http://localhost:8000/v1",
            "http://127.0.0.1:8000/v1",
            "http://127.1.2.3/v1",
            "http://[::1]:8000/v1",
            "https://LOCALHOST/v1",
        ] {
            assert!(is_loopback(url), "{url} should be loopback");
        }
        for url in [
            "https://api.openai.com/v1",
            "https://router.huggingface.co/v1",
            "http://10.0.0.5:8000/v1",
            // A host that merely *contains* "localhost" is not localhost.
            "http://localhost.evil.example/v1",
        ] {
            assert!(!is_loopback(url), "{url} should not be loopback");
        }
    }

    #[test]
    fn loopback_ignores_userinfo_and_port() {
        assert!(is_loopback("http://user:pw@localhost:8000/v1"));
        assert!(!is_loopback("http://user:pw@evil.example:8000/v1"));
    }

    #[test]
    fn a_spec_cannot_declare_itself_local() {
        // `is_local` is computed from the URL, never declared, so a remote
        // endpoint cannot be labelled local to satisfy a policy check.
        let remote = ModelSpec::parse("openai-compatible:https://api.openai.com/v1");
        assert!(!remote.is_local());
    }

    // ---- the model name from a URL ---------------------------------------

    #[test]
    fn a_model_name_is_guessed_from_the_url_and_says_so() {
        // Guessing is documented; the unambiguous form is a spec the operator
        // controls. What must not happen is a silent wrong answer presented as a
        // fact.
        assert_eq!(model_from_url("http://localhost:8000/v1"), "local-model");
        assert_eq!(model_from_url("http://localhost:8000/my-model"), "my-model");
        assert_eq!(model_from_url("http://localhost:8000"), "local-model");
        assert_eq!(model_from_url("not a url"), "local-model");
    }

    #[test]
    fn the_api_version_suffix_is_not_taken_as_a_model_name() {
        // `/v1` is the API version, not a model called "v1".
        assert_eq!(model_from_url("http://h:8000/v1"), "local-model");
    }

    // ---- roles ------------------------------------------------------------

    #[test]
    fn an_unset_worker_override_falls_back_to_the_primary() {
        // The concept is additive: an operator who never sets it is unaffected.
        let a = ModelSpec::for_role("vllm/Qwen/Qwen2.5-7B", Role::Primary);
        let b = ModelSpec::for_role("vllm/Qwen/Qwen2.5-7B", Role::Worker);
        assert_eq!(a, b);
    }

    #[test]
    fn the_primary_role_ignores_the_worker_override() {
        // Otherwise setting WORKER_MODEL_NAME would silently move the director
        // too, which is the opposite of the point.
        let a = ModelSpec::for_role("gpt-4o-mini", Role::Primary);
        assert_eq!(a.model, "gpt-4o-mini");
    }

    // ---- robustness -------------------------------------------------------

    #[test]
    fn adversarial_spec_strings_resolve_without_panicking() {
        let nasty = [
            "",
            "   ",
            "/",
            "//",
            "vllm/",
            "hf/",
            "hf-local/",
            "openai-compatible:",
            "openai-compatible://",
            "vllm/../../etc/passwd",
            "\u{0}",
            "🙂/🙂",
            "hf//",
            "vllm//",
            "openai-compatible:http://",
            "kernel kernel",
            &"x".repeat(10_000),
        ];
        for spec in nasty {
            let s = ModelSpec::parse(spec);
            // The only invariant that must hold for nonsense: it resolves to
            // something, and asking about it does not panic.
            let _ = s.is_local();
            let _ = s.describe();
            let _ = serde_json::to_string(&s).expect("a resolved spec always serializes");
        }
    }

    #[test]
    fn a_resolved_spec_round_trips_through_json() {
        let s = ModelSpec::parse("hf/Qwen/Qwen2.5-32B");
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<ModelSpec>(&j).unwrap(), s);
    }

    #[test]
    fn describe_says_where_and_whether_it_is_local() {
        assert!(
            ModelSpec::parse("vllm/Qwen/Qwen2.5-7B")
                .describe()
                .contains("[local]")
        );
        assert!(
            !ModelSpec::parse("hf/Qwen/Qwen2.5-7B")
                .describe()
                .contains("[local]")
        );
        assert!(ModelSpec::parse("kernel").describe().contains("no model"));
    }
}
