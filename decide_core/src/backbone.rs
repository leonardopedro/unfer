//! D7 — the backbone seam: frozen embeddings in, through the existing C7 model
//! spec.
//!
//! # Why this is a seam and not a client
//!
//! The project's agent interface already names models by spec prefix (`vllm/…`,
//! `hf/…`, `openai-compatible:<url>`) and already knows that different roles can
//! use different models (`WORKER_MODEL_NAME`). D7 reuses that vocabulary instead
//! of introducing a second one: an embedding model is named the same way a chat
//! model is, with an `embed:` prefix to say which capability is wanted.
//!
//! # What is actually transported
//!
//! **Vectors, not text.** The engine never sees a string. A host fetches
//! embeddings from its endpoint and hands them over; this module's job is the
//! part that is easy to get wrong and hard to notice:
//!
//! * **the embedding model's identity**, which is part of engine identity, and
//! * **L2 normalization at the seam**, so the corpus, the fixtures and a
//!   production endpoint agree on what a vector means.
//!
//! Audit defect **A4**'s second half is about the identity: statistics
//! accumulated under one embedding model are not comparable under another, so a
//! version change is refused rather than silently re-projected.

use crate::{DecideCore, DecideError};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Prefix marking a model spec as an *embedding* endpoint.
pub const EMBED_PREFIX: &str = "embed:";

/// A parsed embedding model spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedSpec {
    /// The base URL, or `None` for a named local backend.
    pub backend: String,
    /// The model name as the endpoint knows it.
    pub model: String,
    /// The endpoint, when there is one.
    pub base_url: Option<String>,
}

/// Parse `embed:<spec>`, reusing the C7 prefixes underneath.
///
/// `embed:openai-compatible:http://host/v1` and `embed:hf/all-MiniLM-L6-v2` are
/// both accepted, and anything without the prefix is refused — an embedding
/// endpoint reached by accident under a chat spec would return token ids, and
/// the resulting error would be baffling rather than immediate.
pub fn parse_embed_spec(raw: &str) -> Result<EmbedSpec, String> {
    let rest = raw
        .strip_prefix(EMBED_PREFIX)
        .ok_or_else(|| {
            format!(
                "embedding model spec must start with the embed: prefix; got {raw:?}. \
                 A chat spec pointed at an embedding endpoint would return token ids."
            )
        })?
        .trim();
    if rest.is_empty() {
        return Err("embedding model spec has no model after the prefix".to_string());
    }

    if let Some(url) = rest.strip_prefix("openai-compatible:") {
        return Ok(EmbedSpec {
            backend: "openai-compatible".to_string(),
            model: rest.to_string(),
            base_url: Some(url.to_string()),
        });
    }
    for (prefix, backend) in [
        ("vllm/", "vllm"),
        ("hf-local/", "hf-local"),
        ("hf/", "hf"),
        ("kernel/", "kernel"),
    ] {
        if let Some(model) = rest.strip_prefix(prefix) {
            return Ok(EmbedSpec {
                backend: backend.to_string(),
                model: model.to_string(),
                base_url: None,
            });
        }
    }
    // A bare name is a local model id, matching C7's own rule for chat specs.
    Ok(EmbedSpec {
        backend: "local".to_string(),
        model: rest.to_string(),
        base_url: None,
    })
}

/// Normalize a batch of embeddings to unit length at the seam.
///
/// Called on ingest rather than trusted from the endpoint, because not every
/// endpoint normalizes and a corpus that mixes normalized and unnormalized
/// vectors produces a similarity that means nothing. A zero vector is a caller
/// error and is refused rather than left as a NaN.
pub fn l2_normalize(rows: Vec<DVector<f64>>) -> Result<Vec<DVector<f64>>, String> {
    let mut out = Vec::with_capacity(rows.len());
    for (i, v) in rows.into_iter().enumerate() {
        let n = v.norm();
        if n <= 0.0 || !n.is_finite() {
            return Err(format!(
                "embedding {i} has zero or non-finite norm ({n}); it cannot be normalized"
            ));
        }
        out.push(v / n);
    }
    Ok(out)
}

/// What a host must supply to feed the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub spec: EmbedSpec,
    /// One label per vector. Carried in the batch rather than supplied
    /// separately so a length mismatch is impossible by construction.
    pub labels: Vec<String>,
    pub vectors: Vec<DVector<f64>>,
}

/// Feed a batch through the seam into the engine.
///
/// Four checks, each of which exists because skipping it produces a wrong answer
/// rather than an error:
///
/// 1. `labels` and `vectors` must be the same length — a short label list would
///    otherwise silently train the wrong classes;
/// 2. the embedding model's identity must match the engine's (D7, and audit
///    defect A4);
/// 3. every vector must have the engine's configured dimension;
/// 4. normalization happens here, not in the caller.
pub fn feed(core: &mut DecideCore, batch: &Batch) -> Result<usize, DecideError> {
    if batch.labels.len() != batch.vectors.len() {
        return Err(DecideError::Seam(format!(
            "batch has {} labels but {} vectors",
            batch.labels.len(),
            batch.vectors.len()
        )));
    }
    core.check_embedding_version(&version_string(&batch.spec))?;
    let dim = core.config().embedding_dim;
    for v in &batch.vectors {
        if v.len() != dim {
            return Err(DecideError::EmbeddingDim {
                expected: dim,
                got: v.len(),
            });
        }
    }
    let normalized = l2_normalize(batch.vectors.clone()).map_err(DecideError::Seam)?;
    let n = normalized.len();
    for (v, label) in normalized.into_iter().zip(batch.labels.iter()) {
        core.observe(label, &v)?;
    }
    Ok(n)
}

/// A stable identity string for an embedding spec.
///
/// Includes the backend as well as the model, because `hf:` and `hf-local:` can
/// resolve to the *same weights* behind different endpoints — and that is a
/// legitimate reason to share statistics, whereas two different models of the
/// same architecture is not.
pub fn version_string(spec: &EmbedSpec) -> String {
    match &spec.base_url {
        Some(u) => format!("{}:{}@{}", spec.backend, spec.model, u),
        None => format!("{}:{}", spec.backend, spec.model),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DecideConfig, Mode};

    fn core(version: &str, dim: usize) -> DecideCore {
        let mut c = DecideCore::new(DecideConfig {
            embedding_version: version.to_string(),
            embedding_dim: dim,
            projected_dim: 32,
            heads: 1,
            mode: Mode::Ananc,
            lambda: 1.0,
            alpha: 1.0,
        })
        .unwrap();
        for l in ["a", "b", "c"] {
            c.register(l).unwrap();
        }
        c
    }

    #[test]
    fn the_embed_prefix_is_required_so_a_chat_spec_cannot_be_passed_by_accident() {
        let e = parse_embed_spec("openai-compatible:http://x/v1").unwrap_err();
        assert!(
            e.contains("embed:"),
            "the error must name the missing prefix, got: {e}"
        );
    }

    #[test]
    fn the_c7_prefixes_are_reused_rather_than_reinvented() {
        assert_eq!(
            parse_embed_spec("embed:openai-compatible:http://h/v1").unwrap(),
            EmbedSpec {
                backend: "openai-compatible".into(),
                model: "openai-compatible:http://h/v1".into(),
                base_url: Some("http://h/v1".into()),
            }
        );
        assert_eq!(
            parse_embed_spec("embed:hf/all-MiniLM-L6-v2")
                .unwrap()
                .backend,
            "hf"
        );
        assert_eq!(
            parse_embed_spec("embed:hf-local/all-MiniLM-L6-v2")
                .unwrap()
                .backend,
            "hf-local"
        );
        assert_eq!(
            parse_embed_spec("embed:vllm/qwen-embed").unwrap().backend,
            "vllm"
        );
        assert_eq!(
            parse_embed_spec("embed:some-local-model").unwrap().backend,
            "local"
        );
    }

    #[test]
    fn an_empty_spec_is_refused() {
        assert!(parse_embed_spec("embed:").is_err());
        assert!(parse_embed_spec("embed:   ").is_err());
    }

    #[test]
    fn hf_and_hf_local_of_the_same_weights_are_different_engines() {
        // Deliberate: sharing statistics across endpoints is a migration
        // decision, not something to infer from a matching model name.
        let a = parse_embed_spec("embed:hf/m").unwrap();
        let b = parse_embed_spec("embed:hf-local/m").unwrap();
        assert_ne!(version_string(&a), version_string(&b));
    }

    /// The engine version a host must configure for a given spec: derived from
    /// the parser so the test cannot drift from the identity rule.
    fn version_of(spec_text: &str) -> String {
        version_string(&parse_embed_spec(spec_text).unwrap())
    }

    #[test]
    fn a_version_mismatch_is_refused_before_any_vector_is_read() {
        let mut c = core(&version_of("embed:hf/model-a"), 4);
        let spec = parse_embed_spec("embed:hf/model-b").unwrap();
        let err = feed(
            &mut c,
            &Batch {
                spec,
                vectors: vec![DVector::from_element(4, 1.0)],
                labels: vec!["a".to_string()],
            },
        )
        .unwrap_err();
        assert!(matches!(err, DecideError::EmbeddingVersionChanged { .. }));
    }

    #[test]
    fn a_dimension_mismatch_is_refused() {
        let mut c = core(&version_of("embed:hf/m"), 4);
        let spec = parse_embed_spec("embed:hf/m").unwrap();
        let err = feed(
            &mut c,
            &Batch {
                spec,
                vectors: vec![DVector::from_element(3, 1.0)],
                labels: vec!["a".to_string()],
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            DecideError::EmbeddingDim {
                expected: 4,
                got: 3
            }
        ));
    }

    #[test]
    fn a_zero_vector_is_refused_rather_than_becoming_nan() {
        let e = l2_normalize(vec![DVector::zeros(3)]).unwrap_err();
        assert!(e.contains("zero"), "{e}");
    }

    #[test]
    fn normalization_happens_at_the_seam_not_in_the_caller() {
        let mut c = core(&version_of("embed:hf/m"), 3);
        let spec = parse_embed_spec("embed:hf/m").unwrap();
        let n = feed(
            &mut c,
            &Batch {
                spec: spec.clone(),
                // Deliberately unnormalized and wildly different in magnitude.
                vectors: vec![
                    DVector::from_row_slice(&[100.0, 0.0, 0.0]),
                    DVector::from_row_slice(&[0.0, 0.001, 0.0]),
                ],
                labels: vec!["a".to_string(), "b".to_string()],
            },
        )
        .unwrap();
        assert_eq!(n, 2);
    }
}
