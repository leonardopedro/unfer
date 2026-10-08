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
//!
//! # Choosing a backbone (e.g. ModernBERT vs EmbeddingGemma)
//!
//! The choice is **host-side**. The engine never sees a string, let alone an
//! image or a waveform, so whatever modality a multimodal backbone maps into
//! its shared space arrives here as one vector of `embedding_dim`. What this
//! seam owes the engine is the two things a choice actually changes:
//!
//! * **identity** — which model, at which width, under which instruction —
//!   folded into [`version_string`], so statistics gathered under one choice
//!   are refused under another (A4) rather than silently mixed;
//! * **width** — `embedding_dim` must equal the width the host really feeds.
//!
//! Three options are recognized because each one quietly changes the input
//! space if it is left out of the identity:
//!
//! * **Matryoshka truncation** — `#truncate=N`. An MRL-trained model exposes
//!   the leading `N` coordinates of the same vector as a usable embedding.
//!   Truncation happens *here* and is re-normalized afterwards, because a
//!   truncated vector is no longer unit length and a cosine over unnormalized
//!   vectors is not a cosine. `N` must equal the engine's `embedding_dim`, so
//!   the width and the claim cannot drift apart.
//! * **Instruction prefix** — `#instruction=TEXT`. The host applies the prefix
//!   to the text before embedding; it is recorded in the identity because
//!   prefixed and unprefixed vectors do not sit in the same space.
//! * **Dual-stream concat** — [`concat_streams`]. Two backbones' vectors are
//!   concatenated host-side into `2d`; the engine only needs
//!   `embedding_dim = 2d`, and the identity names the pair.
//!
//! None of the above is a measurement. This module makes no latency, quality,
//! parameter-count or benchmark claim about any model — it is deliberately
//! generic over the C7 model-spec vocabulary rather than pinned to one model,
//! and the choice of *which* model to run is the host's, not the engine's.

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
    /// Matryoshka truncation width, when the host feeds a prefix of a wider
    /// vector. Part of identity: `256` and `768` of the same weights are two
    /// different input spaces, so they must not share statistics.
    pub truncate_dim: Option<usize>,
    /// Instruction prefix applied host-side before embedding. Part of identity
    /// for the same reason: a prefixed query and a bare one do not land in the
    /// same place.
    pub instruction: Option<String>,
}

/// Parse the `#`-fragment of an embed spec: `truncate=N` and `instruction=TEXT`.
///
/// Options ride after a `#` so neither a base URL's own `?query` nor a model
/// id's `/` and `-` can be mistaken for one; a URL fragment is not sent to an
/// endpoint in any case, so splitting on the first `#` cannot truncate a
/// working base URL. `instruction` is free text that may itself contain `&`,
/// so it is allowed to be the last option and absorbs the rest of the
/// fragment.
fn parse_options(fragment: Option<&str>) -> Result<(Option<usize>, Option<String>), String> {
    let Some(frag) = fragment else {
        return Ok((None, None));
    };
    let mut truncate_dim = None;
    let mut instruction = None;
    let mut pos = 0;
    while pos < frag.len() {
        let end = frag[pos..].find('&').map(|i| pos + i).unwrap_or(frag.len());
        let token = &frag[pos..end];
        if let Some(v) = token.strip_prefix("truncate=") {
            let n: usize = v.parse().map_err(|_| {
                format!("embed option 'truncate' must be a positive integer, got {v:?}")
            })?;
            if n == 0 {
                return Err("embed option 'truncate' must be a positive integer".to_string());
            }
            truncate_dim = Some(n);
        } else if token.starts_with("instruction=") {
            let value = &frag[pos + "instruction=".len()..];
            if value.is_empty() {
                return Err("embed option 'instruction' has no text after '='".to_string());
            }
            instruction = Some(value.to_string());
            break;
        } else if !token.is_empty() {
            return Err(format!(
                "unknown embed spec option {token:?}; the known options are \
                 'truncate=N' and 'instruction=TEXT'"
            ));
        }
        pos = end + 1;
    }
    Ok((truncate_dim, instruction))
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

    // Options ride after `#`, so a model id or a base URL is never mangled.
    let (spec_part, fragment) = match rest.split_once('#') {
        Some((s, f)) => (s, Some(f)),
        None => (rest, None),
    };
    if spec_part.trim().is_empty() {
        return Err(format!(
            "embedding model spec has no model before its options: {raw:?}"
        ));
    }
    let (truncate_dim, instruction) = parse_options(fragment)?;

    if let Some(url) = spec_part.strip_prefix("openai-compatible:") {
        return Ok(EmbedSpec {
            backend: "openai-compatible".to_string(),
            model: spec_part.to_string(),
            base_url: Some(url.to_string()),
            truncate_dim,
            instruction,
        });
    }
    for (prefix, backend) in [
        ("vllm/", "vllm"),
        ("hf-local/", "hf-local"),
        ("hf/", "hf"),
        ("kernel/", "kernel"),
    ] {
        if let Some(model) = spec_part.strip_prefix(prefix) {
            return Ok(EmbedSpec {
                backend: backend.to_string(),
                model: model.to_string(),
                base_url: None,
                truncate_dim,
                instruction,
            });
        }
    }
    // A bare name is a local model id, matching C7's own rule for chat specs.
    Ok(EmbedSpec {
        backend: "local".to_string(),
        model: spec_part.to_string(),
        base_url: None,
        truncate_dim,
        instruction,
    })
}

/// Truncate each vector to its leading `dim` coordinates (Matryoshka / MRL).
///
/// Refuses a vector shorter than `dim` rather than padding it: a zero-padded
/// embedding is a different vector, not a smaller one, and the resulting class
/// mean would be wrong in a way nothing downstream can detect. Normalization
/// is deliberately *not* done here — see [`l2_normalize`] and [`feed`], which
/// truncate first and normalize second so the norm is of the vector that is
/// actually used.
pub fn matryoshka_truncate(
    rows: Vec<DVector<f64>>,
    dim: usize,
) -> Result<Vec<DVector<f64>>, String> {
    if dim == 0 {
        return Err("truncation width must be a positive integer".to_string());
    }
    rows.into_iter()
        .enumerate()
        .map(|(i, v)| {
            if v.len() < dim {
                return Err(format!(
                    "embedding {i} has length {} but the truncation width is {dim}; \
                     truncation cannot lengthen a vector",
                    v.len()
                ));
            }
            if v.len() == dim {
                return Ok(v);
            }
            Ok(DVector::from_column_slice(&v.as_slice()[..dim]))
        })
        .collect()
}

/// Concatenate two streams' embeddings pairwise (the dual-stream option).
///
/// A pairing, not a cross product: stream A's `i`-th vector goes with stream
/// B's `i`-th vector, because the two streams are two views of the same items.
/// A length mismatch is refused — silently zipping to the shorter length would
/// drop items and the caller would rank a list shorter than the one on screen.
/// The result's width is `d_a + d_b`, which is what `embedding_dim` must be set
/// to.
pub fn concat_streams(
    a: Vec<DVector<f64>>,
    b: Vec<DVector<f64>>,
) -> Result<Vec<DVector<f64>>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "dual-stream concat has {} vectors in the first stream and {} in the second",
            a.len(),
            b.len()
        ));
    }
    a.into_iter()
        .zip(b)
        .map(|(x, y)| {
            let mut vals = Vec::with_capacity(x.len() + y.len());
            vals.extend_from_slice(x.as_slice());
            vals.extend_from_slice(y.as_slice());
            Ok(DVector::from_row_slice(&vals))
        })
        .collect()
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
    // The truncation claim and the engine's width must agree. A spec that
    // truncates to N while the engine is configured for a different width
    // means the identity describes one input space and the statistics are
    // accumulating under another — the exact mixing A4 exists to refuse.
    if let Some(t) = batch.spec.truncate_dim
        && t != dim
    {
        return Err(DecideError::Seam(format!(
            "embed spec claims truncate={t} but the engine's embedding_dim is {dim}; \
             a Matryoshka truncation must match the width the engine accumulates in"
        )));
    }
    for v in &batch.vectors {
        let ok = match batch.spec.truncate_dim {
            None => v.len() == dim,
            Some(t) => v.len() >= t,
        };
        if !ok {
            return Err(DecideError::EmbeddingDim {
                expected: match batch.spec.truncate_dim {
                    None => dim,
                    Some(t) => t,
                },
                got: v.len(),
            });
        }
    }
    // Truncate before normalizing: a truncated vector's norm is no longer 1,
    // so normalizing first would leave the used vector off the unit sphere.
    let shaped = match batch.spec.truncate_dim {
        Some(t) => matryoshka_truncate(batch.vectors.clone(), t).map_err(DecideError::Seam)?,
        None => batch.vectors.clone(),
    };
    let normalized = l2_normalize(shaped).map_err(DecideError::Seam)?;
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
    let mut s = match &spec.base_url {
        Some(u) => format!("{}:{}@{}", spec.backend, spec.model, u),
        None => format!("{}:{}", spec.backend, spec.model),
    };
    if let Some(t) = spec.truncate_dim {
        s.push_str(&format!("#truncate={t}"));
    }
    if let Some(i) = &spec.instruction {
        s.push_str(&format!("#instruction={i}"));
    }
    s
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
                truncate_dim: None,
                instruction: None,
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

    // ── Backbone choice: ModernBERT vs EmbeddingGemma-style options ──────

    /// The two choices are two spec strings and two widths — nothing in the
    /// engine knows or should know which model produced a vector.
    #[test]
    fn choosing_a_backbone_is_expressed_by_the_spec_and_the_width() {
        let a = parse_embed_spec("embed:hf/answerdotai/ModernBERT-base").unwrap();
        let b = parse_embed_spec(
            "embed:hf/google/embeddinggemma-2#truncate=256&instruction=task: classification",
        )
        .unwrap();
        assert_eq!(a.truncate_dim, None);
        assert_eq!(a.instruction, None);
        assert_eq!(b.truncate_dim, Some(256));
        assert_eq!(
            b.instruction.as_deref(),
            Some("task: classification"),
            "the instruction is free text and must arrive intact"
        );
        // Different models, different widths, different instructions: three
        // independent reasons the two engines must not share statistics.
        assert_ne!(version_string(&a), version_string(&b));
    }

    #[test]
    fn a_truncation_claim_is_part_of_the_identity_not_a_free_lunch() {
        let full = version_of("embed:hf/m");
        let trunc = version_of("embed:hf/m#truncate=256");
        assert_ne!(
            full, trunc,
            "256 and the full width of the same weights are different input spaces"
        );
        let prefixed = version_of("embed:hf/m#instruction=task: classify");
        assert_ne!(
            full, prefixed,
            "prefixed and bare vectors do not sit in the same space"
        );
    }

    /// Truncation then normalization, in that order — the ordering is the
    /// point: normalizing first would leave the vector that is actually used
    /// off the unit sphere.
    #[test]
    fn matryoshka_truncation_runs_before_normalization() {
        let wide = DVector::from_row_slice(&[3.0, 4.0, 12.0, 13.0]);
        let trunc = matryoshka_truncate(vec![wide.clone()], 2).unwrap();
        assert_eq!(trunc[0], DVector::from_row_slice(&[3.0, 4.0]));
        let normed = l2_normalize(trunc).unwrap();
        let expect = DVector::from_row_slice(&[0.6, 0.8]);
        assert!((&normed[0] - &expect).norm() < 1e-12);
        assert!((normed[0].norm() - 1.0).abs() < 1e-12);
        // The whole vector, untouched, is what a no-truncation feed sees.
        assert_eq!(
            matryoshka_truncate(vec![wide], 4).unwrap()[0],
            DVector::from_row_slice(&[3.0, 4.0, 12.0, 13.0])
        );
    }

    #[test]
    fn a_truncation_wider_than_the_vector_is_refused_not_padded() {
        let e = matryoshka_truncate(vec![DVector::from_row_slice(&[1.0, 2.0])], 4).unwrap_err();
        assert!(e.contains("lengthen"), "{e}");
        assert!(matryoshka_truncate(vec![DVector::zeros(4)], 0).is_err());
    }

    #[test]
    fn feeding_a_truncated_backbone_truncates_at_the_seam() {
        // The host sends the full-width vector; the seam truncates to the
        // configured width and normalizes, so `embedding_dim` stays honest.
        let mut c = core(&version_of("embed:hf/m#truncate=2"), 2);
        let spec = parse_embed_spec("embed:hf/m#truncate=2").unwrap();
        let n = feed(
            &mut c,
            &Batch {
                spec,
                vectors: vec![DVector::from_row_slice(&[3.0, 4.0, 12.0, 13.0])],
                labels: vec!["a".to_string()],
            },
        )
        .unwrap();
        assert_eq!(n, 1);
        assert_eq!(c.samples(), 1);
    }

    /// The identity and the accumulation width must agree. One without the
    /// other means statistics gathered under one input space would be read as
    /// another's.
    #[test]
    fn a_truncation_claim_that_disagrees_with_the_engine_width_is_refused() {
        let mut c = core(&version_of("embed:hf/m#truncate=2"), 4);
        let spec = parse_embed_spec("embed:hf/m#truncate=2").unwrap();
        let err = feed(
            &mut c,
            &Batch {
                spec,
                vectors: vec![DVector::from_element(8, 1.0)],
                labels: vec!["a".to_string()],
            },
        )
        .unwrap_err();
        match err {
            DecideError::Seam(msg) => {
                assert!(msg.contains("truncate=2"), "{msg}");
                assert!(msg.contains("embedding_dim is 4"), "{msg}");
            }
            other => panic!("expected a seam refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_truncated_feed_that_is_still_too_short_is_refused() {
        let mut c = core(&version_of("embed:hf/m#truncate=2"), 2);
        let spec = parse_embed_spec("embed:hf/m#truncate=2").unwrap();
        let err = feed(
            &mut c,
            &Batch {
                spec,
                vectors: vec![DVector::from_row_slice(&[1.0])],
                labels: vec!["a".to_string()],
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            DecideError::EmbeddingDim {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn an_unknown_option_is_refused_rather_than_silently_ignored() {
        let e = parse_embed_spec("embed:hf/m#dim=256").unwrap_err();
        assert!(e.contains("unknown embed spec option"), "{e}");
        assert!(parse_embed_spec("embed:hf/m#truncate=abc").is_err());
        assert!(parse_embed_spec("embed:hf/m#truncate=0").is_err());
        assert!(parse_embed_spec("embed:hf/m#instruction=").is_err());
        assert!(parse_embed_spec("embed:#truncate=256").is_err());
    }

    /// A base URL keeps its own `?query`; only `#` starts options, and a URL
    /// fragment is not sent to an endpoint, so nothing reachable is lost.
    #[test]
    fn a_query_string_is_not_mistaken_for_options() {
        let s = parse_embed_spec("embed:openai-compatible:http://h/v1?key=k#truncate=128").unwrap();
        assert_eq!(s.base_url.as_deref(), Some("http://h/v1?key=k"));
        assert_eq!(s.truncate_dim, Some(128));
        let v = version_string(&s);
        assert!(v.contains("?key=k"), "the query survived: {v}");
        assert!(v.ends_with("#truncate=128"), "the option survived: {v}");
    }

    /// Dual stream: the two views pair up item-by-item, and a length mismatch
    /// is refused rather than zipped to the shorter side.
    #[test]
    fn dual_stream_concat_pairs_by_index_and_refuses_a_length_mismatch() {
        let a = vec![
            DVector::from_row_slice(&[1.0, 2.0]),
            DVector::from_row_slice(&[3.0, 4.0]),
        ];
        let b = vec![
            DVector::from_row_slice(&[10.0, 20.0]),
            DVector::from_row_slice(&[30.0, 40.0]),
        ];
        let fused = concat_streams(a, b.clone()).unwrap();
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0], DVector::from_row_slice(&[1.0, 2.0, 10.0, 20.0]));
        assert_eq!(fused[1], DVector::from_row_slice(&[3.0, 4.0, 30.0, 40.0]));
        let e = concat_streams(vec![DVector::zeros(2)], b).unwrap_err();
        assert!(e.contains("first stream"), "{e}");
    }

    /// The fused width is what `embedding_dim` must be set to — asserted
    /// through `feed` rather than by inspecting a number in isolation.
    #[test]
    fn a_dual_stream_feed_accumulates_at_the_concatenated_width() {
        let fused = concat_streams(
            vec![DVector::from_row_slice(&[3.0, 4.0])],
            vec![DVector::from_row_slice(&[12.0, 13.0])],
        )
        .unwrap();
        assert_eq!(fused[0].len(), 4);
        let mut c = core(&version_of("embed:hf/dual-stream"), 4);
        let spec = parse_embed_spec("embed:hf/dual-stream").unwrap();
        let n = feed(
            &mut c,
            &Batch {
                spec,
                vectors: fused,
                labels: vec!["a".to_string()],
            },
        )
        .unwrap();
        assert_eq!(n, 1);
    }
}
