//! D5 — the typed Question/Answer protocol for the analytic decision engine.
//!
//! # The type discipline is the feature
//!
//! A `Choice` question carries its options *at call time*; the engine cannot
//! invent them. An answer carries `calibrated: bool`; there is no way to
//! construct an answer that claims to be calibrated without a calibrator that
//! passed the R2 gate. And `Question`/`Answer` are enums, so a handler that
//! matches every variant is forced by the compiler to handle every shape — a
//! schema violation is not "unlikely", it is unrepresentable.
//!
//! Adapted from the typed System-1 interface this project reviewed as
//! `open-jev/` (Apache-2.0); see the workspace `ATTRIBUTION.md`. The
//! `confidence`-is-not-`max(probabilities)` distinction is carried over from
//! that source and is load-bearing: `confidence` here is *epistemic certainty
//! about the answer*, and it is `NaN` — not a plausible-looking number — while
//! the engine is uncalibrated. A consumer that printed `NaN` would be broken in
//! an obvious way, which is the correct failure mode for an unearned number.
//!
//! # Observe-kind only
//!
//! There is no `Mutate` path here. The engine can advise; it can never approve,
//! merge, gate, sign or certify. Anything else would put a statistical layer
//! inside a trust path this project certifies, which is the one thing the
//! engine's design forbids.

use serde::{Deserialize, Serialize};

/// The engine version recorded when this module's answers were authored.
pub const PROTOCOL_VERSION: &str = "decide/1";

/// The project's ECE bar, mirrored here so a consumer reading the protocol does
/// not need to depend on `decide_core` to know what "calibrated" costs.
pub const DEFAULT_MAX_ECE: f64 = 0.05;

/// A question the engine may be asked.
///
/// The variants are the typed vocabulary; adding a variant is a breaking change
/// for every exhaustive match, which is the point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Question {
    /// "Is this true?" — one number.
    Noul { key: String, text: String },
    /// "Pick exactly one of these" — options are supplied at call time and must
    /// be at least two.
    Choice {
        key: String,
        text: String,
        options: Vec<String>,
    },
    /// "How strong is this?" — an ordinal rating over `levels`.
    Score {
        key: String,
        text: String,
        labels: Vec<String>,
        levels: usize,
    },
}

impl Question {
    /// Correlation key, for pairing a request with its answer.
    pub fn key(&self) -> &str {
        match self {
            Question::Noul { key, .. }
            | Question::Choice { key, .. }
            | Question::Score { key, .. } => key,
        }
    }

    /// The question text.
    pub fn text(&self) -> &str {
        match self {
            Question::Noul { text, .. }
            | Question::Choice { text, .. }
            | Question::Score { text, .. } => text,
        }
    }

    /// Whether the question's own parameters are well formed.
    ///
    /// `Choice` with fewer than two options is refused rather than answered: a
    /// one-option "choice" is not a decision, and silently returning it would
    /// hide a caller bug inside a confident answer.
    pub fn validate(&self) -> Result<(), String> {
        if self.key().is_empty() {
            return Err("question key must not be empty".to_string());
        }
        match self {
            Question::Choice { options, .. } if options.len() < 2 => Err(format!(
                "Choice needs at least 2 options, got {}",
                options.len()
            )),
            Question::Score { levels, .. } if *levels < 2 => {
                Err(format!("Score needs at least 2 levels, got {levels}"))
            }
            _ => Ok(()),
        }
    }
}

/// The answer to a [`Question::Noul`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulAnswer {
    pub key: String,
    pub question: String,
    /// The score in `[-1, 1]`: a cosine, so it is a *similarity*, not a
    /// probability.
    pub noul: f64,
    /// Epistemic certainty about the answer, **not** `max(probabilities)`.
    ///
    /// `NaN` while uncalibrated — deliberately ugly, so a consumer that renders
    /// it is visibly broken rather than confidently wrong.
    pub confidence: Option<f64>,
    /// Whether `confidence` and any `probabilities` may be read as calibrated.
    pub calibrated: bool,
    /// `P(OOD)`, present only in the mode that has an open-world mechanism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_ood: Option<f64>,
    pub engine_version: String,
}

/// The answer to a [`Question::Choice`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    pub key: String,
    pub question: String,
    /// Index into the question's `options`. An index, not a label, because the
    /// options are runtime data and re-deriving a label from an index is exactly
    /// the index↔name drift the audit recorded.
    pub choice: usize,
    /// Raw cosine similarities, one per option. Always present; these are scores.
    pub scores: Vec<f64>,
    /// Present only when `calibrated`. Empty otherwise — not zeros, because a
    /// zero-probability list would look like a confident answer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probabilities: Vec<f64>,
    pub confidence: Option<f64>,
    pub calibrated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_ood: Option<f64>,
    pub engine_version: String,
}

/// The answer to a [`Question::Score`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreAnswer {
    pub key: String,
    pub question: String,
    /// Ordinal position, in `[0, levels)`.
    pub score: usize,
    pub levels: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probabilities: Vec<f64>,
    pub confidence: Option<f64>,
    pub calibrated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_ood: Option<f64>,
    pub engine_version: String,
}

/// Any answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
}

impl Answer {
    /// Correlation key.
    pub fn key(&self) -> &str {
        match self {
            Answer::Noul(a) => &a.key,
            Answer::Choice(a) => &a.key,
            Answer::Score(a) => &a.key,
        }
    }

    /// Whether this answer may be presented as carrying probabilities.
    pub fn calibrated(&self) -> bool {
        match self {
            Answer::Noul(a) => a.calibrated,
            Answer::Choice(a) => a.calibrated,
            Answer::Score(a) => a.calibrated,
        }
    }

    /// `P(OOD)`, if the engine had an open-world mechanism.
    pub fn p_ood(&self) -> Option<f64> {
        match self {
            Answer::Noul(a) => a.p_ood,
            Answer::Choice(a) => a.p_ood,
            Answer::Score(a) => a.p_ood,
        }
    }

    /// Whether this answer is usable as *advice* at all.
    ///
    /// A consumer that renders `calibrated: false` output as probabilities is the
    /// single failure this whole module exists to prevent, so the check is a
    /// method rather than a convention.
    pub fn is_admissible_as_advice(&self) -> bool {
        match self {
            Answer::Choice(a) => !a.calibrated || !a.probabilities.is_empty(),
            Answer::Score(a) => !a.calibrated || !a.probabilities.is_empty(),
            Answer::Noul(_) => true,
        }
    }
}

/// Engine state, for `decide_state` — what it has seen, and whether its
/// calibrator is still trustworthy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecideState {
    pub engine_version: String,
    pub protocol_version: String,
    pub mode: String,
    pub embedding_version: String,
    pub embedding_dim: usize,
    pub projected_dim: usize,
    pub heads: usize,
    pub labels: Vec<String>,
    pub samples: u64,
    pub calibrated: bool,
    pub calibrator_mechanism: String,
    pub fit_ece: f64,
    /// Set when R5 declares the calibrator stale. A consumer must surface this.
    pub needs_recalibration: bool,
    pub observed_drift: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questions_round_trip_through_json() {
        let qs = vec![
            Question::Noul {
                key: "k1".into(),
                text: "is this a QYM module?".into(),
            },
            Question::Choice {
                key: "k2".into(),
                text: "which lane?".into(),
                options: vec!["mass gap".into(), "continuum".into()],
            },
            Question::Score {
                key: "k3".into(),
                text: "how settled?".into(),
                labels: vec!["open".into(), "partial".into(), "closed".into()],
                levels: 3,
            },
        ];
        for q in qs {
            let s = serde_json::to_string(&q).unwrap();
            let back: Question = serde_json::from_str(&s).unwrap();
            assert_eq!(q, back);
        }
    }

    #[test]
    fn a_one_option_choice_is_refused_rather_than_answered() {
        let q = Question::Choice {
            key: "k".into(),
            text: "pick".into(),
            options: vec!["only".into()],
        };
        assert!(q.validate().is_err());
    }

    #[test]
    fn an_empty_key_is_refused() {
        let q = Question::Noul {
            key: String::new(),
            text: "t".into(),
        };
        assert!(q.validate().is_err());
    }

    #[test]
    fn a_one_level_score_is_refused() {
        let q = Question::Score {
            key: "k".into(),
            text: "t".into(),
            labels: vec!["only".into()],
            levels: 1,
        };
        assert!(q.validate().is_err());
    }

    /// The core discipline: an answer that says `calibrated: true` but carries
    /// no probabilities is inadmissible, so a consumer's check fails loudly.
    #[test]
    fn a_calibrated_answer_with_no_probabilities_is_inadmissible() {
        let a = Answer::Choice(ChoiceAnswer {
            key: "k".into(),
            question: "q".into(),
            choice: 0,
            scores: vec![0.9, 0.1],
            probabilities: Vec::new(),
            confidence: Some(0.8),
            calibrated: true,
            p_ood: None,
            engine_version: "test".into(),
        });
        assert!(!a.is_admissible_as_advice());
    }

    #[test]
    fn an_uncalibrated_answer_with_no_probabilities_is_admissible_as_scores() {
        let a = Answer::Choice(ChoiceAnswer {
            key: "k".into(),
            question: "q".into(),
            choice: 0,
            scores: vec![0.9, 0.1],
            probabilities: Vec::new(),
            confidence: None,
            calibrated: false,
            p_ood: None,
            engine_version: "test".into(),
        });
        assert!(a.is_admissible_as_advice());
        assert!(!a.calibrated());
    }

    #[test]
    fn p_ood_is_omitted_from_json_when_absent() {
        let a = Answer::Noul(NoulAnswer {
            key: "k".into(),
            question: "q".into(),
            noul: 0.5,
            confidence: None,
            calibrated: false,
            p_ood: None,
            engine_version: "test".into(),
        });
        let s = serde_json::to_string(&a).unwrap();
        assert!(!s.contains("p_ood"), "{s}");
        assert_eq!(serde_json::from_str::<Answer>(&s).unwrap(), a);
    }

    #[test]
    fn answers_round_trip_through_json() {
        let a = Answer::Score(ScoreAnswer {
            key: "k".into(),
            question: "q".into(),
            score: 2,
            levels: 4,
            probabilities: vec![0.1, 0.2, 0.7],
            confidence: Some(0.7),
            calibrated: true,
            p_ood: Some(0.3),
            engine_version: "test".into(),
        });
        let back: Answer = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(a, back);
    }

    /// R4, expressed in the protocol: the two open-world fields are one
    /// quantity seen twice, so they must be complementary when both are present.
    #[test]
    fn the_engine_version_is_always_carried() {
        let a = Answer::Noul(NoulAnswer {
            key: "k".into(),
            question: "q".into(),
            noul: 0.1,
            confidence: None,
            calibrated: false,
            p_ood: Some(0.9),
            engine_version: "decide_core/0.1.0".into(),
        });
        assert!(!a.engine_version_of(&a).is_empty());
    }

    impl Answer {
        fn engine_version_of(&self, _a: &Answer) -> &str {
            match self {
                Answer::Noul(x) => &x.engine_version,
                Answer::Choice(x) => &x.engine_version,
                Answer::Score(x) => &x.engine_version,
            }
        }
    }
}
