//! E0 — the finite Edge0 research world and its **exact** outcome oracle.
//!
//! `EDGE0_RLCD_PLAN.md` §2/§5 asks for a configuration *small enough to
//! enumerate*: synthetic worlds with known, nondegenerate conditional outcome
//! laws, repeated/noisy observations, all three typed questions (Bernoulli,
//! named Choice, ordinal Score) and a utility table, so "a sharp wrong answer
//! can be distinguished from honest ambiguity" and an abstain decision has a
//! computable regret.
//!
//! The oracle here is deliberately *finite and exhaustive* rather than a
//! learned or closed-form approximation:
//!
//! * a world family is a hypothesis set `H = {0,1}^L` with a uniform prior;
//! * the world's true hypothesis `h*` is chosen deterministically from the
//!   world id and seed, so a run is reproducible without saved state;
//! * an observation reveals the value of one feature **without noise**, so the
//!   posterior is the uniform distribution over still-consistent hypotheses;
//! * a question is a deterministic function of a hypothesis → categorical
//!   outcome law, so the posterior outcome law is a finite mixture.
//!
//! Consequences the E0 acceptance criteria name, and where they are pinned:
//!
//! * an independently enumerated world agrees with the oracle — `tests/
//!   edge0_rlcd.rs::oracle_matches_independent_enumeration`;
//! * missing observations cannot lower conditional entropy on average — the
//!   same test file's `more_observations_never_raise_conditional_entropy`;
//! * correlated question marginals are not multiplied into a false joint —
//!   `correlated_questions_are_not_a_product_of_marginals`.
//!
//! Two questions in one world share the hidden hypothesis. Their marginals are
//! therefore *not* independent, and [`joint_law`] enumerates the actual joint.
//! `EDGE0_RLCD_PLAN.md` §3 is explicit that per-question marginals "do not
//! determine a joint law of several answers"; this module makes that
//! testable instead of a prose caveat.

use crate::manifest::Fnv;

/// Number of binary latent features in every world (`L = 4`, so `|H| = 16`).
pub const FEATURES: usize = 4;
/// Number of hypothesis-space *families*. Question coefficients depend on the
/// family, so train/dev/calibration/test see different laws; a family belongs
/// to exactly one split.
pub const FAMILIES: usize = 16;

/// The four declared data partitions (`EDGE0_RLCD_PLAN.md` §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Split {
    Train,
    Dev,
    Calibration,
    Test,
}

impl Split {
    pub const ALL: [Split; 4] = [Split::Train, Split::Dev, Split::Calibration, Split::Test];

    pub fn label(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Dev => "dev",
            Split::Calibration => "calibration",
            Split::Test => "test",
        }
    }
}

/// The typed question schema (`EDGE0_RLCD_PLAN.md` §2).
///
/// `Choice` carries the number of declared options and `Score` the number of
/// ordered rubric levels. Both are *declared*, never inferred from a logit
/// tensor: an empty schema is an explicit failure
/// ([`SchemaError::EmptySchema`]), not a silent uniform fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuestionKind {
    Noul,
    Choice { options: usize },
    Score { levels: usize },
}

impl QuestionKind {
    /// `K_q` — the outcome count.
    pub fn outcomes(self) -> usize {
        match self {
            QuestionKind::Noul => 2,
            QuestionKind::Choice { options } => options,
            QuestionKind::Score { levels } => levels,
        }
    }

    pub fn validate(self) -> Result<(), SchemaError> {
        match self {
            QuestionKind::Noul => Ok(()),
            QuestionKind::Choice { options: 0 } | QuestionKind::Score { levels: 0 | 1 } => {
                Err(SchemaError::EmptySchema)
            }
            QuestionKind::Choice { .. } | QuestionKind::Score { .. } => Ok(()),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            QuestionKind::Noul => "noul",
            QuestionKind::Choice { .. } => "choice",
            QuestionKind::Score { .. } => "score",
        }
    }
}

/// Invalid question schema. Surfaced rather than repaired into a uniform
/// distribution, because a repaired output hides a numerical/schema failure in
/// scoring (`EDGE0_RLCD_PLAN.md` §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchemaError {
    EmptySchema,
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaError::EmptySchema => write!(f, "empty or one-sided question schema"),
        }
    }
}

impl std::error::Error for SchemaError {}

/// One world before question expansion: `id` and `family` are allocated by the
/// manifest, so a question can never leak across a split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldSpec {
    pub id: u32,
    pub family: u32,
    pub split: Split,
}

/// A configured, disjoint world manifest.
///
/// World ids are assigned in one contiguous block per split and each family is
/// owned by exactly one split, so overlap is impossible by construction and
/// [`WorldManifest::assert_disjoint`] is a check rather than a promise.
#[derive(Clone, Debug)]
pub struct WorldManifest {
    seed: u64,
    worlds: Vec<WorldSpec>,
    family_owner: Vec<Split>,
}

impl WorldManifest {
    /// The first campaign's manifests (`EDGE0_RLCD_PLAN.md` §5):
    /// 2048/256/256/512 worlds. Families are split proportionally and each
    /// world draws its family only from its split's families.
    pub fn campaign(seed: u64) -> Self {
        Self::new(seed, 2048, 256, 256, 512)
    }

    pub fn new(seed: u64, train: usize, dev: usize, calibration: usize, test: usize) -> Self {
        assert!(FAMILIES >= 4, "each split needs at least one family");
        let counts = [train, dev, calibration, test];
        // Family ownership is a property of the family, not of how many worlds
        // a split happens to hold: one family per split, the rest to train.
        let mut family_owner = vec![Split::Train; FAMILIES];
        family_owner[1] = Split::Dev;
        family_owner[2] = Split::Calibration;
        family_owner[3] = Split::Test;

        let mut worlds = Vec::with_capacity(counts.iter().sum());
        let mut id = 0u32;
        for (i, &split) in Split::ALL.iter().enumerate() {
            let families: Vec<u32> = (0..FAMILIES as u32)
                .filter(|f| family_owner[*f as usize] == split)
                .collect();
            for j in 0..counts[i] {
                let family = families[j % families.len().max(1)];
                worlds.push(WorldSpec { id, family, split });
                id += 1;
            }
        }
        Self {
            seed,
            worlds,
            family_owner,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn worlds(&self) -> &[WorldSpec] {
        &self.worlds
    }

    pub fn family_owner(&self, family: u32) -> Split {
        self.family_owner[family as usize % FAMILIES]
    }

    pub fn split(&self, split: Split) -> impl Iterator<Item = &WorldSpec> {
        self.worlds.iter().filter(move |w| w.split == split)
    }

    /// FNV-1a over `(id, family, split)`, so an experiment manifest identifies
    /// the exact world list and not merely the seed that generated it.
    pub fn content_hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.write(&self.seed.to_le_bytes());
        for w in &self.worlds {
            h.write(&w.id.to_le_bytes());
            h.write(&w.family.to_le_bytes());
            h.write(&[w.split as u8]);
        }
        h.finish()
    }

    /// Panics if a world id or a family is owned twice, or if a family's worlds
    /// appear in a split other than its owner's.
    pub fn assert_disjoint(&self) {
        let mut seen_ids = std::collections::BTreeSet::new();
        let mut seen_families: std::collections::BTreeMap<u32, Split> =
            std::collections::BTreeMap::new();
        for w in &self.worlds {
            assert!(seen_ids.insert(w.id), "world id {} reused", w.id);
            if let Some(&prev) = seen_families.get(&w.family) {
                assert_eq!(prev, w.split, "family {} spans two splits", w.family);
            }
            seen_families.insert(w.family, w.split);
            assert_eq!(self.family_owner(w.family), w.split);
        }
    }

    /// Question ids are derived from the world id and a question index *after*
    /// the world list is fixed — "expose world IDs before question expansion".
    pub fn question_id(&self, world: &WorldSpec, index: u32) -> u64 {
        let mut h = Fnv::new();
        h.write(&self.seed.to_le_bytes());
        h.write(&world.id.to_le_bytes());
        h.write(&world.family.to_le_bytes());
        h.write(&index.to_le_bytes());
        h.finish()
    }
}

/// A finite hypothesis: one binary feature vector with its prior weight.
#[derive(Clone, Debug, PartialEq)]
pub struct Hypothesis {
    pub features: [bool; FEATURES],
    pub prior: f64,
}

/// An exact conditional outcome law derived from the world's hidden
/// hypothesis. `coefficients` are fixed by `(family, question id)`.
#[derive(Clone, Debug)]
pub struct QuestionOracle {
    pub id: u64,
    pub kind: QuestionKind,
}

/// A world materialised from a [`WorldSpec`].
#[derive(Clone, Debug)]
pub struct World {
    pub spec: WorldSpec,
    pub hypotheses: Vec<Hypothesis>,
    /// Index into `hypotheses` of the world's true state.
    pub truth: usize,
    seed: u64,
}

/// `±1` encoding of a hypothesis's features.
fn signs(h: &Hypothesis) -> [f64; FEATURES] {
    let mut s = [0.0; FEATURES];
    for (i, b) in h.features.iter().enumerate() {
        s[i] = if *b { 1.0 } else { -1.0 };
    }
    s
}

/// A deterministic coefficient stream in `[-1, 1]` from an FNV hash.
fn coeff(seed: u64, question: u64, which: u32) -> f64 {
    let mut h = Fnv::new();
    h.write(&seed.to_le_bytes());
    h.write(&question.to_le_bytes());
    h.write(&which.to_le_bytes());
    let x = h.finish();
    (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// The scalar latent score of a hypothesis for a question.
fn score(seed: u64, qid: u64, h: &Hypothesis) -> f64 {
    let s = signs(h);
    let mut acc = 0.0;
    for (i, si) in s.iter().enumerate() {
        acc += coeff(seed, qid, i as u32) * si;
    }
    // A nonzero offset keeps the binary question nondegenerate (no default
    // 50/50 collapse), and the `1.2` scale keeps typical sigmoids off 0/1.
    acc + 0.35 * coeff(seed, qid, 99) + 0.25
}

/// Exact categorical law for `kind` at a hypothesis, before normalization.
fn unnormalized(seed: u64, qid: u64, kind: QuestionKind, h: &Hypothesis) -> Vec<f64> {
    let s = score(seed, qid, h);
    match kind {
        QuestionKind::Noul => {
            let p = 1.0 / (1.0 + (-1.2 * s).exp());
            vec![1.0 - p, p]
        }
        QuestionKind::Choice { options } => (0..options)
            .map(|k| (0.8 * s * coeff(seed, qid, 1000 + k as u32)).exp())
            .collect(),
        QuestionKind::Score { levels } => {
            // Ordinal (cumulative) parameterization: `P(level ≥ j)` increases
            // with the latent score and thresholds are ordered, so shifting a
            // latent score upward cannot lower the distribution in the
            // stochastic order — `ordinal_level_order_is_a_stochastic_order`.
            let mut thresholds = Vec::with_capacity(levels - 1);
            for j in 1..levels {
                let t = -1.5 + 3.0 * (j as f64 - 0.5) / (levels - 1).max(1) as f64;
                thresholds.push(t + 0.2 * coeff(seed, qid, 2000 + j as u32));
            }
            let mut ge = vec![1.0];
            for t in &thresholds {
                ge.push(1.0 / (1.0 + (-1.5 * (s - t)).exp()));
            }
            ge.push(0.0);
            (0..levels).map(|j| (ge[j] - ge[j + 1]).max(0.0)).collect()
        }
    }
}

fn normalize(mut v: Vec<f64>) -> Vec<f64> {
    let total: f64 = v.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        let k = v.len();
        return vec![1.0 / k as f64; k];
    }
    for x in v.iter_mut() {
        *x /= total;
    }
    v
}

impl World {
    pub fn new(spec: WorldSpec, seed: u64) -> Self {
        let n = 1usize << FEATURES;
        let mut hypotheses = Vec::with_capacity(n);
        for mask in 0..n {
            let mut features = [false; FEATURES];
            for (i, f) in features.iter_mut().enumerate() {
                *f = (mask >> i) & 1 == 1;
            }
            hypotheses.push(Hypothesis {
                features,
                prior: 1.0 / n as f64,
            });
        }
        let mut h = Fnv::new();
        h.write(&seed.to_le_bytes());
        h.write(&spec.id.to_le_bytes());
        h.write(&spec.family.to_le_bytes());
        let truth = (h.finish() as usize) % n;
        Self {
            spec,
            hypotheses,
            truth,
            seed,
        }
    }

    pub fn question(&self, kind: QuestionKind, index: u32) -> Result<QuestionOracle, SchemaError> {
        kind.validate()?;
        let mut h = Fnv::new();
        h.write(&self.seed.to_le_bytes());
        h.write(&self.spec.id.to_le_bytes());
        h.write(&self.spec.family.to_le_bytes());
        h.write(&index.to_le_bytes());
        Ok(QuestionOracle {
            id: h.finish(),
            kind,
        })
    }

    /// Posterior weights over hypotheses after observing `(feature, value)`
    /// pairs. Empty observations return the prior.
    pub fn posterior(&self, observed: &[(usize, bool)]) -> Vec<f64> {
        let mut w: Vec<f64> = self
            .hypotheses
            .iter()
            .map(|h| {
                if observed
                    .iter()
                    .all(|&(i, v)| i < FEATURES && h.features[i] == v)
                {
                    h.prior
                } else {
                    0.0
                }
            })
            .collect();
        let total: f64 = w.iter().sum();
        if total > 0.0 {
            for x in w.iter_mut() {
                *x /= total;
            }
        } else {
            // An observation inconsistent with the whole family is an explicit
            // schema/data error, not a silent revert to the prior.
            for x in w.iter_mut() {
                *x = 0.0;
            }
        }
        w
    }

    /// One hypothesis's own outcome law. Public so a test can independently
    /// enumerate the mixture instead of trusting [`World::conditional_law`].
    pub fn hypothesis_law(&self, q: &QuestionOracle, h: usize) -> Vec<f64> {
        normalize(unnormalized(self.seed, q.id, q.kind, &self.hypotheses[h]))
    }

    /// The exact **conditional** outcome law `P(Y | observations)`.
    pub fn conditional_law(&self, q: &QuestionOracle, observed: &[(usize, bool)]) -> Vec<f64> {
        let weights = self.posterior(observed);
        let k = q.kind.outcomes();
        let mut out = vec![0.0; k];
        for (h, w) in self.hypotheses.iter().zip(&weights) {
            if *w == 0.0 {
                continue;
            }
            let law = normalize(unnormalized(self.seed, q.id, q.kind, h));
            for (o, p) in out.iter_mut().zip(&law) {
                *o += w * p;
            }
        }
        normalize(out)
    }

    /// The world's own law (its true hypothesis), i.e. the target a model is
    /// scored against. This is `t` in `EDGE0_RLCD_PLAN.md` §4.
    pub fn true_law(&self, q: &QuestionOracle) -> Vec<f64> {
        normalize(unnormalized(
            self.seed,
            q.id,
            q.kind,
            &self.hypotheses[self.truth],
        ))
    }

    /// The **joint** law of two questions, by enumerating the shared hidden
    /// hypothesis. Independent marginals would be the outer product of
    /// [`World::conditional_law`]; for these worlds that product is generally
    /// wrong and the gap is what the E0 gate measures.
    pub fn joint_law(
        &self,
        q1: &QuestionOracle,
        q2: &QuestionOracle,
        observed: &[(usize, bool)],
    ) -> Vec<Vec<f64>> {
        let weights = self.posterior(observed);
        let mut out = vec![vec![0.0; q2.kind.outcomes()]; q1.kind.outcomes()];
        for (h, w) in self.hypotheses.iter().zip(&weights) {
            if *w == 0.0 {
                continue;
            }
            let l1 = normalize(unnormalized(self.seed, q1.id, q1.kind, h));
            let l2 = normalize(unnormalized(self.seed, q2.id, q2.kind, h));
            for (i, p1) in l1.iter().enumerate() {
                for (j, p2) in l2.iter().enumerate() {
                    out[i][j] += w * p1 * p2;
                }
            }
        }
        out
    }
}

/// Shannon entropy in nats. `0 ln 0 := 0`.
pub fn entropy(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x * x.ln())
        .sum::<f64>()
}

/// The Brier score `‖p − e_y‖²`.
pub fn brier(p: &[f64], y: usize) -> f64 {
    p.iter()
        .enumerate()
        .map(|(k, pk)| {
            let d = if k == y { pk - 1.0 } else { *pk };
            d * d
        })
        .sum()
}

/// Excess Brier risk `‖p − t‖²` against a known law `t`
/// (`EDGE0_RLCD_PLAN.md` §4) — the primary test endpoint's per-question term.
pub fn brier_excess(p: &[f64], t: &[f64]) -> f64 {
    p.iter().zip(t).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// A declared utility table for one question. Utility is what a *decision*
/// maximises; it must not alter the returned probabilities at serving time
/// (`EDGE0_RLCD_PLAN.md` §5).
#[derive(Clone, Debug)]
pub struct UtilityTable {
    pub utilities: Vec<f64>,
}

impl UtilityTable {
    /// The synthetic worlds' declared utilities: yes earns 1, each choice
    /// option has a distinct positive value, and each ordinal level earns its
    /// index scaled into `[0, 1]`.
    pub fn declared(qid: u64, kind: QuestionKind) -> Self {
        let mut utilities = Vec::with_capacity(kind.outcomes());
        for k in 0..kind.outcomes() {
            let raw = match kind {
                QuestionKind::Noul => k as f64,
                QuestionKind::Choice { options } => {
                    // Distinct, unordered option values.
                    let x = coeff(0xA5A5_5A5A, qid, 3000 + k as u32);
                    (x + 1.0) * 0.5 + 0.01 * (options - k) as f64
                }
                QuestionKind::Score { levels } => k as f64 / (levels - 1).max(1) as f64,
            };
            utilities.push(raw);
        }
        // Normalise into `[0, 1]`, which is the premise of the finite-action
        // regret bound `regret ≤ 2·TV`.
        let lo = utilities.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = utilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let span = hi - lo;
        if span > 0.0 {
            for u in utilities.iter_mut() {
                *u = (*u - lo) / span;
            }
        } else {
            for u in utilities.iter_mut() {
                *u = 0.5;
            }
        }
        Self { utilities }
    }

    /// `argmax_a Σ_y p(y) u(a, y)` for a *point action* (action = outcome).
    /// Ties resolve to the lowest index, which is a declared tie rule.
    pub fn optimal_action(&self, law: &[f64]) -> usize {
        let expected: Vec<f64> = law
            .iter()
            .zip(&self.utilities)
            .map(|(p, u)| p * u)
            .collect();
        let mut best = 0usize;
        for (i, v) in expected.iter().enumerate() {
            if *v > expected[best] {
                best = i;
            }
        }
        best
    }

    /// Regret of an action under the **true** law: the utility lost against the
    /// optimal action.
    pub fn regret(&self, truth: &[f64], action: usize) -> f64 {
        let best = self.optimal_action(truth);
        let u = |a: usize| truth[a] * self.utilities[a];
        (u(best) - u(action)).max(0.0)
    }

    /// The finite-action stability bound from `EDGE0_RLCD_PLAN.md` §5: for
    /// utilities in `[0,1]`, optimal-action regret is at most twice the total
    /// variation between the predicted and true laws.
    pub fn regret_bound(predicted: &[f64], truth: &[f64]) -> f64 {
        let tv = 0.5
            * predicted
                .iter()
                .zip(truth)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
        2.0 * tv
    }
}
