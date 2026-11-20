//! R12 — a signed, norm-preserving rotation head and its Born likelihood.
//!
//! The R10/R11 heads end in a softmax, so a single draw only ever exposes
//! non-negative probabilities.  R12 asks for a normalized **real state** with
//! signed amplitudes.  For `K` ordered outcomes this module maps a frozen
//! embedding to the `K(K−1)/2` angles of a fixed ordered product of real plane
//! rotations,
//!
//! ```text
//! ψ(θ, e) = U_θ(e) b₀,    U_θ(e) = R_{p_m}(a_m) … R_{p_1}(a_1),
//! ```
//!
//! where `b₀` is the first basis vector and `p_j` is the `j`-th pair in
//! [`Layout::pairs`].  Every `R` is orthogonal, so `‖ψ‖ = 1` for every input
//! and every parameter vector; no division by a possibly-zero output norm is
//! needed.  The first listed pair acts first.
//!
//! The 2×2 convention is the plan's:
//!
//! ```text
//! R(a) = [[ cos a, −sin a],
//!         [ sin a,  cos a]]
//! ```
//!
//! so `R(π/4)(1,1)/√2 = (0,1)` and `R(π/4)(1,−1)/√2 = (1,0)`: equal
//! measurement probabilities can still compose differently.
//!
//! `apply` is the action on an arbitrary state, not just on `b₀`.  A prepared
//! state therefore carries one column of its operator, while `apply` exposes
//! the rest.  Sequential application requires the same dimension and ordered
//! basis; [`PureState::tensor`] builds the ordered pair register.
//!
//! The likelihood is the **sum of unweighted Born NLL** over a unique evidence
//! prefix:
//!
//! ```text
//! ℓ_i(θ) = −2 log |ψ_{y_i}(θ)|,    L(θ) = (1/n) Σ_i ℓ_i(θ).
//! ```
//!
//! Zero amplitude is an infinite loss, hence a rejected proposal.  The
//! gradient is the exact derivative of the value away from nodes; the target
//! seam is [`crate::target::Potential`], so the existing exact MH-pCN kernel
//! can sample it without a new ratio.

use crate::normalize::NormGrad;
use crate::target::Potential;
use candle_core::{DType, Device, Tensor, Var};
use rand::{Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// A text input and the frozen encoder's embedding for it.
///
/// The live-text adapter belongs to the caller; this module deliberately sees
/// only the vector the encoder produced.  Empty text is rejected here so a
/// caller cannot silently turn a missing input into a zero embedding.
#[derive(Clone, Copy, Debug)]
pub struct TextInput<'a> {
    pub text: &'a str,
    pub embedding: &'a [f32],
}

impl<'a> TextInput<'a> {
    pub fn new(text: &'a str, embedding: &'a [f32], dim: usize) -> Result<Self, RotationError> {
        if text.trim().is_empty() {
            return Err(RotationError::Value("empty text".into()));
        }
        if embedding.len() != dim {
            return Err(RotationError::Shape(format!(
                "embedding has {} values, expected {dim}",
                embedding.len()
            )));
        }
        if embedding.iter().any(|v| !v.is_finite()) {
            return Err(RotationError::Value("non-finite embedding".into()));
        }
        Ok(Self { text, embedding })
    }
}

#[derive(Debug)]
pub enum RotationError {
    Shape(String),
    Value(String),
    Basis(String),
    Draw(String),
    Conflict(String),
    Io(String),
    Format(String),
    Candle(String),
}

impl std::fmt::Display for RotationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(m) => write!(f, "rotation shape: {m}"),
            Self::Value(m) => write!(f, "rotation value: {m}"),
            Self::Basis(m) => write!(f, "rotation basis: {m}"),
            Self::Draw(m) => write!(f, "rotation draw: {m}"),
            Self::Conflict(m) => write!(f, "evidence conflict: {m}"),
            Self::Io(m) => write!(f, "rotation I/O: {m}"),
            Self::Format(m) => write!(f, "rotation format: {m}"),
            Self::Candle(m) => write!(f, "rotation tensor: {m}"),
        }
    }
}

impl std::error::Error for RotationError {}

impl From<candle_core::Error> for RotationError {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e.to_string())
    }
}

/// Ordered basis and rotation-pair layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pairs: Vec<(usize, usize)>,
    labels: Vec<u64>,
}

impl Layout {
    /// Lexicographic pairs `(0,1), (0,2), …` with labels `0..K`.
    pub fn for_classes(classes: usize) -> Result<Self, RotationError> {
        if classes < 2 {
            return Err(RotationError::Shape(
                "a rotation head needs at least two outcomes".into(),
            ));
        }
        let mut pairs = Vec::with_capacity(classes * (classes - 1) / 2);
        for i in 0..classes {
            for j in (i + 1)..classes {
                pairs.push((i, j));
            }
        }
        Ok(Self {
            pairs,
            labels: (0..classes as u64).collect(),
        })
    }

    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    pub fn labels(&self) -> &[u64] {
        &self.labels
    }

    pub fn angles(&self) -> usize {
        self.pairs.len()
    }
}

/// A signed real state in one declared ordered basis.
#[derive(Clone, Debug, PartialEq)]
pub struct PureState {
    amplitudes: Vec<f64>,
    basis: BasisId,
    draw: u64,
    checkpoint: u64,
}

/// The ordered basis identity carried by a state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasisId {
    id: String,
    labels: Vec<u64>,
}

impl BasisId {
    pub fn head(model: &str, classes: usize) -> Self {
        Self {
            id: format!("{model}/out"),
            labels: (0..classes as u64).collect(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn labels(&self) -> &[u64] {
        &self.labels
    }

    pub fn dim(&self) -> usize {
        self.labels.len()
    }
}

impl PureState {
    /// Validate an explicit state.  Used by tests, external adapters and any
    /// caller that has already prepared amplitudes; a state that is not
    /// normalized is rejected rather than silently renormalized.
    pub fn new(
        amplitudes: Vec<f64>,
        basis: BasisId,
        draw: u64,
        checkpoint: u64,
    ) -> Result<Self, RotationError> {
        if amplitudes.is_empty() || amplitudes.len() != basis.dim() {
            return Err(RotationError::Shape(format!(
                "state has {} amplitudes for a {}-label basis",
                amplitudes.len(),
                basis.dim()
            )));
        }
        if amplitudes.iter().any(|v| !v.is_finite()) {
            return Err(RotationError::Value("non-finite amplitude".into()));
        }
        let norm2: f64 = amplitudes.iter().map(|v| v * v).sum();
        if (norm2 - 1.0).abs() > 1e-6 {
            return Err(RotationError::Value(format!(
                "state norm² is {norm2}, expected 1"
            )));
        }
        Ok(Self {
            amplitudes,
            basis,
            draw,
            checkpoint,
        })
    }

    pub fn amplitudes(&self) -> &[f64] {
        &self.amplitudes
    }

    pub fn basis(&self) -> &BasisId {
        &self.basis
    }

    pub fn draw(&self) -> u64 {
        self.draw
    }

    pub fn checkpoint(&self) -> u64 {
        self.checkpoint
    }

    pub fn dim(&self) -> usize {
        self.amplitudes.len()
    }

    /// Projective measurement probabilities in this state's ordered basis.
    pub fn measure(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|v| v * v).collect()
    }

    /// Continue from outcome `j` after a projective measurement in this basis.
    pub fn project(&self, outcome: usize) -> Result<Self, RotationError> {
        if outcome >= self.dim() {
            return Err(RotationError::Value(format!(
                "outcome {outcome} outside 0..{}",
                self.dim()
            )));
        }
        let mut amplitudes = vec![0.0; self.dim()];
        amplitudes[outcome] = 1.0;
        Ok(Self {
            amplitudes,
            basis: self.basis.clone(),
            draw: self.draw,
            checkpoint: self.checkpoint,
        })
    }

    /// Ordered tensor product `a ⊗ b`, with `(i,j)` at `i·dim(b)+j`.
    ///
    /// The two factors must come from the same posterior draw: pairing draws
    /// is a factorized-posterior choice the caller has to make explicitly.
    pub fn tensor(a: &Self, b: &Self) -> Result<Self, RotationError> {
        if a.draw != b.draw {
            return Err(RotationError::Draw(format!(
                "tensor factors use draws {} and {}",
                a.draw, b.draw
            )));
        }
        let db = b.dim();
        let mut amplitudes = Vec::with_capacity(a.dim() * db);
        let mut labels = Vec::with_capacity(a.dim() * db);
        for (i, &ai) in a.amplitudes.iter().enumerate() {
            for (j, &bj) in b.amplitudes.iter().enumerate() {
                amplitudes.push(ai * bj);
                labels.push(a.basis.labels[i] * db as u64 + b.basis.labels[j]);
            }
        }
        Self::new(
            amplitudes,
            BasisId {
                id: format!("({}⊗{})", a.basis.id, b.basis.id),
                labels,
            },
            a.draw,
            a.checkpoint.max(b.checkpoint),
        )
    }
}

/// Apply one plane rotation to a two-level state.  Exposed so the composition
/// oracle can state its angle directly instead of inverting a network.
pub fn rotate_pair(
    state: &PureState,
    pair: (usize, usize),
    angle: f64,
) -> Result<PureState, RotationError> {
    let (a, b) = pair;
    if a == b || a >= state.dim() || b >= state.dim() {
        return Err(RotationError::Shape(format!(
            "pair ({a},{b}) outside a {}-level state",
            state.dim()
        )));
    }
    let mut amplitudes = state.amplitudes.clone();
    let (c, s) = (angle.cos(), angle.sin());
    amplitudes[a] = c * state.amplitudes[a] - s * state.amplitudes[b];
    amplitudes[b] = s * state.amplitudes[a] + c * state.amplitudes[b];
    PureState::new(
        amplitudes,
        state.basis.clone(),
        state.draw,
        state.checkpoint,
    )
}

/// A frozen-embedding → angles map with a fixed rotation layout.
#[derive(Clone, Debug)]
pub struct RotationHead {
    name: String,
    dim: usize,
    classes: usize,
    hidden: Vec<usize>,
    layout: Layout,
}

impl RotationHead {
    pub fn new(dim: usize, classes: usize, hidden: &[usize]) -> Result<Self, RotationError> {
        Self::with_name("rotation-head", dim, classes, hidden)
    }

    /// A named head.  The name is part of the state's basis identity, so two
    /// independently trained heads cannot be composed by accident.
    pub fn with_name(
        name: &str,
        dim: usize,
        classes: usize,
        hidden: &[usize],
    ) -> Result<Self, RotationError> {
        if name.trim().is_empty() {
            return Err(RotationError::Value("head name is empty".into()));
        }
        if dim == 0 {
            return Err(RotationError::Shape("embedding dimension is zero".into()));
        }
        if hidden.is_empty() || hidden.iter().any(|&h| h == 0) {
            return Err(RotationError::Shape(
                "the head needs at least one non-empty tanh layer".into(),
            ));
        }
        Ok(Self {
            name: name.to_string(),
            dim,
            classes,
            hidden: hidden.to_vec(),
            layout: Layout::for_classes(classes)?,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn hidden(&self) -> &[usize] {
        &self.hidden
    }

    /// `(in, out)` per affine layer, in execution order.
    pub fn layers(&self) -> Vec<(usize, usize)> {
        let mut widths = Vec::with_capacity(self.hidden.len() + 2);
        widths.push(self.dim);
        widths.extend_from_slice(&self.hidden);
        widths.push(self.layout.angles());
        widths.windows(2).map(|w| (w[0], w[1])).collect()
    }

    pub fn flat_size(&self) -> usize {
        self.layers().iter().map(|&(i, o)| i * o + o).sum()
    }

    fn angles_tensor(&self, theta: &Tensor, embedding: &[f32]) -> candle_core::Result<Tensor> {
        let mut at = 0usize;
        let mut h = Tensor::from_vec(embedding.to_vec(), (1, self.dim), &Device::Cpu)?;
        let layers = self.layers();
        for (i, &(fin, fout)) in layers.iter().enumerate() {
            let w = theta.narrow(0, at, fin * fout)?.reshape((fin, fout))?;
            at += fin * fout;
            let b = theta.narrow(0, at, fout)?;
            at += fout;
            h = h.matmul(&w)?.broadcast_add(&b)?;
            if i + 1 < layers.len() {
                h = h.tanh()?;
            }
        }
        h.reshape(self.layout.angles())
    }

    /// The orthogonal operator for one embedding, in the declared basis.
    pub fn operator(&self, theta: &[f32], embedding: &[f32]) -> Result<Tensor, RotationError> {
        if theta.len() != self.flat_size() {
            return Err(RotationError::Shape(format!(
                "head has {} parameters, got {}",
                self.flat_size(),
                theta.len()
            )));
        }
        if theta.iter().any(|v| !v.is_finite()) {
            return Err(RotationError::Value("non-finite head parameter".into()));
        }
        if embedding.len() != self.dim {
            return Err(RotationError::Shape(format!(
                "embedding has {} values, expected {}",
                embedding.len(),
                self.dim
            )));
        }
        let t = Tensor::from_vec(theta.to_vec(), theta.len(), &Device::Cpu)?;
        let angles = self.angles_tensor(&t, embedding)?;
        let k = self.classes;
        let mut u = Tensor::eye(k, DType::F32, &Device::Cpu)?;
        for (index, &(a, b)) in self.layout.pairs.iter().enumerate() {
            let angle = angles.narrow(0, index, 1)?.reshape(())?;
            let c = angle.cos()?;
            let s = angle.sin()?;
            let mut diagonal = vec![0.0f32; k * k];
            diagonal[a * k + a] = 1.0;
            diagonal[b * k + b] = 1.0;
            let mut cross = vec![0.0f32; k * k];
            cross[b * k + a] = 1.0;
            cross[a * k + b] = -1.0;
            let diagonal = Tensor::from_vec(diagonal, (k, k), &Device::Cpu)?;
            let cross = Tensor::from_vec(cross, (k, k), &Device::Cpu)?;
            let r = Tensor::eye(k, DType::F32, &Device::Cpu)?
                .broadcast_add(&diagonal.broadcast_mul(&c.affine(1.0, -1.0)?)?)?
                .broadcast_add(&cross.broadcast_mul(&s)?)?;
            u = r.matmul(&u)?;
        }
        Ok(u)
    }

    /// `ψ = U_θ(e) b₀`, with the state's metadata attached.
    pub fn prepare(
        &self,
        theta: &[f32],
        input: &TextInput<'_>,
        draw: u64,
        checkpoint: u64,
    ) -> Result<PureState, RotationError> {
        if input.embedding.len() != self.dim {
            return Err(RotationError::Shape(format!(
                "embedding has {} values, expected {}",
                input.embedding.len(),
                self.dim
            )));
        }
        let u = self.operator(theta, input.embedding)?;
        let column = u.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        PureState::new(
            column.into_iter().map(f64::from).collect(),
            BasisId::head(self.name(), self.classes),
            draw,
            checkpoint,
        )
    }

    /// `ψ' = U_θ(e) ψ` in the same ordered basis and posterior draw.
    pub fn apply(
        &self,
        theta: &[f32],
        input: &TextInput<'_>,
        state: &PureState,
        draw: u64,
    ) -> Result<PureState, RotationError> {
        if state.draw != draw {
            return Err(RotationError::Draw(format!(
                "state draw {} does not match apply draw {draw}",
                state.draw
            )));
        }
        let want = BasisId::head(self.name(), self.classes);
        if state.basis != want {
            return Err(RotationError::Basis(format!(
                "state basis `{}` does not match head basis `{}`",
                state.basis.id, want.id
            )));
        }
        let u = self.operator(theta, input.embedding)?;
        let psi = Tensor::from_vec(
            state
                .amplitudes
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>(),
            (state.dim(), 1),
            &Device::Cpu,
        )?;
        let out = u.matmul(&psi)?.flatten_all()?.to_vec1::<f32>()?;
        PureState::new(
            out.into_iter().map(f64::from).collect(),
            state.basis.clone(),
            state.draw,
            state.checkpoint,
        )
    }

    pub fn label(&self) -> &'static str {
        "rotation-head"
    }
}

/// One uniquely identified observation.  `steps` are the recorded embedding
/// applications in order; `observed` is the final measured outcome.  A
/// single-step label is a one-element `steps`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    pub steps: Vec<Vec<f32>>,
    pub observed: u32,
}

/// Append-only evidence prefix with stable IDs and a prefix checkpoint.
#[derive(Clone, Debug)]
pub struct EvidenceStore {
    dim: usize,
    classes: usize,
    events: Vec<Event>,
    index: HashMap<String, usize>,
}

impl EvidenceStore {
    pub fn new(dim: usize, classes: usize) -> Result<Self, RotationError> {
        if dim == 0 || classes < 2 {
            return Err(RotationError::Shape(
                "evidence needs a positive embedding dimension and at least two classes".into(),
            ));
        }
        Ok(Self {
            dim,
            classes,
            events: Vec::new(),
            index: HashMap::new(),
        })
    }

    /// Append one event.  An identical retry is a no-op (`Ok(false)`);
    /// reusing an ID for different evidence is an error (`Err`), as is an
    /// empty step list or a label outside the head's basis.
    pub fn append(
        &mut self,
        id: impl Into<String>,
        steps: Vec<Vec<f32>>,
        observed: u32,
    ) -> Result<bool, RotationError> {
        let id = id.into();
        if id.is_empty() {
            return Err(RotationError::Value("event ID is empty".into()));
        }
        if steps.is_empty() {
            return Err(RotationError::Value(format!(
                "event `{id}` has no applications"
            )));
        }
        if observed as usize >= self.classes {
            return Err(RotationError::Value(format!(
                "event `{id}` observed {observed} outside 0..{}",
                self.classes
            )));
        }
        for (index, step) in steps.iter().enumerate() {
            if step.len() != self.dim {
                return Err(RotationError::Shape(format!(
                    "event `{id}` step {index} has {} values, expected {}",
                    step.len(),
                    self.dim
                )));
            }
            if step.iter().any(|v| !v.is_finite()) {
                return Err(RotationError::Value(format!(
                    "event `{id}` step {index} has a non-finite value"
                )));
            }
        }
        if let Some(&at) = self.index.get(&id) {
            let old = &self.events[at];
            if old.steps == steps && old.observed == observed {
                return Ok(false);
            }
            return Err(RotationError::Conflict(format!(
                "event `{id}` is already bound to different evidence"
            )));
        }
        self.index.insert(id.clone(), self.events.len());
        self.events.push(Event {
            id,
            steps,
            observed,
        });
        Ok(true)
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Number of committed events.  This is the evidence-prefix version.
    pub fn checkpoint(&self) -> u64 {
        self.events.len() as u64
    }

    pub fn potential(&self, head: RotationHead) -> Result<BornPotential, RotationError> {
        if head.dim != self.dim || head.classes != self.classes {
            return Err(RotationError::Shape(
                "evidence and head dimensions/classes do not match".into(),
            ));
        }
        BornPotential::new(head, self.events.clone())
    }
}

/// The unweighted Born-NLL potential over an evidence prefix.
#[derive(Clone, Debug)]
pub struct BornPotential {
    head: RotationHead,
    events: Vec<Event>,
}

impl BornPotential {
    pub fn new(head: RotationHead, events: Vec<Event>) -> Result<Self, RotationError> {
        if events.is_empty() {
            return Err(RotationError::Value(
                "a Born potential needs at least one event".into(),
            ));
        }
        for event in &events {
            if event.steps.is_empty() {
                return Err(RotationError::Value(format!(
                    "event `{}` has no applications",
                    event.id
                )));
            }
            if event.observed as usize >= head.classes {
                return Err(RotationError::Value(format!(
                    "event `{}` observed {} outside 0..{}",
                    event.id, event.observed, head.classes
                )));
            }
            for step in &event.steps {
                if step.len() != head.dim {
                    return Err(RotationError::Shape(format!(
                        "event `{}` step has {} values, expected {}",
                        event.id,
                        step.len(),
                        head.dim
                    )));
                }
            }
        }
        Ok(Self { head, events })
    }

    pub fn head(&self) -> &RotationHead {
        &self.head
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    fn observed_amplitudes(&self, theta: &[f32]) -> Result<Vec<f64>, RotationError> {
        let mut out = Vec::with_capacity(self.events.len());
        for event in &self.events {
            let mut psi: Vec<f64> = (0..self.head.classes)
                .map(|i| if i == 0 { 1.0 } else { 0.0 })
                .collect();
            for step in &event.steps {
                let u = self.head.operator(theta, step)?;
                let u = u.to_vec2::<f32>()?;
                let mut next = vec![0.0f64; self.head.classes];
                for (i, row) in u.iter().enumerate() {
                    for (j, &v) in row.iter().enumerate() {
                        next[i] += f64::from(v) * psi[j];
                    }
                }
                psi = next;
            }
            out.push(psi[event.observed as usize]);
        }
        Ok(out)
    }

    fn state_tensor(&self, theta: &Tensor, event: &Event) -> candle_core::Result<Tensor> {
        let mut psi = vec![0.0f32; self.head.classes];
        psi[0] = 1.0;
        let mut psi = Tensor::from_vec(psi, (self.head.classes, 1), &Device::Cpu)?;
        for step in &event.steps {
            let u = self.head.angles_tensor(theta, step)?;
            let k = self.head.classes;
            let mut u_matrix = Tensor::eye(k, DType::F32, &Device::Cpu)?;
            for (index, &(a, b)) in self.head.layout.pairs.iter().enumerate() {
                let angle = u.narrow(0, index, 1)?.reshape(())?;
                let c = angle.cos()?;
                let s = angle.sin()?;
                let mut diagonal = vec![0.0f32; k * k];
                diagonal[a * k + a] = 1.0;
                diagonal[b * k + b] = 1.0;
                let mut cross = vec![0.0f32; k * k];
                cross[b * k + a] = 1.0;
                cross[a * k + b] = -1.0;
                let diagonal = Tensor::from_vec(diagonal, (k, k), &Device::Cpu)?;
                let cross = Tensor::from_vec(cross, (k, k), &Device::Cpu)?;
                let r = Tensor::eye(k, DType::F32, &Device::Cpu)?
                    .broadcast_add(&diagonal.broadcast_mul(&c.affine(1.0, -1.0)?)?)?
                    .broadcast_add(&cross.broadcast_mul(&s)?)?;
                u_matrix = r.matmul(&u_matrix)?;
            }
            psi = u_matrix.matmul(&psi)?;
        }
        Ok(psi.flatten_all()?)
    }
}

impl Potential for BornPotential {
    fn n_terms(&self) -> usize {
        self.events.len()
    }

    fn mean_loss(&self, theta: &Tensor) -> f64 {
        let theta = match theta.flatten_all().and_then(|v| v.to_vec1::<f32>()) {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        };
        let amplitudes = match self.observed_amplitudes(&theta) {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        };
        let mut sum = 0.0;
        for amplitude in amplitudes {
            if amplitude == 0.0 || !amplitude.is_finite() {
                return f64::INFINITY;
            }
            sum += -2.0 * amplitude.abs().ln();
        }
        sum / self.events.len() as f64
    }

    fn mean_grad_with(&self, theta: &Tensor, _norm: NormGrad) -> Option<Tensor> {
        let flat = theta.flatten_all().ok()?.to_vec1::<f32>().ok()?;
        let amplitudes = self.observed_amplitudes(&flat).ok()?;
        if amplitudes.iter().any(|v| *v == 0.0 || !v.is_finite()) {
            return None;
        }
        let var = Var::from_tensor(theta).ok()?;
        let mut losses = Vec::with_capacity(self.events.len());
        for event in &self.events {
            let psi = self.state_tensor(var.as_tensor(), event).ok()?;
            let amplitude = psi
                .narrow(0, event.observed as usize, 1)
                .ok()?
                .reshape(())
                .ok()?;
            let loss = amplitude.abs().ok()?.log().ok()?.affine(-2.0, 0.0).ok()?;
            losses.push(loss);
        }
        let total = Tensor::stack(&losses, 0).ok()?.sum_all().ok()?;
        let mean = total.affine(1.0 / self.events.len() as f64, 0.0).ok()?;
        if !mean.to_scalar::<f32>().ok()?.is_finite() {
            return None;
        }
        let grads = mean.backward().ok()?;
        grads.get(var.as_tensor())?.flatten_all().ok()
    }

    fn label(&self) -> &'static str {
        "rotation-born-nll"
    }
}

fn prior_draw_with(flat: usize, sigma: f64, rng: &mut impl Rng) -> Vec<f32> {
    (0..flat)
        .map(|_| {
            let z: f64 = rng.sample(rand_distr::StandardNormal);
            (z * sigma) as f32
        })
        .collect()
}

fn phi_value(potential: &BornPotential, lambda: f64, sigma: f64, theta: &[f32]) -> f64 {
    let t = match Tensor::from_vec(theta.to_vec(), theta.len(), &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return f64::INFINITY,
    };
    let loss = potential.mean_loss(&t);
    if !loss.is_finite() {
        return f64::INFINITY;
    }
    lambda * potential.n_terms() as f64 * loss
        + 0.5 * theta.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>()
            / (sigma * sigma)
}

/// One exact MH step on the full head, with the plan's Gaussian proposal
///
/// ```text
/// m(θ) = √(1−β²) θ − (β² σ²/2) ∇Φ(θ),   θ' ~ N(m(θ), β² σ² I)
/// Φ(θ) = λ N mean_loss(θ) + ‖θ‖²/(2σ²).
/// ```
///
/// The full posterior difference and the reverse-minus-forward proposal
/// density are both included.  `phi` is the caller's cached `Φ(θ)`; on
/// acceptance it is updated with `θ`.  A proposal with a non-finite loss or
/// missing gradient is rejected, as is the current point if it cannot be
/// evaluated.  This is a full-target kernel, not a restricted-subspace one.
pub fn mh_pcn_full_step<P: Potential>(
    potential: &P,
    lambda: f64,
    sigma: f64,
    beta: f64,
    theta: &mut [f32],
    phi: &mut f64,
    rng: &mut impl Rng,
) -> bool {
    assert!(sigma > 0.0, "prior sigma must be positive");
    assert!((0.0..=1.0).contains(&beta), "beta must be in [0, 1]");
    if !phi.is_finite() {
        return false;
    }
    let n = potential.n_terms() as f64;
    let dim = theta.len();
    let current = match Tensor::from_vec(theta.to_vec(), dim, &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let Some(grad) = potential.mean_grad(&current) else {
        return false;
    };
    let Ok(grad) = grad.flatten_all().and_then(|v| v.to_vec1::<f32>()) else {
        return false;
    };
    if grad.len() != dim {
        return false;
    }
    let prior_var = sigma * sigma;
    let shrink = (1.0 - beta * beta).max(0.0).sqrt();
    let drift = 0.5 * beta * beta * prior_var;
    let mut mean = vec![0.0f64; dim];
    for i in 0..dim {
        let grad_phi = lambda * n * f64::from(grad[i]) + f64::from(theta[i]) / prior_var;
        mean[i] = shrink * f64::from(theta[i]) - drift * grad_phi;
    }
    let step = beta * sigma;
    if step == 0.0 {
        return false;
    }
    let mut proposal = vec![0.0f32; dim];
    let mut log_q_forward = 0.0f64;
    let log_norm = (2.0 * std::f64::consts::PI * step * step).ln();
    for i in 0..dim {
        let z: f64 = rng.sample(rand_distr::StandardNormal);
        proposal[i] = (mean[i] + step * z) as f32;
        log_q_forward += -0.5 * (z * z + log_norm);
    }
    let proposal_theta = match Tensor::from_vec(proposal.clone(), dim, &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let proposal_loss = potential.mean_loss(&proposal_theta);
    if !proposal_loss.is_finite() {
        return false;
    }
    let proposal_phi = lambda * n * proposal_loss
        + 0.5
            * proposal
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
            / prior_var;
    let Some(proposal_grad) = potential.mean_grad(&proposal_theta) else {
        return false;
    };
    let Ok(proposal_grad) = proposal_grad.flatten_all().and_then(|v| v.to_vec1::<f32>()) else {
        return false;
    };
    if proposal_grad.len() != dim {
        return false;
    }
    let mut reverse_mean = vec![0.0f64; dim];
    for i in 0..dim {
        let grad_phi =
            lambda * n * f64::from(proposal_grad[i]) + f64::from(proposal[i]) / prior_var;
        reverse_mean[i] = shrink * f64::from(proposal[i]) - drift * grad_phi;
    }
    let mut log_q_reverse = 0.0f64;
    for i in 0..dim {
        let z = (f64::from(theta[i]) - reverse_mean[i]) / step;
        log_q_reverse += -0.5 * (z * z + log_norm);
    }
    let log_alpha = -(proposal_phi - *phi) + log_q_reverse - log_q_forward;
    let accept = log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha;
    if accept {
        theta.copy_from_slice(&proposal);
        *phi = proposal_phi;
    }
    accept
}

/// Bayesian predictive from a posterior draw ensemble: the diagonal of
/// `S⁻¹ Σ_s ψ_s ψ_sᵀ`, never a normalized mean amplitude.
pub fn predictive(states: &[PureState], classes: usize) -> Result<Vec<f64>, RotationError> {
    if states.is_empty() {
        return Err(RotationError::Value("empty draw ensemble".into()));
    }
    let mut p = vec![0.0; classes];
    for state in states {
        if state.dim() != classes {
            return Err(RotationError::Shape(format!(
                "state has {} outcomes, expected {classes}",
                state.dim()
            )));
        }
        for (pi, &a) in p.iter_mut().zip(state.amplitudes()) {
            *pi += a * a;
        }
    }
    let n = states.len() as f64;
    for v in &mut p {
        *v /= n;
    }
    Ok(p)
}

/// Small deterministic pseudo-embedding used by the finite checks and the
/// composition example.  This is not a text encoder.
pub fn synthetic_embedding(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x7a11_0b0e);
    (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

/// Deterministic counter-based RNG.  Unlike a platform RNG, its complete
/// state (seed and counter) is serializable, so an interrupted update can
/// resume the same proposal stream exactly.
#[derive(Clone, Debug)]
pub struct CounterRng {
    seed: u64,
    counter: u64,
}

impl CounterRng {
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    pub fn from_parts(seed: u64, counter: u64) -> Self {
        Self { seed, counter }
    }

    pub fn counter(&self) -> u64 {
        self.counter
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl RngCore for CounterRng {
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.counter = self.counter.wrapping_add(1);
        splitmix64(self.seed ^ splitmix64(self.counter))
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut at = 0;
        while at < dst.len() {
            let word = self.next_u64().to_le_bytes();
            let n = (dst.len() - at).min(8);
            dst[at..at + n].copy_from_slice(&word[..n]);
            at += n;
        }
    }
}

/// Immutable description of the head stored in a checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadSpec {
    pub name: String,
    pub dim: usize,
    pub classes: usize,
    pub hidden: Vec<usize>,
}

impl HeadSpec {
    pub fn from_head(head: &RotationHead) -> Self {
        Self {
            name: head.name.clone(),
            dim: head.dim,
            classes: head.classes,
            hidden: head.hidden.clone(),
        }
    }

    pub fn head(&self) -> Result<RotationHead, RotationError> {
        RotationHead::with_name(&self.name, self.dim, self.classes, &self.hidden)
    }
}

/// Sampler and prior settings.  These are target/kernel settings, not tuning
/// that may change silently between a saved run and its continuation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionConfig {
    pub sigma: f64,
    pub lambda: f64,
    pub beta: f64,
    pub burn: usize,
    pub steps: usize,
    pub thin: usize,
    pub chains: usize,
    pub draws: usize,
    pub seed: u64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            sigma: 0.5,
            lambda: 1.0,
            beta: 0.2,
            burn: 400,
            steps: 1_200,
            thin: 10,
            chains: 4,
            draws: 40,
            seed: 20261118,
        }
    }
}

impl SessionConfig {
    pub fn validate(&self) -> Result<(), RotationError> {
        if !(self.sigma.is_finite() && self.sigma > 0.0) {
            return Err(RotationError::Value("sigma must be finite and > 0".into()));
        }
        if !(self.lambda.is_finite() && self.lambda >= 0.0) {
            return Err(RotationError::Value(
                "lambda must be finite and >= 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.beta) || !self.beta.is_finite() {
            return Err(RotationError::Value("beta must be in [0, 1]".into()));
        }
        if self.steps == 0 || self.thin == 0 || self.chains == 0 || self.draws == 0 {
            return Err(RotationError::Value(
                "steps, thin, chains and draws must all be positive".into(),
            ));
        }
        if self.burn >= self.steps {
            return Err(RotationError::Value(
                "burn must be smaller than steps".into(),
            ));
        }
        Ok(())
    }
}

/// One chain's resumable state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainState {
    pub seed: u64,
    pub counter: u64,
    pub theta: Vec<f32>,
    pub phi: f64,
}

impl ChainState {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            counter: 0,
            theta: Vec::new(),
            phi: 0.0,
        }
    }

    fn rng(&self) -> CounterRng {
        CounterRng::from_parts(self.seed, self.counter)
    }
}

/// A prediction from the retained posterior draw ensemble.
#[derive(Clone, Debug)]
pub struct Prediction {
    pub states: Vec<PureState>,
    pub probabilities: Vec<f64>,
}

/// Report from one committed update.
#[derive(Clone, Debug)]
pub struct UpdateReport {
    pub checkpoint: u64,
    pub proposals: usize,
    pub accepted: usize,
    pub retained_draws: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Checkpoint {
    version: u32,
    head: HeadSpec,
    config: SessionConfig,
    evidence: Vec<Event>,
    chains: Vec<ChainState>,
    retained: Vec<Vec<f32>>,
    updates: u64,
}

const CHECKPOINT_VERSION: u32 = 1;

/// A resumable evidence/sampler loop for the signed rotation head.
///
/// The checkpoint is one JSON document containing the evidence prefix, head
/// layout, target settings, every chain's parameter and RNG counter, and the
/// retained predictive draws.  `save` writes a sibling temporary file, syncs
/// it, then renames it into place, so a crash either leaves the old checkpoint
/// or the fully written new one.  A stale `.tmp` file is never read.
#[derive(Clone, Debug)]
pub struct RotationSession {
    head: RotationHead,
    config: SessionConfig,
    evidence: EvidenceStore,
    chains: Vec<ChainState>,
    retained: Vec<Vec<f32>>,
    updates: u64,
}

impl RotationSession {
    pub fn new(head: RotationHead, config: SessionConfig) -> Result<Self, RotationError> {
        config.validate()?;
        let evidence = EvidenceStore::new(head.dim, head.classes)?;
        let chains = (0..config.chains)
            .map(|i| ChainState::new(config.seed.wrapping_add(i as u64 * 0x9e37_79b9)))
            .collect();
        Ok(Self {
            head,
            config,
            evidence,
            chains,
            retained: Vec::new(),
            updates: 0,
        })
    }

    pub fn head(&self) -> &RotationHead {
        &self.head
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub fn events(&self) -> &[Event] {
        self.evidence.events()
    }

    pub fn checkpoint(&self) -> u64 {
        self.evidence.checkpoint()
    }

    pub fn updates(&self) -> u64 {
        self.updates
    }

    pub fn retained_draws(&self) -> &[Vec<f32>] {
        &self.retained
    }

    /// A deterministic prior draw, used to initialise a mode search without
    /// exposing the session's internal RNG.  The update still starts chains
    /// from the prior or from a supplied warm start, never from a hidden fit.
    pub fn prior_theta(&self, seed: u64) -> Vec<f32> {
        prior_draw_with(
            self.head.flat_size(),
            self.config.sigma,
            &mut CounterRng::new(seed),
        )
    }

    /// Set every chain's initial parameter vector.  This is an explicit
    /// optimisation start (for example a finite-density anchor); it does not
    /// change the target, and `update` still recomputes `Φ` at the new
    /// evidence prefix before proposing.
    pub fn warm_start(&mut self, theta: &[f32]) -> Result<(), RotationError> {
        if theta.len() != self.head.flat_size() || theta.iter().any(|v| !v.is_finite()) {
            return Err(RotationError::Shape(
                "warm start must match the head and be finite".into(),
            ));
        }
        for chain in &mut self.chains {
            chain.theta = theta.to_vec();
            chain.phi = 0.0;
        }
        self.retained.clear();
        Ok(())
    }

    pub fn append_event(
        &mut self,
        id: impl Into<String>,
        steps: Vec<Vec<f32>>,
        observed: u32,
    ) -> Result<bool, RotationError> {
        let added = self.evidence.append(id, steps, observed)?;
        if added {
            self.retained.clear();
        }
        Ok(added)
    }

    fn prior_draw(&self, rng: &mut impl Rng) -> Vec<f32> {
        prior_draw_with(self.head.flat_size(), self.config.sigma, rng)
    }

    /// Run every chain on the committed evidence prefix.  Each chain starts at
    /// its preceding checkpoint (or a prior draw at round zero), keeps the
    /// thinned post-burn draws, and stores its RNG counter for exact resumption.
    pub fn update(&mut self) -> Result<UpdateReport, RotationError> {
        if self.evidence.is_empty() {
            return Err(RotationError::Value(
                "cannot update an empty evidence prefix".into(),
            ));
        }
        let potential = self.evidence.potential(self.head.clone())?;
        let flat = self.head.flat_size();
        let (lambda, sigma) = (self.config.lambda, self.config.sigma);
        let mut retained = Vec::new();
        let mut proposals = 0usize;
        let mut accepted = 0usize;
        for chain in &mut self.chains {
            let mut rng = chain.rng();
            if chain.theta.len() != flat || chain.theta.iter().any(|v| !v.is_finite()) {
                chain.theta = prior_draw_with(flat, sigma, &mut rng);
            }
            chain.phi = phi_value(&potential, lambda, sigma, &chain.theta);
            if !chain.phi.is_finite() {
                chain.theta = prior_draw_with(flat, sigma, &mut rng);
                chain.phi = phi_value(&potential, lambda, sigma, &chain.theta);
                if !chain.phi.is_finite() {
                    return Err(RotationError::Value(
                        "could not find a finite-density starting draw".into(),
                    ));
                }
            }
            for step in 0..self.config.steps {
                proposals += 1;
                if mh_pcn_full_step(
                    &potential,
                    lambda,
                    sigma,
                    self.config.beta,
                    &mut chain.theta,
                    &mut chain.phi,
                    &mut rng,
                ) {
                    accepted += 1;
                }
                if step >= self.config.burn && (step - self.config.burn) % self.config.thin == 0 {
                    retained.push(chain.theta.clone());
                }
            }
            chain.counter = rng.counter();
        }
        if retained.is_empty() {
            return Err(RotationError::Value(
                "the update retained no posterior draws".into(),
            ));
        }
        if retained.len() > self.config.draws {
            retained.truncate(self.config.draws);
        }
        self.retained = retained;
        self.updates += 1;
        Ok(UpdateReport {
            checkpoint: self.evidence.checkpoint(),
            proposals,
            accepted,
            retained_draws: self.retained.len(),
        })
    }

    fn ensure_retained(&mut self) -> Result<(), RotationError> {
        if !self.retained.is_empty() {
            return Ok(());
        }
        let mut rng = CounterRng::new(self.config.seed ^ self.updates.wrapping_mul(0xd1b5_4a32));
        self.retained = (0..self.config.draws)
            .map(|_| self.prior_draw(&mut rng))
            .collect();
        Ok(())
    }

    /// Predict from the retained posterior ensemble.  At round zero this is a
    /// prior draw ensemble; after `update` it is the thinned chain ensemble.
    pub fn predict(&mut self, input: &TextInput<'_>) -> Result<Prediction, RotationError> {
        self.predict_sequence(std::slice::from_ref(input))
    }

    /// Compose the whole sequence within one draw, then square and average
    /// over draws.  This is the plan's posterior contract: the ensemble is a
    /// mixture, not a normalized mean amplitude.
    pub fn predict_sequence(
        &mut self,
        inputs: &[TextInput<'_>],
    ) -> Result<Prediction, RotationError> {
        if inputs.is_empty() {
            return Err(RotationError::Value("empty sequence".into()));
        }
        self.ensure_retained()?;
        let mut states = Vec::with_capacity(self.retained.len());
        for (draw, theta) in self.retained.iter().enumerate() {
            let mut amplitudes = vec![0.0; self.head.classes];
            amplitudes[0] = 1.0;
            let mut state = PureState::new(
                amplitudes,
                BasisId::head(self.head.name(), self.head.classes),
                draw as u64,
                self.evidence.checkpoint(),
            )?;
            for input in inputs {
                state = self.head.apply(theta, input, &state, draw as u64)?;
            }
            states.push(state);
        }
        let probabilities = predictive(&states, self.head.classes)?;
        Ok(Prediction {
            states,
            probabilities,
        })
    }

    /// Atomic checkpoint.  A partial write lives in a sibling `.tmp` path and
    /// is never considered a committed state.
    pub fn save(&self, path: &Path) -> Result<(), RotationError> {
        let ck = Checkpoint {
            version: CHECKPOINT_VERSION,
            head: HeadSpec::from_head(&self.head),
            config: self.config.clone(),
            evidence: self.evidence.events().to_vec(),
            chains: self.chains.clone(),
            retained: self.retained.clone(),
            updates: self.updates,
        };
        let bytes = serde_json::to_vec_pretty(&ck)
            .map_err(|e| RotationError::Format(format!("serialize checkpoint: {e}")))?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| RotationError::Io(format!("create {}: {e}", parent.display())))?;
            }
        }
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| RotationError::Io(format!("create {}: {e}", tmp.display())))?;
        f.write_all(&bytes)
            .map_err(|e| RotationError::Io(format!("write {}: {e}", tmp.display())))?;
        f.sync_all()
            .map_err(|e| RotationError::Io(format!("sync {}: {e}", tmp.display())))?;
        drop(f);
        std::fs::rename(&tmp, path).map_err(|e| {
            RotationError::Io(format!(
                "rename {} -> {}: {e}",
                tmp.display(),
                path.display()
            ))
        })?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, RotationError> {
        let bytes = std::fs::read(path)
            .map_err(|e| RotationError::Io(format!("read {}: {e}", path.display())))?;
        let ck: Checkpoint = serde_json::from_slice(&bytes)
            .map_err(|e| RotationError::Format(format!("parse checkpoint: {e}")))?;
        if ck.version != CHECKPOINT_VERSION {
            return Err(RotationError::Format(format!(
                "checkpoint version {}, expected {CHECKPOINT_VERSION}",
                ck.version
            )));
        }
        ck.config.validate()?;
        let head = ck.head.head()?;
        let mut evidence = EvidenceStore::new(head.dim, head.classes)?;
        for event in ck.evidence {
            evidence.append(event.id, event.steps, event.observed)?;
        }
        if ck.chains.len() != ck.config.chains {
            return Err(RotationError::Format(
                "checkpoint chain count does not match its config".into(),
            ));
        }
        for chain in &ck.chains {
            if !chain.theta.is_empty() && chain.theta.len() != head.flat_size() {
                return Err(RotationError::Format(
                    "checkpoint chain parameter count does not match the head".into(),
                ));
            }
        }
        for draw in &ck.retained {
            if draw.len() != head.flat_size() {
                return Err(RotationError::Format(
                    "checkpoint retained draw does not match the head".into(),
                ));
            }
        }
        Ok(Self {
            head,
            config: ck.config,
            evidence,
            chains: ck.chains,
            retained: ck.retained,
            updates: ck.updates,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis(k: usize) -> BasisId {
        BasisId::head("rotation-head", k)
    }

    fn state(amplitudes: Vec<f64>, k: usize) -> PureState {
        PureState::new(amplitudes, basis(k), 7, 3).unwrap()
    }

    fn random_theta(head: &RotationHead, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..head.flat_size())
            .map(|_| rng.random_range(-0.4f32..0.4))
            .collect()
    }

    #[test]
    fn oracle_rotations_distinguish_relative_signs() {
        let plus = state(vec![1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt()], 2);
        let minus = state(vec![1.0 / 2f64.sqrt(), -1.0 / 2f64.sqrt()], 2);
        assert_eq!(plus.measure(), minus.measure());

        let angle = std::f64::consts::FRAC_PI_4;
        let p_plus = rotate_pair(&plus, (0, 1), angle).unwrap();
        let p_minus = rotate_pair(&minus, (0, 1), angle).unwrap();
        assert!((p_plus.measure()[0] - 0.0).abs() < 1e-12);
        assert!((p_plus.measure()[1] - 1.0).abs() < 1e-12);
        assert!((p_minus.measure()[0] - 1.0).abs() < 1e-12);
        assert!((p_minus.measure()[1] - 0.0).abs() < 1e-12);

        // A projective measurement and discard is the classical control.
        let measured_plus = plus.project(0).unwrap();
        let measured_minus = minus.project(0).unwrap();
        assert_eq!(measured_plus.measure(), measured_minus.measure());
    }

    #[test]
    fn prepared_states_are_signed_and_unit_norm_in_any_sequence() {
        let head = RotationHead::new(3, 4, &[5, 4]).unwrap();
        let theta = random_theta(&head, 11);
        let e0 = synthetic_embedding(1, 3);
        let e1 = synthetic_embedding(2, 3);
        let t0 = TextInput::new("first", &e0, 3).unwrap();
        let t1 = TextInput::new("second", &e1, 3).unwrap();
        let a = head.prepare(&theta, &t0, 9, 0).unwrap();
        assert_eq!(a.dim(), 4);
        assert!((a.amplitudes().iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-6);
        assert!(a.amplitudes().iter().any(|v| v.abs() > 1e-9));
        let ab = head.apply(&theta, &t1, &a, 9).unwrap();
        assert!((ab.amplitudes().iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-6);
        assert_eq!(ab.draw(), 9);
    }

    #[test]
    fn applying_a_then_b_equals_the_composed_operator() {
        let head = RotationHead::new(2, 3, &[4]).unwrap();
        let theta = random_theta(&head, 12);
        let e0 = synthetic_embedding(3, 2);
        let e1 = synthetic_embedding(4, 2);
        let t0 = TextInput::new("a", &e0, 2).unwrap();
        let t1 = TextInput::new("b", &e1, 2).unwrap();
        let start = state(vec![1.0, 0.0, 0.0], 3);
        let stepwise = head
            .apply(&theta, &t1, &head.apply(&theta, &t0, &start, 7).unwrap(), 7)
            .unwrap();
        let ua = head.operator(&theta, &e0).unwrap();
        let ub = head.operator(&theta, &e1).unwrap();
        let composed = ub.matmul(&ua).unwrap().to_vec2::<f32>().unwrap();
        let mut want = vec![0.0; 3];
        for i in 0..3 {
            want[i] = f64::from(composed[i][0]);
        }
        for (got, want) in stepwise.amplitudes().iter().zip(want) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_angle_is_the_identity() {
        let before = state(vec![0.6, -0.8], 2);
        let after = rotate_pair(&before, (0, 1), 0.0).unwrap();
        for (a, b) in before.amplitudes().iter().zip(after.amplitudes()) {
            assert!((a - b).abs() < 1e-15);
        }
        assert_eq!(before.measure(), after.measure());
    }

    #[test]
    fn tensor_indexing_and_marginals_are_ordered_pairs() {
        let a = state(vec![0.6, 0.8], 2);
        let b = state(vec![0.0, 1.0], 2);
        let ab = PureState::tensor(&a, &b).unwrap();
        assert_eq!(ab.dim(), 4);
        assert_eq!(ab.basis().labels(), &[0, 1, 2, 3]);
        assert!((ab.amplitudes()[1] - 0.6).abs() < 1e-12);
        assert!((ab.amplitudes()[3] - 0.8).abs() < 1e-12);
        let p = ab.measure();
        let left: f64 = (0..2).map(|i| p[i * 2] + p[i * 2 + 1]).sum();
        let right: f64 = (0..2).map(|j| p[j] + p[2 + j]).sum();
        assert!((left - 1.0).abs() < 1e-12);
        assert!((right - 1.0).abs() < 1e-12);
    }

    #[test]
    fn overall_sign_does_not_change_measurements() {
        let s = state(vec![0.6, -0.8], 2);
        let neg = state(vec![-0.6, 0.8], 2);
        assert_eq!(s.measure(), neg.measure());
    }

    #[test]
    fn incompatible_basis_and_invalid_inputs_are_rejected() {
        assert!(TextInput::new("", &[1.0, 0.0], 2).is_err());
        assert!(TextInput::new("x", &[1.0], 2).is_err());
        assert!(TextInput::new("x", &[f32::NAN, 0.0], 2).is_err());
        let head = RotationHead::new(2, 2, &[3]).unwrap();
        let theta = vec![0.0; head.flat_size()];
        let e = [1.0, 0.0];
        let input = TextInput::new("x", &e, 2).unwrap();
        let mut wrong = state(vec![1.0, 0.0], 2);
        wrong.basis = BasisId {
            id: "other".into(),
            labels: vec![0, 1],
        };
        assert!(head.apply(&theta, &input, &wrong, 7).is_err());
        assert!(
            head.apply(&theta, &input, &state(vec![1.0, 0.0], 2), 8)
                .is_err()
        );
        assert!(state(vec![1.0, 0.0], 2).project(2).is_err());
        assert!(PureState::new(vec![1.1, 0.0], basis(2), 0, 0).is_err());
    }

    #[test]
    fn evidence_ids_are_idempotent_and_conflicts_are_errors() {
        let mut store = EvidenceStore::new(2, 2).unwrap();
        let step = vec![1.0, 0.0];
        assert!(store.append("e1", vec![step.clone()], 0).unwrap());
        assert!(!store.append("e1", vec![step.clone()], 0).unwrap());
        assert!(store.append("e1", vec![step.clone()], 1).is_err());
        assert!(store.append("", vec![step.clone()], 0).is_err());
        assert!(store.append("e2", vec![], 0).is_err());
        assert!(store.append("e3", vec![step], 2).is_err());
        assert_eq!(store.checkpoint(), 1);
    }

    #[test]
    fn zero_amplitude_has_infinite_born_loss() {
        let head = RotationHead::new(2, 2, &[2]).unwrap();
        let theta = vec![0.0; head.flat_size()];
        let mut store = EvidenceStore::new(2, 2).unwrap();
        store.append("e", vec![vec![1.0, 0.0]], 1).unwrap();
        let potential = store.potential(head).unwrap();
        let t = Tensor::from_vec(theta, potential.head.flat_size(), &Device::Cpu).unwrap();
        assert!(potential.mean_loss(&t).is_infinite());
        assert!(potential.mean_grad(&t).is_none());
    }

    #[test]
    fn born_gradient_matches_finite_differences() {
        let head = RotationHead::new(2, 2, &[3]).unwrap();
        let mut theta = random_theta(&head, 21);
        let mut store = EvidenceStore::new(2, 2).unwrap();
        store
            .append("a", vec![synthetic_embedding(5, 2)], 0)
            .unwrap();
        store
            .append("b", vec![synthetic_embedding(6, 2)], 1)
            .unwrap();
        store
            .append(
                "seq",
                vec![synthetic_embedding(7, 2), synthetic_embedding(8, 2)],
                1,
            )
            .unwrap();
        let potential = store.potential(head).unwrap();
        let tensor = Tensor::from_vec(theta.clone(), theta.len(), &Device::Cpu).unwrap();
        let grad = potential
            .mean_grad(&tensor)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let h = 1e-3f32;
        for i in [0usize, 2, theta.len() - 1] {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[i] += h;
            minus[i] -= h;
            let lp =
                potential.mean_loss(&Tensor::from_vec(plus, theta.len(), &Device::Cpu).unwrap());
            let lm =
                potential.mean_loss(&Tensor::from_vec(minus, theta.len(), &Device::Cpu).unwrap());
            let fd = (lp - lm) / (2.0 * h as f64);
            assert!(
                (fd - f64::from(grad[i])).abs() < 3e-3,
                "coord {i}: finite difference {fd}, gradient {}",
                grad[i]
            );
        }
        theta[0] += 0.1;
        assert!(theta.iter().all(|v| v.is_finite()));
    }

    struct GaussianTarget {
        mu: Vec<f32>,
    }

    impl Potential for GaussianTarget {
        fn n_terms(&self) -> usize {
            1
        }

        fn mean_loss(&self, theta: &Tensor) -> f64 {
            let values = match theta.flatten_all().and_then(|v| v.to_vec1::<f32>()) {
                Ok(v) => v,
                Err(_) => return f64::INFINITY,
            };
            if values.len() != self.mu.len() {
                return f64::INFINITY;
            }
            0.5 * values
                .iter()
                .zip(&self.mu)
                .map(|(a, b)| {
                    let d = f64::from(*a - *b);
                    d * d
                })
                .sum::<f64>()
        }

        fn mean_grad_with(&self, theta: &Tensor, _norm: NormGrad) -> Option<Tensor> {
            let values = theta.flatten_all().ok()?.to_vec1::<f32>().ok()?;
            let grad: Vec<f32> = values.iter().zip(&self.mu).map(|(a, b)| a - b).collect();
            Tensor::from_vec(grad, values.len(), &Device::Cpu).ok()
        }

        fn label(&self) -> &'static str {
            "gaussian-canary"
        }
    }

    #[test]
    fn full_space_pcn_recovers_a_gaussian_posterior() {
        let target = GaussianTarget {
            mu: vec![1.0, -0.5],
        };
        let sigma = 1.0f64;
        let theta = &mut [0.0f32, 0.0f32];
        let phi = &mut (0.5
            * target.mean_loss(&Tensor::from_vec(theta.to_vec(), 2, &Device::Cpu).unwrap())
            + 0.5 * theta.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>());
        let mut rng = rand::rngs::StdRng::seed_from_u64(41);
        let mut sums = [0.0f64; 2];
        let mut n = 0usize;
        let mut accepted = 0usize;
        for step in 0..30_000 {
            if mh_pcn_full_step(&target, 1.0, sigma, 0.6, theta, phi, &mut rng) {
                accepted += 1;
            }
            if step >= 3_000 && step % 10 == 0 {
                for (s, t) in sums.iter_mut().zip(theta.iter()) {
                    *s += f64::from(*t);
                }
                n += 1;
            }
        }
        assert!(accepted > 1_000, "acceptance {accepted}");
        // Prior precision 1 and likelihood precision 1 give posterior
        // mean mu/2 and precision 2.
        for (mean, mu) in sums.into_iter().map(|s| s / n as f64).zip(target.mu) {
            assert!(
                (mean - 0.5 * f64::from(mu)).abs() < 0.08,
                "mean {mean} for mu {mu}"
            );
        }
    }

    #[test]
    fn partition_and_permutation_preserve_the_log_target() {
        let head = RotationHead::new(2, 2, &[3]).unwrap();
        let theta = random_theta(&head, 31);
        let events = vec![
            Event {
                id: "a".into(),
                steps: vec![synthetic_embedding(11, 2)],
                observed: 0,
            },
            Event {
                id: "b".into(),
                steps: vec![synthetic_embedding(12, 2)],
                observed: 1,
            },
            Event {
                id: "c".into(),
                steps: vec![synthetic_embedding(13, 2)],
                observed: 0,
            },
        ];
        let base = BornPotential::new(head.clone(), events.clone()).unwrap();
        let permuted = BornPotential::new(
            head.clone(),
            vec![events[2].clone(), events[0].clone(), events[1].clone()],
        )
        .unwrap();
        let a = Tensor::from_vec(theta.clone(), theta.len(), &Device::Cpu).unwrap();
        assert!((base.mean_loss(&a) - permuted.mean_loss(&a)).abs() < 1e-12);

        // Two non-overlapping partitions have the same total, and the new
        // batch's summed NLL is exactly the negative-log increment.
        let first = BornPotential::new(head.clone(), events[..2].to_vec()).unwrap();
        let second = BornPotential::new(head.clone(), events[2..].to_vec()).unwrap();
        let sum = 2.0 * first.mean_loss(&a) + 1.0 * second.mean_loss(&a);
        assert!((sum - 3.0 * base.mean_loss(&a)).abs() < 1e-12);
    }

    #[test]
    fn checkpoint_reopens_to_the_same_prediction() {
        let head = RotationHead::new(2, 2, &[2]).unwrap();
        let config = SessionConfig {
            sigma: 0.5,
            lambda: 1.0,
            beta: 0.3,
            burn: 5,
            steps: 25,
            thin: 2,
            chains: 2,
            draws: 4,
            seed: 91,
        };
        let mut session = RotationSession::new(head, config).unwrap();
        session
            .append_event("e1", vec![vec![1.0, 0.0]], 0)
            .unwrap();
        assert!(!session
            .append_event("e1", vec![vec![1.0, 0.0]], 0)
            .unwrap());
        let report = session.update().unwrap();
        assert_eq!(report.checkpoint, 1);
        assert!(report.accepted <= report.proposals);
        let input = TextInput::new("held out", &[0.4, -0.2], 2).unwrap();
        let before = session.predict(&input).unwrap();
        assert_eq!(before.probabilities.len(), 2);
        assert!(before.states.iter().all(|s| s.draw() < 4));

        let path = std::env::temp_dir().join(format!(
            "pcnl_rotation_checkpoint_{}.json",
            std::process::id()
        ));
        session.save(&path).unwrap();
        let stale = path.with_extension("tmp.999999");
        std::fs::write(&stale, b"not a checkpoint").unwrap();
        let mut reopened = RotationSession::load(&path).unwrap();
        assert_eq!(reopened.checkpoint(), 1);
        assert_eq!(reopened.updates(), 1);
        assert_eq!(reopened.retained_draws(), session.retained_draws());
        let after = reopened.predict(&input).unwrap();
        assert_eq!(before.probabilities, after.probabilities);
        for (a, b) in before.states.iter().zip(&after.states) {
            assert_eq!(a.amplitudes(), b.amplitudes());
            assert_eq!(a.basis(), b.basis());
        }
        std::fs::remove_file(&stale).ok();
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn predictive_is_a_mixture_not_a_normalized_mean_amplitude() {
        let a = state(vec![1.0, 0.0], 2);
        let b = state(vec![0.0, 1.0], 2);
        let p = predictive(&[a, b], 2).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[1] - 0.5).abs() < 1e-12);
    }
}
