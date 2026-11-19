//! A frozen text encoder plus a trainable categorical head.
//!
//! The encoder is not part of the posterior.  It produces a fixed
//! `N × D` embedding table — here UltraBERT-Embed-150M's CLS vectors,
//! L2-normalised — and the only parameters this module knows about are the
//! head's weights and biases.  The head is a multi-layer perceptron:
//!
//! ```text
//! p_i  = softmax(W_L tanh(… tanh(W_1 e_i + b_1) …) + b_L)
//! loss = NLL, or the RLCD proper score Σ_c (p_ic − y_ic)²
//! ```
//!
//! It sits on the `Potential` seam the rest of the crate samples:
//! [`crate::target::SubspaceTarget`] lifts a `z` through an
//! [`crate::subspace::AffineSubspace`] and MH-pCN moves it with the exact
//! target value, so the head's posterior can be explored without touching
//! the 150M-parameter encoder.  `LossKind::Brier` is a **generalized Bayes**
//! target, not a literal likelihood; it is the score the RLCD track optimizes
//! and is strictly proper.  [`EmbeddingTable::calibration_split`] holds rows out
//! of the fit set and optionally corrupts their labels; [`CalibratedMlpPotential`]
//! combines NLL on the fit rows with Brier on that held-out calibration side.
//! Freezing the encoder is a *different* posterior from one that integrates its
//! weights, and the reports say so.
//!
//! The fixture format is one safetensors file with:
//!
//! * `embeddings` `[N, D]` f32 — fixed, finite encoder outputs;
//! * `labels` `[N]` u32 or i64 — class indices in `0..C`;
//! * `split` `[N]` u8 — `0` is train, `1` is test.
//!
//! No preprocessing is hidden here: the caller supplies normalised
//! embeddings and an explicit split, and the loader rejects non-finite
//! values, out-of-range labels and an empty side rather than silently
//! substituting zeros.

use crate::normalize::NormGrad;
use crate::target::Potential;
use candle_core::{D, Device, Tensor, Var};
use safetensors::tensor::{Dtype, SafeTensors};
use std::path::Path;

#[derive(Debug)]
pub enum EmbeddingError {
    Io(String),
    Format(String),
    Shape(String),
    Value(String),
    Candle(String),
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m) => write!(f, "embedding fixture I/O: {m}"),
            Self::Format(m) => write!(f, "embedding fixture format: {m}"),
            Self::Shape(m) => write!(f, "embedding fixture shape: {m}"),
            Self::Value(m) => write!(f, "embedding fixture value: {m}"),
            Self::Candle(m) => write!(f, "embedding tensor: {m}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

fn f32s(t: &safetensors::tensor::TensorView<'_>, name: &str) -> Result<Vec<f32>, EmbeddingError> {
    if t.dtype() != Dtype::F32 {
        return Err(EmbeddingError::Format(format!(
            "`{name}` is {:?}, expected F32",
            t.dtype()
        )));
    }
    Ok(t.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn u32s(t: &safetensors::tensor::TensorView<'_>, name: &str) -> Result<Vec<u32>, EmbeddingError> {
    match t.dtype() {
        Dtype::U32 => Ok(t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()),
        Dtype::I64 => {
            let mut out = Vec::with_capacity(t.data().len() / 8);
            for c in t.data().as_chunks::<8>().0 {
                let v = i64::from_le_bytes(*c);
                if !(0..=u32::MAX as i64).contains(&v) {
                    return Err(EmbeddingError::Value(format!(
                        "`{name}` contains {v}, outside the u32 label range"
                    )));
                }
                out.push(v as u32);
            }
            Ok(out)
        }
        other => Err(EmbeddingError::Format(format!(
            "`{name}` is {other:?}, expected U32 or I64"
        ))),
    }
}

/// A split of the frozen table.  Both sides are materialised separately so a
/// likelihood evaluation never slices a tensor at run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Test,
}

/// The head's scalar objective.
///
/// `Nll` is the ordinary categorical likelihood.  `Brier` is the RLCD proper
/// score `Σ_c (p_c − y_c)²`; using it as the negative log target gives a
/// **generalized** posterior, not a literal likelihood, and its predictive is
/// pulled toward calibrated probabilities because the Brier risk is strictly
/// proper.  The reports label which objective produced a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossKind {
    Nll,
    Brier,
}

impl LossKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Nll => "nll",
            Self::Brier => "brier",
        }
    }
}

/// Fixed encoder outputs and labels, split once at load.
#[derive(Clone, Debug)]
pub struct EmbeddingTable {
    dim: usize,
    classes: usize,
    train_x: Vec<f32>,
    train_y: Vec<u32>,
    test_x: Vec<f32>,
    test_y: Vec<u32>,
}

impl EmbeddingTable {
    /// Read and validate a safetensors fixture (see the module docs).
    pub fn load(path: &Path) -> Result<Self, EmbeddingError> {
        let buf = std::fs::read(path)
            .map_err(|e| EmbeddingError::Io(format!("read {}: {e}", path.display())))?;
        let st = SafeTensors::deserialize(&buf)
            .map_err(|e| EmbeddingError::Format(format!("deserialize: {e}")))?;
        let embeddings = st
            .tensor("embeddings")
            .map_err(|e| EmbeddingError::Format(format!("missing `embeddings`: {e}")))?;
        let labels = st
            .tensor("labels")
            .map_err(|e| EmbeddingError::Format(format!("missing `labels`: {e}")))?;
        let split = st
            .tensor("split")
            .map_err(|e| EmbeddingError::Format(format!("missing `split`: {e}")))?;
        if embeddings.shape().len() != 2 {
            return Err(EmbeddingError::Shape(format!(
                "`embeddings` must be [N, D], got {:?}",
                embeddings.shape()
            )));
        }
        let n = embeddings.shape()[0];
        let dim = embeddings.shape()[1];
        if n == 0 || dim == 0 {
            return Err(EmbeddingError::Shape("empty embedding table".into()));
        }
        if labels.shape() != [n] || split.shape() != [n] {
            return Err(EmbeddingError::Shape(format!(
                "expected labels/split [{n}], got {:?}/{:?}",
                labels.shape(),
                split.shape()
            )));
        }
        let x = f32s(&embeddings, "embeddings")?;
        let y = u32s(&labels, "labels")?;
        if split.dtype() != Dtype::U8 {
            return Err(EmbeddingError::Format(format!(
                "`split` is {:?}, expected U8",
                split.dtype()
            )));
        }
        let which = split.data().to_vec();
        if x.iter().any(|v| !v.is_finite()) {
            return Err(EmbeddingError::Value("non-finite embedding".into()));
        }
        let classes = y.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);
        if classes == 0 {
            return Err(EmbeddingError::Value("no labels".into()));
        }
        if y.iter().any(|&v| v as usize >= classes) {
            return Err(EmbeddingError::Value("label outside 0..C".into()));
        }
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();
        for i in 0..n {
            match which[i] {
                0 => {
                    train_x.extend_from_slice(&x[i * dim..(i + 1) * dim]);
                    train_y.push(y[i]);
                }
                1 => {
                    test_x.extend_from_slice(&x[i * dim..(i + 1) * dim]);
                    test_y.push(y[i]);
                }
                other => {
                    return Err(EmbeddingError::Value(format!(
                        "split[{i}] = {other}, expected 0 or 1"
                    )));
                }
            }
        }
        if train_y.is_empty() || test_y.is_empty() {
            return Err(EmbeddingError::Value(
                "both train and test splits must be non-empty".into(),
            ));
        }
        Ok(Self {
            dim,
            classes,
            train_x,
            train_y,
            test_x,
            test_y,
        })
    }

    /// Small in-memory table, used by the finite checks.  `rows` are
    /// `(embedding, label, is_test)`.
    pub fn from_rows(
        dim: usize,
        classes: usize,
        rows: &[(Vec<f32>, u32, bool)],
    ) -> Result<Self, EmbeddingError> {
        if dim == 0 || classes == 0 {
            return Err(EmbeddingError::Shape("dim and classes must be > 0".into()));
        }
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();
        for (row, class, test) in rows {
            if row.len() != dim {
                return Err(EmbeddingError::Shape(format!(
                    "row has {} values, expected {dim}",
                    row.len()
                )));
            }
            if *class as usize >= classes {
                return Err(EmbeddingError::Value(format!(
                    "label {class} outside 0..{classes}"
                )));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(EmbeddingError::Value("non-finite embedding".into()));
            }
            if *test {
                test_x.extend_from_slice(row);
                test_y.push(*class);
            } else {
                train_x.extend_from_slice(row);
                train_y.push(*class);
            }
        }
        if train_y.is_empty() || test_y.is_empty() {
            return Err(EmbeddingError::Value(
                "both train and test splits must be non-empty".into(),
            ));
        }
        Ok(Self {
            dim,
            classes,
            train_x,
            train_y,
            test_x,
            test_y,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    pub fn n_train(&self) -> usize {
        self.train_y.len()
    }

    pub fn n_test(&self) -> usize {
        self.test_y.len()
    }

    /// A stratified training subset, `per_class` rows per class, with the test
    /// split unchanged.  Used to put a deliberately small labelled set under
    /// the head posterior; `per_class` must not exceed the smallest class.
    pub fn train_subset(&self, per_class: usize, seed: u64) -> Result<Self, EmbeddingError> {
        if per_class == 0 {
            return Err(EmbeddingError::Value("per_class must be > 0".into()));
        }
        use rand::seq::SliceRandom;
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed ^ 0x7a11_5eed);
        let n = self.train_y.len();
        let mut take = vec![false; n];
        let mut sampled = false;
        for class in 0..self.classes {
            let mut ids: Vec<usize> = (0..n)
                .filter(|&i| self.train_y[i] as usize == class)
                .collect();
            if ids.is_empty() {
                continue;
            }
            if ids.len() < per_class {
                return Err(EmbeddingError::Value(format!(
                    "class {class} has {} training rows, fewer than {per_class}",
                    ids.len()
                )));
            }
            ids.shuffle(&mut rng);
            for &i in &ids[..per_class] {
                take[i] = true;
            }
            sampled = true;
        }
        if !sampled {
            return Err(EmbeddingError::Value("no training rows to sample".into()));
        }
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        for i in 0..n {
            if take[i] {
                train_x.extend_from_slice(&self.train_x[i * self.dim..(i + 1) * self.dim]);
                train_y.push(self.train_y[i]);
            }
        }
        Ok(Self {
            dim: self.dim,
            classes: self.classes,
            train_x,
            train_y,
            test_x: self.test_x.clone(),
            test_y: self.test_y.clone(),
        })
    }

    /// Concatenate the training rows of two compatible tables.  The test split
    /// is copied from `self`; replay uses this to keep a bounded set of old
    /// examples beside the current task.
    pub fn concat_train(&self, other: &Self) -> Result<Self, EmbeddingError> {
        if self.dim != other.dim || self.classes != other.classes {
            return Err(EmbeddingError::Shape(
                "cannot concatenate tables with different dimensions or classes".into(),
            ));
        }
        let mut train_x = self.train_x.clone();
        train_x.extend_from_slice(&other.train_x);
        let mut train_y = self.train_y.clone();
        train_y.extend_from_slice(&other.train_y);
        Ok(Self {
            dim: self.dim,
            classes: self.classes,
            train_x,
            train_y,
            test_x: self.test_x.clone(),
            test_y: self.test_y.clone(),
        })
    }

    /// Training rows whose labels are in `classes`, with the full test split
    /// unchanged.  Used for sequential class/task batches: the likelihood sees
    /// only those labels, while every metric still covers all test classes.
    pub fn training_for_classes(&self, classes: &[u32]) -> Result<Self, EmbeddingError> {
        if classes.is_empty() {
            return Err(EmbeddingError::Value("class list is empty".into()));
        }
        if let Some(&bad) = classes.iter().find(|&&c| c as usize >= self.classes) {
            return Err(EmbeddingError::Value(format!(
                "class {bad} outside 0..{}",
                self.classes
            )));
        }
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        for (i, &y) in self.train_y.iter().enumerate() {
            if classes.contains(&y) {
                train_x.extend_from_slice(&self.train_x[i * self.dim..(i + 1) * self.dim]);
                train_y.push(y);
            }
        }
        if train_y.is_empty() {
            return Err(EmbeddingError::Value(
                "no training rows for the requested classes".into(),
            ));
        }
        Ok(Self {
            dim: self.dim,
            classes: self.classes,
            train_x,
            train_y,
            test_x: self.test_x.clone(),
            test_y: self.test_y.clone(),
        })
    }

    /// Re-partition all rows into a new stratified train/test split with
    /// `test_per_class` rows per class on the test side.  The original fixture
    /// has 60 train + 30 test rows per class; this method pools them so a
    /// calibration result is not tied to one 120-row test set.
    pub fn resplit(&self, test_per_class: usize, seed: u64) -> Result<Self, EmbeddingError> {
        if test_per_class == 0 {
            return Err(EmbeddingError::Value("test_per_class must be > 0".into()));
        }
        use rand::seq::SliceRandom;
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5e9_11e7);
        let n_train = self.train_y.len();
        let n = n_train + self.test_y.len();
        let mut x = Vec::with_capacity(n * self.dim);
        let mut y = Vec::with_capacity(n);
        for split in [Split::Train, Split::Test] {
            x.extend_from_slice(self.x(split));
            y.extend_from_slice(self.y(split));
        }
        let mut is_test = vec![false; n];
        for class in 0..self.classes {
            let mut ids: Vec<usize> = (0..n).filter(|&i| y[i] as usize == class).collect();
            if ids.len() <= test_per_class {
                return Err(EmbeddingError::Value(format!(
                    "class {class} has {} rows; need more than {test_per_class} to keep a train row",
                    ids.len()
                )));
            }
            ids.shuffle(&mut rng);
            for &i in &ids[..test_per_class] {
                is_test[i] = true;
            }
        }
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();
        for i in 0..n {
            let row = &x[i * self.dim..(i + 1) * self.dim];
            if is_test[i] {
                test_x.extend_from_slice(row);
                test_y.push(y[i]);
            } else {
                train_x.extend_from_slice(row);
                train_y.push(y[i]);
            }
        }
        Ok(Self {
            dim: self.dim,
            classes: self.classes,
            train_x,
            train_y,
            test_x,
            test_y,
        })
    }

    /// Split the training rows into a fit set and a held-out calibration set,
    /// then corrupt the calibration labels with probability `label_noise`.
    /// The test split is copied unchanged to both sides, so every metric is
    /// still computed on the clean test labels.  Returns
    /// `(fit, calibration, n_flipped)`; `per_class` calibration rows are taken
    /// per class and the fit side keeps the rest.
    pub fn calibration_split(
        &self,
        per_class: usize,
        seed: u64,
        label_noise: f64,
    ) -> Result<(Self, Self, usize), EmbeddingError> {
        if per_class == 0 {
            return Err(EmbeddingError::Value("per_class must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&label_noise) {
            return Err(EmbeddingError::Value(
                "label_noise must be in [0, 1]".into(),
            ));
        }
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed ^ 0xca11_b5eed);
        let n = self.train_y.len();
        let mut is_cal = vec![false; n];
        for class in 0..self.classes {
            let mut ids: Vec<usize> = (0..n)
                .filter(|&i| self.train_y[i] as usize == class)
                .collect();
            if ids.len() <= per_class {
                return Err(EmbeddingError::Value(format!(
                    "class {class} has {} training rows; need more than {per_class} to keep a fit row",
                    ids.len()
                )));
            }
            ids.shuffle(&mut rng);
            for &i in &ids[..per_class] {
                is_cal[i] = true;
            }
        }
        let mut fit_x = Vec::new();
        let mut fit_y = Vec::new();
        let mut cal_x = Vec::new();
        let mut cal_y = Vec::new();
        let mut flipped = 0usize;
        for i in 0..n {
            let row = &self.train_x[i * self.dim..(i + 1) * self.dim];
            if is_cal[i] {
                let mut y = self.train_y[i];
                if self.classes > 1 && rng.random::<f64>() < label_noise {
                    let offset = rng.random_range(1..self.classes) as u32;
                    y = (y + offset) % self.classes as u32;
                    flipped += 1;
                }
                cal_x.extend_from_slice(row);
                cal_y.push(y);
            } else {
                fit_x.extend_from_slice(row);
                fit_y.push(self.train_y[i]);
            }
        }
        let fit = Self {
            dim: self.dim,
            classes: self.classes,
            train_x: fit_x,
            train_y: fit_y,
            test_x: self.test_x.clone(),
            test_y: self.test_y.clone(),
        };
        let cal = Self {
            dim: self.dim,
            classes: self.classes,
            train_x: cal_x,
            train_y: cal_y,
            test_x: self.test_x.clone(),
            test_y: self.test_y.clone(),
        };
        Ok((fit, cal, flipped))
    }

    pub fn x(&self, split: Split) -> &[f32] {
        match split {
            Split::Train => &self.train_x,
            Split::Test => &self.test_x,
        }
    }

    pub fn y(&self, split: Split) -> &[u32] {
        match split {
            Split::Train => &self.train_y,
            Split::Test => &self.test_y,
        }
    }
}

/// `(accuracy %, mean NLL, mean Brier)` from row-major probabilities.
fn metrics_from_probs(
    table: &EmbeddingTable,
    p: &[f32],
    split: Split,
) -> Result<(f64, f64, f64), EmbeddingError> {
    let y = table.y(split);
    let c = table.classes;
    let mut hits = 0usize;
    let mut nll = 0.0f64;
    let mut brier = 0.0f64;
    for (i, &t) in y.iter().enumerate() {
        let row = &p[i * c..(i + 1) * c];
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(j, _)| j)
            .unwrap_or(0);
        if best == t as usize {
            hits += 1;
        }
        nll -= (row[t as usize] as f64).max(1e-12).ln();
        brier += crate::readout::brier(
            &row.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            t as usize,
        );
    }
    let n = y.len() as f64;
    Ok((100.0 * hits as f64 / n, nll / n, brier / n))
}

/// A multi-layer perceptron head: `D → h₁ → … → hₖ → C`, with `tanh`
/// activations and every weight and bias in the sampled vector.
///
/// This is the only head in `examples/ultrabert_head_pcnl`.  The layout is
/// row-major weights then biases per layer, in order, so `flat_size` is
/// `Σ(in·out + out)`.  No layer is frozen: the pCNL target sees the whole
/// flat vector.
#[derive(Clone, Debug)]
pub struct MlpHeadPotential {
    dim: usize,
    classes: usize,
    hidden: Vec<usize>,
    loss: LossKind,
    x: Vec<f32>,
    y: Vec<u32>,
}

impl MlpHeadPotential {
    pub fn new(table: &EmbeddingTable, hidden: &[usize]) -> Result<Self, EmbeddingError> {
        Self::new_with_loss(table, hidden, LossKind::Nll)
    }

    pub fn new_with_loss(
        table: &EmbeddingTable,
        hidden: &[usize],
        loss: LossKind,
    ) -> Result<Self, EmbeddingError> {
        if hidden.is_empty() {
            return Err(EmbeddingError::Shape(
                "an MLP head needs at least one hidden layer".into(),
            ));
        }
        if hidden.iter().any(|&h| h == 0) {
            return Err(EmbeddingError::Shape("a hidden layer has width 0".into()));
        }
        Ok(Self {
            dim: table.dim,
            classes: table.classes,
            hidden: hidden.to_vec(),
            loss,
            x: table.train_x.clone(),
            y: table.train_y.clone(),
        })
    }

    pub fn loss(&self) -> LossKind {
        self.loss
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    pub fn hidden(&self) -> &[usize] {
        &self.hidden
    }

    /// `(in, out)` for each affine layer, in execution order.
    pub fn layers(&self) -> Vec<(usize, usize)> {
        let mut widths = Vec::with_capacity(self.hidden.len() + 2);
        widths.push(self.dim);
        widths.extend_from_slice(&self.hidden);
        widths.push(self.classes);
        widths.windows(2).map(|w| (w[0], w[1])).collect()
    }

    pub fn flat_size(&self) -> usize {
        self.layers().iter().map(|&(i, o)| i * o + o).sum()
    }

    fn onehot(&self) -> Result<Tensor, EmbeddingError> {
        let mut v = vec![0.0f32; self.y.len() * self.classes];
        for (i, &t) in self.y.iter().enumerate() {
            v[i * self.classes + t as usize] = 1.0;
        }
        Tensor::from_vec(v, (self.y.len(), self.classes), &Device::Cpu)
            .map_err(|e| EmbeddingError::Candle(e.to_string()))
    }

    fn objective(
        &self,
        logits: &Tensor,
        y: &Tensor,
        onehot: &Tensor,
    ) -> candle_core::Result<Tensor> {
        match self.loss {
            LossKind::Nll => candle_nn::loss::cross_entropy(logits, y),
            LossKind::Brier => {
                let p = candle_nn::ops::softmax(logits, D::Minus1)?;
                let n = self.y.len().max(1) as f64;
                p.broadcast_sub(onehot)?
                    .sqr()?
                    .sum_all()?
                    .affine(1.0 / n, 0.0)
            }
        }
    }

    fn tensors(&self) -> Result<(Tensor, Tensor), EmbeddingError> {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(self.x.clone(), (self.y.len(), self.dim), &dev)
            .map_err(|e| EmbeddingError::Candle(e.to_string()))?;
        let y = Tensor::from_vec(self.y.clone(), self.y.len(), &dev)
            .map_err(|e| EmbeddingError::Candle(e.to_string()))?;
        Ok((x, y))
    }

    fn logits(
        theta: &Tensor,
        x: &Tensor,
        hidden: &[usize],
        dim: usize,
        classes: usize,
    ) -> candle_core::Result<Tensor> {
        let mut widths = Vec::with_capacity(hidden.len() + 2);
        widths.push(dim);
        widths.extend_from_slice(hidden);
        widths.push(classes);
        let mut at = 0usize;
        let mut h = x.clone();
        for (i, pair) in widths.windows(2).enumerate() {
            let (fin, fout) = (pair[0], pair[1]);
            let w = theta.narrow(0, at, fin * fout)?.reshape((fin, fout))?;
            at += fin * fout;
            let b = theta.narrow(0, at, fout)?;
            at += fout;
            h = h.matmul(&w)?.broadcast_add(&b)?;
            if i + 1 < widths.len() - 1 {
                h = h.tanh()?;
            }
        }
        Ok(h)
    }
}

/// Softmax probabilities of an [`MlpHeadPotential`] on one split.
pub fn predict_proba_mlp(
    table: &EmbeddingTable,
    theta: &[f32],
    hidden: &[usize],
    split: Split,
) -> Result<Vec<f32>, EmbeddingError> {
    let n = table.y(split).len();
    let pot = MlpHeadPotential::new(table, hidden)?;
    let expected = pot.flat_size();
    if theta.len() != expected {
        return Err(EmbeddingError::Shape(format!(
            "MLP head has {} values, expected {expected}",
            theta.len()
        )));
    }
    let dev = Device::Cpu;
    let x = Tensor::from_vec(table.x(split).to_vec(), (n, table.dim), &dev)
        .map_err(|e| EmbeddingError::Candle(e.to_string()))?;
    let t = Tensor::from_vec(theta.to_vec(), expected, &dev)
        .map_err(|e| EmbeddingError::Candle(e.to_string()))?;
    let logits = MlpHeadPotential::logits(&t, &x, hidden, table.dim, table.classes)
        .map_err(|e| EmbeddingError::Candle(e.to_string()))?;
    let probs = candle_nn::ops::softmax(&logits, D::Minus1)
        .map_err(|e| EmbeddingError::Candle(e.to_string()))?;
    probs
        .flatten_all()
        .and_then(|v| v.to_vec1::<f32>())
        .map_err(|e| EmbeddingError::Candle(e.to_string()))
}

/// `(accuracy %, mean NLL, mean Brier)` of an MLP head on one split.
pub fn metrics_mlp(
    table: &EmbeddingTable,
    theta: &[f32],
    hidden: &[usize],
    split: Split,
) -> Result<(f64, f64, f64), EmbeddingError> {
    let p = predict_proba_mlp(table, theta, hidden, split)?;
    metrics_from_probs(table, &p, split)
}

impl Potential for MlpHeadPotential {
    fn n_terms(&self) -> usize {
        self.y.len()
    }

    fn mean_loss(&self, theta: &Tensor) -> f64 {
        let (x, y) = match self.tensors() {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        };
        let onehot = match self.onehot() {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        };
        MlpHeadPotential::logits(theta, &x, &self.hidden, self.dim, self.classes)
            .and_then(|logits| self.objective(&logits, &y, &onehot))
            .and_then(|l| l.to_scalar::<f32>())
            .map(|l| l as f64)
            .ok()
            .filter(|l| l.is_finite())
            .unwrap_or(f64::INFINITY)
    }

    fn mean_grad_with(&self, theta: &Tensor, _norm: NormGrad) -> Option<Tensor> {
        let (x, y) = self.tensors().ok()?;
        let onehot = self.onehot().ok()?;
        let var = Var::from_tensor(theta).ok()?;
        let logits =
            MlpHeadPotential::logits(var.as_tensor(), &x, &self.hidden, self.dim, self.classes)
                .ok()?;
        let loss = self.objective(&logits, &y, &onehot).ok()?;
        let value = loss.to_scalar::<f32>().ok()?;
        if !value.is_finite() {
            return None;
        }
        let grads = loss.backward().ok()?;
        grads.get(var.as_tensor())?.flatten_all().ok()
    }

    fn label(&self) -> &'static str {
        "ultrabert-mlp-head"
    }
}

/// NLL on a fit split plus the Brier proper score on a held-out calibration
/// split.  The two terms are averaged by their row counts, so the target is
/// one potential over the same flat head vector; the calibration rows never
/// enter the fit likelihood.
#[derive(Clone, Debug)]
pub struct CalibratedMlpPotential {
    fit: MlpHeadPotential,
    cal: MlpHeadPotential,
}

impl CalibratedMlpPotential {
    pub fn new(
        fit_table: &EmbeddingTable,
        cal_table: &EmbeddingTable,
        hidden: &[usize],
    ) -> Result<Self, EmbeddingError> {
        if fit_table.dim != cal_table.dim
            || fit_table.classes != cal_table.classes
            || fit_table.train_y.is_empty()
            || cal_table.train_y.is_empty()
        {
            return Err(EmbeddingError::Shape(
                "calibration and fit tables must be non-empty and compatible".into(),
            ));
        }
        Ok(Self {
            fit: MlpHeadPotential::new_with_loss(fit_table, hidden, LossKind::Nll)?,
            cal: MlpHeadPotential::new_with_loss(cal_table, hidden, LossKind::Brier)?,
        })
    }

    pub fn flat_size(&self) -> usize {
        self.fit.flat_size()
    }
}

impl Potential for CalibratedMlpPotential {
    fn n_terms(&self) -> usize {
        self.fit.n_terms() + self.cal.n_terms()
    }

    fn mean_loss(&self, theta: &Tensor) -> f64 {
        let n_fit = self.fit.n_terms() as f64;
        let n_cal = self.cal.n_terms() as f64;
        let total = n_fit + n_cal;
        (n_fit * self.fit.mean_loss(theta) + n_cal * self.cal.mean_loss(theta)) / total
    }

    fn mean_grad_with(&self, theta: &Tensor, norm: NormGrad) -> Option<Tensor> {
        let n_fit = self.fit.n_terms() as f64;
        let n_cal = self.cal.n_terms() as f64;
        let total = n_fit + n_cal;
        let g_fit = self
            .fit
            .mean_grad_with(theta, norm)?
            .affine(n_fit / total, 0.0)
            .ok()?;
        let g_cal = self
            .cal
            .mean_grad_with(theta, norm)?
            .affine(n_cal / total, 0.0)
            .ok()?;
        g_fit.add(&g_cal).ok()
    }

    fn label(&self) -> &'static str {
        "ultrabert-mlp-fit-nll-cal-brier"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> EmbeddingTable {
        EmbeddingTable::from_rows(
            2,
            3,
            &[
                (vec![1.0, 0.0], 0, false),
                (vec![0.0, 1.0], 1, false),
                (vec![1.0, 1.0], 2, false),
                (vec![1.0, 0.0], 0, true),
                (vec![0.0, 1.0], 1, true),
            ],
        )
        .unwrap()
    }

    #[test]
    fn mlp_zero_head_is_uniform_and_gradient_matches_fd() {
        let t = tiny();
        let hidden = [4usize, 3];
        let pot = MlpHeadPotential::new(&t, &hidden).unwrap();
        assert_eq!(pot.layers(), vec![(2, 4), (4, 3), (3, 3)]);
        assert_eq!(pot.flat_size(), 2 * 4 + 4 + 4 * 3 + 3 + 3 * 3 + 3);
        let zero = vec![0.0f32; pot.flat_size()];
        let p = predict_proba_mlp(&t, &zero, &hidden, Split::Test).unwrap();
        for row in p.chunks(3) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            for &v in row {
                assert!((v - 1.0 / 3.0).abs() < 1e-6, "{v}");
            }
        }
        let mut theta = zero.clone();
        for (i, v) in theta.iter_mut().enumerate() {
            *v = 0.02 * (i as f32 * 0.41).sin();
        }
        let tensor = Tensor::from_vec(theta.clone(), &[theta.len()], &Device::Cpu).unwrap();
        let g = pot
            .mean_grad_with(&tensor, NormGrad::Exact)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let h = 1e-3f32;
        for i in [0usize, 5, 20, pot.flat_size() - 1] {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[i] += h;
            minus[i] -= h;
            let lp = pot.mean_loss(&Tensor::from_vec(plus, &[theta.len()], &Device::Cpu).unwrap());
            let lm = pot.mean_loss(&Tensor::from_vec(minus, &[theta.len()], &Device::Cpu).unwrap());
            let fd = (lp - lm) / (2.0 * h as f64);
            assert!(
                (fd - g[i] as f64).abs() < 3e-4,
                "coord {i}: finite difference {fd}, gradient {}",
                g[i]
            );
        }
    }

    #[test]
    fn brier_objective_and_gradient_are_finite() {
        let t = tiny();
        let hidden = [4usize, 3];
        let pot = MlpHeadPotential::new_with_loss(&t, &hidden, LossKind::Brier).unwrap();
        let zero = vec![0.0f32; pot.flat_size()];
        let at_zero =
            pot.mean_loss(&Tensor::from_vec(zero.clone(), &[zero.len()], &Device::Cpu).unwrap());
        assert!((at_zero - 2.0 / 3.0).abs() < 1e-6, "{at_zero}");
        let mut theta = vec![0.0f32; pot.flat_size()];
        for (i, v) in theta.iter_mut().enumerate() {
            *v = 0.02 * (i as f32 * 0.29).cos();
        }
        let tensor = Tensor::from_vec(theta.clone(), &[theta.len()], &Device::Cpu).unwrap();
        let g = pot
            .mean_grad_with(&tensor, NormGrad::Exact)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let h = 1e-3f32;
        for i in [0usize, 9, pot.flat_size() - 1] {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[i] += h;
            minus[i] -= h;
            let lp = pot.mean_loss(&Tensor::from_vec(plus, &[theta.len()], &Device::Cpu).unwrap());
            let lm = pot.mean_loss(&Tensor::from_vec(minus, &[theta.len()], &Device::Cpu).unwrap());
            let fd = (lp - lm) / (2.0 * h as f64);
            assert!(
                (fd - g[i] as f64).abs() < 3e-4,
                "coord {i}: {fd} vs {}",
                g[i]
            );
        }
    }

    #[test]
    fn full_target_matches_restricted_at_zero_complement() {
        use crate::subspace::{AffineSubspace, Basis};
        use crate::target::{FullPotential, PotentialFullTarget, SubspaceTarget, ZPotential};
        let t = tiny();
        let pot = MlpHeadPotential::new(&t, &[4, 3]).unwrap();
        let p = pot.flat_size();
        let subspace =
            AffineSubspace::isotropic(vec![0.0; p], Basis::random(p, 2, 7).unwrap(), 0.05).unwrap();
        let restricted = SubspaceTarget::with_potential(subspace.clone(), 1.0, pot.clone());
        let full = PotentialFullTarget::new(subspace, 1.0, pot);
        let w = vec![0.0f32; p];
        for z in [vec![0.0, 0.0], vec![0.3, -0.7], vec![-1.2, 0.4]] {
            let a = restricted.phi(&z);
            let b = full.phi(&z, &w);
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn train_subset_is_balanced() {
        let t = tiny();
        let s = t.train_subset(1, 3).unwrap();
        assert_eq!(s.n_train(), 3);
        assert_eq!(s.n_test(), t.n_test());
        let mut seen = std::collections::BTreeSet::new();
        for &y in s.y(Split::Train) {
            seen.insert(y);
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn concat_train_keeps_both_sides_and_the_test_split() {
        let t = tiny();
        let a = t.training_for_classes(&[0]).unwrap();
        let b = t.training_for_classes(&[1]).unwrap();
        let both = a.concat_train(&b).unwrap();
        assert_eq!(both.n_train(), 2);
        assert_eq!(both.y(Split::Train), &[0, 1]);
        assert_eq!(both.n_test(), t.n_test());
    }

    #[test]
    fn training_for_classes_keeps_only_the_requested_labels() {
        let t = tiny();
        let one = t.training_for_classes(&[2]).unwrap();
        assert_eq!(one.n_train(), 1);
        assert_eq!(one.y(Split::Train), &[2]);
        assert_eq!(one.n_test(), t.n_test());
        assert!(t.training_for_classes(&[]).is_err());
        assert!(t.training_for_classes(&[9]).is_err());
    }

    #[test]
    fn resplit_is_stratified_and_reproducible() {
        let t = EmbeddingTable::from_rows(
            2,
            2,
            &[
                (vec![1.0, 0.0], 0, false),
                (vec![0.9, 0.1], 0, false),
                (vec![0.8, 0.2], 0, false),
                (vec![0.0, 1.0], 1, false),
                (vec![0.1, 0.9], 1, false),
                (vec![0.2, 0.8], 1, false),
                (vec![1.0, 0.0], 0, true),
                (vec![0.0, 1.0], 1, true),
            ],
        )
        .unwrap();
        let a = t.resplit(2, 13).unwrap();
        let b = t.resplit(2, 13).unwrap();
        assert_eq!(a.n_train(), 4);
        assert_eq!(a.n_test(), 4);
        assert_eq!(a.y(Split::Train), b.y(Split::Train));
        assert_eq!(a.y(Split::Test), b.y(Split::Test));
        assert_eq!(a.y(Split::Test).iter().filter(|&&y| y == 0).count(), 2);
        assert_eq!(a.y(Split::Test).iter().filter(|&&y| y == 1).count(), 2);
        assert!(t.resplit(0, 13).is_err());
        assert!(t.resplit(4, 13).is_err());
    }

    #[test]
    fn calibration_split_holds_out_rows_and_flips_labels() {
        let t = EmbeddingTable::from_rows(
            2,
            2,
            &[
                (vec![1.0, 0.0], 0, false),
                (vec![0.9, 0.1], 0, false),
                (vec![0.8, 0.2], 0, false),
                (vec![0.0, 1.0], 1, false),
                (vec![0.1, 0.9], 1, false),
                (vec![0.2, 0.8], 1, false),
                (vec![1.0, 0.0], 0, true),
            ],
        )
        .unwrap();
        let (fit, cal, flipped) = t.calibration_split(1, 7, 1.0).unwrap();
        assert_eq!(fit.n_train(), 4);
        assert_eq!(cal.n_train(), 2);
        assert_eq!(flipped, 2);
        assert_eq!(fit.n_test(), t.n_test());
        assert_eq!(cal.n_test(), t.n_test());
        assert_eq!(cal.y(Split::Test), t.y(Split::Test));
        assert!(cal.y(Split::Train).iter().all(|&y| y == 1 || y == 0));
    }

    #[test]
    fn calibration_split_rejects_an_empty_fit_side() {
        let t = tiny();
        assert!(t.calibration_split(1, 7, 0.0).is_err());
        assert!(t.calibration_split(0, 7, 0.0).is_err());
        assert!(t.calibration_split(1, 7, 1.5).is_err());
    }

    #[test]
    fn calibrated_potential_averages_fit_nll_and_cal_brier() {
        let t = tiny();
        let hidden = [4usize, 3];
        let fit = MlpHeadPotential::new_with_loss(&t, &hidden, LossKind::Nll).unwrap();
        let cal = MlpHeadPotential::new_with_loss(&t, &hidden, LossKind::Brier).unwrap();
        let both = CalibratedMlpPotential::new(&t, &t, &hidden).unwrap();
        let theta = Tensor::from_vec(
            (0..both.flat_size())
                .map(|i| 0.01 * (i as f32 * 0.17).sin())
                .collect::<Vec<_>>(),
            both.flat_size(),
            &Device::Cpu,
        )
        .unwrap();
        let expected = 0.5 * fit.mean_loss(&theta) + 0.5 * cal.mean_loss(&theta);
        assert!((both.mean_loss(&theta) - expected).abs() < 1e-6);
    }

    #[test]
    fn a_bad_split_is_rejected() {
        let err = EmbeddingTable::from_rows(
            2,
            2,
            &[(vec![0.0, 0.0], 0, false), (vec![0.0, 0.0], 1, false)],
        );
        assert!(matches!(err, Err(EmbeddingError::Value(_))));
    }
}
