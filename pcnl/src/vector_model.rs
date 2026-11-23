//! R13 §1–§2: a four-block transformer with a raw numerical-vector readout.
//!
//! A fixed English embedding `e ∈ R^D` is linearly projected into four ordered
//! feature vectors of width `W`.  The sequence is passed through four distinct
//! standard attention/MLP blocks and the final position is mapped by an untied
//! affine readout to `f_θ(e) ∈ R^q`.  There is no softmax: coordinates may be
//! negative, zero or larger than one.
//!
//! The sampling target is the additive Gaussian regression objective
//!
//! ```text
//! J(θ) = Σ_i ‖y_i − f_θ(e_i)‖² / (2 s²),      U(θ) = λ J(θ),
//! ```
//!
//! with a fixed diagonal Gaussian prior.  [`VectorRegression`] reports the
//! **mean** loss and its exact gradient on [`crate::target::Potential`]; a
//! caller forms the sum exactly once as `λ · n_terms · mean_loss`.  The
//! optimizer and sampler in this crate both take the gradient from the same
//! function, so they cannot silently optimize different objectives.
//!
//! The CN proposal is the R13 §3 kernel
//!
//! ```text
//! g(θ) = m + ρ(θ−m) − (1−ρ) C ∇U(θ),     θ' ~ N(g, (1−ρ²)C),
//! ```
//!
//! accepted with the full posterior ratio and the forward/reverse Gaussian
//! proposal densities.  A state-independent Bernoulli draw replaces the local
//! move by an independent draw from the prior with the exact independence MH
//! ratio.  Both preserve the same target; neither is an unadjusted optimizer.

use crate::model::{LayerSlices, Variant, layer};
use crate::normalize::NormGrad;
use crate::target::Potential;
use candle_core::{DType, Device, Tensor, Var};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// Shape of the R13 vector head.  `tokens` is the number of ordered feature
/// vectors extracted from one embedding; the active plan fixes it at four.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VectorShapes {
    pub input: usize,
    pub tokens: usize,
    pub width: usize,
    pub ff: usize,
    pub layers: usize,
    pub output: usize,
}

impl VectorShapes {
    /// The plan's small design: four positions, width 16, four distinct blocks.
    pub fn new(input: usize, output: usize) -> candle_core::Result<Self> {
        Self::with_width(input, output, 16, 32, 4, 4)
    }

    pub fn with_width(
        input: usize,
        output: usize,
        width: usize,
        ff: usize,
        layers: usize,
        tokens: usize,
    ) -> candle_core::Result<Self> {
        if input == 0 || output == 0 || width == 0 || ff == 0 || layers == 0 || tokens == 0 {
            return Err(candle_core::Error::Msg(
                "vector model dimensions must be non-zero".into(),
            ));
        }
        Ok(Self {
            input,
            tokens,
            width,
            ff,
            layers,
            output,
        })
    }

    pub fn flat_size(&self) -> usize {
        let projection = self.input * self.tokens * self.width;
        let block = 4 * self.width * self.width + 2 * self.width * self.ff;
        let readout = self.width * self.output + self.output;
        projection + self.layers * block + readout
    }

    /// `(start, len, shape)` in a fixed order: projection, four blocks, readout,
    /// readout bias.  Every matrix is `[in, out]`.
    pub fn layout(&self) -> Vec<(usize, usize, Vec<usize>)> {
        let mut out = Vec::new();
        let mut at = 0usize;
        let mut push = |shape: Vec<usize>| {
            let len: usize = shape.iter().product();
            out.push((at, len, shape));
            at += len;
        };
        push(vec![self.input, self.tokens * self.width]);
        for _ in 0..self.layers {
            push(vec![self.width, self.width]); // wq
            push(vec![self.width, self.width]); // wk
            push(vec![self.width, self.width]); // wv
            push(vec![self.width, self.width]); // wo
            push(vec![self.width, self.ff]); // w1
            push(vec![self.ff, self.width]); // w2
        }
        push(vec![self.width, self.output]);
        push(vec![self.output]);
        debug_assert_eq!(at, self.flat_size());
        out
    }

    /// Per-parameter prior scale: matrix entries divide by `sqrt(fan_in)`,
    /// biases stay at one.  This mirrors the rotation head's layer scales.
    pub fn prior_scales(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.flat_size());
        for (_, _, shape) in self.layout() {
            if shape.len() == 2 {
                out.extend(std::iter::repeat((shape[0] as f64).sqrt()).take(shape[0] * shape[1]));
            } else {
                out.extend(std::iter::repeat(1.0).take(shape[0]));
            }
        }
        out
    }
}

struct VectorSlices {
    projection: Tensor,
    layers: Vec<LayerSlices>,
    readout: Tensor,
    bias: Tensor,
}

fn unflatten(theta: &Tensor, shapes: &VectorShapes) -> candle_core::Result<VectorSlices> {
    let mut tensors = Vec::new();
    for (start, len, shape) in shapes.layout() {
        tensors.push(theta.narrow(0, start, len)?.reshape(shape)?);
    }
    let mut it = tensors.into_iter();
    let projection = it.next().expect("layout has a projection");
    let layers = (0..shapes.layers)
        .map(|_| LayerSlices {
            wq: it.next().unwrap(),
            wk: it.next().unwrap(),
            wv: it.next().unwrap(),
            wo: it.next().unwrap(),
            w1: it.next().unwrap(),
            w2: it.next().unwrap(),
        })
        .collect();
    let readout = it.next().unwrap();
    let bias = it.next().unwrap();
    Ok(VectorSlices {
        projection,
        layers,
        readout,
        bias,
    })
}

fn check_embedding(embedding: &Tensor, shapes: &VectorShapes) -> candle_core::Result<Tensor> {
    if embedding.dims() != [shapes.input] {
        return Err(candle_core::Error::Msg(format!(
            "embedding must be [{}], got {:?}",
            shapes.input,
            embedding.dims()
        )));
    }
    embedding.reshape((1, shapes.input))
}

/// Final-position hidden vector `[width]`: the readout's feature vector with the
/// backbone held at `theta`.  Exposed because the readout-conditional posterior
/// is exactly Gaussian in these features.
pub fn last_hidden(
    theta: &Tensor,
    shapes: &VectorShapes,
    embedding: &Tensor,
) -> candle_core::Result<Tensor> {
    if theta.dims() != [shapes.flat_size()] {
        return Err(candle_core::Error::Msg(format!(
            "theta must be [{}], got {:?}",
            shapes.flat_size(),
            theta.dims()
        )));
    }
    let s = unflatten(theta, shapes)?;
    let e = check_embedding(embedding, shapes)?;
    let mut x = e
        .broadcast_matmul(&s.projection)?
        .reshape((1, shapes.tokens, shapes.width))?;
    let tau = (shapes.width as f64).sqrt();
    for w in &s.layers {
        x = layer(&x, w, Variant::Standard, tau, NormGrad::Exact)?;
    }
    x.narrow(1, shapes.tokens - 1, 1)?.squeeze(1)?.squeeze(0)
}

/// `f_θ(e)` for one fixed embedding, shape `[q]`.
pub fn forward(
    theta: &Tensor,
    shapes: &VectorShapes,
    embedding: &Tensor,
) -> candle_core::Result<Tensor> {
    let s = unflatten(theta, shapes)?;
    let last = last_hidden(theta, shapes, embedding)?.reshape((1, shapes.width))?;
    last.broadcast_matmul(&s.readout)?
        .broadcast_add(&s.bias.reshape((1, shapes.output))?)?
        .squeeze(0)
}

/// Flat-vector convenience wrapper around [`forward`].
pub fn predict(
    theta: &[f32],
    shapes: &VectorShapes,
    embedding: &[f32],
) -> candle_core::Result<Vec<f32>> {
    let theta = Tensor::from_vec(theta.to_vec(), shapes.flat_size(), &Device::Cpu)?;
    let e = Tensor::from_vec(embedding.to_vec(), shapes.input, &Device::Cpu)?;
    forward(&theta, shapes, &e)?.to_vec1::<f32>()
}

/// Deterministic prior/working-scale initial vector.
pub fn init_flat(shapes: &VectorShapes, std: f64, seed: u64) -> Tensor {
    let mut rng = StdRng::seed_from_u64(seed);
    let v: Vec<f32> = (0..shapes.flat_size())
        .map(|_| {
            let z: f64 = rng.sample(StandardNormal);
            (z * std) as f32
        })
        .collect();
    Tensor::from_vec(v, shapes.flat_size(), &Device::Cpu).expect("cpu tensor")
}

/// A fixed synthetic teacher on the frozen embedding.  This is the §4.2
/// reproducible regression stand-in until an application supplies a schema.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SyntheticTeacher {
    pub input: usize,
    pub output: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl SyntheticTeacher {
    pub fn new(input: usize, output: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x7631_9a2d);
        let scale = 1.0 / (input as f64).sqrt();
        let weights = (0..input * output)
            .map(|_| {
                let z: f64 = rng.sample(StandardNormal);
                (z * scale) as f32
            })
            .collect();
        // Fixed offsets keep the schema non-unit and make negative coordinates
        // routine rather than a lucky draw.
        let bias = (0..output)
            .map(|j| (j as f32 * 0.75 - 1.5).clamp(-1.5, 1.5))
            .collect();
        Self {
            input,
            output,
            weights,
            bias,
        }
    }

    pub fn predict(&self, embedding: &[f32]) -> Vec<f32> {
        assert_eq!(embedding.len(), self.input);
        (0..self.output)
            .map(|j| {
                let mut v = self.bias[j];
                for (i, &e) in embedding.iter().enumerate() {
                    v += e * self.weights[i * self.output + j];
                }
                v
            })
            .collect()
    }

    /// Noisy targets, one `q`-vector per input row, deterministic in `seed`.
    pub fn targets(&self, x: &[f32], noise: f64, seed: u64) -> Vec<f32> {
        assert_eq!(x.len() % self.input, 0);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x51ed_270b);
        let mut out = Vec::with_capacity(x.len() / self.input * self.output);
        for row in x.chunks(self.input) {
            let clean = self.predict(row);
            for v in clean {
                let z: f64 = rng.sample(StandardNormal);
                out.push(v + (z * noise) as f32);
            }
        }
        out
    }
}

/// Full-data Gaussian vector-regression potential.
///
/// `mean_loss(θ) = (1/n) Σ_i ‖y_i − f_θ(e_i)‖² / (2 s²)`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VectorRegression {
    shapes: VectorShapes,
    x: Vec<f32>,
    y: Vec<f32>,
    noise: f64,
    lambda: f64,
    n: usize,
}

impl VectorRegression {
    pub fn new(
        shapes: VectorShapes,
        x: Vec<f32>,
        y: Vec<f32>,
        noise: f64,
        lambda: f64,
    ) -> candle_core::Result<Self> {
        if !noise.is_finite() || noise <= 0.0 {
            return Err(candle_core::Error::Msg(
                "noise must be finite and > 0".into(),
            ));
        }
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(candle_core::Error::Msg(
                "lambda must be finite and > 0".into(),
            ));
        }
        if x.len() % shapes.input != 0 || x.is_empty() {
            return Err(candle_core::Error::Msg(
                "x is empty or not a whole number of embeddings".into(),
            ));
        }
        let n = x.len() / shapes.input;
        if y.len() != n * shapes.output {
            return Err(candle_core::Error::Msg(format!(
                "y must have {} values, got {}",
                n * shapes.output,
                y.len()
            )));
        }
        if x.iter().chain(&y).any(|v| !v.is_finite()) {
            return Err(candle_core::Error::Msg("x and y must be finite".into()))?;
        }
        Ok(Self {
            shapes,
            x,
            y,
            noise,
            lambda,
            n,
        })
    }

    pub fn shapes(&self) -> &VectorShapes {
        &self.shapes
    }
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
    pub fn noise(&self) -> f64 {
        self.noise
    }
    pub fn rows(&self) -> usize {
        self.n
    }
    pub fn x(&self) -> &[f32] {
        &self.x
    }
    pub fn y(&self) -> &[f32] {
        &self.y
    }

    pub fn predict_row(&self, theta: &Tensor, row: usize) -> candle_core::Result<Vec<f32>> {
        if row >= self.n {
            return Err(candle_core::Error::Msg("row out of range".into()));
        }
        let e = Tensor::from_vec(
            self.x[row * self.shapes.input..(row + 1) * self.shapes.input].to_vec(),
            self.shapes.input,
            theta.device(),
        )?;
        forward(theta, &self.shapes, &e)?.to_vec1::<f32>()
    }

    /// One batched forward over all rows.  `layer` already accepts `[B, T, D]`
    /// and rows never interact, so this is the same function as a stack of
    /// per-row forwards — just not `n` separate graph launches.
    fn mean_loss_tensor(&self, theta: &Tensor) -> candle_core::Result<Tensor> {
        let s = unflatten(theta, &self.shapes)?;
        let e = Tensor::from_vec(self.x.clone(), (self.n, self.shapes.input), theta.device())?;
        let mut x =
            e.matmul(&s.projection)?
                .reshape((self.n, self.shapes.tokens, self.shapes.width))?;
        let tau = (self.shapes.width as f64).sqrt();
        for w in &s.layers {
            x = layer(&x, w, Variant::Standard, tau, NormGrad::Exact)?;
        }
        let last = x.narrow(1, self.shapes.tokens - 1, 1)?.squeeze(1)?;
        let preds = last
            .matmul(&s.readout)?
            .broadcast_add(&s.bias.reshape((1, self.shapes.output))?)?;
        let y = Tensor::from_vec(self.y.clone(), (self.n, self.shapes.output), theta.device())?;
        let diff = preds.broadcast_sub(&y)?;
        let sum = diff.sqr()?.sum_all()?;
        sum.affine(1.0 / (2.0 * self.noise * self.noise * self.n as f64), 0.0)
    }

    /// One full objective/gradient evaluation, for an optimizer or a caller
    /// that needs both.  `mean` is the reduction and `n_terms` the multiplier.
    pub fn mean_loss_and_grad(&self, theta: &Tensor) -> Option<(f64, Tensor)> {
        let var = Var::from_tensor(theta).ok()?;
        let loss = self.mean_loss_tensor(var.as_tensor()).ok()?;
        let value = loss.to_scalar::<f32>().ok()? as f64;
        if !value.is_finite() {
            return None;
        }
        let grads = loss.backward().ok()?;
        let grad = grads.get(var.as_tensor())?.clone();
        Some((value, grad))
    }
}

impl Potential for VectorRegression {
    fn n_terms(&self) -> usize {
        self.n
    }

    fn mean_loss(&self, theta: &Tensor) -> f64 {
        self.mean_loss_tensor(theta)
            .ok()
            .and_then(|t| t.to_scalar::<f32>().ok())
            .map(|v| v as f64)
            .filter(|v| v.is_finite())
            .unwrap_or(f64::INFINITY)
    }

    fn mean_grad_with(&self, theta: &Tensor, _norm: NormGrad) -> Option<Tensor> {
        self.mean_loss_and_grad(theta).map(|(_, g)| g)
    }

    fn label(&self) -> &'static str {
        "vector-regression"
    }
}

/// Full-batch Adam on the same mean objective the sampler uses.  Kept explicit
/// because R13 §2 requires both call paths to share the objective.
pub fn adam_fit(
    target: &VectorRegression,
    theta0: Tensor,
    steps: usize,
    lr: f64,
) -> (Tensor, Vec<f64>) {
    let mut theta = theta0;
    let mut m = Tensor::zeros(theta.dims(), DType::F32, theta.device()).expect("cpu");
    let mut v = Tensor::zeros(theta.dims(), DType::F32, theta.device()).expect("cpu");
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut trace = Vec::with_capacity(steps);
    for t in 1..=steps {
        let Some((loss, grad)) = target.mean_loss_and_grad(&theta) else {
            break;
        };
        trace.push(loss);
        m = (m.affine(b1, 0.0).expect("m") + grad.affine(1.0 - b1, 0.0).expect("g")).expect("m'");
        let g2 = grad.sqr().expect("g²");
        v = (v.affine(b2, 0.0).expect("v") + g2.affine(1.0 - b2, 0.0).expect("g²")).expect("v'");
        let mh = m
            .affine(1.0 / (1.0 - b1.powi(t as i32)), 0.0)
            .expect("mhat");
        let vh = v
            .affine(1.0 / (1.0 - b2.powi(t as i32)), 0.0)
            .expect("vhat");
        let denom = (vh.sqrt().expect("sqrt") + eps).expect("sqrt+eps");
        let step = (mh / denom).expect("ratio").affine(lr, 0.0).expect("lr");
        theta = (&theta - &step).expect("theta'");
    }
    (theta, trace)
}

/// R13 §3 transition settings.  `rho` is fixed after warmup.
#[derive(Clone, Copy, Debug)]
pub struct VectorSamplerConfig {
    pub steps: usize,
    pub burn: usize,
    pub thin: usize,
    pub rho: f64,
    pub refresh: f64,
    pub prior_sigma: f64,
    pub draws: usize,
}

impl Default for VectorSamplerConfig {
    fn default() -> Self {
        Self {
            steps: 1_000,
            burn: 200,
            thin: 1,
            rho: 0.99,
            refresh: 0.01,
            prior_sigma: 0.05,
            draws: 40,
        }
    }
}

impl VectorSamplerConfig {
    pub fn validate(&self, shapes: &VectorShapes) -> candle_core::Result<()> {
        shapes.prior_scales();
        self.validate_dim(shapes.flat_size())
    }

    fn validate_dim(&self, _dim: usize) -> candle_core::Result<()> {
        if self.steps == 0 || self.thin == 0 || self.draws == 0 {
            return Err(candle_core::Error::Msg(
                "steps, thin and draws must be non-zero".into(),
            ));
        }
        if self.burn >= self.steps {
            return Err(candle_core::Error::Msg("burn must be < steps".into()));
        }
        if !(0.0..1.0).contains(&self.rho) {
            return Err(candle_core::Error::Msg("rho must lie in (0,1)".into()));
        }
        if !(0.0..1.0).contains(&self.refresh) {
            return Err(candle_core::Error::Msg("refresh must lie in [0,1)".into()));
        }
        if !self.prior_sigma.is_finite() || self.prior_sigma <= 0.0 {
            return Err(candle_core::Error::Msg("prior_sigma must be > 0".into()));
        }
        let available = (self.steps - self.burn + self.thin - 1) / self.thin;
        if self.draws > available {
            return Err(candle_core::Error::Msg(format!(
                "draws={} exceeds {available} post-burn thinned steps",
                self.draws
            )));
        }
        Ok(())
    }
}

/// The energy seam for the R13 kernel.  A target reports `U(θ)` and its
/// gradient; the prior is supplied separately as the diagonal covariance `C`.
/// Keeping this generic lets a known-target check use the exact production
/// transition code rather than a test-only reimplementation.
pub trait EnergyTarget {
    fn dim(&self) -> usize;
    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)>;
}

impl EnergyTarget for VectorRegression {
    fn dim(&self) -> usize {
        self.shapes.flat_size()
    }

    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let t = Tensor::from_vec(theta.to_vec(), theta.len(), &Device::Cpu).ok()?;
        let (mean, grad) = self.mean_loss_and_grad(&t)?;
        let scale = self.lambda * self.n as f64;
        // The CN drift consumes this gradient as the gradient of `u`, so it
        // must carry the same `lambda * n` scaling as the energy.  Returning
        // the raw mean-loss gradient left the drift 240x too small at n=240.
        let grad = (grad * scale).ok()?.to_vec1::<f32>().ok()?;
        Some((scale * mean, grad))
    }
}

/// The affine readout occupies the last `width * output + output` flat
/// coordinates; see [`VectorShapes::layout`].
pub fn readout_range(shapes: &VectorShapes) -> std::ops::Range<usize> {
    let len = shapes.width * shapes.output + shapes.output;
    shapes.flat_size() - len..shapes.flat_size()
}

/// Posterior target over the affine readout only, with the backbone held at a
/// fixed flat parameter vector.
///
/// This is the low-dimensional target for a four-chain diagnostic: at the full
/// 1538-dimensional R13 size the CN kernel's integrated autocorrelation time is
/// a few thousand steps, so a promotion-grade ESS is out of reach.  Sampling
/// the 34 readout coordinates conditional on a point-estimated backbone is a
/// *different, explicitly conditional* target, not the full posterior.
pub struct ReadoutTarget {
    base: VectorRegression,
    backbone: Vec<f32>,
    range: std::ops::Range<usize>,
}

impl ReadoutTarget {
    pub fn new(base: VectorRegression, flat: Vec<f32>) -> candle_core::Result<Self> {
        if flat.len() != base.shapes().flat_size() {
            return Err(candle_core::Error::Msg(
                "backbone vector does not match the model's flat size".into(),
            ));
        }
        let range = readout_range(base.shapes());
        Ok(Self {
            base,
            backbone: flat,
            range,
        })
    }

    pub fn base(&self) -> &VectorRegression {
        &self.base
    }

    pub fn start(&self) -> Vec<f32> {
        self.backbone[self.range.clone()].to_vec()
    }

    /// Insert a readout draw into the frozen backbone.
    pub fn expand(&self, theta: &[f32]) -> Vec<f32> {
        let mut full = self.backbone.clone();
        full[self.range.clone()].copy_from_slice(theta);
        full
    }
}

impl EnergyTarget for ReadoutTarget {
    fn dim(&self) -> usize {
        self.range.len()
    }

    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let full = self.expand(theta);
        let t = Tensor::from_vec(full, self.backbone.len(), &Device::Cpu).ok()?;
        let (mean, grad) = self.base.mean_loss_and_grad(&t)?;
        let scale = self.base.lambda * self.base.n as f64;
        let grad = grad
            .narrow(0, self.range.start, self.range.len())
            .ok()?
            .affine(scale, 0.0)
            .ok()?
            .to_vec1::<f32>()
            .ok()?;
        Some((scale * mean, grad))
    }
}

// ---------------------------------------------------------------------
// Dense-covariance linear algebra (d is small: 34 readout coordinates)
// ---------------------------------------------------------------------

/// Lower Cholesky factor `L` of a symmetric positive-definite matrix `A`,
/// with `A = L Lᵀ`, row-major.  Also returns `ln det A`.
fn cholesky_lower(a: &[f64], n: usize) -> Option<(Vec<f64>, f64)> {
    let mut l = vec![0.0f64; n * n];
    let mut logdet = 0.0f64;
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(s > 0.0) || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
                logdet += s.ln();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some((l, logdet))
}

/// Solve `L y = b` for lower-triangular `L`.
fn forward_substitute(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    y
}

/// Solve `Lᵀ x = b` for lower-triangular `L`.
fn back_substitute_transpose(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// A positive-definite **covariance** held as the Cholesky factor of its
/// precision: `A = L Lᵀ`, `C = A⁻¹ = (L Lᵀ)⁻¹`.
///
/// Storing the precision factor is what makes both the CN proposal and the
/// log-density cheap: the drift needs `C g = A⁻¹ g` (two triangular solves) and
/// the proposal density needs `δᵀ C⁻¹ δ = δᵀ A δ` (a direct multiply) plus
/// `ln det C = −ln det A`.
#[derive(Clone, Debug)]
pub struct DenseCovariance {
    dim: usize,
    chol_precision: Vec<f64>,
    logdet_precision: f64,
}

impl DenseCovariance {
    /// `precision` is the row-major `d × d` inverse covariance.
    pub fn from_precision(precision: &[f64], dim: usize) -> candle_core::Result<Self> {
        if dim == 0 || precision.len() != dim * dim {
            return Err(candle_core::Error::Msg(
                "precision must be a non-empty square matrix".into(),
            ));
        }
        let (chol_precision, logdet_precision) =
            cholesky_lower(precision, dim).ok_or_else(|| {
                candle_core::Error::Msg("precision matrix is not positive definite".into())
            })?;
        Ok(Self {
            dim,
            chol_precision,
            logdet_precision,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn logdet_precision(&self) -> f64 {
        self.logdet_precision
    }

    /// The covariance itself, row-major (one precision solve per column).
    pub fn to_covariance(&self) -> Vec<f64> {
        let mut cov = vec![0.0f64; self.dim * self.dim];
        for col in 0..self.dim {
            let mut e = vec![0.0f64; self.dim];
            e[col] = 1.0;
            let x = self.solve_precision(&e);
            for row in 0..self.dim {
                cov[row * self.dim + col] = x[row];
            }
        }
        cov
    }

    /// `A⁻¹ b`.
    pub fn solve_precision(&self, b: &[f64]) -> Vec<f64> {
        let y = forward_substitute(&self.chol_precision, self.dim, b);
        back_substitute_transpose(&self.chol_precision, self.dim, &y)
    }

    /// `L⁻ᵀ v`: the covariance factor `G` with `G Gᵀ = C`, applied to a vector.
    pub fn apply_cov_factor(&self, v: &[f64]) -> Vec<f64> {
        back_substitute_transpose(&self.chol_precision, self.dim, v)
    }

    /// `L⁻¹ v = Gᵀ v`, the transpose of [`Self::apply_cov_factor`].
    pub fn apply_cov_factor_transpose(&self, v: &[f64]) -> Vec<f64> {
        forward_substitute(&self.chol_precision, self.dim, v)
    }

    /// `δᵀ A δ = δᵀ C⁻¹ δ = ‖Lᵀ δ‖²` (columns of `L`, not rows: `L Lᵀ = A`).
    pub fn quadratic(&self, delta: &[f64]) -> f64 {
        let l = &self.chol_precision;
        let n = self.dim;
        let mut sum = 0.0f64;
        for i in 0..n {
            let mut column = 0.0;
            for k in i..n {
                column += l[k * n + i] * delta[k];
            }
            sum += column * column;
        }
        sum
    }

    /// `z ~ N(0, A⁻¹)`: solve `Lᵀ x = z₀`.
    pub fn draw(&self, rng: &mut StdRng) -> Vec<f64> {
        let z: Vec<f64> = (0..self.dim).map(|_| rng.sample(StandardNormal)).collect();
        back_substitute_transpose(&self.chol_precision, self.dim, &z)
    }
}

/// Proposal covariance for the CN kernel: either the diagonal form the R13
/// runs used, or a full dense covariance (the exact readout posterior).
#[derive(Clone, Debug)]
pub enum ProposalCov {
    Diagonal(Vec<f64>),
    Dense(DenseCovariance),
}

impl ProposalCov {
    pub fn dim(&self) -> usize {
        match self {
            Self::Diagonal(c) => c.len(),
            Self::Dense(d) => d.dim(),
        }
    }

    /// `C v`.
    pub fn cov_times(&self, v: &[f64]) -> Vec<f64> {
        match self {
            Self::Diagonal(c) => c.iter().zip(v).map(|(ci, vi)| ci * vi).collect(),
            Self::Dense(d) => d.solve_precision(v),
        }
    }

    /// `sqrt(scale) · N(0, C)`.
    pub fn draw_scaled(&self, scale: f64, rng: &mut StdRng) -> Vec<f64> {
        match self {
            Self::Diagonal(c) => c
                .iter()
                .map(|ci| {
                    let z: f64 = rng.sample(StandardNormal);
                    z * (scale * ci).sqrt()
                })
                .collect(),
            Self::Dense(d) => d.draw(rng).into_iter().map(|v| v * scale.sqrt()).collect(),
        }
    }

    /// `δᵀ C⁻¹ δ`.
    pub fn quadratic(&self, delta: &[f64]) -> f64 {
        match self {
            Self::Diagonal(c) => delta.iter().zip(c).map(|(&di, ci)| di * di / ci).sum(),
            Self::Dense(d) => d.quadratic(delta),
        }
    }

    /// `ln N(δ; 0, scale·C)`.
    pub fn log_density(&self, delta: &[f64], scale: f64) -> f64 {
        const LOG_2PI: f64 = 1.8378770664093453;
        match self {
            Self::Diagonal(c) => delta
                .iter()
                .zip(c)
                .map(|(&di, ci)| {
                    let vi = scale * ci;
                    -0.5 * (di * di / vi + (2.0 * std::f64::consts::PI * vi).ln())
                })
                .sum(),
            Self::Dense(d) => {
                let n = d.dim() as f64;
                -0.5 * (d.quadratic(delta) / scale + n * (LOG_2PI + scale.ln())
                    - d.logdet_precision)
            }
        }
    }
}

/// Exact Gaussian posterior of the readout-conditional target.
///
/// With the backbone frozen the readout is an ordinary linear model
/// `y = Φ θ + ε`, `ε ~ N(0, s²)`, against the diagonal Gaussian prior on the
/// readout coordinates, so the posterior is
/// `N(m, A⁻¹)` with `A = (λ/s²) ΦᵀΦ + diag(1/c)`.  The `λ` (and not `λn`) is
/// exact: the target's energy is `λ Σ_i ‖y_i − f_i‖²/(2s²)`.
#[derive(Clone, Debug)]
pub struct ExactPosterior {
    mean: Vec<f64>,
    dense: DenseCovariance,
}

impl ExactPosterior {
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    pub fn covariance(&self) -> Vec<f64> {
        self.dense.to_covariance()
    }

    /// Posterior variances, i.e. the diagonal of the exact covariance.
    pub fn variance(&self) -> Vec<f64> {
        (0..self.dim())
            .map(|i| {
                let mut e = vec![0.0f64; self.dim()];
                e[i] = 1.0;
                self.dense.solve_precision(&e)[i]
            })
            .collect()
    }

    pub fn dense(&self) -> DenseCovariance {
        self.dense.clone()
    }

    /// Proposal covariance for the CN kernel.
    pub fn proposal(&self) -> ProposalCov {
        ProposalCov::Dense(self.dense.clone())
    }

    /// Exact posterior draw `m + Lᵀ⁻¹ z`.
    pub fn draw(&self, rng: &mut StdRng) -> Vec<f32> {
        let scale = self.dense.draw(rng);
        self.mean
            .iter()
            .zip(&scale)
            .map(|(&m, &s)| (m + s) as f32)
            .collect()
    }
}

impl ReadoutTarget {
    /// Exact posterior of this conditional target for the given diagonal prior.
    /// `prior_c` is the readout slice of the prior covariance, exactly as the
    /// sampler receives it.
    pub fn exact_posterior(&self, prior_c: &[f64]) -> candle_core::Result<ExactPosterior> {
        let shapes = self.base.shapes();
        let d = self.range.len();
        if prior_c.len() != d {
            return Err(candle_core::Error::Msg(format!(
                "prior_c must have {d} values, got {}",
                prior_c.len()
            )));
        }
        if prior_c.iter().any(|c| !c.is_finite() || *c <= 0.0) {
            return Err(candle_core::Error::Msg(
                "prior_c must be finite and positive".into(),
            ));
        }
        let q = shapes.output;
        let w = shapes.width;
        let scale = self.base.lambda() / (self.base.noise() * self.base.noise());
        let backbone = Tensor::from_vec(self.backbone.clone(), shapes.flat_size(), &Device::Cpu)?;
        let mut xtx = vec![0.0f64; d * d];
        let mut xty = vec![0.0f64; d];
        let mut phi = vec![0.0f64; d];
        for row in 0..self.base.rows() {
            let e = Tensor::from_vec(
                self.base.x()[row * shapes.input..(row + 1) * shapes.input].to_vec(),
                shapes.input,
                &Device::Cpu,
            )?;
            let h = last_hidden(&backbone, shapes, &e)?.to_vec1::<f32>()?;
            for j in 0..q {
                phi.fill(0.0);
                for (k, &hk) in h.iter().enumerate() {
                    phi[k * q + j] = f64::from(hk);
                }
                phi[w * q + j] = 1.0;
                let y = f64::from(self.base.y()[row * q + j]);
                for r in 0..d {
                    let pr = phi[r];
                    if pr == 0.0 {
                        continue;
                    }
                    xty[r] += pr * y;
                    for c in r..d {
                        xtx[r * d + c] += pr * phi[c];
                    }
                }
            }
        }
        let mut precision = vec![0.0f64; d * d];
        for r in 0..d {
            for c in r..d {
                let v = scale * xtx[r * d + c];
                precision[r * d + c] = v;
                precision[c * d + r] = v;
            }
            precision[r * d + r] += 1.0 / prior_c[r];
        }
        let rhs: Vec<f64> = xty.iter().map(|v| scale * v).collect();
        let dense = DenseCovariance::from_precision(&precision, d)?;
        let y = forward_substitute(&dense.chol_precision, d, &rhs);
        let mean = back_substitute_transpose(&dense.chol_precision, d, &y);
        Ok(ExactPosterior { mean, dense })
    }
}

#[derive(Clone, Debug)]
pub struct VectorChain {
    pub theta: Vec<f32>,
    pub draws: Vec<Vec<f32>>,
    pub proposals: usize,
    pub accepted: usize,
    pub cn_attempts: usize,
    pub cn_accepted: usize,
    pub refresh_attempts: usize,
    pub refresh_accepted: usize,
}

fn prior_penalty(theta: &[f32], c: &[f64]) -> f64 {
    0.5 * theta
        .iter()
        .zip(c)
        .map(|(&t, &ci)| f64::from(t) * f64::from(t) / ci)
        .sum::<f64>()
}

fn draw_prior(c: &[f64], rng: &mut StdRng) -> Vec<f32> {
    c.iter()
        .map(|&ci| {
            let z: f64 = rng.sample(StandardNormal);
            (z * ci.sqrt()) as f32
        })
        .collect()
}

struct Energies {
    u: f64,
    grad: Vec<f32>,
}

impl Energies {
    fn eval<T: EnergyTarget>(target: &T, theta: &[f32]) -> Option<Self> {
        let (u, grad) = target.energy_and_grad(theta)?;
        Some(Self { u, grad })
    }
}
/// One chain of the exact CN/prior-refresh mixture for the vector target.
pub fn run_chain(
    target: &VectorRegression,
    config: VectorSamplerConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<VectorChain> {
    run_chain_preconditioned(target, config, theta0, seed, 1.0)
}

/// The vector target with a separate diagonal CN proposal scale.  The prior in
/// the target stays fixed; `proposal_scale` only shrinks or widens the proposal
/// covariance relative to the prior covariance.  The MH ratio still uses the
/// forward and reverse proposal densities, so any positive scale is exact.
pub fn run_chain_preconditioned(
    target: &VectorRegression,
    config: VectorSamplerConfig,
    theta0: Vec<f32>,
    seed: u64,
    proposal_scale: f64,
) -> candle_core::Result<VectorChain> {
    if !proposal_scale.is_finite() || proposal_scale <= 0.0 {
        return Err(candle_core::Error::Msg(
            "proposal_scale must be finite and > 0".into(),
        ));
    }
    let shapes = target.shapes();
    config.validate(shapes)?;
    let scales = shapes.prior_scales();
    let prior_c: Vec<f64> = scales
        .iter()
        .map(|s| config.prior_sigma.powi(2) * s * s)
        .collect();
    let proposal_c: Vec<f64> = prior_c
        .iter()
        .map(|c| c * proposal_scale * proposal_scale)
        .collect();
    run_chain_with_covariances(target, &prior_c, &proposal_c, config, theta0, seed)
}

/// The same kernel with an explicit diagonal prior covariance.  This is the
/// entry point for the known-target check; `run_chain` fixes `C` from the
/// vector model's layer scales.
pub fn run_chain_with_prior<T: EnergyTarget>(
    target: &T,
    c: &[f64],
    config: VectorSamplerConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<VectorChain> {
    run_chain_with_covariances(target, c, c, config, theta0, seed)
}

/// Exact CN/prior-refresh kernel with independent prior and proposal
/// covariances.  Refresh draws from the prior; CN uses `proposal_c` in both the
/// forward mean/variance and the reverse density.
pub fn run_chain_with_covariances<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    proposal_c: &[f64],
    config: VectorSamplerConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<VectorChain> {
    if proposal_c.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(candle_core::Error::Msg(
            "proposal covariance must be finite and positive".into(),
        ));
    }
    run_chain_with_proposal(
        target,
        prior_c,
        &ProposalCov::Diagonal(proposal_c.to_vec()),
        config,
        theta0,
        seed,
    )
}

/// Exact CN/prior-refresh kernel with an arbitrary positive-definite proposal
/// covariance.  A dense `C` makes the drift `ρθ − (1−ρ)C∇U` fully coupled and
/// keeps the proposal density exact through `C⁻¹` and `ln det C`.
pub fn run_chain_with_proposal<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    proposal: &ProposalCov,
    config: VectorSamplerConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<VectorChain> {
    let dim = target.dim();
    run_chain_with_reference(
        target,
        prior_c,
        &vec![0.0; dim],
        proposal,
        false,
        config,
        theta0,
        seed,
    )
}

/// CN kernel referenced to an arbitrary Gaussian `N(reference, C)` instead of
/// the prior `N(0, diag prior_c)`.  The drift uses the **full** target
/// gradient (likelihood plus prior), which is the gradient of the target the
/// MH ratio evaluates, so at `θ = reference` the proposal is mean-preserving.
/// With `C` the exact posterior covariance and `reference` its mean this is the
/// exact-covariance proposal for the readout conditional; the prior-reference
/// kernel above is the `reference = 0`, likelihood-only special case the R13
/// runs used.
pub fn run_chain_with_posterior_reference<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    reference: &[f64],
    proposal: &ProposalCov,
    config: VectorSamplerConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<VectorChain> {
    run_chain_with_reference(
        target, prior_c, reference, proposal, true, config, theta0, seed,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_chain_with_reference<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    reference: &[f64],
    proposal: &ProposalCov,
    full_gradient: bool,
    config: VectorSamplerConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<VectorChain> {
    config.validate_dim(target.dim())?;
    if prior_c.len() != target.dim() || prior_c.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(candle_core::Error::Msg(
            "prior covariance must be finite, positive and target-sized".into(),
        ));
    }
    if proposal.dim() != target.dim() {
        return Err(candle_core::Error::Msg(format!(
            "proposal covariance is {}x{}, target is {}",
            proposal.dim(),
            proposal.dim(),
            target.dim()
        )));
    }
    if theta0.len() != target.dim() {
        return Err(candle_core::Error::Msg(format!(
            "theta0 must have {} values, got {}",
            target.dim(),
            theta0.len()
        )));
    }
    if reference.len() != target.dim() || reference.iter().any(|r| !r.is_finite()) {
        return Err(candle_core::Error::Msg(
            "reference mean must be finite and target-sized".into(),
        ));
    }
    let c = prior_c.to_vec();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut theta = theta0;
    let mut energy = Energies::eval(target, &theta)
        .ok_or_else(|| candle_core::Error::Msg("initial vector has no finite objective".into()))?;
    let mut prior = prior_penalty(&theta, &c);
    let mut draws = Vec::with_capacity(config.draws);
    let mut chain = VectorChain {
        theta: Vec::new(),
        draws: Vec::new(),
        proposals: 0,
        accepted: 0,
        cn_attempts: 0,
        cn_accepted: 0,
        refresh_attempts: 0,
        refresh_accepted: 0,
    };
    let rho2 = config.rho * config.rho;
    let proposal_scale = 1.0 - rho2;
    for step in 0..config.steps {
        chain.proposals += 1;
        if rng.random::<f64>() < config.refresh {
            chain.refresh_attempts += 1;
            let proposed = draw_prior(&c, &mut rng);
            if let Some(next) = Energies::eval(target, &proposed) {
                let next_prior = prior_penalty(&proposed, &c);
                // Independence proposal from the prior: both prior densities
                // cancel, leaving exp(U(θ) − U(θ')).
                let log_alpha = energy.u - next.u;
                if log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha {
                    theta = proposed;
                    energy = next;
                    prior = next_prior;
                    chain.accepted += 1;
                    chain.refresh_accepted += 1;
                }
            }
        } else {
            chain.cn_attempts += 1;
            let grad: Vec<f64> = energy
                .grad
                .iter()
                .zip(&theta)
                .zip(&c)
                .map(|((g, t), ci)| {
                    f64::from(*g)
                        + if full_gradient {
                            f64::from(*t) / ci
                        } else {
                            0.0
                        }
                })
                .collect();
            let drift = proposal.cov_times(&grad);
            let mean: Vec<f64> = theta
                .iter()
                .zip(reference)
                .zip(&drift)
                .map(|((&t, &r), d)| r + config.rho * (f64::from(t) - r) - (1.0 - config.rho) * d)
                .collect();
            let noise = proposal.draw_scaled(proposal_scale, &mut rng);
            let proposed: Vec<f32> = mean
                .iter()
                .zip(&noise)
                .map(|(&m, &n)| (m + n) as f32)
                .collect();
            if let Some(next) = Energies::eval(target, &proposed) {
                let next_prior = prior_penalty(&proposed, &c);
                let next_grad: Vec<f64> = next
                    .grad
                    .iter()
                    .zip(&proposed)
                    .zip(&c)
                    .map(|((g, t), ci)| {
                        f64::from(*g)
                            + if full_gradient {
                                f64::from(*t) / ci
                            } else {
                                0.0
                            }
                    })
                    .collect();
                let next_drift = proposal.cov_times(&next_grad);
                let reverse_mean: Vec<f64> = proposed
                    .iter()
                    .zip(reference)
                    .zip(&next_drift)
                    .map(|((&t, &r), d)| {
                        r + config.rho * (f64::from(t) - r) - (1.0 - config.rho) * d
                    })
                    .collect();
                let forward_delta: Vec<f64> = proposed
                    .iter()
                    .zip(&mean)
                    .map(|(&p, &m)| f64::from(p) - m)
                    .collect();
                let reverse_delta: Vec<f64> = theta
                    .iter()
                    .zip(&reverse_mean)
                    .map(|(&t, &m)| f64::from(t) - m)
                    .collect();
                let log_q_forward = proposal.log_density(&forward_delta, proposal_scale);
                let log_q_reverse = proposal.log_density(&reverse_delta, proposal_scale);
                let log_alpha =
                    (energy.u + prior) - (next.u + next_prior) + log_q_reverse - log_q_forward;
                if log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha {
                    theta = proposed;
                    energy = next;
                    prior = next_prior;
                    chain.accepted += 1;
                    chain.cn_accepted += 1;
                }
            }
        }
        if step >= config.burn
            && (step - config.burn) % config.thin == 0
            && draws.len() < config.draws
        {
            draws.push(theta.clone());
        }
    }
    chain.theta = theta;
    chain.draws = draws;
    Ok(chain)
}

/// Fixed-length Hamiltonian Monte Carlo settings.  The adapted step size is
/// frozen at the end of warmup, so every retained transition is an exact MH
/// move; `leapfrog` is the trajectory length and `step_size` the initial (or
/// only) integrator step.
#[derive(Clone, Copy, Debug)]
pub struct HmcConfig {
    pub steps: usize,
    pub burn: usize,
    pub thin: usize,
    pub draws: usize,
    pub leapfrog: usize,
    pub step_size: f64,
    pub adapt_step_size: bool,
    pub target_accept: f64,
}

impl Default for HmcConfig {
    fn default() -> Self {
        Self {
            steps: 1_000,
            burn: 400,
            thin: 1,
            draws: 600,
            leapfrog: 8,
            step_size: 1.0,
            adapt_step_size: true,
            target_accept: 0.8,
        }
    }
}

impl HmcConfig {
    pub fn validate(&self) -> candle_core::Result<()> {
        if self.steps == 0 || self.thin == 0 || self.draws == 0 || self.leapfrog == 0 {
            return Err(candle_core::Error::Msg(
                "steps, thin, draws and leapfrog must be non-zero".into(),
            ));
        }
        if self.burn >= self.steps {
            return Err(candle_core::Error::Msg("burn must be < steps".into()));
        }
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(candle_core::Error::Msg(
                "step_size must be finite and > 0".into(),
            ));
        }
        if !(0.0..1.0).contains(&self.target_accept) {
            return Err(candle_core::Error::Msg(
                "target_accept must lie in (0,1)".into(),
            ));
        }
        let available = (self.steps - self.burn + self.thin - 1) / self.thin;
        if self.draws > available {
            return Err(candle_core::Error::Msg(format!(
                "draws={} exceeds {available} post-burn thinned steps",
                self.draws
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct HmcChain {
    pub theta: Vec<f32>,
    pub draws: Vec<Vec<f32>>,
    pub proposals: usize,
    pub accepted: usize,
    pub grad_evals: usize,
    pub step_size: f64,
    pub mean_accept: f64,
}

struct TotalEnergy {
    u: f64,
    grad: Vec<f64>,
}

/// The sampler's target is `−ln` posterior: likelihood energy from the target's
/// own reduction plus the diagonal Gaussian prior the CN kernel adds separately.
fn total_energy<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    theta: &[f32],
) -> Option<TotalEnergy> {
    let (u, grad) = target.energy_and_grad(theta)?;
    if grad.len() != prior_c.len() || theta.len() != prior_c.len() {
        return None;
    }
    let mut pu = 0.0f64;
    let mut total_grad = Vec::with_capacity(prior_c.len());
    for ((&g, &c), &t) in grad.iter().zip(prior_c).zip(theta) {
        let t = f64::from(t);
        pu += 0.5 * t * t / c;
        total_grad.push(f64::from(g) + t / c);
    }
    let u = u + pu;
    if !u.is_finite() || total_grad.iter().any(|g| !g.is_finite()) {
        return None;
    }
    Some(TotalEnergy {
        u,
        grad: total_grad,
    })
}

/// HMC metric: a diagonal `K(p) = ½ Σ p_i²/m_i`, or the exact whitened metric
/// `u = Lᵀ(θ − m)` with unit mass, where `L Lᵀ` is the posterior precision.
/// In the whitened coordinates a Gaussian posterior is exactly standard normal,
/// so unit step sizes are well scaled.
#[derive(Clone, Copy)]
enum HmcMetric<'a> {
    Diagonal(&'a [f64]),
    Whitened { cov: &'a DenseCovariance },
}

impl HmcMetric<'_> {
    /// Momentum draw: `N(0, M)`.
    fn draw(&self, rng: &mut StdRng) -> Vec<f64> {
        match self {
            Self::Diagonal(m) => m
                .iter()
                .map(|mi| rng.sample::<f64, _>(StandardNormal) * mi.sqrt())
                .collect(),
            Self::Whitened { .. } => (0..self.dim())
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            Self::Diagonal(m) => m.len(),
            Self::Whitened { cov } => cov.dim(),
        }
    }

    /// `dθ/dt` for momentum `p`.
    fn velocity(&self, p: &[f64]) -> Vec<f64> {
        match self {
            Self::Diagonal(m) => p.iter().zip(*m).map(|(pi, mi)| pi / mi).collect(),
            Self::Whitened { cov, .. } => cov.apply_cov_factor(p),
        }
    }

    /// `dp/dt` for a potential gradient.
    fn force(&self, grad: &[f64]) -> Vec<f64> {
        match self {
            Self::Diagonal(_) => grad.to_vec(),
            Self::Whitened { cov, .. } => cov.apply_cov_factor_transpose(grad),
        }
    }

    fn kinetic(&self, p: &[f64]) -> f64 {
        match self {
            Self::Diagonal(m) => 0.5 * p.iter().zip(*m).map(|(pi, mi)| pi * pi / mi).sum::<f64>(),
            Self::Whitened { .. } => 0.5 * p.iter().map(|pi| pi * pi).sum::<f64>(),
        }
    }
}

/// Fixed-length HMC (leapfrog) with optional dual-averaging step-size
/// adaptation during warmup.  `mass` is the diagonal metric
/// `K(p) = ½ Σ p_i²/m_i`; pass the posterior variances for a whitened target.
/// Every accepted move uses the full `exp(H₀ − H₁)` MH ratio, so a rejected
/// leapfrog trajectory leaves the state untouched.
pub fn run_hmc_chain<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    mass: &[f64],
    config: HmcConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<HmcChain> {
    if mass.len() != target.dim() || mass.iter().any(|m| !m.is_finite() || *m <= 0.0) {
        return Err(candle_core::Error::Msg(
            "HMC mass must be finite, positive and target-sized".into(),
        ));
    }
    run_hmc_with_metric(
        target,
        prior_c,
        &HmcMetric::Diagonal(mass),
        config,
        theta0,
        seed,
    )
}

/// HMC in the exact whitened coordinates `u = Lᵀ(θ − m)`, where `L Lᵀ` is the
/// posterior precision.  A Gaussian posterior is exactly standard normal in
/// `u`, so this is the dense metric `M = precision` and unit step sizes are well
/// scaled; the MH ratio still uses the full target in `θ`.
pub fn run_hmc_chain_whitened<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    mass: &DenseCovariance,
    config: HmcConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<HmcChain> {
    if mass.dim() != target.dim() {
        return Err(candle_core::Error::Msg(
            "dense HMC mass must be target-sized".into(),
        ));
    }
    run_hmc_with_metric(
        target,
        prior_c,
        &HmcMetric::Whitened { cov: mass },
        config,
        theta0,
        seed,
    )
}

fn run_hmc_with_metric<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    metric: &HmcMetric,
    config: HmcConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<HmcChain> {
    config.validate()?;
    let dim = target.dim();
    if prior_c.len() != dim || prior_c.iter().any(|c| !c.is_finite() || *c <= 0.0) {
        return Err(candle_core::Error::Msg(
            "prior covariance must be finite, positive and target-sized".into(),
        ));
    }
    if theta0.len() != dim {
        return Err(candle_core::Error::Msg(format!(
            "theta0 must have {dim} values, got {}",
            theta0.len()
        )));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut theta = theta0;
    let mut current = total_energy(target, prior_c, &theta)
        .ok_or_else(|| candle_core::Error::Msg("initial vector has no finite objective".into()))?;
    let mut draws = Vec::with_capacity(config.draws);
    let (mut proposals, mut accepted, mut grad_evals) = (0usize, 0usize, 0usize);
    let mut accept_sum = 0.0f64;
    let (gamma, t0, kappa) = (0.05f64, 10.0f64, 0.75f64);
    let mut log_eps = config.step_size.ln();
    let mut log_eps_bar = log_eps;
    let mut hbar = 0.0f64;
    let mut eps = config.step_size;
    let mut adapting = config.adapt_step_size;
    for step in 0..config.steps {
        if adapting && step == config.burn {
            eps = log_eps_bar.exp();
            adapting = false;
        } else if adapting {
            eps = log_eps.exp();
        }
        proposals += 1;
        let mut p = metric.draw(&mut rng);
        let mut proposed = theta.clone();
        let (init_u, init_k) = (current.u, metric.kinetic(&p));
        let mut grad = current.grad.clone();
        let mut force = metric.force(&grad);
        for i in 0..dim {
            p[i] -= 0.5 * eps * force[i];
        }
        let mut last: Option<TotalEnergy> = None;
        let mut finite = true;
        for l in 0..config.leapfrog {
            let velocity = metric.velocity(&p);
            for i in 0..dim {
                proposed[i] = (f64::from(proposed[i]) + eps * velocity[i]) as f32;
            }
            match total_energy(target, prior_c, &proposed) {
                Some(ev) => {
                    grad_evals += 1;
                    grad = ev.grad.clone();
                    last = Some(ev);
                }
                None => {
                    finite = false;
                    break;
                }
            }
            force = metric.force(&grad);
            if l + 1 < config.leapfrog {
                for i in 0..dim {
                    p[i] -= eps * force[i];
                }
            }
        }
        let mut next_state: Option<TotalEnergy> = None;
        let accept_prob = if !finite {
            0.0
        } else {
            for i in 0..dim {
                p[i] -= 0.5 * eps * force[i];
            }
            let next = last.take().expect("a finite leapfrog has a final state");
            let log_alpha = (init_u + init_k) - (next.u + metric.kinetic(&p));
            let prob = if log_alpha >= 0.0 {
                1.0
            } else {
                log_alpha.exp()
            };
            next_state = Some(next);
            prob
        };
        if accept_prob >= 1.0 || (accept_prob > 0.0 && rng.random::<f64>() < accept_prob) {
            if let Some(next) = next_state {
                theta = proposed;
                current = next;
                accepted += 1;
            }
        }
        accept_sum += accept_prob;
        if adapting {
            let m = (step + 1) as f64;
            let eta = 1.0 / (m + t0);
            hbar = (1.0 - eta) * hbar + eta * (config.target_accept - accept_prob);
            log_eps = (10.0 * config.step_size).ln() - (m.sqrt() / gamma) * hbar;
            let w = m.powf(-kappa);
            log_eps_bar = w * log_eps + (1.0 - w) * log_eps_bar;
        }
        if step >= config.burn
            && (step - config.burn) % config.thin == 0
            && draws.len() < config.draws
        {
            draws.push(theta.clone());
        }
    }
    let step_size = if config.adapt_step_size {
        log_eps_bar.exp()
    } else {
        config.step_size
    };
    Ok(HmcChain {
        theta,
        draws,
        proposals,
        accepted,
        grad_evals,
        step_size,
        mean_accept: accept_sum / proposals as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> VectorShapes {
        VectorShapes::with_width(4, 2, 3, 5, 4, 4).unwrap()
    }

    /// `U(θ) = ½ θᵀ A θ` with a known precision, for exactness checks.
    struct Quadratic {
        a: Vec<f64>,
        dim: usize,
    }

    /// `U(θ) = ½ Σ p_i (θ_i − μ_i)²`: a shifted Gaussian likelihood whose
    /// posterior mean is far from the prior mean.
    struct ShiftedQuadratic {
        p: Vec<f64>,
        mu: Vec<f64>,
    }

    impl EnergyTarget for ShiftedQuadratic {
        fn dim(&self) -> usize {
            self.p.len()
        }
        fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
            let mut u = 0.0;
            let mut grad = Vec::with_capacity(self.p.len());
            for i in 0..self.p.len() {
                let d = f64::from(theta[i]) - self.mu[i];
                u += 0.5 * self.p[i] * d * d;
                grad.push((self.p[i] * d) as f32);
            }
            Some((u, grad))
        }
    }

    impl EnergyTarget for Quadratic {
        fn dim(&self) -> usize {
            self.dim
        }
        fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
            let x: Vec<f64> = theta.iter().map(|v| f64::from(*v)).collect();
            let mut grad = vec![0.0f64; self.dim];
            for i in 0..self.dim {
                for j in 0..self.dim {
                    grad[i] += self.a[i * self.dim + j] * x[j];
                }
            }
            let u = 0.5 * x.iter().zip(&grad).map(|(xi, gi)| xi * gi).sum::<f64>();
            Some((u, grad.iter().map(|g| *g as f32).collect()))
        }
    }

    fn quadratic_draws_agree(got: &[Vec<f32>], mean: &[f64], cov: &[f64], tol: f64) {
        let n = got.len() as f64;
        let dim = mean.len();
        let mut emp_mean = vec![0.0f64; dim];
        for d in got {
            for (i, v) in d.iter().enumerate() {
                emp_mean[i] += f64::from(*v) / n;
            }
        }
        let mut emp_cov = vec![0.0f64; dim * dim];
        for d in got {
            for i in 0..dim {
                for j in 0..dim {
                    emp_cov[i * dim + j] +=
                        (f64::from(d[i]) - emp_mean[i]) * (f64::from(d[j]) - emp_mean[j]) / n;
                }
            }
        }
        for i in 0..dim {
            let se = (cov[i * dim + i] / n).sqrt();
            assert!(
                (emp_mean[i] - mean[i]).abs() < 5.0 * se + 1e-3,
                "coordinate {i}: mean {} vs {}",
                emp_mean[i],
                mean[i]
            );
        }
        for i in 0..dim {
            for j in 0..dim {
                let scale = (cov[i * dim + i] * cov[j * dim + j]).sqrt().max(1e-9);
                assert!(
                    (emp_cov[i * dim + j] - cov[i * dim + j]).abs() < tol * scale,
                    "cov({i},{j}): {} vs {}",
                    emp_cov[i * dim + j],
                    cov[i * dim + j]
                );
            }
        }
    }

    #[test]
    fn layout_and_prior_scales_match_flat_size() {
        let s = tiny();
        assert_eq!(s.layout().iter().map(|x| x.1).sum::<usize>(), s.flat_size());
        assert_eq!(s.prior_scales().len(), s.flat_size());
        assert_eq!(s.layers, 4);
    }

    #[test]
    fn readout_carries_negative_and_nonunit_values() {
        let s = tiny();
        let mut theta = vec![0.0; s.flat_size()];
        let bias = s.flat_size() - s.output;
        theta[bias] = -2.0;
        theta[bias + 1] = 3.5;
        let out = predict(&theta, &s, &[0.0; 4]).unwrap();
        assert_eq!(out, vec![-2.0, 3.5]);
    }

    #[test]
    fn gradient_matches_finite_differences_in_every_block() {
        let s = tiny();
        let x: Vec<f32> = (0..3 * s.input).map(|i| (i as f32 * 0.13).sin()).collect();
        let y: Vec<f32> = (0..3 * s.output).map(|i| i as f32 * 0.2 - 0.4).collect();
        let target = VectorRegression::new(s, x, y, 0.3, 1.0).unwrap();
        let mut theta = init_flat(&s, 0.15, 9).to_vec1::<f32>().unwrap();
        let t = Tensor::from_vec(theta.clone(), s.flat_size(), &Device::Cpu).unwrap();
        let grad = target
            .mean_loss_and_grad(&t)
            .unwrap()
            .1
            .to_vec1::<f32>()
            .unwrap();
        // One coordinate inside each block, including projection and readout.
        let mut probes = vec![0usize, s.input * s.tokens * s.width + 1];
        let block = 4 * s.width * s.width + 2 * s.width * s.ff;
        for l in 0..s.layers {
            probes.push(s.input * s.tokens * s.width + l * block + 7);
        }
        probes.push(s.flat_size() - s.output);
        for i in probes {
            let eps = 1e-2f32;
            let old = theta[i];
            theta[i] = old + eps;
            let plus = target
                .mean_loss(&Tensor::from_vec(theta.clone(), s.flat_size(), &Device::Cpu).unwrap());
            theta[i] = old - eps;
            let minus = target
                .mean_loss(&Tensor::from_vec(theta.clone(), s.flat_size(), &Device::Cpu).unwrap());
            theta[i] = old;
            let fd = (plus - minus) / (2.0 * eps as f64);
            assert!(
                (fd - grad[i] as f64).abs() < 2e-2,
                "coordinate {i}: fd {fd} grad {}",
                grad[i]
            );
        }
    }

    #[test]
    fn readout_target_matches_the_full_energy_and_gradient_slice() {
        let s = tiny();
        let x: Vec<f32> = (0..3 * s.input).map(|i| (i as f32 * 0.11).cos()).collect();
        let y: Vec<f32> = (0..3 * s.output).map(|i| i as f32 * 0.3 - 0.2).collect();
        let base = VectorRegression::new(s, x, y, 0.3, 1.0).unwrap();
        let flat = init_flat(&s, 0.2, 7).to_vec1::<f32>().unwrap();
        let target = ReadoutTarget::new(base.clone(), flat.clone()).unwrap();
        let range = readout_range(&s);
        assert_eq!(target.dim(), range.len());
        assert_eq!(target.start(), flat[range.clone()].to_vec());
        let draw: Vec<f32> = (0..target.dim()).map(|i| 0.05 * i as f32 - 0.4).collect();
        let (u, grad) = target.energy_and_grad(&draw).unwrap();
        let expanded = target.expand(&draw);
        assert_eq!(expanded[range.clone()], draw[..]);
        assert_eq!(expanded[..range.start], flat[..range.start]);
        let t = Tensor::from_vec(expanded, s.flat_size(), &Device::Cpu).unwrap();
        let (mean, full_grad) = base.mean_loss_and_grad(&t).unwrap();
        let scale = base.lambda() * base.rows() as f64;
        assert!((u - scale * mean).abs() < 1e-9);
        let full = full_grad.to_vec1::<f32>().unwrap();
        for (i, g) in grad.iter().enumerate() {
            assert!((f64::from(*g) - scale * f64::from(full[range.start + i])).abs() < 1e-3);
        }
    }

    #[test]
    fn additive_rows_partition_exactly_once() {
        let s = tiny();
        let x: Vec<f32> = (0..4 * s.input).map(|i| (i as f32 * 0.07).cos()).collect();
        let y: Vec<f32> = (0..4 * s.output).map(|i| i as f32 * 0.1).collect();
        let all = VectorRegression::new(s.clone(), x.clone(), y.clone(), 0.5, 1.0).unwrap();
        let a = VectorRegression::new(
            s.clone(),
            x[..2 * s.input].to_vec(),
            y[..2 * s.output].to_vec(),
            0.5,
            1.0,
        )
        .unwrap();
        let b = VectorRegression::new(
            s.clone(),
            x[2 * s.input..].to_vec(),
            y[2 * s.output..].to_vec(),
            0.5,
            1.0,
        )
        .unwrap();
        let theta = init_flat(&s, 0.1, 4);
        let loss_all = all.mean_loss(&theta);
        let mean_parts = (a.mean_loss(&theta) + b.mean_loss(&theta)) / 2.0;
        assert!((loss_all - mean_parts).abs() < 1e-6);
        assert_eq!(all.n_terms(), 4);
    }

    #[test]
    fn exact_readout_posterior_is_the_gaussian_stationary_point() {
        let s = tiny();
        let x: Vec<f32> = (0..5 * s.input).map(|i| (i as f32 * 0.17).sin()).collect();
        let y: Vec<f32> = (0..5 * s.output).map(|i| i as f32 * 0.21 - 0.3).collect();
        let base = VectorRegression::new(s.clone(), x, y, 0.3, 1.2).unwrap();
        let flat = init_flat(&s, 0.2, 21).to_vec1::<f32>().unwrap();
        let target = ReadoutTarget::new(base, flat).unwrap();
        let prior_c: Vec<f64> = (0..target.dim())
            .map(|i| 2.5e-3 * (1.0 + 0.05 * i as f64))
            .collect();
        let post = target.exact_posterior(&prior_c).unwrap();
        assert_eq!(post.dim(), target.dim());
        // A m = b is exactly the stationarity condition of the total energy,
        // so the gradient of likelihood + prior vanishes at the posterior mean.
        let mean_f32: Vec<f32> = post.mean().iter().map(|v| *v as f32).collect();
        let (_, grad) = target.energy_and_grad(&mean_f32).unwrap();
        for i in 0..target.dim() {
            let total = f64::from(grad[i]) + f64::from(mean_f32[i]) / prior_c[i];
            assert!(
                total.abs() < 2e-2,
                "coordinate {i}: total gradient {total} at the posterior mean"
            );
        }
        // Exact draws must reproduce the analytic mean and covariance.
        let mut rng = StdRng::seed_from_u64(7);
        let draws: Vec<Vec<f32>> = (0..9_000).map(|_| post.draw(&mut rng)).collect();
        quadratic_draws_agree(&draws, post.mean(), &post.covariance(), 0.08);
    }

    #[test]
    fn dense_cn_proposal_samples_a_correlated_gaussian() {
        let a = vec![4.0, 1.5, 1.5, 2.0];
        let det = 4.0 * 2.0 - 1.5 * 1.5;
        let cov = vec![2.0 / det, -1.5 / det, -1.5 / det, 4.0 / det];
        let target = Quadratic { a, dim: 2 };
        // A negligible prior isolates the quadratic target.
        let prior_c = vec![1e6, 1e6];
        let proposal =
            ProposalCov::Dense(DenseCovariance::from_precision(&[4.0, 1.5, 1.5, 2.0], 2).unwrap());
        let cfg = VectorSamplerConfig {
            steps: 40_000,
            burn: 5_000,
            thin: 1,
            rho: 0.5,
            refresh: 0.0,
            prior_sigma: 1.0,
            draws: 35_000,
        };
        let chain =
            run_chain_with_proposal(&target, &prior_c, &proposal, cfg, vec![1.0, -1.0], 3).unwrap();
        assert!(
            chain.accepted > 5_000,
            "acceptance {} too low",
            chain.accepted
        );
        let draws: Vec<Vec<f32>> = chain.draws;
        quadratic_draws_agree(&draws, &[0.0, 0.0], &cov, 0.05);
    }

    #[test]
    fn posterior_reference_cn_centers_the_proposal_on_the_posterior() {
        let p = vec![4.0, 1.0];
        let mu = vec![1.0, -2.0];
        let prior_c = vec![1.0, 4.0];
        let target = ShiftedQuadratic {
            p: p.clone(),
            mu: mu.clone(),
        };
        let mut precision = vec![0.0f64; 4];
        let mut cov = vec![0.0f64; 4];
        let mut mean = vec![0.0f64; 2];
        for i in 0..2 {
            let a = p[i] + 1.0 / prior_c[i];
            precision[i * 2 + i] = a;
            cov[i * 2 + i] = 1.0 / a;
            mean[i] = p[i] * mu[i] / a;
        }
        let dense = DenseCovariance::from_precision(&precision, 2).unwrap();
        let cfg = VectorSamplerConfig {
            steps: 20_000,
            burn: 4_000,
            thin: 1,
            rho: 0.5,
            refresh: 0.0,
            prior_sigma: 1.0,
            draws: 16_000,
        };
        let centered = run_chain_with_posterior_reference(
            &target,
            &prior_c,
            &mean,
            &ProposalCov::Dense(dense.clone()),
            cfg,
            vec![0.0, 0.0],
            9,
        )
        .unwrap();
        let accept = centered.accepted as f64 / centered.proposals as f64;
        assert!(accept > 0.2, "centered acceptance {accept}");
        quadratic_draws_agree(&centered.draws, &mean, &cov, 0.08);
    }

    #[test]
    fn hmc_with_the_posterior_diagonal_recovers_the_gaussian() {
        let a = vec![4.0, 1.5, 1.5, 2.0];
        let det = 4.0 * 2.0 - 1.5 * 1.5;
        let cov = vec![2.0 / det, -1.5 / det, -1.5 / det, 4.0 / det];
        let target = Quadratic { a, dim: 2 };
        let prior_c = vec![1e6, 1e6];
        let mass = vec![cov[0], cov[3]];
        let cfg = HmcConfig {
            steps: 6_000,
            burn: 1_500,
            thin: 1,
            draws: 4_500,
            leapfrog: 5,
            step_size: 1.0,
            adapt_step_size: true,
            target_accept: 0.8,
        };
        let chain = run_hmc_chain(&target, &prior_c, &mass, cfg, vec![1.0, -1.0], 4).unwrap();
        let accept = chain.accepted as f64 / chain.proposals as f64;
        assert!(accept > 0.4, "HMC acceptance {accept}");
        assert!(chain.step_size.is_finite() && chain.step_size > 0.0);
        quadratic_draws_agree(&chain.draws, &[0.0, 0.0], &cov, 0.08);
    }

    #[test]
    fn cn_and_refresh_chains_resume_deterministically() {
        let s = VectorShapes::with_width(3, 1, 2, 4, 2, 4).unwrap();
        let x: Vec<f32> = (0..2 * s.input).map(|i| i as f32 * 0.1).collect();
        let y = vec![0.2, -0.3];
        let target = VectorRegression::new(s.clone(), x, y, 1.0, 1.0).unwrap();
        let cfg = VectorSamplerConfig {
            steps: 20,
            burn: 5,
            thin: 1,
            rho: 0.9,
            refresh: 0.2,
            prior_sigma: 0.12,
            draws: 10,
        };
        let a = run_chain(&target, cfg, init_flat(&s, 0.1, 3).to_vec1().unwrap(), 11).unwrap();
        let b = run_chain(&target, cfg, init_flat(&s, 0.1, 3).to_vec1().unwrap(), 11).unwrap();
        assert_eq!(a.draws, b.draws);
        assert_eq!(a.accepted, b.accepted);
        assert_eq!(a.refresh_attempts + a.cn_attempts, 20);
        assert!(a.refresh_attempts > 0 && a.cn_attempts > 0);
    }
}
