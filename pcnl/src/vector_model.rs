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
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

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

/// Initial vector drawn at the prior's own scale: coordinate `i` is
/// `N(0, prior_c[i])` with `prior_c[i] = prior_sigma² · prior_scales[i]²`.
///
/// A flat `std` far below the prior sd leaves the attention logits at ≈0.
/// There `∂logits/∂W_q ∝ W_k x` is second order, so the prior term wins and
/// pulls `W_q, W_k` toward zero — the fixed point where every logit is 0, the
/// softmax is exactly uniform, and the q/k gradient is exactly zero.  The
/// projections then never move and a "four-block" model behaves like a
/// single-path MLP.  At prior scale the logits are `O(1)` and the q/k gradients
/// are first order, so all blocks get to use attention.
pub fn init_flat_prior(shapes: &VectorShapes, prior_sigma: f64, seed: u64) -> Tensor {
    let mut rng = StdRng::seed_from_u64(seed);
    let v: Vec<f32> = shapes
        .prior_scales()
        .iter()
        .map(|scale| {
            let z: f64 = rng.sample(StandardNormal);
            (z * prior_sigma * scale) as f32
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

/// Sampled potential `U(θ) + ½ Σ θ_i² / prior_c_i` and its gradient.  [`adam_fit`]
/// and the samplers use this same objective, so a mode of it is a stationary
/// point for the target they actually explore.
fn potential_energy_and_grad<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    theta: &[f32],
) -> Option<(f64, Vec<f64>)> {
    let (u, grad) = target.energy_and_grad(theta)?;
    let n = theta.len();
    let mut total_u = u;
    let mut total_grad = vec![0.0f64; n];
    for i in 0..n {
        let th = f64::from(theta[i]);
        total_u += 0.5 * th * th / prior_c[i];
        total_grad[i] = f64::from(grad[i]) + th / prior_c[i];
    }
    Some((total_u, total_grad))
}

/// Hutchinson probe of the sampled potential's Hessian at `theta`.
///
/// Rademacher vectors `z ∈ {±1}^d` give exact Hessian–vector products by a
/// central difference of the *gradient* over a step of `step` on every
/// coordinate, so `(g(θ+hz) − g(θ−hz)) / 2h ≈ Hz`; `E[z ∘ Hz] = diag(H)`.
/// The returned `columns` are the unscaled `Hz` vectors, usable as a
/// limited-memory basis.  Cost is `2·probes` gradient evaluations, which is
/// the only way to see the stiff directions a short pilot walk never visits.
#[derive(Debug, Clone)]
pub struct HessianProbe {
    /// Hutchinson estimate of the Hessian diagonal (the metric's diagonal).
    pub diag: Vec<f64>,
    /// `H z_i` for each probe vector `z_i`.
    pub columns: Vec<Vec<f64>>,
}

pub fn hessian_probe<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    theta: &[f32],
    probes: usize,
    step: f64,
    seed: u64,
) -> Option<HessianProbe> {
    let d = theta.len();
    assert_eq!(prior_c.len(), d);
    assert!(probes > 0 && step > 0.0);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut diag = vec![0.0f64; d];
    let mut columns = Vec::with_capacity(probes);
    for _ in 0..probes {
        let z: Vec<f64> = (0..d)
            .map(|_| {
                if rng.sample::<f64, _>(StandardNormal) >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let plus: Vec<f32> = (0..d)
            .map(|i| (f64::from(theta[i]) + step * z[i]) as f32)
            .collect();
        let minus: Vec<f32> = (0..d)
            .map(|i| (f64::from(theta[i]) - step * z[i]) as f32)
            .collect();
        let (_, gp) = potential_energy_and_grad(target, prior_c, &plus)?;
        let (_, gm) = potential_energy_and_grad(target, prior_c, &minus)?;
        let col: Vec<f64> = (0..d).map(|i| (gp[i] - gm[i]) / (2.0 * step)).collect();
        for i in 0..d {
            diag[i] += z[i] * col[i] / probes as f64;
        }
        columns.push(col);
    }
    Some(HessianProbe { diag, columns })
}

/// Adam on the sampled potential `U(θ) + ½ Σ θ_i² / prior_c_i`, i.e. a mode of
/// the target the samplers actually use.  [`adam_fit`] minimizes the mean loss
/// alone, so with the sampling prior added its optimum is not stationary; on
/// the full 57,378-coordinate target the residual gradient is large enough to
/// freeze an HMC step size.  Returns the iterate and the potential trace.
pub fn adam_fit_energy<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    theta0: Vec<f32>,
    steps: usize,
    lr: f64,
) -> (Vec<f32>, Vec<f64>) {
    assert_eq!(theta0.len(), target.dim());
    assert_eq!(prior_c.len(), target.dim());
    let mut theta = theta0;
    let n = theta.len();
    let (mut m, mut v) = (vec![0.0f64; n], vec![0.0f64; n]);
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut trace = Vec::with_capacity(steps);
    for t in 1..=steps {
        let Some((total_u, total_grad)) = potential_energy_and_grad(target, prior_c, &theta) else {
            break;
        };
        trace.push(total_u);
        let (c1, c2) = (1.0 - b1.powi(t as i32), 1.0 - b2.powi(t as i32));
        for i in 0..n {
            let g = total_grad[i];
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            let step = lr * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
            theta[i] = (f64::from(theta[i]) - step) as f32;
        }
    }
    (theta, trace)
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
/// 57,378-coordinate R13 size the CN kernel's integrated autocorrelation time is
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
    /// Draw the trajectory length uniformly from `1..=2·leapfrog` for each
    /// proposal instead of using a fixed `leapfrog`.  Both the forward map and
    /// its reverse use the same drawn length, so the proposal stays symmetric.
    /// This breaks leapfrog resonance, where a fixed length returns the
    /// trajectory near its start and the chain appears frozen while the MH
    /// acceptance stays high.
    pub jitter_leapfrog: bool,
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
            jitter_leapfrog: false,
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

/// Low-rank-plus-diagonal Gaussian covariance `C = D + W Λ Wᵀ` estimated from
/// pilot draws, with `D = diag(v)`, unit-norm columns `W` and `k` leading
/// sample-covariance directions.
///
/// It is used as the HMC metric through its inverse (`M = C⁻¹`), applied with
/// Woodbury's identity, so a leapfrog step costs `O(d·k)` plus one `k × k`
/// solve.  A diagonal metric cannot whiten the data Hessian's off-diagonal
/// structure; the low-rank part carries the directions the pilot actually
/// explored.  As with any pilot estimate, directions the pilot never visited
/// fall back to the diagonal residual and are not whitened.
#[derive(Clone, Debug)]
pub struct LowRankCovariance {
    dim: usize,
    rank: usize,
    v: Vec<f64>,
    w: Vec<f64>,        // dim × rank, row-major
    lam: Vec<f64>,      // rank
    w_over_v: Vec<f64>, // dim × rank, row-major
    chol: Vec<f64>,     // rank × rank lower factor of Λ⁻¹ + WᵀD⁻¹W
}

impl LowRankCovariance {
    /// Estimate `C` from pilot draws.  `rank` leading sample-covariance
    /// directions are kept; the diagonal residual is floored at
    /// `floor_frac × max_i var_i` for numerical stability.
    pub fn from_draws(
        draws: &[Vec<f32>],
        rank: usize,
        floor_frac: f64,
    ) -> candle_core::Result<Self> {
        let n = draws.len();
        if n < 2 || rank == 0 {
            return Err(candle_core::Error::Msg(
                "low-rank covariance needs at least 2 draws and rank >= 1".into(),
            ));
        }
        let dim = draws[0].len();
        if dim == 0 || draws.iter().any(|d| d.len() != dim) {
            return Err(candle_core::Error::Msg(
                "pilot draws must share one non-zero dimension".into(),
            ));
        }
        let rank = rank.min(n - 1).min(dim);
        let nf = n as f64;
        let dof = (nf - 1.0).max(1.0);
        let mean: Vec<f64> = (0..dim)
            .into_par_iter()
            .map(|k| draws.iter().map(|d| f64::from(d[k])).sum::<f64>() / nf)
            .collect();
        let var: Vec<f64> = (0..dim)
            .into_par_iter()
            .map(|k| {
                draws
                    .iter()
                    .map(|d| {
                        let x = f64::from(d[k]) - mean[k];
                        x * x
                    })
                    .sum::<f64>()
                    / dof
            })
            .collect();
        // Gram of the centered draws: G[i, j] = (x_i − m)·(x_j − m) / dof.
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let di = &draws[i];
                (0..=i)
                    .map(|j| {
                        let dj = &draws[j];
                        let mut s = 0.0f64;
                        for k in 0..dim {
                            s += (f64::from(di[k]) - mean[k]) * (f64::from(dj[k]) - mean[k]);
                        }
                        s / dof
                    })
                    .collect()
            })
            .collect();
        let mut g = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                g[i * n + j] = rows[i][j];
                g[j * n + i] = rows[i][j];
            }
        }
        let eig = DMatrix::from_row_slice(n, n, &g).symmetric_eigen();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
        let sqrt_dof = dof.sqrt();
        let mut w = vec![0.0f64; dim * rank];
        let mut lam = vec![0.0f64; rank];
        for (slot, &idx) in order.iter().take(rank).enumerate() {
            if !(eig.eigenvalues[idx] > 0.0) {
                continue;
            }
            let u = eig.eigenvectors.column(idx);
            let col: Vec<f64> = (0..dim)
                .into_par_iter()
                .map(|k| {
                    let mut s = 0.0f64;
                    for i in 0..n {
                        s += (f64::from(draws[i][k]) - mean[k]) * u[i];
                    }
                    s / sqrt_dof
                })
                .collect();
            let norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if !(norm > 0.0) {
                continue;
            }
            lam[slot] = norm * norm;
            for k in 0..dim {
                w[k * rank + slot] = col[k] / norm;
            }
        }
        let var_max = var.iter().cloned().fold(0.0f64, f64::max);
        let floor = (floor_frac * var_max).max(f64::MIN_POSITIVE);
        let mut v = vec![0.0f64; dim];
        for k in 0..dim {
            let mut explained = 0.0f64;
            for j in 0..rank {
                let wj = w[k * rank + j];
                explained += lam[j] * wj * wj;
            }
            v[k] = (var[k] - explained).max(floor);
        }
        let mut small = vec![0.0f64; rank * rank];
        for a in 0..rank {
            for b in 0..=a {
                let mut acc = 0.0f64;
                for k in 0..dim {
                    acc += w[k * rank + a] * w[k * rank + b] / v[k];
                }
                if a == b {
                    acc += 1.0 / lam[a].max(f64::MIN_POSITIVE);
                }
                small[a * rank + b] = acc;
                small[b * rank + a] = acc;
            }
        }
        let (chol, _) = cholesky_lower(&small, rank).ok_or_else(|| {
            candle_core::Error::Msg("low-rank metric is not positive definite".into())
        })?;
        let w_over_v = (0..dim * rank)
            .map(|i| w[i] / v[i / rank])
            .collect::<Vec<_>>();
        Ok(Self {
            dim,
            rank,
            v,
            w,
            lam,
            w_over_v,
            chol,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// `C x = D x + W Λ Wᵀ x`.
    fn apply_cov(&self, x: &[f64]) -> Vec<f64> {
        let mut coords = vec![0.0f64; self.rank];
        for j in 0..self.rank {
            let mut s = 0.0f64;
            for k in 0..self.dim {
                s += self.w[k * self.rank + j] * x[k];
            }
            coords[j] = self.lam[j] * s;
        }
        (0..self.dim)
            .map(|k| {
                let mut y = self.v[k] * x[k];
                for j in 0..self.rank {
                    y += self.w[k * self.rank + j] * coords[j];
                }
                y
            })
            .collect()
    }

    /// `C⁻¹ x` by Woodbury's identity.
    fn solve(&self, x: &[f64]) -> Vec<f64> {
        let t: Vec<f64> = (0..self.dim).map(|k| x[k] / self.v[k]).collect();
        let mut rhs = vec![0.0f64; self.rank];
        for j in 0..self.rank {
            let mut s = 0.0f64;
            for k in 0..self.dim {
                s += self.w[k * self.rank + j] * t[k];
            }
            rhs[j] = s;
        }
        let y = forward_substitute(&self.chol, self.rank, &rhs);
        let y = back_substitute_transpose(&self.chol, self.rank, &y);
        (0..self.dim)
            .map(|k| {
                let mut out = t[k];
                for j in 0..self.rank {
                    out -= self.w_over_v[k * self.rank + j] * y[j];
                }
                out
            })
            .collect()
    }

    /// `N(0, C⁻¹)` momentum: solve a `N(0, C)` draw.
    fn draw(&self, rng: &mut StdRng) -> Vec<f64> {
        let z: Vec<f64> = (0..self.dim)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        let zk: Vec<f64> = (0..self.rank)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        let q: Vec<f64> = (0..self.dim)
            .map(|k| {
                let mut y = self.v[k].sqrt() * z[k];
                for j in 0..self.rank {
                    y += self.w[k * self.rank + j] * self.lam[j].sqrt() * zk[j];
                }
                y
            })
            .collect();
        self.solve(&q)
    }
}

/// HMC metric: a diagonal `K(p) = ½ Σ p_i²/m_i`, or the exact whitened metric
/// `u = Lᵀ(θ − m)` with unit mass, where `L Lᵀ` is the posterior precision.
/// In the whitened coordinates a Gaussian posterior is exactly standard normal,
/// so unit step sizes are well scaled.
#[derive(Clone, Copy)]
enum HmcMetric<'a> {
    Diagonal(&'a [f64]),
    Whitened { cov: &'a DenseCovariance },
    LowRank { cov: &'a LowRankCovariance },
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
            Self::LowRank { cov } => cov.draw(rng),
        }
    }

    fn dim(&self) -> usize {
        match self {
            Self::Diagonal(m) => m.len(),
            Self::Whitened { cov } => cov.dim(),
            Self::LowRank { cov } => cov.dim(),
        }
    }

    /// `dθ/dt` for momentum `p`.
    fn velocity(&self, p: &[f64]) -> Vec<f64> {
        match self {
            Self::Diagonal(m) => p.iter().zip(*m).map(|(pi, mi)| pi / mi).collect(),
            Self::Whitened { cov, .. } => cov.apply_cov_factor(p),
            // `M = C⁻¹`, so `dθ/dt = M⁻¹p = C p`.
            Self::LowRank { cov } => cov.apply_cov(p),
        }
    }

    /// `dp/dt` for a potential gradient.
    fn force(&self, grad: &[f64]) -> Vec<f64> {
        match self {
            Self::Diagonal(_) => grad.to_vec(),
            Self::Whitened { cov, .. } => cov.apply_cov_factor_transpose(grad),
            Self::LowRank { .. } => grad.to_vec(),
        }
    }

    fn kinetic(&self, p: &[f64]) -> f64 {
        match self {
            Self::Diagonal(m) => 0.5 * p.iter().zip(*m).map(|(pi, mi)| pi * pi / mi).sum::<f64>(),
            Self::Whitened { .. } => 0.5 * p.iter().map(|pi| pi * pi).sum::<f64>(),
            Self::LowRank { cov } => {
                0.5 * p
                    .iter()
                    .zip(cov.apply_cov(p))
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            }
        }
    }
}

/// Fixed-length HMC (leapfrog) with optional dual-averaging step-size
/// adaptation during warmup.  `mass` is the diagonal metric
/// `K(p) = ½ Σ p_i²/m_i`, so pass the posterior **precision** (inverse
/// variance) to whiten a Gaussian target: with `U = ½θᵀAθ` the dynamics are
/// `θ″ = −M⁻¹Aθ`, which is unit-frequency only for `M = A`.  Passing the
/// variance instead squares the curvature range and drives the adapted step
/// size to zero on ill-conditioned targets.  Every accepted move uses the full
/// `exp(H₀ − H₁)` MH ratio, so a rejected trajectory leaves the state untouched.
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

/// HMC with the low-rank-plus-diagonal metric `M = C⁻¹`, where `C` is a
/// covariance estimated from pilot draws.  Every MH ratio still uses the full
/// target in `θ`; the estimate only sets the metric.
pub fn run_hmc_chain_low_rank<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    cov: &LowRankCovariance,
    config: HmcConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<HmcChain> {
    if cov.dim() != target.dim() {
        return Err(candle_core::Error::Msg(
            "low-rank HMC metric must be target-sized".into(),
        ));
    }
    run_hmc_with_metric(
        target,
        prior_c,
        &HmcMetric::LowRank { cov },
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
        let leapfrog = if config.jitter_leapfrog && config.leapfrog > 1 {
            rng.gen_range(1..=2 * config.leapfrog)
        } else {
            config.leapfrog
        };
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
        for l in 0..leapfrog {
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
            if l + 1 < leapfrog {
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

/// One tempered HMC proposal at inverse temperature `beta` with a fixed metric.
///
/// `current` holds the *tempered* potential `βU` and gradient `β∇U`, so the
/// acceptance ratio is the ordinary one and the density is `exp(−βU)`.
fn tempered_step<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    metric: &HmcMetric,
    beta: f64,
    leapfrog: usize,
    eps: f64,
    rng: &mut StdRng,
    theta: &mut Vec<f32>,
    current: &mut TotalEnergy,
) -> (bool, f64) {
    let dim = theta.len();
    let mut p = metric.draw(rng);
    let mut proposed = theta.clone();
    let (init_u, init_k) = (current.u, metric.kinetic(&p));
    let mut grad = current.grad.clone();
    let mut force = metric.force(&grad);
    for i in 0..dim {
        p[i] -= 0.5 * eps * force[i];
    }
    let mut last: Option<TotalEnergy> = None;
    for l in 0..leapfrog {
        let velocity = metric.velocity(&p);
        for i in 0..dim {
            proposed[i] = (f64::from(proposed[i]) + eps * velocity[i]) as f32;
        }
        match total_energy(target, prior_c, &proposed) {
            Some(ev) => {
                let scaled = TotalEnergy {
                    u: beta * ev.u,
                    grad: ev.grad.iter().map(|g| beta * g).collect(),
                };
                grad = scaled.grad.clone();
                last = Some(scaled);
            }
            None => return (false, 0.0),
        }
        force = metric.force(&grad);
        if l + 1 < leapfrog {
            for i in 0..dim {
                p[i] -= eps * force[i];
            }
        }
    }
    let Some(next) = last else {
        return (false, 0.0);
    };
    for i in 0..dim {
        p[i] -= 0.5 * eps * force[i];
    }
    let log_alpha = (init_u + init_k) - (next.u + metric.kinetic(&p));
    let prob = if log_alpha >= 0.0 {
        1.0
    } else {
        log_alpha.exp()
    };
    if prob >= 1.0 || (prob > 0.0 && rng.random::<f64>() < prob) {
        *theta = proposed;
        *current = next;
        (true, prob)
    } else {
        (false, prob)
    }
}

/// Parallel-tempering (replica-exchange) HMC configuration.
#[derive(Clone, Debug)]
pub struct TemperConfig {
    pub steps: usize,
    pub burn: usize,
    pub thin: usize,
    pub draws: usize,
    pub leapfrog: usize,
    pub step_size: f64,
    pub adapt_step_size: bool,
    pub target_accept: f64,
    /// Inverse temperatures, ascending; the last must be `1.0` (the target).
    pub ladder: Vec<f64>,
    /// Swap adjacent replicas every this many steps.
    pub swap_interval: usize,
}

impl TemperConfig {
    fn validate(&self) -> candle_core::Result<()> {
        if self.ladder.is_empty() || *self.ladder.last().unwrap() != 1.0 {
            return Err(candle_core::Error::Msg(
                "tempering ladder must end at beta = 1".into(),
            ));
        }
        if self.ladder.windows(2).any(|w| !(w[0] < w[1]))
            || self.ladder.iter().any(|b| !(*b > 0.0 && *b <= 1.0))
        {
            return Err(candle_core::Error::Msg(
                "tempering ladder must be strictly increasing in (0, 1]".into(),
            ));
        }
        if self.steps == 0 || self.thin == 0 || self.swap_interval == 0 || self.leapfrog == 0 {
            return Err(candle_core::Error::Msg(
                "tempering steps/thin/swap_interval/leapfrog must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

/// Cold-replica result of a parallel-tempering run.
#[derive(Clone, Debug)]
pub struct TemperChain {
    /// The `β = 1` replica, shaped like an ordinary HMC chain.
    pub cold: HmcChain,
    pub swaps: usize,
    pub swap_accepted: usize,
    /// Mean acceptance per replica, hottest first.
    pub replica_accept: Vec<f64>,
    /// Untempered potential at each replica's final state, hottest first.
    pub replica_final_u: Vec<f64>,
}

/// Parallel tempering over the full parameter vector.
///
/// Replicas sample `exp(−β_k U)` with the same metric, and adjacent replicas
/// swap states with the exact ratio `min(1, exp((β_i − β_j)(U_i − U_j)))`, so the
/// cold chain's marginal is unchanged.  Draws are recorded from the cold chain
/// only; the point is to let it cross between basins the single-temperature
/// chains never leave.  All randomness comes from `seed`-derived streams, so a
/// run is reproducible.
#[allow(clippy::too_many_arguments)]
pub fn run_parallel_tempering<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    mass: &[f64],
    config: TemperConfig,
    theta0: Vec<f32>,
    seed: u64,
) -> candle_core::Result<TemperChain> {
    config.validate()?;
    let dim = target.dim();
    if mass.len() != dim || mass.iter().any(|m| !m.is_finite() || *m <= 0.0) {
        return Err(candle_core::Error::Msg(
            "tempering mass must be finite, positive and target-sized".into(),
        ));
    }
    if theta0.len() != dim {
        return Err(candle_core::Error::Msg(
            "tempering theta0 must be target-sized".into(),
        ));
    }
    let metric = HmcMetric::Diagonal(mass);
    let k = config.ladder.len();
    let (gamma, t0, kappa) = (0.05f64, 10.0f64, 0.75f64);
    let mut thetas: Vec<Vec<f32>> = vec![theta0; k];
    let mut currents: Vec<TotalEnergy> = Vec::with_capacity(k);
    for theta in &thetas {
        let ev = total_energy(target, prior_c, theta).ok_or_else(|| {
            candle_core::Error::Msg("initial vector has no finite objective".into())
        })?;
        currents.push(ev);
    }
    let mut rngs: Vec<StdRng> = (0..k)
        .map(|i| StdRng::seed_from_u64(seed ^ (0x7e11 + i as u64 * 7919)))
        .collect();
    let mut log_eps = vec![config.step_size.ln(); k];
    let mut log_eps_bar = log_eps.clone();
    let mut hbar = vec![0.0f64; k];
    let mut eps = vec![config.step_size; k];
    let mut adapting = vec![config.adapt_step_size; k];
    let mut accept_sum = vec![0.0f64; k];
    let mut accepted = vec![0usize; k];
    let mut proposals = vec![0usize; k];
    let mut grad_evals = 0usize;
    let mut cold_draws = Vec::with_capacity(config.draws);
    let (mut swaps, mut swap_accepted) = (0usize, 0usize);

    for step in 0..config.steps {
        for i in 0..k {
            if adapting[i] && step == config.burn {
                eps[i] = log_eps_bar[i].exp();
                adapting[i] = false;
            } else if adapting[i] {
                eps[i] = log_eps[i].exp();
            }
            proposals[i] += 1;
            // tempered_step evaluates `leapfrog + 1` potentials for an accepted
            // trajectory; count what it actually spends.
            grad_evals += config.leapfrog + 1;
            let (ok, prob) = tempered_step(
                target,
                prior_c,
                &metric,
                config.ladder[i],
                config.leapfrog,
                eps[i],
                &mut rngs[i],
                &mut thetas[i],
                &mut currents[i],
            );
            if ok {
                accepted[i] += 1;
            }
            accept_sum[i] += prob;
            if adapting[i] {
                let m = (step + 1) as f64;
                let eta = 1.0 / (m + t0);
                hbar[i] = (1.0 - eta) * hbar[i] + eta * (config.target_accept - prob);
                log_eps[i] = (10.0 * config.step_size).ln() - (m.sqrt() / gamma) * hbar[i];
                let w = m.powf(-kappa);
                log_eps_bar[i] = w * log_eps[i] + (1.0 - w) * log_eps_bar[i];
            }
        }

        if step % config.swap_interval == 0 {
            for i in 0..k - 1 {
                swaps += 1;
                let (bi, bj) = (config.ladder[i], config.ladder[i + 1]);
                let (ui, uj) = (currents[i].u / bi, currents[i + 1].u / bj);
                let log_alpha = (bi - bj) * (ui - uj);
                let prob = if log_alpha >= 0.0 {
                    1.0
                } else {
                    log_alpha.exp()
                };
                if prob >= 1.0 || (prob > 0.0 && rngs[0].random::<f64>() < prob) {
                    let (left, right) = thetas.split_at_mut(i + 1);
                    std::mem::swap(&mut left[i], &mut right[0]);
                    // Recompute so each replica holds its own β-scaled energy.
                    for (j, beta) in [(i, bi), (i + 1, bj)] {
                        if let Some(ev) = total_energy(target, prior_c, &thetas[j]) {
                            currents[j] = TotalEnergy {
                                u: beta * ev.u,
                                grad: ev.grad.iter().map(|g| beta * g).collect(),
                            };
                        }
                    }
                    grad_evals += 2;
                    swap_accepted += 1;
                }
            }
        }

        if step >= config.burn
            && (step - config.burn) % config.thin == 0
            && cold_draws.len() < config.draws
        {
            cold_draws.push(thetas[k - 1].clone());
        }
    }

    let cold_eps = if config.adapt_step_size {
        log_eps_bar[k - 1].exp()
    } else {
        config.step_size
    };
    Ok(TemperChain {
        cold: HmcChain {
            theta: thetas[k - 1].clone(),
            draws: cold_draws,
            proposals: proposals[k - 1],
            accepted: accepted[k - 1],
            grad_evals,
            step_size: cold_eps,
            mean_accept: accept_sum[k - 1] / proposals[k - 1].max(1) as f64,
        },
        swaps,
        swap_accepted,
        replica_accept: accept_sum
            .iter()
            .zip(&proposals)
            .map(|(s, p)| s / (*p).max(1) as f64)
            .collect(),
        replica_final_u: (0..k).map(|i| currents[i].u / config.ladder[i]).collect(),
    })
}

/// Hit-and-run slice-sampling configuration.
#[derive(Clone, Debug)]
pub struct HitRunConfig {
    pub moves: usize,
    pub burn: usize,
    pub thin: usize,
    pub draws: usize,
    /// Initial slice width in prior-sd units along the sampled direction.
    pub width: f64,
    /// Cap on stepping-out expansions per side.
    pub max_expand: usize,
    /// Cap on shrink steps per move.
    pub max_shrink: usize,
}

impl HitRunConfig {
    fn validate(&self) -> candle_core::Result<()> {
        if self.moves == 0 || self.thin == 0 || self.width <= 0.0 {
            return Err(candle_core::Error::Msg(
                "hit-and-run moves/thin must be non-zero and width positive".into(),
            ));
        }
        Ok(())
    }
}

/// Hit-and-run with slice sampling: a *global* move that does not depend on
/// gradient scale.
///
/// Each move draws a random unit direction in prior-whitened coordinates, so a
/// step of `t = 1` is one prior sd, and samples the exact conditional of the
/// target along that ray with Neal's stepping-out/shrinkage slice sampler.  The
/// move is exact (`π`-invariant), it needs no gradient, and — unlike HMC — a
/// slice that spans a barrier can cross it in one step, which is what a
/// multimodal target needs.  The trade-off is that one move costs tens of
/// potential evaluations instead of `leapfrog + 1`.
///
/// Draws are recorded after `burn`; `step_size` reports the mean `|t|` moved and
/// `grad_evals` the number of potential evaluations spent.
pub fn run_hit_and_run<T: EnergyTarget>(
    target: &T,
    prior_c: &[f64],
    config: HitRunConfig,
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
        return Err(candle_core::Error::Msg(
            "hit-and-run theta0 must be target-sized".into(),
        ));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut theta = theta0;
    let mut current = total_energy(target, prior_c, &theta)
        .ok_or_else(|| candle_core::Error::Msg("initial vector has no finite objective".into()))?;
    let mut evals = 1usize;
    let mut draws = Vec::with_capacity(config.draws);
    let mut moved = 0usize;
    let mut step_sum = 0.0f64;
    let sqrt_c: Vec<f64> = prior_c.iter().map(|c| c.sqrt()).collect();
    let mut point = vec![0.0f32; dim];
    for move_index in 0..config.moves {
        // Random direction, unit in prior-whitened coordinates.
        let mut dir = vec![0.0f64; dim];
        let mut norm2 = 0.0f64;
        for (d, sc) in dir.iter_mut().zip(&sqrt_c) {
            let z = rng.sample::<f64, _>(StandardNormal);
            *d = sc * z;
            norm2 += *d * *d;
        }
        if !(norm2 > 0.0) {
            continue;
        }
        let inv = 1.0 / norm2.sqrt();
        for d in dir.iter_mut() {
            *d *= inv;
        }

        let eval = |t: f64, point: &mut Vec<f32>, evals: &mut usize| -> Option<f64> {
            *evals += 1;
            for i in 0..dim {
                point[i] = (f64::from(theta[i]) + t * dir[i]) as f32;
            }
            total_energy(target, prior_c, point).map(|e| e.u)
        };

        let log_slice = -current.u + rng.random::<f64>().ln();
        let width = config.width;
        let mut left = -width * rng.random::<f64>();
        let mut right = left + width;
        let mut expansions = 0;
        while expansions < config.max_expand {
            match eval(left, &mut point, &mut evals) {
                Some(u) if -u > log_slice => left -= width,
                _ => break,
            }
            expansions += 1;
        }
        while expansions < config.max_expand {
            match eval(right, &mut point, &mut evals) {
                Some(u) if -u > log_slice => right += width,
                _ => break,
            }
            expansions += 1;
        }

        let mut t = left + (right - left) * rng.random::<f64>();
        let mut accepted = false;
        let mut shrinks = 0;
        loop {
            let u = eval(t, &mut point, &mut evals);
            match u {
                Some(u) if -u >= log_slice => {
                    let ev = total_energy(target, prior_c, &point)
                        .expect("the accepted point is finite");
                    evals += 1;
                    theta = point.clone();
                    current = ev;
                    accepted = true;
                    break;
                }
                _ => {
                    if t < 0.0 {
                        left = t;
                    } else {
                        right = t;
                    }
                    t = left + (right - left) * rng.random::<f64>();
                    shrinks += 1;
                    if shrinks >= config.max_shrink {
                        break;
                    }
                }
            }
        }
        if accepted {
            moved += 1;
            step_sum += (t / width).abs();
        }
        if move_index >= config.burn
            && (move_index - config.burn) % config.thin == 0
            && draws.len() < config.draws
        {
            draws.push(theta.clone());
        }
    }
    Ok(HmcChain {
        theta,
        draws,
        proposals: config.moves,
        accepted: moved,
        grad_evals: evals,
        step_size: step_sum / moved.max(1) as f64,
        mean_accept: moved as f64 / config.moves as f64,
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

    /// Inverse of a row-major symmetric 3×3 matrix, by cofactors.
    fn invert3(a: &[f64]) -> Vec<f64> {
        let c = [
            a[4] * a[8] - a[5] * a[7],
            a[5] * a[6] - a[3] * a[8],
            a[3] * a[7] - a[4] * a[6],
        ];
        let det = a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
        assert!(det.abs() > 1e-12);
        let d = 1.0 / det;
        vec![
            c[0] * d,
            (a[2] * a[7] - a[1] * a[8]) * d,
            (a[1] * a[5] - a[2] * a[4]) * d,
            c[1] * d,
            (a[0] * a[8] - a[2] * a[6]) * d,
            (a[2] * a[3] - a[0] * a[5]) * d,
            c[2] * d,
            (a[1] * a[6] - a[0] * a[7]) * d,
            (a[0] * a[4] - a[1] * a[3]) * d,
        ]
    }

    #[test]
    fn adam_fit_energy_finds_the_sampled_potential_mode() {
        // A zero-mean Gaussian with a diagonal prior: the mode of
        // U(θ) + ½Σθ²/c is the origin.  `adam_fit` alone would also find it
        // here, but this checks the prior term is included and the iterate
        // converges from a distant start.
        let a = vec![4.0, 1.5, 1.5, 2.0];
        let target = Quadratic {
            a: a.clone(),
            dim: 2,
        };
        let prior_c = vec![0.5, 0.25];
        let (mode, trace) = adam_fit_energy(&target, &prior_c, vec![3.0, -2.0], 4_000, 1e-2);
        assert!(trace.len() == 4_000);
        assert!(
            trace.last().copied().unwrap() < trace[0],
            "potential did not decrease"
        );
        assert!(
            mode.iter().map(|v| f64::from(*v).abs()).fold(0.0, f64::max) < 1e-2,
            "mode {mode:?} is not the origin"
        );
        // The residual gradient of the sampled potential vanishes there.
        let (_, grad) = target.energy_and_grad(&mode).unwrap();
        for i in 0..2 {
            let total = f64::from(grad[i]) + f64::from(mode[i]) / prior_c[i];
            assert!(total.abs() < 1e-1, "coordinate {i}: residual {total}");
        }
    }

    #[test]
    fn hit_and_run_recovers_the_gaussian_and_crosses_a_barrier() {
        let cov = vec![2.0, 1.2, 0.0, 1.2, 1.5, 0.3, 0.0, 0.3, 0.8];
        let target = Quadratic {
            a: invert3(&cov),
            dim: 3,
        };
        let cfg = HitRunConfig {
            moves: 6_000,
            burn: 1_000,
            thin: 2,
            draws: 2_500,
            width: 1.0,
            max_expand: 32,
            max_shrink: 32,
        };
        let out = run_hit_and_run(&target, &[1e6; 3], cfg, vec![2.0, 2.0, 2.0], 9).unwrap();
        assert_eq!(out.draws.len(), 2_500);
        assert!(
            out.mean_accept > 0.99,
            "slice moves always land: {}",
            out.mean_accept
        );
        quadratic_draws_agree(&out.draws, &[0.0; 3], &cov, 0.12);
    }

    #[test]
    fn prior_scale_init_keeps_the_attention_projections_trainable() {
        let shapes = VectorShapes::with_width(64, 2, 8, 16, 2, 3).unwrap();
        let layout = shapes.layout();
        let prior_sigma = 0.05;
        // The prior's own sd, per coordinate.
        let flat = init_flat_prior(&shapes, prior_sigma, 3)
            .to_vec1::<f32>()
            .unwrap();
        let proj = &flat[layout[0].0..layout[0].0 + layout[0].1];
        let proj_sd =
            (proj.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / proj.len() as f64).sqrt();
        let want = prior_sigma * (shapes.input as f64).sqrt();
        assert!(
            (proj_sd - want).abs() < 0.1 * want,
            "projection sd {proj_sd} vs prior sd {want}"
        );

        // The point of the scale: q/k gradients are first order, not the
        // second-order remnant a near-uniform softmax leaves.
        let teacher = SyntheticTeacher::new(shapes.input, shapes.output, 7);
        let x: Vec<f32> = (0..128)
            .map(|i| ((i * 37 % 61) as f32 - 30.0) / 30.0)
            .collect();
        let y = teacher.targets(&x, 0.05, 7);
        let target = VectorRegression::new(shapes, x, y, 0.05, 1.0).unwrap();
        let (start, len, _) = layout[1]; // block 0, wq
        let q_rms = |theta: Vec<f32>| {
            let (_, g) = EnergyTarget::energy_and_grad(&target, &theta).unwrap();
            (g[start..start + len]
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                / len as f64)
                .sqrt()
        };
        let legacy = q_rms(init_flat(&shapes, 0.02, 3).to_vec1::<f32>().unwrap());
        let prior = q_rms(flat);
        assert!(
            prior > 5.0 * legacy,
            "prior-scale q gradient {prior:e} vs legacy {legacy:e}"
        );
    }

    #[test]
    fn parallel_tempering_cold_chain_recovers_the_gaussian() {
        let cov = vec![2.0, 1.2, 0.0, 1.2, 1.5, 0.3, 0.0, 0.3, 0.8];
        let target = Quadratic {
            a: invert3(&cov),
            dim: 3,
        };
        let cfg = TemperConfig {
            steps: 4_000,
            burn: 1_000,
            thin: 1,
            draws: 3_000,
            leapfrog: 4,
            step_size: 1.0,
            adapt_step_size: true,
            target_accept: 0.8,
            ladder: vec![0.1, 0.35, 0.7, 1.0],
            swap_interval: 5,
        };
        let out =
            run_parallel_tempering(&target, &[1e6; 3], &[1.0; 3], cfg, vec![1.0, -1.0, 0.5], 5)
                .unwrap();
        let accept = out.cold.accepted as f64 / out.cold.proposals as f64;
        assert!(accept > 0.3, "cold acceptance {accept}");
        assert!(out.swap_accepted > 0, "no replica swaps were accepted");
        assert_eq!(out.replica_accept.len(), 4);
        assert_eq!(out.replica_final_u.len(), 4);
        // The cold chain's marginal is the untempered target.
        quadratic_draws_agree(&out.cold.draws, &[0.0; 3], &cov, 0.12);
    }

    #[test]
    fn hessian_probe_recovers_the_analytic_diagonal() {
        // ½θᵀAθ plus ½Σθ²/c has Hessian A + diag(1/c), so the Hutchinson
        // diagonal must reproduce that exactly (up to probe noise).
        let a = vec![4.0, 1.5, 1.5, 2.0];
        let target = Quadratic {
            a: a.clone(),
            dim: 2,
        };
        let prior_c = vec![0.5, 0.25];
        let probe = hessian_probe(&target, &prior_c, &[0.3, -0.2], 2048, 1e-4, 7).expect("probe");
        let want = [4.0 + 1.0 / 0.5, 2.0 + 1.0 / 0.25];
        for i in 0..2 {
            let rel = (probe.diag[i] - want[i]).abs() / want[i];
            assert!(
                rel < 2e-2,
                "coordinate {i}: {} vs {}",
                probe.diag[i],
                want[i]
            );
        }
        assert_eq!(probe.columns.len(), 2048);
        assert_eq!(probe.columns[0].len(), 2, "columns are full-dimension Hz");
        // Hutchinson is unbiased: a second probe seed must agree on the trace.
        let again = hessian_probe(&target, &prior_c, &[0.3, -0.2], 2048, 1e-4, 11).expect("probe");
        let (t1, t2) = (
            probe.diag.iter().sum::<f64>(),
            again.diag.iter().sum::<f64>(),
        );
        assert!((t1 - t2).abs() < 2e-1, "trace {t1} vs {t2}");
    }

    #[test]
    fn low_rank_metric_hmc_recovers_a_correlated_gaussian() {
        let cov = vec![2.0, 1.2, 0.0, 1.2, 1.5, 0.3, 0.0, 0.3, 0.8];
        let precision = invert3(&cov);
        let target = Quadratic {
            a: precision,
            dim: 3,
        };
        // Reference draws define the pilot-based covariance estimate.
        let (chol, _) = cholesky_lower(&cov, 3).unwrap();
        let mut rng = StdRng::seed_from_u64(31);
        let pilot: Vec<Vec<f32>> = (0..2_000)
            .map(|_| {
                let z: Vec<f64> = (0..3)
                    .map(|_| rng.sample::<f64, _>(StandardNormal))
                    .collect();
                let mut x = vec![0.0f64; 3];
                for i in 0..3 {
                    for j in 0..=i {
                        x[i] += chol[i * 3 + j] * z[j];
                    }
                }
                x.iter().map(|v| *v as f32).collect()
            })
            .collect();
        let metric = LowRankCovariance::from_draws(&pilot, 2, 1e-6).unwrap();
        assert_eq!(metric.dim(), 3);
        assert_eq!(metric.rank(), 2);
        let cfg = HmcConfig {
            steps: 6_000,
            burn: 1_500,
            thin: 1,
            draws: 4_500,
            leapfrog: 5,
            step_size: 1.0,
            adapt_step_size: true,
            target_accept: 0.8,
            jitter_leapfrog: false,
        };
        let chain =
            run_hmc_chain_low_rank(&target, &[1e6; 3], &metric, cfg, vec![1.0, -1.0, 0.5], 4)
                .unwrap();
        let accept = chain.accepted as f64 / chain.proposals as f64;
        assert!(accept > 0.4, "low-rank HMC acceptance {accept}");
        assert!(chain.step_size.is_finite() && chain.step_size > 0.0);
        quadratic_draws_agree(&chain.draws, &[0.0; 3], &cov, 0.08);
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
        let target = Quadratic {
            a: a.clone(),
            dim: 2,
        };
        let prior_c = vec![1e6, 1e6];
        // The metric is the precision, not the covariance.
        let mass = vec![a[0], a[3]];
        let cfg = HmcConfig {
            steps: 6_000,
            burn: 1_500,
            thin: 1,
            draws: 4_500,
            leapfrog: 5,
            step_size: 1.0,
            adapt_step_size: true,
            target_accept: 0.8,
            jitter_leapfrog: false,
        };
        let chain = run_hmc_chain(&target, &prior_c, &mass, cfg, vec![1.0, -1.0], 4).unwrap();
        let accept = chain.accepted as f64 / chain.proposals as f64;
        assert!(accept > 0.4, "HMC acceptance {accept}");
        assert!(chain.step_size.is_finite() && chain.step_size > 0.0);
        quadratic_draws_agree(&chain.draws, &[0.0, 0.0], &cov, 0.08);
    }

    #[test]
    fn hmc_with_the_exact_whitened_metric_recovers_the_gaussian() {
        let a = vec![4.0, 1.5, 1.5, 2.0];
        let det = 4.0 * 2.0 - 1.5 * 1.5;
        let cov = vec![2.0 / det, -1.5 / det, -1.5 / det, 4.0 / det];
        let target = Quadratic {
            a: a.clone(),
            dim: 2,
        };
        let prior_c = vec![1e6, 1e6];
        let mass = DenseCovariance::from_precision(&a, 2).unwrap();
        let cfg = HmcConfig {
            steps: 6_000,
            burn: 1_500,
            thin: 1,
            draws: 4_500,
            leapfrog: 5,
            step_size: 1.0,
            adapt_step_size: true,
            target_accept: 0.8,
            jitter_leapfrog: false,
        };
        let chain =
            run_hmc_chain_whitened(&target, &prior_c, &mass, cfg, vec![1.0, -1.0], 4).unwrap();
        let accept = chain.accepted as f64 / chain.proposals as f64;
        assert!(accept > 0.4, "whitened HMC acceptance {accept}");
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
