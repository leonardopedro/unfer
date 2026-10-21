//! Phase 1 — the normalized architecture (plan §2), in miniature.
//!
//! One flat parameter vector `θ ∈ R^P` drives *both* variants so the
//! Phase-0 comparison isolates the forward map and nothing else:
//! identical `P`, identical shapes, identical task.
//!
//! * `Variant::Cosine` — nGPT-lite: embeddings on the unit sphere,
//!   `Q,K` from Frobenius-normalised weights, logits `τ·(q̂·k̂)` with
//!   `τ = √d` (so the logit *range* matches standard scaled attention,
//!   only the hypersphere constrains it), geodesic (nlerp) residuals
//!   that re-normalise after every sub-layer, and a cosine head tied
//!   to the embedding table.
//! * `Variant::Standard` — the plain transformer: affine-free LN,
//!   dot-product attention with the `1/√d` scale, additive residuals,
//!   tied raw head scaled by `1/√d`.
//!
//! Deviations from the plan, recorded rather than hidden: residuals
//! use nlerp (`normalize(x + f(x))`) instead of a true slerp — same
//! fixed point, a fraction of the cost; the head is weight-tied in
//! both variants to keep `P` identical; `τ` is fixed, not learnable.

use crate::normalize::{normalize_fro, normalize_last};
use candle_core::{D, Device, Result, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Cosine,
    Standard,
}

/// Static shapes of the toy model. `P` is derived, never stored twice.
#[derive(Clone, Copy, Debug)]
pub struct Shapes {
    pub vocab: usize,
    pub d: usize,
    pub ff: usize,
    pub layers: usize,
}

impl Shapes {
    /// Embedding table + per-layer (Q,K,V,O + two MLP matrices).
    /// Both variants: no biases, no LN affine — `P` is variant-independent.
    pub fn flat_size(&self) -> usize {
        let per_layer = 4 * self.d * self.d + 2 * self.d * self.ff;
        self.vocab * self.d + self.layers * per_layer
    }

    /// Parameter layout as `(start, len, shape)` in a fixed order:
    /// embedding `[V, D]`, then per layer `wq, wk, wv, wo [D, D]`,
    /// `w1 [D, F]`, `w2 [F, D]`.
    pub fn layout(&self) -> Vec<(usize, usize, Vec<usize>)> {
        let mut out = Vec::new();
        let mut at = 0usize;
        let mut push = |rows: usize, cols: usize| {
            out.push((at, rows * cols, vec![rows, cols]));
            at += rows * cols;
        };
        push(self.vocab, self.d);
        for _ in 0..self.layers {
            push(self.d, self.d); // wq
            push(self.d, self.d); // wk
            push(self.d, self.d); // wv
            push(self.d, self.d); // wo
            push(self.d, self.ff); // w1
            push(self.ff, self.d); // w2
        }
        debug_assert_eq!(at, self.flat_size());
        out
    }
}

/// A named view over the flat vector: `θ ⇄ {W_q, W_k, …}` (plan §2).
pub struct Slices {
    pub embed: Tensor,
    pub layers: Vec<LayerSlices>,
}

pub struct LayerSlices {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub w1: Tensor,
    pub w2: Tensor,
}

/// Unflatten `θ ∈ [P]` into named tensors (the plan's bidirectional
/// view). Cheap at toy sizes; gradients flow because every tensor is
/// a `narrow`+`reshape` of the caller's leaf.
pub fn unflatten(theta: &Tensor, shapes: &Shapes) -> Result<Slices> {
    let mut tensors = Vec::new();
    for (start, len, shape) in shapes.layout() {
        tensors.push(theta.narrow(0, start, len)?.reshape(shape)?);
    }
    let mut it = tensors.into_iter();
    let embed = it.next().expect("layout has an embedding");
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
    Ok(Slices { embed, layers })
}

/// Affine-free layer normalisation: `(x − μ)/√(var + 1e-5)`. The
/// standard variant's LN; the cosine variant replaces it with the
/// hyperspherical projection itself.
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&((var + 1e-5)?.sqrt()?))
}

/// Causal cosine (or standard dot-product) attention over `[B, T, D]`.
/// Returns `[B, T, D]`.
pub fn attention(
    h: &Tensor, // [B, T, D]
    w: &LayerSlices,
    variant: Variant,
    tau: f64,
) -> Result<Tensor> {
    let (b, t, d) = h.dims3()?;
    // Storage convention (Shapes::layout): every weight is [in, out],
    // so projections are plain `x @ W` — no transpose. (Square Q/K/V/O
    // hid the convention; W1 [D, F] exposed it.)
    let (q, k) = match variant {
        Variant::Cosine => (
            // Q, K from Frobenius-normalised weights, then rows on the
            // sphere: logits = τ·(q̂ᵢ · k̂ⱼ) ∈ [−τ, τ].
            normalize_last(&h.broadcast_matmul(&normalize_fro(&w.wq)?)?)?,
            normalize_last(&h.broadcast_matmul(&normalize_fro(&w.wk)?)?)?,
        ),
        Variant::Standard => (h.broadcast_matmul(&w.wq)?, h.broadcast_matmul(&w.wk)?),
    };
    let v = h.broadcast_matmul(&w.wv)?;
    let mut scores = q.broadcast_matmul(&k.transpose(D::Minus2, D::Minus1)?)?; // [B, T, T]
    scores = match variant {
        Variant::Cosine => scores.affine(tau, 0.0)?,
        Variant::Standard => scores.affine(1.0 / (d as f64).sqrt(), 0.0)?,
    };
    // Causal mask: `-inf` strictly above the diagonal (hand-built —
    // small t×t, and the 0·(−∞) trap of a scaled boolean keeps away).
    let mut mask_v = vec![0f32; t * t];
    for r in 0..t {
        for c in r + 1..t {
            mask_v[r * t + c] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(mask_v, (t, t), h.device())?;
    let scores = scores.broadcast_add(&mask)?;
    let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
    let out = probs.broadcast_matmul(&v)?; // [B, T, D]
    debug_assert_eq!(out.dims(), &[b, t, d]);
    out.broadcast_matmul(&w.wo)
}

/// One transformer layer: norm → attention → geodesic residual →
/// norm → MLP → geodesic residual (cosine), or LN/add (standard).
pub fn layer(x: &Tensor, w: &LayerSlices, variant: Variant, tau: f64) -> Result<Tensor> {
    match variant {
        Variant::Cosine => {
            let a = attention(&normalize_last(x)?, w, variant, tau)?;
            let x = normalize_last(&(x + a)?)?;
            let m = normalize_last(&x)?
                .broadcast_matmul(&normalize_fro(&w.w1)?)?
                .gelu()?
                .broadcast_matmul(&normalize_fro(&w.w2)?)?;
            normalize_last(&(x + m)?)
        }
        Variant::Standard => {
            let a = attention(&layer_norm(x)?, w, variant, tau)?;
            let x = (x + a)?;
            let m = layer_norm(&x)?
                .broadcast_matmul(&w.w1)?
                .gelu()?
                .broadcast_matmul(&w.w2)?;
            x + m
        }
    }
}

/// Forward pass over token sequences `[B, T]`, returning next-position
/// logits `[B, V]` (the last position's prediction).
pub fn forward(
    theta: &Tensor,
    shapes: &Shapes,
    variant: Variant,
    tokens: &Tensor,
) -> Result<Tensor> {
    let tau = (shapes.d as f64).sqrt();
    let s = unflatten(theta, shapes)?;
    // gather rows of the (possibly normalised) table for each token:
    // candle 0.8 has no free `ops::embedding`, and index_select keeps
    // the graph rooted at `theta` either way.
    let gather = |table: &Tensor| -> Result<Tensor> {
        let (b, t) = tokens.dims2()?;
        let flat = tokens.flatten_all()?;
        table.index_select(&flat, 0)?.reshape((b, t, table.dim(1)?))
    };
    let emb = match variant {
        Variant::Cosine => normalize_last(&gather(&s.embed)?)?,
        Variant::Standard => gather(&s.embed)?,
    };
    let mut x = emb;
    for w in &s.layers {
        x = layer(&x, w, variant, tau)?;
    }
    let last = x.narrow(1, x.dim(1)? - 1, 1)?.squeeze(1)?; // [B, D]
    match variant {
        Variant::Cosine => {
            // cosine head, tied to the embedding table, same logit range
            let e = normalize_last(&s.embed)?;
            normalize_last(&last)?.matmul(&e.t()?)?.affine(tau, 0.0)
        }
        Variant::Standard => last
            .matmul(&s.embed.t()?)?
            .affine(1.0 / (shapes.d as f64).sqrt(), 0.0),
    }
}

/// Cross-entropy of the model on `(tokens [B,T], targets [B])` plus
/// `∇_θ` — the loss-and-gradient seam the pCNL kernel depends on.
pub fn loss_and_grad(
    theta: &Tensor,
    shapes: &Shapes,
    variant: Variant,
    tokens: &Tensor,
    targets: &Tensor,
) -> Result<(f64, Tensor)> {
    use candle_core::Var;
    let var = Var::from_tensor(theta)?;
    let logits = forward(var.as_tensor(), shapes, variant, tokens)?;
    let loss = candle_nn::loss::cross_entropy(&logits, targets)?;
    let grads = loss.backward()?;
    let grad = grads
        .get(var.as_tensor())
        .ok_or_else(|| candle_core::Error::Msg("theta got no gradient".into()))?;
    Ok((loss.to_scalar::<f32>()? as f64, grad.clone()))
}

/// A flat start vector: i.i.d. N(0, init_std²) on CPU, seeded.
pub fn init_flat(shapes: &Shapes, init_std: f64, seed: u64) -> Tensor {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::StandardNormal;
    let v: Vec<f32> = (0..shapes.flat_size())
        .map(|_| {
            let z: f32 = rng.sample(normal);
            (z as f64 * init_std) as f32
        })
        .collect();
    Tensor::from_vec(v, vec![shapes.flat_size()], &Device::Cpu).expect("cpu tensor")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Shapes {
        Shapes {
            vocab: 9,
            d: 8,
            ff: 16,
            layers: 2,
        }
    }

    #[test]
    fn flat_layout_covers_all_parameters() {
        let s = tiny();
        let total: usize = s.layout().iter().map(|(_, len, _)| *len).sum();
        assert_eq!(total, s.flat_size());
        // P = V·D + L·(4D² + 2DF) = 72 + 2·(256 + 256) = 1080
        assert_eq!(s.flat_size(), 9 * 8 + 2 * (4 * 64 + 2 * 8 * 16));
    }

    #[test]
    fn unflatten_roundtrips_through_reshape() {
        let s = tiny();
        let theta = init_flat(&s, 0.1, 7);
        let u = unflatten(&theta, &s).unwrap();
        assert_eq!(u.embed.dims(), &[9, 8]);
        assert_eq!(u.layers.len(), 2);
        assert_eq!(u.layers[0].w1.dims(), &[8, 16]);
        assert_eq!(u.layers[1].w2.dims(), &[16, 8]);
        // reshape is a view: reading it back reproduces theta exactly
        let back = u.layers[0]
            .wq
            .reshape(64)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let orig = theta.narrow(0, 72, 64).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(back, orig);
    }

    #[test]
    fn both_variants_forward_and_backprop() {
        let s = tiny();
        let theta = init_flat(&s, 0.2, 3);
        let tokens = Tensor::from_vec(vec![0u32, 1, 2, 0, 3, 4], &[2, 3], &Device::Cpu).unwrap();
        let targets = Tensor::from_vec(vec![3u32, 5], &[2], &Device::Cpu).unwrap();
        for variant in [Variant::Cosine, Variant::Standard] {
            let (loss, g) = loss_and_grad(&theta, &s, variant, &tokens, &targets).unwrap();
            assert!(loss.is_finite() && loss > 0.0, "{variant:?} loss {loss}");
            assert_eq!(g.dims(), &[s.flat_size()]);
            assert!(g.to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite()));
        }
    }
}
