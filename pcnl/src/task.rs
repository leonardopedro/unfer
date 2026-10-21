//! The toy sequence tasks (plan Phase 0: "character prediction or
//! modular arithmetic") and the deterministic-training baseline the
//! adaptation is measured against (Phase 4's "SGD/Adam" comparator).
//!
//! Token layout: `0 = BOS`, digits `0..K` are tokens `1..=K`.
//! Sequence `[BOS, a, b]` predicts the next token:
//!
//! * [`Task::A`] — `(a + b) mod K`  (the base task)
//! * [`Task::B`] — `(a + 2b) mod K` (the adaptation target)
//!
//! The full training set *is* the pCNL potential Φ — no minibatching
//! inside the MCMC loop, so the Metropolis ratio is exact for the
//! data the plan's `Φ(θ) = L(D; θ)` names.

use crate::model::{Shapes, Variant, loss_and_grad};
use candle_core::{Device, Result, Tensor};
use rand::SeedableRng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    A,
    B,
}

impl Task {
    pub fn label(&self, a: usize, b: usize, k: usize) -> usize {
        match self {
            Task::A => (a + b) % k,
            Task::B => (a + 2 * b) % k,
        }
    }
}

pub struct Dataset {
    pub k: usize,
    pub train_x: Tensor,
    pub train_y: Tensor,
    pub test_x: Tensor,
    pub test_y: Tensor,
}

fn materialize(task: Task, k: usize, n: usize, seed: u64) -> (Tensor, Tensor) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut xs = Vec::with_capacity(n * 3);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let a = rng.random_range(0..k);
        let b = rng.random_range(0..k);
        xs.push(0u32); // BOS
        xs.push(a as u32 + 1);
        xs.push(b as u32 + 1);
        ys.push(task.label(a, b, k) as u32 + 1);
    }
    (
        Tensor::from_vec(xs, vec![n, 3], &Device::Cpu).unwrap(),
        Tensor::from_vec(ys, vec![n], &Device::Cpu).unwrap(),
    )
}

impl Dataset {
    pub fn new(task: Task, k: usize, n_train: usize, n_test: usize, seed: u64) -> Self {
        let (train_x, train_y) = materialize(task, k, n_train, seed);
        let (test_x, test_y) = materialize(task, k, n_test, seed.wrapping_add(1));
        Self {
            k,
            train_x,
            train_y,
            test_x,
            test_y,
        }
    }

    /// The exact data potential: mean cross-entropy over the whole
    /// training set (constant-factor scaling of Φ only affects the
    /// temperature, so it is mean — and stated here, not implied).
    pub fn loss_and_grad(
        &self,
        theta: &Tensor,
        shapes: &Shapes,
        variant: Variant,
    ) -> Result<(f64, Tensor)> {
        loss_and_grad(theta, shapes, variant, &self.train_x, &self.train_y)
    }

    /// Classification accuracy on the held-out split, in percent.
    pub fn accuracy(&self, theta: &Tensor, shapes: &Shapes, variant: Variant) -> Result<f64> {
        let logits = crate::model::forward(theta, shapes, variant, &self.test_x)?;
        let preds = logits.argmax(1)?.to_vec1::<u32>()?;
        let targets = self.test_y.to_vec1::<u32>()?;
        let hit = preds.iter().zip(&targets).filter(|(p, t)| p == t).count();
        Ok(100.0 * hit as f64 / targets.len() as f64)
    }

    /// Negative log-likelihood (nats) on the held-out split — the
    /// Phase-4 NLL metric.
    pub fn nll(&self, theta: &Tensor, shapes: &Shapes, variant: Variant) -> Result<f64> {
        let logits = crate::model::forward(theta, shapes, variant, &self.test_x)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &self.test_y)?;
        Ok(loss.to_scalar::<f32>()? as f64)
    }

    /// Sum-form mini-batch gradient for SG-pCNL: the mean gradient ×
    /// B ( = Σ_{i∈B} ∇ℓ_i ), which `SgPcnlState` scales by N/B so the
    /// full-potential estimate is N·mean — the plan §3A `g_batch`.
    /// Returns `(mean_loss, sum_grad)`.
    pub fn batch_grad(
        &self,
        theta: &Tensor,
        shapes: &Shapes,
        variant: Variant,
        x: &Tensor,
        y: &Tensor,
    ) -> Result<(f64, Tensor)> {
        let (mean_loss, mean_grad) = loss_and_grad(theta, shapes, variant, x, y)?;
        let b = y.dim(0)? as f64;
        Ok((mean_loss, mean_grad.affine(b, 0.0)?))
    }
}

/// Streaming mini-batch loader (plan Phase 2: "streaming Mini-Batch
/// DataLoader in Candle"): uniform samples *without replacement per
/// batch* over the training split. Each chain owns one loader, so the
/// batch stream is independent of the noise stream (two rngs — a
/// shared one would correlate ξ_t with B_t).
pub struct BatchLoader {
    n: usize,
    batch: usize,
    rng: rand::rngs::StdRng,
}

impl BatchLoader {
    pub fn new(n: usize, batch: usize, seed: u64) -> Self {
        Self {
            n,
            batch: batch.min(n),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// One `(X, Y)` batch `[B, …]` from the training split.
    pub fn next(&mut self, data: &Dataset) -> (Tensor, Tensor) {
        let pick = rand::seq::index::sample(&mut self.rng, self.n, self.batch);
        let ids: Vec<u32> = pick.iter().map(|i| i as u32).collect();
        let n = ids.len();
        let idx = Tensor::from_vec(ids, vec![n], &Device::Cpu).expect("index tensor");
        let x = data.train_x.index_select(&idx, 0).expect("batch x");
        let y = data.train_y.index_select(&idx, 0).expect("batch y");
        (x, y)
    }
}

/// Plain momentum-SGD on the flat vector — the deterministic baseline
/// Phase 4 benchmarks pCNL against, and Phase 3's Task-A pre-trainer.
/// Gradients are clipped to norm `PRETRAIN_CLIP` (both variants alike):
/// the clip is part of the *baseline's* definition, so an unnormalized
/// stack's spike cannot kill the pre-train and leave the adaptation
/// comparison with no starting checkpoint. Returns the final
/// parameters and the loss trace.
pub const PRETRAIN_CLIP: f64 = 5.0;

pub fn momentum_sgd(
    data: &Dataset,
    shapes: &Shapes,
    variant: Variant,
    theta0: Tensor,
    steps: usize,
    lr: f64,
    momentum: f64,
) -> (Tensor, Vec<f64>) {
    let mut theta = theta0;
    let mut vel =
        Tensor::zeros(theta.dims(), candle_core::DType::F32, theta.device()).expect("cpu");
    let mut trace = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (loss, grad) = data.loss_and_grad(&theta, shapes, variant).expect("grad");
        trace.push(loss);
        if !loss.is_finite() {
            break; // the stack diverged; keep the trace as evidence
        }
        // clip to the shared norm cap, then v ← μ·v − lr·g; θ ← θ + v
        let g_norm = grad
            .sqr()
            .and_then(|s| s.sum_all())
            .and_then(|s| s.sqrt())
            .and_then(|s| s.to_scalar::<f32>())
            .map(|v| v as f64)
            .unwrap_or(f64::NAN);
        let grad = if g_norm.is_finite() && g_norm > PRETRAIN_CLIP {
            grad.affine(PRETRAIN_CLIP / g_norm, 0.0).expect("clip")
        } else {
            grad
        };
        vel =
            (vel.affine(momentum, 0.0).expect("v") - grad.affine(lr, 0.0).expect("g")).expect("v'");
        theta = (&theta + &vel).expect("theta'");
    }
    (theta, trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::init_flat;

    fn tiny() -> Shapes {
        Shapes {
            vocab: 9,
            d: 16,
            ff: 32,
            layers: 2,
        }
    }

    #[test]
    fn tasks_are_deterministic_and_distinct() {
        let k = 8;
        assert_eq!(Task::A.label(3, 5, k), 0); // (3+5)%8
        assert_eq!(Task::B.label(3, 5, k), 5); // (3+10)%8
        // A = a+b, B = a+2b coincide exactly when b ≡ 0 (mod K); for
        // b ≠ 0 they must differ, which is what makes B a real target.
        for a in 0..k {
            for b in 1..k {
                assert_ne!(
                    Task::A.label(a, b, k),
                    Task::B.label(a, b, k),
                    "labels must differ at ({a},{b})"
                );
            }
        }
    }

    #[test]
    fn datasets_shape_and_bounds() {
        let d = Dataset::new(Task::A, 8, 64, 32, 5);
        assert_eq!(d.train_x.dims(), &[64, 3]);
        assert_eq!(d.train_y.dims(), &[64]);
        let ys = d.train_y.to_vec1::<u32>().unwrap();
        assert!(
            ys.iter().all(|&y| (1..=8).contains(&y)),
            "labels in token range"
        );
        let xs = d.train_x.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert!(xs.iter().all(|&x| (0..=8).contains(&x)));
    }

    #[test]
    fn momentum_sgd_reduces_loss_on_task_a() {
        let s = tiny();
        let data = Dataset::new(Task::A, 8, 128, 64, 9);
        let theta0 = init_flat(&s, 0.1, 4);
        let (theta, trace) = momentum_sgd(&data, &s, Variant::Cosine, theta0, 150, 0.5, 0.9);
        let first = trace[0];
        let last = *trace.last().unwrap();
        assert!(last < 0.5 * first, "loss {first} -> {last}");
        let acc = data.accuracy(&theta, &s, Variant::Cosine).unwrap();
        assert!(acc > 30.0, "accuracy after pre-training {acc}%"); // chance is 12.5%
    }
}
