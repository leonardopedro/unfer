//! R14/P2 — intra-basin SWA (online averaging, interpolation barriers, output
//! means).
//!
//! `RESEARCH_PLAN.md` §14.2 states the two boundaries this module keeps
//! explicit:
//!
//! 1. constant-LR SGD is not automatically scalar-temperature Langevin, so a
//!    plain SGD knee samples an optimizer-endpoint law, not
//!    `π_{τ,κ,α}`;
//! 2. nonlinear curvature can separate the weight average
//!    `f_{θ_SWA}(x)` from the output average `K⁻¹ Σ_k f_{θ_k}(x)`.
//!
//! Every SWA arm therefore compares the SWA weight mean against the last
//! iterate and against the snapshot output mean on identical inputs, and
//! reports straight-line interpolation loss barriers.  Weight-space averaging
//! is only meaningful inside one connected basin: [`SwaCheckpoint`] records the
//! basin/window provenance so an unaligned restart is never averaged silently.

use serde::{Deserialize, Serialize};

/// Online average `K⁻¹ Σ_k θ_k` without storing the snapshots.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunningAverage {
    count: usize,
    sum: Vec<f64>,
}

impl RunningAverage {
    pub fn new(dim: usize) -> Self {
        Self {
            count: 0,
            sum: vec![0.0; dim],
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn dim(&self) -> usize {
        self.sum.len()
    }

    pub fn push(&mut self, theta: &[f32]) -> Result<(), String> {
        if theta.len() != self.sum.len() {
            return Err(format!(
                "swa: dimension mismatch: got {}, expected {}",
                theta.len(),
                self.sum.len()
            ));
        }
        if theta.iter().any(|v| !v.is_finite()) {
            return Err("swa: snapshot contains a non-finite weight".into());
        }
        for (s, v) in self.sum.iter_mut().zip(theta) {
            *s += f64::from(*v);
        }
        self.count += 1;
        Ok(())
    }

    pub fn mean(&self) -> Result<Vec<f32>, String> {
        if self.count == 0 {
            return Err("swa: average of zero snapshots".into());
        }
        Ok(self
            .sum
            .iter()
            .map(|s| (s / self.count as f64) as f32)
            .collect())
    }

    pub fn mean_f64(&self) -> Result<Vec<f64>, String> {
        if self.count == 0 {
            return Err("swa: average of zero snapshots".into());
        }
        Ok(self.sum.iter().map(|s| s / self.count as f64).collect())
    }
}

/// Batch average, used to check the online accumulator.
pub fn batch_average(snapshots: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    let first = snapshots.first().ok_or("swa: no snapshots")?;
    if first.is_empty() {
        return Err("swa: empty snapshots".into());
    }
    if snapshots.iter().any(|s| s.len() != first.len()) {
        return Err("swa: snapshots differ in dimension".into());
    }
    let mut sum = vec![0.0f64; first.len()];
    for s in snapshots {
        for (acc, v) in sum.iter_mut().zip(s) {
            if !v.is_finite() {
                return Err("swa: snapshot contains a non-finite weight".into());
            }
            *acc += f64::from(*v);
        }
    }
    let k = snapshots.len() as f64;
    Ok(sum.iter().map(|s| (s / k) as f32).collect())
}

/// Straight-line interpolation `(1−t)a + t b`.
pub fn interpolate(a: &[f32], b: &[f32], t: f64) -> Result<Vec<f32>, String> {
    if a.len() != b.len() {
        return Err("swa: interpolation endpoints differ in dimension".into());
    }
    if !t.is_finite() {
        return Err("swa: interpolation parameter must be finite".into());
    }
    Ok(a.iter()
        .zip(b)
        .map(|(&x, &y)| ((1.0 - t) * f64::from(x) + t * f64::from(y)) as f32)
        .collect())
}

/// One evaluated point on a straight line.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarrierPoint {
    pub t: f64,
    pub loss: f64,
}

/// Evaluate the loss along the segment `a → b` at `steps + 1` equally spaced
/// points, including both endpoints.
pub fn straight_line_barrier<F>(
    a: &[f32],
    b: &[f32],
    steps: usize,
    loss: F,
) -> Result<Vec<BarrierPoint>, String>
where
    F: Fn(&[f32]) -> f64,
{
    if steps == 0 {
        return Err("swa: barrier steps must be non-zero".into());
    }
    let mut out = Vec::with_capacity(steps + 1);
    for k in 0..=steps {
        let t = k as f64 / steps as f64;
        let theta = interpolate(a, b, t)?;
        let value = loss(&theta);
        if !value.is_finite() {
            return Err("swa: non-finite loss on the interpolation path".into());
        }
        out.push(BarrierPoint { t, loss: value });
    }
    Ok(out)
}

/// `max_loss(path) − max(loss(a), loss(b))`.  Nonpositive on a convex segment
/// (the midpoint cannot be worse than both endpoints), positive when the
/// straight line crosses a barrier between two basins.
pub fn barrier_height(points: &[BarrierPoint]) -> Option<f64> {
    let max = points
        .iter()
        .map(|p| p.loss)
        .fold(f64::NEG_INFINITY, f64::max);
    let first = points.first()?.loss;
    let last = points.last()?.loss;
    if !max.is_finite() {
        return None;
    }
    Some(max - first.max(last))
}

/// Equal-weight output average `K⁻¹ Σ_k f_k(x)`.
pub fn output_average(outputs: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    let first = outputs.first().ok_or("swa: no outputs")?;
    if outputs.iter().any(|o| o.len() != first.len()) {
        return Err("swa: outputs differ in dimension".into());
    }
    let mut sum = vec![0.0; first.len()];
    for o in outputs {
        for (acc, v) in sum.iter_mut().zip(o) {
            if !v.is_finite() {
                return Err("swa: non-finite output".into());
            }
            *acc += *v;
        }
    }
    let k = outputs.len() as f64;
    Ok(sum.into_iter().map(|s| s / k).collect())
}

/// RMSE between two equal-length vectors.
pub fn rmse(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();
    Some((s / a.len() as f64).sqrt())
}

/// Provenance of a SWA center.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwaKind {
    /// Constant-learning-rate SGD snapshots after the knee.
    SgdConstantLr,
    /// pCNL snapshots inside one basin.
    PcnlWithinBasin,
}

impl SwaKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::SgdConstantLr => "sgd-constant-lr",
            Self::PcnlWithinBasin => "pcnl-within-basin",
        }
    }
}

/// A serializable SWA center with its declared window/spacing and basin
/// provenance.  `basin` is the member/run id the center came from.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwaCheckpoint {
    pub kind: SwaKind,
    pub basin: usize,
    pub window: usize,
    pub spacing: usize,
    pub snapshots: usize,
    pub weights: Vec<f32>,
}

impl SwaCheckpoint {
    pub fn new(
        kind: SwaKind,
        basin: usize,
        window: usize,
        spacing: usize,
        snapshots: &[Vec<f32>],
    ) -> Result<Self, String> {
        if window == 0 || spacing == 0 {
            return Err("swa: window and spacing must be non-zero".into());
        }
        if snapshots.len() != window {
            return Err(format!(
                "swa: declared window {window} but got {} snapshots",
                snapshots.len()
            ));
        }
        Ok(Self {
            kind,
            basin,
            window,
            spacing,
            snapshots: snapshots.len(),
            weights: batch_average(snapshots)?,
        })
    }
}
