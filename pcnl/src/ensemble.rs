//! R14/P2 — adapted ensemble runner: moderate warm entry, the predeclared
//! slowdown knee rule, member provenance and deterministic checkpoints.
//!
//! The warm-entry phase is deliberately *not* a full-convergence optimizer.
//! It halts at the first of:
//!
//! * entry into the declared good-enough set `L ≤ τ`,
//! * R13 §5's slowdown knee (three consecutive full-loss checks whose
//!   fractional decrease is below `1 %` with a decreasing trend),
//! * the declared step cap (recorded as a capped run, never dropped).
//!
//! Everything that follows runs inside `Φ_{τ,κ,α}` (see
//! [`crate::satisficing`]); the ensemble is a set of independent basins from
//! `init_flat_prior`, never an average of unaligned restarts.  Averaging is
//! [`crate::swa`]'s job and carries its own basin provenance.

use crate::satisficing::Satisficing;
use crate::swa::SwaCheckpoint;
use crate::vector_model::EnergyTarget;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// R13 §5's knee rule, frozen before execution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct KneeRule {
    /// Full-loss checks are taken every `check_every` updates.
    pub check_every: usize,
    /// Fractional-decrease threshold (`0.01` = 1 %).
    pub fraction: f64,
    /// Number of consecutive windows that must satisfy the rule.
    pub window: usize,
    /// Floor for the preceding loss in the fractional denominator.
    pub epsilon: f64,
}

impl Default for KneeRule {
    fn default() -> Self {
        Self {
            check_every: 50,
            fraction: 0.01,
            window: 3,
            epsilon: 1e-12,
        }
    }
}

impl KneeRule {
    pub fn validate(&self) -> Result<(), String> {
        if self.check_every == 0 || self.window == 0 {
            return Err("knee: check_every and window must be non-zero".into());
        }
        if !self.fraction.is_finite() || self.fraction <= 0.0 {
            return Err("knee: fraction must be finite and > 0".into());
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err("knee: epsilon must be finite and > 0".into());
        }
        Ok(())
    }

    /// The index of the last check in the first qualifying window.  Losses
    /// must be strictly decreasing and the fractional drops must be below
    /// `fraction` across `window` consecutive intervals.
    pub fn detect(&self, trace: &[f64]) -> Option<usize> {
        if self.validate().is_err() || trace.len() < self.window + 1 {
            return None;
        }
        let mut drops: Vec<(usize, f64)> = Vec::new();
        for k in 1..trace.len() {
            let prev = trace[k - 1];
            let cur = trace[k];
            if !prev.is_finite() || !cur.is_finite() {
                drops.clear();
                continue;
            }
            let denom = prev.abs().max(self.epsilon);
            drops.push((k, (prev - cur) / denom));
            if drops.len() >= self.window {
                let window = &drops[drops.len() - self.window..];
                // Strictly decreasing losses (`f > 0`) with a fractional
                // drop below the declared threshold on every one of the
                // `window` consecutive intervals.  A plateau that merely
                // oscillates has a non-positive drop and does not qualify.
                let small = window
                    .iter()
                    .all(|(_, f)| f.is_finite() && *f > 0.0 && *f < self.fraction);
                if small {
                    return Some(k);
                }
            }
        }
        None
    }
}

/// Declared warm-entry schedule.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WarmEntryConfig {
    pub max_steps: usize,
    pub lr: f64,
    pub tau: f64,
    pub knee: KneeRule,
}

impl WarmEntryConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_steps == 0 {
            return Err("warm-entry: max_steps must be non-zero".into());
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            return Err("warm-entry: lr must be finite and > 0".into());
        }
        if !self.tau.is_finite() || self.tau <= 0.0 {
            return Err("warm-entry: tau must be finite and > 0".into());
        }
        self.knee.validate()
    }
}

/// Why the warm-entry phase stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarmStop {
    /// Entered the declared ceiling `L ≤ τ`.
    Ceiling,
    /// R13 §5 slowdown knee.
    Knee,
    /// Declared step cap (recorded verbatim, never dropped).
    Cap,
}

impl WarmStop {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ceiling => "ceiling",
            Self::Knee => "knee",
            Self::Cap => "cap",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WarmEntry {
    pub theta: Vec<f32>,
    /// Full-loss checks only (`check_every` spacing).
    pub trace: Vec<f64>,
    pub steps: usize,
    pub stop: WarmStop,
}

/// Adam on the declared loss `L` (no prior), with the knee/ceiling stop rule.
pub fn warm_entry<T: EnergyTarget>(
    target: &T,
    theta0: &[f32],
    config: WarmEntryConfig,
) -> Result<WarmEntry, String> {
    config.validate()?;
    if theta0.len() != target.dim() {
        return Err("warm-entry: start dimension mismatch".into());
    }
    let n = theta0.len();
    let mut theta = theta0.to_vec();
    let mut m = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut trace = Vec::new();
    let mut stop = WarmStop::Cap;
    let mut steps = 0usize;

    let check = |theta: &[f32], steps: usize, trace: &mut Vec<f64>| -> Option<WarmStop> {
        let Some((loss, _)) = target.energy_and_grad(theta) else {
            return Some(WarmStop::Cap);
        };
        if !loss.is_finite() {
            return Some(WarmStop::Cap);
        }
        trace.push(loss);
        if loss <= config.tau {
            return Some(WarmStop::Ceiling);
        }
        if config.knee.detect(trace).is_some() {
            return Some(WarmStop::Knee);
        }
        let _ = steps;
        None
    };

    if let Some(s) = check(&theta, 0, &mut trace) {
        return Ok(WarmEntry {
            theta,
            trace,
            steps,
            stop: s,
        });
    }

    for t in 1..=config.max_steps {
        let Some((loss, grad)) = target.energy_and_grad(&theta) else {
            stop = WarmStop::Cap;
            break;
        };
        if !loss.is_finite() || grad.len() != n || grad.iter().any(|g| !g.is_finite()) {
            stop = WarmStop::Cap;
            break;
        }
        let c1 = 1.0 - b1.powi(t as i32);
        let c2 = 1.0 - b2.powi(t as i32);
        for i in 0..n {
            let g = f64::from(grad[i]);
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            let step = config.lr * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
            theta[i] = (f64::from(theta[i]) - step) as f32;
        }
        steps = t;
        if t % config.knee.check_every == 0
            && let Some(s) = check(&theta, t, &mut trace)
        {
            stop = s;
            break;
        }
    }

    Ok(WarmEntry {
        theta,
        trace,
        steps,
        stop,
    })
}

/// One ensemble member with provenance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Member {
    pub weights: Vec<f32>,
    pub seed: u64,
    pub warm_stop: WarmStop,
    pub warm_steps: usize,
    pub train_loss: f64,
    /// Hinge value at the retained configuration (0 inside a Huberized
    /// plateau).
    pub hinge: f64,
}

/// A homogeneous ensemble of weight configurations plus its intra-basin SWA
/// centers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ensemble {
    pub label: String,
    pub members: Vec<Member>,
    pub swa_centers: Vec<SwaCheckpoint>,
}

impl Ensemble {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            members: Vec::new(),
            swa_centers: Vec::new(),
        }
    }

    pub fn push(&mut self, member: Member) -> Result<(), String> {
        if let Some(first) = self.members.first()
            && first.weights.len() != member.weights.len()
        {
            return Err("ensemble: heterogeneous weight dimensions are not allowed".into());
        }
        if member.weights.iter().any(|w| !w.is_finite()) {
            return Err("ensemble: member contains a non-finite weight".into());
        }
        self.members.push(member);
        Ok(())
    }

    pub fn dim(&self) -> Option<usize> {
        self.members.first().map(|m| m.weights.len())
    }

    pub fn weights(&self) -> Vec<Vec<f32>> {
        self.members.iter().map(|m| m.weights.clone()).collect()
    }

    /// Equal-weight mean of the member weights (only meaningful when the
    /// members are aligned; used for logging, not as a merge).
    pub fn weight_mean(&self) -> Option<Vec<f32>> {
        crate::swa::batch_average(&self.weights()).ok()
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, self.to_json()?).map_err(|e| e.to_string())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::from_json(&text)
    }
}

/// Evaluate a satisficing member's training loss and hinge value.
pub fn member_record<T: EnergyTarget>(
    satisficing: &Satisficing<T>,
    weights: &[f32],
    seed: u64,
    warm: &WarmEntry,
) -> Result<Member, String> {
    let loss = satisficing
        .loss(weights)
        .ok_or("ensemble: non-finite member loss")?;
    let config = satisficing.config();
    Ok(Member {
        weights: weights.to_vec(),
        seed,
        warm_stop: warm.stop,
        warm_steps: warm.steps,
        train_loss: loss,
        hinge: config.psi(loss - config.tau),
    })
}
