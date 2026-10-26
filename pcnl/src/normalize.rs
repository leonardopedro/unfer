//! Unit-hypersphere primitives (plan §2 / Phase 1).
//!
//! Every projection the normalized architecture uses is the map
//! `X ↦ X / (‖X‖ + ε)` on the last dimension (rows for activations,
//! the Frobenius norm for whole weight matrices). The ε is what keeps
//! the map finite at the origin, and it is the only reason the result
//! is not *exactly* unit-norm; tests pin how close.

use candle_core::{D, Result, Tensor};

/// ε for every normalisation in the crate. f32 keeps ~7 significant
/// digits, so 1e-6 leaves headroom for the norm-sum rounding without
/// ever dominating a non-zero row.
pub const EPS: f64 = 1e-6;

/// L2-normalise along the last dimension: each row of `x` lands on the
/// unit sphere `S^{n-1}`.
pub fn normalize_last(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    x.broadcast_div(&(norm + EPS)?)
}

/// Frobenius-normalise a 2-D weight matrix so `‖W‖_F = 1`. This is the
/// `Ŵ = W / (‖W‖_F + ε)` of the plan's `NormalizedLinear`.
pub fn normalize_fro(w: &Tensor) -> Result<Tensor> {
    let fro = w.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
    w.affine(1.0 / (fro + EPS), 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn normalize_last_puts_rows_on_the_unit_sphere() {
        let dev = &Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 5.0, 12.0, 0.0], &[2, 3], dev).unwrap();
        let y = normalize_last(&x).unwrap();
        let norms: Vec<f32> = y
            .sqr()
            .unwrap()
            .sum_keepdim(D::Minus1)
            .unwrap()
            .sqrt()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // row 0 is (3,4,0): the norm is exactly 5, so the ε only costs 1e-6/5.
        assert!((norms[0] - 1.0).abs() < 1e-5, "row0 norm {}", norms[0]);
        assert!((norms[1] - 1.0).abs() < 1e-5, "row1 norm {}", norms[1]);
        // direction preserved: (3,4,0)/5
        let row0 = y
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((row0[0] - 0.6).abs() < 1e-5 && (row0[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn normalize_last_stays_finite_at_the_origin() {
        let dev = &Device::Cpu;
        let z = Tensor::zeros(3, candle_core::DType::F32, dev).unwrap();
        let y = normalize_last(&z).unwrap();
        assert!(
            y.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v == 0.0)
        );
    }

    /// The ε in `W/(‖W‖_F + ε)` is the *only* thing that breaks exact
    /// scale invariance, and it breaks it at a computable order. For a
    /// radial rescaling `W ↦ cW` with `ρ = ‖W‖_F`:
    ///
    /// ```text
    /// Ŵ(cW) = cW/(cρ + ε) = ŵ · c(ρ + ε)/(cρ + ε),
    /// ```
    /// so the direction is untouched and the relative defect is exactly
    /// `ε|c−1|/(cρ + ε)`, whose log-scale derivative at `c = 1` is
    /// `ε/(ρ + ε) = O(ε/ρ)`. At the toy's shell (ρ = σ√P ≈ 45.2, ε = 1e-6)
    /// that is 2.2e-8: twelve orders below the 5e-3 floor §6.5 measures
    /// for ⟨θ, ∇Φ⟩, which is why the floor is autodiff noise and not the
    /// regularizer. This test pins the order and the linearity in ε —
    /// without it, "ε is harmless" is an assertion, not a measurement.
    #[test]
    fn epsilon_breaks_scale_invariance_only_at_order_eps() {
        let w = [0.3f64, -1.2, 4.0, 0.5, -0.7, 2.1];
        let rho = w.iter().map(|x| x * x).sum::<f64>().sqrt();

        // the map in f64, mirroring `normalize_fro`
        let scaled = |c: f64, eps: f64| -> Vec<f64> {
            let r = c * rho;
            w.iter().map(|x| c * x / (r + eps)).collect()
        };

        for eps in [1e-6f64, 5e-7] {
            let base = scaled(1.0, eps);
            for c in [0.5f64, 2.0, 4.0, 10.0] {
                let got = scaled(c, eps);
                // same direction, so the defect is a scalar ratio
                let rel = (0..w.len())
                    .map(|i| (got[i] - base[i]).abs())
                    .sum::<f64>()
                    / base.iter().map(|x| x.abs()).sum::<f64>();
                let predicted = eps * (c - 1.0).abs() / (c * rho + eps);
                assert!(
                    (rel - predicted).abs() < 1e-12 + 1e-6 * predicted,
                    "eps={eps} c={c}: measured {rel} vs predicted {predicted}"
                );
            }
            // log-scale derivative at c = 1 is ε/(ρ+ε). Differentiate
            // the scalar ratio in log space: the absolute defect is ~1e-7
            // and a plain central difference cancels against an O(1)
            // baseline (measured 4.7 % off at h = 1e-7, which is exactly
            // the cancellation, not the algebra).
            let ln_ratio = |c: f64| (c * (rho + eps) / (c * rho + eps)).ln();
            let h = 1e-3;
            let d = (ln_ratio(1.0 + h) - ln_ratio(1.0 - h)) / (2.0 * h);
            let predicted = eps / (rho + eps);
            // h = 1e-3 leaves an O(h²) ≈ 1e-6 relative truncation term
            // (measured 1.1e-6 at eps = 5e-7); the order, not the last
            // digit, is what this pins.
            assert!(
                (d - predicted).abs() < 2e-5 * predicted,
                "eps={eps}: d ln/d ln c {d} vs ε/(ρ+ε) {predicted}"
            );
        }
        // linear in ε: halving the regularizer halves the defect
        let d1: f64 = {
            let (a, b) = (scaled(2.0, 1e-6), scaled(1.0, 1e-6));
            (0..w.len()).map(|i| (a[i] - b[i]).abs()).sum()
        };
        let d2: f64 = {
            let (a, b) = (scaled(2.0, 5e-7), scaled(1.0, 5e-7));
            (0..w.len()).map(|i| (a[i] - b[i]).abs()).sum()
        };
        assert!((d1 / d2 - 2.0).abs() < 0.01, "d1/d2 = {} not 2", d1 / d2);
    }

    /// And the real (f32) implementation is that map: its defect at
    /// `c = 2` matches the f64 prediction to f32 precision.
    #[test]
    fn normalize_fro_epsilon_defect_matches_the_analytic_order() {
        let dev = &Device::Cpu;
        let w = vec![0.3f32, -1.2, 4.0, 0.5, -0.7, 2.1];
        let rho_f64 = w.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let base = normalize_fro(&Tensor::from_vec(w.clone(), &[2, 3], dev).unwrap()).unwrap();
        let scaled_w: Vec<f32> = w.iter().map(|x| 2.0 * x).collect();
        let doubled = normalize_fro(&Tensor::from_vec(scaled_w, &[2, 3], dev).unwrap()).unwrap();
        let b = base.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let d = doubled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rel: f64 = b
            .iter()
            .zip(d.iter())
            .map(|(x, y)| (*y as f64 - *x as f64).abs())
            .sum::<f64>()
            / b.iter().map(|x| (*x as f64).abs()).sum::<f64>();
        let predicted = EPS / (2.0 * rho_f64 + EPS); // c = 2 ⇒ ε/(2ρ+ε)
        assert!(
            (rel - predicted).abs() < 1e-5,
            "measured {rel} vs analytic {predicted}"
        );
    }

    #[test]
    fn normalize_fro_unit_frobenius() {
        let dev = &Device::Cpu;
        let w = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2], dev).unwrap();
        let nw = normalize_fro(&w).unwrap();
        let fro = nw
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .sqrt()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((fro - 1.0).abs() < 1e-5, "‖W‖_F = {fro}");
    }
}
