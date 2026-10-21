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
