use candle_core::{Device, Tensor};
use num_complex::Complex64;
use crate::registry::StateDictionary;
use nested_fock_algebra::QuantumState;

pub struct TensorState {
    pub real: Tensor,
    pub imag: Tensor,
    pub dim: usize,
}

impl TensorState {
    pub fn from_quantum_state(
        state: &QuantumState,
        registry: &mut StateDictionary,
        device: &Device
    ) -> candle_core::Result<Self> {
        let dim = registry.len();
        let mut real_vec = vec![0.0f64; dim];
        let mut imag_vec = vec![0.0f64; dim];

        for (outer, amp) in &state.components {
            let idx = registry.get_or_insert(outer.clone());
            if idx >= real_vec.len() {
                real_vec.resize(idx + 1, 0.0);
                imag_vec.resize(idx + 1, 0.0);
            }
            real_vec[idx] = amp.re;
            imag_vec[idx] = amp.im;
        }

        let dim = real_vec.len();
        let real = Tensor::from_vec(real_vec, (dim,), device)?;
        let imag = Tensor::from_vec(imag_vec, (dim,), device)?;

        Ok(Self { real, imag, dim })
    }

    pub fn inner_product(&self, other: &Self) -> candle_core::Result<Complex64> {
        // <self | other> = (self_re * other_re + self_im * other_im) + i(self_re * other_im - self_im * other_re)
        
        let re_re = (&self.real * &other.real)?.sum_all()?.to_scalar::<f64>()?;
        let im_im = (&self.imag * &other.imag)?.sum_all()?.to_scalar::<f64>()?;
        let re_im = (&self.real * &other.imag)?.sum_all()?.to_scalar::<f64>()?;
        let im_re = (&self.imag * &other.real)?.sum_all()?.to_scalar::<f64>()?;

        Ok(Complex64::new(re_re + im_im, re_im - im_re))
    }

    pub fn norm_sqr(&self) -> candle_core::Result<f64> {
        let re_sq = (&self.real * &self.real)?.sum_all()?.to_scalar::<f64>()?;
        let im_sq = (&self.imag * &self.imag)?.sum_all()?.to_scalar::<f64>()?;
        Ok(re_sq + im_sq)
    }
}
