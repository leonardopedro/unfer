//! Posterior-trajectory persistence — plan §1's `safetensors` row
//! ("zero-copy weight serialization to save and stream posterior
//! parameter trajectories to disk during sampling", Week 2 milestone).
//!
//! One chain's thinned trajectory becomes one `.safetensors` file whose
//! tensors are `s00000, s00001, …`, each the flat `[P]` parameter vector
//! in little-endian f32. Writing is per chain so the peak extra buffer is
//! a single chain's trajectory, not the whole run.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Result, Tensor};
use safetensors::tensor::{Dtype, SafeTensors, TensorView, serialize_to_file};

fn candle_err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("safetensors: {e}"))
}

/// Slugs a variant name (`"cosine/nGPT-lite"`) into a filename stem.
pub fn slug(variant: &str) -> String {
    variant
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Write `samples` (each a flat `[P]` f32 tensor) to
/// `{dir}/{variant_slug}_chain{chain}.safetensors`, creating `dir` if
/// needed. Returns the path written and the sample count.
pub fn save_chain_trajectory(
    dir: &Path,
    variant: &str,
    chain: usize,
    samples: &[Tensor],
) -> Result<(String, usize)> {
    if samples.is_empty() {
        return Ok((String::new(), 0));
    }
    std::fs::create_dir_all(dir)
        .map_err(|e| candle_err(format!("create_dir_all {}: {e}", dir.display())))?;
    let path = dir.join(format!("{}_chain{chain}.safetensors", slug(variant)));

    // Two phases so the byte buffers outlive the views borrowing them.
    let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(samples.len());
    for s in samples {
        let flat = s.to_vec1::<f32>()?;
        let mut buf = Vec::with_capacity(flat.len() * 4);
        for v in flat {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        bufs.push(buf);
    }
    let mut views: HashMap<String, TensorView> = HashMap::with_capacity(bufs.len());
    for (i, (buf, s)) in bufs.iter().zip(samples.iter()).enumerate() {
        let view = TensorView::new(Dtype::F32, s.dims().to_vec(), buf).map_err(candle_err)?;
        views.insert(format!("s{i:05}"), view);
    }
    serialize_to_file(views, &None, &path).map_err(candle_err)?;
    Ok((path.display().to_string(), samples.len()))
}

/// Read a trajectory file back as `[[P]; K]`, ordered by tensor name.
/// The verification counterpart of [`save_chain_trajectory`].
pub fn load_trajectory(path: &Path) -> Result<Vec<Vec<f32>>> {
    let buf =
        std::fs::read(path).map_err(|e| candle_err(format!("read {}: {e}", path.display())))?;
    let tensors = SafeTensors::deserialize(&buf).map_err(candle_err)?;
    let mut names: Vec<&str> = tensors.iter().map(|(n, _)| n).collect();
    names.sort_unstable();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let t = tensors.tensor(name).map_err(candle_err)?;
        if t.dtype() != Dtype::F32 {
            return Err(candle_err(format!(
                "tensor {name} is {:?}, expected F32",
                t.dtype()
            )));
        }
        let bytes = t.data();
        let (words, _rest) = bytes.as_chunks::<4>();
        let row: Vec<f32> = words.iter().map(|c| f32::from_le_bytes(*c)).collect();
        out.push(row);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_is_filename_safe() {
        assert_eq!(slug("cosine/nGPT-lite"), "cosine_nGPT-lite");
        assert_eq!(slug("standard softmax"), "standard_softmax");
    }

    #[test]
    fn trajectory_roundtrips_bit_exactly() {
        let dir = std::env::temp_dir().join(format!("pcnl_persist_{}", std::process::id()));
        let samples = vec![
            Tensor::from_vec(
                (0..8).map(|i| i as f32 * 0.25).collect(),
                &[8usize],
                &candle_core::Device::Cpu,
            )
            .unwrap(),
            Tensor::from_vec(vec![1.5f32; 8], &[8usize], &candle_core::Device::Cpu).unwrap(),
        ];
        let (path, n) = save_chain_trajectory(&dir, "cosine/nGPT-lite", 3, &samples).unwrap();
        assert_eq!(n, 2);
        let back = load_trajectory(Path::new(&path)).unwrap();
        assert_eq!(back.len(), 2);
        for (orig, got) in samples.iter().zip(back.iter()) {
            assert_eq!(orig.to_vec1::<f32>().unwrap(), *got);
        }
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn empty_trajectory_writes_nothing() {
        let dir = std::env::temp_dir().join(format!("pcnl_persist_empty_{}", std::process::id()));
        let (path, n) = save_chain_trajectory(&dir, "x", 0, &[]).unwrap();
        assert_eq!((path.as_str(), n), ("", 0));
        assert!(!dir.exists());
    }
}
