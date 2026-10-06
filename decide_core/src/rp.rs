//! The random-projection (RP / ELM) layer.
//!
//! # What this is
//!
//! Both methods expand the `d`-dimensional embedding to `D`-dimensional random
//! features **before** any analytic layer, because that expansion is where the
//! nonlinearity lives. With `R ∈ R^{d×D}` fixed and random,
//!
//! ```text
//! z = GELU(x R)
//! ```
//!
//! This is the recorded audit defect A3: a linear projection of raw features is
//! not the method.
//!
//! # Determinism
//!
//! `R` is drawn once from a seeded PRNG and then **never** re-derived. That is
//! load-bearing twice over: the accumulated RP-space statistics below stay
//! meaningful because the map is fixed, and two runs with the same seed produce
//! bit-identical state. There is deliberately no way to re-seed an existing
//! `RpLayer`; the embedding model *version* is what invalidates state, not the
//! projection seed (see `decide_update` version checks).

use nalgebra::{DMatrix, DVector};

/// GELU, the activation the papers use after the projection.
///
/// The tanh approximation, matching the reference implementations. `erf`-based
/// GELU differs by ~1e-7 in the mid-range and the choice is immaterial to a
/// ridge fit, but it must be *one* choice or the seeded fixtures stop being
/// reproducible.
///
/// Note this is **not** monotone: it has an interior minimum near `x ≈ -0.75`
/// and approaches zero from below as `x → -∞`. `gelu_has_the_shape_the_papers_actually_use`
/// pins that shape, because an earlier version of this crate asserted
/// monotonicity and was simply wrong.
#[inline]
pub fn gelu(x: f64) -> f64 {
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let x3 = x * x * x;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x3)).tanh())
}

/// A fixed, seeded random projection.
#[derive(Debug, Clone)]
pub struct RpLayer {
    /// `d × D`.
    r: DMatrix<f64>,
}

impl RpLayer {
    /// Draw a fresh `d → D` projection from `seed`.
    ///
    /// Entries are drawn from `N(0, 1/D)`, the standard random-feature scaling:
    /// it keeps `E[‖z‖²] ≈ ‖x‖²` so the ridge penalty `λ` means roughly the same
    /// thing across `D`, which is what makes `D` tunable at all.
    pub fn new(d: usize, big_d: usize, seed: u64) -> Self {
        assert!(d > 0, "embedding dimension must be positive");
        assert!(big_d > 0, "projection dimension must be positive");
        let mut rng = Rng(seed);
        let scale = 1.0 / (big_d as f64).sqrt();
        let mut r = DMatrix::zeros(d, big_d);
        for i in 0..d {
            for j in 0..big_d {
                r[(i, j)] = rng.normal() * scale;
            }
        }
        RpLayer { r }
    }

    /// Input dimension.
    pub fn in_dim(&self) -> usize {
        self.r.nrows()
    }

    /// Projected dimension.
    pub fn out_dim(&self) -> usize {
        self.r.ncols()
    }

    /// The projection matrix itself, for tests and diagnostics.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.r
    }

    /// Expand one embedding: `z = GELU(x R)`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != in_dim()`.
    pub fn expand(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(
            x.len(),
            self.in_dim(),
            "embedding dimension {} does not match the projection's {}",
            x.len(),
            self.in_dim()
        );
        let z = self.r.transpose() * x;
        let mut out = DVector::zeros(self.out_dim());
        for i in 0..self.out_dim() {
            out[i] = gelu(z[i]);
        }
        out
    }

    /// Expand a batch, as a `n × D` matrix.
    pub fn expand_batch(&self, xs: &[DVector<f64>]) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(xs.len(), self.out_dim());
        for (i, x) in xs.iter().enumerate() {
            for j in 0..self.out_dim() {
                m[(i, j)] = self.expand(x)[j];
            }
        }
        m
    }
}

/// xorshift64\*, fixed-seed. Reproducible across platforms, which is the point:
/// the pinned fixtures in D6 are only pinnable if this is.
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Standard normal via Box–Muller.
    fn normal(&mut self) -> f64 {
        // `(next >> 11) as f64 / 2^53` is uniform in [0, 1); shifting keeps the
        // mantissa full-width, which a `% 1_000_000` style reduction would not.
        let u1 = ((self.next() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let u2 = ((self.next() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_is_reproducible_from_its_seed() {
        let a = RpLayer::new(4, 8, 42);
        let b = RpLayer::new(4, 8, 42);
        assert_eq!(a.matrix(), b.matrix(), "same seed, same projection");
        let c = RpLayer::new(4, 8, 43);
        assert_ne!(
            a.matrix(),
            c.matrix(),
            "different seed, different projection"
        );
    }

    #[test]
    fn expansion_lands_in_the_projected_space() {
        let rp = RpLayer::new(3, 16, 7);
        let z = rp.expand(&DVector::from_row_slice(&[1.0, -2.0, 0.5]));
        assert_eq!(z.len(), 16);
    }

    #[test]
    fn gelu_has_the_shape_the_papers_actually_use() {
        // NOT monotone, and the earlier version of this test asserted that it
        // was. GELU has a single interior minimum near x ≈ -0.75 (value ≈ -0.17)
        // and approaches 0 from below as x → -∞, so it dips and then recovers.
        // That is the activation the source methods specify, so the property to
        // pin is its *shape*, not a monotonicity it does not have.
        assert!(gelu(0.0).abs() < 1e-12, "GELU(0) = 0");
        assert!(gelu(1.0) > 0.0);
        assert!(gelu(-0.5) < 0.0);
        // Sign follows the argument: the layer is sign-preserving.
        for x in [-5.0f64, -2.0, -1.0, -0.25, 0.25, 1.0, 3.0, 6.0] {
            assert_eq!(gelu(x).signum(), x.signum(), "sign flipped at x={x}");
        }
        // Strictly increasing on [0, ∞): a ridge fit relies on the positive half
        // being injective.
        let mut prev = f64::NEG_INFINITY;
        for i in 0..200 {
            let x = 12.0 * (i as f64) / 199.0;
            let g = gelu(x);
            assert!(g >= prev, "gelu dipped on the positive half at x={x}");
            prev = g;
        }
        // Exactly one interior minimum: decreasing then increasing on (-∞, 0].
        let mut signs = Vec::new();
        let mut prev = gelu(-6.0);
        for i in 1..400 {
            let x = -6.0 + 6.0 * (i as f64) / 399.0;
            let g = gelu(x);
            let d = g - prev;
            if d.abs() > 1e-12 {
                let s = if d > 0.0 { 1 } else { -1 };
                if signs.last() != Some(&s) {
                    signs.push(s);
                }
            }
            prev = g;
        }
        assert_eq!(
            signs,
            vec![-1, 1],
            "expected exactly one minimum on the negative half, got {signs:?}"
        );
    }

    #[test]
    fn the_projection_does_not_explode_or_collapse_the_scale() {
        // NOT a claim that ‖z‖ ≈ ‖x‖. GELU has a gain of ~0.5 for small
        // arguments, so an ELM layer legitimately shrinks near-zero inputs by
        // about that factor; asserting `ratio ≈ 1` would be asserting a property
        // the activation does not have. What must hold is that the layer is
        // *order-preserving in magnitude* — it neither blows up nor annihilates
        // the signal — which is what would silently break a downstream ridge.
        let d = 8;
        let big_d = 512;
        let rp = RpLayer::new(d, big_d, 11);
        let mut worst: f64 = 0.0;
        for k in 0..400u32 {
            let mut x = DVector::zeros(d);
            for i in 0..d {
                x[i] = ((k * 31 + i as u32 * 17) % 7) as f64 * 0.1 - 0.3 + i as f64 * 0.05;
            }
            let want = x.norm_squared();
            let got = rp.expand(&x).norm_squared();
            let ratio = got / want;
            assert!(
                (0.05..20.0).contains(&ratio),
                "projection changed scale by {ratio}x; the 1/sqrt(D) factor is probably wrong"
            );
            worst = worst.max((ratio - 1.0).abs());
        }
        // Over a whole ensemble the *average* must stay in a sane band, which
        // is what catches a missing or double-applied 1/√D.
        let mut sum = 0.0;
        for k in 0..400u32 {
            let mut x = DVector::zeros(d);
            for i in 0..d {
                x[i] = ((k * 31 + i as u32 * 17) % 7) as f64 * 0.1 - 0.3 + i as f64 * 0.05;
            }
            sum += rp.expand(&x).norm_squared() / x.norm_squared();
        }
        let avg = sum / 400.0;
        assert!(
            (0.05..4.0).contains(&avg),
            "mean scale ratio {avg} is outside the plausible band"
        );
        assert!(worst < 20.0);
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn expanding_a_wrong_dimension_is_a_loud_error() {
        RpLayer::new(3, 8, 1).expand(&DVector::from_element(4, 1.0));
    }
}
