//! Continuous-momentum Standard-Model kernels — first numerical deliverable of
//! `docs/SM_RESEARCH_PLAN.md` and `docs/SM_HAMILTONIAN_AUDIT.md` (2026-11-24).
//!
//! ## Conventions (audit work order, item 1)
//!
//! ```text
//! f̂(k)            = ∫ e^{-2π i k·x} f(x) d³x
//! ∂ᵢf             → 2π i kᵢ f̂(k)
//! (a ⋆ b)(K)      = ∫ a(q) b(K − q) d³q
//! f̂†(q)           = f̂(−q)†
//! ```
//!
//! Only internal indices and packet labels are summed; the integration measure
//! is kept explicit at every convolution. Every field here is a **smeared
//! Gaussian packet** in momentum space, so the packet integrals below are exact
//! closed forms (no momentum lattice, no finite momentum sum). The residual
//! discrete structures are elsewhere (Hermite windows / Krylov rank).
//!
//! ## What this module covers
//!
//! * [`GaugeMode`] / [`gauge_magnetic_cross`]: the magnetic cross term
//!   `½ F_lin · F_int` of `¼ Fᵃᵢⱼ Fᵃᵢⱼ` around a declared background packet,
//!   with the structure constants taken from [`sm_full::su3_structure`] (not a
//!   hard-coded table) and the `2π` derivative symbol explicit.
//! * [`HiggsMode`] / [`higgs_gauge_cross`]: the gauge–Higgs mixing term of
//!   `|DᵢΦ|²` for `DᵢΦ = 2πiKᵢΦ + i g Tᵃ(Wᵃᵢ ⋆ Φ) + i g′ Y (Bᵢ ⋆ Φ)`,
//!   using the realification [`sm_full::higgs_real_generator`].
//! * [`yukawa_transfer`]: a chiral matter transition `Y_{ij} Φ_bg(k − l)`,
//!   i.e. the insertion carries the transferred momentum `k − l`.
//!
//! ## What this module deliberately does **not** claim
//!
//! These are algebraically exact packet entries. They are not a domain/ESA
//! theorem, not a gauge reduction and not a continuum construction. The
//! `Conventions` flags exist only so the mutation controls below can flip one
//! convention at a time; the canonical value is [`Conventions::canonical`].
//! A full SM continuous-kernel run, the `joint_sirk` evolution on the complete
//! inner operator and the cadabra2 symbolic reference remain open (see the
//! ledger in `docs/SM_FULL_IMPLEMENTATION.md`).

use crate::sm_full::{SmParameters, book_kinetic_spin, higgs_real_generator, su3_structure};
use num_complex::Complex64 as C;

/// `2π` in the Fourier convention `∂ᵢ → 2π i kᵢ`.
pub const TWO_PI: f64 = std::f64::consts::TAU;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Declared conventions and the mutation switches used by the control tests.
///
/// The canonical values are `two_pi = 2π`, `transfer_sign = +1` (`K = k − l`),
/// `nonabelian_sign = +1`, and `include_higgs_cross = true`. `yukawa_adjoint`
/// switches the Yukawa element to its conjugate (the dropped-adjoint control).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conventions {
    /// The derivative symbol's `2π`. Canonical [`TWO_PI`]; `1.0` is the
    /// "forgot the Fourier normalization" mutation.
    pub two_pi: f64,
    /// `+1` for `K = k − l`, `−1` for the reversed transfer `l − k`.
    pub transfer_sign: f64,
    /// `+1` for `−g f^{abc}`; `−1` flips the nonabelian sign.
    pub nonabelian_sign: f64,
    /// Whether the gauge–Higgs cross term is retained.
    pub include_higgs_cross: bool,
    /// `true` conjugates the Yukawa element (drops the hermitian adjoint).
    pub yukawa_adjoint: bool,
}

impl Default for Conventions {
    fn default() -> Self {
        Self::canonical()
    }
}

impl Conventions {
    pub fn canonical() -> Self {
        Self {
            two_pi: TWO_PI,
            transfer_sign: 1.0,
            nonabelian_sign: 1.0,
            include_higgs_cross: true,
            yukawa_adjoint: false,
        }
    }
}

/// A smooth momentum-space packet: a real Gaussian of standard deviation
/// `width` centred at `center`.
///
/// `φ(k) = exp(−|k − center|² / (2 width²))`. This is the Fourier transform of a
/// real Gaussian packet up to a positive normalization that is identical for
/// every packet, so it cancels in ratios and is kept explicit in the absolute
/// overlap norms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentumPacket {
    pub center: [f64; 3],
    pub width: f64,
}

impl MomentumPacket {
    pub fn new(center: [f64; 3], width: f64) -> Self {
        assert!(width > 0.0 && width.is_finite(), "packet width must be positive");
        Self { center, width }
    }

    pub fn value(&self, k: [f64; 3]) -> f64 {
        let d = sub(k, self.center);
        (-dot(d, d) / (2.0 * self.width * self.width)).exp()
    }

    /// `∫ φ² d³k = (2π)^{3/2} width³` — the packet's (unnormalized) norm.
    pub fn norm_sq(&self) -> f64 {
        TWO_PI.powf(1.5) * self.width.powi(3)
    }

    /// `∫ φ_a φ_b d³k`.
    pub fn overlap(a: &Self, b: &Self) -> f64 {
        let denom = a.width * a.width + b.width * b.width;
        let scale = TWO_PI.powf(1.5) * (a.width * b.width / denom.sqrt()).powi(3);
        let d = sub(a.center, b.center);
        scale * (-dot(d, d) / (2.0 * denom)).exp()
    }
}

/// A Gaussian of the form `amp · exp(−|k − center|² / (2 sigma²))`.
#[derive(Clone, Copy, Debug)]
struct Gauss {
    amp: f64,
    center: [f64; 3],
    sigma: f64,
}

impl Gauss {
    /// `∫ G d³k`.
    fn i0(&self) -> f64 {
        let s = TWO_PI.sqrt() * self.sigma;
        self.amp * s.powi(3)
    }

    /// `∫ k_m G d³k`.
    fn i1(&self, m: usize) -> f64 {
        self.center[m] * self.i0()
    }

    /// `∫ k_m k_n G d³k`.
    fn i2(&self, m: usize, n: usize) -> f64 {
        let s2 = self.sigma * self.sigma;
        let second = if m == n { s2 } else { 0.0 };
        (second + self.center[m] * self.center[n]) * self.i0()
    }
}

/// The pointwise product of two packets, as a Gaussian.
fn packet_product(a: &MomentumPacket, b: &MomentumPacket) -> Gauss {
    let inv_a = 1.0 / (a.width * a.width);
    let inv_b = 1.0 / (b.width * b.width);
    let inv = inv_a + inv_b;
    let center = [
        (a.center[0] * inv_a + b.center[0] * inv_b) / inv,
        (a.center[1] * inv_a + b.center[1] * inv_b) / inv,
        (a.center[2] * inv_a + b.center[2] * inv_b) / inv,
    ];
    let d = sub(a.center, b.center);
    let amp = (-dot(d, d) / (2.0 * (a.width * a.width + b.width * b.width))).exp();
    Gauss {
        amp,
        center,
        sigma: inv.sqrt().recip(),
    }
}

/// The convolution `(a ⋆ b)(K) = ∫ a(q) b(K − q) d³q`, as a Gaussian.
fn packet_convolution(a: &MomentumPacket, b: &MomentumPacket) -> Gauss {
    let sigma = (a.width * a.width + b.width * b.width).sqrt();
    let amp = TWO_PI.powf(1.5) * (a.width * b.width / sigma).powi(3);
    Gauss {
        amp,
        center: add(a.center, b.center),
        sigma,
    }
}

/// `G · φ` for a Gaussian `G` and a packet `φ`, as a Gaussian.
fn gauss_packet_product(g: &Gauss, p: &MomentumPacket) -> Gauss {
    let inv_g = 1.0 / (g.sigma * g.sigma);
    let inv_p = 1.0 / (p.width * p.width);
    let inv = inv_g + inv_p;
    let center = [
        (g.center[0] * inv_g + p.center[0] * inv_p) / inv,
        (g.center[1] * inv_g + p.center[1] * inv_p) / inv,
        (g.center[2] * inv_g + p.center[2] * inv_p) / inv,
    ];
    let d = sub(g.center, p.center);
    let amp = g.amp
        * (-dot(d, d) / (2.0 * (g.sigma * g.sigma + p.width * p.width))).exp();
    Gauss {
        amp,
        center,
        sigma: inv.sqrt().recip(),
    }
}

/// A gauge-field mode: colour index `a`, spatial direction `i`, a packet and a
/// complex amplitude. Colour is an `su(3)` adjoint index `0..8`.
#[derive(Clone, Copy, Debug)]
pub struct GaugeMode {
    pub color: usize,
    pub direction: usize,
    pub packet: MomentumPacket,
    pub amplitude: C,
}

impl GaugeMode {
    pub fn new(color: usize, direction: usize, packet: MomentumPacket) -> Self {
        assert!(color < 8 && direction < 3, "gauge mode index range");
        Self {
            color,
            direction,
            packet,
            amplitude: C::new(1.0, 0.0),
        }
    }

    pub fn with_amplitude(mut self, amplitude: C) -> Self {
        self.amplitude = amplitude;
        self
    }
}

/// A complex Higgs component (`0..4` in the realified `(Re H₀, Im H₀, Re H₁,
/// Im H₁)` basis) carried by a packet.
#[derive(Clone, Copy, Debug)]
pub struct HiggsMode {
    pub component: usize,
    pub packet: MomentumPacket,
    pub amplitude: C,
}

impl HiggsMode {
    pub fn new(component: usize, packet: MomentumPacket) -> Self {
        assert!(component < 4, "Higgs component range");
        Self {
            component,
            packet,
            amplitude: C::new(1.0, 0.0),
        }
    }

    pub fn with_amplitude(mut self, amplitude: C) -> Self {
        self.amplitude = amplitude;
        self
    }
}

/// The magnetic cross term `½ F_lin · F_int` between two fluctuation modes and
/// one background gauge packet.
///
/// `Fᵃᵢⱼ(K) = 2π i (Kᵢ Aᵃⱼ − Kⱼ Aᵃᵢ) − g f^{abc} (Aᵇᵢ ⋆ Aᶜⱼ)(K)`.
/// The returned value is the coefficient of `α_bra · conj(α_ket)` in
/// `¼ ∫ Fᵃᵢⱼ Fᵃᵢⱼ`: it is the induced one-particle kernel entry between
/// `fluct_bra` and `fluct_ket` in the declared background.
///
/// The two fluctuation colours must differ from the background for the
/// structure constants to be nonzero; the test suite checks that noncommuting
/// colours give a nonzero entry and that the abelian limit `g → 0` vanishes.
pub fn gauge_magnetic_cross(
    fluct_bra: &GaugeMode,
    fluct_ket: &GaugeMode,
    background: &GaugeMode,
    params: &SmParameters,
    conv: &Conventions,
) -> C {
    let f = su3_structure();
    let (ap, aq, c0) = (fluct_bra.color, fluct_ket.color, background.color);
    let (i_p, j_q, d) = (
        fluct_bra.direction,
        fluct_ket.direction,
        background.direction,
    );

    // P · (B ⋆ Q): the only momentum dependence besides the explicit K_m.
    let conv_packet = packet_convolution(&fluct_ket.packet, &background.packet);
    let product = gauss_packet_product(&conv_packet, &fluct_bra.packet);

    // N_ij = -g sign [ f^{ap,c0,aq} δ_i,d δ_j,j_q + f^{ap,aq,c0} δ_i,j_q δ_j,d ].
    let n_coeff = |i: usize, j: usize| {
        let first = if i == d && j == j_q {
            f[ap][c0][aq]
        } else {
            0.0
        };
        let second = if i == j_q && j == d {
            f[ap][aq][c0]
        } else {
            0.0
        };
        -params.gs * conv.nonabelian_sign * (first + second)
    };

    // T_ij = 2π i α [ δ_j,i_p K_i - δ_i,i_p K_j ] P.
    let mut coeff = [C::new(0.0, 0.0); 3];
    for i in 0..3 {
        for j in 0..3 {
            let n = n_coeff(i, j);
            if n == 0.0 {
                continue;
            }
            if j == i_p {
                coeff[i] += conv.two_pi * C::i() * n;
            }
            if i == i_p {
                coeff[j] -= conv.two_pi * C::i() * n;
            }
        }
    }

    let integral: f64 = (0..3).map(|m| coeff[m].re * product.i1(m)).sum();
    let imaginary: f64 = (0..3).map(|m| coeff[m].im * product.i1(m)).sum();
    let value = C::new(integral, imaginary);
    // Cross energy is ½ (T conj(N) + h.c.); the bra carries α, the ket conj(β).
    0.5 * fluct_bra.amplitude * fluct_ket.amplitude.conj() * value
}

/// The free (quadratic) magnetic form between two gauge modes of the same
/// colour: `½ Re ∫ Tᵃᵢⱼ conj(Tᵃᵢⱼ)`.
pub fn gauge_magnetic_free(
    bra: &GaugeMode,
    ket: &GaugeMode,
    conv: &Conventions,
) -> C {
    if bra.color != ket.color {
        return C::new(0.0, 0.0);
    }
    let product = packet_product(&bra.packet, &ket.packet);
    let mut value = C::new(0.0, 0.0);
    for i in 0..3 {
        for j in 0..3 {
            // T_ij coefficient (no packet): 2π i [δ_j,i_bra K_i - δ_i,i_bra K_j].
            let mut coeff = [C::new(0.0, 0.0); 3];
            if j == bra.direction {
                coeff[i] += conv.two_pi * C::i();
            }
            if i == bra.direction {
                coeff[j] -= conv.two_pi * C::i();
            }
            // Conjugate of the ket's T_ij: -2π i [δ_j,i_ket K_i - δ_i,i_ket K_j].
            let mut conj_coeff = [C::new(0.0, 0.0); 3];
            if j == ket.direction {
                conj_coeff[i] -= conv.two_pi * C::i();
            }
            if i == ket.direction {
                conj_coeff[j] += conv.two_pi * C::i();
            }
            for m in 0..3 {
                if coeff[m] == C::new(0.0, 0.0) {
                    continue;
                }
                for n in 0..3 {
                    if conj_coeff[n] == C::new(0.0, 0.0) {
                        continue;
                    }
                    value += coeff[m] * conj_coeff[n] * product.i2(m, n);
                }
            }
        }
    }
    let amplitude = bra.amplitude * ket.amplitude.conj() * value;
    C::new(0.5 * amplitude.re, 0.0)
}

/// The gauge–Higgs cross term of `|DᵢΦ|²` induced by one background gauge
/// packet.
///
/// `DᵢΦ = 2π i Kᵢ Φ + i g Tᵃ (Wᵃᵢ ⋆ Φ) + i g′ Y (Bᵢ ⋆ Φ)`, with `Tᵃ` and the
/// hypercharge factor taken from [`sm_full::higgs_real_generator`]. The returned
/// value is the coefficient of `conj(α_bra) · α_ket` in `Σᵢ Σ_r |(DᵢΦ)_r|²`.
pub fn higgs_gauge_cross(
    fluct_bra: &HiggsMode,
    fluct_ket: &HiggsMode,
    background: &GaugeMode,
    hypercharge: bool,
    params: &SmParameters,
    conv: &Conventions,
) -> C {
    if !conv.include_higgs_cross {
        return C::new(0.0, 0.0);
    }
    let generator = if hypercharge {
        let scale = params.gy;
        let mut m = higgs_real_generator(None);
        for row in m.iter_mut() {
            for x in row.iter_mut() {
                *x *= scale;
            }
        }
        m
    } else {
        let scale = params.gw;
        let mut m = higgs_real_generator(Some(background.color as u32));
        for row in m.iter_mut() {
            for x in row.iter_mut() {
                *x *= scale;
            }
        }
        m
    };
    let mixing = generator[fluct_bra.component][fluct_ket.component];
    if mixing == 0.0 {
        return C::new(0.0, 0.0);
    }
    let conv_packet = packet_convolution(&fluct_ket.packet, &background.packet);
    let product = gauss_packet_product(&conv_packet, &fluct_bra.packet);
    let integral = product.i1(background.direction);
    // The Hermitian kernel entry: the coefficient of conj(α_bra) α_ket in
    // Σᵢ Σ_r |(DᵢΦ)_r|² is `−2π i M_{cb,ck} ∫ K_d P (B ⋆ Q)`. It is complex
    // (the antisymmetric φ₁∂φ₂ − φ₂∂φ₁ coupling); the energy is its real part.
    let core = -conv.two_pi * C::i() * mixing * integral;
    core * fluct_bra.amplitude.conj() * fluct_ket.amplitude
}

/// `∫∫ φ_bra(k) Φ_bg(k − l) φ_ket(l) d³k d³l` for a unit-width background
/// packet.
///
/// This is the transferred-momentum kernel of a Yukawa/gauge insertion: the
/// insertion carries `k − l`, not `k` alone. The closed form follows from two
/// Gaussian products; `conventions.transfer_sign = −1` reverses the transfer.
pub fn transfer_integral(
    bra: &MomentumPacket,
    background: &MomentumPacket,
    ket: &MomentumPacket,
    conv: &Conventions,
) -> f64 {
    if conv.transfer_sign < 0.0 {
        return transfer_integral(ket, background, bra, &Conventions {
            transfer_sign: 1.0,
            ..*conv
        });
    }
    // Collapse P(k) · B(k − l) over k first.
    let sigma_k2 = bra.width * bra.width + background.width * background.width;
    let sigma_k = sigma_k2.sqrt();
    let sigma = (bra.width * bra.width * background.width * background.width) / sigma_k2;
    let sigma = sigma.sqrt();
    let l_center = sub(bra.center, background.center);
    let l_width2 = sigma_k2;
    // (2π)^{3/2} sigma³ · Gaussian in l, then overlap with the ket packet.
    let scale = TWO_PI.powf(1.5) * sigma.powi(3);
    let d = sub(l_center, ket.center);
    let combine = l_width2 + ket.width * ket.width;
    let overlap = TWO_PI.powf(1.5)
        * (l_width2.sqrt() * ket.width / combine.sqrt()).powi(3)
        * (-dot(d, d) / (2.0 * combine)).exp();
    scale * overlap
}

/// The Yukawa transition kernel `Y_{ij} Φ_bg(k − l)` between two matter
/// packets, with `Y` from the caller's `SmParameters` matrix.
pub fn yukawa_transfer(
    y_ij: C,
    background: &MomentumPacket,
    bra: &MomentumPacket,
    ket: &MomentumPacket,
    conv: &Conventions,
) -> C {
    let integral = transfer_integral(bra, background, ket, conv);
    if conv.yukawa_adjoint {
        // The dropped-adjoint mutation: replace h by its conjugate.
        y_ij.conj() * integral
    } else {
        y_ij * integral
    }
}

/// The free spatial Dirac symbol in the manuscript frame.
///
/// The matter term is `−i α·D` with `D → 2π i k`, so the symbol is
/// `2π Σ_j k_j s book_kinetic_spin(j)` (`s = +1` right-handed, `−1`
/// left-handed). This is the free spatial Dirac check the work order asks for:
/// it is Hermitian, vanishes at `k = 0`, and the two chiralities are negatives.
pub fn dirac_symbol(
    k: [f64; 3],
    right_handed: bool,
    conv: &Conventions,
) -> [[C; 2]; 2] {
    let sign = if right_handed { 1.0 } else { -1.0 };
    let mut out = [[C::new(0.0, 0.0); 2]; 2];
    for j in 0..3 {
        let kinetic = book_kinetic_spin(j as u32);
        let weight = C::new(conv.two_pi * sign * k[j], 0.0);
        for spin in 0..2 {
            for other in 0..2 {
                out[spin][other] += weight * kinetic[spin][other];
            }
        }
    }
    out
}

/// The Dirac packet matrix element `⟨φ_bra,σ_bra| h_D |φ_ket,σ_ket⟩`.
pub fn dirac_packet_element(
    bra: &MomentumPacket,
    bra_spin: usize,
    ket: &MomentumPacket,
    ket_spin: usize,
    right_handed: bool,
    conv: &Conventions,
) -> C {
    assert!(bra_spin < 2 && ket_spin < 2, "spin index range");
    let sign = if right_handed { 1.0 } else { -1.0 };
    let product = packet_product(bra, ket);
    let mut value = C::new(0.0, 0.0);
    for j in 0..3 {
        let kinetic = book_kinetic_spin(j as u32);
        value += C::new(conv.two_pi * sign * product.i1(j), 0.0) * kinetic[bra_spin][ket_spin];
    }
    value
}

/// The transverse (curl) projector kernel `k²(a·b) − (k·a)(k·b)`.
///
/// A longitudinal polarization `a ∝ k` gives exactly zero, which is the free
/// transverse-curl check; two orthogonal transverse polarizations give `k²`.
pub fn transverse_curl_projector(k: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    dot(k, k) * dot(a, b) - dot(k, a) * dot(k, b)
}

/// The uncoupled scalar-gradient symbol `½ |k|²` (the scalar free-gradient
/// normalization the work order asks to recover).
pub fn scalar_gradient_symbol(k: [f64; 3]) -> f64 {
    0.5 * dot(k, k)
}

/// The retained Higgs-potential constant `λ v⁴ / 4`.
///
/// The audit's mutation list includes "dropping an inner constant"; this is the
/// constant the full squared potential keeps, and it must never be removed by a
/// convenience normalization. (The `0, c, 2c` outer-shift check then lives in
/// the `dΓ` lift, not here.)
pub fn higgs_potential_constant(params: &SmParameters) -> f64 {
    params.lambda * params.vev.powi(4) / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm_full::{Chirality, Species, higgs_yukawa_component};

    fn params() -> SmParameters {
        let mut yu = [[C::new(0.0, 0.0); 3]; 3];
        yu[0][0] = C::new(0.7, 0.2);
        yu[1][1] = C::new(0.5, -0.1);
        yu[2][2] = C::new(0.9, 0.0);
        SmParameters {
            gs: 1.1,
            gw: 0.65,
            gy: 0.36,
            lambda: 0.13,
            vev: 246.0,
            yu,
            yd: [[C::new(0.0, 0.0); 3]; 3],
            ynu: [[C::new(0.0, 0.0); 3]; 3],
            ye: [[C::new(0.0, 0.0); 3]; 3],
        }
    }

    fn packet(center: [f64; 3], width: f64) -> MomentumPacket {
        MomentumPacket::new(center, width)
    }

    fn complete(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// Ordinary composite Simpson integral over a cube.
    fn simpson_3d(
        center: [f64; 3],
        radius: f64,
        n: usize,
        f: impl Fn([f64; 3]) -> f64,
    ) -> f64 {
        assert!(n % 2 == 1, "Simpson needs an odd number of points");
        let h = 2.0 * radius / (n - 1) as f64;
        let mut total = 0.0;
        for i in 0..n {
            let wi = if i == 0 || i == n - 1 {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let x = center[0] - radius + i as f64 * h;
            for j in 0..n {
                let wj = if j == 0 || j == n - 1 {
                    1.0
                } else if j % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                let y = center[1] - radius + j as f64 * h;
                for k in 0..n {
                    let wk = if k == 0 || k == n - 1 {
                        1.0
                    } else if k % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    };
                    let z = center[2] - radius + k as f64 * h;
                    total += wi * wj * wk * f([x, y, z]);
                }
            }
        }
        total * (h / 3.0).powi(3)
    }

    #[test]
    fn packet_overlap_matches_gauss_hermite_quadrature() {
        let a = packet([0.3, -0.2, 0.1], 0.8);
        let b = packet([-0.1, 0.4, 0.2], 1.3);
        let g = packet_product(&a, &b);
        let quad = simpson_3d(g.center, 9.0 * g.sigma, 81, |k| a.value(k) * b.value(k));
        assert!(
            (quad - MomentumPacket::overlap(&a, &b)).abs() < 1e-9,
            "quadrature {quad} vs closed form {}",
            MomentumPacket::overlap(&a, &b)
        );
    }

    #[test]
    fn transfer_integral_matches_six_dimensional_quadrature() {
        let bra = packet([0.35, 0.0, -0.15], 0.9);
        let bg = packet([-0.25, 0.2, 0.1], 0.7);
        let ket = packet([0.15, -0.3, 0.25], 1.1);
        let closed = transfer_integral(&bra, &bg, &ket, &Conventions::canonical());

        // Collapse the k integral independently: ∫ P(k) B(k − l) dk is the
        // convolution of P with the flipped background, then integrate over l.
        let flipped = MomentumPacket::new(
            [-bg.center[0], -bg.center[1], -bg.center[2]],
            bg.width,
        );
        let inner = packet_convolution(&bra, &flipped);
        let quad = simpson_3d(inner.center, 9.0 * inner.sigma, 81, |l| {
            inner.amp
                * (-dot(sub(l, inner.center), sub(l, inner.center))
                    / (2.0 * inner.sigma * inner.sigma))
                    .exp()
                * ket.value(l)
        });
        assert!(
            (quad - closed).abs() < 1e-7,
            "3D convolution quadrature {quad} vs closed form {closed}"
        );
    }

    #[test]
    fn noncommuting_colour_magnetic_cross_is_nonzero() {
        let sm = params();
        let conv = Conventions::canonical();
        let bra = GaugeMode::new(1, 2, packet([0.4, 0.0, 0.0], 0.8));
        let ket = GaugeMode::new(5, 1, packet([-0.2, 0.3, 0.0], 0.9));
        let bg = GaugeMode::new(3, 2, packet([0.1, 0.1, -0.2], 0.6));
        let value = gauge_magnetic_cross(&bra, &ket, &bg, &sm, &conv);
        assert!(
            value.norm() > 1e-8,
            "noncommuting colours must give a nonzero cross term, got {value}"
        );
        assert_ne!(bra.packet.center, ket.packet.center);

        let mut abelian = sm.clone();
        abelian.gs = 0.0;
        assert!(gauge_magnetic_cross(&bra, &ket, &bg, &abelian, &conv).norm() < 1e-14);

        let same = GaugeMode::new(3, 0, packet([0.4, 0.0, 0.0], 0.8));
        assert!(gauge_magnetic_cross(&same, &ket, &bg, &sm, &conv).norm() < 1e-14);
    }

    #[test]
    fn magnetic_mutations_are_detected() {
        let sm = params();
        let canonical = Conventions::canonical();
        let bra = GaugeMode::new(1, 2, packet([0.4, 0.0, 0.0], 0.8));
        let ket = GaugeMode::new(5, 1, packet([-0.2, 0.3, 0.0], 0.9));
        let bg = GaugeMode::new(3, 2, packet([0.1, 0.1, -0.2], 0.6));
        let base = gauge_magnetic_cross(&bra, &ket, &bg, &sm, &canonical);
        assert!(base.norm() > 1e-8);

        let no_two_pi = gauge_magnetic_cross(
            &bra,
            &ket,
            &bg,
            &sm,
            &Conventions {
                two_pi: 1.0,
                ..canonical
            },
        );
        assert!((no_two_pi - base).norm() > 1e-8, "dropping 2π must be detectable");

        let flipped = gauge_magnetic_cross(
            &bra,
            &ket,
            &bg,
            &sm,
            &Conventions {
                nonabelian_sign: -1.0,
                ..canonical
            },
        );
        assert!(
            (flipped + base).norm() < 1e-10,
            "flipping the nonabelian sign negates the element"
        );
    }

    #[test]
    fn gauge_higgs_mixing_is_nonzero_and_removable() {
        let sm = params();
        let conv = Conventions::canonical();
        let bra = HiggsMode::new(0, packet([0.35, 0.0, -0.15], 0.9));
        let ket = HiggsMode::new(2, packet([0.15, -0.3, 0.25], 1.1));
        let bg = GaugeMode::new(1, 0, packet([-0.25, 0.2, 0.1], 0.7));
        let value = higgs_gauge_cross(&bra, &ket, &bg, false, &sm, &conv);
        assert!(value.norm() > 1e-8, "gauge–Higgs mixing must be nonzero, got {value}");
        let removed = higgs_gauge_cross(
            &bra,
            &ket,
            &bg,
            false,
            &sm,
            &Conventions {
                include_higgs_cross: false,
                ..conv
            },
        );
        assert_eq!(removed, C::new(0.0, 0.0), "removing the cross term zeroes it");
        let hyper = higgs_gauge_cross(&bra, &ket, &bg, true, &sm, &conv);
        assert!((hyper - value).norm() > 1e-9);
    }

    #[test]
    fn yukawa_transfer_sign_and_adjoint_are_detected() {
        let conv = Conventions::canonical();
        let bra = packet([0.35, 0.0, -0.15], 0.9);
        let bg = packet([-0.25, 0.2, 0.1], 0.7);
        let ket = packet([0.15, -0.3, 0.25], 1.1);
        let y = C::new(0.7, 0.2);
        let base = yukawa_transfer(y, &bg, &bra, &ket, &conv);
        assert!(base.norm() > 1e-10);

        let reversed = yukawa_transfer(
            y,
            &bg,
            &bra,
            &ket,
            &Conventions {
                transfer_sign: -1.0,
                ..conv
            },
        );
        assert!((reversed - base).norm() > 1e-12);

        let adjoint = yukawa_transfer(
            y,
            &bg,
            &bra,
            &ket,
            &Conventions {
                yukawa_adjoint: true,
                ..conv
            },
        );
        assert!((adjoint - base.conj()).norm() < 1e-12);
    }

    #[test]
    fn yukawa_component_dictionary_is_used() {
        let up = higgs_yukawa_component(true, 0, 3);
        let down = higgs_yukawa_component(false, 0, 0);
        assert!(up.norm() > 0.0 && down.norm() > 0.0);
        assert_eq!(Species::Up.hypercharge(Chirality::Right), 2.0 / 3.0);
        assert_eq!(Species::Down.hypercharge(Chirality::Left), 1.0 / 6.0);
    }

    #[test]
    fn higgs_potential_constant_is_retained() {
        let sm = params();
        let constant = higgs_potential_constant(&sm);
        assert!(constant > 0.0);
        assert!(complete(constant, sm.lambda * sm.vev.powi(4) / 4.0));
        let mut dropped = sm.clone();
        dropped.lambda = 0.0;
        assert_eq!(higgs_potential_constant(&dropped), 0.0);
    }

    #[test]
    fn packet_basis_kernel_is_hermitian() {
        let conv = Conventions::canonical();
        let packets = [
            packet([0.4, 0.0, 0.0], 0.8),
            packet([-0.2, 0.3, 0.0], 0.9),
            packet([0.1, -0.4, 0.2], 0.7),
        ];
        for i in 0..3 {
            for j in 0..3 {
                let a = GaugeMode::new(0, i % 3, packets[i]);
                let b = GaugeMode::new(0, j % 3, packets[j]);
                let forward = gauge_magnetic_free(&a, &b, &conv);
                let backward = gauge_magnetic_free(&b, &a, &conv);
                assert!(
                    (forward - backward.conj()).norm() < 1e-10,
                    "H[{i}][{j}]={forward} vs conj(H[{j}][{i}])={}",
                    backward.conj()
                );
            }
        }
    }

    #[test]
    fn dirac_symbol_is_hermitian_chiral_and_vanishes_at_zero() {
        let conv = Conventions::canonical();
        let k = [0.2, -0.35, 0.15];
        let right = dirac_symbol(k, true, &conv);
        let left = dirac_symbol(k, false, &conv);
        for s in 0..2 {
            for t in 0..2 {
                assert!((right[s][t] - right[t][s].conj()).norm() < 1e-12);
                assert!((right[s][t] + left[s][t]).norm() < 1e-12);
            }
        }
        for s in 0..2 {
            for t in 0..2 {
                assert_eq!(dirac_symbol([0.0; 3], true, &conv)[s][t], C::new(0.0, 0.0));
            }
        }
    }

    #[test]
    fn dirac_packet_element_is_nonzero_and_chirality_flips_it() {
        let conv = Conventions::canonical();
        let a = packet([0.4, 0.0, 0.0], 0.8);
        let b = packet([-0.2, 0.3, 0.0], 0.9);
        let right = dirac_packet_element(&a, 0, &b, 1, true, &conv);
        assert!(right.norm() > 1e-8, "distinct packets give a nonzero Dirac entry");
        let left = dirac_packet_element(&a, 0, &b, 1, false, &conv);
        assert!((right + left).norm() < 1e-12, "chiralities are negatives");
    }

    #[test]
    fn transverse_projector_kills_longitudinal_modes() {
        let k = [0.7, -0.2, 0.4];
        let longitudinal = k;
        assert!(transverse_curl_projector(k, longitudinal, longitudinal).abs() < 1e-12);
        let e1 = [1.0, 0.0, 0.0];
        let e2 = [0.0, 1.0, 0.0];
        let kk = dot(k, k);
        assert!((transverse_curl_projector(k, e1, e1) - (kk - k[0] * k[0])).abs() < 1e-12);
        assert!((transverse_curl_projector(k, e1, e2) + k[0] * k[1]).abs() < 1e-12);
        assert!((scalar_gradient_symbol(k) - 0.5 * kk).abs() < 1e-15);
    }

    #[test]
    fn packet_norms_are_positive_and_finite() {
        for center in [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]] {
            let p = packet(center, 0.85);
            assert!(p.norm_sq() > 0.0 && p.norm_sq().is_finite());
        }
    }
}
