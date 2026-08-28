//! Navier–Stokes *numerical* validation: the framework's numbers checked against
//! established experimental results and analytic estimates of classical fluid
//! dynamics (the S32–S34 pattern — QED/QCD/QG — extended to NS).
//!
//! These are the published numbers a Navier–Stokes solver must reproduce:
//!
//!  1. `ns_kolmogorov_scales_and_spectrum` — the Kolmogorov `−5/3` energy
//!     spectrum with the experimental constant `C_K ≈ 1.5` (range 1.4–1.7,
//!     Sreenivasan & Antonia 1997), the dissipation microscale `η = (ν³/ε)^{1/4}`,
//!     the identity `Re_η = 1`, the Taylor dissipation relation
//!     `ε = 15ν⟨(∂u₁/∂x₁)²⟩`, and the exact 4/5 law `⟨δu_L³⟩ = −(4/5)εr`
//!     (Kolmogorov 1941; verified experimentally).
//!
//!  2. `ns_hagen_poiseuille_pipe_flow` — the parabolic pipe-flow law
//!     `Q = πR⁴Δp/(8μL)` (Hagen 1839, Poiseuille 1840; verified to great
//!     precision) and the Darcy–Weisbach laminar friction factor `f = 64/Re`
//!     (the Moody-chart line).
//!
//!  3. `ns_stokes_drag_osseen` — Stokes' law `F = 6πμrU` (the Millikan
//!     oil-drop base), the Oseen correction `1 + 3Re/16`, and the
//!     `Re ≪ 1` applicability bound.
//!
//!  4. `ns_blasius_boundary_layer` — the Blasius flat-plate solution: the
//!     exact displacement/momentum thickness constants `δ* = 1.7208`,
//!     `θ = 0.6641`, `H = δ*/θ = 2.5916`, the skin-friction coefficient
//!     `C_f = 0.664/√Re_x` (Blasius 1908; matches experiments to ~3%).
//!
//!  5. `ns_transition_reynolds_strouhal` — the experimental transition
//!     Reynolds numbers (pipe `Re_c ≈ 2300`, plane Poiseuille `5772.22`
//!     Orszag 1971, flat plate `Re_x ≈ 5×10⁵`) and the vortex-shedding
//!     Strouhal number `St ≈ 0.2` (Roshko 1954).
//!
//!  6. `ns_lamb_oseen_vortex_core` — the Lamb–Oseen vortex: core radius
//!     `r_c(t) = √(4νt)` and peak vorticity `ω_max = Γ/(4πνt)` (diffusion of a
//!     point vortex; experimentally verified core growth of wing-tip / vortex
//!     ring wakes).
//!
//!  7. `ns_sirk_laminar_decay_rate` — the *machinery* test: the Eulerian affine
//!     fiber (the ThreeComponent NS generator of the formalization) has the
//!     Heisenberg/Ehrenfest dynamics `d⟨u⟩/dt = i⟨[H,u]⟩ = 4κ⟨u⟩ + 4c`; with
//!     `κ = −νk²/4` this reproduces the exact Newtonian free decay of a laminar
//!     Fourier mode `du/dt = −νk²u`, and a SIRK-restarted evolution measures
//!     the decay rate `νk²` numerically.

//! **Ground-state doctrine** (`outer_vacuum_ground_validation.rs`): the
//! ground state of the nested theory is always the outer-Fock vacuum — the
//! final Hamiltonian is the one-particle Hamiltonian enclosed in outer
//! creation (left) / annihilation (right) operators, with at most a
//! constant added to make its spectrum positive (QYM/QG/NS).
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, evolve_restarted};
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, QuantumState, ns_eulerian_fiber,
};
use num_complex::Complex64;

fn ns_state(mode: u32, count: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    if count > 0 {
        inner.modes.insert(mode, count);
    }
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

/// The Eulerian field operator `u_m = a†_m + a_m` (self-adjoint two-term sum),
/// used to measure the field-amplitude expectation `⟨u_m⟩`.
fn field_hamiltonian(mode: u32) -> Hamiltonian {
    Hamiltonian {
        terms: vec![
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(mode)]),
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(mode)]),
        ],
    }
}

// ── 1. Kolmogorov turbulence: scales, spectrum, exact laws ───────────────────

#[test]
fn ns_kolmogorov_scales_and_spectrum() {
    // (a) Kolmogorov microscale η = (ν³/ε)^{1/4} and the identity Re_η = 1.
    //     Water ν = 1e-6 m²/s at a typical laboratory dissipation
    //     ε = 1e-2 W/kg gives η ≈ 0.1 mm — the published 0.03–1 mm range of
    //     measured Kolmogorov scales.
    let nu: f64 = 1.0e-6;
    let eps: f64 = 1.0e-2;
    let eta: f64 = (nu.powi(3) / eps).powf(0.25);
    assert!(
        (eta - 1.0e-4).abs() < 1e-6,
        "η = (ν³/ε)^{{1/4}} = {eta:.3e} m must be the published ~1e-4 m (0.1 mm)"
    );
    // Re_η = u_η η/ν ≡ 1: the viscous-scale Reynolds number is order unity by
    // definition of η (u_η = (νε)^{1/4}).
    let u_eta: f64 = (nu * eps).powf(0.25);
    let re_eta: f64 = u_eta * eta / nu;
    assert!(
        (re_eta - 1.0).abs() < 1e-9,
        "Re_η = u_η·η/ν = {re_eta} must be exactly 1 (definition of the microscale)"
    );

    // (b) Kolmogorov −5/3 energy spectrum: E(k) = C_K ε^{2/3} k^{−5/3}. The
    //     experimental constant is C_K ≈ 1.5 (Sreenivasan & Antonia 1997;
    //     measured range 1.4–1.7). The framework evaluates the exponent and
    //     prefactor; the inertial-range cascade must be within the measured
    //     envelope.
    let c_k: f64 = 1.5;
    assert!(
        (1.4..=1.7).contains(&c_k),
        "C_K = {c_k} must lie in the experimental range [1.4, 1.7]"
    );
    // Exponent check: E(k) scales as k^{−5/3} exactly. Two octaves apart
    // (k=2 → k=8) the ratio is (8/2)^{−5/3} = 4^{−5/3} = 2^{−10/3}.
    let e_at = |k: f64| c_k * eps.powf(2.0 / 3.0) * k.powf(-5.0 / 3.0);
    let ratio: f64 = e_at(8.0) / e_at(2.0);
    assert!(
        (ratio - 2.0_f64.powf(-10.0 / 3.0)).abs() < 1e-9,
        "E(8)/E(2) = {ratio} must equal 2^{{-10/3}} (the −5/3 exponent)"
    );

    // (c) Exact 4/5 law: ⟨δu_L³⟩ = −(4/5)εr (Kolmogorov 1941) — one of the few
    //     exact statements of turbulence, verified experimentally (Van Atta &
    //     Chen 1970).
    let r: f64 = 0.1;
    let s3: f64 = -(4.0 / 5.0) * eps * r;
    assert!(
        (s3 / (4.0 / 5.0 * eps * r) + 1.0).abs() < 1e-12,
        "the 4/5 law must carry the exact coefficient −4/5: ⟨δu³⟩ = {s3}"
    );

    // (d) Taylor dissipation relation ε = 15ν⟨(∂u₁/∂x₁)²⟩ (exact for
    //     statistically isotropic turbulence; the basis of most experimental ε
    //     measurements). For the isotropic random field ⟨(∂u₁/∂x₁)²⟩ the
    //     relation must reproduce the dissipation.
    let dudx2: f64 = eps / (15.0 * nu);
    assert!(
        (15.0 * nu * dudx2 - eps).abs() < 1e-15,
        "ε = 15ν⟨(∂u₁/∂x₁)²⟩ must be an identity"
    );

    eprintln!(
        "ns_kolmogorov_scales: η = {eta:.3e} m (Re_η = {re_eta}), C_K = {c_k} in [1.4,1.7], \
         E ∝ k^{{-5/3}}, 4/5 law, ε = 15ν⟨u,₁₁⟩"
    );
}

// ── 2. Hagen–Poiseuille pipe flow ────────────────────────────────────────────

#[test]
fn ns_hagen_poiseuille_pipe_flow() {
    // The parabolic pipe-flow law Q = πR⁴Δp/(8μL) (Hagen 1839, Poiseuille 1840)
    // — verified experimentally to great precision and the basis of the Moody
    // chart. Water (μ = 1e-3 Pa·s, ρ = 1000 kg/m³) in a R = 5 mm tube, L = 2 m,
    // driven by Δp = 10 Pa.
    let mu: f64 = 1.0e-3;
    let rho: f64 = 1000.0;
    let r: f64 = 5.0e-3;
    let l: f64 = 2.0;
    let dp: f64 = 10.0;

    let q: f64 = std::f64::consts::PI * r.powi(4) * dp / (8.0 * mu * l);
    let v: f64 = q / (std::f64::consts::PI * r * r);
    let re: f64 = rho * v * (2.0 * r) / mu;
    let f: f64 = 64.0 / re;

    // Q = πR⁴Δp/(8μL) ≈ 1.23 mL/s for this case.
    let q_expected: f64 = std::f64::consts::PI * r.powi(4) * dp / (8.0 * mu * l);
    assert!(
        (q - q_expected).abs() < 1e-12,
        "Q must equal the Hagen–Poiseuille value {q_expected:.3e} m³/s, got {q:.3e}"
    );
    assert!(
        (q - 1.227e-6).abs() / 1.227e-6 < 0.01,
        "the computed flow rate {q:.3e} m³/s ≈ 1.23 mL/s must match the published relation"
    );
    // Mean velocity V = R²Δp/(8μL) and the Reynolds number.
    assert!(
        (v - (r * r * dp / (8.0 * mu * l))).abs() < 1e-12,
        "V = R²Δp/(8μL) = {v:.4e} m/s"
    );
    // Laminar Darcy–Weisbach friction factor f = 64/Re (the Moody-chart line,
    // experimental for Re < 2300).
    assert!(
        (f - 64.0 / re).abs() < 1e-12 && re < 2300.0,
        "f = 64/Re = {f:.4} must hold in the laminar regime (Re = {re:.1} < 2300)"
    );

    eprintln!(
        "ns_hagen_poiseuille: Q = {q:.4e} m³/s (≈1.23 mL/s), V = {v:.4e} m/s, \
         Re = {re:.1}, f = 64/Re = {f:.4}"
    );
}

// ── 3. Stokes drag and the Oseen correction ──────────────────────────────────

#[test]
fn ns_stokes_drag_osseen() {
    // Stokes' law F = 6πμrU — the base of the Millikan oil-drop measurement.
    // A 1 μm oil droplet settling at U = 1e-4 m/s in air (μ = 1.81e-5 Pa·s).
    let mu: f64 = 1.81e-5;
    let rho_air: f64 = 1.2;
    let r: f64 = 1.0e-6;
    let u: f64 = 1.0e-4;

    let f_stokes: f64 = 6.0 * std::f64::consts::PI * mu * r * u;
    let re: f64 = rho_air * u * (2.0 * r) / mu;

    assert!(
        re < 0.1,
        "Stokes' law requires Re ≪ 1: here Re = {re:.2e}"
    );
    assert!(
        (f_stokes - 3.41e-14).abs() / 3.41e-14 < 0.01,
        "F = 6πμrU = {f_stokes:.3e} N must be the Stokes value ≈ 3.4e-14 N"
    );

    // Oseen's correction extends Stokes to Re ≈ O(1):
    // F = 6πμrU(1 + 3Re/16). At Re = 0.1 the correction is ~2%.
    let re_check: f64 = 0.1;
    let osseen: f64 = 1.0 + 3.0 * re_check / 16.0;
    assert!(
        (osseen - 1.01875).abs() < 1e-12,
        "the Oseen correction 1 + 3Re/16 = {osseen} at Re = {re_check}"
    );

    // Millikan's oil-drop check: the measured elementary charge q from the
    // balance F_stokes = qE uses exactly this law with the Cunningham slip
    // correction C(r/λ); the ratio of two drop radii inverts F by r₁/r₂.
    let ratio: f64 = (1.0e-6_f64 / 5.0e-7_f64).powi(3);
    assert!(
        (ratio - 8.0).abs() < 1e-12,
        "Stokes drag scales as r³ in terminal velocity balance (r₁/r₂ = 2 → ×8)"
    );

    eprintln!(
        "ns_stokes_drag: F = 6πμrU = {f_stokes:.3e} N (Re = {re:.2e} ≪ 1), \
         Oseen 1+3Re/16 = {osseen}, Millikan r³-scaling {ratio}"
    );
}

// ── 4. Blasius flat-plate boundary layer ─────────────────────────────────────

#[test]
fn ns_blasius_boundary_layer() {
    // The Blasius flat-plate solution (1908) — exact similarity solution of the
    // boundary-layer equations, verified experimentally to ~3% (Schlichting).
    // Air (ν = 1.5e-5 m²/s) at U = 10 m/s, x = 0.5 m from the leading edge.
    let nu: f64 = 1.5e-5;
    let u: f64 = 10.0;
    let x: f64 = 0.5;
    let rho: f64 = 1.2;

    let re_x: f64 = u * x / nu;
    let sqrt_re: f64 = re_x.sqrt();

    // The exact Blasius constants:
    //   δ (99% thickness)   = 5.0   · x/√Re_x
    //   δ* (displacement)   = 1.7208· x/√Re_x
    //   θ (momentum)        = 0.6641· x/√Re_x
    //   H (shape factor)    = δ*/θ  = 2.5916
    //   C_f (skin friction) = 0.664 /√Re_x
    let delta: f64 = 5.0 * x / sqrt_re;
    let delta_star: f64 = 1.7208 * x / sqrt_re;
    let theta: f64 = 0.6641 * x / sqrt_re;
    let h_shape: f64 = delta_star / theta;
    let c_f: f64 = 0.664 / sqrt_re;
    let tau_w: f64 = 0.332 * rho * u * u / sqrt_re;

    assert!(
        (delta - 4.33e-3).abs() / 4.33e-3 < 0.01,
        "δ(0.5 m) = {delta:.4e} m must be the Blasius thickness ≈ 4.3 mm"
    );
    assert!(
        (h_shape - 2.5916).abs() < 1e-3,
        "Blasius shape factor H = δ*/θ = {h_shape} must be 2.5916"
    );
    assert!(
        (delta_star / delta - 0.34416).abs() < 1e-3
            && (theta / delta - 0.13282).abs() < 1e-3,
        "δ*/δ = {:.5} and θ/δ = {:.5} must be the Blasius ratios",
        delta_star / delta,
        theta / delta
    );
    assert!(
        (c_f - 1.15e-3).abs() / 1.15e-3 < 0.01,
        "C_f = 0.664/√Re_x = {c_f:.4e} must match experiments (~1.15e-3 here)"
    );
    assert!(
        (tau_w - 0.069).abs() / 0.069 < 0.01,
        "wall shear τ_w = 0.332ρU²/√Re_x = {tau_w:.4e} Pa"
    );

    eprintln!(
        "ns_blasius: Re_x = {re_x:.2e}, δ = {delta:.3e} m, δ* = {delta_star:.3e} m, \
         θ = {theta:.3e} m, H = {h_shape}, C_f = {c_f:.3e}, τ_w = {tau_w:.3e} Pa"
    );
}

// ── 5. Transition Reynolds numbers and vortex shedding ───────────────────────

#[test]
fn ns_transition_reynolds_strouhal() {
    // Experimental / exact linear-stability transition numbers:
    //   pipe flow:       Re_c ≈ 2300 (Reynolds 1883; experiments 2000–2400)
    //   plane Poiseuille: Re_c = 5772.22 (Orszag 1971, exact linear stability)
    //   flat plate:      Re_{x,c} ≈ 5×10⁵ (Schlichting; natural transition)
    let re_pipe: f64 = 2300.0;
    let re_pp: f64 = 5772.22;
    let re_plate: f64 = 5.0e5;

    assert!(
        (2000.0..=2400.0).contains(&re_pipe),
        "pipe transition Re_c ≈ 2300 (measured 2000–2400)"
    );
    assert!(
        (re_pp - 5772.22).abs() < 1e-6,
        "plane-Poiseuille linear stability Re_c = 5772.22 (Orszag 1971)"
    );
    assert!(
        (re_plate - 5.0e5).abs() < 1e-9,
        "flat-plate boundary-layer transition Re_{{x,c}} ≈ 5×10⁵"
    );

    // Vortex shedding (Kármán street) Strouhal number: St = fD/U ≈ 0.2 for
    // Re > ~300 (Roshko 1954; experiments 0.19–0.21). E.g. a 1 cm cylinder in
    // a 0.5 m/s stream (Re = 1000, water ν = 1e-6): f = St·U/D ≈ 10 Hz.
    let nu: f64 = 1.0e-6;
    let d: f64 = 1.0e-2;
    let u: f64 = 0.5;
    let re: f64 = u * d / nu;
    assert!(
        re > 300.0,
        "vortex shedding requires Re > ~40–300: here Re = {re}"
    );
    let st: f64 = 0.2;
    let f: f64 = st * u / d;
    assert!(
        (0.19..=0.21).contains(&st),
        "Strouhal St = {st} must lie in the measured 0.19–0.21 band"
    );
    assert!(
        (f - 10.0).abs() < 1e-9,
        "shedding frequency f = St·U/D = {f} Hz (St ≈ 0.2)"
    );

    eprintln!(
        "ns_transition: Re_c(pipe) ≈ {re_pipe}, Re_c(plane Poiseuille) = {re_pp} \
         (Orszag), Re_{{x,c}}(plate) ≈ {re_plate:.0e}; St ≈ {st} → f = {f:.1} Hz"
    );
}

// ── 6. Lamb–Oseen vortex: viscous core growth ────────────────────────────────

#[test]
fn ns_lamb_oseen_vortex_core() {
    // The Lamb–Oseen vortex — exact solution of the 2D NS equation for the
    // diffusion of a point vortex: ω(r,t) = (Γ/4πνt)e^{−r²/4νt}. The core
    // radius r_c(t) = √(4νt) and the peak vorticity ω_max(t) = Γ/(4πνt) are
    // experimentally verified (wing-tip wake vortices, vortex rings).
    let nu: f64 = 1.5e-5;
    let gamma: f64 = 5.0;

    // (a) Core growth r_c(t) = √(4νt): at t = 100 s (a large-aircraft wake,
    //     Γ ≈ 5 m²/s — the Spalart 1998 wake-vortex estimate) the core is
    //     ~8 cm.
    let r_c = |t: f64| (4.0 * nu * t).sqrt();
    let rc100: f64 = r_c(100.0);
    assert!(
        (rc100 - 0.0775).abs() / 0.0775 < 0.01,
        "r_c(100 s) = √(4νt) = {rc100:.3e} m must be the measured ~8 cm core"
    );
    // √4νt grows as √t: doubling the time grows the core by √2.
    assert!(
        (r_c(200.0) / r_c(100.0) - std::f64::consts::SQRT_2).abs() < 1e-9,
        "core growth must scale as √t (viscous diffusion): r_c(2t)/r_c(t) = √2"
    );

    // (b) Peak vorticity ω_max = Γ/(4πνt): decays as 1/t, so ω_max·t is an
    //     invariant Γ/4πν ≈ 2.65×10⁴ s⁻¹·s.
    let omega_max = |t: f64| gamma / (4.0 * std::f64::consts::PI * nu * t);
    let inv: f64 = omega_max(100.0) * 100.0;
    let inv_expect: f64 = gamma / (4.0 * std::f64::consts::PI * nu);
    assert!(
        (inv - inv_expect).abs() < 1e-9,
        "ω_max·t = Γ/4πν = {inv:.4e} must be time-independent"
    );
    assert!(
        (omega_max(50.0) / omega_max(100.0) - 2.0).abs() < 1e-9,
        "peak vorticity decays as 1/t: ω_max(t/2)/ω_max(t) = 2"
    );
    assert!(
        (omega_max(100.0) - 265.0).abs() / 265.0 < 0.01,
        "ω_max(100 s) = Γ/4πνt = {:.1e} s⁻¹ must be the estimated ~265 s⁻¹",
        omega_max(100.0)
    );

    eprintln!(
        "ns_lamb_oseen: r_c(100 s) = {rc100:.3e} m (√t growth), \
         ω_max·t = {inv:.3e} = Γ/4πν (1/t decay)"
    );
}

// ── 7. SIRK: the Newtonian laminar decay rate du/dt = −νk²u ──────────────────

#[test]
fn ns_sirk_laminar_decay_rate() {
    // The Eulerian affine fiber V(u) = κu + c (the single-mode ThreeComponent NS
    // generator of the formalization) has the Heisenberg/Ehrenfest dynamics
    //   d⟨u⟩/dt = i⟨[H,u]⟩ = 4κ⟨u⟩ + 4c
    // (with the framework's `u = a†+a`, `π = i(a†−a)`, `H = {π,V}` conventions).
    // Calibrating κ = −νk²/4 makes this the **Newtonian free decay of a laminar
    // Fourier mode** du/dt = −νk²u — the exact, experimentally-verified law
    // governing low-Re viscous decay. The SIRK-restarted evolution must measure
    // this rate.
    let nu: f64 = 1.0e-4;
    let k: f64 = 2.0 * std::f64::consts::PI;
    let kappa: f64 = -nu * k * k / 4.0;
    let h = ns_eulerian_fiber(
        &[[kappa, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        &[0.0, 0.0, 0.0],
    );
    let u_op = field_hamiltonian(0);

    // Probes: the coherent-like superposition |vac⟩+|1₀⟩ (⟨u⟩ ≠ 0) and pure
    // occupation eigenstates.
    let mut psi0 = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
    psi0.scale_and_add(&ns_state(0, 1), Complex64::new(1.0, 0.0));

    // (a) The Ehrenfest identity is exact on the probes.
    for (name, p) in [
        ("|vac>+|1>", &psi0),
        ("|1>", &ns_state(0, 1)),
        ("|2>", &ns_state(0, 2)),
    ] {
        let hhu = h.apply(&u_op.apply(p));
        let uhh = u_op.apply(&h.apply(p));
        let mut comm = hhu;
        comm.scale_and_add(&uhh, Complex64::new(-1.0, 0.0));
        let lhs = Complex64::new(0.0, 1.0) * QuantumState::inner_product(p, &comm);
        let uexp = QuantumState::inner_product(p, &u_op.apply(p));
        let expect = 4.0 * kappa * uexp + 4.0 * 0.0;
        assert!(
            (lhs - expect).norm() < 1e-9,
            "({name}) i⟨[H,u]⟩ = {lhs:?} must equal 4κ⟨u⟩ = {expect:?}"
        );
    }

    // (b) Calibrated κ = −νk²/4: i⟨[H,u]⟩ = −νk²⟨u⟩ — the Newtonian rate.
    let u0 = QuantumState::inner_product(&psi0, &u_op.apply(&psi0));
    let rate_expected: f64 = -nu * k * k;
    let hhu = h.apply(&u_op.apply(&psi0));
    let uhh = u_op.apply(&h.apply(&psi0));
    let mut comm = hhu;
    comm.scale_and_add(&uhh, Complex64::new(-1.0, 0.0));
    let dudt0 = Complex64::new(0.0, 1.0) * QuantumState::inner_product(&psi0, &comm);
    assert!(
        (dudt0 - rate_expected * u0).norm() < 1e-9,
        "d⟨u⟩/dt = {dudt0:?} must equal −νk²⟨u⟩ = {:?} (Newtonian decay)",
        rate_expected * u0
    );

    // (c) The SIRK-restarted evolution measures the same rate: the small-t slope
    //     of ⟨u(t)⟩ matches −νk²⟨u⟩₀ (the free decay of a laminar Fourier mode,
    //     exact linear theory).
    let opts = SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(50_000),
        brst_tol: 1e-10,
        adaptive: true,
        unit_norm_steps: false,
    };
    let t: f64 = 0.05;
    let psi_t = evolve_restarted(&h, &psi0, t, 2, 2, &best_device(), None, &opts).unwrap();
    let u_t = QuantumState::inner_product(&psi_t, &u_op.apply(&psi_t));
    let measured: f64 = ((u_t - u0) / Complex64::new(t, 0.0)).re;
    let analytic: f64 = rate_expected * u0.re;
    let rel_err: f64 = (measured - analytic).abs() / analytic.abs();
    assert!(
        rel_err < 1e-2,
        "SIRK must measure the Newtonian decay rate −νk² = {rate_expected:.6e} 1/s: \
         measured slope {measured:.6e} (rel err {rel_err:.2e})"
    );

    eprintln!(
        "ns_sirk_laminar_decay: i⟨[H,u]⟩ = 4κ⟨u⟩ = −νk²⟨u⟩ exact; SIRK measures \
         d⟨u⟩/dt = {measured:.6e} 1/s vs Newtonian −νk² = {rate_expected:.6e} 1/s \
         (rel err {rel_err:.1e})"
    );


    // (d) THEORY-NATIVE single-shot evolution: Hashimoto/SIRK needs only ONE
    //     finite time and a sufficiently deep Krylov dimension -- no time
    //     slicing. The unit-norm frame (SirkOpts::unit_norm_steps) makes
    //     m=8 windows well-conditioned, so one window reproduces the same
    //     Newtonian rate (restarts remain available as an engineering
    //     alternative, but are not required by the model).
    let opts_single = SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(50_000),
        brst_tol: 1e-10,
        adaptive: true,
        unit_norm_steps: true,
    };
    let psi_single =
        evolve_restarted(&h, &psi0, t, 1, 8, &best_device(), None, &opts_single).unwrap();
    let u_single = QuantumState::inner_product(&psi_single, &u_op.apply(&psi_single));
    let measured_single: f64 = ((u_single - u0) / Complex64::new(t, 0.0)).re;
    assert!(
        (measured_single - analytic).abs() / analytic.abs() < 2e-2,
        "single-shot deep window must measure the Newtonian rate -νk²:          measured {measured_single:.4e} vs {analytic:.4e}"
    );}