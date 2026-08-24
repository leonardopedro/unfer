//! Theorem 4.1 (Hashimoto–Nodera 2019, JJIAM) ERROR BANDS for
//! program-sector models. The measured SIRK state error is checked against
//! the paper's a-priori envelope (Eq. 12):
//!
//!   ‖φ₀(A)v − SIRK_m(v)‖ ≤ 2C‖v‖ e^{-hm} E_m ,   C ∈ [2, 11.08],
//!
//! with the paper's OWN ingredients:
//!   • shift ladder γ_j = N − h j > 0 fed to the solver — mapped into the
//!     inverse-free recurrence as z_j = i γ_j / t for A = −i H t, so the
//!     Krylov span equals the theorem's Q_m exactly;
//!   • the literal SIRK denominator q(z) = Π_{j=1..m} (1 + h j z);
//!   • E_m = min_{p ∈ P_m} ‖f_{0,N} − p/q‖_{∞,Σ}, computed by Lawson's
//!     iteratively-reweighted minimax on a discretized numerical-range hull
//!     Σ of the resolvents X_j = (γ_j I − A)^{-1};
//!   • the C-interval [2, 11.08] giving band edges (lo, hi).
//!
//! Models are bounded program Hamiltonians with CLOSED-FORM evolutions so
//! the true relative state error is measurable:
//!   QED free photon field · QED Jaynes–Cummings (one-excitation Rabi)
//!   QG scalaron band · QG free graviton.
//!
//! Each test prints its band table [m, err, lo(C=2), hi(C=11.08), E_m,
//! e^{-hm}] and asserts: measurement ≤ upper edge; overall error decay
//! across m; geometric tightening of the band.

mod hashimoto_support;

use fock_sirk::device::best_device;
use fock_sirk::{solve_forward_sirk_with_opts, SirkOpts};
use hashimoto_support::{sirk_paper_shifts, BandParams};
use nested_fock_algebra::{
    qed_free_photon, qed_jaynes_cummings, qg_free_graviton,
    qg_starobinsky_scalaron_field, InnerBosonicState, Operator, QuantumState,
};
use num_complex::Complex64;

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 0.0,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: true,
    }
}

fn empty_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn boson_universe(mode: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(mode, 1);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

fn fermion_universe(mode: u32) -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterFermionCreate(
        nested_fock_algebra::InnerFermionicState {
            modes: std::collections::BTreeSet::from([mode]),
        },
    ))
}

/// Solve at time t with the PAPER shift ladder mapped as z_j = i γ_j / t.
fn solve_paper(
    ham: &nested_fock_algebra::Hamiltonian,
    psi: &QuantumState,
    bp: &BandParams,
    m: usize,
) -> QuantumState {
    let shifts: Vec<Complex64> = sirk_paper_shifts(m, bp.big_n, bp.h)
        .iter()
        .map(|g| Complex64::new(0.0, g.re / bp.t))
        .collect();
    let res =
        solve_forward_sirk_with_opts(ham, psi, &shifts, &best_device(), None, &opts())
            .expect("paper-shift SIRK solve");
    let coeffs = res.time_evolve(bp.t);
    res.reconstruct(&coeffs)
}

struct BandRow {
    m: usize,
    err: f64,
    lo: f64,
    hi: f64,
    e_m: f64,
    exp_factor: f64,
}

/// Shared driver for DIAGONAL models: ψ₀ = Σ cᵢ |sector_i⟩ (+ vacuum),
/// exact evolution = per-sector phases.
fn run_diagonal_banded(
    label: &str,
    ham: &nested_fock_algebra::Hamiltonian,
    psi0: &QuantumState,
    omegas: &[f64],
    coeffs: &[Complex64],
    bp: &BandParams,
    depths: &[usize],
    lam_abs_max: f64,
) {
    // Exact evolved state: vacuum component stays put; each sector rotates.
    let mut psi_exact = empty_vacuum();
    for (i, (&w, &c)) in omegas.iter().zip(coeffs.iter()).enumerate() {
        let phase = Complex64::new(0.0, -w * bp.t).exp() * c;
        psi_exact.scale_and_add(&boson_universe(i as u32), phase);
    }
    let v_norm = psi0.norm();

    println!("— {label}: N={} h={} t={}", bp.big_n, bp.h, bp.t);
    println!("  m   err         lo(C=2)      hi(C=11.08)  E_m        e^(-hm)");
    let mut rows: Vec<BandRow> = Vec::new();
    for &m in depths {
        let out = solve_paper(ham, psi0, bp, m);
        // ZERO-BASED difference: adding `empty_vacuum()` first would inject
        // a spurious +1 vacuum amplitude that never cancels.
        let mut diff = QuantumState::zero();
        diff.scale_and_add(&out, Complex64::new(1.0, 0.0));
        diff.scale_and_add(&psi_exact, Complex64::new(-1.0, 0.0));
        let err = QuantumState::inner_product(&diff, &diff).re.sqrt() / v_norm;
        let b = bp.band(lam_abs_max, m, v_norm, 0, 240, 40);
        println!(
            "  {:<3} {:.3e}   {:.3e}    {:.3e}    {:.3e}  {:.3e}",
            m, err, b.lo, b.hi, b.e_m, b.exp_factor
        );
        rows.push(BandRow { m, err, lo: b.lo, hi: b.hi, e_m: b.e_m, exp_factor: b.exp_factor });
    }

    // Assertions.
    for r in &rows {
        assert!(
            r.err <= r.hi * (1.0 + 1e-9),
            "{label}: measured {:.3e} exceeds Theorem 4.1 upper edge {:.3e} (m={})",
            r.err,
            r.hi,
            r.m
        );
    }
    assert!(
        rows.last().unwrap().err < rows.first().unwrap().err,
        "{label}: error must decay across depths"
    );
    // Measured exponential slope vs the theorem's h: err(m) ≈ K e^{-c m}.
    // The paper's qualitative claim is EXPONENTIAL decay (vs polynomial for
    // Arnoldi/SIA/RK); we require the measured slope to reach a healthy
    // fraction of the shift spacing.
    let n_pts = rows.len();
    let floor = 1e-13_f64;
    let e_first = rows[0].err.max(floor);
    let e_last = rows[n_pts - 1].err.max(floor);
    let slope = (e_first.ln() - e_last.ln())
        / (rows[n_pts - 1].m - rows[0].m) as f64;
    println!(
        "  measured decay slope c = {slope:.3} (theorem h = {:.3})",
        bp.h
    );
    assert!(
        e_last < 1e-10 || slope > 0.15,
        "{label}: decay must be at least mildly exponential or machine-converged          (slope {slope:.3}, last {e_last:.3e})"
    );
    for w in rows.windows(2) {
        assert!(w[1].hi < w[0].hi, "{label}: band must tighten with m");
    }
}

// ─────────────────────────────── the four models ──────────────────────────

#[test]
fn band_qed_free_photon_multimode() {
    let omegas = [0.9_f64, 1.4, 2.1, 2.8];
    let ham = qed_free_photon(&omegas);
    let coeffs = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.8, 0.2),
        Complex64::new(-0.5, 0.3),
        Complex64::new(0.6, -0.4),
    ];
    let mut psi0 = empty_vacuum();
    for (i, _) in omegas.iter().enumerate() {
        psi0.scale_and_add(&boson_universe(i as u32), coeffs[i]);
    }
    run_diagonal_banded(
        "QED free photon (4 modes)",
        &ham,
        &psi0,
        &omegas,
        &coeffs,
        
        &BandParams { big_n: 8.0, h: 0.5, t: 0.8 },
        
        &[4, 6, 8],
        omegas[3],
    );
}

#[test]
fn band_qed_jaynes_cummings_rabi() {
    let (wc, wa, g) = (1.0_f64, 1.35_f64, 0.18_f64);
    let ham = qed_jaynes_cummings(wc, wa, g);
    let t = 0.8_f64;

    // Exact one-excitation Rabi solution.
    let delta = wc - wa;
    let omega_r = 2.0 * (g * g + delta * delta / 4.0).sqrt();
    let common = Complex64::new(0.0, -(wc + wa) * t / 2.0).exp();
    let half = omega_r * t / 2.0;
    let amp_pg = common * Complex64::new(half.cos(), -delta / omega_r * half.sin());
    let amp_ae = common * Complex64::new(0.0, -2.0 * g / omega_r * half.sin());
    // TOPOLOGY: the JC builder lives in ONE universe holding BOTH the
    // photon (inner boson mode 0) and the atom (inner fermion mode 1).
    let excitation_base = empty_vacuum().apply(&Operator::InnerBosonCreate(0));
    let ket_photon = excitation_base.clone(); // |1_ph, g>
    let ket_atom = excitation_base
        .apply(&Operator::InnerBosonAnnihilate(0))
        .apply(&Operator::InnerFermionCreate(1)); // |0_ph, e>
    let psi_exact = {
        let mut acc = QuantumState::zero();
        acc.scale_and_add(&ket_photon, amp_pg);
        acc.scale_and_add(&ket_atom, amp_ae);
        acc
    };

    // Solver start: one cavity photon (atomic ground).
    let psi0 = ket_photon.clone();

    let _ = &psi_exact;
    let bp = BandParams { big_n: 8.0, h: 0.5, t };
    let psi0_ref = &psi0;
    println!("— QED Jaynes–Cummings Rabi: N=50 h=1 t={t}");
    println!("  m   err         lo(C=2)      hi(C=11.08)  E_m        e^(-hm)");
    let mut prev_err: Option<f64> = None;
    let mut prev_hi: Option<f64> = None;
    for m in [6_usize, 9, 12] {
        let out = solve_paper(&ham, psi0_ref, &bp, m);
        // ZERO-BASED full state difference (see run_diagonal_banded note).
        let mut diff = QuantumState::zero();
        diff.scale_and_add(&out, Complex64::new(1.0, 0.0));
        diff.scale_and_add(&psi_exact, Complex64::new(-1.0, 0.0));
        let err = diff.norm() / psi0_ref.norm();
        // Spectral extent bound of the one-excitation block.
        let lam_abs = (wa + g).abs().max(wc + g).max((wc - wa).abs() / 2.0 + g);
        let b = bp.band(lam_abs.max(wc), m, psi0_ref.norm(), 0, 240, 40);
        println!(
            "  {m:<3} {err:.3e}   {:.3e}    {:.3e}    {:.3e}  {:.3e}",
            b.lo, b.hi, b.e_m, b.exp_factor
        );
        assert!(err <= b.hi * (1.0 + 1e-9), "JC err {err:.3e} > band {:.3e} (m={m})", b.hi);
        if let Some(p) = prev_err {
            assert!(err < p * 1.05 || err < 1e-7, "JC decay {p:.3e} → {err:.3e}");
        }
        if let Some(ph) = prev_hi {
            assert!(b.hi < ph, "JC band must tighten");
        }
        prev_err = Some(err);
        prev_hi = Some(b.hi);
    }
}

#[test]
fn band_qg_scalaron_band_two_k() {
    let mass = 0.8_f64;
    let ks = [0.9_f64, 1.6];
    let omegas: Vec<f64> = ks.iter().map(|k| (k * k + mass * mass).sqrt()).collect();
    let ham = qg_starobinsky_scalaron_field(&ks, mass);
    let coeffs = [Complex64::new(1.0, 0.0), Complex64::new(0.7, 0.4)];
    let mut psi0 = empty_vacuum();
    for (i, _) in ks.iter().enumerate() {
        psi0.scale_and_add(&boson_universe(i as u32), coeffs[i]);
    }
    run_diagonal_banded(
        "QG scalaron band (k=0.9, 1.6)",
        &ham,
        &psi0,
        &omegas,
        &coeffs,
        
        &BandParams { big_n: 8.0, h: 0.5, t: 0.8 },
        
        &[4, 6, 8],
        omegas[1],
    );
}

#[test]
fn band_qg_graviton_three_k() {
    let ks = [0.5_f64, 1.1, 1.9];
    let ham = qg_free_graviton(&ks); // ω = |k| (ħ = c = 1)
    let omegas = ks.to_vec();
    let coeffs = [
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.6, 0.5),
        Complex64::new(0.4, 0.9),
    ];
    let mut psi0 = empty_vacuum();
    for (i, _) in ks.iter().enumerate() {
        psi0.scale_and_add(&boson_universe(i as u32), coeffs[i]);
    }
    run_diagonal_banded(
        "QG free graviton (3 modes)",
        &ham,
        &psi0,
        &omegas,
        &coeffs,
        
        &BandParams { big_n: 8.0, h: 0.5, t: 0.8 },
        
        &[4, 6, 8],
        ks[2],
    );
}

// ───────────────────────── tiny state-building helpers ────────────────────

trait StateExt {
    fn pipe_boson(self, inner: &InnerBosonicState, amp: Complex64) -> QuantumState;
    fn pipe_fermion(self, mode: u32, amp: Complex64) -> QuantumState;
    fn pipe_fermion_amp(self, mode: u32, amp: Complex64) -> QuantumState;
    fn scale_diff(self, other: &QuantumState) -> QuantumState;
    fn sub_amplitude(self, inner: &InnerBosonicState, amp: Complex64) -> QuantumState;
    fn norm(self) -> f64;
}

impl StateExt for QuantumState {
    fn pipe_boson(mut self, inner: &InnerBosonicState, amp: Complex64) -> QuantumState {
        self.scale_and_add(
            &QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner.clone())),
            amp,
        );
        self
    }
    fn pipe_fermion(self, mode: u32, amp: Complex64) -> QuantumState {
        self.pipe_fermion_amp(mode, amp)
    }
    fn pipe_fermion_amp(mut self, mode: u32, amp: Complex64) -> QuantumState {
        self.scale_and_add(
            &QuantumState::vacuum().apply(&Operator::OuterFermionCreate(
                nested_fock_algebra::InnerFermionicState {
                    modes: std::collections::BTreeSet::from([mode]),
                },
            )),
            amp,
        );
        self
    }
    fn scale_diff(mut self, other: &QuantumState) -> QuantumState {
        self.scale_and_add(other, Complex64::new(-1.0, 0.0));
        self
    }
    fn sub_amplitude(mut self, inner: &InnerBosonicState, amp: Complex64) -> QuantumState {
        self.scale_and_add(
            &QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner.clone())),
            -amp,
        );
        self
    }
    fn norm(self) -> f64 {
        QuantumState::inner_product(&self, &self).re.sqrt()
    }
}
