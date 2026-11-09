//! Historical single-real-field Hashimoto-SIRK comparison dynamics.
//!
//! The sm_higgs_hamiltonian builder uses independent field/momentum modes and
//! legacy contraction-dropping ordering. This suite is NOT full nested-SM
//! validation, a canonical radial quantization, or a proof of gauge invariance
//! or EWSB. A whole-line coherent state does not become comparison by
//! calling its coordinate radial. A physical radial reduction would require
//! the correct measure, constraints, operator/domain and an identification map.
//!
//! Historical names/assertions remain reproducible comparison diagnostics.
//! Full-SM numerical acceptance must instead use Hashimoto-SIRK on the complete
//! identified inner operator and its actual joint-state outer enclosure; see
//! docs/SM_HAMILTONIAN_AUDIT.md N1-N5 and timepiece/SM_ESA_COMPLETION_PLAN.md.

use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, evolve_restarted};
use nested_fock_algebra::{
    Operator, QuantumState, sm_higgs_classical_potential, sm_higgs_hamiltonian, sm_higgs_vev,
};
use num_complex::Complex64;

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(1_000_000),
        brst_tol: 1e-10,
        adaptive: false,
        // Unit-norm frame: exact reparametrization, the admitted
        // conditioning device (guide §4.5 item 3) — required for the
        // quartic Higgs wall.
        unit_norm_steps: true,
    }
}

/// One empty bosonic universe (so inner boson ops can act), the outer-vacuum
/// comparison background (not the outer vacuum).
fn boson_background() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(
        nested_fock_algebra::InnerBosonicState::vacuum(),
    ))
}

/// Radial-mode coherent state |α⟩ = e^{−α²/2} Σ_n (α^n/n!) (a†)^n |0⟩ on
/// mode 0 (the legacy real comparison field), truncated at `n_max`.
/// For real α: ⟨φ⟩ = ⟨a†+a⟩ = 2α.
fn coherent_radial(alpha: f64, n_max: usize) -> QuantumState {
    let vac = boson_background();
    let w0 = (-0.5 * alpha * alpha).exp();
    // ψ ← w0 · |0⟩
    let mut psi = vac.clone();
    psi.scale_and_add(&psi.clone(), Complex64::new(w0 - 1.0, 0.0));
    let mut cur = vac;
    let mut fact = 1.0f64;
    for n in 1..=n_max {
        cur = cur.apply(&Operator::InnerBosonCreate(0));
        fact *= n as f64;
        let w = w0 * alpha.powi(n as i32) / fact;
        psi.scale_and_add(&cur, Complex64::new(w, 0.0));
    }
    psi
}

/// ⟨φ⟩ = ⟨ψ|(a†+a)|ψ⟩ / ⟨ψ|ψ⟩ on mode 0.
fn phi_expectation(psi: &QuantumState) -> f64 {
    let mut up = psi.apply(&Operator::InnerBosonCreate(0));
    up.scale_and_add(
        &psi.apply(&Operator::InnerBosonAnnihilate(0)),
        Complex64::new(1.0, 0.0),
    );
    QuantumState::inner_product(&up, psi).re / QuantumState::inner_product(psi, psi).re
}

/// Comparison observable ⟨φ†φ⟩ = ⟨φ²/2⟩ = ⟨(a†+a)²⟩/2 on mode 0.
fn phi_dag_phi_expectation(psi: &QuantumState) -> f64 {
    let mut u = psi.apply(&Operator::InnerBosonCreate(0));
    u.scale_and_add(
        &psi.apply(&Operator::InnerBosonAnnihilate(0)),
        Complex64::new(1.0, 0.0),
    );
    let mut u2 = u.apply(&Operator::InnerBosonCreate(0));
    u2.scale_and_add(
        &u.apply(&Operator::InnerBosonAnnihilate(0)),
        Complex64::new(1.0, 0.0),
    );
    QuantumState::inner_product(&u2, psi).re / (2.0 * QuantumState::inner_product(psi, psi).re)
}

fn energy_expectation(h: &nested_fock_algebra::Hamiltonian, psi: &QuantumState) -> f64 {
    QuantumState::inner_product(&h.apply(psi), psi).re / QuantumState::inner_product(psi, psi).re
}

// ── 1. condensate stability: the broken-like state survives SIRK ────────────

#[test]
fn sm_higgs_mechanism_condensate_mimics_broken_vacuum() {
    let (mu2, lambda) = (1.0f64, 0.5f64);
    let v = sm_higgs_vev(mu2, lambda); // √(μ²/λ) = √2
    let h = sm_higgs_hamiltonian(mu2, lambda); // the UNBROKEN Mexican hat

    // displaced condensate: ⟨φ⟩ = v + δ (broken-state mimic), small δ.
    let delta = 0.2 * v;
    let alpha = (v + delta) / 2.0; // ⟨φ⟩ = 2α
    let psi0 = coherent_radial(alpha, 14);
    let n0 = psi0.norm();
    assert!((n0 - 1.0).abs() < 1e-9, "probe must be normalized, n0={n0}");

    let phi0 = phi_expectation(&psi0);
    assert!(
        (phi0 - (v + delta)).abs() < 1e-8,
        "initial condensate must sit at v+δ: ⟨φ⟩={phi0}"
    );
    let e0 = energy_expectation(&h, &psi0);

    // Scalar comparison condensate content: ⟨φ†φ⟩ = ⟨φ⟩²/2 + ½·Δφ², and the
    // coherent state has Δφ² = 1 exactly — pin the sharp identity at t = 0.
    let xd0 = phi_dag_phi_expectation(&psi0);
    assert!(
        (xd0 - (0.5 * phi0 * phi0 + 0.5)).abs() < 1e-9,
        "coherent condensate must satisfy ⟨φ†φ⟩ = ⟨φ⟩²/2 + ½: {xd0} vs {}",
        0.5 * phi0 * phi0 + 0.5
    );

    // one oscillation period: T = 2π/m_h, m_h = √(2λ) v.
    let m_h = (2.0 * lambda).sqrt() * v;
    let period = 2.0 * std::f64::consts::PI / m_h;

    // Hashimoto-SIRK evolution to a set of times within the period.
    for &frac in &[0.0f64, 0.25, 0.5, 0.75, 1.0] {
        let t = frac * period;
        let psi_t = evolve_restarted(&h, &psi0, t, 6, 6, &best_device(), None, &opts())
            .expect("SIRK evolve must complete");
        let n_t = psi_t.norm();
        assert!(
            (n_t - n0).abs() < 1e-8,
            "norm must be conserved at t={t}: |Δ‖ψ‖| = {:.2e}",
            (n_t - n0).abs()
        );
        let e_t = energy_expectation(&h, &psi_t);
        assert!(
            (e_t - e0).abs() < 1e-6,
            "energy must be conserved at t={t}: |Δ⟨H⟩| = {:.2e}",
            (e_t - e0).abs()
        );
        let phi_t = phi_expectation(&psi_t);
        // the condensate must NOT collapse to the symmetric point 0 — it
        // mimics the broken state for the whole evolution:
        assert!(
            phi_t > 0.5 * v,
            "condensate must stay near the broken minimum at t={t}: ⟨φ⟩={phi_t}"
        );
        // comparison ⟨φ†φ⟩ stays within 25% of its condensate value —
        // the comparison content of the broken-like state does not
        // deplete (measured orbit stays within ~10%):
        let xd_t = phi_dag_phi_expectation(&psi_t);
        assert!(
            (xd_t - xd0).abs() < 0.25 * xd0,
            "⟨φ†φ⟩ must stay within 25% of its condensate value {xd0} at t={t}: {xd_t}"
        );
        eprintln!("higgs-mechanism t={t:.3}: ⟨φ⟩={phi_t:.6}, ⟨φ†φ⟩={xd_t:.6}, ⟨H⟩={e_t:.6}");
    }
}

// ── 2. the radial mode oscillates at the Higgs mass ─────────────────────────

#[test]
fn sm_higgs_mechanism_radial_mass_from_oscillation() {
    let (mu2, lambda) = (1.0f64, 0.5f64);
    let v = sm_higgs_vev(mu2, lambda);
    let h = sm_higgs_hamiltonian(mu2, lambda);
    let m_h = (2.0 * lambda).sqrt() * v;

    // m_h² = 2λv² must equal the numerical curvature V''(v) of the
    // UNBROKEN potential (the module's CHECK 14, second route):
    let hv = 1e-5 * v.max(1.0);
    let v_of = |x: f64| sm_higgs_classical_potential(x, mu2, lambda);
    let d2 = (v_of(v + hv) - 2.0 * v_of(v) + v_of(v - hv)) / (hv * hv);
    assert!(
        (d2 - m_h * m_h).abs() < 1e-4 * m_h * m_h,
        "V''(v) must equal m_h² = 2λv²: V''={d2}, m_h²={}",
        m_h * m_h
    );

    // Sample ⟨φ(t)⟩ on a grid under SIRK evolution.
    let delta = 0.15 * v;
    let alpha = (v + delta) / 2.0;
    let psi0 = coherent_radial(alpha, 14);
    let period = 2.0 * std::f64::consts::PI / m_h;
    let n_samples = 7usize;
    let mut ts = Vec::with_capacity(n_samples);
    let mut phis = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let t = period * k as f64 / (n_samples - 1) as f64;
        let psi_t = evolve_restarted(&h, &psi0, t, 6, 6, &best_device(), None, &opts())
            .expect("SIRK evolve must complete");
        ts.push(t);
        phis.push(phi_expectation(&psi_t));
    }

    // Least-squares fit c + A cos ωt + B sin ωt on a grid of candidate ω;
    // the best ω is the measured radial-mode frequency.
    let mut best = (f64::INFINITY, 0.0f64);
    let mut omega = 0.3 * m_h;
    while omega < 2.5 * m_h {
        // normal equations for (c, A, B) at this ω
        let mut mtx = [[0.0f64; 3]; 3];
        let mut rhs = [0.0f64; 3];
        for (&t, &p) in ts.iter().zip(phis.iter()) {
            let basis = [1.0, (omega * t).cos(), (omega * t).sin()];
            for i in 0..3 {
                for j in 0..3 {
                    mtx[i][j] += basis[i] * basis[j];
                }
                rhs[i] += basis[i] * p;
            }
        }
        // solve 3x3 by Cramer (well-conditioned on this grid)
        let det = |m: [[f64; 3]; 3]| {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        let d = det(mtx);
        if d.abs() > 1e-30 {
            let mut cols = [mtx; 3];
            let mut sol = [0.0f64; 3];
            for i in 0..3 {
                cols[i] = mtx;
                for r in 0..3 {
                    cols[i][r][i] = rhs[r];
                }
                sol[i] = det(cols[i]) / d;
            }
            let mut res = 0.0f64;
            for (&t, &p) in ts.iter().zip(phis.iter()) {
                let model = sol[0] + sol[1] * (omega * t).cos() + sol[2] * (omega * t).sin();
                res += (model - p) * (model - p);
            }
            if res < best.0 {
                best = (res, omega);
            }
        }
        omega += 0.01 * m_h;
    }
    let measured = best.1;
    assert!(
        (measured - m_h).abs() < 0.25 * m_h,
        "measured radial-mode frequency {measured} must match the Higgs mass {m_h} \
         (anharmonic shift of the displaced condensate is small)"
    );
    eprintln!(
        "higgs-mechanism radial mass: measured ω={measured:.6}, m_h={m_h:.6}, \
         v={v:.6}, m_h²=2λv²={:.6}",
        m_h * m_h
    );
}

// ── 3. comparison in/out states return to the condensate ──────────────

#[test]
fn sm_higgs_mechanism_gauge_invariant_in_out_states() {
    let (mu2, lambda) = (1.0f64, 0.5f64);
    let v = sm_higgs_vev(mu2, lambda);
    let h = sm_higgs_hamiltonian(mu2, lambda);
    let m_h = (2.0 * lambda).sqrt() * v;
    let period = 2.0 * std::f64::consts::PI / m_h;

    // comparison (radial-only) condensate — the state that mimics the
    // broken vacuum:
    let delta = 0.1 * v;
    let psi0 = coherent_radial((v + delta) / 2.0, 14);
    let x0 = phi_dag_phi_expectation(&psi0);

    // evolve through one full period and compare in/out:
    let psi_t = evolve_restarted(&h, &psi0, period, 6, 6, &best_device(), None, &opts())
        .expect("SIRK evolve must complete");
    let xt = phi_dag_phi_expectation(&psi_t);
    assert!(
        (xt - x0).abs() < 0.25 * x0.abs().max(1.0),
        "comparison ⟨φ†φ⟩ must return to its initial value after one \
         period: in={x0}, out={xt}"
    );
    // the fidelity |⟨ψ(T)|ψ(0)⟩| is large: the condensate state is
    // preserved by the dynamics (a full revival at the radial period):
    let ov = QuantumState::inner_product(&psi_t, &psi0).norm();
    assert!(
        ov > 0.5,
        "in/out fidelity must be large after one period: |⟨ψ(T)|ψ(0)⟩|={ov}"
    );
    // and both ends sit at the broken-like condensate, not the symmetric
    // point:
    let phi_in = phi_expectation(&psi0);
    let phi_out = phi_expectation(&psi_t);
    assert!(
        phi_in > 0.5 * v,
        "in-state must be the broken-like condensate"
    );
    assert!(
        phi_out > 0.5 * v,
        "out-state must be the broken-like condensate: ⟨φ⟩={phi_out}"
    );
    eprintln!(
        "higgs-mechanism in/out: ⟨φ†φ⟩ {x0:.6}→{xt:.6}, fidelity {ov:.6}, \
         ⟨φ⟩ {phi_in:.6}→{phi_out:.6}"
    );
}
