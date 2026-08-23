//! Precision Quantum Electrodynamics: numerical QED predictions checked against
//! published experimental measurements and perturbative calculations — the
//! counterpart of `qed_validation.rs` (which exercises the Fock-space / SIRK
//! machinery itself). Every test here compares a well-defined QED number with
//! its published value.
//!
//!   1. `qed_electron_anomalous_moment_g_minus_2` — the electron g−2 anomaly:
//!      the Schwinger term α/2π, the two-loop coefficient A₂ = −0.3284789656,
//!      and the CODATA a_e = 0.00115965218059.
//!   2. `qed_compton_kinematics_and_thomson_limit` — λ_C = h/mc = 2.4263 pm,
//!      Δλ = λ_C(1−cosθ), the Cs-137 184 keV backscatter peak, the Thomson
//!      cross-section σ_T = 8πr_e²/3, and the Klein–Nishina limits.
//!   3. `qed_positronium_spectrum_and_lifetimes` — E = −Ry/2, the 203.392 GHz
//!      ground-state hyperfine splitting, the 125.14 ps para-Ps (2γ) and
//!      142.05 ns ortho-Ps (3γ) lifetimes.
//!   4. `qed_uehling_vacuum_polarization_component` — the −27.1 MHz
//!      vacuum-polarization component of the 2S Lamb shift.
//!   5. `qed_hydrogen_lamb_shift_bethe_estimate` — Bethe's 1947 non-relativistic
//!      self-energy formula against the measured 2S (1057.8 MHz) and 3S
//!      Lamb shifts.
//!   6. `qed_hydrogen_fine_structure_and_rydberg_spectrum` — the Rydberg ladder
//!      and the 10.95 GHz n=2 fine-structure splitting.
//!   7. `qed_casimir_energy_and_force` — the zeta-regularized zero-point sums
//!      (Σn → −1/12, Σn³ → 1/120) and the published E/A = −π²ħc/720d³,
//!      F/A = −π²ħc/240d⁴ (1.30 mN/m² at 1 µm, Lamoreaux 1997).
//!   8. `qed_blackbody_photon_gas_planck_spectrum` — statistical QED: U/V =
//!      π²T⁴/15, the Stefan–Boltzmann constant, and Wien's displacement law.

/// CODATA 2018/2022 physical constants.
const ALPHA: f64 = 1.0 / 137.035999084; // fine-structure constant
const M_EV: f64 = 0.51099895000e6; // electron rest energy, eV
const RY_EV: f64 = 13.605693122994; // Rydberg constant, eV
const H_EV_S: f64 = 4.135667696e-15; // Planck constant, eV·s
const HBAR_EV_S: f64 = 6.582119569e-16; // ħ, eV·s
const C_MS: f64 = 299792458.0; // speed of light, m/s

// ── 1. Electron anomalous magnetic moment (g−2) ──────────────────────────────

#[test]
fn qed_electron_anomalous_moment_g_minus_2() {
    // CODATA 2022 electron g−2 anomaly (theory and experiment agree to ~0.1 ppb
    // — the most precise test of QED). The perturbative expansion is
    //   a_e = α/2π + (α/π)² A₂ + (α/π)³ A₃ + …,
    // with A₂ = −0.32847896557919378 the exact two-loop (Sommerfield–Peterman)
    // coefficient and A₃ = 1.181241456… the three-loop value.
    let a_e_exp = 0.00115965218059;

    // (a) The leading-order Schwinger term α/2π = 0.0011614…
    let schwinger = ALPHA / (2.0 * std::f64::consts::PI);
    let rel_schw = (schwinger - a_e_exp).abs() / a_e_exp;
    assert!(
        rel_schw < 2e-3,
        "α/2π = {schwinger:.9e} must reproduce a_e = {a_e_exp:.9e} to the leading order \
         (rel err {:.4}%)",
        rel_schw * 100.0
    );

    // (b) Adding the two-loop term brings the prediction to ~1e-5 of experiment
    //     (the residual is the O(α³) contribution, which is itself ~1e-8).
    let a2 = -0.32847896557919378;
    let a_pred = schwinger + (ALPHA / std::f64::consts::PI).powi(2) * a2;
    let rel_two = (a_pred - a_e_exp).abs() / a_e_exp;
    assert!(
        rel_two < 3e-5,
        "α/2π + (α/π)²A₂ = {a_pred:.9e} must match a_e to ~1e-5 (rel err {:.4}%)",
        rel_two * 100.0
    );

    // (c) Inverted: the two-loop coefficient extracted from experiment.
    let a2_derived = (a_e_exp - schwinger) / (ALPHA / std::f64::consts::PI).powi(2);
    assert!(
        (a2_derived - a2).abs() / a2.abs() < 0.01,
        "two-loop coefficient derived from experiment {a2_derived:.6} must match the \
         published A₂ = {a2}"
    );

    eprintln!(
        "qed_g_minus_2: α/2π = {schwinger:.10}, α/2π+(α/π)²A₂ = {a_pred:.10}, a_e = {a_e_exp}; \
         A₂(derived) = {a2_derived:.8} vs published {a2}"
    );
}

// ── 2. Compton scattering kinematics and the Thomson limit ──────────────────

#[test]
fn qed_compton_kinematics_and_thomson_limit() {
    // (a) The Compton wavelength λ_C = h/(m_e c) — a CODATA exact value.
    let lambda_c = H_EV_S * C_MS / M_EV;
    assert!(
        (lambda_c - 2.42631023867e-12).abs() / 2.42631023867e-12 < 1e-6,
        "Compton wavelength must be h/mc = 2.42631023867e-12 m, got {lambda_c:.6e}"
    );

    // (b) The Compton shift Δλ = λ_C(1 − cos θ) — the exact kinematics.
    for &(theta_deg, factor) in &[(90.0f64, 1.0f64), (180.0, 2.0), (60.0, 0.5), (0.0, 0.0)] {
        let th = theta_deg.to_radians();
        let dl = lambda_c * (1.0 - th.cos());
        assert!(
            (dl - lambda_c * factor).abs() < 1e-15 * lambda_c,
            "Δλ(θ = {theta_deg}°) must be λ_C·(1−cosθ) = {:.3e}, got {dl:.3e}",
            lambda_c * factor
        );
    }

    // (c) Outgoing photon energy E′ = E/(1 + (E/mc²)(1 − cos θ)). For a 662 keV
    //     γ (¹³⁷Cs) backscattered at 180° the peak sits at 184 keV — the famous
    //     Cs-137 Compton backscatter peak (measured 184.4 keV).
    let e_cs = 662.0e3;
    let e_back = e_cs / (1.0 + (e_cs / M_EV) * 2.0);
    assert!(
        (e_back - 184.4e3).abs() / 184.4e3 < 0.005,
        "Cs-137 180° backscatter must peak at 184.4 keV, got {e_back:.1} eV"
    );
    // A 511 keV photon at 180°: E′ = E/3; at 90°: E′ = E/2.
    assert!(((M_EV / 3.0) - M_EV / (1.0 + 2.0)).abs() < 1.0);
    assert!(((M_EV / 2.0) - M_EV / (1.0 + 1.0)).abs() < 1.0);

    // (d) Thomson cross-section: σ_T = (8π/3)r_e² with r_e = α·ħc/(mc²) the
    //     classical electron radius.
    let hbar_c = HBAR_EV_S * C_MS; // ħc = 1.9733e-7 eV·m
    let r_e = ALPHA * hbar_c / M_EV;
    assert!(
        (r_e - 2.8179403262e-15).abs() / 2.8179403262e-15 < 1e-6,
        "classical electron radius r_e = αħc/mc² = 2.81794e-15 m, got {r_e:.6e}"
    );
    let sigma_t = 8.0 * std::f64::consts::PI / 3.0 * r_e * r_e;
    assert!(
        (sigma_t - 6.6524587321e-29).abs() / 6.6524587321e-29 < 1e-4,
        "Thomson cross-section σ_T = 6.65246e-29 m², got {sigma_t:.6e}"
    );

    // (e) The Klein–Nishina total cross-section:
    //     σ(ε) = (πr_e²/ε)[(1−2(ε+1)/ε²)ln(2ε+1) + 1/2 + 4/ε − 1/(2(2ε+1)²)]
    //     must reduce to Thomson at low energy and to the log form
    //     (πr_e²/ε)(ln 2ε + 1/2) at high energy.
    let kn = |eps: f64| -> f64 {
        let f = (1.0 - 2.0 * (eps + 1.0) / (eps * eps)) * (2.0 * eps + 1.0).ln()
            + 0.5
            + 4.0 / eps
            - 1.0 / (2.0 * (2.0 * eps + 1.0).powi(2));
        std::f64::consts::PI * r_e * r_e / eps * f
    };
    let ratio_low = kn(0.005) / sigma_t;
    assert!(
        (ratio_low - 1.0).abs() < 0.012,
        "Klein–Nishina must → Thomson as ε→0: σ(0.005)/σ_T = {ratio_low:.5}"
    );
    let asym = |eps: f64| {
        std::f64::consts::PI * r_e * r_e / eps * ((2.0 * eps).ln() + 0.5)
    };
    assert!(
        (kn(100.0) - asym(100.0)).abs() / kn(100.0) < 0.02,
        "Klein–Nishina must → (πr_e²/ε)(ln 2ε + ½) at high energy: \
         σ(100) = {:.5} vs asymptotic {:.5}",
        kn(100.0),
        asym(100.0)
    );

    eprintln!(
        "qed_compton: λ_C = {lambda_c:.6e} m, Cs-137 backscatter {e_back:.1} eV (184.4), \
         σ_T = {sigma_t:.5e} m², σ(0.005)/σ_T = {ratio_low:.5}"
    );
}

// ── 3. Positronium: spectrum and annihilation lifetimes ──────────────────────

#[test]
fn qed_positronium_spectrum_and_lifetimes() {
    // (a) Ground-state binding: the reduced mass is m_e/2, so
    //     E_gs = −Ry·(μ/m_e) = −Ry/2 = −6.80 eV.
    let e_gs = -RY_EV / 2.0;
    assert!(
        (e_gs + 6.8029).abs() / 6.8029 < 1e-3,
        "positronium ground state must be −Ry/2 = −6.8029 eV, got {e_gs}"
    );

    // (b) Ground-state hyperfine splitting (ortho–para): the leading-order
    //     ν = (7/12)α⁴mc²/h = 204.4 GHz; the measured value is 203.392 GHz
    //     (the −0.5% shift is the radiative/recoil corrections — this is one
    //     of the most precise QED-vs-experiment comparisons, agreement ~1e-4).
    let a4 = ALPHA.powi(4);
    let hfs_lead = (7.0 / 12.0) * a4 * M_EV / H_EV_S;
    let hfs_exp = 203.392e9;
    assert!(
        (hfs_lead - hfs_exp).abs() / hfs_exp < 0.01,
        "leading-order Ps hyperfine splitting (7/12)α⁴mc²/h = {:.1} GHz must be within \
         1% of the measured 203.392 GHz",
        hfs_lead / 1e9
    );

    // (c) Para-positronium (2γ) decay: Γ = α⁵mc²/(2ħ) → τ = 124.5 ps; the
    //     measured lifetime is 125.14 ps (theory 125.17 ps).
    let gamma_para = ALPHA.powi(5) * M_EV / (2.0 * HBAR_EV_S);
    let tau_para = 1.0 / gamma_para;
    assert!(
        (tau_para - 125.14e-12).abs() / 125.14e-12 < 0.01,
        "para-Ps lifetime α⁵mc²/(2ħ) → τ = {:.3} ps must match the measured 125.14 ps",
        tau_para * 1e12
    );

    // (d) Ortho-positronium (3γ) decay: Γ = (2(π²−9)/9π)α⁶mc²/ħ → τ = 138.7 ns
    //     (leading order); the O(α) radiative correction brings it to 142.0 ns,
    //     matching the measured 142.05 ns.
    let a6 = ALPHA.powi(6);
    let coeff = 2.0 * (std::f64::consts::PI * std::f64::consts::PI - 9.0)
        / (9.0 * std::f64::consts::PI);
    let gamma_ortho = coeff * a6 * M_EV / HBAR_EV_S;
    let tau_ortho = 1.0 / gamma_ortho;
    assert!(
        (tau_ortho - 142.05e-9).abs() / 142.05e-9 < 0.03,
        "ortho-Ps leading-order lifetime {:.3} ns must be within 3% of 142.05 ns",
        tau_ortho * 1e9
    );
    let tau_corr = tau_ortho * (1.0 + 10.286 * ALPHA / std::f64::consts::PI);
    assert!(
        (tau_corr - 142.05e-9).abs() / 142.05e-9 < 0.01,
        "ortho-Ps with the O(α) correction {:.3} ns must match 142.05 ns",
        tau_corr * 1e9
    );

    eprintln!(
        "qed_positronium: E_gs = {e_gs:.4} eV, hfs {:.1} GHz (203.392), \
         τ_para = {:.3} ps (125.14), τ_ortho = {:.3} ns ({:.3} with O(α)) (142.05)",
        hfs_lead / 1e9,
        tau_para * 1e12,
        tau_ortho * 1e9,
        tau_corr * 1e9
    );
}

// ── 4. Uehling: the vacuum-polarization component of the Lamb shift ─────────

#[test]
fn qed_uehling_vacuum_polarization_component() {
    // The Uehling (vacuum-polarization) energy shift of an S level:
    //   ΔE_VP(nS) = −(4/15)(α/π)(Zα)⁴ mc² / n³.
    // For hydrogen Z=1, n=2: Δν = −27.1 MHz — the published VP component of
    // the 2S₁/₂ Lamb shift (the self-energy component is +1085 MHz, and the
    // sum with the relativistic corrections is the 1057.8 MHz total).
    let dvp_hz = -(4.0 / 15.0) * (ALPHA / std::f64::consts::PI) * ALPHA.powi(4) * M_EV / 8.0
        / H_EV_S;
    assert!(
        (dvp_hz + 27.1e6).abs() / 27.1e6 < 0.02,
        "Uehling shift of 2S must be −27.1 MHz, got {dvp_hz:.3e} Hz"
    );

    eprintln!("qed_uehling: Δν_VP(2S) = {dvp_hz:.3e} Hz = {:.3} MHz (published −27.1)", dvp_hz / 1e6);
}

// ── 5. Bethe's Lamb shift estimate ──────────────────────────────────────────

#[test]
fn qed_hydrogen_lamb_shift_bethe_estimate() {
    // Bethe (1947): the non-relativistic self-energy of a hydrogenic level is
    //   ΔE_n = (8/3π)(α³/n³) Ry · ln(mc²/ε̄),
    // with ε̄ the Bethe average excitation energy — for the 2S state
    // ε̄ = 16.6 Ry. For n=2 this gives 1048 MHz against the measured
    // 2S₁/₂−2P₁/₂ Lamb shift 1057.845 MHz (Bethe's original estimate was
    // 1040 MHz; the relativistic and higher radiative corrections add the
    // remaining ~1%). The 1/n³ scaling gives the 3S shift ~310 MHz.
    let a3 = ALPHA.powi(3);
    let ebar = 16.6 * RY_EV;
    let ln_ratio = (M_EV / ebar).ln();
    for &(n, mhz_pub) in &[(2u32, 1057.845f64), (3, 313.4)] {
        let de = (8.0 / (3.0 * std::f64::consts::PI)) * (a3 / (n as f64).powi(3))
            * RY_EV
            * ln_ratio;
        let mhz = de / H_EV_S / 1e6;
        assert!(
            (mhz - mhz_pub).abs() / mhz_pub < 0.02,
            "Bethe estimate for the {n}S Lamb shift {mhz:.1} MHz must be within 2% of \
             the published {mhz_pub} MHz"
        );
    }

    eprintln!("qed_lamb_shift_bethe: 2S {:.1} MHz (1057.8), 3S {:.1} MHz (313)", {
        let de = (8.0 / (3.0 * std::f64::consts::PI)) * (a3 / 8.0) * RY_EV * ln_ratio;
        de / H_EV_S / 1e6
    }, {
        let de = (8.0 / (3.0 * std::f64::consts::PI)) * (a3 / 27.0) * RY_EV * ln_ratio;
        de / H_EV_S / 1e6
    });
}

// ── 6. Rydberg ladder and the n=2 fine structure ────────────────────────────

#[test]
fn qed_hydrogen_fine_structure_and_rydberg_spectrum() {
    // (a) The Rydberg ladder E_n = −Ry/n² (Ry∞ convention): 1S−2S = 3Ry/4.
    let e12 = (3.0 / 4.0) * RY_EV;
    assert!(
        (e12 - 10.2043).abs() < 1e-3,
        "1S−2S must be 3Ry/4 = 10.2043 eV, got {e12}"
    );

    // (b) The n=2 fine-structure splitting (Dirac theory):
    //     ΔE(2P₃/₂ − 2P₁/₂) = α²Ry/16 = 4.53e-5 eV = 10.95 GHz
    //     (measured 10.969 GHz). The 2S₁/₂−2P₁/₂ degeneracy of Dirac theory is
    //     the one QED breaks with the 1057.8 MHz Lamb shift (test above).
    let fs_hz = ALPHA.powi(2) * RY_EV / 16.0 / H_EV_S;
    assert!(
        (fs_hz - 10.969e9).abs() / 10.969e9 < 0.01,
        "n=2 fine structure α²Ry/16 = {:.3} GHz must be within 1% of 10.969 GHz",
        fs_hz / 1e9
    );

    eprintln!(
        "qed_fine_structure: 1S−2S = {e12:.4} eV, 2P₃/₂−2P₁/₂ = {:.4} GHz (10.969)",
        fs_hz / 1e9
    );
}

// ── 7. Casimir energy and force ─────────────────────────────────────────────

#[test]
fn qed_casimir_energy_and_force() {
    // (a) The zeta-regularized zero-point sums behind the Casimir derivation:
    //     with a smooth cutoff, Σ n e^{−δn} − 1/δ² → ζ(−1) = −1/12 and
    //     Σ n³ e^{−δn} − 6/δ⁴ → ζ(−3) = 1/120 as δ → 0.
    for &delta in &[0.05, 0.02, 0.01] {
        let s1: f64 = (1..200_000)
            .map(|n| (n as f64) * (-delta * n as f64).exp())
            .sum();
        let z1 = s1 - 1.0 / (delta * delta);
        assert!(
            (z1 + 1.0 / 12.0).abs() < 1e-3,
            "Σn e^{{−δn}} − 1/δ² must → ζ(−1) = −1/12 (δ = {delta}): got {z1:.6}"
        );
        let s3: f64 = (1..200_000)
            .map(|n| (n as f64).powi(3) * (-delta * n as f64).exp())
            .sum();
        let z3 = s3 - 6.0 / (delta * delta * delta * delta);
        assert!(
            (z3 - 1.0 / 120.0).abs() < 1e-3,
            "Σn³ e^{{−δn}} − 6/δ⁴ must → ζ(−3) = 1/120 (δ = {delta}): got {z3:.6}"
        );
    }

    // (b) The published Casimir values: E/A = −π²ħc/(720d³) and
    //     F/A = −π²ħc/(240d⁴). At d = 1 µm the force is −1.30 mN/m² — the
    //     value Lamoreaux (1997) measured to 5% (and modern experiments to
    //     ~0.1%).
    let hbar_c = HBAR_EV_S * C_MS; // eV·m
    let d: f64 = 1.0e-6;
    let e_area = -std::f64::consts::PI * std::f64::consts::PI * hbar_c
        / (720.0 * d.powi(3)); // eV/m²
    let f_area = -std::f64::consts::PI * std::f64::consts::PI * hbar_c
        / (240.0 * d.powi(4)); // eV/m³
    let e_j = e_area * 1.602176634e-19; // J/m²
    let f_n = f_area * 1.602176634e-19; // N/m²
    assert!(
        (f_n + 1.30e-3).abs() / 1.30e-3 < 0.01,
        "Casimir force at 1 µm must be −1.30 mN/m², got {f_n:.3e} N/m²"
    );
    assert!(
        (e_j + 4.33e-10).abs() / 4.33e-10 < 0.01,
        "Casimir energy at 1 µm must be −4.33e-10 J/m², got {e_j:.3e}"
    );

    // (c) The force is the derivative of the energy: E ∝ d⁻³, so F = 3E/d —
    //     i.e. 3/720 = 1/240 exactly.
    assert!(
        (f_area - 3.0 * e_area / d).abs() / f_area.abs() < 1e-12,
        "F = −dE/dd with E ∝ d⁻³ must give 3/720 = 1/240"
    );

    // (d) 1D scalar cavity: the regularized zero-point sum (πħc/2d)·Σn with
    //     Σn → −1/12 gives E = −πħc/(24d).
    let e_1d = -std::f64::consts::PI * hbar_c / (24.0 * d);
    assert!(
        (e_1d - (std::f64::consts::PI * hbar_c / (2.0 * d)) * (-1.0 / 12.0)).abs() < 1e-12,
        "1D zero-point sum must give E = −πħc/24d"
    );

    eprintln!(
        "qed_casimir: zeta sums → −1/12, 1/120 verified; F/A(1 µm) = {f_n:.3e} N/m², \
         E/A = {e_j:.3e} J/m²"
    );
}

// ── 8. Blackbody: the thermal photon gas ────────────────────────────────────

#[test]
fn qed_blackbody_photon_gas_planck_spectrum() {
    // (a) The photon-gas internal energy integral: with x = ħω/kT,
    //     U/V = (k⁴T⁴/π²ħ³c³)∫₀^∞ x³/(e^x−1) dx and the integral is exactly
    //     π⁴/15, so U/V = π²T⁴/15 (in ħ=c=k=1 units). Numerically:
    let integral: f64 = {
        let (a, b, n) = (0.0, 60.0, 2_000_000);
        let h = (b - a) / n as f64;
        // f(0) = 0 is the limit x³/(e^x−1) → x² as x → 0 (avoid 0/0 = NaN).
        let f = |x: f64| {
            if x < 1e-9 {
                0.0
            } else {
                x.powi(3) / (x.exp() - 1.0)
            }
        };
        let mut s = 0.5 * (f(a) + f(b));
        for i in 1..n {
            s += f(a + i as f64 * h);
        }
        s * h
    };
    let target = std::f64::consts::PI.powi(4) / 15.0;
    assert!(
        (integral - target).abs() / target < 1e-4,
        "∫x³/(e^x−1) must be π⁴/15 = {target}, got {integral}"
    );

    // (b) The Stefan–Boltzmann constant σ = π²k⁴/(60ħ³c²) = 5.6704e-8 W/m²K⁴
    //     (CODATA 2018: 5.670374419e-8, agreement ~1e-8).
    let k_si: f64 = 1.380649e-23;
    let hbar_si: f64 = 1.054571817e-34;
    let sigma = std::f64::consts::PI * std::f64::consts::PI * k_si.powi(4)
        / (60.0 * hbar_si.powi(3) * C_MS * C_MS);
    assert!(
        (sigma - 5.670374419e-8).abs() / 5.670374419e-8 < 1e-6,
        "Stefan–Boltzmann σ = {sigma:.6e} must be 5.670374419e-8 W/m²K⁴"
    );

    // (c) Wien's displacement law: the Planck peak. Per frequency,
    //     x_max solves x = 3(1 − e^{−x}) → x = 2.821439; per wavelength,
    //     x_max solves x = 5(1 − e^{−x}) → x = 4.965114, giving
    //     λ_max T = hc/(k·4.965114) = 2.897771955e-3 m·K.
    let mut xf: f64 = 2.5;
    for _ in 0..100 {
        xf = 3.0 * (1.0 - (-xf).exp());
    }
    assert!(
        (xf - 2.821439).abs() < 1e-5,
        "frequency-peak Wien root must be 2.821439, got {xf}"
    );
    let mut xl: f64 = 4.5;
    for _ in 0..100 {
        xl = 5.0 * (1.0 - (-xl).exp());
    }
    let lam_t = 1.986445857e-25 / (k_si * xl); // hc/(k·x) with hc in J·m
    assert!(
        (lam_t - 2.897771955e-3).abs() / 2.897771955e-3 < 1e-5,
        "λ_max T = hc/(kx) must be 2.897771955e-3 m·K, got {lam_t:.6e}"
    );

    // (d) Photon number density: n/V = (2ζ(3)/π²)T³ (ζ(3) = 1.2020569), and
    //     the radiation pressure P = U/(3V) (blackbody equation of state).
    let zeta3 = 1.2020569031595942854;
    let n_coeff = 2.0 * zeta3 / std::f64::consts::PI.powi(2);
    assert!(
        (n_coeff - 0.24357).abs() < 1e-3,
        "n/V = 2ζ(3)T³/π² coefficient {n_coeff:.6} must be 0.24357"
    );

    eprintln!(
        "qed_blackbody: ∫x³/(e^x−1) = {integral:.6} (π⁴/15 = {target}), σ_SB = {sigma:.6e}, \
         λ_max T = {lam_t:.6e} m·K (2.897772e-3), Wien roots {xf:.6} / {xl:.6}"
    );
}
