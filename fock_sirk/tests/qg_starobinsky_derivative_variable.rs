//! Starobinsky scalaron derivative-variable gauge fixing: the PHYSICAL
//! observables (the S34/S35 pattern — QED/QCD/QG/NS — applied to the R²
//! scalar sector with the Navier-Stokes derivatives-as-fields construction).
//!
//! The `.cdb`-derived gauge-fixed R² scalar sector is
//! `H_final = ½π² + ½(∇φ)² + V(φ)` with the massive scalaron `m² = M²/(12α)`.
//! Following the Navier-Stokes module (book.tex §4159-4197), the scalaron's
//! **spatial gradients** `g_i = ∂_iφ` are promoted to independent canonical
//! fields (inner Hermite-basis ladder modes, no lattice) and *fixed to the
//! values of the field derivatives* by the BRST charge `Ω = Σ_i g_i·c_i`
//! (the `gf_check_Dphi2 → 0` fixing of `docs/qg_starobinsky_hamiltonian.cdb`
//! Part 4; the gravity analogue of NS's `Ω = Σ_j u_{j,j}·c_j`).  In the
//! nested Fock space the field is the first moment of its inner space,
//! `φ(x) = Σ_n φ_n H_n(x)` with `φ_n = a†_n + a_n` and `∂_x H_n = 2n H_{n−1}`,
//! so the spatial derivative of the field operator is
//!
//!     ∂_x φ(x) = Σ_n φ_n·2n H_{n−1}(x) = Σ_m (2(m+1) φ_{m+1}) H_m(x),
//!
//! i.e. the mode-m component of the derivative is the operator
//! `D_m = 2(m+1) φ_{m+1}` — exactly the NS `D_m = 2(m+1) u_{m+1}` structure.
//!
//! THE GAUGE CONDITION IS VERIFIED BY CONSTRUCTION — it is not what these
//! tests check.  The physical initial wave-function *sets* the promoted
//! derivative variable `g_m = a†+a` (its own ladder mode) to the actual
//! derivative value `⟨g_m⟩ = 2(m+1)⟨φ_{m+1}⟩`, and because the derivative-
//! content modes carry no momenta, `[H, C_m] = 0` (`C_m = g_m − D_m`) makes
//! the condition an exact constant of the motion under the Hamiltonian flow
//! (bare or BRST-projected).  The question these tests answer is whether the
//! REMAINING numerical observables are CONSISTENT and CALCULABLE while the
//! gauge condition holds:
//!
//!  1. `qg_starobinsky_derivative_variable_physical_observables_1d` — with
//!     the massive fiber `H = m·N_0 + ½g_0²` (the normal-ordered massive
//!     mode plus the promoted gradient energy): the gradient content is
//!     frozen, the ENERGY `⟨H⟩` is conserved (unitary solver, to machine
//!     precision), the bare and the BRST-projected flows give IDENTICAL
//!     physical observables, and the Ehrenfest equation of motion holds
//!     numerically: `d⟨φ_0⟩/dt = ⟨i[H,φ_0]⟩ = m⟨π_0⟩` — the massive
//!     Klein–Gordon oscillation `⟨φ_0(t)⟩ = ⟨φ_0(0)⟩cos(mt)` of the scalaron
//!     (the R² scalar mode oscillates at its mass).  The one thing that is
//!     NOT exact under the *truncated* restarted-Krylov solver is the gauge
//!     condition itself: its drift is a controlled, quadratically convergent
//!     (in dt) numerical artifact — the exact flow conserves `C_0`
//!     identically, and the test verifies the drift shrinks as dt → 0.
//!
//!  2. `qg_starobinsky_derivative_variable_higher_hermite_modes` — the FULL
//!     multi-level fiber: derivative content `φ_1, φ_2, φ_3` and promoted
//!     `g_0 = 2φ_1, g_1 = 4φ_2, g_2 = 6φ_3`, so the field
//!     `φ(x) = Σ φ_n H_n(x)` and its derivative carry a GENUINE polynomial
//!     profile (not just the constant gradient `2φ_1`).  The pointwise
//!     identity `⟨g(x)⟩ = Σ⟨g_m⟩H_m(x) = ∂_x⟨φ(x)⟩` holds at every x, the
//!     promoted composite observables `⟨φ_m g_m⟩ = 2(m+1)⟨φ_m φ_{m+1}⟩` match
//!     the field-only values, and the gauge-fixed Hamiltonian carries the
//!     products of the field derivatives (`½Σg_m²` — allowed because the
//!     variables are fixed by BRST, exactly as the NS Hamiltonian carries
//!     `u_j·u_{i,j}`).
//!
//!  3. `qg_starobinsky_derivative_variable_unphysical_data_inconsistent` —
//!     unphysical initial data (promoted variable NOT set to the derivative
//!     value) is detected: `⟨C_0⟩ ≠ 0`, the physical observable is
//!     INCONSISTENT (`4⟨φ_0 g_0⟩ ≠ 8⟨φ_0 φ_1⟩`: the dynamics the Hamiltonian
//!     generates disagrees with the value read off the field alone), the
//!     violation carries Ω-content and is conserved (no self-fixing) — gauge
//!     fixing is genuinely required, exactly the NS pattern.
//!
//! Mode layout (minimal slices):
//!   1D: 0 = scalaron value φ_0, 1 = gradient-content φ_1,
//!       2 = promoted derivative variable g_0 = a†_2 + a_2, ghost c_0.
//!   multi: 0 = value, 1 = φ_1, 2 = φ_2, 3 = φ_3 (content),
//!       4 = g_0, 5 = g_1, 6 = g_2 (promoted), ghosts c_0, c_1, c_2.
//!
//! The fiber Hamiltonian `H = m·N_0 + ½Σg_m²` is Hermitian by construction
//! and normal-ordered (`⟨0|H|0⟩ = 0`): the value mode carries the massive
//! dynamics while the derivative-content modes are untouched —
//! `[H, φ_i] = [H, g_m] = 0` exactly, so the gauge conditions `C_m = g_m −
//! 2(m+1)φ_{m+1}` are constants of the motion and `[H, Ω] = 0`, `Ω² = 0`
//! (first-class constraints).

use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, evolve_restarted};
use nested_fock_algebra::{Hamiltonian, InnerBosonicState, Operator, QuantumState};
use num_complex::Complex64;

/// A bosonic occupation eigenstate: one universe whose inner occupation of
/// `mode` is `count` (zero drops the mode, matching the true vacuum).
fn bos_state(mode: u32, count: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    if count > 0 {
        inner.modes.insert(mode, count);
    }
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

/// A ghost-carrying state: a bosonic universe plus a fermionic universe with
/// one ghost in `ghost_mode`.
fn ghost_state(bosonic: InnerBosonicState, ghost_mode: u32) -> QuantumState {
    QuantumState::vacuum()
        .apply(&Operator::OuterBosonCreate(bosonic))
        .apply(&Operator::OuterFermionCreate(nested_fock_algebra::InnerFermionicState {
            modes: std::collections::BTreeSet::from([ghost_mode]),
        }))
}

/// The field operator `φ_m = a†_m + a_m` terms: `(c, op)` pairs.
fn field_ops(mode: u32) -> Vec<(Complex64, Operator)> {
    vec![
        (Complex64::new(1.0, 0.0), Operator::InnerBosonCreate(mode)),
        (Complex64::new(1.0, 0.0), Operator::InnerBosonAnnihilate(mode)),
    ]
}

/// The momentum operator `π_m = i(a†_m − a_m)` terms.
fn momentum_ops(mode: u32) -> Vec<(Complex64, Operator)> {
    vec![
        (Complex64::i(), Operator::InnerBosonCreate(mode)),
        (Complex64::new(0.0, -1.0), Operator::InnerBosonAnnihilate(mode)),
    ]
}

/// The field operator `φ_m = a†_m + a_m` as a Hamiltonian (used to measure
/// the field-amplitude expectation `⟨φ_m⟩`).
fn field_hamiltonian(mode: u32) -> Hamiltonian {
    Hamiltonian {
        terms: field_ops(mode)
            .into_iter()
            .map(|(c, op)| (c, vec![op]))
            .collect(),
    }
}

/// Expectation `⟨ψ|A|ψ⟩ / ⟨ψ|ψ⟩` of a field operator (real).
fn field_expect(psi: &QuantumState, mode: u32) -> f64 {
    let a = field_hamiltonian(mode);
    let num = QuantumState::inner_product(psi, &a.apply(psi)).re;
    let den = QuantumState::inner_product(psi, psi).re;
    num / den
}

/// Expectation of the momentum operator `π_m = i(a†_m − a_m)`.
fn momentum_expect(psi: &QuantumState, mode: u32) -> f64 {
    let h = Hamiltonian {
        terms: vec![
            (Complex64::i(), vec![Operator::InnerBosonCreate(mode)]),
            (Complex64::new(0.0, -1.0), vec![Operator::InnerBosonAnnihilate(mode)]),
        ],
    };
    let num = QuantumState::inner_product(psi, &h.apply(psi)).re;
    let den = QuantumState::inner_product(psi, psi).re;
    num / den
}

/// Expectation of a product of field operators on DISTINCT modes,
/// `⟨ψ| ∏_m φ_m |ψ⟩ / ⟨ψ|ψ⟩` (the operators commute across modes, so the
/// order is irrelevant).
fn composite_expect(psi: &QuantumState, modes: &[u32]) -> f64 {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = vec![(Complex64::new(1.0, 0.0), vec![])];
    for &m in modes {
        let mut next = Vec::new();
        for (c, ops) in &terms {
            let mut o1 = ops.clone();
            o1.push(Operator::InnerBosonCreate(m));
            let mut o2 = ops.clone();
            o2.push(Operator::InnerBosonAnnihilate(m));
            next.push((*c, o1));
            next.push((*c, o2));
        }
        terms = next;
    }
    let h = Hamiltonian { terms };
    let num = QuantumState::inner_product(psi, &h.apply(psi)).re;
    let den = QuantumState::inner_product(psi, psi).re;
    num / den
}

/// Expectation of a Hermitian Hamiltonian `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`.
fn energy_expect(psi: &QuantumState, h: &Hamiltonian) -> f64 {
    let num = QuantumState::inner_product(psi, &h.apply(psi)).re;
    let den = QuantumState::inner_product(psi, psi).re;
    num / den
}

/// The two-level amplitude for a target field expectation t = 2α/(1+α²):
/// α = (1−√(1−t²))/t  (invertible on [0,1]).
fn amp(t: f64) -> f64 {
    if t.abs() < 1e-12 {
        0.0
    } else {
        (1.0 - (1.0 - t * t).sqrt()) / t
    }
}

/// Tensor product of two-level states (|0⟩+α_i|1⟩) over the given (mode, α)
/// pairs — the coherent-like occupation superpositions used as initial
/// wave-functions.  Unnormalized (the *_expect helpers divide by ⟨ψ|ψ⟩).
fn product_state(parts: &[(u32, f64)]) -> QuantumState {
    let mut psi = QuantumState::zero();
    let n = parts.len();
    for mask in 0..(1u32 << n) {
        let mut inner = InnerBosonicState::vacuum();
        let mut weight = 1.0_f64;
        let mut used = false;
        for (i, &(mode, a)) in parts.iter().enumerate() {
            if mask & (1 << i) != 0 {
                inner.modes.insert(mode, 1);
                weight *= a;
                used = true;
            }
        }
        if !used {
            psi.scale_and_add(
                &QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum())),
                Complex64::new(1.0, 0.0),
            );
        } else {
            let univ = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner));
            psi.scale_and_add(&univ, Complex64::new(weight, 0.0));
        }
    }
    psi
}

/// Normalize a state in place (returns a fresh normalized state).
fn normalize(psi: &QuantumState) -> QuantumState {
    let n = psi.norm();
    let mut out = QuantumState::zero();
    out.scale_and_add(psi, Complex64::new(1.0 / n, 0.0));
    out
}

/// Physicists' Hermite polynomial H_n(x): H_0 = 1, H_1 = 2x,
/// H_2 = 4x²−2, H_3 = 8x³−12x, ∂_x H_n = 2n H_{n−1}.
fn hermite(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => 2.0 * x,
        2 => 4.0 * x * x - 2.0,
        3 => 8.0 * x * x * x - 12.0 * x,
        _ => {
            // H_{n+1} = 2x H_n − 2n H_{n−1}
            let mut h0 = 1.0;
            let mut h1 = 2.0 * x;
            for k in 1..n {
                let h2 = 2.0 * x * h1 - 2.0 * (k as f64) * h0;
                h0 = h1;
                h1 = h2;
            }
            h1
        }
    }
}

/// The 1D massive scalaron fiber with the promoted derivative variable:
/// `H = m·N_0 + ½g_0²` where `N_0 = a†_0 a_0` is the massive mode (the
/// diagonal normal-ordered realization of ½π² + ½m²φ², so ⟨0|H|0⟩ = 0) and
/// `½g_0² = ½(a†_2 + a_2)²` is the promoted gradient energy (mode 2).  The
/// derivative-content mode 1 carries no terms: `[H, φ_1] = [H, g_0] = 0`, so
/// `C_0 = g_0 − 2φ_1` is a constant of the motion, while `[H, φ_0] ≠ 0`
/// (the value genuinely evolves — the massive Klein–Gordon oscillator).  By
/// Ehrenfest, `d⟨φ_0⟩/dt = m⟨π_0⟩`.
fn fiber_1d(m: f64) -> Hamiltonian {
    let mut terms = vec![
        (
            Complex64::new(m, 0.0),
            vec![
                Operator::InnerBosonCreate(0),
                Operator::InnerBosonAnnihilate(0),
            ],
        ),
        // ½g_0² on mode 2, normal-ordered (⟨0|½g²|0⟩ = 0): ½a†² + a†a + ½a²
        (
            Complex64::new(0.5, 0.0),
            vec![
                Operator::InnerBosonCreate(2),
                Operator::InnerBosonCreate(2),
            ],
        ),
        (
            Complex64::new(1.0, 0.0),
            vec![
                Operator::InnerBosonCreate(2),
                Operator::InnerBosonAnnihilate(2),
            ],
        ),
        (
            Complex64::new(0.5, 0.0),
            vec![
                Operator::InnerBosonAnnihilate(2),
                Operator::InnerBosonAnnihilate(2),
            ],
        ),
    ];
    Hamiltonian { terms }
}

/// The multi-level massive scalaron fiber: `H = m·N_0 + ½Σ_{m=0}^{2} g_m²`
/// with the promoted derivative variables `g_m` on modes 4, 5, 6 and the
/// derivative content `φ_1, φ_2, φ_3` on modes 1, 2, 3 (frozen — no terms).
/// `[H, φ_i] = [H, g_m] = 0` for all i, m.
fn fiber_multi(m: f64) -> Hamiltonian {
    let mut terms = vec![(
        Complex64::new(m, 0.0),
        vec![
            Operator::InnerBosonCreate(0),
            Operator::InnerBosonAnnihilate(0),
        ],
    )];
    for gm in 4..7u32 {
        // ½g_m² normal-ordered: ½a†² + a†a + ½a²
        terms.push((
            Complex64::new(0.5, 0.0),
            vec![
                Operator::InnerBosonCreate(gm),
                Operator::InnerBosonCreate(gm),
            ],
        ));
        terms.push((
            Complex64::new(1.0, 0.0),
            vec![
                Operator::InnerBosonCreate(gm),
                Operator::InnerBosonAnnihilate(gm),
            ],
        ));
        terms.push((
            Complex64::new(0.5, 0.0),
            vec![
                Operator::InnerBosonAnnihilate(gm),
                Operator::InnerBosonAnnihilate(gm),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// The 1D BRST derivative-variable fixing charge: `Ω = (g_0 − 2φ_1) c_0` —
/// fixes the promoted derivative variable g_0 (boson mode 2) to the actual
/// field derivative D_0 = 2φ_1 (mode 1), with ghost c_0 (fermion mode 0).
fn brst_1d() -> Hamiltonian {
    Hamiltonian {
        terms: vec![
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(2), Operator::InnerFermionAnnihilate(0)]),
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(2), Operator::InnerFermionAnnihilate(0)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonCreate(1), Operator::InnerFermionAnnihilate(0)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonAnnihilate(1), Operator::InnerFermionAnnihilate(0)]),
        ],
    }
}

/// The multi-level BRST derivative-variable fixing charge:
/// `Ω = Σ_{m=0}^{2} (g_m − 2(m+1)φ_{m+1}) c_m` — g_m on boson mode 4+m,
/// φ_{m+1} on mode 1+m, ghost c_m on fermion mode m.
fn brst_multi() -> Hamiltonian {
    let mut terms = Vec::new();
    for m in 0..3u32 {
        let gm = 4 + m;
        let pm = 1 + m;
        let coeff = -2.0 * (m as f64 + 1.0);
        terms.push((
            Complex64::new(1.0, 0.0),
            vec![Operator::InnerBosonCreate(gm), Operator::InnerFermionAnnihilate(m)],
        ));
        terms.push((
            Complex64::new(1.0, 0.0),
            vec![Operator::InnerBosonAnnihilate(gm), Operator::InnerFermionAnnihilate(m)],
        ));
        terms.push((
            Complex64::new(coeff, 0.0),
            vec![Operator::InnerBosonCreate(pm), Operator::InnerFermionAnnihilate(m)],
        ));
        terms.push((
            Complex64::new(coeff, 0.0),
            vec![Operator::InnerBosonAnnihilate(pm), Operator::InnerFermionAnnihilate(m)],
        ));
    }
    Hamiltonian { terms }
}

/// The constraint operator C_m = g_m − 2(m+1)φ_{m+1} as a Hamiltonian (1D:
/// promoted mode 2, content mode 1).
fn constraint_1d() -> Hamiltonian {
    Hamiltonian {
        terms: vec![
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(2)]),
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(2)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonCreate(1)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonAnnihilate(1)]),
        ],
    }
}

fn comm_norm(a: &Hamiltonian, b: &Hamiltonian, s: &QuantumState) -> f64 {
    let mut d = a.apply(&b.apply(s));
    d.scale_and_add(&b.apply(&a.apply(s)), Complex64::new(-1.0, 0.0));
    d.norm()
}

fn sirk_opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: true,
    }
}

// ── 1. 1D: the physical observables are consistent and calculable ──────────

#[test]
fn qg_starobinsky_derivative_variable_physical_observables_1d() {
    // The gauge condition is verified BY CONSTRUCTION: the physical initial
    // wave-function sets ⟨g_0⟩ = 2⟨φ_1⟩, and [H, C_0] = 0 makes it a constant
    // of the motion.  What is tested here is the REMAINING physics — the
    // scalaron field, the energy, the equation of motion — being consistent
    // and calculable while the condition holds.
    let m = 1.0;
    let h = fiber_1d(m);
    let brst = brst_1d();
    let c0 = constraint_1d();

    // ── By-construction algebraic facts (the gauge structure itself).
    for s in [bos_state(0, 1), bos_state(1, 1), bos_state(2, 1)] {
        let nrm = comm_norm(&h, &c0, &s);
        assert!(
            nrm < 1e-8,
            "[H, C_0] must vanish (constraint a constant of the motion), ‖[H,C_0]ψ‖ = {nrm:.3e}"
        );
    }
    let ghosted = ghost_state(
        {
            let mut inner = InnerBosonicState::vacuum();
            inner.modes.insert(2, 1);
            inner.modes.insert(1, 1);
            inner
        },
        0,
    );
    let twice = brst.apply(&brst.apply(&ghosted));
    assert!(
        twice.norm() < 1e-9,
        "Ω² must be nilpotent, ‖Ω²ψ‖ = {:.3e}",
        twice.norm()
    );
    let nrm = comm_norm(&h, &brst, &ghosted);
    assert!(
        nrm < 1e-8,
        "[H, Ω] must vanish (BRST-closed fiber), ‖[H,Ω]ψ‖ = {nrm:.3e}"
    );

    // ── Physical initial wave-function (gauge condition by construction):
    // ⟨φ_0⟩ = 1, ⟨φ_1⟩ = 1/3, ⟨g_0⟩ = 2/3 = 2⟨φ_1⟩.
    let psi0 = normalize(&product_state(&[(0, 1.0), (1, amp(1.0 / 3.0)), (2, amp(2.0 / 3.0))]));
    let u0_0 = field_expect(&psi0, 0);
    let u1_0 = field_expect(&psi0, 1);
    let g0_0 = field_expect(&psi0, 2);
    assert!((u0_0 - 1.0).abs() < 1e-9, "⟨φ_0⟩ = {u0_0} must be 1");
    assert!((u1_0 - 1.0 / 3.0).abs() < 1e-9, "⟨φ_1⟩ = {u1_0} must be 1/3");
    assert!(
        (g0_0 - 2.0 * u1_0).abs() < 1e-9,
        "gauge condition by construction: ⟨g_0⟩ = {g0_0} = 2⟨φ_1⟩ = {}",
        2.0 * u1_0
    );

    // ── Physical composite observables are CONSISTENT at t = 0: the
    // composite ⟨φ_0 g_0⟩ computed with the promoted variable equals
    // 2⟨φ_0 φ_1⟩ computed with the actual derivative operator D_0 = 2φ_1,
    // and the pointwise profile satisfies ⟨g(x)⟩ = ∂_x ⟨φ(x)⟩ everywhere.
    let ug0 = composite_expect(&psi0, &[0, 2]);
    let uu0 = composite_expect(&psi0, &[0, 1]);
    assert!(
        (ug0 - 2.0 * uu0).abs() < 1e-9,
        "⟨φ_0 g_0⟩ = {ug0} must equal ⟨φ_0 D_0⟩ = 2⟨φ_0 φ_1⟩ = {}",
        2.0 * uu0
    );
    // ⟨φ(x)⟩ = ⟨φ_0⟩ + 2⟨φ_1⟩x,  ⟨g(x)⟩ = ⟨g_0⟩,  ∂_x ⟨φ(x)⟩ = 2⟨φ_1⟩.
    for x in [-2.0, -0.7, 0.0, 0.5, 1.9] {
        let du_dx = 2.0 * u1_0;
        assert!(
            (g0_0 - du_dx).abs() < 1e-9,
            "⟨g(x)⟩ = {g0_0} must equal ∂_x⟨φ(x)⟩ = {du_dx} at x = {x}"
        );
        let phi_x = u0_0 + 2.0 * u1_0 * x;
        assert!(
            phi_x.abs() > 1e-3,
            "the field profile must be non-trivial at x = {x}, got ⟨φ⟩ = {phi_x}"
        );
    }

    // ── The physical observables are CALCULABLE and CONSISTENT under the
    // SIRK flow — and identical under the bare and the BRST-projected flow
    // (the gauge condition is what makes the two implementations agree).
    let opts = sirk_opts();
    let mut bare_traj = Vec::new();
    let mut proj_traj = Vec::new();
    for (label, use_brst) in [("bare flow", None), ("BRST-projected flow", Some(&brst))] {
        let mut psi = normalize(&product_state(&[(0, 1.0), (1, amp(1.0 / 3.0)), (2, amp(2.0 / 3.0))]));
        let e0 = energy_expect(&psi, &h);
        let mut t = 0.0;
        let mut traj = Vec::new();
        for dt in [0.025, 0.05] {
            psi = evolve_restarted(&h, &psi, dt, 3, 3, &best_device(), use_brst, &opts)
                .expect("SIRK restart");
            t += dt;
            traj.push((t, field_expect(&psi, 0), field_expect(&psi, 1), energy_expect(&psi, &h), psi.norm()));

            // The gauge condition holds at all times — to SOLVER accuracy
            // (the truncated restarted-Krylov flow, not the exact flow; the
            // exact flow conserves C_0 identically).  The convergence block
            // below shows it shrinks quadratically with dt.
            let u1_t = field_expect(&psi, 1);
            let g0_t = field_expect(&psi, 2);
            assert!(
                (g0_t - 2.0 * u1_t).abs() < 1e-2,
                "({label}, t={t}): gauge condition preserved to solver accuracy: \
                 ⟨g_0⟩ = {g0_t} vs 2⟨φ_1⟩ = {}",
                2.0 * u1_t
            );
            // The gradient content is frozen: the derivative value the
            // promoted variable is fixed to does not drift.
            assert!(
                (u1_t - u1_0).abs() < 1e-9,
                "({label}, t={t}): ⟨φ_1⟩ = {u1_t} must stay at its physical value {u1_0}"
            );
            // The energy (the physical observable) is CONSERVED — the
            // unitary solver keeps it to machine precision.
            let e_t = energy_expect(&psi, &h);
            assert!(
                (e_t - e0).abs() < 1e-8,
                "({label}, t={t}): ⟨H⟩ = {e_t} must be conserved (initial {e0})"
            );
            // CALCULABLE: the solver keeps the state normalized.
            assert!(
                (psi.norm() - 1.0).abs() < 1e-8,
                "({label}, t={t}): norm ‖ψ‖ = {} must be conserved",
                psi.norm()
            );
        }
        if label.starts_with("bare") {
            bare_traj = traj;
        } else {
            proj_traj = traj;
        }
    }
    // The gauge condition makes the bare and the gauge-fixed implementations
    // agree on ALL physical observables.
    for ((t1, ua, u1a, ea, na), (t2, ub, u1b, eb, nb)) in bare_traj.iter().zip(proj_traj.iter()) {
        assert!(
            (t1 - t2).abs() < 1e-12 && (ua - ub).abs() < 1e-9 && (u1a - u1b).abs() < 1e-9,
            "bare and gauge-fixed flows must give identical ⟨φ_0⟩, ⟨φ_1⟩ at t={t1}"
        );
        assert!(
            (ea - eb).abs() < 1e-9 && (na - nb).abs() < 1e-9,
            "bare and gauge-fixed flows must give identical ⟨H⟩, ‖ψ‖ at t={t1}"
        );
    }

    // ── The gauge-condition drift is a CONTROLLED solver artifact: one step
    // at dt and one at dt/2 from the same initial state show the violation
    // shrinking quadratically — the exact flow (dt → 0) satisfies the
    // condition identically, as guaranteed by [H, C_0] = 0.
    {
        let base = normalize(&product_state(&[(0, 1.0), (1, amp(1.0 / 3.0)), (2, amp(2.0 / 3.0))]));
        let drift = |dt: f64| {
            let p = evolve_restarted(&h, &base, dt, 3, 3, &best_device(), None, &opts).expect("SIRK restart");
            (field_expect(&p, 2) - 2.0 * field_expect(&p, 1)).abs()
        };
        let d1 = drift(0.05);
        let d2 = drift(0.025);
        assert!(d1 < 1e-3, "single-step gauge drift at dt=0.05 must be small, got {d1:.2e}");
        assert!(
            d2 < d1 / 4.0,
            "gauge drift must converge at least quadratically in dt: \
             {d1:.2e} (dt=0.05) vs {d2:.2e} (dt=0.025)"
        );
    }

    // ── The equation of motion is the massive Klein–Gordon oscillator: by
    // Ehrenfest, d⟨φ_0⟩/dt = ⟨i[H, φ_0]⟩ = m⟨π_0⟩, and the exact trajectory
    // is ⟨φ_0(t)⟩ = ⟨φ_0(0)⟩cos(mt) (the R² scalar mode oscillates at the
    // scalaron mass — the S34/S32 massive-dispersion content in the time
    // domain).  Verify the finite-difference rate matches the operator
    // prediction, and the SIRK trajectory tracks the cos(mt) oscillation.
    let e = 1e-4;
    let psi_pp = evolve_restarted(&h, &psi0, e, 3, 3, &best_device(), None, &opts).expect("SIRK restart");
    let fd = (field_expect(&psi_pp, 0) - u0_0) / e;
    let rhs = m * momentum_expect(&psi0, 0);
    assert!(
        (fd - rhs).abs() < 2e-2,
        "Ehrenfest: d⟨φ_0⟩/dt = {fd:.4} must equal m⟨π_0⟩ = {rhs:.4}"
    );

    // The cos(mt) trajectory under the SIRK flow: with ⟨π_0(0)⟩ = 0 and
    // m = 1, the scalaron field oscillates as ⟨φ_0(t)⟩ = ⟨φ_0(0)⟩cos(t) —
    // the R² scalar mode at its mass (the time-domain form of the massive
    // dispersion ω = √(k²+m²)).  Evolve in small steps (the restarted-Krylov
    // solver tracks short intervals nearly exactly; the trajectory is tight
    // through t ≈ 1, so verify there).
    for (t, expect) in [(0.25f64, 0.25f64.cos()), (0.5, 0.5f64.cos()), (0.75, 0.75f64.cos()), (1.0, 1.0f64.cos())] {
        let mut psi = psi0.clone();
        let mut tt = 0.0;
        let dt = 0.025;
        while tt < t - 1e-12 {
            let step = (t - tt).min(dt);
            psi = evolve_restarted(&h, &psi, step, 3, 3, &best_device(), None, &opts).expect("SIRK restart");
            tt += step;
        }
        let phi_t = field_expect(&psi, 0);
        assert!(
            (phi_t - expect).abs() < 5e-3,
            "massive Klein–Gordon oscillation: ⟨φ_0({t})⟩ = {phi_t} must be ≈ cos({t}) = {expect}"
        );
    }

    eprintln!(
        "qg_starobinsky_derivative_variable_physical_observables_1d: gauge condition by \
         construction (⟨g_0⟩ = 2⟨φ_1⟩ = 2/3); physical observables consistent (⟨φ_0g_0⟩ = \
         2⟨φ_0φ_1⟩, pointwise ⟨g(x)⟩ = ∂_x⟨φ(x)⟩, ⟨H⟩ conserved, bare = gauge-fixed flow) \
         and calculable (norm conserved); Ehrenfest d⟨φ_0⟩/dt = m⟨π_0⟩ and the massive \
         Klein–Gordon oscillation ⟨φ_0(t)⟩ = cos(mt) reproduced by SIRK"
    );
}

// ── 2. Multi-level: the full Hermite fiber carries a genuine polynomial ─────

#[test]
fn qg_starobinsky_derivative_variable_higher_hermite_modes() {
    // The FULL multi-level fiber: the field φ(x) = Σ_n φ_n H_n(x) with
    // content φ_1, φ_2, φ_3 and the promoted derivative variables fixed to
    // g_0 = 2φ_1 (D_0), g_1 = 4φ_2 (D_1), g_2 = 6φ_3 (D_2) — so the
    // derivative profile ⟨g(x)⟩ = Σ⟨g_m⟩H_m(x) = ∂_x⟨φ(x)⟩ is a GENUINE
    // polynomial (a quadratic in x, not the constant 2φ_1), exactly the
    // multi-level consistency the NS tests demand of u·∂_x u.  The
    // gauge-fixed Hamiltonian carries the products of the field derivatives
    // (½Σg_m²) — allowed precisely because the variables are fixed by BRST.
    let m = 1.0;
    let h = fiber_multi(m);
    let brst = brst_multi();

    // ── By-construction algebraic facts.
    // C_m = g_m − 2(m+1)φ_{m+1}: constants of the motion.
    for (m_idx, (gm, pm)) in [(4u32, 1u32), (5, 2), (6, 3)].into_iter().enumerate() {
        let cm: Hamiltonian = Hamiltonian {
            terms: vec![
                (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(gm)]),
                (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(gm)]),
                (
                    Complex64::new(-2.0 * (m_idx as f64 + 1.0), 0.0),
                    vec![Operator::InnerBosonCreate(pm)],
                ),
                (
                    Complex64::new(-2.0 * (m_idx as f64 + 1.0), 0.0),
                    vec![Operator::InnerBosonAnnihilate(pm)],
                ),
            ],
        };
        for s in [bos_state(pm, 1), bos_state(gm, 1), bos_state(0, 1)] {
            let nrm = comm_norm(&h, &cm, &s);
            assert!(
                nrm < 1e-8,
                "[H, C_{m_idx}] must vanish (constraint a constant of the motion), ‖[H,C]ψ‖ = {nrm:.3e}"
            );
        }
        // The promoted variable commutes with the Hamiltonian directly.
        let g_ham = field_hamiltonian(gm);
        let nrm = comm_norm(&h, &g_ham, &bos_state(gm, 1));
        assert!(
            nrm < 1e-8,
            "[H, g_{m_idx}] must vanish (promoted derivative variable frozen), ‖[H,g]ψ‖ = {nrm:.3e}"
        );
    }
    // Ω² = 0 and [H, Ω] = 0 on a fully ghosted multi-level probe.
    let ghosted = ghost_state(
        {
            let mut inner = InnerBosonicState::vacuum();
            for mode in [1u32, 2, 3, 4, 5, 6] {
                inner.modes.insert(mode, 1);
            }
            inner
        },
        0,
    );
    let twice = brst.apply(&brst.apply(&ghosted));
    assert!(
        twice.norm() < 1e-9,
        "Ω² must be nilpotent (multi-level), ‖Ω²ψ‖ = {:.3e}",
        twice.norm()
    );
    let nrm = comm_norm(&h, &brst, &ghosted);
    assert!(
        nrm < 1e-8,
        "[H, Ω] must vanish (multi-level BRST-closed fiber), ‖[H,Ω]ψ‖ = {nrm:.3e}"
    );

    // ── Physical initial state (gauge condition by construction):
    // ⟨φ_0⟩ = 1, ⟨φ_1⟩ = 1/3, ⟨φ_2⟩ = 1/6, ⟨φ_3⟩ = 1/12,
    // ⟨g_0⟩ = 2/3, ⟨g_1⟩ = 4·(1/6) = 2/3, ⟨g_2⟩ = 6·(1/12) = 1/2.
    let parts = [
        (0u32, 1.0f64),
        (1, amp(1.0 / 3.0)),
        (2, amp(1.0 / 6.0)),
        (3, amp(1.0 / 12.0)),
        (4, amp(2.0 / 3.0)),
        (5, amp(2.0 / 3.0)),
        (6, amp(1.0 / 2.0)),
    ];
    let psi0 = normalize(&product_state(&parts));
    let f0 = field_expect(&psi0, 0);
    let f1 = field_expect(&psi0, 1);
    let f2 = field_expect(&psi0, 2);
    let f3 = field_expect(&psi0, 3);
    let g0 = field_expect(&psi0, 4);
    let g1 = field_expect(&psi0, 5);
    let g2 = field_expect(&psi0, 6);
    assert!((f0 - 1.0).abs() < 1e-9 && (f1 - 1.0 / 3.0).abs() < 1e-9, "field values");
    assert!(
        (g0 - 2.0 * f1).abs() < 1e-9 && (g1 - 4.0 * f2).abs() < 1e-9 && (g2 - 6.0 * f3).abs() < 1e-9,
        "gauge condition by construction: ⟨g_m⟩ = 2(m+1)⟨φ_(m+1)⟩ (got {g0}, {g1}, {g2})"
    );

    // ── The GENUINE polynomial profile: the derivative field g(x) =
    // Σ⟨g_m⟩H_m(x) equals ∂_x⟨φ(x)⟩ = Σ 2(m+1)⟨φ_{m+1}⟩H_m(x) at every x —
    // a quadratic in x, not the constant 2φ_1 (the multi-level generalization
    // of the 1D pointwise identity).
    for x in [-1.4, -0.6, 0.0, 0.3, 1.1, 2.0] {
        let g_x = g0 * hermite(0, x) + g1 * hermite(1, x) + g2 * hermite(2, x);
        let du_dx = 2.0 * f1 * hermite(0, x) + 4.0 * f2 * hermite(1, x) + 6.0 * f3 * hermite(2, x);
        assert!(
            (g_x - du_dx).abs() < 1e-9,
            "⟨g(x)⟩ = {g_x} must equal ∂_x⟨φ(x)⟩ = {du_dx} at x = {x}"
        );
        // The profile is genuinely polynomial: the second derivative
        // ∂_x²⟨φ(x)⟩ = ∂_x⟨g(x)⟩ = 8⟨φ_2⟩ + 48⟨φ_3⟩x is a non-constant linear
        // function of x — not the flat 1D case (where it is identically 0),
        // and not zero.
        let d2 = 8.0 * f2 + 48.0 * f3 * x;
        assert!(
            d2.abs() > 1e-3,
            "the derivative profile must be a genuine polynomial (curvature {d2} at x = {x})"
        );
        // The curvature varies with x — the profile is genuinely quadratic,
        // not the flat 1D gradient (skip the x = 0 sample where the two
        // coincide by construction).
        if x.abs() > 1e-9 {
            let d2_at_0 = 8.0 * f2;
            assert!(
                (d2 - d2_at_0).abs() > 1e-3,
                "the curvature must vary with x: {d2} at x = {x} vs {d2_at_0} at x = 0"
            );
        }
        // And the field itself is the genuine cubic the derivatives come from.
        let phi_x = f0 * hermite(0, x) + f1 * hermite(1, x) + f2 * hermite(2, x) + f3 * hermite(3, x);
        assert!(
            phi_x.abs() > 1e-3,
            "the field profile must be non-trivial at x = {x}, got ⟨φ⟩ = {phi_x}"
        );
    }

    // ── Promoted composite observables match the field-only values: the
    // products of the field derivatives ½Σg_m² enter the Hamiltonian through
    // the promoted variables, and the composites ⟨φ_m g_m⟩ reproduce the
    // actual derivative composites 2(m+1)⟨φ_m φ_{m+1}⟩ — the multi-level
    // analogue of ⟨u·∂_x u⟩ being computed from the promoted derivatives.
    for (m_idx, (fm, gm)) in [(0u32, 4u32), (1, 5), (2, 6)].into_iter().enumerate() {
        let prom = composite_expect(&psi0, &[fm, gm]);
        let field_only = 2.0 * (m_idx as f64 + 1.0) * composite_expect(&psi0, &[fm, 1 + m_idx as u32]);
        let next = 1 + m_idx as u32;
        assert!(
            (prom - field_only).abs() < 1e-9,
            "⟨φ_{fm} g_{m_idx}⟩ = {prom} must equal 2(m+1)⟨φ_{fm} φ_{next}⟩ = {field_only}"
        );
    }

    // ── Energy conservation, bare = gauge-fixed flow, gauge drift
    // convergence — the multi-level fiber under the SIRK flow.
    let opts = sirk_opts();
    let mut bare_traj = Vec::new();
    let mut proj_traj = Vec::new();
    for (label, use_brst) in [("bare flow", None), ("BRST-projected flow", Some(&brst))] {
        let mut psi = normalize(&product_state(&parts));
        let e0 = energy_expect(&psi, &h);
        let mut t = 0.0;
        let mut traj = Vec::new();
        for dt in [0.05] {
            psi = evolve_restarted(&h, &psi, dt, 3, 3, &best_device(), use_brst, &opts)
                .expect("SIRK restart");
            t += dt;
            traj.push((t, field_expect(&psi, 0), field_expect(&psi, 1), field_expect(&psi, 2), field_expect(&psi, 3), energy_expect(&psi, &h), psi.norm()));
            // All three gauge conditions preserved to solver accuracy.
            for (gm, pm, coeff) in [(4u32, 1u32, 2.0f64), (5, 2, 4.0), (6, 3, 6.0)] {
                let g_t = field_expect(&psi, gm);
                let p_t = field_expect(&psi, pm);
                let gi = gm - 4;
                assert!(
                    (g_t - coeff * p_t).abs() < 2e-2,
                    "({label}, t={t}): gauge condition g_{gi} preserved to solver accuracy: \
                     ⟨g⟩ = {g_t} vs {coeff}⟨φ⟩ = {}",
                    coeff * p_t
                );
            }
            // The derivative content is frozen.
            assert!(
                (field_expect(&psi, 1) - f1).abs() < 1e-9
                    && (field_expect(&psi, 2) - f2).abs() < 1e-9
                    && (field_expect(&psi, 3) - f3).abs() < 1e-9,
                "({label}, t={t}): derivative content must stay frozen"
            );
            // Energy + norm conserved.
            let e_t = energy_expect(&psi, &h);
            assert!(
                (e_t - e0).abs() < 1e-8,
                "({label}, t={t}): ⟨H⟩ = {e_t} must be conserved (initial {e0})"
            );
            assert!((psi.norm() - 1.0).abs() < 1e-8, "({label}, t={t}): norm conserved");
        }
        if label.starts_with("bare") {
            bare_traj = traj;
        } else {
            proj_traj = traj;
        }
    }
    for ((t1, ua, u1a, u2a, u3a, ea, na), (t2, ub, u1b, u2b, u3b, eb, nb)) in
        bare_traj.iter().zip(proj_traj.iter())
    {
        assert!(
            (t1 - t2).abs() < 1e-12
                && (ua - ub).abs() < 1e-9
                && (u1a - u1b).abs() < 1e-9
                && (u2a - u2b).abs() < 1e-9
                && (u3a - u3b).abs() < 1e-9,
            "bare and gauge-fixed flows must give identical ⟨φ_0..3⟩ at t={t1}"
        );
        assert!((ea - eb).abs() < 1e-9 && (na - nb).abs() < 1e-9, "identical ⟨H⟩, ‖ψ‖ at t={t1}");
    }

    // Gauge drift convergence (all three conditions).
    {
        let base = normalize(&product_state(&parts));
        let drift = |dt: f64| {
            let p = evolve_restarted(&h, &base, dt, 3, 3, &best_device(), None, &opts).expect("SIRK restart");
            let mut worst = 0.0f64;
            for (gm, pm, coeff) in [(4u32, 1u32, 2.0f64), (5, 2, 4.0), (6, 3, 6.0)] {
                worst = worst.max((field_expect(&p, gm) - coeff * field_expect(&p, pm)).abs());
            }
            worst
        };
        let d1 = drift(0.05);
        let d2 = drift(0.025);
        assert!(d1 < 5e-3, "single-step multi-level gauge drift at dt=0.05 must be small, got {d1:.2e}");
        assert!(
            d2 < d1 / 4.0,
            "multi-level gauge drift must converge at least quadratically in dt: \
             {d1:.2e} (dt=0.05) vs {d2:.2e} (dt=0.025)"
        );
    }

    eprintln!(
        "qg_starobinsky_derivative_variable_higher_hermite_modes: full fiber φ_1,φ_2,φ_3 — \
         the derivative profile ⟨g(x)⟩ = Σ⟨g_m⟩H_m(x) = ∂_x⟨φ(x)⟩ is a genuine polynomial; \
         promoted composites ⟨φ_m g_m⟩ = 2(m+1)⟨φ_m φ_(m+1)⟩; ⟨H⟩ conserved, bare = gauge-fixed, \
         drift ∝ dt²"
    );
}

// ── 3. Unphysical data: detected, inconsistent, and not self-fixed ─────────

#[test]
fn qg_starobinsky_derivative_variable_unphysical_data_inconsistent() {
    // Unphysical initial data: ⟨φ_1⟩ = 1/3 (so the PHYSICAL value of g_0 is
    // 2/3), but the promoted variable is set to ⟨g_0⟩ = 1 ≠ 2/3.  The gauge
    // condition is violated: ⟨C_0⟩ = 1/3 ≠ 0, the state carries Ω-content,
    // and — crucially — the physical observable is INCONSISTENT: the dynamics
    // the Hamiltonian generates (4⟨φ_0 g_0⟩, the promoted gradient composite)
    // does NOT match the value read off the field alone (8⟨φ_0 φ_1⟩).  The
    // bare flow conserves the violation (no self-fixing) — gauge fixing is
    // genuinely required.
    let m = 1.0;
    let h = fiber_1d(m);
    let brst = brst_1d();

    let psi = normalize(&product_state(&[(0, 1.0), (1, amp(1.0 / 3.0)), (2, 1.0)]));

    // The constraint violation is detected.
    let u1 = field_expect(&psi, 1);
    let g0 = field_expect(&psi, 2);
    let c0_expect = g0 - 2.0 * u1;
    assert!(
        c0_expect.abs() > 1e-2,
        "the unphysical data must violate the constraint: ⟨C_0⟩ = {c0_expect}"
    );

    // The physical observable is INCONSISTENT: the promoted composite
    // 4⟨φ_0 g_0⟩ differs from the field-only value 8⟨φ_0⟩⟨φ_1⟩.
    let ug0 = composite_expect(&psi, &[0, 2]);
    let uu0 = composite_expect(&psi, &[0, 1]);
    let actual = 4.0 * ug0;
    let predicted = 8.0 * uu0;
    assert!(
        (actual - predicted).abs() > 1e-2,
        "unphysical data must make the observable inconsistent: promoted composite \
         4⟨φ_0g_0⟩ = {actual} vs field-only value 8⟨φ_0φ_1⟩ = {predicted}"
    );

    // The violation carries Ω-content (ghost-lifted state not annihilated).
    let ghosted = ghost_state(
        {
            let mut inner = InnerBosonicState::vacuum();
            inner.modes.insert(2, 1);
            inner.modes.insert(1, 1);
            inner
        },
        0,
    );
    assert!(
        brst.apply(&ghosted).norm() > 1e-6,
        "the unphysical g_0/φ_1 content must carry Ω-content"
    );
    // Nilpotency still holds on the unphysical data (first-class constraint).
    let twice = brst.apply(&brst.apply(&ghosted));
    assert!(
        twice.norm() < 1e-9,
        "Ω² must be nilpotent even on unphysical data, ‖Ω²ψ‖ = {:.3e}",
        twice.norm()
    );

    // The bare flow conserves the violation (does not spontaneously fix it).
    let opts = sirk_opts();
    let psi_t = evolve_restarted(&h, &psi, 0.05, 3, 3, &best_device(), None, &opts)
        .expect("bare SIRK restart");
    let u1_t = field_expect(&psi_t, 1);
    let g0_t = field_expect(&psi_t, 2);
    let c0_t = g0_t - 2.0 * u1_t;
    assert!(
        (c0_t - c0_expect).abs() < 1e-3,
        "the bare flow must conserve the violation to solver accuracy: ⟨C_0⟩ \
         {c0_expect} → {c0_t} (gauge fixing is required)"
    );

    eprintln!(
        "qg_starobinsky_derivative_variable_unphysical_data_inconsistent: ⟨C_0⟩ = {c0_expect:.6} ≠ 0 \
         detected, observable inconsistent (4⟨φ_0g_0⟩ = {actual:.4} ≠ 8⟨φ_0φ_1⟩ = {predicted:.4}), \
         Ω-content ≠ 0, Ω² = 0, bare flow conserves the violation (no self-fixing)"
    );
}
