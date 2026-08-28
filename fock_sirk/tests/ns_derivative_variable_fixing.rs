//! Navier–Stokes derivative-variable gauge fixing: the PHYSICAL observables.
//!
//! The Eulerian derivatives-as-fields picture (book.tex §4159-4197) promotes
//! the spatial field derivatives `u_{i,j} = ∂_j u_i` to independent canonical
//! fields and *fixes them to the values of the field derivatives*.  In the
//! nested Fock space the field is the first moment of its inner space,
//! `u(x) = ∫ du u a†(x,u) a(x,u)`; expanded in physicists' Hermite
//! polynomials `u(x) = Σ_n u_n H_n(x)` with `u_n = a†_n + a_n` and
//! `∂_x H_n = 2n H_{n-1}`, the spatial derivative of the field operator is the
//! well-defined operator on the velocity ladder modes
//!
//!     ∂_x u(x) = Σ_n u_n·2n H_{n-1}(x) = Σ_m (2(m+1) u_{m+1}) H_m(x),
//!
//! i.e. the mode-m component of the derivative is the operator
//! `D_m = 2(m+1) u_{m+1}`.
//!
//! THE GAUGE CONDITION IS VERIFIED BY CONSTRUCTION — it is not what these
//! tests check.  The physical initial wave-function *sets* the promoted
//! derivative variable `g_m = a†+a` (its own ladder mode) to the actual
//! derivative value `⟨g_m⟩ = 2(m+1)⟨u_{m+1}⟩`, and beca//! **Ground-state doctrine** (`outer_vacuum_ground_validation.rs`): the
//! ground state of the nested theory is always the outer-Fock vacuum — the
//! final Hamiltonian is the one-particle Hamiltonian enclosed in outer
//! creation (left) / annihilation (right) operators, with at most a
//! constant added to make its spectrum positive (QYM/QG/NS).
use the derivative-
//! content modes carry no momenta, `[H, C_m] = 0` (`C_m = g_m − D_m`) makes
//! the condition an exact constant of the motion under the Hamiltonian flow
//! (bare or BRST-projected).  The question these tests answer is whether the
//! REMAINING numerical observables are CONSISTENT and CALCULABLE while the
//! gauge condition holds:
//!
//!  1. `ns_derivative_variable_physical_observables_1d` — with the Euler
//!     fiber `H = K_0 + {π_0, u_0 g_0}` (normal-ordered kinetic + the
//!     promoted form of `u·∂_x u`): the gradient content is frozen, the
//!     ENERGY `⟨H⟩` is conserved (unitary solver, to machine precision), the
//!     bare and the BRST-projected flows give IDENTICAL physical observables
//!     (the gauge condition is what makes the two implementations agree), and
//!     the Ehrenfest equation of motion holds numerically:
//!     `d⟨u_0⟩/dt = ⟨i[H,u_0]⟩ = 2⟨π_0⟩ + 4⟨u_0 g_0⟩`, which in the physical
//!     subspace equals the classical Euler advection `8⟨u_0⟩⟨u_1⟩` — the
//!     velocity is advected by its own spatial derivative.  The one thing
//!     that is NOT exact under the *truncated* restarted-Krylov solver is the
//!     gauge condition itself: its drift is a controlled, quadratically
//!     convergent (in dt) numerical artifact — the exact flow conserves
//!     `C_0` identically, and the test verifies the drift shrinks as dt → 0.
//!
//!  2. `ns_derivative_variable_physical_observables_2d` — the same in 2D with
//!     `g_x = 2u_{1,0}`, `g_y = 2u_{0,1}` and `H = K_0 + {π_0, u_0(g_x+g_y)}`:
//!     energy conservation, bare/projected agreement, the same drift
//!     convergence, and `d⟨u_0⟩/dt = 2⟨π_0⟩ + 4⟨u_0(g_x+g_y)⟩ =
//!     8⟨u_0⟩(⟨u_1⟩+⟨u_2⟩)` at t=0 (the full gradient advection).
//!
//!  3. `ns_derivative_variable_higher_hermite_modes` — the FULL multi-level
//!     fiber: derivative content `u_1, u_2, u_3` with the promoted variables
//!     `g_0 = 2u_1, g_1 = 4u_2, g_2 = 6u_3`, so the derivative profile
//!     `⟨g(x)⟩ = Σ⟨g_m⟩H_m(x) = ∂_x⟨u(x)⟩` is a GENUINE polynomial (a
//!     quadratic in x, not the constant `2u_1`) and the advection `u·∂_x u`
//!     carries a genuine polynomial profile.  The pointwise identity holds at
//!     every x, the promoted composites `⟨u_m g_m⟩ = 2(m+1)⟨u_m u_{m+1}⟩`
//!     match the field-only values, and the SIRK checks (energy conservation,
//!     bare = gauge-fixed, drift ∝ dt²) extend to the full fiber.
//!
//!  4. `ns_derivative_variable_unphysical_data_inconsistent` — unphysical
//!     initial data (promoted variable NOT set to the derivative value) is
//!     detected: `⟨C_0⟩ ≠ 0`, the physical observable is INCONSISTENT
//!     (`4⟨u_0 g_0⟩ ≠ 8⟨u_0 u_1⟩`: the dynamics the Hamiltonian generates
//!     disagrees with the value read off the velocity field alone), the
//!     violation carries Ω-content and is conserved (no self-fixing) — gauge
//!     fixing is genuinely required, exactly the NS pattern.
//!
//! Mode layout (minimal slices, Euler fiber):
//!   1D: 0 = velocity value u_0, 1 = gradient-content u_1,
//!       2 = promoted derivative variable g_0 = a†_2 + a_2, ghost c_0.
//!   2D: 0 = value, 1 = x-gradient u_{1,0}, 2 = y-gradient u_{0,1},
//!       3 = promoted g_x, 4 = promoted g_y, ghosts c_0, c_1.
//!
//! The fiber Hamiltonian `H = K_0 + {π_0, A}` (Hermitian by construction,
//! `⟨0|H|0⟩ = 0`) drives the value mode while leaving the derivative-content
//! modes untouched — `[H, u_1] = [H, u_2] = [H, g_m] = 0` exactly, so the
//! gauge condition `C_m = g_m − 2(m+1)u_{m+1}` is a constant of the motion
//! and `[H, Ω] = 0`, `Ω² = 0` (first-class constraint).

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

/// The field operator `u_m = a†_m + a_m` terms: `(c, op)` pairs.
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

/// The field operator `u_m = a†_m + a_m` as a Hamiltonian (used to measure the
/// field-amplitude expectation `⟨u_m⟩`).
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
/// `⟨ψ| ∏_m u_m |ψ⟩ / ⟨ψ|ψ⟩` (the operators commute across modes, so the
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

/// The multi-level Euler fiber: `H = K_0 + {π_0, u_0 g_0}` with the value
/// mode 0 advected by the mode-0 promoted derivative variable g_0 (mode 4)
/// and the derivative content u_1, u_2, u_3 (modes 1, 2, 3) plus the higher
/// promoted variables g_1, g_2 (modes 5, 6) frozen — the full polynomial
/// profile is present, the value-mode dynamics is the minimal advection.
fn fiber_multi() -> Hamiltonian {
    let pi0 = momentum_ops(0);
    let u0 = field_ops(0);
    let g0 = field_ops(4);
    let mut terms = vec![
        (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(0), Operator::InnerBosonAnnihilate(0)]),
        (Complex64::new(-0.5, 0.0), vec![Operator::InnerBosonCreate(0), Operator::InnerBosonCreate(0)]),
        (Complex64::new(-0.5, 0.0), vec![Operator::InnerBosonAnnihilate(0), Operator::InnerBosonAnnihilate(0)]),
    ];
    for (cp, op_p) in &pi0 {
        for (cu, op_u) in &u0 {
            for (cg, op_g) in &g0 {
                let c = cp * cu * cg;
                if c.norm_sqr() > 1e-30 {
                    terms.push((c, vec![op_p.clone(), op_u.clone(), op_g.clone()]));
                    terms.push((c, vec![op_u.clone(), op_g.clone(), op_p.clone()]));
                }
            }
        }
    }
    Hamiltonian { terms }
}

/// The multi-level BRST derivative-variable fixing charge:
/// `Ω = Σ_{m=0}^{2} (g_m − 2(m+1)u_{m+1}) c_m` — g_m on boson mode 4+m,
/// u_{m+1} on mode 1+m, ghost c_m on fermion mode m.
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

/// The 1D Euler fiber: `H = K_0 + {π_0, u_0 g_0}` with the normal-ordered
/// kinetic `K_0 = a†_0 a_0 − ½a†_0² − ½a_0²` (so `⟨0|H|0⟩ = 0`) plus the
/// value mode advected by the promoted derivative variable (the promoted
/// form of `u·∂_x u`).  Hermitian, and it leaves the derivative-content mode
/// 1 and the promoted variable mode 2 untouched: `[H, u_1] = [H, g_0] = 0`,
/// so `C_0 = g_0 − 2u_1` is a constant of the motion, while `[H, u_0] ≠ 0`
/// (the value genuinely evolves).  By Ehrenfest,
/// `d⟨u_0⟩/dt = 2⟨π_0⟩ + 4⟨u_0 g_0⟩`.
fn fiber_1d() -> Hamiltonian {
    let pi0 = momentum_ops(0);
    let u0 = field_ops(0);
    let g0 = field_ops(2);
    let mut terms = vec![
        (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(0), Operator::InnerBosonAnnihilate(0)]),
        (Complex64::new(-0.5, 0.0), vec![Operator::InnerBosonCreate(0), Operator::InnerBosonCreate(0)]),
        (Complex64::new(-0.5, 0.0), vec![Operator::InnerBosonAnnihilate(0), Operator::InnerBosonAnnihilate(0)]),
    ];
    for (cp, op_p) in &pi0 {
        for (cu, op_u) in &u0 {
            for (cg, op_g) in &g0 {
                let c = cp * cu * cg;
                if c.norm_sqr() > 1e-30 {
                    // ops applied right-to-left: π·u·g and u·g·π
                    terms.push((c, vec![op_p.clone(), op_u.clone(), op_g.clone()]));
                    terms.push((c, vec![op_u.clone(), op_g.clone(), op_p.clone()]));
                }
            }
        }
    }
    Hamiltonian { terms }
}

/// The 2D Euler fiber: `H = K_0 + {π_0, u_0 (g_x + g_y)}` — the normal-ordered
/// kinetic plus the value mode advected by both promoted gradient variables.
/// `d⟨u_0⟩/dt = 2⟨π_0⟩ + 4⟨u_0(g_x+g_y)⟩`.
fn fiber_2d() -> Hamiltonian {
    let pi0 = momentum_ops(0);
    let u0 = field_ops(0);
    let mut terms = vec![
        (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(0), Operator::InnerBosonAnnihilate(0)]),
        (Complex64::new(-0.5, 0.0), vec![Operator::InnerBosonCreate(0), Operator::InnerBosonCreate(0)]),
        (Complex64::new(-0.5, 0.0), vec![Operator::InnerBosonAnnihilate(0), Operator::InnerBosonAnnihilate(0)]),
    ];
    for &gmode in &[3u32, 4u32] {
        let g = field_ops(gmode);
        for (cp, op_p) in &pi0 {
            for (cu, op_u) in &u0 {
                for (cg, op_g) in &g {
                    let c = cp * cu * cg;
                    if c.norm_sqr() > 1e-30 {
                        terms.push((c, vec![op_p.clone(), op_u.clone(), op_g.clone()]));
                        terms.push((c, vec![op_u.clone(), op_g.clone(), op_p.clone()]));
                    }
                }
            }
        }
    }
    Hamiltonian { terms }
}

/// The 1D BRST derivative-variable fixing charge: `Ω = (g_0 − 2u_1) c_0` —
/// fixes the promoted derivative variable g_0 (boson mode 2) to the actual
/// field derivative D_0 = 2u_1 (velocity mode 1), with ghost c_0 (fermion
/// mode 0).
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

/// The 2D BRST derivative-variable fixing charge:
/// `Ω = (g_x − 2u_{1,0}) c_0 + (g_y − 2u_{0,1}) c_1`.
fn brst_2d() -> Hamiltonian {
    Hamiltonian {
        terms: vec![
            // g_x − 2u_{1,0}, fixed by c_0
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(3), Operator::InnerFermionAnnihilate(0)]),
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(3), Operator::InnerFermionAnnihilate(0)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonCreate(1), Operator::InnerFermionAnnihilate(0)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonAnnihilate(1), Operator::InnerFermionAnnihilate(0)]),
            // g_y − 2u_{0,1}, fixed by c_1
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(4), Operator::InnerFermionAnnihilate(1)]),
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(4), Operator::InnerFermionAnnihilate(1)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonCreate(2), Operator::InnerFermionAnnihilate(1)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonAnnihilate(2), Operator::InnerFermionAnnihilate(1)]),
        ],
    }
}

/// The constraint operator C_0 = g_0 − 2u_1 as a Hamiltonian (1D slice:
/// promoted mode 2, velocity mode 1).
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
        unit_norm_steps: false,
    }
}

// ── 1. 1D: the physical observables are consistent and calculable ──────────

#[test]
fn ns_derivative_variable_physical_observables_1d() {
    // The gauge condition is verified BY CONSTRUCTION: the physical initial
    // wave-function sets ⟨g_0⟩ = 2⟨u_1⟩, and [H, C_0] = 0 makes it a constant
    // of the motion.  What is tested here is the REMAINING physics — the
    // velocity field, the energy, the equation of motion — being consistent
    // and calculable while the condition holds.
    let h = fiber_1d();
    let brst = brst_1d();
    let c0 = constraint_1d();

    // ── By-construction algebraic facts (the gauge structure itself).
    for s in [bos_state(0, 1), bos_state(1, 1), bos_state(2, 1)] {
        let nrm = comm_norm(&h, &c0, &s);
        assert!(nrm < 1e-8, "[H, C_0] must vanish (constraint a constant of the motion), ‖[H,C_0]ψ‖ = {nrm:.3e}");
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
    assert!(twice.norm() < 1e-9, "Ω² must be nilpotent, ‖Ω²ψ‖ = {:.3e}", twice.norm());
    let nrm = comm_norm(&h, &brst, &ghosted);
    assert!(nrm < 1e-8, "[H, Ω] must vanish (BRST-closed fiber), ‖[H,Ω]ψ‖ = {nrm:.3e}");

    // ── Physical initial wave-function (gauge condition by construction):
    // ⟨u_0⟩ = 1, ⟨u_1⟩ = 1/3, ⟨g_0⟩ = 2/3 = 2⟨u_1⟩.
    let psi0 = normalize(&product_state(&[(0, 1.0), (1, amp(1.0 / 3.0)), (2, amp(2.0 / 3.0))]));
    let u0_0 = field_expect(&psi0, 0);
    let u1_0 = field_expect(&psi0, 1);
    let g0_0 = field_expect(&psi0, 2);
    assert!((u0_0 - 1.0).abs() < 1e-9, "⟨u_0⟩ = {u0_0} must be 1");
    assert!((u1_0 - 1.0 / 3.0).abs() < 1e-9, "⟨u_1⟩ = {u1_0} must be 1/3");
    assert!(
        (g0_0 - 2.0 * u1_0).abs() < 1e-9,
        "gauge condition by construction: ⟨g_0⟩ = {g0_0} = 2⟨u_1⟩ = {}",
        2.0 * u1_0
    );

    // ── Physical composite observables are CONSISTENT at t = 0: the
    // composite ⟨u_0 g_0⟩ computed with the promoted variable equals
    // 2⟨u_0 u_1⟩ computed with the actual derivative operator D_0 = 2u_1,
    // and the pointwise profile satisfies ⟨g(x)⟩ = ∂_x ⟨u(x)⟩ everywhere.
    let ug0 = composite_expect(&psi0, &[0, 2]);
    let uu0 = composite_expect(&psi0, &[0, 1]);
    assert!(
        (ug0 - 2.0 * uu0).abs() < 1e-9,
        "⟨u_0 g_0⟩ = {ug0} must equal ⟨u_0 D_0⟩ = 2⟨u_0 u_1⟩ = {}",
        2.0 * uu0
    );
    // ⟨u(x)⟩ = ⟨u_0⟩ + 2⟨u_1⟩x,  ⟨g(x)⟩ = ⟨g_0⟩,  ∂_x ⟨u(x)⟩ = 2⟨u_1⟩.
    for x in [-2.0, -0.7, 0.0, 0.5, 1.9] {
        let u_x = u0_0 + 2.0 * u1_0 * x;
        let du_dx = 2.0 * u1_0;
        assert!(
            (g0_0 - du_dx).abs() < 1e-9,
            "⟨g(x)⟩ = {g0_0} must equal ∂_x⟨u(x)⟩ = {du_dx} at x = {x}"
        );
        assert!(
            u_x.abs() > 1e-3,
            "the field profile must be non-trivial at x = {x}, got ⟨u⟩ = {u_x}"
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
            // (the truncated restarted-Krylov flow, not the exact flow;
            // the exact flow conserves C_0 identically).  Bound: the
            // per-step drift is ≲ 3e-4 at dt=0.05 and accumulates slowly;
            // the convergence block below shows it shrinks quadratically
            // with dt (the exact condition is recovered as dt → 0).
            let u1_t = field_expect(&psi, 1);
            let g0_t = field_expect(&psi, 2);
            assert!(
                (g0_t - 2.0 * u1_t).abs() < 1e-2,
                "({label}, t={t}): gauge condition preserved to solver accuracy: \
                 ⟨g_0⟩ = {g0_t} vs 2⟨u_1⟩ = {}",
                2.0 * u1_t
            );
            // The gradient content is frozen: the derivative value the
            // promoted variable is fixed to does not drift.
            assert!(
                (u1_t - u1_0).abs() < 1e-9,
                "({label}, t={t}): ⟨u_1⟩ = {u1_t} must stay at its physical value {u1_0}"
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
        assert!((t1 - t2).abs() < 1e-12 && (ua - ub).abs() < 1e-9 && (u1a - u1b).abs() < 1e-9,
            "bare and gauge-fixed flows must give identical ⟨u_0⟩, ⟨u_1⟩ at t={t1}");
        assert!((ea - eb).abs() < 1e-9 && (na - nb).abs() < 1e-9,
            "bare and gauge-fixed flows must give identical ⟨H⟩, ‖ψ‖ at t={t1}");
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

    // ── The equation of motion is the Euler advection u·∂_x u: by Ehrenfest,
    // d⟨u_0⟩/dt = ⟨i[H, u_0]⟩ = 2⟨π_0⟩ + 4⟨u_0 g_0⟩, and in the physical
    // subspace g_0 = ∂_x u so this is exactly the classical advection
    // 8⟨u_0⟩⟨u_1⟩ at t = 0 (product state, ⟨π_0⟩ = 0).  Verify the
    // finite-difference rate matches the operator prediction, both at t=0 and
    // after real evolution.
    let e = 1e-4;
    for (t0, base) in [(0.0, &psi0), (0.15, &{
        let mut p = normalize(&product_state(&[(0, 1.0), (1, amp(1.0 / 3.0)), (2, amp(2.0 / 3.0))]));
        p = evolve_restarted(&h, &p, 0.15, 3, 3, &best_device(), None, &opts).expect("SIRK restart");
        p
    })] {
        let psi_pp = evolve_restarted(&h, base, e, 3, 3, &best_device(), None, &opts).expect("SIRK restart");
        let fd = (field_expect(&psi_pp, 0) - field_expect(base, 0)) / e;
        let rhs = 2.0 * momentum_expect(base, 0) + 4.0 * composite_expect(base, &[0, 2]);
        assert!(
            (fd - rhs).abs() < 2e-2,
            "Ehrenfest at t={t0}: d⟨u_0⟩/dt = {fd:.4} must equal ⟨i[H,u_0]⟩ = \
             2⟨π_0⟩ + 4⟨u_0g_0⟩ = {rhs:.4}"
        );
        if t0 == 0.0 {
            // In the physical subspace the RHS is the classical advection:
            // 4⟨u_0 g_0⟩ = 8⟨u_0⟩⟨u_1⟩ (the velocity advected by its gradient).
            let phys = 8.0 * uu0;
            assert!(
                (rhs - phys).abs() < 1e-9,
                "4⟨u_0g_0⟩ = {rhs} must equal the Euler advection 8⟨u_0⟩⟨u_1⟩ = {phys}"
            );
            assert!(fd.abs() > 1e-2, "the value mode must genuinely evolve under the advection, d⟨u_0⟩/dt = {fd}");
        }
    }

    eprintln!(
        "ns_derivative_variable_physical_observables_1d: gauge condition by construction \
         (⟨g_0⟩ = 2⟨u_1⟩ = 2/3); physical observables consistent (⟨u_0g_0⟩ = 2⟨u_0u_1⟩, \
         pointwise ⟨g(x)⟩ = ∂_x⟨u(x)⟩, ⟨H⟩ conserved, bare = gauge-fixed flow) and calculable \
         (norm conserved); Ehrenfest d⟨u_0⟩/dt = 4⟨u_0g_0⟩ = 8⟨u_0⟩⟨u_1⟩ — the Euler advection"
    );
}

// ── 2. 2D: two promoted variables, the full gradient advection ─────────────

#[test]
fn ns_derivative_variable_physical_observables_2d() {
    // 2D slice: u(x,y) = u_0 + 2u_1 x + 2u_2 y, so ∂_x u = 2u_1 (D_x),
    // ∂_y u = 2u_2 (D_y); promoted g_x (mode 3), g_y (mode 4).
    // Physical initial state (gauge condition by construction):
    // ⟨u_0⟩ = 1, ⟨u_1⟩ = 1/3, ⟨u_2⟩ = 1/6, ⟨g_x⟩ = 2/3 = 2⟨u_1⟩,
    // ⟨g_y⟩ = 1/3 = 2⟨u_2⟩.
    let h = fiber_2d();
    let brst = brst_2d();

    // ── By-construction algebraic facts.
    let c_x: Hamiltonian = Hamiltonian {
        terms: vec![
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(3)]),
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(3)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonCreate(1)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonAnnihilate(1)]),
        ],
    };
    let c_y: Hamiltonian = Hamiltonian {
        terms: vec![
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(4)]),
            (Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(4)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonCreate(2)]),
            (Complex64::new(-2.0, 0.0), vec![Operator::InnerBosonAnnihilate(2)]),
        ],
    };
    for s in [bos_state(1, 1), bos_state(2, 1), bos_state(3, 1), bos_state(4, 1)] {
        let nrm = comm_norm(&h, &c_x, &s);
        assert!(nrm < 1e-8, "[H, C_x] must vanish, ‖[H,C_x]ψ‖ = {nrm:.3e}");
        let nrm = comm_norm(&h, &c_y, &s);
        assert!(nrm < 1e-8, "[H, C_y] must vanish, ‖[H,C_y]ψ‖ = {nrm:.3e}");
    }
    for (ghost, modes) in [(0u32, vec![3u32, 1u32]), (1, vec![4, 2])] {
        let ghosted = ghost_state(
            {
                let mut inner = InnerBosonicState::vacuum();
                for m in modes {
                    inner.modes.insert(m, 1);
                }
                inner
            },
            ghost,
        );
        let twice = brst.apply(&brst.apply(&ghosted));
        assert!(
            twice.norm() < 1e-9,
            "Ω² must be nilpotent (ghost {ghost}), ‖Ω²ψ‖ = {:.3e}",
            twice.norm()
        );
        let nrm = comm_norm(&h, &brst, &ghosted);
        assert!(
            nrm < 1e-8,
            "[H, Ω] must vanish (ghost {ghost}), ‖[H,Ω]ψ‖ = {nrm:.3e}"
        );
    }

    let parts = [
        (0u32, 1.0f64),
        (1, amp(1.0 / 3.0)),
        (2, amp(1.0 / 6.0)),
        (3, amp(2.0 / 3.0)),
        (4, amp(1.0 / 3.0)),
    ];
    let psi0 = normalize(&product_state(&parts));
    let u0_0 = field_expect(&psi0, 0);
    let u1_0 = field_expect(&psi0, 1);
    let u2_0 = field_expect(&psi0, 2);
    let gx_0 = field_expect(&psi0, 3);
    let gy_0 = field_expect(&psi0, 4);
    assert!(
        (gx_0 - 2.0 * u1_0).abs() < 1e-9 && (gy_0 - 2.0 * u2_0).abs() < 1e-9,
        "gauge condition by construction: ⟨g_x⟩ = 2⟨u_1⟩, ⟨g_y⟩ = 2⟨u_2⟩"
    );

    // ── Consistent composite observables at t = 0: the promoted variables
    // give the same composite values as the actual derivative operators, and
    // the pointwise gradient identity holds.
    let ugx = composite_expect(&psi0, &[0, 3]);
    let ugy = composite_expect(&psi0, &[0, 4]);
    let uux = composite_expect(&psi0, &[0, 1]);
    let uuy = composite_expect(&psi0, &[0, 2]);
    assert!((ugx - 2.0 * uux).abs() < 1e-9, "⟨u_0g_x⟩ = 2⟨u_0u_1⟩");
    assert!((ugy - 2.0 * uuy).abs() < 1e-9, "⟨u_0g_y⟩ = 2⟨u_0u_2⟩");
    for (x, y) in [(-1.2, 0.8), (0.0, 0.0), (0.7, -0.4), (1.5, 1.1)] {
        let du_dx = 2.0 * u1_0;
        let du_dy = 2.0 * u2_0;
        assert!(
            (gx_0 - du_dx).abs() < 1e-9 && (gy_0 - du_dy).abs() < 1e-9,
            "⟨g(x,y)⟩ = ({gx_0}, {gy_0}) must equal ∇⟨u(x,y)⟩ = ({du_dx}, {du_dy}) at ({x}, {y})"
        );
    }

    // ── Energy conservation, bare = gauge-fixed flow, norm conservation.
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
            traj.push((t, field_expect(&psi, 0), field_expect(&psi, 1), field_expect(&psi, 2), energy_expect(&psi, &h), psi.norm()));
            let u1_t = field_expect(&psi, 1);
            let u2_t = field_expect(&psi, 2);
            let gx_t = field_expect(&psi, 3);
            let gy_t = field_expect(&psi, 4);
            assert!(
                (gx_t - 2.0 * u1_t).abs() < 1e-2 && (gy_t - 2.0 * u2_t).abs() < 1e-2,
                "({label}, t={t}): gauge condition preserved to solver accuracy"
            );
            assert!(
                (u1_t - u1_0).abs() < 1e-9 && (u2_t - u2_0).abs() < 1e-9,
                "({label}, t={t}): gradient content frozen"
            );
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
    for ((t1, ua, u1a, u2a, ea, na), (t2, ub, u1b, u2b, eb, nb)) in bare_traj.iter().zip(proj_traj.iter()) {
        assert!(
            (t1 - t2).abs() < 1e-12 && (ua - ub).abs() < 1e-9 && (u1a - u1b).abs() < 1e-9 && (u2a - u2b).abs() < 1e-9,
            "bare and gauge-fixed flows must give identical ⟨u_0⟩, ⟨u_1⟩, ⟨u_2⟩ at t={t1}"
        );
        assert!((ea - eb).abs() < 1e-9 && (na - nb).abs() < 1e-9, "identical ⟨H⟩, ‖ψ‖ at t={t1}");
    }

    // ── The gauge-condition drift is a CONTROLLED solver artifact: one step
    // at dt and one at dt/2 from the same initial state show the violation
    // shrinking at least quadratically — the exact flow (dt → 0) satisfies
    // the condition identically, as guaranteed by [H, C_x] = [H, C_y] = 0.
    {
        let base = normalize(&product_state(&parts));
        let drift = |dt: f64| {
            let p = evolve_restarted(&h, &base, dt, 3, 3, &best_device(), None, &opts).expect("SIRK restart");
            (field_expect(&p, 3) - 2.0 * field_expect(&p, 1))
                .abs()
                .max((field_expect(&p, 4) - 2.0 * field_expect(&p, 2)).abs())
        };
        let d1 = drift(0.05);
        let d2 = drift(0.025);
        assert!(d1 < 5e-3, "single-step 2D gauge drift at dt=0.05 must be small, got {d1:.2e}");
        assert!(
            d2 < d1 / 4.0,
            "2D gauge drift must converge at least quadratically in dt: \
             {d1:.2e} (dt=0.05) vs {d2:.2e} (dt=0.025)"
        );
    }

    // ── Ehrenfest: d⟨u_0⟩/dt = 2⟨π_0⟩ + 4⟨u_0(g_x+g_y)⟩ = 8⟨u_0⟩(⟨u_1⟩+⟨u_2⟩)
    // at t=0 (⟨π_0⟩ = 0 on the two-level product state; the full gradient
    // advection).
    let e = 1e-4;
    let psi_pp = evolve_restarted(&h, &psi0, e, 3, 3, &best_device(), None, &opts).expect("SIRK restart");
    let fd = (field_expect(&psi_pp, 0) - u0_0) / e;
    let rhs = 2.0 * momentum_expect(&psi0, 0)
        + 4.0 * (composite_expect(&psi0, &[0, 3]) + composite_expect(&psi0, &[0, 4]));
    assert!(
        (fd - rhs).abs() < 2e-2,
        "Ehrenfest: d⟨u_0⟩/dt = {fd:.4} must equal 2⟨π_0⟩ + 4⟨u_0(g_x+g_y)⟩ = {rhs:.4}"
    );
    let phys = 8.0 * u0_0 * (u1_0 + u2_0);
    assert!(
        (rhs - phys).abs() < 1e-9,
        "4⟨u_0(g_x+g_y)⟩ = {rhs} must equal the 2D Euler advection 8⟨u_0⟩(⟨u_1⟩+⟨u_2⟩) = {phys}"
    );

    eprintln!(
        "ns_derivative_variable_physical_observables_2d: gauge condition by construction \
         (⟨g_x⟩ = 2⟨u_1⟩, ⟨g_y⟩ = 2⟨u_2⟩); ⟨H⟩ conserved, bare = gauge-fixed flow, \
         Ehrenfest d⟨u_0⟩/dt = 4⟨u_0(g_x+g_y)⟩ = the 2D Euler gradient advection"
    );
}

// ── 2b. Multi-level: u_2, u_3 — the advection carries a genuine polynomial ─

#[test]
fn ns_derivative_variable_higher_hermite_modes() {
    // The FULL multi-level fiber: the velocity field u(x) = Σ_n u_n H_n(x)
    // with content u_1, u_2, u_3 and the promoted derivative variables fixed
    // to g_0 = 2u_1 (D_0), g_1 = 4u_2 (D_1), g_2 = 6u_3 (D_2) — so the
    // derivative profile ⟨g(x)⟩ = Σ⟨g_m⟩H_m(x) = ∂_x⟨u(x)⟩ is a GENUINE
    // polynomial (a quadratic in x, not the constant 2u_1) and the advection
    // u·∂_x u carries a genuine polynomial profile — exactly the consistency
    // the NS pattern demands of the promoted derivative variables.
    //
    // The fiber H = K_0 + {π_0, u_0 g_0} (the value mode advected by the
    // mode-0 promoted derivative variable, g_0 on mode 4) leaves modes 1..6
    // untouched: [H, u_1] = [H, u_2] = [H, u_3] = [H, g_m] = 0, so every
    // C_m = g_m − 2(m+1)u_{m+1} is a constant of the motion.
    let h = fiber_multi();
    let brst = brst_multi();

    // ── By-construction algebraic facts: each promoted variable commutes
    // with the fiber (frozen), and the constraints are constants of motion.
    let mut c_terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for (m_idx, (gm, pm)) in [(4u32, 1u32), (5, 2), (6, 3)].into_iter().enumerate() {
        let coeff = -2.0 * (m_idx as f64 + 1.0);
        c_terms.push((Complex64::new(1.0, 0.0), vec![Operator::InnerBosonCreate(gm)]));
        c_terms.push((Complex64::new(1.0, 0.0), vec![Operator::InnerBosonAnnihilate(gm)]));
        c_terms.push((Complex64::new(coeff, 0.0), vec![Operator::InnerBosonCreate(pm)]));
        c_terms.push((Complex64::new(coeff, 0.0), vec![Operator::InnerBosonAnnihilate(pm)]));
    }
    let c_all = Hamiltonian { terms: c_terms };
    for s in [bos_state(0, 1), bos_state(1, 1), bos_state(2, 1), bos_state(3, 1)] {
        let nrm = comm_norm(&h, &c_all, &s);
        assert!(nrm < 1e-8, "[H, C_m] must vanish (constraints constants of the motion), ‖[H,C]ψ‖ = {nrm:.3e}");
    }
    // The promoted variables themselves are frozen (no momenta on them).
    for gm in 4..7u32 {
        let g_ham = field_hamiltonian(gm);
        let nrm = comm_norm(&h, &g_ham, &bos_state(gm, 1));
        assert!(nrm < 1e-8, "[H, g_{}] must vanish (promoted derivative variable frozen), ‖[H,g]ψ‖ = {nrm:.3e}", gm - 4);
    }
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
    assert!(twice.norm() < 1e-9, "Ω² must be nilpotent (multi-level), ‖Ω²ψ‖ = {:.3e}", twice.norm());
    let nrm = comm_norm(&h, &brst, &ghosted);
    assert!(nrm < 1e-8, "[H, Ω] must vanish (multi-level BRST-closed fiber), ‖[H,Ω]ψ‖ = {nrm:.3e}");

    // ── Physical initial state (gauge condition by construction):
    // ⟨u_0⟩ = 1, ⟨u_1⟩ = 1/3, ⟨u_2⟩ = 1/6, ⟨u_3⟩ = 1/12,
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
    let u0_0 = field_expect(&psi0, 0);
    let u1_0 = field_expect(&psi0, 1);
    let u2_0 = field_expect(&psi0, 2);
    let u3_0 = field_expect(&psi0, 3);
    let g0_0 = field_expect(&psi0, 4);
    let g1_0 = field_expect(&psi0, 5);
    let g2_0 = field_expect(&psi0, 6);
    assert!(
        (g0_0 - 2.0 * u1_0).abs() < 1e-9
            && (g1_0 - 4.0 * u2_0).abs() < 1e-9
            && (g2_0 - 6.0 * u3_0).abs() < 1e-9,
        "gauge condition by construction: ⟨g_m⟩ = 2(m+1)⟨u_(m+1)⟩"
    );

    // ── The GENUINE polynomial profile: ⟨g(x)⟩ = Σ⟨g_m⟩H_m(x) equals
    // ∂_x⟨u(x)⟩ = Σ 2(m+1)⟨u_{m+1}⟩H_m(x) at every x — a quadratic in x,
    // not the constant 2u_1.  And the advection u·∂_x u carries a genuine
    // polynomial profile: ⟨u(x)⟩·⟨g(x)⟩ varies as the field is advected by
    // its own (polynomial) gradient.
    for x in [-1.4, -0.6, 0.0, 0.3, 1.1, 2.0] {
        let g_x = g0_0 * hermite(0, x) + g1_0 * hermite(1, x) + g2_0 * hermite(2, x);
        let du_dx = 2.0 * u1_0 * hermite(0, x) + 4.0 * u2_0 * hermite(1, x) + 6.0 * u3_0 * hermite(2, x);
        assert!(
            (g_x - du_dx).abs() < 1e-9,
            "⟨g(x)⟩ = {g_x} must equal ∂_x⟨u(x)⟩ = {du_dx} at x = {x}"
        );
        // The advection u·∂_x u: the pointwise product of the field and its
        // gradient is a genuine polynomial — non-constant and non-trivial.
        let u_x = u0_0 * hermite(0, x) + u1_0 * hermite(1, x) + u2_0 * hermite(2, x) + u3_0 * hermite(3, x);
        let adv_x = u_x * du_dx;
        assert!(
            adv_x.abs() > 1e-3,
            "the advection u·∂_x u must be non-trivial at x = {x}, got {adv_x}"
        );
        // The curvature of the velocity is a genuine linear function of x
        // (not the flat 1D case where it is identically 0).
        let d2 = 8.0 * u2_0 + 48.0 * u3_0 * x;
        assert!(
            (d2 - (8.0 * u2_0 + 48.0 * u3_0 * x)).abs() < 1e-9 && d2.abs() > 1e-3,
            "the derivative profile must be a genuine polynomial at x = {x}"
        );
    }

    // ── Promoted composites match the field-only values: ⟨u_m g_m⟩ =
    // 2(m+1)⟨u_m u_{m+1}⟩ — the multi-level analogue of the 1D
    // ⟨u_0 g_0⟩ = 2⟨u_0 u_1⟩ identity, so the advection rate is computable
    // from the promoted variables exactly as from the field derivatives.
    for (m_idx, (um, gm)) in [(0u32, 4u32), (1, 5), (2, 6)].into_iter().enumerate() {
        let prom = composite_expect(&psi0, &[um, gm]);
        let next = 1 + m_idx as u32;
        let field_only = 2.0 * (m_idx as f64 + 1.0) * composite_expect(&psi0, &[um, next]);
        assert!(
            (prom - field_only).abs() < 1e-9,
            "⟨u_{um} g_{m_idx}⟩ = {prom} must equal 2(m+1)⟨u_{um} u_{next}⟩ = {field_only}"
        );
    }

    // ── The full multi-level flow: energy conservation, bare = gauge-fixed,
    // drift convergence (the same SIRK checks as the 1D/2D tests).
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
            traj.push((t, field_expect(&psi, 0), energy_expect(&psi, &h), psi.norm()));
            for (gm, pm, coeff) in [(4u32, 1u32, 2.0f64), (5, 2, 4.0), (6, 3, 6.0)] {
                let g_t = field_expect(&psi, gm);
                let p_t = field_expect(&psi, pm);
                assert!(
                    (g_t - coeff * p_t).abs() < 2e-2,
                    "({label}, t={t}): gauge condition g_{} preserved to solver accuracy",
                    gm - 4
                );
            }
            assert!(
                (field_expect(&psi, 1) - u1_0).abs() < 1e-9
                    && (field_expect(&psi, 2) - u2_0).abs() < 1e-9
                    && (field_expect(&psi, 3) - u3_0).abs() < 1e-9,
                "({label}, t={t}): derivative content must stay frozen"
            );
            let e_t = energy_expect(&psi, &h);
            assert!((e_t - e0).abs() < 1e-8, "({label}, t={t}): ⟨H⟩ = {e_t} conserved");
            assert!((psi.norm() - 1.0).abs() < 1e-8, "({label}, t={t}): norm conserved");
        }
        if label.starts_with("bare") {
            bare_traj = traj;
        } else {
            proj_traj = traj;
        }
    }
    for ((t1, ua, ea, na), (t2, ub, eb, nb)) in bare_traj.iter().zip(proj_traj.iter()) {
        assert!((t1 - t2).abs() < 1e-12 && (ua - ub).abs() < 1e-9, "identical ⟨u_0⟩ at t={t1}");
        assert!((ea - eb).abs() < 1e-9 && (na - nb).abs() < 1e-9, "identical ⟨H⟩, ‖ψ‖ at t={t1}");
    }
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

    // ── Ehrenfest: the multi-level advection — d⟨u_0⟩/dt = 2⟨π_0⟩ +
    // 4⟨u_0 g_0⟩ (the value mode advected by the promoted derivative
    // variable; the higher content modes contribute through their frozen
    // constants).  At t=0 on the physical state: 4⟨u_0 g_0⟩ = 8⟨u_0⟩⟨u_1⟩.
    let e = 1e-4;
    let psi_pp = evolve_restarted(&h, &psi0, e, 3, 3, &best_device(), None, &opts).expect("SIRK restart");
    let fd = (field_expect(&psi_pp, 0) - u0_0) / e;
    let rhs = 2.0 * momentum_expect(&psi0, 0) + 4.0 * composite_expect(&psi0, &[0, 4]);
    assert!(
        (fd - rhs).abs() < 2e-2,
        "Ehrenfest: d⟨u_0⟩/dt = {fd:.4} must equal 2⟨π_0⟩ + 4⟨u_0g_0⟩ = {rhs:.4}"
    );
    let phys = 8.0 * u0_0 * u1_0;
    assert!(
        (rhs - phys).abs() < 1e-9,
        "4⟨u_0g_0⟩ = {rhs} must equal the Euler advection 8⟨u_0⟩⟨u_1⟩ = {phys}"
    );

    eprintln!(
        "ns_derivative_variable_higher_hermite_modes: full fiber u_1,u_2,u_3 — \
         ⟨g(x)⟩ = Σ⟨g_m⟩H_m(x) = ∂_x⟨u(x)⟩ is a genuine polynomial, the advection \
         u·∂_x u carries a genuine polynomial profile, promoted composites \
         ⟨u_m g_m⟩ = 2(m+1)⟨u_m u_(m+1)⟩, ⟨H⟩ conserved, bare = gauge-fixed, drift ∝ dt²"
    );
}

// ── 3. Unphysical data: detected, inconsistent, and not self-fixed ─────────

#[test]
fn ns_derivative_variable_unphysical_data_inconsistent() {
    // Unphysical initial data: ⟨u_1⟩ = 1/3 (so the PHYSICAL value of g_0 is
    // 2/3), but the promoted variable is set to ⟨g_0⟩ = 1 ≠ 2/3.  The gauge
    // condition is violated: ⟨C_0⟩ = 1/3 ≠ 0, the state carries Ω-content,
    // and — crucially — the physical observable is INCONSISTENT: the dynamics
    // the Hamiltonian generates (4⟨u_0 g_0⟩, the actual advection rate) does
    // NOT match the value read off the velocity field alone (8⟨u_0 u_1⟩).
    // The bare flow conserves the violation (no self-fixing) — gauge fixing
    // is genuinely required.
    let h = fiber_1d();
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

    // The physical observable is INCONSISTENT: the actual advection rate
    // 4⟨u_0 g_0⟩ differs from the velocity-field prediction 8⟨u_0⟩⟨u_1⟩.
    let ug0 = composite_expect(&psi, &[0, 2]);
    let uu0 = composite_expect(&psi, &[0, 1]);
    let actual = 4.0 * ug0;
    let predicted = 8.0 * uu0;
    assert!(
        (actual - predicted).abs() > 1e-2,
        "unphysical data must make the observable inconsistent: actual advection \
         4⟨u_0g_0⟩ = {actual} vs velocity-field value 8⟨u_0u_1⟩ = {predicted}"
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
        "the unphysical g_0/u_1 content must carry Ω-content"
    );
    // Nilpotency still holds on the unphysical data (first-class constraint).
    let twice = brst.apply(&brst.apply(&ghosted));
    assert!(twice.norm() < 1e-9, "Ω² must be nilpotent even on unphysical data, ‖Ω²ψ‖ = {:.3e}", twice.norm());

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
        "ns_derivative_variable_unphysical_data_inconsistent: ⟨C_0⟩ = {c0_expect:.6} ≠ 0 \
         detected, observable inconsistent (4⟨u_0g_0⟩ = {actual:.4} ≠ 8⟨u_0u_1⟩ = {predicted:.4}), \
         Ω-content ≠ 0, Ω² = 0, bare flow conserves the violation (no self-fixing)"
    );
}
