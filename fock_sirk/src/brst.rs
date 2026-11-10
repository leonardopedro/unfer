//! Proper BRST projection onto the physical (gauge-invariant) subspace.
//!
//! The original solver enforced the gauge constraint by *subtracting* `Q w`
//! from `w`, which is not a projection (not idempotent, not self-adjoint, and
//! leaves a gauge residue). Here we implement the true orthogonal projector onto
//! `ker Q`:
//!
//! ```text
//!   P w = w − Q† z,   where  (Q Q†) z = Q w
//! ```
//!
//! `Q Q†` is Hermitian positive-semidefinite, so we solve for `z` matrix-free
//! with the conjugate-gradient method, never materializing a dense operator.
//!
//! # Conditioning (measured 2026-11-10)
//!
//! CG runs on the normal equations, so its residual is *exactly* the
//! projection's physical residual:
//!
//! ```text
//!   QQ†z − Qw = −Q(w − Q†z) = −Q(Pw),
//! ```
//!
//! i.e. the `tol` the caller passes bounds `‖Q(Pw)‖` directly. There are two
//! regimes, and they are not the same phenomenon.
//!
//! ## Charges with finite conditioning
//!
//! The QG probe that used to fail (`Ω = (a†₂ + a₂)c₉`, a single outer ladder
//! pair) has a CG–Lanczos tridiagonal at 60 steps with `λ_min =
//! 1.648430476329554e-1`, `λ_max = 4.444237534453834e2`, and its residual
//! decays as CG theory predicts for that κ (`0.390` after 50 iterations,
//! `9.12e-2` after 60) — a budget problem, cured by supplying the closing
//! charge. The *promoted* charge `Ω = X·c₉`,
//! `X = Σ x[ie][je]C†(e_i)A(e_j)`, converges on the same problem in **one**
//! iteration.
//!
//! ## Charges built from an unbounded Gauss generator
//!
//! The abelian YM and NS charges (`Ω = Σ_m i(a†_m − a_m)c_g`, `Ω = Σ_j
//! u_{j,j}c_j`) do **not** converge, and no `max_iter` fixes them. Instrumented
//! CG on `Ω|ψ⟩` for the one-ghost probe of `brst_constraint_validation.rs`
//! (`‖Ωw‖ = 2.0`), residual `‖Ω(Pw)‖`:
//!
//! ```text
//!   it    0      5     10     50     80     115
//!   ‖r‖  1.388  0.777 0.631  0.388  0.338  0.306      (4.75 s at 115)
//!   |z|    3     48    143   2548   5549   9861 components
//! ```
//!
//! The decay is algebraic (ratio `0.79` for `2.3×` the iterations), not
//! geometric, and the iterate's support grows without bound — the signature of
//! a normal equation whose solution is not in the truncated Fock space, not of
//! a large finite κ. The structure explains it exactly. Write the charge as
//! `Q = A·c_g` with `A = Σ_m i(a†_m − a_m)`; then `A = Σ_m p_m` is the **total
//! momentum**, whose spectrum is all of `ℝ` with no eigenvalue, and
//!
//! ```text
//!   QQ† = −A² ⊗ P₀,   P₀ = c_g c_g†  (the ghost-free projector),
//!   z    = −A⁻¹ c_g w   ⇒   A⁻¹ c_g w  is in L² but not in any Fock
//!                            truncation (A's spectrum is continuous),
//!   Pw   = w − Q†z = w − c_g†c_g w = P₀w        (A cancels),
//! ```
//!
//! so the **answer is trivial** while the intermediate the CG solve is chasing
//! is unrepresentable. `A⁻¹c_g w` is a smeared state spread over the whole
//! oscillator ladder; extending the truncation only moves the plateau.
//! [`project_ghost_free`] exploits the cancellation directly and is exact and
//! `O(len(w))` for this class of charge. Its one precondition — that each
//! bosonic co-factor be injective — is **checked** per charge
//! (`cofactor_is_injective`), not assumed; a co-factor with a kernel is
//! declined and the caller keeps the numeric path.

use nested_fock_algebra::{Hamiltonian, Operator, QuantumState};
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

use crate::linalg::SirkError;

fn norm(s: &QuantumState) -> f64 {
    QuantumState::inner_product(s, s).re.max(0.0).sqrt()
}

/// Is the bosonic co-factor built from `terms` **injective**?
///
/// `terms` is the collection of `(coeff, ladders)` that belong to one ghost
/// mode; this decides `ker A = {0}` for
///
/// ```text
///   A = c·1 + Σ_m (α_m a_m + β_m a†_m)
/// ```
///
/// by a *sufficient* criterion, returning `false` ("not proved") rather than
/// guessing for anything outside it:
///
/// * every term is a constant or a **single** inner boson ladder — a product of
///   two or more is declined (degree ≥ 2 is not decided here);
/// * if the whole degree-1 part cancels, `A = c·1` is injective iff `c ≠ 0`;
/// * otherwise `A` must be Hermitian (`β_m = ᾱ_m`) or anti-Hermitian
///   (`β_m = −ᾱ_m`). Such an `A` is, by a metaplectic (Bogoliubov) rotation of
///   the modes, a constant plus a multiple of one quadrature `x`; multiplying
///   by `c + μx` has no square-integrable kernel, so `ker A = {0}`.
///
/// The two charges that need the structural projector pass: YM's
/// `Σ_m i(a†_m − a_m) = Σ_m π_m` (Hermitian) and NS's `Σ_k A_{jk} u_k` with
/// real `A` (Hermitian). A co-factor that is *not* covered — e.g. a bare
/// annihilation `a`, whose kernel is the vacuum — is declined, which is the
/// whole point: the previous version of this function assumed the condition.
fn cofactor_is_injective(terms: &[(Complex64, Vec<Operator>)]) -> bool {
    let mut constant = Complex64::new(0.0, 0.0);
    let mut ann: BTreeMap<u32, Complex64> = BTreeMap::new();
    let mut cre: BTreeMap<u32, Complex64> = BTreeMap::new();
    for (coeff, ops) in terms {
        match ops.as_slice() {
            [] => constant += coeff,
            [Operator::InnerBosonAnnihilate(m)] => {
                *ann.entry(*m).or_insert(Complex64::new(0.0, 0.0)) += coeff
            }
            [Operator::InnerBosonCreate(m)] => {
                *cre.entry(*m).or_insert(Complex64::new(0.0, 0.0)) += coeff
            }
            _ => return false,
        }
    }
    if ann.is_empty() && cre.is_empty() {
        return constant != Complex64::new(0.0, 0.0);
    }
    // Hermitian, anti-Hermitian, or not decided.
    let mut hermitian = true;
    let mut anti = true;
    for m in ann.keys().chain(cre.keys()) {
        let a = ann.get(m).copied().unwrap_or(Complex64::new(0.0, 0.0));
        let b = cre.get(m).copied().unwrap_or(Complex64::new(0.0, 0.0));
        hermitian &= (b - a.conj()).norm() <= 1e-12 * (1.0 + b.norm() + a.norm());
        anti &= (b + a.conj()).norm() <= 1e-12 * (1.0 + b.norm() + a.norm());
    }
    hermitian || anti
}

/// The ghost modes an *inner* ghost-lowering charge annihilates, provided every
/// co-factor is *proved* injective.
///
/// Returns `Some({g_k})` when every term of `q` is `A_k · c_{g_k}` — a product
/// of inner boson ladders followed by one inner ghost annihilation, with no
/// fermion anywhere else in the term — and the per-mode co-factor
/// `A_g = Σ_{k: g_k = g} A_k` is injective ([`cofactor_is_injective`]). For
/// such a charge `Q = Σ_k A_k c_{g_k}` with `A_k` bosonic, so `c_{g_k}`
/// annihilates every state carrying no fermion in the returned set.
fn ghost_lowering_modes(q: &Hamiltonian) -> Option<BTreeSet<u32>> {
    if q.terms.is_empty() {
        return None;
    }
    let mut cofactors: BTreeMap<u32, Vec<(Complex64, Vec<Operator>)>> = BTreeMap::new();
    for (coeff, ops) in &q.terms {
        let (last, bosonic) = ops.split_last()?;
        let Operator::InnerFermionAnnihilate(g) = last else {
            return None;
        };
        // Everything else must be a *bosonic* inner ladder: a fermionic or an
        // outer factor would break `[A_k, c_g] = 0` / the sector argument.
        for op in bosonic {
            match op {
                Operator::InnerBosonCreate(_) | Operator::InnerBosonAnnihilate(_) => {}
                _ => return None,
            }
        }
        cofactors
            .entry(*g)
            .or_default()
            .push((*coeff, bosonic.to_vec()));
    }
    // The lemma needs each co-factor injective; check it instead of assuming it.
    if !cofactors.values().all(|terms| cofactor_is_injective(terms)) {
        return None;
    }
    Some(cofactors.into_keys().collect())
}

/// Exact orthogonal projection onto `ker Q` for a ghost-lowering charge
/// (`Q = Σ_k A_k c_{g_k}`, see [`ghost_lowering_modes`]).
///
/// `P w` deletes every Fock component whose inner state carries a fermion in
/// one of `Q`'s ghost modes. That is exact, needs no solve, and is the
/// projection the CG path cannot reach:
///
/// * **`ran P ⊆ ker Q`** — `c_g` annihilates a state with no `g`, so
///   `Q(Pw) = 0` identically, whatever the `A_k` are.
/// * **`ker Q ⊆ ran P`** — write `w = Σ_G w_G` by ghost number; `Q` raises it
///   by one, so `Qw = 0` requires `Σ_k A_k c_{g_k} w_G = 0` for every `G ≥ 1`.
///   In the one-ghost sector `g` this reads `A_g b_g w = 0`, so `b_g w = 0`
///   (hence `w` is ghost-free) **provided `A_g` has trivial kernel**.
/// * `P` is diagonal in the Fock basis, hence self-adjoint and idempotent.
///
/// The injectivity of `A_g` is **checked**, not assumed: [`ghost_lowering_modes`]
/// proves it for a co-factor that is a constant plus a Hermitian (or
/// anti-Hermitian) degree-1 ladder combination — which covers both charges
/// that need this path, YM's `Σ_m π_m` and NS's real `Σ_k A_{jk} u_k` —
/// and declines anything else. A co-factor *with* a kernel (a bare `a`, say)
/// therefore returns `None` rather than a state that sits in `ker Q` but is
/// not the closest point to `w`.
///
/// Returns `None` if `q` does not have the ghost-lowering shape, or if a
/// co-factor's injectivity cannot be established.
pub fn project_ghost_free(w: &QuantumState, q: &Hamiltonian) -> Option<QuantumState> {
    let modes = ghost_lowering_modes(q)?;
    let mut out = QuantumState::zero();
    for (state, amp) in &w.components {
        let carries_ghost = state
            .fermionic
            .iter()
            .any(|f| f.modes.iter().any(|m| modes.contains(m)));
        if !carries_ghost {
            out.components.insert(state.clone(), *amp);
        }
    }
    Some(out)
}

/// Project `w` onto the physical subspace `ker Q` along `im Q†`.
///
/// `q` is the BRST charge `Q`. Returns [`SirkError::BrstNotConverged`] if the
/// inner CG solve does not reach `tol` within `max_iter` iterations; for an
/// unbounded Gauss generator that is the expected outcome at every budget —
/// see the module docs and [`project_ghost_free`].
pub fn project_physical(
    w: &QuantumState,
    q: &Hamiltonian,
    tol: f64,
    max_iter: usize,
) -> Result<QuantumState, SirkError> {
    let q_dag = q.adjoint();

    // The normal operator A = Q Q† (Hermitian PSD), applied matrix-free.
    let apply_a = |x: &QuantumState| -> QuantumState { q.apply(&q_dag.apply(x)) };

    // Right-hand side b = Q w.
    let b = q.apply(w);
    if norm(&b) < tol {
        // w already (numerically) physical: nothing to project out.
        return Ok(w.clone());
    }

    // Conjugate gradient solve of A z = b, starting from z = 0.
    let mut z = QuantumState::zero();
    let mut r = b.clone(); // r = b - A z = b
    let mut p = r.clone();
    let mut rs_old = QuantumState::inner_product(&r, &r).re;

    let mut converged = false;
    let mut residual = rs_old.sqrt();
    for _ in 0..max_iter {
        let ap = apply_a(&p);
        let p_ap = QuantumState::inner_product(&p, &ap).re;
        if p_ap.abs() < 1e-30 {
            // Breakdown: p lies in ker A; remaining residual is unreachable.
            break;
        }
        let alpha = rs_old / p_ap;
        z.scale_and_add(&p, Complex64::new(alpha, 0.0));
        r.scale_and_add(&ap, Complex64::new(-alpha, 0.0));

        let rs_new = QuantumState::inner_product(&r, &r).re;
        residual = rs_new.sqrt();
        if residual < tol {
            converged = true;
            break;
        }
        let beta = rs_new / rs_old;
        let mut new_p = r.clone();
        new_p.scale_and_add(&p, Complex64::new(beta, 0.0));
        p = new_p;
        rs_old = rs_new;
    }

    if !converged {
        return Err(SirkError::BrstNotConverged { residual });
    }

    // P w = w − Q† z.
    let mut projected = w.clone();
    let correction = q_dag.apply(&z);
    projected.scale_and_add(&correction, Complex64::new(-1.0, 0.0));
    Ok(projected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nested_fock_algebra::{InnerBosonicState, InnerFermionicState, Operator, QuantumState};

    /// Build the ghost-lowering BRST charge Q = C_φ (outer fermion annihilation)
    /// plus the two basis states |0> (no ghost) and |1> (one ghost universe).
    fn ghost_setup() -> (Hamiltonian, QuantumState, QuantumState) {
        let phi = InnerFermionicState::vacuum();
        let q = Hamiltonian {
            terms: vec![(
                Complex64::new(1.0, 0.0),
                vec![Operator::OuterFermionAnnihilate(phi.clone())],
            )],
        };
        let s0 = QuantumState::vacuum(); // |0>: no fermionic universe
        let s1 = QuantumState::vacuum().apply(&Operator::OuterFermionCreate(phi)); // |1>
        (q, s0, s1)
    }

    fn combine(a: &QuantumState, ca: Complex64, b: &QuantumState, cb: Complex64) -> QuantumState {
        let mut s = QuantumState::zero();
        s.scale_and_add(a, ca);
        s.scale_and_add(b, cb);
        s
    }

    #[test]
    fn projection_kills_gauge_part() {
        let (q, s0, s1) = ghost_setup();
        let w = combine(&s0, Complex64::new(0.6, 0.1), &s1, Complex64::new(0.0, 0.8));
        let pw = project_physical(&w, &q, 1e-12, 100).unwrap();
        // Q(Pw) must vanish.
        assert!(norm(&q.apply(&pw)) < 1e-8, "Q(Pw) not annihilated");
        // Pw should equal the physical part 0.6+0.1i times |0>.
        let phys = combine(&s0, Complex64::new(0.6, 0.1), &s1, Complex64::new(0.0, 0.0));
        let mut diff = pw.clone();
        diff.scale_and_add(&phys, Complex64::new(-1.0, 0.0));
        assert!(norm(&diff) < 1e-8, "Pw != physical part");
    }

    #[test]
    fn projection_is_idempotent() {
        let (q, s0, s1) = ghost_setup();
        let w = combine(
            &s0,
            Complex64::new(0.3, -0.2),
            &s1,
            Complex64::new(0.7, 0.4),
        );
        let pw = project_physical(&w, &q, 1e-12, 100).unwrap();
        let ppw = project_physical(&pw, &q, 1e-12, 100).unwrap();
        let mut diff = ppw.clone();
        diff.scale_and_add(&pw, Complex64::new(-1.0, 0.0));
        assert!(norm(&diff) < 1e-10, "P is not idempotent");
    }

    #[test]
    fn projection_is_self_adjoint() {
        let (q, s0, s1) = ghost_setup();
        let v = combine(&s0, Complex64::new(0.5, 0.0), &s1, Complex64::new(0.2, 0.3));
        let w = combine(
            &s0,
            Complex64::new(0.1, -0.4),
            &s1,
            Complex64::new(0.9, 0.0),
        );
        let pv = project_physical(&v, &q, 1e-12, 100).unwrap();
        let pw = project_physical(&w, &q, 1e-12, 100).unwrap();
        // <v, Pw> == <Pv, w>
        let lhs = QuantumState::inner_product(&v, &pw);
        let rhs = QuantumState::inner_product(&pv, &w);
        assert!(
            (lhs - rhs).norm() < 1e-9,
            "P is not self-adjoint: {lhs} vs {rhs}"
        );
    }

    /// Off `ker Q` with a *structured* charge — a second-quantized position
    /// times a ghost, `Q = X·c` with `X = (|0><1| + |1><0|)/√2` on a one-mode
    /// outer boson truncation, not the rank-one `C_φ` of the tests above. This
    /// is the shape of the QG charge; it converges in a couple of CG steps,
    /// which is the point: the budget failure recorded for the QG probe came
    /// from a badly conditioned (non-closing) charge, not from this solver.
    #[test]
    fn projection_converges_off_kernel_for_a_second_quantized_charge() {
        let mut bos = InnerBosonicState::vacuum();
        bos.modes.insert(0, 1);
        let x = Hamiltonian {
            terms: vec![
                (
                    Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                    vec![
                        Operator::OuterBosonCreate(bos.clone()),
                        Operator::OuterBosonAnnihilate(InnerBosonicState::vacuum()),
                    ],
                ),
                (
                    Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                    vec![
                        Operator::OuterBosonCreate(InnerBosonicState::vacuum()),
                        Operator::OuterBosonAnnihilate(bos.clone()),
                    ],
                ),
            ],
        };
        let q = Hamiltonian {
            terms: x
                .terms
                .iter()
                .map(|(c, ops)| {
                    let mut o = ops.clone();
                    o.push(Operator::OuterFermionAnnihilate(
                        InnerFermionicState::vacuum(),
                    ));
                    (*c, o)
                })
                .collect(),
        };

        let physical = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(bos.clone()));
        let ghostly = physical
            .clone()
            .apply(&Operator::OuterFermionCreate(InnerFermionicState::vacuum()));
        let w = combine(
            &physical,
            Complex64::new(0.6, 0.0),
            &ghostly,
            Complex64::new(0.8, 0.0),
        );
        assert!(
            norm(&q.apply(&w)) > 1e-3,
            "the probe must be off ker Q: ‖Qw‖ = {:.3e}",
            norm(&q.apply(&w))
        );

        let pw = project_physical(&w, &q, 1e-12, 100).expect("CG must converge here");
        assert!(norm(&q.apply(&pw)) < 1e-8, "Q(Pw) not annihilated");
        let phys = combine(
            &physical,
            Complex64::new(0.6, 0.0),
            &ghostly,
            Complex64::new(0.0, 0.0),
        );
        let mut diff = pw.clone();
        diff.scale_and_add(&phys, Complex64::new(-1.0, 0.0));
        assert!(
            norm(&diff) < 1e-8,
            "Pw must be the physical component: residual {:.3e}",
            norm(&diff)
        );
        assert!(
            norm(&q.apply(&q.apply(&w))) < 1e-12,
            "the structured charge must be nilpotent: Q²w = {:.3e}",
            norm(&q.apply(&q.apply(&w)))
        );
    }

    /// The abelian Gauss charge `Q = Σ_m i(a†_m − a_m)c_g`: a ghost-lowering
    /// charge whose bosonic factor is the *unbounded* total momentum. CG cannot
    /// project it (the module docs' trajectory), while the structural
    /// projector is exact and `O(len(w))`.
    fn gauss_charge(ghost: u32) -> Hamiltonian {
        let mut terms = Vec::new();
        for mode in [0u32, 1] {
            terms.push((
                Complex64::new(0.0, 1.0),
                vec![
                    Operator::InnerBosonCreate(mode),
                    Operator::InnerFermionAnnihilate(ghost),
                ],
            ));
            terms.push((
                Complex64::new(0.0, -1.0),
                vec![
                    Operator::InnerBosonAnnihilate(mode),
                    Operator::InnerFermionAnnihilate(ghost),
                ],
            ));
        }
        Hamiltonian { terms }
    }

    fn universe(boson: Option<u32>, ghost: Option<u32>) -> QuantumState {
        let mut inner = InnerBosonicState::vacuum();
        if let Some(m) = boson {
            inner.modes.insert(m, 1);
        }
        let s = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner));
        match ghost {
            Some(g) => s.apply(&Operator::OuterFermionCreate(InnerFermionicState {
                modes: BTreeSet::from([g]),
            })),
            None => s,
        }
    }

    #[test]
    fn structural_projection_is_exact_where_cg_diverges() {
        let q = gauss_charge(9);
        let physical = universe(Some(1), None);
        let ghostly = universe(Some(1), Some(9));
        let w = combine(
            &physical,
            Complex64::new(0.6, 0.0),
            &ghostly,
            Complex64::new(0.8, 0.0),
        );

        // The CG route does not converge — at any budget (see module docs).
        assert!(
            matches!(
                project_physical(&w, &q, 1e-10, 50),
                Err(SirkError::BrstNotConverged { .. })
            ),
            "CG is expected to stall on an unbounded Gauss generator"
        );

        let pw = project_ghost_free(&w, &q).expect("ghost-lowering shape");
        assert!(
            norm(&q.apply(&pw)) == 0.0,
            "Q(Pw) must vanish exactly: {:.3e}",
            norm(&q.apply(&pw))
        );
        // P is the Fock-diagonal ghost projector, so it is self-adjoint and
        // idempotent exactly.
        let ppw = project_ghost_free(&pw, &q).unwrap();
        assert!(norm(&ppw) >= 0.0);
        let mut diff = ppw.clone();
        diff.scale_and_add(&pw, Complex64::new(-1.0, 0.0));
        assert!(norm(&diff) == 0.0, "P must be idempotent exactly");
        // And it returns the physical part, not just *some* element of ker Q.
        let mut resid = pw.clone();
        resid.scale_and_add(&physical, Complex64::new(-0.6, 0.0));
        assert!(
            norm(&resid) == 0.0,
            "P(0.6·phys + 0.8·ghost) must be 0.6·phys: {:.3e}",
            norm(&resid)
        );
    }

    #[test]
    fn structural_projection_declines_a_charge_it_does_not_recognise() {
        // The rank-one outer-fermion charge of the tests above is not
        // ghost-lowering (its fermionic factor is *outer*), so the structural
        // path must decline rather than guess — `project_physical` handles it.
        let (q, _, _) = ghost_setup();
        assert!(
            project_ghost_free(&QuantumState::vacuum(), &q).is_none(),
            "a charge outside the ghost-lowering shape must be rejected"
        );
        // A charge with a *creation* factor is likewise not projected onto its
        // annihilator's kernel: c† raises ghost number, so `c†` alone does not
        // define a ghost-free kernel.
        let q_create = Hamiltonian {
            terms: vec![(
                Complex64::new(1.0, 0.0),
                vec![
                    Operator::InnerBosonCreate(0),
                    Operator::InnerFermionCreate(9),
                ],
            )],
        };
        assert!(project_ghost_free(&QuantumState::vacuum(), &q_create).is_none());
    }

    #[test]
    fn structural_projection_declines_a_cofactor_with_a_kernel() {
        // `Q = a_0 c_9` has the ghost-lowering *shape* but its co-factor is not
        // injective: `a_0` annihilates the bosonic vacuum. Since `c_9` maps the
        // ghost sector to the empty-fermion sector, `Q` annihilates the state
        // `|vac>_bos ⊗ |ghost>_9` even though it carries the ghost. The
        // Fock-diagonal filter would delete it, i.e. return a state in ker Q
        // that is not the closest point to `w` — exactly the failure the
        // kernel condition guards.
        let q = Hamiltonian {
            terms: vec![(
                Complex64::new(1.0, 0.0),
                vec![
                    Operator::InnerBosonAnnihilate(0),
                    Operator::InnerFermionAnnihilate(9),
                ],
            )],
        };
        let ghostly = universe(None, Some(9));
        assert!(
            norm(&q.apply(&ghostly)) == 0.0,
            "|vac>_bos ⊗ |g> is in ker Q (‖Qw‖ = {:.3e}) — that is why the shape \
             alone is not enough",
            norm(&q.apply(&ghostly))
        );
        assert!(
            project_ghost_free(&ghostly, &q).is_none(),
            "a non-injective co-factor must be declined, not silently filtered"
        );
        // Degree ≥ 2 is outside the decided class (e.g. `a†a`, kernel |0>).
        let q_deg2 = Hamiltonian {
            terms: vec![(
                Complex64::new(1.0, 0.0),
                vec![
                    Operator::InnerBosonCreate(0),
                    Operator::InnerBosonAnnihilate(0),
                    Operator::InnerFermionAnnihilate(9),
                ],
            )],
        };
        assert!(project_ghost_free(&ghostly, &q_deg2).is_none());
        // A *Hermitian* degree-1 co-factor with a constant offset is decided:
        // `(1 + u_0) c_9` is injective (`1 + x` has no L² kernel).
        let q_affine = Hamiltonian {
            terms: vec![
                (
                    Complex64::new(1.0, 0.0),
                    vec![Operator::InnerFermionAnnihilate(9)],
                ),
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::InnerBosonCreate(0),
                        Operator::InnerFermionAnnihilate(9),
                    ],
                ),
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::InnerBosonAnnihilate(0),
                        Operator::InnerFermionAnnihilate(9),
                    ],
                ),
            ],
        };
        assert!(project_ghost_free(&ghostly, &q_affine).is_some());
    }
}
