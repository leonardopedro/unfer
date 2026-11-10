//! **The campaign re-run under the exclusion certificate** (`docs/MASS_GAP_SPEC.md`
//! §6c/§6d).
//!
//! `mass_gap_campaign.rs` emits the T6 expression `θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ)` for a
//! grid of couplings and records that `lo > 0` is *not* a gap: the expression
//! needs the odd solve to have resolved the odd sector's ground, and no term of
//! it witnesses that. This file runs the same grid through the certificate that
//! does witness it —
//!
//! ```text
//!   λ₁ − λ₀ ≥ temple(θᵒ₀, rᵒ, β) − θᵉ₀,   temple: λ₁ ≥ θ − r²/(β − θ)
//! ```
//!
//! — on each cell of the grid, with `β ≤ λ₂` supplied by a **sparse inertia
//! count** ([`inertia_below`]): `H_N − βI` is eliminated and Sylvester's law of
//! inertia decides how many eigenvalues lie below `β`, so a bisection on `β`
//! brackets `λ₂` from below without a single diagonalization. The count is
//! monotone in `β`, which is what makes the bisection valid; the bound it
//! returns is the largest deciding `β`, so Temple's `r²/(β − θ)` penalty is as
//! small as the bracket allows.
//!
//! Two weaker witnesses are still measured beside it, for contrast rather than
//! for certification: [`DiagonalExclusion`] (Weyl on the diagonal, `O(n)`
//! matvecs, decays like `√n` against a fixed gap —
//! `the_sparse_witness_is_sound_but_does_not_certify` pins that it does not
//! fire) and the dense [`certify_spectrum`] count, which is no longer on the
//! certification path at all. The exact spectrum below is a *test oracle* — it
//! checks every bound; it supplies none of them.
//!
//! and with the Ritz pair `(θ, r)` supplied two ways as well ([`Solver`]): the
//! forward truncated SIRK solve of §6d, and matvec-only Lanczos on the same
//! `H_N`. The certificate is identical; only the solve changes, and it is the
//! solve — not the witness — that decides most of these verdicts. `θᵉ₀` is
//! consumed only as an upper bound on `λ₀`, so the even side needs no witness at
//! all, just a value close enough to `λ₀` to leave room for the odd side.
//!
//! The grid runs `N ∈ {6, 8, 10}`. A cell where the odd ground sits *below* the
//! even one inside `H_N` has no gap in T6's orientation (the expression is the
//! negative of its gap), and the set of such cells is **not monotone in `N`**
//! (`g = 2` crosses at `N ≤ 8` and not at `N ≤ 10`; `g = 1` the other way round),
//! so a crossing is an oscillation of the truncation, not a sign that a deeper
//! `N` will cure it. Those cells are not left uncertified: Temple's inequality is
//! sector-local, so the same
//! assembly runs again with the sectors **swapped** — the even side templed
//! against its own `β ≤ λ₂(even)` witness, the odd side consumed only as the
//! Rayleigh–Ritz upper bound `λ₁(odd) ≤ θᵒ₀` — and certifies
//! `λ₀(even) − λ₁(odd)`, which is positive exactly there. The two orientations
//! are complements, so every cell on the grid carries a certificate of the gap
//! it actually has, and the parity of the ground is reported rather than hidden.
//!
//! **What the rows mean.** These are `H_N` rows: `P_N H P_N` on the truncated
//! inner-occupation basis. A certified positive gap here is a statement about
//! the truncated operator only — at `g = 0` it certifies a positive gap on a
//! theory with no continuum gap at all, which is exactly why the `g = 0` row is
//! the control that keeps the table honest.
//!
//! Every emitted bound is checked against the exact parity-resolved spectrum of
//! the same `H_N`: a bound above the exact gap fails the test.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::mass_gap_spec::certified_gap_lower_bound_with_exclusion;
use fock_sirk::{
    DiagonalExclusion, MatchingExclusion, SirkOpts, certified_ground_state, inertia_below,
    lanczos_ritz, solve_forward_sirk_truncated,
};
use nalgebra::DMatrix;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, OuterState, QuantumState, qcd_ym_hamiltonian,
};
use num_complex::Complex64;
use std::collections::HashMap;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: true,
    }
}

fn empty_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn fock_state(occ: &[(u32, u32)]) -> QuantumState {
    let mut s = empty_vacuum();
    for &(m, n) in occ {
        for _ in 0..n {
            s = s.apply(&Operator::InnerBosonCreate(m));
        }
    }
    let norm = s.norm();
    if (norm - 1.0).abs() > 1e-12 {
        s.scale_and_add(&s.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));
    }
    s
}

/// `(|1₀⟩ + |1₁⟩)/√2` — the exact reflection-odd one-quantum start.
fn r_odd_start() -> QuantumState {
    let s0 = fock_state(&[(0, 1)]);
    let s1 = fock_state(&[(1, 1)]);
    let mut s = s0;
    s.scale_and_add(&s1, Complex64::new(1.0, 0.0));
    let inv = 1.0 / 2.0f64.sqrt();
    s.scale_and_add(&s.clone(), Complex64::new(inv - 1.0, 0.0));
    s
}

/// The truncation `P_N`: every single-universe inner state with total occupation
/// `≤ N`, with one sparse column of `H_N` per basis state, plus the reflection's
/// action `R|i⟩ = sign·|j⟩` on that basis.
///
/// The columns are built by applying `H` to each basis state and looking up the
/// components that survive the truncation. Components above the ceiling are
/// exactly the `(I − P_N) H P_N` part: they are dropped and their norm is
/// reported as `leak`, so the truncation is measured rather than assumed away.
///
/// **The reflection sector is not a parity filter.** `R` flips modes 0↔1 and
/// multiplies by `(−1)^(n₀+n₁)`, so for `n₀ ≠ n₁` it maps `|n₀,n₁⟩` to
/// `|n₁,n₀⟩` and its eigenvectors are the ± *combinations*
/// `(|n₀,n₁⟩ ± (−1)^(n₀+n₁)|n₁,n₀⟩)/√2`. Selecting states by `(n₀+n₁)%2`
/// therefore does **not** give an `R` sector — and at `g > 0` it does not even
/// give an `H`-invariant subspace, because `B²` carries three-operator products
/// that change that parity. That is why this struct stores `R` explicitly and
/// builds sectors by pairing ([`Truncated::sector`]) instead.
struct Truncated {
    basis: Vec<QuantumState>,
    index: HashMap<OuterState, usize>,
    columns: Vec<Vec<(usize, f64)>>,
    /// `reflection[i] = (j, sign)` with `R|i⟩ = sign·|j⟩`.
    reflection: Vec<(usize, f64)>,
    leak: f64,
}

impl Truncated {
    fn build(h: &Hamiltonian, n_max: u32) -> Self {
        let mut basis = Vec::new();
        for n0 in 0..=n_max {
            for n1 in 0..=n_max - n0 {
                for n2 in 0..=n_max - n0 - n1 {
                    for n3 in 0..=n_max - n0 - n1 - n2 {
                        let mut occ = Vec::new();
                        for (mode, count) in [(0, n0), (1, n1), (2, n2), (3, n3)] {
                            if count > 0 {
                                occ.push((mode, count));
                            }
                        }
                        basis.push(fock_state(&occ));
                    }
                }
            }
        }
        let mut index = HashMap::new();
        for (i, s) in basis.iter().enumerate() {
            index.insert(s.components.keys().next().unwrap().clone(), i);
        }
        let mut columns = Vec::with_capacity(basis.len());
        let mut leak = 0.0_f64;
        for s in basis.iter() {
            let hs = h.apply(s);
            let mut column = Vec::new();
            for (key, amp) in hs.components.iter() {
                match index.get(key) {
                    Some(&i) => {
                        if amp.norm() > 1e-14 {
                            column.push((i, amp.re));
                        }
                    }
                    None => leak += amp.norm_sqr(),
                }
            }
            columns.push(column);
        }
        // R's action, read off the state itself rather than assumed.
        let mut reflection = Vec::with_capacity(basis.len());
        for s in basis.iter() {
            let rs = apply_reflection(s);
            let (key, amp) = rs
                .components
                .iter()
                .next()
                .expect("R image is a basis state");
            let j = *index.get(key).expect("R preserves the truncation");
            reflection.push((j, amp.re.signum()));
        }
        Self {
            basis,
            index,
            columns,
            reflection,
            leak: leak.sqrt(),
        }
    }

    fn dim(&self) -> usize {
        self.basis.len()
    }

    /// `x ↦ H_N x` over the sparse columns.
    fn matvec(&self) -> impl Fn(&[f64]) -> Vec<f64> + '_ {
        move |x: &[f64]| {
            let mut y = vec![0.0_f64; self.dim()];
            for (j, &xj) in x.iter().enumerate() {
                if xj == 0.0 {
                    continue;
                }
                for &(i, value) in &self.columns[j] {
                    y[i] += value * xj;
                }
            }
            y
        }
    }

    /// `x ↦ P_sector H_N x` in sector coordinates — the same operator the dense
    /// [`Truncated::sector_matrix`] represents, reached with matvecs only.
    fn sector_matvec<'a>(
        &'a self,
        sector: &'a [Vec<(usize, f64)>],
    ) -> impl Fn(&[f64]) -> Vec<f64> + 'a {
        let full = self.matvec();
        move |x: &[f64]| {
            let mut v = vec![0.0_f64; self.dim()];
            for (b, u) in sector.iter().enumerate() {
                if x[b] == 0.0 {
                    continue;
                }
                for &(i, coef) in u {
                    v[i] += x[b] * coef;
                }
            }
            let y = full(&v);
            sector
                .iter()
                .map(|u| u.iter().map(|&(i, coef)| coef * y[i]).sum())
                .collect()
        }
    }

    /// Coordinates of a state in `sector`, with the containment residual
    /// alongside: a start carrying weight outside the sector would silently
    /// solve a different problem, so the caller asserts on the second element.
    fn coordinates(&self, sector: &[Vec<(usize, f64)>], state: &QuantumState) -> (Vec<f64>, f64) {
        let v: Vec<f64> = self
            .basis
            .iter()
            .map(|s| QuantumState::inner_product(s, state).re)
            .collect();
        let coords: Vec<f64> = sector
            .iter()
            .map(|u| u.iter().map(|&(i, coef)| coef * v[i]).sum())
            .collect();
        let mut recon = vec![0.0_f64; self.dim()];
        for (b, u) in sector.iter().enumerate() {
            for &(i, coef) in u {
                recon[i] += coords[b] * coef;
            }
        }
        let residual = v
            .iter()
            .zip(&recon)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        (coords, residual)
    }

    /// Orthonormal basis of one reflection sector, as sparse combinations of the
    /// full-basis indices.
    ///
    /// Each `R`-orbit is a singleton `(|n,n⟩ ⟹ +)` or a pair `i ↔ j` with the
    /// same sign `s` both ways (`R² = 1`); the orbit contributes
    /// `(e_i + s·e_j)/√2` to the even sector and `(e_i − s·e_j)/√2` to the odd one.
    fn sector(&self, even: bool) -> Vec<Vec<(usize, f64)>> {
        let n = self.dim();
        let mut used = vec![false; n];
        let mut out = Vec::new();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        for i in 0..n {
            if used[i] {
                continue;
            }
            let (j, sign) = self.reflection[i];
            if j == i {
                used[i] = true;
                let is_even = sign > 0.0;
                if is_even == even {
                    out.push(vec![(i, 1.0)]);
                }
            } else {
                used[i] = true;
                used[j] = true;
                let s = if even { 1.0 } else { -1.0 };
                out.push(vec![(i, inv_sqrt2), (j, s * sign * inv_sqrt2)]);
            }
        }
        out
    }

    /// `H_N` compressed to one reflection sector, as an explicit matrix.
    ///
    /// Also **verifies the sector is `H`-invariant**: if `H` leaked out of it, the
    /// matrix below would not be the sector's operator and every eigenvalue read
    /// from it would be wrong. The residual is asserted here, once, so the
    /// property cannot rot silently.
    fn sector_matrix(&self, even: bool) -> DMatrix<f64> {
        let sector = self.sector(even);
        let k = sector.len();
        let n = self.dim();
        let mut a = DMatrix::<f64>::zeros(k, k);
        for (b, vector) in sector.iter().enumerate() {
            // H u_b in full-basis coordinates.
            let mut hu = vec![0.0_f64; n];
            for &(i, coef) in vector {
                for &(row, value) in &self.columns[i] {
                    hu[row] += coef * value;
                }
            }
            for (c, other) in sector.iter().enumerate() {
                a[(c, b)] = other.iter().map(|&(i, coef)| coef * hu[i]).sum();
            }
            // Leakage out of the sector: ‖(I − P) H u‖, formed as an explicit
            // residual — subtracting Σ(dot²) from ‖Hu‖² cancels catastrophically
            // and would report a spurious 2e-8 leak on an exactly invariant sector.
            let mut residual = hu.clone();
            for other in sector.iter() {
                let dot: f64 = other.iter().map(|&(i, coef)| coef * hu[i]).sum();
                for &(i, coef) in other {
                    residual[i] -= dot * coef;
                }
            }
            let leak = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
            let scale = hu.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
            assert!(
                leak < 1e-9 * scale,
                "the {} reflection sector must be H-invariant, leaked ‖(I−P)Hu‖ = {leak:e} \
                 against ‖Hu‖ = {scale:e}",
                if even { "even" } else { "odd" }
            );
        }
        for r in 0..k {
            for c in r + 1..k {
                let mean = 0.5 * (a[(r, c)] + a[(c, r)]);
                a[(r, c)] = mean;
                a[(c, r)] = mean;
            }
        }
        a
    }
}

/// The QYM reflection `R|n₀,n₁,n₂,n₃⟩ = (−1)^(n₀+n₁)|n₁,n₀,n₂,n₃⟩`.
fn apply_reflection(s: &QuantumState) -> QuantumState {
    let mut out = QuantumState::zero();
    for (outer, amp) in &s.components {
        for (inner, mult) in &outer.bosonic {
            let n0 = inner.modes.get(&0).copied().unwrap_or(0);
            let n1 = inner.modes.get(&1).copied().unwrap_or(0);
            let mut flipped = InnerBosonicState::vacuum();
            for (mode, n) in &inner.modes {
                let target = match *mode {
                    0 => 1,
                    1 => 0,
                    other => other,
                };
                if *n > 0 {
                    flipped.modes.insert(target, *n);
                }
            }
            let phase = if (n0 + n1) * mult % 2 == 0 { 1.0 } else { -1.0 };
            out.components.insert(
                OuterState {
                    bosonic: std::collections::BTreeMap::from([(flipped, *mult)]),
                    fermionic: std::collections::BTreeSet::new(),
                },
                *amp * Complex64::new(phase, 0.0),
            );
        }
    }
    out
}

fn ascending_spectrum(a: &DMatrix<f64>) -> Vec<f64> {
    let mut v: Vec<f64> = a
        .clone()
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .cloned()
        .collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v
}

fn dense_matvec(a: &DMatrix<f64>) -> impl Fn(&[f64]) -> Vec<f64> {
    let a = a.clone();
    move |x: &[f64]| {
        (0..a.nrows())
            .map(|i| (0..a.ncols()).map(|j| a[(i, j)] * x[j]).sum())
            .collect()
    }
}

/// The two solves put into the same certificate.
///
/// A Temple step only requires two things of its Ritz pair: `θ` is a Rayleigh
/// quotient of the sector (`θ ≥ λ₁`, true of any Ritz value) and the residual is
/// small enough that `θ + r < β`. Neither needs SIRK. On the *truncated*
/// operator the forward SIRK chain collapses (`rank 10 of m+1 = 15` at `g = 0`,
/// recorded in `qcd_mass_gap_certified.rs`), so its Ritz value stalls above
/// `λ₀` and does not improve with `m` — measured at `g = 1, N ≤ 6`: the even
/// excess is `3.71e-1` at `m = 12` and `3.99e-1` at `m = 40`, with the residual
/// flat at `~8e-1`. Lanczos on the same operator reaches machine precision in
/// 60 matvecs. That difference, not the certificate, is what moves the verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Solver {
    /// `solve_forward_sirk_truncated` at `m = 12` — the §6d baseline.
    SirkM12,
    /// Matvec-only Lanczos on the same `H_N`, [`CONVERGED_STEPS`] steps.
    Lanczos60,
}

impl Solver {
    const ALL: [Solver; 2] = [Solver::SirkM12, Solver::Lanczos60];

    fn label(self) -> &'static str {
        match self {
            Solver::SirkM12 => "SIRK m=12",
            Solver::Lanczos60 => "Lanczos 60",
        }
    }
}

/// Lanczos steps for the converged route — comfortably above the sector
/// dimensions on this grid (`≤ 200`), so the ground converges rather than
/// merely acquiring a small residual: measured `r ≤ 3e-3` at the worst cell and
/// `≤ 3e-11` at most.
const CONVERGED_STEPS: usize = 60;

/// Everything a row needs that is a property of the *cell* rather than of the
/// solve, so both routes are compared on identical exact references.
struct Cell {
    g: f64,
    n_max: u32,
    dim: usize,
    lambda_odd_1: f64,
    lambda_odd_2: f64,
    lambda_even_1: f64,
    lambda_even_2: f64,
    /// The matvec-only witness, kept for contrast: it is sound and far weaker,
    /// and having both on the same row is what shows the count is what fires.
    beta_exclusion: Option<f64>,
    exact_gap: Option<f64>,
    /// `λ₀(even) − λ₁(odd)`, present exactly when the odd ground is the lower
    /// of the two — the gap the swapped orientation certifies.
    exact_gap_swapped: Option<f64>,
    leak: f64,
}

/// One measured campaign row.
///
/// The exact columns are present so a refused or violated bound can be
/// attributed: which of Temple's hypotheses failed (`θᵒ₀ ∈ [λ₁, λ₂)`, `β ≤ λ₂`,
/// `θᵉ₀ ≥ λ₀(even)`) is visible in the row itself rather than inferred later.
struct CertifiedRow {
    solver: Solver,
    g: f64,
    n_max: u32,
    dim: usize,
    theta_odd: f64,
    residual_odd: f64,
    lambda_odd_1: f64,
    lambda_odd_2: f64,
    theta_even: f64,
    residual_even: f64,
    lambda_even_1: f64,
    lambda_even_2: f64,
    /// `λ₂` witness for each sector, from the sparse count (one per row: the
    /// bracket depends on the Ritz pair's residual, so it cannot be a property
    /// of the cell alone).
    beta_inertia: Option<f64>,
    beta_even_inertia: Option<f64>,
    beta_exclusion: Option<f64>,
    certified: Option<f64>,
    certified_swapped: Option<f64>,
    exact_gap: Option<f64>,
    exact_gap_swapped: Option<f64>,
    leak: f64,
}

impl Cell {
    #[allow(clippy::too_many_arguments)]
    fn row(
        &self,
        solver: Solver,
        theta_odd: f64,
        residual_odd: f64,
        theta_even: f64,
        residual_even: f64,
        beta_inertia: Option<f64>,
        beta_even_inertia: Option<f64>,
    ) -> CertifiedRow {
        CertifiedRow {
            solver,
            g: self.g,
            n_max: self.n_max,
            dim: self.dim,
            theta_odd,
            residual_odd,
            lambda_odd_1: self.lambda_odd_1,
            lambda_odd_2: self.lambda_odd_2,
            theta_even,
            residual_even,
            lambda_even_1: self.lambda_even_1,
            lambda_even_2: self.lambda_even_2,
            beta_inertia,
            beta_even_inertia,
            beta_exclusion: self.beta_exclusion,
            certified: beta_inertia.and_then(|beta| {
                certified_gap_lower_bound_with_exclusion(theta_odd, residual_odd, beta, theta_even)
            }),
            // The mirror image: Temple on the even sector bounds `λ₀(even)`
            // from below, and the odd side enters only through `λ₁(odd) ≤ θᵒ₀`,
            // which is Rayleigh–Ritz and needs no witness at all.
            certified_swapped: beta_even_inertia.and_then(|beta| {
                certified_gap_lower_bound_with_exclusion(theta_even, residual_even, beta, theta_odd)
            }),
            exact_gap: self.exact_gap,
            exact_gap_swapped: self.exact_gap_swapped,
            leak: self.leak,
        }
    }
}

/// Why a row did or did not certify — the distinction that matters, because the
/// three failure modes call for completely different work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// A valid positive lower bound on the gap of `H_N`.
    Certified,
    /// Temple applied and the bound is valid, but not positive.
    NotPositive,
    /// Temple's hypothesis failed: the witness sits at or below `θ`, or the
    /// residual ball reaches it. The honest output is no bound at all.
    Refused,
}

impl Verdict {
    fn of(certified: Option<f64>) -> Self {
        match certified {
            Some(lo) if lo > 0.0 => Verdict::Certified,
            Some(_) => Verdict::NotPositive,
            None => Verdict::Refused,
        }
    }
}

#[test]
fn exclusion_certificate_over_the_campaign_grid() {
    let m = 12;
    let mut rows = Vec::new();
    let mut crossing = 0usize;

    for g in [0.0, 0.5, 1.0, 2.0, 3.0] {
        let h = qcd_ym_hamiltonian(g);
        for n_max in [6u32, 8, 10] {
            let trunc = Truncated::build(&h, n_max);
            let even_sector = trunc.sector(true);
            let odd_sector = trunc.sector(false);
            let even_block = trunc.sector_matrix(true);
            let odd_block = trunc.sector_matrix(false);

            let even_spectrum = ascending_spectrum(&even_block);
            let odd_spectrum = ascending_spectrum(&odd_block);
            // The T6 expression is `λ(odd) − λ(even)`, so it is *the* gap only
            // while the odd ground sits below the even one. Where the ordering
            // flips inside `H_N` the cell has no gap in this orientation, and
            // `None` here is what stops such a row from being read as a failure
            // of the certificate.
            let exact_gap = if even_spectrum[0] < odd_spectrum[0] {
                Some(odd_spectrum[0] - even_spectrum[0])
            } else {
                crossing += 1;
                None
            };
            // The other orientation, live exactly where the first is not.
            let exact_gap_swapped = if even_spectrum[0] > odd_spectrum[0] {
                Some(even_spectrum[0] - odd_spectrum[0])
            } else {
                None
            };

            // The sparse sector matrices — the input to the inertia count, built
            // from the same columns the solves use. No diagonalization.
            let odd_columns = sector_columns(&trunc, &odd_sector);
            let even_columns = sector_columns(&trunc, &even_sector);
            // A `λ₂` witness for one sector from a Ritz pair: bisect the count.
            //
            // `β` must clear the residual ball (`θ + r < β`, Temple's own
            // hypothesis) and the count `#{λ < β} = 1` is exactly `β ≤ λ₂`. The
            // count is monotone in `β`, so the largest deciding `β` below `λ₂`
            // is reachable by bisection and is the one that minimises Temple's
            // `r²/(β − θ)` penalty. A `β` whose pivots cannot be decided ends
            // the search with `None` — refused, not guessed.
            let witness = |columns: &[Vec<(usize, f64)>], theta: f64, residual: f64| {
                let count_at = |beta: f64| inertia_below(columns, beta).map(|c| c.below);
                let scale = theta.abs().max(1.0);
                // `θ + r < β` is Temple's own hypothesis; a converged row has
                // `r ~ 1e-14`, so the first margin is what keeps the left end
                // out of the residual ball without landing on an eigenvalue.
                let ball = theta + residual * (1.0 + 1e-6);
                // An undecidable count is a property of *this* shift, not of the
                // sector: `β` within ~1e-8 of an eigenvalue leaves a pivot whose
                // sign cannot be decided. So a refusal is retried at a coarser
                // margin rather than ending the search.
                let mut base = None;
                for margin in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0] {
                    match count_at(ball + margin * scale) {
                        Some(1) => {
                            base = Some(ball + margin * scale);
                            break;
                        }
                        // Two or more below: this shift is already past `λ₂`, and
                        // every larger margin is worse.
                        Some(_) => break,
                        None => continue,
                    }
                }
                let base = base?;
                // Grow until the count says `λ₂` has been passed, then bisect
                // back down: the largest deciding `β` is the one that minimises
                // Temple's `r²/(β − θ)` penalty.
                let mut high = base;
                let mut step = 1e-4 * scale;
                let limit = 1e8 * scale;
                loop {
                    high += step;
                    match count_at(high) {
                        Some(1) | None => {
                            step *= 4.0;
                            if step > limit {
                                return None;
                            }
                        }
                        Some(_) => break,
                    }
                }
                let (mut best, mut high) = (base, high);
                for _ in 0..30 {
                    let mid = 0.5 * (best + high);
                    match count_at(mid) {
                        Some(1) => best = mid,
                        Some(_) => high = mid,
                        // Undecidable this close to `λ₂`: keep the last decided
                        // `β`, which is still a valid witness.
                        None => break,
                    }
                }
                Some(best)
            };
            // The same quantity from matvecs only, kept for the comparison the
            // sibling test pins.
            let odd_dense_matvec = dense_matvec(&odd_block);
            let beta_exclusion =
                DiagonalExclusion::from_matvec(&odd_dense_matvec, odd_block.nrows())
                    .and_then(|exclusion| exclusion.second_eigenvalue_lower_bound());

            let cell = Cell {
                g,
                n_max,
                dim: odd_block.nrows(),
                lambda_odd_1: odd_spectrum[0],
                lambda_odd_2: odd_spectrum[1],
                lambda_even_1: even_spectrum[0],
                lambda_even_2: even_spectrum[1],
                beta_exclusion,
                exact_gap,
                exact_gap_swapped,
                leak: trunc.leak,
            };

            // Route 1 — the §6d baseline: forward truncated SIRK.
            let odd_solve = solve_forward_sirk_truncated(
                &h,
                &r_odd_start(),
                &shifts(m),
                &best_device(),
                None,
                &opts(),
                n_max,
            )
            .expect("truncated odd solve");
            let even_solve = solve_forward_sirk_truncated(
                &h,
                &empty_vacuum(),
                &shifts(m),
                &best_device(),
                None,
                &opts(),
                n_max,
            )
            .expect("truncated even solve");
            let odd_cert = certified_ground_state(&odd_solve).expect("odd certificate");
            let even_cert = certified_ground_state(&even_solve).expect("even certificate");
            rows.push(cell.row(
                Solver::SirkM12,
                odd_cert.value,
                odd_cert.residual,
                even_cert.value,
                even_cert.residual,
                witness(&odd_columns, odd_cert.value, odd_cert.residual),
                witness(&even_columns, even_cert.value, even_cert.residual),
            ));

            // Route 2 — matvec-only Lanczos on the same `H_N`. The starts are
            // the same states, projected into sector coordinates, with the
            // containment residual asserted so the route cannot be running on a
            // start that leaks out of its sector.
            let odd_start = trunc.coordinates(&odd_sector, &r_odd_start());
            let even_start = trunc.coordinates(&even_sector, &empty_vacuum());
            assert!(
                odd_start.1 < 1e-9 && even_start.1 < 1e-9,
                "the sector starts must lie inside their sectors: R-odd leak {:.2e}, vacuum leak {:.2e}",
                odd_start.1,
                even_start.1
            );
            let odd_mv = trunc.sector_matvec(&odd_sector);
            let even_mv = trunc.sector_matvec(&even_sector);
            let odd_ritz = lanczos_ritz(&odd_mv, odd_sector.len(), &odd_start.0, CONVERGED_STEPS)
                .expect("odd Lanczos");
            let even_ritz =
                lanczos_ritz(&even_mv, even_sector.len(), &even_start.0, CONVERGED_STEPS)
                    .expect("even Lanczos");
            assert!(
                odd_ritz.orthogonality < 1e-10 && even_ritz.orthogonality < 1e-10,
                "the Lanczos bases must be orthonormal at g = {g}, N <= {n_max}: {:.1e} / {:.1e}",
                odd_ritz.orthogonality,
                even_ritz.orthogonality
            );
            let (theta_odd, residual_odd) = odd_ritz.ground().expect("non-empty");
            let (theta_even, residual_even) = even_ritz.ground().expect("non-empty");
            rows.push(cell.row(
                Solver::Lanczos60,
                theta_odd,
                residual_odd,
                theta_even,
                residual_even,
                witness(&odd_columns, theta_odd, residual_odd),
                witness(&even_columns, theta_even, residual_even),
            ));
        }
    }

    let header = |solver: Solver| {
        println!("\n{} — {}", solver.label(), "H_N rows, m = 12");
        println!(
            "{:>4} {:>3} {:>4} {:>5} {:>12} {:>9} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} \
             {:>10}",
            "g",
            "N",
            "dim",
            "leak",
            "θᵒ₀",
            "rᵒ",
            "θᵉ₀",
            "λ₁ odd",
            "λ₂ odd",
            "β inertia",
            "β excl",
            "certified",
            "exact"
        );
    };
    for solver in Solver::ALL {
        header(solver);
        for row in rows.iter().filter(|r| r.solver == solver) {
            println!(
                "{:>4} {:>3} {:>4} {:>5.0e} {:>12.9} {:>9.1e} {:>12.9} {:>12.9} {:>12.9} \
                 {:>12.9} {:>12.3} {:>12} {:>10}",
                row.g,
                row.n_max,
                row.dim,
                row.leak,
                row.theta_odd,
                row.residual_odd,
                row.theta_even,
                row.lambda_odd_1,
                row.lambda_odd_2,
                row.beta_inertia.unwrap_or(f64::NAN),
                row.beta_exclusion.unwrap_or(f64::NAN),
                row.certified
                    .map(|v| format!("{v:.9}"))
                    .unwrap_or_else(|| "refused".to_string()),
                row.exact_gap
                    .map(|v| format!("{v:.9}"))
                    .unwrap_or_else(|| "—".to_string()),
            );
        }
    }

    // The swapped orientation: the even side templed against its own `λ₂`
    // witness, the odd side plain. This is the table the crossing cells appear
    // in, side by side with the gap they actually have.
    println!("\nswapped — even side templed, odd side plain (Lanczos 60)");
    println!(
        "{:>4} {:>3} {:>12} {:>9} {:>12} {:>12} {:>12} {:>12}",
        "g", "N", "θᵉ₀", "rᵉ", "λ₂ even", "β even", "certified⇄", "exact⇄"
    );
    for row in rows.iter().filter(|r| r.solver == Solver::Lanczos60) {
        println!(
            "{:>4} {:>3} {:>12.9} {:>9.1e} {:>12.9} {:>12.9} {:>12} {:>12}",
            row.g,
            row.n_max,
            row.theta_even,
            row.residual_even,
            row.lambda_even_2,
            row.beta_even_inertia.unwrap_or(f64::NAN),
            row.certified_swapped
                .map(|v| format!("{v:.9}"))
                .unwrap_or_else(|| "refused".to_string()),
            row.exact_gap_swapped
                .map(|v| format!("{v:.9}"))
                .unwrap_or_else(|| "—".to_string()),
        );
    }

    // Soundness, checked row by row **after** the tables are printed, so a
    // failure arrives with its own numbers beside it.
    for row in &rows {
        let label = format!("{} g = {}, N <= {}", row.solver.label(), row.g, row.n_max);
        if let Some(beta) = row.beta_inertia {
            // The count said "one eigenvalue below β", so β may not exceed the
            // exact λ₂ — this is the assertion that the count is right rather
            // than plausible.
            assert!(
                beta <= row.lambda_odd_2 + 1e-9,
                "{label}: inertia β = {beta} exceeds the exact λ₂ = {}",
                row.lambda_odd_2
            );
            assert!(
                beta > row.theta_odd,
                "{label}: the witness {beta} must sit above θᵒ₀ = {} for Temple to apply",
                row.theta_odd
            );
        }
        if let Some(beta) = row.beta_exclusion {
            assert!(
                beta <= row.lambda_odd_2 + 1e-9,
                "{label}: exclusion β = {beta} exceeds the exact λ₂ = {}",
                row.lambda_odd_2
            );
        }
        if let (Some(lo), Some(exact)) = (row.certified, row.exact_gap) {
            assert!(
                lo <= exact + 1e-9,
                "{label}: certified bound {lo} exceeds the exact gap {exact}"
            );
        }
        // The swapped orientation is held to the same standard: its witness is
        // a `λ₂(even)` bound, and its bound may not exceed the exact swapped
        // gap. When it produces a number, `θᵉ₀ < β ≤ λ₂(even)` must hold, which
        // is exactly Temple's `θ ∈ [λ₁, λ₂)` hypothesis for the even sector.
        if let Some(beta) = row.beta_even_inertia {
            assert!(
                beta <= row.lambda_even_2 + 1e-9,
                "{label}: even β = {beta} exceeds the exact λ₂(even) = {}",
                row.lambda_even_2
            );
        }
        if row.certified_swapped.is_some() {
            assert!(
                row.theta_even < row.lambda_even_2 + 1e-9,
                "{label}: swapped certificate with θᵉ₀ = {} at or above λ₂(even) = {}",
                row.theta_even,
                row.lambda_even_2
            );
        }
        if let (Some(lo), Some(exact)) = (row.certified_swapped, row.exact_gap_swapped) {
            assert!(
                lo <= exact + 1e-9,
                "{label}: swapped bound {lo} exceeds the exact swapped gap {exact}"
            );
        }
        // Rayleigh–Ritz, both sides: any Ritz value is at or above its sector's
        // ground, and the even side is consumed only as an upper bound on `λ₀`
        // — which is why the even side needs no witness at all, only a value
        // that is close enough to be useful.
        assert!(
            row.theta_odd >= row.lambda_odd_1 - 1e-8,
            "{label}: θᵒ₀ = {} must not sit below the odd ground {}",
            row.theta_odd,
            row.lambda_odd_1
        );
        assert!(
            row.theta_even >= row.lambda_even_1 - 1e-8,
            "{label}: θᵉ₀ = {} must not sit below the even ground {}",
            row.theta_even,
            row.lambda_even_1
        );
    }

    for solver in Solver::ALL {
        println!("\n{} verdicts:", solver.label());
        for row in rows.iter().filter(|r| r.solver == solver) {
            let verdict = Verdict::of(row.certified);
            let why = match verdict {
                Verdict::Certified => format!(
                    "certified {:.9} (exact {:.9})",
                    row.certified.unwrap(),
                    row.exact_gap.unwrap_or(f64::NAN)
                ),
                Verdict::NotPositive => format!(
                    "bound {:.9} ≤ 0 — even side θᵉ₀ = {:.9} sits {:+.3e} from λ₀(even) = {:.9}, \
                     odd side θᵒ₀ = {:.9} sits {:+.3e} from λ₁(odd) = {:.9}",
                    row.certified.unwrap(),
                    row.theta_even,
                    row.theta_even - row.lambda_even_1,
                    row.lambda_even_1,
                    row.theta_odd,
                    row.theta_odd - row.lambda_odd_1,
                    row.lambda_odd_1
                ),
                Verdict::Refused => format!(
                    "refused — β = {:.9} vs θᵒ₀ = {:.9}, rᵒ = {:.3e} (residual ball reaches λ₂)",
                    row.beta_inertia.unwrap_or(f64::NAN),
                    row.theta_odd,
                    row.residual_odd,
                ),
            };
            println!("  g = {}, N <= {}: {why}", row.g, row.n_max);
        }
    }

    let tally = |solver: Solver| {
        let of = |v: Verdict| {
            rows.iter()
                .filter(|r| r.solver == solver && Verdict::of(r.certified) == v)
                .count()
        };
        (
            of(Verdict::Certified),
            of(Verdict::NotPositive),
            of(Verdict::Refused),
        )
    };
    // The crossing set, pinned because it is *not* a monotone function of `N`:
    // `g = 2` crosses at `N <= 8` and not at `N <= 10`; `g = 1` does not cross at
    // `N <= 8` and does at `N <= 10`. So a crossing is an oscillation of the
    // truncation rather than a sign that a deeper `N` will cure it, and the pin
    // is what keeps that finding from being quietly re-derived later.
    let crossing_cells: Vec<(f64, u32)> = rows
        .iter()
        .filter(|r| r.solver == Solver::Lanczos60 && r.exact_gap.is_none())
        .map(|r| (r.g, r.n_max))
        .collect();
    assert_eq!(
        crossing_cells,
        vec![(0.5, 8), (1.0, 10), (2.0, 8), (3.0, 6), (3.0, 8)],
        "the crossing cells must be exactly the measured set; a change that moves them means \
         the non-monotonicity finding has to be re-measured"
    );

    println!();
    println!(
        "cells with the odd ground BELOW the even one inside H_N (T6's expression is then the \
         negative of their gap): {crossing}"
    );
    println!(
        "  {crossing_cells:?} — non-monotone in N, so the truncated ground's parity is not \
         settling with depth"
    );
    for solver in Solver::ALL {
        let (c, n, r) = tally(solver);
        println!(
            "{}: {c} certified, {n} valid-but-not-positive, {r} refused (of {})",
            solver.label(),
            rows.len() / Solver::ALL.len()
        );
    }

    // Every certified row must have an exact reference it was checked against.
    assert!(
        rows.iter()
            .filter(|r| r.certified.map(|v| v > 0.0).unwrap_or(false))
            .all(|r| r.exact_gap.is_some()),
        "a certified row must have an exact reference to have been checked against"
    );
    assert!(
        tally(Solver::SirkM12).0 > 0,
        "the dense witness must certify somewhere on the SIRK grid — every row refusing would \
         mean the certificate stopped working, not that the gap vanished"
    );

    // **The measured unlock.** With both sector grounds resolved, exactly the
    // cells whose truncated ordering is T6's own certify: `θᵒ₀ = λ₁(odd)` and
    // `θᵉ₀ = λ₀(even)` leave only Temple's `r²/(β−θ)` slack, which at these
    // residuals is `O(1e-14)`. Everything else on the grid is *not* a
    // convergence problem — `crossing` cells have the odd ground below the even
    // one inside `H_N`, so T6's expression is the negative of their gap and no
    // amount of accuracy can make it positive.
    let oriented: Vec<&CertifiedRow> = rows
        .iter()
        .filter(|r| r.solver == Solver::Lanczos60 && r.exact_gap.is_some())
        .collect();
    assert!(
        !oriented.is_empty(),
        "the grid must contain cells whose truncated ordering matches T6's"
    );
    for row in &oriented {
        let lo = row.certified.unwrap_or_else(|| {
            panic!(
                "g = {}, N <= {}: Temple refused a converged solve",
                row.g, row.n_max
            )
        });
        assert!(
            lo > 0.0,
            "with both sector grounds resolved the bound must be positive at g = {}, N <= {}: \
             lo = {lo}, exact {:.9}",
            row.g,
            row.n_max,
            row.exact_gap.unwrap()
        );
    }
    let (sirk_certified, _, _) = tally(Solver::SirkM12);
    let (converged_certified, _, _) = tally(Solver::Lanczos60);
    assert!(
        converged_certified > sirk_certified,
        "the converged route must certify strictly more rows than the m=12 baseline \
         ({converged_certified} vs {sirk_certified})"
    );
    assert_eq!(
        converged_certified,
        oriented.len(),
        "on this grid the converged route certifies exactly the oriented cells ({converged_certified} \
         vs {}), all {} crossing cells aside — a difference would mean convergence, not \
         orientation, is the remaining limit",
        oriented.len(),
        crossing
    );

    // **The swapped orientation is the complement, not a second chance on the
    // same cells.** Where the odd ground is the lower one, the same assembly
    // with the sectors exchanged must certify the gap those cells actually have
    // — and in T6's own cells it must certify nothing, because there the gap has
    // the other sign.
    let swapped_oriented: Vec<&CertifiedRow> = rows
        .iter()
        .filter(|r| r.solver == Solver::Lanczos60 && r.exact_gap_swapped.is_some())
        .collect();
    assert_eq!(
        swapped_oriented.len(),
        crossing,
        "the swapped orientation must be live on exactly the {} crossing cells",
        crossing
    );
    assert!(
        !swapped_oriented.is_empty(),
        "every cell refusing in both orientations would mean the assembly stopped working"
    );
    for row in &swapped_oriented {
        // What makes the swapped bound a statement about the *global* gap rather
        // than about two sector grounds: no odd level lies between them, so
        // `λ₁(odd)` and `λ₀(even)` really are the two lowest levels of `H_N`.
        // (The even sector's own second level is above its ground by definition.)
        assert!(
            row.lambda_odd_2 > row.lambda_even_1,
            "g = {}, N <= {}: λ₂(odd) = {} is not above λ₀(even) = {} — the swapped bound \
             would then separate two ground states with a level in between, and could not be \
             read as the gap of H_N",
            row.g,
            row.n_max,
            row.lambda_odd_2,
            row.lambda_even_1
        );
        let lo = row.certified_swapped.unwrap_or_else(|| {
            panic!(
                "g = {}, N <= {}: the even side refused (β = {:?}, θᵉ₀ = {}, rᵉ = {:.3e}, \
                 λ₂(even) = {})",
                row.g,
                row.n_max,
                row.beta_even_inertia,
                row.theta_even,
                row.residual_even,
                row.lambda_even_2
            )
        });
        assert!(
            lo > 0.0,
            "swapped bound at g = {}, N <= {} must be positive: lo = {lo}, exact {:.9}",
            row.g,
            row.n_max,
            row.exact_gap_swapped.unwrap()
        );
    }
    for row in rows
        .iter()
        .filter(|r| r.solver == Solver::Lanczos60 && r.exact_gap.is_some())
    {
        assert_ne!(
            Verdict::of(row.certified_swapped),
            Verdict::Certified,
            "g = {}, N <= {}: the swapped orientation must not certify in T6's own cells \
             (a positive swapped bound there would exceed the exact negative gap {:.9})",
            row.g,
            row.n_max,
            row.exact_gap.unwrap()
        );
    }
    let swapped_certified = swapped_oriented
        .iter()
        .filter(|r| Verdict::of(r.certified_swapped) == Verdict::Certified)
        .count();
    assert_eq!(
        converged_certified + swapped_certified,
        rows.len() / Solver::ALL.len(),
        "every cell must carry a certified gap in one orientation or the other \
         ({converged_certified} + {swapped_certified} of {})",
        rows.len() / Solver::ALL.len()
    );
    println!(
        "\nwith the sectors swapped: {swapped_certified} more certified, so {}/{} cells carry a \
         certified gap in one orientation or the other",
        converged_certified + swapped_certified,
        rows.len() / Solver::ALL.len()
    );
}

/// The sparse witness is sound at a truncation the dense one cannot reach — and
/// too weak to certify. Both halves are the finding, so both are measured.
#[test]
fn the_sparse_witness_is_sound_but_does_not_certify() {
    let g = 1.0;
    let n_max = 12;
    let h = qcd_ym_hamiltonian(g);
    // No sector reduction here: the point is to reach a size where dense
    // diagonalization is out of the question, and the Krylov start stays in the
    // odd sector anyway. Only the sparse representation is needed.
    let trunc = Truncated::build(&h, n_max);
    let n = trunc.dim();
    println!("N <= {n_max}: truncated dimension {n} (no sector reduction)");

    let matvec = trunc.matvec();
    let exclusion = DiagonalExclusion::from_matvec(&matvec, n).expect("measurable");
    let beta = exclusion.second_eigenvalue_lower_bound().expect("n ≥ 2");

    // Lanczos on the same operator: matvec-only, and the Ritz pair it returns is
    // the input Temple takes at this size.
    let start = trunc
        .basis
        .iter()
        .map(|s| QuantumState::inner_product(s, &r_odd_start()).re)
        .collect::<Vec<f64>>();
    let ritz = lanczos_ritz(&matvec, n, &start, 12).expect("twelve Lanczos steps");
    assert!(
        ritz.orthogonality < 1e-10,
        "Lanczos basis must be orthonormal, measured defect {}",
        ritz.orthogonality
    );
    let (theta, residual) = ritz.ground().expect("non-empty");
    println!(
        "Lanczos: θ = {theta:.9}, r = {residual:.2e}, orthogonality {:.1e}, diag β = {beta:.3}",
        ritz.orthogonality
    );

    // Soundness is not checkable by dense diagonalization here (that is the
    // point), but it is checkable against the *smaller* truncation in the one
    // direction min–max allows: compressions raise eigenvalues, so
    // λ₂(H_6) ≥ λ₂(H_12) — an upper bound, never a lower one. Recording the
    // direction is what stops the nesting hierarchy from being mistaken for a
    // witness.
    let small = Truncated::build(&h, 6);
    let small_spectrum = ascending_spectrum(&small.sector_matrix(false));
    println!(
        "nesting: λ₂(H_6) = {:.9}, so the hierarchy bounds λ₂(H_12) from ABOVE",
        small_spectrum[1]
    );

    // The measured verdict: the bound is already below the ground, so Temple
    // refuses and nothing is certified. This is pinned rather than merely
    // printed — if a future change makes the sparse route fire here, the test
    // that records the negative result has to be rewritten, not deleted.
    assert!(
        beta <= theta,
        "β = {beta} now clears θ = {theta}: the sparse witness certifies at N <= {n_max}, so \
         the negative result this test records no longer holds"
    );
    assert!(
        certified_gap_lower_bound_with_exclusion(theta, residual, beta, theta).is_none(),
        "Temple must refuse a β at or below θ rather than return a bound"
    );
    println!(
        "sparse verdict: refused (β = {beta:.3} ≤ θ = {theta:.3}; the dense route would need \
         O(n³) at n = {n})"
    );
}

/// The count-free witness measured against the window it has to fit.
///
/// Colouring the sparsity graph into matchings replaces the Frobenius split's
/// `√(#edges)` with `#classes`, so the slack becomes degree-bounded instead of
/// `√n`-bounded. That is a change in *growth*, not in magnitude, and this test
/// records both halves on rows where the exact `λ₂` exists (so the bound is
/// checked, not merely argued) and on the row the dense route cannot reach.
#[test]
fn the_matching_split_does_not_reach_the_window() {
    let g = 1.0;
    // The smallest size, kept for the growth comparison below.
    let mut small: Option<(f64, f64)> = None;
    println!("g = {g}: matching split on the odd sector, where λ₂ is known");
    println!(
        "{:>3} {:>5} {:>10} {:>6} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "N", "n", "d₂", "class", "‖V‖_F", "colour slack", "β colour", "β diagonal", "θ", "λ₂ exact"
    );
    // Rows where the exact λ₂ exists, so the new bound is *checked* sound rather
    // than only argued sound.
    for n_max in [6u32, 8] {
        let h = qcd_ym_hamiltonian(g);
        let trunc = Truncated::build(&h, n_max);
        let odd_block = trunc.sector_matrix(false);
        let spectrum = ascending_spectrum(&odd_block);
        let n = odd_block.nrows();
        let mv = dense_matvec(&odd_block);
        let matching = MatchingExclusion::from_matvec(&mv, n).expect("measurable");
        let beta_colour = matching.second_eigenvalue_lower_bound().expect("n >= 2");
        let beta_diagonal = DiagonalExclusion::from_matvec(&mv, n)
            .and_then(|e| e.second_eigenvalue_lower_bound())
            .expect("n >= 2");
        let slack: f64 = matching.class_widths.iter().sum();
        println!(
            "{n_max:>3} {n:>5} {:>10.6} {:>6} {:>11.3} {:>11.3} {:>11.6} {:>11.3} {:>11.6} \
             {:>11.6}",
            matching.diagonal[1],
            matching.class_count(),
            matching.offdiagonal_frobenius,
            slack,
            beta_colour,
            beta_diagonal,
            spectrum[0],
            spectrum[1]
        );
        assert!(
            beta_colour <= spectrum[1] + 1e-9,
            "N <= {n_max}: β_colour = {beta_colour} exceeds the exact λ₂ = {} — the matching \
             split is not sound",
            spectrum[1]
        );
        // Whether the colouring beats the single Frobenius term is measured, not
        // assumed: it trades `√(#edges)` for `#classes`, and which of the two is
        // smaller depends on how dense the coupling graph is at this size.
        println!(
            "  N <= {n_max}: β colour − β diagonal = {:+.3} (positive = the colouring is tighter)",
            beta_colour - beta_diagonal
        );
        if n_max == 6 {
            small = Some((matching.offdiagonal_frobenius, slack));
        }
    }

    // The row the dense route cannot reach: no sector reduction, so the witness
    // is the only thing standing between this size and a certificate.
    let n_max = 12u32;
    let h = qcd_ym_hamiltonian(g);
    let trunc = Truncated::build(&h, n_max);
    let n = trunc.dim();
    let matvec = trunc.matvec();
    let matching = MatchingExclusion::from_matvec(&matvec, n).expect("measurable");
    let beta_colour = matching.second_eigenvalue_lower_bound().expect("n >= 2");
    let beta_diagonal = DiagonalExclusion::from_matvec(&matvec, n)
        .and_then(|e| e.second_eigenvalue_lower_bound())
        .expect("n >= 2");
    let start = trunc
        .basis
        .iter()
        .map(|s| QuantumState::inner_product(s, &r_odd_start()).re)
        .collect::<Vec<f64>>();
    let ritz = lanczos_ritz(&matvec, n, &start, 12).expect("twelve Lanczos steps");
    let (theta, _residual) = ritz.ground().expect("non-empty");
    let slack: f64 = matching.class_widths.iter().sum();
    let admissible = matching.diagonal[1] - theta;
    println!(
        "{n_max:>3} {n:>5} {:>10.6} {:>6} {:>11.3} {:>11.3} {:>11.6} {:>11.3} {:>11.6} {:>11}",
        matching.diagonal[1],
        matching.class_count(),
        matching.offdiagonal_frobenius,
        slack,
        beta_colour,
        beta_diagonal,
        theta,
        "—"
    );
    println!(
        "N <= {n_max}: the certificate needs the slack inside d₂ − θ = {admissible:.3}, and the \
         colouring delivers {slack:.1} in {} classes — flat in N where ‖V‖_F is {:.0}. ",
        matching.class_count(),
        matching.offdiagonal_frobenius
    );
    assert!(
        beta_colour <= theta,
        "β_colour = {beta_colour} now clears θ = {theta} at N <= {n_max}: the count-free route \
         certifies there, so the negative result this test records no longer holds"
    );
    assert!(
        certified_gap_lower_bound_with_exclusion(theta, _residual, beta_colour, theta).is_none(),
        "Temple must refuse a β at or below θ rather than return a bound"
    );
    assert!(
        slack > admissible,
        "the colouring's slack ({slack}) must still exceed the whole admissible budget d₂ − θ = \
         {admissible}; if it does not, the certificate is one bound away from firing and this \
         negative result has to be re-measured"
    );
    // **The growth result.** At the smallest size the colouring is *looser* than
    // one Frobenius term (it charges the largest entry of each of ~13 classes
    // instead of the `l2` norm of ~600 edges); by `N ≤ 12` it is tighter, and
    // the gap between them widens with `n` because `‖V‖_F` grows like `√(#edges)`
    // while the class count grows like the graph's degree. Both directions are
    // pinned so the trade cannot be reported as a win without its price.
    let (frobenius_small, slack_small) = small.expect("N <= 6 row");
    assert!(
        slack_small > frobenius_small,
        "at N <= 6 the colouring ({slack_small}) is expected to be looser than ‖V‖_F \
         ({frobenius_small}); a tighter colouring would mean the graph is already sparse enough \
         for the matchings to pay off"
    );
    assert!(
        slack < matching.offdiagonal_frobenius,
        "at N <= {n_max} the colouring ({slack}) must be tighter than ‖V‖_F ({}), which is the \
         whole reason to pay for the colouring",
        matching.offdiagonal_frobenius
    );
    // The slack is degree-bounded, so it is at worst the whole off-diagonal
    // Frobenius norm (never worse than the diagonal split) and — the point of
    // the colouring — it grows far more slowly than `√n` does. Both halves are
    // pinned so this test cannot rot into printing.
    assert!(
        slack <= matching.offdiagonal_frobenius,
        "the colour slack ({slack}) must not exceed ‖V‖_F ({}) — that would make the split \
         strictly worse than the diagonal one",
        matching.offdiagonal_frobenius
    );
}

/// The odd/even sector of `H_N` as a sparse matrix, built from the full sparse
/// columns — the input the inertia counter needs, and the only linear algebra
/// here that is not a matvec.
fn sector_columns(trunc: &Truncated, sector: &[Vec<(usize, f64)>]) -> Vec<Vec<(usize, f64)>> {
    let n_full = trunc.dim();
    // Which sector vectors each full basis state participates in. The sector
    // basis is orthonormal with at most two states per vector, so this is the
    // exact inverse map.
    let mut members: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_full];
    for (b, u) in sector.iter().enumerate() {
        for &(j, c) in u {
            members[j].push((b, c));
        }
    }
    let mut columns = Vec::with_capacity(sector.len());
    for u in sector.iter() {
        // z = H_N u, then column b of `Uᵀ H U` is `Uᵀ z`.
        let mut z: HashMap<usize, f64> = HashMap::new();
        for &(i, c) in u {
            for &(j, v) in &trunc.columns[i] {
                *z.entry(j).or_insert(0.0) += c * v;
            }
        }
        let mut acc: HashMap<usize, f64> = HashMap::new();
        for (&j, &zj) in z.iter() {
            for &(b, c) in &members[j] {
                *acc.entry(b).or_insert(0.0) += c * zj;
            }
        }
        let mut column: Vec<(usize, f64)> =
            acc.into_iter().filter(|&(_, v)| v.abs() > 1e-14).collect();
        column.sort_by_key(|&(i, _)| i);
        columns.push(column);
    }
    columns
}

/// Fill-reducing ordering versus bandwidth-reducing ordering, on the same
/// matrices, out to the size where the elimination stops being cheap.
///
/// The count is invariant under both (a permutation preserves inertia), so the
/// comparison is purely fill and wall time — and the count cannot be used to pick
/// a winner by accident, because it is checked to agree first.
#[test]
fn nested_dissection_cuts_the_fill_of_the_inertia_count() {
    use fock_sirk::{Ordering, inertia_below_with};
    use std::time::Instant;

    let g = 1.0;
    println!("g = {g}: inertia count under RCM vs nested dissection, odd sector, β = θ₀ + 0.5");
    println!(
        "{:>3} {:>6} {:>10} {:>10} {:>9} {:>9}",
        "N", "n", "fill RCM", "fill ND", "secs RCM", "secs ND"
    );
    let mut rows: Vec<(
        u32,
        usize,
        Option<fock_sirk::InertiaCount>,
        Option<fock_sirk::InertiaCount>,
    )> = Vec::new();
    for n_max in [6u32, 8, 10, 12, 14, 16, 18, 20, 22, 24] {
        let h = qcd_ym_hamiltonian(g);
        let trunc = Truncated::build(&h, n_max);
        let odd_sector = trunc.sector(false);
        let odd_columns = sector_columns(&trunc, &odd_sector);
        let n = odd_columns.len();

        // A shift above the ground by half a unit: the count is then ≥ 1 on both
        // orderings, and large enough that no pivot sits near the boundary — the
        // comparison is about fill, not about a marginal sign.
        let odd_mv = trunc.sector_matvec(&odd_sector);
        let start = trunc.coordinates(&odd_sector, &r_odd_start()).0;
        let ritz = lanczos_ritz(&odd_mv, n, &start, 60.min(n)).expect("Lanczos");
        let (theta, _) = ritz.ground().expect("non-empty");
        let beta = theta + 0.5;

        let clock = Instant::now();
        let rcm = inertia_below_with(&odd_columns, beta, Ordering::ReverseCuthillMckee);
        let secs_rcm = clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let nd = inertia_below_with(&odd_columns, beta, Ordering::NestedDissection);
        let secs_nd = clock.elapsed().as_secs_f64();

        // The count is only comparable where both decide — and each ordering
        // stops at its own size, so the refusal is recorded rather than skipped.
        let label = |r: &Option<fock_sirk::InertiaCount>| match r {
            Some(c) => format!("{:>10}", c.fill),
            None => format!("{:>10}", "refused"),
        };
        println!(
            "{n_max:>3} {n:>6} {} {} {secs_rcm:>9.2} {secs_nd:>9.2}",
            label(&rcm),
            label(&nd)
        );
        rows.push((n_max, n, rcm, nd));
    }

    let decided = |r: &Option<fock_sirk::InertiaCount>| r.is_some();
    // (1) Where both decide, the answer is the same count and nested dissection
    // has the smaller fill. If either half ever fails, the comparison in the
    // docs is wrong and has to be rewritten, not the assertion relaxed.
    let both: Vec<_> = rows
        .iter()
        .filter(|r| decided(&r.2) && decided(&r.3))
        .collect();
    assert!(
        both.len() >= 6,
        "at least six sizes must be decided by both orderings, measured {}",
        both.len()
    );
    for (n_max, _, rcm, nd) in &both {
        let (rcm, nd) = (rcm.as_ref().unwrap(), nd.as_ref().unwrap());
        assert_eq!(
            (rcm.below, rcm.pivots),
            (nd.below, nd.pivots),
            "N <= {n_max}: RCM counts {} below β in {} pivots, nested dissection {} in {} — the \
             ordering must not change the inertia",
            rcm.below,
            rcm.pivots,
            nd.below,
            nd.pivots
        );
        assert!(
            nd.fill < rcm.fill,
            "N <= {n_max}: nested dissection's fill ({}) must stay below RCM's ({}), which is the \
             whole reason to pay for the recursion",
            nd.fill,
            rcm.fill
        );
    }

    // (2) The wall, per ordering: the last size that decides and the first that
    // does not. These are measurements of the error bound's reach, so a change
    // in either direction is a change in what can be claimed.
    let last_decided = |col: usize| -> u32 {
        rows.iter()
            .filter(|r| {
                let c = if col == 2 { &r.2 } else { &r.3 };
                decided(c)
            })
            .map(|r| r.0)
            .max()
            .expect("some size decides")
    };
    let rcm_wall = last_decided(2);
    let nd_wall = last_decided(3);
    println!("last size that decides: RCM N <= {rcm_wall}, nested dissection N <= {nd_wall}");
    // Measured walls, and the reason the assertion is now on the grid's own
    // boundary: with a one-term value the bound compounded `u ≈ 1e-16` per step
    // and both orderings refused (RCM at 20, nested dissection at 18). A
    // two-term value injects `u²` instead, so nothing in this grid refuses any
    // more — the count's reach is past the largest truncation the campaign
    // builds, and what limits it next is time (`the_count_reaches_past_the_grid`).
    assert_eq!(
        (rcm_wall, nd_wall),
        (24, 24),
        "the measured walls moved: RCM decides through N <= {rcm_wall}, nested dissection through \
         N <= {nd_wall} (were 18 and 16 with a one-term value; the grid ends at 24)"
    );

    // (3) Fill is not what binds: at the largest size the fill is a few percent
    // of `n`, so the elimination is nowhere near a memory limit — the cost that
    // does grow is the elimination's own `fill²` work.
    let largest = rows
        .iter()
        .filter(|r| decided(&r.2))
        .max_by_key(|r| r.1)
        .expect("a decided row");
    let fill = largest.2.as_ref().unwrap().fill;
    println!(
        "at N <= {}, n = {}: fill {fill} ({:.2}% of n), so fill is not the binding limit",
        largest.0,
        largest.1,
        100.0 * fill as f64 / largest.1 as f64
    );
    assert!(
        fill * 4 < largest.1,
        "the measured fill ({fill}) on n = {} must stay well under a quarter of the matrix — if \
         it does not, the wall is a fill effect and this explanation is wrong",
        largest.1
    );
}

/// The same comparison one grid step past the campaign's own: the point of the
/// two-term value is that the bound no longer decides the reach. `#[ignore]`d
/// only because the two truncations cost ~50 s together; the assertions are the
/// same ones the campaign applies, so a regression here is a real refusal.
#[test]
#[ignore = "~50 s: run with `cargo test --release -p fock_sirk --test mass_gap_exclusion_campaign -- --ignored`"]
fn the_count_reaches_past_the_grid() {
    use fock_sirk::{Ordering, inertia_below_with};
    use std::time::Instant;

    let g = 1.0;
    println!("g = {g}: inertia count past the campaign grid, odd sector, β = θ₀ + 0.5");
    println!(
        "{:>3} {:>6} {:>9} {:>8} {:>9} {:>8}",
        "N", "n", "fill RCM", "secs", "fill ND", "secs"
    );
    for n_max in [26u32, 28] {
        let h = qcd_ym_hamiltonian(g);
        let trunc = Truncated::build(&h, n_max);
        let odd_sector = trunc.sector(false);
        let odd_columns = sector_columns(&trunc, &odd_sector);
        let n = odd_columns.len();
        let odd_mv = trunc.sector_matvec(&odd_sector);
        let start = trunc.coordinates(&odd_sector, &r_odd_start()).0;
        let ritz = lanczos_ritz(&odd_mv, n, &start, 60.min(n)).expect("Lanczos");
        let (theta, _) = ritz.ground().expect("non-empty");
        let beta = theta + 0.5;

        let clock = Instant::now();
        let rcm = inertia_below_with(&odd_columns, beta, Ordering::ReverseCuthillMckee)
            .expect("RCM must decide here — a refusal is the wall moving back");
        let secs_rcm = clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let nd = inertia_below_with(&odd_columns, beta, Ordering::NestedDissection)
            .expect("nested dissection must decide here");
        let secs_nd = clock.elapsed().as_secs_f64();
        println!(
            "{n_max:>3} {n:>6} {:>9} {secs_rcm:>8.2} {:>9} {secs_nd:>8.2}",
            rcm.fill, nd.fill
        );
        assert_eq!(
            (rcm.below, rcm.pivots),
            (nd.below, nd.pivots),
            "N <= {n_max}: both orderings must count the same inertia"
        );
        assert!(
            rcm.below >= 1,
            "N <= {n_max}: β sits above the ground, so the count cannot be 0"
        );
    }
}

/// The gap of `H_N` bracketed from both sides with **no diagonalization**: a
/// sparse inertia count supplies each sector's `λ₂` witness, and the two Temple
/// bounds turn the Ritz pairs into `[lower, upper]`. This is what the dense route
/// spends `O(n³)` to produce, and the only step that needed the count.
#[test]
fn the_inertia_count_brackets_the_gap_without_diagonalizing() {
    use fock_sirk::inertia_below;
    use std::time::Instant;

    let g = 1.0;
    println!("g = {g}: sparse inertia of H_N − βI, both sectors, bracket vs exact");
    println!(
        "{:>3} {:>6} {:>7} {:>9} {:>12} {:>12} {:>12} {:>10} {:>8}",
        "N", "n", "fill", "count(β)", "lower", "upper", "exact", "secs", "verdict"
    );
    for n_max in [6u32, 8, 10, 12, 14, 16] {
        let h = qcd_ym_hamiltonian(g);
        let trunc = Truncated::build(&h, n_max);
        let clock = Instant::now();
        let odd_sector = trunc.sector(false);
        let even_sector = trunc.sector(true);
        let odd_columns = sector_columns(&trunc, &odd_sector);
        let even_columns = sector_columns(&trunc, &even_sector);
        let n = odd_columns.len();

        if n_max <= 10 {
            // The sparse sector matrix must be the dense one, entry for entry,
            // and symmetric: everything downstream assumes both.
            let dense = trunc.sector_matrix(false);
            for (j, column) in odd_columns.iter().enumerate() {
                for &(i, v) in column {
                    assert!(
                        (v - dense[(i, j)]).abs() < 1e-12,
                        "sector column ({i},{j}) = {v} vs dense {}",
                        dense[(i, j)]
                    );
                    assert!(
                        odd_columns[i]
                            .iter()
                            .any(|&(jj, vv)| jj == j && (vv - v).abs() < 1e-12),
                        "sector matrix is not symmetric at ({i},{j})"
                    );
                }
            }
        }

        let odd_mv = trunc.sector_matvec(&odd_sector);
        let even_mv = trunc.sector_matvec(&even_sector);
        let odd_start = trunc.coordinates(&odd_sector, &r_odd_start()).0;
        let even_start = trunc.coordinates(&even_sector, &empty_vacuum()).0;
        let steps = 120.min(n);
        let odd_ritz = lanczos_ritz(&odd_mv, n, &odd_start, steps).expect("odd Lanczos");
        let even_ritz =
            lanczos_ritz(&even_mv, even_columns.len(), &even_start, steps).expect("even Lanczos");
        let (theta_odd, residual_odd) = odd_ritz.ground().expect("non-empty");
        let (theta_even, residual_even) = even_ritz.ground().expect("non-empty");

        // `β` sits above `θ`, and the count then says "exactly one eigenvalue
        // below β" iff `β ≤ λ₂`, which is Temple's witness. The margin is a
        // ladder rather than a hair: at a margin of `1e-9` the shifted matrix has
        // an eigenvalue `1e-9` from `β`, so one elimination pivot is tiny and its
        // sign cannot be decided (measured at `N ≤ 14`). Any margin that decides
        // with a count of one is a valid witness; a margin that overshoots `λ₂`
        // decides with a count of two and is rejected by the caller below.
        let probe = |columns: &[Vec<(usize, f64)>], theta: f64| {
            let mut found = None;
            for margin in [1e-9, 1e-7, 1e-5, 1e-3, 1e-2] {
                let beta = theta + margin * theta.abs().max(1.0);
                if let Some(count) = inertia_below(columns, beta) {
                    found = Some((beta, count));
                    break;
                }
            }
            found
        };
        let odd_probe = probe(&odd_columns, theta_odd);
        let even_probe = probe(&even_columns, theta_even);
        let (beta_odd, odd_count) = match odd_probe {
            Some((beta, count)) => (beta, Some(count)),
            None => (f64::NAN, None),
        };
        let (beta_even, even_count) = match even_probe {
            Some((beta, count)) => (beta, Some(count)),
            None => (f64::NAN, None),
        };

        let mut verdict = "pivot undecided";
        let mut lower = f64::NAN;
        let mut upper = f64::NAN;
        let mut fill = 0usize;
        let mut count = "—".to_string();
        if let (Some(odd), Some(even)) = (odd_count, even_count) {
            fill = odd.fill.max(even.fill);
            count = format!("{}/{}", odd.below, even.below);
            let temple_odd =
                fock_sirk::mass_gap_spec::temple_lower_bound(theta_odd, residual_odd, beta_odd);
            let temple_even =
                fock_sirk::mass_gap_spec::temple_lower_bound(theta_even, residual_even, beta_even);
            // `temple(odd) − θᵉ₀ ≤ λ₁(odd) − λ₀(even) ≤ θᵒ₀ − temple(even)`: both
            // Temple bounds are sector-local, so this needs no orientation
            // assumption and holds whether or not the even state is lower.
            if let (Some(to), Some(te)) = (temple_odd, temple_even) {
                lower = to - theta_even;
                upper = theta_odd - te;
                verdict = if lower > 0.0 {
                    "CERTIFIED"
                } else {
                    "not positive"
                };
            } else {
                verdict = "Temple refused";
            }
        }
        let secs = clock.elapsed().as_secs_f64();

        let exact = if n_max <= 10 {
            let odd_spectrum = ascending_spectrum(&trunc.sector_matrix(false));
            let even_spectrum = ascending_spectrum(&trunc.sector_matrix(true));
            Some(odd_spectrum[0] - even_spectrum[0])
        } else {
            None
        };
        println!(
            "{n_max:>3} {n:>6} {fill:>7} {count:>9} {lower:>12.9} {upper:>12.9} {:>12} {secs:>10.2} \
             {verdict}",
            exact
                .map(|v| format!("{v:.9}"))
                .unwrap_or_else(|| "—".to_string())
        );

        if let Some(exact) = exact {
            assert!(
                lower <= exact + 1e-9 && exact <= upper + 1e-9,
                "N <= {n_max}: the bracket [{lower}, {upper}] does not contain the exact gap \
                 {exact} — a bound or the count is wrong"
            );
            // The count itself, at three shifts, against the exact spectrum of
            // the same sector: the one place the inertia route can be wrong and
            // still look right.
            let odd_spectrum = ascending_spectrum(&trunc.sector_matrix(false));
            for beta in [theta_odd - 0.5, beta_odd, theta_odd + 0.5] {
                let Some(measured) = inertia_below(&odd_columns, beta) else {
                    continue;
                };
                let exact_below = odd_spectrum.iter().filter(|&&v| v < beta).count();
                assert_eq!(
                    measured.below, exact_below,
                    "N <= {n_max}, β = {beta}: the inertia count says {} below β, the exact \
                     spectrum says {exact_below}",
                    measured.below
                );
            }
        }
    }
}
