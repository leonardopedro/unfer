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
//! — on each cell of the grid, with `β ≤ λ₂` supplied two ways so the two can be
//! compared on the same rows:
//!
//! * **dense** — [`certify_spectrum`] on the odd block: `n` Parlett intervals
//!   merged into disjoint clusters, so each cluster's count is exact and the
//!   second cluster's lower edge is a rigorous `λ₂` bound. Costs `O(n³)`.
//! * **sparse** — [`DiagonalExclusion`]: Weyl's `λ_i(H) ≥ λ_i(diag H) − ‖V‖₂`
//!   with `‖V‖_F` measured from matvecs. `O(n)` matvecs at any `n`, but the
//!   bound decays like `√n` against a fixed gap, so it is expected to be far
//!   weaker (`the_sparse_witness_is_sound_but_does_not_certify` pins that).
//!
//! and with the Ritz pair `(θ, r)` supplied two ways as well ([`Solver`]): the
//! forward truncated SIRK solve of §6d, and matvec-only Lanczos on the same
//! `H_N`. The certificate is identical; only the solve changes, and it is the
//! solve — not the witness — that decides most of these verdicts. `θᵉ₀` is
//! consumed only as an upper bound on `λ₀`, so the even side needs no witness at
//! all, just a value close enough to `λ₀` to leave room for the odd side.
//!
//! A cell where the odd ground sits *below* the even one inside `H_N` has no gap
//! in T6's orientation (the expression is the negative of its gap), and such a
//! cell is reported as crossing rather than as a failure of the certificate.
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
    DiagonalExclusion, SirkOpts, certified_ground_state, certify_spectrum, lanczos_ritz,
    solve_forward_sirk_truncated,
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
/// 40 matvecs. That difference, not the certificate, is what moves the verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Solver {
    /// `solve_forward_sirk_truncated` at `m = 12` — the §6d baseline.
    SirkM12,
    /// Matvec-only Lanczos on the same `H_N`, [`CONVERGED_STEPS`] steps.
    Lanczos40,
}

impl Solver {
    const ALL: [Solver; 2] = [Solver::SirkM12, Solver::Lanczos40];

    fn label(self) -> &'static str {
        match self {
            Solver::SirkM12 => "SIRK m=12",
            Solver::Lanczos40 => "Lanczos 40",
        }
    }
}

/// Lanczos steps for the converged route — comfortably above the sector
/// dimensions on this grid (`≤ 200`), so the ground converges rather than
/// merely acquiring a small residual: measured `r ≤ 3e-3` at the worst cell and
/// `≤ 3e-11` at most.
const CONVERGED_STEPS: usize = 40;

/// Everything a row needs that is a property of the *cell* rather than of the
/// solve, so both routes are compared on identical exact references.
struct Cell {
    g: f64,
    n_max: u32,
    dim: usize,
    lambda_odd_1: f64,
    lambda_odd_2: f64,
    lambda_even_1: f64,
    beta_dense: Option<f64>,
    beta_sparse: Option<f64>,
    exact_gap: Option<f64>,
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
    lambda_even_1: f64,
    beta_dense: Option<f64>,
    beta_sparse: Option<f64>,
    certified: Option<f64>,
    exact_gap: Option<f64>,
    leak: f64,
}

impl Cell {
    fn row(
        &self,
        solver: Solver,
        theta_odd: f64,
        residual_odd: f64,
        theta_even: f64,
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
            lambda_even_1: self.lambda_even_1,
            beta_dense: self.beta_dense,
            beta_sparse: self.beta_sparse,
            certified: self.beta_dense.and_then(|beta| {
                certified_gap_lower_bound_with_exclusion(theta_odd, residual_odd, beta, theta_even)
            }),
            exact_gap: self.exact_gap,
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
        for n_max in [6u32, 8] {
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

            // Dense witness: a `λ₂` bound from the exact eigenvalue count.
            let beta_dense =
                certify_spectrum(&odd_block).and_then(|cert| cert.second_eigenvalue_lower_bound());
            // The same quantity from matvecs only, kept for the comparison the
            // sibling test pins.
            let odd_dense_matvec = dense_matvec(&odd_block);
            let beta_sparse = DiagonalExclusion::from_matvec(&odd_dense_matvec, odd_block.nrows())
                .and_then(|exclusion| exclusion.second_eigenvalue_lower_bound());

            let cell = Cell {
                g,
                n_max,
                dim: odd_block.nrows(),
                lambda_odd_1: odd_spectrum[0],
                lambda_odd_2: odd_spectrum[1],
                lambda_even_1: even_spectrum[0],
                beta_dense,
                beta_sparse,
                exact_gap,
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
            let (theta_even, _) = even_ritz.ground().expect("non-empty");
            rows.push(cell.row(Solver::Lanczos40, theta_odd, residual_odd, theta_even));
        }
    }

    let header = |solver: Solver| {
        println!("\n{} — {}", solver.label(), "H_N rows, m = 12");
        println!(            "{:>4} {:>3} {:>4} {:>5} {:>12} {:>9} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} \
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
            "β dense",
            "β sparse",
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
                row.beta_dense.unwrap_or(f64::NAN),
                row.beta_sparse.unwrap_or(f64::NAN),
                row.certified
                    .map(|v| format!("{v:.9}"))
                    .unwrap_or_else(|| "refused".to_string()),
                row.exact_gap
                    .map(|v| format!("{v:.9}"))
                    .unwrap_or_else(|| "—".to_string()),
            );
        }
    }

    // Soundness, checked row by row **after** the tables are printed, so a
    // failure arrives with its own numbers beside it.
    for row in &rows {
        let label = format!("{} g = {}, N <= {}", row.solver.label(), row.g, row.n_max);
        if let Some(beta) = row.beta_dense {
            assert!(
                beta <= row.lambda_odd_2 + 1e-9,
                "{label}: dense β = {beta} exceeds the exact λ₂ = {}",
                row.lambda_odd_2
            );
        }
        if let Some(beta) = row.beta_sparse {
            assert!(
                beta <= row.lambda_odd_2 + 1e-9,
                "{label}: sparse β = {beta} exceeds the exact λ₂ = {}",
                row.lambda_odd_2
            );
        }
        if let (Some(beta_s), Some(beta_d)) = (row.beta_sparse, row.beta_dense) {
            assert!(
                beta_s <= beta_d + 1e-9,
                "{label}: the sparse witness must not beat the dense one ({beta_s} > {beta_d})"
            );
        }
        if let (Some(lo), Some(exact)) = (row.certified, row.exact_gap) {
            assert!(
                lo <= exact + 1e-9,
                "{label}: certified bound {lo} exceeds the exact gap {exact}"
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
                    row.beta_dense.unwrap_or(f64::NAN),
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
    println!();
    println!(
        "cells with the odd ground BELOW the even one inside H_N (T6's expression is then the \
         negative of their gap): {crossing}"
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
        .filter(|r| r.solver == Solver::Lanczos40 && r.exact_gap.is_some())
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
    let (converged_certified, _, _) = tally(Solver::Lanczos40);
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
