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
//! — with `β ≤ λ₂` supplied two ways, so the two can be compared on the same
//! rows:
//!
//! * **dense** — [`certify_spectrum`] on the parity block: `n` Parlett intervals
//!   merged into disjoint clusters, so each cluster's count is exact and the
//!   second cluster's lower edge is a rigorous `λ₂` bound. Costs `O(n³)`.
//! * **sparse** — [`DiagonalExclusion`]: Weyl's `λ_i(H) ≥ λ_i(diag H) − ‖V‖₂`
//!   with `‖V‖_F` measured from matvecs. `O(n)` matvecs at any `n`, but the
//!   bound decays like `√n` against a fixed gap, so it is expected to be far
//!   weaker (`the_sparse_witness_is_sound_but_does_not_certify` pins that).
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

/// One measured campaign row.
///
/// The exact columns are present so a refused or violated bound can be
/// attributed: which of Temple's hypotheses failed (`θᵒ₀ ∈ [λ₁, λ₂)`, `β ≤ λ₂`,
/// `θᵉ₀ ≥ λ₀(even)`) is visible in the row itself rather than inferred later.
struct CertifiedRow {
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

    for g in [0.0, 0.5, 1.0, 2.0, 3.0] {
        let h = qcd_ym_hamiltonian(g);
        for n_max in [6u32, 8] {
            let trunc = Truncated::build(&h, n_max);
            let even_block = trunc.sector_matrix(true);
            let odd_block = trunc.sector_matrix(false);

            // The solves the certificate consumes: the odd one supplies the
            // Ritz pair, the even one the upper bound on λ₀.
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

            // Dense witness: a λ₂ bound for the odd sector, from the exact count.
            let beta_dense =
                certify_spectrum(&odd_block).and_then(|cert| cert.second_eigenvalue_lower_bound());
            // Sparse witness: the same quantity from matvecs only.
            let odd_matvec = dense_matvec(&odd_block);
            let beta_sparse = DiagonalExclusion::from_matvec(&odd_matvec, odd_block.nrows())
                .and_then(|exclusion| exclusion.second_eigenvalue_lower_bound());

            let certified = beta_dense.and_then(|beta| {
                certified_gap_lower_bound_with_exclusion(
                    odd_cert.value,
                    odd_cert.residual,
                    beta,
                    even_cert.value,
                )
            });

            // Exact parity-resolved gap of the same H_N: the ground must be the
            // even sector's for the cross-sector difference to be *the* gap.
            let even_spectrum = ascending_spectrum(&even_block);
            let odd_spectrum = ascending_spectrum(&odd_block);
            let exact_gap = if even_spectrum[0] < odd_spectrum[0] {
                Some(odd_spectrum[0] - even_spectrum[0])
            } else {
                None
            };

            rows.push(CertifiedRow {
                g,
                n_max,
                dim: odd_block.nrows(),
                theta_odd: odd_cert.value,
                residual_odd: odd_cert.residual,
                lambda_odd_1: odd_spectrum[0],
                lambda_odd_2: odd_spectrum[1],
                theta_even: even_cert.value,
                lambda_even_1: even_spectrum[0],
                beta_dense,
                beta_sparse,
                certified,
                exact_gap,
                leak: trunc.leak,
            });
        }
    }

    println!(
        "{:>4} {:>3} {:>4} {:>12} {:>10} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "g",
        "N",
        "dim",
        "θᵒ₀",
        "rᵒ",
        "λ₁ odd",
        "λ₂ odd",
        "β dense",
        "β sparse",
        "certified",
        "exact"
    );
    for row in &rows {
        println!(
            "{:>4} {:>3} {:>4} {:>12.9} {:>10.1e} {:>12.9} {:>12.9} {:>12.9} {:>12.3} \
             {:>12} {:>10}",
            row.g,
            row.n_max,
            row.dim,
            row.theta_odd,
            row.residual_odd,
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

    // Soundness, checked row by row **after** the table is printed, so a failure
    // arrives with its own numbers beside it.
    for row in &rows {
        let label = format!("g = {}, N <= {}", row.g, row.n_max);
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
    }

    println!("\nverdicts:");
    for row in &rows {
        let verdict = Verdict::of(row.certified);
        let why = match verdict {
            Verdict::Certified => format!(
                "certified {:.9} (exact {:.9})",
                row.certified.unwrap(),
                row.exact_gap.unwrap_or(f64::NAN)
            ),
            Verdict::NotPositive => format!(
                "bound {:.9} ≤ 0 — the even-side Rayleigh quotient θᵉ₀ = {:.9} sits {:.3e} \
                 above λ₀(even) = {:.9}",
                row.certified.unwrap(),
                row.theta_even,
                row.theta_even - row.lambda_even_1,
                row.lambda_even_1
            ),
            Verdict::Refused => format!(
                "refused — β = {:.9} vs θᵒ₀ = {:.9}, rᵒ = {:.3e} (residual ball reaches λ₂)",
                row.beta_dense.unwrap_or(f64::NAN),
                row.theta_odd,
                row.residual_odd
            ),
        };
        println!("  g = {}, N <= {}: {why}", row.g, row.n_max);
    }

    let certified: Vec<&CertifiedRow> = rows
        .iter()
        .filter(|r| r.certified.map(|v| v > 0.0).unwrap_or(false))
        .collect();
    let tally = |v: Verdict| {
        rows.iter()
            .filter(|r| Verdict::of(r.certified) == v)
            .count()
    };
    println!(
        "rows: {} certified, {} valid-but-not-positive, {} refused (of {})",
        certified.len(),
        tally(Verdict::NotPositive),
        tally(Verdict::Refused),
        rows.len()
    );
    for row in &certified {
        println!(
            "  g = {}, N <= {}: {:.9} (exact {:.9})",
            row.g,
            row.n_max,
            row.certified.unwrap(),
            row.exact_gap.unwrap()
        );
    }
    assert!(
        certified.iter().all(|r| r.exact_gap.is_some()),
        "a certified row must have an exact reference to have been checked against"
    );

    assert!(
        !certified.is_empty(),
        "the dense witness must certify somewhere on this grid — every row refusing would \
         mean the certificate stopped working, not that the gap vanished"
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
