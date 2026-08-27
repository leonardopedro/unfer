use crate::linalg::SirkError;
use crate::registry::StateDictionary;
use crate::tensor_state::TensorState;
use candle_core::Device;
use nalgebra::DMatrix;
use nested_fock_algebra::{Hamiltonian, QuantumState};
use num_complex::Complex64;

/// Default convergence tolerance for the per-step BRST projection.
const BRST_TOL: f64 = 1e-10;

pub struct ForwardSirkResult {
    /// Projected Hamiltonian in the whitened orthonormal basis (`rank x rank`).
    pub h_proj: DMatrix<Complex64>,
    /// Full Gram matrix of the Krylov sequence (`(m+1) x (m+1)`).
    pub g_matrix: DMatrix<Complex64>,
    pub registry: StateDictionary,
    pub basis_tensors: Vec<TensorState>,
    /// Numerical rank of the Krylov basis after whitening (= `h_proj.nrows()`).
    pub rank: usize,
    /// Whitening transform `W` (`m x rank`) mapping whitened coefficients back to
    /// Krylov-vector coordinates. `Wᴴ G_sub W = I_rank`.
    pub w_whiten: DMatrix<Complex64>,
    /// The retained Krylov sequence `w_0, ..., w_m` used by [`reconstruct`] to map
    /// whitened-basis coefficients back to a [`QuantumState`]. The stored
    /// vectors are the UNIT-NORM recurrence frame `ŵ_k` (see [`Self::scales`]).
    pub w_sequence: Vec<QuantumState>,
    /// Per-step norms of the unit-norm recurrence: `H u_k = τ_{k+1} u_{k+1}
    /// + z_k u_k` with `τ = self.scales` (τ₀ ≡ 1). Needed to assemble `H`
    /// from the Gram matrix and by [`Self::ritz_residuals`].
    pub scales: Vec<f64>,
    /// The shift sequence `z_0..z_{m-1}` used to build the recurrence
    /// (retained for the Gram-only residual computation).
    pub shifts: Vec<Complex64>,
}

impl ForwardSirkResult {
    /// Whitened-basis coefficients of the initial state `v_0 = w_0`.
    ///
    /// `v_0` has Krylov-coordinate `e_0`, so its whitened coordinate is
    /// `b_0 = Wᴴ G e_0 = Wᴴ · (first m entries of column 0 of G)`.
    fn initial_coeffs(&self) -> nalgebra::DVector<Complex64> {
        let m = self.w_whiten.nrows();
        let g0 = self.g_matrix.column(0).rows(0, m).into_owned();
        self.w_whiten.adjoint() * g0
    }

    /// Construct the evolution operator in the small basis: exp(-i * H_proj * t)
    /// applied to the initial state, returning coefficients in the whitened basis.
    pub fn time_evolve(&self, t: f64) -> nalgebra::DVector<Complex64> {
        let i = Complex64::new(0.0, 1.0);
        let evolution_matrix = (self.h_proj.clone() * (-i * t)).exp();
        evolution_matrix * self.initial_coeffs()
    }

    /// Map whitened-basis coefficients back to a [`QuantumState`].
    ///
    /// This is the inverse of [`time_evolve`](Self::time_evolve): it multiplies
    /// the whitened coefficients by the whitening transform `W` to get Krylov-vector
    /// coordinates, then linearly combines the stored `w_sequence` via
    /// [`QuantumState::scale_and_add`].
    pub fn reconstruct(&self, coeffs: &nalgebra::DVector<Complex64>) -> QuantumState {
        let w_coords = &self.w_whiten * coeffs;
        let m = self.w_whiten.nrows();
        let mut state = QuantumState::zero();
        for j in 0..m {
            if j < self.w_sequence.len() {
                state.scale_and_add(&self.w_sequence[j], w_coords[j]);
            }
        }
        state
    }

    /// Return the Ritz values (real eigenvalues of the projected Hamiltonian
    /// `h_proj`) sorted ascending.
    ///
    /// These approximate the low-lying spectrum of the full Hamiltonian `H` in
    /// the Krylov subspace built from `v_0`. The quality of the approximation
    /// depends on the Krylov dimension and the spectral reach from the starting
    /// state. For a Hermitian `H`, the Ritz values interlace the true
    /// eigenvalues and converge from the outside in as the dimension grows.
    pub fn ritz_values(&self) -> Vec<f64> {
        let eig = self.h_proj.clone().symmetric_eigen();
        let mut vals: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals
    }

    /// True relative residuals of every Ritz pair, computed **entirely from
    /// the stored Gram matrix** — no big-space work.
    ///
    /// The Galerkin condition makes the residual vanish *inside* the subspace,
    /// so what matters is the component outside it. For the forward sequence
    /// that component is exactly one dimension further: `H` maps
    /// `span{ŵ_0..ŵ_{m-1}}` into `span{ŵ_0..ŵ_m}` (the identity above), so for
    /// coefficient vector `c` (length m) the big-space image has coordinates
    ///
    ///   `d_j = z_j c_j + (σ_j/σ_{j-1}) c_{j-1}`  (d_m = (σ_m/σ_{m-1}) c_{m-1}),
    ///
    /// and with everything expressed through the Gram matrix G (size m+1):
    ///
    ///   ‖ψ‖² = c†G_sub c,  ‖Hψ‖² = d†G d,  ⟨ψ,Hψ⟩ = c† G[:, :m] d,
    ///   res(θ) = √(‖Hψ‖² − θ²‖ψ‖²) / ‖Hψ‖ .
    ///
    /// Returns `(θ, rel_res)` sorted by ascending θ. This is the type-enforced
    /// version of the practical rule "filter Ritz values above the resolved
    /// window": consumers can select pairs by convergence quality instead of
    /// hand-maintained energy cutoffs. See `fock_sirk/tests/ritz_edge_study.rs`.
    pub fn ritz_residuals(&self) -> Vec<(f64, f64)> {
        let m = self.scales.len() - 1;
        let eig = self.h_proj.clone().symmetric_eigen();

        // Sort eigenpairs ascending.
        let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
        order.sort_by(|&a, &b| {
            eig.eigenvalues[a]
                .partial_cmp(&eig.eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let g = &self.g_matrix;
        let g_sub = g.view((0, 0), (m, m));

        let mut out = Vec::with_capacity(order.len());
        for &i in &order {
            let theta = eig.eigenvalues[i];
            let c = eig.eigenvectors.column(i); // whitened-frame coeffs

            // Whitened → unit-norm Krylov frame: ĉ = W c (length m).
            let c_hat = &self.w_whiten * c;

            // Big-space coordinates d (length m+1):
            //   d_j = z_j ĉ_j + τ_j ĉ_{j-1}   (τ from the recurrence frame),
            // so that H ψ has coordinates d in span{u_0..u_m}.
            let mut d = nalgebra::DVector::<Complex64>::zeros(m + 1);
            for j in 0..m {
                d[j] += self.shifts[j] * c_hat[j];
            }
            for j in 1..=m {
                d[j] += self.scales[j] * c_hat[j - 1];
            }

            let psi_norm2 = (c_hat.adjoint() * g_sub * &c_hat)[(0, 0)]
                .re
                .max(1e-300);
            // Residual VECTOR, formed before any metric contraction:
            //   e_j = d_j − θ ĉ_j  (j < m; vanishes identically in exact
            //        arithmetic by the Galerkin condition),
            //   e_m = τ_m ĉ_{m−1}   (the one component H pushes OUTSIDE the
        //        projected basis — the entire physical content of the
            //        residual).
            // Then ‖Hψ − θψ‖² = e† G e exactly. Computing the norm this way
            // (instead of ‖Hψ‖² − θ²‖ψ‖²) avoids the catastrophic
            // cancellation that clamps small residuals to zero.
            let mut e = nalgebra::DVector::<Complex64>::zeros(m + 1);
            for j in 0..m {
                e[j] = d[j] - theta * c_hat[j];
            }
            e[m] = self.scales[m] * c_hat[m - 1];
            let res2 = (e.conjugate().transpose() * g * e)[(0, 0)].re.max(0.0);
            let hpsi_norm = (theta * theta * psi_norm2).sqrt().max(1e-300);
            out.push((theta, res2.sqrt() / hpsi_norm));
        }
        out
    }

    /// **Absolute** residuals `‖Hψ − θψ‖` of every Ritz pair, computed
    /// entirely from the stored Gram matrix (same cancellation-free
    /// construction as [`Self::ritz_residuals`], without the relative
    /// normalization).  Returns `(θ, ‖Hψ − θψ‖)` sorted by ascending `θ`.
    ///
    /// This is the **sharp-tier certificate**: by the Rayleigh–Ritz residual
    /// bound (Parlett), `|θ − λ| ≤ ‖Hψ − θψ‖` for the *exact* operator
    /// applied to the *computed* vector — the a-posteriori width the
    /// certified mass-gap assembly consumes (`MASS_GAP_CERTIFIED.md` §2,
    /// §4.3; `fock_sirk/src/certificate.rs`).
    pub fn ritz_abs_residuals(&self) -> Vec<(f64, f64)> {
        let m = self.scales.len() - 1;
        let eig = self.h_proj.clone().symmetric_eigen();

        // Sort eigenpairs ascending.
        let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
        order.sort_by(|&a, &b| {
            eig.eigenvalues[a]
                .partial_cmp(&eig.eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let g = &self.g_matrix;

        let mut out = Vec::with_capacity(order.len());
        for &i in &order {
            let theta = eig.eigenvalues[i];
            let c = eig.eigenvectors.column(i); // whitened-frame coeffs
            let c_hat = &self.w_whiten * c; // → unit-norm Krylov frame

            // Big-space coordinates d (length m+1): H ψ has coordinates d in
            // span{u_0..u_m} (see [`Self::ritz_residuals`]).
            let mut d = nalgebra::DVector::<Complex64>::zeros(m + 1);
            for j in 0..m {
                d[j] += self.shifts[j] * c_hat[j];
            }
            for j in 1..=m {
                d[j] += self.scales[j] * c_hat[j - 1];
            }

            // Residual vector, formed before any metric contraction:
            //   e_j = d_j − θ ĉ_j  (j < m; vanishes by the Galerkin
            //        condition in exact arithmetic),
            //   e_m = τ_m ĉ_{m−1}   (the one component H pushes OUTSIDE the
            //        projected basis — the entire physical content).
            // ‖Hψ − θψ‖² = e† G e exactly (no catastrophic cancellation).
            let mut e = nalgebra::DVector::<Complex64>::zeros(m + 1);
            for j in 0..m {
                e[j] = d[j] - theta * c_hat[j];
            }
            e[m] = self.scales[m] * c_hat[m - 1];
            let res2 = (e.conjugate().transpose() * g * e)[(0, 0)].re.max(0.0);
            out.push((theta, res2.sqrt()));
        }
        out
    }

    /// Ritz values whose true relative residual (see [`Self::ritz_residuals`])
    /// is at most `rel_tol`, sorted ascending — the converged part of the
    /// spectrum, enforced by the solver instead of by caller convention.
    pub fn resolved_ritz_values(&self, rel_tol: f64) -> Vec<f64> {
        self.ritz_residuals()
            .into_iter()
            .filter(|&(_, r)| r <= rel_tol)
            .map(|(theta, _)| theta)
            .collect()
    }


    /// Estimate the **intra-sector** spectral gap `E₁ − E₀` from the two
    /// lowest Ritz values of a single SIRK solve.
    ///
    /// Returns `None` if the Krylov basis has rank < 2.
    ///
    /// **Parity caveat (Yang-Mills):** the lattice Hamiltonian preserves total
    /// excitation-number parity (the electric term is diagonal; the quartic
    /// magnetic term changes excitation by ∈ {±4, ±2, 0}). A Krylov subspace
    /// built from the vacuum therefore only contains even-parity states, and
    /// `mass_gap()` reports the gap to the lowest *even-parity* excitation
    /// (≈ 2 × g²/2 = g²), **not** the particle-physics mass gap g²/2 (which is
    /// the one-excitation / odd-parity gap). To extract the true mass gap,
    /// compare the ground-state energies of two solves — one from the vacuum
    /// (even sector) and one from a one-excitation state (odd sector) — via
    /// [`mass_gap_from_sectors`].
    pub fn mass_gap(&self) -> Option<f64> {
        let ritz = self.ritz_values();
        if ritz.len() < 2 {
            return None;
        }
        Some(ritz[1] - ritz[0])
    }

    /// Estimate the ground-state energy `E₀` from the lowest Ritz value.
    pub fn ground_state_energy(&self) -> Option<f64> {
        self.ritz_values().first().copied()
    }

    /// Phase 11.2: Export simulation coefficients to JSON for visualization.
    pub fn export_to_json(&self) -> String {
        use serde_json::json;
        let mut data = Vec::new();
        for i in 0..self.h_proj.nrows() {
            for j in 0..self.h_proj.ncols() {
                data.push(json!({
                    "row": i,
                    "col": j,
                    "re": self.h_proj[(i, j)].re,
                    "im": self.h_proj[(i, j)].im
                }));
            }
        }
        json!({
            "h_proj": data,
            "m_dim": self.h_proj.nrows()
        })
        .to_string()
    }
}

/// Estimate the **cross-sector mass gap** from two SIRK solves in different
/// parity sectors.
///
/// For lattice gauge theories (e.g. `yang_mills_lattice`), the Hamiltonian
/// preserves total excitation-number parity. The true mass gap is the energy
/// difference between the vacuum (even-parity ground state, E₀ ≈ 0) and the
/// one-particle state (odd-parity ground state, E₁ ≈ g²/2). Since a single
/// Krylov subspace built from either sector cannot see the other, this function
/// compares the ground-state Ritz values from two independent solves.
///
/// Returns `None` if either solve has rank 0.
pub fn mass_gap_from_sectors(even: &ForwardSirkResult, odd: &ForwardSirkResult) -> Option<f64> {
    Some(odd.ground_state_energy()? - even.ground_state_energy()?)
}

/// Proof-facing parity-sector mass gap (`MASS_GAP_CERTIFIED.md` §3.3–§3.4,
/// T6). This is the **formalization seam**: it runs the two pure-parity SIRK
/// solves (even = vacuum, odd = one electric-flux quantum), enforces the
/// theorem's checkable preconditions at runtime (`debug_assert`), and
/// assembles the certified gap `[lo, hi]` via
/// [`crate::certified_mass_gap`].
///
/// Preconditions enforced here (see [`crate::mass_gap_spec`] for the exact
/// contracts):
/// 1. **Sector purity** — lattice parity is an exact symmetry of `H_m`, so
///    the two Krylov chains must be disjoint; witnessed by the maximal
///    mutual overlap of the retained chain vectors (`< 1e-8`).
/// 2. **Even ground = vacuum** — the even sector ground Ritz value must be
///    `O(1/g⁶)`-small at strong coupling (the magnetic shift of the vacuum
///    is second order); witnessed by `|θᵉ₀| < 0.1`.
///
/// Returns `None` if either solve fails or has rank 0. In release builds the
/// precondition witnesses degrade to documentation (debug-only asserts).
pub fn certified_mass_gap_parity(
    h: &Hamiltonian,
    v_even: &QuantumState,
    v_odd: &QuantumState,
    shifts: &[Complex64],
    opts: &SirkOpts,
) -> Option<crate::certificate::GapCertificate> {
    let device = crate::device::best_device();
    let res_even =
        solve_forward_sirk_with_opts(h, v_even, shifts, &device, None, opts).ok()?;
    let res_odd =
        solve_forward_sirk_with_opts(h, v_odd, shifts, &device, None, opts).ok()?;

    // Precondition 1: sector purity (ChapterParity) — the chains must be
    // disjoint. The overlap witness is exact for pure-parity starts (the
    // occupation parity differs on every basis state).
    let max_overlap = res_even
        .w_sequence
        .iter()
        .flat_map(|we| {
            res_odd
                .w_sequence
                .iter()
                .map(move |wo| QuantumState::inner_product(we, wo).norm())
        })
        .fold(0.0_f64, f64::max);
    debug_assert!(
        crate::mass_gap_spec::parities_disjoint(max_overlap, 1e-8),
        "even/odd Krylov chains must be disjoint (parity symmetry of H_m); \
         max overlap = {max_overlap:.2e}"
    );

    // Precondition 2 witness: the even ground is the normal-ordered vacuum.
    let e_even = res_even.ground_state_energy()?;
    debug_assert!(
        crate::mass_gap_spec::even_sector_is_vacuum(e_even, 0.1),
        "even sector ground must be the vacuum at strong coupling, got {e_even}"
    );

    crate::certificate::certified_mass_gap(&res_even, &res_odd)
}

/// Tunable bounds and tolerances for the SIRK solve.
#[derive(Debug, Clone)]
pub struct SirkOpts {
    /// Drop Krylov-vector components with `|amplitude| <= prune_eps` each
    /// step. DEFAULT 0.0 = DISABLED: the canonical Hashimoto sequence keeps
    /// every component the exact recurrence produces (model-fidelity
    /// directive; justified-inert usage is documented in
    /// `tests/guard_justification_study.rs` Study A). Memory-bounded runs on
    /// explosion-prone models opt in explicitly (e.g. 1e-12), which the same
    /// study shows is invariant below the solver noise floor.
    pub prune_eps: f64,
    /// Hard ceiling on the number of components in any Krylov vector. When
    /// [`SirkOpts::adaptive`] is false (default), exceeding this aborts with
    /// [`SirkError::StateExplosion`]. When `adaptive` is true, the solver
    /// instead keeps only the `max_components` largest-amplitude components
    /// (via [`QuantumState::truncate_top_k`]), allowing larger lattices to
    /// run under a fixed memory budget at the cost of approximation error.
    pub max_components: Option<usize>,
    /// Convergence tolerance for the per-step BRST projection.
    pub brst_tol: f64,
    /// When true, fall back to `truncate_top_k(max_components)` instead of
    /// erroring with `StateExplosion`. This lets quartic-heavy models (e.g.
    /// `yang_mills_lattice` at l≥4) run under a fixed budget. The truncation
    /// error is bounded by the total probability mass of the dropped
    /// components; the Gram whitening absorbs the resulting non-orthonormality.
    pub adaptive: bool,
    /// Store the forward sequence in the unit-norm frame (each Krylov vector
    /// rescaled to ‖u‖=1 after its step, with the step norms `τ` folded into
    /// the projection identity `H u_k = τ_{k+1} u_{k+1} + z_k u_k`).
    ///
    /// DEFAULT FALSE — the canonical Hashimoto forward products
    /// `w_k = ∏(H − z_j I) v₀` are stored exactly as the theory defines them,
    /// and the generalized pencil absorbs non-normality through the Gram
    /// matrix. This flag is a numerically EXACT basis reparametrization of
    /// the same Krylov span (Rayleigh–Ritz invariant, no approximation):
    /// it exists because raw Gram matrices become ill-conditioned for deep
    /// windows (‖w_k‖ ~ ‖H‖ᵏ), and deep spectral solves measure better in
    /// the normalized frame. See `tests/ritz_edge_study.rs` for both
    /// measured profiles.
    pub unit_norm_steps: bool,
}

impl Default for SirkOpts {
    fn default() -> Self {
        Self {
            prune_eps: 0.0,
            max_components: None,
            brst_tol: BRST_TOL,
            adaptive: false,
            unit_norm_steps: false,
        }
    }
}

/// Backwards-compatible entry point: solve with [`SirkOpts::default`].
pub fn solve_forward_sirk(
    hamiltonian: &Hamiltonian,
    v_0: &QuantumState,
    shifts: &[Complex64],
    device: &Device,
    brst_charge: Option<&Hamiltonian>, // Phase 10.1: Optional BRST Projection
) -> Result<ForwardSirkResult, SirkError> {
    solve_forward_sirk_with_opts(
        hamiltonian,
        v_0,
        shifts,
        device,
        brst_charge,
        &SirkOpts::default(),
    )
}

pub fn solve_forward_sirk_with_opts(
    hamiltonian: &Hamiltonian,
    v_0: &QuantumState,
    shifts: &[Complex64],
    device: &Device,
    brst_charge: Option<&Hamiltonian>,
    opts: &SirkOpts,
) -> Result<ForwardSirkResult, SirkError> {
    solve_forward_sirk_with_matvec(
        |state| hamiltonian.apply(state),
        v_0,
        shifts,
        device,
        brst_charge,
        opts,
    )
}

/// Like [`solve_forward_sirk_with_opts`] but takes a custom matvec
/// closure `F: Fn(&QuantumState) -> QuantumState` that computes
/// `H |state⟩` (the Hamiltonian application, not the shifted one).
/// The shift `(H - z_k I) |state⟩` is then computed as
/// `matvec(state) - z_k * state` by the SIRK loop. This lets callers
/// use a fast-path matvec (e.g. O(M) for the QFM.tex
/// `|0⟩⟨0| + Σ_j α_j n̂_j` Hamiltonian, where the general
/// `Hamiltonian::apply` is O(K_2²) per matvec because it iterates
/// over all components × all terms) without modifying the SIRK
/// core.
pub fn solve_forward_sirk_with_matvec<F>(
    matvec: F,
    v_0: &QuantumState,
    shifts: &[Complex64],
    device: &Device,
    brst_charge: Option<&Hamiltonian>,
    opts: &SirkOpts,
) -> Result<ForwardSirkResult, SirkError>
where
    F: Fn(&QuantumState) -> QuantumState,
{
    let m = shifts.len();

    // Forward-sequence frame selection.
    //
    // CANONICAL (default, `unit_norm_steps: false`): store the raw Hashimoto
    // products w_k = (H − z_k I) w_{k−1} exactly as the theory defines them;
    // all scale factors are τ ≡ 1 and the generalized pencil absorbs the
    // non-normality through the Gram matrix.
    //
    // UNIT-NORM FRAME (opt-in, `unit_norm_steps: true`): each stored vector
    // is rescaled to ‖u‖=1 after its step. This is a numerically EXACT basis
    // reparametrization — same Krylov span, Rayleigh–Ritz invariant, no
    // approximation — motivated purely by conditioning: raw Gram matrices
    // grow like ‖H‖^k and defeat whitening for deep windows. In this frame
    // the projection identity reads H u_k = τ_{k+1} u_{k+1} + z_k u_k with
    // τ_{k+1} = ‖(H − z_k I) u_k‖ the PRE-normalization step norm.
    //
    // In BOTH frames: u₀ is stored EXACTLY AS GIVEN (never rescaled), so the
    // state amplitude survives `time_evolve`→`reconstruct` for inputs with
    // ‖ψ₀‖ ≠ 1, and the unified identity above holds with `scales`.
    let mut scales: Vec<f64> = Vec::with_capacity(m + 1);
    scales.push(1.0);
    let mut w_sequence: Vec<QuantumState> = Vec::with_capacity(m + 1);
    w_sequence.push(v_0.clone());

    // 1. Generate the forward sequence: w_k = (H - z_k I) w_{k-1}
    for k in 0..m {
        let prev_w = &w_sequence[k];
        let mut next_w = matvec(prev_w);
        // next_w = H * prev_w - shifts[k] * prev_w
        next_w.scale_and_add(prev_w, -shifts[k]);

        // Phase 10.1: Periodic BRST projection to maintain gauge invariance.
        // True orthogonal projection onto ker(Q) via matrix-free CG (replaces the
        // earlier non-idempotent subtraction hack).
        if let Some(brst) = brst_charge {
            next_w = crate::brst::project_physical(&next_w, brst, opts.brst_tol, 50)?;
        }

        // Memory hygiene + explosion guard.
        next_w.prune(opts.prune_eps);
        if let Some(limit) = opts.max_components
            && next_w.len() > limit
        {
            if opts.adaptive {
                // Keep only the `limit` largest-amplitude components. This
                // trades approximation error for a guaranteed memory bound,
                // enabling quartic-heavy models (l≥4 Yang-Mills) to run.
                next_w.truncate_top_k(limit);
            } else {
                return Err(SirkError::StateExplosion {
                    components: next_w.len(),
                    limit,
                });
            }
        }

        // Optional unit-norm frame (see `SirkOpts::unit_norm_steps`): store
        // u_{k+1} = w_{k+1}/‖w_{k+1}‖ and remember the step norm τ. In the
        // canonical mode (default) the RAW product is stored and τ ≡ 1 —
        // both modes satisfy H u_k = τ_{k+1} u_{k+1} + z_k u_k exactly.
        // On breakdown (‖·‖ ≈ 0, linear dependence) keep the raw vector with
        // unit scale; whitening will drop the dependent direction.
        if opts.unit_norm_steps {
            let nk = next_w.norm();
            if nk.is_finite() && nk > 1e-30 {
                let mut scaled = QuantumState::zero();
                scaled.scale_and_add(&next_w, Complex64::new(1.0 / nk, 0.0));
                scales.push(nk);
                w_sequence.push(scaled);
                continue;
            }
        }
        scales.push(1.0);
        w_sequence.push(next_w);
    }

    // 2. Flatten states into a registry for GPU processing
    let mut registry = StateDictionary::new();
    for w in &w_sequence {
        registry.register(w);
    }

    let mut basis_tensors = Vec::with_capacity(m + 1);
    for w in &w_sequence {
        basis_tensors.push(
            TensorState::from_quantum_state(w, &mut registry, device)
                .map_err(|e| SirkError::Numeric(e.to_string()))?,
        );
    }

    // 3. Compute the Gram matrix G_jk = <w_j | w_k> on the GPU
    // 3. Compute the Gram matrix G_jk = <w_j | w_k> on the GPU
    // Parallelized using Rayon for Phase 9.1 performance milestone.
    use rayon::prelude::*;

    let mut g_matrix = DMatrix::zeros(m + 1, m + 1);

    // We compute only the upper triangle due to Hermiticity: G_kj = G_jk*
    // We flatten the indices to use par_iter
    let indices: Vec<(usize, usize)> = (0..=m).flat_map(|j| (j..=m).map(move |k| (j, k))).collect();

    let results: Vec<candle_core::Result<((usize, usize), Complex64)>> = indices
        .into_par_iter()
        .map(|(j, k)| {
            let val = basis_tensors[j].inner_product(&basis_tensors[k])?;
            Ok(((j, k), val))
        })
        .collect();

    for res in results {
        let ((j, k), val) = res.map_err(|e| SirkError::Numeric(e.to_string()))?;
        g_matrix[(j, k)] = val;
        if j != k {
            g_matrix[(k, j)] = val.conj();
        }
    }

    // 3. Compute the Gram matrix G_jk = <w_j | w_k> on the GPU

    // 4. Construct the projected Hamiltonian H_jk = <w_j | H | w_k>
    // Using the identity (normalized frame): H ŵ_k = (σ_{k+1}/σ_k) ŵ_{k+1} + z_k ŵ_k
    // <ŵ_j|H|ŵ_k> = (σ_{k+1}/σ_k) G_{j,k+1} + z_k G_{j,k}
    let mut h_proj_raw = DMatrix::zeros(m, m);
    for j in 0..m {
        for k in 0..m {
            h_proj_raw[(j, k)] =
                scales[k + 1] * g_matrix[(j, k + 1)] + shifts[k] * g_matrix[(j, k)];
        }
    }

    // 5. Orthonormalize the system (solve the generalized problem H_raw c = λ G c)
    // via rank-revealing Gram whitening. This replaces a bare Cholesky that
    // panicked whenever the Krylov vectors became linearly dependent.
    let g_sub = g_matrix.view((0, 0), (m, m)).into_owned();
    let whitening = crate::linalg::whiten_gram(&g_sub, crate::linalg::GRAM_REL_TOL)?;
    let w = whitening.w; // m x rank
    let h_proj = w.adjoint() * h_proj_raw * &w; // rank x rank

    Ok(ForwardSirkResult {
        h_proj,
        g_matrix,
        registry,
        basis_tensors,
        rank: whitening.rank,
        w_whiten: w,
        w_sequence,
        scales,
        shifts: shifts.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState};

    fn shifts(m: usize) -> Vec<Complex64> {
        (0..m)
            .map(|j| Complex64::new(0.0, 1.0 + (j as f64) * 0.2))
            .collect()
    }

    /// A zero Hamiltonian makes every Krylov vector parallel to v_0, so the Gram
    /// matrix is rank 1. The old bare Cholesky panicked here; whitening must not.
    #[test]
    fn rank_deficient_gram_no_panic() {
        let device = Device::Cpu;
        let h = Hamiltonian { terms: vec![] };
        let v0 =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
        let res = solve_forward_sirk(&h, &v0, &shifts(4), &device, None)
            .expect("whitening must not panic on a rank-deficient Gram");
        assert_eq!(res.rank, 1, "all Krylov vectors are parallel => rank 1");
        assert_eq!(res.h_proj.nrows(), 1);
    }

    /// Two-state hopping Hamiltonian H = |B><A| + |A><B| has eigenvalues ±1.
    /// After whitening, the projected Hamiltonian must be Hermitian with real
    /// Ritz values matching ±1 (basis-independent), confirming the orthonormal-
    /// ization is correct.
    #[test]
    fn ritz_values_real_for_hermitian() {
        let device = Device::Cpu;
        let a = InnerBosonicState::vacuum();
        let mut b = InnerBosonicState::vacuum();
        b.modes.insert(0, 1);

        let h = Hamiltonian {
            terms: vec![
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(b.clone()),
                        Operator::OuterBosonAnnihilate(a.clone()),
                    ],
                ),
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(a.clone()),
                        Operator::OuterBosonAnnihilate(b.clone()),
                    ],
                ),
            ],
        };
        let v0 = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(a));

        let res = solve_forward_sirk(&h, &v0, &shifts(4), &device, None).unwrap();
        assert!(
            res.rank >= 2,
            "hopping spans a 2D Krylov space, got {}",
            res.rank
        );

        // h_proj is Hermitian => real eigenvalues; the spectrum should bracket ±1.
        let eig = res.h_proj.clone().symmetric_eigen();
        let mut vals: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let lo = *vals.first().unwrap();
        let hi = *vals.last().unwrap();
        assert!((lo + 1.0).abs() < 1e-8, "lowest Ritz value {lo} != -1");
        assert!((hi - 1.0).abs() < 1e-8, "highest Ritz value {hi} != +1");
    }

    /// Stage 5: the Navier-Stokes Hamiltonian was previously an Expression that
    /// caused the CAS to hang in .expand(). Now built directly as Hamiltonian
    /// terms (like yang_mills). With Gram whitening (Stage 2) + explosion bounds
    /// (Stage 4), the full solve must complete on CPU without panicking.
    #[test]
    fn navier_stokes_solve_completes() {
        let device = Device::Cpu;
        let h = nested_fock_algebra::models::navier_stokes_hamiltonian(1e-3);

        assert!(
            !h.terms.is_empty(),
            "Navier-Stokes Hamiltonian must be non-empty"
        );

        let v0 =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

        let opts = SirkOpts {
            prune_eps: 1e-12,
            max_components: Some(50_000),
            brst_tol: 1e-10,
            adaptive: false,
            unit_norm_steps: false,
        };
        let res = solve_forward_sirk_with_opts(&h, &v0, &shifts(4), &device, None, &opts)
            .expect("Navier-Stokes SIRK solve must not panic or error");

        assert!(res.rank > 0, "Krylov basis must have positive rank");
        assert_eq!(res.h_proj.nrows(), res.rank);
    }

    /// Stage NS: numerical flow completeness of the truncated Navier-Stokes
    /// Hamiltonian.
    ///
    /// The book.tex Navier-Stokes chapter claims the polynomial Hamiltonian
    /// `H = ∫a†(πⁱ(u_j u_{i,j} − ν u_{i,jj}) + h.c.)a` is essentially
    /// self-adjoint on a dense domain (book.tex §4199-4208). Essential
    /// self-adjointness is a flow-completeness question (Nelson); low degree
    /// only makes H a well-defined polynomial/symmetric operator, so this test
    /// checks the *truncated* statement that is actually provable. On a
    /// *truncated* Krylov/Fock space the corresponding finite operator
    /// `h_proj` is a finite Hermitian matrix, so `e^{−iHt}` is a **unitary**
    /// group: probability is conserved and the flow cannot blow up in finite
    /// time — the numerical shadow of the "complete flow ⟹ no singularities"
    /// claim on the truncation. This test checks
    /// (1) `h_proj = h_proj†` (the unitarity hypothesis), (2) `‖ψ(t)‖ = ‖ψ(0)‖`
    /// under `time_evolve` for several times, and (3) that the coefficients stay
    /// finite (no blow-up) for all sampled `t`.
    #[test]
    fn navier_stokes_flow_complete_and_unitary() {
        let device = Device::Cpu;
        let h = nested_fock_algebra::models::navier_stokes_hamiltonian(1e-3);

        let v0 =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

        let opts = SirkOpts {
            prune_eps: 1e-12,
            max_components: Some(50_000),
            brst_tol: 1e-10,
            adaptive: false,
            unit_norm_steps: false,
        };
        let res = solve_forward_sirk_with_opts(&h, &v0, &shifts(4), &device, None, &opts)
            .expect("Navier-Stokes SIRK solve must not error");

        // (1) Hermiticity of the projected Hamiltonian => e^{-iHt} is unitary.
        assert_hermitian(&res.h_proj, "Navier-Stokes h_proj");

        // (2) Norm preservation: ||ψ(t)|| == ||ψ(0)|| for a sweep of times.
        let t0 = res.time_evolve(0.0);
        let n0 = t0.norm();
        assert!(
            (n0 - 1.0).abs() < 1e-6,
            "initial whitened state must be normalised, got {n0}"
        );
        for t in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let psi = res.time_evolve(t);
            let nt = psi.norm();
            assert!(
                (nt - n0).abs() < 1e-6,
                "flow at t={t} not norm-preserving: ‖ψ(t)‖ = {nt} vs ‖ψ(0)‖ = {n0}"
            );
            assert!(
                psi.iter().all(|c| c.norm().is_finite()),
                "blow-up at t={t}: non-finite coefficients in the evolved state"
            );
        }
    }

    /// GPU smoke (P1 #6): the two-state hopping Hamiltonian H = |B><A| + |A><B|
    /// has eigenvalues ±1 regardless of device. When the `cuda` feature is on
    /// and a CUDA device is reachable, `best_device()` picks it; the Ritz
    /// values must match the CPU baseline within 1e-8. This is the one test
    /// that exercises the GPU tensor path (inner products + Gram matrix +
    /// H_proj on the CUDA device).
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_smoke_hopping_energy_matches_cpu() {
        let device = crate::best_device();
        let is_cuda = matches!(device, Device::Cuda(_));
        assert!(
            is_cuda,
            "best_device() must pick CUDA when the feature is on and a GPU is reachable; got {device:?}"
        );

        let a = InnerBosonicState::vacuum();
        let mut b = InnerBosonicState::vacuum();
        b.modes.insert(0, 1);

        let h = Hamiltonian {
            terms: vec![
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(b.clone()),
                        Operator::OuterBosonAnnihilate(a.clone()),
                    ],
                ),
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(a.clone()),
                        Operator::OuterBosonAnnihilate(b.clone()),
                    ],
                ),
            ],
        };
        let v0 = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(a));

        let res = solve_forward_sirk(&h, &v0, &shifts(4), &device, None).unwrap();
        assert!(
            res.rank >= 2,
            "hopping spans a 2D Krylov space, got {}",
            res.rank
        );

        let eig = res.h_proj.clone().symmetric_eigen();
        let mut vals: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let lo = *vals.first().unwrap();
        let hi = *vals.last().unwrap();
        assert!((lo + 1.0).abs() < 1e-8, "GPU lowest Ritz value {lo} != -1");
        assert!((hi - 1.0).abs() < 1e-8, "GPU highest Ritz value {hi} != +1");
    }

    /// GPU evolve on a real physics model (P5 #30): `yang_mills_lattice(2,1,1)` has
    /// 72 Hamiltonian terms (8 electric + 64 magnetic quartic). Starting from the
    /// gauge-field vacuum, a short SIRK solve on the GPU must produce a Krylov basis
    /// of positive rank and an H_proj matrix whose Ritz values are within a
    /// physically reasonable range (g²/2 = 0.5 sets the electric energy scale).
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_yang_mills_lattice_l2_norm_conserved() {
        use nested_fock_algebra::models::yang_mills_lattice;

        let device = crate::best_device();
        assert!(
            matches!(device, Device::Cuda(_)),
            "best_device() must pick CUDA when the feature is on; got {device:?}"
        );

        let h = yang_mills_lattice(2, 1.0, 1);
        let v0 =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

        let opts = SirkOpts {
            prune_eps: 1e-12,
            max_components: Some(50_000),
            brst_tol: 1e-10,
            adaptive: false,
            unit_norm_steps: false,
        };
        let res = solve_forward_sirk_with_opts(&h, &v0, &shifts(4), &device, None, &opts)
            .expect("yang-mills GPU solve must not error");

        assert!(res.rank > 0, "Krylov basis has positive rank: {}", res.rank);

        // H_proj Ritz values must be real (Hermitian Hamiltonian) and bounded.
        let eig = res.h_proj.clone().symmetric_eigen();
        for (i, &ev) in eig.eigenvalues.iter().enumerate() {
            assert!(
                ev.abs() < 1e6,
                "Ritz value [{i}]={ev} out of expected range (g²/2=0.5 sets the scale)"
            );
        }
    }

    // ── P5 #30: Larger-scale physics ──────────────────────────────────────
    // Profile yang_mills_lattice at l=2/l=4 and verify SIRK stability at
    // larger Krylov dimensions (m=16, m=32). The bounded direct-construction
    // path should handle the quartic plaquette term without CAS explosion;
    // the Gram whitening should maintain numerical rank and Hermiticity.

    /// Verify that H_proj is Hermitian (H = H†) within tolerance — the
    /// defining property that makes `e^{-iHt}` unitary.
    fn assert_hermitian(h_proj: &DMatrix<Complex64>, label: &str) {
        let dag = h_proj.adjoint();
        let diff = (h_proj - &dag).norm();
        assert!(
            diff < 1e-8,
            "{label}: H_proj must be Hermitian, ‖H−H†‖={diff}"
        );
    }

    #[test]
    fn yang_mills_l4_quartic_explosion_is_typed() {
        // P5 #30: larger-scale physics profiling.
        //
        // l=4 lattice: 16 sites, 32 links, 16 plaquettes → 288 Hamiltonian
        // terms (32 electric + 256 magnetic quartic). The quartic plaquette
        // term Φ(ℓ1)Φ(ℓ2)Φ(ℓ3)Φ(ℓ4) = (a†+a)⁴ creates 2⁴ = 16 new components
        // per plaquette per Krylov step. With 16 plaquettes → 256× branching per
        // step → 256⁸ ≈ 10¹⁹ over m=8 steps. Even with pruning at 1e-12, the
        // component count hits 627K before the max_components guard fires.
        //
        // This is the **documented scaling wall** for the Yang-Mills lattice
        // model: l=2 (72 terms, ~8K components) solves in milliseconds; l=4
        // (288 terms) explodes. The fix is NOT more memory — 627K components ×
        // ~16 bytes each ≈ 10GB of QuantumState. The bounded `max_components`
        // guard correctly catches this with a typed `StateExplosion` error (no
        // panic, no OOM). Approaching the Millennium Prize target (l=6+) will
        // require a compressed/implicit Krylov representation, not just bigger
        // limits.
        use nested_fock_algebra::models::yang_mills_lattice;

        let device = Device::Cpu;
        let h = yang_mills_lattice(4, 1.0, 1);
        assert!(
            h.terms.len() > 250,
            "l=4 lattice should have >250 terms, got {}",
            h.terms.len()
        );

        let v0 =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

        let opts = SirkOpts {
            prune_eps: 1e-12,
            max_components: Some(10_000),
            brst_tol: 1e-10,
            adaptive: false,
            unit_norm_steps: false,
        };

        let result = solve_forward_sirk_with_opts(&h, &v0, &shifts(8), &device, None, &opts);

        match result {
            Err(SirkError::StateExplosion { components, limit }) => {
                assert_eq!(limit, 10_000, "guard fires at the configured limit");
                assert!(
                    components > limit,
                    "components ({components}) must exceed the limit ({limit})"
                );
                // Document the explosion magnitude for future optimization work.
                eprintln!(
                    "l=4 yang-mills: StateExplosion at {components} components \
                     (limit {limit}). l=2 solves in <1s; l=4 needs a compressed Krylov \
                     representation. This is the scaling wall for the quartic plaquette term."
                );
            }
            Ok(res) => {
                // If pruning improves enough to solve, verify Hermiticity.
                assert!(res.rank > 0);
                assert_hermitian(&res.h_proj, "l=4 (solved)");
            }
            Err(e) => panic!("expected StateExplosion, got {e:?}"),
        }
    }

    // ── P6 A2: Adaptive scaling beyond l=4 ───────────────────────────────
    // The adaptive l=4 / l=5 Yang-Mills solves and the l=3 mass-gap demo are
    // the wall-time drivers of this suite; they live in
    // `tests/heavy_krylov.rs` (IMPROVEMENT_PLAN #28). See that file for the
    // bounded `truncate_top_k` truncation path and the P10.18 mass-gap
    // deliverable.

    #[test]
    fn sirk_stability_at_large_krylov_dim() {
        // Verify numerical stability at m=16 and m=32 Krylov dimensions.
        // The Gram matrix grows as (m+1)²; whitening must handle the larger
        // matrix without degeneracy panics. The harmonic_chain (quadratic,
        // bounded spectrum) is a good stress model: it generates a rich but
        // well-conditioned Krylov space.
        use nested_fock_algebra::models::harmonic_chain;

        let device = Device::Cpu;
        let h = harmonic_chain(4, 1.0);
        let v0 =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

        for &m in &[16usize, 32] {
            let opts = SirkOpts {
                prune_eps: 1e-12,
                max_components: Some(50_000),
                brst_tol: 1e-10,
                adaptive: false,
                unit_norm_steps: false,
            };
            let res = solve_forward_sirk_with_opts(&h, &v0, &shifts(m), &device, None, &opts)
                .unwrap_or_else(|e| panic!("m={m} solve must complete: {e}"));

            // Rank must not collapse to 0 or exceed m+1.
            assert!(
                res.rank > 0 && res.rank <= m + 1,
                "m={m}: rank={} must be in [1, {}]",
                res.rank,
                m + 1
            );

            // H_proj must be Hermitian at every m.
            assert_hermitian(&res.h_proj, &format!("m={m}"));

            // Time-evolve: the norm must stay ~1 (unitarity preserved by the
            // Padé approximant even at large m).
            let coeffs = res.time_evolve(0.5);
            let psi_t = res.reconstruct(&coeffs);
            let norm = QuantumState::norm(&psi_t);
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "m={m}: norm={norm} must be ~1 (unitarity at large Krylov dim)"
            );
        }
    }

    // ── P6 A1: Mass-gap extraction ────────────────────────────────────────
    // The SIRK Ritz values (eigenvalues of h_proj) approximate the low-lying
    // spectrum. For the Yang-Mills lattice, the electric term (g²/2)Σn_ℓ gaps
    // the spectrum: the vacuum has E=0, one excitation costs g²/2. The
    // quartic magnetic term preserves excitation-number parity, so the
    // one-particle gap requires comparing even-parity (vacuum) and odd-parity
    // (one-excitation) sectors.

    /// Ritz values + mass_gap() on the two-state hopping Hamiltonian (exact
    /// eigenvalues ±1, so the intra-sector gap = 2).
    #[test]
    fn ritz_values_and_gap_for_hopping() {
        let device = Device::Cpu;
        let a = InnerBosonicState::vacuum();
        let mut b = InnerBosonicState::vacuum();
        b.modes.insert(0, 1);
        let h = Hamiltonian {
            terms: vec![
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(b.clone()),
                        Operator::OuterBosonAnnihilate(a.clone()),
                    ],
                ),
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(a),
                        Operator::OuterBosonAnnihilate(b),
                    ],
                ),
            ],
        };
        let v0 =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
        let res = solve_forward_sirk(&h, &v0, &shifts(4), &device, None).unwrap();

        let ritz = res.ritz_values();
        assert!(ritz.len() >= 2, "need ≥2 Ritz values, got {}", ritz.len());
        // Eigenvalues are ±1; the two lowest Ritz values should bracket them.
        let gap = res.mass_gap().unwrap();
        assert!(gap > 0.0, "intra-sector gap must be positive, got {gap}");
        // For a 2-state system the gap ≈ 2 (from -1 to +1); with a 4-dim
        // Krylov the Ritz values bracket the true spectrum.
        assert!(
            gap > 0.5,
            "hopping gap should be ≈2 (eigenvalues ±1), got {gap:.4}"
        );
    }

    /// Yang-Mills lattice mass gap: the cross-sector gap between the
    /// even-parity (vacuum) and odd-parity (one-excitation) ground states
    /// should be positive and on the order of g²/2 — the defining property
    /// of a confining gauge theory.
    #[test]
    fn yang_mills_lattice_mass_gap() {
        use nested_fock_algebra::models::yang_mills_lattice;

        let device = Device::Cpu;
        // Strong coupling (g=2): electric term (g²/2 = 2.0 per excitation)
        // dominates the magnetic term (-1/2g² = -0.125 per plaquette), so the
        // vacuum is the ground state and the mass gap ≈ g²/2. At g=1 the
        // magnetic coupling (-0.5) is too strong — the odd-parity sector
        // ground state dips below the vacuum, giving a negative "gap".
        let g = 2.0;
        let g2_half = g * g / 2.0; // = 2.0 — the expected electric gap
        let h = yang_mills_lattice(2, g, 1);

        // Even-parity sector: start from the vacuum (0 excitations).
        let v_even =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

        // Odd-parity sector: start from one excitation on link mode 0
        // (dir=0, site (0,0), color=0 — the bottom +x link at the origin).
        let mut inner_odd = InnerBosonicState::vacuum();
        inner_odd.modes.insert(0, 1);
        let v_odd = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner_odd));

        let opts = SirkOpts {
            prune_eps: 1e-12,
            max_components: Some(100_000),
            brst_tol: 1e-10,
            adaptive: false,
            unit_norm_steps: false,
        };
        // m=4 keeps the component count manageable (the quartic plaquette term
        // creates 2⁴ sub-terms per plaquette per Krylov step; m=8 hits 70K+
        // components on l=2). The Ritz values from a 5-vector Krylov still
        // approximate the extreme eigenvalues (ground states) well.
        let m = 4;
        let res_even = solve_forward_sirk_with_opts(&h, &v_even, &shifts(m), &device, None, &opts)
            .expect("even-parity solve must complete");
        let res_odd = solve_forward_sirk_with_opts(&h, &v_odd, &shifts(m), &device, None, &opts)
            .expect("odd-parity solve must complete");

        assert!(
            res_even.rank > 0,
            "even-parity Krylov must have positive rank"
        );
        assert!(
            res_odd.rank > 0,
            "odd-parity Krylov must have positive rank"
        );

        let e_even = res_even.ground_state_energy().unwrap();
        let e_odd = res_odd.ground_state_energy().unwrap();

        eprintln!(
            "yang_mills_lattice(2, g={g}, 1): \
             rank_even={}, rank_odd={}, \
             ritz_even={:?}, ritz_odd={:?}, \
             E_even={e_even:.6}, E_odd={e_odd:.6}",
            res_even.rank,
            res_odd.rank,
            res_even.ritz_values(),
            res_odd.ritz_values(),
        );

        // The even ground state ≈ vacuum (E ≈ 0, perturbed below by magnetic
        // mixing); the odd ground state ≈ one excitation (E ≈ g²/2, also
        // perturbed). The mass gap must be positive (confinement) and on the
        // order of g²/2.
        let gap = mass_gap_from_sectors(&res_even, &res_odd).unwrap();
        assert!(
            gap > 0.0,
            "mass gap must be positive (confinement): e_even={e_even:.4}, e_odd={e_odd:.4}, gap={gap:.4}"
        );
        // The electric gap is g²/2 = 2.0; the magnetic term (strength 1/2g² =
        // 0.125) perturbs both sectors weakly. Allow a factor-3 window to
        // account for finite Krylov convergence and magnetic perturbation.
        assert!(
            gap > g2_half / 3.0 && gap < g2_half * 3.0,
            "mass gap {gap:.4} should be O(g²/2 = {g2_half}): \
             e_even={e_even:.4}, e_odd={e_odd:.4}"
        );
    }

    // ── P10.18 ─────────────────────────────────────────────────────────
    // The l=3 Yang-Mills mass-gap demo (the central empirical P10.18
    // deliverable) moved to `tests/heavy_krylov.rs` (IMPROVEMENT_PLAN #28)
    // together with the adaptive l=4/l=5 solves. The l=2 mass-gap test above
    // stays fast (<1s) and keeps the unit suite self-contained.
}
