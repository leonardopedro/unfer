# QFM star topology — empirical findings

Extracted from `AGENTS.md`'s "Resolved Limitations" (project review item **X3**).
These are measured results about `qfm` / `qfm_text` training behaviour, not
design intent — each one took a parameter search to establish, and none of them is
recoverable by reading the code. That is why they were being carried as prose in
an agent guide and why they now have a page.

Every claim below is about the **single-mode-per-word (star) topology** unless
stated otherwise. That qualifier is load-bearing: the headline finding is that
this topology does not generalize, which is a statement about the *encoding*, not
about the solver.

---

## The findings

- **Star topology degeneracy**: Single-mode-per-word star topology produces within-class degenerate W-rows because modes sharing the same label have identical Hamiltonian columns. **Distributed multi-mode encoding** (each word is a superposition of 2+ unique dedicated feature modes) breaks this degeneracy and enables 7/7 training accuracy.
- **Gram whitening vs full-rank orthogonalization**: Gram eigendecomposition whitening (with `rel_tol=1e-12` rank truncation) and full-rank orthogonalization (keep all positive eigenvalues) give identical results when no near-null eigenvalues exist. Raw non-orthogonal bases violate unitarity and give random 8/16 classification.
- **Asymmetric label distribution**: Balanced label counts (6e/6o) cause permutation-symmetric Krylov subspace → random 8/16. Asymmetric (5e/7o) breaks symmetry → 12/12 training at m≥3.
- **`compile_channels` API now has `per_mode_weights` parameter**: optional `Option<&HashMap<(u32,u32),f64>>` for per-transition amplitude weights. Pass `None` for uniform λ₁.
- **Krylov dimension m=2 insufficient**: regardless of λ₀ value, m=2 cannot distinguish 12 training inputs in the star topology parity test. Minimum m=3 required.
- **Lambda0 sweet spot**: λ₀=1.0 at m=3 gives 12/12 training; λ₀>1.5 degrades (projector dominates transitions).
- **Anti-learning at m=3**: The 3-dimensional Krylov subspace inverts the label structure at moderate λ₀ (0 < λ₀ ≤ ~5), giving 0% training accuracy. This disappears at λ₀=0 (random) and λ₀≥10 (projector dominates, 100% training even at m=3). The sweet spot m≥5 always works regardless of λ₀.
- **Rank saturation**: For single-mode-per-input parity at any scale (4-bit, 7-bit, 8-bit), the effective gram rank of the Krylov subspace caps at 6. The Krylov dimension m saturates in useful spectral directions at ~6, regardless of mode count (18 to 258) or training set size (12 to 200).
- **No generalization in single-mode star topology**: The Hermitian Hamiltonian with single-mode-per-input encoding achieves 100% training at m≥5 but gives 50% on held-out modes (extrapolation) and only ~54% on within-range held-out (interpolation). Each input is an independent mode with no shared structure — the uniform projector provides no mode-specific generalization.

## API note

`compile_channels` takes a `per_mode_weights` parameter —
`Option<&HashMap<(u32, u32), f64>>` for per-transition amplitude weights. Pass
`None` for uniform `λ₁`.

## Related

`AGENTS.md` → "Resolved Limitations" points here. `docs/QFM_TEXT_STATUS.md` and
`docs/QFM_TEXT_HRM_PLAN.md` cover the text-domain model; the numeric side is in
`docs/NUMERICAL_VALIDATION_GUIDE.md`.
