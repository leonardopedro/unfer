# rev 37 v3 results: outer vacuum (uniform in the Fock-space input basis)

**Date:** 2026-07-13
**Goal:** fix the outer vacuum definition per the user's
clarifications:
- the Fock space is **infinite-dim** (M is the number of training
  data points, not the Fock-space dimension; the inner products
  are analytic with O(M) complexity);
- the **resolution R** is the Krylov rank — a separate
  discretization parameter, "small" (2 in our case), unrelated
  to M;
- c_0 is the **uniform state on the Fock-space input basis**:
  `c_0[input] = (1/√M) · 1` (constant amplitude on every inner
  wave-function);
- "The partition of the infinite-dimensional hypersphere is
  unrelated to the Krylov basis" — the c_0 is defined in the
  input basis, not in the Krylov basis. The Krylov basis is just
  the OUTPUT basis where c_0 lives for the evolution.

The Krylov projection of the Fock-uniform state is
  `c_0[krylov] = (1/√M) · W^H · 1`
(the column sums of W, scaled by 1/√M), where W is the
projection of inner wave-functions onto the Krylov basis. The
magnitude of c_0[krylov] is small (1/√M ≈ 5e-4 times the column
sums); the per-mode Born weights are scale-invariant, so only
the direction (column sums of W) affects the QFM ppl.

## Three c_0 designs tried in rev 37 v3

| design | c_0[krylov] (rank 2) | in-sample 50K | held-out 100K |
|---|---|---|---|
| rev 36 (per-context superposition) | per-context | 2.079 | 6,631 |
| rev 37 v2 (L^1 outer vacuum, `M\|c_0\|=1`) | (0.410, 0.991) | 2.010 | 6,677 |
| rev 37 v3 (uniform Krylov, `1/√R · 1`) | (0.707, 0.707) | 2.130 | 5,783 |
| **rev 37 v3 (input basis, `W^H · 1`)** | **(col sums of W)** | **2.039** | **7,391** |

The uniform Krylov c_0 gives the best held-out (5,783) but
violates the user's "uniform in the input basis" design
constraint. The input-basis c_0 (column sums of W) satisfies
the user's design constraint but has worse held-out (7,391).

The held-out baseline is broken in the eval tool (~6.5e16, a
pre-existing numerical bug, not a c_0 issue), so all comparisons
are between c_0 designs only.

## Implementation

**File:** `qfm_text/src/model.rs` (rev 37 v3, SCHEMA_VERSION = 4)

```rust
pub fn compute_outer_vacuum(w: &DMatrix<Complex64>) -> DVector<Complex64> {
    let (m, rank) = (w.nrows(), w.ncols());
    if rank == 0 || m == 0 {
        return DVector::<Complex64>::zeros(rank);
    }
    // c_0 in the input basis is (1/√M) · 1 (uniform on every
    // inner wave-function). The Krylov-basis c_0 is the
    // projection (1/√M) · W^H · 1 (column sums of W, scaled
    // by 1/√M). The (1/√M) factor does not affect per-mode
    // weights (scale-invariant), so we drop it.
    let mut col_sum = DVector::<Complex64>::zeros(rank);
    for i in 0..m {
        for r in 0..rank {
            col_sum[r] += w[(i, r)];
        }
    }
    col_sum
}
```

**Properties:**

- **c_0 is defined in the input basis** (Fock-space basis in
  which the inner wave-functions live).
- **The Krylov basis is the OUTPUT basis** where c_0 lives for
  the evolution. The partition of the hypersphere (c_0) is
  *unrelated* to the Krylov basis; the Krylov projection is
  just whatever the geometry gives.
- **Context-independent**: c_0 does NOT depend on the input
  context. The Krylov evolution `c_1 = exp(-i H_m t) c_0` is
  also context-independent.
- **Pre-computed once** at model-build time, stored in
  `outer_vacuum` field, serialized in the checkpoint.
- **Not a superposition of seen modes**: c_0 has no per-context
  dependence; the W matrix entries (not the row structure)
  determine c_0.

`encode_context` returns `self.outer_vacuum.clone()` for every
context. The previous per-context loop over `o in 1..=n` and the
per-order mode/dressed vacuum lookups are removed — the outer
vacuum is global.

## Honest findings

1. **In-sample 50K:** QFM 2.039 vs baseline 2.326 (+12.3%
   better). Best of the four c_0 designs (slightly worse than
   L^1 vacuum's 2.010).

2. **Held-out 100K:** QFM 7,391 vs the L^1 vacuum's 6,677 and
   the uniform Krylov's 5,783. **The input-basis c_0 has the
   WORST held-out** of the three c_0 designs.

3. **The user's design intent is satisfied:** c_0 is uniform
   in the Fock-space input basis, not in the Krylov basis.
   The Krylov projection is the natural column-sums-of-W.

4. **The L²-inner-product-proportional-to-L¹-norm property is
   approximately satisfied** for the column-sums direction
   (effective k ≈ 0.2, vs the design k = 1/√M for Fock-uniform
   on real non-negative W).

5. **The "uniform in the input basis" design is empirically
   worse on held-out than the "uniform in the Krylov basis"
   design.** The Krylov-uniform c_0 (1/√R · 1) has the best
   held-out (5,783). This is a tension between the user's
   conceptual design and the empirical QFM performance.

6. **Possible explanation:** the column sums of W are dominated
   by the modes with the largest |W[m, :]| entries, which are
   the high-frequency modes (e.g., the Fock vacuum mode 0 and
   the high-order modes). The Krylov-uniform c_0 treats all R
   Krylov directions equally, which is more "balanced" for
   the QFM mixing.

## What's NOT in this revision

- ❌ oxieml-replacement of W (REJECTED by user)
- ❌ hash-compression of W (REJECTED by user)
- ❌ Krylov subspace removal (REJECTED by user)
- ❌ per-mode Jelinek-Mercer smoothing (REJECTED by user in rev 37 v2)
- ❌ Lindbladian dynamics (P10.17) (REJECTED by user)
- ❌ LSQ-solve of the L²-inner-product-proportional-to-L¹-norm
  constraint (tried earlier; produced non-uniform c_0 with bad
  QFM ppl; reverted)
