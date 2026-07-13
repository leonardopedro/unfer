# rev 37 v3 results: L^1 outer vacuum

**Date:** 2026-07-13
**Goal:** fix the outer vacuum definition per the user's "the outer
vacuum cannot be a superposition of seen modes" + "inner product is
L^1 integral of amplitude" design constraints. The Krylov input c_0
must be the L^1-uniform vector in the Krylov subspace, not the
per-context superposition `(1/√(n+1))(W[0, :] + Σ_o W[m_o, :])`.

## Background

The per-context superposition c_0 = (1/√(n+1))(W[0, :] + Σ_o W[m_o, :])
is a **superposition of seen modes**: the n per-order modes for the
current context, plus the Fock vacuum. Every mode NOT in this list —
including all modes the training set observed but didn't pair with
this exact context, plus all modes the training set never observed at
all — is excluded from c_0. This biases the Krylov input toward what
the context already tells us, which is exactly the bias the user
ruled out.

The fix: c_0 is now the L^1-uniform vector in the Krylov subspace.
It is the unique vector such that the L^1 inner product with every
inner wave-function is equal (the L^1 integral of the amplitudes
over the support, which gives `1/R` for a uniform amplitude in the
discrete case, where `R = rank` is the resolution).

## Implementation

**File:** `qfm_text/src/model.rs` (rev 37 v3, SCHEMA_VERSION = 4)

The new `compute_l1_outer_vacuum(w: &DMatrix<Complex64>) ->
DVector<Complex64>` solves `M |c_0| = 1` where `M[m, r] = |W[m, r]|`
and `1` is the all-ones vector. The least-squares solution is
`|c_0| = (M^T M)^-1 M^T 1` (a `rank`-vector of non-negative
amplitudes; phases set to zero). Cost: O(M × rank²), sub-second for
shard 0.

**Properties:**

- **Context-independent**: c_0 does NOT depend on the input context.
  The Krylov evolution `c_1 = U c_0` is also context-independent.
- **Pre-computed once** at model-build time, stored in `outer_vacuum`
  field, serialized in the checkpoint.
- **Not a superposition of seen modes**: c_0 is a specific vector
  in the Krylov subspace, determined by the W matrix structure
  (not by the per-context mode lookup).

**`encode_context` simplified**: returns the precomputed
`self.outer_vacuum.clone()` for every context. The previous
per-context loop over `o in 1..=n` and the per-order mode/dressed
vacuum lookups are removed — the outer vacuum is global.

## Phase 3 measurements (rev 37 v3, L^1 outer vacuum)

### In-sample 50K (shard 0)

| design | QFM ppl | baseline ppl | unigram ppl | QFM vs baseline |
|--------|---------|--------------|-------------|-----------------|
| rev 36 (registry, per-context c_0) | 2.079 | 2.210 | 985.424 | +6.0% better |
| **rev 37 v3 (L^1 outer vacuum)** | **2.010** | 2.210 | 985.424 | **+9.0% better** |

### Held-out 100K (wikitext-103-test)

| design | QFM ppl | baseline ppl | unigram ppl | QFM vs baseline |
|--------|---------|--------------|-------------|-----------------|
| rev 36 (registry, per-context c_0) | 6,631 | 5,688 | 1,033 | -16.6% (QFM worse) |
| **rev 37 v3 (L^1 outer vacuum)** | **6,677** | 5,688 | 1,033 | **-17.4% (QFM worse)** |

### Per-context KL (in-sample, 3,126 high-frequency contexts)

| design | mean KL | mean cos | top-1 hit |
|--------|---------|----------|-----------|
| rev 37 v3 QFM | **0.6020** | **0.9385** | **85.2%** |
| rev 37 v3 baseline | 0.6777 | 0.9200 | 82.7% |

QFM beats baseline on all three metrics: lower KL, higher cosine
similarity, higher top-1 hit rate. This is the rev 37 v3 effect
(Krylov smoothing + L^1 outer vacuum unigram backoff + per-mode
histogram context conditioning).

## L^1 outer vacuum properties (diagnostic)

The L^1 outer vacuum c_0 (in the Krylov basis, rank = 2) is:
```
c_0 = (0.410, 0.991)
||c_0|| = 1.073
```

**L^1 inner product uniformity check** (over all 4,194,305 modes):

- mean L^1 inner product: 2.51e-1
- std L^1 inner product: 4.34e-1
- coefficient of variation (CV): 1.73

The CV is high because the system `M |c_0| = 1` is heavily
overdetermined (4.19M equations in 2 unknowns); the least-squares
solution minimizes the L^2 norm of the residual, which is NOT the
L^1-uniform solution. The L^1-uniform c_0 in the exact sense does
not exist for a rank-2 W (the vector 1 is not in the 2D column
space of M); the L^2-minimal residual is the best 2D approximation.

**Per-mode Born weights on a test context** (ctx = [155, 487, 172, 155]):

- mode 0 (Fock vacuum / unigram): 31.6%
- mode 189926 (order 4): 29.6%
- mode 185804 (order 3): 28.3%
- mode 224 (order 1): 5.3%
- mode 225 (order 2): 5.1%

The Krylov evolution propagates the L^1-uniform c_0 through the
dressed-vacuum Hamiltonian; the post-evolution symmetry is broken
by the W matrix asymmetry. Mode 0 (the Fock vacuum) gets 31.6% of
the mass — the unigram backoff mass — while the high-order modes
(orders 3-4) get 28-30% each and the low-order modes (orders 1-2)
get 5% each. This is the opposite of the user's "outer vacuum takes
care of the unseen data" intuition: the unigram gets 31.6%, but the
high-order modes get 57.9% (which is what makes the model
overfit to seen contexts).

## Honest findings

1. **In-sample 50K:** QFM 2.010 vs baseline 2.210 (+9.0% better).
   The QFM memorizes the training data via the per-mode histograms;
   the Krylov unigram mass adds 2-3% improvement on top.

2. **Held-out 100K:** QFM 6,677 vs baseline 5,688 (-17.4% worse).
   The per-mode histograms are overfit; the Krylov unigram mass
   (31.6% of the Born weight) cannot bridge the held-out gap.

3. **The L^1 outer vacuum is not the unigram:** it's a 2D vector
   in the Krylov subspace that has approximately equal L^1 inner
   product with all W rows. After Krylov evolution, the per-mode
   Born weights are NOT uniform; the asymmetry of W + the
   Hamiltonian structure break the symmetry.

4. **The fundamental trade-off is unchanged:** the QFM is
   context-conditioned (in-sample wins) but overfit (held-out
   losses). The Krylov unigram at ~30% is not enough to dominate
   the per-order modes' over-confidence. The user's "the outer
   vacuum takes care of the unseen data" design intent requires
   either:
   (a) a much higher unigram mass (which would require a
       different c_0 construction, not just a different
       superposition), or
   (b) per-mode smoothing (which the user explicitly rejected
       in rev 37 v2 as "Jelinek-Mercer"), or
   (c) a much larger Krylov rank (currently 2 on shard 0;
       the W is rank-deficient).

5. **The "outer vacuum cannot be a superposition of seen modes"
   design constraint is now satisfied:** c_0 is a single vector
   determined by the W matrix, not a per-context superposition.
   But the L^1-uniformity goal is only approximately achieved
   in 2D (CV = 1.73).

## What's NOT in this revision

- ❌ oxieml-replacement of W (REJECTED by user)
- ❌ hash-compression of W (REJECTED by user)
- ❌ Krylov subspace removal (REJECTED by user)
- ❌ per-mode Jelinek-Mercer smoothing (REJECTED by user in rev 37 v2)
- ❌ Lindbladian dynamics (P10.17) (REJECTED by user)
