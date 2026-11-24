# Verifying the continuous-momentum λφ⁴ suite (`P11`–`P13`)

This note records what was built, what was actually measured, how to replay it,
and — importantly — where the certification stops. It accompanies
[`PHI4_ANALYTIC_PLAN.md`](PHI4_ANALYTIC_PLAN.md) and
[`PHI4_NUMERICAL_PLAN.md`](PHI4_NUMERICAL_PLAN.md).

The arena is `L²(ℝ³_k × ℝ_φ)`: momentum `k ∈ ℝ³` and field `φ ∈ ℝ` stay
**continuous**. Nothing here discretizes momentum into sums or a lattice; the
only discrete structures are the Mehler–Hermite occupation window and the Krylov
rank.

**How to read the response intervals.** In the recorded sweep, `exact` denotes a dense
floating-point Padé reference on the same window. The tier labelled `certified` uses
time samples and trapezoidal refinement. Its width-response interval excludes zero
and its coupling interval includes zero. Rigorous bounds on the adjoint approximation,
the error between time samples and the floating-point operations remain open for these
response cells. This qualification applies to the tier names and conclusions in §§5–7.
The rational rank-two identities in the symbolic fixture are evaluated separately.
[Response enclosure work order](PHI4_NUMERICAL_PLAN.md)

## 1. What exists

| artifact | role |
|---|---|
| `nested_fock_algebra/src/phi4_momentum.rs` | `P11` operator: `ScalarMomCont`, `h_cont` on four continuous coordinates, Mehler widths, even/parity windows, `S₃` mode table, frozen comparisons, 8 unit tests |
| `docs/phi4_momentum_nested.cdb` | Cadabra2 derivation of `h_cont`, the 11×3 `S₃` table, the seed moments and the rank-2 return interval |
| `docs/fixtures/phi4_momentum_mehler.json` | exact rational output of that `.cdb` (schema `phi4-momentum-cont-exact/v1`) |
| `fock_sirk/tests/phi4_momentum_sirk.rs` | `P12` suite: rank-2 certificate, cutoff/rank sweeps, paired response estimates, parity + comparisons, fixture cross-check |
| `fock_sirk/examples/phi4_momentum_sirk.rs` | `P13` runner writing the evidence record |
| `docs/fixtures/phi4_momentum_sirk_sweep.json` | the evidence record itself (schema `phi4-momentum-sirk-sweep/v1`) |

The operator keeps the full nonnegative square including `λv⁴/4`, forms the
`φ⁴` product **inside** `h` before `dΓ` (`dΓ(B²) ≠ dΓ(B)²`), and lifts an inner
`cI` to `cN_out`. `dΓ(h)Ω = 0` holds for the outer vacuum.

## 2. Versions and digests

* Cadabra2 `2.5.15 private` (`nix shell nixpkgs#cadabra2`)
* `rustc 1.99.0 (b940084d7 2026-09-28)`
* repository `HEAD = 65a86f85d0ea` when the record was regenerated, with the
  `P11`–`P13` files (including the parallel action and the map-operation fixes
  of §5.5–§5.6) uncommitted

| file | SHA-256 |
|---|---|
| `docs/phi4_momentum_nested.cdb` | `669510d3c55d334b065b0d94c6f7a4d402cdf29408865856178c3b475fb8d709` |
| `docs/fixtures/phi4_momentum_mehler.json` | `9b1e9c15ccbb53c876a4818f8d2898d8faec5f26c54d0664406cbcc9912d15bb` |
| `docs/fixtures/phi4_momentum_sirk_sweep.json` | `11edac4097cc40a4eb858eea56a8068e2b3c504a8ed053aa909dbebe81c9cc3c` |

## 3. Replay

```bash
# symbolic half (needs cadabra2-cli; PYTHONHOME is required on this host)
env PYTHONHOME=<python-in-cadabra-closure> \
  nix shell nixpkgs#cadabra2 --command \
  bash scripts/verify_cdb_modules.sh --strict

# unit tests for the continuous operator
cargo test -p nested_fock_algebra phi4_momentum

# the P12 suite (needs a C compiler, e.g. via `nix shell nixpkgs#gcc`)
cargo test -p fock_sirk --test phi4_momentum_sirk   # 6 tests; ~5 s release

# regenerate the evidence record (~30 min in release: the eight cutoff cells
# cost 22 ms, 0.13 s, 0.75 s, 5.7 s, 35.9 s, 85.1 s, 184.9 s and 406.4 s at
# m = 2 and 46 ms, 0.2 s, 1.7 s, 11.6 s, 70.3 s, 167.2 s, 317.5 s and 534.3 s
# at m = 4; the debug build is impractical for them)
cargo run -p fock_sirk --release --example phi4_momentum_sirk

# the measurements of section 5.5 (serial, then default workers), plus the
# per-operation profile of the Krylov solve
FOCK_SIRK_ACTION_THREADS=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe
cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe
PHI4_PROFILE=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe
```

The fixture itself is regenerated from the module's own `PHI4_MOM_FIXTURE|`
line:

```bash
cadabra2-cli -q -n docs/phi4_momentum_nested.cdb | tee /tmp/phi4-mom.log
python3 - <<'PY'
import json, pathlib
rows = pathlib.Path("/tmp/phi4-mom.log").read_text().splitlines()
ok = [r for r in rows if "(0 = ok)" in r]
assert len(ok) >= 20 and all(r.rsplit(":", 1)[1].strip() == "0" for r in ok)
[fix] = [r.split("|", 1)[1] for r in rows if r.startswith("PHI4_MOM_FIXTURE|")]
pathlib.Path("docs/fixtures/phi4_momentum_mehler.json").write_text(
    json.dumps(json.loads(fix), indent=2, sort_keys=True) + "\n")
PY
```

The `--strict` sweep reports:

```
[PASS] phi4_mehler — ran; 23 zero-intent check(s) all reached 0
[PASS] phi4_momentum_nested — ran; 32 zero-intent check(s) all reached 0
```

Two modules fail that sweep for reasons outside this work:
`nested_fock_enclosure` (`AssertionError: ENC 3: composition defect
mismatch`) and `yang_mills_hamiltonian` (`AssertionError: YMH reflection
changed the inner magnetic square`). Both were failing before this suite was
added and neither is a λφ⁴ module; they are recorded here so the gate's verdict
is not mistaken for a clean run.

## 4. What the symbolic half certifies

Every `(0 = ok)` handle in `docs/phi4_momentum_nested.cdb` reduces to `0`,
including the Mehler-window matrix, the `11 × 3` `S₃` table, `dΓ(h)Ω = 0`, the
outer shifts `0, c, 2c`, and `dΓ(B)² ≠ dΓ(B²)`. The fixture carries the same
table as exact rationals:

* mean `17/8`
* full variance `553/32`
* squared norm outside the three-mode window `9/2`
* `He₂(φ)K₂` coupling `1/2`
* rank-2 two-sided return interval `[319447/320000, 39999/40000]`

The module also certifies, and the units test, the following exact content:

* the continuous quadratic gradient `½(k₁²+k₂²+k₃²)φ²` — six expanded terms,
  with no momentum sum and no spatial-derivative jet constraint;
* retention of the energy constant `λv⁴/4` and the shifted wall
  `λv²x² + λvx³ + λx⁴/4`;
* the continuous Fourier symbol `‖𝓕(∂ₓf)‖² = (2πk)²‖𝓕f‖²` on the Gaussian
  packet `f = exp(−πx²)`, plus `∫(f²)²dx = ∫|Ψ(Q)|²dQ = 1/2`;
* the unconditional coercion square completion, value `7/64` at
  `λ = v = 1, α = 1/2`;
* outer `dΓ(h)` vacuum annihilation, the shift lift `n·c`, and the nonzero
  cross-particle defect `dΓ(B)² − dΓ(B²) = c²N(N−1)`.

`mehler_fixture_matches_the_live_model` in the `P12` suite parses that fixture
and compares it to the live operator entry by entry (3 columns, the 3×3 window,
the three moments) at `1e-12`. The fixture is therefore not decorative: if the
operator's conventions drift, the test fails.

### Mutation controls

The fixture carries the exact differences produced by the deliberate breakages
below, so a silent convention change is detectable:

| mutation | exact difference |
|---|---|
| decouple `k` from `φ` (frozen `jetModel`) — variance | `51/4` (`553/32` → `145/32`) |
| decouple `k` from `φ` — `He₂(φ)K₂` coupling | `1/2` → `0` |
| drop `λv⁴/4` — `U(0)` | `1/4` |
| drop the double contraction — `φ⁴(1)` | `3` |

## 5. What the numerical suite measures

All trajectories use `JointKrylovMethod::ShiftInvert` with Hashimoto decreasing
shifts `γ̃_j = N − h j`; `ForwardProducts` appears only as a labeled diagnostic.

### 5.1 Rank-2 cell, `t = 0.01`, `γ̃ = [10, 5]`, degree-≤2 even window

| quantity | value |
|---|---|
| attained rank | 2 |
| `hermiticity_defect` | `< 1e-12` |
| `Re G_m ≤ 0` (dissipativity) | `≤ 1e-12` |
| outside-window tail² (`projected_action_tail_norms[0]²`) | `4.500000000000001` = `9/2` |
| compressed return probability | `0.9978594340244482` |
| exact return of the same window | `0.998724351897` |
| exact return at `C = 8` / `C = 12` | `0.998298600132` / `0.998298583024` |

The plan's interval `[319447/320000, 39999/40000]` is a statement about the
**physical** return probability; it is met by the converged window
(`0.9982986 ∈ [0.998271875, 0.999975]`). The rank-2 compressed flow is confined
to the degree-2 window, so it reports the *windowed* return and sits `8.65e-4`
below that window's own exact value; the omitted tail (`‖·‖² = 9/2`) enters at
second order. The test asserts the converged interval, the tail value, the
structure of the rank-2 flow, and the measured truncation charge (`≤ 2e-3`)
instead of pretending rank 2 reproduces the full-action number.

### 5.2 Cutoff and rank sweeps, `t = 0.05`

Rank-2 tail sweep (`N = 8, h = 3`), `max_k ‖(I − P_C) H V_k‖`. The `C = 16 … 26`
rows extend the plan's `C ≤ 12` range past 20: the tail keeps falling, but the
Arnoldi steps the smallest scheduled shift (`γ = 40`) needs grow with the
window, and the runtime grows faster than the label count.

| `C` | labels | tail | Arnoldi steps | cell runtime |
|---|---|---|---|---|
| 4 | 15 | `1.7866` | `[6, 6]` | `22 ms` |
| 8 | 70 | `1.4001` | `[17, 18]` | `0.13 s` |
| 12 | 210 | `0.8946` | `[38, 43]` | `0.75 s` |
| 16 | 495 | `0.5067` | `[70, 82]` | `5.7 s` |
| 20 | 1001 | `0.3189` | `[112, 138]` | `35.9 s` |
| 22 | 1365 | `0.2615` | `[137, 174]` | `85.1 s` |
| 24 | 1820 | `0.2129` | `[163, 212]` | `184.9 s` |
| 26 | 2380 | `0.1703` | `[190, 242]` | `406.4 s` |

The `C ≥ 16` cells solve to a `1e-10` relative residual with a 1024-step cap:
at `C = 20` the `1e-13` target stalls at `2.3e-13` and exhausts even 512 steps.
The looser target is six orders below the `1e-3`-scale inside-window error
being measured, and every cell records its policy (`solve_policy`,
`relative_tolerance`, `max_iterations`, and the realised step counts) so the
numbers are never silently mixed. `C ≥ 20` also needs a larger component
budget: the window alone is 1001 labels at `C = 20`, 1820 at `C = 24` and 2380
at `C = 26` (`5.71e6` components), and the dictionary charges one component per
(outer label, inner state) pair, so the default `1e6` limit (and `C = 20`'s
`4e6`) is hit during span registration; the runner now uses `8e6`. Note the tail
is a norm of the *action* outside `P_C`; it need not reach zero as the window
grows, and it is still `0.17` at `C = 26` because `H` moves the seed to labels
the window does not contain. The tail falls by only `0.32 → 0.17` from `C = 20`
to `C = 26` while the cell cost grows `35.9 s → 406.4 s` (§5.5 explains where
that time goes).

Rank sweep on the fixed `C = 8` window (`N = 3m + 2`, `h = 3`):

| `m` | inside-window error | `ForwardProducts` error |
|---|---|---|
| 2 | `1.32e-2` | `1.35e-3` |
| 4 | `3.07e-3` | `5.44e-5` |
| 6 | `2.53e-4` | `6.30e-8` |

Method contrast at fixed rank 4 over the cutoff sweep:

| `C` | `ShiftInvert` error | `ForwardProducts` error | bounded-resolvent steps |
|---|---|---|---|
| 4 | `8.02e-4` | `1.61e-8` | `[6, 6, 6, 6]` |
| 8 | `3.07e-3` | `5.44e-5` | `[18, 18, 19, 19]` |
| 12 | `2.43e-3` | `1.40e-4` | `[43, 44, 45, 46]` |
| 16 | `2.73e-3` | `1.74e-4` | `[81, 82, 83, 83]` |
| 20 | `2.99e-3` | `1.86e-4` | `[136, 137, 139, 141]` |
| 22 | `2.90e-3` | `1.82e-4` | `[171, 173, 175, 176]` |
| 24 | `2.85e-3` | `1.85e-4` | `[209, 212, 214, 216]` |
| 26 | `2.84e-3` | `1.84e-4` | `[239, 242, 245, 247]` |

Two readings, both from the same table. The plan's Hashimoto §3.1 claim
reproduces as **growth with the window** (≈ 10³× from `C = 4` to `C = 12`,
`1.2e4×` to `C = 20`, `1.1e4×` at `C = 26` where it levels off), not as a
rank-for-rank deficit: on these bounded windows the shifted polynomial flow is
the *more* accurate of the two. And the *inside-window* error of the primary
solver is flat — `8.0e-4, 3.07e-3, 2.43e-3, 2.73e-3, 2.99e-3, 2.90e-3,
2.85e-3, 2.84e-3` — i.e. once the window covers the seed's support (from
`C = 8` on) the remaining error is set by the rank, not by `C`, while the tail
keeps shrinking. The `C = 20 … 26` entry is a shallow plateau with no monotone
trend: `2.99e-3, 2.90e-3, 2.85e-3, 2.84e-3` differs in the third digit, which
is the `solve_policy` tolerance's neighbourhood, so those four cells are read as
equal, not as a decaying sequence. Growing `C` past 12 therefore buys tail, not
accuracy, at a steeply rising cost (1.7 s → 11.6 s → 70.3 s → 167.2 s →
317.5 s → 534.3 s per `m = 4` cell).
That is exactly why it stays a labeled diagnostic. At `m = 2` the contrast is
not yet visible (`1.9e-3, 1.3e-3, 1.3e-3`), so the m = 2 sweep is used for the
plan-literal tail measurement and the contrast is recorded at `m = 4`.

### 5.3 Paired return responses, `C = 8, m = 4, t = 0.05, N = 250, h = 50`

`p_exact` is the dense floating-point Padé reference for the *declared window*. Each cell
carries three numbers:

* `sharp` = `|p_sirk − p_exact|`, the measured compressed-flow discrepancy
  (S43 sharp tier);
* `state_charge` = `2ε + ε²` on the sampled full-action **state** estimate
  `ε = evolution_bound` (see §6.1);
* `certified` = `2a + a²` on the sampled **amplitude-overlap** estimate
  `a` from `sm_defect::amplitude_terms`, which prices `<ψ|e>` instead of `‖e‖`.

| cell | `p_sirk` | `p_exact` | `sharp` | `ε` (state) | `a` (amplitude) | measured `\|Δamplitude\|` | `certified` |
|---|---|---|---|---|---|---|---|
| base (`λ=1`, `width_k=1`) | `0.966749778031` | `0.966736506315` | `1.33e-5` | `0.1805` | `3.814e-4` | `3.553e-5` | `7.629e-4` |
| null control (same again) | `0.966749778031` | `0.966736506315` | `1.33e-5` | `0.1805` | `3.814e-4` | `3.553e-5` | `7.629e-4` |
| coupled (`λ=1.1`) | `0.965119884383` | `0.965123360010` | `3.48e-6` | `0.2045` | `5.117e-4` | `4.771e-5` | `1.024e-3` |
| spread (`width_k=1.5`) | `0.909965726101` | `0.910575997445` | `6.10e-4` | `0.6293` | `5.929e-3` | `3.647e-4` | `1.189e-2` |

The penultimate numeric column is the *measured* amplitude gap
`|⟨ψ|ψ(t)⟩ − ⟨ψ|w(t)⟩|` on the declared window, recorded beside the bound
(`measured_amplitude_gap`, with `amplitude_bound_dominates_gap`), and the measured `a` exceeds
the gap by `10.7×`, `10.7×` and `16.3×`.

On the base cell the sampled amplitude estimate is `3.81e-4` and the state
estimate is `0.181`. The sampled overlap is `9.06e-6`. The replacement
integral is `3.72e-4`, about 6.8 times smaller than the calculated `M²/2`.

The tests compare both charges against the dense-window reference. The last
column below reports zero exclusion for the amplitude estimate.

| response | exact difference | sharp interval | certified (amplitude) interval | excludes 0 |
|---|---|---|---|---|
| null | `0` | `[-2.65e-5, 2.65e-5]` | `[-1.53e-3, 1.53e-3]` | no (correct) |
| `λ` | `+1.6131e-3` | `[1.5964e-3, 1.6299e-3]` | `[-1.57e-4, 3.416e-3]` | **no** (1.1× short) |
| `width_k` | `+5.6161e-2` | `[5.5537e-2, 5.6784e-2]` | `[4.413e-2, 6.944e-2]` | **yes** |

The recorded differences use `a − b`. Raising λ lowers the reference return
probability by about 0.00161. Increasing the physical packet width lowers it
by about 0.0562. The amplitude estimates contain both reference differences
and resolve the sign only for the wider packet. The identical-cell interval
contains zero.

### 5.4 Parity, inner bound, and comparison contrast

* Both parity cells conserve `φ ↦ −φ` exactly: every reconstructed label of an
  even seed has even `φ` occupation and vice versa.
* Every Ritz value satisfies `θ ≥ 7/64` (unconditional inner bound); the seeded
  values are `1.1503` (even) and `2.7509` (odd), an inner excitation splitting
  of `1.6006`. This is an *inner* excitation, never the outer vacuum gap.
* Frozen comparisons: continuous `(mean 17/8, variance 553/32, He₂K₂ 1/2)`,
  static jet `(17/8, 145/32, 0)`, 2-site lattice `(13/4, 337/16)`.

The continuous coupling `1/2` is the discriminator: the static-jet comparison
has the same mean and zero momentum–field coupling, and the lattice comparison
is a different operator entirely.

### 5.5 The parallel action, and what it does *not* buy (`P13`)

Pushing the sweep past `C = 20` was attempted by parallelising the action in
`nested_fock_algebra/src/exact_inner.rs`. Two splits were built and measured on
this host (12 cores, 30 GB); only one survived.

* **Kept — sum over terms.** A top-level `InnerExpr::Sum` gives each term its
  own worker; the parent then merges each term's image **as a whole map, in term
  order**. This is bit-identical to the serial path, not merely close: a term
  contributes at most one value per output label, so every label still sees its
  contributions in term order, and `Budget::add`'s remove-on-zero behaves the
  same. Workers keep private work counters and the parent charges the aggregate
  through `Budget::charge`, which errors only when the total *exceeds*
  `max_work` — the limit is unchanged.
* **Kept — outer labels.** `apply_outer_labels` (the multi-particle branch of
  `d_gamma`) splits contiguous label chunks across workers and returns
  per-entry contributions, merged in the original order. This sweep never
  exercises it: every window label is 1-particle, so these states take the
  `d_gamma` fast path.
* **Rejected — the inner per-entry map.** Splitting `Scalar`/`Ladder` maps
  across workers measured **slower**: `46.0 ms → 62.5 ms` per `C = 20` matvec.
  One fork/join per factor does not amortise against a per-entry cost of a few
  map lookups. It was removed rather than left behind a threshold.

`FOCK_SIRK_ACTION_THREADS` forces the worker count (that is how the serial
column below was measured); the width is also capped by the term count, and
`h_cont` has **three** top-level terms — `kinetic`, `gradient`, `wall` — so the
sum split can never exceed 3-way.

Measured with `fock_sirk/examples/phi4_momentum_parallel_probe.rs`:

| quantity (`C = 20`, 1001 labels) | serial | parallel | ratio |
|---|---|---|---|
| `d_gamma` on a dense window state, three runs | `45.55 / 45.61 / 46.12 ms` | `29.78 / 30.34 / 29.95 ms` | **1.50–1.54×** |
| `m = 2` Krylov solve | `61.67 / 62.92 s` | `62.35 / 63.30 s` | `0.99×` |
| total Arnoldi steps | `528` | `528` | — |
| return probability | `9.80001510278448618e-1` | identical to all 17 digits | — |

The third run is after the worker-count helpers were factored into
`requested_threads`/`available_threads`; the probability it reports is the
same `0.9800015102784486` the record stores for `C = 20`, which is the
end-to-end check that the split really is order-preserving and not merely
accurate.

So the action parallelisation alone is a **1.5× matvec and a 1.00× sweep** (the
solve column predates §5.6's fixes). The matvec is not the binding cost at this
size: 528 steps at `45.6 ms` is ≈ `24 s` of the `62 s` solve (≈ 39 %), while the
rest is full-GMRES orthogonalisation — quadratic in the step count,
`528 → 918` steps from `C = 20` to `C = 26` — plus nested-vector arithmetic per
1001-label `axpy`. §5.6 measures that arithmetic, rebuilds it, and reports what
parallelising it would have cost.

Bit-identity is asserted, not assumed: `mod parallel_action_tests` in
`exact_inner.rs` runs three release tests — an order-preserving worker merge on
a 96-label 2-particle state (threads `1/2/3/4/7/16`, bit-equal vectors and
equal `d_gamma`), a bisected `max_work` limit that must trip at the same work
total serial and at 4 threads, and a 1001-label window state mapped through
forced 1 / default / 7 threads. They pass in `0.2 s`.

### 5.6 What actually cost the time: the two map operations (`P13`)

`PHI4_PROFILE=1` makes `gmres_shift` accumulate wall time for the action
matvec, the nested `axpy`, the nested inner product and the per-step basis SVD.
At `C = 20` (1001 labels), `m = 2`, the two shift solves of the first basis entry
(`[112, 138]` steps) are:

| phase | 112-step solve | 138-step solve |
|---|---|---|
| action matvec | `0.18 s` | `0.23 s` |
| `joint_axpy` inside MGS (the row below is inside it) | `3.25 s` (12656 calls, `0.257 ms`) | `5.53 s` (19182 calls, `0.288 ms`) |
| `joint_inner_product` | `1.02 s` (12656 calls, `0.081 ms`) | `1.77 s` (19182 calls, `0.092 ms`) |
| per-step basis SVD | `0.19 s` | `0.42 s` |

So the action is ≈ **1.4 %** of the solve and the two nested-vector map
operations are ≈ **90 %** of it. Rebuilding GMRES with Givens rotations would
target the `0.6 s` of SVD — about 2 % — while changing every pinned digit, so it
was rejected on the measurement, not on taste.

Both operations were also doing gratuitous work per entry:

* `joint_inner_product` walked `a` and ran a `b.get(&label)` **binary search**
  per entry. Both vectors come from the same window dictionary in label order,
  so it is now a two-pointer merge: same products, same left-to-right
  accumulation order, no search. `dot_ms` `0.27 → 0.039 ms` at 1001 labels.
* `joint_axpy` recomputed each summand and then `insert(basis.clone(), …)`,
  cloning a `JointOuterBasis` (and forcing a node split) per entry. It now
  updates in place through `get_mut`, inserting only genuinely new labels and
  still deleting zeros. `axpy_ms` `0.85 → 0.39 ms`.

Bit-identity is structural rather than lucky: no product changes and no
association changes, so every label still sees the same contributions in the
same order. The end-to-end check at `C = 20`, `m = 2` is
**`63.2 s → 24.1 s` (2.6×)** with the same 528 Arnoldi steps and the same
`9.80001510278448618e-1`. In the record, where the fix applies to all eight
cells and both ranks, the `m = 2` cells went `74.7 → 35.9 s` (2.08×),
`172.2 → 85.1 s` (2.02×) and `367.1 → 184.9 s` (1.99×).

The regression check is stronger than a spot comparison: the record was
regenerated with the new map operations and compared field by field against the
previous record with `runtime_ms` excluded. The only differences are the two
new `C = 26` rows — every tail, probability, residual, defect, step count and
component count of the seven existing cells is **exactly** equal. Bit-identity
modulo the timing fields, in other words — the earlier record's numbers stay
quotable.

**Parallelising these two operations was tried and loses:**

| operation (`C = 20`, 1001 labels) | serial | 2 workers | 4 workers | 8 workers |
|---|---|---|---|---|
| `joint_inner_product` (chunk partials, chunk order) | `0.039 ms` | `0.19 ms` | `0.21 ms` | `0.21 ms` |
| `joint_axpy` (chunk deltas, applied in label order) | `0.39 ms` | `1.32 ms` | `1.21 ms` | `1.21 ms` |

The axpy row needs a fresh copy of the target per repetition (the operation
mutates it), so all four columns share that cost; a 1-worker run of the same
chunked helper still measures `0.65 ms` against `0.39 ms` for the direct call,
i.e. the split's bookkeeping alone is a 1.7× tax. A fork/join on this host
costs `0.1–0.2 ms`, several times the whole 1001-label dot and a comparable
share of the axpy — the same lesson as the rejected per-entry map split in
§5.5, and the reason the sweep is not made faster by adding workers. The
parallel dot is additionally **not bit-identical**: summing chunk partials in
chunk order associates the additions differently from the left-to-right fold,
and the returned scalar differs from the serial one by `1.1e-15` relative,
which is enough to break the pinned 17-digit probability.

So the honest headline for `P13` is: a **1.5× action matvec, no gain from
parallelising the Krylov arithmetic, and a 2.0–2.6× end-to-end win from
replacing two per-entry map operations with structural ones**. `C = 22 … 26`
were reached with that 2× plus wall clock (`30 min` for the recorded sweep).

## 6. Limitations — read this before quoting a number

1. **How the defect estimates changed.**

   **(a) The state envelope is no longer shift-scaled.** `defect_action_envelope`
   used to be a grid-plus-Lipschitz bound whose dominant term was
   `t/8 · ‖DG‖_F · ‖c‖ · e^{ωt}`. On this operator `G = γI − B⁻¹` with `γ` up to
   `200`, every entry of `G` carries the shift while `e^{sG}` stays `O(1)`, so
   `‖DG‖_F ≈ 1.55e4` against a trajectory whose actual maximum is `3.61`: the
   product prices the shift, not the defect. It is now a grid-converged sampled
   maximum along the returned trajectory.

   | cell | envelope before | now | state bound before | now |
   |---|---|---|---|---|
   | base | `100.2` | `3.611` | `5.949` | `0.1805` |
   | coupled | `111.2` | `4.090` | `6.537` | `0.2045` |
   | spread | `167.4` | `12.586` | `10.601` | `0.6293` |

   **(b) The charge is on the amplitude, not the state norm.** A probability is
   quadratic in the amplitude, so `| |a|² − |b|² | ≤ 2|a−b| + |a−b|²`; with
   `A = ⟨ψ|ψ(t)⟩ − ⟨ψ|w(t)⟩` and `e(0) = ℓ`, variation of constants gives
   `A = ⟨ψ|e(t)⟩` with `e(t) = e^{-itH}ℓ − ∫₀ᵗ e^{-i(t−s)H} r(s) ds` and
   `r(s) = D e^{sG}c`. In the return-amplitude pairing the exact test vector is
   `exp(+i(t−s)H)ψ`. The code uses the forward reduced state `w(t−s)` in its
   sampled overlap and a cumulative trapezoid for the replacement term. An
   adjoint residual bound or a proved symmetry on this pairing, together with
   verified quadrature remainders, is needed to turn that calculation into
   an upper bound.

   * The **overlap** is computed, not majorised. This is where the
     near-orthogonality lives: it is `9.06e-6` on the base cell, against a
     measured amplitude difference of `3.553e-5` and a state bound of `0.18`.
   * The **replacement** term is the price of substituting `w` for `u`. It is
     bounded above by `M²/2` (`M = ∫₀ᵗ‖r‖`, the whole square being twice the
     triangle) but computed directly as `∫₀ᵗ F(t−s)‖r(s)‖ds`, which is `6.8×`
     smaller here because the defect grows with `s` — `F` is smallest exactly
     where `‖r(s)‖` is largest.

   **(c) Why the coupling response still misses.** The replacement term is the
   floor of the technique, and its slack is `‖u−w‖ ≤ F`, i.e. a
   Cauchy–Schwarz statement about the error *direction*. Going one order
   further would require propagating `r(σ)`, which lives outside the Krylov
   basis, so it needs its own state bound — the recursion returns to (a). The
   measured facts at this operator/window:

   | cell | amplitude bound `a` | `2a + a²` | measured `\|Δamplitude\|` | true state error | true prob. error |
   |---|---|---|---|---|---|
   | base | `3.814e-4` | `7.629e-4` | `3.553e-5` | `7.85e-3` | `1.33e-5` |
   | coupled | `5.117e-4` | `1.024e-3` | `4.771e-5` | — | `3.48e-6` |
   | spread | `5.929e-3` | `1.189e-2` | `3.647e-4` | `2.14e-2` | `6.10e-4` |

   On the base cell the amplitude estimate is `10.7×` the measured amplitude
   gap. The sampled width interval excludes zero;
   the coupling interval includes it. The dense-window reference resolves both
   differences at the precision reported in §5.3.
2. **The outside-window tail is reported separately** and is first order in
   neither `t` nor the rank; it must be shrunk by raising `C`
   (`1.79 → 1.40 → 0.89 → 0.51 → 0.32 → 0.26 → 0.21 → 0.17` for
   `C = 4, 8, 12, 16, 20, 22, 24, 26`). It does not reach zero, and the last
   rows cost 5.7 s, 36 s, 85 s, 185 s and 406 s per `m = 2` cell, so the window
   cannot be pushed much further by brute force — and, as §5.5–§5.6 measure,
   neither by parallelising the action matvec nor by parallelising the Krylov
   arithmetic.
3. **`ForwardProducts` is not worse here** at equal rank (see §5.2). Do not
   quote the plan's "Arnoldi deteriorates" as a rank comparison on bounded
   windows; the observable degradation is growth with `C`.
4. **GMRES iteration counts are exposed and recorded.** `JointSirkResult`
   carries `shift_solve_iterations` (one Arnoldi step count per rational basis
   solve, in the same order as `shift_solve_error_bounds`) and
   `function_solve_iterations` (one per bounded-resolvent solve), with the
   `max_*` / `total_solve_iterations` accessors. The response cells need
   `[12, 13, 14, 15]` basis steps and `[15, 15, 15, 15]` resolvent steps   under the `64`-step cap at `1e-13`; the cutoff sweep climbs to `[163, 212]` at
   `C = 24` and `[190, 242]` at `C = 26` (`[209, 212, 214, 216]` and
   `[239, 242, 245, 247]` at rank 4) and needs the looser `1e-10`/1024 policy
   described in §5.2 (528 total steps at `C = 20`, 800 at `C = 24`, 918 at
   `C = 26`). Every count is in the record — no `null` stands in for a
   measurement.
5. **`cadabra2-cli` and a C compiler are Nix-only on this host**; the CDB gate
   needs `PYTHONHOME` from the Cadabra2 closure to avoid
   `ModuleNotFoundError: _struct`.
6. The two pre-existing `.cdb` failures in §3 are unrelated to this suite and
   remain open.

## 7. Honesty summary

* Symbolic (`cadabra`/`.cdb`): all 32 λφ⁴ zero-intent checks reduce to `0`;
  the fixture is cross-checked against the live operator.
* Dense reference: floating-point Padé evolution of each declared window.
* Measured: every probability, tail, Ritz value, response and charge above,
  reproducible with the commands in §3.
* Measured and reported against the stated goal: the parallel action is
  **1.53×** on the `C = 20` matvec and **0.99×** end to end, because full-GMRES
  orthogonalisation dominates that solve (§5.5). Parallelising the nested-vector
  inner product and axpy instead made them **5× and 3× slower** and the
  parallel dot is not even bit-identical, so it was not adopted (§5.6).
* Measured and adopted: rebuilding those two map operations (merge walk instead
  of per-entry binary search; in-place `get_mut` instead of clone-and-insert)
  is bit-identical and gives **2.6×** on one `C = 20` solve and **2.0–2.1×** on
  the recorded cells, which is what made `C = 26` affordable (§5.6).
* On the base cell the probability charge fell from about `0.394` (state)
  to `0.000763` (amplitude). The paired amplitude charge is `0.001786` for
  coupling and `0.01266` for momentum spread. Rigorous response signs for
  the full fibre orbit remain open. [Enclosure work order](PHI4_NUMERICAL_PLAN.md)
