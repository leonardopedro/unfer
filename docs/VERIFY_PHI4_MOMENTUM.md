# Verifying the continuous-momentum λφ⁴ suite (`P11`–`P13`)

This note records what was built, what was actually measured, how to replay it,
and — importantly — where the certification stops. It accompanies
[`PHI4_ANALYTIC_PLAN.md`](PHI4_ANALYTIC_PLAN.md) and
[`PHI4_NUMERICAL_PLAN.md`](PHI4_NUMERICAL_PLAN.md).

The arena is `L²(ℝ³_k × ℝ_φ)`: momentum `k ∈ ℝ³` and field `φ ∈ ℝ` stay
**continuous**. Nothing here discretizes momentum into sums or a lattice; the
only discrete structures are the Mehler–Hermite occupation window and the Krylov
rank.

## 1. What exists

| artifact | role |
|---|---|
| `nested_fock_algebra/src/phi4_momentum.rs` | `P11` operator: `ScalarMomCont`, `h_cont` on four continuous coordinates, Mehler widths, even/parity windows, `S₃` mode table, frozen comparisons, 8 unit tests |
| `docs/phi4_momentum_nested.cdb` | Cadabra2 derivation of `h_cont`, the 11×3 `S₃` table, the seed moments and the rank-2 return interval |
| `docs/fixtures/phi4_momentum_mehler.json` | exact rational output of that `.cdb` (schema `phi4-momentum-cont-exact/v1`) |
| `fock_sirk/tests/phi4_momentum_sirk.rs` | `P12` suite: rank-2 certificate, cutoff/rank sweeps, certified responses, parity + comparisons, fixture cross-check |
| `fock_sirk/examples/phi4_momentum_sirk.rs` | `P13` runner writing the evidence record |
| `docs/fixtures/phi4_momentum_sirk_sweep.json` | the evidence record itself (schema `phi4-momentum-sirk-sweep/v1`) |

The operator keeps the full nonnegative square including `λv⁴/4`, forms the
`φ⁴` product **inside** `h` before `dΓ` (`dΓ(B²) ≠ dΓ(B)²`), and lifts an inner
`cI` to `cN_out`. `dΓ(h)Ω = 0` holds for the outer vacuum.

## 2. Versions and digests

* Cadabra2 `2.5.15 private` (`nix shell nixpkgs#cadabra2`)
* `rustc 1.99.0 (b940084d7 2026-09-28)`
* repository `HEAD = c36aa895b42d` when the record was generated, with the
  `P11`–`P13` files uncommitted

| file | SHA-256 |
|---|---|
| `docs/phi4_momentum_nested.cdb` | `669510d3c55d334b065b0d94c6f7a4d402cdf29408865856178c3b475fb8d709` |
| `docs/fixtures/phi4_momentum_mehler.json` | `9b1e9c15ccbb53c876a4818f8d2898d8faec5f26c54d0664406cbcc9912d15bb` |
| `docs/fixtures/phi4_momentum_sirk_sweep.json` | `2f09e65b472a4b0ec5e0d1ac57f83fb52e7a736c7eafc09b07430e39d441fd87` |

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

# regenerate the evidence record (~4 min in release: the C = 16 and 20 cells
# dominate; the debug build is impractical for them)
cargo run -p fock_sirk --release --example phi4_momentum_sirk
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

## 5. What the numerical suite certifies

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

Rank-2 tail sweep (`N = 8, h = 3`), `max_k ‖(I − P_C) H V_k‖`. The `C = 16, 20`
rows extend the plan's `C ≤ 12` range: the tail keeps falling, but each doubling
of the window roughly doubles the Arnoldi steps the smallest scheduled shift
(`γ = 40`) needs, and the runtime grows superlinearly.

| `C` | labels | tail | Arnoldi steps | cell runtime |
|---|---|---|---|---|
| 4 | 15 | `1.7866` | `[6, 6]` | `2 ms` |
| 8 | 70 | `1.4001` | `[17, 18]` | `67 ms` |
| 12 | 210 | `0.8946` | `[38, 43]` | `1.1 s` |
| 16 | 495 | `0.5067` | `[70, 82]` | `11.5 s` |
| 20 | 1001 | `0.3189` | `[112, 138]` | `73.4 s` |

The `C ≥ 16` cells solve to a `1e-10` relative residual with a 1024-step cap:
at `C = 20` the `1e-13` target stalls at `2.3e-13` and exhausts even 512 steps.
The looser target is six orders below the `1e-3`-scale inside-window error
being measured, and every cell records its policy (`solve_policy`,
`relative_tolerance`, `max_iterations`, and the realised step counts) so the
numbers are never silently mixed. `C = 20` also needs a larger component
budget: the window alone is 1001 labels and the dictionary charges one
component per (outer label, inner state) pair, so the default `1e6` limit is
hit during span registration. Note the tail is a norm of the *action* outside
`P_C`; it need not reach zero as the window grows, and it is still `0.32` at
`C = 20` because `H` moves the seed to labels the window does not contain.

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

Two readings, both from the same table. The plan's Hashimoto §3.1 claim
reproduces as **growth with the window** (≈ 10³× from `C = 4` to `C = 12`,
`1.2e4×` to `C = 20`), not as a rank-for-rank deficit: on these bounded windows
the shifted polynomial flow is the *more* accurate of the two. And the *inside-
window* error of the primary solver is flat — `8.0e-4, 3.07e-3, 2.43e-3,
2.73e-3, 2.99e-3` — i.e. once the window covers the seed's support (from
`C = 8` on) the remaining error is set by the rank, not by `C`, while the tail
keeps shrinking. Growing `C` past 12 therefore buys tail, not accuracy, at a
steeply rising cost (2.2 s → 21 s → 127 s per `m = 4` cell).
That is exactly why it stays a labeled diagnostic. At `m = 2` the contrast is
not yet visible (`1.9e-3, 1.3e-3, 1.3e-3`), so the m = 2 sweep is used for the
plan-literal tail measurement and the contrast is recorded at `m = 4`.

### 5.3 Certified responses, `C = 8, m = 4, t = 0.05, N = 250, h = 50`

`p_exact` is the exact dense Padé evolution of the *declared window*. Each cell
carries **two** charges:

* `sharp` = `|p_sirk − p_exact|`, the measured compressed-flow discrepancy
  (S43 sharp tier);
* `certified` = `2ε + ε²`, where `ε` is the full-action Duhamel state bound
  `evolution_bound` from `sm_defect` (`|a|²` is Lipschitz with constant `2|a|`
  on the unit ball).

| cell | `p_sirk` | `p_exact` | `sharp` | `ε` | `certified` |
|---|---|---|---|---|---|
| base (`λ=1`, `width_k=1`) | `0.966749778031` | `0.966736506315` | `1.33e-5` | `0.1805` | `0.3937` |
| null control (same again) | `0.966749778031` | `0.966736506315` | `1.33e-5` | `0.1805` | `0.3937` |
| coupled (`λ=1.1`) | `0.965119884383` | `0.965123360010` | `3.48e-6` | `0.2045` | `0.4508` |
| spread (`width_k=1.5`) | `0.909965726101` | `0.910575997445` | `6.10e-4` | `0.6293` | `1.6547` |

The full-action state bound is now what the record charges, and its soundness is
asserted (each response's certified interval contains the exact difference).
What it cannot do is *certify* the response: at `±0.84` (`λ`) and `±2.05`
(`width_k`) the certified intervals contain `0`, while the sharp-tier intervals
exclude it. The reason is measured, not conjectural: on the base cell the true
state error is `7.85e-3` and `ε` is `1.81e-1` (23×), but the true *probability*
error is `1.33e-5` (600× smaller still). The evolution error is nearly
orthogonal to the seed, so a bound on `‖e‖` alone cannot resolve a `1e-3`-scale
probability. See §6.1.

Paired differences:

The sign convention in the record is `a − b`, so the coupling difference reads
positive because raising `λ` *lowers* the return probability.

| response | exact difference | sharp interval | excludes 0 | certified interval | excludes 0 |
|---|---|---|---|---|---|
| null | `0` | `[-2.65e-5, 2.65e-5]` | no (correct) | `[-0.79, 0.79]` | no |
| `λ` | `+1.6131e-3` | `[1.5964e-3, 1.6299e-3]` | yes | `[-0.84, 0.84]` | no |
| `width_k` | `+5.6161e-2` | `[5.5537e-2, 5.6784e-2]` | yes | `[-2.05, 2.05]` | no |

The momentum-spread response is 35× the coupling response, as expected: the
reference Gaussian width in `k` multiplies `|k|²φ²`, so it feeds the quartic
window directly. The null control contains `0` while both physical intervals
exclude it.

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

## 6. Limitations — read this before quoting a number

1. **The full-action defect does not close — but it is now 33× tighter, and the
   reason it still cannot certify is measured.** `defect_action_envelope` used
   to be a grid-plus-Lipschitz bound whose dominant term was
   `t/8 · ‖DG‖_F · ‖c‖ · e^{ωt}`. On this operator `G = γI − B⁻¹` with
   `γ` up to `200`, every entry of `G` carries the shift while `e^{sG}` stays
   `O(1)`, so `‖DG‖_F ≈ 1.55e4` against a trajectory whose actual maximum is
   `3.61` — the product prices the shift, not the defect. The envelope is now a
   grid-converged sampled maximum along the returned trajectory (no shift
   factor, basis-covariant):

   | cell | envelope before | envelope now | `evolution_bound` before | now |
   |---|---|---|---|---|
   | base | `100.2` | `3.611` | `5.949` | `0.1805` |
   | coupled | `111.2` | `4.090` | `6.537` | `0.2045` |
   | spread | `167.4` | `12.586` | `10.601` | `0.6293` |

   That is the honest ceiling of this route. The measured state error is
   `7.85e-3` (base) and `2.14e-2` (spread), so the tightened bound is within
   ~25× of it; but the probability error is `1.33e-5` and `6.10e-4`, ~600× and
   ~35× smaller, because the evolution error is nearly orthogonal to the seed.
   A bound that knows only `‖e‖` cannot separate those two, so it cannot
   certify a `1e-3` response at any tightness it can actually reach. Responses
   are therefore charged by the measured compressed-flow discrepancy against
   the exact evolution of the declared window (the S43 *sharp* tier), with the
   full-action charge recorded beside it and asserted sound. These remain
   certified *discretized-window* statements plus a measured SIRK-versus-exact
   agreement, not window-independent error bounds.
2. **The outside-window tail is reported separately** and is first order in
   neither `t` nor the rank; it must be shrunk by raising `C`
   (`1.79 → 1.40 → 0.89 → 0.51 → 0.32` for `C = 4, 8, 12, 16, 20`). It does not
   reach zero, and the last two rows cost 11.5 s and 73 s per cell, so the
   window cannot be pushed much further by brute force on one core.
3. **`ForwardProducts` is not worse here** at equal rank (see §5.2). Do not
   quote the plan's "Arnoldi deteriorates" as a rank comparison on bounded
   windows; the observable degradation is growth with `C`.
4. **GMRES iteration counts are exposed and recorded.** `JointSirkResult`
   carries `shift_solve_iterations` (one Arnoldi step count per rational basis
   solve, in the same order as `shift_solve_error_bounds`) and
   `function_solve_iterations` (one per bounded-resolvent solve), with the
   `max_*` / `total_solve_iterations` accessors. The response cells need
   `[12, 13, 14, 15]` basis steps and `[15, 15, 15, 15]` resolvent steps under
   the `64`-step cap at `1e-13`; the cutoff sweep climbs to `[112, 138]` and
   `[136, 137, 139, 141]` at `C = 20` and needs the looser `1e-10`/1024 policy
   described in §5.2. Every count is in the record — no `null` stands in for a
   measurement.
5. **`cadabra2-cli` and a C compiler are Nix-only on this host**; the CDB gate
   needs `PYTHONHOME` from the Cadabra2 closure to avoid
   `ModuleNotFoundError: _struct`.
6. The two pre-existing `.cdb` failures in §3 are unrelated to this suite and
   remain open.

## 7. Honesty summary

* Symbolic (`cadabra`/`.cdb`): all 32 λφ⁴ zero-intent checks reduce to `0`;
  the fixture is cross-checked against the live operator.
* Kernel/dense reference: exact Padé evolution of each declared window.
* Measured: every probability, tail, Ritz value, response and charge above,
  reproducible with the commands in §3.
* Not certified: a `1e-3`-scale probability bound from the full-action defect.
  The bound was tightened 33× (`5.95 → 0.18` state) and is charged and asserted
  sound, but it is limited by the state error (~`1e-2`) while the probability
  error is ~`1e-5`; that gap is measured, not assumed, and is reported rather
  than papered over. The sharp tier carries the response claims.
