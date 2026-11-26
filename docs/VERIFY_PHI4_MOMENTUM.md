# Verifying the continuous-momentum λφ⁴ suite (`P11`–`P13`)

This note records what was built, what was actually measured, how to replay it,
and — importantly — where the certification stops. It accompanies
[`PHI4_ANALYTIC_PLAN.md`](PHI4_ANALYTIC_PLAN.md) and
[`PHI4_NUMERICAL_PLAN.md`](PHI4_NUMERICAL_PLAN.md).

The arena is `L²(ℝ³_k × ℝ_φ)`: momentum `k ∈ ℝ³` and field `φ ∈ ℝ` stay
**continuous**. Nothing here discretizes momentum into sums or a lattice; the
only discrete structures are the Mehler–Hermite occupation window and the Krylov
rank.

**How to read the response intervals.** In the recorded sweep, `exact` denotes the exact
evolution of the same window — since §5.8 a Hermitian-Lanczos seed column of
`exp(-i t H_window)`, before that a dense floating-point Padé `exp` of the window
matrix, whose values it reproduces (§5.8). The tier labelled `certified` uses
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
* repository `HEAD = 9d862ec` when the record was regenerated, with the
  `C = 42/44` rows of §5.12 uncommitted (the fixture's
  `source_revision.git_head` records the same commit)

| file | SHA-256 |
|---|---|
| `docs/phi4_momentum_nested.cdb` | `669510d3c55d334b065b0d94c6f7a4d402cdf29408865856178c3b475fb8d709` |
| `docs/fixtures/phi4_momentum_mehler.json` | `9b1e9c15ccbb53c876a4818f8d2898d8faec5f26c54d0664406cbcc9912d15bb` |
| `docs/fixtures/phi4_momentum_sirk_sweep.json` | `92625892a8d4288e141f6f50716a6340f390f132e1cef29d424299686dae31d2` |

The whole record comes from that one run: `source_revision.git_head` and the
digest above are the run's own. The record predates the twelve-digit reporting
now in the tree: `6f0d9c6` routes `AmplitudeBound::to_json` through
`sm_bridge::reported_residual`, and `ea651ed` does not, so this record's
`responses.cells.*.amplitude_defect` block is still stored at full precision and
the next regeneration rounds those seven measured fields per cell. Two field
classes are machine- rather than model-dependent and move between runs:
`runtime_ms` / `peak_rss_kb`, and the solver diagnostics already at the `1e-16`
floor (`hermiticity_defect`, `orthogonality_defect`), whose *relative* spread is
large while the absolute defect stays at rounding noise.

## 3. Replay

```bash
# symbolic half (needs cadabra2-cli; PYTHONHOME is required on this host)
env PYTHONHOME=<python-in-cadabra-closure> \
  nix shell nixpkgs#cadabra2 --command \
  bash scripts/verify_cdb_modules.sh --strict

# unit tests for the continuous operator
cargo test -p nested_fock_algebra phi4_momentum

# the P12 suite (needs a C compiler, e.g. via `nix shell nixpkgs#gcc`)
cargo test -p fock_sirk --test phi4_momentum_sirk   # 7 tests; ~5 s release

# regenerate the evidence record (~83 min in release as run here, of which
# ~48 min is the four new C = 42/44 cells; the thirteen older cutoffs alone
# took ~36 min on a quiet host). The seventeen cutoff cells cost 21 ms,
# 0.11 s, 0.36 s, 0.95 s, 2.5 s, 4.1 s, 6.7 s, 10.9 s, 17.8 s, 31.7 s, 54.5 s,
# 81.9 s, 122 s, 178 s, 266 s, 391 s and 574 s at m = 2 and 35 ms, 0.17 s,
# 0.55 s, 1.5 s, 4.0 s, 6.7 s, 11.3 s, 18.6 s, 30.8 s, 52.3 s, 83.2 s, 131 s,
# 202 s, 308 s, 470 s, 717 s and 1183 s at m = 4; it peaks at 4.20 GB RSS.
# These are wall clocks on a shared host and the recorded run was loaded (see
# section 5.12); two reruns measure the spread at <= 12 %, so they are upper
# bounds by that much; the debug build is impractical here)
cargo run -p fock_sirk --release --example phi4_momentum_sirk

# the measurements of section 5.5 (serial, then default workers), plus the
# per-operation profile of the Krylov solve
FOCK_SIRK_ACTION_THREADS=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe
cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe
PHI4_PROFILE=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe

# the section 5.11 profile: same probe at C = 26 (~40 s: an 8.5 s solve after
# the Givens least squares — 9.8 s before the NEON inner product of section
# 5.12 — then a 2.5 s seed-column window reference;
# PROBE_COMPONENT_BUDGET must exceed the window's component count, 5.71e6 here)
PROBE_CUTOFF=26 PROBE_COMPONENT_BUDGET=25000000 PHI4_PROFILE=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe

# the section 5.11 C = 28 cell (~1.3 min, the same budget; the same solve took
# 40.5 s before section 5.11)
PROBE_CUTOFF=28 PROBE_COMPONENT_BUDGET=25000000 PHI4_PROFILE=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe

# the section 5.12 profile: the same probe at C = 32 (~1.8 min; the budget must
# exceed that window's components, 2.36e7). The gap between `matvec_ms` (one
# sparse nested action) and the solve's per-step `PROFILE matvec=` is the
# measured price of not having the projected gemv. This is a timing probe: on a
# loaded host every phase clocks ~1.4x high, so run it quiet (the recorded run
# started at load average 2.3 and its gemv lands on the bench's 20.868 ms/step).
PROBE_CUTOFF=32 PROBE_COMPONENT_BUDGET=25000000 PHI4_PROFILE=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe

# agreement check between the coordinate and map solves (section 5.11), and the
# bit-identity pin for the two map operations (previous implementations as the
# oracle, amplitudes compared on to_bits())
cargo test -p fock_sirk --test joint_map_ops
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

Rank-2 tail sweep (`N = 8, h = 3`), `max_k ‖(I − P_C) H V_k‖`. The `C = 16 … 40`
rows extend the plan's `C ≤ 12` range past 20: the tail keeps falling, but the
shift-solve steps the smallest scheduled shift (`γ = 40`) needs grow with the
window, and the runtime grows faster than the label count. The step column is
`shift_solve_iterations` — one Arnoldi count per rational basis solve, in shift
order; the bounded-resolvent counts are in the record's
`function_solve_iterations` (the rank-4 table below reports those).

| `C` | labels | tail | shift-solve steps | cell runtime |
|---|---|---|---|---|
| 4 | 15 | `1.7866` | `[6, 6]` | `21 ms` |
| 8 | 70 | `1.4001` | `[17, 18]` | `0.11 s` |
| 12 | 210 | `0.8946` | `[39, 43]` | `0.37 s` |
| 16 | 495 | `0.5067` | `[70, 82]` | `0.96 s` |
| 20 | 1001 | `0.3189` | `[112, 139]` | `2.5 s` |
| 22 | 1365 | `0.2615` | `[137, 173]` | `4.1 s` |
| 24 | 1820 | `0.2129` | `[163, 212]` | `6.7 s` |
| 26 | 2380 | `0.1703` | `[190, 241]` | `10.9 s` |
| 28 | 3060 | `0.1337` | `[213, 281]` | `17.9 s` |
| 30 | 3876 | `0.1037` | `[240, 312]` | `30.8 s` |
| 32 | 4845 | `0.0802` | `[266, 351]` | `48.9 s` |
| 34 | 5985 | `0.0625` | `[289, 385]` | `76.1 s` |
| 36 | 7315 | `0.0495` | `[315, 420]` | `117 s` |
| 38 | 8855 | `0.0397` | `[340, 459]` | `178 s` |
| 40 | 10626 | `0.0321` | `[367, 498]` | `267 s` |

The `C ≥ 16` cells solve to a `1e-10` relative residual with a 1024-step cap:
at `C = 20` the `1e-13` target stalls at `2.3e-13` and exhausts even 512 steps.
The looser target is six orders below the `1e-3`-scale inside-window error
being measured, and every cell records its policy (`solve_policy`,
`relative_tolerance`, `max_iterations`, and the realised step counts) so the
numbers are never silently mixed. `C ≥ 20` also needs a larger component
budget: the window alone is 1001 labels at `C = 20`, 1820 at `C = 24`, 2380 at
`C = 26` (`5.71e6` components), 3060 at `C = 28` (`9.42e6`), 3876 at `C = 30`
(`1.51e7`), 4845 at `C = 32` (`2.36e7`), 5985 at `C = 34` (`3.59e7`), 7315 at
`C = 36` (`5.37e7`), 8855 at `C = 38` (`7.84e7`), 10626 at `C = 40`
(`1.129e8`), 12650 at `C = 42` and 14950 at `C = 44` (`2.235e8`), and the
dictionary charges one component per (outer label, inner state) pair, so the
default `1e6` limit (and `C = 20`'s `4e6`) is hit during span registration; the
runner now uses `3.0e8`, which is the `C = 44` window with `34 %` headroom. The
component budget was not the only ceiling: the
projected action is a dense `n × n` product, so `ActionLimits::max_work`
(default `1e7`, which the `C = 28` window's `9.36e6` entries nearly fill) binds
at `C = 30`. The runner raises it to `1.2e9` (`4 × 3.0e8`) rather than charging
the same `gemv` less work — the count is a resource policy, and the action
really does touch `n²` entries.

Note the tail is a norm of the *action* outside `P_C`; it need not reach zero as
the window grows, and it is still `0.021` at `C = 44` because `H` moves the seed
to labels the window does not contain. The tail falls by only `0.32 → 0.021`
from `C = 20` to `C = 44` while the cell cost grows `2.5 s → 574 s`. That cost is
now the Arnoldi steps themselves, not the window and no longer the least
squares: of the `C = 28` cell the exact reference is `3.5 s` and the Krylov
solve `16.7 s` (§5.11), and the per-step least-squares stage is below the
profile's `10 ms` print resolution. Within the solve it is now the projected
`gemv` — `71 %` of the `C = 32` solve (§5.12) — that a wider window charges.

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
| 16 | `2.73e-3` | `1.74e-4` | `[81, 83, 83, 84]` |
| 20 | `2.99e-3` | `1.86e-4` | `[136, 138, 139, 141]` |
| 22 | `2.90e-3` | `1.82e-4` | `[171, 172, 174, 176]` |
| 24 | `2.85e-3` | `1.85e-4` | `[209, 212, 214, 216]` |
| 26 | `2.84e-3` | `1.84e-4` | `[240, 242, 246, 247]` |
| 28 | `2.86e-3` | `1.83e-4` | `[278, 286, 284, 286]` |
| 30 | `2.88e-3` | `1.84e-4` | `[311, 319, 320, 322]` |
| 32 | `2.88e-3` | `1.84e-4` | `[349, 359, 361, 358]` |
| 34 | `2.88e-3` | `1.84e-4` | `[384, 396, 399, 400]` |
| 36 | `2.88e-3` | `1.84e-4` | `[419, 432, 434, 435]` |
| 38 | `2.88e-3` | `1.84e-4` | `[459, 472, 476, 475]` |
| 40 | `2.88e-3` | `1.84e-4` | `[500, 514, 518, 519]` |
| 42 | `2.88e-3` | `1.84e-4` | `[539, 553, 559, 558]` |
| 44 | `2.88e-3` | `1.84e-4` | `[578, 593, 599, 601]` |

Two readings, both from the same table. The plan's Hashimoto §3.1 claim
reproduces as **growth with the window** (≈ 10³× from `C = 4` to `C = 12`,
`1.2e4×` to `C = 20`, `1.1e4×` at `C = 26` where it levels off), not as a
rank-for-rank deficit: on these bounded windows the shifted polynomial flow is
the *more* accurate of the two. And the *inside-window* error of the primary
solver is flat — `8.0e-4, 3.07e-3, 2.43e-3, 2.73e-3, 2.99e-3, 2.90e-3,
2.85e-3, 2.84e-3, 2.86e-3, 2.88e-3, 2.88e-3, 2.88e-3, 2.88e-3` — i.e. once the
window covers the seed's support (from `C = 8` on) the remaining error is set by
the rank, not by `C`, while the tail keeps shrinking. The `C = 20 … 44` entry is
a shallow plateau with no monotone trend: `2.99e-3, 2.90e-3, 2.85e-3, 2.84e-3,
2.86e-3, 2.88e-3, 2.88e-3, 2.884e-3, 2.882e-3, 2.880e-3, 2.879e-3, 2.8788e-3,
2.8793e-3` differs in the third digit (`1.04×` end to end), which is the
`solve_policy` tolerance's neighbourhood, so those thirteen cells are read as
equal, not as a decaying sequence. **The plateau does not break.** `C = 42` and `C = 44` put the same
question to the two widest windows: `2.8788e-3` and `2.8793e-3`, against
`2.8787e-3` at `C = 40` — the minimum over the whole sweep sits at `C = 40` and
the last two cells are `0.006 %` and `0.02 %` above it, an upturn of `5.9e-7`
absolute, two orders below the `1.04×` spread of the plateau itself. So there is
no cutoff in `C = 8 … 44` at which the rank-4 inside-window error leaves its
plateau: it is rank-limited throughout, and widening the window cannot recover
it. Growing `C` past 12 therefore buys tail, not accuracy, at a rising cost
(0.55 s → 1.5 s → 4.0 s → 6.7 s → 11.3 s → 18.8 s → 31.0 s → 52.5 s → 83.9 s →
131 s → 203 s → 311 s → 535 s → 717 s → 1183 s per `m = 4` cell).
That is exactly why it stays a labeled diagnostic. At `m = 2` the contrast is
not yet visible (`1.9e-3, 1.3e-3, 1.3e-3`), so the m = 2 sweep is used for the
plan-literal tail measurement and the contrast is recorded at `m = 4`.

### 5.3 Paired return responses, `C = 8, m = 4, t = 0.05, N = 250, h = 50`

`p_exact` is the exact evolution of the *declared window* (§5.8). Each cell
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
same `0.9800015102784486` that record stored for `C = 20`, which is the
end-to-end check that the split really is order-preserving and not merely
accurate. (The checked-in record at `ea651ed` stores `0.9800015102784502` for
that cell — the same value to 13 digits, the remainder being Krylov accumulation
order rather than the map operations, which are bit-identical by construction.)

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
`172.2 → 85.1 s` (2.02×) and `367.1 → 184.9 s` (1.99×). (Those cell figures are
as recorded then, beside the dense window reference. §5.8 subsequently replaced
that reference, which lowers every cell again — the map-operation comparison is
the one measured here and is unaffected by it.)

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

### 5.7 C = 26: the same two operations, but the cell is now half reference

The question for the extension was whether the `C = 20` profile survives the
move to `C = 26`. It does, and the arithmetic share edges up. `PROBE_CUTOFF=26
PHI4_PROFILE=1` on 2380 labels, `m = 2`, four shift solves, 918 steps total:

| phase | 190-step | 242-step | 242-step | 244-step | total | share of the solve |
|---|---|---|---|---|---|---|
| action matvec | `1.36 s` | `1.76 s` | `1.75 s` | `1.76 s` | `6.63 s` | **3.4 %** |
| `joint_axpy` in MGS (contains the dot) | `27.44 s` | `47.74 s` | `47.57 s` | `47.87 s` | `170.62 s` | **87.1 %** |
| — `joint_inner_product` inside it | `8.98 s` | `15.32 s` | `15.26 s` | `15.32 s` | `54.88 s` | `28.0 %` |
| per-step basis SVD | `1.48 s` | `3.74 s` | `3.70 s` | `3.82 s` | `12.74 s` | `6.5 %` |

Solve wall clock `196.0 s` (`918` Arnoldi steps, probability
`9.79979792832502916e-1`, which is the record's `C = 26` value). A second run,
with compilation on the same machine, reproduced the shares within `3 %` and the
wall clock at `202.1 s`. (This profile is the *before* picture of §5.9; the same
probe measures `104.0 s` after §5.9 and `22.5 s` after §5.10, for the same steps
and the same probability.)

The per-call cost grew roughly with the label count, not faster: `axpy`
`0.76–0.81 ms` against `0.26–0.29 ms` at `C = 20`, `dot` `0.25–0.26 ms` against
`0.08–0.09 ms`, for `2.38×` the labels. The in-solve `dot` is `≈ 3×` the
standalone `dot_ms = 0.0864 ms` the probe prints, because the Krylov basis
vectors are dense in the inner sector while the probe's pair is not — quote the
`PROFILE` shares for cost, the probe numbers for the operations in isolation.

**What changed is not the solver but the cell.** The recorded `C = 26` cell is
`406.4 s` then (`104.8 s` after §5.8 and §5.9, `26.0 s` after §5.10), and the
probe then
times the other half of it — the exact dense window
reference (`WindowOperator::build` plus the Padé `exp`) that the sweep evaluates
beside every solve:

| cell at `C = 26` | time | share of the `406.4 s` cell |
|---|---|---|
| Krylov solve (4 shifts, 918 steps) | `196.0 s` | `48 %` |
| exact reference: dense build / Padé `exp` | `1.73 s` / `174.98 s` | `43 %` |
| not individually timed (window, seed, residual bookkeeping, run variance) | `≈ 34 s` | `8 %` |

The reference is a dense `2380 × 2380` Padé exponential, so it grows like `n³`:
the same probe at `C = 20` (`1001` labels) measures `0.73 s` to build and
`11.83 s` to exponentiate, `12.6 s` against `177 s` here — a `14.8×` rise for
`2.38×` the labels, exponent `3.1`. Past `C = 24` the experiment is no longer
solver-bound: **the Krylov solve is half the cell and the exact reference tier
is the other half**, and it is the reference, not GMRES, that sets what `C = 28`
would cost. Lengthening the sweep therefore needs either a cheaper reference (a
Krylov or Lanczos `exp` action on the seed column instead of a full dense
exponential) or a decision to record the sharp tier on a subsample of cutoffs.
§5.8 takes the first route.

The two halves are measured by different programs — the cell figure is the
record's, the split is the probe's — and they sum to `373 s` against `406 s`, so
the third row is a residual, not an attribution.

The parallel map operations still lose at this size, by more than at `C = 20`:

| operation (`C = 26`, 2380 labels) | direct | 1 worker (chunked) | 2 workers | 4 workers | 8 workers |
|---|---|---|---|---|---|
| `joint_inner_product` | `0.086 ms` | `0.080 ms` | `0.46 ms` | `0.29 ms` | `0.30 ms` |
| `joint_axpy` | `0.96 ms` | `1.31 ms` | `2.59 ms` | `3.13 ms` | `3.12 ms` |

and the chunked dot's deviation from the serial scalar is `3.5e-15` relative
here against `1.1e-15` at `C = 20` — the association error grows with the entry
count, so the bit-identity objection gets worse, not better, as the window
widens.

### 5.8 C = 28, and the reference that stopped being cubic

The first question about the next cutoff was what it would cost *under the
reference as it then stood*. `PROBE_CUTOFF=28 PROBE_COMPONENT_BUDGET=12000000`
on 3060 labels (`C = 28` needs `9,363,600` components, so the sweep's `8e6`
ceiling is not enough there), `m = 2`, four shift solves, 1060 steps:

| phase of the `C = 28` cell (old reference) | time | share |
|---|---|---|
| Krylov solve (4 shifts, 1060 steps) | `356.0 s` | `49 %` |
| reference: window build / dense Padé `exp` (`3060 × 3060`) | `2.26 s` / `364.6 s` | `51 %` |
| cell (solve + reference) | `723 s` (`12.1 min`) | — |

Peak RSS `3.2 GB`, almost all of it the dense window matrix and the `exp`
temporaries. The cell essentially doubles from `C = 26` (`406 s` → `723 s`)
while the *solve* only grows `1.8×` (`196 s` → `356 s`): under the dense
reference the marginal cost of the next cutoff was more than half reference,
which is the measurement §5.7 predicted and the reason it was replaced.

**The reference is now a seed column.** `fock_sirk/examples/common/window_reference.rs`
replaced the dense `n × n` Padé exponential with the column the sweep actually
uses: `exp(-i t H_window) e_j`. The window matrix is `P H P` in the orthonormal
Fock basis, hence Hermitian, so Hermitian Lanczos with full reorthogonalisation
builds a tridiagonal `T_m` and the column is `V_m exp(-i t T_m) e_1` —
`O(m n² + m² n)` for Krylov dimension `m`, against `O(n³)` for the dense
factorisation. The dimension is chosen by measurement: the column is formed at
`m = 64, 128, 256, 512` and accepted when two successive checkpoints agree to
`1e-13` on a column of norm 1; an invariant Krylov subspace returns exactly, and
non-convergence panics rather than silently degrading the reference tier.

The same `C = 28` cell with the seed column:

| phase | old dense reference | seed-column reference |
|---|---|---|
| window build (one action per label) | `2.26 s` | `2.28 s` |
| exponential action | `364.6 s` | `1.20 s` |
| reference total | `366.9 s` | `3.47 s` |
| return probability of the window | `9.66679088641535622e-1` | `9.66679088641579365e-1` |

The two references differ by `4.4e-15` absolute (`4.5e-15` relative) — four
orders below the `1e-3`-scale discrepancy the reference is used to certify — and
the reference is now `106×` cheaper: the `C = 28` cell is `360 s` instead of
`723 s`, and what remains is the solve.

The whole evidence record was then regenerated with the seed column and compared
field by field against the previous record. Every `return_probability_window_exact`
entry moved by at most `7.1e-14` absolute (at `C = 16`; most are `≤ 1e-14`), and
nothing else changed except the timings and the source revision. The record now
carries `reference_tier` saying which implementation produced the column. The
cells came down with it — the `m = 2` cells are `21 ms, 0.12 s, 0.49 s, 2.5 s,
14.0 s, 29.9 s, 58.8 s, 104.8 s` (the `C = 26` cell was `406.4 s`) and the `m = 4`
cells `34 ms … 187.5 s` (was `46 ms … 534.3 s`), peak RSS `0.69 GB` (was
`2.1 GB`), and the full run `18.5 min` (was `30.4 min`). §5.9 then halves those
cells again and adds `C = 28` to the sweep itself, so the `C = 28` figures in the
tables above are the *probe* measurement of a cell the record now contains as a
row: `188.7 s` at `m = 2`, `334.7 s` at `m = 4`. §5.10 then removes the map walk
from the inner solve and takes those recorded cells to `26.0 s` and `44.8 s` at
`m = 2`, `42.8 s` and `75.5 s` at `m = 4`, leaving every probability in this
section byte-identical.

The dense implementation is still in `fock_sirk/tests/phi4_momentum_sirk.rs` on
the `C ≤ 12` windows, where it costs milliseconds, as an independent oracle for
the seed-column tier.

### 5.9 The solve was the map walk, not the arithmetic

§5.7 left a lopsided cell: the reference was `3.5 s` and the solve was
`104–196 s`, with `87 %` of the solve inside `joint_axpy`. That number is worth
reading twice, because `axpy` is one complex multiply-add per label. The cost
was not the arithmetic: `joint_axpy` found each label with `get_mut`, i.e. a
fresh `BTreeMap` descent — about a dozen comparisons into a key that is itself a
nested dictionary — *per entry of the source vector*. Synchronising the two
vectors' zeros at `C = 26` cost `196 s`.

Both maps are ordered by that same comparison, so the descent is redundant:
`joint_axpy` now walks `a` in order with a forward cursor over `b`, matching as it
goes (one comparison per visited entry), collecting fresh labels and recording
exact cancellations to apply after the walk (the map cannot be restructured
while it is borrowed). Each label's value still gets exactly the arithmetic it
got before, and no label's value depends on another, so the operation is
bit-identical — `tests/joint_map_ops.rs`, which keeps the previous
implementations as the oracle and compares every amplitude on `to_bits()`, still
passes unchanged.

| labels | per-entry search | ordered walk |
|---|---|---|
| 210 | `0.075 ms` | `0.005 ms` (**15.9×**) |
| 715 | `0.313 ms` | `0.015 ms` (**20.8×**) |
| 1365 | `0.675 ms` | `0.032 ms` (**21.4×**, `23 ns`/entry) |

End to end, with the same step counts and bit-identical probabilities:

| cell | solve before | solve after | probability (both) |
|---|---|---|---|
| `C = 20`, `m = 2` | `24.1 s` | `12.9 s` | `9.80001510278448618e-1` |
| `C = 26`, `m = 2` | `196.0 s` | `104.0 s` | `9.79979792832502916e-1` |
| `C = 28`, `m = 2` (recorded row) | — | `188.7 s` cell | `9.79988805406555197e-1` |

The strongest check is not the spot comparison: the record regenerated after the
rewrite has **exactly** the same `return_probability_sirk` and the same Arnoldi
step counts as the record regenerated before it, in all eight cells that both
contain — so §5.8's numbers stay quotable and only timings move.

The `C = 26` split after the change (`PROBE_CUTOFF=26 PHI4_PROFILE=1`, 918 steps,
four shift solves):

| phase | before §5.9 | after §5.9 | share of the `104.0 s` solve |
|---|---|---|---|
| action matvec | `6.63 s` | `6.63 s` | `6.4 %` |
| `joint_axpy` inside MGS | `170.62 s` | `79.37 s` | `76.3 %` |
| — the `joint_inner_product` inside it | `54.88 s` | `54.60 s` | `52.5 %` |
| per-step basis SVD | `12.74 s` | `12.73 s` | `12.2 %` |

So the rewrite moved the bottleneck rather than removing it. What is left is one
deep label comparison per entry in the inner product (`0.25 ms` per call at 2380
labels, `≈ 40 ns`/entry) plus a real complex multiply-add — and that is already
minimal for this representation. The next factor of 5–10 is a *representation*
change, not an algorithmic one: inside a declared window the label set is fixed,
so `(labels, Vec<C>)` makes every dot and `axpy` a `2–5 ns`/entry memory sweep,
and the action is already a cached dense matrix on the same labels
(`FiniteWindowAction::matrix`). §5.10 does exactly that, with the `to_bits()` pin
this section asked for; what §5.9 reports is `1.88×` measured, not a projection
of what it might have been.

### 5.10 The solve in coordinates of the window

§5.9 ended on a measurement and a prediction: the surviving cost was the
*representation* — a deep label comparison per entry — and the fix was
coordinates over the fixed label set, with its own `to_bits()` pin. This is that
change, and the prediction held: **4.0–4.4×** on the recorded cells, with every
recorded number unchanged to the last bit. (That bit-identity held until §5.11
replaced the least-squares stage; the coordinate *representation* of this
section is unchanged, only what it solves with.)

Inside a declared window the labels are known before the first Arnoldi step, and
`FiniteWindowAction` already cached exactly what a coordinate solver needs: the
sorted `labels`, their `indices`, and the `gemv`-ready `matrix`. GMRES now runs
on `Vec<C>` over those labels (`JointSirkOptions::dense_window_solve`, default
on). The nested maps remain the representation of the *result*: the right-hand
side is converted once on entry, the accepted solution once on exit, and in
between — projected action, modified Gram–Schmidt, norm, least squares — the
solver touches flat memory. The acceptance test still evaluates the **full**
action on the nested solution, so tails, residuals and the `shift_solve_error_bounds`
are computed exactly as before, tails included.

**Why this is identical and not merely close.** Each coordinate operation is the
map operation with the map removed, not a rearrangement of it:

| operation | map path | coordinate path |
|---|---|---|
| projected action | `gemv` on the cached `matrix` | the same `gemv` on the same matrix (`image_coords`) |
| inner product | products in ascending label order over the intersection | the same products **in index order**, skipping exactly the labels the map does not store (`coords_inner_product`) |
| axpy | `slot + scale * value` per visited label, exact zeros dropped | the same per-label arithmetic, zeros skipped (`coords_axpy`) |
| norm | `hypot(re).hypot(im)` folded in label order | the same fold over the stored entries (`coords_norm`) |

No label's value depends on another, and no summation order changes, so the only
difference is that the intermediate zeros stay in an array instead of being a
missing map entry. Three checks said so rather than asserting it:

* `tests/phi4_momentum_sirk.rs` runs the same `C = 12`, `m = 4` window both ways
  — the map path is kept behind `dense_window_solve: false` for exactly this —
  and compares the forward sequence, the orthonormal basis, the Gram matrix,
  `H_proj`, the bounded-resolvent compression, the function generator, the
  unit-norm scales, the action tails, the solve error bounds, the Arnoldi step
  counts and the return probabilities at `t`, `2t`, `5t` (`0.6 s`). While the
  SVD was the least-squares stage this was `to_bits()` equality
  (`dense_window_solve_is_bit_identical_to_the_map_walk`); since §5.11 it is
  `window_solve_matches_the_map_walk_at_solver_precision`, a `1e-12` relative
  agreement check, because the two paths now round differently in that stage.
* The record regenerated on the new path differed from the previous one in **33
  leaves out of the whole file**: `runtime_ms` on the 31 measured cells,
  `peak_rss_kb`, and `git_head`. Every probability, tail, error bound, step
  count and `reference_tier` field was byte-identical (that was the state before
  §5.11).
* The suites that share this solver pass unchanged on the new path: `sm_validation`
  13/13 (`sm_full_hashimoto`), `qym_magnetic_packet` 2/2, `joint_map_ops` 2/2
  (the map-operation oracle of §5.9).

Measured on the recorded cells (whole-cell wall clock, so this includes the seed
column reference and the residual bookkeeping):

| cell | map walk (§5.9) | coordinates | speedup | probability (both) |
|---|---|---|---|---|
| `C = 26`, `m = 2` | `104.8 s` | `26.0 s` | **4.0×** | `9.79979792832502916e-1` |
| `C = 28`, `m = 2` | `188.7 s` | `44.8 s` | **4.2×** | `9.79988805406555197e-1` |
| `C = 26`, `m = 4` | `187.5 s` | `42.8 s` | **4.4×** | — (rank-4 error unchanged) |
| `C = 28`, `m = 4` | `334.7 s` | `75.5 s` | **4.4×** | — |
| whole record | `18.7 min` | `4.5 min` | **4.1×** | all 18 rows identical then |

The in-loop operations fell by more than the cells because the reference and the
bookkeeping do not move: `dot` is `0.007 ms` per call at 2380 labels against
`0.25 ms`, and `axpy` `0.010 ms`. Peak RSS also fell, `1.01 GB → 0.29 GB`: the
Krylov basis is `n` complex numbers per vector instead of a `BTreeMap` of nested
dictionaries.

The `C = 26` split after the change (`PROBE_CUTOFF=26 PHI4_PROFILE=1`, 918 steps,
four shift solves, `22.5 s` solve):

| phase | map walk (§5.9) | coordinates | share of the `22.5 s` solve |
|---|---|---|---|
| action `gemv` | `6.63 s` | `4.66 s` | `20.7 %` |
| MGS `axpy` | `79.37 s` | `2.23 s` | `9.9 %` |
| — the inner product inside it | `54.60 s` | `1.65 s` | `7.3 %` |
| per-step basis SVD | `12.73 s` | `12.66 s` | **`56.3 %`** |

The bottleneck moved again, and this time it is an algorithmic one rather than a
representational one. Every Arnoldi step runs a dense SVD of the `(k+2)×(k+1)`
Hessenberg to estimate the residual, so a solve with `k` steps pays `Σ k³` — at
`C = 28` the four solves' SVDs total `23.4 s` of the `40.5 s` solve. Standard
GMRES updates the same least-squares problem with Givens rotations in `O(k)` per
step; that changes the floating-point path of every returned coefficient, so it
is *not* bit-identical and needs its own record — which is §5.11. This section
reports what was measured before that: `4.0–4.4×`, bit-identical, with the SVD
then the largest single item.

### 5.11 The least squares was the solve — Givens, and two more cutoffs

§5.10 ended on an algorithmic bottleneck: with the maps gone from the inner loop,
a dense SVD of the `(k+2)×(k+1)` Hessenberg was `56 %` of the `C = 26` solve and
`58 %` of the `C = 28` one, and a `k`-step solve paid `Σ k³` for it. This section
replaces it, and then spends what it saved on a wider window.

**The change.** `gmres_shift_window` now triangularises the Hessenberg
incrementally, one Givens rotation per Arnoldi step: each new column is rotated
by the accumulated rotations, one new rotation zeroes its subdiagonal entry, and
the same rotations are applied to the right-hand side `βe₁`. The residual
estimate is the last entry of that rotated vector, and on acceptance the
coefficients come from back substitution on the triangular factor. That is the
standard GMRES construction, `O(k)` per step and `O(k²)` per solve instead of
`Σ k³`. The rotation keeps the phase of the diagonal entry
(`givens_rotation` returns `r = top/|top| · ρ`), so the factor is a complex
unitary triangularisation, not a modulus operation. Only the coordinate path
changed: `gmres_shift`, the map walk, still runs its SVD and stays the reference
the agreement test compares against.

**What it bought.** The least-squares stage went from the largest item to below
the profile's print resolution:

| `C = 26` phase | SVD (§5.10) | Givens | share of the `9.8 s` solve |
|---|---|---|---|
| action `gemv` | `4.66 s` | `4.65 s` | `48 %` |
| MGS `axpy` | `2.23 s` | `2.21 s` | `23 %` |
| — the inner product inside it | `1.65 s` | `1.63 s` | `17 %` |
| per-step least squares | `12.66 s` | `< 0.02 s` | **`< 0.2 %`** |

The matvec is unchanged to the second decimal, which is the point: the saving is
entirely the stage that was removed. Whole-solve wall clock `22.5 s → 9.77 s` at
`C = 26` and `40.5 s → 16.72 s` at `C = 28` (**2.3×** and **2.4×**). Cell times
move less where the cell is not mostly solve: the seven `C ≥ 12` cells fell
`2.30×` at `m = 2` (`89.4 s → 38.8 s`) and `2.05×` at `m = 4` (`168.6 s →
82.1 s`); the `C ≤ 8` cells are dominated by the seed-column reference and move
by seconds at most.

**What it cost: the bit-identity of §5.10.** A different rounding path for every
returned coefficient means the coordinate and map solves no longer agree on
`to_bits()`. The pin was replaced by
`window_solve_matches_the_map_walk_at_solver_precision`, which requires identical
Krylov rank and component counts, Arnoldi counts within one step, and every
amplitude, Gram/`H_proj`/generator/resolvent entry, tail, bound and probability
within `1e-12` relative (probabilities at `1e-13`). The record was regenerated
and compared leaf by leaf with the previous one:

* `188` leaves change once `runtime_ms`, `peak_rss_kb` and `git_head` are
  excluded — the two appended rows, some step counts, and floats. No relative
  change exceeds `8.9e-10`, and that worst case is a `sharp_charge` — a
  difference of two return probabilities that are both near `0.96`, so a `1e-16`
  absolute move divides by `3.5e-6`. The return probabilities themselves move by
  at most `3.3e-15` relative, and in the rank-2 sweep every
  `return_probability_window_exact` is byte-identical while
  `window_tail_norm_max` and `inside_window_error` agree to `≤ 6.2e-13`
  relative.
* No Arnoldi or bounded-resolvent step count moves by more than one in any of the
  nine overlapping cells. The acceptance test is still the **full-action**
  residual, so a step either meets its cell's recorded `solve_policy` or the solve
  does not return; the rank-4 error at `C = 28` is `2.860e-3` before and after.
* The shared suites pass on the new path: `sm_full_hashimoto` 13/13,
  `qym_magnetic_packet` 2/2, `joint_map_ops` 2/2, and the φ⁴ suite 7/7 including
  the agreement test.

**Two more cutoffs.** With the SVD gone, a `C = 28` cell costs `20.2 s` at
`m = 2` instead of `44.8 s`, so the sweep now runs `C = 30` and `C = 32`
(`SWEEP_CUTOFFS`, `SWEEP_COMPONENT_BUDGET = 2.5e7`):

| `C` | labels | tail | Arnoldi steps (`m = 2`) | `m = 2` cell | `m = 4` cell |
|---|---|---|---|---|---|
| 28 | 3060 | `0.1337` | `[213, 281]` | `20.2 s` | `34.9 s` |
| 30 | 3876 | `0.1037` | `[240, 312]` | `34.6 s` | `58.7 s` |
| 32 | 4845 | `0.0802` | `[266, 350]` | `54.7 s` | `95.6 s` |

The tail keeps shrinking (`0.1337 → 0.1037 → 0.0802`, both steps `0.77×`), the
rank-4 inside-window error stays flat (`2.86e-3, 2.88e-3, 2.88e-3`) and the
window-exact probability settles into a `8.0e-7` band (`9.666790886e-1,
9.666782885e-1, 9.666785798e-1`, not monotone in the last two), so the two new
rows say what the plan says they should say:
more window buys tail, not accuracy. They also raise the ceiling the component
budget sets — `4845` labels are `2.36e7` components — and they are where
`ActionLimits::max_work` first bound, which §5.2 records.

Cost of the whole thing: the record is `11` cutoffs × `2` ranks in `6.3 min` and
`0.60 GB` peak RSS, against `9` cutoffs in `4.5 min` and `0.29 GB` before. That
is `+62 %` window labels (`12352 → 20073` summed over the cutoffs) for `+39 %`
wall clock, because every individual cell is now cheaper than it was.

**One table correction.** The rank-4 `C = 28` step row of §5.2 used to read
`[156, 190, 220, 276]`. Those are that cell's *shift-solve* counts; the column is
defined as bounded-resolvent counts, which were `[278, 286, 284, 286]` and still
are. The row now shows the resolvent counts, and the shift counts are in the
record under `shift_solve_iterations`.

### 5.12 The projected `gemv` is now the solve, and six more cutoffs

§5.11 left the least-squares stage below print resolution, so the next question
is what a `C = 32` solve actually spends its time on. `PHI4_PROFILE=1` at
`C = 32` (`PROBE_CUTOFF=32`, budget `2.5e7`), four shift solves, `1327` Arnoldi
steps in total, **re-run on the adopted NEON kernel on a quiet host** (load
average `2.3` on 12 cores when it started; its per-step `gemv` lands on the
standalone bench's `20.868 ms/step`):

| phase | 266 | 351 | 351 | 359 | total | share of the `39.4 s` solve |
|---|---|---|---|---|---|---|
| action `gemv` | `5.61 s` | `7.38 s` | `7.35 s` | `7.50 s` | `27.84 s` | **`70.7 %`** |
| MGS `axpy` | `0.59 s` | `1.03 s` | `1.03 s` | `1.08 s` | `3.73 s` | `9.5 %` |
| — the inner product inside it | `0.22 s` | `0.38 s` | `0.38 s` | `0.40 s` | `1.38 s` | `3.5 %` |
| Givens least squares | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.03 %` |

**Every row here is a clock from this run.** The previous version of this table
was taken while the host carried a load average of `11` on 12 cores (other
repositories' proof jobs): its solve took `55.268 s`, and only its `gemv` row
could be corrected, by substituting the standalone bench's `20.868 ms/step` on
the same `1327` steps — a *substitution*, not a measurement. The run below is
the measurement: `39.383 s` for the solve and `20.98 ms/step` for the `gemv`,
which the substitution predicted to `0.5 %`. What the old normalisation could
not correct was the load on the phases it did not touch, and those clocked
`1.4×` high too (`5.26 s → 3.73 s` Gram–Schmidt, `1.92 s → 1.38 s` inner
product, `11.49 s → 7.81 s` everything else) — hence the direct solve is
`39.4 s` rather than the `44.45 s` the normalised table implied. Nothing
structural moved either way: the Arnoldi counts (`266/351/351/359`), the solve
probability (`9.79994613340724019e-1`) and the `1327`-step total are identical
in the loaded and the quiet run, digit for digit.

Read the table as nested, not additive: the inner-product row sits inside the
MGS block and the block's clock contains it (§5.12 note below). Against the
`39.38 s` solve the disjoint phases account for `31.57 s`; the remaining
`7.81 s` (`19.8 %`) is everything outside them — the `coords`/`from_coords`
round-trip at the two ends of each solve (`1.5049 ms` per clone here), each
step's image construction, and the norm of every new basis vector. The `gemv` is
`20.98 ms` per step by construction: `H_proj` is `4845²` complex entries,
`375.6 MB`, streamed once per Arnoldi step. That is `17.9 GB/s`, so it is
bandwidth-bound, not arithmetic-bound — the flop count (`~9.4e7` per step) would
finish an order of magnitude sooner.

**What removing it would cost, measured.** There is exactly one way to keep the
window solve and not multiply by `n²` per step, and it is to go back to the
sparse nested action: the same probe times one `d_gamma` on the same dense
window state at `169.518 ms`, **`8.08×`** the `gemv`'s `20.98 ms`.
Substituted into the same `1327` steps that is `~225 s` of action instead of
`27.8 s`, i.e. a `~236 s` solve and a `~246 s` cell instead of the probe's own
`39.4 s` and `49.1 s`. That is a floor, not an estimate in the other direction:
the map route also pays a reserialisation per step (`map_clone_ms = 1.5049`
here, `42 %` above the earlier quiet run's `1.06`) that the coordinate loop does
not. So the
`gemv` is not a tax on the solve — it is the *cheap* action, already `8×` below
the sparse one, and it only reaches `71 %` because §5.11 made everything else
free. Removing the `gemv` leaves an `11.5 s` solve; removing every *other* term
leaves `27.8 s`. Either way the `gemv` is where a wider window pays.

One accounting caveat on the `PROFILE` line, because it is easy to misread the
split: the `axpy=` figure is the **whole Gram–Schmidt block**, not the axpys
alone — its clock starts before the two passes and stops after them, so it
contains the `dot=` figure. The disjoint split of the `39.38 s` solve is
therefore `70.7 %` `gemv`, `3.5 %` inner products, `6.0 %` real axpys and
`19.8 %` everything else (norm, image construction, the Givens stage).

**This is the adopted-kernel run, and it moves the shares, not the structure.**
The earlier table here was taken on `56374c7`, before the NEON inner product; it
read `61.9 %` `gemv`, `21.0 %` Gram–Schmidt (which contains the inner products),
`15.6 %` inner products and `17.0 %` other on a `44.9 s` solve. Both quiet runs
agree on the three phases the kernel does not touch — the `gemv` (`27.79 s`
there, `27.84 s` here), the real axpys (`2.44 s` against `2.35 s`) and
everything else (`7.63 s` against `7.81 s`) — so the `5.5 s` between the two
solves **is** the inner product: `6.99 s → 1.38 s` (`5.07×`, on `1327` steps
rather than `1328`), measured inside the real solve rather than in the
microbenchmark. The `gemv`'s *share* rises `61.9 % → 70.7 %` only because its
absolute cost did not change while the solve around it shrank; that is what
"the `gemv` is the solve" looks like as a number. The inner-product phase is
now `3.5 %` of the solve. §5.2's record agrees independently: the adopted
kernel took the `C = 32` rank-2 cell from `55.0 s` to `48.9 s` (`−11.1 %`).

Three other removals were considered and are reported as *not* done. Storing
`H_proj` in `f32` halves the stream to `188 MB` (`~10.5 ms` per step, a `~31 s`
solve), but `f32` rounding is `~1e-7` relative against a `1e-10` solve target,
so the recorded residual would stop describing the arithmetic. A device-resident
`gemv` is not measured: the loop's `axpy`s, inner products and vectors would all
have to move with the matrix, and `fock_sirk`'s Krylov loop is CPU `nalgebra`
today, so no claim is made that it would win. And there is no sparsity to
exploit: `H_proj = W† H_raw W` is dense in a `4845`-dimensional window basis by
construction; dropping the projection *is* the map walk, i.e. the `8.08×` above.

**Six more cutoffs.** With the per-step cost known, the sweep now runs `C = 34`,
`36`, `38`, `40`, `42` and `44` (`SWEEP_CUTOFFS`, `SWEEP_COMPONENT_BUDGET =
3.0e8`):

| `C` | labels | tail | shift-solve steps (`m = 2`) | `m = 2` cell | `m = 4` cell |
|---|---|---|---|---|---|
| 32 | 4845 | `0.0802` | `[266, 351]` | `54.5 s` | `83.2 s` |
| 34 | 5985 | `0.0625` | `[289, 385]` | `81.9 s` | `131 s` |
| 36 | 7315 | `0.0495` | `[315, 420]` | `122 s` | `202 s` |
| 38 | 8855 | `0.0397` | `[340, 459]` | `178 s` | `308 s` |
| 40 | 10626 | `0.0321` | `[367, 498]` | `266 s` | `470 s` |
| 42 | 12650 | `0.0259` | `[393, 535]` | `391 s` | `717 s` |
| 44 | 14950 | `0.0208` | `[420, 575]` | `574 s` | `1183 s` |

The tail keeps shrinking (`0.0802 → 0.0625 → 0.0495 → 0.0397 → 0.0321 →
0.0259 → 0.0208`, ratios `0.78×`, `0.79×`, `0.80×`, `0.81×`, `0.81×`, `0.80×`),
the rank-4 inside-window error does **not** leave its plateau (`2.884e-3,
2.882e-3, 2.880e-3, 2.879e-3, 2.8788e-3, 2.8793e-3` — minimum at `C = 40`, then
`0.006 %` and `0.02 %` above it; §5.2) and the window-exact probability stays
inside the same `8.0e-7` band (`9.66678739861e-1, 9.66678514261e-1,
9.66678568574e-1, 9.66678645686e-1`, extended by `9.66678581179e-1` at `C = 42`
and `9.66678573227e-1` at `C = 44`), so the rows again buy tail, not accuracy.
The component budget had to rise from `2.5e7` to `1.3e8` for `C = 40` (`1.129e8`
components, `C = 38` is `7.84e7`) and then to `3.0e8` for `C = 44`, whose window
is `14950` labels, i.e. `2.235e8` before the span images;
`ActionLimits::max_work` is `4 ×` the budget for the same reason as before. The
`C = 44` cell also sets the record's memory: `4.20 GB` peak RSS, up from
`2.29 GB` at `C = 40` and `1.19 GB` at `C = 36`.

**These cell times are wall clock on a loaded host, and the spread is now
measured rather than guessed.** The run that produced this record shared the
12-core host with other repositories' jobs and sampled a load average of `9.6`
at the end (`30.7` at peak), so the `s` columns moved by up to `±12 %` against
the previous quiet run in both directions (`C = 32`'s `48.9 s → 54.5 s`,
`C = 40`'s `535 s → 470 s`). Two full reruns were then made to replace that
guessed band with a measured one. This host did not offer the ~`80 min` window
a sweep needs: the first rerun completed in `80.1 min` (`4535 s` of CPU) with
the load unsampled, the second in `1:23:49` (`4755 s` of CPU) with the load
sampled every `30 s` — median `6.0`, minimum `1.7`, maximum `10.9`, `139` of
its `168` samples above `4.0`. So the reruns are measurements of the same
contended host, not of a quiet one, and they say how far this record can be
from a quiet figure: `19` of the `34` cells reproduce within `3 %` and none is
further than `12 %` away. The pattern is the load, not the code — the first
rerun started while the host was still busy and its small `m = 2` windows
(`C ≤ 20`) came out `5–10 %` slower, while the three `m = 2` cells at `C = 32`,
`34`, `36` and the `C = 44` `m = 4` cell came out `4–12 %` **faster** in both
reruns (`C = 44` at `m = 4`: `1183 s` recorded, `1036 s` and `1088 s` rerun);
the load-sampled rerun, run under the heavier median (`6.0`), is the one that
put the `C = 38 … 42` `m = 4` cells `8–11 %` above the record. The record keeps
the slowest of the three runs on purpose:
with no quiet window available, a conservative figure with a measured band is
worth more than a fast one that cannot be reproduced, so the `s` columns stand
as upper bounds **by at most `12 %`**, that ceiling being measured here. The
ratios between neighbouring rows and every physics leaf are unaffected — all
three runs reproduce the step counts and every tail, error and probability
exactly, and the reruns differ only in `runtime_ms` and in the
`amplitude_defect` fields, which the current tree rounds to twelve significant
digits (`6f0d9c6`) and this record still stores at full precision (extension
note below).

**Trying to cut the other `36 %`, and failing.** Three variants were measured
against the §5.11 `C = 26` baseline (`axpy 2.21 s`, `dot 1.63 s`, solve
`9.77 s`) on one machine state, each A/B'd against the committed kernel:

| variant | `axpy` | `dot` | solve | verdict |
|---|---|---|---|---|
| committed (branch on a zero entry) | `2.49 s` | `1.86 s` | `12.06 s` | baseline of this A/B |
| branch-free (`select` on the same test) | `2.57 s` | `1.93 s` | `12.14 s` | no gain |
| no zero-skip at all | `2.34 s` | `1.86 s` | `12.14 s` | within noise, and drops the skip |
| **one** Gram–Schmidt pass instead of two | `1.29 s` | `0.97 s` | `10.73 s` | **fails the orthogonality pin** |

The two branch variants do nothing because the loop is not branch-limited: it
runs at `4.3–5.0 ns` per element in every form, i.e. it is not vectorised at
all and the `if` costs nothing next to the scalar complex multiply-add. The
one-pass variant is the real saving — `11 %` of the solve, `916` steps instead
of `917` — but it pushes the reorthogonalisation defect to the pin's edge:
`window_solve_matches_the_map_walk_at_solver_precision` fails with
`orthonormal basis 3: got 1.0004e-12, expected 0 (tolerance 1e-12)`. Two passes
are what hold that defect two orders below the tolerance, so the pass count is
not a free parameter and nothing here was adopted; `joint_sirk.rs` is unchanged.
So the only lever left on that `36.6 %` is an explicitly SIMD complex kernel —
which needs no new dependency (the target's intrinsics are in `core::arch`) —
or moving the loop to a device.

**An explicit NEON kernel: `2.5×` on the axpy, `3.75×` on the inner product,
`12.6 %` off the `C = 26` solve — adopted, record regenerated.** This host is `aarch64`
(Qualcomm Oryon, 12 cores, NEON + `fcma`), so "explicit SIMD" here means NEON,
not AVX2. `examples/phi4_kernel_simd_probe.rs` rebuilds both kernels with
`core::arch::aarch64` intrinsics — deinterleaving the array-of-structs pairs
with `vuzp1q/vuzp2q`, four independent accumulators for the inner product so
the dependent-add chain is broken — and times them against the committed shape:

| measurement | scalar | NEON | ratio |
|---|---|---|---|
| `axpy`, microbenchmark (`n = 2380`) | `2.31 us` | `0.93 us` | `2.48x` |
| `dot`, microbenchmark (`n = 2380`) | `1.44–2.81 us` | `1.39 us` | `1.03–2.07x` |
| `dot`, in the `C = 26` solve | `1.65 s` | `0.44 s` | `3.75x` |
| whole `C = 26` solve | `9.760 s` | `8.528 s` | `12.6 %` |

The `dot` microbenchmark row is a range on purpose: LLVM vectorises the
*scalar reference* in some compilation contexts and not others, so the
microbenchmark cannot settle that kernel. The solver can, and does — there the
scalar accumulator is latency-bound, which is why the end-to-end gain (`3.75×`)
exceeds the microbenchmark's. `matvec` was identical in both runs of that A/B
(`4.63 s` vs `4.62 s`), which is the control showing the two runs are
comparable. Correctness is unaffected in kind: `axpy` agrees to `2.2e-16` and
`dot` to `1.9e-14` relative.

It is **adopted**: `coords_inner_product` dispatches to this kernel on
`aarch64` and keeps the scalar loop as the fallback everywhere else. The two
objections that stood in the way were resolved rather than argued down. The
record was regenerated on this host: the `C = 26` rank-2 cell went
`12.261 s → 10.964 s` (`−10.6 %`) with `917 → 915` bounded-resolvent steps, the
`C = 32` cell `55.0 s → 48.9 s`, and the whole `13 × 2` sweep
`994.978 s → 865.507 s` (`14.4 min`, previous `16.5 min`) at the same `1.19 GB`
peak RSS. A handful of per-cutoff step counts moved by one — the rows updated in
§5.2 — and the recorded probabilities moved in their last places, at most
`1.4e-14` (`C = 36`: `0.9695602895694879 → 0.9695602895695016`), below the
precision at which §5.2–§5.4 quote them, so no quoted probability in this
document changed. The target dependence is now a stated property of the commit
rather than a reason to refuse it, and it is measured instead of assumed: the
scalar fallback is cross-compiled for `x86_64-unknown-linux-gnu` (GCC `16.2.0`,
glibc `2.44`) and run under `qemu-x86_64` `11.1.2`, where all seven tests in
`tests/phi4_momentum_sirk.rs` pass — including
`recorded_cutoff_sweep_matches_the_live_model`, which recomputes the `C = 12`
cell live and compares it against this record at `1e-12`. The digits are **not**
bit-identical and it would be wrong to say they are: of the `23` numbers the
suite prints under `--nocapture`, `15` differ, by at most `3.1e-15` absolute and
`2.7e-15` relative on the `O(1)` values — the summation-order difference between
four NEON accumulators and the serial scalar loop. One *discrete* number moves
too: the rank-2 Krylov count at `C = 8` is `18` on `aarch64` and `19` under
emulation, one extra iteration because the `1e-13` stop test sits on the
boundary the last-place difference crosses; the error it converges to agrees to
`1e-15`. The fallback therefore reproduces the record to solver precision but
not to the bit, the frozen record stays the `aarch64` run, and its
`source_revision` pins the commit that produced it. The prototype stays as
`examples/phi4_kernel_simd_probe.rs` so the number remains re-measurable against
the fallback.

**The device `gemv` is now runnable, and the host still cannot run it.**
`examples/phi4_gemv_device_bench.rs` builds the real `C = 32` window operator
and times the same dense product both ways; its CPU leg reproduces the solve's
per-step figure (`20.868 ms/step` here against the profile's `20.98 ms`),
which is what makes the comparison meaningful rather than a proxy. On this host
the device leg stops after the CPU leg and prints the project's own triage line:

```text
cutoff=32 labels=4845 matrix_mb=375.6 reps=5
cpu gemv: 20.868 ms/step (checksum 3.050567e0)
UK-GPU-OTHER → see the candle error below (…) (candle: the candle crate has not been built with cuda support)
device leg not run: no usable CUDA device.
```

To get the device number, run it where a driver and toolkit are installed
(`nvcc` is needed at build time; the CSR also has no CUDA device):

```text
PROBE_CUTOFF=32 cargo run -q -p fock_sirk --release --features cuda \
    --example phi4_gemv_device_bench
```

Two caveats are printed with the result rather than buried: the device leg
uploads the complex matrix as its real `2n × 2n` block form in `f32`, i.e. four
times the real multiplies of the complex product (conservative *against* the
device), and it times one `gemv` per step, so the per-step host↔device transfer
of the `4845`-complex vector (`77.5 KB` each way, order `10 µs` at PCIe rates)
is not included. Until that run exists, what the host side supports is
arithmetic only: the step streams `375.6 MB` in `21.0 ms`, a device-resident
matrix would be read at device bandwidth (order `0.4 ms` at `1 TB/s`), and the
transfer is negligible — plausible, **not** demonstrated, and no speedup is
claimed.

**This regeneration moves the last bits, by design.** The record was generated
on the adopted kernel, so every cell differs from the previous fixture: runtime
down (`994.978 s → 865.507 s`; `994978 → 865507` in the file), step counts by
one at a handful of cutoffs (`917 → 915` at `C = 26`, `1328 → 1327` at `C = 32`),
and probabilities and defects in their twelfth significant digit or beyond
(largest `1.4e-14`, at `C = 36`). Tails, label counts and rank structure are
unchanged; the `C = 34/36` rows still buy tail, not accuracy. The `13` cutoffs ×
`2` ranks cost `14.4 min` at `1.19 GB` peak RSS, the memory being the `C = 36`
cache of `5.37e7` components. Compare with §5.11's `11` cells in `6.3 min` at
`0.60 GB`: summed rank-2 window labels rose `19137 → 32437` (`+70 %`) for
`+131 %` wall clock, because a step now costs `n²` entries and there are more
steps in a wider window.

**The extension to `C = 44` adds rows and changes no existing cell.** The
regenerated record has `17 × 2` cells in `4978 s` (`83.0 min`) at `4.20 GB` peak
RSS; `2865 s` (`47.8 min`) of that is the four new cells (`391 s`, `717 s`,
`574 s`, `1183 s`), and the thirty pre-existing cells reproduce their step
counts, tails, errors and probabilities exactly. The diff against `078d29c`'s
fixture is therefore four new cells, thirteen `runtime_ms` leaves,
`peak_rss_kb`, `git_head` and the record's own `note` string (which now says the
sweep runs to `C = 44`) — nothing was retuned to make the wider windows
affordable. This run was **not** quiet: other repositories' jobs took the host to
a load average of `9.6` on 12 cores at the end (`30.7` at peak), and the
re-measured clocks moved by up to `±12 %` in both directions against the previous
quiet run. The physics is load-invariant — every one of the thirty pre-existing
cells is identical leaf for leaf — which is the same conclusion the earlier
loaded/quiet pair reached (`3593560` loaded against `2158332` quiet for the
`15 × 2` record, with no physics difference). So treat the new cell times as
upper bounds — by at most `12 %`, once two reruns measured the spread (§5.12) —
and compare ratios inside the record, not against another machine.

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
   (`1.79 → 1.40 → 0.89 → 0.51 → 0.32 → 0.26 → 0.21 → 0.17 → 0.13 → 0.10 →
   0.08 → 0.063 → 0.049` for `C = 4, 8, 12, 16, 20, 22, 24, 26, 28, 30, 32, 34,
   36`). It does not reach zero, and the last rows cost 0.96 s, 2.5 s, 4.0 s,
   6.7 s, 11.0 s, 17.9 s, 30.9 s, 48.9 s, 76.3 s and 118 s per `m = 2` cell, so
   the window cannot be pushed much further by brute force — and, as §5.5–§5.6
   measure, neither by parallelising the action matvec nor by parallelising the
   Krylov arithmetic. The remaining cost is the Arnoldi steps themselves, of
   which the projected `gemv` is now `62 %` (§5.12); the per-step least squares
   is unmeasurable (§5.11). Removing that `gemv` in favour of the sparse action
   costs `8.1×`, measured, so the step is already the cheap one.
3. **`ForwardProducts` is not worse here** at equal rank (see §5.2). Do not
   quote the plan's "Arnoldi deteriorates" as a rank comparison on bounded
   windows; the observable degradation is growth with `C`.
4. **GMRES iteration counts are exposed and recorded.** `JointSirkResult`
   carries `shift_solve_iterations` (one Arnoldi step count per rational basis
   solve, in the same order as `shift_solve_error_bounds`) and
   `function_solve_iterations` (one per bounded-resolvent solve), with the
   `max_*` / `total_solve_iterations` accessors. The response cells need
   `[12, 13, 14, 15]` basis steps and `[15, 15, 15, 15]` resolvent steps   under the `64`-step cap at `1e-13`; the cutoff sweep climbs to `[163, 212]` at
   `C = 24`, `[190, 241]` at `C = 26`, `[266, 351]` at `C = 32`, `[315, 420]` at
   `C = 36`, `[340, 459]` at `C = 38` and `[367, 498]` at `C = 40`
   (`[209, 211, 214, 216]`, `[239, 242, 245, 247]`, `[419, 432, 433, 435]` and
   `[500, 514, 518, 519]` at rank 4) and needs the looser `1e-10`/1024 policy
   described in §5.2 (528 total steps at `C = 20`, 800 at `C = 24`, 915 at
   `C = 26`, 1327 at `C = 32`, 1590 at `C = 36`, 1733 at `C = 38`, 1879 at
   `C = 40`). Every count is in the record —
   no `null` stands in for a measurement. These counts carry about one step of
   rounding sensitivity to the least-squares path (§5.11); they are diagnostics,
   not the contract — the contract is the full-action residual.
5. **`cadabra2-cli` and a C compiler are Nix-only on this host**; the CDB gate
   needs `PYTHONHOME` from the Cadabra2 closure to avoid
   `ModuleNotFoundError: _struct`.
6. The two pre-existing `.cdb` failures in §3 are unrelated to this suite and
   remain open.

## 7. Honesty summary

* Symbolic (`cadabra`/`.cdb`): all 32 λφ⁴ zero-intent checks reduce to `0`;
  the fixture is cross-checked against the live operator.
* Window reference: exact evolution of each declared window — a Hermitian-Lanczos
  seed column of `exp(-i t H_window)` with a `1e-13` convergence check on the column
  (§5.8), not the dense floating-point Padé `n³` factorisation it replaced.
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
  the recorded cells, which is what made `C = 26` affordable (§5.6). Bit-identity
  is pinned by `fock_sirk/tests/joint_map_ops.rs`, which keeps the previous
  implementations as the oracle and compares every amplitude on `to_bits()`.
* Measured and *not* extrapolated: before §5.9 the map operations dominated the
  solve (`87 %` axpy including a `28 %` dot, `3.4 %` matvec, `6.5 %` SVD at
  `C = 26`), and in the cell as recorded then they were only half of it — `177 s`
  of `406.4 s` was the dense Padé reference, whose `n³` growth set the cost of
  the next cutoff (`723 s` for a `C = 28` cell); the reference is now the seed
  column. The parallel variants lose by more than at `C = 20`, and the chunked
  dot's association error rose to `3.5e-15` (§5.7).
* Measured and adopted: `joint_axpy` is now one ordered walk over the two maps
  instead of a tree descent per entry. **20.8×** per call at 715 labels and
  **21.4×** at 1365, which is **1.88×** end to end (`C = 26`: `196.0 s →
  104.0 s`; `C = 20`: `24.1 s → 12.9 s`) with *bit-identical* probabilities and
  step counts — the eight previously recorded cells are unchanged to the last
  bit, which is how the record regenerated after the change stays comparable
  (§5.9).
* Measured and *not claimed*: §5.9 moved the bottleneck rather than removing it —
  the inner product inside the same orthogonalisation pass was then `>50 %` of
  the solve, at about one deep label comparison per entry, and that is what §5.10
  then removed.
* Measured and adopted: the inner shift solves now run in coordinates of the
  declared window (`dense_window_solve`, default on), with the nested map path
  kept behind the same flag as the reference. `4.0–4.4×` on the recorded cells
  (`C = 26`: `104.8 s → 26.0 s`; `C = 28`: `188.7 s → 44.8 s`; `m = 4`:
  `187.5 s → 42.8 s` and `334.7 s → 75.5 s`), the whole record `18.7 min →
  4.5 min`, peak RSS `1.01 GB → 0.29 GB` — and, at the time, *bit-identical*:
  that record differed from the previous one in only `runtime_ms`, `peak_rss_kb`
  and `git_head` (§5.10). It does put a second arithmetic path into a solver the
  SM certificates share, which is why the map path is still there to be compared
  against and `sm_full_hashimoto` (13/13) and `qym_magnetic_packet` (2/2) are
  re-run on it.
* Measured, and then removed: with the maps gone from the inner loop, the
  per-step least-squares SVD was `56 %` of the `C = 26` solve and `58 %` of the
  `C = 28` one. §5.11 replaced it with the standard GMRES construction — one
  complex Givens rotation per Arnoldi step — and the stage is now below the
  profile's `10 ms` resolution (`< 0.2 %`), taking the `C = 26` solve from
  `22.5 s` to `9.77 s` and `C = 28` from `40.5 s` to `16.72 s`.
* Measured and reported as a *loss*: that change ends the `to_bits()` equality of
  §5.10. The coordinate path now agrees with the map walk to `1e-12` relative
  (same rank, step counts within one) rather than exactly, the test was renamed
  to say so, and the record differs from the previous one in 188 numeric leaves —
  probabilities by ≤ `3.3e-15` relative, nothing above `8.9e-10`, that worst case
  a difference of two near-equal probabilities. Step counts stay part of the
  record but are now a one-step-rounding diagnostic, not the acceptance test: the
  acceptance is still the full-action residual against each cell's recorded
  `solve_policy` (§5.11).
* Measured and adopted: the sweep now runs `C = 30` and `C = 32`. The tail keeps
  shrinking (`0.1337 → 0.1037 → 0.0802`, `0.77×` per step), the rank-4
  inside-window error stays flat (`2.86e-3, 2.88e-3, 2.88e-3`) and the
  window-exact probability settles to a `8.0e-7` band, so a wider window still
  buys tail and not accuracy. `C = 32` is `4845` labels, `2.36e7` components, and
  is where `ActionLimits::max_work` (default `1e7`) first bound — the runner
  raises the limit with the window instead of charging the same `gemv` less
  work. The record is `11 × 2` cells in `6.3 min` and `0.60 GB` (§5.11).
* Measured: at `C = 32` the projected
  `gemv` is `70.7 %` of the `39.4 s` solve measured on a quiet host (`27.84 s`
  of `1327` steps, `20.98 ms` per step, `17.9 GB/s` on a `375.6 MB` matrix),
  against `9.5 %` Gram–Schmidt, `3.5 %` inner products and `< 0.03 %` least
  squares on the adopted NEON kernel (§5.12). Removing it is only possible by
  going back to the sparse nested action, measured at `169.518 ms` per step
  (`8.08×` the `gemv`, a `~246 s` cell instead of the probe's `49.1 s`), so the
  removal is a regression, not a saving. `f32` storage and a GPU `gemv` are reported as
  considered and not done — the first contradicts the recorded `1e-10`
  residual, the second is unmeasured (§5.12).
* Measured and adopted: the sweep now runs `C = 34` and `C = 36` as well
  (`5985` and `7315` labels, `3.59e7` and `5.37e7` components). The tail keeps
  shrinking (`0.0802 → 0.0625 → 0.0495`, `0.78×` / `0.79×`), the rank-4
  inside-window error stays flat (`2.884e-3, 2.882e-3`) and the window-exact
  probability stays in the same `8.0e-7` band, so again more window buys tail
  and not accuracy. The record was regenerated on the adopted NEON inner
  product: every cell differs from the previous fixture in `runtime_ms` (down),
  in a handful of step counts (by one), and in probabilities and defects from
  the twelfth significant digit on (≤ `1.4e-14`); tails and rank structure are
  unchanged. It is `13 × 2` cells in `14.4 min` and `1.19 GB` peak RSS (previous
  `16.5 min`, `1.19 GB`) (§5.12).
* Measured and adopted: the same sweep now runs `C = 38` and `C = 40` as well
  (`8855` and `10626` labels, `7.84e7` and `1.129e8` components, the budget
  raised to `1.3e8`). The tail keeps shrinking
  (`0.0495 → 0.0397 → 0.0321`, `0.80×` / `0.81×`) and the rank-4 inside-window
  error stays flat (`2.880e-3, 2.879e-3`), so the wider windows buy tail and not
  accuracy, at `311 s` and `535 s` per `m = 4` cell. The pre-existing
  twenty-six cells are unchanged to the last bit; the record is `15 × 2` cells
  in `35.9 min` and `2.29 GB` peak RSS, the memory being the `C = 40` cache of
  `1.129e8` components (§5.12).
* Measured: the sweep now runs `C = 42` and `C = 44` as well (`12650` and
  `14950` labels, `2.235e8` components at `C = 44`, the budget raised to
  `3.0e8`). The tail keeps shrinking (`0.0321 → 0.0259 → 0.0208`, `0.81×` /
  `0.80×`) and the rank-4 inside-window error **does not leave its plateau**:
  `2.8787e-3, 2.8788e-3, 2.8793e-3` at `C = 40/42/44`, i.e. minimal at `C = 40`
  and `0.02 %` above that at `C = 44` — an upturn of `5.9e-7` absolute, two
  orders below the plateau's own `1.04×` spread. So no cutoff in `C = 8 … 44`
  recovers the rank-4 error: it is rank-limited, and more window buys tail only,
  at `717 s` and `1183 s` per `m = 4` cell. The thirty pre-existing cells are
  unchanged leaf for leaf (thirteen `runtime_ms` values and the record's `note`
  are the only other differences); the record is `17 × 2` cells in `83.0 min`
  and `4.20 GB` peak RSS. The generating run was loaded (load average `9.6`,
  peak `30.7`), and two further reruns — one with the load sampled every
  `30 s`, median `6.0`, minimum `1.7` — keep every cell within `12 %` of it
  (`19` within `3 %`), the `C = 44` `m = 4` cell `8–12 %` lower, so the `s`
  columns are upper bounds by at most `12 %`, that ceiling measured rather than
  assumed. The record keeps the slowest run (§5.12).
* Measured: the scalar fallback is not asserted to reproduce the `aarch64`
  record, it is run. Cross-compiled for `x86_64-unknown-linux-gnu` (GCC `16.2.0`,
  glibc `2.44`) and executed under `qemu-x86_64` `11.1.2`, all seven tests of
  `tests/phi4_momentum_sirk.rs` pass — `recorded_cutoff_sweep_matches_the_live_model`
  included, which pins the live `C = 12` cell to this record at `1e-12`. The
  printed digits differ from this host's by at most `3.1e-15` absolute and
  `2.7e-15` relative on the `O(1)` values, and the `C = 8` rank-2 Krylov count
  moves `18 → 19` where the `1e-13` stop test sits on the crossed boundary
  (§5.12).
* Corrected: the rank-4 `C = 28` step row of §5.2 had been quoting that cell's
  shift-solve counts under a bounded-resolvent heading. It now shows the
  resolvent counts; no measurement changed, the label was wrong (§5.11).
* Measured and adopted: the reference tier is now the seed column of
  `exp(-i t H_window)` (Hermitian Lanczos, `1e-13` on the column), not the dense
  `n³` Padé exponential. At `C = 28` the reference falls from `366.9 s` to
  `3.47 s` for a probability that moves by `4.4e-15`, taking the cell from
  `723 s` to `360 s` — after which the solve, not the reference, is again what a
  wider window costs (§5.8). §5.9 then halves the solve — the same cell is a
  recorded row at `188.7 s` — and §5.10 halves it again to `44.8 s`.
* On the base cell the probability charge fell from about `0.394` (state)
  to `0.000763` (amplitude). The paired amplitude charge is `0.001786` for
  coupling and `0.01266` for momentum spread. Rigorous response signs for
  the full fibre orbit remain open. [Enclosure work order](PHI4_NUMERICAL_PLAN.md)
