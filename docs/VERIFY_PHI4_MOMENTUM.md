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
| `nested_fock_algebra/src/phi4_momentum.rs` | `P11` operator: `ScalarMomCont`, `h_cont` on four continuous coordinates, Mehler widths, even/parity windows, `S₃` mode table, frozen comparisons, 9 unit tests |
| `docs/phi4_momentum_nested.cdb` | Cadabra2 derivation of `h_cont`, the 11×3 `S₃` table, the seed moments and the rank-2 return interval |
| `docs/fixtures/phi4_localized_wall.json` | Classical two-packet localized wall, checked against a spatial Gaussian integral and replayed by Rust. |
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
* repository `HEAD = 2ea6356` when the record was regenerated (the fixture's
  `source_revision.git_head` records the same commit)

| file | SHA-256 |
|---|---|
| `docs/phi4_momentum_nested.cdb` | `669510d3c55d334b065b0d94c6f7a4d402cdf29408865856178c3b475fb8d709` |
| `docs/fixtures/phi4_momentum_mehler.json` | `9b1e9c15ccbb53c876a4818f8d2898d8faec5f26c54d0664406cbcc9912d15bb` |
| `docs/fixtures/phi4_momentum_sirk_sweep.json` | `44b38944912732bb37c5be292ec9a7791283b98390c7b0bd668cf4e372356cc0` |

The whole record comes from that one run: `source_revision.git_head` and the
digest above are the run's own. It was produced natively on an **`x86_64`
host** on the **AVX2 kernels**: `coords_inner_product` and `coords_axpy` both
dispatch to their AVX2 path here, so the digits are no longer the scalar
fallback's (`~1e-15` away, which the suite's `1e-12` checks would have absorbed
either way). `coords_norm`'s AVX2 kernel (§5.12) postdates this run, so the
record's last digits are the path *before* that third kernel: on this host the
`C = 26` live solve now differs from the recorded value by `3.3e-15` in the
return probability (`916` steps instead of `915`), which is the same order as
the reassociation already in the record and far inside the reported precision. The per-cell rows of §5.2 are this record's,
and the profile measurements of §§5.5–5.12 were re-run on the same `x86_64`
host; the tables that remain the earlier `aarch64` campaign's are labelled as
such in place. The run is the first record to apply the
twelve-digit reporting now in the tree: `AmplitudeBound::to_json` routes measured
fields through `sm_bridge::reported_residual`, so
`responses.cells.*.amplitude_defect` is stored at twelve significant digits
(`4.44614460857e-16` for the base `initial_leakage`, raw
`4.446144608574858e-16` in `full_action_defect`). Two field classes are
machine- rather than model-dependent and move between runs: `runtime_ms` /
`peak_rss_kb`, and the solver diagnostics already at the `1e-16` floor
(`hermiticity_defect`, `orthogonality_defect`), whose *relative* spread is large
while the absolute defect stays at rounding noise.

## 3. Replay

```bash
# symbolic half (needs cadabra2-cli; PYTHONHOME is required on this host)
env PYTHONHOME=<python-in-cadabra-closure> \
  nix shell nixpkgs#cadabra2 --command \
  bash scripts/verify_cdb_modules.sh --strict

# unit tests for the continuous operator
cargo test -p nested_fock_algebra phi4_momentum

# the P12 suite (needs a C compiler, e.g. via `nix shell nixpkgs#gcc`)
cargo test -p fock_sirk --test phi4_momentum_sirk   # 9 tests; ~5 s release

# regenerate the evidence record (151.0 min in release on the x86_64 host of
# record; the aarch64 record it replaced took 83.0 min). The seventeen cutoff
# cells cost 22 ms, 0.12 s, 0.44 s, 1.23 s, 3.40 s, 6.1 s, 10.6 s, 17.7 s,
# 29.1 s, 52.6 s, 88.9 s, 137 s, 215 s, 346 s, 508 s, 753 s and 1114 s at
# m = 2 and 36 ms, 0.24 s, 0.88 s, 2.94 s, 7.7 s, 12.5 s, 18.6 s, 29.9 s,
# 54.3 s, 93.5 s, 148 s, 252 s, 386 s, 597 s, 903 s, 1309 s and 1951 s at
# m = 4; it peaks at 4.24 GB RSS. These are wall clocks on a shared host, so
# they are upper bounds; the debug build is impractical here)
cargo run -p fock_sirk --release --example phi4_momentum_sirk

# the measurements of section 5.5 (serial, then default workers), plus the
# per-operation profile of the Krylov solve
FOCK_SIRK_ACTION_THREADS=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe
cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe
PHI4_PROFILE=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe

# the section 5.11 profile: same probe at C = 26 (20 s on this x86_64 host: a
# 16.2 s solve after the Givens least squares — 9.8 s on the NEON aarch64 host
# of section 5.12 — then a 3.3 s seed-column window reference;
# PROBE_COMPONENT_BUDGET must exceed the window's component count, 5.71e6 here)
PROBE_CUTOFF=26 PROBE_COMPONENT_BUDGET=25000000 PHI4_PROFILE=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe

# the section 5.11 C = 28 cell (35 s, the same budget; the same solve took
# 16.7 s on the aarch64 host after section 5.11, and 40.5 s before it)
PROBE_CUTOFF=28 PROBE_COMPONENT_BUDGET=25000000 PHI4_PROFILE=1 \
  cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe

# the section 5.12 profile: the same probe at C = 32 (98 s; the budget must
# exceed that window's components, 2.36e7). The gap between `matvec_ms` (one
# sparse nested action, 321 ms here) and the solve's per-step `PROFILE matvec=`
# (the projected gemv, 41.27 ms) is the measured price of not having the gemv.
# This is a timing probe, so run it quiet: the recorded run started at load
# average 0.5 and its gemv lands on the bench's 41.301 ms/step; the aarch64
# campaign host ran the same two at 20.98 and 20.868 ms.
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
for tag, name in [("PHI4_MOM_FIXTURE", "phi4_momentum_mehler"),
                  ("PHI4_LOCALIZED_FIXTURE", "phi4_localized_wall")]:
    [payload] = [r.split("|", 1)[1] for r in rows if r.startswith(tag + "|")]
    pathlib.Path("docs/fixtures", name + ".json").write_text(
        json.dumps(json.loads(payload), indent=2, sort_keys=True) + "\n")
PY
```

The historical fibre run at the revision above reported:

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

Rank-2 tail sweep (`N = 8, h = 3`), `max_k ‖(I − P_C) H V_k‖`. The `C = 16 … 44`
rows extend the plan's `C ≤ 12` range past 20: the tail keeps falling, but the
shift-solve steps the smallest scheduled shift (`γ = 40`) needs grow with the
window, and the runtime grows faster than the label count. The step column is
`shift_solve_iterations` — one Arnoldi count per rational basis solve, in shift
order; the bounded-resolvent counts are in the record's
`function_solve_iterations` (the rank-4 table below reports those).

| `C` | labels | tail | shift-solve steps | cell runtime |
|---|---|---|---|---|
| 4 | 15 | `1.7866` | `[6, 6]` | `22 ms` |
| 8 | 70 | `1.4001` | `[17, 19]` | `0.12 s` |
| 12 | 210 | `0.8946` | `[38, 43]` | `0.44 s` |
| 16 | 495 | `0.5067` | `[70, 82]` | `1.23 s` |
| 20 | 1001 | `0.3189` | `[112, 138]` | `3.40 s` |
| 22 | 1365 | `0.2615` | `[138, 173]` | `6.1 s` |
| 24 | 1820 | `0.2129` | `[163, 212]` | `10.6 s` |
| 26 | 2380 | `0.1703` | `[190, 241]` | `17.7 s` |
| 28 | 3060 | `0.1337` | `[213, 281]` | `29.1 s` |
| 30 | 3876 | `0.1037` | `[240, 312]` | `52.6 s` |
| 32 | 4845 | `0.0802` | `[266, 350]` | `88.9 s` |
| 34 | 5985 | `0.0625` | `[289, 385]` | `137 s` |
| 36 | 7315 | `0.0495` | `[315, 419]` | `215 s` |
| 38 | 8855 | `0.0397` | `[340, 458]` | `346 s` |
| 40 | 10626 | `0.0321` | `[367, 497]` | `508 s` |
| 42 | 12650 | `0.0259` | `[393, 535]` | `753 s` |
| 44 | 14950 | `0.0208` | `[419, 575]` | `1114 s` |

The `C ≥ 16` cells solve to a `1e-10` relative residual with a 1024-step cap.
This paragraph used to say that at `C = 20` the `1e-13` target stalls at
`2.3e-13` and exhausts even 512 steps; that no longer describes the solver.
Re-tested on 2026-11-27, the `C = 16, 20, 22, 24, 26, 28` cells (both ranks)
accept `1e-13` in `83, 142, 177, 217, 261, 308` steps (`m = 2`) and
`83, 143, 177, 217, 260, 306` (`m = 4`), with no stall and no cap hit. The
`C ≥ 30` cells were not re-tested at `1e-13` and keep the looser target, which
is six orders below the `1e-3`-scale inside-window error
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
from `C = 20` to `C = 44` while the cell cost grows `3.4 s → 1114 s`. That cost is
now the Arnoldi steps themselves, not the window and no longer the least
squares: of the `C = 28` cell the exact reference is `4.7 s` and the Krylov
solve `24.9 s` (§5.11), and the per-step least-squares stage is below the
profile's `10 ms` print resolution. Within the solve it is now the projected
`gemv` — `75 %` of the `C = 32` solve (§5.12) — that a wider window charges.

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
| 12 | `2.43e-3` | `1.40e-4` | `[43, 43, 45, 45]` |
| 16 | `2.73e-3` | `1.74e-4` | `[81, 83, 83, 84]` |
| 20 | `2.99e-3` | `1.86e-4` | `[137, 137, 139, 141]` |
| 22 | `2.90e-3` | `1.82e-4` | `[171, 173, 175, 176]` |
| 24 | `2.85e-3` | `1.85e-4` | `[209, 212, 214, 216]` |
| 26 | `2.84e-3` | `1.84e-4` | `[240, 242, 246, 247]` |
| 28 | `2.86e-3` | `1.83e-4` | `[278, 286, 284, 286]` |
| 30 | `2.88e-3` | `1.84e-4` | `[310, 318, 320, 322]` |
| 32 | `2.88e-3` | `1.84e-4` | `[350, 359, 361, 359]` |
| 34 | `2.88e-3` | `1.84e-4` | `[384, 396, 398, 399]` |
| 36 | `2.88e-3` | `1.84e-4` | `[419, 432, 433, 435]` |
| 38 | `2.88e-3` | `1.84e-4` | `[459, 472, 476, 475]` |
| 40 | `2.88e-3` | `1.84e-4` | `[500, 514, 518, 518]` |
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
2.8793e-3` differs in the third digit (`1.05×` end to end), which is the
`solve_policy` tolerance's neighbourhood, so those thirteen cells are read as
equal, not as a decaying sequence. **The plateau does not break.** `C = 42` and `C = 44` put the same
question to the two widest windows: `2.8788e-3` and `2.8793e-3`, against
`2.8787e-3` at `C = 40` — the floor of the `C = 34 … 44` tail sits at `C = 40`
and the last two cells are `0.006 %` and `0.02 %` above it, an upturn of `5.9e-7`
absolute, two orders below the `1.05×` spread of the plateau itself. (The sweep's
lowest value is `2.842e-3` at `C = 26`, not at `C = 40`; the plateau's `1.9e-6` band
is read as rank-limited noise, not as structure that could pick out a cutoff.) So there is
no cutoff in `C = 8 … 44` at which the rank-4 inside-window error leaves its
plateau: it is rank-limited throughout, and widening the window cannot recover
it. Growing `C` past 12 therefore buys tail, not accuracy, at a rising cost
(0.88 s → 2.94 s → 7.7 s → 12.5 s → 18.6 s → 29.9 s → 54.3 s → 93.5 s →
148 s → 252 s → 386 s → 597 s → 903 s → 1309 s → 1951 s per `m = 4` cell).
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

**Change-of-basis control (`docs/PHI4_NUMERICAL_PLAN.md` §3.4).** The
identical-cell null above re-runs one tuple; the control transports a different
one. The diagonal phase unitary `U = exp(i Σ_m θ_m n_m)` with
`θ = (0, 0.3, −0.45, 1.1)` sends `h → U h U†` and `ψ → Uψ`, so its return
probability is invariant by construction. The seed is one `φ` quantum in the
`φ`-parity-1 window (`C = 8`, 35 labels); `θ_φ = 0` leaves it bit-identical
because it carries no momentum quanta, while the momentum quadratures transform
as `q_r → cos θ_r q_r + 2 sin θ_r p_r` and
`p_r → cos θ_r p_r − ½ sin θ_r q_r` in the model's unnormalized pair. The
window matrix therefore really changes
(`‖P H' P − P H P‖_F = 78.6`), and the conjugation is checked symbolically
before any solve: `h' Γ(U)ψ = Γ(U) h ψ` to `1.1e-15`.

| arm | `p_sirk` | `p_exact` | sharp |
|---|---|---|---|
| base | `0.838723759846` | `0.838683297014` | `4.05e-5` |
| transported | `0.838723759846` | `0.838683297014` | `4.05e-5` |

The exact response is `0` — identical bits, not merely small — and the
compressed response is `3.1e-15` inside an `8.09e-5` sharp budget, so the
control reports no physical response. That is the point of transporting the
tuple: the `λ` and `width_k` responses above change the physical problem and do
exclude zero, whereas a pure change of basis cannot.

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
the original `aarch64` host (12 cores, 30 GB); only one survived. The table and
paragraphs below were re-measured for this section on the `x86_64` host of the
record (12 cores, 15 GB).

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
  across workers measured **slower**: `46.0 ms → 62.5 ms` per `C = 20` matvec
  (`aarch64`; the `x86_64` serial matvec of the table below is `77.6–81.5 ms`).
  One fork/join per factor does not amortise against a per-entry cost of a few
  map lookups. It was removed rather than left behind a threshold.

`FOCK_SIRK_ACTION_THREADS` forces the worker count (that is how the serial
column below was measured); the width is also capped by the term count, and
`h_cont` has **three** top-level terms — `kinetic`, `gradient`, `wall` — so the
sum split can never exceed 3-way.

Measured with `fock_sirk/examples/phi4_momentum_parallel_probe.rs`:

| quantity (`C = 20`, 1001 labels) | serial | parallel | ratio |
|---|---|---|---|
| `d_gamma` on a dense window state, three runs | `72.60 / 76.54 / 78.05 ms` | `46.12 / 47.36 / 49.00 ms` | **1.48–1.69×** |
| `m = 2` Krylov solve | `1.766 / 1.768 / 1.779 s` | `2.491 / 2.498 / 2.565 s` | `0.69–0.71×` |
| total Arnoldi steps | `527` | `527` | — |
| return probability | `9.80001510278452170e-1` | identical to all 17 digits | — |

All six runs report the same probability, which is the end-to-end check that
the split really is order-preserving and not merely accurate; the checked-in
record, generated on the scalar kernel, stores `0.9800015102784558` for this
cell, so the AVX2 reassociation moves it by `3.7e-15` and nothing else. The solve column is
`0.68–0.73×` rather than `1.00×` because a fork/join is not free at this size: the
same probe's window build, which applies one action per label, costs `0.060 s`
serial against `0.779 s` parallel — ≈ `0.7 ms` of per-call overhead — and the
solve's full-action acceptance step pays it too. The window solve itself uses the
projected `gemv`, not `d_gamma`, so no split of the action can touch it.

So the action parallelisation alone is a **1.5× matvec** that the coordinate
solver never calls, and a **1.00× sweep**. It matters only on the map path: the
standalone `d_gamma` is `7.8×` the `gemv` at `C = 32` (§5.12), so the serial map
walk is what the split would rescue, not this solve. On the earlier `aarch64` host
the same solve column read `61.67 / 62.92 s` because it predates §5.6–§5.11's
map-operation, coordinate and Givens changes; those sections measure the sequence
that took `C = 20` from `62.9 s` to the `1.8 s` above.

Bit-identity is asserted, not assumed: `mod parallel_action_tests` in
`exact_inner.rs` runs three release tests — an order-preserving worker merge on
a 96-label 2-particle state (threads `1/2/3/4/7/16`, bit-equal vectors and
equal `d_gamma`), a bisected `max_work` limit that must trip at the same work
total serial and at 4 threads, and a 1001-label window state mapped through
forced 1 / default / 7 threads. They pass in `0.2 s`.

### 5.6 What actually cost the time: the two map operations (`P13`)

`PHI4_PROFILE=1` makes `gmres_shift` accumulate wall time for the action
matvec, the nested `axpy`, the nested inner product and the per-step basis SVD.
The table below is the `aarch64` profile that motivated §5.9, taken while the map
walk was still the inner loop; the code that produced it no longer exists, so it
is a historical record, not a re-runnable probe. The same probe on the current
`x86_64` solver is the second table.

`aarch64`, before §5.9 — at `C = 20` (1001 labels), `m = 2`, the two shift solves
of the first basis entry (`[112, 138]` steps):

| phase | 112-step solve | 138-step solve |
|---|---|---|
| action matvec | `0.18 s` | `0.23 s` |
| `joint_axpy` inside MGS (the row below is inside it) | `3.25 s` (12656 calls, `0.257 ms`) | `5.53 s` (19182 calls, `0.288 ms`) |
| `joint_inner_product` | `1.02 s` (12656 calls, `0.081 ms`) | `1.77 s` (19182 calls, `0.092 ms`) |
| per-step basis SVD | `0.19 s` | `0.42 s` |

So the action was ≈ **1.4 %** of the solve and the two nested-vector map
operations ≈ **90 %** of it. Rebuilding GMRES with Givens rotations would
target the `0.6 s` of SVD — about 2 % — while changing every pinned digit, so it
was rejected on the measurement, not on taste. (§5.11 later did exactly that,
because by then the SVD was 56 % of the solve; the statement above is about the
map-walk distribution, not the coordinate one.)

`x86_64`, current coordinate solver with the AVX2 `dot` and `axpy` (the `norm`
kernel, below in §5.12, postdates this table) — the same four
solves of that cell (`112`, `138`, `137`, `140` steps, `2.669 s` total):

| phase | 112-step | 138-step | 137-step | 140-step | total | share |
|---|---|---|---|---|---|---|
| action `gemv` | `0.19 s` | `0.23 s` | `0.23 s` | `0.24 s` | `0.89 s` | `33.3 %` |
| MGS `axpy` (contains the dot) | `0.03 s` | `0.04 s` | `0.04 s` | `0.04 s` | `0.15 s` | `5.6 %` |
| — the inner product inside it | `0.01 s` | `0.01 s` | `0.01 s` | `0.01 s` | `0.04 s` | `1.5 %` |
| Givens least squares | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.4 %` |

The remaining `1.63 s` (`61.1 %`) is the same "everything else" as in §5.12: the
`coords`/`from_coords` round-trip at the ends of each solve, each step's image
construction and the norm of each new basis vector. At 1001 labels the fixed
per-solve overhead dominates, which is why its share falls steadily as the window
grows (`61.1 %` here, `29.4 %` at `C = 26`, `23.4 %` at `C = 28`, `17.0 %` at
`C = 32`).

Both map operations were also doing gratuitous work (`aarch64` measurements of
the steps that §5.9 and §5.10 then replaced):

* `joint_inner_product` walked `a` and ran a `b.get(&label)` **binary search**
  per entry. Both vectors come from the same window dictionary in label order,
  so it is now a two-pointer merge: same products, same left-to-right
  accumulation order, no search. `dot_ms` `0.27 → 0.039 ms` at 1001 labels
  (`aarch64`).
* `joint_axpy` recomputed each summand and then `insert(basis.clone(), …)`,
  cloning a `JointOuterBasis` (and forcing a node split) per entry. It now
  updates in place through `get_mut`, inserting only genuinely new labels and
  still deleting zeros. `axpy_ms` `0.85 → 0.39 ms` (`aarch64`).

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

| operation (`C = 20`, 1001 labels, `x86_64`) | direct | 1 worker (chunked) | 2 workers | 4 workers | 8 workers |
|---|---|---|---|---|---|
| `joint_inner_product` (chunk partials, chunk order) | `0.086 ms` | `0.066 ms` | `0.307 ms` | `0.274 ms` | `0.269 ms` |
| `joint_axpy` (chunk deltas, applied in label order) | `0.101 ms` | `0.470 ms` | `1.684 ms` | `1.699 ms` | `1.868 ms` |

The axpy row needs a fresh copy of the target per repetition (the operation
mutates it), so all four columns share that cost (`map_clone_ms = 0.254`); even
so, a 1-worker run of the same chunked helper measures `0.470 ms` against
`0.101 ms` for the direct call, i.e. the split's bookkeeping alone is a `4.7×`
tax. A fork/join on this host costs ≈ `0.2 ms` (the 2-worker dot is `0.307 ms`
against `0.086 ms` direct), several times the whole 1001-label dot and a
comparable share of the axpy — the same lesson as the rejected per-entry map
split in §5.5, and the reason the sweep is not made faster by adding workers.
At `C = 26` (2380 labels) the direct dot is `0.149 ms` and the 1-worker chunked
axpy `1.40 ms` against `0.255 ms`, so the tax grows with the entry count. The
parallel dot is additionally **not bit-identical**: summing chunk partials in
chunk order associates the additions differently from the left-to-right fold,
and the returned scalar differs from the serial one by `1.1e-15` relative at
`C = 20` (`3.5e-15` at `C = 26`), which is enough to break the pinned 17-digit
probability.

So the honest headline for `P13` is: a **1.5× action matvec, no gain from
parallelising the Krylov arithmetic, and a 2.0–2.6× end-to-end win from
replacing two per-entry map operations with structural ones**. `C = 22 … 26`
were reached with that 2× plus wall clock (`30 min` for the recorded sweep).

### 5.7 C = 26: the same two operations, but the cell is now half reference

The question for the extension was whether the `C = 20` profile survives the
move to `C = 26`. It did, and the arithmetic share edged up. That profile is the
`aarch64` *before* picture of §5.9 (190/242/242/244 steps), when the map walk was
the inner loop; the current solver's profile of the same cell follows it.

`aarch64`, before §5.9 — `PROBE_CUTOFF=26 PHI4_PROFILE=1` on 2380 labels,
`m = 2`, four shift solves, 918 steps total:

| phase | 190-step | 242-step | 242-step | 244-step | total | share of the solve |
|---|---|---|---|---|---|---|
| action matvec | `1.36 s` | `1.76 s` | `1.75 s` | `1.76 s` | `6.63 s` | **3.4 %** |
| `joint_axpy` in MGS (contains the dot) | `27.44 s` | `47.74 s` | `47.57 s` | `47.87 s` | `170.62 s` | **87.1 %** |
| — `joint_inner_product` inside it | `8.98 s` | `15.32 s` | `15.26 s` | `15.32 s` | `54.88 s` | `28.0 %` |
| per-step basis SVD | `1.48 s` | `3.74 s` | `3.70 s` | `3.82 s` | `12.74 s` | `6.5 %` |

Solve wall clock `196.0 s` (`918` Arnoldi steps; probability
`9.79979792832502916e-1`, the `aarch64` record's `C = 26` value). A second run,
with compilation on the same machine, reproduced the shares within `3 %` and the
wall clock at `202.1 s`. (The same probe measured `104.0 s` after §5.9 and
`22.5 s` after §5.10, for the same steps and the same probability.)

`x86_64`, current solver with the AVX2 `dot` and `axpy` (the `norm` kernel,
§5.12, postdates this table) — the same cell on this host
(`190`, `241`, `241`, `243` steps, `915` total, `13.586 s` solve):

| phase | 190-step | 241-step | 241-step | 243-step | total | share of the solve |
|---|---|---|---|---|---|---|
| action `gemv` | `1.76 s` | `2.24 s` | `2.23 s` | `2.23 s` | `8.46 s` | **62.3 %** |
| MGS `axpy` (contains the dot) | `0.19 s` | `0.31 s` | `0.31 s` | `0.32 s` | `1.13 s` | `8.3 %` |
| — the inner product inside it | `0.07 s` | `0.12 s` | `0.12 s` | `0.13 s` | `0.44 s` | `3.2 %` |
| Givens least squares | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.1 %` |

The probability is `9.79979792832510244e-1`, within `4e-17` of the checked-in
`x86_64` record's `C = 26` value; the solve is `1.39×` the `aarch64` `9.77 s`
after §5.11, and the `gemv` is `1.82×` its `4.65 s`, close to the per-step ratio
the bench in §5.12 reports.

Under the map walk the per-call cost grew roughly with the label count, not
faster (`aarch64`): `axpy` `0.76–0.81 ms` against `0.26–0.29 ms` at `C = 20`,
`dot` `0.25–0.26 ms` against `0.08–0.09 ms`, for `2.38×` the labels. On the
current path the two representations must be kept apart: the probe's standalone
`dot_ms = 0.141 ms` / `axpy_ms = 0.196 ms` at 2380 labels are the *nested-map*
operations, while the solve's `PROFILE dot=` is `0.0024 ms` per call because it
is the AVX2 coordinate kernel over cached labels. Quote the `PROFILE` shares for
the solve's cost and the probe numbers for the map operations in isolation; the
`x86_64` probe prints the map dot at `≈ 60×` the coordinate one.

**What changed is not the solver but the cell.** The recorded `C = 26` cell was
`406.4 s` then (`104.8 s` after §5.8 and §5.9, `26.0 s` after §5.10), and the
probe then
times the other half of it — the exact dense window
reference (`WindowOperator::build` plus the Padé `exp`) that the sweep evaluated
beside every solve:

| cell at `C = 26` (`aarch64`, dense reference) | time | share of the `406.4 s` cell |
|---|---|---|
| Krylov solve (4 shifts, 918 steps) | `196.0 s` | `48 %` |
| exact reference: dense build / Padé `exp` | `1.73 s` / `174.98 s` | `43 %` |
| not individually timed (window, seed, residual bookkeeping, run variance) | `≈ 34 s` | `8 %` |

With the seed column the split is the same two halves but the reference is no
longer one of them. `x86_64`, current solver (AVX2), both halves measured in the
same probe run:

| cell at `C = 26` | time | share of the `16.805 s` sum |
|---|---|---|
| Krylov solve (4 shifts, 915 steps) | `13.586 s` | `80.8 %` |
| exact reference: seed-column build / Lanczos `exp` | `1.867 s` / `1.353 s` | `19.2 %` |

The checked-in record's cell is `19.575 s`; it predates both AVX2 kernels, and
its solve is the scalar `16.205 s` measured for that section. With the kernels
committed the same cell is `16.8 s`, which is what the regenerated record now
carries.

Under the dense reference the Padé exponential is `2380 × 2380`, so it grows like `n³`:
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
record's, the split is the probe's — and they sum to `373 s` against `406 s`
under the dense reference, and to `18.401 s` against the record's `19.575 s`
now, so the third row is a residual, not an attribution.

The parallel map operations still lose at this size, by more than at `C = 20`
(`x86_64`, current code):

| operation (`C = 26`, 2380 labels) | direct | 1 worker (chunked) | 2 workers | 4 workers | 8 workers |
|---|---|---|---|---|---|
| `joint_inner_product` | `0.149 ms` | `0.166 ms` | `0.720 ms` | `0.478 ms` | `0.451 ms` |
| `joint_axpy` | `0.255 ms` | `1.402 ms` | `4.714 ms` | `5.006 ms` | `4.711 ms` |

and the chunked dot's deviation from the serial scalar is `3.5e-15` relative
here against `1.1e-15` at `C = 20` — the association error grows with the entry
count (`2.6e-15` at `C = 28`), so the bit-identity objection gets worse, not
better, as the window widens.

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

The same `C = 28` cell with the seed column (`aarch64`; the dense column no
longer exists, so this pair is the historical comparison):

| phase | old dense reference | seed-column reference |
|---|---|---|
| window build (one action per label) | `2.26 s` | `2.28 s` |
| exponential action | `364.6 s` | `1.20 s` |
| reference total | `366.9 s` | `3.47 s` |
| return probability of the window | `9.66679088641535622e-1` | `9.66679088641579365e-1` |

The two references differ by `4.4e-15` absolute (`4.5e-15` relative) — four
orders below the `1e-3`-scale discrepancy the reference is used to certify — and
the reference is `106×` cheaper: the `C = 28` cell became `360 s` instead of
`723 s`, and what remains is the solve. Re-measured on `x86_64` with the seed
column the same cell is `2.479 s` to build plus `2.254 s` to exponentiate
(`4.732 s` total) and the window probability is
`9.66679088641579365e-1`, bit-identical to the `aarch64` column above; the whole
cell is `33.239 s` in the checked-in record, of which the reference is `14.2 %`.

The whole evidence record was then regenerated with the seed column and compared
field by field against the previous record. Every `return_probability_window_exact`
entry moved by at most `7.1e-14` absolute (at `C = 16`; most are `≤ 1e-14`), and
nothing else changed except the timings and the source revision. The record now
carries `reference_tier` saying which implementation produced the column. The
cells came down with it (`aarch64`) — the `m = 2` cells are `21 ms, 0.12 s,
0.49 s, 2.5 s, 14.0 s, 29.9 s, 58.8 s, 104.8 s` (the `C = 26` cell was `406.4 s`)
and the `m = 4` cells `34 ms … 187.5 s` (was `46 ms … 534.3 s`), peak RSS
`0.69 GB` (was `2.1 GB`), and the full run `18.5 min` (was `30.4 min`). The
checked-in `x86_64` record's cells are §5.2's rows (`22 ms` … `1114 s` at
`m = 2`). §5.9 then halves those
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

`aarch64` microbenchmark of the map operation that was replaced (the code that
produced the `per-entry search` column no longer exists):

| labels | per-entry search | ordered walk |
|---|---|---|
| 210 | `0.075 ms` | `0.005 ms` (**15.9×**) |
| 715 | `0.313 ms` | `0.015 ms` (**20.8×**) |
| 1365 | `0.675 ms` | `0.032 ms` (**21.4×**, `23 ns`/entry) |

End to end, with the same step counts and bit-identical probabilities
(`aarch64`, before the coordinate solver of §5.10):

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
four shift solves, `aarch64`):

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
| norm | `hypot(re).hypot(im)` folded in label order | the same fold over the stored entries (`coords_norm`) — except on `x86_64`, where the AVX2 kernel (§5.12) sums `re²+im²` and takes one `sqrt`, so the two paths now differ by a few ulp there |

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
column reference and the residual bookkeeping; the `aarch64` cells of that
section):

| cell | map walk (§5.9) | coordinates | speedup | probability (both) |
|---|---|---|---|---|
| `C = 26`, `m = 2` | `104.8 s` | `26.0 s` | **4.0×** | `9.79979792832502916e-1` |
| `C = 28`, `m = 2` | `188.7 s` | `44.8 s` | **4.2×** | `9.79988805406555197e-1` |
| `C = 26`, `m = 4` | `187.5 s` | `42.8 s` | **4.4×** | — (rank-4 error unchanged) |
| `C = 28`, `m = 4` | `334.7 s` | `75.5 s` | **4.4×** | — |
| whole record | `18.7 min` | `4.5 min` | **4.1×** | all 18 rows identical then |

Those are the pre-§5.11 cells. The checked-in `x86_64` record's coordinate cells
are `17.654 s` (`C = 26`, `m = 2`), `29.107 s` (`C = 28`, `m = 2`), `29.888 s`
(`C = 26`, `m = 4`) and `54.283 s` (`C = 28`, `m = 4`) — the same cutoffs, on the
AVX2 kernels that generated the record, after §5.11 and §5.12 had also
removed the SVD.

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
the profile's print resolution (`aarch64`):

| `C = 26` phase | SVD (§5.10) | Givens | share of the `9.8 s` solve |
|---|---|---|---|
| action `gemv` | `4.66 s` | `4.65 s` | `48 %` |
| MGS `axpy` | `2.23 s` | `2.21 s` | `23 %` |
| — the inner product inside it | `1.65 s` | `1.63 s` | `17 %` |
| per-step least squares | `12.66 s` | `< 0.02 s` | **`< 0.2 %`** |

Re-measured on `x86_64` with the same Givens stage and the AVX2 `dot` and `axpy`
(the `norm` kernel came later, §5.12), that
cell's solve is `13.586 s`: `62.3 %` `gemv`, `8.3 %` MGS `axpy`, `3.2 %` inner
product, `< 0.1 %` least squares. The `gemv` is `1.82×` the `aarch64` figure and
the SIMD share is far smaller because its kernels are committed here (the dot
reassociates, the axpy does not); the shares move and the structure does not.

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
(`SWEEP_CUTOFFS`, `SWEEP_COMPONENT_BUDGET = 2.5e7`). These are the
`aarch64` cells measured when this change landed; the checked-in record's rows
are in §5.2:

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
`C = 32` (`PROBE_CUTOFF=32`, budget `2.5e7`), four shift solves, `1328` Arnoldi
steps in total, on this `x86_64` host at load average `0.5` on 12 cores. This is
the host of the checked-in record; it dispatches both map operations to **AVX2**
kernels rather than the `aarch64` NEON kernels below, and its per-step `gemv`
(`38.44 ms`) sits inside the spread of the standalone bench's
`41.301 ms/step`:

| phase | 266 | 350 | 352 | 360 | total | share of the `67.80 s` solve |
|---|---|---|---|---|---|---|
| action `gemv` | `10.23 s` | `13.46 s` | `13.54 s` | `13.81 s` | `51.04 s` | **`75.3 %`** |
| MGS `axpy` | `0.88 s` | `1.43 s` | `1.47 s` | `1.43 s` | `5.21 s` | `7.7 %` |
| — the inner product inside it | `0.36 s` | `0.62 s` | `0.63 s` | `0.65 s` | `2.26 s` | `3.3 %` |
| Givens least squares | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.01 s` | `< 0.02 %` |

**Every row here is a clock from this run**, on an idle host (load average
`0.5` when it started; the same cell is `67.80–70.62 s` across two runs of this
build and `74.83–75.43 s` with the `axpy` dispatch removed, see the A/B below). The `aarch64`/NEON run measured the same cell at
`39.383 s` with `20.98 ms/step`, `70.7 %` `gemv` and `3.5 %` inner products.
The two host tables are separate measurements, not a normalisation of one
another: this host's `gemv` is `2.0×` the NEON host's (`41.0` against
`20.98 ms/step`, `9.2` against `17.9 GB/s`) and its AVX2 inner product `1.7×`
(`2.36` against `1.38 s`), which together account for the `1.9×` solve
difference (`74.8` against `39.4 s`).
The Arnoldi counts (`266/350/352/360` against `266/351/351/359`) and the
probability (`9.79994613340702814e-1` against `9.79994613340724019e-1`) agree;
the one-step differences are the known last-place sensitivity of the `1e-13`
stop test, not a different algorithm.

Read the table as nested, not additive: the inner-product row sits inside the
MGS block and the block's clock contains it (§5.12 note below). Against the
`67.80 s` solve the disjoint phases account for `56.25 s`; the remaining
`11.55 s` (`17.0 %`) is everything outside them — the `coords`/`from_coords`
round-trip at the two ends of each solve (`1.66 ms` per clone at this cutoff),
each step's image construction, and the norm of every new basis vector. The
`gemv` is `38.44 ms` per step: `H_proj` is `4845²` complex entries,
`375.6 MB`, streamed once per Arnoldi step. That is `9.8 GB/s`, so it is
bandwidth-bound, not arithmetic-bound — the flop count (`~9.4e7` per step) would
finish an order of magnitude sooner.

**What removing it would cost, measured.** There is exactly one way to keep the
window solve and not multiply by `n²` per step, and it is to go back to the
sparse nested action: the same probe times one `d_gamma` on the same dense
window state at `287.986 ms`, **`7.5×`** the `gemv`'s `38.44 ms`.
Substituted into the same `1328` steps that is `~382 s` of action instead of
`51.0 s`, i.e. a `~399 s` solve and a `~416 s` cell instead of the probe's own
`67.8 s` and `84.9 s` (reference `17.128 s` included; the previous fixture's
cell, on the scalar path, is `96.7 s`). That is a floor, not an estimate in the other direction:
the map route also pays a reserialisation per step (`map_clone_ms = 1.6594`)
that the coordinate loop does not. So the
`gemv` is not a tax on the solve — it is the *cheap* action, already `7×` below
the sparse one, and it reaches `75 %` because §5.11 made everything else
free. Removing the `gemv` leaves a `16.8 s` solve; removing every *other* term
leaves `51.0 s`. Either way the `gemv` is where a wider window pays.

One accounting caveat on the `PROFILE` line, because it is easy to misread the
split: the `axpy=` figure is the **whole Gram–Schmidt block**, not the axpys
alone — its clock starts before the two passes and stops after them, so it
contains the `dot=` figure. The disjoint split of the `67.80 s` solve is
therefore `75.3 %` `gemv`, `3.3 %` inner products, `4.4 %` real axpys (`2.95 s`)
and `17.0 %` everything else (norm, image construction, the Givens stage).

**The kernel comparison, and what the SIMD rows buy.** On the `aarch64`
campaign host the pre-NEON table read `61.9 %` `gemv`, `21.0 %` Gram–Schmidt
(which contains the inner products), `15.6 %` inner products and `17.0 %` other
on a `44.9 s` solve; after the NEON kernel it read `70.7 %` / `9.5 %` / `3.5 %` /
`19.8 %` on `39.4 s`. Both `aarch64` quiet runs agree on the three phases the
kernel does not touch — the `gemv` (`27.79 s` before the kernel, `27.84 s`
after), the real
axpys (`2.44 s` against `2.35 s`) and everything else (`7.63 s` against
`7.81 s`) — so the `5.5 s` between those solves **is** the inner product:
`6.99 s → 1.38 s` (`5.07×`, on `1327` steps rather than `1328`).

The `x86_64` counterparts are the AVX2 `dot` (then) and `axpy`, measured the same
way. The inner product first: the `C = 26` in-solve dot goes `1.39 s → 0.50 s` and the
solve `16.205 s → 15.027 s` (`−7.3 %`), and at `C = 32` the dot `6.00 s →
2.36 s` and the solve `79.35 s → 74.81 s` (`−5.7 %`). The microbenchmark
(`examples/phi4_kernel_simd_probe.rs`) reports `2.18–2.76×` across `n = 2380`
and `4845` over two runs each — the range is the scalar reference's, not the
kernel's — with agreement to `7.3e-15` and `3.2e-15` relative. It helps less
than NEON's `3.75×` because LLVM already auto-vectorises this scalar loop on
`x86_64` (`2.98–3.44 us` against `1.25 us` in the probe), whereas the `aarch64`
scalar reference is not vectorised at all. The reassociation moves the solve's
last bits — the `C = 20` probability by `3.7e-15` and the `C = 32` one in its
17th digit — which is why the record's digits are the scalar path's. The
`aarch64` campaign record agreed independently before it was replaced: that
kernel took the `C = 32` rank-2 cell from `55.0 s` to `48.9 s` (`−11.1 %`).

The `axpy` came next, and it is the better of the two because it reassociates
nothing: unfused multiply and add in the scalar loop's order give a
**bit-identical** result (`max |diff| 0.000e0` in the probe), with the same
zero-skip and the same in-loop non-finiteness report — any four-element block
holding a zero `b` entry, or one whose result would be non-finite, falls back to
the scalar loop for those elements. It is `2.65–2.76×` at `n = 2380` and
`2.43–2.47×` at `4845` (`6.79–7.01 us → 2.54–2.56 us`, `13.71–14.56 us →
5.56–5.98 us`).

The in-solve A/B is a link swap and nothing else: same binary otherwise, same
steps, same probability to the last printed digit (`9.79994613340702814e-1` at
`C = 32`, `9.79979792832510244e-1` at `26`, `9.79988805406569519e-1` at `28`),
which the bit-identity guarantees. The Gram–Schmidt block, which is the only
place the kernel runs, moves as follows:

| cutoff | steps | MGS block, scalar `axpy` | MGS block, AVX2 `axpy` | ratio | solve, scalar | solve, AVX2 |
|---|---|---|---|---|---|---|
| `C = 20` | `527` | `0.23 s` | `0.15 s` | `1.53×` | — | `2.41–2.67 s` |
| `C = 26` | `915` | `1.74 s` | `1.13 s` | `1.54×` | `14.02–16.02 s` | `13.59–14.54 s` |
| `C = 28` | `1061` | `3.25 s` | `2.13 s` | `1.53×` | `24.92–28.59 s` | `23.80–26.55 s` |
| `C = 32` | `1328` | `8.03–8.05 s` | `5.18–5.30 s` | `1.52–1.55×` | `74.83–75.43 s` | `67.80–70.62 s` |

The block is a clean `1.5×` at every size, and it is also the honest ceiling on
the solve gain: `2.8 s` of a `1328`-step solve at `C = 32`, `0.6 s` at `C = 26`.
The solve ranges above are two runs each (three at `C = 26`) on a 12-core host
whose run-to-run spread at this size is `±5 %`, so at `C = 26` and `C = 28` the
two arms overlap and only the block separates them; at `C = 32` the arms do not
overlap (`75.43/74.83` against `70.62/67.80`), but the wider gap is partly a
slower `gemv` in the scalar runs (`54.9–55.2 s` against `51.0–52.1 s`) and no
part of it is claimed as an `axpy` effect. Because the kernel is exact, these
solves differ from the scalar ones only through the inner product that was
already in the tree: the `C = 32` probability moves in its 16th significant
digit (`0.9799946133407069 → 0.9799946133407028`), which is the AVX2 `dot`'s
reassociation and nothing else. So the record regenerated below on this path is
an arithmetic match of the previous fixture up to that kernel, not a
re-baselining.

The `norm` came last. Its scalar reference is a `hypot` fold — two libm calls per
entry, which is why the fold is the most expensive of the three scalar loops,
far above `dot` and `axpy` — so replacing it with a `re²+im²` FMA sum plus one
`sqrt` is not a marginal win at the kernel level: `85.4–86.8 us → 0.89 us`
(`95–97×`) at `n = 2380` and `174.1–174.3 us → 2.08–2.19 us` (`80–84×`) at
`n = 4845` (`examples/phi4_kernel_simd_probe.rs`, two runs each). A third run on
a loaded host moved the scalar reference to `123.8 us` against `1.37 us`
(`90.5×`), so read the range as `90–97×` at that size; the kernel's own clock is
stable and the scalar `hypot` fold is what varies. It is
nevertheless the smallest solve-level change of the three, and this note says so
rather than quoting the `95×`: `coords_norm` is called **once per Arnoldi step**
over one vector, where the inner product is called `k+1` times per step over two
vectors. At `C = 26` the whole norm cost is `174 us × 916 steps = 0.16 s` of a
`13.6 s` solve (`1.2 %`), so the end-to-end A/B is inside this host's `±5 %`:
three runs each give `13.544`, `13.838`, `14.663 s` with the kernel against
`13.772`, `14.633`, `13.875 s` without, and at `C = 20` `2.730 s` against
`2.723 s`. `1.2 %` is the ceiling the call count allows; the runs neither
confirm nor refute it.

The kernel reassociates, as the inner product does and the `axpy` does not: a
`hypot` fold's rounding depends on the order of its arguments, so no vector form
reproduces the last bits. Agreement is `1.8e-15` relative at `n = 2380` and
exact at `n = 4845` on the probe's inputs, and the solve moves as the `dot`
kernel did — at `C = 26` the return probability goes `9.79979792832510244e-1 →
9.79979792832513574e-1` with `915 → 916` steps, i.e. the last Arnoldi step is
decided the other way. Because `hypot`'s scaling is what keeps the fold safe at
the ends of the `f64` range, the kernel tracks the largest component it sees and
hands any slice outside `[1e-150, 1e150]` — or any slice whose squared sum is not
finite — to `coords_norm_scalar`, so extreme amplitudes keep the fold's verdict
and its accuracy. The runtime gate is the inner product's (`avx2` **and** `fma`),
and only the `x86_64` path has a `norm` kernel; `aarch64` keeps the fold and the
record built there was untouched by this change.

**The NEON arm can now be run without an `aarch64` host — but not timed.**
`scripts/check-phi4-neon-probe` installs the `aarch64-unknown-linux-musl`
target, builds `examples/phi4_kernel_simd_probe.rs` as a one-file crate (a
symlink back to the committed example, `num-complex` its only dependency) with
`rust-lld`'s self-contained linking, and runs it under `qemu-aarch64-static`. It
is built that way rather than through `-p fock_sirk` because fock_sirk's
dependency graph emits host `libc` search paths, which made the aarch64 link
drag in the x86_64 `libc.a`; the probe needs nothing from the crate. The
result: the NEON modules **execute** and the probe's own agreement lines are
produced (`dot` `1.85e-14` relative, `axpy` `2.24e-16` absolute at `n = 2380` —
the probe's NEON `axpy` uses `vfmaq`, so unlike the committed AVX2 `axpy` it is
fused and reassociates). Those are the only numbers this route may be quoted
for. qemu is not a performance measurement: the same run reports `dot` at
`0.51×` and `axpy` at `1.46×` against the scalar loop, where the real `aarch64`
host measures `dot` at `3.75×` in-solve. Emulation cost moves with the shape of
the emulated instruction stream, so a ratio taken under qemu says nothing about
hardware — §5.12's NEON speedups stand on the direct-host runs, and the
docstring in the script says the same. What is still missing is a real
`aarch64` host to re-run the end-to-end A/B on the current tree.

Three other removals were considered and are reported as *not* done. Storing
`H_proj` in `f32` halves the stream to `188 MB` (`~20.4 ms` per step at this
host's measured `9.2 GB/s`, a `~27 s` solve), but `f32` rounding is `~1e-7`
relative against a `1e-10` solve target,
so the recorded residual would stop describing the arithmetic. A device-resident
`gemv` is not measured: the loop's `axpy`s, inner products and vectors would all
have to move with the matrix, and `fock_sirk`'s Krylov loop is CPU `nalgebra`
today, so no claim is made that it would win. And there is no sparsity to
exploit: `H_proj = W† H_raw W` is dense in a `4845`-dimensional window basis by
construction; dropping the projection *is* the map walk, i.e. the `7.03×` above.

**Six more cutoffs.** With the per-step cost known, the sweep now runs `C = 34`,
`36`, `38`, `40`, `42` and `44` (`SWEEP_CUTOFFS`, `SWEEP_COMPONENT_BUDGET =
3.0e8`). These are the `aarch64` cells of that campaign, from the quiet run of
§5.12; the record's current rows are in §5.2:

| `C` | labels | tail | shift-solve steps (`m = 2`) | `m = 2` cell | `m = 4` cell |
|---|---|---|---|---|---|
| 32 | 4845 | `0.0802` | `[266, 351]` | `49.1 s` | `83.7 s` |
| 34 | 5985 | `0.0625` | `[289, 385]` | `76.2 s` | `131.2 s` |
| 36 | 7315 | `0.0495` | `[315, 420]` | `116.7 s` | `202.0 s` |
| 38 | 8855 | `0.0397` | `[340, 459]` | `182.2 s` | `308.2 s` |
| 40 | 10626 | `0.0321` | `[367, 498]` | `266.2 s` | `464.3 s` |
| 42 | 12650 | `0.0259` | `[393, 535]` | `391.7 s` | `685.9 s` |
| 44 | 14950 | `0.0208` | `[420, 575]` | `573.3 s` | `1004.4 s` |

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
`C = 44` cell also set that run's memory: `4.20 GB` peak RSS, up from
`2.29 GB` at `C = 40` and `1.19 GB` at `C = 36`.

**These cell times were wall clock on a loaded host; a sixth attempt finally
replaced them.** That `aarch64` extension run shared the
12-core host with other repositories' jobs and sampled a load average of `9.6`
at the end (`30.7` at peak), so the `s` columns moved by up to `±12 %` against
the previous quiet run in both directions (`C = 32`'s `48.9 s → 54.5 s`,
`C = 40`'s `535 s → 470 s`). Two full reruns were then made to replace that
guessed band with a measured one. This host did not offer the ~`80 min` window
a sweep needs: the first rerun completed in `80.1 min` (`4535 s` of CPU) with
the load unsampled, the second in `1:23:49` (`4755 s` of CPU) with the load
sampled every `30 s` — median `6.0`, minimum `1.7`, maximum `10.9`, `139` of
its `168` samples above `4.0`. So the reruns are measurements of the same
contended host, not of a quiet one, and they say how far that table can be
from a quiet figure: `19` of the `34` cells reproduce within `3 %` and none is
further than `12 %` away. The pattern is the load, not the code — the first
rerun started while the host was still busy and its small `m = 2` windows
(`C ≤ 20`) came out `5–10 %` slower, while the three `m = 2` cells at `C = 32`,
`34`, `36` and the `C = 44` `m = 4` cell came out `4–12 %` **faster** in both
reruns (`C = 44` at `m = 4`: `1183 s` recorded, `1036 s` and `1088 s` rerun);
the load-sampled rerun, run under the heavier median (`6.0`), is the one that
put the `C = 38 … 42` `m = 4` cells `8–11 %` above the `aarch64` table. That
table keeps the slowest of the three runs on purpose: with no quiet window
available, a conservative figure with a measured band is worth more than a
fast one that cannot be reproduced, so the `s` columns stood as upper bounds
**by at most `12 %`**, that ceiling being measured here. Six later attempts
were made specifically to lift them and only the last produced a usable run.
This is a 12-core `aarch64` box shared with other agents, so each was gated on
a five-minute quiet window and abandoned if the host was taken mid-run:

| attempt | gate | ran for | killed at |
|---|---|---:|---:|
| 1 | `load1 < 1.2` to start | `45 min` | `load1 15–16` |
| 2 | `load1 < 1.2` to start | — | no window in the following hour (`10–12`) |
| 3 | `load1 < 2.0`, abort `≥ 3.0` | `5 min` | `3.41` |
| 4 | `load1 < 2.5`, abort `≥ 6.0` | `72 min` | `9.91` |
| 5 | `load1 < 2.5`, abort `≥ 6.0` | `27 min` | `6.74` |
| 6 | `load1 < 2.5`, abort `≥ 6.0` | `79.0 min` | **clean** |

Attempt 4 is the near miss: it started at `load1 0.53` and held `2.3–3.7` for
`72` of the `~85 min` before a burst took the box to `9.91`. Its cell times were
discarded rather than recorded. Attempt 6 started at `load1 0.32` and held a
mean of `1.45` over `316` fifteen-second samples, with a single sample at
`6.39` — one transient, short of the three-sample abort rule — and finished the
whole sweep in `79.0 min` at `4.24 GB` peak RSS. The table above now carries
**that** run's cell times. Against the upper bounds it replaces, `9` of the
`14` cells move by less than `3 %` (`C = 36` and `38` at `m = 4` agree to
`0.0 %` and `0.1 %`), and the five that move more are all faster except one:
`C = 32`, `34`, `36` at `m = 2` are `9.9 %`, `7.0 %`, `4.3 %` lower (the three
cells the loaded rerun had put `5–10 %` above), `C = 42` and `44` at `m = 4`
are `4.3 %` and `15.1 %` lower (`717 → 686 s`, `1183 → 1004 s`), and `C = 38`
at `m = 2` is `2.4 %` *higher* (`178 → 182.2 s`). So the `12 %` ceiling was a
real measurement rather than a conservative one: it under-predicted the
`C = 44` `m = 4` cell by another three points and over-predicted the small
`m = 2` windows. The step counts (`[266, 351] … [420, 575]`) and every tail in
that table are unchanged by the replacement — this is a timing run, not a
re-measurement of the physics.

**A burst now costs one cell, not the run.** Attempts `4` and `5` lost `72` and
`27` minutes of already-measured cells. The example therefore takes a per-cell
plan — `PHI4_LIST_CELLS=1` prints its `44` cell ids, `PHI4_CELLS` selects the
ones to run, `PHI4_MERGE` supplies the rest from an earlier record and
`PHI4_OUT` keeps a partial record out of the checked-in fixture — and
`scripts/sweep-phi4-cells` drives it one cell per invocation. A killed cell is
dropped, every finished cell is already merged, and the next run resumes at the
first cell the record's own `measured_cells` list does not hold. The timing
contract is unchanged: five idle minutes before each cell, abandoned if `load1`
stays at or above `6.0` for three samples. Cells the invocation did not run are
written as `not run in this invocation` rather than omitted, so a partial record
says what it is. The
ratios between neighbouring rows and every physics leaf are unaffected — all
three runs reproduce the step counts and every tail, error and probability
exactly, and the reruns differ only in `runtime_ms` and in the
`amplitude_defect` fields, which the current tree rounds to twelve significant
digits (`6f0d9c6`) while that `aarch64` run still stored them at full precision
(the checked-in `x86_64` record is the first rounded one — provenance note
below).

**The other target's build, leaf by leaf.** The suite prints its diagnostics with
`{:?}`, so a cross build can be compared leaf for leaf instead of through the
`1e-12` tolerance of the record assertion. `scripts/check-phi4-cross-target`
builds the suite for a named target and runs it under that target's `qemu`: with
`PHI4_TARGET=x86_64-unknown-linux-gnu` that is the scalar loop, with
`PHI4_TARGET=aarch64-unknown-linux-gnu` the `NEON` kernel, so the same script
covers both implementations from either host (CI runs both legs);
`scripts/compare-phi4-leaves` diffs its `29` printed leaves against this host's
`aarch64` run and against the record's `19` stored ones:

| leaves | count | max abs `|aarch64 − x86|` | bit-identical |
|---|---:|---:|---:|
| response probabilities `base`, `coupled`, `spread` | 3 | `8.1e-16` | 0 |
| response differences `lambda`, `width_k` | 2 | `0` | 2 |
| certified charges (`*_charge`) | 2 | `9.0e-16` | 0 |
| tails at `C = 4`, `8`, `12` | 3 | `2.4e-15` | 0 |
| `m = 4` inside-window errors | 3 | `3.1e-15` | 0 |
| `m = 4` forward-product errors | 3 | `4.4e-16` | 0 |
| rank `m = 2, 3, 4` errors | 3 | `2.9e-15` | 0 |
| rank `m = 2, 3, 4` forward errors | 3 | `2.1e-15` | 0 |
| `exact` reference probability | 1 | `0` | 1 |
| shift-solve counts | 6 | `0` (5 of `6`) | 5 |

The one non-float difference is the `C = 8` rank-2 Krylov count: `19` on
`x86_64`, `18` on `aarch64`. That is a stop test sitting on a rounding
boundary, not a wrong answer — the tails it produces differ by `2.0e-15` — and
it is traced rather than inferred. `SIRK_TRACE_SHIFT` printed every gate of that
cell on both builds: the second shift solve accepts at step `18` on `aarch64`
(`8.79e-14` against the `1e-13` tolerance) and at step `19` on `x86_64`
(`1.45e-13` at step `18`, `1.11e-14` at `19`). Both sequences are still falling
by two orders of magnitude per step when they cross, and the iterates have
already diverged (`8.3e-11` against `9.2e-11` at step `17`), so the crossing
moves by one step while the answer does not. The tolerance is not the knob the
flip needs: these solves bottom out at `~1.1e-14` on both builds, so tightening
`1e-13` toward `1e-14` moves the gate *into* the stagnation band and starts
hitting the `64`-step cap; the stall-aware acceptance reported in §5.12 now
reports that instead of spending the cap. The stop
test is left alone, and the suite states the right contract already:
`window_solve_matches_the_map_walk_at_solver_precision` compares step counts
with `abs_diff <= 1` ("the least-squares rounding decides the last step"). Against the
record's stored `12`-digit leaves the `x86_64` build matches `16/19` and the
`aarch64` build `13/19`; the three neither reaches are `m4_err_C12`
(`0.00242965408903` recorded, `…902` from both builds), `m4_fwd_C8` and
`m4_fwd_C12`. So the record's last digit is not a cross-build reproducible
quantity even on its own target, and no leaf moves by more than `3.1e-15`
absolute or `7e-12` relative (on leaves above `1e-4`; the two `~1e-8` forward
errors differ by `2.2e-16` absolute, which reads as `1e-8` relative because the
value itself is tiny). Two caveats: this compares two builds of the same
compiler family, so it says nothing about a different `rustc`; and every leaf on
this table is physics that both builds get right — the differences are in the
last two of fifteen or sixteen significant digits.

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
or moving the loop to a device. Both were taken: NEON on `aarch64` (below) and
AVX2 on `x86_64` (the profile above).

**An explicit NEON kernel: `2.5×` on the axpy, `3.75×` on the inner product,
`12.6 %` off the `C = 26` solve — adopted, record regenerated.** The campaign
host is `aarch64` (Qualcomm Oryon, 12 cores, NEON + `fcma`), so "explicit SIMD"
there means NEON, not AVX2; this `x86_64` host dispatches to the AVX2 kernel
instead, and its `C = 26` inner-product phase is the `0.50 s` / `3.3 %` of the
profile above.
`examples/phi4_kernel_simd_probe.rs` rebuilds both kernels with
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
`aarch64` and keeps the scalar loop as the fallback everywhere else; the
`x86_64` counterpart dispatches to AVX2 on the same terms. The two objections
that stood in the way were resolved rather than argued down. The record was
regenerated on the `aarch64` host: the `C = 26` rank-2 cell went
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
not to the bit. At the time of that comparison the frozen record was still the
`aarch64` run; it has since been regenerated natively on `x86_64` (provenance
at the end of §5.12), and each record's `source_revision` pins the commit that
produced it. The prototype stays as
`examples/phi4_kernel_simd_probe.rs` so the number remains re-measurable against
the fallback.

**The device `gemv` is now runnable, and the host still cannot run it.**
`examples/phi4_gemv_device_bench.rs` builds the real `C = 32` window operator
and times the same dense product both ways; its CPU leg reproduces the solve's
per-step figure (`41.301 ms/step` on `x86_64` against the profile's
`40.96–41.27 ms` across runs; the `aarch64` host ran the same leg at `20.868 ms`
against
`20.98 ms`), which is what makes the comparison meaningful rather than a proxy.
On this host the device leg stops after the CPU leg and prints the project's own
triage line:

```text
cutoff=32 labels=4845 matrix_mb=375.6 reps=5
cpu gemv: 41.301 ms/step (checksum 3.050567e0)
UK-GPU-OTHER → see the candle error below (…) (candle: the candle crate has not been built with cuda support)
device leg not run: no usable CUDA device. Rebuild with `--features cuda` on a host whose driver lists a GPU (`nvidia-smi`); the triage line above names the fix when one is present but unusable.
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
arithmetic only: the step streams `375.6 MB` in `41.3 ms`, a device-resident
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

**The extension to `C = 44` adds rows and changes no existing cell.** That
`aarch64` regenerated record had `17 × 2` cells in `4978 s` (`83.0 min`) at
`4.20 GB` peak RSS; `2865 s` (`47.8 min`) of that is the four new cells (`391 s`, `717 s`,
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
`15 × 2` record, with no physics difference). So those cell times were upper
bounds — by at most `12 %`, once two reruns measured the spread (§5.12). The
sixth attempt there has since replaced all `14` of them with a quiet run
(`49.1 s` against `54.5 s` at `C = 32`, `1004 s` against `1183 s` at `C = 44`);
compare ratios inside the record, not against another machine.

**Record provenance: the checked-in record is the native `x86_64` AVX2 run.** The
fixture was regenerated at `2ea6356`, the commit that carries the AVX2 `dot` and
`axpy` kernels, on the `x86_64` host of this note — so `coords_inner_product`
and `coords_axpy` both took their AVX2 path and the record's digits are that
path's, not the scalar fallback's. The `norm` kernel (§5.12) arrived afterwards,
so the record does **not** carry it: the live `C = 26` rank-2 solve now returns
`9.79979792832513574e-1` against the recorded `9.79979792832510244e-1`
(`3.3e-15`, one Arnoldi step later), and no other digit is expected to move by
more than that order. Neither value is quoted as exact — the reassociation was
already `1.2e-14` wide across the record's own sweep — but a rerun of the
generator would now differ from the checked-in file at that level, so the record
is *pre-norm* provenance: regenerate it (about `151 min`) before quoting a digit
as the live solver's. All `17 × 2` cells completed in `9057523 ms`
(`151.0 min`) at `4239804 kB` peak RSS, against `9737331 ms` (`162.3 min`) and
`4239900 kB` for the scalar-path record it replaces. The physics reproduces
that record leaf for leaf: all `42` `return_probability_window_exact` values are
byte-identical, the `return_probability_sirk` values move by at most `1.2e-14`
absolute (`C = 40`), the tails by at most `1.0e-11`, and seven shift-solve
entries across the sweep differ, each by exactly one. Of `2389` non-timing
leaves `472` differ, and the largest relative move anywhere (`1.4e-10`) is on a
solver *error bound* (`m = 4`, `C = 20`), not on a physics leaf. Cell times are
`0.93×` the scalar record's in total and fall most at the widest windows
(`C = 44`, `m = 4`: `2309 s → 1951 s`), the direction the `axpy` kernel's
measured `1.5×` on the Gram–Schmidt block predicts. The record keeps the
twelve-digit reporting of §2: its `responses.cells.*.amplitude_defect` fields
are stored rounded (`4.44614460857e-16` for the base `initial_leakage`). The
performance tables of §§5.5–5.12 were re-run on this host; the per-cell rows of
§5.2 are this run's, and the tables that remain `aarch64` measurements — the
map-walk era of §5.6 and §5.9, the dense reference of §5.8, the before/after
columns of §5.10 and §5.11, the NEON and A/B microbenchmarks of §5.12, and the
`C = 34 … 44` campaign — are labelled as such in place.

**The `aarch64` and `x86_64` tails differ by summation order, not by
architecture or revision.** Comparing the clean `aarch64` record (`0e85e65`)
with the checked-in `x86_64` record (`2ea6356`), all `34`
`return_probability_window_exact` values are byte-identical — window, basis,
`H_proj` and the exact projected reference are the same to the bit — while
`window_tail_norm_max` moves by `2.2e-16` at `C = 4` and `1.25e-11` at `C = 44`
(`6.0e-10` relative). Revision is excluded: the only source difference under
`fock_sirk/src` and `nested_fock_algebra/src` between those commits is the AVX2
`coords_norm`, which is `cfg(target_arch = "x86_64")` and never compiled on
`aarch64` (the `aarch64` dispatcher still resolves to the same scalar fold).
The cause was then isolated on one host at one revision: with the `aarch64`
dispatcher temporarily forcing `coords_inner_product_scalar`, the `C = 32` tail
is `0.08017007880954587` against `0.08017007881086600` with NEON and
`0.08017007880697495` on the `x86_64` AVX2 record — `1.3e-12` and `2.6e-12`
apart, so the *same-architecture* scalar-vs-NEON gap (`1.7e-11` relative) is as
large as the cross-architecture gap (`4.9e-11` relative). Four-accumulator
reassociation, not NEON's arithmetic, is the whole effect.

The tail is `‖(I − P_C) H v_k‖`, so it inherits the Krylov-vector difference at
the `1e-13` acceptance scale multiplied by the operator's scale; the tail itself
shrinks `1.79 → 0.0208` as `C` grows, which is why the same absolute error reads
as `1e-16` relative at `C = 4` and `6e-10` at `C = 44`. It is also why six cells
(`C = 8`, `20`, `32`, `36`, `38`, `40`) flip one Arnoldi step on the boundary
already traced at `C = 8`. Independent confirmation of that sensitivity:
re-solving the `C = 20` cell at `1e-13` instead of the policy's `1e-10` accepts
`3` steps later and moves its tail `8.4e-11` absolute (`2.6e-10` relative). The
quoted tail is three digits (`0.0208`), and the SIRK probability moves at most
`7.4e-14` relative.

**A solve that stalls at its rounding floor now reports that instead of
spending the cap.** The acceptance gate recomputes the true residual whenever
the Givens estimate dips below the tolerance; near the floor the estimate can
read below it while the recomputed residual cannot improve, and the old loop
then re-tested (one full action per candidate) until the budget ran out. It also
reported the *cap* as the iteration count even when a lucky breakdown had
already stopped the loop earlier. `StallWatch` in `joint_sirk.rs` now counts
consecutive acceptance checks that fail to beat the best recomputed residual;
three in a row stop the solve and return
`ShiftSolve { residual: <best>, iterations: <steps actually taken> }`. Accepted
solves are untouched (the watch only runs after a failed acceptance). Measured
on the `C = 12` cell with the probe tolerance `1e-16`: the solve reports
`residual: 1.97e-15, iterations: 48`; with the watch disabled the same residual
takes `210` steps, where lucky breakdown stops the loop (and the old code would
have reported `512`). The watch's two unit tests live in `joint_sirk.rs`, and
the suite's ten tests pass unchanged.

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
   0.080 → 0.063 → 0.049 → 0.040 → 0.032 → 0.026 → 0.021` for `C = 4, 8, 12,
   16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44`). It does not reach
   zero, and the last rows cost 1.23 s, 3.40 s, 6.1 s, 10.6 s, 17.7 s, 29.1 s,
   52.6 s, 88.9 s, 137 s, 215 s, 346 s, 508 s, 753 s and 1114 s per `m = 2`
   cell, so
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
   `C = 24`, `[190, 242]` at `C = 26`, `[266, 350]` at `C = 32`, `[315, 420]` at
   `C = 36`, `[340, 459]` at `C = 38`, `[367, 497]` at `C = 40`, `[393, 535]` at
   `C = 42` and `[420, 575]` at `C = 44`
   (`[209, 211, 214, 216]`, `[239, 242, 245, 247]`, `[419, 432, 433, 435]`,
   `[500, 514, 518, 519]`, `[539, 553, 559, 558]` and `[578, 593, 599, 601]` at
   rank 4) and needs the looser `1e-10`/1024 policy
   described in §5.2 (528 total steps at `C = 20`, 800 at `C = 24`, 916 at
   `C = 26`, 1327 at `C = 32`, 1590 at `C = 36`, 1732 at `C = 38`, 1878 at
   `C = 40`, 2018 at `C = 42`, 2170 at `C = 44`). Every count is in the record —
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
  `gemv` is `75.3 %` of the `67.80 s` solve on the `x86_64` host of the record
  (`51.04 s` of `1328` steps, `38.44 ms` per step, `9.8 GB/s` on a `375.6 MB`
  matrix), against `7.7 %` Gram–Schmidt, `3.3 %` inner products and `< 0.02 %`
  least squares; the `aarch64` NEON host measured `70.7 %` of a `39.4 s` solve
  at `20.98 ms/step` (§5.12). Removing it is only possible by
  going back to the sparse nested action, measured at `287.986 ms` per step
  (`7.5×` the `gemv`, a `~416 s` cell instead of the probe's `84.9 s`), so the
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
  orders below the plateau's own `1.05×` spread. So no cutoff in `C = 8 … 44`
  recovers the rank-4 error: it is rank-limited, and more window buys tail only,
  at `1309 s` and `1951 s` per `m = 4` cell on the checked-in `x86_64` record
  (`686 s` and `1004 s` on the `aarch64` host that measured the campaign, from
  the quiet run of §5.12; the loaded runs it replaced read `717 s` and
  `1183 s`). That campaign's cell times were upper bounds — by at most `12 %`,
  measured across two reruns rather than assumed against its previous quiet run
  — until that sixth attempt replaced them.
  The checked-in record is `17 × 2` cells in `151.0 min` and `4.24 GB` peak RSS.
* Measured and adopted: §3.4's change-of-basis control. The diagonal phase
  unitary transports seed, operator and observable together; with `θ_φ = 0` the
  one-`φ`-quantum seed is bit-identical and the momentum quadratures rotate, so
  the window matrix changes by `78.6` in Frobenius norm while the exact response
  is exactly `0` and the compressed response is `3.1e-15` inside an `8.09e-5`
  sharp budget (§5.3). The conjugation `h' Γ(U)ψ = Γ(U) h ψ` is pinned to
  `1e-12` in `nested_fock_algebra` and the record assertion is in
  `fock_sirk/tests/phi4_momentum_sirk.rs`.
* Measured: neither kernel implementation is merely asserted, each is run —
  `scripts/check-phi4-cross-target` cross-compiles `tests/phi4_momentum_sirk.rs`
  for a target other than the host's and executes it under that target's `qemu`.
  The scalar loop was checked from this `aarch64` host with
  `PHI4_TARGET=x86_64-unknown-linux-gnu` (GCC `16.2.0`, glibc `2.44`, `qemu-x86_64`
  `11.1.2`, `92.9 s`); against the checked-in native `x86_64` record all ten
  tests pass, the record assertions included (§5.12, which compares **all
  29** printed leaves rather than the `C`-sweep rows). The `aarch64` branch of
  the same script was exercised from this host at `PHI4_TARGET=aarch64-unknown-linux-gnu`
  against `qemu-aarch64` `11.1.2` (`60.2 s`, ten tests, using the host's own
  kernel under emulation — plumbing, not the missing coverage); the Ubuntu cross
  toolchain CI installs for that leg (`gcc-aarch64-linux-gnu`, sysroot
  `/usr/aarch64-linux-gnu`) cannot be reproduced here and is unverified. The first four
  GitHub Actions runs that carried the leg (runs `448`, `449`, `450` at
  `a53714f`, `6aae00f`, `be874e5`, then `451` at `c402761`) did not exercise it.
  The first three skipped it because the earlier
  `cargo test (release, workspace excluding fock_sirk)` step exited `101`; run
  `451` moved the cross-checks ahead of that suite and the x86_64 leg then
  failed with exit `69`. That is the script's own precondition, not a toolchain
  gap: it tested the bare names from the workflow (`gcc`, `qemu-x86_64`) with
  `[ -x ]`, which looks in the current directory rather than `PATH`. The script
  now resolves a bare name through `command -v` before checking it, and the
  aarch64 step carries `if: success() || failure()` so the NEON leg reports even
  when the x86_64 leg trips. The resolution fix was exercised here with bare
  names on the aarch64 leg (`gcc`, `qemu-aarch64`, ten tests passed); the `if:`
  guard is configuration and was not exercised. Run `453` (`6ea2b02`) then
  executed both legs on `ubuntu-latest`: the `x86_64` second-toolchain leg and
  the `aarch64` NEON leg both finished green — each step exits with the test
  suite's status, so all ten tests passed under `qemu-aarch64` with Ubuntu's
  `gcc-aarch64-linux-gnu` and the `/usr/aarch64-linux-gnu` sysroot. That job's
  later workspace-suite step is the same unrelated failure; the kernel
  cross-checks no longer wait on it. CI `ubuntu-latest` is
  `x86_64`, where a plain `cargo test` can only ever run the scalar loop, so it
  now runs both legs of the same script: the host target through a second
  toolchain, and `aarch64-unknown-linux-gnu` under `qemu-aarch64`, which is the
  only place the `NEON` kernel executes at all. That leg builds the cross suite
  from `gcc-aarch64-linux-gnu` and the `aarch64` rust target.
* Measured: the sweep is resumable per cell, so a load burst no longer discards
  the cells it has not reached. `fock_sirk/examples/phi4_momentum_sirk` takes a
  per-cell plan (`PHI4_LIST_CELLS`, `PHI4_CELLS`, `PHI4_MERGE`, `PHI4_OUT`) and
  `scripts/sweep-phi4-cells` runs one cell per invocation, resuming from the
  record's own `measured_cells`. Checked end to end on this host: the example
  lists `44` ids; `rank_two_cell` alone then `rank_sweep.m2` merged into the same
  record reports `measured 2 of 44`, keeps the first cell's digits and labels
  every cell it did not run `not run in this invocation`; a stubbed binary
  driven with `PHI4_BURST_MAX=0` exits `75`, drops the in-flight cell and writes
  nothing (§5.12).
* Measured across every printed leaf: `scripts/compare-phi4-leaves` diffs the
  `{:?}` leaves of the native `aarch64` run and the emulated `x86_64` run. Of
  `29` leaves `8` are bit-identical across builds (`lambda` and `width_k`
  responses, `exact`, the `m = 4` counts at all three cutoffs, `m = 2` counts at
  `C = 4` and `12`); the rest differ by at most `3.1e-15` absolute and `7e-12`
  relative on leaves above `1e-4`; **one integer** moves — the `C = 8` rank-2
  Krylov count is `19` on `x86_64` and `18` on `aarch64`, the two sides of a
  `1e-13` stop test. Against the record's stored 12-digit leaves the `x86_64`
  build reproduces `16/19` and the `aarch64` build `13/19`; the three neither
  reaches are `m4_err_C12`, `m4_fwd_C8` and `m4_fwd_C12`, so the record's own
  last digit is not a cross-build reproducible quantity even on its own target.
  The cross build therefore reproduces the checked-in *x86* arithmetic and it is
  the `aarch64` NEON kernel that sits on the other side of that boundary — a
  difference between two builds of one target, not a verdict on the target
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
