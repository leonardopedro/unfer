# Full SM E1 assembly and joint-state Hashimoto integration

2026-11-09c. This is the executable independent-collective-coordinate endpoint
E1 from `timepiece/SM_ESA_COMPLETION_PLAN.md`. Its inner space is
`L²(R¹⁶³) tensor CAR(96)`, and its final operator is

```text
h_SM = h_B + spatial Dirac + gauge–matter + all four Higgs/Yukawa components,
H_SM = dGamma_out(h_SM) = sum h_ij Cdag_i A_j.
```

Every inner term and the specified Higgs constant is assembled before the
single outer enclosure. This implementation supplies the numerical operator
and symbolic dictionary for the Lean specialist. It does not prove its full
ESA, physical BRST reduction, graded-statistics Hilbert/core identification, or an
infinite inner matter-mode ultraviolet limit. No Lean compilation is involved.

## Source and operator dictionary

`nested_fock_algebra/src/sm_full.rs` defines the complete parameterized
operator. `docs/sm_full_nested.cdb` derives an independent component dictionary
in Cadabra2; its actual LaTeX export is
`docs/fixtures/sm_full_cadabra.latex`. The fixture contains 40 field momenta,
49 bosonic forms and all 1,152 nonzero symbolic entries of the 96-by-96 matter
matrix: **1,241 exported component expressions**. Couplings and all complex
three-generation Yukawa entries remain symbolic until parameter binding.

| Term | Concrete representation |
|---|---|
| Bosonic coordinate | Named blocks `x; G,dG; W,dW; B,dB; phi,dphi`, totaling `3+24+72+9+27+3+9+4+12=163` |
| Field kinetic energy | `1/2 sum_active p_m²`, all 40 gauge/Higgs modes; no extra jet momenta |
| Magnetic energy | All 24 strong, 9 weak and 3 hypercharge magnetic components, squared with weight `1/2` |
| Covariant Higgs energy | All 12 real covariant derivative components, squared with weight `1/2` |
| Higgs potential | `lambda/4 (sum_a phi_a²-vev²)²`, including its fixed constant |
| Spatial Dirac derivative | `sum_j p_xj tensor A_j`, not a fixed plane-wave momentum |
| Gauge–matter interaction | Every strong/weak/hypercharge coordinate multiplied by its spin/chiral/internal CAR bilinear |
| Yukawa interaction | Four real components, all up/down/neutrino/charged-lepton matrices, all generations/colors/spins, and h.c. |

The explicit numeric bosonic labels are a named-coordinate bijection. Lean's
`smIdx` is noncomputable; equality is through named blocks, not a claim that
its internal enumeration has been evaluated. SM retains all independent jet
coordinates. The spatial derivative here acts at fixed collective coordinates;
NS/QG momentum-convolution elimination is not applied to redefine this operator.
A total-jet derivative interpretation would require its separate representation
map and extra terms; it is not silently substituted for E1.

With `D=d+i g T A`, commutation gives the manuscript's
`F=dA-dA-g f A A`. The strong constants are derived from the actual Gell-Mann
matrices in Rust and independently tabulated in Cadabra. Weak constants have
the same minus-epsilon convention. Thus the arbitrary-data Lean bosonic
formula is instantiated with the corresponding signed structure constants.
The four-real-component Higgs generators are the realifications of `i tau/2`
and `i I/2`, not arbitrary real matrices.

## Real/Majorana and complex CAR coordinates

Choose the independent complex chiral coordinates after the manuscript's
reality/chirality constraints. The inventories are:

| Multiplet | Independent complex coordinates | Real coordinates |
|---|---:|---:|
| Left quark doublet | `3 generations * 3 colors * 2 weak * 2 spin = 36` | 72 |
| Right up, right down | 18 each | 36 each |
| Left lepton doublet | 12 | 24 |
| Right neutrino, right charged lepton | 6 each | 12 each |
| Total | 96 | 192 |

The manuscript's conjugate singlets use the opposite complex structure from
the physical right-handed coordinates: `gamma0` anticommutes with `gamma5`.
Changing `J=i gamma5` to `-J` conjugates their complex matrices, converting the
antitriplet to the physical triplet and reversing the book singlet charges.
The implemented complex charges are therefore `Q_L=1/6`, `u_R=2/3`,
`d_R=-1/3`, `L_L=-1/2`, `nu_R=0`, `e_R=-1`. No extra independent
antiparticle copy or Majorana mass is appended. The actual manuscript spin frame is retained:
`alpha_L=(-sigma1,+sigma3,+sigma2)`, `alpha_R=-alpha_L`. An independent
regression starts from the fixed integer `i gamma^mu` and `i gamma5` matrices
in `ChapterA3`/`ChapterCPTHamiltonian`, transports the opposite complex
structures and singlet spin basis, and checks every kinetic block and the
mass-spin contraction. No conventional spin-frame formula is presumed.

For each complex CAR coordinate, the real Clifford coordinates are
`gamma_x=c+c†`, `gamma_y=i(c†-c)`. For a Hermitian matrix `K=A+i B`,

```text
c† K c = Tr(K)/2 + (i/4) gamma^T R gamma,
R_xx=B, R_yy=B, R_xy=A, R_yx=-A.
```

The regression applies this real antisymmetric quadratic form and the complex
CAR bilinear to the same states, retaining the trace constant. Full gauge
covariance is checked for every Higgs component, color, generation and spin.
These executable identities are numerical evidence for the dictionary;
formal representation/intertwining theorems remain P0/P1 Lean work.

The complex doublet is `H=(phi0+i phi1, phi2+i phi3)/sqrt(2)` and its partner
is `tilde H=(conj H1,-conj H0)`. The Hamiltonian includes
`Q_L† H Y_d d_R + Q_L† tilde H Y_u u_R` and the lepton analogues, plus h.c.
This makes the real manuscript `-L_Y/2` convention explicit as the full paired
Hermitian interaction after choosing independent canonical variables.
`SmParameters` supplies canonical Yukawa coefficients, not measured masses;
a mass calibration must specify its `vev/sqrt(2)` and field-normalization map.
The test instance has nonzero gauge/quartic couplings and distinct complex,
fully populated three-generation matrices; it makes no measured-mass prediction.

## Cadabra LaTeX really enters the operator

The Cadabra module checks the full bosonic Legendre expression, Hermiticity of
every matter entry, and completeness of all 96 internal coordinates, then
exports actual LaTeX through `_latex_()`. The import in
`sm_full::cadabra_import` uses the existing **mathhook** calculus crate.
Its adapter handles Cadabra's braced indices/fractions/powers and protects
multi-letter scalar parameters from implicit-multiplication splitting.

Only commuting coefficient polynomials cross this bridge. Spatial momenta
commute with the independent field/jet coordinates; same-mode `q*p` imports
are rejected. Unknown AST nodes, missing/extra components, wrong momenta and
mutated coefficients fail. The imported expressions are compared termwise
with the independent native dictionary and then used to build the numerical
operator. The unsafe legacy scalar/contraction-dropping compiler is not used.
This is a deliberate exact import path inside the existing calculus integration,
not a claim that the legacy compiler's ordering behavior has been repaired.

Cadabra remains a subprocess; no GPL engine code is linked into Rust.

## Actual rational SIRK on joint states

`fock_sirk/src/joint_sirk.rs` accepts the joint vectors directly and builds a
bijective registry of joint outer labels. Its default method constructs
sequential **resolvent** vectors using matrix-free GMRES with two-pass modified
Gram–Schmidt and measured shifted residuals. For `A=-i h`, the paper's
positive shifts `gamma_j=(N-j delta)/t`, `j=1..m`, correspond to
`z_j=i gamma_j` in `(h-z_j)^-1`. The budget requires `m<N/delta`.

Forward products `(h-z_j)w` remain an explicitly named polynomial diagnostic.
They are not substituted for the rational space or automatically assigned
Hashimoto–Nodera's Theorem 4.1 bound. Positive shifts alone do not discharge
the paper's operator-function/numerical-range or best-approximation hypotheses.

For the rational basis V, the function uses exactly the bounded-resolvent
compression `B=V†(gamma_m I+iH)^-1 V`, then
`G=gamma_m I-B^-1`, and `V exp(tG) V† v`. The separately computed
`V† H V` is used for Hamiltonian Ritz diagnostics. Substituting its exponential
would give a different method. With a declared numerical projection P, the
shifted solves use `P H P`; complete outside-window residuals report the
resulting difference from the target full resolvent.

The existing Gram-whitening routine is followed by two-pass reorthogonalization
of the actual physical sparse vectors. Projections, reconstruction and full
residuals use those vectors directly, avoiding cancellation in nearly dependent
Gram coordinates. The small exponential uses nalgebra's Padé implementation.
Rank loss is reported as numerical rank. Finite rational approximations may
have norm defects; tests measure them against independent references rather
than claim exact unitarity. Restart diagnostics retain norms, full-operator
energy expectations and all solve/function/projection error data.

A declared finite basis window can cache the complete operator images and
its compression. Internal GMRES iterations reuse that same compression while
full images are retained for residual/tail calculations. Cache-entry budgets
fail explicitly. This speeds the complete operator without removing terms.

There is no hidden amplitude threshold, top-k deletion or interaction removal.
An optional total-occupation cutoff or explicit basis window is a declared
compression of the same full operator. Each action is computed before that
projection, and its discarded norm is recorded. GMRES failures/resource limits
propagate. Full shifted residuals divided by `|Im z|` are recorded as conditional
self-adjoint resolvent error bounds, including projection error.

Ritz residuals are recomputed from the full operator images, so omitted-window
components cannot disappear behind a Gram recurrence. Finite residuals do not
exclude unseen spectrum or certify a physical mass gap. Floating-point/rounding
certification and full cutoff-removal estimates remain separate proof work.

## Reproducible commands

Cadabra2 2.5.14 was provisioned through Nix and all locked Cargo dependencies,
including `ahash`, were fetched. A clean environment can use:

```sh
cargo fetch --locked
nix-shell -p cadabra2 --run 'cadabra2-cli -q -n docs/sm_full_nested.cdb'
cargo test --locked --offline --release -p nested_fock_algebra --features latex --test exact_inner_contract --test sm_full_contract -- --test-threads=1
cargo test --locked --offline --release -p fock_sirk --test sm_full_hashimoto --test sm_gauge_fixed_validation -- --test-threads=1
```

### Host prerequisites

Three conditions are not satisfied by a bare shell on this host and must be
handled before the commands above behave as written.

**Toolchain on `PATH`.** `cargo`, `rustc` and a C linker are not installed
globally; the pinned toolchain is taken from the Nix store and prepended. The
versions match `rust-toolchain.toml`:

```sh
export PATH=/nix/store/c4svgcgjzph9cy4r1fbq5hh9g55krv5r-rustc-wrapper-1.99.0/bin:\
/nix/store/fbw7w7blbj5yvxz4fz2gc0ihm898lwgq-cargo-1.99.0/bin:\
/nix/store/6gj4dzcv6gaxh5raqyrq7jmy605qaj3z-gcc-wrapper-16.2.0/bin:$PATH
cargo --version   # cargo 1.99.0 (5f94df478 2026-08-27)
```

**One online fetch first.** `cargo fetch --locked --offline` cannot succeed on a
cold registry cache: dependencies of other workspace members (for example
`wasm-bindgen`, `ahash`) are not vendored yet. Run the online `cargo fetch
--locked` once; the `--offline` test commands then work.

**`rustfmt` is a separate package.** The cargo store path above ships no
`cargo-fmt`, and no `rustup` is installed, so formatting comes from nixpkgs at
the pinned version:

```sh
nix profile install 'nixpkgs#rustfmt'     # rustfmt-1.99.0; ~/.nix-profile/bin
cargo fmt -p fock_sirk --check           # clean
```

The binary reports `rustfmt 1.10.0`, which is the formatter's own version, not
the toolchain's. `cargo fmt --all --check` is **not** clean: 20 files in
`logos`, `mm0web`, `prob_kernel`, `unfer_measure` and `unfer_taler` predate
this and would need their own change, so a workspace-wide gate cannot be turned
on as-is.

**Cadabra2 needs its own Python.** `cadabra2-cli` is a wrapper that starts an
embedded Python. When the host has a system Python of the same major version it
resolves `/usr/lib/python3.14` instead of its bundled stdlib and aborts with
`ModuleNotFoundError: No module named '_struct'` (exit 134, core dumped). Point
it at the Nix Python rather than exporting into the caller's shell, and let
`CADABRA_CLI` name the wrapper:

```sh
cat > target/tmp/cadabra2-cli <<'EOF'
#!/usr/bin/env bash
export PYTHONHOME=/nix/store/lq2fsnmg0q3y0vlxn91w8s8wvj1m6fld-python3-3.14.8
export MPLCONFIGDIR=/tmp/mplcfg
exec /nix/store/6qy19bwzbzvxk24sa10wgdxvs6160izd-cadabra2-2.5.14-p1/bin/cadabra2-cli "$@"
EOF
chmod +x target/tmp/cadabra2-cli
export CADABRA_CLI=$PWD/target/tmp/cadabra2-cli
target/tmp/cadabra2-cli -q -n docs/sm_full_nested.cdb | cmp - docs/fixtures/sm_full_cadabra.latex
```

`nix-shell -p cadabra2` has no shell environment on this host; the same package
was realized with `nix build --no-link 'nixpkgs#cadabra2'`, a binary-cache
substitution of `cadabra2-2.5.14-p1` (no source build). The engine's version
banner prints `2.5.15 private` even at that store path; the export is still
byte-identical to the fixture.

Set `CADABRA_CLI` to the engine path for the fresh-engine/fixture equality test;
without that variable the test explicitly reports an engine-check skip. The
operator/fixture import tests still run. `scripts/check-sm-symbolic` regenerates
and checks the complete export independently of Rust compilation.


A runnable driver imports the same complete LaTeX dictionary and prints JSON
with shifts, numerical window, norms, energies, full action tails and resolvent
error data:

```sh
cargo run --locked --offline --release -p fock_sirk --example sm_full_hashimoto -- --time 0.01 --steps 1 --krylov 4 --occupation-cutoff 1
```

Its illustrative parameter instance is not a measured-mass calibration; every
physical interaction remains in the applied operator. Wider occupation cutoffs
need explicit resource budgets, because the cached window grows quadratically:
`--cache-limit` (cached operator entries, default 1,000,000),
`--action-components` (entries in one complete image, default 500,000) and
`--action-work` (accumulated image entries per action, default 500,000,000).
Exceeding any of them is an explicit error naming the requirement; the operator
is never truncated to fit.

`scripts/run-sm-full-validation` applies the host prerequisites above and runs
this whole section end to end (symbolic export and CHECK rows, both test suites,
then the JSON driver). It records the engine CLI and `PYTHONHOME` it settled on
in `target/tmp/cadabra-env` and reuses that record instead of re-probing every
store Python; delete the file, or set `SM_FULL_CADABRA_ENV`, to override it.
`SM_FULL_WIDE_CUTOFFS=1` additionally prints the cutoff 2-4 JSON tails, which
need the raised budgets above and peak near 7 GB.


## Executed validation (2026-11-09c)

Rust `1.99.0`; locked Cargo dependencies; CPU backend. All **37 tests passed**
through normal Cargo commands: 13 exact-action contracts, 7 complete dictionary/
Majorana/Cadabra contracts, 6 primary full-SM rational tests, and 11 supporting
SM algebra checks. The fresh-engine equality test ran with `CADABRA_CLI` set;
it was not skipped. No Lean compilation was performed.

The primary full-SM tests use parity-graded outer states, including a mixed
even/odd two-particle action check. The test instance uses a 70-state declared finite reference window
and independently assembled full-SM matrix columns; occupation refinement
expands the window to 230 states. All terms remain in the complete operator.
The actual rational evolution errors against that same finite reference are:

| Configuration | Measured state error |
|---|---:|
| t=0.01, m=2, shift budget rate 8 | 9.15134e-5 |
| t=0.01, m=4, shift budget rate 9 | 4.17940e-7 |
| t=0.01, m=5, shift budget rate 12 | 5.90702e-7 |
| t=0.02, m=4, four restart steps | 8.25354e-7 |

The first complete-action tail norm drops from about **9.78477** at occupation
cutoff 1 to **4.03868** at cutoff 2. These substantial tails are reported, not
hidden behind the small finite-reference errors. The run does not establish a
small error for the unbounded full-operator evolution or remove its cutoff.
Ritz and shifted residuals include those tails. The independent dense resolvent
checks the paper's function compression separately from the Ritz Hamiltonian.

The cache-enabled six-test run with graded outer states took 80.79 seconds in this environment;
this is one observed run, not a performance guarantee. No pruning was enabled,
and cache/action budget counterfactuals returned errors rather than simplify H.

## Executed validation (2026-11-09, host re-run)

A repeat of the commands in *Reproducible commands* on the same working tree,
from a shell where only the host prerequisites above were added. Rust `1.99.0`;
locked dependencies; CPU backend. All **40 tests passed**: 13 exact-action
contracts, 7 complete dictionary/Majorana/Cadabra contracts, 9 primary full-SM
rational tests and 11 supporting SM algebra checks. The fresh-engine equality
test ran with `CADABRA_CLI` set and was not skipped. No Lean compilation was
performed.

Symbolic side: `docs/sm_full_nested.cdb` regenerated the export byte-identically
(1245 lines, `cmp` clean), and `scripts/check-sm-symbolic` reported 19
`sm_gauge_fixed_hamiltonian` CHECK rows, 49 `faris_lavine_n_sm` CHECK rows, and
the full-SM inventory (163 coordinates, 40 momenta, 49 forms, 96 CAR
coordinates, 1152 matter entries) exactly.

The measured rational errors and tails reproduce the recorded values:

| Configuration | Measured state error |
|---|---:|
| t=0.01, m=2, shift budget rate 8 | 9.15134e-5 |
| t=0.01, m=4, shift budget rate 9 | 4.17940e-7 |
| t=0.01, m=5, shift budget rate 12 | 5.90702e-7 |

The m=5 row is historical: the default refinement now solves m=2 and m=4 only,
because the tighter budget measured *less* accurate (`5.90702e-7` against
`4.17940e-7`) and cost a full extra set of shifted solves.

The first complete-action tail norm closes monotonically with the declared
occupation window: **9.784770722594882** at cutoff 1 (70 states),
**4.038679200865053** at cutoff 2 (230 states), **1.2179612832516469** at cutoff
3 (638 states) and **0.0** at cutoff 4 (1176 states), where the first image of
the full operator lies entirely inside the window. Later tails remain nonzero,
as expected for a window built from that first image. `sm_full_hashimoto_budget_refinement`
asserts all four windows and both endpoints (strict decrease, then exactly
zero). The tails are now computed from the first complete action image rather
than by one costly solve per cutoff — the tail the solver reports *is* the part
of `H|psi_0>` its window omits — so the test dropped from 97.56 s (4.5 GB peak)
to **23.08 s**, with the 230-state solve kept as a cross-check of the reported
tail. Cutoffs 3 and 4 exceed the shipped
budgets, so they need
`--cache-limit 4000000 --action-components 8000000 --action-work 4000000000`
(65 s / 5.2 GB and 114 s / 7.0 GB peak RSS for the example here). With the
defaults the same runs refuse explicitly: 1,001,177 and 1,382,976 cached
entries against a 1,000,000 limit, and images over the 500,000-entry action
limit. The
restarted-evolution case at t=0.02, m=4 measured
`9.126129501984431e-7` and `8.253536719381402e-7` for 1 and 4 restart steps; the
2-step split (`8.35880401995079e-7`) is covered by the energy test. The
`sm_full_hashimoto` example at cutoff 1 exited 0
and printed the same window, energy (`51.32175249999999`), tail and a norm
defect of `3.6e-14`; at cutoff 2 it reports the 230-state window
(dimension 230, support 230) and the 4.03868 tail. The primary six-test block
took 77.84 seconds before the wider cutoff assertions and 148.84 seconds with
them. The nine-test primary block now runs in **115.07 and 118.53 s** in two runs
(`--test-threads=1`), against 329.92 and 335.74 seconds for the earlier
eight-test block (8.1 GB peak). The 936-state Ehrenfest point is the
remaining expensive case and is opt-in: `SM_FULL_WIDE_CUTOFFS=1` re-measures it.
These are observations, not performance guarantees.

### Physics invariants under the compressed flow

Three further tests ask what the SM operator does, not whether the solve is
self-consistent.

**Energy is conserved.** The exact flow is unitary, so `<H>` is fixed; the
compressed rational flow can only move it by its own truncation. At t=0.02 the
reference energy is `51.32175249999999` and the drift is
`3.2670754990249407e-11` (1 step) and `5.186961971048731e-13` (2) — roundoff, so
the assertion is the bound, not an ordering. (The 4-step split measured
`1.8417267710901797e-11` in the earlier run and is covered by the
restarted-evolution test.) Every recorded energy is real to `1e-9`.

**Canonical field velocity.** With `q = a + a†` and the documented SM momentum
`p = i(a† − a)/2`, Ehrenfest gives `d<q>/dt = <p>`. The test uses the two-universe
state `(|0> + e^{iπ/4}|1>)/√2` on `coord::g(0,0)` with the same quark/lepton
pair, whose analytic values are `<q> = cos(π/4) = 0.7071067811865475` and
`<p> = sin(π/4)/2 = 0.3535533905932737` — the second pins the momentum
normalization that `[q,p] = i` depends on, and both are asserted to `1e-12`. At
dt=1e-3 the measured velocity converges to `<p>` as the window closes:

| Window | States | Leading tail | `d<q>/dt` | Defect vs `<p>` |
|---|---:|---:|---:|---:|
| cutoff 1 | 72 | 9.9275 | 0.55751 | 2.0395e-1 |
| cutoff 2 | 325 | 7.5488 | 0.32488 | 2.8678e-2 |
| cutoff 3 | 936 | 3.0897 | 0.34923 | 4.3185e-3 |

The defect is the omitted action of the window, not solver error: it falls by
roughly a factor 7 per cutoff while the tail falls by 1.3–2.4. The default run
uses cutoffs 1 and 2 and asserts that the defect both decreases and stays under
10% of `<p>`; `SM_FULL_WIDE_CUTOFFS=1` adds cutoff 3, where it reproduces
`4.31846279135123e-3` and must stay under 1.25%.

**The BRST charge annihilates the ground state.** `sm_full` carries no ghost
sector, so the test ports the temporal-gauge charge `Q = Σ_g P_{A_g} c_g` of
`sm_brst_charge` into this representation, with the twelve ghosts on fermionic
modes `96..108` — outside the 96-mode matter block, where no `h_SM` term can
reach them. On the window ground state (`θ₀ = 49.7633053378465`) `Q|gs> = 0`
exactly. Because the repository's charge uses ghost *lowering* (the ladder
`sm_brst_charge` itself uses), that would hold of every ghost-free state, so the
test is not vacuous: `Q` acts nontrivially on the one-ghost companion,
`|Q|gs,ghost>| = √3/2 = 0.8660254037844387` (three spatial Gauss terms at
`p = i(a† − a)/2`), and `Q² = 0` on it. The printed `|P|gs>| = 0.5010457745164838`
records the scope: the Gauss generator alone does *not* annihilate this state, so
the claim is ghost-vacuum closure of the window ground state, not Gauss's law and
not a BRST reduction — the E1 dictionary has neither ghosts nor a gauge-fixing
term.
