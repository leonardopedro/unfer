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
physical interaction remains in the applied operator.


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
