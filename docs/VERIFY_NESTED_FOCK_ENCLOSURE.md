# Nested-Fock symbolic contract and pending validation

This 2026-11-08 source audit aligns the Cadabra2 modules with the operator
contract in `../timepiece/CONSOLIDATED_PLAN.md`, entry 2026-11-08f. It does
**not** report a Cadabra run, Lean compilation or renewed numerical certificate.
Historical CHECK transcripts remain evidence for their earlier source versions.

The final physical QYM/QED/QG/NS Hamiltonian is

$$H=d\Gamma(h)=\sum_{ij}h_{ij}C^\dagger(e_i)A(e_j).$$

Keep every inner interaction inside `h`, with creation left and annihilation
right at the outer level. Form products before lifting: `dΓ(B²)` generally
differs from `dΓ(B)²`. A scalar inner shift gives `c N_out`, not a harmless
outer energy constant. Vacuum annihilation alone proves zero energy; a ground
state requires `h≥0`, and a gap above the unique vacuum requires an absolute
`h≥mI`, `m>0`, on the whole inner space. Inner zero modes give further outer
ground states. An inner excitation or reflection difference is a diagnostic.

QG means R² vielbein/TEGR with the full exponential wall and interactions.
The Jordan-frame vielbein, metric scalaron, coupled base-gravity and densitized
modules have different subjects. No shared `H_final` handle or builder name
identifies all of them. The exact scalaron wall and the full vielbein coupling
must coexist in the identified record. A Taylor model is comparison-only.
Fourier convolution for spatial derivative products belongs to NS/QG only,
inside `h`, with the symbol attached to the differentiated leg.

## Source changes and runtime status

All existing `.cdb` files now have a contract specifying their inner/chart/
comparison role. Existing extraction handles and printed markers remain.
The momentum-product comments no longer claim a physical outer pair-sector
Hamiltonian from commuting field identities. Densitized checks no longer
promote formal symmetry or frame determinant scaling to a unitary/ESA theorem.
In particular, `det(tilde e)=y³det(e)` is a frame-matrix identity, not the
Jacobian of a map of independent field coordinates. That Jacobian, its inverse,
measure, onto isometry, domains and graph-core transfer remain proof obligations.

`yang_mills_hamiltonian.cdb` adds checks 3–4 for the two-mode reflection and
its full cubic/quartic magnetic expansion. They refer to the diagnostic's inner
operator, not a physical full-spectrum gap. Execution is pending.

The new `nested_fock_enclosure.cdb` has assertion-bearing checks:

| Check | Expected | Purpose |
|---|---|---|
| 1 | 0 | Right outer annihilator kills a formal vacuum ket |
| 2 | 0 | Inner scalar shift lifts to a number term |
| 3 | 0 | CCR composition retains the two-body cross term |
| 4 | 0 | Inner composition agrees on the one-particle restriction |
| 5 | 0 | At occupation two the defect is `2 h k` |
| 6 | 2 | Negative control against false multiplicativity |
| 7 | -2 | Negative inner energy defeats vacuum-ground inference |
| 8 | 0 | Exact exponential and interaction matrix entries survive linear lifting |
| 9 | 1 | Independent probe retains the interaction entry |

This is finite formal CCR/scalar-entry algebra. The last checks do not model
noncommuting inner fields or prove the complete coupled QG Hamiltonian. None
proves positivity, ESA, an infinite-mode limit or a physical spectral gap.

## Next engine-equipped checks

From the `unfer` root, run (do not treat a missing engine as a pass):

```bash
cadabra2-cli -q -n docs/nested_fock_enclosure.cdb
cadabra2-cli -q -n docs/yang_mills_hamiltonian.cdb
cadabra2-cli -q -n docs/momentum_space_products.cdb
cadabra2-cli -q -n docs/ns_qg_fourier_elimination.cdb
cadabra2-cli -q -n docs/qg_starobinsky_vielbein_hamiltonian.cdb
```

Replay the remaining modules' CHECK rows and pinned sizes, then the appropriate
symbolic Rust extraction suites. Pin the transcript and engine version only
after a successful run. Read [the Cadabra gotchas](CADABRA2_GOTCHAS.md): preserve
`@()` names, `.copy()` semantics, integral wrappers and standalone comments on
assignment lines. A printed zero is not an operator-domain proof.

No Hamiltonian builder or numerical fixture is changed in this wave. If later
work changes an actual operator/solver/certificate, execute the full SIRK,
Aeneas, Lean-export and nanoda regeneration workflow before publishing status.
Vendored `timepiece/unfer_contracts` copies remain reference material and should
be synchronized deliberately after the engine validation, never executed by
the Lean specialist as an undocumented prerequisite.
