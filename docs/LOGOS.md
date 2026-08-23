# Logos — CNL-to-Verified-Execution Compiler

> Controlled Natural Language compiler producing verified execution graphs
> with content-addressable (UNF hash) identity.

## Pipeline

```
sentence → harper_gate → CCG parse → CoreIR compile → linearity →
           interaction net → reduce → readback → UNF hash
```

| Stage | Module | Entry point |
|-------|--------|-------------|
| 1. Pre-parse lint | `harper_gate` | `HarperGate::lint(input) -> GateResult` |
| 2. CCG chart parse | `ccg` | `parse_sentence(tokens, lexicon) -> Vec<DerivationTree>` |
| 3. Compile to CoreIR | `core_ir` | `compile_to_core_ir(tree, lexicon) -> CoreIR` |
| 4. Linearity enforcement | `core_ir::linearity` | `insert_linearity(term) -> CoreIR` |
| 5. Interaction net | `deltanet` | `compile_to_net(ir) -> Net` |
| 6. Reduce | `deltanet::reducer` | `reduce(&mut net)` (≤1M iterations) |
| 7. Readback | `deltanet::readback` | `readback(net) -> String` |
| 8. UNF hash | `deltanet::unf` | `unf_hash(net) -> [u8; 32]` (SHA-256) |

## L0 Grammar (CNL subset)

```
Sentence    := NP VP
VP          := Vtrans NP | Vintrans | Vditrans NP NP | Copula NP
NP          := Det N | Det Adj N | ProperNoun | Numeral | Det N RelClause
RelClause   := RelPron VP
```

Vocabulary: proper nouns (John, Mary, Bob, Alice), determiners (the, a),
nouns (number, cat, dog), adjectives (big, small, red, blue + "very"),
transitive verbs (loves, sees, likes, eats), intransitive (sleeps, runs),
ditransitive (adds, multiplies, subtracts, give), copulas (is, equals,
greater, less), numerals (zero–ten), relative pronouns (that, which),
conjunction (and), negation (not).

Lexicon: `corpus/lexicon.tsv` (46 entries mapping words → CCG categories +
semantic templates).

## CCG Parser

CKY-style bottom-up chart parser. Four combinators:

- Forward Application: `X/Y + Y → X`
- Backward Application: `Y + X\Y → X`
- Forward Composition: `X/Y + Y/Z → X/Z`
- Backward Composition: `Y\Z + X\Y → X\Z`

Returns all derivation trees with root category `S`.

## CoreIR

```rust
enum CoreIR {
    Var(String),
    Lit(Literal),           // Int64(i64) | Bool(bool)
    Con(TagId, Vec<CoreIR>), // constructor (Love=1, See=2, ..., And=25)
    Lam(String, Box<CoreIR>),
    App(Box<CoreIR>, Box<CoreIR>),
    Let(String, Box<CoreIR>, Box<CoreIR>),
    Match(Box<CoreIR>, Vec<(Pattern, CoreIR)>),
    Fold(Box<CoreIR>, Box<CoreIR>, Box<CoreIR>),
    Prim(PrimOp, Vec<CoreIR>), // Add64, Sub64, Mul64, Eq64, Gt64, Lt64,
                               // AddF64, SubF64, MulF64, DivF64, EqF64,
                               // GtF64, LtF64, And, Or, Not
    Clone(Box<CoreIR>),
    Drop(Box<CoreIR>),
}
```

## Linearity Enforcement

After compilation, `insert_linearity` transforms the term so every variable
is used exactly once (linear logic):

- Variables used 0 times → explicit `Drop` node.
- Variables used N>1 times → explicit `Clone` (Dup) nodes.

`check_linearity` then verifies the invariant; any violation is a
`LinearityError` (`UsedMultipleTimes` | `Unused`).

## Interaction Net (deltanet)

Agents: `App`, `Abs`, `Con(TagId, arity)`, `Fold`, `Dup(level)`, `Era`,
`Prim(PrimOp)`, `Lit(Literal)`, `Entity(String)`.

Reduction fires on active pairs (principal-to-principal connections):

| Active pair | Rule |
|-------------|------|
| App ↔ Abs | Beta reduction (rewire argument to bound variable) |
| Dup ↔ Dup (same level) | Annihilation |
| Dup ↔ Dup (different level) | Commutation (4 new Dup nodes) |
| Dup ↔ other | Commutation through the agent |
| Era ↔ anything | Erasure (propagate Era to aux ports) |
| Fold ↔ Con(Nil) | Return init |
| Fold ↔ Con(Cons) | Unfold: `f(head, Fold(f, init, tail))` |
| Prim ↔ Lit, Lit | Native arithmetic/boolean evaluation |

`Prim` nodes evaluate when both auxiliary ports feed literals. The reducer
additionally scans for "ready" `Prim` nodes (both auxes are `Lit`), so a
top-level primitive whose principal is the root — which never forms a
principal-to-principal active pair — still reduces natively. This is the
**numerical operations** path: `Int64` (`Add64`…`Lt64`) and `F64`
(`AddF64`, `SubF64`, `MulF64`, `DivF64`, `EqF64`, `GtF64`, `LtF64`)
arithmetic lowers to a literal normal form. `F64` and `Int64` are never
silently coerced; a type-mismatched fold yields no result and the term is
left stuck rather than mis-evaluated.

Bounded at 1,000,000 iterations.

## UNF Hash

Canonical serialization walks the net from root, emitting type-tagged bytes
(0x01=Int64, 0x02=Bool, 0x03=Con, 0x04=Entity, 0x05=F64, 0xFF=other). SHA-256
of the byte sequence gives a deterministic, content-addressable fingerprint.
`F64` serializes via its IEEE-754 bit pattern, so `F64(3.0)` hashes
distinctly from `Int64(3)`.

Properties:
- Same sentence → same hash (determinism).
- Different sentences → different hashes (discrimination).
- Intensional equivalence: same sentence run twice → identical result + hash.

## Harper Gate

Pre-parse linter that runs before the CCG parse:
1. Tokenizes (whitespace split, strip non-alpha except apostrophes).
2. POS-tags via hardcoded lookup (NNP, DT, VBZ, CD, JJ, ...).
3. Rejects sentences with <2 words.

Purpose: reject obviously malformed input early and provide POS tags for
disambiguation.

## L1 Probabilistic Layer

`l1` module implements world-splitting for modal adverbs ("probably",
"possibly"):
- `split_l1(tree, triggers) -> Vec<(f64, DerivationTree)>` — splits a
  derivation into weighted worlds.
- `aggregate_results(worlds) -> Vec<(String, f64)>` — merges duplicate
  results, summing probabilities.
- `verify_world_probabilities(worlds, tol) -> bool` — asserts weights sum
  to 1.0.

## Austral Codegen

`austral_codegen::emit_austral(term) -> String` emits Austral source from
CoreIR, providing a path to independently verified native execution via the
australVM JIT.

## Austral ↔ DeltaNets ↔ UNF ↔ TED (the software translation)

The reverse direction — **AustralVM-language software → DeltaNets** — is the
`logos::translate` orchestrator (Phase 2–3 of the Total-Austral-Subset plan,
adapted to the existing net engine). A parsed Austral source fragment
(statement list, expression, function, or whole `module body …`) is:

1. **Lowered** to CoreIR (`austral_codegen::parser`, `parse_austral_*`);
2. **Validated** (Phase 1.3, `austral_codegen::validate`): totality — no
   raw recursion (only `fold`/`match`; direct or mutual call cycles are
   rejected with a diagnostic, since the inliner would loop) — and
   linearity — every variable used exactly once unless explicitly
   `clone x -> y, z;` (replicator) or `destroy x;` (eraser);
3. **Compiled** to an interaction net and **reduced** to its unique normal
   form (`deltanet::compile_to_net` + `reducer::reduce`);
4. **Read back** as a symbolic expression (`deltanet::symbolic`,
   `sym_readback`): a closed term (no unknowns) collapses to the numerical
   result of its calculation (e.g. `ADD` of two 64-bit integers → `5`); an
   open term stays symbolic (`Add64(x, 3)`);
5. **Canonicalized** to a word-level **TED** (Phase 3,
   `deltanet::ted`): the Int64-arithmetic fragment becomes a sorted
   polynomial over ℤ/2⁶⁴ — products distribute, like terms combine
   (`x + x → 2*x`, `0*y → 0`), monomials order deterministically — and
   its SHA-256 is the content-addressable *algebraic* UNF hash, on top of
   the net-level `unf_hash`;
6. **Verified** by the confluence self-check: a second independent
   reduction reproduces the identical UNF.

The deltanet reducer is the **hybrid evaluator** (Phase 5): when a `Prim`
agent's ports are all reduced literals, the reduction engine evaluates it
natively (the graph-to-native trap), so closed arithmetic blocks are
replaced by their numerical results during reduction, not after.

## Kernel integration (uk_austral_unf)

`prob_kernel::logos::austral_unf` exposes the translation over the kernel C
ABI as `uk_austral_unf`. The report (`AustralReport`) carries the symbolic
readback (`sym_expr`), the infix arithmetic (`infix`), the numerical result
(`value`, when closed), both UNF digests (`unf_hash` net-level,
`ted_hash` algebraic), the canonical TED polynomial (`ted`), the confluence
self-check (`verified`), and the echoed source. Validation rejections
(recursion, double use) surface as `UK-4804` (`AUSTRAL_UNF_FAILED`) with
the precise diagnostic. The symbol follows the S29 registration checklist.

The **OCaml/Why3/AustralVM cycle** closes the loop: australVM's
`Deltanet_plugin` compiler pass (`australVM/lib/deltanet_plugin.ml`,
registered in `Vm_plugin.boot` alongside the Why3 gate) serializes every
top-level constant expression of a compiled Austral module to the subset,
calls the kernel's `uk_austral_unf` through the rust bridge, and rejects
the module when the kernel's UNF value disagrees with the compiler's own
evaluation — the same proof-carrying-extension pattern as the Why3 gate,
with the kernel's unique normal form as the independent arbiter.

## Verified-Execution Guarantee

1. **Grammar correctness**: only L0-conformant sentences produce derivation
   trees; the CCG type system rejects ungrammatical input.
2. **Linearity**: every variable used exactly once → well-formed interaction
   net (no dangling/duplicated wires).
3. **Confluence**: interaction net reduction is confluent — normal form is
   unique regardless of reduction order.
4. **UNF hash**: cryptographic fingerprint of the execution result enables
   independent verification.
5. **Corpus validation**: 99+ grammatical + 50 ungrammatical sentences in
   `corpus/` with >50% parse rate assertion.

## CLI

```bash
logos parse "John loves Mary"     # CCG derivation tree
logos run "John adds two three"   # full pipeline → readback
logos verify "one is one" "Eq(1, 1)"  # assert expected output
logos hash "the cat sleeps"       # UNF hash (hex)
logos l1 "probably John loves Mary"   # probabilistic worlds
```

## Tests

- 12 unit test modules (ccg, core_ir, deltanet, harper_gate, lexicon, l1,
  austral_codegen).
- 15 integration tests (`tests/integration.rs`): end-to-end pipeline,
  UNF hash determinism/discrimination, L1 world splitting, linearity,
  corpus parse rate.

## Kernel integration (uk_logos_compile)

The pipeline is exposed over the unfer kernel C ABI as `uk_logos_compile` (the
CNL side):
a CNL sentence (NUL-free UTF-8) is parsed with an embedded L0 lexicon,
compiled to CoreIR, reduced to a UNF, and read back. The report — readback
result, content-addressed `unf_hash`, a confluence self-check (`verified`),
and the echoed sentence — is stored in `uk_get_result` and broadcast as a
`logos_compiled` event. Unparseable input fails with `UK-4803`
(`LOGOS_COMPILE_FAILED`). The symbol follows the S29 registration checklist
(EXPECTED_SYMBOLS.txt, generated C header, australVM `UNFER_SYMBOLS`,
`GrantSet.kernel`).

## Formal confluence

`lean/Confluence.lean` (checked by `lean`, the authoritative type-checker)
machine-verifies the confluence-critical structure of the reduction: the
diamond property, Church–Rosser confluence, and the uniqueness of normal
forms are decided by kernel-computed `rfl` over the explicitly enumerated
finite state space. This is the formal backbone of the "unique normal form"
guarantee; the runtime confluence self-check in `uk_logos_compile`
corroborates it empirically. The proof is exported through the S29
`lean4export` pipeline: the official `leanprover/lean4export` emits the
NDJSON format 3.1.0 payload (pinned at
`prob_kernel/tests/fixtures/confluence.ndjson`), which `verify_export`
re-verifies independently in nanoda. The proof term must stay
`rfl`/kernel-computed — `native_decide`/`decide` emit `Lean.ofReduceBool` +
`_nativeDecide_*` terms that nanoda (an independent checker) cannot reduce.
Regeneration needs the official `lean4export` on the matching toolchain (a
documented provisioning step).

