# NL → CNL: the `logos::formalize` pipeline

ProofFlow does *faithful proof autoformalization*: natural-language proofs →
Lean 4, through a dependency-graph pipeline, verified by a Lean compiler.
This document describes the adaptation of that pipeline to **logos'
Controlled Natural Language**, where the verification substrate is
interaction-net reduction rather than a type checker.

It is the second half of the rewrite plan (`emthin/docs/REWRITE_PLAN.md`
§11–§17); the first half is the emthin/mathed document work, which is where
these proofs are authored.

---

## 1. What changed, and what did not

| | ProofFlow | here |
|---|---|---|
| target language | Lean 4 | logos L0 CNL (`logos/src/lexicon`, `corpus/lexicon.tsv`) |
| oracle | `lean_check.py` — a Lean 4 process | `deltanet::reduce` → `readback` → SHA-256 |
| "it compiles" | `lean_pass` | gates, parses, compiles to a net |
| "it really works" | `lean_verify` (no `sorry`) | reduces to a **unique** normal form |
| tactic completer | `proof_prover.py` | reformulation loop (`completer.rs`) |
| node identity | the name `l4` | **the UNF hash** of its normal form |
| scoring | ProofScore | LogosScore (`score.rs`) |
| memory | none | engram-keyed lemma store (`memory.rs`) |

Everything else — the graph model, the DAG rules, the retry loop over typed
errors, the fuzzy-score aggregation, the centrality weighting — is a **port**,
not a redesign, and `eval_corpus.rs` checks the port against ProofFlow's own
committed human DAGs.

## 2. The substitution that drives it all

> **A node's identity is the UNF hash of its normal form.**

ProofFlow identifies a step by a name. Here, two CNL sentences that denotate
the same term *are* the same node. Three consequences run through the whole
design:

- **Content-addressed memory.** Two nodes denoting the same term share one
  lemma (`memory.rs`), and a store of lemmas is a store of normal forms.
- **Collapsing is a failure.** A formalization whose hash equals its
  dependency's is rejected (`FormalizeError::DuplicateIdentity`) — it is a
  duplicate of another step, not a formalization of anything.
- **Identity is what the compiler gate checks.** `Formalize_plugin` in
  australVM has no second opinion to disagree with, so where `Deltanet_plugin`
  compares a value it compares *identity*: two `cnl_` constants reducing to the
  same hash is a module that is making a mistake.

## 3. Pipeline

```
natural-language proof
        │  graph::build_proof_graph          (ProofFlow's proof_graph.py)
        ▼
   proof-graph DAG  ── validate ── check_DAG  (cycles, orphans, forward refs)
        │
        │  formalizer::formalize_node        (per node, in topological order)
        │     ├─ provenance cache hit?  ── yes ── reuse
        │     └─ no: LLM emits one ```cnl sentence
        │            └─ verify_cnl: gate → CCG → CoreIR → linearity
        │                            → net → reduce → readback → SHA-256
        │            └─ typed error → correction turn → retry
        ▼
   CNL sentence + UNF hash + readback          (identity established)
        │
        │  completer::complete_node           (reformulate until *uniquely* reduced)
        │  memory::insert                      (UNF-keyed, verified only)
        ▼
   score::judge_graph                         (LogosScore)
        ▼
   report.json + dag.html
```

### 3.1 Two stop conditions, not one

ProofFlow's formalizer only needs `lean_pass`, because its output still
contains a `sorry` and the tactic completer finishes it. That split is
preserved, because it is what the stages *are*:

- `formalize_node` stops at **compiles**.
- `formalize_node_verified` — the same function with the stop condition moved —
  is what the completer drives, and it demands a **unique** normal form.

## 4. The honest limit

§17.1 of the plan called L0's 46-word vocabulary the top risk. It is. The
measurement is in `eval_corpus.rs` and it is not flattering:

| | full 184-graph corpus |
|---|---|
| node statements using only L0 vocabulary | **0 of 1463** |
| distinct alphabetic words already in L0 | 16 of 852 (1.9%) |
| a domain lexicon would need | **~836 more** |

So: **the pipeline is correct and the language is too small.** No amount of
correctness changes that number, and the pipeline says so rather than hiding
it — an out-of-lexicon word produces `VerifyError::OutOfLexicon` *naming the
offending words*, which is the diagnostic a lexicon extension needs.

L0 coverage is extended by data, not by grammar changes:
`DomainLexicon::with_extension` appends a TSV in the lexicon's own format.

**An extension may add words, not constructors.** `core_ir::compiler::tag_id`
maps an unrecognized constructor name to tag `0`, which is a *legal* tag — so
the name is discarded, the term reads back as `Unknown(…)`, and two different
unknown constructors of equal arity produce the same term and therefore the
same UNF hash. Under §2's identity rule that is a silent identity collapse, so
`with_extension` rejects such a lexicon by name
(`LexiconError::UnknownConstructor`). Lifting the restriction properly needs a
deterministic name→tag registry in `core_ir`; that is a change to that crate.

## 5. Memory

Two caches, both needed:

- **Provenance cache**, keyed `(node statement, prompt version, model)` —
  §17.3's answer to LLM nondeterminism. An identical re-request is served from
  memory instead of being re-asked, so a run is reproducible even though the
  model is not. Deliberately *not* keyed on node id: `l3` in the next proof is a
  different statement.
- **Lemma store**, keyed by UNF hash. A lemma without a unique normal form has
  no content address and is refused, not stored — admitting one would make the
  store's hit rate a lie.

Reused: `engram::engram_key` for derivation (so a key here *is* an engram key),
`engram::Granularity` for its "a lookup at `g` must not fall back to a coarser
`g`" rule, and `engram::l1keys` for weighted retrieval.

Not reused: `EngramTable` / `TieredStore` / `Spill` are typed to
`Embedding = Vec<f32>` — vector payloads. A lemma's payload is text, and none
of that survives a round trip through `f32`. Their spill *addressing* is
`(granularity, unf_hash)`, which is exactly this store's key, so lifting the
payload type is a small change there rather than a reimplementation here.

### 5.1 An L1 design that could not work

The obvious use of `l1keys` is looking lemmas up by L1 *world* key. It cannot
work: a world's key comes from its `DerivationTree`, i.e. from the
sub-derivation *under* the trigger, while a lemma's key is the UNF of a whole
sentence. For `probably John sees Mary` the identity world's key is identical
for every fragment carrying that trigger — it is the key of the bare `probably`
modifier — so a sentence-keyed store never matches and every lookup silently
returns nothing.

So `l1keys` is used for what it can answer: how much probability mass reached
the UNF path. `memory::certainty` returns that (0.8 for `probably X`, since the
0.2 negate lands on a tagged fallback) and it scales the lexical score, so a
hedged fragment retrieves the same examples while reporting less confidence.

Retrieval is **lexical**, deliberately. There is no embedding model in the
pipeline — `placeholder_embedding` is derived from the key, so similarity on it
would be similarity of *hashes*, which says nothing about meaning. Overlap of
content words is weak but honest, and a weak real signal beats a
strong-looking one that measures nothing.

## 6. Scoring

All three layers of `proof_scorer.py`:

| layer | port |
|---|---|
| per-node judgement | LLM tags each component A/B/C (`Perfectly match` / `Minor` / `Major`) |
| per-node score | fuzzy measure `mu` + Sugeno integral, rule for rule |
| graph score | `semantic_score` weighted by Katz (default), Laplacian, or equal |

Two judgement details are **pinned rather than tidied**. ProofFlow's
down-sampling computes `b = 10 - a`, so a major inconsistency is counted as a
minor and then lost; and `sugeno_integral` guards on a `C` *before* calling
`generate_mu`, so the integral is always 0 and never reaches the
down-sampling. Both are reproduced, and `downsampling_loses_major_inconsistencies`
calls `generate_mu` directly to show the loss is real.

A **dependency check** is added on top: the property ProofScore asks the model
about — does the DAG faithfully reflect the proof's premises — is structural, so
it is *computed* rather than asked. Orphans, dangling refs, forward references,
and §2's identity requirement. `is_clean()` includes `unformalized`, so a
report with a gap cannot read as a clean bill of health.

### 6.1 The numeric check, and what it does not establish

`prob_kernel::Session` is unreachable from here (`prob_kernel` depends on
`logos`), so the numeric check is local and narrow: `numeric_agreement`
compares a node's closed value against its TED canonical form — two
independent canonicalizations of the same reduced term.

That catches a real class of bug. It is **not** a check that the CNL sentence
means what the natural language says; that is the judge's job, and every
`NumericVerdict` says so.

It is also **vacuous under stock L0**: every L0 predicate compiles to a
constructor application, so no L0 sentence denotes a number and every verdict
is `Open`. `stock_l0_denotes_no_numbers_so_the_numeric_check_is_open` asserts
that, so the vacuity cannot be mistaken for a working check. `Open` is
reported rather than treated as agreement — claiming "no disagreement" about a
term that never claimed a number would be a fabricated agreement.

## 7. Using it

```sh
cargo run -p logos --features llm-http -- formalize proof.md \
    --out report.json --vis dag.html \
    --lexicon corpus/lexicon.tsv --memory-out lemmas.json
```

Pointing at a local vLLM is a base-URL change and nothing else — the OpenAI
chat schema is what vLLM, llama.cpp, Ollama's shim, TGI and LM Studio all
implement:

```sh
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_MODEL=qwen2.5-7b cargo run …
```

`<input>` may also be a **proof-graph JSON file**, which skips the graph stage
entirely — useful for re-running the later stages against a fixed DAG.

**Exit codes.** `0` ran and trustworthy · `1` bad usage or a stage failed ·
`2` ran but the result is not trustworthy. A run that produced a report and
found problems is not a run that could not happen.

`logos unf <sentence> --json` is the single-sentence kernel surface, used by
the australVM plugin. Exit `1` is a rejected sentence, `2` is one that compiled
without a unique normal form — a distinction the plugin needs, because
collapsing them would make an unverifiable declaration look like a rejected one.

## 8. The Austral VM gate

`australVM/lib/formalize_plugin.ml` registers a `gate_pass` that verifies `cnl_`
string constants at compile time, and
`examples/modules/logos_formalize/module.toml` packages the pipeline as a hosted
module granting `uk_logos_compile` and nothing else.

Three properties are carried over from `Deltanet_plugin`: opt-in via
`UNFER_LOGOS=1`; **no-op when the kernel is unavailable**, with every other
outcome rejecting; and a protocol mismatch rejecting rather than passing. The
second is a distinction `Deltanet_plugin` cannot make (a bridge is present or
it is not), and getting it wrong lets a broken kernel read as an absent one —
which is the silent-pass failure mode that gate's own header warns about.

## 9. The document front end

`\formal(#s, #f, "CNL"[, "readback"[, "hash"]])` is a mathed property statement:
the span between `#s` and `#f` is the caption, the rest is the claim and its
expected kernel answer. It is **non-visual and non-kernel** — it produces no
figure and does not decide anything. Its only job is to carry a CNL sentence to
a verifier and say what came back.

emthin asks `logos unf <cnl> --json` during an ordinary relayout
(`crates/emthin/src/docui/formals.rs`), so a claim in the document is checked by
the same kernel surface §7 describes and the australVM plugin uses — one seam,
two callers. One subprocess per distinct sentence, cached by sentence text:
two steps with the same CNL are asked once, which is the transport half of the
identity rule above.

The rules that matter, all of them inherited from §8:

- **No kernel, no-op.** `EMTHIN_LOGOS_BIN`, then `$PATH`, then nothing. A
  checkout without `logos` still opens the editor.
- **A kernel that ran and failed is not silent.** Exit 1 and exit 2 become
  named verdicts in the document, so an out-of-lexicon word appears as
  *"words not in the lexicon: Euler"* rather than as a block that simply has no
  verdict.
- **Declared and verified are different things.** The caption shows the declared
  CNL in grey; the verdict arrives as a separate annotation. A document cannot
  assert a hash it did not check.

## 10. The DAG as a figure

The HTML dependency graph from `--vis` is self-contained — no external `src`,
no network — so it is an ordinary `\app` payload and needs nothing from the
compositor beyond what any application needs:

```typst
#1 proof DAG #2 \app(#1, #2, 900, 600, "dag")
#3 Mary sees Bob #4 \formal(#3, #4, "Mary sees Bob")
```

The first line reserves the slot and binds the app by glob; the second states
the claim. They coexist in one document and neither disturbs the other, because
a `\formal` step is not a figure and a DAG viewer is not special. That
coexistence is itself a test.

## 11. Verification

| | result |
|---|---|
| `cargo test -p logos` | 258 lib + 11 eval + 44 integration pass |
| `logos --features llm-http` | 258 lib + 8 live-socket HTTP tests pass |
| `eval_corpus` on the full 184-graph corpus | 173 validate; the 11 rejections are all real DAG violations in the source data |
| `australVM` | `dune build @check` clean; `dune runtest --force` green, including 16 `FormalizePluginTest` cases with and without a kernel |
| `cargo test -p emthin` | 164 lib + 20 integration pass; 2 ignored |
| the two ignored emthin tests | pass against a real `logos` build — they check the subprocess contract, including that the kernel's own rejection reason survives to the document |

Two known environmental limitations, both pre-existing and unrelated to this
work:

- `confluence_lean_proof_type_checks` shells out to `lean`, and elan has no
  toolchain configured. That test fails on a clean tree.
- The `unfer` workspace does not build here: `libz-ng-sys` (via `loro`, via
  `prob_kernel`) needs `cmake`, which is absent. Identically on a clean tree.
  This is why the kernel seam is exposed as `logos unf` rather than by calling
  `prob_kernel` directly.

## 12. Not done

- **Batch `cnl_formalize` protocol op.** §16 lists it as an optional
  optimization; `module.toml` says so in its `max_ms` note.
- **The `TieredStore`/`Spill` payload lift** described in §5.
- **A real model run.** Everything above is verified against the deterministic
  half of the pipeline and a scripted transport. No end-to-end run against a
  live model has been performed, so nothing here claims what accuracy the
  pipeline achieves — §4 predicts it is very low, and that prediction is
  untested.