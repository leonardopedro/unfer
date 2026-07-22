# Critical Review of the Project Logos Plan (from CNLs.txt)

## 1. What the Plan Gets Right

The core architectural insight is genuinely strong: **linear types (Austral) and interaction nets (Delta-Nets) share the same mathematical foundation (Linear Logic)**, making a bidirectional transpilation between them far more natural than, say, compiling C to lambda calculus. The observation that Austral's explicit `clone`/`destroy` maps 1:1 to Delta-Net Replicator/Eraser agents, and that Austral's linearity checker statically guarantees what Delta-Nets enforce dynamically, is correct and well-motivated.

The "total subset" design (banning general recursion, allowing only `fold`/`map`) is the right call: it guarantees strong normalization, which is the prerequisite for any canonical normal form to exist at all. The hybrid execution model (sequential Cranelift regions as "Native Agents" inside a parallel Delta-Net graph) is a legitimate and well-studied pattern in the interaction-net literature (cf. HVM/Bend's approach to native operations).

## 2. Critical Problems

### 2.1 Scope Inflation Without Validation

The conversation escalates from "a CCG parser for a controlled English subset" to "an alternative to LLMs that also proves the Yang-Mills mass gap" over ~15 exchanges, with every proposal validated in escalating superlatives ("brilliant," "breathtaking," "profound," "revolutionary") and zero pushback. The `QFM-17.pdf` report's own numbers show its QFM-Text model **losing to a plain n-gram baseline by 5–39% perplexity** with Krylov rank collapsing to 1–2. Building an elaborate CCG/TED front-end on top of a component that currently underperforms counting word frequencies is putting the cart before the horse.

### 2.2 The "Unique Normal Form" Claims Are Overstated

The plan conflates three distinct things:
- **Intensional equivalence** (same graph structure → same hash): achievable, but only proves two programs have identical structure, not identical behavior.
- **Extensional equivalence** (same input→output for all inputs): undecidable in general (Rice's theorem). No canonical form can solve this.
- **Algebraic equivalence** (same polynomial/circuit): achievable for pure arithmetic or pure boolean logic separately, but **not for their combination**. The plan's discussion of TEDs and "word-level canonical forms" never confronts that mixing `+ mod 2^64` with `XOR` has no known clean unified canonical form — this is precisely why ARX ciphers (ChaCha20, Salsa20) resist algebraic cryptanalysis.

### 2.3 TEDs and Delta-Nets Solve Different Problems

The plan treats TEDs and Delta-Nets as competing alternatives for the same task. They are not:
- **Delta-Nets** handle control flow, higher-order functions, variable binding, and β-reduction. They cannot simplify `(b + a) + a` to `b + 2a`.
- **TEDs** handle word-level algebraic simplification. They cannot represent a lambda, a closure, or a `fold`.

The correct relationship is sequential: Delta-Nets reduce the functional/control-flow structure, leaving a flat arithmetic DAG; TEDs then canonicalize that DAG. The plan eventually arrives at this but only after several confused exchanges.

### 2.4 The Hamiltonian Discussion Is Disconnected From Validation

The extensive discussion of `V D V†`, symbolic operators, complex phases for CCG roles, and "structural attention" is mathematically interesting but entirely unvalidated. No experiment is proposed to test whether any of this actually improves over the existing (failing) QFM-Text baseline, let alone over a random control. The plan assumes the answer to "will this fix the rank collapse?" is yes, without ever asking "is the rank collapse even caused by the coupling structure, or by the coupling magnitude?" (the weights are ~1e-5, making vacuum domination almost inevitable regardless of structure).

### 2.5 Missing Implementation Details

For a plan that aspires to be implementable, critical details are absent:
- No concrete data structures for the Delta-Net graph representation.
- No specification of the interaction rule table beyond informal descriptions.
- No algorithm for the CCG→Core-IR compilation (how exactly do combinators map to IR nodes?).
- No specification of how "Native Agents" (Cranelift-compiled sequential regions) interface with the Delta-Net runtime (calling convention, data marshalling, strictness boundary).
- No error handling strategy.
- No test corpus specification.

## 3. Constructive Recommendations

1. **Split into two independent projects.** Track A (the CNL→verified-graph compiler) is buildable now with no dependency on QFM. Track B (the QFM coupling experiment) should be a separate, gated research effort that cannot begin until a root-cause report explains why QFM-Text currently fails.

2. **Start with a correctness oracle, not an optimization.** Build a slow-but-correct reference Delta-Net interpreter first. Only optimize (WASM+TCO, parallel execution, Cranelift native agents) against a passing differential test suite.

3. **Stay on native-agent UNF for v1.** Do not Church-encode. Do not attempt bit-blasting or word-level algebraic canonical forms that mix arithmetic and bitwise operations. The native-agent normal form (`Cons(1, Cons(2, Nil))`) already gives byte-identical hashes for structurally identical results.

4. **Bound the probabilistic layer.** The "Exploding Worlds" design needs a hard cap (e.g., ≤4 triggers per sentence) and structural hash-consing of shared sub-world fragments.

5. **Validate before extending.** Every new component should have a test that can fail before it is built. The single biggest structural fix is: nothing gets added without a milestone and acceptance criterion.

---

# Project Logos v3 — Implementation Plan

## A Controlled Natural Language Compiler to Verified Execution Graphs

**Status:** Draft v3. Self-contained; no external context required.
**Audience:** An implementer (human or AI agent) with no prior knowledge of this project. Every term is defined before use. Every phase has explicit inputs, deliverables, and pass/fail acceptance criteria.

---

## Table of Contents

0. [Purpose and Rules](#0-purpose-and-rules)
1. [Glossary](#1-glossary)
2. [Non-Goals](#2-non-goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Phase 1 — L0 Language Definition](#4-phase-1--l0-language-definition)
5. [Phase 2 — Harper Gate and Tokenizer](#5-phase-2--harper-gate-and-tokenizer)
6. [Phase 3 — CCG Chart Parser](#6-phase-3--ccg-chart-parser)
7. [Phase 4 — Core IR and Linearity Checker](#7-phase-4--core-ir-and-linearity-checker)
8. [Phase 5 — Delta-Net Reference Interpreter](#8-phase-5--delta-net-reference-interpreter)
9. [Phase 6 — Austral Code Generation](#9-phase-6--austral-code-generation)
10. [Phase 7 — Canonical Normal Form (UNF)](#10-phase-7--canonical-normal-form-unf)
11. [Phase 8 — L1 Probabilistic Layer](#11-phase-8--l1-probabilistic-layer)
12. [Phase 9 — Hybrid Execution (Post-v1)](#12-phase-9--hybrid-execution-post-v1)
13. [Testing Strategy](#13-testing-strategy)
14. [Milestone Checklist](#14-milestone-checklist)
15. [Risk Register](#15-risk-register)
16. [Repository Layout](#16-repository-layout)

---

## 0. Purpose and Rules

**What this project builds:** A compiler that takes sentences from a small, closed Controlled Natural Language (a restricted subset of English plus integer arithmetic) and produces two outputs:

1. A **native binary** (via the Austral compiler) that executes the sentence's meaning.
2. A **canonical hash** (via a Delta-Net interaction-net reducer) such that two differently-phrased sentences computing the same thing produce the same hash.

**Rules for the implementer:**

- Work through phases in order. Do not skip.
- Every phase ends with **Acceptance Criteria**. These are pass/fail gates. Do not proceed with a failing or unmeasured criterion.
- If a design decision is not specified here, choose the smallest, most testable option. Record it in `DECISIONS.md` (one line per decision, with rationale).
- Do not add a module, dependency, or research direction not in this document without first adding a milestone and acceptance test here.
- Correctness before performance. Always.

---

## 1. Glossary

| Term | Definition |
|------|-----------|
| **CNL** (Controlled Natural Language) | A deliberately restricted, unambiguous subset of a natural language designed to be both human-readable and machine-parseable. This project's CNL is called **L0** (deterministic) and **L1** (probabilistic extension). |
| **CCG** (Combinatory Categorial Grammar) | A grammar formalism where every word has a syntactic **category** (e.g., `NP`, `S\NP`, `(S\NP)/NP`) and sentences are built by a small fixed set of **combinators** (forward application `>`, backward application `<`, forward composition `>B`, backward composition `<B`). Categories double as semantic function types. |
| **Interaction Net** | A graph-rewriting system (Lafont 1990). A net is a graph of **agents** (nodes) connected by **wires** (edges). Each agent has exactly one **principal port** and zero or more **auxiliary ports**. An **active pair** is two agents whose principal ports are wired together. Reduction rewrites active pairs according to a fixed rule table. Reduction is **confluent**: any reduction order yields the same result. |
| **Delta-Net** | A specific interaction-net system (Salvadori, arXiv:2505.20314) optimized for parallel λ-reduction. Uses a single **Replicator** agent with level annotations instead of Lamping's multiple delimiter nodes. |
| **Agent** | A node type in a Delta-Net (e.g., `App`, `Abs`, `Dup`, `Era`, `Con`, `Prim`). |
| **Austral** | An existing systems programming language with **linear types** (values used exactly once unless explicitly cloned) and **capability-based effect tracking** (side effects require holding a capability value). Compiles to C. No garbage collector. |
| **Core IR** | The small, purpose-built intermediate representation defined in Phase 4. Emitted by the CCG compiler; consumed by both backends. |
| **UNF** (Unique Normal Form) | A canonical byte-string serialization of a fully-reduced Delta-Net graph. Two graphs representing the same computation produce byte-identical UNFs. |
| **Linearity** | The property that every bound variable is used exactly once on every control-flow path. Enforced statically by the linearity checker (Phase 4) and dynamically by the Delta-Net wire discipline (Phase 5). |
| **Fold** | A bounded iteration primitive: `fold(f, init, list)`. The only iteration primitive allowed in Core IR. Guarantees termination because lists are finite and there is no general recursion. |
| **Native Agent** | A Delta-Net agent that, instead of graph-rewriting, calls out to compiled machine code (Cranelift) for sequential computation. Used in Phase 9 (post-v1). |
| **Harper** | Automattic's existing Rust-native, offline grammar checker. Used as a pre-parse linting gate. |

---

## 2. Non-Goals

The following are explicitly **out of scope** for this project. Do not implement them. Do not add them without a future revision of this document that includes milestones and acceptance tests.

- Competing with or replacing large language models.
- Parsing unrestricted, general English. L0/L1 are closed CNLs with a curated lexicon.
- Physics simulations or "validation" (lattice gauge theory, Ising models, gravitational waveforms, CFD).
- GPU/CUDA acceleration of any kind.
- Bit-blasted boolean circuit synthesis, ROBDDs, or word-level algebraic canonical forms (TEDs, GF(2^64) polynomials) that mix modular arithmetic with bitwise operations. (Reason: no known clean unified canonical form exists for mixed arithmetic+bitwise algebra. This is a recognized hard problem, not an implementation detail.)
- Church/Scott-encoding the final normal form. (Reason: comparing Church-encoded graphs requires graph-isomorphism-level algorithms. The native-agent UNF is sufficient for the stated requirement.)
- Any quantum-flow Hamiltonian, QFM integration, or "structural attention" mechanism. (Reason: the existing QFM system's own reported numbers show it underperforming a plain n-gram baseline. Do not build on a failing foundation.)
- Proving general program equivalence (extensional equivalence is undecidable by Rice's theorem). This project proves **intensional** equivalence only (same graph structure → same hash).

---

## 3. Architecture Overview

```
L0/L1 text (English + arithmetic)
        │
        ▼
┌─────────────────────────┐
│  Harper Gate            │  Reject ungrammatical input with actionable message.
│  (Phase 2)              │  POS-tag for soft re-ranking (not hard constraint).
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  CCG Chart Parser       │  Produce one or more derivation trees.
│  (Phase 3)              │  Ambiguity is preserved, never silently resolved.
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  CCG → Core IR          │  Insert explicit Clone/Drop from linear-usage count.
│  Compiler (Phase 4)     │  Type-check under linearity checker.
└───────────┬─────────────┘
            │
            ├────────────────────────────┐
            │                            │
            ▼                            ▼
┌───────────────────────┐  ┌───────────────────────────┐
│  Austral Codegen      │  │  Delta-Net Codegen        │
│  (Phase 6)            │  │  (Phase 5)                │
│                       │  │                           │
│  Core IR → Austral    │  │  Core IR → Δ-Net graph    │
│  → C → native binary  │  │  → reduce → readback      │
│                       │  │  → UNF (canonical hash)   │
│  EXECUTION path       │  │  EQUIVALENCE path         │
└───────────────────────┘  └───────────────────────────┘
            │                            │
            └──────────┬─────────────────┘
                       │
                       ▼
              Three-way agreement:
              gold result == Δ-Net readback == Austral output
              (This is the M5 correctness oracle.)
```

**Key design principle:** The two backends (Austral, Delta-Net) are independent consumers of the same Core IR. They must agree on results. Their agreement is the project's primary correctness test.

---

## 4. Phase 1 — L0 Language Definition

**Goal:** Precisely define the controlled natural language L0 before writing any code.

### 4.1 Lexicon

Create `corpus/lexicon.tsv` with columns: `word`, `ccg_category`, `semantic_template`.

Seed entries (minimum 40 words):

```tsv
John        NP                  John
Mary        NP                  Mary
Bob         NP                  Bob
the         NP/N                λn.n
a           NP/N                λn.n
number      N                   Number
cat         N                   Cat
dog         N                   Dog
loves       (S\NP)/NP           λy.λx.Love(x,y)
sees        (S\NP)/NP           λy.λx.See(x,y)
adds        ((S\NP)/NP)/NP      λz.λy.λx.Assign(x,Add(y,z))
multiplies  ((S\NP)/NP)/NP      λz.λy.λx.Assign(x,Mul(y,z))
is          (S\NP)/NP           λy.λx.Eq(x,y)
equals      (S\NP)/NP           λy.λx.Eq(x,y)
greater     (S\NP)/NP           λy.λx.Gt(x,y)
less        (S\NP)/NP           λy.λx.Lt(x,y)
and         conj                (coordination combinator)
that        (NP\NP)/S           λs.λn.Restrict(n,s)
which       (NP\NP)/S           λs.λn.Restrict(n,s)
zero        NP                  Lit(0)
one         NP                  Lit(1)
two         NP                  Lit(2)
three       NP                  Lit(3)
four        NP                  Lit(4)
five        NP                  Lit(5)
ten         NP                  Lit(10)
true        NP                  Lit(true)
false       NP                  Lit(false)
not         (S\NP)\(S\NP)      λvp.λx.Not(vp(x))
```

### 4.2 Combinator Set (Fixed)

| Combinator | Schema | Example |
|-----------|--------|---------|
| Forward Application (`>`) | `X/Y  Y  →  X` | `(S\NP)/NP  NP  →  S\NP` |
| Backward Application (`<`) | `Y  X\Y  →  X` | `NP  S\NP  →  S` |
| Forward Composition (`>B`) | `X/Y  Y/Z  →  X/Z` | `(S\NP)/NP  NP/N  →  (S\NP)/N` |
| Backward Composition (`<B`) | `Y\Z  X\Y  →  X\Z` | |
| Type-Raising | Only for a **fixed, named list** of trigger words. Do NOT allow unrestricted type-raising. | |

### 4.3 Grammar (BNF)

Write `docs/L0_GRAMMAR.md` containing:

```
Sentence    := NP VP
VP          := Vtrans NP | Vintrans | Vditrans NP NP | Copula NP
NP          := Det N | Det N RelClause | ProperNoun | Numeral
RelClause   := RelPron VP
Det         := "the" | "a"
N           := "number" | "cat" | "dog" | ...
Vtrans      := "loves" | "sees" | ...
Vintrans    := "sleeps" | ...
Vditrans    := "adds" | "multiplies" | ...
Copula      := "is" | "equals" | ...
Numeral     := "zero" | "one" | "two" | ... | "ten"
RelPron     := "that" | "which"
Coordination := NP "and" NP | VP "and" VP
```

### 4.4 Semantic Templates

Each lexical entry's `semantic_template` is a lambda term in the following mini-language:

```
SemExpr := Var(id)
         | Lit(Int64(n) | Bool(b))
         | Con(tag, [SemExpr])       -- e.g., Love(x,y), Add(x,y)
         | Lam(id, SemExpr)
         | App(SemExpr, SemExpr)
```

### Acceptance Criteria (Milestone M0)

- [ ] `lexicon.tsv` has ≥40 entries, each with a valid CCG category and semantic template.
- [ ] `L0_GRAMMAR.md` is written and covers every sentence shape the lexicon can generate.
- [ ] A 200-sentence L0 test corpus exists in `corpus/l0_seed/`, covering every grammar rule ≥3 times.
- [ ] Each corpus sentence has a hand-written gold CCG derivation tree (in JSON) and a hand-computed gold result.
- [ ] A 50-sentence intentionally-ungrammatical corpus exists in `corpus/l0_ungrammatical/`.

---

## 5. Phase 2 — Harper Gate and Tokenizer

**Goal:** Reject ungrammatical input early with actionable error messages.

### 5.1 Integration

Use Harper (or an equivalent deterministic, rule-based grammar/POS tool) as a **linter**, not a parser.

```rust
// Pseudocode for the gate
fn harper_gate(input: &str) -> Result<TokenStream, GateError> {
    let harper_result = harper::lint(input);
    if harper_result.has_errors() {
        return Err(GateError::Grammar(harper_result.errors()));
    }
    let tokens = tokenize(input);  // simple whitespace + punctuation tokenizer
    Ok(tokens)
}
```

### 5.2 POS Tags as Soft Signal

Do **not** use Harper's POS tags as hard constraints on CCG categories. Harper is tuned for general prose, not for feeding a formal chart parser. Instead:

- Record Harper's POS tags alongside each token.
- After CCG parsing, if multiple derivations exist, use POS agreement as a **re-ranking signal** (prefer derivations whose CCG categories are consistent with Harper's POS tags).
- **Measure** the agreement rate between Harper POS and gold CCG categories on the test corpus. Record this number. If agreement is below 80%, do not use POS for re-ranking at all.

### Acceptance Criteria (Milestone M1)

- [ ] 100% of the 200-sentence corpus passes the Harper gate.
- [ ] 100% of the 50-sentence ungrammatical corpus is rejected with an actionable message (e.g., "verb 'sleep' requires a singular subject").
- [ ] Harper-POS / gold-CCG agreement rate is measured and recorded in `docs/METRICS.md`.

---

## 6. Phase 3 — CCG Chart Parser

**Goal:** Parse tokenized L0 sentences into one or more CCG derivation trees.

### 6.1 Algorithm

Standard CYK-style bottom-up chart parsing.

```rust
// Pseudocode
struct ChartCell {
    entries: Vec<ChartEntry>,  // may be >1 if ambiguous
}

struct ChartEntry {
    category: CCGCategory,
    derivation: DerivationTree,
    score: f64,  // for re-ranking if POS tags are used
}

fn parse(tokens: &[Token], lexicon: &Lexicon) -> Vec<DerivationTree> {
    let n = tokens.len();
    let mut chart = vec![vec![ChartCell::default(); n + 1]; n + 1];

    // Lexical insertion
    for (i, tok) in tokens.iter().enumerate() {
        for entry in lexicon.lookup(tok) {
            chart[i][i+1].entries.push(ChartEntry::lexical(entry));
        }
    }

    // Chart filling
    for span in 2..=n {
        for i in 0..=(n - span) {
            let j = i + span;
            for k in (i+1)..j {
                for left in &chart[i][k].entries {
                    for right in &chart[k][j].entries {
                        for result in apply_combinators(left, right) {
                            chart[i][j].entries.push(result);
                        }
                    }
                }
            }
        }
    }

    // Extract all S-category entries spanning [0, n]
    chart[0][n].entries.iter()
        .filter(|e| e.category == CCGCategory::S)
        .map(|e| e.derivation.clone())
        .collect()
}
```

### 6.2 Combinator Application

```rust
fn apply_combinators(left: &ChartEntry, right: &ChartEntry) -> Vec<ChartEntry> {
    let mut results = Vec::new();

    // Forward Application: X/Y + Y → X
    if let CCGCategory::Slash(fwd, x, y) = &left.category {
        if fwd && y == &right.category {
            results.push(ChartEntry::application(x.clone(), left, right, Forward));
        }
    }

    // Backward Application: Y + X\Y → X
    if let CCGCategory::Slash(fwd, x, y) = &right.category {
        if !fwd && y == &left.category {
            results.push(ChartEntry::application(x.clone(), left, right, Backward));
        }
    }

    // Forward Composition: X/Y + Y/Z → X/Z
    // Backward Composition: Y\Z + X\Y → X\Z
    // (implement similarly)

    results
}
```

### 6.3 Ambiguity Handling

- If a sentence has >1 valid derivation, return **all** of them.
- In L0 (v1): treat >1 derivation as a **hard parse error** to be fixed by tightening the lexicon. Log a warning with all derivations.
- In L1 (Phase 8): multiple derivations are expected and handled by the probabilistic layer.

### 6.4 Derivation Tree Format

```json
{
  "type": "Application",
  "direction": "forward",
  "result_category": "S",
  "left": {
    "type": "Application",
    "direction": "backward",
    "result_category": "S\\NP",
    "left": { "type": "Leaf", "word": "loves", "category": "(S\\NP)/NP" },
    "right": { "type": "Leaf", "word": "Mary", "category": "NP" }
  },
  "right": { "type": "Leaf", "word": "John", "category": "NP" }
}
```

### Acceptance Criteria (Milestone M2)

- [ ] Exact match against gold CCG trees for 100% of the 200-sentence corpus.
- [ ] Parse time vs. sentence length is measured and recorded (sanity check: should be polynomial, not exponential).
- [ ] Any sentence with >1 valid parse returns all parses (logged, not silently dropped).

---

## 7. Phase 4 — Core IR and Linearity Checker

**Goal:** Compile CCG derivation trees into a small, typed, linear intermediate representation.

### 7.1 Core IR Definition

```rust
enum CoreIR {
    Var(Id),
    Lit(Literal),                          // Int64(i64) | Bool(bool)
    Con(TagId, Vec<CoreIR>),               // Cons, Nil, Some, None, Love, Add, ...
    Lam(Id, Box<CoreIR>),
    App(Box<CoreIR>, Box<CoreIR>),
    Let(Id, Box<CoreIR>, Box<CoreIR>),
    Match(Box<CoreIR>, Vec<(Pattern, CoreIR)>),
    Fold(Box<CoreIR>, Box<CoreIR>, Box<CoreIR>),  // fold(f, init, list)
    Prim(PrimOp, Vec<CoreIR>),             // Add64, Sub64, Mul64, Eq64, Gt64, And, Or, Not
    Clone(Id, Id, Id, Box<CoreIR>),        // clone `id` into two fresh bindings
    Drop(Id, Box<CoreIR>),                 // explicitly discard `id`
}

enum Pattern {
    Tag(TagId, Vec<Id>),                   // e.g., Cons(head, tail)
}

enum PrimOp {
    Add64, Sub64, Mul64, Eq64, Gt64, Lt64, And, Or, Not,
}

enum Literal {
    Int64(i64),
    Bool(bool),
}
```

**Critical rules:**
- `Clone` and `Drop` are **explicit and mandatory**. There is no implicit sharing or implicit discarding anywhere.
- `Fold` is the **only** iteration primitive. No general recursion. No `while` loops. No self-referential `Let`.
- All arithmetic is `i64` with **wraparound** semantics (`wrapping_add`, `wrapping_sub`, `wrapping_mul`).

### 7.2 CCG Derivation → Core IR Compiler

Walk the derivation tree bottom-up:

```rust
fn compile_derivation(node: &DerivationTree, lexicon: &Lexicon) -> CoreIR {
    match node {
        DerivationTree::Leaf(word) => {
            // Look up semantic template from lexicon
            let template = lexicon.semantic_template(word);
            instantiate_template(template)
        }
        DerivationTree::Application { left, right, .. } => {
            let f = compile_derivation(left, lexicon);
            let arg = compile_derivation(right, lexicon);
            CoreIR::App(Box::new(f), Box::new(arg))
        }
        DerivationTree::Composition { left, right, .. } => {
            // X/Y + Y/Z → X/Z
            // λz. (compile(left))( (compile(right))(z) )
            let f = compile_derivation(left, lexicon);
            let g = compile_derivation(right, lexicon);
            let z = fresh_id();
            CoreIR::Lam(z, Box::new(CoreIR::App(
                Box::new(f),
                Box::new(CoreIR::App(Box::new(g), Box::new(CoreIR::Var(z))))
            )))
        }
    }
}
```

### 7.3 Linear Usage Counting Pass

After building the Core IR term, run a pass that counts variable usage:

```rust
fn insert_linearity(term: CoreIR) -> CoreIR {
    // For every Let(id, value, body) and Lam(id, body):
    //   count = count_uses(id, body)
    //   if count == 0: wrap body in Drop(id, body)
    //   if count == 1: leave as-is
    //   if count > 1: insert Clone(id, id1, id2, ...) to give each use a fresh binding
    // Recurse into subterms.
}
```

### 7.4 Linearity Checker (Standalone Tool)

A separate function that verifies: every bound variable is used **exactly once** on every control-flow path.

```rust
fn check_linearity(term: &CoreIR) -> Result<(), LinearityError> {
    // Walk the term. Maintain a context: Map<Id, UsageCount>.
    // At Lam(id, body) and Let(id, _, body): id must be used exactly once in body.
    // At Match(scrutinee, arms): id must be used exactly once in EACH arm.
    // At Clone(id, id1, id2, body): id is consumed; id1 and id2 must each be used exactly once.
    // At Drop(id, body): id is consumed; must not appear in body.
    // Return error if any violation found.
}
```

### Acceptance Criteria (Milestone M3)

- [ ] 100% of corpus sentences compile to Core IR that passes the linearity checker.
- [ ] Pretty-print Core IR to S-expressions. For the full corpus, a human (or independent LLM pass) confirms the S-expression matches the sentence's intended meaning.
- [ ] The linearity checker rejects at least 5 hand-crafted ill-formed Core IR terms (variable used twice, variable unused, etc.).

---

## 8. Phase 5 — Delta-Net Reference Interpreter

**Goal:** A correct (not yet fast) Delta-Net reducer for Core IR. This serves as the **correctness oracle** for the Austral backend.

### 8.1 Data Structures

```rust
type NodeId = u32;

struct Port {
    node: NodeId,
    slot: u8,  // slot 0 = principal port; slots 1..N = auxiliary ports
}

enum AgentKind {
    App,                    // 2 aux ports: func, arg
    Abs,                    // 2 aux ports: var, body
    Con(TagId, u8),         // tag + arity aux ports
    Fold,                   // 3 aux ports: f, init, list
    Dup(u16),               // level; 2 aux ports
    Era,                    // 0 aux ports
    Prim(PrimOp),           // 2 aux ports: left, right
    Lit(Literal),           // 0 aux ports (value carried in agent)
}

struct Node {
    kind: AgentKind,
    ports: Vec<Option<Port>>,  // ports[0] = principal
}

struct Net {
    nodes: Vec<Option<Node>>,
    free_list: Vec<NodeId>,
    active_pairs: VecDeque<(NodeId, NodeId)>,
    root: Port,  // the output port of the whole computation
}
```

### 8.2 Core IR → Delta-Net Compilation

```rust
fn compile_to_net(term: &CoreIR) -> Net {
    let mut net = Net::new();
    let root_port = emit(term, &mut net);
    net.root = root_port;
    net.collect_active_pairs();
    net
}

fn emit(term: &CoreIR, net: &mut Net) -> Port {
    match term {
        CoreIR::Var(id) => {
            // Return the port previously allocated for this variable
            net.lookup_var_port(id)
        }
        CoreIR::Lit(lit) => {
            let node = net.alloc_node(AgentKind::Lit(lit.clone()), 0);
            Port { node, slot: 0 }
        }
        CoreIR::Con(tag, args) => {
            let arity = args.len() as u8;
            let node = net.alloc_node(AgentKind::Con(*tag, arity), arity);
            for (i, arg) in args.iter().enumerate() {
                let arg_port = emit(arg, net);
                net.wire(Port { node, slot: (i + 1) as u8 }, arg_port);
            }
            Port { node, slot: 0 }
        }
        CoreIR::Lam(id, body) => {
            let node = net.alloc_node(AgentKind::Abs, 2);
            // Allocate a port for the bound variable
            let var_port = Port { node, slot: 1 };
            net.bind_var(id, var_port);
            let body_port = emit(body, net);
            net.wire(Port { node, slot: 2 }, body_port);
            Port { node, slot: 0 }
        }
        CoreIR::App(f, arg) => {
            let node = net.alloc_node(AgentKind::App, 2);
            let f_port = emit(f, net);
            let arg_port = emit(arg, net);
            net.wire(Port { node, slot: 1 }, f_port);
            net.wire(Port { node, slot: 2 }, arg_port);
            Port { node, slot: 0 }
        }
        CoreIR::Fold(f, init, list) => {
            let node = net.alloc_node(AgentKind::Fold, 3);
            let f_port = emit(f, net);
            let init_port = emit(init, net);
            let list_port = emit(list, net);
            net.wire(Port { node, slot: 1 }, f_port);
            net.wire(Port { node, slot: 2 }, init_port);
            net.wire(Port { node, slot: 3 }, list_port);
            Port { node, slot: 0 }
        }
        CoreIR::Prim(op, args) => {
            assert_eq!(args.len(), 2);
            let node = net.alloc_node(AgentKind::Prim(*op), 2);
            let l = emit(&args[0], net);
            let r = emit(&args[1], net);
            net.wire(Port { node, slot: 1 }, l);
            net.wire(Port { node, slot: 2 }, r);
            Port { node, slot: 0 }
        }
        CoreIR::Clone(id, id1, id2, body) => {
            let node = net.alloc_node(AgentKind::Dup(0), 2);
            let orig_port = net.lookup_var_port(id);
            net.wire(Port { node, slot: 0 }, orig_port);  // principal meets the wire
            let p1 = Port { node, slot: 1 };
            let p2 = Port { node, slot: 2 };
            net.bind_var(id1, p1);
            net.bind_var(id2, p2);
            emit(body, net)
        }
        CoreIR::Drop(id, body) => {
            let node = net.alloc_node(AgentKind::Era, 0);
            let orig_port = net.lookup_var_port(id);
            net.wire(Port { node, slot: 0 }, orig_port);
            emit(body, net)
        }
        CoreIR::Let(id, value, body) => {
            let val_port = emit(value, net);
            net.bind_var(id, val_port);
            emit(body, net)
        }
        CoreIR::Match(scrutinee, arms) => {
            // Compile as a chain of Con-interactions.
            // For v1: support only non-overlapping constructor patterns.
            // Emit the scrutinee, then for each arm, emit a Con agent
            // whose principal port faces the scrutinee's output.
            // (Detailed implementation depends on the specific ADT encoding.)
            todo!("Implement Match compilation in Phase 5")
        }
    }
}
```

### 8.3 Interaction Rule Table (Exhaustive for v1)

Implement **exactly** these rules. No others.

```rust
fn interact(net: &mut Net, a: NodeId, b: NodeId) {
    let kind_a = net.nodes[a].kind;
    let kind_b = net.nodes[b].kind;

    match (kind_a, kind_b) {
        // Beta reduction: App >< Abs
        (AgentKind::App, AgentKind::Abs) => {
            // Wire App.arg → Abs.var;  App.func → Abs.body
            let app_arg = net.get_aux(a, 2);
            let abs_var = net.get_aux(b, 1);
            let app_func = net.get_aux(a, 1);
            let abs_body = net.get_aux(b, 2);
            net.wire(app_arg, abs_var);
            net.wire(app_func, abs_body);
            net.free_node(a);
            net.free_node(b);
        }

        // Dup >< Dup (same level): annihilate
        (AgentKind::Dup(l1), AgentKind::Dup(l2)) if l1 == l2 => {
            let a1 = net.get_aux(a, 1);
            let a2 = net.get_aux(a, 2);
            let b1 = net.get_aux(b, 1);
            let b2 = net.get_aux(b, 2);
            net.wire(a1, b1);
            net.wire(a2, b2);
            net.free_node(a);
            net.free_node(b);
        }

        // Dup >< Dup (different level): commute
        (AgentKind::Dup(l1), AgentKind::Dup(l2)) if l1 != l2 => {
            // Create two new Dup agents, cross-wire
            let new_a = net.alloc_node(AgentKind::Dup(l1), 2);
            let new_b = net.alloc_node(AgentKind::Dup(l2), 2);
            // ... cross-wiring logic ...
            net.free_node(a);
            net.free_node(b);
        }

        // Dup >< Con/Abs/App/Fold/Prim: commute (push Dup through)
        (AgentKind::Dup(_), _) | (_, AgentKind::Dup(_)) => {
            // Standard interaction-combinator commutation:
            // create two copies of the non-Dup agent,
            // wire each to one aux port of the Dup.
            // ... implementation ...
        }

        // Era >< anything: erase
        (AgentKind::Era, _) => {
            // Wire all aux ports of the other agent to fresh Era nodes.
            for slot in 1..net.nodes[b].ports.len() {
                let era = net.alloc_node(AgentKind::Era, 0);
                net.wire(net.get_aux(b, slot as u8), Port { node: era, slot: 0 });
            }
            net.free_node(a);
            net.free_node(b);
        }
        (_, AgentKind::Era) => {
            // Symmetric case
            for slot in 1..net.nodes[a].ports.len() {
                let era = net.alloc_node(AgentKind::Era, 0);
                net.wire(net.get_aux(a, slot as u8), Port { node: era, slot: 0 });
            }
            net.free_node(a);
            net.free_node(b);
        }

        // Fold >< Con(Nil, 0): reduce to init
        (AgentKind::Fold, AgentKind::Con(tag, 0)) if tag == NIL_TAG => {
            let init_port = net.get_aux(a, 2);
            let result_port = net.get_aux(a, 0);  // principal
            net.wire(result_port, init_port);
            // Free the Fold and Nil agents
            net.free_node(a);
            net.free_node(b);
        }

        // Fold >< Con(Cons, 2): reduce to f(head, Fold(f, init, tail))
        (AgentKind::Fold, AgentKind::Con(tag, 2)) if tag == CONS_TAG => {
            let f_port = net.get_aux(a, 1);
            let init_port = net.get_aux(a, 2);
            let head_port = net.get_aux(b, 1);
            let tail_port = net.get_aux(b, 2);

            // Create: App(App(f, head), Fold(f, init, tail))
            let inner_fold = net.alloc_node(AgentKind::Fold, 3);
            // Wire inner_fold's f, init, list ports
            // ... (clone f and init via Dup agents) ...

            let inner_app = net.alloc_node(AgentKind::App, 2);
            let outer_app = net.alloc_node(AgentKind::App, 2);
            // Wire: outer_app.func = inner_app, outer_app.arg = head
            // inner_app.func = f, inner_app.arg = inner_fold

            net.free_node(a);
            net.free_node(b);
        }

        // Prim >< Lit, Lit: native evaluation
        (AgentKind::Prim(op), AgentKind::Lit(_)) => {
            // Check if both aux ports of Prim are connected to Lit agents.
            // If so, compute the result natively.
            if let (Some(lit1), Some(lit2)) = (net.get_connected_lit(a, 1), net.get_connected_lit(a, 2)) {
                let result = eval_prim(op, lit1, lit2);
                let result_node = net.alloc_node(AgentKind::Lit(result), 0);
                let prim_principal = Port { node: a, slot: 0 };
                net.wire(prim_principal, Port { node: result_node, slot: 0 });
                net.free_node(a);
                // Free the two Lit nodes
            }
        }

        // Con >< Con: STUCK TERM (bug detector)
        (AgentKind::Con(_, _), AgentKind::Con(_, _)) => {
            panic!("STUCK TERM: two Con agents met principal-to-principal. This indicates a type error in the source program.");
        }

        _ => {
            // Not an active pair. Do nothing.
        }
    }
}

fn eval_prim(op: PrimOp, a: Literal, b: Literal) -> Literal {
    match (op, a, b) {
        (PrimOp::Add64, Literal::Int64(x), Literal::Int64(y)) => Literal::Int64(x.wrapping_add(y)),
        (PrimOp::Sub64, Literal::Int64(x), Literal::Int64(y)) => Literal::Int64(x.wrapping_sub(y)),
        (PrimOp::Mul64, Literal::Int64(x), Literal::Int64(y)) => Literal::Int64(x.wrapping_mul(y)),
        (PrimOp::Eq64, Literal::Int64(x), Literal::Int64(y)) => Literal::Bool(x == y),
        (PrimOp::Gt64, Literal::Int64(x), Literal::Int64(y)) => Literal::Bool(x > y),
        (PrimOp::Lt64, Literal::Int64(x), Literal::Int64(y)) => Literal::Bool(x < y),
        (PrimOp::And, Literal::Bool(x), Literal::Bool(y)) => Literal::Bool(x && y),
        (PrimOp::Or, Literal::Bool(x), Literal::Bool(y)) => Literal::Bool(x || y),
        _ => panic!("Type error in Prim evaluation"),
    }
}
```

### 8.4 Reduction Loop

```rust
fn reduce(net: &mut Net) {
    net.collect_active_pairs();
    while let Some((a, b)) = net.active_pairs.pop_front() {
        if net.is_freed(a) || net.is_freed(b) { continue; }
        interact(net, a, b);
        net.collect_new_active_pairs();
    }
}
```

### 8.5 Readback

After reduction halts, traverse from `net.root` and serialize:

```rust
fn readback(net: &Net) -> String {
    readback_port(net, net.root)
}

fn readback_port(net: &Net, port: Port) -> String {
    let node = &net.nodes[port.node];
    match &node.kind {
        AgentKind::Lit(Literal::Int64(n)) => format!("{}", n),
        AgentKind::Lit(Literal::Bool(b)) => format!("{}", b),
        AgentKind::Con(tag, arity) => {
            let args: Vec<String> = (1..=*arity)
                .map(|s| readback_port(net, net.get_aux(port.node, s)))
                .collect();
            format!("{}({})", tag_name(tag), args.join(", "))
        }
        // For other agent types that remain after reduction:
        // this indicates an incomplete reduction (should not happen for well-typed programs)
        other => format!("<STUCK:{:?}>", other),
    }
}
```

### Acceptance Criteria (Milestone M4)

- [ ] For every corpus program, Delta-Net reduction of its Core IR **halts** (non-halting = P0 bug).
- [ ] Readback output exactly matches the gold expected result for 100% of the corpus.
- [ ] The `Con >< Con` stuck-term detector fires on at least 2 hand-crafted ill-typed Core IR inputs.

---

## 9. Phase 6 — Austral Code Generation

**Goal:** Compile Core IR to Austral source code, then use the Austral compiler to produce a native binary.

### 9.1 Emission Mapping

| Core IR | Austral emission |
|---------|-----------------|
| `Con(tag, args)` | Linear record/variant construction |
| `Lam` / `App` | Closure conversion: each lambda → top-level function + linear environment record |
| `Fold(f, init, list)` | A `while`-style loop over the native list, calling `f` once per element |
| `Prim(op, [a, b])` | Direct native operator (`+`, `-`, `*`, comparisons) with `i64` wraparound |
| `Clone(id, id1, id2, body)` | Explicit `clone` call on the linear value |
| `Drop(id, body)` | Explicit `destroy` call |
| `Lit(Int64(n))` | Literal integer |
| `Lit(Bool(b))` | Literal boolean |

### 9.2 Closure Conversion (Defunctionalization)

```rust
// For each Lam(id, body) with free variables [fv1, fv2, ...]:
// 1. Generate a record type:
//    record Env_Lambda_N is
//        fv1: Type1;
//        fv2: Type2;
//    end;
// 2. Generate a top-level function:
//    function impl_lambda_N(env: Env_Lambda_N, id: ArgType): RetType is
//        -- body, with free variables accessed as env.fv1, env.fv2, ...
//    end;
// 3. At the Lam site, emit:
//    let closure_N: Closure = { ptr: impl_lambda_N, env: Env_Lambda_N(fv1, fv2, ...) };
// 4. At App(closure, arg) sites, emit:
//    let result = closure.ptr(closure.env, arg);
```

### 9.3 Performance Target (Modest and Checkable)

The Austral-compiled binary must run **at least 2× faster** than a naive tree-walking interpreter of the same Core IR, on the corpus's arithmetic-heavy programs.

Target **native Austral → C** compilation only for v1. Do not target WASM, Cranelift, or parallel execution.

### Acceptance Criteria (Milestone M5 — Core Correctness Oracle)

For every program in the corpus, **all three** must agree exactly:

1. The gold expected result (hand-computed, from M0).
2. The Delta-Net reducer's readback result (Phase 5).
3. The compiled-and-executed Austral binary's output.

**Any mismatch among these three is a P0 bug.** Do not proceed past this milestone with a known, unresolved mismatch.

Additionally:
- [ ] The 2× performance target is measured and met.

---

## 10. Phase 7 — Canonical Normal Form (UNF)

**Goal:** Produce a canonical byte-string hash from a fully-reduced Delta-Net graph.

### 10.1 Design Decision: Native-Agent UNF (Not Church-Encoding)

The UNF is the **native-agent normal form**: the literal reduced data structure, serialized canonically.

Example: the program "the number that is two plus two" reduces to:

```
Add64(Lit(2), Lit(2))  →  Lit(4)
```

UNF serialization: `"Lit(Int64(4))"` → SHA-256 hash.

Two sentences that compute the same value produce the same UNF hash. Two sentences that compute different values produce different hashes.

**What this does NOT prove:** That two different *algorithms* computing the same function are equivalent. (Rice's theorem: undecidable in general.) This proves **intensional** equivalence only.

### 10.2 Canonical Serialization Algorithm

```rust
fn canonical_serialize(net: &Net) -> Vec<u8> {
    let mut output = Vec::new();
    serialize_port(net, net.root, &mut output);
    output
}

fn serialize_port(net: &Net, port: Port, out: &mut Vec<u8>) {
    let node = &net.nodes[port.node];
    match &node.kind {
        AgentKind::Lit(Literal::Int64(n)) => {
            out.push(0x01);  // tag for Int64
            out.extend_from_slice(&n.to_le_bytes());
        }
        AgentKind::Lit(Literal::Bool(b)) => {
            out.push(0x02);
            out.push(if *b { 1 } else { 0 });
        }
        AgentKind::Con(tag, arity) => {
            out.push(0x03);
            out.extend_from_slice(&tag.to_le_bytes());
            out.push(*arity);
            // Serialize aux ports in slot order (deterministic)
            for s in 1..=*arity {
                serialize_port(net, net.get_aux(port.node, s), out);
            }
        }
        _ => {
            // Should not happen after full reduction of a well-typed program.
            out.push(0xFF);  // STUCK marker
        }
    }
}

fn unf_hash(net: &Net) -> [u8; 32] {
    let bytes = canonical_serialize(net);
    sha256(&bytes)
}
```

### 10.3 Arithmetic Simplification (Optional, Post-UNF)

A small term-rewriting pass on the flat arithmetic DAG, **restricted to one algebra at a time**:

- Constant folding: `Add64(Lit(2), Lit(3))` → `Lit(5)`
- Identity: `Add64(x, Lit(0))` → `x`; `Mul64(x, Lit(1))` → `x`
- Annihilation: `Mul64(x, Lit(0))` → `Lit(0)`
- Idempotence (boolean): `And(x, x)` → `x`; `Or(x, x)` → `x`

**Do NOT mix arithmetic and bitwise simplification in the same pass.** If cross-algebra equivalence is ever needed, use an external SMT solver (Z3/CVC5 with bitvector theory) as an oracle.

### Acceptance Criteria (part of M4/M5)

- [ ] UNF hash is byte-identical for two different L0 sentences that compute the same value (e.g., "two plus two" vs. "one plus three").
- [ ] UNF hash is different for two sentences that compute different values.
- [ ] Arithmetic simplification correctly reduces the identities listed above, verified on a dedicated test set.

---

## 11. Phase 8 — L1 Probabilistic Layer

**Goal:** Handle epistemic/frequency/vague vocabulary without making the deterministic core non-deterministic.

### 11.1 Design

L1 is a **compile-time expansion**, not a runtime feature. An L1 sentence is expanded into a weighted set of L0 sentences ("worlds"), each compiled and evaluated independently.

### 11.2 Trigger Table

| Trigger word | CCG category | Splitting rule |
|-------------|-------------|---------------|
| `probably` | `S/S` | `[(0.8, S), (0.2, Negate(S))]` |
| `might` | `(S\NP)/(S\NP)` | `[(0.5, VP), (0.5, NullAction)]` |
| `or` (stochastic) | `(X\X)/X` | `[(0.5, X1), (0.5, X2)]` |
| `usually` | `S/S` | `[(0.9, S), (0.1, Negate(S))]` |
| `big` (vague adj, modifying `cat`) | `N/N` | `[(0.48, Lion), (0.48, Tiger), (0.04, LargeHouseCat)]` |

The splitting table is **hand-authored**, not learned. Each entry is a small, explicit mapping.

### 11.3 Splitting Compiler

```rust
fn split_l1(tree: &CCGTree, triggers: &TriggerTable) -> Vec<(f64, CCGTree)> {
    // Walk the tree. At each trigger node, fork into all combinations.
    // Multiply probabilities along each path.
    // Return a list of (probability, deterministic_L0_tree) pairs.
}
```

### 11.4 Blow-Up Control (Critical)

The original brainstorm named this the "Exploding Worlds Paradigm" as a positive, without bounding it. **This is a bug, not a feature.**

- **Hard cap:** At most **4 L1 triggers per sentence** in v1. Sentences exceeding this are rejected at the Harper gate with an actionable message.
- **Structural hash-consing:** Identical Core IR fragments across worlds (very common, since most triggers affect only a small part of the sentence) are compiled and evaluated **once**, not once per world.

### 11.5 Aggregation

To answer a query against an L1 sentence's worlds:

```rust
fn answer_query(worlds: &[(f64, UNF)], query: &UNF) -> f64 {
    worlds.iter()
        .filter(|(_, unf)| unf == query)
        .map(|(p, _)| p)
        .sum()
}
```

### Acceptance Criteria (Milestone M6)

- [ ] A ≥30-sentence L1 test corpus exists, with hand-computed gold probability distributions.
- [ ] Gold distributions are reproduced within floating-point tolerance (1e-9).
- [ ] Per-sentence world probabilities sum to 1.0 within tolerance.
- [ ] Sentences exceeding the 4-trigger cap are rejected with an actionable message (verified by test).

---

## 12. Phase 9 — Hybrid Execution (Post-v1)

**This phase is explicitly post-v1.** Do not begin until Milestones M0–M6 are all green.

### 12.1 Concept

Mark certain functions in the source as `pragma Sequential`. These compile to Cranelift machine code and are embedded as **Native Agents** inside the Delta-Net graph. The Delta-Net handles parallelism, data routing, and duplication; the Native Agent handles dense sequential computation.

### 12.2 Strictness Boundary

A Native Agent **cannot execute until all its inputs are fully reduced to primitive values** (unboxed `i64`, `bool`, or contiguous memory arrays). The interaction rule for a Native Agent is:

1. **Wait:** Do not reduce this active pair until all input ports are connected to `Lit` agents.
2. **Unbox:** Read primitive values from the graph.
3. **Execute:** Call the Cranelift-compiled function pointer.
4. **Box:** Write the result back as a new `Lit` or `Con` agent.

### 12.3 Why This Preserves UNF

The Native Agent is a **pure function** (no side effects, guaranteed to terminate because the source is total). To the Delta-Net, it is just a deterministic rewrite rule. The UNF is unaffected by whether the rewrite was performed by graph-rewriting or by machine code.

### 12.4 Acceptance Criteria (Milestone M7, post-v1)

- [ ] A program with a `pragma Sequential` region produces the same UNF hash as the same program without the pragma.
- [ ] The sequential region executes at least 5× faster than the pure graph-rewriting path on a compute-heavy benchmark (e.g., matrix multiplication).

---

## 13. Testing Strategy

### 13.1 Golden-File Tests

Every stage's output (CCG trees, Core IR, Delta-Net readback, Austral output, UNF hashes) is checked into the repo alongside the corpus. CI runs all golden-file tests on every commit.

### 13.2 Differential Testing (The M5 Oracle)

The three-way agreement test (gold / Delta-Net / Austral) is the project's single most important test. It runs in CI on every commit. A failure blocks merging.

### 13.3 Fuzzing (Post-M5)

Once M0–M5 are green, add:
- A random generator of L0-grammar-conforming sentences (from the lexicon and grammar).
- A random generator of well-typed Core IR programs within the total/terminating fragment.
- Run the three-way differential check against both.

### 13.4 No Performance Work Without Passing Correctness

Do not optimize the Delta-Net reducer or the Austral backend until the differential test suite passes.

---

## 14. Milestone Checklist

| Milestone | Phase | Size | Prereq | Key Acceptance Criterion |
|-----------|-------|------|--------|------------------------|
| **M0** | 1 | S | none | Corpus + gold files committed |
| **M1** | 2 | S | M0 | 100% accept/reject on grammatical/ungrammatical corpora |
| **M2** | 3 | M | M1 | 100% exact match against gold CCG trees |
| **M3** | 4 | M | M2 | 100% of corpus passes linearity checker |
| **M4** | 5 | M/L | M3 | Δ-Net readback matches gold for 100% of corpus |
| **M5** | 6 | M/L | M3 | **Three-way differential test passes on 100% of corpus** |
| **M6** | 8 | M | M5 | L1 gold distributions reproduced within 1e-9 |
| **M7** | 9 | M/L | M6 | Hybrid execution: same UNF, ≥5× speedup on benchmark |

Sizes: S = small, M = medium, L = large. These are effort estimates, not calendar deadlines.

---

## 15. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CCG parse ambiguity blow-up | Exponential parse times | Keep lexicon small; log parse count per sentence in CI; treat spikes as regressions |
| Δ-Net reduction non-termination | Infinite loop | Cannot happen for well-typed Core IR (totality). If it happens, it's a P0 bug in the type system or the Fold compilation. |
| Austral closure conversion bugs | Wrong results | Differential test (M5) catches these immediately |
| UNF hash collision | False equivalence | Use SHA-256; collision probability is negligible. For extra safety, compare the full canonical byte string, not just the hash. |
| L1 world explosion | Exponential compile/eval time | Hard cap of 4 triggers; structural hash-consing |
| Scope creep | Project never ships | §2 Non-Goals is binding. No new module without a milestone and acceptance test. |
| Rice's theorem misunderstanding | Users expect general equivalence checking | State limitation explicitly in all documentation and CLI output |
| Harper POS mismatch | Silent mis-parses | Measure agreement at M1; use as soft signal only |

---

## 16. Repository Layout

```
logos/
├── docs/
│   ├── L0_GRAMMAR.md
│   ├── LEXICON_GUIDE.md
│   ├── DECISIONS.md
│   └── METRICS.md
├── corpus/
│   ├── lexicon.tsv
│   ├── l0_seed/              # 200 sentences + gold CCG trees + gold results
│   ├── l0_ungrammatical/     # 50 sentences that must be rejected
│   └── l1_seed/              # ≥30 sentences + gold probability distributions
├── src/
│   ├── lexicon/              # lexicon.tsv loader + types
│   ├── harper_gate/          # Harper integration + tokenizer
│   ├── ccg/                  # chart parser
│   ├── core_ir/              # Core IR types, linearity checker, CCG→IR compiler
│   ├── deltanet/             # agent/net/reducer/readback/UNF
│   ├── austral_codegen/      # Core IR → Austral emitter
│   ├── l1/                   # possible-worlds splitting compiler
│   └── cli/                  # `logos` binary: parse/run/verify/hash subcommands
├── tests/
│   ├── differential/         # three-way agreement tests
│   ├── fuzz/                 # random sentence + Core IR generators
│   └── unit/                 # per-module unit tests
├── Cargo.toml
└── README.md
```

---

*End of plan.*