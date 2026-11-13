# `logos mm0` — the math fragment, certified by MM0

A sentence of a small controlled natural language becomes an MM0 theorem, and
the *shape of the certificate* is the answer:

| sentence | certificate | answer |
| --- | --- | --- |
| `two plus three equals five` | `theorem … : $ teq L R $` with a proof | **yes** (exit 0) |
| `is two plus three six?` | `theorem … (h: $ teq L R $): $ tfals $` | **no** (exit 2) |
| `for all x: x plus zero equals x` | a chain of recursion-law rewrites, checked by mm0 | **proved** (exit 0) |
| `for all x, y: x plus y equals y plus x` | nothing | **unproved** (exit 1) |

There is no fourth case: the pipeline never answers "probably". An LLM can
propose the intermediate rule instantiations that an open claim needs, but its
proposals enter the artifact only through mm0's elaborator — they are candidates
until MM0 says otherwise.

## The chain

```text
sentence
  │  logos::mm0::cnl           grammar-checked claim (one reading per sentence)
  ▼
Claim { lhs, rhs, question }
  │  logos::mm0::bridge        CoreIR → DeltaNet net → reducer → readback
  │                           (the *existing* lowering: deltanet::compile_to_net,
  │                            deltanet::reduce, deltanet::sym_readback)
  ▼                           value + unf_hash + isomorphism witness π
logos::mm0::proof             net encoding, Peano encoding, proof terms
  │
  ▼
logos::mm0::emit              <theorem>.mm0 (statements) + <theorem>.mm1 (proofs)
  │  mm0-rs compile           → <theorem>.mmb
  ▼  mm0-c <mm0                → the root of trust's verdict
answer
```

Two machines compute the same number, and they must agree. The DeltaNet reducer
folds a ground `Prim` natively in one interaction (`reducer::eval_prim`); the
MM0 proof re-derives it from Peano recursion with no trusted evaluator. If they
disagree, `bridge::corroborate` refuses to certify the claim — a certificate
whose number came from a broken reducer would be a lie about which machine
produced it.

## The fragment

```text
claim    := ["for all" var {"," var} (":" | ",")] body
body     := expr eq expr
eq       := "equals" | "=" | "is" | "is equal to" | "equal" ["to"]
expr     := term {("plus" | "+" | "and") term}
term     := atom {("times" | "*" | "multiplied by") atom}
atom     := number | var | "suc" atom | ["the"] "successor" ["of"] atom
          | "(" expr ")"
question := body "?" | "is" body | "does" expr eq expr
```

Numbers are decimal (`30`) or words (`thirty`, `twenty five`). `*` binds
tighter than `+`; parentheses override. A trailing `?`, or a leading `is`/`does`,
makes the sentence a question — the certificate is identical, only the reporting
differs. Everything outside this grammar is refused with the offending token
named, because a sentence that *nearly* parses is one that would be certified
about the wrong claim.

`suc a` is a *constructor* taking one atom, spelled three ways (`suc a`, `the
successor of a`, `successor of a`): `suc a plus b` is `(suc a) + b`, and `suc (a
plus b)` is the successor of the sum — exactly the precedence an MM0 Peano
development gives `suc`, which is why the upstream theorems below can be stated
in the CNL unchanged. At the net level `suc e` is `e + 1` (DeltaNet has no
successor agent, and `Add64(e, 1)` is the agent that computes it); at the Peano
level it is the numeral constructor `ts e`.

`for all x, y:` is accepted for readability and then checked against the
variables the body actually uses: a quantifier that quantifies nothing is an
error, not a silently unused binder, and a name listed twice (`for all a, a:`)
is refused rather than collapsed. It is not *needed* — a free variable is
universally quantified by the theorem's MM0 binders — which is why the two
spellings of a claim are the same certificate.

## The theory (`logos/mm0/logos_math.mm0`)

Three layers, each with one job.

1. **The agent layer** — `tapp`, `tprim_add`, `tprim_mul`, `tz`, `ts`. This is
   the DeltaNet encoding itself: `Prim(Add64)` with a literal on each auxiliary
   port is `tapp (tapp tprim_add a) b`, the shape `deltanet::unf::
   canonical_serialize` walks and the reducer rewrites. `ts^n tz` is a
   `Lit(Int64 n)` payload as a Peano numeral — MM0 has no decimal literals
   (mm0-rs rejects a bare `1` in a math string unless a numeral token is declared
   for it, as `australVM/mm0/gap_certificate.mm0` does), and structural numerals
   are what the refutation rule needs anyway.
2. **The bridge** — `prim_add_denote`, `prim_mul_denote` say what the agents
   *mean*. This is the only place the encoding is interpreted, and it is a
   definition, not a fact about particular numbers.
3. **Arithmetic and absurdity** — `add_z`/`add_s`/`mul_z`/`mul_s` (Peano
   recursion) and `s_inj`/`z_ne_s`/`fals_elim`. `tfals` is inhabited only by
   `z_ne_s`, i.e. only if `tz` and `ts _` were equal, so a `tfals` proof really
   is a refutation and "no" is an answer rather than a failure to prove.

The theory has a standard model (the term algebra of finite numerals), which is
what makes a refutation sound. It has **no induction rule** — that is the honest
limit of the fragment, and the reason `x + y = y + x` comes back unproved.

## Proving an open claim

A claim with a variable has no value, so it is proved by **rewriting both sides
to normal form** (`proof::rewrite_claim`):

```text
teq (net L) (net R)
  ← bridge_l          teq (net L) (peano L)
  ← L normalises      teq (peano L) nf        add_z | add_s | mul_z | mul_s,
                                               under eq_add_l/eq_add_r/…
  → R normalises back                        teq nf (peano R)   (eq_sym)
  → bridge_r                                 teq (peano R) (net R)  (eq_sym)
```

The rewrite system is the theory's own recursion, applied where the axioms
apply — on the *second* operand — with congruence for the subterms. It is
left-linear, non-overlapping and terminating (each step consumes a constructor
of that operand), so it is convergent: the normal form exists, is unique, and two
terms are equal in the congruence the laws generate exactly when their normal
forms coincide. For the equalities this fragment states, that makes the rewrite
step a *decision*, not a heuristic. A claim the system cannot join is then
offered to the theory's implication-free rules one at a time, mm0-rs deciding
each candidate; if none typechecks, the verdict is `unproved` and the reason
names both normal forms, e.g.

```text
the left side rewrites to `(tadd v_x v_y)`, the right to `(tadd v_y v_x)`
(no candidate typechecked: …). Closing that gap needs a rule this theory does
not have — induction is the usual one — so the claim is left as a goal …
```

That message is the fragment's boundary stated as data, not as a shrug.

Style note, inherited from `australVM/docs/cic.mm0`: premises are stated with
`>` as *hypothesis binders*, never as `->` inside one formula. MM0's logic has no
internal modus ponens, so an `->`-shaped axiom can be proved but never applied.

## Certificates

A closed true claim:

```text
theorem logos_462dbcf1b6537a1f: $ teq (tapp (tapp tprim_add (ts (ts tz))) …) (ts (ts (ts (ts (ts tz))))) $;
```
with a proof that is a chain of `eq_trans`/`eq_app`/`eq_sym` over
`prim_add_denote`, `add_s`, `add_z`, `eq_ts` — i.e. the net's own reduction,
step for step, in the theory's rules.

A closed false claim is stated in the other direction and proved by peeling:
`eq_sym` the hypothesis across both normalisations, `s_inj` off the shared
successors, and let `z_ne_s` refuse the residue.

Two details of the emitted files are load-bearing and were found the hard way;
both are commented at their definitions in `src/mm0/emit.rs`:

* the `.mm0` theorem statement ends with `;` — mm0-rs tolerates its absence, the
  C checker walks off the end of the claim without it;
* the `.mm1` theorem is `pub` — a private theorem's proof reaches the `.mmb` but
  its *declaration* does not, and `mm0-c` then reports `expecting a theorem`.
* the checkers are spawned with a raised stack (`emit.rs::tool_cmd`): `99 times
  99` builds the numeral `ts^9801 tz`, deep enough that mm0-rs recursed through
  the default 8MB stack and aborted with `thread 'main' has overflowed its
  stack` before any verdict could be printed — the very same artifact verifies
  once the child has room, so the tool is `exec`'d through `ulimit -s` (argv,
  stdin and the exit code unchanged), while a *missing* tool still fails as a
  spawn error rather than reading as a rejection. The corpus keeps one large
  line (`99 * 99 = 9801` and its refutation) so the boundary stays tested.

An unproved claim writes `<theorem>.goal` instead: a statement without a proof is
a `sorry`, `mm0-c` rejects those (correctly), and calling the file a certificate
would misrepresent it. The `.goal` file is exactly what the LLM loop below needs.

## CLI

```bash
logos mm0 theory [--out FILE] [--kopperman] [--mehler]  # the theory, byte for byte
logos mm0 ask "<sentence>" [options] # certify one sentence
logos mm0 claims <FILE> [options]    # certify a fixture, check its expectations
logos mm0 check <FILE.mm0 | DIR>     # mm0-rs + mm0-c over written artifacts

ask/claims options:
  --out DIR      write <theorem>.mm0 / .mm1 (or .goal)
  --verify       compile with mm0-rs and re-check with mm0-c
  --json         one JSON object instead of prose (per claim in `claims`)
  --hint TERM    try TERM (an MM0 proof term) for an open claim; repeatable
  --llm          ask the LLM for hint terms (needs `--features llm-http`)
  --mehler       use the Kopperman/Mehler tail-determination theory
                 (a `theory: mehler` section in a claims file does the same)
  --kernel-certificate FILE.mm0
                 decide `theory: kernel` lines by citing that kernel certificate
  --kernel-obligations FILE.json
                 read this sidecar instead of the one the certificate pins

exit codes: 0 yes (true/proved), 2 no (refuted), 1 unproved or error
```

Fixture lines carry their expectation, so a corpus run fails if any line reaches
a different verdict:

```text
true: two plus three equals five
false: is two plus three six?
proved: for all x: x plus zero equals x
unproved: for all x, y: x plus y equals y plus x
```

A `theory:` line changes the target for the claims below it: `base`,
`mehler`, or `kernel`.

## Kernel content: deciding by citation

`theory: kernel` switches to [`kernel`](../logos/src/mm0/kernel.rs), whose
subject is a **symbolic kernel certificate** — the artifacts the unfer kernel's
prover emits (`australVM/mm0/<stem>_certificate.mm0`). Those theorems state a
numerical scheme's obligations as polynomial identities over the state's
components (`numFlux u u = flux u`, `delta l r + delta r l = 0`, …), which is
content the arithmetic fragment cannot express: they range over state vectors
rather than over numbers.

The verdict is a **citation**, not a fresh proof. A sentence is lowered to the
same polynomials the certificate used (by expanding the kernel's definitions
from the obligations sidecar), and it is `proved` when some theorem of that
certificate states exactly the same identity up to a renaming of the
variables — including its hypothesis, so `delta l r = 0 when l = r` cites the
uniform-state theorem while `delta l r = 0` does not. Nothing here invents a
proof term: the cited theorem is true because `mm0-rs` + `mm0-c` accepted the
certificate, which the gate re-runs.

The citation is *pinned*: the certificate's `-- Source:` line carries the
sha256 of the obligations file it was emitted from, and a sidecar that does not
hash to it is refused rather than believed. What the fragment will not do is
refute: "not an identity" is not "false", and turning one into the other would
be a semantic claim about the numeric domain that this layer has no proof of, so
those sentences come back `unproved` with the reason. `true:`/`false:` are
accordingly not part of this fragment.

```text
theory: kernel
proved: for all u: numerical flux of u and u agrees with flux of u
proved: delta of l and r equals zero when l equals r
unproved: the scheme is second order accurate
```

## The LLM seam

Open claims (`for all x: …`) are first normalised with the recursion laws, which
settles everything those laws justify without any model in the loop. What is
left is offered to *one* instantiation of a theory rule: the emitter tries the
rules in `emit::RULES` plus any `--hint`, hands each candidate to mm0-rs, and
keeps the first that elaborates. MM0 infers the arguments, so a candidate is
just a rule name in a proof term — a hallucinated rule fails to typecheck and is
discarded.

A `--hint` is a *complete* mm0 proof term, which is how the premise-carrying
rules are offered at all: `eq_add_l`, `eq_add_r`, `eq_mul_l`, `eq_mul_r`,
`eq_ts`, `eq_app`, `eq_sym`, `eq_trans`, `s_inj` and `fals_elim` take their
premises as ordinary arguments, so a bare `(eq_mul_l)` has holes nothing fills.
A hint carries the rule's term arguments first, then the proofs of its
premises:

```text
--hint '(eq_mul_l (tadd v_x tz) v_x v_y (add_z v_x))'
```

reads "congruence on the left factor of `·`, with `add_z v_x :
teq (tadd v_x tz) v_x` discharging the premise". Whatever the hint, its fate is
the rules' own — it is *elaborated*, and if it does not prove the goal it is
rejected and **named** in the unproved reason:

```text
no candidate typechecked: (prim_add_denote), (prim_mul_denote), (add_z),
  (add_s), (mul_z), (mul_s), (eq_refl), (eq_mul_l (tadd v_x tz) v_x v_y (add_z v_x))
```

For an equational claim in *this* theory a hint is never the winning step —
the rewriter normalises operands in place, so anything `eq_mul_l` could close
is already closed before candidates run. The form is what matters here: the
induction rule the theory is missing will be premise-carrying too, and that is
the day a proposed hint can change a verdict.

`--llm` asks a model for candidate proof terms (`formalize::llm`, the same
transport `formalize` uses; `OPENAI_BASE_URL`/`OPENAI_MODEL`/`OPENAI_API_KEY`),
then verifies every one. For a claim the emitter cannot close, the `.goal` file
is the goal statement to hand back, and `--hint` is how the answer returns. That
is the loop: **the LLM proposes, MM0 disposes.**

The endpoint is any OpenAI-shaped server, so "the model" can be a hosted key, a
local vLLM, or — through `scripts/llm_proxy` — an out-of-band answerer: the
proxy parks each request in a state dir (`/tmp/llm/req.json` by default,
`LLM_PROXY_STATE` to relocate, port from `argv[1]`), waits for `ans.json`, and
serves it as the reply. That is how an agent or a person answers the *real*
pipeline prompt by hand — write `ans.json` and the seam continues. The same
deferred flow runs in `tests/mm0.rs` against a private state dir and an
ephemeral port, so the transport → parse → judge path has a regression
test with no hosted service in it: the proposals arrive over HTTP and are still
judged only by mm0-rs.

## Checked against existing MM0 theorems

`logos/mm0/upstream.txt` is a second corpus, and its sentences were not written
to fit the fragment: each one is a rewrite of a theorem from MM0's own
Peano development (`peano.mm0`/`peano.mm1`), with the upstream name and statement
quoted in a comment. That is what turns "the fragment works" into a claim with a
witness, and it is where the boundary shows:

| upstream | statement | CNL verdict |
| --- | --- | --- |
| `add02` | `a + 0 = a` | **proved** (`add_z`) |
| `mul02` | `a * 0 = 0` | **proved** (`mul_z`) |
| `addS2` | `a + suc b = suc (a + b)` | **proved** (`add_s`) |
| `mulS2` | `a * suc b = a * b + a` | **proved** (`mul_s`) |
| `add12` | `a + 1 = suc a` | **proved** (`add_s`, `add_z`) |
| `add22` | `a + 2 = suc (suc a)` | **proved** (`add_s`, `add_z`) |
| `add01`, `add11`, `addS1`, `addSass` | `0 + a = a`, `1 + a = suc a`, … | unproved (induction) |
| `addcom`, `addass`, `addlass`, `addrass`, `add4` | commutativity, associativity | unproved (induction) |
| `mul01`, `mul12`, `mul11`, `mul22`, `mul21`, `mulS1`, `mulcom` | the multiplicative versions | unproved (induction) |
| `muladd`, `addmul`, `mulass`, `mullass`, `mulrass` | distributivity, associativity | unproved (induction) |

Every upstream theorem that the equational theory can state *and* settle is
proved, and every one that needs `peano5` is reported unproved with the reason —
no line is claimed by a proof that does not check. The corpus also carries the
*instances*: one ground assignment per upstream theorem (`addcom` at 3, 4;
`muladd` at 2, 3, 4; …), each decided by evaluation, and a set of
misquotations of the same theorems (`3 + 4 = 5 + 3`, `4 + 1 = suc 5`, …), each
refuted. Those exercise the machinery the open claims do not: the isomorphism
witness π over the reduced nets, `s_inj`/`z_ne_s`, and the exit-code contract.

Three things the corpus found and the code now does:

* `suc` was not in the fragment at all, so six of upstream's statements could
  not even be written down. It is now an atom constructor (three spellings),
  lowered as `Add64(e, 1)` — see the grammar above.
* The single-rule prover was a false-negative machine: `a + 2 = suc (suc a)` and
  `(a + 0) + (b + 0) = a + b` were reported unproved because no *one* axiom is
their proof. Rewriting under congruence settles them (§ Proving an open claim).
* `suc a plus b` versus `suc (a plus b)`: the successor takes one atom, so the
  second needs parentheses, exactly as in the MM0 source. Getting this wrong does
  not silently prove the wrong thing — the claim is just not the upstream one.

## The Kopperman layer

The second target theory answers a second kind of sentence: Hilbert-space
claims over `logos/mm0/kopperman.mm0`, an *overlay* concatenated onto
`logos_math.mm0` (`logos mm0 theory --kopperman` dumps the pair; the gate
re-checks it on its own). Both fragments share `Prop`, `Term`, `teq`, the
numerals and the bridge, so a certificate for either is one self-contained
`.mm0`/`.mm1` pair. The prover is based on Kopperman's Hilbert-space axioms as
packaged by the sibling `timepiece` repo (`Kopperman_Tutorial.p.tex`,
`BookProof/Toolbox/PhysMehler.lean`); the theory's own header records the
three layers and what each is for.

```text
sentence ──cnl──► KClaim ──raw──► NF ──renorm──► normal form + proof
                                          │
                logos_math.mm0 + kopperman.mm0 ──► mm0-rs ──► mm0-c
```

**The surface syntax** (`logos/src/mm0/kop.rs`) is a second fragment with its
own words — `the inner product of V and W`, `the coordinate i of V`,
`the vector of c, …`, `the sum of V, W`, `the scalar s times V`,
`the zero vector` — and its own discipline: *formation checking*. The parser is
sort-directed (scalar position parses scalars, vector position parses
vectors), a name may be used at only one sort in one sentence, and a number in
vector position is refused rather than read as a variable. That is the whole
soft-type story, and it lives in the CNL because MM0's `Term` is one sort: the
catcher must be the parser.

**The normaliser** rewrites a term with the K2 recursion rules (`coord_*`,
`vadd_*`, `vscal_*`, `inpr_*`, `tail_*`) and the Peano laws, lifting each step
through the congruence axioms (`eq_tinpr`, `eq_tvcons`, …). The rules are
left-linear, non-overlapping and terminating, so the fragment decides its own
equations — and normal-form *equality* is spelling equality: the goal text is
what mm0 checks, so two structures that spell the same term are the same term.
An inner product with a stuck first argument and a literal second flips
(`inpr_sym`) before expanding, which is what makes `u · tvnil` compute to zero.

**Ground claims** are decided by evaluation, cross-checked against the DeltaNet
reducer on both sides' arithmetic *expansions* (`KExpr::expand` — the inner
product of two literal heads is the Σ of coordinate products, an expression the
existing lowering already compiles and folds). A false one is refuted in the
other direction (`teq L R` implies `tfals`, peeled with `s_inj`/`z_ne_s`), and
a true one records the isomorphism witness π, exactly like the arithmetic
fragment.

**The bounded obvious layer** is what the recursion laws cannot reach. The
catalogue is the theory's *equality* laws the normaliser deliberately does not
apply — commutativity, associativity, distributivity, the five inner-product
laws, and the Peano recursion offered again so a restructured term can peel.
Each law is tried at the root first (one step — the obvious inference), then
anywhere in the term in chains of bounded depth, composed with `eq_trans` and
lifted through the term contexts (the relators `eq_add_l`, `eq_tinpr`, …):
*calculational composition*. Every candidate that meets is **replayed by
mm0-rs** — the search proposes a chain, the elaborator accepts or rejects it,
and only an accepted one becomes a certificate. A claim like
`2 * (u·u) = (u·u) + (u·u)` needs two steps and a re-normalisation between
them; one obvious step is not enough, and a chain that does not elaborate is
not an answer.

**The lexicon** (`kop::LEXICON`) is one table from theory symbol to CNL phrase,
with `None` marking a sealed symbol no sentence may name (the K3 sequence layer
— Kopperman's Warning: one named infinite constant destroys decidability — the
order predicates, and `ttail`, which appears only in normal forms). The parser
consumes the table and `logos/tests/kopperman.rs` checks it against
`kopperman.mm0`'s own `term` declarations *both ways*: a theory symbol cannot
be added without the CNL knowing it, and the lexicon cannot name a symbol the
theory does not declare.

**The LLM seam** speaks this layer's vocabulary: `logos mm0 ask --llm` on a
Kopperman sentence sends `kop::LLM_SYSTEM` — the K2 rule catalogue *and the
Kopperman goal format* (the `goal: $ teq L R $` line the seam sends, the `v_`
prefix on variables, which constructors build a scalar side and which a vector
one, and which vocabularies are sealed) — instead of the arithmetic fragment's
agent-spine rules, and whatever comes back is replayed by mm0-rs like any
`--hint`, with each failed proposal *named in the unproved reason*.
`logos/tests/kopperman.rs` pins both halves without a hosted service: the
catalogue against the theory's own `axiom` declarations (every offered rule
really exists; every overlay axiom that could close a `teq` is offered, the
Prop-valued/hypothesis-bound/sealed ones named as exceptions), and the
transport against the same deferred proxy `tests/mm0.rs` uses. Under `--mehler`
the seam sends `LLM_SYSTEM_MEHLER` — the base catalogue plus the four
determination rules — checked against the extended theory the same way.

**The optional Kopperman/Mehler overlay** (`--mehler`). The base overlay
leaves a variable vector's tail *unknown*: the recursion rules fire only on
literal heads, so `tcoord (ttail u) i` is stuck — the "infinite, unknown
tail" that `timepiece/RandomMap/RandomMap2.lean` makes explicit as
`InnerSpace N = InnerHead N × InnerTail`, with the Mehler/Kopperman prior
over the tail. `logos mm0 theory --mehler` (and `ask --mehler`, or a
`theory: mehler` section in a claims file) concatenates
`logos/mm0/mehler.mm0` on top of the overlay: four axioms that *determine*
the tail — `coord_tail` (the tail reads the parent's coordinates one
position on), `vadd_coord`/`vscal_coord` (the constructors act
coordinatewise for **any** arguments — the general laws whose literal-head
instances are the recursion rules), and `vec_eta` (head atom plus tail
reconstructs the vector, the unit/counit of the `InnerHead × InnerTail`
product). All four are first-order equations, sound in the standard model
where a vector denotes its coordinate function, and none is an induction
principle or quantifies inside a formula — Kopperman's Warning is untouched.
The decision procedure is untouched too: the laws are deliberately *not*
normaliser rewrite rules (overlapping the recursion rules at variable
indices would break the confluence normal-form equality rests on); they
join the bounded obvious layer, each candidate replayed by mm0-rs, and
`LLM_SYSTEM_MEHLER` offers them by name.

What `--mehler` buys: the `2·u + 3·v` case-split goal closes, and coordinate
facts about variable vectors (`the coordinate i of the sum of u, v`, the
scalar-multiplication twin) become certifiable — the certificates embed the
extended theory and cite its axioms, and the gate re-checks that theory on
its own. What it does *not* buy: the vector-equality goals (commutativity,
associativity, distributivity over vectors) need *extensionality* — equal at
every coordinate, a quantifier inside the formula — which stays out of the
fragment by design. `kop_claims.txt` carries both theories in one file via
the `theory: base` / `theory: mehler` directive, so one gate run covers both,
and an ignored directive would fail the mehler expectations immediately.

**What stays a goal** is recorded in `logos/mm0/kop_claims.txt` and counted by
the gate (a corpus that cannot say "I do not know" passes vacuously):

* `u · u = 0` — positive definiteness is an *implication* (`inpr_pos` takes a
  hypothesis binder), so nothing derives it;
* `u + v = v + u` for vectors — not an axiom of the overlay; it follows from
  the coordinatewise laws plus induction, which this theory does not have;
* `2·u + 3·v` against a literal head — expanding an inner product whose second
  argument is a variable reads `tcoord (ttail u)`, and relating that to
  `tcoord u (s u)` needs a case split on `u` the theory has no rule for
  (the base fragment's boundary; the `--mehler` overlay's `coord_tail`
  closes it — same sentence, certified under `theory: mehler`);
* the coordinatewise vector laws — `c·(d·v) = (c·d)·v`, `(u+v)+w = u+(v+w)`,
  `c·(u+v) = c·u + c·v` — true in the standard model, each needing a head
  induction (or a case split on a variable) the theory does not have.

## What is not there yet

* **No induction — in the fragment.** `x + y = y + x`, `x * (y + z) = x*y + x*z`,
  associativity — all true, all unproved *by the pipeline*, and `upstream.txt`
  records exactly which upstream theorems are out of reach and why. Adding
  induction to `logos_math` is the next real step, and it is a change to the
  *theory*: `peano5` needs an object logic with substitution (`wff x`, `[a/x]p`),
  not one more axiom (mm0's parser rejects the short route outright — a binder
  cannot have an arrow type, so there is no way to pass a motive to an
  induction rule). What *can* be done today is committed:
  `logos/mm0/addass.mm1` states associativity — the sentence
  `for all a, b, c: (a plus b) plus c equals a plus (b plus c)`, whose automatic
  translation `logos mm0 ask … --out` writes as `logos_f2488ffcf247ed68.goal` —
  and proves it by importing the upstream development (vendored verbatim under
  `logos/mm0/upstream/`, digama0/mm0, CC0) and applying its existing induction
  proof `'(addass)`. The pipeline's `unproved` verdict is untouched: both
  answers are kept, and neither is weakened to make the other look better.
* **Two operators.** `+` and `*` on numerals and variables. Subtraction,
  division, comparison and lists have no bridge axioms yet; each is a small
  addition (a `tprim_*` constant, a `*_denote` axiom, and — for a recursive one —
  a rewrite pair), which is how `add`/`mul` are done.
* **No `mm0web` publishing.** `unfer/mm0web` can publish and serve MM0, but this
  pipeline stops at verified local artifacts.
* **Variables are terms, not entity agents.** A CNL variable encodes as an MM0
  variable of sort `Term` rather than as `tentity (sym …)`, because an open claim
  must be able to instantiate its binders.

## Where the checks live

| check | where |
| --- | --- |
| fragment parsing, encodings, refusals | `logos/tests/mm0.rs` |
| reducer/arithmetic agreement, witness | `logos/tests/mm0.rs` (`bridge::corroborate`) |
| mm0-rs + mm0-c accept proof and refutation | `logos/tests/mm0.rs` |
| a **tampered** certificate is rejected | `logos/tests/mm0.rs`, `scripts/logos2mm0` |
| the corpus reaches its expected verdicts | `unfer/logos/mm0/claims.txt` + `australVM/scripts/logos2mm0` |
| kernel fragment: parsing, citation, hypothesis sensitivity, refusals | `logos/src/mm0/kernel.rs` (`#[cfg(test)]`) |
| kernel corpus cites the certificate's four obligations, and keeps the accuracy line unproved | `unfer/logos/mm0/claims.txt` (`theory: kernel`) + `australVM/scripts/logos2mm0` |
| a kernel certificate whose pinned sidecar digest drifted is refused | `logos/src/mm0/kernel.rs`, `australVM/scripts/logos2mm0` |
| existing MM0 theorems, rewritten in the CNL, reach theirs | `unfer/logos/mm0/upstream.txt` + `australVM/scripts/logos2mm0` |
| Kopperman fragment: formation checks, verdicts, obvious/calculational layers, lexicon coverage | `logos/tests/kopperman.rs` |
| Kopperman LLM catalogue matches the theory and the goal format; proposals are judged | `logos/tests/kopperman.rs` (the deferred-proxy half needs `llm-http`) |
| the `--mehler` tail determination: laws close their claims, both theories re-check | `logos/tests/kopperman.rs`, `kop_claims.txt` (`theory: mehler`) + `australVM/scripts/logos2mm0` |
| Kopperman corpus reaches its expected verdicts (incl. the goals) | `unfer/logos/mm0/kop_claims.txt` + `australVM/scripts/logos2mm0` |
| the Kopperman overlay alone still compiles and checks | `australVM/scripts/logos2mm0` |
| rewriting closes what the recursion laws justify | `logos/tests/mm0.rs` (no toolchain needed) |
| the theory alone still compiles and checks | `australVM/scripts/logos2mm0` |
| a CNL theorem proved from upstream's induction proof | `logos/mm0/addass.mm1` + `logos/tests/mm0.rs` (mm0-rs) |

`australVM/scripts/verify-invariants` runs `australVM/scripts/logos2mm0`, which
gaps (rather than fails) when the mm0 toolchain or cargo is absent, and fails
when the emitter, the corpus, the exit-code contract, or the tamper anchor
breaks.
