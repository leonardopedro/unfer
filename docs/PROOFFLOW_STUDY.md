# ProofFlow study note — S4-1

**Date:** 2026-10-14 · **Staged by:** `PLAN_S4_BACKLOG_2026-10-14.md` §S4-1
· **Source:** `../ProofFlow/` (Huawei-AI4Math/ProofFlow, ICLR 2026,
arXiv 2510.15981), **MIT** — `LICENSE` read 2026-10-14, "Copyright (c) 2024
ProofFlow Team".

## Verdict first

The question S3 §6 queued — *"is ProofFlow's dependency-graph pipeline a
candidate discipline for `logos`' formalize pipeline, and may a non-Lean agent
implement the Rust-side orchestration?"* — has a sharper answer than a
go/no-go: **the orchestration is already implemented.** `logos::formalize` *is*
the ProofFlow adaptation (`docs/FORMALIZE.md`, rewrite plan §11–§17 in
`emthin/docs/REWRITE_PLAN.md`): a module-for-module port, parity-locked
against the Python oracle, with the verification substrate deliberately
retargeted from a Lean server to deltanet reduction. There is nothing left for
the specialist's output contract to gate *in the orchestration itself*.

What this note does: (a) the mechanism mapping, (b) what was deliberately not
taken and why, (c) go/no-go on the remaining delta — plus one compliance
finding fixed here.

## (a) Mechanism mapping — what already landed

The full mapping table and porting rationale live in `FORMALIZE.md` §1 and the
module headers; summarised, with the ProofFlow original named:

| ProofFlow (`proofflow/`) | logos counterpart | status |
|---|---|---|
| `proof_graph.py` (316 ln) — graph build + DAG rules | `formalize/graph.rs` | ported, **parity-locked** (`eval_corpus`: 184 graphs, 173 validate, 11 rejections are real DAG violations in the source data) |
| `proof_formalize.py` (201 ln) — per-node generation | `formalize/formalizer.rs` | ported, retargeted (CNL sentences; no `lean_pass` needed — there is no Lean output) |
| `proof_prover.py` (138 ln) — solver/tactic completion | `formalize/completer.rs` | ported |
| `proof_scorer.py` (699 ln) — ProofScore | `formalize/score.rs` | ported; two judgements pinned rather than tidied (documented in `FORMALIZE.md` §6) |
| `utils.py` `LLMManager` (446 ln) | `formalize/llm.rs` | own implementation (OpenAI chat schema; see contract in `FORMALIZE.md` §7) |
| `vis.py` (854 ln) interactive/static HTML | `--vis dag.html` self-contained HTML + the in-document DAG figure (`FORMALIZE.md` §10) | own implementation; ProofFlow's Plotly surface not taken |

Node identity substitution, already on record: ProofFlow keys a step by its
name string (`l4`); here two CNL sentences denotating the same normal form
*are* the same node (UNF hash), which is what makes the lemma store and the
readback round-trip content-addressed rather than string-matched.

## (b) What was not adopted, and why

| ProofFlow mechanism | verdict | reason |
|---|---|---|
| `lean_check.py` (396 ln) `LeanServer` | **substituted, permanently** | logos verifies L0 CNL by interaction-net reduction, not a type checker. The substitution is the whole point of the adaptation and is documented from the first line of `FORMALIZE.md`. Revisit only if Lean output re-enters the pipeline — that path *is* inside the Lean4-specialist boundary. |
| **per-stage model managers** (ProofFlow passes separate graph/formalize/solver/score `LLMManager`s) | **candidate — adopt** | `formalize/llm.rs` uses one `OPENAI_MODEL` for every stage (`Config::from_env`). Per-stage models are the reason ProofFlow's class supports cheap-graph/expensive-solver splits. Rust-only, output contract unchanged (same request/response schema per call): implementable without the specialist. Small: one env var (`OPENAI_MODEL_<STAGE>`) + config plumbing. |
| `llm_call_logs` (every LLM API call logged) | **candidate — adopt** | `llm.rs` reports per-call errors but keeps no transcript. §12's open item ("a real model run") needs exactly this to be diagnosable post-hoc. Rust-only, no contract change. |
| ProofFlowBench + `benchmark.sh` (pass@5 over think/no-think × DAG/no-DAG) | **defer** | There has been no live-model run at all (`FORMALIZE.md` §12). A benchmark harness before the first run measures nothing. Reopen after a live run exists; the HF dataset (`ProofFlowBench`) is then a ready-made external-validity corpus for `eval_corpus`. |
| `io.py` pickle/`benchmark_results` formats, `cli.py` | **not adopted** | formats for their own harness; logos reads proof-graph JSON / document input and has its own CLI. No benefit to mirroring file formats. |

## (c) Go / no-go

1. **Rust-side orchestration: moot — already implemented** (see verdict).
   Nothing about the existing pipeline needs the specialist's output contract;
   it was built under §11–§17 and is verified as documented in
   `FORMALIZE.md` §11.
2. **Go (non-Lean agent, small):** the two candidates in (b) — per-stage model
   config and the LLM call log. Both sit behind the existing `llm-http`
   feature, change no output schema, and write no Lean4 code.
3. **No-go on accuracy claims until a live run happens:** `FORMALIZE.md` §4
   predicts very low end-to-end accuracy and §12 says no live run has been
   performed. Nothing in this study upgrades that — a run needs the owner's
   model server (`OPENAI_BASE_URL`), not the specialist. The specialist
   boundary applies only where Lean output re-enters (the `confluence_lean_*`
   test, elan toolchain absent).
4. **No-go on ProofFlowBench adoption now** — see (b); it becomes meaningful
   with the first live run.

## Compliance finding (fixed in this change)

`logos/src/formalize/graph.rs` calls itself "a faithful Rust port of
ProofFlow's `proof_graph.py`", yet **no ProofFlow row existed in either
ledger** — the claim phrases in the code were phrased ("port of",
"adaptation") so `check-attribution` never matched them, which is exactly the
case a human owes the ledger. Policy #4 requires the row when the adaptation
lands; this note adds it to `../ATTRIBUTION.md` (workspace) and
`ATTRIBUTION.md` (this repo), with `LICENSE` read firsthand. MIT, so the port
was always legally clean — the gap was the record, now closed.
