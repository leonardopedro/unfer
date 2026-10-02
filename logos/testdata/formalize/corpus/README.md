# formalize evaluation corpus

`sample_graphs.json` — 15 real human proof graphs, copied from ProofFlow.

## Provenance

Extracted from `/home/leo/Projects/ProofFlow/data/benchmark_0409.json`, which
holds 184 graphs. The 15 here are the **first graph of each distinct node
count**, 2 through 16, so the sample spans the whole size distribution of the
full corpus (median 8) rather than being 15 arbitrary small ones.

Regenerate with:

```sh
node -e '
const fs=require("fs");
const d=JSON.parse(fs.readFileSync("/home/leo/Projects/ProofFlow/data/benchmark_0409.json","utf8"));
const seen=new Set(); const picked=[];
for (const g of d) { const n=g.proof_graph.length; if (!seen.has(n)) { seen.add(n); picked.push(g); } }
picked.sort((a,b)=>a.proof_graph.length-b.proof_graph.length);
fs.writeFileSync("sample_graphs.json", JSON.stringify(picked.map(g=>({
  origin:g.origin, id:g.id, nl_theorem:g.nl_theorem, nl_proof:g.nl_proof, proof_graph:g.proof_graph
})),null,1));
'
```

## Licensing

ProofFlow is **MIT**, and this is a copy of a substantial part of its data, so
its notice ships here in full:

```
MIT License

Copyright (c) 2024 ProofFlow Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## What these graphs are, and are not

They are **human DAGs**: ProofFlow's graph-builder output on 184 benchmark
proofs, committed to its `data/` directory. So they are ground truth for the
*P1 stage* — a port of `proof_graph.py` should reproduce them exactly, and
`logos/tests/eval_corpus.rs` asserts it does, on every graph in the file.

They are **not** ground truth for the CNL stage: they carry no CNL sentences,
because ProofFlow's target was Lean and the graphs' `lean_hint` fields are Lean
tactic plans. Producing CNL targets is P8's other half, and §17.1 says the
honest answer is that stock L0 cannot formalize most of these steps — which is
exactly what `lexicon_coverage_is_the_measured_limit` measures rather than
assumes.