# logos

"Project Logos" — a controlled natural language (CNL) compiler to verified
execution graphs. Phase sequence: parse → compile → reduce → readback →
hash. `l1` + `lexicon` define the CNL subset, `ccg` the combinatory-
categorial parser, `core_ir`/`deltanet` the execution-graph IR and reducer,
`harper_gate` the verification gate, `austral_codegen` the Austral backend,
and `cli` the CLI driver. See `docs/LOGOS.md` for the full architecture
and supported grammar.