# TODO ledger — reviewed code markers

Policy (PLAN_CROSS_PROJECT_REVIEW_2026-10-12 §B1): code files (`*.rs`,
`*.sh`, `*.py`, `*.toml`) carry **no** `TODO`/`FIXME`/`HACK`/`XXX`
marker unless the file is listed below with the reviewed reason.
Markdown is out of scan scope by policy — historical status prose
records a past state and is not rewritten to satisfy a gate.
`scripts/check-todo-ledger` enforces this; a ledger row whose file
carries no marker anymore is reported `[stale]` and should be removed.

The ledger starts empty because the initial full-tree scan found every
"TODO" in this repo to be markdown history, not code debt.

| file | why markers are allowed |
|---|---|
