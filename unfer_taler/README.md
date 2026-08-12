# unfer_taler

GNU Taler-style exchange adapter over the unfer certificate ledger.
Implements reserves, two-phase wire gateway, denominations, and the
e-coin withdraw/deposit/peg-out flows, with the
`fiat_in - fiat_out = reserves + merchants + outstanding` audit
(`UK-7101..7107`, Plan R Phase 5). `wire` is the wire-gateway abstraction,
`denom` the denominations table, `exchange` the exchange operations.