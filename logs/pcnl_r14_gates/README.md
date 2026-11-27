# R14 gate logs (recorded 2026-11-27)

Verified output of the R14 mechanics, replayed with the commands below on the
recording host.  The measured numbers, including the negative/stalled
contrasts, are quoted in [`docs/PCNL_STATUS.md`](../../docs/PCNL_STATUS.md)
§R14.  These are **not** the declared multi-seed/full-parameter campaigns.

| log | command | verdict |
|---|---|---|
| `plateau_quadrature.log` | `cargo run --offline --release -p pcnl --example r14_plateau_quadrature` | `gate=true` — three temperatures with `R̂ < 1.01`, bulk ESS `≥ 400`, masses within 3 MCSE |
| `supervised_campaign.log` | `cargo run --offline --release -p pcnl --example r14_supervised_campaign` | reduced-budget P4 pilot; curved-boundary stall and negative distillation contrast reported |
| `rl_campaign.log` | `cargo run --offline --release -p pcnl --example r14_rl_campaign` | reduced-budget non-RLCD P5 pilot; boundary-shell in-region `0/4` reported |
| `supervised_multiseed.log` | `cargo run --offline --release -p pcnl --example r14_supervised_multiseed` | 3 seeds × declared 57 378-parameter architecture; pCNL stall `0/900`, hit-and-run `240/240` with 179/180 in-region draws, paired intervals recorded |
| `rl_multiseed.log` | `cargo run --offline --release -p pcnl --example r14_rl_multiseed` | 3 seeds × `M = 3` RL policies; pCNL in-region `0.222`, hit-and-run `1.000` (256/270 draws), paired return contrasts recorded |
| `serving_bench.log` | `cargo run --offline --release -p pcnl --example r14_serving_bench` | measured single-query latency and model bytes: P4 student `3.04×` faster and `3.00×` smaller than the 3-member ensemble; P5 student `3.2×` faster, `12.5 %` smaller; not replay-gated (machine-dependent timings) |
| `rl_edge0_multiseed.log` | `cargo run --offline --release -p pcnl --example r14_rl_edge0_multiseed` | 3 families × RLCD/Edge0 checkpoints reloaded from `target/`; ensemble − single `−0.00275 ± 0.00399` (n.s.), pCNL in-region `0.000`, hit-and-run in-region `1.000` |
| `return_aware.log` | `cargo run --offline --release -p pcnl --example r14_return_aware` | return-aware hit-and-run: in-region `1.000`, `+2.2412 ± 0.7961` nominal and `+3.4798 ± 0.9910` shifted vs plain hit-and-run; diversity `0.334 → 0.174` |
| `heterogeneous.log` | `cargo run --offline --release -p pcnl --example r14_heterogeneous` | P6 scoped evaluation under `docs/PCNL_P6_REVIEW.md`; 12/17/19 vs 16/16/16 at matched params, negative result (in-region 0.778, rank 1.812, latency +3.5 %), promotion blocked |

Deterministic module checks: `cargo test --release -p pcnl --test r14` (18
tests).  Full crate suite: `cargo test --release -p pcnl --locked` (354 tests, all
passing on 2026-11-27).

The examples are seeded, but the committed numbers are the recording host's.
`scripts/check-pcnl-r14-gates` replays four rows and compares their logs
(`--check-manifest`, `--band R`, `--structure`): the fast RL multiseed row is
gated on every push, and the ~2.5 min supervised multiseed row is replayed
manually.  All rows run on the scheduled lane (`--band 0.2` in
`pcnl-gates.yml`).  `scripts/check-pcnl-gates` owns R13 only.
