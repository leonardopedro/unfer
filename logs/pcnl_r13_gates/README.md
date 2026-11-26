# R13 gate evidence logs

Cited by `docs/PCNL_STATUS.md` ("Legacy-gate reproducibility after the init
change"). These files live here rather than in `/tmp` because `/tmp` was cleared
mid-session on 2026-11-24 and took several cited logs with it; `logs/*` is
otherwise git-ignored, so the two negation lines in `.gitignore` are what keep
this directory trackable.

`scripts/check-pcnl-gates` re-runs every command below and diffs the result
against the committed log (`posterior`, `calibration`, `chain_diagnostics`,
`diag_observable`, `summary`, `run=`, `arm=` lines), so a log that stops
matching fails loudly:

```
./scripts/check-pcnl-gates              # every gate under 900 s
./scripts/check-pcnl-gates --all        # includes hit-and-run 5,000
./scripts/check-pcnl-gates --quick      # per-push: the three *-canary rows (~94 s)
./scripts/check-pcnl-gates qk-gate      # one gate by name
```

Every run is from the repository root, against
`./target/release/examples/vector_regression` (or `vector_basins` /
`vector_stream`), fixture `docs/fixtures/ultrabert_20ng4_embeddings.safetensors`,
`teacher_seed=20261122`, with output redirected via `> <file> 2>&1`.

| file | command | recorded result |
|---|---|---|
| `readout-exact.log` | `vector_regression --readout-only --kernel hmc --init mean --hmc-mass exact --hmc-leapfrog 4 --steps 1000 --burn 250 --thin 1 --draws 750 --chains 4` | R̂ 1.00935, ESS 3000.0, tail 906.4, RMSE 0.105302 |
| `pilot.log` | `vector_regression --kernel hmc --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass pilot --pilot-steps 600 --pilot-rho 0.99999 --pilot-inflate 1 --hmc-leapfrog 4 --steps 400 --chains 4 --start-jitter 1e-4 --thin 1 --draws 300` | R̂ 2.61047, ESS 4.7, RMSE 0.144261 |
| `hessian-64.log` | `vector_regression --kernel hmc --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass hessdiag --hessian-probes 64 --hmc-leapfrog 4 --steps 1000 --burn 250 --thin 1 --draws 750 --chains 4 --start-jitter 1e-4` | R̂ 2.50388, ESS 4.8, RMSE 0.150189, `clipped_to_prior` 27,622 |
| `lowrank-8.log` | `vector_regression --kernel hmc --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass lowrank --lowrank-rank 8 --pilot-steps 600 --pilot-rho 0.99999 --pilot-inflate 1 --hmc-leapfrog 4 --steps 400 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | R̂ 2.39408, ESS 4.9 |
| `lowrank-32.log` | `vector_regression --kernel hmc --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass lowrank --lowrank-rank 32 --pilot-steps 600 --pilot-rho 0.99999 --pilot-inflate 1 --hmc-leapfrog 4 --steps 1000 --burn 250 --thin 1 --draws 750 --chains 4 --start-jitter 1e-4` | R̂ 2.01334, ESS 5.3 |
| `jitter-lf4.log` | `lowrank-32.log`'s command plus `--hmc-jitter` | R̂ 2.14626, ESS 5.1, tail 10.0, RMSE 0.092472 |
| `qk-gate.log` | `vector_regression --kernel hmc --init mode --mode-steps 8000 --mode-lr 5e-3 --hmc-mass hessdiag --hmc-leapfrog 4 --steps 400 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | R̂ 2.60278, ESS 4.7, tail 9.3, RMSE 0.173727 |
| `qk-alt-lf8-lr5e-3.log` | same without `--hmc-leapfrog 4` | R̂ 1.76121, ESS 5.9, RMSE 0.137976 |
| `qk-alt-lf8-lr2e-3.log` | same, default `--mode-lr 2e-3` and default leapfrog | R̂ 2.12514, ESS 5.2, RMSE 0.123945 |
| `hitrun-400.log` | `vector_regression --kernel hitrun --init mode --mode-steps 8000 --mode-lr 5e-3 --steps 400 --hr-width 1 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | R̂ 2.39717, ESS 4.9, RMSE 0.087048 |
| `hitrun-5000.log` | `vector_regression --kernel hitrun --init point --steps 5000 --hr-width 1 --thin 2 --draws 1875 --chains 4` | R̂ 3.03445, ESS 4.5, RMSE 0.121815 |
| `temper-4.log` | `vector_regression --kernel hmc --temper 4 --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass hessdiag --hmc-leapfrog 4 --steps 400 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | swaps 0/240 on all chains; R̂ 2.44665, ESS 4.8, tail 10.0, RMSE 0.168598 (matches the recorded row exactly) |
| `cn-exact-dense.log` | `vector_regression --readout-only --kernel cn --proposal exact --init mean --rho 0.5 --refresh 0.01 --steps 6000 --burn 1500 --thin 1 --draws 4500 --chains 4` | posterior only (`readout_conditional`, dim 34); R̂ 1.00616, ESS 620.9, tail 201.3, RMSE 0.105528; `gate=true` |
| `cn-exact-dense-12k.log` (alt) | the same with `--steps 12000 --burn 3000 --draws 9000` | ESS grows with length: R̂ 1.00423, ESS 1720.5, tail 1107.3, RMSE 0.105396; `gate=true` |
| `cn-exact-dense-24k.log` (alt) | the same with `--steps 24000 --burn 3000 --draws 21000` | ESS keeps scaling: R̂ 1.00274, ESS 3920.6, tail 2144.3, RMSE 0.105362; `gate=true` |
| `basins.log` | `vector_basins --runs 2 --steps 8000` | potentials 28.0901/38.7083, RMSE 0.076978/0.087130 |
| `stream.log` | `vector_stream --prefixes 4 --adam-steps 300 --lr 0.02` | warm coverage 1.0/1.0/0.9917/0.9833, z² 0.032/0.024/0.275/0.356 |
| `stream-canary.log` (canary) | the same with `--adam-steps 50` | per-push drift baseline (6 s): joint test RMSE 0.114816, coverage90 1.0/1.0/1.0/1.0, z² 0.029/0.016/0.038/0.047 |
| `pilot-canary.log` (canary) | `vector_regression --kernel hmc --init mode --mode-steps 300 --mode-lr 5e-3 --start-scale legacy --hmc-mass pilot --pilot-steps 100 --pilot-rho 0.99999 --pilot-inflate 1 --hmc-leapfrog 4 --steps 100 --chains 4 --start-jitter 1e-4 --thin 1 --draws 75` | per-push drift baseline (45 s); full-parameter R̂ 2.60209, RMSE 0.104758, `gate=false` by length |
| `readout-canary.log` (canary) | `vector_regression --readout-only --kernel hmc --init mean --hmc-mass exact --hmc-leapfrog 4 --steps 100 --burn 25 --thin 1 --draws 75 --chains 4` | per-push drift baseline (43 s); R̂ 1.07057, ESS 296.0, RMSE 0.105268, plus the `calibration` line |
| `pilot-calibrated.log` (alt) | `pilot.log`'s command plus `--calibration` | full-parameter calibration contrast: train param 90% 478/480, predictive 480/480, band half-width 0.066869, residual sd 0.018003, df −56,898 → df-adjusted `NaN`; every other summary line identical to `pilot.log` |

Three details worth keeping: hit-and-run takes `--steps` and `--hr-width` (burn is
`steps/4`, there is no `--burn`); `--mode-lr 5e-3` is not the default (the
default `2e-3` leaves the mode at potential 93.87 instead of 33.81); and
**`--start-scale legacy` is not the default** (`prior`) — it decides the probe
base point and the prior clip, so the `pilot`/`hessian`/`lowrank` rows, the
jitter row and the `temper-4` row only reproduce with it, while the later
`qk`/`hitrun` rows reproduce without it. The tempering cold chain was the last
row still unmatched for exactly this reason: with `prior` it gives R̂ 2.15397 /
ESS 5.1 / RMSE 0.096604, and adding `legacy` (keeping the 3000-step, `5e-3`
mode fit rather than an 8000-step one) reproduces 2.44665 / 4.8 / 0.168598.

The CN row needs `--readout-only` before `--proposal exact` / `--init mean`:
the analytic posterior only exists for the readout conditional, so the pairing
is rejected outright ("require `--readout-only`") without it, and the run then
reports `sample_space=readout_conditional` rather than the full parameter. The
`12k` alternate exists to separate "poor mixing" from "short chain": at twice
the draws with the same start and proposal the bulk ESS goes 620.9 → 1720.5 and
the tail ESS 201.3 → 1107.3, so the CN row's ESS was estimator-truncated rather
than a stuck chain. The `24k` row (four times the original, `18,000` draws per
chain) continues the line: bulk ESS `3920.6`, tail `2144.3`, R̂ `1.00274`, point
estimate `0.105362` — 2.28× the 12k ESS for 2× the steps, so the growth is not
plateauing. It measured ~3,100 s here; its budget is 3,200 s.

Only `--all` runs every row. Budgets are runtime estimates and anything
budgeted above 900 s (currently `temper-4` and `hitrun-5000`) is skipped by the
default sweep, which is why the checker now prints a `not selected:` line for
every row it left out, with the flag that would select it. The `alt` rows
(`cn-exact-dense-12k`, `cn-exact-dense-24k` and the two `qk` alternates) need
`--alts`. `--quick` is the per-push lane: it selects only the three `canary`
rows, which twin `stream`, `pilot` and `readout-exact` at a length that fits
the budget (94 s measured idle against 492 s for the full rows; 1,410 s under
load), and prints every row it skipped, so it can never be mistaken for the
full sweep. Canary numbers are a drift baseline, not a gate: their `gate=`
verdict is `false` at that chain length by construction, and the default sweep
skips them because the full rows subsume them.

The scripts are byte-exact by default, which is only true on the machine that
recorded the logs. On GitHub's hosted runner, with no code change, a 50-step
Adam canary moved `test_rmse_mean` by 0.1% and one `z2` by 4.7%, and a 100-step
HMC canary moved `rhat_rank` `2.54906 → 1.91680` with `ess_tail` `13.4 → 22.8`
(at 75 draws those are noise, and one accept/reject flip moves the counts in
`(93/240)`). The per-push lane therefore uses `--structure`: key sequence,
`gate=`/`sample_space=` words and finiteness must match, magnitudes need not.
The scheduled `--all` lane uses `--band 0.2`, since its rows are long enough for
R̂/ESS to mean something. Byte-exact reproduction is still
`./scripts/check-pcnl-gates` on the recording machine, and the scheduled
workflow has a second job (`gates-byte-exact`) that runs exactly that command
with no `--band` on a self-hosted runner carrying the `r13-reference` label.
That job is inert until such a runner is registered and the
`RUN_R13_BYTE_EXACT` repository variable is set — the same gate the CUDA smoke
job uses, so an unregistered label cannot leave a run queued forever.

Readout-only rows also print a `calibration` line, which scores the same 90%
bands on the split the conditional likelihood was fitted on (train) and reports
the residual sd there, its df-adjusted value, and the declared noise. Its
train-residual estimate is ~`0.0044`, well under the declared `0.05`, because
the frozen backbone memorised the train split; the held-out residual is the
operative scale behind the `0.158` coverage. See `docs/PCNL_STATUS.md`,
"Why the readout-conditional 90% coverage is 0.158".

`--calibration` adds the same line to a full-parameter run and is opt-in for
that reason: it is a new summary line, so turning it on by default would
invalidate every committed full-parameter log at once. `pilot-calibrated` is
the contrast row — same train rows, same 90% band, all 57,378 coordinates
marginalised — and its band is 3.5× wider (`0.0669` vs `0.0190`). Its `df` is
negative (480 observations, 57,378 parameters), so the df-adjusted residual sd
is `NaN`: a full-parameter run has no unbiased in-sample noise estimate.

Measured on 2026-11-25 on the local 12-core machine: `--all` = 13 rows /
6,645 s / ~1 h 55 m, `--all --alts` = 16 rows / 8,349 s / ~2 h 20 m, both
reproducing every summary line (exit 0); after the canaries landed `--all` is
16 rows and adds their 94 s (~6,740 s expected — not re-measured end to end),
and `--all --alts` is 21 rows adding the same 94 s plus `pilot-calibrated`'s
196 s. The manifest now has 21 entries (13
gates, five alternates — `cn-exact-dense-12k`, `cn-exact-dense-24k`, the two
`qk` alternates and `pilot-calibrated` — and the three canaries); the 24k row
alone is ~3,100 s, so a full `--all --alts` sweep is
now above the recorded 2 h 20 m. Those figures are what the scheduled lane's
`timeout-minutes` is sized against; the per-push lane is `--quick` instead.
