# R13 gate evidence logs

Cited by `docs/PCNL_STATUS.md` ("Legacy-gate reproducibility after the init
change"). These files live here rather than in `/tmp` because `/tmp` was cleared
mid-session on 2026-11-24 and took several cited logs with it; `logs/*` is
otherwise git-ignored, so the two negation lines in `.gitignore` are what keep
this directory trackable.

Every run is from the repository root, against
`./target/release/examples/vector_regression` (or `vector_basins` /
`vector_stream`), fixture `docs/fixtures/ultrabert_20ng4_embeddings.safetensors`,
`teacher_seed=20261122`. Output was redirected with `> <file> 2>&1`.

| file | command | recorded result |
|---|---|---|
| `qk-gate.log` | `vector_regression --kernel hmc --init mode --mode-steps 8000 --mode-lr 5e-3 --hmc-mass hessdiag --hmc-leapfrog 4 --steps 400 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | R̂ 2.60278, ESS 4.7, tail 9.3, RMSE 0.173727 |
| `qk-alt-lf8-lr5e-3.log` | same without `--hmc-leapfrog 4` | R̂ 1.76121, ESS 5.9, RMSE 0.137976 |
| `qk-alt-lf8-lr2e-3.log` | same, default `--mode-lr 2e-3` and default leapfrog | R̂ 2.12514, ESS 5.2, RMSE 0.123945 |
| `hitrun-400.log` | `vector_regression --kernel hitrun --init mode --mode-steps 8000 --mode-lr 5e-3 --steps 400 --hr-width 1 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | R̂ 2.39717, ESS 4.9, RMSE 0.087048 |
| `hitrun-5000.log` | `vector_regression --kernel hitrun --init point --steps 5000 --hr-width 1 --thin 2 --draws 1875 --chains 4` | R̂ 3.03445, ESS 4.5, RMSE 0.121815 |
| `jitter-lf4.log` | `vector_regression --kernel hmc --init mode --mode-steps 3000 --mode-lr 5e-3 --hmc-mass lowrank --lowrank-rank 32 --pilot-steps 600 --pilot-rho 0.99999 --pilot-inflate 1 --hmc-leapfrog 4 --hmc-jitter --steps 1000 --burn 250 --thin 1 --draws 750 --chains 4 --start-jitter 1e-4` | R̂ 2.48617, ESS 4.8 (recorded 2.14626 not reproduced) |
| `jitter-lf8.log` | same without `--hmc-leapfrog 4` | R̂ 2.34143, ESS 4.9 (recorded 2.14626 not reproduced) |
| `basins.log` | `vector_basins --runs 2 --steps 8000` | potentials 28.0901/38.7083, RMSE 0.076978/0.087130 |
| `stream.log` | `vector_stream --prefixes 4 --adam-steps 300 --lr 0.02` | warm coverage 1.0/1.0/0.9917/0.9833, z² 0.032/0.024/0.275/0.356 |
| `temper-4.log` | `vector_regression --kernel hmc --temper 4 --init mode --mode-steps 3000 --mode-lr 5e-3 --hmc-mass hessdiag --steps 400 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | swaps 0/240 on all chains (matches); R̂ 2.15397, ESS 5.1, RMSE 0.096604 (recorded 2.44665/4.8/0.168598, start-dependent) |

Two details worth keeping: hit-and-run takes `--steps` and `--hr-width` (burn is
`steps/4`, there is no `--burn`), and `--mode-lr 5e-3` is not the default — the
default `2e-3` leaves the mode at potential 93.87 instead of 33.81 and changes
every downstream number.
