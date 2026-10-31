//! # mm0web — a webservice for mm0 proofs
//!
//! The [`prove2.me`](https://prove2.me) analogue for **mm0** rather than Lean:
//! a submitted proof is verified by the frozen toolchain this project already
//! trusts (`mm0-rs` elaborates `.mm0`+`.mm1` into a `.mmb`; `mm0-c` re-verifies
//! that `.mmb` against the `.mm0` on stdin), and the verified artifact is
//! published to **Zenodo** as a project-scoped deposit that grows by
//! snapshot + incremental delta — the storage contract
//! `unfer_ffi::zenodo` (`docs/altgit.md`, P11.24) defines.
//!
//! ```text
//! browser ──POST /verify──▶ checker ──▶ Accepted / Rejected (+ toolchain log)
//!         └─POST /publish─▶ store ────▶ Zenodo deposit ──▶ record URL + DOI
//!         └─GET /graph|/search|/stats ▶ derived views of what is stored
//! ```
//!
//! The third line is read-only and derives everything it shows: `/graph`
//! renders one submission's dependency graph (its declarations, what they
//! rest on, the faults in that structure), `/search` finds declarations
//! across the index, `/stats` totals it. None of them can accept, reject or
//! publish — the two checkers remain the only verdict in the service. The
//! graph is this project's reading of the point ProofFlow makes with its
//! `check_DAG` stage (`../../../ProofFlow/`, study in `../../docs/PROOFFLOW_STUDY.md`):
//! a proof is easier to trust when you can see what rests on what. Its rules
//! are the ones already ported in `logos::formalize::graph`, parity-tested
//! from `graph::tests`, and the figure comes from `logos::dagviz` — the same
//! canvas `logos --vis` writes.
//!
//! Three deliberate scope decisions, from the operator's answers when this was
//! scoped:
//!
//! 1. **mm0 in, verify, publish.** No translator chain: `lean2mm0` /
//!    `ledger2mm0` / ProofFlow stay upstream, and this service takes mm0 source.
//! 2. **One deposit per project, deltas inside.** The first push uploads
//!    `snapshot_0.bin` plus `manifest.json`; later pushes upload `delta_N.bin`
//!    and a refreshed manifest, exactly the file convention of
//!    `unfer_ffi/src/zenodo.rs` — including the squash rule (a new snapshot
//!    every `squash_after` deltas).
//! 3. **Open submission**, rate-limited per client address, bound to loopback
//!    by default: the service is meant for a tailnet-only host (see
//!    `../tailscale-ssh-only-oracle-linux-10.md`), not for the open internet.
//!
//! What this crate does *not* claim: the module documented here is the
//! HTTP/verify/publish plumbing. It does not implement a checker, and it does
//! not implement the Zenodo wire protocol itself — it shells out to the
//! binaries and to `curl`, because this workspace builds without `ureq`
//! available offline (see `README.md`, "Why curl").

pub mod audit;
pub mod checker;
pub mod graph;
pub mod http;
pub mod publisher;
pub mod store;
pub mod web;

pub use audit::Audit;
pub use checker::{CheckOutcome, CheckerConfig};
pub use publisher::Publisher;
pub use store::{StoreConfig, ZenodoConfig};
pub use web::{Config, RateLimiter, handle};
