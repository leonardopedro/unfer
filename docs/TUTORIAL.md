# unfer — Tutorial

> A hands-on, narrative tour of **unfer** and its modules. Read this first for the guided
> walk-through, then dive into the reference docs (`ARCHITECTURE.md`, `PROTOCOL.md`,
> `MODULE_RECIPE.md`). All paths abbreviate `$ROOT = /media/leo/.../` — the parent folder
> that holds `unfer/` and its sibling checkouts (`australVM/`, `velysterm/`, and the P11
> siblings `pattern/`, `pattern-main/`, `dynamic-arctic/`, `hayagriva/` for integration;
> `zentinel/` as a design reference only — see §7.4).

---

## 0. The north star

The long-term goal is an **internet-connected AI-being** that the user and an AI
**co-inhabit through one shared surface** — like `../pattern` (a persistent agent on Loro +
atproto/bluesky), but with a **math editor** as that surface and a **physics/probability
kernel** as the foundation. The roles:

- **velysterm math editor** — the **main interface, used by both the user and the AI**: a
  shared Loro-CRDT document, not a chat box.
- **unfer** — the **math / AI foundation** (the probability kernel below).
- **australVM** — the **module framework**: how capabilities are defined, integrated, and
  authorized.
- **Nix + cloud-hypervisor** (§7.5) — the **practical way to turn existing software into
  modules** (wrap as a Nix derivation, run in the GPU-shared VM).
- **Loro + atproto/bluesky** — the being's **CRDT memory + internet identity** (via §7.1
  and §7.2).

The five P11 integrations each supply one organ of this being (memory/identity, authority,
citation, network edge, module-from-existing-code). The rest of this tutorial is the
foundation that makes it possible.

## 1. What unfer is

unfer is a **modular probability kernel**: a quantum-field-theory engine repurposed to
**compute probabilities of events**. The substrate is a symbolic Fock-space algebra plus a
GPU/CPU time-evolution solver; on top of it sits a **Born-rule layer** that turns quantum
states into event probabilities. Around that kernel is a **module system** (Austral cells,
authorized per-module) and a **UI / AI-agent interface** (velysterm + an NDJSON agent loop).

The one-sentence mental model:

> **Priors** are an initial quantum state + Hamiltonian; **data** conditions/projects the
> state; an **event's probability** is the squared-amplitude mass of the Fock states that
> match it. Modules and humans both drive the same `Session`.

### The crates (in `unfer/`)

| crate | role |
|-------|------|
| `nested_fock_algebra` | symbolic Fock-space engine; LaTeX → Hamiltonian (`compile_latex`) |
| `fock_sirk` | Shift-Invert Rational Krylov time-evolution solver (CPU default, `cuda` feature) |
| `unfer_protocol` | the **contract**: serde types, `UK-####` codes, repair hints |
| `prob_kernel` | the **Born-rule layer**: `Session`, `EventPredicate`, `condition()`, `bayesian_update()` |
| `unfer_ffi` | handle-based C ABI: the `uk_*()` symbols modules call |
| `qfm` | Tomographic QFM Subspace Recovery pipeline + Quantum Bayesian Updating |

As of **rev 19** the workspace is **201 tests** green on CPU.

---

## 2. The three layers

```
   ┌──────────────────────────────────────────────────────────────┐
   │  UI / AI INTERFACE                                            │
   │  velysterm (Bevy + Typst math editor, Loro)                  │
   │  unfer_agent (NDJSON stdin/stdout for AI agents)            │
   ├──────────────────────────────────────────────────────────────┤
   │  MODULES  (Austral cells, sibling folders)                   │
   │  demo_module · qfm_module · qfm_tomo_module · data_source    │
   │  bayes_update_module · iterated_bayes_module                 │
   │     ↓ call uk_*() in-process, authorized per-module          │
   ├──────────────────────────────────────────────────────────────┤
   │  THE KERNEL  (unfer)                                          │
   │  prob_kernel::Session  ← unfer_ffi (uk_*)  ← unfer_protocol   │
   │  nested_fock_algebra · fock_sirk · qfm                       │
   └──────────────────────────────────────────────────────────────┘
```

- **Modules** are Austral `.aui/.aum` cells. They are linear-typechecked, lowered to a CPS
  binary IR, and JIT-compiled by `australVM`'s `safestos/cranelift` backend. They call the
  kernel **in-process** through native `uk_*` symbols registered in the JIT — the same
  mechanism as the runtime's built-in `au_print_int`.
- **Authorization is per-module, and two-tiered.** In the lineage of **Theseus OS**,
  australVM's safety is *intralingual* — most capabilities are enforced by Austral's **linear
  types at compile time**, so they cost nothing at runtime (a direct, better-than-C call).
  Capabilities *declared dynamic at compile time* route to a **runtime** decision (Cedar or
  the unfer kernel); a denied call fails with **`UK-4001 CallDenied`**. The manifest
  `[grants]` allow-list (`auth.rs` / `ManifestAuthEngine`, Cedar optional) is today's dynamic
  lane. See `IMPLEMENTATION_PLAN.md` §"australVM authorization: the two-tier capability model"
  for the design direction (compile-time default / dynamic opt-in).
- **velysterm** drives the *same* `prob_kernel::Session` directly (a Rust dependency), and
  **AI agents** drive it over the `unfer_agent` NDJSON loop. One code path, three front ends.

---

## 3. The `uk_*` C ABI (what a module sees)

All parameters are `i64`-compatible (pointer + length; scalars travel inside JSON). Return
`≥ 0` = success (a handle, or a byte count); `< 0` = `-code` (a `UK-####`). Buffers use the
**probe protocol**: pass a null/short buffer to learn the byte count, then re-call with a
big-enough buffer.

The 17 exported symbols (`nm -D target/release/libunfer_ffi.so | grep uk_`):

```
uk_version            uk_init               uk_model_create       uk_model_free
uk_set_prior          uk_set_hamiltonian    uk_evolve             uk_condition
uk_event_probability  uk_observe            uk_bayesian_update    uk_get_result
uk_last_error         uk_snapshot           uk_restore            uk_subscribe   uk_poll
```

A typical module lifecycle: `uk_model_create(spec)` → handle, then `uk_set_prior` /
`uk_evolve` / `uk_event_probability` / `uk_bayesian_update`, drain JSON with
`uk_get_result`, and `uk_model_free`. On any error, `uk_last_error` returns a parseable
`Diagnostic` (code + message + repair hints).

The JSON shapes (`ModelSpec`, `HamiltonianSpec`, `PriorSpec`, `EventPredicate`,
`BayesianUpdateResult`, …) are defined in `unfer_protocol` and documented in
`docs/PROTOCOL.md`.

---

## 4. Hands-on: run a module

Every module ships a `run_demo.sh` that builds the kernel FFI, builds the `australVM`
cranelift bridge, JIT-runs the module, and exercises the authorization gate. From a clean
sibling checkout (`$ROOT/{unfer,australVM}` present):

```bash
bash $ROOT/unfer/demo_module/run_demo.sh
```

It does three things:
1. `cargo build --release -p unfer_ffi` — produce the `uk_*` symbols.
2. Build the `safestos/cranelift` bridge + `modhost`, with the kernel linked in.
3. Run the demo Austral cell through the CPS-JIT (a **live** `uk_*` call in-process),
   then run the **negative test**: revoke `uk_evolve` from the manifest grants and confirm
   the call now fails with `UK-4001`.

The `demo_module/module.toml` manifest is the normative example:

```toml
[module]
name = "demo_module"
version = "0.1.0"
archetypes = ["actor"]          # prior_provider | data_source | actor
entry = "src/DemoModule"

[grants]
kernel = ["uk_version", "uk_model_create", "uk_set_prior", "uk_evolve",
          "uk_event_probability", "uk_get_result", "uk_last_error", "uk_model_free"]
```

Remove a symbol from `[grants].kernel` and the corresponding call is denied — authorization
is the whole point of the module boundary.

---

## 5. Tour of the six existing modules

Each module is **executable documentation** of one slice of the architecture.

| module | archetype | demonstrates |
|--------|-----------|--------------|
| `demo_module/` | actor | the spine: create → set prior → evolve → read `P(event)`; the `UK-4001` negative test |
| `qfm_module/` | actor | the QFM Mehler-prior model end-to-end |
| `qfm_tomo_module/` | actor | the **smallest full pipeline**: TSR online inference (`compile/encode/evolve/decode`) + buffer-protocol result drain |
| `demo_module/data_source/` | data_source | a standalone Rust module ingesting external observations via `uk_observe` |
| `bayes_update_module/` | actor | the Quantum Bayesian Update (`uk_bayesian_update`) on a `qfm_tomography` model |
| `iterated_bayes_module/` | actor | the **full §7+§8 pipeline in a loop**: 3 iterations of `uk_bayesian_update` + `uk_get_result` + `uk_evolve`, one `freeModel` per iteration |

**Recommended order:** start with `qfm_tomo_module/` (the smallest end-to-end exercise of
online inference + authorization + the buffer protocol), then `iterated_bayes_module/` (the
full QFM-TSR + Bayesian pipeline driven in a loop).

### The Bayesian update + the rev-19 posterior mean

`uk_bayesian_update` conditions a TSR-evolved prior on new observations (the QFM.tex §8
algorithm) and runs HMC on the unit sphere of ℂᵐ. As of **rev 19** it returns, alongside the
single representative draw, a **posterior-mean point estimate** — the **Karcher (Fréchet)
mean** of the post-burn-in chain on the projective sphere ℂP^{m-1} (each sample phase-aligned
to quotient out the Born-rule `e^{iφ}` gauge). The result JSON now carries
`posterior_mean` + `n_samples` in addition to `image`. This is `qfm::bayes::karcher_mean`,
wired through `Session::bayesian_update` → `BayesianUpdateResult` → `uk_bayesian_update`.

---

## 6. The AI-agent interface (`unfer_agent`)

For machine callers, `velysterm/crates/kernel_client` ships `unfer_agent`: an **NDJSON**
request/response loop on stdin/stdout, in the spirit of a structured-error "Zero language" —
stable codes, typed repair hints.

```bash
printf '{"id":"1","op":"version","params":{}}\n' \
  | cargo run -p kernel_client --bin unfer_agent
# → {"id":"1","ok":true,"result":{...}}
```

Ops: `version`, `create_model`, `set_prior`, `evolve`, `condition`, `probability`,
`snapshot`, `bayesian_update`, `list_codes` (dumps the whole `UK-####` table so an agent can
self-document). Every failure carries a `Diagnostic` with repair hints; an unknown op →
`UK-1001` + a hint listing valid ops.

---

## 7. External modules (P11, all five DONE as of rev 28)

unfer is designed to grow by **reusing sibling projects across an arms-length interface**,
not by absorbing their code. All five integrations are now built (see
`IMPLEMENTATION_PLAN.md` §P11 for the full design, the rev 28 status paragraph, and the
per-item license analysis).

> **The license boundary is the protocol boundary.** The unfer kernel is `MIT OR
> Apache-2.0`. Where a sibling is *not* permissively compatible, **no code is copied into
> unfer** — the module lives in its own repo under its own license and talks to the kernel
> only via `unfer_protocol` JSON (the `uk_*` ABI or the `unfer_agent` NDJSON loop). Crossing
> only that interface keeps every side's licensing intact.

### 7.1 `pattern_unfer` — persistent CRDT session memory (P11.19, DONE rev 28)

- **Status:** implemented as `../pattern/crates/pattern_unfer/` — a new crate inside
  `../pattern` (workspace-declared `AGPL-3.0`; per-file MPL-2.0 notices matching the rest of
  that repo). `src/kernel_client.rs`'s `UnferKernelClient` spawns the `unfer_agent` binary as
  a child process and speaks its NDJSON wire protocol directly (no Cargo dependency on any
  unfer crate — genuinely arms-length, not just license-arms-length). `src/session.rs`'s
  `PatternUnferSession` wraps a `prob_kernel::SessionBlob` in a
  `pattern_core::memory::document::StructuredDocument`: `checkpoint` calls the real
  `save_session` op and writes the returned blob into a `StructuredDocument` field (bumping a
  Loro-committed revision counter); `restore` calls `restore_session` to rehydrate a model
  from the document's last checkpoint. 10 unit tests, plus 1 `#[ignore]`d integration test
  that spawns the *real* `unfer_agent` binary and round-trips create → evolve → save →
  restore → evolve-again — run and passing in this session (`UNFER_AGENT_BIN=<path> cargo
  test -p pattern-unfer -- --ignored`).
- **From:** `../pattern` (**MPL-2.0** per-file) — a dedicated `pattern_memory` crate with a
  full **Loro-CRDT** memory subsystem (`MemoryCache`, `SharedBlockManager`,
  `StructuredDocument`, `loro_sync` subscribers, filesystem `mount`, `jj`/VCS adapter,
  `backup`/`sharing`); `pattern_unfer` depends only on `pattern-core` (for
  `StructuredDocument`), not the rest of that subsystem yet (see IMPLEMENTATION_PLAN.md's
  "Next steps" for wiring into `MemoryCache`/`SharedBlockManager`).
- **License rule (honored):** the adapted module code stays inside `../pattern`; nothing was
  vendored into unfer, and unfer's own crates gained no new dependency at all.
- **What it gives unfer:** a probability-kernel session becomes a **persistent, versioned,
  CRDT document** — the Loro commit on each `checkpoint()` is exactly the undo/redo,
  filesystem-sync, sharing substrate `StructuredDocument` already provides for every other
  Pattern memory block.
- **Bridge (actual, not the FFI names originally sketched):** `save_session`/
  `restore_session` over the `unfer_agent` NDJSON loop — these are the real op names in
  `unfer_agent.rs`; `uk_snapshot`/`uk_restore` are the corresponding FFI-side symbols for
  modules calling in-process via the JIT, not what an external NDJSON client uses.

### 7.2 `arctic_authority` — threshold-signed collective authority (P11.20, DONE rev 28)

- **Status:** implemented as `australVM/arctic_authority/`, a new sibling crate (MIT). Depends on
  `../dynamic-arctic` for `arctic_core::{PubKey, Signature, verify}` and the
  `DelegationCertificate`/`DelegationRequest` wire types already defined there.
  `ArcticAuthEngine::register_certificate` verifies a certificate's threshold Schnorr
  signature once (against the group public key) and registers it; `check(principal, action,
  resource)` then authorizes purely by capability-list membership + expiry, so the
  authorization hot path never re-verifies a signature. 9 unit tests. Bridged into
  `australVM/safestos/cranelift` as a new optional `arctic-auth` feature: `src/arctic_auth.rs`
  defines `ArcticVmEngine: AuthorizationEngine` (delegates to `ArcticAuthEngine::check`) and
  `install()` (wires it into `auth::set_auth_engine`, the same mechanism
  `safestos_load_auth_manifest` uses for `ManifestAuthEngine`); 1 more test. Builds and tests
  clean alone, with `default` features, and combined with `default` — no interference with
  Cedar or `ManifestAuthEngine`.
- **From:** `../dynamic-arctic` (**MIT**) — the Arctic threshold-signature scheme
  (`arctic_core`, `shine_core`, Lagrange interpolation): a stateless, robust collective
  authority. (Its own `src/main.rs` — not vendored, only its wire types are reused —
  demonstrates the did:web / AT-Protocol delegation-certificate issuance flow this crate
  consumes.)
- **Integration point (a), authorization — done:** a sensitive `uk_*` call can now be
  authorized against a valid **n-of-t** threshold-signed `DelegationCertificate` instead of
  (or alongside) the single-policy Cedar / `ManifestAuthEngine` + `UK-4001` path.
- **Integration point (b), signed observations — not yet:** a `data_source` module whose
  `uk_observe` payloads themselves carry an Arctic threshold signature is follow-on work (see
  IMPLEMENTATION_PLAN.md's "Next steps").
- **License (honored):** MIT ↔ MIT throughout — `arctic_authority` and the `arctic-auth`
  cranelift feature; no copyleft code crossed anywhere in this integration.

### 7.3 `hayagriva` — bibliography & citation in the math editor (P11.21, DONE rev 28)

- **Status:** implemented as `velysterm/crates/mathed_biblio/` — a direct Cargo dependency on
  `../hayagriva` (both MIT/Apache). `load_yaml`/`load_bibtex` parse a `hayagriva::Library`;
  `CitationStyle::by_name` resolves one of hayagriva's ~2600 bundled CSL styles (rejecting
  dependent styles, which only override locale/terms); `Bibliography::cite` renders a
  grouped in-text citation via `hayagriva::standalone_citation`, `Bibliography::reference_list`
  renders the full bibliography via `BibliographyDriver`. 11 unit tests. `mathed_core` gained,
  additively: `PropKind::{Bibliography, Cite}` (+ `is_biblio()`, mirroring `is_kernel()`), a
  new `SemanticIndex.biblio_statements: Vec<BiblioStatement>` collected in `build_index`
  alongside (not mixed into) `kernel_statements`, and two new `AccessRole` variants
  (`Bibliography → Group`, `Citation → Link`) wired through `mathed_mini`'s AccessKit bridge.
  `mathed_biblio::resolve_citations` bridges `biblio_statements` → rendered strings, keyed by
  each `\cite`'s document span. 3 new `mathed_core` tests (75 total, was 72); the full
  velysterm workspace (`cargo test --workspace`) stays green.
- **From:** `../hayagriva` (**MIT OR Apache-2.0**) — the Typst project's bibliography
  manager (`Library`, the `io` reader/writer for Hayagriva-YAML + BibTeX, CSL `lang`/style
  formatting, in-text citations + reference lists).
- **License:** permissive and same-family as velysterm — so unlike P11.19/.20 it is a
  **direct Cargo dependency**, not an arms-length bridge, exactly as planned.
- **What it gives velysterm:** attach a bibliography (YAML/BibTeX) via `\bibliography(#1,#2,
  name, format: "yaml", style: "apa")`, insert in-text citations via `\cite(#1,#2, "key-a",
  "key-b", bib: "name")` — the `#mark1…#mark2` marker convention (the P3.10 translator pivot
  — users add meta-info through markers + a translator, **not** hand-written Typst-math) —
  and render a formatted reference list. The natural way to cite the physics literature (e.g.
  the `Layden2025` wavefunction-flow paper QFM.tex builds on) inside the editor.
- **Not yet done:** wiring `resolve_citations`'s output into `mathed`'s Bevy overlay / the
  `mathed_mini` translator panel (rendering next to the `\cite` span, mirroring how
  `\prob` results already render as `= 0.4231`/`UK-####` annotations) — see
  IMPLEMENTATION_PLAN.md's "Next steps."

### 7.4 `unfer_edge` — Pingora proxy module *inspired by* zentinel (P11.22, DONE rev 27)

- **Status:** implemented as `unfer_edge/` (8th workspace crate — a binary, not
  an Austral module, since it fronts the network edge rather than calling
  `uk_*` from inside the JIT). `src/filter.rs` implements the
  `ai-gateway`-style op allowlist (`validate_request`, UK-1001 on bad JSON,
  UK-4001 on a denied `op`); `src/mask.rs` implements the
  `data-masking`/`secret-inject`-style redaction of `AgentRequest`/
  `AgentResponse` envelopes (recursive JSON walk, redacts any key containing
  `api_key`/`secret`/`token`/`password`/`authorization`/`credential`);
  `src/main.rs` wires both into a `pingora_proxy::ProxyHttp` impl —
  `request_filter` rejects disallowed ops before the backend is reached,
  `upstream_response_body_filter` + `response_body_filter` buffer and mask the
  full upstream response body (dropping `content-length` since masking can
  change the byte length). 11 unit tests, clippy-clean.
- **Not** an integration of `../zentinel`. zentinel is itself a full security-first reverse
  proxy, and as a whole project it is **redundant with unfer** — so it is **not** depended
  on or deployed. Instead, build a **new, focused proxy-edge module directly on Cloudflare
  **Pingora**** (Apache-2.0, a direct dependency), reusing only the *design ideas* from
  zentinel and the user's fork-local agent modules (`zentinel/agents/`: **`ai-gateway`**
  route/authn, **`data-masking`** + **`secret-inject`** payload protection,
  **`wasm-allowlist`** sandbox/capability gating).
- **What it gives unfer:** the **edge** in front of the kernel's machine interface. Front the
  `unfer_agent` NDJSON loop (or a future kernel HTTP gateway) with a hardened Pingora proxy,
  re-implementing as Pingora request/response filters just the behaviours the kernel needs:
  an `ai-gateway`-style router/authn ahead of the agent ops, `data-masking`/`secret-inject`-
  style protection of the `AgentRequest`/`AgentResponse` JSON envelopes, and an
  `allowlist`-style gate on callers.
- **Relationship to arctic:** `unfer_edge` is the **network** complement to arctic's
  **cryptographic** authority — arctic decides *who is allowed* (threshold signature),
  `unfer_edge` does *edge enforcement + transport hardening*. Lives as a sibling
  `unfer_edge/` crate (or a `kernel_client` bin) depending only on `pingora` +
  `unfer_protocol` — no zentinel code vendored, only the ideas.

---

### 7.5 `unfer_nixvm` — GPU-shared Nix execution sandbox (P11.23, DONE rev 28)

- **Status:** two pieces, composed rather than merged. (1) `../unfer/flake.nix` gained
  `packages.x86_64-linux.unfer-ffi` — a real, **actually-built** `rustPlatform.buildRustPackage`
  derivation of the CPU-only `unfer_ffi` cdylib+rlib, using a new `nixpkgs-unstable` flake
  input (the workspace's `edition = "2024"` needs a newer rustc than the CUDA devShell's
  deliberately-pinned `nixos-23.05` ships; that devShell is untouched). (2) new subdirectory
  `unfer/unfer_nixvm/flake.nix` (it lives inside the unfer repo, not as a separate sibling
  checkout, since its `packages.*` output is entirely about the unfer kernel) takes
  `../cloud-hypervisor-build`'s `configuration.nix` as a
  raw (`flake = false`) path input — not edited, upstream stays the single source of truth —
  and layers a NixOS module installing `unfer-ffi` on top, producing `vm-perf-with-unfer` /
  `vm-sec-with-unfer` image outputs. `nix eval` confirmed the whole composition is
  well-formed, reaching real NixOS `system.build.toplevel` evaluation, before this sandboxed
  session's disk (~17G free) ran out attempting to realize the full image via `nix build` —
  an environment resource limit hit while verifying, not a flake defect.
- **A real bug found and fixed:** `../cloud-hypervisor-build/full-stack-vm-launch.sh` invoked
  virtiofsd with `--shared-dir ../nix`, resolved against the *caller's* cwd rather than
  `$SCRIPT_DIR` — silently pointing at a nonexistent directory unless invoked from one
  specific working directory, which would have quietly broken the "host store IS guest
  store" mechanism below. Fixed in place to the absolute `/nix`.
- **The recipe, committed:** `../cloud-hypervisor-build` itself (~930MB — full vendored
  `crosvm`/`cloud-hypervisor` upstream checkouts plus a built `.deb`) stays external and
  ungitted, same as `../dynamic-arctic`. But the small, hand-authored recipe behind it (the
  Nix/shell files, ~150KB — the two SpectrumOS GPU-sharing patches are downloaded
  fresh by `setup.sh` at run time, not vendored, per the third-party-license caution
  applied here) is committed at
  `australVM/cloud_hypervisor_vm/` (grouped with `arctic_authority` in `australVM` as
  runtime/module infrastructure, not kernel code) — its `setup.sh` clones
  cloud-hypervisor/crosvm/vhost at pinned commits, applies the patches, and builds both
  binaries, regenerating an equivalent working tree from source.
- **A decision left to the user, not made unilaterally:** `configuration.nix` mounts the
  shared `/nix` **read-only** in the guest — the safe default (a compromised/buggy guest
  process can't corrupt the host's store), but it also means the "packages installed in the
  VM transfer to the host" direction doesn't work yet as configured (a read-only mount can't
  receive the guest's `nix-daemon` writes). Flipping `ro`→`rw` is a real security-boundary
  decision; see `unfer_nixvm/README.md` for the two safer alternatives instead of an
  unreviewed flip.
- **From:** `../cloud-hypervisor-build` (**Apache-2.0 / BSD-3-Clause**) — a Nix +
  **cloud-hypervisor 50.0** stack patched with the **spectrum0** GPU-sharing patches
  (SpectrumOS). Its own `flake.nix` builds two NixOS images (`vm-perf` = Nix-store sharing +
  git from `/nix`; `vm-sec` = same + SSH-agent socket forwarding), and
  `full-stack-vm-launch.sh` wires a crosvm **vhost-user-gpu** backend plus **virtiofsd**
  sharing of the host's `/nix`. `unfer_nixvm` composes with these rather than modifying them
  (except the one-line path-bug fix above).
- **The mechanism:** because Nix store paths are content-addressed and immutable, **the host
  store *is* the guest store** once `configuration.nix`'s virtiofs `/nix` mount is backed by
  the host's real store — a package built on the host (`nix build` → `/nix/store/…`) is
  instantly usable in the guest, no copy, no rebuild.
- **What it gives unfer (done vs. not yet):** (a) **done** — a reproducible Nix package for
  `unfer_ffi` that the VM's shared store already contains once built on the host; (b) **not
  yet** — `australVM`/`qfm` as sibling packages, a `cuda`-feature package variant (CUDA is
  unfree and heavier to build through `nixpkgs-unstable`'s `rustPlatform` than the CPU
  default), and actually booting `vm-perf`/`vm-sec` to confirm `unfer_ffi` runs against the
  shared GPU — all deliberately left for the user to do by hand (`sudo`, real GPU/network
  device access; see `unfer_nixvm/README.md`'s "Launching" section). (c) still valid as
  designed: a sandboxed compute backend that `unfer_agent` / `unfer_edge` (§7.4) can front,
  gated by `arctic_authority` (§7.2).
- **`../claurst`** (a Rust terminal coding agent, GPL-3.0, related to `../pattern`'s
  persistent-agent work) inspires the **agent-drives-the-VM** UX — but being **copyleft** it
  is **inspiration only / arms-length**, never vendored into the permissive kernel, exactly
  like `../pattern`/`../pattern-main` in §7.1. No claurst code exists anywhere in
  `unfer_nixvm/`.

## 8. Add your own module

The full recipe (folder layout, manifest schema, the three archetype contracts, the
build → load → grant → run → hot-swap lifecycle, and a numbered checklist) is in
`docs/MODULE_RECIPE.md` and `docs/MODULES.md`. In brief:

1. Create a sibling/`unfer/<name>/` folder with a `module.toml` (declare `archetypes` and
   the `[grants].kernel` allow-list).
2. Write the Austral cell `src/<Name>.aui/.aum`, importing `UnferKernel` and the `uk_*`
   foreign functions you need.
3. Implement one of the archetype entry points:
   - **prior_provider** — `provide_prior(model: Int64): Int64`
   - **data_source** — `update(model: Int64, payload: Address, len: Int64): Int64`
   - **actor** — `act(model: Int64): Int64`
4. Write a `run_demo.sh` (assert the sibling layout, build the FFI + bridge, JIT-run, and
   include the `UK-4001` grant-removal negative test).
5. Wire a CI job mirroring an existing `*-e2e` job.

To add a new **kernel op** instead (a new `uk_*`): protocol type → `Session` method →
`uk_` shim → JIT symbol → agent op → `UK-####` allocation. See `docs/ARCHITECTURE.md`
extension-point #2 for the exact file list.

---

## 9. Where to go next

- `docs/ARCHITECTURE.md` — system diagram + per-crate extension-point checklists.
- `docs/PROTOCOL.md` — every JSON request/response schema and the full `UK-####` table.
- `docs/MODULE_RECIPE.md` / `docs/MODULES.md` — the module recipe.
- `docs/IMPLEMENTATION_PLAN.md` — the per-revision implementation record (currently rev 28)
  and the future roadmap (P8–P11), including the P11 external-module designs above.
- `QFM.tex` — the algorithm specification (the QFM-TSR pipeline §7, the Bayesian update §8,
  the rev-19 Karcher-mean subsection).
