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

`uk_bayesian_update` conditions a TSR-evolved prior on new observations (the QMF.tex §8
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

## 7. External modules (planned — P11)

unfer is designed to grow by **reusing sibling projects across an arms-length interface**,
not by absorbing their code. Five integrations are planned (see `IMPLEMENTATION_PLAN.md` §P11
for the full design and the per-item license analysis).

> **The license boundary is the protocol boundary.** The unfer kernel is `MIT OR
> Apache-2.0`. Where a sibling is *not* permissively compatible, **no code is copied into
> unfer** — the module lives in its own repo under its own license and talks to the kernel
> only via `unfer_protocol` JSON (the `uk_*` ABI or the `unfer_agent` NDJSON loop). Crossing
> only that interface keeps every side's licensing intact.

### 7.1 `pattern_unfer` — persistent CRDT session memory (P11.19)

- **From:** `../pattern` (**MPL-2.0**) — a dedicated `pattern_memory` crate with a full
  **Loro-CRDT** memory subsystem (`MemoryCache`, `SharedBlockManager`, `StructuredDocument`,
  `loro_sync` subscribers, filesystem `mount`, `jj`/VCS adapter, `backup`/`sharing`) —
  **combined with** `../pattern-main` (**AGPL-3.0**), the more recent branch that has the
  newer agent-platform features (`pattern_auth`/`pattern_mcp`/`pattern_api`, coordination
  patterns) but **lacks** that dedicated memory crate.
- **License rule:** because `../pattern` is MPL (file-level copyleft) and `../pattern-main`
  is AGPL, **the adapted module code stays inside `../pattern`** and reaches unfer only over
  the protocol boundary. It is *not* vendored into the permissive kernel.
- **What it gives unfer:** a probability-kernel session becomes a **persistent, versioned,
  CRDT document** — wrap `prob_kernel::SessionBlob` and the Bayesian-update history
  (including the rev-19 `posterior_mean`) in a Loro `StructuredDocument` with undo/redo,
  filesystem sync, and sharing.
- **Bridge:** `uk_snapshot` (SessionBlob out) / `uk_restore` (SessionBlob in), or
  `snapshot`/`create_model` over NDJSON. Natural partner of velysterm, which already speaks
  Loro for the math editor.

### 7.2 `arctic_authority` — threshold-signed collective authority (P11.20)

- **From:** `../dynamic-arctic` (**MIT**) — the Arctic threshold-signature scheme
  (`arctic_core`, `shine_core`, Lagrange interpolation): a stateless, robust collective
  authority for the AT Protocol (did:web identities + delegation certificates).
- **Two integration points:**
  1. **Authorization** — implement australVM's `AuthorizationEngine` trait
     (`safestos/cranelift/src/auth.rs`) with an Arctic threshold check, so a sensitive call
     (`uk_bayesian_update`, `uk_observe`, `uk_set_hamiltonian`) is allowed only against a
     valid **n-of-t** threshold-signed delegation certificate — generalizing the
     single-policy Cedar / `ManifestAuthEngine` + `UK-4001` path to a collective one.
  2. **Signed observations** — a `data_source` module whose `uk_observe` payloads carry an
     Arctic threshold signature + did:web identity, so the data conditioning the Born-rule
     update is cryptographically attested.
- **License:** MIT — can live as a sibling `arctic_authority/` crate / australVM auth backend.

### 7.3 `hayagriva` — bibliography & citation in the math editor (P11.21)

- **From:** `../hayagriva` (**MIT OR Apache-2.0**) — the Typst project's bibliography
  manager (`Library`, the `io` reader/writer for Hayagriva-YAML + BibTeX, CSL `lang`/style
  formatting, in-text citations + reference lists).
- **License:** permissive and same-family as velysterm — so unlike P11.19/.20 it is a
  **direct Cargo dependency**, not an arms-length bridge.
- **Where it lives — module vs. core:** add a focused velysterm crate (e.g.
  `crates/mathed_biblio`) wrapping `hayagriva::Library` + CSL formatting, consumed by
  `mathed_core`/`mathed`.
- **What it gives velysterm:** attach a bibliography (YAML/BibTeX), insert in-text citations
  via the `#mark1…#mark2` marker convention (the P3.10 translator pivot — users add
  meta-info through markers + a translator, **not** hand-written Typst-math), and render a
  formatted reference list. The natural way to cite the physics literature (e.g. the
  `Layden2025` wavefunction-flow paper QMF.tex builds on) inside the editor.

### 7.4 `unfer_edge` — Pingora proxy module *inspired by* zentinel (P11.22)

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

### 7.5 `unfer_nixvm` — GPU-shared Nix execution sandbox (P11.23)

- **From:** `../cloud-hypervisor-build` (**Apache-2.0 / BSD-3-Clause**) — a Nix +
  **cloud-hypervisor 50.0** stack patched with the **spectrum0** GPU-sharing patches
  (SpectrumOS). A `flake.nix` builds two NixOS images (`vm-perf` = Nix-store sharing + git
  from `/nix`; `vm-sec` = same + SSH-agent socket forwarding), and `full-stack-vm-launch.sh`
  wires a crosvm **vhost-user-gpu** backend plus **virtiofsd** sharing of the host's `/nix`.
- **The mechanism:** `configuration.nix` mounts the host Nix store into the guest at `/nix`
  over **virtiofs + DAX**. Because Nix store paths are content-addressed and immutable, **the
  host store *is* the guest store** — a package built on either side (`nix build` →
  `/nix/store/…`) is instantly usable on the other, no copy. That is the project's goal:
  *packages installed in the VM transfer to the host, and host packages are accessed from the
  VM directly.*
- **What it gives unfer:** (a) a **reproducible, GPU-accelerated execution sandbox** for the
  kernel — package `unfer_ffi`, the `australVM` runtime, `qfm`, and CUDA deps as Nix
  derivations and run the `fock_sirk` `cuda` solves inside `vm-perf`/`vm-sec` against the
  spectrum0-shared GPU; (b) a clean, reproducible fix for the **CUDA toolkit pinning** pain
  (P9.12 / AGENTS.md §5) — pinned by the flake, not `LD_LIBRARY_PATH` hacks; (c) a sandboxed
  compute backend that `unfer_agent` / `unfer_edge` (§7.4) can front, gated by
  `arctic_authority` (§7.2).
- **`../claurst`** (a Rust terminal coding agent, GPL-3.0, related to `../pattern`'s
  persistent-agent work) inspires the **agent-drives-the-VM** UX — but being **copyleft** it
  is **inspiration only / arms-length**, never vendored into the permissive kernel, exactly
  like `../pattern`/`../pattern-main` in §7.1.

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
- `docs/IMPLEMENTATION_PLAN.md` — the per-revision implementation record (currently rev 19)
  and the future roadmap (P8–P11), including the P11 external-module designs above.
- `QMF.tex` — the algorithm specification (the QFM-TSR pipeline §7, the Bayesian update §8,
  the rev-19 Karcher-mean subsection).
