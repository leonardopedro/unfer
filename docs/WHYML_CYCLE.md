# The WhyML Cycle: the probability kernel extends the AustralVM compiler

**Goal.** Use the probability kernel to *produce WhyML code* (or call the
kernel *from* WhyML) that compiles — via Why3 — to an OCaml module extending
the AustralVM compiler, with machine-checked properties on the AustralVM
code. The cycle: the kernel supplies the compiler extension, and the extended
compiler compiles the modules that call the kernel.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. prob_kernel (Rust)                          ┌──────────────────────┐  │
│    uk_whyml_emit(session, WhymlSpec)          │ the kernel's own      │  │
│      → a .mlw program with provable           │ S21 grant-subset      │  │
│        postconditions (the authorization      │ semantics — the       │  │
│        gate: authorize grants required =      │ compiler check is the │  │
│        true ⇔ required ⊆ grants)              │ kernel's capability   │  │
└─────────────────────────┬──────────────────────┴──── lattice, compiled │
                          │ .mlw                                      │  │
                          ▼                                           │  │
┌───────────────────────────────────────────────────────────────────────┐  │
│ 2. why3 (external, LGPL — SUBPROCESS ONLY, never linked;              │  │
│    the same license seam as Cadabra2/GPL in S30)                      │  │
│    why3 prove  -P alt-ergo authorize_gate.mlw   → 5 goals discharged  │  │
│    why3 extract -D unfer_ocaml.drv authorize_gate.mlw                 │  │
│      → authorize_gate.ml (custom driver: Why3 int → native OCaml int; │  │
│        the stock ocaml64 driver emits Zarith Z.t, pulling a zarith    │  │
│        dependency into the plugin — unfer_ocaml.drv avoids it)        │  │
└─────────────────────────┬─────────────────────────────────────────────┘  │
                          │ authorize_gate.ml (pure OCaml; the user's own  │
                          │ program, Apache-able — not Why3's code)        │
                          ▼                                                │
┌──────────────────────────────────────────────────────────────────────┐   │
│ 3. australVM compiler (OCaml, Apache-2.0)                            │   │
│    lib/Compiler_plugin.ml — the plugin seam: every registered pass   │   │
│      runs on each typed module after typing, before codegen; a       │   │
│      VerdictReject aborts compilation.                               │   │
│    lib/why3_plugin/why3_plugin.ml — registers the pass: for every    │   │
│      TForeignFunction whose external name is uk_*/uz_*, the module's │   │
│      grant set must cover it — decided by Authorize_gate.gate_verdict│   │
│      (the extracted function; sound by the Why3-proved postcondition)│   │
└─────────────────────────┬────────────────────────────────────────────┘   │
                          │ compiles Austral cells                        │
                          ▼                                                │
┌──────────────────────────────────────────────────────────────────────┐   │
│ 4. The compiled cell calls uk_whyml_emit again (JIT-linked kernel)   │   │
│    → emits WhyML → verify+extract → the same plugin.                 │   │
│    SELF-EXTENSION: the compiler is extended by code the kernel       │   │
│    produced, verified by Why3, loaded through its own plugin seam.   │   │
└──────────────────────────────────────────────────────────────────────┘   │
                                                                           │
The reverse direction ("calling the kernel from WhyML") is the `val`        │
declarations: the emitted program may declare uk_* kernel calls as          │
externals; the extracted OCaml binds them at link time to the unfer C ABI  │
via the `CamlCompiler_stubs` shim (see the template in whyml.rs).          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Why the authorization gate?

The kernel's `GrantSet::is_subset_of` (S21) is the capability-lattice
semantics that already gates every `uk_*` call at the JIT loopback
(UK-4001). Writing it in WhyML makes the *compiler itself* enforce it as a
machine-checked property:

- **Soundness** — `authorize grants required = True` only when every
  required symbol is granted. A compiled module can never import a `uk_*`
  symbol its grant set lacks.
- **Completeness** — `authorize` returns `False` exactly when a required
  symbol is missing; granted imports always pass.
- **No escalation path** — the subset lattice lemmas (reflexivity,
  transitivity) are proved as theorems in the same `.mlw`.

Both are discharged by Why3's provers (`why3 prove`), and Why3 extraction is
semantics-preserving, so the OCaml module the compiler loads satisfies the
postcondition. The property is *machine-checked*, not test-asserted.

## The pieces

### 1. Kernel-side emission — `prob_kernel::whyml` (`uk_whyml_emit`)

- Wire types `WhymlSpec` / `WhymlReport` / `WhymlOp` live in
  `unfer_protocol` (JSON schema over the C ABI).
- The emitter validates the spec against the kernel's **own symbol registry**
  (`unfer_protocol::symbols::SYMBOL_REGISTRY`): it refuses to emit a gate for
  a symbol the kernel does not know (`UK-4904 WHYML_SPEC_INVALID`). The
  kernel's knowledge bounds the codegen.
- `WhymlOp::Emit` is pure Rust — always works, no external tool. `WhymlOp::Prove`
  additionally runs `why3 prove` as a subprocess (`WHY3_CLI` override, then
  PATH) and reduces the outcome to `verified` (`UK-4903 WHYML_ENGINE_UNAVAILABLE`
  when the binary is missing — the Cadabra2 pattern).
- The emitted `.mlw` embeds the grant context for audit, the exact
  verify/extract commands, and (optionally) commented `val` declarations for
  kernel-call externals.

### 2. Why3 (external toolchain)

```
why3 prove  -P alt-ergo authorize_gate.mlw
why3 extract -D unfer_ocaml.drv authorize_gate.mlw -o authorize_gate.ml
```

`authorize_gate.mlw` is checked in at
`australVM/lib/why3_plugin/authorize_gate.mlw` (byte-identical to what the
kernel emits for the sample grant set); the expected extraction
`authorize_gate.ml` is pinned and its interface `authorize_gate.mli` is the
plugin's contract. The extraction driver `unfer_ocaml.drv` sits next to the
`.mlw`; the stock `ocaml64` driver would map Why3 `int` to Zarith `Z.t` and
thereby drag the zarith OCaml library into the compiler plugin, so the
custom driver maps `int` to native OCaml `int` (the `int32`/`int63` pattern
inside `ocaml64.drv` itself).

**Why3 1.8 note**: `list.Mem`'s `mem` is a *logical predicate* and Why3 1.8
refuses it in program code — the emitted program therefore defines a
program-level recursive `mem` function whose postcondition
(`result = True <-> Mem.mem x l`) links it to the predicate. That adds a
fifth proof obligation (`mem`'s own postcondition) beyond the two subset
lemmas and the two gate postconditions; all five are discharged by
alt-ergo (verified against Why3 1.8.2 + alt-ergo 2.6.3 from the dev shell).

### 3. Compiler-side — the plugin seam

`australVM/lib/Compiler_plugin.ml(mli)`:

```ocaml
type verdict = VerdictOk | VerdictReject of string
val register : name:string -> (module_name:string -> foreign_externals:string list -> verdict) -> unit
val run_on_typed : Stages.Tast.typed_module -> verdict
```

`Compiler.compile_mod` (and the hot-swap path) run every registered pass on
the typed module right after typing, before codegen; a `VerdictReject`
aborts compilation with an Austral error. `why3_plugin.ml` installs the
authorization-gate pass (idempotent, from `empty_compiler`).

The gate pass reads the module's grant set from `AUSTRAL_UK_GRANTS`
(comma-separated `uk_*` symbols). This stands in for the full
`module.toml` manifest plumbing, which today lives in the JIT/modhost layer
— with the variable unset the pass is a no-op, so existing builds are
unaffected. **Extension point:** thread the real manifest into the compiler
(per-module grant map) so the gate is always armed; the WhyML check is then
the single enforcement point shared by the compiler and the JIT.

The gate core is **pure** — `check_with_grants` takes the grant list
explicitly — with `check` as the env-reading wrapper, so tests exercise
every branch without mutating the process environment (ounit2 flags env
changes between tests; OCaml 4.14 has no `Unix.unsetenv`).

### 4. The compiler itself is a plugin of the VM (unified application/VM)

`australVM/lib/Vm_plugin.ml(mli)` makes the **whole compiler a plugin of
the application/VM**, following the Mirage-unikernel pattern (see
`../mirage-skeleton/` and the `australVM/unikernel/` scaffold):

```ocaml
type compiler_service = {
  name : string;
  compile : Compiler.module_source list -> Compiler.compiler;
}
val register_compiler : compiler_service -> unit
val run_compiler : Compiler.module_source list -> Compiler.compiler
val list_compilers : unit -> string list
val boot : unit -> unit
```

- **`Vm_plugin.boot ()`** registers the built-in compiler as the
  application's `austral-builtin` compiler plugin (idempotent), installing
  the Why3 gate as a pass plugin of that compiler. `Cli.main'` calls it at
  startup — the application boots by loading its plugins.
- **`CliEngine` routes every compile through `Vm_plugin.run_compiler`**
  instead of a hard-coded `compile_multiple empty_compiler` call: the
  compiler is discovered through the registry, so it is swappable/extensible
  exactly like a unikernel job.
- The gate is therefore a plugin of a plugin: application/VM → compiler
  plugin (`austral-builtin`) → Why3 pass plugin (`why3_gate`).

### 5. The cycle closes

An Austral cell can call `uk_whyml_emit` (granted like any `uk_*` symbol);
the emitted `.mlw` verifies and extracts to the OCaml plugin that extends
the very compiler compiling that cell. The compiler extension is
kernel-produced, Why3-verified, and loaded through the compiler's own plugin
seam — and the compiler itself is loaded through the application/VM's
plugin seam. Self-hosting: the compiler runs inside the same application
that hosts the JIT'd modules it produces.

### 6. Unikernel packaging (Mirage)

`australVM/unikernel/` mirrors the mirage-skeleton layout:

- `config.ml` — `main "Unikernel" job ~local_libs:[ "austral_lib" ]`,
  `register "australvm-compiler" [ main ]`.
- `unikernel.ml` — the job's `start`: `Vm_plugin.boot (); Vm_plugin.run_compiler
  [probe]`, i.e. the exact CLI boot, inside a single self-contained
  unikernel binary.
- `compiler_vm_test.ml` + `dune` — the Mirage-free executable that runs the
  same `boot_and_compile` under the unix backend, so the workspace build
  exercises the unified boot without the Mirage toolchain
  (`opam install mirage` is optional).

Build the actual unikernel with `mirage configure -t unix && make depend &&
make`; the result is one binary that is both the VM and the application —
compiler, Why3 gate, and module host in a single artifact.

## License (Why3 is LGPL — attention required)

Why3 is distributed under the GNU **Lesser** GPL (LGPL-2.1). The design
keeps the Apache-2.0 australVM clean:

- `why3` is invoked only as a **subprocess** — the Rust and OCaml binaries
  never link against Why3's code. This is the identical "independent work"
  seam this repo already uses for Cadabra2 (GPL-3.0, S30) and Lean4.
- The **extracted OCaml** is a mechanical translation of *your* WhyML
  program — it is not Why3's code, so it can be Apache-2.0 (the same
  reasoning as compiler output: the tool does not infect the work it
  processes).
- Keep the Why3 **library** out of any Apache-licensed binary; if a
  checked-in artifact (the pinned extraction) ever needs regenerating, run
  `why3 extract` in a separate toolchain and commit only the output.

## New-symbol registration checklist (`uk_whyml_emit`)

1. `unfer_protocol/src/symbols.rs` — `SymbolRecord` row (`Observe`).
2. `python3 scripts/gen_symbol_artifacts` — regenerates
   `EXPECTED_SYMBOLS.txt` and `include/unfer_kernel.h`.
3. `australVM/safestos/cranelift/src/lib.rs` — `UNFER_SYMBOLS` entry
   (cross-checked by the `symbol_sync` test).
4. `GrantSet.kernel` namespace — automatic (symbols.rs is the source).
5. Codes `UK-4903` / `UK-4904` in `unfer_protocol::codes`.

## Running the verification (when Why3 is available)

```bash
# From the unfer repo — emit the gate and prove it:
export WHY3_CLI=/path/to/why3
cargo test -p prob_kernel whyml            # emission tests (pure)
# From australVM — regenerate the pinned extraction and diff:
cd ../australVM && nix develop
why3 prove -P alt-ergo lib/why3_plugin/authorize_gate.mlw
why3 extract -D lib/why3_plugin/unfer_ocaml.drv lib/why3_plugin/authorize_gate.mlw \
  -o /tmp/extracted/authorize_gate.ml
diff /tmp/extracted/authorize_gate.ml lib/authorize_gate.ml
```

## Extensions

- **New emitted templates**: the emitter is a pure function of `WhymlSpec`;
  add templates for other provable passes (e.g. a linearity/borrow verdict on
  a small IR, a normalization check, a budget invariant) behind `WhymlOp`.
- **Kernel calls from WhyML**: `kernel_externals` in the spec emits `val`
  declarations; the extracted OCaml binds them to the unfer C ABI at link
  time (the `CamlCompiler_stubs` shim), so WhyML-verified passes can *call*
  the kernel with the same guarantees.
- **Manifest threading**: replace the `AUSTRAL_UK_GRANTS` env stand-in with
  the per-module `module.toml` grant set so the gate is always armed.
