# unfer_nixvm

GPU-shared Nix execution sandbox for the unfer kernel — P11.23 in
`../docs/IMPLEMENTATION_PLAN.md`.

This is glue, not a fork: it composes two existing, independently-licensed
projects rather than vendoring either into the other.

- **`../flake.nix`** (`MIT OR Apache-2.0`) — new `packages.x86_64-linux.unfer-ffi`
  output: a reproducible Nix derivation of the kernel's handle-based C ABI
  (`unfer_ffi`, cdylib+rlib), built with the workspace's own `Cargo.lock` via
  `rustPlatform.buildRustPackage`, CPU-only (default features, no `cuda`) —
  matching the workspace's own CPU-default convention (S1).
- **`../../cloud-hypervisor-build`** (`Apache-2.0`/`BSD-3-Clause`) — the existing
  `vm-perf`/`vm-sec` NixOS image builder: cloud-hypervisor 50.0 patched with
  the spectrum0 GPU-sharing patches, a `configuration.nix` that mounts the
  host's `/nix` into the guest over virtiofs+DAX, and
  `full-stack-vm-launch.sh` which wires a crosvm `vhost-user-gpu` backend plus
  virtiofsd sharing. The small, hand-authored recipe behind this directory (the
  Nix/shell files, ~150KB — the two SpectrumOS GPU-sharing patches are a third
  party's licensed work, so they're downloaded fresh and checksum-verified by
  `setup.sh` at run time instead of being vendored — without the ~930MB of
  vendored upstream checkouts and build output) is committed at
  `../../australVM/cloud_hypervisor_vm/` — see that directory's `setup.sh` to
  regenerate an equivalent working tree from source.
- **`../../claurst`** (`GPL-3.0`) — **inspiration only, never vendored**: its
  agent-drives-the-VM interaction style (a terminal coding agent that issues
  commands into a sandboxed environment) is the intended long-run UX for
  `unfer_agent`/`unfer_edge` fronting a kernel running inside this VM — no
  claurst code is used here.

This flake (`unfer_nixvm/flake.nix`) does not modify `../../cloud-hypervisor-build`'s
`configuration.nix` — it takes it as a flake input (`path:../../cloud-hypervisor-build`)
and layers a second NixOS module on top that adds `unfer.packages.x86_64-linux.unfer-ffi`
to the guest's `environment.systemPackages`, producing `vm-perf-with-unfer` /
`vm-sec-with-unfer` image outputs.

## The mechanism (why this is the "practical way to make new modules out of existing code")

Nix store paths are content-addressed and immutable: a given input always
hashes to the same `/nix/store/<hash>-<name>` path, on any machine. Because
`configuration.nix` mounts the *host's* `/nix` into the guest at `/nix` over
virtiofs (tag `host_nix`), once `nix build .#unfer-ffi` (from `../`)
realizes that derivation on the host, the exact same store path already
exists from the guest's point of view — no copy, no rebuild, no version
skew. That's the general pattern P11.23 exists to demonstrate: **package
existing software as a Nix derivation and it becomes usable inside the
GPU-shared VM for free.** The same pattern applies to `australVM` and `qfm`
as follow-on packages (see "What's not done yet" below).

This also gives a clean, *reproducible* resolution to the CUDA toolkit
pinning pain documented in `AGENTS.md §5` / P9.12 (`../flake.nix`'s
CUDA devShell): the toolkit version is pinned by the flake's `nixpkgs` input,
not by `LD_LIBRARY_PATH` juggling on whatever host happens to be running.

## Building (host-side, safe — no VM, no sudo, no GPU/network device access)

```sh
# From the parent unfer/ directory: build just the kernel's C ABI.
cd .. && nix build .#unfer-ffi
# -> ./result/lib/libunfer_ffi.so, ./result/lib/libunfer_ffi.a

# From here: build a full VM disk image with unfer_ffi pre-installed.
# NOTE: this *does* build a complete NixOS closure (kernel, systemd, the
# agent user, sshd, opengl libs, ...) — a large, slow, network-heavy build.
nix build .#vm-perf-with-unfer
# or
nix build .#vm-sec-with-unfer
```

**Verification status (this session, 2026-07-01):**
- `..#unfer-ffi` was actually built (not just evaluated) end-to-end via
  `nixpkgs-unstable`'s `rustPlatform.buildRustPackage` — a real
  `libunfer_ffi.so`/`libunfer_ffi.a` landed in `../result/lib/`. This is
  the one piece of P11.23 fully exercised, not just written.
- `.#vm-perf-with-unfer` was evaluated (not built) with `nix eval`: the input
  resolution, the `configuration.nix` + `unferGuestModule` composition, and
  NixOS's own module system all evaluated correctly deep into
  `system.build.toplevel` (past `hardware.opengl.enable`'s deprecation
  warning, past `nixos-generators`' own deprecation warning) — i.e. the flake
  itself is well-formed and composes as intended. Realizing the full
  derivation (`nix build`) was not completed: it exhausted this session's
  sandboxed disk (`error: writing to file: No space left on device` with
  ~17G free at the time — a full NixOS image closure plus the
  `nixpkgs-unstable` toolchain already pulled in for `unfer-ffi` did not fit).
  This is an environment resource limit encountered while validating, not a
  defect found in the flake; it should build fine on a host with more free
  disk (a NixOS raw image is commonly several GB once systemd, the kernel,
  and Mesa/OpenGL libs are included).

## Launching (host-side, NOT run by this session — requires `sudo`, real GPU/network device access)

`../../cloud-hypervisor-build/full-stack-vm-launch.sh` still does the actual
launch (GPU backend, virtiofsd exporters, tap networking, `cloud-hypervisor`
invocation). Point its `--disk` argument at this flake's built image instead
of `./result/nixos.img`:

```sh
cd ../../cloud-hypervisor-build
sudo ./full-stack-vm-launch.sh --strategy perf   # or --strategy sec
```

**This session deliberately did not run this script**, or anything that
mounts real host directories, opens GPU device sockets, configures tap
networking, or invokes `cloud-hypervisor`/`sudo` — those are exactly the
kind of hard-to-reverse, host-affecting actions that call for the user's own
hands on the keyboard, not an agent running them unattended. Building the
Nix derivations above is safe (sandboxed, no host mutation beyond
`/nix/store`); actually booting the VM is a separate, deliberate step for a
human to take.

## A bug this flake documents (and fixes) rather than silently working around

`full-stack-vm-launch.sh` invoked virtiofsd with `--shared-dir ../nix` — a
path relative to whatever directory the script happened to be *invoked
from*, not `$SCRIPT_DIR`. Unless the caller's cwd was exactly one level
above `cloud-hypervisor-build/`, this pointed at a nonexistent directory
rather than the host's real Nix store, silently breaking the "host store IS
guest store" mechanism this whole design depends on. This session's P11.23
work fixed it in place to `--shared-dir /nix` (the host's real, absolute Nix
store) — see the commit/diff in `../../cloud-hypervisor-build/full-stack-vm-launch.sh`.

## A design decision this flake does *not* make unilaterally: read-only vs. read-write `/nix`

`../../cloud-hypervisor-build/configuration.nix` currently mounts the shared
`/nix` **read-only** (`options = [ "ro" ]`) in the guest. That's the safe
default: a compromised or buggy guest process cannot corrupt the host's
store. It also means the plan's aspirational "packages installed in the VM
transfer to the host" direction does not yet work as configured — the
guest's `nix-daemon` cannot write new store paths back through a read-only
virtiofs mount. Making it writable is a real security-boundary decision
(the guest OS would gain write access to the host's package store) that
this session left to the user to make deliberately, rather than flipping
`ro` → `rw` as an unreviewed side effect of an "adapt the sibling repo" task.
If bidirectional transfer is wanted, the two options are: (a) mount `/nix`
read-write and accept that trust boundary, or (b) run a *second*,
guest-local Nix store and sync build results explicitly (e.g. `nix copy`
over the existing SSH path) — safer, more code, not implemented here.

## What's not done yet

- `australVM` and `qfm` as Nix derivations alongside `unfer_ffi` (the plan's
  full P11.23 scope) — `unfer_ffi` alone was packaged and is the piece the
  `unfer_agent`/`unfer_edge` (P11.22) surface actually needs to run.
- A `cuda`-feature variant of `unfer.packages.x86_64-linux.unfer-ffi` (the
  GPU-accelerated `fock_sirk` solves) — building the unfree CUDA toolkit
  through `nixpkgs-unstable`'s `rustPlatform` is a heavier, license-gated
  undertaking than the CPU-default package; the existing CI GPU job
  (self-hosted runner with CUDA pre-installed, `.github/workflows/ci.yml`)
  remains the tested path for CUDA builds until this is picked up.
- Actually booting `vm-perf`/`vm-sec` and confirming `unfer_ffi` runs inside
  the guest against the shared GPU — deliberately left for the user to run
  by hand (see "Launching" above).
