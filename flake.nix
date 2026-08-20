# P9 P12: CUDA toolkit pinning for the unfer workspace (Nix shell).
#
# Pinned CUDA: 12.6 (from nixpkgs-unstable). Older toolkits are no longer
# usable: nixos-23.05's monolithic `cudaPackages_12_1.cudatoolkit` fails to
# build (auto-patchelf cannot satisfy its Qt6/gstreamer deps), and CUDA
# 12.1–12.5 have been removed from nixpkgs-unstable as unmaintained upstream.
# The unfer GPU tests were originally developed against 12.2
# (libcublas 12.2, libcudart 12.2, CUFFT 11.0, CUSOLVER 11.4), but candle-core's
# cudarc backend (cudarc 0.13.9) supports 12.6, which is what this shell
# provides. On systems with a system-installed toolkit elsewhere, the
# CUBLAS_STATUS_ARCH_MISMATCH error (AGENTS.md §5) can still occur when the
# driver and toolkit versions disagree.
#
# Use with:  nix-shell
# or:        nix develop
#
# This flake is opt-in: it's only loaded if the user has Nix
# installed and explicitly invokes nix-shell/nix develop in
# the unfer workspace. The CUDA-on-CI job
# (qfm-tomo-e2e-cuda in .github/workflows/ci.yml) does NOT
# require Nix; it runs on a self-hosted runner with the CUDA
# toolkit pre-installed.

{
  description = "unfer kernel + CUDA toolkit pinned environment (P9 P12) + reproducible-build packages (P11.23) + Haskell Egison workspace";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
  # Separate, newer channel for the CUDA toolkit AND for `packages.*` (P11.23,
  # unfer_nixvm): the workspace's Cargo.toml pins `edition = "2024"`, which needs
  # a rustc newer than nixos-23.05 ships, and the 23.05 CUDA 12.1.1 cudatoolkit
  # derivation is broken (auto-patchelf fails). The devShell keeps the 23.05 pin
  # only for the non-CUDA tooling (gcc/make/rustup/Haskell); the CUDA toolkit now
  # comes from the unstable channel, whose `cudaPackages` build cleanly.
  inputs.nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, nixpkgs-unstable }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;  # CUDA toolkit is unfree
      };
      pkgsUnstable = import nixpkgs-unstable {
        system = "x86_64-linux";
        config.allowUnfree = true;  # CUDA toolkit is unfree
      };
      # The merged CUDA 12.6 toolkit from nixpkgs-unstable (cudarc 0.13.9
      # supports 12.6). `cudatoolkit` on the unstable channel already points at
      # the merged output, which contains bin/nvcc, include/cuda.h and the
      # libcudart/libcublas/cusolver shared libs in `lib/`.
      cudaToolkit = pkgsUnstable.cudaPackages_12_6.cudatoolkit;

      # Configure Haskell package set with targeted Egison modules from the unstable channel
      haskellEnv = pkgsUnstable.haskellPackages.ghcWithPackages (ps: [
        ps.egison-pattern-src          # Core Egison pattern AST
        ps.egison-pattern-src-th-mode # Template Haskell bindings
        ps.template-haskell           # Built-in Template Haskell library
      ]);

      # S7: pinned workerd npm version (Cloudflare ships daily builds).
      workerdVersion = "1.20260808.1";
    in
    {
      # P11.23: `unfer_ffi` (the handle-based C ABI, cdylib+rlib) built as a
      # reproducible Nix derivation from the workspace's own Cargo.lock — CPU-only
      # (default features; no `cuda`), matching the workspace's CPU-default
      # convention (S1). This is the artifact `unfer_nixvm/` installs into the
      # GPU-shared VM's `/nix/store`, from which — because the store is content-
      # addressed and shared with the host over virtiofs — the exact same build is
      # usable on either side with no copy (see `../unfer_nixvm/README.md`).
      packages.x86_64-linux.unfer-ffi = pkgsUnstable.rustPlatform.buildRustPackage {
        pname = "unfer_ffi";
        version = "0.1.0";
        src = ./.;
        cargoLock.lockFile = ./Cargo.lock;
        buildAndTestSubdir = "unfer_ffi";
        # The full workspace test suite (fock_sirk, qfm, etc.) is exercised by CI;
        # this derivation only needs to produce the unfer_ffi artifacts.
        doCheck = false;
      };

      # P11.23/S7: the data plane crate exposing the content-addressed blueprint store
      # (`store_cell`/`verify_cell`, AES-GCM envelope, magnet URIs, content publishing).
      # Built reproducibly like unfer-ffi.
      packages.x86_64-linux.unfer-data = pkgsUnstable.rustPlatform.buildRustPackage {
        pname = "unfer_data";
        version = "0.1.0";
        src = ./.;
        cargoLock.lockFile = ./Cargo.lock;
        buildAndTestSubdir = "unfer_data";
        # unfer_data is a plain rlib (no cdylib, no bins), so the default
        # cargo-install leaves $out empty; ship the rlib explicitly so the
        # store path carries the content-plane crate itself.
        installPhase = ''
          runHook preInstall
          mkdir -p $out/lib
          rlib=$(find target -name 'libunfer_data*.rlib' -print -quit)
          if [ -n "$rlib" ]; then cp -v "$rlib" $out/lib/; fi
          runHook postInstall
        '';
        doCheck = false;
      };

      # S7: the workerd runtime, packaged from Cloudflare's npm tarballs (no Bazel/clang build,
      # no node shim). The meta tarball's `bin/workerd` is a node wrapper; we overwrite it with
      # the statically-linked platform binary from `@cloudflare/workerd-linux-64` and keep the
      # `workerd.capnp` schema beside it, exactly the npm layout `ecma.rs` expects
      # (`<pkg>/bin/workerd` + `<pkg>/workerd.capnp`). Reproducible: both URLs are pinned with
      # sha256, so the store path is content-addressed and shared with the guest over virtiofs
      # just like unfer-ffi (see `unfer_nixvm/`).
      packages.x86_64-linux.unfer-workerd = pkgsUnstable.stdenv.mkDerivation {
        pname = "unfer-workerd";
        version = workerdVersion;
        src = pkgsUnstable.fetchurl {
          url = "https://registry.npmjs.org/@cloudflare/workerd-linux-64/-/workerd-linux-64-${workerdVersion}.tgz";
          sha256 = "02235d1d5e56a655b587bedc803c75d64e6b5f97cbb13e6cf289d73b1082c17d";
        };
        srcCapnp = pkgsUnstable.fetchurl {
          url = "https://registry.npmjs.org/workerd/-/workerd-${workerdVersion}.tgz";
          sha256 = "78bf07ea96b4b5d4b1f515859fa725712d05f21ab16fc0c7cb1b7b0300b78135";
        };
        dontUnpack = true;
        installPhase = ''
          runHook preInstall
          mkdir -p $out
          # Capnp first (meta tarball: bin/workerd shim + workerd.capnp)...
          tar xzf $srcCapnp -C $out --strip-components=1
          # ...so the platform's real binary overwrites the shim at the same path.
          tar xzf $src -C $out --strip-components=1
          chmod +x $out/bin/workerd
          runHook postInstall
        '';
        meta = {
          description = "Cloudflare workerd runtime (npm tarball, S7 sidecar)";
          homepage = "https://github.com/cloudflare/workerd";
          license = pkgsUnstable.lib.licenses.asl20;
          mainProgram = "workerd";
        };
      };

      devShells.x86_64-linux.default = pkgs.mkShell {
        name = "unfer-cuda-12.6";

        # The CUDA toolkit, system utilities, and Haskell tools
        packages = with pkgs; [
          cudaToolkit
          gcc
          gnumake
          pkg-config
          rustup

          # S30: Cadabra2 symbolic CAS (external subprocess engine).
          pkgsUnstable.cadabra2
          
          # Haskell Integration
          haskellEnv
          pkgsUnstable.cabal-install
          pkgsUnstable.haskell-language-server
        ];

        # Prepend the CUDA toolkit libraries to LD_LIBRARY_PATH so
        # the linker picks them up first (the load-bearing Stage 2
        # Gram eigendecomp uses cuSOLVER; the Stage 6 reconstruction
        # uses cuBLAS for the per-row renormalization).
        shellHook = ''
          # gcc-15 libstdc++ (from the unstable channel) FIRST: cadabra2-2.5.14
          # is built against a GCC-15 toolchain and needs GLIBCXX_3.4.32 /
          # CXXABI_1.3.15, which the 23.05 stdenv's gcc-12 libstdc++ lacks
          # (runtime failure: "GLIBCXX_3.4.32 not found"). Prepend it so the
          # symbolic-CAS subprocess (prob_kernel::symbolic) loads a compatible
          # libstdc++ instead of the older one from stdenv.cc.cc. It is ABI-
          # backward-compatible, so all other tools keep working.
          export LD_LIBRARY_PATH="${cudaToolkit}/lib:${pkgsUnstable.gcc.cc.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          export CUDA_HOME="${cudaToolkit}"
          # cudarc's build.rs (candle-core's CUDA backend) locates the toolkit
          # via CUDA_ROOT/CUDA_PATH/CUDA_TOOLKIT_ROOT_DIR — NOT CUDA_HOME — plus
          # nvcc on PATH. Without these, `--all-features` fails at compile time
          # with "Unable to find `include/cuda.h` under any of: [...]".
          export CUDA_ROOT="${cudaToolkit}"
          export CUDA_PATH="${cudaToolkit}"
          export CUDA_TOOLKIT_ROOT_DIR="${cudaToolkit}"
          # bindgen_cuda (candle's CUDA-kernel codegen) queries `nvidia-smi` for
          # the compute capability unless CUDA_COMPUTE_CAP is set. Machines
          # without an NVIDIA GPU (e.g. AMD-only hosts) have no nvidia-smi, so
          # pin a capability to keep `--features cuda` builds working; 75
          # (Turing) is safe because nvcc 12.6 supports sm_75..sm_120.
          export CUDA_COMPUTE_CAP="75"
          export PATH="${cudaToolkit}/bin:$PATH"
          echo "[unfer-cuda-shell] CUDA ${cudaToolkit.version or "12.6"} on LD_LIBRARY_PATH"
          echo "[unfer-cuda-shell] CUDA_ROOT/CUDA_PATH set for cudarc/candle-core build-time detection"
          echo "[unfer-cuda-shell] Haskell environment with Egison loaded"
        '';

        # Rustup default toolchain.
        RUSTUP_TOOLCHAIN = "stable";
      };
    };
}