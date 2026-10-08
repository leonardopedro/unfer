{
  description = ''
    unfer_nixvm — GPU-shared Nix execution sandbox for the unfer kernel (P11.23).

    Adapts the cloud-hypervisor-build stack (Apache-2.0/BSD-3-Clause:
    cloud-hypervisor 50.0 + the spectrum0 GPU-sharing patches +
    nixos-generators vm-perf/vm-sec images), whose hand-authored recipe now
    lives at ../../australVM/cloud_hypervisor_vm/ (2026-10-08 — that tracked
    directory is what the path input below points at, so evaluation depends on
    committed files only; the regenerated `cloud-hypervisor-build/` tree is
    launch-only, carrying the two patched binaries). Does not vendor code from
    it — this flake *composes* with its
    existing configuration.nix by adding a NixOS module on top, so upstream's
    module stays the single source of truth for the VM's virtiofs/GPU/network
    setup. UX is inspired (never vendored, GPL-3.0) by ../claurst's
    agent-drives-the-VM interaction style.

    Mechanism: because /nix/store paths are content-addressed and immutable,
    once `unfer.packages.x86_64-linux.unfer-ffi` (built by the parent
    ../flake.nix, P11.23's Nix-package half) is realized on the host, that
    exact store path is what the guest's virtiofs-shared /nix already
    contains — no copy, no rebuild inside the VM. See README.md for the full
    picture and the known bug this flake documents (but does not silently
    work around) in the recipe's full-stack-vm-launch.sh.

    Lives at unfer/unfer_nixvm/ (a subdirectory of the unfer repo itself,
    not a separate sibling checkout) since its packages output is entirely
    about the unfer kernel; the recipe (../../australVM/cloud_hypervisor_vm)
    stays an external sibling (composed via a path input, not vendored — see
    the cloud-hypervisor-build input below).
  '';

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixos-generators.url = "github:nix-community/nixos-generators";
    # The parent unfer flake (one level up — this is unfer/unfer_nixvm/),
    # for its `packages.x86_64-linux.unfer-ffi` output.
    unfer.url = "path:..";
    # `flake = false`: we only want the recipe's raw configuration.nix as a
    # NixOS module to import, not to evaluate its own flake.nix (which builds
    # its own, separate vm-perf/vm-sec outputs).
    # Absolute path deliberately: the recipe is an external sibling checkout
    # (see AGENTS.md's sibling-checkout convention), not a subdirectory of
    # this repo, so a relative "path:../../x" would silently break once this
    # flake is evaluated from a git store copy rather than the working tree.
    # 2026-10-08 (owner note): this points at the *tracked recipe*
    # (`australVM/cloud_hypervisor_vm/`), NOT at the regenerated
    # `cloud-hypervisor-build/` working tree — the two configuration.nix files
    # are byte-identical (setup.sh copies this one over), so composition
    # evaluates from committed files only. The build tree is launch-only: it
    # carries the two spectrum0-patched binaries (cloud-hypervisor --gpu, the
    # crosvm vhost-user-gpu backend) that `full-stack-vm-launch.sh` runs, and
    # `setup.sh` regenerates it.
    cloud-hypervisor-build = {
      url = "path:/home/leo/Projects/australVM/cloud_hypervisor_vm";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, nixos-generators, unfer, cloud-hypervisor-build }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # the recipe's own guest module (australVM/cloud_hypervisor_vm/
      # configuration.nix — byte-identical to any regenerated tree's copy):
      # virtiofs /nix +
      # optional ssh-agent share, GPU passthrough (hardware.opengl.enable),
      # the `agent` user, sshd. We depend on it as a flake input rather than
      # copying configuration.nix, so upstream changes there are picked up
      # automatically.
      baseModule = "${cloud-hypervisor-build}/configuration.nix";

      # Layered on top: install the unfer kernel's C ABI (unfer_ffi, built
      # reproducibly by the parent flake's `packages.x86_64-linux.unfer-ffi`)
      # into the guest, so the shared-store mechanism above is exercised by a
      # real, useful package rather than a toy example.
      unferGuestModule = { config, pkgs, ... }: {
        environment.systemPackages = [
          unfer.packages.${system}.unfer-ffi
        ];
      };
    in {
      packages.${system} = {
        # Performance strategy (Nix store sharing + git from /nix, no SSH
        # socket) + the unfer kernel pre-installed.
        vm-perf-with-unfer = nixos-generators.nixosGenerate {
          inherit system;
          format = "raw";
          modules = [ baseModule unferGuestModule ];
        };

        # Secure/agent strategy (same as perf + SSH agent socket forwarding)
        # + the unfer kernel pre-installed.
        vm-sec-with-unfer = nixos-generators.nixosGenerate {
          inherit system;
          format = "raw";
          modules = [ baseModule unferGuestModule ];
        };

        # S7: the sandboxed ECMAScript sidecar stack, packaged reproducibly.
        # The default path stays sidecar + OS sandbox WITHOUT a full VM (§2.1);
        # the VM outputs above remain the opt-in extra isolation tier. The store
        # paths are content-addressed, so the same exact workerd / unfer-ffi /
        # unfer-data builds are usable inside the guest over virtiofs with no
        # copy. `run_sandboxed_sidecar.sh` is the smoke gate: it drives the
        # packaged workerd through the S3 OS sandbox against the kernel.
        sandboxed-sidecar = pkgs.stdenv.mkDerivation {
          pname = "unfer-sandboxed-sidecar";
          version = "0.1.0";
          src = ./.;
          buildPhase = "true";
          installPhase = ''
            runHook preInstall
            mkdir -p $out
            ln -s ${unfer.packages.${system}.unfer-workerd} $out/workerd
            ln -s ${unfer.packages.${system}.unfer-data} $out/unfer-data
            ln -s ${unfer.packages.${system}.unfer-ffi} $out/unfer-ffi
            install -m 0755 run_sandboxed_sidecar.sh $out/run_sandboxed_sidecar.sh
            runHook postInstall
          '';
        };
      };

      # S7: `nix run .#sandboxed-sidecar-smoke` = the packaging gate (builds
      # workerd + the blueprint store + the kernel ABI, then runs the sandboxed
      # sidecar smoke against the packaged runtime).
      apps.${system}.sandboxed-sidecar-smoke = {
        type = "app";
        program = "${self.packages.${system}.sandboxed-sidecar}/run_sandboxed_sidecar.sh";
      };
    };
}
