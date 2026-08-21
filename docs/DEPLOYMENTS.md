# Deployments

Deployment guide for the unfer stack (H14). The **core stays byte-identical**
across deployments — the S23/S24 release golden gate is the guard. Org material
lives in `deploy/layers/<org>/` and never touches the repo root.

## Layers

- **Core** (this repo): the kernel, `unfer_ffi`, `prob_kernel`, `unfer_edge`,
  `unfer_taler`, `unfer_consensus`, and the module/VM tier. Byte-identical —
  the release-manifest golden gate (`unfer_data/tests/release_manifest_golden.rs`,
  `UPDATE_GOLDEN=1` to regenerate) maps every deployable artifact byte→sha256.
- **Org layer** (`deploy/layers/<org>/`): org-specific material — grants,
  operator config, onboarding. Kept out of the byte-identical core.

## Stack components

| Component        | Role                                                        | Repo      |
|------------------|-------------------------------------------------------------|-----------|
| `unfer_ffi`      | C ABI kernel surface (`uk_*`/`uz_*`)                        | unfer     |
| `unfer_edge`     | Pingora edge (agent protocol over HTTP, S22 admin seam)     | unfer     |
| `unfer_taler`    | GNU Taler exchange adapter over the cert ledger (Plan R)    | unfer     |
| `unfer_consensus`| QuePaxa-style consensus, certificate/auction ledgers        | unfer     |
| `unfer_nixvm`    | Nix flake packaging `unfer_ffi` in a cloud-hypervisor VM    | unfer     |
| `australVM`      | module host (`modhost`), JIT, workerd/ecma/tidepool/capstd  | australVM |
| `velysterm`      | frontend: `kernel_client`, `mathed`, `mathed_mini`          | velysterm |

## Security posture (H9)

Set per deployment via `uk_posture_set` (`dangerous|auto|strict`,
operator-console only). The strict posture pauses every `EffectKind::Mutate`
`uk_*` for approval except the two no-effect turn enders. See PROTOCOL.md
"three portal-only walls".

## Onboarding

`tools/onboard_federation.sh --org <slug> --dry-run` prints the onboarding steps
without writing to the core. Run the dry-run first; it must pass with the core
`git status` clean (nothing in the repo root was modified).

## Releasing

A release bumps the golden manifest: `UPDATE_GOLDEN=1 cargo test -p unfer_data
-t release_manifest_golden`, reviewed as an explicit, deliberate byte change.