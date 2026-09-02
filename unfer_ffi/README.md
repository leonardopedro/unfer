# unfer_ffi

Handle-based C ABI for the unfer probability kernel. Exposes the `uk_*`
function set (and `uz_*` under the optional `zenodo` feature) as
`extern "C"` symbols over opaque session handles. Every call is
sanity-checked and unwinds panics across the FFI boundary.

- ABI surface: `uk_*`/`uz_*` `extern "C"` symbols. The authoritative count
  is the generated census (`EXPECTED_SYMBOLS.txt` / `EXPECTED_SYMBOLS_ZENODO.txt`),
  kept in sync by `scripts/gen_symbol_artifacts` — never a hand-maintained
  number in prose.
- The C header `include/unfer_kernel.h` is **generated** — run
  `python3 gen_unfer_kernel_h.py` to regenerate; never hand-edit it.
- `unfer_ffi` is packaged for Nix (`nix build .#unfer-ffi`) and linked
  statically by the australVM JIT (see `docs/FEDERATION.md`).