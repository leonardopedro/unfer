#!/usr/bin/env python3
"""Regenerate unfer_ffi/include/unfer_kernel.h from the Rust ABI sources.

Parses `pub extern "C" fn NAME(...) -> RET` declarations in src/lib.rs and
src/zenodo.rs and emits a C header declaring every symbol with the same
signature. Run from the crate root:

    python3 gen_unfer_kernel_h.py

The header is checked by tests/abi_header.rs (drift fails CI).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SOURCES = [ROOT / "src/lib.rs", ROOT / "src/zenodo.rs"]
HEADER = ROOT / "include/unfer_kernel.h"

TYPE_MAP = {
    "i64": "int64_t",
    "u64": "uint64_t",
    "i32": "int32_t",
    "u32": "uint32_t",
    "usize": "size_t",
    "isize": "ssize_t",
    "f64": "double",
    "f32": "float",
    "u8": "uint8_t",
    "i8": "int8_t",
    "*const u8": "const uint8_t*",
    "*mut u8": "uint8_t*",
    "*const i8": "const char*",
    "*mut i8": "char*",
    "*const void": "const void*",
    "*mut void": "void*",
}

FN_RE = re.compile(
    r'pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+([a-z0-9_]+)\s*\('
    r"(?P<args>[^)]*)\)\s*->\s*(?P<ret>[^\s{]+)",
    re.S,
)


def parse_sig(src: str):
    out = []
    for m in FN_RE.finditer(src):
        name = m.group(1)
        ret = m.group("ret").strip()
        if ret not in (None, "", "impl", "async"):
            ret = ret.rstrip("}")
        args = m.group("args").strip()
        params = []
        if args and args != "void":
            for chunk in args.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                parts = chunk.split(":")
                if len(parts) != 2:
                    continue
                argname = parts[0].strip()
                ty = parts[1].strip()
                if ty.endswith("}"):
                    ty = ty.rsplit("}", 1)[0].rsplit("{", 1)[0].strip()
                if ty not in TYPE_MAP:
                    print(f"  !! unknown arg type {ty!r} in {name} arg {argname}", file=sys.stderr)
                    ty = f"/* {ty} */ int64_t"
                params.append((ty, argname))
        out.append((name, ret, params))
    return out


def c_decl(name, ret, params, is_zenodo):
    ret_c = TYPE_MAP.get(ret)
    if not ret_c:
        print(f"  !! unknown return type {ret!r} in {name}", file=sys.stderr)
        ret_c = "int64_t"
    pad = " " * (len(name) + 1)
    line = f"{ret_c} {name}("
    for i, (ty, an) in enumerate(params):
        ty_c = TYPE_MAP.get(ty, "int64_t")
        sep = ",\n" + pad if i else ""
        line += f"{sep}{ty_c} {an}"
    line += ");"
    return line


def main():
    sigs = []
    for src_path in SOURCES:
        src = src_path.read_text()
        for (name, ret, params) in parse_sig(src):
            is_zenodo = name.startswith("uz_")
            sigs.append((name, ret, params, is_zenodo, src_path.name))
    sigs.sort(key=lambda s: s[0])
    if not sigs:
        print("no symbols parsed!", file=sys.stderr)
        return 1
    head = ["/*", " * unfer_kernel.h — C ABI for the unfer probability kernel.", " *",
            " * GENERATED FILE — do not edit by hand. Regenerate with:", " *",
            " *     python3 gen_unfer_kernel_h.py", " *",
            " * All functions use i64-compatible parameters (ptr+len; ...) to match the CPS IR",
            " * calling convention. Return convention:",
            " *   >= 0 : success (handle, byte count, or 0)",
            " *   <  0 : error (-code); call uk_last_error() for a Diagnostic JSON.",
            " *",
            " * The `uz_*` declarations require building unfer_ffi with `--features zenodo`.",
            " */",
            "#ifndef UNFER_KERNEL_H",
            "#define UNFER_KERNEL_H",
            "",
            "#include <stdint.h>",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            "/* ABI version (see unfer_protocol::KERNEL_VERSION). */",
    ]
    body = []
    body.append("int64_t uk_version(void);")
    pending_zenodo = []
    for (name, ret, params, is_zenodo, _src) in sigs:
        if name == "uk_version":
            continue
        decl = c_decl(name, ret, params, is_zenodo)
        (pending_zenodo if is_zenodo else body).append(decl)
    tail = ["#ifdef __cplusplus", "}", "#endif",
            "", "#endif /* UNFER_KERNEL_H */", ""]
    HEADER.write_text("\n".join(head + body + [""] + pending_zenodo + [""] + tail))
    print(f"wrote {HEADER}: {len(sigs)} symbols ({sum(1 for s in sigs if s[0].startswith('uk_'))} uk_, {sum(1 for s in sigs if s[0].startswith('uz_'))} uz_)")
    return 0


if __name__ == "__main__":
    sys.exit(main())