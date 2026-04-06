"""Post-process the TileLang-generated CUDA so wgmma descriptors are
constructed by-value (returning the full uint64) instead of by-reference
through bitfield RMW.

Background: tl::initialize_wgmma_descriptor takes `GmmaDescriptor &` and
performs five bitfield writes. ptxas cannot promote the descriptor to
registers (its address is taken), so each bitfield op becomes a stack
RMW. This shows up as ~43 % of all spill traffic in v2 (260 of 607
static LDLs attributed to common.h:541-546).

Fix: emit a helper that computes the full uint64 descriptor in pure
register ops, and rewrite the call sites to assign via that helper.
After the rewrite the descriptor variable is only ever assigned and
read by value, so ptxas can SSA-promote it to a register pair.

Usage:
    python3 _postproc_wgmma_desc.py <input.cu>
    # then nvcc -arch=sm_90a ... -cubin -Xptxas=-v <input>_descv.cu
"""
import re
import subprocess
import sys
from pathlib import Path

# Helper definition that computes the descriptor's uint64 directly,
# avoiding any address-taken local struct on the producer side.
HELPER = r"""
// POST-PROCESSED: register-resident wgmma descriptor builder.
// Returns the descriptor by value so the caller-side variable can be
// SSA-promoted to a register pair (instead of being stack-resident due
// to the by-reference bitfield writes in tl::initialize_wgmma_descriptor).
template <int __LT = 0, int __LBO = 0, int __SBO = 0, typename __T>
__device__ __forceinline__ tl::GmmaDescriptor
make_wgmma_descriptor_v(__T *__addr) {
    tl::GmmaDescriptor __d;
    uint64_t __a14 = (cute::cast_smem_ptr_to_uint(__addr) >> 4) & 0x3fffull;
    __d.desc_ = __a14
              | (uint64_t(__LBO & 0x3fff) << 16)
              | (uint64_t(__SBO & 0x3fff) << 32)
              | ((uint64_t)(__LT & 0x3) << 62);
    return __d;
}
"""

# Match: tl::initialize_wgmma_descriptor<A, B, C>(name, ptr_expr);
# Capture: 1=template args, 2=descriptor name, 3=pointer expression
CALL_RE = re.compile(
    r"tl::initialize_wgmma_descriptor<\s*([^>]+?)\s*>\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(.+?)\s*\)\s*;",
    re.DOTALL,
)

ANCHOR = "extern \"C\" __global__"


def transform(src: str) -> tuple[str, dict]:
    stats = {"helper_inserted": False, "calls_rewritten": 0}

    if "main_kernel" not in src or "initialize_wgmma_descriptor" not in src:
        return src, stats

    # Insert helper just before the kernel decl.
    idx = src.find(ANCHOR)
    if idx < 0:
        # Fall back to inserting after the last #include.
        m = list(re.finditer(r"^#include[^\n]*\n", src, re.M))
        if m:
            idx = m[-1].end()
        else:
            idx = 0
    out = src[:idx] + HELPER + "\n" + src[idx:]
    stats["helper_inserted"] = True

    def repl(m):
        tpl_args = m.group(1).strip()
        name = m.group(2).strip()
        ptr = m.group(3).strip()
        return f"{name} = make_wgmma_descriptor_v<{tpl_args}>({ptr});"

    out, n = CALL_RE.subn(repl, out)
    stats["calls_rewritten"] = n
    return out, stats


def compile_and_capture(cu_path: Path, out_cubin: Path) -> str:
    include_dirs = [
        "/home/ga/anaconda3/envs/env_tilelang_20260119/lib/python3.12/site-packages/tilelang/src",
        "/home/ga/anaconda3/envs/env_tilelang_20260119/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include",
    ]
    cmd = [
        "nvcc",
        "-arch=sm_90a",
        "-O3",
        "-DENABLE_BF16",
    ]
    for d in include_dirs:
        cmd.extend(["-I", d])
    cmd.extend([
        "-Xptxas=-v,-warn-lmem-usage",
        "-cubin",
        "-o", str(out_cubin),
        str(cu_path),
    ])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + "\n" + r.stderr


def main():
    if len(sys.argv) != 2:
        print("usage: python3 _postproc_wgmma_desc.py <input.cu>")
        sys.exit(1)

    src_path = Path(sys.argv[1])
    src = src_path.read_text()

    print("=" * 70)
    print(f"Input: {src_path}")
    print("=" * 70)

    out, stats = transform(src)
    out_path = src_path.with_name(src_path.stem + "_descv.cu")
    out_path.write_text(out)
    print(f"Transform stats: {stats}")
    print(f"Wrote {out_path}")
    print()

    print("=" * 70)
    print("Compiling ORIGINAL:")
    print("=" * 70)
    log = compile_and_capture(src_path, Path("/tmp/v2_orig_descv_orig.cubin"))
    for line in log.splitlines():
        if any(kw in line for kw in [
            "ptxas info", "ptxas warn", "stack frame", "spill", "registers", "C75"
        ]) and "error" not in line.lower():
            print(f"  {line}")
    print()

    print("=" * 70)
    print("Compiling POST-PROCESSED (descriptor by value):")
    print("=" * 70)
    log = compile_and_capture(out_path, Path("/tmp/v2_descv.cubin"))
    err_lines = [l for l in log.splitlines() if "error" in l.lower()]
    if err_lines:
        print("ERROR:")
        for l in err_lines[:10]:
            print(f"  {l}")
        return
    for line in log.splitlines():
        if any(kw in line for kw in [
            "ptxas info", "ptxas warn", "stack frame", "spill", "registers", "C75"
        ]) and "error" not in line.lower():
            print(f"  {line}")


if __name__ == "__main__":
    main()
