"""Post-process v2's lowered CUDA to alias _1 and _2 fragments to the
same physical storage.

Insight: per thread, only one warpgroup branch executes, so a thread that
enters WG1's body never touches the _2 fragments and vice versa. ptxas
allocates per-thread storage for the UNION of both branches' fragments
because its register allocator is not warpgroup-aware. By aliasing
`_2` names to `_1` storage at the C++ level, we tell the compiler the
two are the same physical buffer — halving the per-thread fragment
budget.

Implementation: find each `<type> name_1[N];` declaration, then for each
matching `<type> name_2[N];` declaration, delete it and inject a
`#define name_2 name_1` near the top.
"""
import re
import subprocess
import sys
from pathlib import Path

# Match a function-scope local declaration like "  float acc_s_1[64];"
DECL_RE = re.compile(
    r"^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*_1)\[(\d+)\];\s*$",
    re.M,
)

def transform(src: str) -> tuple[str, dict]:
    stats = {"aliased": [], "removed": []}
    if "main_kernel" not in src:
        return src, stats

    # Pass 1: collect _1 declarations
    decls_1 = {}  # name_1 -> (dtype, size)
    for m in DECL_RE.finditer(src):
        indent, dtype, name1, size = m.groups()
        decls_1[name1] = (dtype, int(size))

    # Pass 2: for each _1, look for matching _2 with same dtype/size
    aliases = []  # list of (name_1, name_2)
    new_src = src
    for name1, (dtype, size) in decls_1.items():
        name2 = name1[:-2] + "_2"
        # Match the _2 declaration with same dtype/size
        decl_2_re = re.compile(
            r"^(\s*)" + re.escape(dtype) + r"\s+" + re.escape(name2) + r"\[" + str(size) + r"\];\s*$",
            re.M,
        )
        m2 = decl_2_re.search(new_src)
        if m2:
            # Remove the _2 declaration
            new_src = new_src[:m2.start()] + new_src[m2.end():]
            aliases.append((name1, name2))
            stats["aliased"].append((name1, name2, dtype, size))

    if not aliases:
        return src, stats

    # Inject #define directives just before extern "C" __global__
    anchor = 'extern "C" __global__'
    idx = new_src.find(anchor)
    if idx < 0:
        return src, stats

    define_block = "// POST-PROCESSED: alias _2 fragments to _1 storage\n"
    define_block += "// (each thread is in exactly one WG; storage can be shared)\n"
    for n1, n2 in aliases:
        define_block += f"#define {n2} {n1}\n"
    define_block += "\n"

    new_src = new_src[:idx] + define_block + new_src[idx:]
    return new_src, stats


def compile_and_capture(cu_path: Path, out_cubin: Path) -> str:
    include_dirs = [
        "/home/ga/anaconda3/envs/env_tilelang_20260119/lib/python3.12/site-packages/tilelang/src",
        "/home/ga/anaconda3/envs/env_tilelang_20260119/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include",
    ]
    cmd = [
        "nvcc", "-arch=sm_90a", "-O3", "-DENABLE_BF16",
    ]
    for d in include_dirs:
        cmd.extend(["-I", d])
    cmd.extend([
        "-Xptxas=-v,-warn-lmem-usage",
        "-cubin", "-o", str(out_cubin), str(cu_path),
    ])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + "\n" + r.stderr


def main():
    if len(sys.argv) != 2:
        print("usage: python3 _postproc_alias_fragments.py <input.cu>")
        sys.exit(1)

    src_path = Path(sys.argv[1])
    src = src_path.read_text()
    out, stats = transform(src)
    out_path = src_path.with_name(src_path.stem + "_alias.cu")
    out_path.write_text(out)

    print(f"Input:  {src_path}")
    print(f"Output: {out_path}")
    print(f"Aliased {len(stats['aliased'])} fragment pairs:")
    for n1, n2, dt, sz in stats["aliased"]:
        bytes_each = sz * (4 if dt == "float" else 2 if "half" in dt else 4)
        print(f"  {n2} → {n1}  ({dt}[{sz}], saves {bytes_each} B/thread)")
    print()

    print("=" * 70)
    print("Compiling ORIGINAL:")
    print("=" * 70)
    log = compile_and_capture(src_path, Path("/tmp/v2_alias_orig.cubin"))
    for line in log.splitlines():
        if any(k in line for k in ["stack frame", "spill", "registers", "C75"]):
            print(f"  {line}")

    print()
    print("=" * 70)
    print("Compiling ALIASED:")
    print("=" * 70)
    log = compile_and_capture(out_path, Path("/tmp/v2_alias.cubin"))
    err = [l for l in log.splitlines() if "error" in l.lower()]
    if err:
        print("ERROR:")
        for l in err[:10]:
            print(f"  {l}")
        return
    for line in log.splitlines():
        if any(k in line for k in ["stack frame", "spill", "registers", "C75"]):
            print(f"  {line}")


if __name__ == "__main__":
    main()
