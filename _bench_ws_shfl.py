"""Benchmark v2 with the shfl-broadcast WG-id post-process applied via a
TileLang nvcc-callback hook.

The hook intercepts CUDA source on its way to nvcc, runs the same regex
transform that _postproc_ws_shfl.py does, and feeds the patched source to
nvcc. Everything else (TileLang adapter, host wrapper, TMA descriptors,
launch path) stays untouched.

Usage:
    TILELANG_DISABLE_CACHE=1 CUDA_VISIBLE_DEVICES=1 \
        python3 _bench_ws_shfl.py
"""
import os
import re
import sys
import time
from pathlib import Path

# Force a clean compile so the patched callback actually runs.
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import torch
import tvm_ffi

import tilelang
from tilelang.contrib import nvcc
from tilelang.engine.lower import tilelang_callback_cuda_compile as _orig_cb
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.transform import PassConfigKey


# ---------------------------------------------------------------------------
# Shfl post-process (same logic as _postproc_ws_shfl.py)
# ---------------------------------------------------------------------------
WG_PATTERNS = [
    (
        "WG1_range",
        r"\(\(128 <= \(\(int\)threadIdx\.x\)\) && \(\(\(int\)threadIdx\.x\) < 256\)\)",
        "(__wg_id == 1)",
    ),
    (
        "WG2_range",
        r"\(256 <= \(\(int\)threadIdx\.x\)\)",
        "(__wg_id == 2)",
    ),
    (
        "WG0_range",
        r"\(\(\(int\)threadIdx\.x\) < 128\)",
        "(__wg_id == 0)",
    ),
]

# wgmma descriptor by-value rewrite (path C-2a):
# - inject helper near top
# - rewrite tl::initialize_wgmma_descriptor<...>(name, ptr) → name = make_wgmma_descriptor_v<...>(ptr)
WGMMA_DESC_HELPER = r"""
// POST-PROCESSED: register-resident wgmma descriptor builder.
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

WGMMA_CALL_RE = re.compile(
    r"tl::initialize_wgmma_descriptor<\s*([^>]+?)\s*>\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(.+?)\s*\)\s*;",
    re.DOTALL,
)

ENABLE_DESC_REWRITE = os.environ.get("DESC_REWRITE", "0") == "1"
ENABLE_ALIAS = os.environ.get("ALIAS", "0") == "1"

# Path C-2b: alias _2 fragments to _1 storage via #define so per-thread
# fragment storage halves (each thread is in only one WG branch).
ALIAS_DECL_RE = re.compile(
    r"^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*_1)\[(\d+)\];\s*$",
    re.M,
)


def alias_rewrite(src: str) -> tuple[str, dict]:
    stats = {"alias_applied": False, "alias_pairs": 0}
    if "main_kernel" not in src:
        return src, stats

    decls_1 = {}
    for m in ALIAS_DECL_RE.finditer(src):
        _indent, dtype, name1, size = m.groups()
        decls_1[name1] = (dtype, int(size))

    out = src
    aliases = []
    for name1, (dtype, size) in decls_1.items():
        name2 = name1[:-2] + "_2"
        decl_2_re = re.compile(
            r"^(\s*)" + re.escape(dtype) + r"\s+" + re.escape(name2) +
            r"\[" + str(size) + r"\];\s*$",
            re.M,
        )
        m2 = decl_2_re.search(out)
        if m2:
            out = out[:m2.start()] + out[m2.end():]
            aliases.append((name1, name2))

    if not aliases:
        return src, stats

    anchor = 'extern "C" __global__'
    idx = out.find(anchor)
    if idx < 0:
        return src, stats

    define_block = "// POST-PROCESSED: alias _2 fragments to _1 storage\n"
    for n1, n2 in aliases:
        define_block += f"#define {n2} {n1}\n"
    define_block += "\n"
    out = out[:idx] + define_block + out[idx:]

    stats["alias_applied"] = True
    stats["alias_pairs"] = len(aliases)
    return out, stats

WG_ID_DECL = (
    "  // POST-PROCESSED: warp-uniform warpgroup idx via shuffle broadcast\n"
    "  int __wg_id = __shfl_sync(0xffffffff, ((int)threadIdx.x) / 128, 0);\n"
)

INSERT_ANCHOR = re.compile(r"^(\s*)if \(tl::tl_shuffle_elect<0>\(\)\) \{", re.M)


def shfl_transform(src: str) -> tuple[str, dict]:
    stats = {"applied": False}
    if "main_kernel" not in src:
        return src, stats
    if "(128 <= ((int)threadIdx.x))" not in src:
        # Not the v2 shape — leave alone.
        return src, stats

    m = INSERT_ANCHOR.search(src)
    if not m:
        return src, stats

    out = src[:m.start()] + WG_ID_DECL + src[m.start():]
    for tag, pat, repl in WG_PATTERNS:
        n = len(re.findall(pat, out))
        out = re.sub(pat, repl, out)
        stats[tag] = n
    stats["applied"] = True
    return out, stats


def desc_rewrite(src: str) -> tuple[str, dict]:
    """Path C-2a: rewrite wgmma descriptor init to return-by-value."""
    stats = {"desc_applied": False, "desc_calls": 0}
    if "initialize_wgmma_descriptor" not in src:
        return src, stats
    # Inject helper before the kernel decl.
    anchor = "extern \"C\" __global__"
    idx = src.find(anchor)
    if idx < 0:
        return src, stats
    out = src[:idx] + WGMMA_DESC_HELPER + "\n" + src[idx:]
    out, n = WGMMA_CALL_RE.subn(
        lambda m: f"{m.group(2).strip()} = make_wgmma_descriptor_v<{m.group(1).strip()}>({m.group(3).strip()});",
        out,
    )
    stats["desc_applied"] = True
    stats["desc_calls"] = n
    return out, stats


# ---------------------------------------------------------------------------
# Patched nvcc callback — registered as a TVM FFI global func.
# ---------------------------------------------------------------------------
@tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
def patched_cuda_compile(code, target, pass_config=None):
    patched, stats = shfl_transform(code)
    if ENABLE_DESC_REWRITE:
        patched, dstats = desc_rewrite(patched)
        stats.update(dstats)
    if ENABLE_ALIAS:
        patched, astats = alias_rewrite(patched)
        stats.update(astats)

    dump_dir = Path("/tmp")
    if stats.get("applied") or stats.get("desc_applied") or stats.get("alias_applied"):
        (dump_dir / "v2_orig_hook.cu").write_text(code)
        (dump_dir / "v2_shfl_hook.cu").write_text(patched)
        print(f"[shfl-hook] applied: {stats}", file=sys.stderr, flush=True)
    else:
        print(f"[shfl-hook] skipped (not v2 shape)", file=sys.stderr, flush=True)

    # Reproduce the original nvcc invocation logic with the patched source.
    target_arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))
    arch = [f"-arch=sm_{target_arch}"]

    cfg = pass_config or {}
    enable_fast_math = bool(cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False))
    ptxas_usage_level = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
    verbose_ptxas_output = bool(cfg.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False))

    options = [
        "-std=c++17",
        "-I" + TILELANG_TEMPLATE_PATH,
        "-I" + CUTLASS_INCLUDE_DIR,
    ]
    extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
    if extra_flags:
        import shlex
        if isinstance(extra_flags, str):
            options += shlex.split(extra_flags)
        else:
            for f in extra_flags:
                if isinstance(f, str):
                    options += shlex.split(f)
                else:
                    options.append(str(f))

    if enable_fast_math:
        options.append("--use_fast_math")
    if ptxas_usage_level is not None:
        options.append(f"--ptxas-options=--register-usage-level={ptxas_usage_level}")
    verbose = False
    if verbose_ptxas_output:
        options.append("--ptxas-options=--verbose")
        options.append("-w")
        verbose = True

    return nvcc.compile_cuda(patched, "cubin", arch, options=options, verbose=verbose)


# ---------------------------------------------------------------------------
# Now import the kernel builder — must happen AFTER the hook is registered.
# ---------------------------------------------------------------------------
from _test_ws_fa3_v2 import build_fa3_v2, ref_gqa


# ---------------------------------------------------------------------------
# Correctness + benchmark
# ---------------------------------------------------------------------------
def fa3_flops(B, S, H, D, is_causal):
    # 2 matmuls in attention; 4*B*H*S^2*D total flops, halved if causal.
    f = 4.0 * B * H * S * S * D
    if is_causal:
        f *= 0.5
    return f


def bench(B, S, H, Hkv, D, is_causal, n_warmup=10, n_iter=50):
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

    kernel = build_fa3_v2(B, S, H, Hkv, D, is_causal)(block_m=128, block_n=128)

    # Correctness
    o, lse = kernel(q, k, v)
    o_ref = ref_gqa(q, k, v, is_causal)
    diff = (o.float() - o_ref.float()).abs().max().item()

    # Warm up
    for _ in range(n_warmup):
        kernel(q, k, v)
    torch.cuda.synchronize()

    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        kernel(q, k, v)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / n_iter

    flops = fa3_flops(B, S, H, D, is_causal)
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    cfg = f"B={B} S={S} H={H} Hkv={Hkv} D={D} causal={is_causal}"
    print(f"  {cfg}: {elapsed_ms:.3f} ms  {tflops:.1f} TFLOPS  diff={diff:.4f}")
    return diff < 0.5, tflops


def disable_shfl_hook():
    """Re-register the original callback so the next compile uses unpatched code."""
    tvm_ffi.register_global_func(
        "tilelang_callback_cuda_compile", _orig_cb, override=True)


def enable_shfl_hook():
    tvm_ffi.register_global_func(
        "tilelang_callback_cuda_compile", patched_cuda_compile, override=True)


if __name__ == "__main__":
    print("=" * 70)
    print("v2 ablation: original vs shfl post-process")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"  TILELANG_DISABLE_CACHE = {os.environ.get('TILELANG_DISABLE_CACHE')}")
    print(f"  device = {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    shape = (4, 2048, 64, 4, 128, False)

    print()
    print("--- A: baseline (C-1 source only) ---")
    disable_shfl_hook()
    _ok_a, tflops_a = bench(*shape)

    saved_desc = ENABLE_DESC_REWRITE

    print()
    print("--- B: + shfl WG-id ---")
    ENABLE_DESC_REWRITE = False
    enable_shfl_hook()
    _ok_b, tflops_b = bench(*shape)

    print()
    print("--- C: + desc-by-value (no shfl) ---")
    ENABLE_DESC_REWRITE = True
    # Hook detects no WG-range pattern when shfl is off — emulate by
    # bypassing shfl insertion: temporarily disable hook then a desc-only path.
    # Simpler: keep shfl_transform a no-op and only run desc_rewrite.
    # (We'll just call enable_shfl_hook with desc enabled — shfl portion still
    # runs since the source has the WG-range branches; mark this as "B+desc".)
    # To get pure C, we need to skip shfl. Quick hack: use a local override.
    @tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
    def desc_only_cb(code, target, pass_config=None):
        patched, dstats = desc_rewrite(code)
        print(f"[desc-only] {dstats}", file=sys.stderr, flush=True)
        target_arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))
        opts = ["-std=c++17", "-I" + TILELANG_TEMPLATE_PATH, "-I" + CUTLASS_INCLUDE_DIR]
        cfg = pass_config or {}
        extra = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
        if extra:
            import shlex
            for f in extra: opts += shlex.split(f) if isinstance(f, str) else [str(f)]
        if cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False): opts.append("--use_fast_math")
        return nvcc.compile_cuda(patched, "cubin", [f"-arch=sm_{target_arch}"], options=opts)
    _ok_c, tflops_c = bench(*shape)

    print()
    print("--- D: + shfl + desc-by-value ---")
    enable_shfl_hook()  # restores both
    _ok_d, tflops_d = bench(*shape)

    print()
    print(f"A baseline (C-1 only):     {tflops_a:.1f} TFLOPS")
    print(f"B + shfl:                  {tflops_b:.1f} TFLOPS  ({(tflops_b-tflops_a):+.1f}, {tflops_b/tflops_a:.3f}x)")
    print(f"C + desc-by-value (alone): {tflops_c:.1f} TFLOPS  ({(tflops_c-tflops_a):+.1f}, {tflops_c/tflops_a:.3f}x)")
    print(f"D + shfl + desc-by-value:  {tflops_d:.1f} TFLOPS  ({(tflops_d-tflops_a):+.1f}, {tflops_d/tflops_a:.3f}x)")
    ENABLE_DESC_REWRITE = saved_desc
