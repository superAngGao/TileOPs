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


# ---------------------------------------------------------------------------
# Patched nvcc callback — registered as a TVM FFI global func.
# ---------------------------------------------------------------------------
@tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
def patched_cuda_compile(code, target, pass_config=None):
    patched, stats = shfl_transform(code)

    dump_dir = Path("/tmp")
    if stats.get("applied"):
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
    print("--- Baseline (no shfl patch) ---")
    disable_shfl_hook()
    _ok_a, tflops_a = bench(*shape)

    print()
    print("--- Patched (shfl WG-id) ---")
    enable_shfl_hook()
    _ok_b, tflops_b = bench(*shape)

    print()
    print(f"Speedup: {tflops_b / tflops_a:.3f}x  ({tflops_b - tflops_a:+.1f} TFLOPS)")
