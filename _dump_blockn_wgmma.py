"""Dump lowered CUDA for block_n=128 and 176, count wgmma instructions
to verify the TileLang wgmma split hypothesis."""
import os
os.environ.setdefault("SHFL", "0")
os.environ.setdefault("ALIAS", "0")
os.environ.setdefault("DESC_REWRITE", "0")
os.environ.setdefault("IF_ELSE_CHAIN", "0")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import re
from _test_ws_fa3_v2_threadbind import build_fa3_v2

B, H, Hkv, D = 4, 64, 8, 128
S = 4224
is_causal = False


def get_source(kernel_jit):
    """Try several attribute names to extract lowered CUDA source."""
    for attr in ("get_kernel_source", "get_source", "source"):
        if hasattr(kernel_jit, attr):
            v = getattr(kernel_jit, attr)
            return v() if callable(v) else v
    # Try the inner kernel
    inner = getattr(kernel_jit, "kernel", None)
    if inner is not None:
        for attr in ("get_kernel_source", "get_source", "source"):
            if hasattr(inner, attr):
                v = getattr(inner, attr)
                return v() if callable(v) else v
    return None


def analyze(block_n):
    print(f"\n=== block_n={block_n} ===")
    kernel_factory = build_fa3_v2(B, S, H, Hkv, D, is_causal)
    compiled = kernel_factory(block_m=128, block_n=block_n)
    src = get_source(compiled)
    if src is None:
        print(f"  could not extract source")
        return
    out_path = f"/tmp/v2tb_b{block_n}.cu"
    with open(out_path, "w") as f:
        f.write(src)
    print(f"  wrote {out_path} ({len(src)} bytes)")

    # Count various wgmma-related patterns
    patterns = {
        "tl::gemm_ss":            r"tl::gemm_ss",
        "tl::gemm_rs":            r"tl::gemm_rs",
        "wgmma.mma_async (ptx)":  r"wgmma\.mma_async",
        "warpgroup_arrive":       r"warpgroup_arrive",
        "warpgroup_commit_batch": r"warpgroup_commit_batch",
        "wgmma_fence":            r"wgmma_fence",
        "fence_proxy_async":      r"fence_proxy_async",
    }
    for name, pat in patterns.items():
        n = len(re.findall(pat, src))
        print(f"  {name:30s}: {n}")


if __name__ == "__main__":
    for bn in (128, 176, 192):
        analyze(bn)
