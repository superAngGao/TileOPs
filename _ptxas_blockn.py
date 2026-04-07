"""ptxas verbose for block_n = 128 / 176 / 192 to check reg spill cliff."""
import os
os.environ.setdefault("SHFL", "0")
os.environ.setdefault("ALIAS", "0")
os.environ.setdefault("DESC_REWRITE", "0")
os.environ.setdefault("IF_ELSE_CHAIN", "0")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import re
import subprocess
import tempfile
from _test_ws_fa3_v2_threadbind import build_fa3_v2

B, H, Hkv, D = 4, 64, 8, 128
S = 4224


def get_source(compiled):
    for attr in ("get_kernel_source", "get_source", "source"):
        if hasattr(compiled, attr):
            v = getattr(compiled, attr)
            return v() if callable(v) else v
    return None


for bn in (128, 176, 192):
    print(f"\n=== block_n={bn} ===")
    kf = build_fa3_v2(B, S, H, Hkv, D, False)
    src = get_source(kf(block_m=128, block_n=bn))
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
        f.write(src)
        cu_path = f.name
    cmd = [
        "nvcc", "-O3", "-arch=sm_90a", "-DENABLE_BF16",
        "-I", "/home/ga/anaconda3/envs/env_tilelang_20260119/lib/python3.12/site-packages/tilelang/src",
        "-Xptxas", "-v",
        "-cubin", "-o", "/dev/null", cu_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stderr + p.stdout
    for ln in out.splitlines():
        if any(s in ln for s in ("ptxas info", "warning", "C75", "registers", "stack", "spill", "smem")):
            print(f"  {ln}")
