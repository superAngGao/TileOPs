"""ncu driver for the best v2 configuration so far:
- C-1 source fix (committed)
- Option 1 in-place common.h patch (in conda env)
- alias _2 → _1 fragments (post-process via ALIAS=1)
- shfl WG-id (post-process, always-on in patched callback)
- block_n=64

396 TFLOPS / 61% of FA3.

Run with:
    ncu --set full --nvtx --nvtx-include "v2_best/" \
        -o /tmp/v2_best.ncu-rep \
        python3 _ncu_v2_best.py
"""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
os.environ["ALIAS"] = "1"
os.environ["DESC_REWRITE"] = "1"

import torch
import _bench_ws_shfl as bw
bw.enable_shfl_hook()

from _test_ws_fa3_v2 import build_fa3_v2

B, S, H, Hkv, D = 4, 2048, 64, 4, 128
torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

kernel = build_fa3_v2(B, S, H, Hkv, D, False,
                     block_m=128, block_n=64)(block_m=128, block_n=64)

# Warm up
for _ in range(5):
    kernel(q, k, v)
torch.cuda.synchronize()

torch.cuda.nvtx.range_push("v2_best")
kernel(q, k, v)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
