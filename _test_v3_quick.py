"""Quick smoke test for v3 — just try to build and check correctness on
the main bench shape with all post-processing hooks active.
"""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
os.environ["ALIAS"] = "1"
os.environ["DESC_REWRITE"] = "1"

import torch
import _bench_ws_shfl as bw
bw.enable_shfl_hook()

import _test_ws_fa3_v3 as v3
print(f"NUM_SMS = {v3.NUM_SMS}")

from _test_ws_fa3_v3 import build_fa3_v2 as build_fa3_v3, ref_gqa


B, S, H, Hkv, D = 4, 2048, 64, 4, 128
torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

print("Building v3 (block_m=128, block_n=64)...")
kernel = build_fa3_v3(B, S, H, Hkv, D, False,
                      block_m=128, block_n=64)(block_m=128, block_n=64)

print("Running v3 once...")
o, lse = kernel(q, k, v)
o_ref = ref_gqa(q, k, v, False)
diff = (o.float() - o_ref.float()).abs().max().item()
print(f"v3 diff vs ref: {diff:.4f}  {'PASS' if diff < 0.5 else 'FAIL'}")

# Quick TFLOPS measurement
for _ in range(10):
    kernel(q, k, v)
torch.cuda.synchronize()
import torch.cuda
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(50):
    kernel(q, k, v)
e.record()
torch.cuda.synchronize()
ms = s.elapsed_time(e) / 50
flops = 4.0 * B * H * S * S * D
tflops = flops / (ms * 1e-3) / 1e12
print(f"v3 bench: {ms:.3f} ms  {tflops:.1f} TFLOPS  ({tflops/647.3*100:.1f}% of FA3)")
