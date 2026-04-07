"""Anchor bench: v2 thread-bind b128 at S=4096 on locked GPU.

Compares against historical '525 TFLOPS' which was on unlocked GPU.
"""
import os
os.environ.setdefault("SHFL", "0")
os.environ.setdefault("ALIAS", "0")
os.environ.setdefault("DESC_REWRITE", "0")
os.environ.setdefault("IF_ELSE_CHAIN", "0")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import torch
import time
from _test_ws_fa3_v2_threadbind import build_fa3_v2

B, H, Hkv, D = 4, 64, 8, 128
S = 4096
is_causal = False
n_warmup = 5
n_trials = 5

print(f"=== v2 thread-bind b128 anchor ===")
print(f"shape: B={B} S={S} H={H} Hkv={Hkv} D={D} causal={is_causal}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

kernel = build_fa3_v2(B, S, H, Hkv, D, is_causal)
compiled = kernel(block_m=128, block_n=128)

# Correctness
o, lse = compiled(q, k, v)

# Warmup
for _ in range(n_warmup):
    compiled(q, k, v)
torch.cuda.synchronize()

flops = 4 * B * H * S * S * D
times = []
for i in range(n_trials):
    torch.cuda.synchronize()
    t0 = time.time()
    compiled(q, k, v)
    torch.cuda.synchronize()
    dt = time.time() - t0
    times.append(dt)
    print(f"  trial {i+1}: {dt*1000:.4f} ms  {flops/dt/1e12:.1f} TFLOPS")
times.sort()
median = times[len(times) // 2]
print(f"  median: {median*1000:.4f} ms  {flops/median/1e12:.1f} TFLOPS")
