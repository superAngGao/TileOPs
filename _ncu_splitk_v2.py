"""ncu launcher: split-K v2 (block_m=64, block_n=64, double-buf per stream).
Production reference shape S=4096 H=64 Hkv=8 D=128 fp16 non-causal.
"""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import torch
from _test_ws_fa3_v2_tb_splitk import build_fa3_v2_splitk

B, S, H, Hkv, D = 4, 4096, 64, 8, 128
print(f"Building split-K v2 b{64} S={S}...")
kernel = build_fa3_v2_splitk(B, S, H, Hkv, D, False)(block_m=64, block_n=64)
print("Built.")

torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

for _ in range(5):
    kernel(q, k, v)
torch.cuda.synchronize()

for _ in range(3):
    kernel(q, k, v)
torch.cuda.synchronize()
print("done")
