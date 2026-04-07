"""Minimal FA3 bench for the same GQA shape we use for v2 thread-bind.

Shape: B=4, S=4096 or 4224, H=64, Hkv=8, D=128, fp16, non-causal.
Run on locked-freq GPU (1, 2, 7) for clean comparison.
"""
import sys
import time
import torch
from flash_attn_interface import flash_attn_func as flash_attn_func_v3

# CLI: python3 _bench_fa3_gqa.py [S]
S = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
B = 4
H = 64
Hkv = 8
D = 128
is_causal = False
n_warmup = 5
n_trials = 5

print(f"=== FA3 GQA bench ===")
print(f"shape: B={B} S={S} H={H} Hkv={Hkv} D={D} causal={is_causal} dtype=fp16")
print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

# Sanity: do one call to make sure shape is accepted
try:
    out = flash_attn_func_v3(q, k, v, causal=is_causal)
    if isinstance(out, tuple):
        out = out[0]
    print(f"  output shape: {tuple(out.shape)}")
except Exception as e:
    print(f"  FA3 call failed: {type(e).__name__}: {e}")
    sys.exit(1)

# Warmup
for _ in range(n_warmup):
    flash_attn_func_v3(q, k, v, causal=is_causal)
torch.cuda.synchronize()

# 2 * (QK + PV) = 4 mat-muls, but with GQA the K/V matmul stays at full H
# Total flops: 4 * B * H * S * S * D (matches both v2 and FA3 conventions)
flops = 4 * B * H * S * S * D
if is_causal:
    flops //= 2

times = []
for i in range(n_trials):
    torch.cuda.synchronize()
    t0 = time.time()
    flash_attn_func_v3(q, k, v, causal=is_causal)
    torch.cuda.synchronize()
    dt = time.time() - t0
    times.append(dt)
    tflops = flops / dt / 1e12
    print(f"  trial {i+1}: {dt*1000:.4f} ms  {tflops:.1f} TFLOPS")

times.sort()
median = times[len(times) // 2]
print(f"  median: {median*1000:.4f} ms  {flops/median/1e12:.1f} TFLOPS")
