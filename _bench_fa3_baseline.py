"""Measure FA3 reference TFLOPS on the same shape we use for v2."""
import torch
from flash_attn_interface import flash_attn_func as fa3

B, S, H, Hkv, D = 4, 2048, 64, 4, 128
is_causal = False

torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Shape: B={B} S={S} H={H} Hkv={Hkv} D={D} causal={is_causal}")

# Warm up
for _ in range(20):
    out = fa3(q, k, v, causal=is_causal)
torch.cuda.synchronize()

# Time
n_iter = 100
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(n_iter):
    out = fa3(q, k, v, causal=is_causal)
e.record()
torch.cuda.synchronize()
ms = s.elapsed_time(e) / n_iter

flops = 4.0 * B * H * S * S * D
if is_causal:
    flops *= 0.5
tflops = flops / (ms * 1e-3) / 1e12

print(f"FA3: {ms:.3f} ms  {tflops:.1f} TFLOPS")
