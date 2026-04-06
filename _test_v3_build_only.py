"""Just try to build v3 with no hooks. Time it."""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import time
import torch
from _test_ws_fa3_v3 import build_fa3_v2 as build_fa3_v3

print("Building v3 (block_m=128, block_n=128, NO HOOKS)...")
t0 = time.time()
try:
    kernel_builder = build_fa3_v3(4, 2048, 64, 4, 128, False,
                                   block_m=128, block_n=128)
    print(f"  build_fa3_v3() returned in {time.time()-t0:.1f}s")
    t1 = time.time()
    kernel = kernel_builder(block_m=128, block_n=128)
    print(f"  kernel_builder() returned in {time.time()-t1:.1f}s")
    print(f"Total: {time.time()-t0:.1f}s")
    src = kernel.get_kernel_source()
    with open("/tmp/v3_raw.cu", "w") as f:
        f.write(src)
    print(f"Wrote /tmp/v3_raw.cu ({len(src.splitlines())} lines)")
    print()
    # Run for correctness
    print()
    print("Running v3 for correctness check...")
    B, S, H, Hkv, D = 4, 2048, 64, 4, 128
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    o, lse = kernel(q, k, v)

    from _test_ws_fa3_v3 import ref_gqa
    o_ref = ref_gqa(q, k, v, False)
    diff = (o.float() - o_ref.float()).abs().max().item()
    print(f"  diff: {diff:.4f}  {'PASS' if diff < 0.5 else 'FAIL'}")

    # Quick TFLOPS
    for _ in range(5):
        kernel(q, k, v)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        kernel(q, k, v)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 20
    flops = 4.0 * B * H * S * S * D
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"  bench: {ms:.3f} ms  {tflops:.1f} TFLOPS")
except Exception as e:
    print(f"FAILED at {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
