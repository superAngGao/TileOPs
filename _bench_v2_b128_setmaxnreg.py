"""Bench v2 b128 with the restructured per-WG body layout that finally
makes setmaxnreg work (stack:0, no C7507).

Reference shape: B=4 S=2048 H=64 Hkv=4 D=128 non-causal fp16
Prior baselines on the same shape:
  - v2 best (block_n=64, alias+shfl+option1):           396 TFLOPS (61% FA3)
  - v2 b128 (alias+shfl+option1, no setmaxnreg):       ~329 TFLOPS (51% FA3)
  - v3 b128 NUM_SMS=132 (persistent, ws3 fixes):       ~327 TFLOPS (51% FA3)
  - FA3 reference:                                      647 TFLOPS (100%)

Active hooks (post-ablation, see issue #9):
  - shfl WG-id (always on, dependency of if_else_chain)
  - if_else_chain (default IF_ELSE_CHAIN=1, REQUIRED, +66% perf vs disabled)

Hooks NO LONGER needed after the per-WG restructure:
  - alias _2→_1 (restructure makes WG fragment live ranges disjoint)
  - desc_rewrite (Hack 1 covered it; Hack 1 itself also redundant)
  - common.h Option 1 in-place patch (kept reverted to upstream)
"""
import os
os.environ.setdefault("IF_ELSE_CHAIN", "1")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import time
import torch
import _bench_ws_shfl  # registers shfl/desc/alias/if-else hook chain
from _test_ws_fa3_v2 import build_fa3_v2, ref_gqa


def fa3_flops(B, S, H, D, is_causal):
    f = 4.0 * B * H * S * S * D
    if is_causal:
        f *= 0.5
    return f


def bench(B, S, H, Hkv, D, is_causal, block_m=128, block_n=128,
          n_warmup=20, n_iter=100):
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

    print(f"  building (block_m={block_m}, block_n={block_n})...", end=" ", flush=True)
    t0 = time.time()
    kernel = build_fa3_v2(B, S, H, Hkv, D, is_causal)(
        block_m=block_m, block_n=block_n)
    print(f"{time.time()-t0:.1f}s")

    # Correctness
    o, lse = kernel(q, k, v)
    o_ref = ref_gqa(q, k, v, is_causal)
    diff = (o.float() - o_ref.float()).abs().max().item()
    ok = diff < 0.5

    # Warmup
    for _ in range(n_warmup):
        kernel(q, k, v)
    torch.cuda.synchronize()

    # Bench
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        kernel(q, k, v)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / n_iter

    flops = fa3_flops(B, S, H, D, is_causal)
    tflops = flops / (ms * 1e-3) / 1e12

    print(f"  shape B={B} S={S} H={H} Hkv={Hkv} D={D} causal={is_causal}")
    print(f"    diff:    {diff:.4f}  {'PASS' if ok else 'FAIL'}")
    print(f"    time:    {ms:.3f} ms")
    print(f"    TFLOPS:  {tflops:.1f}")
    return tflops, diff, ok


if __name__ == "__main__":
    print("=" * 70)
    print("v2 b128 + per-WG self-contained body + setmaxnreg(24/240)")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    print(f"  device = {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    print("\n--- Main shape: B=4 S=2048 H=64 Hkv=4 D=128 non-causal ---")
    tflops_b128, _, _ = bench(4, 2048, 64, 4, 128, False, block_n=128)

    print("\n--- For comparison: same shape, block_n=64 (v2 prev best path) ---")
    tflops_b64, _, _ = bench(4, 2048, 64, 4, 128, False, block_n=64)

    print()
    print("=" * 70)
    print("Summary on B=4 S=2048 H=64 Hkv=4 D=128 non-causal")
    print("=" * 70)
    print(f"  FA3 ref:                  647.3 TFLOPS  (100%)")
    print(f"  v2 best (prev, b64):      396.0 TFLOPS   (61%)")
    print(f"  v2 b128 prev (no nreg):   329.4 TFLOPS   (51%)")
    print(f"  v3 b128 NUM_SMS=132:      327.8 TFLOPS   (51%)")
    print()
    print(f"  v2 b128 NEW (this run):   {tflops_b128:.1f} TFLOPS"
          f"   ({tflops_b128/647.3*100:.0f}%)")
    print(f"  v2 b64  NEW (this run):   {tflops_b64:.1f} TFLOPS"
          f"   ({tflops_b64/647.3*100:.0f}%)")
