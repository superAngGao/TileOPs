"""block_n sweep on thread-binding v2 — direction 4 from issue #9 followup.

Goal: see if larger block_n (matching FA3's 176) reduces the long_scoreboard
gap by giving consumer more wgmma work per K/V load (better producer/consumer
balance).

Constraints:
  - non-causal kernel does not mask tail iter, so S must be divisible by block_n
  - S=4224 divides 128/176/192 cleanly
  - block_n=192 may exceed 228KB smem budget (k+v double = 196KB + q 32KB = 228KB
    + mbarriers)
"""
import os
os.environ.setdefault("SHFL", "0")
os.environ.setdefault("ALIAS", "0")
os.environ.setdefault("DESC_REWRITE", "0")
os.environ.setdefault("IF_ELSE_CHAIN", "0")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import torch
import time
from _test_ws_fa3_v2_threadbind import build_fa3_v2, ref_gqa


def bench_blockn(B, S, H, Hkv, D, is_causal, block_n, n_warmup=5, n_trials=5):
    assert S % block_n == 0, f"S={S} not divisible by block_n={block_n}"

    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

    print(f"\n=== block_n={block_n} (B={B} S={S} H={H} Hkv={Hkv} D={D}) ===")
    t0 = time.time()
    try:
        kernel = build_fa3_v2(B, S, H, Hkv, D, is_causal)
        compiled = kernel(block_m=128, block_n=block_n)
    except Exception as e:
        print(f"  BUILD FAIL: {type(e).__name__}: {e}")
        return None
    print(f"  build: {time.time()-t0:.1f}s")

    # Correctness
    try:
        o, lse = compiled(q, k, v)
        o_ref = ref_gqa(q, k, v, is_causal)
        diff = (o.float() - o_ref.float()).abs().max().item()
        ok = diff < 0.5
        print(f"  diff={diff:.4f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            return None
    except Exception as e:
        print(f"  RUN FAIL: {type(e).__name__}: {e}")
        return None

    # FLOPs: 2 * (QK^T) + 2 * (PV)  with M=S, N=S, K=D, batch=B*H, causal halves
    flops = 2 * 2 * B * H * S * S * D
    if is_causal:
        flops //= 2

    # Warmup
    for _ in range(n_warmup):
        compiled(q, k, v)
    torch.cuda.synchronize()

    times = []
    for i in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.time()
        compiled(q, k, v)
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        tflops = flops / dt / 1e12
        print(f"  trial {i+1}: {dt*1000:.4f} ms  {tflops:.1f} TFLOPS")

    times.sort()
    median = times[len(times) // 2]
    median_tflops = flops / median / 1e12
    print(f"  median: {median*1000:.4f} ms  {median_tflops:.1f} TFLOPS")
    return median_tflops


if __name__ == "__main__":
    # Reference shape: GQA fwd, B=4, H=64, Hkv=8, D=128, S=4224
    B, H, Hkv, D = 4, 64, 8, 128
    S = 4224
    is_causal = False

    print(f"=== block_n sweep on thread-bind v2 ===")
    print(f"shape: B={B} S={S} H={H} Hkv={Hkv} D={D} causal={is_causal}")

    results = {}
    for block_n in [128, 176, 192]:
        try:
            tflops = bench_blockn(B, S, H, Hkv, D, is_causal, block_n)
            if tflops is not None:
                results[block_n] = tflops
        except Exception as e:
            print(f"  block_n={block_n} threw: {type(e).__name__}: {e}")

    print(f"\n=== Summary ===")
    if 128 in results:
        baseline = results[128]
        for bn, tf in sorted(results.items()):
            delta = (tf / baseline - 1) * 100
            print(f"  block_n={bn:3d}: {tf:6.1f} TFLOPS  ({delta:+.1f}%)")
    else:
        for bn, tf in sorted(results.items()):
            print(f"  block_n={bn:3d}: {tf:6.1f} TFLOPS")
