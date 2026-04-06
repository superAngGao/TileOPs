"""Compare block_n=128 vs block_n=64 with all post-processes enabled.
Goal: see if shrinking block_n frees register slots for wgmma pipelining
and helps the C7512 bottleneck.
"""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
os.environ["ALIAS"] = "1"
os.environ["DESC_REWRITE"] = "1"

import torch
import _bench_ws_shfl as bw  # registers patched callback
bw.enable_shfl_hook()

from _test_ws_fa3_v2 import build_fa3_v2, ref_gqa


def fa3_flops(B, S, H, D, is_causal):
    f = 4.0 * B * H * S * S * D
    return f * 0.5 if is_causal else f


def bench(B, S, H, Hkv, D, is_causal, block_m, block_n, n_warmup=10, n_iter=50):
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

    kernel = build_fa3_v2(B, S, H, Hkv, D, is_causal,
                          block_m=block_m, block_n=block_n)(
        block_m=block_m, block_n=block_n)

    o, lse = kernel(q, k, v)
    o_ref = ref_gqa(q, k, v, is_causal)
    diff = (o.float() - o_ref.float()).abs().max().item()
    for _ in range(n_warmup):
        kernel(q, k, v)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        kernel(q, k, v)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / n_iter
    tflops = fa3_flops(B, S, H, D, is_causal) / (ms * 1e-3) / 1e12

    print(f"  block_m={block_m} block_n={block_n}: {ms:.3f} ms  {tflops:7.1f} TFLOPS  diff={diff:.4f}")
    return tflops


if __name__ == "__main__":
    print("=" * 60)
    print("v2 block_n ablation (all hooks: shfl + desc + alias)")
    print("=" * 60)

    shape = (4, 2048, 64, 4, 128, False)
    print(f"Shape: B={shape[0]} S={shape[1]} H={shape[2]} Hkv={shape[3]} D={shape[4]} causal={shape[5]}")
    print()

    # Only block_n values that divide S=2048 (otherwise residual is wrong)
    results = {}
    for bn in [32, 64, 128, 256]:
        try:
            print(f"--- block_n={bn} ---")
            results[bn] = bench(*shape, block_m=128, block_n=bn)
            print()
        except Exception as ex:
            print(f"  FAILED: {type(ex).__name__}: {str(ex)[:200]}")
            print()

    print("Summary (only valid block_n that divide S=2048):")
    for bn, tf in results.items():
        print(f"  block_n={bn:3}: {tf:7.1f} TFLOPS  ({tf/647.3*100:.1f}% of FA3)")
