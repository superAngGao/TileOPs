"""Validate block_n=64 + all post-processes on all 5 v2 shapes + 3x bench."""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
os.environ["ALIAS"] = "1"
os.environ["DESC_REWRITE"] = "1"

import torch
import _bench_ws_shfl as bw
bw.enable_shfl_hook()

from _test_ws_fa3_v2 import build_fa3_v2, ref_gqa


def fa3_flops(B, S, H, D, is_causal):
    f = 4.0 * B * H * S * S * D
    return f * 0.5 if is_causal else f


def bench_one(B, S, H, Hkv, D, is_causal, block_m=128, block_n=64,
              n_warmup=10, n_iter=50):
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
    cfg = f"B={B} S={S} H={H} Hkv={Hkv} D={D} causal={is_causal}"
    ok = "PASS" if diff < 0.5 else "FAIL"
    print(f"  {cfg}: {ms:6.3f} ms  {tflops:7.1f} TFLOPS  diff={diff:.4f}  {ok}")
    return diff < 0.5, tflops


if __name__ == "__main__":
    print("=" * 70)
    print("v2 + all hooks + block_n=64 — 5-shape correctness + 3x stability")
    print("=" * 70)

    print("\n--- Stability (main bench shape, 3 runs) ---")
    tflops_runs = []
    for i in range(3):
        _, tf = bench_one(4, 2048, 64, 4, 128, False)
        tflops_runs.append(tf)
    import statistics
    mean = statistics.mean(tflops_runs)
    stdev = statistics.stdev(tflops_runs) if len(tflops_runs) > 1 else 0.0
    print(f"  Stable mean: {mean:.1f} ± {stdev:.1f} TFLOPS  ({mean/647.3*100:.1f}% of FA3)")

    print("\n--- All 5 v2 correctness shapes ---")
    ok = True
    ok &= bench_one(1, 256, 8, 4, 128, False)[0]
    ok &= bench_one(4, 512, 64, 4, 128, False)[0]
    ok &= bench_one(4, 512, 64, 4, 128, True)[0]
    ok &= bench_one(1, 1024, 32, 8, 128, True)[0]
    ok &= bench_one(2, 2048, 32, 8, 128, True)[0]
    print(f"\n{'All passed' if ok else 'SOME FAILED'}")
