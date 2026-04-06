"""Run all 5 v2 correctness shapes with alias rewrite enabled (no shfl).
Also benchmarks alias-alone to ablate from alias+shfl.
"""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
os.environ["ALIAS"] = "1"

import torch
import tvm_ffi
import _bench_ws_shfl as bw

# Build an alias-only callback that skips shfl WG-id rewriting.
from tilelang.contrib import nvcc
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.transform import PassConfigKey


@tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
def alias_only_cb(code, target, pass_config=None):
    patched, stats = bw.alias_rewrite(code)
    print(f"[alias-only] {stats}", flush=True)
    target_arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))
    opts = ["-std=c++17", "-I" + TILELANG_TEMPLATE_PATH, "-I" + CUTLASS_INCLUDE_DIR]
    cfg = pass_config or {}
    if cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False):
        opts.append("--use_fast_math")
    extra = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
    if extra:
        import shlex
        for f in extra:
            opts += shlex.split(f) if isinstance(f, str) else [str(f)]
    return nvcc.compile_cuda(patched, "cubin", [f"-arch=sm_{target_arch}"], options=opts)


from _test_ws_fa3_v2 import build_fa3_v2, ref_gqa


def fa3_flops(B, S, H, D, is_causal):
    f = 4.0 * B * H * S * S * D
    return f * 0.5 if is_causal else f


def bench_one(B, S, H, Hkv, D, is_causal):
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    kernel = build_fa3_v2(B, S, H, Hkv, D, is_causal)(block_m=128, block_n=128)
    o, lse = kernel(q, k, v)
    o_ref = ref_gqa(q, k, v, is_causal)
    diff = (o.float() - o_ref.float()).abs().max().item()
    for _ in range(10):
        kernel(q, k, v)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        kernel(q, k, v)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 50
    tflops = fa3_flops(B, S, H, D, is_causal) / (ms * 1e-3) / 1e12
    cfg = f"B={B} S={S} H={H} Hkv={Hkv} D={D} causal={is_causal}"
    ok = "PASS" if diff < 0.5 else "FAIL"
    print(f"  {cfg}: {ms:.3f} ms  {tflops:7.1f} TFLOPS  diff={diff:.4f}  {ok}")
    return diff < 0.5, tflops


if __name__ == "__main__":
    print("=" * 70)
    print("Alias-only ablation + 5-shape correctness")
    print("=" * 70)

    print("\n--- Bench (alias only, no shfl) ---")
    bench_one(4, 2048, 64, 4, 128, False)

    print("\n--- Correctness, all 5 v2 shapes (alias only) ---")
    ok = True
    ok &= bench_one(1, 256, 8, 4, 128, False)[0]
    ok &= bench_one(4, 512, 64, 4, 128, False)[0]
    ok &= bench_one(4, 512, 64, 4, 128, True)[0]
    ok &= bench_one(1, 1024, 32, 8, 128, True)[0]
    ok &= bench_one(2, 2048, 32, 8, 128, True)[0]
    print(f"\n{'All passed' if ok else 'SOME FAILED'}")
