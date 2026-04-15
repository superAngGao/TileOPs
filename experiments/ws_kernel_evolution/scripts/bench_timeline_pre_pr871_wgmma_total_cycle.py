#!/usr/bin/env python3
"""Measure pre-PR871 WGMMA-pipelined coarse tile timing.

This probe intentionally uses coarse timing boundaries only. Injecting
fine-grained clock reads inside the `T.Pipelined + T.gemm` loop currently hits
an internal TileLang/TVM `WgmmaSyncRewriter` crash on this environment.

So for pre-PR871 we first capture:

- `loop_body_total`: the software-pipelined causal loop over K/V tiles
- `epilogue_total`: final normalize + store path

This is enough to support milestone comparison and to define how this kernel
should be drawn: as a single-CTA software pipeline, not as a WS handoff graph.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")

ROOT = Path("/home/ga/TileOPs")
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.online_softmax import make_log2e_scale, make_online_softmax, make_rescale


CLOCK_PRELUDE = r"""
namespace clk {
__device__ __forceinline__ int64_t read_clock() {
    int64_t ret; asm volatile("mov.u64 %0, %%clock64;" : "=l"(ret)); return ret;
}
__device__ __forceinline__ void clock_accum(int64_t* addr, int64_t delta) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr), static_cast<unsigned long long>(delta));
}
__device__ __forceinline__ void clock_accum_count(int64_t* addr) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr), 1ULL);
}
}  // namespace clk
"""


def build_clock_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    dtype: str = "float16",
):
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError(f"expected dim=128, got {dim}")

    block_m = 128
    block_n = 128
    threads = 256
    num_stages = 2
    if seq_len % block_m != 0 or seq_len % block_n != 0:
        raise ValueError("seq_len must be divisible by block_m and block_n")

    scale = make_log2e_scale(dim)
    groups = heads // heads_kv
    accum_dtype = "float"
    m_blocks = seq_len // block_m

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def func(block_m: int, block_n: int, num_stages: int, threads: int):
        online_softmax = make_online_softmax(scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def main(
            q: T.Tensor((batch, seq_len, heads, dim), dtype),
            k: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
            v: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
            output: T.Tensor((batch, seq_len, heads, dim), dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
            timing: T.Tensor([3], "int64"),
        ) -> None:
            with T.Kernel(1, 1, 1, threads=threads, prelude=CLOCK_PRELUDE) as (_bx, _by, _bz):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                o_shared = T.alloc_shared([block_m, dim], dtype)

                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.annotate_layout({o_shared: tilelang.layout.make_swizzled_layout(o_shared)})

                tile_b = 0
                tile_h = 0
                tile_hkv = tile_h // groups
                tile_m = m_blocks - 1
                row_base = tile_m * block_m
                loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)

                T.copy(q[tile_b, row_base: row_base + block_m, tile_h, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                t0 = T.call_extern("int64", "clk::read_clock")
                for k_idx in T.Pipelined(
                    loop_range,
                    num_stages=num_stages,
                    order=[-1, 0, 3, 1, -1, 2],
                    stage=[-1, 0, 0, 1, -1, 1],
                    group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]],
                ):
                    T.copy(k[tile_b, k_idx * block_n:(k_idx + 1) * block_n, tile_hkv, :], k_shared)
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.if_then_else(
                            row_base + i >= k_idx * block_n + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                    T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    online_softmax(
                        acc_s,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                    )
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.copy(v[tile_b, k_idx * block_n:(k_idx + 1) * block_n, tile_hkv, :], v_shared)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                t1 = T.call_extern("int64", "clk::read_clock")

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, output[tile_b, row_base: row_base + block_m, tile_h, :])
                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[tile_b, tile_h, row_base: row_base + block_m])
                t2 = T.call_extern("int64", "clk::read_clock")

                T.call_extern("handle", "clk::clock_accum", timing.access_ptr("w", offset=0), t1 - t0)
                T.call_extern("handle", "clk::clock_accum", timing.access_ptr("w", offset=1), t2 - t1)
                T.call_extern("handle", "clk::clock_accum_count", timing.access_ptr("w", offset=2))

        return main

    return func


if __name__ == "__main__":
    batch, seq_len, heads, heads_kv, dim = 4, 4096, 64, 8, 128

    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    timing = torch.zeros(3, device="cuda", dtype=torch.int64)

    print("Building kernel (pre_pr871_wgmma total cycle)...")
    kernel_fn = build_clock_kernel(batch, seq_len, heads, heads_kv, dim)
    kernel = kernel_fn(block_m=128, block_n=128, num_stages=2, threads=256)

    for _ in range(3):
        timing.zero_()
        kernel(q, k, v, timing)

    timing.zero_()
    torch.cuda.synchronize()
    kernel(q, k, v, timing)
    torch.cuda.synchronize()

    loop_total, epilogue_total, count = timing.cpu().numpy().astype(float)
    if count <= 0:
        raise RuntimeError("no samples captured")
    print("Pre-PR871 WGMMA coarse timing")
    print(f"  loop_body_total: {loop_total / count:.1f} cycles")
    print(f"  epilogue_total: {epilogue_total / count:.1f} cycles")
    print(f"Samples: {int(count)}")
