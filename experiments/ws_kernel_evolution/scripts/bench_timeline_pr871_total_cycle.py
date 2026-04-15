#!/usr/bin/env python3
"""Measure PR 871 steady-state total iteration cycles.

Supports:

- `base`: original PR 871 persistent ordering
- `reorder`: KV-head-friendly persistent ordering

Measured interval for WG1 steady-state (`n_idx > 0`):

  t_start: before `barrier_wait(k_full, ...)`
  t_end:   after `T.copy(acc_s_1, acc_s_cast_1)`

This gives a directly measured local steady-state iteration total for the
consumer path, without stitching front/steady/tail probes together.
"""

import argparse
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

from tileops.kernels.online_softmax import (
    make_log2e_scale,
    make_online_softmax_with_mask_guard,
    make_rescale,
)


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
    variant: str = "base",
    dtype: str = "float16",
):
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError(f"expected dim=128, got {dim}")
    if variant not in {"base", "reorder"}:
        raise ValueError(f"unknown variant {variant}")

    block_m = 128
    block_n = 128
    if seq_len % block_m != 0 or seq_len % block_n != 0:
        raise ValueError("seq_len must be divisible by block_m and block_n")

    m_blocks = seq_len // block_m
    if m_blocks % 2 != 0:
        raise ValueError("persistent causal pairing requires even M_blocks")

    half_m = block_m // 2
    half_m_blocks = m_blocks // 2
    groups = heads // heads_kv
    total_pairs = batch * heads * half_m_blocks
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    effective_num_sms = min(num_sms, total_pairs)
    scale = make_log2e_scale(dim)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def func(block_m: int, block_n: int):
        softmax_1 = make_online_softmax_with_mask_guard(scale, accum_dtype, half_m, block_n)
        softmax_2 = make_online_softmax_with_mask_guard(scale, accum_dtype, half_m, block_n)
        rescale_1 = make_rescale(half_m, dim)
        rescale_2 = make_rescale(half_m, dim)

        @T.prim_func
        def main(
            q: T.Tensor((batch, seq_len, heads, dim), dtype),
            k: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
            v: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
            output: T.Tensor((batch, seq_len, heads, dim), dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
            timing: T.Tensor([2], "int64"),
        ) -> None:
            with T.Kernel(effective_num_sms, 1, 1, threads=384, prelude=CLOCK_PRELUDE) as (bx, _by, _bz):
                q_shared_1 = T.alloc_shared([half_m, dim], dtype)
                q_shared_2 = T.alloc_shared([half_m, dim], dtype)
                k_smem_0 = T.alloc_shared([block_n, dim], dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], dtype)
                v_smem_0 = T.alloc_shared([block_n, dim], dtype)
                v_smem_1 = T.alloc_shared([block_n, dim], dtype)

                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_1 = T.alloc_fragment([half_m, block_n], dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_2 = T.alloc_fragment([half_m, block_n], dtype)
                acc_o_2 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_2 = T.alloc_fragment([half_m], accum_dtype)
                smp_2 = T.alloc_fragment([half_m], accum_dtype)
                ss_2 = T.alloc_fragment([half_m], accum_dtype)
                ssum_2 = T.alloc_fragment([half_m], accum_dtype)
                ls_2 = T.alloc_fragment([half_m], accum_dtype)

                k_full = T.alloc_barrier(arrive_count=128)
                k_empty = T.alloc_barrier(arrive_count=256)
                v_full = T.alloc_barrier(arrive_count=128)
                v_empty = T.alloc_barrier(arrive_count=256)
                wg_sched_12 = T.alloc_barrier(arrive_count=128)
                wg_sched_21 = T.alloc_barrier(arrive_count=128)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)

                T.annotate_layout({
                    q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                    q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                })
                T.sync_threads()

                gi_kp = T.alloc_var("int32", init=0)
                gi_vp = T.alloc_var("int32", init=0)
                gi_kc1 = T.alloc_var("int32", init=0)
                gi_vc1 = T.alloc_var("int32", init=0)
                gi_kc2 = T.alloc_var("int32", init=0)
                gi_vc2 = T.alloc_var("int32", init=0)
                gi_q1 = T.alloc_var("int32", init=0)
                gi_q2 = T.alloc_var("int32", init=0)

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    if variant == "base":
                        for tile_b, tile_h, pair_idx in T.Persistent(
                            [batch, heads, half_m_blocks],
                            wave_size=effective_num_sms,
                            index=bx,
                            group_size=8,
                        ):
                            head_kv = tile_h // groups
                            for sub_idx in range(2):
                                tile_m = pair_idx + sub_idx * (m_blocks - 1 - 2 * pair_idx)
                                loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)
                                for n_idx in T.Pipelined(loop_range, num_stages=0):
                                    T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                                    if gi_kp % 2 == 0:
                                        T.tma_copy(k[tile_b, n_idx * block_n : (n_idx + 1) * block_n, head_kv, :], k_smem_0, barrier=k_full)
                                    else:
                                        T.tma_copy(k[tile_b, n_idx * block_n : (n_idx + 1) * block_n, head_kv, :], k_smem_1, barrier=k_full)
                                    T.barrier_arrive(k_full)
                                    if n_idx > 0:
                                        T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                                        if gi_vp % 2 == 0:
                                            T.tma_copy(v[tile_b, (n_idx - 1) * block_n : n_idx * block_n, head_kv, :], v_smem_0, barrier=v_full)
                                        else:
                                            T.tma_copy(v[tile_b, (n_idx - 1) * block_n : n_idx * block_n, head_kv, :], v_smem_1, barrier=v_full)
                                        T.barrier_arrive(v_full)
                                        gi_vp = gi_vp + 1
                                    gi_kp = gi_kp + 1
                                T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                                if gi_vp % 2 == 0:
                                    T.tma_copy(v[tile_b, (loop_range - 1) * block_n : loop_range * block_n, head_kv, :], v_smem_0, barrier=v_full)
                                else:
                                    T.tma_copy(v[tile_b, (loop_range - 1) * block_n : loop_range * block_n, head_kv, :], v_smem_1, barrier=v_full)
                                T.barrier_arrive(v_full)
                                gi_vp = gi_vp + 1
                    else:
                        for tile_b, head_kv, pair_idx, group_idx in T.Persistent(
                            [batch, heads_kv, half_m_blocks, groups],
                            wave_size=effective_num_sms,
                            index=bx,
                            group_size=8,
                        ):
                            for sub_idx in range(2):
                                tile_m = pair_idx + sub_idx * (m_blocks - 1 - 2 * pair_idx)
                                loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)
                                for n_idx in T.Pipelined(loop_range, num_stages=0):
                                    T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                                    if gi_kp % 2 == 0:
                                        T.tma_copy(k[tile_b, n_idx * block_n : (n_idx + 1) * block_n, head_kv, :], k_smem_0, barrier=k_full)
                                    else:
                                        T.tma_copy(k[tile_b, n_idx * block_n : (n_idx + 1) * block_n, head_kv, :], k_smem_1, barrier=k_full)
                                    T.barrier_arrive(k_full)
                                    if n_idx > 0:
                                        T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                                        if gi_vp % 2 == 0:
                                            T.tma_copy(v[tile_b, (n_idx - 1) * block_n : n_idx * block_n, head_kv, :], v_smem_0, barrier=v_full)
                                        else:
                                            T.tma_copy(v[tile_b, (n_idx - 1) * block_n : n_idx * block_n, head_kv, :], v_smem_1, barrier=v_full)
                                        T.barrier_arrive(v_full)
                                        gi_vp = gi_vp + 1
                                    gi_kp = gi_kp + 1
                                T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                                if gi_vp % 2 == 0:
                                    T.tma_copy(v[tile_b, (loop_range - 1) * block_n : loop_range * block_n, head_kv, :], v_smem_0, barrier=v_full)
                                else:
                                    T.tma_copy(v[tile_b, (loop_range - 1) * block_n : loop_range * block_n, head_kv, :], v_smem_1, barrier=v_full)
                                T.barrier_arrive(v_full)
                                gi_vp = gi_vp + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    T.barrier_arrive(wg_sched_21)
                    if variant == "base":
                        for tile_b, tile_h, pair_idx in T.Persistent(
                            [batch, heads, half_m_blocks],
                            wave_size=effective_num_sms,
                            index=bx,
                            group_size=8,
                        ):
                            for sub_idx in range(2):
                                tile_m = pair_idx + sub_idx * (m_blocks - 1 - 2 * pair_idx)
                                row_base = tile_m * block_m
                                loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)
                                T.tma_copy(q[tile_b, row_base : row_base + half_m, tile_h, :], q_shared_1, barrier=q_full_1)
                                T.barrier_arrive(q_full_1)
                                T.barrier_wait(q_full_1, gi_q1 % 2)
                                gi_q1 = gi_q1 + 1
                                T.clear(acc_o_1)
                                T.clear(ls_1)
                                T.fill(sm_1, -T.infinity(accum_dtype))
                                for n_idx in T.Pipelined(loop_range, num_stages=0):
                                    if n_idx == 0:
                                        T.barrier_wait(k_full, gi_kc1 % 2)
                                        T.barrier_wait(wg_sched_21, gi_kc1 % 2)
                                        T.clear(acc_s_1)
                                        if gi_kc1 % 2 == 0:
                                            T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        T.barrier_arrive(wg_sched_12)
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                                        T.barrier_arrive(k_empty)
                                        if n_idx == loop_range - 1:
                                            for i, j in T.Parallel(half_m, block_n):
                                                acc_s_1[i, j] = T.if_then_else(row_base + i >= n_idx * block_n + j, acc_s_1[i, j], -T.infinity(accum_dtype))
                                        softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                        T.copy(acc_s_1, acc_s_cast_1)
                                    else:
                                        t_start = T.call_extern("int64", "clk::read_clock")
                                        T.barrier_wait(k_full, gi_kc1 % 2)
                                        T.barrier_wait(wg_sched_21, gi_kc1 % 2)
                                        T.clear(acc_s_1)
                                        if gi_kc1 % 2 == 0:
                                            T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        rescale_1(acc_o_1, ss_1)
                                        T.barrier_wait(v_full, gi_vc1 % 2)
                                        if gi_vc1 % 2 == 0:
                                            T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                        T.barrier_arrive(wg_sched_12)
                                        T.wait_wgmma(1)
                                        T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                                        T.barrier_arrive(k_empty)
                                        if n_idx == loop_range - 1:
                                            for i, j in T.Parallel(half_m, block_n):
                                                acc_s_1[i, j] = T.if_then_else(row_base + i >= n_idx * block_n + j, acc_s_1[i, j], -T.infinity(accum_dtype))
                                        softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                        T.barrier_arrive(v_empty)
                                        T.copy(acc_s_1, acc_s_cast_1)
                                        t_end = T.call_extern("int64", "clk::read_clock")
                                        gi_vc1 = gi_vc1 + 1
                                        T.call_extern("handle", "clk::clock_accum", timing.access_ptr("w", offset=0), t_end - t_start)
                                        T.call_extern("handle", "clk::clock_accum_count", timing.access_ptr("w", offset=1))
                                    gi_kc1 = gi_kc1 + 1
                                rescale_1(acc_o_1, ss_1)
                                T.barrier_wait(v_full, gi_vc1 % 2)
                                if gi_vc1 % 2 == 0:
                                    T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                else:
                                    T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                T.barrier_arrive(v_empty)
                                gi_vc1 = gi_vc1 + 1
                                for i, j in T.Parallel(half_m, dim):
                                    acc_o_1[i, j] /= ls_1[i]
                                T.copy(acc_o_1, q_shared_1)
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=3, arrive_count=128)
                                T.copy(q_shared_1, output[tile_b, row_base : row_base + half_m, tile_h, :])
                                for i in T.Parallel(half_m):
                                    ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                                T.copy(ls_1, lse[tile_b, tile_h, row_base : row_base + half_m])
                    else:
                        for tile_b, head_kv, pair_idx, group_idx in T.Persistent(
                            [batch, heads_kv, half_m_blocks, groups],
                            wave_size=effective_num_sms,
                            index=bx,
                            group_size=8,
                        ):
                            tile_h = head_kv * groups + group_idx
                            for sub_idx in range(2):
                                tile_m = pair_idx + sub_idx * (m_blocks - 1 - 2 * pair_idx)
                                row_base = tile_m * block_m
                                loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)
                                T.tma_copy(q[tile_b, row_base : row_base + half_m, tile_h, :], q_shared_1, barrier=q_full_1)
                                T.barrier_arrive(q_full_1)
                                T.barrier_wait(q_full_1, gi_q1 % 2)
                                gi_q1 = gi_q1 + 1
                                T.clear(acc_o_1)
                                T.clear(ls_1)
                                T.fill(sm_1, -T.infinity(accum_dtype))
                                for n_idx in T.Pipelined(loop_range, num_stages=0):
                                    if n_idx == 0:
                                        T.barrier_wait(k_full, gi_kc1 % 2)
                                        T.barrier_wait(wg_sched_21, gi_kc1 % 2)
                                        T.clear(acc_s_1)
                                        if gi_kc1 % 2 == 0:
                                            T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        T.barrier_arrive(wg_sched_12)
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                                        T.barrier_arrive(k_empty)
                                        if n_idx == loop_range - 1:
                                            for i, j in T.Parallel(half_m, block_n):
                                                acc_s_1[i, j] = T.if_then_else(row_base + i >= n_idx * block_n + j, acc_s_1[i, j], -T.infinity(accum_dtype))
                                        softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                        T.copy(acc_s_1, acc_s_cast_1)
                                    else:
                                        t_start = T.call_extern("int64", "clk::read_clock")
                                        T.barrier_wait(k_full, gi_kc1 % 2)
                                        T.barrier_wait(wg_sched_21, gi_kc1 % 2)
                                        T.clear(acc_s_1)
                                        if gi_kc1 % 2 == 0:
                                            T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        rescale_1(acc_o_1, ss_1)
                                        T.barrier_wait(v_full, gi_vc1 % 2)
                                        if gi_vc1 % 2 == 0:
                                            T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                        T.barrier_arrive(wg_sched_12)
                                        T.wait_wgmma(1)
                                        T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                                        T.barrier_arrive(k_empty)
                                        if n_idx == loop_range - 1:
                                            for i, j in T.Parallel(half_m, block_n):
                                                acc_s_1[i, j] = T.if_then_else(row_base + i >= n_idx * block_n + j, acc_s_1[i, j], -T.infinity(accum_dtype))
                                        softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                        T.barrier_arrive(v_empty)
                                        T.copy(acc_s_1, acc_s_cast_1)
                                        t_end = T.call_extern("int64", "clk::read_clock")
                                        gi_vc1 = gi_vc1 + 1
                                        T.call_extern("handle", "clk::clock_accum", timing.access_ptr("w", offset=0), t_end - t_start)
                                        T.call_extern("handle", "clk::clock_accum_count", timing.access_ptr("w", offset=1))
                                    gi_kc1 = gi_kc1 + 1
                                rescale_1(acc_o_1, ss_1)
                                T.barrier_wait(v_full, gi_vc1 % 2)
                                if gi_vc1 % 2 == 0:
                                    T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                else:
                                    T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                T.barrier_arrive(v_empty)
                                gi_vc1 = gi_vc1 + 1
                                for i, j in T.Parallel(half_m, dim):
                                    acc_o_1[i, j] /= ls_1[i]
                                T.copy(acc_o_1, q_shared_1)
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=3, arrive_count=128)
                                T.copy(q_shared_1, output[tile_b, row_base : row_base + half_m, tile_h, :])
                                for i in T.Parallel(half_m):
                                    ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                                T.copy(ls_1, lse[tile_b, tile_h, row_base : row_base + half_m])

                else:
                    T.inc_max_nreg(240)
                    if variant == "base":
                        for tile_b, tile_h, pair_idx in T.Persistent(
                            [batch, heads, half_m_blocks],
                            wave_size=effective_num_sms,
                            index=bx,
                            group_size=8,
                        ):
                            for sub_idx in range(2):
                                tile_m = pair_idx + sub_idx * (m_blocks - 1 - 2 * pair_idx)
                                row_base = tile_m * block_m
                                loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)
                                T.tma_copy(q[tile_b, row_base + half_m : row_base + block_m, tile_h, :], q_shared_2, barrier=q_full_2)
                                T.barrier_arrive(q_full_2)
                                T.barrier_wait(q_full_2, gi_q2 % 2)
                                gi_q2 = gi_q2 + 1
                                T.clear(acc_o_2)
                                T.clear(ls_2)
                                T.fill(sm_2, -T.infinity(accum_dtype))
                                for n_idx in T.Pipelined(loop_range, num_stages=0):
                                    T.barrier_wait(k_full, gi_kc2 % 2)
                                    T.barrier_wait(wg_sched_12, gi_kc2 % 2)
                                    T.clear(acc_s_2)
                                    if n_idx == 0:
                                        if gi_kc2 % 2 == 0:
                                            T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        T.barrier_arrive(wg_sched_21)
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                                        T.barrier_arrive(k_empty)
                                        if n_idx == loop_range - 1:
                                            for i, j in T.Parallel(half_m, block_n):
                                                acc_s_2[i, j] = T.if_then_else(row_base + half_m + i >= n_idx * block_n + j, acc_s_2[i, j], -T.infinity(accum_dtype))
                                        softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                        T.copy(acc_s_2, acc_s_cast_2)
                                    else:
                                        if gi_kc2 % 2 == 0:
                                            T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        rescale_2(acc_o_2, ss_2)
                                        T.barrier_wait(v_full, gi_vc2 % 2)
                                        if gi_vc2 % 2 == 0:
                                            T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                        T.barrier_arrive(wg_sched_21)
                                        T.wait_wgmma(1)
                                        T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                                        T.barrier_arrive(k_empty)
                                        if n_idx == loop_range - 1:
                                            for i, j in T.Parallel(half_m, block_n):
                                                acc_s_2[i, j] = T.if_then_else(row_base + half_m + i >= n_idx * block_n + j, acc_s_2[i, j], -T.infinity(accum_dtype))
                                        softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                        T.barrier_arrive(v_empty)
                                        T.copy(acc_s_2, acc_s_cast_2)
                                        gi_vc2 = gi_vc2 + 1
                                    gi_kc2 = gi_kc2 + 1
                                rescale_2(acc_o_2, ss_2)
                                T.barrier_wait(v_full, gi_vc2 % 2)
                                if gi_vc2 % 2 == 0:
                                    T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                else:
                                    T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                T.barrier_arrive(v_empty)
                                gi_vc2 = gi_vc2 + 1
                                for i, j in T.Parallel(half_m, dim):
                                    acc_o_2[i, j] /= ls_2[i]
                                T.copy(acc_o_2, q_shared_2)
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=4, arrive_count=128)
                                T.copy(q_shared_2, output[tile_b, row_base + half_m : row_base + block_m, tile_h, :])
                                for i in T.Parallel(half_m):
                                    ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                                T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m : row_base + block_m])
                    else:
                        for tile_b, head_kv, pair_idx, group_idx in T.Persistent(
                            [batch, heads_kv, half_m_blocks, groups],
                            wave_size=effective_num_sms,
                            index=bx,
                            group_size=8,
                        ):
                            tile_h = head_kv * groups + group_idx
                            for sub_idx in range(2):
                                tile_m = pair_idx + sub_idx * (m_blocks - 1 - 2 * pair_idx)
                                row_base = tile_m * block_m
                                loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)
                                T.tma_copy(q[tile_b, row_base + half_m : row_base + block_m, tile_h, :], q_shared_2, barrier=q_full_2)
                                T.barrier_arrive(q_full_2)
                                T.barrier_wait(q_full_2, gi_q2 % 2)
                                gi_q2 = gi_q2 + 1
                                T.clear(acc_o_2)
                                T.clear(ls_2)
                                T.fill(sm_2, -T.infinity(accum_dtype))
                                for n_idx in T.Pipelined(loop_range, num_stages=0):
                                    T.barrier_wait(k_full, gi_kc2 % 2)
                                    T.barrier_wait(wg_sched_12, gi_kc2 % 2)
                                    T.clear(acc_s_2)
                                    if n_idx == 0:
                                        if gi_kc2 % 2 == 0:
                                            T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        T.barrier_arrive(wg_sched_21)
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                                        T.barrier_arrive(k_empty)
                                        if n_idx == loop_range - 1:
                                            for i, j in T.Parallel(half_m, block_n):
                                                acc_s_2[i, j] = T.if_then_else(row_base + half_m + i >= n_idx * block_n + j, acc_s_2[i, j], -T.infinity(accum_dtype))
                                        softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                        T.copy(acc_s_2, acc_s_cast_2)
                                    else:
                                        if gi_kc2 % 2 == 0:
                                            T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                        rescale_2(acc_o_2, ss_2)
                                        T.barrier_wait(v_full, gi_vc2 % 2)
                                        if gi_vc2 % 2 == 0:
                                            T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                        else:
                                            T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                        T.barrier_arrive(wg_sched_21)
                                        T.wait_wgmma(1)
                                        T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                                        T.barrier_arrive(k_empty)
                                        if n_idx == loop_range - 1:
                                            for i, j in T.Parallel(half_m, block_n):
                                                acc_s_2[i, j] = T.if_then_else(row_base + half_m + i >= n_idx * block_n + j, acc_s_2[i, j], -T.infinity(accum_dtype))
                                        softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                        T.barrier_arrive(v_empty)
                                        T.copy(acc_s_2, acc_s_cast_2)
                                        gi_vc2 = gi_vc2 + 1
                                    gi_kc2 = gi_kc2 + 1
                                rescale_2(acc_o_2, ss_2)
                                T.barrier_wait(v_full, gi_vc2 % 2)
                                if gi_vc2 % 2 == 0:
                                    T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                else:
                                    T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                T.barrier_arrive(v_empty)
                                gi_vc2 = gi_vc2 + 1
                                for i, j in T.Parallel(half_m, dim):
                                    acc_o_2[i, j] /= ls_2[i]
                                T.copy(acc_o_2, q_shared_2)
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=4, arrive_count=128)
                                T.copy(q_shared_2, output[tile_b, row_base + half_m : row_base + block_m, tile_h, :])
                                for i in T.Parallel(half_m):
                                    ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                                T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m : row_base + block_m])

        return main

    return func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "reorder"), default="base")
    args = parser.parse_args()

    batch, seq_len, heads, heads_kv, dim = 4, 4096, 64, 8, 128

    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    timing = torch.zeros(2, device="cuda", dtype=torch.int64)

    print(f"Building kernel ({args.variant} total cycle)...")
    kernel_fn = build_clock_kernel(batch, seq_len, heads, heads_kv, dim, variant=args.variant)
    kernel = kernel_fn(block_m=128, block_n=128)

    for _ in range(3):
        timing.zero_()
        kernel(q, k, v, timing)

    timing.zero_()
    torch.cuda.synchronize()
    kernel(q, k, v, timing)
    torch.cuda.synchronize()

    t = timing.cpu().numpy().astype(float)
    count = t[1]
    if count > 0:
        print(f"steady_iter_total:  {t[0] / count:.0f} cycles")
        print(f"\nSamples: {int(count)} (WG1 steady-state iters across all persistent tiles)")
    else:
        print("No timing data collected!")
