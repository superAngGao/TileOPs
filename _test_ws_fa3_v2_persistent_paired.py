"""FA3-aligned IntraWGOverlap v2 — PERSISTENT + CAUSAL TILE PAIRING.

Built on top of `_test_ws_fa3_v2_persistent.py`. Adds causal tile
pairing to flatten the long-tail load imbalance.

## The pairing trick

Causal mode: tile_m=k has loop_range = k+1, so tile_m=0 does 1 K-iter
and tile_m=M-1 does M K-iters. Long-tail CTAs drag the avg TFLOPS to
~50% FA3.

Fix: pair tile_m=k with tile_m=M-1-k INTO THE SAME CTA's persistent
stream. Each pair has total work (k+1) + (M-k) = M+1 iters, CONSTANT.
Every "unit of work" in the persistent loop is now the same size. No
long-tail.

  M = 16:
    Pair 0: tile_m=0  (1 iter)  + tile_m=15 (16 iters) = 17 iters
    Pair 1: tile_m=1  (2 iters) + tile_m=14 (15 iters) = 17 iters
    ...
    Pair 7: tile_m=7  (8 iters) + tile_m=8  (9 iters)  = 17 iters

This brings causal from ~50% FA3 (long-tail-bound) toward the
non-causal % FA3 (which is the actual kernel ceiling).

## Implementation

The persistent loop iterates over `(B, H, M_blocks // 2)` pairs.
Inside each pair, a Python `for sub_idx in range(2)` unrolls 2 sub-tile
bodies sequentially in the TIR. Both sub-tiles share the same per-WG
global counters (gi_kp / gi_vp / gi_kc / gi_vc / gi_q), so Approach A
extends without modification — the counters accumulate continuously
across both sub-tiles.

Per pair, gi_kp / gi_kc each increase by exactly M_blocks + 1, the same
for any pair_idx.

Restrictions:
- Causal only (non-causal has no imbalance to fix).
- M_blocks must be even (asserted).

## Other design choices

Same as `_test_ws_fa3_v2_persistent.py`:
- Approach A global iteration counters
- Per-WG Q smem ownership (q_shared_1 = WG1, q_shared_2 = WG2)
- TMA-based Q load
- Bootstrap stays OUTSIDE the persistent loop
- Consumer epilogue v_empty release (1 extra arrive per consumer per
  sub-tile, matches the producer's wait count for that sub-tile)

Sub-tile body is the EXACT SAME structure as v2_persistent's causal
path. The only diff is the pairing wrapper outside it.
"""
import os
import time

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.online_softmax import (
    make_log2e_scale,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

NUM_SMS = int(os.environ.get("V2P_NUM_SMS", "132"))


def build_fa3_v2_persistent_paired(B, S, H, Hkv, D,
                                   block_m=128, block_n=128):
    assert H % Hkv == 0 and block_m % 2 == 0 and D == 128
    half_m = block_m // 2
    M_blocks = (S + block_m - 1) // block_m
    assert M_blocks % 2 == 0, (
        f"M_blocks={M_blocks} (S={S}, block_m={block_m}) "
        f"must be even for pairing")
    half_M_blocks = M_blocks // 2
    total_pairs = B * H * half_M_blocks
    # Clamp NUM_SMS to total_pairs to avoid idle CTAs in the single-wave
    # case. T.Persistent doesn't emit loop_break when waves == 1, so idle
    # CTAs would leak out-of-range pair_idx values, which the pairing
    # arithmetic turns into negative tile_m → negative loop_range →
    # CUDA_ERROR_ILLEGAL_INSTRUCTION inside wgmma. Clamping at build
    # time guarantees grid_size <= total_pairs and every CTA gets a
    # valid (tile_b, tile_h, pair_idx) triple.
    effective_num_sms = min(NUM_SMS, total_pairs)
    groups = H // Hkv
    scale = make_log2e_scale(D)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def func(block_m: int, block_n: int):
        q_shape = (B, S, H, D)
        kv_shape = (B, S, Hkv, D)

        softmax_1 = make_online_softmax_with_mask_guard(
            scale, accum_dtype, half_m, block_n)
        softmax_2 = make_online_softmax_with_mask_guard(
            scale, accum_dtype, half_m, block_n)
        rescale_1 = make_rescale(half_m, D)
        rescale_2 = make_rescale(half_m, D)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float16"),
            k: T.Tensor(kv_shape, "float16"),
            v: T.Tensor(kv_shape, "float16"),
            output: T.Tensor(q_shape, "float16"),
            lse: T.Tensor([B, H, S], accum_dtype),
        ) -> None:
            with T.Kernel(
                effective_num_sms, 1, 1, threads=384,
            ) as (bx, _by, _bz):
                # ---- Shared memory ----
                q_shared_1 = T.alloc_shared([half_m, D], "float16")
                q_shared_2 = T.alloc_shared([half_m, D], "float16")
                k_smem_0 = T.alloc_shared([block_n, D], "float16")
                k_smem_1 = T.alloc_shared([block_n, D], "float16")
                v_smem_0 = T.alloc_shared([block_n, D], "float16")
                v_smem_1 = T.alloc_shared([block_n, D], "float16")

                # ---- Fragments ----
                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_1 = T.alloc_fragment(
                    [half_m, block_n], "float16")
                acc_o_1 = T.alloc_fragment([half_m, D], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_2 = T.alloc_fragment(
                    [half_m, block_n], "float16")
                acc_o_2 = T.alloc_fragment([half_m, D], accum_dtype)
                sm_2 = T.alloc_fragment([half_m], accum_dtype)
                smp_2 = T.alloc_fragment([half_m], accum_dtype)
                ss_2 = T.alloc_fragment([half_m], accum_dtype)
                ssum_2 = T.alloc_fragment([half_m], accum_dtype)
                ls_2 = T.alloc_fragment([half_m], accum_dtype)

                # ---- Pipeline barriers ----
                k_full = T.alloc_barrier(arrive_count=128)
                k_empty = T.alloc_barrier(arrive_count=256)
                v_full = T.alloc_barrier(arrive_count=128)
                v_empty = T.alloc_barrier(arrive_count=256)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)

                T.annotate_layout({
                    q_shared_1:
                        tilelang.layout.make_swizzled_layout(q_shared_1),
                    q_shared_2:
                        tilelang.layout.make_swizzled_layout(q_shared_2),
                })

                T.sync_threads()  # after barrier init

                # ---- Per-WG global iter counters (Approach A) ----
                gi_kp = T.alloc_var("int32", init=0)
                gi_vp = T.alloc_var("int32", init=0)
                gi_kc1 = T.alloc_var("int32", init=0)
                gi_vc1 = T.alloc_var("int32", init=0)
                gi_kc2 = T.alloc_var("int32", init=0)
                gi_vc2 = T.alloc_var("int32", init=0)
                gi_q1 = T.alloc_var("int32", init=0)
                gi_q2 = T.alloc_var("int32", init=0)

                tx = T.get_thread_binding()

                # ===== WG0 (producer, tx < 128) =====
                if tx < 128:
                    T.dec_max_nreg(24)
                    for tile_b, tile_h, pair_idx in T.Persistent(
                        [B, H, half_M_blocks],
                        wave_size=effective_num_sms,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_h // groups
                        # Inner Python loop unrolls into 2 sub-tile bodies.
                        # sub_idx=0: short side (tile_m = pair_idx)
                        # sub_idx=1: long  side (tile_m = M-1-pair_idx)
                        for sub_idx in range(2):
                            # tile_m as a single TIR expression (no Python
                            # if-frame). sub_idx is Python int 0 or 1:
                            #   sub_idx=0 → tile_m = pair_idx
                            #   sub_idx=1 → tile_m = M_blocks - 1 - pair_idx
                            tile_m = (
                                pair_idx
                                + sub_idx * (M_blocks - 1 - 2 * pair_idx))
                            loop_range = T.ceildiv(
                                (tile_m + 1) * block_m, block_n)

                            for n_idx in T.Pipelined(
                                    loop_range, num_stages=0):
                                T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                                if gi_kp % 2 == 0:
                                    T.tma_copy(
                                        k[tile_b,
                                          n_idx * block_n:
                                          (n_idx + 1) * block_n,
                                          head_kv, :],
                                        k_smem_0, barrier=k_full)
                                else:
                                    T.tma_copy(
                                        k[tile_b,
                                          n_idx * block_n:
                                          (n_idx + 1) * block_n,
                                          head_kv, :],
                                        k_smem_1, barrier=k_full)
                                T.barrier_arrive(k_full)
                                if n_idx > 0:
                                    T.barrier_wait(
                                        v_empty, (gi_vp + 1) % 2)
                                    if gi_vp % 2 == 0:
                                        T.tma_copy(
                                            v[tile_b,
                                              (n_idx - 1) * block_n:
                                              n_idx * block_n,
                                              head_kv, :],
                                            v_smem_0, barrier=v_full)
                                    else:
                                        T.tma_copy(
                                            v[tile_b,
                                              (n_idx - 1) * block_n:
                                              n_idx * block_n,
                                              head_kv, :],
                                            v_smem_1, barrier=v_full)
                                    T.barrier_arrive(v_full)
                                    gi_vp = gi_vp + 1
                                gi_kp = gi_kp + 1
                            # Producer epilogue: tail load V[loop_range-1]
                            T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                            if gi_vp % 2 == 0:
                                T.tma_copy(
                                    v[tile_b,
                                      (loop_range - 1) * block_n:
                                      loop_range * block_n,
                                      head_kv, :],
                                    v_smem_0, barrier=v_full)
                            else:
                                T.tma_copy(
                                    v[tile_b,
                                      (loop_range - 1) * block_n:
                                      loop_range * block_n,
                                      head_kv, :],
                                    v_smem_1, barrier=v_full)
                            T.barrier_arrive(v_full)
                            gi_vp = gi_vp + 1

                # ===== WG1 (consumer 1, 128 <= tx < 256) =====
                elif tx < 256:
                    T.inc_max_nreg(240)
                    # Bootstrap: ONCE per CTA, OUTSIDE persistent loop
                    T.call_extern("handle",
                                  "tl::barrier_arrive_named", 1, 256)
                    for tile_b, tile_h, pair_idx in T.Persistent(
                        [B, H, half_M_blocks],
                        wave_size=effective_num_sms,
                        index=bx,
                        group_size=8,
                    ):
                        for sub_idx in range(2):
                            # tile_m as a single TIR expression (no Python
                            # if-frame). sub_idx is Python int 0 or 1:
                            #   sub_idx=0 → tile_m = pair_idx
                            #   sub_idx=1 → tile_m = M_blocks - 1 - pair_idx
                            tile_m = (
                                pair_idx
                                + sub_idx * (M_blocks - 1 - 2 * pair_idx))
                            row_base = tile_m * block_m
                            loop_range = T.ceildiv(
                                (tile_m + 1) * block_m, block_n)

                            # Per-sub-tile Q load (per-WG ownership)
                            T.tma_copy(
                                q[tile_b,
                                  row_base:row_base + half_m,
                                  tile_h, :],
                                q_shared_1, barrier=q_full_1)
                            T.barrier_arrive(q_full_1)
                            T.barrier_wait(q_full_1, gi_q1 % 2)
                            gi_q1 = gi_q1 + 1

                            # Per-sub-tile state reset
                            T.clear(acc_o_1)
                            T.clear(ls_1)
                            T.fill(sm_1, -T.infinity(accum_dtype))

                            for n_idx in T.Pipelined(
                                    loop_range, num_stages=0):
                                T.barrier_wait(k_full, gi_kc1 % 2)
                                T.sync_threads(
                                    barrier_id=1, arrive_count=256)
                                # Causal mask: only diagonal block needs it
                                if n_idx == loop_range - 1:
                                    for i, j in T.Parallel(
                                            half_m, block_n):
                                        acc_s_1[i, j] = T.if_then_else(
                                            row_base + i
                                            >= n_idx * block_n + j,
                                            0,
                                            -T.infinity(accum_dtype))
                                else:
                                    T.clear(acc_s_1)
                                if n_idx == 0:
                                    if gi_kc1 % 2 == 0:
                                        T.wgmma_gemm(
                                            q_shared_1, k_smem_0, acc_s_1,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            q_shared_1, k_smem_1, acc_s_1,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    T.call_extern(
                                        "handle",
                                        "tl::barrier_arrive_named",
                                        2, 256)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(
                                        acc_s_1, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    softmax_1(acc_s_1, sm_1, smp_1,
                                              ss_1, ssum_1, ls_1)
                                    T.copy(acc_s_1, acc_s_cast_1)
                                else:
                                    if gi_kc1 % 2 == 0:
                                        T.wgmma_gemm(
                                            q_shared_1, k_smem_0, acc_s_1,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            q_shared_1, k_smem_1, acc_s_1,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    rescale_1(acc_o_1, ss_1)
                                    T.barrier_wait(v_full, gi_vc1 % 2)
                                    if gi_vc1 % 2 == 0:
                                        T.wgmma_gemm(
                                            acc_s_cast_1, v_smem_0,
                                            acc_o_1,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            acc_s_cast_1, v_smem_1,
                                            acc_o_1,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    T.call_extern(
                                        "handle",
                                        "tl::barrier_arrive_named",
                                        2, 256)
                                    T.wait_wgmma(1)
                                    T.warpgroup_fence_operand(
                                        acc_s_1, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    softmax_1(acc_s_1, sm_1, smp_1,
                                              ss_1, ssum_1, ls_1)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(
                                        acc_o_1, num_regs=64)
                                    T.barrier_arrive(v_empty)
                                    T.copy(acc_s_1, acc_s_cast_1)
                                    gi_vc1 = gi_vc1 + 1
                                gi_kc1 = gi_kc1 + 1
                            # Consumer 1 epilogue: rescale + last PV
                            rescale_1(acc_o_1, ss_1)
                            T.barrier_wait(v_full, gi_vc1 % 2)
                            if gi_vc1 % 2 == 0:
                                T.wgmma_gemm(
                                    acc_s_cast_1, v_smem_0, acc_o_1,
                                    policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(
                                    acc_s_cast_1, v_smem_1, acc_o_1,
                                    policy=T.GemmWarpPolicy.FullRow)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_o_1, num_regs=64)
                            T.barrier_arrive(v_empty)
                            gi_vc1 = gi_vc1 + 1
                            # Output write for half 1
                            for i, j in T.Parallel(half_m, D):
                                acc_o_1[i, j] /= ls_1[i]
                            T.copy(acc_o_1, q_shared_1)
                            T.fence_proxy_async()
                            T.sync_threads(
                                barrier_id=3, arrive_count=128)
                            T.copy(q_shared_1,
                                   output[tile_b,
                                          row_base:row_base + half_m,
                                          tile_h, :])
                            for i in T.Parallel(half_m):
                                ls_1[i] = (T.log2(ls_1[i])
                                           + sm_1[i] * scale)
                            T.copy(ls_1,
                                   lse[tile_b, tile_h,
                                       row_base:row_base + half_m])

                # ===== WG2 (consumer 2, tx >= 256) =====
                else:
                    T.inc_max_nreg(240)
                    for tile_b, tile_h, pair_idx in T.Persistent(
                        [B, H, half_M_blocks],
                        wave_size=effective_num_sms,
                        index=bx,
                        group_size=8,
                    ):
                        for sub_idx in range(2):
                            # tile_m as a single TIR expression (no Python
                            # if-frame). sub_idx is Python int 0 or 1:
                            #   sub_idx=0 → tile_m = pair_idx
                            #   sub_idx=1 → tile_m = M_blocks - 1 - pair_idx
                            tile_m = (
                                pair_idx
                                + sub_idx * (M_blocks - 1 - 2 * pair_idx))
                            row_base = tile_m * block_m
                            loop_range = T.ceildiv(
                                (tile_m + 1) * block_m, block_n)

                            T.tma_copy(
                                q[tile_b,
                                  row_base + half_m:
                                  row_base + block_m,
                                  tile_h, :],
                                q_shared_2, barrier=q_full_2)
                            T.barrier_arrive(q_full_2)
                            T.barrier_wait(q_full_2, gi_q2 % 2)
                            gi_q2 = gi_q2 + 1

                            T.clear(acc_o_2)
                            T.clear(ls_2)
                            T.fill(sm_2, -T.infinity(accum_dtype))

                            for n_idx in T.Pipelined(
                                    loop_range, num_stages=0):
                                T.barrier_wait(k_full, gi_kc2 % 2)
                                T.sync_threads(
                                    barrier_id=2, arrive_count=256)
                                if n_idx == loop_range - 1:
                                    for i, j in T.Parallel(
                                            half_m, block_n):
                                        acc_s_2[i, j] = T.if_then_else(
                                            row_base + half_m + i
                                            >= n_idx * block_n + j,
                                            0,
                                            -T.infinity(accum_dtype))
                                else:
                                    T.clear(acc_s_2)
                                if n_idx == 0:
                                    if gi_kc2 % 2 == 0:
                                        T.wgmma_gemm(
                                            q_shared_2, k_smem_0, acc_s_2,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            q_shared_2, k_smem_1, acc_s_2,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    T.call_extern(
                                        "handle",
                                        "tl::barrier_arrive_named",
                                        1, 256)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(
                                        acc_s_2, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    softmax_2(acc_s_2, sm_2, smp_2,
                                              ss_2, ssum_2, ls_2)
                                    T.copy(acc_s_2, acc_s_cast_2)
                                else:
                                    if gi_kc2 % 2 == 0:
                                        T.wgmma_gemm(
                                            q_shared_2, k_smem_0, acc_s_2,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            q_shared_2, k_smem_1, acc_s_2,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    rescale_2(acc_o_2, ss_2)
                                    T.barrier_wait(v_full, gi_vc2 % 2)
                                    if gi_vc2 % 2 == 0:
                                        T.wgmma_gemm(
                                            acc_s_cast_2, v_smem_0,
                                            acc_o_2,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            acc_s_cast_2, v_smem_1,
                                            acc_o_2,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    T.call_extern(
                                        "handle",
                                        "tl::barrier_arrive_named",
                                        1, 256)
                                    T.wait_wgmma(1)
                                    T.warpgroup_fence_operand(
                                        acc_s_2, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    softmax_2(acc_s_2, sm_2, smp_2,
                                              ss_2, ssum_2, ls_2)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(
                                        acc_o_2, num_regs=64)
                                    T.barrier_arrive(v_empty)
                                    T.copy(acc_s_2, acc_s_cast_2)
                                    gi_vc2 = gi_vc2 + 1
                                gi_kc2 = gi_kc2 + 1
                            # Consumer 2 epilogue
                            rescale_2(acc_o_2, ss_2)
                            T.barrier_wait(v_full, gi_vc2 % 2)
                            if gi_vc2 % 2 == 0:
                                T.wgmma_gemm(
                                    acc_s_cast_2, v_smem_0, acc_o_2,
                                    policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(
                                    acc_s_cast_2, v_smem_1, acc_o_2,
                                    policy=T.GemmWarpPolicy.FullRow)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_o_2, num_regs=64)
                            T.barrier_arrive(v_empty)
                            gi_vc2 = gi_vc2 + 1
                            # Output write for half 2
                            for i, j in T.Parallel(half_m, D):
                                acc_o_2[i, j] /= ls_2[i]
                            T.copy(acc_o_2, q_shared_2)
                            T.fence_proxy_async()
                            T.sync_threads(
                                barrier_id=4, arrive_count=128)
                            T.copy(q_shared_2,
                                   output[tile_b,
                                          row_base + half_m:
                                          row_base + block_m,
                                          tile_h, :])
                            for i in T.Parallel(half_m):
                                ls_2[i] = (T.log2(ls_2[i])
                                           + sm_2[i] * scale)
                            T.copy(ls_2,
                                   lse[tile_b, tile_h,
                                       row_base + half_m:
                                       row_base + block_m])

        return main

    return func


# ---------------------------------------------------------------------------
def ref_gqa_causal(q, k, v):
    B, S, H, D = q.shape
    Hkv = k.shape[2]
    groups = H // Hkv
    k_exp = k[:, :, :, None, :].expand(
        B, S, Hkv, groups, D).reshape(B, S, H, D)
    v_exp = v[:, :, :, None, :].expand(
        B, S, Hkv, groups, D).reshape(B, S, H, D)
    qt = q.transpose(1, 2).float()
    kt = k_exp.transpose(1, 2).float()
    vt = v_exp.transpose(1, 2).float()
    sm = 1.0 / D**0.5
    attn = (qt @ kt.transpose(-2, -1)) * sm
    mask = torch.triu(
        torch.ones(S, S, device=q.device, dtype=torch.bool),
        diagonal=1)
    attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return (attn @ vt).transpose(1, 2).half()


def test(B, S, H, Hkv, D):
    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    o_ref = ref_gqa_causal(q, k, v)
    kernel = build_fa3_v2_persistent_paired(B, S, H, Hkv, D)
    o, lse = kernel(block_m=128, block_n=128)(q, k, v)
    diff = (o.float() - o_ref.float()).abs().max().item()
    ok = diff < 0.5
    M_blocks = (S + 128 - 1) // 128
    n_pairs = B * H * (M_blocks // 2)
    tag = (f"B={B} S={S} H={H} Hkv={Hkv} D={D} causal=1 "
           f"NUM_SMS={NUM_SMS} pairs={n_pairs}")
    print(f"  {tag}: diff={diff:.4f} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"=== v2 PERSISTENT PAIRED (causal, NUM_SMS={NUM_SMS}) ===")
    ok = True
    ok &= test(1, 256, 8, 4, 128)
    ok &= test(4, 512, 64, 4, 128)
    ok &= test(1, 1024, 32, 8, 128)
    ok &= test(2, 2048, 32, 8, 128)
    ok &= test(4, 2048, 64, 4, 128)
    ok &= test(4, 4096, 64, 8, 128)
    print(f"\n{'All passed!' if ok else 'SOME FAILED!'}")
