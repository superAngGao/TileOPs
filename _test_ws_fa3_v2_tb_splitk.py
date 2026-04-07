"""FA3-aligned IntraWGOverlap v2 — SPLIT-K + thread-binding.

Architecture change vs row-split thread-bind:
  - Row-split: each consumer handles half the M rows, sees ALL K iterations
  - Split-K (this file): each consumer handles FULL block_m rows, sees HALF
    the K iterations (consumer 0 = even K's, consumer 1 = odd K's)

Why: with split-K, the producer can issue 2 K TMAs in parallel per outer
iter (one to even buf, one to odd buf), and 2 V TMAs likewise. This
doubles the K/V load issue rate, hiding more of the long_scoreboard
HBM/L2 latency that ncu showed dominates the 14% gap to FA3.

Mbarrier design (per-stream, total 4):
  - mbar_k_e: mbarrier for K[even] stream. TMA arrives → ready;
              consumer 0 arrives → empty. Phase tracks ready/empty cycle.
  - mbar_k_o: same for K[odd] stream / consumer 1.
  - mbar_v_e: V[even] stream / consumer 0.
  - mbar_v_o: V[odd] stream / consumer 1.
  - merge_bar: epilogue-only sync for the partial-O dump (arrive_count=256,
                arrived once per consumer thread). NOT used during main loop.

Each mbarrier handles BOTH ready and empty signaling for one stream via
phase rotation. The producer's TMA contributes "ready" arrives; the
consumer contributes "empty" arrives after consuming.

block_m = 64 (single-tile, both consumers process the FULL block_m rows
on different K subsets). This gives each consumer a fragment footprint
identical to the row-split half_m=64 case (no register spill).

Epilogue merge (D-split):
  1. Each consumer dumps its acc_o (full block_m × D fp32) to its own
     smem region (no overlap).
  2. Both consumers arrive merge_bar; both wait merge_bar.
  3. WG0 reads m, l from both consumers (small per-row scalars), computes
     the merged m, l, then merges acc_o for D[0..64] half and writes to
     global output.
  4. WG1 does the same for D[64..128].
  5. m, l exchange goes via tiny smem buffers (block_m floats each).

Producer iter cadence:
  Outer loop iter j = 0 .. (loop_range/2 - 1):
    - TMA K[2j]   → k_smem_e (binds mbar_k_e)
    - TMA K[2j+1] → k_smem_o (binds mbar_k_o)
    - if j > 0:
        TMA V[2j-2] → v_smem_e (binds mbar_v_e)  // for consumer 0 iter j PV
        TMA V[2j-1] → v_smem_o (binds mbar_v_o)  // for consumer 1 iter j PV
  Producer epilogue:
    - TMA V[2N-2] → v_smem_e
    - TMA V[2N-1] → v_smem_o
  (where N = loop_range / 2)

Consumer 0 / 1 iter cadence (per WG, j = 0 .. N-1):
  - wait mbar_k_*  (K[2j] or K[2j+1] ready)
  - wgmma QK
  - online softmax (own m_i, l_i, sumexp updates)
  - if j > 0:
      wait mbar_v_* (V[2(j-1)] or V[2(j-1)+1] ready)
      wgmma PV with previous P @ V
      arrive mbar_k_* (release K buf)
      arrive mbar_v_* (release V buf)
  - save P (acc_s_cast)
Consumer epilogue:
  - wait last V, do final PV
  - rescale acc_o, dump to smem
  - merge_bar arrive + wait
  - merge with peer's m, l, do D-split write

Edge cases:
  - loop_range must be EVEN for clean split-K. Production ref shape
    (S=4096, block_n=128) gives loop_range=32 → N=16, even. For odd
    loop_range, the last K tile would be processed by only one consumer
    and the other would have one fewer iter — special-case logic needed
    (not implemented in v1).
"""
import time

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.online_softmax import (
    make_log2e_scale,
    make_online_softmax_with_mask_guard,
    make_rescale,
)


def build_fa3_v2_splitk(B, S, H, Hkv, D, is_causal,
                        block_m=64, block_n=64):
    assert H % Hkv == 0 and D == 128
    assert block_m == 64, "split-K only supports block_m=64 (single tile per CTA)"
    assert block_n == 64, "split-K v2 uses block_n=64 for double-buffer fit"
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

        # Each consumer's softmax operates on full block_m rows (since
        # split-K means each consumer owns full Q rows, partial K).
        softmax_e = make_online_softmax_with_mask_guard(
            scale, accum_dtype, block_m, block_n)
        softmax_o = make_online_softmax_with_mask_guard(
            scale, accum_dtype, block_m, block_n)
        rescale_e = make_rescale(block_m, D)
        rescale_o = make_rescale(block_m, D)

        D_half = D // 2  # 64; for the epilogue D-split

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float16"),
            k: T.Tensor(kv_shape, "float16"),
            v: T.Tensor(kv_shape, "float16"),
            output: T.Tensor(q_shape, "float16"),
            lse: T.Tensor([B, H, S], accum_dtype),
        ) -> None:
            with T.Kernel(
                T.ceildiv(S, block_m), H, B, threads=384,
            ) as (bx, by, bz):
                # ---- Shared memory ----
                # Single Q tile, full block_m rows. Both consumers read it.
                q_shared = T.alloc_shared([block_m, D], "float16")
                # Per-stream double-buffered K and V (block_n=64 fits)
                k_smem_e_0 = T.alloc_shared([block_n, D], "float16")
                k_smem_e_1 = T.alloc_shared([block_n, D], "float16")
                k_smem_o_0 = T.alloc_shared([block_n, D], "float16")
                k_smem_o_1 = T.alloc_shared([block_n, D], "float16")
                v_smem_e_0 = T.alloc_shared([block_n, D], "float16")
                v_smem_e_1 = T.alloc_shared([block_n, D], "float16")
                v_smem_o_0 = T.alloc_shared([block_n, D], "float16")
                v_smem_o_1 = T.alloc_shared([block_n, D], "float16")
                # ---- Epilogue merge dumps (smem) ----
                # Each consumer dumps its full acc_o (fp32) here.
                o_dump_e = T.alloc_shared([block_m, D], accum_dtype)
                o_dump_o = T.alloc_shared([block_m, D], accum_dtype)
                # m and l per consumer; rows packed as [2, block_m]
                # row 0 = even, row 1 = odd
                ml_dump = T.alloc_shared([4, block_m], accum_dtype)
                # Layout: ml_dump[0]=m_e, ml_dump[1]=l_e, ml_dump[2]=m_o, ml_dump[3]=l_o

                # ---- Fragments (per consumer) ----
                # Consumer 0 (even K stream): full block_m rows
                acc_s_e = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast_e = T.alloc_fragment(
                    [block_m, block_n], "float16")
                acc_o_e = T.alloc_fragment([block_m, D], accum_dtype)
                sm_e = T.alloc_fragment([block_m], accum_dtype)
                smp_e = T.alloc_fragment([block_m], accum_dtype)
                ss_e = T.alloc_fragment([block_m], accum_dtype)
                ssum_e = T.alloc_fragment([block_m], accum_dtype)
                ls_e = T.alloc_fragment([block_m], accum_dtype)

                # Consumer 1 (odd K stream)
                acc_s_o = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast_o = T.alloc_fragment(
                    [block_m, block_n], "float16")
                acc_o_o = T.alloc_fragment([block_m, D], accum_dtype)
                sm_o = T.alloc_fragment([block_m], accum_dtype)
                smp_o = T.alloc_fragment([block_m], accum_dtype)
                ss_o = T.alloc_fragment([block_m], accum_dtype)
                ssum_o = T.alloc_fragment([block_m], accum_dtype)
                ls_o = T.alloc_fragment([block_m], accum_dtype)

                # ---- Mbarriers — strictly following the 2-stage baseline
                #      pattern: separate full/empty mbarriers per direction.
                # 8 mbarriers total (4 per stream × 2 streams).
                # Each stream is single-buffered, so no in-stream phase
                # alternation — phase tracking same as 2-stage.
                # k_*_full: arrive_count=128 (producer warp arrives via TMA
                #          + manual T.barrier_arrive after T.tma_copy)
                # k_*_empty: arrive_count=128 (single consumer warp arrives
                #           when done with that stream's buffer)
                k_e_full = T.alloc_barrier(arrive_count=128)
                k_e_empty = T.alloc_barrier(arrive_count=128)
                k_o_full = T.alloc_barrier(arrive_count=128)
                k_o_empty = T.alloc_barrier(arrive_count=128)
                v_e_full = T.alloc_barrier(arrive_count=128)
                v_e_empty = T.alloc_barrier(arrive_count=128)
                v_o_full = T.alloc_barrier(arrive_count=128)
                v_o_empty = T.alloc_barrier(arrive_count=128)
                # Epilogue merge sync: 2 consumer WGs arrive (256 threads).
                merge_bar = T.alloc_barrier(arrive_count=256)

                T.annotate_layout({
                    q_shared:
                        tilelang.layout.make_swizzled_layout(q_shared),
                })

                T.sync_threads()  # after barrier init

                head_kv = by // groups
                row_base = bx * block_m
                # In v1 we assume non-causal + S divisible by block_n*2,
                # so loop_range is a Python int and N = loop_range // 2.
                # This lets us use plain Python ints in T.Pipelined etc.
                assert not is_causal, "split-K v1: causal not supported"
                loop_range_py = (S + block_n - 1) // block_n
                assert loop_range_py % 2 == 0, (
                    f"split-K v1 needs loop_range even, got {loop_range_py}")
                N = loop_range_py // 2
                loop_range = loop_range_py

                # ---- Q load (CTA-wide, full block_m rows) ----
                T.copy(q[bz, row_base:row_base + block_m, by, :],
                       q_shared)
                T.sync_threads()  # after Q load

                tx = T.get_thread_binding()

                # =========================================================
                # WG0 (producer, tx < 128) — issues TMA pairs
                # =========================================================
                # Per outer iter j = 0..N-1:
                #   - K[2j]   → k_smem_e
                #   - K[2j+1] → k_smem_o
                #   - V[2j-2] → v_smem_e   (j > 0)
                #   - V[2j-1] → v_smem_o   (j > 0)
                # Epilogue: V[2N-2] → v_smem_e, V[2N-1] → v_smem_o
                if tx < 128:
                    T.dec_max_nreg(24)
                    for j in T.Pipelined(N, num_stages=0):
                        # K wait empty: parity (j+1)%2 (matches reference)
                        T.barrier_wait(k_e_empty, (j + 1) % 2)
                        T.barrier_wait(k_o_empty, (j + 1) % 2)
                        # Alternate buf 0 / buf 1 based on j%2
                        if j % 2 == 0:
                            T.tma_copy(
                                k[bz, (2 * j) * block_n:
                                  (2 * j + 1) * block_n,
                                  head_kv, :],
                                k_smem_e_0, barrier=k_e_full)
                            T.tma_copy(
                                k[bz, (2 * j + 1) * block_n:
                                  (2 * j + 2) * block_n,
                                  head_kv, :],
                                k_smem_o_0, barrier=k_o_full)
                        else:
                            T.tma_copy(
                                k[bz, (2 * j) * block_n:
                                  (2 * j + 1) * block_n,
                                  head_kv, :],
                                k_smem_e_1, barrier=k_e_full)
                            T.tma_copy(
                                k[bz, (2 * j + 1) * block_n:
                                  (2 * j + 2) * block_n,
                                  head_kv, :],
                                k_smem_o_1, barrier=k_o_full)
                        T.barrier_arrive(k_e_full)
                        T.barrier_arrive(k_o_full)
                        # V load (for j>0): V[2(j-1)] / V[2(j-1)+1]
                        # Buf alternates based on (j-1)%2.
                        if j > 0:
                            T.barrier_wait(v_e_empty, j % 2)
                            T.barrier_wait(v_o_empty, j % 2)
                            if (j - 1) % 2 == 0:
                                T.tma_copy(
                                    v[bz, (2 * j - 2) * block_n:
                                      (2 * j - 1) * block_n,
                                      head_kv, :],
                                    v_smem_e_0, barrier=v_e_full)
                                T.tma_copy(
                                    v[bz, (2 * j - 1) * block_n:
                                      (2 * j) * block_n,
                                      head_kv, :],
                                    v_smem_o_0, barrier=v_o_full)
                            else:
                                T.tma_copy(
                                    v[bz, (2 * j - 2) * block_n:
                                      (2 * j - 1) * block_n,
                                      head_kv, :],
                                    v_smem_e_1, barrier=v_e_full)
                                T.tma_copy(
                                    v[bz, (2 * j - 1) * block_n:
                                      (2 * j) * block_n,
                                      head_kv, :],
                                    v_smem_o_1, barrier=v_o_full)
                            T.barrier_arrive(v_e_full)
                            T.barrier_arrive(v_o_full)
                    # Producer epilogue: tail V loads V[2N-2], V[2N-1]
                    # v_idx = N-1 → buf (N-1)%2
                    T.barrier_wait(v_e_empty, N % 2)
                    T.barrier_wait(v_o_empty, N % 2)
                    if (N - 1) % 2 == 0:
                        T.tma_copy(
                            v[bz, (2 * N - 2) * block_n:
                              (2 * N - 1) * block_n,
                              head_kv, :],
                            v_smem_e_0, barrier=v_e_full)
                        T.tma_copy(
                            v[bz, (2 * N - 1) * block_n:
                              (2 * N) * block_n,
                              head_kv, :],
                            v_smem_o_0, barrier=v_o_full)
                    else:
                        T.tma_copy(
                            v[bz, (2 * N - 2) * block_n:
                              (2 * N - 1) * block_n,
                              head_kv, :],
                            v_smem_e_1, barrier=v_e_full)
                        T.tma_copy(
                            v[bz, (2 * N - 1) * block_n:
                              (2 * N) * block_n,
                              head_kv, :],
                            v_smem_o_1, barrier=v_o_full)
                    T.barrier_arrive(v_e_full)
                    T.barrier_arrive(v_o_full)

                # =========================================================
                # WG1 (consumer 0, even K stream, 128 <= tx < 256)
                # =========================================================
                elif tx < 256:
                    T.inc_max_nreg(240)
                    T.clear(acc_o_e)
                    T.clear(ls_e)
                    T.fill(sm_e, -T.infinity(accum_dtype))
                    for j in T.Pipelined(N, num_stages=0):
                        T.barrier_wait(k_e_full, j % 2)
                        T.clear(acc_s_e)
                        # QK with the buf for this iter (j%2 alternates)
                        if j % 2 == 0:
                            T.wgmma_gemm(
                                q_shared, k_smem_e_0, acc_s_e,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow)
                        else:
                            T.wgmma_gemm(
                                q_shared, k_smem_e_1, acc_s_e,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow)
                        if j == 0:
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_s_e, num_regs=64)
                            T.barrier_arrive(k_e_empty)
                            softmax_e(acc_s_e, sm_e, smp_e,
                                      ss_e, ssum_e, ls_e)
                            T.copy(acc_s_e, acc_s_cast_e)
                        else:
                            rescale_e(acc_o_e, ss_e)
                            T.barrier_wait(v_e_full, (j - 1) % 2)
                            # PV with v_buf[(j-1)%2]
                            if (j - 1) % 2 == 0:
                                T.wgmma_gemm(
                                    acc_s_cast_e, v_smem_e_0, acc_o_e,
                                    policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(
                                    acc_s_cast_e, v_smem_e_1, acc_o_e,
                                    policy=T.GemmWarpPolicy.FullRow)
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(
                                acc_s_e, num_regs=64)
                            T.barrier_arrive(k_e_empty)
                            softmax_e(acc_s_e, sm_e, smp_e,
                                      ss_e, ssum_e, ls_e)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_o_e, num_regs=64)
                            T.barrier_arrive(v_e_empty)
                            T.copy(acc_s_e, acc_s_cast_e)
                    # Consumer 0 epilogue: do final PV with V[2(N-1)]
                    rescale_e(acc_o_e, ss_e)
                    T.barrier_wait(v_e_full, (N - 1) % 2)
                    if (N - 1) % 2 == 0:
                        T.wgmma_gemm(
                            acc_s_cast_e, v_smem_e_0, acc_o_e,
                            policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.wgmma_gemm(
                            acc_s_cast_e, v_smem_e_1, acc_o_e,
                            policy=T.GemmWarpPolicy.FullRow)
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(
                        acc_o_e, num_regs=64)
                    T.barrier_arrive(v_e_empty)
                    # ---- Epilogue merge (D-split) ----
                    # Step 1: dump unnormalized acc_o_e + (m_e, l_e) to smem
                    T.copy(acc_o_e, o_dump_e)
                    for i in T.Parallel(block_m):
                        ml_dump[0, i] = sm_e[i]
                        ml_dump[1, i] = ls_e[i]
                    T.fence_proxy_async()
                    T.barrier_arrive(merge_bar)
                    T.barrier_wait(merge_bar, 0)
                    # Step 2: read peer's m, l; compute global m, l
                    for i in T.Parallel(block_m):
                        m_self = ml_dump[0, i]
                        l_self = ml_dump[1, i]
                        m_peer = ml_dump[2, i]
                        l_peer = ml_dump[3, i]
                        m_max = T.max(m_self, m_peer)
                        ss_e[i] = T.exp2((m_self - m_max) * scale)
                        ssum_e[i] = T.exp2((m_peer - m_max) * scale)
                        ls_e[i] = (l_self * ss_e[i]
                                   + l_peer * ssum_e[i])
                        sm_e[i] = m_max
                    # Step 3: WG0 owns D[0..D_half] of output. Merge each row's
                    # left-half D from o_dump_e and o_dump_o, normalize.
                    for i, d in T.Parallel(block_m, D_half):
                        acc_o_e[i, d] = (
                            (o_dump_e[i, d] * ss_e[i]
                             + o_dump_o[i, d] * ssum_e[i])
                            / ls_e[i])
                    # Step 4: write left-half D to global
                    for i, d in T.Parallel(block_m, D_half):
                        output[bz, row_base + i, by, d] = T.cast(
                            acc_o_e[i, d], "float16")
                    # Step 5: write LSE (only WG0 writes, full block_m rows)
                    for i in T.Parallel(block_m):
                        ls_e[i] = (T.log2(ls_e[i])
                                   + sm_e[i] * scale)
                    T.copy(ls_e,
                           lse[bz, by,
                               row_base:row_base + block_m])

                # =========================================================
                # WG2 (consumer 1, odd K stream, tx >= 256)
                # =========================================================
                else:
                    T.inc_max_nreg(240)
                    T.clear(acc_o_o)
                    T.clear(ls_o)
                    T.fill(sm_o, -T.infinity(accum_dtype))
                    for j in T.Pipelined(N, num_stages=0):
                        T.barrier_wait(k_o_full, j % 2)
                        T.clear(acc_s_o)
                        if j % 2 == 0:
                            T.wgmma_gemm(
                                q_shared, k_smem_o_0, acc_s_o,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow)
                        else:
                            T.wgmma_gemm(
                                q_shared, k_smem_o_1, acc_s_o,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow)
                        if j == 0:
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_s_o, num_regs=64)
                            T.barrier_arrive(k_o_empty)
                            softmax_o(acc_s_o, sm_o, smp_o,
                                      ss_o, ssum_o, ls_o)
                            T.copy(acc_s_o, acc_s_cast_o)
                        else:
                            rescale_o(acc_o_o, ss_o)
                            T.barrier_wait(v_o_full, (j - 1) % 2)
                            if (j - 1) % 2 == 0:
                                T.wgmma_gemm(
                                    acc_s_cast_o, v_smem_o_0, acc_o_o,
                                    policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(
                                    acc_s_cast_o, v_smem_o_1, acc_o_o,
                                    policy=T.GemmWarpPolicy.FullRow)
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(
                                acc_s_o, num_regs=64)
                            T.barrier_arrive(k_o_empty)
                            softmax_o(acc_s_o, sm_o, smp_o,
                                      ss_o, ssum_o, ls_o)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_o_o, num_regs=64)
                            T.barrier_arrive(v_o_empty)
                            T.copy(acc_s_o, acc_s_cast_o)
                    # Consumer 1 epilogue
                    rescale_o(acc_o_o, ss_o)
                    T.barrier_wait(v_o_full, (N - 1) % 2)
                    if (N - 1) % 2 == 0:
                        T.wgmma_gemm(
                            acc_s_cast_o, v_smem_o_0, acc_o_o,
                            policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.wgmma_gemm(
                            acc_s_cast_o, v_smem_o_1, acc_o_o,
                            policy=T.GemmWarpPolicy.FullRow)
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(
                        acc_o_o, num_regs=64)
                    T.barrier_arrive(v_o_empty)
                    # ---- Epilogue merge (D-split) ----
                    T.copy(acc_o_o, o_dump_o)
                    for i in T.Parallel(block_m):
                        ml_dump[2, i] = sm_o[i]
                        ml_dump[3, i] = ls_o[i]
                    T.fence_proxy_async()
                    T.barrier_arrive(merge_bar)
                    T.barrier_wait(merge_bar, 0)
                    for i in T.Parallel(block_m):
                        m_self = ml_dump[2, i]
                        l_self = ml_dump[3, i]
                        m_peer = ml_dump[0, i]
                        l_peer = ml_dump[1, i]
                        m_max = T.max(m_self, m_peer)
                        ss_o[i] = T.exp2((m_self - m_max) * scale)
                        ssum_o[i] = T.exp2((m_peer - m_max) * scale)
                        ls_o[i] = (l_self * ss_o[i]
                                   + l_peer * ssum_o[i])
                    # WG2 owns D[D_half..D]; merge right half
                    for i, d in T.Parallel(block_m, D_half):
                        acc_o_o[i, d] = (
                            (o_dump_o[i, D_half + d] * ss_o[i]
                             + o_dump_e[i, D_half + d] * ssum_o[i])
                            / ls_o[i])
                    # Write right half D to global
                    for i, d in T.Parallel(block_m, D_half):
                        output[bz, row_base + i, by, D_half + d] = T.cast(
                            acc_o_o[i, d], "float16")

        return main

    return func


def ref_gqa(q, k, v, is_causal):
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
    if is_causal:
        mask = torch.triu(
            torch.ones(S, S, device=q.device, dtype=torch.bool),
            diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return (attn @ vt).transpose(1, 2).half()


def test(B, S, H, Hkv, D, is_causal):
    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    o_ref = ref_gqa(q, k, v, is_causal)
    kernel = build_fa3_v2_splitk(B, S, H, Hkv, D, is_causal)
    o, lse = kernel(block_m=64, block_n=128)(q, k, v)
    diff = (o.float() - o_ref.float()).abs().max().item()
    ok = diff < 0.5
    tag = (f"B={B} S={S} H={H} Hkv={Hkv} D={D} "
           f"causal={is_causal}")
    print(f"  {tag}: diff={diff:.4f} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=== Split-K v2 Correctness ===")
    ok = True
    ok &= test(4, 4096, 64, 8, 128, False)  # production reference
    print(f"\n{'PASS' if ok else 'FAIL'}")
