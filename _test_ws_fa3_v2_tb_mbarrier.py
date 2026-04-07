"""FA3-aligned IntraWGOverlap v2 — THREAD-BINDING variant.

Same logic as `_test_ws_fa3_v2.py`, but the producer/consumer/consumer
split uses `tx = T.get_thread_binding()` + raw `if/elif/else` instead of
`with T.ws(N):` blocks. Pattern modeled after
`tile-ai/tilelang::examples/deepseek_v32/sparse_mla_fwd_seesaw.py`.

Why this matters: TileLang's `T.ws(N)` lowering produces three independent
`if (cond_N) {...}` blocks in the lowered CUDA, which ptxas cannot prove
are mutually exclusive. v2 worked around that with two post-process hooks
(`shfl_transform` + `if_else_chain_rewrite`). With raw thread binding,
the Python `if/elif/else` is converted by the TIR parser into a proper
nested `T.If/T.Then/T.Else` chain natively, and the lowered CUDA already
has the `if/else if/else` shape ptxas needs. **No post-process required.**

Match for v2 b128 / 524 TFLOPS reference shape:
  - Q load CTA-wide before WG split (same as v2)
  - WG0 (tx < 128)         = producer:  TMA load K/V, dec(24) regs/thread
  - WG1 (128 <= tx < 256)  = consumer 1: rows 0..half_m, inc(240)
  - WG2 (tx >= 256)        = consumer 2: rows half_m..block_m, inc(240)
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


def build_fa3_v2(B, S, H, Hkv, D, is_causal,
                 block_m=128, block_n=128):
    assert H % Hkv == 0 and block_m % 2 == 0 and D == 128
    half_m = block_m // 2
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
                T.ceildiv(S, block_m), H, B, threads=384,
            ) as (bx, by, bz):
                # ---- Shared memory ----
                q_shared_1 = T.alloc_shared([half_m, D], "float16")
                q_shared_2 = T.alloc_shared([half_m, D], "float16")
                # Double-buffered K and V
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

                # ---- Pipeline barriers (FA3-aligned) ----
                # K pipeline: producer→consumer (data ready)
                k_full = T.alloc_barrier(arrive_count=128)
                # K pipeline: consumer→producer (buffer free)
                # arrive_count=256: both WG1(128) + WG2(128) must arrive
                k_empty = T.alloc_barrier(arrive_count=256)
                # V pipeline
                v_full = T.alloc_barrier(arrive_count=128)
                v_empty = T.alloc_barrier(arrive_count=256)
                # Scheduler mbarriers (replaces named-barrier ping-pong).
                # bar_sched_1: WG2 arrives after wgmma_QK -> releases WG1.
                # bar_sched_2: WG1 arrives after wgmma_QK -> releases WG2.
                bar_sched_1 = T.alloc_barrier(arrive_count=128)
                bar_sched_2 = T.alloc_barrier(arrive_count=128)

                T.annotate_layout({
                    q_shared_1:
                        tilelang.layout.make_swizzled_layout(q_shared_1),
                    q_shared_2:
                        tilelang.layout.make_swizzled_layout(q_shared_2),
                })

                T.sync_threads()  # after barrier init

                head_kv = by // groups
                row_base = bx * block_m
                loop_range = (
                    T.ceildiv((bx + 1) * block_m, block_n)
                    if is_causal else T.ceildiv(S, block_n))

                T.copy(q[bz, row_base:row_base + half_m, by, :],
                       q_shared_1)
                T.copy(q[bz, row_base + half_m:row_base + block_m,
                         by, :], q_shared_2)

                T.sync_threads()  # after Q loads

                # =============================================
                # FA3-aligned per-warpgroup body layout (THREAD-BIND variant)
                # =============================================
                # tx is the raw threadIdx.x. Python if/elif/else lowers via
                # the TIR parser into nested T.If/T.Then/T.Else, which ptxas
                # sees as a true if/elseif/else tree (mutually exclusive
                # branches). No need for shfl_transform / if_else_chain
                # post-process.
                #
                # FA3-style register reallocation (setmaxnreg):
                #   Producer dec:  128 * (168 - 24)  = 18432 regs released
                #   Consumer inc:  256 * (240 - 168) = 18432 regs claimed
                # 24/240 are the only numbers that match for 1+2 WG split.

                tx = T.get_thread_binding()

                # ===== WG0 (producer, tx < 128) — entire producer life cycle =====
                if tx < 128:
                    T.dec_max_nreg(24)
                    for n_idx in T.Pipelined(loop_range, num_stages=0):
                        # Acquire K stage: wait for consumers to free it
                        T.barrier_wait(k_empty, (n_idx + 1) % 2)
                        if n_idx % 2 == 0:
                            T.tma_copy(
                                k[bz, n_idx * block_n:
                                  (n_idx + 1) * block_n,
                                  head_kv, :],
                                k_smem_0, barrier=k_full)
                        else:
                            T.tma_copy(
                                k[bz, n_idx * block_n:
                                  (n_idx + 1) * block_n,
                                  head_kv, :],
                                k_smem_1, barrier=k_full)
                        T.barrier_arrive(k_full)
                        # Load V[n-1] into v_smem[(n-1)%2]
                        if n_idx > 0:
                            T.barrier_wait(v_empty, n_idx % 2)
                            if (n_idx - 1) % 2 == 0:
                                T.tma_copy(
                                    v[bz,
                                      (n_idx - 1) * block_n:
                                      n_idx * block_n,
                                      head_kv, :],
                                    v_smem_0, barrier=v_full)
                            else:
                                T.tma_copy(
                                    v[bz,
                                      (n_idx - 1) * block_n:
                                      n_idx * block_n,
                                      head_kv, :],
                                    v_smem_1, barrier=v_full)
                            T.barrier_arrive(v_full)
                    # Producer epilogue: tail load V[loop_range-1]
                    T.barrier_wait(v_empty, loop_range % 2)
                    if (loop_range - 1) % 2 == 0:
                        T.tma_copy(
                            v[bz,
                              (loop_range - 1) * block_n:
                              loop_range * block_n,
                              head_kv, :],
                            v_smem_0, barrier=v_full)
                    else:
                        T.tma_copy(
                            v[bz,
                              (loop_range - 1) * block_n:
                              loop_range * block_n,
                              head_kv, :],
                            v_smem_1, barrier=v_full)
                    T.barrier_arrive(v_full)

                # ===== WG1 (consumer 1, 128 <= tx < 256) — entire life cycle =====
                elif tx < 256:
                    T.inc_max_nreg(240)
                    # No bootstrap needed: WG1 leads by half a phase, skips
                    # the wait on iter 0 (WG2 hasn't released bar_sched_1 yet).
                    T.clear(acc_o_1)
                    T.clear(ls_1)
                    T.fill(sm_1, -T.infinity(accum_dtype))
                    for n_idx in T.Pipelined(loop_range, num_stages=0):
                        T.barrier_wait(k_full, n_idx % 2)
                        # Wait on WG2's release from previous iter (skip iter 0).
                        if n_idx > 0:
                            T.barrier_wait(bar_sched_1, (n_idx - 1) % 2)
                        if is_causal:
                            for i, j in T.Parallel(half_m, block_n):
                                acc_s_1[i, j] = T.if_then_else(
                                    row_base + i
                                    >= n_idx * block_n + j,
                                    0, -T.infinity(accum_dtype))
                        else:
                            T.clear(acc_s_1)
                        if n_idx == 0:
                            T.wgmma_gemm(
                                q_shared_1, k_smem_0, acc_s_1,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow)
                            # Release WG2 (it can issue its wgmma_QK now).
                            T.barrier_arrive(bar_sched_2)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_s_1, num_regs=64)
                            T.barrier_arrive(k_empty)
                            softmax_1(acc_s_1, sm_1, smp_1,
                                      ss_1, ssum_1, ls_1)
                            T.copy(acc_s_1, acc_s_cast_1)
                        else:
                            if n_idx % 2 == 0:
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
                            T.barrier_wait(v_full, (n_idx - 1) % 2)
                            if (n_idx - 1) % 2 == 0:
                                T.wgmma_gemm(
                                    acc_s_cast_1, v_smem_0,
                                    acc_o_1,
                                    policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(
                                    acc_s_cast_1, v_smem_1,
                                    acc_o_1,
                                    policy=T.GemmWarpPolicy.FullRow)
                            # Release WG2 (it can issue its wgmma_QK now).
                            T.barrier_arrive(bar_sched_2)
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
                    # Consumer 1 epilogue: rescale + last PV
                    rescale_1(acc_o_1, ss_1)
                    T.barrier_wait(v_full, (loop_range - 1) % 2)
                    if (loop_range - 1) % 2 == 0:
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
                    # Output write for half 1
                    for i, j in T.Parallel(half_m, D):
                        acc_o_1[i, j] /= ls_1[i]
                    T.copy(acc_o_1, q_shared_1)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=3, arrive_count=128)
                    T.copy(q_shared_1,
                           output[bz, row_base:row_base + half_m,
                                  by, :])
                    for i in T.Parallel(half_m):
                        ls_1[i] = (T.log2(ls_1[i])
                                   + sm_1[i] * scale)
                    T.copy(ls_1,
                           lse[bz, by,
                               row_base:row_base + half_m])

                # ===== WG2 (consumer 2, tx >= 256) — entire life cycle =====
                else:
                    T.inc_max_nreg(240)
                    T.clear(acc_o_2)
                    T.clear(ls_2)
                    T.fill(sm_2, -T.infinity(accum_dtype))
                    for n_idx in T.Pipelined(loop_range, num_stages=0):
                        T.barrier_wait(k_full, n_idx % 2)
                        # Wait on WG1's release of bar_sched_2 (every iter,
                        # WG1 leads so iter 0 already has WG1's release).
                        T.barrier_wait(bar_sched_2, n_idx % 2)
                        if is_causal:
                            for i, j in T.Parallel(half_m, block_n):
                                acc_s_2[i, j] = T.if_then_else(
                                    row_base + half_m + i
                                    >= n_idx * block_n + j,
                                    0, -T.infinity(accum_dtype))
                        else:
                            T.clear(acc_s_2)
                        if n_idx == 0:
                            T.wgmma_gemm(
                                q_shared_2, k_smem_0, acc_s_2,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow)
                            # Release WG1 (so WG1 iter 1 can proceed).
                            T.barrier_arrive(bar_sched_1)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_s_2, num_regs=64)
                            T.barrier_arrive(k_empty)
                            softmax_2(acc_s_2, sm_2, smp_2,
                                      ss_2, ssum_2, ls_2)
                            T.copy(acc_s_2, acc_s_cast_2)
                        else:
                            if n_idx % 2 == 0:
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
                            T.barrier_wait(v_full, (n_idx - 1) % 2)
                            if (n_idx - 1) % 2 == 0:
                                T.wgmma_gemm(
                                    acc_s_cast_2, v_smem_0,
                                    acc_o_2,
                                    policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(
                                    acc_s_cast_2, v_smem_1,
                                    acc_o_2,
                                    policy=T.GemmWarpPolicy.FullRow)
                            # Release WG1.
                            T.barrier_arrive(bar_sched_1)
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
                    # Consumer 2 epilogue: rescale + last PV
                    rescale_2(acc_o_2, ss_2)
                    T.barrier_wait(v_full, (loop_range - 1) % 2)
                    if (loop_range - 1) % 2 == 0:
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
                    # Output write for half 2
                    for i, j in T.Parallel(half_m, D):
                        acc_o_2[i, j] /= ls_2[i]
                    T.copy(acc_o_2, q_shared_2)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=4, arrive_count=128)
                    T.copy(q_shared_2,
                           output[bz, row_base + half_m:
                                  row_base + block_m, by, :])
                    for i in T.Parallel(half_m):
                        ls_2[i] = (T.log2(ls_2[i])
                                   + sm_2[i] * scale)
                    T.copy(ls_2,
                           lse[bz, by,
                               row_base + half_m:
                               row_base + block_m])

        return main

    return func


# ---------------------------------------------------------------------------
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
    kernel = build_fa3_v2(B, S, H, Hkv, D, is_causal)
    o, lse = kernel(block_m=128, block_n=128)(q, k, v)
    diff = (o.float() - o_ref.float()).abs().max().item()
    ok = diff < 0.5
    tag = (f"B={B} S={S} H={H} Hkv={Hkv} D={D} "
           f"causal={is_causal}")
    print(f"  {tag}: diff={diff:.4f} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=== FA3-aligned v2 Correctness ===")
    ok = True
    ok &= test(1, 256, 8, 4, 128, False)
    ok &= test(4, 512, 64, 4, 128, False)
    ok &= test(4, 512, 64, 4, 128, True)
    ok &= test(1, 1024, 32, 8, 128, True)
    ok &= test(2, 2048, 32, 8, 128, True)
    print(f"\n{'All passed!' if ok else 'SOME FAILED!'}")
