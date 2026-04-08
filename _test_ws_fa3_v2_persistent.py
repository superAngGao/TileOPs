"""FA3-aligned IntraWGOverlap v2 — PERSISTENT variant (Approach A).

Built on top of `_test_ws_fa3_v2_threadbind.py`. Adds persistent CTA
scheduling: each CTA processes multiple (B, H, M-block) tiles via
`T.Persistent`, instead of grid = (M, H, B) one-tile-per-CTA.

Key design choices (= why this differs from v3 and v4 attempts):

1. **Approach A: global iteration counters for mbarrier parity**
   Per-WG `T.alloc_var("int32")` counters (gi_kp/gi_vp/gi_kc/gi_vc) that
   monotonically accumulate across tiles. ALL barrier_wait parities use
   `gi % 2` instead of `n_idx % 2`. The mbarrier hardware phase is
   cumulative across tiles, so per-tile n_idx parity is wrong; gi % 2
   is right.

2. **Per-WG Q smem ownership**
   Q load is INSIDE each consumer branch (q_shared_1 owned by WG1,
   q_shared_2 owned by WG2). WG0 (producer) never touches q_shared_*.
   This eliminates the cross-WG race that would otherwise need a
   tma_store_wait fence at tile boundaries: same-WG cp.async + tma_store
   on the same smem region are ordered by warp-scheduler program order +
   async-proxy in-warp FIFO.

3. **Bootstrap stays OUTSIDE the persistent loop**
   `tl::barrier_arrive_named(1, 256)` is fired ONCE per CTA at WG1 entry
   (not per tile). Bar 1 leftover counter (=128 after each tile) is the
   stable steady state; the next tile's first WG1 sync naturally absorbs
   it without re-bootstrapping. v4's per-tile drain was wrong.

4. **No v_empty extra arrive, no bar 1 drain**
   Both were v4 hacks that masked symptoms of the n_idx parity bug.
   With Approach A, neither is needed.

Non-causal only for now. Causal needs additional handling for the
loop_range=1 edge case at tile_m=0 where bar 1/2 carryover becomes
asymmetric.

Per-WG global counter increments:
  gi_kp, gi_kc1, gi_kc2 += 1 after every K load / K wgmma (every iter)
  gi_vp += 1 after every V load (skips n_idx=0, +1 in epilogue tail load)
  gi_vc1, gi_vc2 += 1 after every V wgmma (skips n_idx=0, +1 in epilogue)
  Per tile, all 6 counters increase by exactly loop_range.
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


def build_fa3_v2_persistent(B, S, H, Hkv, D, is_causal,
                            block_m=128, block_n=128):
    assert H % Hkv == 0 and block_m % 2 == 0 and D == 128
    assert not is_causal, "non-causal only for v1 (loop_range=1 edge case)"
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
                NUM_SMS, 1, 1, threads=384,
            ) as (bx, _by, _bz):
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
                # Q load barriers (one per consumer): consumer issues
                # tma_copy for its own q_shared, waits via mbarrier.
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
                # All declared at kernel scope; only the owning WG
                # actually reads/writes each. T.alloc_var is per-thread,
                # value persists across persistent loop iterations.
                gi_kp = T.alloc_var("int32", init=0)   # WG0 K loads
                gi_vp = T.alloc_var("int32", init=0)   # WG0 V loads
                gi_kc1 = T.alloc_var("int32", init=0)  # WG1 K wgmmas
                gi_vc1 = T.alloc_var("int32", init=0)  # WG1 V wgmmas
                gi_kc2 = T.alloc_var("int32", init=0)  # WG2 K wgmmas
                gi_vc2 = T.alloc_var("int32", init=0)  # WG2 V wgmmas
                gi_q1 = T.alloc_var("int32", init=0)   # WG1 Q loads
                gi_q2 = T.alloc_var("int32", init=0)   # WG2 Q loads

                tx = T.get_thread_binding()

                # ===== WG0 (producer, tx < 128) =====
                if tx < 128:
                    T.dec_max_nreg(24)
                    for tile_b, tile_h, tile_m in T.Persistent(
                        [B, H, T.ceildiv(S, block_m)],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_h // groups
                        loop_range = T.ceildiv(S, block_n)

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            # Acquire K stage: wait for consumers to free it
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
                            # Load V[n-1] into v_smem[gi_vp%2]
                            if n_idx > 0:
                                T.barrier_wait(v_empty, (gi_vp + 1) % 2)
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
                    # Bootstrap: arrive on own scheduler barrier so first
                    # bar.sync(1, 256) has 128+128=256. ONCE per CTA,
                    # OUTSIDE the persistent loop.
                    T.call_extern("handle",
                                  "tl::barrier_arrive_named", 1, 256)
                    for tile_b, tile_h, tile_m in T.Persistent(
                        [B, H, T.ceildiv(S, block_m)],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        row_base = tile_m * block_m
                        loop_range = T.ceildiv(S, block_n)

                        # Per-WG Q load via TMA (faster than cp.async,
                        # no L1TEX stall). Issued by 1 thread, completion
                        # via mbarrier.
                        T.tma_copy(
                            q[tile_b, row_base:row_base + half_m,
                              tile_h, :],
                            q_shared_1, barrier=q_full_1)
                        T.barrier_arrive(q_full_1)
                        T.barrier_wait(q_full_1, gi_q1 % 2)
                        gi_q1 = gi_q1 + 1

                        # Per-tile state reset
                        T.clear(acc_o_1)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc1 % 2)
                            T.sync_threads(barrier_id=1, arrive_count=256)
                            # Non-causal: just clear acc_s_1
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
                                T.call_extern("handle",
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
                                T.call_extern("handle",
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
                        # Release the last V buffer (V[loop_range-1]).
                        # v2 single-tile didn't need this because the
                        # initial v_empty pre-credit covered the off-by-one,
                        # but persistent multi-tile MUST release it.
                        T.barrier_arrive(v_empty)
                        gi_vc1 = gi_vc1 + 1
                        # Output write for half 1
                        for i, j in T.Parallel(half_m, D):
                            acc_o_1[i, j] /= ls_1[i]
                        T.copy(acc_o_1, q_shared_1)
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=3, arrive_count=128)
                        T.copy(q_shared_1,
                               output[tile_b, row_base:row_base + half_m,
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
                    for tile_b, tile_h, tile_m in T.Persistent(
                        [B, H, T.ceildiv(S, block_m)],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        row_base = tile_m * block_m
                        loop_range = T.ceildiv(S, block_n)

                        # Per-WG Q load via TMA (own q_shared_2)
                        T.tma_copy(
                            q[tile_b,
                              row_base + half_m:row_base + block_m,
                              tile_h, :],
                            q_shared_2, barrier=q_full_2)
                        T.barrier_arrive(q_full_2)
                        T.barrier_wait(q_full_2, gi_q2 % 2)
                        gi_q2 = gi_q2 + 1

                        # Per-tile state reset
                        T.clear(acc_o_2)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc2 % 2)
                            T.sync_threads(barrier_id=2, arrive_count=256)
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
                                T.call_extern("handle",
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
                                T.call_extern("handle",
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
                        # Release the last V buffer (see WG1 above).
                        T.barrier_arrive(v_empty)
                        gi_vc2 = gi_vc2 + 1
                        # Output write for half 2
                        for i, j in T.Parallel(half_m, D):
                            acc_o_2[i, j] /= ls_2[i]
                        T.copy(acc_o_2, q_shared_2)
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        T.copy(q_shared_2,
                               output[tile_b,
                                      row_base + half_m:
                                      row_base + block_m, tile_h, :])
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
    kernel = build_fa3_v2_persistent(B, S, H, Hkv, D, is_causal)
    o, lse = kernel(block_m=128, block_n=128)(q, k, v)
    diff = (o.float() - o_ref.float()).abs().max().item()
    ok = diff < 0.5
    n_tiles = B * H * (S // 128)
    tag = (f"B={B} S={S} H={H} Hkv={Hkv} D={D} causal={int(is_causal)} "
           f"NUM_SMS={NUM_SMS} tiles={n_tiles}")
    print(f"  {tag}: diff={diff:.4f} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"=== v2 PERSISTENT (NUM_SMS={NUM_SMS}) ===")
    ok = True
    ok &= test(1, 256, 8, 4, 128, False)
    ok &= test(4, 512, 64, 4, 128, False)
    ok &= test(2, 2048, 32, 8, 128, False)
    ok &= test(4, 2048, 64, 4, 128, False)
    print(f"\n{'All passed!' if ok else 'SOME FAILED!'}")
