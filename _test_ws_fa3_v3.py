"""FA3-aligned IntraWGOverlap v3: persistent tile scheduling.

v3 adds the FA3 structural change we identified as the largest remaining
gap: the kernel launches NUM_SMS CTAs (one per SM) and each CTA loops
internally over multiple tiles via T.Persistent. Per-tile setup
(Q load, mbarrier wait, accumulator clear) is paid inside the loop, so
the GPU never goes back to the launch scheduler between tiles.

Differences from v2:
  - launch grid: (NUM_SMS, 1, 1) instead of (S/block_m, H, B)
  - body wrapped in `for (tile_b, tile_h, tile_m) in T.Persistent(...)`
  - bx/by/bz replaced with tile_m/tile_h/tile_b inside the loop

Mbarrier parity: with non-causal and even loop_range (e.g. S=2048,
block_n=64 → loop_range=32), parity naturally returns to 0 at the end
of each tile. Causal shapes may need an explicit reset; not handled
in this first cut.
"""

# Number of SMs on the target device. H200 = 132. Set to dev count at
# import time when imported by the bench script.
import os as _os
NUM_SMS = int(_os.environ.get("V3_NUM_SMS", "132"))
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
                NUM_SMS, 1, 1, threads=384,
            ) as (bx, _by_unused, _bz_unused):
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

                T.annotate_layout({
                    q_shared_1:
                        tilelang.layout.make_swizzled_layout(q_shared_1),
                    q_shared_2:
                        tilelang.layout.make_swizzled_layout(q_shared_2),
                })

                T.sync_threads()  # after barrier init

                # Persistent loop over tile space (B, H, ceildiv(S, block_m)).
                # group_size groups the last (M) dim for L2 reuse.
                # Each CTA processes ~ceil(B*H*M / NUM_SMS) tiles.
                for tile_b, tile_h, tile_m in T.Persistent(
                    [B, H, T.ceildiv(S, block_m)],
                    wave_size=NUM_SMS,
                    index=bx,
                    group_size=8,
                ):
                    head_kv = tile_h // groups
                    row_base = tile_m * block_m
                    loop_range = (
                        T.ceildiv((tile_m + 1) * block_m, block_n)
                        if is_causal else T.ceildiv(S, block_n))

                    T.copy(q[tile_b, row_base:row_base + half_m, tile_h, :],
                           q_shared_1)
                    T.copy(q[tile_b, row_base + half_m:row_base + block_m,
                             tile_h, :], q_shared_2)

                    T.sync_threads()  # after Q loads

                    with T.ws(1):
                        # Bootstrap: arrive on own scheduler barrier
                        # so first bar.sync(1, 256) has 128+128=256
                        T.call_extern("handle",
                                      "tl::barrier_arrive_named", 1, 256)
                        T.clear(acc_o_1)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))
                    with T.ws(2):
                        T.clear(acc_o_2)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))
    
                    # =============================================
                    # Main loop: FA3 IntraWGOverlap with 2-stage KV
                    # =============================================
                    # Each iter:
                    #   Producer: acquire k_empty → load K[n] → commit k_full
                    #             acquire v_empty → load V[n-1] → commit v_full
                    #   Consumer: wait k_full → QK[n] async
                    #             wait v_full → PV[n-1] async
                    #             wait<1> QK → release K (k_empty)
                    #             softmax
                    #             wait<0> PV → release V (v_empty)
    
                    for n_idx in T.Pipelined(loop_range, num_stages=0):
    
                        # -- WG0 (producer) --
                        with T.ws(0):
                            # Acquire K stage: wait for consumers to free it
                            T.barrier_wait(k_empty, (n_idx + 1) % 2)
                            # Load K[n] into k_smem[n%2]
                            if n_idx % 2 == 0:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:
                                      (n_idx + 1) * block_n,
                                      head_kv, :],
                                    k_smem_0, barrier=k_full)
                            else:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:
                                      (n_idx + 1) * block_n,
                                      head_kv, :],
                                    k_smem_1, barrier=k_full)
                            T.barrier_arrive(k_full)
    
                            # Load V[n-1] into v_smem[(n-1)%2]
                            if n_idx > 0:
                                T.barrier_wait(
                                    v_empty, n_idx % 2)
                                if (n_idx - 1) % 2 == 0:
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
    
                        # -- WG1 (consumer) --
                        with T.ws(1):
                            # Wait for K[n]
                            T.barrier_wait(k_full, n_idx % 2)
                            # Scheduler: wait for WG2's prev-iter arrive
                            T.sync_threads(barrier_id=1, arrive_count=256)
    
                            if is_causal:
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] = T.if_then_else(
                                        row_base + i
                                        >= n_idx * block_n + j,
                                        0, -T.infinity(accum_dtype))
                            else:
                                T.clear(acc_s_1)
    
                            if n_idx == 0:
                                # Prologue: QK[0] sync from k_smem_0
                                T.wgmma_gemm(
                                    q_shared_1, k_smem_0, acc_s_1,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow)
                                # Scheduler: signal WG2 (prologue)
                                T.call_extern("handle",
                                              "tl::barrier_arrive_named",
                                              2, 256)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(
                                    acc_s_1, num_regs=64)
                                # Release K stage 0
                                T.barrier_arrive(k_empty)
                                softmax_1(acc_s_1, sm_1, smp_1,
                                          ss_1, ssum_1, ls_1)
                                T.copy(acc_s_1, acc_s_cast_1)
                            else:
                                # Steady state: QK[n] from k_smem[n%2]
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
                                # Rescale O while QK runs
                                rescale_1(acc_o_1, ss_1)
                                # PV[n-1] from v_smem[(n-1)%2]
                                T.barrier_wait(
                                    v_full, (n_idx - 1) % 2)
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
                                # Scheduler: signal WG2 (arrive on barrier 2)
                                T.call_extern("handle",
                                              "tl::barrier_arrive_named",
                                              2, 256)
                                # wait<1>: QK done
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(
                                    acc_s_1, num_regs=64)
                                # Early K release
                                T.barrier_arrive(k_empty)
                                softmax_1(acc_s_1, sm_1, smp_1,
                                          ss_1, ssum_1, ls_1)
                                # PV must drain before we overwrite
                                # acc_s_cast_1 (PV's input registers).
                                # Swapping cast and wait<0> avoids C7513
                                # (compiler-inserted WG.DP) and conforms to
                                # the wgmma spec: do not modify in-flight
                                # input registers before the matching wait.
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(
                                    acc_o_1, num_regs=64)
                                T.barrier_arrive(v_empty)
                                T.copy(acc_s_1, acc_s_cast_1)
    
                        # -- WG2 (consumer) --
                        with T.ws(2):
                            T.barrier_wait(k_full, n_idx % 2)
                            T.sync_threads(barrier_id=2, arrive_count=256)
    
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
                                T.call_extern("handle",
                                              "tl::barrier_arrive_named",
                                              1, 256)  # signal WG1
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
                                T.barrier_wait(
                                    v_full, (n_idx - 1) % 2)
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
                                T.call_extern("handle",
                                              "tl::barrier_arrive_named",
                                              1, 256)  # signal WG1
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(
                                    acc_s_2, num_regs=64)
                                T.barrier_arrive(k_empty)
                                softmax_2(acc_s_2, sm_2, smp_2,
                                          ss_2, ssum_2, ls_2)
                                # See WG1 above: cast must wait for PV to
                                # drain (avoid C7513 / wgmma spec violation).
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(
                                    acc_o_2, num_regs=64)
                                T.barrier_arrive(v_empty)
                                T.copy(acc_s_2, acc_s_cast_2)
    
                    # ===== Epilogue: last PV[N-1] =====
                    with T.ws(0):
                        T.barrier_wait(v_empty, loop_range % 2)
                        if (loop_range - 1) % 2 == 0:
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
    
                    with T.ws(1):
                        rescale_1(acc_o_1, ss_1)
                        T.barrier_wait(
                            v_full, (loop_range - 1) % 2)
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
    
                    with T.ws(2):
                        rescale_2(acc_o_2, ss_2)
                        T.barrier_wait(
                            v_full, (loop_range - 1) % 2)
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
    
                    # ===== Write output =====
                    with T.ws(1):
                        for i, j in T.Parallel(half_m, D):
                            acc_o_1[i, j] /= ls_1[i]
                        T.copy(acc_o_1, q_shared_1)
                        # Fence: register->smem writes must be visible to
                        # async proxy (TMA store reads smem via async proxy)
                        T.fence_proxy_async()
                        # WG-local sync so all 128 WG1 threads have finished
                        # writing q_shared_1 before TMA store reads it
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
    
                    with T.ws(2):
                        for i, j in T.Parallel(half_m, D):
                            acc_o_2[i, j] /= ls_2[i]
                        T.copy(acc_o_2, q_shared_2)
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        T.copy(q_shared_2,
                               output[tile_b, row_base + half_m:
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
