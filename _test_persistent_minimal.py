"""Minimal test: does T.Persistent compile at all in v2-shape kernels?
Strip everything except Q load and a single TMA-load + wgmma to see if
persistent + ws + pipelined nesting blows up TileLang lowering.
"""
import os
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

import tilelang
import tilelang.language as T

NUM_SMS = 132


def build():
    """Persistent + ws(0/1) + Pipelined + alloc_fragment — closer to v2."""
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )
    def func():
        @T.prim_func
        def main(A: T.Tensor((4, 2048, 8, 128), "float16"),
                 B: T.Tensor((4, 2048, 8, 128), "float16")):
            with T.Kernel(NUM_SMS, 1, 1, threads=256) as (bx, _, _):
                a_smem_0 = T.alloc_shared([64, 128], "float16")
                a_smem_1 = T.alloc_shared([64, 128], "float16")
                acc = T.alloc_fragment([64, 128], "float")
                full_bar = T.alloc_barrier(arrive_count=128)
                empty_bar = T.alloc_barrier(arrive_count=128)
                T.sync_threads()
                for tile_b, tile_h, tile_m in T.Persistent(
                    [4, 8, 16],
                    wave_size=NUM_SMS,
                    index=bx,
                    group_size=8,
                ):
                    with T.ws(1):
                        T.clear(acc)
                    for n in T.Pipelined(8, num_stages=0):
                        with T.ws(0):
                            T.barrier_wait(empty_bar, (n + 1) % 2)
                            if n % 2 == 0:
                                T.tma_copy(
                                    A[tile_b, tile_m * 128 + n * 16:tile_m * 128 + (n + 1) * 16,
                                      tile_h, :],
                                    a_smem_0, barrier=full_bar)
                            else:
                                T.tma_copy(
                                    A[tile_b, tile_m * 128 + n * 16:tile_m * 128 + (n + 1) * 16,
                                      tile_h, :],
                                    a_smem_1, barrier=full_bar)
                            T.barrier_arrive(full_bar)
                        with T.ws(1):
                            T.barrier_wait(full_bar, n % 2)
                            T.barrier_arrive(empty_bar)
                    with T.ws(1):
                        T.copy(acc, B[tile_b, tile_m * 128:(tile_m + 1) * 128, tile_h, :])
        return main
    return func


if __name__ == "__main__":
    print("Building minimal persistent + tma + barrier kernel...")
    import time
    t0 = time.time()
    k = build()()
    elapsed = time.time() - t0
    print(f"Compiled in {elapsed:.1f}s")
    print()
    src = k.get_kernel_source()
    print("Generated CUDA (first 80 lines):")
    for i, line in enumerate(src.splitlines()[:80]):
        print(f"  {i:3}: {line}")
