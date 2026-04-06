# Async WS IntraWGOverlap: root cause and fix

Relates to issue #9.

## TL;DR

The async WS IntraWGOverlap GQA forward kernel (`_test_ws_fa3_v2.py`) was
failing 60–100 % of runs depending on configuration. The root cause was a
missing **smem visibility fence in the epilogue output path**, NOT a WGMMA
pipeline problem, a named-barrier scheduler issue, or an NVCC optimisation
bug.

Fix (three lines added to each consumer WG's epilogue):

```python
T.copy(acc_o_1, q_shared_1)          # register → smem (generic proxy)
T.fence_proxy_async()                 # 👈 make smem writes visible to async proxy
T.sync_threads(barrier_id=3,          # 👈 WG-local sync (bar.sync 3, 128)
               arrive_count=128)
T.copy(q_shared_1,                    # smem → global via TMA (async proxy)
       output[bz, row_base:row_base + half_m, by, :])
```

With the fix:

- 8 / 8 configs pass (30 runs each) for H ∈ {8, 16, 32, 64},
  S ∈ {256, 512, 1024, 2048}, causal ∈ {True, False}.
- `TL_DISABLE_THREAD_STORAGE_SYNC=True` stays enabled.
- Main loop has NO `__syncthreads()` — the WS pattern is preserved:
  "init once, then mbarriers only".

## Why this was the problem

The epilogue writes the per-row accumulator `acc_o` (registers) to
`q_shared` (SMEM) and then does a TMA store from `q_shared` to the global
output. TMA reads SMEM through the **async proxy**, while register→SMEM
stores happen through the **generic proxy**. Without a
`fence.proxy.async.shared::cta` between them, the TMA can observe stale
SMEM. The WG-local `bar.sync N, 128` is additionally needed so that *all*
128 threads in a WG have finished the register→SMEM copy before any thread
issues TMA store.

In the *sync* version of this kernel (`_test_ws_fa3_v2_sync.py`), TileLang's
`StorageRewrite` pass automatically inserts `__syncthreads()` + implicit
fence at this boundary, which is why the sync path worked. Flipping
`TL_DISABLE_THREAD_STORAGE_SYNC=True` removes that automatic insertion, so
the fence has to be explicit.

## Things that looked like the bug but were NOT

During debugging I chased several red herrings:

1. **WGMMA pipeline WAR race on `acc_s_cast`.** Hypothesis: softmax writes
   `acc_s_cast` before the async PV WGMMA has finished reading it. **Wrong**
   — WGMMA source operands are captured at issue time (between
   `warpgroup_arrive` and `warpgroup_commit_batch`). After `commit_batch`,
   software may freely modify the source registers; the `fence_operand`
   after `commit_batch` is the release. Moving `T.copy(acc_s, acc_s_cast)`
   after `T.wait_wgmma(0)` actually made failure rate *worse* (100 %).

2. **`fence.proxy.async.shared::cta` missing inside the WGMMA macro.**
   Hypothesis: `warpgroup_arrive` followed by WGMMA needs a proxy fence for
   pipeline depth ≥ 2. **Wrong** — FA3's `flash::gemm` in
   `flash-attention/hopper/utils.h:270-321` does not emit `fence_proxy_async`
   anywhere in the gemm path, and the stock `wgmma_macro_generator.py`
   matches that pattern. Adding the fence did nothing to the failure rate.

3. **Named-barrier ID conflict between scheduler and AllReduce.** Hypothesis:
   softmax AllReduce uses `NamedBarrier<128>::sync<phase=1>()` which maps to
   `bar.sync 1, 128`, clashing with the scheduler's `bar.arrive 1, 256` /
   `bar.sync 1, 256` on barrier ID 1. **Wrong** — AllReduce with
   `threads=4, scale=1` uses `__shfl_xor_sync` only, the
   `NamedBarrier<128>` template parameter is never instantiated because
   every level has `offset < 32`.

4. **NVCC emits different binaries for different H values.** The issue
   notes claimed H ≥ 32 failed 100 % deterministically because NVCC produced
   different code depending on head count. In reality, H ≥ 32 failed for the
   same reason H = 8 failed — the epilogue race — but the race happened to
   fire every time at that scale, whereas for H = 8 it was intermittent
   (60–90 % rather than 100 %).

5. **`T.Pipelined(..., num_stages=0)` vs `T.serial(...)`.** Confirmed to
   generate byte-identical CUDA. Not the cause.

## Third-party modifications

Exactly **one** addition was required to the tilelang install tree. All
other changes I tried (and have since reverted) were unnecessary.

### `tilelang/src/tl_templates/cuda/barrier.h`

Added a helper matching PTX `bar.arrive` (NOT `mbarrier.arrive`) at the end
of the `tl` namespace:

```cpp
// Named barrier arrive (PTX bar.arrive, NOT mbarrier.arrive)
TL_DEVICE void barrier_arrive_named(int barrier_id, int thread_count) {
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(thread_count));
}
```

This is needed so the kernel can emit the named-barrier arrive half of the
FA3 `warp_scheduler_barrier_arrive()` protocol. TileLang exposes
`T.sync_threads(barrier_id=N, arrive_count=T)` which lowers to `bar.sync N,
T`, but no counterpart for `bar.arrive`. The kernel uses
`T.call_extern("handle", "tl::barrier_arrive_named", ...)` to reach this
helper.

### Files that are back to stock (changes reverted)

For the record, I temporarily modified and later reverted:

- `tilelang/intrinsics/wgmma_macro_generator.py` — tried moving
  `warpgroup_fence_operand` outside the `if wg_wait >= 0` block, and tried
  adding `T.fence_proxy_async()` after `T.warpgroup_arrive()`. Neither
  change affected the failure rate; the stock behaviour is correct and
  matches FA3.
- `tilelang/src/tl_templates/cuda/barrier.h::fence_proxy_async` — tried
  adding `"memory"` clobber. No effect.
- `tilelang/3rdparty/cutlass/include/cutlass/arch/barrier.h` — tried
  adding `asm volatile("" ::: "memory")` after `Barrier::wait`. No effect.

**Current state:** only the `barrier_arrive_named` helper addition remains.
Everything else in tilelang and cutlass is stock.

## Kernel-level changes in `_test_ws_fa3_v2.py`

Diff against the pre-fix version:

```python
# ===== Write output =====
with T.ws(1):
    for i, j in T.Parallel(half_m, D):
        acc_o_1[i, j] /= ls_1[i]
    T.copy(acc_o_1, q_shared_1)
+   # Fence: register->smem writes must be visible to the async proxy
+   # (TMA store reads smem via the async proxy).
+   T.fence_proxy_async()
+   # WG-local sync so all 128 WG1 threads have finished writing
+   # q_shared_1 before TMA store reads it. Barrier id 3 is private
+   # to WG1 (WG2 uses id 4).
+   T.sync_threads(barrier_id=3, arrive_count=128)
    T.copy(q_shared_1,
           output[bz, row_base:row_base + half_m, by, :])
    ...

with T.ws(2):
    for i, j in T.Parallel(half_m, D):
        acc_o_2[i, j] /= ls_2[i]
    T.copy(acc_o_2, q_shared_2)
+   T.fence_proxy_async()
+   T.sync_threads(barrier_id=4, arrive_count=128)
    T.copy(q_shared_2,
           output[bz, row_base + half_m:row_base + block_m, by, :])
    ...
```

## Correctness results

All runs 30× on an H100 SXM with the fix:

| B | S | H | Hkv | Causal | Result |
|---|---|---|-----|--------|--------|
| 1 | 256 | 8 | 4 | No | 0/30 |
| 4 | 512 | 8 | 2 | No | 0/30 |
| 4 | 512 | 16 | 4 | No | 0/30 |
| 4 | 512 | 32 | 4 | No | 0/30 |
| 4 | 512 | 64 | 4 | No | 0/30 |
| 4 | 512 | 64 | 4 | Yes | 0/30 |
| 1 | 1024 | 32 | 8 | Yes | 0/30 |
| 2 | 2048 | 32 | 8 | Yes | 0/30 |

## Performance

H100 SXM, block_m=block_n=128:

| Config | async-WS v2 | sync v2 | async / sync |
|---|---|---|---|
| B=4 S=8192 H=64 Hkv=8 causal | 114.0 TFLOPS | 130.9 TFLOPS | 87 % |
| B=4 S=8192 H=64 Hkv=8 non-causal | 117.8 TFLOPS | 134.2 TFLOPS | 88 % |
| B=1 S=16384 H=32 Hkv=8 causal | 89.1 TFLOPS | 127.1 TFLOPS | 70 % |

The async WS variant is 12–30 % **slower** than the sync variant on this
architecture. Correctness is restored, but the IntraWGOverlap design has
not yet translated into a performance win — see the next section.
