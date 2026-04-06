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

## Why async WS is slow: ncu-driven analysis (2026-04-06)

Profiled with `ncu --set full` against FA3 on H200 GPU1 (locks released), config
`B=4 S=2048 H=64 Hkv=4 D=128 causal=False` (official throughput-fp16 shape).

### Speed-of-Light comparison

| Metric | v2-WS | FA3 | Note |
|---|---|---|---|
| Duration | 5.06 ms | 0.893 ms | 5.67× slower |
| Compute (SM) Throughput | 15.48 % | 79.20 % | FA3 keeps SMs 5× busier |
| SM Busy | **13.83 %** | **82.20 %** | v2 SMs idle 86 % of cycles |
| L2 Cache Throughput | **58.75 %** | **26.09 %** | v2 re-reads L2 2.25× more |
| DRAM Throughput | 8.82 % | 5.99 % | both low, not DRAM-bound |
| Achieved Occupancy | 18.52 % | 14.06 % | v2 is *higher* |
| Registers / Thread | 168 | 168 | identical |
| Dyn SMEM / Block | 160 KB | 211 KB | FA3 uses bigger `block_n` |
| Warp Cycles / Issued Inst | **22.29** | **6.02** | v2 stalls 3.7× more |
| Eligible Warps / Scheduler | 0.14 | 0.52 | v2 almost never ready |

Occupancy is not the problem — v2 has *more* resident warps than FA3 but
only 0.14 of them are eligible to issue per cycle.

### Stall reason breakdown (warp cycles per issued instruction)

| Stall reason | v2-WS | FA3 | Δ | Interpretation |
|---|---:|---:|---:|---|
| **long_scoreboard** | **11.55** | 0.25 | **+11.30** | Global / LSU / SMEM load dependency |
| **barrier** | **5.89** | 0.82 | **+5.07** | `bar.sync` (named barrier) |
| wait | 1.73 | 1.64 | +0.09 | **WGMMA fence / async wait — essentially identical** |
| lg_throttle | 1.29 | 0.00 | +1.29 | LSU pipe throttle (classic spill signature) |
| selected | 1.00 | 1.00 | 0 | Issue cycle itself |
| gmma | 0.24 | 0.35 | **−0.11** | v2 has *less* WGMMA issue pressure than FA3 |
| no_instruction | 0.25 | 0.06 | +0.19 | I-cache / branch |
| dispatch_stall | 0.01 | 0.76 | −0.75 | FA3 is more compute-bound |
| other | <0.10 each | <0.30 each | — | — |
| **total** | **22.30** | **6.03** | +16.27 | |

### What is NOT the bottleneck

**Fence / WGMMA overlap.** The `wait` stall (which includes `wait_wgmma<N>`
and memory fences) is essentially identical between v2 and FA3 (1.73 vs
1.64 cycles per issued inst), and `gmma` stall is actually *lower* for v2
(0.24 vs 0.35). IntraWGOverlap — issue QK → issue PV → `wait<1>` → softmax
→ `wait<0>` — is working as intended. Adding more fences, or removing the
existing ones, will not close the gap.

**Register budget.** Explicitly declaring FA3-style `set_max_nreg(32, 0)` on
the producer WG and `set_max_nreg(160, 1)` on each consumer WG is verified
in the generated CUDA (`warpgroup_reg_dealloc<32>` / `warpgroup_reg_alloc<160>`)
but has zero effect on runtime (129.7 → 129.5 TFLOPS). Both kernels end up
at 168 reg/thread regardless.

**Occupancy.** v2 already has higher achieved occupancy than FA3. Reducing
shared memory or register usage to get more CTAs/SM won't help — FA3 hits
65 % SOL with only 14 % occupancy because its resident warps stall less.

### What IS the bottleneck

### `ptxas -v` output (the smoking gun)

Recompiling `/tmp/ga-tvm-tmp/.../tvm_kernels.cu` with
`nvcc -Xptxas=-v,-warn-lmem-usage` exposes two critical compiler warnings
that are invisible from the ncu GPU profile:

```
ptxas info  : (C7507) Potential Performance Loss: 'setmaxnreg' ignored
              to maintain minimum register requirements.
ptxas info  : (C7518) Potential Performance Loss: wgmma.mma_async
              instructions are serialized due to program dependence on
              compiler-inserted WG.DP in divergent path in the function
              'main_kernel'
ptxas warn  : Local memory used for function 'main_kernel',
              size of stack frame: 1128 bytes
ptxas info  : Function properties for main_kernel
              1128 bytes stack frame, 3172 bytes spill stores,
              3380 bytes spill loads
ptxas info  : Used 168 registers, used 5 barriers,
              1128 bytes cumulative stack size, 1024 bytes smem
```

Compare to FA3's `sm_90a` fwd kernels (from
`cuobjdump --dump-resource-usage` of the flash-attention `.so`):

```
 Function _ZN...FlashAttnFwd...Sm90...:
   REG:168 STACK:0 SHARED:1024 LOCAL:0
```

Same register count (168), but FA3 has **STACK:0** (zero spill) while v2
has **STACK:1128** — 282 fp32 values per thread living in local memory.

#### C7507: `setmaxnreg` ignored

The explicit `T.set_max_nreg(160, 1)` annotation I added on the consumer
WGs (and `(32, 0)` on the producer) was **silently dropped by ptxas**.
The compiler determined that 168 registers were the minimum needed and
refused to lower the budget. This is also why explicit register-budget
tuning produced zero performance change: the annotation never took
effect.

#### C7518: WGMMA instructions serialized by compiler-inserted WG.DP

This is the more important finding. ptxas is inserting a
`WARPGROUP.DEPBAR` (WG.DP — wait-for-all-pending-WGMMA) into what it
labels a "divergent path" in `main_kernel`, **forcing every WGMMA to
complete before the next one can issue**. The async WGMMA pipeline the
kernel was designed around is being silently defeated by the compiler.

This explains the previously-confusing ncu data point: `gmma` stall is
0.24 for v2 vs 0.35 for FA3. I read this as "v2 has less WGMMA pressure",
but the real interpretation is "v2's WGMMAs never queue up because
ptxas inserted a depbar between each pair, so there is nothing for the
scheduler to stall on". The `wait_wgmma<1>` / `wait<0>` sequence in the
generated code is in place, but ptxas degrades it to effectively `wait<0>`
every time by inserting the depbar ahead of the software wait.

The likely cause of the "divergent path" label is TileLang lowering
`T.ws(N)` to plain `if (128 <= threadIdx.x && threadIdx.x < 256) { ... }`
conditionals. ptxas does not recognize this as a warpgroup-uniform split
and conservatively inserts the depbar. FA3 avoids this by computing
`canonical_warp_group_idx_async()` (CUTLASS helper) and using a
warpgroup-uniform branch that ptxas can prove is uniform across the
warpgroup.

**1. `long_scoreboard` 11.55 vs 0.25 (46×)** — confirmed to be **register
spilling**. The ncu raw metrics are conclusive:

| Metric | v2-WS | FA3 | Ratio |
|---|---:|---:|---:|
| `sass__inst_executed_register_spilling` | **148,406,272** | 32,768 | 4,529× |
| `sass__inst_executed_local_loads` | 71,417,856 | 16,384 | 4,359× |
| `sass__inst_executed_local_stores` | 76,988,416 | 16,384 | 4,698× |
| L1 local-load sectors | 374,513,132 | 53,248 | 7,034× |
| **L1 local-load miss → L2** | **141,921,244** | **0** | ∞ |
| L1 local-store sectors | 396,235,660 | 41,356 | 9,581× |

v2 executes **148 million spill instructions per launch**, of which
**142 million local loads miss L1 and go all the way to L2** (37.9 % miss
rate). FA3's 32 K spill count is essentially zero — likely just kernel
prologue stack setup. This is a 4,500× gap, and it chains directly to
every other symptom:

- `long_scoreboard` = 11.55 → waiting on spill reloads returning from L2
- `lg_throttle` = 1.29 → LSU pipe saturated by spill traffic
- `L2 Throughput` = 58.75 % vs 26.09 % → L2 bandwidth is serving spills,
  not K/V data (not the cluster/multicast difference I initially suspected)
- `SM Busy` = 13.83 % → SMs are idle waiting on local memory

`set_max_nreg(160, 1)` had no effect because the compiler was already
pinned at 168 regs/thread and the live set was already far too large to
fit — the allocator was spilling before and after the annotation.

**2. `barrier` 5.89 vs 0.82 (7×).** Most likely the softmax reductions,
which currently emit `AllReduce<..., tl::NamedBarrier<128>>` — two
reductions (max + sum) per half, two consumer WGs, 16 iterations ≈ 64
inter-warp `bar.sync`s per CTA. FA3's softmax uses warp-level
`__shfl_xor_sync` plus at most one inter-warp reduce, avoiding the repeated
named-barrier cost.

**3. L2 traffic.** Initially I suspected the missing `ClusterM = 2` TMA
multicast (FA3 uses it for `kHeadDim >= 128, fp16, !causal, seqlen_q/blockM
even`, all matching our config — see `flash_fwd_launch_template.h:212-218`).
But the raw L2 metrics show `memory_l2_theoretical_sectors_local` is
**770 M for v2 vs 94 K for FA3** — the L2 pressure is dominated by spill
reload traffic, not by K/V re-fetches. Cluster multicast is still a
worthwhile optimization, but it is downstream of the spill fix: until the
spill is resolved, any K/V bandwidth win will be swamped by spill traffic.

### Instruction-count comparison

From ncu raw metrics:

| Metric | v2-WS | FA3 | v2/FA3 |
|---|---:|---:|---:|
| Total executed instructions (`smsp__inst_executed.sum`) | 512.2 M | 251.1 M | 2.04× |
| of which, register spilling | 148.4 M | 32.8 K | 4,529× |
| of which, local loads | 71.4 M | 16.4 K | 4,359× |
| of which, local stores | 77.0 M | 16.4 K | 4,698× |
| Inst/cycle elapsed | 68.3 | 189.6 | 0.36× |
| `pipe_lsu` active (% of peak) | **15.79 %** | **1.36 %** | **11.6×** |
| `pipe_tensor_gmma` active (% of peak) | 0.87 % | 4.44 % | 0.19× |
| `pipe_fma` fp32 active (% of peak) | 3.62 % | 15.50 % | 0.23× |
| `pipe_alu` int active (% of peak) | 6.56 % | 19.97 % | 0.33× |
| `pipe_xu` SFU active (% of peak) | 6.25 % | 20.05 % | 0.31× |
| TMA load instructions | 278.5 K | 204.8 K | 1.36× |
| TMA store instructions | 16.4 K | 8.2 K | 2.00× |
| TMA bytes read (GB) | 4.43 | 4.56 | 0.97× |

Key cross-checks:

- Even after subtracting the 148 M spill instructions, v2 still executes
  364 M useful instructions vs FA3's 251 M — v2 has 1.45× more non-spill
  work. This likely comes from extra address-computation instructions
  around the spill traffic and from running at lower per-instruction
  efficiency (IPC 0.53 vs 1.49).
- **LSU active rate 11.6× higher** for v2, directly reflecting the spill
  traffic. FA3's LSU is at 1.36 % of peak — essentially idle.
- **Tensor GMMA rate 5.1× lower** for v2 despite serializing the same
  amount of WGMMA work. This is C7518 at work: v2's WGMMAs cannot issue
  back-to-back, so the tensor-core pipe utilization collapses.
- **TMA bytes read are within 3 %**. Raw TMA traffic is NOT a differentiator;
  FA3's `ClusterM = 2` multicast does not show up as a reduction in this
  metric. TMA is not the bottleneck for either kernel (both `pipe_tma_active`
  < 0.3 %).

### Next steps (in expected payoff order)

1. **Fix C7518: eliminate the compiler-inserted WG.DP.** Rewrite the
   TileLang consumer/producer split so ptxas recognizes it as
   warpgroup-uniform instead of thread-divergent. Two concrete options:
   - Emit the WG selection via a uniform predicate
     (`warpgroup_uniform()` / `__shfl_sync` broadcast of
     `threadIdx.x / 128`) before the `if`, so ptxas sees a uniform branch.
   - Or restructure so WGMMA calls are lifted out of the conditional
     entirely (each WG enters the same code, with data differences
     encoded in SMEM offsets). This mirrors how FA3's mainloop works.

   Until this is fixed, the software IntraWGOverlap design has no effect
   — ptxas is silently degrading every `wait<1>` to `wait<0>`.

2. **Fix the register spill (1128 bytes stack frame → 0).** The compiler
   is at its 168-register limit and spilling 3172 B of stores and
   3380 B of loads per thread (statically; 148 M dynamically). Live
   candidates by inspection of `_test_ws_fa3_v2.py`:
   - `acc_o_1/2` (16 float4 each = 256 B), kept live across the whole
     KV loop.
   - `acc_s_1/2` (64 floats = 256 B) — QK accumulator.
   - `acc_s_cast_1/2` (32 halves = 64 B) — staged for the PV gemm; live
     across the softmax → cast → PV issue window.
   - Softmax scratch `smp_*`, `sm_*_clear`, `ss_*`, `ssum_*` — small but
     kept alongside.

   Highest-leverage fix: eliminate the `acc_s_cast` staging step and feed
   the PV gemm directly from `acc_s` in register-source form, matching
   FA3's `MmaPV_is_RS` path. This removes a whole second copy of the P
   matrix from the live set.

3. **Replace `NamedBarrier<128>` softmax reductions with warp-shuffle +
   one `bar.sync`.** Independently of (1) and (2). Should cut `barrier`
   stall from ~5.9 to ~1 cycles/inst — estimated 15–20 % speedup.

4. **Enable TMA multicast / cluster launch** (`ClusterM = 2`). Worthwhile
   only after (1)–(3) are fixed — today's L2 traffic is dominated by
   spill reloads, not K/V fetches, and the tensor pipe is capped by
   WGMMA serialization.

### Raw reports

`/tmp/v2_ws_prof.ncu-rep`, `/tmp/fa3_prof.ncu-rep` on the H200 host,
reproducible via `_ncu_v2.py` / `_ncu_fa3.py` with
`ncu --set full --nvtx --nvtx-include "<range>/"`.
