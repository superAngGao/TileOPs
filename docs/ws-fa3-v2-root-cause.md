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

## Path A result: shfl-broadcast WG selection (C7518 fix)

End-to-end validation of "Next step 1" above. Implemented as a regex
post-process on the TileLang-generated CUDA, hooked in via a monkey-patch
of `tilelang_callback_cuda_compile` (so the TileLang adapter, host
wrapper, TMA descriptors, and launch path stay untouched).

The transform replaces three forms of WG-selection branch:

| Original                                                  | Replacement     | Count |
|-----------------------------------------------------------|-----------------|-------|
| `(128 <= ((int)threadIdx.x)) && (((int)threadIdx.x) < 256)` | `(__wg_id == 1)` | 4 |
| `(256 <= ((int)threadIdx.x))`                             | `(__wg_id == 2)` | 4 |
| `(((int)threadIdx.x) < 128)`                              | `(__wg_id == 0)` | 3 |

with `int __wg_id = __shfl_sync(0xffffffff, ((int)threadIdx.x) / 128, 0);`
inserted at the top of `main_kernel`, before any divergent op (so warps
are still converged at the shuffle point).

The `__shfl_sync` output is the only construct ptxas reliably tags as
warp-uniform. `((int)tid / 128) == N` does **not** work — confirmed by an
explicit divide-form post-process; ptxas still treats the result as
divergent and inserts WG.DP.

### Result on H200 (B=4 S=2048 H=64 Hkv=4 D=128, non-causal)

| Metric                          | Baseline (range branch) | Patched (shfl)        | Δ                        |
|---------------------------------|-------------------------|-----------------------|--------------------------|
| Latency                         | 4.227 ± 0.001 ms        | 3.887 ± 0.003 ms      | -8.0 %                   |
| TFLOPS                          | 130.0                   | 141.4                 | **+11.4 (+8.8 %)**       |
| Max diff vs torch fp32 ref      | 0.0002                  | 0.0002                | unchanged                |
| Stack frame                     | 1128 B                  | 1120 B                | -8 B (just `__wg_id`)    |
| Spill stores / loads            | 3172 / 3380             | 3164 / 3372           | -8 / -8                  |
| Registers                       | 168                     | 168                   | unchanged                |
| C7518 (divergent path WG.DP)    | present                 | **gone**              | resolved                 |
| C7507 (`set_max_nreg` ignored)  | present                 | present               | unchanged                |
| C7513 (non-WGMMA → wgmma_rs)    | hidden                  | **newly visible**     | exposed by removing 7518 |

Reproduced 3 times, jitter ±0.2 TFLOPS. Reproducer:
`_bench_ws_shfl.py` — runs ablation in one process via FFI re-registration
of `tilelang_callback_cuda_compile`.

### What this proves and what it disproves

**Proves**: C7518 is real and costly. Even though removing it does not
free a single spilled byte (per-thread live set is unchanged), the
WGMMA pipeline parallelism it had been silently destroying is worth
~9 % on this shape. The IntraWGOverlap design was effectively a no-op
before this fix.

**Disproves**: the hypothesis that branch-uniformity analysis would let
ptxas separate `_1` / `_2` fragment live ranges. Stack frame moved by
exactly 8 bytes (the new `__wg_id` register itself), which means ptxas'
per-thread register allocator does not use branch uniformity to prune
function-scope local arrays — it tracks them at function scope
regardless of which branches use them. This was a wrong intuition;
spill is unrelated to C7518 / branch form.

### What still bottlenecks v2 after C7518 is fixed

C7513 is now the visible WGMMA-stalling warning. Its mechanism is
direct: between `warpgroup_arrive` and `commit_batch` there are non-WGMMA
instructions (the `F2FP` cast that stages `acc_s` → `acc_s_cast`) that
write registers consumed by the very next WGMMA (PV `wgmma_rs`). ptxas
must insert a `warpgroup.wait` to make those writes visible before
issuing the dependent WGMMA. This collapses QK/PV overlap on the PV
side, independently of branch divergence.

Both C7513 and the 1120-byte spill point at the same structural fix
(still item 2 in the next-steps list above, now elevated to highest
priority): **eliminate the `acc_s_cast` staging buffer** and feed PV
directly from `acc_s` (cast on the wgmma input). This:

- removes the F2FP → wgmma_rs dependency chain → resolves C7513
- removes 256 B per warpgroup of fragment storage → should let 168 regs
  hold the live set without spill → resolves C7507 and the spill
- aligns with FA3's `MmaPV_is_RS` path

The remaining 3-4× gap to FA3 is expected to come almost entirely from
this single change. Path A's +9 % is the ceiling of what fixing only
divergence can buy.

### Productionizing path A

Currently the shfl rewrite lives in a monkey-patch in `_bench_ws_shfl.py`.
Two options to make v2 inherit it on the regular registration path:

1. Move the FFI override into the v2 op `__init__.py` so any importer of
   v2 sees the patch. Cheap, but global side effect on the FFI registry.
2. Fix it properly in TileLang C++: make `WarpSpecialize` lower the WG
   condition to `__shfl_sync(...) == N` instead of
   `(N*128 <= tid) && (tid < (N+1)*128)`. Permanent and benefits every
   WS kernel, but needs a TileLang rebuild.

Recommendation: defer productionizing path A until path C is in. If C
brings v2 into the FA3 ballpark, we then go back and fix the lowering
in TileLang the right way.

## Path C-1 result: reorder F2FP cast after wait_wgmma(0)

End-to-end validation that the C7513 warning identifies a real wgmma
spec violation in v2's source, not just a compiler quirk.

### The bug

In v2's WG1/WG2 consumer steady-state body, the original ordering was:

```python
T.wait_wgmma(1)                          # QK done (B1 drained)
softmax(acc_s)
T.copy(acc_s, acc_s_cast)                # ★ F2FP writes acc_s_cast
T.wait_wgmma(0)                          # PV (B2) drains here
T.barrier_arrive(v_empty)
```

`acc_s_cast` is PV's input register (PV is `wgmma_rs`). PV (B2) was
committed earlier in the same iteration but is still in flight at the
point of the cast. NVIDIA WGMMA spec is explicit:

> Until the corresponding `wait_group`, the input registers of in-flight
> wgmma operations should not be modified by other instructions.

ptxas correctly inserts `WG.DP` (warpgroup dependency barrier) to enforce
this — it cannot let the F2FP writes proceed before B2 drains, because
the hardware specification says those registers belong to B2 until then.
The diagnostic surfaces as C7513 (data dependency between non-WGMMA
write and in-flight WGMMA input).

### The fix

Swap the order so the cast happens **after** PV drains:

```python
T.wait_wgmma(1)                          # QK done
softmax(acc_s)
T.wait_wgmma(0)                          # PV done — registers free
T.barrier_arrive(v_empty)
T.copy(acc_s, acc_s_cast)                # safe to overwrite now
```

Two call sites (`_test_ws_fa3_v2.py` WG1 line 273 and WG2 line 341).
Total source diff: ~6 lines (including comments).

### Result on H200 (B=4 S=2048 H=64 Hkv=4 D=128, non-causal)

| Configuration                  | TFLOPS         | C7507 | C7518 | C7513 |
|-------------------------------|----------------|-------|-------|-------|
| Baseline (no fixes)            | 130.0          | y     | y     | y     |
| Path A only (shfl, hack)       | 141.4          | y     | n     | y (newly visible) |
| **Path C-1 only (cast reorder)** | **139.7 ± 0.05** | **y** | **y** | **n** |
| C-1 + Path A combined          | 141.1 ± 0.05   | y     | n     | n     |

3 runs, jitter ±0.05 TFLOPS. Correctness: PASS on all 5 v2 test shapes
(diff ≤ 0.002 vs torch fp32 reference).

### Interpretation: C7518 and C7513 are the same wgmma stalls, two views

The four-cell ablation table reveals a non-obvious fact: **fixing either
the divergent-path issue (Path A) or the data-dependency issue (C-1)
yields almost the same speedup, and stacking them adds only ~1 % more**.
They are not orthogonal bottlenecks.

The mechanism: ptxas inserts a single `WG.DP` per affected wgmma. It
attributes that barrier to whichever reason it finds first when emitting
the diagnostic. Removing one reason exposes the other in the warning
text, but the underlying barrier insertion logic operates on the same
set of wgmmas. Once both reasons are removed for a given wgmma, the
barrier is gone — but if either reason still applies, ptxas inserts the
barrier and reports whichever reason it noticed.

So **Path A and C-1 are alternative fixes for the same set of stalls**,
not complementary. C-1 wins on every dimension that matters:

1. **Production-quality**: clean source-code change vs. monkey-patched
   regex post-process on generated CUDA.
2. **Spec correctness**: it fixes a real WGMMA spec violation. v2's
   original code happened to produce correct output today only because
   ptxas was conservatively inserting the barrier on our behalf. Without
   the barrier (e.g. if a future ptxas got more aggressive), the cast
   could race with the in-flight wgmma's input read.
3. **Reviewable**: ~6 lines, two call sites, with a comment explaining
   why. The reorder cost (losing cast/PV overlap) is at most a few tens
   of cycles per iteration vs. PV's thousands of cycles — negligible.

### Path A retired

Path A's monkey-patch is no longer needed for production. It remains in
`_bench_ws_shfl.py` purely as a research/diagnostic tool — it lets us
ablate the C7518 dimension specifically, which was useful for proving
the C7518/C7513 equivalence above. The TileLang C++ fix to lower
`T.ws(N)` to `__shfl_sync(...) == N` is still potentially worth doing
(it benefits any other WS kernel that hits the spec issue or has
genuinely different spill characteristics), but is no longer on the v2
critical path.

### What still bottlenecks v2 after C-1

C7507 remains, and the spill numbers haven't moved at all:

- 1128 B stack frame
- 3172 spill stores / 3380 spill loads (statically)
- 148 M dynamic spill loads (from earlier ncu profile)
- LSU 11.6× FA3's

The remaining 3-4× gap to FA3 (130 → 460-500 TFLOPS) is now attributable
almost entirely to spill. Path A and C-1 combined buy ~9 % by fixing the
WGMMA pipelining issue; the spill needs a structurally different attack:

1. **PTX-level inspection** of `main_kernel.ptx` to identify which
   fragment contributes the most spill traffic — register pressure is
   not uniform across the fragments.
2. Possible levers if the worst offender is identified:
   - Move that fragment to shared memory (loses some async, gains regs).
   - Reduce `block_n` from 128 → 64 (halves all fragment sizes; may also
     halve TFLOPS — needs measurement).
   - Restructure so the fragment's live range is shorter.

Option 1 (PTX inspection) is the next concrete step.

## Path C-2a result: wgmma descriptor by-value (+12.9 % on top of C-1)

End-to-end validation that **the largest single source of spill traffic
in v2 is not user data — it is TileLang's wgmma descriptor template
forcing GmmaDescriptor to be stack-resident**.

### SASS-level spill breakdown (after C-1, before C-2a)

Compiled with `-lineinfo`, disassembled via `nvdisasm -c -g`, then
attributed each LDL/STL to its nearest source-line annotation:

| Source category | LDL | STL | Share of LDL |
|---|---|---|---|
| `tl_templates/cuda/common.h` (descriptor bitfield ops) | **260** | 231 | **42.8 %** |
| `_test_ws_fa3_v2.py` (user code: acc_o, acc_s, acc_s_cast spills) | 198 | 228 | 32.6 % |
| `cute/arch/mma_sm90_gmma.hpp` (wgmma SASS expansion glue) | 88 | 38 | 14.5 % |
| `cutlass/arch/barrier.h` (mbarrier ops) | 61 | 42 | 10.0 % |
| **Total** | **607** | **539** | 100 % |

The static byte counts cross-check with the ptxas summary:
`23×4 + ... = 3172` for stores, `64×4 + 305×4 + 238×8 = 3380` for loads.

The single biggest line attribution is `common.h:546`
(`descriptor.bitfield.stride_byte_offset_ = stride_byte_offset`), with
**151 LDLs** alone — that one bitfield write is generating ~25 % of all
spill reloads in the kernel.

The user fragment spills are dominated by acc_o (lines 248 and 559 in
the lowered CUDA — the rescale and epilogue normalize loops respectively
read all 64 floats sequentially), **not by acc_s_cast as initially
guessed**. acc_s_cast is fully stack-resident (all 128 of its halves
stored via STL.S16 in the `[R1+0x1ec..0x2e8]` window) but the dynamic
load count is small because PV reads it through `wgmma_rs` register
pointers, not via per-element LDLs.

### The mechanism

`tl::initialize_wgmma_descriptor` in `tl_templates/cuda/common.h:537-547`
takes the descriptor by reference:

```cpp
template <int layout_type = 0, int leading_byte_offset = 0,
          int stride_byte_offset = 0, typename T>
TL_DEVICE void initialize_wgmma_descriptor(GmmaDescriptor &descriptor,
                                           T *start_address) {
  descriptor.bitfield.start_address_ =
      cute::cast_smem_ptr_to_uint(start_address) >> 4;
  descriptor.bitfield.layout_type_ = layout_type;
  descriptor.bitfield.base_offset_ = 0;
  descriptor.bitfield.leading_byte_offset_ = leading_byte_offset;
  descriptor.bitfield.stride_byte_offset_ = stride_byte_offset;
}
```

The `&descriptor` parameter forces ptxas to treat the caller-side
`GmmaDescriptor desc_a_1, desc_b_1, ...` locals as address-taken, which
prevents register promotion. Each of the five bitfield writes becomes a
partial-word stack RMW. With 18 descriptor variables in v2's main_kernel,
that's ~250 spill ops attributed to one template function.

`GmmaDescriptor` itself is a CUTLASS union with implicit `uint64_t`
decay, so the storage is exactly 2 32-bit registers per descriptor when
register-resident. ptxas's failure to promote it is purely a consequence
of the by-reference signature.

### The fix

Inject a return-by-value helper at the top of the lowered CUDA, then
rewrite each `tl::initialize_wgmma_descriptor<L,LBO,SBO>(name, ptr);`
into `name = make_wgmma_descriptor_v<L,LBO,SBO>(ptr);`:

```cpp
template <int __LT = 0, int __LBO = 0, int __SBO = 0, typename __T>
__device__ __forceinline__ tl::GmmaDescriptor
make_wgmma_descriptor_v(__T *__addr) {
    tl::GmmaDescriptor __d;
    uint64_t __a14 = (cute::cast_smem_ptr_to_uint(__addr) >> 4) & 0x3fffull;
    __d.desc_ = __a14
              | (uint64_t(__LBO & 0x3fff) << 16)
              | (uint64_t(__SBO & 0x3fff) << 32)
              | ((uint64_t)(__LT & 0x3) << 62);
    return __d;
}
```

Three things are different from the original:

1. **No reference parameter** — local `__d` is not address-taken, so
   ptxas can SSA-promote it.
2. **Single 64-bit write to `desc_`** (the union's underlying field)
   instead of five bitfield RMWs. All template constants get
   constant-folded into the OR expression at compile time.
3. **`__forceinline__`** so the temporary `__d` becomes an SSA value in
   the caller after inlining.

The post-process is in `_bench_ws_shfl.py::desc_rewrite()`. For v2 it
matches **18 call sites**.

### Result on H200 (B=4 S=2048 H=64 Hkv=4 D=128, non-causal)

Four-cell ablation, 3 runs, jitter ±0.3 TFLOPS:

| Configuration | TFLOPS | Δ vs C-1 | Speedup |
|---|---|---|---|
| **A**: C-1 source only (committed) | 140.2 | — | 1.000 |
| **B**: + shfl WG-id (path A) | 141.1 | +0.9 | 1.006 |
| **C**: + descriptor by-value (path C-2a, no shfl) | **158.2** | **+18.0** | **1.129** |
| **D**: + shfl + desc by-value | 157.2 | +17.0 | 1.121 |

ptxas summary for C-2a:

| | C-1 only | + desc by-value | Δ |
|---|---|---|---|
| Stack frame | 1128 B | **880 B** | **−22 %** |
| Spill stores | 3172 | 2948 | −7 % |
| Spill loads | 3380 | **3668** | **+8.5 %** |
| Registers | 168 | 168 | — |

Correctness: PASS on all 5 v2 test shapes (max diff ≤ 0.002), validated
via `_test_correctness_descv.py`.

### Why the kernel runs faster despite ~equal total spill bytes

**This is the load-bearing insight from path C-2a.** Naive intuition
says: if static spill bytes are roughly conserved (6552 → 6616 total
LDL+STL bytes), TFLOPS should not change. The actual measurement is
+12.9 %.

The reason is **where in the SASS the spill operations land**.

Original v2 (C-1 only): each `wgmma_ss` / `wgmma_rs` issue stalls on the
LDL→arithmetic→wgmma chain, because the descriptor needs to be loaded
from stack into a register pair *right at issue time*. The descriptor
LDL is on the wgmma issue critical path. With 18 descriptors × 8 ki
unrolls × many loop iters, this is the dominant runtime cost.

After C-2a: descriptors live in registers across their entire lifetime
(from the assignment to the last wgmma read). wgmmas issue back-to-back
without stalls. The new spill load traffic is on `acc_o` (the rescale
and epilogue normalize loops), which is sequential fp32 work with no
data dependency on wgmma issue. Those LDLs are hidden inside the time
the WGMMA pipe is busy with previous batches.

**Static spill bytes are not a useful TFLOPS predictor on Hopper. What
matters is whether the spill operations land on or off the WGMMA issue
critical path.**

This also explains why we cannot simply chase "minimum stack frame" as
the goal. The next attack on spill needs to consider WHERE the
remaining spills land, not just how many bytes they consume.

### Path A retired

Cell B (`+ shfl`) buys only +0.6 % over C-1 alone, and stacking shfl on
top of C-2a (cell D) actually *hurts* by 1 TFLOPS (157.2 vs 158.2). The
shfl monkey-patch is no longer on the v2 critical path; `_bench_ws_shfl.py`
keeps it as a diagnostic-only knob.

The reason C-2a is better than B+C-2a is plausible but not proven:
shfl rewrites WG-selection branches into `__wg_id == N` checks, which
adds an extra integer compare per branch. In a workload that already
has the WGMMA pipe humming (as is the case after C-2a), this minor
serialization overhead is no longer hidden.

### The remaining bottleneck — fragment union, NOT branch divergence

After C-2a the kernel still has 168 registers used and 880 B of stack
spill. The remaining spill is the **fragment union** problem:

- Per-warpgroup, the actual live set (per WG1 thread: acc_s_1 + acc_o_1
  + acc_s_cast_1 + softmax scratch + 3 descriptors + temps) is roughly
  ~188 registers.
- But ptxas does per-thread allocation **without** warpgroup awareness.
  It allocates the union of all branches: 2× acc_s + 2× acc_o + 2×
  acc_s_cast + 2× softmax scratch + 18 descriptors + temps ≈ 410
  registers per thread.
- 410 needed vs 168 available → ~242 registers' worth of state must
  spill, which matches the ~250 B per-thread spill we observe.

This is **not** a branch-divergence problem. Path A's shfl rewrite
already proved that fixing the branch predicate does not change ptxas'
per-thread fragment live range allocation. The two analyses run on
separate code paths inside ptxas: branch uniformity affects the WGMMA
pipelining check (C7518), but does not feed into the function-scope
local allocator.

To force ptxas to recognize that `_1` and `_2` fragments have mutually
exclusive live ranges, we need to declare them inside the corresponding
`with T.ws(N):` frames so they have **scoped lifetime** in the lowered
CUDA — i.e., the lowered code becomes:

```cpp
if (__wg_id == 1) {
    float acc_s_1[64];   // declared inside the branch
    float acc_o_1[64];
    half  acc_s_cast_1[64];
    // ... WG1 body ...
}
if (__wg_id == 2) {
    float acc_s_2[64];   // independent declaration
    // ... WG2 body ...
}
```

C++ scope rules then guarantee `acc_s_1` does not exist outside the
WG1 branch, and ptxas' standard live range analysis (no special
inter-branch reasoning required) will correctly compute non-overlapping
live ranges.

TileLang currently does **not** support `T.alloc_fragment` inside a
`with T.ws(N):` frame — the fragment alloc gets hoisted to the kernel
top. Lifting this restriction is the right next attack. It requires
either a TileLang lowering pass change or a careful manual workaround.

### Productionizing C-2a

The current implementation is a regex post-process via the FFI hook on
`tilelang_callback_cuda_compile`. Three options to make v2 inherit it
on the standard registration path:

1. **Patch `tl_templates/cuda/common.h` body only** ✅ **VALIDATED** —
   see "Option 1 validation" section below. Single ~10-line edit, byte-
   identical ptxas output to the post-process variant. **This is the
   chosen path forward.**

2. **Patch TileLang's `codegen_cuda.cc`** to emit
   `name = tl::make_wgmma_descriptor<...>(ptr);` instead of
   `tl::initialize_wgmma_descriptor<...>(name, ptr);`, alongside adding
   the new template to `common.h`. Two-file TileLang upstream PR.
   Strictly correct but unnecessary now that Option 1 works.

3. **Hook the FFI override into the v2 op `__init__.py`** so any
   importer of v2 transparently inherits the rewrite. Single Python
   file change. Global side effect on the FFI registry — affects every
   kernel built in the same process.

Recommendation: **Option 1 is the production path** — single-file, ~10
line patch, no codegen change, no global FFI hook. The patch is saved
in `tools/tilelang_common_h_descv.patch` for upstream submission.

### Option 1 validation

Direct in-place edit of `$CONDA_PREFIX/lib/python3.12/site-packages/tilelang/src/tl_templates/cuda/common.h`,
replacing the five-bitfield-RMW body with a single 64-bit OR-and-store
into the union's `desc_` field. Reference parameter signature kept
unchanged. After the edit, ptxas reports **byte-identical** spill
metrics to the post-process variant:

| | C-1 only (no Option 1) | C-1 + Option 1 in-place | C-1 + post-process variant | Match |
|---|---|---|---|---|
| Stack frame | 1128 B | **880 B** | 880 B | ✓ |
| Spill stores | 3172 | 2948 | 2948 | ✓ |
| Spill loads | 3380 | 3668 | 3668 | ✓ |
| Registers | 168 | 168 | 168 | — |

Confirms the hypothesis: ptxas's pessimism about by-reference parameters
is keyed on whether the body **does** RMW operations on the referenced
struct. A single 64-bit store via `descriptor.desc_ = expr` is detected
as a write-only, non-aliasing operation, and after inlining the caller's
local can be SSA-promoted. The five separate bitfield writes
(`descriptor.bitfield.foo = ...`) were treated as RMWs because each
bitfield write involves loading the surrounding word, masking, and
storing.

Bench (3 runs, jitter ±0.3 TFLOPS):

| Configuration | TFLOPS (mean) |
|---|---|
| C-1 only (Option 1 not applied) | 140.2 |
| **C-1 + Option 1 in-place** | **156.8 ± 0.3** |
| **C-1 + Option 1 + shfl** | **159.1 ± 0.1** |
| C-1 + post-process desc-by-value | 158.2 ± 0.3 |

**Best total configuration: C-1 source + Option 1 common.h patch + shfl
post-process = 159.1 TFLOPS, +22.4 % over the original 130.0 baseline.**

Note that with Option 1 in place, **shfl is now reliably useful again**
(+2.3 TFLOPS, +1.5 %), whereas without Option 1 it was within noise.
Most likely explanation: with the WGMMA pipe finally hot enough to be
the bottleneck, the integer-compare overhead from the range-form WG
selection branches becomes visible.

### Applying Option 1 to a fresh environment

The patch lives at `tools/tilelang_common_h_descv.patch` in this repo.
Apply via:

```sh
TILELANG_SRC=$CONDA_PREFIX/lib/python3.12/site-packages/tilelang/src
patch -p1 -d $TILELANG_SRC < tools/tilelang_common_h_descv.patch
```

The patch is idempotent: it replaces 6 lines with 9 lines inside the
existing `initialize_wgmma_descriptor` template body. No other TileLang
files are touched. Re-applying to an already-patched file will fail
cleanly (use `patch --dry-run` to check).

This is currently a manual local patch on the dev box. The next step is
opening an upstream PR against tile-ai/tilelang. The patch is small
enough to land on its own; the only reason to bundle it with anything
else is if we also want to upstream the `T.ws(N)` shfl-broadcast lowering
fix at the same time (which would benefit any other WS kernel that
shares v2's branch divergence).

## Path C-2b result: alias _2 fragments to _1 storage (+76 % over Option 1)

The biggest remaining bottleneck after C-1 + Option 1 was the **fragment
union** problem: ptxas does per-thread register allocation as the union
of all warpgroups' fragments, even though no single thread ever reads
both `_1` and `_2` versions. The fix turns out to be a single regex
post-process — and the impact dwarfs every previous optimization
combined.

### The insight

Each thread is in exactly one warpgroup at runtime. A WG1 thread enters
the `if (128 <= tid && tid < 256)` branch and only ever touches `_1`
fragments; a WG2 thread enters `if (256 <= tid)` and only touches `_2`.
The two sets are mutually exclusive at the thread level, but ptxas
allocates separate register/stack slots for both because its register
allocator does not perform branch-uniform live-range coalescing across
threadIdx-based predicates.

We can tell the compiler the truth at the C++ level by literally
aliasing the names — `#define acc_s_2 acc_s_1` etc. — so that there is
only one underlying storage. ptxas then sees one fragment, not two, and
the per-thread spill drops correspondingly.

### The fix

A regex post-process in `_bench_ws_shfl.py::alias_rewrite()`:

1. Find every function-scope local declaration of the form
   `<type> <name>_1[<size>];`
2. For each, look for a matching `<type> <name>_2[<size>];` declaration
3. Delete the `_2` declaration and inject `#define <name>_2 <name>_1`
   right before `extern "C" __global__`

For v2 the regex matches **8 fragment pairs**:

| `_2` name | aliased to | type | bytes saved per thread |
|---|---|---|---|
| `acc_o_2` | `acc_o_1` | `float[64]` | 256 |
| `acc_s_2` | `acc_s_1` | `float[64]` | 256 |
| `acc_s_cast_2` | `acc_s_cast_1` | `half_t[64]` | 128 |
| `ls_2`, `sm_2`, `smp_2`, `ss_2`, `ssum_2` | `ls_1, ...` | `float[2]` each | 8 each (40 total) |
| **Total** | | | **680 B/thread** |

### ptxas summary

| | C-1 + Option 1 (no alias) | + alias | Δ |
|---|---|---|---|
| Stack frame | 880 B | **160 B** | **−82 %** |
| Spill stores | 2948 | 640 | −78 % |
| Spill loads | 3668 | 800 | −78 % |
| Registers used | 168 | 168 | unchanged |
| C7507 ignored set_max_nreg | gone | gone | — |
| C7518 divergent path WG.DP | present (with shfl: gone) | present (with shfl: gone) | — |
| C7513 (data dep) | gone | gone | — |
| **C7512 (insufficient regs for wgmma pipelining)** | absent | **newly visible** | **new bottleneck** |

The 680 B/thread we expected is matched by the 720 B drop in stack
frame (160 vs 880). The 8 alias pairs together delete five 256-B spill
slots' worth of stack — almost the entire previous spill.

### Result (3 runs, jitter ±0.3 TFLOPS, all 5 v2 correctness shapes PASS)

Bench shape `B=4 S=2048 H=64 Hkv=4 D=128 non-causal`. **FA3 reference on
the same shape: 647.3 TFLOPS** (measured via `_bench_fa3_baseline.py`,
flash_attn_interface.flash_attn_func).

| Configuration | TFLOPS | % of FA3 | Δ vs prev row |
|---|---|---|---|
| v2 original (no fixes) | 130.0 | 20.1 % | — |
| + C-1 (cast reorder) | 140.2 | 21.7 % | +7.8 % |
| + Option 1 (descriptor body single-store) | 156.8 | 24.2 % | +11.8 % |
| **+ alias _1/_2 fragments** | **276.9** | **42.8 %** | **+76.6 %** |
| **+ shfl WG-id broadcast** | **329.4** | **50.9 %** | **+19.0 %** |

So: alias alone is the single biggest improvement to date — bigger than
all previous fixes combined. With shfl on top we cross the 50 %-of-FA3
mark for the first time. **From 4.98× behind FA3 down to 1.97× behind.**

Note that **shfl is now contributing reliably +19 %** on top of alias
(it was within noise before alias). Most likely the WGMMA pipe is
finally running fast enough that the integer-compare overhead of the
range-form WG selection branches has become measurable. With both
alias and shfl applied, all "easy" remaining bottlenecks are visible:

- C7512 says wgmma pipelining is now register-bound. 168 regs used per
  thread is essentially the H200 maximum at 384 threads/CTA
  (65536 reg file / 384 threads ≈ 170 regs/thread). Each in-flight
  wgmma needs dedicated register slots for its accumulator, so to
  pipeline more wgmmas we need either more regs/thread (lower thread
  count, e.g. drop block_m to 64 to remove one consumer) or smaller
  per-wgmma fragments (lower block_n).
- LSU is no longer the dominant pipe (from 11.6× FA3 down to 0.5× FA3
  estimated, since spill load bytes dropped from 3380 → 800).

### Why this is a "C-2b" fix and what it isn't

This is *the* fix for the fragment-union problem we predicted in the
C-2a section. It does not change the kernel structure, does not change
TileLang lowering, and does not require any C++ patching of TileLang
templates. It is purely a textual rewrite of the lowered CUDA.

What it isn't: a permanent solution. The real fix is for TileLang's
register allocator (or codegen) to recognize that fragments declared
inside or around `T.ws(N)` frames have warpgroup-exclusive live ranges
and can share storage. The post-process is doing manually what TileLang
should be doing automatically. There are at least three places to put
the actual fix:

1. **TileLang `WarpSpecialize` lowering in `src/ir.cc`**: currently the
   `with T.ws(N):` frame creates an `If` + `Then` + `Attr` frame stack
   but does not push a `Block` frame. As a result, any
   `T.alloc_buffer` calls inside the `with` block get attached to the
   *outer* Kernel block's `alloc_buffers` list rather than to a block
   inside the if-then-else. Fixing this would let `LowerOpaqueBlock`
   emit the `Allocate` node *inside* the if-then-else body, where ptxas
   would see scoped lifetime. This is the cleanest fix.

2. **Per-warpgroup register hint in TileLang's storage scope**: invent
   a new fragment scope like `local.fragment.ws<N>` that tells the
   storage rewriter "this fragment is only live in WG N". Storage
   coalescing can then merge fragments with disjoint WG hints.

3. **Post-process the lowered CUDA** (what we do today): cheap, works,
   but doesn't propagate to other TileOPs kernels and is fragile to
   future TileLang codegen changes.

### Productionizing path

Currently the alias rewrite is in `_bench_ws_shfl.py::alias_rewrite()`,
gated on `ALIAS=1`. To make v2 inherit it on the standard registration
path, the same three options apply as for path C-2a (in-place common.h
patch / TileLang codegen change / FFI hook in v2 op). Given how dramatic
the speedup is, **the right next step is upstreaming option 1 above
into TileLang's `WarpSpecialize` lowering**. That fix would benefit any
other WS kernel as well.

### Cross-references

- `_postproc_alias_fragments.py` — standalone repro: `python3 _postproc_alias_fragments.py /tmp/v2_orig_hook.cu`
- `_bench_ws_shfl.py` (with `ALIAS=1`) — 4-cell ablation harness
- `_test_alias_correctness.py` — 5-shape correctness validator
- `_bench_fa3_baseline.py` — FA3 reference TFLOPS measurement

## Path C-3 partial: block_n=64 buys +20 % over block_n=128

After alias + shfl + Option 1, the visible warning is C7512 (insufficient
registers for wgmma pipelining). 168 regs/thread is the H200 cap at
384 threads/CTA (65536 / 384 ≈ 170). Each in-flight wgmma needs
dedicated accumulator register slots, so the # of pipelined wgmmas is
register-bound.

The cheapest way to relax that pressure is to **shrink the per-wgmma
accumulator** by halving `block_n` from 128 to 64.

### Result on the main bench shape (3 runs ±0.3 TFLOPS)

| `block_n` | TFLOPS | % of FA3 | Notes |
|---|---|---|---|
| 32 | 253.5 | 39.2 % | Too many K iterations, pipeline overhead dominates |
| **64** | **396.0** | **61.2 %** | **Sweet spot — registers freed, wgmma pipelines** |
| 128 | 329.5 | 50.9 % | Register-bound (C7512) |
| 256 | FAIL | — | smem 294 KB > H200 228 KB limit |

ptxas with block_n=64 is **completely clean**:

| | block_n=128 | block_n=64 | Δ |
|---|---|---|---|
| Stack frame | 160 B | **0 B** | gone |
| Spill stores | 640 B | **0 B** | gone |
| Spill loads | 800 B | **0 B** | gone |
| Registers used | 168 | **138** | -30 (under cap) |
| C7512 (insufficient regs) | present | **gone** | resolved |

All 5 v2 correctness shapes PASS at block_n=64.

### Caveat — block_n=64 is not the path forward to FA3

This is +20 % "for free" but is **not** the right structural answer.
FA3 uses block_n=128 (or larger) and gets 647 TFLOPS — bigger blocks
amortize per-wgmma issue overhead better and use the SM register file
more efficiently. The reason we have to *shrink* the block to gain
performance is that v2's per-thread register budget is bounded by 384
threads/CTA, which is itself a consequence of having three warpgroups
(producer + 2 consumers) all in one main_kernel function.

The fundamental fix is path 3: separate `__device__` functions per
warpgroup, dispatched at the top of `main_kernel`, so each warpgroup's
register budget is independent. With FA3-style structural separation we
should be able to use block_n=128 and reach the 80-100 % of FA3 range.
For now block_n=64 is a useful empirical data point but should not
become the production setting if path 3 lands.

Note that the wrong-output diff at non-divisor block_n (96, 48) is
NOT a kernel bug — those values just don't divide S=2048 cleanly and
v2 doesn't handle the residual block. Only `block_n ∈ {32, 64, 128}`
are valid choices for our test shape.

### Cross-reference

- `_bench_block_n.py` — block_n sweep on the main bench shape
- `_test_block_n64_all.py` — 5-shape correctness + 3x stability at block_n=64

## ncu profile of v2 best vs FA3: persistent scheduling is the real gap

After block_n=64, profiled both v2_best (`_ncu_v2_best.py`) and FA3
(`_ncu_fa3.py`) on the same shape `B=4 S=2048 H=64 Hkv=4 D=128
non-causal`:

| Metric | v2 best | FA3 | Notes |
|---|---|---|---|
| Compute (SM) Throughput | **41.3 %** | **79.4 %** | FA3 keeps SM busy 1.92× more |
| Memory Throughput | 34.3 % | 48.9 % | proportional difference |
| DRAM Throughput | 3.31 % | 5.97 % | both well below DRAM limit |
| L1/TEX Cache Throughput | 35.0 % | 50.7 % | similar ratio |
| L2 Cache Throughput | 17.2 % | 26.0 % | similar ratio |
| **Achieved Occupancy** | **18.4 %** | **14.1 %** | **FA3 is LOWER!** |
| Theoretical Occupancy | 18.75 % | 18.75 % | identical |
| Block Limit Registers | 1 | 1 | both 1 block/SM |
| Block Limit Shared Mem | 1 | 1 | both 1 block/SM |
| Active warps/SM | 11.78 | 9.00 | FA3 uses fewer warps |
| **Block Size** | (384,1,1) | (384,1,1) | identical |
| **Grid Size** | **(16, 64, 4) = 4096** | **(132, 1, 1) = 132** | **FA3 uses persistent scheduling** |

### The two big findings

**(1) Occupancy is NOT the bottleneck.** FA3 achieves *lower* occupancy
than v2 (14.1 % vs 18.4 %), but reaches almost twice the SM compute
throughput. So the gap is not "v2 needs more warps" — it's "v2's warps
are stalling on something FA3's aren't."

**(2) FA3 launches 132 CTAs (= the H200 SM count) while v2 launches
4096.** FA3 uses *persistent tile scheduling*: each SM gets exactly one
CTA at the start, and that CTA loops over `4096 / 132 ≈ 31` tiles
internally without exiting. The kernel name even confirms it:
`StaticPersistentTileScheduler<0>`. v2 launches a fresh CTA per tile,
which means every tile pays the full setup cost (smem init, mbarrier
init, TMA descriptor setup, register file init) and the GPU pays the
full CTA scheduler latency in between.

The 38.5 % long-scoreboard stall in v2 (top stall reason) is largely
this CTA-startup latency repeated 31 times per SM.

### What about fragment scoping?

I tried three approaches to get `T.alloc_fragment` to be scoped inside
a `T.ws(N)` frame in the lowered CUDA: bare alloc-in-ws (M1),
`@T.macro` encapsulation (M2), and explicit `T.block` inside the ws
frame (M3). All three still hoist the alloc to function scope. The
hoisting happens in some downstream pass that's not undoable from the
Python source level. See `_test_path3_alloc_scoping.py`.

This means **fragment scoping is not the path forward** — it's already
solved by the alias post-process (C-2b). The remaining gap is
elsewhere.

### Path forward: persistent tile scheduling via T.Persistent

TileLang already has `T.Persistent` (in `tilelang/language/loop.py:90`)
which constructs a persistent for loop bound to a tile domain. The
signature is:

```python
T.Persistent(domain=[tile_x_extent, tile_y_extent, tile_z_extent],
             wave_size=132,           # = num SMs on H200
             index=blockIdx.x,        # this CTA's index within wave
             group_size=8)            # tile grouping for L2 reuse
```

Each CTA enters the body in a loop, with the iteration variables
binding to a tile coordinate computed from `(wave_index * 132 + bx)`.
The loop exits when all tiles are processed.

To convert v2 to persistent: change the launch grid from
`(ceildiv(S, block_m), H, B)` to `(132, 1, 1)` and wrap the kernel
body in `for tile_x, tile_y, tile_z in T.Persistent([...]):`.

### Risks

The main correctness risk is mbarrier state across tile boundaries.
v2's k_full / k_empty / v_full / v_empty mbarriers track parity for
the K/V pipeline; if they're not reset between tiles, the parity bit
will be wrong on the second tile and waits will block forever or
release immediately. Two ways to handle this:

1. Re-initialize mbarriers at the top of each tile (single thread, fast)
2. Make the parity tracking reset based on tile index (more invasive)

Option 1 is simpler but adds a few hundred ns per tile. Acceptable.

Other risks: smem state from previous tile persists (we need to clear
acc_o_1 / sm_1 / ls_1 at the start of each tile, which we already do
in the WG1/WG2 setup blocks), TMA descriptors are per-kernel-launch so
they're fine, scheduler-named-barriers reset implicitly each tile.

### Cross-references

- `_ncu_v2_best.py` — ncu driver for v2 best config
- `_ncu_fa3.py` — ncu driver for FA3 reference
- `_test_path3_alloc_scoping.py` — fragment scoping experiments (all
  three approaches fail to scope the alloc)
- `/tmp/v2_best.ncu-rep`, `/tmp/fa3_best.ncu-rep` — ncu reports

### Cross-references

- `_postproc_wgmma_desc.py` — standalone repro: prints ptxas comparison
- `_bench_ws_shfl.py` (with `DESC_REWRITE=1`) — 4-cell ablation harness
- `_test_correctness_descv.py` — runs all 5 correctness shapes with the
  desc rewrite hook enabled
- GitHub issue [superAngGao/TileOPs-report-static#9](https://github.com/superAngGao/TileOPs-report-static/issues/9)
  comments 4192727756 and 4192736668 — same content with the full
  patching options writeup
