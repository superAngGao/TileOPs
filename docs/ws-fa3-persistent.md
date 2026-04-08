# WS GQA forward: persistent CTA scheduling (Approach A)

Relates to issue #9. Builds on `ws-fa3-v2-root-cause.md` (the v2 epilogue
fence fix).

## TL;DR

Convert the FA3-aligned IntraWGOverlap GQA forward kernel from
one-tile-per-CTA grid launch (`grid = (M, H, B)`) to **persistent CTA
scheduling** (`grid = (NUM_SMS, 1, 1)`). Each CTA processes multiple
`(B, H, M)` tiles via `T.Persistent`.

The implementation is `_test_ws_fa3_v2_persistent.py`. Performance is
within run-to-run noise of the per-tile baseline; correctness validated
across `NUM_SMS ∈ {1, 8, 132}`.

The five design choices below are NOT optional — they are the difference
between "works correctly + perf neutral" and "deadlocks or silently
corrupts data". v3 / v4 attempted persistent before and failed because
they got at least one of these wrong.

## Why per-tile parity tracking is wrong

`mbarrier`'s `phase parity` is a **cumulative hardware counter**. Every
`arrive_count` arrives flips the phase. The phase is NOT reset at any
software boundary — including the start of a new persistent loop iteration.

v2 (single-tile-per-CTA) used `barrier_wait(b, n_idx % 2)` where `n_idx`
is the local K-loop iter index, restarting at 0 for each kernel call.
That works **only because** each kernel call is a fresh CTA with a fresh
mbarrier (parity 0 / parity 1 pre-credit, depending on barrier type).

In a persistent kernel, the CTA processes tile 0, then tile 1, then ...
The mbarrier accumulates phase flips across all tiles. At the start of
tile 1, the mbarrier is at `parity = (initial + tile_0_flip_count) % 2`,
NOT at the initial state. Using `n_idx % 2` (which restarts at 0) for
the wait parity in tile 1 races / deadlocks against the actual hardware
state.

This is the root cause of v3's slowness and v4's NUM_SMS=8 correctness
failure. They both used `n_idx % 2` and tried to "patch" the symptom
with extra arrives, drains, etc. None of those patches are needed if you
fix the parity formula.

## The five design choices

### 1. Approach A: per-WG global iteration counters

Each warp group maintains its own monotonic global iteration counter via
`T.alloc_var("int32", init=0)`, declared at kernel scope, persisting
across persistent loop iterations:

```python
gi_kp  = T.alloc_var("int32", init=0)   # WG0 K loads
gi_vp  = T.alloc_var("int32", init=0)   # WG0 V loads
gi_kc1 = T.alloc_var("int32", init=0)   # WG1 K wgmmas
gi_vc1 = T.alloc_var("int32", init=0)   # WG1 V wgmmas
gi_kc2 = T.alloc_var("int32", init=0)   # WG2 K wgmmas
gi_vc2 = T.alloc_var("int32", init=0)   # WG2 V wgmmas
gi_q1  = T.alloc_var("int32", init=0)   # WG1 Q loads
gi_q2  = T.alloc_var("int32", init=0)   # WG2 Q loads
```

**Wait formulas (single-barrier-multi-buffer model, num_stages=2)**:

| Barrier type | Formula | Init parity |
|---|---|---|
| `*_full` (data ready, producer→consumer) | `gi % 2` | 0 |
| `*_empty` (buffer free, consumer→producer) | `(gi + 1) % 2` | 1 (pre-credited) |
| `q_full_*` (per-WG Q load) | `gi_q % 2` | 0 |

**Stage selection** (which buffer to use): also `gi % 2`. Same counter,
both for the wait parity AND the buffer index — that's the invariant.

**Increment**: AFTER each load / wgmma, increment the corresponding
counter:
```python
T.tma_copy(k[...], k_smem_0 if gi_kp % 2 == 0 else k_smem_1, barrier=k_full)
T.barrier_arrive(k_full)
gi_kp = gi_kp + 1
```

For the V pipeline (lagged by 1 — V[n-1] loaded at iter n), the V
counter starts at 0 and only increments when a V load / V wgmma actually
happens (i.e., NOT at the n_idx=0 prologue iter, but DOES increment in
the epilogue). Per tile, both `gi_kp` and `gi_vp` increase by exactly
`loop_range`.

### 2. Per-WG Q smem ownership

v2 loaded Q via 384-thread CTA-wide `cp.async` followed by
`T.sync_threads()`, then split into WGs. In persistent, this creates a
cross-WG smem race: WG0 (producer) participates in the Q load, but the
output store at the end of the previous tile is owned by WG1/WG2. WG0's
next-tile cp.async could race with WG1/WG2's in-flight tma_store on the
same `q_shared_*` buffer.

Fix: **move Q load INTO each consumer branch**. Each `q_shared_*`
becomes owned by exactly one WG:

```python
elif tx < 256:  # WG1
    for tile in T.Persistent(...):
        T.tma_copy(q[tile_b, row_base:row_base+half_m, tile_h, :],
                   q_shared_1, barrier=q_full_1)
        T.barrier_arrive(q_full_1)
        T.barrier_wait(q_full_1, gi_q1 % 2)
        gi_q1 = gi_q1 + 1
        # ... wgmma loop using q_shared_1 ...
        # ... output tma_store from q_shared_1 ...

else:  # WG2 — same with q_shared_2

if tx < 128:  # WG0 — never touches q_shared_*
    for tile in T.Persistent(...):
        # K/V loads only
```

Within a single WG, the cp.async / tma_load and the tma_store on the
same `q_shared_*` are issued in program order by the same warp scheduler,
so the async proxy's in-warp FIFO handles the hazard. **No fence
needed.** This is the WS idiom: ownership isolation eliminates the need
for explicit fences.

### 3. TMA-based Q load (not cp.async)

Within the consumer branch, use `tma_copy` for the Q load (not `T.copy`
which lowers to cp.async):

```python
T.tma_copy(q[...], q_shared_1, barrier=q_full_1)
T.barrier_arrive(q_full_1)
T.barrier_wait(q_full_1, gi_q1 % 2)
```

This requires a per-WG `q_full_1` mbarrier (`arrive_count=128`).

Why TMA: cp.async with only 128 issuing threads (one WG) added a +1.8
cycle/inst L1TEX scoreboard stall in NCU vs the v2 baseline. TMA goes
through the async proxy directly, doesn't hit L1TEX, and uses bulk
transfer. Without this, persistent ran at ~−10% vs baseline; with this,
it's at −1.4% (within noise).

### 4. Bootstrap stays OUTSIDE the persistent loop

The named-barrier bootstrap that v2 uses to make WG1 not deadlock on its
first `bar.sync(1, 256)` is fired **once per CTA**, before the persistent
loop starts:

```python
elif tx < 256:  # WG1
    T.inc_max_nreg(240)
    # Bootstrap: ONCE per CTA, OUTSIDE the persistent loop
    T.call_extern("handle", "tl::barrier_arrive_named", 1, 256)
    for tile in T.Persistent(...):
        # ... rest of WG1 work ...
```

Bar 1's leftover counter (=128 after each tile) is the **stable steady
state**. The next tile's first `T.sync_threads(barrier_id=1, arrive_count=256)`
on WG1 contributes 128, the leftover provides 128, total=256, flip,
counter=0, WG1 unblocks. This is functionally equivalent to having a
fresh bootstrap at the start of each tile.

**v4's "per-tile drain" was wrong.** It added an extra
`barrier_arrive_named(1, 256)` at the top of every tile after the first,
which over-drained the leftover and reversed the WG1↔WG2 ordering. Don't
do this.

### 5. Consumer epilogue v_empty release

Per tile, the producer issues `loop_range` V waits (in-loop `loop_range-1`
+ epilogue 1), but v2's consumer only arrives v_empty `loop_range - 1`
times (n_idx > 0 in-loop, no arrive in epilogue). v2 single-tile got
away with this because the initial v_empty pre-credit (parity 1) covered
the off-by-one.

In persistent multi-tile, the pre-credit is consumed only once. Tile 1's
producer first wait races / deadlocks because v_empty has only had
`loop_range - 1` flips, parity drifted from the pre-credit value.

Fix: **add `T.barrier_arrive(v_empty)` to each consumer's epilogue**,
after the last PV wgmma + warpgroup_fence_operand:

```python
# Consumer 1 epilogue
rescale_1(acc_o_1, ss_1)
T.barrier_wait(v_full, gi_vc1 % 2)
T.wgmma_gemm(acc_s_cast_1, v_smem_? , acc_o_1, ...)
T.wait_wgmma(0)
T.warpgroup_fence_operand(acc_o_1, num_regs=64)
# ↓ Release the last V buffer (v2 single-tile didn't need this).
T.barrier_arrive(v_empty)
gi_vc1 = gi_vc1 + 1
# ... output store from q_shared_1 ...
```

This makes per-tile consumer arrives = `loop_range` (matching producer
loads), restoring symmetry across tile boundaries. This is structural
correctness, not a hack: the consumer SHOULD release the last V buffer.
v2 just got away without it because there was no "next tile".

## Validation methodology

**Always validate the parity formula on a stripped 1P+2C minimal GEMM
first.** Don't try to debug parity bugs in the full attention kernel —
too many moving parts.

`_test_persistent_gemm_minimal.py` is exactly this: strips out softmax,
the V pipeline, causal masking, and the named-barrier scheduler. Only
keeps the K-dim full/empty pipeline + WS scheduling + persistent loop.
If approach A fails on this, the formula is wrong. If it passes, the
formula is right and any failure in the full kernel is from somewhere
else (Q load, bar 1/2, V pipeline lag, etc.).

The minimal GEMM was validated on:
- `NUM_SMS ∈ {1, 2, 4}` (1 stresses cross-tile carryover maximally)
- `K_blocks ∈ {1, 2, 4}` (covers odd and even loop counts per tile)

For the full attention kernel:
- `NUM_SMS=132` (H200's actual SM count, normal operation)
- `NUM_SMS=8` (heavy carryover, ~4096/8 = 512 tiles per CTA)
- `NUM_SMS=1` (single CTA serially processing all tiles, max stress)

All shapes × all NUM_SMS PASS for non-causal.

## Performance (GPU 7 H200 locked, B=4 S=4096 H=64 Hkv=8 D=128, block_n=176)

| Variant | TFLOPS | Δ vs v2_tb | % FA3 |
|---|---|---|---|
| v2_threadbind (production baseline) | 584.7 | — | 94.6% |
| **v2_persistent** | **576.5** | **−1.4%** | 93.3% |
| FA3 reference | 618.0 | +5.7% | 100% |

Persistent and per-tile-launch are perf-neutral within run-to-run noise.
The persistent kernel is ready as a foundation; further improvements
(causal tile pairing, cross-tile epilogue pipelining) build on top of it.

## Common pitfalls

1. **Don't bench on a busy GPU.** GPUs 1, 2, 7 are locked-frequency on
   this box, but only 2 and 7 are usually free. Always check
   `nvidia-smi --query-gpu=index,utilization.gpu --format=csv` before
   benching. A polluted GPU 1 measurement gave us a fake −10% slowdown
   that took an hour to chase.

2. **Use `block_n=176`, not 128.** The production baseline is 612 TFLOPS
   with `block_n=176` (FA3's sweet spot, requires the wgmma N gcd patch
   active in this conda env). `block_n=128` is ~558 TFLOPS, and the
   per-tile fixed overhead in persistent is a larger fraction of that
   smaller per-tile work, making the gap look worse.

3. **Don't add `tma_store_wait` or `fence_proxy_async` between tiles in
   the consumer.** WS doesn't need fences. The mbarrier graph + per-WG
   smem ownership + warp-scheduler program order handles all hazards.
   Adding fences either hides a real ownership bug or wastes cycles.

4. **Don't reuse v4's "fixes".** v4 added (a) per-tile bar 1 drain and
   (b) extra v_empty arrives that don't increment gi_vc. Both were
   misguided patches for the n_idx parity bug. With Approach A, neither
   is needed; the bar 1 leftover self-corrects, and the v_empty epilogue
   release in choice 5 above is what actually closes the V pipeline
   asymmetry.

5. **Don't try to skip the minimal GEMM validation.** Approach A is
   correct in principle, but parity formulas are easy to typo. Validate
   the formula on a minimum kernel first, then port. The minimal GEMM
   takes ~5 minutes to write and ~30 seconds to compile+run; it'll save
   you from hour-long debugging in the full attention kernel.

## Causal mode (NOT yet implemented)

The non-causal version above does not handle causal mode. Causal needs
two extra things:

1. **Diagonal-block-only mask optimization** — only the last
   (`n_idx == loop_range - 1`) block needs the mask; earlier blocks are
   strictly below the diagonal. This is already in v4 / fwd.py, just
   needs porting.

2. **`loop_range == 1` edge case at `tile_m == 0`** — the consumer's
   `n_idx == 0` branch skips all V pipeline operations, so a tile with
   `loop_range == 1` (which causal `tile_m=0` always has) does NOT do
   any V load / arrive at all. This breaks the per-tile bar 1/2 op
   count symmetry that the self-correct argument relies on.

   Possible fixes: special-case `tile_m == 0`, or convert bar 1/2 from
   named scheduler to mbarrier with global counter, or always issue
   one dummy V iteration.

3. **Tile pairing** (separate optimization) — pair `(tile_m=k,
   tile_m=M-1-k)` into the same CTA's persistent stream so each pair
   has constant total work `M+1` iters. This flattens the causal load
   imbalance and is what brings causal from ~46% FA3 up to ~85% FA3.

## Files

| File | Purpose |
|---|---|
| `_test_ws_fa3_v2_threadbind.py` | Production baseline (single-tile, 612 TFLOPS) |
| `_test_ws_fa3_v2_persistent.py` | **This doc's implementation** (Approach A) |
| `_test_persistent_gemm_minimal.py` | 1P+2C persistent GEMM, validates parity formula |
| `_test_ws_fa3_v3.py` | First (failed) persistent attempt — `T.ws()` style |
| `_test_ws_fa3_v4_persistent.py` | Second (failed) persistent attempt — has wrong drains |
