# FP8 GQA AKO Log

## Status

- Maximum rounds: 300
- Selected production candidate: current main baseline
- Best validated candidate: current main baseline
- Current structural question: whether the latest TileLang/lowering stack can
  preserve FA3-style grouped QK/PV overlap without fragment-layout conversion,
  register spilling, or conservative scoreboard serialization.

## Round 000: Contract Freeze

**Hypothesis**

A clean, fixed contract is required before interpreting any optimization as a
kernel improvement.

**Action**

- cloned current upstream `main`;
- created branch `perf/gqa-fp8-ako-fa3-level`;
- fixed GPU 4 and the runtime images listed in `README.md`;
- separated TileOps-only validation from the FA3 A/B environment;
- retained the previous dirty worktree as historical evidence only.

**Gate result**

Passed:

- official-runner correctness: `8 passed`;
- stable benchmark: `warmup=5`, `repeat=20`, `trials=3`;
- same-process TileOps/FA3 comparison on GPU 4.

## Round 001: Clean Main Baseline

**Hypothesis**

The public `[B, Hkv]` descale wrapper may explain part of the observed gap, but
the serial QK/softmax/PV mainloop is expected to remain the dominant wall.

**Action**

Measured:

- canonical TileOps op with 2D descales;
- the same TileOps kernel with scales pre-expanded outside the timed region;
- FA3 with the same FP8 Q/K/V and descales.

**Result**

| Shape | TileOps canonical | TileOps pre-expanded | FA3 | Canonical / FA3 | Descale expansion |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 H32/Hkv8 FP16 | 0.050490 ms | 0.041937 ms | 0.028675 ms | 1.761x | 20.4% |
| S896 H32/Hkv8 BF16 | 0.050643 ms | 0.042058 ms | 0.028728 ms | 1.763x | 20.4% |
| S1792 H32/Hkv8 FP16 | 0.140337 ms | 0.131501 ms | 0.086254 ms | 1.627x | 6.7% |
| S1792 H32/Hkv8 BF16 | 0.139363 ms | 0.130789 ms | 0.086067 ms | 1.619x | 6.6% |
| S3584 H64/Hkv8 FP16 | 0.812658 ms | 0.803397 ms | 0.534705 ms | 1.520x | 1.2% |
| S3584 H64/Hkv8 BF16 | 0.812694 ms | 0.805893 ms | 0.529788 ms | 1.534x | 0.8% |
| S7168 H64/Hkv8 FP16 | 3.054555 ms | 3.045751 ms | 2.040679 ms | 1.497x | 0.3% |
| S7168 H64/Hkv8 BF16 | 3.060255 ms | 3.047775 ms | 2.033469 ms | 1.505x | 0.4% |

**Decision**

Direct 2D descale consumption is selected as a low-risk short-sequence
improvement. It is not treated as the structural answer: long-sequence
performance still requires about a 1.5x improvement to reach FA3.

## Round 002: Direct 2D Descale Consumption

**Hypothesis**

The kernel can consume the public FA3 `[B, Hkv]` descale ABI directly. Removing
the repeated host-side expansion should recover the pre-expanded baseline
without changing the mainloop.

**Action**

- changed the TileLang kernel tensor contract from internal 3D scale tensors to
  direct 2D descales;
- mapped each query head to its owning KV head inside the persistent task;
- removed `repeat_interleave`, `expand`, and `contiguous` from kernel dispatch;
- removed the undocumented direct-kernel 3D scale compatibility path.

**Gate result**

- ruff: pass;
- official-runner correctness: `8 passed`;
- no CUDA-events fallback accepted in benchmark rows.

| Shape | Round 001 | Round 002 | Change | FA3 | Round 002 / FA3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 H32/Hkv8 FP16 | 0.050490 ms | 0.042019 ms | -16.8% | 0.028622 ms | 1.468x |
| S3584 H64/Hkv8 FP16 | 0.812658 ms | 0.803116 ms | -1.2% | 0.534503 ms | 1.503x |
| S7168 H64/Hkv8 FP16 | 3.054555 ms | 3.034873 ms | -0.6% | 2.044088 ms | 1.485x |

**Decision**

Accepted. This closes the wrapper allocation gap and makes the public ABI the
native kernel ABI. The long-sequence structural wall is unchanged.

## Round 003: Lowering And NCU Audit

**Hypothesis**

The remaining long-sequence gap is caused by exposed dependency latency in the
consumer mainloop, not DRAM bandwidth. Generated code may also reveal local
work that can be removed before changing the pipeline structure.

**Action**

- profiled one warmed S3584 H64/Hkv8 FP16 call with NCU `--set full`;
- isolated the generated TileLang cache for source inspection;
- inspected registers, shared memory, occupancy, scheduler eligibility,
  scoreboard stalls, local-memory instructions, and static Hopper SASS.

**Result**

- duration: `802.272 us`;
- compute / memory throughput: `38.06% / 48.77%`;
- tensor-pipe active: `32.75%`;
- registers: `168` per thread;
- dynamic shared memory: `197.63 KiB`;
- achieved occupancy: `18.70%`, one CTA per SM;
- no eligible warp: `60.46%`;
- dominant PC samples: long scoreboard `14,495`, wait `13,882`, short
  scoreboard `9,148`, barrier `8,831`;
- DRAM throughput was only `1.87%`.

The generated CUDA also performs two complete `acc_o_layout_seed` conversions
and shared-memory writes. The custom output helper reads `acc_o` directly and
does not consume either seed buffer, so these writes are a concrete removable
candidate rather than a speculative pipeline rewrite.

**Decision**

The profile confirms the serial dependency wall. First remove the dead
layout-seed work and re-run all gates; then probe whether typed QK/PV overlap is
expressible without fragment conversion or spill growth.

## Round 004: Remove Layout Seeds Without A Replacement Contract

**Hypothesis**

Because the custom output helper does not read `acc_o_layout_seed_1/2`, deleting
the buffers and their copies should remove dead runtime work.

**Action**

Removed both shared buffers and both `T.copy(acc_o, acc_o_layout_seed)` calls
without adding an explicit fragment layout.

**Gate result**

Rejected at lowering:

```text
InternalError: In PrimFunc main variables (acc_o_1, acc_o_2) are used,
but are not passed in as API arguments
```

The copies were not semantically required by the output calculation, but they
were the only TileLang-visible consumers anchoring the raw accumulator
fragments and their layouts. A no-contract deletion is therefore invalid.

## Round 005: Explicit TileLang PV Accumulator Layout

**Hypothesis**

An explicit `tilelang.layout.Fragment` matching the FA3-style 64x128 PV
accumulator mapping can preserve the compiler contract without materializing
the dummy seed copies.

**Action**

- defined the 64x128 PV accumulator thread/register mapping as a TileLang
  fragment;
- annotated both consumer warpgroup accumulators with that layout;
- removed the two 16 KiB seed buffers and generated FP32-to-output conversion
  loops.

**Gate result**

- ruff: pass;
- official-runner correctness: `8 passed`;
- generated CUDA seed/local-cast markers: `0`;
- dynamic shared memory: about `197.6 KiB -> 161.0 KiB`;
- no CUDA-events fallback accepted.

| Shape | Round 002 | Round 005 | Change | FA3 | Round 005 / FA3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 H32/Hkv8 FP16 | 0.042019 ms | 0.041315 ms | -1.7% | 0.028664 ms | 1.441x |
| S3584 H64/Hkv8 FP16 | 0.803116 ms | 0.802621 ms | -0.1% | 0.534570 ms | 1.501x |
| S7168 H64/Hkv8 FP16 | 3.034873 ms | 3.040450 ms | +0.2% | 2.038617 ms | 1.491x |

**Decision**

Accepted. The long-sequence change is within measurement noise, while the
implementation replaces hidden runtime work with an explicit TileLang layout
contract and improves the short-sequence row.

## Round 006: Same-Contract FA3 NCU Reference

**Hypothesis**

A same-input NCU profile can identify which execution property, rather than
which source-level feature, separates FA3 from the selected TileOps path.

**Action**

Profiled one warmed FA3 S3584 H64/Hkv8 FP16 call with the same GPU, input
generator, NVTX boundary, and NCU `--set full` collection.

**Result**

| Metric | TileOps Round 003 | FA3 |
| --- | ---: | ---: |
| NCU duration | 802.27 us | 530.46 us |
| Tensor-pipe active | 32.75% | 49.21% |
| Compute throughput | 38.06% | 50.15% |
| Eligible warps / scheduler | 0.55 | 0.65 |
| Registers / thread | 168 | 168 |
| Dynamic shared memory | 197.63 KiB | 191.49 KiB |

The static SASS shapes expose the strongest difference: FA3 issues wide
`QGMMA.64x224x32` QK operations, whereas the selected TileOps path emits QK as
many `QGMMA.64x32x32` instructions.

**Decision**

Use wide-QK lowering as the next isolated structural target. Register count
alone is not the explanation because both kernels use 168 registers per
thread.

## Round 007: Wide QK With An Incomplete Layout Contract

**Hypothesis**

Replacing TileLang's seven n32 QK issues with one CUTE-selected m64n224 group
should preserve the existing TileLang-visible softmax path.

**Action**

- added a typed CUTE wide-QK helper;
- annotated Q/K shared memory and the QK accumulator;
- left the softmax row fragments on inferred layouts.

**Gate result**

Rejected by correctness: the kernel compiled, but output contained non-finite
values. Generated CUDA showed that `ss` and `ls` materialization had been
eliminated, leaving the PV and epilogue helpers to read uninitialized shared
state.

**Decision**

The wide instruction itself was not disproven. The candidate lacked the
softmax row-fragment portion of the compiler contract.

## Round 008: Wide QK With Complete Row Layouts

**Hypothesis**

Explicit replicated row layouts for `sm`, `smp`, `ss`, `ssum`, and `ls` should
preserve the TileLang softmax dataflow around the wide QK accumulator.

**Action**

Added a four-way replicated row fragment for both consumer warpgroups and
retained the Round 005 explicit PV accumulator layout.

**Gate result**

- official-runner correctness: `8 passed`;
- static SASS QK: `112 x m64n32 -> 16 x m64n224`;
- registers: unchanged at `168` per thread;
- NCU tensor-pipe active: `32.75% -> 35.96%`;
- NCU duration: `802.27 us -> 729.95 us`;
- no CUDA-events fallback accepted.

| Shape | Round 005 | Round 008 | Change | FA3 | Round 008 / FA3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 H32/Hkv8 FP16 | 0.041315 ms | 0.038858 ms | -5.9% | 0.028688 ms | 1.354x |
| S3584 H64/Hkv8 FP16 | 0.802621 ms | 0.733979 ms | -8.6% | 0.529436 ms | 1.386x |
| S7168 H64/Hkv8 FP16 | 3.040450 ms | 2.757815 ms | -9.3% | 2.039602 ms | 1.352x |

**Decision**

Accepted. This is the first structural mainloop improvement in the clean AKO
ladder. The remaining gap tracks exposed softmax/PV dependency more than QK
instruction shape.

## Round 009: Direct PV Accumulation With Deferred V Descale

**Hypothesis**

The per-KV-head V descale is constant across all N tiles. Keeping the output
accumulator in raw PV-WGMMA units, applying online-softmax row rescaling before
each PV issue, and applying V descale once in the epilogue should be equivalent
to materializing a second delta accumulator on every tile.

**Action**

- split the synchronous PV helper into an asynchronous direct-accumulate issue;
- replaced the 64-float per-thread delta accumulator and scalar merge loop with
  an explicit raw-layout row-rescale helper;
- deferred the constant V descale to the final output conversion;
- retained the serial QK -> softmax -> PV wait order for this round.

**Gate result**

- official-runner correctness: `8 passed`;
- no timing fallback accepted;
- stable GPU4 results:

| Shape | Round 008 | Round 009 | Change | FA3 | Round 009 / FA3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 H32/Hkv8 FP16 | 0.038858 ms | 0.039155 ms | +0.8% | 0.028694 ms | 1.365x |
| S3584 H64/Hkv8 FP16 | 0.733979 ms | 0.722727 ms | -1.5% | 0.528954 ms | 1.366x |
| S7168 H64/Hkv8 FP16 | 2.757815 ms | 2.719297 ms | -1.4% | 2.036495 ms | 1.335x |

**Decision**

Accepted. The short-row regression is negligible, the long rows improve, and
the direct accumulator representation removes the extra delta lifetime needed
by the old synchronous helper. Round 010 may now move the PV wait across the
next QK issue without changing the arithmetic representation again.

## Round 010: QK(n) / PV(n-1) WGMMA Overlap

**Hypothesis**

After Round 009, the previous PV writes only `acc_o` while the next QK writes
only `acc_s`. Issuing both groups before `wait_group<1>` should overlap Tensor
Core work without changing softmax order or adding another accumulator.

**Action**

- issued `QK(n)` while `PV(n-1)` remained outstanding;
- used `wait_group<1>` to retire the previous PV before reading/rescaling
  `acc_o`;
- used `wait_group<0>` before consuming the current QK scores;
- delayed each V-buffer release by one loop iteration while preserving the
  existing double-buffer phases.

**Gate result**

- official-runner correctness: `8 passed`;
- no timing fallback accepted;

| Shape | Round 009 | Round 010 | Change | FA3 | Round 010 / FA3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 H32/Hkv8 FP16 | 0.039155 ms | 0.038082 ms | -2.7% | 0.028710 ms | 1.326x |
| S3584 H64/Hkv8 FP16 | 0.722727 ms | 0.703467 ms | -2.7% | 0.536064 ms | 1.312x |
| S7168 H64/Hkv8 FP16 | 2.719297 ms | 2.656325 ms | -2.3% | 2.040118 ms | 1.302x |

S3584 NCU:

| Metric | Round 008 | Round 010 |
| --- | ---: | ---: |
| Duration | 729.95 us | 701.82 us |
| Tensor-pipe active | 35.96% | 37.28% |
| Eligible warps / scheduler | 0.56 | 0.58 |
| Registers / thread | 168 | 168 |
| Static `WARPGROUP.ARRIVE` | 12 | 8 |
| Static `WARPGROUP.DEPBAR` | 6 | 6 |
| Local load instructions | 229376 | 458752 |
| Local store instructions | 121856 | 237072 |

**Decision**

Accepted. The intended QK/PV overlap survives lowering and improves every
representative shape. The remaining local-memory traffic shows that fragment
state still escapes across the helper boundary; reducing that spill/lifetime
cost is now more promising than adding another pipeline stage.

## Round 011: Pass Softmax Row Fragments Directly Across the Helper Boundary

**Hypothesis**

The generated CUDA represents each consumer's `ss` and `ls` row state as two
thread-local floats. Passing those fragments directly to the CUTE rescale and
output-store helpers should remove four 64-element shared arrays and their
fragment-to-shared copies.

**Action**

- removed the two `ss_shared` and two `ls_shared` arrays;
- passed `ss.data` and `ls.data` to the C++ helpers;
- selected the two local row values from the CUTE row coordinate.

**Gate result**

- official-runner correctness: `8 passed`;
- the first formal S3584 benchmark compile remained CPU-bound for more than
  11 minutes and did not produce a runnable kernel;
- the benchmark was terminated, so no latency result is reported.

**Decision**

Rejected. Correct arithmetic is insufficient when a representation change
causes pathological compile time. Passing TileLang fragment storage through
this helper boundary expands the lowering/compiler problem substantially.
The `ss` rescale path and the final `ls` output path will be tested separately
before considering any combined change.

## Round 012: Pass Only the Online-Softmax Rescale Fragment Directly

**Hypothesis**

The pathological compile time in Round 011 may have come from combining local
`ss` and `ls` pointers with the final CUTE output-store template. Keeping the
final `ls` path unchanged while passing only the two-float `ss` fragment should
isolate that interaction.

**Action**

- retained the original shared-memory `ls` path;
- removed only the two `ss_shared` arrays and copies;
- passed each consumer's `ss` fragment to the accumulator-rescale helper.

**Gate result**

- formatting and lint gates passed;
- the formal S3584 benchmark remained CPU-bound in compilation and did not
  produce a runnable kernel within a four-minute hard limit.

**Decision**

Rejected. The compile-time failure follows the local `ss` fragment across the
extern/helper boundary. Shared staging remains the practical compiler contract
for this row state in the current TileLang/CUTE integration.

## Round 013: Pass Only the Final Normalization Fragment Directly

**Hypothesis**

The final output-store helper is called once per consumer CTA rather than once
per K/V tile. Passing only `ls` directly may remove its shared round trip
without triggering the repeated online-rescale compile behavior from Round 012.

**Action**

- retained the original shared-memory `ss` path;
- removed only the two `ls_shared` arrays and copies;
- passed each consumer's `ls` fragment to the final CUTE output-store helper.

**Gate result**

- formatting and lint gates passed;
- the formal S3584 benchmark remained CPU-bound in compilation and did not
  produce a runnable kernel within a four-minute hard limit.

**Decision**

Rejected. Both row-state fragment handoffs independently trigger pathological
compile time. The current compiler-friendly boundary is therefore explicit
shared staging for `ss` and `ls`. Subsequent rounds will change scheduling or
softmax organization rather than remove these copies through raw fragment
pointers.

## Round 014: Persistent Scheduler Group Size 4

**Hypothesis**

Reducing the persistent scheduler's grouping from eight tiles to four may trade
some GQA-group locality for better tail-wave balance without changing the
kernel arithmetic.

**Action**

Changed the producer and both consumer `T.Persistent` loops from
`group_size=8` to `group_size=4`.

**Gate result**

- lowering completed and produced a cached executable;
- the first S3584 warmup did not terminate;
- GPU4 remained at 100% utilization until the experiment container was
  explicitly stopped.

**Decision**

Rejected as a liveness failure. In this kernel, `group_size` is part of the
three-warpgroup persistent scheduling and barrier-phase protocol, not an
independent locality parameter. Future scheduler changes must derive and test
the producer/consumer tile sequence together rather than sweep this value in
isolation.

## Round 015: Skip First-Tile Output Rescale

**Hypothesis**

The output accumulator is explicitly zeroed before the first PV. FA3 handles
the first tile separately, so the first shared `ss` copy and multiplication of
the zero accumulator appear arithmetically redundant.

**Action**

Guarded each consumer's `ss` staging and accumulator-rescale helper with
`n_idx > 0`; all steady-state iterations and barriers were unchanged.

**Gate result**

- lowering completed;
- the official-runner correctness process entered the generated kernel but did
  not terminate;
- GPU4 remained at 100% utilization until the named test container was stopped.

**Decision**

Rejected as a liveness failure. In the current lowering, the rescale helper is
also an accumulator-lifetime/compiler anchor. Its first-iteration presence
cannot be removed as if it were only scalar arithmetic. A future first-tile
specialization would need an explicit WGMMA operand/fence contract.

## Round 016: Reuse Previous-Max Row Storage for the Current Sum

**Hypothesis**

After the online-softmax rescale factor is computed, the previous-max row
fragment is dead for the current tile. Reusing it as the reduction-sum
destination should shorten live row state while keeping all softmax operations
in TileLang.

**Action**

- added an experimental scaled online-softmax macro that reuses the
  previous-max fragment;
- removed the separate `ssum` fragment and layout annotation from both
  consumers;
- kept QK, PV, barriers, and arithmetic order unchanged.

**Gate result**

- official-runner correctness: `8 passed`;
- no timing fallback accepted;

| Shape | Round 010 | Round 016 | Change | FA3 | Round 016 / FA3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 H32/Hkv8 FP16 | 0.038082 ms | 0.038125 ms | +0.1% | 0.028690 ms | 1.329x |
| S3584 H64/Hkv8 FP16 | 0.703467 ms | 0.703243 ms | -0.0% | 0.534624 ms | 1.315x |
| S7168 H64/Hkv8 FP16 | 2.656325 ms | 2.654613 ms | -0.1% | 2.039317 ms | 1.302x |

S3584 NCU remained effectively identical to Round 010:

- duration: `701.056 us`;
- registers: `168` per thread;
- dynamic shared memory: `164.864 KiB`;
- local loads/stores: `458752 / 237072`;
- tensor-pipe active: `37.19%`.

**Decision**

Rejected as source-only churn. The compiler already reuses the row lifetime:
the generated resource use, local-memory traffic, and runtime are unchanged.
The existing shared softmax utility remains the smaller public API.

## Round 017: Independent P Fragment for True Softmax/PV Overlap

**Hypothesis**

Round 010 reuses the QK accumulator as the FP8 P operand. It therefore issues
`PV(n-1)` before `QK(n)` and waits for PV before softmax. Keeping a separate
28-register FP8 P fragment should enable the FA3 ordering:
`QK(n) -> PV(n-1) -> wait<1> -> softmax(n) -> wait<0>`.

**Action**

- added an explicitly annotated per-consumer FP8 P fragment;
- split score-to-P packing from the RS-WGMMA issue helper;
- handled the first QK tile and final PV tile separately;
- reordered the steady state so PV remains outstanding during the next
  softmax.

**Gate result**

- S896 H32/Hkv8 completed at `0.040299 ms` versus FA3 `0.028971 ms`;
- this was `5.8%` slower than the Round 010 short-shape row;
- S3584 did not terminate after entering the generated kernel;
- a `V2P_NUM_SMS=1792` probe assigning one work item per persistent CTA also
  failed to terminate, ruling out cross-work-item phase reuse as the sole
  cause;
- the small official correctness shape likewise failed the liveness gate.

**Decision**

Rejected. The independent P representation exposes the desired algorithmic
schedule, but the current TileLang fragment-to-extern-to-async-RS-WGMMA
boundary does not maintain a valid long-loop scoreboard contract.

## Round 018: Explicit P-Fragment WGMMA Fences

**Hypothesis**

The compiler may not know that the raw FP8 P fragment remains an asynchronous
RS-WGMMA operand until `wait_group<0>`. Explicit operand fences after packing
and after the wait may make the fragment's read/overwrite lifetime visible.

**Action**

Added `T.warpgroup_fence_operand(..., num_regs=28)` around both consumers'
pack/overwrite boundaries without changing the Round 017 schedule.

**Gate result**

The S3584 probe again entered the generated kernel and did not terminate.

**Decision**

Rejected. Source-level operand fences are insufficient for this cross-helper
fragment lifetime. The true-overlap direction remains structurally relevant,
but its P packing and RS-WGMMA issue must share a compiler-visible boundary or
move into one typed helper.

## Round 019: Shared-Memory P Mailbox

**Hypothesis**

A per-consumer `[28 registers, 128 lanes]` shared mailbox can keep P out of a
cross-helper TileLang fragment. Packing writes coalesced lane-major words and
the PV helper immediately reloads them into its own local P registers.

**Action**

- allocated two 14 KiB P mailboxes, keeping total shared memory below the H200
  one-CTA-per-SM limit;
- split the existing fused score-pack/PV helper into a shared pack and a
  mailbox-load/PV issue;
- retained the proven Round 010 schedule to isolate representation cost and
  liveness before attempting true overlap.

**Gate result**

The S896 probe entered the generated kernel but did not terminate.

**Decision**

Rejected. Even without schedule reordering, separating P packing from the
asynchronous RS-WGMMA issue breaks the current compiler/scoreboard contract.
P packing, register lifetime, and PV issue must remain in one typed boundary.

## Round 020: Typed Pack-QK-PV Helper

**Hypothesis**

Keeping the previous-tile P packing, current-tile QK issue, and previous-tile
PV issue inside one C++ helper should preserve the register-source lifetime
that Rounds 017-019 lost while exposing the FA3 ordering:
`pack P(n-1) -> QK(n) -> PV(n-1) -> wait<1> -> softmax(n) -> wait<0>`.

**Action**

- added a typed helper that packs the previous score accumulator into 28 local
  FP8 words, issues and commits current QK, then issues and commits previous PV;
- split the first QK and final PV out of the steady-state loop;
- kept barriers, online softmax, accumulator rescaling, and the final epilogue
  in TileLang.

**Gate result**

- official-runner correctness: `8 passed`;
- S896 H32/Hkv8 FP16: `0.039301 ms` versus Round 010 `0.038082 ms`,
  a `3.2%` regression; FA3 measured `0.028709 ms`;
- S3584 entered the generated kernel and remained at 100% GPU utilization
  without terminating; the named container was stopped after the liveness
  gate failed;
- no S7168 timing or profiler run was attempted after the long-loop failure.

**Decision**

Rejected. A single typed boundary is sufficient for the short loop and
preserves numerical correctness, but it does not establish a valid long-loop
scoreboard protocol. It also adds a material short-shape regression. The
accepted implementation remains Round 010.

## Round 021: Split V-Transpose Source and Destination

**Hypothesis**

Round 010 transposes V in place and synchronizes all 128 producer threads
before and after every LDSM/STSM micro-step. Allocating distinct source and
Tensor-Core-layout destination buffers should remove those internal barriers,
reduce shared-memory conflicts, and move the producer closer to FA3.

**Action**

- collected a focused same-contract NCU comparison before changing source;
- allocated two additional `[128, 224]` FP8 shared buffers for transposed V;
- changed the transpose helper to read and write distinct buffers without
  per-step barriers;
- preserved TileLang TMA, mbarriers, persistent scheduling, QK/PV order,
  softmax, and epilogue.

The focused baseline comparison confirmed a large execution-level gap:

| Metric | Round 010 | FA3 |
| --- | ---: | ---: |
| Duration | 705.44 us | 530.50 us |
| Tensor-pipe active | 37.15% | 49.30% |
| Dynamic instructions | 208.29 M | 183.20 M |
| Shared bank conflicts | 3.140 M | 0.716 M |
| Local loads / stores | 458752 / 237072 | 444416 / 109104 |

**Gate result**

- official-runner dequantized-reference smoke: `1 passed`;
- S896 H32/Hkv8 FP16: `0.038327 ms`, a `0.6%` regression from Round 010;
- S3584 H64/Hkv8 FP16: `0.706746 ms`, a `0.5%` regression from Round 010;
- focused candidate NCU duration: `704.54 us`;
- candidate shared bank conflicts increased to `3.287 M`; tensor-pipe active
  and local traffic were unchanged.

**Decision**

Rejected. The in-place synchronization was not the source of the remaining
shared-memory conflict or performance gap. Separate V buffers consume another
56 KiB of shared memory without improving the mainloop. The accepted
implementation remains Round 010, and subsequent work should target the
LDSM/STSM/PV access pattern or consumer schedule rather than buffer aliasing.

## Round 022: Vectorized CUTE Output Copy

**Hypothesis**

Source-correlated NCU counters refined the Round 021 diagnosis: 2.753 M of the
shared-memory excessive wavefronts came from 128 scalar `LDS.U16` instructions
in the output epilogue, not from the V transpose. Replacing the hand-written
scalar shared loads with a 128-bit CUTE tiled copy should remove that conflict
without changing the accepted QK/PV schedule.

**Action**

- kept the existing STSM register-to-shared output conversion;
- replaced eight scalar shared loads per output vector with
  `AutoVectorizingCopyWithAssumedAlignment<128>`;
- used the same 128-thread, 64x128 output tile and swizzled shared layout;
- preserved the Round 010 mainloop, barriers, online softmax, and descale ABI.

**Gate result**

- official-runner correctness: `8 passed`;
- S896 H32/Hkv8 FP16: `0.034846 ms` versus Round 010 `0.038082 ms`,
  an `8.5%` improvement; FA3 measured `0.028663 ms`;
- S1792 H32/Hkv8 FP16: `0.108867 ms`; FA3 measured `0.086161 ms`;
- S3584 H64/Hkv8 FP16: `0.666609 ms` versus Round 010 `0.703467 ms`,
  a `5.2%` improvement; FA3 measured `0.530303 ms`;
- S7168 H64/Hkv8 FP16: `2.550040 ms` versus Round 010 `2.656325 ms`,
  a `4.0%` improvement; FA3 measured `2.043426 ms`;
- focused S3584 NCU duration fell from `705.44 us` to `662.85 us`;
- shared-memory bank conflicts fell from `3.140 M` to `0.344 M`;
- dynamic instructions fell from `208.29 M` to `202.42 M`;
- tensor-pipe active rose from `37.15%` to `39.60%`.

**Decision**

Accepted as the new baseline. The change removes a measured epilogue
bottleneck with a standard CUTE vector-copy primitive and improves every
measured FP16 shape. The remaining gap to FA3 is approximately 22-26%, so the
next search should return to mainloop issue efficiency, local-memory traffic,
and producer/consumer overlap.
