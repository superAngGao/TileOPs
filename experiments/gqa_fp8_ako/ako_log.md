# FP8 GQA AKO Log

## Status

- Maximum rounds: 300
- Selected production candidate: Round 042 compressed producer phase state
- Best validated candidate: Round 042 compressed producer phase state
- Current structural question: whether the latest TileLang/lowering stack can
  preserve FA3-style grouped QK/PV overlap without fragment-layout conversion,
  register spilling, or conservative scoreboard serialization.
- Compiler-artifact rule: every candidate is compiled with a unique empty
  TileLang cache. Shared-cache results are not accepted for helper-only edits.

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

## Round 023: Derive K-Producer Phase From Tile Index

**Hypothesis**

The 24-register producer warpgroup appeared to spill its persistent K phase
counter. Because the kernel contract makes the number of 224-wide K tiles a
multiple of four, the K buffer and barrier phase can be derived from `n_idx`
without carrying `gi_kp` across persistent tasks.

**Action**

- removed the producer `gi_kp` scalar;
- selected the K buffer and empty-barrier phase from `n_idx`;
- left V production and both consumer warpgroup schedules unchanged.

**Gate result**

- dequantized-reference smoke: `1 passed`;
- S3584 H64/Hkv8 FP16: `0.661659 ms` versus Round 022 `0.666609 ms`,
  a `0.7%` improvement; FA3 measured `0.530045 ms`;
- focused NCU local loads increased from `458752` to `573440`;
- local stores remained `237072`, and dynamic instructions increased slightly.

**Decision**

Rejected. The small timing movement is not supported by the lowering evidence:
deriving the phase from `n_idx` causes more local reloads under the constrained
producer register budget. The accepted implementation remains Round 022.

## Round 024: Launch K Before Completing V Transpose

**Hypothesis**

FA3 launches the next K TMA before transposing the previously loaded V tile.
Moving the independent K launch between the TileOps V TMA launch and V
wait/transpose could hide K latency behind the producer-side transpose.

**Action**

- launched V TMA as before;
- acquired and launched the current K TMA before waiting for V completion;
- completed the V transpose and published `v_full` afterward;
- left all consumer code unchanged.

**Gate result**

- dequantized-reference smoke: `1 passed`;
- S3584 H64/Hkv8 FP16 regressed from Round 022 `0.666609 ms` to
  `0.800958 ms`; FA3 measured `0.530934 ms`.

**Decision**

Rejected. The producer blocks on `k_empty` while a completed V tile is waiting
to be transposed, delaying the PV consumer. FA3's ordering depends on a fuller
pipeline-state protocol and cannot be copied as a local statement reorder.
The accepted implementation remains Round 022.

## Round 025: Consumer Warpgroup Scheduler Token

**Hypothesis**

FA3 alternates its two consumer warpgroups through named scheduler barriers.
Giving each TileOps consumer a token before QK issue and passing it immediately
after issue could reduce Tensor Core issue-queue contention while preserving
the existing `PV(n-1) + QK(n)` overlap.

**Action**

- seeded a 256-thread named barrier for the first consumer warpgroup;
- alternated barrier IDs between the two consumers before each QK issue;
- passed the token to the peer warpgroup immediately after QK issue;
- kept softmax, PV issue, and producer pipelines unchanged.

**Gate result**

- dequantized-reference smoke: `1 passed`;
- S3584 H64/Hkv8 FP16 regressed from Round 022 `0.666609 ms` to
  `0.683065 ms`; FA3 measured `0.529283 ms`.

**Decision**

Rejected. FA3's scheduler token is coupled to a step that issues QK and PV
together. On the Round 022 schedule, adding the token only introduces another
wait and does not improve Tensor Core utilization. The accepted implementation
remains Round 022.

## Round 026: TileLang-Owned Register-P and RS-WGMMA

**Hypothesis**

If TileLang owns the FP8 register-A fragment, RS-WGMMA issue, waits, output
accumulator layout, and epilogue, the compiler may preserve the P operand
lifetime that failed across the fragment/extern boundary in Rounds 017-020.
Only the QK-accumulator-to-P layout conversion would remain a typed helper.

**Action**

- allocated explicit `[64, 224]` FP8 P fragments for both consumer warpgroups;
- derived and annotated the register-A fragment layout inferred by TileLang;
- replaced the fused raw-PTX PV helper with `T.wgmma_gemm`;
- moved output rescaling, normalization, and shared/global copies to TileLang;
- kept producer TMA, V transpose, barriers, QK issue, and persistent scheduling
  unchanged.

**Gate result**

- an unannotated probe failed layout inference because the standard TileLang
  RS-WGMMA output layout differs from the accepted raw-PTX accumulator ABI;
- after moving the complete output path to the inferred layout, direct
  `T.copy` from the QK accumulator to P exposed the expected QK/PV fragment
  layout conflict;
- a narrow typed conversion helper resolved both compile-time layout
  conflicts, and the candidate entered the S896 kernel;
- the S896 dequantized-reference smoke then failed the liveness gate: the
  kernel stayed resident without producing output and the named container was
  stopped.

**Decision**

Rejected. TileLang can infer and lower the FP8 RS-WGMMA path, but the current
register-P conversion plus asynchronous persistent-loop protocol still does
not establish a valid runtime scoreboard contract. The experiment confirms
that the remaining issue is not merely the raw PV issue helper. The accepted
implementation remains Round 022.

## Round 027: Source-Correlated Mainloop Profile

**Hypothesis**

The remaining 22-26% gap should be localized before another structural edit.
A same-shape source profile can distinguish producer latency, consumer
arithmetic, and scheduler overhead without inferring from aggregate duration.

**Action**

- collected Nsight Compute 2025.2.1 reports for Round 022 and FA3 at
  `B=1, S=3584, H=64, Hkv=8, D=128`, FP16 output;
- used the same `132 x 384` launch geometry and 1500 MHz H200 clock;
- compared dynamic instruction classes, scheduler state, source-correlated
  stalls, and the selected FA3 kernel template;
- mapped the largest TileOps barrier wait back to the generated producer
  control flow.

**Gate result**

- TileOps measured `659.040 us`; FA3 measured `533.664 us`;
- TileOps executed `202.45 M` instructions versus FA3 `183.21 M`;
- tensor-pipe active was `40.06%` versus `49.10%`;
- the largest TileOps stall, 5,766 long-scoreboard samples, is the producer
  waiting on `k_empty` after publishing V, not a consumer waiting for V;
- TileOps executes about `6.34 M` more `PRMT` instructions while both kernels
  execute the same number of FP8 conversion instructions;
- the 28 FP8x2-to-FP8x4 pack permutations per PV step account for about
  `6.42 M` dynamic instructions across the two consumers.

**Decision**

Diagnostic round. A deeper V producer pipeline is not the immediate answer:
the producer already catches the consumer and blocks on K-buffer reuse.
Target the redundant register-A pack permutation while preserving the accepted
raw-WGMMA layout and scoreboard protocol. The accepted implementation remains
Round 022.

## Round 028: CUTLASS FP8x4 Register-A Conversion

**Hypothesis**

Using CUTLASS's four-element FP32-to-E4M3 converter should lower the second
FP8x2 conversion with merge semantics and remove the 28 explicit P-register
pack permutations without changing the accumulator index map.

**Action**

- replaced two FP8x2 intrinsics plus shift/or packing with
  `NumericArrayConverter<float_e4m3_t, float, 4>`;
- preserved the accepted QK accumulator layout, source-index map, raw PV
  WGMMA helper, barriers, and persistent schedule;
- tested the dequantized-reference smoke before the S3584 performance gate.

**Gate result**

- S896 dequantized-reference smoke: `1 passed`;
- a fresh-cache S896 profile measured `36.51 us` versus `34.11 us` for the
  fresh Round 022 baseline, with identical `PRMT` and `F2FP` counts;
- a fresh-cache S3584 run failed the liveness gate: it remained resident at
  full GPU utilization for more than three minutes without producing output;
- the named container was stopped and the source was restored exactly to
  Round 022.

**Decision**

Rejected. As in the earlier typed-fragment probes, introducing a C++ array
conversion boundary changes register allocation or asynchronous lowering
enough to invalidate the long persistent loop, even though the short shape is
numerically correct. A follow-up may use a narrower inline conversion that
does not materialize `Array` temporaries. The accepted implementation remains
Round 022.

## Round 029: Inline FP8x4 Merge Conversion

**Hypothesis**

Inlining the exact four-scalar PTX conversion used by CUTLASS, without any
`Array` or fragment object, should preserve the outer register lifetime while
allowing the second FP8x2 conversion to merge into the same 32-bit P operand.

**Action**

- loaded the same four mapped QK-accumulator values as Round 022;
- emitted two `cvt.rn.satfinite.e4m3x2.f32` instructions and one `mov.b32`
  inside the existing pack helper;
- preserved every outer helper call, WGMMA issue, wait, barrier, and loop
  statement;
- ran the dequantized-reference smoke before the S3584 gate.

**Gate result**

- S896 dequantized-reference smoke: `1 passed`;
- a fresh-cache S896 profile measured `36.74 us` versus `34.11 us` for the
  fresh Round 022 baseline;
- the fresh artifact retained the same `412,160` PRMT and `458,752` F2FP
  instructions as Round 022, so the intended merge lowering did not occur;
- the earlier shared-cache S3584 liveness observation is excluded from the
  decision because the compiler artifact was not trustworthy;
- the source was restored exactly to Round 022.

**Decision**

Rejected. The inline rewrite did not reproduce FA3's merged conversion
lowering and regressed the fresh-cache S896 kernel by 7.7%. The accepted
implementation remains Round 022.

## Round 032: Compiler Cache-Key Audit

**Hypothesis**

Helper-only edits may not participate in the generated-kernel cache key. A
fresh compiler-artifact comparison is required before interpreting the
Round 028/029 profiles.

**Action**

- created a unique empty `/ci-cache/tilelang` mount for each candidate;
- rebuilt and profiled Round 022, Round 028, and Round 029 at S896;
- reran the Round 028 S3584 liveness gate with a fresh cache;
- compared duration, executed instructions, scheduler state, PRMT, and F2FP.

**Gate result**

| Candidate | Duration | Instructions | No eligible | PRMT | F2FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Round 022 | 34.11 us | 7,068,440 | 64.71% | 412,160 | 458,752 |
| Round 028 | 36.51 us | 7,064,890 | 66.58% | 412,160 | 458,752 |
| Round 029 | 36.74 us | 7,079,181 | 66.60% | 412,160 | 458,752 |

The old shared-cache profile measured 38.30 us and 469,504 PRMT instructions;
it was not the fresh Round 022 artifact.

**Decision**

Protocol correction accepted. Unique empty TileLang caches are mandatory for
all subsequent candidate compiles. Round 028 remains rejected on both
performance and fresh-cache liveness evidence. Round 029 is rejected on
fresh-cache lowering and performance evidence, without relying on its earlier
shared-cache long-shape observation. Full details are recorded in
`results/round032/round032_cache_audit.md`.

## Round 033: Fold Consumer K/V Phase Counters

**Hypothesis**

Each consumer maintains separate persistent K and V barrier counters even
though they identify the same tile at PV issue time. Folding them into one
counter may remove one high-frequency local spill without changing the
barrier phase sequence.

**Action**

- removed `gi_vc1` and `gi_vc2`;
- used the current K phase for the current V tile and the opposite phase when
  releasing the previous V tile;
- delayed the single counter increment until after PV issue;
- compiled every candidate shape from the isolated Round 033 cache.

**Gate result**

- official-runner correctness: `8 passed`;
- S896 FP16: `0.034728 ms`, effectively unchanged from Round 022;
- paired GPU0 S3584 FP16: `0.682017 ms` versus a freshly compiled Round 022
  baseline of `0.664097 ms`, a 2.7% regression;
- local loads/stores remained exactly `458,752 / 237,072`;
- dynamic instructions fell slightly to `201.03 M`, but the intended spill
  removal did not occur.

**Decision**

Rejected. The compiler preserved the same local-memory lifetime despite the
source-level counter fold, while the longer dependence through one counter
regressed the mainloop. The source was restored exactly to Round 022.

## Round 034: Fold Producer K/V Phase Counters

**Hypothesis**

Fresh source correlation maps the two dominant local stores to producer
updates of `gi_vp` and `gi_kp`, not to the consumer counters tested in Round
033. Within one work item, K production is one tile ahead of V production, and
the counters become equal again after the V tail. Encoding that relation in
one counter may remove the actual producer spill.

**Action**

- removed `gi_vp`;
- derived the previous V buffer, raw-V barrier phase, and V-empty phase from
  `gi_kp - 1`;
- retained the existing K load order, V transpose, and all barrier objects;
- compiled and tested from a new isolated Round 034 cache.

**Gate result**

- official-runner correctness: `8 passed`;
- the formal S896 benchmark failed the liveness gate: the kernel remained
  resident at 100% GPU utilization for more than two minutes;
- the named container was stopped before any longer-shape or profiler run.

**Decision**

Rejected. The within-work-item arithmetic relation is not sufficient to
reconstruct the producer's persistent barrier phase across grouped scheduler
work. Independent K and V producer counters are part of the live scheduler
contract. The source was restored exactly to Round 022.

## Round 035: Derive All K/V Phases From the Tile Index

**Hypothesis**

The kernel contract requires `seq_len` to be divisible by both 224 and 128, so
the producer loop length is always a multiple of four. All K/V double-buffer
phases therefore return to their initial value at each persistent work-item
boundary and can be derived from `n_idx` without persistent scalar counters.

**Action**

- removed the six producer and consumer K/V phase counters;
- derived K/V buffer selection, full-barrier phase, and empty-barrier phase
  from `n_idx` and the final `loop_range - 1` tile;
- retained the two independent Q consumer counters;
- compiled and tested from an isolated empty Round 035 cache.

**Gate result**

- official-runner correctness: `8 passed`;
- S896 FP16: `0.034963 ms`;
- paired GPU0 S3584 FP16: `0.682377 ms` versus the fresh Round 022 baseline
  of `0.664097 ms`, a 2.75% regression;
- S3584 local loads/stores fell from `458,752 / 237,072` to
  `336,896 / 121,856`;
- dynamic instructions fell from `202.45 M` to `200.35 M`;
- despite the lower spill traffic, eligible warps per scheduler fell from
  `0.59` to `0.55`, while no-eligible cycles rose from `58.69%` to `60.84%`.

**Decision**

Rejected. The phase counters were a real source of local-memory traffic, but
binding every barrier decision to the pipelined loop index lengthened the
producer control dependence and reduced scheduler eligibility. Lower
instruction and spill counts did not translate into lower latency.

## Round 036: Rebalance Dynamic Registers Around the Derived Phase

**Hypothesis**

Round 035 may spill the producer loop index because the producer warpgroup is
capped at 24 registers. Giving the producer eight more dynamic registers,
while reducing each consumer request by eight, may retain the spill reduction
without the scheduler regression.

**Action**

- first tested `dec_max_nreg(16)`, which the assembler rejected because
  `setmaxnreg.dec` accepts values in `[24, 256]`;
- corrected the direction to producer `dec_max_nreg(32)` and consumer
  `inc_max_nreg(232)`;
- retained the Round 035 phase derivation unchanged;
- compiled from a new isolated empty Round 036 cache.

**Gate result**

- targeted S896 FP16 correctness: `1 passed`;
- S896 FP16 latency: `0.036158 ms`, slower than both Round 035
  (`0.034963 ms`) and the accepted Round 022 range;
- the short-shape performance gate rejected the candidate before a formal
  long-shape/profile sweep.

**Decision**

Rejected. Dynamic register redistribution does not recover the scheduling
loss caused by the derived-phase dependency. The accepted implementation
remains Round 022.

## Round 037: Rebalance Registers With Explicit Phase Counters

**Hypothesis**

The accepted Round 022 producer counters may spill only because its producer
warpgroup is capped at 24 registers. Raising the producer cap to 32 while
lowering both consumer requests from 240 to 232 tests that possibility without
changing any phase or barrier dependency.

**Action**

- retained the exact accepted Round 022 producer, consumer, and barrier logic;
- changed only producer `dec_max_nreg(24) -> dec_max_nreg(32)`;
- changed both consumers `inc_max_nreg(240) -> inc_max_nreg(232)`;
- compiled from an isolated empty Round 037 cache.

**Gate result**

- targeted S896 FP16 correctness: `1 passed`;
- S896 FP16 latency: `0.037434 ms`, roughly 9% slower than the fresh accepted
  baseline.

**Decision**

Rejected at the short-shape gate. The dynamic register partition is itself
performance-sensitive; reallocating eight registers from each consumer does
not provide a useful producer-spill trade. The accepted implementation remains
Round 022.

## Round 038: Fully Unroll the K/V Tile Loops

**Hypothesis**

Because `seq_len` is a compile-time specialization and the K/V loop length is
known, explicitly unrolling the producer and both consumer loops may eliminate
the loop-index dependency and expose constant barrier phases to the compiler.

**Action**

- replaced the three `T.Pipelined(loop_range, num_stages=0)` loops with
  `T.unroll(loop_range)`;
- retained the accepted counters, barriers, WGMMA order, and register budget;
- compiled S896 from an isolated empty Round 038 cache.

**Gate result**

- compilation and launch took substantially longer than the accepted source;
- the S896 kernel remained resident at full GPU utilization for more than
  three minutes without returning;
- the named container was stopped and no longer-shape run was attempted.

**Decision**

Rejected on the liveness and compilation-cost gates. Explicit unrolling does
not preserve the persistent barrier/scoreboard contract even at the four-tile
S896 shape. The accepted implementation remains Round 022.

## Round 039: Same-GPU FA3 Gap and Counter-Store Diagnosis

**Hypothesis**

A same-GPU, same-image paired benchmark and profile can separate the remaining
FA3 gap into WGMMA utilization, scheduling, instruction, and spill components
before another structural edit.

**Action**

- moved the paired baseline to idle GPU1 after GPU0 was claimed by nightly CI;
- compiled Round 022 from an isolated empty cache in the official development
  runner;
- benchmarked TileOps and FA3 in the same process at S896 and S3584;
- collected full NCU reports for both S3584 kernels.

**Gate result**

| Metric | Round 022 | FA3 |
| --- | ---: | ---: |
| S896 FP16 | `0.034653 ms` | `0.028912 ms` |
| S3584 FP16 | `0.665510 ms` | `0.538036 ms` |
| NCU duration | `665.344 us` | `537.888 us` |
| dynamic instructions | `202.43 M` | `183.21 M` |
| local loads | `458,752` | `456,898` |
| local stores | `237,072` | `121,178` |
| tensor active | `39.67%` | `49.32%` |
| eligible warps / scheduler | `0.598` | `0.648` |
| registers / thread | `168` | `168` |

**Decision**

Diagnostic round. Register count and local loads are already comparable.
TileOps has about `116 K` excess local stores, lower scheduler eligibility,
and lower Tensor Core occupancy. The excess stores match the producer phase
counter stores removed by Round 035, so the next probe should preserve the
explicit barrier contract while changing the storage of those uniform states.

## Round 040: Warp-Shared Producer Phase Counters

**Hypothesis**

The producer K/V phases are warp-uniform. Storing one shared counter per
producer warp instead of one spilled scalar per thread may remove the excess
local stores while retaining the accepted explicit phase control flow.

**Action**

- replaced `gi_kp` and `gi_vp` with two four-element shared arrays;
- indexed each array by producer warp;
- retained all original phase expressions, barriers, and update points;
- compiled and profiled from an isolated empty Round 040 cache.

**Gate result**

- targeted S896 FP16 correctness: `1 passed`;
- S896 FP16: `0.034862 ms`;
- S3584 FP16: `0.677190 ms` versus paired Round 022 `0.665510 ms`;
- local loads/stores fell to `14,560 / 7,696`;
- dynamic instructions rose to `204.16 M`;
- eligible warps per scheduler fell to `0.587`.

**Decision**

Rejected. Shared storage removes almost all local spill traffic, proving the
counter diagnosis, but repeated shared loads in branch expressions cost more
than the removed local traffic.

## Round 041: Hoist Shared Phases Into Short-Lived Scalars

**Hypothesis**

Loading each warp-shared phase once per tile into a short-lived scalar and
writing it back once should preserve Round 040's spill reduction while
removing repeated shared-memory expressions.

**Action**

- hoisted V and K phases into per-iteration `T.alloc_var` scalars;
- hoisted the V-tail phase once outside the main loop;
- retained the Round 040 shared arrays and original barrier logic.

**Gate result**

- targeted S896 FP16 correctness: `1 passed`;
- S896 FP16: `0.034946 ms`;
- S3584 FP16: `0.672305 ms`, better than Round 040 but still about 1.0% slower
  than paired Round 022.

**Decision**

Rejected. The repeated-load cost was real, but a shared counter update per
tile remains slower than the accepted spilled-register control path. The
next probe will exploit the stronger four-tile phase-reset invariant rather
than preserving full counters.

## Round 042: Compress Producer State to One Warm-Up Bit

**Hypothesis**

The public kernel contract requires `seq_len` to be divisible by both 224 and
128, so `loop_range = seq_len / 224` is a multiple of four. Both K and V
double-buffer phases therefore reset at every persistent work-item boundary.
The only cross-work-item producer state is whether the first two V buffers
have ever been populated.

**Action**

- removed the persistent per-thread K and V phase counters;
- derived K/V buffer and barrier phases from `n_idx`;
- retained one shared warm-up bit per producer warp so only the first work
  item skips the initial V-empty waits;
- preserved the accepted QK/PV issue order, consumer counters, register
  partition, output epilogue, and public ABI;
- compiled every candidate shape from an isolated empty Round 042 cache.

**Gate result**

- official-runner correctness: `8 passed`;
- S3584 local stores fell from `237,072` to `129,024`, close to FA3's
  `121,178`;
- S3584 eligible warps per scheduler improved from `0.598` to `0.603`;
- S3584 tensor-pipe active improved from `39.67%` to `40.33%`;
- paired GPU1 FP16 comparison against a fresh Round 022 worktree:

| Shape | Round 022 | Round 042 | Change |
| --- | ---: | ---: | ---: |
| S896 | `0.034653 ms` | `0.034937 ms` | `+0.82%` |
| S1792 | `0.109168 ms` | `0.108229 ms` | `-0.86%` |
| S3584 | `0.665510 ms` | `0.653060 ms` | `-1.87%` |
| S7168 | `2.557244 ms` | `2.495464 ms` | `-2.42%` |

The isolated candidate/FA3 surface was:

| Shape | Round 042 | FA3 | Round 042 / FA3 |
| --- | ---: | ---: | ---: |
| S896 FP16 | `0.034937 ms` | `0.028867 ms` | `1.210x` |
| S896 BF16 | `0.035024 ms` | `0.028874 ms` | `1.213x` |
| S1792 FP16 | `0.108229 ms` | `0.087929 ms` | `1.231x` |
| S1792 BF16 | `0.108107 ms` | `0.088072 ms` | `1.228x` |
| S3584 FP16 | `0.653060 ms` | `0.538542 ms` | `1.213x` |
| S3584 BF16 | `0.654170 ms` | `0.537248 ms` | `1.218x` |
| S7168 FP16 | `2.495464 ms` | `2.046073 ms` | `1.220x` |
| S7168 BF16 | `2.498358 ms` | `2.049279 ms` | `1.219x` |

**Decision**

Accepted as the new baseline. The sub-1% S896 regression is within the
predeclared negligible-regression budget, while every longer shape improves
and the mechanism removes the measured excess producer stores. The remaining
roughly 21-23% gap to FA3 still requires better consumer issue overlap rather
than more phase-counter tuning.

## Round 043: Materialize an Independent Register P Fragment

**Hypothesis**

The earlier independent-P probes predated the isolated-cache protocol. Keeping
the accepted schedule while materializing 28 FP8 P registers per consumer can
revalidate the representation and liveness boundary before reordering WGMMA.

**Action**

- split score-to-P packing from the RS-WGMMA issue helper;
- allocated 28 per-thread `uint32` P registers for each consumer;
- added a typed helper that issues PV directly from those registers;
- retained the exact Round 042 QK/softmax/PV ordering and barriers.

**Gate result**

- targeted S896 FP16 correctness: `1 passed`;
- S896 FP16: `0.035043 ms` versus Round 042 `0.034937 ms`;
- S3584 FP16: `0.656588 ms` versus Round 042 `0.653060 ms`;
- unlike the old pre-audit probes, S3584 completed without a liveness failure.

**Decision**

Representation gate passed, performance candidate rejected. An independent P
fragment is viable in the current compiler/runtime, with about 0.3-0.5%
overhead before any overlap benefit.

## Round 044: True Softmax/PV Overlap With Independent P

**Hypothesis**

With a stable independent P fragment, the FA3-style steady state should expose
PV latency behind the next softmax:
`QK(n) -> rescale O -> PV(n-1) -> wait<1> -> softmax(n) -> wait<0> -> pack P(n)`.

**Action**

- handled the first P and final PV separately;
- issued current QK before previous PV;
- waited for QK with one WGMMA group left outstanding;
- ran current softmax while previous PV remained outstanding;
- waited for PV before overwriting P and releasing V.

**Gate result**

- targeted S896 FP16 correctness: `1 passed`;
- S896 FP16: `0.040782 ms`;
- S3584 FP16: `0.838896 ms`;
- S3584 dynamic instructions and local traffic remained near Round 042;
- eligible warps per scheduler collapsed from `0.603` to `0.450`;
- tensor-pipe active fell from `40.33%` to `32.36%`.

**Decision**

Rejected. Generated commit/wait ordering matched the source, but the
cross-iteration P dependency and outstanding RS-WGMMA interaction created a
much longer scoreboard stall. This source-level ordering does not reproduce
FA3's compiler-visible fragment/pipeline contract.

## Round 045: Complete QK/PV Before Softmax

**Hypothesis**

If Round 044's regression is specifically caused by running TileLang softmax
while RS-WGMMA is outstanding, waiting for both QK and PV before softmax should
recover the independent-P baseline.

**Action**

- retained QK-then-PV issue order and the cross-iteration P fragment;
- changed the steady-state wait to `wait_wgmma<0>`;
- released K/V before running softmax;
- removed the second post-softmax WGMMA wait.

**Gate result**

- targeted S896 FP16 correctness: `1 passed`;
- S896 FP16: `0.041489 ms`, slower than Round 044.

**Decision**

Rejected at the short-shape gate. Outstanding-PV softmax overlap is not the
sole source of the regression; the cross-iteration P pipeline itself lowers
poorly in this form. The accepted implementation remains Round 042.

## Round 046: Rescale the Output Accumulator From the Row Fragment

**Hypothesis**

The softmax row scale already exists as a two-element register fragment per
thread. Broadcasting those values with warp shuffles should remove the
register-to-shared-to-register round trip before PV without changing the
accepted Round 042 schedule.

**Action**

- removed the two 64-element shared scale arrays;
- removed the two local-to-shared scale copies;
- added a fragment-layout-aware helper that broadcasts each row scale from
  the owning lane and applies it directly to the output accumulator;
- retained Round 042's producer state, WGMMA issue order, barriers, and output
  epilogue;
- compiled the candidate from isolated empty caches.

**Gate result**

- official-runner correctness suite: `8 passed`;
- S896 FP16: `0.033526 ms` versus Round 042 `0.034937 ms`;
- S896 BF16: `0.033582 ms` versus Round 042 `0.035024 ms`;
- S3584 FP16: `0.635262 ms` versus Round 042 `0.653060 ms`;
- S3584 NCU dynamic instructions fell from `202.454 M` to `199.764 M`;
- S3584 tensor-pipe active rose from `40.33%` to `41.34%`;
- S1792 FP16 did not complete while holding GPU utilization at 100% for more
  than three minutes;
- the S1792 failure reproduced from a second independent empty TileLang cache.

The two archived Round 046 JSONL files are a partial short-shape diagnostic,
not a completed benchmark surface.

**Decision**

Rejected. The direct row-fragment rescale is a real performance improvement
on the shapes that complete, but it violates the required S1792/H32 liveness
contract. The accepted implementation remains Round 042. This round also
exposed a gate gap: the general correctness suite does not instantiate the
exact S1792/H32 manifest shape, so subsequent candidates must run a
manifest-shape completion probe before the formal timing surface.

## Round 047: Restore a Warpgroup Convergence Point

**Hypothesis**

The removed fragment-to-shared copy may have supplied a warpgroup convergence
point in addition to moving data. A 128-thread named-barrier sync after online
softmax should preserve direct row-fragment broadcasting while restoring that
execution contract.

**Action**

- retained Round 046's direct register-fragment row-scale helper;
- added a separate 128-thread named-barrier sync for each consumer warpgroup
  immediately after online softmax;
- compiled the candidate from an isolated empty cache.

**Gate result**

- S896 FP16 completed at `0.033354 ms`;
- S1792 FP16 completed at `0.101739 ms`, versus Round 042 `0.108229 ms`;
- FA3 measured `0.028893 ms` and `0.088064 ms` in the same processes;
- S3584 FP16 did not complete while holding GPU utilization at 100% for more
  than three minutes.

**Decision**

Rejected. Restoring a convergence point fixes the H32 liveness failure and
preserves the performance gain, but applying it unconditionally breaks the H64
persistent workload. The synchronization requirement is therefore tied to the
compiled dispatch shape rather than being a universal replacement for the
shared-memory path.

## Round 048: Compile-Time H32/H64 Synchronization Split

**Hypothesis**

Round 046 completed on H64 without an explicit convergence point, while Round
047 completed on H32 with one. Selecting the sync behavior at specialization
time may match the different persistent head-group workloads.

**Action**

- retained direct row-fragment scale broadcasting;
- emitted the 128-thread consumer sync only for `heads == 32`;
- required every manifest shape to complete before accepting formal timing;
- used a per-process 180-second timeout for the formal surface.

**Gate result**

- quick completion probes passed:
  - S896 FP16: `0.033354 ms`;
  - S1792 FP16: `0.101739 ms`;
  - S3584 FP16: `0.632158 ms`;
  - S7168 FP16: `2.434213 ms`;
- official-runner correctness suite: `8 passed`;
- formal S896 FP16/BF16 measured `0.033302 / 0.033285 ms`;
- formal S1792 FP16 timed out after 180 seconds under
  `warmup=5, repeat=20, trials=3`.

**Decision**

Rejected. Compile-time shape selection made short completion probes pass, but
the H32 path remained nondeterministically unsafe under the formal repeated
launch contract. The direct register broadcast lacks a synchronization or
visibility property provided by the accepted shared-scale path; timing-based
or shape-based barriers are not a valid substitute. The accepted
implementation remains Round 042.
