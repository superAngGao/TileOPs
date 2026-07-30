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
