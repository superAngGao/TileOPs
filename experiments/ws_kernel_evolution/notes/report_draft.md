# WS Kernel Evolution: From Schedule Redesign To Hardware-Facing Refinements

## 1. Introduction

Grouped Query Attention (GQA) is now a central building block in large language
model inference. It reduces the `K/V` footprint relative to full multi-head
attention, improves serving efficiency at long context, and therefore appears
throughout modern prefill and decode workloads. Because GQA sits directly on
the critical path of end-to-end latency, its kernel efficiency matters not only
for isolated microbenchmarks but for practical AI systems.

At the same time, optimizing GQA well on Hopper is unusually difficult.
Hopper exposes a powerful but highly specific warp-specialized (WS) programming
regime built around warpgroup-level tensor-core execution, explicit
producer-consumer orchestration, and tight interaction among `TMA`, `WGMMA`,
shared memory, and register allocation. These mechanisms create large
performance opportunities, but they also make kernel behavior much more
sensitive to schedule shape, locality policy, synchronization placement, and
compiler-lowered register flow than in more conventional CTA-level designs.

This makes Hopper GQA optimization a particularly interesting systems problem.
The challenge is not only to map the attention equations onto tensor cores, but
to construct a WS execution schedule that matches the causal workload, feeds the
hardware efficiently, and avoids losing throughput to memory-system or register
side effects. FlashAttention-3 (FA3) provides the most important prior point of
reference here: its Hopper results show that high-performance attention depends
on a carefully designed warp-specialized schedule rather than on arithmetic
formulation alone. However, reproducing that level of performance inside a
different operator stack still requires turning those high-level ideas into a
concrete kernel design and then understanding which remaining gaps come from
which hardware-facing causes.

This report studies that problem in the `TileOPs` operator library, using
`TileLang` as the kernel construction framework. Following the design direction
suggested by FA3, we first build a Hopper warp-specialized GQA forward kernel
in TileOPs. We then go beyond the initial WS schedule and show that the next
stage of optimization is unlocked by the causal workload itself: because causal
attention leaves real freedom in traversal and handoff order, it exposes a
schedule space that can be used to improve both `L2` behavior and register
usage. Starting from that observation, this report develops a systematic
optimization path with three layers:

1. `Single-CTA WGMMA Pipeline -> Baseline WS Pipeline` is a schedule redesign.
2. `Baseline WS Pipeline -> KV-Locality Reorder` is an `L2`-locality
   refinement.
3. `KV-Locality Reorder -> Post-wait0 Delayed Rescale` is a register-flow
   refinement.

The broader claim of this report is not limited to one kernel revision on one
GPU. For compute-bound operators, especially those that depend strongly on
tensor-core utilization, performance is often determined by how much schedule
freedom can be extracted from the workload and then translated into
hardware-compatible locality and register behavior. In that sense, the methods
documented here are relevant beyond Hopper itself. They should also be useful
for future tensor-core-dominated architectures, including platforms such as
Blackwell, where the exact instructions may change but the need for coordinated
schedule, memory, and register design remains.

## 2. End-To-End Results Vs FA3

Before analyzing mechanisms, we first summarize the end-to-end outcome on a set
of production-prefill shapes measured against FA3 on the same GPU. This table
is not meant to replace the later causal analysis; rather, it establishes the
practical performance envelope that the rest of the report aims to explain. For
the final causal entry, we report the best available anchor strategy per shape:
either the original paired anchor or the newer single-tile outer-scheduler
variant.

| Shape | Base ms | Reorder ms | Best Anchor ms | Anchor Variant | FA3 ms | Best Anchor TFLOPS | FA3 TFLOPS | Best Anchor / FA3 TF |
| --- | ---: | ---: | ---: | :-- | ---: | ---: | ---: | ---: |
| llama8b-4k | 0.3165 | 0.3099 | 0.2665 | paired | 0.2758 | 515.8 | 498.3 | 103.5% |
| llama8b-8k | 1.0321 | 0.9938 | 0.9151 | paired | 0.8371 | 600.7 | 656.7 | 91.5% |
| llama8b-32k | 17.0063 | 16.2642 | 14.3983 | single-tile | 12.9126 | 610.9 | 681.2 | 89.7% |
| llama8b-128k | 273.6086 | 267.2308 | 254.9966 | paired | 216.1482 | 551.9 | 651.1 | 84.8% |
| llama8b-256k | 1130.9110 | 1081.5928 | 1033.3350 | paired | 873.7935 | 544.8 | 644.3 | 84.5% |
| llama70b-4k | 0.5636 | 0.5575 | 0.4863 | paired | 0.4680 | 565.2 | 587.3 | 96.2% |
| llama405b-4k | 1.0463 | 1.0164 | 0.9440 | paired | 0.8598 | 582.4 | 639.4 | 91.1% |

Several observations are immediate.

- At `4k`, the final anchor-style kernel is already in the FA3 neighborhood,
  and on `llama8b-4k` it is slightly faster in elapsed time on this
  measurement set.
- The first WS milestone is the dominant step-change. Later milestones matter,
  but they build on top of a new execution organization rather than rescuing an
  already-good kernel.
- At longer contexts, the path still helps materially, but it does not fully
  close the remaining throughput gap to FA3. That is exactly why the later
  sections separate schedule, memory-system, and register-flow effects instead
  of treating them as one undifferentiated story.

The `Best Anchor` column is intentionally dispatch-oriented. For the
`llama8b` causal rows, it takes the better of the paired and single-tile
anchor strategies from a follow-up paired-vs-single sweep. For the larger-model
`4k` rows, only the paired anchor has been measured so far, so `Best Anchor`
remains the paired result there.

The `llama8b-256k` row was added in a follow-up run under the same measurement
setup as the rest of the table, using
[20260416_gqa_milestones_256k_gpu1.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260416_gqa_milestones_256k_gpu1.json).

## 3. Problem Setup And Measurement

We evaluate the evolution path on one representative causal analysis point,
`B=4, S=4096, H=64, Hkv=8, D=128`, and we also track end-to-end prefill
performance on production-aligned shapes. All milestone comparisons are taken on
the same GPU and under the same environment so that schedule effects,
locality effects, and codegen effects can be compared directly rather than
through anecdotal profiler screenshots.

The production-prefill sweep already indicates the overall trend, but the core
goal of this report is explanatory rather than merely comparative. The detailed
supporting numbers are collected in
[milestone_summary_table.md](/home/ga/TileOPs/experiments/ws_kernel_evolution/notes/milestone_summary_table.md).

The measurement discipline matters for the later mechanistic argument. We do
not only compare end-to-end times. We also compare tensor-pipe utilization,
cycle-level timeline splits, and generated CUDA / `ptxas` / SASS. That extra
instrumentation is what lets us distinguish "the kernel does less work" from
"the kernel performs the same work with a better schedule."

## 4. Single-CTA Baseline

The pre-WS baseline for this study is the existing Hopper-oriented single-CTA
WGMMA pipeline, which we refer to as the `Single-CTA WGMMA Pipeline`. This
baseline already uses WGMMA and software pipelining, so it should be viewed as
a competent predecessor rather than as an artificially weak comparison point.

Its structural limitation is that one CTA still owns the entire local loop over
`K/V` tiles. There is no explicit producer warp group, no consumer handoff, and
no stable two-consumer ping-pong. As a result, Tensor Core activity is gated by
the progress of one CTA-local execution path. That execution model is shown in
[pre_pr871_schematic.png](/home/ga/TileOPs/experiments/ws_kernel_evolution/figures/pre_pr871_schematic.png).

This distinction matters for the interpretation of the entire report. The later
milestones are organized around explicit producer / consumer handoff between
specialized warp groups, whereas the pre-WS baseline is better understood as a
single-CTA software pipeline with only local overlap across loop iterations.
Accordingly, it serves as the correct reference point for identifying which
benefits come specifically from warp specialization.

## 5. Baseline WS As A Schedule Win

The first major gain comes from replacing the single-CTA structure with an
explicit warp-specialized schedule. This `Baseline WS Pipeline` is the primary
structural transition in the optimization path. It is not best understood as an
instruction-level cleanup; rather, it changes how data movement is assigned,
how consumption is partitioned, and how Tensor Core work is phased across the
CTA.

Concretely, the baseline WS milestone introduces three structural changes
relative to the single-CTA baseline. First, `K/V` movement is pulled into a
dedicated producer warp group instead of remaining fused with the consumer body.
Second, the consumer path is split into two warp groups, `WG1` and `WG2`, with
an explicit scheduler handoff between them so that one consumer can release the
next while the producer is already feeding future buffers. Third, the kernel
switches to the persistent WS execution style used in this study, so the same
resident CTA keeps stepping through macro-tiles instead of rebuilding the local
pipeline around a single CTA-owned loop body every time. Those are visible
execution-model changes, not local cleanups inside an otherwise unchanged loop.

At a high level, the schedule adopts the same class of information-flow idea
emphasized by FlashAttention-3: one producer warp group feeds `K/V`, while two
consumer warp groups alternate their Tensor Core work. That split matters
because it decouples data movement from Tensor Core issue. In the pre-WS
kernel, those responsibilities remain chained behind one CTA-local loop. In the
WS kernel, they are explicitly staged and handed off between specialized warp
groups. The steady-state picture is illustrated in
[ws_two_wg_schematic.png](/home/ga/TileOPs/experiments/ws_kernel_evolution/figures/ws_two_wg_schematic.png), and the full-cycle organization in
[ws_three_kernel_full_cycle.png](/home/ga/TileOPs/experiments/ws_kernel_evolution/figures/ws_three_kernel_full_cycle.png).

The cleanest evidence that this is a schedule win comes from the tensor-pipe
metrics. Across the pre-WS and WS milestones, the amount of Tensor Core work is
essentially unchanged: the GMMA instruction count and tensor cycles active are
the same. What changes is elapsed packing. Tensor pipe utilization rises from
`35.2%` in the pre-WS kernel to `60.0%` in the baseline WS kernel, even though
the kernel is doing the same GMMA work. That is exactly what a schedule win
should look like: the kernel is not doing less Tensor Core math, it is feeding
and phasing that math more effectively. Those measurements are summarized in
[milestone_summary_table.md](/home/ga/TileOPs/experiments/ws_kernel_evolution/notes/milestone_summary_table.md).

This interpretation also matches the size of the end-to-end gain. A
`30% ~ 40%` latency improvement is difficult to explain as a minor cleanup in a
local code region. The more plausible explanation is that the work partition
itself changed. Once producer and consumers are separated, the kernel can
maintain a steadier Tensor Core rhythm than the single-CTA baseline can
sustain.

The claim should still be phrased carefully. The point is not that this kernel
reproduces FA3 in a strict one-to-one sense. The more precise statement is that
it adopts the same class of information-flow organization: explicit warp
specialization, explicit staging, and tighter Tensor Core packing.

## 6. Reorder As A Memory-System Win

The next milestone, `KV-Locality Reorder`, is a smaller but still real
improvement. What is interesting about this step is that its local steady-state
timing does not collapse in the same way that the first WS jump does. In the
fine-grained core split, `PR871 base` and `PR871 reorder` remain very close:

- `qk_issue`: `198 -> 200`
- `pv_issue`: `215 -> 216`
- `softmax_core`: `1309 -> 1313`

That makes sense once we state the source-level change precisely. Reorder does
not introduce a new producer/consumer decomposition. It keeps the same
`1 producer + 2 consumers` WS skeleton, the same basic scheduler protocol, and
nearly the same consumer-side core body. The main change is in how the
persistent kernel traverses work and therefore how it touches the `K/V`
footprint: the loop order becomes more `KV`-head-friendly, so neighboring
iterations reuse a more similar `K/V` working set instead of stressing the
memory system with a less locality-aware traversal. In other words, this step
changes the visitation order and memory footprint of the WS schedule more than
it changes the arithmetic body executed by each consumer.

Yet the measured total still improves from `2919` cycles to `2841` cycles, and
the production sweep also improves. That pattern makes a pure "better local
compute schedule" explanation too weak. If the local compute windows barely move
but end-to-end time improves, the kernel is likely benefiting from better
locality or cache behavior.

The new Nsight Compute L2 pass sharpens that interpretation, but also makes it
more precise. Reorder does **not** simply lower the L2 miss rate. On the
canonical shape, the measured L2 read lookup miss rate actually increases from
`0.0808` to `0.1145`. What drops instead is the total amount of read traffic:

- total L2 read lookups: `656k -> 388k`
- L2 read misses in absolute count: `53.0k -> 44.5k`
- DRAM read bytes: `8.04M -> 7.00M`

So the stronger statement is not "reorder has a better L2 hit ratio." It is
that reorder changes the memory-system footprint of the WS schedule. It appears
to require fewer L2 read lookups and fewer DRAM reads overall, even though the
remaining read stream does not have a lower miss ratio. That is still a
locality- or memory-system-facing win, but it is more accurately described as a
traffic-shaping win than as a naive miss-rate win. Those results are collected
in [l2_profile_findings.md](/home/ga/TileOPs/experiments/ws_kernel_evolution/notes/l2_profile_findings.md), alongside the broader milestone measurements in
[milestone_summary_table.md](/home/ga/TileOPs/experiments/ws_kernel_evolution/notes/milestone_summary_table.md). The visual context remains
[milestone_stitched_timelines.png](/home/ga/TileOPs/experiments/ws_kernel_evolution/figures/milestone_stitched_timelines.png).

This prevents us from telling one vague story for all improvements. The first WS
jump is mainly a schedule story. The reorder step is mainly a memory-system /
locality story.

Just as importantly, reorder also clarifies what it does **not** fix. Once the
memory-system footprint is improved, the remaining gap to the final anchor-style
kernel is no longer well explained by access ordering alone. At that point the
most exposed bottleneck shifts back inside the consumer hot window itself, which
is exactly where the delayed-rescale change operates.

## 7. Delayed Rescale As A Register-Flow Win

With that in mind, the final major gain is best read as a second-stage cleanup
inside an already-good WS schedule. `Post-wait0 Delayed Rescale` does not
primarily change the producer/consumer organization, and it is not mainly a
memory-footprint improvement. Instead, it changes how work is staged around the
consumer hot window. Superficially, that result looks puzzling because some
measured launch-side windows become much larger:

- `qk_issue`: `200 -> 1066`
- `pv_issue`: `216 -> 941`

Here again the concrete source-level modification matters. In the reorder
kernel, the consumer body still performs `rescale(acc_o)` before `PV`, so the
steady-state ordering is effectively `QK -> rescale(acc_o) -> wait(v_full) -> PV
-> wait1 -> preamble/softmax -> wait0`. The delayed-rescale variant removes that
old-output rescale from the `pre-PV` region and moves it to after `wait0`,
making the critical region read more like `QK -> wait(v_full) -> PV -> wait1 ->
preamble/softmax -> wait0 -> rescale(acc_o) -> copy`. The full anchor kernel
keeps that same post-`wait0` rescale placement and pairs it with anchor-specific
wait placement and handoff cleanup, but the isolated delayed-rescale-only
experiment shows that the rescale move is already the dominant source-level
change.

At the same time, the kernel still gets faster overall:

- `reorder`: `2841` cycles
- `reorder + delayed rescale only`: `2716` cycles
- `anchor`: `2668` cycles

If we read those `issue` windows too literally, the result seems contradictory.
The key point is that delayed rescale does not make the `QK` or `PV` arithmetic
intrinsically cheaper. What it changes is the register-flow picture around the
hottest part of the loop: the region spanning `pre-PV`, `wait1`, and softmax.

In the reorder kernel, the old output path `acc_o *= ss` still lives before
`PV`. That means one crowded region has to carry the current `acc_s` path, the
softmax state, the casted `acc_s` path consumed by `PV`, and the old output
accumulator path at the same time. In the delayed-rescale variant, that `acc_o`
work is moved to after `wait0`. The code reflects that change directly: the
rescale loop appears before `PV` in
[bench_timeline_pr871_core_split.py](/home/ga/TileOPs/experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_core_split.py), but after `wait0` in
[bench_timeline_pr871_reorder_delayed_rescale_core_split.py](/home/ga/TileOPs/experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_reorder_delayed_rescale_core_split.py) and
[bench_timeline_anchor_causal_core_split.py](/home/ga/TileOPs/experiments/ws_kernel_evolution/scripts/bench_timeline_anchor_causal_core_split.py).

That source-level shift is already enough to reproduce most of the anchor
behavior. The delayed-rescale-only experiment isolates the variable: without
adding the anchor waits or changing the handoff mechanism, moving rescale alone
reduces `softmax_core` from `1313` to `1056`, very close to anchor's `1030`, and
improves the measured total from `2841` to `2716`. Those results are summarized in
[milestone_summary_table.md](/home/ga/TileOPs/experiments/ws_kernel_evolution/notes/milestone_summary_table.md).

The codegen evidence then makes the mechanism concrete. At the generated CUDA
level, the `QK` loops still look broadly similar across `reorder`,
`delayed-rescale-only`, and `anchor`: descriptor setup, `warpgroup_arrive()`,
eight `wgmma_ss(...)` operations, and `warpgroup_commit_batch()`. But the
generated SASS is not similar. In `reorder`, `QK` lowers to a relatively compact
burst of `HGMMA` instructions. In `delayed-rescale-only` and `anchor`, almost
every `HGMMA` is interleaved with `WARPGROUP.DEPBAR + ARRIVE`. That evidence is
collected in
[register_flow_codegen_findings.md](/home/ga/TileOPs/experiments/ws_kernel_evolution/notes/register_flow_codegen_findings.md).

The `ptxas` diagnostics point in the same direction. `reorder` gets
compiler-injected `warpgroup.wait` and spills, while `delayed-rescale-only` and
`anchor` instead expose WGMMA serialization hazards tied to accumulator access.
So the most defensible reading is not "the compiler was hacked" and not "the
hardware alone did this by magic." The stronger explanation is a coupled one:
the source-level register-flow change causes the compiler to lower the same
logical `QK/PV` stages into a different Hopper `WARPGROUP/HGMMA` schedule, and
that new schedule interacts differently with real accumulator-pipeline
constraints.

This also resolves the apparent paradox of longer local `QK/PV` windows but
better steady-state throughput. The cost does not disappear; it moves. `reorder`
appears to pay more in the downstream `wait1 / softmax-side` region through
spills and injected waits. Delayed rescale pays more in the launch-side
`QK/PV` windows, but in return it makes the downstream hot window much cleaner.
That trade is favorable because the pipeline does not expose every local timing
window equally on the end-to-end critical path. Delayed rescale is therefore
best understood as a register-flow win realized through changed codegen under
real Hopper WGMMA constraints, not as a cheaper arithmetic kernel.

## 8. A Causal-Specific Scheduler Choice: Paired Vs Single-Tile

The three-step mainline above explains how the final anchor kernel emerged, but
it does not fully explain the remaining long-sequence gap to FA3. That gap led
to one more causal-specific question: should the outer scheduler continue to use
the current paired causal work unit, or should it switch to a single-tile work
unit that more closely matches FA3's causal scheduler?

This is a causal-specific design choice because pairing is not arbitrary. The
current anchor kernel uses a paired outer work unit `(k, M-1-k)` to flatten the
causal triangle imbalance: each outer work item mixes one light tile from the
top of the triangle with one heavy tile from the bottom. That is a sensible
local balancing strategy, especially at short sequence lengths. But it also
makes the outer scheduler coarser. FA3 does not use this paired work unit. Its
causal scheduler issues single tiles, then relies on reverse-`m_block`
ordering, query-head-space sectioning, and dynamic persistent issuance to
recover load balance and locality.

To separate these two strategies, we built a single-tile outer-scheduler
variant that keeps the inner anchor compute body largely unchanged while
removing pairing from the outer work unit. The result is not a new mainline
kernel yet, but it is already useful as a design probe.

![Paired vs single-tile scheduler](../figures/pair_vs_single_scheduler_strategy.png)

The schematic above uses a small causal example to show the difference in
grouping and traversal. The paired strategy binds one light tile and one heavy
tile into the same outer work item, while the single-tile strategy lets the
outer scheduler issue one tile at a time in reverse `m_block` order. Both still
operate inside the same query-head section, but they expose very different
scheduler granularity to the outer policy.

| Shape | Paired Anchor ms | Single-Tile ms | Single / Paired |
| --- | ---: | ---: | ---: |
| llama8b-4k | 0.2665 | 0.3428 | 128.6% |
| llama8b-8k | 0.9151 | 1.0156 | 111.0% |
| llama8b-16k | 3.4973 | 3.4788 | 99.5% |
| llama8b-32k | 15.2967 | 14.3983 | 94.1% |
| llama8b-64k | 61.8354 | 60.4092 | 97.7% |
| llama8b-128k | 254.9966 | 256.7521 | 100.7% |
| llama8b-256k | 1033.3350 | 1034.1890 | 100.1% |

This table shows a clear crossover shape. Pairing is the right outer strategy
for small shapes, where the extra scheduler freedom of single-tile does not pay
for itself. Around `16k`, the two become nearly identical. At `32k-64k`,
single-tile becomes meaningfully better. And by `128k-256k`, the single-tile
outer scheduler remains competitive but does not yet deliver a decisive
end-to-end win on its own.

The Nsight Compute data makes that last point more precise. At `64k`, `128k`,
and `256k`, the single-tile outer scheduler consistently produces a more
FA3-like memory-system signature:

- lower L2 miss rate
- lower or comparable DRAM read
- higher total L2 bytes

So single-tile is not a dead end. It really does release some scheduler freedom
and improve cache behavior. But it is also not a complete explanation by
itself. In this report, that result is best read as a new causal-specific
design lesson rather than as a replacement for the earlier three-step story:

- pairing is still the right outer strategy for short contexts
- single-tile becomes the more interesting outer strategy once context grows
- but closing the residual long-sequence gap still requires more than just
  changing scheduler granularity

This also suggests a practical dispatch intuition for future kernels. A
conservative policy would keep paired scheduling below `32k` and only consider
single-tile outer scheduling at `32k+`. An exploratory policy could already
treat `16k` as a crossover region worth benchmarking, while still keeping
paired scheduling as the safer default there.

## 9. One Schedule Win, Then Two Hardware Wins, Plus One Causal Scheduler Choice

Taken together, the kernel evolution path is not one long blur of tuning. It has
a simple structure:

- `Single-CTA WGMMA Pipeline -> Baseline WS Pipeline`: schedule / information-flow win
- `Baseline WS Pipeline -> KV-Locality Reorder`: memory-system / locality win
- `KV-Locality Reorder -> Post-wait0 Delayed Rescale`: register-flow win
- `Paired -> Single-Tile (causal outer scheduler choice)`: scheduler-granularity tradeoff

That decomposition is useful because it gives each step a distinct mechanistic
role. The baseline WS kernel matters because it makes the producer / consumer
schedule explicit. The reorder step matters because it improves how that
schedule interacts with Hopper's memory hierarchy. The delayed-rescale step
matters because it improves how that schedule interacts with Hopper's register
and accumulator constraints. The paired-vs-single result matters because it
shows that causal kernels have one more outer scheduling degree of freedom
above the mainline itself.

This is also the most practical way to explain why the later milestones are
smaller than the first jump. Once the schedule is structurally correct, the
remaining work is no longer "find a better pipeline." It is "make this pipeline
fit the hardware more cleanly," and in the causal case, also "choose the right
outer scheduling granularity for the target context range."

## 10. Design Lessons

The experiments suggest a few practical design lessons for future WS kernel
work.

First, optimize exposed throughput, not just local window length. A measured
window getting longer is not automatically bad if it moves cost into a part of
the pipeline that is easier to hide. The delayed-rescale result is the clearest
example of this.

Second, treat `wait` as a stage boundary, not only as a correctness primitive.
`wait1` and `wait0` separate regions with different live values and different
accumulator pressure. That makes them natural cut points for scheduling heavy
register traffic. The main lesson from anchor is not simply "add more waits,"
but "use waits to keep different heavy data paths from colliding in the same hot
window."

Third, keep old-state repair work away from the current hot path whenever
possible. Moving `acc_o *= ss` out of the `pre-PV` region was effective because
it stopped the old output path from competing with the current `acc_s` and
softmax path at the worst possible moment.

Fourth, design kernels with a full source-to-codegen-to-hardware feedback loop.
The important unit of optimization is not just source code structure, and it is
not just SASS shape. The effective pattern is to reshape dataflow and live
ranges at the source level, inspect how the compiler lowers that structure, and
then judge whether the lowering fits the target hardware's pipeline behavior.

## 11. Failed Directions And Practical Constraints

Several side paths were useful precisely because they did **not** become the
main explanation of the milestone gains. Their value was in narrowing the search
space and clarifying what the winning kernels were not doing. The supporting
details are collected in
[failed_directions.md](/home/ga/TileOPs/experiments/ws_kernel_evolution/notes/failed_directions.md).

First, more aggressive overlap schemes did not become the central story. We
explored variants that tried to make overlap more explicit through full-overlap,
cooperative, or looser-wait structures, including
[_test_ws_fa3_full_overlap.py](/home/ga/TileOPs/_test_ws_fa3_full_overlap.py),
[_test_ws_fa3_cooperative.py](/home/ga/TileOPs/_test_ws_fa3_cooperative.py),
[_test_ws_fa3_v2_nowait.py](/home/ga/TileOPs/_test_ws_fa3_v2_nowait.py), and
[_test_ws_fa3_v2_tb_regflip.py](/home/ga/TileOPs/_test_ws_fa3_v2_tb_regflip.py).
Those experiments were useful for mapping the design space, but they did not
turn out to be the clean explanation for the measured milestone improvements.
The final path is not "more overlap at any cost." It is a more selective story
about schedule quality, memory-system footprint, and register-flow cleanup.

Second, zero-clear and `ScaleOut::Zero` were not a general solution in the
causal path. The causal anchor implementation explicitly records why the old
pre-fill mask approach breaks when `clear_accum=True` zeroes `acc_s` on the
first `ki`:
[_test_ws_fa3_v2_persistent_anchor_causal.py](/home/ga/TileOPs/_test_ws_fa3_v2_persistent_anchor_causal.py:8).
The working fix was not to zero more aggressively, but to move the mask to the
post-WGMMA point where the dataflow is already stable. The broader lesson is
that zeroing only helps when it respects both the kernel's semantics and the
compiler's WGMMA scheduling model.

Third, wait placement mattered more than simply "releasing earlier." Some
variants tried to make WG2 or the producer advance more aggressively, but the
high-value lesson was not that earlier release is always better. The stronger
lesson was that `wait1` and `wait0` define stage boundaries in the register-flow
picture. The final anchor-style gain came from using those boundaries to keep
heavy data paths from colliding in the same hot window, not from eliminating the
boundaries altogether.

Finally, some constraints were methodological rather than algorithmic. Fine-
grained in-loop timing for the pre-WS kernel still triggers a TileLang/TVM
`WgmmaSyncRewriter` crash in this environment, which is why the pre-WS node has
coarse timing but not the same internal cycle split as the WS milestones.
Likewise, the profiler runs had to be kept serial on the same GPU and bracketed
carefully with `cudaProfilerStart/Stop` to keep the measurements trustworthy.
Those constraints do not weaken the core argument, but they do explain why the
report uses different evidence types for different milestones.
