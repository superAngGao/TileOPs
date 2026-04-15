# WS Kernel Evolution Summary

This note collects the current milestone comparison into one place so we can
discuss the next steps before drawing final timeline figures.

## Scope

- Performance table:
  - source: [20260415_gqa_milestones_prefill_gpu1_merged.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_gqa_milestones_prefill_gpu1_merged.json)
  - additional pre-WS source: [20260415_gqa_pre_pr871_wgmma_gpu1.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_gqa_pre_pr871_wgmma_gpu1.json)
  - workload: production-prefill shapes aligned with TileOPs `bench_gqa.py`
  - measurement: `GPU 1`, serial run, `env_tilelang_20260119`
- Cycle-level table:
  - sources:
    - [20260415_timeline_probes_pre_pr871.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_timeline_probes_pre_pr871.json)
    - [20260415_timeline_probes_pr871_base.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_timeline_probes_pr871_base.json)
    - [20260415_timeline_probes_pr871_reorder.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_timeline_probes_pr871_reorder.json)
    - [20260415_timeline_probes_anchor_causal.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_timeline_probes_anchor_causal.json)
  - workload: `B=4, S=4096, H=64, Hkv=8, D=128, causal`
  - measurement: `GPU 1`, serial run, `env_tilelang_20260119`

## Production Performance

| Shape | Pre-PR871 ms / TF | Base ms / TF | Reorder ms / TF | Anchor ms / TF | FA3 ms / TF | Base vs Pre | Reorder vs Base | Anchor vs Reorder | Anchor / FA3 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama8b-4k | 0.5353 / 256.8 | 0.3165 / 434.3 | 0.3099 / 443.5 | 0.2630 / 522.6 | 0.2758 / 498.3 | -40.9% | -2.1% | -15.1% | 95.3% |
| llama8b-8k | 1.7506 / 314.0 | 1.0321 / 532.7 | 0.9938 / 553.2 | 0.9132 / 602.0 | 0.8371 / 656.7 | -41.0% | -3.7% | -8.1% | 109.1% |
| llama8b-32k | 24.3772 / 360.8 | 17.0063 / 517.2 | 16.2642 / 540.8 | 15.4076 / 570.9 | 12.9126 / 681.2 | -30.2% | -4.4% | -5.3% | 119.3% |
| llama8b-128k | 424.9382 / 331.2 | 273.6086 / 514.4 | 267.2308 / 526.7 | 259.1798 / 543.0 | 216.1482 / 651.1 | -35.6% | -2.3% | -3.0% | 119.9% |
| llama70b-4k | 0.9462 / 290.5 | 0.5636 / 487.7 | 0.5575 / 493.0 | 0.4863 / 565.2 | 0.4680 / 587.3 | -40.4% | -1.1% | -12.8% | 103.9% |
| llama405b-4k | 1.7308 / 317.6 | 1.0463 / 525.4 | 1.0164 / 540.9 | 0.9440 / 582.4 | 0.8598 / 639.4 | -39.5% | -2.9% | -7.1% | 109.8% |

## Single-CTA WGMMA Performance Reading

The pre-WS Hopper baseline is now pinned to `GqaFwdWgmmaPipelinedKernel`, which
is the natural "before PR 871" milestone for this study:

- it already uses Hopper-oriented WGMMA pipelining
- but it does not yet build the warp-specialized producer/consumer pipeline

The production-prefill sweep shows that PR 871 itself is a major step-change:

- `pr871_base` is about `30% ~ 41%` faster than the pre-WS WGMMA baseline
- `reorder` and `anchor` then stack additional gains on top of that new base

So the story is not only about later pipeline polishing. The first big jump is
from introducing the WS / persistent pipeline at all.

## Single-CTA WGMMA Timing Shape

The `pre_pr871` milestone, which is better presented to readers as
`Single-CTA WGMMA Pipeline`, cannot be drawn with the same vocabulary as the
later milestones.

The reason is structural:

- `pre_pr871` is a single-CTA software pipeline
- `pr871_base / reorder / anchor` are WS kernels with explicit producer /
  consumer handoff

So for `pre_pr871`, the right mental model is not:

- producer WG
- consumer WG1
- consumer WG2

Instead, it is:

- one CTA
- one pipelined loop over K/V tiles
- software overlap across loop iterations

The first coarse timing anchor we have is:

- `loop_body_total = 174390.0 cycles`
- `epilogue_total = 787.6 cycles`

for the representative last causal tile (`B=4, S=4096, H=64, Hkv=8, D=128`,
`block_m=128`, `block_n=128`, `head=0`, `batch=0`).

Important caveat:

- fine-grained clock instrumentation *inside* the `T.Pipelined + T.gemm` loop
  currently crashes the TileLang/TVM `WgmmaSyncRewriter` pass in this
  environment
- so `pre_pr871` currently has only coarse timing, not the same detailed split
  available for the WS milestones

This means the final blog figure should likely use a **different visual form**
for this node:

- `pre_pr871`: single-CTA software-pipeline diagram
- later milestones: multi-lane WS handoff timeline

## Tensor Pipe Utilization

We also profiled the canonical causal shape with Nsight Compute:

- shape: `B=4, S=4096, H=64, Hkv=8, D=128, causal`
- source: [20260415_ncu_tensor_pipe_milestones_4k.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_ncu_tensor_pipe_milestones_4k.json)
- profiler setup:
  - `cudaProfilerStart/Stop` brackets a single profiled invocation
  - `ncu` temporary files redirected under `.tmp/ncu`

The most useful utilization-style metric was:

- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`

| Milestone | Kernel time (ns) | Tensor pipe utilization (% peak elapsed) | Tensor cycles active (sum) | GMMA inst (sum) |
| --- | --- | --- | --- | --- |
| Pre-PR871 WGMMA | 3981920 | 35.2% | 1107296256 | 17301504 |
| PR871 base | 2347200 | 60.0% | 1107296256 | 17301504 |
| PR871 reorder | 2296160 | 61.4% | 1107296256 | 17301504 |
| Anchor causal | 2149312 | 65.4% | 1107296256 | 17301504 |

Immediate reading:

- the amount of GMMA work is essentially the same across these four kernels
- the WS kernels do not win by doing less Tensor Core math
- instead, they complete the same GMMA work in less time and with higher tensor
  pipe utilization during elapsed time

This makes the WS advantage much easier to explain visually:

- `pre_pr871` has lower Tensor pipe utilization despite doing the same GMMA work
- `pr871_base` is the first big jump
- `reorder` and `anchor` then continue to improve Tensor pipe packing

## Cycle Breakdown

The cycle table is grouped as:

- Front-end:
  - `wait(k_full)`
  - scheduler handoff
  - explicit `clear(acc_s)` cost when present
- Steady-state:
  - `wgmma_issue`
  - `wait<1>`
  - `softmax`
  - `wait<0>`
- Tail:
  - post-`wait<0>` work measured in a dedicated probe

| Milestone | Measured total | Front wait(k) | Front sched | Front clear | Front sum | WGMMA issue | wait<1> | softmax | wait<0> | Steady total | Tail detail | Tail sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PR871 base | 2919 | 486 | 66 | 14 | 566 | 660 | 57 | 1390 | 6 | 2113 | `v_empty=18`, `acc_s_cast_copy=116` | 134 |
| PR871 reorder | 2841 | 429 | 68 | 16 | 513 | 677 | 77 | 1419 | 16 | 2189 | `v_empty=19`, `acc_s_cast_copy=124` | 143 |
| Anchor causal | 2668 | 137 | 11 | 0 | 147 | 987 | 59 | 1435 | 29 | 2510 | `v_empty=32`, `delayed_rescale=87` | 120 |

## Current Reading

- `reorder` improves end-to-end performance, but its local steady-state region is not shorter than `base`.
  - This continues to support the locality/L2-reuse explanation.
- The directly measured steady-state total now matches the performance trend.
  - `2919 -> 2841 -> 2668` for `base -> reorder -> anchor`.
- `reorder` does improve the front-end.
  - `566 -> 513` cycles is real, but still far from `anchor`'s `147`.
- `anchor` wins despite a longer measured steady-state core.
  - Its main visible structural advantage is the much lighter front-end handoff.
- `anchor`'s delayed rescale is measurable but not huge by itself.
  - Tail split says `delayed_rescale ~= 87 cycles`.

## Disambiguated Core Split

To reduce ambiguity in the old merged `issue` / `softmax` labels, we added a
finer steady-state core split:

- `qk_issue`
- `rescale_before_pv`
- `wait_v_full`
- `pv_issue`
- `wait1_fence`
- `post_wait1_preamble`
- `softmax_core`
- `wait0_fence`

These values come from the same shape and environment as the cycle table above.

| Milestone | qk_issue | rescale_before_pv | wait_v_full | pv_issue | wait1_fence | post_wait1_preamble | softmax_core | wait0_fence | Core sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PR871 base | 198 | 140 | 100 | 215 | 58 | 81 | 1309 | 6 | 2106 |
| PR871 reorder | 200 | 137 | 98 | 216 | 59 | 83 | 1313 | 6 | 2111 |
| Anchor causal | 1079 | 0 | 82 | 937 | 31 | 78 | 1030 | 25 | 3262 |

Immediate reading from this finer split:

- `base` and `reorder` now look much less mysterious.
  - Their `qk_issue`, `pv_issue`, `softmax_core`, and `acc_s_copy` are all close.
  - This supports the view that `reorder` mainly wins via locality / reuse, not by changing the local compute schedule much.
- `acc_s_cast_copy` is now directly comparable across all three milestones.
  - `base=116`, `reorder=123`, `anchor=111`.
  - This no longer looks like a major structural differentiator.
- `softmax_core` is also now on a clearer footing.
  - `base=1309`, `reorder=1313`, `anchor=1030`.
  - The old ambiguity mostly came from `post_wait1_preamble` being mixed into the previous coarse `softmax` interval.
- `anchor` remains the most surprising case.
  - Its `qk_issue` and `pv_issue` windows are both much larger than the PR871 variants.
  - That suggests the old coarse `wgmma_issue` label was hiding a real schedule/codegen difference rather than only a naming problem.

## Still Missing

- `pre-PR871` milestone is still missing from the cycle table and the milestone summary.
- We do not yet have final polished timeline figures.
- We have not yet converted the issue notes into a single "dead ends / failed ideas" summary note for the blog.

## Delayed-Rescale-Only Check

We also ran a minimal hypothesis check on top of `pr871_reorder`:

- keep the original reorder kernel structure
- do **not** add anchor waits
- do **not** switch to named-barrier handoff
- only move `rescale(acc_o)` from before PV to after `wait<0>`

Scripts:

- [bench_timeline_pr871_reorder_delayed_rescale_core_split.py](/home/ga/TileOPs/experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_reorder_delayed_rescale_core_split.py)
- [bench_timeline_pr871_reorder_delayed_rescale_total_cycle.py](/home/ga/TileOPs/experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_reorder_delayed_rescale_total_cycle.py)

Measured on `GPU 1`, `B=4, S=4096, H=64, Hkv=8, D=128, causal`:

| Variant | qk_issue | rescale_before_pv | wait_v_full | pv_issue | wait1_fence | post_wait1_preamble | softmax_core | wait0_fence | Core sum | Measured total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PR871 reorder | 200 | 137 | 98 | 216 | 59 | 83 | 1313 | 6 | 2111 | 2841 |
| Reorder + delayed rescale only | 1066 | 0 | 78 | 941 | 19 | 78 | 1056 | 5 | 3242 | 2716 |
| Anchor causal | 1079 | 0 | 82 | 937 | 31 | 78 | 1030 | 25 | 3262 | 2668 |

Immediate reading:

- Moving `rescale` alone is enough to reproduce most of the big `softmax_core` drop.
  - `1313 -> 1056` cycles, which lands very close to anchor's `1030`.
- The same minimal change also reproduces the large launch-side windows.
  - `qk_issue / pv_issue` move from `200 / 216` to `1066 / 941`, again very close to anchor.
- The total steady-state cycle count also improves materially.
  - `2841 -> 2716`, closing most of the gap to anchor's `2668`.

This strongly supports the hypothesis that delayed `rescale` is a primary driver
of the surprising softmax-side cycle shift. The remaining gap to full anchor is
much smaller than the original `reorder -> anchor` gap, and can likely be
investigated next via anchor waits / named-barrier handoff / codegen differences.

## Codegen Evidence For The Delayed-Rescale Hypothesis

We also dumped generated CUDA / `ptxas` logs / SASS for:

- [reorder_core_split_codegen.cu](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/codegen_compare/reorder_core_split_codegen.cu)
- [reorder_delayed_rescale_core_split_codegen.cu](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/codegen_compare/reorder_delayed_rescale_core_split_codegen.cu)
- [reorder_core_split_codegen.ptxas.txt](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/codegen_compare/reorder_core_split_codegen.ptxas.txt)
- [reorder_delayed_rescale_core_split_codegen.ptxas.txt](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/codegen_compare/reorder_delayed_rescale_core_split_codegen.ptxas.txt)
- [reorder_core_split_codegen.sass](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/codegen_compare/reorder_core_split_codegen.sass)
- [reorder_delayed_rescale_core_split_codegen.sass](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/codegen_compare/reorder_delayed_rescale_core_split_codegen.sass)

Three points stand out.

1. `ptxas` resource usage changes in a meaningful way.

| Variant | Registers | Stack frame | Spill stores | Spill loads | Key `ptxas` note |
| --- | --- | --- | --- | --- | --- |
| PR871 reorder | 168 | 32 B | 36 B | 36 B | compiler injects `warpgroup.wait` to use GMMA-defined registers |
| Reorder + delayed rescale only | 168 | 0 B | 0 B | 0 B | WGMMA pipeline may serialize because non-WGMMA instructions read accumulator registers |

This is important because the nominal register count stays the same, but the
storage pressure changes sharply. The original reorder codegen spills, while the
delayed-rescale-only codegen does not.

2. The generated CUDA really does move the `acc_o *= ss` loop.

- In [reorder_core_split_codegen.cu](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/codegen_compare/reorder_core_split_codegen.cu), the `acc_o_1[i] = acc_o_1[i] * ss_1[...]` loop appears immediately after the QK block and before `v_full.wait(...)` / PV issue.
- In [reorder_delayed_rescale_core_split_codegen.cu](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/codegen_compare/reorder_delayed_rescale_core_split_codegen.cu), that loop is gone from the pre-PV path and reappears only after `wait_wgmma<0>()` and `v_empty.arrive()`.

So the timing difference is not only a measurement artifact. The generated CUDA
really has a different dependency structure around the softmax window.

3. The `ptxas` warnings point to two different kinds of pressure.

- `reorder`:
  - compiler injects `warpgroup.wait`
  - has spills
- `delayed_rescale_only`:
  - no spills
  - but `ptxas` warns that non-WGMMA instructions read accumulator registers inside the WGMMA pipeline stage

The current reading is:

- moving `rescale(acc_o)` changes the live-range / dependency picture seen by `ptxas`
- that is enough to reshape the generated schedule
- the most visible outcome is:
  - much shorter `softmax_core`
  - much larger launch-side windows
  - fewer spill-related symptoms

This does **not** yet prove exactly which SASS instructions account for the full
`~257 cycle` drop in `softmax_core`, but it does make the main mechanism much
more concrete: delayed rescale is changing codegen, not just moving a small
scalar loop in source.

## Blog-Ready Explanation: Why Delayed Rescale Helps

A concise way to explain the current finding is:

> Delaying `rescale(acc_o)` does not make the rescale math itself cheaper.
> Instead, it removes `acc_o`-side work from the critical softmax-side window,
> which gives the compiler and hardware a cleaner dependency picture.

More concretely:

- `rescale(acc_o)` depends on:
  - the previous `acc_o`
  - the previously computed scale factor `ss`
- it does **not** depend on the new softmax reduction result being produced in
  the current `softmax_core` window

That means `rescale(acc_o)` is logically important, but it is not on the
immediate data path needed to start the current-round softmax update.

When `rescale(acc_o)` sits before PV, the compiler has to schedule all of these
things in a tighter region:

- QK / PV launch
- `acc_o` read-modify-write work
- softmax state (`sm`, `ss`, `ls`)
- the new `acc_s` values that will feed the current softmax

This increases the amount of simultaneously live state and makes the codegen
around the softmax window more crowded.

When `rescale(acc_o)` is delayed until after `wait<0>`, the key effect is:

- the current softmax-side work no longer has to share its hottest window with
  `acc_o *= ss`
- the compiler is freer to pack the softmax-side computation more tightly
- the `acc_o` read path is moved out of a more WGMMA-sensitive region

This matches what we measured:

- `softmax_core` drops sharply:
  - `1313 -> 1056` cycles for `reorder -> delayed_rescale_only`
- the launch-side windows get much larger:
  - `qk_issue: 200 -> 1066`
  - `pv_issue: 216 -> 941`
- `ptxas` behavior changes:
  - original `reorder` shows spill traffic and an injected `warpgroup.wait`
  - delayed-rescale-only removes those spill symptoms

So the most accurate current interpretation is:

> delayed rescale helps mainly because it changes live-range / dependency
> pressure in codegen, not because the scalar rescale loop is inherently fast.

One important nuance:

- this does **not** mean delayed rescale has no register dependency at all
- it still reads and writes `acc_o`, and still uses `ss`
- the key point is that it does not need to happen inside the same critical
  window as the current softmax-side update

For the blog, a simple phrasing that should be both accurate and readable is:

> `rescale(acc_o)` is a necessary step, but it is delayable work rather than
> immediate critical-path work. Moving it later reduces interference with the
> softmax-side window, which leads to better codegen and a shorter measured
> `softmax_core`.
